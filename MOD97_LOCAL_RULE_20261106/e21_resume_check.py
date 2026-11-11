"""E21 acceptance tests: lives and births that survive a restart (brief: OPUS_TASK_E21_RESUMABLE_LIVES_20261110.md).

    python3 e21_resume_check.py [--work DIR] [--base-ref REF] [--jobs N] [--only small|large|all]

Exit code nonzero on any failure; results in exp45/E21_RESUME_CHECK.json. Every run happens in its own scratch directory (a copy of this
folder's *.py and links to the births it needs), so no receipt, checkpoint or birth in exp45/ is touched.

1. Unchanged when not resuming: each command shape run by this branch and by the base commit (exp16_integrated.py from --base-ref, every other
   module from this folder, which must be identical there) gives the same receipt apart from CPU-time fields, and the same final weights hash.
2. Exact resume: a run killed with SIGKILL (the process sends it to itself at an exactly counted update or score call, see --child) and
   restarted with E16_RESUME=1 equals the uninterrupted run in every stage record (CPU fields aside), in the weights hash, and in the pipeline
   digest of every checkpoint and every replayed stage. Three kill points per shape; one shape killed twice in one life.
3. Fail-closed: a changed setting stops naming it; a truncated .ckpt falls back to .ckpt.prev (and the result is still exact); both damaged
   stops; an edited pipeline digest (re-sealed, so the file is intact) stops at the comparison; a kill inside the checkpoint's own write
   (between the two renames) resumes from .ckpt.prev, exactly.
4. The same at E11_STAGE_UPDATES=512 for two shapes, one kill each.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = "73129"
CPU_KEYS = {"cpu", "cpu_scoring_so_far", "cpu_updates_so_far"}

SHAPES = {   # name: (environment, argv)
    "T_drift_online_store": (dict(E16_REPLAY="raw", E16_CAP="65536", E16_ADMIT="online", E16_STREAM="drift"), ["T", SEED]),
    "D_reference": ({}, ["D", SEED]),
    "M_block_admission": (dict(E16_REPLAY="raw", E16_CAP="65536"), ["M", SEED]),
    "bank2_phase4": (dict(E16_ARCH="bank2", E16_BOOKDECAY="off", E16_REPLAY="raw", E16_CAP="65536", E16_TRACEUP="all"), ["T", SEED]),
    "bank2_phase4_lossall": (dict(E16_ARCH="bank2", E16_BOOKDECAY="off", E16_REPLAY="raw", E16_CAP="65536", E16_TRACEUP="all", E16_LOSS="all"), ["T", SEED]),
    "T_compgap6": (dict(E16_COMPGAP="6", E16_REPLAY="raw", E16_CAP="65536"), ["T", SEED]),
    "birth_rule_T_local_heads": (dict(E16_ARCH="bank2", E16_BOOKDECAY="off", E16_HEADS="local", E16_BIRTHRULE="T", E16_TRACEUP="all"), ["dev", SEED]),
}
BIRTHS = ["e11_dev_M_73129", "e17_dev_bank2_nbd_73129"]


# ---------------------------------------------------------------- child: one run, killed by itself at an exact point
def child(spec, argv):
    """spec: 'none' | 'update:N' (SIGKILL as this process's N-th update begins) | 'score:N:J' (SIGKILL at the J-th score call made after
    this process's N-th update, i.e. inside that stage's or curve row's evaluation) | 'ckptrename:S' (SIGKILL inside the S-th checkpoint
    write of this process, after the old .ckpt became .ckpt.prev and before the new one is renamed into place). Then the harness's own main."""
    sys.path.insert(0, os.getcwd())
    import exp16_integrated as H
    E = H.E
    kind, *a = spec.split(":")
    a = [int(v) for v in a]
    n_up, n_sc, n_ck = [0], [0], [0]
    die = lambda: os.kill(os.getpid(), signal.SIGKILL)

    def counted(f):
        def g(*x, **k):
            n_up[0] += 1
            if kind == "update" and n_up[0] == a[0]:
                print(f"e21-check SIGKILL at update {n_up[0]} of this process", flush=True); die()
            return f(*x, **k)
        return g

    def scored(f):
        def g(*x, **k):
            if kind == "score" and n_up[0] >= a[0]:
                n_sc[0] += 1
                if n_sc[0] == a[1]:
                    print(f"e21-check SIGKILL at score call {n_sc[0]} after update {n_up[0]} of this process", flush=True); die()
            return f(*x, **k)
        return g
    birth = argv[0] == "dev"
    plain = H.HEADS16 == "top" and H.BIRTHRULE16 == "D" and H.FIXEDCODES16 == "off" and H.LOSS16 == "answer"
    if birth and plain:
        E.update = counted(E.update)
    else:
        H.update16 = counted(H.update16)
    if birth:
        E.score = scored(E.score)
    else:
        H.score = scored(H.score)
    if kind == "ckptrename":
        real_replace = os.replace

        def ckpt_write(ck, obj):     # the harness's own write, with the kill between its two renames
            def rep(src, dst):
                real_replace(src, dst)
                if str(dst).endswith(".ckpt.prev"):
                    n_ck[0] += 1
                    if n_ck[0] == a[0]:
                        print(f"e21-check SIGKILL inside checkpoint write {n_ck[0]}, after {Path(dst).name}", flush=True); die()
            H.os.replace = rep
            try:
                return real_write(ck, obj)
            finally:
                H.os.replace = real_replace
        real_write = H.ckpt_write
        H.ckpt_write = ckpt_write
    H.OUT.mkdir(exist_ok=True)
    if argv[0] == "dev":
        H.develop17(int(argv[1]))
    else:
        H.life(argv[0], int(argv[1]))


# ---------------------------------------------------------------- parent
class Check:
    def __init__(self, args):
        self.args = args
        self.work = Path(args.work or tempfile.mkdtemp(prefix="e21_"))
        self.work.mkdir(parents=True, exist_ok=True)
        self.results, self.costs = [], {}
        self.failed = False

    # code trees
    def prepare_code(self):
        new, base = self.work / "code_new", self.work / "code_base"
        for d in (new, base):
            shutil.rmtree(d, ignore_errors=True); d.mkdir()
            for f in HERE.glob("*.py"):
                shutil.copy2(f, d / f.name)
        rel = HERE.relative_to(Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip()))
        ref = self.args.base_ref
        self.base_commit = subprocess.check_output(["git", "rev-parse", ref], cwd=HERE, text=True).strip()
        changed = subprocess.check_output(["git", "diff", "--name-only", self.base_commit, "--", "."], cwd=HERE, text=True).split()
        py_changed = sorted(Path(c).name for c in changed if c.endswith(".py"))
        assert set(py_changed) <= {"exp16_integrated.py", "e21_resume_check.py"}, f"modules other than the harness differ from {ref}: {py_changed}"
        (base / "exp16_integrated.py").write_bytes(subprocess.check_output(["git", "show", f"{self.base_commit}:{rel}/exp16_integrated.py"], cwd=HERE))
        self.code = dict(new=new, base=base)
        self.code_sha = {k: hashlib.sha256(b"".join(f.read_bytes() for f in sorted(v.glob("*.py")) if f.name != "e21_resume_check.py")).hexdigest()
                         for k, v in self.code.items()}     # the code under test; the check script itself is not part of the cache key

    def run_dir(self, name, code):
        d = self.work / "runs" / name
        shutil.rmtree(d, ignore_errors=True); (d / "exp45").mkdir(parents=True)
        for f in self.code[code].glob("*.py"):      # copies: the harness finds exp45/ from its own resolved path
            shutil.copy2(f, d / f.name)
        os.symlink(HERE / "exp45" / "BIRTH_MANIFEST.json", d / "exp45" / "BIRTH_MANIFEST.json")
        for b in BIRTHS:
            for ext in (".pt", ".json"):
                if (HERE / "exp45" / (b + ext)).exists():
                    os.symlink(HERE / "exp45" / (b + ext), d / "exp45" / (b + ext))
        return d

    def env(self, shape, dose, extra=None):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}
        e.update(SHAPES[shape][0])
        if SHAPES[shape][1][0] == "dev":
            e["E11_DEV_UPDATES"] = str(dose)
        else:
            e["E11_STAGE_UPDATES"] = str(dose)
        e.update(extra or {})
        return e

    def proc(self, d, env, argv, spec, log):
        cmd = [sys.executable, str(HERE / "e21_resume_check.py"), "--child", spec, *argv] if spec != "plain" else [sys.executable, "exp16_integrated.py", *argv]
        t = time.time()
        with open(log, "a") as f:
            f.write(f"\n### {' '.join(k + '=' + v for k, v in sorted(env.items()) if k.startswith(('E11_', 'E16_')))} {' '.join(cmd[1:])}\n"); f.flush()
            rc = subprocess.run(cmd, cwd=d, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        return rc, round(time.time() - t, 1)

    # one life: phases = [spec, spec, ...]; the first runs without E16_RESUME, the rest with it. Kill specs must end in SIGKILL (rc -9).
    def life_run(self, name, code, shape, dose, phases, extra=None, between=None):
        done = self.work / "runs" / name / "job.json"      # a job that finished before the runner itself was interrupted is kept
        key = dict(code=code, shape=shape, dose=dose, phases=phases, extra=extra, code_sha=self.code_sha[code])
        if done.exists() and json.loads(done.read_text()).get("key") == key:
            return json.loads(done.read_text())["result"]
        r = self._life_run(name, code, shape, dose, phases, extra, between)
        done.write_text(json.dumps(dict(key=key, result=r)))
        return r

    def _life_run(self, name, code, shape, dose, phases, extra=None, between=None):
        d = self.run_dir(name, code); log = d / "run.log"; steps = []
        for i, spec in enumerate(phases):
            env = self.env(shape, dose, dict(extra or {}, **({"E16_RESUME": "1"} if i else {})))
            if i and between:
                between(d, i)
            rc, sec = self.proc(d, env, SHAPES[shape][1], spec, log)
            steps.append(dict(spec=spec, rc=rc, seconds=sec))
            expect_kill = spec not in ("none", "plain")
            if (rc == -signal.SIGKILL) != expect_kill or (not expect_kill and rc != 0):
                return dict(dir=str(d), steps=steps, ok=False, why=f"phase {i} ({spec}) exited {rc}")
        return dict(dir=str(d), steps=steps, ok=True)

    # receipts and digests
    def receipt(self, d, shape):
        r = [p for p in (d / "exp45").glob("*.json") if p.name.startswith(("e16_life", "e17_life", "e17_dev_bank2_nbd_lh_bT"))]
        assert len(r) == 1, f"{d}: receipts {r}"
        return r[0]

    @staticmethod
    def strip(x):
        if isinstance(x, dict):
            return {k: Check.strip(v) for k, v in x.items() if k not in CPU_KEYS and k != "resumed"}
        if isinstance(x, list):
            return [Check.strip(v) for v in x]
        return x

    @staticmethod
    def first_diff(a, b, path="$"):
        if type(a) != type(b):
            return f"{path}: {type(a).__name__} vs {type(b).__name__}"
        if isinstance(a, dict):
            for k in list(a) + [k for k in b if k not in a]:
                if k not in a or k not in b:
                    return f"{path}.{k}: present in one only"
                r = Check.first_diff(a[k], b[k], f"{path}.{k}")
                if r:
                    return r
            if list(a) != list(b):
                return f"{path}: key order differs"
            return None
        if isinstance(a, list):
            if len(a) != len(b):
                return f"{path}: length {len(a)} vs {len(b)}"
            for i, (x, y) in enumerate(zip(a, b)):
                r = Check.first_diff(x, y, f"{path}[{i}]")
                if r:
                    return r
            return None
        return None if a == b else f"{path}: {a!r} vs {b!r}"

    def outcome(self, d, shape):
        rp = self.receipt(d, shape); r = json.loads(rp.read_text())
        if isinstance(r.get("complete"), dict):      # life (a birth's receipt has complete: true)
            w = d / "exp45" / r["complete"]["weights"]; claimed = r["complete"]["weights_sha256"]
        else:                    # birth
            w = d / "exp45" / r["weights"]; claimed = r["weights_sha256"]
        actual = hashlib.sha256(w.read_bytes()).hexdigest()
        assert actual == claimed, f"{w}: hash {actual} is not the receipt's {claimed}"
        logs = list((d / "exp45").glob("*.ckpt.log"))
        log = [json.loads(l) for l in logs[0].read_text().splitlines()] if logs else []
        return dict(receipt=r, sha=actual, log=log, receipt_path=rp)

    def compare(self, ref, run, digests=True):
        """Every stage/curve record apart from CPU fields, the weights hash, and every digest of `run` against `ref`'s checkpoint digests."""
        diff = self.first_diff(self.strip(ref["receipt"]), self.strip(run["receipt"]))
        out = dict(receipt_equal=diff is None, first_difference=diff, weights_equal=ref["sha"] == run["sha"], weights_sha256=run["sha"])
        if digests:
            key = lambda e: e.get("stage", e.get("step"))
            refd = {key(e): e["pipeline"] for e in ref["log"] if e["event"] == "checkpoint"}
            bad = [(e["event"], key(e)) for e in run["log"] if refd.get(key(e)) != e["pipeline"]]
            out.update(digests_compared=len(run["log"]), digests_reference=len(refd), digest_mismatches=bad,
                       replayed=sorted({key(e) for e in run["log"] if e["event"] == "replay"}))
            out["ok"] = out["receipt_equal"] and out["weights_equal"] and not bad and len(refd) > 0
        else:
            out["ok"] = out["receipt_equal"] and out["weights_equal"]
        return out

    def record(self, **r):
        self.results.append(r)
        if not r.get("ok"):
            self.failed = True
        print(f"[{'PASS' if r.get('ok') else 'FAIL'}] {r['test']} {r.get('shape', '')} {r.get('kill', '')} {r.get('why', '') or ''}", flush=True)

    # ---------------------------------------------------------------- the plan
    def kills(self, shape, dose):
        """Three kill points: inside the first stage before any checkpoint; inside a middle stage's updates; inside a middle stage's evaluation."""
        if SHAPES[shape][1][0] == "dev":     # curve rows (checkpoints) every 4096 updates
            return [("first segment, before any checkpoint (update 100)", "update:100", 0),
                    ("updates after the first row (update 5096)", "update:5096", 4096),
                    ("evaluation of the last row, at 8192 (7th score call)", "score:8192:7", 4096)]
        return [("stage 1 updates, before any checkpoint", f"update:{max(3, dose // 4)}", 0),
                (f"stage 3 updates (update {2 * dose + dose // 2})", f"update:{2 * dose + dose // 2}", 2),
                ("stage 4 evaluation (25th score call)", f"score:{4 * dose}:25", 3)]

    def run(self):
        self.prepare_code(); a = self.args
        pool = ThreadPoolExecutor(a.jobs); fut = {}
        small = a.only in ("small", "all"); large = a.only in ("large", "all")
        S = 16; want = set(a.shapes.split(",")) if a.shapes else set(SHAPES); self.want = want
        assert want <= set(SHAPES), f"unknown shapes {want - set(SHAPES)}"
        order = sorted([x for x in SHAPES if x in want], key=lambda x: SHAPES[x][1][0] != "dev")     # births first: the longest runs
        if small:
            for sh in order[:1] if SHAPES[order[0]][1][0] == "dev" else []:
                self.small_jobs(pool, fut, sh, S)
        if large:
            for sh in [x for x in ("T_drift_online_store", "bank2_phase4") if x in want]:
                fut[("A", sh, "L")] = pool.submit(self.life_run, f"{sh}_A512", "new", sh, 512, ["plain"])
                fut[("B", sh, "L")] = pool.submit(self.life_run, f"{sh}_B512", "new", sh, 512, ["update:1300", "plain"])
                self.fail_closed_jobs(pool, fut, sh, 512, "L")
        if small:
            for sh in order:
                if SHAPES[sh][1][0] != "dev":
                    self.small_jobs(pool, fut, sh, S)
        if small and "T_drift_online_store" in want:
            sh = "T_drift_online_store"   # killed twice in one life: in stage 2's updates, then (resumed from stage 1) in stage 5's evaluation
            fut[("B2x", sh)] = pool.submit(self.life_run, f"{sh}_Btwice", "new", sh, S, [f"update:{S + 5}", f"score:{4 * S}:10", "plain"])
            self.fail_closed_jobs(pool, fut, "T_drift_online_store", S, "small")
        res = {k: f.result() for k, f in fut.items()}
        self.evaluate(res, small, large)
        return res

    def small_jobs(self, pool, fut, sh, S):
        dose = 8192 if SHAPES[sh][1][0] == "dev" else S
        fut[("base", sh)] = pool.submit(self.life_run, f"{sh}_base", "base", sh, dose, ["plain"])
        fut[("A", sh)] = pool.submit(self.life_run, f"{sh}_A", "new", sh, dose, ["plain"])
        for i, (label, spec, _) in enumerate(self.kills(sh, dose)):
            fut[("B", sh, i)] = pool.submit(self.life_run, f"{sh}_B{i}", "new", sh, dose, [spec, "plain"])

    def fail_closed_jobs(self, pool, fut, sh, dose, tag):
        """Each starts from a life killed in stage 3 (two checkpoints exist: .ckpt at stage 2, .ckpt.prev at stage 1)."""
        k = f"update:{2 * dose + 3}"

        def damage(how):
            def f(d, i):
                if i != 1:
                    return
                ck = next((d / "exp45").glob("*.ckpt")); prev = Path(str(ck) + ".prev")
                if how in ("truncate", "both"):
                    raw = ck.read_bytes(); ck.write_bytes(raw[:len(raw) // 2])
                if how == "both":
                    raw = bytearray(prev.read_bytes()); raw[len(raw) // 2] ^= 0xFF; prev.write_bytes(bytes(raw))
                if how == "digest":       # an intact, re-sealed checkpoint whose saved pipeline digest differs in one component
                    import io
                    import torch
                    raw = ck.read_bytes(); obj = torch.load(io.BytesIO(raw[40:]), weights_only=True)
                    dg = json.loads(obj["pipeline"]); dg["parts"]["rng"] = "0" * 64
                    dg["overall"] = hashlib.sha256(json.dumps(dg["parts"], sort_keys=True).encode()).hexdigest(); obj["pipeline"] = json.dumps(dg)
                    buf = io.BytesIO(); torch.save(obj, buf); body = buf.getvalue()
                    ck.write_bytes(raw[:8] + hashlib.sha256(body).digest() + body)
            return f
        fut[("FC", sh, tag, "setting")] = pool.submit(self._fc_setting, sh, dose, tag, k)
        fut[("FC", sh, tag, "truncate")] = pool.submit(self.life_run, f"{sh}_{tag}_fc_truncate", "new", sh, dose, [k, "plain"], between=damage("truncate"))
        fut[("FC", sh, tag, "both")] = pool.submit(self._fc_expect_stop, sh, dose, tag, k, "both", damage("both"))
        fut[("FC", sh, tag, "digest")] = pool.submit(self._fc_expect_stop, sh, dose, tag, k, "digest", damage("digest"))
        if tag == "small":
            fut[("FC", sh, tag, "ckptrename")] = pool.submit(self.life_run, f"{sh}_{tag}_fc_ckptrename", "new", sh, dose, ["ckptrename:2", "plain"])

    def cached(self, name, key, fn):
        done = self.work / "runs" / f"{name}.job.json"; key = dict(key, code_sha=self.code_sha["new"])
        if done.exists() and json.loads(done.read_text()).get("key") == key:
            return json.loads(done.read_text())["result"]
        r = fn(); done.parent.mkdir(parents=True, exist_ok=True); done.write_text(json.dumps(dict(key=key, result=r), default=str))
        return r

    def _fc_setting(self, sh, dose, tag, k):
        return self.cached(f"{sh}_{tag}_fc_setting", dict(sh=sh, dose=dose, k=k), lambda: self.__fc_setting(sh, dose, tag, k))

    def __fc_setting(self, sh, dose, tag, k):
        out = {}
        for var, val in (("E16_LEGACY_BIRTH", "allow"), ("E11_SCHEDULE", "interleaved")):
            r = self._life_run(f"{sh}_{tag}_fc_setting_{var}", "new", sh, dose, [k])
            d = Path(r["dir"]); log = d / "run.log"
            rc, _ = self.proc(d, self.env(sh, dose, {"E16_RESUME": "1", var: val}), SHAPES[sh][1], "plain", log)
            text = log.read_text().split("\n### ")[-1]
            msg = [l for l in text.splitlines() if l.startswith("e21:") or "SystemExit" in l]
            out[var] = dict(rc=rc, named=any(var in l and "different configuration" in l for l in msg), started=("after F" in text or "e21 resumed at" in text),
                            message=msg[-1:] or None, ok=r["ok"])
        return out

    def _fc_expect_stop(self, sh, dose, tag, k, how, between):
        return self.cached(f"{sh}_{tag}_fc_{how}", dict(sh=sh, dose=dose, k=k, how=how), lambda: self.__fc_expect_stop(sh, dose, tag, k, how, between))

    def __fc_expect_stop(self, sh, dose, tag, k, how, between):
        r = self._life_run(f"{sh}_{tag}_fc_{how}", "new", sh, dose, [k])
        d = Path(r["dir"]); between(d, 1); log = d / "run.log"
        rc, _ = self.proc(d, self.env(sh, dose, {"E16_RESUME": "1"}), SHAPES[sh][1], "plain", log)
        text = log.read_text().split("\n### ")[-1]
        return dict(first=r, rc=rc, message=[l for l in text.splitlines() if l.startswith("e21")], started=("after F" in text or "e21 resumed at" in text))

    # ---------------------------------------------------------------- verdicts
    def evaluate(self, res, small, large):
        A = {}
        if small:
            for sh in [x for x in SHAPES if x in self.want]:
                ok_runs = all(res[k]["ok"] for k in (("base", sh), ("A", sh)))
                if not ok_runs:
                    self.record(test="1 unchanged", shape=sh, ok=False, why=f"a run failed: {res[('base', sh)]}, {res[('A', sh)]}"); continue
                base = self.outcome(Path(res[("base", sh)]["dir"]), sh); A[sh] = self.outcome(Path(res[("A", sh)]["dir"]), sh)
                c = self.compare(base, A[sh], digests=False)
                self.record(test="1 unchanged", shape=sh, **c)
                for i, (label, spec, K) in enumerate(self.kills(sh, 8192 if SHAPES[sh][1][0] == "dev" else 16)):
                    self.exact(res[("B", sh, i)], A[sh], sh, f"{label} [{spec}]", K)
        if small and "T_drift_online_store" in self.want:
            sh = "T_drift_online_store"
            self.exact(res[("B2x", sh)], A[sh], sh, "twice: stage 2 updates (update 21), then, resumed from stage 1, stage 5 evaluation (10th score call)", [1, 4])
            self.fail_closed(res, A, "T_drift_online_store", "small")
        if large:
            for sh in [x for x in ("T_drift_online_store", "bank2_phase4") if x in self.want]:
                if not res[("A", sh, "L")]["ok"]:
                    self.record(test="4 larger dose", shape=sh, ok=False, why="the uninterrupted run failed"); continue
                AL = self.outcome(Path(res[("A", sh, "L")]["dir"]), sh); A[(sh, "L")] = AL
                self.exact(res[("B", sh, "L")], AL, sh, "dose 512, stage 3 updates (update 1300)", 2, test="4 larger dose: exact resume")
                self.fail_closed(res, {sh: AL}, sh, "L")
                self.costs[sh] = self.cost(AL, res[("B", sh, "L")])

    def exact(self, r, A, sh, label, K, test="2 exact resume"):
        if not r["ok"]:
            self.record(test=test, shape=sh, kill=label, ok=False, why=r.get("why"), steps=r["steps"]); return
        B = self.outcome(Path(r["dir"]), sh); c = self.compare(A, B)
        res_list = B["receipt"].get("resumed", [])
        c["resumed"] = res_list; c["steps"] = r["steps"]
        want = K if isinstance(K, list) else [K]
        if [x["from_stage"] for x in res_list] != want:
            c["ok"] = False; c["why"] = f"expected resumes from {want}, receipt says {res_list}"
        self.record(test=test, shape=sh, kill=label, **c)

    def fail_closed(self, res, A, sh, tag):
        t = "3 fail-closed" if tag == "small" else "4 larger dose: fail-closed"
        s = res[("FC", sh, tag, "setting")]
        for var, v in s.items():
            v = dict(v, first_run_ok=v.pop("ok"))
            self.record(test=t, shape=sh, kill=f"changed setting {var}", ok=v["first_run_ok"] and v["rc"] not in (0, -9) and v["named"] and not v["started"], **v)
        r = res[("FC", sh, tag, "truncate")]
        if r["ok"]:
            B = self.outcome(Path(r["dir"]), sh); c = self.compare(A[sh], B); text = (Path(r["dir"]) / "run.log").read_text()
            c["said_fallback"] = "falling back to" in text and ".ckpt.prev" in text
            c["resumed"] = B["receipt"].get("resumed")
            c["ok"] = c["ok"] and c["said_fallback"] and c["resumed"][-1]["checkpoint"].endswith(".ckpt.prev") and c["resumed"][-1]["from_stage"] == 1
            self.record(test=t, shape=sh, kill="truncated .ckpt -> .ckpt.prev, then exact", **c)
        else:
            self.record(test=t, shape=sh, kill="truncated .ckpt", ok=False, why=r.get("why"))
        for how, need in (("both", "every checkpoint is damaged"), ("digest", "does not match")):
            v = res[("FC", sh, tag, how)]
            ok = v["first"]["ok"] and v["rc"] not in (0, -9) and any(need in l for l in v["message"]) and not v["started"]
            self.record(test=t, shape=sh, kill={"both": "both checkpoints damaged", "digest": "pipeline digest edited (file re-sealed)"}[how], ok=ok, rc=v["rc"], message=v["message"])
        if tag == "small":
            r = res[("FC", sh, tag, "ckptrename")]
            if r["ok"]:
                B = self.outcome(Path(r["dir"]), sh); c = self.compare(A[sh], B); c["resumed"] = B["receipt"].get("resumed")
                c["ok"] = c["ok"] and c["resumed"][-1]["checkpoint"].endswith(".ckpt.prev") and c["resumed"][-1]["from_stage"] == 2
                self.record(test=t, shape=sh, kill="inside the stage-3 checkpoint write, between the renames -> .ckpt.prev (stage 2), then exact", **c)
            else:
                self.record(test=t, shape=sh, kill="inside a checkpoint write", ok=False, why=r.get("why"))

    def cost(self, A, rB):
        ck = [e for e in A["log"] if e["event"] == "checkpoint"]
        B = self.outcome(Path(rB["dir"]), "x")
        rs = [r for r in B["receipt"].get("resumed", []) if r["from_stage"]]
        per_stage = rs[-1]["cpu_replay"] / rs[-1]["from_stage"] if rs else None
        return dict(checkpoint_bytes=[e["bytes"] for e in ck], seconds_to_write=[e["seconds_to_write"] for e in ck],
                    replay_cpu_seconds_per_stage_at_512=per_stage,
                    replay_cpu_seconds_per_stage_at_6144_ESTIMATE=(per_stage * 6144 / 512 if per_stage is not None else None),
                    note="the full-dose figure is an estimate: replay cost at 512 updates per stage scaled linearly to 6,144 (panel building included)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child(sys.argv[2], sys.argv[3:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--work"); ap.add_argument("--base-ref", default="40cb1c208adec3f39351356e52d3f69eba4ade17")
    ap.add_argument("--jobs", type=int, default=os.cpu_count()); ap.add_argument("--only", default="all", choices=("small", "large", "all"))
    ap.add_argument("--shapes", default="", help="comma-separated subset of the shapes (default: all)")
    ap.add_argument("--out", default=str(HERE / "exp45" / "E21_RESUME_CHECK.json"))
    a = ap.parse_args()
    import torch
    c = Check(a); t = time.time(); c.run()
    out = dict(brief="OPUS_TASK_E21_RESUMABLE_LIVES_20261110.md", base_commit=c.base_commit,
               branch_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
               tested_sha256={f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ("exp16_integrated.py", "e21_resume_check.py")},
               dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "--", "exp16_integrated.py"], cwd=HERE, text=True).strip()),
               machine=dict(platform=platform.platform(), processor=platform.processor() or platform.machine(), cpus=os.cpu_count(),
                            python=platform.python_version(), torch=torch.__version__),
               only=a.only, shapes=sorted(c.want), wall_seconds=round(time.time() - t), passed=not c.failed,
               counts=dict(passed=sum(r["ok"] for r in c.results), failed=sum(not r["ok"] for r in c.results)),
               results=c.results, costs=c.costs, work=str(c.work))
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["counts"]), "passed" if out["passed"] else "FAILED", flush=True)
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
