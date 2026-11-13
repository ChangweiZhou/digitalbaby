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
5. (E21b) Fixtures: a source changed after the checkpoint, another birth, a changed dose (stage and birth), .ckpt/.ckpt.prev of two runs or
   non-consecutive, a kill during the final publication of a birth and of a life (after the weights; half way through the receipt), a torn final
   receipt, another torch version, and the qualification gate (absent, listed, not listed, other sources). Each fixture is also run against a harness
   without its guard (a mutant of this branch's, or the base harness), where it must fail.
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
MUTANTS = {"sources": ("sources",), "birth": ("birth_sha256",), "stage_dose": ("E11_STAGE_UPDATES", "receipt"), "dev_dose": ("E11_DEV_UPDATES", "stem"),
           "torch": ("torch_version",), "pairing": "pairing",     # E21b: what each mutant's checkpoint admission ignores
           # E21b amendments: each removes one guard of the production gate, by text replacement
           "closure": [('"exp3_state.py", "mod97_local.py")', '"exp3_state.py")')],
           "runtime": [('        elif rec.get("torch") != T.__version__ or rec.get("dtype") != str(T.get_default_dtype()):', '        elif False:')],
           "record": [("        elif cr is None or not cr.is_file():", "        elif False:"),
                      ('        elif hashlib.sha256(cr.read_bytes()).hexdigest() != rec.get("check_record_sha256"):', "        elif False:")]}


# ---------------------------------------------------------------- child: one run, killed by itself at an exact point
def child(spec, argv):
    """spec: 'none' | 'update:N' (SIGKILL as this process's N-th update begins) | 'score:N:J' (SIGKILL at the J-th score call made after
    this process's N-th update, i.e. inside that stage's or curve row's evaluation) | 'ckptrename:S' (SIGKILL inside the S-th checkpoint
    write of this process, after the old .ckpt became .ckpt.prev and before the new one is renamed into place). Then the harness's own main."""
    sys.path.insert(0, os.getcwd())
    import exp16_integrated as H
    E = H.E
    kind, *a = spec.split(":")
    a = [int(v) for v in a] if kind != "publish" else a
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
    if kind == "publish":      # E21b: a kill during the final publication (weights, then receipt) of a birth or a life
        where = spec.split(":")[1]
        if where == "weights":      # the final weights are in place, the final receipt not yet written
            real_save = H.save_weights

            def save_then_die(m, path):
                h = real_save(m, path); print(f"e21-check SIGKILL after the final weights {Path(path).name}", flush=True); die(); return h
            H.save_weights = save_then_die
        else:                       # 'torn': inside the final receipt's write, half of it written
            def half(path, text):
                print(f"e21-check SIGKILL half way through writing the final receipt {Path(path).name}", flush=True)
                with open(path, "w") as f:
                    f.write(text[:len(text) // 2]); f.flush(); os.fsync(f.fileno())
                die()
            if hasattr(H, "write_json"):      # this branch: the receipt goes to a temporary file, then is renamed
                real_wj = H.write_json

                def wj(path, obj):
                    if isinstance(obj, dict) and obj.get("complete"):
                        half(str(path) + ".tmp", json.dumps(obj, indent=1))
                    return real_wj(path, obj)
                H.write_json = wj
            else:                             # the base harness writes the receipt in place
                real_wt = Path.write_text

                def wt(self, data, *x, **k):
                    if self.suffix == ".json" and ('"complete": {' in data or '"complete": true' in data):
                        half(str(self), data)
                    return real_wt(self, data, *x, **k)
                Path.write_text = wt
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
        deps = {"arch17.py", "exp11_cell.py", "exp45.py", "exp3_state.py", "mod97_local.py"}     # the modules the harness imports (checked by 6 E21b source closure)
        py_changed = sorted(Path(c).name for c in changed if c.endswith(".py") and Path(c).parent == Path(rel) and Path(c).name in deps)
        assert not py_changed, f"modules the harness imports differ from {ref}: {py_changed}"
        (base / "exp16_integrated.py").write_bytes(subprocess.check_output(["git", "show", f"{self.base_commit}:{rel}/exp16_integrated.py"], cwd=HERE))
        self.code = dict(new=new, base=base)
        # E21b: mutants of this branch's harness, each with one guard removed, to show each fixture is able to fail
        src = (new / "exp16_integrated.py").read_text()
        cfg_line = '    old = json.loads(obj["config"])\n    diff = {}'
        assert src.count(cfg_line) == 1 and src.count("        elif p == cur and prev.exists():") == 1
        for name, ign in MUTANTS.items():
            d = self.work / f"code_mut_{name}"; shutil.rmtree(d, ignore_errors=True); shutil.copytree(new, d)
            if ign == "pairing":
                m_src = src.replace("        elif p == cur and prev.exists():", "        elif False:")
            elif isinstance(ign, list):
                m_src = src
                for a_, b_ in ign:
                    assert m_src.count(a_) == 1, (name, a_); m_src = m_src.replace(a_, b_)
            else:
                m_src = src.replace(cfg_line, f'    IGN = {ign!r}; old = {{k: v for k, v in json.loads(obj["config"]).items() if k not in IGN}}; cfg = {{k: v for k, v in cfg.items() if k not in IGN}}\n    diff = {{}}')
            (d / "exp16_integrated.py").write_text(m_src); self.code[f"mut_{name}"] = d
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

    def env(self, shape, dose, extra=None, gate=False):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}
        e.update(SHAPES[shape][0])
        if SHAPES[shape][1][0] == "dev":
            e["E11_DEV_UPDATES"] = str(dose)
        else:
            e["E11_STAGE_UPDATES"] = str(dose)
        e.update(extra or {})
        if e.get("E16_RESUME") == "1" and not gate:     # E21b: resume is gated off until qualified; the checks resume with the explicit allowance
            e["E16_RESUME_UNQUALIFIED"] = "allow"
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
        if a.only in ("e21b", "all"):
            self.e21b_jobs(pool, fut)
        res = {k: f.result() for k, f in fut.items()}
        self.evaluate(res, small, large)
        if a.only in ("e21b", "all"):
            self.e21b_evaluate(res)
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

    # ---------------------------------------------------------------- E21b fixtures (brief OPUS_TASK_E21B_RESUME_QUALIFICATION_20261111.md, items 2-4)
    def fx(self, name, code, shape, steps):
        """One fixture directory: steps = [(spec, extra environment, gate, (label, hook) or None)], run in order; a hook acts on the directory
        before its step. With gate=False a step that resumes is allowed to (E16_RESUME_UNQUALIFIED=allow). Kept once finished."""
        dose = 8192 if SHAPES[shape][1][0] == "dev" else 16
        key = json.loads(json.dumps(dict(code=code, shape=shape, dose=dose, steps=[(sp, ex, g, h[0] if h else None) for sp, ex, g, h in steps])))   # as stored: tuples become lists
        done = self.work / "runs" / name / "job.json"
        if done.exists() and json.loads(done.read_text()).get("key") == dict(key, code_sha=self.code_sha[code]):
            return json.loads(done.read_text())["result"]
        d = self.run_dir(name, code); log = d / "run.log"; out = []
        for spec, extra, gate, hook in steps:
            note = hook[1](d) if hook else None
            rc, sec = self.proc(d, self.env(shape, dose, extra, gate=gate), SHAPES[shape][1], spec, log)
            text = log.read_text().split("\n### ")[-1]
            out.append(dict(spec=spec, rc=rc, seconds=sec, hook=(hook[0] if hook else None), hook_result=note,
                            messages=[l for l in text.splitlines() if l.startswith("e21")][-8:], resumed_ok=("e21 resumed at" in text)))
        r = dict(dir=str(d), steps=out); done.write_text(json.dumps(dict(key=dict(key, code_sha=self.code_sha[code]), result=r), default=str))
        return r

    @staticmethod
    def h_touch(d):
        with open(d / "arch17.py", "a") as f:
            f.write("\n# E21b fixture: this source changed after the checkpoint\n")
        return hashlib.sha256((d / "arch17.py").read_bytes()).hexdigest()

    @staticmethod
    def h_other_birth(d):
        """Another birth under the same name: the weights perturbed, admitted by a manifest written for it (test directory only)."""
        import torch
        b = d / "exp45" / "e11_dev_M_73129.pt"; sd = torch.load(b, weights_only=True)
        k = next(k for k, v in sd.items() if v.is_floating_point()); sd[k] = sd[k] + 1e-3
        b.unlink(); torch.save(sd, b); sha = hashlib.sha256(b.read_bytes()).hexdigest()
        man = d / "exp45" / "BIRTH_MANIFEST.json"; mj = json.loads(man.read_text()); mj[b.name] = dict(mj[b.name], sha256=sha)
        man.unlink(); man.write_text(json.dumps(mj, indent=1))
        return sha

    @staticmethod
    def h_rename(a, b):
        def f(d):
            moved = []
            for p in (d / "exp45").iterdir():
                if a in p.name and ".ckpt" in p.name:
                    p.rename(p.with_name(p.name.replace(a, b))); moved.append(p.name.replace(a, b))
            return sorted(moved)
        return f

    @staticmethod
    def reseal(ck, edit):
        import io
        import torch
        raw = ck.read_bytes(); obj = torch.load(io.BytesIO(raw[40:]), weights_only=True); edit(obj)
        buf = io.BytesIO(); torch.save(obj, buf); body = buf.getvalue(); ck.write_bytes(raw[:8] + hashlib.sha256(body).digest() + body)

    def h_torch(self, d):
        ck = next((d / "exp45").glob("*.ckpt"))
        def edit(o):
            c = json.loads(o["config"]); c["torch_version"] = "0.0.0+edited"; o["config"] = json.dumps(c, sort_keys=True)
        self.reseal(ck, edit); Path(str(ck) + ".prev").unlink()
        return "torch_version edited to 0.0.0+edited in .ckpt (re-sealed); .ckpt.prev removed"

    @staticmethod
    def h_save_prev(d):
        prev = next((d / "exp45").glob("*.ckpt.prev")); shutil.copy2(prev, d / "saved_stage1.ckpt.prev")
        return "copied .ckpt.prev aside"

    @staticmethod
    def h_restore_prev(d):
        prev = next((d / "exp45").glob("*.ckpt.prev")); shutil.copy2(d / "saved_stage1.ckpt.prev", prev)
        return "the run's own stage-1 checkpoint put back as .ckpt.prev beside its stage-4 .ckpt"

    @staticmethod
    def h_prev_from(other):
        def f(d):
            prev = next((d / "exp45").glob("*.ckpt.prev")); shutil.copy2(next((Path(other) / "exp45").glob("*.ckpt.prev")), prev)
            return f".ckpt.prev replaced by the one of another run of the same life ({Path(other).name})"
        return f

    def h_receipt_state(self, d):
        r = [p for p in (d / "exp45").glob("*.json") if p.name.startswith(("e16_life", "e17_life", "e17_dev_bank2_nbd_lh_bT"))]
        if not r:       # the kill came before any receipt of this run was in place: nothing torn to be read
            return dict(receipt=None, state="absent", tmp=sorted(p.name for p in (d / "exp45").glob("*.tmp")))
        r = r[0]; raw = r.read_text()
        try:
            json.loads(raw); st = "parses"
        except ValueError:
            st = "torn"
        return dict(receipt=r.name, state=st, tmp=sorted(p.name for p in (d / "exp45").glob("*.tmp")))

    def h_truncate(self, d):
        r = self.receipt(d, None); raw = r.read_bytes(); r.write_bytes(raw[:len(raw) // 2])
        return f"{r.name} cut to {len(raw) // 2} of {len(raw)} bytes"

    def h_qual(self, shape, mode):
        """Writes a qualification record into the test directory, as the production file is written: the sources the directory's harness binds
        (its own RESUME_SOURCES), this runtime, a check record in the directory with its hash, and this shape listed. Then one thing is made wrong:
        mode 'wrong_source' (exp45.py's hash), 'not_listed' (another shape), 'torch' / 'dtype' (the qualified runtime), 'record_altered' (the record
        changed after), 'record_absent', 'dependency' (mod97_local.py changed after: a transitive import), or 'listed' (nothing)."""
        def f(d):
            env = self.env(shape, 16); arm = SHAPES[shape][1][0]
            info = json.loads(subprocess.check_output([sys.executable, "-c", "import json, torch, exp16_integrated as H; print(json.dumps(dict(src=H.source_hashes(), "
                                                       f"torch=torch.__version__, dtype=str(torch.get_default_dtype()), shape=(H.resume_shape({arm!r}) if hasattr(H, 'resume_shape') else None))))"],
                                                      cwd=d, env=env, text=True).strip().splitlines()[-1])
            srcs = info["src"]
            if mode == "wrong_source":
                srcs["exp45.py"] = "0" * 64
            shapes = [info["shape"]] if mode != "not_listed" else [dict(arm="none", note="a shape that is not this one")]
            cr = d / "exp45" / "E21B_FIXTURE_CHECK.json"; cr.write_text(json.dumps(dict(note="stands for the check record in a gate fixture", passed=True)))
            rec = dict(sources=srcs, torch=info["torch"], dtype=info["dtype"], shapes=shapes, check_record=cr.name,
                       check_record_sha256=hashlib.sha256(cr.read_bytes()).hexdigest(), note="E21b gate fixture, test directory only")
            if mode == "torch":
                rec["torch"] = "0.0.0+edited"
            if mode == "dtype":
                rec["dtype"] = "torch.float64"
            (d / "exp45" / "RESUME_QUALIFIED.json").write_text(json.dumps(rec, indent=1))
            if mode == "record_altered":
                with open(cr, "a") as g:
                    g.write("\n")
            if mode == "record_absent":
                cr.unlink()
            if mode == "dependency":
                with open(d / "mod97_local.py", "a") as g:
                    g.write("\n# E21b fixture: a module the harness imports only through exp45.py and exp3_state.py, changed after qualification\n")
            return dict(mode=mode, shapes=len(shapes), sources=sorted(srcs))
        return f

    def e21b_plan(self):
        S = 16; K3 = f"update:{2 * S + 3}"; R = {"E16_RESUME": "1"}; T, B = "T_drift_online_store", "birth_rule_T_local_heads"
        P = {}      # name: (shape, steps, verdict kind, verdict argument, [(able-to-fail code, ...)])
        P["source_changed"] = (T, [(K3, {}, False, None), ("plain", R, False, ("arch17.py changed", self.h_touch))], "refuse", ["different configuration", "sources.arch17.py"], ["mut_sources"])
        P["other_birth"] = (T, [(K3, {}, False, None), ("plain", R, False, ("another birth under the same name", self.h_other_birth))], "refuse", ["different configuration", "birth_sha256"], ["mut_birth"])
        P["stage_dose"] = (T, [(K3, {}, False, None), ("plain", dict(R, E11_STAGE_UPDATES="32"), False, ("checkpoint presented to the run at E11_STAGE_UPDATES=32", self.h_rename("_dry16_", "_dry32_")))],
                           "refuse", ["different configuration", "E11_STAGE_UPDATES"], ["mut_stage_dose"])
        P["dev_dose"] = (B, [("update:5096", {}, False, None), ("plain", dict(R, E11_DEV_UPDATES="12288"), False, ("checkpoint presented to the birth at E11_DEV_UPDATES=12288", self.h_rename("_dry8192_", "_dry12288_")))],
                         "refuse", ["different configuration", "E11_DEV_UPDATES"], ["mut_dev_dose"])
        P["pairs_nonconsecutive"] = (T, [(K3, {}, False, None), (K3, R, False, ("stage-1 .ckpt.prev copied aside", self.h_save_prev)), ("plain", R, False, ("stage-1 .ckpt.prev beside stage-4 .ckpt", self.h_restore_prev))],
                                     "refuse", ["not one run's two newest checkpoints", "not consecutive"], ["mut_pairing"])
        P["pairs_other_run"] = (T, None, "refuse", ["not one run's two newest checkpoints", "run id"], ["mut_pairing"])     # built in e21b_jobs (needs a second run)
        P["torch_version"] = (T, [(K3, {}, False, None), ("plain", R, False, ("torch version edited in the checkpoint", self.h_torch))], "refuse", ["different configuration", "torch_version"], ["mut_torch"])
        for sh, tag, last in ((T, "life", 6), (B, "birth", 8192)):
            P[f"final_{tag}_after_weights"] = (sh, [("publish:weights", {}, False, None), ("plain", R, False, None)], "exact", last, [])
            P[f"final_{tag}_kill_in_receipt"] = (sh, [("publish:torn", {}, False, None), ("plain", R, False, ("receipt state after the kill", self.h_receipt_state))], "exact_untorn", last, ["base"])
            P[f"final_{tag}_torn_receipt"] = (sh, [("plain", {}, False, None), ("plain", R, False, ("final receipt cut short", self.h_truncate))], "exact_setaside", last, ["base"])
        P["gate_absent"] = (T, [(K3, {}, False, None), ("plain", R, True, None)], "refuse", ["refused", "RESUME_QUALIFIED.json is absent"], [])
        P["gate_listed"] = (T, [(K3, {}, False, None), ("plain", R, True, ("qualification record: these sources, this shape", self.h_qual(T, "listed")))], "exact_qualified", 2, [])
        P["gate_not_listed"] = (T, [(K3, {}, False, None), ("plain", R, True, ("qualification record: these sources, another shape", self.h_qual(T, "not_listed")))], "refuse", ["refused", "not on the qualified list"], ["base"])
        P["gate_other_source"] = (T, [(K3, {}, False, None), ("plain", R, True, ("qualification record: exp45.py hash wrong", self.h_qual(T, "wrong_source")))], "refuse", ["refused", "other sources", "exp45.py"], [])
        # E21b amendments (status note, 12 Nov): through the production gate, without the allowance
        P["gate_dependency_changed"] = (T, [(K3, {}, False, None), ("plain", R, True, ("qualified, then mod97_local.py changed", self.h_qual(T, "dependency")))],
                                        "refuse", ["refused", "other sources", "mod97_local.py"], ["mut_closure"])
        P["gate_runtime_torch"] = (T, [(K3, {}, False, None), ("plain", R, True, ("qualification names torch 0.0.0+edited", self.h_qual(T, "torch")))],
                                   "refuse", ["refused", "runtime is not the qualified one", "0.0.0+edited"], ["mut_runtime"])
        P["gate_runtime_dtype"] = (T, [(K3, {}, False, None), ("plain", R, True, ("qualification names dtype torch.float64", self.h_qual(T, "dtype")))],
                                   "refuse", ["refused", "runtime is not the qualified one", "torch.float64"], ["mut_runtime"])
        P["gate_record_altered"] = (T, [(K3, {}, False, None), ("plain", R, True, ("check record changed after qualification", self.h_qual(T, "record_altered")))],
                                    "refuse", ["refused", "does not have the hash"], ["mut_record"])
        P["gate_record_absent"] = (T, [(K3, {}, False, None), ("plain", R, True, ("check record removed after qualification", self.h_qual(T, "record_absent")))],
                                   "refuse", ["refused", "not on disk"], ["mut_record"])
        D0 = dict(E16_FARHEAD="0", E16_FARCTRL="none", E16_HEAD1LOSS="answer", E16_LOSSW="uniform")     # the four options a301ae8e recorded only when set
        P["explicit_default_on_resume"] = (T, [(K3, {}, False, None), ("plain", dict(R, **D0), False, None)], "exact", 2, ["base"])
        P["explicit_default_on_start"] = (T, [(K3, dict(D0), False, None), ("plain", R, False, None)], "exact", 2, ["base"])
        P["explicit_default_gate"] = (T, [(K3, {}, False, None), ("plain", dict(R, **D0), True, ("qualification record: these sources, this shape", self.h_qual(T, "listed")))],
                                      "exact_qualified", 2, [])
        return P

    def e21b_jobs(self, pool, fut):
        S = 16; K3 = f"update:{2 * S + 3}"; R = {"E16_RESUME": "1"}; T = "T_drift_online_store"
        for name, (sh, steps, kind, arg, fails) in self.e21b_plan().items():
            for code in ["new"] + fails:
                if name == "pairs_other_run":
                    def two(code=code, name=name):     # bound now: a late-bound name ran this fixture under the last fixture's directory
                        x = self.fx(f"e21b_{name}_X_{code}", code, T, [(K3, {}, False, None)])
                        return self.fx(f"e21b_{name}_{code}", code, T, [(K3, {}, False, None), ("plain", R, False, ("another run's .ckpt.prev", self.h_prev_from(x["dir"])))])
                    fut[("E21B", name, code)] = pool.submit(two)
                else:
                    fut[("E21B", name, code)] = pool.submit(self.fx, f"e21b_{name}_{code}", code, sh, steps)
        for code in ("new", "base"):
            for resume in (False, True):
                fut[("E21B_legacy", code, resume)] = pool.submit(self.legacy_rerun, code, resume)
        for sh in ("T_drift_online_store", "birth_rule_T_local_heads"):     # the uninterrupted reference (also a job of the small plan)
            fut[("E21B_A", sh)] = pool.submit(self.life_run, f"{sh}_A", "new", sh, 8192 if SHAPES[sh][1][0] == "dev" else S, ["plain"])

    def legacy_rerun(self, code, resume):
        """E21b default path: a plain rerun of a birth written before the completion record existed (its receipt has no `complete` and no weights hash,
        and there is no checkpoint) must still say 'nothing to do', with resume off and on; the receipt and weights untouched."""
        name = f"e21b_legacy_birth_rerun_{'resume' if resume else 'plain'}_{code}"
        key = dict(code=code, resume=resume, code_sha=self.code_sha[code]); done = self.work / "runs" / name / "job.json"
        if done.exists() and json.loads(done.read_text()).get("key") == key:
            return json.loads(done.read_text())["result"]
        d = self.run_dir(name, code); log = d / "run.log"; stem = "e17_dev_bank2_nbd_73129"
        before = {x: hashlib.sha256((d / "exp45" / x).read_bytes()).hexdigest() for x in (stem + ".json", stem + ".pt")}
        e = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}; e.update(E16_ARCH="bank2", E16_BOOKDECAY="off")
        if resume:
            e.update(E16_RESUME="1", E16_RESUME_UNQUALIFIED="allow")
        rc, sec = self.proc(d, e, ["dev", SEED], "plain", log)
        after = {x: hashlib.sha256((d / "exp45" / x).read_bytes()).hexdigest() for x in (stem + ".json", stem + ".pt")}
        text = log.read_text().split("\n### ")[-1]
        r = dict(dir=str(d), rc=rc, seconds=sec, nothing_to_do=("nothing to do" in text), files_unchanged=(before == after),
                 set_aside=sorted(p.name for p in (d / "exp45").glob(stem + ".json.*")), last_lines=text.strip().splitlines()[-3:])
        done.write_text(json.dumps(dict(key=key, result=r), default=str))
        return r

    def e21b_verdict(self, r, sh, kind, arg, ref):
        st = r["steps"]; last = st[-1]; v = dict(steps=[{k: x[k] for k in ("spec", "rc", "hook", "hook_result", "messages")} for x in st])
        kills_ok = all((x["rc"] == -signal.SIGKILL) == (x["spec"] not in ("plain", "none")) for x in st[:-1])
        if kind == "refuse":
            text = " ".join(last["messages"])
            v["ok"] = kills_ok and last["rc"] not in (0, -signal.SIGKILL) and all(a in text for a in arg) and not last["resumed_ok"]
            return v
        if last["rc"] != 0 or not kills_ok:
            v["ok"] = False; v["why"] = "a step exited wrongly"; return v
        d = Path(r["dir"]); B = self.outcome(d, sh); c = self.compare(ref, B, digests=False)
        res_list = B["receipt"].get("resumed", []); v.update(c); v["resumed"] = res_list
        v["ok"] = c["ok"] and bool(res_list) and res_list[-1]["from_stage"] == arg
        if kind == "exact_untorn":
            v["ok"] = v["ok"] and last["hook_result"]["state"] in ("parses", "absent")
        if kind == "exact_setaside":
            aside = sorted(p.name for p in (d / "exp45").glob("*.torn.*")); v["set_aside"] = aside; v["ok"] = v["ok"] and len(aside) == 1
        if kind == "exact_qualified":
            v["ok"] = v["ok"] and res_list[-1].get("qualification") == "qualified"
        return v

    def static_checks(self, code):
        """E21b amendments 1 and 2, read from the code tree: (1) RESUME_SOURCES is exactly the modules of this folder that exp16_integrated.py imports,
        directly or through another, found by walking the import statements (and __import__/import_module calls with a literal name); (2) every
        E11_*/E16_*/EXP45_* variable any of those modules reads is a key of run_config(), apart from E16_RESUME and E16_RESUME_UNQUALIFIED."""
        import ast
        import re
        d = self.code[code]; local = {p.stem for p in d.glob("*.py")}

        def deps(m):
            out = set()
            for n in ast.walk(ast.parse((d / f"{m}.py").read_text())):
                if isinstance(n, ast.Import):
                    out |= {a.name.split(".")[0] for a in n.names}
                elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                    out.add(n.module.split(".")[0])
                elif isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) in ("__import__", "import_module") and n.args \
                        and isinstance(n.args[0], ast.Constant):
                    out.add(str(n.args[0].value).split(".")[0])
            return out & local
        seen, todo = set(), ["exp16_integrated"]
        while todo:
            m = todo.pop()
            if m not in seen:
                seen.add(m); todo += sorted(deps(m))
        closure = sorted(f"{m}.py" for m in seen)
        e = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}
        info = json.loads(subprocess.check_output([sys.executable, "-c", "import json, exp16_integrated as H; print(json.dumps(dict(src=list(H.RESUME_SOURCES), "
                                                   "cfg=sorted(H.run_config()))))"], cwd=d, env=e, text=True).strip().splitlines()[-1])
        reads = sorted({v for f in closure for v in re.findall(r"environ(?:\.get)?[\[(]\s*[\"']((?:E11|E16|EXP45)_[A-Z0-9_]+)", (d / f).read_text())})
        missing = [v for v in reads if v not in info["cfg"] and v not in ("E16_RESUME", "E16_RESUME_UNQUALIFIED")]
        return dict(closure=closure, resume_sources=sorted(info["src"]), closure_ok=closure == sorted(info["src"]),
                    options_read=reads, options_missing_from_config=missing, options_ok=not missing)

    def e21b_evaluate(self, res):
        for code in ("new", "base"):     # base (a301ae8e) must fail both: five sources, and four options recorded only when set
            st = self.static_checks(code); ok = st["closure_ok"] if code == "new" else not st["closure_ok"]
            self.record(test="6 E21b source closure" + ("" if code == "new" else ": able to fail"), shape="exp16_integrated.py",
                        kill=("this harness" if code == "new" else "against base"), ok=ok, **{k: st[k] for k in ("closure", "resume_sources")})
            ok = st["options_ok"] if code == "new" else not st["options_ok"]
            self.record(test="6 E21b options by effective value" + ("" if code == "new" else ": able to fail"), shape="run_config",
                        kill=("this harness" if code == "new" else "against base"), ok=ok, **{k: st[k] for k in ("options_read", "options_missing_from_config")})
        ref = {sh: self.outcome(Path(res[("E21B_A", sh)]["dir"]), sh) for sh in ("T_drift_online_store", "birth_rule_T_local_heads")}
        for resume in (False, True):
            n, b = res[("E21B_legacy", "new", resume)], res[("E21B_legacy", "base", resume)]
            ok = all(x["rc"] == 0 and x["nothing_to_do"] and x["files_unchanged"] and not x["set_aside"] for x in (n, b))
            self.record(test="5 E21b default path: rerun of a birth older than the completion record", shape="e17_dev_bank2_nbd_73129",
                        kill=f"E16_RESUME={'1' if resume else 'off'}", ok=ok, new=n, base=b)
        for name, (sh, steps, kind, arg, fails) in self.e21b_plan().items():
            v = self.e21b_verdict(res[("E21B", name, "new")], sh, kind, arg, ref[sh])
            self.record(test="5 E21b fixture", shape=sh, kill=name, **v)
            for code in fails:     # the same fixture against a harness without the guard must fail
                w = self.e21b_verdict(res[("E21B", name, code)], sh, kind, arg, ref[sh])
                self.record(test="5 E21b fixture: able to fail", shape=sh, kill=f"{name} against {code}", ok=not w["ok"], fixture_verdict_on_defective_harness=w)

    def cost(self, A, rB):
        ck = [e for e in A["log"] if e["event"] == "checkpoint"]
        B = self.outcome(Path(rB["dir"]), "x")
        rs = [r for r in B["receipt"].get("resumed", []) if r["from_stage"]]
        per_stage = rs[-1]["cpu_replay"] / rs[-1]["from_stage"] if rs else None
        return dict(checkpoint_bytes=[e["bytes"] for e in ck], seconds_to_write=[e["seconds_to_write"] for e in ck],
                    replay_cpu_seconds_per_stage_at_512=per_stage,
                    replay_cpu_seconds_per_stage_at_6144_ESTIMATE=(per_stage * 6144 / 512 if per_stage is not None else None),
                    note="the full-dose figure is an estimate: replay cost at 512 updates per stage scaled linearly to 6,144 (panel building included)")


def write_qualification(rec_path):
    """E21b item 5: exp45/RESUME_QUALIFIED.json from a check record, only if that record passed on the full plan and the harness it tested is this
    folder's exp16_integrated.py. Also run on its own (--qualify-from RECORD) to rebuild the file from an existing record without rerunning it."""
    import socket
    import torch
    out = json.loads(Path(rec_path).read_text()); q = HERE / "exp45" / "RESUME_QUALIFIED.json"
    e0 = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}
    srcs = json.loads(subprocess.check_output([sys.executable, "-c", "import json, exp16_integrated as H; print(json.dumps(H.source_hashes()))"],
                                              cwd=HERE, env=e0, text=True).strip().splitlines()[-1])      # the harness's own list (amendment 1)
    if not (out["passed"] and out["only"] == "all" and sorted(out["shapes"]) == sorted(SHAPES) and not out.get("dirty")
            and out["tested_sha256"]["exp16_integrated.py"] == srcs["exp16_integrated.py"] and out["machine"]["torch"] == torch.__version__):
        print("RESUME_QUALIFIED.json NOT written: the record is not a pass of the full plan on this harness and torch", flush=True)
        return False
    shapes = []
    for name, (env, argv) in SHAPES.items():      # the command shapes that passed, as the harness itself states them
        e = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}; e.update(env)
        sh = json.loads(subprocess.check_output([sys.executable, "-c", f"import json, exp16_integrated as H; print(json.dumps(H.resume_shape({argv[0]!r})))"],
                                                cwd=HERE, env=e, text=True).strip().splitlines()[-1])
        shapes.append(dict(sh, check_name=name))
    rec = dict(sources=srcs, torch=torch.__version__, dtype=str(torch.get_default_dtype()), host=socket.gethostname(), machine=out["machine"],
               shapes=[{k: v for k, v in x.items() if k != "check_name"} for x in shapes], shape_names=[x["check_name"] for x in shapes],
               check_record=Path(rec_path).name, check_record_sha256=hashlib.sha256(Path(rec_path).read_bytes()).hexdigest(),
               brief="OPUS_TASK_E21B_RESUME_QUALIFICATION_20261111.md",
               qualified="exactly the shapes listed: the gate compares the whole shape, so a setting is qualified only in a listed combination. "
                         "Two listed shapes use the default uniform replay (D_reference, a life; birth_rule_T_local_heads, a birth); uniform replay "
                         "is qualified in those two shapes only",
               not_qualified="every shape not listed, including any with balanced/share/reservoir/total/oracle replay, uniform replay outside the two listed "
                             "shapes, E16_FREEZE, E16_HEADS=aux and relay, E16_RSHARE other than 0.5, the far-head options (E16_FARHEAD, E16_FARCTRL), "
                             "E16_HEAD1LOSS and E16_LOSSW=balanced")
    q.write_text(json.dumps(rec, indent=1)); print("wrote", q, flush=True)
    return True


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child(sys.argv[2], sys.argv[3:])
    if len(sys.argv) > 1 and sys.argv[1] == "--qualify-from":
        sys.exit(0 if write_qualification(Path(sys.argv[2])) else 1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--work"); ap.add_argument("--base-ref", default="a301ae8e")
    ap.add_argument("--jobs", type=int, default=3); ap.add_argument("--only", default="all", choices=("small", "large", "e21b", "all"))
    ap.add_argument("--shapes", default="", help="comma-separated subset of the shapes (default: all)")
    ap.add_argument("--out", default=str(HERE / "exp45" / "E21B_RESUME_CHECK.json"))
    ap.add_argument("--write-qualification", action="store_true", help="E21b: write exp45/RESUME_QUALIFIED.json if the full plan passed on a clean harness")
    a = ap.parse_args()
    import torch
    c = Check(a); t = time.time(); c.run()
    out = dict(brief="OPUS_TASK_E21_RESUMABLE_LIVES_20261110.md; E21b: OPUS_TASK_E21B_RESUME_QUALIFICATION_20261111.md", base_commit=c.base_commit,
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
    if a.write_qualification:      # E21b item 5: only when every check passed, on the full plan, with the harness as committed
        write_qualification(Path(a.out))
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
