"""Independent verification of E60 ("the sweep's long life from back-propagation's birth"), arm `sweep-after-naive`, worlds 1-4,
on the branch origin/claude/e44-l2-sweep-long-lives at aecf7b3b (E60's files: those that differ from 738e2d61).

Read-only.  Nothing is trained; no file of the worker's branch is executed.  Every committed file is read with `git show` into memory.
The world is the reviewer's second implementation (e44_world_reader.World); D2-D10 come from the reviewer's e44_l2_checks.l2_quantities;
the remaining per-life machinery (stream regeneration, D9, D14, D15, score-file comparison, log reading) is taken verbatim, by AST
extraction, from e44_l2_sweep_verify.py (the script that verified the arms `sweep` and `sweep+replay` on this branch); that file is
read, not edited and not imported as a module (its module-level code is bound to another host).

    cd <repo>; OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/homebrew/bin/python3.12 -I <this file>

Parts: A settings/completeness  B per world stream/events/logs/quantities  C readings (pairs, intervals, words, sentences, D13,
predictions)  D final/ tables and JSON  E report text  F checks  G saved states in history  H form.
"""
import ast
import calendar
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

T0 = time.time()
ROOT = Path("/private/tmp/claude-501/-Users-pencilbard-Downloads-second-try-copy-BYTE-CORE-FEAS-1-20260923/55245d29-91ee-4093-b8cd-a37e21c17e90/scratchpad/repo")
os.chdir(ROOT)
REV = ROOT / "PROJECT_DOCUMENTS" / "MINIFLY_REVIEW_20261109"
sys.path.insert(0, str(REV))
import e44_l2_checks as C                                                    # noqa: E402
from e44_world_reader import World                                           # noqa: E402

HEAD = "aecf7b3b"           # E60's commit
PREV = "738e2d61"           # the previously verified state (naive/sweep score files, sweep lives)
W1 = "origin/claude/e44-l2-first-long-lives"   # naive lives
REPC = "ba912c03"           # commit that adds the report
M = "MOD97_LOCAL_RULE_20261106/"
L2 = M + "exp45/e44/L2/"
SW = L2 + "sweep/"
SAN = L2 + "sweep_after_naive/"
WORLDS = (1, 2, 3, 4)
W3 = (1, 2, 3)
SEED = {1: 73129, 2: 93683, 3: 116089, 4: 73129}
VS = (256, 1024)
PER, VU = 16 * 256, 256
CONTROL_FAMILY = 55
T95_3 = 2.353363434801823
TAB = {4: 2.3534, 3: 2.9200}                 # l2_score.T95[n-1], the tabled values the brief prescribes (via l2_score.interval90)
EXACT = {4: 2.353363434801823, 3: 2.919985580355516}
ABS = (("1 visit", 1, 1), ("2 to 4", 2, 4), ("5 to 32", 5, 32), ("33 or more", 33, 10 ** 9), ("5 or more", 5, 10 ** 9))
EARLIER = (("1", 1, 1), ("2 to 3", 2, 3), ("4 to 7", 4, 7), ("8 to 15", 8, 15), ("16 or more", 16, 10 ** 9))
ORDER = C.ORDER
N_CMP = [0]
DIFFS = []
TXT_N = [0]
TXT_DIFFS = []
FIND = []          # findings noted by the script (text)

# ---- verbatim functions from e44_l2_sweep_verify.py (AST extraction; the file is only read)
_src = (REV / "e44_l2_sweep_verify.py").read_text()
_want = {"sha", "arrays_sha", "cmp", "tmean", "regen", "later_records", "kept_tables", "by_earlier", "list_shares", "d9", "classify",
         "d14_stream", "d14_list", "quantities", "compare_score_file", "log_runs", "pool_d15"}
_mod = ast.parse(_src)
_got = [n for n in _mod.body if isinstance(n, ast.FunctionDef) and n.name in _want]
assert {n.name for n in _got} == _want, _want - {n.name for n in _got}
exec(compile(ast.Module(body=_got, type_ignores=[]), "e44_l2_sweep_verify.py(extracted)", "exec"), globals())
_cmp_scalar = cmp


def cmp(label, mine, theirs, tol=1e-9):          # lists/dicts compared by equality; scalars by the extracted rule
    if isinstance(mine, (list, tuple, dict)) or isinstance(theirs, (list, tuple, dict)):
        N_CMP[0] += 1
        ok = list(mine) == list(theirs) if not isinstance(mine, dict) else mine == theirs
        if not ok:
            DIFFS.append((label, mine, theirs)); print(f"   DIFFERS: {label}: recomputed {mine}, committed {theirs}")
        return ok
    return _cmp_scalar(label, mine, theirs, tol)


print(f"helpers: e44_l2_sweep_verify.py sha256 {hashlib.sha256(_src.encode()).hexdigest()[:16]}...; extracted {sorted(_want)}")
print(f"         e44_l2_checks.py {hashlib.sha256((REV / 'e44_l2_checks.py').read_bytes()).hexdigest()[:16]}..., "
      f"e44_world_reader.py {hashlib.sha256((REV / 'e44_world_reader.py').read_bytes()).hexdigest()[:16]}...")


def git(*a):
    r = subprocess.run(["git", *a], capture_output=True, cwd=str(ROOT))
    assert r.returncode == 0, (a, r.stderr[:300])
    return r.stdout


def gb(path, ref=HEAD):
    return git("show", f"{ref}:{path}")


def gjson(path, ref=HEAD):
    return json.loads(gb(path, ref))


def gnpz(path, ref=HEAD):
    f = np.load(io.BytesIO(gb(path, ref)))
    return {k: f[k] for k in f.files}


def blob(ref, path):
    return git("rev-parse", f"{ref}:{path}").decode().strip()


def free_gb():
    st = os.statvfs(str(ROOT))
    return st.f_bavail * st.f_frsize / 1e9


def iv(xs, n_expected, exact=False):
    xs = [float(x) for x in xs]
    assert len(xs) == n_expected
    n = len(xs); m = float(np.mean(xs)); sd = float(np.std(xs, ddof=1))
    t = (EXACT if exact else TAB)[n]
    h = t * sd / math.sqrt(n)
    return dict(n=n, mean=m, sd=sd, low=m - h, high=m + h)


def word(i):
    return "higher" if i["low"] > 0 else "lower" if i["high"] < 0 else "not distinguished"


def interval90(xs, t=None):   # used by pool_d15? (no); kept for compatibility with extracted code
    return iv(xs, len(xs))


def stem(arm, w):
    return f"lw_{arm}_w{w}_v1024"


def life_path(arm, w):
    return {"sweep-after-naive": SAN + "lives/", "sweep": SW + "lives/", "naive": L2 + "lives/"}[arm] + stem(arm, w)


def ref_of(arm):
    return {"sweep-after-naive": HEAD, "sweep": PREV, "naive": W1}[arm]


def score_path(arm, w, V):
    return {"sweep-after-naive": SAN + "scores/", "sweep": SW + "scores/", "naive": SW + "main/"}[arm] + f"{stem(arm, w)}_V{V}.json"


def score_ref(arm):
    return HEAD if arm == "sweep-after-naive" else PREV


ARMS = ("sweep-after-naive", "naive", "sweep")
REC = {}
R = {}
E = {}


# ================================================================ A
def part_a():
    print("A. E60's files, settings and completeness of the four lives")
    changed = git("diff", "--name-only", PREV, HEAD).decode().split()
    outside = [p for p in changed if not (p.startswith(SAN) or p.startswith(M + "smallworld/l2_san"))]
    print(f"   files differing {PREV}..{HEAD}: {len(changed)}; outside sweep_after_naive/ and smallworld/l2_san*: {outside or 'none'}")
    st = git("diff", "--name-status", PREV, HEAD).decode().splitlines()
    print(f"   of which modified (not added): {[l for l in st if not l.startswith('A')] or 'none'}")
    # sources against the sweep's check receipt
    ck = gjson(SW + "checks/E44_L2_SWEEP_CHECKS.json", PREV)
    src = ck["sources_sha256"]; unch = ck["checks"]["1"]["unchanged_files_sha256"]
    now = {k: sha(gb(M + k)) for k in src}
    print(f"   twelve sources at {HEAD} equal to sweep/checks receipt: {sum(now[k] == src[k] for k in src)} of {len(src)}")
    nu = {}
    for k, v in unch.items():
        p = k if k.startswith("smallworld/") else "smallworld/" + k
        h = v if isinstance(v, str) else v.get("sha256", v)
        nu[p] = (sha(gb(M + p)), h)
    print(f"   unchanged files: " + ", ".join(f"{k.split('/')[-1]} {a == b}" for k, (a, b) in nu.items()))
    jobs = gjson(SAN + "san_jobs.json")
    want = ["--world", "lw", "--H", "256", "--U", "128", "--visit-updates", "256", "--wd", "0.25", "--lr", "0.001", "--list-every", "8",
            "--decay-distances", "none", "--save-states", "64,128,256,384,512,640,768,896,1024"]
    for j in jobs:
        a = j["args"]; w = int(a[2])
        rest = a[4:a.index("--learner")] + a[a.index("--learner") + 2:]
        ok = (a[:2] == ["life", "sweep-after-naive"] and a[3] == "1024" and rest == want and a[a.index("--learner") + 1].endswith(f"e41_B_h256_f50_{SEED[w]}.pt")
              and j["out"].endswith("sweep_after_naive/lives") and j["stem"] == stem("sweep-after-naive", w))
        cmp(f"jobs {j['id']} args as the brief", ok, True)
    print(f"   san_jobs.json: {len(jobs)} jobs: {[j['id'] for j in jobs]}")
    for arm in ARMS:
        for w in WORLDS:
            REC[arm, w] = gjson(life_path(arm, w) + ".json", ref_of(arm))
    host_sweep = {(REC['sweep', w]['provenance']['host']['python'], REC['sweep', w]['provenance']['host']['torch'], REC['sweep', w]['provenance']['host']['numpy']) for w in WORLDS}
    tot = 0.0
    for w in WORLDS:
        r = REC["sweep-after-naive", w]; st_ = r["settings"]; ln = st_["learner"]; nv = REC["naive", w]
        ok = (r["complete"] is True and r["final"]["visits"] == 1024 and r["final"]["updates"] == 262144 and len(r["by_visit"]) == 1024
              and [b["visit"] for b in r["by_visit"]] == list(range(1024)) and all(b["updates"] == 256 for b in r["by_visit"])
              and st_["arm"] == "sweep-after-naive" and st_["world"] == w and st_["visits"] == 1024 and st_["rho"] is None and st_["lr"] == 0.001
              and st_["wd"] == 0.25 and st_["H"] == 256 and st_["U"] == 128 and st_["visit_updates"] == 256 and st_["list_every"] == 8
              and st_["decay_distances"] == [] and st_["save_states"] == [64, 128, 256, 384, 512, 640, 768, 896, 1024] and st_["world_kind"] == "lw"
              and st_["in_turn"] is False and r["local"] is True and r["replay"] is False and r["replay_items_per_update"] == 0
              and r["optimizer"]["name"] == "AdamW" and r["optimizer"]["lr"] == 0.001 and r["optimizer"]["weight_decay"] == 0.25)
        cmp(f"san w{w} settings and completeness", ok, True)
        lok = (ln["rule"] == "B" and ln["local_in_birth"] is False and ln["seed"] == SEED[w] and ln["file"] == f"exp45/e41/e41_B_h256_f50_{SEED[w]}.pt")
        cmp(f"san w{w} birth is rule B of the world's seed", lok, True)
        lk = sorted(set(ln) | set(nv["settings"]["learner"]))
        dk = [k for k in lk if ln.get(k) != nv["settings"]["learner"].get(k)]
        cmp(f"san w{w} learner equal to naive's key for key (keys differing)", len(dk), 0)
        # the birth file itself
        bb = gb(M + ln["file"]); er = gjson(M + ln["file"].replace(".pt", ".json"))
        try:
            ts = C.tensors_sha(C.plain_weights(bb))
        except Exception as e:                                   # noqa: BLE001
            ts = f"unreadable ({e})"
        cmp(f"san w{w} birth file sha256 = learner.file_sha256", sha(bb), ln["file_sha256"])
        cmp(f"san w{w} birth file sha256 = E41 receipt weights_sha256", sha(bb), er["weights_sha256"])
        cmp(f"san w{w} birth tensors (recomputed) = learner.tensors_sha256", ts, ln["tensors_sha256"])
        cmp(f"san w{w} birth file at {HEAD} same blob as at {W1}", blob(HEAD, M + ln["file"]), blob(W1, M + ln["file"]))
        cmp(f"san w{w} E41 rule B", er["rule"], "B")
        # different from the sweep's birth?
        cmp(f"san w{w} birth differs from sweep's birth (file sha)", ln["file_sha256"] != REC["sweep", w]["settings"]["learner"]["file_sha256"], True)
        # other settings against naive
        sk = [k for k in st_ if k not in ("arm", "learner") and st_[k] != nv["settings"][k]]
        cmp(f"san w{w} settings other than arm/learner differing from naive (count)", len(sk), 0)
        cmp(f"san w{w} optimizer record equal to naive's", r["optimizer"] == nv["optimizer"], True)
        src_ok = r["provenance"]["sources_sha256"] == src and r["provenance"]["git"]["sources_modified_vs_commit"] == []
        cmp(f"san w{w} sources as sweep check receipt", src_ok, True)
        hv = r["provenance"]["host"]
        cmp(f"san w{w} python/torch/numpy as the sweep lives' receipts", (hv["python"], hv["torch"], hv["numpy"]) in host_sweep, True)
        cmp(f"san w{w} threads", hv["torch_threads"], 1)
        chain = sha("".join(b["learned_sha256"] for b in r["by_visit"]).encode()) == r["learned_sha256"]
        cmp(f"san w{w} learned digest chain", chain, True)
        fin = all(math.isfinite(b[k]) for b in r["by_visit"] for k in ("loss_mean", "loss_first", "loss_last"))
        cmp(f"san w{w} losses finite", fin, True)
        cmp(f"san w{w} resumes", len(r["resumes"]), 0)
        cmp(f"san w{w} states recorded", [x["visits"] for x in r["states"]], [64, 128, 256, 384, 512, 640, 768, 896, 1024])
        tot += r["cpu"]["total"]
        print(f"   w{w}: complete/settings {ok}; birth {ln['file'].split('/')[-1]} rule {ln['rule']} seed {ln['seed']}; learner keys differing from naive "
              f"{dk or 'none'}; birth sha ok; host {hv['python']}/{hv['torch']}/{hv['numpy']} (sweep lives: {sorted(host_sweep)}); resumes "
              f"{len(r['resumes'])}; CPU {r['cpu']['total'] / 3600:.4f} h; command {r['provenance']['command'][:60]}...")
        E[f"cpu_{w}"] = r["cpu"]["total"] / 3600
    E["cpu_tot"] = tot / 3600
    print(f"   CPU-hours total {tot / 3600:.4f}")
    # births A/R
    for sd in (73129, 93683, 116089):
        b, s = gjson(M + f"exp45/e41/e41_B_h256_f50_{sd}.json"), gjson(M + f"exp45/e41/e41_S_h256_f50_{sd}.json")
        E[f"birth_B_{sd}"] = (b["curve"][-1]["A"], b["curve"][-1]["R"]); E[f"birth_S_{sd}"] = (s["curve"][-1]["A"], s["curve"][-1]["R"])
    # starts and driver concurrency
    for w in WORLDS:
        sj = [json.loads(x) for x in gb(life_path("sweep-after-naive", w) + ".starts.jsonl").decode().splitlines() if x.strip()]
        cmp(f"san w{w} starts (count)", len(sj), 1)
        E[f"start_{w}"] = sj[0]["wall_utc"]
    dl = gb(SAN + "logs/san_driver.log").decode().splitlines()
    run, mx = set(), 0
    for ln_ in dl:
        m = re.match(r"(\S+) (started|done) (\S+)", ln_)
        if m:
            (run.add if m.group(2) == "started" else run.discard)(m.group(3)); mx = max(mx, len(run))
    print(f"   driver log: largest number of lives running at once {mx}")
    cmp("lives at most three at once", mx <= 3, True)


# ================================================================ B
def part_b():
    print("B. Per world: regenerated stream against every visit's digest; event files against the stream; logs; quantities vs score files")
    tot_n, tot_d = 0, 0
    for w in WORLDS:
        t = time.time()
        W, sched, S, digests, life = regen(w)
        print(f"   world {w}: stream regenerated in {time.time() - t:.0f} s")
        for arm in ARMS:
            r = REC[arm, w]
            bad = [v for v in range(1024) if r["by_visit"][v]["stream_sha256"] != digests[v]]
            cmp(f"{arm} w{w} visits whose stream digest differs from the regenerated", len(bad), 0)
            cmp(f"{arm} w{w} life stream digest", life, r["stream_sha256"])
            cmp(f"{arm} w{w} schedule = reviewer's world", [x["group"] for x in r["schedule"]], sched)
            print(f"      {arm}: visits whose digest differs {len(bad)}; life digest equal {life == r['stream_sha256']}")
        san_eq = sum(a["stream_sha256"] == b["stream_sha256"] for a, b in zip(REC["sweep-after-naive", w]["by_visit"], REC["naive", w]["by_visit"]))
        E[f"san_naive_eq_{w}"] = san_eq
        for arm in ARMS:
            s = stem(arm, w); p = life_path(arm, w); ref = ref_of(arm); r = REC[arm, w]
            raw = gb(p + ".npz", ref)
            f = np.load(io.BytesIO(raw)); ev = {k: f[k] for k in f.files}
            fsha = sha(raw) == r["events"]["file_sha256"]; asha = arrays_sha(ev) == r["events"]["arrays_sha256"]
            del raw, f
            cmp(f"{arm} w{w} event file sha256 and arrays sha256 = receipt", fsha and asha, True)
            upd_ok = bool((ev["update"] == np.repeat(np.arange(1024 * VU), 16)).all()); vis_ok = bool((ev["visit"] == np.repeat(np.arange(1024), PER)).all())
            g = np.stack([ev["g1"], ev["g2"], ev["g3"]], 1).astype(np.int64); tg = S["target"].astype(np.int64)
            right = (g[:, 0] == 48 + tg // 10) & (g[:, 1] == 48 + tg % 10) & (g[:, 2] == 256)
            cols = {k: bool((ev[k] == S[k]).all()) for k in ("task", "wrap", "mirror", "blind", "seen")}
            cols["correct"] = bool((ev["correct"].astype(bool) == right).all())
            cmp(f"{arm} w{w} event rows/updates/visits", upd_ok and vis_ok and len(ev["task"]) == 1024 * PER, True)
            cmp(f"{arm} w{w} event columns = stream ({cols})", all(cols.values()), True)
            sh_v = ev["correct"].reshape(1024, PER).mean(1)
            if arm == "sweep-after-naive":
                log = gb(p + ".log").decode().splitlines()
                runs, mism, dup, dup_diff, nlog = log_runs(s, log, sh_v)
                cmp(f"san w{w} log runs", len(runs), 1); cmp(f"san w{w} log visits", nlog, 1024)
                cmp(f"san w{w} log share right vs event file, differences", int(mism), 0); cmp(f"san w{w} visits logged twice", dup, 0)
                print(f"      log: runs {len(runs)}, visits logged {nlog}, share mismatches {mism}, logged twice {dup} (differing {dup_diff})")
            la = gnpz(p + "_list.npz", ref)
            cmp(f"{arm} w{w} list arrays sha256 = receipt", arrays_sha(la) == r["list"]["arrays_sha256"], True)
            for V in VS:
                q = quantities(W, sched, ev, la, V)
                q["d14_stream"] = d14_stream(W, sched, V, S, ev)
                sf = gjson(score_path(arm, w, V), score_ref(arm))
                nd, nc = compare_score_file(arm, w, V, q, sf)
                tot_n += nc; tot_d += nd
                R[arm, w, V] = q
                print(f"      {arm} V{V}: late level {q['late_level']:.6f} D5 {q['score_trend']:+.6f} D7 {q['trend_of_learning']:+.6f} "
                      f"early {q['early']:.6f}; score-file figures compared {nc:,}, differing {nd}")
            del ev, la
        del S
        print(f"   world {w} done in {time.time() - t:.0f} s")
    E["score_n"] = (tot_n, tot_d)
    print(f"   score-file figures compared (24 files: 8 new, 16 of naive/sweep at {PREV}): {tot_n:,}; differing {tot_d}")


# ================================================================ C
PAIRS = {"R": ("sweep-after-naive", "naive"), "B": ("sweep", "sweep-after-naive"), "RB": ("sweep", "naive")}
AGES = ("early", "256", "1024")
CELL = {}


def lev(arm, w, age):
    return R[arm, w, 1024]["early"] if age == "early" else R[arm, w, int(age)]["late_level"]


def cmp_iv(label, mine, theirs):
    for k in ("mean", "low", "high", "sd", "n"):
        cmp(f"{label} {k}", mine[k], theirs[k])
    cmp(f"{label} word", word(mine), theirs["word"])


def part_c():
    print("C. Readings: levels, the nine cells, pair R extras, D13, sentences, predictions (compared with final/E60.json)")
    J = gjson(SAN + "final/E60.json")
    for arm in ARMS:
        for age in AGES:
            for w in WORLDS:
                cmp(f"levels {arm}|{age} w{w}", lev(arm, w, age), J["levels"][f"{arm}|{age}"][str(w)])
    dev = 0.0
    for pn, (x, y) in PAIRS.items():
        for age in AGES:
            per = {w: lev(x, w, age) - lev(y, w, age) for w in WORLDS}
            i4, i3 = iv([per[w] for w in WORLDS], 4), iv([per[w] for w in W3], 3)
            e4, e3 = iv([per[w] for w in WORLDS], 4, True), iv([per[w] for w in W3], 3, True)
            CELL[pn, age] = dict(per=per, i4=i4, i3=i3)
            t = J["cells"][f"{pn}|{age}"]
            for w in WORLDS:
                cmp(f"cell {pn}|{age} w{w}", per[w], t["per_world"][str(w)])
            cmp_iv(f"cell {pn}|{age} over 1-4", i4, t["over worlds 1 to 4"]); cmp_iv(f"cell {pn}|{age} over 1-3", i3, t["over worlds 1 to 3"])
            if word(e4) != word(i4) or word(e3) != word(i3):
                DIFFS.append((f"cell {pn}|{age} word under the exact t", word(e4) + "/" + word(e3), word(i4) + "/" + word(i3)))
            print(f"   {pn:2s} {age:>5}: " + " ".join(f"{per[w]:+.4f}" for w in WORLDS)
                  + f" | 1-4 {i4['mean']:+.4f} [{i4['low']:+.4f}, {i4['high']:+.4f}] {word(i4)} | 1-3 {i3['mean']:+.4f} [{i3['low']:+.4f}, {i3['high']:+.4f}] {word(i3)}")
    for age in AGES:
        for w in WORLDS:
            dev = max(dev, abs(CELL["RB", age]["per"][w] - CELL["R", age]["per"][w] - CELL["B", age]["per"][w]))
    print(f"   RB - (R + B) largest |deviation| per world {dev:.3e} (E60.json says {J['RB_equals_R_plus_B_largest_deviation']})")
    E["rb_dev"] = dev
    # pair R extras
    pr = J["pair_R"]
    d5 = {w: R["sweep-after-naive", w, 1024]["score_trend"] - R["naive", w, 1024]["score_trend"] for w in WORLDS}
    ch = {w: CELL["R", "1024"]["per"][w] - CELL["R", "early"]["per"][w] for w in WORLDS}
    for nm, d, key in (("D5 difference", d5, "score_trend_difference"), ("change", ch, "change")):
        for w in WORLDS:
            cmp(f"pair R {nm} w{w}", d[w], pr[key]["per_world"][str(w)])
        a4, a3 = iv([d[w] for w in WORLDS], 4), iv([d[w] for w in W3], 3)
        cmp_iv(f"pair R {nm} 1-4", a4, pr[key]["over worlds 1 to 4"]); cmp_iv(f"pair R {nm} 1-3", a3, pr[key]["over worlds 1 to 3"])
        E[f"R_{nm}"] = (d, a4, a3)
        print(f"   pair R {nm}: " + " ".join(f"{d[w]:+.4f}" for w in WORLDS) + f" | 1-4 {a4['mean']:+.4f} [{a4['low']:+.4f}, {a4['high']:+.4f}] {word(a4)}"
              f" | 1-3 {a3['mean']:+.4f} [{a3['low']:+.4f}, {a3['high']:+.4f}] {word(a3)}")
    # D13
    for lab, ws, n in (("1 to 4", WORLDS, 4), ("1 to 3", W3, 3)):
        a = iv([R["sweep-after-naive", w, 1024]["score_trend"] for w in ws], n); b = iv([R["sweep-after-naive", w, 1024]["trend_of_learning"] for w in ws], n)
        stt = "still learning" if (a["low"] > 0 or b["low"] > 0) else "stuck"
        decl = [nm for nm, i in (("score_trend", a), ("trend_of_learning", b)) if i["high"] < 0]
        t = pr[f"D13 over worlds {lab}"]
        cmp_iv(f"D13 {lab} D5", a, t["score_trend"]); cmp_iv(f"D13 {lab} D7", b, t["trend_of_learning"])
        cmp(f"D13 {lab} statement", stt, t["statement"]); cmp(f"D13 {lab} declining", decl, t["declining"])
        E[f"D13_{lab}"] = (a, b, stt, decl)
        print(f"   D13 over worlds {lab}: D5 {a['mean']:+.4f} [{a['low']:+.4f}, {a['high']:+.4f}]; D7 {b['mean']:+.4f} [{b['low']:+.4f}, {b['high']:+.4f}]: {stt}; declining {decl or 'none'}")
    # sentences
    for lab, ws, key in (("1 to 4", WORLDS, "over worlds 1 to 4"), ("1 to 3", W3, "over worlds 1 to 3")):
        ik = "i4" if lab == "1 to 4" else "i3"
        s = J["sentences"][key]
        w1 = [word(CELL["R", a][ik]) for a in AGES]
        kinds = {}
        for w in ws:
            e, l_ = CELL["R", "early"]["per"][w], CELL["R", "1024"]["per"][w]
            kinds[w] = ("nearer to zero" if (np.sign(e) == np.sign(l_) and abs(l_) < abs(e)) else "changed sign" if np.sign(e) * np.sign(l_) < 0 else "not nearer to zero")
        chw = word(E["R_change"][1 if lab == "1 to 4" else 2])
        over = w1[0] == "lower" and w1[2] == "higher"
        smaller = [w for w in ws if CELL["R", "early"]["per"][w] < CELL["R", "1024"]["per"][w] < 0]
        rb, rr, bb = (word(CELL[p, "1024"][ik]) for p in ("RB", "R", "B"))
        if rb != "not distinguished" and rr == rb and bb == "not distinguished":
            asg = "rule"
        elif rb != "not distinguished" and bb == rb and rr == "not distinguished":
            asg = "birth"
        else:
            asg = "nothing"
        E[f"sent_{lab}"] = (w1, kinds, chw, over, smaller, asg)
        for w in ws:
            cmp(f"sentence 2 {lab} world {w} kind", kinds[w], s["world_kinds"][str(w)])
        cmp(f"sentence 2 {lab} overtaking", over, s["overtaking"]); cmp(f"sentence 2 {lab} smaller deficit", smaller, s["smaller_deficit"])
        cmp(f"sentence 2 {lab} change word in text", f" is {chw} (" in s["sentence_2"], True)
        exp_asg = {"rule": "RB is assigned to the rule.", "birth": "RB is assigned to the birth.", "nothing": None}[asg]
        cmp(f"sentence 3 {lab} assignment", (exp_asg == s["assignment"]) if exp_asg else s["assignment"].startswith("Nothing is assigned"), True)
        # sentence 1 words in text
        txt1 = s["sentence_1"]
        found = re.findall(r"(lower than|higher than|not distinguished from)", txt1)
        cmp(f"sentence 1 {lab} words in text", [f.split()[0] if f != "not distinguished from" else "not distinguished" for f in found],
            ["not distinguished" if x == "not distinguished" else x for x in w1])
        print(f"   sentences over worlds {lab}: (1) {w1}; (2) kinds {kinds}, change {chw}, overtaking {over}, smaller deficit in {smaller}; (3) assigned to: {asg}")
    # births
    for sd in (73129, 93683, 116089):
        for rl in "BS":
            cmp(f"births {rl}_{sd} A", E[f"birth_{rl}_{sd}"][0], J["births"][f"{rl}_{sd}"]["A"]); cmp(f"births {rl}_{sd} R", E[f"birth_{rl}_{sd}"][1], J["births"][f"{rl}_{sd}"]["R"])
    # against naive
    for w in WORLDS:
        t = J["against_naive"][str(w)]; r = REC["sweep-after-naive", w]
        cmp(f"against naive w{w} equal visits", E[f"san_naive_eq_{w}"], t["equal"]); cmp(f"against naive w{w} stream", r["stream_sha256"], t["stream_sha256"])
        cmp(f"against naive w{w} stream naive", REC["naive", w]["stream_sha256"], t["stream_sha256_naive"])
        cmp(f"against naive w{w} cpu", r["cpu"]["total"] / 3600, t["cpu_hours"]); cmp(f"against naive w{w} resumes", len(r["resumes"]), len(t["resumes"]))
        cmp(f"against naive w{w} events sha", r["events"]["file_sha256"], t["events_file_sha256"]); cmp(f"against naive w{w} list sha", r["list"]["file_sha256"], t["list_file_sha256"])
    # predictions
    P = {}
    for lab, ik in (("1 to 3", "i3"), ("1 to 4", "i4")):
        w1, kinds, chw, over, smaller, asg = E[f"sent_{lab}"]
        P[lab] = {"P60.1": word(CELL["R", "early"][ik]) == "lower", "P60.2": word(CELL["R", "1024"][ik]),
                  "P60.3": all(k == "nearer to zero" for k in kinds.values()), "P60.4": word(CELL["B", "1024"][ik]) == "not distinguished",
                  "P60.5": E[f"D13_{lab}"][2] == "still learning"}
        print(f"   predictions over worlds {lab}: P60.1 {P[lab]['P60.1']}; P60.2 (lower?) {P[lab]['P60.2'] == 'lower'} (word {P[lab]['P60.2']}); "
              f"P60.3 {P[lab]['P60.3']}; P60.4 {P[lab]['P60.4']}; P60.5 {P[lab]['P60.5']}")
    E["P"] = P
    # curves csv, per visit
    for w in WORLDS:
        rows = gb(SAN + f"final/curves_w{w}.csv").decode().splitlines()
        hdr = rows[0].split(","); n_bad = 0
        for v, ln_ in enumerate(rows[1:]):
            c = ln_.split(",")
            cmp(f"curves w{w} visit {v} index", int(c[0]), v); cmp(f"curves w{w} visit {v} kind", c[1], R["naive", w, 1024]["kind"][v])
            for k, h in enumerate(hdr[2:]):
                arm = h.split(":")[0]
                ok = abs(float(c[2 + k]) - R[arm, w, 1024]["score_by_age"][v]) <= 5e-7 + 1e-12
                N_CMP[0] += 1
                if not ok:
                    n_bad += 1; DIFFS.append((f"curves w{w} v{v} {arm}", R[arm, w, 1024]["score_by_age"][v], c[2 + k]))
        cmp(f"curves w{w} rows", len(rows) - 1, 1024)
        print(f"   curves_w{w}.csv: {len(rows) - 1} rows, columns {hdr}; values differing beyond rounding {n_bad}")


# ================================================================ D. tables file
def fnum(s):
    return float(s.replace("−", "-").replace(",", ""))


IVRE = r"([+\-−]?\d+\.\d+) \[([+\-−]?\d+\.\d+), ([+\-−]?\d+\.\d+)\],? ?(lower|higher|not distinguished)?"


def chk_iv(label, cell, mine, dec, src="T"):
    m = re.search(IVRE, cell)
    if not m:
        DIFFS.append((label + " (not parsed)", None, cell)); return
    for k, g_ in zip(("mean", "low", "high"), m.groups()[:3]):
        txt(f"{src} {label} {k}", fnum(g_), mine[k], dec)
    if m.group(4):
        TXT_N[0] += 1
        if m.group(4) != word(mine):
            TXT_DIFFS.append((f"{src} {label} word", m.group(4), word(mine), 0))


def txt(label, reported, mine, dec):
    TXT_N[0] += 1
    d = float(mine) - float(reported)
    ok = abs(d) <= 0.5 * 10 ** -dec + 1e-9
    if not ok:
        TXT_DIFFS.append((label, reported, mine, d))
    return ok


def table_rows(text, header_start):
    lines = text.splitlines()
    i = next(k for k, l_ in enumerate(lines) if l_.startswith(header_start))
    out = []
    for l_ in lines[i + 2:]:
        if not l_.startswith("|"):
            break
        out.append([c.strip() for c in l_.strip().strip("|").split("|")])
    return out


AGE_OF = {"first 64": "early", "256": "256", "1,024": "1024"}


def age_of(cell):
    return next(v for k, v in AGE_OF.items() if k in cell)


def part_d(text=None, src="TABLES"):
    md = text if text is not None else gb(SAN + "final/E60_TABLES.md").decode()
    print(f"D/E. Figures of {src} against the recomputation")
    n0, d0 = TXT_N[0], len(TXT_DIFFS)
    # levels
    for row in table_rows(md, "| Arm | Age |"):
        arm = row[0].strip("`"); age = age_of(row[1])
        for k, w in enumerate(WORLDS):
            txt(f"{src} level {arm} {age} w{w}", fnum(row[2 + k]), lev(arm, w, age), 3)
    # nine cells
    for row in table_rows(md, "| Pair | Age |"):
        pn = row[0].split(":")[0].split(",")[0].strip("* ").split()[0]; age = age_of(row[1])
        for k, w in enumerate(WORLDS):
            txt(f"{src} cell {pn} {age} w{w}", fnum(row[2 + k]), CELL[pn, age]["per"][w], 4)
        chk_iv(f"cell {pn} {age} 1-4", row[6], CELL[pn, age]["i4"], 3, src); chk_iv(f"cell {pn} {age} 1-3", row[7], CELL[pn, age]["i3"], 3, src)
    # pair R table
    for row in table_rows(md, "| Quantity |"):
        nm = "D5 difference" if "D5" in row[0] else "change"
        d, a4, a3 = E[f"R_{nm}"]
        for k, w in enumerate(WORLDS):
            txt(f"{src} pair R {nm} w{w}", fnum(row[1 + k]), d[w], 4)
        dec = len(re.search(IVRE, row[5]).group(1).split(".")[1])
        chk_iv(f"pair R {nm} 1-4", row[5], a4, dec, src); chk_iv(f"pair R {nm} 1-3", row[6], a3, dec, src)
    print(f"   {src}: figures compared so far {TXT_N[0] - n0}, differing {len(TXT_DIFFS) - d0}")


def part_d_tables():
    md = gb(SAN + "final/E60_TABLES.md").decode()
    part_d(md, "TABLES")
    n0, d0 = TXT_N[0], len(TXT_DIFFS)
    for row in table_rows(md, "| Reading | Score trend"):
        lab = "1 to 4" if "1 to 4" in row[0] else "1 to 3"
        a, b, stt, decl = E[f"D13_{lab}"]
        chk_iv(f"D13 {lab} D5", row[1], a, 4, "TABLES"); chk_iv(f"D13 {lab} D7", row[2], b, 4, "TABLES")
        TXT_N[0] += 1
        if row[3] != stt:
            TXT_DIFFS.append((f"TABLES D13 {lab} statement", row[3], stt, 0))
    for row in table_rows(md, "| Seed | Rule of the birth"):
        sd, rl = int(row[0]), row[1]
        txt(f"TABLES birth {rl} {sd} A", fnum(row[2]), E[f"birth_{rl}_{sd}"][0], 2); txt(f"TABLES birth {rl} {sd} R", fnum(row[3]), E[f"birth_{rl}_{sd}"][1], 2)
    for row in table_rows(md, "| World | Visits with equal"):
        w = int(row[0])
        txt(f"TABLES against naive w{w} equal", fnum(row[1].split()[0]), E[f"san_naive_eq_{w}"], 0)
        txt(f"TABLES against naive w{w} cpu", fnum(row[5]), E[f"cpu_{w}"], 2)
        txt(f"TABLES against naive w{w} resumes", fnum(row[4]), 0, 0)
    # curves blocks
    lines = md.splitlines()
    for w in WORLDS:
        i = lines.index(f"**World {w}.**")
        rows = table_rows("\n".join(lines[i:]), "| Visits |")
        hdr = ["naive", "sweep-after-naive", "sweep"]
        for b, row in enumerate(rows):
            for k, arm in enumerate(hdr):
                txt(f"TABLES curve w{w} block {b} {arm}", fnum(row[1 + k]), float(np.mean(R[arm, w, 1024]["score_by_age"][64 * b:64 * b + 64])), 3)
    # sentences in the tables
    print(f"   TABLES (D13, births, against naive, curves): compared {TXT_N[0] - n0}, differing {len(TXT_DIFFS) - d0}")


# ================================================================ E. report text
def part_e():
    rep = gb(SAN + "E60_REPORT.md").decode()
    E["report"] = rep
    part_d(rep, "REPORT")
    n0, d0 = TXT_N[0], len(TXT_DIFFS)
    # births table in the report (section 4)
    for row in table_rows(rep, "| Seed | Birth of rule B"):
        sd = int(row[0])
        for k, rl in ((1, "B"), (2, "S")):
            a_, r_ = row[k].split("/")
            txt(f"REPORT birth {rl} {sd} A", fnum(a_), E[f"birth_{rl}_{sd}"][0], 2); txt(f"REPORT birth {rl} {sd} R", fnum(r_), E[f"birth_{rl}_{sd}"][1], 2)
    # D13 lines
    for lab in ("1 to 4", "1 to 3"):
        a, b, stt, decl = E[f"D13_{lab}"]
        m = re.search(rf"Over worlds {lab}: score trend {IVRE}, trend of learning {IVRE}\. It is \*\*(\w+ \w+)\*\*", rep)
        if not m:
            TXT_DIFFS.append((f"REPORT D13 {lab} line not parsed", None, None, 0)); continue
        for k, kk in enumerate(("mean", "low", "high")):
            txt(f"REPORT D13 {lab} D5 {kk}", fnum(m.group(1 + k)), a[kk], 3); txt(f"REPORT D13 {lab} D7 {kk}", fnum(m.group(5 + k)), b[kk], 3)
        TXT_N[0] += 1
        if m.group(9) != stt:
            TXT_DIFFS.append((f"REPORT D13 {lab} statement", m.group(9), stt, 0))
    # sentences: every interval in section 5.5
    s55 = rep.split("### 5.5")[1].split("### 5.6")[0]
    four, three = s55.split("**Over worlds 1 to 3**")[0], s55.split("**Over worlds 1 to 3**")[1].split("**Where the two")[0]
    for lab, part, ik in (("1 to 4", four, "i4"), ("1 to 3", three, "i3")):
        ivs = re.findall(IVRE, part)
        mine = [CELL["R", "early"][ik], CELL["R", "256"][ik], CELL["R", "1024"][ik], E["R_change"][1 if lab == "1 to 4" else 2],
                CELL["RB", "1024"][ik], CELL["R", "1024"][ik], CELL["B", "1024"][ik]]
        TXT_N[0] += 1
        if len(ivs) != len(mine):
            TXT_DIFFS.append((f"REPORT sentences {lab}: intervals found", len(ivs), len(mine), 0))
        for k, (g_, mi) in enumerate(zip(ivs, mine)):
            for j, kk in enumerate(("mean", "low", "high")):
                txt(f"REPORT sentences {lab} interval {k + 1} {kk}", fnum(g_[j]), mi[kk], 3)
        w1, kinds, chw, over, smaller, asg = E[f"sent_{lab}"]
        found = [x if x != "not distinguished from" else "not distinguished" for x in re.findall(r"\*\*(lower than|higher than|not distinguished from)\*\*", part)]
        found = [x.split()[0] if x.endswith("than") else x for x in found]
        TXT_N[0] += 1
        if found != w1:
            TXT_DIFFS.append((f"REPORT sentence 1 {lab} words", found, w1, 0))
        n_near = sum(1 for k in kinds.values() if k == "nearer to zero")
        m = re.search(r"In (\d) of the \w+ worlds .*? nearer to zero .*?; in (\d) it changed sign, and in (\d) it is not nearer", part)
        TXT_N[0] += 3
        if not m or (int(m.group(1)), int(m.group(2)), int(m.group(3))) != (n_near, sum(k == "changed sign" for k in kinds.values()),
                                                                             sum(k == "not nearer to zero" for k in kinds.values())):
            TXT_DIFFS.append((f"REPORT sentence 2 {lab} counts", m.groups() if m else None, n_near, 0))
        TXT_N[0] += 1
        if f"is **{chw}**" not in part:
            TXT_DIFFS.append((f"REPORT sentence 2 {lab} change word", None, chw, 0))
        TXT_N[0] += 1
        said = "**RB is assigned to the rule**" in part and "rule" or "**RB is assigned to the birth**" in part and "birth" or "**Nothing is assigned" in part and "nothing"
        if said != asg:
            TXT_DIFFS.append((f"REPORT sentence 3 {lab} assignment", said, asg, 0))
        TXT_N[0] += 1
        if over != False or "**Overtaking is not said" not in part:      # noqa: E712
            TXT_DIFFS.append((f"REPORT overtaking {lab}", "not said", over, 0))
    # predictions table
    for row in table_rows(rep, "| | Statement |"):
        pid = row[0]
        for col, lab in ((3, "1 to 3"), (5, "1 to 4")):
            if pid == "P60.6":
                continue
            v = E["P"][lab][pid]
            yes = (v == "lower") if pid == "P60.2" else bool(v)
            said = row[col].startswith("**Yes**")
            TXT_N[0] += 1
            if said != yes:
                TXT_DIFFS.append((f"REPORT prediction {pid} {lab}", row[col][:12], yes, 0))
            m = re.search(IVRE, row[col])
            if m:
                mine = {"P60.1": CELL["R", "early"], "P60.2": CELL["R", "1024"], "P60.4": CELL["B", "1024"]}[pid]["i3" if lab == "1 to 3" else "i4"]
                for j, kk in enumerate(("mean", "low", "high")):
                    txt(f"REPORT prediction {pid} {lab} {kk}", fnum(m.group(1 + j)), mine[kk], 3)
    # prose figures
    def rng(vals):
        return min(vals), max(vals)
    k33 = lambda arm, key: [R[arm, w, 1024]["kept"]["33 or more"][key] for w in WORLDS]          # noqa: E731
    d9m = lambda arm: [R[arm, w, 1024]["d9"]["33 or more"]["mean"] for w in WORLDS]               # noqa: E731
    pros = [("5.6 D8 san", (0.007, 0.012), rng(k33("sweep-after-naive", "share"))), ("5.6 D8 naive", (0.005, 0.015), rng(k33("naive", "share"))),
            ("5.6 D9 san", (0.013, 0.020), rng(d9m("sweep-after-naive"))), ("5.6 D9 naive", (0.008, 0.012), rng(d9m("naive"))),
            ("5.6 D10 san", (0.071, 0.093), rng(k33("sweep-after-naive", "first16"))), ("5.6 D10 naive", (0.093, 0.115), rng(k33("naive", "first16")))]
    for lab, rep_, mine in pros:
        txt(f"REPORT {lab} low", rep_[0], mine[0], 3); txt(f"REPORT {lab} high", rep_[1], mine[1], 3)
    pool = pool_d15("sweep-after-naive", 1024)["33 or more"]
    for (en, _, _), sh_, nret in zip(EARLIER, (0.005, 0.013, 0.006, 0.012, 0.011), (56, 102, 170, 237, 434)):
        txt(f"REPORT 5.6 D15 {en} share", sh_, pool[en]["share"], 3); txt(f"REPORT 5.6 D15 {en} returns", nret, pool[en]["visits"], 0)
    print("   D15 pooled (san, 33+): " + ", ".join(f"{en}: {pool[en]['share']:.4f} ({pool[en]['visits']} returns)" for en, _, _ in EARLIER))
    print("   D8 33+ share san " + str([round(x, 4) for x in k33("sweep-after-naive", "share")]) + " naive " + str([round(x, 4) for x in k33("naive", "share")])
          + " sweep " + str([round(x, 4) for x in k33("sweep", "share")]))
    # world 4 D5 and the curve
    st4 = R["sweep-after-naive", 4, 1024]["score_trend"]
    txt("REPORT 5.4 world 4 san D5", 0.460, st4, 3)
    bl4 = [float(np.mean(R["sweep-after-naive", 4, 1024]["score_by_age"][64 * b:64 * b + 64])) for b in range(16)]
    txt("REPORT 5.4 w4 block 0", 0.648, bl4[0], 3); txt("REPORT 5.4 w4 block 9", 0.421, bl4[9], 3); txt("REPORT 5.4 w4 block 15", 0.675, bl4[15], 3)
    others = [R["sweep-after-naive", w, 1024]["score_trend"] for w in W3]
    txt("REPORT 5.4 other worlds D5 low", 0.019, min(others), 3); txt("REPORT 5.4 other worlds D5 high", 0.068, max(others), 3)
    print(f"   san D5 at 1,024 by world: {[round(R['sweep-after-naive', w, 1024]['score_trend'], 4) for w in WORLDS]}; naive "
          f"{[round(R['naive', w, 1024]['score_trend'], 4) for w in WORLDS]}; w4 blocks {[round(x, 3) for x in bl4]}; lowest block {int(np.argmin(bl4))}")
    # w4 D5 visits used, and D5 with world 4's returns of the second half only after the dip
    sba = R["sweep-after-naive", 4, 1024]["score_by_age"]; kind = R["sweep-after-naive", 4, 1024]["kind"]
    print(f"   w4 second half: returns used by D5 {sum(1 for v in range(512, 1024) if kind[v] == 'return')}; mean D2 of returns 512-639 "
          f"{np.mean([sba[v] for v in range(512, 640) if kind[v] == 'return']):.4f}, 896-1023 {np.mean([sba[v] for v in range(896, 1024) if kind[v] == 'return']):.4f}")
    # section 2, 3, 7 figures
    for w, x in zip(WORLDS, (3.411, 3.412, 3.411, 3.359)):
        txt(f"REPORT 2 cpu w{w}", x, E[f"cpu_{w}"], 3)
    txt("REPORT 2 cpu total", 13.59, E["cpu_tot"], 2)
    ek = [R["sweep-after-naive", w, 1024]["early_kinds"]["continuation"] for w in WORLDS]
    lk = [R["sweep-after-naive", w, 1024]["late_visits_by_kind"]["return"] for w in WORLDS]
    txt("REPORT 7 continuations low", 23, min(ek), 0); txt("REPORT 7 continuations high", 47, max(ek), 0)
    txt("REPORT 7 returns low", 218, min(lk), 0); txt("REPORT 7 returns high", 223, max(lk), 0)
    print(f"   continuations among visits 0-63: {ek}; returns in the late window of 1,024: {lk}")
    E["c4_right_reported"] = (1893, 1885, 1860, 1893)
    print(f"   report text (other than tables): compared {TXT_N[0] - n0}, differing {len(TXT_DIFFS) - d0}")


# ================================================================ F. checks
def part_f():
    print("F. The six checks")
    CK = gjson(SAN + "checks/E60_CHECKS.json"); c = CK["checks"]
    d = SAN + "checks/"
    # 1
    swck = gjson(SW + "checks/E44_L2_SWEEP_CHECKS.json", PREV)
    rec_ok = c["1"]["sources_sha256_recorded"] == swck["sources_sha256"]
    now_ok = all(sha(gb(M + k, PREV)) == v for k, v in c["1"]["sources_sha256_now"].items())
    unch_ok = all(sha(gb(M + k, HEAD)) == v for k, v in c["1"]["unchanged_files_now"].items())
    san_files = [p for p in git("ls-tree", "-r", "--name-only", HEAD, "--", M + "smallworld/").decode().split() if "/l2_san" in p]
    listed = c["1"]["new_scripts_sha256"]
    print(f"   check 1: recorded sources = sweep check receipt {rec_ok}; sources_now hold at {PREV} {now_ok}; unchanged files hold at {HEAD} {unch_ok}; "
          f"l2_san* scripts committed {[p.split('/')[-1] for p in san_files]}; listed in the receipt {list(listed)}")
    for k, v in listed.items():
        print(f"      {k}: receipt {v[:12]}..., at {HEAD} {sha(gb(M + k))[:12]}..., equal {sha(gb(M + k)) == v}")
    missing = [p.split("/")[-1] for p in san_files if p.replace(M, "") not in listed]
    print(f"      scripts not in the receipt: {missing}")
    E["c1"] = (rec_ok and now_ok and unch_ok, missing)
    # 2
    ok2 = True
    for w in WORLDS:
        r = gjson(d + f"c2_w{w}/lw_sweep-after-naive_w{w}_v1.json")
        nv = REC["naive", w]["settings"]["learner"]; ln = r["settings"]["learner"]
        eq = ln == nv
        ok2 &= eq and r["settings"]["arm"] == "sweep-after-naive" and r["complete"] and r["settings"]["visits"] == 1
        print(f"   check 2 w{w}: one-visit life complete {r['complete']}, arm {r['settings']['arm']}, learner equal to naive's (whole object) {eq}; "
              f"list_every {r['settings']['list_every']}, decay {r['settings']['decay_distances']}, lr {r['settings']['lr']}, out under checks/c2_w{w}")
    ref = gb(d + "c2_refused.txt").decode()
    print(f"   check 2 refusal: text holds the runner's assertion {'not S' in ref and 'AssertionError' in ref}; exit line {ref.strip().splitlines()[-1]!r}")
    E["c2"] = ok2 and "AssertionError" in ref
    # 3
    r3 = gjson(d + "c3_calib/lw_sweep-after-naive_w1_v8_turn.json")
    ev3 = gnpz(d + "c3_calib/lw_sweep-after-naive_w1_v8_turn.npz"); la3 = gnpz(d + "c3_calib/lw_sweep-after-naive_w1_v8_turn_list.npz")
    f1 = M + "exp45/e44/F1/lw_sweep-after-naive_w1_v8_turn"
    e44 = gjson(f1 + ".json"); ee = gnpz(f1 + ".npz"); el = gnpz(f1 + "_list.npz")
    h = dict(events=(arrays_sha(ev3), arrays_sha(ee)), list=(arrays_sha(la3), arrays_sha(el)),
             weights=(r3["final"]["weights_tensors_sha256"], e44["final"]["weights_tensors_sha256"]), optimizer=(r3["final"]["optimizer_sha256"], e44["final"]["optimizer_sha256"]),
             learned=(r3["learned_sha256"], e44["learned_sha256"]), stream=(r3["stream_sha256"], e44["stream_sha256"]))
    E["c3"] = all(a == b for a, b in h.values())
    print(f"   check 3: six hashes equal to E44's F1 receipt/files: {[(k, a == b) for k, (a, b) in h.items()]}; arrays recomputed = c3 receipt "
          f"{arrays_sha(ev3) == r3['events']['arrays_sha256'] and arrays_sha(la3) == r3['list']['arrays_sha256']}; right {int(ev3['correct'].sum())} vs E44 {int(ee['correct'].sum())}; "
          f"settings: in_turn {r3['settings']['in_turn']}, list_every {r3['settings']['list_every']}, decay {r3['settings']['decay_distances']}, save_states {r3['settings']['save_states']}, "
          f"learner {r3['settings']['learner']['file'].split('/')[-1]}; F1 receipt at {HEAD} same blob as at {PREV}: {blob(HEAD, f1 + '.json') == blob(PREV, f1 + '.json')}")
    # 4
    ok4 = True; rights = []
    for w in WORLDS:
        la = gnpz(d + f"c2_w{w}/lw_sweep-after-naive_w{w}_v1_list.npz"); ln_ = gnpz(life_path("naive", w) + "_list.npz", W1)
        r = gjson(d + f"c2_w{w}/lw_sweep-after-naive_w{w}_v1.json")
        m1 = (la["at_visit"] == 65535) & (la["source"] == 0); m2 = (ln_["at_visit"] == 65535) & (ln_["source"] == 0)
        eq = int(m1.sum()) == int(m2.sum()) == 2000 and all(np.array_equal(la[k][m1], ln_[k][m2]) for k in ("task", "a", "b", "g1", "g2", "g3", "correct"))
        s0 = r["by_visit"][0]["stream_sha256"] == REC["naive", w]["by_visit"][0]["stream_sha256"]
        rights.append(int(la["correct"][m1].sum()))
        ok4 &= eq and s0
        print(f"   check 4 w{w}: 2,000 rows equal in 7 columns {eq}; right {int(la['correct'][m1].sum())}; visit-0 stream digest equal {s0}")
    for w, x in zip(WORLDS, E["c4_right_reported"]):
        txt(f"REPORT 3 check 4 right w{w}", x, rights[w - 1], 0)
    E["c4"] = ok4
    # 5
    rw = gjson(d + "c5_whole/lw_sweep-after-naive_w1_v8.json"); rc = gjson(d + "c5_cut/lw_sweep-after-naive_w1_v8.json")
    bw = {k: gb(d + f"c5_whole/lw_sweep-after-naive_w1_v8{k}") for k in (".npz", "_list.npz")}
    bc = {k: gb(d + f"c5_cut/lw_sweep-after-naive_w1_v8{k}") for k in (".npz", "_list.npz")}
    same = dict(weights=rw["final"]["weights_tensors_sha256"] == rc["final"]["weights_tensors_sha256"], optimizer=rw["final"]["optimizer_sha256"] == rc["final"]["optimizer_sha256"],
                events_file=sha(bw[".npz"]) == sha(bc[".npz"]) == rw["events"]["file_sha256"] == rc["events"]["file_sha256"],
                list_file=sha(bw["_list.npz"]) == sha(bc["_list.npz"]) == rw["list"]["file_sha256"] == rc["list"]["file_sha256"],
                events_arrays=rw["events"]["arrays_sha256"] == rc["events"]["arrays_sha256"], list_arrays=rw["list"]["arrays_sha256"] == rc["list"]["arrays_sha256"],
                learned=rw["learned_sha256"] == rc["learned_sha256"])
    st_ = rw["settings"]
    sets = (st_["list_every"], st_["decay_distances"], st_["save_states"], rc["settings"]["save_states"])
    res = rc["resumes"]
    cut = gb(d + "c5_cut.log").decode().splitlines(); rsm = gb(d + "c5_resumed.log").decode().splitlines()
    last_cut = max(int(re.search(r" visit (\d+) ", l_).group(1)) for l_ in cut if " visit " in l_)
    pairs = {}
    for l_ in cut + rsm:
        m = re.search(r" visit (\d+) .* loss ([0-9.]+) right ([0-9.]+)", l_)
        if m:
            pairs.setdefault(int(m.group(1)), set()).add((m.group(2), m.group(3)))
    twice = sorted(v for v in pairs if sum(1 for l_ in cut + rsm if re.search(rf" visit {v} ", l_)) > 1)
    agree = all(len(pairs[v]) == 1 for v in twice)
    E["c5"] = all(same.values()) and len(res) == 1 and res[0]["done_visits"] == 4 and res[0]["path"].endswith("c5_cut/lw_sweep-after-naive_w1_v8_state_v4.pt") and last_cut == 5
    print(f"   check 5: {same}; settings (list_every, decay, save_states whole, save_states cut) {sets}; resumes {[(x['done_visits'], x['path'].split('/')[-1], x['file_sha256'][:12]) for x in res]}; "
          f"last visit in the cut log {last_cut}; visits computed twice {twice}, same loss and share {agree}; resume from the state recorded by the cut run's own receipt states? "
          f"{[x for x in rc['states'] if x['visits'] == 4][0]['file_sha256'] == res[0]['file_sha256'] if any(x['visits'] == 4 for x in rc['states']) else 'no state 4 in states'}")
    # 6
    c6 = c["6"]
    pv = c6["cpu_seconds_per_visit"]
    src6 = [nm for nm, rr in (("c5_whole", rw), ("c3_calib", r3)) if [round(x, 6) for x in rr["cpu"]["per_visit"]] == [round(x, 6) for x in pv]]
    sw_cpu = [REC["sweep", w]["cpu"]["total"] / 3600 for w in WORLDS]
    fc = sum(sw_cpu) * np.mean(pv) / c6["sweep_mean_per_visit_E44_L2_sweep_check_6"]
    print(f"   check 6: per-visit CPU seconds taken from {src6 or 'no committed receipt found'}; mean {np.mean(pv):.4f}; sweep lives' CPU-hours from their receipts {[round(x, 4) for x in sw_cpu]} "
          f"(check receipt {[round(x, 4) for x in c6['sweep_lives_cpu_hours']]}); forecast recomputed {fc:.4f} (receipt {c6['forecast_four_lives_cpu_hours']:.4f}); measured {E['cpu_tot']:.4f}")
    cmp("check 6 forecast", fc, c6["forecast_four_lives_cpu_hours"], 1e-6)
    txt("REPORT 3 check 6 cpu per visit", 12.06, np.mean(pv), 2); txt("REPORT 3 check 6 forecast", 13.85, fc, 2)
    print(f"   checks 1-5 pass by recomputation: 1 {E['c1'][0]} (receipt omits {E['c1'][1]}), 2 {E['c2']}, 3 {E['c3']}, 4 {E['c4']}, 5 {E['c5']}")


# ================================================================ G. states
def part_g():
    print(f"G. Saved states. Free disk {free_gb():.1f} GB. Committed at {HEAD}: "
          f"{[p for p in git('ls-tree', '-r', '--name-only', HEAD, '--', SAN).decode().split() if p.endswith('.pt')] or 'none'}; reading the history")
    seen = {}
    for c_ in git("rev-list", f"{PREV}..{HEAD}").decode().split():
        for ln_ in git("ls-tree", "-r", "-l", c_, "--", SAN).decode().splitlines():
            meta, p = ln_.split("\t")
            if p.endswith(".pt") and p not in seen:
                seen[p] = (c_, int(meta.split()[3]))
    n_ok = 0
    for p, (c_, size) in sorted(seen.items()):
        if size > 200e6:
            print(f"   {p.split('/')[-1]}: {size / 1e6:.0f} MB, skipped"); continue
        b = gb(p, c_)
        m = re.search(r"_w(\d)_v1024_state_v(\d+)\.pt", p); w, v = int(m.group(1)), int(m.group(2))
        rec = [x for x in REC["sweep-after-naive", w]["states"] if x["visits"] == v][0]
        fe = sha(b) == rec["file_sha256"]
        try:
            Wt, s = C.state_weights(b); we = C.tensors_sha(Wt) == rec["weights_tensors_sha256"]
        except Exception as e:                                   # noqa: BLE001
            we = f"unreadable {e}"
        cmp(f"state w{w} v{v} file sha", fe, True); cmp(f"state w{w} v{v} weights tensors sha", we, True)
        n_ok += fe and we is True
        print(f"   w{w} state after {v} (commit {c_[:8]}, {size / 1e6:.1f} MB): file sha256 = receipt {fe}; weights tensors recomputed = receipt {we}")
        del b
    E["states"] = (len(seen), n_ok)
    print(f"   states in history {len(seen)}, matching {n_ok}; v1024 states never committed: {not any('state_v1024' in p for p in seen)}")


# ================================================================ H. form
def part_h():
    print("H. Form")
    rep = E["report"]
    files = [p for p in git("ls-tree", "-r", "--name-only", HEAD, "--", SAN).decode().split() if not p.endswith((".npz", ".pt"))]
    files += [p for p in git("ls-tree", "-r", "--name-only", HEAD, "--", M + "smallworld/").decode().split() if "/l2_san" in p]
    texts = {p: gb(p).decode("utf-8", "replace") for p in files}
    for mark in ("@@", "TODO", "TBD", "XXX", "<w>", "{", "[..]"):
        hits = [p.split("/")[-1] for p, t in texts.items() if mark in t and p.endswith(".md")]
        print(f"   '{mark}' in .md files: {hits}")
    # model names, outside branch names
    terms = ("claude", "opus", "sonnet", "haiku", "gpt", "codex", "gemini", "anthropic", "openai")
    for p, t in texts.items():
        low = t.lower()
        for term in terms:
            for m in re.finditer(term, low):
                nxt = low[m.end():m.end() + 1]
                ctx = low[max(0, m.start() - 15):m.end() + 15].replace("\n", " ")
                if term == "claude" and nxt == "/":
                    continue
                if term == "codex" and ("codex/minifly" in ctx):
                    continue
                print(f"   name '{term}' in {p.split('/')[-1]}: ...{ctx}...")
    # excluded words for a difference
    for term in ("level", "the same", "behind", "ahead", "grown", "shrunk", "not changed", "near zero", "about the size"):
        n = len(re.findall(rf"\b{re.escape(term)}\b", rep))
        print(f"   report uses '{term}' {n} times")
    for m in re.finditer(r"[^\n]*(near zero|about the size|keeps as little|does not rise)[^\n]*", rep):
        print(f"   sentence: {m.group(0)[:220]}")
    # rate limit and shared-births limit by section
    secs = re.split(r"\n(?=#{2,3} )", rep)
    for s in secs:
        head = s.splitlines()[0][:50]
        four = len(re.findall(r"worlds 1 to 4|four worlds|four-world", s))
        print(f"   section '{head}': mentions of the four-world reading {four}; shared-births limit stated {'share their births' in s or 'same births' in s or 'shared births' in s.lower() or 'births of seed 73129 twice' in s}; "
              f"rate 0.001 stated {'0.001' in s}")
    # committed-file list
    lst = re.findall(r"^(MOD97_\S+)$", rep, re.M)
    at_rep = [p for p in git("ls-tree", "-r", "--name-only", REPC, "--", SAN, M + "smallworld").decode().split() if p.startswith(SAN) or "/l2_san" in p]
    at_head = [p for p in git("ls-tree", "-r", "--name-only", HEAD, "--", SAN, M + "smallworld").decode().split() if p.startswith(SAN) or "/l2_san" in p]
    print(f"   committed-file list in the report: {len(lst)} paths; equal to ls-tree at {REPC}: {lst == at_rep}; at {HEAD}: {lst == at_head}")
    # the brief's limit 1 figures
    print(f"   section 7 gives the RB figures at 1,024 (−0.069 [−0.162, +0.024]; −0.066 [−0.120, −0.013]) as the brief's limit 1 words it: "
          f"{'−0.069' in rep.split('## 7.')[1] and '−0.120' in rep.split('## 7.')[1]}")


if __name__ == "__main__":
    print(f"free disk {free_gb():.1f} GB; numpy {np.__version__}; python {sys.version.split()[0]}; repo {ROOT}")
    print(f"HEAD {git('rev-parse', HEAD).decode().strip()}; PREV {git('rev-parse', PREV).decode().strip()}; W1 {git('rev-parse', W1).decode().strip()}")
    for nm, fn in (("A", part_a), ("B", part_b), ("C", part_c), ("D", part_d_tables), ("E", part_e), ("F", part_f), ("G", part_g), ("H", part_h)):
        t = time.time(); fn(); print(f"   part {nm} done in {time.time() - t:.0f} s")
    print("DIFFERENCES among committed-file figures:")
    for d_ in DIFFS:
        print("   ", d_)
    print("DIFFERENCES among tables/report figures (label, reported, recomputed, difference):")
    for d_ in TXT_DIFFS:
        print("   ", d_)
    print(f"SUMMARY: committed-file figures compared {N_CMP[0]:,}, differing {len(DIFFS)}; tables/report figures compared {TXT_N[0]}, differing {len(TXT_DIFFS)}; "
          f"free disk {free_gb():.1f} GB; running time {time.time() - T0:.0f} s")
