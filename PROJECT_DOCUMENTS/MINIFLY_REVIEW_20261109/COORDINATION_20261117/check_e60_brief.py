"""Check of the brief of E60, revision 1: every figure that the check document uses.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/check_e60_brief.py

Files of other branches are read only with `git show <ref>:<path>` and listed only with `git ls-tree -r --name-only`. No file of another
branch is executed or imported; the runner, the scorer and the worker's scripts are read as text. The script writes no file.
Standard library and numpy only.
Branches are found by the end of their names, so that no product name is written in this file.
"""
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

T0 = time.time()
M = "MOD97_LOCAL_RULE_20261106"
R = "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109"
BRIEF = R + "/COORDINATION_20261117/TASK_E60_THE_SWEEPS_LONG_LIFE_FROM_BACK_PROPAGATIONS_BIRTH_20261118.md"
L2 = M + "/exp45/e44/L2"
F1 = M + "/exp45/e44/F1"
T2, T3, T7 = 2.9200, 2.3534, 1.8946          # 0.95 quantiles of Student's t (as l2_score.T95): two-sided 90% intervals


def git(*a, binary=False):
    r = subprocess.run(["git", *a], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode()


def show(ref, path, binary=False):
    return git("show", f"{ref}:{path}", binary=binary)


def js(ref, path):
    return json.loads(show(ref, path))


def npz(ref, path):
    f = np.load(io.BytesIO(show(ref, path, binary=True)))
    return {k: f[k] for k in f.files}


def names(ref, d=""):
    return git("ls-tree", "-r", "--name-only", ref, *(["--", d] if d else [])).split("\n")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def iv(xs, t=None, ddof=1):
    xs = [float(x) for x in xs]
    n = len(xs); m = float(np.mean(xs)); sd = float(np.std(xs, ddof=ddof))
    t = {2: T2, 3: T3, 7: T7}[n - 1] if t is None else t
    h = t * sd / math.sqrt(n)
    return m, m - h, m + h, sd


def word(lo, hi):
    return "higher" if lo > 0 else "lower" if hi < 0 else "not distinguished"


def head(s):
    print("\n== " + s)


refs = git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin").split()
pick = lambda suf: [r for r in refs if r.endswith("/" + suf)][0]
S = pick("e44-l2-sweep-long-lives")          # the worker's branch (the helper of the long lives)
W1 = pick("e44-l2-first-long-lives")         # the main worker's branch
D = pick("minifly-consolidated-20261023")    # the default branch
V = pick("vibrant-cerf-sje726")              # the reviewer's branch (checked out)

head("0. What was read")
print("HEAD", git("rev-parse", "HEAD").strip(), "| reviewer's branch", git("rev-parse", V).strip()[:8])
print("worker's branch head", git("rev-parse", S).strip(), "| main worker's", git("rev-parse", W1).strip()[:8],
      "| default", git("rev-parse", D).strip()[:8])
bb = show("HEAD", BRIEF, binary=True)
print("brief blob at HEAD", git("rev-parse", "HEAD:" + BRIEF).strip()[:12], "| at 1a1a3b4b", git("rev-parse", "1a1a3b4b:" + BRIEF).strip()[:12],
      "| lines", bb.decode().count("\n"))
brief = bb.decode()
BL = brief.splitlines()

# ------------------------------------------------------------------ section 3, the verified lives
head("1. Section 3: the verified lives of worlds 1 to 3 (final/E44_L2_SWEEP_EARLY.json on the worker's branch)")
E = js(S, L2 + "/sweep/final/E44_L2_SWEEP_EARLY.json")["pairs"]["sweep against naive"]
early, late = {}, {}
for V_ in ("256", "1024"):
    for w in ("1", "2", "3", "4"):
        p = E[V_]["per_world"][w]
        early[w] = (p["early_y"], p["early_x"], p["early_difference"])
        late[(V_, w)] = (p["late_y"], p["late_x"], p["late_difference"])
for w in ("1", "2", "3"):
    print(f"world {w}: early naive {early[w][0]:.3f} sweep {early[w][1]:.3f} diff {early[w][2]:+.4f} | late256 {late[('256', w)][0]:.3f} "
          f"{late[('256', w)][1]:.3f} {late[('256', w)][2]:+.4f} | late1024 {late[('1024', w)][0]:.3f} {late[('1024', w)][1]:.3f} {late[('1024', w)][2]:+.4f}")
for lab, xs in (("early", [early[w][2] for w in "123"]), ("late256", [late[("256", w)][2] for w in "123"]),
                ("late1024", [late[("1024", w)][2] for w in "123"])):
    m, lo, hi, sd = iv(xs)
    r4 = [round(x, 4) for x in xs]; m4, lo4, hi4, _ = iv(r4)
    mp, lop, hip, _ = iv(xs, ddof=0)
    print(f"3 worlds {lab}: unrounded {m:+.5f} [{lo:+.5f}, {hi:+.5f}] sd {sd:.5f} {word(lo, hi)} | from 4-decimal values {m4:+.4f} [{lo4:+.4f}, {hi4:+.4f}]"
          f" | with the sd of divisor n (not n-1): [{lop:+.4f}, {hip:+.4f}] {word(lop, hip)}")
xs4 = [late[("1024", w)][2] for w in "1234"]
m, lo, hi, _ = iv(xs4)
print(f"4 worlds late1024: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] {word(lo, hi)}; all four below zero: {all(x < 0 for x in xs4)}")
print(f"t(0.95,2)/sqrt(3) = {T2 / math.sqrt(3):.4f}; t(0.95,3)/sqrt(4) = {T3 / 2:.4f}")
print("RB per world, sign of early and late difference at 1,024 visits; |late| < |early| ('nearer to zero'):")
for w in "123":
    e, l = early[w][2], late[("1024", w)][2]
    print(f"   world {w}: early {e:+.4f} late {l:+.4f} sign change {np.sign(e) != np.sign(l)} nearer to zero {abs(l) < abs(e)} change {l - e:+.4f}")
ch = [late[("1024", w)][2] - early[w][2] for w in "123"]
m, lo, hi, _ = iv(ch)
print(f"RB change (late1024 - early), 3 worlds: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] {word(lo, hi)}")

# kinds of visit in the two windows, worlds 1 to 3 (score files of naive, V = 1,024)
print("kinds of visit (scorer's `kind`), visits 0-63 and 768-1023, worlds 1-3:")
for w in "123":
    q = js(S, L2 + f"/sweep/main/lw_naive_w{w}_v1024_V1024.json")
    k = q["kind"]
    c = lambda a, b: {x: k[a:b].count(x) for x in ("first", "continuation", "return")}
    print(f"   world {w}: early {c(0, 64)} late {c(768, 1024)}")

# D13 and D5 over three worlds for the verified arms, as the worker's `stuck` would form them with three values
head("2. D5, D7, D13 of the verified arms over worlds 1 to 3 (score files on the worker's branch)")
Q = {}
for arm, sub in (("naive", "main"), ("sweep", "scores")):
    for w in "123":
        for V_ in (256, 1024):
            Q[(arm, w, V_)] = js(S, L2 + f"/sweep/{sub}/lw_{arm}_w{w}_v1024_V{V_}.json")
for arm in ("naive", "sweep"):
    for V_ in (256, 1024):
        t5 = iv([Q[(arm, w, V_)]["score_trend"] for w in "123"]); t7 = iv([Q[(arm, w, V_)]["trend_of_learning"] for w in "123"])
        st = "still learning" if (t5[1] > 0 or t7[1] > 0) else "stuck"
        print(f"{arm} V={V_}: D5 {t5[0]:+.4f} [{t5[1]:+.4f}, {t5[2]:+.4f}]  D7 {t7[0]:+.4f} [{t7[1]:+.4f}, {t7[2]:+.4f}]  D13 over 3 worlds: {st}")
for V_ in (256, 1024):
    d5 = iv([Q[("sweep", w, V_)]["score_trend"] - Q[("naive", w, V_)]["score_trend"] for w in "123"])
    print(f"RB difference of D5 at {V_}: {d5[0]:+.4f} [{d5[1]:+.4f}, {d5[2]:+.4f}] {word(d5[1], d5[2])}")
for w in "123":
    e = float(np.mean(Q[("naive", w, 1024)]["score_by_age"][:64]))
    print(f"   check: early level of naive, world {w}, from the score file: {e:.6f} (EARLY.json {early[w][0]:.6f}); late256 {Q[('naive', w, 256)]['late_level']:.6f}")
print("score files hold no None in score_by_age for visits 0-63:",
      all(x is not None for a in ("naive", "sweep") for w in "123" for x in Q[(a, w, 1024)]["score_by_age"][:64]))

# ------------------------------------------------------------------ section 3, the calibration
head("3. Section 3: the calibration (F1 event files at 256 updates a visit, default branch)")
cal, eighths_fig, eighths_pool = {}, {}, {}
for arm in ("naive", "sweep", "sweep-after-naive"):
    cal[arm] = []
    fig8, pool_r, pool_n = np.zeros(8), np.zeros(8), np.zeros(8)
    nf = 0
    for w in range(1, 9):
        stem = f"lw_{arm}_w{w}_v8_turn"
        rec = js(D, f"{F1}/{stem}.json")
        raw = show(D, f"{F1}/{stem}.npz", binary=True)
        assert sha(raw) == rec["events"]["file_sha256"] and rec["complete"]
        assert rec["settings"]["visit_updates"] == 256 and rec["settings"]["in_turn"] and rec["learner"]["seed"] == 73129
        assert rec["learner"]["rule"] == ("S" if arm == "sweep" else "B")
        f = np.load(io.BytesIO(raw)); ev = {k: f[k] for k in f.files}
        vis, upd, task, cor, wrap = (ev[k].astype(np.int64) for k in ("visit", "update", "task", "correct", "wrap"))
        op = task % 4; red = (wrap == 1) & (op < 2)
        i = upd - vis * 256                                     # update within the visit
        figs = []
        for v in range(1, 8):
            assert (task[vis == v] // 4 == v).all()             # in turn: visit v is family v
            m = (vis == v) & red
            last = m & (i >= 192)
            figs.append(np.mean([cor[last & (op == j)].mean() for j in (0, 1)]))
            for e in range(8):
                me = m & (i >= 32 * e) & (i < 32 * e + 32)
                fig8[e] += np.mean([cor[me & (op == j)].mean() for j in (0, 1)])
                pool_r[e] += cor[me].sum(); pool_n[e] += me.sum()
            nf += 1
        cal[arm].append(float(np.median(figs)))
    eighths_fig[arm] = fig8 / nf; eighths_pool[arm] = pool_r / pool_n
    g = cal[arm]
    print(f"{arm}: gate's figure w1-8 {' '.join(f'{x:.3f}' for x in g)} | mean {np.mean(g):.4f} | min {min(g):.4f} max {max(g):.4f} | "
          f"median over worlds {np.median(g):.4f} | inside 0.20-0.90: {sum(0.2 <= x <= 0.9 for x in g)}")
    print(f"   eighths, mean of the 56 family figures: {' '.join(f'{x:.3f}' for x in eighths_fig[arm])}")
    print(f"   eighths, pooled share right over all reduction items: {' '.join(f'{x:.3f}' for x in eighths_pool[arm])}")
for a, b in (("sweep-after-naive", "naive"), ("sweep", "sweep-after-naive"), ("sweep", "naive")):
    d = [x - y for x, y in zip(cal[a], cal[b])]
    m, lo, hi, _ = iv(d)
    print(f"{a} - {b}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] above zero in {sum(x > 0 for x in d)} of 8, below in {sum(x < 0 for x in d)}")

# ------------------------------------------------------------------ what exists
head("4. What exists: files of the arm sweep-after-naive, and of a reverse arm, on every fetched ref")
hits, rev = {}, {}
for r in refs:
    try:
        ns = names(r)
    except subprocess.CalledProcessError:
        continue
    h = [n for n in ns if "after-naive" in n or "after_naive" in n]
    out = [n for n in h if "/exp45/e44/F1/" not in n]
    if out:
        hits[r.split("/", 1)[1]] = out[:3]
    rv = [n for n in ns if re.search(r"after[-_]sweep|naive[-_]after|backprop[-_]after", n)]
    if rv:
        rev[r.split("/", 1)[1][-30:]] = rv[:3]
f1san = [n for n in names(D, F1) if "sweep-after-naive" in n]
print("refs searched", len(refs), "| sweep-after-naive files outside F1:", hits or "none", "| reverse-arm names:", rev or "none")
print("F1 sweep-after-naive receipts (.json, not logs):", sum(1 for n in f1san if n.endswith(".json")),
      "| visits in their names:", sorted({re.search(r"_v(\d+)", n).group(1) for n in f1san if n.endswith(".json")}))
sanl = names(S, L2)
print("worker's branch: any path under L2/sweep_after_naive:", any("/sweep_after_naive/" in n for n in sanl),
      "| any smallworld/l2_san*:", any("/l2_san" in n for n in names(S, M + "/smallworld")))
print("F1 receipt of check 3 on the worker's branch:", (F1 + "/lw_sweep-after-naive_w1_v8_turn.json") in names(S, F1))
for seed in (73129, 93683, 116089):
    for rule in "BS":
        p = f"{M}/exp45/e41/e41_{rule}_h256_f50_{seed}"
        print(f"   birth {rule} {seed}: .pt and .json on the worker's branch:", (p + ".pt") in names(S, M + "/exp45/e41") and (p + ".json") in names(S, M + "/exp45/e41"))

# ------------------------------------------------------------------ the runner, read as text
head("5. The runner (smallworld/sw_run.py on the worker's branch), read as text")
run = show(S, M + "/smallworld/sw_run.py")
RL = run.splitlines()


def where(pat):
    return [i + 1 for i, l in enumerate(RL) if re.search(pat, l)]


print("sha256", sha(run.encode()))
for lab, pat in (("ARMS (sweep-after-naive: local, no replay)", r'"sweep-after-naive": \(True, False\)'),
                 ("load_learner: rule wanted per arm", r'want = \{"naive": "B", "sweep": "S", "sweep-after-naive": "B"'),
                 ("load_learner: refusal of another rule", r'assert rec\["rule"\] == want'),
                 ("load_learner: no check of the seed against the world", r'rec\["seed"\]'),
                 ("fresh optimizer at the start of a life", r'self\.opt = N\.make_opt\(self\.m, lr, wd\)'),
                 ("TAGS (no list_every, decay_distances: they do not enter the file name)", r'^TAGS = '),
                 ("life_stem", r'return f"\{pre\}_\{cfg\[.arm.\]\}_w'),
                 ("--out default exp45/e39", r'^OUT = MDIR / "exp45" / "e39"'),
                 ("--out argument", r'ap\.add_argument\("--out", default=str\(OUT\)\)'),
                 ("--resume takes the settings from the state", r'--resume takes the life.s settings from the state'),
                 ("answer_list_before: the birth family's rows at at_visit 65535", r'self\._list_rows\(self\.m, self\.r_items\(\), BEFORE_LIFE'),
                 ("list_whole_at: whole list also after the last visit", r'return n % self\.cfg\.get\("list_every", 1\) == 0 or n == self\.cfg\["visits"\]'),
                 ("decay_visits with distances [] gives no rows", r'return list\(range\(v0 \+ 1, V\)\) if d is None else'),
                 ("wants_state", r'return \(is_pow2\(n\) or n == self\.cfg\["visits"\]\) if ss is None else n in ss'),
                 ("save_state refuses an existing path", r'refuse_existing\(path, str\(path\) \+ "\.tmp"\)'),
                 ("run_life refuses existing receipt, events, list", r'refuse_existing\(rpath, epath, lpath\)'),
                 ("Life.load: sources must equal the state's", r'the sources differ from the state'),
                 ("--decay-distances none", r'\[\] if a\.decay_distances == "none"'),
                 ("--list-every default 1", r'ap\.add_argument\("--list-every", type=int, default=1\)'),
                 ("per-visit stream digest in by_visit", r'rec = dict\(s, updates=len\(losses\), stream_sha256=hs\.hexdigest\(\)'),
                 ("receipt: learner record", r'world=wd_, learner=self\.cfg\.get\("learner"\)'),
                 ("list file written only at the end of the life", r'\*\*list_record\(life, lpath\)\)'),
                 ("log line prints the 0-based visit index", r'print\(f"\{stem\} visit \{s\[.visit.\]\}'),
                 ("load_learner: complete E41 birth at this H and U", r'assert rec\["experiment"\] == "E41" and rec\["complete"\]'),
                 ("load_learner: file hashes against the receipt", r'the birth file.s hashes differ from its receipt'),
                 ("state saved at the end of visit n (n visits done)", r'life\.save_state\(out / f"\{stem\}_state_v\{n\}\.pt"\)'),
                 ("whole list: the birth family's rows at the end of a visit", r'self\._list_rows\(self\.m, self\.r_items\(\), v, 0, BIRTH_FAMILY'),
                 ("event file written after the last visit", r'np\.savez_compressed\(epath, \*\*arrs\)'),
                 ("receipt written after the last visit", r'write_json\(rpath, rec\)')):
    print(f"   {lab}: lines {where(pat)}")
print("a one-visit life holds the birth family's 2,000 items twice (before visit 0, and after visit 0 as the last visit):", 2 * 2000, "rows")
print("lines of the docstring that mention --birth-from (E39's sweep-after-naive):", where(r"--birth-from"))

# ------------------------------------------------------------------ check 1: sources
head("6. Check 1: sources against sweep/checks/E44_L2_SWEEP_CHECKS.json")
C = js(S, L2 + "/sweep/checks/E44_L2_SWEEP_CHECKS.json")
rec_src = C["sources_sha256"]; rec_unch = C["checks"]["1"]["unchanged_files_sha256"]
smallworld = sorted(n.split("/")[-1] for n in names(S, M + "/smallworld") if n)
recorded = {k.split("/")[-1] for k in list(rec_src) + list(rec_unch) if k.startswith("smallworld/")}
print("smallworld files at the worker's head:", len(smallworld), "| recorded in the receipt:", sorted(recorded))
print("not recorded:", [f for f in smallworld if f not in recorded])
print("recorded outside smallworld/:", sorted(k for k in rec_src if not k.startswith("smallworld/")))
bad = []
for k, h in list(rec_src.items()) + list(rec_unch.items()):
    p = (M + "/" + k) if not k.startswith(M) else k
    if sha(show(S, p, binary=True)) != h:
        bad.append(k)
print("recorded hashes that differ at the worker's head:", bad or "none")

# ------------------------------------------------------------------ receipts of naive and sweep: learners, streams, settings
head("7. Receipts: naive (main worker's branch) and sweep (worker's branch), worlds 1 to 3; calibration receipt of check 3")
NR = {w: js(W1, f"{L2}/lives/lw_naive_w{w}_v1024.json") for w in "123"}
SR = {w: js(S, f"{L2}/sweep/lives/lw_sweep_w{w}_v1024.json") for w in "123"}
F1R = js(D, f"{F1}/lw_sweep-after-naive_w1_v8_turn.json")
for w in "123":
    n, s = NR[w], SR[w]
    same_stream = [a["stream_sha256"] for a in n["by_visit"]] == [b["stream_sha256"] for b in s["by_visit"]]
    keys = ("world_kind", "H", "U", "visit_updates", "p_new", "lr", "wd", "list_every", "decay_distances", "save_states", "taught_share", "lowest", "in_turn")
    print(f"world {w}: naive learner {n['learner']['file']} rule {n['learner']['rule']} seed {n['learner']['seed']} R {n['learner']['final_row']['R']} "
          f"updates {n['learner']['updates']} fraction {n['learner']['fraction']} H {n['learner']['H']} U {n['learner']['U']} "
          f"| sweep learner seed {s['learner']['seed']} rule {s['learner']['rule']} R {s['learner']['final_row']['R']}")
    print(f"   all 1,024 visit stream digests equal naive/sweep: {same_stream}; settings equal on {len(keys)} keys: "
          f"{all(n['settings'][k] == s['settings'][k] for k in keys)}; decay_distances {n['settings']['decay_distances']}; list_every {n['settings']['list_every']}")
    print(f"   naive before visit 0 (list_summary[0]): {n['list_summary'][0]['n']} items, share {n['list_summary'][0]['share']}; sweep: {s['list_summary'][0]['share']}")
    b0 = [e for e in n["list_summary"] if e["family"] == -1]
    print(f"   naive list rows of the birth family: at_visit 65535: {sum(1 for e in b0 if e['at_visit'] == 65535)} entry; at other visits: "
          f"{sum(1 for e in b0 if e['at_visit'] != 65535)} entries (first {[e['at_visit'] for e in b0 if e['at_visit'] != 65535][:3]})")
    print(f"   resumes naive {len(n['resumes'])} sweep {len(s['resumes'])}; host naive {n['provenance']['host']['python']} {n['provenance']['host']['torch']} "
          f"{n['provenance']['host']['numpy']}; initial weights hash equals learner tensors hash: "
          f"{n['model']['initial']['weights_tensors_sha256'] == n['learner']['tensors_sha256']}")
print("calibration receipt (check 3): learner record equals naive world 1's field for field:", F1R["learner"] == NR["1"]["learner"])
print("   its command:", F1R["provenance"]["command"].replace("/home/user/digitalbaby/", ""))
print("   its settings hold list_every / decay_distances:", "list_every" in F1R["settings"], "decay_distances" in F1R["settings"])
print("   hashes: events arrays", F1R["events"]["arrays_sha256"][:12], "list arrays", F1R["list"]["arrays_sha256"][:12], "final weights",
      F1R["final"]["weights_tensors_sha256"][:12], "optimizer", F1R["final"]["optimizer_sha256"][:12])
c2 = C["checks"]["2"]
print("worker's earlier reproduction of the sweep's calibration life: hashes compared", sorted(c2["this_host"]["hashes"]),
      "| differing:", c2["hashes_that_differ"])

# ------------------------------------------------------------------ cost
head("8. Cost: the worker's eight lives (receipts on the worker's branch)")
cpu = {}
for arm in ("sweep", "sweep+replay"):
    for w in "1234":
        stem = f"lw_{arm}_w{w}_v1024" + ("_rho0.5" if "replay" in arm else "")
        r = SR[w] if (arm == "sweep" and w in SR) else js(S, f"{L2}/sweep/lives/{stem}.json")
        cpu[stem] = (r["cpu"]["total"] / 3600, r["cpu"]["mean_per_visit"], len(r["resumes"]))
for k, v in cpu.items():
    print(f"   {k}: {v[0]:.3f} CPU-hours, {v[1]:.2f} s a visit, resumes {v[2]}")
tot = [v[0] for v in cpu.values()]
sw = [cpu[f"lw_sweep_w{w}_v1024"][0] for w in "1234"]
print(f"range {min(tot):.3f} to {max(tot):.3f}; sum {sum(tot):.2f}; three lives at the sweep's mean ({np.mean(sw):.3f} h): {3 * np.mean(sw):.2f} h; at the "
      f"largest: {3 * max(tot):.2f} h")
print(f"CPU-minutes lost at most by a stop between two states (64 visits at the sweep's mean seconds a visit, {np.mean([cpu[f'lw_sweep_w{w}_v1024'][1] for w in '1234']):.2f}):",
      round(64 * np.mean([cpu[f"lw_sweep_w{w}_v1024"][1] for w in "1234"]) / 60, 1))
print("calibration life of check 3 (receipt): CPU", round(F1R["cpu"]["total"], 1), "s")
c6 = C["checks"]["6"]
print("worker's check 6: life of 1,024 visits under sweep forecast", round(c6["life_of_1024_visits_cpu_hours"]["sweep"], 3), "h")
print("naive lives' CPU-hours, worlds 1-3:", [round(NR[w]["cpu"]["total"] / 3600, 3) for w in "123"])

# ------------------------------------------------------------------ the scorer and the report scripts, read as text
head("9. Scorer and report scripts (worker's branch), read as text")
sc = show(S, M + "/smallworld/l2_score.py")
rp = show(S, M + "/smallworld/l2_sweep_report.py")
ea = show(S, M + "/smallworld/l2_sweep_early.py")
print("l2_score.py sha256", sha(sc.encode())[:12], "| T95[2] =", re.search(r"T95 = \{1: [\d.]+, 2: ([\d.]+)", sc).group(1),
      "| interval90 uses ddof=1:", "np.std(xs, ddof=1)" in sc, "| t chosen by n:", "T95[len(xs) - 1]" in sc)
print("T95[1] (t used with two values):", re.search(r"T95 = \{1: ([\d.]+)", sc).group(1))
print("l2_score.py CLI: 'life <events .npz> <list .npz> <world> <visits read>':", "python3 l2_score.py life <events .npz> <list .npz> <world> <visits read>" in sc)
print("l2_sweep_report.py: WORLDS fixed to (1, 2, 3, 4):", "WORLDS = (1, 2, 3, 4)" in rp, "| LIVES = sweep/lives:", 'LIVES, SCORES, MAIN = SW / "lives", SW / "scores", SW / "main"' in rp,
      "| q_of routes any arm containing 'sweep' to sweep/scores:", '(SCORES if "sweep" in arm else MAIN)' in rp)
print("l2_sweep_report.py maps 'not distinguished' to 'level' and 'the same':", '"not distinguished": "level"' in rp and '"not distinguished": "the same"' in rp)
print("l2_sweep_early.py: words grown/shrunk/not changed:", '"higher": "grown"' in ea, "| pairs from RP.PAIRS (sweep/naive, sweep+replay/replay):", "for x, y in RP.PAIRS" in ea)
print("D12 'closed' defined in the roster note for naive and replay only; D13 uses an interval over worlds (stuck() in the report script):",
      "def stuck(arm, worlds, V)" in rp)
ros_name = [n for n in names(S) if n.endswith("_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md") and "/" not in n][0]
ros = show(S, ros_name)
d5 = [l for l in ros.splitlines() if l.startswith("- **D5.")][0]
d13 = [l for l in ros.splitlines() if l.startswith("- **D13.")][0]
print("roster D5 says 'over the returns':", "over the returns" in d5, "| 'ln(v + 1)':", "ln(v + 1)" in d5, "| all-visits slope 'not used':", "not used" in d5)
print("roster 'Stuck' uses 't over worlds':", "(t over worlds)" in ros, "| 'mean ± 1.18 standard deviations' (four worlds):", "1.18 standard deviations" in ros)
print("brief line with D5:", [i + 1 for i, l in enumerate(BL) if "D5, the trend of the score over the second half" in l])

# ------------------------------------------------------------------ the brief's text
head("10. The brief's text")
for pat in ("t(0.95, 2 degrees of freedom)", "standard deviations (t = 2.920)", "nearer to zero", "one half of a two-by-two",
            "and only that birth", "the two newer arguments", "from the state after visit 4", "during visit 6", "A message reaches you only when your turn ends",
            "Keep waits to ten minutes", "D1 to D15", "RB = R + B", "the median level at the first visit", "the pooled share right",
            "commit trailers as", "No name of any language model", "the committed files listed from `git ls-tree`", "--out", "l2_driver", "90%"):
    print(f"   {pat!r}: lines {[i + 1 for i, l in enumerate(BL) if pat in l]}")
ptr = show(S, "PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md").splitlines()
ns_word = S.split("/")[1]
words = [ns_word, D.split("/")[1], ros_name.split("_")[0]]
print("AGENTS_POINTER.md line 17 prescribes a trailer containing the worker's namespace word:", (ns_word + ":") in ptr[16])
print("worker's recent commit trailers:", sorted(set(re.findall(r"Acting-Agent: (\w+) session_", git("log", "-5", "--format=%B", S)))))
cnt = lambda t: sum(len(re.findall(re.escape(w), t, re.I)) for w in words)
print("product-name words (namespace words of the branches and the roster file's prefix) in the brief:", cnt(brief), "on lines",
      [i + 1 for i, l in enumerate(BL) if cnt(l)])
print("   in the worker's committed report E44_L2_SWEEP_REPORT.md:", cnt(show(S, L2 + "/sweep/E44_L2_SWEEP_REPORT.md")))
print("predictions: P60.2 chances sum", 45 + 50 + 5, "| P60.4 chances stated sum", 60 + 35)

print(f"\nrun time {time.time() - T0:.1f} s")
