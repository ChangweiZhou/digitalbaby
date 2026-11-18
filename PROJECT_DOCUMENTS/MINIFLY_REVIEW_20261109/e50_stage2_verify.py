"""Verification of the stage-2 report of E50 (the fixed input code with each cell reading 16 of its 512 values, `C16-in`,
seed 73129, under the sweep and under backpropagation; worker's branch claude/e50-fixed-input-code-017dSE at c6c24d9c).

Read-only.  Nothing is trained, torch is not used, and no code of another branch is run.  Receipts, answers files, logs and
the report's files are read with `git show` into memory.  A weight file is written with `git show` into the scratch folder,
read, and deleted at once, so that at most one weight file is on disk at a time (the folder is the environment variable
E50S2V_SCRATCH, or the default below; nothing else is written, and no bytecode).  The forward pass is the reviewer's reader
e50_code_reader.forward (a cell's input for a byte is the value of a turned-on entry times the sum of the cell's weights on
the values of the byte that the cell reads), with the reader's own draw of the 16 values of each cell
(e49_interface_reader.Interface(seed, 16, 256)); section 2c compares that draw with the masks in the weight files, and
section 2a compares the forward with a second form built from the tensors of the file alone.  The panels are rebuilt from
np_gru.py.  The functions for the receipts and the class Net are those of e50_stage1_verify.py, extended to two readings of a
run (within its first 24,576 updates and at its end) and to greedy generation.

The parts:

   0   what is read: the commits, the runner's files, the receipts' provenance, the times of the runs
   1   the worker's result table for `C16-in`, every figure recomputed from the receipts (updates to 80 on A and on R, to
       50 on R, the levels of A, R and the length test at 40 letters, the change of R over the last 2,048 updates, max, min,
       add and sub at 12 letters; both readings), the taught panel, the figures of the report's text, the five
       predictions, the rows of the tables file and the summary file
   2a  the recorded answers and the recorded counts from the saved weights: the final file and the file at update 24,576
       of both runs
   2b  the code is fixed: the code table, its owners and its value, the byte table and `l1`, bit for bit between the start
       file, the file at update 24,576 and the final file; the SHA-256 of the files
   2c  the masks: each cell reads exactly 16 values, the lists are E49's, every weight outside a mask is zero
   2d  the figures "by what a cell knows" and "other measures", from the weight files
   3a  the use of the cells: the standard deviation of each cell's state between the 300 items of the length-test panel at
       12 letters, at the position of `=`; `C16-in` and `C-dense`, both rules, final weights
   3b  the answers given: first and second byte right and the number of different answers, add and sub together and max
       and min together; `C16-in` and `C-dense`, both rules, final weights
   4   the stream of training items: the running digest of the batches rebuilt here beside the 193 scored rows of each run

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e50_stage2_verify.py [0 1 2a 2b 2c 2d 3a 3b 4]
Without arguments every part is printed.  The running time of each part is printed at its end.
"""
import sys
sys.dont_write_bytecode = True
import hashlib, json, os, re, subprocess, time
from random import Random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from compare_pt_tensors import tensors
from np_gru import split, panel, render, DEV, BOS, EOS
import e42_cells_reader as R42
import e49_interface_reader as R49
import e50_code_reader as R50

REF = "c6c24d9c"                       # claude/e50-fixed-input-code-017dSE: the stage-2 report, the receipts, the weights
BRANCH = "origin/claude/e50-fixed-input-code-017dSE"
STAGE1 = "b18e563b"                    # the commit of the stage-1 verification
BRIEF = "d6606454"                     # the brief, revision 2
RUNNER = "0f3ec4bc"                    # the commit the report names for the runner
E49REF = "a36de661"                    # claude/peaceful-noether-vcvrmf: E49's receipts
E42REF = "44f91edc"                    # claude/gallant-dijkstra-i9zgsr: E42's receipts
M = "MOD97_LOCAL_RULE_20261106/"
P = M + "exp45/"
SCRATCH = os.environ.get("E50S2V_SCRATCH", "/tmp/claude-0/-home-user-digitalbaby/42667b5c-b7ff-5d5d-9cfa-8990d6cdcbc6/scratchpad/e50s2v")
SEED = 73129
U, D, ON, BATCH = 256, 512, 11, 64
FIRST, DOSE = 24576, 49152
DTYPE = {"FloatStorage": "<f4", "LongStorage": "<i8", "BoolStorage": "?"}
RULE = {"S": "the sweep", "B": "backpropagation"}
OPS = ("max", "min", "add", "sub")
PARTS = ("0", "1", "2a", "2b", "2c", "2d", "3a", "3b", "4")
want = set(sys.argv[1:]) or set(PARTS)
assert want <= set(PARTS), f"parts are {PARTS}"
T0 = time.time()


def show(path, ref=REF):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True, cwd=ROOT).stdout


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, check=True, cwd=ROOT, text=True).stdout.strip()


sha = lambda b: hashlib.sha256(b).hexdigest()
fu = lambda v: "not reached" if v is None else f"{v:,}"
yes = lambda b: "yes" if b else "NO"
bit = lambda a, b: a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
same1 = lambda mine, rep: all(f"{a:.1f}" == f"{b:.1f}" for a, b in zip(mine, rep))
_part = [None, 0.0]


def head(n, text):
    if _part[0] is not None:
        print(f"   [part {_part[0]}: {time.time() - _part[1]:.0f} s]", flush=True)
    _part[0], _part[1] = n, time.time()
    if n is not None:
        print(f"\n{n}. {text}", flush=True)


# ---------------------------------------------------------------- weight files: one on disk at a time, deleted when read
_W = {}


def weights(name, ref=REF):
    """The tensors of exp45/e50/<name>, with the file's SHA-256 and size.  The file is written into the scratch folder with
    `git show`, read and deleted; the tensors are kept in memory for the other parts."""
    if (name, ref) not in _W:
        os.makedirs(SCRATCH, exist_ok=True)
        path = os.path.join(SCRATCH, name)
        try:
            with open(path, "wb") as f:
                subprocess.run(["git", "show", f"{ref}:{P}e50/{name}"], stdout=f, check=True, cwd=ROOT)
            with open(path, "rb") as f:
                blob = f.read()
        finally:
            if os.path.exists(path):
                os.remove(path)
        T = {k.lstrip("/"): np.frombuffer(raw, dtype=DTYPE[stype]).reshape(shape).copy() for k, (stype, shape, raw) in tensors(blob).items()}
        _W[(name, ref)] = dict(T=T, sha=sha(blob), size=len(blob))
    return _W[(name, ref)]


wname = lambda setting, r, suffix="": f"e50_{setting}_{r}_{SEED}{suffix}.pt"


# ---------------------------------------------------------------- the receipts, every share computed again from the counts
def receipt(setting, rule):
    if setting == "F10":
        return json.loads(show(f"{P}e50/reference/e42_F10_{rule}_{SEED}.json"))
    if setting == "in16":
        return json.loads(show(f"{P}e50/reference/e49_in16_{rule}_{SEED}.json"))
    return json.loads(show(f"{P}e50/e50_{setting}_{rule}_{SEED}.json"))


def rows_of(rc):
    """The scored rows with every share computed from the row's counts of right answers; `same` says whether the shares
    that the receipt itself records equal them (e50_stage1_verify.rows_of)."""
    out, same = [], True
    for r in rc["curve"]:
        a, rr, ta, ln = r["A_counts"], r["R_counts"], r["taught_counts"], r["length_counts"]
        q = dict(step=r["step"], A=100 * (a["add89"] + a["sub89"]) / 600, R=100 * (rr["add89"] + rr["sub89"]) / 2000,
                 taught=100 * (ta["add89"] + ta["sub89"]) / 600, len12=100 * ln["12"] / 300, len40=100 * ln["40"] / 300,
                 loss=r.get("loss"), **{o: 100 * a[o + "89"] / 300 for o in OPS})
        same = same and all(q[k] == r[k] for k in ("A", "R", "taught", "len12", "len40"))
        if "R_first_byte_counts" in r:
            q["b1"] = 100 * sum(r["R_first_byte_counts"].values()) / 2000
            q["b2"] = 100 * sum(r["R_second_byte_counts"].values()) / 2000
            same = same and q["b1"] == r["R_first_byte"] and q["b2"] == r["R_second_byte"] and all(q[o] == r["ops"][o + "89"] for o in OPS)
        out.append(q)
    return out, same


def reach(curve, key, thr, rows=3):
    """The update of the first scored row from which `rows` successive rows are at or above the threshold."""
    v = [(r["step"], r[key]) for r in curve if r["step"] > 0]
    for i in range(len(v) - rows + 1):
        if all(x[1] >= thr for x in v[i:i + rows]):
            return v[i][0]
    return None


def level(curve, key, back=0):
    """Mean over three scored rows: the last three (back=0) or the three ending `back` updates before the last."""
    end = curve[-1]["step"] - back
    rows = [r[key] for r in curve if end - 512 <= r["step"] <= end]
    assert len(rows) == 3
    return sum(rows) / 3


MA = {"F10": 930560, "in16": 549632, "C-dense": 926464, "C16-in": 545536}     # the first count; C16-in's is recounted in part 2c
MA2 = {"C-dense": 285184, "C16-in": 277000}


def summary(setting, rule, upto=None):
    """One run read within its first `upto` updates, or whole (upto None)."""
    rc = receipt(setting, rule)
    c, same = rows_of(rc)
    whole = len(c)
    if upto is not None:
        c = [r for r in c if r["step"] <= upto]
    t = dict(setting=setting, rule=rule, rc=rc, curve=c, shares_equal_counts=same, dose=c[-1]["step"], rows=len(c), rows_whole=whole, complete=rc["complete"])
    for key in ("A", "R") + OPS:
        for thr in ((50, 80, 90) if key in "AR" else (80,)):
            t[f"{key}{thr}"] = reach(c, key, thr)
            t[f"{key}{thr}s"] = reach(c, key, thr, rows=1)
    for key in ("A", "R", "len40", "len12", "taught") + OPS + (("b1", "b2") if "b1" in c[0] else ()):
        t["lv_" + key] = level(c, key)
    t["rise_R"] = level(c, "R") - level(c, "R", back=2048)
    return t


def describe(x_ref, x_new, d_ref, d_new):
    """E41's four cases read as fewer / more / about the same: candidate against reference; d_ref, d_new are the last
    recordable figures (the dose less two rows in the three-row version)."""
    if x_ref and x_new:
        q = x_new / x_ref
        return "fewer" if q < 2 / 3 else "more" if q > 1.5 else "about the same"
    if x_ref and not x_new:
        return "more" if d_new / x_ref >= 1.5 else "not decided"
    if x_new and not x_ref:
        return "fewer" if x_new / d_ref <= 2 / 3 else "not decided"
    return "neither reaches"


IN_MA = {"fewer": "falls", "more": "rises", "about the same": "stays", "not decided": "not decided", "neither reaches": "neither reaches"}

# the readings: (name in the tables file, setting, reading)
READ = [("C-dense", "C-dense", None), ("C16-in @24,576", "C16-in", FIRST), ("C16-in @end", "C16-in", None), ("in16", "in16", None), ("F10", "F10", None)]
TAB = {}
if want & {"0", "1", "2a", "2d", "4"}:
    for r in "BS":
        for nm, s, upto in READ:
            TAB[(nm, r)] = summary(s, r, upto)

# what the report's result table states (E50_STAGE2_REPORT_73129.md, section "Result"); None is "not reached"
N_ = None
REPORTED = {
    ("C-dense", "B"): dict(A80=7680, R80=6912, R50=4352, lv=(97.7, 97.1, 92.7), rise=1.1, ops80=(1024, 1280, 7680, 7680)),
    ("C16-in @24,576", "B"): dict(A80=N_, R80=N_, R50=18176, lv=(67.9, 62.1, 41.6), rise=2.7, ops80=(5376, 5632, N_, N_)),
    ("C16-in @end", "B"): dict(A80=40704, R80=N_, R50=18176, lv=(83.8, 79.4, 56.2), rise=2.6, ops80=(5376, 5632, 34560, 40704)),
    ("in16", "B"): dict(A80=16640, R80=20480, R50=11776, lv=(85.2, 82.5, 66.1), rise=0.7, ops80=(5120, 5376, 15616, 18432)),
    ("C-dense", "S"): dict(A80=13056, R80=12800, R50=7424, lv=(93.5, 94.1, 90.6), rise=3.0, ops80=(1280, 1536, 12032, 13824)),
    ("C16-in @24,576", "S"): dict(A80=N_, R80=N_, R50=N_, lv=(32.7, 29.0, 9.6), rise=1.7, ops80=(7424, 7680, N_, N_)),
    ("C16-in @end", "S"): dict(A80=N_, R80=N_, R50=N_, lv=(49.5, 43.6, 13.6), rise=1.3, ops80=(7424, 7680, N_, N_)),
    ("in16", "S"): dict(A80=N_, R80=N_, R50=N_, lv=(1.9, 1.4, 1.2), rise=-0.4, ops80=(N_, N_, N_, N_)),
}
# the tables file's levels of the taught panel and of the two answer bytes (the report's text gives the bytes at the end)
REPORTED_MORE = {("C16-in @24,576", "B"): dict(taught=85.28, bytes=(77.13, 78.03)), ("C16-in @end", "B"): dict(taught=96.33, bytes=(84.23, 92.58)),
                 ("C16-in @24,576", "S"): dict(taught=54.61, bytes=(63.10, 39.57)), ("C16-in @end", "S"): dict(taught=74.83, bytes=(71.45, 53.98))}
REPORTED_SHA16 = {"B": "f9b170926b45326d", "S": "143520ef71f5a474"}
ORDER = [(nm, r) for r in "BS" for nm in ("C-dense", "C16-in @24,576", "C16-in @end", "in16")]
C16 = [(nm, r) for r in "BS" for nm in ("C16-in @24,576", "C16-in @end")]


# ---------------------------------------------------------------- 0: what is read
if "0" in want:
    head("0", "what is read")
    print(f"   the worker's branch {BRANCH} resolves to {git('rev-parse', BRANCH)[:8]}; this script reads {REF} ({git('log', '-1', '--format=%cd', '--date=format:%Y-%m-%d %H:%M:%SZ', REF)})")
    brief = "OPUS_TASK_E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE_20261116.md"
    print(f"   the brief's file is the same at {BRIEF} and at {REF}: {yes(git('rev-parse', f'{BRIEF}:{brief}') == git('rev-parse', f'{REF}:{brief}'))}; its first line names revision 2: {yes('revision 2' in show(brief).decode().splitlines()[0])}")
    print("   files that are the same (git blob) at " + REF + " and at the commits named:")
    for f, ref2, what in (("exp50_code.py", RUNNER, f"the runner at {RUNNER}, which the report names"), ("e50_sparse_code.py", RUNNER, f"the same file at {RUNNER}"),
                          ("e50_tables.py", STAGE1, f"the same file at {STAGE1} (the stage-1 verification)"),
                          ("e49_interface.py", E49REF, f"E49's worker's file at {E49REF}"), ("exp49_interface.py", E49REF, f"E49's worker's file at {E49REF}")):
        print(f"      {f}: {what}: {yes(git('rev-parse', f'{REF}:{M}{f}') == git('rev-parse', f'{ref2}:{M}{f}'))}")
    for f in [f"e50_C-dense_{r}_{SEED}{x}" for r in "SB" for x in (".json", ".pt", "_start.pt", "_answers.json")] + [f"reference/e4{n}_{s}_{r}_{SEED}.json" for n, s in ((2, "F10"), (9, "in16")) for r in "SB"]:
        if git("rev-parse", f"{REF}:{P}e50/{f}") != git("rev-parse", f"{STAGE1}:{P}e50/{f}"):
            print(f"      CHANGED since {STAGE1}: {f}")
    print(f"      the eight files of C-dense (receipts, final and start weights, answers) and the four reference receipts are unchanged since {STAGE1}, where the stage-1 verification read them")
    for r in "SB":
        copy = show(f"{P}e50/reference/e49_in16_{r}_{SEED}.json"); orig = show(f"{P}e49/e49_in16_{r}_{SEED}.json", E49REF)
        c42 = show(f"{P}e50/reference/e42_F10_{r}_{SEED}.json"); o42 = show(f"{P}e42/e42_F10_{r}_{SEED}.json", E42REF)
        print(f"      reference receipts under {r}: e49_in16 byte-equal to E49's own at {E49REF}: {yes(copy == orig)}; e42_F10 byte-equal to E42's own at {E42REF}: {yes(c42 == o42)}")
    files = git("ls-tree", "-r", "--name-only", REF, "--", P + "e50/").splitlines()
    print(f"   files of C16-in at {REF}: {len([f for f in files if 'C16-in' in f])}: " + ", ".join(sorted(f.split('/')[-1] for f in files if "C16-in" in f)))
    for r in "SB":
        rc = TAB[("C16-in @end", r)]["rc"]; pv = rc["provenance"]; t = TAB[("C16-in @end", r)]
        print(f"   receipt of C16-in {r}: command `{pv['command']}`; commit {pv['git']['commit'][:8]}, sources modified against it: {pv['git']['sources_modified_vs_commit']};"
              f" the runner's SHA-256 recorded equals that of exp50_code.py at {REF}: {yes(pv['runner_sha256'] == sha(show(M + 'exp50_code.py')))}; of e50_sparse_code.py: {yes(pv['sources_sha256']['e50_sparse_code.py'] == sha(show(M + 'e50_sparse_code.py')))}")
        walls = [q["wall"] for q in rc["curve"]]; gaps = [b - a for a, b in zip(walls, walls[1:])]
        print(f"      complete {rc['complete']}, dose {rc['dose']:,}, {t['rows_whole']} scored rows (every 256 updates from 0 to {rc['curve'][-1]['step']:,}), optimizer steps {rc['optimizer_steps']};"
              f" the shares recorded in every row equal the shares computed from the row's counts: {yes(t['shares_equal_counts'])}; k_in {rc['k_in']}, k_out {rc['k_out']}; processes beside: {rc['processes_beside']}")
        print(f"      wall-clock seconds between two successive scored rows (the receipt's `wall`): least {min(gaps):.1f}, median {sorted(gaps)[len(gaps) // 2]:.1f}, largest {max(gaps):.1f} (before row {rc['curve'][1 + gaps.index(max(gaps))]['step']:,});"
              f" up to row 5,120 (about 22:11Z) the largest is {max(g for g, q in zip(gaps, rc['curve'][1:]) if q['step'] <= 5120):.1f}; wall of the run {rc['wall'] / 3600:.3f} hours; all losses finite: {yes(all(q['finite_losses'] for q in rc['curve']))}")
        print(f"      cost: {rc['cpu_updates'] / 3600:.2f} CPU-hours of updates + {rc['cpu_scoring'] / 3600:.2f} of scoring = {(rc['cpu_updates'] + rc['cpu_scoring']) / 3600:.2f} (reported {'1.76' if r == 'B' else '2.00'})")
        lg = show(f"{P}e50/logs/e50_C16-in_{r}_{SEED}.log").decode().splitlines()
        steps = [int(l.split()[4].rstrip(":")) for l in lg if re.match(rf"e50 C16-in {r} {SEED} \d+:", l)]
        print(f"      the run's log: {len(lg)} lines; scored rows in it {len(steps)}, in order from 0 to {steps[-1]:,} without a repeat: {yes(steps == list(range(0, DOSE + 1, 256)))}; first line: {lg[0]}; last line: {lg[-1]}")
    print("   logs/queue.log:")
    for l in show(f"{P}e50/logs/queue.log").decode().splitlines():
        print(f"      {l}")
    for f in ("E50_STAGE2_REPORT_73129.md", "E50_TABLES_73129_stage2.md", "E50_SUMMARY_73129_stage2.json"):
        print(f"   {f} at {REF}: SHA-256 {sha(show(P + 'e50/' + f))[:16]}")


# ---------------------------------------------------------------- 1: the result table
if "1" in want:
    head("1", "the worker's result table, recomputed from the receipts; recomputed / reported. A threshold is reached at the first of three consecutive scored rows at or above it [a single row]; a level is the mean of the last three scored rows of the reading")
    n_fig = n_ok = 0; differ = []

    def cmp(k, what, ok):
        global n_fig, n_ok
        n_fig += 1; n_ok += bool(ok)
        if not ok:
            differ.append((k, what))
        return yes(ok)

    for k in ORDER:
        t, rep = TAB[k], REPORTED[k]
        mine = (t["lv_A"], t["lv_R"], t["lv_len40"])
        last3 = [q["step"] for q in t["curve"][-3:]]
        print(f"   {k[0]:15s} {k[1]}  (rows of the level: {last3[0]:,}, {last3[1]:,}, {last3[2]:,}; of the comparison: {last3[0] - 2048:,} to {last3[2] - 2048:,})")
        print(f"      80 on A {fu(t['A80'])} [{fu(t['A80s'])}] / {fu(rep['A80'])}: {cmp(k, 'A80', t['A80'] == rep['A80'])} | 80 on R {fu(t['R80'])} [{fu(t['R80s'])}] / {fu(rep['R80'])}: {cmp(k, 'R80', t['R80'] == rep['R80'])}"
              f" | 50 on R {fu(t['R50'])} [{fu(t['R50s'])}] / {fu(rep['R50'])}: {cmp(k, 'R50', t['R50'] == rep['R50'])} | 50 on A {fu(t['A50'])} [{fu(t['A50s'])}]; 90 on A {fu(t['A90'])}, on R {fu(t['R90'])}")
        print(f"      levels A {mine[0]:.2f}, R {mine[1]:.2f}, 40 letters {mine[2]:.2f} / {rep['lv'][0]}, {rep['lv'][1]}, {rep['lv'][2]}: " + ", ".join(cmp(k, 'level ' + w, same1((a,), (b,))) for w, a, b in zip(("A", "R", "40 letters"), mine, rep["lv"]))
              + f" | change of R {t['rise_R']:+.2f} / {rep['rise']:+.1f}: {cmp(k, 'change of R', same1((t['rise_R'],), (rep['rise'],)))} | taught {t['lv_taught']:.2f}, 12 letters {t['lv_len12']:.2f}")
        u = tuple(t[o + "80"] for o in OPS)
        print("      max, min, add, sub to 80: " + ", ".join(f"{fu(t[o + '80'])} [{fu(t[o + '80s'])}]" for o in OPS) + " / " + ", ".join(fu(v) for v in rep["ops80"]) + ": "
              + ", ".join(cmp(k, o + " to 80", a == b) for o, a, b in zip(OPS, u, rep["ops80"])) + " | levels " + ", ".join(f"{t['lv_' + o]:.2f}" for o in OPS))
    print(f"   figures of the result table compared (eight columns, 13 figures each): {n_fig}; equal to the report: {n_ok}" + (f"; DIFFERENT: {differ}" if differ else "; none differs"))

    print("   the taught panel (entry `taught`: 300 taught pairs each of add and sub at 12 letters) and the two answer bytes on R's items; recomputed / the tables file:")
    for k in C16:
        t, rep = TAB[k], REPORTED_MORE[k]
        print(f"   {k[0]:15s} {k[1]}  taught {t['lv_taught']:.2f} / {rep['taught']:.2f}: {yes(f'{t['lv_taught']:.2f}' == f'{rep['taught']:.2f}')} | first byte right {t['lv_b1']:.2f}, second {t['lv_b2']:.2f} / {rep['bytes'][0]:.2f}, {rep['bytes'][1]:.2f}:"
              f" {yes(f'{t['lv_b1']:.2f}' == f'{rep['bytes'][0]:.2f}' and f'{t['lv_b2']:.2f}' == f'{rep['bytes'][1]:.2f}')} | taught at the three rows: {[round(q['taught'], 2) for q in t['curve'][-3:]]}; all four operations of the taught panel at the last row: {t['rc']['curve'][t['rows'] - 1]['taught_counts']}")

    print("   the figures of the report's text:")
    Be, Se, B1, S1 = TAB[("C16-in @end", "B")], TAB[("C16-in @end", "S")], TAB[("C16-in @24,576", "B")], TAB[("C16-in @24,576", "S")]
    Cb, Cs, Ib, Is, Fb = TAB[("C-dense", "B")], TAB[("C-dense", "S")], TAB[("in16", "B")], TAB[("in16", "S")], TAB[("F10", "B")]
    best = max(Be["curve"], key=lambda q: q["R"])
    print(f"      B: R at the last three rows {[round(q['R'], 2) for q in Be['curve'][-3:]]} (reported 80.1, 80.5, 77.7); highest single row {best['R']:.2f} at {best['step']:,} (reported 80.75 at 47,616);"
          f" rows at or above 80 on R: {[q['step'] for q in Be['curve'] if q['R'] >= 80]}; the longest run of successive rows at or above 80: {max(len(s) for s in ''.join('x' if q['R'] >= 80 else ' ' for q in Be['curve']).split(' '))}")
    print(f"      B: 80 on A at {fu(Be['A80'])} = {Be['A80'] / Cb['A80']:.2f} times C-dense's {Cb['A80']:,} (reported 5.30); the last recordable update {DOSE - 512:,} = {(DOSE - 512) / Cb['R80']:.2f} times C-dense's {Cb['R80']:,} to 80 on R (reported 7.04)")
    for r, e, c in (("B", Be, Cb), ("S", Se, Cs)):
        d = [e["lv_" + x] - c["lv_" + x] for x in ("A", "R", "len40")]
        dA = describe(c["A80"], e["A80"], FIRST - 512, DOSE - 512); dR = describe(c["R80"], e["R80"], FIRST - 512, DOSE - 512)
        ma = [IN_MA[describe(c[x] * m[c["setting"]] if c[x] else None, e[x] * m["C16-in"] if e[x] else None, (FIRST - 512) * m[c["setting"]], (DOSE - 512) * m["C16-in"])] for m in (MA, MA2) for x in ("A80", "R80")]
        print(f"      {r}, C16-in at the end against C-dense: updates to 80 on A *{dA}*, on R *{dR}*; multiply-adds, first count: A *{ma[0]}*, R *{ma[1]}*; second count: A *{ma[2]}*, R *{ma[3]}*;"
              f" levels A {d[0]:+.2f}, R {d[1]:+.2f}, 40 letters {d[2]:+.2f}: " + ", ".join("about the same" if abs(x) < 15 else ("higher" if x > 0 else "lower") for x in d))
    for r, e in (("B", Be), ("S", Se)):
        mm = [q[o] for q in e["curve"] for o in ("max", "min") if q["step"] >= max(e["max80"], e["min80"]) + 512]
        l3 = [round(e["lv_" + o], 2) for o in ("max", "min")]
        print(f"      {r}: max and min from the row at which both have reached 80 on: least {min(mm):.2f}, largest {max(mm):.2f} in a row; levels at the end {l3}, within the first dose {[round(TAB[('C16-in @24,576', r)]['lv_' + o], 2) for o in ('max', 'min')]} (reported: stay at 96 to 98)")
    print(f"      in16 against F10 under B: 80 on R {Ib['R80']:,} / {Fb['R80']:,} = {Ib['R80'] / Fb['R80']:.2f} (reported 2.16); 80 on A {Ib['A80']:,} / {Fb['A80']:,} = {Ib['A80'] / Fb['A80']:.2f} (reported 1.91)")
    print(f"      within the first 24,576 updates, C16-in B minus in16 B: A {B1['lv_A'] - Ib['lv_A']:+.2f}, R {B1['lv_R'] - Ib['lv_R']:+.2f}, 40 letters {B1['lv_len40'] - Ib['lv_len40']:+.2f} (reported: below on every level)")
    print(f"      S: in16 R {Is['lv_R']:.2f}, max {Is['lv_max']:.2f}, min {Is['lv_min']:.2f}; in16's max and min over all rows from 4,096 on: {min(min(q['max'], q['min']) for q in Is['curve'] if q['step'] >= 4096):.1f} to {max(max(q['max'], q['min']) for q in Is['curve'] if q['step'] >= 4096):.1f} (reported 1.4, and 39 to 41);"
          f" C16-in S at the end R {Se['lv_R']:.2f}, max {Se['lv_max']:.2f}, min {Se['lv_min']:.2f} (reported 43.6 and 96)")
    for r, e in (("B", Be), ("S", Se)):
        c = [q for q in e["curve"] if q["step"] > 0]
        more = [q["step"] for q in c if q["b1"] > q["b2"]]; less = [q["step"] for q in c if q["b1"] < q["b2"]]
        print(f"      {r}: the two answer bytes at the end: first {e['lv_b1']:.2f}, second {e['lv_b2']:.2f} (reported {'84.2 and 92.6' if r == 'B' else '71.5 and 54.0'}); rows (of {len(c)}) with the first byte right more often {len(more)}, less often {len(less)}, equally {len(c) - len(more) - len(less)};"
              f" last row with the first more often {max(more):,}; first row with the second more often {min(less) if less else 'none'}"
              + (f"; rows up to 20,480 with the first more often {sum(s <= 20480 for s in more)} of 80; rows from 28,672 on with the second more often {sum(s >= 28672 for s in less)} of {sum(q['step'] >= 28672 for q in c)}; between them: first more often {sum(20480 < s < 28672 for s in more)}, second {sum(20480 < s < 28672 for s in less)}" if r == "B" else ""))
    for r in "BS":
        rc = TAB[("C16-in @end", r)]["rc"]; cv = rc["curve"]
        print(f"      {r}: rows (of {len(cv)}) in which the receipt records the code table, the byte table and l1 unchanged: {sum(all(q['fixed'].values()) for q in cv)}; every mask held with the optimizer's averages zero outside: {sum(all(all(v.values()) for v in q['masks'].values()) for q in cv)};"
              f" E42's limit held: {sum(all(q['limit'].values()) for q in cv)}; digest at 4,096: {next(q['data_digest']['sha256'] for q in cv if q['step'] == 4096)[:16]} (reported bbd71ab7ca27b72e);"
              f" weights_sha256 {rc['weights_sha256'][:16]} (reported {REPORTED_SHA16[r]}: {yes(rc['weights_sha256'][:16] == REPORTED_SHA16[r])})")
        cs, cm, ce = rc["cosine_first_batch_start"], rc["first_dose"]["cosine_first_batch"], rc["cosine_first_batch_end"]
        print(f"         cosines between the sweep's and the exact gradient on the first batch (the receipt only; not recomputed), start, 24,576, end: input map {cs['rnn.weight_ih_l0']:.3f}, {cm['rnn.weight_ih_l0']:.3f}, {ce['rnn.weight_ih_l0']:.3f};"
              f" recurrent {cs['rnn.weight_hh_l0']:.3f}, {cm['rnn.weight_hh_l0']:.3f}, {ce['rnn.weight_hh_l0']:.3f}; lm {cs['lm.weight']:.3f}, {cm['lm.weight']:.3f}, {ce['lm.weight']:.3f} (reported at the end: {'0.104, 0.293, 0.664' if r == 'B' else '0.348, 0.431, 0.669'}; at the start 0.709, 0.755, 0.866)")
        cov = rc["coverage"]
        print(f"         the setting (the receipt): a cell has input for {cov['bytes_of_20']['min']} to {cov['bytes_of_20']['max']} of the 20 needed bytes (median {cov['bytes_of_20']['median']}), for {cov['digits_of_10']['min']} to {cov['digits_of_10']['max']} of the ten digits (median {cov['digits_of_10']['median']});"
              f" start scale of the input map {rc['start_scale']['input']:.4f} (reported 5.66)")

    print("   the predictions (the reviewer's, from the brief's revision 2), scored here from the receipts / the worker's outcome:")
    print(f"      P50.4'  C16-in reaches 80 on R within 49,152 updates under backpropagation (55%): {'true' if Be['R80'] else 'false'} / false")
    print(f"      P50.5'  the same under the sweep (25%): {'true' if Se['R80'] else 'false'} / false")
    print(f"      P50.6'  under backpropagation C16-in reaches 80 on R in no more than 3.24 times the updates of C-dense (45%): 3.24 x {Cb['R80']:,} = {3.24 * Cb['R80']:,.1f}; {'true' if Be['R80'] and Be['R80'] <= 3.24 * Cb['R80'] else 'false'} / false (reported: not by 22,395)")
    print(f"      P50.7'  max and min at 12 letters reach 80 under backpropagation (90%): {'true' if Be['max80'] and Be['min80'] else 'false'} ({fu(Be['max80'])}, {fu(Be['min80'])}) / true (5,376 and 5,632)")
    print(f"      P50.8'  the same under the sweep (45%): {'true' if Se['max80'] and Se['min80'] else 'false'} ({fu(Se['max80'])}, {fu(Se['min80'])}) / true (7,424 and 7,680)")
    print(f"      the explanation under test: C16-in under the sweep does not reach 80 on R within its dose ({yes(Se['R80'] is None)}) while C-dense under the sweep does ({yes(Cs['R80'] is not None)}, at {fu(Cs['R80'])}): the brief's condition for *refuted as a sufficient one* is met: {yes(Se['R80'] is None and Cs['R80'] is not None)}")

    # the tables file, row by row, in the worker's formats, from the figures recomputed here
    md = show(f"{P}e50/E50_TABLES_{SEED}_stage2.md").decode(); lines = set(md.splitlines())
    exp = []
    for r in "BS":
        for nm in ("C16-in @24,576", "C16-in @end"):
            t = TAB[(nm, r)]; c = t["curve"]; rc = t["rc"]; pre = f"| {nm} | {r} | "; s = "C16-in"; raw = rc["curve"][t["rows"] - 1]
            mx = lambda key: max(c, key=lambda q: q[key])
            exp.append(("thresholds", pre + " | ".join(f"{fu(t[f'{key}{th}'])} [{fu(t[f'{key}{th}s'])}]" for key in "AR" for th in (50, 80, 90))
                        + f" | {mx('A')['A']:.2f} ({mx('A')['step']:,}) | {mx('R')['R']:.2f} ({mx('R')['step']:,}) | {raw['cpu_updates'] / 3600:.2f} + {raw['cpu_scoring'] / 3600:.2f} |"))
            exp.append(("four operations", pre + " | ".join(f"{fu(t[o + '80'])} [{fu(t[o + '80s'])}]" for o in OPS) + " | " + ", ".join(f"{t['lv_' + o]:.1f}" for o in OPS) + " |"))
            g3 = lambda key, m: "not reached" if t[key] is None else f"{m * BATCH * t[key]:.3g}"
            exp.append(("multiply-adds", pre + f"{MA[s]:,} | {g3('A80', MA[s])} | {g3('R80', MA[s])} | {MA2[s]:,} | {g3('A80', MA2[s])} | {g3('R80', MA2[s])} |"))
            exp.append(("levels", pre + f"{t['lv_A']:.2f} | {t['lv_R']:.2f} | {t['lv_len40']:.2f} | {t['lv_len12']:.2f} | {t['lv_taught']:.2f} | {t['lv_b1']:.2f} | {t['lv_b2']:.2f} | {t['rise_R']:+.2f}{', still rising' if t['rise_R'] >= 5 and t['R80'] is None else ''} |"))
            d4 = next(q["data_digest"]["sha256"] for q in rc["curve"] if q["step"] == 4096)[:16]; cv = rc["curve"][:t["rows"]]
            exp.append(("checks inside the runs", pre + f"{d4} | {all(all(all(v.values()) for v in q['masks'].values()) for q in cv)} | {all(all(q['limit'].values()) for q in cv)} | {all(all(q['fixed'].values()) for q in cv)} |"))
            ref = TAB[("C-dense", r)]; pre2 = f"| {nm} | C-dense | {r} | "
            for key in ("A80", "R80"):
                d3 = describe(ref[key], t[key], FIRST - 512, t["dose"] - 512); d1 = describe(ref[key + "s"], t[key + "s"], FIRST, t["dose"])
                exp.append(("descriptions", pre2 + f"{key} updates | {d3}{'*' if d1 != d3 else ''} | {fu(ref[key])} | {fu(t[key])} | {'' if t[key] is None else f'{t[key] / ref[key]:.2f}'} |"))
                for m, tail in ((MA, ""), (MA2, ", second count")):
                    xr = ref[key] * m["C-dense"] * BATCH; xc = None if t[key] is None else t[key] * m[s] * BATCH
                    xrs = ref[key + "s"] * m["C-dense"] * BATCH; xcs = None if t[key + "s"] is None else t[key + "s"] * m[s] * BATCH
                    m3 = IN_MA[describe(xr, xc, (FIRST - 512) * m["C-dense"] * BATCH, (t["dose"] - 512) * m[s] * BATCH)]; m1 = IN_MA[describe(xrs, xcs, FIRST * m["C-dense"] * BATCH, t["dose"] * m[s] * BATCH)]
                    exp.append(("descriptions", pre2 + f"{key} multiply-adds{tail} | {m3}{'*' if m1 != m3 else ''} | {xr:.3g} | {'not reached' if xc is None else f'{xc:.3g}'} | {'' if xc is None else f'{xc / xr:.2f}'} |"))
            for key, w in (("A", "A"), ("R", "R"), ("len40", "the length test at 40")):
                d = t["lv_" + key] - ref["lv_" + key]
                exp.append(("descriptions", pre2 + f"level of {w} | {'lower' if d <= -15 else 'higher' if d >= 15 else 'about the same'} | {ref['lv_' + key]:.2f} | {t['lv_' + key]:.2f} | |"))
        t = TAB[("C16-in @end", r)]; by = {q["step"]: q for q in t["curve"]}; pre = f"| C16-in | {r} | "
        exp.append(("A and R every 2,048", pre + " | ".join(f"{by[u]['A']:.1f}, {by[u]['R']:.1f} ({by[u]['b1']:.0f}, {by[u]['b2']:.0f})" for u in range(2048, DOSE + 1, 2048)) + " |"))
        exp.append(("operations every 4,096", pre + " | ".join(", ".join(f"{by[u][o]:.0f}" for o in OPS) for u in range(4096, DOSE + 1, 4096)) + " |"))
        exp.append(("training loss", pre + " | ".join(f"{by[u]['loss']:.3f}" for u in (256, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 32768, 40960, 49152)) + " |"))
    missing = [(w, l) for w, l in exp if l not in lines]
    kinds = sorted({w for w, _ in exp}, key=[w for w, _ in exp].index)
    print(f"   the tables file E50_TABLES_{SEED}_stage2.md: the rows of C16-in written here from the recomputed figures, in the tables' own formats, and found in the file character for character: {len(exp) - len(missing)} of {len(exp)}")
    print("      " + "; ".join(f"{w}: {sum(1 for a, _ in exp if a == w) - sum(1 for a, _ in missing if a == w)} of {sum(1 for a, _ in exp if a == w)}" for w in kinds))
    for w, l in missing:
        print(f"      NOT FOUND ({w}): {l}")
    for r in "BS":
        print(f"      the file's rule for stage 3 under {r}: " + next((l for l in md.splitlines() if l.startswith(f"- {r}, stage 3")), "NOT FOUND"))
    sm = json.loads(show(f"{P}e50/E50_SUMMARY_{SEED}_stage2.json")); n_ok = n_all = 0; bad = []
    for r in "BS":
        for nm in ("C16-in @24,576", "C16-in @end", "C-dense", "in16", "F10"):
            t, v = TAB[(nm, r)], sm["runs"].get(f"{nm} {r}")
            if v is None:
                bad.append((nm, r, "not in the summary")); continue
            for key in ("A", "R"):
                for th in (50, 80, 90):
                    for ver, suf in (("three", ""), ("single", "s")):
                        w = v["thresholds"][f"{key}{th}_{ver}"]; n_all += 1; ok = (w["updates"] if w else None) == t[f"{key}{th}{suf}"]; n_ok += ok
                        if not ok:
                            bad.append((nm, r, key, th, ver))
            for o in OPS:
                for ver, suf in (("three", ""), ("single", "s")):
                    w = v["thresholds"][f"{o}80_{ver}"]; n_all += 1; ok = (w["updates"] if w else None) == t[f"{o}80{suf}"]; n_ok += ok
                    if not ok:
                        bad.append((nm, r, o, 80, ver))
            for key, mine in [(x, t["lv_" + x]) for x in ("A", "R", "len40", "len12", "taught") + OPS] + ([("R_first_byte", t["lv_b1"]), ("R_second_byte", t["lv_b2"])] if "lv_b1" in t else []):
                n_all += 1; ok = abs(v["levels"][key] - mine) < 1e-9; n_ok += ok
                if not ok:
                    bad.append((nm, r, "level", key))
            n_all += 1; ok = abs(v["R_last3_minus_2048_earlier"] - t["rise_R"]) < 1e-9; n_ok += ok
            if not ok:
                bad.append((nm, r, "change of R"))
    print(f"   the summary file E50_SUMMARY_{SEED}_stage2.json: thresholds (A and R at 50, 80, 90; the four operations at 80; both versions), levels and the change of R of the ten readings equal to the figures recomputed here: {n_ok} of {n_all}" + (f"; differing: {bad}" if bad else ""))

head(None, "")
print(f"\nrunning time {time.time() - T0:.0f} s")
