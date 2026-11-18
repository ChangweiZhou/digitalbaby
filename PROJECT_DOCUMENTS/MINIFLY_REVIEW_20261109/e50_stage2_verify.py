"""Verification of the stage-2 report of E50 (the fixed input code with each cell reading 16 of its 512 values, `C16-in`,
seed 73129, under the sweep and under backpropagation; the worker's branch of E50 at c6c24d9c).

Read-only.  Nothing is trained, torch is not used, and no code of another branch is run.  Receipts, answers files, logs and
the report's files are read with `git show` into memory.  A weight file is read in one of two ways.  When the environment
variable E50S2V_SCRATCH names a folder, the file is written there with `git show`, read, and deleted at once, so that at most
one weight file is on disk at a time (the way of the first run of this script).  Otherwise the file is read with `git show`
into memory and nothing is written.  No bytecode is written in either case.  The forward pass is the reviewer's reader
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
(with E50S2V_SCRATCH=<folder> before it to pass the weight files through that folder, one at a time).
Without arguments every part is printed (about six and a half minutes on one core: part 2a about 190 seconds, part 3b about
170, part 4 about 18, part 3a about 8, every other part about one second).  The running time of each part is printed at its end.
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

REF = "c6c24d9c"                       # the worker's branch of E50: the stage-2 report, the receipts, the weights
STAGE1 = "b18e563b"                    # the commit of the stage-1 verification
BRIEF = "d6606454"                     # the brief, revision 2
RUNNER = "0f3ec4bc"                    # the commit the report names for the runner
E49REF = "a36de661"                    # the branch of E49's worker: E49's receipts
E42REF = "44f91edc"                    # E42's own branch: E42's receipts
M = "MOD97_LOCAL_RULE_20261106/"
P = M + "exp45/"
SCRATCH = os.environ.get("E50S2V_SCRATCH")          # a folder for one weight file at a time, or nothing: then no file is written
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
    """The tensors of exp45/e50/<name>, with the file's SHA-256 and size.  With a scratch folder the file is written there
    with `git show`, read and deleted; without one it is read into memory.  The tensors are kept for the other parts."""
    if (name, ref) not in _W:
        if SCRATCH:
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
        else:
            blob = show(f"{P}e50/{name}", ref)
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
    heads = [l.split() for l in git("for-each-ref", "--format=%(objectname) %(refname:short)", "refs/remotes/origin").splitlines() if "e50-fixed-input-code" in l]
    print(f"   remote branches of E50's worker in this clone: {len(heads)}; head {heads[0][0][:8] if heads else 'none'}; this script reads {REF} ({git('log', '-1', '--format=%cd', '--date=format:%Y-%m-%d %H:%M:%SZ', REF)}), which is that head: {yes(bool(heads) and heads[0][0].startswith(REF))}")
    brief = next(f for f in git("ls-tree", "--name-only", REF).splitlines() if "E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE" in f)
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
    print(f"   figures of the result table compared (eight columns, eleven figures each): {n_fig}; equal to the report: {n_ok}" + (f"; DIFFERENT: {differ}" if differ else "; none differs"))

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
        both = max(e["max80"], e["min80"])
        mm = [q[o] for q in e["curve"] for o in ("max", "min") if q["step"] >= both + 512]; m2 = [q[o] for q in e["curve"] for o in ("max", "min") if q["step"] > FIRST]
        print(f"      {r}: max and min: levels at the end {e['lv_max']:.2f}, {e['lv_min']:.2f}; within the first dose {TAB[('C16-in @24,576', r)]['lv_max']:.2f}, {TAB[('C16-in @24,576', r)]['lv_min']:.2f} (reported: stay at 96 to 98);"
              f" single rows from update {both + 512:,} on (the third row after both have reached 80): {min(mm):.2f} to {max(mm):.2f}; single rows after update 24,576: {min(m2):.2f} to {max(m2):.2f}, mean {sum(m2) / len(m2):.2f}; rows below 80 after the threshold: {sum(v < 80 for v in mm)}")
    print(f"      in16 against F10 under B: 80 on R {Ib['R80']:,} / {Fb['R80']:,} = {Ib['R80'] / Fb['R80']:.2f} (reported 2.16); 80 on A {Ib['A80']:,} / {Fb['A80']:,} = {Ib['A80'] / Fb['A80']:.2f} (reported 1.91)")
    print(f"      within the first 24,576 updates, C16-in B minus in16 B: A {B1['lv_A'] - Ib['lv_A']:+.2f}, R {B1['lv_R'] - Ib['lv_R']:+.2f}, 40 letters {B1['lv_len40'] - Ib['lv_len40']:+.2f} (reported: below on every level)")
    print(f"      S: in16 R {Is['lv_R']:.2f}, max {Is['lv_max']:.2f}, min {Is['lv_min']:.2f}; in16's max and min over all rows from 4,096 on: {min(min(q['max'], q['min']) for q in Is['curve'] if q['step'] >= 4096):.1f} to {max(max(q['max'], q['min']) for q in Is['curve'] if q['step'] >= 4096):.1f} (reported 1.4, and 39 to 41);"
          f" C16-in S at the end R {Se['lv_R']:.2f}, max {Se['lv_max']:.2f}, min {Se['lv_min']:.2f} (reported 43.6 and 96)")
    for r, e in (("B", Be), ("S", Se)):
        c = [q for q in e["curve"] if q["step"] > 0]
        more = [q["step"] for q in c if q["b1"] > q["b2"]]; less = [q["step"] for q in c if q["b1"] < q["b2"]]
        print(f"      {r}: the two answer bytes at the end: first {e['lv_b1']:.2f}, second {e['lv_b2']:.2f} (reported {'84.2 and 92.6' if r == 'B' else '71.5 and 54.0'}); rows (of {len(c)}) with the first byte right more often {len(more)}, less often {len(less)}, equally {len(c) - len(more) - len(less)};"
              f" last row with the first more often {max(more):,}; first row with the second more often {min(less) if less else 'none'}"
              + (f"; rows up to 20,480 with the first more often {sum(s <= 20480 for s in more)} of 80 (the others: {[s for s in less if s <= 20480] + [q['step'] for q in c if q['b1'] == q['b2'] and q['step'] <= 20480]}); rows from 28,672 on with the second more often {sum(s >= 28672 for s in less)} of {sum(q['step'] >= 28672 for q in c)}; between them: first more often {sum(20480 < s < 28672 for s in more)}, second {sum(20480 < s < 28672 for s in less)}" if r == "B" else f"; the rows with the second more often: {less}"))
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

# ---------------------------------------------------------------- the networks: the reader's draws, the panels, the forward
FIXED = ("code_table", "code_owner", "code_amplitude", "book.weight", "l1.weight", "l1.bias")
STRUCT = ("mask", "made", "periods", "phases", "mask_in", "mask_out", "k_in", "k_out")
TRAINED = ("rnn.weight_ih_l0", "rnn.weight_hh_l0", "rnn.bias_ih_l0", "rnn.bias_hh_l0", "lm.weight", "lm.bias", "l2.weight", "l2.bias", "l3.weight", "l3.bias")
if want & {"2a", "2b", "2c", "2d", "3a", "3b"}:
    CELLS = R42.Cells(SEED, N=10, pset="1")                       # E42's F10: the reader's own draw of the ten connections
    F = {"C16-in": R49.Interface(SEED, 16, U), "C-dense": R49.Interface(SEED, D, U)}   # the reader's own draw of E49's lists
    CODE0 = R50.Code(SEED)


def panels():
    """The panels of exp42_cells.panels for the seed, rebuilt from np_gru.py (e50_stage1_verify.panels): the panel behind A
    (300 untaught pairs of each operation at 12 letters), R (1,000 untaught pairs each of add and sub that need the reduction,
    no letters), the taught panel (300 taught pairs of each operation at 12 letters), the length test (300 untaught pairs of
    add at 12 and at 40 letters)."""
    held = {t: split(t, .5, SEED) for t in DEV}
    name = lambda t: f"{t[0]}{t[1]}"
    AP = {name(t): panel(t, held[t][1], SEED, ks=(12,), n=300)[12] for t in DEV}
    TA = {name(t): panel(t, held[t][0], SEED + 1, ks=(12,), n=300)[12] for t in DEV}
    RP = {}
    for t in DEV[:2]:
        red = [(a, b) for a, b in held[t][1] if (a + b >= 89 if t[0] == "add" else a - b < 0)]
        RP[name(t)] = [render(t, a, b, 0, None) for a, b in Random(f"e41|{SEED}|50|{t[0]}").sample(sorted(red), 1000)]
    LEN_TASK = ("add", 89, b"p")
    LP = panel(LEN_TASK, split(LEN_TASK, .5, SEED)[1], SEED, ks=(12, 40), n=300)
    return AP, RP, TA, LP


def forward_file(T, x):
    """A second form of the forward, from the tensors of the weight file alone: the code of a byte is its row of the file's
    `code_table`, and the three maps are full matrices times the file's masks.  It uses none of the reader's draws."""
    z1 = T["code_table"][x]
    Wi = T["rnn.weight_ih_l0"] * np.tile(T["mask_in"], (3, 1))
    Wh = T["rnn.weight_hh_l0"] * np.tile(T["mask"], (3, 1))
    Wo = T["lm.weight"] * T["mask_out"]
    B, L = x.shape
    h = np.zeros((B, U), np.float32); Hs = np.empty((B, L, U), np.float32)
    for t in range(L):
        gi = z1[:, t] @ Wi.T + T["rnn.bias_ih_l0"]; gh = h @ Wh.T + T["rnn.bias_hh_l0"]
        r = R42.sigmoid(gi[:, :U] + gh[:, :U]); z = R42.sigmoid(gi[:, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:])
        h = ((1 - z) * n + z * h).astype(np.float32); Hs[:, t] = h
    z2 = np.tanh(z1 @ T["l2.weight"].T + T["l2.bias"] + Hs @ Wo.T + T["lm.bias"])
    return Hs, z2 @ T["l3.weight"].T + T["l3.bias"]


class Net:
    """One saved network, evaluated by e50_code_reader.forward with the reader's own draws: the code (the file gives only the
    value of a turned-on entry), the values each cell reads (E49's lists for the setting) and the cells each cell reads."""

    def __init__(self, T, setting):
        self.T = T
        self.W = {k: v for k, v in T.items() if v.dtype == np.float32 and v.ndim > 0 and k != "code_table"}
        self.code = R50.Code(SEED, amplitude=float(T["code_amplitude"]))
        self.F, self.mask = F[setting], CELLS.mask
        self.draws_equal_file = bool(np.array_equal(T["mask"], self.mask) and np.array_equal(T["mask_in"], self.F.mask_in) and np.array_equal(T["mask_out"], self.F.mask_out))

    def run(self, x, chunk=100):
        """States (B, L, 256) and logits (B, L, 257)."""
        parts = [R50.forward(self.W, self.mask, self.F, self.code, x[s:s + chunk]) for s in range(0, len(x), chunk)]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def last(self, x, n_last, chunk=100):
        """Logits at the last n_last positions, (B, n_last, 257)."""
        return np.concatenate([R50.forward(self.W, self.mask, self.F, self.code, x[s:s + chunk])[1][:, -n_last:] for s in range(0, len(x), chunk)])

    def teacher(self, items):
        """Per item: the largest output at each answer position, given the prompt and the right bytes before it.  A whole
        answer is right under greedy generation exactly when all of them are the right bytes."""
        x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64)
        return self.last(x, len(items[0][1])).argmax(-1), np.array([a for _, a in items], dtype=np.int64)

    def exact(self, items):
        got, tgt = self.teacher(items)
        return int((got == tgt).all(1).sum())

    def bytes_apart(self, items):
        """exp50_code.score_all's two counts on R's items under greedy generation: the first byte given is the right one;
        a second byte is given (the first is not the end symbol) and is the right one.  Where the first byte given is
        wrong, the second is generated after the byte that was given, in a pass of its own."""
        got, tgt = self.teacher(items)
        o0 = got[:, 0]; first = o0 == tgt[:, 0]
        o1 = got[:, 1].copy()
        wrong = np.nonzero(~first & (o0 != EOS))[0]
        if len(wrong):
            x = np.array([[BOS] + items[i][0] + [int(o0[i])] for i in wrong], dtype=np.int64)
            o1[wrong] = self.last(x, 1)[:, 0].argmax(-1)
        second = (o0 != EOS) & (o1 == tgt[:, 1])
        return int(first.sum()), int(second.sum()), int((got == tgt).all(1).sum()), int(len(wrong))

    def respond(self, prompts, cap=4):
        """Greedy generation (exp11_cell.respond as the receipts and answers files show it): after the prompt the largest
        output is appended and fed back, until the end symbol or four bytes.  The prompts of one call have one length."""
        outs = [[] for _ in prompts]; active = list(range(len(prompts)))
        for _ in range(cap):
            if not active:
                break
            x = np.array([[BOS] + prompts[i] + outs[i] for i in active], dtype=np.int64)
            nxt = self.last(x, 1)[:, 0].argmax(-1)
            for i, b in zip(active, nxt):
                outs[i].append(int(b))
            active = [i for i, b in zip(active, nxt) if b != EOS]
        return outs


# ---------------------------------------------------------------- 2a: the recorded answers and counts from the saved weights
if "2a" in want:
    head("2a", "the recorded answers and the recorded counts from the saved weights, by the reviewer's forward pass (e50_code_reader.forward with the reader's draws); reader / receipt")
    AP, RP, TA, LP = panels()
    grand = [0, 0]
    for r in "SB":
        rc = receipt("C16-in", r)
        for suffix, row, what, sha_rec in (("", rc["curve"][-1], "the final file", rc["weights_sha256"]), (f"_at{FIRST}", next(q for q in rc["curve"] if q["step"] == FIRST), "the file at update 24,576", rc["first_dose"]["weights_sha256"])):
            t1 = time.time()
            w = weights(wname("C16-in", r, suffix)); net = Net(w["T"], "C16-in")
            print(f"   C16-in {r}, {what} ({wname('C16-in', r, suffix)}, {w['size']:,} bytes; its SHA-256 equals the receipt's: {yes(w['sha'] == sha_rec)}); against the receipt's row {row['step']:,}; the reader's draws equal the file's three masks: {yes(net.draws_equal_file)}")
            x = np.array([[BOS] + p + a[:-1] for p, a in AP["add89"][:6] + AP["sub89"][:6] + AP["max89"][:6] + AP["min89"][:6]], dtype=np.int64)
            la = R50.forward(net.W, net.mask, net.F, net.code, x)[1]; lb = forward_file(w["T"], x)[1]
            print(f"      two forms of the forward on 24 items of the panel behind A (the reader's, by index lists and its own draws; a table form from the file's tensors alone): largest difference of a logit {float(np.abs(la - lb).max()):.1e}"
                  f" (largest logit {float(np.abs(la).max()):.1f}); the same largest output at every position: {yes(bool((la.argmax(-1) == lb.argmax(-1)).all()))}")
            mine = dict(A={n: net.exact(it) for n, it in AP.items()}, R={n: net.exact(it) for n, it in RP.items()}, taught={n: net.exact(it) for n, it in TA.items()})
            theirs = dict(A=row["A_counts"], R=row["R_counts"], taught=row["taught_counts"])
            for k, nm in (("A", "the panel behind A (add, sub, max, min; 300 each)"), ("R", "R (add, sub; 1,000 each)"), ("taught", "the taught panel (add, sub, max, min; 300 each)")):
                print(f"      {nm}: {list(mine[k].values())} / {list(theirs[k].values())}: {yes(mine[k] == theirs[k])}")
            b = {n: net.bytes_apart(it) for n, it in RP.items()}
            b1 = {n: v[0] for n, v in b.items()}; b2 = {n: v[1] for n, v in b.items()}
            print(f"      R's items, first byte right {list(b1.values())} / {list(row['R_first_byte_counts'].values())}: {yes(b1 == row['R_first_byte_counts'])};"
                  f" second byte right {list(b2.values())} / {list(row['R_second_byte_counts'].values())}: {yes(b2 == row['R_second_byte_counts'])}"
                  f" (items whose first byte is wrong and whose second byte was generated after the byte given: {sum(v[3] for v in b.values())})")
            ans = json.loads(show(f"{P}e50/e50_C16-in_{r}_{SEED}{suffix}_answers.json"))
            tot = 0
            for k in (12, 40):
                rec = ans[str(k)]; got = net.respond([p for p, _ in LP[k]])
                same = sum(a == g for a, g in zip(rec, got)); tot += same
                right = sum(a == t for a, (_, t) in zip(rec, LP[k])); mine_right = sum(g == t for g, (_, t) in zip(got, LP[k]))
                print(f"      the answers file, {k} letters: {len(rec)} recorded answers; greedy generation by the reader's forward gives the recorded answer, byte for byte, in {same} of {len(LP[k])};"
                      f" right answers: the reader's {mine_right}, the recorded answers' {right}, the receipt's count {row['length_counts'][str(k)]}: {yes(mine_right == right == row['length_counts'][str(k)])}")
            grand[0] += tot; grand[1] += 600
            print(f"      recorded answers reproduced: {tot} of 600   [{time.time() - t1:.0f} s]", flush=True)
            del net
    print(f"   recorded answers reproduced over the four files: {grand[0]} of {grand[1]}")


# ---------------------------------------------------------------- 2b, 2c, 2d: the three files of each run
if want & {"2b", "2c", "2d"}:
    FILES = {r: {k: weights(wname("C16-in", r, sfx)) for k, sfx in (("start", "_start"), ("at 24,576", f"_at{FIRST}"), ("final", ""))} for r in "SB"}

if "2b" in want:
    head("2b", "the code is fixed: the start file, the file at update 24,576 and the final file of each run")
    print(f"   the digest of the code by the reviewer's reader: {CODE0.digest()} (the receipts' 12111044f0f8bc70: {yes(CODE0.digest() == '12111044f0f8bc70')})")
    dense0 = weights(wname("C-dense", "S", "_start"))["T"]
    for r in "SB":
        f = FILES[r]; S0, T1, T2 = f["start"]["T"], f["at 24,576"]["T"], f["final"]["T"]; rc = receipt("C16-in", r)
        print(f"   C16-in {r}: SHA-256 of the start file {f['start']['sha'][:16]} (receipt's start_sha256: {yes(f['start']['sha'] == rc['start_sha256'])}), of the file at 24,576 {f['at 24,576']['sha'][:16]} (receipt: {yes(f['at 24,576']['sha'] == rc['first_dose']['weights_sha256'])}),"
              f" of the final file {f['final']['sha']} (receipt's weights_sha256: {yes(f['final']['sha'] == rc['weights_sha256'])}; the report's first 16: {yes(f['final']['sha'][:16] == REPORTED_SHA16[r])})")
        print(f"      tensors in each file: {len(S0)}, {len(T1)}, {len(T2)}; the same names: {yes(set(S0) == set(T1) == set(T2))}")
        print("      bit for bit in the three files (start = at 24,576 = final): " + "; ".join(f"{k} {yes(bit(S0[k], T1[k]) and bit(S0[k], T2[k]))}" for k in FIXED))
        print("      the structure, bit for bit in the three files: " + "; ".join(f"{k} {yes(bit(S0[k], T1[k]) and bit(S0[k], T2[k]))}" for k in STRUCT))
        print("      the trained tensors, entries that differ, start to 24,576 | 24,576 to final: " + "; ".join(f"{k} {int((T1[k] != S0[k]).sum()):,} | {int((T2[k] != T1[k]).sum()):,} of {S0[k].size:,}" for k in TRAINED))
        amp = float(S0["code_amplitude"]); code = R50.Code(SEED, amplitude=amp); tb = S0["code_table"]
        z1 = np.tanh(S0["book.weight"][code.used].astype(np.float64) @ S0["l1.weight"].T.astype(np.float64) + S0["l1.bias"]); msl = float((z1 ** 2).sum(1).mean())
        print(f"      code_table {tb.shape}: equal bit for bit to the reader's table with the file's value of a turned-on entry: {yes(bit(tb, code.table()))}; entries that are not zero {int((tb != 0).sum())} (46 x 11 = 506), all equal to {amp:.6f}: {yes(bool((tb[tb != 0] == np.float32(amp)).all()))};"
              f" rows with a code {int((tb != 0).any(1).sum())}, each with {sorted(set((tb != 0).sum(1)[(tb != 0).any(1)].tolist()))} values; values with two owners: {int(((tb != 0).sum(0) > 1).sum())}; values without an owner: {int(((tb != 0).sum(0) == 0).sum())}")
        print(f"      code_owner equals the reader's list of owners: {yes(S0['code_owner'].tolist() == code.owner)}; its digest {R49.digest16(S0['code_owner'].tolist())} (the receipt's {rc['code']['digest']});"
              f" the value of a turned-on entry from the start file's byte table and l1: mean squared length {msl:.4f}, value {np.sqrt(msl / ON):.6f}; in the file {amp:.6f}, in the receipt {rc['code']['amplitude']:.6f}")
        m3 = np.tile(S0["mask_in"], (3, 1)); exp_ih = (dense0["rnn.weight_ih_l0"] * m3 * np.float32((D / 16) ** 0.5)).astype(np.float32)
        rest = [k for k in dense0 if k not in ("rnn.weight_ih_l0", "mask_in", "k_in")]
        print(f"      the start file against the start file of C-dense: the code table, its owners and its value, the byte table and l1 bit for bit: {yes(all(bit(S0[k], dense0[k]) for k in FIXED))}; every tensor but the input map, its mask and k_in bit for bit: {yes(all(bit(S0[k], dense0[k]) for k in rest))};"
              f" the input map is C-dense's start times the mask times the square root of 32, largest difference {float(np.abs(S0['rnn.weight_ih_l0'] - exp_ih).max()):.1e}")
        print(f"      the receipt: byte_table_and_l1_bit_equal_to_start {rc['byte_table_and_l1_bit_equal_to_start']}; code_table_sha256_equal_at_every_row {rc['code_table_sha256_equal_at_every_row']}; the optimizer holds state for: {', '.join(rc['optimizer_state_for'])}")
    print(f"   the two start files hold the same tensors bit for bit: {yes(all(bit(FILES['S']['start']['T'][k], FILES['B']['start']['T'][k]) for k in FILES['S']['start']['T']))}")


if "2c" in want:
    head("2c", "the masks in the three files of each run: what each cell reads, and every weight outside a mask")
    Fr = F["C16-in"]
    print(f"   the reader's draw of E49's lists for 16 values a cell: digest {R49.digest16(Fr.in_src)}; the reader's draw of E42's ten connections: {int(CELLS.mask.sum(1).min())} to {int(CELLS.mask.sum(1).max())} cells a cell, the cell itself included")
    for r in "SB":
        rc = receipt("C16-in", r)
        for k in ("start", "at 24,576", "final"):
            T = FILES[r][k]["T"]; m_in, m_out, m_rec = T["mask_in"], T["mask_out"], T["mask"]
            per = m_in.sum(1); o_in = ~np.tile(m_in, (3, 1)); o_rec = ~np.tile(m_rec, (3, 1))
            wi, wh, wo = T["rnn.weight_ih_l0"], T["rnn.weight_hh_l0"], T["lm.weight"]
            print(f"   C16-in {r}, {k}: values a cell reads (the input map's mask, 256 x 512): {int(per.min())} to {int(per.max())}, cells with exactly 16: {int((per == 16).sum())} of {U}; k_in {int(T['k_in'])}, k_out {int(T['k_out'])};"
                  f" the mask equals the reader's draw: {yes(np.array_equal(m_in, Fr.mask_in))} (the receipt's digest of the lists {rc['draw_digests']['input']}); the three gates of a cell have weights on the same values: {yes(bool(((wi != 0).reshape(3, U, D) <= m_in[None]).all()))}")
            print(f"      weights outside a mask: input map {int(o_in.sum()):,} entries outside, not zero {int((wi[o_in] != 0).sum())}; recurrent map {int(o_rec.sum()):,} outside, not zero {int((wh[o_rec] != 0).sum())}; lm's mask is full ({int(m_out.sum()):,} of {m_out.size:,}): {yes(bool(m_out.all()))}"
                  f" | inside a mask and exactly zero: input map {int((wi[~o_in] == 0).sum())} of {int((~o_in).sum()):,}, recurrent map {int((wh[~o_rec] == 0).sum())} of {int((~o_rec).sum()):,}"
                  f" | the recurrent mask equals the reader's draw: {yes(np.array_equal(m_rec, CELLS.mask))}; periods all 1: {yes(bool((T['periods'] == 1).all()))}")
        T = FILES[r]["final"]["T"]; code = R50.Code(SEED, amplitude=float(T["code_amplitude"]))
        own = np.zeros((len(code.used), D), bool)
        for i, b in enumerate(code.used):
            own[i, np.nonzero(T["code_owner"] == b)[0]] = True                                       # from the file's owners
        reads = (T["mask_in"][:, None, :] & own[None]).sum(2)                                           # (256, 46) from the file's mask
        col = lambda group: [code.used.index(b) for b in group]
        k46, k20, k10 = (reads > 0).sum(1), (reads[:, col(R50.NEEDED)] > 0).sum(1), (reads[:, col(R50.DIGITS)] > 0).sum(1)
        reach_ = (reads[:, col(R50.NEEDED)] > 0).sum(0); med = lambda v: int(sorted(v.tolist())[len(v) // 2])
        behind = np.bincount(reads[:, col(R50.NEEDED)].ravel(), minlength=4)
        print(f"   C16-in {r}, from the final file's mask and owners: a cell has input for {int(k46.min())} to {int(k46.max())} of the 46 used bytes (median {med(k46)}), for {int(k20.min())} to {int(k20.max())} of the 20 needed bytes (median {med(k20)}; the report: 1 to 10, median 6),"
              f" for {int(k10.min())} to {int(k10.max())} of the ten digits (median {med(k10)}; the report: 0 to 7, median 3); a needed byte reaches {int(reach_.min())} to {int(reach_.max())} cells; cells without input {int((k46 == 0).sum())}")
        print(f"      of the 5,120 pairs of a cell and a needed byte, the cell reads 0, 1, 2, 3 values of the byte in {behind.tolist()} (the receipt: {[rc['coverage']['weights_behind_a_cell_and_needed_byte'][str(i)] for i in range(4)]});"
              f" of a cell's 16 values, those of the 20 needed bytes: mean {reads[:, col(R50.NEEDED)].sum(1).mean():.2f}, of the 26 letters {reads[:, col(R50.LETTERS)].sum(1).mean():.2f}, without an owner {(16 - reads.sum(1)).mean():.2f}")
        n_in, n_out, n_rec = 3 * int(T["mask_in"].sum()), int(T["mask_out"].sum()), 3 * int(T["mask"].sum()); l2, l3 = T["l2.weight"].size, T["l3.weight"].size
        per_byte = [3 * int(reads[:, i].sum()) for i in range(len(code.used))]
        print(f"      multiply-adds a position, first count: input map {n_in:,} + lm {n_out:,} + cells {n_rec:,} + l2 {l2:,} + l3 {l3:,} = {n_in + n_out + n_rec + l2 + l3:,} (the brief's 545,536; the receipt {rc['multiply_adds_per_position']:,});"
              f" second count: input map {min(per_byte)} to {max(per_byte)} for a used byte, mean {sum(per_byte) / len(per_byte):.2f}, + l2 {ON * T['l2.weight'].shape[0]:,} + lm + cells + l3 = {sum(per_byte) / len(per_byte) + ON * T['l2.weight'].shape[0] + n_out + n_rec + l3:,.2f} (the receipt {rc['multiply_adds_per_position_second_count']:,.2f})")


if "2d" in want:
    head("2d", "the figures `by what a cell knows` and `other measures`, from the weight files; start, update 24,576, end / the receipt's rows 0, 24,576 and 49,152 / the report")
    kd = R50.knows(CODE0, F["C16-in"], R50.DIGITS)
    groups = [np.nonzero(kd <= 2)[0], np.nonzero((kd >= 3) & (kd <= 4))[0], np.nonzero(kd >= 5)[0]]
    print(f"   cells with input for 0 to 2, 3 or 4, 5 or more of the ten digits under 16 values a cell (the reader's draw): {[len(g) for g in groups]} (reported 95, 131, 30)")
    eye = np.eye(U, dtype=bool); NAMES = ("digits_0_to_2", "digits_3_to_4", "digits_5_or_more")

    def figures(T):
        """exp50_code.knows_figure, input_size and exp49_interface.magnitudes, in double precision from the file's tensors."""
        code = R50.Code(SEED, amplitude=float(T["code_amplitude"])); amp = np.float64(T["code_amplitude"])
        m3i = np.tile(T["mask_in"], (3, 1)); Wi = T["rnn.weight_ih_l0"].astype(np.float64) * m3i
        gi = np.stack([amp * Wi[:, code.own[b]].sum(1) for b in R50.NEEDED])                 # (20, 768): a cell's input for a needed byte, without the bias
        gd = np.stack([amp * Wi[:, code.own[b]].sum(1) for b in R50.DIGITS])
        has = np.stack([m3i[:, code.own[b]].any(1) for b in R50.NEEDED])                       # the cell reads at least one value of the byte
        m3 = np.tile(T["mask"], (3, 1)); Wh = np.abs(T["rnn.weight_hh_l0"].astype(np.float64))
        kn = []
        for g in groups:
            rows_ = np.concatenate([g, g + U, g + 2 * U]); nb = np.tile((T["mask"] & ~eye)[g], (3, 1)); ws = np.tile(T["mask"][g], (3, 1))
            kn.append((float(Wh[rows_][nb].mean()), float(Wh[rows_][ws].mean())))
        return dict(rms=float(np.sqrt((gi ** 2).mean())), rms_d=float(np.sqrt((gd ** 2).mean())), rms_has=float(np.sqrt((gi[has] ** 2).mean())), share_has=float(has.mean()),
                    ih=float(np.abs(T["rnn.weight_ih_l0"].astype(np.float64))[m3i].mean()), lm=float(np.abs(T["lm.weight"].astype(np.float64))[T["mask_out"]].mean()), hh=float(Wh[m3].mean()), kn=kn)

    REP_KN = {("C16-in", "B"): (0.265, 0.240, 0.233), ("C16-in", "S"): (0.226, 0.219, 0.213), ("C-dense", "B"): (0.147, 0.144, 0.145), ("C-dense", "S"): (0.150, 0.152, 0.149)}
    REP_START = (0.150, 0.152, 0.150)
    REP_OTHER = {"B": dict(rms=0.765, ih=0.140, dense=5.14, diff=0.03), "S": dict(rms=0.629, ih=0.117, dense=4.77, diff=0.01)}
    md = show(f"{P}e50/E50_TABLES_{SEED}_stage2.md").decode(); lines = set(md.splitlines()); wrows = []
    rep = []                                               # (what, the figure from the weight file, the report's figure, its decimals)

    for r in "BS":
        rc = receipt("C16-in", r); by = {q["step"]: q for q in rc["curve"]}
        fg = {k: figures(FILES[r][k]["T"]) for k in ("start", "at 24,576", "final")}; rw = {"start": by[0], "at 24,576": by[FIRST], "final": by[DOSE]}
        print(f"   C16-in {r}: from the weight files at the start, at 24,576 and at the end / the receipt's rows 0, 24,576 and 49,152")
        for i, n in enumerate(NAMES):
            print(f"      kept recurrent weights from neighbours into the cells of group {i + 1} ({len(groups[i])} cells) [with the cell's reading of itself]: " + ", ".join(f"{fg[k]['kn'][i][0]:.4f} [{fg[k]['kn'][i][1]:.4f}]" for k in fg)
                  + " / " + ", ".join(f"{rw[k]['by_what_a_cell_knows'][n]['neighbours_only']:.4f} [{rw[k]['by_what_a_cell_knows'][n]['with_self']:.4f}]" for k in fg)
                  + f": {yes(all(abs(fg[k]['kn'][i][j] - rw[k]['by_what_a_cell_knows'][n][w]) < 5e-7 for k in fg for j, w in ((0, 'neighbours_only'), (1, 'with_self'))))}")
            if r == "B":
                rep.append((f"by what a cell knows, group {i + 1}, at the start", fg["start"]["kn"][i][0], REP_START[i], 3))
            rep.append((f"by what a cell knows, group {i + 1}, C16-in {r}, at the end", fg["final"]["kn"][i][0], REP_KN[("C16-in", r)][i], 3))
        rep.append((f"the group with the fewest digits minus the group with the most, C16-in {r}, at the end", fg["final"]["kn"][0][0] - fg["final"]["kn"][2][0], REP_OTHER[r]["diff"], 2))
        print(f"      the group with input for the fewest digits minus the group with the most: at the start {fg['start']['kn'][0][0] - fg['start']['kn'][2][0]:+.4f}, at 24,576 {fg['at 24,576']['kn'][0][0] - fg['at 24,576']['kn'][2][0]:+.4f}, at the end {fg['final']['kn'][0][0] - fg['final']['kn'][2][0]:+.4f}")
        print("      a cell's input for a needed byte, root mean square over all 256 cells, three gates and 20 bytes, without the bias: " + ", ".join(f"{fg[k]['rms']:.4f}" for k in fg) + " / " + ", ".join(f"{rw[k]['input_size']['needed_rms']:.4f}" for k in fg)
              + f": {yes(all(abs(fg[k]['rms'] - rw[k]['input_size']['needed_rms']) < 5e-6 for k in fg))}"
              + f" | for the ten digits: " + ", ".join(f"{fg[k]['rms_d']:.4f}" for k in fg) + " / " + ", ".join(f"{rw[k]['input_size']['digits_rms']:.4f}" for k in fg))
        print(f"         the same over the pairs of a cell and a needed byte for which the cell reads a value of the byte ({100 * fg['start']['share_has']:.1f}% of the pairs; for the others the input is zero by the mask): " + ", ".join(f"{fg[k]['rms_has']:.4f}" for k in fg))
        print("      mean magnitude of a kept weight: input map " + ", ".join(f"{fg[k]['ih']:.4f}" for k in fg) + " / " + ", ".join(f"{rw[k]['kept_mean_magnitude']['rnn.weight_ih_l0']:.4f}" for k in fg)
              + f": {yes(all(abs(fg[k]['ih'] - rw[k]['kept_mean_magnitude']['rnn.weight_ih_l0']) < 5e-7 for k in fg))}"
              + " | lm " + ", ".join(f"{fg[k]['lm']:.4f}" for k in fg) + " / " + ", ".join(f"{rw[k]['kept_mean_magnitude']['lm.weight']:.4f}" for k in fg)
              + f": {yes(all(abs(fg[k]['lm'] - rw[k]['kept_mean_magnitude']['lm.weight']) < 5e-7 for k in fg))}"
              + " | recurrent " + ", ".join(f"{fg[k]['hh']:.4f}" for k in fg) + " / " + ", ".join(f"{rw[k]['kept_mean_magnitude']['rnn.weight_hh_l0']:.4f}" for k in fg)
              + f": {yes(all(abs(fg[k]['hh'] - rw[k]['kept_mean_magnitude']['rnn.weight_hh_l0']) < 5e-7 for k in fg))}")
        if r == "B":
            rep.append(("a cell's input for a needed byte, at the start", fg["start"]["rms"], 0.381, 3)); rep.append(("a kept weight of the input map, at the start", fg["start"]["ih"], 0.177, 3))
        rep.append((f"a cell's input for a needed byte, C16-in {r}, at the end", fg["final"]["rms"], REP_OTHER[r]["rms"], 3)); rep.append((f"a kept weight of the input map, C16-in {r}, at the end", fg["final"]["ih"], REP_OTHER[r]["ih"], 3))
        cs, cm, ce = rc["cosine_first_batch_start"], rc["first_dose"]["cosine_first_batch"], rc["cosine_first_batch_end"]
        for nm, k, cend in (("C16-in @24,576", "at 24,576", cm), ("C16-in @end", "final", ce)):
            a_, b_ = fg["start"], fg[k]
            wrows.append(("sizes", f"| {nm} | {r} | " + " | ".join(f"{cs[m]:.3f}, {cend[m]:.3f}" for m in ("rnn.weight_ih_l0", "lm.weight", "rnn.weight_hh_l0")) + " | " + " | ".join(f"{a_[x]:.4f}, {b_[x]:.4f}" for x in ("ih", "lm", "hh")) + f" | {a_['rms']:.3f}, {b_['rms']:.3f} |"))
            wrows.append(("by what a cell knows", f"| {nm} | {r} | " + " | ".join(f"{a_['kn'][i][0]:.4f}, {b_['kn'][i][0]:.4f} [{a_['kn'][i][1]:.4f}, {b_['kn'][i][1]:.4f}]" for i in range(3)) + " |"))
    for r in "BS":
        a_, b_ = figures(weights(wname("C-dense", r, "_start"))["T"]), figures(weights(wname("C-dense", r))["T"])
        print(f"   C-dense {r}, from its start and final files: recurrent weights from neighbours by the same three groups {', '.join(f'{a_['kn'][i][0]:.4f}' for i in range(3))} -> {', '.join(f'{b_['kn'][i][0]:.4f}' for i in range(3))};"
              f" a cell's input for a needed byte {a_['rms']:.3f} -> {b_['rms']:.3f}; mean magnitude of a kept weight of the input map {a_['ih']:.4f} -> {b_['ih']:.4f}, recurrent {a_['hh']:.4f} -> {b_['hh']:.4f}")
        for i in range(3):
            rep.append((f"by what a cell knows, group {i + 1}, C-dense {r}, at the end", b_["kn"][i][0], REP_KN[("C-dense", r)][i], 3))
        rep.append((f"a cell's input for a needed byte, C-dense {r}, at the end", b_["rms"], REP_OTHER[r]["dense"], 2))
    bad = [(w, v, q, d) for w, v, q, d in rep if f"{v:.{d}f}" != f"{q:.{d}f}"]
    print(f"   the report's figures of the two paragraphs that can be recomputed from the weight files: {len(rep)}; equal to the figure from the weight file rounded to the report's decimals: {len(rep) - len(bad)}")
    for w, v, q, d in bad:
        print(f"      DIFFERENT in the last decimal: {w}: {v:.6f} from the weight file, which is {v:.{d}f} to {d} decimals; the report has {q:.{d}f} (the tables file has {v:.4f})")
    missing = [(w, l) for w, l in wrows if l not in lines]
    print(f"   the tables file: the rows of C16-in of the two tables of sizes, written here from the weight files (the cosines from the receipts), found character for character: {len(wrows) - len(missing)} of {len(wrows)}")
    for w, l in missing:
        print(f"      NOT FOUND ({w}): {l}")


# ---------------------------------------------------------------- 3a: the use of the cells
NETS = [("C16-in", "S"), ("C16-in", "B"), ("C-dense", "S"), ("C-dense", "B")]
if "3a" in want:
    head("3a", "the use of the cells: the standard deviation of each cell's state between the 300 items of the length-test panel at 12 letters (untaught pairs of add), at the position of `=`; final weights")
    LP12 = panels()[3][12]
    x = np.array([[BOS] + p for p, _ in LP12], dtype=np.int64)
    assert bool((x[:, -1] == ord("=")).all())
    kd = R50.knows(CODE0, F["C16-in"], R50.DIGITS)
    groups = [np.nonzero(kd <= 2)[0], np.nonzero((kd >= 3) & (kd <= 4))[0], np.nonzero(kd >= 5)[0]]
    print(f"   items {x.shape[0]}, positions {x.shape[1]} (the start symbol, the task code, two digits, the operation sign, 12 letters, two digits, `=`); the state is read after the last position, where the answer's first byte is read."
          " The standard deviation is taken over the items (divisor 300) for each cell; the magnitude is the absolute value of the state, averaged over items and cells.")
    print("   | Network | Rule | Cells above 0.05 | Cells above 0.10 | Cells at or below 0.01 | Mean over cells | Least, median, largest over cells | Mean magnitude of the state | Cells with mean magnitude below 0.05 | By group of cells (0 to 2, 3 or 4, 5 or more digits): mean; cells above 0.10 |")
    print("   |---|---|---|---|---|---|---|---|---|---|")
    notes = []
    for setting, r in NETS:
        net = Net(weights(wname(setting, r))["T"], setting)
        h = net.run(x)[0][:, -1].astype(np.float64)                 # (300, 256): the state after `=`
        sd = h.std(0); mag = np.abs(h)
        print(f"   | {setting} | {r} | {int((sd > 0.05).sum())} | {int((sd > 0.10).sum())} | {int((sd <= 0.01).sum())} | {sd.mean():.4f} | {sd.min():.4f}, {np.median(sd):.4f}, {sd.max():.4f} | {mag.mean():.4f} | {int((mag.mean(0) < 0.05).sum())} | "
              + "; ".join(f"{sd[g].mean():.4f}, {int((sd[g] > 0.10).sum())} of {len(g)}" for g in groups) + " |", flush=True)
        low = np.nonzero(sd <= 0.10)[0]
        notes.append(f"   {setting} {r}: the {len(low)} cells at or below 0.10 have a mean magnitude of the state of {mag.mean(0)[low].min():.3f} to {mag.mean(0)[low].max():.3f}" if len(low) else f"   {setting} {r}: no cell at or below 0.10")
        del net
    print("   the cells that vary least between the items are not silent cells (a silent cell would have a magnitude near zero):")
    print("\n".join(notes))


# ---------------------------------------------------------------- 3b: the answers given
if "3b" in want:
    head("3b", "the answers given under greedy generation, final weights: first and second byte right, whole answer right, and the number of different answers")
    AP, RP, TA, LP = panels()
    SETS = [("R: add and sub, untaught pairs that need the reduction, no letters", RP["add89"] + RP["sub89"], (RP["add89"], RP["sub89"])),
            ("the panel behind A: add and sub, untaught pairs, 12 letters", AP["add89"] + AP["sub89"], (AP["add89"], AP["sub89"])),
            ("the panel behind A: max and min, untaught pairs, 12 letters", AP["max89"] + AP["min89"], (AP["max89"], AP["min89"]))]
    digit = lambda b: 48 <= b <= 57
    NOTES3B = []

    def counted(got, tgt):
        """What is counted on the greedy answers `got` of a set of items whose right answers are `tgt`."""
        f = [len(o) > 0 and o[0] == t[0] for o, t in zip(got, tgt)]; s = [len(o) > 1 and o[1] == t[1] for o, t in zip(got, tgt)]
        cnt = {}
        for o in got:
            cnt[tuple(o)] = cnt.get(tuple(o), 0) + 1
        top = max(cnt.items(), key=lambda kv: kv[1])
        return dict(n=len(got), first=sum(f), second=sum(s), whole=sum(o == t for o, t in zip(got, tgt)), both=sum(a and b for a, b in zip(f, s)), only1=sum(a and not b for a, b in zip(f, s)),
                    only2=sum(b and not a for a, b in zip(f, s)), neither=sum(not a and not b for a, b in zip(f, s)), given=len(cnt), right=len({tuple(t) for t in tgt}),
                    formed=sum(len(o) == 3 and digit(o[0]) and digit(o[1]) and o[2] == EOS for o in got), top=("".join(chr(b) if b < 256 else "|" for b in top[0]), top[1]),
                    firsts=len({o[0] for o in got}), seconds=len({o[1] for o in got if len(o) > 1}), given_right=len({tuple(o) for o, t in zip(got, tgt) if o == t}))

    print("   An answer is the bytes generated after `=`, up to the end symbol or four bytes; a right answer is two digits and the end symbol. `Different answers given` counts the distinct answers over the items of the set; `different right answers` counts the distinct answers that the set asks for.")
    print("   | Network | Rule | Set of items | Items | First byte right | Second byte right | Whole answer right | Both bytes right; first only; second only; neither | Different answers given | Different right answers in the set | Different first bytes; second bytes given | Answers of two digits and the end symbol | The most frequent answer given: items |")
    print("   |---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for setting, r in NETS:
        net = Net(weights(wname(setting, r))["T"], setting); last = receipt(setting, r)["curve"][-1]; notes = []
        for nm, items, halves in SETS:
            got2 = [net.respond([p for p, _ in it]) for it in halves]; tgt2 = [[t for _, t in it] for it in halves]      # the prompts of one operation have one length
            a = counted(got2[0] + got2[1], tgt2[0] + tgt2[1]); h2 = [counted(g, t) for g, t in zip(got2, tgt2)]
            pc = lambda v: f"{v:,} ({100 * v / a['n']:.1f}%)"
            print(f"   | {setting} | {r} | {nm} | {a['n']:,} | {pc(a['first'])} | {pc(a['second'])} | {pc(a['whole'])} | {a['both']:,}; {a['only1']:,}; {a['only2']:,}; {a['neither']:,} | {a['given']} | {a['right']} | {a['firsts']}; {a['seconds']} | {a['formed']:,} | `{a['top'][0][:2]}`: {a['top'][1]} |", flush=True)
            if nm.startswith("R"):
                ok = [h2[i]["whole"] for i in range(2)] == list(last["R_counts"].values()) and [h2[i]["first"] for i in range(2)] == list(last["R_first_byte_counts"].values()) and [h2[i]["second"] for i in range(2)] == list(last["R_second_byte_counts"].values())
                note = f"the counts of add and of sub apart equal the receipt's last row (whole answer, first byte, second byte): {yes(ok)}"
            else:
                key = [("add89", "sub89"), ("max89", "min89")]["max" in nm]
                note = f"the whole answers right of the two operations apart equal the receipt's last row: {yes([h2[i]['whole'] for i in range(2)] == [last['A_counts'][k] for k in key])}"
            notes.append(f"      {setting} {r}, {nm.split(':')[0]}, {nm.split(':')[1].split(',')[0].strip()}: {note}; the two operations apart: first byte right {h2[0]['first']}, {h2[1]['first']}; second {h2[0]['second']}, {h2[1]['second']}; whole {h2[0]['whole']}, {h2[1]['whole']};"
                         f" different answers given {h2[0]['given']}, {h2[1]['given']}; different right answers {h2[0]['right']}, {h2[1]['right']}; different answers among the right ones given {h2[0]['given_right']}, {h2[1]['given_right']}")
        NOTES3B.extend(notes)
        del net
    print("   the two operations of each set apart, and the counts against the receipts' last rows:")
    print("\n".join(NOTES3B))


# ---------------------------------------------------------------- 4: the stream of training items
def stream_digests(seed, n_batches, every=256):
    """exp26_candidate.Digest rebuilt (e50_stage1_verify.stream_digests): a running SHA-256 over every batch of 64 items of
    exp11_cell.stream (Random(seed + 77), the taught halves of the four tasks, 0 to 12 letters), as the three arrays of
    exp45.batch_tensor (x, y and the answer mask), each preceded by the text of its shape."""
    tp = [(t, split(t, .5, seed)[0]) for t in DEV]
    rng = Random(seed + 77)
    flat = [(t, pr) for t, pairs in tp for pr in pairs]

    def stream():
        while True:
            order = list(range(len(flat)))
            rng.shuffle(order)
            for i in order:
                t, (a, b) = flat[i]
                yield render(t, a, b, rng.randint(0, 12), rng)

    st = stream(); h = hashlib.sha256(); out = {0: h.copy().hexdigest()}
    for n in range(1, n_batches + 1):
        items = [next(st) for _ in range(BATCH)]
        rows = [p + t for p, t in items]
        L = max(len(q) for q in rows) + 1
        data = np.full((BATCH, L), BOS, np.int64)
        for i, q in enumerate(rows):
            data[i, 1:len(q) + 1] = q
        x, y = data[:, :-1], data[:, 1:]
        mask = np.zeros(y.shape, bool)
        for i, (p, t) in enumerate(items):
            mask[i, len(p):len(p) + len(t)] = True
        for a in (x, y, mask):
            h.update(str(tuple(a.shape)).encode()); h.update(np.ascontiguousarray(a).tobytes())
        if n % every == 0:
            out[n] = h.copy().hexdigest()
    return out


if "4" in want:
    head("4", "the stream of training items: the running digest of the batches, rebuilt here from np_gru.py for seed 73129, beside the receipts")
    mine = stream_digests(SEED, DOSE)
    print(f"   rebuilt: {DOSE:,} batches of {BATCH} items; digest after 4,096 batches {mine[4096][:16]} (reported bbd71ab7ca27b72e: {yes(mine[4096][:16] == 'bbd71ab7ca27b72e')}); after 24,576 {mine[FIRST][:16]}; after {DOSE:,} {mine[DOSE][:16]}")
    for r in "BS":
        rc = TAB[("C16-in @end", r)]["rc"]; rows = {q["step"]: q["data_digest"] for q in rc["curve"]}
        dn = {q["step"]: q["data_digest"] for q in TAB[("C-dense", r)]["rc"]["curve"]}
        print(f"   C16-in {r}: rows whose recorded digest equals the rebuilt one: {sum(rows[u]['sha256'] == mine[u] for u in rows)} of {len(rows)} (the number of batches recorded in each row equals its update: {yes(all(rows[u]['batches'] == u for u in rows))});"
              f" the receipt's final digest equals the rebuilt one: {yes(rc['data_digest']['sha256'] == mine[DOSE])}; the first 97 rows equal the 97 rows of C-dense under the same rule: {yes(all(rows[u] == dn[u] for u in dn))};"
              f" the same optimizer record as C-dense: {yes(rc['optimizer'] == TAB[('C-dense', r)]['rc']['optimizer'])}; the same text for the data: {yes(rc['data'] == TAB[('C-dense', r)]['rc']['data'])}; batch {rc['batch']}, letters up to {rc['kmax']}")

head(None, "")
print(f"\nrunning time {time.time() - T0:.0f} s")
