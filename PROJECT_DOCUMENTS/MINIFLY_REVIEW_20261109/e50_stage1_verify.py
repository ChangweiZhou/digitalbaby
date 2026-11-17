"""Verification of the stage-1 report of E50 (the fixed input code with the dense interface, `C-dense`, seed 73129, under
the sweep and under backpropagation; worker's branch claude/e50-fixed-input-code-017dSE at b18e563b).

Read-only.  Nothing is trained, torch is not used, and no code of another branch is run: every file is read with `git show`,
one at a time, into memory.  The forward pass is e50_code_reader.forward (the reviewer's reader, written from the brief's
text: a cell's input for a byte is the value of a turned-on entry times the sum of the cell's weights on the byte's values);
the panels and the training stream are rebuilt from np_gru.py.  The comparison runs (E42's `F10`) are read from E42's own
branch, claude/gallant-dijkstra-i9zgsr at 44f91edc, and not from the worker's copies.

The sections carry the numbers of the claims that were to be checked:

   0  what is read: the commits, the worker's copies of the reference receipts against the originals, the runner's files
   1  updates to 80 on A and on R (three-row and single-row version), the ratios to `F10`, the brief's descriptions
   2  the levels of A, of R and of the length test at 40 letters, and the change of R at the end of the dose
   3  the four operations apart: updates to 80 and levels
   4  the answer's two bytes apart (from the receipts; the last row again from the final weights in section 6)
   5  the weight files: SHA-256, the code table, the byte table and `l1` bit for bit between start and end, the masks,
      every weight outside a mask, the digest of the code by the reader
   6  the recorded scores are those of the saved weights: every count of the last scored row and every recorded answer
      of the length-test panel, recomputed with the reader's forward pass
   7  multiply-adds a position, recounted from the masks in the weight files and from the reader's code
   8  the stream of training items: the running digest of the batches rebuilt here, beside every scored row of the
      E50 receipts and of E42's `F10` receipts
   9  the four predictions, and the receipts of the checks before the runs
  10  the rule for stage 2, the launch script, the stage-2 logs and the setting `C16-in` as this runner builds it
  11  what a seed alone changes: E42's `F10` on three seeds beside the differences `C-dense` minus `F10`
  12  other figures of the report (a cell's input for a needed byte, the size of the weights, the figure "by what a
      cell knows", the cost, the tables file row by row, the summary file)

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e50_stage1_verify.py [0 1 ... 12]
Without arguments every section is printed (about two minutes on one core).
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

REF = "b18e563b"                       # claude/e50-fixed-input-code-017dSE: the stage report, the receipts, the weights
BRANCH = "origin/claude/e50-fixed-input-code-017dSE"
BRIEF = "d6606454"                     # the brief, revision 2
RUNNER = "0f3ec4bc"                    # the commit the receipts name for the runner
E42REF = "44f91edc"                    # claude/gallant-dijkstra-i9zgsr: E42's own receipts, three seeds
E49REF = "a36de661"                    # claude/peaceful-noether-vcvrmf: E49's receipts, its copies of E42's, E42's start of F10
DEFAULT = "origin/codex/minifly-consolidated-20261023"
M = "MOD97_LOCAL_RULE_20261106/"
P = M + "exp45/"
SEED, SEEDS = 73129, (73129, 93683, 116089)
U, D, ON, BATCH = 256, 512, 11, 64
DOSE = 24576
DTYPE = {"FloatStorage": "<f4", "LongStorage": "<i8", "BoolStorage": "?"}
RULE = {"S": "the sweep", "B": "backpropagation"}
OPS = ("max", "min", "add", "sub")
want = set(sys.argv[1:]) or {str(i) for i in range(13)}
T0 = time.time()


def show(path, ref):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True, cwd=ROOT).stdout


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, check=True, cwd=ROOT, text=True).stdout.strip()


def load(blob):
    return {name.lstrip("/"): np.frombuffer(raw, dtype=DTYPE[stype]).reshape(shape).copy() for name, (stype, shape, raw) in tensors(blob).items()}


sha = lambda b: hashlib.sha256(b).hexdigest()
fu = lambda v: "not reached" if v is None else f"{v:,}"
yes = lambda b: "yes" if b else "NO"


def head(n, text):
    print(f"\n{n}. {text}", flush=True)


# ---------------------------------------------------------------- the receipts, every share computed again from the counts
def receipt(setting, rule, seed=SEED):
    if setting == "F10":
        return json.loads(show(f"{P}e42/e42_F10_{rule}_{seed}.json", E42REF))
    if setting == "in16":
        return json.loads(show(f"{P}e49/e49_in16_{rule}_{seed}.json", E49REF))
    return json.loads(show(f"{P}e50/e50_{setting}_{rule}_{seed}.json", REF))


def rows_of(rc):
    """The scored rows with every share computed from the row's counts of right answers; `same` says whether the shares
    that the receipt itself records equal them."""
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


MA = {"F10": 930560, "in16": 549632, "C-dense": 926464}            # recounted in section 7 (C-dense, F10) and in E49's verification (in16)
MA2 = {"C-dense": 285184}


def summary(setting, rule, seed=SEED):
    rc = receipt(setting, rule, seed)
    c, same = rows_of(rc)
    t = dict(setting=setting, rule=rule, rc=rc, curve=c, shares_equal_counts=same, dose=c[-1]["step"], rows=len(c), complete=rc["complete"])
    for key in ("A", "R") + OPS:
        for thr in ((50, 80, 90) if key in "AR" else (80,)):
            t[f"{key}{thr}"] = reach(c, key, thr)
            t[f"{key}{thr}s"] = reach(c, key, thr, rows=1)
    for key in ("A", "R", "len40", "len12", "taught") + OPS + (("b1", "b2") if "b1" in c[0] else ()):
        t["lv_" + key] = level(c, key)
    t["rise_R"] = level(c, "R") - level(c, "R", back=2048)
    return t


def describe(x_ref, x_new, d_ref, d_new):
    """E41's four cases read as fewer / more / about the same (the brief's form for updates to the 80 threshold): candidate
    against reference; d_ref, d_new are the last recordable figures (the dose less two rows in the three-row version)."""
    if x_ref and x_new:
        q = x_new / x_ref
        return "fewer" if q < 2 / 3 else "more" if q > 1.5 else "about the same"
    if x_ref and not x_new:
        return "more" if d_new / x_ref >= 1.5 else "not decided"
    if x_new and not x_ref:
        return "fewer" if x_new / d_ref <= 2 / 3 else "not decided"
    return "neither reaches"


IN_MA = {"fewer": "falls", "more": "rises", "about the same": "stays", "not decided": "not decided", "neither reaches": "neither reaches"}

TAB = {}
if want & {"1", "2", "3", "4", "7", "8", "9", "10", "11", "12"}:
    for r in "BS":
        for s in ("C-dense", "F10", "in16"):
            TAB[(s, r)] = summary(s, r)

# what the worker's message and report state (the report's first two tables, its text, and the task's list of claims)
REPORTED = {("C-dense", "B"): dict(A80=7680, R80=6912, lv=(97.7, 97.1, 92.7), rise=1.1, ops80=(1024, 1280, 7680, 7680), opslv=(97.8, 96.9, 97.3, 98.0), bytes=(98.1, 97.6)),
            ("F10", "B"): dict(A80=8704, R80=9472, lv=(93.2, 94.1, 92.6), rise=2.1, ops80=(2048, 2048, 8704, 9472), opslv=(97.0, 97.4, 92.9, 93.6)),
            ("C-dense", "S"): dict(A80=13056, R80=12800, lv=(93.5, 94.1, 90.6), rise=3.0, ops80=(1280, 1536, 12032, 13824), opslv=(97.7, 95.7, 94.0, 93.0), bytes=(96.0, 94.3)),
            ("F10", "S"): dict(A80=13312, R80=15104, lv=(91.2, 89.3, 89.1), rise=0.7, ops80=(3072, 2816, 11776, 15104), opslv=(97.8, 97.8, 90.1, 92.3))}
REPORTED_RATIO = {("B", "A80"): 0.88, ("B", "R80"): 0.73, ("S", "A80"): 0.98, ("S", "R80"): 0.85}
REPORTED_SHA = {"S": "056ca79d181f1ec2192ea653aca84f1ad9e739f0a6c90c0b06f24245f8ea78c4", "B": "af1315dd2cd048a103238f5fe5517bdae50d1ca397973c1bbe80ccafef3e386a"}
same1 = lambda mine, rep: all(f"{a:.1f}" == f"{b:.1f}" for a, b in zip(mine, rep))
ORDER = [("C-dense", "B"), ("F10", "B"), ("C-dense", "S"), ("F10", "S")]


# ---------------------------------------------------------------- 0: what is read
if "0" in want:
    head(0, "what is read")
    print(f"   the worker's branch {BRANCH} resolves to {git('rev-parse', BRANCH)[:8]}; this script reads {REF} ({git('rev-parse', REF)[:8]});"
          f" the brief at {BRIEF}, E42's receipts at {E42REF} ({git('rev-parse', 'origin/claude/gallant-dijkstra-i9zgsr')[:8]} is the head of its branch), E49's at {E49REF}")
    brief = "OPUS_TASK_E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE_20261116.md"
    print(f"   the brief's file is the same at {BRIEF} and at {REF}: {yes(git('rev-parse', f'{BRIEF}:{brief}') == git('rev-parse', f'{REF}:{brief}'))}; its first line names revision 2: {yes('revision 2' in show(brief, BRIEF).decode().splitlines()[0])}")
    print("   the worker's copies of the reference receipts (exp45/e50/reference/) against the originals:")
    src = show(f"{P}e50/reference/SOURCE.txt", REF).decode()
    for r in "BS":
        copy = show(f"{P}e50/reference/e42_F10_{r}_{SEED}.json", REF); orig = show(f"{P}e42/e42_F10_{r}_{SEED}.json", E42REF)
        via = show(f"{P}e49/reference_e42/e42_F10_{r}_{SEED}.json", E49REF)
        print(f"      e42_F10_{r}_{SEED}.json: {len(copy):,} bytes, SHA-256 {sha(copy)[:16]}; byte-equal to E42's own receipt at {E42REF}: {yes(copy == orig)};"
              f" to E49's worker's copy at {E49REF}: {yes(copy == via)}; its SHA-256 is the one in SOURCE.txt: {yes(sha(copy) in src)}")
    for r in "BS":
        copy = show(f"{P}e50/reference/e49_in16_{r}_{SEED}.json", REF); orig = show(f"{P}e49/e49_in16_{r}_{SEED}.json", E49REF)
        print(f"      e49_in16_{r}_{SEED}.json: byte-equal to E49's own receipt at {E49REF}: {yes(copy == orig)}; its SHA-256 is the one in SOURCE.txt: {yes(sha(copy) in src)}")
    print("   the runner's files (git blob ids):")
    for f, ref2, what in (("exp50_code.py", RUNNER, f"the runner at {RUNNER}"), ("e50_sparse_code.py", RUNNER, f"the same file at {RUNNER}"), ("e50_tables.py", "c85d9af3", "the same file at the report's commit c85d9af3"),
                          ("e49_interface.py", E49REF, f"E49's worker's file at {E49REF}"), ("exp49_interface.py", E49REF, f"E49's worker's file at {E49REF}"),
                          ("e42_cells.py", DEFAULT, "the default branch's file"), ("exp42_cells.py", DEFAULT, "the default branch's file")):
        print(f"      {f}: at {REF} equal to {what}: {yes(git('rev-parse', f'{REF}:{M}{f}') == git('rev-parse', f'{ref2}:{M}{f}'))}")
    for r in "BS":
        pv = receipt("C-dense", r)["provenance"]
        print(f"      receipt of C-dense {r}: command `{pv['command']}`; commit {pv['git']['commit'][:8]}, sources modified against it: {pv['git']['sources_modified_vs_commit']};"
              f" recorded SHA-256 of the runner equals that of exp50_code.py at {REF}: {yes(pv['runner_sha256'] == sha(show(M + 'exp50_code.py', REF)))};"
              f" of e50_sparse_code.py: {yes(pv['sources_sha256']['e50_sparse_code.py'] == sha(show(M + 'e50_sparse_code.py', REF)))}")
    for f in ("E50_STAGE1_REPORT_73129.md", "E50_TABLES_73129_stage1.md", "E50_SUMMARY_73129_stage1.json"):
        print(f"      {f} at {REF}: SHA-256 {sha(show(P + 'e50/' + f, REF))[:16]}")
    for s in ("C-dense", "F10"):
        for r in "BS":
            rc = receipt(s, r); c, same = rows_of(rc)
            print(f"      {s} {r}: complete {rc['complete']}, dose {rc['dose']:,}, {len(c)} scored rows (every 256 updates from 0 to {c[-1]['step']:,}); the shares recorded in every row equal the shares computed from the row's counts: {yes(same)}")


# ---------------------------------------------------------------- 1: updates to 80
if "1" in want:
    head(1, "updates to 80 on A and on R: the first of three consecutive scored rows at or above 80 [a single row]; recomputed / reported")
    for k in ORDER:
        t, rep = TAB[k], REPORTED[k]
        print(f"   {k[0]:7s} {k[1]}  A {fu(t['A80']):>6} [{fu(t['A80s']):>6}] / {rep['A80']:,}: {yes(t['A80'] == rep['A80'])} | R {fu(t['R80']):>6} [{fu(t['R80s']):>6}] / {rep['R80']:,}: {yes(t['R80'] == rep['R80'])}"
              f" | 50 and 90: A {fu(t['A50'])} [{fu(t['A50s'])}], {fu(t['A90'])} [{fu(t['A90s'])}]; R {fu(t['R50'])} [{fu(t['R50s'])}], {fu(t['R90'])} [{fu(t['R90s'])}]")
    print("   C-dense against F10 of the same rule and seed. The brief's bands (E41's four cases): *fewer* below two thirds, *more* above one and a half, otherwise *about the same*.")
    print(f"   The last update at which a threshold can be recorded in the three-row version is {DOSE - 512:,}.")
    for r in "BS":
        c, f = TAB[("C-dense", r)], TAB[("F10", r)]
        for key in ("A80", "R80"):
            q, qs = c[key] / f[key], c[key + "s"] / f[key + "s"]
            d, ds = describe(f[key], c[key], DOSE - 512, DOSE - 512), describe(f[key + "s"], c[key + "s"], DOSE, DOSE)
            qm = q * MA["C-dense"] / MA["F10"]; qms = qs * MA["C-dense"] / MA["F10"]
            im = "falls" if qm < 2 / 3 else "rises" if qm > 1.5 else "stays"; ims = "falls" if qms < 2 / 3 else "rises" if qms > 1.5 else "stays"
            print(f"   {r} {key}: ratio {q:.4f} (reported {REPORTED_RATIO[(r, key)]:.2f}: {yes(f'{q:.2f}' == f'{REPORTED_RATIO[(r, key)]:.2f}')}), *{d}*; difference {c[key] - f[key]:+,} updates"
                  f" | single row: ratio {qs:.4f}, *{ds}* | in multiply-adds (first count, {MA['C-dense']:,} against {MA['F10']:,}): ratio {qm:.4f}, *{im}*; single row {qms:.4f}, *{ims}*")
    print(f"   *fewer* would need 80 on R at or before update {max(u for u in range(256, DOSE + 1, 256) if u / TAB[('F10', 'B')]['R80'] < 2 / 3):,} under backpropagation"
          f" and {max(u for u in range(256, DOSE + 1, 256) if u / TAB[('F10', 'S')]['R80'] < 2 / 3):,} under the sweep (scored rows are 256 updates apart)")


# ---------------------------------------------------------------- 2: levels
if "2" in want:
    head(2, "levels (mean of the last three scored rows: updates 24,064, 24,320 and 24,576) of A, R and the length test at 40 letters; recomputed / reported")
    for k in ORDER:
        t, rep = TAB[k], REPORTED[k]
        mine = (t["lv_A"], t["lv_R"], t["lv_len40"])
        print(f"   {k[0]:7s} {k[1]}  A {mine[0]:.2f}, R {mine[1]:.2f}, 40 letters {mine[2]:.2f} / {rep['lv'][0]}, {rep['lv'][1]}, {rep['lv'][2]}: {yes(same1(mine, rep['lv']))}"
              f" | R: last three rows minus the three ending 2,048 updates earlier (21,504 to 22,016) {t['rise_R']:+.2f} / {rep['rise']:+.1f}: {yes(same1((t['rise_R'],), (rep['rise'],)))}"
              f" | also: 12 letters {t['lv_len12']:.2f}, taught {t['lv_taught']:.2f}")
    for r in "BS":
        c, f = TAB[("C-dense", r)], TAB[("F10", r)]
        d = [c["lv_" + k] - f["lv_" + k] for k in ("A", "R", "len40")]
        print(f"   {r}: C-dense minus F10, A {d[0]:+.2f}, R {d[1]:+.2f}, 40 letters {d[2]:+.2f}; the brief's form (*lower* or *higher* at 15 points or more): "
              + ", ".join("about the same" if abs(x) < 15 else ("higher" if x > 0 else "lower") for x in d))
    for k in ORDER:
        t = TAB[k]
        print(f"   {k[0]:7s} {k[1]}  R at the last nine rows: {[round(x['R'], 2) for x in t['curve'][-9:]]}")


# ---------------------------------------------------------------- 3: the four operations
if "3" in want:
    head(3, "the four operations apart (the panel behind A: 300 untaught pairs each at 12 letters): updates to 80, three rows [single row], and levels; recomputed / reported")
    for k in ORDER:
        t, rep = TAB[k], REPORTED[k]
        u = tuple(t[o + "80"] for o in OPS); lv = tuple(t["lv_" + o] for o in OPS)
        print(f"   {k[0]:7s} {k[1]}  max, min, add, sub to 80: " + ", ".join(f"{fu(t[o + '80'])} [{fu(t[o + '80s'])}]" for o in OPS) + f" / {', '.join(f'{v:,}' for v in rep['ops80'])}: {yes(u == rep['ops80'])}"
              f" | levels {', '.join(f'{v:.2f}' for v in lv)} / {', '.join(str(v) for v in rep['opslv'])}: {yes(same1(lv, rep['opslv']))}")


# ---------------------------------------------------------------- 4: the answer's two bytes
if "4" in want:
    head(4, "the answer's two bytes apart, on R's 2,000 items: share with the first byte right, with the second byte right (levels); recomputed / reported")
    for r in "BS":
        t, rep = TAB[("C-dense", r)], REPORTED[("C-dense", r)]
        mine = (t["lv_b1"], t["lv_b2"])
        last = t["rc"]["curve"][-1]
        print(f"   C-dense {r}  first {mine[0]:.2f}, second {mine[1]:.2f} / {rep['bytes'][0]}, {rep['bytes'][1]}: {yes(same1(mine, rep['bytes']))}"
              f" | counts at the last row: first {last['R_first_byte_counts']}, second {last['R_second_byte_counts']}, whole answer {last['R_counts']}")
    by = {r: {q["step"]: q for q in TAB[("C-dense", r)]["curve"]} for r in "BS"}
    print(f"   early rows quoted by the report: B at 2,048 updates first {by['B'][2048]['b1']:.2f}, second {by['B'][2048]['b2']:.2f} (reported 42 against 13);"
          f" S at 4,096 first {by['S'][4096]['b1']:.2f}, second {by['S'][4096]['b2']:.2f} (reported 60 against 26)")
    for r in "BS":
        c = [q for q in TAB[("C-dense", r)]["curve"] if q["step"] > 0]
        more = sum(q["b1"] > q["b2"] for q in c); eq = sum(q["b1"] == q["b2"] for q in c)
        firstle = next((q["step"] for q in c if q["b1"] <= q["b2"]), None)
        print(f"   C-dense {r}  rows (of {len(c)}) in which the first byte is right more often than the second: {more}; equally often: {eq}; less often: {len(c) - more - eq};"
              f" first row in which it is not more often: {fu(firstle)}; whole answer right never exceeds either byte: {yes(all(q['R'] <= min(q['b1'], q['b2']) for q in c))}")


# ---------------------------------------------------------------- the weights: sections 5, 6, 7, 12
WF = {}
if want & {"5", "6", "7", "12"}:
    cells42 = R42.Cells(SEED, N=10, pset="1")                     # E42's F10: the reader's own draw of the ten connections
    F_full = R49.Interface(SEED, D, U)                            # both masks full
    bit = lambda a, b: a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    TRAINED = ("rnn.weight_ih_l0", "rnn.weight_hh_l0", "rnn.bias_ih_l0", "rnn.bias_hh_l0", "lm.weight", "lm.bias", "l2.weight", "l2.bias", "l3.weight", "l3.bias")
    FIXED = ("code_table", "code_owner", "code_amplitude", "book.weight", "l1.weight", "l1.bias")
    STRUCT = ("mask", "made", "periods", "phases", "mask_in", "mask_out", "k_in", "k_out")
    for r in "SB":
        blob = show(f"{P}e50/e50_C-dense_{r}_{SEED}.pt", REF); blob0 = show(f"{P}e50/e50_C-dense_{r}_{SEED}_start.pt", REF)
        WF[r] = dict(sha=sha(blob), sha0=sha(blob0), size=len(blob), T=load(blob), S0=load(blob0))
        del blob, blob0

if "5" in want:
    head(5, "the weight files of the two runs")
    e42start = load(show(f"{P}e49_qual/check2_73129_runs/e49_F10_S_dry2048_{SEED}_start.pt", E49REF))
    code0 = R50.Code(SEED)
    print(f"   the digest of the code by the reviewer's reader (e50_code_reader.Code({SEED}).digest()): {code0.digest()} (reported 12111044f0f8bc70: {yes(code0.digest() == '12111044f0f8bc70')});"
          f" used bytes {len(code0.used)}, values with an owner {sum(o >= 0 for o in code0.owner)} of {D}")
    for r in "SB":
        w = WF[r]; T, S0 = w["T"], w["S0"]; rc = TAB[("C-dense", r)]["rc"] if TAB else receipt("C-dense", r)
        print(f"   C-dense {r}: final file {w['size']:,} bytes, SHA-256 {w['sha']}")
        print(f"      equal to the receipt's weights_sha256: {yes(w['sha'] == rc['weights_sha256'])}; to the report's: {yes(w['sha'] == REPORTED_SHA[r])}; the start file's SHA-256 equals the receipt's start_sha256: {yes(w['sha0'] == rc['start_sha256'])}")
        print(f"      tensors in the final file: {len(T)}, the same names as in the start file: {yes(set(T) == set(S0))}")
        print("      bit for bit between the start file and the final file: " + "; ".join(f"{k} {yes(bit(T[k], S0[k]))}" for k in FIXED))
        print("      the structure, bit for bit between the two files: " + "; ".join(f"{k} {yes(bit(T[k], S0[k]))}" for k in STRUCT))
        print("      the trained tensors, entries that differ between the two files: " + "; ".join(f"{k} {int((T[k] != S0[k]).sum()):,} of {T[k].size:,}" for k in TRAINED))
        amp = float(T["code_amplitude"]); code = R50.Code(SEED, amplitude=amp)
        z1 = np.tanh(S0["book.weight"][code.used].astype(np.float64) @ S0["l1.weight"].T.astype(np.float64) + S0["l1.bias"])
        msl = float((z1 ** 2).sum(1).mean())
        tb = T["code_table"]
        print(f"      code_table {tb.shape}: equal bit for bit to the reader's table with the file's value of a turned-on entry: {yes(bit(tb, code.table()))}; entries that are not zero {int((tb != 0).sum())} (46 x 11 = 506),"
              f" all equal to {amp:.6f}: {yes(bool((tb[tb != 0] == np.float32(amp)).all()))}; rows with a code {int((tb != 0).any(1).sum())}, each with {sorted(set((tb != 0).sum(1)[(tb != 0).any(1)].tolist()))} values; a value with two owners: {yes(bool(((tb != 0).sum(0) > 1).any())).replace('NO', 'none').replace('yes', 'SOME')}")
        print(f"      code_owner equals the reader's list of owners: {yes(T['code_owner'].tolist() == code.owner)}; its digest {R49.digest16(T['code_owner'].tolist())} (the receipt's {rc['code']['digest']})")
        print(f"      the value of a turned-on entry from the start file's byte table and l1 (the square root of one eleventh of the mean squared length of tanh(l1(byte table)) over the 46 used bytes): mean squared length {msl:.4f}, value {np.sqrt(msl / ON):.6f};"
              f" in the file {amp:.6f}, in the receipt {rc['code']['amplitude']:.6f}")
        m_in, m_out, m_rec = T["mask_in"], T["mask_out"], T["mask"]
        out_rec = np.abs(T["rnn.weight_hh_l0"])[~np.tile(m_rec, (3, 1))]
        print(f"      masks: the input map's is full ({int(m_in.sum()):,} of {m_in.size:,}): {yes(bool(m_in.all()))}; lm's is full ({int(m_out.sum()):,} of {m_out.size:,}): {yes(bool(m_out.all()))}; k_in {int(T['k_in'])}, k_out {int(T['k_out'])};"
              f" the recurrent mask equals the reader's draw of E42's ten connections and the cell itself: {yes(bit(m_rec, cells42.mask))} ({int(m_rec.sum(1).min())} to {int(m_rec.sum(1).max())} cells a cell)")
        print(f"      weights outside a mask in the final file: recurrent map {out_rec.size:,} entries outside, of which not zero {int((out_rec != 0).sum())}, largest magnitude {float(out_rec.max())};"
              f" input map and lm have no entry outside their (full) masks; periods all 1: {yes(bool((T['periods'] == 1).all()))}; in the start file, not zero outside the recurrent mask: {int((S0['rnn.weight_hh_l0'][~np.tile(m_rec, (3, 1))] != 0).sum())}")
        mod = [k for k in e42start if k in S0 and e42start[k].dtype == np.float32]
        print(f"      the start file against E42's start of F10 (the start file of the dry run of E49's check 2, {E49REF}): the thirteen module tensors bit for bit: "
              f"{yes(all(bit(S0[k], e42start[k]) for k in mod))} ({len(mod)} compared); mask, periods, phases: {yes(all(bit(S0[k], e42start[k]) for k in ('mask', 'periods', 'phases')))}")
        print(f"      the receipt: byte_table_and_l1_bit_equal_to_start {rc['byte_table_and_l1_bit_equal_to_start']}; code_table_sha256_equal_at_every_row {rc['code_table_sha256_equal_at_every_row']};"
              f" rows in which code table, byte table and l1 are recorded unchanged: {sum(all(q['fixed'].values()) for q in rc['curve'])} of {len(rc['curve'])};"
              f" rows with every mask held (weights outside zero, the optimizer's averages there zero): {sum(all(all(v.values()) for v in q['masks'].values()) for q in rc['curve'])}; with E42's limit held: {sum(all(q['limit'].values()) for q in rc['curve'])};"
              f" the optimizer holds state for: {', '.join(rc['optimizer_state_for'])}")
    print(f"   the two start files hold the same tensors bit for bit: {yes(all(bit(WF['S']['S0'][k], WF['B']['S0'][k]) for k in WF['S']['S0']))}")
    del e42start


# ---------------------------------------------------------------- 6: the scores from the saved weights
def panels():
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


class Net:
    """One run's final network, evaluated by e50_code_reader.forward (the reviewer's reader) and nothing else."""

    def __init__(self, T):
        self.W = {k: v for k, v in T.items() if v.dtype == np.float32 and v.ndim > 0 and k != "code_table"}
        self.code = R50.Code(SEED, amplitude=float(T["code_amplitude"]))
        self.mask = T["mask"]

    def last(self, x, n_last, chunk=100):
        """Logits at the last n_last positions, (B, n_last, 257)."""
        return np.concatenate([R50.forward(self.W, self.mask, F_full, self.code, x[s:s + chunk])[1][:, -n_last:] for s in range(0, len(x), chunk)])

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

    def recorded(self, items, rec):
        """How many recorded answers are what greedy generation gives: each recorded byte is the largest output after the
        prompt and the recorded bytes before it, and the answer ends at the end symbol or at four bytes (the cap of
        exp11_cell.respond)."""
        same = np.zeros(len(items), bool); by_len = {}
        for i, a in enumerate(rec):
            by_len.setdefault(len(a), []).append(i)
        for n, idx in by_len.items():
            x = np.array([[BOS] + items[i][0] + rec[i][:-1] for i in idx], dtype=np.int64); tgt = np.array([rec[i] for i in idx], dtype=np.int64)
            same[np.array(idx)] = (self.last(x, n).argmax(-1) == tgt).all(1)
        stops = all(EOS not in a[:-1] and (a[-1] == EOS or len(a) == 4) for a in rec)
        return int(same.sum()), stops, sum(a == t for a, (_, t) in zip(rec, items))


if "6" in want:
    head(6, "the recorded scores from the saved final weights, by the reviewer's forward pass (e50_code_reader.forward); reader / receipt's last row")
    AP, RP, TA, LP = panels()
    for r in "SB":
        t1 = time.time()
        T = WF[r]["T"]; net = Net(T); rc = receipt("C-dense", r); last = rc["curve"][-1]
        # the reader's two forms on the trained weights, 24 items
        x = np.array([[BOS] + p + a[:-1] for p, a in AP["add89"][:12] + AP["sub89"][:12]], dtype=np.int64)
        la = R50.forward(net.W, net.mask, F_full, net.code, x)[1]; lb = R50.forward_table(net.W, net.mask, F_full, net.code, x)[1]
        print(f"   C-dense {r}: the reader's two forms of the forward (by index lists; by a table and masked matrices) on 24 items of the panel behind A: largest difference of a logit {float(np.abs(la - lb).max()):.1e}"
              f" (largest logit {float(np.abs(la).max()):.1f}); same largest output at every position: {yes(bool((la.argmax(-1) == lb.argmax(-1)).all()))}")
        mine = dict(A={n: net.exact(it) for n, it in AP.items()}, R={n: net.exact(it) for n, it in RP.items()}, taught={n: net.exact(it) for n, it in TA.items()})
        theirs = dict(A=last["A_counts"], R=last["R_counts"], taught=last["taught_counts"])
        for k, nm in (("A", "the panel behind A (add, sub, max, min; 300 each)"), ("R", "R (add, sub; 1,000 each)"), ("taught", "the taught panel (add, sub, max, min; 300 each)")):
            print(f"      {nm}: {list(mine[k].values())} / {list(theirs[k].values())}: {yes(mine[k] == theirs[k])}")
        b = {n: net.bytes_apart(it) for n, it in RP.items()}
        b1 = {n: v[0] for n, v in b.items()}; b2 = {n: v[1] for n, v in b.items()}
        print(f"      R's items, first byte right {list(b1.values())} / {list(last['R_first_byte_counts'].values())}: {yes(b1 == last['R_first_byte_counts'])};"
              f" second byte right {list(b2.values())} / {list(last['R_second_byte_counts'].values())}: {yes(b2 == last['R_second_byte_counts'])}"
              f" (items whose first byte is wrong and whose second byte was generated after the byte given: {sum(v[3] for v in b.values())})")
        ans = json.loads(show(f"{P}e50/e50_C-dense_{r}_{SEED}_answers.json", REF))
        tot = 0
        for k in (12, 40):
            same, stops, right = net.recorded(LP[k], ans[str(k)])
            tot += same
            print(f"      the answers file, {k} letters: {len(ans[str(k)])} recorded answers; the reader's forward gives the recorded answer, byte by byte, in {same} of {len(LP[k])};"
                  f" recorded answers end at the end symbol or at four bytes: {yes(stops)}; right by the recorded answers {right} / receipt {last['length_counts'][str(k)]}: {yes(right == last['length_counts'][str(k)])}")
        print(f"      recorded answers that agree with the reader's: {tot} of 600   [{time.time() - t1:.0f} s]", flush=True)
        del net


# ---------------------------------------------------------------- 7: multiply-adds
if "7" in want:
    head(7, "multiply-adds for one item at one position, recounted from the masks in the final weight files and from the reader's code")
    for r in "SB":
        T = WF[r]["T"]; code = R50.Code(SEED, amplitude=float(T["code_amplitude"])); rc = TAB[("C-dense", r)]["rc"]
        n_in, n_out, n_rec = 3 * int(T["mask_in"].sum()), int(T["mask_out"].sum()), 3 * int(T["mask"].sum())
        l1, l2, l3 = T["l1.weight"].size, T["l2.weight"].size, T["l3.weight"].size
        first = n_in + n_out + n_rec + l2 + l3
        per_byte = [3 * int(T["mask_in"][:, code.own[b]].sum()) for b in code.used]
        l2_on = ON * T["l2.weight"].shape[0]
        second = sum(per_byte) / len(per_byte) + l2_on + n_out + n_rec + l3
        print(f"   C-dense {r}: first count (every weight of a map inside its mask): input map {n_in:,} + lm {n_out:,} + cells {n_rec:,} + l2 {l2:,} + l3 {l3:,} = {first:,}"
              f" (reported 926,464: {yes(first == 926464)}; receipt {rc['multiply_adds_per_position']:,}); with l1's {l1:,}, which F10 computes: {first + l1:,} (reported for F10 930,560: {yes(first + l1 == 930560)}); ratio {first / (first + l1):.4f}")
        print(f"      second count (products with a value of the code that is not zero): input map {min(per_byte):,} to {max(per_byte):,} for a used byte (three gates x 256 cells x 11 values) + l2 {l2_on:,} + lm {n_out:,} + cells {n_rec:,} + l3 {l3:,} = {second:,.0f}"
              f" (reported 285,184: {yes(second == 285184)}; receipt {rc['multiply_adds_per_position_second_count']:,.0f})")
    print("   to the 80 threshold (count a position x 64 items x updates), three-row version:")
    for k in ORDER:
        t = TAB[k]; s = k[0]
        print(f"   {s:7s} {k[1]}  first count: A {MA[s] * BATCH * t['A80']:.3g}, R {MA[s] * BATCH * t['R80']:.3g}" + (f" | second count: A {MA2[s] * BATCH * t['A80']:.3g}, R {MA2[s] * BATCH * t['R80']:.3g}" if s in MA2 else ""))
    print("   (the report: by the second count C-dense reaches 80 on R after 1.26e+11 products under backpropagation and 2.34e+11 under the sweep)")
    print("   F10's code is tanh(l1(byte table)); if none of its values is zero, the products with a value that is not zero are all of F10's 930,560. From E42's final weight files:")
    used = R50.Code(SEED).used
    for r in "BS":
        Tf = load(show(f"{P}e42/e42_F10_{r}_{SEED}.pt", E42REF))
        z = np.tanh(Tf["book.weight"][used] @ Tf["l1.weight"].T + Tf["l1.bias"])
        print(f"   F10 {r}: values of the codes of the 46 used bytes that are exactly zero: {int((z == 0).sum())} of {z.size:,}; smallest magnitude {float(np.abs(z).min()):.1e}; of magnitude below 0.01: {int((np.abs(z) < 0.01).sum()):,}")
        del Tf
    print("   C-dense's second count against F10's 930,560:")
    for r in "BS":
        c, f = TAB[("C-dense", r)], TAB[("F10", r)]
        print(f"   {r}: a position {MA2['C-dense'] / MA['F10']:.3f} of F10's; to 80 on A {MA2['C-dense'] * c['A80'] / (MA['F10'] * f['A80']):.2f} of F10's products, to 80 on R {MA2['C-dense'] * c['R80'] / (MA['F10'] * f['R80']):.2f}")


# ---------------------------------------------------------------- 8: the stream of training items
def stream_digests(seed, n_batches, every=256):
    """exp26_candidate.Digest rebuilt: a running SHA-256 over every batch of 64 items of exp11_cell.stream (Random(seed + 77),
    the taught halves of the four tasks, 0 to 12 letters), as the three arrays of exp45.batch_tensor (x, y and the answer
    mask), each preceded by the text of its shape.  Returns the digest after every `every` batches."""
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


if "8" in want:
    head(8, "the stream of training items: the running digest of the batches, rebuilt here from np_gru.py for seed 73129, beside the receipts")
    t1 = time.time()
    mine = stream_digests(SEED, DOSE)
    print(f"   rebuilt: {DOSE:,} batches of {BATCH} items; digest after 4,096 batches {mine[4096][:16]} (reported bbd71ab7ca27b72e: {yes(mine[4096][:16] == 'bbd71ab7ca27b72e')}); after {DOSE:,} batches {mine[DOSE][:16]}   [{time.time() - t1:.0f} s]")
    recs = {}
    for k in ORDER:
        rc = TAB[k]["rc"]; recs[k] = rc
        rows = {q["step"]: q["data_digest"] for q in rc["curve"]}
        eq = sum(rows[u]["sha256"] == mine[u] for u in rows); cnt = all(rows[u]["batches"] == u for u in rows)
        print(f"   {k[0]:7s} {k[1]}  rows whose recorded digest equals the rebuilt one: {eq} of {len(rows)} (the number of batches recorded in each row equals its update: {yes(cnt)});"
              f" at 4,096 {rows[4096]['sha256'][:16]}; the receipt's final digest equals the rebuilt one: {yes(rc['data_digest']['sha256'] == mine[DOSE])}; optimizer steps {rc['optimizer_steps']}")
    for r in "BS":
        a, b = recs[("C-dense", r)], recs[("F10", r)]
        print(f"   {r}: C-dense's receipt against F10's: the same digest at every one of the 97 rows: {yes(all(p['data_digest'] == q['data_digest'] for p, q in zip(a['curve'], b['curve'])))};"
              f" the same optimizer record (AdamW, rate {a['optimizer']['lr']}, decay {a['optimizer']['weight_decay']}, betas {a['optimizer']['betas']}, the same two groups): {yes(a['optimizer'] == b['optimizer'])};"
              f" batch {a['batch']} and {b['batch']}, letters up to {a['kmax']} and {b['kmax']}, dose {a['dose']:,} and {b['dose']:,}; the same text for the data: {yes(a['data'] == b['data'])}")
    print("   (the scoring panels are also E42's: section 6 reproduces every count of the last row on the panels that np_gru.py builds, which are those of the E42 and E49 verifications)")


# ---------------------------------------------------------------- 9: the predictions and the checks before the runs
if "9" in want:
    head(9, "the four predictions, scored from the receipts, and the receipts of the checks before the runs (exp45/e50_qual)")
    q = P + "e50_qual/"
    c1, c3, c4, c5, cf = (json.loads(show(q + f"check{n}_{SEED}.json", REF)) for n in ("1", "3", "4", "5", "figures"))
    spec = show("PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e50_sparse_code_spec.py", REF); copy = show(M + "e50_sparse_code.py", REF)
    a, b = spec.decode().splitlines(), copy.decode().splitlines()
    diff = [(i + 1, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    print(f"   check 1 (the worker's record: pass {c1['pass']}, {c1['lines_compared']} lines of the self-checks' output equal {c1['lines_equal']}); read here: the runner's copy has {len(b)} lines, the specification file {len(a)};"
          f" lines that differ: {[d[0] for d in diff]}, both import lines: {yes(len(diff) == 2 and all(d[1].startswith('import ') and d[2].startswith('import ') for d in diff))};"
          f" the specification file at {REF} is the default branch's: {yes(git('rev-parse', REF + ':PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e50_sparse_code_spec.py') == git('rev-parse', DEFAULT + ':PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e50_sparse_code_spec.py'))};"
          f" recorded SHA-256 of the two files equal those computed here: {yes(c1['spec_sha256'] == sha(spec) and c1['copy_sha256'] == sha(copy))}")
    line1 = c1["stdout"][0]
    print(f"      the first line of the self-checks' output names the digest {re.search(r'digest of the code ([0-9a-f]+)', line1).group(1)} and the value {re.search(r'amplitude ([0-9.]+)', line1).group(1)}")
    for r in "BS":
        rc = TAB[("C-dense", r)]["rc"]
        print(f"   check 2 (the code is fixed through a run), C-dense {r}: rows with the code table's SHA-256 equal to the start's {sum(x['fixed']['code_table_sha256_equal'] for x in rc['curve'])} of {len(rc['curve'])} (the worker's record; section 5 compares the start and the final file bit for bit)")
    print(f"      in the six dry runs: {sum(v['code_fixed_every_row'] for v in c5['runs'].values())} of {len(c5['runs'])} hold at every row")
    print(f"   check 3 (the worker's record: pass {c3['pass']}): E49's check 4, digest at 4,096 {c3['e49_check4']['digest_4096'][:16]} (rebuilt in section 8); E49's check 5, largest difference of a count on E26's six births {c3['e49_check5']['max_abs_diff']};"
          f" the masks at every row of the dry runs: {sum(v['masks_every_row'] for v in c5['runs'].values())} of {len(c5['runs'])}")
    bc = c4["the_briefs_check"]
    worst = max(max(v[l]["logits_max_abs_diff"].values()) for v in c4["settings"].values() for l in v)
    print(f"   check 4 (the worker's record: pass {c4['pass']}): {bc['setting']} under {bc['rule']}, largest difference of a logit between the two forms {bc['logits_max_abs_diff']:.1e}, tolerance {c4['tolerance']}; the largest over the three settings, at the start and with the weights moved: {worst:.1e}")
    print(f"   check 5 (the worker's record: pass {c5['pass']}): {len(c5['runs'])} dry runs; the scorer equals E42's in {sum(v['scorer_equals_e42'] for v in c5['runs'].values())}; forecast {sum(v['forecast_cpu_h'] for v in c5['runs'].values()):.2f} CPU-hours for the six runs;"
          f" the reviewer's figures (check `figures`): pass {cf['pass']}")
    Bt, St, Fb = TAB[("C-dense", "B")], TAB[("C-dense", "S")], TAB[("F10", "B")]
    d8 = describe(Fb["R80"], Bt["R80"], DOSE - 512, DOSE - 512)
    print("   the predictions (the coordinator's; chances from the brief, revision 2):")
    print(f"      P50.1  checks 1 to 4 pass (85%): {'true' if all(c['pass'] for c in (c1, c3, c4)) and all(all(x['fixed'].values()) for r in 'BS' for x in TAB[('C-dense', r)]['rc']['curve']) else 'false'} (worker: true)")
    print(f"      P50.2  C-dense, backpropagation, reaches 80 on R within the dose (80%): {'true' if Bt['R80'] else 'false'} ({fu(Bt['R80'])} updates; worker: true, 6,912)")
    print(f"      P50.3  C-dense, the sweep, reaches 80 on R within the dose (55%): {'true' if St['R80'] else 'false'} ({fu(St['R80'])} updates; worker: true, 12,800)")
    print(f"      P50.8  C-dense against F10, backpropagation, updates to 80 on R *about the same* or *fewer* (60%): {'true' if d8 in ('about the same', 'fewer') else 'false'} (*{d8}*, ratio {Bt['R80'] / Fb['R80']:.2f}; worker: true, *about the same*, 0.73)")


# ---------------------------------------------------------------- 10: stage 2
if "10" in want:
    head(10, "stage 2: the brief's rule, the launch script, the logs present at " + REF)
    brief = show("OPUS_TASK_E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE_20261116.md", BRIEF).decode()
    row = next(l for l in brief.splitlines() if l.startswith("| 2 | `C16-in`"))
    print(f"   the brief's row for stage 2: {row}")
    for r in "BS":
        t = TAB[("C-dense", r)]
        print(f"   {RULE[r]}: C-dense reaches 80 on R within its dose: {yes(bool(t['R80']))} ({fu(t['R80'])} updates; the run is complete: {t['complete']}) -> stage 2 runs under this rule: {yes(bool(t['R80']))}")
    Bt, St = TAB[("C-dense", "B")], TAB[("C-dense", "S")]
    f49 = TAB[("in16", "B")]["R80"] / TAB[("F10", "B")]["R80"]
    print(f"   what can be recorded in stage 2 (thresholds up to update {49152 - 512:,}): a factor of up to {(49152 - 512) / Bt['R80']:.2f} (backpropagation) and {(49152 - 512) / St['R80']:.2f} (the sweep) against C-dense (reported 7.0 and 3.8);"
          f" E49's factor for in16 against F10 under backpropagation is {TAB[('in16', 'B')]['R80']:,} / {TAB[('F10', 'B')]['R80']:,} = {f49:.4f}; times C-dense's {Bt['R80']:,} that is update {f49 * Bt['R80']:,.0f} ({2.16 * Bt['R80']:,.0f} with the rounded 2.16; reported 14,930), within the first dose's {DOSE - 512:,}: {yes(f49 * Bt['R80'] <= DOSE - 512)};"
          f" times the sweep's {St['R80']:,} it is {f49 * St['R80']:,.0f}")
    sh = show(f"{P}e50/launch_e50.sh", REF).decode()
    print(f"   launch_e50.sh: one process a job, started by `setsid nohup`: {yes('setsid nohup' in sh)}; one thread (OMP_NUM_THREADS=1 MKL_NUM_THREADS=1): {yes('OMP_NUM_THREADS=1 MKL_NUM_THREADS=1' in sh)};"
          f" the command is `exp50_code.py run <setting> <rule> 73129` with no option for the dose: {yes(bool(re.search(r'exp50_code.py run .\$s. .\$r. 73129 >', sh)) and '--dose' not in sh)}; a log that exists is refused: {yes('log exists, refused' in sh)}")
    src = show(M + "exp50_code.py", REF).decode()
    print(f"   the runner at {REF} (unchanged since {RUNNER}, section 0): {re.search(r'^SETTINGS = .*$', src, re.M).group(0)} ; {re.search(r'^DOSES = .*$', src, re.M).group(0)}")
    print("   logs/queue.log:")
    for l in show(f"{P}e50/logs/queue.log", REF).decode().splitlines():
        print(f"      {l}")
    for r in "SB":
        lg = show(f"{P}e50/logs/e50_C16-in_{r}_{SEED}.log", REF).decode().splitlines()
        print(f"   logs/e50_C16-in_{r}_{SEED}.log ({len(lg)} lines): {lg[0]} | {lg[-1]}")
    files = git("ls-tree", "-r", "--name-only", REF, "--", P + "e50/").splitlines()
    print(f"   files of stage 2 at {REF} other than logs: {[f.split('/')[-1] for f in files if 'C16-in' in f and '/logs/' not in f] or 'none (the receipt is first written at update 1,024)'}")
    print(f"   commits: " + "; ".join(git("log", "-1", "--format=%h %cd", "--date=format:%H:%M:%SZ", c) + f" ({w})" for c, w in (("c85d9af3", "the stage report"), ("f54c6db8", "the report amended"), ("b18e563b", "stage 2 launched, logs"))))
    # the setting as this runner builds it: the dry run of check 5 (the same code path and the same seed as the run of stage 2)
    F16 = R49.Interface(SEED, 16, U); code = R50.Code(SEED)
    for r in "SB":
        Td = load(show(f"{P}e50_qual/check5_73129_runs/e50_C16-in_{r}_dry256_{SEED}_start.pt", REF)); rcd = json.loads(show(f"{P}e50_qual/check5_73129_runs/e50_C16-in_{r}_dry256_{SEED}.json", REF))
        base = WF[r]["S0"] if WF else load(show(f"{P}e50/e50_C-dense_{r}_{SEED}_start.pt", REF))
        exp_ih = (base["rnn.weight_ih_l0"] * np.tile(F16.mask_in, (3, 1)) * np.float32((D / 16) ** 0.5)).astype(np.float32)
        rest = [k for k in base if k not in ("rnn.weight_ih_l0", "mask_in", "k_in")]
        kn = R50.knows(code, F16, R50.NEEDED); reach_ = (code.reads(F16)[:, [code.used.index(b) for b in R50.NEEDED]] > 0).sum(0)
        print(f"   C16-in as built for the dry run of check 5 ({r}): mask_in equals the reader's draw of E49's lists for 16 values a cell: {yes(np.array_equal(Td['mask_in'], F16.mask_in))} (digest {R49.digest16(F16.in_src)}, receipt {rcd['draw_digests']['input']});"
              f" lm's mask full: {yes(bool(Td['mask_out'].all()))}; k_in {int(Td['k_in'])}, k_out {int(Td['k_out'])}; the input map is C-dense's start times the mask times the square root of 32, largest difference {float(np.abs(Td['rnn.weight_ih_l0'] - exp_ih).max()):.1e};"
              f" every other tensor equal to C-dense's start: {yes(all(np.array_equal(Td[k], base[k]) for k in rest))}; needed bytes a cell has input for {int(kn.min())} to {int(kn.max())}, cells a needed byte reaches {int(reach_.min())} to {int(reach_.max())}; dose of the dry run {rcd['dose']}")
        del Td


# ---------------------------------------------------------------- 11: what a seed alone changes
if "11" in want:
    head(11, "what a seed alone changes: E42's F10 on three seeds (E42's branch at " + E42REF + "), beside C-dense minus F10 of seed 73129")
    for r in "BS":
        rows = {seed: summary("F10", r, seed) for seed in SEEDS}
        for seed, t in rows.items():
            print(f"   F10 {r} {seed}: A80 {fu(t['A80'])} [{fu(t['A80s'])}], R80 {fu(t['R80'])} [{fu(t['R80s'])}]; levels A {t['lv_A']:.2f}, R {t['lv_R']:.2f}, 40 letters {t['lv_len40']:.2f}; complete {t['complete']}, rows {t['rows']}")
        c = TAB[("C-dense", r)]; ref = rows[SEED]
        for key in ("A80", "R80"):
            v = [rows[s][key] for s in SEEDS]
            where = "below the least" if c[key] < min(v) else "above the largest" if c[key] > max(v) else "inside the range"
            print(f"   {r} {key}: F10's three seeds {min(v):,} to {max(v):,} (range {max(v) - min(v):,} updates, largest over least {max(v) / min(v):.2f}; least over F10 of 73129 {min(v) / ref[key]:.2f}) | C-dense {c[key]:,}: minus F10 of 73129 {c[key] - ref[key]:+,} (ratio {c[key] / ref[key]:.2f});"
                  f" {where} of F10's three seeds; against the least of them {c[key] - min(v):+,} (ratio {c[key] / min(v):.2f}), in scored rows {(c[key] - min(v)) // 256:+d}")
        for key, nm in (("lv_A", "level of A"), ("lv_R", "level of R"), ("lv_len40", "level at 40 letters")):
            v = [rows[s][key] for s in SEEDS]
            where = "below the least" if c[key] < min(v) else "above the largest" if c[key] > max(v) else "inside the range"
            print(f"   {r} {nm}: F10's three seeds {min(v):.2f} to {max(v):.2f} (range {max(v) - min(v):.2f}) | C-dense {c[key]:.2f}: minus F10 of 73129 {c[key] - ref[key]:+.2f}; {where} of F10's three seeds" + (f" by {c[key] - max(v):+.2f}" if c[key] > max(v) else ""))


# ---------------------------------------------------------------- 12: other figures of the report, the tables file, the summary file
if "12" in want:
    head(12, "other figures of the report")
    code = R50.Code(SEED); F16 = R49.Interface(SEED, 16, U)
    kd = R50.knows(code, F16, R50.DIGITS)
    groups = [np.nonzero(kd <= 2)[0], np.nonzero((kd >= 3) & (kd <= 4))[0], np.nonzero(kd >= 5)[0]]
    print(f"   cells with input for 0 to 2, 3 or 4, 5 or more of the ten digits under 16 values a cell (the reader's draw): {[len(g) for g in groups]} (reported 95, 131, 30)")
    eye = np.eye(U, dtype=bool)

    def figures(T):
        amp = np.float64(T["code_amplitude"]); Wi = T["rnn.weight_ih_l0"].astype(np.float64)
        gi = np.stack([amp * Wi[:, code.own[b]].sum(1) for b in R50.NEEDED])                 # (20, 768): a cell's input for a needed byte, without the bias
        m3 = np.tile(T["mask"], (3, 1)); Wh = np.abs(T["rnn.weight_hh_l0"].astype(np.float64))
        kn = []
        for g in groups:
            rows_ = np.concatenate([g, g + U, g + 2 * U]); nb = np.tile((T["mask"] & ~eye)[g], (3, 1)); ws = np.tile(T["mask"][g], (3, 1))
            kn.append((float(Wh[rows_][nb].mean()), float(Wh[rows_][ws].mean())))
        return dict(rms=float(np.sqrt((gi ** 2).mean())), ih=float(np.abs(Wi).mean()), lm=float(np.abs(T["lm.weight"].astype(np.float64)).mean()), hh=float(Wh[m3].mean()), kn=kn)

    wrows = []                                             # two rows of the tables file written from the weight files
    for r in "BS":
        a, b = figures(WF[r]["S0"]), figures(WF[r]["T"]); rc = TAB[("C-dense", r)]["rc"]; c0, c1 = rc["curve"][0], rc["curve"][-1]
        cs_, ce_ = rc["cosine_first_batch_start"], rc["cosine_first_batch_end"]
        wrows.append(("sizes from the weight files (cosines from the receipt)", f"| C-dense | {r} | " + " | ".join(f"{cs_[m]:.3f}, {ce_[m]:.3f}" for m in ("rnn.weight_ih_l0", "lm.weight", "rnn.weight_hh_l0"))
                      + " | " + " | ".join(f"{a[k]:.4f}, {b[k]:.4f}" for k in ("ih", "lm", "hh")) + f" | {a['rms']:.3f}, {b['rms']:.3f} |"))
        wrows.append(("by what a cell knows, from the weight files", f"| C-dense | {r} | " + " | ".join(f"{a['kn'][i][0]:.4f}, {b['kn'][i][0]:.4f} [{a['kn'][i][1]:.4f}, {b['kn'][i][1]:.4f}]" for i in range(3)) + " |"))
        print(f"   C-dense {r}, from the start and the final weight file / the receipt's first and last row:")
        print(f"      a cell's input for a needed byte, root mean square without the bias: {a['rms']:.3f} -> {b['rms']:.3f} / {c0['input_size']['needed_rms']:.3f} -> {c1['input_size']['needed_rms']:.3f} (reported 0.389 -> {'5.14' if r == 'B' else '4.77'})")
        print(f"      mean magnitude of a kept weight: input map {a['ih']:.4f} -> {b['ih']:.4f} / {c0['kept_mean_magnitude']['rnn.weight_ih_l0']:.4f} -> {c1['kept_mean_magnitude']['rnn.weight_ih_l0']:.4f} (reported 0.031 -> {'0.094' if r == 'B' else '0.086'});"
              f" lm {a['lm']:.4f} -> {b['lm']:.4f} / {c0['kept_mean_magnitude']['lm.weight']:.4f} -> {c1['kept_mean_magnitude']['lm.weight']:.4f}; recurrent {a['hh']:.4f} -> {b['hh']:.4f} / {c0['kept_mean_magnitude']['rnn.weight_hh_l0']:.4f} -> {c1['kept_mean_magnitude']['rnn.weight_hh_l0']:.4f}")
        names = ("digits_0_to_2", "digits_3_to_4", "digits_5_or_more")
        print("      by what a cell knows, kept recurrent weights from neighbours [with the cell's reading of itself]: " + "; ".join(
            f"{a['kn'][i][0]:.4f} -> {b['kn'][i][0]:.4f} [{a['kn'][i][1]:.4f} -> {b['kn'][i][1]:.4f}] / {c0['by_what_a_cell_knows'][n]['neighbours_only']:.4f} -> {c1['by_what_a_cell_knows'][n]['neighbours_only']:.4f}" for i, n in enumerate(names)))
        cs, ce = rc["cosine_first_batch_start"], rc["cosine_first_batch_end"]
        print(f"      cosine between the sweep's and the exact gradient on the first batch (the receipt only; not recomputed): input map {cs['rnn.weight_ih_l0']:.3f} -> {ce['rnn.weight_ih_l0']:.3f}; lm {cs['lm.weight']:.3f} -> {ce['lm.weight']:.3f}; recurrent {cs['rnn.weight_hh_l0']:.3f} -> {ce['rnn.weight_hh_l0']:.3f};"
              f" first-batch loss at the start {rc['first_batch_loss_start']}")
        print(f"      cost: {rc['cpu_updates'] / 3600:.2f} CPU-hours of updates + {rc['cpu_scoring'] / 3600:.2f} of scoring = {(rc['cpu_updates'] + rc['cpu_scoring']) / 3600:.2f} (reported {'0.87' if r == 'B' else '1.00'}); wall {rc['wall'] / 60:.1f} minutes; processes beside: {rc['processes_beside']}")
        lg = show(f"{P}e50/logs/e50_C-dense_{r}_{SEED}.log", REF).decode().splitlines()
        steps = [int(l.split()[4].rstrip(":")) for l in lg if re.match(rf"e50 C-dense {r} {SEED} \d+:", l)]
        print(f"      the run's log: {len(lg)} lines; scored rows in it {len(steps)}, in order from 0 to {steps[-1]:,} without a repeat: {yes(steps == list(range(0, DOSE + 1, 256)))}; last line: {lg[-1]}")

    # the tables file, row by row, in the worker's formats, from the figures recomputed here
    md = show(f"{P}e50/E50_TABLES_{SEED}_stage1.md", REF).decode()
    keys = [(s, r) for r in "BS" for s in ("F10", "in16", "C-dense")]
    exp = []
    for k in keys:
        t = TAB[k]; s, r = k; c = t["curve"]; rc = t["rc"]; by = {q["step"]: q for q in c}; pre = f"| {s} | {r} | "
        mx = lambda key: max(c, key=lambda q: q[key])
        exp.append(("thresholds", pre + " | ".join(f"{fu(t[f'{key}{th}'])} [{fu(t[f'{key}{th}s'])}]" for key in "AR" for th in (50, 80, 90))
                    + f" | {mx('A')['A']:.2f} ({mx('A')['step']:,}) | {mx('R')['R']:.2f} ({mx('R')['step']:,}) | {rc['curve'][-1]['cpu_updates'] / 3600:.2f} + {rc['curve'][-1]['cpu_scoring'] / 3600:.2f} |"))
        exp.append(("four operations", pre + " | ".join(f"{fu(t[o + '80'])} [{fu(t[o + '80s'])}]" for o in OPS) + " | " + ", ".join(f"{t['lv_' + o]:.1f}" for o in OPS) + " |"))
        g3 = lambda key, m: "not reached" if t[key] is None else f"{m * BATCH * t[key]:.3g}"
        exp.append(("multiply-adds", pre + f"{MA[s]:,} | {g3('A80', MA[s])} | {g3('R80', MA[s])} | " + (f"{MA2[s]:,} | {g3('A80', MA2[s])} | {g3('R80', MA2[s])} |" if s in MA2 else " |  |  |")))
        byt = f"{t['lv_b1']:.2f} | {t['lv_b2']:.2f}" if "lv_b1" in t else "(not recorded) | (not recorded)"
        exp.append(("levels", pre + f"{t['lv_A']:.2f} | {t['lv_R']:.2f} | {t['lv_len40']:.2f} | {t['lv_len12']:.2f} | {t['lv_taught']:.2f} | {byt} | {t['rise_R']:+.2f}{', still rising' if t['rise_R'] >= 5 and t['R80'] is None else ''} |"))
        cell = lambda q: f"{q['A']:.1f}, {q['R']:.1f}" + (f" ({q['b1']:.0f}, {q['b2']:.0f})" if "b1" in q else "")
        exp.append(("A and R every 2,048", pre + " | ".join(cell(by[u]) for u in range(2048, DOSE + 1, 2048)) + " |"))
        exp.append(("operations every 4,096", pre + " | ".join(", ".join(f"{by[u][o]:.0f}" for o in OPS) for u in range(4096, DOSE + 1, 4096)) + " |"))
        exp.append(("training loss", pre + " | ".join(f"{by[u]['loss']:.3f}" for u in (256, 2048, 4096, 8192, 12288, 16384, 20480, 24576)) + " |"))
        d4 = next(q["data_digest"]["sha256"] for q in rc["curve"] if q["step"] == 4096)[:16]
        lim = all(all(q["limit"].values()) for q in rc["curve"])
        mk = all(all(all(v.values()) for v in q["masks"].values()) for q in rc["curve"]) if "masks" in rc["curve"][0] else "(E42 receipt)"
        fx = all(all(q["fixed"].values()) for q in rc["curve"]) if "fixed" in rc["curve"][0] else "(learned code)"
        exp.append(("checks inside the runs", pre + f"{d4} | {mk} | {lim} | {fx} |"))
    for r in "BS":
        for cand in ("C-dense", "in16"):
            c, f = TAB[(cand, r)], TAB[("F10", r)]; pre = f"| {cand} | F10 | {r} | "
            for key in ("A80", "R80"):
                d3 = describe(f[key], c[key], DOSE - 512, DOSE - 512); d1 = describe(f[key + "s"], c[key + "s"], DOSE, DOSE)
                exp.append(("descriptions", pre + f"{key} updates | {d3}{'*' if d1 != d3 else ''} | {fu(f[key])} | {fu(c[key])} | {'' if c[key] is None else f'{c[key] / f[key]:.2f}'} |"))
                xr = f[key] * MA["F10"] * BATCH; xc = None if c[key] is None else c[key] * MA[cand] * BATCH
                xrs = f[key + "s"] * MA["F10"] * BATCH; xcs = None if c[key + "s"] is None else c[key + "s"] * MA[cand] * BATCH
                m3 = IN_MA[describe(xr, xc, (DOSE - 512) * MA["F10"] * BATCH, (DOSE - 512) * MA[cand] * BATCH)]; m1 = IN_MA[describe(xrs, xcs, DOSE * MA["F10"] * BATCH, DOSE * MA[cand] * BATCH)]
                exp.append(("descriptions", pre + f"{key} multiply-adds | {m3}{'*' if m1 != m3 else ''} | {xr:.3g} | {'not reached' if xc is None else f'{xc:.3g}'} | {'' if xc is None else f'{xc / xr:.2f}'} |"))
            for key, nm in (("A", "A"), ("R", "R"), ("len40", "the length test at 40")):
                d = c["lv_" + key] - f["lv_" + key]
                exp.append(("descriptions", pre + f"level of {nm} | {'lower' if d <= -15 else 'higher' if d >= 15 else 'about the same'} | {f['lv_' + key]:.2f} | {c['lv_' + key]:.2f} | |"))
    exp += wrows
    lines = set(md.splitlines())
    missing = [(w, l) for w, l in exp if l not in lines]
    kinds = sorted({w for w, _ in exp}, key=[w for w, _ in exp].index)
    print(f"   the tables file E50_TABLES_{SEED}_stage1.md: rows of C-dense, F10 and in16 written here from the recomputed figures, in the tables' own formats, and found in the file character for character:")
    for w in kinds:
        n = sum(1 for a, _ in exp if a == w); miss = sum(1 for a, _ in missing if a == w)
        print(f"      {w}: {n - miss} of {n}")
    for w, l in missing:
        print(f"      NOT FOUND ({w}): {l}")
    for r in "BS":
        print(f"      the file's rule for stage 2 under {r}: " + next((l for l in md.splitlines() if l.startswith(f"- {r}, stage 2")), "NOT FOUND"))
    # the summary file
    sm = json.loads(show(f"{P}e50/E50_SUMMARY_{SEED}_stage1.json", REF)); n_ok = n_all = 0; bad = []
    for k in keys:
        t, v = TAB[k], sm["runs"][f"{k[0]} {k[1]}"]
        for key in ("A", "R"):
            for th in (50, 80, 90):
                for ver, suf in (("three", ""), ("single", "s")):
                    w = v["thresholds"][f"{key}{th}_{ver}"]; n_all += 1
                    ok = (w["updates"] if w else None) == t[f"{key}{th}{suf}"]; n_ok += ok
                    if not ok:
                        bad.append((k, key, th, ver))
        for o in OPS:
            for ver, suf in (("three", ""), ("single", "s")):
                w = v["thresholds"][f"{o}80_{ver}"]; n_all += 1; ok = (w["updates"] if w else None) == t[f"{o}80{suf}"]; n_ok += ok
                if not ok:
                    bad.append((k, o, 80, ver))
        for key, mine in [(x, t["lv_" + x]) for x in ("A", "R", "len40", "len12", "taught") + OPS] + ([("R_first_byte", t["lv_b1"]), ("R_second_byte", t["lv_b2"])] if "lv_b1" in t else []):
            n_all += 1; ok = abs(v["levels"][key] - mine) < 1e-9; n_ok += ok
            if not ok:
                bad.append((k, "level", key))
        n_all += 1; ok = abs(v["R_last3_minus_2048_earlier"] - t["rise_R"]) < 1e-9; n_ok += ok
        if not ok:
            bad.append((k, "rise"))
    print(f"   the summary file E50_SUMMARY_{SEED}_stage1.json: thresholds (A and R at 50, 80, 90; the four operations at 80; both versions), levels and the change of R of the six runs equal to the figures recomputed here: {n_ok} of {n_all}" + (f"; differing: {bad}" if bad else ""))
    print(f"      its rule for stage 2: " + "; ".join(f"{r}: runs {sm['stages'][r]['stage2']['run']} ({sm['stages'][r]['stage2']['reason']})" for r in "BS") + f"; runs in progress when it was written: {sm['in_progress'] or 'none'}")

print(f"\nrunning time {time.time() - T0:.0f} s")
