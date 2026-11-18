"""Verification of stage 3 of E50 (the fixed input code with each cell reading one value of every used byte, `C46-one`,
46 of the 512 values, seed 73129, under the sweep and under backpropagation) and of the final report E50_REPORT.md; the
worker's branch of E50 at 36aa8771 (receipts, weights, answers, logs, tables) and at 1799b2ba (the three reports as amended).

Read-only.  Nothing is trained, torch is not used, and no code of another branch is run.  Every file of the worker's branch is
read with `git show` into memory.  A weight file is read into memory, or, when the environment variable E50S3V_SCRATCH names a
folder, written there, read and deleted at once.  No bytecode is written.

The method and most functions are those of e50_stage2_verify.py, imported here (the functions for the receipts, `summary`,
`reach`, `level`, `describe`, the class Net, `panels`, `forward_file`, `stream_digests`) and pointed at 36aa8771.  Importing
that script runs its part 1 on its own commit c6c24d9c with the output discarded (about two seconds); nothing of it is printed.
The forward pass is the reviewer's reader e50_code_reader.forward.  The 46 values of each cell of `C46-one` are drawn here by
`draw_one_runner`, written from the text of the worker's runner exp50_code.py (`draw_one`) and of the brief; part 2c compares
that draw with e50_code_reader.draw_one, with the digest that the brief and the receipts state, and with the masks in the six
weight files of stage 3.

The parts:

   0   what is read: the commits, the amendments of 1799b2ba, the runner's files, the receipts' provenance, logs, times, costs
   1   every figure of the final report's two result tables (all seven rows of each) and of the tables file for `C46-one`,
       recomputed from the receipts: updates to 50 and to 80 on A and on R (three-row and single-row version), the levels of
       A, R, the taught panel and the length test, the rise of R over the last 2,048 updates, max, min, add and sub at 12
       letters; both readings; the ratios and descriptions against `C-dense` and `C16-in`; the summary file
   2a  the recorded answers and counts from the saved weights: the final file and the file at update 24,576 of both runs
   2b  the code is fixed: code table, owners, value, byte table and `l1`, bit for bit, start = 24,576 = final; the start
       against the start of `C-dense`; the SHA-256 of the files
   2c  the masks: each cell reads exactly 46 values, one of every used byte; the draw; weights outside a mask are zero
   2d  "by what a cell knows" and "other measures" for `C46-one` from the weight files, and the tables file's rows of them
   3   the restart: each completed run against its interrupted predecessor (receipt rows, log rows, starting tensors), and
       the running digest of the batches rebuilt here from the seed beside the 193 rows of each completed run
   4   the figures behind the statements of the final report that concern all three stages, and the predictions
   5a  the use of the cells: standard deviation of each cell's state between the 300 length-test items at 12 letters at `=`
   5b  the answers given: first byte, second byte, whole answer right, number of different answers (R's items and the
       panel behind A)

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e50_stage3_verify.py [0 1 2a 2b 2c 2d 3 4 5a 5b]
Without arguments every part is printed.  The running time of each part is printed at its end.
"""
import sys
sys.dont_write_bytecode = True
import contextlib, hashlib, io, json, os, re, subprocess, time
from random import Random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PARTS = ("0", "1", "2a", "2b", "2c", "2d", "3", "4", "5a", "5b")
want = set(sys.argv[1:]) or set(PARTS)
assert want <= set(PARTS), f"parts are {PARTS}"
T0 = time.time()

# ---------------------------------------------------------------- the stage-2 script, imported and pointed at 36aa8771
_argv = sys.argv
sys.argv = [_argv[0], "1"]
with contextlib.redirect_stdout(io.StringIO()):
    import e50_stage2_verify as S2
sys.argv = _argv
import e42_cells_reader as R42
import e49_interface_reader as R49
import e50_code_reader as R50
from np_gru import BOS, EOS

REF = "36aa8771"                       # the worker's branch: stage 3, the final report, receipts, weights
AMEND = "1799b2ba"                     # the same branch with the three reports amended
STAGE2 = "c6c24d9c"                    # the commit read by the verification of stage 2
RUNNER = "0f3ec4bc"                    # the commit that the report names for the runner
BASE = "7cabb13a"                      # the default branch where the worker's branch starts
S2.REF = REF
S2.show.__defaults__ = (REF,)
S2.weights.__defaults__ = (REF,)
S2.SCRATCH = os.environ.get("E50S3V_SCRATCH")
S2._W.clear()
show, git, weights, wname, receipt, summary, describe = S2.show, S2.git, S2.weights, S2.wname, S2.receipt, S2.summary, S2.describe
sha, fu, yes, bit, head = S2.sha, S2.fu, S2.yes, S2.bit, S2.head
M, P, SEED, U, D, ON, BATCH, FIRST, DOSE, OPS = S2.M, S2.P, S2.SEED, S2.U, S2.D, S2.ON, S2.BATCH, S2.FIRST, S2.DOSE, S2.OPS
E = P + "e50/"
INT = E + "interrupted_20261117T2359Z/"
K = 46
MA = dict(S2.MA, **{"C46-one": 568576})
MA2 = dict(S2.MA2, **{"C46-one": 277504})
IN_MA = S2.IN_MA
r1 = lambda v: f"{v:.1f}"


# ---------------------------------------------------------------- the draw of C46-one, from the runner's text
def draw_one_runner(seed):
    """exp50_code.draw_one as its text and the brief state it: vals = range(512) shuffled by Random(f"e50|code|{seed}"); the
    used symbol number k owns vals[11k : 11k + 11]; for cells 0 to 255 in turn and, within a cell, for the 46 used symbols in
    the code's order (increasing byte, the start symbol last), the cell reads own[k][rng.randrange(11)] with
    rng = Random(f"e50|one|{seed}").  Each list is sorted."""
    vals = list(range(D))
    Random(f"e50|code|{seed}").shuffle(vals)
    used = sorted(list(b"0123456789+-><=") + list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list(b"pqvw")) + [BOS]
    own = {b: vals[ON * k: ON * k + ON] for k, b in enumerate(used)}
    rng = Random(f"e50|one|{seed}")
    return [sorted(own[b][rng.randrange(ON)] for b in used) for _ in range(U)]


class One(R49.Interface):
    """E49's interface object (both masks full) whose input lists are replaced by draw_one_runner's 46 values a cell."""

    def __init__(self, seed):
        super().__init__(seed, D, U)
        self.k_in = K
        self.in_src = draw_one_runner(seed)
        self.in_idx = np.array(self.in_src)
        self.mask_in = np.zeros((U, D), bool)
        for i, s in enumerate(self.in_src):
            self.mask_in[i, s] = True


NEED_NETS = want & {"2a", "2b", "2c", "2d", "5a", "5b"}
if NEED_NETS:
    S2.CELLS = R42.Cells(SEED, N=10, pset="1")
    S2.F = {"C46-one": One(SEED), "C16-in": R49.Interface(SEED, 16, U), "C-dense": R49.Interface(SEED, D, U)}
    S2.CODE0 = R50.Code(SEED)
Net, panels = S2.Net, S2.panels

# ---------------------------------------------------------------- the readings
READ = [("C-dense", "C-dense", None), ("C16-in @24,576", "C16-in", FIRST), ("C16-in @end", "C16-in", None),
        ("C46-one @24,576", "C46-one", FIRST), ("C46-one @end", "C46-one", None), ("in16", "in16", None), ("F10", "F10", None)]
TAB = {}
if want & {"0", "1", "3", "4"}:
    for r in "BS":
        for nm, s, upto in READ:
            TAB[(nm, r)] = summary(s, r, upto)

N_ = None
# the final report's two result tables (E50_REPORT.md at 36aa8771 and at 1799b2ba, section "Results"); None is "not reached"
REPORTED = {
    ("C-dense", "B"): dict(A80=7680, R80=6912, lv=(97.7, 97.1, 92.7), ops80=(1024, 1280, 7680, 7680), rise=1.1),
    ("C16-in @24,576", "B"): dict(A80=N_, R80=N_, lv=(67.9, 62.1, 41.6), ops80=(5376, 5632, N_, N_), rise=2.7),
    ("C16-in @end", "B"): dict(A80=40704, R80=N_, lv=(83.8, 79.4, 56.2), ops80=(5376, 5632, 34560, 40704), rise=2.6),
    ("C46-one @24,576", "B"): dict(A80=14080, R80=15360, lv=(88.5, 88.9, 38.2), ops80=(3584, 3584, 14592, 13824), rise=4.0),
    ("C46-one @end", "B"): dict(A80=14080, R80=15360, lv=(93.8, 94.2, 54.3), ops80=(3584, 3584, 14592, 13824), rise=0.0),
    ("in16", "B"): dict(A80=16640, R80=20480, lv=(85.2, 82.5, 66.1), ops80=(5120, 5376, 15616, 18432), rise=0.7),
    ("F10", "B"): dict(A80=8704, R80=9472, lv=(93.2, 94.1, 92.6), ops80=(2048, 2048, 8704, 9472), rise=2.1),
    ("C-dense", "S"): dict(A80=13056, R80=12800, lv=(93.5, 94.1, 90.6), ops80=(1280, 1536, 12032, 13824), rise=3.0),
    ("C16-in @24,576", "S"): dict(A80=N_, R80=N_, lv=(32.7, 29.0, 9.6), ops80=(7424, 7680, N_, N_), rise=1.7),
    ("C16-in @end", "S"): dict(A80=N_, R80=N_, lv=(49.5, 43.6, 13.6), ops80=(7424, 7680, N_, N_), rise=1.3),
    ("C46-one @24,576", "S"): dict(A80=N_, R80=N_, lv=(63.6, 65.5, 23.0), ops80=(5120, 5120, N_, N_), rise=5.6),
    ("C46-one @end", "S"): dict(A80=35584, R80=34304, lv=(85.9, 87.2, 27.8), ops80=(5120, 5120, 33280, 35584), rise=0.0),
    ("in16", "S"): dict(A80=N_, R80=N_, lv=(1.9, 1.4, 1.2), ops80=(N_, N_, N_, N_), rise=-0.4),
    ("F10", "S"): dict(A80=13312, R80=15104, lv=(91.2, 89.3, 89.1), ops80=(3072, 2816, 11776, 15104), rise=0.7),
}
ORDER = [(nm, r) for r in "BS" for nm, _, _ in READ]
C46 = [(nm, r) for r in "BS" for nm in ("C46-one @24,576", "C46-one @end")]
# the tables file's rows of C46-one for the thresholds at 50 (not in the report's table)
REPORTED_50 = {("C46-one @24,576", "B"): (7936, 7936), ("C46-one @end", "B"): (7936, 7936), ("C46-one @24,576", "S"): (15872, 14592), ("C46-one @end", "S"): (15872, 14592)}


# ---------------------------------------------------------------- 0: what is read
if "0" in want:
    head("0", "what is read")
    heads = [l.split() for l in git("for-each-ref", "--format=%(objectname) %(refname:short)", "refs/remotes/origin").splitlines() if "e50-fixed-input-code" in l]
    print(f"   remote branches of E50's worker in this clone: {len(heads)}; head {heads[0][0][:8] if heads else 'none'}; {REF} is an ancestor of the head: {yes(subprocess.run(['git', 'merge-base', '--is-ancestor', REF, heads[0][0]], cwd=S2.ROOT).returncode == 0)};"
          f" {AMEND} is the head: {yes(heads[0][0].startswith(AMEND))}; {REF} is the parent of {AMEND}: {yes(git('rev-parse', AMEND + '^').startswith(git('rev-parse', REF)))}")
    print(f"   {REF}: {git('log', '-1', '--format=%cd %s', '--date=format:%Y-%m-%d %H:%M:%SZ', REF)}")
    print(f"   {AMEND}: {git('log', '-1', '--format=%cd %s', '--date=format:%Y-%m-%d %H:%M:%SZ', AMEND)}")
    changed = git("diff", "--name-only", REF, AMEND).splitlines()
    print(f"   files changed between {REF} and {AMEND}: {len(changed)}: {', '.join(f.split('/')[-1] for f in changed)}; only report files (.md): {yes(all(f.endswith('_REPORT_73129.md') or f.endswith('E50_REPORT.md') for f in changed))}")
    for f in changed:
        st = git("diff", "--numstat", REF, AMEND, "--", f).split()
        txt = show(f, AMEND).decode()
        print(f"      {f.split('/')[-1]}: {st[0]} lines added, {st[1]} removed; a section 'Amendments' in it: {yes('## Amendments' in txt)}")
    brief = next(f for f in git("ls-tree", "--name-only", REF).splitlines() if "E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE" in f)
    print(f"   the brief's file is the same at {STAGE2}, {REF} and {AMEND}: {yes(git('rev-parse', f'{STAGE2}:{brief}') == git('rev-parse', f'{REF}:{brief}') == git('rev-parse', f'{AMEND}:{brief}'))}")
    for f in ("exp50_code.py", "e50_sparse_code.py", "e50_tables.py", "e49_interface.py", "exp49_interface.py"):
        print(f"   {f}: the same blob at {REF} as at {STAGE2}: {yes(git('rev-parse', f'{REF}:{M}{f}') == git('rev-parse', f'{STAGE2}:{M}{f}'))}" + (f"; as at {RUNNER}: {yes(git('rev-parse', f'{REF}:{M}{f}') == git('rev-parse', f'{RUNNER}:{M}{f}'))}" if f != "e50_tables.py" else ""))
    files = git("ls-tree", "-r", "--name-only", REF, "--", E).splitlines()
    old = [f for f in files if "C46-one" not in f and not f.split("/")[-1].startswith("E50_") and "launch" not in f and "queue" not in f]
    print(f"   files under exp45/e50 at {REF}: {len(files)}; of C46-one: {len([f for f in files if 'C46-one' in f])}; the {len(old)} files of stages 1 and 2 and the references (receipts, weights, answers, logs) unchanged since {STAGE2}: "
          f"{yes(all(git('rev-parse', f'{REF}:{f}') == git('rev-parse', f'{STAGE2}:{f}') for f in old))}")
    for r in "BS":
        t = TAB[("C46-one @end", r)]; rc = t["rc"]; pv = rc["provenance"]
        print(f"   receipt of C46-one {r}: command `{pv['command']}`; commit {pv['git']['commit'][:8]}, sources modified against it: {pv['git']['sources_modified_vs_commit']};"
              f" runner's SHA-256 equals exp50_code.py at {REF}: {yes(pv['runner_sha256'] == sha(show(M + 'exp50_code.py')))}; e50_sparse_code.py: {yes(pv['sources_sha256']['e50_sparse_code.py'] == sha(show(M + 'e50_sparse_code.py')))}")
        walls = [q["wall"] for q in rc["curve"]]; gaps = [b - a for a, b in zip(walls, walls[1:])]
        print(f"      complete {rc['complete']}, dose {rc['dose']:,}, {t['rows_whole']} scored rows (0 to {rc['curve'][-1]['step']:,}), optimizer steps {rc['optimizer_steps']}; shares equal counts in every row: {yes(t['shares_equal_counts'])};"
              f" k_in {rc['k_in']}, k_out {rc['k_out']}; processes beside: {rc['processes_beside']}; wall seconds between rows: least {min(gaps):.1f}, median {sorted(gaps)[len(gaps) // 2]:.1f}, largest {max(gaps):.1f}; wall {rc['wall'] / 3600:.3f} h; losses finite: {yes(all(q['finite_losses'] for q in rc['curve']))}")
        print(f"      cost: {rc['cpu_updates'] / 3600:.3f} + {rc['cpu_scoring'] / 3600:.3f} = {(rc['cpu_updates'] + rc['cpu_scoring']) / 3600:.3f} CPU-hours (reported {'1.90' if r == 'B' else '2.18'})")
        lg = show(f"{E}logs/e50_C46-one_{r}_{SEED}.log").decode().splitlines()
        steps = [int(l.split()[4].rstrip(":")) for l in lg if re.match(rf"e50 C46-one {r} {SEED} \d+:", l)]
        print(f"      the log: {len(lg)} lines; scored rows {len(steps)}, in order 0 to {steps[-1]:,} without a repeat: {yes(steps == list(range(0, DOSE + 1, 256)))}; lines before the first row that are not the run's: {sum(1 for l in lg[:lg.index(next(l for l in lg if l.startswith('e50 C46-one')))])};"
              f" last line: {lg[-1]}")
    print("   logs/queue.log:")
    for l in show(f"{E}logs/queue.log").decode().splitlines():
        print(f"      {l}")
    print(f"   launch_e50.sh at {REF} writes `exit=`: {yes('exit=' in show(E + 'launch_e50.sh').decode())}; commits on the branch from {BASE} to {AMEND} whose message has a Co-Authored-By line: "
          + ", ".join(c[:8] for c in git("log", "--format=%H", f"{BASE}..{AMEND}").split() if "co-authored-by" in git("log", "-1", "--format=%B", c).lower()) + f"; it changes {git('show', '--stat', '--format=', 'b1a516ca').splitlines()[0].split('|')[0].strip()}")
    for f in ("E50_REPORT.md", "E50_TABLES_73129.md", "E50_SUMMARY_73129.json", "E50_STAGE1_REPORT_73129.md", "E50_STAGE2_REPORT_73129.md"):
        print(f"   {f}: SHA-256 at {REF} {sha(show(E + f))[:16]}, at {AMEND} {sha(show(E + f, AMEND))[:16]}")


# ---------------------------------------------------------------- 1: the result tables
def table_rows(t_of, md_lines):
    """The tables file's rows of C46-one, written from the recomputed figures in the file's own formats."""
    exp = []
    for r in "BS":
        for nm in ("C46-one @24,576", "C46-one @end"):
            t = t_of[(nm, r)]; c = t["curve"]; rc = t["rc"]; pre = f"| {nm} | {r} | "; s = "C46-one"; raw = rc["curve"][t["rows"] - 1]
            mx = lambda key: max(c, key=lambda q: q[key])
            exp.append(("thresholds", pre + " | ".join(f"{fu(t[f'{key}{th}'])} [{fu(t[f'{key}{th}s'])}]" for key in "AR" for th in (50, 80, 90))
                        + f" | {mx('A')['A']:.2f} ({mx('A')['step']:,}) | {mx('R')['R']:.2f} ({mx('R')['step']:,}) | {raw['cpu_updates'] / 3600:.2f} + {raw['cpu_scoring'] / 3600:.2f} |"))
            exp.append(("four operations", pre + " | ".join(f"{fu(t[o + '80'])} [{fu(t[o + '80s'])}]" for o in OPS) + " | " + ", ".join(f"{t['lv_' + o]:.1f}" for o in OPS) + " |"))
            g3 = lambda key, m: "not reached" if t[key] is None else f"{m * BATCH * t[key]:.3g}"
            exp.append(("multiply-adds", pre + f"{MA[s]:,} | {g3('A80', MA[s])} | {g3('R80', MA[s])} | {MA2[s]:,} | {g3('A80', MA2[s])} | {g3('R80', MA2[s])} |"))
            exp.append(("levels", pre + f"{t['lv_A']:.2f} | {t['lv_R']:.2f} | {t['lv_len40']:.2f} | {t['lv_len12']:.2f} | {t['lv_taught']:.2f} | {t['lv_b1']:.2f} | {t['lv_b2']:.2f} | {t['rise_R']:+.2f}{', still rising' if t['rise_R'] >= 5 and t['R80'] is None else ''} |"))
            d4 = next(q["data_digest"]["sha256"] for q in rc["curve"] if q["step"] == 4096)[:16]; cv = rc["curve"][:t["rows"]]
            exp.append(("checks inside the runs", pre + f"{d4} | {all(all(all(v.values()) for v in q['masks'].values()) for q in cv)} | {all(all(q['limit'].values()) for q in cv)} | {all(all(q['fixed'].values()) for q in cv)} |"))
            refs = [("C-dense", t_of[("C-dense", r)])] + ([("C16-in @end", t_of[("C16-in @end", r)])] if nm.endswith("end") else [])
            for ref_nm, ref in refs:
                exp.extend(("descriptions", l) for l in description_rows(nm, t, ref_nm, ref, r))
        t = t_of[("C46-one @end", r)]; by = {q["step"]: q for q in t["curve"]}; pre = f"| C46-one | {r} | "
        exp.append(("A and R every 2,048", pre + " | ".join(f"{by[u]['A']:.1f}, {by[u]['R']:.1f} ({by[u]['b1']:.0f}, {by[u]['b2']:.0f})" for u in range(2048, DOSE + 1, 2048)) + " |"))
        exp.append(("operations every 4,096", pre + " | ".join(", ".join(f"{by[u][o]:.0f}" for o in OPS) for u in range(4096, DOSE + 1, 4096)) + " |"))
        exp.append(("training loss", pre + " | ".join(f"{by[u]['loss']:.3f}" for u in (256, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 32768, 40960, 49152)) + " |"))
    missing = [(w, l) for w, l in exp if l not in md_lines]
    return exp, missing


def description_rows(cand_nm, cand, ref_nm, ref, r):
    """The rows of the table "Descriptions fixed beforehand" for one candidate reading against one reference reading."""
    out = []; pre2 = f"| {cand_nm} | {ref_nm} | {r} | "
    sr, sc = ref["setting"], cand["setting"]; dr, dc = ref["dose"], cand["dose"]
    for key in ("A80", "R80"):
        d3 = describe(ref[key], cand[key], dr - 512, dc - 512); d1 = describe(ref[key + "s"], cand[key + "s"], dr, dc)
        ratio = "" if cand[key] is None or ref[key] is None else f"{cand[key] / ref[key]:.2f}"
        out.append(pre2 + f"{key} updates | {d3}{'*' if d1 != d3 else ''} | {fu(ref[key])} | {fu(cand[key])} | {ratio} |")
        for m, tail in ((MA, ""), (MA2, ", second count")):
            X = lambda t, k, s: None if t[k] is None else t[k] * m[s] * BATCH
            xr, xc, xrs, xcs = X(ref, key, sr), X(cand, key, sc), X(ref, key + "s", sr), X(cand, key + "s", sc)
            m3 = IN_MA[describe(xr, xc, (dr - 512) * m[sr] * BATCH, (dc - 512) * m[sc] * BATCH)]; m1 = IN_MA[describe(xrs, xcs, dr * m[sr] * BATCH, dc * m[sc] * BATCH)]
            g = lambda v: "not reached" if v is None else f"{v:.3g}"
            out.append(pre2 + f"{key} multiply-adds{tail} | {m3}{'*' if m1 != m3 else ''} | {g(xr)} | {g(xc)} | {'' if xc is None or xr is None else f'{xc / xr:.2f}'} |")
    for key, w in (("A", "A"), ("R", "R"), ("len40", "the length test at 40")):
        d = cand["lv_" + key] - ref["lv_" + key]
        out.append(pre2 + f"level of {w} | {'lower' if d <= -15 else 'higher' if d >= 15 else 'about the same'} | {ref['lv_' + key]:.2f} | {cand['lv_' + key]:.2f} | |")
    return out


if "1" in want:
    head("1", "the final report's result tables and the tables file, recomputed from the receipts; recomputed / reported. Three-row version [single-row version]; a level is the mean of the last three scored rows of the reading")
    n_fig = n_ok = 0; differ = []
    for k in ORDER:
        t, rep = TAB[k], REPORTED[k]
        mine = [t["A80"], t["R80"], *(t["lv_" + x] for x in ("A", "R", "len40")), *(t[o + "80"] for o in OPS), t["rise_R"]]
        theirs = [rep["A80"], rep["R80"], *rep["lv"], *rep["ops80"], rep["rise"]]
        names = ["80 on A", "80 on R", "level A", "level R", "level 40 letters", "max 80", "min 80", "add 80", "sub 80", "rise of R"]
        oks = [(a == b) if i < 2 or 5 <= i <= 8 else (f"{a:+.1f}" == f"{b:+.1f}") for i, (a, b) in enumerate(zip(mine, theirs))]
        for nmf, ok, a, b in zip(names, oks, mine, theirs):
            n_fig += 1; n_ok += ok
            if not ok:
                differ.append((k, nmf, a, b))
        last3 = [q["step"] for q in t["curve"][-3:]]
        print(f"   {k[0]:15s} {k[1]}  ({t['rows']} rows; level rows {last3[0]:,} to {last3[2]:,}; rise against rows {last3[0] - 2048:,} to {last3[2] - 2048:,})")
        print(f"      80 on A {fu(t['A80'])} [{fu(t['A80s'])}] / {fu(rep['A80'])}; 80 on R {fu(t['R80'])} [{fu(t['R80s'])}] / {fu(rep['R80'])}; 50 on A {fu(t['A50'])} [{fu(t['A50s'])}]; 50 on R {fu(t['R50'])} [{fu(t['R50s'])}]; 90 on A {fu(t['A90'])} [{fu(t['A90s'])}], on R {fu(t['R90'])} [{fu(t['R90s'])}]")
        print(f"      levels A {t['lv_A']:.2f}, R {t['lv_R']:.2f}, 40 letters {t['lv_len40']:.2f} / {rep['lv']}; taught {t['lv_taught']:.2f}; 12 letters {t['lv_len12']:.2f}; rise of R {t['rise_R']:+.2f} / {rep['rise']:+.1f}")
        print("      max, min, add, sub to 80: " + ", ".join(f"{fu(t[o + '80'])} [{fu(t[o + '80s'])}]" for o in OPS) + " / " + ", ".join(fu(v) for v in rep["ops80"]) + " | levels " + ", ".join(f"{t['lv_' + o]:.2f}" for o in OPS)
              + (f" | answer bytes on R's items: first {t['lv_b1']:.2f}, second {t['lv_b2']:.2f}" if "lv_b1" in t else ""))
        print(f"      equal to the report: {sum(oks)} of {len(oks)}")
    print(f"   figures of the final report's two result tables compared (14 rows of ten figures): {n_fig}; equal: {n_ok}" + (f"; DIFFERENT: {differ}" if differ else "; none differs"))
    n46 = sum(1 for k in ORDER if k[0].startswith("C46")) * 10
    print(f"   of them the rows of C46-one: {n46}; equal: {n46 - sum(1 for d in differ if d[0][0].startswith('C46'))}")
    for k in C46:
        t = TAB[k]
        print(f"   {k[0]} {k[1]}: 50 on A {fu(t['A50'])}, on R {fu(t['R50'])} (the tables file: {REPORTED_50[k][0]:,}, {REPORTED_50[k][1]:,}): {yes((t['A50'], t['R50']) == REPORTED_50[k])}")
    Cb, Cs, Ib, Fb = TAB[("C-dense", "B")], TAB[("C-dense", "S")], TAB[("in16", "B")], TAB[("F10", "B")]
    Ob, Os, Qb, Qs = TAB[("C46-one @end", "B")], TAB[("C46-one @end", "S")], TAB[("C16-in @end", "B")], TAB[("C16-in @end", "S")]
    print("   the ratios and words of the report's text:")
    print(f"      C46-one / C-dense, 80 on R: B {Ob['R80']:,} / {Cb['R80']:,} = {Ob['R80'] / Cb['R80']:.4f} (reported 2.22); S {Os['R80']:,} / {Cs['R80']:,} = {Os['R80'] / Cs['R80']:.4f} (reported 2.68)")
    print(f"      C46-one / C-dense, 80 on A: B {Ob['A80'] / Cb['A80']:.4f} (reported 1.83); S {Os['A80'] / Cs['A80']:.4f} (reported 2.73); C46-one / C16-in, 80 on A, B: {Ob['A80'] / Qb['A80']:.4f} (reported 0.35)")
    print(f"      in16 / F10, 80 on R, B: {Ib['R80'] / Fb['R80']:.4f} (reported 2.16); 80 on A {Ib['A80'] / Fb['A80']:.4f} (reported 1.91)")
    for r, o, c, q in (("B", Ob, Cb, Qb), ("S", Os, Cs, Qs)):
        for ref_nm, ref in (("C-dense", c), ("C16-in @end", q)):
            d = {x: o["lv_" + x] - ref["lv_" + x] for x in ("A", "R", "len40")}
            words = {x: "lower" if v <= -15 else "higher" if v >= 15 else "about the same" for x, v in d.items()}
            dsc = [describe(ref[x], o[x], ref["dose"] - 512, DOSE - 512) for x in ("A80", "R80")]
            ma = [IN_MA[describe(None if ref[x] is None else ref[x] * m[ref["setting"]], o[x] * m["C46-one"], (ref["dose"] - 512) * m[ref["setting"]], (DOSE - 512) * m["C46-one"])] for m in (MA, MA2) for x in ("A80", "R80")]
            print(f"      {r}: C46-one at the end against {ref_nm}: updates to 80 on A *{dsc[0]}*, on R *{dsc[1]}*; multiply-adds first count A *{ma[0]}*, R *{ma[1]}*, second count A *{ma[2]}*, R *{ma[3]}*;"
                  f" levels A {d['A']:+.2f} *{words['A']}*, R {d['R']:+.2f} *{words['R']}*, 40 letters {d['len40']:+.2f} *{words['len40']}*")
    print(f"      C46-one B, R at the end: {Ob['lv_R']:.4f}, to one decimal {Ob['lv_R']:.1f} (the report's table 94.2; its text 94.3); minus C16-in's {Qb['lv_R']:.4f}: {Ob['lv_R'] - Qb['lv_R']:.2f} (reported 14.8)")
    print(f"      the last three rows of R: C46-one B {[round(q['R'], 2) for q in Ob['curve'][-3:]]}; S {[round(q['R'], 2) for q in Os['curve'][-3:]]}")
    for r in "BS":
        a, b = TAB[("C46-one @24,576", r)], TAB[("C46-one @end", r)]
        print(f"      {r}: rows at 24,576 and at the end in the two readings: {a['rows']}, {b['rows']}; within the first 24,576 updates *still rising* (5 points or more): C46-one {a['rise_R']:+.2f}, C16-in {TAB[('C16-in @24,576', r)]['rise_R']:+.2f}")
    md = show(f"{E}E50_TABLES_{SEED}.md").decode(); lines = set(md.splitlines())
    exp, missing = table_rows(TAB, lines)
    kinds = sorted({w for w, _ in exp}, key=[w for w, _ in exp].index)
    print(f"   the tables file E50_TABLES_{SEED}.md: rows of C46-one written here in the file's formats and found in it character for character: {len(exp) - len(missing)} of {len(exp)}")
    print("      " + "; ".join(f"{w}: {sum(1 for a, _ in exp if a == w) - sum(1 for a, _ in missing if a == w)} of {sum(1 for a, _ in exp if a == w)}" for w in kinds))
    for w, l in missing:
        print(f"      NOT FOUND ({w}): {l}")
    for r in "BS":
        print(f"      the file's rule for stage 3 under {r}: " + next((l for l in md.splitlines() if l.startswith(f"- {r}, stage 3")), "NOT FOUND"))
    sm = json.loads(show(f"{E}E50_SUMMARY_{SEED}.json")); n_ok = n_all = 0; bad = []
    for r in "BS":
        for nm, _, _ in READ:
            t = TAB[(nm, r)]; v = sm["runs"].get(f"{nm} {r}")
            if v is None:
                bad.append((nm, r, "not in the summary")); continue
            for key in ("A", "R"):
                for th in (50, 80, 90):
                    for ver, suf in (("three", ""), ("single", "s")):
                        w = v["thresholds"][f"{key}{th}_{ver}"]; n_all += 1; ok = (w["updates"] if w else None) == t[f"{key}{th}{suf}"]; n_ok += ok
                        bad += [] if ok else [(nm, r, key, th, ver)]
            for o in OPS:
                for ver, suf in (("three", ""), ("single", "s")):
                    w = v["thresholds"][f"{o}80_{ver}"]; n_all += 1; ok = (w["updates"] if w else None) == t[f"{o}80{suf}"]; n_ok += ok
                    bad += [] if ok else [(nm, r, o, ver)]
            for key, mine in [(x, t["lv_" + x]) for x in ("A", "R", "len40", "len12", "taught") + OPS] + ([("R_first_byte", t["lv_b1"]), ("R_second_byte", t["lv_b2"])] if "lv_b1" in t else []):
                n_all += 1; ok = abs(v["levels"][key] - mine) < 1e-9; n_ok += ok
                bad += [] if ok else [(nm, r, "level", key)]
            n_all += 1; ok = abs(v["R_last3_minus_2048_earlier"] - t["rise_R"]) < 1e-9; n_ok += ok
            bad += [] if ok else [(nm, r, "rise")]
    print(f"   the summary file E50_SUMMARY_{SEED}.json: thresholds, levels and rises of the 14 readings equal to the recomputed figures: {n_ok} of {n_all}" + (f"; differing: {bad}" if bad else "") + f"; its entry in_progress: {sm.get('in_progress')}")


# ---------------------------------------------------------------- 2a: answers and counts from the weights
if "2a" in want:
    head("2a", "the recorded answers and counts of C46-one from the saved weights, by the reviewer's forward (e50_code_reader.forward with the draw of this script); reader / receipt")
    AP, RP, TA, LP = panels()
    grand = [0, 0]; rows_ok = 0
    for r in "SB":
        rc = receipt("C46-one", r)
        for suffix, row, what, sha_rec in (("", rc["curve"][-1], "the final file", rc["weights_sha256"]), (f"_at{FIRST}", next(q for q in rc["curve"] if q["step"] == FIRST), "the file at update 24,576", rc["first_dose"]["weights_sha256"])):
            t1 = time.time()
            w = weights(wname("C46-one", r, suffix)); net = Net(w["T"], "C46-one")
            print(f"   C46-one {r}, {what} ({w['size']:,} bytes; SHA-256 equals the receipt's: {yes(w['sha'] == sha_rec)}); against the receipt's row {row['step']:,}; the draws equal the file's three masks: {yes(net.draws_equal_file)}")
            x = np.array([[BOS] + p + a[:-1] for p, a in AP["add89"][:6] + AP["sub89"][:6] + AP["max89"][:6] + AP["min89"][:6]], dtype=np.int64)
            la = R50.forward(net.W, net.mask, net.F, net.code, x)[1]; lb = S2.forward_file(w["T"], x)[1]
            print(f"      two forms of the forward on 24 items: largest difference of a logit {float(np.abs(la - lb).max()):.1e} (largest logit {float(np.abs(la).max()):.1f}); the same largest output everywhere: {yes(bool((la.argmax(-1) == lb.argmax(-1)).all()))}")
            mine = dict(A={n: net.exact(it) for n, it in AP.items()}, R={n: net.exact(it) for n, it in RP.items()}, taught={n: net.exact(it) for n, it in TA.items()})
            theirs = dict(A=row["A_counts"], R=row["R_counts"], taught=row["taught_counts"])
            ok_all = True
            for k, nm in (("A", "the panel behind A (add, sub, max, min)"), ("R", "R (add, sub)"), ("taught", "the taught panel (add, sub, max, min)")):
                ok_all &= mine[k] == theirs[k]
                print(f"      {nm}: {list(mine[k].values())} / {list(theirs[k].values())}: {yes(mine[k] == theirs[k])}")
            b = {n: net.bytes_apart(it) for n, it in RP.items()}
            b1 = {n: v[0] for n, v in b.items()}; b2 = {n: v[1] for n, v in b.items()}
            ok_all &= b1 == row["R_first_byte_counts"] and b2 == row["R_second_byte_counts"]
            print(f"      R's items, first byte right {list(b1.values())} / {list(row['R_first_byte_counts'].values())}: {yes(b1 == row['R_first_byte_counts'])}; second {list(b2.values())} / {list(row['R_second_byte_counts'].values())}: {yes(b2 == row['R_second_byte_counts'])}")
            ans = json.loads(show(f"{E}e50_C46-one_{r}_{SEED}{suffix}_answers.json"))
            tot = 0
            for k in (12, 40):
                rec = ans[str(k)]; got = net.respond([p for p, _ in LP[k]])
                same = sum(a == g for a, g in zip(rec, got)); tot += same
                right = sum(a == t for a, (_, t) in zip(rec, LP[k])); mine_right = sum(g == t for g, (_, t) in zip(got, LP[k]))
                ok_all &= mine_right == right == row["length_counts"][str(k)]
                print(f"      answers file, {k} letters: {len(rec)} recorded; the reader's greedy answer equals the recorded one in {same} of {len(LP[k])}; right: reader {mine_right}, recorded {right}, receipt {row['length_counts'][str(k)]}")
            grand[0] += tot; grand[1] += 600; rows_ok += ok_all
            print(f"      every count of the row reproduced: {yes(ok_all)}; recorded answers reproduced: {tot} of 600   [{time.time() - t1:.0f} s]", flush=True)
            del net
    print(f"   rows reproduced in every count: {rows_ok} of 4; recorded answers reproduced over the four files: {grand[0]} of {grand[1]}")


# ---------------------------------------------------------------- 2b, 2c, 2d
FIXED, STRUCT, TRAINED = S2.FIXED, S2.STRUCT, S2.TRAINED
if want & {"2b", "2c", "2d"}:
    FILES = {r: {k: weights(wname("C46-one", r, sfx)) for k, sfx in (("start", "_start"), ("at 24,576", f"_at{FIRST}"), ("final", ""))} for r in "SB"}
REPORTED_SHA = {}

if "2b" in want:
    head("2b", "the code is fixed: the start file, the file at update 24,576 and the final file of each run of C46-one")
    print(f"   the digest of the code by the reader: {S2.CODE0.digest()} (the receipts' 12111044f0f8bc70: {yes(S2.CODE0.digest() == '12111044f0f8bc70')})")
    dense0 = weights(wname("C-dense", "S", "_start"))["T"]
    for r in "SB":
        f = FILES[r]; S0, T1, T2 = f["start"]["T"], f["at 24,576"]["T"], f["final"]["T"]; rc = receipt("C46-one", r)
        print(f"   C46-one {r}: SHA-256 of the start file {f['start']['sha'][:16]} (receipt: {yes(f['start']['sha'] == rc['start_sha256'])}), at 24,576 {f['at 24,576']['sha'][:16]} (receipt: {yes(f['at 24,576']['sha'] == rc['first_dose']['weights_sha256'])}),"
              f" final {f['final']['sha'][:16]} (receipt: {yes(f['final']['sha'] == rc['weights_sha256'])}); the log's last line names {show(f'{E}logs/e50_C46-one_{r}_{SEED}.log').decode().splitlines()[-1].split()[-1]}")
        print(f"      tensors in each file: {len(S0)}, {len(T1)}, {len(T2)}; the same names: {yes(set(S0) == set(T1) == set(T2))}")
        print("      bit for bit in the three files: " + "; ".join(f"{k} {yes(bit(S0[k], T1[k]) and bit(S0[k], T2[k]))}" for k in FIXED))
        print("      the structure, bit for bit in the three files: " + "; ".join(f"{k} {yes(bit(S0[k], T1[k]) and bit(S0[k], T2[k]))}" for k in STRUCT))
        print("      trained tensors, entries that differ, start to 24,576 | 24,576 to final: " + "; ".join(f"{k} {int((T1[k] != S0[k]).sum()):,} | {int((T2[k] != T1[k]).sum()):,} of {S0[k].size:,}" for k in TRAINED))
        amp = float(S0["code_amplitude"]); code = R50.Code(SEED, amplitude=amp); tb = S0["code_table"]
        print(f"      code_table equal bit for bit to the reader's table: {yes(bit(tb, code.table()))}; entries not zero {int((tb != 0).sum())}, all {amp:.6f}: {yes(bool((tb[tb != 0] == np.float32(amp)).all()))}; code_owner equals the reader's: {yes(S0['code_owner'].tolist() == code.owner)};"
              f" value in the receipt {rc['code']['amplitude']:.6f}")
        m3 = np.tile(S0["mask_in"], (3, 1)); exp_ih = (dense0["rnn.weight_ih_l0"] * m3 * np.float32((D / K) ** 0.5)).astype(np.float32)
        rest = [k for k in dense0 if k not in ("rnn.weight_ih_l0", "mask_in", "k_in")]
        print(f"      against the start file of C-dense: the fixed tensors bit for bit {yes(all(bit(S0[k], dense0[k]) for k in FIXED))}; every tensor but the input map, its mask and k_in bit for bit {yes(all(bit(S0[k], dense0[k]) for k in rest))};"
              f" the input map is C-dense's start times the mask times sqrt(512/46) = {(D / K) ** 0.5:.4f}: largest difference {float(np.abs(S0['rnn.weight_ih_l0'] - exp_ih).max()):.1e}")
        print(f"      the receipt: byte_table_and_l1_bit_equal_to_start {rc['byte_table_and_l1_bit_equal_to_start']}; code_table_sha256_equal_at_every_row {rc['code_table_sha256_equal_at_every_row']}; the optimizer holds state for: {', '.join(rc['optimizer_state_for'])}")
    print(f"   the two start files hold the same tensors bit for bit: {yes(all(bit(FILES['S']['start']['T'][k], FILES['B']['start']['T'][k]) for k in FILES['S']['start']['T']))}")


if "2c" in want:
    head("2c", "the masks of C46-one in the three files of each run: each cell reads 46 values, one of every used byte")
    mine = draw_one_runner(SEED); theirs = R50.draw_one(SEED)
    print(f"   the draw written here from the runner's text: digest {R49.digest16(mine)} (the brief and the receipts: 6788d5475af8c847: {yes(R49.digest16(mine) == '6788d5475af8c847')}); equal to e50_code_reader.draw_one: {yes(mine == theirs)}")
    for r in "SB":
        rc = receipt("C46-one", r)
        for k in ("start", "at 24,576", "final"):
            T = FILES[r][k]["T"]; m_in, m_out, m_rec = T["mask_in"], T["mask_out"], T["mask"]
            own = T["code_owner"]; per = m_in.sum(1)
            per_byte = np.stack([(m_in & (own == b)[None]).sum(1) for b in R50.USED], 1)       # (256, 46): values of each used byte a cell reads
            unowned = (m_in & (own < 0)[None]).sum()
            o_in = ~np.tile(m_in, (3, 1)); o_rec = ~np.tile(m_rec, (3, 1)); wi, wh = T["rnn.weight_ih_l0"], T["rnn.weight_hh_l0"]
            print(f"   C46-one {r}, {k}: values a cell reads {int(per.min())} to {int(per.max())}; cells with exactly 46: {int((per == K).sum())} of {U}; values of each used byte a cell reads: {int(per_byte.min())} to {int(per_byte.max())} (pairs with exactly one: {int((per_byte == 1).sum()):,} of {per_byte.size:,}); values without an owner read: {int(unowned)};"
                  f" k_in {int(T['k_in'])}, k_out {int(T['k_out'])}; the mask equals the draw: {yes(np.array_equal(m_in, S2.F['C46-one'].mask_in))} (receipt's digest {rc['draw_digests']['input']})")
            print(f"      outside a mask: input map {int(o_in.sum()):,} entries, not zero {int((wi[o_in] != 0).sum())}; recurrent map {int(o_rec.sum()):,}, not zero {int((wh[o_rec] != 0).sum())}; lm's mask full: {yes(bool(m_out.all()))};"
                  f" inside and exactly zero: input map {int((wi[~o_in] == 0).sum())} of {int((~o_in).sum()):,}; the recurrent mask equals the reader's: {yes(np.array_equal(m_rec, S2.CELLS.mask))}; the three gates within the mask: {yes(bool(((wi != 0).reshape(3, U, D) <= m_in[None]).all()))}")
        T = FILES[r]["final"]["T"]
        n_in, n_out, n_rec = 3 * int(T["mask_in"].sum()), int(T["mask_out"].sum()), 3 * int(T["mask"].sum()); l2, l3 = T["l2.weight"].size, T["l3.weight"].size
        second = 3 * U + ON * T["l2.weight"].shape[0] + n_out + n_rec + l3
        print(f"   multiply-adds a position: first count {n_in:,} + {n_out:,} + {n_rec:,} + {l2:,} + {l3:,} = {n_in + n_out + n_rec + l2 + l3:,} (the brief 568,576; the receipt {rc['multiply_adds_per_position']:,});"
              f" second count: input map 3 x 256 (one turned-on value a cell for every byte) + l2 {ON * T['l2.weight'].shape[0]:,} + lm + cells + l3 = {second:,} (the brief 277,504; the receipt {rc['multiply_adds_per_position_second_count']:,.0f})")
        print(f"   the receipt's coverage: {rc['coverage']['bytes_of_20']}, digits {rc['coverage']['digits_of_10']}, weights behind a cell and a needed byte: one in {rc['coverage']['weights_behind_a_cell_and_needed_byte']['1']:,} pairs")


def figures(T, groups, eye=np.eye(U, dtype=bool)):
    """exp50_code.knows_figure, input_size and the magnitudes, in double precision from the file's tensors (e50_stage2_verify, part 2d)."""
    code = R50.Code(SEED, amplitude=float(T["code_amplitude"])); amp = np.float64(T["code_amplitude"])
    m3i = np.tile(T["mask_in"], (3, 1)); Wi = T["rnn.weight_ih_l0"].astype(np.float64) * m3i
    gi = np.stack([amp * Wi[:, code.own[b]].sum(1) for b in R50.NEEDED])
    m3 = np.tile(T["mask"], (3, 1)); Wh = np.abs(T["rnn.weight_hh_l0"].astype(np.float64))
    kn = []
    for g in groups:
        rows_ = np.concatenate([g, g + U, g + 2 * U]); nb = np.tile((T["mask"] & ~eye)[g], (3, 1)); ws = np.tile(T["mask"][g], (3, 1))
        kn.append((float(Wh[rows_][nb].mean()), float(Wh[rows_][ws].mean())))
    return dict(rms=float(np.sqrt((gi ** 2).mean())), ih=float(np.abs(T["rnn.weight_ih_l0"].astype(np.float64))[m3i].mean()),
                lm=float(np.abs(T["lm.weight"].astype(np.float64))[T["mask_out"]].mean()), hh=float(Wh[m3].mean()), kn=kn)


if "2d" in want:
    head("2d", "`by what a cell knows` and `other measures` for C46-one, from the weight files; start, 24,576, end / the receipt's rows / the report")
    kd = R50.knows(S2.CODE0, S2.F["C16-in"], R50.DIGITS)
    groups = [np.nonzero(kd <= 2)[0], np.nonzero((kd >= 3) & (kd <= 4))[0], np.nonzero(kd >= 5)[0]]
    print(f"   the three groups (input for 0 to 2, 3 or 4, 5 or more digits at 16 values a cell): {[len(g) for g in groups]}")
    NAMES = ("digits_0_to_2", "digits_3_to_4", "digits_5_or_more")
    REP_KN = {"B": (0.195, 0.206, 0.192), "S": (0.212, 0.211, 0.208)}; REP_RMS = {"B": 0.85, "S": 0.81}
    md = show(f"{E}E50_TABLES_{SEED}.md").decode(); lines = set(md.splitlines()); wrows = []; rep = []
    for r in "BS":
        rc = receipt("C46-one", r); by = {q["step"]: q for q in rc["curve"]}
        fg = {k: figures(FILES[r][k]["T"], groups) for k in ("start", "at 24,576", "final")}; rw = {"start": by[0], "at 24,576": by[FIRST], "final": by[DOSE]}
        print(f"   C46-one {r}:")
        for i, n in enumerate(NAMES):
            ok = all(abs(fg[k]["kn"][i][j] - rw[k]["by_what_a_cell_knows"][n][w]) < 5e-7 for k in fg for j, w in ((0, "neighbours_only"), (1, "with_self")))
            print(f"      group {i + 1}: recurrent weights from neighbours [with itself] " + ", ".join(f"{fg[k]['kn'][i][0]:.6f} [{fg[k]['kn'][i][1]:.4f}]" for k in fg) + f"; equal to the receipt: {yes(ok)}")
            rep.append((f"by what a cell knows, group {i + 1}, {r}, end", fg["final"]["kn"][i][0], REP_KN[r][i], 3))
        print(f"      input for a needed byte (rms): " + ", ".join(f"{fg[k]['rms']:.6f}" for k in fg) + f" / " + ", ".join(f"{rw[k]['input_size']['needed_rms']:.4f}" for k in fg)
              + f": {yes(all(abs(fg[k]['rms'] - rw[k]['input_size']['needed_rms']) < 5e-6 for k in fg))}")
        rep.append((f"input for a needed byte, {r}, end", fg["final"]["rms"], REP_RMS[r], 2))
        print("      mean magnitude of a kept weight: input map " + ", ".join(f"{fg[k]['ih']:.4f}" for k in fg) + f" (receipt equal: {yes(all(abs(fg[k]['ih'] - rw[k]['kept_mean_magnitude']['rnn.weight_ih_l0']) < 5e-7 for k in fg))}); lm "
              + ", ".join(f"{fg[k]['lm']:.4f}" for k in fg) + f" ({yes(all(abs(fg[k]['lm'] - rw[k]['kept_mean_magnitude']['lm.weight']) < 5e-7 for k in fg))}); recurrent " + ", ".join(f"{fg[k]['hh']:.4f}" for k in fg)
              + f" ({yes(all(abs(fg[k]['hh'] - rw[k]['kept_mean_magnitude']['rnn.weight_hh_l0']) < 5e-7 for k in fg))})")
        cs, cm, ce = rc["cosine_first_batch_start"], rc["first_dose"]["cosine_first_batch"], rc["cosine_first_batch_end"]
        for nm, k, cend in (("C46-one @24,576", "at 24,576", cm), ("C46-one @end", "final", ce)):
            a_, b_ = fg["start"], fg[k]
            wrows.append(f"| {nm} | {r} | " + " | ".join(f"{cs[m]:.3f}, {cend[m]:.3f}" for m in ("rnn.weight_ih_l0", "lm.weight", "rnn.weight_hh_l0")) + " | " + " | ".join(f"{a_[x]:.4f}, {b_[x]:.4f}" for x in ("ih", "lm", "hh")) + f" | {a_['rms']:.3f}, {b_['rms']:.3f} |")
            wrows.append(f"| {nm} | {r} | " + " | ".join(f"{a_['kn'][i][0]:.4f}, {b_['kn'][i][0]:.4f} [{a_['kn'][i][1]:.4f}, {b_['kn'][i][1]:.4f}]" for i in range(3)) + " |")
        print(f"      cosines (the receipt only): input map {cs['rnn.weight_ih_l0']:.3f} -> {ce['rnn.weight_ih_l0']:.3f}, lm {cs['lm.weight']:.3f} -> {ce['lm.weight']:.3f}, recurrent {cs['rnn.weight_hh_l0']:.3f} -> {ce['rnn.weight_hh_l0']:.3f}")
    bad = [(w, v, q, d) for w, v, q, d in rep if f"{v:.{d}f}" != f"{q:.{d}f}"]
    print(f"   the final report's figures for C46-one recomputable from the weights: {len(rep)}; equal to the weight file's figure rounded to the report's decimals: {len(rep) - len(bad)}")
    for w, v, q, d in bad:
        print(f"      DIFFERENT: {w}: {v:.6f} from the weight file, {v:.{d}f} to {d} decimals; the report has {q:.{d}f}")
    missing = [l for l in wrows if l not in lines]
    print(f"   the tables file's rows of C46-one in the two tables of sizes, from the weight files: found {len(wrows) - len(missing)} of {len(wrows)}")
    for l in missing:
        print(f"      NOT FOUND: {l}")


# ---------------------------------------------------------------- 3: the restart
if "3" in want:
    head("3", "the restart: each completed run of C46-one against its interrupted predecessor; and the batch digests rebuilt from the seed")
    SKIP = {"wall", "cpu_updates", "cpu_scoring"}
    for r, upto_log in (("B", 9728), ("S", 2560)):
        done = TAB[("C46-one @end", r)]["rc"]; part = json.loads(show(f"{INT}e50_C46-one_{r}_{SEED}.json"))
        by = {q["step"]: q for q in done["curve"]}; pr = part["curve"]
        diff_rows = [(q["step"], sorted(k for k in q if k not in SKIP and q[k] != by[q["step"]].get(k))) for q in pr]
        diff_rows = [d for d in diff_rows if d[1]]
        keys = sorted(set(k for q in pr for k in q) - SKIP)
        print(f"   C46-one {r}: the interrupted receipt holds {len(pr)} rows (0 to {pr[-1]['step']:,}), complete {part['complete']}; every field of each row but the times ({len(keys)} fields: {', '.join(keys)}) equal to the completed run's row: {yes(not diff_rows)}" + (f"; DIFFERENT: {diff_rows}" if diff_rows else ""))
        top = sorted(k for k in part if k not in ("curve", "provenance") and part[k] != done.get(k))
        print(f"      top-level entries of the interrupted receipt that differ from the completed one's: {top if top else 'none'}; start_sha256 equal: {yes(part['start_sha256'] == done['start_sha256'])}; start_tensors_sha256 equal: {yes(part['start_tensors_sha256'] == done['start_tensors_sha256'])};"
              f" first_batch_loss_start equal: {yes(part['first_batch_loss_start'] == done['first_batch_loss_start'])}; cosines at the start equal: {yes(part['cosine_first_batch_start'] == done['cosine_first_batch_start'])}; runner's and sources' SHA-256 equal: {yes(part['provenance']['runner_sha256'] == done['provenance']['runner_sha256'] and part['provenance']['sources_sha256'] == done['provenance']['sources_sha256'])}")
        pc = part['provenance']['git']['commit']
        print(f"      the commit the interrupted receipt names: {pc[:8]} (sources modified against it: {part['provenance']['git']['sources_modified_vs_commit']}), the completed one {done['provenance']['git']['commit'][:8]}; {pc[:8]} is on the branch before {REF}: {yes(subprocess.run(['git', 'merge-base', '--is-ancestor', pc, REF], cwd=S2.ROOT).returncode == 0)};"
              f" exp50_code.py and e50_sparse_code.py the same blobs there as at {REF}: {yes(all(git('rev-parse', f'{pc}:{M}{f}') == git('rev-parse', f'{REF}:{M}{f}') for f in ('exp50_code.py', 'e50_sparse_code.py')))}")
        a = show(f"{INT}e50_C46-one_{r}_{SEED}_start.pt"); b = show(f"{E}e50_C46-one_{r}_{SEED}_start.pt")
        print(f"      the interrupted start file is byte-equal to the completed run's start file: {yes(a == b)} (SHA-256 {sha(a)[:16]}; the receipts' {done['start_sha256'][:16]})")
        strip = lambda l: re.sub(r" cpu \d+\+\d+s$", "", l)
        li = [l for l in show(f"{INT}e50_C46-one_{r}_{SEED}.log").decode().splitlines() if l.startswith("e50 C46-one")]
        lc = [l for l in show(f"{E}logs/e50_C46-one_{r}_{SEED}.log").decode().splitlines() if l.startswith("e50 C46-one")]
        rows_i = [l for l in li if re.match(rf"e50 C46-one {r} {SEED} \d+:", l)]
        steps_i = [int(l.split()[4].rstrip(":")) for l in rows_i]
        same = [strip(x) == strip(y) for x, y in zip(li, lc)]
        print(f"      the interrupted log: {len(li)} lines of the run, {len(rows_i)} scored rows 0 to {steps_i[-1]:,} (reported {upto_log:,}); each equal to the completed log's line of the same place, the CPU field set aside: {sum(same)} of {len(li)}")
        print(f"      CPU at the interrupted log's last row: {rows_i[-1].split('cpu ')[1]} = {sum(int(v) for v in re.findall(r'(\d+)', rows_i[-1].split('cpu ')[1])) / 3600:.3f} CPU-hours")
    t1 = time.time()
    mine = S2.stream_digests(SEED, DOSE)
    for r in "BS":
        rc = TAB[("C46-one @end", r)]["rc"]; rows = {q["step"]: q["data_digest"] for q in rc["curve"]}
        dn = {q["step"]: q["data_digest"] for q in TAB[("C-dense", r)]["rc"]["curve"]}; c16 = {q["step"]: q["data_digest"] for q in TAB[("C16-in @end", r)]["rc"]["curve"]}
        print(f"   C46-one {r}: rows whose recorded batch digest equals the one rebuilt here from the seed: {sum(rows[u]['sha256'] == mine[u] for u in rows)} of {len(rows)}; equal to C16-in's at all 193 rows: {yes(all(rows[u] == c16[u] for u in rows))};"
              f" to C-dense's at its 97 rows: {yes(all(rows[u] == dn[u] for u in dn))}; the same optimizer record as C-dense: {yes(rc['optimizer'] == TAB[('C-dense', r)]['rc']['optimizer'])}")
    print(f"   (the rebuilt stream: {time.time() - t1:.0f} s)")


# ---------------------------------------------------------------- 4: the figures behind the report's statements on all three stages
if "4" in want:
    head("4", "the figures behind the statements of the final report (as amended at " + AMEND + ") that concern all three stages")
    runs = [("C-dense", "B"), ("C-dense", "S"), ("C16-in", "B"), ("C16-in", "S"), ("C46-one", "B"), ("C46-one", "S")]
    cpu = {k: (TAB[(k[0] if k[0] == "C-dense" else k[0] + " @end", k[1])]["rc"]["cpu_updates"] + TAB[(k[0] if k[0] == "C-dense" else k[0] + " @end", k[1])]["rc"]["cpu_scoring"]) / 3600 for k in runs}
    print("   CPU-hours of the six runs (receipts): " + ", ".join(f"{s} {r} {v:.3f}" for (s, r), v in cpu.items()) + f"; sum {sum(cpu.values()):.3f} (reported 9.72); the sum of the six figures as rounded in the report's table: {sum(round(v, 2) for v in cpu.values()):.2f}")
    inter = {}
    for r in "BS":
        last = [l for l in show(f"{INT}e50_C46-one_{r}_{SEED}.log").decode().splitlines() if re.match(rf"e50 C46-one {r} {SEED} \d+:", l)][-1]
        inter[r] = sum(int(v) for v in re.findall(r"(\d+)", last.split("cpu ")[1])) / 3600
    qual = sum(json.loads(show(f"{P}e50_qual/{f}_{SEED}.json"))["cpu"] for f in ("check1", "check3", "check4", "check5", "checkfigures")) / 3600
    print(f"   interrupted pieces (CPU at the last scored row of each log): B {inter['B']:.3f}, S {inter['S']:.3f}, sum {sum(inter.values()):.3f} (reported 0.34, 0.10, 0.44); the checks (field `cpu` of the five check receipts) {qual:.3f} (reported 0.11);"
          f" in all {sum(cpu.values()) + sum(inter.values()) + qual:.3f} (reported 10.27); without the interrupted pieces {sum(cpu.values()) + qual:.3f} (reported 9.83)")
    print("   completion: " + "; ".join(f"{s} {r}: receipt complete {TAB[(s if s == 'C-dense' else s + ' @end', r)]['rc']['complete']}, log ends with `e50 complete` {yes(show(f'{E}logs/e50_{s}_{r}_{SEED}.log').decode().splitlines()[-1].startswith('e50 complete'))}" for s, r in runs))
    # the descriptions of the report that rest on all three stages
    lv = lambda nm, r, key: TAB[(nm, r)]["lv_" + key]
    for r in "BS":
        print(f"   {r}: answer bytes on R's items, levels at the end, C-dense, C16-in, C46-one: first " + ", ".join(f"{lv(n, r, 'b1'):.2f}" for n in ("C-dense", "C16-in @end", "C46-one @end"))
              + "; second " + ", ".join(f"{lv(n, r, 'b2'):.2f}" for n in ("C-dense", "C16-in @end", "C46-one @end"))
              + (" (reported 98.1, 84.2, 95.6; 97.6, 92.6, 98.2)" if r == "B" else " (reported 96.0, 71.5, 90.2; 94.3, 54.0, 95.5)"))
    for nm in ("C-dense", "C16-in @end", "C46-one @end"):
        print(f"   max and min to 80, {nm}: " + "; ".join(f"{r} {fu(TAB[(nm, r)]['max80'])}, {fu(TAB[(nm, r)]['min80'])}" for r in "BS") + f"; levels at the end " + "; ".join(f"{r} {lv(nm, r, 'max'):.2f}, {lv(nm, r, 'min'):.2f}" for r in "BS"))
    print(f"   add and sub levels at the end, C16-in S: {lv('C16-in @end', 'S', 'add'):.2f}, {lv('C16-in @end', 'S', 'sub'):.2f} (reported 49.4 and 49.6)")
    for nm in ("C-dense", "C16-in @end", "C46-one @end"):
        for r in "BS":
            rc = TAB[(nm, r)]["rc"]; last = rc["curve"][-1]
            print(f"   {nm} {r}: cosine of the input map on the first batch, start -> end {rc['cosine_first_batch_start']['rnn.weight_ih_l0']:.3f} -> {rc['cosine_first_batch_end']['rnn.weight_ih_l0']:.3f}; a cell's input for a needed byte (receipt) start -> end {rc['curve'][0]['input_size']['needed_rms']:.6f} -> {last['input_size']['needed_rms']:.6f};"
                  f" by what a cell knows at the start (receipt, neighbours only) {', '.join(f'{rc['curve'][0]['by_what_a_cell_knows'][g]['neighbours_only']:.6f}' for g in ('digits_0_to_2', 'digits_3_to_4', 'digits_5_or_more'))}")
    Ob, Os, Qb, Qs, Cb, Cs = (TAB[(n, r)] for n, r in (("C46-one @end", "B"), ("C46-one @end", "S"), ("C16-in @end", "B"), ("C16-in @end", "S"), ("C-dense", "B"), ("C-dense", "S")))
    Ib, Fb, Is = TAB[("in16", "B")], TAB[("F10", "B")], TAB[("in16", "S")]
    print(f"   the costs set side by side (updates to 80 on R, ratio to the reference of the same rule): B C16-in / C-dense: not reached, more than {(DOSE - 512) / Cb['R80']:.2f}; B C46-one / C-dense {Ob['R80'] / Cb['R80']:.2f}; B in16 / F10 {Ib['R80'] / Fb['R80']:.2f};"
          f" S C46-one / C-dense {Os['R80'] / Cs['R80']:.2f}; S in16: not reached (R level {Is['lv_R']:.2f}); S C16-in: not reached (R level {Qs['lv_R']:.2f})")
    print(f"   on A: B C46-one / C-dense {Ob['A80'] / Cb['A80']:.2f}; B in16 / F10 {Ib['A80'] / Fb['A80']:.2f}; B C16-in / C-dense {Qb['A80'] / Cb['A80']:.2f}")
    print(f"   the length test at 40 letters, levels at the end: C-dense {lv('C-dense', 'B', 'len40'):.2f}, {lv('C-dense', 'S', 'len40'):.2f}; C16-in {lv('C16-in @end', 'B', 'len40'):.2f}, {lv('C16-in @end', 'S', 'len40'):.2f}; C46-one {lv('C46-one @end', 'B', 'len40'):.2f}, {lv('C46-one @end', 'S', 'len40'):.2f};"
          f" at 12 letters: C-dense {lv('C-dense', 'B', 'len12'):.2f}, {lv('C-dense', 'S', 'len12'):.2f}; C16-in {lv('C16-in @end', 'B', 'len12'):.2f}, {lv('C16-in @end', 'S', 'len12'):.2f}; C46-one {lv('C46-one @end', 'B', 'len12'):.2f}, {lv('C46-one @end', 'S', 'len12'):.2f}")
    print("   the predictions, scored here from the receipts / the report's outcome:")
    print(f"      P50.2  C-dense B reaches 80 on R within the dose: {'true' if Cb['R80'] else 'false'} ({fu(Cb['R80'])}) / true (6,912)")
    print(f"      P50.3  C-dense S: {'true' if Cs['R80'] else 'false'} ({fu(Cs['R80'])}) / true (12,800)")
    print(f"      P50.8  C-dense against F10, B, updates to 80 on R about the same or fewer: {describe(Fb['R80'], Cb['R80'], FIRST - 512, FIRST - 512)}, ratio {Cb['R80'] / Fb['R80']:.2f} / true (about the same, 0.73)")
    print(f"      P50.4' C16-in B reaches 80 on R within 49,152: {'true' if Qb['R80'] else 'false'} (single row {fu(Qb['R80s'])}, level {Qb['lv_R']:.2f}) / false")
    print(f"      P50.5' C16-in S: {'true' if Qs['R80'] else 'false'} (level {Qs['lv_R']:.2f}) / false")
    print(f"      P50.6' C16-in B within 3.24 x {Cb['R80']:,} = {3.24 * Cb['R80']:,.1f}: {'true' if Qb['R80'] and Qb['R80'] <= 3.24 * Cb['R80'] else 'false'} / false")
    print(f"      P50.7' C16-in B max and min reach 80: {'true' if Qb['max80'] and Qb['min80'] else 'false'} ({fu(Qb['max80'])}, {fu(Qb['min80'])}) / true;  P50.8' the same under S: {'true' if Qs['max80'] and Qs['min80'] else 'false'} ({fu(Qs['max80'])}, {fu(Qs['min80'])}) / true")
    print(f"      the brief's rule for C46-one: C46-one reaches 80 on R and C16-in does not: B {yes(Ob['R80'] is not None and Qb['R80'] is None)}, S {yes(Os['R80'] is not None and Qs['R80'] is None)}")
    rep = show(f"{E}E50_REPORT.md", AMEND).decode()
    print(f"   the final report at {AMEND}: 'At the start 0.150, 0.152, 0.150' still in it: {yes('At the start 0.150, 0.152, 0.150' in rep)}; '94.3 against 79.4' in it: {yes('94.3 against 79.4' in rep)}; a section 'Amendments' in it: {yes('## Amendments' in rep)}")


# ---------------------------------------------------------------- 5a: the use of the cells
NETS = [("C46-one", "S"), ("C46-one", "B")]
if "5a" in want:
    head("5a", "the use of the cells: standard deviation of each cell's state between the 300 items of the length-test panel at 12 letters, at `=`; final weights of C46-one")
    LP12 = panels()[3][12]
    x = np.array([[BOS] + p for p, _ in LP12], dtype=np.int64)
    assert bool((x[:, -1] == ord("=")).all())
    kd = R50.knows(S2.CODE0, S2.F["C16-in"], R50.DIGITS)
    groups = [np.nonzero(kd <= 2)[0], np.nonzero((kd >= 3) & (kd <= 4))[0], np.nonzero(kd >= 5)[0]]
    print(f"   items {x.shape[0]}, positions {x.shape[1]}; standard deviation over the items (divisor 300) for each cell; magnitude = absolute value of the state averaged over items and cells")
    print("   | Network | Rule | Cells above 0.05 | Cells above 0.10 | Cells at or below 0.01 | Mean over cells | Least, median, largest | Mean magnitude of the state | Cells with mean magnitude below 0.05 | By the three groups of C16-in: mean; cells above 0.10 |")
    print("   |---|---|---|---|---|---|---|---|---|---|")
    notes = []
    for setting, r in NETS:
        net = Net(weights(wname(setting, r))["T"], setting)
        h = net.run(x)[0][:, -1].astype(np.float64)
        sd = h.std(0); mag = np.abs(h)
        print(f"   | {setting} | {r} | {int((sd > 0.05).sum())} | {int((sd > 0.10).sum())} | {int((sd <= 0.01).sum())} | {sd.mean():.4f} | {sd.min():.4f}, {np.median(sd):.4f}, {sd.max():.4f} | {mag.mean():.4f} | {int((mag.mean(0) < 0.05).sum())} | "
              + "; ".join(f"{sd[g].mean():.4f}, {int((sd[g] > 0.10).sum())} of {len(g)}" for g in groups) + " |", flush=True)
        low = np.nonzero(sd <= 0.10)[0]
        notes.append(f"   {setting} {r}: the {len(low)} cells at or below 0.10 have a mean magnitude of {mag.mean(0)[low].min():.3f} to {mag.mean(0)[low].max():.3f}" if len(low) else f"   {setting} {r}: no cell at or below 0.10")
        del net
    print("\n".join(notes))


# ---------------------------------------------------------------- 5b: the answers given
if "5b" in want:
    head("5b", "the answers given under greedy generation, final weights of C46-one: first and second byte right, whole answer right, different answers")
    AP, RP, TA, LP = panels()
    SETS = [("R: add and sub, untaught pairs that need the reduction, no letters", (RP["add89"], RP["sub89"])),
            ("the panel behind A: add and sub, untaught pairs, 12 letters", (AP["add89"], AP["sub89"])),
            ("the panel behind A: max and min, untaught pairs, 12 letters", (AP["max89"], AP["min89"]))]
    digit = lambda b: 48 <= b <= 57

    def counted(got, tgt):
        f = [len(o) > 0 and o[0] == t[0] for o, t in zip(got, tgt)]; s = [len(o) > 1 and o[1] == t[1] for o, t in zip(got, tgt)]
        cnt = {}
        for o in got:
            cnt[tuple(o)] = cnt.get(tuple(o), 0) + 1
        top = max(cnt.items(), key=lambda kv: kv[1])
        return dict(n=len(got), first=sum(f), second=sum(s), whole=sum(o == t for o, t in zip(got, tgt)), both=sum(a and b for a, b in zip(f, s)), only1=sum(a and not b for a, b in zip(f, s)),
                    only2=sum(b and not a for a, b in zip(f, s)), neither=sum(not a and not b for a, b in zip(f, s)), given=len(cnt), right=len({tuple(t) for t in tgt}),
                    formed=sum(len(o) == 3 and digit(o[0]) and digit(o[1]) and o[2] == EOS for o in got), top=("".join(chr(b) if b < 256 else "|" for b in top[0]), top[1]))

    print("   | Network | Rule | Set of items | Items | First byte right | Second byte right | Whole answer right | Both; first only; second only; neither | Different answers given | Different right answers in the set | Two digits and the end symbol | Most frequent answer: items |")
    print("   |---|---|---|---|---|---|---|---|---|---|---|---|")
    notes = []
    for setting, r in NETS:
        net = Net(weights(wname(setting, r))["T"], setting); last = receipt(setting, r)["curve"][-1]
        for nm, halves in SETS:
            got2 = [net.respond([p for p, _ in it]) for it in halves]; tgt2 = [[t for _, t in it] for it in halves]
            a = counted(got2[0] + got2[1], tgt2[0] + tgt2[1]); h2 = [counted(g, t) for g, t in zip(got2, tgt2)]
            pc = lambda v: f"{v:,} ({100 * v / a['n']:.1f}%)"
            print(f"   | {setting} | {r} | {nm} | {a['n']:,} | {pc(a['first'])} | {pc(a['second'])} | {pc(a['whole'])} | {a['both']:,}; {a['only1']:,}; {a['only2']:,}; {a['neither']:,} | {a['given']} | {a['right']} | {a['formed']:,} | `{a['top'][0][:2]}`: {a['top'][1]} |", flush=True)
            if nm.startswith("R"):
                ok = [h2[i]["whole"] for i in range(2)] == list(last["R_counts"].values()) and [h2[i]["first"] for i in range(2)] == list(last["R_first_byte_counts"].values()) and [h2[i]["second"] for i in range(2)] == list(last["R_second_byte_counts"].values())
            else:
                key = [("add89", "sub89"), ("max89", "min89")]["max" in nm]
                ok = [h2[i]["whole"] for i in range(2)] == [last["A_counts"][k] for k in key]
            notes.append(f"      {setting} {r}, {nm.split(':')[0]} ({nm.split(':')[1].split(',')[0].strip()}): counts equal to the receipt's last row: {yes(ok)}; apart: first {h2[0]['first']}, {h2[1]['first']}; second {h2[0]['second']}, {h2[1]['second']}; whole {h2[0]['whole']}, {h2[1]['whole']}; different answers given {h2[0]['given']}, {h2[1]['given']}")
        del net
    print("\n".join(notes))

head(None, "")
print(f"\nrunning time {time.time() - T0:.0f} s")
