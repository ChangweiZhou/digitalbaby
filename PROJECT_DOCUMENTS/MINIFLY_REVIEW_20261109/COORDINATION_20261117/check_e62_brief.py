"""Check of the brief TASK_E62_THE_NUMBER_OF_CONNECTIONS_OF_A_CELL_20261118.md (revision 1): every figure quoted in
CHECK_OF_TASK_E62_BRIEF_20261118.md.

Run from the repository's root:   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/check_e62_brief.py
Standard library and numpy only; writes no file. It reads, with `git show`, the record's two receipts and e42_cells.py from the default
branch (the remote branch whose name ends in minifly-consolidated-20261023, at e647e7d7); no code of the repository is executed.

Section A re-implements e42_cells.neighbourhood and e42_cells.draws (connections only; the periods and phases use generators of their
own and do not touch the generator of the connections) and checks the re-implementation against the digests that e42_cells.py itself
asserts for seed 73129 (its self-check 9). Section B computes the graph figures of the ten settings. Section C recomputes the record's
figures from the two receipts. Section D recomputes the count of weights that take part and the cost forecast.
"""
import hashlib
import json
import subprocess
from random import Random

import numpy as np

D = [r for r in subprocess.run(["git", "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin/"], capture_output=True, text=True,
                               check=True).stdout.split() if r.endswith("/minifly-consolidated-20261023")][0]       # the default branch
M = "MOD97_LOCAL_RULE_20261106"
SEED = 73129
U, SIDE, RADIUS = 256, 16, 3


def show(path):
    return subprocess.run(["git", "show", f"{D}:{path}"], capture_output=True, check=True).stdout


# ---------------------------------------------------------------- A. the draws, copied from e42_cells.py
def neighbourhood(i, wide=False):
    if wide:
        return [j for j in range(U) if j != i]
    r, c = divmod(i, SIDE)
    return sorted({((r + dr) % SIDE) * SIDE + (c + dc) % SIDE for dr in range(-RADIUS, RADIUS + 1) for dc in range(-RADIUS, RADIUS + 1)} - {i})


def connections(seed, N, wide=False):
    rc = Random(f"e42|connections|{seed}")
    return [[j for j in range(U) if j != i] if N == U - 1 else sorted(rc.sample(sorted(neighbourhood(i, wide)), N)) for i in range(U)]


dg16 = lambda v: hashlib.sha256(json.dumps(v).encode()).hexdigest()[:16]

src = show(f"{M}/e42_cells.py").decode()
print("A. e42_cells.py on the default branch, SHA-256:", hashlib.sha256(src.encode()).hexdigest()[:16])
want = "ecf98006edb8ad9b fde78220c57da3d4 dbe99af8352f428c 4a9feeeb77ff57d2"
assert want in src, "the digests of e42_cells.py's self-check 9 were not found in the file"
mine = " ".join([dg16(connections(SEED, n)) for n in (5, 10, 20)] + [dg16(connections(SEED, 10, wide=True))])
print("A. digests of the connections for N = 5, 10, 20 (neighbourhood) and 10 (all cells):", mine)
print("A. equal to the digests that e42_cells.py asserts for seed 73129:", mine == want)
assert mine == want
print("A. size of the neighbourhood:", sorted({len(neighbourhood(i)) for i in range(U)}))


# ---------------------------------------------------------------- B. the graph of each setting
def adjacency(A):
    """a[j, i] = True if cell i reads cell j (a step goes from j to the cells that read it); the diagonal is True (a cell reads itself)."""
    a = np.eye(U, dtype=bool)
    for i, s in enumerate(A):
        a[s, i] = True
    return a


def reach_within(a, k):
    """r[j, i]: cell i is reached from cell j within k steps (walks of length k with the self-connection = within k)."""
    r = np.eye(U, dtype=bool); ai = a.astype(np.int32)
    for _ in range(k):
        r = (r.astype(np.int32) @ ai) > 0
    return r


def closure(a):
    r = a.copy()
    for _ in range(10):                                  # 2^10 > 256
        r2 = (r.astype(np.int32) @ r.astype(np.int32)) > 0
        if (r2 == r).all():
            break
        r = r2
    return r


def weak_parts(a):
    s = a | a.T; seen = [False] * U; parts = 0
    for v in range(U):
        if seen[v]:
            continue
        parts += 1; stack = [v]; seen[v] = True
        while stack:
            x = stack.pop()
            for y in np.nonzero(s[x])[0]:
                if not seen[y]:
                    seen[y] = True; stack.append(int(y))
    return parts


SETTINGS = [(n, "all") for n in (2, 4, 8, 10, 16, 32, 64, 128)] + [(255, "all"), (10, "near")]
KS = (1, 2, 3, 4, 6, 8)
table = []
for N, choice in SETTINGS:
    wide = choice == "all"
    A = connections(SEED, N, wide)
    ok_count = all(len(s) == N and len(set(s)) == N and i not in s for i, s in enumerate(A))
    a = adjacency(A)
    unread = int(sum(1 for j in range(U) if not any(j in A[i] for i in range(U) if i != j)))
    reached = {}
    for k in KS:
        c = reach_within(a, k).sum(1)                      # for each cell j: the cells reached from it, itself included
        reached[k] = (int(c.min()), float(c.mean()), int(c.max()))
    diam = None
    r = np.eye(U, dtype=bool); ai = a.astype(np.int32)
    for k in range(1, 300):
        r = (r.astype(np.int32) @ ai) > 0
        if r.all():
            diam = k; break
        if k > U:
            break
    cl = closure(a)
    scc = cl & cl.T
    scc_sizes = sorted({tuple(np.nonzero(scc[v])[0]) for v in range(U)}, key=len, reverse=True)
    reach_all = int((cl.sum(1) == U).sum())                # cells from which every cell is reached
    reached_by_all = int((cl.sum(0) == U).sum())           # cells reached from every cell
    nb = [set(neighbourhood(i)) for i in range(U)]
    outside = sum(1 for i in range(U) for j in A[i] if j not in nb[i])
    named_outside = len({j for i in range(U) for j in A[i] if j not in nb[i]})
    readers = a.sum(1) - 1                                  # other cells reading cell j
    row = dict(N=N, choice=choice, exactly_N=ok_count, digest=dg16(A), unread=unread, reached=reached, every_from_every=diam,
               weak_parts=weak_parts(a), scc=len(scc_sizes), largest_scc=len(scc_sizes[0]), from_which_all=reach_all, reached_by_all=reached_by_all,
               outside=outside, named_outside=named_outside, readers_min=int(readers.min()), readers_max=int(readers.max()),
               start_scale=(U / (N + 1)) ** 0.5 if N != U - 1 else 1.0)
    table.append(row)

print("\nB. Graph of each setting, seed 73129. A step goes from a cell to the cells that read it; 'reached within k' counts the cell itself.")
print("B. setting | exactly N | digest | not read by another cell | readers min-max | reached within 1, 2, 3, 4, 6, 8 steps (min/mean/max) | "
      "every cell from every cell within | parts (weak) | strongly connected parts (largest) | cells reaching all | cells reached by all | "
      "connections outside the neighbourhood | different cells named outside | start scale")
for t in table:
    rs = "; ".join(f"{k}: {t['reached'][k][0]}/{t['reached'][k][1]:.1f}/{t['reached'][k][2]}" for k in KS)
    print(f"B. N={t['N']:>3} {t['choice']:<4} | {t['exactly_N']} | {t['digest']} | {t['unread']} | {t['readers_min']}-{t['readers_max']} | {rs} | "
          f"{t['every_from_every'] if t['every_from_every'] else 'none'} | {t['weak_parts']} | {t['scc']} ({t['largest_scc']}) | {t['from_which_all']} | "
          f"{t['reached_by_all']} | {t['outside']} of {U * t['N']} | {t['named_outside']} | {t['start_scale']:.4f}")

# the expected number of unread cells from all cells, for comparison: 256 (1 - N/255)^255
print("B. expected number of cells not read by another cell, 256 (1 - N/255)^255:",
      {n: round(U * (1 - n / 255) ** 255, 2) for n in (2, 4, 8, 10, 16)})
# nested draws? (is the set of N = 4 connections a subset of N = 8's, and so on)
for n1, n2 in ((2, 4), (4, 8), (8, 16), (16, 32)):
    a1, a2 = connections(SEED, n1, True), connections(SEED, n2, True)
    print(f"B. connections of N = {n1} contained in those of N = {n2}, in how many of the 256 cells:", sum(set(x) <= set(y) for x, y in zip(a1, a2)))
# N = 10 from all cells against the record's N = 10 from the neighbourhood: shared connections
aw, an = connections(SEED, 10, True), connections(SEED, 10, False)
print("B. connections shared by N = 10 from all cells and N = 10 from the neighbourhood:", sum(len(set(x) & set(y)) for x, y in zip(aw, an)), "of", 2560)

# ---------------------------------------------------------------- C. the record
print()
rec = {r: json.loads(show(f"{M}/exp45/e50/e50_C46-one_{r}_73129.json")) for r in "SB"}


def reached(curve, key, thr=80.0):
    for i in range(len(curve) - 2):
        if all(curve[i + d][key] >= thr for d in range(3)):
            return curve[i]["step"]
    return None


def single(curve, key, thr=80.0):
    return next((c["step"] for c in curve if c[key] >= thr), None)


def level(curve, u, key):
    i = [c["step"] for c in curve].index(u)
    return sum(curve[j][key] for j in (i - 2, i - 1, i)) / 3


for r, name in (("B", "exact gradient"), ("S", "sweep")):
    c = rec[r]["curve"]
    print(f"C. record, {name}: rows {len(c)}, last {c[-1]['step']}; N {rec[r]['N']}; to 80 on R {reached(c, 'R')} [{single(c, 'R')}]; "
          f"to 80 on A {reached(c, 'A')} [{single(c, 'A')}]; "
          f"levels A, R, T at 24,576: {level(c, 24576, 'A'):.1f}, {level(c, 24576, 'R'):.1f}, {level(c, 24576, 'taught'):.1f}; "
          f"at 49,152: {level(c, 49152, 'A'):.1f}, {level(c, 49152, 'R'):.1f}, {level(c, 49152, 'taught'):.1f}; "
          f"CPU-hours {(rec[r]['cpu_updates'] + rec[r]['cpu_scoring']) / 3600:.3f} (updates {rec[r]['cpu_updates'] / 3600:.3f}, "
          f"scoring {rec[r]['cpu_scoring'] / 3600:.3f}); CPU-s an update {rec[r]['cpu_updates'] / 49152:.4f}; a scored row {rec[r]['cpu_scoring'] / len(c):.2f}; "
          f"processes beside {rec[r]['processes_beside']}; weights taking part {rec[r]['counts']['weights_taking_part']}; "
          f"kept mean magnitude of the recurrent map at row 0 {c[0]['kept_mean_magnitude']['rnn.weight_hh_l0']:.5f}")
rS, rB = reached(rec["S"]["curve"], "R"), reached(rec["B"]["curve"], "R")
print(f"C. ratio of the sweep's updates to 80 on R to the exact gradient's: {rS} / {rB} = {rS / rB:.4f}")
print(f"C. 1.25 times the record's updates to 80 on R: exact gradient {1.25 * rB:.0f}, sweep {1.25 * rS:.0f}; 0.75 times: {0.75 * rB:.0f}, {0.75 * rS:.0f}")
# CPU-seconds of the updates up to the row at which 80 on R is reached (one of the brief's per-run figures)
for r in "BS":
    c = rec[r]["curve"]; u = reached(c, "R"); row = next(x for x in c if x["step"] == u)
    print(f"C. record {r}: CPU-seconds of the updates up to update {u}: {row['cpu_updates']:.0f}")

# the brief's 2.36 and 1.70 against the memo's table of E42 on three seeds (section 28.3, read from the working tree)
memo = open("PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md", encoding="utf-8").read()
sec = memo[memo.index("### 28.3 E42 on three seeds"):memo.index("### 28.4")]
rows = {}
for line in sec.splitlines():
    for key in ("`G`", "`F10`"):
        if line.startswith(f"| {key},"):
            cells = [x.strip() for x in line.strip("|").split("|")]
            rows[key] = [[int(v.replace(",", "")) for v in c.split(", ")] for c in cells[1:3]]
(gS, gB), (fS, fB) = rows["`G`"], rows["`F10`"]
print("C. E42, updates to 80 on A, seeds 73129, 93683, 116089 (memo 28.3): F10 / G under the sweep",
      [round(f / g, 3) for f, g in zip(fS, gS)], "under back-propagation", [round(f / g, 3) for f, g in zip(fB, gB)],
      "; ratio of the sums: sweep", round(sum(fS) / sum(gS), 3), "back-propagation", round(sum(fB) / sum(gB), 3))
print("C. E42 seed 73129 (memo, stage-2 table): W10 (from all cells, connections replaced) 17,408 against A10 13,824 under the sweep:",
      round(17408 / 13824, 3))

# ---------------------------------------------------------------- D. counts and cost
print()
c0 = rec["S"]["counts"]
base = c0["input_map"] + c0["lm"] + c0["l2"] + c0["l3"]
print(f"D. weights taking part without the cells: input map {c0['input_map']} + lm {c0['lm']} + l2 {c0['l2']} + l3 {c0['l3']} = {base}; "
      f"cells at N = 10: {c0['cells']} = 768 x 11 = {768 * 11}")
for N in (2, 4, 8, 10, 16, 32, 64, 128, 255):
    print(f"D. N = {N}: 560,128 + 768 (N + 1) = {base + 768 * (N + 1)}")
hS = (rec["S"]["cpu_updates"] + rec["S"]["cpu_scoring"]) / 3600; hB = (rec["B"]["cpu_updates"] + rec["B"]["cpu_scoring"]) / 3600
print(f"D. eighteen runs at the record's cost a run (nine of each rule): {9 * hS + 9 * hB:.1f} CPU-hours; limit 44 is {44 / (9 * hS + 9 * hB):.2f} times that")
print(f"D. weights file: {rec['S']['params_module']} parameters of the module, about {rec['S']['params_module'] * 4 / 1e6:.1f} MB of float32 a file before "
      f"the code table ({258 * 512 * 4 / 1e6:.2f} MB); three weight files a run, eighteen runs: about {18 * 3 * (rec['S']['params_module'] * 4 + 258 * 512 * 4) / 1e6:.0f} MB")
