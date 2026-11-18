"""Checks behind the audit of the E47 brief and its specification file at 7247682a (16 November 2026).

Read-only.  Nothing is trained, torch is not used, and no code of the default branch or of a worker's branch is run: files are
read with `git show`; everything computed here comes from e47_local_reader.py (a reader written from the brief's text),
e42_cells_reader.py and np_gru.py.

  A  the draws from the text: geometry, connections at 16 a side beside E42's, cells by period, the difference of periods across
     a connection, digests for three seeds
  B  the reader at 256 cells against the E42 reader, on E42's saved F10 and P8 of seed 73129 (states and logits)
  C  weights and multiply-adds at each size and arm beside the brief's table; the present network on the same basis
  D  time: how far every cell is from the interface, and, from the structure alone, which cells can take part in an answer
     (a state that depends on a given byte and can still change the logits at a given position), by arm and number of letters
  E  the time properties on the reader's forward with random recurrent weights: a changed byte against the structural reach;
     the far cells held at zero
  F  which of the far cells' recurrent weights can receive a gradient in training items (the output's 0.90)
  G  cost of the trial and the roster from the brief's CPU figures
  H  what is on file for the brief's check 2 (E42's dry runs of F10), and E42's first seed on period 16
  I  the sweep by index: the two lines of the brief's rule, written in numpy, against the masked matrix form (256 cells)
  J  the interface inside the patch, on E42's trained F10 of seed 73129: how many of the 512 input values a cell's weights rest on,
     how many cells a readout unit's weights rest on, and the score when all but the largest are removed (no retraining)
  K  other rules for periods by place at 4,096 cells (equal groups by distance, shorter longest period): the share of far cells
     that can take part, and the count of multiply-adds

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e47_brief_checks.py [A B C D E F G H I J K]
"""
import hashlib, json, os, subprocess, sys
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_pt_tensors import tensors
from np_gru import split, panel, load_weights, BOS, EOS, P
import e42_cells_reader as R42
import e47_local_reader as R

REF = "7247682a"                      # the brief
E42REF = "7fbdbdc2"                   # E42's worker branch, seed 73129 complete
SEEDS = (73129, 93683, 116089)
DTYPE = {"FloatStorage": "<f4", "LongStorage": "<i8", "BoolStorage": "?"}
want = set(sys.argv[1:]) or set("ABCDEFGHIJK")
digest = lambda obj: hashlib.sha256(json.dumps(obj).encode()).hexdigest()[:16]


def show(path, ref):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def load(path, ref):
    return {n.lstrip("/"): np.frombuffer(raw, dtype=DTYPE[st]).reshape(shape).copy() for n, (st, shape, raw) in tensors(show(path, ref)).items()}


def layout(k):
    """Positions in x for an item with k letters: x = [BOS, code, a1, a2, op, k letters, b1, b2, '=', c1, c2]."""
    names = ["BOS", "code", "a1", "a2", "op"] + [f"d{i}" for i in range(k)] + ["b1", "b2", "=", "c1", "c2"]
    return {n: i for i, n in enumerate(names)}


if "A" in want:
    print("A. the draws, from the brief's text")
    for side in (16, 32, 64):
        C = side * side
        sizes = {len(R.neighbourhood_fast(i, side)) for i in range(C)}
        far = sum(R.distance(i, side) > 0 for i in range(C))
        print(f"   {side} a side: {C} cells; neighbourhood sizes {sorted(sizes)}; interface {len(R.interface(side))} cells; far cells {far}; "
              f"farthest {max(R.distance(i, side) for i in range(C))} grid steps = {-(-max(R.distance(i, side) for i in range(C)) // R.RADIUS)} connections")
    for seed in SEEDS:
        print(f"   connections at 16 a side equal E42's (seed {seed}): {R.draw_connections(seed, 16) == R42.draw_connections(seed, 10)}")
    print("   size, arm | cells by period | mean of 1/period | mean difference in log2 period across a connection (the output: 0.59, 1.13 at 1,024; 0.26, 1.22 at 4,096)")
    for side in (32, 64):
        for arm in R.ARMS:
            for seed in (SEEDS if arm == "scatter" else SEEDS[:1]):
                S = R.Grid(seed, side, arm)
                byp = {int(q): int((S.p == q).sum()) for q in sorted(set(S.p.tolist()))}
                lg = np.log2(S.p)
                gap = float(np.abs(lg[:, None] - lg[S.idx[:, 1:]]).mean())
                print(f"   {S.C:5d} {arm:8s} seed {seed} | {byp} | {float((1 / S.p).mean()):.3f} | {gap:.2f}")
    print("   by distance from the interface under `place` (seed 73129): grid steps -> periods present")
    for side in (32, 64):
        S = R.Grid(73129, side, "place")
        print(f"   {S.C:5d}: " + "; ".join(f"{d}: {sorted(set(S.p[S.dist == d].tolist()))} ({int((S.dist == d).sum())} cells)" for d in sorted(set(S.dist.tolist())) if d))
    print("   digests of the draws (sha256 of json.dumps of the list, first 16 digits): connections | periods `place` | periods `scatter` | phases under `place`")
    for seed in SEEDS:
        for side in (16, 32, 64):
            A = R.draw_connections(seed, side)
            pp, ps = R.draw_periods(seed, side, "place"), R.draw_periods(seed, side, "scatter")
            print(f"   seed {seed} side {side}: {digest(A)} | {digest(pp)} | {digest(ps)} | {digest(R.draw_phases(seed, pp))}")
    A16, A32 = R.draw_connections(73129, 16), R.draw_connections(73129, 32)
    n16, n32 = R.neighbourhood_fast(0, 16), R.neighbourhood_fast(0, 32)
    print(f"   one string for every size: cell 0 takes places {[n16.index(j) for j in A16[0]]} of its sorted neighbourhood at 16 a side and {[n32.index(j) for j in A32[0]]} at 32")
    rp = Random("e42|phases|73129"); before = rp.getstate()[1][:3]; rp.randrange(1); after = rp.getstate()[1][:3]
    print(f"   a cell of period 1 draws its phase with randrange(1): the generator's state changes: {before != after}")

if "B" in want:
    print("B. at 256 cells: the reader by index against the E42 reader, on E42's saved networks of seed 73129")
    LP = panel(("add", 89, b"p"), split(("add", 89, b"p"), .5, 73129)[1], 73129, ks=(12, 40), n=300)
    for cfg, rule in (("F10", "S"), ("F10", "B"), ("P8", "S"), ("P8", "B")):
        T = load(f"{P}e42/e42_{cfg}_{rule}_73129.pt", E42REF)
        class S42:                                                                     # the structure the E42 reader's forward takes
            p, f, mask = T["periods"].astype(int), T["phases"].astype(int), T["mask"].astype(bool)
            K, variant, a, Wx = int(T["periods"].max()), "step", T.get("a"), None
            mask3 = staticmethod(lambda m=T["mask"].astype(bool): np.tile(m, (3, 1)))
        A = [[j for j in range(256) if S42.mask[i, j] and j != i] for i in range(256)]
        assert all(len(a) == 10 for a in A)
        S = R.Grid(73129, 16, A=A, p=list(S42.p), f=list(S42.f))
        Wh, bh = T["rnn.weight_hh_l0"], T["rnn.bias_hh_l0"]
        W = dict(T)
        W["Wc"] = np.stack([np.take_along_axis(Wh[g * 256:(g + 1) * 256], S.idx, 1) for g in range(3)])
        W["bc"], W["bfar"], W["a"] = bh.reshape(3, 256), None, T.get("a")
        line = []
        for k in (12, 40):
            x = np.array([[BOS] + p + a[:-1] for p, a in LP[k][:64]], dtype=np.int64)
            H42, lg42 = R42.forward(T, S42, x, form="afactor")
            H47, lg47, _ = R.forward(W, S, x)
            line.append(f"{k} letters: states differ by at most {float(np.abs(H42 - H47).max()):.1e}, logits by {float(np.abs(lg42 - lg47).max()):.1e}, same largest output at every position {bool((lg42.argmax(-1) == lg47.argmax(-1)).all())}")
        print(f"   {cfg} {rule} (periods {sorted(set(S42.p.tolist()))}): " + " | ".join(line), flush=True)

if "C" in want:
    print("C. weights and multiply-adds (one item, one position, forward)")
    brief = {(16, "one"): (935953, 930560, 930560), (32, "one"): (965905, 955904, 955904), (32, "place"): (971665, 955904, 936500), (32, "scatter"): (971665, 955904, 936500),
             (64, "one"): (1085713, 1057280, 1057280), (64, "place"): (1114513, 1057280, 960260), (64, "scatter"): (1114513, 1057280, 960260)}
    for (side, arm), b in brief.items():
        c = R.Grid(73129, side, arm).counts()
        mine = (c["weights"], c["multiply_adds"], round(c["with_periods"]))
        print(f"   {side * side:5d} {arm:8s} weights {mine[0]:,}  multiply-adds {mine[1]:,}  with periods {mine[2]:,}   equal to the brief's: {mine == b}")
    dense = 8 * 512 + 3 * 256 * 512 + 3 * 256 * 256 + 512 * 256 + 512 * 512 + 257 * 512
    biases = 512 + 2 * 768 + 512 + 512 + 257
    print(f"   the present network on the same basis (products only, as the rows above): {dense:,}; the brief's row says 1,122,048, which is {1122048 - dense:,} more "
          f"(the network's {biases:,} biases less one); weights 1,124,113")
    print("   the `step` variant gives every step of a period its own learned weight, so a cell of period p must add up its 11 sources' values with those weights at every step")
    print("   (11 multiply-adds a step) before its 33 products at the update; the brief's column counts the 33 / p alone:")
    for side in (32, 64):
        S = R.Grid(73129, side, "place"); c = S.counts()
        least = c["with_periods"] + 11 * int((S.p > 1).sum())
        print(f"      {S.C:5d} cells: the brief's {round(c['with_periods']):,}; with the sums {round(least):,} ({least / 930560 - 1:+.1%} against 256 cells; every cell every step {c['multiply_adds'] / 930560 - 1:+.1%})")
    c16, c64 = R.Grid(73129, 16, "one").counts(), R.Grid(73129, 64, "one").counts()
    print(f"   a cell: {3 * 11} multiply-adds; sixteen times the cells: {c64['multiply_adds'] / c16['multiply_adds'] - 1:+.1%} every step, "
          f"{R.Grid(73129, 64, 'place').counts()['with_periods'] / c16['multiply_adds'] - 1:+.1%} with periods; against the present network {c64['multiply_adds'] / dense - 1:+.1%}")
    print(f"   the interface: 3 x 256 x 512 + 512 x 256 = {3 * 256 * 512 + 512 * 256:,} multiply-adds, {2048} a cell of the patch, at every size; "
          f"share of a position's multiply-adds: {524288 / c16['multiply_adds']:.2f} at 256 cells, {524288 / c64['multiply_adds']:.2f} at 4,096")

if "D" in want:
    print("D. time")
    for side in (32, 64):
        S = R.Grid(73129, side, "one")
        hops = -(-S.dist // R.RADIUS)
        print(f"   {S.C} cells: far cells by connections from the interface: " + ", ".join(f"{h}: {int((hops == h).sum())}" for h in sorted(set(hops.tolist())) if h)
              + f"; a round trip takes twice that many positions at period 1")
        for gap, what in ((1, "from b2 to the first answer byte"), (5, "from a1 to the first answer byte, no letters"), (11, "the same, 6 letters"), (17, "the same, 12 letters"), (7, "from the start symbol, no letters"), (19, "from the start symbol, 12 letters")):
            ok = (2 * hops[S.dist > 0] <= gap)
            print(f"      {what} ({gap} positions): {float(ok.mean()):.2f} of the far cells are near enough for a round trip")
    print("   from the structure alone (connections, periods, phases): the share of far cells some state of which depends on the byte and can still change the logits")
    print("   size arm seed | a1 -> first answer byte, by letters 0, 4, 8, 12 | any prompt byte -> any answer byte, by letters 0, 4, 8, 12 | at 40 letters: a1 -> first answer byte; any -> any")
    for side in (32, 64):
        for arm in R.ARMS:
            for seed in (SEEDS if side == 32 else SEEDS[:1]):
                S = R.Grid(seed, side, arm)
                far = S.dist > 0
                a1, anyb = [], []
                for k in (0, 4, 8, 12, 40):
                    lay = layout(k); L = len(lay)
                    tp = R.taking_part(S, L, lay["a1"], lay["="])
                    a1.append(float(tp[far].mean()))
                    u = np.zeros(S.C, bool)
                    outs = [R.reach_to_output(S, L, T) for T in (lay["="], lay["c1"], lay["c2"])]
                    for s in range(lay["="] + 1):
                        d = R.reach_from_byte(S, L, s)
                        for o in outs:
                            u |= (d & o).any(0)
                    anyb.append(float(u[far].mean()))
                    if k == 12:
                        used12 = u.copy()
                    if k == 0:
                        used0 = u.copy()
                print(f"   {S.C:5d} {arm:8s} {seed} | " + " ".join(f"{v:.2f}" for v in a1[:4]) + " | " + " ".join(f"{v:.2f}" for v in anyb[:4]) + f" | {a1[4]:.2f}; {anyb[4]:.2f}", flush=True)
                if arm != "one":
                    print("        at 12 letters, any -> any, by period: " + ", ".join(f"{q}: {float(used12[far & (S.p == q)].mean()):.2f}" for q in (2, 4, 8, 16))
                          + "; without letters: " + ", ".join(f"{q}: {float(used0[far & (S.p == q)].mean()):.2f}" for q in (2, 4, 8, 16)))
    print("   updates of a cell in one training item (10 + k positions, k = 0..12 with equal chance; every phase with equal chance): share with none, one, two or more")
    for q in (8, 16):
        n = [sum(1 for u in range(f, 10 + k, q)) for k in range(13) for f in range(q)]
        print(f"      period {q:2d}: none {sum(v == 0 for v in n) / len(n):.2f}, one {sum(v == 1 for v in n) / len(n):.2f}, two or more {sum(v >= 2 for v in n) / len(n):.2f}; "
              f"none in an item without letters {sum(1 for f in range(q) if f >= 10) / q:.2f}")
    S = R.Grid(73129, 32, "one"); lay = layout(0)
    tp = R.taking_part(S, len(lay), lay["b2"], lay["="])
    print(f"   b2 -> the first answer byte (one position later), 1,024 cells: far cells that can take part {int(tp[S.dist > 0].sum())}; interface cells {int(tp[S.dist == 0].sum())}")

if "E" in want:
    print("E. the reader's forward with drawn recurrent weights (shared parts from E26's birth of seed 73129), 1,024 cells")
    Wb, _ = load_weights(P + "e26/e26_dev_S_73129.pt")
    rng = np.random.default_rng(47)
    LP = panel(("add", 89, b"p"), split(("add", 89, b"p"), .5, 73129)[1], 73129, ks=(12,), n=300)
    x = np.array([[BOS] + p + a[:-1] for p, a in LP[12][:16]], dtype=np.int64)
    for arm in R.ARMS:
        S = R.Grid(73129, 32, arm)
        W = dict(Wb)
        W["Wc"] = rng.uniform(-11 ** -0.5, 11 ** -0.5, (3, S.C, 11)).astype(np.float32)
        W["bc"] = rng.uniform(-1 / 16, 1 / 16, (3, S.C)).astype(np.float32)
        W["bfar"] = rng.uniform(-1 / 16, 1 / 16, (3, len(S.farx))).astype(np.float32)
        W["a"] = (S.inside() / S.p[None, :]).astype(np.float32) if S.K > 1 else None
        _, lg, H = R.forward(W, S, x, keep_all=True)
        t0 = 9
        x2 = x.copy(); x2[:, t0] = (x2[:, t0] + 1) % 256
        _, _, H2 = R.forward(W, S, x2, keep_all=True)
        differs = (H != H2).any(0)                                                   # (L, C)
        reach = R.reach_from_byte(S, x.shape[1], t0)
        first = np.where(differs.any(0), differs.argmax(0), -1)
        ok_dist = all(fi == -1 or fi >= t0 + -(-int(S.dist[i]) // R.RADIUS) for i, fi in enumerate(first))
        up = R.updates(S, x.shape[1])
        only_at_updates = bool((~(H[:, 1:] != H[:, :-1]).any(0) | up[1:]).all())
        _, lg_off, H_off = R.forward(W, S, x, far_off=True, keep_all=True)
        print(f"   {arm:8s}: a changed byte reaches a cell no sooner than its distance allows {ok_dist}; states that differ lie inside the structural reach {bool((differs <= reach).all())} "
              f"and fill {float(differs.sum() / max(1, reach.sum())):.3f} of it; changes only at update positions {only_at_updates}; "
              f"far cells zero when held {bool((H_off[:, :, S.farx] == 0).all())}")
        far = H[:, :, S.farx]                                                         # (B, L, far cells)
        mean = far.mean(0)                                                             # the same items' mean at each position
        _, lg_mean, _ = R.forward(W, S, x, far_hold=mean)
        print(f"             far cells' states at the last position: mean size {float(np.abs(far[:, -1]).mean()):.3f}, of which varying between items (standard deviation) {float(far[:, -1].std(0).mean()):.3f}; "
              f"logits move by at most {float(np.abs(lg - lg_off).max()):.3f} with the far cells at zero and {float(np.abs(lg - lg_mean).max()):.3f} with them at their mean over the items")

if "F" in want:
    print("F. far cells' recurrent weights that can receive a gradient from items of 0 to 12 letters (the output: 0.90 at 1,024 cells, `place`)")
    for side, arm in ((32, "place"), (32, "scatter"), (32, "one"), (64, "place"), (64, "one")):
        S = R.Grid(73129, side, arm)
        can = np.zeros((S.C, 11), bool)
        first_update = S.f.copy()
        for k in range(13):
            lay = layout(k); L = len(lay)
            inf = np.zeros((L, S.C), bool)
            for T in (lay["="], lay["c1"], lay["c2"]):
                inf |= R.reach_to_output(S, L, T)
            up = R.updates(S, L)
            for t in range(L):
                cells = np.nonzero(up[t] & inf[t])[0]
                for kk in range(int(S.p[cells].max()) if len(cells) else 0):
                    tau = t - kk - 1
                    if tau < 0:
                        break
                    live = (kk < S.p[cells])[:, None]
                    can[cells] |= live & (first_update[S.idx[cells]] <= tau)        # the source has a value by then
        far = S.dist > 0
        print(f"   {S.C:5d} {arm:8s}: {float(can[far].mean()):.2f} of the far cells' weights; far cells with none {int((~can[far].any(1)).sum())} of {int(far.sum())}"
              + ("" if arm == "one" else "; by period " + ", ".join(f"{q}: {float(can[far & (S.p == q)].mean()):.2f}" for q in (2, 4, 8, 16))))

if "G" in want:
    print("G. cost, from the brief's CPU-seconds for a forward and backward on the coordinator's machine (scoring not included)")
    mac = {(16, "one"): 0.085, (32, "one"): 0.271, (32, "place"): 0.391, (64, "one"): 1.206, (64, "place"): 1.576}
    mac[(32, "scatter")] = mac[(32, "place")]
    U = 24576
    hours = {k: v * U / 3600 for k, v in mac.items()}
    print("   CPU-hours of updates for one run of 24,576 updates: " + ", ".join(f"{s * s} {a} {h:.1f}" for (s, a), h in hours.items()))
    trial = 2 * hours[(32, "place")]
    roster = 2 * hours[(32, "one")] + 2 * hours[(32, "scatter")] + 2 * hours[(64, "one")] + 2 * hours[(64, "place")]
    print(f"   the trial (two runs at 1,024 cells, `place`): {trial:.1f}; with `one` beside it: {trial + 2 * hours[(32, 'one')]:.1f}")
    print(f"   the roster's eight further runs: {roster:.1f}, of which the four at 4,096 cells {2 * hours[(64, 'one')] + 2 * hours[(64, 'place')]:.1f}; at 1,024 cells alone {2 * hours[(32, 'one')] + 2 * hours[(32, 'scatter')]:.1f}")
    print("   the cloud host against the coordinator's machine, on E42's files (CPU-seconds an update in the receipt over CPU-seconds a forward and backward in OUTPUT_E42_SPEC_MAC.txt):")
    mac42 = {"G": 0.031, "F10": 0.035, "P8": 0.076}                                   # the output's line for A10 stands for F10 (the same layer with fixed connections)
    ratio = {}
    for cfg in mac42:
        c = json.loads(show(f"{P}e42/e42_{cfg}_S_73129.json", E42REF))["curve"][-1]
        ratio[cfg] = c["cpu_updates"] / U / mac42[cfg]
        print(f"      {cfg:4s}: cloud {c['cpu_updates'] / U:.3f} an update and {c['cpu_scoring'] / 96:.1f} a scored row; the coordinator's machine {mac42[cfg]:.3f}; ratio {ratio[cfg]:.1f}")
    lo, hi = min(ratio.values()), max(ratio.values())
    print(f"   at {lo:.1f} to {hi:.1f} times the coordinator's figures: the trial {lo * trial:.0f} to {hi * trial:.0f} CPU-hours of updates; one run at 4,096 cells {lo * hours[(64, 'one')]:.0f} to {hi * hours[(64, 'place')]:.0f}; "
          f"the roster's eight {lo * roster:.0f} to {hi * roster:.0f}, its four at 1,024 cells {lo * (2 * hours[(32, 'one')] + 2 * hours[(32, 'scatter')]):.0f} to {hi * (2 * hours[(32, 'one')] + 2 * hours[(32, 'scatter')]):.0f}")
    c = json.loads(show(f"{P}e42/e42_F10_S_73129.json", E42REF))["curve"][-1]
    row = c["cpu_scoring"] / 96
    print(f"   scoring: E42's F10 took {row:.1f} CPU-seconds a scored row on the cloud host, 96 rows a run; if a scored row grows as the update does (0.391 / 0.085 at 1,024 cells, `place`) "
          f"and is computed twice (with and without the far cells), a trial run adds {2 * row * 0.391 / 0.085 * 96 / 3600:.1f} CPU-hours of scoring")

if "H" in want:
    print("H. what is on file for the brief's check 2, and E42's first seed on period 16")
    ls = subprocess.run(["git", "ls-tree", "-r", "--name-only", E42REF, "--", f"{P}e42_qual/"], capture_output=True, text=True).stdout.split()
    dry = sorted(n.rsplit("/", 1)[1] for n in ls if "F10" in n and n.endswith(".json") and "structure" not in n and "answers" not in n)
    print(f"   E42's dry runs of F10 on its branch: {dry}")
    for cfg in ("P8", "P16", "F10", "G"):
        for rule in ("S", "B"):
            try:
                c = json.loads(show(f"{P}e42/e42_{cfg}_{rule}_73129.json", E42REF))["curve"]
            except subprocess.CalledProcessError:
                print(f"   E42 {cfg:4s} {rule}, seed 73129: no receipt on the branch"); continue
            first80 = next((r["step"] for i, r in enumerate(c) if all(q["R"] >= 80 for q in c[i:i + 3]) and len(c[i:i + 3]) == 3), None)
            print(f"   E42 {cfg:4s} {rule}, seed 73129: at 40 letters, mean of the last three rows {sum(r['len40'] for r in c[-3:]) / 3:.1f}; at 12 letters {sum(r['len12'] for r in c[-3:]) / 3:.1f}; "
                  f"first of three scored rows in a row at 80 or more on R (a plain count, not E41's table): {first80}")

if "I" in want:
    print("I. the sweep's backward by index against the masked matrix form (random values; float64)")
    rng = np.random.default_rng(4747)
    for side in (16, 32):
        S = R.Grid(73129, side, "one"); C = S.C; Bn = 5
        Wc = rng.uniform(-11 ** -0.5, 11 ** -0.5, (3, C, 11)); Wc[0, 0, 3] = 0.0     # one weight exactly zero: nothing crosses it
        g = rng.normal(size=(Bn, 3, C)); h = rng.normal(size=(Bn, C))
        scale = np.abs(Wc).mean()
        # by index, as the brief states the rule: each connection carries the error by the sign of its weight times one scale
        back = np.einsum("bgi,gik->bik", g, np.sign(Wc) * scale)
        gh = np.zeros((Bn, C)); np.add.at(gh, (slice(None), S.idx.reshape(-1)), back.reshape(Bn, -1))
        gw = np.einsum("bgi,bik->gik", g, h[:, S.idx])
        # the masked matrix (3 C x C), E42's form
        Wfull = np.zeros((3 * C, C)); mask = np.zeros((3 * C, C), bool)
        for gate in range(3):
            np.put_along_axis(Wfull[gate * C:(gate + 1) * C], S.idx, Wc[gate], 1)
            np.put_along_axis(mask[gate * C:(gate + 1) * C], S.idx, True, 1)
        scale_m = np.abs(Wfull[mask]).mean()
        gh_m = g.reshape(Bn, 3 * C) @ (np.sign(Wfull) * mask * scale_m)
        gw_m = np.einsum("bq,bj->qj", g.reshape(Bn, 3 * C), h)
        gw_back = np.stack([np.take_along_axis(gw_m[gate * C:(gate + 1) * C], S.idx, 1) for gate in range(3)])
        fwd = np.einsum("bik,gik->bgi", h[:, S.idx], Wc).reshape(Bn, 3 * C)
        print(f"   {C:5d} cells: entries inside the mask {int(mask.sum()):,} = 3 x cells x 11 {mask.sum() == 3 * C * 11}; scale equal {abs(scale - scale_m) < 1e-15}; "
              f"forward differs by {np.abs(fwd - h @ Wfull.T).max():.1e}; error sent back differs by {np.abs(gh - gh_m).max():.1e}; weight gradient differs by {np.abs(gw - gw_back).max():.1e}")

if "J" in want:
    print("J. the interface of E42's trained F10 (seed 73129, 256 cells): length-test items of 12 letters, whole answers, 300 items")
    LP = panel(("add", 89, b"p"), split(("add", 89, b"p"), .5, 73129)[1], 73129, ks=(12,), n=300)[12]
    x = np.array([[BOS] + p + a[:-1] for p, a in LP], dtype=np.int64); tgt = np.array([a for _, a in LP], dtype=np.int64)
    for rule in ("S", "B"):
        T = load(f"{P}e42/e42_F10_{rule}_73129.pt", E42REF)
        mask = T["mask"].astype(bool)
        A = [[j for j in range(256) if mask[i, j] and j != i] for i in range(256)]
        S = R.Grid(73129, 16, A=A, p=[1] * 256, f=[0] * 256)
        W = dict(T)
        W["Wc"] = np.stack([np.take_along_axis(T["rnn.weight_hh_l0"][g * 256:(g + 1) * 256], S.idx, 1) for g in range(3)])
        W["bc"], W["bfar"], W["a"] = T["rnn.bias_hh_l0"].reshape(3, 256), None, None

        def score(Wx):
            lg = R.forward(Wx, S, x)[1]
            return 100 * float((lg[:, -3:].argmax(-1) == tgt).all(1).mean())

        win = (T["rnn.weight_ih_l0"].reshape(3, 256, 512) ** 2).sum(0)               # (cell, input value): squared weight over the three gates
        wout = T["lm.weight"] ** 2                                                    # (readout unit, cell)
        need = lambda sq: np.array([int(np.searchsorted(np.cumsum(np.sort(r)[::-1]) / r.sum(), 0.9)) + 1 for r in sq])
        ni, no = need(win), need(wout)
        print(f"   {rule}: as trained {score(W):.1f}; input values that hold 90% of a cell's squared input weight: median {int(np.median(ni))} of 512 (range {ni.min()} to {ni.max()}); "
              f"cells that hold 90% of a readout unit's squared weight: median {int(np.median(no))} of 256 (range {no.min()} to {no.max()})")
        rows = []
        for keep in (256, 128, 64, 32, 16):
            top = np.argsort(-win, 1)[:, :keep]
            m_in = np.zeros((256, 512), bool); np.put_along_axis(m_in, top, True, 1)
            Wi = dict(W); Wi["rnn.weight_ih_l0"] = (T["rnn.weight_ih_l0"].reshape(3, 256, 512) * m_in[None]).reshape(768, 512)
            k_out = max(1, keep // 2)
            topo = np.argsort(-wout, 1)[:, :k_out]
            m_out = np.zeros((512, 256), bool); np.put_along_axis(m_out, topo, True, 1)
            Wo = dict(W); Wo["lm.weight"] = T["lm.weight"] * m_out
            Wb = dict(Wi); Wb["lm.weight"] = Wo["lm.weight"]
            rows.append(f"{keep} input values a cell: {score(Wi):.1f}; {k_out} cells a readout unit: {score(Wo):.1f}; both: {score(Wb):.1f}")
        print("      largest kept, the rest set to zero, nothing retrained: " + " | ".join(rows), flush=True)

if "K" in want:
    print("K. periods by place with a shorter longest period (equal groups of far cells, nearest first; seed 73129): share of far cells that can take part, any prompt byte -> any answer byte")
    print("   size, periods | by letters 0, 4, 8, 12 | lowest share among the groups at 12 letters | multiply-adds a position with the sums of the `step` variant")
    for side in (32, 64):
        far = sorted((i for i in range(side * side) if R.distance(i, side) > 0), key=lambda i: (R.distance(i, side), i))
        for periods in ((2, 4, 8, 16), (2, 4, 8), (2, 4), (2,)):
            p = [1] * (side * side)
            for n, i in enumerate(far):
                p[i] = periods[n * len(periods) // len(far)]
            S = R.Grid(73129, side, p=p)
            isfar = S.dist > 0
            row = []
            for k in (0, 4, 8, 12):
                lay = layout(k); L = len(lay)
                u = np.zeros(S.C, bool)
                outs = [R.reach_to_output(S, L, T) for T in (lay["="], lay["c1"], lay["c2"])]
                for b in range(lay["="] + 1):
                    d = R.reach_from_byte(S, L, b)
                    for o in outs:
                        u |= (d & o).any(0)
                row.append(float(u[isfar].mean()))
            low = min(float(u[isfar & (S.p == q)].mean()) for q in periods)
            cnt = S.counts()["with_periods"] + 11 * int((S.p > 1).sum())
            print(f"   {S.C:5d} {str(periods):14s} | " + " ".join(f"{v:.2f}" for v in row) + f" | {low:.2f} | {round(cnt):,}", flush=True)
