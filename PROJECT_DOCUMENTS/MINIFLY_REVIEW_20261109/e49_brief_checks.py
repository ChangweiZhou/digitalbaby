"""Checks behind AUDIT_E49_LIMITED_INTERFACE_BRIEF_20261116.md (the E49 brief and e49_interface_spec.py at 7da5c453).

Read-only.  Nothing is trained, torch is not used, and no code of another branch is run: files are read with `git show`.
The forward is e49_interface_reader.py (written from the brief's text) beside e42_cells_reader.py; panels come from np_gru.py.

  A  the draws: the digests for three seeds beside the six the coordinator committed for seed 73129; other readings of
     the text; how many cells read each input value and how many readout units read each cell
  B  with both masks full the reader by index equals the E42 reader on E42's saved F10 of seed 73129 (both rules);
     with partial masks the reader by index equals the E42 reader on matrices multiplied by the masks
  C  multiply-adds and weights of every setting beside the brief's
  D  the start: sizes of the drives with the brief's scaling, on stand-in weights drawn from the stated distributions;
     the sizes of the kept weights beside what the optimizer does to a weight in one update, and E42's precedent
  E  what the masks limit: how well the K_IN values a cell reads identify the byte, for the 46 bytes the task uses
  F  the readings against E42's F10 of seed 73129: thresholds, levels, the four cases in updates and in multiply-adds
  G  what is on file for check 2, and the cost of a run

Run from the repository root:  PYTHONDONTWRITEBYTECODE=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e49_brief_checks.py [A B C D E F G]
"""
import hashlib, json, os, subprocess, sys
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_pt_tensors import tensors
from np_gru import split, panel, BOS
import e42_cells_reader as R42
import e49_interface_reader as R49

WREF = "ed4431f2"                                   # claude/gallant-dijkstra-i9zgsr, E42's worker
P = "MOD97_LOCAL_RULE_20261106/exp45/"
SEED, SEEDS = 73129, (73129, 93683, 116089)
U, D = 256, 512
DOSE, DREC = 24576, 24576 - 512                    # the last update at which a threshold can be recorded, three-row version
DTYPE = {"FloatStorage": "<f4", "LongStorage": "<i8", "BoolStorage": "?"}
COMMITTED = {("in", 64): "eb1dd9417ca1a282", ("out", 32): "e9c8783e0efa8d13", ("in", 16): "0e09b9ff0545e68d",
             ("out", 8): "3d8ba41006917f42", ("out", 256): "b27cc82e87e68c52", ("in", 512): "7b0701bb680e83f3"}
USED = [BOS] + list(b"pqvw") + list(b"0123456789") + list(b"+-><") + list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [ord("=")]
DIGITS = list(range(5, 15))                          # where the ten digits stand in USED
want = set(sys.argv[1:]) or set("ABCDEFG")


def show(path, ref=WREF):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def load(path, ref=WREF):
    return {name.lstrip("/"): np.frombuffer(raw, dtype=DTYPE[stype]).reshape(shape).copy() for name, (stype, shape, raw) in tensors(show(path, ref)).items()}


def length_items(k):
    task = ("add", 89, b"p")
    LP = panel(task, split(task, .5, SEED)[1], SEED, ks=(12, 40), n=300)[k]
    return np.array([[BOS] + p + a[:-1] for p, a in LP], dtype=np.int64), np.array([a for _, a in LP], dtype=np.int64)


def standin(s):
    """Weights drawn from the distributions torch's constructors use for this network (not torch's own values)."""
    g = np.random.default_rng(s)
    u = lambda shape, b: g.uniform(-b, b, shape).astype(np.float32)
    W = {"book.weight": g.standard_normal((258, 8)).astype(np.float32)}
    W["book.weight"][BOS] = 0
    W["l1.weight"], W["l1.bias"] = u((D, 8), 8 ** -.5), u(D, 8 ** -.5)
    W["rnn.weight_ih_l0"], W["rnn.weight_hh_l0"] = u((3 * U, D), 1 / 16), u((3 * U, U), 1 / 16)
    W["rnn.bias_ih_l0"], W["rnn.bias_hh_l0"] = u(3 * U, 1 / 16), u(3 * U, 1 / 16)
    W["lm.weight"], W["lm.bias"] = u((D, U), 1 / 16), u(D, 1 / 16)
    W["l2.weight"], W["l2.bias"] = u((D, D), D ** -.5), u(D, D ** -.5)
    W["l3.weight"], W["l3.bias"] = u((257, D), D ** -.5), u(257, D ** -.5)
    return W


def code(W, bytes_=USED):
    return np.tanh(W["book.weight"][bytes_] @ W["l1.weight"].T + W["l1.bias"]).astype(np.float64)        # (bytes, 512)


def reach(curve, key, thr):
    v = [(r["step"], r[key]) for r in curve]
    return next((v[i][0] for i in range(len(v) - 2) if all(v[i + j][1] >= thr for j in range(3))), None)


level = lambda curve, key: float(np.mean([r[key] for r in curve[-3:]]))

# ------------------------------------------------------------------------------------------------ A
if "A" in want:
    print("A. the draws")
    for seed in SEEDS:
        row = []
        for side, k in (("in", 64), ("out", 32), ("in", 16), ("out", 8), ("in", 512), ("out", 256)):
            v = R49.draw_input(seed, k) if side == "in" else R49.draw_readout(seed, k)
            dg = R49.digest16(v)
            mark = "" if seed != SEED else (" =" if dg == COMMITTED[(side, k)] else " DIFFERS")
            row.append(f"{'K_IN' if side == 'in' else 'K_OUT'} {k}: {dg}{mark}")
        print(f"   seed {seed}: " + "; ".join(row))
    print("   ('=' marks a digest equal to the one in OUTPUT_E49_SPEC_MAC.txt; the full masks do not depend on the seed)")
    v = R49.draw_input(SEED, 16)
    compact = hashlib.sha256(json.dumps(v, separators=(",", ":")).encode()).hexdigest()[:16]
    rng = Random(f"e49|input|{SEED}|16"); unsorted = [rng.sample(range(D), 16) for _ in range(U)]
    percell = [sorted(Random(f"e49|input|{SEED}|16").sample(range(D), 16)) for _ in range(U)]
    flat = hashlib.sha256(json.dumps([j for r in v for j in r]).encode()).hexdigest()[:16]
    print(f"   other readings, K_IN 16, seed {SEED}: json with compact separators {compact}; lists not sorted {R49.digest16(unsorted)}; "
          f"one flat list {flat}; a new generator for every cell {R49.digest16(percell)} (every cell would read the same values)")
    print("   the same draws serve the one-sided settings: in16 uses I16's input lists and out8 uses I16's readout lists (the strings do not name the other side)")
    for seed in SEEDS:
        for k_in, k_out in ((64, 32), (16, 8)):
            F = R49.Interface(seed, k_in, k_out)
            ci, co = F.mask_in.sum(0), F.mask_out.sum(0)       # cells that read an input value; readout units that read a cell
            print(f"   seed {seed}, K_IN {k_in:2d}: an input value is read by {ci.min()} to {ci.max()} cells (median {int(np.median(ci))}, expected {U * k_in / D:.0f}; read by none: {(ci == 0).sum()});"
                  f" K_OUT {k_out:2d}: a cell is read by {co.min()} to {co.max()} readout units (median {int(np.median(co))}, expected {D * k_out / U:.0f})")
    p0 = (1 - 16 / D) ** U
    print(f"   chance that a given input value is read by no cell at K_IN 16: {p0:.5f}; that at least one of the 512 is, for a seed: {1 - (1 - p0) ** D:.2f}")

# ------------------------------------------------------------------------------------------------ B
if "B" in want:
    print("B. the reader by index against the E42 reader, on E42's saved F10 of seed 73129 (300 items at 12 letters, 300 at 40)")
    full = R49.Interface(SEED, D, U)
    for rule in ("S", "B"):
        T = load(f"{P}e42/e42_F10_{rule}_{SEED}.pt")
        assert (T["periods"] == 1).all() and T["mask"].sum(1).tolist() == [11] * U
        mask = T["mask"].astype(bool)
        S = R49.Period1(mask)
        for k in (12, 40):
            x, tgt = length_items(k)
            Hs, lg = R42.forward(T, S, x)
            dH = dL = 0.0; same = True; right = 0
            for a in range(0, len(x), 50):
                Hi, li = R49.forward(T, mask, full, x[a:a + 50])
                dH = max(dH, float(np.abs(Hi - Hs[a:a + 50]).max())); dL = max(dL, float(np.abs(li - lg[a:a + 50]).max()))
                same &= bool((li.argmax(-1) == lg[a:a + 50].argmax(-1)).all()); right += int((li[:, -3:].argmax(-1) == tgt[a:a + 50]).all(1).sum())
            print(f"   F10 {rule}, {k} letters, full masks: largest difference of a state {dH:.1e}, of a logit {dL:.1e}; every predicted byte the same {same}; "
                  f"whole answers right {right}/300 by index, {int((lg[:, -3:].argmax(-1) == tgt).all(1).sum())}/300 by the E42 reader")
        x, tgt = length_items(12)
        for name in ("I64", "I16", "in16", "out8"):
            F = R49.Interface(SEED, *R49.SETTINGS[name])
            Wm = dict(T)
            Wm["rnn.weight_ih_l0"] = T["rnn.weight_ih_l0"] * np.tile(F.mask_in, (3, 1)); Wm["lm.weight"] = T["lm.weight"] * F.mask_out
            Hs, lg = R42.forward(Wm, S, x[:100])
            Hi, li = R49.forward(T, mask, F, x[:100])
            Ws = F.start(T)                                    # the brief's factor applied to the trained weights: not the job, a scale check only
            sc = int((R42.forward(Ws, S, x)[1][:, -3:].argmax(-1) == tgt).all(1).sum())
            un = int((R42.forward(Wm, S, x)[1][:, -3:].argmax(-1) == tgt).all(1).sum())
            print(f"   F10 {rule}, {name:4s} masks put on the trained weights: by index against masked matrices, largest difference of a state {np.abs(Hi - Hs).max():.1e}, of a logit {np.abs(li - lg).max():.1e};"
                  f" nothing retrained, whole answers right {un}/300 (with the start's factors {sc}/300)", flush=True)

# ------------------------------------------------------------------------------------------------ C
if "C" in want:
    print("C. multiply-adds for one item at one position, and weights")
    BRIEF = {"F10": (930560, 524288, None), "I64": (471808, 65536, 477201), "I16": (422656, 16384, 428049), "in16": (549632, 143360, 555025), "out8": (803584, 397312, 808977)}
    ref = R49.Interface(SEED, D, U).counts()
    for name, (k_in, k_out) in R49.SETTINGS.items():
        c = R49.Interface(SEED, k_in, k_out).counts()
        ok = (c["multiply_adds"], c["interface"]) == BRIEF[name][:2] and (BRIEF[name][2] in (None, c["weights"]))
        print(f"   {name:4s} K_IN {k_in:3d} K_OUT {k_out:3d}: {c['multiply_adds']:,} (interface {c['interface']:,}, cells {c['cells']:,}, the rest {c['fixed']:,}); weights {c['weights']:,}; "
              f"a cell with its share of the interface {c['a_cell']:.0f}; of F10's multiply-adds {c['multiply_adds'] / ref['multiply_adds']:.3f}; equal to the brief's {ok}")
    print(f"   the rest: l1 on the byte's row {8 * D:,} + l2 {D * D:,} + l3 {257 * D:,} = {8 * D + D * D + 257 * D:,}; of it, l2 reads all 512 values of the byte's code for each of the 512 readout units")
    print(f"   E42's receipt counts weights, {ref['weights']:,} for F10 (weight_count 935,953), and weight-items = weights x 64 x updates")

# ------------------------------------------------------------------------------------------------ D
if "D" in want:
    print("D. the start, on stand-in weights from the stated distributions (three draws), 300 items at 12 letters")
    x, _ = length_items(12)
    C = R42.Cells(SEED, N=10)
    sd = lambda a: float(np.sqrt((a.astype(np.float64) ** 2).mean()))
    acc = {n: [] for n in ("F10", "I64", "I16")}
    for s in range(3):
        W0 = C.start(standin(s))
        z1 = np.tanh(W0["book.weight"][x] @ W0["l1.weight"].T + W0["l1.bias"])
        for name in acc:
            F = R49.Interface(SEED, *R49.SETTINGS[name])
            W = F.start(W0)
            Hs, lg = R42.forward(W, C, x)
            gi = z1 @ W["rnn.weight_ih_l0"].T                                           # input drive without its bias, (B, L, 768)
            prev = np.concatenate([np.zeros_like(Hs[:, :1]), Hs[:, :-1]], 1)
            gh = prev @ (W["rnn.weight_hh_l0"] * C.mask3()).T
            out = Hs @ W["lm.weight"].T
            percell = (gi.astype(np.float64) ** 2).mean((0, 1)).reshape(3, U).mean(0)   # a cell's own mean squared input drive
            perunit = (out.astype(np.float64) ** 2).mean((0, 1))
            acc[name].append((sd(gi), sd(gh), sd(Hs), sd(out), sd(lg), percell.std() / percell.mean(), perunit.std() / perunit.mean(),
                              float(np.abs(W["rnn.weight_ih_l0"][np.tile(F.mask_in, (3, 1))]).mean()), float(np.abs(W["lm.weight"][F.mask_out]).mean())))
    for name, rows in acc.items():
        m = np.mean(rows, 0)
        print(f"   {name:4s}: root mean square of the input drive {m[0]:.3f}, of the recurrent drive {m[1]:.3f}, of a state {m[2]:.3f}, of lm's output {m[3]:.3f}, of a logit {m[4]:.3f}; "
              f"spread over cells of a cell's mean squared input drive {m[5]:.2f} of the mean; over readout units of lm's output {m[6]:.2f}; mean size of a kept weight: input map {m[7]:.4f}, lm {m[8]:.4f}")
    print("   by the distributions: a weight uniform in +-1/16 has variance 1/768 and mean size 1/32; the variance of a drive over K kept weights times (512 / K) or (256 / K) is that of the dense drive")
    for k, n in ((64, D), (16, D), (32, U), (8, U)):
        f = (n / k) ** .5
        print(f"   {k:2d} of {n}: factor {f:.2f}; mean size of a kept weight {f / 32:.4f}; one update of 0.001 is {0.001 / (f / 32):.3f} of it (dense: {0.001 * 32:.3f}); the decay removes 0.00025 of a weight in an update whatever its size")
    for rule in ("S", "B"):
        T = load(f"{P}e42/e42_F10_{rule}_{SEED}.pt")
        m3 = np.tile(T["mask"].astype(bool), (3, 1))
        print(f"   E42's F10 {rule} after 24,576 updates: mean size of a weight of the input map {np.abs(T['rnn.weight_ih_l0']).mean():.4f} (started near 0.0313), of lm {np.abs(T['lm.weight']).mean():.4f} (0.0313), "
              f"of a kept recurrent weight {np.abs(T['rnn.weight_hh_l0'][m3]).mean():.4f} (started near {(U / 11) ** .5 / 32:.4f})")
    for seed in SEEDS:
        F = R49.Interface(seed, 16, 8)
        ci, co = F.mask_in.sum(0), F.mask_out.sum(0)
        print(f"   seed {seed}, I16: the error an input value receives from the cells has {ci.min() / 8:.2f} to {ci.max() / 8:.2f} of the dense variance (by the number of cells that read it); "
              f"the error a cell receives from the readout {co.min() / 16:.2f} to {co.max() / 16:.2f}")

# ------------------------------------------------------------------------------------------------ E
if "E" in want:
    NONLET = [i for i, b in enumerate(USED) if not 65 <= b <= 90]          # the 20 bytes the task tells apart
    LET = [i for i, b in enumerate(USED) if 65 <= b <= 90]                 # the 26 letters, which stand between the operands and carry nothing
    LABEL = np.array([NONLET.index(i) if i in NONLET else len(NONLET) for i in range(len(USED))])      # 21 classes: 'a letter' is one
    NAME = lambda i: "start" if USED[i] == BOS else chr(USED[i])
    print(f"E. what the K_IN values a cell reads say about the byte.  The task puts in {len(USED)} bytes: {len(NONLET)} it must tell apart (the start symbol, p q v w, ten digits, + - > <, =) and 26 letters")

    def among(d, rows):
        iu = np.triu_indices(len(rows), 1)
        return d[..., rows, :][..., :, rows][..., iu[0], iu[1]]

    def dist(X):                                                            # X (sets, bytes, K) -> (sets, bytes, bytes)
        return np.sqrt(((X[:, :, None] - X[:, None]) ** 2).sum(-1))

    def measures(d):
        """Per set of values: the smallest distance between two of the 20 bytes over the mean distance between two of them;
        the same among the ten digits; the smallest distance from a letter to one of the 20, over the first mean."""
        nn, dg = among(d, NONLET), among(d, DIGITS)
        cross = d[:, LET][:, :, NONLET].reshape(len(d), -1)
        return nn.min(1) / nn.mean(1), dg.min(1) / dg.mean(1), cross.min(1) / nn.mean(1)

    def linear_rule(X, steps=8000):
        """One linear rule on a set of values (with a constant), 21 classes, fitted by gradient descent on the cross-entropy.
        Returns for every set how many of the 46 bytes the rule puts in their class."""
        n = X.shape[1]
        A = np.concatenate([X, np.ones((len(X), n, 1))], 2)
        Y = np.eye(21)[LABEL]
        Wt = np.zeros((len(X), A.shape[2], 21)); m = np.zeros_like(Wt); v = np.zeros_like(Wt)
        for t in range(1, steps + 1):
            lg = A @ Wt
            lg -= lg.max(2, keepdims=True)
            p = np.exp(lg); p /= p.sum(2, keepdims=True)
            g = A.transpose(0, 2, 1) @ (p - Y) / n
            m = .9 * m + .1 * g; v = .999 * v + .001 * g * g
            Wt -= 0.05 * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
            if t % 200 == 0 and ((A @ Wt).argmax(2) == LABEL).all():
                break
        return ((A @ Wt).argmax(2) == LABEL).sum(1)

    def report(tag, Z):
        dfull = dist(Z[None])
        a, b, c = measures(dfull)
        dn = dfull[0][np.ix_(NONLET, NONLET)] + np.eye(len(NONLET)) * 1e9
        i, j = np.unravel_index(dn.argmin(), dn.shape)
        ll = among(dfull, LET)
        print(f"   {tag}, all 512 values: closest two of the 20 bytes ({NAME(NONLET[i])} and {NAME(NONLET[j])}) at {a[0]:.3f} of the mean distance between two of them; closest two digits at {b[0]:.3f} of the digits' mean; "
              f"closest letter to one of the 20 at {c[0]:.3f}; two letters are on average {ll.mean() / among(dfull, NONLET).mean():.3f} apart on the same scale")
        for k_in in (64, 16):
            F = R49.Interface(SEED, k_in, U)
            X = Z[:, F.in_idx].transpose(1, 0, 2)
            dX = dist(X)
            a, b, c = measures(dX)
            distinct = bool((among(dX, list(range(len(USED)))) > 1e-6).all())
            lin = linear_rule(X)
            print(f"      the {k_in} values each cell reads (seed 73129's draw; least over the 256 cells, median): closest two of the 20 at {a.min():.3f}, {np.median(a):.3f}; closest two digits at {b.min():.3f}, {np.median(b):.3f}; "
                  f"closest letter to one of the 20 at {c.min():.3f}, {np.median(c):.3f}; no two of the 46 bytes coincide on any cell {distinct}; a linear rule on the cell's values puts all 46 bytes in their class on {int((lin == len(USED)).sum())} of 256 cells (least {lin.min()} of 46)", flush=True)
        g = np.random.default_rng(49)
        for k_in in (16, 8, 4, 2, 1):
            idx = np.array([g.choice(D, k_in, replace=False) for _ in range(1024)])
            a, b, c = measures(dist(Z[:, idx].transpose(1, 0, 2)))
            print(f"      1,024 other random sets of {k_in:2d} values: closest two of the 20 at least {a.min():.3f}, median {np.median(a):.3f}; closest two digits at least {b.min():.3f}, median {np.median(b):.3f}; "
                  f"sets on which two of the 20 are closer than 0.05 of the mean: {int((a < 0.05).sum())}")

    for rule in ("S", "B"):
        report(f"E42's F10 {rule}, trained", code(load(f"{P}e42/e42_F10_{rule}_{SEED}.pt")))
    for s in range(2):
        report(f"a stand-in start (draw {s})", code(standin(s)))
    print("   for comparison, a code in which a value stands for one byte: each used byte turns on 11 values of its own (46 x 11 = 506 of 512) and no other value")
    Zs = np.zeros((len(USED), D)); own = np.arange(506).reshape(len(USED), 11)
    for b in range(len(USED)):
        Zs[b, own[b]] = 1
    for k_in in (64, 16):
        for seed in SEEDS:
            F = R49.Interface(seed, k_in, U)
            seen = np.array([[bool(Zs[b, F.in_idx[c]].any()) for b in range(len(USED))] for c in range(U)])
            s20, s10 = seen[:, NONLET].sum(1), seen[:, DIGITS].sum(1)
            print(f"      seed {seed}: a cell reading {k_in} values receives a non-zero input for {s20.min()} to {s20.max()} of the 20 bytes (median {int(np.median(s20))}) and for {s10.min()} to {s10.max()} of the ten digits (median {int(np.median(s10))}); "
                  f"cells that receive something for every digit: {int((s10 == 10).sum())} of 256")

# ------------------------------------------------------------------------------------------------ F
if "F" in want:
    print("F. the readings against E42's F10 of seed 73129 (three-row version; a threshold can be recorded up to update 24,064)")
    ref = {}
    for rule in ("S", "B"):
        J = json.loads(show(f"{P}e42/e42_F10_{rule}_{SEED}.json"))
        cv = J["curve"]
        ref[rule] = {m: reach(cv, m, 80) for m in ("A", "R")}
        lv = {m: level(cv, m) for m in ("A", "R", "len12", "len40")}
        ref[rule]["lv"] = lv
        print(f"   F10 {rule}: updates to 50, 80, 90 on A {[reach(cv, 'A', t) for t in (50, 80, 90)]}, on R {[reach(cv, 'R', t) for t in (50, 80, 90)]}; "
              f"levels A {lv['A']:.1f}, R {lv['R']:.1f}, 12 letters {lv['len12']:.1f}, 40 letters {lv['len40']:.1f}; a setting is *lower* below A {lv['A'] - 15:.1f}, R {lv['R'] - 15:.1f}, 40 letters {lv['len40'] - 15:.1f}; "
              f"CPU-hours {(J['cpu_updates'] + J['cpu_scoring']) / 3600:.2f}")
    c0 = R49.Interface(SEED, D, U).counts()["multiply_adds"]
    for name in ("I64", "I16", "in16", "out8"):
        rho = R49.Interface(SEED, *R49.SETTINGS[name]).counts()["multiply_adds"] / c0
        for rule in ("S", "B"):
            for m in ("A", "R"):
                x0 = ref[rule][m]
                up = f"falls below {2 * x0 / 3:,.0f}, rises above {1.5 * x0:,.0f}" + ("" if 1.5 * x0 < DREC else " (beyond the last recordable update: cannot be recorded)")
                up_nr = "rises" if DREC / x0 >= 1.5 else "not decided"
                ma_r = 1.5 * x0 / rho
                ma = f"falls below {2 * x0 / (3 * rho):,.0f}, rises above {ma_r:,.0f}" + ("" if ma_r < DREC else " (beyond the last recordable update: cannot be recorded)")
                ma_nr = "rises" if DREC * rho / x0 >= 1.5 else "not decided"
                print(f"   {name:4s} {rule} on {m} (F10 at {x0:,}; multiply-adds {rho:.3f} of F10's): in updates {up}; not reached: {up_nr} (D/x {DREC / x0:.2f}). "
                      f"In multiply-adds, as updates of the setting: {ma}; not reached: {ma_nr} ({DREC * rho / x0:.2f})")
    print("   the same configuration on E42's three seeds, where on file (updates to 80, three-row version):")
    for cfg, rule in (("G", "S"), ("P8", "S"), ("P8", "B"), ("F10", "S"), ("F10", "B")):
        row = []
        for seed in SEEDS:
            try:
                J = json.loads(show(f"{P}e42/e42_{cfg}_{rule}_{seed}.json"))
            except subprocess.CalledProcessError:
                row.append(f"{seed}: not on file"); continue
            cv = J["curve"]
            row.append(f"{seed}: A {reach(cv, 'A', 80)}, R {reach(cv, 'R', 80)}, levels A {level(cv, 'A'):.1f} R {level(cv, 'R'):.1f} 40 letters {level(cv, 'len40'):.1f}" + ("" if J.get("complete") else " (not complete)"))
        print(f"      {cfg} {rule}: " + " | ".join(row))

# ------------------------------------------------------------------------------------------------ G
if "G" in want:
    print("G. what is on file for check 2")
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", WREF, "--", f"{P}e42_qual"], capture_output=True, check=True).stdout.decode().split()
    dry = sorted({os.path.basename(f) for f in out if "dry2048" in f and f.endswith(".json") and not f.endswith(("_answers.json", "_structure.json"))})
    print(f"   dry runs of 2,048 updates on E42's worker branch at {WREF}: {dry}")
    J = json.loads(show(f"{P}e42_qual/check8_73129_runs/e42_F10_S_dry2048_{SEED}.json"))
    print(f"   e42_F10_S_dry2048_{SEED}: tensors_sha256 {J['tensors_sha256'][:16]}..., rows {len(J['curve'])}, last row A {J['curve'][-1]['A']:.1f} R {J['curve'][-1]['R']:.1f}; rule {J['rule']}; a dry run of F10 under backpropagation: {'on file' if any('F10_B' in f for f in dry) else 'not on file'}")
