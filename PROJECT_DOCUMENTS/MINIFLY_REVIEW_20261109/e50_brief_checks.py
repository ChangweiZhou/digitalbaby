"""Checks behind AUDIT_E50_SPARSE_INPUT_CODE_BRIEF_20261116.md (the E50 brief and e50_sparse_code_spec.py at f8f62df6).

Read-only.  Nothing is trained, torch is not used, and no code of another branch is run: files are read with `git show`.
The code and the forward are e50_code_reader.py (written from the brief's text) beside the E42 and E49 readers.

  A  the code: the digest and the counts for three seeds beside the coordinator's for seed 73129; other readings of the
     text; the symbols that stand in the training stream; the amplitude from a committed starting network
  B  what a cell has input for, per setting and seed: bytes of the 46, of the 20, of the ten digits; cells per byte
  C  travel: connections between cells; how many bytes are within one and two connections of a cell; which cells'
     states depend on the second operand's last digit at `=`, from the connections and from the forward itself
  D  the start, on the starting network of seed 73129: the first-batch losses beside the coordinator's; sizes of the
     drives per setting beside E42's F10; how far one update can move a cell's input; the two forms of the forward
  E  multiply-adds a position per setting under two ways of counting
  F  the readings: E42's F10 of seed 73129; what can be recorded against `C-dense`; E49's partial receipts; cost
  G  the test of whether cells use what arrives through neighbours (the mechanics, on the starting network)

Run from the repository root:  PYTHONDONTWRITEBYTECODE=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e50_brief_checks.py [A B C D E F G]
"""
import hashlib, json, os, subprocess, sys
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_pt_tensors import tensors
from np_gru import split, render, DEV, BOS, EOS
import e42_cells_reader as R42
import e49_interface_reader as R49
import e50_code_reader as R50

WREF = "ed4431f2"                                   # claude/gallant-dijkstra-i9zgsr, E42's worker: F10's records
SREF = "8b48f76b"                                   # claude/e47-worker-stage-0-3d0ya6: a starting network of seed 73129
SPATH = "MOD97_LOCAL_RULE_20261106/exp45/e47_qual/check2_runs/dry/e47_16_one_S_dry2048_73129_start.pt"
EREF = "d9ec911a"                                   # claude/peaceful-noether-vcvrmf, E49's worker: the runner, its checks, partial receipts
EPATH = "MOD97_LOCAL_RULE_20261106/exp45/e49_qual/check2_73129_runs/e49_F10_S_dry2048_73129_start.pt"
P = "MOD97_LOCAL_RULE_20261106/exp45/"
SEED, SEEDS = 73129, (73129, 93683, 116089)
U, D, ON = 256, 512, 11
DOSE, DREC = 24576, 24576 - 512                    # the last update at which a threshold can be recorded, three-row version
DTYPE = {"FloatStorage": "<f4", "LongStorage": "<i8", "BoolStorage": "?"}
COMMITTED = "12111044f0f8bc70"                     # OUTPUT_E50_SPEC_MAC.txt, seed 73129
BRIEF_LOSS = {"C-dense": 5.597, "C64": 5.556, "C16": 5.609}
want = set(sys.argv[1:]) or set("ABCDEFG")
med = lambda v: (int(np.sort(v)[len(v) // 2 - 1]), int(np.sort(v)[len(v) // 2]))     # the two middle values of an even count
rms = lambda a: float(np.sqrt((np.asarray(a, np.float64) ** 2).mean()))


def show(path, ref):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def load(path, ref):
    return {name.lstrip("/"): np.frombuffer(raw, dtype=DTYPE[stype]).reshape(shape).copy() for name, (stype, shape, raw) in tensors(show(path, ref)).items()}


def start_network():
    """The starting network of seed 73129 as E47's worker committed it for the 16 x 16 arm `one`: the byte table, `l1`, the input
    map, `lm`, `l2` and `l3` as the constructor made them, and the kept recurrent weights in E42's connections with E42's scale."""
    T = load(SPATH, SREF)
    W = {k: T[k] for k in ("book.weight", "l1.weight", "l1.bias", "rnn.weight_ih_l0", "rnn.bias_ih_l0", "lm.weight", "lm.bias", "l2.weight", "l2.bias", "l3.weight", "l3.bias")}
    src = T["sources"]
    assert (src[:, 0] == np.arange(U)).all() and src[:, 1:].tolist() == R42.draw_connections(SEED, 10), "the file's connections are E42's"
    Wh = np.zeros((3 * U, U), np.float32)
    mask = np.zeros((U, U), bool)
    for i in range(U):
        mask[i, src[i]] = True
        for g in range(3):
            Wh[g * U + i, src[i]] = T["Wc"][g, i]
    W["rnn.weight_hh_l0"], W["rnn.bias_hh_l0"] = Wh, T["bc"].reshape(-1).copy()
    return W, mask


def stream_items(seed, n):
    """The first n items of the training stream of the first family (exp11_cell.stream: Random(seed + 77), up to 12 letters)."""
    tp = [(t, split(t, .5, seed)[0]) for t in DEV]
    rng = Random(seed + 77)
    flat = [(t, pr) for t, pairs in tp for pr in pairs]
    out = []
    while len(out) < n:
        order = list(range(len(flat)))
        rng.shuffle(order)
        for i in order:
            t, (a, b) = flat[i]
            k = rng.randint(0, 12)
            out.append(render(t, a, b, k, rng))
            if len(out) == n:
                break
    return out


def batch_tensor(items):
    """exp45.batch_tensor: rows padded with the start symbol; x, y, and the positions of y that are scored."""
    rows = [p + t for p, t in items]
    L = max(len(r) for r in rows) + 1
    data = np.full((len(rows), L), BOS, np.int64)
    for i, r in enumerate(rows):
        data[i, 1:len(r) + 1] = r
    x, y = data[:, :-1], data[:, 1:]
    mask = np.zeros(y.shape, bool)
    for i, (p, t) in enumerate(items):
        mask[i, len(p):len(p) + len(t)] = True
    return x, y, mask


def loss_of(logits, y, mask):
    lg = logits[mask].astype(np.float64)
    lg -= lg.max(1, keepdims=True)
    return float(-(lg[np.arange(len(lg)), y[mask]] - np.log(np.exp(lg).sum(1))).mean())


def plain_items(pairs, code=ord("p"), op=ord("+"), m=89):
    """Items without letters, as R's: x = [start, code, a1, a2, op, b1, b2, =, c1, c2]; b2 at position 6, `=` at 7."""
    rows = []
    for a, b in pairs:
        c = (a + b) % m
        rows.append([BOS, code] + list(f"{a:02d}".encode()) + [op] + list(f"{b:02d}".encode()) + [ord("=")] + list(f"{c:02d}".encode()))
    return np.array(rows, np.int64)


def reach(curve, key, thr):
    v = [(r["step"], r[key]) for r in curve]
    return next((v[i][0] for i in range(len(v) - 2) if all(v[i + j][1] >= thr for j in range(3))), None)


level = lambda curve, key: float(np.mean([r[key] for r in curve[-3:]]))
GROUPS = (("the ten digits", R50.DIGITS), ("+ - > <", R50.OPS), ("p q v w", R50.CODES), ("=", [R50.EQ]), ("start", [BOS]))

# ------------------------------------------------------------------------------------------------ A
if "A" in want:
    print("A. the code")
    for seed in SEEDS:
        c = R50.Code(seed, amplitude=1.0)
        owned = sum(o >= 0 for o in c.owner)
        per = sorted(len(v) for v in c.own.values())
        mark = "" if seed != SEED else (" = the coordinator's" if c.digest() == COMMITTED else " DIFFERS from the coordinator's")
        print(f"   seed {seed}: {len(c.used)} used symbols, {per[0]} to {per[-1]} values each, {owned} of {D} values owned, {D - owned} without an owner, no value with two owners (asserted); digest {c.digest()}{mark}")
    c = R50.Code(SEED, amplitude=1.0)
    alt = {}
    alt["the start symbol first"] = R50.Code(SEED, 1.0, used=[BOS] + R50.USED[:-1]).digest()
    vals = list(range(D)); Random(f"e50|code|{SEED}").shuffle(vals)
    own1 = [-1] * D
    for k, b in enumerate(R50.USED):
        for v in vals[ON * (k + 1): ON * (k + 2)]:
            own1[v] = b
    alt["symbol number k counted from 1 (the last symbol then has 6 values)"] = hashlib.sha256(json.dumps(own1).encode()).hexdigest()[:16]
    alt["the digest taken of the 46 lists of values, in the order of the symbols"] = R49.digest16([c.own[b] for b in R50.USED])
    alt["the same lists sorted"] = R49.digest16([sorted(c.own[b]) for b in R50.USED])
    alt["the owner written as the symbol's number, not its byte"] = R49.digest16([R50.USED.index(o) if o >= 0 else -1 for o in c.owner])
    alt["the SHA-256 of the table's bytes (float32, amplitude 3.2387)"] = hashlib.sha256(R50.Code(SEED).table().tobytes()).hexdigest()[:16]
    for k, v in alt.items():
        print(f"   another reading, seed {SEED}: {k}: {v}")
    print("   the brief's text gives the digest's first eight digits and not what is hashed; the formula above (json.dumps of the list of 512 owners, the owning byte or -1) is the specification file's")
    # the symbols in the stream
    items = stream_items(SEED, 64 * 1024)
    x_all = set()
    ends_inside = 0
    for a in range(0, len(items), 64):
        x, y, mask = batch_tensor(items[a:a + 64])
        x_all |= set(np.unique(x).tolist())
        last_scored = np.array([np.nonzero(m)[0].max() for m in mask])
        ends_inside += int(sum((x[i, :last_scored[i] + 1] == EOS).any() for i in range(len(x))))
    print(f"   the first 1,024 batches of seed {SEED}'s stream: input symbols {len(x_all)}; all among the 46 or the end symbol {x_all <= set(R50.USED) | {EOS}}; every one of the 46 occurs {set(R50.USED) <= x_all};"
          f" the end symbol at or before a scored position in {ends_inside} items")
    x, y, mask = batch_tensor(items[:64])
    pad = int(((x == BOS).sum(1) - 1).sum())
    print(f"   in the first batch, {pad} positions hold the start symbol as padding after an item's end (it has a code of its own, so padding now drives the cells; nothing is scored there)")
    # the amplitude
    T = load(SPATH, SREF)
    z1 = np.tanh(T["book.weight"][R50.USED].astype(np.float64) @ T["l1.weight"].T + T["l1.bias"])
    sq = (z1 ** 2).sum(1)
    print(f"   the amplitude, from the starting network of seed {SEED} that E47's worker committed ({SREF}): the code tanh(l1(byte table)) of the 46 used bytes has mean squared length {sq.mean():.3f};"
          f" its root over 11 is {np.sqrt(sq.mean() / ON):.4f} (the brief: 3.2387)")
    print(f"      per byte the squared length runs from {sq.min():.1f} (the start symbol, whose row of the byte table is zero) to {sq.max():.1f}; without the start symbol the mean is {sq[:-1].mean():.1f};"
          f" in the fixed code every used byte has {ON * 3.2387 ** 2:.1f}")
    print(f"      a value of the present code is at most 1 in size; a turned-on value of the fixed code is 3.2387.  Sum of the sizes of a byte's values: present code {np.abs(z1).sum(1).mean():.1f} on average, fixed code {ON * 3.2387:.1f}")

# ------------------------------------------------------------------------------------------------ B
if "B" in want:
    print("B. what a cell has input for (least, the two middle values, most over the 256 cells)")
    for seed in SEEDS:
        c = R50.Code(seed, amplitude=1.0)
        for nm in ("C64", "C16"):
            F = R49.Interface(seed, *R50.SETTINGS[nm])
            m = c.reads(F)                                               # (256 cells, 46 bytes)
            col = lambda group: [c.used.index(b) for b in group]
            k46, k20, k10 = (m > 0).sum(1), (m[:, col(R50.NEEDED)] > 0).sum(1), (m[:, col(R50.DIGITS)] > 0).sum(1)
            f = lambda v: f"{v.min()} to {v.max()} (middle {med(v)[0]}, {med(v)[1]})"
            print(f"   seed {seed}, {nm}: of the 46 used bytes {f(k46)}; of the 20 {f(k20)}; of the ten digits {f(k10)}; cells with input for no digit {int((k10 == 0).sum())}, for at most one of the 20 {int((k20 <= 1).sum())},"
                  f" for every digit {int((k10 == 10).sum())}")
            if nm == "C16":
                print(f"      the brief's three groups of cells at `C16`: input for 0 to 2 digits {int((k10 <= 2).sum())} cells, for 3 or 4 {int(((k10 >= 3) & (k10 <= 4)).sum())}, for 5 or more {int((k10 >= 5).sum())}")
            per = (m > 0).sum(0)                                         # cells per byte
            row = []
            for gname, group in GROUPS:
                v = per[col(group)]
                row.append(f"{gname}: {v.min()} to {v.max()}" if len(group) > 1 else f"{gname}: {v[0]}")
            v20 = per[col(R50.NEEDED)]
            least = R50.NEEDED[int(np.argmin(v20))]
            print(f"      cells that have input for a byte: " + "; ".join(row) + f"; the least-read of the 20 is {R50.name(least)!r} with {v20.min()} cells; letters {per[col(R50.LETTERS)].min()} to {per[col(R50.LETTERS)].max()}")
            pairs = m[:, col(R50.NEEDED)]
            unread = int((F.mask_in.sum(0) == 0).sum())
            ownerless = int(sum(F.mask_in[:, v].sum() for v in range(D) if c.owner[v] < 0))
            letters = int(sum(F.mask_in[:, v].sum() for v in range(D) if c.owner[v] in R50.LETTERS))
            print(f"      of the 256 x 20 pairs of a cell and a needed byte: the cell reads no value of the byte in {100 * (pairs == 0).mean():.1f}%, one in {100 * (pairs == 1).mean():.1f}%, two in {100 * (pairs == 2).mean():.1f}%, three or more in {100 * (pairs >= 3).mean():.1f}%;"
                  f" of the {U * F.k_in:,} values the cells read, {100 * letters / (U * F.k_in):.0f}% belong to letters and {ownerless} to no byte; values read by no cell {unread}")
    print("   by the counts alone: 220 of the 512 values belong to the 20 needed bytes, 286 to the 26 letters, 6 to none; a cell reading 16 values has on average 6.9 values of needed bytes and 8.9 of letters")

# ------------------------------------------------------------------------------------------------ C
if "C" in want:
    print("C. travel between cells (E42's connections; a cell reads 10 cells within 3 rows and columns, and its own previous value)")
    for seed in SEEDS:
        dist = R50.distances(seed)
        off = dist[~np.eye(U, dtype=bool)]
        print(f"   seed {seed}: of the 65,280 ordered pairs of two cells, the state of the first reaches the second within 1 connection in {100 * (off <= 1).mean():.1f}%, within 2 in {100 * (off <= 2).mean():.1f}%, within 3 in {100 * (off <= 3).mean():.1f}%,"
              f" within 4 in {100 * (off <= 4).mean():.2f}% (all but {int((off > 4).sum())} pairs); the farthest pair is {off.max()} connections apart; cells that every other cell reaches within 3: {int((dist.max(0) <= 3).sum())} of 256")
    print("   the grid alone: two cells can be 8 rows or columns apart and a connection covers at most 3, so 3 connections is the least possible for the farthest pairs and needs each step to be of full length in the right direction")
    for seed in SEEDS:
        c = R50.Code(seed, amplitude=1.0)
        for nm in ("C64", "C16"):
            F = R49.Interface(seed, *R50.SETTINGS[nm])
            m = c.reads(F) > 0                                           # (256, 46)
            W_ = {b: R50.within(seed, np.nonzero(m[:, c.used.index(b)])[0], 3) for b in R50.NEEDED}      # byte -> (4, 256)
            cnt = lambda group, s: np.array([W_[b][s] for b in group]).sum(0)                           # per cell: bytes of the group within s connections
            f = lambda v: f"{v.min()} to {v.max()} (middle {med(v)[0]}, {med(v)[1]})"
            row20 = "; ".join(f"within {s} connection{'s' if s != 1 else ''} {f(cnt(R50.NEEDED, s))}" for s in (0, 1, 2))
            row10 = "; ".join(f"within {s} {f(cnt(R50.DIGITS, s))}" for s in (0, 1, 2))
            print(f"   seed {seed}, {nm}: bytes of the 20 a cell can have learned of: {row20}.  Digits: {row10}")
            cells = lambda group, s: np.array([W_[b][s].sum() for b in group])
            print(f"      cells that a digit has reached: at its own position {cells(R50.DIGITS, 0).min()} to {cells(R50.DIGITS, 0).max()}; one position later {cells(R50.DIGITS, 1).min()} to {cells(R50.DIGITS, 1).max()};"
                  f" two positions later {cells(R50.DIGITS, 2).min()} to {cells(R50.DIGITS, 2).max()}; three later {cells(R50.DIGITS, 3).min()} to {cells(R50.DIGITS, 3).max()} (of 256).  The least over all 20 needed bytes one position later: {cells(R50.NEEDED, 1).min()}")
    # the same from the forward: change the second operand's last digit and see which cells' states differ
    W0, mask = start_network()
    code = R50.Code(SEED)
    print(f"   from the forward itself, on the starting network of seed {SEED}: the second operand's last digit changed (40 items for each of the 90 ordered pairs of two digits), and the cells whose state differs counted")
    for nm in ("C64", "C16"):
        F = R49.Interface(SEED, *R50.SETTINGS[nm])
        W = F.start(W0)
        m = code.reads(F) > 0
        g = np.random.default_rng(50)
        agree, n_b2, n_eq, n_c1 = True, [], [], []
        for d in range(10):
            for e in range(10):
                if d == e:
                    continue
                a, b1 = g.integers(0, 89, 40), g.integers(0, 8, 40)
                x0 = plain_items([(int(aa), int(bb) * 10 + d) for aa, bb in zip(a, b1)])
                x1 = x0.copy(); x1[:, 6] = ord("0") + e
                H0, _ = R50.forward(W, mask, F, code, x0)
                H1, _ = R50.forward(W, mask, F, code, x1)
                diff = (H0 != H1).any(0)                                 # (L, 256)
                readers = m[:, code.used.index(ord("0") + d)] | m[:, code.used.index(ord("0") + e)]
                wn = R50.within(SEED, np.nonzero(readers)[0], 2)
                agree &= bool((diff[6] == wn[0]).all() and (diff[7] == wn[1]).all() and (diff[8] == wn[2]).all() and not diff[:6].any())
                n_b2.append(int(diff[6].sum())); n_eq.append(int(diff[7].sum())); n_c1.append(int(diff[8].sum()))
        print(f"      {nm}: cells whose state depends on that digit at the digit's own position {min(n_b2)} to {max(n_b2)}; at `=`, where the answer's first byte is read, {min(n_eq)} to {max(n_eq)};"
              f" one position later, where the second is read, {min(n_c1)} to {max(n_c1)}; equal in every case to the cells within 0, 1 and 2 connections of a cell that reads either digit: {agree}")

# ------------------------------------------------------------------------------------------------ D
if "D" in want:
    print(f"D. the start, on the starting network of seed {SEED} (E47's worker's file at {SREF}; the recurrent weights are E42's kept ones)")
    W0, mask = start_network()
    T49 = load(EPATH, EREF)
    same = all((T49[k] == W0[k]).all() for k in W0) and bool((T49["mask"].astype(bool) == mask).all())
    print(f"   the network built from E47's file is equal, tensor by tensor and in its connections, to the start of `F10` that E49's worker committed ({EREF}): {bool(same)}")
    code = R50.Code(SEED)
    items = stream_items(SEED, 64)
    x, y, ymask = batch_tensor(items)
    real = np.zeros(x.shape, bool)
    for i, (p, t) in enumerate(items):
        real[i, :len(p) + len(t)] = True                                # the start symbol, the prompt and the two answer digits
    Wh_dense = W0["rnn.weight_hh_l0"]
    # E42's F10 at the same start: the present code, dense interface
    full = R49.Interface(SEED, D, U)
    z1 = np.tanh(W0["book.weight"][x] @ W0["l1.weight"].T + W0["l1.bias"])
    Hs, lg = R49.forward(W0, mask, full, x)
    prev = np.concatenate([np.zeros_like(Hs[:, :1]), Hs[:, :-1]], 1)
    gi = z1 @ W0["rnn.weight_ih_l0"].T
    print(f"   E42's F10 (the present code): first-batch loss {loss_of(lg, y, ymask):.3f}; root mean square of the input drive {rms(gi[real]):.3f}, of the recurrent drive {rms((prev @ Wh_dense.T)[real]):.3f}, of a state {rms(Hs[real]):.3f},"
          f" of `l2`'s drive {rms((z1 @ W0['l2.weight'].T)[real]):.3f}, of `lm`'s {rms((Hs @ W0['lm.weight'].T)[real]):.3f}, of a logit {rms(lg[real]):.3f}")
    table = code.table()
    for nm in ("C-dense", "C64", "C16"):
        F = R49.Interface(SEED, *R50.SETTINGS[nm])
        W = F.start(W0)
        Hs, lg = R50.forward(W, mask, F, code, x)
        Ht, lt = R50.forward_table(W, mask, F, code, x)
        GI, L2 = R50.drives(W, F, code)
        gi = GI[x] - W["rnn.bias_ih_l0"]
        prev = np.concatenate([np.zeros_like(Hs[:, :1]), Hs[:, :-1]], 1)
        loss = loss_of(lg, y, ymask)
        m = code.reads(F)                                               # (256, 46)
        need = [code.used.index(b) for b in R50.NEEDED]
        drive = (GI[R50.NEEDED] - W["rnn.bias_ih_l0"]).reshape(len(R50.NEEDED), 3, U)                  # (20, 3, 256)
        mm = np.broadcast_to(m[:, need].T[:, None, :], drive.shape)
        by = "; ".join(f"{k} value{'s' if k != 1 else ''}: {rms(drive[mm == k]):.3f}" for k in (1, 2, 3) if (mm == k).any()) if nm != "C-dense" else f"11 values: {rms(drive):.3f}"
        cellsq = (gi.astype(np.float64) ** 2)[real].reshape(-1, 3, U).mean((0, 1))                       # a cell's own mean squared input drive on the batch
        print(f"   {nm}: first-batch loss {loss:.3f} (the coordinator's {BRIEF_LOSS[nm]}); root mean square of the input drive {rms(gi[real]):.3f} ({100 * (gi[real] == 0).mean():.0f}% of the entries exactly zero),"
              f" of the recurrent drive {rms((prev @ Wh_dense.T)[real]):.3f}, of a state {rms(Hs[real]):.3f}, of `l2`'s drive {rms((L2[x] - W['l2.bias'])[real]):.3f}, of `lm`'s {rms((Hs @ W['lm.weight'].T)[real]):.3f}, of a logit {rms(lg[real]):.3f}")
        print(f"      a cell's input for a needed byte, by how many of the byte's values the cell reads: {by}; a cell's own mean squared input drive over the batch varies over cells with a spread of {cellsq.std() / cellsq.mean():.2f} of the mean"
              f" (least {cellsq.min():.3f}, most {cellsq.max():.3f}); the two forms of the forward differ by at most {np.abs(lg - lt).max():.1e} in a logit and {np.abs(Hs - Ht).max():.1e} in a state (float32)")
    # how far one update can move a cell's input for a byte
    zU = np.tanh(W0["book.weight"][R50.USED].astype(np.float64) @ W0["l1.weight"].T + W0["l1.bias"])     # (46, 512)
    F16 = R49.Interface(SEED, 16, 8)
    l1_all = np.abs(zU).sum(1).mean()
    l1_16 = np.abs(zU[:, F16.in_idx]).sum(2).mean()
    amp = code.amplitude
    print("   how far one update can move a cell's input for a byte, if the optimizer moves each weight by its rate of 0.001 in the helpful direction (the code held as it is):")
    l2_all = np.sqrt((zU ** 2).sum(1)).mean()
    print(f"      E42's F10 at most {0.001 * l1_all:.4f} (512 weights on values of mean size {np.abs(zU).mean():.2f}, every weight shared by all bytes; {0.001 * l2_all:.4f} if the directions of the 512 steps were unrelated to the byte); E49's I16 at most {0.001 * l1_16:.4f} (16 weights);"
          f" `C-dense` {0.001 * ON * amp:.4f} (11 weights on values of {amp}, which no other byte uses); `C64` and `C16` {0.001 * amp:.4f} for each of the byte's values the cell reads, mostly one")
    print(f"      as a share of the size of that input at the start (root mean square): `C-dense` {0.001 * ON * amp / (amp * (ON / 3) ** .5 / 16):.3f}; a cell of `C64` reading one value of the byte {0.001 * amp / (amp * (D / 64) ** .5 / (16 * 3 ** .5)):.4f};"
          f" a cell of `C16` reading one value {0.001 * amp / (amp * (D / 16) ** .5 / (16 * 3 ** .5)):.4f} ({(0.001 * ON * amp / (amp * (ON / 3) ** .5 / 16)) / (0.001 * amp / (amp * (D / 16) ** .5 / (16 * 3 ** .5))):.0f} times less than `C-dense`)")
    print("   the sweep and the fixed code: the code has no gradient, so nothing is carried back through the input map or through `l2`; under the sweep the sign rule therefore acts on the recurrent map, on `lm` and on `l3` only,"
          " and the scale of the sign rule on the masked input map is not used (by the forward's structure; no measurement)")

# ------------------------------------------------------------------------------------------------ E
if "E" in want:
    print("E. multiply-adds for one item at one position (products only), two ways of counting")
    c = R50.Code(SEED, amplitude=1.0)
    n_rec = 3 * U * 11
    f10 = 8 * D + 3 * U * D + n_rec + D * U + D * D + 257 * D
    print(f"   E42's F10 by E49's count: {f10:,} (of it `l1` on the byte's row {8 * D:,})")
    ref = {}
    for nm, (k_in, k_out) in R50.SETTINGS.items():
        F = R49.Interface(SEED, k_in, k_out)
        a = 3 * U * k_in + n_rec + D * k_out + D * D + 257 * D           # every weight of a map counted, `l1` left out
        m = c.reads(F)                                                   # (256, 46)
        nz_in = 3 * m.sum(0)                                             # per byte: products of the input map with a non-zero value
        b = lambda q: q + n_rec + D * k_out + ON * D + 257 * D
        ref[nm] = (a, b(float(nz_in.mean())))
        print(f"   {nm:8s} K_IN {k_in:3d} K_OUT {k_out:3d}: every weight of a map counted, without `l1`: {a:,} ({a / f10:.3f} of F10; E49's function, which still counts `l1`, would give {a + 8 * D:,});"
              f" counting only products with a turned-on value: input map {nz_in.min()} to {nz_in.max()} by byte (mean {nz_in.mean():,.0f}), `l2` {ON * D:,}, total about {b(float(nz_in.mean())):,.0f} ({b(float(nz_in.mean())) / f10:.3f} of F10), of which `l3` {257 * D:,}")
    for nm in ("C64", "C16", "C16-in"):
        q = ref[nm][0] / ref['C-dense'][0]
        print(f"   {nm} against `C-dense`: {q:.3f} by the first count, {ref[nm][1] / ref['C-dense'][1]:.3f} by the second; *more* in multiply-adds by the first count needs {1.5 / q:.2f} times `C-dense`'s updates,"
              f" which can be recorded only if `C-dense` reaches the threshold by update {DREC * q / 1.5:,.0f}")
    print("   the brief gives no figure for any setting, and says only 'multiply-adds beside it, with E49's caution'")

# ------------------------------------------------------------------------------------------------ F
if "F" in want:
    print("F. the readings (three-row version; a threshold can be recorded up to update 24,064)")
    ref = {}
    for rule in ("S", "B"):
        J = json.loads(show(f"{P}e42/e42_F10_{rule}_{SEED}.json", WREF))
        cv = J["curve"]
        ref[rule] = {(m, t): reach(cv, m, t) for m in ("A", "R") for t in (50, 80, 90)}
        lv = {m: level(cv, m) for m in ("A", "R", "len12", "len40")}
        print(f"   E42's F10 {rule}: updates to 50, 80, 90 on A {[ref[rule][('A', t)] for t in (50, 80, 90)]}, on R {[ref[rule][('R', t)] for t in (50, 80, 90)]}; levels A {lv['A']:.1f}, R {lv['R']:.1f}, 12 letters {lv['len12']:.1f}, 40 letters {lv['len40']:.1f};"
              f" CPU-hours {(J['cpu_updates'] + J['cpu_scoring']) / 3600:.2f}")
    print("   `C-dense` against F10 (E41's cases on updates to 80):")
    for rule in ("S", "B"):
        for m in ("A", "R"):
            x0 = ref[rule][(m, 80)]
            print(f"      {rule} on {m}: F10 at {x0:,}; `C-dense` *fewer* below {2 * x0 / 3:,.0f}, *more* above {1.5 * x0:,.0f}" + ("" if 1.5 * x0 <= DREC else " (beyond the last recordable update)")
                  + f"; if `C-dense` does not reach 80: {'*more*' if DREC / x0 >= 1.5 else '*not decided*'} (24,064 / {x0:,} = {DREC / x0:.2f})")
    print(f"   `C64` and `C16` against `C-dense`: a threshold of the masked setting 1.5 times `C-dense`'s can be recorded only if `C-dense` reaches it by update {int(DREC / 1.5):,};"
          f" a masked run that does not reach it is *more* only under the same condition, and otherwise *not decided*")
    for rule in ("S", "B"):
        for m in ("A", "R"):
            x0 = ref[rule][(m, 80)]
            print(f"      {rule} on {m}: if `C-dense` needs what F10 needed ({x0:,}), the largest factor that can be recorded is {DREC / x0:.2f}; `C-dense` may be slower than F10 by at most {DREC / 1.5 / x0:.2f} for *more* to remain possible;"
                  f" for the 50 threshold F10 needed {ref[rule][(m, 50)]:,} (factor {DREC / ref[rule][(m, 50)]:.2f})")
    print("   the two answer bytes: with 1,000 items of each of two operations a share has a standard error of at most 1.1 points, and a difference of two shares on the same items less than 1.6")
    print(f"   E49's runs of the same masks on the learned code, as far as E49's worker has committed them ({EREF}; runs not complete), beside F10 at the same update:")
    for nm in ("I64_S", "I64_B", "I16_S", "I16_B"):
        try:
            J = json.loads(show(f"{P}e49/e49_{nm}_{SEED}.json", EREF))
        except subprocess.CalledProcessError:
            print(f"      {nm}: no receipt on file"); continue
        r = J["curve"][-1]
        Jf = json.loads(show(f"{P}e42/e42_F10_{nm[-1]}_{SEED}.json", WREF))
        rf = next(q for q in Jf["curve"] if q["step"] == r["step"])
        cpu = (r["cpu_updates"] + r["cpu_scoring"]) / 3600 * DOSE / r["step"]
        print(f"      {nm}: at update {r['step']:,} of {DOSE:,}: A {r['A']:.1f}, R {r['R']:.1f}, taught {r['taught']:.1f}, loss {r['loss']:.3f} (complete: {J.get('complete')}); F10 at that update: A {rf['A']:.1f}, R {rf['R']:.1f};"
              f" at this rate a run takes {cpu:.2f} CPU-hours")
    print(f"   cost: at the rate E49's worker measured for the specification's forward (1.11 to 1.22 CPU-hours a run), two runs are about 2.3 CPU-hours, six about 7 and eight about 9.3; with three processes six runs take two rounds")

# ------------------------------------------------------------------------------------------------ G
if "G" in want:
    print(f"G. the test of use (states of chosen cells taken from a run with another last digit), mechanics on the starting network of seed {SEED}")
    W0, mask = start_network()
    code = R50.Code(SEED)
    F = R49.Interface(SEED, *R50.SETTINGS["C16"])
    W = F.start(W0)
    m = code.reads(F) > 0
    g = np.random.default_rng(51)
    a, b1 = g.integers(0, 89, 200), g.integers(0, 8, 200)
    d, e = g.integers(0, 10, 200), g.integers(1, 10, 200)
    e = (d + e) % 10
    x0 = plain_items([(int(aa), int(bb) * 10 + int(dd)) for aa, bb, dd in zip(a, b1, d)])
    x1 = x0.copy(); x1[:, 6] = ord("0") + e
    readers = np.array([m[:, code.used.index(ord("0") + int(dd))] | m[:, code.used.index(ord("0") + int(ee))] for dd, ee in zip(d, e)])   # (200, 256)
    L0, L1, Lall, H0, H1 = R50.patch_test(W, mask, F, code, x0, x1, 6, 7, np.ones((200, U), bool))
    _, _, Lnone, _, _ = R50.patch_test(W, mask, F, code, x0, x1, 6, 7, np.zeros((200, U), bool))
    _, _, Lread, _, _ = R50.patch_test(W, mask, F, code, x0, x1, 6, 7, readers)
    _, _, Lnon, _, _ = R50.patch_test(W, mask, F, code, x0, x1, 6, 7, ~readers)
    gap = np.abs(L1 - L0).max(1)
    print(f"   200 pairs of items that differ in the second operand's last digit only, `C16`: with every cell's state at `=` taken from the other run the logits at `=` are the other run's (largest difference {np.abs(Lall - L1).max():.1e});"
          f" with none taken they are the item's own ({np.abs(Lnone - L0).max():.1e})")
    print(f"   cells that read neither digit: {int((~readers).sum(1).min())} to {int((~readers).sum(1).max())} of 256 an item; their states at the digit's own position are the same in the two runs: {bool((H0[:, 6][~readers] == H1[:, 6][~readers]).all())};"
          f" at `=` they differ in {100 * (H0[:, 7][~readers] != H1[:, 7][~readers]).mean():.0f}% of the cases")
    sh_non = np.abs(Lnon - L0).max(1) / gap
    sh_read = np.abs(Lread - L0).max(1) / gap
    print(f"   at the start, before any learning, the largest change of a logit at `=` when only the cells that read neither digit take the other run's states is {np.median(sh_non):.2f} of the change between the two runs (median over items),"
          f" and {np.median(sh_read):.2f} when only the cells that read one of the two digits do.  On a trained network the same two figures, and the share of items whose first answer byte follows the patched cells, say which cells carry the digit")
