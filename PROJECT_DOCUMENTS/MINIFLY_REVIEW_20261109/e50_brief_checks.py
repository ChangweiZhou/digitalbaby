"""Checks behind AUDIT_E50_SPARSE_INPUT_CODE_BRIEF_20261117.md (the E50 brief and e50_sparse_code_spec.py at f8f62df6).

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
  F  the readings: E42's F10 of seed 73129; what can be recorded against `C-dense`; cost
  G  the test of whether cells use what arrives through neighbours (the mechanics, on the starting network)
  H  the record that arrived after the brief: E49's eight runs (a36de661) and E47's trial (578a877b), recomputed from the
     receipts; what they leave of the brief's settings, dose and predictions
  I  the same eight runs of E49 from their final weights: the masks are the reader's draws, nothing outside them, and
     the counts of A, R and the length test recomputed by a decoder that shares no structure with the runner's forward
  J  the codes E49's networks learned: their size, whether the values a cell reads still tell the needed bytes apart, and
     the size of a cell's input for a needed byte (add the word `linear` for the fitted linear rule, about 7 minutes)
  K  the settings the audit asks for: `C16-in` and the setting in which every cell reads one value of every used byte
     (the draw's digest for three seeds, the reader's first-batch loss, sizes at the start, multiply-adds)

Run from the repository root:
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e50_brief_checks.py [A B C D E F G H I J K]
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
EREF = "a36de661"                                   # claude/peaceful-noether-vcvrmf, E49's worker: the runner, its checks, the eight complete runs
E47REF = "578a877b"                                 # claude/e47-worker-stage-0-3d0ya6: E47's trial complete (four runs at 1,024 cells)
EPATH = "MOD97_LOCAL_RULE_20261106/exp45/e49_qual/check2_73129_runs/e49_F10_S_dry2048_73129_start.pt"
P = "MOD97_LOCAL_RULE_20261106/exp45/"
SEED, SEEDS = 73129, (73129, 93683, 116089)
U, D, ON = 256, 512, 11
DOSE, DREC = 24576, 24576 - 512                    # the last update at which a threshold can be recorded, three-row version
DTYPE = {"FloatStorage": "<f4", "LongStorage": "<i8", "BoolStorage": "?"}
COMMITTED = "12111044f0f8bc70"                     # OUTPUT_E50_SPEC_MAC.txt, seed 73129
BRIEF_LOSS = {"C-dense": 5.597, "C64": 5.556, "C16": 5.609}
want = (set(sys.argv[1:]) & set("ABCDEFGHIJK")) or set("ABCDEFGHIJK")
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
    print("   cost: E42's F10 took 0.87 and 0.86 CPU-hours; E49's eight runs with the specification's masked forward took 0.90 to 1.02 CPU-hours each, scoring included (part H); two runs of the dose are about 2 CPU-hours")

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

# ------------------------------------------------------------------------------------------------ the record after the brief
E49P, E47P = P + "e49/", P + "e47/"
RUNS49 = [(s, r) for r in ("B", "S") for s in ("I64", "I16", "in16", "out8")]
# E49_REPORT.md at a36de661, the summary table: updates to 80 on R (three rows), the level of R, R's rise over the last 2,048 updates
REPORT49 = {("I64", "B"): (None, 65.17, 7.88), ("I16", "B"): (None, 16.87, 3.37), ("in16", "B"): (20480, 82.53, 0.65), ("out8", "B"): (None, 70.77, 8.60),
            ("I64", "S"): (None, 0.93, -0.25), ("I16", "S"): (None, 0.90, -0.02), ("in16", "S"): (None, 1.37, -0.35), ("out8", "S"): (None, 7.75, 2.37)}
rising = lambda cv: float(np.mean([r["R"] for r in cv[-3:]]) - np.mean([r["R"] for r in cv[-11:-8]]))    # against the three rows ending 2,048 updates earlier
num = lambda v: "not reached" if v is None else f"{v:,}"


def receipt49(s, r):
    J = json.loads(show(f"{E49P}e49_{s}_{r}_{SEED}.json", EREF))
    assert J["complete"] and J["setting"] == s and J["rule"] == r and len(J["curve"]) == 97 and J["curve"][-1]["step"] == DOSE
    return J


if "H" in want:
    print(f"H. the record that arrived after the brief, recomputed from the receipts: E49's eight runs ({EREF}) and E47's trial ({E47REF}); seed {SEED}, 24,576 updates, thresholds in the three-row version")
    f10 = {}
    for rule in ("B", "S"):
        cv = json.loads(show(f"{P}e42/e42_F10_{rule}_{SEED}.json", WREF))["curve"]
        f10[rule] = {k: reach(cv, k[0], int(k[1:])) for k in ("A50", "A80", "R50", "R80")}
        f10[rule].update(A=level(cv, "A"), R=level(cv, "R"), len40=level(cv, "len40"))
        print(f"   E42's F10 {rule} (dense interface, learned code): 80 on A at {num(f10[rule]['A80'])}, 80 on R at {num(f10[rule]['R80'])}, 50 on R at {num(f10[rule]['R50'])}; levels A {f10[rule]['A']:.1f}, R {f10[rule]['R']:.1f}, 40 letters {f10[rule]['len40']:.1f}")
    rec, same = {}, True
    for s, r in RUNS49:
        J = receipt49(s, r); cv = J["curve"]
        d = {k: reach(cv, k[0], int(k[1:])) for k in ("A50", "A80", "R50", "R80")}
        d.update(A=level(cv, "A"), R=level(cv, "R"), len40=level(cv, "len40"), rise=rising(cv), loss=cv[-1]["loss"], cpu=(J["cpu_updates"] + J["cpu_scoring"]) / 3600,
                 top=max(max(q["A"], q["R"], q["taught"], q["len12"], q["len40"]) for q in cv), ma=J["multiply_adds_per_position"])
        rec[(s, r)] = d
        rep = REPORT49[(s, r)]
        same &= d["R80"] == rep[0] and abs(d["R"] - rep[1]) < 0.006 and abs(d["rise"] - rep[2]) < 0.006
        print(f"   E49 {s:5s} {r} (K_IN {J['k_in']:3d}, K_OUT {J['k_out']:3d}; {d['ma']:,} multiply-adds a position): 80 on A at {num(d['A80'])}, 80 on R at {num(d['R80'])}, 50 on R at {num(d['R50'])};"
              f" levels A {d['A']:.1f}, R {d['R']:.1f}, 40 letters {d['len40']:.1f}; R over the last three rows against 2,048 updates earlier {d['rise']:+.2f}; the largest score on any panel at any row {d['top']:.1f};"
              f" loss at the end {d['loss']:.3f}; {d['cpu']:.2f} CPU-hours")
    print(f"   the eight rows agree with the summary table of E49_REPORT.md (updates to 80 on R, the level of R, the rise): {same}")
    print("   the four operations apart, on untaught pairs at 12 letters (300 items each; the score A is the mean of add and sub): share right at the last row, and updates to 80 in the three-row version")
    def ops(cv):
        cur = lambda op: [dict(step=q["step"], v=100 * q["A_counts"][op + "89"] / 300) for q in cv]
        return {op: (cv[-1]["A_counts"][op + "89"] / 3, reach(cur(op), "v", 80)) for op in ("add", "sub", "max", "min")}
    for rule in ("B", "S"):
        o = ops(json.loads(show(f"{P}e42/e42_F10_{rule}_{SEED}.json", WREF))["curve"])
        print(f"      F10   {rule}: " + "; ".join(f"{op} {o[op][0]:.0f} ({num(o[op][1])})" for op in o))
    for s_, r in RUNS49:
        o = ops(receipt49(s_, r)["curve"])
        print(f"      {s_:5s} {r}: " + "; ".join(f"{op} {o[op][0]:.0f} ({num(o[op][1])})" for op in o))
    b, f = rec[("in16", "B")], f10["B"]
    print(f"   under backpropagation the one limited setting that reaches 80 on R is `in16` (16 values a cell, dense readout): {b['R80']:,} against F10's {f['R80']:,}, a factor of {b['R80'] / f['R80']:.2f}"
          f" (on A {b['A80']:,} against {f['A80']:,}, {b['A80'] / f['A80']:.2f}); `out8` (dense input, 8 cells a readout unit) does not reach it; under the sweep no setting passes {max(rec[(s, 'S')]['top'] for s in ('I64', 'I16', 'in16')):.1f}"
          f" on any panel with a limited input, and `out8` ends at {rec[('out8', 'S')]['R']:.1f} on R")
    fac = b["R80"] / f["R80"]
    print(f"   what this leaves of E50's dose: a setting that costs against `C-dense` what `in16` costs against F10 (a factor of {fac:.2f} on R) is recorded within 24,064 updates only if `C-dense` reaches 80 on R by update {DREC / fac:,.0f},"
          f" that is at most {DREC / fac / f['R80']:.2f} times F10's {f['R80']:,}; with a dose of 49,152 updates (the last recordable update 48,640) `C-dense` may need up to {48640 / fac:,.0f}, and a factor of up to {48640 / f['R80']:.1f} can be recorded if it needs what F10 needed")
    print("   the brief's predictions beside E49's outcomes on the same masks (the coordinator's chance; what the same masks did on the learned code, where every cell can know every byte):")
    for name, chance, key in (("P50.4 `C64`, backpropagation, reaches 80 on R", 65, ("I64", "B")), ("P50.5 `C16`, backpropagation, reaches 80 on R", 35, ("I16", "B")), ("P50.6 `C16`, sweep, reaches 80 on R", 15, ("I16", "S"))):
        d = rec[key]
        print(f"      {name}: {chance}%; E49's `{key[0]}` under the same rule: 80 on R {num(d['R80'])}, level of R {d['R']:.1f}, rise {d['rise']:+.2f}")
    print("   E47's trial (1,024 cells, of which 256 read the byte's whole code and are read by the readout; the other 768 have no input and are not read):")
    for arm in ("one", "place"):
        for r in ("B", "S"):
            J = json.loads(show(f"{E47P}e47_32_{arm}_{r}_{SEED}.json", E47REF)); cv = J["curve"]
            assert J["complete"] and len(cv) == 97 and J["interface_cells"] == 256 and J["far_cells"] == 768
            R80, A80 = reach(cv, "R", 80), reach(cv, "A", 80)
            last = cv[-1]
            print(f"      `{arm}` {r}: 80 on A at {num(A80)}, 80 on R at {num(R80)} (F10: {num(f10[r]['A80'])}, {num(f10[r]['R80'])}; a factor of {R80 / f10[r]['R80']:.2f} on R); R at the last row {last['R']:.1f},"
                  f" with the cells without input held at zero {last['far_off']['R']:.1f}, held at their mean {last['far_mean']['R']:.1f}; 40 letters, last three rows {level(cv, 'len40'):.1f}")

# ------------------------------------------------------------------------------------------------ I
if "I" in want or "J" in want:
    from np_gru import panel
    NEED = [R50.USED.index(b) for b in R50.NEEDED]
    LETS = [R50.USED.index(b) for b in R50.LETTERS]
    DIGS = [R50.USED.index(b) for b in R50.DIGITS]

    def tables49(W, F):
        """For every byte: each cell's three input sums over the values the cell reads, and each readout unit's sum over the
        byte's code, the biases included.  With a code that depends on the byte alone, this is all the input map and `l2` do."""
        code = np.tanh(W["book.weight"].astype(np.float64) @ W["l1.weight"].T + W["l1.bias"])            # (258, 512)
        Wi, bi = W["rnn.weight_ih_l0"].astype(np.float64), W["rnn.bias_ih_l0"]
        cells = np.arange(U)
        GI = np.empty((258, 3 * U))
        for g in range(3):
            if F.k_in == D:
                GI[:, g * U:(g + 1) * U] = code @ Wi[g * U:(g + 1) * U].T + bi[g * U:(g + 1) * U]
            else:
                GI[:, g * U:(g + 1) * U] = np.einsum("vuk,uk->vu", code[:, F.in_idx], Wi[g * U + cells[:, None], F.in_idx]) + bi[g * U:(g + 1) * U]
        L2 = code @ W["l2.weight"].T + W["l2.bias"]
        return code, GI.astype(np.float32), L2.astype(np.float32)

    def decode49(W, mask_cells, F, GI, L2, prompts, chunk=500):
        """Greedy decoding of three bytes.  The prompt is run once; every sum runs over a cell's own sources and a readout
        unit's own cells; the input is looked up in the two tables.  It shares no structure with the specification's forward."""
        Wh, bh = W["rnn.weight_hh_l0"], W["rnn.bias_hh_l0"]
        cells = np.arange(U)
        src = np.array([np.nonzero(mask_cells[i])[0] for i in range(U)])
        Wh_g = [Wh[g * U + cells[:, None], src] for g in range(3)]
        Wo = None if F.k_out == U else W["lm.weight"][np.arange(D)[:, None], F.out_idx]

        def step(tok, h):
            gi, hin = GI[tok], h[:, src]
            gh = [np.einsum("buk,uk->bu", hin, Wh_g[g]) + bh[g * U:(g + 1) * U] for g in range(3)]
            r = R42.sigmoid(gi[:, :U] + gh[0]); z = R42.sigmoid(gi[:, U:2 * U] + gh[1]); n = np.tanh(gi[:, 2 * U:] + r * gh[2])
            return ((1 - z) * n + z * h).astype(np.float32)

        out = []
        for a in range(0, len(prompts), chunk):
            x = np.array([[BOS] + p for p in prompts[a:a + chunk]], dtype=np.int64)
            h = np.zeros((len(x), U), np.float32)
            for t in range(x.shape[1]):
                h = step(x[:, t], h)
            tok, gen = x[:, -1], []
            for k in range(3):
                o = (h @ W["lm.weight"].T if Wo is None else np.einsum("bjk,jk->bj", h[:, F.out_idx], Wo)) + W["lm.bias"]
                tok = (np.tanh(L2[tok] + o) @ W["l3.weight"].T + W["l3.bias"]).argmax(1)
                gen.append(tok)
                if k < 2:
                    h = step(tok, h)
            out.extend(np.array(gen).T.tolist())
        return out

if "I" in want:
    print(f"I. E49's eight runs from their final weights ({EREF}): the reader's decoder beside the receipt's last row (reader / receipt)")
    AP = {f"{t[0]}{t[1]}": panel(t, split(t, .5, SEED)[1], SEED, ks=(12,), n=300)[12] for t in DEV}
    RP = {}
    for t in DEV[:2]:
        held = split(t, .5, SEED)[1]
        red = [(a, b) for a, b in held if (a + b >= 89 if t[0] == "add" else a - b < 0)]
        RP[f"{t[0]}{t[1]}"] = [render(t, a, b, 0, None) for a, b in Random(f"e41|{SEED}|50|{t[0]}").sample(sorted(red), 1000)]
    LEN_TASK = ("add", 89, b"p")
    LP = panel(LEN_TASK, split(LEN_TASK, .5, SEED)[1], SEED, ks=(12, 40), n=300)
    conn = R42.draw_connections(SEED, 10)
    all_equal = True
    for s, r in RUNS49:
        J = receipt49(s, r); last = J["curve"][-1]
        T = load(f"{E49P}e49_{s}_{r}_{SEED}.pt", EREF)
        F = R49.Interface(SEED, int(T["k_in"]), int(T["k_out"]))
        mask = T["mask"].astype(bool)
        draws = bool((T["mask_in"].astype(bool) == F.mask_in).all() and (T["mask_out"].astype(bool) == F.mask_out).all()) and all(sorted(np.nonzero(mask[i])[0].tolist()) == sorted(conn[i] + [i]) for i in range(U))
        zero = bool((T["rnn.weight_ih_l0"][~np.tile(F.mask_in, (3, 1))] == 0).all() and (T["lm.weight"][~F.mask_out] == 0).all() and (T["rnn.weight_hh_l0"][~np.tile(mask, (3, 1))] == 0).all())
        _, GI, L2 = tables49(T, F)
        count = lambda items: sum(g == list(a) for g, (_, a) in zip(decode49(T, mask, F, GI, L2, [p for p, _ in items]), items))
        mA = {n: count(it) for n, it in AP.items()}; mR = {n: count(it) for n, it in RP.items()}; mL = {str(k): count(LP[k]) for k in (12, 40)}
        eq = all(mA[n] == last["A_counts"][n] for n in mA) and all(mR[n] == last["R_counts"][n] for n in mR) and all(mL[k] == last["length_counts"][k] for k in mL)
        all_equal &= eq and draws and zero
        print(f"   {s:5s} {r}: the file's three masks are the reader's draws {draws}; every weight outside them is zero {zero}; A add, sub, max, min {[mA[n] for n in AP]} / {[last['A_counts'][n] for n in AP]};"
              f" R add, sub {[mR[n] for n in RP]} / {[last['R_counts'][n] for n in RP]}; 12 and 40 letters {[mL[k] for k in mL]} / {[last['length_counts'][k] for k in mL]}; equal {eq}", flush=True)
        del T
    print(f"   all eight runs: masks, zeros and every count equal to the receipt's: {all_equal}")

# ------------------------------------------------------------------------------------------------ J
if "J" in want:
    print("J. the codes E49's networks learned (tanh(l1(byte table)) of the 46 used bytes at the end of the run), on the values each cell reads")

    def closest(X):
        """X (cells, 46, K).  Per cell: the smallest distance between two of the 20 needed bytes over the mean such distance; the
        same among the ten digits; whether two of the 46 used bytes coincide (closer than 1e-6)."""
        d = np.sqrt(((X[:, :, None] - X[:, None]) ** 2).sum(-1))
        iu = np.triu_indices(len(NEED), 1); dn = d[:, NEED][:, :, NEED][:, iu[0], iu[1]]
        i10 = np.triu_indices(10, 1); dd = d[:, DIGS][:, :, DIGS][:, i10[0], i10[1]]
        i46 = np.triu_indices(46, 1)
        return dn.min(1) / dn.mean(1), dd.min(1) / dd.mean(1), (d[:, i46[0], i46[1]] < 1e-6).any(1)

    LABEL = np.array([NEED.index(i) if i in NEED else len(NEED) for i in range(46)])                     # 21 classes: a letter is one class

    def linear_rule(X, steps=8000):
        """One linear rule on a cell's values (with a constant), 21 classes, fitted by gradient descent on the cross-entropy.
        Returns for every cell how many of the 46 bytes the rule puts in their class."""
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

    def describe(tag, W, F, k_show):
        code, GI, _ = tables49(W, F)
        Z = code[R50.USED]                                                                               # (46, 512)
        Zn = code[R50.NEEDED]
        dall = np.sqrt(((Zn[:, None] - Zn[None]) ** 2).sum(-1)); iu = np.triu_indices(len(NEED), 1)
        k0 = int(np.argmin(dall[iu])); pair = f"{R50.name(R50.NEEDED[iu[0][k0]])} and {R50.name(R50.NEEDED[iu[1][k0]])}"
        drive = (GI[R50.NEEDED] - W["rnn.bias_ih_l0"]).astype(np.float64)                               # (20, 768): a cell's input for a needed byte
        part = drive - drive.mean(0)                                                                    # what differs from the mean over the 20 needed bytes
        line = (f"   {tag}: mean size of a weight of `l1` {np.abs(W['l1.weight']).mean():.3f}; of a value of the code {np.abs(Z).mean():.3f}; mean distance between the codes of two needed bytes {dall[iu].mean():.2f},"
                f" the closest two ({pair}) at {dall[iu].min() / dall[iu].mean():.2f} of it; mean size of a kept weight of the input map {np.abs(W['rnn.weight_ih_l0'][np.tile(F.mask_in, (3, 1))]).mean():.3f};"
                f" a cell's input for a needed byte, root mean square {rms(drive):.2f}, of which what differs between bytes {rms(part):.2f}")
        if k_show:
            Fm = R49.Interface(SEED, k_show, U)                                                         # the values a cell reads at K_IN = k_show
            X = Z[:, Fm.in_idx].transpose(1, 0, 2)                                                      # (256, 46, K)
            a, b, same = closest(X)
            line += (f".  On the {k_show} values each cell reads: the closest two needed bytes at {a.min():.3f} of their mean distance on the worst cell, {np.median(a):.3f} on the median cell ({int((a < 0.10).sum())} cells below 0.10);"
                     f" the closest two digits at {b.min():.3f}, {np.median(b):.3f}; cells on which two of the 46 used bytes coincide {int(same.sum())}")
            if "linear" in sys.argv:
                lin = linear_rule(X)
                line += f"; a linear rule on the cell's values puts all 46 bytes in their class on {int((lin == 46).sum())} of 256 cells (least {lin.min()} of 46)"
        print(line, flush=True)

    W0, _ = start_network()
    describe("the start of seed 73129 (dense interface; the values are those `I16` reads)", W0, R49.Interface(SEED, D, U), 16)
    for rule in ("B", "S"):
        T = load(f"{P}e42/e42_F10_{rule}_{SEED}.pt", WREF)
        describe(f"E42's F10 {rule}, trained (dense interface; the values are those `I16` reads)", T, R49.Interface(SEED, D, U), 16)
        del T
    for s, r in RUNS49:
        T = load(f"{E49P}e49_{s}_{r}_{SEED}.pt", EREF)
        F = R49.Interface(SEED, int(T["k_in"]), int(T["k_out"]))
        describe(f"E49 {s:5s} {r}, trained", T, F, F.k_in if F.k_in < D else 0)
        del T
    print(f"   E50's fixed code, for comparison: every two used bytes are {np.sqrt(2 * ON) * 3.2387:.2f} apart, and the code cannot change")

# ------------------------------------------------------------------------------------------------ K
if "K" in want:
    print("K. the settings the audit asks for: a limited input with the dense readout (`C16-in`, the brief's conditional setting), and a setting in which every cell reads one value of every used byte")

    OneOfEach = R50.OneOfEach

    for seed in SEEDS:
        F = OneOfEach(seed)
        c = R50.Code(seed, amplitude=1.0)
        m = c.reads(F)
        per_value = F.mask_in.sum(0)
        owned = np.array([o >= 0 for o in c.owner])
        print(f"   seed {seed}, one value of every used byte: digest of the 256 lists {R49.digest16(F.in_src)}; every cell reads exactly one value of each of the 46 used bytes {bool((m == 1).all())};"
              f" cells that read a value: {per_value[owned].min()} to {per_value[owned].max()} (mean {per_value[owned].mean():.1f}); the six values without an owner are read by no cell {bool((per_value[~owned] == 0).all())}")
    W0, mask = start_network()
    code = R50.Code(SEED)
    items = stream_items(SEED, 64)
    x, y, ymask = batch_tensor(items)
    real = np.zeros(x.shape, bool)
    for i, (p_, t_) in enumerate(items):
        real[i, :len(p_) + len(t_)] = True
    n_rec = 3 * U * 11
    for nm, F in (("`C-dense`", R49.Interface(SEED, D, U)), ("`C16-in` (16 values a cell, dense readout)", R49.Interface(SEED, 16, U)), ("one value of every used byte (46 values a cell, dense readout)", OneOfEach(SEED))):
        W = F.start(W0)
        Hs, lg = R50.forward(W, mask, F, code, x)
        GI, _ = R50.drives(W, F, code)
        gi = GI[x] - W["rnn.bias_ih_l0"]
        mr = code.reads(F)
        need = [code.used.index(b) for b in R50.NEEDED]
        drive = (GI[R50.NEEDED] - W["rnn.bias_ih_l0"]).reshape(len(R50.NEEDED), 3, U)
        mm = np.broadcast_to(mr[:, need].T[:, None, :], drive.shape)
        by = "; ".join(f"{k} value{'s' if k != 1 else ''} {rms(drive[mm == k]):.3f}" for k in (1, 2, 3, 11) if (mm == k).any())
        weights = 3 * U * F.k_in + n_rec + D * F.k_out + D * D + 257 * D
        nz = 3 * float(mr.sum(0).mean()) + n_rec + D * F.k_out + ON * D + 257 * D
        scale = (D / F.k_in) ** .5
        print(f"   {nm}: the kept weights of the input map are scaled by {scale:.3f}; first-batch loss by the reader {loss_of(lg, y, ymask):.3f}; root mean square of the input drive {rms(gi[real]):.3f} ({100 * (gi[real] == 0).mean():.0f}% of the entries exactly zero);"
              f" a cell's input for a needed byte by the number of the byte's values it reads: {by}; one update of 0.001 a weight moves that input by {0.001 * code.amplitude:.4f} for each value read, "
              + (f"that is by {100 * 0.001 * ON * code.amplitude / (code.amplitude * (ON / 3) ** .5 / 16):.1f}% of its size for the 11 values of a byte;" if F.k_in == D else f"{100 * 0.001 * code.amplitude / (code.amplitude * scale / (16 * 3 ** .5)):.2f}% of the size of an input that rests on one value;") +
              f" multiply-adds a position {weights:,} by the weights that take part, about {nz:,.0f} by the products with a turned-on value")
