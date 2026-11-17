"""How close the sweep's gradient is to the exact gradient under other choices of the sweep's scale.  Read-only, nothing is trained.

The document is SWEEP_SCALE_VARIANTS_ALIGNMENT_20261117.md (same folder); this script prints every figure of it.

    python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/sweep_scale_variants.py [part ...]

    parts   stream      the batches rebuilt here are the recorded ones (digest against the receipts)
            forward     this script's forward against the reviewer's readers and against the recorded first-batch loss
            test1       the exact gradient against finite differences of the loss
            test2       V0 on the first batch against the cosines recorded in the receipts of E42, E49 and E50
            test3       V5 (signs times exact magnitudes) against the exact gradient
            magnitudes  how unequal the magnitudes are that a scale replaces
            tables      the cosines, norm ratios and gate blocks of every variant, 16 further batches
            all         every part (the default)
    a part may be followed by network names (for example `tables G_S F10_B`) to restrict it; the names are printed by `stream`.

Rules of the sweep, as the code read with `git show` has them (no line of another branch is run):
    exp3_state.SignLinear   the error crosses a map backward through sign(W) * mean(|W|); the map's own gradient is the outer product
    e42_cells.MaskedSign    the same with the sign and the mean taken inside the mask; the weight gradient is zero outside the mask
    arch26.forward, e42_cells.forward (period 1), e49_interface.forward, e50_sparse_code.forward   the networks
    exp45.batch_tensor, exp26_candidate.step   the mask of the answer's positions and the mean of the cross-entropy over them
    arch20.NetRNN           the byte table has padding_idx = 257: the row of the start symbol receives no gradient

The task and the stream (render, split, batches) are copied from COORDINATION_20261117/check_e54_brief.py, which copied them from
exp11_cell.py (random.Random only).  Weight files are read with `git show` into memory, one at a time, by compare_pt_tensors.tensors
(an unpickler that refuses every class but those that describe a tensor).  Standard library and numpy only; float64 throughout.
"""
import sys
sys.dont_write_bytecode = True                      # nothing is written beside this script
import hashlib, json, os, subprocess, time
from random import Random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_pt_tensors import tensors

T0 = time.time()
SEED = 73129
P = "MOD97_LOCAL_RULE_20261106/exp45/"
REF42, REF49, REF50 = "origin/claude/gallant-dijkstra-i9zgsr", "origin/claude/peaceful-noether-vcvrmf", "origin/claude/e50-fixed-input-code-017dSE"
DTYPE = {"FloatStorage": "<f4", "LongStorage": "<i8", "BoolStorage": "?"}
U, D = 256, 512
N_FURTHER = 16                                      # batches 2 to 17 of the stream; batch 1 is the receipts' batch

# name: (ref, weight file, receipt, kind, what it is)
NETS = {
    "G_S":           (REF42, "e42/e42_G_S_73129.pt", "e42/e42_G_S_73129.json", "end", "E42 G (dense), trained under the sweep"),
    "G_B":           (REF42, "e42/e42_G_B_73129.pt", "e42/e42_G_B_73129.json", "end", "E42 G (dense), trained under back-propagation"),
    "F10_start":     (REF49, "e49_qual/check2_73129_runs/e49_F10_S_dry2048_73129_start.pt", "e49_qual/check2_73129_runs/e49_F10_S_dry2048_73129.json", "start",
                      "E42 F10 at its start (the start file of E49's check 2, a run of F10 by E49's runner)"),
    "F10_S":         (REF42, "e42/e42_F10_S_73129.pt", "e42/e42_F10_S_73129.json", "end", "E42 F10 (10 neighbours and itself), trained under the sweep"),
    "F10_B":         (REF42, "e42/e42_F10_B_73129.pt", "e42/e42_F10_B_73129.json", "end", "E42 F10, trained under back-propagation"),
    "in16_start":    (REF49, "e49/e49_in16_S_73129_start.pt", "e49/e49_in16_S_73129.json", "start", "E49 in16 at its start"),
    "in16_S":        (REF49, "e49/e49_in16_S_73129.pt", "e49/e49_in16_S_73129.json", "end", "E49 in16 (16 of 512 input values a cell), sweep: did not learn"),
    "in16_B":        (REF49, "e49/e49_in16_B_73129.pt", "e49/e49_in16_B_73129.json", "end", "E49 in16, trained under back-propagation"),
    "C-dense_start": (REF50, "e50/e50_C-dense_S_73129_start.pt", "e50/e50_C-dense_S_73129.json", "start", "E50 C-dense at its start (fixed code)"),
    "C-dense_S":     (REF50, "e50/e50_C-dense_S_73129.pt", "e50/e50_C-dense_S_73129.json", "end", "E50 C-dense, trained under the sweep"),
    "C-dense_B":     (REF50, "e50/e50_C-dense_B_73129.pt", "e50/e50_C-dense_B_73129.json", "end", "E50 C-dense, trained under back-propagation"),
}
RECEIPT_REF = {"e42": REF42, "e49": REF49, "e50": REF50}
MAPS = ("l1", "ih", "hh", "l2", "lm", "l3")
WNAME = {"l1": "l1.weight", "ih": "rnn.weight_ih_l0", "hh": "rnn.weight_hh_l0", "l2": "l2.weight", "lm": "lm.weight", "l3": "l3.weight"}
GRU_MAPS = ("ih", "hh")
VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V5")
TENSORS = ("book.weight", "l1.weight", "rnn.weight_ih_l0", "rnn.weight_hh_l0", "lm.weight", "l2.weight", "l3.weight")
SHORT = {"book.weight": "byte table", "l1.weight": "l1", "rnn.weight_ih_l0": "input map", "rnn.weight_hh_l0": "recurrent map",
         "lm.weight": "lm", "l2.weight": "l2", "l3.weight": "l3"}
BIASES = ("l1.bias", "rnn.bias_ih_l0", "rnn.bias_hh_l0", "lm.bias", "l2.bias", "l3.bias")


def show(path, ref):
    return subprocess.run(["git", "show", f"{ref}:{P}{path}"], capture_output=True, check=True, cwd=HERE).stdout


def load(path, ref):
    return {n.lstrip("/"): np.frombuffer(raw, dtype=DTYPE[st]).reshape(sh).copy() for n, (st, sh, raw) in tensors(show(path, ref)).items()}


_RECEIPTS = {}


def receipt(path):
    if path not in _RECEIPTS:
        _RECEIPTS[path] = json.loads(show(path, RECEIPT_REF[path[:3]]))
    return _RECEIPTS[path]


# ---------------------------------------------------------------- the task and the stream (check_e54_brief.py; exp11_cell.py lines 224-265)
EOS, BOS, BATCH = 256, 257, 64
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
LET = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ap = lambda op, m, a, b: {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]


def render(task, a, b, k, rng):
    op, m, code = task
    d = [rng.choice(LET) for _ in range(k)]
    return list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")], list(f"{ap(op, m, a, b):02d}".encode()) + [EOS]


def split(task, frac, seed):
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]).shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def batches(f, seed):
    rng = Random(seed + 77); flat = [(t, pr) for t in DEV for pr in split(t, f, seed)[0]]

    def stream():
        while True:
            order = list(range(len(flat))); rng.shuffle(order)
            for i in order:
                t, (a, b) = flat[i]
                yield render(t, a, b, rng.randint(0, 12), rng)
    st = stream()
    while True:
        yield [next(st) for _ in range(BATCH)]


def batch_arrays(items):
    """exp45.batch_tensor: x, y (int64) and the mask of the answer's positions (the two answer bytes and the end symbol)."""
    rows = [p + t for p, t in items]
    L = max(len(r) for r in rows) + 1
    data = np.full((len(rows), L), BOS, dtype=np.int64)
    for i, r in enumerate(rows):
        data[i, 1:len(r) + 1] = r
    x, y = data[:, :-1].copy(), data[:, 1:].copy()
    mask = np.zeros(y.shape, dtype=bool)
    for i, (p, t) in enumerate(items):
        mask[i, len(p):len(p) + len(t)] = True
    return x, y, mask


_BATCHES = []


def stream_batches(n):
    """The first n batches of the stream every run of E42, E49 and E50 was fed (half of the pairs taught, seed 73129)."""
    if len(_BATCHES) < n:
        bt = batches(.5, SEED); _BATCHES.clear()
        for _ in range(max(n, 1 + N_FURTHER)):
            _BATCHES.append(batch_arrays(next(bt)))
    return _BATCHES[:n]


def stream_digest(n):
    """exp26_candidate.Digest over the first n batches: the shape and the bytes of x, y and the mask, in order."""
    h = hashlib.sha256(); bt = batches(.5, SEED)
    for _ in range(n):
        for t in batch_arrays(next(bt)):
            h.update(str(tuple(t.shape)).encode()); h.update(np.ascontiguousarray(t).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- a network
def sigmoid(v):
    return 0.5 * (1.0 + np.tanh(0.5 * v))


class Net:
    """The parameters (float64), the masks of the three maps that can be masked, and the fixed code where there is one."""

    def __init__(self, name, dt=np.float64):
        ref, path, rpath, self.when, self.what = NETS[name]
        T = load(path, ref)
        self.name, self.receipt_path, self.dt = name, rpath, dt   # dt = float32 only for the comparison of precisions in test 2
        self.coded = name.startswith("C-dense")
        self.code = T["code_table"].astype(dt) if self.coded else None
        ones = lambda s: np.ones(s, dtype=bool)
        self.M = {"l1": ones((D, 8)), "l2": ones((D, D)), "l3": ones((257, D)),
                  "hh": np.tile(T["mask"].astype(bool), (3, 1)),
                  "ih": np.tile(T["mask_in"].astype(bool), (3, 1)) if "mask_in" in T and not name.startswith("F10") else ones((3 * U, D)),
                  "lm": T["mask_out"].astype(bool) if "mask_out" in T else ones((D, U))}
        names = TENSORS + BIASES
        self.P = {k: T[k].astype(dt) for k in names}
        self.outside_zero = {k: bool((self.P[WNAME[k]][~self.M[k]] == 0).all()) for k in MAPS}
        self.trained = [k for k in names if not (self.coded and k in ("book.weight", "l1.weight", "l1.bias"))]
        self.maps = [k for k in MAPS if not (self.coded and k == "l1")]
        self.raw = T

    def Wm(self, k):
        return self.P[WNAME[k]] * self.M[k]


def forward(net, x, y, mask):
    """The forward pass and the loss; everything the backward pass needs is kept.
    Two arrangements that change no value: the byte's code z1 = tanh(l1(book)) (or the fixed code) depends on the byte alone, so it and
    the input map's output are formed once for each of the 258 bytes and looked up; and the readout (z2, logits) is formed at the
    answer's positions only, the only positions that enter the loss."""
    Pm = net.P
    B, L = x.shape
    xf = x.reshape(-1)
    idx = np.flatnonzero(mask.reshape(-1))                          # the answer's positions, in the order of logits[mask]
    Z1 = net.code if net.coded else np.tanh(Pm["book.weight"] @ Pm["l1.weight"].T + Pm["l1.bias"])     # (258, 512)
    gi = (Z1 @ net.Wm("ih").T + Pm["rnn.bias_ih_l0"])[x]            # (B, L, 768)
    WhT, bh = net.Wm("hh").T, Pm["rnn.bias_hh_l0"]
    c = dict(x=x, xf=xf, idx=idx, Z1=Z1)
    for k in ("hp", "r", "z", "n", "ghn"):
        c[k] = np.empty((L, B, U), dtype=net.dt)
    h = np.zeros((B, U), dtype=net.dt); Hs = np.empty((B, L, U), dtype=net.dt)
    for t in range(L):
        gh = h @ WhT + bh
        r = sigmoid(gi[:, t, :U] + gh[:, :U])
        z = sigmoid(gi[:, t, U:2 * U] + gh[:, U:2 * U])
        n = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
        c["hp"][t], c["r"][t], c["z"][t], c["n"][t], c["ghn"][t] = h, r, z, n, gh[:, 2 * U:]
        h = (1 - z) * n + z * h
        Hs[:, t] = h
    z1a, Hsa = Z1[xf[idx]], Hs.reshape(-1, U)[idx]
    z2 = np.tanh(z1a @ Pm["l2.weight"].T + Pm["l2.bias"] + Hsa @ net.Wm("lm").T + Pm["lm.bias"])
    logits = z2 @ Pm["l3.weight"].T + Pm["l3.bias"]                 # (answer positions, 257)
    lg = logits - logits.max(1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
    tgt = y.reshape(-1)[idx]
    c["loss"] = float(-logp[np.arange(len(tgt)), tgt].mean())       # the mean over the answer's positions of the whole batch
    d3 = np.exp(logp); d3[np.arange(len(tgt)), tgt] -= 1.0
    order = np.argsort(xf, kind="stable"); xs = xf[order]
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    c.update(z1a=z1a, Hsa=Hsa, Hs=Hs, z2=z2, logits=logits, d3=(d3 / len(tgt)).astype(net.dt), order=order, starts=starts, present=xs[starts])
    return c


def by_byte(c, rows):
    """(258, K): the rows (one for each position of the batch) summed over the positions that hold the same byte."""
    out = np.zeros((258, rows.shape[1]), dtype=rows.dtype)
    out[c["present"]] = np.add.reduceat(rows[c["order"]], c["starts"], axis=0)
    return out


def backward(net, c, F):
    """The gradient of every trained parameter when the error crosses map k backward through the transpose of F[k].
    F[k] = the (masked) weight matrix itself gives the exact gradient.  A map's own gradient is always the outer product of the
    arriving error and the map's input, inside the mask.  The gates' nonlinearities and products are differentiated exactly.
    The error at the readout is zero outside the answer's positions, so the readout's part is written on those positions only; on the
    input side the sums over positions are taken byte by byte first (z1 depends on the byte alone)."""
    B, L = c["x"].shape
    idx, Z1, z1a, z2, Hsa, d3 = c["idx"], c["Z1"], c["z1a"], c["z2"], c["Hsa"], c["d3"]
    g = {}
    g["l3.weight"], g["l3.bias"] = d3.T @ z2, d3.sum(0)
    da2 = (d3 @ F["l3"]) * (1 - z2 * z2)                            # the error after crossing l3, at the sum inside z2's tanh
    g["l2.weight"], g["l2.bias"] = da2.T @ z1a, da2.sum(0)
    g["lm.weight"], g["lm.bias"] = (da2.T @ Hsa) * net.M["lm"], da2.sum(0)
    dt = net.dt
    dHs = np.zeros((B * L, U), dtype=dt); dHs[idx] = da2 @ F["lm"]; dHs = dHs.reshape(B, L, U)
    dgi = np.empty((B, L, 3 * U), dtype=dt); gWh = np.zeros((3 * U, U), dtype=dt); gbh = np.zeros(3 * U, dtype=dt)
    carry = np.zeros((B, U), dtype=dt); Fh = F["hh"]
    for t in range(L - 1, -1, -1):
        dh = dHs[:, t] + carry
        r, z, n, ghn, hp = c["r"][t], c["z"][t], c["n"][t], c["ghn"][t], c["hp"][t]
        dan = dh * (1 - z) * (1 - n * n)                            # at the sum inside the candidate's tanh
        daz = dh * (hp - n) * z * (1 - z)
        dar = dan * ghn * r * (1 - r)
        dgh = np.concatenate([dar, daz, dan * r], 1)                # at the recurrent map's output (rows r, z, n)
        dgi[:, t, :U], dgi[:, t, U:2 * U], dgi[:, t, 2 * U:] = dar, daz, dan
        gWh += dgh.T @ hp; gbh += dgh.sum(0)
        carry = dh * z + dgh @ Fh                                   # the recurrent map is crossed again at every step back
    g["rnn.weight_hh_l0"], g["rnn.bias_hh_l0"] = gWh * net.M["hh"], gbh
    Dg = by_byte(c, dgi.reshape(-1, 3 * U))                         # (258, 768): the error at the input map's output, by byte
    g["rnn.weight_ih_l0"], g["rnn.bias_ih_l0"] = (Dg.T @ Z1) * net.M["ih"], Dg.sum(0)
    if not net.coded:
        Da2 = np.zeros((258, D), dtype=dt); np.add.at(Da2, c["xf"][idx], da2)
        DA1 = (Da2 @ F["l2"] + Dg @ F["ih"]) * (1 - Z1 * Z1)        # the error at the sum inside z1's tanh, by byte
        g["l1.weight"], g["l1.bias"] = DA1.T @ net.P["book.weight"], DA1.sum(0)
        gb = DA1 @ F["l1"]
        gb[BOS] = 0.0                                               # padding_idx: the start symbol's row receives no gradient
        g["book.weight"] = gb
    return g


# ---------------------------------------------------------------- the scale of the sweep
def scale(W, M, variant, gates):
    """The matrix of magnitudes that replaces |W| (same shape as W); every mean is over the kept entries (M) only."""
    A = np.abs(W) * M
    if variant == "V5":
        return A
    mean = A.sum() / M.sum()
    row = A.sum(1) / np.maximum(M.sum(1), 1)
    col = A.sum(0) / np.maximum(M.sum(0), 1)
    one = np.ones(W.shape, dtype=W.dtype)
    if variant == "V0" or (variant == "V1" and not gates):
        return one * mean
    if variant == "V1":
        blocks = [A[g * U:(g + 1) * U].sum() / M[g * U:(g + 1) * U].sum() for g in range(3)]
        return one * np.repeat(blocks, U).astype(W.dtype)[:, None]
    if variant == "V2":
        return one * row[:, None]
    if variant == "V3":
        return one * col[None, :]
    if variant == "V4":
        return row[:, None] * col[None, :] / mean
    raise ValueError(variant)


def feedback(net, variant, where="all"):
    """F[k] for every map: sign(W) inside the mask times the variant's magnitudes.  where = 'gru': the variant at the two GRU maps,
    V0 at the others."""
    F = {}
    for k in net.maps:
        v = variant if (where == "all" or k in GRU_MAPS) else "V0"
        Wk = net.Wm(k)
        F[k] = (np.sign(Wk) * scale(net.P[WNAME[k]], net.M[k], v, k in GRU_MAPS)).astype(net.dt)
    return F


def exact_feedback(net):
    return {k: net.Wm(k) for k in net.maps}


# ---------------------------------------------------------------- comparisons
def cos(a, b):
    a, b = a.ravel(), b.ravel()
    na, nb = np.sqrt(a @ a), np.sqrt(b @ b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def ratio(a, b):
    nb = np.sqrt((b * b).sum())
    return float(np.sqrt((a * a).sum()) / nb) if nb > 0 else float("nan")


def flat(net, g, names=None):
    return np.concatenate([g[k].ravel() for k in (names or net.trained)])


def compare(net, gs, ge):
    """Cosines and norm ratios (sweep over exact) for each weight tensor, all parameters together, and the gate blocks."""
    out = {}
    for k in TENSORS:
        if k in gs:
            out["cos " + k], out["ratio " + k] = cos(gs[k], ge[k]), ratio(gs[k], ge[k])
    for k in BIASES:
        if k in gs:
            out["cos " + k] = cos(gs[k], ge[k])
    a, b = flat(net, gs), flat(net, ge)
    out["cos all"], out["ratio all"] = cos(a, b), ratio(a, b)
    rest = [k for k in net.trained if not k.startswith("l3.")]
    out["cos all but l3"] = cos(flat(net, gs, rest), flat(net, ge, rest))
    for m in GRU_MAPS:
        for gi_, gate in enumerate("rzn"):
            out[f"cos {m} {gate}"] = cos(gs[WNAME[m]][gi_ * U:(gi_ + 1) * U], ge[WNAME[m]][gi_ * U:(gi_ + 1) * U])
    return out


# ---------------------------------------------------------------- the measurement of one network (kept in memory for the parts that follow)
SETS = [(v, "all") for v in VARIANTS] + [(v, "gru") for v in ("V2", "V3", "V4", "V5")]      # V1 is the same in both tables by definition
_MEASURED = {}


def measure(name):
    """For the first batch and the 16 further ones: the loss, and for every variant the comparison with the exact gradient.
    Also what describes the magnitudes of each map.  Returns (description, rows); rows[0] is the first batch."""
    if name not in _MEASURED:
        net = Net(name)
        Fx, Fs = exact_feedback(net), {key: feedback(net, *key) for key in SETS}
        rows = []
        for x, y, mask in stream_batches(1 + N_FURTHER):
            c = forward(net, x, y, mask)
            ge = backward(net, c, Fx)
            row = {"loss": c["loss"], "norm": float(np.sqrt((flat(net, ge) ** 2).sum()))}
            for key, F in Fs.items():
                row[key] = compare(net, backward(net, c, F), ge)
            rows.append(row)
        _MEASURED[name] = (describe(net), rows)
    return _MEASURED[name]


def describe(net):
    """For every map: the kept entries, the mean magnitude, the mean of each gate's block, the coefficient of variation (standard
    deviation over mean) of the row means, of the column means and of the magnitudes themselves, and for each variant the cosine
    between the matrix that carries the error back and the weight matrix (1 for V5)."""
    out = dict(name=net.name, what=net.what, when=net.when, coded=net.coded, maps={}, outside_zero=net.outside_zero)
    cv = lambda v: float(v.std() / v.mean())
    for k in net.maps:
        W, M = net.P[WNAME[k]], net.M[k]
        A = np.abs(W) * M
        row, col = A.sum(1) / M.sum(1), A.sum(0)[M.sum(0) > 0] / M.sum(0)[M.sum(0) > 0]
        d = dict(kept=int(M.sum()), size=int(M.size), mean=float(A.sum() / M.sum()), cv_rows=cv(row), cv_cols=cv(col), cv_entries=cv(A[M]),
                 align={v: cos(np.sign(W * M) * scale(W, M, v, k in GRU_MAPS), W * M) for v in VARIANTS})
        if k in GRU_MAPS:
            d["blocks"] = [float(A[g * U:(g + 1) * U].sum() / M[g * U:(g + 1) * U].sum()) for g in range(3)]
            d["cv_rows_in_block"] = [cv(row[g * U:(g + 1) * U]) for g in range(3)]
        out["maps"][k] = d
    return out


def stat(rows, key, field):
    v = np.array([r[key][field] for r in rows])
    return float(v.mean()), float(v.min()), float(v.max())


def cell(rows, key, field, digits=3):
    if field not in rows[0][key]:
        return "–"
    m, lo, hi = stat(rows, key, field)
    return f"{m:.{digits}f} ({lo:.{digits}f} to {hi:.{digits}f})"


def table(head, lines):
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(["---"] * len(head)) + "|")
    for ln in lines:
        print("| " + " | ".join(str(v) for v in ln) + " |")
    print()


def label(text):
    print(f"\n===== {text} =====", flush=True)


def decimals(a, b):
    """The number of leading decimals in which a and b agree, written to nine decimals."""
    sa, sb = f"{a:.9f}", f"{b:.9f}"
    n = 0
    for p, q in zip(sa, sb):
        if p != q:
            break
        n += 1
    return max(0, n - sa.index(".") - 1) if sa.split(".")[0] == sb.split(".")[0] else 0


# ---------------------------------------------------------------- part: stream
def part_stream(names):
    label("stream: the batches rebuilt here against the receipts' digest of the first 256 batches")
    mine = stream_digest(256)
    for rp in ("e42/e42_G_S_73129.json", "e42/e42_F10_S_73129.json", "e49/e49_in16_S_73129.json", "e50/e50_C-dense_S_73129.json"):
        row = next(r for r in receipt(rp)["curve"] if r["step"] == 256)
        print(f"{rp}: recorded {row['data_digest']['sha256'][:16]} over {row['data_digest']['batches']} batches; rebuilt {mine[:16]}; equal {row['data_digest']['sha256'] == mine}")
    bs = stream_batches(1 + N_FURTHER)
    print(f"batches used: the first (the receipts' batch) and the next {N_FURTHER}; items a batch {bs[0][0].shape[0]}; positions a batch {sorted({b[0].shape[1] for b in bs})};"
          f" answer positions a batch {sorted({int(b[2].sum()) for b in bs})}")
    print("networks:")
    for n, v in NETS.items():
        print(f"  {n:14s} {v[4]}  [{v[0]}:{P}{v[1]}]")


# ---------------------------------------------------------------- part: forward
def part_forward(names):
    """This script's forward (float64) against the reviewer's readers (float32, written from the briefs), on the first batch: the
    logits at the answer's positions.  And E50's recorded loss on the first batch at the start."""
    import e42_cells_reader as R42
    import e49_interface_reader as R49
    import e50_code_reader as R50
    label("forward: this script's logits at the answer's positions against the reviewer's readers, first batch")
    x, y, mask = stream_batches(1)[0]
    for name in names:
        net = Net(name); T = net.raw
        c = forward(net, x, y, mask)
        cells = T["mask"].astype(bool)
        if name.startswith("C-dense"):
            F = R49.Interface(SEED, D, U); code = R50.Code(SEED, amplitude=float(T["code_amplitude"]))
            assert np.array_equal(code.table(), T["code_table"]), "the reader's code table is not the file's"
            lg = R50.forward(T, cells, F, code, x)[1]; rd = "e50_code_reader.forward"
        elif name.startswith("in16"):
            F = R49.Interface(SEED, 16, U)
            assert np.array_equal(F.mask_in, T["mask_in"].astype(bool)) and np.array_equal(F.mask_out, T["mask_out"].astype(bool))
            lg = R49.forward(T, cells, F, x)[1]; rd = "e49_interface_reader.forward"
        else:
            lg = R42.forward(T, R49.Period1(cells), x)[1]; rd = "e42_cells_reader.forward"
        diff = float(np.abs(lg[mask].astype(np.float64) - c["logits"]).max())
        print(f"{name:14s} {rd}: largest difference of a logit {diff:.1e} (largest logit magnitude {float(np.abs(c['logits']).max()):.2f}); loss here {c['loss']:.9f};"
              f" weights outside the masks all zero {all(net.outside_zero.values())}")
        if name == "C-dense_start":
            rec = receipt(net.receipt_path)["first_batch_loss_start"]
            print(f"               E50's receipt, loss on the first batch at the start: {rec['B']:.9f} (B) {rec['S']:.9f} (S); here {c['loss']:.9f}")


# ---------------------------------------------------------------- part: test 1
T1_NETS = ("G_S", "F10_B", "in16_start", "in16_S", "C-dense_B")


def five_point(net, W, ix, x, y, mask, h):
    """The derivative of the loss with respect to the single weight W[ix] by the five-point central difference."""
    w0 = W[ix]; f = []
    for s in (2, 1, -1, -2):
        W[ix] = w0 + s * h
        f.append(forward(net, x, y, mask)["loss"])
    W[ix] = w0
    return (8 * (f[1] - f[2]) - (f[0] - f[3])) / (12 * h)


def part_test1(names):
    """The exact gradient against a five-point central difference of the loss (step 1e-3, float64), for single weights of every tensor.
    Group A: drawn at random among all kept entries (3 of each weight tensor, 2 of each bias).  Group B: drawn at random among the kept
    entries whose exact gradient is at least a tenth of the tensor's largest in magnitude (3 and 2), where a relative error has a meaning:
    the difference itself carries a rounding error of about 1.5 * 2.2e-16 * loss / step, so a weight whose gradient is near that size
    cannot be compared in relative terms.  The byte table's weights are drawn among the rows of the bytes that occur in the batch, the
    start symbol's row excepted (training gives that row no gradient)."""
    label("test 1: the exact gradient against finite differences of the loss, first batch")
    x, y, mask = stream_batches(1)[0]
    h = 1e-3
    for name in names:
        net = Net(name)
        c = forward(net, x, y, mask)
        g = backward(net, c, exact_feedback(net))
        rng = np.random.default_rng(20261117)
        A = dict(n=0, zero=0, rel=0.0, at=0.0, err=0.0); Bg = dict(n=0, rel=0.0, least=np.inf); per = {}
        for k in net.trained:
            W = net.P[k]
            ok = np.ones(W.shape, dtype=bool)                       # the entries a weight may be drawn from
            if k == "book.weight":
                ok[:] = False; ok[sorted(set(x.ravel().tolist()) - {BOS})] = True
            elif W.ndim == 2:
                ok = net.M[next(m for m in MAPS if WNAME[m] == k)]
            na = 3 if W.ndim == 2 else 2
            pool = np.flatnonzero(ok)
            big = np.flatnonzero(ok & (np.abs(g[k]) >= 0.1 * np.abs(g[k]).max()))
            index = lambda flat_: tuple(int(v) for v in np.unravel_index(int(flat_), W.shape))
            ta = [index(i) for i in rng.choice(pool, na, replace=False)]
            tb = [index(i) for i in rng.choice(big, min(na, len(big)), replace=False)]
            per[k] = 0.0
            for ix in ta:
                fd, ga = five_point(net, W, ix, x, y, mask, h), float(g[k][ix])
                A["n"] += 1
                if fd == 0.0 and ga == 0.0:
                    A["zero"] += 1
                    continue
                e = abs(fd - ga) / max(abs(fd), abs(ga))
                if e > A["rel"]:
                    A.update(rel=e, at=abs(ga), err=abs(fd - ga))
            for ix in tb:
                fd, ga = five_point(net, W, ix, x, y, mask, h), float(g[k][ix])
                e = abs(fd - ga) / max(abs(fd), abs(ga))
                Bg["n"] += 1; Bg["rel"] = max(Bg["rel"], e); Bg["least"] = min(Bg["least"], abs(ga)); per[k] = max(per[k], e)
        print(f"{name:14s} loss {c['loss']:.6f}; rounding error of the difference at least about {1.5 * 2.2e-16 * c['loss'] / h:.0e}; {A['n'] + Bg['n']} weights over {len(net.trained)} tensors")
        print(f"               group A ({A['n']} weights among all kept entries): both exactly zero in {A['zero']}; largest relative error {A['rel']:.1e}, at a weight whose gradient is"
              f" {A['at']:.1e} in magnitude (absolute error there {A['err']:.1e})")
        print(f"               group B ({Bg['n']} weights with a gradient of at least a tenth of the tensor's largest; the least of them {Bg['least']:.1e}): largest relative error {Bg['rel']:.1e}")
        print("               group B, largest relative error by tensor: " + ", ".join(f"{k} {v:.0e}" for k, v in per.items()))
        if not net.coded:
            fd = five_point(net, net.P["book.weight"], (BOS, 0), x, y, mask, h)
            print(f"               the start symbol's row of the byte table: derivative of the loss by finite difference {fd:.3e};"
                  f" gradient used in training (padding row) {float(g['book.weight'][BOS, 0]):.1f}")


# ---------------------------------------------------------------- part: test 2
RECORDED = {"recurrent_matrix": "rnn.weight_hh_l0", "rnn.weight_ih_l0": "rnn.weight_ih_l0", "lm.weight": "lm.weight", "rnn.weight_hh_l0": "rnn.weight_hh_l0"}


def part_test2(names):
    label("test 2: V0 on the first batch against the cosines the receipts record (torch, float32)")
    lines = []
    for name in names:
        desc, rows = measure(name)
        r0 = rows[0][("V0", "all")]
        recs = [(NETS[name][2], "cosine_first_batch_" + desc["when"])]
        if name == "F10_start":
            recs += [("e42/e42_F10_S_73129.json", "cosine_first_batch_start"), ("e42/e42_F10_B_73129.json", "cosine_first_batch_start")]
        if name == "in16_start":
            recs.append(("e49/e49_in16_B_73129.json", "cosine_first_batch_start"))
        if name == "C-dense_start":
            recs.append(("e50/e50_C-dense_B_73129.json", "cosine_first_batch_start"))
        for rp, key in recs:
            for rk, v in receipt(rp)[key].items():
                if rk in RECORDED:
                    mine = r0["cos " + RECORDED[rk]]
                    lines.append((name, SHORT[RECORDED[rk]], rp.split("/")[-1], key[19:], f"{v:.7f}", f"{mine:.7f}", f"{mine - v:+.1e}", decimals(mine, v)))
    table(("network", "tensor", "receipt", "at", "recorded", "here (V0)", "difference", "decimals that agree"), lines)
    d = [abs(float(ln[6])) for ln in lines]
    print(f"{len(lines)} recorded cosines compared; largest difference {max(d):.1e}; decimals that agree: least {min(ln[7] for ln in lines)}, most {max(ln[7] for ln in lines)}")
    a, b = load("e49/e49_in16_S_73129_start.pt", REF49), load("e49/e49_in16_B_73129_start.pt", REF49)
    print("start files of the two rules hold the same tensors: E49 in16", all(np.array_equal(a[k], b[k]) for k in a), end="; ")
    a, b = load("e50/e50_C-dense_S_73129_start.pt", REF50), load("e50/e50_C-dense_B_73129_start.pt", REF50)
    print("E50 C-dense", all(np.array_equal(a[k], b[k]) for k in a))
    f = load(NETS["F10_start"][1], REF49)
    print("the start file of E49's check 2 (F10) and E50's C-dense start file hold the same tensors:", all(np.array_equal(f[k], a[k]) for k in f))


# ---------------------------------------------------------------- part: test 3
def part_test3(names):
    label("test 3: V5 (signs times the exact magnitudes) against the exact gradient, 17 batches")
    for name in names:
        desc, rows = measure(name)
        dev = max(abs(1 - v) for r in rows for k, v in r[("V5", "all")].items() if k.startswith("cos "))
        rat = max(abs(1 - v) for r in rows for k, v in r[("V5", "all")].items() if k.startswith("ratio "))
        nt = sum(1 for k in rows[0][("V5", "all")] if k.startswith("cos "))
        print(f"{name:14s} largest |1 - cosine| over {nt} cosines (weights, biases, all parameters, gate blocks) and {len(rows)} batches {dev:.1e}; largest |1 - norm ratio| {rat:.1e}")


# ---------------------------------------------------------------- part: magnitudes
MAPNAME = {"l1": "l1", "ih": "input map", "hh": "recurrent map", "l2": "l2", "lm": "lm", "l3": "l3"}


def part_magnitudes(names):
    label("magnitudes: how unequal the magnitudes are that a scale replaces (means over the kept entries)")
    for name in names:
        desc, _ = measure(name)
        print(f"**{name}** ({desc['what']})\n")
        lines = []
        for k, d in desc["maps"].items():
            bl = " / ".join(f"{v:.4f}" for v in d["blocks"]) if "blocks" in d else "–"
            lines.append((MAPNAME[k], f"{d['kept']:,} of {d['size']:,}", f"{d['mean']:.4f}", bl, f"{d['cv_rows']:.2f}", f"{d['cv_cols']:.2f}", f"{d['cv_entries']:.2f}",
                          *(f"{d['align'][v]:.3f}" for v in VARIANTS[:5])))
        table(("map", "kept entries", "mean magnitude", "by gate block r / z / n", "CV of row means", "CV of column means", "CV of magnitudes",
               "V0", "V1", "V2", "V3", "V4"), lines)


# ---------------------------------------------------------------- part: tables
COLS = [("cos " + k, SHORT[k]) for k in TENSORS] + [("cos all", "all parameters"), ("cos all but l3", "all but l3")]
RCOLS = [("ratio " + k, SHORT[k]) for k in TENSORS] + [("ratio all", "all parameters")]
GCOLS = [(f"cos {m} {g}", f"{MAPNAME[m]}, {g}") for m in GRU_MAPS for g in "rzn"]
BCOLS = [("cos " + k, k) for k in BIASES]
ROWS_ALL = [("V0", "all"), ("V1", "all"), ("V2", "all"), ("V3", "all"), ("V4", "all")]
ROWS_GRU = [("V1", "all"), ("V2", "gru"), ("V3", "gru"), ("V4", "gru"), ("V5", "gru")]


def part_tables(names):
    label(f"tables: every variant against the exact gradient, batches 2 to {1 + N_FURTHER} of the stream: mean (least to greatest)")
    for name in names:
        desc, rows = measure(name)
        fur = rows[1:]
        cols = [c for c in COLS if c[0] in fur[0][("V0", "all")]]
        rcols = [c for c in RCOLS if c[0] in fur[0][("V0", "all")]]
        ls = [r["loss"] for r in fur]; ns = [r["norm"] for r in fur]
        print(f"**{name}** ({desc['what']}). Loss on the {len(fur)} batches: mean {np.mean(ls):.4f} (least {min(ls):.4f}, greatest {max(ls):.4f});"
              f" first batch {rows[0]['loss']:.4f}. Norm of the exact gradient: mean {np.mean(ns):.3f} ({min(ns):.3f} to {max(ns):.3f}).\n")
        print(f"{name}, cosine with the exact gradient, variant at every map:\n")
        table(("variant",) + tuple(c[1] for c in cols), [("V0, first batch",) + tuple(f"{rows[0][('V0', 'all')][c[0]]:.3f}" for c in cols)]
              + [(key[0],) + tuple(cell(fur, key, c[0]) for c in cols) for key in ROWS_ALL])
        print(f"{name}, cosine with the exact gradient, variant at the two GRU maps only (V0 at the other maps; V5 here is the upper limit for a scale at the GRU maps):\n")
        table(("variant",) + tuple(c[1] for c in cols), [(key[0],) + tuple(cell(fur, key, c[0]) for c in cols) for key in ROWS_GRU])
        print(f"{name}, norm of the sweep's gradient over the norm of the exact gradient:\n")
        table(("variant", "where") + tuple(c[1] for c in rcols),
              [(key[0], "every map" if key[1] == "all" else "GRU maps") + tuple(cell(fur, key, c[0], 2) for c in rcols) for key in ROWS_ALL + ROWS_GRU[1:]])
        print(f"{name}, cosine for each gate's block of the two GRU maps:\n")
        table(("variant", "where") + tuple(c[1] for c in GCOLS),
              [(key[0], "every map" if key[1] == "all" else "GRU maps") + tuple(cell(fur, key, c[0]) for c in GCOLS) for key in ROWS_ALL + ROWS_GRU[1:]])
        bcols = [c for c in BCOLS if c[0] in fur[0][("V0", "all")]]
        print(f"{name}, cosine for the biases, variant at every map (means):\n")
        table(("variant",) + tuple(c[1] for c in bcols), [(key[0],) + tuple(f"{stat(fur, key, c[0])[0]:.3f}" for c in bcols) for key in ROWS_ALL])


def part_summary(names):
    label(f"summary: V0 against the best of V1 to V4 at every map, mean cosine over batches 2 to {1 + N_FURTHER}; in brackets the number of the {N_FURTHER} batches on which the variant is above V0")
    lines = []
    for name in names:
        desc, rows = measure(name)
        fur = rows[1:]
        for f, short in COLS:
            if f not in fur[0][("V0", "all")] or f == "cos l3.weight":
                continue
            v0 = np.array([r[("V0", "all")][f] for r in fur])
            alt = {key[0]: np.array([r[key][f] for r in fur]) for key in ROWS_ALL[1:]}
            best = max(alt, key=lambda k: alt[k].mean())
            gru = {key[0]: np.array([r[key][f] for r in fur]) for key in ROWS_GRU[:4]}
            bg = max(gru, key=lambda k: gru[k].mean())
            ceil = np.array([r[("V5", "gru")][f] for r in fur])
            lines.append((name, short, f"{v0.mean():.3f}", f"{best} {alt[best].mean():.3f}", f"{alt[best].mean() - v0.mean():+.3f} [{int((alt[best] > v0).sum())}]",
                          f"{bg} {gru[bg].mean():.3f}", f"{gru[bg].mean() - v0.mean():+.3f} [{int((gru[bg] > v0).sum())}]", f"{ceil.mean():.3f}"))
    table(("network", "tensor", "V0", "best at every map", "gain", "best at the GRU maps only", "gain", "exact magnitudes at the GRU maps only"), lines)


# ---------------------------------------------------------------- main
PARTS = {"stream": part_stream, "forward": part_forward, "test1": part_test1, "test2": part_test2, "test3": part_test3,
         "magnitudes": part_magnitudes, "tables": part_tables, "summary": part_summary}

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    unknown = [a for a in args if a not in PARTS and a not in NETS and a != "all"]
    assert not unknown, f"unknown part or network: {unknown}; parts {list(PARTS)}; networks {list(NETS)}"
    parts = [a for a in args if a in PARTS] or list(PARTS)
    if "all" in args:
        parts = list(PARTS)
    chosen = [a for a in args if a in NETS]
    print(f"sweep_scale_variants.py; parts {parts}; networks {chosen or 'every one'}; numpy {np.__version__}; threads OPENBLAS {os.environ.get('OPENBLAS_NUM_THREADS')} OMP {os.environ.get('OMP_NUM_THREADS')}")
    for p_ in parts:
        t = time.time()
        PARTS[p_](chosen or (list(T1_NETS) if p_ == "test1" else list(NETS)))
        print(f"[{p_}: {time.time() - t:.0f} s]", flush=True)
    print(f"\ntotal {time.time() - T0:.0f} s")
