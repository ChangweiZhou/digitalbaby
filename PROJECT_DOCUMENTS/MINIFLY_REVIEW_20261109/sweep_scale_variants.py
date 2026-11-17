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
    "F10_start":     (REF50, "e50/e50_C-dense_S_73129_start.pt", "e42/e42_F10_S_73129.json", "start", "E42 F10 at its start (tensors of the E50 start file, byte table and l1 in use)"),
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

    def __init__(self, name):
        ref, path, rpath, self.when, self.what = NETS[name]
        T = load(path, ref)
        self.name, self.receipt_path = name, rpath
        self.coded = name.startswith("C-dense")
        self.code = T["code_table"].astype(np.float64) if self.coded else None
        ones = lambda s: np.ones(s, dtype=bool)
        self.M = {"l1": ones((D, 8)), "l2": ones((D, D)), "l3": ones((257, D)),
                  "hh": np.tile(T["mask"].astype(bool), (3, 1)),
                  "ih": np.tile(T["mask_in"].astype(bool), (3, 1)) if "mask_in" in T and not name.startswith("F10") else ones((3 * U, D)),
                  "lm": T["mask_out"].astype(bool) if "mask_out" in T else ones((D, U))}
        names = TENSORS + BIASES
        self.P = {k: T[k].astype(np.float64) for k in names}
        self.outside_zero = {k: bool((self.P[WNAME[k]][~self.M[k]] == 0).all()) for k in MAPS}
        self.trained = [k for k in names if not (self.coded and k in ("book.weight", "l1.weight", "l1.bias"))]
        self.maps = [k for k in MAPS if not (self.coded and k == "l1")]
        self.raw = T

    def Wm(self, k):
        return self.P[WNAME[k]] * self.M[k]


def forward(net, x, y, mask):
    """The forward pass and the loss; everything the backward pass needs is kept."""
    Pm = net.P
    B, L = x.shape
    c = dict(x=x)
    if net.coded:
        z1 = net.code[x]
    else:
        c["u"] = Pm["book.weight"][x]
        z1 = np.tanh(c["u"] @ Pm["l1.weight"].T + Pm["l1.bias"])
    gi = z1 @ net.Wm("ih").T + Pm["rnn.bias_ih_l0"]
    WhT, bh = net.Wm("hh").T, Pm["rnn.bias_hh_l0"]
    h = np.zeros((B, U))
    for k in ("hp", "r", "z", "n", "ghn"):
        c[k] = np.empty((L, B, U))
    Hs = np.empty((B, L, U))
    for t in range(L):
        gh = h @ WhT + bh
        r = sigmoid(gi[:, t, :U] + gh[:, :U])
        z = sigmoid(gi[:, t, U:2 * U] + gh[:, U:2 * U])
        n = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
        c["hp"][t], c["r"][t], c["z"][t], c["n"][t], c["ghn"][t] = h, r, z, n, gh[:, 2 * U:]
        h = (1 - z) * n + z * h
        Hs[:, t] = h
    z2 = np.tanh(z1 @ Pm["l2.weight"].T + Pm["l2.bias"] + Hs @ net.Wm("lm").T + Pm["lm.bias"])
    logits = z2 @ Pm["l3.weight"].T + Pm["l3.bias"]
    lg = logits[mask]                                               # (answer positions, 257)
    lg = lg - lg.max(1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
    tgt = y[mask]
    c["loss"] = float(-logp[np.arange(len(tgt)), tgt].mean())
    d = np.exp(logp); d[np.arange(len(tgt)), tgt] -= 1.0
    dlog = np.zeros((B, L, 257)); dlog[mask] = d / len(tgt)
    c.update(z1=z1, Hs=Hs, z2=z2, logits=logits, dlog=dlog)
    return c


def loss_only(net, x, y, mask):
    return forward(net, x, y, mask)["loss"]


def backward(net, c, F):
    """The gradient of every trained parameter when the error crosses map k backward through the transpose of F[k].
    F[k] = the (masked) weight matrix itself gives the exact gradient.  A map's own gradient is always the outer product of the
    arriving error and the map's input, inside the mask.  The gates' nonlinearities and products are differentiated exactly."""
    B, L = c["x"].shape
    z1, z2, Hs = c["z1"].reshape(-1, D), c["z2"].reshape(-1, D), c["Hs"].reshape(-1, U)
    g = {}
    d3 = c["dlog"].reshape(-1, 257)
    g["l3.weight"], g["l3.bias"] = d3.T @ z2, d3.sum(0)
    da2 = (d3 @ F["l3"]) * (1 - z2 * z2)
    g["l2.weight"], g["l2.bias"] = da2.T @ z1, da2.sum(0)
    g["lm.weight"], g["lm.bias"] = (da2.T @ Hs) * net.M["lm"], da2.sum(0)
    dHs = (da2 @ F["lm"]).reshape(B, L, U)
    dgi = np.empty((B, L, 3 * U)); gWh = np.zeros((3 * U, U)); gbh = np.zeros(3 * U)
    carry = np.zeros((B, U)); Fh = F["hh"]
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
    dgi = dgi.reshape(-1, 3 * U)
    g["rnn.weight_ih_l0"], g["rnn.bias_ih_l0"] = (dgi.T @ z1) * net.M["ih"], dgi.sum(0)
    if not net.coded:
        da1 = (da2 @ F["l2"] + dgi @ F["ih"]) * (1 - z1 * z1)
        u = c["u"].reshape(-1, 8)
        g["l1.weight"], g["l1.bias"] = da1.T @ u, da1.sum(0)
        gb = np.zeros((258, 8)); np.add.at(gb, c["x"].reshape(-1), da1 @ F["l1"])
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
    one = np.ones(W.shape)
    if variant == "V0" or (variant == "V1" and not gates):
        return one * mean
    if variant == "V1":
        blocks = [A[g * U:(g + 1) * U].sum() / M[g * U:(g + 1) * U].sum() for g in range(3)]
        return one * np.repeat(blocks, U)[:, None]
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
        F[k] = np.sign(Wk) * scale(net.P[WNAME[k]], net.M[k], v, k in GRU_MAPS)
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
