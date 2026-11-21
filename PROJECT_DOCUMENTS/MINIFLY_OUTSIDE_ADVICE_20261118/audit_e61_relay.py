#!/usr/bin/env python3
"""Outside audit of the brief of E61 (the error relayed over a limited number of steps), 18 November 2026.

numpy and the standard library only. It reads files of the default branch with `git show`, executes no code of the repository,
trains nothing and writes nothing. Everything is computed in double precision.

    python3 audit_e61_relay.py            all parts (about ten minutes on one core)
    python3 audit_e61_relay.py quick      parts 1 to 3 and 5 only (about one minute)

Part 1  the network of E50's stage 3 (`C46-one`) rebuilt from the text of the source, and what is dense in it
Part 2  validation of this file's forward and backward passes against figures stored in the record's receipts, and against
        central differences
Part 3  how far an error travels: cells reached directly, and by relay
Part 4  what a limit on relay removes from the gradient of the cells' own weights, in the record's trained networks, for two
        forms of limit: (a) the brief's (every path cut K positions back) and (b) the cell's own path kept for the whole item
        and only the number of crossings between cells limited
Part 5  whether the record's trained networks use the connections between cells: the record's own scores computed again here
        from the saved weights, and the same scores when a cell receives nothing item-specific from the other cells
Part 6  updates to 80 recomputed from the receipts of E50 and of E42 (the figures behind the ratios 2.22, 2.23 and 2.68)
"""
import hashlib
import io
import json
import pickle
import subprocess
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path
from random import Random

import numpy as np

REF = "e647e7d7"                       # the default branch at the commit the brief names
M = "MOD97_LOCAL_RULE_20261106/"
E50 = M + "exp45/e50/"
SEED = 73129
ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                      cwd=str(Path(__file__).resolve().parent)).stdout.strip()


def git_bytes(path):
    r = subprocess.run(["git", "show", f"{REF}:{path}"], capture_output=True, cwd=ROOT)
    assert r.returncode == 0, (path, r.stderr.decode()[:200])
    return r.stdout


# ---------------------------------------------------------------- a torch.save file read without torch
_DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64, "IntStorage": np.int32,
       "BoolStorage": np.bool_}


class _Storage:
    def __init__(self, name):
        self.name = name


def _tensor(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
    n = int(np.prod(size)) if len(size) else 1
    return storage[offset:offset + n].reshape(size).copy()


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, requires_grad, hooks: data
        if module == "torch" and name.endswith("Storage"):
            return _Storage(name)
        raise pickle.UnpicklingError(f"refused {module}.{name}")

    def persistent_load(self, pid):
        kind, st, key, loc, numel = pid
        return np.frombuffer(self.zf.read(f"{self.prefix}/data/{key}"), dtype=_DT[st.name])


def load_pt(raw):
    zf = zipfile.ZipFile(io.BytesIO(raw))
    pk = [n for n in zf.namelist() if n.endswith("/data.pkl")][0]
    return _Unpickler(io.BytesIO(zf.read(pk)), zf, pk[:-len("/data.pkl")]).load()


# ---------------------------------------------------------------- the items and the stream (from the text of exp11_cell.py, exp45.py)
EOS, BOS = 256, 257
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
KMAX, BATCH = 12, 64


def apply(op, m, a, b):
    return {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]


def render(task, a, b, k, rng):
    op, m, code = task
    d = [rng.choice(DISTRACT) for _ in range(k)]
    prompt = list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")]
    return prompt, list(f"{apply(op, m, a, b):02d}".encode()) + [EOS]


def split(task, frac, seed):
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    rng = Random(seed * 1000 + round(frac * 100) + m * 7 + code[0])
    rng.shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def stream(tasks_pairs, seed, kmax):
    rng = Random(seed + 77)
    flat = [(t, pr) for t, pairs in tasks_pairs for pr in pairs]
    while True:
        order = list(range(len(flat)))
        rng.shuffle(order)
        for i in order:
            t, (a, b) = flat[i]
            yield render(t, a, b, rng.randint(0, kmax), rng)


def batch_arrays(items):
    rows = [p + t for p, t in items]
    L = max(len(r) for r in rows) + 1
    data = np.full((len(rows), L), BOS, np.int64)
    for i, r in enumerate(rows):
        data[i, 1:len(r) + 1] = r
    x, y = data[:, :-1], data[:, 1:]
    mask = np.zeros(y.shape, bool)
    for i, (p, t) in enumerate(items):
        mask[i, len(p):len(p) + len(t)] = True
    return np.ascontiguousarray(x), np.ascontiguousarray(y), mask


def letters_of(item):
    return len(item[0]) - 7             # a prompt is the task byte, two digits, the operation, k letters, two digits and "="


# ---------------------------------------------------------------- the network of E50's stage 3, from the text of the source
U, D, ON, SIDE, RADIUS, N_CONN = 256, 512, 11, 16, 3, 10
USED = sorted(set(b"0123456789+-><=ABCDEFGHIJKLMNOPQRSTUVWXYZpqvw")) + [BOS]


def draw_code(seed):                    # exp50_code.draw_code
    vals = list(range(D))
    Random(f"e50|code|{seed}").shuffle(vals)
    return {b: vals[ON * k: ON * k + ON] for k, b in enumerate(USED)}


def draw_one(seed):                     # exp50_code.draw_one: 46 values a cell, one of every used byte
    own = draw_code(seed)
    rng = Random(f"e50|one|{seed}")
    return [sorted(own[b][rng.randrange(ON)] for b in USED) for _ in range(U)]


def neighbourhood(i):                   # e42_cells.neighbourhood
    r, c = divmod(i, SIDE)
    return sorted({((r + dr) % SIDE) * SIDE + (c + dc) % SIDE for dr in range(-RADIUS, RADIUS + 1)
                   for dc in range(-RADIUS, RADIUS + 1)} - {i})


def draw_connections(seed, N=N_CONN):   # e42_cells.draws, the connections only
    rc = Random(f"e42|connections|{seed}")
    return [sorted(rc.sample(sorted(neighbourhood(i)), N)) for i in range(U)]


class Net:
    """The weights of one saved file with the masks, in double precision."""

    def __init__(self, W, amplitude, rule):
        self.rule = rule                                         # "B": exact; "S": the sweep
        own = draw_code(SEED)
        self.table = np.zeros((258, D))
        for b, vs in own.items():
            self.table[b, vs] = amplitude
        self.owner = [-1] * D
        for b, vs in own.items():
            for v in vs:
                self.owner[v] = b
        self.min = np.zeros((U, D), bool)
        for i, s in enumerate(draw_one(SEED)):
            self.min[i, s] = True
        self.src = draw_connections(SEED)
        self.mrec = np.eye(U, dtype=bool)
        for i, a in enumerate(self.src):
            self.mrec[i, a] = True
        f = lambda k: np.asarray(W[k], np.float64)
        self.Wi = f("rnn.weight_ih_l0") * np.tile(self.min, (3, 1))
        self.bi = f("rnn.bias_ih_l0")
        self.Wh = f("rnn.weight_hh_l0") * np.tile(self.mrec, (3, 1))
        self.bh = f("rnn.bias_hh_l0")
        self.Wm, self.bm = f("lm.weight"), f("lm.bias")
        self.W2, self.b2 = f("l2.weight"), f("l2.bias")
        self.W3, self.b3 = f("l3.weight"), f("l3.bias")
        self.raw_outside = float(np.abs(f("rnn.weight_hh_l0")[~np.tile(self.mrec, (3, 1))]).max())
        self.set_rule(rule)

    def set_rule(self, rule):
        """The maps over which an error is sent back: the weights (B), or their signs times one scale for the map (S)."""
        self.rule = rule
        m3 = np.tile(self.mrec, (3, 1))
        if rule == "B":
            self.Sh, self.Sm, self.S3 = self.Wh, self.Wm, self.W3
        else:
            self.Sh = np.sign(self.Wh) * (np.abs(self.Wh).sum() / m3.sum())           # e42_cells.Layer.sign_scale
            self.Sm = np.sign(self.Wm) * np.abs(self.Wm).mean()                       # exp3_state.SignLinear
            self.S3 = np.sign(self.W3) * np.abs(self.W3).mean()
        idx = np.arange(U)
        self.Sh_diag = np.stack([self.Sh[g * U + idx, idx] for g in range(3)])        # (3, U): a cell's reading of itself
        self.Sh_cross = self.Sh.copy()
        for g in range(3):
            self.Sh_cross[g * U + idx, idx] = 0.0


def sig(a):
    return 1.0 / (1.0 + np.exp(-a))


def forward(n, x, y, mask):
    """The loss and everything the backward passes need."""
    B, L = x.shape
    z1 = n.table[x]                                             # (B, L, 512)
    gi = z1 @ n.Wi.T + n.bi
    h = np.zeros((B, U))
    c = dict(r=[], z=[], c=[], ghn=[], hp=[])
    Hs = np.empty((B, L, U))
    for t in range(L):
        gh = h @ n.Wh.T + n.bh
        r = sig(gi[:, t, :U] + gh[:, :U])
        z = sig(gi[:, t, U:2 * U] + gh[:, U:2 * U])
        cand = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
        for k, v in zip(("r", "z", "c", "ghn", "hp"), (r, z, cand, gh[:, 2 * U:], h)):
            c[k].append(v)
        h = (1 - z) * cand + z * h
        Hs[:, t] = h
    c = {k: np.stack(v, 1) for k, v in c.items()}               # each (B, L, U)
    z2 = np.tanh(z1 @ n.W2.T + n.b2 + Hs @ n.Wm.T + n.bm)
    lg = z2 @ n.W3.T + n.b3
    lg = lg - lg.max(-1, keepdims=True)
    lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    nm = mask.sum()
    loss = -lp[mask, y[mask]].sum() / nm
    dl = np.exp(lp)
    dl[np.arange(B)[:, None], np.arange(L)[None, :], np.where(mask, y, 0)] -= 1.0      # outside the mask the factor below is zero
    dl = dl * mask[:, :, None] / nm                             # the error at the outputs
    c.update(z1=z1, Hs=Hs, z2=z2, dl=dl)
    return loss, c


def readout_error(n, c):
    """The error at the states, as it leaves the readout under the rule, and the gradient of `lm`."""
    da2 = (c["dl"] @ n.S3) * (1 - c["z2"] ** 2)                 # (B, L, 512)
    B, L, _ = da2.shape
    g_lm = da2.reshape(B * L, D).T @ c["Hs"].reshape(B * L, U)
    return da2 @ n.Sm, g_lm


def back_parts(n, c, e):
    """One step back for the errors e at the states of every position (B, L, U). Returns the errors at the states of the same
    row's previous position, in three parts (the term z*h(t-1); the cell's reading of itself in the recurrent map; the crossings
    from the cells it reads), each still indexed by the position it came from, and the errors at the gates."""
    r, z, cand, ghn, hp = c["r"], c["z"], c["c"], c["ghn"], c["hp"]
    da_n = e * (1 - z) * (1 - cand ** 2)
    da_r = da_n * ghn * r * (1 - r)
    da_z = e * (hp - cand) * z * (1 - z)
    dgh_n = da_n * r
    own_z = e * z
    own_w = da_r * n.Sh_diag[0] + da_z * n.Sh_diag[1] + dgh_n * n.Sh_diag[2]
    G = np.concatenate([da_r, da_z, dgh_n], -1)                 # (B, L, 768): the errors at the recurrent map's outputs
    cross = G @ n.Sh_cross
    return own_z, own_w, cross, (da_r, da_z, da_n, dgh_n)


def shift(g):
    """The error that leaves position t arrives at the states of position t - 1."""
    d = np.zeros_like(g)
    d[:, :-1] = g[:, 1:]
    return d


def relay_brief(n, c, d0, K):
    """The brief's update: every path, the cell's own included, is cut K positions back."""
    total, d = d0.copy(), d0
    for _ in range(K):
        own_z, own_w, cross, _ = back_parts(n, c, d)
        d = shift(own_z + own_w + cross)
        if not d.any():
            break
        total += d
    return total


def relay_crossings(n, c, d0, X):
    """The cell's own path (the term z*h and its reading of itself) kept for the whole item; at most X crossings between cells."""
    B, L, _ = d0.shape
    total, Dj = np.zeros_like(d0), d0
    for _ in range(X + 1):
        A = Dj.copy()
        for _ in range(L):                                      # the own path, to its end
            own_z, own_w, _, _ = back_parts(n, c, A)
            A_new = Dj + shift(own_z + own_w)
            if np.array_equal(A_new, A):
                break
            A = A_new
        total += A
        Dj = shift(back_parts(n, c, A)[2])
        if not Dj.any():
            break
    return total


def split_recurrent(n):
    """The recurrent map in two parts: a cell's reading of itself (3, U) and its readings of the other cells (3U, U)."""
    idx = np.arange(U)
    Wd = np.stack([n.Wh[g * U + idx, idx] for g in range(3)])
    Wc = n.Wh.copy()
    for g in range(3):
        Wc[g * U + idx, idx] = 0.0
    return Wd, Wc


def loss_levels(n, x, y, mask, X, Hstar):
    """The direct definition of form (b), as a loss: X + 1 copies of the states. Copy v takes a cell's own earlier state from
    itself and the other cells' states from copy v + 1; the last copy takes the other cells' states from Hstar, the states of
    the unchanged network, held constant. The readout reads copy 0. With the weights unchanged every copy equals Hstar; the
    derivative of this loss with respect to the cells' weights is the sum over the paths with at most X crossings."""
    B, L = x.shape
    z1 = n.table[x]
    gi = z1 @ n.Wi.T + n.bi
    Wd, Wc = split_recurrent(n)
    hs = [np.zeros((B, U)) for _ in range(X + 1)]
    H0 = np.empty((B, L, U))
    for t in range(L):
        star = Hstar[:, t - 1] if t else np.zeros((B, U))
        new = []
        for v in range(X + 1):
            other = hs[v + 1] if v < X else star
            gh = other @ Wc.T + np.concatenate([hs[v] * Wd[g] for g in range(3)], -1) + n.bh
            r = sig(gi[:, t, :U] + gh[:, :U])
            z = sig(gi[:, t, U:2 * U] + gh[:, U:2 * U])
            cand = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
            new.append((1 - z) * cand + z * hs[v])
        hs = new
        H0[:, t] = hs[0]
    z2 = np.tanh(z1 @ n.W2.T + n.b2 + H0 @ n.Wm.T + n.bm)
    lg = z2 @ n.W3.T + n.b3
    lg = lg - lg.max(-1, keepdims=True)
    lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    return -lp[mask, y[mask]].sum() / mask.sum()


def cell_grads(n, c, total):
    """The gradients of the recurrent map and of the input map (inside their masks) from the errors at the states."""
    da_r, da_z, da_n, dgh_n = back_parts(n, c, total)[3]
    B, L, _ = total.shape
    Gi = np.concatenate([da_r, da_z, da_n], -1).reshape(B * L, 3 * U)
    Gh = np.concatenate([da_r, da_z, dgh_n], -1).reshape(B * L, 3 * U)
    gWi = (Gi.T @ c["z1"].reshape(B * L, D)) * np.tile(n.min, (3, 1))
    gWh = (Gh.T @ c["hp"].reshape(B * L, U)) * np.tile(n.mrec, (3, 1))
    return gWh, gWi


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a * b).sum() / (na * nb)) if na > 0 and nb > 0 else float("nan")


# ---------------------------------------------------------------- the parts
def rule():
    print("=" * 100)


def part1(nets):
    rule()
    print("PART 1. The network of the brief (E50's stage 3, `C46-one`), rebuilt from the text of the source")
    n = nets["start"]
    code_digest = hashlib.sha256(json.dumps(n.owner).encode()).hexdigest()[:16]
    rec = json.loads(git_bytes(E50 + "e50_C46-one_S_73129.json"))
    print(f"1.1 digest of the code {code_digest}; the receipt's {rec['code']['digest']}; equal: {code_digest == rec['code']['digest']}")
    reads = n.min.sum(1)
    per_byte = [sorted(n.owner[v] for v in np.nonzero(n.min[i])[0]) == sorted(USED) for i in range(U)]
    print(f"1.2 values a cell reads: {int(reads.min())} to {int(reads.max())}; every cell reads exactly one value of every used byte: {all(per_byte)}")
    print(f"1.3 cells a cell reads, itself not counted: {int(n.mrec.sum(1).min()) - 1} to {int(n.mrec.sum(1).max()) - 1}; "
          f"largest stored recurrent weight outside the mask in the start file {nets['start'].raw_outside:.1e}, "
          f"in the sweep's final file {nets['S'].raw_outside:.1e}")
    w_in, w_rec = int(3 * n.min.sum()), int(3 * n.mrec.sum())
    w_cells = w_in + w_rec + n.bi.size + n.bh.size
    w_read = n.Wm.size + n.bm.size + n.W2.size + n.b2.size + n.W3.size + n.b3.size
    print(f"1.4 weights in use: input map {w_in:,}; recurrent map {w_rec:,}; the cells' biases {n.bi.size + n.bh.size:,}; "
          f"readout (lm, l2, l3 with biases) {w_read:,}; the readout's share {w_read / (w_cells + w_read):.3f}")
    print(f"1.5 every readout unit reads {n.Wm.shape[1]} cells and {n.W2.shape[1]} values of the byte's code; every cell is read by "
          f"{n.Wm.shape[0]} readout units; `l2` takes the byte's code to the readout without passing a cell")
    return rec


def part2(nets, rec, bt):
    rule()
    print("PART 2. This file's forward and backward passes against the record")
    h = hashlib.sha256()
    for items in bt[:256]:
        for t in batch_arrays(items):
            h.update(str(tuple(t.shape)).encode())
            h.update(t.tobytes())
    want = rec["curve"][1]["data_digest"]["sha256"]
    print(f"2.1 digest of the first 256 batches of the stream written here {h.hexdigest()[:16]}; the receipt's at update 256 {want[:16]}; equal: {h.hexdigest() == want}")
    x, y, mask = batch_arrays(bt[0])
    n = nets["start"]
    loss, c = forward(n, x, y, mask)
    print(f"2.2 loss of the first batch at the start {loss:.6f}; the receipt's {rec['first_batch_loss_start']['S']:.6f}")
    for key, label in (("start", "cosine_first_batch_start"), ("S", "cosine_first_batch_end")):
        n = nets[key]
        g = {}
        for r in "BS":
            n.set_rule(r)
            loss, c = forward(n, x, y, mask)
            d0, g_lm = readout_error(n, c)
            gWh, gWi = cell_grads(n, c, relay_brief(n, c, d0, 64))
            g[r] = {"rnn.weight_ih_l0": gWi, "lm.weight": g_lm, "rnn.weight_hh_l0": gWh}
        mine = {k: cos(g["B"][k], g["S"][k]) for k in g["B"]}
        print(f"2.3 cosine between the sweep's and the exact gradient on the first batch, {key} weights: "
              + "; ".join(f"{k} {mine[k]:.4f} (receipt {rec[label][k]:.4f})" for k in mine))
    n = nets["B"]
    n.set_rule("B")
    xs, ys, ms = x[:8], y[:8], mask[:8]
    loss, c = forward(n, xs, ys, ms)
    d0, _ = readout_error(n, c)
    gWh, gWi = cell_grads(n, c, relay_brief(n, c, d0, 64))
    rng = np.random.default_rng(0)
    worst = 0.0
    for name, g in (("Wh", gWh), ("Wi", gWi)):
        base = getattr(n, name)
        msk = np.tile(n.mrec if name == "Wh" else n.min, (3, 1))
        for _ in range(3):
            v = rng.standard_normal(base.shape) * msk
            eps = 1e-5
            setattr(n, name, base + eps * v); lp = forward(n, xs, ys, ms)[0]
            setattr(n, name, base - eps * v); lm_ = forward(n, xs, ys, ms)[0]
            setattr(n, name, base)
            num, ana = (lp - lm_) / (2 * eps), float((g * v).sum())
            worst = max(worst, abs(num - ana) / max(abs(num), abs(ana), 1e-30))
    print(f"2.4 the exact whole-item gradient of the two maps against central differences of the loss, six random directions: "
          f"largest relative difference {worst:.1e}")
    n.set_rule("B")
    loss, c = forward(n, x, y, mask)
    d0, _ = readout_error(n, c)
    whole = relay_brief(n, c, d0, 64)
    alt = relay_crossings(n, c, d0, 64)
    print(f"2.5 form (b) with no limit on the crossings against the whole-item error at the states: largest difference "
          f"{np.abs(whole - alt).max():.1e} (largest entry {np.abs(whole).max():.1e})")
    n.set_rule("B")                                              # form (b) against its direct definition, by central differences
    loss, c = forward(n, xs, ys, ms)
    d0, _ = readout_error(n, c)
    Hstar = c["Hs"].copy()
    gw = cell_grads(n, c, relay_brief(n, c, d0, 64))
    rng = np.random.default_rng(1)
    dirs = [(name, rng.standard_normal(getattr(n, name).shape) * np.tile(n.mrec if name == "Wh" else n.min, (3, 1)))
            for name in ("Wh", "Wi") for _ in range(2)]
    for X in (0, 1, 2, 3):
        gX = cell_grads(n, c, relay_crossings(n, c, d0, X))
        worst, apart = 0.0, 0.0
        for name, v in dirs:
            base = getattr(n, name)
            eps = 1e-5
            setattr(n, name, base + eps * v); lp = loss_levels(n, xs, ys, ms, X, Hstar)
            setattr(n, name, base - eps * v); lm_ = loss_levels(n, xs, ys, ms, X, Hstar)
            setattr(n, name, base)
            k = 0 if name == "Wh" else 1
            num, ana, full = (lp - lm_) / (2 * eps), float((gX[k] * v).sum()), float((gw[k] * v).sum())
            worst = max(worst, abs(num - ana) / max(abs(num), abs(ana), 1e-30))
            apart = max(apart, abs(full - ana) / max(abs(full), abs(ana), 1e-30))
        print(f"2.6 form (b) with at most {X} crossings against central differences of its direct definition (copies of the states), "
              f"four random directions: largest relative difference {worst:.1e}; from the whole-item gradient it differs by up to {apart:.1e}")
    for r in "BS":
        n.set_rule(r)
        loss, c = forward(n, x, y, mask)
        d0, _ = readout_error(n, c)
        cut = d0.copy()                                          # form (a) with K = 0: nothing leaves its position
        print(f"2.7 rule {r}: form (a) with K = 0 equals the readout's error alone: {np.array_equal(relay_brief(n, c, d0, 0), cut)}")


def part3(nets, bt):
    rule()
    print("PART 3. How far an error travels")
    n = nets["B"]
    x, y, mask = batch_arrays(bt[0])
    for r in "BS":
        n.set_rule(r)
        loss, c = forward(n, x, y, mask)
        d0, _ = readout_error(n, c)
        hit = (np.abs(d0) > 0).any(0)                            # (L, U)
        per_item = (np.abs(d0[mask]) > 0).mean()
        print(f"3.1 rule {r}: before any relay (K = 0) the readout's error is not zero at {int(hit.any(0).sum())} of {U} cells; "
              f"share of (item, answer position, cell) entries that are not zero {per_item:.3f}")
    reach = np.eye(U, dtype=bool)                               # reach[j, i]: an error at cell i has reached cell j
    A = n.mrec.T.copy()                                         # an error at cell i goes to the cells that cell i reads, and stays
    steps_all = np.zeros(U, int)
    cur = np.eye(U, dtype=bool)
    for s in range(1, 13):
        cur = (cur.astype(int) @ A.astype(int)) > 0             # row i: the cells reached from cell i after s steps
        done = cur.all(1)
        steps_all[(steps_all == 0) & done] = s
    print(f"3.2 by relay alone over the recurrent connections, an error that starts at one cell has reached every cell after "
          f"{int(steps_all.min())} to {int(steps_all.max())} steps (median {int(np.median(steps_all))})")
    for P, cells in (("one cell", [0]), ("a patch of 16 (4 by 4)", [r_ * SIDE + c_ for r_ in range(4) for c_ in range(4)]),
                     ("a patch of 64 (8 by 8)", [r_ * SIDE + c_ for r_ in range(8) for c_ in range(8)])):
        cur = np.zeros(U, bool)
        cur[cells] = True
        row = [int(cur.sum())]
        for s in range(8):
            cur = ((cur.astype(int) @ A.astype(int)) > 0)
            row.append(int(cur.sum()))
        print(f"3.3 if only {P} were read by the readout: cells that have received an error after 0, 1, 2, ... steps of relay: {row}")
    print("3.4 one step of relay is one position back: an error relayed s steps reaches a cell at the position s before the answer's;")
    print("    the second operand stands 1 and 2 positions before the first answer position, the first operand 4 + k and 5 + k (k letters)")


def part4(nets, bt):
    rule()
    print("PART 4. What a limit removes from the whole-item gradient of the cells' own weights (cosine with the whole-item")
    print("        gradient of the same rule; mean over batches of 64; recurrent map / input map), and in square brackets the")
    print("        share of the whole-item error at the states of the first operand's two positions (2 and 3) that the form")
    print("        delivers (its projection on the whole-item error there, as a share of that error's squared length)")
    Ks, Xs = (0, 1, 2, 4, 8, 16), (0, 1, 2, 4)
    stream_items = [it for b in bt[1:65] for it in b]            # batches 2 to 65 of the stream
    sets = {"training items (0 to 12 letters)": bt[1:17],
            "items with no letters, as R's": None, "items with 12 letters, as A's and T's": None}
    for k, name in ((0, "items with no letters, as R's"), (12, "items with 12 letters, as A's and T's")):
        pick = [it for it in stream_items if letters_of(it) == k][:256]
        sets[name] = [pick[i:i + BATCH] for i in range(0, len(pick) - BATCH + 1, BATCH)]
    for key, r, label in (("S", "S", "the sweep's trained network, under the sweep"),
                          ("B", "B", "the exact gradient's trained network, under the exact gradient"),
                          ("start", "S", "the start, under the sweep"), ("start", "B", "the start, under the exact gradient")):
        n = nets[key]
        n.set_rule(r)
        print(f"-- {label}")
        for sname, batches_ in sets.items():
            acc = {("a", K): [] for K in Ks}
            acc.update({("b", X): [] for X in Xs})
            norms = {k: [] for k in acc}
            for items in batches_:
                x, y, mask = batch_arrays(items)
                loss, c = forward(n, x, y, mask)
                d0, _ = readout_error(n, c)
                whole = relay_brief(n, c, d0, 64)
                gh0, gi0 = cell_grads(n, c, whole)
                w1 = whole[:, 2:4]
                for form, lim, fn in [("a", K, relay_brief) for K in Ks] + [("b", X, relay_crossings) for X in Xs]:
                    tot = fn(n, c, d0, lim)
                    gh, gi = cell_grads(n, c, tot)
                    acc[(form, lim)].append((cos(gh, gh0), cos(gi, gi0)))
                    norms[(form, lim)].append(float((tot[:, 2:4] * w1).sum() / (w1 * w1).sum()))
            one = lambda f, lab, v: (f"{lab}={v}: {np.mean([q[0] for q in acc[(f, v)]]):.2f}/{np.mean([q[1] for q in acc[(f, v)]]):.2f}"
                                     f" [{np.mean(norms[(f, v)]):.2f}]")
            fa = "  ".join(one("a", "K", K) for K in Ks)
            fb = "  ".join(one("b", "X", X) for X in Xs)
            print(f"   {sname} ({len(batches_)} batches)")
            print(f"      (a) every path cut K positions back:        {fa}")
            print(f"      (b) own path kept, at most X crossings:     {fb}")


def panel(task, pairs, seed, ks, n):                            # exp11_cell.panel
    rng = Random(seed + 3)
    sub = rng.sample(pairs, n)
    return {k: [render(task, a, b, k, rng) for a, b in sub] for k in ks}


def panels(seed, f=.5):                                         # exp42_cells.panels, without the length test
    name = lambda t: f"{t[0]}{t[1]}"
    th = {t: split(t, f, seed) for t in DEV}
    A = {name(t): panel(t, th[t][1], seed, (12,), 300)[12] for t in DEV}
    TA = {name(t): panel(t, th[t][0], seed + 1, (12,), 300)[12] for t in DEV}
    Rs = {}
    for t in DEV[:2]:
        pairs = [(a, b) for a, b in th[t][1] if (a + b >= 89 if t[0] == "add" else a - b < 0)]
        s = Random(f"e41|{seed}|{round(f * 100)}|{t[0]}").sample(sorted(pairs), 1000)
        Rs[name(t)] = [render(t, a, b, 0, Random(0)) for a, b in s]
    return A, Rs, TA


def answered(n, items, cross="intact", means=None):
    """How many of the items (all of one length) the network answers exactly: both digits and the end symbol. The scorer's answer
    is the greedy one, which is exact if and only if the three answer positions are right when the right bytes are fed.
    `cross` is what a cell receives from the other cells over the recurrent map: their states (intact); nothing (none); or, at
    each position, the mean over these items of what it received in the intact network (mean), which keeps the size of the
    input and removes everything about the item. A cell's reading of itself and the term z*h are kept in every case."""
    x, y, mask = batch_arrays(items)
    B, L = x.shape
    z1 = n.table[x]
    gi = z1 @ n.Wi.T + n.bi
    Wd, Wc = split_recurrent(n)
    h = np.zeros((B, U))
    Hs, got = np.empty((B, L, U)), np.empty((L, 3 * U))
    for t in range(L):
        cr = h @ Wc.T
        got[t] = cr.mean(0)
        if cross == "none":
            cr = np.zeros_like(cr)
        elif cross == "mean":
            cr = np.broadcast_to(means[t], cr.shape)
        gh = cr + np.concatenate([h * Wd[g] for g in range(3)], -1) + n.bh
        r = sig(gi[:, t, :U] + gh[:, :U])
        z = sig(gi[:, t, U:2 * U] + gh[:, U:2 * U])
        cand = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
        h = (1 - z) * cand + z * h
        Hs[:, t] = h
    pred = (np.tanh(z1 @ n.W2.T + n.b2 + Hs @ n.Wm.T + n.bm) @ n.W3.T + n.b3).argmax(-1)
    return int(((pred == y) | ~mask).all(1).sum()), got


def part5(nets):
    rule()
    print("PART 5. Do the record's trained networks use the connections between cells? (forward passes only, on the saved weights)")
    A, Rs, TA = panels(SEED)
    for key, label in (("B", "the exact gradient's"), ("S", "the sweep's")):
        n = nets[key]
        row = json.loads(git_bytes(E50 + f"e50_C46-one_{key}_73129.json"))["curve"][-1]
        mine = {"A_counts": {k: answered(n, it)[0] for k, it in A.items()}, "R_counts": {k: answered(n, it)[0] for k, it in Rs.items()},
                "taught_counts": {k: answered(n, it)[0] for k, it in TA.items()}}
        same = all(mine[k] == row[k] for k in mine)
        print(f"5.1 {label} final weights: exact answers computed here {mine}")
        print(f"    equal to the last scored row of the record's receipt (update {row['step']}) in all ten counts: {same}")
        for name, sets in (("R (2,000 items, no letters)", Rs), ("A (600 items of addition and subtraction, 12 letters)",
                                                                   {k: A[k] for k in ("add89", "sub89")}),
                           ("T (600 taught items of addition and subtraction, 12 letters)", {k: TA[k] for k in ("add89", "sub89")})):
            tot = sum(len(it) for it in sets.values())
            res = {}
            for mode in ("intact", "none", "mean"):
                cnt = 0
                for k, it in sets.items():
                    means = answered(n, it)[1] if mode == "mean" else None
                    cnt += answered(n, it, mode, means)[0]
                res[mode] = 100 * cnt / tot
            print(f"5.2 {label} network, {name}: intact {res['intact']:.2f}; a cell receives nothing from the other cells {res['none']:.2f}; "
                  f"it receives only the mean over the items {res['mean']:.2f}")


def to80(rec, key):
    """Updates at the first of three consecutive scored rows at or above 80, and [the first single row]."""
    rows = rec["curve"]
    at = "step" if "step" in rows[0] else "update"
    ok = [(r.get(key) or 0) >= 80 for r in rows]
    three = next((rows[i][at] for i in range(len(rows) - 2) if all(ok[i:i + 3])), None)
    single = next((rows[i][at] for i in range(len(rows)) if ok[i]), None)
    return three, single


def part6():
    rule()
    print("PART 6. Updates to 80 from the receipts (first of three consecutive scored rows [first single row]; None: not reached)")
    got = {}
    for folder, names in ((E50, [f"e50_{s}_{r}_73129" for s in ("C-dense", "C16-in", "C46-one") for r in "BS"]),
                          (M + "exp45/e42/", [f"e42_{s}_{r}_73129" for s in ("G", "F10") for r in "BS"])):
        for nm in names:
            rec = json.loads(git_bytes(folder + nm + ".json"))
            got[nm] = to80(rec, "R")
            a = to80(rec, "A")
            print(f"6.1 {nm}: on R {got[nm][0]} [{got[nm][1]}]; on A {a[0]} [{a[1]}]")
    s46, b46, sd, bd = (got[f"e50_{k}_73129"][0] for k in ("C46-one_S", "C46-one_B", "C-dense_S", "C-dense_B"))
    print(f"6.2 the sweep with 46 values a cell against the sweep with every value: {s46} / {sd} = {s46 / sd:.4f}; "
          f"the exact gradient likewise: {b46} / {bd} = {b46 / bd:.4f}; "
          f"the sweep against the exact gradient, both with 46 values a cell: {s46} / {b46} = {s46 / b46:.4f}")


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    rec0 = json.loads(git_bytes(E50 + "e50_C46-one_S_73129.json"))
    amp = rec0["code"]["amplitude"]
    files = {"start": "e50_C46-one_S_73129_start.pt", "S": "e50_C46-one_S_73129.pt", "B": "e50_C46-one_B_73129.pt"}
    nets = {}
    for k, fn in files.items():
        W = load_pt(git_bytes(E50 + fn))
        W = W if "rnn.weight_hh_l0" in W else next(v for v in W.values() if isinstance(v, dict) and "rnn.weight_hh_l0" in v)
        nets[k] = Net(W, amp, "B")
    tp = [(t, split(t, .5, SEED)[0]) for t in DEV]
    st = stream(tp, SEED, KMAX)
    bt = [[next(st) for _ in range(BATCH)] for _ in range(256)]
    rec = part1(nets)
    part2(nets, rec, bt)
    part3(nets, bt)
    part5(nets)
    part6()
    if not quick:
        part4(nets, bt)
    rule()


if __name__ == "__main__":
    main()
