"""Check for revision 3 of the brief of E61 (TASK_E61_THE_ERROR_RELAYED_OVER_A_LIMITED_NUMBER_OF_STEPS_20261118.md).

Prints every figure that E61_REVISION_3_NOTE_20261118.md (beside this file) quotes. Run from the repository's root:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/e61_revision_3_check.py

The Python standard library and numpy only; double precision throughout. Files of the default branch are read into memory with
`git show`, one weight file at a time; no code of the repository is executed (the stream and the panels are written again here from
the text of `exp11_cell.py`, `exp45.py` and `exp42_cells.py`, and the rewritten stream is tested against the record's batch digest).
The reviewer's memo is read at commit 8434d095, which the audit read, and from the working tree. No file is written.

Part 1  form (b) of the audit (a cell's own path kept for the whole item, at most X crossings between cells), written here as one
        backward pass over the positions that carries X + 1 errors indexed by the number of crossings made; tested on a small random
        gated network against central differences of the audit's "X + 1 copies" definition, under the exact rule and under the sign
        rule, and against the whole-item gradient
Part 2  the record's network read from its weight files: the weights in use and the readout's share; the readout's error at K = 0;
        the share of the whole-item credit at the first operand that each form delivers (the audit's section 3.4), computed with
        the method of part 1
Part 3  updates to 80 and the ratios of E50's receipts; the limits of 1.25 times the record
Part 4  the distances of the first operand from the first answer position in the panels R, A and T
Part 5  the quotations of the reviewer's memo in the audit's sections 3.6 and 5.2, and E49's readout `out8`
"""
import hashlib
import io
import json
import pickle
import subprocess
import sys
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from random import Random

import numpy as np

D = "origin/codex/minifly-consolidated-20261023"
M = "MOD97_LOCAL_RULE_20261106"
E50 = f"{M}/exp45/e50"
MEMO = Path("PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md")
SEED = 73129
T0 = time.time()


def show(path):
    r = subprocess.run(["git", "show", f"{D}:{path}"], capture_output=True)
    if r.returncode != 0:
        sys.exit(f"git show failed for {path}: {r.stderr.decode()[:200]}")
    return r.stdout


def sig(a):
    return 1.0 / (1.0 + np.exp(-a))


def line():
    print("=" * 110)


# =====================================================================================================================
# The relay, written once for both networks: a backward pass over the positions carrying several errors ("levels")
# =====================================================================================================================
def step_back(e, c, t, Sd, Sc, U):
    """The error e (..., B, U) at the states of position t sent to the states of position t - 1, in two parts: the cell's own path
    (the term z * h(t-1) and the cell's reading of itself, Sd (3, U)) and the crossings (Sc (3U, U), the recurrent map's backward
    value with its diagonal set to zero). Also the errors at the gates' pre-activations (for the weight gradients)."""
    r, z, cand, ghn, hp = c["r"][:, t], c["z"][:, t], c["c"][:, t], c["ghn"][:, t], c["hp"][:, t]
    da_n = e * (1 - z) * (1 - cand ** 2)
    da_r = da_n * ghn * r * (1 - r)
    da_z = e * (hp - cand) * z * (1 - z)
    dgh_n = da_n * r
    own = e * z + da_r * Sd[0] + da_z * Sd[1] + dgh_n * Sd[2]
    cross = np.concatenate([da_r, da_z, dgh_n], -1) @ Sc
    return own, cross


def relay(d0, c, Sd, Sc, U, levels, own_up, cross_up):
    """One backward pass over the positions. levels: the number of errors carried; an error moves `own_up` levels up when it takes the
    own path and `cross_up` levels up when it crosses to another cell; an error that would pass the last level is dropped.
      whole item: levels 1, own_up 0, cross_up 0
      form (b), at most X crossings: levels X + 1, own_up 0, cross_up 1
      form (a), at most K positions: levels K + 1, own_up 1, cross_up 1
    Returns the error at the states of every position (B, L, U), the sum over the levels."""
    B, L, _ = d0.shape
    e = np.zeros((levels, B, U))
    tot = np.zeros_like(d0)
    for t in range(L - 1, -1, -1):
        e[0] += d0[:, t]
        tot[:, t] = e.sum(0)
        if t == 0:
            break
        own, cross = step_back(e, c, t, Sd, Sc, U)
        new = np.zeros_like(e)
        new[own_up:] += own[:levels - own_up]
        new[cross_up:] += cross[:levels - cross_up]
        e = new
    return tot


# =====================================================================================================================
# Part 1. Form (b) on a small random gated network
# =====================================================================================================================
def part1():
    line()
    print("PART 1. Form (b) written as one backward pass with X + 1 levels, against central differences of the audit's 'X + 1 copies'")
    print("        definition, on a small random gated network in double precision (the form of e50_sparse_code.forward)")
    rng = np.random.default_rng(61003)
    U, DIN, H, V, B, L = 9, 8, 6, 5, 4, 14
    ks = np.array([0, 2, 3, 1])                                     # letters of each item
    ans = np.stack([8 + ks, 9 + ks, 10 + ks], 1)                     # three answer positions an item, different between items
    Mh = np.eye(U, dtype=bool)
    for i in range(U):
        Mh[i, rng.choice([j for j in range(U) if j != i], 3, replace=False)] = True
    Mh3 = np.tile(Mh, (3, 1)).astype(float)
    Mi3 = np.tile(rng.random((U, DIN)) < 0.5, (3, 1)).astype(float)
    P0 = dict(Wi=rng.normal(0, .6, (3 * U, DIN)) * Mi3, bi=rng.normal(0, .2, 3 * U), Wh=rng.normal(0, .7, (3 * U, U)) * Mh3,
              bh=rng.normal(0, .2, 3 * U), W2=rng.normal(0, .4, (H, DIN)), b2=rng.normal(0, .1, H), Wm=rng.normal(0, .5, (H, U)),
              bm=rng.normal(0, .1, H), W3=rng.normal(0, .6, (V, H)), b3=rng.normal(0, .1, V))
    z1 = rng.normal(0, 1, (6, DIN))[rng.integers(0, 6, (B, L))]      # a fixed input code
    tgt = rng.integers(0, V, (B, 3))
    mask = np.zeros((B, L), bool)
    for b in range(B):
        mask[b, ans[b]] = True
    y = np.zeros((B, L), int)
    for b in range(B):
        y[b, ans[b]] = tgt[b]
    nm = mask.sum()
    idx = np.arange(U)
    diag = lambda W: np.stack([W[g * U + idx, idx] for g in range(3)])          # (3, U): a cell's reading of itself

    def back_values(P, rule, own_exact=False):
        """The maps over which the error is sent back: the weights (B); sign(W) times one scale (S): for the recurrent map the mean
        magnitude inside its mask, diagonal included (e42_cells.Layer.sign_scale), for lm and l3 the mean magnitude
        (exp3_state.SignLinear). own_exact: a variant of S in which a cell's reading of itself is sent back by its weight."""
        if rule == "B":
            Sh, Sm, S3 = P["Wh"], P["Wm"], P["W3"]
        else:
            Sh = np.sign(P["Wh"]) * (np.abs(P["Wh"]) * Mh3).sum() / Mh3.sum()
            Sm = np.sign(P["Wm"]) * np.abs(P["Wm"]).mean()
            S3 = np.sign(P["W3"]) * np.abs(P["W3"]).mean()
        Sd = diag(P["Wh"]) if own_exact else diag(Sh)
        Sc = Sh.copy()
        for g in range(3):
            Sc[g * U + idx, idx] = 0.0
        return Sd, Sc, Sm, S3

    def forward(P):
        gi = z1 @ (P["Wi"] * Mi3).T + P["bi"]
        Wh = P["Wh"] * Mh3
        h = np.zeros((B, U))
        c = {k: [] for k in ("r", "z", "c", "ghn", "hp")}
        Hs = np.empty((B, L, U))
        for t in range(L):
            gh = h @ Wh.T + P["bh"]
            r = sig(gi[:, t, :U] + gh[:, :U]); z = sig(gi[:, t, U:2 * U] + gh[:, U:2 * U])
            cand = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
            for k, v in zip(("r", "z", "c", "ghn", "hp"), (r, z, cand, gh[:, 2 * U:], h)):
                c[k].append(v)
            h = (1 - z) * cand + z * h
            Hs[:, t] = h
        c = {k: np.stack(v, 1) for k, v in c.items()}
        z2 = np.tanh(z1 @ P["W2"].T + P["b2"] + Hs @ P["Wm"].T + P["bm"])
        lg = z2 @ P["W3"].T + P["b3"]
        c.update(Hs=Hs, z2=z2, lg=lg)
        return c

    def readout_error(P, c, rule):
        lg = c["lg"] - c["lg"].max(-1, keepdims=True)
        p = np.exp(lg); p /= p.sum(-1, keepdims=True)
        p[np.arange(B)[:, None], np.arange(L)[None, :], y] -= 1.0
        dl = p * mask[:, :, None] / nm
        _, _, Sm, S3 = back_values(P, rule)
        return ((dl @ S3) * (1 - c["z2"] ** 2)) @ Sm

    def cell_grads(P, c, tot, rule):
        """The gradients of Wi, bi, Wh, bh from the errors at the states of every position."""
        r, z, cand, ghn, hp = c["r"], c["z"], c["c"], c["ghn"], c["hp"]
        da_n = tot * (1 - z) * (1 - cand ** 2); da_r = da_n * ghn * r * (1 - r); da_z = tot * (hp - cand) * z * (1 - z)
        Gi = np.concatenate([da_r, da_z, da_n], -1).reshape(-1, 3 * U)
        Gh = np.concatenate([da_r, da_z, da_n * r], -1).reshape(-1, 3 * U)
        return dict(Wi=(Gi.T @ z1.reshape(-1, DIN)) * Mi3, bi=Gi.sum(0), Wh=(Gh.T @ hp.reshape(-1, U)) * Mh3, bh=Gh.sum(0))

    def grads(P, rule, levels, own_up, cross_up, own_exact=False):
        c = forward(P)
        Sd, Sc, _, _ = back_values(P, rule, own_exact)
        tot = relay(readout_error(P, c, rule), c, Sd, Sc, U, levels, own_up, cross_up)
        return cell_grads(P, c, tot, rule), tot

    def copies_loss(P, P_ref, X, rule):
        """The audit's direct definition of form (b) (section 3.7): X + 1 copies of the states; copy v takes a cell's own earlier state
        from itself and the other cells' earlier states from copy v + 1; the last copy takes them from the reference states, held
        constant; the readout reads copy 0. For the sign rule, every map over which the rule sends an error back by another value than
        its weight is written as W h_ref + S (h - h_ref), with h_ref the reference states and S the rule's backward value at the
        reference weights, both held constant: at the reference weights the values are unchanged and the ordinary derivative is the
        rule's. Under the exact rule S is the reference weight, which leaves the first derivative unchanged."""
        c0 = forward(P_ref)
        Sd, Sc, Sm, S3 = back_values(P_ref, rule)
        H0, z20 = c0["Hs"], c0["z2"]
        gi = z1 @ (P["Wi"] * Mi3).T + P["bi"]
        Wd = diag(P["Wh"])
        Wc = P["Wh"] * Mh3
        for g in range(3):
            Wc[g * U + idx, idx] = 0.0
        hs = [np.zeros((B, U)) for _ in range(X + 1)]
        Hc = np.empty((B, L, U))
        for t in range(L):
            ref = H0[:, t - 1] if t else np.zeros((B, U))
            new = []
            for v in range(X + 1):
                other = hs[v + 1] if v < X else ref
                gh = (ref @ Wc.T + (other - ref) @ Sc.T
                      + np.concatenate([ref * Wd[g] + (hs[v] - ref) * Sd[g] for g in range(3)], -1) + P["bh"])
                r = sig(gi[:, t, :U] + gh[:, :U]); z = sig(gi[:, t, U:2 * U] + gh[:, U:2 * U])
                cand = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
                new.append((1 - z) * cand + z * hs[v])
            hs = new
            Hc[:, t] = hs[0]
        z2 = np.tanh(z1 @ P["W2"].T + P["b2"] + H0 @ P["Wm"].T + (Hc - H0) @ Sm.T + P["bm"])
        lg = z20 @ P["W3"].T + (z2 - z20) @ S3.T + P["b3"]
        lg = lg - lg.max(-1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        return -lp[mask, y[mask]].sum() / nm

    def rel(a, b):
        return max(float(np.abs(a[k] - b[k]).max() / np.abs(b[k]).max()) for k in b)

    # the hand-written whole-item gradient against central differences of the plain loss (exact rule)
    dirs = []
    for _ in range(6):
        dirs.append({"Wi": rng.normal(0, 1, P0["Wi"].shape) * Mi3, "bi": rng.normal(0, 1, 3 * U),
                     "Wh": rng.normal(0, 1, P0["Wh"].shape) * Mh3, "bh": rng.normal(0, 1, 3 * U)})

    def fd(fun, v, eps=1e-6):
        Pp = {k: P0[k] + eps * v.get(k, 0) for k in P0}; Pm = {k: P0[k] - eps * v.get(k, 0) for k in P0}
        return (fun(Pp) - fun(Pm)) / (2 * eps)

    def plain_loss(P):
        lg = forward(P)["lg"]; lg = lg - lg.max(-1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        return -lp[mask, y[mask]].sum() / nm

    gw, _ = grads(P0, "B", 1, 0, 0)
    worst = max(abs(fd(plain_loss, v) - sum(float((gw[k] * v[k]).sum()) for k in v)) / abs(sum(float((gw[k] * v[k]).sum()) for k in v))
                for v in dirs)
    print(f"1.1 test network: {U} cells, each reading 3 others and itself; {L} positions; {B} items with answer positions {ans.tolist()}")
    print(f"1.2 the whole-item gradient written here (exact rule) against central differences of the loss, 6 random directions: "
          f"largest relative difference {worst:.1e}")
    results = {}
    for rule in ("B", "S"):
        full, _ = grads(P0, rule, 1, 0, 0)
        print(f"-- rule {rule}: {'the exact gradient' if rule == 'B' else 'the sign rule: sign(W) times one scale at every map, the gates and z * h(t-1) exact'}")
        for X in (0, 1, 2, 4, L, L + 5):
            gX, _ = grads(P0, rule, X + 1, 0, 1)
            w = 0.0
            for v in dirs:
                num = fd(lambda P: copies_loss(P, P0, X, rule), v)
                ana = sum(float((gX[k] * v[k]).sum()) for k in v)
                w = max(w, abs(num - ana) / max(abs(num), abs(ana)))
            results[(rule, X)] = w
            print(f"   X = {X:2d}: one pass with X + 1 levels against central differences of the copies definition: largest relative "
                  f"difference {w:.1e}; against the whole-item gradient {rel(gX, full):.1e}")
        g0, _ = grads(P0, rule, 1, 0, 1)                    # X = 0: one level, every crossing dropped
        gK0, _ = grads(P0, rule, 1, 1, 1)                   # form (a) with K = 0
        print(f"   X = 0 against form (a) with K = 0 (the state a constant at every position): {rel(g0, gK0):.2f}")
        g1, _ = grads(P0, rule, 2, 0, 1); g2, _ = grads(P0, rule, 3, 0, 1)
        print(f"   an error of one level is visible: X = 1 against X = 2 {rel(g1, g2):.2f}; a crossing kept on its level (form (b) with "
              f"cross_up = 0, which is the whole item) against X = 2 {rel(grads(P0, rule, 3, 0, 0)[0], g2):.2f}")
        if rule == "S":
            gd, _ = grads(P0, "S", 3, 0, 1, own_exact=True)
            gdL, _ = grads(P0, "S", L + 1, 0, 1, own_exact=True)
            print(f"   the diagonal under the sign rule: X = 2 with a cell's reading of itself sent back by its weight instead of sign(W) times "
                  f"the map's scale differs by {rel(gd, g2):.2f}; with that variant X = {L} differs from the sign rule's whole-item update "
                  f"by {rel(gdL, full):.2f}")
    gS, _ = grads(P0, "S", 1, 0, 0)
    print(f"1.3 the two rules differ (whole item) by {rel(gS, grads(P0, 'B', 1, 0, 0)[0]):.2f}")
    ok = max(results.values()) < 1e-6
    print(f"1.4 THE TEST {'PASSED' if ok else 'FAILED'}: largest difference from the copies definition {max(results.values()):.1e} "
          f"over X = 0, 1, 2, 4, {L}, {L + 5} and both rules")


# =====================================================================================================================
# The record's network, read from its weight files
# =====================================================================================================================
_DT = {"FloatStorage": np.float32, "LongStorage": np.int64, "BoolStorage": np.bool_}


class _St:
    def __init__(self, name):
        self.name = name


def load_pt(path):
    """A torch.save file read without torch: only OrderedDict, tensors and the storages above are admitted."""
    zf = zipfile.ZipFile(io.BytesIO(show(path)))
    pk = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
    pre = pk[:-len("/data.pkl")]

    def tensor(storage, offset, size, stride, *rest):
        size = tuple(size)
        want = tuple(int(np.prod(size[i + 1:])) for i in range(len(size)))
        assert tuple(stride) == want, (size, stride)
        n = int(np.prod(size)) if size else 1
        return storage[offset:offset + n].reshape(size).copy()

    class Unp(pickle.Unpickler):
        def find_class(self, mod, name):
            if (mod, name) == ("collections", "OrderedDict"):
                return OrderedDict
            if (mod, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return tensor
            if mod == "torch" and name in _DT:
                return _St(name)
            raise pickle.UnpicklingError(f"refused {mod}.{name}")

        def persistent_load(self, pid):
            return np.frombuffer(zf.read(f"{pre}/data/{pid[2]}"), dtype=_DT[pid[1].name])

    return Unp(io.BytesIO(zf.read(pk))).load()


class Net:
    U = 256

    def __init__(self, path):
        W = load_pt(path)
        f = lambda k: np.asarray(W[k], np.float64)
        U = self.U
        self.table = f("code_table")
        self.mask = np.asarray(W["mask"], bool)
        self.mask_in = np.asarray(W["mask_in"], bool)
        self.mask_out = np.asarray(W["mask_out"], bool)
        self.Wi = f("rnn.weight_ih_l0") * np.tile(self.mask_in, (3, 1))
        self.Wh = f("rnn.weight_hh_l0") * np.tile(self.mask, (3, 1))
        self.bi, self.bh = f("rnn.bias_ih_l0"), f("rnn.bias_hh_l0")
        self.Wm, self.bm, self.W2, self.b2, self.W3, self.b3 = (f(k) for k in ("lm.weight", "lm.bias", "l2.weight", "l2.bias",
                                                                               "l3.weight", "l3.bias"))
        self.idx = np.arange(U)

    def back_values(self, rule):
        U, idx, m3 = self.U, self.idx, np.tile(self.mask, (3, 1))
        if rule == "B":
            Sh, Sm, S3 = self.Wh, self.Wm, self.W3
        else:
            Sh = np.sign(self.Wh) * (np.abs(self.Wh) * m3).sum() / m3.sum()
            Sm = np.sign(self.Wm) * np.abs(self.Wm).mean()
            S3 = np.sign(self.W3) * np.abs(self.W3).mean()
        Sd = np.stack([Sh[g * U + idx, idx] for g in range(3)])
        Sc = Sh.copy()
        for g in range(3):
            Sc[g * U + idx, idx] = 0.0
        return Sd, Sc, Sm, S3

    def forward(self, x, y, mask, rule):
        U = self.U
        B, L = x.shape
        z1 = self.table[x]
        gi = z1 @ self.Wi.T + self.bi
        h = np.zeros((B, U))
        c = {k: np.empty((B, L, U)) for k in ("r", "z", "c", "ghn", "hp")}
        Hs = np.empty((B, L, U))
        for t in range(L):
            gh = h @ self.Wh.T + self.bh
            r = sig(gi[:, t, :U] + gh[:, :U]); z = sig(gi[:, t, U:2 * U] + gh[:, U:2 * U])
            cand = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
            c["r"][:, t], c["z"][:, t], c["c"][:, t], c["ghn"][:, t], c["hp"][:, t] = r, z, cand, gh[:, 2 * U:], h
            h = (1 - z) * cand + z * h
            Hs[:, t] = h
        z2 = np.tanh(z1 @ self.W2.T + self.b2 + Hs @ self.Wm.T + self.bm)
        lg = z2 @ self.W3.T + self.b3
        lg = lg - lg.max(-1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        nm = mask.sum()
        loss = -lp[mask, y[mask]].sum() / nm
        p = np.exp(lp)
        p[np.arange(B)[:, None], np.arange(L)[None, :], np.where(mask, y, 0)] -= 1.0
        dl = p * mask[:, :, None] / nm
        _, _, Sm, S3 = self.back_values(rule)
        d0 = ((dl @ S3) * (1 - z2 ** 2)) @ Sm
        return loss, c, d0


# ---------------------------------------------------------------- the items and the stream, from exp11_cell.py and exp45.py
EOS, BOS, BATCH = 256, 257, 64
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def apply_op(op, m, a, b):
    return {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]


def render(task, a, b, k, rng):
    op, m, code = task
    d = [rng.choice(DISTRACT) for _ in range(k)]
    prompt = list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")]
    return prompt, list(f"{apply_op(op, m, a, b):02d}".encode()) + [EOS]


def split(task, frac, seed):
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    rng = Random(seed * 1000 + round(frac * 100) + m * 7 + code[0])
    rng.shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def stream(seed, kmax=12):
    rng = Random(seed + 77)
    flat = [(t, pr) for t in DEV for pr in split(t, .5, seed)[0]]
    while True:
        order = list(range(len(flat)))
        rng.shuffle(order)
        for i in order:
            t, (a, b) = flat[i]
            yield render(t, a, b, rng.randint(0, kmax), rng)


def arrays(items):
    rows = [p + t for p, t in items]
    L = max(len(r) for r in rows) + 1
    data = np.full((len(rows), L), BOS, np.int64)
    for i, r in enumerate(rows):
        data[i, 1:len(r) + 1] = r
    x, y = np.ascontiguousarray(data[:, :-1]), np.ascontiguousarray(data[:, 1:])
    mask = np.zeros(y.shape, bool)
    for i, (p, t) in enumerate(items):
        mask[i, len(p):len(p) + len(t)] = True
    return x, y, mask


def part2():
    line()
    print("PART 2. The record's network (E50's stage 3, C46-one, seed 73129), read from its weight files")
    st = stream(SEED)
    bt = [[next(st) for _ in range(BATCH)] for _ in range(256)]
    h = hashlib.sha256()
    for items in bt:
        for a in arrays(items):
            h.update(str(tuple(a.shape)).encode()); h.update(a.tobytes())
    rec = json.loads(show(f"{E50}/e50_C46-one_S_73129.json"))
    want = next(r for r in rec["curve"] if r["step"] == 256)["data_digest"]["sha256"]
    print(f"2.1 the stream written here gives the record's batch digest at update 256: {h.hexdigest() == want} ({h.hexdigest()[:16]})")
    files = (("start", "e50_C46-one_S_73129_start.pt", ("S", "B")), ("the sweep's final network", "e50_C46-one_S_73129.pt", ("S",)),
             ("the exact gradient's final network", "e50_C46-one_B_73129.pt", ("B",)))
    stream_items = [it for b in bt[1:65] for it in b]
    by_k = {k: [it for it in stream_items if len(it[0]) - 7 == k][:256] for k in (0, 12)}
    sets = {"no letters (4 batches, as R's)": [by_k[0][i:i + 64] for i in range(0, 256, 64)],
            "12 letters (4 batches, as A's and T's)": [by_k[12][i:i + 64] for i in range(0, 256, 64)]}
    Xs, Ks = (0, 1, 2, 4), (0, 8, 16)
    for label, fn, rules in files:
        n = Net(f"{E50}/{fn}")
        if label == "start":
            U = n.U
            w_in, w_rec = 3 * int(n.mask_in.sum()), 3 * int(n.mask.sum())
            w_bias = n.bi.size + n.bh.size
            w_read = n.Wm.size + n.bm.size + n.W2.size + n.b2.size + n.W3.size + n.b3.size
            tot = w_in + w_rec + w_bias + w_read
            print(f"2.2 weights in use: input map {w_in:,} ({int(n.mask_in.sum(1).min())} to {int(n.mask_in.sum(1).max())} values a cell); "
                  f"recurrent map {w_rec:,} ({int(n.mask.sum(1).min()) - 1} to {int(n.mask.sum(1).max()) - 1} other cells a cell, and itself); "
                  f"the cells' biases {w_bias:,}; readout lm {n.Wm.size + n.bm.size:,}, l2 {n.W2.size + n.b2.size:,}, l3 {n.W3.size + n.b3.size:,}, "
                  f"together {w_read:,}; in all {tot:,}; the readout's share {w_read / tot:.4f}")
            nb = n.Wm.size + n.W2.size + n.W3.size
            print(f"    without biases (the receipt's count `weights_taking_part`, {rec['multiply_adds_per_position']:,}): "
                  f"{w_in + w_rec + nb:,}, readout share {nb / (w_in + w_rec + nb):.4f}; of the readout, l2 ({n.W2.size + n.b2.size:,}) reads "
                  f"the byte's code and no cell; lm and l3 alone are {(w_read - n.W2.size - n.b2.size) / tot:.4f} of the weights in use; "
                  f"mask_out full: {bool(n.mask_out.all())}")
            x, y, mask = arrays(bt[0])
            for rule in ("B", "S"):
                loss, c, d0 = n.forward(x, y, mask, rule)
                hit = int((np.abs(d0).sum((0, 1)) > 0).sum())
                share = float((np.abs(d0[mask]) > 0).mean())
                print(f"2.3 rule {rule}, start, first batch: loss {loss:.6f} (receipt {rec['first_batch_loss_start'][rule]:.6f}); the readout's "
                      f"error (K = 0, X = 0, before any relay) is not zero at {hit} of {U} cells; share of (item, answer position, cell) "
                      f"entries not zero {share:.3f}")
            # the share at the start on the brief's 16 cosine batches (batches 2 to 17 of the stream), for every form
            print("2.4 at the start, on the 16 batches of the brief's cosines (batches 2 to 17 of the stream, training items with 0 to 12")
            print("    letters): the share of the whole-item credit at the first operand (states of positions 2 and 3) that each form delivers;")
            print("    mean over the 16 batches of the ratio computed on each batch; in brackets the ratio pooled over the items of the 16")
            print("    batches with no letters and with 12 letters")
            for rule in ("S", "B"):
                Sd, Sc, _, _ = n.back_values(rule)
                acc = {}
                for items in bt[1:17]:
                    x, y, mask = arrays(items)
                    _, c, d0 = n.forward(x, y, mask, rule)
                    whole = relay(d0, c, Sd, Sc, U, 1, 0, 0)
                    w1 = whole[:, 2:4]
                    kk = np.array([len(p) - 7 for p, _ in items])
                    forms = [("X", X, relay(d0, c, Sd, Sc, U, X + 1, 0, 1)) for X in Xs] + \
                            [("K", K, relay(d0, c, Sd, Sc, U, K + 1, 1, 1)) for K in Ks]
                    for f_, v, tot_ in forms:
                        a = acc.setdefault((f_, v), [[], [0, 0], [0, 0]])
                        a[0].append(float((tot_[:, 2:4] * w1).sum() / (w1 * w1).sum()))
                        for j, k in ((1, 0), (2, 12)):
                            s = kk == k
                            a[j][0] += float((tot_[s, 2:4] * w1[s]).sum()); a[j][1] += float((w1[s] * w1[s]).sum())
                print(f"    rule {rule}: " + "  ".join(f"{f_}={v}: {np.mean(a[0]):.3f} [{a[1][0] / a[1][1]:.2f}; {a[2][0] / a[2][1]:.2f}]"
                                                     for (f_, v), a in acc.items()))
        for rule in rules:
            Sd, Sc, _, _ = n.back_values(rule)
            out = []
            for sname, batches in sets.items():
                sh = {f: [] for f in [("X", X) for X in Xs] + [("K", K) for K in Ks]}
                for items in batches:
                    x, y, mask = arrays(items)
                    _, c, d0 = n.forward(x, y, mask, rule)
                    w1 = relay(d0, c, Sd, Sc, n.U, 1, 0, 0)[:, 2:4]
                    for (f_, v) in sh:
                        t_ = relay(d0, c, Sd, Sc, n.U, v + 1, 0 if f_ == "X" else 1, 1)
                        sh[(f_, v)].append(float((t_[:, 2:4] * w1).sum() / (w1 * w1).sum()))
                out.append(f"{sname}: " + ", ".join(f"{f_}={v} {np.mean(a):.2f}" for (f_, v), a in sh.items()))
            print(f"2.5 {label}, rule {rule}, share at the first operand (the audit's table of section 3.4; items of batches 2 "
                  f"to 65 of the stream): " + "; ".join(out))
        del n


# =====================================================================================================================
# Part 3. The receipts
# =====================================================================================================================
def reached(curve, key, th=80):
    v = [r[key] for r in curve]
    st = [r["step"] for r in curve]
    three = next((st[i] for i in range(len(v) - 2) if min(v[i:i + 3]) >= th), None)
    one = next((st[i] for i in range(len(v)) if v[i] >= th), None)
    return three, one


def part3():
    line()
    print("PART 3. Updates to 80 (first of three consecutive scored rows at or above 80 [first single row]) from E50's receipts")
    got = {}
    for s in ("C-dense", "C16-in", "C46-one"):
        for r in "BS":
            d = json.loads(show(f"{E50}/e50_{s}_{r}_73129.json"))
            assert d["setting"] == s and d["rule"] == r
            got[(s, r, "R")], got[(s, r, "A")] = reached(d["curve"], "R"), reached(d["curve"], "A")
            print(f"3.1 {s} {r}: R {got[(s, r, 'R')]}; A {got[(s, r, 'A')]}")
    q = lambda s, r, k: got[(s, r, k)][0]
    print(f"3.2 the sweep, 46 values a cell against every value: {q('C46-one', 'S', 'R')} / {q('C-dense', 'S', 'R')} = "
          f"{q('C46-one', 'S', 'R') / q('C-dense', 'S', 'R'):.4f}; the exact gradient likewise: {q('C46-one', 'B', 'R')} / {q('C-dense', 'B', 'R')} = "
          f"{q('C46-one', 'B', 'R') / q('C-dense', 'B', 'R'):.4f}; the sweep against the exact gradient, both with 46 values a cell: "
          f"{q('C46-one', 'S', 'R')} / {q('C46-one', 'B', 'R')} = {q('C46-one', 'S', 'R') / q('C46-one', 'B', 'R'):.4f}")
    print(f"3.3 1.25 times the record (C46-one): on R, exact gradient {1.25 * q('C46-one', 'B', 'R'):,.0f}, sweep {1.25 * q('C46-one', 'S', 'R'):,.0f}; "
          f"on A, exact gradient {1.25 * q('C46-one', 'B', 'A'):,.0f}, sweep {1.25 * q('C46-one', 'S', 'A'):,.0f}")
    for r in "BS":
        d = json.loads(show(f"{M}/exp45/e42/e42_G_{r}_73129.json"))
        print(f"3.4 decision 31's reference, E42's G under rule {r}: R {reached(d['curve'], 'R')}; A {reached(d['curve'], 'A')}")


# =====================================================================================================================
# Part 4. The distances in the panels
# =====================================================================================================================
def part4():
    line()
    print("PART 4. The distances of the first operand from the first answer position in the panels")
    src = show(f"{M}/exp42_cells.py").decode()
    for s in ('    A = {E.NAME(*t): E.panel(t, th[t][1], seed, ks=(12,), n=300)[12] for t in E.DEV}',
              '    TA = {E.NAME(*t): E.panel(t, th[t][0], seed + 1, ks=(12,), n=300)[12] for t in E.DEV}',
              '        Rs[E.NAME(*t)] = [E.render(t, a, b, 0, Random(0)) for a, b in s]'):
        print(f"4.1 {s in src!s:5} exp42_cells.panels: {s.strip()}")
    for name, k in (("R", 0), ("A and T", 12)):
        p, t = render(DEV[0], 12, 34, k, Random(0))
        x, y, mask = arrays([(p, t)])
        first_answer = int(np.nonzero(mask[0])[0][0])
        pos = [i for i in range(x.shape[1]) if int(x[0, i]) in set(b"0123456789")][:2]          # the two digits of the first operand
        print(f"4.2 panel {name} ({k} letters): input positions 0 to {x.shape[1] - 1}; first operand at {pos}; first answer position "
              f"{first_answer}; distances {first_answer - pos[0]} and {first_answer - pos[1]}")


# =====================================================================================================================
# Part 5. The memo's quotations and E49's readout
# =====================================================================================================================
def part5():
    line()
    print("PART 5. The reviewer's memo at 8434d095 (the commit the audit read), sections quoted in the audit's sections 3.6 and 5.2")
    r = subprocess.run(["git", "show", f"8434d095:{MEMO}"], capture_output=True)
    if r.returncode != 0:
        sys.exit("git show failed for the memo at 8434d095")
    memo = r.stdout.decode()

    def section(head):
        i = memo.index(head)
        j = memo.find("\n### ", i + len(head))
        k = memo.find("\n## ", i + len(head))
        return memo[i:min(x for x in (j, k) if x > 0)]
    checks = [
        ("### 28.4", "With the 540 cells of the second and third rings held at their mean, or given the states of another item, no score changes by more than 0.7 points"),
        ("### 28.4", "1,024 cells of which 256 read the byte and are read by the readout"),
        ("### 28.4", "The first ring does take part."),
        ("### 28.5", "| `out8` (512 / 8; 0.86) | 74 / 71; not reached, rising | 10.6 / 7.8 |"),
        ("### 28.6", "**E42's networks lose the family as the dense one does**: `F10` and `P8` are at or below 1% on reduction pairs by update 64. Limiting the connections between cells does not separate what is learned."),
        ("### 29.3", "every item drives every cell, including an item made of bytes the network has never met"),
        ("### 29.3", "the limit removes connections, it does not make any cell idle"),
        ("### 29.3", "With a code in which a value stands for one byte, as in E50, a new byte would drive only connections that no earlier byte drives."),
        ("### 33.27", "One visit to another family (256 updates) takes what is answered from about 0.8 to about 0.02, under both rules"),
        ("### 33.23", "With a fixed code it learns only when each cell has an input for every byte, 2.68 times slower than back-propagation (E50)."),
    ]
    for head, s in checks:
        print(f"5.1 {s in section(head)!s:5} memo {head[4:]}: {s[:120]}{'...' if len(s) > 120 else ''}")
    now = MEMO.read_text()
    print(f"5.1 in the working tree the sentence of 33.23 with 2.68 against back-propagation is still present: "
          f"{checks[-1][1] in now}; the corrected text with 2.23 is present: {'2.23 times the exact gradient' in now}")
    src = show(f"{M}/e49_interface.py").decode()
    ok = 'ro = Random(f"e49|readout|{seed}|{k_out}")' in src and "sorted(ro.sample(range(U), k_out))" in src
    ro = Random(f"e49|readout|{SEED}|8")
    out = [sorted(ro.sample(range(256), 8)) for _ in range(512)]
    per_cell = np.bincount([j for s in out for j in s], minlength=256)
    print(f"5.2 e49_interface.draws for out8 (text present: {ok}): readings a cell {per_cell.min()} to {per_cell.max()} (mean {per_cell.mean():.0f}); "
          f"cells read by no readout unit {int((per_cell == 0).sum())}")


def main():
    part1()
    part3()
    part4()
    part5()
    part2()
    line()
    print(f"done in {time.time() - T0:.0f} s")


if __name__ == "__main__":
    main()
