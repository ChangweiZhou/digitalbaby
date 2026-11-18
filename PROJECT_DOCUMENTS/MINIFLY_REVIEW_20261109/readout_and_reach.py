#!/usr/bin/env python3
"""Readout and reach: read-only measurements on the six saved networks of E50 (C-dense, C46-one, C16-in; each under the rule
by signs, S, and under the exact gradient, B; seed 73129), 18 November 2026.

numpy and the standard library only. Every file is read with `git show` from the default branch into memory; nothing is
trained, no code of the repository is executed, and no file is written. Arithmetic is in double precision.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/readout_and_reach.py

Run from the repository's root. The reader of the weight files (`load_pt`) and the construction of the panels (`render`,
`split`, `panel`, `panels`, `batch_arrays`) are copied from the outside audit's script audit_e61_relay.py, which wrote them from
the text of exp11_cell.py, exp42_cells.py and exp45.py. The forward pass follows e50_sparse_code.forward with period 1
(E42's masked recurrent map, E49's masked input map): z1 is the row of the fixed code table for the current byte; the input map
and the recurrent map are multiplied by their masks (the masks are read from the weight file; the recurrent mask contains the
diagonal); a GRU step; z2 = tanh(l2 z1 + lm h); output l3 z2.

Sections:
  0  what is read; the check against the receipts (greedy generation, and the same counts from the answer positions fed the
     right bytes)
  1  the direct path l2 z1: intact, replaced by its mean over items at each position, removed
  2  the cells silenced: lm h replaced by its mean over items at each position, removed
  3  linear readability of the answer digits (and of b2, b1, a2, a1 and the operation) from h and from z2
  4  share of each cell's variance at `=` explained by b2, b1, a2, a1
"""
import io
import json
import pickle
import subprocess
import sys
import time
import zipfile
from collections import OrderedDict
from random import Random

import numpy as np

T0 = time.time()
REF = "origin/codex/minifly-consolidated-20261023"
E50 = "MOD97_LOCAL_RULE_20261106/exp45/e50/"
SEED = 73129
U, D = 256, 512
SETTINGS = ("C46-one", "C16-in", "C-dense")
DIGITS = list(b"0123456789")


def git_bytes(path):
    r = subprocess.run(["git", "show", f"{REF}:{path}"], capture_output=True)
    assert r.returncode == 0, (path, r.stderr.decode()[:200])
    return r.stdout


# ---------------------------------------------------------------- a torch.save file read without torch (from audit_e61_relay.py)
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


# ---------------------------------------------------------------- the items and the panels (from audit_e61_relay.py)
EOS, BOS = 256, 257
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")


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


def panel(task, pairs, seed, ks, n):
    rng = Random(seed + 3)
    sub = rng.sample(pairs, n)
    return {k: [render(task, a, b, k, rng) for a, b in sub] for k in ks}


def panels(seed, f=.5):
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


# ---------------------------------------------------------------- the network
def sig(a):
    return 1.0 / (1.0 + np.exp(-a))


class Net:
    def __init__(self, W):
        f = lambda k: np.asarray(W[k], np.float64)
        self.table = f("code_table")                                    # (258, 512), fixed
        self.mask_in = np.asarray(W["mask_in"], bool)                   # (256, 512)
        self.mask = np.asarray(W["mask"], bool)                         # (256, 256), diagonal included
        self.mask_out = np.asarray(W["mask_out"], bool)                 # (512, 256)
        self.owner = np.asarray(W["code_owner"]).astype(int)
        self.Wi = f("rnn.weight_ih_l0") * np.tile(self.mask_in, (3, 1))
        self.bi = f("rnn.bias_ih_l0")
        self.Wh = f("rnn.weight_hh_l0") * np.tile(self.mask, (3, 1))
        self.bh = f("rnn.bias_hh_l0")
        self.Wm, self.bm = f("lm.weight") * self.mask_out, f("lm.bias")
        self.W2, self.b2 = f("l2.weight"), f("l2.bias")
        self.W3, self.b3 = f("l3.weight"), f("l3.bias")

    def step(self, z1t, h):
        gi = z1t @ self.Wi.T + self.bi
        gh = h @ self.Wh.T + self.bh
        r = sig(gi[:, :U] + gh[:, :U])
        z = sig(gi[:, U:2 * U] + gh[:, U:2 * U])
        n = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:])
        return (1 - z) * n + z * h

    def states(self, x):
        """Teacher-forced pass: the code z1 and the states h after every position, (B, L, 512) and (B, L, 256)."""
        z1 = self.table[x]
        B, L = x.shape
        h = np.zeros((B, U))
        Hs = np.empty((B, L, U))
        for t in range(L):
            h = self.step(z1[:, t], h)
            Hs[:, t] = h
        return z1, Hs

    def out(self, z1, h):
        return np.tanh(z1 @ self.W2.T + self.b2 + h @ self.Wm.T + self.bm) @ self.W3.T + self.b3

    def greedy(self, prompts, n_new=3):
        """The scorer's greedy generation: the prompt after the start symbol, then n_new bytes, each the largest output."""
        x = np.array([[BOS] + p for p in prompts], np.int64)
        z1 = self.table[x]
        h = np.zeros((len(prompts), U))
        for t in range(x.shape[1]):
            h = self.step(z1[:, t], h)
        got = []
        for _ in range(n_new):
            b = self.out(self.table[x[:, -1]] if not got else self.table[got[-1]], h).argmax(-1)
            got.append(b)
            h = self.step(self.table[b], h)
        return np.stack(got, 1)


def exact_from(pred, y, mask):
    return int(((pred == y) | ~mask).all(1).sum())


# ---------------------------------------------------------------- the classifier of section 3
def ridge_classifier(Xtr, ytr, Xte, yte, ncls):
    """Least-squares classifier: features standardized with the training half's mean and standard deviation; one-hot targets
    centred; ridge penalty lambda * n_train * I, lambda chosen from {1e-4, 1e-3, ..., 1e2} by 5-fold cross-validation inside
    the training half (largest share right); the class is the largest of the ten (or two, or four) fitted outputs."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    A, Bm = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = np.eye(ncls)[ytr]
    lams = [10.0 ** e for e in range(-4, 3)]
    folds = np.arange(len(A)) % 5

    def fit(X, Yy, lam):
        xm, ym = X.mean(0), Yy.mean(0)
        Xc = X - xm
        W = np.linalg.solve(Xc.T @ Xc + lam * len(X) * np.eye(X.shape[1]), Xc.T @ (Yy - ym))
        return lambda Z: (Z - xm) @ W + ym

    cv = []
    for lam in lams:
        ok = 0
        for k in range(5):
            tr, va = folds != k, folds == k
            ok += int((fit(A[tr], Y[tr], lam)(A[va]).argmax(1) == ytr[va]).sum())
        cv.append(ok)
    lam = lams[int(np.argmax(cv))]
    pred = fit(A, Y, lam)(Bm).argmax(1)
    return 100.0 * float((pred == yte).mean()), lam


def softmax_classifier(Xtr, ytr, Xte, yte, ncls, iters=300):
    """Multinomial logistic regression (a linear map and a softmax): features standardized as above; the mean cross-entropy
    plus mu * |W|^2 / 2 (bias not penalized) minimized by 300 steps of Nesterov's accelerated gradient with step 1 / L, L the
    bound 0.5 * (largest eigenvalue of X'X / n) + mu; mu chosen from {1e-4, 1e-3, 1e-2} by 5-fold cross-validation inside the
    training half; the class is the largest output."""
    mu_, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-8] = 1.0
    A, Bm = (Xtr - mu_) / sd, (Xte - mu_) / sd
    folds = np.arange(len(A)) % 5

    def fit(X, y, mu):
        X1 = np.concatenate([X, np.ones((len(X), 1))], 1)
        Y = np.eye(ncls)[y]
        L = 0.5 * np.linalg.eigvalsh(X1.T @ X1 / len(X1))[-1] + mu
        W = np.zeros((X1.shape[1], ncls)); V = W.copy(); tk = 1.0
        pen = np.ones((X1.shape[1], 1)); pen[-1] = 0.0
        for _ in range(iters):
            S = X1 @ V
            S -= S.max(1, keepdims=True)
            P = np.exp(S); P /= P.sum(1, keepdims=True)
            Wn = V - (X1.T @ (P - Y) / len(X1) + mu * pen * V) / L
            tn = (1 + (1 + 4 * tk * tk) ** 0.5) / 2
            V = Wn + (tk - 1) / tn * (Wn - W)
            W, tk = Wn, tn
        return lambda Z: np.concatenate([Z, np.ones((len(Z), 1))], 1) @ W

    mus = [1e-4, 1e-3, 1e-2]
    cv = [sum(int((fit(A[folds != k], ytr[folds != k], mu)(A[folds == k]).argmax(1) == ytr[folds == k]).sum()) for k in range(5)) for mu in mus]
    mu = mus[int(np.argmax(cv))]
    return 100.0 * float((fit(A, ytr, mu)(Bm).argmax(1) == yte).mean()), mu


def majority(ytr, yte):
    return 100.0 * float((yte == np.bincount(ytr).argmax()).mean())


# ---------------------------------------------------------------- the run
def head(s, text):
    print("=" * 110)
    print(f"{s}. {text}")


def main():
    commit = subprocess.run(["git", "rev-parse", REF], capture_output=True, text=True).stdout.strip()
    A, R, TA = panels(SEED)
    PAN = {"A": A, "R": R, "T": TA}
    PNAME = {"A": "A (never-taught pairs, 12 letters; add, sub, max, min, 300 items each)",
             "R": "R (pairs that need the reduction, no letters; add, sub, 1,000 items each)",
             "T": "T (taught pairs, 12 letters; add, sub, max, min, 300 items each)"}
    head("0", f"what is read: {REF} at {commit[:12]}, folder {E50}; the panels rebuilt for seed {SEED}: "
              + "; ".join(f"{k} {sum(len(v) for v in PAN[k].values())} items" for k in "ART"))
    S1, S3, S4 = {}, {}, {}
    for s in SETTINGS:
        for r in "SB":
            key = f"{s} {r}"
            t1 = time.time()
            W = load_pt(git_bytes(f"{E50}e50_{s}_{r}_{SEED}.pt"))
            rc = json.loads(git_bytes(f"{E50}e50_{s}_{r}_{SEED}.json"))
            n = Net(W)
            del W
            row = rc["curve"][-1]
            # ---- 0: the check against the receipt
            cache = {}
            mine_g, mine_t = {}, {}
            for p, sets in PAN.items():
                for task, items in sets.items():
                    x, y, mask = batch_arrays(items)
                    z1, Hs = n.states(x)
                    cache[(p, task)] = (items, x, y, mask, z1, Hs)
                    g = n.greedy([it[0] for it in items])
                    want = np.array([it[1] for it in items])
                    mine_g[(p, task)] = int((g == want).all(1).sum())
                    mine_t[(p, task)] = exact_from(n.out(z1, Hs).argmax(-1), y, mask)
            rec = {"A": row["A_counts"], "R": row["R_counts"], "T": row["taught_counts"]}
            ok = all(mine_g[(p, k)] == rec[p][k] == mine_t[(p, k)] for p in rec for k in rec[p])
            diag = bool(np.diag(n.mask).all())
            print(f"0.{key}: receipt's last scored row, update {row['step']:,}; cells read per cell {int(n.mask_in.sum(1).min())} to {int(n.mask_in.sum(1).max())} values, "
                  f"{int(n.mask.sum(1).min()) - 1} to {int(n.mask.sum(1).max()) - 1} other cells (diagonal in the mask: {diag}); lm's mask full: {bool(n.mask_out.all())}")
            for p in "ART":
                print(f"     {p}: greedy here {[mine_g[(p, k)] for k in rec[p]]}; fed the right bytes here {[mine_t[(p, k)] for k in rec[p]]}; receipt {list(rec[p].values())}")
            print(f"     all {sum(len(v) for v in rec.values())} counts equal in the three forms: {ok}")
            assert ok, key
            # ---- 1 and 2: the readout's two inputs
            res = {}
            for p in "ART":
                cnt = {c: np.zeros(4, int) for c in ("intact", "l2_mean", "l2_zero", "lm_mean", "lm_zero")}
                for task in PAN[p]:
                    items, x, y, mask, z1, Hs = cache[(p, task)]
                    a2 = z1 @ n.W2.T                     # the direct path, without its bias
                    mh = Hs @ n.Wm.T                     # the path through the cells, without its bias
                    a2m = np.broadcast_to(a2.mean(0), a2.shape)
                    mhm = np.broadcast_to(mh.mean(0), mh.shape)
                    for c, (u, v) in dict(intact=(a2, mh), l2_mean=(a2m, mh), l2_zero=(0.0, mh),
                                          lm_mean=(a2, mhm), lm_zero=(a2, 0.0)).items():
                        pred = (np.tanh(u + n.b2 + v + n.bm) @ n.W3.T + n.b3).argmax(-1)
                        q = len(items[0][0])
                        right = pred[:, q:q + 3] == y[:, q:q + 3]       # the three answer outputs, fed the right bytes
                        cnt[c] += np.array([exact_from(pred, y, mask), *right.sum(0)])
                tot = sum(len(v) for v in PAN[p].values())
                res[p] = {c: (v, tot) for c, v in cnt.items()}
            S1[key] = res
            # ---- 3 and 4: what is held at `=` and at the next position
            S3[key], S4[key] = {}, {}
            for p in "RA":
                Hq, Z2q, Z2n, Hn, lab = [], [], [], [], {k: [] for k in ("d1", "d2", "b2", "b1", "a2", "a1", "op")}
                for oi, task in enumerate(PAN[p]):
                    items, x, y, mask, z1, Hs = cache[(p, task)]
                    q = len(items[0][0])                       # index of `=` in x (the start symbol is at index 0)
                    assert all(len(it[0]) == q for it in items) and (x[:, q] == ord("=")).all()
                    z2 = np.tanh(z1 @ n.W2.T + n.b2 + Hs @ n.Wm.T + n.bm)
                    Hq.append(Hs[:, q]); Hn.append(Hs[:, q + 1]); Z2q.append(z2[:, q]); Z2n.append(z2[:, q + 1])
                    for it in items:
                        pr, an = it
                        lab["d1"].append(an[0] - 48); lab["d2"].append(an[1] - 48)
                        lab["a1"].append(pr[1] - 48); lab["a2"].append(pr[2] - 48)
                        lab["b1"].append(pr[-3] - 48); lab["b2"].append(pr[-2] - 48); lab["op"].append(oi)
                Hq, Hn, Z2q, Z2n = map(np.concatenate, (Hq, Hn, Z2q, Z2n))
                lab = {k: np.array(v) for k, v in lab.items()}
                N = len(Hq)
                perm = np.array(Random(f"readout|{p}").sample(range(N), N))
                tr, te = perm[:N // 2], perm[N // 2:]
                out3 = {}
                for nm, X, k in (("h at = -> first digit", Hq, "d1"), ("z2 at = -> first digit", Z2q, "d1"),
                                 ("h next -> second digit", Hn, "d2"), ("z2 next -> second digit", Z2n, "d2"),
                                 ("h at = -> b2", Hq, "b2"), ("h at = -> b1", Hq, "b1"), ("h at = -> a2", Hq, "a2"),
                                 ("h at = -> a1", Hq, "a1"), ("h at = -> operation", Hq, "op")):
                    nc = int(lab[k].max()) + 1 if k == "op" else 10
                    acc, lam = ridge_classifier(X[tr], lab[k][tr], X[te], lab[k][te], nc)
                    sm = softmax_classifier(X[tr], lab[k][tr], X[te], lab[k][te], nc)[0] if k in ("d1", "d2") else None
                    out3[nm] = (acc, majority(lab[k][tr], lab[k][te]), lam, len(tr), len(te), sm)
                S3[key][p] = out3
                # 4: eta squared per cell at `=`
                var = Hq.var(0)
                out4 = {}
                S4[key].setdefault("groups", {})[p] = {k: int(len(set(lab[k].tolist()))) for k in ("b2", "b1", "a2", "a1")}
                for k in ("b2", "b1", "a2", "a1"):
                    g = lab[k]
                    between = sum((g == v).sum() * (Hq[g == v].mean(0) - Hq.mean(0)) ** 2 for v in range(10) if (g == v).any()) / N
                    eta = np.where(var > 1e-12, between / np.maximum(var, 1e-300), 0.0)
                    out4[k] = eta
                S4[key][p] = (out4, N, int((var <= 1e-12).sum()))
            # which cells read a digit byte directly
            own = n.owner
            ndig = np.array([len({int(own[v]) for v in np.nonzero(n.mask_in[i])[0] if own[v] in DIGITS}) for i in range(U)])
            S4[key]["ndig"] = ndig
            print(f"     [{time.time() - t1:.0f} s]", flush=True)
            del n, cache

    head("1", "the direct path l2 z1 at the readout: exact answers (greedy) with l2 z1 intact; replaced at each position by its mean over "
              "the items of the same operation of the panel; removed (the product set to zero, the bias of l2 kept). Per cent exact (in brackets: per cent right at the three answer outputs, first digit, second digit, end symbol, each fed the right bytes) [items]")
    for p in "ART":
        print(f"   panel {PNAME[p]}")
        for key in S1:
            r = S1[key][p]
            f3 = lambda v: f"{100 * v[0][0] / v[1]:6.2f} ({', '.join(f'{100 * w / v[1]:.1f}' for w in v[0][1:])})"
            print(f"      1.{p} {key:10s}: intact {f3(r['intact'])}  mean {f3(r['l2_mean'])}  removed {f3(r['l2_zero'])}   [{r['intact'][1]} items]")
    head("2", "the cells silenced at the readout: exact answers (greedy) with lm h replaced at each position by its mean over the items of "
              "the same operation of the panel; removed (the product set to zero, the bias of lm kept; the same as the source's `silence`). Per cent exact (in brackets: per cent right at the first digit, second digit, end symbol, fed the right bytes) [items]")
    for p in "ART":
        print(f"   panel {PNAME[p]}")
        for key in S1:
            r = S1[key][p]
            f3 = lambda v: f"{100 * v[0][0] / v[1]:6.2f} ({', '.join(f'{100 * w / v[1]:.1f}' for w in v[0][1:])})"
            print(f"      2.{p} {key:10s}: intact {f3(r['intact'])}  mean {f3(r['lm_mean'])}  removed {f3(r['lm_zero'])}   [{r['intact'][1]} items]")
    head("3", "share of test items classified right (per cent) by a least-squares classifier (ridge, lambda by 5-fold cross-validation in the "
              "training half) fitted on one half of the panel's items and tested on the other; [share of the most frequent training class "
              "in the test half]; `=` is the position where the first answer digit is read out, `next` the following position (input: the right first digit)")
    for p in "RA":
        any_key = next(iter(S3))
        ntr, nte = S3[any_key][p]["h at = -> b2"][3:5]
        print(f"   panel {p}: {ntr} training items, {nte} test items")
        for key in S3:
            o = S3[key][p]
            print(f"      3.{p} {key:10s}: " + "; ".join(f"{nm} {v[0]:.1f}" + (f" (softmax {v[5]:.1f})" if v[5] is not None else "") + f" [{v[1]:.1f}]" for nm, v in o.items()))
    head("4", "for each cell, eta squared = (variance between the ten groups of a digit's value) / (total variance) of its state at `=`, over "
              "the panel's items (groups with no item left out); number of cells with eta squared above 0.05 and above 0.20 (of 256)")
    for p in "RA":
        any_key = next(iter(S4))
        print(f"   panel {p}: {S4[any_key][p][1]} items; values present: " + ", ".join(f"{k} {v}" for k, v in S4[any_key]['groups'][p].items())
              + f"; with g groups and N items the expected eta squared of a variable unrelated to the digit is (g - 1) / (N - 1)")
        for key in S4:
            out4, N, zero = S4[key][p]
            print(f"      4.{p} {key:10s}: " + "; ".join(f"{k} {int((e > .05).sum())} / {int((e > .20).sum())}" for k, e in out4.items())
                  + f"   (cells with no variance: {zero}; mean eta2 b2 {out4['b2'].mean():.3f})")
            if key.startswith("C16-in"):
                nd = S4[key]["ndig"]
                for lab_, sel in (("read no digit byte", nd == 0), ("read 1 or 2 digit bytes", (nd >= 1) & (nd <= 2)),
                                  ("read 3 or 4 digit bytes", (nd >= 3) & (nd <= 4)), ("read 5 or more digit bytes", nd >= 5),
                                  ("read at least one digit byte", nd >= 1)):
                    print(f"         {lab_:30s} ({int(sel.sum()):3d} cells): " + "; ".join(
                        f"{k} {int((e[sel] > .05).sum())} / {int((e[sel] > .20).sum())}" for k, e in out4.items()))
    for key in S4:
        nd = S4[key]["ndig"]
        print(f"   4.c {key}: digit bytes of which a cell reads at least one value: least {int(nd.min())}, most {int(nd.max())}; cells reading none {int((nd == 0).sum())}")
    print("=" * 110)
    print(f"total time {time.time() - T0:.0f} s")


if __name__ == "__main__":
    main()
