"""What 64 updates change when a network that has lived learns a new family (E40, seed 73129).  Read-only.

Files: the life-end weights (row 0 of E40's lived runs) on the default branch, and rows 64 and 8,192 of the same runs on
claude/vibrant-allen-8hziln at 45b6fd79.  Nothing is trained; every figure is a forward pass of the numpy reader on saved
or recombined weights.  Whole answers (two digits and the end symbol), no letters.

  A  scores at rows 0, 64 and 8,192: the new family (mod 83, a tenth taught) on never-taught pairs, with and without the
     reduction; the two earlier families of the life (mod 97, mod 73)
  B  how far each tensor moved in 64 updates and in 8,192, relative to its size at the life's end; the byte table's rows
  C  grafts: the life-end network with one group of tensors taken from row 64, and row 64 with one group put back
  D  how many changed weights are needed: the life-end network plus only the largest changes (by size), beside the same
     number chosen at random
  E  how many directions are needed: every matrix's change replaced by its best approximation of rank k

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e40_row64_changes.py [A B C D E] [armS armB armK]
"""
import os, sys
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import load_weights, split, sigmoid, apply, SYM, BOS, EOS, P

DREF = 'origin/codex/minifly-consolidated-20261023'
WREF = '45b6fd79'
SEED = 73129
E40 = 'MOD97_LOCAL_RULE_20261106/exp45/e40/'
LIFE = {"S": P + "e26/e26_life_S_73129.pt", "B": P + "e26/e26_life_B_73129.pt", "K": P + "e31/e31_life_K_73129.pt"}
F4 = [("add", 83, b"e"), ("sub", 83, b"f")]
OLD = [("add", 97, b"r"), ("sub", 97, b"t"), ("add", 73, b"s"), ("sub", 73, b"u")]
GROUPS = {
    "new code rows": None,                       # the four rows of the byte table for e f g h (handled apart)
    "byte table": ["book.weight"],
    "l1": ["l1.weight", "l1.bias"],
    "input map": ["rnn.weight_ih_l0", "rnn.bias_ih_l0"],
    "recurrent": ["rnn.weight_hh_l0", "rnn.bias_hh_l0"],
    "lm": ["lm.weight", "lm.bias"],
    "l2": ["l2.weight", "l2.bias"],
    "l3": ["l3.weight", "l3.bias"],
}
CODES = [ord(c) for c in "efgh"]
args = sys.argv[1:]
want = {a for a in args if a in ("A", "B", "C", "D", "E")} or set("ABCDE")
arms = [a[3:] for a in args if a in ("armS", "armB", "armK")] or ["S", "B", "K"]


def run_items(W, items):
    x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64); tgt = np.array([a for _, a in items], dtype=np.int64)
    z1 = np.tanh(W['book.weight'] @ W['l1.weight'].T + W['l1.bias'])
    gi_tok = z1 @ W['rnn.weight_ih_l0'].T + W['rnn.bias_ih_l0']; l2_tok = z1 @ W['l2.weight'].T + W['l2.bias']
    Whh, bhh = W['rnn.weight_hh_l0'], W['rnn.bias_hh_l0']; U = Whh.shape[1]
    h = np.zeros((len(x), U), np.float32); ok = np.ones(len(x), bool); L = x.shape[1]
    for t in range(L):
        gi = gi_tok[x[:, t]]; gh = h @ Whh.T + bhh
        r = sigmoid(gi[:, :U] + gh[:, :U]); z = sigmoid(gi[:, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:]); h = (1 - z) * n + z * h
        if t >= L - 3:
            ok &= (np.tanh(l2_tok[x[:, t]] + h @ W['lm.weight'].T + W['lm.bias']) @ W['l3.weight'].T + W['l3.bias']).argmax(1) == tgt[:, t - (L - 3)]
    return ok


def plain(task, pairs):
    op, m, code = task
    return [(list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + list(f"{b:02d}".encode()) + [ord("=")], list(f"{apply(op, m, a, b):02d}".encode()) + [EOS]) for a, b in pairs]


# the probes, fixed once
PROBE = {}
for t in F4:
    _, held = split(t, .10, SEED)
    wrap = [(a + b >= 83) if t[0] == "add" else (a - b < 0) for a, b in held]
    red = [p for p, w in zip(held, wrap) if w]; nor = [p for p, w in zip(held, wrap) if not w]
    PROBE[("red", t[0])] = plain(t, Random(f"probe|red|{t[0]}").sample(red, 1500))
    PROBE[("nor", t[0])] = plain(t, Random(f"probe|nor|{t[0]}").sample(nor, 1000))
for t in OLD:
    allp = [(a, b) for a in range(t[1]) for b in range(t[1])]
    PROBE[("old", f"{t[0]}{t[1]}")] = plain(t, Random(f"probe|old|{t[0]}{t[1]}").sample(allp, 600))


def scores(W):
    """(new family: reduction pairs, pairs without it; mod 97; mod 73), each the mean of add and sub, percent."""
    g = lambda keys: 100 * float(np.mean([run_items(W, PROBE[k]).mean() for k in keys]))
    return (g([("red", "add"), ("red", "sub")]), g([("nor", "add"), ("nor", "sub")]),
            g([("old", "add97"), ("old", "sub97")]), g([("old", "add73"), ("old", "sub73")]))


fmt = lambda s: f"new: reduction {s[0]:5.1f}, none {s[1]:5.1f} | mod 97 {s[2]:5.1f} | mod 73 {s[3]:5.1f}"


def graft(base, other, group):
    W = {k: v.copy() for k, v in base.items()}
    if group == "new code rows":
        W["book.weight"][CODES] = other["book.weight"][CODES]
    else:
        for k in GROUPS[group]:
            W[k] = other[k].copy()
    return W


for arm in arms:
    W0, _ = load_weights(LIFE[arm], DREF)
    W64, _ = load_weights(f"{E40}e40_lived_{arm}_{SEED}_row64.pt", WREF)
    W8k, _ = load_weights(f"{E40}e40_lived_{arm}_{SEED}_row8192.pt", WREF)
    print(f"===== lived {arm}, seed {SEED}")
    if "A" in want:
        print("A. scores")
        for name, W in (("life end (row 0)", W0), ("row 64", W64), ("row 8,192", W8k)):
            print(f"   {name:17s} {fmt(scores(W))}")
    if "B" in want:
        print("B. change of each tensor, as a share of its size at the life's end (Frobenius norm)")
        for k in W0:
            n0 = np.linalg.norm(W0[k])
            print(f"   {k:22s} {str(W0[k].shape):12s} after 64: {np.linalg.norm(W64[k] - W0[k]) / n0:6.3f}   after 8,192: {np.linalg.norm(W8k[k] - W0[k]) / n0:6.3f}"
                  f"   largest single change after 64: {np.abs(W64[k] - W0[k]).max():.3f}; mean size of a weight at the life's end {np.abs(W0[k]).mean():.3f}")
        d = np.linalg.norm(W64["book.weight"] - W0["book.weight"], axis=1); n = np.linalg.norm(W0["book.weight"], axis=1)
        r3 = lambda v: [round(float(x), 3) for x in v]
        old_codes = [ord(c) for c in "pqrtsu"]
        print(f"   byte table rows after 64 updates: the four new codes moved {r3(d[CODES])} (row sizes {r3(n[CODES])});")
        print(f"      digits moved on average {d[48:58].mean():.3f} (size {n[48:58].mean():.3f}); + - = moved {r3(d[[43, 45, 61]])};"
              f" codes of earlier families (p q r t s u) moved {r3(d[old_codes])} (sizes {r3(n[old_codes])})")
    if "C" in want:
        print("C. grafts between the life's end and row 64")
        print("   group            | life end + this group from row 64                                | row 64 with this group put back")
        for g in GROUPS:
            print(f"   {g:16s} | {fmt(scores(graft(W0, W64, g)))} | {fmt(scores(graft(W64, W0, g)))}")
        both = graft(graft(W0, W64, "input map"), W64, "recurrent")
        print(f"   input map + recurrent from row 64: {fmt(scores(both))}")
        rd = graft(graft(graft(W0, W64, "lm"), W64, "l2"), W64, "l3")
        print(f"   readout (lm, l2, l3) from row 64:  {fmt(scores(rd))}")
        fr = graft(graft(W0, W64, "byte table"), W64, "l1")
        print(f"   byte table + l1 from row 64:       {fmt(scores(fr))}")
    if "D" in want:
        print("D. the life-end network plus only a share of the changes made in 64 updates")
        D = {k: W64[k] - W0[k] for k in W0}
        flat = np.concatenate([np.abs(v).ravel() for v in D.values()]); total = flat.size
        rng = np.random.default_rng(20261116)
        for q in (0.001, 0.01, 0.05, 0.2, 0.5):
            thr = np.partition(flat, total - int(q * total))[total - int(q * total)]
            Wt = {k: W0[k] + np.where(np.abs(D[k]) >= thr, D[k], 0) for k in W0}
            Wr = {k: W0[k] + np.where(rng.random(D[k].shape) < q, D[k], 0) for k in W0}
            kept = sum(int((np.abs(D[k]) >= thr).sum()) for k in W0)
            by = {k.split('.')[0] + ('.ih' if 'ih' in k else '.hh' if 'hh' in k else ''): 0 for k in W0}
            for k in W0:
                by[k.split('.')[0] + ('.ih' if 'ih' in k else '.hh' if 'hh' in k else '')] += int((np.abs(D[k]) >= thr).sum())
            print(f"   largest {100 * q:5.1f}% ({kept:7,} weights; by part {by})")
            print(f"        largest: {fmt(scores(Wt))}")
            print(f"        random:  {fmt(scores(Wr))}")
    if "E" in want:
        print("E. every matrix's change replaced by its best approximation of rank k (biases kept whole)")
        D = {k: W64[k] - W0[k] for k in W0}
        sv = {k: np.linalg.svd(v, full_matrices=False) for k, v in D.items() if v.ndim == 2}
        for k, (u, s, vt) in sv.items():
            e = s ** 2 / (s ** 2).sum()
            print(f"   {k:22s} share of the squared change in the first 1, 4, 16 directions: {e[:1].sum():.2f}, {e[:4].sum():.2f}, {e[:16].sum():.2f}")
        for kk in (1, 2, 4, 8, 16, 64):
            Wk = {}
            for k in W0:
                if k in sv:
                    u, s, vt = sv[k]; r = min(kk, len(s))
                    Wk[k] = W0[k] + (u[:, :r] * s[:r]) @ vt[:r]
                else:
                    Wk[k] = W64[k]
            print(f"   rank {kk:2d}: {fmt(scores(Wk))}")
