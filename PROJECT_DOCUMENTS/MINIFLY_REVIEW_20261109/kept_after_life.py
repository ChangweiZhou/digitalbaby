"""What survives a life of the first family (mod 89), which the life never rehearses.  Read-only.

The six E26 births and the nine lives made from them (E26 arms S and B, E31 arm K), on the default branch.  For each file:
whole answers on 1,500 pairs of add, sub, max and min mod 89 without letters; add and sub split by whether the answer needs
the reduction.  Nothing is trained.

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/kept_after_life.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from random import Random
import numpy as np
from np_gru import load_weights, sigmoid, apply, SYM, BOS, EOS, P, DEV
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
allp = [(a, b) for a in range(89) for b in range(89)]
probe = {t[0]: Random(f"kept|{t[0]}").sample(allp, 1500) for t in DEV}
wrap = {"add": lambda a, b: a + b >= 89, "sub": lambda a, b: a - b < 0}
print("the birth family (mod 89), never rehearsed during the life; 1,500 pairs each, no letters, whole answers, percent")
print("file                         add: needs reduction / not | sub: needs reduction / not | max  | min")
for kind, files in (("birth", ["e26/e26_dev_S_{s}.pt", "e26/e26_dev_B_{s}.pt"]), ("life", ["e26/e26_life_S_{s}.pt", "e26/e26_life_B_{s}.pt", "e31/e31_life_K_{s}.pt"])):
    for f in files:
        for s in (73129, 93683, 116089):
            W, _ = load_weights(P + f.format(s=s))
            out = {}
            for t in DEV:
                ok = run_items(W, plain(t, probe[t[0]]))
                if t[0] in wrap:
                    w = np.array([wrap[t[0]](a, b) for a, b in probe[t[0]]])
                    out[t[0]] = f"{100 * ok[w].mean():5.1f} / {100 * ok[~w].mean():5.1f}"
                else:
                    out[t[0]] = f"{100 * ok.mean():5.1f}"
            print(f"{f.format(s=s):28s} {out['add']}             | {out['sub']}             | {out['max']} | {out['min']}", flush=True)
