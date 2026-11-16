"""Where the first family is lost when a born network learns an unrelated task (E43, seed 73129).  Read-only.

Files: the E26 births on the default branch, and rows 64, 1,024 and 4,096 of E43's `unrelated` runs (no decay) on
claude/peaceful-hopper-8ofq4i.  Nothing is trained; every figure is a forward pass of the numpy reader on saved or
recombined weights.  The first family's reduction pairs of add and sub (1,500 each, no letters), percent.

  whole answers      the two digits and the end symbol, each the largest output
  digits only        the two digits right when each is chosen among the ten digit outputs only (the end symbol not asked)
  grafts             the unrelated network with one group of tensors put back to the birth's, and the birth with one group
                     taken from the unrelated network

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e43_where_the_loss_sits.py
"""
import os, sys
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import load_weights, sigmoid, apply, SYM, BOS, EOS, P, DEV

WREF = 'origin/claude/peaceful-hopper-8ofq4i'
E43 = 'MOD97_LOCAL_RULE_20261106/exp45/e43/'
GROUPS = {"byte table and l1": ["book.weight", "l1.weight", "l1.bias"],
          "input map": ["rnn.weight_ih_l0", "rnn.bias_ih_l0"],
          "recurrent": ["rnn.weight_hh_l0", "rnn.bias_hh_l0"],
          "readout": ["lm.weight", "lm.bias", "l2.weight", "l2.bias", "l3.weight", "l3.bias"]}
DIG = list(range(48, 58))


def run(W, items):
    """(whole answers right, both digits right among the digit outputs only), booleans per item."""
    x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64); tgt = np.array([a for _, a in items], dtype=np.int64)
    z1 = np.tanh(W['book.weight'] @ W['l1.weight'].T + W['l1.bias'])
    gi_tok = z1 @ W['rnn.weight_ih_l0'].T + W['rnn.bias_ih_l0']; l2_tok = z1 @ W['l2.weight'].T + W['l2.bias']
    Whh, bhh = W['rnn.weight_hh_l0'], W['rnn.bias_hh_l0']; U = Whh.shape[1]
    h = np.zeros((len(x), U), np.float32); ok = np.ones(len(x), bool); okd = np.ones(len(x), bool); L = x.shape[1]
    for t in range(L):
        gi = gi_tok[x[:, t]]; gh = h @ Whh.T + bhh
        r = sigmoid(gi[:, :U] + gh[:, :U]); z = sigmoid(gi[:, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:]); h = (1 - z) * n + z * h
        if t >= L - 3:
            lg = np.tanh(l2_tok[x[:, t]] + h @ W['lm.weight'].T + W['lm.bias']) @ W['l3.weight'].T + W['l3.bias']; j = t - (L - 3)
            ok &= lg.argmax(1) == tgt[:, j]
            if j < 2:
                okd &= (np.array(DIG)[lg[:, DIG].argmax(1)] == tgt[:, j])
    return ok, okd


def plain(task, pairs):
    op, m, code = task
    return [(list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + list(f"{b:02d}".encode()) + [ord("=")], list(f"{apply(op, m, a, b):02d}".encode()) + [EOS]) for a, b in pairs]


allp = [(a, b) for a in range(89) for b in range(89)]
ITEMS = []
for t in DEV[:2]:
    red = [(a, b) for a, b in allp if ((a + b >= 89) if t[0] == "add" else (a - b < 0))]
    ITEMS.append(plain(t, Random(f"where|{t[0]}").sample(red, 1500)))


def score(W):
    r = [run(W, it) for it in ITEMS]
    return 100 * float(np.mean([a.mean() for a, _ in r])), 100 * float(np.mean([b.mean() for _, b in r]))


def swap(base, other, group):
    W = {k: v.copy() for k, v in base.items()}
    for k in GROUPS[group]:
        W[k] = other[k].copy()
    return W


for rule in "SB":
    Wb, _ = load_weights(P + f"e26/e26_dev_{rule}_73129.pt")
    print(f"===== birth under {'the sweep' if rule == 'S' else 'backpropagation'}, seed 73129; reduction pairs of add and sub")
    print(f"   birth                         whole answers {score(Wb)[0]:5.1f}   digits only {score(Wb)[1]:5.1f}")
    for row in (64, 1024, 4096):
        Wu, _ = load_weights(f"{E43}e43_unrelated_{rule}_73129_row{row}.pt", WREF)
        s = score(Wu)
        print(f"   after {row:5,} unrelated updates  whole answers {s[0]:5.1f}   digits only {s[1]:5.1f}")
        for g in GROUPS:
            a = score(swap(Wu, Wb, g)); b = score(swap(Wb, Wu, g))
            print(f"        {g:18s} put back to the birth's: whole {a[0]:5.1f}, digits only {a[1]:5.1f}   |   the birth with only this group changed: whole {b[0]:5.1f}, digits only {b[1]:5.1f}")
        ab = swap(swap(Wu, Wb, "input map"), Wb, "recurrent"); s2 = score(ab)
        print(f"        input map and recurrent put back: whole {s2[0]:5.1f}, digits only {s2[1]:5.1f}")
