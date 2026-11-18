"""Directions in use: whether the change made by an unrelated task can be kept out of what the first family uses (E43's networks).  Read-only.

Files: the E26 births on the default branch; rows 64 and 4,096 of E43's `unrelated` runs (no decay) and row 4,096 of its
`unrelated-rehearse` runs (no decay) on claude/peaceful-hopper-8ofq4i.  Nothing is trained; every figure is a forward pass of
the numpy reader on saved or recombined weights.

The idea tested.  A matrix of connections W receives input vectors x.  If a change dW satisfies dW x = 0 for every input x
that the first family's items present to that matrix, the matrix's output on those items is unchanged; if this holds for
every matrix, the whole forward pass on those items is unchanged, and the family is kept exactly.  The inputs that the
family presents to a matrix span a set of directions; "the directions in use" below are the leading eigenvectors of the
second-moment matrix of those inputs (a constant 1 appended for the bias), enough of them to hold a stated share of its
trace.  Projecting a change away from them, dW (I - P), is a rule local to the matrix: P depends only on the activity of the
matrix's own inputs.

  F1  how many directions the first family's taught items use at each matrix (of 513, 257 or 9 with the bias)
  F1b how much of the code of a byte that never occurred lies in the directions of the bytes in use
  F2  what share of the change made by unrelated learning lies in those directions, without rehearsal and with it
  F3  the birth plus the change projected away from those directions: first family and unrelated task
      (the byte table's rows of bytes the family uses, and l1, are put back to the birth's: the family uses all 8 of l1's
      input directions)

The family's inputs are taken from its taught half only (split 0.5 of each of the four operations, 0 to 12 letters, 40 items
at each number of letters); the family's score is on reduction pairs of add and sub without letters drawn from all pairs,
and on a 12-letter panel of held-out pairs.

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/directions_in_use.py [seed93683] [ruleS ruleB]
"""
import os, sys
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import load_weights, sigmoid, apply, split, SYM, BOS, EOS, P, DEV
import e43_unrelated_reader as U43

WREF = 'origin/claude/peaceful-hopper-8ofq4i'
E43 = 'MOD97_LOCAL_RULE_20261106/exp45/e43/'
args = sys.argv[1:]
SEED = next((int(a[4:]) for a in args if a.startswith('seed')), 73129)
RULES = [a[4:] for a in args if a in ("ruleS", "ruleB")] or ["S", "B"]
LETTERS = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MATS = {"input map": ("rnn.weight_ih_l0", "rnn.bias_ih_l0"), "recurrent": ("rnn.weight_hh_l0", "rnn.bias_hh_l0"),
        "lm": ("lm.weight", "lm.bias"), "l2": ("l2.weight", "l2.bias"), "l3": ("l3.weight", "l3.bias")}
SHARES = (0.99, 0.999, 0.9999, 1.0)


def forward(W, items, collect=False):
    x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64); tgt = np.array([a for _, a in items], dtype=np.int64)
    z1tok = np.tanh(W['book.weight'] @ W['l1.weight'].T + W['l1.bias'])
    Wih, bih, Whh, bhh = W['rnn.weight_ih_l0'], W['rnn.bias_ih_l0'], W['rnn.weight_hh_l0'], W['rnn.bias_hh_l0']; U = Whh.shape[1]
    h = np.zeros((len(x), U), np.float32); ok = np.ones(len(x), bool); L = x.shape[1]; n3 = tgt.shape[1]
    X = {k: [] for k in MATS}
    for t in range(L):
        z1 = z1tok[x[:, t]]
        if collect:
            X["input map"].append(z1); X["recurrent"].append(h.copy())
        gi = z1 @ Wih.T + bih; gh = h @ Whh.T + bhh
        r = sigmoid(gi[:, :U] + gh[:, :U]); z = sigmoid(gi[:, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:])
        h = (1 - z) * n + z * h
        if t >= L - n3:
            z2 = np.tanh(z1 @ W['l2.weight'].T + W['l2.bias'] + h @ W['lm.weight'].T + W['lm.bias'])
            if collect:
                X["lm"].append(h.copy()); X["l2"].append(z1); X["l3"].append(z2)
            ok &= (z2 @ W['l3.weight'].T + W['l3.bias']).argmax(1) == tgt[:, t - (L - n3)]
    return (ok, {k: np.concatenate(v) for k, v in X.items()}, set(x.ravel().tolist())) if collect else ok


def render(task, a, b, k, rng):
    op, m, code = task
    d = [rng.choice(LETTERS) for _ in range(k)]
    return (list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")], list(f"{apply(op, m, a, b):02d}".encode()) + [EOS])


# the family's taught items, from which the directions are taken
rng = Random(f"directions|{SEED}")
TAUGHT = {k: [] for k in range(13)}
for t in DEV:
    taught, _ = split(t, .5, SEED)
    for k in range(13):
        TAUGHT[k] += [render(t, a, b, k, rng) for a, b in rng.sample(taught, 40)]
# the scores
allp = [(a, b) for a in range(89) for b in range(89)]
RED = []
for t in DEV[:2]:
    red = [(a, b) for a, b in allp if ((a + b >= 89) if t[0] == "add" else (a - b < 0))]
    RED.append([render(t, a, b, 0, rng) for a, b in Random(f"inuse|{t[0]}").sample(red, 600)])
HELD12 = []
for t in DEV[:2]:
    _, held = split(t, .5, SEED)
    r2 = Random(f"directions|held|{t[0]}")
    HELD12.append([render(t, a, b, 12, r2) for a, b in r2.sample(held, 300)])
UNR12 = U43.panel(SEED, "held")


def scores(W):
    return (100 * float(np.mean([forward(W, it).mean() for it in RED])), 100 * float(np.mean([forward(W, it).mean() for it in HELD12])),
            100 * float(forward(W, UNR12).mean()))


fmt = lambda s: f"family: reduction pairs {s[0]:5.1f}, held-out pairs at 12 letters {s[1]:5.1f} | unrelated {s[2]:5.1f}"

for rule in RULES:
    name = 'the sweep' if rule == 'S' else 'back-propagation'
    Wb, _ = load_weights(P + f"e26/e26_dev_{rule}_{SEED}.pt")
    ends = {"64 unrelated updates": load_weights(f"{E43}e43_unrelated_{rule}_{SEED}_row64.pt", WREF)[0]} if SEED == 73129 else {}   # row 64: seed 73129 only
    ends.update({"4,096 unrelated updates": load_weights(f"{E43}e43_unrelated_{rule}_{SEED}_row4096.pt", WREF)[0],
            "4,096 with rehearsal": load_weights(f"{E43}addendum2/e43_unrelated-rehearse_{rule}_{SEED}_row4096.pt", WREF)[0]})
    print(f"===== birth under {name}, seed {SEED}")
    print(f"   birth: {fmt(scores(Wb))}")
    for nm, W in ends.items():
        print(f"   after {nm}: {fmt(scores(W))}")
    # inputs the family's taught items present to each matrix, in the birth
    parts = [forward(Wb, TAUGHT[k], collect=True) for k in range(13)]
    used_bytes = sorted(set().union(*[p[2] for p in parts]))
    eig = {}
    for m in MATS:
        X = np.concatenate([p[1][m] for p in parts]).astype(np.float64)
        X = np.concatenate([X, np.ones((len(X), 1))], 1)
        w, V = np.linalg.eigh(X.T @ X / len(X))
        eig[m] = (w[::-1].clip(min=0), V[:, ::-1], len(X))
    print(f"F1. directions the family's taught items use (bytes in use: {len(used_bytes)}; vectors per matrix: " + ", ".join(f"{m} {eig[m][2]:,}" for m in MATS) + ")")
    kk = {}
    for m in MATS:
        w, V, _ = eig[m]; c = np.cumsum(w) / w.sum()
        kk[m] = {s: (int(np.searchsorted(c, s)) + 1 if s < 1 else int((w > 1e-10 * w[0]).sum())) for s in SHARES}
        print(f"   {m:10s} of {len(w):3d}: " + ", ".join(f"{100 * s:g}% of the trace in {kk[m][s]}" for s in SHARES[:-1]) + f"; eigenvalues above 1e-10 of the largest: {kk[m][1.0]}")
    # F1b: how much of the code of a byte that never occurred lies in the directions of the bytes in use
    z1all = np.tanh(Wb['book.weight'].astype(np.float64) @ Wb['l1.weight'].T.astype(np.float64) + Wb['l1.bias'])
    Xall = np.concatenate([z1all, np.ones((len(z1all), 1))], 1)
    _, _, Vt = np.linalg.svd(Xall[used_bytes], full_matrices=False)
    Xn = Xall[list(range(128, 226))]
    sh_all = np.linalg.norm(Xn @ Vt.T, axis=1) ** 2 / np.linalg.norm(Xn, axis=1) ** 2
    sh_9 = np.linalg.norm(Xn @ Vt[:9].T, axis=1) ** 2 / np.linalg.norm(Xn, axis=1) ** 2
    print(f"F1b. the code (tanh(l1(byte table)), constant appended) of the 98 bytes of the unrelated task, none of which has occurred: share of its squared size inside the span of the "
          f"{len(used_bytes)} bytes in use: mean {sh_all.mean():.4f}, least {sh_all.min():.4f}; inside nine leading directions of those bytes' codes: mean {sh_9.mean():.4f} (a random direction: {len(used_bytes) / 513:.3f})")
    print("F2. share of the squared change that lies in the family's directions (99.9% of the trace); a change in random directions would give the share in brackets")
    for nm, W in ends.items():
        row = []
        for m, (wk, bk) in MATS.items():
            d = np.concatenate([W[wk] - Wb[wk], (W[bk] - Wb[bk])[:, None]], 1).astype(np.float64)
            V = eig[m][1][:, :kk[m][0.999]]
            row.append(f"{m} {np.linalg.norm(d @ V) ** 2 / np.linalg.norm(d) ** 2:.3f} ({kk[m][0.999] / d.shape[1]:.3f})")
        print(f"   {nm:24s} " + "; ".join(row))
    print("F3. the birth plus the change projected away from the family's directions")
    for nm, W in ends.items():
        for s in SHARES:
            Wp = {k: v.copy() for k, v in Wb.items()}
            new_rows = [b for b in range(Wb['book.weight'].shape[0]) if b not in used_bytes]
            Wp['book.weight'][new_rows] = W['book.weight'][new_rows]
            for m, (wk, bk) in MATS.items():
                d = np.concatenate([W[wk] - Wb[wk], (W[bk] - Wb[bk])[:, None]], 1).astype(np.float64)
                V = eig[m][1][:, :kk[m][s]]
                d = d - (d @ V) @ V.T
                Wp[wk] = (Wb[wk] + d[:, :-1]).astype(np.float32); Wp[bk] = (Wb[bk] + d[:, -1]).astype(np.float32)
            print(f"   {nm:24s} directions holding {100 * s:g}% of the trace removed: {fmt(scores(Wp))}")
        # control: the same number of directions removed, chosen at random
        rs = np.random.default_rng(20261117)
        Wp = {k: v.copy() for k, v in Wb.items()}
        Wp['book.weight'][new_rows] = W['book.weight'][new_rows]
        for m, (wk, bk) in MATS.items():
            d = np.concatenate([W[wk] - Wb[wk], (W[bk] - Wb[bk])[:, None]], 1).astype(np.float64)
            Q, _ = np.linalg.qr(rs.standard_normal((d.shape[1], kk[m][0.999])))
            d = d - (d @ Q) @ Q.T
            Wp[wk] = (Wb[wk] + d[:, :-1]).astype(np.float32); Wp[bk] = (Wb[bk] + d[:, -1]).astype(np.float32)
        print(f"   {nm:24s} control, as many random directions removed as at 99.9%: {fmt(scores(Wp))}")
