"""Which cells are in use, and where an unrelated task's solution lies relative to the birth (E43's networks).  Read-only.

Files: the E26 births on the default branch; rows 64 and 4,096 of E43's `unrelated` runs (no decay) and row 4,096 of its
`unrelated-rehearse` runs (no decay) on claude/peaceful-hopper-8ofq4i.  Nothing is trained; every figure is a forward pass of
the numpy reader on saved or recombined weights.

Terms.  The first family is the family of the birth (modulus 89); its score here is the share of whole answers right on
reduction pairs of add and sub without letters (600 each).  The unrelated task is E43's (copy two symbols after fillers; its
bytes are 128 to 225, none of which occurs in the first family); its score is the share of whole answers right on the
held-out panel (300 pairs at 12 fillers).

For a cell at one position of one item, with update gate z, candidate n and previous state h (torch's GRU, h' = (1-z) n + z h):
  change factor   (1-z)(1-n^2) + z(1-z)|h - n| : every change to the cell's incoming connections (candidate and update gate)
                  made at that position, under back-propagation and under the sweep alike, is a product containing this factor;
                  where it is 0 the cell's incoming connections cannot change, whatever the error
  state size      |h'| : every change to a connection leaving the cell is a product containing the cell's state

  A  the two quantities per cell on first-family items and on unrelated items, in the birth and after unrelated learning
  B  how the change made by unrelated learning is spread over cells (incoming and outgoing connections of each cell)
  C  the straight path from the birth to the network after unrelated learning: both scores along it; and both scores when
     only one group of tensors is taken from the unrelated network
  D  the change made with rehearsal beside the change made without: sizes, and the cosine between them
  E  the straight paths from the birth to the rehearsed network and from the unrehearsed to the rehearsed network

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/cells_in_use.py [A B C D E] [seed93683] [ruleS ruleB]
"""
import os, sys
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import load_weights, sigmoid, apply, SYM, BOS, EOS, P, DEV
import e43_unrelated_reader as U43

WREF = 'origin/claude/peaceful-hopper-8ofq4i'
E43 = 'MOD97_LOCAL_RULE_20261106/exp45/e43/'
args = sys.argv[1:]
SEED = next((int(a[4:]) for a in args if a.startswith('seed')), 73129)
RULES = [a[4:] for a in args if a in ("ruleS", "ruleB")] or ["S", "B"]
want = {a for a in args if a in "ABCDE" and len(a) == 1} or set("ABCDE")
GROUPS = {"byte table and l1": ["book.weight", "l1.weight", "l1.bias"],
          "input map": ["rnn.weight_ih_l0", "rnn.bias_ih_l0"],
          "recurrent": ["rnn.weight_hh_l0", "rnn.bias_hh_l0"],
          "readout": ["lm.weight", "lm.bias", "l2.weight", "l2.bias", "l3.weight", "l3.bias"]}


def run(W, items, stats=False):
    """Whole answers right per item; with stats, also the per-cell means of the change factor and of the state size over
    the positions after the start symbol, and the share of (cell, position) pairs at which the change factor is below 0.01."""
    x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64); tgt = np.array([a for _, a in items], dtype=np.int64)
    z1 = np.tanh(W['book.weight'] @ W['l1.weight'].T + W['l1.bias'])
    gi_tok = z1 @ W['rnn.weight_ih_l0'].T + W['rnn.bias_ih_l0']; l2_tok = z1 @ W['l2.weight'].T + W['l2.bias']
    Whh, bhh = W['rnn.weight_hh_l0'], W['rnn.bias_hh_l0']; U = Whh.shape[1]
    h = np.zeros((len(x), U), np.float32); ok = np.ones(len(x), bool); L = x.shape[1]; n3 = tgt.shape[1]
    cf = np.zeros(U); st = np.zeros(U); low = 0.0; cnt = 0
    for t in range(L):
        gi = gi_tok[x[:, t]]; gh = h @ Whh.T + bhh
        r = sigmoid(gi[:, :U] + gh[:, :U]); z = sigmoid(gi[:, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:])
        if stats and t >= 1:
            f = (1 - z) * (1 - n * n) + z * (1 - z) * np.abs(h - n)
            cf += f.mean(0); low += float((f < 0.01).mean()); cnt += 1
        h = (1 - z) * n + z * h
        if stats and t >= 1:
            st += np.abs(h).mean(0)
        if t >= L - n3:
            ok &= (np.tanh(l2_tok[x[:, t]] + h @ W['lm.weight'].T + W['lm.bias']) @ W['l3.weight'].T + W['l3.bias']).argmax(1) == tgt[:, t - (L - n3)]
    return (ok, cf / cnt, st / cnt, low / cnt) if stats else ok


def plain(task, pairs, k=0, rng=None):
    op, m, code = task
    out = []
    for a, b in pairs:
        letters = [rng.choice(list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")) for _ in range(k)] if k else []
        out.append((list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + letters + list(f"{b:02d}".encode()) + [ord("=")],
                    list(f"{apply(op, m, a, b):02d}".encode()) + [EOS]))
    return out


allp = [(a, b) for a in range(89) for b in range(89)]
FAM0, FAM12 = [], []
for t in DEV[:2]:
    red = [(a, b) for a, b in allp if ((a + b >= 89) if t[0] == "add" else (a - b < 0))]
    chosen = Random(f"inuse|{t[0]}").sample(red, 600)
    FAM0.append(plain(t, chosen)); FAM12.append(plain(t, chosen[:300], 12, Random(f"inuse|letters|{t[0]}")))
UNR12 = U43.panel(SEED, "held")
UNR0 = [U43.item(p[1], p[2], 0, None) for p, _ in UNR12]


def fam(W):
    return 100 * float(np.mean([run(W, it).mean() for it in FAM0]))


def unr(W):
    return 100 * float(run(W, UNR12).mean())


def mix(Wa, Wb, alpha):
    return {k: ((1 - alpha) * Wa[k] + alpha * Wb[k]).astype(np.float32) for k in Wa}


def swap(base, other, group):
    W = {k: v.copy() for k, v in base.items()}
    for k in GROUPS[group]:
        W[k] = other[k].copy()
    return W


def q(v):
    return "5%/50%/95% of cells " + "/".join(f"{x:.3f}" for x in np.quantile(v, [.05, .5, .95]))


def cell_change(Wa, Wb):
    """Per cell: relative change of its incoming connections (its rows of the input map and of the recurrent matrix, three
    gates) and of its outgoing connections (its column of the recurrent matrix and of lm)."""
    Un = Wa['rnn.weight_hh_l0'].shape[1]
    def rows(W):
        return np.concatenate([np.concatenate([W['rnn.weight_ih_l0'][g * Un:(g + 1) * Un], W['rnn.weight_hh_l0'][g * Un:(g + 1) * Un]], 1) for g in range(3)], 1)
    def cols(W):
        return np.concatenate([W['rnn.weight_hh_l0'], W['lm.weight']], 0).T
    inc = np.linalg.norm(rows(Wb) - rows(Wa), axis=1) / np.linalg.norm(rows(Wa), axis=1)
    out = np.linalg.norm(cols(Wb) - cols(Wa), axis=1) / np.linalg.norm(cols(Wa), axis=1)
    inc2 = np.sort(np.linalg.norm(rows(Wb) - rows(Wa), axis=1) ** 2)[::-1]
    half = int(np.searchsorted(np.cumsum(inc2) / inc2.sum(), 0.5)) + 1
    return inc, out, half


for rule in RULES:
    name = 'the sweep' if rule == 'S' else 'back-propagation'
    Wb, _ = load_weights(P + f"e26/e26_dev_{rule}_{SEED}.pt")
    try:
        W64, _ = load_weights(f"{E43}e43_unrelated_{rule}_{SEED}_row64.pt", WREF)
    except AssertionError:                     # row 64 was saved for seed 73129 only
        W64 = None
    W4k, _ = load_weights(f"{E43}e43_unrelated_{rule}_{SEED}_row4096.pt", WREF)
    Wr, _ = load_weights(f"{E43}addendum2/e43_unrelated-rehearse_{rule}_{SEED}_row4096.pt", WREF)
    print(f"===== birth under {name}, seed {SEED}")
    print(f"   scores (first family, unrelated task): birth {fam(Wb):5.1f}, {unr(Wb):5.1f}" + (f" | after 64 unrelated updates {fam(W64):5.1f}, {unr(W64):5.1f}" if W64 else "")
          + f" | after 4,096 {fam(W4k):5.1f}, {unr(W4k):5.1f} | after 4,096 with rehearsal {fam(Wr):5.1f}, {unr(Wr):5.1f}")
    if "A" in want:
        print("A. cells in use (256 cells)")
        for label, W in (("birth", Wb), ("after 64 unrelated updates", W64), ("after 4,096 unrelated updates", W4k)):
            if W is None:
                continue
            res = {}
            for nm, sets in (("first family, no letters", FAM0), ("first family, 12 letters", FAM12), ("unrelated, no fillers", [UNR0]), ("unrelated, 12 fillers", [UNR12])):
                r = [run(W, it, stats=True) for it in sets]
                res[nm] = (np.mean([a[1] for a in r], 0), np.mean([a[2] for a in r], 0), float(np.mean([a[3] for a in r])))
            print(f"   {label}")
            for nm, (cf, st, low) in res.items():
                print(f"      {nm:26s} change factor: {q(cf)}; cells below 0.01: {int((cf < 0.01).sum()):3d}, below 0.05: {int((cf < 0.05).sum()):3d};"
                      f" (cell, position) pairs below 0.01: {100 * low:4.1f}% | state size: {q(st)}; cells below 0.05: {int((st < 0.05).sum()):3d}")
            a, b = res["first family, 12 letters"], res["unrelated, 12 fillers"]
            ratio = (a[0] + 1e-9) / (b[0] + 1e-9)
            print(f"      between first family (12 letters) and unrelated (12 fillers): correlation of the change factor over cells {np.corrcoef(a[0], b[0])[0, 1]:+.2f},"
                  f" of the state size {np.corrcoef(a[1], b[1])[0, 1]:+.2f}; cells whose change factor is 10 times larger on the family: {int((ratio > 10).sum())},"
                  f" on the unrelated task: {int((ratio < 0.1).sum())}; within a factor of 2: {int(((ratio > 0.5) & (ratio < 2)).sum())}")
    if "B" in want:
        print("B. how the change made by unrelated learning is spread over cells")
        for label, W in (("64 updates", W64), ("4,096 updates", W4k), ("4,096 updates with rehearsal", Wr)):
            if W is None:
                continue
            inc, out, half = cell_change(Wb, W)
            print(f"   {label:28s} incoming connections, change relative to size at birth: {q(inc)}; half of the squared change is in {half} of 256 cells"
                  f" | outgoing: {q(out)}")
    if "C" in want:
        print("C. the straight path from the birth (0) to the network after unrelated learning (1): first family, unrelated task")
        for label, W in (("row 64", W64), ("row 4,096", W4k)):
            if W is None:
                continue
            row = [f"{al:g}: {fam(mix(Wb, W, al)):5.1f}, {unr(mix(Wb, W, al)):5.1f}" for al in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85)]
            print(f"   to {label:9s} " + " | ".join(row))
        print("   one group of tensors taken from the unrelated network at row 4,096, the rest the birth's; and that group put back to the birth's")
        for g in GROUPS:
            a = swap(Wb, W4k, g); b = swap(W4k, Wb, g)
            print(f"      {g:18s} only this group changed: {fam(a):5.1f}, {unr(a):5.1f}   |   this group put back: {fam(b):5.1f}, {unr(b):5.1f}")
        a = swap(swap(Wb, W4k, "byte table and l1"), W4k, "readout"); b = swap(swap(Wb, W4k, "input map"), W4k, "recurrent")
        print(f"      byte table, l1 and readout changed, cells' connections the birth's: {fam(a):5.1f}, {unr(a):5.1f}   |   cells' connections changed, the rest the birth's: {fam(b):5.1f}, {unr(b):5.1f}")
    if "D" in want:
        print("D. the change made in 4,096 unrelated updates without rehearsal (d1) and with it (d2), relative to the birth")
        for g, keys in list(GROUPS.items()) + [("whole network", list(Wb))]:
            d1 = np.concatenate([(W4k[k] - Wb[k]).ravel() for k in keys]); d2 = np.concatenate([(Wr[k] - Wb[k]).ravel() for k in keys])
            b0 = np.concatenate([Wb[k].ravel() for k in keys])
            print(f"   {g:18s} |d1|/|birth| {np.linalg.norm(d1) / np.linalg.norm(b0):.3f}   |d2|/|birth| {np.linalg.norm(d2) / np.linalg.norm(b0):.3f}"
                  f"   cosine(d1, d2) {float(d1 @ d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)):+.3f}   |d2 - d1|/|d1| {np.linalg.norm(d2 - d1) / np.linalg.norm(d1):.3f}")
    if "E" in want:
        print("E. straight paths to the rehearsed network: first family, unrelated task")
        for label, Wa in (("from the birth", Wb), ("from the unrehearsed network (row 4,096)", W4k)):
            row = [f"{al:g}: {fam(mix(Wa, Wr, al)):5.1f}, {unr(mix(Wa, Wr, al)):5.1f}" for al in (0.1, 0.25, 0.5, 0.75, 0.9)]
            print(f"   {label:42s} " + " | ".join(row))
