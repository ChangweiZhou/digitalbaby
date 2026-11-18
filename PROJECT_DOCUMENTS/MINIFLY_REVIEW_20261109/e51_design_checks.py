"""Design check for the job proposed in section 30.3 of the memo (E51: reserve cells, and changes kept out of the directions
that the first family uses).  Read-only and without training: forward passes of a numpy reader on E41's half-width births.

Note on the two forms of the reserve's gates.  The check below uses the form in which the update-gate rows of the reserve
cells are projected and the gates start shut (a bias b0).  In the arm `keyed` of section 30.3 those rows are free and the
gates are held shut on the first family's items by the item's first byte; the family is then kept for the same reason
(the reserve cells' states are zero on its items), and exactly.  The projected form is the arm that section 30.3 leaves
out for the present code and keeps for a birth with E50's code.

The construction checked.  A half-width birth (feedforward width 256, 128 cells; E41, taught fraction 0.5) is placed inside a
network of the present size (512, 256).  The 128 added cells are the reserve: their update gates carry a bias b0 that keeps
them shut (torch's GRU, h' = (1-z) n + z h, so a large positive bias on z keeps the state at its start, 0).  Weights from
added feedforward units into old units are 0; all other added weights are drawn at the scale of a fresh network.

The rule checked.  For every matrix, the directions in use are the eigenvectors of the second-moment matrix of the inputs
that the first family's taught items present to it (a constant 1 appended for the bias) whose eigenvalue exceeds 1e-6 of the
largest.  A change to a row is multiplied on the input side by (I - P), P the projector on those directions, for: every row
of the input map and of the recurrent matrix that belongs to an old cell; the update-gate rows of the reserve cells; every
row of lm, l2 and l3.  The candidate and reset rows of the reserve cells are free.  l1 and the byte table's rows of bytes
the family uses are not changed; the other rows of the byte table are free.

  1  the embedded network answers the first family as the half-width birth does
  2  directions in use and free directions at each matrix, and the size of the projectors
  3  a random change that obeys the rule (every changed entry drawn at a scale of half a fresh weight) leaves the family's
     answers unchanged; the same change without the projection destroys them; the size of the drift for several b0
  4  what share of a reserve cell's state survives on the family's items (the leak through a shut gate), by b0

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e51_design_checks.py [seed93683] [ruleS ruleB]
"""
import os, sys
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import load_weights, sigmoid, apply, split, SYM, BOS, EOS, P, DEV

args = sys.argv[1:]
SEED = next((int(a[4:]) for a in args if a.startswith('seed')), 73129)
RULES = [a[4:] for a in args if a in ("ruleS", "ruleB")] or ["S", "B"]
LETTERS = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
H0, U0, H1, U1 = 256, 128, 512, 256
MATS = {"input map": ("rnn.weight_ih_l0", "rnn.bias_ih_l0"), "recurrent": ("rnn.weight_hh_l0", "rnn.bias_hh_l0"),
        "lm": ("lm.weight", "lm.bias"), "l2": ("l2.weight", "l2.bias"), "l3": ("l3.weight", "l3.bias")}


def forward(W, items, collect=False):
    x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64); tgt = np.array([a for _, a in items], dtype=np.int64)
    z1tok = np.tanh(W['book.weight'] @ W['l1.weight'].T + W['l1.bias'])
    Wih, bih, Whh, bhh = W['rnn.weight_ih_l0'], W['rnn.bias_ih_l0'], W['rnn.weight_hh_l0'], W['rnn.bias_hh_l0']; U = Whh.shape[1]
    h = np.zeros((len(x), U)); L = x.shape[1]; n3 = tgt.shape[1]
    X = {k: [] for k in MATS}; logits = []
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
            logits.append(z2 @ W['l3.weight'].T + W['l3.bias'])
    lg = np.stack(logits, 1)                                   # (items, 3, 257)
    ok = (lg.argmax(-1) == tgt).all(1)
    return ok, lg, ({k: np.concatenate(v) for k, v in X.items()} if collect else None), set(x.ravel().tolist()), h


def render(task, a, b, k, rng):
    op, m, code = task
    d = [rng.choice(LETTERS) for _ in range(k)]
    return (list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")], list(f"{apply(op, m, a, b):02d}".encode()) + [EOS])


rng = Random(f"e51|{SEED}")
TAUGHT = {k: [] for k in range(13)}
for t in DEV:
    taught, _ = split(t, .5, SEED)
    for k in range(13):
        TAUGHT[k] += [render(t, a, b, k, rng) for a, b in rng.sample(taught, 40)]
allp = [(a, b) for a in range(89) for b in range(89)]
RED = []
for t in DEV[:2]:
    red = [(a, b) for a, b in allp if ((a + b >= 89) if t[0] == "add" else (a - b < 0))]
    RED += [render(t, a, b, 0, rng) for a, b in Random(f"e51|red|{t[0]}").sample(red, 400)]
HELD12 = []
for t in DEV[:2]:
    _, held = split(t, .5, SEED)
    r2 = Random(f"e51|held|{t[0]}")
    HELD12 += [render(t, a, b, 12, r2) for a, b in r2.sample(held, 300)]


def embed(Wh, b0, gen):
    """The half-width birth inside a network of the present size."""
    f = lambda shape, bound: gen.uniform(-bound, bound, shape)
    W = {"book.weight": Wh["book.weight"].astype(np.float64).copy()}
    W["l1.weight"] = np.concatenate([Wh["l1.weight"], f((H1 - H0, 8), 8 ** -0.5)]); W["l1.bias"] = np.concatenate([Wh["l1.bias"], f((H1 - H0,), 8 ** -0.5)])
    s = U1 ** -0.5                                            # nn.GRU draws every weight uniformly in +-1/sqrt(units)
    Wih = f((3 * U1, H1), s); Whh = f((3 * U1, U1), s); bih = f((3 * U1,), s); bhh = f((3 * U1,), s)
    for g in range(3):
        o1 = slice(g * U1, g * U1 + U0); o0 = slice(g * U0, (g + 1) * U0)
        Wih[o1, :H0] = Wh["rnn.weight_ih_l0"][o0]; Wih[o1, H0:] = 0.0
        Whh[o1, :U0] = Wh["rnn.weight_hh_l0"][o0]             # old rows keep fresh-scale weights from the reserve cells
        bih[o1] = Wh["rnn.bias_ih_l0"][o0]; bhh[o1] = Wh["rnn.bias_hh_l0"][o0]
    bih[U1 + U0:2 * U1] = b0; bhh[U1 + U0:2 * U1] = 0.0       # update-gate bias of the reserve cells
    W.update({"rnn.weight_ih_l0": Wih, "rnn.weight_hh_l0": Whh, "rnn.bias_ih_l0": bih, "rnn.bias_hh_l0": bhh})
    lm = f((H1, U1), U1 ** -0.5); lm[:H0, :U0] = Wh["lm.weight"]; lmb = f((H1,), U1 ** -0.5); lmb[:H0] = Wh["lm.bias"]
    l2 = f((H1, H1), H1 ** -0.5); l2[:H0, :H0] = Wh["l2.weight"]; l2[:H0, H0:] = 0.0; l2b = f((H1,), H1 ** -0.5); l2b[:H0] = Wh["l2.bias"]
    l3 = np.zeros((257, H1)); l3[:, :H0] = Wh["l3.weight"]
    W.update({"lm.weight": lm, "lm.bias": lmb, "l2.weight": l2, "l2.bias": l2b, "l3.weight": l3, "l3.bias": Wh["l3.bias"].astype(np.float64).copy()})
    return W


def directions(W):
    parts = [forward(W, TAUGHT[k], collect=True) for k in range(13)]
    used = sorted(set().union(*[p[3] for p in parts])); V = {}; info = {}
    for m in MATS:
        X = np.concatenate([p[2][m] for p in parts]); X = np.concatenate([X, np.ones((len(X), 1))], 1)
        w, E = np.linalg.eigh(X.T @ X / len(X)); w = w[::-1].clip(min=0); E = E[:, ::-1]
        k = int((w > 1e-6 * w[0]).sum()); c = np.cumsum(w) / w.sum()
        V[m] = E[:, :k]; info[m] = (len(w), k, int(np.searchsorted(c, 0.9999)) + 1)
    return V, used, info


def changed(W, V, used, gen, scale, project=True):
    """W plus a random change of every entry the rule allows to change, drawn at `scale`; with project=False the same
    entries are changed without the projection."""
    Wn = {k: v.copy() for k, v in W.items()}
    new_rows = [b for b in range(W['book.weight'].shape[0]) if b not in used]
    Wn['book.weight'][new_rows] += gen.normal(0, scale, (len(new_rows), 8))
    for m, (wk, bk) in MATS.items():
        D = gen.normal(0, scale, (W[wk].shape[0], W[wk].shape[1] + 1))
        if project:
            Dp = D - (D @ V[m]) @ V[m].T
            if m in ("input map", "recurrent"):                # candidate and reset rows of the reserve cells stay free
                free = np.zeros(3 * U1, bool); free[U0:U1] = True; free[2 * U1 + U0:] = True
                Dp[free] = D[free]
            D = Dp
        Wn[wk] = W[wk] + D[:, :-1]; Wn[bk] = W[bk] + D[:, -1]
    return Wn


def score(W):
    a, la, _, _, _ = forward(W, RED); b, lb, _, _, _ = forward(W, HELD12)
    return 100 * a.mean(), 100 * b.mean(), la, lb


for rule in RULES:
    Wh, _ = load_weights(P + f"e41/e41_{rule}_h256_f50_{SEED}.pt")
    Wh = {k: v.astype(np.float64) for k, v in Wh.items()}
    r0, h0, l0a, l0b = score(Wh)
    print(f"===== half-width birth under {'the sweep' if rule == 'S' else 'back-propagation'}, seed {SEED}: reduction pairs {r0:5.1f}, held-out pairs at 12 letters {h0:5.1f}")
    for b0 in (6.0, 8.0, 10.0, 14.0):
        gen = np.random.default_rng(20261117)
        W = embed(Wh, b0, gen)
        r1, h1, l1a, l1b = score(W)
        _, _, _, _, hend = forward(W, HELD12)
        V, used, info = directions(W)
        print(f"   b0 = {b0:g}")
        print(f"      1. embedded: reduction pairs {r1:5.1f}, held-out {h1:5.1f}; largest difference of an output from the half-width birth's {max(np.abs(l1a - l0a).max(), np.abs(l1b - l0b).max()):.2e}")
        print(f"      4. largest state of a reserve cell at the end of a family item {np.abs(hend[:, U0:]).max():.2e} (old cells: median size {np.median(np.abs(hend[:, :U0])):.2f})")
        if b0 == 10.0:
            mem = sum(info[m][1] * info[m][0] for m in MATS)
            print("      2. directions in use (eigenvalue above 1e-6 of the largest) of the matrix's input dimensions; in brackets the number holding 99.99% of the trace: "
                  + "; ".join(f"{m} {info[m][1]} of {info[m][0]} ({info[m][2]})" for m in MATS))
            print(f"         bytes in use {len(used)}; numbers needed to store the directions {mem:,} ({mem / 1124113:.3f} of the network's parameters)")
        for scale in (0.03,):
            g2 = np.random.default_rng(7)
            Wp = changed(W, V, used, g2, scale, True); rp, hp, lpa, lpb = score(Wp)
            g2 = np.random.default_rng(7)
            Wc = changed(W, V, used, g2, scale, False); rc, hc, _, _ = score(Wc)
            rel = np.sqrt(sum(np.linalg.norm(Wp[k] - W[k]) ** 2 for k in W) / sum(np.linalg.norm(W[k]) ** 2 for k in W))
            print(f"      3. random change obeying the rule (size {rel:.2f} of the network's): reduction pairs {rp:5.1f}, held-out {hp:5.1f}; largest change of an output {max(np.abs(lpa - l1a).max(), np.abs(lpb - l1b).max()):.2e}"
                  f" | the same entries changed without the projection: {rc:5.1f}, {hc:5.1f}")
