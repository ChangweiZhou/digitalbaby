# Run from the repository root:  python3 -I -B check_e51_brief.py [S|B] [updates in part E, default 8]      (numpy and the standard library; about one minute)
# It reads one half-width birth of seed 73129 through git and writes nothing. No network is trained; part E moves the readout matrix l3 alone.
# The fresh weights are numpy's draws at torch's scales.
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109")
from random import Random
import numpy as np
from np_gru import load_weights                      # used only to read the saved tensors

RULE, SEED, EOS, BOS = (sys.argv[1] if len(sys.argv) > 1 else "S"), 73129, 256, 257
N_UP = int(sys.argv[2]) if len(sys.argv) > 2 else 8        # updates in part E
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
OLD = np.concatenate([np.arange(g * 256, g * 256 + 128) for g in range(3)])          # the old cells' rows of the two GRU matrices
MATS = {"ih": ("rnn.weight_ih_l0", "rnn.bias_ih_l0"), "hh": ("rnn.weight_hh_l0", "rnn.bias_hh_l0"), "lm": ("lm.weight", "lm.bias"),
        "l2": ("l2.weight", "l2.bias"), "l3": ("l3.weight", "l3.bias")}

def render(task, a, b, k, rng):                      # exp11_cell.render
    op, m, code = task
    d = [rng.choice(list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")) for _ in range(k)]
    ans = {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]
    return list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")], list(f"{ans:02d}".encode()) + [EOS]

def split(task, frac, seed):                         # exp11_cell.split
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]).shuffle(pairs)
    return pairs[:round(frac * len(pairs))], pairs[round(frac * len(pairs)):]

def panel(task, pairs, seed, k, n):                  # exp11_cell.panel for one number of letters
    rng = Random(seed + 3); sub = rng.sample(pairs, n)
    return [render(task, a, b, k, rng) for a, b in sub]

def unrelated_batches(seed):                         # e43_task.batches
    pairs = [(a, b) for a in range(128, 192) for b in range(128, 192)]
    Random(f"e43|{seed}|split").shuffle(pairs); taught = pairs[:2048]; rng = Random(f"e43|{seed}|stream")
    def stream():
        while True:
            order = list(range(len(taught))); rng.shuffle(order)
            for i in order:
                a, b = taught[i]; k = rng.randint(0, 12)
                yield [224, a, b] + [rng.choice(list(range(192, 224))) for _ in range(k)] + [225], [a, b, EOS]
    st = stream()
    while True:
        yield [next(st) for _ in range(64)]

def sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))

def forward(W, x, in_use=None, zero_added=False):
    """arch26.forward in float64. in_use: the brief's added line (z = 1 for cells 128.. on items whose first byte is in use).
    zero_added: the repair of defect 1 (units 256.. of z2 held at zero on the same items). Returns outputs and input vectors."""
    U = W["rnn.weight_hh_l0"].shape[1]
    z1t = np.tanh(W["book.weight"] @ W["l1.weight"].T + W["l1.bias"]); git = z1t @ W["rnn.weight_ih_l0"].T + W["rnn.bias_ih_l0"]
    h = np.zeros((len(x), U)); key = in_use[x[:, 1]] if in_use is not None else np.zeros(len(x), bool)
    out = []; col = {m: [] for m in MATS}
    for t in range(x.shape[1]):
        z1 = z1t[x[:, t]]; gi = git[x[:, t]]; col["ih"].append(z1); col["hh"].append(h.copy())
        gh = h @ W["rnn.weight_hh_l0"].T + W["rnn.bias_hh_l0"]
        r = sig(gi[:, :U] + gh[:, :U]); z = sig(gi[:, U:2 * U] + gh[:, U:2 * U]); z[key, 128:] = 1.0
        h = (1 - z) * np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:]) + z * h
        if t >= x.shape[1] - 3:                      # the three answer positions
            z2 = np.tanh(z1 @ W["l2.weight"].T + W["l2.bias"] + h @ W["lm.weight"].T + W["lm.bias"])
            if zero_added: z2[key, 256:] = 0.0
            col["lm"].append(h.copy()); col["l2"].append(z1); col["l3"].append(z2); out.append(z2 @ W["l3.weight"].T + W["l3.bias"])
    return np.stack(out, 1), {m: np.concatenate(v) for m, v in col.items()}

def run(W, items, in_use=None, zero_added=False):
    """Teacher forcing, x = [start] + prompt + answer[:-1], items grouped by length. Outputs (items, 3, 257), the collected vectors, the targets."""
    out = np.empty((len(items), 3, 257)); col = {m: [] for m in MATS}; groups = {}; tgt = []
    for i, (p, a) in enumerate(items): groups.setdefault(len(p), []).append(i)
    for idx in groups.values():
        o, c = forward(W, np.array([[BOS] + items[i][0] + items[i][1][:-1] for i in idx]), in_use, zero_added); out[idx] = o
        for m in MATS: col[m].append(c[m])
        tgt += [items[i][1][j] for j in range(3) for i in idx]                          # in the order of the vectors collected for lm, l2 and l3
    return out, {m: np.concatenate(v) for m, v in col.items()}, np.array(tgt)

def one(X):
    return np.concatenate([X, np.ones((len(X), 1))], 1)

def away(D, V):
    return D - (D @ V) @ V.T

sd0 = {k: v.astype(np.float64) for k, v in load_weights(f"MOD97_LOCAL_RULE_20261106/exp45/e41/e41_{RULE}_h256_f50_{SEED}.pt")[0].items()}
g = np.random.default_rng(1); u = lambda shape, bound: g.uniform(-bound, bound, shape)
sd = {"book.weight": sd0["book.weight"].copy(), "l1.weight": u((512, 8), 8 ** -.5), "l1.bias": u(512, 8 ** -.5),
      "rnn.weight_ih_l0": u((768, 512), 1 / 16), "rnn.weight_hh_l0": u((768, 256), 1 / 16), "rnn.bias_ih_l0": u(768, 1 / 16), "rnn.bias_hh_l0": u(768, 1 / 16),
      "lm.weight": u((512, 256), 1 / 16), "lm.bias": u(512, 1 / 16), "l2.weight": u((512, 512), 512 ** -.5), "l2.bias": u(512, 512 ** -.5),
      "l3.weight": u((257, 512), 512 ** -.5), "l3.bias": u(257, 512 ** -.5)}
sd["l1.weight"][:256] = sd0["l1.weight"]; sd["l1.bias"][:256] = sd0["l1.bias"]                     # the brief's fragment, line for line
for gate in range(3):
    new, old = slice(gate * 256, gate * 256 + 128), slice(gate * 128, (gate + 1) * 128)
    sd["rnn.weight_ih_l0"][new, :256] = sd0["rnn.weight_ih_l0"][old]; sd["rnn.weight_ih_l0"][new, 256:] = 0.0
    sd["rnn.weight_hh_l0"][new, :128] = sd0["rnn.weight_hh_l0"][old]
    sd["rnn.bias_ih_l0"][new] = sd0["rnn.bias_ih_l0"][old]; sd["rnn.bias_hh_l0"][new] = sd0["rnn.bias_hh_l0"][old]
sd["lm.weight"][:256, :128] = sd0["lm.weight"]; sd["lm.bias"][:256] = sd0["lm.bias"]
sd["l2.weight"][:256, :256] = sd0["l2.weight"]; sd["l2.weight"][:256, 256:] = 0.0; sd["l2.bias"][:256] = sd0["l2.bias"]
sd["l3.weight"][:, :256] = sd0["l3.weight"]; sd["l3.weight"][:, 256:] = 0.0; sd["l3.bias"][:] = sd0["l3.bias"]
op = {k: v.copy() for k, v in sd.items()}; op["rnn.bias_ih_l0"][384:512] = 10.0; op["rnn.bias_hh_l0"][384:512] = 0.0   # the start of `open`

rng = Random(f"e51|{SEED}"); taught_items = []       # the 2,080 items, k = 0 first
by_k = {k: [] for k in range(13)}
for t in DEV:
    taught = split(t, .5, SEED)[0]
    for k in range(13): by_k[k] += [render(t, a, b, k, rng) for a, b in rng.sample(taught, 40)]
for k in range(13): taught_items += by_k[k]
in_use = np.zeros(258, bool); in_use[[BOS] + [b for p, a in taught_items for b in p + a[:-1]]] = True
held12 = [it for t in DEV for it in panel(t, split(t, .5, SEED)[1], SEED, 12, 300)]                 # E43's score 2 (and E41's panel A)
plain = Random(f"audit|{SEED}").sample([render(t, a, b, 0, None) for t in DEV for a in range(89) for b in range(89)], 2000)
held20 = [it for t in DEV for it in panel(t, split(t, .5, SEED)[1], SEED + 11, 20, 300)]
SETS = {"the 2,080 items": taught_items, "held-out pairs, 12 letters": held12, "2,000 items, no letters": plain, "held-out pairs, 20 letters": held20}
ref = {n: run(sd, it, in_use) for n, it in SETS.items()}
right = lambda o, it: (o.argmax(-1) == np.array([a for _, a in it])).all(1)

print(f"A. rule {RULE}, seed {SEED}; bytes in use {int(in_use.sum())}; half-width birth right on the held-out panel at 12 letters: {int(right(run(sd0, held12)[0], held12).sum())} of 1200")
for n, it in SETS.items():
    b = run(sd0, it)[0]; nokey = run(sd, it)[0]; o = run(op, it)[0]
    print(f"   {n}: keyed start against the birth {np.abs(ref[n][0] - b).max():.1e}; `open` start {np.abs(o - b).max():.1e}; keyed weights without the added line "
          f"{np.abs(nokey - b).max():.2f}, answers changed {int((nokey.argmax(-1) != b.argmax(-1)).any(1).sum())} of {len(it)}")

V, spec = {}, {}
for m in MATS:                                       # the brief's directions: eigenvalue above one millionth of the largest
    X = one(ref["the 2,080 items"][1][m]); w, E = np.linalg.eigh(X.T @ X / len(X)); w, E = w[::-1], E[:, ::-1]
    k = int((w > 1e-6 * w[0]).sum()); V[m], spec[m] = E[:, :k], (w, E)
    print(f"B. {m}: {k} directions of {X.shape[1]}; rank of the collected vectors {np.linalg.matrix_rank(X)}; eigenvalues: largest {w[0]:.2e}, last kept {w[k - 1]:.2e}, "
          f"first dropped {w[k] if k < len(w) else 0:.2e}; kept at 1e-8, 1e-10, 1e-12 of the largest: {', '.join(str(int((w > t * w[0]).sum())) for t in (1e-8, 1e-10, 1e-12))}")

for n in SETS:                                       # C. what the first family has outside the protected directions
    r3 = away(one(ref[n][1]["l3"]), V["l3"]); ri = away(one(ref[n][1]["ih"]), V["ih"]); rh = away(one(ref[n][1]["hh"]), V["hh"])
    print(f"C. {n}: part outside the directions, largest 2-norm / 1-norm: l3 {np.linalg.norm(r3, axis=1).max():.3f} / {np.abs(r3).sum(1).max():.2f}; "
          f"input map {np.linalg.norm(ri, axis=1).max():.4f} / {np.abs(ri).sum(1).max():.3f}; recurrent matrix {np.linalg.norm(rh, axis=1).max():.0e}")
codes = one(np.tanh(sd["book.weight"] @ sd["l1.weight"].T + sd["l1.bias"])); rc = away(codes, V["ih"]); new = np.arange(128, 226)
half = one(np.tanh(sd0["book.weight"] @ sd0["l1.weight"].T + sd0["l1.bias"])); wq, Eq = np.linalg.eigh(half[in_use].T @ half[in_use]); rq = away(half, Eq[:, wq > 1e-12 * wq.max()])
print(f"C. a byte of the unrelated task: share of its code's squared norm inside the directions in use: embedded network median {1 - np.median((np.linalg.norm(rc[new], axis=1) / np.linalg.norm(codes[new], axis=1)) ** 2):.4f}, "
      f"half-width birth alone {1 - np.median((np.linalg.norm(rq[new], axis=1) / np.linalg.norm(half[new], axis=1)) ** 2):.4f}; outside part in the embedded network: 2-norm median {np.median(np.linalg.norm(rc[new], axis=1)):.2f}, "
      f"1-norm median {np.median(np.abs(rc[new]).sum(1)):.1f} (a byte in use: at most {np.abs(rc[in_use]).sum(1).max():.2f}; a whole code: {np.median(np.abs(codes[new]).sum(1)):.0f})")

def changed(W, scale, only, Vs, zero_added=False):   # D. a random change of every entry that the table lets change
    gen = np.random.default_rng(100); Wn = {k: v.copy() for k, v in W.items()}
    if "book" in only: Wn["book.weight"][~in_use] += gen.normal(0, scale, (int((~in_use).sum()), 8))
    for m in only:
        if m == "book": continue
        D = gen.normal(0, scale, (W[MATS[m][0]].shape[0], W[MATS[m][0]].shape[1] + 1))
        if not zero_added:
            rows = OLD if m in ("ih", "hh") else np.arange(len(D)); D[rows] = away(D[rows], Vs[m])
        elif m == "l3": D[:, :256] = 0.0; D[:, -1] = 0.0                               # the repair: l3[:, :256] and l3.bias not changed
        else:
            rows = OLD if m in ("ih", "hh") else np.arange(256); D[rows] = away(D[rows], Vs[m])   # rows 256.. of lm and l2 free
        Wn[MATS[m][0]] += D[:, :-1]; Wn[MATS[m][1]] += D[:, -1]
    return Wn
for only in (("l3",), ("ih",), ("hh", "lm", "l2", "book")):
    for scale in (0.03, 4.0):
        Wn = changed(sd, scale, only, V); line = []
        for n, it in SETS.items():
            o = run(Wn, it, in_use)[0]; line.append(f"{n} {np.abs(o - ref[n][0]).max():.1e} ({int((o.argmax(-1) != ref[n][0].argmax(-1)).any(1).sum())} answers)")
        print(f"D. change of {'+'.join(only)}, entries N(0, {scale}^2) and then the rule: largest change of an output: " + "; ".join(line))

bt = unrelated_batches(SEED); feats = []             # E. checks 3(a) and 3(b) for l3 alone: AdamW's arithmetic (rate 0.001, betas 0.9 and 0.98, no decay)
for _ in range(N_UP):
    o, c, y = run(sd, next(bt), in_use); feats.append((one(c["l3"]), y))
W3 = np.concatenate([sd["l3.weight"], sd["l3.bias"][:, None]], 1); w3, E3 = spec["l3"]
Zh, Zp = one(ref["held-out pairs, 12 letters"][1]["l3"]), one(ref["2,000 items, no letters"][1]["l3"])
srt = np.sort(ref["2,000 items, no letters"][0], -1); margin = (srt[..., -1] - srt[..., -2]).min(1)
print(f"E. first family, 2,000 items without letters: share whose smallest gap between the two largest outputs is below 0.1, 0.3, 1.0: {(margin < .1).mean():.3f}, {(margin < .3).mean():.3f}, {(margin < 1).mean():.3f}; "
      f"the unrelated task's first batch at l3: 2-norm of a vector {np.median(np.linalg.norm(feats[0][0], axis=1)):.1f}, of its part outside the brief's directions {np.median(np.linalg.norm(away(feats[0][0], V['l3']), axis=1)):.2f}, "
      f"outside those kept at 1e-8 {np.median(np.linalg.norm(away(feats[0][0], E3[:, :int((w3 > 1e-8 * w3[0]).sum())]), axis=1)):.2f}")
for label, Vp in (("the brief's rule (1e-6)", V["l3"]), ("threshold 1e-8", E3[:, :int((w3 > 1e-8 * w3[0]).sum())]), ("projections off", None)):
    W = W3.copy(); mo = np.zeros_like(W); ve = np.zeros_like(W)
    for step, (X, y) in enumerate(feats, 1):
        lg = X @ W.T; lg -= lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True); p[np.arange(len(y)), y] -= 1; G = p.T @ X / len(y)
        if Vp is not None: G = away(G, Vp)
        mo = .9 * mo + .1 * G; ve = .98 * ve + .02 * G * G; S = -1e-3 * (mo / (1 - .9 ** step)) / (np.sqrt(ve / (1 - .98 ** step)) + 1e-8)
        W = W + (away(S, Vp) if Vp is not None else S)
        if step in (8, N_UP):
            print(f"E. {step} updates of l3 alone on the unrelated stream, {label}: largest change of an output at an answer position: held-out panel "
                  f"{np.abs(Zh @ (W - W3).T).max():.1e}, 2,000 items without letters {np.abs(Zp @ (W - W3).T).max():.1e}")

D = np.zeros((768, 513)); gen = np.random.default_rng(5)                               # F. an allowed change of the old cells' rows of the input map
for b in (224, 225) + tuple(range(128, 136)): D[OLD] += np.outer(gen.choice([-1.0, 1.0], len(OLD)), rc[b] / (rc[b] @ rc[b]))
assert np.abs(D[OLD] @ V["ih"]).max() < 1e-9
Wn = {k: v.copy() for k, v in sd.items()}; Wn["rnn.weight_ih_l0"] += D[:, :-1]; Wn["rnn.bias_ih_l0"] += D[:, -1]
print(f"F. old cells' rows changed inside what the rule allows (largest entry {np.abs(D).max():.2f}): an old cell's gate input moves by {np.abs(codes[[224, 225]] @ D[OLD].T).mean():.2f} on the new task's code and query bytes "
      f"and by at most {np.abs(codes[in_use] @ D[OLD].T).max():.3f} on a byte in use; largest change of a first-family output {max(np.abs(run(Wn, it, in_use)[0] - ref[n][0]).max() for n, it in SETS.items()):.3f}")

ref2 = {n: run(sd, it, in_use, True) for n, it in SETS.items()}; V2 = {}                # G. the repair of defect 1
for m in ("ih", "hh", "lm", "l2"):
    X = one(ref2["the 2,080 items"][1][m]); w, E = np.linalg.eigh(X.T @ X / len(X)); V2[m] = E[:, w > 1e-12 * w.max()]
Wn = changed(sd, 4.0, ("book", "ih", "hh", "lm", "l2", "l3"), V2, True)
print(f"G. repair (added units of z2 keyed, all directions kept: {', '.join(f'{m} {V2[m].shape[1]}' for m in V2)}; l3[:, :256] and l3.bias fixed): start against the birth "
      f"{max(np.abs(ref2[n][0] - run(sd0, it)[0]).max() for n, it in SETS.items()):.1e}; after a change with entries N(0, 4^2), largest change of an output "
      f"{max(np.abs(run(Wn, it, in_use, True)[0] - ref2[n][0]).max() for n, it in SETS.items()):.1e}")
