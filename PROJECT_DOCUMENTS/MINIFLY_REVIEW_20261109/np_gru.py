"""Independent numpy re-implementation, first written for the E29 audit (13 November 2026) and reused since: items, GRU forward (torch's convention), the E29 measures. No torch, no harness code."""
import hashlib, json, subprocess, sys
from random import Random
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_pt_tensors import tensors

REF = 'origin/codex/minifly-consolidated-20261023'
P = 'MOD97_LOCAL_RULE_20261106/exp45/'
EOS, BOS = 256, 257
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
SYM_OP = {v: k for k, v in SYM.items()}
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
LIFE = [("add", 97, b"r"), ("sub", 97, b"t"), ("max", 97, b"x"), ("min", 97, b"n")]
PAIRS = [(a, b) for a in SYM.values() for b in SYM.values()]
HELD_PAIRS = [("+", "-"), ("-", "+"), (">", "<"), ("<", ">")]
TAUGHT_PAIRS = [p for p in PAIRS if p not in HELD_PAIRS]

def git_bytes(path, ref=REF):
    r = subprocess.run(['git', 'show', f'{ref}:{path}'], capture_output=True, cwd=str(HERE))
    assert r.returncode == 0, path
    return r.stdout

# ---------------------------------------------------------------- items (re-derived from reading exp11_cell.py / exp16_integrated.py)
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

def triples(pair, seed, n, exclude=frozenset(), split_="taught"):
    rng = Random(seed * 31 + ord(pair[0]) * 7 + ord(pair[1])) if split_ == "taught" else Random(f"{split_}|{seed}|{pair[0]}{pair[1]}")
    out, seen = [], set(exclude)
    while len(out) < n:
        tr = (rng.randrange(97), rng.randrange(97), rng.randrange(97))
        if tr not in seen:
            seen.add(tr); out.append(tr)
    return out

def render_pair(o1, o2, a, b, c):
    i = apply(SYM_OP[o1], 97, a, b); f = apply(SYM_OP[o2], 97, i, c)
    prompt = list(b"c" + f"{a:02d}".encode() + o1.encode() + f"{b:02d}".encode() + o2.encode() + f"{c:02d}".encode() + b"=")
    return prompt, list(f"{i:02d}".encode()) + [ord(";")] + list(f"{f:02d}".encode()) + [EOS]

def comp_panels(seed):
    taught = {p: set(triples(p, seed, 2000)) for p in TAUGHT_PAIRS}
    all_taught = frozenset().union(*taught.values())
    comp = {f"{o1}{o2}": [render_pair(o1, o2, *tr) for tr in triples((o1, o2), seed, 600, exclude=all_taught, split_="held")] for o1, o2 in HELD_PAIRS}
    comp["within"] = [render_pair(*p, *tr) for p in TAUGHT_PAIRS for tr in triples(p, seed, 50, exclude=all_taught, split_="within")]
    comp["taught_sample"] = {p: sorted(taught[p])[:0] for p in TAUGHT_PAIRS}
    return comp, taught

def sha_items(items):
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()

def batch(items):
    rows = [[BOS] + p + t[:-1] for p, t in items]
    assert len({len(r) for r in rows}) == 1
    x = np.array(rows, dtype=np.int64)
    tgt = np.array([t for _, t in items], dtype=np.int64)          # the last len(t) inputs predict these
    return x, tgt

# ---------------------------------------------------------------- weights and forward
def load_weights(path, ref=REF):
    blob = git_bytes(path, ref)
    sha = hashlib.sha256(blob).hexdigest()
    t = tensors(blob)
    W = {}
    for name, (stype, shape, raw) in t.items():
        assert stype == 'FloatStorage', (name, stype)
        W[name.lstrip('/')] = np.frombuffer(raw, dtype='<f4').reshape(shape).copy()
    return W, sha

def sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos]); out[~pos] = e / (1.0 + e)
    return out

def forward(W, x, want_logits=True):
    """torch's GRU: r = s(Wir x + Whr h), z = s(Wiz x + Whz h), n = tanh(Win x + r*(Whn h)), h' = (1-z) n + z h; chunks ordered r, z, n."""
    u = W['book.weight'][x]                                              # (N, L, 8)
    z1 = np.tanh(u @ W['l1.weight'].T + W['l1.bias'])                    # (N, L, 512)
    gi = z1 @ W['rnn.weight_ih_l0'].T + W['rnn.bias_ih_l0']              # (N, L, 3U)
    Whh, bhh = W['rnn.weight_hh_l0'], W['rnn.bias_hh_l0']
    U = Whh.shape[1]
    N, L = x.shape
    h = np.zeros((N, U), dtype=np.float32)
    Z = np.empty((N, L, U), dtype=np.float32); R = np.empty_like(Z); Hs = np.empty_like(Z)
    for t in range(L):
        gh = h @ Whh.T + bhh
        r = sigmoid(gi[:, t, :U] + gh[:, :U])
        z = sigmoid(gi[:, t, U:2*U] + gh[:, U:2*U])
        n = np.tanh(gi[:, t, 2*U:] + r * gh[:, 2*U:])
        h = (1 - z) * n + z * h
        Z[:, t], R[:, t], Hs[:, t] = z, r, h
    logits = None
    if want_logits:
        z2 = np.tanh(z1 @ W['l2.weight'].T + W['l2.bias'] + Hs @ W['lm.weight'].T + W['lm.bias'])
        logits = z2 @ W['l3.weight'].T + W['l3.bias']
    return Z, R, Hs, logits

def exact_teacher_forced(logits, tgt):
    """Whole answer right under teacher forcing == whole answer right under greedy decoding."""
    n = tgt.shape[1]
    pred = logits[:, -n:, :].argmax(-1)
    return (pred == tgt).all(1)

# ---------------------------------------------------------------- the E29 measures, as the brief defines them
def roles_of(xrow):
    b = [int(v) for v in xrow]
    eq = b.index(ord('='))
    out = ['start', 'code']
    for v in b[2:eq]:
        out.append('digit' if 48 <= v <= 57 else 'operator' if v in set(b'+-<>') else 'letter' if 65 <= v <= 90 else '?')
    out.append('=')
    out += ['answer'] * (len(b) - eq - 1)
    return out

def measures(Z, x, zthr=0.9):
    roles = np.array([roles_of(r) for r in x])
    letter = roles == 'letter'; start = roles == 'start'; off = ~letter & ~start
    N, L, U = Z.shape
    res = {}
    zo = np.where(off[..., None], Z, 2.0).reshape(N * L, U)
    res['low'] = zo.min(0); arg = zo.argmin(0)
    res['rewrite_role'] = roles.reshape(-1)[arg]
    res['mean_z'] = (Z * (~start)[..., None]).sum((0, 1)) / (~start).sum()
    if letter.any():
        res['hold'] = ((Z > zthr) & letter[..., None]).sum((0, 1)) / letter.sum()
        prod = np.where(letter[..., None], Z.astype(np.float64), 1.0).prod(1)
        res['ret'] = prod.mean(0)
        res['holder'] = (res['hold'] >= 0.9) & (res['low'] <= 0.1)
        # per item: is the unit rewritten (z <= 0.1) somewhere off letters in this item?
        res['rewritten_item_share'] = (np.where(off[..., None], Z, 2.0).min(1) <= 0.1).mean(0)
    return res, roles
