"""The lived network's forward in numpy (arch26.forward, with e48_gated.forward's added terms), and the harness's panels.
The task lines (split, render, panel) are copied from exp11_cell.py; the forward follows arch26.py and e48_gated.py line by line.
Standard library and numpy only. Nothing is trained."""
from random import Random
import numpy as np

EOS, BOS = 256, 257
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
LIFE = [("add", 97, b"r"), ("sub", 97, b"t"), ("max", 97, b"x"), ("min", 97, b"n")]
F3 = [("add", 73, b"s"), ("sub", 73, b"u"), ("max", 73, b"y"), ("min", 73, b"z")]
F4 = [("add", 83, b"e"), ("sub", 83, b"f"), ("max", 83, b"g"), ("min", 83, b"h")]
CAP = 6


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


def panel(task, pairs, seed, ks=(0, 8, 12, 20), n=400):
    rng = Random(seed + 3)
    sub = rng.sample(pairs, n)
    return {k: [render(task, a, b, k, rng) for a, b in sub] for k in ks}


def earlier_panels(seed):
    """exp16_integrated.panels with E16_ARCH=bank2: k = 0, 8, 12, 20, 40 for the life's families; k = 12, n = 300 for the first family."""
    P = {f"{t[0]}{t[1]}": panel(t, split(t, .25, seed)[1], seed, ks=(0, 8, 12, 20, 40))[12] for t in LIFE + F3}
    P.update({f"{t[0]}{t[1]}": panel(t, split(t, .5, seed)[1], seed, ks=(12,), n=300)[12] for t in DEV})
    return P


def needs_reduction(op, a, b):
    return a + b >= 83 if op == "add" else a - b < 0


def reduction_panel(t, frac, seed):
    return [render(t, a, b, 0, Random(0)) for a, b in split(t, frac, seed)[1] if needs_reduction(t[0], a, b)]


sig = lambda v: 1.0 / (1.0 + np.exp(-v))


class Net:
    def __init__(self, sd, term=None, dtype=np.float32):
        self.w = {k: np.asarray(v, dtype=dtype) for k, v in sd.items()}
        self.term = None if term is None else {k: np.asarray(v, dtype=dtype) for k, v in term.items()}
        self.dtype = dtype

    def scaled(self, s):
        """Every tensor outside the byte table multiplied by s."""
        n = Net({k: (v if k == "book.weight" else v * self.dtype(s)) for k, v in self.w.items()}, self.term, self.dtype)
        return n

    def embed(self, x, gate):
        w = self.w
        z1 = np.tanh(w["book.weight"][x] @ w["l1.weight"].T + w["l1.bias"])
        gi = z1 @ w["rnn.weight_ih_l0"].T + w["rnn.bias_ih_l0"]
        if gate:
            gi = gi + (z1 @ self.term["down_i"].T) @ self.term["up_i"].T
        return z1, gi

    def step(self, h, z1, gi, gate):
        w = self.w
        gh = h @ w["rnn.weight_hh_l0"].T + w["rnn.bias_hh_l0"]
        if gate:
            gh = gh + (h @ self.term["down_h"].T) @ self.term["up_h"].T
        n = h.shape[1]
        r = sig(gi[:, :n] + gh[:, :n]); z = sig(gi[:, n:2 * n] + gh[:, n:2 * n]); nn_ = np.tanh(gi[:, 2 * n:] + r * gh[:, 2 * n:])
        h = (1 - z) * nn_ + z * h
        z2 = np.tanh(z1 @ w["l2.weight"].T + w["l2.bias"] + h @ w["lm.weight"].T + w["lm.bias"])
        return h, z2 @ w["l3.weight"].T + w["l3.bias"]

    def respond(self, prompts, gate):
        out = [None] * len(prompts)
        groups = {}
        for i, p in enumerate(prompts):
            groups.setdefault(len(p), []).append(i)
        for L, idx in groups.items():
            for j in range(0, len(idx), 1024):
                ids = idx[j:j + 1024]
                x = np.array([[BOS, *prompts[i]] for i in ids])
                z1, gi = self.embed(x, gate)
                h = np.zeros((len(ids), 256), dtype=self.dtype)
                for t in range(x.shape[1]):
                    h, lg = self.step(h, z1[:, t], gi[:, t], gate)
                gen = [[] for _ in ids]; done = [False] * len(ids)
                for _ in range(CAP):
                    yy = lg.argmax(1)
                    for q in range(len(ids)):
                        if not done[q]:
                            gen[q].append(int(yy[q]))
                            if int(yy[q]) == EOS:
                                done[q] = True
                    if all(done):
                        break
                    z1n, gin = self.embed(yy[:, None], gate)
                    h, lg = self.step(h, z1n[:, 0], gin[:, 0], gate)
                for q, i in enumerate(ids):
                    out[i] = gen[q]
        return out

    def score(self, items, gate=False):
        outs = self.respond([p for p, _ in items], gate)
        return sum(o == t for o, (_, t) in zip(outs, items)), len(items)
