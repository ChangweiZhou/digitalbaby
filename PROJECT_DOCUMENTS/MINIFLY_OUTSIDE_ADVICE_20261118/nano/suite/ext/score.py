#!/usr/bin/env python3
"""The measure of the test at the level of nanoGPT (see MEASURE.md). Plain numpy; meant to be imported by every line under nano/.

    import score
    x, cut, S = score.load()            # the whole text as byte indices 0..64, the cut (1,003,854), 65
    stream, last = x[:cut], x[cut:]      # the stream that may be read; the last tenth, never read
    score.score(P)                       # P: (111540, 65), row i = probabilities for the byte last[i]
    score.score_fn(f, window=64)         # f(ctx) -> (m, 65) for ctx of shape (m, window), oldest byte first

Byte indices: the 65 byte values of the text in ascending order (nanoGPT's own table), index 0 = newline.
Every one of the 111,540 bytes of the last tenth is scored. The context of a byte is the bytes before it in the
whole text, so the first bytes of the last tenth are predicted from the end of the stream.
Nothing is cut off or floored: a probability of zero for the byte that arrives gives an infinite loss."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = os.path.join(HERE, "..", "..", "data", "tinyshakespeare.txt")
S = 65

def load(path=TEXT):
    raw = np.frombuffer(open(path, "rb").read(), np.uint8)
    values = np.unique(raw)
    assert len(raw) == 1115394 and len(values) == S, (len(raw), len(values))
    table = np.full(256, -1, np.int64); table[values] = np.arange(S)
    x = table[raw]; cut = int(0.9 * len(x))               # nanoGPT's prepare.py: data[:int(n*0.9)]
    assert cut == 1003854
    return x, cut, S

def byte_values(path=TEXT):
    return np.unique(np.frombuffer(open(path, "rb").read(), np.uint8))

def score(P, y=None, tol=1e-4):
    """P: (n, 65) probabilities aligned with y (default: the last tenth). Returns the stream loss and the share named right."""
    if y is None:
        x, cut, _ = load(); y = x[cut:]
    P = np.asarray(P, np.float64); y = np.asarray(y)
    assert P.shape == (len(y), S), P.shape
    assert np.all(P >= 0) and np.all(np.abs(P.sum(1) - 1) < tol), "rows must be probabilities that sum to one"
    with np.errstate(divide="ignore"):
        nats = float(-np.log(P[np.arange(len(y)), y]).mean())
    right = float((P.argmax(1) == y).mean())               # ties go to the lowest byte index
    return dict(n=int(len(y)), nats=nats, bits=float(nats / np.log(2)), right=100 * right)

def contexts(window, lo=None, hi=None):
    """The window of bytes before each byte lo..hi-1 of the whole text (default: the last tenth), oldest first, and those bytes."""
    x, cut, _ = load(); lo = cut if lo is None else lo; hi = len(x) if hi is None else hi
    assert lo >= window
    idx = np.arange(lo, hi)[:, None] + np.arange(-window, 0)[None, :]
    return x[idx], x[lo:hi]

def score_fn(f, window, batch=2048):
    """f(ctx) -> (m, 65) probabilities of the next byte, ctx: (m, window) byte indices, oldest first.
    Each byte of the last tenth is predicted from exactly the `window` bytes before it."""
    ctx, y = contexts(window)
    P = np.concatenate([np.asarray(f(ctx[i:i + batch]), np.float64) for i in range(0, len(y), batch)])
    return score(P, y)

if __name__ == "__main__":                                 # self-check: the uniform answer, and the stream's byte frequencies
    x, cut, _ = load(); n = len(x) - cut
    u = score(np.full((n, S), 1 / S)); assert abs(u["nats"] - np.log(S)) < 1e-12
    f = np.bincount(x[:cut], minlength=S) / cut
    print("uniform:", u); print("frequencies of the stream:", score(np.tile(f, (n, 1))))
    print("as a callback:", score_fn(lambda c: np.tile(f, (len(c), 1)), window=8))
