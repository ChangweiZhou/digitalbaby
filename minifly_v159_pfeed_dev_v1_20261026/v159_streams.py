"""V159 streams (design section 2): V155's same-future fixture (X1), the class-transfer task (X2), front-end traces.
Module names carry a v159_ prefix: MiniFly's native code imports generic names (it has its own `fixture`).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V155 = HERE.parent / "minifly_v155_keys_dev_v1_20261026"
sys.path.append(str(V155))
import cache as v155_cache  # noqa: E402  (front end P81, untaught: V153 states; record_traces)
import codes  # noqa: E402,F401  (V155: binary top-k codes from traces)

CACHE = HERE / "cache"
SET = json.loads((HERE / "SETTINGS.json").read_text())
EOS = 256
S1, S2, F1, F2 = b"abcdefgh", b"ijklmnop", b"qrstuvwx", b"yzABCDEF"
NEUTRAL, G = ord("-"), (ord("!"), ord("?"))


def samefuture_stream(seed, n_lines=300, length=40):
    """V155 technical.py's fixture generator, unchanged."""
    g = np.random.default_rng(seed)
    out_lines = []
    for _ in range(n_lines):
        cur = (S1 + S2)[g.integers(16)]
        line = [cur]
        while len(line) < length:
            if cur in S1:
                cur = F1[g.integers(8)]
            elif cur in S2:
                cur = F2[g.integers(8)]
            else:
                cur = (S1 + S2)[g.integers(16)]
            line.append(cur)
        out_lines.append(bytes(line))
    return out_lines


def transfer_stream(seed, n_lines, test):
    """Class-transfer lines: segments 's f' (f from F of s's class) or 's - g' (g = '!' class 1, '?' class 2).
    Type 2 only for seen members (index < 4) unless test=True, with probability p_type2 when allowed."""
    g = np.random.default_rng(seed)
    S, F = (S1, S2), (F1, F2)
    lines = []
    for _ in range(n_lines):
        segs = []
        for _ in range(SET["x2"]["segments"]):
            c, m = int(g.integers(2)), int(g.integers(8))
            s = S[c][m]
            if (m < 4 or test) and g.random() < SET["x2"]["p_type2"]:
                segs.append(bytes([s, NEUTRAL, G[c]]))
            else:
                segs.append(bytes([s, F[c][g.integers(8)]]))
        lines.append(b" ".join(segs))
    return lines


def byte_class(b):
    return 1 if b in S1 else 2 if b in S2 else 0


def traces(name, recs):
    """Front-end traces before every symbol (cached): P (positions x 88), targets y, line lengths n."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.npz"
    if path.exists():
        z = np.load(path)
        return z["P"], z["y"], z["n"]
    mem = v155_cache.states.new_memory(v155_cache.CELL)
    P, y = [], []
    for r in recs:
        ps, _ = v155_cache.record_traces(mem, r)
        P += ps
        y += list(r) + [EOS]
    P, y, n = np.array(P), np.array(y), np.array([len(r) + 1 for r in recs])
    np.savez(path, P=P, y=y, n=n)
    return P, y, n


def previous_bytes(y, n, k):
    """For every position, the symbol k positions earlier in the same line (-1 if none)."""
    out = np.full(len(y), -1)
    e = np.cumsum(np.concatenate([[0], n]))
    for lo, hi in zip(e[:-1], e[1:]):
        out[lo + k:hi] = y[lo:hi - k]
    return out


def transfer_set(seed):
    """Train / calibration / test streams of X2 for one seed, with their traces."""
    x2 = SET["x2"]
    out = {}
    for part, n_lines, off, test in (("train", x2["train_lines"], 0, False), ("calib", x2["calib_lines"], 100, False),
                                     ("test", x2["test_lines"], 200, True)):
        recs = transfer_stream(seed + off, n_lines, test)
        P, y, n = traces(f"x2_{seed}_{part}", recs)
        out[part] = {"P": P, "y": y, "n": n}
    return out
