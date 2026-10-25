"""MiniFly side of V148: the unchanged native FastMemory (minifly_cls_dev_20261017/fastmem.py, default settings).

Training: one record per line, FastMemory.train(b"", line): every byte is taught from the context of the preceding
bytes of the line, then EOS. Probing runs on a clone: the record is opened exactly as in training (begin control,
end-of-empty-prompt control), then for each position the clone reads its own 257 native logits, the probability of
the true next symbol is recorded, and the true byte is fed as a passive (untaught) event. The original memory is not
changed (checked by state hash in checks.py).
"""
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "minifly_cls_dev_20261017"))
import fastmem  # noqa: E402

EOS = fastmem.EOS


def new_memory():
    return fastmem.FastMemory()          # defaults: perm front end, delta teach, fresh adaptation, homeostasis


def state_hash(mem):
    h = hashlib.sha256()
    for s in mem.bank:
        m = s.fly.m
        for a in (m.fast, m.slow, m.adapt):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(repr((float(m.elapsed), s.brain_t)).encode())
    h.update(mem.theta.tobytes())
    h.update(mem.freq.tobytes())
    h.update(repr((mem.time, mem.records)).encode())
    return h.hexdigest()


def line_logits(mem, rec):
    """Native 257-way logits before each symbol of rec (bytes then EOS), teacher-forced, on a clone."""
    view = mem.clone()
    view._control(True)
    view._control(False)                  # empty prompt, exactly as train(b"", line)
    out = []
    for b in rec:
        out.append(fastmem.logits(view.values()))
        view._event(b)                    # passive: no teaching
    out.append(fastmem.logits(view.values()))
    return np.array(out)


def collect(mem, recs):
    """Logits and targets for a list of records (for temperature calibration and scoring)."""
    L, Y = [], []
    for r in recs:
        L.append(line_logits(mem, r))
        Y.extend(list(r) + [EOS])
    return np.vstack(L), np.array(Y)


def bits_at(L, Y, T):
    Z = L / T
    Z = Z - Z.max(1, keepdims=True)
    logp = Z[np.arange(len(Y)), Y] - np.log(np.exp(Z).sum(1))
    return float(-logp.sum() / math.log(2)), len(Y)


TGRID = np.geomspace(0.02, 50.0, 61)


def calibrate(L, Y):
    vals = [bits_at(L, Y, t)[0] for t in TGRID]
    i = int(np.argmin(vals))
    return float(TGRID[i]), vals[i] / len(Y)


def accuracy(L, Y):
    return float(np.mean(L.argmax(1) == Y))


def generate(mem, prefix, cap=64):
    """Greedy own continuation after a teacher-forced line prefix; stops at own EOS or cap bytes."""
    view = mem.clone()
    view._control(True)
    view._control(False)
    for b in prefix:
        view._event(b)
    out = bytearray()
    for _ in range(cap):
        tok = int(fastmem.logits(view.values()).argmax())
        if tok == EOS:
            return bytes(out), "EOS"
        out.append(tok)
        view._event(tok)
    return bytes(out), "CAP"
