"""V149 states: the native front-end and KC state of an untaught FastMemory before each symbol of a record.

The memory is minifly_cls_dev_20261017/fastmem.py with V148's settings. Nothing is ever taught. A record is opened
exactly as in V148's probe (begin control, end-of-empty-prompt control). Before each symbol the state is read exactly
as FastMemory.values() reads it (fastmem.py:194-205), and then the true byte is fed as a passive event:
  x   binary per-side top-k KC code of the front end advanced one step   (values(): f.advance(t + DELTA); encode)
  dx  observed KC activity after the compartment's adaptation             (values(): fly.rest(DELTA); observed_activity)
With no teaching all 16 compartments stay identical, so one compartment carries the state (checked in checks.py).
"""
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "minifly_cls_dev_20261017"))
import fastmem  # noqa: E402

EOS = fastmem.EOS
ARMS = ("PERM-obs", "PERM-x", "FE0-x", "PERM-last")


def new_memory(front="perm"):
    mem = fastmem.FastMemory(front=front)
    mem.bank = mem.bank[:1]
    return mem


def _read(view):
    """x and dx exactly as values() computes them, without the compartment readout."""
    f = view.fe.clone()
    f.advance(view.time + fastmem.DELTA)
    x = view.encode(f.read())[0]
    fly = view.bank[0].fly.clone()
    fly.rest(fastmem.DELTA)
    dx = fastmem.NATIVE.common.observed_activity(fly.m, x[None, :])[0]
    return x, dx


def _last_only(view, prev):
    """Binary code of a fresh FE of the same kind fed only the previous byte, read at the same time as _read."""
    f = view._new_front()
    if prev is not None:
        f.t = view.time
        f.feed(prev, view.time)
    f.advance(view.time + fastmem.DELTA)
    return view.encode(f.read())[0]


def record_states(mem, rec):
    """Dict arm -> array (len(rec) + 1, n_kc) of states before each symbol of rec (bytes, then EOS)."""
    view = mem.clone()
    view._control(True)
    view._control(False)
    xs, dxs, lasts = [], [], []
    for i in range(len(rec) + 1):
        x, dx = _read(view)
        xs.append(x)
        dxs.append(dx)
        lasts.append(_last_only(view, rec[i - 1] if i else None))
        if i < len(rec):
            view._event(rec[i])
    return np.array(xs), np.array(dxs), np.array(lasts)


def collect(recs):
    """Sparse state matrices for every arm over a list of records, plus the target symbols."""
    perm, fe0 = new_memory("perm"), new_memory("fe0")
    rows = {a: [] for a in ARMS}
    ys = []
    for r in recs:
        x, dx, last = record_states(perm, r)
        x0 = record_states(fe0, r)[0]
        rows["PERM-obs"].append(sparse.csr_matrix(dx))
        rows["PERM-x"].append(sparse.csr_matrix(x))
        rows["FE0-x"].append(sparse.csr_matrix(x0))
        rows["PERM-last"].append(sparse.csr_matrix(last))
        ys.extend(list(r) + [EOS])
    return {a: sparse.vstack(v).tocsr() for a, v in rows.items()}, np.array(ys)
