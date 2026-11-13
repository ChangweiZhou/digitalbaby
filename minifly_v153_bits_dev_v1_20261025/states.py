"""V153 states: native front-end/KC state of an untaught FastMemory before each symbol of a record (V149's reading,
fastmem.py values() path), for the four front-end cells of the 2x2 factorial, plus FE0's observed KC activity.

Cells (scrambling = the fixed 88-channel permutation applied at every byte; per-byte fade d):
  P95  scrambling on,  d = 0.95     V148's FE-PERM (fastmem.FEPerm default)
  P81  scrambling on,  d = 0.807    FEPerm with FE0's time constant
  N95  scrambling off, d = 0.95     BC.FE0 with FE-PERM's time constant
  N81  scrambling off, d = 0.807    native FE0 (BC.FE0, tau = 10 s)
Nothing is taught; homeostatic thresholds stay zero; one compartment carries the state (all 16 are identical).
"""
import math
import sys
import types
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "minifly_cls_dev_20261017"))
import fastmem  # noqa: E402

BC = fastmem.BC
EOS = fastmem.EOS
TAU_PERM = fastmem.FEPerm().tau                       # fade 0.95 per byte step
TAU_FE0 = BC.FE_PARAMS["FE0"]["tau_p"]                # 10 s: fade 0.807 per byte step
CELLS = ("P95", "P81", "N95", "N81")


def make_front(cell):
    if cell == "P95":
        return fastmem.FEPerm()
    if cell == "P81":
        return fastmem.FEPerm(step_decay=math.exp(-fastmem.DELTA / TAU_FE0))
    if cell == "N81":
        return BC.FE0()
    if cell == "N95":
        fe = BC.FE0()
        fe.tau = TAU_PERM
        return fe
    raise ValueError(cell)


def new_memory(cell, bank=1, homeo=True):
    front = "perm" if cell.startswith("P") else "fe0"
    mem = fastmem.FastMemory(front=front, homeo=homeo)
    if cell in ("P81", "N95"):
        def _new_front(self, _cell=cell):
            fe = make_front(_cell)
            fe.t = self.time
            return fe
        mem._new_front = types.MethodType(_new_front, mem)
    mem.bank = mem.bank[:bank]
    return mem


def _read(view):
    f = view.fe.clone()
    f.advance(view.time + fastmem.DELTA)
    x = view.encode(f.read())[0]
    fly = view.bank[0].fly.clone()
    fly.rest(fastmem.DELTA)
    dx = fastmem.NATIVE.common.observed_activity(fly.m, x[None, :])[0]
    return x, dx


def record_states(mem, rec):
    view = mem.clone()
    view._control(True)
    view._control(False)
    xs, dxs = [], []
    for i in range(len(rec) + 1):
        x, dx = _read(view)
        xs.append(x)
        dxs.append(dx)
        if i < len(rec):
            view._event(rec[i])
    return np.array(xs), np.array(dxs)


def collect(recs, cells=CELLS, obs_cell="N81"):
    """Sparse binary codes for each requested cell, the observed activity of obs_cell, and the target symbols."""
    mems = {c: new_memory(c) for c in cells}
    rows = {c: [] for c in cells}
    obs = []
    ys = []
    for r in recs:
        for c in cells:
            x, dx = record_states(mems[c], r)
            rows[c].append(sparse.csr_matrix(x))
            if c == obs_cell:
                obs.append(sparse.csr_matrix(dx))
        ys.extend(list(r) + [EOS])
    out = {c: sparse.vstack(v).tocsr() for c, v in rows.items()}
    if obs:
        out[obs_cell + "-obs"] = sparse.vstack(obs).tocsr()
    return out, np.array(ys)
