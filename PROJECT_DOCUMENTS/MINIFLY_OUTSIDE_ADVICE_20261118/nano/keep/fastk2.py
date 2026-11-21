#!/usr/bin/env python3
"""The active-cell kernel (kern2.c) for cells2.py with clip=p (PLAN_8.md); `fast=2`. The numpy program remains the reference (check_fast.py)."""
import ctypes, os, subprocess
import numpy as np
from fastk import FLAGS, I64, F64, PD, PI, P32, pd
HERE = os.path.dirname(os.path.abspath(__file__)); SO = os.path.join(HERE, "kern2.so"); SRC = os.path.join(HERE, "kern2.c")
class SNet(ctypes.Structure):
    _fields_ = [(k, I64) for k in "H NL S D learn_steps active_sum changing_sum".split()] + [(k, F64) for k in "eta1 eta2 lam clip kap cap".split()] + \
               [(k, PD) for k in "W1 b1 W2 b2 R".split()] + [("hist", PI)]
_lib = None
def lib():
    global _lib
    if _lib is None:
        if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
            tmp = f"{SO}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", *FLAGS, "-o", tmp, SRC, "-lm"]); os.replace(tmp, SO)
        _lib = ctypes.CDLL(SO); _lib.sparse_steps.restype = I64; _lib.sparse_steps.argtypes = [ctypes.POINTER(SNet), I64, P32, P32]
    return _lib
def run(s):
    xs = np.asarray(s.bx, np.int32); ys = np.asarray(s.by, np.int32); s.bx = []; s.by = []
    if not len(xs): return
    assert s.clip and s.M is None and not s.dale and not s.H2 and s.live is None and not s.two and not s.stiff1 and not s.rewire, "fast=2 is for the plain system with clip"
    n = SNet(); NL = s.W1.shape[1]; S = s.W2.shape[0]; hist = np.asarray(s.hist, np.int64)
    n.H, n.NL, n.S, n.D = s.H, NL, S, NL // S; n.eta1, n.eta2, n.lam, n.clip, n.kap, n.cap = s.eta1, s.eta2, s.lam, s.clip, s.kap, min(s.cap, 1e300)
    n.W1, n.b1, n.W2, n.b2, n.R = pd(s.W1), pd(s.b1), pd(s.W2), pd(s.b2), pd(s.R); n.hist = hist.ctypes.data_as(PI)
    lib().sparse_steps(ctypes.byref(n), len(xs), xs.ctypes.data_as(P32), ys.ctypes.data_as(P32)); s.hist = [int(v) for v in hist]
    s.stat[0] += int(n.learn_steps); s.stat[1] += int(n.active_sum); s.stat[2] += int(n.changing_sum)
if __name__ == "__main__": lib(); print("built", SO)
