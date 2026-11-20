#!/usr/bin/env python3
"""nano/kinds: a copy of loop_small_case/fastk3.py at 260c5d73, with the fields of the two additions.
The dense kernel with lazy fading (kern3.c) for the plain system of cells2.py; `fast=3`. The numpy program remains the reference (check_fast.py)."""
import ctypes, os, subprocess
import numpy as np
from fastk import FLAGS, I64, F64, PD, PI, P32, pd
HERE = os.path.dirname(os.path.abspath(__file__)); SO = os.path.join(HERE, "kern3.so"); SRC = os.path.join(HERE, "kern3.c")
class DNet(ctypes.Structure):
    _fields_ = [(k, I64) for k in "H NL S D".split()] + [(k, F64) for k in "eta1 eta2 lam".split()] + [(k, PD) for k in "W1 b1 W2 b2 R".split()] + [("hist", PI), ("rate", PD)] + \
               [(k, I64) for k in "sig_on sn".split()] + [(k, F64) for k in "beta pw kappa sa sS sM".split()]
_lib = None
def lib():
    global _lib
    if _lib is None:
        if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
            tmp = f"{SO}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", *FLAGS, "-o", tmp, SRC, "-lm"]); os.replace(tmp, SO)
        _lib = ctypes.CDLL(SO); _lib.dense_steps.restype = I64; _lib.dense_steps.argtypes = [ctypes.POINTER(DNet), I64, P32, P32]
    return _lib
def run(s):
    xs = np.asarray(s.bx, np.int32); ys = np.asarray(s.by, np.int32); s.bx = []; s.by = []
    if not len(xs): return
    assert not s.clip and s.M is None and not s.dale and not s.H2 and s.live is None and not s.two and not s.stiff1 and not s.rewire, "fast=3 is for the plain system"
    n = DNet(); NL = s.W1.shape[1]; S = s.W2.shape[0]; hist = np.asarray(s.hist, np.int64)
    n.H, n.NL, n.S, n.D = s.H, NL, S, NL // S; n.eta1, n.eta2, n.lam = s.eta1, s.eta2, s.lam
    n.W1, n.b1, n.W2, n.b2, n.R = pd(s.W1), pd(s.b1), pd(s.W2), pd(s.b2), pd(s.R); n.hist = hist.ctypes.data_as(PI)
    if s.rate is not None: n.rate = pd(s.rate)
    n.sig_on = int(bool(s.sig))
    if s.sig: n.beta, n.pw, n.kappa, n.sa, n.sS, n.sM, n.sn = s.sig, s.sigp, s.sigk, s.sa, s.sS, s.sM, s.sn
    lib().dense_steps(ctypes.byref(n), len(xs), xs.ctypes.data_as(P32), ys.ctypes.data_as(P32)); s.hist = [int(v) for v in hist]
    if s.sig: s.sa, s.sS, s.sM, s.sn = float(n.sa), float(n.sS), float(n.sM), int(n.sn)
if __name__ == "__main__": lib(); print("built", SO)
