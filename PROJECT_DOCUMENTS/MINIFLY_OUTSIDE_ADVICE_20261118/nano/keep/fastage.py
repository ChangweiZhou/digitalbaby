#!/usr/bin/env python3
"""The compiled inner loop (kage.c) of kage.py, called through ctypes. The numpy program remains the reference (check_kage.py)."""
import ctypes, os, subprocess
import numpy as np
from fastk import FLAGS, I64, F64, PD, PI, PU, P32, pd
HERE = os.path.dirname(os.path.abspath(__file__)); SO = os.path.join(HERE, "kagek.so"); SRC = os.path.join(HERE, "kage.c")
class ANet(ctypes.Structure):
    _fields_ = [(k, I64) for k in "H NL S D n law young usemode sigjoin nl blaw meet".split()] + [(k, F64) for k in "eta1 eta2 lam tau E rhoE C G rhoS delta btau A rhoA K".split()] + \
               [(k, PD) for k in "W1 b1 W2 b2 R U thr".split()] + [("born", PI), ("hist", PI), ("seen", PU)] + [(k, PD) for k in "h rate m".split()]
_lib = None
def lib():
    global _lib
    if _lib is None:
        if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
            tmp = f"{SO}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", *FLAGS, "-o", tmp, SRC, "-lm"]); os.replace(tmp, SO)
        _lib = ctypes.CDLL(SO); _lib.age_steps.restype = None; _lib.age_steps.argtypes = [ctypes.POINTER(ANet), I64, P32, P32]
    return _lib
def run(s):
    xs = np.asarray(s.bx, np.int32); ys = np.asarray(s.by, np.int32); s.bx = []; s.by = []
    if not len(xs): return
    q = ANet(); NL = s.W1.shape[1]; S = s.W2.shape[0]; hist = np.asarray(s.hist, np.int64); H = s.H
    q.H, q.NL, q.S, q.D, q.n, q.law, q.young, q.usemode, q.sigjoin, q.nl, q.blaw, q.meet = H, NL, S, NL // S, s.n, s.law, s.young, s.usemode, s.sigjoin, s.nl, s.blaw, s.meet
    q.eta1, q.eta2, q.lam, q.tau, q.E, q.rhoE, q.C, q.G, q.rhoS, q.delta, q.btau, q.A, q.rhoA, q.K = s.eta1, s.eta2, s.lam, s.tau, s.E, s.rhoE, s.C, s.G, s.rhoS, s.delta, s.btau, s.A, s.rhoA, s.K
    assert s.born.dtype == np.int64 and s.seen.dtype == np.uint8 and s.seen.shape == (H, NL) and s.seen.flags.c_contiguous
    q.W1, q.b1, q.W2, q.b2, q.R, q.U, q.thr = pd(s.W1), pd(s.b1), pd(s.W2), pd(s.b2), pd(s.R), pd(s.U), pd(s.thr)
    q.born = s.born.ctypes.data_as(PI); q.hist = hist.ctypes.data_as(PI); q.seen = s.seen.ctypes.data_as(PU)
    work = [np.zeros(k) for k in (H, H, S)]; q.h, q.rate, q.m = [pd(w) for w in work]
    lib().age_steps(ctypes.byref(q), len(xs), xs.ctypes.data_as(P32), ys.ctypes.data_as(P32))
    s.hist = [int(v) for v in hist]; s.n = int(q.n); s.nl = int(q.nl); s.E = float(q.E); s.C = float(q.C); s.G = float(q.G); s.A = float(q.A)
if __name__ == "__main__": lib(); print("built", SO)
