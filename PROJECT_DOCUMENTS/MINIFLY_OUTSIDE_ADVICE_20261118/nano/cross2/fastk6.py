#!/usr/bin/env python3
"""The compiled pass backward over a line (kern6.c) for hold_x.py at X=inf. The numpy program hold_x.py remains the reference (check_bp.py)."""
import ctypes, os, subprocess
import numpy as np
from fastk5 import FLAGS, I64, F64, PD, PI, P32, pd
HERE = os.path.dirname(os.path.abspath(__file__)); SO = os.path.join(HERE, "kern6.so"); SRC = os.path.join(HERE, "kern6.c"); TMAX = 64
class BNet(ctypes.Structure):
    _fields_ = [(k, I64) for k in "H S K right C NI T adam nadam".split()] + [(k, F64) for k in "eta1 eta2 lam alpha b1 b2a eps".split()] + [("idx", PI), ("prev", PI), ("cols", PI)] + \
               [(k, PD) for k in "Ac Az Bc Bz cc cz R W2 b2 h zs kcs kzs rs pres c m d dn G M V".split()]
_lib = None
def lib():
    global _lib
    if _lib is None:
        if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
            tmp = f"{SO}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", *FLAGS, "-o", tmp, SRC, "-lm"]); os.replace(tmp, SO)
        _lib = ctypes.CDLL(SO); _lib.hx_steps.restype = None; _lib.hx_steps.argtypes = [ctypes.POINTER(BNet), I64, P32, P32]
    return _lib
def hx_run(s):
    xs = np.asarray(s.bx, np.int32); ys = np.asarray(s.by, np.int32); s.bx = []; s.by = []
    if not len(xs): return
    assert ys[-1] < 0, "whole lines only"
    n = BNet(); H, K = s.Bc.shape; S = s.W2.shape[0]; NI = s.Ac.shape[1]; n.H, n.S, n.K, n.right, n.C, n.NI, n.T = H, S, K, 0, s.C, NI, 0; NP = 2 * H * (NI + K + 1)
    n.eta1, n.eta2, n.lam = s.eta1, s.eta2, s.lam; n.adam = int(bool(getattr(s, "adam", 0)))
    if n.adam:
        if not hasattr(s, "aM"): s.aM = np.zeros(NP); s.aV = np.zeros(NP); s.nadam = 0
        n.alpha, n.b1, n.b2a, n.eps, n.nadam, n.M, n.V = s.alpha, 0.9, 0.999, 1e-8, s.nadam, pd(s.aM), pd(s.aV)
    prev = np.asarray(s.prev if s.C else [0], np.int64); n.prev = prev.ctypes.data_as(PI); cols = np.zeros(TMAX * 16, np.int64); n.cols = cols.ctypes.data_as(PI); n.idx = s.idx.ctypes.data_as(PI)
    for k in "Ac Az Bc Bz cc cz R W2 b2 h".split(): setattr(n, k, pd(getattr(s, k)))
    work = [np.zeros(k) for k in (TMAX * H, TMAX * H, TMAX * H, TMAX * H, TMAX * H * K, H, S, H, H, NP)]; n.zs, n.kcs, n.kzs, n.rs, n.pres, n.c, n.m, n.d, n.dn, n.G = [pd(w) for w in work]
    lib().hx_steps(ctypes.byref(n), len(xs), xs.ctypes.data_as(P32), ys.ctypes.data_as(P32)); s.right_acc += int(n.right)
    if s.C: s.prev = [int(v) for v in prev]
    if n.adam: s.nadam = int(n.nadam)
if __name__ == "__main__": lib(); print("built", SO)
