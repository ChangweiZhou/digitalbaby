#!/usr/bin/env python3
"""The gated-cell kernel with a short chain of delay lines (kern4.c), for hold.py with chain=C. The numpy program remains the reference."""
import ctypes, os, subprocess
import numpy as np
from fastk import FLAGS, I64, F64, PD, PI, P32, pd
HERE = os.path.dirname(os.path.abspath(__file__)); SO = os.path.join(HERE, "kern4.so"); SRC = os.path.join(HERE, "kern4.c")
class HNet4(ctypes.Structure):
    _fields_ = [(k, I64) for k in "H S K cross retmode right C NI".split()] + [(k, F64) for k in "eta1 eta2 lam lamB qs".split()] + [("idx", PI), ("prev", PI)] + \
               [(k, PD) for k in "Ac Az Bc Bz cc cz R W2 b2 h Ecx Ezx Ecr Ezr Ecb Ezb Qc Qz hp pre z c kc kz r rho m".split()]
_lib = None
def lib():
    global _lib
    if _lib is None:
        if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
            tmp = f"{SO}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", *FLAGS, "-o", tmp, SRC, "-lm"]); os.replace(tmp, SO)
        _lib = ctypes.CDLL(SO); _lib.hold4_steps.restype = None; _lib.hold4_steps.argtypes = [ctypes.POINTER(HNet4), I64, P32, P32]
    return _lib
def hold_run(s):
    xs = np.asarray(s.bx, np.int32); ys = np.asarray(s.by, np.int32); s.bx = []; s.by = []
    if not len(xs): return
    n = HNet4(); H, K = s.Bc.shape; S = s.W2.shape[0]; n.H, n.S, n.K, n.cross, n.right, n.C, n.NI = H, S, K, int(bool(s.cross)), 0, s.C, s.Ac.shape[1]
    assert s.Ac.shape == (H, S * (1 + s.C)) and s.C <= 15
    n.retmode = {"exact": 0, "sign": 1, "learned": 2, "grown": 2, "local": 3}[s.ret] if s.cross else 0
    n.eta1, n.eta2, n.lam, n.lamB, n.qs = s.eta1, s.eta2, s.lam, s.lamB, s.qs
    assert s.idx.dtype == np.int64 and s.idx.flags.c_contiguous; n.idx = s.idx.ctypes.data_as(PI); prev = np.asarray(s.prev if s.C else [0], np.int64); n.prev = prev.ctypes.data_as(PI)
    for k in "Ac Az Bc Bz cc cz R W2 b2 h Ecx Ezx Ecr Ezr Ecb Ezb".split(): setattr(n, k, pd(getattr(s, k)))
    if s.learned: n.Qc, n.Qz = pd(s.Qc), pd(s.Qz)
    work = [np.zeros(k) for k in (H, H * K, H, H, H, H, H, H, S)]; n.hp, n.pre, n.z, n.c, n.kc, n.kz, n.r, n.rho, n.m = [pd(w) for w in work]
    lib().hold4_steps(ctypes.byref(n), len(xs), xs.ctypes.data_as(P32), ys.ctypes.data_as(P32)); s.right_acc += int(n.right)
    if s.C: s.prev = [int(v) for v in prev]
if __name__ == "__main__": lib(); print("built", SO)
