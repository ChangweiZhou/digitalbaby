#!/usr/bin/env python3
"""The compiled inner loop (kernl.c) of layered.py, called through ctypes. The numpy program remains the reference; `fast=1` runs the same
arithmetic here.    python3 fastl.py   builds kernl.so (gcc);   python3 check_fastl.py   compares the two.
The flags and the calling pattern follow loop_small_case/fastk.py and fastk4.py at commit 0ba9f241."""
import ctypes, os, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SO = os.path.join(HERE, "kernl.so"); SRC = os.path.join(HERE, "kernl.c")
FLAGS = ["-O3", "-march=native", "-ffast-math", "-shared", "-fPIC"]
I64, F64, PD, PI, P32 = ctypes.c_int64, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32)
ARR = "Ac Az Ao cc cz co WB V bB RB R W2 b2 h Ecx Ezx Ecb Ezb".split()
class LNet(ctypes.Structure):
    _fields_ = [(k, I64) for k in "NA NB S C NI og right".split()] + [(k, F64) for k in "etaA etaB nB eta2 lam".split()] + [("prev", PI)] + [(k, PD) for k in ARR + "z o u b m sig".split()]
def pd(a):
    assert a.dtype == np.float64 and a.flags.c_contiguous; return a.ctypes.data_as(PD)
_lib = None
def lib():
    global _lib
    if _lib is None:
        if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
            tmp = f"{SO}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", *FLAGS, "-o", tmp, SRC, "-lm"]); os.replace(tmp, SO)
        _lib = ctypes.CDLL(SO); _lib.lay_steps.restype = None; _lib.lay_steps.argtypes = [ctypes.POINTER(LNet), I64, P32, P32]
    return _lib
def run(s):
    xs = np.asarray(s.bx, np.int32); ys = np.asarray(s.by, np.int32); s.bx = []; s.by = []
    if not len(xs): return
    n = LNet(); NB, NA = s.WB.shape; S = s.W2.shape[0]; n.NA, n.NB, n.S, n.C, n.NI, n.og, n.right = NA, NB, S, s.C, s.Ac.shape[1], int(bool(s.og)), 0
    assert s.Ac.shape == (NA, S * (1 + s.C)) and s.C <= 15
    n.etaA, n.etaB, n.nB, n.eta2, n.lam = s.etaA, s.etaB, s.nB, s.eta2, s.lam
    prev = np.asarray(s.prev if s.C else [0], np.int64); n.prev = prev.ctypes.data_as(PI)
    dummy = np.zeros(1)
    for k in ARR: a = getattr(s, k); setattr(n, k, pd(a if a.size else dummy))
    work = [np.zeros(k) for k in (NA, NA, NA, NB, S, NB)]; n.z, n.o, n.u, n.b, n.m, n.sig = [pd(w) for w in work]
    lib().lay_steps(ctypes.byref(n), len(xs), xs.ctypes.data_as(P32), ys.ctypes.data_as(P32)); s.right_acc += int(n.right)
    if s.C: s.prev = [int(v) for v in prev]
if __name__ == "__main__": lib(); print("built", SO)
