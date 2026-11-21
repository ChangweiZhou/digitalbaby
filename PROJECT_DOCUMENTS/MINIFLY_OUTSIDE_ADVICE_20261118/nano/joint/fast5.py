#!/usr/bin/env python3
"""The compiled inner loop (kern5.c) of joint5.py; `fast=1`. The numpy program remains the reference (check_joint5.py)."""
import ctypes, os, subprocess
import numpy as np
from fastj import FLAGS, I64, F64, PD, PI, P32, pd
HERE = os.path.dirname(os.path.abspath(__file__)); SO = os.path.join(HERE, "kern5.so"); SRC = os.path.join(HERE, "kern5.c")
class SNet(ctypes.Structure):
    _fields_ = [(k, I64) for k in "H NL S D ans cel".split()] + [(k, F64) for k in "eta1 eta2 lam p q typ typw".split()] + [(k, PD) for k in "W1 b1 W2 b2 R".split()] + [("hist", PI)]
_lib = None
def lib():
    global _lib
    if _lib is None:
        if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
            tmp = f"{SO}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", *FLAGS, "-o", tmp, SRC, "-lm"]); os.replace(tmp, SO)
        _lib = ctypes.CDLL(SO); _lib.rule_steps.restype = I64; _lib.rule_steps.argtypes = [ctypes.POINTER(SNet), I64, P32, P32]
    return _lib
def run(s, xs, ys):
    xs = np.ascontiguousarray(xs, np.int32); ys = np.ascontiguousarray(ys, np.int32)
    if not len(xs): return
    n = SNet(); NL = s.W1.shape[1]; S = s.A.shape[0]; hist = np.asarray(s.hist, np.int64)
    n.H, n.NL, n.S, n.D = s.H, NL, S, NL // S; n.eta1, n.eta2, n.lam = s.eta1, s.eta2, s.lam; import joint5; n.ans, n.cel, n.p, n.q, n.typ, n.typw = s.ans, s.cel, s.p, s.q, joint5.TYP, joint5.TYPW
    n.W1, n.b1, n.W2, n.b2, n.R = pd(s.W1), pd(s.b1), pd(s.A), pd(s.b2), pd(s.R); n.hist = hist.ctypes.data_as(PI)
    lib().rule_steps(ctypes.byref(n), len(xs), xs.ctypes.data_as(P32), ys.ctypes.data_as(P32)); s.hist = [int(v) for v in hist]
if __name__ == "__main__": lib(); print("built", SO)
