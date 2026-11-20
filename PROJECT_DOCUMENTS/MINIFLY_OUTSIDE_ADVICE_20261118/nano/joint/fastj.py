#!/usr/bin/env python3
"""The compiled inner loop (kernj.c) of joint.py; `fast=1`. The numpy program remains the reference (check_joint.py).
The build and the call are copied from loop_small_case/fastk.py and fastk3.py at commit fe397ca."""
import ctypes, os, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SO = os.path.join(HERE, "kernj.so"); SRC = os.path.join(HERE, "kernj.c")
FLAGS = ["-O3", "-march=native", "-ffast-math", "-shared", "-fPIC"]
I64, F64, PD, PI, P32 = ctypes.c_int64, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32)
class JNet(ctypes.Structure):
    _fields_ = [(k, I64) for k in "H NL S D form".split()] + [(k, F64) for k in "eta1 eta2 lam etaB".split()] + [(k, PD) for k in "W1 b1 A B b2 R".split()] + [("hist", PI)]
_lib = None
def lib():
    global _lib
    if _lib is None:
        if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
            tmp = f"{SO}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", *FLAGS, "-o", tmp, SRC, "-lm"]); os.replace(tmp, SO)
        _lib = ctypes.CDLL(SO); _lib.joint_steps.restype = I64; _lib.joint_steps.argtypes = [ctypes.POINTER(JNet), I64, P32, P32]
    return _lib
def pd(a):
    assert a.dtype == np.float64 and a.flags.c_contiguous and a.flags.writeable, "the kernel changes the arrays in place"; return a.ctypes.data_as(PD)
def run(s, xs, ys):
    xs = np.ascontiguousarray(xs, np.int32); ys = np.ascontiguousarray(ys, np.int32)
    if not len(xs): return
    n = JNet(); NL = s.W1.shape[1]; S = s.A.shape[0]; hist = np.asarray(s.hist, np.int64)
    n.H, n.NL, n.S, n.D, n.form = s.H, NL, S, NL // S, s.form; n.eta1, n.eta2, n.lam, n.etaB = s.eta1, s.eta2, s.lam, s.etaB
    n.W1, n.b1, n.A, n.B, n.b2, n.R = pd(s.W1), pd(s.b1), pd(s.A), pd(s.B), pd(s.b2), pd(s.R); n.hist = hist.ctypes.data_as(PI)
    lib().joint_steps(ctypes.byref(n), len(xs), xs.ctypes.data_as(P32), ys.ctypes.data_as(P32)); s.hist = [int(v) for v in hist]
if __name__ == "__main__": lib(); print("built", SO)
