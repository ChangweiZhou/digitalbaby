#!/usr/bin/env python3
"""The compiled inner loop (kernq.c) of prod.py, called through ctypes. The numpy program remains the reference; `fast=1` runs the same arithmetic
here.    python3 fastq.py   builds kernq.so (gcc);   python3 check_prod.py   compares the two."""
import ctypes, os, subprocess
import numpy as np
from fastm import FLAGS, I64, F64, PD, PI, P32, pd
HERE = os.path.dirname(os.path.abspath(__file__)); SO = os.path.join(HERE, "kernq.so"); SRC = os.path.join(HERE, "kernq.c")

class QNet(ctypes.Structure):
    _fields_ = [(k, I64) for k in "H H2 NL S D KE KP joint rewire nbytes due right".split()] + [(k, F64) for k in "eta1 eta2 lam rho".split()] + \
               [(k, PD) for k in "W1 b1 W2 b2 R bB EW ER PW PR QA QB".split()] + [(k, PI) for k in "EA PA PB hist".split()]

def build():
    if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
        tmp = f"{SO}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", *FLAGS, "-o", tmp, SRC, "-lm"]); os.replace(tmp, SO)
    return SO
_lib = None
def lib():
    global _lib
    if _lib is None: _lib = ctypes.CDLL(build()); _lib.prod_steps.restype = I64; _lib.prod_steps.argtypes = [ctypes.POINTER(QNet), I64, P32, P32]
    return _lib

def run(s, xs, ys):
    """prod.Sys3: the bytes are read and learned from; the strengths change in place; the turnover of connections is done by prod.py."""
    xs = np.ascontiguousarray(xs, np.int32); ys = np.ascontiguousarray(ys, np.int32)
    if not len(xs): return
    hist = np.asarray(s.hist, np.int64); pos = 0; L = lib(); n = QNet()
    n.H, n.H2, n.NL, n.S, n.D, n.KE, n.KP, n.joint, n.rewire = s.H, s.H2, s.W1.shape[1], s.W2.shape[0], len(s.hist), s.KE, s.KP, int(s.joint), int(s.rewire)
    n.eta1, n.eta2, n.lam, n.rho = s.eta1, s.eta2, s.lam, s.rho
    for k in "W1 b1 W2 b2 R bB EW ER PW PR QA QB".split(): setattr(n, k, pd(getattr(s, k)))
    for k in "EA PA PB".split():
        a = getattr(s, k); assert a.dtype == np.int64 and a.flags.c_contiguous; setattr(n, k, a.ctypes.data_as(PI))
    assert s.EW.shape == (s.H2, s.KE) and s.PW.shape == (s.H2, s.KP) and s.R.shape == (s.H2, n.S) and n.S <= 64 and n.D <= 64; n.hist = hist.ctypes.data_as(PI)
    while pos < len(xs):
        n.nbytes, n.due, n.right = int(s.nbytes), 0, 0
        pos += L.prod_steps(ctypes.byref(n), len(xs) - pos, xs[pos:].ctypes.data_as(P32), ys[pos:].ctypes.data_as(P32)); s.nbytes = int(n.nbytes); s.right += int(n.right)
        if n.due: s.turnover()                        # the arrays are changed in place, so the pointers stay good
    s.hist = [int(v) for v in hist]

if __name__ == "__main__": print("built", build())
