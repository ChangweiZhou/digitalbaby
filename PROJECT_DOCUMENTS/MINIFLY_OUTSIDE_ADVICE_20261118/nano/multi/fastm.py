#!/usr/bin/env python3
"""The compiled inner loop (kernm.c) of multi.py, called through ctypes, after loop_small_case/fastk.py (commit 260c5d7). The numpy program remains
the reference; `fast=1` runs the same arithmetic here.    python3 fastm.py   builds kernm.so (gcc);   python3 check_multi.py   compares the two."""
import ctypes, os, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SO = os.path.join(HERE, "kernm.so"); SRC = os.path.join(HERE, "kernm.c")
FLAGS = ["-O3", "-march=native", "-ffast-math", "-shared", "-fPIC"]
I64, F64, PD, PI, PU, P32 = ctypes.c_int64, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32)

class MNet(ctypes.Structure):
    _fields_ = [(k, I64) for k in "H NL S D P has_M comp rewire nbytes due right".split()] + [(k, F64) for k in "eta1 eta2 lam nats".split()] + \
               [(k, PD) for k in "W1 b1 W2 b2 R PW".split()] + [("M", PU), ("PA", PI), ("PB", PI), ("hist", PI)]

def build():
    if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
        tmp = f"{SO}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", *FLAGS, "-o", tmp, SRC, "-lm"]); os.replace(tmp, SO)
    return SO
_lib = None
def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build()); _lib.multi_steps.restype = I64; _lib.multi_steps.argtypes = [ctypes.POINTER(MNet), I64, P32, P32]
        _lib.multi_drive.restype = None; _lib.multi_drive.argtypes = [ctypes.POINTER(MNet), I64, P32, PD]
    return _lib
def pd(a):
    assert a.dtype == np.float64 and a.flags.c_contiguous and a.flags.writeable, "the kernel changes the arrays in place"; return a.ctypes.data_as(PD)

def net(s, hist):
    n = MNet(); n.H, n.NL, n.S, n.D, n.P, n.comp, n.rewire, n.nbytes, n.due, n.right = s.H, s.NL, s.S, s.D, s.P, int(s.comp), int(s.rewire), int(s.nbytes), 0, 0
    n.eta1, n.eta2, n.lam, n.nats = s.eta1, s.eta2, s.lam, 0.0; n.W1, n.b1, n.W2, n.b2, n.R, n.PW = pd(s.W1), pd(s.b1), pd(s.W2), pd(s.b2), pd(s.R), pd(s.PW)
    assert s.W1.shape == (s.H, s.NL) and s.W2.shape == (s.S, s.H) and s.R.shape == (s.H, s.S) and s.D <= 64
    n.has_M = int(s.M is not None)
    if s.M is not None: assert s.M.dtype == np.bool_ and s.M.flags.c_contiguous; n.M = s.M.ctypes.data_as(PU)
    assert s.PA.dtype == np.int64 and s.PB.dtype == np.int64 and s.PA.flags.c_contiguous and s.PB.flags.c_contiguous
    n.PA = s.PA.ctypes.data_as(PI); n.PB = s.PB.ctypes.data_as(PI); n.hist = hist.ctypes.data_as(PI); return n

def run(s, xs, ys):
    """multi.Sys: the bytes are read and learned from; the strengths change in place; the turnover of connections is done by multi.py."""
    xs = np.ascontiguousarray(xs, np.int32); ys = np.ascontiguousarray(ys, np.int32)
    if not len(xs): return
    hist = np.asarray(s.hist, np.int64); pos = 0; L = lib()
    while pos < len(xs):
        n = net(s, hist)                              # made afresh after a turnover: the pairs a cell holds may have changed
        pos += L.multi_steps(ctypes.byref(n), len(xs) - pos, xs[pos:].ctypes.data_as(P32), ys[pos:].ctypes.data_as(P32))
        s.nbytes = int(n.nbytes); s.right += int(n.right); s.nats += float(n.nats); s.hist = [int(v) for v in hist]
        if n.due: s.turnover()
    s.hist = [int(v) for v in hist]

def drive(s, h):
    h = np.ascontiguousarray(h, np.int32); out = np.zeros((len(h), s.H)); hist = np.asarray(s.hist, np.int64); n = net(s, hist)
    lib().multi_drive(ctypes.byref(n), len(h), h.ctypes.data_as(P32), pd(out)); return out

if __name__ == "__main__": print("built", build())
