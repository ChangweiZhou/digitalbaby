#!/usr/bin/env python3
"""The compiled inner loops (kern.c) of cells2.py and hold.py, called through ctypes. The numpy programs remain the reference; `fast=1` on either
program's command line runs the same arithmetic here.    python3 fastk.py     builds kern.so (gcc);   python3 check_fast.py   compares the two."""
import ctypes, os, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SO = os.path.join(HERE, "kern.so"); SRC = os.path.join(HERE, "kern.c")
FLAGS = ["-O3", "-march=native", "-ffast-math", "-shared", "-fPIC"]
I64, F64, PD, PI, PU, P32 = ctypes.c_int64, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32)

class Net(ctypes.Structure):
    _fields_ = [(k, I64) for k in "H NL S D top H2 has_M dale has_live late rewire nbytes due".split()] + [(k, F64) for k in "eta1 eta2 lam".split()] + \
               [(k, PD) for k in "W1 b1 W2 b2 R WB bB RB sign live freec".split()] + [("M", PU), ("hist", PI)] + [(k, PD) for k in "hA hB m sig sA d".split()]
class HNet(ctypes.Structure):
    _fields_ = [(k, I64) for k in "H S K cross retmode right".split()] + [(k, F64) for k in "eta1 eta2 lam lamB qs".split()] + [("idx", PI)] + \
               [(k, PD) for k in "Ac Az Bc Bz cc cz R W2 b2 h Ecx Ezx Ecr Ezr Ecb Ezb Qc Qz hp pre z c kc kz r rho m".split()]

def build():
    if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
        tmp = f"{SO}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", *FLAGS, "-o", tmp, SRC, "-lm"]); os.replace(tmp, SO)
    return SO
_lib = None
def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build()); _lib.loop_steps.restype = I64; _lib.loop_steps.argtypes = [ctypes.POINTER(Net), I64, P32, P32]
        _lib.hold_steps.restype = None; _lib.hold_steps.argtypes = [ctypes.POINTER(HNet), I64, P32, P32]
    return _lib
def pd(a):
    assert a.dtype == np.float64 and a.flags.c_contiguous and a.flags.writeable, "the kernel changes the arrays in place"; return a.ctypes.data_as(PD)

def loop_run(s):
    """cells2.Sys: the buffered bytes are read and learned from; the strengths change in place."""
    xs = np.asarray(s.bx, np.int32); ys = np.asarray(s.by, np.int32); s.bx = []; s.by = []
    if not len(xs): return
    NL = s.W1.shape[1]; S = s.W2.shape[0]; D = NL // S                 # taken from the arrays: the program may be running as a script
    assert not s.two and not s.stiff1, "ans2 and stiff1 are in the numpy program only"
    n = Net(); H = s.H; top = s.H2 or H; hist = np.asarray(s.hist, np.int64); z = np.zeros(1); keep = [hist]
    n.H, n.NL, n.S, n.D, n.top, n.H2, n.dale, n.rewire, n.nbytes, n.due = H, NL, S, D, top, s.H2, int(bool(s.dale)), int(s.rewire), int(getattr(s, "nbytes", 0)), 0
    n.eta1, n.eta2, n.lam = s.eta1, s.eta2, s.lam; n.W1, n.b1, n.W2, n.b2, n.R = pd(s.W1), pd(s.b1), pd(s.W2), pd(s.b2), pd(s.R)
    assert s.W1.shape == (H, NL) and s.W2.shape == (S, top) and s.R.shape == (top, S)
    if s.H2: n.WB, n.bB, n.RB = pd(s.WB), pd(s.bB), pd(s.RB)
    if s.dale: n.sign = pd(s.sign)
    n.has_M = int(s.M is not None)
    if s.M is not None: assert s.M.dtype == np.bool_ and s.M.flags.c_contiguous; n.M = s.M.ctypes.data_as(PU)
    n.has_live = int(s.live is not None); n.late = int(bool(s.late))
    if s.live is not None: live = s.live.astype(np.float64); free = np.ascontiguousarray(s.free, np.float64); keep += [live, free]; n.live, n.freec = pd(live), pd(free)
    work = [np.zeros(k) for k in (H, max(s.H2, 1), S, top, H, H)]; keep += work; n.hA, n.hB, n.m, n.sig, n.sA, n.d = [pd(w) for w in work]; n.hist = hist.ctypes.data_as(PI)
    pos = 0; L = lib()
    while pos < len(xs):
        pos += L.loop_steps(ctypes.byref(n), len(xs) - pos, xs[pos:].ctypes.data_as(P32), ys[pos:].ctypes.data_as(P32))
        if n.due: n.due = 0; s.nbytes = int(n.nbytes); s.turnover()
    s.hist = [int(v) for v in hist]
    if s.rewire: s.nbytes = int(n.nbytes)

def hold_run(s):
    """hold.Hold: the buffered bytes are read; traces, strengths and the state change in place; the bytes named right are counted in s.right_acc."""
    xs = np.asarray(s.bx, np.int32); ys = np.asarray(s.by, np.int32); s.bx = []; s.by = []
    if not len(xs): return
    n = HNet(); H, K = s.Bc.shape; S = s.Ac.shape[1]; n.H, n.S, n.K, n.cross, n.right = H, S, K, int(bool(s.cross)), 0
    n.retmode = {"exact": 0, "sign": 1, "learned": 2, "grown": 2, "local": 3}[s.ret] if s.cross else 0
    n.eta1, n.eta2, n.lam, n.lamB, n.qs = s.eta1, s.eta2, s.lam, s.lamB, s.qs
    assert s.idx.dtype == np.int64 and s.idx.flags.c_contiguous and s.Ac.shape == (H, S) and s.W2.shape == (S, H); n.idx = s.idx.ctypes.data_as(PI)
    for k in "Ac Az Bc Bz cc cz R W2 b2 h Ecx Ezx Ecr Ezr Ecb Ezb".split(): setattr(n, k, pd(getattr(s, k)))
    if s.learned: n.Qc, n.Qz = pd(s.Qc), pd(s.Qz)
    work = [np.zeros(k) for k in (H, H * K, H, H, H, H, H, H, S)]; n.hp, n.pre, n.z, n.c, n.kc, n.kz, n.r, n.rho, n.m = [pd(w) for w in work]
    lib().hold_steps(ctypes.byref(n), len(xs), xs.ctypes.data_as(P32), ys.ctypes.data_as(P32)); s.right_acc += int(n.right)

if __name__ == "__main__": print("built", build())
