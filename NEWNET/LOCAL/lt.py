"""Python side of lt.c (24 Nov 2026): birth, phases (one C call each), snapshot, audit. Usage: see ltb.py."""
import ctypes, os, subprocess
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_so = os.path.join(_here, "liblt.so")
if not os.path.exists(_so) or os.path.getmtime(_so) < os.path.getmtime(os.path.join(_here, "lt.c")):
    _tmp = _so + ".%d.tmp" % os.getpid()
    subprocess.check_call(["cc", "-O3", "-march=native", "-shared", "-fPIC", "-o", _tmp, os.path.join(_here, "lt.c"), "-lm"]); os.replace(_tmp, _so)
_lib = ctypes.CDLL(_so)
_lib.lt_new.restype = ctypes.c_void_p
_lib.lt_new.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
for f in ("lt_snapshot", "lt_restore"): getattr(_lib, f).argtypes = [ctypes.c_void_p]
_lib.lt_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_lib.lt_audit.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

IK = ["R", "C", "M", "S", "P", "B", "KB", "NG", "FO", "FOP", "NO", "FOO", "lat", "td", "outr", "seed", "graded"]
DK = ["rho_p", "rho_d", "rho_o", "eta", "eta_m", "w_on", "w_new", "th_on", "th_m", "eta_o", "eta_b", "eta_t"]
DEF = dict(R=6, C=256, M=16, S=10, P=6, B=32, KB=2, NG=8, FO=12, FOP=12, NO=12, FOO=12, lat=0, td=0, outr=0, seed=1, graded=1,
           rho_p=0.1, rho_d=0.15, rho_o=0.5, eta=0.2, eta_m=0.05, w_on=0.5, w_new=0.4, th_on=0.75, th_m=0.5,
           eta_o=0.2, eta_b=0.01, eta_t=0.01)


class Net:
    def __init__(self, **kw):
        bad = [k for k in kw if k not in DEF]
        if bad: raise TypeError(f"unknown options {bad}")
        self.p = dict(DEF, **kw)
        ip = np.array([int(self.p[k]) for k in IK], np.int32); dp = np.array([float(self.p[k]) for k in DK], np.float64)
        self.h = _lib.lt_new(ip.ctypes.data, dp.ctypes.data)

    def run(self, x, learn=True):
        """read the bytes x (uint8); returns (nats per scored byte, right share)"""
        x = np.ascontiguousarray(x, np.uint8); st = np.zeros(3)
        _lib.lt_run(self.h, x.ctypes.data, len(x), 1 if learn else 0, st.ctypes.data)
        return st[0] / max(st[2], 1), st[1] / max(st[2], 1)

    def test(self, x):
        """score x with learning off, then put the dynamic state back (the life is not disturbed)"""
        _lib.lt_snapshot(self.h); r = self.run(x, False); _lib.lt_restore(self.h); return r

    def audit(self):
        a = np.zeros(7, np.int64); _lib.lt_audit(self.h, a.ctypes.data)
        return dict(zip(["fanin", "fanout_d", "fanout_p", "fanin_out", "fanout_out", "syn", "cells_used"], a.tolist()))
