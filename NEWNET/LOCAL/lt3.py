"""Python side of lt3.c (K populations). A candidate is a dict: g (global options) and pops (one dict per population).
Population options: n, FP, FD, G, B, a (share firing per block), lp, ld, ad, in, rd, np (patches), prox (list of earlier populations
read proximally), dist (populations read distally, one byte late), tlo, thi (time constants), rho, sth, dg."""
import ctypes, os, subprocess, json
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_so = os.path.join(_here, "liblt3.so")
if not os.path.exists(_so) or os.path.getmtime(_so) < os.path.getmtime(os.path.join(_here, "lt3.c")):
    _tmp = _so + ".%d.tmp" % os.getpid()
    subprocess.check_call(["cc", "-O3", "-march=native", "-shared", "-fPIC", "-o", _tmp, os.path.join(_here, "lt3.c"), "-lm"]); os.replace(_tmp, _so)
_lib = ctypes.CDLL(_so)
_lib.lt3_new.restype = ctypes.c_void_p
_lib.lt3_new.argtypes = [ctypes.c_void_p] * 4
for f in ("lt3_snapshot", "lt3_restore"): getattr(_lib, f).argtypes = [ctypes.c_void_p]
_lib.lt3_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_lib.lt3_audit.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

GDEF = dict(NL=512, KIN=16, FOI=16, FO=16, NO=16, FOO=16, NGP=6, NGD=8, seed=1,
            eta=0.2, eta_m=0.05, w_new=0.4, eta_o=0.2, eta_b=0.01, eta_t=0.01, rho_o=0.5, orep=0.0, tag=0, gate=0, hubm=4, tautag=2000.0, cap=0.5, gth=0.5, turn=0.0, boost=4.0, otr=1.0, hubf=0.0)
PDEF = dict(n=2048, FP=6, FD=10, G=10, B=64, a=0.05, lp=0, ld=1, ad=0, inp=1, rd=1, np=0, prox=[], dist=[0],
            tlo=1.0, thi=1.0, rho=0.15, pf=0.0, sth=0.6, dg=1.0)
GI = ["K", "NL", "KIN", "FOI", "FO", "NO", "FOO", "NGP", "NGD", "seed", "tag", "gate", "hubm"]
GD = ["eta", "eta_m", "w_new", "eta_o", "eta_b", "eta_t", "rho_o", "orep", "tautag", "cap", "gth", "turn", "boost", "otr", "hubf"]


def full(cand):
    g = dict(GDEF, **cand.get("g", {})); pops = [dict(PDEF, **p) for p in cand["pops"]]; return g, pops


class Net3:
    def __init__(self, cand, scale=1):
        g, pops = full(cand); K = len(pops)
        assert 1 <= K <= 6 and any(p["rd"] for p in pops) and any(p["inp"] for p in pops)
        g["K"] = K
        gi = np.array([int(g[k]) for k in GI], np.int32); gd = np.array([float(g[k]) for k in GD])
        pi, pd = [], []
        for q, p in enumerate(pops):
            n = int(p["n"] * scale); k = max(1, int(round(p["a"] * p["B"])))
            pm = sum(1 << j for j in p["prox"] if j < q); dm = sum(1 << j for j in p["dist"] if 0 <= j < K)
            pi.append([n, p["FP"], p["FD"], p["G"], p["B"], k, p["lp"], p["ld"], p["ad"], p["inp"], p["rd"], p["np"], pm, dm])
            pd.append([p["tlo"], p["thi"], p["rho"], p["pf"], p["sth"], p["dg"]])
        pi = np.array(pi, np.int32).ravel(); pd = np.array(pd, np.float64).ravel()
        self.cand = cand; self.h = _lib.lt3_new(gi.ctypes.data, gd.ctypes.data, pi.ctypes.data, pd.ctypes.data)

    def run(self, x, learn=True):
        x = np.ascontiguousarray(x, np.uint8); st = np.zeros(3)
        _lib.lt3_run(self.h, x.ctypes.data, len(x), 1 if learn else 0, st.ctypes.data)
        return st[0] / max(st[2], 1), st[1] / max(st[2], 1)

    def test(self, x):
        _lib.lt3_snapshot(self.h); r = self.run(x, False); _lib.lt3_restore(self.h); return r

    def audit(self):
        a = np.zeros(6, np.int64); _lib.lt3_audit(self.h, a.ctypes.data)
        return dict(zip(["fanin", "fanout", "fanout_line", "fanin_out", "fanout_out", "syn"], a.tolist()))
