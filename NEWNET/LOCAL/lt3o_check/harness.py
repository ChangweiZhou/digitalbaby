"""Load any build of the lt3 API (prefix lt3 or lt3o) from a given .so; same Net3 interface as lt3.py."""
import ctypes, numpy as np, sys
sys.path.insert(0, "..")
from lt3 import GI, GD, full
class Lib:
    def __init__(self, so, pre):
        L = ctypes.CDLL(so); self.f = {k: getattr(L, pre + "_" + k) for k in ("new", "run", "snapshot", "restore", "audit")}
        self.f["new"].restype = ctypes.c_void_p; self.f["new"].argtypes = [ctypes.c_void_p] * 4
        for k in ("snapshot", "restore"): self.f[k].argtypes = [ctypes.c_void_p]
        self.f["run"].argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.f["audit"].argtypes = [ctypes.c_void_p, ctypes.c_void_p]
class Net:
    def __init__(self, lib, cand, scale=1, seed=None):
        g, pops = full(cand); K = len(pops); g["K"] = K
        if seed is not None: g["seed"] = seed
        gi = np.array([int(g[k]) for k in GI], np.int32); gd = np.array([float(g[k]) for k in GD]); pi, pd = [], []
        for q, p in enumerate(pops):
            n = int(p["n"] * scale); k = max(1, int(round(p["a"] * p["B"])))
            pm = sum(1 << j for j in p["prox"] if j < q); dm = sum(1 << j for j in p["dist"] if 0 <= j < K)
            pi.append([n, p["FP"], p["FD"], p["G"], p["B"], k, p["lp"], p["ld"], p["ad"], p["inp"], p["rd"], p["np"], pm, dm])
            pd.append([p["tlo"], p["thi"], p["rho"], p["pf"], p["sth"], p["dg"]])
        self._k = (gi, gd, np.array(pi, np.int32).ravel(), np.array(pd, np.float64).ravel())
        self.L = lib; self.h = lib.f["new"](*[a.ctypes.data for a in self._k])
    def run(self, x, learn=True):
        x = np.ascontiguousarray(x, np.uint8); st = np.zeros(3); self.L.f["run"](self.h, x.ctypes.data, len(x), int(learn), st.ctypes.data); return st[0] / max(st[2], 1)
    def test(self, x):
        self.L.f["snapshot"](self.h); r = self.run(x, False); self.L.f["restore"](self.h); return r
