"""net5 with the context part of its step in C (net7.c). Same definition and parameters as net5.py (see its docstring).
Differences: among equally weak slots a new connection takes the lowest slot (net5: numpy's sort order), and the random sample
of new sources (only when more sources are active than n_grow) uses the C generator. With no sampling the cells, surprise and
predictions are those of net5 (stable=1) step for step (check_net7.py). The forward part and the hold are as in net5 (numpy).
slow=TAU: the hold part of every cell's context is a slow code (per region: the SLOW_K columns whose trace with time constant
TAU is most above its long-term mean, time constant SLOW_LT), every cell of those columns; it is required as the hold is.
mesh=K: regions form a random directed graph: each takes its columns from the byte's lines (its own wiring) and its context
from its own columns at the byte before and the cells of K other regions at the byte before (loops allowed); use with ctx=2.
nodef=1: surprise never makes a column fire its default cell (a predicted cell fires whenever there is one).
recall=1 (also in net5): surprise also when most winning columns were not predicted (else a hold left by another stream can
never be refreshed: nothing matches it, so nothing is predicted, so no surprise)."""
import ctypes, os, subprocess
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_so = os.path.join(_here, "libnet7.so")
if not os.path.exists(_so) or os.path.getmtime(_so) < os.path.getmtime(os.path.join(_here, "net7.c")):
    subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", _so, os.path.join(_here, "net7.c")])
_lib = ctypes.CDLL(_so)


class _P(ctypes.Structure):
    _fields_ = [(n, ctypes.c_double) for n in ("eta", "eta_minus", "w_on", "w_new", "th_on", "th_match", "s_th", "lam")] + \
               [(n, ctypes.c_int) for n in ("n_grow", "burst", "ctx_mode", "learn", "recall", "nodef")]


def _p(a): return a.ctypes.data_as(ctypes.c_void_p)


class Tissue:
    def __init__(self, seed=1, L=12, C=64, M=8, k_win=8, fan_in=32, S=48, n_values=32, lines_in=8, eta=0.2, eta_minus=0.05,
                 w_on=0.5, w_new=0.4, th_on=0.75, th_match=0.6, s_th=0.75, lam=0.1, n_grow=0, burst=0, ctx=0, hold=1, recall=0,
                 slow=0, slow_k=8, slow_lt=20000, nodef=0, mesh=0, mesh_base=1, mesh_dag=0, theta=1e-9, sweep=1):
        rng = np.random.default_rng(seed)
        self.L, self.C, self.M, self.k, self.S, self.theta, self.n_values, self.sweep = L, C, M, k_win, S, theta, n_values, sweep
        self.ctx_mode, self.use_hold = ctx, hold
        self.slow, self.slow_k, self.slow_lt = slow, slow_k, slow_lt                     # slow=tau: the hold part is a slow code instead
        self.tr = np.zeros((L, C)); self.lt = np.full((L, C), 1e-3)
        self.mesh_base = mesh_base                                                        # mesh_base=0: no own columns of the byte before
        self.mesh = mesh                                                                  # mesh=K: regions in a random graph (see step)
        if mesh:
            mr = np.random.default_rng(seed + 55)
            if mesh_dag:                                                                   # mesh_dag=1: a random graph without loops: region d reads only
                if mesh_dag == 2:                                                          # mesh_dag=2: the region just before, and K-1 random earlier ones
                    self.msrc = [np.r_[d - 1, mr.choice(d - 1, min(mesh - 1, d - 1), replace=False)].astype(int) if d else np.zeros(0, int) for d in range(L)]
                else:
                    self.msrc = [mr.choice(d, min(mesh, d), replace=False) if d else np.zeros(0, int) for d in range(L)]   # regions before it
            else:
                self.msrc = [mr.choice([j for j in range(L) if j != d], min(mesh, L - 1), replace=False) for d in range(L)]
            self.mlsrc = mr.integers(0, n_values, (L, C, lines_in)); self.mlw = mr.random((L, C, lines_in))
        self.fsrc = np.stack([rng.choice(C, (C, fan_in)) for _ in range(L)])
        self.fw = rng.random((L, C, fan_in))
        self.lsrc = rng.integers(0, n_values, (C, lines_in)); self.lw = rng.random((C, lines_in))
        self.rsrc = np.zeros((L, C, M, S), np.int64); self.rw = np.zeros((L, C, M, S))
        self.gsrc = np.zeros((L, C, M, S), np.int64) + (np.arange(L) * 2 * C * M)[:, None, None, None]
        self.use = np.zeros((L, C, M)); self.use[:, :, 0] = np.inf
        self.rel = np.full((L, C, M), 1.0)
        self.cells = np.zeros((L, C, M)); self.cols = np.zeros((L, C)); self.hold = np.zeros((L, C, M))
        self.predicted = 0.0; self.surprise = np.zeros(L, bool)
        self.par = _P(eta, eta_minus, w_on, w_new, th_on, th_match, s_th, lam, n_grow, burst, ctx, 1, recall, nodef)
        _lib.seed_rng(ctypes.c_uint64(seed + 7))
        self._cells = np.zeros(L * C * M, np.uint8); self._sur = np.zeros(L, np.uint8); self._st = np.zeros(1)

    def forward(self, byte):                                                        # as net5
        if self.mesh:                                                               # mesh: every region takes its columns from the lines
            F = (self.mlw * (self.mlsrc == byte)).sum(2); cols = np.zeros_like(F)
            for d in range(self.L):
                w_ = np.argpartition(-F[d], self.k - 1)[: self.k]; cols[d, w_] = 1.0; cols[d, F[d] <= self.theta] = 0.0
            return cols
        F = np.empty((self.L, self.C)); F[0] = (self.lw * (self.lsrc == byte)).sum(1)
        cols = np.zeros_like(F)
        for d in range(self.L):
            if d:
                src = cols[d - 1] if self.sweep else self.cols[d - 1]
                F[d] = (self.fw[d] * src[self.fsrc[d]]).sum(1)
            w_ = np.argpartition(-F[d], self.k - 1)[: self.k]
            cols[d, w_] = 1.0; cols[d, F[d] <= self.theta] = 0.0
        return cols

    def step(self, byte, learn=True):
        L, C, M = self.L, self.C, self.M
        cols = self.forward(byte)
        if self.slow:                                                                     # each column's slow trace and its long-term mean (its own state);
            self.tr += (cols - self.tr) / self.slow; self.lt += (cols - self.lt) / self.slow_lt   # the code: the SLOW_K columns most above their mean
            top = np.argsort(-(self.tr / self.lt), 1)[:, : self.slow_k]
            sc = np.zeros((L, C)); np.put_along_axis(sc, top, 1.0, 1); self.hold = np.repeat(sc[:, :, None], M, 2)
        prevc = np.repeat(self.cols[:, :, None], M, 2) if self.ctx_mode else self.cells
        if self.mesh:                                                                     # mesh: the byte before (own columns, every cell) and the
            base = np.repeat(self.cols[:, :, None], M, 2)                                 # cells of K other regions at the byte before
            prevc = np.zeros_like(self.cells)
            for d in range(L):
                if len(self.msrc[d]): prevc[d] = self.cells[self.msrc[d]].max(0)
                if self.mesh_base or not len(self.msrc[d]): prevc[d] = np.maximum(prevc[d], base[d])   # a region with no sources reads the byte before
        if self.ctx_mode == 2 and not self.mesh: prevc = np.concatenate([prevc[:1], self.cells[:-1]], 0)
        ctx = np.ascontiguousarray(np.concatenate([prevc.reshape(L, -1), self.hold.reshape(L, -1)], 1) > 0, np.uint8)
        cu = np.ascontiguousarray(cols > 0, np.uint8)
        self.par.learn = 1 if learn else 0
        _lib.step(L, C, M, self.S, _p(self.gsrc), _p(self.rsrc), _p(self.rw), _p(ctx), _p(cu), _p(self.use), _p(self.rel),
                  _p(self._cells), _p(self._sur), _p(self._st), ctypes.byref(self.par))
        cells = self._cells.reshape(L, C, M).astype(float); surprise = self._sur.astype(bool)
        if self.use_hold: self.hold = np.where(surprise[:, None, None], cells, self.hold)
        self.predicted = float(self._st[0]); self.surprise = surprise
        self.cells, self.cols = cells, cols
        return cells, cols
