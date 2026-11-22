"""net7 with the context kernel by events (net7e.c, see EVENTK_PLAN.md): for every source, the list of the slots that read it
(an inverted index kept on the Tissue as ix_head, ix_nxt, ix_prv); each byte only the slots of the active sources are visited.
The same constructor, attributes, definition and results as net7.py (check_net7e.py); the random sample of new sources uses
net7e.c's own copy of net7.c's generator, drawn in the same order. The index is built from (rw, gsrc) at the first step; if rw,
rsrc or gsrc are set from outside after that, call reindex()."""
import ctypes, os, subprocess
import numpy as np
import net7

_here = os.path.dirname(os.path.abspath(__file__))
_so = os.path.join(_here, "libnet7e.so")
if not os.path.exists(_so) or os.path.getmtime(_so) < os.path.getmtime(os.path.join(_here, "net7e.c")):
    subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", _so, os.path.join(_here, "net7e.c")])
_lib = ctypes.CDLL(_so)
_P, _p = net7._P, net7._p


class Tissue(net7.Tissue):                                                           # forward is net7's; __init__ is net7's but seeds net7e.c's generator
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
        assert L * C * M * S < 2 ** 31, 'slot index is int32'
        self.ix_head = np.full(L * 2 * C * M, -1, np.int32); self.ix_nxt = np.full(L * C * M * S, -1, np.int32)
        self.ix_prv = np.full(L * C * M * S, -2, np.int32); self._ix_built = False   # the inverted index (built at the first step)
        self._cells = np.zeros(L * C * M, np.uint8); self._sur = np.zeros(L, np.uint8); self._st = np.zeros(1)

    def _context(self):
        """net7's context array (prev part, then hold part, > 0), built from the active cells and columns only: the same values"""
        L, C, M = self.L, self.C, self.M; CM = C * M
        ctx = np.zeros((L, 2 * CM), np.uint8)
        cnz = self.cells.reshape(L, CM) > 0; cells_on = [np.flatnonzero(cnz[d]) for d in range(L)]
        cols_on = [(np.flatnonzero(self.cols[d] > 0)[:, None] * M + np.arange(M)).ravel() for d in range(L)]   # every cell of an active column
        for d in range(L):
            if self.mesh:                                                             # as net7: K other regions' cells, plus the own columns if mesh_base or no source
                for e in self.msrc[d]: ctx[d, cells_on[e]] = 1
                if self.mesh_base or not len(self.msrc[d]): ctx[d, cols_on[d]] = 1
            elif self.ctx_mode == 2: ctx[d, cols_on[0] if d == 0 else cells_on[d - 1]] = 1
            elif self.ctx_mode: ctx[d, cols_on[d]] = 1
            else: ctx[d, cells_on[d]] = 1
        ctx[:, CM:] = self.hold.reshape(L, CM) > 0
        return ctx

    def reindex(self):
        _lib.build_index(self.L, self.C, self.M, self.S, _p(self.gsrc), _p(self.rw), _p(self.ix_head), _p(self.ix_nxt), _p(self.ix_prv))
        self._ix_built = True

    def step(self, byte, learn=True):
        L, C, M = self.L, self.C, self.M
        cols = self.forward(byte)
        if self.slow:                                                                     # each column's slow trace and its long-term mean (its own state);
            self.tr += (cols - self.tr) / self.slow; self.lt += (cols - self.lt) / self.slow_lt   # the code: the SLOW_K columns most above their mean
            top = np.argsort(-(self.tr / self.lt), 1)[:, : self.slow_k]
            sc = np.zeros((L, C)); np.put_along_axis(sc, top, 1.0, 1); self.hold = np.repeat(sc[:, :, None], M, 2)
        ctx = self._context()
        cu = np.ascontiguousarray(cols > 0, np.uint8)
        self.par.learn = 1 if learn else 0
        if not self._ix_built: self.reindex()
        _lib.step(L, C, M, self.S, _p(self.gsrc), _p(self.rsrc), _p(self.rw), _p(ctx), _p(cu), _p(self.use), _p(self.rel),
                  _p(self._cells), _p(self._sur), _p(self._st), ctypes.byref(self.par),
                  _p(self.ix_head), _p(self.ix_nxt), _p(self.ix_prv))
        cells = self._cells.reshape(L, C, M).astype(float); surprise = self._sur.astype(bool)
        if self.use_hold: self.hold = np.where(surprise[:, None, None], cells, self.hold)
        self.predicted = float(self._st[0]); self.surprise = surprise
        self.cells, self.cols = cells, cols
        return cells, cols
