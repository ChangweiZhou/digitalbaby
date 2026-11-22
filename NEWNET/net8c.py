"""net8 with its per-byte step in C (net8c.c), for the configuration net9 uses. Same definition, constructor and attributes as
net8.py (a subclass: the strengths at birth are net8's own draws). The C step is used when fb=none, gran > 0, k=0, rec=0,
send_y=0, norm_rec=0 (any tau_max, center, out_var, eta_o, rule, n_values, fbm=teach|yesno, learn on or off); otherwise net8's
numpy step is used unchanged (self.c says which). Differences from net8 in the C configuration: exp is the C library's (numpy's
may differ in the last bit), O sums only the firing expansion cells (BLAS sums all of them: last-bit differences), ties at the
expansion's threshold go to the lowest index; the traces E are not kept (with fb=none and an expansion net8 never reads them).
Storage: A, Wc, Wg, Wd and gline are kept transposed (contiguous for what one byte reads); each attribute is that array seen as
net8 has it (a view), so reading, copying or writing into net.A[...] works as with net8. Check: check_net8c.py. Plan: SLOWC_PLAN.md."""
import ctypes, os, subprocess
import numpy as np
from net8 import Tissue as _Net8

_here = os.path.dirname(os.path.abspath(__file__))
_so = os.path.join(_here, "libnet8c.so")
if not os.path.exists(_so) or os.path.getmtime(_so) < os.path.getmtime(os.path.join(_here, "net8c.c")):
    subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", _so, os.path.join(_here, "net8c.c"), "-lm"])
_lib = ctypes.CDLL(_so)
_I, _D, _P = ctypes.c_int, ctypes.c_double, ctypes.c_void_p
_lib.learn.argtypes = [_I, _I, _I, _I, _I, _I, _I, _D, _D] + [_P] * 7
_lib.forward.argtypes = [_I] * 8 + [_D, _I] + [_P] * 23


def _p(a): return a.ctypes.data


def _f(a):                                                                         # an attribute as C reads it (contiguous float64)
    return a if a.dtype == np.float64 and a.flags.c_contiguous else np.ascontiguousarray(a, np.float64)


class Tissue(_Net8):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.c = (self.fb == "none" and self.gran > 0 and not self.k and not self.rec and not self.send_y and not self.norm_rec
                  and self.F <= 256 and self.gsrc.shape[1] <= 256)
        if self.c:
            self.src = np.ascontiguousarray(self.src, np.int64); self.gsrc = np.ascontiguousarray(self.gsrc, np.int64)
            G = self.gran
            self._yg = np.zeros(self.N); self._zg = np.zeros(G); self._sc = np.zeros(G)
            self._act = np.zeros(G, np.int32); self._e = np.zeros(self.V)
            for k in ("A", "Wc", "Wg", "Wd", "gline"): self._t(k)                      # stored transposed: A by sender [G, V], the byte lines
                                                                                           # by byte [V, N] / [V, G]; the attribute is the same
                                                                                           # array seen as net8 has it (a view), so changes are shared

    def _t(self, k):                                                               # attribute k as C reads it: its transpose, contiguous
        a = getattr(self, k)                                                       # (made again if the attribute was replaced)
        if not (a.T.flags.c_contiguous and a.dtype == np.float64):
            a = np.ascontiguousarray(a.T, np.float64).T; setattr(self, k, a)
        return a.T

    def step(self, byte, learn=True):
        if not self.c: return super().step(byte, learn)
        if learn and self.byte is not None: self._learn(byte)
        self.yprev_byte = self.yprev; self.yprev = self.y.copy(); self.hprev_byte = self.h.copy(); self.t += 1
        N, G, V = self.N, self.gran, self.V
        hp = _f(self.h); self.m = _f(self.m); self.vy = _f(self.vy)
        h = np.empty(N); y = np.empty(N) if self.center else h; z = np.empty(G); O = np.empty(V)
        pb = -1 if self.prev_byte is None else int(self.prev_byte)
        _lib.forward(N, self.F, V, G, self.gsrc.shape[1], self.gk, int(byte), pb, float(self.center), 1 if self.out_var else 0,
                     _p(self.src), _p(self._t('Wc')), _p(self._t('Wg')), _p(self.Uc), _p(self.Ug), _p(self.bc), _p(self.bg), _p(self._t('Wd')),
                     _p(hp), _p(h), _p(self.m), _p(y), _p(self.vy), _p(self.gsrc), _p(self.gw), _p(self._t('gline')), _p(self._t('A')),
                     _p(z), _p(O), _p(self._yg), _p(self._zg), _p(self._sc), _p(self._act))
        self.prev_byte = self.byte; self.h = h; self.byte = byte
        self.y = y; self.z = z; self.O = O
        return h

    def _learn(self, nxt):
        if not self.c: return super()._learn(nxt)
        yes = self.fbm == "yesno" and self.gave is not None
        self.vy = _f(self.vy); O, z, y = _f(self.O), _f(self.z), _f(self.y)               # (held here while C reads them)
        _lib.learn(self.V, self.gran, self.N, int(nxt), int(self.gave) if yes else -1, 1 if yes else 0, 1 if self.rule == "gated" else 0,
                   float(self.eta_o), float(self.out_var), _p(O), _p(z), _p(y), _p(self.vy), _p(self._t('A')), _p(self._act), _p(self._e))
