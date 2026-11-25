"""Python side of lt7.c (local tissue from the owner's definition of a cell, 25 Nov 2026). See the header of lt7.c.
A candidate is a dict of options; missing options take DEF. scale multiplies the tissue, inhibitory and relay cell counts."""
import ctypes, os, subprocess
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_so = os.path.join(_here, "liblt7.so"); _c = os.path.join(_here, "lt7.c")
if not os.path.exists(_so) or os.path.getmtime(_so) < os.path.getmtime(_c):
    _tmp = _so + ".%d.tmp" % os.getpid()
    subprocess.check_call(["cc", "-O3", "-march=native", "-shared", "-fPIC", "-o", _tmp, _c, "-lm"]); os.replace(_tmp, _so)
_lib = ctypes.CDLL(_so)
_lib.lt7_new.restype = ctypes.c_void_p; _lib.lt7_new.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
for f in ("lt7_snapshot", "lt7_restore"): getattr(_lib, f).argtypes = [ctypes.c_void_p]
_lib.lt7_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_lib.lt7_audit.argtypes = [ctypes.c_void_p, ctypes.c_void_p]; _lib.lt7_fanin.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.lt7_set_norelay.argtypes = [ctypes.c_void_p, ctypes.c_int]

GI = ["L", "KIN", "KOUT", "KFB", "ntis", "ninh", "nrly", "S", "H", "seed"]
GD = ["r", "r_out", "r_fb", "tau_lo", "tau_hi", "rho_lo", "rho_hi", "tau_r", "eta_th", "th0", "w0", "wi0", "d0", "g_d", "p_off", "p_dis",
      "lam", "w_min", "eta_h", "eta_e", "eta_o", "eta_b", "eta_d", "eta_i", "eta_q", "eta_r", "eps", "eta_t", "g_acc", "b0", "tau_el", "tr_out", "kap", "mu", "p_ret", "B_exc", "B_inh", "B_dis", "B_out", "B_rd", "B_rly"]
DEF = dict(KIN=2, KOUT=16, KFB=8, ntis=4096, ninh=1024, nrly=1024, S=3, H=2, seed=1,
           r=0.12, r_out=0.25, r_fb=0.15, tau_lo=1.0, tau_hi=16.0, rho_lo=0.02, rho_hi=0.1, tau_r=500.0, eta_th=0.01, th0=0.3,
           w0=0.4, wi0=0.3, d0=0.2, g_d=0.3, p_off=0.3, p_dis=0.3, lam=2e-5, w_min=0.01, eta_h=0.02, eta_e=0.005, eta_o=0.5,
           eta_b=0.01, eta_d=0.05, eta_i=0.01, eta_q=0.05, eta_r=0.05, eps=0.03, eta_t=0.01, g_acc=5.0, b0=-4.0, tau_el=4.0, tr_out=8.0, kap=0.01, mu=1e-6, p_ret=0.0, B_exc=3.0, B_inh=2.0, B_dis=2.0, B_out=20.0, B_rd=2.0, B_rly=3.0)
# input passages: (offset, length); the lags are their union. Short passages from different places, not one window.
PASSAGES = [(d, 1) for d in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192)]   # per-synapse delays, log-spaced


def lags_of(passages): return sorted({o + k for o, l in passages for k in range(l)})


AUD = ["syn_exc", "syn_inh", "syn_dis", "syn_out", "syn_rly", "syn_rd", "locality_violations", "fanin_tis_max", "fanin_tis_mean_x1000",
       "fanin_out_max", "tis_with_input", "tis_rate_mean_x1e6", "tis_rate_p10_x1e6", "tis_rate_p90_x1e6", "made", "cut", "tis_reading_relay",
       "relay_with_input", "syn_touching_answer", "syn_into_interface", "fanout_tis_max", "tis_silent_x1000"]


class Net7:
    def __init__(self, cand=None, scale=1):
        o = dict(DEF, **(cand or {})); lags = lags_of(o.get("passages", PASSAGES)); o["L"] = len(lags)
        for k in ("ntis", "ninh", "nrly"): o[k] = int(o[k] * scale)
        self.o = o; gi = np.array([int(o[k]) for k in GI], np.int32); gd = np.array([float(o[k]) for k in GD])
        lg = np.array(lags, np.int32); self.lags = lags
        self._keep = (gi, gd, lg); self.h = _lib.lt7_new(gi.ctypes.data, gd.ctypes.data, lg.ctypes.data)

    def run(self, x, learn=True):
        x = np.ascontiguousarray(x, np.uint8); st = np.zeros(3)
        _lib.lt7_run(self.h, x.ctypes.data, len(x), 1 if learn else 0, st.ctypes.data)
        return st[0] / max(st[2], 1), st[1] / max(st[2], 1)

    def test(self, x):
        _lib.lt7_snapshot(self.h); r = self.run(x, False); _lib.lt7_restore(self.h); return r

    def audit(self):
        a = np.zeros(22, np.int64); _lib.lt7_audit(self.h, a.ctypes.data); return dict(zip(AUD, a.tolist()))

    def fanin(self):
        a = np.zeros(64, np.int64); _lib.lt7_fanin(self.h, a.ctypes.data); return a

    def norelay(self, v=True): _lib.lt7_set_norelay(self.h, 1 if v else 0)


_lib.lt7_run_trace.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]


def run_trace(net, x, learn=True):
    """per-byte loss in nats (-1 where no prediction yet)"""
    x = np.ascontiguousarray(x, np.uint8); st = np.zeros(3); per = np.zeros(len(x), np.float32)
    _lib.lt7_run_trace(net.h, x.ctypes.data, len(x), 1 if learn else 0, st.ctypes.data, per.ctypes.data); return per


def test_trace(net, x):
    _lib.lt7_snapshot(net.h); per = run_trace(net, x, False); _lib.lt7_restore(net.h); return per
_lib.lt7_set_biasonly.argtypes = [ctypes.c_void_p, ctypes.c_int]; _lib.lt7_outstats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def biasonly_test(net, x):
    _lib.lt7_set_biasonly(net.h, 1); r = net.test(x); _lib.lt7_set_biasonly(net.h, 0); return r


def outstats(net):
    a = np.zeros(7, np.int64); _lib.lt7_outstats(net.h, a.ctypes.data)
    return dict(zip(["syn", "outputs_with_syn", "mean_w_x1000", "max_w_x1000", "mean_bias_x1000", "mean_drive_x1000", "driven"], a.tolist()))
