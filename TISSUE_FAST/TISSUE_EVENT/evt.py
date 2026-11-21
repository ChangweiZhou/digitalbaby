"""E63, the tissue's program that works only where something changes: the Python side of the compiled core (evt_core.c).

The birth, the stream and the readings are E63's (TISSUE_STEP1/tissue_step1.py, imported unchanged); the tissue's steps are computed
by evt_core.c. EvTissue has the interface of tissue_step1.Tissue that tissue_step1.Run uses (step, run_byte, x, th, k, live, counts,
distances, made, given_up), so E63's records and readings are used without change.

Switches (section 3, point 5 of the task): floor (0 or a value), tol (0 or a value), trace_sender (False: a trace for each
connection), strength_byte (False: strengths at every step). With all four off the program is meant to compute E63's tissue.
"""
import ctypes
import subprocess
import sys
from pathlib import Path
from random import Random

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "TISSUE_STEP1"))
import tissue_step1 as ts  # noqa: E402

SO = HERE / "evt_core.so"
SRC = HERE / "evt_core.c"
CFLAGS = ["-O3", "-march=native", "-ffp-contract=off", "-shared", "-fPIC"]


def build():
    if not SO.exists() or SO.stat().st_mtime < SRC.stat().st_mtime:
        subprocess.run(["gcc", *CFLAGS, str(SRC), "-o", str(SO), "-lm"], check=True)


build()
L = ctypes.CDLL(str(SO))
_d, _i, _u8, _i64, _u32, _i16 = (np.ctypeslib.ndpointer(t, flags="C_CONTIGUOUS") for t in
                                 (np.float64, np.int32, np.uint8, np.int64, np.uint32, np.int16))
L.evt_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double] + [ctypes.c_int] * 8
L.evt_set_senders.argtypes = [_d, _u8, _u8]
L.evt_set_cells.argtypes = [_u8, _u8]
L.evt_set_members.argtypes = [_i, _i]
L.evt_set_drivers.argtypes = [_i, ctypes.c_int]
L.evt_set_cell_state.argtypes = [_d, _d, _d, _d, ctypes.c_void_p]
L.evt_add_connection.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int64]
L.evt_add_connection.restype = ctypes.c_int
L.evt_set_life_cells.argtypes = [_u8]
L.evt_set_line.argtypes = [ctypes.c_int, ctypes.c_double]
L.evt_set_rng.argtypes = [_u32, ctypes.c_int]
L.evt_get_rng.argtypes = [_u32, ctypes.POINTER(ctypes.c_int)]
L.evt_set_astar.argtypes = [ctypes.c_double]
L.evt_set_withdraw_use.argtypes = [ctypes.c_int, _u8, ctypes.c_int]
L.evt_withdrawn_by_use.restype = ctypes.c_int64
L.evt_step.argtypes = [ctypes.c_int]
L.evt_step.restype = ctypes.c_int
L.evt_run_bytes.argtypes = [_u8, ctypes.c_int, _i16]
for f in ("evt_k", "evt_made", "evt_given_up"):
    getattr(L, f).restype = ctypes.c_int64
L.evt_nlive.restype = ctypes.c_int
for f in ("evt_ptr_x", "evt_ptr_th", "evt_ptr_v", "evt_ptr_A", "evt_ptr_m"):
    getattr(L, f).restype = ctypes.POINTER(ctypes.c_double)
L.evt_counters.argtypes = [_d]
L.evt_live.argtypes = [_i, _i, _d, _d, _i64, _d]
L.evt_sleep_stats.argtypes = [_d]
L.evt_set_cap.argtypes = [ctypes.c_int]
L.evt_set_turnover.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
L.evt_trial_counts.argtypes = [_i64]
L.evt_set_gain.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]
L.evt_gains.argtypes = [_d]
L.evt_set_norm.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double]
L.evt_norm_calls.restype = ctypes.c_int64
L.evt_cells.argtypes = [_d, _d, _d, _d, ctypes.c_void_p, _i64]
L.evt_events_made.argtypes = [_i64, ctypes.c_int]
L.evt_events_made.restype = ctypes.c_int
L.evt_events_gone.argtypes = [_i64, ctypes.c_int]
L.evt_events_gone.restype = ctypes.c_int
COUNTER_NAMES = ("cell_updates", "senders_passing_on", "visits_passing_on", "visits_strengths", "visits_life", "steps",
                 "unused", "lazy_updates")


def view(ptr, n):
    return np.ctypeslib.as_array(ptr, shape=(n,))


class EvTissue:
    """One tissue of E63 (first-form, coincidence or fixed; formula 11 as written for the 512-byte tissue), computed by evt_core.c."""

    def __init__(self, seed, rule, f11_written=False, floor=0.0, tol=0.0, trace_sender=False, strength_byte=False, B=None,
                 astar=ts.ASTAR, withdraw_by_use=False, use_exempt_feedback=False, long_frac=0.0, long_r=0.5):
        assert rule in ("first", "coincidence", "fixed")
        self.seed, self.rule, self.f11w, self.by_use = seed, rule, f11_written, False
        self.switches = dict(floor=floor, tol=tol, trace_sender=bool(trace_sender), strength_byte=bool(strength_byte), astar=float(astar),
                             withdraw_by_use=bool(withdraw_by_use), use_exempt_feedback=bool(use_exempt_feedback))
        B = B if B is not None else ts.birth(seed)
        if long_frac > 0:                                    # plan 5: a share of the excitatory cells reach farther (given at birth)
            import copy
            B = copy.deepcopy(B)
            P = B["P"]
            exc = [j for j in range(ts.N) if j not in B["fbset"] and B["sig"][j] > 0]
            rl = Random(seed + 700000)
            longs = sorted(rl.sample(exc, int(round(long_frac * len(exc)))))
            r2 = long_r * long_r
            for i in range(ts.N):
                if i in B["fbset"]:
                    continue
                have = set(B["members"][i])
                xi, yi = P[i]
                B["members"][i] = B["members"][i] + [j for j in longs if j != i and j not in have and (P[j][0] - xi) ** 2 + (P[j][1] - yi) ** 2 <= r2]
            B["long_cells"] = longs
        self.B = B
        self.kind = B["kind"]; self.sig = B["sig"]
        self.isfb = self.kind == ts.FBK
        self.isinner = self.kind == ts.INNER
        self.members = B["members"]
        self.driver = np.array(B["driver"], dtype=np.int64)
        N, NZ = ts.N, ts.NZ
        life = rule != "fixed"
        L.evt_set_astar(float(astar))                       # (T) a*: before evt_init, which starts the sliding threshold at a*
        L.evt_init(N, NZ, float(floor), float(tol), int(trace_sender), int(strength_byte), int(life), int(f11_written),
                   int(rule == "coincidence"), ts.S, ts.LINE0, ts.NIN)
        lrn = np.zeros(NZ, np.uint8); lrn[:N] = 1
        ex = np.zeros(NZ, np.uint8); ex[ts.YES_LINE:] = 1
        L.evt_set_senders(np.ascontiguousarray(B["sig"], np.float64), lrn, ex)
        fbs = np.zeros(NZ, np.uint8); fbs[:N] = self.isfb
        L.evt_set_withdraw_use(int(withdraw_by_use), fbs, int(use_exempt_feedback))   # (W)
        rk = np.zeros(N, np.uint8)
        if rule == "first":
            rk[~self.isfb] = 1
        elif rule == "coincidence":
            rk[~self.isfb] = 1
            rk[self.isinner] = 2
        L.evt_set_cells(rk, (~self.isfb).astype(np.uint8))
        ptr = np.zeros(N + 1, np.int32)
        idx = []
        for i in range(N):
            idx += self.members[i]
            ptr[i + 1] = len(idx)
        L.evt_set_members(ptr, np.array(idx, np.int32))
        L.evt_set_drivers(np.array(B["driver"], np.int32), len(B["driver"]))
        for i in range(N):                                   # the order of E63's program: by receiver, then the feedback lines
            for j in B["src"][i]:
                L.evt_add_connection(j, i, ts.START_W, 0.0, 0)
        for i in B["YES"]:
            L.evt_add_connection(ts.YES_LINE, i, 1.0, 0.0, 0)
        for i in B["NO"]:
            L.evt_add_connection(ts.NO_LINE, i, 1.0, 0.0, 0)
        L.evt_start()
        if life:
            L.evt_set_life_cells((~self.isfb).astype(np.uint8))
        self.rl = Random(seed + 500000)                      # the life generator (formula 12): its state is handed to the core
        st = self.rl.getstate()
        L.evt_set_rng(np.array(st[1][:624], np.uint32), int(st[1][624]))
        self.x = view(L.evt_ptr_x(), N)
        self.th = view(L.evt_ptr_th(), N)
        self.audit = None

    # ---------------------------------------------------------------------------------------- the interface of tissue_step1.Tissue
    @property
    def k(self):
        return int(L.evt_k())

    @property
    def made(self):
        return int(L.evt_made())

    @property
    def withdrawn_by_use(self):
        return int(L.evt_withdrawn_by_use())

    @property
    def given_up(self):
        return int(L.evt_given_up())

    def step(self, byte, hook=None):
        r = L.evt_step(int(byte))
        if hook is not None:
            hook(self, self.k - 1, None, None, None, self.x)
        if r == -2:
            return None
        return r

    def run_byte(self, byte, hook=None):
        em = None
        for _ in range(ts.S):
            r = self.step(byte, hook)
            if r is not None:
                em = r
        return em

    def run_bytes(self, bytes_):
        b = np.ascontiguousarray(bytes_, np.uint8)
        em = np.zeros(len(b), np.int16)
        L.evt_run_bytes(b, len(b), em)
        return em

    def rng_state(self):
        s = np.zeros(624, np.uint32); pos = ctypes.c_int()
        L.evt_get_rng(s, ctypes.byref(pos))
        return (3, tuple(int(q) for q in s) + (pos.value,), None)

    def live_full(self):
        n = int(L.evt_nlive())
        snd = np.zeros(n, np.int32); rcv = np.zeros(n, np.int32); w = np.zeros(n); e = np.zeros(n); k0 = np.zeros(n, np.int64)
        ub = np.zeros(n)
        L.evt_live(snd, rcv, w, e, k0, ub)
        return snd.astype(np.int64), rcv.astype(np.int64), w, e, k0, ub

    def live(self):
        snd, rcv, w, e, k0, ub = self.live_full()
        return snd, rcv, w, k0

    def cells(self):
        N = ts.N
        v = np.zeros(N); p = np.zeros(N); th = np.zeros(N); ab = np.zeros(N); vt = np.zeros(N); kl = np.zeros(N, np.int64)
        L.evt_cells(v, p, th, ab, vt.ctypes.data if self.rule == "coincidence" else None, kl)
        return dict(v=v, p=p, th=th, abar=ab, vt=vt if self.rule == "coincidence" else None, klast=kl, x=self.x.copy())

    @property
    def srcset(self):
        snd, rcv, _, _ = self.live()
        out = [set() for _ in range(ts.N)]
        for s_, r_ in zip(snd.tolist(), rcv.tolist()):
            if s_ < ts.YES_LINE:
                out[r_].add(s_)
        return out

    def counts(self, used_values):
        self._srcset_cache = self.srcset
        return _counts(self, used_values)

    distances = ts.Tissue.distances

    def counters(self):
        c = np.zeros(8)
        L.evt_counters(c)
        return dict(zip(COUNTER_NAMES, c.tolist()))

    def events(self):
        nm = L.evt_events_made(np.zeros(0, np.int64), 0)
        m = np.zeros(3 * nm, np.int64); L.evt_events_made(m, nm)
        ng = L.evt_events_gone(np.zeros(0, np.int64), 0)
        g = np.zeros(4 * ng, np.int64); L.evt_events_gone(g, ng)
        return m.reshape(-1, 3), g.reshape(-1, 4)


class _SrcsetView:
    """counts() of tissue_step1.Tissue reads self.srcset several times; it is built once per call."""


def _counts(t, used_values):
    class Shim:
        pass
    s = Shim()
    s.live, s.isfb, s.kind, s.sig, s.members, s.B = t.live, t.isfb, t.kind, t.sig, t.members, t.B
    s.srcset = t._srcset_cache
    s.given_up, s.made, s.withdrawn_by_use, s.k = t.given_up, t.made, t.withdrawn_by_use, t.k
    return ts.Tissue.counts(s, used_values)
