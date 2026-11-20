#!/usr/bin/env python3
"""PLAN_10.md: the kinds of cells found on 20 November, together.
    python3 all.py [without=companions|relay|since|pool]
T2 (the table's cells, K = 6, and the cells that step once a word) with: companions whose counts fade over 10,000 bytes (two.py); the relay system of nano/loop as one more
sender (relay.py); cells that time how long ago each byte value last came, and a pool of cells keyed on two or three of the last eight bytes (place.py), born with no trust."""
import ctypes, json, os, subprocess, sys, time
import numpy as np
import score, table, words, trust, two, place, relay
HERE = os.path.dirname(os.path.abspath(__file__)); S = 65; NL = 40; ETA, E0, H = 0.00005, 0.02, 10000.0
PD = ctypes.POINTER(ctypes.c_double); PF = ctypes.POINTER(ctypes.c_float); PI = ctypes.POINTER(ctypes.c_int64); PY = ctypes.POINTER(ctypes.c_int32)
SRC = two.SRC.replace("void two_steps(", "void all_steps(").replace("s[40][S]; int lev[40];", "s[192][S]; int lev[192];").replace("const double *half, double *last, double clock0) {", "const double *half, double *last, double clock0, const float *X, const int32_t *xlev) {")
SRC = SRC.replace("        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];", "        if (X) { const float *sx = X + t * S; const double tk = T[K1 * NL + xlev[t]]; for (int v = 0; v < S; v++) a[v] += tk * sx[v]; }      /* one more sender whose message is given (relay.py) */\n        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];")
SRC = SRC.replace("        if (learn) {\n", "        if (learn) {\n            if (X) { const float *sx = X + t * S; double g = sx[y[t]]; for (int v = 0; v < S; v++) g -= p[v] * sx[v]; T[K1 * NL + xlev[t]] += eta * g; }\n")
assert "s[192][S]" in SRC and SRC.count("const float *sx") == 2 and "all_steps" in SRC

def kern():
    so = os.path.join(HERE, "kern_a.so"); src = os.path.join(HERE, "kern_a.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, src, "-lm"])
    L = ctypes.CDLL(so); L.all_steps.argtypes = [ctypes.c_int64, ctypes.c_int64, PI, PY, PF, PF, PF, PD, ctypes.c_double, ctypes.c_double, ctypes.c_int, PD, PD, PD, ctypes.c_double, PF, PY]; L.all_steps.restype = None; return L

class SysAll:
    def __init__(s, x, without=()):
        K = 6; keys = words.word_keys(x); born0 = 0
        if "since" not in without: keys += [(place.since(x, v) * S + x).astype(np.int64) for v in range(S)]; born0 += S
        if "pool" not in without: k, _ = place.pool(x); keys += k; born0 += len(k)
        ids, ncell, kinds, _ = words.make_ids(x, keys, K, len(x), len(x), K, 1); ntab = ncell - sum(kinds); tot = ncell; nk = ids.shape[1]; comp = "companions" not in without
        if comp: ids = np.hstack([ids, np.where(ids[:, :K + 1] >= 0, ids[:, :K + 1] + ncell, -1)]); tot = ncell + ntab
        s.x, s.K, s.ids, s.K1, s.L = x, K, np.ascontiguousarray(ids), ids.shape[1], kern(); assert s.K1 <= 192; s.half = np.zeros(s.K1)
        if comp: s.half[-(K + 1):] = H
        s.C = np.zeros((tot, S), np.float32); s.N = np.zeros(tot, np.float32); s.U = np.zeros(tot, np.float32); s.last = np.zeros(tot); s.T = np.full((s.K1 + 1, NL), 0.25); s.T[10:10 + born0] = 0.0; s.clock = 0.0; s.X = s.xlev = None
        if "relay" not in without:
            f = os.environ.get("RELAY_CACHE", "")
            if f and os.path.exists(f + ".npy"): s.X = np.load(f + ".npy"); s.xlev = np.load(f + "_lev.npy")
            else:
                s.X, s.xlev, s.alone = relay.opinions(0)
                if f: np.save(f + ".npy", s.X); np.save(f + "_lev.npy", s.xlev)
    def go(s, lo, hi, learn, want_p=False):
        ids = np.ascontiguousarray(s.ids[lo - s.K:hi - s.K]); y = np.ascontiguousarray(s.x[lo:hi].astype(np.int32)); P = np.zeros((hi - lo, S)) if want_p else None
        X = np.ascontiguousarray(s.X[lo:hi]) if s.X is not None else None; xl = np.ascontiguousarray(s.xlev[lo:hi]) if s.X is not None else None
        s.L.all_steps(hi - lo, s.K1, ids.ctypes.data_as(PI), y.ctypes.data_as(PY), s.C.ctypes.data_as(PF), s.N.ctypes.data_as(PF), s.U.ctypes.data_as(PF), s.T.ctypes.data_as(PD), ETA, E0, int(learn),
                      P.ctypes.data_as(PD) if want_p else None, s.half.ctypes.data_as(PD), s.last.ctypes.data_as(PD), s.clock, X.ctypes.data_as(PF) if X is not None else None, xl.ctypes.data_as(PY) if X is not None else None)
        if learn: s.clock += hi - lo
        return P

def run(without):
    x, cut, _ = score.load(); t0 = time.process_time(); s = SysAll(x, without); s.go(6, cut, True); r = dict(score.score(s.go(cut, len(x), False, True))); on = score.score(s.go(cut, len(x), True, True))
    r.update(without=list(without), kinds=int(s.K1), cells=int((s.N > 0).sum()), going_on_nats=on["nats"], going_on_right=on["right"], cpu=round(time.process_time() - t0, 1)); return r

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:]); w = tuple(v for v in kw.get("without", "").split(",") if v); r = run(w); tag = "all" + ("_without_" + "_".join(w) if w else "")
    json.dump(r, open(os.path.join(HERE, "res", tag + ".json"), "w")); print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "n"}, flush=True)
