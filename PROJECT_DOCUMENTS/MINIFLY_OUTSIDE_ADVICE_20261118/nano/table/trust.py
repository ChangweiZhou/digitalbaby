#!/usr/bin/env python3
"""PLAN_3.md: cells that count; an answering cell that learns how far to trust a cell of each kind and experience.
    python3 trust.py arm=0|1|2|3|4 [eta=...] [e=...]      settings chosen on the last twentieth of the stream unless both are given
    python3 trust.py check                                the compiled loop against numpy on a prefix
Arms: 0 the table's cells, K = 6; 1 as 0 with one trust for each kind; 2 as 0 with the cells that step once a word; 3 as 2 with the blind clock; 4 as 0 with K = 8."""
import ctypes, json, os, subprocess, sys, time
import numpy as np
import score, table, words
HERE = os.path.dirname(os.path.abspath(__file__)); S = 65; NL = 40
PD = ctypes.POINTER(ctypes.c_double); PF = ctypes.POINTER(ctypes.c_float); PI = ctypes.POINTER(ctypes.c_int64); PY = ctypes.POINTER(ctypes.c_int32)

SRC = r'''
#include <math.h>
#include <stdint.h>
#define S 65
#define NL 40
static inline int level(float n, float u, int flat) {
    if (flat) return 0;
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* at each byte: every active cell that has counted says s(v) = log((c(v)+e)/(n+65e)); the answering cell's branches compete on the sum of trust times s;
   then the trust that was used changes by eta times (mismatch . s), and the cells count */
void mix_steps(int64_t n, int64_t K1, const int64_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double eta, double e,
               int learn_c, int learn_t, int flat, double *P) {
    static double lt[4096]; double a[S], p[S], s[32][S]; int lev[32];
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int64_t t = 0; t < n; t++) {
        const int64_t *id = ids + t * K1; double mx = -1e300, z = 0;
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int64_t k = 0; k < K1; k++) {
            lev[k] = -1; if (id[k] < 0 || N[id[k]] < 1) continue;
            const float *c = C + id[k] * S; lev[k] = level(N[id[k]], U[id[k]], flat); const double tk = T[k * NL + lev[k]], den = log(N[id[k]] + S * e);
            for (int v = 0; v < S; v++) { const double sv = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den; s[k][v] = sv; a[v] += tk * sv; }
        }
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (learn_t) for (int64_t k = 0; k < K1; k++) if (lev[k] >= 0) {
            double g = s[k][y[t]]; for (int v = 0; v < S; v++) g -= p[v] * s[k][v];
            T[k * NL + lev[k]] += eta * g;
        }
        if (learn_c) for (int64_t k = 0; k < K1; k++) if (id[k] >= 0) {
            float *c = C + id[k] * S; if (c[y[t]] == 0) U[id[k]] += 1; c[y[t]] += 1; N[id[k]] += 1;
        }
    }
}
'''
def kern():
    so = os.path.join(HERE, "kern_m.so"); src = os.path.join(HERE, "kern_m.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, src, "-lm"])
    L = ctypes.CDLL(so); L.mix_steps.argtypes = [ctypes.c_int64, ctypes.c_int64, PI, PY, PF, PF, PF, PD, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, PD]; L.mix_steps.restype = None; return L

def level_np(n, u):
    bn = 0; m = int(n)
    while m > 1 and bn < 9: m >>= 1; bn += 1
    return bn * 4 + (0 if u <= 1 else 1 if u <= 2 else 2 if u <= 4 else 3)

def steps_np(ids, y, C, N, U, T, eta, e, learn=True, flat=False):
    P = np.zeros((len(y), S))
    for t in range(len(y)):
        act = [(k, i) for k, i in enumerate(ids[t]) if i >= 0 and N[i] >= 1]; a = np.zeros(S); ss = []
        for k, i in act:
            l = 0 if flat else level_np(N[i], U[i]); sv = np.log(C[i].astype(np.float64) + e) - np.log(N[i] + S * e); ss.append((k, l, sv)); a += T[k, l] * sv
        p = np.exp(a - a.max()); p /= p.sum(); P[t] = p
        if learn:
            for k, l, sv in ss: T[k, l] += eta * (sv[y[t]] - p @ sv)
            for i in ids[t]:
                if i >= 0:
                    if C[i, y[t]] == 0: U[i] += 1
                    C[i, y[t]] += 1; N[i] += 1
    return P

class Sys:
    """The cells (counts) and the answering cell's trust; ids are built over the whole text so that a cell first met in the last tenth can be made there when learning goes on."""
    def __init__(s, x, keys, K, flat=False):
        s.x, s.K, s.flat = x, K, int(flat); s.ids, s.ncell, s.kinds, _ = words.make_ids(x, keys, K, len(x), len(x), K, 1); s.K1 = s.ids.shape[1]; s.L = kern()
        s.C = np.zeros((s.ncell, S), np.float32); s.N = np.zeros(s.ncell, np.float32); s.U = np.zeros(s.ncell, np.float32); s.T = np.full((s.K1, NL), 0.25)
    def go(s, lo, hi, eta, e, learn, want_p=False):
        """Bytes lo..hi-1 are named (lo >= K); with learn the trust and the counts change."""
        ids = np.ascontiguousarray(s.ids[lo - s.K:hi - s.K]); y = np.ascontiguousarray(s.x[lo:hi].astype(np.int32)); P = np.zeros((hi - lo, S)) if want_p else None
        s.L.mix_steps(hi - lo, s.K1, ids.ctypes.data_as(PI), y.ctypes.data_as(PY), s.C.ctypes.data_as(PF), s.N.ctypes.data_as(PF), s.U.ctypes.data_as(PF), s.T.ctypes.data_as(PD), eta, e, int(learn), int(learn), s.flat, P.ctypes.data_as(PD) if want_p else None)
        return P
    def copy_state(s): return s.C.copy(), s.N.copy(), s.U.copy(), s.T.copy()
    def set_state(s, st): s.C, s.N, s.U, s.T = (a.copy() for a in st)

def arm_setup(x, arm):
    K = 8 if arm == 4 else 6; keys = words.word_keys(x, clock=5 if arm == 3 else 0) if arm in (2, 3) else []
    return K, keys, arm == 1

def run(kw):
    x, cut, _ = score.load(); arm = int(kw.get("arm", 0)); K, keys, flat = arm_setup(x, arm); dev = cut - cut // 20; t0 = time.process_time(); grid = {}
    if "eta" in kw and "e" in kw: eta, e = float(kw["eta"]), float(kw["e"])
    else:
        for e in ((0.005, 0.02, 0.1) if kw.get("low") else (0.005, 0.02, 0.1, 0.5)):
            for eta in ((0.00005, 0.0001, 0.0002, 0.0005) if kw.get("low") else (0.0005, 0.002, 0.008, 0.03)):
                s = Sys(x, keys, K, flat); s.go(K, dev, eta, e, True); grid[f"{eta}/{e}"] = score.score(s.go(dev, cut, eta, e, False, True), x[dev:cut])["nats"]
        eta, e = (float(v) for v in min(grid, key=grid.get).split("/"))
    s = Sys(x, keys, K, flat); s.go(K, cut, eta, e, True); cells = int((s.N > 0).sum()); above = int((s.C > 0).sum()); trust = np.round(s.T, 3).tolist()
    r = score.score(s.go(cut, len(x), eta, e, False, True)); on = score.score(s.go(cut, len(x), eta, e, True, True))
    r.update(arm=arm, K=K, eta=eta, e=e, grid=grid, cells=cells, counts_above_zero=above, cells_by_kind=[int((s.N[np.unique(s.ids[:cut - K, k][s.ids[:cut - K, k] >= 0])] > 0).sum()) for k in range(s.K1)],
             trusts=int(s.T.size if not flat else s.K1), trust=trust, learning_on_nats=on["nats"], learning_on_right=on["right"], cpu=round(time.process_time() - t0, 1)); return r

def check():
    x, cut, _ = score.load(); K = 4; n = 4000; keys = words.word_keys(x)[:2]; e, eta = 0.05, 0.01
    for flat in (False, True):
        s = Sys(x[:n], [k[:n] for k in keys], K, flat); P2 = s.go(K, n, eta, e, True, True)
        C = np.zeros((s.ncell, S), np.float32); N = np.zeros(s.ncell, np.float32); U = np.zeros(s.ncell, np.float32); T = np.full((s.K1, NL), 0.25); P1 = steps_np(s.ids, x[K:n], C, N, U, T, eta, e, True, flat)
        d = max(np.abs(P1 - P2).max(), np.abs(T - s.T).max(), np.abs(C - s.C).max()); print(f"trust form, {n - K} bytes, K = {K}, two added kinds, flat = {flat}: compiled against numpy, largest difference {d:.1e}", "PASSED" if d < 1e-9 else "FAILED")

if __name__ == "__main__":
    if sys.argv[1] == "check": check()
    else:
        kw = dict(a.split("=") for a in sys.argv[1:]); r = run(kw); tag = "trust_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
        json.dump(r, open(os.path.join(HERE, "res", tag + ".json"), "w"))
        print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("n", "grid", "trust")}, "best of grid:", sorted((round(v, 4), k) for k, v in r["grid"].items())[:4], flush=True)
