#!/usr/bin/env python3
"""PLAN_12.md: the answering cell's trust in a companion is held separately for each number of long companions alive at the byte (0 to 4).
    python3 fresh.py T2 0|1        the life and measures of two.py's `moments` (PLAN_11.md); 0: the change switched off (must give PLAN_11.md's figures)
    python3 fresh.py check         the compiled loop against numpy on a prefix, and against two.py's loop with the change off
Companions: half-life 10,000 bytes, born with no trust. Everything else as two.py."""
import ctypes, json, os, subprocess, sys
import numpy as np
import score, table, words, trust, life, two
HERE = os.path.dirname(os.path.abspath(__file__)); S = 65; NL = 40; NF = 5; ETA, E = 0.00005, 0.02; H = 10000.0
PD = ctypes.POINTER(ctypes.c_double); PF = ctypes.POINTER(ctypes.c_float); PI = ctypes.POINTER(ctypes.c_int64); PY = ctypes.POINTER(ctypes.c_int32)
SRC = r'''
#include <math.h>
#include <stdint.h>
#define S 65
#define NL 40
#define NF 5
static inline int level(double n, int u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* two.py's loop with one change: the trust in a companion (a kind with half[k] > 0) is T[k][NL * fresh + level], where fresh is the number of companions
   of long runs (kinds first_long and above) that are alive at this byte, 0 to NF - 1; with use_fresh = 0 fresh is always 0 and the loop is two.py's. */
void fresh_steps(int64_t n, int64_t K1, const int64_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double eta, double e,
                 int learn, double *P, const double *half, double *last, double clock0, int64_t first_long, int use_fresh) {
    static double lt[4096]; double a[S], p[S], s[40][S]; int lev[40]; int64_t ix[40];
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int64_t t = 0; t < n; t++) {
        const int64_t *id = ids + t * K1; double mx = -1e300, z = 0; const double clock = clock0 + (learn ? (double)t : 0.0); int fresh = 0;
        for (int64_t k = 0; k < K1; k++) {
            lev[k] = -1; const int64_t i = id[k]; if (i < 0) continue;
            float *c = C + i * S;
            if (half[k] > 0) {
                if (N[i] <= 0) continue;
                const double f = exp2(-(clock - last[i]) / half[k]), nn = N[i] * f;
                if (learn) { for (int v = 0; v < S; v++) c[v] = (float)(c[v] * f); N[i] = (float)nn; last[i] = clock; }
                if (nn < 1) continue;
                int u = 0; const double den = log(nn + S * e);
                for (int v = 0; v < S; v++) { const double cv = learn ? (double)c[v] : c[v] * f; if (cv >= 0.5) u++; s[k][v] = log(cv + e) - den; }
                lev[k] = level(nn, u); if (k >= first_long) fresh++;
            } else {
                if (N[i] < 1) continue;
                lev[k] = level(N[i], (int)U[i]); const double den = log(N[i] + S * e);
                for (int v = 0; v < S; v++) s[k][v] = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den;
            }
        }
        if (!use_fresh) fresh = 0;
        if (fresh > NF - 1) fresh = NF - 1;
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int64_t k = 0; k < K1; k++) if (lev[k] >= 0) {
            ix[k] = k * NL * NF + lev[k] + (half[k] > 0 ? NL * fresh : 0); const double tk = T[ix[k]];
            for (int v = 0; v < S; v++) a[v] += tk * s[k][v];
        }
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (learn) {
            for (int64_t k = 0; k < K1; k++) if (lev[k] >= 0) {
                double g = s[k][y[t]]; for (int v = 0; v < S; v++) g -= p[v] * s[k][v];
                T[ix[k]] += eta * g;
            }
            for (int64_t k = 0; k < K1; k++) if (id[k] >= 0) {
                float *c = C + id[k] * S; if (half[k] > 0) last[id[k]] = clock; else if (c[y[t]] == 0) U[id[k]] += 1;
                c[y[t]] += 1; N[id[k]] += 1;
            }
        }
    }
}
'''
def kern():
    so = os.path.join(HERE, "kern_f.so"); src = os.path.join(HERE, "kern_f.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, src, "-lm"])
    L = ctypes.CDLL(so); L.fresh_steps.argtypes = [ctypes.c_int64, ctypes.c_int64, PI, PY, PF, PF, PF, PD, ctypes.c_double, ctypes.c_double, ctypes.c_int, PD, PD, PD, ctypes.c_double, ctypes.c_int64, ctypes.c_int]; L.fresh_steps.restype = None; return L

class SysF:
    def __init__(s, x, keys, K, use_fresh, half=H):
        ids, ncell, kinds, _ = words.make_ids(x, keys, K, len(x), len(x), K, 1); ntab = ncell - sum(kinds); nk = ids.shape[1]
        ids = np.hstack([ids, np.where(ids[:, :K + 1] >= 0, ids[:, :K + 1] + ncell, -1)]); tot = ncell + ntab
        s.x, s.K, s.ids, s.K1, s.L, s.use = x, K, np.ascontiguousarray(ids), ids.shape[1], kern(), int(use_fresh); s.half = np.zeros(s.K1); s.half[nk:] = half; s.first_long = nk + 3
        s.C = np.zeros((tot, S), np.float32); s.N = np.zeros(tot, np.float32); s.U = np.zeros(tot, np.float32); s.last = np.zeros(tot); s.T = np.full((s.K1, NL * NF), 0.25); s.T[nk:] = 0.0; s.clock = 0.0
    def go(s, lo, hi, learn, want_p=False):
        ids = np.ascontiguousarray(s.ids[lo - s.K:hi - s.K]); y = np.ascontiguousarray(s.x[lo:hi].astype(np.int32)); P = np.zeros((hi - lo, S)) if want_p else None
        s.L.fresh_steps(hi - lo, s.K1, ids.ctypes.data_as(PI), y.ctypes.data_as(PY), s.C.ctypes.data_as(PF), s.N.ctypes.data_as(PF), s.U.ctypes.data_as(PF), s.T.ctypes.data_as(PD), ETA, E, int(learn),
                        P.ctypes.data_as(PD) if want_p else None, s.half.ctypes.data_as(PD), s.last.ctypes.data_as(PD), s.clock, s.first_long, s.use)
        if learn: s.clock += hi - lo
        return P

def steps_np(sy, lo, hi, learn):
    """The reference for one stretch, on the state of a SysF (changed in place as the compiled loop changes it)."""
    P = np.zeros((hi - lo, S)); C, N, U, T, half, last = sy.C, sy.N, sy.U, sy.T, sy.half, sy.last
    for t in range(hi - lo):
        ids = sy.ids[lo - sy.K + t]; y = sy.x[lo + t]; clock = sy.clock + (t if learn else 0); ss = []; fresh = 0
        for k, i in enumerate(ids):
            if i < 0: continue
            if half[k] > 0:
                if N[i] <= 0: continue
                f = 2.0 ** (-(clock - last[i]) / half[k]); nn = float(N[i]) * f
                if learn: C[i] = (C[i].astype(np.float64) * f).astype(np.float32); N[i] = np.float32(nn); last[i] = clock
                if nn < 1: continue
                cv = C[i].astype(np.float64) if learn else C[i].astype(np.float64) * f; sv = np.log(cv + E) - np.log(nn + S * E); l = trust.level_np(nn, int((cv >= 0.5).sum())); fresh += k >= sy.first_long
            else:
                if N[i] < 1: continue
                sv = np.log(C[i].astype(np.float64) + E) - np.log(float(N[i]) + S * E); l = trust.level_np(N[i], U[i])
            ss.append((k, l, sv))
        fresh = min(fresh, NF - 1) if sy.use else 0; a = np.zeros(S); ix = []
        for k, l, sv in ss: j = l + (NL * fresh if half[k] > 0 else 0); ix.append(j); a += T[k, j] * sv
        p = np.exp(a - a.max()); p /= p.sum(); P[t] = p
        if learn:
            for (k, l, sv), j in zip(ss, ix): T[k, j] += ETA * (sv[y] - p @ sv)
            for k, i in enumerate(ids):
                if i < 0: continue
                if half[k] > 0: last[i] = clock
                elif C[i, y] == 0: U[i] += 1
                C[i, y] += 1; N[i] += 1
    if learn: sy.clock += hi - lo
    return P

def moments(name, use):
    x, segs, _ = life.load_all(); K = 6; keys = words.word_keys(x) if name == "T2" else []; r = dict(system=name, use_fresh=use); sc = lambda sy, k: score.score(sy.go(segs[k][1], segs[k][2], False, True), x[segs[k][1]:segs[k][2]])["nats"]
    sy = SysF(x, keys, K, use); r["A_B"] = {}
    for k in segs: sy.go(max(segs[k][0], K), segs[k][1], True); r["A_B"][k] = sc(sy, k)
    r["A_at_end"] = sc(sy, "A"); nk = sy.K1 - (K + 1); r["trust_of_short_companions_by_long_ones_alive"] = {f"run of {j} bytes": [round(float(sy.T[nk + j, NL * f:NL * (f + 1)][sy.T[nk + j, NL * f:NL * (f + 1)] != 0].mean()) if (sy.T[nk + j, NL * f:NL * (f + 1)] != 0).any() else 0.0, 3) for f in range(NF)] for j in range(K + 1)}
    s2 = SysF(x, keys, K, use); r["B_only"] = {}
    for k in segs:
        if k != "A": s2.go(segs[k][0], segs[k][1], True); r["B_only"][k] = sc(s2, k)
    return r

def check():
    x, cut, _ = score.load(); K = 4; n = 4000; x = x[:n]; keys = [k[:n] for k in words.word_keys(x)[:1]]
    for use in (0, 1):
        a = SysF(x, keys, K, use, half=300.0); b = SysF(x, keys, K, use, half=300.0); P2 = np.vstack([a.go(K, 3000, True, True), a.go(3000, 3500, False, True), a.go(3500, n, True, True)])
        P1 = np.vstack([steps_np(b, K, 3000, True), steps_np(b, 3000, 3500, False), steps_np(b, 3500, n, True)]); d = max(np.abs(P1 - P2).max(), np.abs(a.T - b.T).max(), np.abs(a.C - b.C).max())
        print(f"use_fresh = {use}: compiled against numpy, {n - K} bytes (learn, score, learn), largest difference {d:.1e}", "PASSED" if d < 1e-6 else "FAILED")
    os.environ["COMPANION_BIRTH"] = "0"; two.BIRTH = 0.0; a = SysF(x, keys, K, 0, half=300.0); t = two.Sys2(x, keys, K, 300); d = np.abs(a.go(K, n, True, True) - t.go(K, n, True, True)).max(); print(f"change switched off against two.py's loop: largest difference {d:.1e}", "PASSED" if d < 1e-12 else "FAILED")

if __name__ == "__main__":
    if sys.argv[1] == "check": check()
    else:
        name, use = sys.argv[1], int(sys.argv[2]); r = moments(name, use); json.dump(r, open(os.path.join(HERE, "res", f"fresh_{name}_{use}.json"), "w"))
        print(name, "use_fresh", use, "| A, B in order:", {k[:6]: round(v, 4) for k, v in r["A_B"].items()}, "| B only:", {k[:6]: round(v, 4) for k, v in r["B_only"].items()}, "| A at the end:", round(r["A_at_end"], 4), "rise", round(r["A_at_end"] - r["A_B"]["A"], 4), flush=True)
