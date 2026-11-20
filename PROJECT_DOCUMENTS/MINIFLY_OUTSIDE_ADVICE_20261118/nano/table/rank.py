#!/usr/bin/env python3
"""PLAN_14.md: the trust of a kind and level is held separately for the number (0, 1, 2, 3 or more) of active senders that have counted fewer times
than this one and at least twice.
    python3 rank.py 0|1          0: the change switched off (must give T2's figures); 1: on; 2: on, with the situation's part born at nothing beside the kind's own trust (addition 1). The lives A, B and B only; the test text's last tenth once.
    python3 rank.py check"""
import ctypes, json, os, subprocess, sys
import numpy as np
import score, table, words, trust, life
HERE = os.path.dirname(os.path.abspath(__file__)); S = 65; NL = 40; NR = 4; ETA, E = 0.00005 * float(os.environ.get("STEP_TIMES", 1)), 0.02          # STEP_TIMES=2: the control of PLAN_14.md, section 8
PD = ctypes.POINTER(ctypes.c_double); PF = ctypes.POINTER(ctypes.c_float); PI = ctypes.POINTER(ctypes.c_int64); PY = ctypes.POINTER(ctypes.c_int32)
SRC = r'''
#include <math.h>
#include <stdint.h>
#define S 65
#define NL 40
#define NR 4
static inline int level(float n, float u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* trust.py's loop (constant step) with one change: the trust used for sender k is T[k][NL * r + level], r = min(3, number of active senders j with
   2 <= N[j] < N[k]); with use = 0, r is always 0 and the loop is trust.py's. */
void rank_steps(int64_t n, int64_t K1, const int64_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double eta, double e, int learn, double *P, int use, double *Tb) {
    static double lt[4096]; double a[S], p[S], s[32][S]; int lev[32]; int64_t ix[32]; double nk[32];
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int64_t t = 0; t < n; t++) {
        const int64_t *id = ids + t * K1; double mx = -1e300, z = 0;
        for (int64_t k = 0; k < K1; k++) {
            lev[k] = -1; if (id[k] < 0 || N[id[k]] < 1) continue;
            const float *c = C + id[k] * S; lev[k] = level(N[id[k]], U[id[k]]); nk[k] = N[id[k]]; const double den = log(N[id[k]] + S * e);
            for (int v = 0; v < S; v++) s[k][v] = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den;
        }
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int64_t k = 0; k < K1; k++) if (lev[k] >= 0) {
            int r = 0; if (use) { for (int64_t j = 0; j < K1; j++) if (lev[j] >= 0 && nk[j] >= 2 && nk[j] < nk[k]) r++; if (r > NR - 1) r = NR - 1; }
            ix[k] = k * NL * NR + NL * r + lev[k]; const double tk = T[ix[k]] + (Tb ? Tb[k * NL + lev[k]] : 0.0);          /* Tb: the kind's and level's own trust; T is then the situation's part */
            for (int v = 0; v < S; v++) a[v] += tk * s[k][v];
        }
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (learn) {
            for (int64_t k = 0; k < K1; k++) if (lev[k] >= 0) { double g = s[k][y[t]]; for (int v = 0; v < S; v++) g -= p[v] * s[k][v]; T[ix[k]] += eta * g; if (Tb) Tb[k * NL + lev[k]] += eta * g; }
            for (int64_t k = 0; k < K1; k++) if (id[k] >= 0) { float *c = C + id[k] * S; if (c[y[t]] == 0) U[id[k]] += 1; c[y[t]] += 1; N[id[k]] += 1; }
        }
    }
}
'''
def kern():
    so = os.path.join(HERE, "kern_k.so"); src = os.path.join(HERE, "kern_k.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, src, "-lm"])
    L = ctypes.CDLL(so); L.rank_steps.argtypes = [ctypes.c_int64, ctypes.c_int64, PI, PY, PF, PF, PF, PD, ctypes.c_double, ctypes.c_double, ctypes.c_int, PD, ctypes.c_int, PD]; L.rank_steps.restype = None; return L

class SysR:
    def __init__(s, x, keys, K, use):
        s.x, s.K, s.use = x, K, int(use); s.ids, s.ncell, _, _ = words.make_ids(x, keys, K, len(x), len(x), K, 1); s.K1 = s.ids.shape[1]; s.L = kern()
        s.C = np.zeros((s.ncell, S), np.float32); s.N = np.zeros(s.ncell, np.float32); s.U = np.zeros(s.ncell, np.float32); s.T = np.full((s.K1, NL * NR), 0.25); s.Tb = None
        if s.use == 2: s.T[:] = 0.0; s.Tb = np.full((s.K1, NL), 0.25)                 # PLAN_14.md, addition 1: the situation's part born at nothing
    def go(s, lo, hi, learn, want_p=False):
        ids = np.ascontiguousarray(s.ids[lo - s.K:hi - s.K]); y = np.ascontiguousarray(s.x[lo:hi].astype(np.int32)); P = np.zeros((hi - lo, S)) if want_p else None
        s.L.rank_steps(hi - lo, s.K1, ids.ctypes.data_as(PI), y.ctypes.data_as(PY), s.C.ctypes.data_as(PF), s.N.ctypes.data_as(PF), s.U.ctypes.data_as(PF), s.T.ctypes.data_as(PD), ETA, E, int(learn), P.ctypes.data_as(PD) if want_p else None, s.use, s.Tb.ctypes.data_as(PD) if s.Tb is not None else None); return P

def steps_np(sy, lo, hi):
    """The reference (learning on), on a SysR's state."""
    P = np.zeros((hi - lo, S)); C, N, U, T = sy.C, sy.N, sy.U, sy.T
    for t in range(hi - lo):
        ids = sy.ids[lo - sy.K + t]; y = sy.x[lo + t]; act = [(k, i) for k, i in enumerate(ids) if i >= 0 and N[i] >= 1]; a = np.zeros(S); ss = []
        for k, i in act:
            r = min(NR - 1, sum(1 for _, j in act if 2 <= N[j] < N[i])) if sy.use else 0; sv = np.log(C[i].astype(np.float64) + E) - np.log(float(N[i]) + S * E); l = trust.level_np(N[i], U[i]); j = NL * r + l; ss.append((k, j, sv, l)); a += (T[k, j] + (sy.Tb[k, l] if sy.Tb is not None else 0.0)) * sv
        p = np.exp(a - a.max()); p /= p.sum(); P[t] = p
        for k, j, sv, l in ss:
            g = ETA * (sv[y] - p @ sv); T[k, j] += g
            if sy.Tb is not None: sy.Tb[k, l] += g
        for i in ids:
            if i >= 0:
                if C[i, y] == 0: U[i] += 1
                C[i, y] += 1; N[i] += 1
    return P

def lives(use):
    x, segs, _ = life.load_all(); K = 6; keys = words.word_keys(x); sc = lambda sy, k: score.score(sy.go(segs[k][1], segs[k][2], False, True), x[segs[k][1]:segs[k][2]])["nats"]; r = dict(use=use, A_B={}, B_only={})
    sy = SysR(x, keys, K, use)
    for k in segs:
        sy.go(max(segs[k][0], K), segs[k][1], True)
        if k != "A": r["A_B"][k] = sc(sy, k)
    T = sy.T.reshape(sy.K1, NR, NL); b0 = 0.0 if sy.Tb is not None else 0.25; r["mean_trust_by_kind_and_number_of_more_specific_senders"] = [[round(float(T[k, q][T[k, q] != b0].mean()) if (T[k, q] != b0).any() else b0, 3) for q in range(NR)] for k in range(sy.K1)]
    s2 = SysR(x, keys, K, use)
    for k in segs:
        if k != "A": s2.go(segs[k][0], segs[k][1], True); r["B_only"][k] = sc(s2, k)
    s3 = SysR(x, keys, K, use); a, c, b = segs["A"]; s3.go(K, c, True); r["test_text_last_tenth"] = score.score(s3.go(c, b, False, True), x[c:b])["nats"]; return r

def check():
    x, cut, _ = score.load(); K = 4; n = 3000; x = x[:n]; keys = [k[:n] for k in words.word_keys(x)[:1]]
    for use in (0, 1, 2):
        a = SysR(x, keys, K, use); b = SysR(x, keys, K, use); P2 = a.go(K, n, True, True); P1 = steps_np(b, K, n); d = max(np.abs(P1 - P2).max(), np.abs(a.T - b.T).max(), np.abs(a.Tb - b.Tb).max() if a.Tb is not None else 0.0); print(f"use = {use}: compiled against numpy, {n - K} bytes, largest difference {d:.1e}", "PASSED" if d < 1e-9 else "FAILED")
    a = SysR(x, keys, K, 0); t = trust.Sys(x, keys, K); d = np.abs(a.go(K, n, True, True) - t.go(K, n, ETA, E, True, True)).max(); print(f"change switched off against trust.py's loop: largest difference {d:.1e}", "PASSED" if d < 1e-12 else "FAILED")

if __name__ == "__main__":
    if sys.argv[1] == "check": check()
    else:
        use = int(sys.argv[1]); r = lives(use); json.dump(r, open(os.path.join(HERE, "res", f"rank_{use}" + (f"_step{os.environ['STEP_TIMES']}" if "STEP_TIMES" in os.environ else "") + ".json"), "w"))
        print("use", use, "| life A, B:", {k[:6]: round(v, 4) for k, v in r["A_B"].items()}, "| B only:", {k[:6]: round(v, 4) for k, v in r["B_only"].items()}, "| test text:", round(r["test_text_last_tenth"], 4), flush=True)
        if use: print("mean trust by kind (rows) and number of more specific senders 0,1,2,3+:", r["mean_trust_by_kind_and_number_of_more_specific_senders"])
