#!/usr/bin/env python3
"""nano/both, PLAN_2.md: a trust is learned from the answer of the senders that have counted at least as much. The system is both.py's; only the
mismatch that teaches each trust (and each part of the second cell) changes: the speaking senders are grouped by the doubling of their counts, and a
sender of group b learns from the competition of the branches summed over groups b and higher. The answer given is that of all senders, as before.
The relay system's count is the number of bytes it has been taught.
    python3 casc.py check
    python3 casc.py sums seed=1 arm=alone|joined cascade=1|0 [lines=400000] [tag=...]"""
import ctypes, hashlib, json, os, subprocess, sys
import numpy as np
import both as B
from both import G, NL, XL, PD, PF, PI, p32, pf, pd
HERE = B.HERE

SRC = r'''
#include <math.h>
#include <stdint.h>
#define NL 40
#define XL 10
#define KMAX 400
#define SMAX 65
#define DMAX 16
#define GB 40
static inline int level(float n, float u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
static inline int grp(float n) { int b = 0; int64_t m = (int64_t)n; while (m > 1) { m >>= 1; b++; } return b; }
static inline void compete(const double *a, const double *b, int S, double *p) {
    double mx = -1e300, z = 0;
    for (int v = 0; v < S; v++) { p[v] = a[v] + (b ? b[v] : 0.0); if (p[v] > mx) mx = p[v]; }
    for (int v = 0; v < S; v++) { p[v] = exp(p[v] - mx); z += p[v]; }
    for (int v = 0; v < S; v++) p[v] /= z;
}
/* both.py's mix_steps3 with the groups. cascade 0: one group (both.py's arithmetic, to rounding). xg: the group of the given sender. */
void mix_steps4(int64_t n, int K1, int S, const int32_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double *Tu,
                double eta, double e, int learn, int shrink, double *P, int D, const int32_t *cb, double *Gt, double *Gu, double *P2,
                const float *X, const int32_t *xlev, double *Tx, double *Txu, double *Gx, double *Gxu, int cascade, int xg) {
    static double lt[4096], s[KMAX][SMAX], ga[GB][SMAX], gb2[GB][SMAX], pg[GB][SMAX], pg2[GB][SMAX]; static int lev[KMAX], gk[KMAX]; static int64_t gix[KMAX][DMAX];
    int64_t xix[DMAX]; int has[GB]; double r1[SMAX], r2[SMAX], uni[SMAX];
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int v = 0; v < S; v++) uni[v] = 1.0 / S;
    for (int64_t t = 0; t < n; t++) {
        const int32_t *id = ids + t * K1; const int xl = (X && xlev[t] >= 0) ? xlev[t] : -1; const float *sx = X ? X + t * S : 0; const int xgk = cascade ? xg : 0;
        for (int b = 0; b < GB; b++) has[b] = 0;
#define OPEN(b) if (!has[b]) { has[b] = 1; for (int v = 0; v < S; v++) { ga[b][v] = 0; gb2[b][v] = 0; } }
        for (int k = 0; k < K1; k++) {
            lev[k] = -1; if (id[k] < 0 || N[id[k]] < 1) continue;
            const float *c = C + (int64_t)id[k] * S; lev[k] = level(N[id[k]], U[id[k]]); const int g = cascade ? grp(N[id[k]]) : 0; gk[k] = g; OPEN(g)
            const double tk = T[k * NL + lev[k]], den = log(N[id[k]] + S * e); double tk2 = 0;
            for (int j = 0; j < D; j++) { gix[k][j] = (((int64_t)j * K1 + k) * NL + lev[k]) * S + cb[t * D + j]; tk2 += Gt[gix[k][j]]; }
            for (int v = 0; v < S; v++) { const double sv = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den; s[k][v] = sv; ga[g][v] += tk * sv; gb2[g][v] += tk2 * sv; }
        }
        if (xl >= 0) { const double tk = Tx[xl]; double tk2 = 0; OPEN(xgk)
            for (int j = 0; j < D; j++) { xix[j] = ((int64_t)j * XL + xl) * S + cb[t * D + j]; tk2 += Gx[xix[j]]; }
            for (int v = 0; v < S; v++) { ga[xgk][v] += tk * sx[v]; gb2[xgk][v] += tk2 * sx[v]; } }
        int low = -1; for (int v = 0; v < S; v++) { r1[v] = 0; r2[v] = 0; }
        for (int b = GB - 1; b >= 0; b--) if (has[b]) { for (int v = 0; v < S; v++) { r1[v] += ga[b][v]; r2[v] += gb2[b][v]; } compete(r1, 0, S, pg[b]); compete(r1, r2, S, pg2[b]); low = b; }
        const double *p = low >= 0 ? pg[low] : uni, *p2 = low >= 0 ? pg2[low] : uni;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (P2) for (int v = 0; v < S; v++) P2[t * S + v] = p2[v];
        if (learn) {
            for (int k = 0; k < K1; k++) if (lev[k] >= 0) {
                const double *q = pg[gk[k]], *q2 = pg2[gk[k]]; double g = s[k][y[t]], g2 = s[k][y[t]]; for (int v = 0; v < S; v++) { g -= q[v] * s[k][v]; g2 -= q2[v] * s[k][v]; }
                T[k * NL + lev[k]] += (shrink ? eta / sqrt(1.0 + Tu[k * NL + lev[k]]) : eta) * g; Tu[k * NL + lev[k]] += 1.0;
                for (int j = 0; j < D; j++) { const int64_t i = gix[k][j]; Gt[i] += (shrink ? eta / sqrt(1.0 + Gu[i]) : eta) * g2; Gu[i] += 1.0; }
            }
            if (xl >= 0) {
                const double *q = pg[xgk], *q2 = pg2[xgk]; double g = sx[y[t]], g2 = sx[y[t]]; for (int v = 0; v < S; v++) { g -= q[v] * sx[v]; g2 -= q2[v] * sx[v]; }
                Tx[xl] += (shrink ? eta / sqrt(1.0 + Txu[xl]) : eta) * g; Txu[xl] += 1.0;
                for (int j = 0; j < D; j++) { const int64_t i = xix[j]; Gx[i] += (shrink ? eta / sqrt(1.0 + Gxu[i]) : eta) * g2; Gxu[i] += 1.0; }
            }
            if (learn == 1) for (int k = 0; k < K1; k++) { const int64_t i = id[k] < -1 ? -2 - (int64_t)id[k] : id[k];
                if (i >= 0) { float *c = C + i * S; if (c[y[t]] == 0) U[i] += 1; c[y[t]] += 1; N[i] += 1; } }
        }
    }
}
'''
def kern():
    h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_casc_{h}.so"); src = os.path.join(HERE, "kern_casc.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd = ctypes.c_int, ctypes.c_double
    L.mix_steps4.argtypes = [ctypes.c_int64, ci, ci, PI, PI, PF, PF, PF, PD, PD, cd, cd, ci, ci, PD, ci, PI, PD, PD, PD, PF, PI, PD, PD, PD, PD, ci, ci]; L.mix_steps4.restype = None; return L

def grp_np(n):
    b = 0; m = int(n)
    while m > 1: m >>= 1; b += 1
    return b

class SysC(B.Sys3):
    CASCADE = 1
    def __init__(s, *a, **kw): super().__init__(*a, **kw); s.L4 = kern(); s.cascade = SysC.CASCADE; s.xtaught = 0
    def _run(s, ctx, y, learn, want_p):
        P = np.zeros((len(ctx), s.S)) if want_p else None
        for a in range(0, len(ctx), 50000):
            idd = s.ids(ctx[a:a + 50000], create=learn and not s.only_trust); yy = np.ascontiguousarray(y[a:a + 50000].astype(np.int32)) if y is not None else np.zeros(len(idd), np.int32)
            cb = np.ascontiguousarray(ctx[a:a + 50000, :s.D].astype(np.int32)); P1 = np.zeros((len(idd), s.S)) if want_p else None; P2 = np.zeros((len(idd), s.S)) if want_p else None
            X = xl = None
            if s.xq is not None: X = s.xq[0][s.xpos + a:s.xpos + a + len(idd)]; xl = s.xq[1][s.xpos + a:s.xpos + a + len(idd)]; assert len(X) == len(idd)
            s.L4.mix_steps4(len(idd), s.K1, s.S, p32(idd), p32(yy), pf(s.C), pf(s.N), pf(s.U), pd(s.T), pd(s.Tu), s.eta, s.e, (2 if s.only_trust else 1) if learn else 0, s.shrink, pd(P1) if want_p else None,
                            s.D, p32(cb), pd(s.Gt), pd(s.Gu), pd(P2) if want_p else None, pf(X) if X is not None else None, p32(xl) if X is not None else None, pd(s.Tx), pd(s.Txu), pd(s.Gx), pd(s.Gxu),
                            s.cascade, grp_np(max(1, s.xtaught)))                      # the relay system's count: the bytes it had been taught when this piece of at most 50,000 bytes began
            if learn and xl is not None: s.xtaught += int((xl >= 0).sum())
            if want_p: P[a:a + 50000] = P2 if s.cell == 2 else P1
        s.xpos += len(ctx); return P

def steps4_np(ids, y, cb, C, N, U, T, Tu, Gt, Gu, eta, e, X, xlev, Tx, Txu, Gx, Gxu, cascade, xg, learn=1, shrink=False):
    """The reference: every group's answer is computed from its definition."""
    S = C.shape[1]; n = len(y); P1 = np.zeros((n, S)); P2 = np.zeros((n, S)); D = Gt.shape[0]; st = lambda u: eta / np.sqrt(1.0 + u) if shrink else eta
    sm = lambda a: np.exp(a - a.max()) / np.exp(a - a.max()).sum()
    for t in range(n):
        gv = [int(cb[t, j]) for j in range(D)]; snd = []                                # (group, trust, sum of parts, message, what to change)
        for k, i in enumerate(ids[t]):
            if i >= 0 and N[i] >= 1:
                l = G.level_np(N[i], U[i]); sv = np.log(C[i].astype(np.float64) + e) - np.log(N[i] + S * e); snd.append((grp_np(N[i]) if cascade else 0, T[k, l], sum(Gt[j, k, l, gv[j]] for j in range(D)), sv, ("k", k, l)))
        if X is not None and xlev[t] >= 0: xl = int(xlev[t]); snd.append((xg if cascade else 0, Tx[xl], sum(Gx[j, xl, gv[j]] for j in range(D)), X[t].astype(np.float64), ("x", xl)))
        def ans(b): a1 = sum((q[1] * q[3] for q in snd if q[0] >= b), np.zeros(S)); a2 = a1 + sum((q[2] * q[3] for q in snd if q[0] >= b), np.zeros(S)); return sm(a1), sm(a2)
        P1[t], P2[t] = ans(0) if snd else (np.full(S, 1 / S), np.full(S, 1 / S)); ups = []
        if learn:
            for b, _, _, sv, w in snd: q1, q2 = ans(b); ups.append((w, sv[y[t]] - q1 @ sv, sv[y[t]] - q2 @ sv))
            for w, g, g2 in ups:
                if w[0] == "k":
                    _, k, l = w; T[k, l] += st(Tu[k, l]) * g; Tu[k, l] += 1.0
                    for j in range(D): Gt[j, k, l, gv[j]] += st(Gu[j, k, l, gv[j]]) * g2; Gu[j, k, l, gv[j]] += 1.0
                else:
                    xl = w[1]; Tx[xl] += st(Txu[xl]) * g; Txu[xl] += 1.0
                    for j in range(D): Gx[j, xl, gv[j]] += st(Gxu[j, xl, gv[j]]) * g2; Gxu[j, xl, gv[j]] += 1.0
            if learn == 1:
                for i in ids[t]:
                    if i < -1: i = -2 - i
                    if i >= 0:
                        if C[i, y[t]] == 0: U[i] += 1
                        C[i, y[t]] += 1; N[i] += 1
    return P1, P2

def check():
    ok = True; rng = np.random.default_rng(6); L = kern(); L3 = B.kern()
    for S, shrink, D, withx, cascade, xg in ((14, True, 9, True, 1, 4), (14, True, 9, False, 1, 0), (65, False, 6, True, 1, 2), (14, True, 9, True, 0, 4)):
        K1, n, nc = 9, 3000, 150; ids = rng.integers(-1, nc, (n, K1)).astype(np.int32); ids[:, :3] = np.minimum(ids[:, :3], 5); sil = rng.random((n, K1)) < 0.1; ids[sil & (ids >= 0)] = -2 - ids[sil & (ids >= 0)]; y = rng.integers(0, S, n).astype(np.int32)
        cb = rng.integers(0, S, (n, D)).astype(np.int32); X = rng.normal(0, 1, (n, S)); X = (X - np.log(np.exp(X).sum(1, keepdims=True))).astype(np.float32); xlev = rng.integers(-1, 10, n).astype(np.int32)
        fresh = lambda: [np.zeros((nc, S), np.float32), np.zeros(nc, np.float32), np.zeros(nc, np.float32), np.full((K1, NL), 0.1), np.zeros((K1, NL)), np.zeros((D, K1, NL, S)), np.zeros((D, K1, NL, S)), np.zeros(XL), np.zeros(XL), np.zeros((D, XL, S)), np.zeros((D, XL, S))]
        A = fresh(); Bq = fresh(); Cm = fresh(); P1 = np.zeros((n, S)); P2 = np.zeros((n, S)); xa = (pf(X), p32(xlev)) if withx else (None, None)
        L.mix_steps4(n, K1, S, p32(ids), p32(y), pf(A[0]), pf(A[1]), pf(A[2]), pd(A[3]), pd(A[4]), 0.05, 0.3, 1, int(shrink), pd(P1), D, p32(cb), pd(A[5]), pd(A[6]), pd(P2), *xa, pd(A[7]), pd(A[8]), pd(A[9]), pd(A[10]), cascade, xg)
        R1, R2 = steps4_np(ids, y, cb, Bq[0], Bq[1], Bq[2], Bq[3], Bq[4], Bq[5], Bq[6], 0.05, 0.3, X if withx else None, xlev, Bq[7], Bq[8], Bq[9], Bq[10], cascade, xg, 1, shrink)
        d = max([np.abs(P1 - R1).max(), np.abs(P2 - R2).max()] + [np.abs(u - v).max() for u, v in zip(A, Bq)]); ngr = len({grp_np(v) for v in A[1] if v >= 1})
        Q1 = np.zeros((n, S)); Q2 = np.zeros((n, S)); L3.mix_steps3(n, K1, S, p32(ids), p32(y), pf(Cm[0]), pf(Cm[1]), pf(Cm[2]), pd(Cm[3]), pd(Cm[4]), 0.05, 0.3, 1, int(shrink), pd(Q1), D, p32(cb), pd(Cm[5]), pd(Cm[6]), pd(Q2), *xa, pd(Cm[7]), pd(Cm[8]), pd(Cm[9]), pd(Cm[10]))
        d3 = max(np.abs(P2 - Q2).max(), np.abs(A[3] - Cm[3]).max(), np.abs(A[5] - Cm[5]).max()); good = d < 1e-9 and ((d3 < 1e-9) if not cascade else (d3 > 1e-3 and ngr >= 3)); ok &= bool(good)
        print(f"groups by count (cascade = {cascade}), given sender {'given' if withx else 'absent'}, S = {S}, shrinking step = {shrink}, {n} bytes, cells in {ngr} groups at the end: compiled against numpy, largest difference {d:.1e}; against both.py's kernel {d3:.1e} ({'must agree' if not cascade else 'must differ'})", "PASSED" if good else "FAILED")
    return ok

def install(cascade):
    SysC.CASCADE = int(cascade); B.Sys3 = SysC; orig = B.reading
    def reading(cs, r, Ls, prevs, floor, arms):
        out = orig(cs, r, Ls, prevs, floor, arms)
        if cs is not None:                                                              # a look: the first cell's trusts by level of count (mean weighted by uses), base kinds and class kinds
            for nm, sl in (("base", slice(0, cs.KB)), ("class", slice(cs.KB, cs.K1))):
                T = cs.T[sl].reshape(-1, 10, 4); U = cs.Tu[sl].reshape(-1, 10, 4); out[f"look_trust_by_count_level_{nm}"] = [float((T[:, b] * U[:, b]).sum() / max(1.0, U[:, b].sum())) for b in range(10)]; out[f"look_uses_by_count_level_{nm}"] = [float(U[:, b].sum()) for b in range(10)]
            for th in (8, 64, 512):                                                       # a look: the second cell with the cells that have counted fewer than th times silent at the score (two of the five pasts)
                keep = cs.N.copy(); cs.N[cs.N < th] = 0
                out[f"look_cell2_only_counts_from_{th}"] = float(np.mean([B.answer(cs, r, Ls, pv, floor, 2, False)[0] for pv in prevs[:2]]))
                if r is not None: out[f"look_cell2_only_counts_from_{th}_relay_silent"] = float(np.mean([B.answer(cs, r, Ls, pv, floor, 2, True)[0] for pv in prevs[:2]]))
                cs.N[:] = keep
            keep = cs.N.copy(); cs.N[:cs.nbase] = 0; out["look_cell2_class_kinds_alone"] = float(np.mean([B.answer(cs, r, Ls, pv, floor, 2, True)[0] for pv in prevs[:2]])); cs.N[:] = keep
        return out
    B.reading = reading

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if (B.check() and check()) else 1)
    kw = dict(a.split("=") for a in sys.argv[2:]); install(kw.get("cascade", "1")); res, cs, r = B.run_sums(kw)
    tag = kw.get("tag") or "casc_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items())); json.dump(res, open(os.path.join(HERE, "res", tag + ".json"), "w")); print(tag, "| cpu", res["cpu"], flush=True)
