#!/usr/bin/env python3
"""PLAN_1.md: look-ahead guesses used as lines, on prose. T2 of nano/table/PLAN_3.md (arm 2, step 0.00005, e = 0.02) with one change.
    python3 ahead.py check                       the compiled loop against the numpy reference on a prefix (three cases)
    python3 ahead.py dev  la=0|1 [lag=0] [etan=0.00005]     cells taught on the stream less its last twentieth; that twentieth scored
    python3 ahead.py life la=0|1 [lag=0] [etan=...] [scoreA=1]   the life A, B: the stream of the test text, then three books; each last tenth scored right after its own stream
la=0 is T2 itself. lag=64: every cell keyed on a guess is keyed on the guess made 64 bytes earlier (the control).
Places: step t names the byte x[p], p = lo + t. g3 is the guess of x[p+2], g2 of x[p+1].
PLAN_2.md: la=2 adds a fourth set of counts (the first letter of the word that begins next at or after p), the guess w from it, and kinds (f), (g), (h) keyed on w;
lagw=64: those three kinds are keyed on the w of 64 bytes earlier (the control of PLAN_2.md)."""
import ctypes, json, os, subprocess, sys, time
import numpy as np
import score, words
HERE = os.path.dirname(os.path.abspath(__file__)); S = 65; NL = 40; ETA, E0, K = 0.00005, 0.02, 6
DATA = os.path.join(HERE, "..", "..", "..", "..", "minifly_v148_pretrain_probe_dev_20261025", "data")
BT = ("bryant-stories.txt", "burgess-busterbrown.txt", "carroll-alice.txt")
NA = S * S + S * S * S; NB = S * S + 2 * S * S * S          # cells of kinds (a), (b); of kinds (c), (d), (e); addressed directly
PD = ctypes.POINTER(ctypes.c_double); PF = ctypes.POINTER(ctypes.c_float); PI = ctypes.POINTER(ctypes.c_int64); PY = ctypes.POINTER(ctypes.c_int32)

SRC = r'''
#include <math.h>
#include <stdint.h>
#include <string.h>
#define S 65
#define NL 40
#define KM 16
static double lt[4096]; static double E;
static inline int level(float n, float u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* what a cell says: s(v) = log((c(v)+e)/(n+65e)); returns its level of experience */
static inline int says(const float *c, float n, float u, double *s) {
    const double den = log(n + S * E);
    for (int v = 0; v < S; v++) s[v] = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + E)) - den;
    return level(n, u);
}
static inline int compete(const double *a, double *p) {
    double mx = -1e300, z = 0; int g = 0;
    for (int v = 0; v < S; v++) if (a[v] > mx) { mx = a[v]; g = v; }
    for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
    for (int v = 0; v < S; v++) p[v] /= z;
    return g;
}
static inline void count(float *C, float *N, float *U, int64_t i, int y) { float *c = C + i * S; if (c[y] == 0) U[i] += 1; c[y] += 1; N[i] += 1; }
/* xs points at the byte named at step 0; xs[-1], xs[-2] are the two bytes before it.
   CP, NP, UP: three sets for T2's cells (next byte, two ahead, three ahead). CA, NA_, UA: two sets for kinds (a), (b). CB, NB_, UB: kinds (c), (d), (e).
   TP[4][KP][NL], TA[2][2][NL], TB[3][NL].
   la = 2 (PLAN_2.md): CP[3] counts the first letter of the word that begins next; CC, NC, UC, TC[3][NL]: kinds (f), (g), (h) keyed on the guess w. */
void la_steps(int64_t n, int64_t KP, const int64_t *ids, const int32_t *xs,
              float **CP, float **NP, float **UP, float **CA, float **NA_, float **UA, float *CB, float *NB_, float *UB,
              double *TP, double *TA, double *TB, double eta, double etan, double e, int learn, int la, int lag, double *P, int32_t *G2, int32_t *G3,
              float *CC, float *NC, float *UC, double *TC, int lagw, int32_t *W) {
    static double s1[KM + 8][S], s2[2][KM + 2][S], s3[3][KM][S], sw[32][KM][S]; double a[S], p[S], q2[2][KM + 2], q3[3][KM], qw[32][KM];
    int l1[KM + 8], l2[2][KM + 2], l3[3][KM], lw[32][KM]; int64_t ida[2][2], idb[3], idc[3], pend_from = 0;
    E = e; for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    const int64_t OA1 = S * S, OB1 = S * S, OB2 = S * S + (int64_t)S * S * S;
    for (int64_t t = 0; t < n; t++) {
        const int64_t *id = ids + t * KP; const int x1 = xs[t - 1], x2 = xs[t - 2]; int g3 = 0, g2 = 0, k3 = 0, k2 = 0, kw = 0; int64_t *ia = ida[t % 2];
        if (la >= 2) {                         /* the guess of the next word's first letter: T2's cells' fourth counts */
            const int rw = t % 32;
            for (int v = 0; v < S; v++) a[v] = 0;
            for (int64_t k = 0; k < KP; k++) {
                lw[rw][k] = -1; if (id[k] < 0 || NP[3][id[k]] < 1) continue;
                lw[rw][k] = says(CP[3] + id[k] * S, NP[3][id[k]], UP[3][id[k]], sw[rw][k]); const double tk = TP[(3 * KP + k) * NL + lw[rw][k]];
                for (int v = 0; v < S; v++) a[v] += tk * sw[rw][k][v];
            }
            const int w = compete(a, p); W[t] = w; kw = lagw ? (t >= lagw ? W[t - lagw] : 0) : w;
            for (int64_t k = 0; k < KP; k++) if (lw[rw][k] >= 0) { double q = 0; for (int v = 0; v < S; v++) q += p[v] * sw[rw][k][v]; qw[rw][k] = q; }
        }
        if (la) {
            const int r3 = t % 3, r2 = t % 2;
            /* the guess three ahead: T2's cells' third counts */
            for (int v = 0; v < S; v++) a[v] = 0;
            for (int64_t k = 0; k < KP; k++) {
                l3[r3][k] = -1; if (id[k] < 0 || NP[2][id[k]] < 1) continue;
                l3[r3][k] = says(CP[2] + id[k] * S, NP[2][id[k]], UP[2][id[k]], s3[r3][k]); const double tk = TP[(2 * KP + k) * NL + l3[r3][k]];
                for (int v = 0; v < S; v++) a[v] += tk * s3[r3][k][v];
            }
            g3 = compete(a, p); G3[t] = g3; k3 = lag ? (t >= lag ? G3[t - lag] : 0) : g3;
            for (int64_t k = 0; k < KP; k++) if (l3[r3][k] >= 0) { double q = 0; for (int v = 0; v < S; v++) q += p[v] * s3[r3][k][v]; q3[r3][k] = q; }
            /* the guess two ahead: T2's cells' second counts and those of kinds (a), (b) */
            ia[0] = x1 * S + k3; ia[1] = OA1 + ((int64_t)x2 * S + x1) * S + k3;
            for (int v = 0; v < S; v++) a[v] = 0;
            for (int64_t k = 0; k < KP + 2; k++) {
                l2[r2][k] = -1; double tk;
                if (k < KP) { if (id[k] < 0 || NP[1][id[k]] < 1) continue; l2[r2][k] = says(CP[1] + id[k] * S, NP[1][id[k]], UP[1][id[k]], s2[r2][k]); tk = TP[(KP + k) * NL + l2[r2][k]]; }
                else { const int64_t i = ia[k - KP]; if (NA_[1][i] < 1) continue; l2[r2][k] = says(CA[1] + i * S, NA_[1][i], UA[1][i], s2[r2][k]); tk = TA[(2 + k - KP) * NL + l2[r2][k]]; }
                for (int v = 0; v < S; v++) a[v] += tk * s2[r2][k][v];
            }
            g2 = compete(a, p); G2[t] = g2; k2 = lag ? (t >= lag ? G2[t - lag] : 0) : g2;
            for (int64_t k = 0; k < KP + 2; k++) if (l2[r2][k] >= 0) { double q = 0; for (int v = 0; v < S; v++) q += p[v] * s2[r2][k][v]; q2[r2][k] = q; }
            idb[0] = x1 * S + k2; idb[1] = OB1 + ((int64_t)x2 * S + x1) * S + k2; idb[2] = OB2 + ((int64_t)x1 * S + k2) * S + k3;
            idc[0] = x1 * S + kw; idc[1] = OB1 + ((int64_t)x2 * S + x1) * S + kw; idc[2] = OB2 + ((int64_t)x1 * S + k2) * S + kw;
        }
        /* the answer */
        const int64_t K1 = la >= 2 ? KP + 8 : la ? KP + 5 : KP;
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int64_t k = 0; k < K1; k++) {
            l1[k] = -1; double tk;
            if (k < KP) { if (id[k] < 0 || NP[0][id[k]] < 1) continue; l1[k] = says(CP[0] + id[k] * S, NP[0][id[k]], UP[0][id[k]], s1[k]); tk = TP[k * NL + l1[k]]; }
            else if (k < KP + 2) { const int64_t i = ia[k - KP]; if (NA_[0][i] < 1) continue; l1[k] = says(CA[0] + i * S, NA_[0][i], UA[0][i], s1[k]); tk = TA[(k - KP) * NL + l1[k]]; }
            else if (k < KP + 5) { const int64_t i = idb[k - KP - 2]; if (NB_[i] < 1) continue; l1[k] = says(CB + i * S, NB_[i], UB[i], s1[k]); tk = TB[(k - KP - 2) * NL + l1[k]]; }
            else { const int64_t i = idc[k - KP - 5]; if (NC[i] < 1) continue; l1[k] = says(CC + i * S, NC[i], UC[i], s1[k]); tk = TC[(k - KP - 5) * NL + l1[k]]; }
            for (int v = 0; v < S; v++) a[v] += tk * s1[k][v];
        }
        compete(a, p);
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (!learn) continue;
        const int y = xs[t];
        /* trusts: the answer's now; a guess's when the guessed byte arrives */
        for (int64_t k = 0; k < K1; k++) if (l1[k] >= 0) {
            double g = s1[k][y]; for (int v = 0; v < S; v++) g -= p[v] * s1[k][v];
            if (k < KP) TP[k * NL + l1[k]] += eta * g; else if (k < KP + 2) TA[(k - KP) * NL + l1[k]] += etan * g; else if (k < KP + 5) TB[(k - KP - 2) * NL + l1[k]] += etan * g; else TC[(k - KP - 5) * NL + l1[k]] += etan * g;
        }
        if (la && t >= 1) { const int r = (t - 1) % 2;
            for (int64_t k = 0; k < KP + 2; k++) if (l2[r][k] >= 0) { const double g = s2[r][k][y] - q2[r][k]; if (k < KP) TP[(KP + k) * NL + l2[r][k]] += eta * g; else TA[(2 + k - KP) * NL + l2[r][k]] += etan * g; } }
        if (la && t >= 2) { const int r = (t - 2) % 3;
            for (int64_t k = 0; k < KP; k++) if (l3[r][k] >= 0) TP[(2 * KP + k) * NL + l3[r][k]] += eta * (s3[r][k][y] - q3[r][k]); }
        const int wstart = la >= 2 && y >= 13 && x1 < 13; const int64_t wfrom = pend_from > t - 31 ? pend_from : t - 31;      /* a word begins: the places that waited for its first letter, at most 32 */
        if (wstart) for (int64_t u = wfrom; u <= t; u++) { const int r = u % 32;
            for (int64_t k = 0; k < KP; k++) if (lw[r][k] >= 0) TP[(3 * KP + k) * NL + lw[r][k]] += eta * (sw[r][k][y] - qw[r][k]); }
        /* counts */
        for (int64_t k = 0; k < KP; k++) if (id[k] >= 0) count(CP[0], NP[0], UP[0], id[k], y);
        if (la) {
            for (int j = 0; j < 2; j++) count(CA[0], NA_[0], UA[0], ia[j], y);
            for (int j = 0; j < 3; j++) count(CB, NB_, UB, idb[j], y);
            if (t >= 1) { const int64_t *im = id - KP; for (int64_t k = 0; k < KP; k++) if (im[k] >= 0) count(CP[1], NP[1], UP[1], im[k], y);
                          for (int j = 0; j < 2; j++) count(CA[1], NA_[1], UA[1], ida[(t - 1) % 2][j], y); }
            if (t >= 2) { const int64_t *im = id - 2 * KP; for (int64_t k = 0; k < KP; k++) if (im[k] >= 0) count(CP[2], NP[2], UP[2], im[k], y); }
        }
        if (la >= 2) for (int j = 0; j < 3; j++) count(CC, NC, UC, idc[j], y);
        if (wstart) { for (int64_t u = wfrom; u <= t; u++) { const int64_t *iu = ids + u * KP; for (int64_t k = 0; k < KP; k++) if (iu[k] >= 0) count(CP[3], NP[3], UP[3], iu[k], y); } pend_from = t + 1; }
    }
}
'''
def kern():
    so = os.path.join(HERE, "kern_la.so"); src = os.path.join(HERE, "kern_la.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so + ".tmp%d" % os.getpid(), src, "-lm"]); os.replace(so + ".tmp%d" % os.getpid(), so)
    L = ctypes.CDLL(so); PPF = ctypes.POINTER(PF)
    L.la_steps.argtypes = [ctypes.c_int64, ctypes.c_int64, PI, PY, PPF, PPF, PPF, PPF, PPF, PPF, PF, PF, PF, PD, PD, PD, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, PD, PY, PY, PF, PF, PF, PD, ctypes.c_int, PY]; L.la_steps.restype = None; return L

def level_np(n, u):
    bn = 0; m = int(n)
    while m > 1 and bn < 9: m >>= 1; bn += 1
    return bn * 4 + (0 if u <= 1 else 1 if u <= 2 else 2 if u <= 4 else 3)

class Sys:
    """T2's cells with three sets of counts, the five added kinds, and the answering cell's trusts. ids are built over the whole of x."""
    def __init__(s, x, la=1, lag=0, keys=None, lagw=0):
        s.x = np.ascontiguousarray(x.astype(np.int32)); s.la, s.lag, s.lagw = int(la), int(lag), int(lagw); s.ids, s.ncell, s.kinds, _ = words.make_ids(x, words.word_keys(x) if keys is None else keys, K, len(x), len(x), K, 1); s.KP = s.ids.shape[1]; s.L = kern()
        ns = 4 if la >= 2 else 3 if la else 1; s.CP = [np.zeros((s.ncell, S), np.float32) for _ in range(ns)]; s.NP = [np.zeros(s.ncell, np.float32) for _ in range(ns)]; s.UP = [np.zeros(s.ncell, np.float32) for _ in range(ns)]
        na, nb = (NA, NB) if la else (1, 1); s.CA = [np.zeros((na, S), np.float32) for _ in range(2)]; s.NA = [np.zeros(na, np.float32) for _ in range(2)]; s.UA = [np.zeros(na, np.float32) for _ in range(2)]
        s.CB = np.zeros((nb, S), np.float32); s.NB = np.zeros(nb, np.float32); s.UB = np.zeros(nb, np.float32)
        s.TP = np.full((4, s.KP, NL), 0.25); s.TA = np.zeros((2, 2, NL)); s.TB = np.zeros((3, NL))                    # the added kinds are born with no trust
        nc = NB if la >= 2 else 1; s.CC = np.zeros((nc, S), np.float32); s.NC = np.zeros(nc, np.float32); s.UC = np.zeros(nc, np.float32); s.TC = np.zeros((3, NL))
    def go(s, lo, hi, etan, learn, want_p=False):
        """Bytes lo..hi-1 are named (lo >= K); with learn the trusts and the counts change. Returns P (or None), g2, g3, w."""
        n = hi - lo; ids = np.ascontiguousarray(s.ids[lo - K:hi - K]); xs = np.ascontiguousarray(s.x[lo - 2:hi]); P = np.zeros((n, S)) if want_p else None; G2 = np.zeros(n, np.int32); G3 = np.zeros(n, np.int32); W = np.zeros(n, np.int32)
        arr = lambda lst: (PF * 4)(*[a.ctypes.data_as(PF) for a in lst] + [None] * (4 - len(lst)))
        s.L.la_steps(n, s.KP, ids.ctypes.data_as(PI), ctypes.cast(xs.ctypes.data + 8, PY), arr(s.CP), arr(s.NP), arr(s.UP), arr(s.CA), arr(s.NA), arr(s.UA), s.CB.ctypes.data_as(PF), s.NB.ctypes.data_as(PF), s.UB.ctypes.data_as(PF),
                     s.TP.ctypes.data_as(PD), s.TA.ctypes.data_as(PD), s.TB.ctypes.data_as(PD), ETA, etan, E0, int(learn), s.la, s.lag, P.ctypes.data_as(PD) if want_p else None, G2.ctypes.data_as(PY), G3.ctypes.data_as(PY),
                     s.CC.ctypes.data_as(PF), s.NC.ctypes.data_as(PF), s.UC.ctypes.data_as(PF), s.TC.ctypes.data_as(PD), s.lagw, W.ctypes.data_as(PY))
        return P, G2, G3, W
    def sizes(s):
        r = dict(cells_T2=int((s.NP[0] > 0).sum()), counts_above_zero_first_set=int((s.CP[0] > 0).sum()))
        if s.la:
            r.update(counts_above_zero_second_set=int((s.CP[1] > 0).sum()), counts_above_zero_third_set=int((s.CP[2] > 0).sum()),
                     cells_a=int((s.NA[0][:S * S] > 0).sum()), cells_b=int((s.NA[0][S * S:] > 0).sum()), cells_c=int((s.NB[:S * S] > 0).sum()), cells_d=int((s.NB[S * S:S * S + S ** 3] > 0).sum()), cells_e=int((s.NB[S * S + S ** 3:] > 0).sum()),
                     trust_added_answer=dict(a=np.round(s.TA[0, 0], 3).tolist(), b=np.round(s.TA[0, 1], 3).tolist(), c=np.round(s.TB[0], 3).tolist(), d=np.round(s.TB[1], 3).tolist(), e=np.round(s.TB[2], 3).tolist()),
                     trust_added_guess_two_ahead=dict(a=np.round(s.TA[1, 0], 3).tolist(), b=np.round(s.TA[1, 1], 3).tolist()),
                     trust_added_largest=dict(a=float(np.abs(s.TA[0, 0]).max()), b=float(np.abs(s.TA[0, 1]).max()), c=float(np.abs(s.TB[0]).max()), d=float(np.abs(s.TB[1]).max()), e=float(np.abs(s.TB[2]).max())))
        if s.la >= 2:
            r.update(counts_above_zero_fourth_set=int((s.CP[3] > 0).sum()), cells_f=int((s.NC[:S * S] > 0).sum()), cells_g=int((s.NC[S * S:S * S + S ** 3] > 0).sum()), cells_h=int((s.NC[S * S + S ** 3:] > 0).sum()),
                     trust_w_kinds=dict(f=np.round(s.TC[0], 3).tolist(), g=np.round(s.TC[1], 3).tolist(), h=np.round(s.TC[2], 3).tolist()), trust_w_largest=dict(f=float(np.abs(s.TC[0]).max()), g=float(np.abs(s.TC[1]).max()), h=float(np.abs(s.TC[2]).max())))
        return r

def steps_np(ids, x, lo, hi, etan, la, lag, learn=True, lagw=0):
    """The reference: the same loop in plain numpy with the cells of the added kinds in dictionaries. Returns P, G2, G3 and the state."""
    KP = ids.shape[1]; nc = int(ids.max()) + 1; n = hi - lo; e = E0
    CP = [np.zeros((nc, S)) for _ in range(4)]; TP = np.full((4, KP, NL), 0.25); TA = np.zeros((2, 2, NL)); TB = np.zeros((3, NL)); TC = np.zeros((3, NL)); DA = [dict(), dict()]; DB = dict(); DC = dict()
    P = np.zeros((n, S)); G2 = np.zeros(n, np.int64); G3 = np.zeros(n, np.int64); W = np.zeros(n, np.int64); pend2 = {}; pend3 = {}; aids = {}; pendw = {}
    def says(c): nn = c.sum(); return np.log(c + e) - np.log(nn + S * e), level_np(nn, (c > 0).sum())
    def comp(a): p = np.exp(a - a.max()); return int(np.argmax(a)), p / p.sum()
    for t in range(n):
        p_ = lo + t; row = ids[p_ - K]; x1, x2 = int(x[p_ - 1]), int(x[p_ - 2]); used1 = []; a = np.zeros(S); ia = ib = ic = ()
        if la >= 2:
            uw = []; aw = np.zeros(S)
            for k in range(KP):
                i = row[k]
                if i >= 0 and CP[3][i].sum() >= 1: sv, l = says(CP[3][i]); uw.append((k, l, sv)); aw += TP[3, k, l] * sv
            w, pw = comp(aw); W[t] = w; pendw[t] = [(k, l, sv, pw @ sv) for k, l, sv in uw]; kw = (int(W[t - lagw]) if t >= lagw else 0) if lagw else w
            for u in [u for u in pendw if u < t - 31]: del pendw[u]                    # at most the last 32 places wait
        if la:
            u3 = []; a3 = np.zeros(S)
            for k in range(KP):
                i = row[k]
                if i >= 0 and CP[2][i].sum() >= 1: sv, l = says(CP[2][i]); u3.append((k, l, sv)); a3 += TP[2, k, l] * sv
            g3, p3 = comp(a3); G3[t] = g3; pend3[t] = [(k, l, sv, p3 @ sv) for k, l, sv in u3]; k3 = (int(G3[t - lag]) if t >= lag else 0) if lag else g3
            ia = (("a", x1, k3), ("b", x2, x1, k3)); aids[t] = ia; u2 = []; a2 = np.zeros(S)
            for k in range(KP):
                i = row[k]
                if i >= 0 and CP[1][i].sum() >= 1: sv, l = says(CP[1][i]); u2.append(("P", k, l, sv)); a2 += TP[1, k, l] * sv
            for j in range(2):
                c = DA[1].get(ia[j])
                if c is not None and c.sum() >= 1: sv, l = says(c); u2.append(("A", j, l, sv)); a2 += TA[1, j, l] * sv
            g2, p2 = comp(a2); G2[t] = g2; pend2[t] = [(f, k, l, sv, p2 @ sv) for f, k, l, sv in u2]; k2 = (int(G2[t - lag]) if t >= lag else 0) if lag else g2
            ib = (("c", x1, k2), ("d", x2, x1, k2), ("e", x1, k2, k3))
            if la >= 2: ic = (("f", x1, kw), ("g", x2, x1, kw), ("h", x1, k2, kw))
        for k in range(KP):
            i = row[k]
            if i >= 0 and CP[0][i].sum() >= 1: sv, l = says(CP[0][i]); used1.append(("P", k, l, sv)); a += TP[0, k, l] * sv
        for j in range(len(ia)):
            c = DA[0].get(ia[j])
            if c is not None and c.sum() >= 1: sv, l = says(c); used1.append(("A", j, l, sv)); a += TA[0, j, l] * sv
        for j in range(len(ib)):
            c = DB.get(ib[j])
            if c is not None and c.sum() >= 1: sv, l = says(c); used1.append(("B", j, l, sv)); a += TB[j, l] * sv
        for j in range(len(ic)):
            c = DC.get(ic[j])
            if c is not None and c.sum() >= 1: sv, l = says(c); used1.append(("C", j, l, sv)); a += TC[j, l] * sv
        _, p = comp(a); P[t] = p
        if not learn: continue
        y = int(x[p_])
        for f, k, l, sv in used1:
            g = sv[y] - p @ sv
            if f == "P": TP[0, k, l] += ETA * g
            elif f == "A": TA[0, k, l] += etan * g
            elif f == "B": TB[k, l] += etan * g
            else: TC[k, l] += etan * g
        if la and t >= 1:
            for f, k, l, sv, q in pend2.pop(t - 1):
                if f == "P": TP[1, k, l] += ETA * (sv[y] - q)
                else: TA[1, k, l] += etan * (sv[y] - q)
        if la and t >= 2:
            for k, l, sv, q in pend3.pop(t - 2): TP[2, k, l] += ETA * (sv[y] - q)
        wstart = la >= 2 and y >= 13 and x1 < 13; waiting = sorted(pendw) if wstart else []
        for u in waiting:
            for k, l, sv, q in pendw[u]: TP[3, k, l] += ETA * (sv[y] - q)
        for k in range(KP):
            if row[k] >= 0: CP[0][row[k], y] += 1
        if la:
            for key in ia: DA[0].setdefault(key, np.zeros(S))[y] += 1
            for key in ib: DB.setdefault(key, np.zeros(S))[y] += 1
            if t >= 1:
                for i in ids[p_ - 1 - K]:
                    if i >= 0: CP[1][i, y] += 1
                for key in aids[t - 1]: DA[1].setdefault(key, np.zeros(S))[y] += 1
            if t >= 2:
                for i in ids[p_ - 2 - K]:
                    if i >= 0: CP[2][i, y] += 1
        for key in ic: DC.setdefault(key, np.zeros(S))[y] += 1
        for u in waiting:
            for i in ids[lo + u - K]:
                if i >= 0: CP[3][i, y] += 1
        if wstart: pendw.clear()
    return P, G2, G3, CP, TP, TA, TB, W, TC

def check():
    x, cut, _ = score.load(); n = 3000; x = x[:n]; ok = True
    for la, lag, etan, lagw in ((0, 0, 0.0005, 0), (1, 0, 0.0005, 0), (1, 64, 0.0005, 0), (1, 0, 0.05, 0), (2, 0, 0.0005, 0), (2, 0, 0.05, 64), (2, 0, 0.05, 0)):
        s = Sys(x, la, lag, lagw=lagw); m = 2400; P2, g2, g3, w2 = s.go(K, m, etan, True, True)
        P1, G2, G3, CP, TP, TA, TB, W, TC = steps_np(s.ids, x, K, m, etan, la, lag, True, lagw); ns = len(s.CP)
        d = max(np.abs(P1 - P2).max(), np.abs(TP[:ns] - s.TP[:ns]).max(), np.abs(TA - s.TA).max(), np.abs(TB - s.TB).max(), np.abs(TC - s.TC).max(), max(np.abs(CP[j] - s.CP[j]).max() for j in range(ns)))
        same = bool((G2 == g2).all() and (G3 == g3).all() and (W == w2).all())
        print(f"look-ahead = {la}, lag = {lag}, lag of w = {lagw}, step of the added trusts = {etan}: {m - K} bytes read, compiled against numpy: largest difference {d:.1e}, guesses equal: {same}, largest added trust {max(np.abs(s.TA).max(), np.abs(s.TB).max(), np.abs(s.TC).max()):.3f}", "PASSED" if d < 1e-9 and same else "FAILED", flush=True); ok &= d < 1e-9 and same
    return ok

def load_all():
    """As nano/table/life.py: A = the test text; then three books. A byte that A does not have becomes a space; carriage returns are removed."""
    xa, cut, _ = score.load(); vals = score.byte_values(); tab = np.full(256, 1, np.int64); tab[vals] = np.arange(S); parts = [("A", xa)]
    for f in BT:
        raw = np.frombuffer(open(os.path.join(DATA, f), "rb").read(), np.uint8); raw = raw[raw != 13]; parts.append((f.split(".")[0], tab[raw]))
    x = np.concatenate([p[1] for p in parts]); segs = {}; off = 0
    for name, p in parts:
        c = cut if name == "A" else int(0.9 * len(p)); segs[name] = (off, off + c, off + len(p)); off += len(p)
    return x, segs

def guesses(x, c, b, P, G2, G3, W=None):
    """How often the guesses are right on x[c:b], by where the guessed byte stands: the first letter of a word, a later letter, or another byte. A letter is a byte index of 13 or more."""
    out = {}
    for name, G, d in (("answer", P.argmax(1), 0), ("two_ahead", G2, 1), ("three_ahead", G3, 2)):
        t = np.arange(0, b - c - d); tgt = x[c + t + d]; before = x[c + t + d - 1]; ok = G[t] == tgt; first = (tgt >= 13) & (before < 13); later = (tgt >= 13) & (before >= 13); other = tgt < 13
        out[name] = dict(all=round(100 * float(ok.mean()), 2), first_letter_of_a_word=round(100 * float(ok[first].mean()), 2), later_letter=round(100 * float(ok[later].mean()), 2), other_byte=round(100 * float(ok[other].mean()), 2),
                         share_first=round(100 * float(first.mean()), 2), share_later=round(100 * float(later.mean()), 2), share_other=round(100 * float(other.mean()), 2))
    if W is not None:                                     # w against the first letter of the word that begins next at or after each place, by where the NEXT byte stands
        y = x[c:b]; before = x[c - 1:b - 1]; ws = (y >= 13) & (before < 13); nxt = np.where(ws, np.arange(b - c), b - c); nxt = np.minimum.accumulate(nxt[::-1])[::-1]; has = nxt < b - c; tgt = y[np.minimum(nxt, b - c - 1)]; ok = (W == tgt)
        first = ws & has; later = (y >= 13) & ~ws & has; other = (y < 13) & has
        out["next_word_first_letter"] = dict(all=round(100 * float(ok[has].mean()), 2), next_byte_is_first_letter=round(100 * float(ok[first].mean()), 2), next_byte_is_later_letter=round(100 * float(ok[later].mean()), 2), next_byte_is_other=round(100 * float(ok[other].mean()), 2),
                                             most_common_first_letter_share=round(100 * float(np.bincount(y[ws]).max() / ws.sum()), 2))
    return out

def where(x, c, b, P):
    """The loss on x[c:b] by where the named byte stands."""
    y = x[c:b]; l = -np.log(P[np.arange(b - c), y]); before = x[c - 1:b - 1]; first = (y >= 13) & (before < 13); later = (y >= 13) & (before >= 13); other = y < 13
    return dict(nats_first_letter=float(l[first].mean()), nats_later_letter=float(l[later].mean()), nats_other=float(l[other].mean()), share_of_loss_first=round(100 * float(l[first].sum() / l.sum()), 1))

def dev(la, lag, etan, lagw=0):
    x, cut, _ = score.load(); d = cut - cut // 20; t0 = time.process_time(); s = Sys(x[:cut], la, lag, lagw=lagw); s.go(K, d, etan, True); P, G2, G3, W = s.go(d, cut, etan, False, True)
    r = dict(score.score(P, x[d:cut])); r.update(what="last twentieth of the stream, cells taught on the stream before it", la=la, lag=lag, etan=etan, where=where(x, d, cut, P), cpu=round(time.process_time() - t0, 1))
    if la: r["guesses"] = guesses(x, d, cut, P, G2, G3, W if la >= 2 else None)
    r.update(s.sizes()); return r

def life(la, lag, etan, scoreA, lagw=0):
    x, segs = load_all(); t0 = time.process_time(); s = Sys(x, la, lag, lagw=lagw); r = dict(la=la, lag=lag, lagw=lagw, etan=etan, segments={k: list(map(int, v)) for k, v in segs.items()}, nats={}, right={}, where={}, guesses={}, sizes={})
    for name, (a, c, b) in segs.items():
        s.go(max(a, K), c, etan, True)
        if name == "A" and not scoreA: r["sizes"][name] = s.sizes(); continue
        P, G2, G3, W = s.go(c, b, etan, False, True); sc = score.score(P, x[c:b]); r["nats"][name] = sc["nats"]; r["right"][name] = sc["right"]; r["where"][name] = where(x, c, b, P); r["sizes"][name] = s.sizes()
        if la: r["guesses"][name] = guesses(x, c, b, P, G2, G3, W if la >= 2 else None)
    r["cpu"] = round(time.process_time() - t0, 1); return r

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=") for a in sys.argv[2:]); la = int(kw.get("la", 1)); lag = int(kw.get("lag", 0)); etan = float(kw.get("etan", ETA)); tag = sys.argv[1] + "_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    lagw = int(kw.get("lagw", 0)); r = dev(la, lag, etan, lagw) if sys.argv[1] == "dev" else life(la, lag, etan, int(kw.get("scoreA", 1)), lagw)
    json.dump(r, open(os.path.join(HERE, "res", tag + ".json"), "w"), indent=1)
    print(tag, "| nats", round(r["nats"], 4) if sys.argv[1] == "dev" else {k[:6]: round(v, 4) for k, v in r["nats"].items()}, "| largest added trust", (r if sys.argv[1] == "dev" else list(r["sizes"].values())[-1]).get("trust_added_largest"), (r if sys.argv[1] == "dev" else list(r["sizes"].values())[-1]).get("trust_w_largest"), "| cpu", r["cpu"], flush=True)
