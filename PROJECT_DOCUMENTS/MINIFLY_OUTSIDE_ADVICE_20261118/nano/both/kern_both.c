
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#define NL 40
#define XL 10
#define KMAX 400
#define SMAX 65
#define DMAX 16
static inline int level(float n, float u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* nano/table/gate.py's mix_steps2 at gate 0, with one more sender whose message X[t] (S numbers) and level xlev[t] (0..9; below 0: silent) are given.
   First cell: its trust Tx[level]. Second cell: its parts Gx[delay][level][byte at that delay]. Same rule and step as the other senders.
   learn: 1 everything learns; 2 only the trusts and parts learn (nothing is counted). */
void mix_steps3(int64_t n, int K1, int S, const int32_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double *Tu,
                double eta, double e, int learn, int shrink, double *P, int D, const int32_t *cb, double *Gt, double *Gu, double *P2,
                const float *X, const int32_t *xlev, double *Tx, double *Txu, double *Gx, double *Gxu) {
    static double lt[4096], s[KMAX][SMAX]; double a[SMAX], p[SMAX], a2[SMAX], p2[SMAX]; static int lev[KMAX]; static int64_t gix[KMAX][DMAX]; int64_t xix[DMAX];
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int64_t t = 0; t < n; t++) {
        const int32_t *id = ids + t * K1; double mx = -1e300, z = 0; const int xl = (X && xlev[t] >= 0) ? xlev[t] : -1; const float *sx = X ? X + t * S : 0;
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int k = 0; k < K1; k++) {
            lev[k] = -1; if (id[k] < 0 || N[id[k]] < 1) continue;
            const float *c = C + (int64_t)id[k] * S; lev[k] = level(N[id[k]], U[id[k]]); const double tk = T[k * NL + lev[k]], den = log(N[id[k]] + S * e);
            for (int v = 0; v < S; v++) { const double sv = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den; s[k][v] = sv; a[v] += tk * sv; }
        }
        if (xl >= 0) { const double tk = Tx[xl]; for (int v = 0; v < S; v++) a[v] += tk * sx[v]; }
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        for (int v = 0; v < S; v++) a2[v] = a[v];
        for (int k = 0; k < K1; k++) if (lev[k] >= 0) {
            double tk = 0;
            for (int j = 0; j < D; j++) { gix[k][j] = (((int64_t)j * K1 + k) * NL + lev[k]) * S + cb[t * D + j]; tk += Gt[gix[k][j]]; }
            if (tk != 0) for (int v = 0; v < S; v++) a2[v] += tk * s[k][v];
        }
        if (xl >= 0) { double tk = 0; for (int j = 0; j < D; j++) { xix[j] = ((int64_t)j * XL + xl) * S + cb[t * D + j]; tk += Gx[xix[j]]; }
            if (tk != 0) for (int v = 0; v < S; v++) a2[v] += tk * sx[v]; }
        mx = -1e300; z = 0;
        for (int v = 0; v < S; v++) if (a2[v] > mx) mx = a2[v];
        for (int v = 0; v < S; v++) { p2[v] = exp(a2[v] - mx); z += p2[v]; }
        for (int v = 0; v < S; v++) p2[v] /= z;
        if (P2) for (int v = 0; v < S; v++) P2[t * S + v] = p2[v];
        if (learn) {
            for (int k = 0; k < K1; k++) if (lev[k] >= 0) {
                double g = s[k][y[t]], g2 = s[k][y[t]]; for (int v = 0; v < S; v++) { g -= p[v] * s[k][v]; g2 -= p2[v] * s[k][v]; }
                T[k * NL + lev[k]] += (shrink ? eta / sqrt(1.0 + Tu[k * NL + lev[k]]) : eta) * g; Tu[k * NL + lev[k]] += 1.0;
                for (int j = 0; j < D; j++) { const int64_t i = gix[k][j]; Gt[i] += (shrink ? eta / sqrt(1.0 + Gu[i]) : eta) * g2; Gu[i] += 1.0; }
            }
            if (xl >= 0) {
                double g = sx[y[t]], g2 = sx[y[t]]; for (int v = 0; v < S; v++) { g -= p[v] * sx[v]; g2 -= p2[v] * sx[v]; }
                Tx[xl] += (shrink ? eta / sqrt(1.0 + Txu[xl]) : eta) * g; Txu[xl] += 1.0;
                for (int j = 0; j < D; j++) { const int64_t i = xix[j]; Gx[i] += (shrink ? eta / sqrt(1.0 + Gxu[i]) : eta) * g2; Gxu[i] += 1.0; }
            }
            if (learn == 1) for (int k = 0; k < K1; k++) { const int64_t i = id[k] < -1 ? -2 - (int64_t)id[k] : id[k];
                if (i >= 0) { float *c = C + i * S; if (c[y[t]] == 0) U[i] += 1; c[y[t]] += 1; N[i] += 1; } }
        }
    }
}
