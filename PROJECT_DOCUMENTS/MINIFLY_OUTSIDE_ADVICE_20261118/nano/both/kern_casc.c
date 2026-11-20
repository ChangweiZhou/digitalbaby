
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#define NL 40
#define XL 10
#define KMAX 400
#define SMAX 65
#define DMAX 16
#define GB 40
static int cmpf(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
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
                const float *X, const int32_t *xlev, double *Tx, double *Txu, double *Gx, double *Gxu, int cascade, int xg, int half) {
    static float cnt[KMAX]; static double lt[4096], s[KMAX][SMAX], ga[GB][SMAX], gb2[GB][SMAX], pg[GB][SMAX], pg2[GB][SMAX]; static int lev[KMAX], gk[KMAX]; static int64_t gix[KMAX][DMAX];
    int64_t xix[DMAX]; int has[GB]; double r1[SMAX], r2[SMAX], uni[SMAX];
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int v = 0; v < S; v++) uni[v] = 1.0 / S;
    for (int64_t t = 0; t < n; t++) {
        const int32_t *id = ids + t * K1; const int xl = (X && xlev[t] >= 0) ? xlev[t] : -1; const float *sx = X ? X + t * S : 0; const int xgk = cascade ? xg : 0;
        for (int b = 0; b < GB; b++) has[b] = 0;
        float thr = 0;                                     /* half: only the speakers that have counted at least the middle count among the speakers are heard */
        if (half) { int m = 0; for (int k = 0; k < K1; k++) if (id[k] >= 0 && N[id[k]] >= 1) cnt[m++] = N[id[k]]; if (m) { qsort(cnt, m, sizeof(float), cmpf); thr = cnt[m / 2]; } }
#define OPEN(b) if (!has[b]) { has[b] = 1; for (int v = 0; v < S; v++) { ga[b][v] = 0; gb2[b][v] = 0; } }
        for (int k = 0; k < K1; k++) {
            lev[k] = -1; if (id[k] < 0 || N[id[k]] < 1 || N[id[k]] < thr) continue;
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
