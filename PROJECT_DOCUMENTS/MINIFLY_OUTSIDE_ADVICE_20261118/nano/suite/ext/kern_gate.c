
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#define NL 40
#define KMAX 400
#define SMAX 128
#define DMAX 16
static inline int level(float n, float u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
static int cmpf(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
/* the first cell exactly as nano/merge's mix_steps; the second cell's branches are the first cell's plus, for each speaking sender, (the sum of its
   parts) times the sender's message.  Parts: Gt[part][kind][level][gate value]; gate 0: part j reads the byte at delay j + 1; gate 1: every part reads 0;
   gate 2: one part, reading the number (0..3) of speaking senders that have counted fewer times than this one and at least twice. */
void mix_steps2(int64_t n, int K1, int S, const int32_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double *Tu,
                double eta, double e, int learn, int shrink, double *P, int D, const int32_t *cb, double *Gt, double *Gu, int gate, double *P2, double birth) {
    static double lt[4096], s[KMAX][SMAX]; double a[SMAX], p[SMAX], a2[SMAX], p2[SMAX]; static int lev[KMAX]; static int64_t gix[KMAX][DMAX]; static float srt[KMAX];
    const int NG = gate == 2 ? 4 : S, np = gate == 2 ? 1 : D;
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int64_t t = 0; t < n; t++) {
        const int32_t *id = ids + t * K1; double mx = -1e300, z = 0; int m = 0;
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int k = 0; k < K1; k++) {
            lev[k] = -1; if (id[k] < 0 || N[id[k]] < 1) continue;
            const float *c = C + (int64_t)id[k] * S; lev[k] = level(N[id[k]], U[id[k]]); const double tk = T[k * NL + lev[k]], den = log(N[id[k]] + S * e);
            for (int v = 0; v < S; v++) { const double sv = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den; s[k][v] = sv; a[v] += tk * sv; }
            if (gate == 2 && N[id[k]] >= 2) srt[m++] = N[id[k]];
        }
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        /* the second cell */
        if (gate == 2) qsort(srt, m, sizeof(float), cmpf);
        for (int v = 0; v < S; v++) a2[v] = a[v];
        for (int k = 0; k < K1; k++) if (lev[k] >= 0) {
            double tk = 0;
            for (int j = 0; j < np; j++) {
                int gv;
                if (gate == 0) gv = cb[t * D + j]; else if (gate == 1) gv = 0;
                else { const float nk = N[id[k]]; int lo = 0, hi = m; while (lo < hi) { const int mid = (lo + hi) / 2; if (srt[mid] < nk) lo = mid + 1; else hi = mid; } gv = lo > 3 ? 3 : lo; }
                gix[k][j] = (((int64_t)j * K1 + k) * NL + lev[k]) * NG + gv; tk += Gt[gix[k][j]];
            }
            if (tk != 0) for (int v = 0; v < S; v++) a2[v] += tk * s[k][v];
        }
        mx = -1e300; z = 0;
        for (int v = 0; v < S; v++) if (a2[v] > mx) mx = a2[v];
        for (int v = 0; v < S; v++) { p2[v] = exp(a2[v] - mx); z += p2[v]; }
        for (int v = 0; v < S; v++) p2[v] /= z;
        if (P2) for (int v = 0; v < S; v++) P2[t * S + v] = p2[v];
        if (learn) {
            for (int k = 0; k < K1; k++) if (lev[k] >= 0) {
                double g = s[k][y[t]], g2 = s[k][y[t]]; for (int v = 0; v < S; v++) { g -= p[v] * s[k][v]; g2 -= p2[v] * s[k][v]; }
                T[k * NL + lev[k]] += (shrink ? eta / sqrt(1.0 + Tu[k * NL + lev[k]]) : eta) * g; Tu[k * NL + lev[k]] += 1.0;
                for (int j = 0; j < np; j++) { const int64_t i = gix[k][j]; Gt[i] += (shrink ? eta / sqrt(1.0 + Gu[i]) : eta) * g2; Gu[i] += 1.0; }
            }
            for (int k = 0; k < K1; k++) { const int64_t i = id[k] < -1 ? -2 - (int64_t)id[k] : id[k];
                if (i >= 0) { if (birth > 0 && N[i] == 0 && p[y[t]] >= birth) continue;      /* PLAN_19.md: a cell is not born at a byte to which the first cell gave at least this probability */
                    float *c = C + i * S; if (c[y[t]] == 0) U[i] += 1; c[y[t]] += 1; N[i] += 1; } }
        }
    }
}
