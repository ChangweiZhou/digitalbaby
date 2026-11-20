
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
void rank_steps(int64_t n, int64_t K1, const int64_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double eta, double e, int learn, double *P, int use) {
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
            ix[k] = k * NL * NR + NL * r + lev[k]; const double tk = T[ix[k]];
            for (int v = 0; v < S; v++) a[v] += tk * s[k][v];
        }
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (learn) {
            for (int64_t k = 0; k < K1; k++) if (lev[k] >= 0) { double g = s[k][y[t]]; for (int v = 0; v < S; v++) g -= p[v] * s[k][v]; T[ix[k]] += eta * g; }
            for (int64_t k = 0; k < K1; k++) if (id[k] >= 0) { float *c = C + id[k] * S; if (c[y[t]] == 0) U[id[k]] += 1; c[y[t]] += 1; N[id[k]] += 1; }
        }
    }
}
