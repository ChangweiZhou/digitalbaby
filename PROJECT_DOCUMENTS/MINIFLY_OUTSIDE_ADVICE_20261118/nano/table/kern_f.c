
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
