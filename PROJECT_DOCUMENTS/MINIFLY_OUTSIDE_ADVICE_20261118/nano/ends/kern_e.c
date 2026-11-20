
#include <math.h>
#include <stdint.h>
#define S 65
#define NL 40
static inline int level(float n, float u, int flat) {
    if (flat) return 0;
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* trust.py's loop (nano/table) with one more output: Q[t], the probability given to the byte that arrived, formed before anything changes.
   at each byte: every active cell that has counted says s(v) = log((c(v)+e)/(n+65e)); the answering cell's branches compete on the sum of trust times s;
   then the trust that was used changes by eta times (mismatch . s), and the cells count */
void mix_steps(int64_t n, int64_t K1, const int64_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double eta, double e,
               int learn_c, int learn_t, int flat, double *P, double *Q) {
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
        if (Q) Q[t] = p[y[t]];
        if (learn_t) for (int64_t k = 0; k < K1; k++) if (lev[k] >= 0) {
            double g = s[k][y[t]]; for (int v = 0; v < S; v++) g -= p[v] * s[k][v];
            T[k * NL + lev[k]] += eta * g;
        }
        if (learn_c) for (int64_t k = 0; k < K1; k++) if (id[k] >= 0) {
            float *c = C + id[k] * S; if (c[y[t]] == 0) U[id[k]] += 1; c[y[t]] += 1; N[id[k]] += 1;
        }
    }
}
