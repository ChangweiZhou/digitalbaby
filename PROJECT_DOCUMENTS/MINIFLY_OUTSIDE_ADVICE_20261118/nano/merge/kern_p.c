
#include <math.h>
#include <stdint.h>
#define NL 40
#define KMAX 400
#define SMAX 65
static inline int level(float n, float u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* shrink 1: the step falls with the uses of the trust; shrink 2 (addition 1): with the sum of the squares of the mismatch it has met */
static inline double step(double eta, double used, double met, int shrink) { return shrink == 2 ? eta / sqrt(1.0 + met) : shrink ? eta / sqrt(1.0 + used) : eta; }
void mix_steps_p(int64_t n, int K1, int KB, int S, const int32_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double *Tu, double *T2, double *Tu2, double *G, double *G2,
                 const int32_t *par, double eta, double e, int learn, int shrink, double *P) {
    static double lt[4096], s[KMAX][SMAX]; double a[SMAX], p[SMAX], a2[SMAX], p2[SMAX]; static int lev[KMAX], use2[KMAX], plist[KMAX]; static char mark[KMAX];
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int64_t t = 0; t < n; t++) {
        const int32_t *id = ids + t * K1; double mx = -1e300, z = 0;
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int k = 0; k < K1; k++) {
            lev[k] = -1; if (id[k] < 0 || N[id[k]] < 1) continue;
            const float *c = C + (int64_t)id[k] * S; lev[k] = level(N[id[k]], U[id[k]]); const double den = log(N[id[k]] + S * e);
            for (int v = 0; v < S; v++) s[k][v] = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den;
        }
        for (int k = 0; k < K1; k++) if (lev[k] >= 0) {       /* a class kind is on its second row where the member's own cell does not speak */
            use2[k] = k >= KB && par[k] >= 0 && lev[par[k]] < 0; const double tk = use2[k] ? T2[k * NL + lev[k]] : T[k * NL + lev[k]];
            for (int v = 0; v < S; v++) a[v] += tk * s[k][v];
        }
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (learn) {
            int np = 0;                                        /* practice: for each speaking R with class kinds beside it, the answer without that R-cell */
            for (int c = KB; c < K1; c++) if (lev[c] >= 0 && par[c] >= 0 && lev[par[c]] >= 0 && !mark[par[c]]) { mark[par[c]] = 1; plist[np++] = par[c]; }
            for (int i = 0; i < np; i++) {
                const int kp = plist[i]; const double tp = T[kp * NL + lev[kp]]; double m2 = -1e300, z2 = 0; mark[kp] = 0;
                for (int v = 0; v < S; v++) a2[v] = a[v] - tp * s[kp][v];
                for (int c = KB; c < K1; c++) if (lev[c] >= 0 && par[c] == kp) { const double d = T2[c * NL + lev[c]] - T[c * NL + lev[c]]; for (int v = 0; v < S; v++) a2[v] += d * s[c][v]; }
                for (int v = 0; v < S; v++) if (a2[v] > m2) m2 = a2[v];
                for (int v = 0; v < S; v++) { p2[v] = exp(a2[v] - m2); z2 += p2[v]; }
                for (int v = 0; v < S; v++) p2[v] /= z2;
                for (int c = KB; c < K1; c++) if (lev[c] >= 0 && par[c] == kp) {
                    double g = s[c][y[t]]; for (int v = 0; v < S; v++) g -= p2[v] * s[c][v];
                    T2[c * NL + lev[c]] += step(eta, Tu2[c * NL + lev[c]], G2[c * NL + lev[c]], shrink) * g; Tu2[c * NL + lev[c]] += 1.0; G2[c * NL + lev[c]] += g * g;
                }
            }
            for (int k = 0; k < K1; k++) if (lev[k] >= 0) {
                double g = s[k][y[t]]; for (int v = 0; v < S; v++) g -= p[v] * s[k][v];
                if (use2[k]) { T2[k * NL + lev[k]] += step(eta, Tu2[k * NL + lev[k]], G2[k * NL + lev[k]], shrink) * g; Tu2[k * NL + lev[k]] += 1.0; G2[k * NL + lev[k]] += g * g; }
                else { T[k * NL + lev[k]] += step(eta, Tu[k * NL + lev[k]], G[k * NL + lev[k]], shrink) * g; Tu[k * NL + lev[k]] += 1.0; G[k * NL + lev[k]] += g * g; }
            }
            for (int k = 0; k < K1; k++) if (id[k] >= 0) { float *c = C + (int64_t)id[k] * S; if (c[y[t]] == 0) U[id[k]] += 1; c[y[t]] += 1; N[id[k]] += 1; }
        }
    }
}
