
#include <math.h>
#include <stdint.h>
#define S 65
#define NL 40
static inline int level(double n, int u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* as mix_steps of trust.py; a kind with half[k] > 0 is a kind of companions: before a companion speaks or counts, its counts are multiplied by
   2^(-(time since it was last active)/half[k]); it is silent when its faded total is under one; its level of u is the number of counts of 1/2 or more.
   With learn = 0 nothing changes and the time stands at clock0. */
void gate_steps(int64_t n, int64_t K1, const int64_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double eta, double e,
               int learn, double *P, const double *half, double *last, double clock0, const float *X, const int32_t *xlev, const int32_t *cb, int D, double *Gt, int gate, double *P2, int nolevel) {
    static double lt[4096]; double a[S], p[S], a2[S], p2[S], s[192][S]; int lev[192]; int64_t gix[193][16];
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int64_t t = 0; t < n; t++) {
        const int64_t *id = ids + t * K1; double mx = -1e300, z = 0; const double clock = clock0 + (learn ? (double)t : 0.0);
        for (int v = 0; v < S; v++) a[v] = 0;
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
                lev[k] = level(nn, u); const double tk = T[k * NL + lev[k]];
                for (int v = 0; v < S; v++) a[v] += tk * s[k][v];
            } else {
                if (N[i] < 1) continue;
                lev[k] = level(N[i], (int)U[i]); const double tk = T[k * NL + lev[k]], den = log(N[i] + S * e);
                for (int v = 0; v < S; v++) { const double sv = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den; s[k][v] = sv; a[v] += tk * sv; }
            }
        }
        if (X) { const float *sx = X + t * S; const double tk = T[K1 * NL + xlev[t]]; for (int v = 0; v < S; v++) a[v] += tk * sx[v]; }      /* one more sender whose message is given (relay.py) */
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        for (int v = 0; v < S; v++) a2[v] = a[v];
        for (int64_t k = 0; k <= K1; k++) {
            const int lk = k < K1 ? lev[k] : (X ? xlev[t] : -1); if (lk < 0) continue;
            const double *sk = k < K1 ? s[k] : 0; const float *sx = X ? X + t * S : 0; double tk = 0;
            for (int j = 0; j < D; j++) { gix[k][j] = (((int64_t)j * (K1 + 1) + k) * NL + (nolevel ? 0 : lk)) * S + (gate ? 0 : cb[t * D + j]); tk += Gt[gix[k][j]]; }
            if (tk != 0) { if (sk) for (int v = 0; v < S; v++) a2[v] += tk * sk[v]; else for (int v = 0; v < S; v++) a2[v] += tk * sx[v]; }
        }
        { double m2 = -1e300, z2 = 0; for (int v = 0; v < S; v++) if (a2[v] > m2) m2 = a2[v];
          for (int v = 0; v < S; v++) { p2[v] = exp(a2[v] - m2); z2 += p2[v]; }
          for (int v = 0; v < S; v++) p2[v] /= z2; }
        if (P2) for (int v = 0; v < S; v++) P2[t * S + v] = p2[v];
        if (learn) {
            for (int64_t k = 0; k <= K1; k++) {
                const int lk = k < K1 ? lev[k] : (X ? xlev[t] : -1); if (lk < 0) continue;
                double g2;
                if (k < K1) { g2 = s[k][y[t]]; for (int v = 0; v < S; v++) g2 -= p2[v] * s[k][v]; }
                else { const float *sx = X + t * S; g2 = sx[y[t]]; for (int v = 0; v < S; v++) g2 -= p2[v] * sx[v]; }
                for (int j = 0; j < D; j++) Gt[gix[k][j]] += eta * g2;
            }
            if (X) { const float *sx = X + t * S; double g = sx[y[t]]; for (int v = 0; v < S; v++) g -= p[v] * sx[v]; T[K1 * NL + xlev[t]] += eta * g; }
            for (int64_t k = 0; k < K1; k++) if (lev[k] >= 0) {
                double g = s[k][y[t]]; for (int v = 0; v < S; v++) g -= p[v] * s[k][v];
                T[k * NL + lev[k]] += eta * g;
            }
            for (int64_t k = 0; k < K1; k++) if (id[k] >= 0) {
                float *c = C + id[k] * S; if (half[k] > 0) last[id[k]] = clock; else if (c[y[t]] == 0) U[id[k]] += 1;
                c[y[t]] += 1; N[id[k]] += 1;
            }
        }
    }
}
