
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
   TP[3][KP][NL], TA[2][2][NL], TB[3][NL]. */
void la_steps(int64_t n, int64_t KP, const int64_t *ids, const int32_t *xs,
              float **CP, float **NP, float **UP, float **CA, float **NA_, float **UA, float *CB, float *NB_, float *UB,
              double *TP, double *TA, double *TB, double eta, double etan, double e, int learn, int la, int lag, double *P, int32_t *G2, int32_t *G3) {
    static double s1[KM + 5][S], s2[2][KM + 2][S], s3[3][KM][S]; double a[S], p[S], q2[2][KM + 2], q3[3][KM];
    int l1[KM + 5], l2[2][KM + 2], l3[3][KM]; int64_t ida[2][2], idb[3];
    E = e; for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    const int64_t OA1 = S * S, OB1 = S * S, OB2 = S * S + (int64_t)S * S * S;
    for (int64_t t = 0; t < n; t++) {
        const int64_t *id = ids + t * KP; const int x1 = xs[t - 1], x2 = xs[t - 2]; int g3 = 0, g2 = 0, k3 = 0, k2 = 0; int64_t *ia = ida[t % 2];
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
        }
        /* the answer */
        const int64_t K1 = la ? KP + 5 : KP;
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int64_t k = 0; k < K1; k++) {
            l1[k] = -1; double tk;
            if (k < KP) { if (id[k] < 0 || NP[0][id[k]] < 1) continue; l1[k] = says(CP[0] + id[k] * S, NP[0][id[k]], UP[0][id[k]], s1[k]); tk = TP[k * NL + l1[k]]; }
            else if (k < KP + 2) { const int64_t i = ia[k - KP]; if (NA_[0][i] < 1) continue; l1[k] = says(CA[0] + i * S, NA_[0][i], UA[0][i], s1[k]); tk = TA[(k - KP) * NL + l1[k]]; }
            else { const int64_t i = idb[k - KP - 2]; if (NB_[i] < 1) continue; l1[k] = says(CB + i * S, NB_[i], UB[i], s1[k]); tk = TB[(k - KP - 2) * NL + l1[k]]; }
            for (int v = 0; v < S; v++) a[v] += tk * s1[k][v];
        }
        compete(a, p);
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (!learn) continue;
        const int y = xs[t];
        /* trusts: the answer's now; a guess's when the guessed byte arrives */
        for (int64_t k = 0; k < K1; k++) if (l1[k] >= 0) {
            double g = s1[k][y]; for (int v = 0; v < S; v++) g -= p[v] * s1[k][v];
            if (k < KP) TP[k * NL + l1[k]] += eta * g; else if (k < KP + 2) TA[(k - KP) * NL + l1[k]] += etan * g; else TB[(k - KP - 2) * NL + l1[k]] += etan * g;
        }
        if (la && t >= 1) { const int r = (t - 1) % 2;
            for (int64_t k = 0; k < KP + 2; k++) if (l2[r][k] >= 0) { const double g = s2[r][k][y] - q2[r][k]; if (k < KP) TP[(KP + k) * NL + l2[r][k]] += eta * g; else TA[(2 + k - KP) * NL + l2[r][k]] += etan * g; } }
        if (la && t >= 2) { const int r = (t - 2) % 3;
            for (int64_t k = 0; k < KP; k++) if (l3[r][k] >= 0) TP[(2 * KP + k) * NL + l3[r][k]] += eta * (s3[r][k][y] - q3[r][k]); }
        /* counts */
        for (int64_t k = 0; k < KP; k++) if (id[k] >= 0) count(CP[0], NP[0], UP[0], id[k], y);
        if (la) {
            for (int j = 0; j < 2; j++) count(CA[0], NA_[0], UA[0], ia[j], y);
            for (int j = 0; j < 3; j++) count(CB, NB_, UB, idb[j], y);
            if (t >= 1) { const int64_t *im = id - KP; for (int64_t k = 0; k < KP; k++) if (im[k] >= 0) count(CP[1], NP[1], UP[1], im[k], y);
                          for (int j = 0; j < 2; j++) count(CA[1], NA_[1], UA[1], ida[(t - 1) % 2][j], y); }
            if (t >= 2) { const int64_t *im = id - 2 * KP; for (int64_t k = 0; k < KP; k++) if (im[k] >= 0) count(CP[2], NP[2], UP[2], im[k], y); }
        }
    }
}
