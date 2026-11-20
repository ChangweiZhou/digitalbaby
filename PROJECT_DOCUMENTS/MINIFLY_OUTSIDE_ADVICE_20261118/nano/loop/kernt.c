/* nano/loop, PLAN_2.md: kernp.c for the arm "born at nothing" only. Return strengths born at nothing receive at every byte the same change as the
   forward strengths of the answering cell and fade alike, so the two are the same numbers throughout (check.py: it agrees with kernp.c and with numpy to rounding).
   This kernel therefore keeps one copy and reads the return strengths from it; nothing else differs from kernp.c. Built by loop.py (tied=1).      */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct { int64_t H, NL, S, D, soft, right; double eta1, eta2, lam, loss; double *W1t, *b1, *W2, *b2; int64_t *hist; } TNet;

int64_t t_steps(TNet *n, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t H = n->H, S = n->S, D = n->D, soft = n->soft;
    double *restrict u = malloc(H * sizeof(double)), *restrict h = malloc(H * sizeof(double)), *restrict r = malloc(H * sizeof(double)), *restrict dv = malloc(H * sizeof(double));
    double m[256], a[256];
    double *restrict W1t = n->W1t, *restrict W2 = n->W2, *restrict b1 = n->b1;
    double sc = 1.0, loss = 0.0; int64_t right = 0;     /* the true strengths of the answering cell and of the return connections are sc times the stored ones */
    for (int64_t i = 0; i < nsteps; i++) {
        for (int64_t d = D - 1; d > 0; d--) n->hist[d] = n->hist[d - 1];
        n->hist[0] = xs[i];
        const int64_t y = ys[i];
        if (y < 0) continue;
        for (int64_t j = 0; j < H; j++) u[j] = b1[j];
        for (int64_t d = 0; d < D; d++) { const double *restrict w = W1t + (d * S + n->hist[d]) * H; for (int64_t j = 0; j < H; j++) u[j] += w[j]; }
        for (int64_t j = 0; j < H; j++) h[j] = 1.0 / (1.0 + exp(-u[j]));
        double hh = 0; for (int64_t j = 0; j < H; j++) hh += h[j] * h[j];
        const double g = n->eta2 / (hh + 1.0);
        int64_t best = 0;
        for (int64_t v = 0; v < S; v++) { const double *restrict w = W2 + v * H; double q = 0; for (int64_t j = 0; j < H; j++) q += w[j] * h[j]; a[v] = sc * q + n->b2[v]; if (a[v] > a[best]) best = v; }
        right += best == y;
        if (soft) {                                     /* the branches compete */
            double z = 0; for (int64_t v = 0; v < S; v++) { m[v] = exp(a[v] - a[best]); z += m[v]; }
            for (int64_t v = 0; v < S; v++) m[v] = -m[v] / z;
            loss -= log(-m[y]); m[y] += 1.0;
        } else {                                        /* least squares; its loss is crude: outputs cut off below at 1e-3 and made to sum to one */
            double z = 0; for (int64_t v = 0; v < S; v++) z += a[v] > 1e-3 ? a[v] : 1e-3;
            loss -= log((a[y] > 1e-3 ? a[y] : 1e-3) / z);
            for (int64_t v = 0; v < S; v++) m[v] = -a[v];
            m[y] += 1.0;
        }
        for (int64_t j = 0; j < H; j++) r[j] = 0.0;
        for (int64_t v = 0; v < S; v++) {
            const double mv = m[v], f = g * mv / sc; double *restrict w = W2 + v * H;
            for (int64_t j = 0; j < H; j++) { const double q = f * h[j]; r[j] += mv * w[j]; w[j] += q; }
            n->b2[v] += g * mv;
        }
        for (int64_t j = 0; j < H; j++) dv[j] = n->eta1 * sc * r[j] * h[j] * (1.0 - h[j]);
        for (int64_t d = 0; d < D; d++) { double *restrict w = W1t + (d * S + n->hist[d]) * H; for (int64_t j = 0; j < H; j++) w[j] += dv[j]; }
        for (int64_t j = 0; j < H; j++) b1[j] += dv[j];
        sc *= 1.0 - n->lam;
        if (sc < 1e-100) { for (int64_t k = 0; k < S * H; k++) W2[k] *= sc; sc = 1.0; }
    }
    for (int64_t k = 0; k < S * H; k++) W2[k] *= sc;
    n->loss += loss; n->right += right;
    free(u); free(h); free(r); free(dv);
    return nsteps;
}
