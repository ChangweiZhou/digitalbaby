/* PLAN_8.md: the inner loop of cells2.py for cells that are silent below a threshold (clip=p), working only on the cells that are active.
   Same arithmetic as the numpy program (Sys.learn with clip), with two things done lazily and exactly: the fading of the answering cell's strengths
   and of the return strengths is one scale factor, and the drift of every threshold toward the target is one common offset.
   Only the plain system: every line read, one step of relay, no sign constraint, no rewiring.  Built by fastk2.py.                              */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int64_t H, NL, S, D, learn_steps, active_sum, changing_sum;
    double eta1, eta2, lam, clip, kap, cap;
    double *W1, *b1, *W2, *b2, *R;
    int64_t *hist;
} SNet;

int64_t sparse_steps(SNet *n, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t H = n->H, NL = n->NL, S = n->S, D = n->D;
    const double eta1 = n->eta1, cap = n->cap;
    double *restrict W1t = malloc(NL * H * sizeof(double)), *restrict W2c = malloc(H * S * sizeof(double)), *restrict u = malloc(H * sizeof(double));
    double *restrict hv = malloc(H * sizeof(double)), a[64], m[64];
    int32_t *restrict ia = malloc(H * sizeof(int32_t));
    for (int64_t j = 0; j < H; j++) for (int64_t l = 0; l < NL; l++) W1t[l * H + j] = n->W1[j * NL + l];      /* by line, so that a line's strengths lie together */
    for (int64_t v = 0; v < S; v++) for (int64_t j = 0; j < H; j++) W2c[j * S + v] = n->W2[v * H + j];        /* by cell */
    double sc = 1.0, G = 0.0;                           /* the true strengths are sc times the stored ones; every threshold has drifted by G */
    for (int64_t i = 0; i < nsteps; i++) {
        for (int64_t d = D - 1; d > 0; d--) n->hist[d] = n->hist[d - 1];
        n->hist[0] = xs[i];
        const int64_t y = ys[i];
        if (y < 0) continue;
        for (int64_t j = 0; j < H; j++) u[j] = n->b1[j] + G;
        for (int64_t d = 0; d < D; d++) { const double *restrict w = W1t + (d * S + n->hist[d]) * H; for (int64_t j = 0; j < H; j++) u[j] += w[j]; }
        int64_t nA = 0;
        for (int64_t j = 0; j < H; j++) if (u[j] > 0.0) { ia[nA] = (int32_t)j; hv[nA] = u[j] < cap ? u[j] : cap; nA++; }
        double hh = 0;
        for (int64_t v = 0; v < S; v++) a[v] = 0.0;
        for (int64_t q = 0; q < nA; q++) { const double h = hv[q]; const double *w = W2c + ia[q] * S; hh += h * h; for (int64_t v = 0; v < S; v++) a[v] += w[v] * h; }
        for (int64_t v = 0; v < S; v++) m[v] = -(sc * a[v] + n->b2[v]);
        m[y] += 1.0;
        const double g = n->eta2 / (hh + 1.0);
        for (int64_t q = 0; q < nA; q++) {              /* only the active cells: the relay, the cell's own change, the answering cell's change */
            const int64_t j = ia[q]; const double h = hv[q]; double *rr = n->R + j * S, *w = W2c + j * S;
            if (u[j] < cap) {
                double r = 0; for (int64_t v = 0; v < S; v++) r += rr[v] * m[v];
                const double dj = eta1 * sc * r;
                for (int64_t d = 0; d < D; d++) W1t[(d * S + n->hist[d]) * H + j] += dj;
                n->b1[j] += dj; n->changing_sum++;
            }
            const double f = g * h / sc;
            for (int64_t v = 0; v < S; v++) { const double p = f * m[v]; w[v] += p; rr[v] += p; }
            n->b1[j] -= n->kap * h;
        }
        for (int64_t v = 0; v < S; v++) n->b2[v] += g * m[v];
        sc *= 1.0 - n->lam; G += n->kap * n->clip; n->learn_steps++; n->active_sum += nA;
        if (sc < 1e-100) {                              /* keep the stored numbers in range */
            for (int64_t k = 0; k < H * S; k++) { W2c[k] *= sc; n->R[k] *= sc; }
            sc = 1.0;
        }
    }
    for (int64_t j = 0; j < H; j++) { for (int64_t l = 0; l < NL; l++) n->W1[j * NL + l] = W1t[l * H + j]; n->b1[j] += G; }
    for (int64_t v = 0; v < S; v++) for (int64_t j = 0; j < H; j++) n->W2[v * H + j] = sc * W2c[j * S + v];
    for (int64_t k = 0; k < H * S; k++) n->R[k] *= sc;
    free(W1t); free(W2c); free(u); free(hv); free(ia);
    return nsteps;
}
