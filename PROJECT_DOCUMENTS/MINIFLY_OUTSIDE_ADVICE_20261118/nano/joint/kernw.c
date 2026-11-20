/* PLAN_2.md (line C, step C2): the inner loop of joint2.py. A copy of loop_small_case/kern3.c at commit fe397ca (the plain system, the fading of the
   answering cell's strengths and of the return strengths done exactly as one scale factor, the strengths laid out by line), with one addition:
   K[v * H + j] is 1 for a living connection from cell j to branch v and 0 for a withdrawn one, whose forward and return strengths are zero when the
   kernel is called and are not changed.  The withdrawing itself is done in joint2.py between calls.  Built by fastw.py.                           */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct { int64_t H, NL, S, D; double eta1, eta2, lam; double *W1, *b1, *W2, *b2, *R, *K; int64_t *hist; } WNet;

int64_t withdraw_steps(WNet *n, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t H = n->H, NL = n->NL, S = n->S, D = n->D;
    double *restrict W1t = malloc(NL * H * sizeof(double)), *restrict Rt = malloc(S * H * sizeof(double)), *restrict u = malloc(H * sizeof(double));
    double *restrict h = malloc(H * sizeof(double)), *restrict r = malloc(H * sizeof(double)), *restrict dv = malloc(H * sizeof(double)), m[64];
    double *restrict W2 = n->W2, *restrict b1 = n->b1; const double *restrict K = n->K;
    for (int64_t j = 0; j < H; j++) for (int64_t l = 0; l < NL; l++) W1t[l * H + j] = n->W1[j * NL + l];
    for (int64_t j = 0; j < H; j++) for (int64_t v = 0; v < S; v++) Rt[v * H + j] = n->R[j * S + v];
    double sc = 1.0;                                    /* the true strengths of the answering cell and of the return connections are sc times the stored ones */
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
        for (int64_t v = 0; v < S; v++) { const double *restrict w = W2 + v * H; double a = 0; for (int64_t j = 0; j < H; j++) a += w[j] * h[j]; m[v] = -(sc * a + n->b2[v]); }
        m[y] += 1.0;
        for (int64_t j = 0; j < H; j++) r[j] = 0.0;
        for (int64_t v = 0; v < S; v++) { const double mv = m[v]; const double *restrict w = Rt + v * H; for (int64_t j = 0; j < H; j++) r[j] += mv * w[j]; }
        for (int64_t j = 0; j < H; j++) dv[j] = n->eta1 * sc * r[j] * h[j] * (1.0 - h[j]);
        for (int64_t v = 0; v < S; v++) {
            const double f = g * m[v] / sc; double *restrict w = W2 + v * H, *restrict t = Rt + v * H; const double *restrict k = K + v * H;
            for (int64_t j = 0; j < H; j++) { const double q = f * h[j] * k[j]; w[j] += q; t[j] += q; }
            n->b2[v] += g * m[v];
        }
        for (int64_t d = 0; d < D; d++) { double *restrict w = W1t + (d * S + n->hist[d]) * H; for (int64_t j = 0; j < H; j++) w[j] += dv[j]; }
        for (int64_t j = 0; j < H; j++) b1[j] += dv[j];
        sc *= 1.0 - n->lam;
        if (sc < 1e-100) { for (int64_t k2 = 0; k2 < S * H; k2++) { W2[k2] *= sc; Rt[k2] *= sc; } sc = 1.0; }
    }
    for (int64_t k2 = 0; k2 < S * H; k2++) W2[k2] *= sc;
    for (int64_t j = 0; j < H; j++) { for (int64_t v = 0; v < S; v++) n->R[j * S + v] = sc * Rt[v * H + j]; for (int64_t l = 0; l < NL; l++) n->W1[j * NL + l] = W1t[l * H + j]; }
    free(W1t); free(Rt); free(u); free(h); free(r); free(dv);
    return nsteps;
}
