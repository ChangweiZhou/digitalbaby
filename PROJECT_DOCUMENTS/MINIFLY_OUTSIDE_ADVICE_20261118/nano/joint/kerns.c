/* PLAN_5.md (line C, step C4): the inner loop of joint4.py. A copy of kernw.c (itself loop_small_case/kern3.c at commit fe397ca with the living
   connections as a mask K) with the one sign for each cell of cells2.py's dale=1: after every byte a strength from an active line below zero is set
   to zero, and a strength of the answering cell, or a return strength, whose sign is not its cell's is set to zero.  Built by fasts.py.          */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct { int64_t H, NL, S, D, dale; double eta1, eta2, lam; double *W1, *b1, *W2, *b2, *R, *K, *sign; int64_t *hist; } GNet;

int64_t sign_steps(GNet *n, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t H = n->H, NL = n->NL, S = n->S, D = n->D;
    double *restrict W1t = malloc(NL * H * sizeof(double)), *restrict Rt = malloc(S * H * sizeof(double)), *restrict u = malloc(H * sizeof(double));
    double *restrict h = malloc(H * sizeof(double)), *restrict r = malloc(H * sizeof(double)), *restrict dv = malloc(H * sizeof(double)), m[64];
    double *restrict W2 = n->W2, *restrict b1 = n->b1; const double *restrict K = n->K, *restrict sgn = n->sign; const int64_t dale = n->dale;
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
            if (dale) for (int64_t j = 0; j < H; j++) { if (!(w[j] * sgn[j] > 0)) w[j] = 0.0; if (!(t[j] * sgn[j] > 0)) t[j] = 0.0; }
            n->b2[v] += g * m[v];
        }
        for (int64_t d = 0; d < D; d++) { double *restrict w = W1t + (d * S + n->hist[d]) * H; for (int64_t j = 0; j < H; j++) w[j] += dv[j]; }
        if (dale) for (int64_t d = 0; d < D; d++) { double *restrict w = W1t + (d * S + n->hist[d]) * H; for (int64_t j = 0; j < H; j++) if (w[j] < 0) w[j] = 0.0; }
        for (int64_t j = 0; j < H; j++) b1[j] += dv[j];
        sc *= 1.0 - n->lam;
        if (sc < 1e-100) { for (int64_t k2 = 0; k2 < S * H; k2++) { W2[k2] *= sc; Rt[k2] *= sc; } sc = 1.0; }
    }
    for (int64_t k2 = 0; k2 < S * H; k2++) W2[k2] *= sc;
    for (int64_t j = 0; j < H; j++) { for (int64_t v = 0; v < S; v++) n->R[j * S + v] = sc * Rt[v * H + j]; for (int64_t l = 0; l < NL; l++) n->W1[j * NL + l] = W1t[l * H + j]; }
    free(W1t); free(Rt); free(u); free(h); free(r); free(dv);
    return nsteps;
}
