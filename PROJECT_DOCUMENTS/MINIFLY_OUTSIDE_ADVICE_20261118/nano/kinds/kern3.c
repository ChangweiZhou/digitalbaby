/* nano/kinds: a copy of loop_small_case/kern3.c at 260c5d73 with two additions, marked below.
   PLAN_8.md, section 1 (b): the inner loop of cells2.py for the plain system (logistic cells, every line read, one step of relay), with what
   need not be dense done lazily and exactly: the fading of the answering cell's strengths and of the return strengths is one scale factor, and the
   strengths are laid out so that the ten active lines are ten contiguous vectors. Same arithmetic as the numpy program.  Built by fastk3.py.     */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct { int64_t H, NL, S, D; double eta1, eta2, lam; double *W1, *b1, *W2, *b2, *R; int64_t *hist;
                 double *rate; int64_t sig_on, sn; double beta, pw, kappa, sa, sS, sM; } DNet;      /* nano/kinds: the second line (see cells2.py) */

int64_t dense_steps(DNet *n, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t H = n->H, NL = n->NL, S = n->S, D = n->D;
    double *restrict W1t = malloc(NL * H * sizeof(double)), *restrict Rt = malloc(S * H * sizeof(double)), *restrict u = malloc(H * sizeof(double));
    double *restrict h = malloc(H * sizeof(double)), *restrict r = malloc(H * sizeof(double)), *restrict dv = malloc(H * sizeof(double)), m[64];
    double *restrict W2 = n->W2, *restrict b1 = n->b1;
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
        double e1 = n->eta1;
        if (n->sig_on) {                                /* nano/kinds: the signal as it was sent before this byte; then the running mean takes this byte's mismatch */
            const double M = n->sn ? n->kappa * pow(n->sa / (n->sS / (double)n->sn), n->pw) : n->kappa; double sz = 0;
            for (int64_t v = 0; v < S; v++) sz += m[v] * m[v];
            sz = sqrt(sz); e1 = n->eta1 * M; n->sa = n->sn ? n->sa + n->beta * (sz - n->sa) : sz; n->sS += sz; n->sn++; n->sM += M;
        }
        for (int64_t j = 0; j < H; j++) dv[j] = e1 * sc * r[j] * h[j] * (1.0 - h[j]);
        if (n->rate) for (int64_t j = 0; j < H; j++) dv[j] *= n->rate[j];      /* nano/kinds: each cell's own clock */
        for (int64_t v = 0; v < S; v++) {
            const double f = g * m[v] / sc; double *restrict w = W2 + v * H, *restrict t = Rt + v * H;
            for (int64_t j = 0; j < H; j++) { const double q = f * h[j]; w[j] += q; t[j] += q; }
            n->b2[v] += g * m[v];
        }
        for (int64_t d = 0; d < D; d++) { double *restrict w = W1t + (d * S + n->hist[d]) * H; for (int64_t j = 0; j < H; j++) w[j] += dv[j]; }
        for (int64_t j = 0; j < H; j++) b1[j] += dv[j];
        sc *= 1.0 - n->lam;
        if (sc < 1e-100) { for (int64_t k = 0; k < S * H; k++) { W2[k] *= sc; Rt[k] *= sc; } sc = 1.0; }
    }
    for (int64_t k = 0; k < S * H; k++) W2[k] *= sc;
    for (int64_t j = 0; j < H; j++) { for (int64_t v = 0; v < S; v++) n->R[j * S + v] = sc * Rt[v * H + j]; for (int64_t l = 0; l < NL; l++) n->W1[j * NL + l] = W1t[l * H + j]; }
    free(W1t); free(Rt); free(u); free(h); free(r); free(dv);
    return nsteps;
}
