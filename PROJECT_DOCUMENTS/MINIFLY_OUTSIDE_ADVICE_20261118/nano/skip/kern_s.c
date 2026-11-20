/* The inner loop of skip.py (Sk.read), compiled. It is kern3.c of loop_small_case (the plain system of cells2.py, the fading of the answering
   cell's strengths and of the return strengths done as one scale factor) with three additions: a clock for the delay chain, a fading of a cell's
   strengths from the lines that are active (lam1), and for each line the sum of the sizes of the strengths that cells hold from it, kept up to date.
   The numpy program is the reference (check_skip.py). Built by skip.py:  gcc -O3 -march=native -ffast-math -shared -fPIC -o kern_s.so kern_s.c -lm */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    int64_t H, NL, S, D, mode, pend;                    /* mode 0: every byte steps the chain; 2: the rule. pend: with flags, whether the present byte will enter the chain */
    double eta1, eta2, lam, lam1, theta;
    double *W1, *b1, *W2, *b2, *R, *L0;                 /* L0: for each byte value, the reference that its sum is compared with (S numbers) */
    int64_t *hist, *nskip;                              /* hist[0]: the present byte; hist[1..D-1]: the chain. nskip: bytes passed over, by byte value */
} KNet;

/* Each step: the byte xs[i] arrives. The byte that was present enters the chain (which moves on by one) or is passed over; then, if ys[i] >= 0, the
   system names the next byte, ys[i] arrives, and the cells change. fl (may be null): fl[i] says whether byte i will enter the chain (given clocks). */
int64_t skip_steps(KNet *n, int64_t nsteps, const int32_t *xs, const int32_t *ys, const uint8_t *fl) {
    const int64_t H = n->H, NL = n->NL, S = n->S, D = n->D;
    double *restrict W1t = malloc(NL * H * sizeof(double)), *restrict Rt = malloc(S * H * sizeof(double)), *restrict u = malloc(H * sizeof(double));
    double *restrict h = malloc(H * sizeof(double)), *restrict r = malloc(H * sizeof(double)), *restrict dv = malloc(H * sizeof(double)), m[64], Lv[64];
    double *restrict W2 = n->W2, *restrict b1 = n->b1;
    const double keep1 = 1.0 - n->lam1;
    for (int64_t j = 0; j < H; j++) for (int64_t l = 0; l < NL; l++) W1t[l * H + j] = n->W1[j * NL + l];
    for (int64_t j = 0; j < H; j++) for (int64_t v = 0; v < S; v++) Rt[v * H + j] = n->R[j * S + v];
    for (int64_t v = 0; v < S; v++) { double a = 0; for (int64_t d = 1; d < D; d++) { const double *w = W1t + (d * S + v) * H; for (int64_t j = 0; j < H; j++) a += fabs(w[j]); } Lv[v] = a; }
    double sc = 1.0;                                    /* the true strengths of the answering cell and of the return connections are sc times the stored ones */
    for (int64_t i = 0; i < nsteps; i++) {
        const int64_t p = n->hist[0];
        const int adm = fl ? (int)n->pend : (n->mode == 2 ? (Lv[p] >= n->theta * n->L0[p]) : 1);
        if (adm) { for (int64_t d = D - 1; d > 1; d--) n->hist[d] = n->hist[d - 1]; n->hist[1] = p; } else n->nskip[p]++;
        n->hist[0] = xs[i];
        if (fl) n->pend = fl[i];
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
            const double f = g * m[v] / sc; double *restrict w = W2 + v * H, *restrict t = Rt + v * H;
            for (int64_t j = 0; j < H; j++) { const double q = f * h[j]; w[j] += q; t[j] += q; }
            n->b2[v] += g * m[v];
        }
        for (int64_t d = 0; d < D; d++) {               /* the active lines: the change, the fading of what is used, and the line's sum kept up to date */
            double *restrict w = W1t + (d * S + n->hist[d]) * H; double was = 0, is = 0;
            for (int64_t j = 0; j < H; j++) { was += fabs(w[j]); w[j] = (w[j] + dv[j]) * keep1; is += fabs(w[j]); }
            if (d) Lv[n->hist[d]] += is - was;
        }
        for (int64_t j = 0; j < H; j++) b1[j] += dv[j];
        sc *= 1.0 - n->lam;
        if (sc < 1e-100) { for (int64_t k = 0; k < S * H; k++) { W2[k] *= sc; Rt[k] *= sc; } sc = 1.0; }
    }
    for (int64_t k = 0; k < S * H; k++) W2[k] *= sc;
    for (int64_t j = 0; j < H; j++) { for (int64_t v = 0; v < S; v++) n->R[j * S + v] = sc * Rt[v * H + j]; for (int64_t l = 0; l < NL; l++) n->W1[j * NL + l] = W1t[l * H + j]; }
    free(W1t); free(Rt); free(u); free(h); free(r); free(dv);
    return nsteps;
}
