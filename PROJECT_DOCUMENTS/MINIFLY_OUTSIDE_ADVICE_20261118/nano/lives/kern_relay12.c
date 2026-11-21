/* nano/loop, PLAN_1.md: the inner loop of loop.py (class Sys). A copy of loop_small_case/kern3.c (commit fe397ca) with these changes, all stated:
   (1) soft = 1: the branches of the answering cell compete, p = exp(a) / sum of exp(a), and the mismatch is the indicator of the byte that arrived
       less p; soft = 0 is kern3.c's answering cell (mismatch = indicator less a);
   (2) room for 256 byte values (kern3.c had room for 64; the test text has 65);
   (3) the strengths are held by line in Python as well, so nothing is copied on entry and exit;
   (4) the sum a cell receives over its return strengths is formed in the same sweep that changes those strengths, from their values before the
       change (the same arithmetic as kern3.c, one sweep over the return strengths less);
   (5) the loss of the stream as it is read (each byte scored before it arrives, learning on) and the bytes named right are counted.
   The fading of the answering cell's strengths and of the return strengths is one scale factor, as in kern3.c. Built by loop.py.               */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct { int64_t H, NL, S, D, soft, right; double eta1, eta2, lam, loss; double *W1t, *b1, *W2, *b2, *Rt; int64_t *hist; int64_t gate, hold, rect, holdb; } PNet;      /* PLAN_12: four switches */

int64_t p_steps(PNet *n, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t H = n->H, S = n->S, D = n->D, soft = n->soft;
    double *restrict u = malloc(H * sizeof(double)), *restrict h = malloc(H * sizeof(double)), *restrict r = malloc(H * sizeof(double)), *restrict dv = malloc(H * sizeof(double));
    double m[256], a[256]; double *restrict cs = malloc(H * sizeof(double)); for (int64_t j = 0; j < H; j++) cs[j] = 1.0;      /* gate: the true strengths from cell j are cs[j] times the stored ones */
    double *restrict W1t = n->W1t, *restrict Rt = n->Rt, *restrict W2 = n->W2, *restrict b1 = n->b1;
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
        for (int64_t v = 0; v < S; v++) { const double *restrict w = W2 + v * H; double q = 0; for (int64_t j = 0; j < H; j++) q += w[j] * (cs[j] * h[j]); a[v] = sc * q + n->b2[v]; if (a[v] > a[best]) best = v; }
        right += best == y;
        if (soft) {                                     /* the branches compete */
            double z = 0; for (int64_t v = 0; v < S; v++) { m[v] = exp(a[v] - a[best]); z += m[v]; }
            for (int64_t v = 0; v < S; v++) m[v] = -m[v] / z;
            loss -= log(-m[y]); m[y] += 1.0;
        } else {                                        /* least squares; its loss is crude: outputs cut off below at 1e-3 and made to sum to one */
            double z = 0; for (int64_t v = 0; v < S; v++) z += a[v] > 1e-3 ? a[v] : 1e-3;
            loss -= log((a[y] > 1e-3 ? a[y] : 1e-3) / z);
            for (int64_t v = 0; v < S; v++) m[v] = (n->rect && a[v] < 0) ? 0.0 : -a[v];      /* rect: a branch below nought is silent */
            m[y] += 1.0;
        }
        for (int64_t j = 0; j < H; j++) r[j] = 0.0;
        for (int64_t v = 0; v < S; v++) {
            const double mv = m[v], f = g * mv / sc; double *restrict w = W2 + v * H, *restrict t = Rt + v * H;
            for (int64_t j = 0; j < H; j++) { const double q = f * h[j] / cs[j]; r[j] += mv * t[j] * cs[j]; w[j] += q; t[j] += q; }
            if (!n->holdb) n->b2[v] += g * mv;      /* holdb: the branches' constants stay at nought */
        }
        for (int64_t j = 0; j < H; j++) dv[j] = n->eta1 * sc * r[j] * h[j] * (1.0 - h[j]);
        for (int64_t d = 0; d < D; d++) { double *restrict w = W1t + (d * S + n->hist[d]) * H; for (int64_t j = 0; j < H; j++) w[j] += dv[j]; }
        if (!n->hold) for (int64_t j = 0; j < H; j++) b1[j] += dv[j];      /* hold: the thresholds stay where they were born */
        if (n->gate) { for (int64_t j = 0; j < H; j++) { cs[j] *= 1.0 - n->lam * h[j]; if (cs[j] < 1e-100) { for (int64_t v = 0; v < S; v++) { W2[v * H + j] *= cs[j]; Rt[v * H + j] *= cs[j]; } cs[j] = 1.0; } } }
        else sc *= 1.0 - n->lam;
        if (sc < 1e-100) { for (int64_t k = 0; k < S * H; k++) { W2[k] *= sc; Rt[k] *= sc; } sc = 1.0; }
    }
    for (int64_t k = 0; k < S * H; k++) { W2[k] *= sc; Rt[k] *= sc; }
    for (int64_t v = 0; v < S; v++) for (int64_t j = 0; j < H; j++) { W2[v * H + j] *= cs[j]; Rt[v * H + j] *= cs[j]; }
    free(cs);
    n->loss += loss; n->right += right;
    free(u); free(h); free(r); free(dv);
    return nsteps;
}
