/* PLAN_4.md (line C, step C5): the inner loop of joint5.py. A copy of loop_small_case/kern3.c at commit fe397ca (the plain system, the fading done
   exactly as one scale factor, the strengths laid out by line) with the rule of change of the answering cell's strengths (ans) and of the cells in
   between (cel) as options; ans = 1 and cel = 1 are the present rules.  Same arithmetic as the numpy program.  Built by fast5.py.               */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct { int64_t H, NL, S, D, ans, cel; double eta1, eta2, lam, p, q, typ, typw; double *W1, *b1, *W2, *b2, *R; int64_t *hist; } SNet;

int64_t rule_steps(SNet *n, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t H = n->H, NL = n->NL, S = n->S, D = n->D, ans = n->ans, cel = n->cel; const double p = n->p, q = n->q;
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
        if (cel == 3) for (int64_t j = 0; j < H; j++) { const double sg = sc * r[j] * h[j] * (1.0 - h[j]); dv[j] = sg > 0 ? q : (sg < 0 ? -q : 0.0); }
        else for (int64_t j = 0; j < H; j++) dv[j] = n->eta1 * sc * r[j] * h[j] * (1.0 - h[j]);
        for (int64_t v = 0; v < S; v++) {
            const double f = g * m[v] / sc; double *restrict w = W2 + v * H, *restrict t = Rt + v * H;
            if (ans == 1) for (int64_t j = 0; j < H; j++) { const double c = f * h[j]; w[j] += c; t[j] += c; }
            else if (ans == 2) { const double e = p * m[v] / sc; for (int64_t j = 0; j < H; j++) { const double c = e * h[j]; w[j] += c; t[j] += c; } }
            else if (ans == 3) { const double sm = m[v] > 0 ? p / sc : (m[v] < 0 ? -p / sc : 0.0); for (int64_t j = 0; j < H; j++) { const double c = h[j] > 0 ? sm : 0.0; w[j] += c; t[j] += c; } }
            else if (ans == 5) { const double sm = m[v] > 0 ? p / sc : (m[v] < 0 ? -p / sc : 0.0); for (int64_t j = 0; j < H; j++) { const double c = h[j] > 0.5 ? sm : (h[j] < 0.5 ? -sm : 0.0); w[j] += c; t[j] += c; } }
            else { const double e = f / (n->typ + p); for (int64_t j = 0; j < H; j++) { const double c = e * h[j] * (sc * fabs(w[j]) + p); w[j] += c; t[j] += c; } }
            n->b2[v] += g * m[v];
        }
        if (cel == 4) { const double e = 1.0 / (n->typw + q); for (int64_t d = 0; d < D; d++) { double *restrict w = W1t + (d * S + n->hist[d]) * H; for (int64_t j = 0; j < H; j++) w[j] += dv[j] * (fabs(w[j]) + q) * e; } }
        else for (int64_t d = 0; d < D; d++) { double *restrict w = W1t + (d * S + n->hist[d]) * H; for (int64_t j = 0; j < H; j++) w[j] += dv[j]; }
        for (int64_t j = 0; j < H; j++) b1[j] += dv[j];
        sc *= 1.0 - n->lam;
        if (sc < 1e-100) { for (int64_t k = 0; k < S * H; k++) { W2[k] *= sc; Rt[k] *= sc; } sc = 1.0; }
    }
    for (int64_t k = 0; k < S * H; k++) W2[k] *= sc;
    for (int64_t j = 0; j < H; j++) { for (int64_t v = 0; v < S; v++) n->R[j * S + v] = sc * Rt[v * H + j]; for (int64_t l = 0; l < NL; l++) n->W1[j * NL + l] = W1t[l * H + j]; }
    free(W1t); free(Rt); free(u); free(h); free(r); free(dv);
    return nsteps;
}
