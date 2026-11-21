/* PLAN_1.md (line C, step C1): the inner loop of joint.py. A copy of loop_small_case/kern3.c at commit fe397ca (the plain system, with the fading
   of the answering cell's parts A and of the return strengths done exactly as one scale factor, and the strengths laid out by line), with the
   connection from a cell in between to a branch in three forms:  1: w = A;  2: w = A B;  3: w = sign(A) min(|A|, B);  4: form 2 with the return strength following w (a diagnostic);  5: form 2 with the step of a branch set from the product's own change (a diagnostic).  B is the sending cell's part,
   kept at its true value (it does not fade).  Same arithmetic as the numpy program.  Built by fastj.py.                                          */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct { int64_t H, NL, S, D, form; double eta1, eta2, lam, etaB; double *W1, *b1, *A, *B, *b2, *R; int64_t *hist; } JNet;

int64_t joint_steps(JNet *n, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t H = n->H, NL = n->NL, S = n->S, D = n->D, form = n->form;
    double *restrict W1t = malloc(NL * H * sizeof(double)), *restrict Rt = malloc(S * H * sizeof(double)), *restrict u = malloc(H * sizeof(double));
    double *restrict h = malloc(H * sizeof(double)), *restrict r = malloc(H * sizeof(double)), *restrict dv = malloc(H * sizeof(double)), m[64], nz[64];
    double *restrict A = n->A, *restrict B = n->B, *restrict b1 = n->b1;
    for (int64_t j = 0; j < H; j++) for (int64_t l = 0; l < NL; l++) W1t[l * H + j] = n->W1[j * NL + l];
    for (int64_t j = 0; j < H; j++) for (int64_t v = 0; v < S; v++) Rt[v * H + j] = n->R[j * S + v];
    double sc = 1.0;                                    /* the true parts A and the true return strengths are sc times the stored ones */
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
        for (int64_t v = 0; v < S; v++) {
            const double *restrict a = A + v * H, *restrict b = B + v * H; double z = 0;
            if (form == 1) { for (int64_t j = 0; j < H; j++) z += a[j] * h[j]; z *= sc; }
            else if (form == 2 || form == 4) { for (int64_t j = 0; j < H; j++) z += a[j] * b[j] * h[j]; z *= sc; }
            else if (form == 5) { double q2 = 0; const double e2 = n->etaB * sc * sc; for (int64_t j = 0; j < H; j++) { z += a[j] * b[j] * h[j]; q2 += h[j] * h[j] * (b[j] * b[j] + e2 * a[j] * a[j]); } z *= sc; nz[v] = q2; }
            else for (int64_t j = 0; j < H; j++) { const double at = sc * a[j], aa = fabs(at); z += (aa < b[j] ? at : (at > 0 ? b[j] : (at < 0 ? -b[j] : 0.0))) * h[j]; }
            m[v] = -(z + n->b2[v]);
        }
        m[y] += 1.0;
        for (int64_t j = 0; j < H; j++) r[j] = 0.0;
        for (int64_t v = 0; v < S; v++) { const double mv = m[v]; const double *restrict w = Rt + v * H; for (int64_t j = 0; j < H; j++) r[j] += mv * w[j]; }
        for (int64_t j = 0; j < H; j++) dv[j] = n->eta1 * sc * r[j] * h[j] * (1.0 - h[j]);
        for (int64_t v = 0; v < S; v++) {
            const double gm = g * m[v], f = gm / sc; double *restrict a = A + v * H, *restrict b = B + v * H, *restrict t = Rt + v * H;
            if (form == 1) for (int64_t j = 0; j < H; j++) { const double q = f * h[j]; a[j] += q; t[j] += q; }
            else if (form == 2) { const double e = n->etaB * gm * sc;
                for (int64_t j = 0; j < H; j++) { const double q = f * h[j], a0 = a[j], b0 = b[j], nb = b0 + e * h[j] * a0; a[j] = a0 + q * b0; b[j] = nb > 0 ? nb : 0.0; t[j] += q; } }
            else if (form == 5) { const double g5 = n->eta2 / (nz[v] + 1.0) * m[v], f5 = g5 / sc, e = n->etaB * g5 * sc;      /* form 2, the step set from the product's own change */
                for (int64_t j = 0; j < H; j++) { const double a0 = a[j], b0 = b[j], nb = b0 + e * h[j] * a0; a[j] = a0 + f5 * h[j] * b0; b[j] = nb > 0 ? nb : 0.0; t[j] += f * h[j]; } }
            else if (form == 4) { const double e = n->etaB * gm * sc, e2 = n->etaB * sc * sc;      /* form 2, the return strength following w */
                for (int64_t j = 0; j < H; j++) { const double q = f * h[j], a0 = a[j], b0 = b[j], nb = b0 + e * h[j] * a0; a[j] = a0 + q * b0; b[j] = nb > 0 ? nb : 0.0; t[j] += q * (b0 * b0 + e2 * a0 * a0); } }
            else for (int64_t j = 0; j < H; j++) {
                const double q = f * h[j], at = sc * a[j];
                if (fabs(at) < b[j]) a[j] += q;
                else { const double nb = b[j] + (at > 0 ? gm * h[j] : (at < 0 ? -gm * h[j] : 0.0)); b[j] = nb > 0 ? nb : 0.0; }
                t[j] += q; }
            n->b2[v] += gm;
        }
        for (int64_t d = 0; d < D; d++) { double *restrict w = W1t + (d * S + n->hist[d]) * H; for (int64_t j = 0; j < H; j++) w[j] += dv[j]; }
        for (int64_t j = 0; j < H; j++) b1[j] += dv[j];
        sc *= 1.0 - n->lam;
        if (sc < 1e-100) { for (int64_t k = 0; k < S * H; k++) { A[k] *= sc; Rt[k] *= sc; } sc = 1.0; }
    }
    for (int64_t k = 0; k < S * H; k++) A[k] *= sc;
    for (int64_t j = 0; j < H; j++) { for (int64_t v = 0; v < S; v++) n->R[j * S + v] = sc * Rt[v * H + j]; for (int64_t l = 0; l < NL; l++) n->W1[j * NL + l] = W1t[l * H + j]; }
    free(W1t); free(Rt); free(u); free(h); free(r); free(dv);
    return nsteps;
}
