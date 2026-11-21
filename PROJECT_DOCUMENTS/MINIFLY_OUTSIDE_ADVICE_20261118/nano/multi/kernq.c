/* nano/multi, step 3: the inner loop of prod.py (Sys3.step), compiled. The arithmetic is that of the numpy program, which remains the reference;
   sums are taken in another order, so the two agree to rounding (check_prod.py). After loop_small_case/kern.c (commit 260c5d7), its part for two
   steps of relay; a cell B here holds KE edges and KP pair-connections.  Built by fastq.py.                                                   */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    int64_t H, H2, NL, S, D, KE, KP, joint, rewire, nbytes, due, right;
    double eta1, eta2, lam, rho;
    double *W1, *b1, *W2, *b2, *R, *bB, *EW, *ER, *PW, *PR, *QA, *QB;
    int64_t *EA, *PA, *PB, *hist;
} QNet;

int64_t prod_steps(QNet *n, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t H = n->H, H2 = n->H2, NL = n->NL, S = n->S, D = n->D, KE = n->KE, KP = n->KP;
    const double eta1 = n->eta1, lam = n->lam, rho = n->rho;
    double *hA = malloc(H * sizeof(double)), *hB = malloc(H2 * sizeof(double)), *sig = malloc(H2 * sizeof(double)), *raw = malloc(H * sizeof(double)), *nrm = malloc(H2 * sizeof(double)), m[64];
    int64_t act[64], done = nsteps;
    for (int64_t i = 0; i < nsteps; i++) {
        for (int64_t d = D - 1; d > 0; d--) n->hist[d] = n->hist[d - 1];
        n->hist[0] = xs[i];
        const int64_t y = ys[i];
        if (y < 0) continue;
        for (int64_t d = 0; d < D; d++) act[d] = d * S + n->hist[d];
        for (int64_t j = 0; j < H; j++) { const double *w = n->W1 + j * NL; double u = 0; for (int64_t d = 0; d < D; d++) u += w[act[d]]; hA[j] = 1.0 / (1.0 + exp(-(u + n->b1[j]))); }
        for (int64_t k = 0; k < H2; k++) {
            double u = 0, q = 0;
            const int64_t *ea = n->EA + k * KE; const double *ew = n->EW + k * KE;
            for (int64_t e = 0; e < KE; e++) { const double a = hA[ea[e]]; u += ew[e] * a; q += a * a; }
            const int64_t *pa = n->PA + k * KP, *pb = n->PB + k * KP; const double *pw = n->PW + k * KP;
            for (int64_t e = 0; e < KP; e++) { const double a = hA[pa[e]] * hA[pb[e]]; u += pw[e] * a; q += a * a; }
            hB[k] = 1.0 / (1.0 + exp(-(u + n->bB[k]))); nrm[k] = q + 1.0;
        }
        double hh = 0; for (int64_t k = 0; k < H2; k++) hh += hB[k] * hB[k];
        const double g = n->eta2 / (hh + 1.0);
        double best = -1e300; int64_t arg = 0;
        for (int64_t v = 0; v < S; v++) { const double *w = n->W2 + v * H2; double a = 0; for (int64_t k = 0; k < H2; k++) a += w[k] * hB[k]; a += n->b2[v]; if (a > best) { best = a; arg = v; } m[v] = -a; }
        m[y] += 1.0; n->right += (arg == y);
        for (int64_t k = 0; k < H2; k++) { const double *r = n->R + k * S; double s = 0; for (int64_t v = 0; v < S; v++) s += r[v] * m[v]; sig[k] = s * hB[k] * (1.0 - hB[k]); }
        for (int64_t j = 0; j < H; j++) raw[j] = 0.0;
        for (int64_t k = 0; k < H2; k++) {              /* a cell B passes on what it received over each connection's return strength, then changes */
            const double sk = sig[k], c = eta1 * (double)D / nrm[k] * sk;
            const int64_t *ea = n->EA + k * KE; double *ew = n->EW + k * KE, *er = n->ER + k * KE;
            for (int64_t e = 0; e < KE; e++) { const double a = hA[ea[e]]; raw[ea[e]] += er[e] * sk; const double q = c * a; ew[e] = (ew[e] + q) * (1.0 - lam); er[e] = (er[e] + q) * (1.0 - lam); }
            const int64_t *pa = n->PA + k * KP, *pb = n->PB + k * KP; double *pw = n->PW + k * KP, *pr = n->PR + k * KP;
            for (int64_t e = 0; e < KP; e++) {
                const double a1 = hA[pa[e]], a2 = hA[pb[e]], cA = pr[e] * sk * a2, cB = pr[e] * sk * a1;      /* to each sending cell: times the other's output */
                raw[pa[e]] += cA; raw[pb[e]] += cB;
                if (n->joint) { double *qa = n->QA + k * KP + e, *qb = n->QB + k * KP + e; *qa += rho * (fabs(cA) - *qa); *qb += rho * (fabs(cB) - *qb); }
                const double q = c * a1 * a2; pw[e] = (pw[e] + q) * (1.0 - lam); pr[e] = (pr[e] + q) * (1.0 - lam);
            }
            n->bB[k] += eta1 * sk;
        }
        for (int64_t v = 0; v < S; v++) {
            const double gm = g * m[v]; double *w = n->W2 + v * H2;
            for (int64_t k = 0; k < H2; k++) { const double q = gm * hB[k]; w[k] = (w[k] + q) * (1.0 - lam); n->R[k * S + v] = (n->R[k * S + v] + q) * (1.0 - lam); }
            n->b2[v] += gm;
        }
        for (int64_t j = 0; j < H; j++) { const double dj = eta1 * raw[j] * hA[j] * (1.0 - hA[j]); double *w = n->W1 + j * NL; for (int64_t d = 0; d < D; d++) w[act[d]] += dj; n->b1[j] += dj; }
        if (n->rewire) { n->nbytes++; if (n->nbytes % n->rewire == 0) { n->due = 1; done = i + 1; break; } }
    }
    free(hA); free(hB); free(sig); free(raw); free(nrm);
    return done;
}
