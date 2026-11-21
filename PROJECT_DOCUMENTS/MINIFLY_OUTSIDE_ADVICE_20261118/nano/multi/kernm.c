/* nano/multi: the inner loop of multi.py (Sys.step), compiled. The arithmetic is that of the numpy program, which remains the reference; sums are
   taken in another order, so the two agree to rounding (check_multi.py). After loop_small_case/kern.c and kern3.c (commit 260c5d7): the fading of
   the answering cell's strengths and of the return strengths is one scale factor, the strengths from lines are laid out by line for the length
   of a call, and the pair-connections are found through a list for each line.  Built by fastm.py.                                               */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int64_t H, NL, S, D, P, has_M, comp, rewire, nbytes, due, right;
    double eta1, eta2, lam, nats;
    double *W1, *b1, *W2, *b2, *R, *PW;
    uint8_t *M;
    int64_t *PA, *PB, *hist;
} MNet;

/* for each line, the pair-connections whose nearer line it is: start[NL + 1], ent[H * P] (cell * P + slot) */
static void pair_index(const MNet *n, int64_t *start, int64_t *ent) {
    const int64_t NL = n->NL, T = n->H * n->P;
    memset(start, 0, (NL + 1) * sizeof(int64_t));
    for (int64_t e = 0; e < T; e++) start[n->PA[e] + 1]++;
    for (int64_t l = 0; l < NL; l++) start[l + 1] += start[l];
    int64_t *fill = malloc(NL * sizeof(int64_t)); memcpy(fill, start, NL * sizeof(int64_t));
    for (int64_t e = 0; e < T; e++) ent[fill[n->PA[e]]++] = e;
    free(fill);
}

/* Each step: the byte xs[i] is read (the delay chain moves on); if ys[i] >= 0 the system names the next byte, ys[i] arrives, and the cells change.
   Returns the number of steps done; it stops early when a turnover of connections is due (multi.py does the turnover). */
int64_t multi_steps(MNet *n, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t H = n->H, NL = n->NL, S = n->S, D = n->D, P = n->P;
    double *restrict W1t = malloc(NL * H * sizeof(double)), *restrict Mt = malloc((n->has_M ? NL * H : 1) * sizeof(double)), *restrict Rt = malloc(S * H * sizeof(double));
    double *restrict u = malloc(H * sizeof(double)), *restrict h = malloc(H * sizeof(double)), *restrict r = malloc(H * sizeof(double)), *restrict dv = malloc(H * sizeof(double));
    double *restrict W2 = n->W2, *restrict b1 = n->b1, *m = malloc(S * sizeof(double));
    int64_t *start = malloc((NL + 1) * sizeof(int64_t)), *ent = malloc((H * P + 1) * sizeof(int64_t)), *actp = malloc((H * P + 1) * sizeof(int64_t)), act[64];
    uint8_t *on = calloc(NL, 1);
    for (int64_t j = 0; j < H; j++) for (int64_t l = 0; l < NL; l++) W1t[l * H + j] = n->W1[j * NL + l];
    if (n->has_M) for (int64_t j = 0; j < H; j++) for (int64_t l = 0; l < NL; l++) Mt[l * H + j] = n->M[j * NL + l] ? 1.0 : 0.0;
    for (int64_t j = 0; j < H; j++) for (int64_t v = 0; v < S; v++) Rt[v * H + j] = n->R[j * S + v];
    if (P) pair_index(n, start, ent);
    double sc = 1.0; int64_t done = nsteps;             /* the true strengths of the answering cell and of the return connections are sc times the stored ones */
    for (int64_t i = 0; i < nsteps; i++) {
        for (int64_t d = D - 1; d > 0; d--) n->hist[d] = n->hist[d - 1];
        n->hist[0] = xs[i];
        const int64_t y = ys[i];
        if (y < 0) continue;
        for (int64_t d = 0; d < D; d++) act[d] = d * S + n->hist[d];
        for (int64_t j = 0; j < H; j++) u[j] = b1[j];
        for (int64_t d = 0; d < D; d++) { const double *restrict w = W1t + act[d] * H; for (int64_t j = 0; j < H; j++) u[j] += w[j]; }
        int64_t na = 0;
        if (P) {                                        /* a pair-connection is active when both its lines are */
            for (int64_t d = 0; d < D; d++) on[act[d]] = 1;
            for (int64_t d = 0; d < D; d++) for (int64_t e = start[act[d]]; e < start[act[d] + 1]; e++) { const int64_t q = ent[e]; if (on[n->PB[q]]) { u[q / P] += n->PW[q]; actp[na++] = q; } }
            for (int64_t d = 0; d < D; d++) on[act[d]] = 0;
        }
        for (int64_t j = 0; j < H; j++) h[j] = 1.0 / (1.0 + exp(-u[j]));
        double hh = 0; for (int64_t j = 0; j < H; j++) hh += h[j] * h[j];
        const double g = n->eta2 / (hh + 1.0);
        double best = -1e300; int64_t arg = 0;
        for (int64_t v = 0; v < S; v++) { const double *restrict w = W2 + v * H; double a = 0; for (int64_t j = 0; j < H; j++) a += w[j] * h[j]; a = sc * a + n->b2[v]; m[v] = a; if (a > best) { best = a; arg = v; } }
        n->right += (arg == y);
        if (n->comp) {                                  /* branches that compete: the share given to each branch */
            double z = 0; for (int64_t v = 0; v < S; v++) { m[v] = exp(m[v] - best); z += m[v]; }
            n->nats -= log(m[y] / z);
            for (int64_t v = 0; v < S; v++) m[v] = -m[v] / z;
        } else for (int64_t v = 0; v < S; v++) m[v] = -m[v];
        m[y] += 1.0;
        for (int64_t j = 0; j < H; j++) r[j] = 0.0;
        for (int64_t v = 0; v < S; v++) { const double mv = m[v]; const double *restrict w = Rt + v * H; for (int64_t j = 0; j < H; j++) r[j] += mv * w[j]; }
        for (int64_t j = 0; j < H; j++) dv[j] = n->eta1 * sc * r[j] * h[j] * (1.0 - h[j]);
        for (int64_t v = 0; v < S; v++) {
            const double f = g * m[v] / sc; double *restrict w = W2 + v * H, *restrict t = Rt + v * H;
            for (int64_t j = 0; j < H; j++) { const double q = f * h[j]; w[j] += q; t[j] += q; }
            n->b2[v] += g * m[v];
        }
        if (n->has_M) for (int64_t d = 0; d < D; d++) { double *restrict w = W1t + act[d] * H; const double *restrict k = Mt + act[d] * H; for (int64_t j = 0; j < H; j++) w[j] += dv[j] * k[j]; }
        else for (int64_t d = 0; d < D; d++) { double *restrict w = W1t + act[d] * H; for (int64_t j = 0; j < H; j++) w[j] += dv[j]; }
        for (int64_t j = 0; j < H; j++) b1[j] += dv[j];
        for (int64_t k = 0; k < na; k++) n->PW[actp[k]] += dv[actp[k] / P];
        sc *= 1.0 - n->lam;
        if (sc < 1e-100) { for (int64_t k = 0; k < S * H; k++) { W2[k] *= sc; Rt[k] *= sc; } sc = 1.0; }
        if (n->rewire) { n->nbytes++; if (n->nbytes % n->rewire == 0) { n->due = 1; done = i + 1; break; } }
    }
    for (int64_t k = 0; k < S * H; k++) W2[k] *= sc;
    for (int64_t j = 0; j < H; j++) { for (int64_t v = 0; v < S; v++) n->R[j * S + v] = sc * Rt[v * H + j]; for (int64_t l = 0; l < NL; l++) n->W1[j * NL + l] = W1t[l * H + j]; }
    free(W1t); free(Mt); free(Rt); free(u); free(h); free(r); free(dv); free(m); free(start); free(ent); free(actp); free(on);
    return done;
}

/* The drives of the cells for np positions: hist[np * D] (newest first), out[np * H]. */
void multi_drive(const MNet *n, int64_t np, const int32_t *hist, double *out) {
    const int64_t H = n->H, NL = n->NL, S = n->S, D = n->D, P = n->P;
    int64_t *start = malloc((NL + 1) * sizeof(int64_t)), *ent = malloc((H * P + 1) * sizeof(int64_t)), act[64];
    double *W1t = malloc(NL * H * sizeof(double)); uint8_t *on = calloc(NL, 1);
    for (int64_t j = 0; j < H; j++) for (int64_t l = 0; l < NL; l++) W1t[l * H + j] = n->W1[j * NL + l];
    if (P) pair_index(n, start, ent);
    for (int64_t i = 0; i < np; i++) {
        double *restrict u = out + i * H;
        for (int64_t d = 0; d < D; d++) act[d] = d * S + hist[i * D + d];
        for (int64_t j = 0; j < H; j++) u[j] = n->b1[j];
        for (int64_t d = 0; d < D; d++) { const double *restrict w = W1t + act[d] * H; for (int64_t j = 0; j < H; j++) u[j] += w[j]; }
        if (P) {
            for (int64_t d = 0; d < D; d++) on[act[d]] = 1;
            for (int64_t d = 0; d < D; d++) for (int64_t e = start[act[d]]; e < start[act[d] + 1]; e++) { const int64_t q = ent[e]; if (on[n->PB[q]]) u[q / P] += n->PW[q]; }
            for (int64_t d = 0; d < D; d++) on[act[d]] = 0;
        }
    }
    free(start); free(ent); free(W1t); free(on);
}
