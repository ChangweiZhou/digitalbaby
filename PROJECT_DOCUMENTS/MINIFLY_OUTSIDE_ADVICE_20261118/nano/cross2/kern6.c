/* The instrument hold_x.py at X=inf, compiled: the gated cells of hold.py; the answering cell and its return connections learn at every byte as
   there; the cells' strengths change once at the end of each line, by a full pass backward over that line (back-propagation through the line,
   cut at the line's first byte). With adam=1 the same accumulated change of a line (taken for eta1 = 1) is given to Adam, one step a line.
   A line ends at the step whose ys is negative. Same arithmetic as the numpy program (check_bp.py).  Built by fastk6.py.                       */
#include <math.h>
#include <stdint.h>
#include <string.h>
#define TMAX 64

typedef struct {
    int64_t H, S, K, right, C, NI, T, adam, nadam;     /* T: steps of the present line already stored (0 between lines) */
    double eta1, eta2, lam, alpha, b1, b2a, eps;
    int64_t *idx, *prev, *cols;                         /* cols: TMAX * 16 */
    double *Ac, *Az, *Bc, *Bz, *cc, *cz, *R, *W2, *b2, *h;
    double *zs, *kcs, *kzs, *rs, *pres;                 /* stored for the line: TMAX*H each, pres TMAX*H*K */
    double *c, *m, *d, *dn, *G, *M, *V;                 /* work: H, S, H, H; G: the line's change, NP numbers; M, V: Adam's two means, NP each */
} BNet;

void hx_steps(BNet *n, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t H = n->H, S = n->S, K = n->K, C = n->C, NI = n->NI, nc = 1 + n->C;
    const int64_t oAc = 0, oAz = H * NI, oBc = 2 * H * NI, oBz = oBc + H * K, occ = oBz + H * K, ocz = occ + H, NP = ocz + H;
    const double eta1 = n->eta1, lam = n->lam;
    double *restrict h = n->h, *restrict c = n->c, *restrict m = n->m, *restrict G = n->G;
    for (int64_t i = 0; i < nsteps; i++) {
        const int64_t x = xs[i], y = ys[i], t = n->T;
        int64_t *cols = n->cols + t * 16;
        double *restrict z = n->zs + t * H, *restrict kc = n->kcs + t * H, *restrict kz = n->kzs + t * H, *restrict r = n->rs + t * H, *restrict pre = n->pres + t * H * K;
        cols[0] = x; for (int64_t d = 0; d < C; d++) cols[d + 1] = S * (d + 1) + n->prev[d];
        for (int64_t d = C - 1; d > 0; d--) n->prev[d] = n->prev[d - 1];
        if (C) n->prev[0] = x;
        for (int64_t j = 0; j < H; j++) {
            const int64_t *id = n->idx + j * K; const double *bz = n->Bz + j * K, *bc = n->Bc + j * K; double *p = pre + j * K; double zu = 0, cu = 0;
            for (int64_t s = 0; s < K; s++) { const double v = h[id[s]]; p[s] = v; zu += bz[s] * v; cu += bc[s] * v; }
            for (int64_t q = 0; q < nc; q++) { zu += n->Az[j * NI + cols[q]]; cu += n->Ac[j * NI + cols[q]]; }
            z[j] = -(zu + n->cz[j]); c[j] = -(cu + n->cc[j]);
        }
        for (int64_t j = 0; j < H; j++) z[j] = 1.0 / (1.0 + exp(z[j]));
        for (int64_t j = 0; j < H; j++) c[j] = 1.0 / (1.0 + exp(c[j]));
        for (int64_t j = 0; j < H; j++) { const double hp = h[j]; kc[j] = (1.0 - z[j]) * c[j] * (1.0 - c[j]); kz[j] = (hp - c[j]) * z[j] * (1.0 - z[j]); h[j] = z[j] * hp + (1.0 - z[j]) * c[j]; }
        if (y >= 0) {
            double best = -1e300, hh = 0; int64_t arg = 0;
            for (int64_t v = 0; v < S; v++) { const double *w = n->W2 + v * H; double a = 0; for (int64_t k = 0; k < H; k++) a += w[k] * h[k]; a += n->b2[v]; if (a > best) { best = a; arg = v; } m[v] = -a; }
            m[y] += 1.0; n->right += (arg == y);
            for (int64_t k = 0; k < H; k++) hh += h[k] * h[k];
            const double g = n->eta2 / (hh + 1.0);
            for (int64_t j = 0; j < H; j++) { const double *rr = n->R + j * S; double s = 0; for (int64_t v = 0; v < S; v++) s += rr[v] * m[v]; r[j] = s; }      /* without eta1: it multiplies the line's change at the end */
            for (int64_t v = 0; v < S; v++) {
                const double gm = g * m[v]; double *w = n->W2 + v * H;
                for (int64_t k = 0; k < H; k++) { const double q = gm * h[k]; w[k] = (w[k] + q) * (1.0 - lam); n->R[k * S + v] = (n->R[k * S + v] + q) * (1.0 - lam); }
                n->b2[v] += gm;
            }
            n->T = t + 1; if (n->T < TMAX) continue;      /* a line longer than TMAX is cut there */
        } else { for (int64_t j = 0; j < H; j++) r[j] = 0.0; n->T = t + 1; }
        /* the end of the line: the pass backward */
        const int64_t T = n->T; double *d = n->d, *dn = n->dn;
        for (int64_t k = 0; k < NP; k++) G[k] = 0.0;
        for (int64_t u = T - 1; u >= 0; u--) {
            const double *zu = n->zs + u * H, *kcu = n->kcs + u * H, *kzu = n->kzs + u * H, *ru = n->rs + u * H, *pu = n->pres + u * H * K; const int64_t *cu = n->cols + u * 16;
            if (u == T - 1) for (int64_t j = 0; j < H; j++) d[j] = ru[j];
            else {
                const double *z1 = n->zs + (u + 1) * H, *kc1 = n->kcs + (u + 1) * H, *kz1 = n->kzs + (u + 1) * H;
                for (int64_t j = 0; j < H; j++) dn[j] = ru[j] + z1[j] * d[j];
                for (int64_t k = 0; k < H; k++) {
                    const int64_t *id = n->idx + k * K; const double a = d[k] * kc1[k], b = d[k] * kz1[k]; const double *qc = n->Bc + k * K, *qz = n->Bz + k * K;
                    for (int64_t s = 0; s < K; s++) dn[id[s]] += a * qc[s] + b * qz[s];
                }
                double *sw = d; d = dn; dn = sw;
            }
            for (int64_t j = 0; j < H; j++) {
                const double gc = d[j] * kcu[j], gz = d[j] * kzu[j]; const double *p = pu + j * K; double *gb = G + oBc + j * K, *gy = G + oBz + j * K;
                for (int64_t q = 0; q < nc; q++) { G[oAc + j * NI + cu[q]] += gc; G[oAz + j * NI + cu[q]] += gz; }
                for (int64_t s = 0; s < K; s++) { gb[s] += gc * p[s]; gy[s] += gz * p[s]; }
                G[occ + j] += gc; G[ocz + j] += gz;
            }
        }
        double *P[6] = { n->Ac, n->Az, n->Bc, n->Bz, n->cc, n->cz }; const int64_t off[7] = { oAc, oAz, oBc, oBz, occ, ocz, NP };
        if (!n->adam) { for (int a = 0; a < 6; a++) { double *p = P[a]; const double *g = G + off[a]; const int64_t len = off[a + 1] - off[a]; for (int64_t k = 0; k < len; k++) p[k] += eta1 * g[k]; } }
        else if (eta1 != 0.0) {
            n->nadam += 1; const double b1 = n->b1, b2 = n->b2a, f = n->alpha * sqrt(1.0 - pow(b2, (double)n->nadam)) / (1.0 - pow(b1, (double)n->nadam)), eps = n->eps;
            for (int a = 0; a < 6; a++) { double *p = P[a]; const double *g = G + off[a]; double *M = n->M + off[a], *V = n->V + off[a]; const int64_t len = off[a + 1] - off[a];
                for (int64_t k = 0; k < len; k++) { M[k] = b1 * M[k] + (1.0 - b1) * g[k]; V[k] = b2 * V[k] + (1.0 - b2) * g[k] * g[k]; p[k] += f * M[k] / (sqrt(V[k]) + eps); } }
        }
        n->T = 0;
    }
}
