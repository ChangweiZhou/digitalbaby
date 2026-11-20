/* The gated cells of hold.py with a short chain of delay lines before the present byte (chain=C): kern.c's hold_steps with C lines more.
   Same arithmetic as the numpy program (check_fast.py).  Built by fastk4.py.                                                              */
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    int64_t H, S, K, cross, retmode, right, C, NI;      /* C: delay lines before the present byte; NI = S (1 + C). retmode: 0 exact, 1 sign, 2 learned or grown (kept at the reader), 3 local, 4 local2 (kept at the cell read) */
    double eta1, eta2, lam, lamB, qs;
    int64_t *idx, *prev;
    double *Ac, *Az, *Bc, *Bz, *cc, *cz, *R, *W2, *b2, *h, *Ecx, *Ezx, *Ecr, *Ezr, *Ecb, *Ezb, *Qc, *Qz;
    double *hp, *pre, *z, *c, *kc, *kz, *r, *rho, *m;   /* work: H, H*K, H, H, H, H, H, H, S */
} HNet4;

void hold4_steps(HNet4 *n, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t H = n->H, S = n->S, K = n->K, rm = n->retmode, C = n->C, NI = n->NI, nc = 1 + n->C;
    int64_t cols[16];
    const double eta1 = n->eta1, lam = n->lam, lamB = n->lamB, qs = n->qs;
    double *restrict h = n->h, *restrict hp = n->hp, *restrict pre = n->pre, *restrict z = n->z, *restrict c = n->c, *restrict kc = n->kc, *restrict kz = n->kz;
    double *restrict r = n->r, *restrict rho = n->rho, *restrict m = n->m;
    for (int64_t i = 0; i < nsteps; i++) {
        const int64_t x = xs[i], y = ys[i];
        cols[0] = x; for (int64_t d = 0; d < C; d++) cols[d + 1] = S * (d + 1) + n->prev[d];
        for (int64_t d = C - 1; d > 0; d--) n->prev[d] = n->prev[d - 1];
        if (C) n->prev[0] = x;
        memcpy(hp, h, H * sizeof(double));
        for (int64_t j = 0; j < H; j++) {
            const int64_t *id = n->idx + j * K; const double *bz = n->Bz + j * K, *bc = n->Bc + j * K; double *p = pre + j * K; double zu = 0, cu = 0;
            for (int64_t s = 0; s < K; s++) { const double v = hp[id[s]]; p[s] = v; zu += bz[s] * v; cu += bc[s] * v; }
            for (int64_t q = 0; q < nc; q++) { zu += n->Az[j * NI + cols[q]]; cu += n->Ac[j * NI + cols[q]]; }
            z[j] = -(zu + n->cz[j]); c[j] = -(cu + n->cc[j]);
        }
        for (int64_t j = 0; j < H; j++) z[j] = 1.0 / (1.0 + exp(z[j]));
        for (int64_t j = 0; j < H; j++) c[j] = 1.0 / (1.0 + exp(c[j]));
        for (int64_t j = 0; j < H; j++) { h[j] = z[j] * hp[j] + (1.0 - z[j]) * c[j]; kc[j] = (1.0 - z[j]) * c[j] * (1.0 - c[j]); kz[j] = (hp[j] - c[j]) * z[j] * (1.0 - z[j]); }
        double g = 0; int learn = 0;
        if (y >= 0) {
            double best = -1e300, hh = 0; int64_t arg = 0;
            for (int64_t v = 0; v < S; v++) { const double *w = n->W2 + v * H; double a = 0; for (int64_t k = 0; k < H; k++) a += w[k] * h[k]; a += n->b2[v]; if (a > best) { best = a; arg = v; } m[v] = -a; }
            m[y] += 1.0; n->right += (arg == y);
            for (int64_t k = 0; k < H; k++) hh += h[k] * h[k];
            g = n->eta2 / (hh + 1.0); learn = (eta1 != 0.0);
        }
        if (learn) for (int64_t j = 0; j < H; j++) { const double *rr = n->R + j * S; double s = 0; for (int64_t v = 0; v < S; v++) s += rr[v] * m[v]; r[j] = eta1 * s; }
        const int cross = learn && n->cross;
        if (cross) {                                    /* every reader sends back what it received times its sensitivity to each connection */
            for (int64_t j = 0; j < H; j++) rho[j] = 0.0;
            for (int64_t k = 0; k < H; k++) {
                const int64_t *id = n->idx + k * K; const double a = r[k] * kc[k], b = r[k] * kz[k];
                if (rm == 0) { const double *qc = n->Bc + k * K, *qz = n->Bz + k * K; for (int64_t s = 0; s < K; s++) rho[id[s]] += a * qc[s] + b * qz[s]; }
                else if (rm == 1) { const double *qc = n->Bc + k * K, *qz = n->Bz + k * K;
                    for (int64_t s = 0; s < K; s++) rho[id[s]] += qs * (a * ((qc[s] > 0.0) - (qc[s] < 0.0)) + b * ((qz[s] > 0.0) - (qz[s] < 0.0))); }
                else { const double *qc = n->Qc + k * K, *qz = n->Qz + k * K; for (int64_t s = 0; s < K; s++) rho[id[s]] += a * qc[s] + b * qz[s]; }
            }
        }
        for (int64_t j = 0; j < H; j++) {               /* every trace moves on; a strength changes by what reached its cell times its trace */
            const double zj = z[j], kcj = kc[j], kzj = kz[j], rj = learn ? r[j] : 0.0, oj = cross ? rho[j] : 0.0;
            double *ec = n->Ecx + j * NI, *ez = n->Ezx + j * NI, *ac = n->Ac + j * NI, *az = n->Az + j * NI;
            if (learn) {
                for (int64_t v = 0; v < NI; v++) { const double eo = ec[v], en = zj * eo; ac[v] += rj * en + oj * eo; ec[v] = en; }
                for (int64_t v = 0; v < NI; v++) { const double eo = ez[v], en = zj * eo; az[v] += rj * en + oj * eo; ez[v] = en; }
                for (int64_t q = 0; q < nc; q++) { ec[cols[q]] += kcj; ac[cols[q]] += rj * kcj; ez[cols[q]] += kzj; az[cols[q]] += rj * kzj; }
            } else {
                for (int64_t v = 0; v < NI; v++) ec[v] *= zj;
                for (int64_t v = 0; v < NI; v++) ez[v] *= zj;
                for (int64_t q = 0; q < nc; q++) { ec[cols[q]] += kcj; ez[cols[q]] += kzj; }
            }
            double *fc = n->Ecr + j * K, *fz = n->Ezr + j * K, *bc = n->Bc + j * K, *bz = n->Bz + j * K; const double *p = pre + j * K;
            if (learn) {
                if (rm >= 2 && n->cross) {
                    double *qc = n->Qc + j * K, *qz = n->Qz + j * K;
                    for (int64_t s = 0; s < K; s++) {
                        const double eo = fc[s], en = zj * eo + kcj * p[s], dl = rj * en + oj * eo; fc[s] = en; bc[s] += dl;
                        qc[s] = (qc[s] + (rm == 3 ? rj * kcj * p[s] : rm == 4 ? rj * en : dl)) * (1.0 - lamB);
                        const double eo2 = fz[s], en2 = zj * eo2 + kzj * p[s], dl2 = rj * en2 + oj * eo2; fz[s] = en2; bz[s] += dl2;
                        qz[s] = (qz[s] + (rm == 3 ? rj * kzj * p[s] : rm == 4 ? rj * en2 : dl2)) * (1.0 - lamB);
                    }
                } else for (int64_t s = 0; s < K; s++) {
                    const double eo = fc[s], en = zj * eo + kcj * p[s]; fc[s] = en; bc[s] += rj * en + oj * eo;
                    const double eo2 = fz[s], en2 = zj * eo2 + kzj * p[s]; fz[s] = en2; bz[s] += rj * en2 + oj * eo2;
                }
                if (lamB != 0.0) for (int64_t s = 0; s < K; s++) { bc[s] *= 1.0 - lamB; bz[s] *= 1.0 - lamB; }
                { const double eo = n->Ecb[j], en = zj * eo + kcj; n->cc[j] += rj * en + oj * eo; n->Ecb[j] = en; }
                { const double eo = n->Ezb[j], en = zj * eo + kzj; n->cz[j] += rj * en + oj * eo; n->Ezb[j] = en; }
            } else {
                for (int64_t s = 0; s < K; s++) { fc[s] = zj * fc[s] + kcj * p[s]; fz[s] = zj * fz[s] + kzj * p[s]; }
                n->Ecb[j] = zj * n->Ecb[j] + kcj; n->Ezb[j] = zj * n->Ezb[j] + kzj;
            }
        }
        if (y >= 0) for (int64_t v = 0; v < S; v++) {   /* the answering cell and its return connections */
            const double gm = g * m[v]; double *w = n->W2 + v * H;
            for (int64_t k = 0; k < H; k++) { const double q = gm * h[k]; w[k] = (w[k] + q) * (1.0 - lam); n->R[k * S + v] = (n->R[k * S + v] + q) * (1.0 - lam); }
            n->b2[v] += gm;
        }
    }
}
