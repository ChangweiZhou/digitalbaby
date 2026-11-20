/* nano/kinds: a copy of loop_small_case/kern.c at 260c5d73; hold_steps has two additions (each cell's own rate; the common signal), marked below.
   The inner loops of cells2.py (Sys.learn) and hold.py (Hold.read), compiled. The arithmetic is that of the numpy programs, which remain the
   reference; sums are taken in a different order, so the two agree to rounding and not bit for bit (check_fast.py).
   Built by fastk.py:  gcc -O3 -march=native -ffast-math -shared -fPIC -o kern.so kern.c -lm                                                   */
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    int64_t H, NL, S, D, top, H2, has_M, dale, has_live, late, rewire, nbytes, due;
    double eta1, eta2, lam;
    double *W1, *b1, *W2, *b2, *R, *WB, *bB, *RB, *sign, *live, *freec;
    uint8_t *M;
    int64_t *hist;
    double *hA, *hB, *m, *sig, *sA, *d;                 /* work: H, H2, S, top, H, H */
} Net;

/* Each step: the byte xs[i] is read (the delay chain moves on); if ys[i] >= 0 the system names the next byte, ys[i] arrives, and the cells change.
   Returns the number of steps done; it stops early when a turnover of connections is due (cells2.py does the turnover). */
int64_t loop_steps(Net *n, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t H = n->H, NL = n->NL, S = n->S, D = n->D, top = n->top, H2 = n->H2;
    const double eta1 = n->eta1, lam = n->lam;
    double *restrict hA = n->hA, *restrict hB = n->hB, *restrict m = n->m, *restrict sig = n->sig, *restrict sA = n->sA, *restrict dd = n->d;
    int64_t act[64];
    for (int64_t i = 0; i < nsteps; i++) {
        for (int64_t d = D - 1; d > 0; d--) n->hist[d] = n->hist[d - 1];
        n->hist[0] = xs[i];
        const int64_t y = ys[i];
        if (y < 0) continue;
        for (int64_t d = 0; d < D; d++) act[d] = d * S + n->hist[d];
        for (int64_t j = 0; j < H; j++) { const double *w = n->W1 + j * NL; double u = 0; for (int64_t d = 0; d < D; d++) u += w[act[d]]; hA[j] = -(u + n->b1[j]); }
        for (int64_t j = 0; j < H; j++) hA[j] = 1.0 / (1.0 + exp(hA[j]));
        if (n->has_live) for (int64_t j = 0; j < H; j++) hA[j] *= n->live[j];
        double *restrict h = hA;
        if (H2) {
            for (int64_t k = 0; k < H2; k++) { const double *w = n->WB + k * H; double u = 0; for (int64_t j = 0; j < H; j++) u += w[j] * hA[j]; hB[k] = -(u + n->bB[k]); }
            for (int64_t k = 0; k < H2; k++) hB[k] = 1.0 / (1.0 + exp(hB[k]));
            h = hB;
        }
        double hh = 0; for (int64_t k = 0; k < top; k++) hh += h[k] * h[k];
        const double g = n->eta2 / (hh + 1.0);
        for (int64_t v = 0; v < S; v++) { const double *w = n->W2 + v * top; double a = 0; for (int64_t k = 0; k < top; k++) a += w[k] * h[k]; m[v] = -(a + n->b2[v]); }
        m[y] += 1.0;
        for (int64_t k = 0; k < top; k++) { const double *r = n->R + k * S; double s = 0; for (int64_t v = 0; v < S; v++) s += r[v] * m[v]; sig[k] = s * h[k] * (1.0 - h[k]); }
        if (n->has_live) for (int64_t k = 0; k < top; k++) sig[k] *= n->freec[k];
        if (H2) {                                       /* cells B change, and pass on what they received to cells A over their own return connections */
            double aa = 0; for (int64_t j = 0; j < H; j++) aa += hA[j] * hA[j];
            const double cB = eta1 * (double)D / (aa + 1.0);
            for (int64_t j = 0; j < H; j++) { const double *r = n->RB + j * H2; double s = 0; for (int64_t k = 0; k < H2; k++) s += r[k] * sig[k]; sA[j] = s * hA[j] * (1.0 - hA[j]); }
            for (int64_t k = 0; k < H2; k++) {
                const double f = cB * sig[k]; double *w = n->WB + k * H;
                for (int64_t j = 0; j < H; j++) { const double q = f * hA[j]; w[j] = (w[j] + q) * (1.0 - lam); n->RB[j * H2 + k] = (n->RB[j * H2 + k] + q) * (1.0 - lam); }
                n->bB[k] += eta1 * sig[k];
            }
            for (int64_t j = 0; j < H; j++) dd[j] = eta1 * sA[j];
        } else for (int64_t j = 0; j < H; j++) dd[j] = eta1 * sig[j];
        for (int64_t v = 0; v < S; v++) {               /* the answering cell and the return connections: the same product, the same fading */
            const double gm = g * m[v]; double *w = n->W2 + v * top;
            if (n->has_live) {
                for (int64_t k = 0; k < top; k++) { const double q = gm * h[k] * n->freec[k], f = 1.0 - lam * n->freec[k]; w[k] = (w[k] + q) * f; n->R[k * S + v] = (n->R[k * S + v] + q) * f; }
                if (!n->late) n->b2[v] += gm;
            } else {
                for (int64_t k = 0; k < top; k++) { const double q = gm * h[k]; w[k] = (w[k] + q) * (1.0 - lam); n->R[k * S + v] = (n->R[k * S + v] + q) * (1.0 - lam); }
                n->b2[v] += gm;
            }
        }
        for (int64_t j = 0; j < H; j++) {               /* the cells in between: the lines that are active, and the threshold */
            double *w = n->W1 + j * NL; const double dj = dd[j];
            if (n->has_M) { const uint8_t *mk = n->M + j * NL; for (int64_t d = 0; d < D; d++) if (mk[act[d]]) w[act[d]] += dj; }
            else for (int64_t d = 0; d < D; d++) w[act[d]] += dj;
            n->b1[j] += dj;
            if (n->dale) for (int64_t d = 0; d < D; d++) if (w[act[d]] < 0.0) w[act[d]] = 0.0;
        }
        if (n->dale) for (int64_t v = 0; v < S; v++) for (int64_t k = 0; k < top; k++) {
            if (!(n->W2[v * top + k] * n->sign[k] > 0.0)) n->W2[v * top + k] = 0.0;
            if (!(n->R[k * S + v] * n->sign[k] > 0.0)) n->R[k * S + v] = 0.0;
        }
        if (n->rewire) { n->nbytes++; if (n->nbytes % n->rewire == 0) { n->due = 1; return i + 1; } }
    }
    return nsteps;
}

typedef struct {
    int64_t H, S, K, cross, retmode, right;             /* retmode: 0 exact, 1 sign, 2 learned or grown (kept at the reader), 3 local (kept at the cell read) */
    double eta1, eta2, lam, lamB, qs;
    int64_t *idx;
    double *Ac, *Az, *Bc, *Bz, *cc, *cz, *R, *W2, *b2, *h, *Ecx, *Ezx, *Ecr, *Ezr, *Ecb, *Ezb, *Qc, *Qz;
    double *hp, *pre, *z, *c, *kc, *kz, *r, *rho, *m;   /* work: H, H*K, H, H, H, H, H, H, S */
    double *rate;                                       /* nano/kinds: each cell's own rate over eta1, or null */
    int64_t sig_on, sn;                                 /* nano/kinds: the common signal (see hold.py) */
    double beta, pw, kappa, sa, sS, sM;
} HNet;

void hold_steps(HNet *n, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t H = n->H, S = n->S, K = n->K, rm = n->retmode;
    const double eta1 = n->eta1, lam = n->lam, lamB = n->lamB, qs = n->qs;
    double *restrict h = n->h, *restrict hp = n->hp, *restrict pre = n->pre, *restrict z = n->z, *restrict c = n->c, *restrict kc = n->kc, *restrict kz = n->kz;
    double *restrict r = n->r, *restrict rho = n->rho, *restrict m = n->m;
    for (int64_t i = 0; i < nsteps; i++) {
        const int64_t x = xs[i], y = ys[i];
        memcpy(hp, h, H * sizeof(double));
        for (int64_t j = 0; j < H; j++) {
            const int64_t *id = n->idx + j * K; const double *bz = n->Bz + j * K, *bc = n->Bc + j * K; double *p = pre + j * K; double zu = 0, cu = 0;
            for (int64_t s = 0; s < K; s++) { const double v = hp[id[s]]; p[s] = v; zu += bz[s] * v; cu += bc[s] * v; }
            z[j] = -(n->Az[j * S + x] + zu + n->cz[j]); c[j] = -(n->Ac[j * S + x] + cu + n->cc[j]);
        }
        for (int64_t j = 0; j < H; j++) z[j] = 1.0 / (1.0 + exp(z[j]));
        for (int64_t j = 0; j < H; j++) c[j] = 1.0 / (1.0 + exp(c[j]));
        for (int64_t j = 0; j < H; j++) { h[j] = z[j] * hp[j] + (1.0 - z[j]) * c[j]; kc[j] = (1.0 - z[j]) * c[j] * (1.0 - c[j]); kz[j] = (hp[j] - c[j]) * z[j] * (1.0 - z[j]); }
        double g = 0, e1 = eta1; int learn = 0;
        if (y >= 0) {
            double best = -1e300, hh = 0; int64_t arg = 0;
            for (int64_t v = 0; v < S; v++) { const double *w = n->W2 + v * H; double a = 0; for (int64_t k = 0; k < H; k++) a += w[k] * h[k]; a += n->b2[v]; if (a > best) { best = a; arg = v; } m[v] = -a; }
            m[y] += 1.0; n->right += (arg == y);
            for (int64_t k = 0; k < H; k++) hh += h[k] * h[k];
            g = n->eta2 / (hh + 1.0); learn = (eta1 != 0.0);
            if (n->sig_on) {                            /* the signal as it was sent before this byte; then the running mean takes this byte's mismatch */
                const double M = n->sn ? n->kappa * pow(n->sa / (n->sS / (double)n->sn), n->pw) : n->kappa; double sz = 0;
                for (int64_t v = 0; v < S; v++) sz += m[v] * m[v];
                sz = sqrt(sz); e1 = eta1 * M; n->sa = n->sn ? n->sa + n->beta * (sz - n->sa) : sz; n->sS += sz; n->sn++; n->sM += M;
            }
        }
        if (learn) for (int64_t j = 0; j < H; j++) { const double *rr = n->R + j * S; double s = 0; for (int64_t v = 0; v < S; v++) s += rr[v] * m[v]; r[j] = e1 * s; }
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
            const double zj = z[j], kcj = kc[j], kzj = kz[j], gj = n->rate ? n->rate[j] : 1.0, rj = learn ? gj * r[j] : 0.0, oj = cross ? gj * rho[j] : 0.0;
            double *ec = n->Ecx + j * S, *ez = n->Ezx + j * S, *ac = n->Ac + j * S, *az = n->Az + j * S;
            if (learn) {
                for (int64_t v = 0; v < S; v++) { const double eo = ec[v], en = zj * eo + (v == x ? kcj : 0.0); ac[v] += rj * en + oj * eo; ec[v] = en; }
                for (int64_t v = 0; v < S; v++) { const double eo = ez[v], en = zj * eo + (v == x ? kzj : 0.0); az[v] += rj * en + oj * eo; ez[v] = en; }
            } else {
                for (int64_t v = 0; v < S; v++) ec[v] *= zj;
                for (int64_t v = 0; v < S; v++) ez[v] *= zj;
                ec[x] += kcj; ez[x] += kzj;
            }
            double *fc = n->Ecr + j * K, *fz = n->Ezr + j * K, *bc = n->Bc + j * K, *bz = n->Bz + j * K; const double *p = pre + j * K;
            if (learn) {
                if (rm >= 2 && n->cross) {
                    double *qc = n->Qc + j * K, *qz = n->Qz + j * K;
                    for (int64_t s = 0; s < K; s++) {
                        const double eo = fc[s], en = zj * eo + kcj * p[s], dl = rj * en + oj * eo; fc[s] = en; bc[s] += dl;
                        qc[s] = (qc[s] + (rm == 3 ? rj * kcj * p[s] : dl)) * (1.0 - lamB);
                        const double eo2 = fz[s], en2 = zj * eo2 + kzj * p[s], dl2 = rj * en2 + oj * eo2; fz[s] = en2; bz[s] += dl2;
                        qz[s] = (qz[s] + (rm == 3 ? rj * kzj * p[s] : dl2)) * (1.0 - lamB);
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
