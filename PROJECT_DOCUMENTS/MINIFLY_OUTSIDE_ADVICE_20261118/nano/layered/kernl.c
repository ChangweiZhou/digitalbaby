/* The layered system of layered.py: holding cells A that read only lines, ordinary cells B that read them at the same byte, the mismatch
   relayed over two steps of return strengths. Same arithmetic as the numpy program (check_fastl.py). Built by fastl.py.               */
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    int64_t NA, NB, S, C, NI, og, right;
    double etaA, etaB, nB, eta2, lam;
    int64_t *prev;
    double *Ac, *Az, *Ao, *cc, *cz, *co, *WB, *V, *bB, *RB, *R, *W2, *b2, *h, *Ecx, *Ezx, *Ecb, *Ezb;
    double *z, *o, *u, *b, *m, *sig;                    /* work: NA, NA, NA, NB, S, NB */
} LNet;

void lay_steps(LNet *n, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t NA = n->NA, NB = n->NB, S = n->S, C = n->C, NI = n->NI, nc = 1 + n->C, og = n->og;
    const double etaA = n->etaA, etaB = n->etaB, lam = n->lam;
    int64_t cols[16];
    double *restrict h = n->h, *restrict z = n->z, *restrict o = n->o, *restrict u = n->u, *restrict b = n->b, *restrict m = n->m, *restrict sig = n->sig;
    for (int64_t i = 0; i < nsteps; i++) {
        const int64_t x = xs[i], y = ys[i];
        cols[0] = x; for (int64_t d = 0; d < C; d++) cols[d + 1] = S * (d + 1) + n->prev[d];
        for (int64_t d = C - 1; d > 0; d--) n->prev[d] = n->prev[d - 1];
        if (C) n->prev[0] = x;
        for (int64_t j = 0; j < NA; j++) {              /* cells A: gate, candidate, state; every trace moves on */
            double zu = n->cz[j], cu = n->cc[j];
            for (int64_t q = 0; q < nc; q++) { zu += n->Az[j * NI + cols[q]]; cu += n->Ac[j * NI + cols[q]]; }
            const double zj = 1.0 / (1.0 + exp(-zu)), cj = 1.0 / (1.0 + exp(-cu)), hp = h[j];
            const double kc = (1.0 - zj) * cj * (1.0 - cj), kz = (hp - cj) * zj * (1.0 - zj);
            h[j] = zj * hp + (1.0 - zj) * cj; z[j] = zj;
            double *ec = n->Ecx + j * NI, *ez = n->Ezx + j * NI;
            for (int64_t v = 0; v < NI; v++) ec[v] *= zj;
            for (int64_t v = 0; v < NI; v++) ez[v] *= zj;
            for (int64_t q = 0; q < nc; q++) { ec[cols[q]] += kc; ez[cols[q]] += kz; }
            n->Ecb[j] = zj * n->Ecb[j] + kc; n->Ezb[j] = zj * n->Ezb[j] + kz;
        }
        if (y < 0) continue;
        double uu = 0;
        for (int64_t j = 0; j < NA; j++) {              /* what the cells A give out */
            if (og) { double ou = n->co[j]; for (int64_t q = 0; q < nc; q++) ou += n->Ao[j * NI + cols[q]]; o[j] = 1.0 / (1.0 + exp(-ou)); u[j] = o[j] * h[j]; }
            else u[j] = h[j];
            uu += u[j] * u[j];
        }
        double bb = 0;
        for (int64_t k = 0; k < NB; k++) {              /* cells B */
            const double *w = n->WB + k * NA; double a = n->V[k * S + x] + n->bB[k];
            for (int64_t j = 0; j < NA; j++) a += w[j] * u[j];
            b[k] = 1.0 / (1.0 + exp(-a)); bb += b[k] * b[k];
        }
        double best = -1e300; int64_t arg = 0;
        for (int64_t v = 0; v < S; v++) { const double *w = n->W2 + v * NB; double a = n->b2[v]; for (int64_t k = 0; k < NB; k++) a += w[k] * b[k]; if (a > best) { best = a; arg = v; } m[v] = -a; }
        m[y] += 1.0; n->right += (arg == y);
        const double g = n->eta2 / (bb + 1.0);
        for (int64_t k = 0; k < NB; k++) { const double *rr = n->R + k * S; double s = 0; for (int64_t v = 0; v < S; v++) s += rr[v] * m[v]; sig[k] = s * b[k] * (1.0 - b[k]); }
        if (etaA != 0.0) for (int64_t j = 0; j < NA; j++) {   /* cells A change: what reaches each over its return strengths, times each strength's trace */
            const double *rb = n->RB + j * NB; double s = 0; for (int64_t k = 0; k < NB; k++) s += rb[k] * sig[k];
            const double rA = etaA * s, ra = og ? rA * o[j] : rA;
            if (og) { const double q = rA * h[j] * o[j] * (1.0 - o[j]); for (int64_t c = 0; c < nc; c++) n->Ao[j * NI + cols[c]] += q; n->co[j] += q; }
            double *ac = n->Ac + j * NI, *az = n->Az + j * NI; const double *ec = n->Ecx + j * NI, *ez = n->Ezx + j * NI;
            for (int64_t v = 0; v < NI; v++) ac[v] += ra * ec[v];
            for (int64_t v = 0; v < NI; v++) az[v] += ra * ez[v];
            n->cc[j] += ra * n->Ecb[j]; n->cz[j] += ra * n->Ezb[j];
        }
        if (etaB != 0.0) {                              /* cells B change; the cells A's return strengths receive the same product; both fade */
            const double f = etaB * n->nB / (uu + 1.0), keep = 1.0 - lam;
            for (int64_t k = 0; k < NB; k++) { double *w = n->WB + k * NA; const double fs = f * sig[k]; for (int64_t j = 0; j < NA; j++) w[j] = (w[j] + fs * u[j]) * keep; }
            for (int64_t j = 0; j < NA; j++) { double *rb = n->RB + j * NB; const double fu = f * u[j]; for (int64_t k = 0; k < NB; k++) rb[k] = (rb[k] + fu * sig[k]) * keep; }
            for (int64_t k = 0; k < NB; k++) { n->V[k * S + x] += etaB * sig[k]; n->bB[k] += etaB * sig[k]; }
        }
        for (int64_t v = 0; v < S; v++) {               /* the answering cell and its return strengths */
            const double gm = g * m[v]; double *w = n->W2 + v * NB;
            for (int64_t k = 0; k < NB; k++) { const double q = gm * b[k]; w[k] = (w[k] + q) * (1.0 - lam); n->R[k * S + v] = (n->R[k * S + v] + q) * (1.0 - lam); }
            n->b2[v] += gm;
        }
    }
}
