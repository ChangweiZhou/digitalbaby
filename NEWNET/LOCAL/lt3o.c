/* lt3o.c - lt3.c with the same mechanism and faster code (24 Nov 2026). Not bit-identical to lt3: random draws and visiting order
   differ. Changes, each keeping lt3's rule:
   1. grow: lt3 scanned every active source of every allowed population for every learning cell (cost ~ active x active). Here the
      active lists are kept sorted by cell index (= position on the ring), the reach regions of the cell (its own position and its
      patches) become index ranges found by binary search, and candidates are drawn uniformly from the union of those ranges with
      rejection (full fan-out, self, already connected, outside the exact reach, or counted in an earlier region). The result is the
      same rule: `need` sources drawn uniformly without replacement from the eligible set. If the draws keep failing, the ranges
      (in-reach active sources only) are enumerated as in lt3.
   2. local inhibition: per block, the k-th largest value by quickselect, then winners in index order (same winners and tie rule as
      lt3's repeated argmax: the k largest above 1e-6, ties to the lower index); cost ~ n instead of n x k.
   3. outgoing synapses of a source: proximal entries packed from the front, distal from the back, so each spread walks only the kind
      it needs; each synapse slot keeps its position in the list, so removal is O(1); target cell and slot are stored, no division.
   4. tables for the labile decay (TAG) and the output trace (OTR) replace powf/pow calls.
   5. turnover: the r least-used cells by one selection (same set as lt3's r repeated scans), cost ~ n instead of r x n.
   6. the active lists used for growth are filtered once per byte (or per population step) to sources below their fan-out cap, so
      draws rarely hit full sources; a source freed later in the same byte is not offered until the next byte (lt3 would offer it).
   Interface: lt3o_new/run/snapshot/restore/audit as lt3, plus lt3o_stats. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define V 128
#define MAXK 6
#define DTN 65536
enum { P_N, P_FP, P_FD, P_G, P_B, P_K, P_LP, P_LD, P_AD, P_IN, P_RD, P_NP, P_PM, P_DM, P_NI };
enum { Q_TLO, Q_THI, Q_RHO, Q_PF, Q_STH, Q_DG, Q_ND };
enum { G_K, G_NL, G_KIN, G_FOI, G_FO, G_NO, G_FOO, G_NGP, G_NGD, G_SEED, G_TAG, G_GATE, G_HUBM, G_NI };
enum { H_ETA, H_ETAM, H_WNEW, H_ETAO, H_ETAB, H_ETAT, H_RHOO, H_OREP, H_TAUTAG, H_CAP, H_GTH, H_TURN, H_BOOST, H_OTR, H_HUBF, H_ND };

typedef struct {
    int n, FP, FD, FI, G, nseg, B, k, lp, ld, ad, in, rd, np, pm, dm, off;
    float tlo, thi, rho, pf, sth, dg;
    float *pos, *dec, *h, *m, *val, *pc;
    int *ssrc; float *sw, *sl; int *tl, *use, *born, *bk; int segof[256];
    uint8_t *y; int *act, nact;
    float *pacc, *sacc; uint8_t *tp, *ts; int *tpl, ntp, *tsl, nts;
} Pop;

typedef struct {
    int gi[G_NI]; double gd[H_ND]; int K, NG, FOS;
    Pop p[MAXK];
    float *gpos; int *gpop, *fo, *fop, *fod; int *opop, *ocell, *oslot;
    uint8_t *cur, *prv; int *code;
    int *prvl[MAXK + 1], nprv[MAXK + 1];
    int *osrc; float *ow, *opos; int *ocnt; double b[V], o[V], beta, ct[V];
    int have; uint64_t rs; int lastbyte; int T; int *capa, *lastf; float etamul;
    int *cand; void *snap;
    float *dtab, *otab; int notab; float *vtmp; uint64_t *keys;
    long st_grow, st_draw, st_fall, st_fM, st_fnc, st_fneed, st_rej[6];
    int *fcur[MAXK], nfcur[MAXK], *fprv[MAXK + 1], nfprv[MAXK + 1];   /* active lists filtered to sources below their fan-out cap */
} Net;

static inline uint64_t xr(Net *h) { h->rs ^= h->rs << 13; h->rs ^= h->rs >> 7; h->rs ^= h->rs << 17; return h->rs; }
static inline double ur(Net *h) { return (xr(h) >> 11) * (1.0 / 9007199254740992.0); }
static inline float rdist(float a, float b) { float d = fabsf(a - b); return d > 0.5f ? 1.0f - d : d; }

static inline float lab(Net *h, Pop *P, size_t e) {
    if (!h->gi[G_TAG]) return 0;
    int dt = h->T - P->tl[e];
    if (dt > 0) { P->sl[e] *= dt < DTN ? h->dtab[dt] : powf((float)(1.0 - 1.0 / h->gd[H_TAUTAG]), (float)dt); P->tl[e] = h->T; }
    return P->sl[e];
}
static inline float wef(Net *h, Pop *P, size_t e) { return P->sw[e] + lab(h, P, e); }
static inline void pot(Net *h, Pop *P, size_t e, double eta) {
    if (!h->gi[G_TAG]) { P->sw[e] += (float)(eta * (1 - P->sw[e])); return; }
    float l = lab(h, P, e); P->sl[e] = l + (float)(eta * (1 - P->sw[e] - l));
}
static inline void dep(Net *h, Pop *P, size_t e, double etam) { P->sw[e] -= (float)(etam * P->sw[e]); if (h->gi[G_TAG]) { float l = lab(h, P, e); P->sl[e] = l - (float)(etam * l); } }
static int reach(Pop *P, int i, float sp) {
    if (rdist(P->pos[i], sp) <= P->rho) return 1;
    for (int k = 0; k < P->np; k++) if (rdist(P->pc[i * P->np + k], sp) <= P->rho * 0.5f) return 1;
    return 0;
}
static int srcpop_ok(int mask, int gp) { return (mask >> (gp < 0 ? 31 : gp)) & 1; }
static inline int cap(Net *h, int src) { return h->capa[src]; }

static void add_out(Net *h, int src, int pop, int i, int s) {
    Pop *P = h->p + pop; size_t b = (size_t)src * h->FOS; int k = s < P->FP ? h->fop[src]++ : h->FOS - 1 - h->fod[src]++;
    h->fo[src]++; h->opop[b + k] = pop; h->ocell[b + k] = i; h->oslot[b + k] = s; P->bk[(size_t)i * P->FI + s] = k;
}
static void del_out(Net *h, int src, int pop, int e) {
    Pop *P = h->p + pop; size_t b = (size_t)src * h->FOS; int s = e % P->FI, k = P->bk[e], l;
    if (s < P->FP) l = --h->fop[src]; else l = h->FOS - h->fod[src]--;
    h->fo[src]--;
    if (l != k) { int pq = h->opop[b + l], ci = h->ocell[b + l], cs = h->oslot[b + l]; h->opop[b + k] = pq; h->ocell[b + k] = ci; h->oslot[b + k] = cs; h->p[pq].bk[(size_t)ci * h->p[pq].FI + cs] = k; }
}

static int pick_source(Net *h, int q, int i, int mask) {
    Pop *P = h->p + q;
    for (int t = 0; t < 400; t++) {
        int s = (int)(xr(h) % (uint64_t)h->NG), gp = h->gpop[s];
        if (!srcpop_ok(mask, gp) || h->fo[s] >= cap(h, s)) continue;
        if (gp == q && s - P->off == i) continue;
        if (reach(P, i, h->gpos[s])) return s;
    }
    return -1;
}

Net *lt3o_new(const int *gi, const double *gd, const int *pi, const double *pd) {
    Net *h = calloc(1, sizeof(Net)); memcpy(h->gi, gi, sizeof(int) * G_NI); memcpy(h->gd, gd, sizeof(double) * H_ND);
    h->rs = 88172645463325252ULL ^ ((uint64_t)gi[G_SEED] * 2654435761ULL); for (int i = 0; i < 20; i++) xr(h);
    int K = h->K = gi[G_K], NL = gi[G_NL], maxn = 1, maxB = 1;
    h->NG = NL;
    for (int q = 0; q < K; q++) {
        Pop *P = h->p + q; const int *a = pi + q * P_NI; const double *d = pd + q * Q_ND;
        P->n = a[P_N]; P->FP = a[P_FP]; P->FD = a[P_FD]; P->FI = P->FP + P->FD; P->G = a[P_G] > 0 ? a[P_G] : (P->FD > 0 ? P->FD : 1);
        P->nseg = P->FD ? (P->FD + P->G - 1) / P->G : 0; P->B = a[P_B]; P->k = a[P_K]; P->lp = a[P_LP]; P->ld = a[P_LD]; P->ad = a[P_AD];
        P->in = a[P_IN]; P->rd = a[P_RD]; P->np = a[P_NP]; P->pm = a[P_PM]; P->dm = a[P_DM];
        P->tlo = d[Q_TLO]; P->thi = d[Q_THI]; P->rho = d[Q_RHO]; P->pf = d[Q_PF]; P->sth = d[Q_STH]; P->dg = d[Q_DG];
        P->off = h->NG; h->NG += P->n; if (P->n > maxn) maxn = P->n; if (P->B > maxB) maxB = P->B;
        for (int s = 0; s < P->FI && s < 256; s++) P->segof[s] = s < P->FP ? 0 : (s - P->FP) / P->G;
    }
    int hm = gd[H_HUBF] > 0 ? gi[G_HUBM] : 1; h->FOS = gi[G_FOI] > gi[G_FO] * hm ? gi[G_FOI] : gi[G_FO] * hm;
    h->gpos = malloc(sizeof(float) * h->NG); h->gpop = malloc(sizeof(int) * h->NG);
    h->fo = calloc(h->NG, sizeof(int)); h->fop = calloc(h->NG, sizeof(int)); h->fod = calloc(h->NG, sizeof(int));
    h->opop = malloc(sizeof(int) * (size_t)h->NG * h->FOS); h->ocell = malloc(sizeof(int) * (size_t)h->NG * h->FOS); h->oslot = malloc(sizeof(int) * (size_t)h->NG * h->FOS);
    h->capa = malloc(sizeof(int) * h->NG); h->lastf = malloc(sizeof(int) * h->NG); for (int s = 0; s < h->NG; s++) h->lastf[s] = -1000000;
    h->cur = calloc(h->NG, 1); h->prv = calloc(h->NG, 1); h->cand = malloc(sizeof(int) * h->NG);
    h->vtmp = malloc(sizeof(float) * maxB); h->keys = malloc(sizeof(uint64_t) * maxn);
    for (int s = 0; s < NL; s++) { h->gpos[s] = (s + 0.5f) / NL; h->gpop[s] = -1; h->capa[s] = gi[G_FOI]; }
    for (int q = 0; q < K; q++) {
        Pop *P = h->p + q; int n = P->n;
        P->pos = malloc(sizeof(float) * n); P->dec = malloc(sizeof(float) * n); P->h = calloc(n, sizeof(float)); P->m = calloc(n, sizeof(float));
        P->val = malloc(sizeof(float) * n); P->pc = malloc(sizeof(float) * (n * P->np + 1));
        for (int i = 0; i < n; i++) {
            P->pos[i] = (i + 0.5f) / n; h->gpos[P->off + i] = P->pos[i]; h->gpop[P->off + i] = q;
            double tau = exp(log(P->tlo) + ur(h) * (log(P->thi) - log(P->tlo))); P->dec[i] = (float)(1.0 - 1.0 / tau);
            for (int k = 0; k < P->np; k++) { double dd = 2 * P->rho + ur(h) * (0.5 - 2 * P->rho); if (dd < 0) dd = 0; float c = P->pos[i] + (float)(ur(h) < 0.5 ? dd : -dd); c -= floorf(c); P->pc[i * P->np + k] = c; }
        }
        P->ssrc = malloc(sizeof(int) * (size_t)n * P->FI); P->sw = calloc((size_t)n * P->FI, sizeof(float)); P->bk = calloc((size_t)n * P->FI + 1, sizeof(int));
        P->sl = calloc((size_t)n * P->FI, sizeof(float)); P->tl = calloc((size_t)n * P->FI, sizeof(int)); P->use = calloc(n, sizeof(int)); P->born = malloc(sizeof(int) * n); for (int i = 0; i < n; i++) P->born[i] = -1000000;
        for (size_t e = 0; e < (size_t)n * P->FI; e++) P->ssrc[e] = -1;
        P->y = calloc(n, 1); P->act = malloc(sizeof(int) * n);
        P->pacc = calloc(n, sizeof(float)); P->sacc = calloc((size_t)n * (P->nseg + 1), sizeof(float));
        P->tp = calloc(n, 1); P->ts = calloc(n, 1); P->tpl = malloc(sizeof(int) * n); P->tsl = malloc(sizeof(int) * n);
        h->prvl[q] = malloc(sizeof(int) * n); h->fcur[q] = malloc(sizeof(int) * n); h->fprv[q] = malloc(sizeof(int) * n);
    }
    h->prvl[K] = malloc(sizeof(int) * NL);
    for (int s = NL; s < h->NG; s++) h->capa[s] = gi[G_FO];
    if (gd[H_HUBF] > 0) { uint64_t r = 0x9E3779B97F4A7C15ULL ^ (uint64_t)gi[G_SEED]; for (int s = NL; s < h->NG; s++) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; if ((r >> 11) * (1.0 / 9007199254740992.0) < gd[H_HUBF]) h->capa[s] = gi[G_FO] * gi[G_HUBM]; } }
    for (int q = 0; q < K; q++) {
        Pop *P = h->p + q; int pm = P->pm | (P->in ? (1 << 31) : 0);
        for (int i = 0; i < P->n; i++) {
            for (int s = 0; s < P->FP; s++) { int src = pick_source(h, q, i, pm); if (src < 0) continue; int e = i * P->FI + s; P->ssrc[e] = src; P->sw[e] = (float)(0.5 + 0.5 * ur(h)); add_out(h, src, q, i, s); }
            if (!P->ld) for (int s = 0; s < P->FD; s++) { int src = pick_source(h, q, i, P->dm); if (src < 0) continue; int e = i * P->FI + P->FP + s; P->ssrc[e] = src; P->sw[e] = (float)(0.3 + 0.7 * ur(h)); add_out(h, src, q, i, P->FP + s); }
        }
    }
    h->code = malloc(sizeof(int) * V * gi[G_KIN]);
    for (int v = 0; v < V; v++) for (int k = 0; k < gi[G_KIN]; k++) h->code[v * gi[G_KIN] + k] = (int)(xr(h) % (uint64_t)NL);
    int NO = gi[G_NO];
    h->osrc = malloc(sizeof(int) * V * NO); h->ow = calloc(V * NO, sizeof(float)); h->opos = malloc(sizeof(float) * V); h->ocnt = calloc(h->NG, sizeof(int));
    for (int i = 0; i < V * NO; i++) h->osrc[i] = -1;
    for (int v = 0; v < V; v++) h->opos[v] = (float)ur(h);
    h->beta = 1.0; h->lastbyte = -1;
    if (gi[G_TAG]) { h->dtab = malloc(sizeof(float) * DTN); float bse = (float)(1.0 - 1.0 / gd[H_TAUTAG]); for (int d = 0; d < DTN; d++) h->dtab[d] = powf(bse, (float)d); }
    if (gd[H_OTR] > 1) { h->notab = (int)(60 * gd[H_OTR]) + 1; h->otab = malloc(sizeof(float) * (h->notab + 1)); for (int d = 0; d <= h->notab; d++) h->otab[d] = (float)pow(1.0 - 1.0 / gd[H_OTR], d); }
    return h;
}

static void spread(Net *h, int src, int distal) {
    size_t b = (size_t)src * h->FOS; int k0 = distal ? h->FOS - h->fod[src] : 0, k1 = distal ? h->FOS : h->fop[src];
    for (int k = k0; k < k1; k++) {
        Pop *P = h->p + h->opop[b + k]; int i = h->ocell[b + k], s = h->oslot[b + k]; size_t e = (size_t)i * P->FI + s; float w = wef(h, P, e);
        if (w <= 0) continue;
        if (!distal) { if (!P->tp[i]) { P->tp[i] = 1; P->tpl[P->ntp++] = i; } P->pacc[i] += w; }
        else { if (!P->ts[i]) { P->ts[i] = 1; P->tsl[P->nts++] = i; } P->sacc[(size_t)i * P->nseg + P->segof[s]] += w; }
    }
}

static float best_seg(Pop *P, int i) { float b = 0; for (int g = 0; g < P->nseg; g++) { float v = P->sacc[(size_t)i * P->nseg + g]; if (v > b) b = v; } return b; }

static int lbound(const int *L, int n, int v) { int lo = 0, hi = n; while (lo < hi) { int m = (lo + hi) >> 1; if (L[m] < v) lo = m + 1; else hi = m; } return lo; }

typedef struct { const int *L; int lo, hi, gp, reg; } Rg;

/* is source src (of population gp, list entry a of range r) eligible for slots s0..s1 of cell i, and counted only in this range */
static int eligible(Net *h, Pop *P, int q, int i, int s0, int s1, Rg *r, int a, int src, const float *rc, const float *rr, const int *chosen, int nch) {
    if (h->fo[src] >= cap(h, src)) { h->st_rej[0]++; return 0; }
    if (r->gp == q && src - P->off == i) { h->st_rej[1]++; return 0; }
    float sp = h->gpos[src];
    if (r->gp < 0) { if (!reach(P, i, sp)) { h->st_rej[2]++; return 0; } for (int b = r->lo; b < a; b++) if (r->L[b] == r->L[a]) { h->st_rej[3]++; return 0; } }
    else { if (rdist(rc[r->reg], sp) > rr[r->reg]) { h->st_rej[2]++; return 0; } for (int g = 0; g < r->reg; g++) if (rdist(rc[g], sp) <= rr[g]) { h->st_rej[3]++; return 0; } }
    for (int s = s0; s < s1; s++) if (P->ssrc[(size_t)i * P->FI + s] == src) { h->st_rej[4]++; return 0; }
    for (int c = 0; c < nch; c++) if (chosen[c] == src) { h->st_rej[5]++; return 0; }
    return 1;
}

static void grow(Net *h, Pop *P, int q, int i, int s0, int s1, int mask, const uint8_t *mark, int need) {
    if (need <= 0) return;
    if (need > 64) need = 64;
    h->st_grow++;
    Rg R[MAXK * 2 * 9 + 2]; int nr = 0; long M = 0;
    float rc[9], rr[9]; int nreg = 1 + P->np;
    rc[0] = P->pos[i]; rr[0] = P->rho; for (int k = 0; k < P->np; k++) { rc[1 + k] = P->pc[i * P->np + k]; rr[1 + k] = P->rho * 0.5f; }
    if (srcpop_ok(mask, -1)) {
        const int *L; int nl;
        if (mark == h->cur) { L = h->lastbyte < 0 ? h->code : h->code + h->lastbyte * h->gi[G_KIN]; nl = h->lastbyte < 0 ? 0 : h->gi[G_KIN]; }
        else { L = h->prvl[h->K]; nl = h->nprv[h->K]; }
        static int sens[1024]; int m = 0;                  /* sensor lines below their fan-out cap (a byte's KIN lines) */
        for (int a = 0; a < nl && m < 1024; a++) if (h->fo[L[a]] < cap(h, L[a])) sens[m++] = L[a];
        if (m) { R[nr] = (Rg){ sens, 0, m, -1, 0 }; M += m; nr++; }
    }
    for (int gp = 0; gp < h->K; gp++) {
        if (!srcpop_ok(mask, gp)) continue;
        const int *L; int nl;
        if (mark == h->cur) { L = h->fcur[gp]; nl = gp < q ? h->nfcur[gp] : 0; } else { L = h->fprv[gp]; nl = h->nfprv[gp]; }
        if (!nl) continue;
        int ng = h->p[gp].n;
        for (int g = 0; g < nreg; g++) {
            int iv[4], ni = 0;
            if (rr[g] >= 0.5f) { iv[0] = 0; iv[1] = ng - 1; ni = 1; }
            else {
                int a = (int)floorf((rc[g] - rr[g]) * ng - 0.5f) - 1, b = (int)ceilf((rc[g] + rr[g]) * ng - 0.5f) + 1;
                if (b - a + 1 >= ng) { iv[0] = 0; iv[1] = ng - 1; ni = 1; }
                else if (a < 0) { iv[0] = a + ng; iv[1] = ng - 1; iv[2] = 0; iv[3] = b; ni = 2; }
                else if (b >= ng) { iv[0] = a; iv[1] = ng - 1; iv[2] = 0; iv[3] = b - ng; ni = 2; }
                else { iv[0] = a; iv[1] = b; ni = 1; }
            }
            for (int u = 0; u < ni; u++) {
                int lo = lbound(L, nl, iv[2 * u]), hi = lbound(L, nl, iv[2 * u + 1] + 1);
                if (hi > lo) { R[nr] = (Rg){ L, lo, hi, gp, g }; M += hi - lo; nr++; }
            }
        }
    }
    if (!M) return;
    int chosen[64], nch = 0, trials = 0, maxt = M <= 2 * need + 8 ? 0 : 4 * need + 16;   /* few candidates: enumerate them directly */
    while (nch < need && trials < maxt) {
        trials++; h->st_draw++;
        long u = (long)(xr(h) % (uint64_t)M); int t = 0; while (u >= R[t].hi - R[t].lo) { u -= R[t].hi - R[t].lo; t++; }
        int a = R[t].lo + (int)u, src = R[t].gp < 0 ? R[t].L[a] : h->p[R[t].gp].off + R[t].L[a];
        if (eligible(h, P, q, i, s0, s1, R + t, a, src, rc, rr, chosen, nch)) chosen[nch++] = src;
    }
    if (nch < need) {                                   /* draws keep failing: enumerate the in-reach ranges, draw from what is left */
        h->st_fall++; h->st_fM += M; h->st_fneed += need - nch; int nc = 0;
        for (int t = 0; t < nr; t++) for (int a = R[t].lo; a < R[t].hi; a++) {
            int src = R[t].gp < 0 ? R[t].L[a] : h->p[R[t].gp].off + R[t].L[a];
            if (eligible(h, P, q, i, s0, s1, R + t, a, src, rc, rr, chosen, nch)) h->cand[nc++] = src;
        }
        h->st_fnc += nc;
        for (int t = 0; t < nc && nch < need; t++) { int j = t + (int)(xr(h) % (uint64_t)(nc - t)); int z = h->cand[t]; h->cand[t] = h->cand[j]; h->cand[j] = z; chosen[nch++] = h->cand[t]; }
    }
    for (int t = 0; t < nch; t++) {
        int c = chosen[t]; if (h->fo[c] >= cap(h, c)) continue;
        int bs = -1; float bw = 2;
        for (int s = s0; s < s1; s++) { int e = i * P->FI + s, u = P->ssrc[e]; if (u >= 0 && mark[u]) continue; float w = u < 0 ? -1 : wef(h, P, e); if (w < bw) { bw = w; bs = s; } }
        if (bs < 0) break;
        int e = i * P->FI + bs, old = P->ssrc[e];
        if (old >= 0) del_out(h, old, q, e);
        P->ssrc[e] = c; add_out(h, c, q, i, bs);
        if (h->gi[G_TAG]) { P->sw[e] = 0; P->sl[e] = (float)h->gd[H_WNEW]; P->tl[e] = h->T; } else P->sw[e] = (float)h->gd[H_WNEW];
    }
}

static float kth_largest(float *a, int m, int k) {               /* value of rank m-k in ascending order (a is permuted) */
    int lo = 0, hi = m - 1, want = m - k;
    while (lo < hi) {
        float piv = a[(lo + hi) >> 1]; int i = lo, j = hi;
        while (i <= j) { while (a[i] < piv) i++; while (a[j] > piv) j--; if (i <= j) { float t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; } }
        if (want <= j) hi = j; else if (want >= i) lo = i; else return a[want];
    }
    return a[want];
}

static void pop_step(Net *h, int q, int learn) {
    Pop *P = h->p + q; int n = P->n;
    for (int a = 0; a < P->nact; a++) { P->y[P->act[a]] = 0; }
    P->nact = 0;
    float dg = P->dg, sth = P->sth * P->G;
    for (int i = 0; i < n; i++) {
        float d = P->pacc[i];
        if (P->nseg && P->ts[i]) { float bsg = best_seg(P, i); d += bsg >= sth ? dg : dg * 0.3f * bsg / P->G; }
        P->h[i] = P->dec[i] * P->h[i] + (1 - P->dec[i]) * d;
        float v = P->h[i];
        if (P->ad) { P->m[i] += (P->h[i] - P->m[i]) * 0.001f; v -= P->m[i]; }
        P->val[i] = v;
    }
    for (int b0 = 0; b0 < n; b0 += P->B) {                       /* the k largest above 1e-6 in each block, ties to the lower index */
        int b1 = b0 + P->B < n ? b0 + P->B : n, m = 0;
        for (int i = b0; i < b1; i++) if (P->val[i] > 1e-6f) h->vtmp[m++] = P->val[i];
        if (m <= P->k) { for (int i = b0; i < b1; i++) if (P->val[i] > 1e-6f) { P->y[i] = 1; P->act[P->nact++] = i; } continue; }
        float th = kth_largest(h->vtmp, m, P->k); int ngt = 0;
        for (int i = b0; i < b1; i++) ngt += P->val[i] > th;
        int eq = P->k - ngt;
        for (int i = b0; i < b1; i++) { float v = P->val[i]; if (v > th || (v == th && eq-- > 0)) { P->y[i] = 1; P->act[P->nact++] = i; } }
    }
    for (int a = 0; a < P->nact; a++) { int i = P->act[a]; h->cur[P->off + i] = 1; h->lastf[P->off + i] = h->T; if (learn) P->use[i]++; }
    if (learn) {
        int pm = P->pm | (P->in ? (1 << 31) : 0);
        uint8_t blk[4096]; int nb = (n + P->B - 1) / P->B; if (nb > 4096) nb = 4096;
        if (h->gi[G_GATE]) {
            static int nf[4096], nu[4096]; memset(nf, 0, sizeof(int) * nb); memset(nu, 0, sizeof(int) * nb);
            for (int a = 0; a < P->nact; a++) { int i = P->act[a], bb = i / P->B; if (bb >= nb) continue; nf[bb]++; if (!(P->nseg && P->ts[i] && best_seg(P, i) >= sth)) nu[bb]++; }
            for (int bb = 0; bb < nb; bb++) blk[bb] = nf[bb] && nu[bb] >= h->gd[H_GTH] * nf[bb];
        } else memset(blk, 1, nb);
        double etam = h->gd[H_ETAM] * h->etamul;
        if (P->ld) for (int k = 0; k < P->nts; k++) {
            int i = P->tsl[k]; if (P->y[i] || best_seg(P, i) < sth || !blk[i / P->B < nb ? i / P->B : nb - 1]) continue;
            for (int s = P->FP; s < P->FI; s++) { int e = i * P->FI + s, u = P->ssrc[e]; if (u >= 0 && h->prv[u]) dep(h, P, e, etam); }
        }
        for (int a = 0; a < P->nact; a++) {
            int i = P->act[a];
            int strong = P->nseg && P->ts[i] && best_seg(P, i) >= sth;
            if (h->gi[G_TAG] && strong) for (int s = 0; s < P->FI; s++) { size_t e = (size_t)i * P->FI + s; float l = lab(h, P, e); float c = (float)h->gd[H_CAP] * l; P->sw[e] += c; P->sl[e] = l - c; }
            if (!blk[i / P->B < nb ? i / P->B : nb - 1]) continue;
            double bo = (h->gd[H_TURN] > 0 && h->T - P->born[i] < 2000) ? h->gd[H_BOOST] : 1.0;
            double eta = h->gd[H_ETA] * bo * h->etamul;
            if (P->lp) {
                int na = 0;
                for (int s = 0; s < P->FP; s++) { int e = i * P->FI + s, u = P->ssrc[e]; if (u < 0) continue; if (h->cur[u]) { pot(h, P, e, eta); na++; } else dep(h, P, e, etam); }
                grow(h, P, q, i, 0, P->FP, pm, h->cur, h->gi[G_NGP] - na);
            }
            if (P->ld && P->FD) {
                int na = 0;
                for (int s = P->FP; s < P->FI; s++) { int e = i * P->FI + s, u = P->ssrc[e]; if (u >= 0 && h->prv[u]) { pot(h, P, e, eta); na++; } }
                grow(h, P, q, i, P->FP, P->FI, P->dm, h->prv, h->gi[G_NGD] - na);
            }
        }
    }
    for (int k = 0; k < P->ntp; k++) { int i = P->tpl[k]; P->pacc[i] = 0; P->tp[i] = 0; }
    P->ntp = 0;
}

static void clear_distal(Net *h) {
    for (int q = 0; q < h->K; q++) { Pop *P = h->p + q;
        for (int k = 0; k < P->nts; k++) { int i = P->tsl[k]; memset(P->sacc + (size_t)i * P->nseg, 0, sizeof(float) * P->nseg); P->ts[i] = 0; }
        P->nts = 0; }
}

static uint64_t kth_key(uint64_t *a, int m, int r) {            /* r-th smallest (0-based) of distinct keys (a is permuted) */
    int lo = 0, hi = m - 1;
    while (lo < hi) {
        uint64_t piv = a[(lo + hi) >> 1]; int i = lo, j = hi;
        while (i <= j) { while (a[i] < piv) i++; while (a[j] > piv) j--; if (i <= j) { uint64_t t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; } }
        if (r <= j) hi = j; else if (r >= i) lo = i; else return a[r];
    }
    return a[r];
}

static void turnover(Net *h) {                                  /* the r least-used old cells (ties to the lower index) */
    for (int q = 0; q < h->K; q++) { Pop *P = h->p + q; int r = (int)(h->gd[H_TURN] * 0.1 * P->n + 0.5), m = 0;
        if (r <= 0) continue;
        for (int i = 0; i < P->n; i++) if (h->T - P->born[i] > 2000 && !P->y[i]) h->keys[m++] = ((uint64_t)(uint32_t)P->use[i] << 32) | (uint32_t)i;
        if (!m) continue;
        if (r > m) r = m;
        uint64_t th = kth_key(h->keys, m, r - 1);
        for (int i = 0; i < P->n; i++) {
            if (!(h->T - P->born[i] > 2000 && !P->y[i])) continue;
            if ((((uint64_t)(uint32_t)P->use[i] << 32) | (uint32_t)i) > th) continue;
            for (int s = P->FP; s < P->FI; s++) { int e = i * P->FI + s, u = P->ssrc[e]; if (u >= 0 && P->ld) { del_out(h, u, q, e); P->ssrc[e] = -1; P->sw[e] = 0; P->sl[e] = 0; } }
            P->h[i] = 0; P->m[i] = 0; P->use[i] = 0; P->born[i] = h->T;
        } }
}
static inline double otr(Net *h, int s) { int d = h->T - h->lastf[s]; if (d == 0) return 1.0; if (h->gd[H_OTR] <= 1 || d > 60 * h->gd[H_OTR]) return 0; return h->otab[d]; }

void lt3o_run(Net *h, const uint8_t *x, int n, int learn, double *stats) {
    if (h->etamul == 0) h->etamul = 1;
    int NO = h->gi[G_NO], KIN = h->gi[G_KIN], FOO = h->gi[G_FOO];
    double eo = h->gd[H_ETAO], eb = h->gd[H_ETAB], et = h->gd[H_ETAT], rhoo = h->gd[H_RHOO]; double p[V];
    for (int t = 0; t < n; t++) {
        int y = x[t] & 127;
        if (h->have) {
            double mx = -1e300; int arg = 0;
            for (int v = 0; v < V; v++) { double z = h->beta * h->o[v] + h->ct[v]; p[v] = z; if (z > mx) mx = z; if (h->o[v] > h->o[arg]) arg = v; }
            double Z = 0; for (int v = 0; v < V; v++) { p[v] = exp(p[v] - mx); Z += p[v]; }
            for (int v = 0; v < V; v++) p[v] /= Z;
            stats[0] += -log(p[y] + 1e-12); stats[1] += arg == y; stats[2] += 1;
            if (learn) {
                double gb = 0; for (int v = 0; v < V; v++) { double gv = (v == y) - p[v]; gb += gv * h->o[v]; h->ct[v] += et * gv; }
                h->beta += et * gb;
                for (int v = 0; v < V; v++) {
                    if (v != y && h->o[v] <= 0) continue;
                    double e = (v == y) - h->o[v], na = 0;
                    for (int k = 0; k < NO; k++) { int s = h->osrc[v * NO + k]; if (s >= 0) na += otr(h, s); }
                    h->b[v] += eb * e;
                    if (na > 0) for (int k = 0; k < NO; k++) { int s = h->osrc[v * NO + k]; double tr; if (s >= 0 && (tr = otr(h, s)) > 0) h->ow[v * NO + k] += (float)(eo * e * tr / na); }
                }
                int have = 0; for (int k = 0; k < NO; k++) have += h->osrc[y * NO + k] >= 0;
                for (int q = 0; q < h->K && have < NO; q++) {
                    Pop *P = h->p + q; if (!P->rd) continue;
                    for (int a = 0; a < P->nact && have < NO; a++) {
                        int s = P->off + P->act[a]; if (h->ocnt[s] >= FOO || rdist(h->opos[y], h->gpos[s]) > rhoo) continue;
                        int dup = 0; for (int k = 0; k < NO; k++) if (h->osrc[y * NO + k] == s) dup = 1;
                        if (dup) continue;
                        for (int k = 0; k < NO; k++) if (h->osrc[y * NO + k] < 0) { h->osrc[y * NO + k] = s; h->ow[y * NO + k] = 0; h->ocnt[s]++; have++; break; }
                    }
                }
                if (have >= NO && h->gd[H_OREP] > 0) {
                    int wk = -1; float ww = (float)h->gd[H_OREP];
                    for (int k = 0; k < NO; k++) { float w = fabsf(h->ow[y * NO + k]); if (w < ww) { ww = w; wk = k; } }
                    for (int q = 0; q < h->K && wk >= 0; q++) { Pop *P = h->p + q; if (!P->rd || !P->nact) continue;
                        for (int t2 = 0; t2 < 4; t2++) {
                            int s = P->off + P->act[(int)(xr(h) % (uint64_t)P->nact)]; if (h->ocnt[s] >= FOO || rdist(h->opos[y], h->gpos[s]) > rhoo) continue;
                            int dup = 0; for (int k = 0; k < NO; k++) if (h->osrc[y * NO + k] == s) dup = 1;
                            if (dup) continue;
                            h->ocnt[h->osrc[y * NO + wk]]--; h->osrc[y * NO + wk] = s; h->ow[y * NO + wk] = 0; h->ocnt[s]++; wk = -1; break; } }
                }
            }
        }
        h->T++;
        if (learn && h->gd[H_TURN] > 0 && h->T % 1000 == 0) turnover(h);
        for (int q = 0; q <= h->K; q++) { int *L = h->prvl[q]; for (int a = 0; a < h->nprv[q]; a++) h->prv[q < h->K ? h->p[q].off + L[a] : L[a]] = 0; }
        for (int q = 0; q <= h->K; q++) {
            if (q < h->K) { Pop *P = h->p + q; memcpy(h->prvl[q], P->act, sizeof(int) * P->nact); h->nprv[q] = P->nact; }
            else { h->nprv[q] = h->lastbyte < 0 ? 0 : KIN; if (h->lastbyte >= 0) memcpy(h->prvl[q], h->code + h->lastbyte * KIN, sizeof(int) * KIN); }
        }
        uint8_t *tmp = h->prv; h->prv = h->cur; h->cur = tmp;
        h->lastbyte = y;
        if (learn) for (int q = 0; q < h->K; q++) { int m = 0, off = h->p[q].off; for (int a = 0; a < h->nprv[q]; a++) { int c = h->prvl[q][a]; if (h->fo[off + c] < cap(h, off + c)) h->fprv[q][m++] = c; } h->nfprv[q] = m; }
        for (int k = 0; k < KIN; k++) { int s = h->code[y * KIN + k]; if (!h->cur[s]) { h->cur[s] = 1; spread(h, s, 0); } }
        for (int q = 0; q < h->K; q++) {
            pop_step(h, q, learn);
            Pop *P = h->p + q; for (int a = 0; a < P->nact; a++) spread(h, P->off + P->act[a], 0);
            if (learn) { int m = 0; for (int a = 0; a < P->nact; a++) { int c = P->act[a]; if (h->fo[P->off + c] < cap(h, P->off + c)) h->fcur[q][m++] = c; } h->nfcur[q] = m; }
        }
        clear_distal(h);
        for (int q = 0; q < h->K; q++) { Pop *P = h->p + q; for (int a = 0; a < P->nact; a++) spread(h, P->off + P->act[a], 1); }
        for (int q = 0; q < h->K; q++) { Pop *P = h->p + q; for (int k = 0; k < P->ntp; k++) { int i = P->tpl[k]; P->pacc[i] = 0; P->tp[i] = 0; } P->ntp = 0; }
        for (int v = 0; v < V; v++) {
            double s = h->b[v];
            for (int k = 0; k < NO; k++) { int src = h->osrc[v * NO + k]; double tr; if (src >= 0 && (tr = otr(h, src)) > 0) s += h->ow[v * NO + k] * tr; }
            h->o[v] = s;
        }
        h->have = 1;
    }
}

#define SNAPF(X) X(&h->T, 1); X(h->lastf, h->NG); X(h->cur, h->NG); X(h->prv, h->NG); X(h->o, V); X(&h->have, 1); X(&h->rs, 1); X(&h->lastbyte, 1); X(h->nprv, MAXK + 1); \
    for (int q = 0; q <= h->K; q++) X(h->prvl[q], q < h->K ? h->p[q].n : h->gi[G_NL]); \
    for (int q = 0; q < h->K; q++) { Pop *P = h->p + q; X(P->h, P->n); X(P->m, P->n); X(P->y, P->n); X(P->act, P->n); X(&P->nact, 1); \
        X(P->sacc, (size_t)P->n * (P->nseg + 1)); X(P->ts, P->n); X(P->tsl, P->n); X(&P->nts, 1); }
#define SZ(p, k) sz += sizeof(*(p)) * (size_t)(k);
#define SV(p, k) { memcpy(c, p, sizeof(*(p)) * (size_t)(k)); c += sizeof(*(p)) * (size_t)(k); }
#define LD(p, k) { memcpy(p, c, sizeof(*(p)) * (size_t)(k)); c += sizeof(*(p)) * (size_t)(k); }
void lt3o_snapshot(Net *h) { size_t sz = 0; SNAPF(SZ); if (!h->snap) h->snap = malloc(sz); char *c = h->snap; SNAPF(SV); }
void lt3o_restore(Net *h) { char *c = h->snap; SNAPF(LD); }

void lt3o_audit(Net *h, long *a) {
    memset(a, 0, sizeof(long) * 6);
    for (int q = 0; q < h->K; q++) { Pop *P = h->p + q;
        for (int i = 0; i < P->n; i++) { int c = 0; for (int s = 0; s < P->FI; s++) c += P->ssrc[i * P->FI + s] >= 0; if (c > a[0]) a[0] = c; a[5] += c; } }
    for (int s = 0; s < h->NG; s++) { long *t = s < h->gi[G_NL] ? &a[2] : &a[1]; if (h->fo[s] > *t) *t = h->fo[s]; if (h->ocnt[s] > a[4]) a[4] = h->ocnt[s]; }
    int NO = h->gi[G_NO]; for (int v = 0; v < V; v++) { int c = 0; for (int k = 0; k < NO; k++) c += h->osrc[v * NO + k] >= 0; if (c > a[3]) a[3] = c; }
}
/* [0] grow calls, [1] draws, [2] calls that fell back to enumerating the in-reach ranges; [3] 1 if every out-list is consistent;
   [4] sum of range sizes at fallbacks, [5] eligible found at fallbacks, [6] still needed at fallbacks;
   [7..12] rejections: full, self, outside exact reach, counted in an earlier region, already connected, already chosen */
void lt3o_stats(Net *h, long *a) {
    a[0] = h->st_grow; a[1] = h->st_draw; a[2] = h->st_fall; a[3] = 1; a[4] = h->st_fM; a[5] = h->st_fnc; a[6] = h->st_fneed; for (int j = 0; j < 6; j++) a[7 + j] = h->st_rej[j];
    for (int s = 0; s < h->NG && a[3]; s++) {
        if (h->fop[s] + h->fod[s] != h->fo[s] || h->fo[s] > cap(h, s) || h->fop[s] + h->fod[s] > h->FOS) { a[3] = 0; break; }
        size_t b = (size_t)s * h->FOS;
        for (int k = 0; k < h->FOS; k++) {
            if (!(k < h->fop[s] || k >= h->FOS - h->fod[s])) continue;
            Pop *P = h->p + h->opop[b + k]; size_t e = (size_t)h->ocell[b + k] * P->FI + h->oslot[b + k];
            if (P->ssrc[e] != s || P->bk[e] != k || ((h->oslot[b + k] < P->FP) != (k < h->fop[s]))) { a[3] = 0; break; }
        }
    }
}
