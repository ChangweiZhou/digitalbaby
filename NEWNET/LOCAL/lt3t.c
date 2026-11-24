/* lt3.c - lt2 plus mechanism dimensions (24 Nov 2026), each off by default (off = lt2's computation):
   TAG (synaptic tagging and capture: changes go to a labile part decaying with TAUTAG bytes; a cell that fires with a dendritic
   event moves CAP of its labile part into the lasting part), GATE (plasticity in a block only when the block is surprised: at least GTH
   of its firing cells had no dendritic event), TURN (every 1,000 bytes the share TURN/10 of the least-used cells of each population
   are cleared and learn BOOST times faster for 2,000 bytes), OTR (output presynaptic trace with time constant OTR bytes),
   HUBF/HUBM (a share HUBF of cells may make HUBM times FO outgoing synapses).
   Base: lt2.c - a local tissue of K populations (cell kinds), 24 Nov 2026. Event-driven synapses; one call reads a whole phase.
   Sheet: every population and the sensor array lie on one ring [0,1). A cell reads only sources within its reach (radius rho) or
   inside its own npatch distant patches (fixed at birth); every cell has FP proximal + FD distal slots (fan-in) and every source has
   at most FO outgoing synapses (fan-out; sensor lines FOI). Only populations with reads_in read the sensor lines; only readable
   populations are read by the output cells (fan-in NO, fan-out to outputs FOO).
   Byte -> sensor array: NL lines, each byte a fixed random set of KIN lines (a distributed code over the whole sheet).
   Order of a byte (sweep): population p takes proximal input from the sensor lines (if reads_in) and from populations q < p at this
   byte; distal input from populations q >= p (and itself) one byte late, in segments of G synapses (a segment over SEG_TH of its
   size is a dendritic event). drive = proximal + dgain * (dendritic event of the best segment, else 0.3 * its fraction);
   h = dec*h + (1-dec)*drive with each cell's own time constant (log-uniform tau_lo..tau_hi); adapt: send h less its running mean;
   local inhibition: the k largest of each block of B cells fire.
   Learning (local, per population flags): learnp - firing cells strengthen active proximal slots, weaken inactive ones, and grow
   slots to active sources; learnd - firing cells strengthen distal slots active one byte back and grow to them; cells with a
   dendritic event that did not fire weaken their active distal slots. Output cell v: o_v = b_v + sum w * y (readable cells now);
   delta rule on its own target when the next byte arrives; output cell y grows synapses to readable cells active one byte back.
   Instrument (not the net): p = softmax(beta*o + c), learned online by log loss; nats = -log p[y]. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define V 128
#define MAXK 6
enum { P_N, P_FP, P_FD, P_G, P_B, P_K, P_LP, P_LD, P_AD, P_IN, P_RD, P_NP, P_PM, P_DM, P_NI };
enum { Q_TLO, Q_THI, Q_RHO, Q_PF, Q_STH, Q_DG, Q_ND };
enum { G_K, G_NL, G_KIN, G_FOI, G_FO, G_NO, G_FOO, G_NGP, G_NGD, G_SEED, G_TAG, G_GATE, G_HUBM, G_NI };
enum { H_ETA, H_ETAM, H_WNEW, H_ETAO, H_ETAB, H_ETAT, H_RHOO, H_OREP, H_TAUTAG, H_CAP, H_GTH, H_TURN, H_BOOST, H_OTR, H_HUBF, H_ND };

typedef struct {
    int n, FP, FD, FI, G, nseg, B, k, lp, ld, ad, in, rd, np, pm, dm, off;
    float tlo, thi, rho, pf, sth, dg;
    float *pos, *dec, *h, *m, *val, *pc;
    int *ssrc; float *sw, *sl; int *tl, *use, *born; int *ki;     /* ki: position of each slot in its source's outgoing list */
    uint8_t *y; int *act, nact;
    float *pacc, *sacc; uint8_t *tp, *ts; int *tpl, ntp, *tsl, nts;
} Pop;

typedef struct {
    int gi[G_NI]; double gd[H_ND]; int K, NG, FOS;
    Pop p[MAXK];
    float *gpos; int *gpop, *fo; int *opop, *oslot;          /* sources: 0..NL-1 sensor lines, then each population's cells */
    uint8_t *cur, *prv; int *code;                           /* byte -> KIN lines */
    int *prvl[MAXK + 1], nprv[MAXK + 1];                     /* active one byte back, per population (+ sensor at index K) */
    int *osrc; float *ow, *opos; int *ocnt; double b[V], o[V], beta, ct[V];
    int have; uint64_t rs; int lastbyte; int T; int *capa, *lastf; float etamul;
    int *cand; void *snap;
    /* exact growth search (lt3t): per population p and version v (0 = firing now, 1 = one byte back), the active list's block
       starts, a Fenwick tree over list positions of 'source not full', the full flags, and each cell's list position */
    float ptab[4096]; int ptab_ok;
    int *bs[MAXK][2], *fw[MAXK][2], bT[MAXK][2], bm[MAXK][2]; uint8_t *ff[MAXK][2]; int *lp_[MAXK][2], *lst[MAXK][2]; const int *bl[MAXK][2];
} Net;

static inline uint64_t xr(Net *h) { h->rs ^= h->rs << 13; h->rs ^= h->rs >> 7; h->rs ^= h->rs << 17; return h->rs; }
static inline double ur(Net *h) { return (xr(h) >> 11) * (1.0 / 9007199254740992.0); }
static inline float rdist(float a, float b) { float d = fabsf(a - b); return d > 0.5f ? 1.0f - d : d; }

static inline float lab(Net *h, Pop *P, size_t e) {           /* labile part, decayed to now */
    if (!h->gi[G_TAG]) return 0;
    int dt = h->T - P->tl[e];
    if (dt > 0) {                                                /* the same powf values, from a table for short gaps */
        float d = (float)(1.0 - 1.0 / h->gd[H_TAUTAG]);
        if (!h->ptab_ok) { for (int k = 0; k < 4096; k++) h->ptab[k] = powf(d, (float)k); h->ptab_ok = 1; }
        P->sl[e] *= dt < 4096 ? h->ptab[dt] : powf(d, (float)dt); P->tl[e] = h->T; }
    return P->sl[e];
}
static inline float wef(Net *h, Pop *P, size_t e) { return P->sw[e] + lab(h, P, e); }
static inline void pot(Net *h, Pop *P, size_t e, double eta) {   /* potentiation: lasting part (lt2) or labile part (TAG) */
    if (!h->gi[G_TAG]) { P->sw[e] += (float)(eta * (1 - P->sw[e])); return; }
    float l = lab(h, P, e); P->sl[e] = l + (float)(eta * (1 - P->sw[e] - l));
}
static inline void dep(Net *h, Pop *P, size_t e, double etam) { P->sw[e] -= (float)(etam * P->sw[e]); if (h->gi[G_TAG]) { float l = lab(h, P, e); P->sl[e] = l - (float)(etam * l); } }
static int reach(Pop *P, int i, float sp) {                  /* can cell i of P connect to a source at position sp */
    if (rdist(P->pos[i], sp) <= P->rho) return 1;
    for (int k = 0; k < P->np; k++) if (rdist(P->pc[i * P->np + k], sp) <= P->rho * 0.5f) return 1;
    return 0;
}
static int srcpop_ok(int mask, int gp) { return (mask >> (gp < 0 ? 31 : gp)) & 1; }   /* bit 31: sensor lines */

static inline void fw_add(int *f, int m, int a, int d) { for (a++; a <= m; a += a & -a) f[a] += d; }
static inline int fw_pre(const int *f, int a) { int s = 0; for (; a > 0; a -= a & -a) s += f[a]; return s; }   /* sum over [0, a) */
static inline int fw_kth(const int *f, int m, int k) {          /* smallest position a with sum over [0, a] > k */
    int pos = 0, lg = 1; while (lg * 2 <= m) lg *= 2;
    for (; lg; lg >>= 1) if (pos + lg <= m && f[pos + lg] <= k) { pos += lg; k -= f[pos]; }
    return pos;
}
static int capf(Net *h, int src);
static void fo_changed(Net *h, int src) {                        /* keep the 'not full' trees of this byte's lists current */
    int gp = h->gpop[src]; if (gp < 0) return; int c = src - h->p[gp].off;
    for (int v = 0; v < 2; v++) {
        if (!h->lst[gp][v] || h->bT[gp][v] != h->T || h->lst[gp][v][c] != h->T) continue;
        int a = h->lp_[gp][v][c]; uint8_t nf = h->fo[src] < capf(h, src);
        if (nf != h->ff[gp][v][a]) { h->ff[gp][v][a] = nf; fw_add(h->fw[gp][v], h->bm[gp][v], a, nf ? 1 : -1); }
    }
}
static void add_out(Net *h, int src, int pop, int slot) { int k = h->fo[src]++; h->opop[(size_t)src * h->FOS + k] = pop; h->oslot[(size_t)src * h->FOS + k] = slot; h->p[pop].ki[slot] = k; fo_changed(h, src); }
static void del_out(Net *h, int src, int pop, int slot) {
    size_t b = (size_t)src * h->FOS;
    int k = h->p[pop].ki[slot];                                    /* direct position; same swap with the last entry as the search did */
    if (0) {
        int l = --h->fo[src]; h->opop[b + k] = h->opop[b + l]; h->oslot[b + k] = h->oslot[b + l]; h->p[h->opop[b + k]].ki[h->oslot[b + k]] = k; fo_changed(h, src); return; }
    for (k = 0; k < h->fo[src]; k++) if (h->opop[b + k] == pop && h->oslot[b + k] == slot) { int l = --h->fo[src]; h->opop[b + k] = h->opop[b + l]; h->oslot[b + k] = h->oslot[b + l]; h->p[h->opop[b + k]].ki[h->oslot[b + k]] = k; fo_changed(h, src); return; }
}
static int cap(Net *h, int src) { return h->capa[src]; }
static int capf(Net *h, int src) { return h->capa[src]; }

/* a random source for cell i of population q, among allowed populations, within reach, below its fan-out cap */
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

Net *lt3t_new(const int *gi, const double *gd, const int *pi, const double *pd) {
    Net *h = calloc(1, sizeof(Net)); memcpy(h->gi, gi, sizeof(int) * G_NI); memcpy(h->gd, gd, sizeof(double) * H_ND);
    h->rs = 88172645463325252ULL ^ ((uint64_t)gi[G_SEED] * 2654435761ULL); for (int i = 0; i < 20; i++) xr(h);
    int K = h->K = gi[G_K], NL = gi[G_NL];
    h->NG = NL;
    for (int q = 0; q < K; q++) {
        Pop *P = h->p + q; const int *a = pi + q * P_NI; const double *d = pd + q * Q_ND;
        P->n = a[P_N]; P->FP = a[P_FP]; P->FD = a[P_FD]; P->FI = P->FP + P->FD; P->G = a[P_G] > 0 ? a[P_G] : (P->FD > 0 ? P->FD : 1);
        P->nseg = P->FD ? (P->FD + P->G - 1) / P->G : 0; P->B = a[P_B]; P->k = a[P_K]; P->lp = a[P_LP]; P->ld = a[P_LD]; P->ad = a[P_AD];
        P->in = a[P_IN]; P->rd = a[P_RD]; P->np = a[P_NP]; P->pm = a[P_PM]; P->dm = a[P_DM];
        P->tlo = d[Q_TLO]; P->thi = d[Q_THI]; P->rho = d[Q_RHO]; P->pf = d[Q_PF]; P->sth = d[Q_STH]; P->dg = d[Q_DG];
        P->off = h->NG; h->NG += P->n;
    }
    int hm = gd[H_HUBF] > 0 ? gi[G_HUBM] : 1; h->FOS = gi[G_FOI] > gi[G_FO] * hm ? gi[G_FOI] : gi[G_FO] * hm;
    h->gpos = malloc(sizeof(float) * h->NG); h->gpop = malloc(sizeof(int) * h->NG); h->fo = calloc(h->NG, sizeof(int));
    h->opop = malloc(sizeof(int) * (size_t)h->NG * h->FOS); h->oslot = malloc(sizeof(int) * (size_t)h->NG * h->FOS);
    h->capa = malloc(sizeof(int) * h->NG); h->lastf = malloc(sizeof(int) * h->NG); for (int s = 0; s < h->NG; s++) h->lastf[s] = -1000000;
    h->cur = calloc(h->NG, 1); h->prv = calloc(h->NG, 1); h->cand = malloc(sizeof(int) * h->NG);
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
        P->ssrc = malloc(sizeof(int) * (size_t)n * P->FI); P->sw = calloc((size_t)n * P->FI, sizeof(float));
        P->ki = malloc(sizeof(int) * (size_t)n * P->FI);
        P->sl = calloc((size_t)n * P->FI, sizeof(float)); P->tl = calloc((size_t)n * P->FI, sizeof(int)); P->use = calloc(n, sizeof(int)); P->born = malloc(sizeof(int) * n); for (int i = 0; i < n; i++) P->born[i] = -1000000;
        for (size_t e = 0; e < (size_t)n * P->FI; e++) P->ssrc[e] = -1;
        P->y = calloc(n, 1); P->act = malloc(sizeof(int) * n);
        P->pacc = calloc(n, sizeof(float)); P->sacc = calloc((size_t)n * (P->nseg + 1), sizeof(float));
        P->tp = calloc(n, 1); P->ts = calloc(n, 1); P->tpl = malloc(sizeof(int) * n); P->tsl = malloc(sizeof(int) * n);
        h->prvl[q] = malloc(sizeof(int) * n);
    }
    h->prvl[K] = malloc(sizeof(int) * NL);
    for (int s = NL; s < h->NG; s++) h->capa[s] = gi[G_FO];
    if (gd[H_HUBF] > 0) { uint64_t r = 0x9E3779B97F4A7C15ULL ^ (uint64_t)gi[G_SEED]; for (int s = NL; s < h->NG; s++) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; if ((r >> 11) * (1.0 / 9007199254740992.0) < gd[H_HUBF]) h->capa[s] = gi[G_FO] * gi[G_HUBM]; } }
    /* birth wiring: proximal slots from allowed sources (sensor lines only for reads_in populations); distal slots only when they do not learn */
    for (int q = 0; q < K; q++) {
        Pop *P = h->p + q; int pm = P->pm | (P->in ? (1 << 31) : 0);
        for (int i = 0; i < P->n; i++) {
            for (int s = 0; s < P->FP; s++) { int src = pick_source(h, q, i, pm); if (src < 0) continue; int e = i * P->FI + s; P->ssrc[e] = src; P->sw[e] = (float)(0.5 + 0.5 * ur(h)); add_out(h, src, q, e); }
            if (!P->ld) for (int s = 0; s < P->FD; s++) { int src = pick_source(h, q, i, P->dm); if (src < 0) continue; int e = i * P->FI + P->FP + s; P->ssrc[e] = src; P->sw[e] = (float)(0.3 + 0.7 * ur(h)); add_out(h, src, q, e); }
        }
    }
    h->code = malloc(sizeof(int) * V * gi[G_KIN]);
    for (int v = 0; v < V; v++) for (int k = 0; k < gi[G_KIN]; k++) h->code[v * gi[G_KIN] + k] = (int)(xr(h) % (uint64_t)NL);
    int NO = gi[G_NO];
    h->osrc = malloc(sizeof(int) * V * NO); h->ow = calloc(V * NO, sizeof(float)); h->opos = malloc(sizeof(float) * V); h->ocnt = calloc(h->NG, sizeof(int));
    for (int i = 0; i < V * NO; i++) h->osrc[i] = -1;
    for (int v = 0; v < V; v++) h->opos[v] = (float)ur(h);
    for (int q = 0; q < K; q++) for (int v = 0; v < 2; v++) { Pop *P = h->p + q; int nb = P->n / P->B + 2;
        h->bs[q][v] = malloc(sizeof(int) * (nb + 1)); h->fw[q][v] = malloc(sizeof(int) * (P->n + 1)); h->ff[q][v] = malloc(P->n + 1);
        h->lp_[q][v] = malloc(sizeof(int) * P->n); h->lst[q][v] = malloc(sizeof(int) * P->n); for (int i = 0; i < P->n; i++) h->lst[q][v][i] = -1; h->bT[q][v] = -1; }
    h->beta = 1.0; h->lastbyte = -1;
    return h;
}

/* walk the outgoing synapses of an active source; proximal slots add to pacc, distal slots to sacc */
static void spread(Net *h, int src, int distal) {
    size_t b = (size_t)src * h->FOS;
    for (int k = 0; k < h->fo[src]; k++) {
        Pop *P = h->p + h->opop[b + k]; int e = h->oslot[b + k], i = e / P->FI, s = e % P->FI; float w = wef(h, P, e);
        if (w <= 0) continue;
        if (s < P->FP) { if (distal) continue; if (!P->tp[i]) { P->tp[i] = 1; P->tpl[P->ntp++] = i; } P->pacc[i] += w; }
        else { if (!distal) continue; if (!P->ts[i]) { P->ts[i] = 1; P->tsl[P->nts++] = i; } P->sacc[(size_t)i * P->nseg + (s - P->FP) / P->G] += w; }
    }
}

static float best_seg(Pop *P, int i) { float b = 0; for (int g = 0; g < P->nseg; g++) { float v = P->sacc[(size_t)i * P->nseg + g]; if (v > b) b = v; } return b; }

static void grow_scan(Net *h, Pop *P, int q, int i, int s0, int s1, int gp, const int *L, int nl, int *out, int *no) {   /* the original scan */
    for (int a = 0; a < nl; a++) {
        int src = gp < 0 ? L[a] : h->p[gp].off + L[a];
        if (h->fo[src] >= cap(h, src) || (gp == q && src - P->off == i) || !reach(P, i, h->gpos[src])) continue;
        int dup = 0; for (int s = s0; s < s1; s++) if (P->ssrc[i * P->FI + s] == src) { dup = 1; break; }
        if (!dup) out[(*no)++] = src;
    }
}
/* build the structures of active list L of population gp, version v (once a byte) */
static void build(Net *h, int gp, int v, const int *L, int nl) {
    if (h->bT[gp][v] == h->T) return;
    Pop *S = h->p + gp; int nb = S->n / S->B, *bs = h->bs[gp][v], *f = h->fw[gp][v]; uint8_t *ff = h->ff[gp][v];
    memset(bs, 0, sizeof(int) * (nb + 1));
    for (int a = 0; a < nl; a++) bs[L[a] / S->B + 1]++;
    for (int b = 0; b < nb; b++) bs[b + 1] += bs[b];
    memset(f, 0, sizeof(int) * (nl + 1));
    for (int a = 0; a < nl; a++) { int src = S->off + L[a]; ff[a] = h->fo[src] < cap(h, src); f[a + 1] += ff[a]; int j = (a + 1) + ((a + 1) & -(a + 1)); if (j <= nl) f[j] += f[a + 1]; h->lp_[gp][v][L[a]] = a; h->lst[gp][v][L[a]] = h->T; }
    h->bm[gp][v] = nl; h->bl[gp][v] = L; h->bT[gp][v] = h->T;
}
static inline int passc(const Pop *S, int c, float x, float r) { int cm = ((c % S->n) + S->n) % S->n; return rdist(x, (cm + 0.5f) / S->n) <= r; }
typedef struct { int lo, hi, partial; } Piece;                     /* list positions [lo, hi): a run of whole blocks, or one partial block */
static int cmp_piece(const void *a, const void *b) { return ((const Piece *)a)->lo - ((const Piece *)b)->lo; }
/* exact candidates of population gp for cell i: the same set and order as the scan, found by blocks and the Fenwick tree.
   Returns the count; elements are then read by rank with pick(). */
typedef struct { Piece pc[64]; int npc; int ex[64]; int nex; int avail[64]; int gp, v; } Seg;
static int seg_count(Net *h, Pop *P, int q, int i, int s0, int s1, Seg *g) {
    int gp = g->gp, v = g->v; Pop *S = h->p + gp; int nb = S->n / S->B, B = S->B; const int *bs = h->bs[gp][v], *L = h->bl[gp][v];
    int ilo[16], ihi[16], ni = 0, part[32], npart = 0;
    float xs[8], rs[8]; int nw = 0; xs[nw] = P->pos[i]; rs[nw++] = P->rho; for (int k = 0; k < P->np && nw < 8; k++) { xs[nw] = P->pc[i * P->np + k]; rs[nw++] = P->rho * 0.5f; }
    for (int w = 0; w < nw; w++) {
        float x = xs[w], r = rs[w]; int n = S->n;
        int c0 = (int)floorf((x - r) * n - 0.5f), c1 = (int)ceilf((x + r) * n - 0.5f);
        if (passc(S, c0, x, r)) { while (passc(S, c0 - 1, x, r) && c0 > c1 - n) c0--; } else { while (!passc(S, c0, x, r) && c0 <= c1) c0++; }
        if (passc(S, c1, x, r)) { while (passc(S, c1 + 1, x, r) && c1 < c0 + n - 1) c1++; } else { while (!passc(S, c1, x, r) && c1 >= c0) c1--; }
        if (c1 < c0) continue;
        int ub0 = (int)floor((double)c0 / B), ub1 = (int)floor((double)c1 / B);
        int i0 = (c0 % B == 0 || (c0 < 0 && (-c0) % B == 0)) ? ub0 : ub0 + 1, i1 = ((c1 + 1) % B == 0) ? ub1 : ub1 - 1;
        if (ub0 == ub1 && !(i0 <= i1)) { part[npart++] = ((ub0 % nb) + nb) % nb; continue; }
        if (i0 > ub0) part[npart++] = ((ub0 % nb) + nb) % nb;
        if (i1 < ub1) part[npart++] = ((ub1 % nb) + nb) % nb;
        if (i0 <= i1) {                                             /* whole blocks i0..i1 (unwrapped) -> actual ranges */
            int a0 = ((i0 % nb) + nb) % nb, len = i1 - i0 + 1; if (len >= nb) { ilo[ni] = 0; ihi[ni++] = nb - 1; continue; }
            if (a0 + len <= nb) { ilo[ni] = a0; ihi[ni++] = a0 + len - 1; } else { ilo[ni] = a0; ihi[ni++] = nb - 1; ilo[ni] = 0; ihi[ni++] = a0 + len - 1 - nb; }
        }
    }
    /* merge whole-block ranges */
    for (int a = 1; a < ni; a++) for (int b = a; b > 0 && ilo[b] < ilo[b - 1]; b--) { int t = ilo[b]; ilo[b] = ilo[b - 1]; ilo[b - 1] = t; t = ihi[b]; ihi[b] = ihi[b - 1]; ihi[b - 1] = t; }
    int mlo[16], mhi[16], nm = 0;
    for (int a = 0; a < ni; a++) { if (nm && ilo[a] <= mhi[nm - 1] + 1) { if (ihi[a] > mhi[nm - 1]) mhi[nm - 1] = ihi[a]; } else { mlo[nm] = ilo[a]; mhi[nm++] = ihi[a]; } }
    g->npc = 0;
    for (int a = 0; a < nm; a++) { Piece pc = { bs[mlo[a]], bs[mhi[a] + 1], 0 }; if (pc.hi > pc.lo) g->pc[g->npc++] = pc; }
    for (int a = 0; a < npart; a++) {                               /* partial blocks not inside a whole range, once each */
        int b = part[a], inside = 0, dup = 0;
        for (int m = 0; m < nm; m++) if (b >= mlo[m] && b <= mhi[m]) inside = 1;
        for (int c = 0; c < a; c++) if (part[c] == b) dup = 1;
        if (inside || dup || bs[b + 1] == bs[b]) continue;
        Piece pc = { bs[b], bs[b + 1], 1 }; g->pc[g->npc++] = pc;
    }
    qsort(g->pc, g->npc, sizeof(Piece), cmp_piece);
    /* excluded: self and already connected sources that are in this list, not full, and inside a whole-block piece */
    g->nex = 0; const int *lp = h->lp_[gp][v], *ls = h->lst[gp][v]; const uint8_t *ff = h->ff[gp][v];
    for (int s = s0 - 1; s < s1; s++) {
        int src = s < s0 ? (gp == q ? S->off + i : -1) : P->ssrc[i * P->FI + s];
        if (src < 0 || h->gpop[src] != gp) continue; int c = src - S->off; if (ls[c] != h->T) continue;
        int a = lp[c]; if (!ff[a]) continue;
        int in = 0; for (int k = 0; k < g->npc; k++) if (!g->pc[k].partial && a >= g->pc[k].lo && a < g->pc[k].hi) in = 1;
        if (!in) continue;
        int d = 0; for (int k = 0; k < g->nex; k++) if (g->ex[k] == a) d = 1;
        if (!d && g->nex < 64) g->ex[g->nex++] = a;
    }
    for (int a = 1; a < g->nex; a++) for (int b = a; b > 0 && g->ex[b] < g->ex[b - 1]; b--) { int t = g->ex[b]; g->ex[b] = g->ex[b - 1]; g->ex[b - 1] = t; }
    int tot = 0; const int *f = h->fw[gp][v];
    for (int k = 0; k < g->npc; k++) {
        Piece *pc = g->pc + k; int av;
        if (pc->partial) {                                          /* check each element as the scan does; keep the eligible ones */
            av = 0; for (int a = pc->lo; a < pc->hi; a++) {
                int src = S->off + L[a];
                if (h->fo[src] >= cap(h, src) || (gp == q && src - P->off == i) || !reach(P, i, h->gpos[src])) continue;
                int dup = 0; for (int s = s0; s < s1; s++) if (P->ssrc[i * P->FI + s] == src) { dup = 1; break; }
                if (!dup) av++;
            }
        } else { av = fw_pre(f, pc->hi) - fw_pre(f, pc->lo); for (int e = 0; e < g->nex; e++) if (g->ex[e] >= pc->lo && g->ex[e] < pc->hi) av--; }
        g->avail[k] = av; tot += av;
    }
    return tot;
}
static int seg_pick(Net *h, Pop *P, int q, int i, int s0, int s1, Seg *g, int r) {   /* the r-th candidate (0-based) of this segment */
    int gp = g->gp, v = g->v; Pop *S = h->p + gp; const int *L = h->bl[gp][v], *f = h->fw[gp][v];
    for (int k = 0; k < g->npc; k++) {
        Piece *pc = g->pc + k;
        if (r >= g->avail[k]) { r -= g->avail[k]; continue; }
        if (pc->partial) {
            for (int a = pc->lo; a < pc->hi; a++) {
                int src = S->off + L[a];
                if (h->fo[src] >= cap(h, src) || (gp == q && src - P->off == i) || !reach(P, i, h->gpos[src])) continue;
                int dup = 0; for (int s = s0; s < s1; s++) if (P->ssrc[i * P->FI + s] == src) { dup = 1; break; }
                if (dup) continue;
                if (r-- == 0) return src;
            }
            return -1;
        }
        int base = fw_pre(f, pc->lo), t = 0;
        for (int e = 0; e < g->nex; e++) {
            int x = g->ex[e]; if (x < pc->lo || x >= pc->hi) continue;
            int before = fw_pre(f, x) - base - t; if (r < before) break;
            t++;
        }
        return S->off + L[fw_kth(f, h->bm[gp][v], base + r + t)];
    }
    return -1;
}
static void grow(Net *h, Pop *P, int q, int i, int s0, int s1, int mask, const uint8_t *mark, int need) {
    if (need <= 0) return;
    int nc = 0, ndir = 0; Seg segs[MAXK]; int nseg = 0, segn[MAXK];
    int direct = 1;                                                 /* the scan only for the sensor lines and short lists */
    for (int gp = -1; gp < h->K; gp++) {
        if (!srcpop_ok(mask, gp)) continue;
        int idx = gp < 0 ? h->K : gp; int *L = h->prvl[idx]; int nl = h->nprv[idx], v = 1;
        if (mark == h->cur) { v = 0; if (gp < 0) { L = h->code + (h->lastbyte < 0 ? 0 : h->lastbyte) * h->gi[G_KIN]; nl = h->lastbyte < 0 ? 0 : h->gi[G_KIN]; } else { L = h->p[gp].act; nl = gp < q ? h->p[gp].nact : 0; } }
        if (!nl) continue;
        Pop *S = gp >= 0 ? h->p + gp : NULL;
        if (gp < 0 || nl < 128 || S->n % S->B || S->B * 8 > S->n || !direct || P->np > 7) {
            if (nseg) { direct = 0; }                               /* a scanned list after a tree list: keep order by scanning the rest too */
            if (!direct) break;
            grow_scan(h, P, q, i, s0, s1, gp, L, nl, h->cand, &nc); ndir = nc; continue;
        }
        build(h, gp, v, L, nl);
        Seg *g = segs + nseg; g->gp = gp; g->v = v; segn[nseg] = seg_count(h, P, q, i, s0, s1, g); nc += segn[nseg]; nseg++;
    }
    if (!direct) {                                                  /* mixed order case: fall back to the full scan */
        nc = 0;
        for (int gp = -1; gp < h->K; gp++) {
            if (!srcpop_ok(mask, gp)) continue;
            int idx = gp < 0 ? h->K : gp; int *L = h->prvl[idx]; int nl = h->nprv[idx];
            if (mark == h->cur) { if (gp < 0) { L = h->code + (h->lastbyte < 0 ? 0 : h->lastbyte) * h->gi[G_KIN]; nl = h->lastbyte < 0 ? 0 : h->gi[G_KIN]; } else { L = h->p[gp].act; nl = gp < q ? h->p[gp].nact : 0; } }
            grow_scan(h, P, q, i, s0, s1, gp, L, nl, h->cand, &nc);
        }
    } else if (nseg) {                                              /* the same draws as the partial Fisher-Yates over the full list */
        int pos[64], val[64], np = 0;
        #define GET(r, out) { int _r = (r), _f = 0; for (int _k = 0; _k < np; _k++) if (pos[_k] == _r) { out = val[_k]; _f = 1; } \
            if (!_f) { if (_r < ndir) out = h->cand[_r]; else { int _x = _r - ndir; for (int _s = 0; _s < nseg; _s++) { if (_x < segn[_s]) { out = seg_pick(h, P, q, i, s0, s1, segs + _s, _x); break; } _x -= segn[_s]; } } } }
        #define SET(r, v_) { int _f = 0; for (int _k = 0; _k < np; _k++) if (pos[_k] == (r)) { val[_k] = (v_); _f = 1; } if (!_f) { pos[np] = (r); val[np++] = (v_); } }
        int take = nc > need ? need : nc, outv[64];
        if (nc > need) {
            for (int t = 0; t < need; t++) { int j = t + (int)(xr(h) % (uint64_t)(nc - t)); int a, b; GET(t, a); GET(j, b); SET(t, b); SET(j, a); }
        }
        for (int t = 0; t < take; t++) GET(t, outv[t]);
        for (int t = 0; t < take; t++) h->cand[t] = outv[t];
        nc = take;
        #undef GET
        #undef SET
        goto assign;
    }
    if (nc > need) { for (int t = 0; t < need; t++) { int j = t + (int)(xr(h) % (uint64_t)(nc - t)); int z = h->cand[t]; h->cand[t] = h->cand[j]; h->cand[j] = z; } nc = need; }
assign:
    for (int t = 0; t < nc; t++) {
        int bs = -1; float bw = 2;
        for (int s = s0; s < s1; s++) { int e = i * P->FI + s, u = P->ssrc[e]; if (u >= 0 && mark[u]) continue; float w = u < 0 ? -1 : wef(h, P, e); if (w < bw) { bw = w; bs = s; } }
        if (bs < 0) break;
        int e = i * P->FI + bs, old = P->ssrc[e];
        if (old >= 0) del_out(h, old, q, e);
        P->ssrc[e] = h->cand[t]; add_out(h, h->cand[t], q, e);
        if (h->gi[G_TAG]) { P->sw[e] = 0; P->sl[e] = (float)h->gd[H_WNEW]; P->tl[e] = h->T; } else P->sw[e] = (float)h->gd[H_WNEW];
    }
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
    for (int b0 = 0; b0 < n; b0 += P->B) {
        int b1 = b0 + P->B < n ? b0 + P->B : n;
        for (int k = 0; k < P->k; k++) {
            int best = -1; float bv = 1e-6f;
            for (int i = b0; i < b1; i++) if (!P->y[i] && P->val[i] > bv) { bv = P->val[i]; best = i; }
            if (best < 0) break;
            P->y[best] = 1; P->act[P->nact++] = best;
        }
    }
    for (int a = 0; a < P->nact; a++) { int i = P->act[a]; h->cur[P->off + i] = 1; h->lastf[P->off + i] = h->T; if (learn) P->use[i]++; }
    if (learn) {
        double eta = h->gd[H_ETA] * h->etamul, etam = h->gd[H_ETAM] * h->etamul; int pm = P->pm | (P->in ? (1 << 31) : 0);
        uint8_t blk[4096]; int nb = (n + P->B - 1) / P->B; if (nb > 4096) nb = 4096;
        if (h->gi[G_GATE]) {                                    /* GATE: a block learns only when it is surprised */
            int nf[4096] = {0}, nu[4096] = {0};
            for (int a = 0; a < P->nact; a++) { int i = P->act[a], bb = i / P->B; if (bb >= nb) continue; nf[bb]++; if (!(P->nseg && P->ts[i] && best_seg(P, i) >= sth)) nu[bb]++; }
            for (int bb = 0; bb < nb; bb++) blk[bb] = nf[bb] && nu[bb] >= h->gd[H_GTH] * nf[bb];
        } else memset(blk, 1, nb);
        if (P->ld) for (int k = 0; k < P->nts; k++) {           /* a dendritic event without firing weakens the active distal slots */
            int i = P->tsl[k]; if (P->y[i] || best_seg(P, i) < sth || !blk[i / P->B < nb ? i / P->B : nb - 1]) continue;
            for (int s = P->FP; s < P->FI; s++) { int e = i * P->FI + s, u = P->ssrc[e]; if (u >= 0 && h->prv[u]) dep(h, P, e, etam); }
        }
        for (int a = 0; a < P->nact; a++) {
            int i = P->act[a];
            int strong = P->nseg && P->ts[i] && best_seg(P, i) >= sth;
            if (h->gi[G_TAG] && strong) for (int s = 0; s < P->FI; s++) { size_t e = (size_t)i * P->FI + s; float l = lab(h, P, e); float c = (float)h->gd[H_CAP] * l; P->sw[e] += c; P->sl[e] = l - c; }   /* capture */
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

static void turnover(Net *h) {                                    /* TURN: clear the least-used cells; they learn faster for a while */
    for (int q = 0; q < h->K; q++) { Pop *P = h->p + q; int r = (int)(h->gd[H_TURN] * 0.1 * P->n + 0.5);
        for (int t = 0; t < r; t++) { int bi = -1, bu = 1 << 30;
            for (int i = 0; i < P->n; i++) if (P->use[i] < bu && h->T - P->born[i] > 2000 && !P->y[i]) { bu = P->use[i]; bi = i; }
            if (bi < 0) break;
            for (int s = P->FP; s < P->FI; s++) { int e = bi * P->FI + s, u = P->ssrc[e]; if (u >= 0 && P->ld) { del_out(h, u, q, e); P->ssrc[e] = -1; P->sw[e] = 0; P->sl[e] = 0; } }
            P->h[bi] = 0; P->m[bi] = 0; P->use[bi] = 0; P->born[bi] = h->T; } }
}
static inline double otr(Net *h, int s) { int d = h->T - h->lastf[s]; if (d == 0) return 1.0; if (h->gd[H_OTR] <= 1 || d > 60 * h->gd[H_OTR]) return 0; return pow(1.0 - 1.0 / h->gd[H_OTR], d); }
void lt3t_run(Net *h, const uint8_t *x, int n, int learn, double *stats) {
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
                for (int v = 0; v < V; v++) {                       /* delta rule; presynaptic = readable cells firing one byte back (cur) */
                    if (v != y && h->o[v] <= 0) continue;
                    double e = (v == y) - h->o[v], na = 0;
                    for (int k = 0; k < NO; k++) { int s = h->osrc[v * NO + k]; if (s >= 0) na += otr(h, s); }
                    h->b[v] += eb * e;
                    if (na > 0) for (int k = 0; k < NO; k++) { int s = h->osrc[v * NO + k]; double tr; if (s >= 0 && (tr = otr(h, s)) > 0) h->ow[v * NO + k] += (float)(eo * e * tr / na); }
                }
                int have = 0; for (int k = 0; k < NO; k++) have += h->osrc[y * NO + k] >= 0;
                for (int q = 0; q < h->K && have < NO; q++) {    /* output cell y grows to readable cells that fired one byte back */
                    Pop *P = h->p + q; if (!P->rd) continue;
                    for (int a = 0; a < P->nact && have < NO; a++) {
                        int s = P->off + P->act[a]; if (h->ocnt[s] >= FOO || rdist(h->opos[y], h->gpos[s]) > rhoo) continue;
                        int dup = 0; for (int k = 0; k < NO; k++) if (h->osrc[y * NO + k] == s) dup = 1;
                        if (dup) continue;
                        for (int k = 0; k < NO; k++) if (h->osrc[y * NO + k] < 0) { h->osrc[y * NO + k] = s; h->ow[y * NO + k] = 0; h->ocnt[s]++; have++; break; }
                    }
                }
                if (have >= NO && h->gd[H_OREP] > 0) {             /* orep: the weakest output synapse below OREP gives way to a cell that fired one byte back */
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
        /* this byte: rotate marks (cur -> prv), sensor lines, then the populations in order */
        for (int q = 0; q <= h->K; q++) { int *L = h->prvl[q]; for (int a = 0; a < h->nprv[q]; a++) h->prv[q < h->K ? h->p[q].off + L[a] : L[a]] = 0; }   /* clear two bytes back */
        for (int q = 0; q <= h->K; q++) {                           /* the lists of one byte back */
            if (q < h->K) { Pop *P = h->p + q; memcpy(h->prvl[q], P->act, sizeof(int) * P->nact); h->nprv[q] = P->nact; }
            else { h->nprv[q] = h->lastbyte < 0 ? 0 : KIN; if (h->lastbyte >= 0) memcpy(h->prvl[q], h->code + h->lastbyte * KIN, sizeof(int) * KIN); }
        }
        uint8_t *tmp = h->prv; h->prv = h->cur; h->cur = tmp;       /* prv: one byte back; cur: cleared */
        h->lastbyte = y;
        for (int k = 0; k < KIN; k++) { int s = h->code[y * KIN + k]; if (!h->cur[s]) { h->cur[s] = 1; spread(h, s, 0); } }
        for (int q = 0; q < h->K; q++) {
            pop_step(h, q, learn);
            Pop *P = h->p + q; for (int a = 0; a < P->nact; a++) spread(h, P->off + P->act[a], 0);
        }
        clear_distal(h);                                            /* distal input for the next byte: everything firing now */
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

/* snapshot of the dynamic state (tests with learning off leave the life undisturbed) */
#define SNAPF(X) X(&h->T, 1); X(h->lastf, h->NG); X(h->cur, h->NG); X(h->prv, h->NG); X(h->o, V); X(&h->have, 1); X(&h->rs, 1); X(&h->lastbyte, 1); X(h->nprv, MAXK + 1); \
    for (int q = 0; q <= h->K; q++) X(h->prvl[q], q < h->K ? h->p[q].n : h->gi[G_NL]); \
    for (int q = 0; q < h->K; q++) { Pop *P = h->p + q; X(P->h, P->n); X(P->m, P->n); X(P->y, P->n); X(P->act, P->n); X(&P->nact, 1); \
        X(P->sacc, (size_t)P->n * (P->nseg + 1)); X(P->ts, P->n); X(P->tsl, P->n); X(&P->nts, 1); }
#define SZ(p, k) sz += sizeof(*(p)) * (size_t)(k);
#define SV(p, k) { memcpy(c, p, sizeof(*(p)) * (size_t)(k)); c += sizeof(*(p)) * (size_t)(k); }
#define LD(p, k) { memcpy(p, c, sizeof(*(p)) * (size_t)(k)); c += sizeof(*(p)) * (size_t)(k); }
void lt3t_snapshot(Net *h) { size_t sz = 0; SNAPF(SZ); if (!h->snap) h->snap = malloc(sz); char *c = h->snap; SNAPF(SV); }
void lt3t_restore(Net *h) { char *c = h->snap; SNAPF(LD); for (int q = 0; q < h->K; q++) h->bT[q][0] = h->bT[q][1] = -1; }

/* audit: [0] max tissue fan-in in use, [1] max fan-out (cells), [2] max sensor-line fan-out, [3] max output fan-in, [4] max fan-out to outputs, [5] synapses */
void lt3t_audit(Net *h, long *a) {
    memset(a, 0, sizeof(long) * 6);
    for (int q = 0; q < h->K; q++) { Pop *P = h->p + q;
        for (int i = 0; i < P->n; i++) { int c = 0; for (int s = 0; s < P->FI; s++) c += P->ssrc[i * P->FI + s] >= 0; if (c > a[0]) a[0] = c; a[5] += c; } }
    for (int s = 0; s < h->NG; s++) { long *t = s < h->gi[G_NL] ? &a[2] : &a[1]; if (h->fo[s] > *t) *t = h->fo[s]; if (h->ocnt[s] > a[4]) a[4] = h->ocnt[s]; }
    int NO = h->gi[G_NO]; for (int v = 0; v < V; v++) { int c = 0; for (int k = 0; k < NO; k++) c += h->osrc[v * NO + k] >= 0; if (c > a[3]) a[3] = c; }
}
