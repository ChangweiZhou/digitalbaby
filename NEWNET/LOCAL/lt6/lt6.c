/* lt6.c - a local tissue designed from the owner's definition of a cell (PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/
   COORDINATION_20261117/WHAT_A_NEURON_HAS_20261118.md, sections 7-19, and the rulings of 16-25 Nov 2026). New code; nothing is
   taken from lt/lt2/lt3/lt3o (non-compliant references) except the scoring instrument.

   Requirement (owner)                                      -> how it is met here
   R1 no global connection; neighbourhood fixed in advance  -> every cell has a random coordinate on the unit torus (s.15). Every
                                                               synapse joins cells within the radius of its type (r; r_out for output
                                                               synapses). Feedback cells read output cells only within r_fb.
                                                               lt6_audit counts violations (must be 0).
   R2 cells decide their own connection counts             -> no fan-in or fan-out number anywhere. Active cells offer a synapse to a
                                                               random nearby cell; the cell accepts by its own need (activity below its
                                                               set point; an output cell by its recent error). Every synapse pays a
                                                               price (its strength decays by lambda per byte) and is withdrawn when its
                                                               strength falls below w_min, i.e. when what arrives over it keeps being
                                                               of no use (s.19). The count is the balance of the two.
   R3 cells decide their own activity                       -> no firing share. A cell fires when its potential exceeds its own
                                                               threshold; the threshold follows the cell's activity set point
                                                               (homeostasis, multiplicative and symmetric in the relative deviation); inhibitory synapses learn from the target's activity
                                                               (inhibitory plasticity). Set points are rates drawn per cell.
   R4 each cell its own period                              -> time constant per cell, log-uniform in [tau_lo, tau_hi].
   R5 feedback: the output end is told when it is wrong;    -> the right answer arrives after the attempt, as activity on answer lines
      dedicated cells receive it and pass it on over          at the output end (s.3.3, s.7.1). A feedback site (an answer line, a +
      learned connections; no interior cell sees the answer   cell and a - cell, at each output cell, with its value) compares the
                                                               answer with the output cells of its value within r_fb: + if the value
                                                               came and they were low, - if they were high and it did not come.
                                                               Output cells read the + and - signals of their value near them (a local
                                                               signal, s.9). Relay cells (a + and a - channel) pass the signals on, hop
                                                               by hop, over their own learned synapses (s.16; "1->50->50^2", 18 Nov).
                                                               Tissue cells read relay and feedback cells on a separate part of the
                                                               cell (s.7.1) and use the deviation of what they read from their own
                                                               running average as a third factor on their eligible synapses. No error goes back along a forward connection (s.4.1).
                                                               Only feedback cells read answer lines; lt6_audit checks it.
   R6 no long byte window                                   -> input lines carry the byte at a set of lags made of short passages
                                                               from different places (e.g. 0-7, 12-15, 24-27, 48-51, 96-99, 192-195);
                                                               a cell connected to lines of several passages reads separate pieces of
                                                               the text (s.8). The lag set is a dimension; no cell reads the document.
   R7 each cell's own task                                  -> every tissue cell predicts its own firing from what fired one byte
                                                               earlier (distal part, local rule) and grows distal synapses when it fires
                                                               unpredicted (s.8).
   R8 Dale; outputs >= 0                                    -> excitatory strengths >= 0; inhibitory cells subtract; activities >= 0.
   R9 cost follows the active part                          -> event-driven: only cells reached by active cells are touched; leak,
                                                               traces, thresholds and synapse decay are caught up lazily. A cleanup
                                                               sweep every 2,000 bytes is linear in synapses (amortised).
   Kinds: SEN input lines, ANS answer lines, TIS excitatory tissue, INH inhibitory, OUT output (value-tagged, graded), FBP/FBN feedback
   cells (one site per output cell), RLP/RLN relay cells. An output cell told it should have fired tries tr_out cells (a rate per
   byte) that were active near it in the last byte; its learning step is divided by its own number of active inputs. The number of cells of each kind and the interface sizes (L, KIN, KOUT) are architecture and
   scaling factors, not connection counts. S = micro-steps per byte, H = relay hops per byte.
   Not part of the net: the instrument p = softmax(beta * D_v / KOUT + c_v), D_v = summed drive of the output cells of value v, learned
   by log loss. It is the experimenter's readout and changes nothing in the net. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define V 256
enum { K_SEN, K_ANS, K_TIS, K_INH, K_OUT, K_FBP, K_FBN, K_RLP, K_RLN, NK };
enum { T_EXC, T_INH, T_DIS, T_OUT, T_RLY, T_RD, NT };
enum { I_L, I_KIN, I_KOUT, I_KFB, I_NTIS, I_NINH, I_NRLY, I_S, I_H, I_SEED, I_NI };
enum { D_R, D_ROUT, D_RFB, D_TLO, D_THI, D_RLO, D_RHI, D_TAUR, D_ETH, D_TH0, D_W0, D_WI0, D_D0, D_GD, D_POFF, D_PDIS, D_LAM, D_WMIN,
       D_EH, D_EE, D_EO, D_EB, D_ED, D_EI, D_EQ, D_ER, D_EPS, D_ETAT, D_GACC, D_B0, D_TEL, D_TRY, D_KAP, D_MU, D_ND };

typedef struct { int pre, post; float w, sl, el; int tl, ki, ko; uint8_t ty; } Syn;   /* w fast part, sl slow (consolidated) part */
typedef struct { int *a, n, cap; } Vec;
static void vpush(Vec *v, int x) { if (v->n == v->cap) { v->cap = v->cap ? 2 * v->cap : 8; v->a = realloc(v->a, sizeof(int) * v->cap); } v->a[v->n++] = x; }
static void ensure(Vec *v, int n) { if (v->cap < n) { v->cap = n; v->a = realloc(v->a, sizeof(int) * n); } }

typedef struct {
    int gi[I_NI]; double gd[D_ND];
    int nk[NK], off[NK], NC, G, nbins, nsite;
    uint8_t *kind; float *px, *py, *ld, *lel, *rho; int *val, *bin;
    float *h, *th, *xl, *rt, *acc, *m, *zc, *b, *pd, *eav, *sig; int *nin;
    int *tt, *xs, *rts, *fired, *mark, *mst, *pdst, *sigst, *zst;
    Vec *in[NT], *out[NT];
    Syn *S; int ns, cap; int *fre, nfre;
    Vec *bins[NK];                                   /* static: cells of each kind per grid bin */
    Vec *pbin; Vec pbt;                              /* tissue cells that fired in the last byte, per bin; and the bins used */
    Vec fired_now, fired_prev, inh_now, sen_now, out_touch, out_prev, cur, nxt, tl, sl, ml;
    int *so0, *so, *os0, *os;                        /* static: output cells near each feedback site, and sites near each output */
    int *hist, *lag, HL, head; int T, have; uint64_t rs;   /* input history ring (HL bytes) and the lag of each input-line group */
    double Ds[V], beta, c[V];
    float lnl; long st_made, st_cut, st_offers, st_viol, st_fall; int norelay, biasonly;
    void *snap; size_t snapsz;
} Net;

static inline uint64_t xr(Net *h) { h->rs ^= h->rs << 13; h->rs ^= h->rs >> 7; h->rs ^= h->rs << 17; return h->rs; }
static inline double ur(Net *h) { return (xr(h) >> 11) * (1.0 / 9007199254740992.0); }
static inline float tdist2(Net *h, int i, int j) { float dx = fabsf(h->px[i] - h->px[j]), dy = fabsf(h->py[i] - h->py[j]); if (dx > 0.5f) dx = 1 - dx; if (dy > 0.5f) dy = 1 - dy; return dx * dx + dy * dy; }
static inline float radius(Net *h, int ty) { return (float)(ty == T_OUT ? h->gd[D_ROUT] : h->gd[D_R]); }
static inline float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ---- lazy per-cell state ---- */
static inline float trace(Net *h, int j) { int d = h->T - h->xs[j]; return d <= 0 ? h->xl[j] : h->xl[j] * expf(h->lel[j] * d); }
static inline float rate(Net *h, int i) { int d = h->T - h->rts[i]; return d <= 0 ? h->rt[i] : h->rt[i] * expf(-(float)d / (float)h->gd[D_TAUR]); }
static void catch_up(Net *h, int i, int learn) {          /* leak and homeostatic threshold from the last update to now */
    int d = h->T - h->tt[i]; if (d <= 0) return;
    h->h[i] *= expf(h->ld[i] * d);
    if (learn) { float q = (rate(h, i) - h->rho[i]) / h->rho[i]; q = q > 1 ? 1 : (q < -1 ? -1 : q);   /* relative, symmetric */
                 float t = h->th[i] * expf((float)h->gd[D_ETH] * q * d); h->th[i] = t < 0.02f ? 0.02f : (t > 20 ? 20 : t); }
    h->tt[i] = h->T;
}
static void fire_mark(Net *h, int i, int learn) {
    h->fired[i] = h->T; h->xl[i] = trace(h, i) + 1; h->xs[i] = h->T;
    if (learn) { h->rt[i] = rate(h, i) + 1.0f / (float)h->gd[D_TAUR]; h->rts[i] = h->T; }
}
static inline void decay2(Net *h, Syn *s, int d, float *f, float *sl) {     /* exact two-stage decay over d bytes */
    float a = (float)(h->gd[D_LAM] + h->gd[D_KAP]), mu = (float)h->gd[D_MU], ea = expf(-a * d), em = expf(-mu * d);
    *f = s->w * ea; *sl = s->sl * em + (fabsf(a - mu) > 1e-12f ? s->w * (float)h->gd[D_KAP] / (a - mu) * (em - ea) : 0); }
static inline float weff(Net *h, Syn *s) { int d = h->T - s->tl; if (d <= 0) return s->w + s->sl; float f, g; decay2(h, s, d, &f, &g); return f + g; }
static inline float need(Net *h, int i) { float q = (h->rho[i] - rate(h, i)) / h->rho[i]; return q < 0 ? 0 : (q > 1 ? 1 : q); }
static inline float excess(Net *h, int i) { float q = (rate(h, i) - h->rho[i]) / h->rho[i]; return q < 0 ? 0 : (q > 1 ? 1 : q); }

/* ---- synapses ---- */
static void syn_cut(Net *h, int id) {
    Syn *s = h->S + id; Vec *vi = &h->in[s->ty][s->post], *vo = &h->out[s->ty][s->pre];
    int a = vi->a[--vi->n]; if (s->ki != vi->n) { vi->a[s->ki] = a; h->S[a].ki = s->ki; }
    int b = vo->a[--vo->n]; if (s->ko != vo->n) { vo->a[s->ko] = b; h->S[b].ko = s->ko; }
    s->pre = -1; h->fre[h->nfre++] = id; h->st_cut++;
}
static int connected(Net *h, int pre, int post, int ty) { Vec *v = &h->in[ty][post]; for (int k = 0; k < v->n; k++) if (h->S[v->a[k]].pre == pre) return 1; return 0; }
static void syn_make(Net *h, int pre, int post, int ty, float w) {
    float r = radius(h, ty); if (tdist2(h, pre, post) > r * r) { h->st_viol++; return; }
    if (pre == post || connected(h, pre, post, ty)) return;
    int id; if (h->nfre) id = h->fre[--h->nfre]; else { if (h->ns == h->cap) { h->cap = h->cap ? 2 * h->cap : 4096; h->S = realloc(h->S, sizeof(Syn) * h->cap); h->fre = realloc(h->fre, sizeof(int) * h->cap); } id = h->ns++; }
    Syn *s = h->S + id; s->pre = pre; s->post = post; s->w = w; s->sl = 0; s->el = 0; s->tl = h->T; s->ty = ty;
    s->ki = h->in[ty][post].n; vpush(&h->in[ty][post], id); s->ko = h->out[ty][pre].n; vpush(&h->out[ty][pre], id); h->st_made++;
}
static int syn_touch(Net *h, int id) {                     /* bring to now; withdraw if of no use. 0 = withdrawn */
    Syn *s = h->S + id; int d = h->T - s->tl; if (d > 0) { float f, g; decay2(h, s, d, &f, &g); s->w = f; s->sl = g; } s->tl = h->T;
    if (s->w + s->sl < h->gd[D_WMIN]) { syn_cut(h, id); return 0; } return 1;
}
static inline float elig(Net *h, Syn *s) { int d = h->T - s->tl; return d > 0 ? s->el * expf(h->lel[s->post] * d) : s->el; }

/* a random grid bin among those that can hold cells within radius r of cell j (bins are 1/G wide) */
static int rand_bin(Net *h, int j, float r) {
    int G = h->G, sp = (int)ceilf(r * G); if (2 * sp + 1 >= G) return (int)(xr(h) % (uint64_t)(G * G));
    int bx = h->bin[j] % G, by = h->bin[j] / G, dx = (int)(xr(h) % (uint64_t)(2 * sp + 1)) - sp, dy = (int)(xr(h) % (uint64_t)(2 * sp + 1)) - sp;
    return ((by + dy + G) % G) * G + (bx + dx + G) % G;
}
/* one trial: a random cell of kind k within radius r of cell j, or -1 */
static int near_cell(Net *h, int j, int k, float r) {
    int nb = rand_bin(h, j, r);
    Vec *v = &h->bins[k][nb]; if (!v->n) return -1;
    int c = v->a[xr(h) % (uint64_t)v->n]; if (c == j || tdist2(h, c, j) > r * r) return -1; return c;
}

Net *lt6_new(const int *gi, const double *gd, const int *lags) {
    Net *h = calloc(1, sizeof(Net)); memcpy(h->gi, gi, sizeof(int) * I_NI); memcpy(h->gd, gd, sizeof(double) * D_ND);
    h->rs = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)gi[I_SEED] * 2654435761ULL); for (int i = 0; i < 20; i++) xr(h);
    if (h->gi[I_H] > 6) h->gi[I_H] = 6;
    int L = gi[I_L], KIN = gi[I_KIN], KOUT = gi[I_KOUT], KFB = gi[I_KFB];
    (void)KFB; h->nk[K_SEN] = L * V * KIN; h->nk[K_ANS] = h->nk[K_FBP] = h->nk[K_FBN] = h->nsite = V * KOUT; h->nk[K_TIS] = gi[I_NTIS]; h->nk[K_INH] = gi[I_NINH];
    h->nk[K_OUT] = V * KOUT; h->nk[K_RLP] = h->nk[K_RLN] = gi[I_NRLY];
    for (int k = 0; k < NK; k++) { h->off[k] = h->NC; h->NC += h->nk[k]; }
    int N = h->NC;
    h->G = (int)floor(1.0 / gd[D_R]);                 /* bins as wide as the tissue radius; larger radii span more bins */ if (h->G < 1) h->G = 1; h->nbins = h->G * h->G;
#define AL(p, t) p = calloc(N, sizeof(t))
    AL(h->kind, uint8_t); AL(h->px, float); AL(h->py, float); AL(h->ld, float); AL(h->lel, float); AL(h->rho, float); AL(h->val, int); AL(h->bin, int);
    AL(h->h, float); AL(h->th, float); AL(h->xl, float); AL(h->rt, float); AL(h->acc, float); AL(h->m, float); AL(h->zc, float); AL(h->b, float);
    AL(h->pd, float); AL(h->eav, float); AL(h->sig, float); AL(h->nin, int);
    AL(h->tt, int); AL(h->xs, int); AL(h->rts, int); AL(h->fired, int); AL(h->mark, int); AL(h->mst, int); AL(h->pdst, int); AL(h->sigst, int); AL(h->zst, int);
    for (int t = 0; t < NT; t++) { h->in[t] = calloc(N, sizeof(Vec)); h->out[t] = calloc(N, sizeof(Vec)); }
    for (int k = 0; k < NK; k++) h->bins[k] = calloc(h->nbins, sizeof(Vec));
    h->pbin = calloc(h->nbins, sizeof(Vec)); h->lnl = logf(1.0f - (float)gd[D_LAM]);
    for (int k = 0; k < NK; k++) for (int q = 0; q < h->nk[k]; q++) {
        int i = h->off[k] + q; h->kind[i] = k; h->px[i] = (float)ur(h); h->py[i] = (float)ur(h);
        double tau = exp(log(gd[D_TLO]) + ur(h) * (log(gd[D_THI]) - log(gd[D_TLO])));
        h->ld[i] = tau <= 1.0 ? -30.f : (float)log(1.0 - 1.0 / tau);
        double tel = tau > gd[D_TEL] ? tau : gd[D_TEL]; h->lel[i] = tel <= 1.0 ? -30.f : (float)log(1.0 - 1.0 / tel);
        h->rho[i] = (float)exp(log(gd[D_RLO]) + ur(h) * (log(gd[D_RHI]) - log(gd[D_RLO])));
        h->th[i] = (float)gd[D_TH0];
        h->fired[i] = h->mark[i] = h->mst[i] = h->pdst[i] = h->sigst[i] = h->zst[i] = -1000000;
        if (k == K_SEN) h->val[i] = q / KIN;
        else if (k == K_ANS || k == K_FBP || k == K_FBN) h->val[i] = q / KOUT;
        else if (k == K_OUT) { h->val[i] = q / KOUT; h->b[i] = (float)gd[D_B0]; }
        else h->val[i] = -1;
    }
    for (int q = 0; q < h->nsite; q++) { int o = h->off[K_OUT] + q;       /* one feedback site at each output cell */
        for (int k = K_ANS; k <= K_FBN; k++) { if (k == K_OUT || k == K_TIS || k == K_INH) continue; h->px[h->off[k] + q] = h->px[o]; h->py[h->off[k] + q] = h->py[o]; } }
    for (int i = 0; i < N; i++) { int bx = (int)(h->px[i] * h->G) % h->G, by = (int)(h->py[i] * h->G) % h->G; h->bin[i] = by * h->G + bx; vpush(&h->bins[h->kind[i]][h->bin[i]], i); }
    float rf2 = (float)(gd[D_RFB] * gd[D_RFB]); Vec *a1 = calloc(h->nsite, sizeof(Vec)), *a2 = calloc(h->nk[K_OUT], sizeof(Vec));
    for (int q = 0; q < h->nsite; q++) { int a = h->off[K_ANS] + q;
        for (int u = 0; u < KOUT; u++) { int o = h->off[K_OUT] + h->val[a] * KOUT + u; if (tdist2(h, a, o) <= rf2) { vpush(&a1[q], o); vpush(&a2[o - h->off[K_OUT]], q); } } }
    h->so0 = malloc(sizeof(int) * (h->nsite + 1)); h->os0 = malloc(sizeof(int) * (h->nk[K_OUT] + 1)); int t1 = 0, t2 = 0;
    for (int q = 0; q < h->nsite; q++) { h->so0[q] = t1; t1 += a1[q].n; } h->so0[h->nsite] = t1;
    for (int o = 0; o < h->nk[K_OUT]; o++) { h->os0[o] = t2; t2 += a2[o].n; } h->os0[h->nk[K_OUT]] = t2;
    h->so = malloc(sizeof(int) * (t1 + 1)); h->os = malloc(sizeof(int) * (t2 + 1));
    for (int q = 0; q < h->nsite; q++) { memcpy(h->so + h->so0[q], a1[q].a, sizeof(int) * a1[q].n); free(a1[q].a); }
    for (int o = 0; o < h->nk[K_OUT]; o++) { memcpy(h->os + h->os0[o], a2[o].a, sizeof(int) * a2[o].n); free(a2[o].a); }
    free(a1); free(a2);
    h->lag = malloc(sizeof(int) * L); h->HL = 1; for (int l = 0; l < L; l++) { h->lag[l] = lags[l]; if (lags[l] + 1 > h->HL) h->HL = lags[l] + 1; }
    h->hist = malloc(sizeof(int) * h->HL); for (int l = 0; l < h->HL; l++) h->hist[l] = -1; h->head = 0;
    for (int v = 0; v < V; v++) h->Ds[v] = 0;
    for (int q = 0; q < h->nk[K_OUT]; q++) h->Ds[h->val[h->off[K_OUT] + q]] += h->b[h->off[K_OUT] + q];
    h->beta = 1.0;
    return h;
}

static inline float zout(Net *h, int o) { return h->b[o] + (h->zst[o] == h->T - 1 ? h->zc[o] : 0.0f); }   /* drive of the last byte */

/* ---- feedback for the prediction made from the last byte, its relay, and the learning it drives ---- */
static void feedback(Net *h, int y) {
    int KFB = h->gi[I_KFB]; float eps = (float)h->gd[D_EPS];
    h->cur.n = 0; h->sl.n = 0; h->ml.n = 0;
    for (int u = 0; u < h->gi[I_KOUT]; u++) vpush(&h->cur, y * h->gi[I_KOUT] + u);   /* sites of the answer's value */
    for (int k = 0; k < h->out_touch.n; k++) { int o = h->out_touch.a[k] - h->off[K_OUT]; for (int j = h->os0[o]; j < h->os0[o + 1]; j++) vpush(&h->cur, h->os[j]); }
    for (int k = 0; k < h->cur.n; k++) {                                        /* each site compares the answer with its outputs */
        int q = h->cur.a[k], p = h->off[K_FBP] + q, n = h->off[K_FBN] + q; if (h->sigst[p] == h->T) continue;
        float ab = 0; int j0 = h->so0[q], j1 = h->so0[q + 1];
        for (int j = j0; j < j1; j++) ab += sigm(zout(h, h->so[j]));
        ab = j1 > j0 ? ab / (j1 - j0) : 0;
        float ans = h->val[h->off[K_ANS] + q] == y ? 1.0f : 0.0f;
        h->sig[p] = ans > ab ? ans - ab : 0; h->sig[n] = ab > ans ? ab - ans : 0; h->sigst[p] = h->sigst[n] = h->T;
        if (h->sig[p] > eps) vpush(&h->sl, p); if (h->sig[n] > eps) vpush(&h->sl, n);
    }
    /* output cells read the + and - signals of their value near them, and learn from what drove them in the last byte */
    h->nxt.n = 0;
    for (int k = 0; k < h->sl.n; k++) { int f = h->sl.a[k], q = f - (h->kind[f] == K_FBP ? h->off[K_FBP] : h->off[K_FBN]);
        for (int j = h->so0[q]; j < h->so0[q + 1]; j++) { int o = h->so[j]; if (h->mst[o] != h->T) { h->mst[o] = h->T; h->m[o] = 0; vpush(&h->nxt, o); }
            h->m[o] += h->kind[f] == K_FBP ? h->sig[f] : -h->sig[f]; } }
    for (int k = 0; k < h->nxt.n; k++) {
        int o = h->nxt.a[k], oq = o - h->off[K_OUT], ns = h->os0[oq + 1] - h->os0[oq]; float e = h->m[o] / (float)(ns > 0 ? ns : 1);
        h->eav[o] = 0.98f * h->eav[o] + 0.02f * fabsf(e);
        float db = (float)h->gd[D_EB] * e; h->b[o] += db; h->Ds[h->val[o]] += db;
        Vec *v = &h->in[T_OUT][o]; int na = 0;
        for (int u = 0; u < v->n; u++) na += h->fired[h->S[v->a[u]].pre] == h->T - 1;
        float step = (float)h->gd[D_EO] * e / (float)(na + 1);   /* normalized by the cell's own number of active inputs */
        for (int u = v->n - 1; u >= 0; u--) { int id = v->a[u]; if (h->fired[h->S[id].pre] != h->T - 1) continue; if (!syn_touch(h, id)) continue;
            Syn *s = h->S + id; s->w += step; if (s->w + s->sl < 0) s->w = -s->sl; }
        /* an output cell told it should have fired tries a nearby cell that was active in the last byte (the receiver chooses) */
        if (e > eps) { float ro = (float)h->gd[D_ROUT]; double tr = h->gd[D_TRY]; int nt = (int)tr + (ur(h) < tr - (int)tr);   /* trials per byte: a rate */
            for (int t = 0; t < nt; t++) { Vec *pv = &h->pbin[rand_bin(h, o, ro)]; if (!pv->n) continue; int j = pv->a[xr(h) % (uint64_t)pv->n];
                if (tdist2(h, j, o) <= ro * ro) syn_make(h, j, o, T_OUT, (float)(2 * h->gd[D_WMIN])); } }
    }
    /* relay: + and - channels, hop by hop; tissue cells read relay sources on their separate part */
    float r2 = (float)h->gd[D_R];
    if (h->norelay) return;
    for (int hop = 0; ; hop++) {
        h->nxt.n = 0;
        for (int k = 0; k < h->sl.n; k++) {
            int f = h->sl.a[k], sg = (h->kind[f] == K_FBP || h->kind[f] == K_RLP) ? 1 : -1; float sf = h->sig[f];
            Vec *rd = &h->out[T_RD][f];
            for (int u = rd->n - 1; u >= 0; u--) { int id = rd->a[u]; if (!syn_touch(h, id)) continue; Syn *s = h->S + id; int i = s->post;
                float xi = trace(h, i); s->w += (float)h->gd[D_EQ] * sf * xi * (1 - s->w);
                if (h->mst[i] != h->T) { h->mst[i] = h->T; h->m[i] = 0; vpush(&h->ml, i); }
                h->m[i] += sg * s->w * sf; }
            if (hop < h->gi[I_H]) {
                Vec *ry = &h->out[T_RLY][f];
                for (int u = ry->n - 1; u >= 0; u--) { int id = ry->a[u]; if (!syn_touch(h, id)) continue; Syn *s = h->S + id; int r = s->post;
                    if (h->mark[r] != h->T * 8 + hop) { h->mark[r] = h->T * 8 + hop; h->acc[r] = 0; vpush(&h->nxt, r); }
                    h->acc[r] += s->w * sf; }
            }
            if (ur(h) < h->gd[D_POFF]) { int r = near_cell(h, f, sg > 0 ? K_RLP : K_RLN, r2); if (r >= 0 && ur(h) < need(h, r)) syn_make(h, f, r, T_RLY, (float)h->gd[D_W0]); }
            if (ur(h) < h->gd[D_POFF]) { int i = near_cell(h, f, K_TIS, r2); if (i >= 0 && ur(h) < need(h, i) * (trace(h, i) > 1 ? 1 : trace(h, i))) syn_make(h, f, i, T_RD, (float)h->gd[D_W0]); }
        }
        if (hop >= h->gi[I_H]) break;
        h->sl.n = 0;
        for (int k = 0; k < h->nxt.n; k++) { int r = h->nxt.a[k]; float a = h->acc[r] > 1 ? 1 : h->acc[r]; h->mark[r] = -1;
            if (a <= eps) continue;
            h->sig[r] = a; h->sigst[r] = h->T; vpush(&h->sl, r); fire_mark(h, r, 1);
            Vec *v = &h->in[T_RLY][r];
            for (int u = v->n - 1; u >= 0; u--) { int id = v->a[u]; int pre = h->S[id].pre; if (h->sigst[pre] != h->T) continue; if (!syn_touch(h, id)) continue;
                Syn *s = h->S + id; s->w += (float)h->gd[D_ER] * h->sig[pre] * a * (1 - s->w); } }
        if (!h->sl.n) break;
    }
    /* tissue cells: what they read is a third factor on their eligible excitatory synapses */
    for (int k = 0; k < h->ml.n; k++) {
        int i = h->ml.a[k]; float mi = h->m[i] - h->eav[i]; h->eav[i] += 0.01f * (h->m[i] - h->eav[i]);   /* deviation from what it usually reads */
        if (fabsf(mi) < eps) continue;
        Vec *v = &h->in[T_EXC][i];
        for (int u = v->n - 1; u >= 0; u--) { int id = v->a[u]; Syn *s = h->S + id; float el = elig(h, s); if (el < 1e-3f) continue;
            if (!syn_touch(h, id)) continue; s = h->S + id; s->el = el; s->w += (float)h->gd[D_EE] * mi * el; if (s->w + s->sl < 0) s->w = -s->sl; if (s->w + s->sl > 1) s->w = 1 - s->sl; }
    }
}

/* ---- forward pass of one byte ---- */
static void spread(Net *h, int j, int ty) {
    Vec *v = &h->out[ty][j];
    for (int u = 0; u < v->n; u++) { Syn *s = h->S + v->a[u]; int i = s->post; float w = weff(h, s);
        if (h->mark[i] != h->T * 8 + 7) { h->mark[i] = h->T * 8 + 7; h->acc[i] = 0; h->nin[i] = 0; vpush(&h->tl, i); }
        h->acc[i] += ty == T_INH ? -w : w; if (ty == T_OUT) h->nin[i]++; }
}

static void forward(Net *h, int y, int learn) {
    int L = h->gi[I_L], KIN = h->gi[I_KIN];
    h->head = (h->head + 1) % h->HL; h->hist[h->head] = y;
    Vec t = h->fired_prev; h->fired_prev = h->fired_now; h->fired_now = t; h->fired_now.n = 0; h->inh_now.n = 0; h->sen_now.n = 0;
    t = h->out_prev; h->out_prev = h->out_touch; h->out_touch = t; h->out_touch.n = 0;
    for (int l = 0; l < L; l++) { int b = h->hist[(h->head - h->lag[l] + h->HL) % h->HL]; if (b < 0) continue;
        for (int k = 0; k < KIN; k++) { int s = h->off[K_SEN] + (l * V + b) * KIN + k; vpush(&h->sen_now, s); h->fired[s] = h->T; } }
    for (int k = 0; k < h->fired_prev.n; k++) { int j = h->fired_prev.a[k]; Vec *v = &h->out[T_DIS][j];      /* distal prediction */
        for (int u = 0; u < v->n; u++) { Syn *s = h->S + v->a[u]; int i = s->post; if (h->pdst[i] != h->T) { h->pdst[i] = h->T; h->pd[i] = 0; } h->pd[i] += weff(h, s); } }
    h->cur.n = 0;
    for (int st = 0; st < h->gi[I_S]; st++) {                                    /* micro-steps */
        h->tl.n = 0;
        if (st == 0) { for (int k = 0; k < h->sen_now.n; k++) spread(h, h->sen_now.a[k], T_EXC);
                       for (int k = 0; k < h->fired_prev.n; k++) spread(h, h->fired_prev.a[k], T_EXC); }        /* recurrence from the last byte */
        else for (int k = 0; k < h->cur.n; k++) { int j = h->cur.a[k]; spread(h, j, h->kind[j] == K_INH ? T_INH : T_EXC); }
        h->nxt.n = 0;
        for (int k = 0; k < h->tl.n; k++) {
            int i = h->tl.a[k]; h->mark[i] = -1; if (h->fired[i] == h->T) continue;
            catch_up(h, i, learn);
            float d = h->acc[i]; if (st == 0 && h->kind[i] == K_TIS && h->pdst[i] == h->T) d += (float)h->gd[D_GD] * (h->pd[i] > 1 ? 1 : h->pd[i]);
            h->h[i] += d;
            if (h->h[i] > h->th[i]) { fire_mark(h, i, learn); h->h[i] = 0; vpush(&h->nxt, i); vpush(h->kind[i] == K_TIS ? &h->fired_now : &h->inh_now, i); }
        }
        Vec tv = h->cur; h->cur = h->nxt; h->nxt = tv;
        if (!h->cur.n) break;
    }
    /* output cells: drive from the tissue that fired this byte and from the input lines */
    for (int k = 0; k < h->out_prev.n; k++) { int o = h->out_prev.a[k]; h->Ds[h->val[o]] -= h->zc[o]; h->zc[o] = 0; }
    h->tl.n = 0;
    for (int k = 0; k < h->fired_now.n; k++) spread(h, h->fired_now.a[k], T_OUT);
    for (int k = 0; k < h->sen_now.n; k++) spread(h, h->sen_now.a[k], T_OUT);
    for (int k = 0; k < h->tl.n; k++) { int o = h->tl.a[k]; h->mark[o] = -1; float z = h->acc[o] / (1.0f + h->nin[o]);   /* feedforward inhibition */
        h->zc[o] = z; h->zst[o] = h->T; h->Ds[h->val[o]] += z; vpush(&h->out_touch, o); }
    if (!learn) return;
    float eh = (float)h->gd[D_EH], ed = (float)h->gd[D_ED], ei = (float)h->gd[D_EI], r2 = (float)h->gd[D_R];
    for (int k = 0; k < h->fired_now.n; k++) {                                   /* eligibility, own feature rule, inhibitory plasticity */
        int i = h->fired_now.a[k]; Vec *v = &h->in[T_EXC][i];
        for (int u = v->n - 1; u >= 0; u--) { int id = v->a[u]; float el = elig(h, h->S + id); if (!syn_touch(h, id)) continue; Syn *s = h->S + id;
            float x = h->fired[s->pre] >= h->T - 1 ? 1.0f : 0.0f; s->el = el + x; s->w += eh * x * (1 - s->w - s->sl); }   /* co-active inputs strengthen; the price withdraws the rest */
        v = &h->in[T_INH][i]; float rr = rate(h, i);
        for (int u = v->n - 1; u >= 0; u--) { int id = v->a[u]; if (!syn_touch(h, id)) continue; Syn *s = h->S + id;
            s->w += ei * trace(h, s->pre) * (rr - h->rho[i]) / h->rho[i]; if (s->w < 0) s->w = 0; }
    }
    for (int k = 0; k < h->fired_now.n; k++) {                                   /* own task: predict one's own firing */
        int i = h->fired_now.a[k]; float p = h->pdst[i] == h->T ? (h->pd[i] > 1 ? 1 : h->pd[i]) : 0;
        Vec *v = &h->in[T_DIS][i];
        for (int u = v->n - 1; u >= 0; u--) { int id = v->a[u]; if (h->fired[h->S[id].pre] != h->T - 1) continue; if (!syn_touch(h, id)) continue; Syn *s = h->S + id;
            s->w += ed * (1 - p); if (s->w > 1) s->w = 1; }
        if (p < 0.5f && ur(h) < h->gd[D_PDIS]) {
            Vec *pv = &h->pbin[rand_bin(h, i, r2)]; if (pv->n) { int j = pv->a[xr(h) % (uint64_t)pv->n]; if (j != i && h->kind[j] == K_TIS && tdist2(h, i, j) <= r2 * r2) syn_make(h, j, i, T_DIS, (float)h->gd[D_D0]); } }
    }
    for (int k = 0; k < h->fired_prev.n; k++) { Vec *v = &h->out[T_DIS][h->fired_prev.a[k]];          /* predicted, did not fire */
        for (int u = v->n - 1; u >= 0; u--) { int id = v->a[u]; int i = h->S[id].post; if (h->fired[i] == h->T) continue; float p = h->pd[i] > 1 ? 1 : h->pd[i];
            if (!syn_touch(h, id)) continue; Syn *s = h->S + id; s->w -= ed * p; if (s->w < 0) s->w = 0; } }
    for (int pass = 0; pass < 3; pass++) {                                       /* offers; each cell accepts by its own need */
        Vec *src = pass == 0 ? &h->sen_now : (pass == 1 ? &h->fired_now : &h->inh_now);
        for (int k = 0; k < src->n; k++) { int j = src->a[k]; if (ur(h) >= h->gd[D_POFF]) continue; h->st_offers++;
            if (pass == 2) { int i = near_cell(h, j, K_TIS, r2); if (i >= 0 && ur(h) < excess(h, i)) syn_make(h, j, i, T_INH, (float)h->gd[D_WI0]); continue; }
            int roll = (int)(xr(h) % 3);
            if (roll == 0) { int i = near_cell(h, j, K_TIS, r2); if (i >= 0 && ur(h) < need(h, i)) syn_make(h, j, i, T_EXC, (float)h->gd[D_W0]); }
            else if (roll == 1) { int o = near_cell(h, j, K_OUT, (float)h->gd[D_ROUT]); if (o >= 0 && ur(h) < h->gd[D_GACC] * h->eav[o]) syn_make(h, j, o, T_OUT, (float)(3 * h->gd[D_WMIN])); }
            else if (pass == 1) { int i = near_cell(h, j, K_INH, r2); if (i >= 0 && ur(h) < need(h, i)) syn_make(h, j, i, T_EXC, (float)h->gd[D_W0]); } } }
    for (int k = 0; k < h->pbt.n; k++) h->pbin[h->pbt.a[k]].n = 0; h->pbt.n = 0;     /* where the tissue fired, for distal growth */
    for (int k = 0; k < h->fired_now.n; k++) { int i = h->fired_now.a[k], bb = h->bin[i]; if (!h->pbin[bb].n) vpush(&h->pbt, bb); vpush(&h->pbin[bb], i); }
    for (int k = 0; k < h->sen_now.n; k++) { int i = h->sen_now.a[k], bb = h->bin[i]; if (!h->pbin[bb].n) vpush(&h->pbt, bb); vpush(&h->pbin[bb], i); }
}

void lt6_run_trace(Net *h, const uint8_t *x, int n, int learn, double *stats, float *per);
void lt6_run(Net *h, const uint8_t *x, int n, int learn, double *stats) { lt6_run_trace(h, x, n, learn, stats, 0); }
/* per (optional): the loss of each byte in nats (-1 where there was no prediction yet) */
void lt6_run_trace(Net *h, const uint8_t *x, int n, int learn, double *stats, float *per) {
    double p[V]; int KOUT = h->gi[I_KOUT];
    for (int t = 0; t < n; t++) {
        int y = x[t];
        if (h->have) {
            double mx = -1e300; int arg = 0;
            for (int v = 0; v < V; v++) { double dv = h->Ds[v]; if (h->biasonly) { dv = 0; for (int u = 0; u < KOUT; u++) dv += h->b[h->off[K_OUT] + v * KOUT + u]; }
                p[v] = h->beta * dv / KOUT + h->c[v]; if (p[v] > mx) { mx = p[v]; arg = v; } }
            double Z = 0; for (int v = 0; v < V; v++) { p[v] = exp(p[v] - mx); Z += p[v]; } for (int v = 0; v < V; v++) p[v] /= Z;
            stats[0] += -log(p[y] + 1e-12); stats[1] += arg == y; stats[2] += 1; if (per) per[t] = (float)-log(p[y] + 1e-12);
            if (learn) { double et = h->gd[D_ETAT], gb = 0; for (int v = 0; v < V; v++) { double g = (v == y) - p[v]; gb += g * h->Ds[v] / KOUT; } h->beta += et * gb; }
        } else if (per) per[t] = -1;
        h->T++;
        if (learn && h->have) feedback(h, y);
        forward(h, y, learn); h->have = 1;
    }
}

#define SNAPF(X) X(h->h, h->NC); X(h->th, h->NC); X(h->tt, h->NC); X(h->xl, h->NC); X(h->xs, h->NC); X(h->fired, h->NC); X(h->zc, h->NC); \
    X(h->zst, h->NC); X(h->pd, h->NC); X(h->pdst, h->NC); X(h->mark, h->NC); X(h->hist, h->HL); X(&h->head, 1); X(&h->T, 1); X(&h->have, 1); X(&h->rs, 1); X(h->Ds, V);
#define SZ(p, k) sz += sizeof(*(p)) * (size_t)(k);
#define SV(p, k) { memcpy(c, p, sizeof(*(p)) * (size_t)(k)); c += sizeof(*(p)) * (size_t)(k); }
#define LD(p, k) { memcpy(p, c, sizeof(*(p)) * (size_t)(k)); c += sizeof(*(p)) * (size_t)(k); }
#define VSZ(v) sz += sizeof(int) * ((size_t)(v).n + 1);
#define VSV(v) { memcpy(c, &(v).n, sizeof(int)); c += sizeof(int); memcpy(c, (v).a, sizeof(int) * (v).n); c += sizeof(int) * (v).n; }
#define VLD(v) { memcpy(&(v).n, c, sizeof(int)); c += sizeof(int); ensure(&(v), (v).n + 1); memcpy((v).a, c, sizeof(int) * (v).n); c += sizeof(int) * (v).n; }
#define VECS(X) X(h->fired_prev); X(h->fired_now); X(h->inh_now); X(h->sen_now); X(h->out_touch); X(h->out_prev)
void lt6_snapshot(Net *h) { size_t sz = 0; SNAPF(SZ); VECS(VSZ); if (sz > h->snapsz) { h->snap = realloc(h->snap, sz); h->snapsz = sz; } char *c = h->snap; SNAPF(SV); VECS(VSV); }
void lt6_restore(Net *h) { char *c = h->snap; SNAPF(LD); VECS(VLD); }

static int cmpf(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return x < y ? -1 : x > y; }
/* audit: [0..5] synapses by type, [6] locality violations, [7] max fan-in TIS exc, [8] mean fan-in TIS exc x1000, [9] max fan-in OUT,
   [10] tissue cells with input, [11] TIS mean rate x1e6, [12] TIS rate p10 x1e6, [13] p90 x1e6, [14] made, [15] cut, [16] tissue cells
   reading relay, [17] relay cells with input, [18] synapses touching answer lines (must be 0), [19] synapses into SEN/ANS/FB (must be 0),
   [20] max fan-out TIS exc, [21] TIS silent share x1000 (rate < 1e-4) */
void lt6_audit(Net *h, long *a) {
    memset(a, 0, sizeof(long) * 22);
    for (int id = 0; id < h->ns; id++) { Syn *s = h->S + id; if (s->pre < 0) continue; a[s->ty]++; float r = radius(h, s->ty);
        if (tdist2(h, s->pre, s->post) > r * r) a[6]++;
        if (h->kind[s->pre] == K_ANS || h->kind[s->post] == K_ANS) a[18]++;
        int kp = h->kind[s->post]; if (kp == K_SEN || kp == K_ANS || kp == K_FBP || kp == K_FBN) a[19]++; }
    a[6] += h->st_viol;
    int nt = h->nk[K_TIS], o0 = h->off[K_TIS]; long sum = 0; float *rr = malloc(sizeof(float) * (nt + 1)); long sil = 0;
    for (int q = 0; q < nt; q++) { int i = o0 + q, f = h->in[T_EXC][i].n, fo = h->out[T_EXC][i].n; sum += f; if (f > a[7]) a[7] = f; if (fo > a[20]) a[20] = fo;
        if (f) a[10]++; if (h->in[T_RD][i].n) a[16]++; rr[q] = rate(h, i); if (rr[q] < 1e-4f) sil++; }
    a[8] = nt ? sum * 1000 / nt : 0; a[21] = nt ? sil * 1000 / nt : 0;
    for (int q = 0; q < h->nk[K_OUT]; q++) { int f = h->in[T_OUT][h->off[K_OUT] + q].n; if (f > a[9]) a[9] = f; }
    double ms = 0; for (int q = 0; q < nt; q++) ms += rr[q]; a[11] = (long)(1e6 * ms / (nt ? nt : 1));
    qsort(rr, nt, sizeof(float), cmpf); if (nt) { a[12] = (long)(1e6 * rr[(int)(0.1 * (nt - 1))]); a[13] = (long)(1e6 * rr[(int)(0.9 * (nt - 1))]); } free(rr);
    a[14] = h->st_made; a[15] = h->st_cut;
    for (int k = K_RLP; k <= K_RLN; k++) for (int q = 0; q < h->nk[k]; q++) if (h->in[T_RLY][h->off[k] + q].n) a[17]++;
}
/* fan-in histogram of tissue excitatory synapses into hist[0..63] (bucket = min(fan-in, 63)) */
void lt6_fanin(Net *h, long *hist) { memset(hist, 0, sizeof(long) * 64); for (int q = 0; q < h->nk[K_TIS]; q++) { int f = h->in[T_EXC][h->off[K_TIS] + q].n; hist[f > 63 ? 63 : f]++; } }
/* ablation for checks: with norelay = 1 the feedback reaches the output cells only; no relay, no reading, no tissue third factor */
void lt6_set_norelay(Net *h, int v) { h->norelay = v; }

/* diagnostic for checks: the instrument reads only the output biases (what the net gives without any synaptic drive) */
void lt6_set_biasonly(Net *h, int v) { h->biasonly = v; }
/* output-end statistics: [0] output synapses, [1] outputs with any synapse, [2] mean |w| x1000, [3] max |w| x1000,
   [4] mean output bias x1000, [5] mean drive of driven outputs in the last byte x1000, [6] driven outputs in the last byte */
void lt6_outstats(Net *h, long *a) {
    memset(a, 0, sizeof(long) * 7); double sw = 0; long ns = 0; double sb = 0;
    for (int q = 0; q < h->nk[K_OUT]; q++) { int o = h->off[K_OUT] + q; Vec *v = &h->in[T_OUT][o]; if (v->n) a[1]++; sb += h->b[o];
        for (int u = 0; u < v->n; u++) { float w = fabsf(weff(h, h->S + v->a[u])); sw += w; ns++; if (w * 1000 > a[3]) a[3] = (long)(w * 1000); } }
    a[0] = ns; a[2] = ns ? (long)(1000 * sw / ns) : 0; a[4] = (long)(1000 * sb / h->nk[K_OUT]);
    double sz = 0; for (int k = 0; k < h->out_touch.n; k++) sz += h->zc[h->out_touch.a[k]]; a[6] = h->out_touch.n; a[5] = h->out_touch.n ? (long)(1000 * sz / h->out_touch.n) : 0;
}
