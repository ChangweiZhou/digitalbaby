/* lt2.c - a local tissue of K populations (cell kinds), 24 Nov 2026. Event-driven synapses; one call reads a whole phase.
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
enum { G_K, G_NL, G_KIN, G_FOI, G_FO, G_NO, G_FOO, G_NGP, G_NGD, G_SEED, G_NI };
enum { H_ETA, H_ETAM, H_WNEW, H_ETAO, H_ETAB, H_ETAT, H_RHOO, H_OREP, H_ND };

typedef struct {
    int n, FP, FD, FI, G, nseg, B, k, lp, ld, ad, in, rd, np, pm, dm, off;
    float tlo, thi, rho, pf, sth, dg;
    float *pos, *dec, *h, *m, *val, *pc;
    int *ssrc; float *sw;
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
    int have; uint64_t rs; int lastbyte;
    int *cand; void *snap;
} Net;

static inline uint64_t xr(Net *h) { h->rs ^= h->rs << 13; h->rs ^= h->rs >> 7; h->rs ^= h->rs << 17; return h->rs; }
static inline double ur(Net *h) { return (xr(h) >> 11) * (1.0 / 9007199254740992.0); }
static inline float rdist(float a, float b) { float d = fabsf(a - b); return d > 0.5f ? 1.0f - d : d; }

static int reach(Pop *P, int i, float sp) {                  /* can cell i of P connect to a source at position sp */
    if (rdist(P->pos[i], sp) <= P->rho) return 1;
    for (int k = 0; k < P->np; k++) if (rdist(P->pc[i * P->np + k], sp) <= P->rho * 0.5f) return 1;
    return 0;
}
static int srcpop_ok(int mask, int gp) { return (mask >> (gp < 0 ? 31 : gp)) & 1; }   /* bit 31: sensor lines */

static void add_out(Net *h, int src, int pop, int slot) { int k = h->fo[src]++; h->opop[(size_t)src * h->FOS + k] = pop; h->oslot[(size_t)src * h->FOS + k] = slot; }
static void del_out(Net *h, int src, int pop, int slot) {
    size_t b = (size_t)src * h->FOS;
    for (int k = 0; k < h->fo[src]; k++) if (h->opop[b + k] == pop && h->oslot[b + k] == slot) { int l = --h->fo[src]; h->opop[b + k] = h->opop[b + l]; h->oslot[b + k] = h->oslot[b + l]; return; }
}
static int cap(Net *h, int src) { return src < h->gi[G_NL] ? h->gi[G_FOI] : h->gi[G_FO]; }

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

Net *lt2_new(const int *gi, const double *gd, const int *pi, const double *pd) {
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
    h->FOS = gi[G_FOI] > gi[G_FO] ? gi[G_FOI] : gi[G_FO];
    h->gpos = malloc(sizeof(float) * h->NG); h->gpop = malloc(sizeof(int) * h->NG); h->fo = calloc(h->NG, sizeof(int));
    h->opop = malloc(sizeof(int) * (size_t)h->NG * h->FOS); h->oslot = malloc(sizeof(int) * (size_t)h->NG * h->FOS);
    h->cur = calloc(h->NG, 1); h->prv = calloc(h->NG, 1); h->cand = malloc(sizeof(int) * h->NG);
    for (int s = 0; s < NL; s++) { h->gpos[s] = (s + 0.5f) / NL; h->gpop[s] = -1; }
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
        for (size_t e = 0; e < (size_t)n * P->FI; e++) P->ssrc[e] = -1;
        P->y = calloc(n, 1); P->act = malloc(sizeof(int) * n);
        P->pacc = calloc(n, sizeof(float)); P->sacc = calloc((size_t)n * (P->nseg + 1), sizeof(float));
        P->tp = calloc(n, 1); P->ts = calloc(n, 1); P->tpl = malloc(sizeof(int) * n); P->tsl = malloc(sizeof(int) * n);
        h->prvl[q] = malloc(sizeof(int) * n);
    }
    h->prvl[K] = malloc(sizeof(int) * NL);
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
    h->beta = 1.0; h->lastbyte = -1;
    return h;
}

/* walk the outgoing synapses of an active source; proximal slots add to pacc, distal slots to sacc */
static void spread(Net *h, int src, int distal) {
    size_t b = (size_t)src * h->FOS;
    for (int k = 0; k < h->fo[src]; k++) {
        Pop *P = h->p + h->opop[b + k]; int e = h->oslot[b + k], i = e / P->FI, s = e % P->FI; float w = P->sw[e];
        if (w <= 0) continue;
        if (s < P->FP) { if (distal) continue; if (!P->tp[i]) { P->tp[i] = 1; P->tpl[P->ntp++] = i; } P->pacc[i] += w; }
        else { if (!distal) continue; if (!P->ts[i]) { P->ts[i] = 1; P->tsl[P->nts++] = i; } P->sacc[(size_t)i * P->nseg + (s - P->FP) / P->G] += w; }
    }
}

static float best_seg(Pop *P, int i) { float b = 0; for (int g = 0; g < P->nseg; g++) { float v = P->sacc[(size_t)i * P->nseg + g]; if (v > b) b = v; } return b; }

static void grow(Net *h, Pop *P, int q, int i, int s0, int s1, int mask, const uint8_t *mark, int need) {
    if (need <= 0) return;
    int nc = 0;
    for (int gp = -1; gp < h->K; gp++) {
        if (!srcpop_ok(mask, gp)) continue;
        int idx = gp < 0 ? h->K : gp; int *L = h->prvl[idx]; int nl = h->nprv[idx];
        if (mark == h->cur) { if (gp < 0) { L = h->code + (h->lastbyte < 0 ? 0 : h->lastbyte) * h->gi[G_KIN]; nl = h->lastbyte < 0 ? 0 : h->gi[G_KIN]; } else { L = h->p[gp].act; nl = gp < q ? h->p[gp].nact : 0; } }
        for (int a = 0; a < nl; a++) {
            int src = gp < 0 ? L[a] : h->p[gp].off + L[a];
            if (h->fo[src] >= cap(h, src) || (gp == q && src - P->off == i) || !reach(P, i, h->gpos[src])) continue;
            int dup = 0; for (int s = s0; s < s1; s++) if (P->ssrc[i * P->FI + s] == src) { dup = 1; break; }
            if (!dup) h->cand[nc++] = src;
        }
    }
    if (nc > need) { for (int t = 0; t < need; t++) { int j = t + (int)(xr(h) % (uint64_t)(nc - t)); int z = h->cand[t]; h->cand[t] = h->cand[j]; h->cand[j] = z; } nc = need; }
    for (int t = 0; t < nc; t++) {
        int bs = -1; float bw = 2;
        for (int s = s0; s < s1; s++) { int e = i * P->FI + s, u = P->ssrc[e]; if (u >= 0 && mark[u]) continue; float w = u < 0 ? -1 : P->sw[e]; if (w < bw) { bw = w; bs = s; } }
        if (bs < 0) break;
        int e = i * P->FI + bs, old = P->ssrc[e];
        if (old >= 0) del_out(h, old, q, e);
        P->ssrc[e] = h->cand[t]; P->sw[e] = (float)h->gd[H_WNEW]; add_out(h, h->cand[t], q, e);
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
    for (int a = 0; a < P->nact; a++) h->cur[P->off + P->act[a]] = 1;
    if (learn) {
        double eta = h->gd[H_ETA], etam = h->gd[H_ETAM]; int pm = P->pm | (P->in ? (1 << 31) : 0);
        if (P->ld) for (int k = 0; k < P->nts; k++) {           /* a dendritic event without firing weakens the active distal slots */
            int i = P->tsl[k]; if (P->y[i] || best_seg(P, i) < sth) continue;
            for (int s = P->FP; s < P->FI; s++) { int e = i * P->FI + s, u = P->ssrc[e]; if (u >= 0 && h->prv[u]) P->sw[e] -= (float)(etam * P->sw[e]); }
        }
        for (int a = 0; a < P->nact; a++) {
            int i = P->act[a];
            if (P->lp) {
                int na = 0;
                for (int s = 0; s < P->FP; s++) { int e = i * P->FI + s, u = P->ssrc[e]; if (u < 0) continue; if (h->cur[u]) { P->sw[e] += (float)(eta * (1 - P->sw[e])); na++; } else P->sw[e] -= (float)(etam * P->sw[e]); }
                grow(h, P, q, i, 0, P->FP, pm, h->cur, h->gi[G_NGP] - na);
            }
            if (P->ld && P->FD) {
                int na = 0;
                for (int s = P->FP; s < P->FI; s++) { int e = i * P->FI + s, u = P->ssrc[e]; if (u >= 0 && h->prv[u]) { P->sw[e] += (float)(eta * (1 - P->sw[e])); na++; } }
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

void lt2_run(Net *h, const uint8_t *x, int n, int learn, double *stats) {
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
                    double e = (v == y) - h->o[v]; int na = 0;
                    for (int k = 0; k < NO; k++) { int s = h->osrc[v * NO + k]; if (s >= 0 && h->cur[s]) na++; }
                    h->b[v] += eb * e;
                    if (na) for (int k = 0; k < NO; k++) { int s = h->osrc[v * NO + k]; if (s >= 0 && h->cur[s]) h->ow[v * NO + k] += (float)(eo * e / na); }
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
            for (int k = 0; k < NO; k++) { int src = h->osrc[v * NO + k]; if (src >= 0 && h->cur[src]) s += h->ow[v * NO + k]; }
            h->o[v] = s;
        }
        h->have = 1;
    }
}

/* snapshot of the dynamic state (tests with learning off leave the life undisturbed) */
#define SNAPF(X) X(h->cur, h->NG); X(h->prv, h->NG); X(h->o, V); X(&h->have, 1); X(&h->rs, 1); X(&h->lastbyte, 1); X(h->nprv, MAXK + 1); \
    for (int q = 0; q <= h->K; q++) X(h->prvl[q], q < h->K ? h->p[q].n : h->gi[G_NL]); \
    for (int q = 0; q < h->K; q++) { Pop *P = h->p + q; X(P->h, P->n); X(P->m, P->n); X(P->y, P->n); X(P->act, P->n); X(&P->nact, 1); \
        X(P->sacc, (size_t)P->n * (P->nseg + 1)); X(P->ts, P->n); X(P->tsl, P->n); X(&P->nts, 1); }
#define SZ(p, k) sz += sizeof(*(p)) * (size_t)(k);
#define SV(p, k) { memcpy(c, p, sizeof(*(p)) * (size_t)(k)); c += sizeof(*(p)) * (size_t)(k); }
#define LD(p, k) { memcpy(p, c, sizeof(*(p)) * (size_t)(k)); c += sizeof(*(p)) * (size_t)(k); }
void lt2_snapshot(Net *h) { size_t sz = 0; SNAPF(SZ); if (!h->snap) h->snap = malloc(sz); char *c = h->snap; SNAPF(SV); }
void lt2_restore(Net *h) { char *c = h->snap; SNAPF(LD); }

/* audit: [0] max tissue fan-in in use, [1] max fan-out (cells), [2] max sensor-line fan-out, [3] max output fan-in, [4] max fan-out to outputs, [5] synapses */
void lt2_audit(Net *h, long *a) {
    memset(a, 0, sizeof(long) * 6);
    for (int q = 0; q < h->K; q++) { Pop *P = h->p + q;
        for (int i = 0; i < P->n; i++) { int c = 0; for (int s = 0; s < P->FI; s++) c += P->ssrc[i * P->FI + s] >= 0; if (c > a[0]) a[0] = c; a[5] += c; } }
    for (int s = 0; s < h->NG; s++) { long *t = s < h->gi[G_NL] ? &a[2] : &a[1]; if (h->fo[s] > *t) *t = h->fo[s]; if (h->ocnt[s] > a[4]) a[4] = h->ocnt[s]; }
    int NO = h->gi[G_NO]; for (int v = 0; v < V; v++) { int c = 0; for (int k = 0; k < NO; k++) c += h->osrc[v * NO + k] >= 0; if (c > a[3]) a[3] = c; }
}
