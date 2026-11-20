
#include <math.h>
#include <stdint.h>
#include <string.h>
#define S 14
#define NL 40
#define MAXK 256
typedef struct { int32_t D, nP, nG, look, levown, lag; double eta, e; int32_t *P, *A, *B; double *T, *U;
                 const int32_t *pd, *pn, *gd, *gn; const int64_t *pbase, *gbase; double ncap; } St;
typedef struct { int64_t pid[MAXK], aid[MAXK], bid[MAXK]; int lev[6][MAXK]; double s[6][MAXK][S]; double p[3][S]; int g3, g2, ans; } Rec;
static Rec ring[3], scratch; static int hist[16]; static int64_t tt; static int gq3[64], gq2[64];
static double lt[8192], ln_[65536], cur_e = -1;
static void tables(double e) { if (e == cur_e) return; cur_e = e; for (int c = 0; c < 8192; c++) lt[c] = log(c + e); for (int n = 0; n < 65536; n++) ln_[n] = log(n + S * e); }
static inline int level(int64_t n, int u) { int bn = 0; while (n > 1 && bn < 9) { n >>= 1; bn++; } return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3); }
static inline void say(const St *st, const int32_t *c, const int32_t *cl, int f, int k, Rec *r, double *a) {
    int64_t n = 0; int u = 0;
    for (int v = 0; v < S; v++) { n += c[v]; u += c[v] > 0; }
    if (n < 1) { r->lev[f][k] = -1; return; }
    int64_t nl = n; int ul = u;
    if (cl) { nl = 0; ul = 0; for (int v = 0; v < S; v++) { nl += cl[v]; ul += cl[v] > 0; } }
    const int lv = level(nl, ul); r->lev[f][k] = lv;
    const double tk = st->T[((int64_t)f * MAXK + k) * NL + lv], den = n < 65536 ? ln_[n] : log(n + S * st->e); double *s = r->s[f][k];
    if (st->ncap > 0 && n > st->ncap) {      /* PLAN_3.md: the proportions at the weight of ncap counts */
        const double den2 = log(st->ncap + S * st->e), zero = log(st->e) - den2;
        for (int v = 0; v < S; v++) { const double sv = c[v] ? log(c[v] * st->ncap / n + st->e) - den2 : zero; s[v] = sv; a[v] += tk * sv; }
        return; }
    for (int v = 0; v < S; v++) { const double sv = (c[v] < 8192 ? lt[c[v]] : log(c[v] + st->e)) - den; s[v] = sv; a[v] += tk * sv; }
}
static inline int share(const double *a, double *p) {
    int g = 0; double z = 0; for (int v = 1; v < S; v++) if (a[v] > a[g]) g = v;
    for (int v = 0; v < S; v++) { p[v] = exp(a[v] - a[g]); z += p[v]; }
    for (int v = 0; v < S; v++) p[v] /= z;
    return g;
}
static void form(const St *st, const int *h, Rec *r, int kg3, int kg2) {
    double a[S]; int64_t gkey[MAXK];
    for (int k = 0; k < st->nP; k++) { int64_t idx = 0, mul = 1; for (int i = 0; i < st->pn[k]; i++) { idx += h[st->pd[k * 4 + i]] * mul; mul *= S; } r->pid[k] = st->pbase[k] + idx; }
    if (st->look) {
        for (int j = 0; j < st->nG; j++) { int64_t idx = 0, mul = 1; for (int i = 0; i < st->gn[j]; i++) { idx += h[st->gd[j * 4 + i]] * mul; mul *= S; } gkey[j] = st->gbase[j] + idx * S; }
        memset(a, 0, sizeof a);
        for (int k = 0; k < st->nP; k++) { const int32_t *c = st->P + r->pid[k] * 42; say(st, c + 28, st->levown ? 0 : c, 2, k, r, a); }
        r->g3 = share(a, r->p[2]); const int k3 = kg3 >= 0 ? kg3 : r->g3;
        memset(a, 0, sizeof a);
        for (int k = 0; k < st->nP; k++) { const int32_t *c = st->P + r->pid[k] * 42; say(st, c + 14, st->levown ? 0 : c, 1, k, r, a); }
        for (int j = 0; j < st->nG; j++) { r->aid[j] = gkey[j] + k3; const int32_t *c = st->A + r->aid[j] * 28; say(st, c + 14, st->levown ? 0 : c, 4, j, r, a); }
        r->g2 = share(a, r->p[1]); const int k2 = kg2 >= 0 ? kg2 : r->g2;
        for (int j = 0; j < st->nG; j++) r->bid[j] = gkey[j] + k2;
    } else { r->g3 = -1; r->g2 = -1; }
    memset(a, 0, sizeof a);
    for (int k = 0; k < st->nP; k++) say(st, st->P + r->pid[k] * 42, 0, 0, k, r, a);
    if (st->look) {
        for (int j = 0; j < st->nG; j++) say(st, st->A + r->aid[j] * 28, 0, 3, j, r, a);
        for (int j = 0; j < st->nG; j++) say(st, st->B + r->bid[j] * 14, 0, 5, j, r, a);
    }
    r->ans = share(a, r->p[0]);
}
static inline void teach(const St *st, const Rec *r, int f, int nk, const double *p, int y) {
    for (int k = 0; k < nk; k++) { const int lv = r->lev[f][k]; if (lv < 0) continue;
        const double *s = r->s[f][k]; double g = s[y]; for (int v = 0; v < S; v++) g -= p[v] * s[v];
        const int64_t i = ((int64_t)f * MAXK + k) * NL + lv; st->T[i] += st->eta / sqrt(1.0 + st->U[i]) * g; st->U[i] += 1.0; }
}
static void arrive(const St *st, int y, const Rec *r0, const Rec *r1, const Rec *r2) {
    teach(st, r0, 0, st->nP, r0->p[0], y);
    if (st->look) {
        teach(st, r0, 3, st->nG, r0->p[0], y); teach(st, r0, 5, st->nG, r0->p[0], y);
        if (r1) { teach(st, r1, 1, st->nP, r1->p[1], y); teach(st, r1, 4, st->nG, r1->p[1], y); }
        if (r2) teach(st, r2, 2, st->nP, r2->p[2], y);
    }
    for (int k = 0; k < st->nP; k++) st->P[r0->pid[k] * 42 + y]++;
    if (st->look) {
        for (int j = 0; j < st->nG; j++) { st->A[r0->aid[j] * 28 + y]++; st->B[r0->bid[j] * 14 + y]++; }
        if (r1) { for (int k = 0; k < st->nP; k++) st->P[r1->pid[k] * 42 + 14 + y]++; for (int j = 0; j < st->nG; j++) st->A[r1->aid[j] * 28 + 14 + y]++; }
        if (r2) for (int k = 0; k < st->nP; k++) st->P[r2->pid[k] * 42 + 28 + y]++;
    }
}
void reset(int end) { tt = 0; for (int d = 0; d < 16; d++) hist[d] = end; }
/* read n bytes of the stream: before each, the guesses and the answer; at its arrival, the trusts and the counts */
void run(const St *st, const int32_t *x, int64_t n, double *outp, int32_t *outg) {
    tables(st->e);
    for (int64_t i = 0; i < n; i++, tt++) {
        Rec *r0 = &ring[tt % 3], *r1 = tt >= 1 ? &ring[(tt + 2) % 3] : 0, *r2 = tt >= 2 ? &ring[(tt + 1) % 3] : 0; int kg3 = -1, kg2 = -1;
        if (st->lag > 0 && tt >= st->lag) { kg3 = gq3[(tt - st->lag) % 64]; kg2 = gq2[(tt - st->lag) % 64]; }
        form(st, hist, r0, kg3, kg2); gq3[tt % 64] = r0->g3; gq2[tt % 64] = r0->g2;
        if (outp) { memcpy(outp + i * 3 * S, r0->p, sizeof r0->p); outg[i * 3] = r0->ans; outg[i * 3 + 1] = r0->g2; outg[i * 3 + 2] = r0->g3; }
        arrive(st, x[i], r0, r1, r2);
        for (int d = st->D; d > 1; d--) hist[d] = hist[d - 1];
        hist[1] = x[i];
    }
}
/* the answer and the guesses for a given history (h[1..D]); nothing changes */
void answer(const St *st, const int32_t *h, int kg3, int kg2, int32_t *out) {
    int hh[16]; tables(st->e); for (int d = 0; d <= st->D; d++) hh[d] = h[d];
    form(st, hh, &scratch, kg3, kg2); out[0] = scratch.ans; out[1] = scratch.g2; out[2] = scratch.g3;
}
void sizes(const St *st, int64_t nP, int64_t nA, int64_t *out) {
    int64_t cells = 0, above = 0;
    for (int64_t i = 0; i < nP; i++) { const int32_t *c = st->P + i * 42; int64_t n = 0; for (int v = 0; v < 14; v++) n += c[v]; cells += n > 0; for (int v = 0; v < 42; v++) above += c[v] > 0; }
    if (st->look) for (int64_t i = 0; i < nA; i++) {
        const int32_t *c = st->A + i * 28; int64_t n = 0; for (int v = 0; v < 14; v++) n += c[v]; cells += n > 0; for (int v = 0; v < 28; v++) above += c[v] > 0;
        c = st->B + i * 14; n = 0; for (int v = 0; v < 14; v++) { n += c[v]; above += c[v] > 0; } cells += n > 0; }
    out[0] = cells; out[1] = above;
}
