
#include <math.h>
#include <stdint.h>
#include <string.h>
#define NL 40
#define MAXK 256
#define MAXS 18
#define MAXL 4
#define MAXF 10
typedef struct { int32_t S, D, nP, nG, L; double eta, e; int32_t *P; int32_t *F[MAXL + 1]; double *T, *U;
                 const int32_t *pd, *pn, *gd, *gn; const int64_t *pbase, *gbase; int32_t fam[MAXL + 1][MAXL]; } St;
typedef struct { int64_t pid[MAXK], fid[MAXL + 1][MAXK]; int lev[MAXF][MAXK]; double s[MAXF][MAXK][MAXS]; double p[MAXL][MAXS]; int g[MAXL + 1], ans; } Rec;
static Rec ring[MAXL], scratch; static int hist[16]; static int64_t tt; static int pend = 1;
static double lt[8192], ln_[65536], cur_e = -1; static int cur_S = -1;
static void tables(double e, int S) { if (e == cur_e && S == cur_S) return; cur_e = e; cur_S = S; for (int c = 0; c < 8192; c++) lt[c] = log(c + e); for (int n = 0; n < 65536; n++) ln_[n] = log(n + S * e); }
static inline int level(int64_t n, int u) { int bn = 0; while (n > 1 && bn < 9) { n >>= 1; bn++; } return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3); }
static inline void say(const St *st, const int32_t *c, int f, int k, Rec *r, double *a) {
    const int S = st->S; int64_t n = 0; int u = 0;
    for (int v = 0; v < S; v++) { n += c[v]; u += c[v] > 0; }
    if (n < 1) { r->lev[f][k] = -1; return; }
    const int lv = level(n, u); r->lev[f][k] = lv;
    const double tk = st->T[((int64_t)f * MAXK + k) * NL + lv], den = n < 65536 ? ln_[n] : log(n + S * st->e); double *s = r->s[f][k];
    for (int v = 0; v < S; v++) { const double sv = (c[v] < 8192 ? lt[c[v]] : log(c[v] + st->e)) - den; s[v] = sv; a[v] += tk * sv; }
}
static inline int share(int S, const double *a, double *p) {
    int g = 0; double z = 0; for (int v = 1; v < S; v++) if (a[v] > a[g]) g = v;
    for (int v = 0; v < S; v++) { p[v] = exp(a[v] - a[g]); z += p[v]; }
    for (int v = 0; v < S; v++) p[v] /= z;
    return g;
}
static void form(const St *st, const int *h, Rec *r) {
    const int S = st->S, L = st->L, L1 = L > 1 ? L : 1; double a[MAXS]; int64_t gkey[MAXK];
    for (int k = 0; k < st->nP; k++) { int64_t idx = 0, mul = 1; for (int i = 0; i < st->pn[k]; i++) { idx += h[st->pd[k * 4 + i]] * mul; mul *= S; } r->pid[k] = st->pbase[k] + idx; }
    if (L > 1) for (int j = 0; j < st->nG; j++) { int64_t idx = 0, mul = 1; for (int i = 0; i < st->gn[j]; i++) { idx += h[st->gd[j * 4 + i]] * mul; mul *= S; } gkey[j] = st->gbase[j] + idx * S; }
    for (int j = L; j > 1; j--) {
        const int w = j - 1; memset(a, 0, sizeof a);
        for (int k = 0; k < st->nP; k++) say(st, st->P + r->pid[k] * L1 * S + w * S, w, k, r, a);
        for (int i = L; i > j; i--) for (int q = 0; q < st->nG; q++) say(st, st->F[i] + r->fid[i][q] * (i - 1) * S + w * S, st->fam[i][w], q, r, a);
        r->g[j] = share(S, a, r->p[w]);
        for (int q = 0; q < st->nG; q++) r->fid[j][q] = gkey[q] + r->g[j];
    }
    memset(a, 0, sizeof a);
    for (int k = 0; k < st->nP; k++) say(st, st->P + r->pid[k] * L1 * S, 0, k, r, a);
    for (int i = L; i > 1; i--) for (int q = 0; q < st->nG; q++) say(st, st->F[i] + r->fid[i][q] * (i - 1) * S, st->fam[i][0], q, r, a);
    r->ans = share(S, a, r->p[0]);
}
static inline void teach(const St *st, const Rec *r, int f, int nk, const double *p, int y) {
    const int S = st->S;
    for (int k = 0; k < nk; k++) { const int lv = r->lev[f][k]; if (lv < 0) continue;
        const double *s = r->s[f][k]; double g = s[y]; for (int v = 0; v < S; v++) g -= p[v] * s[v];
        const int64_t i = ((int64_t)f * MAXK + k) * NL + lv; st->T[i] += st->eta / sqrt(1.0 + st->U[i]) * g; st->U[i] += 1.0; }
}
void reset(int end) { tt = 0; pend = 1; for (int d = 0; d < 16; d++) hist[d] = end; }
/* read n bytes of the stream: before each, the guesses and the answer; at its arrival, the trusts and the counts; then the chain steps or does not.
   fl[i]: whether the byte x[i] enters the chain when the next byte arrives (0: all enter). Returns the sum of the answering cell's squared mismatches. */
double run(const St *st, const int32_t *x, const uint8_t *fl, int64_t n, double *outp, int32_t *outg) {
    const int S = st->S, L = st->L, L1 = L > 1 ? L : 1; double esum = 0; tables(st->e, S);
    for (int64_t i = 0; i < n; i++, tt++) {
        Rec *rec[MAXL]; const int nr = tt + 1 < L1 ? (int)(tt + 1) : L1; const int y = x[i];
        for (int w = 0; w < nr; w++) rec[w] = &ring[(tt - w) % L1];
        Rec *r0 = rec[0]; form(st, hist, r0);
        if (outp) { for (int w = 0; w < L1; w++) memcpy(outp + (i * L1 + w) * S, r0->p[w], S * sizeof(double)); outg[i * (MAXL + 1)] = r0->ans; for (int j = 2; j <= L; j++) outg[i * (MAXL + 1) + j] = r0->g[j]; }
        for (int v = 0; v < S; v++) { const double m = (v == y) - r0->p[0][v]; esum += m * m; }
        for (int w = 0; w < nr; w++) { teach(st, rec[w], w, st->nP, rec[w]->p[w], y); for (int j = w + 2; j <= L; j++) teach(st, rec[w], st->fam[j][w], st->nG, rec[w]->p[w], y); }
        for (int w = 0; w < nr; w++) {
            for (int k = 0; k < st->nP; k++) st->P[rec[w]->pid[k] * L1 * S + w * S + y]++;
            for (int j = w + 2; j <= L; j++) for (int q = 0; q < st->nG; q++) st->F[j][rec[w]->fid[j][q] * (j - 1) * S + w * S + y]++;
        }
        if (pend) for (int d = st->D; d > 1; d--) hist[d] = hist[d - 1];
        hist[1] = y; pend = fl ? fl[i] : 1;
    }
    return esum;
}
/* the answer and the guesses for a given history (h[1..D]); nothing changes */
void answer(const St *st, const int32_t *h, int32_t *out) {
    int hh[16]; tables(st->e, st->S); for (int d = 0; d <= st->D; d++) hh[d] = h[d];
    form(st, hh, &scratch); out[0] = scratch.ans; for (int j = 2; j <= st->L; j++) out[j] = scratch.g[j];
}
void state(int32_t *h, int32_t *p) { for (int d = 0; d < 16; d++) h[d] = hist[d]; *p = pend; }
/* cells that have counted, counts above zero; and, given a set of byte values (dead[v]), the cells keyed on one of them at a chain position (2..D) */
static void census1(const St *st, const int32_t *C, int stride, int nk, const int32_t *kd, const int32_t *kn, const int64_t *base, int guess, const uint8_t *dead, int64_t *out) {
    const int S = st->S;
    for (int k = 0; k < nk; k++) {
        int64_t cnt = 1; for (int i = 0; i < kn[k]; i++) cnt *= S;
        for (int64_t idx = 0; idx < cnt; idx++) {
            int isdead = 0; int64_t q = idx; for (int i = 0; i < kn[k]; i++) { const int v = q % S; q /= S; if (kd[k * 4 + i] >= 2 && dead[v]) isdead = 1; }
            for (int g = 0; g < (guess ? S : 1); g++) {
                const int32_t *c = C + (guess ? base[k] + idx * S + g : base[k] + idx) * stride; int64_t n = 0, ab = 0;
                for (int v = 0; v < S; v++) n += c[v];
                if (!n) continue;
                for (int v = 0; v < stride; v++) ab += c[v] > 0;
                out[0]++; out[1] += ab; if (isdead) { out[2]++; out[3] += ab; }
            }
        }
    }
}
void census(const St *st, const uint8_t *dead, int64_t *out) {
    const int L1 = st->L > 1 ? st->L : 1; out[0] = out[1] = out[2] = out[3] = 0;
    census1(st, st->P, L1 * st->S, st->nP, st->pd, st->pn, st->pbase, 0, dead, out);
    for (int j = 2; j <= st->L; j++) census1(st, st->F[j], (j - 1) * st->S, st->nG, st->gd, st->gn, st->gbase, 1, dead, out);
}
