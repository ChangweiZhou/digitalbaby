
#include <math.h>
#include <stdint.h>
#include <string.h>
#define S 14
#define NL 40
#define MAXK 256
#define MAXH 4
typedef struct { int32_t D, H, nk[MAXH]; double eta, e; int32_t *C[MAXH]; double *T, *U; const int32_t *kd[MAXH], *kn[MAXH]; const int64_t *base[MAXH]; } St;
typedef struct { int64_t id[MAXH][MAXK]; int lev[MAXH][MAXH][MAXK]; double s[MAXH][MAXH][MAXK][S]; double p[MAXH + 1][S]; int g[MAXH + 1]; } Rec;
static Rec ring[MAXH], scratch; static int hist[16]; static int64_t tt;
static double lt[8192], ln_[65536], cur_e = -1;
static void tables(double e) { if (e == cur_e) return; cur_e = e; for (int c = 0; c < 8192; c++) lt[c] = log(c + e); for (int n = 0; n < 65536; n++) ln_[n] = log(n + S * e); }
static inline int level(int64_t n, int u) { int bn = 0; while (n > 1 && bn < 9) { n >>= 1; bn++; } return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3); }
#define TI(q, d, k, lv) (((((int64_t)(q) * MAXH + (d)) * MAXK + (k)) * NL) + (lv))
static void form(const St *st, const int *h, Rec *r) {
    const int H = st->H; double a[S]; int64_t key[MAXH][MAXK];
    for (int q = 0; q < H; q++) for (int k = 0; k < st->nk[q]; k++) {
        int64_t idx = 0, mul = 1; for (int i = 0; i < st->kn[q][k]; i++) { idx += h[st->kd[q][k * 4 + i]] * mul; mul *= S; }
        key[q][k] = st->base[q][k] + (q ? idx * S : idx); }
    for (int k = 0; k < st->nk[0]; k++) r->id[0][k] = key[0][k];
    for (int d = H; d >= 1; d--) {
        memset(a, 0, sizeof a);
        for (int q = 0; q <= H - d; q++) { const int ns = H - q;
            for (int k = 0; k < st->nk[q]; k++) {
                const int32_t *c = st->C[q] + (r->id[q][k] * ns + (d - 1)) * S; int64_t n = 0; int u = 0;
                for (int v = 0; v < S; v++) { n += c[v]; u += c[v] > 0; }
                if (n < 1) { r->lev[q][d - 1][k] = -1; continue; }
                const int lv = level(n, u); r->lev[q][d - 1][k] = lv; const double tk = st->T[TI(q, d - 1, k, lv)], den = n < 65536 ? ln_[n] : log(n + S * st->e); double *s = r->s[q][d - 1][k];
                for (int v = 0; v < S; v++) { const double sv = (c[v] < 8192 ? lt[c[v]] : log(c[v] + st->e)) - den; s[v] = sv; a[v] += tk * sv; }
            } }
        int g = 0; double z = 0, *p = r->p[d]; for (int v = 1; v < S; v++) if (a[v] > a[g]) g = v;
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - a[g]); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        r->g[d] = g;
        if (d >= 2) { const int q = H - d + 1; for (int k = 0; k < st->nk[q]; k++) r->id[q][k] = key[q][k] + g; }
    }
}
static void arrive(const St *st, int y, Rec **recs, int nrec) {
    const int H = st->H;
    for (int d = 1; d <= H && d <= nrec; d++) { const Rec *r = recs[d - 1]; const double *p = r->p[d];
        for (int q = 0; q <= H - d; q++) for (int k = 0; k < st->nk[q]; k++) { const int lv = r->lev[q][d - 1][k]; if (lv < 0) continue;
            const double *s = r->s[q][d - 1][k]; double g = s[y]; for (int v = 0; v < S; v++) g -= p[v] * s[v];
            const int64_t i = TI(q, d - 1, k, lv); st->T[i] += st->eta / sqrt(1.0 + st->U[i]) * g; st->U[i] += 1.0; } }
    for (int d = 1; d <= H && d <= nrec; d++) { const Rec *r = recs[d - 1];
        for (int q = 0; q <= H - d; q++) { const int ns = H - q; for (int k = 0; k < st->nk[q]; k++) st->C[q][(r->id[q][k] * ns + (d - 1)) * S + y]++; } }
}
void reset(int end) { tt = 0; for (int d = 0; d < 16; d++) hist[d] = end; }
void run(const St *st, const int32_t *x, int64_t n, double *outp, int32_t *outg) {
    const int H = st->H; tables(st->e);
    for (int64_t i = 0; i < n; i++, tt++) {
        Rec *recs[MAXH]; int nrec = 0; for (int d = 0; d < H && d <= tt; d++) recs[nrec++] = &ring[(tt - d) % H];
        form(st, hist, recs[0]);
        if (outp) { memcpy(outp + i * (MAXH + 1) * S, recs[0]->p, sizeof recs[0]->p); memcpy(outg + i * (MAXH + 1), recs[0]->g, sizeof recs[0]->g); }
        arrive(st, x[i], recs, nrec);
        for (int d = st->D; d > 1; d--) hist[d] = hist[d - 1];
        hist[1] = x[i];
    }
}
void answer(const St *st, const int32_t *h, int32_t *out) {
    int hh[16]; tables(st->e); for (int d = 0; d <= st->D; d++) hh[d] = h[d];
    form(st, hh, &scratch); for (int d = 0; d <= MAXH; d++) out[d] = scratch.g[d];
}
void sizes(const St *st, const int64_t *ncell, int64_t *out) {
    int64_t cells = 0, above = 0;
    for (int q = 0; q < st->H; q++) { const int ns = st->H - q; for (int64_t i = 0; i < ncell[q]; i++) { const int32_t *c = st->C[q] + i * ns * S; int64_t n = 0;
        for (int v = 0; v < S; v++) n += c[v]; cells += n > 0; for (int v = 0; v < ns * S; v++) above += c[v] > 0; } }
    out[0] = cells; out[1] = above;
}
