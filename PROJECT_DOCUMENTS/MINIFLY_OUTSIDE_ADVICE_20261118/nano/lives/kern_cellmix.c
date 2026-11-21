
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define NL 40
#define SMAX 256
#define SIG (1.0 / 32)
#define KMAX 600
#define JMAX 8
#define LMAX 2048
typedef struct {
    int S, K, gate, J, M, rule, age; double eta, e, birth, spow;
    int ktype[KMAX], knd[KMAX], kdel[KMAX][8], kcap[KMAX], ksil[KMAX]; float *cw;
    double *T, *Tu, *Ta;            /* K x NL: the first cell's trusts, times used, sum of squared drivers */
    double *G, *Gu, *Ga; int64_t gsize; int GL;   /* the second cell's parts: J x K x GL x S (GL = 1 without the level, NL with it) */
    /* cells */
    uint64_t *hkey; uint16_t *hkind; int32_t *hcell; int64_t hcap, ncell;
    uint32_t *cN; uint16_t *cU, *ccap; int64_t *cptr; uint8_t *cL; uint32_t *cT; int64_t ccells;
    uint8_t *pv; uint32_t *pc; int64_t ptop, pcap;
    /* recall */
    int32_t *head, *prev; int64_t nmax; int hbits;
    /* since */
    int64_t last[SMAX]; int64_t synced; int64_t now; uint32_t *ptime; double gmin; int nb, bcap; int64_t bper, lnow; double share; double *BT, *BG, *w2, *pb2, *ab, *tb, *mxv, *zv, *mo2; int32_t *to2;      /* nano/lives: now = bytes read so far; last[] and cT hold times; ptime = the time at which a place was written */
    double lt[4096];
} Sy;
static inline int level(uint32_t n, int u) { int bn = 0; int64_t m = n; while (m > 1 && bn < 9) { m >>= 1; bn++; } return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3); }
static inline int reclevel(int L, int M) { int d = L - M; if (d < 16) return d; int b = 16; int64_t m = (d - 16) / 8; while (m > 0 && b < NL - 1) { m >>= 1; b++; } return b; }
/* PLAN_7.md: levels by the age of what a sender of recency says. last: (different bytes the cell has met: 1, 2, 3-4, more) x (bytes since the cell last counted: 1-3, 4-15, ..., four-fold, ten steps);
   recall: (bytes of agreement beyond M: 0, 1, 2-3, ..., 64 and more; eight steps) x (bytes back to the place: under 64, 1,024, 16,384, 262,144, more) */
static inline int agelevel_last(int u, int64_t d) { int b = 0; while (d >= 4 && b < 9) { d >>= 2; b++; } return (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3) * 10 + b; }
static inline int agelevel_recall(int L, int M, int64_t d) { int n = L - M, b = 0, c = 0; while (n >= 1 && b < 7) { n >>= 1; b++; } d >>= 6; while (d > 0 && c < 4) { d >>= 4; c++; } return b * 5 + c; }
static inline uint64_t mix(uint64_t z) { z += 0x9E3779B97F4A7C15ULL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
static inline int64_t slot_of(const Sy *s, uint64_t key, int kind) {
    int64_t i = mix(key ^ ((uint64_t)kind * 0xD6E8FEB86659FD93ULL)) & (s->hcap - 1);
    while (s->hcell[i] >= 0 && !(s->hkey[i] == key && s->hkind[i] == kind)) i = (i + 1) & (s->hcap - 1);
    return i;
}
static void grow(Sy *s) {
    int64_t oc = s->hcap; uint64_t *ok = s->hkey; uint16_t *oki = s->hkind; int32_t *oce = s->hcell;
    s->hcap = oc * 2; s->hkey = malloc(s->hcap * 8); s->hkind = malloc(s->hcap * 2); s->hcell = malloc(s->hcap * 4); memset(s->hcell, 0xff, s->hcap * 4);
    for (int64_t i = 0; i < oc; i++) if (oce[i] >= 0) { int64_t j = slot_of(s, ok[i], oki[i]); s->hkey[j] = ok[i]; s->hkind[j] = oki[i]; s->hcell[j] = oce[i]; }
    free(ok); free(oki); free(oce);
}
static int32_t new_cell(Sy *s) {
    if (s->ncell == s->ccells) { s->ccells *= 2; s->cN = realloc(s->cN, s->ccells * 4); s->cU = realloc(s->cU, s->ccells * 2); s->ccap = realloc(s->ccap, s->ccells * 2); s->cptr = realloc(s->cptr, s->ccells * 8); s->cL = realloc(s->cL, s->ccells); s->cT = realloc(s->cT, s->ccells * 4); s->cw = realloc(s->cw, s->ccells * 12); }
    int32_t c = (int32_t)s->ncell++; s->cN[c] = 0; s->cU[c] = 0; s->ccap[c] = 0; s->cptr[c] = 0; s->cw[3 * (int64_t)c] = s->cw[3 * (int64_t)c + 1] = s->cw[3 * (int64_t)c + 2] = (float)(1.0 / 3.0); return c;
}
static void count(Sy *s, int32_t c, int y, int64_t t) {
    uint8_t *pv = s->pv + s->cptr[c]; uint32_t *pc = s->pc + s->cptr[c]; int u = s->cU[c]; s->cL[c] = (uint8_t)y; s->cT[c] = (uint32_t)t;
    for (int i = 0; i < u; i++) if (pv[i] == y) { pc[i]++; s->cN[c]++; return; }
    if (u == s->ccap[c]) {
        int nc = u ? u * 2 : 1; if (nc > SMAX) nc = SMAX;
        if (s->ptop + nc > s->pcap) { s->pcap = s->pcap * 2 + nc; s->pv = realloc(s->pv, s->pcap); s->pc = realloc(s->pc, s->pcap * 4); }
        memcpy(s->pv + s->ptop, s->pv + s->cptr[c], u); memcpy(s->pc + s->ptop, s->pc + s->cptr[c], (size_t)u * 4); s->cptr[c] = s->ptop; s->ptop += nc; s->ccap[c] = nc;
    }
    s->pv[s->cptr[c] + u] = y; s->pc[s->cptr[c] + u] = 1; s->cU[c] = u + 1; s->cN[c]++;
}
static void halve(Sy *s, int32_t c) {      /* PLAN_5: every count of the cell is halved; a count that falls to nothing is dropped */
    uint8_t *pv = s->pv + s->cptr[c]; uint32_t *pc = s->pc + s->cptr[c]; const int u = s->cU[c]; int w = 0; uint32_t n = 0;
    for (int i = 0; i < u; i++) { const uint32_t h = pc[i] >> 1; if (h > 0) { pv[w] = pv[i]; pc[w] = h; w++; n += h; } }
    s->cU[c] = (uint16_t)w; s->cN[c] = n;
}
void sy_silent(Sy *s, const int32_t *sil) { for (int k = 0; k < s->K; k++) s->ksil[k] = sil[k]; }
void sy_caps(Sy *s, const int32_t *caps) { for (int k = 0; k < s->K; k++) s->kcap[k] = caps[k]; }
void sy_clear_kind(Sy *s, int kind) { for (int64_t i = 0; i < s->hcap; i++) if (s->hcell[i] >= 0 && s->hkind[i] == kind) { s->cN[s->hcell[i]] = 0; s->cU[s->hcell[i]] = 0; } }
void sy_clear_places(Sy *s) { memset(s->head, 0xff, sizeof(int32_t) << s->hbits); }
Sy *sy_new(int S, int K, const int32_t *ktype, const int32_t *knd, const int32_t *kdel, const double *kborn, double eta, double e, int rule, int gate, int J, double birth, int M, int64_t nmax, int age, double spow) {
    Sy *s = calloc(1, sizeof(Sy)); s->S = S; s->K = K; s->gate = gate; s->J = J; s->M = M; s->rule = rule; s->eta = eta; s->e = e; s->birth = birth; s->nmax = nmax; s->age = age; s->spow = spow;
    for (int k = 0; k < K; k++) { s->ktype[k] = ktype[k]; s->knd[k] = knd[k]; for (int i = 0; i < 8; i++) s->kdel[k][i] = kdel[k * 8 + i]; }
    s->T = malloc(sizeof(double) * K * NL); s->Tu = calloc(K * NL, sizeof(double)); s->Ta = calloc(K * NL, sizeof(double));
    for (int k = 0; k < K; k++) for (int l = 0; l < NL; l++) s->T[k * NL + l] = kborn[k];
    s->GL = gate == 2 ? NL : 1; s->gsize = gate ? (int64_t)J * K * s->GL * S : 0;
    if (gate) { s->G = calloc(s->gsize, sizeof(double)); s->Gu = calloc(s->gsize, sizeof(double)); s->Ga = calloc(s->gsize, sizeof(double)); }
    s->hcap = 1 << 16; s->hkey = malloc(s->hcap * 8); s->hkind = malloc(s->hcap * 2); s->hcell = malloc(s->hcap * 4); memset(s->hcell, 0xff, s->hcap * 4);
    s->ccells = 1 << 16; s->cN = malloc(s->ccells * 4); s->cU = malloc(s->ccells * 2); s->ccap = malloc(s->ccells * 2); s->cptr = malloc(s->ccells * 8); s->cL = malloc(s->ccells); s->cT = malloc(s->ccells * 4); s->cw = malloc(s->ccells * 12);
    s->pcap = 1 << 16; s->pv = malloc(s->pcap); s->pc = malloc(s->pcap * 4);
    s->hbits = 22; s->head = malloc(sizeof(int32_t) << s->hbits); memset(s->head, 0xff, sizeof(int32_t) << s->hbits); s->prev = malloc(sizeof(int32_t) * (nmax + 1)); s->ptime = malloc(sizeof(uint32_t) * (nmax + 1)); s->now = 0;
    for (int v = 0; v < SMAX; v++) s->last[v] = -1; s->synced = 0;
    for (int c = 0; c < 4096; c++) s->lt[c] = log(c + e) - log(e);
    return s;
}
void sy_free(Sy *s) { free(s->T); free(s->Tu); free(s->Ta); free(s->G); free(s->Gu); free(s->Ga); free(s->hkey); free(s->hkind); free(s->hcell); free(s->cN); free(s->cU); free(s->ccap); free(s->cptr); free(s->cL); free(s->cT); free(s->cw); free(s->pv); free(s->pc); free(s->head); free(s->prev); free(s->ptime); free(s->BT); free(s->BG); free(s->w2); free(s->pb2); free(s->ab); free(s->tb); free(s->mxv); free(s->zv); free(s); }
int64_t sy_cells(const Sy *s) { int64_t n = 0; for (int64_t c = 0; c < s->ncell; c++) n += s->cN[c] > 0; return n; }
int64_t sy_counts(const Sy *s) { int64_t n = 0; for (int64_t c = 0; c < s->ncell; c++) n += s->cU[c]; return n; }
int64_t sy_cells_of(const Sy *s, int kind) { int64_t n = 0; for (int64_t i = 0; i < s->hcap; i++) if (s->hcell[i] >= 0 && s->hkind[i] == kind && s->cN[s->hcell[i]] > 0) n++; return n; }
int64_t sy_once(const Sy *s) { int64_t n = 0; for (int64_t c = 0; c < s->ncell; c++) n += s->cN[c] == 1; return n; }
void sy_gmin(Sy *s, double g) { s->gmin = g; }
void sy_books(Sy *s, int64_t bper, int bcap, double share) {
    s->bper = bper; s->bcap = bcap; s->share = share; s->nb = 0; s->lnow = 0;
    s->BT = calloc((size_t)s->K * NL * bcap, sizeof(double)); s->BG = calloc((size_t)s->gsize * bcap, sizeof(double)); s->w2 = calloc(bcap + 1, sizeof(double)); s->pb2 = calloc(bcap + 1, sizeof(double)); s->w2[0] = 1.0;
    s->ab = calloc((size_t)SMAX * bcap, sizeof(double)); s->tb = calloc(bcap, sizeof(double)); s->mxv = calloc(bcap, sizeof(double)); s->zv = calloc(bcap, sizeof(double));
}
void sy_mixout(Sy *s, double *mo2, int32_t *to2) { s->mo2 = mo2; s->to2 = to2; }
int sy_nb(Sy *s) { return s->nb; }
double *sy_w2(Sy *s) { return s->w2; }
  double *sy_Ta(Sy *s) { return s->Ta; }  double *sy_Ga(Sy *s) { return s->Ga; }  int64_t sy_now(Sy *s) { return s->now; }  double *sy_T(Sy *s) { return s->T; }  double *sy_Tu(Sy *s) { return s->Tu; }  double *sy_G(Sy *s) { return s->G; }  double *sy_Gu(Sy *s) { return s->Gu; }  int64_t sy_gsize(Sy *s) { return s->gsize; }
static inline uint32_t rhash(const int32_t *x, int64_t end, int M, int hbits) { uint64_t h = 0; for (int i = 1; i <= M; i++) h = h * 0x100000001B3ULL + (uint64_t)(x[end - i] + 1); return (uint32_t)(mix(h) >> (64 - hbits)); }
static inline double step_of(const Sy *s, double used, double acc, double sc) { return s->rule == 0 ? s->eta : s->rule == 1 ? s->eta / sqrt(1.0 + used) : s->eta * sc / sqrt(acc); }
/* reads or answers the bytes x[lo..hi-1] of the stream x (the same array at every call). nl1, nl2: minus the log of the probability each answering cell gave
   to the byte that came; P1, P2 (may be null): all probabilities.  learn = 0: nothing changes (the since clocks, which are context, still move). */
void sy_run(Sy *s, const int32_t *x, int64_t lo, int64_t hi, int learn, double *nl1, double *nl2, double *P1, double *P2) {
    const int S = s->S, K = s->K, M = s->M; double a[SMAX], p[SMAX], a2[SMAX], p2[SMAX];
    static int lev[KMAX], nv[KMAX]; static const uint8_t *vv[KMAX]; static double dv[KMAX][SMAX]; static int64_t sl[KMAX]; static uint64_t keys[KMAX]; static uint8_t rb[KMAX]; static int64_t gix[KMAX][JMAX];
    const double done = log(1 + s->e) - log(s->e);
    for (int64_t t = lo; t < hi; t++) {
        const int y = x[t]; int64_t rq[8]; int nrq = -1; const int64_t tau = s->now + (t - lo);
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int k = 0; k < K; k++) {
            lev[k] = -1;
            if (s->ktype[k] == 2) {                                   /* recall */
                if (nrq < 0) { nrq = 0; if (t >= M) { int32_t q = s->head[rhash(x, t, M, s->hbits)];
                    while (q >= 0 && nrq < 8) { int ok = 1; for (int i = 1; i <= M; i++) if (x[q - i] != x[t - i]) { ok = 0; break; } if (ok) rq[nrq++] = q; q = s->prev[q]; } } }
                const int r = s->knd[k]; if (r >= nrq) continue;
                const int64_t q = rq[r]; int L = M; while (L < LMAX && q - L - 1 >= 0 && x[q - L - 1] == x[t - L - 1]) L++;
                lev[k] = s->age ? agelevel_recall(L, M, tau - (int64_t)s->ptime[q]) : reclevel(L, M); rb[k] = (uint8_t)x[q]; nv[k] = 1; vv[k] = &rb[k]; dv[k][0] = done; a[rb[k]] += s->T[k * NL + lev[k]] * done; continue;
            }
            if (s->ktype[k] == 3) {                                   /* last: the byte the cell of kind knd[k] counted last, said as a cell that has counted once; trust by that cell's level */
                const int32_t c = s->hcell[sl[s->knd[k]]]; if (c < 0 || s->cN[c] < 1) continue;
                lev[k] = s->age ? agelevel_last(s->cU[c], tau - (int64_t)s->cT[c]) : level(s->cN[c], s->cU[c]); rb[k] = s->cL[c]; nv[k] = 1; vv[k] = &rb[k]; dv[k][0] = done; a[rb[k]] += s->T[k * NL + lev[k]] * done; continue;
            }
            if (s->ktype[k] == 6) {                                   /* PLAN_7: the mixture of a cell's three memories, by the cell's own shares */
                const int32_t cl = s->hcell[sl[s->knd[k]]]; if (cl < 0 || s->cN[cl] < 1) continue;
                static double cnt[3][SMAX]; int32_t cm[3]; double wn[3]; const float *w = s->cw + 3 * (int64_t)cl; cm[0] = cl; cm[1] = s->hcell[sl[s->kdel[k][0]]]; cm[2] = s->hcell[sl[s->kdel[k][1]]];
                for (int q = 0; q < 3; q++) { const int32_t c = cm[q]; wn[q] = (double)w[q] / ((c >= 0 ? (double)s->cN[c] : 0.0) + S * s->e); if (c >= 0) { const uint8_t *pv = s->pv + s->cptr[c]; const uint32_t *pc = s->pc + s->cptr[c]; for (int i = 0; i < s->cU[c]; i++) cnt[q][pv[i]] = pc[i]; } }
                const double lun = log((wn[0] + wn[1] + wn[2]) * s->e); const int u = s->cU[cl]; lev[k] = level(s->cN[cl], u); nv[k] = u; vv[k] = s->pv + s->cptr[cl]; const double tk = s->T[k * NL + lev[k]];
                for (int i = 0; i < u; i++) { const int v = vv[k][i]; const double d = log(wn[0] * (cnt[0][v] + s->e) + wn[1] * (cnt[1][v] + s->e) + wn[2] * (cnt[2][v] + s->e)) - lun; dv[k][i] = d; a[v] += tk * d; }
                for (int q = 0; q < 3; q++) { const int32_t c = cm[q]; if (c >= 0) { const uint8_t *pv = s->pv + s->cptr[c]; for (int i = 0; i < s->cU[c]; i++) cnt[q][pv[i]] = 0; } }
                continue;
            }
            uint64_t key = 0;
            if (s->ktype[k] == 0) { for (int i = 0; i < s->knd[k]; i++) { const int64_t q = t - s->kdel[k][i]; key |= (uint64_t)(q >= 0 ? x[q] : 0) << (8 * i); } }
            else { const int v = s->knd[k]; if (s->last[v] < 0) { sl[k] = -1; continue; } const int64_t sn = tau - 1 - s->last[v] > 63 ? 63 : tau - 1 - s->last[v]; key = (uint64_t)sn * 256 + (t >= 1 ? x[t - 1] : 0); }
            keys[k] = key; sl[k] = slot_of(s, key, k); const int32_t c = s->hcell[sl[k]]; if (c < 0 || s->cN[c] < 1 || s->ksil[k]) continue;
            const int u = s->cU[c]; lev[k] = level(s->cN[c], u); nv[k] = u; vv[k] = s->pv + s->cptr[c]; const uint32_t *pc = s->pc + s->cptr[c]; const double tk = s->T[k * NL + lev[k]];
            for (int i = 0; i < u; i++) { const double d = pc[i] < 4096 ? s->lt[pc[i]] : log(pc[i] + s->e) - log(s->e); dv[k][i] = d; a[vv[k][i]] += tk * d; }
        }
        double mx = -1e300, z = 0;
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (nl1) nl1[t - lo] = -log(p[y]);
        if (P1) for (int v = 0; v < S; v++) P1[(t - lo) * S + v] = p[v];
        if (s->gate) {
            for (int v = 0; v < S; v++) a2[v] = a[v];
            for (int k = 0; k < K; k++) if (lev[k] >= 0) {
                double tk = 0;
                for (int j = 0; j < s->J; j++) { const int64_t q = t - 1 - j; gix[k][j] = (((int64_t)j * K + k) * s->GL + (s->GL > 1 ? lev[k] : 0)) * S + (q >= 0 ? x[q] : 0); tk += s->G[gix[k][j]]; }
                if (tk != 0) for (int i = 0; i < nv[k]; i++) a2[vv[k][i]] += tk * dv[k][i];
            }
            mx = -1e300; z = 0;
            for (int v = 0; v < S; v++) if (a2[v] > mx) mx = a2[v];
            for (int v = 0; v < S; v++) { p2[v] = exp(a2[v] - mx); z += p2[v]; }
            for (int v = 0; v < S; v++) p2[v] /= z;
            if (nl2) nl2[t - lo] = -log(p2[y]);
            if (P2) for (int v = 0; v < S; v++) P2[(t - lo) * S + v] = p2[v];
        }
        if (s->bcap > 0 && s->gate) {                    /* PLAN_3: every bookmark answers; the shares' mixture; the shares move */
            const int nb = s->nb, B1 = nb + 1, bc = s->bcap; const double sh = s->share; double mix = 0; int top = 0; double topv = -1;
            s->pb2[0] = p2[y];
            if (nb > 0) {
                static unsigned char mark[SMAX]; static int pos[SMAX]; int tl[SMAX], nt = 0;
                for (int k = 0; k < K; k++) if (lev[k] >= 0) for (int i = 0; i < nv[k]; i++) { const int v = vv[k][i]; if (!mark[v]) { mark[v] = 1; pos[v] = nt; tl[nt++] = v; } }
                if (!mark[y]) { mark[y] = 1; pos[y] = nt; tl[nt++] = y; }
                for (int i = 0; i < nt; i++) { double *r = s->ab + (int64_t)i * bc; for (int b = 0; b < nb; b++) r[b] = 0; }
                for (int k = 0; k < K; k++) if (lev[k] >= 0) {
                    const double *bt = s->BT + (int64_t)(k * NL + lev[k]) * bc; for (int b = 0; b < nb; b++) s->tb[b] = bt[b];
                    for (int j = 0; j < s->J; j++) { const double *bg = s->BG + gix[k][j] * bc; for (int b = 0; b < nb; b++) s->tb[b] += bg[b]; }
                    for (int i = 0; i < nv[k]; i++) { double *r = s->ab + (int64_t)pos[vv[k][i]] * bc; const double d = dv[k][i]; for (int b = 0; b < nb; b++) r[b] += s->tb[b] * d; }
                }
                for (int b = 0; b < nb; b++) s->mxv[b] = nt < S ? 0.0 : -1e300;      /* a value no sender names has the branch sum zero */
                for (int i = 0; i < nt; i++) { const double *r = s->ab + (int64_t)i * bc; for (int b = 0; b < nb; b++) if (r[b] > s->mxv[b]) s->mxv[b] = r[b]; }
                for (int b = 0; b < nb; b++) s->zv[b] = nt < S ? (S - nt) * exp(-s->mxv[b]) : 0.0;
                for (int i = 0; i < nt; i++) { const double *r = s->ab + (int64_t)i * bc; for (int b = 0; b < nb; b++) s->zv[b] += exp(r[b] - s->mxv[b]); }
                { const double *r = s->ab + (int64_t)pos[y] * bc; for (int b = 0; b < nb; b++) s->pb2[b + 1] = exp(r[b] - s->mxv[b]) / s->zv[b]; }
                for (int i = 0; i < nt; i++) mark[tl[i]] = 0;
            }
            for (int b = 0; b < B1; b++) { const double v_ = (1.0 - sh) * s->w2[b] + sh / B1; if (v_ > topv) { topv = v_; top = b; } mix += v_ * s->pb2[b]; s->w2[b] = v_ * s->pb2[b]; }
            for (int b = 0; b < B1; b++) s->w2[b] /= mix;
            if (s->mo2) s->mo2[t - lo] = -log(mix);
            if (s->to2) s->to2[t - lo] = top;
        }
        if (learn) {
            int nact = 0; for (int k = 0; k < K; k++) nact += lev[k] >= 0;
            const double sc = s->spow > 0 && nact > 0 ? pow(12.0 / nact, s->spow) : 1.0;      /* PLAN_8.md: the rule's number divided by a power of the senders that speak, 12 as the unit */
            for (int k = 0; k < K; k++) if (lev[k] >= 0) {
                double g = 0, g2 = 0;
                for (int i = 0; i < nv[k]; i++) { const int v = vv[k][i]; const double d = dv[k][i]; g += ((v == y) - p[v]) * d; if (s->gate) g2 += ((v == y) - p2[v]) * d; }
                const int64_t i = k * NL + lev[k]; s->Ta[i] += g * g; if (s->Ta[i] > s->gmin || s->rule != 2) s->T[i] += step_of(s, s->Tu[i], s->Ta[i], sc) * g; s->Tu[i] += 1.0;
                if (s->gate) for (int j = 0; j < s->J; j++) { const int64_t i2 = gix[k][j]; s->Ga[i2] += g2 * g2; if (s->Ga[i2] > s->gmin || s->rule != 2) s->G[i2] += step_of(s, s->Gu[i2], s->Ga[i2], sc) * g2; s->Gu[i2] += 1.0; }
            }
            for (int k = 0; k < K; k++) if (s->ktype[k] == 6 && lev[k] >= 0) {      /* PLAN_7: the cell's shares move by how well each memory foretold the byte */
                const int32_t cl = s->hcell[sl[s->knd[k]]]; float *w = s->cw + 3 * (int64_t)cl; double o[3], z = 0;
                for (int q = 0; q < 3; q++) { const int32_t c = q == 0 ? cl : s->hcell[sl[s->kdel[k][q - 1]]]; double cy = 0, n = 0; if (c >= 0) { n = s->cN[c]; const uint8_t *pv = s->pv + s->cptr[c]; const uint32_t *pc = s->pc + s->cptr[c]; for (int i = 0; i < s->cU[c]; i++) if (pv[i] == y) { cy = pc[i]; break; } }
                    o[q] = ((1.0 - SIG) * (double)w[q] + SIG / 3.0) * (cy + s->e) / (n + S * s->e); z += o[q]; }
                for (int q = 0; q < 3; q++) w[q] = (float)(o[q] / z);
            }
            for (int k = 0; k < K; k++) if (s->ktype[k] < 2 && sl[k] >= 0) {
                int32_t c = s->hcell[sl[k]];
                if (c < 0) {
                    if (s->birth > 0 && p[y] >= s->birth) continue;
                    if ((s->ncell + 1) * 10 > s->hcap * 6) { grow(s); for (int j = 0; j < K; j++) if (s->ktype[j] < 2 && sl[j] >= 0) sl[j] = slot_of(s, keys[j], j); }
                    c = new_cell(s); s->hkey[sl[k]] = keys[k]; s->hkind[sl[k]] = (uint16_t)k; s->hcell[sl[k]] = c;
                    for (int j = k + 1; j < K; j++) if (s->ktype[j] < 2 && sl[j] >= 0) sl[j] = slot_of(s, keys[j], j);      /* a slot found empty may now be taken */
                }
                else if (s->birth > 0 && s->cN[c] == 0 && p[y] >= s->birth) continue;
                count(s, c, y, tau); if (s->kcap[k] > 0 && s->cN[c] >= (uint32_t)s->kcap[k]) halve(s, c);
            }
            if (M > 0 && t >= M && t <= s->nmax) { const uint32_t h = rhash(x, t, M, s->hbits); s->prev[t] = s->head[h]; s->head[h] = (int32_t)t; s->ptime[t] = (uint32_t)tau; }      /* the place t: the run that ended before it, and the byte x[t] just read */
            if (s->bcap > 0) { s->lnow++; if (s->lnow % s->bper == 0 && s->nb < s->bcap) { const int b = s->nb, bc = s->bcap; for (int64_t i = 0; i < (int64_t)K * NL; i++) s->BT[i * bc + b] = s->T[i]; for (int64_t i = 0; i < s->gsize; i++) s->BG[i * bc + b] = s->G[i]; s->w2[b + 1] = 0; s->nb++; } }      /* PLAN_3: a bookmark */
        }
        s->last[y] = tau;
    }
    s->now += hi - lo;
}
