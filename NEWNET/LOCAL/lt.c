/* lt.c - a local tissue (24 Nov 2026). Event-driven; a whole phase of bytes runs in one call.
   Locality: every cell has bounded fan-in (P proximal + S distal slots) and bounded fan-out (FO distal, FOP proximal, FOO to the
   output cells); a connection is made only within a radius (ring coordinates fixed at birth); only region 0 reads the input lines;
   the output cells read only the boundary region (outr: 0 = last region, 1 = region 0).
   Regions 0..R-1, each C minicolumns of M cells. Columns of region r are chosen by proximal input (region 0: the byte's input line;
   r > 0: region r-1's columns at this byte) and local inhibition (KB winners in each block of B columns). The cell of a winning
   column is chosen by its distal prediction (predicted > matching > least used). Distal sources (the region's pool, one byte late):
   lower part (r = 0: region 0's columns; r > 0: region r-1's cells), lat: own cells, td: region r+1's cells.
   Distal learning (local, by the arriving activity): the chosen cell strengthens its active slots and grows slots to active sources
   (up to NG active), replacing its weakest inactive slots; predicted cells not chosen weaken their active slots.
   Output cell v reads up to NO boundary columns (their graded prediction for the next byte); delta rule on its own target.
   Instrument (not the net): p = softmax(beta*o + c), beta and c learned online by log loss while learning; nats are -log p[y]. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define V 128
enum { I_R, I_C, I_M, I_S, I_P, I_B, I_KB, I_NG, I_FO, I_FOP, I_NO, I_FOO, I_LAT, I_TD, I_OUTR, I_SEED, I_GRADED, I_NI };
enum { D_RHOP, D_RHOD, D_RHOO, D_ETA, D_ETAM, D_WON, D_WNEW, D_THON, D_THM, D_ETAO, D_ETAB, D_ETAT, D_ND };

typedef struct {
    int C, M, S, P, n, npool, lat_off, td_off, nlow;
    int *psrc; float *pw;                 /* proximal [C*P] */
    int *ssrc; float *sw;                 /* distal [n*S] */
    int *olist, *on;                      /* outgoing distal synapses of each pool unit [npool*FO] */
    float *ppos;                          /* pool unit positions */
    int *pfo;                             /* proximal fan-out of the source units (region below / input lines) */
    float *use;
    uint8_t *col_on, *cell_on, *mark;
    int *acols, nacols, *acells, nacells;
    int *pool, npool_act;                 /* pool units active one byte back */
    int *cnt_on, *cnt_m, *touched, ntouched, *nref;
    float *ov;
} Reg;

typedef struct {
    int ip[I_NI]; double dp[D_ND];
    int R; Reg *g; float *ipos;
    int *osrc; float *ow, *opos; int *ocnt; double b[V], o[V], beta, ct[V];
    float *pcol; int *pcl, npcl;
    int have; uint64_t rs;
    int *cand, *tmp;
    void *snap; size_t snapsz;
} Net;

static inline uint64_t xr(Net *h) { h->rs ^= h->rs << 13; h->rs ^= h->rs >> 7; h->rs ^= h->rs << 17; return h->rs; }
static inline double ur(Net *h) { return (xr(h) >> 11) * (1.0 / 9007199254740992.0); }
static inline float rd(float a, float b) { float d = fabsf(a - b); return d > 0.5f ? 1.0f - d : d; }

/* ---- dynamic state snapshot (for tests with learning off) ---- */
#define SNAP_FIELDS(X) \
    for (int r = 0; r < h->R; r++) { Reg *g = h->g + r; \
        X(g->col_on, g->C); X(g->cell_on, g->n); X(g->acols, g->C); X(g->acells, g->C); X(g->pool, g->npool); \
        X(g->mark, g->npool); X(g->cnt_on, g->n); X(g->cnt_m, g->n); X(g->touched, g->n); X(g->nref, g->C); \
        X(&g->nacols, 1); X(&g->nacells, 1); X(&g->npool_act, 1); X(&g->ntouched, 1); } \
    X(h->o, V); X(h->pcol, h->ip[I_C]); X(h->pcl, h->ip[I_C]); X(&h->npcl, 1); X(&h->have, 1); X(&h->rs, 1);

#define SZ(p, k) sz += sizeof(*(p)) * (size_t)(k);
#define SV(p, k) { memcpy(q, p, sizeof(*(p)) * (size_t)(k)); q += sizeof(*(p)) * (size_t)(k); }
#define LD(p, k) { memcpy(p, q, sizeof(*(p)) * (size_t)(k)); q += sizeof(*(p)) * (size_t)(k); }
void lt_snapshot(Net *h) { size_t sz = 0; SNAP_FIELDS(SZ); if (!h->snap) { h->snap = malloc(sz); h->snapsz = sz; } char *q = h->snap; SNAP_FIELDS(SV); }
void lt_restore(Net *h) { char *q = h->snap; SNAP_FIELDS(LD); }

/* ---- birth ---- */
Net *lt_new(const int *ip, const double *dp) {
    Net *h = calloc(1, sizeof(Net)); memcpy(h->ip, ip, sizeof(int) * I_NI); memcpy(h->dp, dp, sizeof(double) * D_ND);
    h->rs = 88172645463325252ULL ^ ((uint64_t)ip[I_SEED] * 2654435761ULL); for (int i = 0; i < 20; i++) xr(h);
    int R = h->R = ip[I_R], C = ip[I_C], M = ip[I_M], S = ip[I_S], P = ip[I_P], FO = ip[I_FO];
    h->g = calloc(R, sizeof(Reg)); h->ipos = malloc(sizeof(float) * V);
    for (int v = 0; v < V; v++) h->ipos[v] = (float)ur(h);
    int maxpool = 0;
    for (int r = 0; r < R; r++) {
        Reg *g = h->g + r; g->C = C; g->M = M; g->S = S; g->P = P; g->n = C * M;
        g->nlow = r == 0 ? C : C * M;
        g->lat_off = g->nlow; g->td_off = g->lat_off + (ip[I_LAT] ? g->n : 0);
        g->npool = g->td_off + (ip[I_TD] && r + 1 < R ? C * M : 0);
        if (g->npool > maxpool) maxpool = g->npool;
        g->psrc = malloc(sizeof(int) * C * P); g->pw = malloc(sizeof(float) * C * P);
        g->ssrc = malloc(sizeof(int) * g->n * S); g->sw = calloc((size_t)g->n * S, sizeof(float));
        for (int i = 0; i < g->n * S; i++) g->ssrc[i] = -1;
        g->olist = malloc(sizeof(int) * (size_t)g->npool * FO); g->on = calloc(g->npool, sizeof(int));
        g->ppos = malloc(sizeof(float) * g->npool);
        for (int u = 0; u < g->npool; u++) {
            int k = u < g->nlow ? (r == 0 ? u : u / M) : u < g->td_off ? (u - g->lat_off) / M : (u - g->td_off) / M;
            g->ppos[u] = (k + 0.5f) / C;
        }
        g->use = calloc(g->n, sizeof(float)); g->col_on = calloc(C, 1); g->cell_on = calloc(g->n, 1); g->mark = calloc(g->npool, 1);
        g->acols = malloc(sizeof(int) * C); g->acells = malloc(sizeof(int) * C); g->pool = malloc(sizeof(int) * g->npool);
        g->cnt_on = calloc(g->n, sizeof(int)); g->cnt_m = calloc(g->n, sizeof(int)); g->touched = malloc(sizeof(int) * g->n);
        g->nref = calloc(C, sizeof(int)); g->ov = malloc(sizeof(float) * C);
        /* proximal wiring within rho_p, source fan-out <= FOP */
        int nsrc = r == 0 ? V : C; g->pfo = calloc(nsrc, sizeof(int));
        int *el = malloc(sizeof(int) * nsrc);
        for (int c = 0; c < C; c++) {
            float x = (c + 0.5f) / C; int ne = 0;
            for (int u = 0; u < nsrc; u++) {
                float pu = r == 0 ? h->ipos[u] : (u + 0.5f) / C;
                if (rd(x, pu) <= dp[D_RHOP] && g->pfo[u] < ip[I_FOP]) el[ne++] = u;
            }
            for (int k = 0; k < P; k++) {
                if (k < ne) { int j = k + (int)(xr(h) % (uint64_t)(ne - k)); int t = el[k]; el[k] = el[j]; el[j] = t;
                              g->psrc[c * P + k] = el[k]; g->pw[c * P + k] = (float)(0.5 + 0.5 * ur(h)); g->pfo[el[k]]++; }
                else { g->psrc[c * P + k] = -1; g->pw[c * P + k] = 0; }
            }
        }
        free(el);
    }
    h->cand = malloc(sizeof(int) * maxpool); h->tmp = malloc(sizeof(int) * maxpool);
    int NO = ip[I_NO];
    h->osrc = malloc(sizeof(int) * V * NO); h->ow = calloc(V * NO, sizeof(float)); h->opos = malloc(sizeof(float) * V);
    for (int i = 0; i < V * NO; i++) h->osrc[i] = -1;
    for (int v = 0; v < V; v++) h->opos[v] = (float)ur(h);
    h->ocnt = calloc(C, sizeof(int)); h->pcol = calloc(C, sizeof(float)); h->pcl = malloc(sizeof(int) * C);
    h->beta = 1.0;
    return h;
}

static void remove_out(Reg *g, int u, int e, int FO) {
    int *L = g->olist + (size_t)u * FO;
    for (int k = 0; k < g->on[u]; k++) if (L[k] == e) { L[k] = L[--g->on[u]]; return; }
}

/* predictions for the next byte from the pool units active now; also the boundary columns' graded prediction */
static void predict(Net *h, int r) {
    Reg *g = h->g + r; int S = g->S, FO = h->ip[I_FO], NG = h->ip[I_NG];
    for (int k = 0; k < g->ntouched; k++) { int i = g->touched[k]; g->cnt_on[i] = g->cnt_m[i] = 0; }
    g->ntouched = 0;
    double won = h->dp[D_WON];
    for (int a = 0; a < g->npool_act; a++) {
        int u = g->pool[a]; int *L = g->olist + (size_t)u * FO;
        for (int k = 0; k < g->on[u]; k++) {
            int e = L[k], i = e / S; float w = g->sw[e];
            if (w <= 0) continue;
            if (!g->cnt_m[i] && !g->cnt_on[i]) g->touched[g->ntouched++] = i;
            g->cnt_m[i]++; if (w >= won) g->cnt_on[i]++;
        }
    }
    double rho = h->dp[D_RHOD];
    for (int c = 0; c < g->C; c++) {
        float x = (c + 0.5f) / g->C; int n = 0;
        for (int a = 0; a < g->npool_act; a++) if (rd(x, g->ppos[g->pool[a]]) <= rho) n++;
        g->nref[c] = n < NG ? n : NG;
    }
}

static void boundary(Net *h) {
    int ob = h->ip[I_OUTR] ? 0 : h->R - 1; Reg *g = h->g + ob; double th = h->dp[D_THON];
    for (int k = 0; k < h->npcl; k++) h->pcol[h->pcl[k]] = 0; h->npcl = 0;
    for (int k = 0; k < g->ntouched; k++) {
        int i = g->touched[k], c = i / g->M; if (!g->nref[c]) continue;
        double gr = g->cnt_on[i] / (th * g->nref[c]);
        if (!h->ip[I_GRADED]) gr = gr >= 1.0 ? 1.0 : 0.0;
        if (gr > 1) gr = 1; if (gr <= 0) continue;
        if (h->pcol[c] == 0) h->pcl[h->npcl++] = c;
        if (gr > h->pcol[c]) h->pcol[c] = (float)gr;
    }
    int NO = h->ip[I_NO];
    for (int v = 0; v < V; v++) {
        double s = h->b[v];
        for (int k = 0; k < NO; k++) { int c = h->osrc[v * NO + k]; if (c >= 0) s += h->ow[v * NO + k] * h->pcol[c]; }
        h->o[v] = s;
    }
}

/* one region at this byte: columns, cells, distal learning */
static void region_step(Net *h, int r, int byte, int learn) {
    Reg *g = h->g + r; int C = g->C, M = g->M, S = g->S, P = g->P, FO = h->ip[I_FO], NG = h->ip[I_NG];
    for (int k = 0; k < g->nacells; k++) g->cell_on[g->acells[k]] = 0;
    for (int k = 0; k < g->nacols; k++) g->col_on[g->acols[k]] = 0;
    g->nacols = g->nacells = 0;
    /* proximal overlap and local inhibition */
    const uint8_t *below = r ? h->g[r - 1].col_on : NULL;
    for (int c = 0; c < C; c++) {
        float s = 0; const int *ps = g->psrc + c * P; const float *pw = g->pw + c * P;
        for (int k = 0; k < P; k++) { int u = ps[k]; if (u < 0) continue; if (r ? below[u] : u == byte) s += pw[k]; }
        g->ov[c] = s;
    }
    int B = h->ip[I_B], KB = h->ip[I_KB];
    for (int b0 = 0; b0 < C; b0 += B) {
        int b1 = b0 + B < C ? b0 + B : C;
        for (int k = 0; k < KB; k++) {
            int best = -1; float bv = 0;
            for (int c = b0; c < b1; c++) if (!g->col_on[c] && g->ov[c] > bv) { bv = g->ov[c]; best = c; }
            if (best < 0) break;
            g->col_on[best] = 1; g->acols[g->nacols++] = best;
        }
    }
    double thon = h->dp[D_THON], thm = h->dp[D_THM];
    /* choose cells */
    for (int a = 0; a < g->nacols; a++) {
        int c = g->acols[a], best = c * M; double bs = -1e300; int nr = g->nref[c];
        for (int m = 0; m < M; m++) {
            int i = c * M + m; double sc;
            if (nr && g->cnt_on[i] >= thon * nr) sc = 1e6 + g->cnt_on[i];
            else if (nr && g->cnt_m[i] >= thm * nr) sc = 1e3 + g->cnt_m[i];
            else sc = -g->use[i];
            if (sc > bs) { bs = sc; best = i; }
        }
        g->cell_on[best] = 1; g->acells[g->nacells++] = best;
    }
    if (!learn) return;
    double eta = h->dp[D_ETA], etam = h->dp[D_ETAM], wnew = h->dp[D_WNEW], rho = h->dp[D_RHOD];
    /* predicted cells that did not fire weaken their active slots */
    for (int k = 0; k < g->ntouched; k++) {
        int i = g->touched[k], c = i / M, nr = g->nref[c];
        if (!nr || g->cnt_on[i] < thon * nr || g->cell_on[i]) continue;
        for (int s = 0; s < S; s++) { int e = i * S + s, u = g->ssrc[e]; if (u >= 0 && g->mark[u] && g->sw[e] > 0) g->sw[e] -= (float)(etam * g->sw[e]); }
    }
    /* the chosen cells learn and grow */
    for (int a = 0; a < g->nacells; a++) {
        int i = g->acells[a], nact = 0; float x = (i / M + 0.5f) / C;
        g->use[i] += 1;
        for (int s = 0; s < S; s++) { int e = i * S + s, u = g->ssrc[e]; if (u >= 0 && g->mark[u]) { g->sw[e] += (float)(eta * (1 - g->sw[e])); nact++; } }
        int need = NG - nact; if (need <= 0 || !g->npool_act) continue;
        int nc = 0;
        for (int q = 0; q < g->npool_act; q++) {
            int u = g->pool[q]; if (g->on[u] >= FO || rd(x, g->ppos[u]) > rho) continue;
            int have = 0; for (int s = 0; s < S; s++) if (g->ssrc[i * S + s] == u) { have = 1; break; }
            if (!have) h->cand[nc++] = u;
        }
        if (nc > need) { for (int t = 0; t < need; t++) { int j = t + (int)(xr(h) % (uint64_t)(nc - t)); int z = h->cand[t]; h->cand[t] = h->cand[j]; h->cand[j] = z; } nc = need; }
        for (int t = 0; t < nc; t++) {                                  /* weakest inactive slot (empty first) */
            int bs = -1; float bw = 2;
            for (int s = 0; s < S; s++) { int e = i * S + s, u = g->ssrc[e]; if (u >= 0 && g->mark[u]) continue; float w = u < 0 ? -1 : g->sw[e]; if (w < bw) { bw = w; bs = s; } }
            if (bs < 0) break;
            int e = i * S + bs, old = g->ssrc[e];
            if (old >= 0) remove_out(g, old, e, FO);
            int u = h->cand[t]; g->ssrc[e] = u; g->sw[e] = (float)wnew; g->olist[(size_t)u * FO + g->on[u]++] = e;
        }
    }
}

static void build_pools(Net *h) {
    for (int r = 0; r < h->R; r++) {
        Reg *g = h->g + r; int n = 0;
        for (int k = 0; k < g->npool_act; k++) g->mark[g->pool[k]] = 0;
        if (r == 0) for (int k = 0; k < g->nacols; k++) g->pool[n++] = g->acols[k];
        else { Reg *d = h->g + r - 1; for (int k = 0; k < d->nacells; k++) g->pool[n++] = d->acells[k]; }
        if (h->ip[I_LAT]) for (int k = 0; k < g->nacells; k++) g->pool[n++] = g->lat_off + g->acells[k];
        if (h->ip[I_TD] && r + 1 < h->R) { Reg *u = h->g + r + 1; for (int k = 0; k < u->nacells; k++) g->pool[n++] = g->td_off + u->acells[k]; }
        g->npool_act = n;
        for (int k = 0; k < n; k++) g->mark[g->pool[k]] = 1;
    }
}

/* read n bytes. learn: tissue, output and instrument learn. stats: [0] nats summed, [1] right, [2] scored */
void lt_run(Net *h, const uint8_t *x, int n, int learn, double *stats) {
    int NO = h->ip[I_NO], ob = h->ip[I_OUTR] ? 0 : h->R - 1, FOO = h->ip[I_FOO];
    double eo = h->dp[D_ETAO], eb = h->dp[D_ETAB], et = h->dp[D_ETAT], rhoo = h->dp[D_RHOO];
    double p[V];
    for (int t = 0; t < n; t++) {
        int y = x[t] & 127;
        if (h->have) {                                                   /* score and learn the prediction made one byte ago */
            double mx = -1e300; int arg = 0;
            for (int v = 0; v < V; v++) { double z = h->beta * h->o[v] + h->ct[v]; p[v] = z; if (z > mx) mx = z; if (h->o[v] > h->o[arg]) arg = v; }
            double Z = 0; for (int v = 0; v < V; v++) { p[v] = exp(p[v] - mx); Z += p[v]; }
            for (int v = 0; v < V; v++) p[v] /= Z;
            stats[0] += -log(p[y] + 1e-12); stats[1] += arg == y; stats[2] += 1;
            if (learn) {
                double gb = 0; for (int v = 0; v < V; v++) { double gv = (v == y) - p[v]; gb += gv * h->o[v]; h->ct[v] += et * gv; }
                h->beta += et * gb;
                int na = h->npcl;
                for (int v = 0; v < V; v++) {
                    if (v != y && h->o[v] <= 0) continue;
                    double e = (v == y) - h->o[v];
                    h->b[v] += eb * e;
                    if (!na) continue;
                    for (int k = 0; k < NO; k++) { int c = h->osrc[v * NO + k]; if (c >= 0 && h->pcol[c] > 0) h->ow[v * NO + k] += (float)(eo * e * h->pcol[c] / na); }
                }
            }
        }
        for (int r = 0; r < h->R; r++) region_step(h, r, y, learn);
        if (learn) {                                                     /* output cell y grows synapses to the boundary columns of y */
            Reg *g = h->g + ob; int have = 0;
            for (int k = 0; k < NO; k++) have += h->osrc[y * NO + k] >= 0;
            for (int a = 0; a < g->nacols && have < NO; a++) {
                int c = g->acols[a]; if (h->ocnt[c] >= FOO || rd(h->opos[y], (c + 0.5f) / g->C) > rhoo) continue;
                int dup = 0; for (int k = 0; k < NO; k++) if (h->osrc[y * NO + k] == c) dup = 1;
                if (dup) continue;
                for (int k = 0; k < NO; k++) if (h->osrc[y * NO + k] < 0) { h->osrc[y * NO + k] = c; h->ow[y * NO + k] = 0; h->ocnt[c]++; have++; break; }
            }
        }
        build_pools(h);
        for (int r = 0; r < h->R; r++) predict(h, r);
        boundary(h);
        h->have = 1;
    }
}

/* locality audit: [0] max fan-in of a tissue cell (proximal + distal in use), [1] max distal fan-out, [2] max proximal fan-out,
   [3] max output fan-in, [4] max fan-out to outputs, [5] distal synapses in use, [6] cells ever used */
void lt_audit(Net *h, long *a) {
    memset(a, 0, sizeof(long) * 7);
    for (int r = 0; r < h->R; r++) {
        Reg *g = h->g + r; int np = 0; for (int k = 0; k < g->P; k++) np += g->psrc[k] >= 0;
        for (int i = 0; i < g->n; i++) { int d = 0; for (int s = 0; s < g->S; s++) d += g->ssrc[i * g->S + s] >= 0; if (np + d > a[0]) a[0] = np + d; a[5] += d; a[6] += g->use[i] > 0; }
        for (int u = 0; u < g->npool; u++) if (g->on[u] > a[1]) a[1] = g->on[u];
        int nsrc = r == 0 ? V : g->C; for (int u = 0; u < nsrc; u++) if (g->pfo[u] > a[2]) a[2] = g->pfo[u];
    }
    int NO = h->ip[I_NO];
    for (int v = 0; v < V; v++) { int k2 = 0; for (int k = 0; k < NO; k++) k2 += h->osrc[v * NO + k] >= 0; if (k2 > a[3]) a[3] = k2; }
    for (int c = 0; c < h->ip[I_C]; c++) if (h->ocnt[c] > a[4]) a[4] = h->ocnt[c];
}

/* debug: [nacols, npool_act, sum nref, predicted cells for next byte, synapses, boundary cols predicted, answer] */
void lt_dbg(Net *h, int r, long *a) {
    Reg *g = h->g + r; a[0] = g->nacols; a[1] = g->npool_act; a[2] = 0; a[3] = 0; a[4] = 0;
    for (int k = 0; k < g->nacols; k++) a[2] += g->nref[g->acols[k]];
    for (int k = 0; k < g->ntouched; k++) { int i = g->touched[k], c = i / g->M; if (g->nref[c] && g->cnt_on[i] >= h->dp[D_THON] * g->nref[c]) a[3]++; }
    a[5] = h->npcl; { int b = 0; for (int v = 1; v < V; v++) if (h->o[v] > h->o[b]) b = v; a[6] = b; }
    for (int i = 0; i < g->n * g->S; i++) a[4] += g->ssrc[i] >= 0;
}
