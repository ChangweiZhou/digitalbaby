
#include <math.h>
#include <stdint.h>
#define NL 40
#define MAXS 64
#define MAXK 512
#define MAXP 70000
#define HP 16
#define R 4
typedef struct { int64_t S, K, nplace, END, ncell, cap; const int32_t *nd, *del; const int64_t *off; int32_t *map; float *C, *N, *U; double *T, *Tu; double eta, e; int64_t flat, shrink;
                 int64_t KZ, ZTOT, ncell2, cap2; int32_t *map2; float *C2, *N2, *U2; double *T2, *Tu2; } sys_t;
static int64_t ids[MAXK], keyv[MAXK]; static int tix[MAXK]; static double sv[MAXK][MAXS]; static double lt[4096]; static double lte = -1;
static int8_t rk[MAXK][MAXS]; static int n2, m2; static struct { int kx, t; double sr[R + 1]; } act2[MAXP]; static struct { int kx; int64_t id; } all2[MAXP];
static inline int level(float n, float u, int flat) {
    if (flat) return 0;
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
static inline double lg(float c, double e) { return c < 4096 ? lt[(int)c] : log(c + e); }
static void look(sys_t *q, const int32_t *h, int make, double *p) {
    const int64_t S = q->S; double a[MAXS], mx = -1e300, z = 0; int pl = 0;
    if (lte != q->e) { for (int c = 0; c < 4096; c++) lt[c] = log(c + q->e); lte = q->e; }
    if (q->nplace > 1) { while (pl < q->nplace - 1 && h[-1 - pl] != q->END) pl++; }
    for (int v = 0; v < S; v++) a[v] = 0;
    for (int64_t k = 0; k < q->K; k++) {
        int64_t key = 0; for (int j = 0; j < q->nd[k]; j++) key = key * S + h[-q->del[k * 4 + j]];
        keyv[k] = key; int64_t id = q->map[q->off[k] + key];
        if (id < 0 && make) { id = q->ncell++; q->map[q->off[k] + key] = (int32_t)id; }
        ids[k] = id; tix[k] = -1; if (id < 0 || q->N[id] < 1) continue;
        const float *c = q->C + id * S; tix[k] = (int)((k * q->nplace + pl) * NL + level(q->N[id], q->U[id], (int)q->flat));
        const double tk = q->T[tix[k]], den = log(q->N[id] + S * q->e);
        for (int v = 0; v < S; v++) { const double s = lg(c[v], q->e) - den; sv[k][v] = s; a[v] += tk * s; }
    }
    n2 = 0; m2 = 0;
    if (q->KZ) for (int64_t kx = 0; kx < q->K; kx++) {
        const int64_t id = ids[kx]; if (id < 0 || q->N[id] < 1 || q->U[id] < 2 || q->U[id] > R) continue;
        const float *c = q->C + id * S; int8_t *r = rk[kx]; int f[R], nf = 0;
        for (int v = 0; v < S; v++) { r[v] = R; if (c[v] > 0) f[nf++] = v; }
        for (int i = 1; i < nf; i++) { int x = f[i], j = i; while (j > 0 && c[f[j - 1]] < c[x]) { f[j] = f[j - 1]; j--; } f[j] = x; }
        for (int j = 0; j < nf; j++) r[f[j]] = (int8_t)j;
        const double lognf = log((double)(S - nf));
        for (int64_t kz = 0; kz < q->KZ; kz++) {
            const int64_t slot = kx * q->ZTOT + q->off[kz] + keyv[kz]; int64_t id2 = q->map2[slot];
            if (id2 < 0 && make) { id2 = q->ncell2++; q->map2[slot] = (int32_t)id2; }
            all2[m2].kx = (int)kx; all2[m2].id = id2; m2++;
            if (id2 < 0 || q->N2[id2] < 1) continue;
            const float *c2 = q->C2 + id2 * (R + 1); const int t = (int)((kx * q->KZ + kz) * NL + level(q->N2[id2], q->U2[id2], 0));
            const double tk = q->T2[t], den = log(q->N2[id2] + (R + 1) * q->e); double *sr = act2[n2].sr;
            for (int j = 0; j < R; j++) sr[j] = lg(c2[j], q->e) - den;
            sr[R] = lg(c2[R], q->e) - den - lognf;
            for (int v = 0; v < S; v++) a[v] += tk * sr[r[v]];
            act2[n2].kx = (int)kx; act2[n2].t = t; n2++;
        }
    }
    for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
    for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
    for (int v = 0; v < S; v++) p[v] /= z;
}
int64_t steps(sys_t *q, int64_t n, const int32_t *x, int64_t learn_c, int64_t learn_t, double *P) {
    const int64_t S = q->S; double p[MAXS];
    for (int64_t t = 0; t < n; t++) {
        if (learn_c && (q->ncell + q->K > q->cap || q->ncell2 + q->K * q->KZ > q->cap2)) return t;
        look(q, x + t, (int)learn_c, p); const int y = x[t];
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (learn_t) {
            for (int64_t k = 0; k < q->K; k++) if (tix[k] >= 0) {
                double g = sv[k][y]; for (int v = 0; v < S; v++) g -= p[v] * sv[k][v];
                q->T[tix[k]] += q->eta * g; q->Tu[tix[k]] += 1.0;
            }
            for (int i = 0; i < n2; i++) {
                const int8_t *r = rk[act2[i].kx]; const double *sr = act2[i].sr; double g = sr[r[y]]; for (int v = 0; v < S; v++) g -= p[v] * sr[r[v]];
                q->T2[act2[i].t] += q->eta * g; q->Tu2[act2[i].t] += 1.0;
            }
        }
        if (learn_c) {
            for (int i = 0; i < m2; i++) { float *c2 = q->C2 + all2[i].id * (R + 1); const int j = rk[all2[i].kx][y]; if (c2[j] == 0) q->U2[all2[i].id] += 1; c2[j] += 1; q->N2[all2[i].id] += 1; }
            for (int64_t k = 0; k < q->K; k++) { float *c = q->C + ids[k] * S; if (c[y] == 0) q->U[ids[k]] += 1; c[y] += 1; q->N[ids[k]] += 1; }
        }
    }
    return n;
}
void answers(sys_t *q, int64_t n, const int32_t *lines, int64_t L, int64_t from, int64_t cnt, int64_t own, int32_t *out, double *acc) {
    int32_t buf[HP + 64]; double p[MAXS];
    for (int64_t i = 0; i < n; i++) {
        for (int j = 0; j < HP; j++) buf[j] = (int32_t)q->END;
        for (int j = 0; j < L; j++) buf[HP + j] = lines[i * L + j];
        for (int64_t c = 0; c < cnt; c++) {
            look(q, buf + HP + from + c, 0, p); int b = 0; for (int v = 1; v < q->S; v++) if (p[v] > p[b]) b = v;
            out[i * cnt + c] = b; if (own) buf[HP + from + c] = b;
            if (acc) for (int64_t k = 0; k < q->K; k++) if (tix[k] >= 0) { acc[(c * q->K + k) * 2] += q->T[tix[k]]; acc[(c * q->K + k) * 2 + 1] += 1; }
        }
    }
}
/* the second-layer cells active before the byte at h, for inspection: pair index and trust index, up to cap of them; returns how many */
int64_t active2(sys_t *q, const int32_t *h, int64_t cap, int32_t *t_out, double *sr_out) {
    double p[MAXS]; look(q, h, 0, p); int64_t n = n2 < cap ? n2 : cap;
    for (int64_t i = 0; i < n; i++) { t_out[i] = act2[i].t; for (int j = 0; j <= R; j++) sr_out[i * (R + 1) + j] = act2[i].sr[j]; }
    return n;
}
