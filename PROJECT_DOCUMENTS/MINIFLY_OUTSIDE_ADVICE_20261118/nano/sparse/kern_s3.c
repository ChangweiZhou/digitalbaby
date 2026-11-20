
#include <math.h>
#include <stdint.h>
#define NL 40
#define MAXS 64
#define MAXK 2048
#define HP 16
typedef struct { int64_t S, K, nplace, END, ncell, cap; const int32_t *nd, *del; const int64_t *off; int32_t *map; float *C, *N, *U; double *T, *Tu; double eta, e; int64_t flat, shrink; } sys_t;
static int64_t ids[MAXK]; static int tix[MAXK]; static double sv[MAXK][MAXS]; static double lt[4096]; static double lte = -1;
static inline int level(float n, float u, int flat) {
    if (flat) return 0;
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* h[-d] is the byte d places back. The active cells that have counted speak; p receives the answering cell's probabilities. */
static void look(sys_t *q, const int32_t *h, int make, double *p, const float *sx, int xl) {
    const int64_t S = q->S; double a[MAXS], mx = -1e300, z = 0; int pl = 0;
    if (lte != q->e) { for (int c = 0; c < 4096; c++) lt[c] = log(c + q->e); lte = q->e; }
    if (q->nplace > 1) { while (pl < q->nplace - 1 && h[-1 - pl] != q->END) pl++; }
    for (int v = 0; v < S; v++) a[v] = 0;
    for (int64_t k = 0; k < q->K; k++) {
        int64_t key = 0; for (int j = 0; j < q->nd[k]; j++) key = key * S + h[-q->del[k * 4 + j]];
        int64_t id = q->map[q->off[k] + key];
        if (id < 0 && make) { id = q->ncell++; q->map[q->off[k] + key] = (int32_t)id; }
        ids[k] = id; tix[k] = -1; if (id < 0 || q->N[id] < 1) continue;
        const float *c = q->C + id * S; tix[k] = (int)((k * q->nplace + pl) * NL + level(q->N[id], q->U[id], (int)q->flat));
        const double tk = q->T[tix[k]], den = log(q->N[id] + S * q->e);
        for (int v = 0; v < S; v++) { const double s = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + q->e)) - den; sv[k][v] = s; a[v] += tk * s; }
    }
    if (sx) { const double tk = q->T[q->K * q->nplace * NL + xl]; for (int v = 0; v < S; v++) a[v] += tk * sx[v]; }      /* one more sender: its message is given */
    for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
    for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
    for (int v = 0; v < S; v++) p[v] /= z;
}
/* x[t] is named from the bytes before it (HP bytes before x[0] are valid). Returns how many bytes were done (fewer than n if cells ran out). */
int64_t steps(sys_t *q, int64_t n, const int32_t *x, int64_t learn_c, int64_t learn_t, double *P, const float *X, const int32_t *xlev) {
    const int64_t S = q->S; double p[MAXS];
    for (int64_t t = 0; t < n; t++) {
        if (learn_c && q->ncell + q->K > q->cap) return t;
        look(q, x + t, (int)learn_c, p, X ? X + t * S : 0, X ? xlev[t] : 0); const int y = x[t];
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (learn_t) for (int64_t k = 0; k < q->K; k++) if (tix[k] >= 0) {
            double g = sv[k][y]; for (int v = 0; v < S; v++) g -= p[v] * sv[k][v];
            q->T[tix[k]] += (q->shrink ? q->eta / sqrt(1.0 + q->Tu[tix[k]]) : q->eta) * g; q->Tu[tix[k]] += 1.0;
        }
        if (learn_t && X) { const float *sx = X + t * S; double g = sx[y]; for (int v = 0; v < S; v++) g -= p[v] * sx[v]; const int64_t i = q->K * q->nplace * NL + xlev[t]; q->T[i] += q->eta * g; q->Tu[i] += 1.0; }
        if (learn_c) for (int64_t k = 0; k < q->K; k++) { float *c = q->C + ids[k] * S; if (c[y] == 0) q->U[ids[k]] += 1; c[y] += 1; q->N[ids[k]] += 1; }
    }
    return n;
}
/* For each line (L bytes; before it only line ends): the bytes at places from..from+cnt-1 are named; own=1: the system's own byte is written in
   place of the line's before the next is named. acc (cnt x K x 2, may be null) receives the sum of the trust used and the number of times active. */
void answers(sys_t *q, int64_t n, const int32_t *lines, int64_t L, int64_t from, int64_t cnt, int64_t own, int32_t *out, double *acc, const float *X, const int32_t *xlev) {
    int32_t buf[HP + 64]; double p[MAXS];
    for (int64_t i = 0; i < n; i++) {
        for (int j = 0; j < HP; j++) buf[j] = (int32_t)q->END;
        for (int j = 0; j < L; j++) buf[HP + j] = lines[i * L + j];
        for (int64_t c = 0; c < cnt; c++) {
            look(q, buf + HP + from + c, 0, p, X ? X + i * q->S : 0, X ? xlev[i] : 0); int b = 0; for (int v = 1; v < q->S; v++) if (p[v] > p[b]) b = v;
            out[i * cnt + c] = b; if (own) buf[HP + from + c] = b;
            if (acc) for (int64_t k = 0; k < q->K; k++) if (tix[k] >= 0) { acc[(c * q->K + k) * 2] += q->T[tix[k]]; acc[(c * q->K + k) * 2 + 1] += 1; }
        }
    }
}
