
#include <math.h>
#include <stdint.h>
#define NL 40
#define MAXS 16
#define MAXK 300
#define MAXA 140
#define MAXN (MAXK + 2 * MAXA)
#define HP 16
typedef struct { int64_t S, K, KA, ZTOT, PTOT, END, ncell, cap, step, shrink; const int32_t *nd, *del; const int64_t *off; int32_t *map; float *C, *N, *U; double *T, *Tu; double eta, e, ncap; } sys_t;
typedef struct { double p[3][MAXS]; int n[3]; int tix[3][MAXN]; double s[3][MAXN][MAXS]; int64_t idP[MAXK], idA[MAXA], idB[MAXA]; int g3, g2; } rec_t;
static rec_t ring[3], tmp; static int64_t keyv[MAXK];
static inline int level(float n, float u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* what the cell id says from its counts in `slot` (0: of the next byte, 1: of the one after, 2: of the one after that); returns the trust index or -1 */
static inline int say(const sys_t *q, int64_t id, int slot, int fam, int64_t k, double *s) {
    if (id < 0) return -1; const float n = q->N[id * 3 + slot]; if (n < 1) return -1;
    const int64_t S = q->S; const float *c = q->C + (id * 3 + slot) * S;
    if (q->ncap > 0 && n > q->ncap) { const double sc = q->ncap / n, den = log(q->ncap + S * q->e); for (int v = 0; v < S; v++) s[v] = log(c[v] * sc + q->e) - den; }
    else { const double den = log(n + S * q->e); for (int v = 0; v < S; v++) s[v] = log(c[v] + q->e) - den; }
    return (int)(((fam * (q->K + 1) + k) * NL) + level(n, q->U[id * 3 + slot]));
}
static inline void add(const sys_t *q, rec_t *r, int d, int64_t id, int slot, int fam, int64_t k, double *a) {
    const int i = r->n[d], t = say(q, id, slot, fam, k, r->s[d][i]); if (t < 0) return;
    const double tk = q->T[t]; for (int v = 0; v < q->S; v++) a[v] += tk * r->s[d][i][v]; r->tix[d][i] = t; r->n[d] = i + 1;
}
static inline int soft(const sys_t *q, const double *a, double *p) {
    double mx = -1e300, z = 0; int b = 0;
    for (int v = 0; v < q->S; v++) if (a[v] > mx) { mx = a[v]; b = v; }
    for (int v = 0; v < q->S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
    for (int v = 0; v < q->S; v++) p[v] /= z;
    return b;
}
static inline int64_t cell(sys_t *q, int64_t slot, int make) { int64_t id = q->map[slot]; if (id < 0 && make) { id = q->ncell++; q->map[slot] = (int32_t)id; } return id; }
/* h[-d] is the byte d places back */
static void look(sys_t *q, const int32_t *h, int make, rec_t *r) {
    const int64_t S = q->S, K = q->K, KA = q->KA; double a[MAXS];
    for (int64_t k = 0; k < K; k++) { int64_t key = 0; for (int j = 0; j < q->nd[k]; j++) key = key * S + h[-q->del[k * 4 + j]]; keyv[k] = key; r->idP[k] = cell(q, q->off[k] + key, make); }
    r->n[0] = r->n[1] = r->n[2] = 0;
    for (int v = 0; v < S; v++) a[v] = 0;
    for (int64_t k = 0; k < K; k++) add(q, r, 2, r->idP[k], 2, 2, k, a);
    r->g3 = soft(q, a, r->p[2]);
    for (int64_t k = 0; k <= KA; k++) r->idA[k] = cell(q, q->PTOT + (k < KA ? q->off[k] + keyv[k] : q->ZTOT) * S + r->g3, make);
    for (int v = 0; v < S; v++) a[v] = 0;
    for (int64_t k = 0; k < K; k++) add(q, r, 1, r->idP[k], 1, 1, k, a);
    for (int64_t k = 0; k <= KA; k++) add(q, r, 1, r->idA[k], 1, 4, k, a);
    r->g2 = soft(q, a, r->p[1]);
    for (int64_t k = 0; k <= KA; k++) r->idB[k] = cell(q, q->PTOT + (q->ZTOT + 1) * S + (k < KA ? q->off[k] + keyv[k] : q->ZTOT) * S + r->g2, make);
    for (int v = 0; v < S; v++) a[v] = 0;
    for (int64_t k = 0; k < K; k++) add(q, r, 0, r->idP[k], 0, 0, k, a);
    for (int64_t k = 0; k <= KA; k++) add(q, r, 0, r->idA[k], 0, 3, k, a);
    for (int64_t k = 0; k <= KA; k++) add(q, r, 0, r->idB[k], 0, 5, k, a);
    soft(q, a, r->p[0]);
}
static inline void trust(sys_t *q, const rec_t *r, int d, int y) {
    for (int i = 0; i < r->n[d]; i++) {
        const double *s = r->s[d][i]; double g = s[y]; for (int v = 0; v < q->S; v++) g -= r->p[d][v] * s[v];
        const int t = r->tix[d][i]; q->T[t] += (q->shrink ? q->eta / sqrt(1.0 + q->Tu[t]) : q->eta) * g; q->Tu[t] += 1.0;
    }
}
static inline void count(sys_t *q, int64_t id, int slot, int y) { float *c = q->C + (id * 3 + slot) * q->S; if (c[y] == 0) q->U[id * 3 + slot] += 1; c[y] += 1; q->N[id * 3 + slot] += 1; }
int64_t steps(sys_t *q, int64_t n, const int32_t *x, double *P) {
    const int64_t S = q->S, K = q->K, KA = q->KA;
    for (int64_t t = 0; t < n; t++) {
        if (q->ncell + K + 2 * (KA + 1) > q->cap) return t;
        rec_t *cur = &ring[q->step % 3], *prev = q->step >= 1 ? &ring[(q->step + 2) % 3] : 0, *pp = q->step >= 2 ? &ring[(q->step + 1) % 3] : 0;
        look(q, x + t, 1, cur); const int y = x[t];
        if (P) for (int d = 0; d < 3; d++) for (int v = 0; v < S; v++) P[(t * 3 + d) * S + v] = cur->p[d][v];
        trust(q, cur, 0, y); if (prev) trust(q, prev, 1, y); if (pp) trust(q, pp, 2, y);
        for (int64_t k = 0; k < K; k++) count(q, cur->idP[k], 0, y);
        for (int64_t k = 0; k <= KA; k++) { count(q, cur->idA[k], 0, y); count(q, cur->idB[k], 0, y); }
        if (prev) { for (int64_t k = 0; k < K; k++) count(q, prev->idP[k], 1, y); for (int64_t k = 0; k <= KA; k++) count(q, prev->idA[k], 1, y); }
        if (pp) for (int64_t k = 0; k < K; k++) count(q, pp->idP[k], 2, y);
        q->step++;
    }
    return n;
}
/* out: the byte named at each place; gs: at each place the guesses (three ahead, two ahead) made there */
void answers(sys_t *q, int64_t n, const int32_t *lines, int64_t L, int64_t from, int64_t cnt, int64_t own, int32_t *out, int32_t *gs) {
    int32_t buf[HP + 64];
    for (int64_t i = 0; i < n; i++) {
        for (int j = 0; j < HP; j++) buf[j] = (int32_t)q->END;
        for (int j = 0; j < L; j++) buf[HP + j] = lines[i * L + j];
        for (int64_t c = 0; c < cnt; c++) {
            look(q, buf + HP + from + c, 0, &tmp); int b = 0; for (int v = 1; v < q->S; v++) if (tmp.p[0][v] > tmp.p[0][b]) b = v;
            out[i * cnt + c] = b; gs[(i * cnt + c) * 2] = tmp.g3; gs[(i * cnt + c) * 2 + 1] = tmp.g2; if (own) buf[HP + from + c] = b;
        }
    }
}
