
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define NMAX 64
typedef struct {
    int N; int64_t W; double s;
    double h[NMAX][256], n[NMAX], b[NMAX], cost[NMAX];
    int64_t inblock; int32_t *wlog, *xlog;      /* the block so far: the cell that counted each byte (-1: none), and the byte */
} Det;
Det *det_new(int64_t W) { Det *d = calloc(1, sizeof(Det)); d->W = W; d->s = 1.0 / (double)W; d->N = 1; d->b[0] = 1.0; d->wlog = malloc(4 * W); d->xlog = malloc(4 * W); return d; }
void det_free(Det *d) { free(d->wlog); free(d->xlog); free(d); }
int det_n(Det *d) { return d->N; }
double *det_h(Det *d) { return &d->h[0][0]; }
double *det_b(Det *d) { return d->b; }
/* reads x[lo..hi). awake[t - lo]: the cell with the largest share before byte t is read. born_end[c]: the place just after the block that gave situation c its birth.
   learn = 0: no cell counts and none is born; the belief still moves. */
int det_run(Det *d, const int32_t *x, int64_t lo, int64_t hi, int learn, int32_t *awake, int64_t *born_end) {
    for (int64_t t = lo; t < hi; t++) {
        const int y = x[t]; int w = 0; double z = 0;
        for (int c = 1; c < d->N; c++) if (d->b[c] > d->b[w]) w = c;
        awake[t - lo] = w;
        for (int c = 0; c < d->N; c++) { const double p = (d->h[c][y] + 0.5) / (d->n[c] + 128.0); d->cost[c] -= log(p); d->b[c] = ((1.0 - d->s) * d->b[c] + d->s / d->N) * p; z += d->b[c]; }
        for (int c = 0; c < d->N; c++) d->b[c] /= z;
        if (learn) { d->h[w][y] += 1; d->n[w] += 1; }
        d->wlog[d->inblock] = learn ? w : -1; d->xlog[d->inblock] = y; d->inblock++;
        if (d->inblock == d->W) {
            double fresh = 0, best = 1e300, cnt[256]; memset(cnt, 0, sizeof(cnt));
            for (int64_t i = 0; i < d->W; i++) { const int v = d->xlog[i]; fresh -= log((cnt[v] + 0.5) / ((double)i + 128.0)); cnt[v] += 1; }
            for (int c = 0; c < d->N; c++) if (d->cost[c] < best) best = d->cost[c];
            if (learn && fresh < best && d->N < NMAX) {
                for (int64_t i = 0; i < d->W; i++) if (d->wlog[i] >= 0) { d->h[d->wlog[i]][d->xlog[i]] -= 1; d->n[d->wlog[i]] -= 1; }
                const int c = d->N++; for (int v = 0; v < 256; v++) d->h[c][v] = cnt[v]; d->n[c] = (double)d->W;
                for (int k = 0; k < d->N; k++) d->b[k] = 0; d->b[c] = 1.0; born_end[c] = t + 1;
            }
            memset(d->cost, 0, sizeof(d->cost)); d->inblock = 0;
        }
    }
    return d->N;
}
