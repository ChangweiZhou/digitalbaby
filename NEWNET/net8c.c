/* net8's step in C for the configuration net9 uses (fb=none, an expansion layer, k=0, rec=0, send_y=0, norm_rec=0); see net8.py
   for the definition and net8c.py for the wrapper. Sums over a cell's inputs follow numpy's pairwise order (so that they are
   bit-equal); exp is the C library's (numpy's own exp may differ in the last bit). Build: gcc -O3 -march=native -shared -fPIC. */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* numpy's pairwise summation (DOUBLE_pairwise_sum, block size 128, unrolled by 8) of x[0..n) */
static double pw_sum(const double *x, int n)
{
    if (n < 8) { double r = 0.; for (int i = 0; i < n; i++) r += x[i]; return r; }
    if (n <= 128) {
        double r[8]; int i;
        for (int j = 0; j < 8; j++) r[j] = x[j];
        for (i = 8; i < n - (n % 8); i += 8) for (int j = 0; j < 8; j++) r[j] += x[i + j];
        double s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) s += x[i];
        return s;
    }
    int n2 = n / 2; n2 -= n2 % 8;
    return pw_sum(x, n2) + pw_sum(x + n2, n - n2);
}

static inline double sig(double x) { return 1.0 / (1.0 + exp(-x)); }

/* k-th largest (1 <= k <= n) of v[0..n): a min-heap of the k largest so far (w: k scratch places) */
static double kth_largest(const double *v, double *w, int n, int k)
{
    for (int i = 0; i < k; i++) {                       /* heap of the first k */
        int x = i; double val = v[i];
        while (x > 0 && w[(x - 1) / 2] > val) { w[x] = w[(x - 1) / 2]; x = (x - 1) / 2; }
        w[x] = val;
    }
    for (int i = k; i < n; i++) {
        double val = v[i]; if (val <= w[0]) continue;
        int x = 0;                                      /* replace the smallest, sift down */
        for (;;) {
            int l = 2 * x + 1, r = l + 1, s = x; double sv = val;
            if (l < k && w[l] < sv) { s = l; sv = w[l]; }
            if (r < k && w[r] < sv) { s = r; sv = w[r]; }
            if (s == x) break;
            w[x] = w[s]; x = s;
        }
        w[x] = val;
    }
    return w[0];
}

/* learning from the last answer: the output end only (fb=none). e from O (the last answer's outputs), z and y (the last states). */
void learn(int V, int G, int N, int nxt, int gave, int yesno, int gated, double eta_o, double out_var,
           const double *O, const double *z, const double *y, double *vy, double *AT, int *act, double *e)
{
    for (int v = 0; v < V; v++) {
        double tg = v == nxt ? 1.0 : 0.0; e[v] = tg - O[v];
        if (gated && !(tg > 0 || O[v] > 0)) e[v] = 0.0;
    }
    if (yesno && gave >= 0) { for (int v = 0; v < V; v++) e[v] = 0.0; e[gave] = (gave == nxt ? 1.0 : 0.0) - O[gave]; }
    if (out_var > 0) for (int i = 0; i < N; i++) vy[i] += (y[i] * y[i] - vy[i]) / out_var;
    int na = 0; for (int j = 0; j < G; j++) if (z[j] != 0) act[na++] = j;
    if (!na) return;
    double c = eta_o / (double)na;
    for (int t = 0; t < na; t++) {                       /* AT: the output strengths stored by sender, [G][V] (self.A is its transpose) */
        double *a = AT + (size_t)act[t] * V;
        for (int v = 0; v < V; v++) if (e[v] != 0) a[v] += c * e[v];
    }
}

/* reading the byte (the byte lines stored by byte: WcT, WgT, WdT [V][N], glT [V][G]): tissue state h (new), running mean m (in place), adapted output y, expansion z, outputs O */
void forward(int N, int F, int V, int G, int gfan, int gk, int byte, int pb, double center, int use_var,
             const int64_t *src, const double *WcT, const double *WgT, const double *Uc, const double *Ug, const double *bc,
             const double *bg, const double *WdT, const double *hp, double *h, double *m, double *y, const double *vy,
             const int64_t *gsrc, const double *gw, const double *glT, const double *AT, double *z, double *O,
             double *yg, double *zg, double *scratch, int *act)
{
    double pc[256], pg[256];
    for (int i = 0; i < N; i++) {
        const int64_t *s = src + (size_t)i * F; const double *uc = Uc + (size_t)i * F, *ug = Ug + (size_t)i * F;
        for (int f = 0; f < F; f++) { double p = hp[s[f]]; pc[f] = uc[f] * p; pg[f] = ug[f] * p; }
        double zc = WcT[(size_t)byte * N + i] + pw_sum(pc, F) + bc[i];
        if (pb >= 0) zc += WdT[(size_t)pb * N + i];
        double c = sig(zc), g = sig(WgT[(size_t)byte * N + i] + pw_sum(pg, F) + bg[i]);
        double hn = g * hp[i] + (1 - g) * c; h[i] = hn;
        if (center > 0) { m[i] += (hn - m[i]) / center; y[i] = hn - m[i]; } else y[i] = hn;
        yg[i] = use_var ? y[i] / sqrt(vy[i]) : y[i];
    }
    for (int j = 0; j < G; j++) {
        const int64_t *s = gsrc + (size_t)j * gfan; const double *w = gw + (size_t)j * gfan;
        double sj;
        if (gfan == 4) sj = (((0.0 + w[0] * yg[s[0]]) + w[1] * yg[s[1]]) + w[2] * yg[s[2]]) + w[3] * yg[s[3]];   /* = pw_sum for n = 4 */
        else { for (int k = 0; k < gfan; k++) pc[k] = w[k] * yg[s[k]]; sj = pw_sum(pc, gfan); }
        zg[j] = sj + glT[(size_t)byte * G + j];
    }
    double th = kth_largest(zg, scratch, G, gk);
    int na = 0;
    for (int j = 0; j < G; j++) { z[j] = 0.0; if (zg[j] > th) { z[j] = 1.0; na++; } }
    for (int j = 0; j < G && na < gk; j++) if (zg[j] == th) { z[j] = 1.0; na++; }   /* ties at the threshold: lowest index */
    int n = 0; for (int j = 0; j < G; j++) if (z[j] != 0) act[n++] = j;
    for (int v = 0; v < V; v++) O[v] = 0.0;
    for (int t = 0; t < n; t++) { const double *a = AT + (size_t)act[t] * V; for (int v = 0; v < V; v++) O[v] += a[v]; }
}
