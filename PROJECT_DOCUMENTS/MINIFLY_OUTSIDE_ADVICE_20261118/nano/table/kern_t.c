
#include <math.h>
#include <stdint.h>
/* at each byte: the active cells' strengths are summed, the branches compete, and every active cell changes by its step times the mismatch */
void table_steps(int64_t n, int64_t K1, const int64_t *ids, const int32_t *y, double *W, double *uses, double eta, int learn, double *P) {
    double a[65], p[65];
    for (int64_t t = 0; t < n; t++) {
        const int64_t *id = ids + t * K1; double mx = -1e300, z = 0;
        for (int v = 0; v < 65; v++) a[v] = 0;
        for (int64_t k = 0; k < K1; k++) if (id[k] >= 0) { const double *w = W + id[k] * 65; for (int v = 0; v < 65; v++) a[v] += w[v]; }
        for (int v = 0; v < 65; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < 65; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < 65; v++) p[v] /= z;
        if (P) for (int v = 0; v < 65; v++) P[t * 65 + v] = p[v];
        if (learn) for (int64_t k = 0; k < K1; k++) if (id[k] >= 0) {
            double *w = W + id[k] * 65; const double st = eta / sqrt(1.0 + uses[id[k]]);
            for (int v = 0; v < 65; v++) w[v] -= st * p[v];
            w[y[t]] += st; uses[id[k]] += 1.0;
        }
    }
}
