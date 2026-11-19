#include <stdint.h>
/* one step of the tissue's per-connection work, connections sorted by receiver */
void step(int64_t n, int64_t N, const int32_t *snd, const int32_t *rcv, double *w, double *e,
          const double *sig, const double *z, const double *m, const double *xp, double *A,
          const int64_t *k0, int64_t k, uint8_t *gone) {
    for (int64_t i = 0; i < N; i++) A[i] = 0.0;
    for (int64_t c = 0; c < n; c++) { int32_t s = snd[c]; A[rcv[c]] += sig[s] * w[c] * z[s]; }
    for (int64_t c = 0; c < n; c++) {
        int32_t s = snd[c], r = rcv[c];
        double nw = w[c] - 0.125 * sig[s] * m[r] * xp[r] * e[c];
        w[c] = nw < 0.0 ? 0.0 : (nw > 1.0 ? 1.0 : nw);
        e[c] += (z[s] - e[c]) / 32.0;
        gone[c] = (k - k0[c] >= 128) && (w[c] < 1.0 / 128.0);
    }
}
