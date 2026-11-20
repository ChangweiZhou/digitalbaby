/* nano/replay, PLAN_6.md: the judging of candidate lines in replay6.py, compiled. For each line of ten bytes read alone after a newline: the sum,
   over its nine learned bytes and the S branches of the answering cell, of the squared mismatch, with the strengths as given; no strength is
   changed. The forward arithmetic is that of loop_small_case/kern3.c (copied in kernr.c). The numpy function replay6.mismatch is the reference
   (check_replay6.py).  Built by fastm.py.                                                                                                    */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

void judge_lines(int64_t H, int64_t NL, int64_t S, int64_t D, int64_t END, const double *W1, const double *b1, const double *W2, const double *b2,
                 int64_t n, const int32_t *lines, double *out) {
    double *restrict W1t = malloc(NL * H * sizeof(double)), *restrict u = malloc(H * sizeof(double)), *restrict h = malloc(H * sizeof(double));
    for (int64_t j = 0; j < H; j++) for (int64_t l = 0; l < NL; l++) W1t[l * H + j] = W1[j * NL + l];
    for (int64_t i = 0; i < n; i++) {
        const int32_t *L = lines + i * 10; int64_t hist[64]; double tot = 0;
        for (int64_t d = 0; d < D; d++) hist[d] = END;
        for (int64_t t = 0; t < 9; t++) {
            for (int64_t d = D - 1; d > 0; d--) hist[d] = hist[d - 1];
            hist[0] = L[t];
            for (int64_t j = 0; j < H; j++) u[j] = b1[j];
            for (int64_t d = 0; d < D; d++) { const double *restrict w = W1t + (d * S + hist[d]) * H; for (int64_t j = 0; j < H; j++) u[j] += w[j]; }
            for (int64_t j = 0; j < H; j++) h[j] = 1.0 / (1.0 + exp(-u[j]));
            for (int64_t v = 0; v < S; v++) { const double *restrict w = W2 + v * H; double a = 0; for (int64_t j = 0; j < H; j++) a += w[j] * h[j]; const double m = (v == L[t + 1] ? 1.0 : 0.0) - (a + b2[v]); tot += m * m; }
        }
        out[i] = tot;
    }
    free(W1t); free(u); free(h);
}
