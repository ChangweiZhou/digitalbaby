/* An independent program of the tissue, written from THE_TISSUE_AS_RUN_20261119.md (reviewer's branch, 83dc294d) alone.
   Plain and slow on purpose: every cell, every connection, every step, in the order of the text's sections 1 and 2.
   The birth, the stream and the time constants are prepared by my_tissue.py and read from a text file.

   usage: my_tissue <birth.txt> <bytes.bin> <nbytes> <rec_lo> <rec_hi> <out_prefix> <trace_steps> <u1> [ftz]
     rec_lo..rec_hi : the bytes whose outputs at phi = 31 are recorded (float32, all cells) into <out_prefix>.rec
     trace_steps    : the outputs of every cell at every step k < trace_steps are written as float64 into <out_prefix>.trace
     u1             : 1 = step 8 uses x_j of this step for a cell; 0 = it uses z_j, the output of step k - 1 (the literal z)
     ftz            : 1 (default) = numbers below the smallest normal number are treated as zero (FTZ and DAZ)
   Writes <out_prefix>.conn (the final connections: receiver sender strength made lead lag) and <out_prefix>.stats. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <xmmintrin.h>

#define NC 2048
#define NL 256
#define STEPS_A_BYTE 32
#define RATE (1.0 / 16384.0)

typedef struct { int n, cap; int *s; double *w, *lead, *lag; long *made; } Conns;

static int is_fb[NC], nmem[NC], *mem[NC];
static double sig[NC + NL], tau[NC];
static double v[NC], th[NC], g[NC], F[NC + NL], M[NC + NL], P[NC], Q[NC], xo[NC], xn[NC], zl[NL];
static long last_make[NC];
static Conns C[NC];
static uint64_t rng;

static void add(Conns *c, int s, double w, long made) {
    if (c->n == c->cap) {
        c->cap = c->cap ? 2 * c->cap : 16;
        c->s = realloc(c->s, c->cap * sizeof(int)); c->w = realloc(c->w, c->cap * sizeof(double));
        c->lead = realloc(c->lead, c->cap * sizeof(double)); c->lag = realloc(c->lag, c->cap * sizeof(double));
        c->made = realloc(c->made, c->cap * sizeof(long));
    }
    c->s[c->n] = s; c->w[c->n] = w; c->made[c->n] = made; c->lead[c->n] = 0; c->lag[c->n] = 0; c->n++;
}
static void withdraw(Conns *c, int q) {          /* the last connection of the list takes the place of the withdrawn one */
    int l = c->n - 1;
    c->s[q] = c->s[l]; c->w[q] = c->w[l]; c->made[q] = c->made[l]; c->lead[q] = c->lead[l]; c->lag[q] = c->lag[l]; c->n--;
}
static uint64_t draw(void) {
    rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
    return rng * 2685821657736338717ULL;
}
static double clamp01(double a) { return a < 0 ? 0 : (a > 1 ? 1 : a); }

int main(int argc, char **argv) {
    if (argc < 9) { fprintf(stderr, "usage\n"); return 2; }
    long nbytes = atol(argv[3]), rec_lo = atol(argv[4]), rec_hi = atol(argv[5]), trace_steps = atol(argv[7]);
    int u1 = atoi(argv[8]), ftz = argc > 9 ? atoi(argv[9]) : 1;
    if (ftz) _mm_setcsr(_mm_getcsr() | 0x8040);   /* flush to zero and denormals are zero */
    FILE *fb = fopen(argv[1], "r"); long seed; int n_;
    if (fscanf(fb, "%ld", &seed) != 1) return 3;
    for (int i = 0; i < NC; i++) {
        int k, ns; if (fscanf(fb, "%d %d %lf %lf %d", &k, &is_fb[i], &sig[i], &tau[i], &ns) != 5 || k != i) return 4;
        for (int q = 0; q < ns; q++) { int s; if (fscanf(fb, "%d", &s) != 1) return 5; add(&C[i], s, 1.0 / 8.0, 0); }
        if (fscanf(fb, "%d", &n_) != 1) return 6;
        nmem[i] = n_; mem[i] = malloc((n_ + 1) * sizeof(int));
        for (int q = 0; q < n_; q++) if (fscanf(fb, "%d", &mem[i][q]) != 1) return 7;
    }
    fclose(fb);
    for (int j = 0; j < NL; j++) sig[NC + j] = 1.0;
    unsigned char *bytes = malloc(nbytes); FILE *fs = fopen(argv[2], "rb");
    if (fread(bytes, 1, nbytes, fs) != (size_t)nbytes) return 8;
    fclose(fs);
    rng = 88172645463325252ULL + (uint64_t)seed;
    for (int i = 0; i < NC; i++) { v[i] = 0; th[i] = 0; g[i] = 1; F[i] = 0; M[i] = 1.0 / 64; P[i] = 0; Q[i] = 0; last_make[i] = 0; xo[i] = 0; }
    for (int j = 0; j < NL; j++) { F[NC + j] = 0; M[NC + j] = 1.0 / 64; zl[j] = 0; }
    char path[4096];
    snprintf(path, sizeof path, "%s.rec", argv[6]); FILE *fr = fopen(path, "wb");
    FILE *ft = NULL; if (trace_steps > 0) { snprintf(path, sizeof path, "%s.trace", argv[6]); ft = fopen(path, "wb"); }
    long made_n = 0, withdrawn_n = 0;
    double dec = 1.0 - RATE, Fbefore[NC + NL];
    float recrow[NC];
    long K = nbytes * STEPS_A_BYTE;
    for (long k = 0; k < K; k++) {
        int phi = (int)(k % STEPS_A_BYTE);
        if (phi == 0) { int b = bytes[k / STEPS_A_BYTE]; for (int j = 0; j < NL; j++) zl[j] = (j == b) ? 1.0 : 0.0; }
        memcpy(Fbefore, F, sizeof F);                 /* "F_i and F_j the fast traces before this step" */
#define Z(s) ((s) < NC ? xo[(s)] : zl[(s) - NC])     /* z_j: a cell's output of step k - 1, a line's value of step k */
        for (int i = 0; i < NC; i++) {
            if (is_fb[i]) { xn[i] = 0; continue; }
            Conns *c = &C[i];
            double A = 0;                                                 /* 1. arrival */
            for (int q = 0; q < c->n; q++) A += sig[c->s[q]] * c->w[q] * Z(c->s[q]);
            A = g[i] * A;
            v[i] = v[i] + (A - v[i]) / tau[i];                            /* 2. potential */
            double x = clamp01(v[i] - th[i]);                             /* 3. output */
            xn[i] = x;
            if (x != 0 || Fbefore[i] != 0) {                              /* 4. order in time */
                for (int q = 0; q < c->n; q++) {
                    int s = c->s[q]; if (sig[s] < 0) continue;
                    c->lead[q] = dec * c->lead[q] + Fbefore[s] * x;
                    c->lag[q] = dec * c->lag[q] + Fbefore[i] * Z(s);
                }
            }
            double se = 0, si = 0;                                        /* 5. conservation */
            for (int q = 0; q < c->n; q++) { if (sig[c->s[q]] > 0) se += c->w[q]; else si += c->w[q]; }
            double fe = se > 0 ? 0.75 / se : 0, fi = si > 0 ? 0.25 / si : 0;
            for (int q = 0; q < c->n; q++) {
                double f = sig[c->s[q]] > 0 ? fe : fi;
                if (f == 0) continue;
                double w = c->w[q] * f; c->w[q] = w > 1 ? 1 : w;
            }
            double t = th[i] + (x - 1.0 / 8) / 128;                       /* 6. threshold */
            th[i] = t > 0 ? t : 0;
            double m = 0;                                                 /* 7. gain */
            for (int q = 0; q < c->n; q++) { int s = c->s[q]; if (sig[s] > 0 && Z(s) > m) m = Z(s); }
            double qd = dec * Q[i]; Q[i] = m > qd ? m : qd;
            double pd = dec * P[i]; P[i] = x > pd ? x : pd;
            double gg = g[i] * (1 + (1.0 / 4096) * (Q[i] - P[i]) / (Q[i] + P[i] + 1e-30));
            g[i] = gg < 1.0 / 64 ? 1.0 / 64 : (gg > 8 ? 8 : gg);
        }
        for (int j = 0; j < NC; j++) {                                    /* 8. traces */
            double zz = u1 ? xn[j] : xo[j];
            F[j] = F[j] + (zz - F[j]) / 4; M[j] = M[j] + (zz - M[j]) * RATE;
        }
        for (int j = 0; j < NL; j++) { double zz = zl[j]; F[NC + j] = F[NC + j] + (zz - F[NC + j]) / 4; M[NC + j] = M[NC + j] + (zz - M[NC + j]) * RATE; }
        for (int i = 0; i < NC; i++) {                                    /* 9. withdrawal and making */
            if (is_fb[i]) continue;
            Conns *c = &C[i];
            for (int q = c->n - 1; q >= 0; q--) {
                int s = c->s[q]; if (s >= NC) continue;
                long age = k - c->made[q];
                if (age < 128) continue;
                if (c->w[q] < 1.0 / 128) { withdraw(c, q); withdrawn_n++; continue; }
                if (sig[s] > 0 && age >= 4096 && c->lag[q] > 0 && c->lag[q] > c->lead[q]) { withdraw(c, q); withdrawn_n++; }
            }
            if (th[i] == 0 && M[i] < 1.0 / 4096 && k - last_make[i] >= 1024) {
                int cand[NC + NL], n = 0;
                for (int q = 0; q < nmem[i]; q++) {
                    int s = mem[i][q], here = 0;
                    for (int r = 0; r < c->n; r++) if (c->s[r] == s) { here = 1; break; }
                    if (!here) cand[n++] = s;
                }
                if (n > 0) {
                    uint64_t val = draw();
                    add(c, cand[val % (uint64_t)n], 1.0 / 8, k); last_make[i] = k; made_n++;
                }
            }
        }
        memcpy(xo, xn, sizeof xo);                                         /* 10. x_i becomes the output of step k */
        if (ft && k < trace_steps) fwrite(xn, sizeof(double), NC, ft);
        long byte = k / STEPS_A_BYTE;
        if (phi == STEPS_A_BYTE - 1 && byte >= rec_lo && byte < rec_hi) { for (int i = 0; i < NC; i++) recrow[i] = (float)xn[i]; fwrite(recrow, sizeof(float), NC, fr); }
    }
    fclose(fr); if (ft) fclose(ft);
    snprintf(path, sizeof path, "%s.conn", argv[6]); FILE *fc = fopen(path, "w");
    long total = 0, nonfb = 0;
    for (int i = 0; i < NC; i++) {
        if (!is_fb[i]) { total += C[i].n; nonfb++; }
        for (int q = 0; q < C[i].n; q++) fprintf(fc, "%d %d %.17g %ld %.17g %.17g\n", i, C[i].s[q], C[i].w[q], C[i].made[q], C[i].lead[q], C[i].lag[q]);
    }
    fclose(fc);
    snprintf(path, sizeof path, "%s.stats", argv[6]); FILE *fo = fopen(path, "w");
    fprintf(fo, "{\"nbytes\": %ld, \"made\": %ld, \"withdrawn\": %ld, \"connections\": %ld, \"cells_not_feedback\": %ld, \"connections_a_cell\": %.6f, \"u1\": %d, \"ftz\": %d}\n",
            nbytes, made_n, withdrawn_n, total, nonfb, (double)total / nonfb, u1, ftz);
    fclose(fo);
    return 0;
}
