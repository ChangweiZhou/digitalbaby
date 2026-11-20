/* nano/keep2: my own kernel for the plain system of cells2.py with three rules (PLAN_1.md, section 2). Checked against rep.py's numpy (check.py). */
#include <math.h>
#include <string.h>
#define S 14
#define D 10
#define NL 140
typedef struct { double eta1, eta2, lam, K, theta, btau, rate, A, sumX; long nans; int P, H0, young, meet, njoined, closed, fade_silent, allbytes, join; } Par;

static void youth(Par *p, const int *act, unsigned char *cnt, unsigned char *open, int *age) {
    for (int j = 0; j < p->njoined; j++) if (age[j] < p->young) {
        unsigned char *c = cnt + (long)j * NL;
        for (int d = 0; d < D; d++) if (c[act[d]] < 255) c[act[d]]++;
        if (++age[j] == p->young) { unsigned char *o = open + (long)j * NL; for (int l = 0; l < NL; l++) o[l] = c[l] >= p->meet; }
    }
}

void run(Par *p, long n, const int *x, const int *y, double *W1, double *b1, double *W2, double *b2, double *R, unsigned char *cnt, unsigned char *open, int *age, int *hist, double *h, unsigned char *sp) {
    int act[D]; double a[S], m[S]; const double keep = 1 - p->lam;
    for (long i = 0; i < n; i++) {
        memmove(hist + 1, hist, (D - 1) * sizeof(int)); hist[0] = x[i];
        for (int d = 0; d < D; d++) act[d] = d * S + hist[d];
        if (y[i] < 0) { if (p->closed && p->allbytes) youth(p, act, cnt, open, age); continue; }
        int nj = p->njoined, heard = 0; double hh = 0;
        for (int k = 0; k < S; k++) a[k] = b2[k];
        for (int j = 0; j < nj; j++) {
            int s = 1;
            if (p->closed && age[j] >= p->young) { const unsigned char *o = open + (long)j * NL; for (int d = 0; d < D; d++) if (!o[act[d]]) { s = 0; break; } }
            sp[j] = s; if (!s) continue;
            const double *w = W1 + (long)j * NL; double u = b1[j];
            for (int d = 0; d < D; d++) u += w[act[d]];
            double v = 1 / (1 + exp(-u)); h[j] = v; hh += v * v; heard++;
            const double *w2 = W2 + (long)j * S; for (int k = 0; k < S; k++) a[k] += w2[k] * v;
        }
        for (int k = 0; k < S; k++) m[k] = -a[k];
        m[y[i]] += 1; double g = p->eta2 / (hh + 1);
        for (int j = 0; j < nj; j++) {
            double *w2 = W2 + (long)j * S, *r = R + (long)j * S;
            if (!sp[j]) { if (p->fade_silent) for (int k = 0; k < S; k++) { w2[k] *= keep; r[k] *= keep; } continue; }
            double v = h[j], sg = 0;
            for (int k = 0; k < S; k++) sg += r[k] * m[k];
            double d1 = p->eta1 * sg * v * (1 - v), gv = g * v;
            for (int k = 0; k < S; k++) { double d2 = gv * m[k]; r[k] = (r[k] + d2) * keep; w2[k] = (w2[k] + d2) * keep; }
            double *w = W1 + (long)j * NL; for (int d = 0; d < D; d++) w[act[d]] += d1;
            b1[j] += d1;
        }
        double sc = p->btau > 0 ? exp(-(double)p->nans / p->btau) : 1.0; p->nans++;
        for (int k = 0; k < S; k++) b2[k] += g * m[k] * sc;
        if (p->closed) youth(p, act, cnt, open, age);
        if (p->join) {
            p->A += p->rate * (heard - p->A); double X = 1 - p->A / p->K; if (X > 0) p->sumX += X;
            while (p->njoined < p->P && p->sumX >= (p->njoined - p->H0 + 1) * p->theta) { age[p->njoined] = 0; p->njoined++; }
        }
    }
}
