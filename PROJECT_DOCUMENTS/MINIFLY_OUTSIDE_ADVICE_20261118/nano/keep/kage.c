/* PLAN_2.md and later (nano/keep): the inner loop of kage.py (AgeSys.learn), compiled. The numpy program is the reference (check_kage.py).
   Cells in between join at their own byte count (born[j], ascending in j); a cell's steps, the answering cell's steps on the strength from it, the
   return strength's steps and the fading of both are multiplied by the cell's rate, a function of its age (law 0: 1; 1: exp(-age/tau); 2: 1/(1+age/tau)).
   young > 0: a strength from a line is open once that line was active while the cell was younger than `young` bytes; a cell with a closed
   strength on an active line is silent at that byte and unchanged by it (rate nothing: no fading either).
   blaw (PLAN_4.md): the answering cell's steps on its constants shrink with its age n. sigjoin (PLAN_5.md): cells join on the common signal.
   sigjoin 2 (PLAN_6.md): the signal is the shortfall of the number of cells the answering cell hears. meet (PLAN_7.md): a strength is open after youth
   if its line was active on at least `meet` young bytes.
   usemode > 0: the rate is 1/(1+U[j]/tau) with U[j] the cell's own accumulated use.
   Built by fastage.py:  gcc -O3 -march=native -ffast-math -shared -fPIC -o kagek.so kage.c -lm                                               */
#include <math.h>
#include <stdint.h>

typedef struct {
    int64_t H, NL, S, D, n, law, young, usemode, sigjoin, nl, blaw, meet;
    double eta1, eta2, lam, tau, E, rhoE, C, G, rhoS, delta, btau, A, rhoA, K;
    double *W1, *b1, *W2, *b2, *R, *U, *thr;
    int64_t *born, *hist;
    uint8_t *seen;
    double *h, *rate, *m;                               /* work: H, H, S */
} ANet;

void age_steps(ANet *q, int64_t nsteps, const int32_t *xs, const int32_t *ys) {
    const int64_t H = q->H, NL = q->NL, S = q->S, D = q->D;
    const double eta1 = q->eta1, lam = q->lam, tau = q->tau;
    double *restrict h = q->h, *restrict rate = q->rate, *restrict m = q->m;
    int64_t act[64];
    for (int64_t i = 0; i < nsteps; i++) {
        for (int64_t d = D - 1; d > 0; d--) q->hist[d] = q->hist[d - 1];
        q->hist[0] = xs[i];
        const int64_t y = ys[i];
        if (y < 0) continue;
        const int64_t n = q->n;
        int64_t nl = q->nl;
        if (q->sigjoin) { while (nl < H && q->thr[nl] <= q->C) { q->born[nl] = n; nl++; } }        /* joined by the common signal: born is set at joining */
        else while (nl < H && q->born[nl] <= n) nl++;
        q->nl = nl;
        for (int64_t d = 0; d < D; d++) act[d] = d * S + q->hist[d];
        for (int64_t j = 0; j < nl; j++) { const double *w = q->W1 + j * NL; double u = 0; for (int64_t d = 0; d < D; d++) u += w[act[d]]; h[j] = -(u + q->b1[j]); }
        for (int64_t j = 0; j < nl; j++) h[j] = 1.0 / (1.0 + exp(h[j]));
        if (q->usemode) for (int64_t j = 0; j < nl; j++) rate[j] = 1.0 / (1.0 + q->U[j] / tau);
        else if (q->law == 1) for (int64_t j = 0; j < nl; j++) rate[j] = exp(-(double)(n - q->born[j]) / tau);
        else if (q->law == 2) for (int64_t j = 0; j < nl; j++) rate[j] = 1.0 / (1.0 + (double)(n - q->born[j]) / tau);
        else for (int64_t j = 0; j < nl; j++) rate[j] = 1.0;
        int64_t heard = nl;
        if (q->young) for (int64_t j = 0; j < nl; j++) {
            uint8_t *sn = q->seen + j * NL;
            if (n - q->born[j] < q->young) { for (int64_t d = 0; d < D; d++) if (sn[act[d]] < 255) sn[act[d]]++; }      /* in its youth a cell counts the bytes at which each line is active */
            else { int ok = 1; for (int64_t d = 0; d < D; d++) ok &= (sn[act[d]] >= q->meet); if (!ok) { h[j] = 0.0; rate[j] = 0.0; heard--; } }
        }
        double hh = 0; for (int64_t j = 0; j < nl; j++) hh += h[j] * h[j];
        const double g = q->eta2 / (hh + 1.0);
        for (int64_t v = 0; v < S; v++) { const double *w = q->W2 + v * H; double a = 0; for (int64_t j = 0; j < nl; j++) a += w[j] * h[j]; m[v] = -(a + q->b2[v]); }
        m[y] += 1.0;
        double mm = 0; for (int64_t v = 0; v < S; v++) mm += m[v] * m[v];
        for (int64_t j = 0; j < nl; j++) {
            double *r = q->R + j * S; double s = 0; for (int64_t v = 0; v < S; v++) s += r[v] * m[v];
            const double rj = rate[j], sig = s * h[j] * (1.0 - h[j]) * rj, dj = eta1 * sig, f = 1.0 - lam * rj, gh = g * h[j] * rj;
            double *w = q->W1 + j * NL;
            for (int64_t d = 0; d < D; d++) w[act[d]] += dj;
            q->b1[j] += dj;
            for (int64_t v = 0; v < S; v++) { const double p = gh * m[v]; q->W2[v * H + j] = (q->W2[v * H + j] + p) * f; r[v] = (r[v] + p) * f; }
            if (q->usemode == 1) q->U[j] += fabs(s * h[j] * (1.0 - h[j]));
            else if (q->usemode == 2) q->U[j] += h[j];
        }
        const double gb = g * (q->blaw == 1 ? exp(-(double)n / q->btau) : q->blaw == 2 ? 1.0 / (1.0 + (double)n / q->btau) : 1.0);
        for (int64_t v = 0; v < S; v++) q->b2[v] += gb * m[v];
        q->E += q->rhoE * (mm - q->E); q->G += q->rhoS * (mm - q->G);
        q->A += q->rhoA * ((double)heard - q->A);
        { const double x = q->sigjoin == 2 ? 1.0 - q->A / q->K : q->E - (1.0 + q->delta) * q->G; if (x > 0.0) q->C += x; }      /* the common signal: the mismatch above its usual size */
        q->n = n + 1;
    }
}
