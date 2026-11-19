/* The tissue of step 1, written by the reviewer for quick runs (19 November 2026).
   The formulas and their order are those of E63's tissue_step1.py (method Tissue.step), with these differences, none of
   which changes the mathematics:
     - a trace is held once for a sending cell; a connection's trace is that trace less what it was when the connection
       was made, decayed (the same number up to rounding);
     - strengths are changed only at receiving cells whose activity at the previous step was not zero, where the rule
       contains that factor (exactly zero otherwise);
     - a cell whose neighbourhood is full is not searched for a new source;
     - the draws for new sources come from a generator of this file, not from Python's, so a life is not bit for bit
       E63's life.
   Options: an output floor; the target of activity a*; withdrawal of connections by use (zero for a silent cell). */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

#define NCELL 2048
#define NIN 256
#define LINE0 NCELL
#define NZ (NCELL + NIN + 2)
#define S 32
#define PS 128
#define TAU 32.0
#define TAU_VT 4096.0
#define ETA 0.125
#define ETA_T (1.0 / 128.0)
#define GIVE_UP (1.0 / 128.0)
#define START_W 0.125
enum { ENC = 0, INNER = 1, OUTE = 2, FBK = 3 };
enum { FIRST = 0, COINC = 1, FIXED = 2, TIMING = 3 };

typedef struct {
    int maxd;                       /* capacity of a cell's list of connections */
    int rule; double floor_x, astar; int by_use;
    int two_sided; double eta_g; double *gain;      /* 1: a gain on the cell's arrival that follows its activity toward a*; 2: toward rho x the mean activity of its sources */
    double rho; double *ibar; double gmax;
    double *F; double tau_f, eta_s; double norm, we, wi;        /* norm 1: after its strengths change, a cell rescales them so that those from excitatory senders sum to we and those from inhibitory senders to wi */
    double *tauv;                   /* time constant of the potential of each cell, in steps (2 in the definition); set by the driver (option tau=..) */
    double mk_long;                 /* mk_long 1: the condition of making uses the cell's long mean M (rate 1/16384 a step) in place of abar (after E52's worker's mk_mean) */
    double prune_only, prune_ratio;  /* prune_only 1: under TIMING no strength changes by the rule; every connection from an excitatory cell keeps two slow sums, of the times its sender was active before the receiver (c_cu) and after it (c_e0), and is withdrawn when the second exceeds prune_ratio times the first; the strengths stay shared by norm=1 */
    double wcap; double *pk;        /* wcap h: with norm=1 no excitatory connection holds more than h of the excitatory total (the excess goes to the cell's other excitatory connections); pk: slow peak of the cell's own output, for the control two_sided == 3 */
    double inh_learn;               /* inh_learn 1: under TIMING a connection from an inhibitory sender grows with the joint activity of its two ends (dw = eta_s F_j x_i); with norm=1 the inhibitory total is conserved, so a cell's inhibition moves to the senders active when it is */
    double rate_norm; double *M;     /* rate_norm 1: under TIMING the change is divided by the sender's own slow mean activity M (rate 1/16384), so that a sender active rarely is not at a disadvantage */
    double a_make;                  /* a cell makes a connection only while its mean activity is below a_make (0: the definition's a*) */
    double start_w, make_period;   /* strength of a new connection; least number of steps between two makings by one cell */
    double no_make, lines_learn;   /* no_make 1: no connection is made; lines_learn 1: under TIMING the connections from input lines change too (F has NZ entries) */                 /* rule TIMING: a fast trace of every cell's output, its time constant, the rate */
         /* the largest gain a cell can take */
    /* birth (given) */
    const int8_t *kind; const double *sig;          /* sig has NZ entries */
    const int32_t *nmem; const int16_t *mem;        /* members of each cell's neighbourhood: NCELL x maxd */
    /* connections, by receiving cell: NCELL x maxd */
    int32_t *deg; int16_t *c_src; int16_t *c_mi; double *c_w, *c_e0; int32_t *c_t0, *c_k0; double *c_cu; int32_t *c_nb;
    uint8_t *issrc;                 /* NCELL x maxd: is member m a source */
    /* cells */
    double *v, *xp, *p, *th, *abar, *vt, *E; int32_t *klast;
    double *z;                      /* NZ */
    uint64_t rng;
    /* counters */
    int64_t made, given, given_by_use, k;
    double active_sum;              /* sum over steps of the number of cells with output not zero */
    double qpow[4097];
    /* added in TISSUE_DELAY2 (NULL: as before): a delay in steps for every pair (cell, member of its neighbourhood), drawn once at birth
       (c_dm, NCELL x maxd, indexed like mem), so that a connection made later has the delay of its pair; histories of outputs zh and of
       fast traces Fh (each hlen x NCELL). A connection with delay D carries the sender's output of 1 + D steps before. Under the rule
       TIMING, both the strength change and the two withdrawal sums of prune_only compare the receiver with the sender's activity AS IT
       ARRIVES: the delayed output and the sender's fast trace as it was D steps before (the trace of the delayed output), since that is
       what the receiving end has. The control two_sided == 3 takes the largest arriving output as it arrives as well. */
    int32_t *c_dm; double *zh; double *Fh; int hlen;
} Tis;

static inline uint64_t rnd(Tis *t) { uint64_t x = t->rng; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; t->rng = x; return x * 2685821657736338717ULL; }

static void remove_conn(Tis *t, int i, int d) {
    int md = t->maxd, last = t->deg[i] - 1; int64_t b = (int64_t)i * md;
    t->issrc[b + t->c_mi[b + d]] = 0;
    if (d != last) {
        t->c_src[b + d] = t->c_src[b + last]; t->c_mi[b + d] = t->c_mi[b + last]; t->c_w[b + d] = t->c_w[b + last];
        t->c_e0[b + d] = t->c_e0[b + last]; t->c_t0[b + d] = t->c_t0[b + last]; t->c_k0[b + d] = t->c_k0[b + last];
        t->c_cu[b + d] = t->c_cu[b + last]; t->c_nb[b + d] = t->c_nb[b + last];
    }
    t->deg[i] = last;
}

static inline double conn_trace(const Tis *t, int64_t idx, int s, int64_t k) {      /* the connection's trace before step k */
    int64_t d = k - t->c_t0[idx];
    double e0 = t->c_e0[idx];
    if (e0 == 0.0) return t->E[s];
    return t->E[s] - e0 * (d <= 4096 ? t->qpow[d] : 0.0);
}

/* TISSUE_DELAY2: what arrives through connection b + d (receiver base b) from sender s at step k: x_s(k - 1 - D) for a delayed pair, z[s]
   otherwise; and the sender's fast trace as it arrives: F_s as it was before step k - D. */
static inline int pair_delay(const Tis *t, int64_t b, int d, int s) {
    if (!t->c_dm || s >= NCELL) return 0;
    return t->c_dm[b + t->c_mi[b + d]];
}
static inline double arriving(const Tis *t, int64_t b, int d, int s, int64_t k) {
    const int D = pair_delay(t, b, d, s);
    if (D <= 0) return t->z[s];
    return (k - 1 - D >= 0) ? t->zh[((k - 1 - D) % t->hlen) * NCELL + s] : 0.0;
}
static inline double arriving_trace(const Tis *t, int64_t b, int d, int s, int64_t k) {
    const int D = pair_delay(t, b, d, s);
    if (D <= 0) return t->F[s];
    return (k - D >= 0) ? t->Fh[((k - D) % t->hlen) * NCELL + s] : 0.0;
}

/* one step; byte is the byte of the present position; returns the number of cells with output not zero */
static int step(Tis *t, int byte, double *x /* NCELL, out */) {
    const int md = t->maxd; const int64_t k = t->k; const int phi = (int)(k % S);
    double *z = t->z; const double a = t->astar;
    if (phi == 0) { memset(z + LINE0, 0, NIN * sizeof(double)); z[LINE0 + byte] = 1.0; }
    memcpy(z, t->xp, NCELL * sizeof(double));
    if (t->Fh) memcpy(t->Fh + (k % t->hlen) * NCELL, t->F, NCELL * sizeof(double));   /* the fast traces before step k */
    int nact = 0;
    for (int i = 0; i < NCELL; i++) {
        if (t->kind[i] == FBK) { x[i] = 0.0; continue; }
        const int64_t b = (int64_t)i * md; const int dg = t->deg[i];
        double A = 0.0;
        double zs = 0.0, zmx = 0.0;
        for (int d = 0; d < dg; d++) { int s = t->c_src[b + d]; const double zz = arriving(t, b, d, s, k);
            A += t->sig[s] * t->c_w[b + d] * zz; zs += zz; if (t->sig[s] > 0.0 && zz > zmx) zmx = zz; }
        A *= t->gain[i];
        const double m = A - t->p[i];
        const double v = t->v[i] + (A - t->v[i]) / t->tauv[i];       /* TAU_V = 2 in the definition: v/2 + A/2 */
        double xi = v - t->th[i]; if (xi < 0.0) xi = 0.0; if (xi > 1.0) xi = 1.0; if (xi < t->floor_x) xi = 0.0;
        const double xp = t->xp[i];
        /* strength changes, from the traces before the step and x(k-1) */
        if (t->rule != FIXED) {
            const int inner_coinc = (t->rule == COINC && t->kind[i] == INNER);
            if (t->rule == TIMING) {
                /* a connection from an excitatory cell grows when the sender was active before the receiver and falls when it
                   was active after it: dw = eta_s ( F_j x_i(k) - F_i x_j(k-1) ), F the fast trace of a cell's output before the step */
                const double Fi = t->F[i];
                if (xi != 0.0 || Fi != 0.0)
                    for (int d = 0; d < dg; d++) { int s = t->c_src[b + d];
                        if (t->sig[s] < 0.0) { if (t->inh_learn != 0.0 && xi != 0.0) { double w = t->c_w[b + d] + t->eta_s * arriving_trace(t, b, d, s, k) * xi; t->c_w[b + d] = w > 1.0 ? 1.0 : w; } continue; }
                        if (s >= NCELL && t->lines_learn == 0.0) continue;
                        const double Fs_a = arriving_trace(t, b, d, s, k), zs_a = arriving(t, b, d, s, k);   /* as it arrives */
                        if (t->prune_only != 0.0) { t->c_cu[b + d] = t->c_cu[b + d] * (1.0 - 1.0 / 16384.0) + Fs_a * xi; t->c_e0[b + d] = t->c_e0[b + d] * (1.0 - 1.0 / 16384.0) + Fi * zs_a; continue; }
                        double dw = t->eta_s * (Fs_a * xi - Fi * zs_a); if (t->rate_norm != 0.0) dw /= (t->M[s] + 1.0 / 1024);
                        double w = t->c_w[b + d] + dw;
                        t->c_w[b + d] = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w); }
            } else if (!inner_coinc) {
                if (xp != 0.0) {
                    const double f = ETA * m * xp;
                    for (int d = 0; d < dg; d++) { int s = t->c_src[b + d]; if (s >= NCELL) continue;
                        double w = t->c_w[b + d] - f * t->sig[s] * conn_trace(t, b + d, s, k);
                        t->c_w[b + d] = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w); }
                }
            } else {
                const double fe = ETA * xp * (xp - t->vt[i]), fi = ETA * (xp - a);
                for (int d = 0; d < dg; d++) { int s = t->c_src[b + d]; if (s >= NCELL) continue;
                    const double sg = t->sig[s];
                    if (sg > 0.0 && xp == 0.0) continue;
                    double w = t->c_w[b + d] + (sg > 0.0 ? fe : fi) * conn_trace(t, b + d, s, k);
                    t->c_w[b + d] = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w); }
            }
            if (t->norm != 0.0) {
                double se = 0.0, si = 0.0;
                for (int d = 0; d < dg; d++) { if (t->sig[t->c_src[b + d]] > 0.0) se += t->c_w[b + d]; else si += t->c_w[b + d]; }
                const double fe = se > 0.0 ? t->we / se : 1.0, fi = si > 0.0 ? t->wi / si : 1.0;
                for (int d = 0; d < dg; d++) { double w = t->c_w[b + d] * (t->sig[t->c_src[b + d]] > 0.0 ? fe : fi); t->c_w[b + d] = w > 1.0 ? 1.0 : w; }
                if (t->wcap > 0.0) {
                    const double cap = t->wcap * t->we;
                    for (int it = 0; it < 4; it++) {
                        double exc = 0.0, un = 0.0;
                        for (int d = 0; d < dg; d++) { if (t->sig[t->c_src[b + d]] <= 0.0) continue;
                            if (t->c_w[b + d] > cap) { exc += t->c_w[b + d] - cap; t->c_w[b + d] = cap; } else if (t->c_w[b + d] < cap) un += t->c_w[b + d]; }
                        if (exc <= 0.0 || un <= 0.0) break;
                        const double f = (un + exc) / un;
                        for (int d = 0; d < dg; d++) if (t->sig[t->c_src[b + d]] > 0.0 && t->c_w[b + d] < cap) t->c_w[b + d] *= f;
                    }
                }
            }
            /* withdrawal by use: only when the cell was active */
            if (t->by_use && xp != 0.0) {
                double mx = 0.0;
                for (int d = 0; d < dg; d++) { int s = t->c_src[b + d]; if (s >= NCELL) continue;
                    double cu = (1.0 - 1.0 / PS) * t->c_cu[b + d] + (1.0 / PS) * conn_trace(t, b + d, s, k) * xp;
                    t->c_cu[b + d] = cu; if (cu > mx) mx = cu; }
                const double lvl = GIVE_UP * mx;
                for (int d = 0; d < dg; d++) { int s = t->c_src[b + d]; if (s >= NCELL) continue;
                    if (t->c_cu[b + d] < lvl) t->c_nb[b + d]++; else t->c_nb[b + d] = 0; }
            }
        }
        /* prediction, threshold, averages */
        t->p[i] += m / TAU;
        double th = t->th[i] + ETA_T * (xi - a); t->th[i] = th < 0.0 ? 0.0 : th;
        t->abar[i] = (1.0 - 1.0 / PS) * t->abar[i] + (1.0 / PS) * xi;
        if (t->rule == COINC) t->vt[i] = (1.0 - 1.0 / TAU_VT) * t->vt[i] + (1.0 / TAU_VT) * xi * xi / a;
        if (t->two_sided == 3) {
            /* the cell scales its arrival so that its own largest outputs are as large as the largest outputs arriving from its excitatory senders:
               both peaks decay with rate 1/16384 a step; ibar holds the peak of what arrives, pk the cell's own */
            const double q = t->ibar[i] * (1.0 - 1.0 / 16384.0), pp = t->pk[i] * (1.0 - 1.0 / 16384.0);
            t->ibar[i] = zmx > q ? zmx : q; t->pk[i] = xi > pp ? xi : pp;
            const double Q = t->rho * t->ibar[i], P = t->pk[i];
            double g = t->gain[i] * (1.0 + t->eta_g * (Q - P) / (Q + P + 1e-30)); t->gain[i] = g < 1.0 / 64 ? 1.0 / 64 : (g > t->gmax ? t->gmax : g);
        } else if (t->two_sided == 2) {
            t->ibar[i] = (1.0 - 1.0 / PS) * t->ibar[i] + (1.0 / PS) * (dg ? zs / dg : 0.0);
            const double tg = t->rho * t->ibar[i], ab = t->abar[i];
            double g = t->gain[i] * (1.0 + t->eta_g * (tg - ab) / (tg + ab + 1e-30)); t->gain[i] = g < 1.0 / 64 ? 1.0 / 64 : (g > t->gmax ? t->gmax : g);
        } else if (t->two_sided) { double g = t->gain[i] * (1.0 + t->eta_g * (a - xi) / a); t->gain[i] = g < 1.0 / 64 ? 1.0 / 64 : (g > t->gmax ? t->gmax : g); }
        t->v[i] = v; x[i] = xi; if (xi != 0.0) nact++;
    }
    /* traces of the sending cells: z of this step is x(k-1) */
    for (int j = 0; j < NCELL; j++) t->E[j] = (1.0 - 1.0 / TAU) * t->E[j] + (1.0 / TAU) * z[j];
    if (t->rule == TIMING) for (int j = 0; j < NZ; j++) t->F[j] += (z[j] - t->F[j]) / t->tau_f;
    if (t->rate_norm != 0.0 || t->mk_long != 0.0) for (int j = 0; j < NZ; j++) t->M[j] += (z[j] - t->M[j]) / 16384.0;
    /* giving up, then making */
    if (t->rule != FIXED) {
        for (int i = 0; i < NCELL; i++) {
            if (t->kind[i] == FBK) continue;
            const int64_t b = (int64_t)i * md;
            if (t->xp[i] != 0.0 || (k % PS) == 0 || t->rule == TIMING) {                /* a strength can only have fallen at an active cell; ages are looked at once a period */
                for (int d = t->deg[i] - 1; d >= 0; d--) { int s = t->c_src[b + d]; if (s >= NCELL) continue;
                    if (k - t->c_k0[b + d] < PS) continue;
                    if (t->c_w[b + d] < GIVE_UP) { remove_conn(t, i, d); t->given++; }
                    else if (t->prune_only != 0.0 && t->sig[s] > 0.0 && (k - t->c_k0[b + d]) >= 4096 && t->c_e0[b + d] > 0.0 && t->c_e0[b + d] > t->prune_ratio * t->c_cu[b + d]) { remove_conn(t, i, d); t->given++; t->given_by_use++; }
                    else if (t->by_use && t->c_nb[b + d] >= PS) { remove_conn(t, i, d); t->given++; t->given_by_use++; }
                }
            }
            if (t->no_make == 0.0 && t->th[i] == 0.0 && (t->mk_long != 0.0 ? t->M[i] : t->abar[i]) < (t->a_make > 0.0 ? t->a_make : a) && (k - t->klast[i]) >= (int64_t)t->make_period) {
                const int free_n = t->nmem[i] - t->deg[i];
                if (free_n > 0) {
                    int pick = (int)(rnd(t) % (uint64_t)free_n), mi = -1;
                    for (int q = 0; q < t->nmem[i]; q++) if (!t->issrc[b + q]) { if (pick == 0) { mi = q; break; } pick--; }
                    int d = t->deg[i]++; int s = t->mem[b + mi];
                    t->c_src[b + d] = (int16_t)s; t->c_mi[b + d] = (int16_t)mi; t->c_w[b + d] = t->start_w; t->c_k0[b + d] = (int32_t)k;
                    t->c_e0[b + d] = (s < NCELL && t->rule != TIMING) ? t->E[s] : 0.0; t->c_t0[b + d] = (int32_t)(k + 1);
                    t->c_cu[b + d] = 0.0; t->c_nb[b + d] = 0; t->issrc[b + mi] = 1;
                    t->klast[i] = (int32_t)k; t->made++;
                }
            }
        }
    }
    memcpy(t->xp, x, NCELL * sizeof(double));
    if (t->zh) memcpy(t->zh + (k % t->hlen) * NCELL, x, NCELL * sizeof(double));      /* the outputs of step k */
    t->k = k + 1; t->active_sum += nact;
    return nact;
}

void tis_init(Tis *t) { for (int d = 0; d <= 4096; d++) t->qpow[d] = pow(1.0 - 1.0 / TAU, (double)d); }

/* Reads bytes[b0..b1); records the outputs at the reading step of the bytes in [r0, r1) into rec (float32, (r1-r0) x NCELL). */
void tis_run(Tis *t, const uint8_t *bytes, int64_t b0, int64_t b1, int64_t r0, int64_t r1, float *rec) {
    double x[NCELL];
    /* numbers below the smallest normal number are taken as zero: they cost about a hundred times a normal operation and carry nothing */
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON); _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    for (int64_t b = b0; b < b1; b++) {
        for (int ph = 0; ph < S; ph++) step(t, bytes[b], x);
        if (b >= r0 && b < r1) { float *o = rec + (b - r0) * NCELL; for (int i = 0; i < NCELL; i++) o[i] = (float)x[i]; }
    }
}
