/* E63, the tissue's program written so that it works only where something changes: the compiled core.

   Task: TASK_E63_THE_TISSUE_PROGRAM_EVENT_DRIVEN_20261119.md (reviewer's branch). The tissue is the definition as E63's program
   TISSUE_STEP1/tissue_step1.py computes it (operations 1 to 8 of section 5.4 of the brief of step 1), with four switches:

     floor          an output below it is 0 (0: no floor)
     tol            a sender passes on the change of its output since it last sent only when that change exceeds tol (0: any change)
     trace_sender   1: one trace for each sending cell or line; 0: a trace for each connection
     strength_byte  1: strengths changed once a byte (at the reading step) from sums kept by the receiving cell; 0: at every step

   Design (section 3 of the task):
   1. Connections are stored once, in slots; each receiving cell holds the list of its incoming slots in the order of making, and each
      sender (cell or line) the list of its outgoing slots. Both lists are kept correct when a connection is made or given up.
   2. Each cell keeps its arrival A from step to step. At the start of a step each sender whose value moved passes on sigma*w*(change)
      to its receivers, which go on the work list. The arrivals of a step are all collected before any cell is computed, so the result
      does not depend on the order of the cells. A cell leaves the work list when it is stationary: computing it again would leave its
      state, output and threshold unchanged and change no strength.
   3. What only relaxes while nothing arrives (prediction p, activity average abar, sliding threshold vt, traces e) carries the step
      up to which it is brought; it is brought up to date when next needed: by the recurrence itself for gaps up to 64 steps, by the
      closed form with pow() beyond.
   4. Making is looked at only for cells whose period has passed (a queue of timers); a cell whose neighbourhood is full is left out
      until one of its connections is given up. Giving up is tested where a strength has just changed and when a connection reaches
      the age of 128 steps. The draws of the life generator are Python's Mersenne Twister and Random.choice, reproduced exactly.

   One instance per process (global state). Built as a shared library: gcc -O2 -ffp-contract=off -shared -fPIC evt_core.c -o evt_core.so
*/
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TAU_V_A 0.5
#define DEC32 0.96875          /* 1 - 1/32 */
#define INC32 0.03125
#define DEC128 0.9921875       /* 1 - 1/128 */
#define INC128 0.0078125
#define DEC4096 0.999755859375 /* 1 - 1/4096 */
#define INC4096 (1.0 / 4096)
#define ETA 0.125
#define ETA_T 0.0078125
static double ASTAR_V = 0.125;   /* a*, the target of activity (a parameter: evt_set_astar before evt_init) */
#define ASTAR ASTAR_V
#define PS 128
#define GIVE_UP 0.0078125
#define START_W 0.125
#define EMIT 0.5
#define LAZY_ITER 64

typedef struct { int *a; int n, cap; } ivec;

static void iv_push(ivec *v, int x) {
    if (v->n == v->cap) { v->cap = v->cap ? 2 * v->cap : 8; v->a = realloc(v->a, sizeof(int) * v->cap); }
    v->a[v->n++] = x;
}
static void iv_remove_ordered(ivec *v, int x) {
    for (int i = 0; i < v->n; i++) if (v->a[i] == x) { memmove(v->a + i, v->a + i + 1, sizeof(int) * (v->n - i - 1)); v->n--; return; }
}
static void iv_remove_swap(ivec *v, int x) {
    for (int i = 0; i < v->n; i++) if (v->a[i] == x) { v->a[i] = v->a[--v->n]; return; }
}

/* ---------------------------------------------------------------------------------------------- Python's Mersenne Twister */
static uint32_t mt[624];
static int mti;
static uint32_t genrand(void) {
    uint32_t y;
    static const uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
    if (mti >= 624) {
        int kk;
        for (kk = 0; kk < 624 - 397; kk++) { y = (mt[kk] & 0x80000000U) | (mt[kk + 1] & 0x7fffffffU); mt[kk] = mt[kk + 397] ^ (y >> 1) ^ mag01[y & 1U]; }
        for (; kk < 623; kk++) { y = (mt[kk] & 0x80000000U) | (mt[kk + 1] & 0x7fffffffU); mt[kk] = mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 1U]; }
        y = (mt[623] & 0x80000000U) | (mt[0] & 0x7fffffffU); mt[623] = mt[396] ^ (y >> 1) ^ mag01[y & 1U];
        mti = 0;
    }
    y = mt[mti++];
    y ^= (y >> 11); y ^= (y << 7) & 0x9d2c5680U; y ^= (y << 15) & 0xefc60000U; y ^= (y >> 18);
    return y;
}
static int randbelow(int n) {           /* Random._randbelow_with_getrandbits, n < 2**31 */
    int k = 0; while ((n >> k) != 0) k++;  /* n.bit_length() */
    uint32_t r = genrand() >> (32 - k);
    while (r >= (uint32_t)n) r = genrand() >> (32 - k);
    return (int)r;
}
void evt_set_rng(const uint32_t *state, int pos) { memcpy(mt, state, sizeof(mt)); mti = pos; }
void evt_get_rng(uint32_t *state, int *pos) { memcpy(state, mt, sizeof(mt)); *pos = mti; }

/* ---------------------------------------------------------------------------------------------- state */
static int NC, NZ, S_STEPS, LINE0, NIN;     /* cells; senders (cells then lines); steps a byte; first input line; input lines */
static double FLOOR, TOL;
static int TRACE_SENDER, STRENGTH_BYTE, LIFE, F11W;
static int64_t K;                            /* the next step */

/* senders */
static double *sig, *z, *zsent;              /* sigma; present value; value last passed on */
static uint8_t *learnable, *exempt;          /* strengths of its connections change; its connections are never given up */
static double *E; static int64_t *Est;       /* trace of a sender (trace_sender) and the step it holds */
static ivec *out;
/* cells */
static double *A, *v, *x, *p, *th, *abar, *vt, *dA, *sm, *sexc, *sinh_;
static int64_t *pst, *ast, *vst, *klast;
static int *ncomp;
static uint8_t *rule, *theta_on, *inwl, *due, *full, *touched, *inlearn, *ischg;
static int *ninh;                            /* connections from inhibitory learnable senders into a cell (coincidence rule) */
static ivec *in, *members;
static int *driver, NDRV;
static double *Astep, *mstep;
static uint64_t *duebits, *sortbits; static int NW;
static inline void due_set(int i, int on) { due[i] = (uint8_t)on; if (on) duebits[i >> 6] |= 1ULL << (i & 63); else duebits[i >> 6] &= ~(1ULL << (i & 63)); }
/* Plan 2 (turnover with a trial period): TURN 1 makes a connection every PMK steps whatever the cell's activity; a connection from a
   cell made during the life is on trial for TTR steps; at the end of the trial it is withdrawn if its use (the mean over the steps at
   which the receiving cell was active of the sender's trace times that activity) is 0, or below THTR times the mean running use of
   the receiving cell's other connections from cells. */
static int TURN = 0, PMK = 128, TTR = 512; static double THTR = 0.25; static int64_t TRIAL_GONE = 0, TRIAL_KEPT = 0;
static uint8_t *c_trial; static double *c_tsum; static int *c_tn;
static int64_t *trq_t, *trq_seq; static int *trq_c, trq_n, trq_h, trq_cap;
static int is_cell_sender(int j);
void evt_set_turnover(int on, int pmk, int ttr, double th) { TURN = on; PMK = pmk; TTR = ttr; THTR = th; }
void evt_trial_counts(int64_t *o) { o[0] = TRIAL_GONE; o[1] = TRIAL_KEPT; }
/* Plan 3 (a cell's total incoming strength held): NORM 1 scales the strengths of a cell's learnable incoming connections so that those
   from excitatory senders sum to C_E and those from inhibitory senders to C_I (each kind apart; a kind with sum 0 is left), after the
   cell's strengths change at the end of a byte, after it makes or loses a connection, and once at the start. */
static int NORM = 0; static double C_E = 0.75, C_I = 0.25; static int64_t NORM_CALLS = 0;
void evt_set_norm(int on, double ce, double ci) { NORM = on; C_E = ce; C_I = ci; }
static void normalise(int i);
/* Plan 5 (a cell's own gain): GAIN 1 multiplies a cell's arrival by its gain g. Once a byte, at the reading step, g is moved so that
   the cell's running mean output (abar) goes toward RHO times the running mean of its sources' values (ib, a running mean, 1/128 a
   step, of the mean value of the senders of its incoming connections): g <- g (1 + ETA_G (t - abar) / (t + abar)), t = RHO ib, kept
   within [GMIN, GMAX]. The reviewer's form of 19 November (gain by ratio), moved once a byte in place of every step. */
static int GAIN = 0; static double ETA_G = 1.0 / 128, RHO = 0.875, GMAX = 8.0, GMIN = 1.0 / 64;
static double *gain, *Zs, *ib; static int64_t *ibst;
void evt_set_gain(int on, double eta, double rho, double gmax) { GAIN = on; ETA_G = eta; RHO = rho; GMAX = gmax; }
void evt_gains(double *o) { memcpy(o, gain, 8 * (size_t)NC); }
static int CAPIN = 1 << 30; void evt_set_cap(int c) { CAPIN = c; }
static int64_t *slk; static int NSLEEP; static double SLEEP_EMUL, SLEEP_WAKES;   /* step from which a cell sleeps (-1: awake) */                /* the arrival and mismatch of the last step a cell was computed (for checks) */
/* connections */
static int CAP, NSLOT;
static int *c_snd, *c_rcv;
static double *c_w, *c_e, *c_ub, *c_use;
static int *c_cnt;
static int USE_W, USE_EXEMPT_FB; static uint8_t *usefb; static int64_t WITHDRAWN_USE;
static int64_t *c_est, *c_k0, *c_seq, SEQ;
static uint8_t *c_alive, *c_mark;
static ivec freeslots;
static int NLIVE;
/* lists */
static ivec wl, wl2, chg, chg2, gone, learnl, dal;
static int64_t *tq_t; static int *tq_c, tq_n, tq_h, tq_cap;    /* timers of making: (step, cell) */
static int64_t *aq_t, *aq_seq; static int *aq_c, aq_n, aq_h, aq_cap;  /* ages: (step, slot, seq) */
/* events and counters */
static int64_t *ev_made, *ev_gone; static size_t nev_made, cap_made, nev_gone, cap_gone;   /* size_t: these exceed an int near 250,000 bytes */
static int64_t MADE, GIVEN_UP;
static double CNT[8];   /* 0 cell updates, 1 senders passing on, 2 connection visits in passing on, 3 visits for strengths,
                           4 visits for making and giving up, 5 steps, 6 cells active (x > 0) after a step, 7 lazy updates */

static double relax(double q, double target, double dec, double inc, int64_t n) {
    if (n <= 0) return q;
    if (n <= LAZY_ITER) { for (int64_t i = 0; i < n; i++) q = dec * q + inc * target; return q; }
    return target + (q - target) * pow(dec, (double)n);
}
static double relax_vt(double q, double xx, int64_t n) {
    if (n <= 0) return q;
    double t = INC4096 * xx * xx / ASTAR;
    if (n <= LAZY_ITER) { for (int64_t i = 0; i < n; i++) q = DEC4096 * q + t; return q; }
    double fix = xx * xx / ASTAR;
    return fix + (q - fix) * pow(DEC4096, (double)n);
}
static inline void bring_p(int i, int64_t upto) { if (pst[i] < upto) { p[i] = relax(p[i], A[i], DEC32, INC32, upto - pst[i]); pst[i] = upto; CNT[7]++; } }
static inline void bring_a(int i, int64_t upto) { if (ast[i] < upto) { abar[i] = relax(abar[i], x[i], DEC128, INC128, upto - ast[i]); ast[i] = upto; } }
static inline void bring_vt(int i, int64_t upto) { if (vt && vst[i] < upto) { vt[i] = relax_vt(vt[i], x[i], upto - vst[i]); vst[i] = upto; } }
static inline void bring_ib(int i, int64_t upto) { if (GAIN && ibst[i] < upto) { int n = in[i].n; ib[i] = relax(ib[i], n ? Zs[i] / n : 0.0, DEC128, INC128, upto - ibst[i]); ibst[i] = upto; } }
static inline void touch(int i, int64_t upto) { bring_p(i, upto); bring_a(i, upto); bring_vt(i, upto); bring_ib(i, upto); }
static inline double trace_of(int c, int64_t upto) {       /* the trace of connection c held at the start of step `upto` */
    if (TRACE_SENDER) { int j = c_snd[c]; if (Est[j] < upto) { E[j] = relax(E[j], zsent[j], DEC32, INC32, upto - Est[j]); Est[j] = upto; } return E[j]; }
    if (c_est[c] < upto) { c_e[c] = relax(c_e[c], zsent[c_snd[c]], DEC32, INC32, upto - c_est[c]); c_est[c] = upto; }
    return c_e[c];
}
static inline void wl_add(ivec *l, int i) { if (!inwl[i]) { inwl[i] = 1; iv_push(l, i); } }


/* A sleeping cell is one the present program would compute at every step with nothing observable changing: its output is 0 before and
   after, its threshold is 0, its arrival A is fixed and below the floor (0 with no floor), nothing changes its strengths. Then its
   output stays 0 (v moves between its present value and A, both below the floor) and only v, p, abar and vt move. The wake computes
   those steps one by one with the present program's own formulas, in the same order, so every number is identical, and it stops at
   the step where the present program would have found the cell stationary and taken it off the list. Returns 1 if the present program
   would still have the cell on its list for step `upto`. */
static int wake(int i, int64_t upto) {
    int64_t s = slk[i];
    slk[i] = -1; NSLEEP--; SLEEP_WAKES++;
    int on = 1;
    const double xn = 0.0;
    for (; s < upto; s++) {
        SLEEP_EMUL++;
        double Ai = gain[i] * A[i], m = Ai - p[i];
        double vn = TAU_V_A * v[i] + TAU_V_A * Ai;
        Astep[i] = Ai; mstep[i] = m;
        if (GAIN) { int nn = in[i].n; ib[i] = DEC128 * ib[i] + INC128 * (nn ? Zs[i] / nn : 0.0); ibst[i] = s + 1; }
        p[i] = DEC32 * p[i] + INC32 * Ai; pst[i] = s + 1;
        abar[i] = DEC128 * abar[i] + INC128 * xn; ast[i] = s + 1;
        if (vt) { vt[i] = DEC4096 * vt[i] + INC4096 * xn * xn / ASTAR; vst[i] = s + 1; }
        int still = (vn == v[i]);
        v[i] = vn;
        if (still) { on = 0; break; }
    }
    inwl[i] = 0;
    return on;
}
static inline void wake_to(int i, int64_t upto, ivec *l) { if (slk[i] >= 0) { if (wake(i, upto)) wl_add(l, i); } }
void evt_wake_all(void) { for (int i = 0; i < NC; i++) wake_to(i, K, &wl); }
double evt_sleep_stats(double *o) { o[0] = NSLEEP; o[1] = SLEEP_EMUL; o[2] = SLEEP_WAKES; return 0; }
static int new_slot(void) {
    if (freeslots.n) return freeslots.a[--freeslots.n];
    if (NSLOT == CAP) {
        CAP = CAP ? 2 * CAP : 4096;
        c_snd = realloc(c_snd, sizeof(int) * CAP); c_rcv = realloc(c_rcv, sizeof(int) * CAP);
        c_w = realloc(c_w, sizeof(double) * CAP); c_e = realloc(c_e, sizeof(double) * CAP); c_ub = realloc(c_ub, sizeof(double) * CAP); c_use = realloc(c_use, sizeof(double) * CAP); c_cnt = realloc(c_cnt, sizeof(int) * CAP);
        c_est = realloc(c_est, sizeof(int64_t) * CAP); c_k0 = realloc(c_k0, sizeof(int64_t) * CAP); c_seq = realloc(c_seq, sizeof(int64_t) * CAP);
        c_alive = realloc(c_alive, CAP); c_mark = realloc(c_mark, CAP);
        c_trial = realloc(c_trial, CAP); c_tsum = realloc(c_tsum, sizeof(double) * CAP); c_tn = realloc(c_tn, sizeof(int) * CAP);
    }
    return NSLOT++;
}
static void aq_push(int64_t t, int c, int64_t seq) {
    if (aq_n == aq_cap) { aq_cap = aq_cap ? 2 * aq_cap : 4096; aq_t = realloc(aq_t, 8 * aq_cap); aq_seq = realloc(aq_seq, 8 * aq_cap); aq_c = realloc(aq_c, 4 * aq_cap); }
    aq_t[aq_n] = t; aq_seq[aq_n] = seq; aq_c[aq_n] = c; aq_n++;
}
static void tq_push(int64_t t, int i) {
    if (tq_n == tq_cap) { tq_cap = tq_cap ? 2 * tq_cap : 4096; tq_t = realloc(tq_t, 8 * tq_cap); tq_c = realloc(tq_c, 4 * tq_cap); }
    tq_t[tq_n] = t; tq_c[tq_n] = i; tq_n++;
}
static void compact_queues(void) {
    if (aq_h > 65536 && aq_h > aq_n / 2) { memmove(aq_t, aq_t + aq_h, 8 * (aq_n - aq_h)); memmove(aq_seq, aq_seq + aq_h, 8 * (aq_n - aq_h)); memmove(aq_c, aq_c + aq_h, 4 * (aq_n - aq_h)); aq_n -= aq_h; aq_h = 0; }
    if (trq_h > 65536 && trq_h > trq_n / 2) { memmove(trq_t, trq_t + trq_h, 8 * (trq_n - trq_h)); memmove(trq_seq, trq_seq + trq_h, 8 * (trq_n - trq_h)); memmove(trq_c, trq_c + trq_h, 4 * (trq_n - trq_h)); trq_n -= trq_h; trq_h = 0; }
    if (tq_h > 65536 && tq_h > tq_n / 2) { memmove(tq_t, tq_t + tq_h, 8 * (tq_n - tq_h)); memmove(tq_c, tq_c + tq_h, 4 * (tq_n - tq_h)); tq_n -= tq_h; tq_h = 0; }
}

/* ---------------------------------------------------------------------------------------------- set-up */
#define FREE(q) do { free(q); q = NULL; } while (0)
static void free_all(void) {
    if (out) { for (int j = 0; j < NZ; j++) free(out[j].a); FREE(out); }
    if (in) { for (int i = 0; i < NC; i++) { free(in[i].a); free(members[i].a); } FREE(in); FREE(members); }
    FREE(sig); FREE(z); FREE(zsent); FREE(learnable); FREE(exempt); FREE(E); FREE(Est);
    FREE(A); FREE(v); FREE(x); FREE(p); FREE(th); FREE(abar); FREE(vt); FREE(dA); FREE(sm); FREE(sexc); FREE(sinh_);
    FREE(pst); FREE(ast); FREE(vst); FREE(klast); FREE(ncomp); FREE(rule); FREE(theta_on); FREE(inwl); FREE(due); FREE(full);
    FREE(touched); FREE(inlearn); FREE(ischg); FREE(ninh); FREE(driver); FREE(Astep); FREE(mstep); FREE(slk); FREE(gain); FREE(Zs); FREE(ib); FREE(ibst); FREE(duebits); FREE(sortbits);
    FREE(c_snd); FREE(c_rcv); FREE(c_w); FREE(c_e); FREE(c_ub); FREE(c_use); FREE(c_cnt); FREE(usefb); FREE(c_est); FREE(c_k0); FREE(c_seq); FREE(c_alive); FREE(c_mark); FREE(c_trial); FREE(c_tsum); FREE(c_tn); FREE(trq_t); FREE(trq_seq); FREE(trq_c); trq_n = trq_h = trq_cap = 0; TRIAL_GONE = TRIAL_KEPT = 0;
    FREE(freeslots.a); freeslots.n = freeslots.cap = 0;
    ivec *ls[] = {&wl, &wl2, &chg, &chg2, &gone, &learnl, &dal};
    for (int q = 0; q < 7; q++) { FREE(ls[q]->a); ls[q]->n = ls[q]->cap = 0; }
    FREE(tq_t); FREE(tq_c); FREE(aq_t); FREE(aq_seq); FREE(aq_c); tq_n = tq_h = tq_cap = aq_n = aq_h = aq_cap = 0;
    FREE(ev_made); FREE(ev_gone); nev_made = cap_made = nev_gone = cap_gone = 0;
    CAP = NSLOT = NLIVE = 0;
}

/* nc cells, nz senders; params: floor, tol; flags: trace_sender, strength_byte, life (making and giving up), f11w, coincidence
   (hold vt); steps a byte, first input line, number of input lines */
void evt_init(int nc, int nz, double floor_, double tol, int trace_sender, int strength_byte, int life, int f11w, int hold_vt,
              int steps, int line0, int nin) {
    free_all();
    NC = nc; NZ = nz; FLOOR = floor_; TOL = tol; TRACE_SENDER = trace_sender; STRENGTH_BYTE = strength_byte; LIFE = life; F11W = f11w;
    S_STEPS = steps; LINE0 = line0; NIN = nin; K = 0; SEQ = 0; MADE = GIVEN_UP = 0; memset(CNT, 0, sizeof(CNT));
    sig = calloc(nz, 8); z = calloc(nz, 8); zsent = calloc(nz, 8); learnable = calloc(nz, 1); exempt = calloc(nz, 1);
    E = calloc(nz, 8); Est = calloc(nz, 8); out = calloc(nz, sizeof(ivec));
    A = calloc(nc, 8); v = calloc(nc, 8); x = calloc(nc, 8); p = calloc(nc, 8); th = calloc(nc, 8); abar = calloc(nc, 8);
    vt = hold_vt ? calloc(nc, 8) : NULL; dA = calloc(nc, 8); sm = calloc(nc, 8); sexc = calloc(nc, 8); sinh_ = calloc(nc, 8);
    pst = calloc(nc, 8); ast = calloc(nc, 8); vst = calloc(nc, 8); klast = calloc(nc, 8); ncomp = calloc(nc, 4);
    rule = calloc(nc, 1); theta_on = calloc(nc, 1); inwl = calloc(nc, 1); due = calloc(nc, 1); full = calloc(nc, 1);
    touched = calloc(nc, 1); inlearn = calloc(nc, 1); ischg = calloc(nc, 1); ninh = calloc(nc, 4);
    in = calloc(nc, sizeof(ivec)); members = calloc(nc, sizeof(ivec)); Astep = calloc(nc, 8); mstep = calloc(nc, 8);
    NW = (nc + 63) / 64; duebits = calloc(NW, 8); sortbits = calloc((nz + 63) / 64, 8);
    gain = malloc(8 * nc); for (int i = 0; i < nc; i++) gain[i] = 1.0; Zs = calloc(nc, 8); ib = calloc(nc, 8); ibst = calloc(nc, 8);
    slk = malloc(8 * nc); for (int i = 0; i < nc; i++) slk[i] = -1; NSLEEP = 0; SLEEP_EMUL = SLEEP_WAKES = 0;
    driver = NULL; NDRV = 0;
    if (vt) for (int i = 0; i < nc; i++) vt[i] = ASTAR;
}
/* per sender: sigma, learnable, exempt; per cell: rule (0 none, 1 formula 7, 2 coincidence rule), theta_on */
void evt_set_astar(double a) { ASTAR_V = a; }
/* withdrawal by use (W): on/off; fbcells marks senders that are feedback cells; exempt_fb: their connections are not withdrawn by use */
void evt_set_withdraw_use(int on, const uint8_t *fbcells, int exempt_fb) {
    USE_W = on; USE_EXEMPT_FB = exempt_fb; WITHDRAWN_USE = 0;
    usefb = realloc(usefb, NZ); if (fbcells) memcpy(usefb, fbcells, NZ); else memset(usefb, 0, NZ);
}
int64_t evt_withdrawn_by_use(void) { return WITHDRAWN_USE; }
void evt_set_senders(const double *s, const uint8_t *lrn, const uint8_t *ex) {
    memcpy(sig, s, 8 * NZ); memcpy(learnable, lrn, NZ); memcpy(exempt, ex, NZ);
}
void evt_set_cells(const uint8_t *r, const uint8_t *t) { memcpy(rule, r, NC); memcpy(theta_on, t, NC); }
void evt_set_members(const int *ptr, const int *idx) {
    for (int i = 0; i < NC; i++) { members[i].n = 0; for (int q = ptr[i]; q < ptr[i + 1]; q++) iv_push(&members[i], idx[q]); }
}
void evt_set_drivers(const int *d, int n) { driver = malloc(4 * n); memcpy(driver, d, 4 * n); NDRV = n; }
/* the state of the cells before step 0 (the worked example starts from given values) */
void evt_set_cell_state(const double *v0, const double *p0, const double *th0, const double *x0, const double *vt0) {
    memcpy(v, v0, 8 * NC); memcpy(p, p0, 8 * NC); memcpy(th, th0, 8 * NC); memcpy(x, x0, 8 * NC);
    if (vt && vt0) memcpy(vt, vt0, 8 * NC);
    for (int i = 0; i < NC; i++) z[i] = x[i];
}
static void link_slot(int c) {
    int j = c_snd[c], i = c_rcv[c];
    iv_push(&in[i], c); iv_push(&out[j], c);
    if (learnable[j] && sig[j] < 0) ninh[i]++;
    NLIVE++;
}
/* a connection present at birth (or in the example): sender, receiver, strength, trace, step of making */
int evt_add_connection(int j, int i, double w, double e, int64_t k0) {
    int c = new_slot();
    c_snd[c] = j; c_rcv[c] = i; c_w[c] = w; c_e[c] = e; c_ub[c] = 0.0; c_use[c] = 0.0; c_cnt[c] = 0; c_trial[c] = 0; c_est[c] = K; c_k0[c] = k0; c_seq[c] = SEQ++;
    c_alive[c] = 1; c_mark[c] = 0;
    link_slot(c);
    if (LIFE) aq_push(k0 + PS, c, c_seq[c]);
    return c;
}
/* after all connections: arrivals are built by the passing on of step 0; every cell starts on the work list, the timers of making */
void evt_start(void) {
    if (NORM) for (int i = 0; i < NC; i++) normalise(i);
    for (int i = 0; i < NC; i++) wl_add(&wl, i);
    for (int j = 0; j < NC; j++) iv_push(&chg, j);
}
void evt_set_life_cells(const uint8_t *can_make) {     /* cells that make connections (not feedback cells) */
    tq_n = tq_h = 0;
    for (int i = 0; i < NC; i++) if (can_make[i]) tq_push(klast[i] + PMK, i);
}
static int nlinechg; static int *linechg;
void evt_set_line(int j, double val) {                  /* a line's value from this step on (the worked example) */
    if (z[j] != val) { z[j] = val; linechg = realloc(linechg, 4 * (nlinechg + 1)); linechg[nlinechg++] = j; }
}

/* ---------------------------------------------------------------------------------------------- one step */
static void pass_on(int j) {                           /* sender j passes on the change of its value */
    double d = z[j] - zsent[j];
    if (d == 0.0 || fabs(d) <= TOL) return;
    CNT[1]++;
    ivec *o = &out[j];
    if (TRACE_SENDER && Est[j] < K) { E[j] = relax(E[j], zsent[j], DEC32, INC32, K - Est[j]); Est[j] = K; }
    for (int q = 0; q < o->n; q++) {
        int c = o->a[q], i = c_rcv[c];
        if (!TRACE_SENDER && c_est[c] < K) { c_e[c] = relax(c_e[c], zsent[j], DEC32, INC32, K - c_est[c]); c_est[c] = K; }
        if (!inwl[i]) { if (slk[i] >= 0) (void)wake(i, K); touch(i, K); wl_add(&wl, i); }
        A[i] += sig[j] * c_w[c] * d;
        if (GAIN) Zs[i] += d;
    }
    CNT[2] += o->n;
    zsent[j] = z[j];
}
static inline double bound(double w) { return w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w); }
static inline void strength_change(int c, double w_new, int i) {   /* w of connection c into cell i becomes w_new (step K) */
    double w = c_w[c];
    if (w_new == w) return;
    if (dA[i] == 0.0) iv_push(&dal, i);
    dA[i] += sig[c_snd[c]] * (w_new - w) * zsent[c_snd[c]];
    c_w[c] = w_new;
    if (LIFE && !F11W && !exempt[c_snd[c]] && K - c_k0[c] >= PS && w_new < GIVE_UP && !c_mark[c]) { c_mark[c] = 1; iv_push(&gone, c); }
}
static void normalise(int i) {
    ivec *ip = &in[i]; double se = 0.0, si = 0.0;
    for (int q = 0; q < ip->n; q++) { int c = ip->a[q], j = c_snd[c]; if (!learnable[j]) continue; if (sig[j] > 0) se += c_w[c]; else si += c_w[c]; }
    double fe = se > 0.0 ? C_E / se : 1.0, fi = si > 0.0 ? C_I / si : 1.0;
    if (fe == 1.0 && fi == 1.0) return;
    NORM_CALLS++; CNT[3] += 2 * ip->n;
    for (int q = 0; q < ip->n; q++) { int c = ip->a[q], j = c_snd[c]; if (!learnable[j]) continue; strength_change(c, bound(c_w[c] * (sig[j] > 0 ? fe : fi)), i); }
}
static void remove_conn(int c) {
    int j = c_snd[c], i = c_rcv[c];
    if (!inwl[i]) { if (slk[i] >= 0) (void)wake(i, K + 1); touch(i, K + 1); wl_add(&wl2, i); }
    A[i] -= sig[j] * c_w[c] * zsent[j];
    if (GAIN) Zs[i] -= zsent[j];
    iv_remove_ordered(&in[i], c); iv_remove_swap(&out[j], c);
    if (learnable[j] && sig[j] < 0) ninh[i]--;
    if (USE_W && c_w[c] >= GIVE_UP) WITHDRAWN_USE++;
    c_alive[c] = 0; c_w[c] = 0.0; c_mark[c] = 0;
    iv_push(&freeslots, c);
    NLIVE--; GIVEN_UP++;
    CNT[4] += in[i].n;
    if (nev_gone == cap_gone) { cap_gone = cap_gone ? 2 * cap_gone : 1024; ev_gone = realloc(ev_gone, sizeof(int64_t) * 4 * cap_gone); }
    ev_gone[4 * nev_gone] = K; ev_gone[4 * nev_gone + 1] = j; ev_gone[4 * nev_gone + 2] = i; ev_gone[4 * nev_gone + 3] = c_seq[c]; nev_gone++;
    if (full[i]) { full[i] = 0; if (K - klast[i] >= PMK) due_set(i, 1); }
    if (NORM) normalise(i);
}
static int *mark_s; static int mark_cap; static int *cand; static int cand_cap;
static void make_for(int i) {
    if (mark_cap < NZ) { mark_s = realloc(mark_s, 4 * NZ); memset(mark_s, 0, 4 * NZ); mark_cap = NZ; }
    ivec *ip = &in[i], *mb = &members[i];
    for (int q = 0; q < ip->n; q++) { int j = c_snd[ip->a[q]]; if (!exempt[j]) mark_s[j] = 1; }
    if (cand_cap < mb->n) { cand = realloc(cand, 4 * mb->n); cand_cap = mb->n; }
    int nc = 0;
    for (int q = 0; q < mb->n; q++) if (!mark_s[mb->a[q]]) cand[nc++] = mb->a[q];
    for (int q = 0; q < ip->n; q++) mark_s[c_snd[ip->a[q]]] = 0;
    CNT[4] += ip->n + mb->n;
    if (nc == 0) { full[i] = 1; due_set(i, 0); return; }
    int j = cand[randbelow(nc)];
    if (!inwl[i]) { if (slk[i] >= 0) (void)wake(i, K + 1); touch(i, K + 1); wl_add(&wl2, i); }
    int c = new_slot();
    c_snd[c] = j; c_rcv[c] = i; c_w[c] = START_W; c_e[c] = 0.0; c_ub[c] = 0.0; c_use[c] = 0.0; c_cnt[c] = 0; c_trial[c] = 0; c_est[c] = K + 1; c_k0[c] = K; c_seq[c] = SEQ++;
    c_alive[c] = 1; c_mark[c] = 0;
    link_slot(c);
    A[i] += sig[j] * START_W * zsent[j];
    if (GAIN) Zs[i] += zsent[j];
    aq_push(K + PS, c, c_seq[c]);
    if (TURN && is_cell_sender(j)) {
        c_trial[c] = 1; c_tsum[c] = 0.0; c_tn[c] = 0;
        if (trq_n == trq_cap) { trq_cap = trq_cap ? 2 * trq_cap : 4096; trq_t = realloc(trq_t, 8 * trq_cap); trq_seq = realloc(trq_seq, 8 * trq_cap); trq_c = realloc(trq_c, 4 * trq_cap); }
        trq_t[trq_n] = K + TTR; trq_seq[trq_n] = c_seq[c]; trq_c[trq_n] = c; trq_n++;
    }
    klast[i] = K; due_set(i, 0); tq_push(K + PMK, i);
    MADE++;
    if (nev_made == cap_made) { cap_made = cap_made ? 2 * cap_made : 4096; ev_made = realloc(ev_made, sizeof(int64_t) * 3 * cap_made); }
    ev_made[3 * nev_made] = K; ev_made[3 * nev_made + 1] = j; ev_made[3 * nev_made + 2] = i; nev_made++;
    if (NORM) normalise(i);
}
static int cmp_int(const void *a, const void *b) { int x1 = *(const int *)a, x2 = *(const int *)b; return (x1 > x2) - (x1 < x2); }
static int cmp_seq(const void *a, const void *b) { int64_t x1 = c_seq[*(const int *)a], x2 = c_seq[*(const int *)b]; return (x1 > x2) - (x1 < x2); }

/* strengths of the incoming connections of cell i at step K from the rule of that step (strength_byte 0) */
static void strengths_step(int i, double m, double xp) {
    ivec *ip = &in[i];
    CNT[3] += ip->n;
    if (rule[i] == 1) {
        double mx = m * xp;
        if (mx == 0.0) return;
        for (int q = 0; q < ip->n; q++) {
            int c = ip->a[q], j = c_snd[c];
            if (!learnable[j]) continue;
            double e = trace_of(c, K);
            strength_change(c, bound(c_w[c] + ((-ETA * sig[j]) * mx) * e), i);
        }
    } else if (rule[i] == 2) {
        double vti = vt[i];
        for (int q = 0; q < ip->n; q++) {
            int c = ip->a[q], j = c_snd[c];
            if (!learnable[j]) continue;
            double e = trace_of(c, K), dw;
            if (sig[j] > 0) dw = ((ETA * e) * xp) * (xp - vti); else dw = (ETA * e) * (xp - ASTAR);
            strength_change(c, bound(c_w[c] + dw), i);
        }
    }
}
/* strengths once a byte, from the sums of the byte (strength_byte 1), at the reading step */
static void strengths_byte(int i) {
    ivec *ip = &in[i];
    double sinh_all = sinh_[i] - (double)(S_STEPS - ncomp[i]) * ASTAR;   /* steps not computed had x(k - 1) = 0 */
    CNT[3] += ip->n;
    for (int q = 0; q < ip->n; q++) {
        int c = ip->a[q], j = c_snd[c];
        if (!learnable[j]) continue;
        double e = trace_of(c, K);
        if (e == 0.0) continue;
        double dw;
        if (rule[i] == 1) dw = (-ETA * sig[j]) * sm[i] * e;
        else dw = sig[j] > 0 ? ETA * e * sexc[i] : ETA * e * sinh_all;
        strength_change(c, bound(c_w[c] + dw), i);
    }
}
static int is_cell_sender(int j) { return j < NC; }
static double *ubuf; static int ubuf_cap;
static void use_step(int i, double xp) {
    ivec *ip = &in[i];
    int n = ip->n;
    if (ubuf_cap < n) { ubuf_cap = 2 * n + 16; ubuf = realloc(ubuf, sizeof(double) * ubuf_cap); }
    double mx = 0.0;
    for (int q = 0; q < n; q++) {
        int c = ip->a[q];
        if (c_snd[c] >= NC) { ubuf[q] = -1.0; continue; }          /* not a cell: its use is not measured */
        double e = trace_of(c, K);                       /* the trace held before the step */
        double u = DEC128 * c_use[c] + INC128 * e * xp;
        c_use[c] = u; ubuf[q] = u;
        if (c_trial[c]) { c_tsum[c] += e * xp; c_tn[c]++; }
        if (u > mx) mx = u;
    }
    double level = INC128 * mx;
    for (int q = 0; q < n; q++) {
        double u = ubuf[q];
        if (u < 0.0) continue;
        int c = ip->a[q];
        if (u < level) c_cnt[c]++; else c_cnt[c] = 0;
        if (c_cnt[c] >= PS && K - c_k0[c] >= PS && !c_mark[c] && !(USE_EXEMPT_FB && usefb[c_snd[c]])) { c_mark[c] = 1; iv_push(&gone, c); }
    }
    CNT[3] += 2 * n;
}
static int stationary_inh(int i, double xn) {         /* coincidence rule: no connection from an inhibitory cell would change */
    ivec *ip = &in[i];
    for (int q = 0; q < ip->n; q++) {
        int c = ip->a[q], j = c_snd[c];
        if (!learnable[j] || sig[j] > 0) continue;
        double w = c_w[c];
        if (!((w == 0.0 && xn <= ASTAR) || (w == 1.0 && xn >= ASTAR))) { if (trace_of(c, K + 1) != 0.0) return 0; }
    }
    return 1;
}

/* one step K; byte >= 0 sets the input lines at phi = 0 (the tissue); returns -2 if phi != 31, else the emission (-1: none) */
int evt_step(int byte) {
    int phi = (int)(K % S_STEPS);
    /* 1. input lines */
    if (byte >= 0 && phi == 0) {
        for (int q = 0; q < NIN; q++) { int j = LINE0 + q; double val = (q == byte) ? 1.0 : 0.0; if (z[j] != val) evt_set_line(j, val); }
    }
    /* 2. passing on, all before any cell is computed */
    for (int q = 0; q < nlinechg; q++) pass_on(linechg[q]);
    nlinechg = 0;
    if (chg.n > 1) {                                   /* senders pass on in the order of their numbers (a bitmap in place of a sort) */
        int lo = 1 << 30, hi = -1;
        for (int q = 0; q < chg.n; q++) { int j = chg.a[q]; sortbits[j >> 6] |= 1ULL << (j & 63); if (j < lo) lo = j; if (j > hi) hi = j; }
        int n = 0;
        for (int w = lo >> 6; w <= (hi >> 6); w++) { uint64_t b = sortbits[w]; sortbits[w] = 0; while (b) { int t = __builtin_ctzll(b); chg.a[n++] = (w << 6) + t; b &= b - 1; } }
    }
    for (int q = 0; q < chg.n; q++) { int j = chg.a[q]; ischg[j] = 0; pass_on(j); }
    chg.n = 0;
    /* 3 to 6 for the cells on the work list */
    wl2.n = 0;
    int last = (phi == S_STEPS - 1);
    for (int q = 0; q < wl.n; q++) {
        int i = wl.a[q];
        CNT[0]++;
        touch(i, K);
        double Ai = gain[i] * A[i], m = Ai - p[i];
        double vn = TAU_V_A * v[i] + TAU_V_A * Ai;
        double xn = vn - th[i]; xn = xn > 1.0 ? 1.0 : (xn < 0.0 ? 0.0 : xn);
        if (xn < FLOOR) xn = 0.0;
        double xp = x[i];
        Astep[i] = Ai; mstep[i] = m;
        /* 5. strengths */
        if (rule[i]) {
            if (!STRENGTH_BYTE) strengths_step(i, m, xp);
            else {
                if (rule[i] == 1) sm[i] += m * xp; else { sexc[i] += xp * (xp - vt[i]); sinh_[i] += xp - ASTAR; }
                ncomp[i]++;
                if (!inlearn[i] && (xp != 0.0 || (rule[i] == 2 && ninh[i] > 0))) { inlearn[i] = 1; iv_push(&learnl, i); }
            }
        }
        /* W: use of each connection from a cell, only at a step where x(k - 1) of the receiving cell is not zero */
        if (USE_W && LIFE && xp != 0.0) use_step(i, xp);
        /* 6. prediction, threshold, activity average, sliding threshold (traces are brought up to date when used) */
        p[i] = DEC32 * p[i] + INC32 * Ai; pst[i] = K + 1;
        double thn = theta_on[i] ? fmax(0.0, th[i] + ETA_T * (xn - ASTAR)) : th[i];
        abar[i] = DEC128 * abar[i] + INC128 * xn; ast[i] = K + 1;
        if (GAIN) { int nn = in[i].n; ib[i] = DEC128 * ib[i] + INC128 * (nn ? Zs[i] / nn : 0.0); ibst[i] = K + 1; }
        if (vt) { vt[i] = DEC4096 * vt[i] + INC4096 * xn * xn / ASTAR; vst[i] = K + 1; }
        int still = (vn == v[i]) && (xn == xp) && (thn == th[i]) && (dA[i] == 0.0);
        if (still && rule[i]) {
            if (xn != 0.0) still = 0;
            else if (rule[i] == 2 && !STRENGTH_BYTE && ninh[i] > 0 && !stationary_inh(i, xn)) still = 0;
        }
        v[i] = vn; th[i] = thn;
        if (xn != xp) { x[i] = xn; z[i] = xn; if (!ischg[i]) { ischg[i] = 1; iv_push(&chg2, i); } }
        if (!still) {
            if (xn == 0.0 && xp == 0.0 && thn == 0.0 && dA[i] == 0.0 && rule[i] != 2 && !inlearn[i]
                && (FLOOR > 0.0 ? (vn < FLOOR && Ai < FLOOR) : (vn <= 0.0 && Ai <= 0.0))) { slk[i] = K + 1; NSLEEP++; inwl[i] = 0; }
            else iv_push(&wl2, i);
        } else inwl[i] = 0;
    }
    /* strengths once a byte */
    if (STRENGTH_BYTE && last) {
        for (int i = 0; i < NC; i++) if (rule[i] == 2 && ninh[i] > 0 && !inlearn[i]) { inlearn[i] = 1; iv_push(&learnl, i); }
        for (int q = 0; q < learnl.n; q++) {
            int i = learnl.a[q];
            wake_to(i, K + 1, &wl2);
            if (!inwl[i]) touch(i, K + 1);
            strengths_byte(i);
            if (NORM) normalise(i);
            inlearn[i] = 0; sm[i] = sexc[i] = sinh_[i] = 0.0;
        }
        learnl.n = 0;
        for (int i = 0; i < NC; i++) ncomp[i] = 0;
    }
    if (GAIN && last) {
        for (int i = 0; i < NC; i++) {
            if (slk[i] >= 0) { if (wake(i, K + 1)) wl_add(&wl2, i); }
            bring_a(i, K + 1); bring_ib(i, K + 1);
            double tg = RHO * ib[i], ab = abar[i];
            double g = gain[i] * (1.0 + ETA_G * (tg - ab) / (tg + ab + 1e-30));
            g = g < GMIN ? GMIN : (g > GMAX ? GMAX : g);
            if (g != gain[i]) { gain[i] = g; if (!inwl[i]) { touch(i, K + 1); wl_add(&wl2, i); } }
        }
    }
    /* the strength changes of this step reach the arrivals of the next */
    ivec tmp = wl; wl = wl2; wl2 = tmp; wl2.n = 0;
    for (int q = 0; q < dal.n; q++) {
        int i = dal.a[q];
        if (dA[i] == 0.0) continue;
        if (!inwl[i]) { if (slk[i] >= 0) (void)wake(i, K + 1); touch(i, K + 1); wl_add(&wl, i); }
        A[i] += dA[i]; dA[i] = 0.0;
    }
    dal.n = 0;
    /* the 512-byte tissue: running average of the sending side for every connection */
    if (F11W) {
        for (int c = 0; c < NSLOT; c++) if (c_alive[c]) c_ub[c] = DEC128 * c_ub[c] + INC128 * zsent[c_snd[c]];
    }
    /* 7. giving up, then making */
    if (LIFE) {
        if (F11W) {
            gone.n = 0;
            for (int c = 0; c < NSLOT; c++)
                if (c_alive[c] && !exempt[c_snd[c]] && K - c_k0[c] >= PS && (c_w[c] < GIVE_UP || c_ub[c] < GIVE_UP)) iv_push(&gone, c);
            CNT[4] += NSLOT;
        } else {
            while (aq_h < aq_n && aq_t[aq_h] <= K) {
                int c = aq_c[aq_h];
                if (c_alive[c] && c_seq[c] == aq_seq[aq_h] && !exempt[c_snd[c]] && !c_mark[c]
                    && (c_w[c] < GIVE_UP || (USE_W && is_cell_sender(c_snd[c]) && c_cnt[c] >= PS && !(USE_EXEMPT_FB && usefb[c_snd[c]])))) {
                    c_mark[c] = 1; iv_push(&gone, c);
                }
                aq_h++; CNT[4]++;
            }
        }
        /* the end of a trial (plan 2) */
        while (trq_h < trq_n && trq_t[trq_h] <= K) {
            int c = trq_c[trq_h];
            if (c_alive[c] && c_seq[c] == trq_seq[trq_h] && c_trial[c]) {
                c_trial[c] = 0;
                if (!c_mark[c]) {
                    int i = c_rcv[c]; double sum = 0.0; int n = 0;
                    for (int q = 0; q < in[i].n; q++) { int c2 = in[i].a[q]; if (c2 != c && !c_trial[c2] && is_cell_sender(c_snd[c2])) { sum += c_use[c2]; n++; } }
                    double mean = n ? sum / n : 0.0, u = c_tn[c] ? c_tsum[c] / c_tn[c] : 0.0;
                    CNT[4] += in[i].n;
                    if (u == 0.0 || u < THTR * mean) { c_mark[c] = 1; iv_push(&gone, c); TRIAL_GONE++; } else TRIAL_KEPT++;
                }
            }
            trq_h++;
        }
        if (gone.n) {
            qsort(gone.a, gone.n, sizeof(int), cmp_seq);
            for (int q = 0; q < gone.n; q++) remove_conn(gone.a[q]);
            gone.n = 0;
        }
        while (tq_h < tq_n && tq_t[tq_h] <= K) {
            int i = tq_c[tq_h];
            if (K - klast[i] >= PMK && !full[i]) due_set(i, 1);
            tq_h++;
        }
        for (int w = 0; w < NW; w++) for (uint64_t b = duebits[w]; b; b &= b - 1) {
            int i = (w << 6) + __builtin_ctzll(b);
            if (!due[i]) continue;
            if (in[i].n >= CAPIN) { full[i] = 1; due_set(i, 0); continue; }
            wake_to(i, K + 1, &wl2);
            bring_a(i, K + 1);
            if (in[i].n >= CAPIN) { full[i] = 1; due_set(i, 0); continue; }   /* at the cap: as a full neighbourhood */
            if (TURN || (th[i] == 0.0 && abar[i] < ASTAR)) make_for(i);
        }
        compact_queues();
    }
    /* cells put on the work list by giving up and making are in wl2 */
    for (int q = 0; q < wl2.n; q++) iv_push(&wl, wl2.a[q]);
    wl2.n = 0;
    tmp = chg; chg = chg2; chg2 = tmp; chg2.n = 0;
    CNT[5]++;
    K++;
    if (last && NDRV) {
        int b = 0; double best = x[driver[0]];
        for (int q = 1; q < NDRV; q++) if (x[driver[q]] > best) { best = x[driver[q]]; b = q; }
        return best > EMIT ? b : -1;
    }
    return -2;
}
/* run many bytes (no recording); emissions into em */
void evt_run_bytes(const uint8_t *bytes, int n, int16_t *em) {
    for (int t = 0; t < n; t++) { int r = -2; for (int s = 0; s < S_STEPS; s++) r = evt_step(bytes[t]); em[t] = (int16_t)r; }
}

/* ---------------------------------------------------------------------------------------------- reading the state */
int64_t evt_k(void) { return K; }
int evt_nlive(void) { return NLIVE; }
int64_t evt_made(void) { return MADE; }
int64_t evt_given_up(void) { return GIVEN_UP; }
double *evt_ptr_x(void) { return x; }
double *evt_ptr_th(void) { return th; }
double *evt_ptr_v(void) { return v; }
double *evt_ptr_A(void) { return Astep; }
double *evt_ptr_m(void) { return mstep; }
void evt_counters(double *o) { memcpy(o, CNT, sizeof(CNT)); }
void evt_cells_active(int *o) { int n = 0; for (int i = 0; i < NC; i++) n += x[i] != 0.0; *o = n; }
/* bring every lazily held quantity up to the start of step K, for reading */
void evt_sync(void) {
    evt_wake_all();
    for (int i = 0; i < NC; i++) touch(i, K);
    for (int c = 0; c < NSLOT; c++) if (c_alive[c]) (void)trace_of(c, K);
}
static int *order; static int norder;
static double peek_trace(int c, int64_t upto) {          /* the trace of connection c at the start of step `upto`, not stored */
    if (TRACE_SENDER) { int j = c_snd[c]; return relax(E[j], zsent[j], DEC32, INC32, upto - Est[j]); }
    return relax(c_e[c], zsent[c_snd[c]], DEC32, INC32, upto - c_est[c]);
}
static int cmp_seq2(const void *a, const void *b) { return cmp_seq(a, b); }
/* the live connections in the order of making: sender, receiver, strength, trace, step of making, ub */
void evt_live(int *snd, int *rcv, double *w, double *e, int64_t *k0, double *ub) {
    norder = 0; order = realloc(order, 4 * (NLIVE + 1));
    for (int c = 0; c < NSLOT; c++) if (c_alive[c]) order[norder++] = c;
    qsort(order, norder, sizeof(int), cmp_seq2);
    for (int q = 0; q < norder; q++) {
        int c = order[q];
        snd[q] = c_snd[c]; rcv[q] = c_rcv[c]; w[q] = c_w[c]; k0[q] = c_k0[c]; ub[q] = c_ub[c];
        e[q] = peek_trace(c, K);
    }
}
void evt_cells(double *pv, double *pp, double *pth, double *pab, double *pvt, int64_t *pkl) {
    /* reading does not change the state: lazily held quantities are computed up to step K without being stored */
    evt_wake_all();
    for (int i = 0; i < NC; i++) {
        pp[i] = relax(p[i], A[i], DEC32, INC32, K - pst[i]);
        pab[i] = relax(abar[i], x[i], DEC128, INC128, K - ast[i]);
        if (vt && pvt) pvt[i] = relax_vt(vt[i], x[i], K - vst[i]);
    }
    memcpy(pv, v, 8 * NC); memcpy(pth, th, 8 * NC); memcpy(pkl, klast, 8 * NC);
}
int evt_events_made(int64_t *o, int cap) { size_t n = nev_made < (size_t)cap ? nev_made : (size_t)cap; memcpy(o, ev_made, sizeof(int64_t) * 3 * n); return (int)nev_made; }
int evt_events_gone(int64_t *o, int cap) { size_t n = nev_gone < (size_t)cap ? nev_gone : (size_t)cap; memcpy(o, ev_gone, sizeof(int64_t) * 4 * n); return (int)nev_gone; }
void evt_clear_events(void) { nev_made = nev_gone = 0; }
/* the use and counter of withdrawal by use (W) of the live connections, in the order of evt_live (call evt_live first) */
void evt_live_use(double *use, int *cnt) {
    for (int q = 0; q < norder; q++) { int c = order[q]; use[q] = c_use[c]; cnt[q] = c_cnt[c]; }
}

int64_t evt_norm_calls(void) { return NORM_CALLS; }
