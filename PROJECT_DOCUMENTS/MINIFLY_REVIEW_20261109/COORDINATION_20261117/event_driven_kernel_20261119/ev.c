#include <stdint.h>
#include <math.h>
/* Event-driven step of the tissue's per-connection and per-cell work.
   Cells and connections hold the step at which they were last brought up to date; a cell is
   visited only if something arrived at it or it was active at the previous step.
   Connections are stored once, sorted by receiver (in_ptr); out_ptr/out_conn list them by sender. */
typedef struct {
    int64_t N, n;
    const int32_t *snd, *rcv; const int64_t *in_ptr, *out_ptr; const int32_t *out_conn;
    double *w, *e; int32_t *te;            /* strength, trace, step the trace was last brought up to date */
    const double *sig;
    double *v, *p, *th, *abar, *x; int32_t *tc;   /* cell state and the step it was last brought up to date */
    double *A; uint8_t *touched; int32_t *tl;     /* arrivals of this step, and the cells they reached */
    int32_t *act, *act2; int64_t na;              /* cells with output above the floor at the previous step */
    const double *dec_v, *dec_p;                  /* (1-1/tau)^d for d = 0..4095 */
    double floor_x;
} T;
static inline double decay(const double *tab, int64_t d) { return d < 4096 ? tab[d] : 0.0; }
static int64_t ev_step(T *t, int32_t k, int64_t *visited) {
    int64_t nt = 0, vis = 0;
    /* 1. arrivals from the cells that were active; their outgoing traces are brought up to date */
    for (int64_t a = 0; a < t->na; a++) {
        int32_t s = t->act[a]; double xs = t->x[s], g = t->sig[s] * xs;
        for (int64_t q = t->out_ptr[s]; q < t->out_ptr[s + 1]; q++) {
            int32_t c = t->out_conn[q], r = t->rcv[c];
            t->e[c] = t->e[c] * decay(t->dec_p, k - t->te[c]) + xs / 32.0; t->te[c] = k;
            t->A[r] += g * t->w[c];
            if (!t->touched[r]) { t->touched[r] = 1; t->tl[nt++] = r; }
            vis++;
        }
    }
    /* 2. strength changes at the cells that were active at the previous step (formula 7 with its bound) */
    for (int64_t a = 0; a < t->na; a++) {
        int32_t r = t->act[a];
        double pr = t->p[r] * decay(t->dec_p, k - t->tc[r]);
        double m = t->A[r] - pr, f = 0.125 * m * t->x[r];
        for (int64_t c = t->in_ptr[r]; c < t->in_ptr[r + 1]; c++) {
            double ec = t->e[c] * decay(t->dec_p, k - t->te[c]);
            double nw = t->w[c] - f * t->sig[t->snd[c]] * ec;
            t->w[c] = nw < 0.0 ? 0.0 : (nw > 1.0 ? 1.0 : nw);
            vis++;
        }
        if (!t->touched[r]) { t->touched[r] = 1; t->tl[nt++] = r; }
    }
    /* 3. the cells reached: state brought up to date over the steps in which nothing arrived, then this step */
    int64_t na2 = 0;
    for (int64_t i = 0; i < nt; i++) {
        int32_t r = t->tl[i]; int64_t d = k - t->tc[r];
        double v = t->v[r] * decay(t->dec_v, d > 0 ? d - 1 : 0);
        double p = t->p[r] * decay(t->dec_p, d > 0 ? d - 1 : 0);
        double th = t->th[r] - (d > 1 ? (d - 1) * (1.0 / 1024.0) : 0.0); if (th < 0.0) th = 0.0;
        v = 0.5 * v + 0.5 * t->A[r];
        p = p + (t->A[r] - p) / 32.0;
        double x = v - th; if (x < t->floor_x) x = 0.0; if (x > 1.0) x = 1.0;
        th += (x - 0.125) / 128.0; if (th < 0.0) th = 0.0;
        t->v[r] = v; t->p[r] = p; t->th[r] = th; t->x[r] = x; t->tc[r] = k;
        t->A[r] = 0.0; t->touched[r] = 0;
        if (x > 0.0) t->act2[na2++] = r;
    }
    for (int64_t a = 0; a < t->na; a++) { int32_t s = t->act[a]; if (t->tc[s] != k) t->x[s] = 0.0; }
    int32_t *tmp = t->act; t->act = t->act2; t->act2 = tmp; t->na = na2;
    *visited += vis;
    return na2;
}
/* a whole run: at each byte's first step `drive` cells of the input end are set active, then 32 steps */
int64_t ev_run(T *t, int64_t nbytes, const int32_t *drive, int64_t per_byte, int64_t *visited, int64_t *active_sum) {
    int32_t k = 1; int64_t tot = 0;
    for (int64_t b = 0; b < nbytes; b++) {
        for (int ph = 0; ph < 32; ph++, k++) {
            for (int64_t j = 0; j < per_byte; j++) {           /* the byte's input lines drive their cells at every step */
                int32_t r = drive[b * per_byte + j];
                if (t->x[r] == 0.0) { t->act[t->na++] = r; }
                t->x[r] = 0.25; 
            }
            tot += ev_step(t, k, visited);
        }
    }
    *active_sum = tot; return k;
}
