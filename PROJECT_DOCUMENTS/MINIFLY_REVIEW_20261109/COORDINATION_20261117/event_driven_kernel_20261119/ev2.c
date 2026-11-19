#include <stdint.h>
#include <math.h>
/* Second event-driven kernel. Three changes of algorithm against ev.c:
   (1) a cell's arrival is kept from step to step and only CHANGES of a sender's output are sent on
       (an output that has not moved by more than `delta` since it was last sent costs nothing);
   (2) the trace of an input is held once for the sending cell, not once for each of its connections;
   (3) strengths are changed once a byte from a sum kept by the receiving cell, not at every step.
   Cells are worked on only while their arrival or state is still moving. */
typedef struct {
    int64_t N, n, S;
    const uint16_t *snd, *rcv;                 /* connections sorted by receiver */
    const int64_t *in_ptr, *out_ptr; const int32_t *out_conn;
    float *w; const float *sig;
    double *A, *v, *p, *th, *x, *xs, *acc, *e; /* arrival, state, prediction, threshold, output, output last sent, sum for learning, trace */
    uint8_t *inW, *inD; int32_t *W, *W2, *D;   /* cells still moving; cells with something to learn this byte */
    int64_t nW, nD;
    double floor_x, delta;
    int64_t visits_push, visits_learn, cell_updates;
} T2;
static inline void wake(T2 *t, int32_t r, int32_t *list, int64_t *n) { if (!t->inW[r]) { t->inW[r] = 1; list[(*n)++] = r; } }
static void one_step(T2 *t) {
    int64_t n2 = 0;
    for (int64_t i = 0; i < t->nW; i++) t->inW[t->W[i]] = 0;
    for (int64_t i = 0; i < t->nW; i++) {
        int32_t r = t->W[i];
        double m = t->A[r] - t->p[r];
        if (t->x[r] != 0.0) { t->acc[r] += m * t->x[r]; if (!t->inD[r]) { t->inD[r] = 1; t->D[t->nD++] = r; } }
        t->p[r] += m / 32.0;
        double v = 0.5 * t->v[r] + 0.5 * t->A[r]; t->v[r] = v;
        double x = v - t->th[r]; if (x < t->floor_x) x = 0.0; else if (x > 1.0) x = 1.0;
        double th = t->th[r] + (x - 0.125) / 128.0; t->th[r] = th < 0.0 ? 0.0 : th;
        t->x[r] = x; t->cell_updates++;
        double dx = x - t->xs[r];
        if (dx > t->delta || dx < -t->delta) {
            double g = t->sig[r] * dx; t->xs[r] = x;
            for (int64_t q = t->out_ptr[r]; q < t->out_ptr[r + 1]; q++) {
                int32_t c = t->out_conn[q]; uint16_t tgt = t->rcv[c];
                t->A[tgt] += g * t->w[c];
                wake(t, tgt, t->W2, &n2);
            }
            t->visits_push += t->out_ptr[r + 1] - t->out_ptr[r];
        }
        double dv = v - t->A[r];
        if (dv > t->delta || dv < -t->delta || x != 0.0 && fabs(m) > t->delta) wake(t, r, t->W2, &n2);
    }
    int32_t *tmp = t->W; t->W = t->W2; t->W2 = tmp; t->nW = n2;
}
static void learn(T2 *t) {                      /* once a byte: strengths of the cells that have a sum; arrivals corrected */
    for (int64_t i = 0; i < t->nD; i++) {
        int32_t r = t->D[i]; double f = 0.125 * t->acc[r], dA = 0.0;
        for (int64_t c = t->in_ptr[r]; c < t->in_ptr[r + 1]; c++) {
            uint16_t s = t->snd[c]; double es = t->e[s];
            if (es == 0.0) continue;
            double w0 = t->w[c], nw = w0 - f * t->sig[s] * es;
            nw = nw < 0.0 ? 0.0 : (nw > 1.0 ? 1.0 : nw);
            t->w[c] = (float)nw; dA += (nw - w0) * t->sig[s] * t->xs[s];
        }
        t->visits_learn += t->in_ptr[r + 1] - t->in_ptr[r];
        t->A[r] += dA; t->acc[r] = 0.0; t->inD[r] = 0;
        if (dA != 0.0 && !t->inW[r]) { t->inW[r] = 1; t->W[t->nW++] = r; }
    }
    t->nD = 0;
}
int64_t ev2_run(T2 *t, int64_t nbytes, const int32_t *drive, int64_t per_byte, double *active_sum) {
    double asum = 0.0;
    for (int64_t b = 0; b < nbytes; b++) {
        for (int64_t j = 0; j < per_byte; j++) {            /* the byte's lines come on, the last byte's go off */
            int32_t r = drive[b * per_byte + j]; t->A[r] += 0.125;
            if (!t->inW[r]) { t->inW[r] = 1; t->W[t->nW++] = r; }
            if (b > 0) { int32_t o = drive[(b - 1) * per_byte + j]; t->A[o] -= 0.125; if (!t->inW[o]) { t->inW[o] = 1; t->W[t->nW++] = o; } }
        }
        for (int64_t ph = 0; ph < t->S; ph++) { one_step(t); }
        for (int64_t i = 0; i < t->N; i++) if (t->xs[i] != 0.0) asum += 1.0;   /* counted for the report only */
        /* traces of the sending cells, once a byte (the cells that sent something; the others decay) */
        double q = pow(1.0 - 1.0 / 32.0, (double)t->S);
        for (int64_t i = 0; i < t->N; i++) { double e = t->e[i] * q + (1.0 - q) * t->xs[i]; t->e[i] = e < 1e-6 ? 0.0 : e; }
        learn(t);
    }
    *active_sum = asum; return 0;
}
