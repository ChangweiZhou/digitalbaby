#include <stdint.h>
#include <math.h>
/* Third kernel: the same algorithm as ev2.c with different data structures.
   (1) Each connection is stored once, in the order of its SENDING cell, as (target, strength): sending on a change
       walks one contiguous block. The list by receiving cell holds, for each incoming connection, its source and
       the position of that record, and is walked only when the cell learns.
   (2) The quantities of a cell that are used together at a step sit in one record of one cache line. */
typedef struct { double A, v, p, th, x, acc; float sig; int32_t pad; double pad2; } Cell;   /* 64 bytes */
typedef struct {
    int64_t N, n, S;
    const int64_t *out_ptr; const uint16_t *out_tgt; float *out_w;      /* by sending cell */
    const int64_t *in_ptr;  const uint16_t *in_src; const int32_t *in_q; /* by receiving cell: source, position in out_* */
    Cell *c; double *xs, *e;                                             /* output last sent and trace: read from other cells */
    uint8_t *inW, *inD; int32_t *W, *W2, *D; int64_t nW, nD;
    double floor_x, delta; int64_t visits_push, visits_learn, cell_updates;
} T3;
static void one_step(T3 *t) {
    int64_t n2 = 0; Cell *C = t->c; const double delta = t->delta;
    for (int64_t i = 0; i < t->nW; i++) t->inW[t->W[i]] = 0;
    for (int64_t i = 0; i < t->nW; i++) {
        int32_t r = t->W[i]; Cell *k = &C[r];
        double m = k->A - k->p;
        if (k->x != 0.0) { k->acc += m * k->x; if (!t->inD[r]) { t->inD[r] = 1; t->D[t->nD++] = r; } }
        k->p += m / 32.0;
        double v = 0.5 * k->v + 0.5 * k->A; k->v = v;
        double x = v - k->th; if (x < t->floor_x) x = 0.0; else if (x > 1.0) x = 1.0;
        double th = k->th + (x - 0.125) / 128.0; k->th = th < 0.0 ? 0.0 : th;
        k->x = x; t->cell_updates++;
        double dx = x - t->xs[r];
        if (dx > delta || dx < -delta) {
            double g = k->sig * dx; t->xs[r] = x;
            const int64_t q1 = t->out_ptr[r + 1];
            for (int64_t q = t->out_ptr[r]; q < q1; q++) {
                uint16_t tg = t->out_tgt[q];
                C[tg].A += g * t->out_w[q];
                if (!t->inW[tg]) { t->inW[tg] = 1; t->W2[n2++] = tg; }
            }
            t->visits_push += q1 - t->out_ptr[r];
        }
        double dv = v - k->A;
        if ((dv > delta || dv < -delta || (x != 0.0 && fabs(m) > delta)) && !t->inW[r]) { t->inW[r] = 1; t->W2[n2++] = r; }
    }
    int32_t *tmp = t->W; t->W = t->W2; t->W2 = tmp; t->nW = n2;
}
static void learn(T3 *t) {
    Cell *C = t->c;
    for (int64_t i = 0; i < t->nD; i++) {
        int32_t r = t->D[i]; double f = 0.125 * C[r].acc, dA = 0.0;
        for (int64_t c = t->in_ptr[r]; c < t->in_ptr[r + 1]; c++) {
            uint16_t s = t->in_src[c]; double es = t->e[s];
            if (es == 0.0) continue;
            int32_t q = t->in_q[c]; double sg = C[s].sig;
            double w0 = t->out_w[q], nw = w0 - f * sg * es;
            nw = nw < 0.0 ? 0.0 : (nw > 1.0 ? 1.0 : nw);
            t->out_w[q] = (float)nw; dA += (nw - w0) * sg * t->xs[s];
        }
        t->visits_learn += t->in_ptr[r + 1] - t->in_ptr[r];
        C[r].A += dA; C[r].acc = 0.0; t->inD[r] = 0;
        if (dA != 0.0 && !t->inW[r]) { t->inW[r] = 1; t->W[t->nW++] = r; }
    }
    t->nD = 0;
}
int64_t ev3_run(T3 *t, int64_t nbytes, const int32_t *drive, int64_t per_byte, double *active_sum) {
    double asum = 0.0; Cell *C = t->c;
    double q = pow(1.0 - 1.0 / 32.0, (double)t->S);
    for (int64_t b = 0; b < nbytes; b++) {
        for (int64_t j = 0; j < per_byte; j++) {
            int32_t r = drive[b * per_byte + j]; C[r].A += 0.125;
            if (!t->inW[r]) { t->inW[r] = 1; t->W[t->nW++] = r; }
            if (b > 0) { int32_t o = drive[(b - 1) * per_byte + j]; C[o].A -= 0.125; if (!t->inW[o]) { t->inW[o] = 1; t->W[t->nW++] = o; } }
        }
        for (int64_t ph = 0; ph < t->S; ph++) { if (t->nW == 0) break; one_step(t); }
        for (int64_t i = 0; i < t->N; i++) { double xs = t->xs[i]; if (xs != 0.0) asum += 1.0;
            double e = t->e[i] * q + (1.0 - q) * xs; t->e[i] = e < 1e-6 ? 0.0 : e; }
        learn(t);
    }
    *active_sum = asum; return 0;
}
