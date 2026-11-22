/* The context part of net5's step (prediction, choice of the cell, learning, making of connections), in C.
   Same definition as net5.py; differences: ties among equally weak slots go to the lowest slot number (net5: numpy's sort),
   and the random sample of new sources uses its own generator (xorshift). Build: gcc -O3 -march=native -shared -fPIC. */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

static uint64_t rs = 88172645463325252ULL;
void seed_rng(uint64_t s) { rs = s * 2654435761ULL + 1; if (!rs) rs = 1; }
static inline uint64_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

typedef struct { double eta, eta_minus, w_on, w_new, th_on, th_match, s_th, lam; int n_grow, burst, ctx_mode, learn, recall; } P;

/* arrays: gsrc, rsrc [L*C*M*S] int64; rw [L*C*M*S] double; ctx [L*2CM] uint8; cols [L*C] uint8;
   use, rel [L*C*M] double; out: cells [L*C*M] uint8, surprise [L] uint8; stats[0] = predicted share of winning columns */
void step(int L, int C, int M, int S, int64_t *gsrc, int64_t *rsrc, double *rw, const uint8_t *ctx, const uint8_t *cols,
          double *use, double *rel, uint8_t *cells, uint8_t *surprise, double *stats, const P *p)
{
    const int CM = C * M, NC = 2 * CM;
    int *n_on = malloc(sizeof(int) * CM), *n_act = malloc(sizeof(int) * CM), *sH = malloc(sizeof(int) * CM), *aH = malloc(sizeof(int) * CM);
    uint8_t *pred = malloc(CM), *match = malloc(CM), *act = malloc((size_t)CM * S), *f = malloc(CM);
    int *prev = malloc(sizeof(int) * NC), *miss = malloc(sizeof(int) * NC), *tmp = malloc(sizeof(int) * NC);
    int *order = malloc(sizeof(int) * S);
    long pc_win = 0, win_tot = 0;
    for (int d = 0; d < L; d++) {
        const uint8_t *cx = ctx + (size_t)d * NC, *cl = cols + (size_t)d * C;
        int np_ = 0, nh = 0;
        for (int i = 0; i < CM; i++) np_ += cx[i] != 0;
        for (int i = CM; i < NC; i++) nh += cx[i] != 0;
        double nap = p->n_grow && np_ > p->n_grow ? p->n_grow : np_, nah = p->n_grow && nh > p->n_grow ? p->n_grow : nh;
        size_t base = (size_t)d * CM;
        for (int i = 0; i < CM; i++) {
            size_t o = (base + i) * S; int a1 = 0, a2 = 0, a3 = 0, a4 = 0;
            for (int s = 0; s < S; s++) {
                double w = rw[o + s]; uint8_t a = (w > 0) && ctx[gsrc[o + s]]; act[(size_t)i * S + s] = a;
                if (a) { int st = w >= p->w_on, h = rsrc[o + s] >= CM; a2++; a1 += st; a4 += h; a3 += st && h; }
            }
            n_on[i] = a1; n_act[i] = a2; sH[i] = a3; aH[i] = a4;
            int ok = np_ > 0;
            pred[i] = ok && a1 > 0 && (a1 - a3) >= p->th_on * nap && a3 >= p->th_on * nah;
            match[i] = ok && a2 > 0 && (a2 - a4) >= p->th_match * nap && a4 >= p->th_match * nah;
        }
        int npc = 0, hit = 0, npw = 0; double rsum = 0;
        for (int c = 0; c < C; c++) {
            int pc = 0;
            for (int m = 0; m < M; m++) if (pred[c * M + m]) { pc = 1; if (cl[c]) { npw++; rsum += rel[base + c * M + m]; } }
            npc += pc; hit += pc && cl[c];
            if (cl[c]) { win_tot++; pc_win += pc; }
        }
        double relw = npw ? rsum / npw : 1.0;
        int nwin = 0; for (int cc = 0; cc < C; cc++) nwin += cl[cc] != 0;
        int sur = (npc > 0 && hit < p->s_th * npc) || relw < p->s_th || (p->recall && hit < p->s_th * nwin);   /* recall: most winners unpredicted */
        surprise[d] = sur;
        memset(f, 0, CM);
        for (int c = 0; c < C; c++) {
            int best = 0; double bs = -1e300; int pc = 0;
            for (int m = 0; m < M; m++) {
                int i = c * M + m; pc |= pred[i];
                double sc = pred[i] ? 1e6 + n_on[i] : (match[i] ? 1e3 + n_act[i] : -use[base + i]);
                if (sc > bs) { bs = sc; best = m; }
            }
            int pick = p->ctx_mode ? best : (sur ? 0 : best);
            if (p->burst && !pc) pick = 0;
            for (int m = 0; m < M; m++) cells[base + c * M + m] = cl[c] && m == pick;
            if (p->learn) {
                for (int m = 0; m < M; m++) { int i = c * M + m; if (pred[i]) rel[base + i] += p->lam * ((double)(m == best && cl[c]) - rel[base + i]); }
                if (cl[c] && best != 0) f[c * M + best] = 1;
            }
        }
        if (!p->learn) continue;
        for (int i = 0; i < CM; i++) {
            size_t o = (base + i) * S; const uint8_t *a = act + (size_t)i * S;
            if (f[i]) { for (int s = 0; s < S; s++) if (a[s]) rw[o + s] += p->eta * (1 - rw[o + s]); }
            else if (pred[i]) { for (int s = 0; s < S; s++) if (a[s]) rw[o + s] -= p->eta_minus * rw[o + s]; }
        }
        /* making of connections, for each cell that learns */
        int P_ = 0; for (int j = 0; j < NC; j++) if (cx[j]) prev[P_++] = j;
        if (!P_) { for (int i = 0; i < CM; i++) use[base + i] += f[i]; continue; }
        for (int i = 0; i < CM; i++) {
            if (!f[i]) continue;
            size_t o = (base + i) * S; const uint8_t *a = act + (size_t)i * S;
            int nm = 0;
            for (int j = 0; j < P_; j++) {                                   /* sources it has no slot for, in order */
                int have = 0;
                for (int s = 0; s < S; s++) if (rw[o + s] > 0 && rsrc[o + s] == prev[j]) { have = 1; break; }
                if (!have) miss[nm++] = prev[j];
            }
            if (p->n_grow) {                                                 /* per part: at most n_grow - (active slots of the part), a random sample */
                int ap = 0, ah = 0;
                for (int s = 0; s < S; s++) if (a[s]) { if (rsrc[o + s] >= CM) ah++; else ap++; }
                int keep[2] = { p->n_grow - ap > 0 ? p->n_grow - ap : 0, p->n_grow - ah > 0 ? p->n_grow - ah : 0 };
                int out = 0; memcpy(tmp, miss, sizeof(int) * nm);
                for (int part = 0; part < 2; part++) {
                    int st = out, n = 0;
                    for (int j = 0; j < nm; j++) if ((tmp[j] >= CM) == part) miss[out + n++] = tmp[j];
                    if (n > keep[part]) {                                    /* choose keep[part] of the n, keep their order */
                        for (int t = 0; t < keep[part]; t++) { int r = t + (int)(xr() % (uint64_t)(n - t)); int tmp = miss[st + t]; miss[st + t] = miss[st + r]; miss[st + r] = tmp; }
                        n = keep[part];
                        for (int x = 1; x < n; x++) { int v = miss[st + x], y = x - 1; while (y >= 0 && miss[st + y] > v) { miss[st + y + 1] = miss[st + y]; y--; } miss[st + y + 1] = v; }
                    }
                    out = st + n;
                }
                /* parts were laid out prev first then hold, each in order: the same order as net5 (prev indices < hold indices) */
                nm = out;
            }
            int no = 0;                                                      /* weakest inactive slots first (ties: lower slot) */
            for (int s = 0; s < S; s++) if (!a[s]) order[no++] = s;
            for (int x = 1; x < no; x++) { int v = order[x], y = x - 1; while (y >= 0 && rw[o + order[y]] > rw[o + v]) { order[y + 1] = order[y]; y--; } order[y + 1] = v; }
            for (int j = 0; j < nm && j < no; j++) {
                int s = order[j]; rsrc[o + s] = miss[j]; rw[o + s] = p->w_new; gsrc[o + s] = miss[j] + (int64_t)d * NC;
            }
        }
        for (int i = 0; i < CM; i++) use[base + i] += f[i];
    }
    stats[0] = win_tot ? (double)pc_win / win_tot : 0.0;
    free(n_on); free(n_act); free(sH); free(aH); free(pred); free(match); free(act); free(f); free(prev); free(miss); free(tmp); free(order);
}
