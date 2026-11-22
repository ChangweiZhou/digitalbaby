/* net7.c's step by events (see EVENTK_PLAN.md): the same definition, the same results, at a cost that follows the active
   sources instead of the number of slots. For every global source g (gsrc value) a doubly linked list of the slots that read it
   and have strength > 0 (head[g], nxt[slot], prv[slot]; prv = -2: in no list). Each byte only the lists of the active sources
   are walked. The random sample of new sources uses the same generator as net7.c, drawn in the same order.
   Build: gcc -O3 -march=native -shared -fPIC. */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

static uint64_t rs = 88172645463325252ULL;
void seed_rng(uint64_t s) { rs = s * 2654435761ULL + 1; if (!rs) rs = 1; }
static inline uint64_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

typedef struct { double eta, eta_minus, w_on, w_new, th_on, th_match, s_th, lam; int n_grow, burst, ctx_mode, learn, recall, nodef; } P;

static inline void ix_remove(int32_t *head, int32_t *nxt, int32_t *prv, const int64_t *gsrc, int32_t sl)
{
    int32_t p = prv[sl], n = nxt[sl];
    if (p == -2) return;
    if (p == -1) head[gsrc[sl]] = n; else nxt[p] = n;
    if (n >= 0) prv[n] = p;
    prv[sl] = -2; nxt[sl] = -1;
}
static inline void ix_insert(int32_t *head, int32_t *nxt, int32_t *prv, int64_t g, int32_t sl)
{
    int32_t h = head[g]; nxt[sl] = h; prv[sl] = -1; if (h >= 0) prv[h] = sl; head[g] = sl;
}

/* build the index from (rw, gsrc): every slot with rw > 0 in the list of its gsrc */
void build_index(int L, int C, int M, int S, const int64_t *gsrc, const double *rw, int32_t *head, int32_t *nxt, int32_t *prv)
{
    const size_t NS = (size_t)L * C * M * S, NG = (size_t)L * 2 * C * M;
    for (size_t g = 0; g < NG; g++) head[g] = -1;
    for (size_t sl = 0; sl < NS; sl++) { nxt[sl] = -1; prv[sl] = -2; }
    for (size_t sl = NS; sl-- > 0;) if (rw[sl] > 0) ix_insert(head, nxt, prv, gsrc[sl], (int32_t)sl);
}

/* the same arguments as net7.c's step, plus the index */
void step(int L, int C, int M, int S, int64_t *gsrc, int64_t *rsrc, double *rw, const uint8_t *ctx, const uint8_t *cols,
          double *use, double *rel, uint8_t *cells, uint8_t *surprise, double *stats, const P *p,
          int32_t *head, int32_t *nxt, int32_t *prv)
{
    const int CM = C * M, NC = 2 * CM;
    int *n_on = calloc(CM, sizeof(int)), *n_act = calloc(CM, sizeof(int)), *sH = calloc(CM, sizeof(int)), *aH = calloc(CM, sizeof(int));
    uint8_t *pred = calloc(CM, 1), *match = calloc(CM, 1), *f = calloc(CM, 1), *af = malloc((size_t)C * S), *colp = calloc(C, 1);
    int *touched = malloc(sizeof(int) * CM), *prev = malloc(sizeof(int) * NC), *miss = malloc(sizeof(int) * NC), *tmp = malloc(sizeof(int) * NC);
    int *stamp = calloc(NC, sizeof(int)), *order = malloc(sizeof(int) * S), *wins = malloc(sizeof(int) * C), *fcell = malloc(sizeof(int) * C);
    size_t cap = 1024, nact; int32_t *actl = malloc(sizeof(int32_t) * cap);
    int stamp_id = 0;
    long pc_win = 0, win_tot = 0;
    for (int d = 0; d < L; d++) {
        const uint8_t *cx = ctx + (size_t)d * NC, *cl = cols + (size_t)d * C;
        size_t base = (size_t)d * CM, sbase = base * S;
        int P_ = 0, np_ = 0;
        for (int j = 0; j < NC; j++) if (cx[j]) prev[P_++] = j;               /* the active sources, in order */
        while (np_ < P_ && prev[np_] < CM) np_++;
        int nh = P_ - np_;
        double nap = p->n_grow && np_ > p->n_grow ? p->n_grow : np_, nah = p->n_grow && nh > p->n_grow ? p->n_grow : nh;
        /* counts, by the lists of the active sources */
        int nt = 0; nact = 0;
        for (int j = 0; j < P_; j++) {
            int h = prev[j] >= CM;
            for (int32_t sl = head[(size_t)d * NC + prev[j]]; sl >= 0; sl = nxt[sl]) {
                double w = rw[sl];
                if (!(w > 0)) continue;
                int i = (int)(((size_t)sl - sbase) / S);
                if (!n_act[i]) touched[nt++] = i;
                int st = w >= p->w_on; n_act[i]++; n_on[i] += st; aH[i] += h; sH[i] += st && h;
                if (nact == cap) { cap *= 2; actl = realloc(actl, sizeof(int32_t) * cap); }
                actl[nact++] = sl;
            }
        }
        int ok = np_ > 0;
        for (int t = 0; t < nt; t++) {
            int i = touched[t], a1 = n_on[i], a2 = n_act[i], a3 = sH[i], a4 = aH[i];
            pred[i] = ok && a1 > 0 && (a1 - a3) >= p->th_on * nap && a3 >= p->th_on * nah;
            match[i] = ok && a2 > 0 && (a2 - a4) >= p->th_match * nap && a4 >= p->th_match * nah;
            if (pred[i]) colp[i / M] = 1;
        }
        int npc = 0, hit = 0, npw = 0, nw = 0; double rsum = 0;
        for (int t = 0; t < nt; t++) { int c = touched[t] / M; if (colp[c] == 1) { colp[c] = 2; npc++; hit += cl[c] != 0; } }
        for (int c = 0; c < C; c++) if (cl[c]) {
            wins[nw++] = c; win_tot++; pc_win += colp[c] != 0;
            for (int m = 0; m < M; m++) if (pred[c * M + m]) { npw++; rsum += rel[base + c * M + m]; }
        }
        double relw = npw ? rsum / npw : 1.0;
        int sur = (npc > 0 && hit < p->s_th * npc) || relw < p->s_th || (p->recall && hit < p->s_th * nw);
        surprise[d] = sur;
        memset(cells + base, 0, CM);
        int nf = 0;
        for (int w_ = 0; w_ < nw; w_++) {                                     /* only winning columns fire a cell or learn by best */
            int c = wins[w_], best = 0; double bs = -1e300; int pc = colp[c] != 0;
            for (int m = 0; m < M; m++) {
                int i = c * M + m;
                double sc = pred[i] ? 1e6 + n_on[i] : (match[i] ? 1e3 + n_act[i] : -use[base + i]);
                if (sc > bs) { bs = sc; best = m; }
            }
            int pick = (p->ctx_mode || p->nodef) ? best : (sur ? 0 : best);
            if (p->burst && !pc) pick = 0;
            cells[base + c * M + pick] = 1;
            if (p->learn && best != 0) { f[c * M + best] = 1; fcell[nf++] = c * M + best; }
            if (p->learn) for (int m = 0; m < M; m++) { int i = c * M + m; if (pred[i]) rel[base + i] += p->lam * ((double)(m == best) - rel[base + i]); }
        }
        if (p->learn) {
            /* rel of predicted cells in columns that do not win: target 0 */
            for (int t = 0; t < nt; t++) { int i = touched[t]; if (pred[i] && !cl[i / M]) rel[base + i] += p->lam * (0.0 - rel[base + i]); }
            for (int q = 0; q < nf; q++) {                                       /* the f cells' active slots, before learning (as net7.c's act) */
                size_t o = (base + fcell[q]) * S; uint8_t *aq = af + (size_t)q * S;
                for (int s = 0; s < S; s++) aq[s] = (rw[o + s] > 0) && ctx[gsrc[o + s]];
            }
            for (size_t k = 0; k < nact; k++) {
                int32_t sl = actl[k]; int i = (int)(((size_t)sl - sbase) / S);
                if (f[i]) rw[sl] += p->eta * (1 - rw[sl]);
                else if (pred[i]) rw[sl] -= p->eta_minus * rw[sl];
            }
            if (P_) for (int q = 0; q < nf; q++) {                                /* making of connections, cells ascending */
                int i = fcell[q];
                size_t o = (base + i) * S;
                if (++stamp_id == 0x7fffffff) { memset(stamp, 0, sizeof(int) * NC); stamp_id = 1; }
                const uint8_t *a = af + (size_t)q * S;
                for (int s = 0; s < S; s++) {
                    if (rw[o + s] > 0 && rsrc[o + s] >= 0 && rsrc[o + s] < NC) stamp[rsrc[o + s]] = stamp_id;
                }
                int nm = 0;
                for (int j = 0; j < P_; j++) if (stamp[prev[j]] != stamp_id) miss[nm++] = prev[j];
                if (p->n_grow) {                                                 /* as net7.c */
                    int ap = 0, ah = 0;
                    for (int s = 0; s < S; s++) if (a[s]) { if (rsrc[o + s] >= CM) ah++; else ap++; }
                    int keep[2] = { p->n_grow - ap > 0 ? p->n_grow - ap : 0, p->n_grow - ah > 0 ? p->n_grow - ah : 0 };
                    int out = 0; memcpy(tmp, miss, sizeof(int) * nm);
                    for (int part = 0; part < 2; part++) {
                        int st = out, n = 0;
                        for (int j = 0; j < nm; j++) if ((tmp[j] >= CM) == part) miss[out + n++] = tmp[j];
                        if (n > keep[part]) {
                            for (int t = 0; t < keep[part]; t++) { int r = t + (int)(xr() % (uint64_t)(n - t)); int x = miss[st + t]; miss[st + t] = miss[st + r]; miss[st + r] = x; }
                            n = keep[part];
                            for (int x = 1; x < n; x++) { int v = miss[st + x], y = x - 1; while (y >= 0 && miss[st + y] > v) { miss[st + y + 1] = miss[st + y]; y--; } miss[st + y + 1] = v; }
                        }
                        out = st + n;
                    }
                    nm = out;
                }
                int no = 0;
                for (int s = 0; s < S; s++) if (!a[s]) order[no++] = s;
                for (int x = 1; x < no; x++) { int v = order[x], y = x - 1; while (y >= 0 && rw[o + order[y]] > rw[o + v]) { order[y + 1] = order[y]; y--; } order[y + 1] = v; }
                for (int j = 0; j < nm && j < no; j++) {
                    int s = order[j]; int32_t sl = (int32_t)(o + s);
                    ix_remove(head, nxt, prv, gsrc, sl);
                    rsrc[o + s] = miss[j]; rw[o + s] = p->w_new; gsrc[o + s] = miss[j] + (int64_t)d * NC;
                    if (rw[o + s] > 0) ix_insert(head, nxt, prv, gsrc[o + s], sl);
                }
            }
            for (int q = 0; q < nf; q++) use[base + fcell[q]] += 1.0;
        }
        /* clear the per-region state */
        for (int t = 0; t < nt; t++) { int i = touched[t]; n_on[i] = n_act[i] = sH[i] = aH[i] = 0; pred[i] = match[i] = 0; colp[i / M] = 0; }
        for (int q = 0; q < nf; q++) f[fcell[q]] = 0;
    }
    stats[0] = win_tot ? (double)pc_win / win_tot : 0.0;
    free(n_on); free(n_act); free(sH); free(aH); free(pred); free(match); free(f); free(af); free(colp); free(touched); free(prev);
    free(miss); free(tmp); free(stamp); free(order); free(wins); free(fcell); free(actl);
}
