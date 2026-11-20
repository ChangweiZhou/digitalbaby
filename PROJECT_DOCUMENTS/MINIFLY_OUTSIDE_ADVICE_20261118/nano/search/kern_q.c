/* nano/search, PLAN_1.md: cells that count, kinds found by search, read by learned trust. The plain Python reference is search.py (class Ref). */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define MAXL 6
#define D 16
#define NLEV 40
#define MAXS 80
#define MAXK 512
typedef struct {
    int64_t S, H, K, NLAY, NB, stepmode, pos, evict, ins;
    double eta, e, ncap;
    uint64_t *KEY; uint32_t *KU; uint32_t *NN; uint8_t *UU; uint16_t *CC; uint8_t *dead;
    int32_t *nl, *ln, *lay; uint32_t *uid;
    double *T, *A, *Wa, *Wg;
    uint64_t *rkey; int64_t *rslot; int8_t *rlev; double *rs, *rp; int8_t *rvalid;
} St;

static inline uint64_t mix0(uint32_t uid) { return ((uint64_t)uid + 1) * 0x9E3779B97F4A7C15ULL; }
static inline uint64_t mix1(uint64_t h, int v) { return (h ^ (uint64_t)(v + 1)) * 0x100000001B3ULL; }
static inline uint64_t fin(uint64_t h) { h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 29; return h ? h : 1; }
static inline int level(uint32_t n, int u) { int bn = 0; uint32_t m = n; while (m > 1 && bn < 9) { m >>= 1; bn++; } return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3); }
static inline int64_t lookup(const St *z, uint64_t key) {
    int64_t b = (int64_t)(key % (uint64_t)z->NB) * 4;
    for (int i = 0; i < 4; i++) if (z->KEY[b + i] == key) return b + i;
    return -1;
}
static int64_t insert(St *z, uint64_t key, uint32_t uid) {
    int64_t b = (int64_t)(key % (uint64_t)z->NB) * 4, w = -1;
    for (int i = 0; i < 4 && w < 0; i++) if (z->KEY[b + i] == 0 || z->dead[z->KU[b + i]]) w = b + i;
    if (w < 0) { w = b; for (int i = 1; i < 4; i++) if (z->NN[(b + i) * z->H] < z->NN[w * z->H]) w = b + i; z->evict++; }
    z->KEY[w] = key; z->KU[w] = uid; z->ins++;
    memset(z->NN + w * z->H, 0, z->H * sizeof(uint32_t)); memset(z->UU + w * z->H, 0, z->H); memset(z->CC + w * z->H * z->S, 0, z->H * z->S * sizeof(uint16_t));
    return w;
}
/* what the system says before the byte at xt arrives; xt[-d] is the byte d places back. Fills ring entry a. */
static void fwd(const St *z, const int32_t *xt, int64_t a, int8_t *gout) {
    const int64_t S = z->S, H = z->H, K = z->K, NLAY = z->NLAY, R = NLAY + 1;
    int G[8][8]; double br[MAXS];
    uint64_t *keys = z->rkey + a * K; int64_t *slots = z->rslot + a * K; int8_t *lev = z->rlev + a * K * H; double *s = z->rs + a * K * H * S, *p = z->rp + a * R * H * S;
    const double le = log(z->e);
    for (int64_t l = 0; l <= NLAY; l++) {
        for (int64_t k = 0; k < K; k++) if (z->lay[k] == l) {
            uint64_t h = mix0(z->uid[k]);
            for (int i = 0; i < z->nl[k]; i++) { const int line = z->ln[k * MAXL + i]; h = mix1(h, line < D ? xt[-(line + 1)] : G[(line - D) / H][(line - D) % H]); }
            keys[k] = fin(h); const int64_t sl = lookup(z, keys[k]); slots[k] = sl;
            for (int64_t hh = 0; hh < H; hh++) {
                lev[k * H + hh] = -1; if (sl < 0) continue;
                const uint32_t n = z->NN[sl * H + hh]; if (n < 1) continue;
                lev[k * H + hh] = (int8_t)level(n, z->UU[sl * H + hh]);
                const uint16_t *c = z->CC + (sl * H + hh) * S; double *sv = s + (k * H + hh) * S;
                const double sc = n > z->ncap ? z->ncap / n : 1.0, den = log((n > z->ncap ? z->ncap : (double)n) + S * z->e), lz = le - den;
                for (int64_t v = 0; v < S; v++) sv[v] = c[v] ? log(c[v] * sc + z->e) - den : lz;
            }
        }
        const int64_t nh = l < NLAY ? H : 1;
        for (int64_t hh = 0; hh < nh; hh++) {
            for (int64_t v = 0; v < S; v++) br[v] = 0;
            for (int64_t k = 0; k < K; k++) if (z->lay[k] <= l && lev[k * H + hh] >= 0) {
                const double t = z->T[((l * H + hh) * K + k) * NLEV + lev[k * H + hh]]; const double *sv = s + (k * H + hh) * S;
                if (t != 0) for (int64_t v = 0; v < S; v++) br[v] += t * sv[v];
            }
            int g = 0; double mx = br[0], zz = 0; for (int64_t v = 1; v < S; v++) if (br[v] > mx) { mx = br[v]; g = (int)v; }
            double *pp = p + (l * H + hh) * S; for (int64_t v = 0; v < S; v++) { pp[v] = exp(br[v] - mx); zz += pp[v]; }
            for (int64_t v = 0; v < S; v++) pp[v] /= zz;
            G[l][hh] = g; if (gout && l < NLAY) gout[l * H + hh] = (int8_t)g;
        }
    }
}
/* the byte y has arrived; ring entry of the present byte is pos % H */
static void learn(St *z, int y) {
    const int64_t S = z->S, H = z->H, K = z->K, NLAY = z->NLAY, R = NLAY + 1;
    for (int64_t hh = 0; hh < H; hh++) {
        if (z->pos - hh < 0) continue; const int64_t a = (z->pos - hh) % H; if (!z->rvalid[a]) continue;
        const int8_t *lev = z->rlev + a * K * H; const double *s = z->rs + a * K * H * S;
        for (int64_t r = 0; r <= NLAY; r++) {
            if (r == NLAY && hh > 0) continue;
            const double *pp = z->rp + ((a * R + r) * H + hh) * S;
            for (int64_t k = 0; k < K; k++) if (z->lay[k] <= r && lev[k * H + hh] >= 0) {
                const double *sv = s + (k * H + hh) * S; double g = sv[y]; for (int64_t v = 0; v < S; v++) g -= pp[v] * sv[v];
                const int64_t i = ((r * H + hh) * K + k) * NLEV + lev[k * H + hh];
                if (r == NLAY) z->Wa[k] += z->T[i] * g; else z->Wg[k] += z->T[i] * g;
                if (z->stepmode == 0) { z->T[i] += z->eta / sqrt(1.0 + z->A[i]) * g; z->A[i] += 1.0; }
                else { z->A[i] += g * g; if (z->A[i] > 0) z->T[i] += z->eta * g / sqrt(z->A[i]); }
            }
        }
    }
    for (int64_t hh = 0; hh < H; hh++) {
        if (z->pos - hh < 0) continue; const int64_t a = (z->pos - hh) % H; if (!z->rvalid[a]) continue;
        for (int64_t k = 0; k < K; k++) {
            if (z->lay[k] == NLAY && hh > 0) continue;
            const uint64_t key = z->rkey[a * K + k]; if (!key) continue;
            int64_t sl = z->rslot[a * K + k];
            if (sl < 0 || z->KEY[sl] != key) { sl = hh == 0 ? lookup(z, key) : -1; if (sl < 0 && hh == 0) sl = insert(z, key, z->uid[k]); z->rslot[a * K + k] = sl; }
            if (sl < 0) continue;
            uint16_t *c = z->CC + (sl * H + hh) * S;
            if (c[y] == 65535) { uint32_t n = 0; int u = 0; for (int64_t v = 0; v < S; v++) { c[v] >>= 1; n += c[v]; u += c[v] > 0; } z->NN[sl * H + hh] = n; z->UU[sl * H + hh] = (uint8_t)u; }
            if (c[y] == 0) z->UU[sl * H + hh]++;
            c[y]++; z->NN[sl * H + hh]++;
        }
    }
}
/* bytes lo..hi-1 of x (lo >= D). P (may be null): the answer's probabilities, (hi-lo) by S. cont = 0: nothing pending is carried in. */
void run(St *z, const int32_t *x, int64_t lo, int64_t hi, int do_learn, int cont, double *P, int8_t *Gout) {
    const int64_t S = z->S, H = z->H, R = z->NLAY + 1;
    if (do_learn && !cont) { z->pos = 0; memset(z->rvalid, 0, H); }
    for (int64_t t = lo; t < hi; t++) {
        const int64_t a = do_learn ? z->pos % H : H;
        fwd(z, x + t, a, Gout ? Gout + (t - lo) * z->NLAY * H : 0);
        if (P) memcpy(P + (t - lo) * S, z->rp + ((a * R + z->NLAY) * H) * S, S * sizeof(double));
        if (do_learn) { z->rvalid[a] = 1; learn(z, x[t]); z->pos++; }
    }
}
/* n contexts of D bytes (oldest first); the system writes ngen bytes of its own after each. Nothing changes. */
void gen(St *z, const int32_t *ctx, int64_t n, int64_t ngen, int32_t *out) {
    const int64_t S = z->S, H = z->H, R = z->NLAY + 1; int32_t buf[D + 64];
    for (int64_t i = 0; i < n; i++) {
        memcpy(buf, ctx + i * D, D * sizeof(int32_t));
        for (int64_t j = 0; j < ngen; j++) {
            fwd(z, buf + D + j, H, 0); const double *pp = z->rp + ((H * R + z->NLAY) * H) * S; int g = 0;
            for (int64_t v = 1; v < S; v++) if (pp[v] > pp[g]) g = (int)v;
            buf[D + j] = g; out[i * ngen + j] = g;
        }
    }
}
/* PLAN_1.md, addition 1: newly born kinds count over the bytes kept (x[lo..hi-1] with the guesses g as they were made then); nothing else changes */
void backfill(St *z, const int32_t *x, const int8_t *g, int64_t lo, int64_t hi, const int32_t *ks, int64_t nk) {
    const int64_t S = z->S, H = z->H, NLAY = z->NLAY;
    for (int64_t t = lo; t < hi; t++) for (int64_t j = 0; j < nk; j++) {
        const int64_t k = ks[j]; uint64_t h = mix0(z->uid[k]); const int8_t *gt = g + (t - lo) * NLAY * H;
        for (int i = 0; i < z->nl[k]; i++) { const int line = z->ln[k * MAXL + i]; h = mix1(h, line < D ? x[t - (line + 1)] : gt[line - D]); }
        const uint64_t key = fin(h); int64_t sl = lookup(z, key); if (sl < 0) sl = insert(z, key, z->uid[k]);
        for (int64_t hh = 0; hh < (z->lay[k] == NLAY ? 1 : H) && t + hh < hi; hh++) {
            const int y = x[t + hh]; uint16_t *c = z->CC + (sl * H + hh) * S;
            if (c[y] == 65535) { uint32_t n = 0; int u = 0; for (int64_t v = 0; v < S; v++) { c[v] >>= 1; n += c[v]; u += c[v] > 0; } z->NN[sl * H + hh] = n; z->UU[sl * H + hh] = (uint8_t)u; }
            if (c[y] == 0) z->UU[sl * H + hh]++;
            c[y]++; z->NN[sl * H + hh]++;
        }
    }
}
int64_t sizeof_st(void) { return sizeof(St); }
