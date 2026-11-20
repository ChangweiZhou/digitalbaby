#!/usr/bin/env python3
"""nano/suite (PLAN_1.md): one system of counting cells with ONE setting, on byte streams of different kinds.
    python3 suite.py check                                   the compiled kernel against the plain-Python reference; and against nano/table/gate.py on tiny Shakespeare
    python3 suite.py run stream=<name> [kinds=runs|runs+since|runs+pool|runs+since+pool] [gate=0|1|2] [birth=0] [recall=0] [M=8] [rule=0] [eta=0.00005] [e=0.02] [born=0.25] [tag=..]
The system is nano/merge/merge.py's (cells that count; an answering cell whose branches compete on the sum of trust times message; a trust for each kind and
level of counts), re-written so that a cell holds only the counts it has (the message of a cell is the same number as in merge.py: log((c(v)+e)/(n+S e));
the part of it that is the same for every value cancels in the answer and in the change of the trust, so only values the cell has met are touched).
The number of values S is a parameter (at most 128). Kinds of sender:
    a run or set of delays (type 0): keyed on the bytes at those delays (delay 1 = the byte just read);
    since (type 1), one kind for each value v: keyed on (bytes since v last came, capped at 63; the byte just read) (nano/table/place.py);
    recall (type 2), rank r: the (r+1)-th last place where the last M bytes stood before; it says the byte that followed there, as a cell that has counted once;
        its trust is indexed by how long the two runs agree (PLAN_3.md).
gate=1: a second answering cell with, for each of the last J delays, a part of the trust indexed by (kind, byte at that delay) (nolevel);  gate=2: by (kind, level, byte).
birth=b>0: a cell is not born at a byte to which the first cell gave at least probability b (nano/table/PLAN_19.md).
rule: the step of a trust. 0: eta;  1: eta / sqrt(1 + times used);  2: eta / sqrt(sum of its own squared changes' drivers) (PLAN_4.md)."""
import ctypes, hashlib, itertools, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); NL = 40; LMAX = 2048
PD = ctypes.POINTER(ctypes.c_double); PI = ctypes.POINTER(ctypes.c_int32)

SRC = r'''
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define NL 40
#define SMAX 128
#define KMAX 600
#define JMAX 8
#define LMAX 2048
typedef struct {
    int S, K, gate, J, M, rule; double eta, e, birth;
    int ktype[KMAX], knd[KMAX], kdel[KMAX][8];
    double *T, *Tu, *Ta;            /* K x NL: the first cell's trusts, times used, sum of squared drivers */
    double *G, *Gu, *Ga; int64_t gsize; int GL;   /* the second cell's parts: J x K x GL x S (GL = 1 without the level, NL with it) */
    /* cells */
    uint64_t *hkey; uint16_t *hkind; int32_t *hcell; int64_t hcap, ncell;
    uint32_t *cN; uint16_t *cU, *ccap; int64_t *cptr; int64_t ccells;
    uint8_t *pv; uint32_t *pc; int64_t ptop, pcap;
    /* recall */
    int32_t *head, *prev; int64_t nmax; int hbits;
    /* since */
    int64_t last[SMAX]; int64_t synced;
    double lt[4096];
} Sy;
static inline int level(uint32_t n, int u) { int bn = 0; int64_t m = n; while (m > 1 && bn < 9) { m >>= 1; bn++; } return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3); }
static inline int reclevel(int L, int M) { int d = L - M; if (d < 16) return d; int b = 16; int64_t m = (d - 16) / 8; while (m > 0 && b < NL - 1) { m >>= 1; b++; } return b; }
static inline uint64_t mix(uint64_t z) { z += 0x9E3779B97F4A7C15ULL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
static inline int64_t slot_of(const Sy *s, uint64_t key, int kind) {
    int64_t i = mix(key ^ ((uint64_t)kind * 0xD6E8FEB86659FD93ULL)) & (s->hcap - 1);
    while (s->hcell[i] >= 0 && !(s->hkey[i] == key && s->hkind[i] == kind)) i = (i + 1) & (s->hcap - 1);
    return i;
}
static void grow(Sy *s) {
    int64_t oc = s->hcap; uint64_t *ok = s->hkey; uint16_t *oki = s->hkind; int32_t *oce = s->hcell;
    s->hcap = oc * 2; s->hkey = malloc(s->hcap * 8); s->hkind = malloc(s->hcap * 2); s->hcell = malloc(s->hcap * 4); memset(s->hcell, 0xff, s->hcap * 4);
    for (int64_t i = 0; i < oc; i++) if (oce[i] >= 0) { int64_t j = slot_of(s, ok[i], oki[i]); s->hkey[j] = ok[i]; s->hkind[j] = oki[i]; s->hcell[j] = oce[i]; }
    free(ok); free(oki); free(oce);
}
static int32_t new_cell(Sy *s) {
    if (s->ncell == s->ccells) { s->ccells *= 2; s->cN = realloc(s->cN, s->ccells * 4); s->cU = realloc(s->cU, s->ccells * 2); s->ccap = realloc(s->ccap, s->ccells * 2); s->cptr = realloc(s->cptr, s->ccells * 8); }
    int32_t c = (int32_t)s->ncell++; s->cN[c] = 0; s->cU[c] = 0; s->ccap[c] = 0; s->cptr[c] = 0; return c;
}
static void count(Sy *s, int32_t c, int y) {
    uint8_t *pv = s->pv + s->cptr[c]; uint32_t *pc = s->pc + s->cptr[c]; int u = s->cU[c];
    for (int i = 0; i < u; i++) if (pv[i] == y) { pc[i]++; s->cN[c]++; return; }
    if (u == s->ccap[c]) {
        int nc = u ? u * 2 : 1; if (nc > SMAX) nc = SMAX;
        if (s->ptop + nc > s->pcap) { s->pcap = s->pcap * 2 + nc; s->pv = realloc(s->pv, s->pcap); s->pc = realloc(s->pc, s->pcap * 4); }
        memcpy(s->pv + s->ptop, s->pv + s->cptr[c], u); memcpy(s->pc + s->ptop, s->pc + s->cptr[c], (size_t)u * 4); s->cptr[c] = s->ptop; s->ptop += nc; s->ccap[c] = nc;
    }
    s->pv[s->cptr[c] + u] = y; s->pc[s->cptr[c] + u] = 1; s->cU[c] = u + 1; s->cN[c]++;
}
Sy *sy_new(int S, int K, const int32_t *ktype, const int32_t *knd, const int32_t *kdel, const double *kborn, double eta, double e, int rule, int gate, int J, double birth, int M, int64_t nmax) {
    Sy *s = calloc(1, sizeof(Sy)); s->S = S; s->K = K; s->gate = gate; s->J = J; s->M = M; s->rule = rule; s->eta = eta; s->e = e; s->birth = birth; s->nmax = nmax;
    for (int k = 0; k < K; k++) { s->ktype[k] = ktype[k]; s->knd[k] = knd[k]; for (int i = 0; i < 8; i++) s->kdel[k][i] = kdel[k * 8 + i]; }
    s->T = malloc(sizeof(double) * K * NL); s->Tu = calloc(K * NL, sizeof(double)); s->Ta = calloc(K * NL, sizeof(double));
    for (int k = 0; k < K; k++) for (int l = 0; l < NL; l++) s->T[k * NL + l] = kborn[k];
    s->GL = gate == 2 ? NL : 1; s->gsize = gate ? (int64_t)J * K * s->GL * S : 0;
    if (gate) { s->G = calloc(s->gsize, sizeof(double)); s->Gu = calloc(s->gsize, sizeof(double)); s->Ga = calloc(s->gsize, sizeof(double)); }
    s->hcap = 1 << 16; s->hkey = malloc(s->hcap * 8); s->hkind = malloc(s->hcap * 2); s->hcell = malloc(s->hcap * 4); memset(s->hcell, 0xff, s->hcap * 4);
    s->ccells = 1 << 16; s->cN = malloc(s->ccells * 4); s->cU = malloc(s->ccells * 2); s->ccap = malloc(s->ccells * 2); s->cptr = malloc(s->ccells * 8);
    s->pcap = 1 << 16; s->pv = malloc(s->pcap); s->pc = malloc(s->pcap * 4);
    s->hbits = 22; s->head = malloc(sizeof(int32_t) << s->hbits); memset(s->head, 0xff, sizeof(int32_t) << s->hbits); s->prev = malloc(sizeof(int32_t) * (nmax + 1));
    for (int v = 0; v < SMAX; v++) s->last[v] = -1; s->synced = 0;
    for (int c = 0; c < 4096; c++) s->lt[c] = log(c + e) - log(e);
    return s;
}
void sy_free(Sy *s) { free(s->T); free(s->Tu); free(s->Ta); free(s->G); free(s->Gu); free(s->Ga); free(s->hkey); free(s->hkind); free(s->hcell); free(s->cN); free(s->cU); free(s->ccap); free(s->cptr); free(s->pv); free(s->pc); free(s->head); free(s->prev); free(s); }
int64_t sy_cells(const Sy *s) { int64_t n = 0; for (int64_t c = 0; c < s->ncell; c++) n += s->cN[c] > 0; return n; }
int64_t sy_counts(const Sy *s) { int64_t n = 0; for (int64_t c = 0; c < s->ncell; c++) n += s->cU[c]; return n; }
int64_t sy_cells_of(const Sy *s, int kind) { int64_t n = 0; for (int64_t i = 0; i < s->hcap; i++) if (s->hcell[i] >= 0 && s->hkind[i] == kind && s->cN[s->hcell[i]] > 0) n++; return n; }
int64_t sy_once(const Sy *s) { int64_t n = 0; for (int64_t c = 0; c < s->ncell; c++) n += s->cN[c] == 1; return n; }
double *sy_T(Sy *s) { return s->T; }  double *sy_Tu(Sy *s) { return s->Tu; }  double *sy_G(Sy *s) { return s->G; }  double *sy_Gu(Sy *s) { return s->Gu; }  int64_t sy_gsize(Sy *s) { return s->gsize; }
static inline uint32_t rhash(const int32_t *x, int64_t end, int M, int hbits) { uint64_t h = 0; for (int i = 1; i <= M; i++) h = h * 0x100000001B3ULL + (uint64_t)(x[end - i] + 1); return (uint32_t)(mix(h) >> (64 - hbits)); }
static inline double step_of(const Sy *s, double used, double acc) { return s->rule == 0 ? s->eta : s->rule == 1 ? s->eta / sqrt(1.0 + used) : s->eta / sqrt(acc); }
/* reads or answers the bytes x[lo..hi-1] of the stream x (the same array at every call). nl1, nl2: minus the log of the probability each answering cell gave
   to the byte that came; P1, P2 (may be null): all probabilities.  learn = 0: nothing changes (the since clocks, which are context, still move). */
void sy_run(Sy *s, const int32_t *x, int64_t lo, int64_t hi, int learn, double *nl1, double *nl2, double *P1, double *P2) {
    const int S = s->S, K = s->K, M = s->M; double a[SMAX], p[SMAX], a2[SMAX], p2[SMAX];
    static int lev[KMAX], nv[KMAX]; static const uint8_t *vv[KMAX]; static double dv[KMAX][SMAX]; static int64_t sl[KMAX]; static uint64_t keys[KMAX]; static uint8_t rb[KMAX]; static int64_t gix[KMAX][JMAX];
    const double done = log(1 + s->e) - log(s->e);
    if (s->synced != lo) { for (int v = 0; v < SMAX; v++) s->last[v] = -1; for (int64_t t = 0; t < lo; t++) s->last[x[t]] = t; s->synced = lo; }
    for (int64_t t = lo; t < hi; t++) {
        const int y = x[t]; int64_t rq[8]; int nrq = -1;
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int k = 0; k < K; k++) {
            lev[k] = -1;
            if (s->ktype[k] == 2) {                                   /* recall */
                if (nrq < 0) { nrq = 0; if (t >= M) { int32_t q = s->head[rhash(x, t, M, s->hbits)];
                    while (q >= 0 && nrq < 8) { int ok = 1; for (int i = 1; i <= M; i++) if (x[q - i] != x[t - i]) { ok = 0; break; } if (ok) rq[nrq++] = q; q = s->prev[q]; } } }
                const int r = s->knd[k]; if (r >= nrq) continue;
                const int64_t q = rq[r]; int L = M; while (L < LMAX && q - L - 1 >= 0 && x[q - L - 1] == x[t - L - 1]) L++;
                lev[k] = reclevel(L, M); rb[k] = (uint8_t)x[q]; nv[k] = 1; vv[k] = &rb[k]; dv[k][0] = done; a[rb[k]] += s->T[k * NL + lev[k]] * done; continue;
            }
            uint64_t key = 0;
            if (s->ktype[k] == 0) { for (int i = 0; i < s->knd[k]; i++) { const int64_t q = t - s->kdel[k][i]; key |= (uint64_t)(q >= 0 ? x[q] : 0) << (7 * i); } }
            else { const int v = s->knd[k]; const int64_t sn = s->last[v] < 0 ? 63 : (t - 1 - s->last[v] > 63 ? 63 : t - 1 - s->last[v]); key = (uint64_t)sn * 128 + (t >= 1 ? x[t - 1] : 0); }
            keys[k] = key; sl[k] = slot_of(s, key, k); const int32_t c = s->hcell[sl[k]]; if (c < 0 || s->cN[c] < 1) continue;
            const int u = s->cU[c]; lev[k] = level(s->cN[c], u); nv[k] = u; vv[k] = s->pv + s->cptr[c]; const uint32_t *pc = s->pc + s->cptr[c]; const double tk = s->T[k * NL + lev[k]];
            for (int i = 0; i < u; i++) { const double d = pc[i] < 4096 ? s->lt[pc[i]] : log(pc[i] + s->e) - log(s->e); dv[k][i] = d; a[vv[k][i]] += tk * d; }
        }
        double mx = -1e300, z = 0;
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (nl1) nl1[t - lo] = -log(p[y]);
        if (P1) for (int v = 0; v < S; v++) P1[(t - lo) * S + v] = p[v];
        if (s->gate) {
            for (int v = 0; v < S; v++) a2[v] = a[v];
            for (int k = 0; k < K; k++) if (lev[k] >= 0) {
                double tk = 0;
                for (int j = 0; j < s->J; j++) { const int64_t q = t - 1 - j; gix[k][j] = (((int64_t)j * K + k) * s->GL + (s->GL > 1 ? lev[k] : 0)) * S + (q >= 0 ? x[q] : 0); tk += s->G[gix[k][j]]; }
                if (tk != 0) for (int i = 0; i < nv[k]; i++) a2[vv[k][i]] += tk * dv[k][i];
            }
            mx = -1e300; z = 0;
            for (int v = 0; v < S; v++) if (a2[v] > mx) mx = a2[v];
            for (int v = 0; v < S; v++) { p2[v] = exp(a2[v] - mx); z += p2[v]; }
            for (int v = 0; v < S; v++) p2[v] /= z;
            if (nl2) nl2[t - lo] = -log(p2[y]);
            if (P2) for (int v = 0; v < S; v++) P2[(t - lo) * S + v] = p2[v];
        }
        if (learn) {
            for (int k = 0; k < K; k++) if (lev[k] >= 0) {
                double g = 0, g2 = 0;
                for (int i = 0; i < nv[k]; i++) { const int v = vv[k][i]; const double d = dv[k][i]; g += ((v == y) - p[v]) * d; if (s->gate) g2 += ((v == y) - p2[v]) * d; }
                const int64_t i = k * NL + lev[k]; s->Ta[i] += g * g; if (g != 0 || s->rule != 2) s->T[i] += step_of(s, s->Tu[i], s->Ta[i]) * g; s->Tu[i] += 1.0;
                if (s->gate) for (int j = 0; j < s->J; j++) { const int64_t i2 = gix[k][j]; s->Ga[i2] += g2 * g2; if (g2 != 0 || s->rule != 2) s->G[i2] += step_of(s, s->Gu[i2], s->Ga[i2]) * g2; s->Gu[i2] += 1.0; }
            }
            for (int k = 0; k < K; k++) if (s->ktype[k] != 2) {
                int32_t c = s->hcell[sl[k]];
                if (c < 0) {
                    if (s->birth > 0 && p[y] >= s->birth) continue;
                    if ((s->ncell + 1) * 10 > s->hcap * 6) { grow(s); for (int j = 0; j < K; j++) if (s->ktype[j] != 2) sl[j] = slot_of(s, keys[j], j); }
                    c = new_cell(s); s->hkey[sl[k]] = keys[k]; s->hkind[sl[k]] = (uint16_t)k; s->hcell[sl[k]] = c;
                    for (int j = k + 1; j < K; j++) if (s->ktype[j] != 2) sl[j] = slot_of(s, keys[j], j);      /* a slot found empty may now be taken */
                }
                else if (s->birth > 0 && s->cN[c] == 0 && p[y] >= s->birth) continue;
                count(s, c, y);
            }
            if (M > 0 && t + 1 >= M && t + 1 <= s->nmax) { const uint32_t h = rhash(x, t + 1, M, s->hbits); s->prev[t + 1] = s->head[h]; s->head[h] = (int32_t)(t + 1); }
        }
        s->last[y] = t; s->synced = t + 1;
    }
}
'''

def kern():
    h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_suite_{h}.so"); src = os.path.join(HERE, "kern_suite.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd, cl, vp = ctypes.c_int, ctypes.c_double, ctypes.c_int64, ctypes.c_void_p
    L.sy_new.argtypes = [ci, ci, PI, PI, PI, PD, cd, cd, ci, ci, ci, cd, ci, cl]; L.sy_new.restype = vp; L.sy_free.argtypes = [vp]; L.sy_free.restype = None
    L.sy_run.argtypes = [vp, PI, cl, cl, ci, PD, PD, PD, PD]; L.sy_run.restype = None
    for f in ("sy_cells", "sy_counts", "sy_once", "sy_gsize"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = cl
    L.sy_cells_of.argtypes = [vp, ci]; L.sy_cells_of.restype = cl
    for f in ("sy_T", "sy_Tu", "sy_G", "sy_Gu"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = PD
    return L
p32 = lambda a: a.ctypes.data_as(PI); pd = lambda a: a.ctypes.data_as(PD) if a is not None else None

# ---------------------------------------------------------------- kinds
def kinds_of(spec, S, recall=0, born=0.25):
    """A list of (type, nd or value or rank, delays, trust at birth, name). spec: 'runs', 'runs+since', 'runs+pool', 'runs+since+pool', 'sums' (all sets of at most four of nine delays)."""
    ks = []; parts = spec.split("+")
    if "runs" in parts: ks += [(0, k, tuple(range(1, k + 1)), born, f"run{k}") for k in range(7)]
    if "sums" in parts: ks += [(0, len(c), c, born, "set" + "".join(map(str, c))) for m in range(5) for c in itertools.combinations(range(1, 10), m)]
    if "since" in parts: ks += [(1, v, (), 0.0, f"since{v}") for v in range(S)]                       # nano/table/place.py arm B: born with no trust (PLAN_8.md, addition 1)
    if "pool" in parts:                                                                            # nano/table/place.py pool(): two or three places among the last eight, less the two that are runs
        for m in (2, 3):
            for c in itertools.combinations(range(8), m):
                if c in ((0, 1), (0, 1, 2)): continue
                ks.append((0, m, tuple(d + 1 for d in c), 0.0, "pool" + "".join(map(str, c))))
    ks += [(2, r, (), 0.0, f"recall{r}") for r in range(recall)]
    return ks

class Sys:
    def __init__(s, S, kinds, x, eta=0.00005, e=0.02, rule=0, gate=0, J=6, birth=0.0, M=8):
        s.S, s.kinds, s.K, s.gate, s.J, s.M = S, kinds, len(kinds), gate, J, M; assert S <= 128 and s.K <= 600 and J <= 8; s.L = kern(); s.x = np.ascontiguousarray(x.astype(np.int32)); assert s.x.max() < S
        kt = np.array([k[0] for k in kinds], np.int32); kn = np.array([k[1] for k in kinds], np.int32); kd = np.zeros((s.K, 8), np.int32); kb = np.array([k[3] for k in kinds], np.float64)
        for i, k in enumerate(kinds): kd[i, :len(k[2])] = k[2]
        s.h = s.L.sy_new(S, s.K, p32(kt), p32(kn), p32(np.ascontiguousarray(kd)), pd(kb), eta, e, rule, gate, J, birth, M if any(k[0] == 2 for k in kinds) else 0, len(s.x))
    def run(s, lo, hi, learn, want_p=False):
        n = hi - lo; nl1 = np.zeros(n); nl2 = np.zeros(n) if s.gate else None; P1 = np.zeros((n, s.S)) if want_p else None; P2 = np.zeros((n, s.S)) if want_p and s.gate else None
        s.L.sy_run(s.h, p32(s.x), lo, hi, int(learn), pd(nl1), pd(nl2), pd(P1), pd(P2)); return (nl1, nl2, P1, P2) if want_p else (nl1, nl2)
    def T(s): return np.ctypeslib.as_array(s.L.sy_T(s.h), (s.K, NL)).copy()
    def Tu(s): return np.ctypeslib.as_array(s.L.sy_Tu(s.h), (s.K, NL)).copy()
    def G(s): n = s.L.sy_gsize(s.h); return np.ctypeslib.as_array(s.L.sy_G(s.h), (n,)).copy() if n else np.zeros(0)
    def Gu(s): n = s.L.sy_gsize(s.h); return np.ctypeslib.as_array(s.L.sy_Gu(s.h), (n,)).copy() if n else np.zeros(0)
    def sizes(s):
        return dict(cells=int(s.L.sy_cells(s.h)), counts_above_zero=int(s.L.sy_counts(s.h)), cells_met_once=int(s.L.sy_once(s.h)), kinds=s.K, trusts_used=int((s.Tu() > 0).sum()), parts_used=int((s.Gu() > 0).sum()), parts=int(s.L.sy_gsize(s.h)))
    def __del__(s):
        try: s.L.sy_free(s.h)
        except Exception: pass

# ---------------------------------------------------------------- the reference in plain Python and numpy (every cell holds all S counts; the message is written out in full)
def level_np(n, u):
    bn = 0; m = int(n)
    while m > 1 and bn < 9: m >>= 1; bn += 1
    return bn * 4 + (0 if u <= 1 else 1 if u <= 2 else 2 if u <= 4 else 3)
def reclevel_np(L, M):
    d = L - M
    if d < 16: return d
    b = 16; m = (d - 16) // 8
    while m > 0 and b < NL - 1: m >>= 1; b += 1
    return b

def ref_run(S, kinds, x, lo, hi, state=None, learn=True, eta=0.00005, e=0.02, rule=0, gate=0, J=6, birth=0.0, M=8):
    K = len(kinds); GL = NL if gate == 2 else 1
    if state is None: state = dict(cells={}, T=np.array([[k[3]] * NL for k in kinds], np.float64), Tu=np.zeros((K, NL)), Ta=np.zeros((K, NL)), G=np.zeros((J, K, GL, S)), Gu=np.zeros((J, K, GL, S)), Ga=np.zeros((J, K, GL, S)), places={})
    cells, T, Tu, Ta, G, Gu, Ga, places = (state[k] for k in ("cells", "T", "Tu", "Ta", "G", "Gu", "Ga", "places")); P1 = np.zeros((hi - lo, S)); P2 = np.zeros((hi - lo, S)); xb = lambda q: int(x[q]) if q >= 0 else 0
    step = lambda used, acc: eta if rule == 0 else eta / np.sqrt(1.0 + used) if rule == 1 else eta / np.sqrt(acc)
    for t in range(lo, hi):
        y = int(x[t]); a = np.zeros(S); act = []; keys = []; rq = places.get(tuple(int(v) for v in x[t - M:t]), []) if t >= M else []
        for k, (typ, nd, ds, _, _) in enumerate(kinds):
            if typ == 2:
                keys.append(None)
                if nd < len(rq):
                    q = rq[-1 - nd]; L = M
                    while L < LMAX and q - L - 1 >= 0 and x[q - L - 1] == x[t - L - 1]: L += 1
                    c = np.zeros(S); c[int(x[q])] = 1; sv = np.log(c + e) - np.log(1 + S * e); l = reclevel_np(L, M); act.append((k, l, sv)); a += T[k, l] * sv
                continue
            if typ == 0: key = (k,) + tuple(xb(t - d) for d in ds)
            else:
                lastv = max([q for q in range(t) if x[q] == nd], default=-1) if t < 300 else None
                if lastv is None: w = np.flatnonzero(x[max(0, t - 64):t] == nd); lastv = max(0, t - 64) + int(w[-1]) if len(w) else -1
                key = (k, 63 if lastv < 0 else min(63, t - 1 - lastv), xb(t - 1))
            keys.append(key); c = cells.get(key)
            if c is not None and c.sum() >= 1:
                n = c.sum(); l = level_np(n, int((c > 0).sum())); sv = np.log(c + e) - np.log(n + S * e); act.append((k, l, sv)); a += T[k, l] * sv
        p = np.exp(a - a.max()); p /= p.sum(); P1[t - lo] = p; p2 = p
        if gate:
            a2 = a.copy()
            for k, l, sv in act: a2 += sum(G[j, k, l if GL > 1 else 0, xb(t - 1 - j)] for j in range(J)) * sv
            p2 = np.exp(a2 - a2.max()); p2 /= p2.sum(); P2[t - lo] = p2
        if learn:
            for k, l, sv in act:
                g = sv[y] - p @ sv; Ta[k, l] += g * g
                if g != 0 or rule != 2: T[k, l] += step(Tu[k, l], Ta[k, l]) * g
                Tu[k, l] += 1
                if gate:
                    g2 = sv[y] - p2 @ sv
                    for j in range(J):
                        i = (j, k, l if GL > 1 else 0, xb(t - 1 - j)); Ga[i] += g2 * g2
                        if g2 != 0 or rule != 2: G[i] += step(Gu[i], Ga[i]) * g2
                        Gu[i] += 1
            for k, key in enumerate(keys):
                if key is None: continue
                if key not in cells:
                    if birth > 0 and p[y] >= birth: continue
                    cells[key] = np.zeros(S)
                cells[key][y] += 1
            if any(k[0] == 2 for k in kinds) and t + 1 >= M: places.setdefault(tuple(int(v) for v in x[t + 1 - M:t + 1]), []).append(t + 1)
    return P1, P2, state

def check():
    import make_streams as MS
    ok = True; rng = np.random.default_rng(3)
    cases = [("shakespeare", "runs", dict()), ("py", "runs", dict(gate=1)), ("xml", "runs", dict(gate=2, birth=0.5)), ("c", "runs+since+pool", dict(gate=1)), ("table", "runs", dict(gate=1, recall=1)), ("py", "runs", dict(gate=1, recall=3, M=4)),
             ("book", "runs", dict(rule=2, eta=0.05, gate=1)), ("book", "runs", dict(rule=1, eta=0.1, e=2.0, gate=2)), ("random", "sums", dict(rule=1, eta=0.1, e=2.0, gate=1, J=8, lo=9)), ("repeats", "runs", dict(gate=1, recall=3, M=8))]
    for name, spec, kw in cases:
        kw = dict(kw); n = kw.pop("n", 2500); lo = kw.pop("lo", 6); recall = kw.pop("recall", 0)
        if name == "random": x = rng.integers(0, 14, n).astype(np.int32); S = 14
        elif name == "repeats": b = rng.integers(0, 5, 300); x = np.concatenate([b[rng.integers(0, 250):][:rng.integers(5, 60)] for _ in range(200)])[:n].astype(np.int32); S = 5
        else: x, cut, S = MS.load(name); x = x[:n]
        kinds = kinds_of(spec, S, recall); cutn = n - 500
        s = Sys(S, kinds, x, **kw); a1, a2, A1, A2 = s.run(lo, cutn, True, True); f1, f2, F1, F2 = s.run(cutn, n, False, True); Tf = s.T(); g1, g2, G1, G2 = s.run(cutn, n, True, True)
        R1, R2, st = ref_run(S, kinds, x, lo, cutn, None, True, **kw); Q1, Q2, _ = ref_run(S, kinds, x, cutn, n, st, False, **kw); Tr = st["T"].copy(); W1, W2, _ = ref_run(S, kinds, x, cutn, n, st, True, **kw)
        d = max(np.abs(A1 - R1).max(), np.abs(F1 - Q1).max(), np.abs(G1 - W1).max(), np.abs(Tf - Tr).max(), np.abs(s.T() - st["T"]).max())
        if kw.get("gate"): d = max(d, np.abs(A2 - R2).max(), np.abs(F2 - Q2).max(), np.abs(G2 - W2).max(), np.abs(s.G() - st["G"].ravel()).max())
        dn = np.abs(a1 + np.log(A1[np.arange(len(a1)), x[lo:cutn]])).max(); cells_ref = sum(1 for c in st["cells"].values() if c.sum() > 0); counts_ref = sum(int((c > 0).sum()) for c in st["cells"].values()); sz = s.sizes()
        used = int((st["Tu"][[i for i, k in enumerate(kinds) if k[0] == 2]] > 0).sum()) if recall else -1
        good = d < 1e-9 and dn < 1e-12 and sz["cells"] == cells_ref and sz["counts_above_zero"] == counts_ref and (not recall or used > 0); ok &= good
        print(f"{name} ({n} bytes, S = {S}), kinds {spec} ({len(kinds)}), options {kw}, recall {recall}: compiled against the reference, largest difference {d:.1e} (answers while reading, with nothing changing, going on; trusts; parts); cells {sz['cells']} = {cells_ref}, counts {sz['counts_above_zero']} = {counts_ref}" + (f"; recall levels used {used}" if recall else ""), "PASSED" if good else "FAILED", flush=True)
    return ok

def check_gate():
    """With 65 values on tiny Shakespeare the base is nano/table/gate.py's system (copied to ext/ with room for 128 values): the same answers on a prefix, and the same figures on the last tenth."""
    sys.path.insert(0, os.path.join(HERE, "ext")); import gate as GT, merge as G, make_streams as MS
    x, cut, S = MS.load("shakespeare"); ok = True
    for gate in (2, 1):                                           # gate.py's own second cell has the level (our gate=2); gate=1 has no counterpart there and is checked against the reference above
        n = 60000; GT.Sys2.GATE = 0; G.Sys = GT.Sys2; s0 = G.prose_sys(x[:n], dict(mode="none")); s0.learn(G.ctx_of(x, 6, n - 5000, 6), x[6:n - 5000]); s0.cell = 1; A1 = s0.say(G.ctx_of(x, n - 5000, n, 6)); s0.cell = 2; A2 = s0.say(G.ctx_of(x, n - 5000, n, 6))
        s = Sys(S, kinds_of("runs", S), x[:n], gate=gate); s.run(6, n - 5000, True); _, _, B1, B2 = s.run(n - 5000, n, False, True); d1 = np.abs(A1 - B1).max(); d2 = np.abs(A2 - B2).max() if gate == 2 else float("nan")
        good = d1 < 1e-9 and (gate != 2 or d2 < 1e-9); ok &= good; print(f"against nano/table/gate.py (ext/ copy) on a prefix of {n} bytes, last 5,000 answered with nothing changing: first cell differs by at most {d1:.1e}; second cell (with the level) {d2:.1e}", "PASSED" if good else "FAILED", flush=True)
    s = Sys(S, kinds_of("runs", S), x, gate=2); s.run(6, cut, True); f1, f2 = s.run(cut, len(x), False); sz = s.sizes(); good = abs(f1.mean() - 1.5226) < 5e-5; ok &= good
    print(f"the whole stream, last tenth with nothing changing: first cell {f1.mean():.4f} (gate.py: 1.5226), second cell with the level {f2.mean():.4f}; cells {sz['cells']}, counts above zero {sz['counts_above_zero']}", "PASSED" if good else "FAILED", flush=True); return ok

# ---------------------------------------------------------------- a run on a stream
def run_stream(kw):
    import make_streams as MS
    name = kw["stream"]; x, cut, S = MS.load(name); t0 = time.process_time(); recall = int(kw.get("recall", 0)); kinds = kinds_of(kw.get("kinds", "runs"), S, recall, float(kw.get("born", 0.25)))
    s = Sys(S, kinds, x, eta=float(kw.get("eta", 0.00005)), e=float(kw.get("e", 0.02)), rule=int(kw.get("rule", 0)), gate=int(kw.get("gate", 0)), J=int(kw.get("J", 6)), birth=float(kw.get("birth", 0)), M=int(kw.get("M", 8)))
    lo = 6; a1, a2 = s.run(lo, cut, True); t1 = time.process_time(); f1, f2 = s.run(cut, len(x), False); sz = s.sizes(); g1, g2 = s.run(cut, len(x), True); m = lambda v: None if v is None else float(v.mean())
    half = (cut - lo) // 2; out = dict(stream=name, args=kw, S=S, length=len(x), cut=cut, kinds=len(kinds), frozen=dict(cell1=m(f1), cell2=m(f2)), going_on=dict(cell1=m(g1), cell2=m(g2)), while_reading=dict(cell1=m(a1), cell2=m(a2)),
               while_reading_second_half=dict(cell1=m(a1[half:]), cell2=m(a2[half:]) if a2 is not None else None), sizes_at_the_frozen_scoring=sz, sizes_at_the_end=s.sizes(), cpu=round(time.process_time() - t0, 1), cpu_reading=round(t1 - t0, 1))
    if recall:
        Tu = s.Tu(); T = s.T(); out["recall_trust_by_level"] = [[round(float(T[i, l]), 3) if Tu[i, l] > 0 else None for l in range(NL)] for i, k in enumerate(kinds) if k[0] == 2]; out["recall_uses_by_level"] = [[int(Tu[i, l]) for l in range(NL)] for i, k in enumerate(kinds) if k[0] == 2]
    return out

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    if sys.argv[1] == "check_gate": sys.exit(0 if check_gate() else 1)
    kw = dict(a.split("=") for a in sys.argv[2:]); out = run_stream(kw); tag = kw.pop("tag", None) or "_".join(f"{k}{v}" for k, v in sorted(kw.items()) if k != "stream"); os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "res", f"{out['stream']}__{tag or 'base'}.json"), "w")); r = lambda v: None if v is None else round(v, 4)
    print(out["stream"], tag or "base", "| nothing changing: first cell", r(out["frozen"]["cell1"]), "second", r(out["frozen"]["cell2"]), "| going on:", r(out["going_on"]["cell1"]), r(out["going_on"]["cell2"]),
          "| cells", out["sizes_at_the_frozen_scoring"]["cells"], "counts", out["sizes_at_the_frozen_scoring"]["counts_above_zero"], "parts used", out["sizes_at_the_frozen_scoring"]["parts_used"], "| cpu", out["cpu"], flush=True)
