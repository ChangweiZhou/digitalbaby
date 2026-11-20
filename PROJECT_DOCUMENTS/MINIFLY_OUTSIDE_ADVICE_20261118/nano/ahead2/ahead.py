#!/usr/bin/env python3
"""PLAN_1.md of nano/ahead2: cells that count, keyed on a few delays, read with a learned trust, with look-ahead guesses.
Written from the prose of nano/sparse (PLAN_1, 2, 5, 10) and not from that line's programs.
The text, the split and the form of the score are those of loop_small_case/cells2.py (symbols, text(), the permutations).

    python3 ahead.py check                         the compiled kernel against the plain Python/numpy reference on prefixes
    python3 ahead.py run label seed=1 [look=1] [plan=add:40000,sub:20000] [every=5000] [lag=0] [levown=1] [t23=0.02] [mg=3]
        look=0: plain cells, first counts only.  lag=10: cells keyed on a guess are keyed on the guess made ten bytes earlier (control).
        levown=0: the level of a cell's second and third counts is that of its first counts (other choice C1).
        ncap=N (PLAN_3.md): a set of counts with total above N says its proportions at the weight of N counts.
        t23=0: the trusts of the plain cells' second and third counts are born at nothing (other choice C2).
        mg: the largest number of delays in a kind keyed on a guess (3: the plan's prose; 4: other choice C3).
"""
import ctypes, itertools, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = list(b"0123456789+=\n-"); S = len(SYMS); SI = {c: i for i, c in enumerate(SYMS)}; END = SI[10]; NL = 40; MAXK = 256

def text(op, a, b):                                   # as cells2.py
    if op == "-" and a < b: a, b = b, a
    return [SI[c] for c in f"{a:02d}{op}{b:02d}={(a + b if op == '+' else a - b):03d}\n".encode()]

def split(seed):                                      # as cells2.py: 500 scored, 2,000 never shown, 8,000 shown, for each operator
    rng = np.random.default_rng([seed, 99]); held, shown = {}, {}
    for op in "+-":
        o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = P[:500]; shown[op] = P[2000:]
    return rng, held, shown

def kinds(D, m, with_none=False):
    """Every set of at most m of the delays 1..D (and the empty set if with_none)."""
    out = [()] if with_none else []
    for r in range(1, m + 1): out += list(itertools.combinations(range(1, D + 1), r))
    return out

def level(n, u):
    bn = 0; n = int(n)
    while n > 1 and bn < 9: n >>= 1; bn += 1
    return bn * 4 + (0 if u <= 1 else 1 if u <= 2 else 2 if u <= 4 else 3)

def birth(tp, t23, tg):
    T = np.zeros((6, MAXK, NL)); T[0] = tp; T[1] = t23; T[2] = t23; T[3:] = tg; return T

# ------------------------------------------------------------------ the reference: plain Python, cells in dictionaries
class Ref:
    """Families: 0 plain first counts, 1 plain second, 2 plain third, 3 A first, 4 A second, 5 B."""
    def __init__(s, D=9, m=4, mg=3, look=1, levown=1, lag=0, eta=0.1, e=2.0, tp=0.02, t23=0.02, tg=0.0, ncap=0):
        s.D, s.look, s.levown, s.lag, s.eta, s.e = D, look, levown, lag, eta, e; s.ncap = ncap
        s.pk = kinds(D, m); s.gk = kinds(D, mg, True); s.T = birth(tp, t23, tg); s.U = np.zeros_like(s.T)
        s.cells = {}; s.h = [END] * (D + 1); s.past = []; s.gs = []
    def cell(s, fam, k, key, sets):
        return s.cells.setdefault((fam, k, key), np.zeros((sets, S)))
    def say(s, c, f, k, a, used, cl=None):
        n = c.sum()
        if n < 1: return
        if cl is None: cl = c
        lv = level(cl.sum(), (cl > 0).sum())
        if s.ncap and n > s.ncap: sv = np.log(c * s.ncap / n + s.e) - np.log(s.ncap + S * s.e)      # PLAN_3.md: the proportions at the weight of ncap counts
        else: sv = np.log(c + s.e) - np.log(n + S * s.e)
        a += s.T[f, k, lv] * sv; used.append((f, k, lv, sv))
    @staticmethod
    def share(a):
        p = np.exp(a - a.max()); return p / p.sum()
    def form(s, h, kg3=-1, kg2=-1):
        r = dict(used=[[], [], []], p=[None, None, None]); P = [s.cell("P", k, tuple(h[d] for d in ds), 3) for k, ds in enumerate(s.pk)]; r["P"] = P
        if s.look:
            a = np.zeros(S)
            for k, c in enumerate(P): s.say(c[2], 2, k, a, r["used"][2], None if s.levown else c[0])
            r["g3"] = int(a.argmax()); r["p"][2] = s.share(a); k3 = r["g3"] if kg3 < 0 else kg3
            A = [s.cell("A", j, tuple(h[d] for d in ds) + (k3,), 2) for j, ds in enumerate(s.gk)]; r["A"] = A; a = np.zeros(S)
            for k, c in enumerate(P): s.say(c[1], 1, k, a, r["used"][1], None if s.levown else c[0])
            for j, c in enumerate(A): s.say(c[1], 4, j, a, r["used"][1], None if s.levown else c[0])
            r["g2"] = int(a.argmax()); r["p"][1] = s.share(a); k2 = r["g2"] if kg2 < 0 else kg2
            B = [s.cell("B", j, tuple(h[d] for d in ds) + (k2,), 1) for j, ds in enumerate(s.gk)]; r["B"] = B
        a = np.zeros(S)
        for k, c in enumerate(P): s.say(c[0], 0, k, a, r["used"][0])
        if s.look:
            for j, c in enumerate(A): s.say(c[0], 3, j, a, r["used"][0])
            for j, c in enumerate(B): s.say(c[0], 5, j, a, r["used"][0])
        r["ans"] = int(a.argmax()); r["p"][0] = s.share(a); return r
    def teach(s, r, w, y):
        for f, k, lv, sv in r["used"][w]:
            s.T[f, k, lv] += s.eta / np.sqrt(1.0 + s.U[f, k, lv]) * (sv[y] - r["p"][w] @ sv); s.U[f, k, lv] += 1
    def step(s, y):
        t = len(s.gs); kg = s.gs[t - s.lag] if s.lag and t >= s.lag else (-1, -1)
        r = s.form(s.h, *kg); s.gs.append((r.get("g3", -1), r.get("g2", -1))); r1 = s.past[-1] if len(s.past) >= 1 else None; r2 = s.past[-2] if len(s.past) >= 2 else None
        s.teach(r, 0, y)
        if s.look:
            if r1: s.teach(r1, 1, y)
            if r2: s.teach(r2, 2, y)
        for c in r["P"]: c[0, y] += 1
        if s.look:
            for c in r["A"]: c[0, y] += 1
            for c in r["B"]: c[0, y] += 1
            if r1:
                for c in r1["P"]: c[1, y] += 1
                for c in r1["A"]: c[1, y] += 1
            if r2:
                for c in r2["P"]: c[2, y] += 1
        s.past = (s.past + [r])[-2:]; s.h = [0, y] + s.h[1:s.D]; return r
    def sizes(s):
        cells = sum(1 for c in s.cells.values() if c[0].sum() > 0); above = int(sum((c > 0).sum() for c in s.cells.values())); return cells, above, int((s.U > 0).sum())

# ------------------------------------------------------------------ the compiled kernel (the same arithmetic; cells addressed directly)
SRC = r'''
#include <math.h>
#include <stdint.h>
#include <string.h>
#define S 14
#define NL 40
#define MAXK 256
typedef struct { int32_t D, nP, nG, look, levown, lag; double eta, e; int32_t *P, *A, *B; double *T, *U;
                 const int32_t *pd, *pn, *gd, *gn; const int64_t *pbase, *gbase; double ncap; } St;
typedef struct { int64_t pid[MAXK], aid[MAXK], bid[MAXK]; int lev[6][MAXK]; double s[6][MAXK][S]; double p[3][S]; int g3, g2, ans; } Rec;
static Rec ring[3], scratch; static int hist[16]; static int64_t tt; static int gq3[64], gq2[64];
static double lt[8192], ln_[65536], cur_e = -1;
static void tables(double e) { if (e == cur_e) return; cur_e = e; for (int c = 0; c < 8192; c++) lt[c] = log(c + e); for (int n = 0; n < 65536; n++) ln_[n] = log(n + S * e); }
static inline int level(int64_t n, int u) { int bn = 0; while (n > 1 && bn < 9) { n >>= 1; bn++; } return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3); }
static inline void say(const St *st, const int32_t *c, const int32_t *cl, int f, int k, Rec *r, double *a) {
    int64_t n = 0; int u = 0;
    for (int v = 0; v < S; v++) { n += c[v]; u += c[v] > 0; }
    if (n < 1) { r->lev[f][k] = -1; return; }
    int64_t nl = n; int ul = u;
    if (cl) { nl = 0; ul = 0; for (int v = 0; v < S; v++) { nl += cl[v]; ul += cl[v] > 0; } }
    const int lv = level(nl, ul); r->lev[f][k] = lv;
    const double tk = st->T[((int64_t)f * MAXK + k) * NL + lv], den = n < 65536 ? ln_[n] : log(n + S * st->e); double *s = r->s[f][k];
    if (st->ncap > 0 && n > st->ncap) {      /* PLAN_3.md: the proportions at the weight of ncap counts */
        const double den2 = log(st->ncap + S * st->e), zero = log(st->e) - den2;
        for (int v = 0; v < S; v++) { const double sv = c[v] ? log(c[v] * st->ncap / n + st->e) - den2 : zero; s[v] = sv; a[v] += tk * sv; }
        return; }
    for (int v = 0; v < S; v++) { const double sv = (c[v] < 8192 ? lt[c[v]] : log(c[v] + st->e)) - den; s[v] = sv; a[v] += tk * sv; }
}
static inline int share(const double *a, double *p) {
    int g = 0; double z = 0; for (int v = 1; v < S; v++) if (a[v] > a[g]) g = v;
    for (int v = 0; v < S; v++) { p[v] = exp(a[v] - a[g]); z += p[v]; }
    for (int v = 0; v < S; v++) p[v] /= z;
    return g;
}
static void form(const St *st, const int *h, Rec *r, int kg3, int kg2) {
    double a[S]; int64_t gkey[MAXK];
    for (int k = 0; k < st->nP; k++) { int64_t idx = 0, mul = 1; for (int i = 0; i < st->pn[k]; i++) { idx += h[st->pd[k * 4 + i]] * mul; mul *= S; } r->pid[k] = st->pbase[k] + idx; }
    if (st->look) {
        for (int j = 0; j < st->nG; j++) { int64_t idx = 0, mul = 1; for (int i = 0; i < st->gn[j]; i++) { idx += h[st->gd[j * 4 + i]] * mul; mul *= S; } gkey[j] = st->gbase[j] + idx * S; }
        memset(a, 0, sizeof a);
        for (int k = 0; k < st->nP; k++) { const int32_t *c = st->P + r->pid[k] * 42; say(st, c + 28, st->levown ? 0 : c, 2, k, r, a); }
        r->g3 = share(a, r->p[2]); const int k3 = kg3 >= 0 ? kg3 : r->g3;
        memset(a, 0, sizeof a);
        for (int k = 0; k < st->nP; k++) { const int32_t *c = st->P + r->pid[k] * 42; say(st, c + 14, st->levown ? 0 : c, 1, k, r, a); }
        for (int j = 0; j < st->nG; j++) { r->aid[j] = gkey[j] + k3; const int32_t *c = st->A + r->aid[j] * 28; say(st, c + 14, st->levown ? 0 : c, 4, j, r, a); }
        r->g2 = share(a, r->p[1]); const int k2 = kg2 >= 0 ? kg2 : r->g2;
        for (int j = 0; j < st->nG; j++) r->bid[j] = gkey[j] + k2;
    } else { r->g3 = -1; r->g2 = -1; }
    memset(a, 0, sizeof a);
    for (int k = 0; k < st->nP; k++) say(st, st->P + r->pid[k] * 42, 0, 0, k, r, a);
    if (st->look) {
        for (int j = 0; j < st->nG; j++) say(st, st->A + r->aid[j] * 28, 0, 3, j, r, a);
        for (int j = 0; j < st->nG; j++) say(st, st->B + r->bid[j] * 14, 0, 5, j, r, a);
    }
    r->ans = share(a, r->p[0]);
}
static inline void teach(const St *st, const Rec *r, int f, int nk, const double *p, int y) {
    for (int k = 0; k < nk; k++) { const int lv = r->lev[f][k]; if (lv < 0) continue;
        const double *s = r->s[f][k]; double g = s[y]; for (int v = 0; v < S; v++) g -= p[v] * s[v];
        const int64_t i = ((int64_t)f * MAXK + k) * NL + lv; st->T[i] += st->eta / sqrt(1.0 + st->U[i]) * g; st->U[i] += 1.0; }
}
static void arrive(const St *st, int y, const Rec *r0, const Rec *r1, const Rec *r2) {
    teach(st, r0, 0, st->nP, r0->p[0], y);
    if (st->look) {
        teach(st, r0, 3, st->nG, r0->p[0], y); teach(st, r0, 5, st->nG, r0->p[0], y);
        if (r1) { teach(st, r1, 1, st->nP, r1->p[1], y); teach(st, r1, 4, st->nG, r1->p[1], y); }
        if (r2) teach(st, r2, 2, st->nP, r2->p[2], y);
    }
    for (int k = 0; k < st->nP; k++) st->P[r0->pid[k] * 42 + y]++;
    if (st->look) {
        for (int j = 0; j < st->nG; j++) { st->A[r0->aid[j] * 28 + y]++; st->B[r0->bid[j] * 14 + y]++; }
        if (r1) { for (int k = 0; k < st->nP; k++) st->P[r1->pid[k] * 42 + 14 + y]++; for (int j = 0; j < st->nG; j++) st->A[r1->aid[j] * 28 + 14 + y]++; }
        if (r2) for (int k = 0; k < st->nP; k++) st->P[r2->pid[k] * 42 + 28 + y]++;
    }
}
void reset(int end) { tt = 0; for (int d = 0; d < 16; d++) hist[d] = end; }
/* read n bytes of the stream: before each, the guesses and the answer; at its arrival, the trusts and the counts */
void run(const St *st, const int32_t *x, int64_t n, double *outp, int32_t *outg) {
    tables(st->e);
    for (int64_t i = 0; i < n; i++, tt++) {
        Rec *r0 = &ring[tt % 3], *r1 = tt >= 1 ? &ring[(tt + 2) % 3] : 0, *r2 = tt >= 2 ? &ring[(tt + 1) % 3] : 0; int kg3 = -1, kg2 = -1;
        if (st->lag > 0 && tt >= st->lag) { kg3 = gq3[(tt - st->lag) % 64]; kg2 = gq2[(tt - st->lag) % 64]; }
        form(st, hist, r0, kg3, kg2); gq3[tt % 64] = r0->g3; gq2[tt % 64] = r0->g2;
        if (outp) { memcpy(outp + i * 3 * S, r0->p, sizeof r0->p); outg[i * 3] = r0->ans; outg[i * 3 + 1] = r0->g2; outg[i * 3 + 2] = r0->g3; }
        arrive(st, x[i], r0, r1, r2);
        for (int d = st->D; d > 1; d--) hist[d] = hist[d - 1];
        hist[1] = x[i];
    }
}
/* the answer and the guesses for a given history (h[1..D]); nothing changes */
void answer(const St *st, const int32_t *h, int kg3, int kg2, int32_t *out) {
    int hh[16]; tables(st->e); for (int d = 0; d <= st->D; d++) hh[d] = h[d];
    form(st, hh, &scratch, kg3, kg2); out[0] = scratch.ans; out[1] = scratch.g2; out[2] = scratch.g3;
}
void sizes(const St *st, int64_t nP, int64_t nA, int64_t *out) {
    int64_t cells = 0, above = 0;
    for (int64_t i = 0; i < nP; i++) { const int32_t *c = st->P + i * 42; int64_t n = 0; for (int v = 0; v < 14; v++) n += c[v]; cells += n > 0; for (int v = 0; v < 42; v++) above += c[v] > 0; }
    if (st->look) for (int64_t i = 0; i < nA; i++) {
        const int32_t *c = st->A + i * 28; int64_t n = 0; for (int v = 0; v < 14; v++) n += c[v]; cells += n > 0; for (int v = 0; v < 28; v++) above += c[v] > 0;
        c = st->B + i * 14; n = 0; for (int v = 0; v < 14; v++) { n += c[v]; above += c[v] > 0; } cells += n > 0; }
    out[0] = cells; out[1] = above;
}
'''
PI32 = ctypes.POINTER(ctypes.c_int32); PI64 = ctypes.POINTER(ctypes.c_int64); PD = ctypes.POINTER(ctypes.c_double)
class St(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int32) for n in ("D", "nP", "nG", "look", "levown", "lag")] + [("eta", ctypes.c_double), ("e", ctypes.c_double)] + \
               [(n, PI32) for n in ("P", "A", "B")] + [(n, PD) for n in ("T", "U")] + [(n, PI32) for n in ("pd", "pn", "gd", "gn")] + [(n, PI64) for n in ("pbase", "gbase")] + [("ncap", ctypes.c_double)]

def kern():
    so = os.path.join(HERE, "kern_a.so"); src = os.path.join(HERE, "kern_a.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); L.run.argtypes = [ctypes.POINTER(St), PI32, ctypes.c_int64, PD, PI32]; L.answer.argtypes = [ctypes.POINTER(St), PI32, ctypes.c_int, ctypes.c_int, PI32]
    L.sizes.argtypes = [ctypes.POINTER(St), ctypes.c_int64, ctypes.c_int64, PI64]; L.reset.argtypes = [ctypes.c_int]
    for f in (L.run, L.answer, L.sizes, L.reset): f.restype = None
    return L

class Fast:
    def __init__(s, D=9, m=4, mg=3, look=1, levown=1, lag=0, eta=0.1, e=2.0, tp=0.02, t23=0.02, tg=0.0, ncap=0):
        s.D = D; s.pk = kinds(D, m); s.gk = kinds(D, mg, True); s.L = kern(); s.look = look; assert len(s.pk) <= MAXK and len(s.gk) <= MAXK
        def lay(ks):
            d = np.zeros((len(ks), 4), np.int32); n = np.zeros(len(ks), np.int32); base = np.zeros(len(ks), np.int64); tot = 0
            for k, ds in enumerate(ks): d[k, :len(ds)] = ds; n[k] = len(ds); base[k] = tot; tot += S ** len(ds)
            return d, n, base, tot
        s.pd, s.pn, s.pbase, s.nPc = lay(s.pk); s.gd, s.gn, s.gbase, nk = lay(s.gk); s.gbase *= S; s.nGc = nk * S
        s.P = np.zeros(s.nPc * 42, np.int32); s.A = np.zeros(s.nGc * 28 if look else 1, np.int32); s.B = np.zeros(s.nGc * 14 if look else 1, np.int32)
        s.T = birth(tp, t23, tg); s.U = np.zeros_like(s.T); p = lambda a, t: a.ctypes.data_as(t)
        s.st = St(D, len(s.pk), len(s.gk), look, levown, lag, eta, e, p(s.P, PI32), p(s.A, PI32), p(s.B, PI32), p(s.T, PD), p(s.U, PD),
                  p(s.pd, PI32), p(s.pn, PI32), p(s.gd, PI32), p(s.gn, PI32), p(s.pbase, PI64), p(s.gbase, PI64), float(ncap)); s.L.reset(END)
    def run(s, x, want=False):
        x = np.ascontiguousarray(x, np.int32); P = np.zeros((len(x), 3, S)) if want else None; G = np.zeros((len(x), 3), np.int32) if want else None
        s.L.run(ctypes.byref(s.st), x.ctypes.data_as(PI32), len(x), P.ctypes.data_as(PD) if want else None, G.ctypes.data_as(PI32) if want else None); return P, G
    def answer(s, h, kg3=-1, kg2=-1):
        hh = np.zeros(16, np.int32); hh[:len(h)] = h; out = np.zeros(3, np.int32); s.L.answer(ctypes.byref(s.st), hh.ctypes.data_as(PI32), kg3, kg2, out.ctypes.data_as(PI32)); return int(out[0]), int(out[1]), int(out[2])
    def sizes(s):
        out = np.zeros(2, np.int64); s.L.sizes(ctypes.byref(s.st), s.nPc, s.nGc, out.ctypes.data_as(PI64)); return int(out[0]), int(out[1]), int((s.U > 0).sum())

# ------------------------------------------------------------------ the score: the system writes its own answer bytes; nothing changes
def score(sys_, op, pairs, D=9, other=False):
    """Before the line stand line ends only. Returns whole %, by place %, and how often the guesses are right.
    other=True (the control): the cells keyed on a guess are keyed on the guess made at the same place of the preceding scored pair."""
    Ls = [text(op, a, b) for a, b in pairs]; whole = 0; place = np.zeros(3); gr = np.zeros(3)      # gr: g3 = units and g2 = tens at the first answer byte; g2 = units at the second
    def one(L, keys):
        h = [0] + [END] * D; res = []
        for c in L[:6]: h = [0, c] + h[1:D]
        for i in range(3):
            ans, g2, g3 = sys_.answer(h, *(keys[i] if keys else (-1, -1))); res.append((ans, g2, g3)); h = [0, ans] + h[1:D]
        return res
    prev = one(Ls[-1], None) if other else None
    for L in Ls:
        res = one(L, [(r[2], r[1]) for r in prev] if other else None); ok = [res[i][0] == L[6 + i] for i in range(3)]; whole += all(ok); place += ok
        gr += [res[0][2] == L[8], res[0][1] == L[7], res[1][1] == L[8]]
        if other: prev = res
    n = len(pairs); return 100 * whole / n, (100 * place / n).round(1).tolist(), (100 * gr / n).round(1).tolist()

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng, held, shown = split(seed); look = int(kw.get("look", 1)); lag = int(kw.get("lag", 0)); every = int(kw.get("every", 5000))
    s = Fast(mg=int(kw.get("mg", 3)), look=look, levown=int(kw.get("levown", 1)), lag=lag, t23=float(kw.get("t23", 0.02)), ncap=int(kw.get("ncap", 0)))
    TAB = {op: np.array([text(op, a, b) for a, b in shown[op]], np.int32) for op in "+-"}
    lines_shown = {tuple(L) for L in TAB["-"].tolist()}; clean = [p for p in held["-"] if tuple(text("-", *p)) not in lines_shown]      # held-out subtractions whose line no shown pair writes
    assert not {tuple(text("+", *p)) for p in held["+"]} & {tuple(L) for L in TAB["+"].tolist()}
    out = dict(label=kw.get("label"), args=kw, rows=[], clean_pairs=len(clean))
    for pi, ph in enumerate(kw.get("plan", "add:40000,sub:20000").split(",")):
        name, n = ph.split(":"); op = {"add": "+", "sub": "-"}[name]; n = int(n); assert n % every == 0
        for b in range(n // every):
            s.run(TAB[op][rng.integers(8000, size=every)].ravel())
            a, ap, ag = score(s, "+", held["+"], other=bool(lag)); row = dict(phase=pi, name=name, lines=(b + 1) * every, add=a, add_place=ap, add_guess=ag)
            if pi > 0 or b == n // every - 1:
                row["sub_all"], row["sub_all_place"], _ = score(s, "-", held["-"], other=bool(lag)); row["sub_clean"], row["sub_clean_place"], row["sub_guess"] = score(s, "-", clean, other=bool(lag))
            out["rows"].append(row)
        c, ab, tu = s.sizes(); out.setdefault("sizes", []).append(dict(phase=pi, cells=c, counts_above_zero=ab, trusts_used=tu, cpu=round(time.process_time() - t0, 1)))
    r0 = [r["add"] for r in out["rows"] if r["phase"] == 0]; out["last3_add_phase0"] = float(np.mean(r0[-3:])); out["last_add_phase0"] = r0[-1]
    out["last_add_end"] = out["rows"][-1]["add"]; out["cpu"] = round(time.process_time() - t0, 1)
    fam = ["plain first", "plain second", "plain third", "A first", "A second", "B"]; ks = [s.pk, s.pk, s.pk, s.gk, s.gk, s.gk]
    out["trust_abs_sum_by_family"] = {fam[f]: float(np.abs(s.T[f] * (s.U[f] > 0)).sum()) for f in range(6)}
    out["top_kinds_by_family"] = {fam[f]: [(list(ks[f][k]), round(float(v), 3)) for k, v in sorted(enumerate((s.T[f] * (s.U[f] > 0)).max(1)[:len(ks[f])]), key=lambda q: -q[1])[:6]] for f in range(6)}
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"{kw['label']}.json"), "w")); return out

def check():
    ok_all = True
    for kw, nb in ((dict(D=5, m=2, mg=1), 1500), (dict(D=6, m=3, mg=2, levown=0), 1500), (dict(D=6, m=2, mg=2, lag=10), 1500), (dict(D=5, m=2, mg=1, look=0), 1500),
                   (dict(D=6, m=2, mg=1, t23=0.0, tg=0.05, eta=0.03, e=0.5), 1500), (dict(D=5, m=2, mg=1, ncap=4), 3000), (dict(D=6, m=3, mg=2, ncap=16), 3000), (dict(D=9, m=4, mg=3), 400), (dict(D=9, m=4, mg=3, lag=10, levown=0), 300)):
        rng, held, shown = split(7); x = np.concatenate([text("+-"[i % 7 == 3], *shown["+"][int(j)]) for i, j in enumerate(rng.integers(8000, size=nb // 10))]).astype(np.int32)
        f = Fast(**kw); P, G = f.run(x[:nb // 2], True); P2, G2 = f.run(x[nb // 2:], True); P = np.concatenate([P, P2]); G = np.concatenate([G, G2])      # two calls: the stream carries on
        r = Ref(**kw); dp = 0.0; same = True; look = kw.get("look", 1)
        for t in range(nb):
            q = r.step(int(x[t]))
            for w in range(3 if look else 1): dp = max(dp, float(np.abs(q["p"][w] - P[t, w]).max()))
            same &= q["ans"] == G[t, 0] and (not look or (q["g2"] == G[t, 1] and q["g3"] == G[t, 2]))
        dT = float(np.abs(r.T - f.T).max()); sz = r.sizes() == f.sizes(); hs = [[0] + [int(v) for v in rng.integers(S, size=kw["D"])] for _ in range(50)]
        ans = all(f.answer(h, a, b) == (lambda q: (q["ans"], q.get("g2", -1), q.get("g3", -1)))(r.form(h, a, b)) for h in hs for a, b in ((-1, -1), (3, 7)))
        ok = dp < 1e-9 and dT < 1e-9 and same and sz and ans; ok_all &= ok
        print(f"{kw}: {nb} bytes, {len(r.pk)} plain kinds, {len(r.gk)} kinds keyed on each guess, sizes (cells, counts above zero, trusts used) {r.sizes()}: compiled against the reference: "
              f"shares {dp:.1e}, trusts {dT:.1e}; answers and guesses the same: {bool(same)}; sizes the same: {sz}; answers without learning the same: {ans}", "PASSED" if ok else "FAILED", flush=True)
    return ok_all

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=") for a in sys.argv[3:]); kw["label"] = sys.argv[2]; o = run(kw)
    print(o["label"], "last3", round(o["last3_add_phase0"], 1), "|", " ".join(f"{r['phase']}:{r['lines'] // 1000}k {r['add']:.1f}" + (f"/{r['sub_clean']:.0f}" if "sub_clean" in r else "") for r in o["rows"]),
          "| place", o["rows"][-1]["add_place"], "guess", o["rows"][-1]["add_guess"], "|", o["sizes"][-1], flush=True)
