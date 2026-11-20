#!/usr/bin/env python3
"""nano/unite: cells that count, with look-ahead, keyed on a chain that has a clock. One system out of two that were shown apart.

Taken from other lines and said so:
  nano/ahead2/ahead.py (commit 7765872): the cells that count (kinds, levels, shares, trusts, the look-ahead guesses and the cells keyed on them),
      here written for a look-ahead of L bytes (L=3 is that program's arithmetic) and for a chain that need not step at every byte.
  nano/skip/skip.py (commit 7016262): the trial clock in its second form (form=evid), the stream of the text g2 and t3, the letters used when scoring.
  loop_small_case/cells2.py (commit f3e632c): the split into pairs shown and never shown for each operator, the life add then sub.

The lines a cell may be keyed on: position 1 is the present byte (always); positions 2 to D are the chain: the last D-1 bytes that ENTERED it.
When a byte arrives, the byte that was present either enters the chain (which moves on by one) or is passed over (as in skip.py).
The look-ahead is in real bytes: a cell counts the bytes that arrive 1, 2, ... L bytes after its reading, whatever the clock does.

    python3 unite.py check
    python3 unite.py run <label> seed=1 [text=g2|t3] [clock=none|oracle|trial] [look=3] [plan=add:200000] [every=10000] [gap=8]
            [alpha=2 p0=0.2 P=256 win=3000 eps=0.01 beta=0.02] [freeze=N] [ncap=N] [past=5] [nosub=1] [m=4 mg=3 D=9]
        clock=trial: every P bytes each byte value's entry is closed by chance c = logistic(u) or open; u = logit(p0) + alpha * t, t the difference of
        the entry's two means of z (closed less open) in units of its uncertainty; z is the period's mean of the one number the answering cell sends
        out (the sum of its squared mismatches), against its running mean, in units of its running deviation (skip.py, form=evid).
        Scores are taken with every entry in its likelier state and no drawing; the system writes its own answer bytes.
"""
import ctypes, itertools, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NL = 40; MAXK = 256; MAXS = 18; MAXL = 4; MAXF = 10; LET = b"wxyz"

def setup(sub=True):
    """The byte values: digits + = end mark, '-' only where subtraction is read, and four letters."""
    global SYMS, S, SI, END, EQ, LETS
    SYMS = list(b"0123456789+=\n" + (b"-" if sub else b"") + LET); S = len(SYMS); SI = {c: i for i, c in enumerate(SYMS)}; END = SI[10]; EQ = SI[ord("=")]; LETS = [SI[c] for c in LET]
setup()

def fill(a, b, n): return bytes(LET[(7 * a + 3 * b + 5 * i * i + i) % 4] for i in range(n)).decode()          # the letters used when scoring: fixed by the pair (hold.py, skip.py)
def text(tx, op, a, b, f=""):
    if op == "-" and a < b: a, b = b, a                # as cells2.py
    c = a + b if op == "+" else a - b
    t = f"{a:02d}{op}{f}{b:02d}={c:03d}\n" if tx == "g2" else f"{a:03d}{op}{f}{b:03d}={c:04d}\n"
    return [SI[c] for c in t.encode()]

def split(tx, seed):
    """g2: as cells2.py, for each operator 500 pairs scored and never shown, 8,000 shown (the 500 of addition are those of hold.py and skip.py).
    t3: as skip.py, 500 pairs never shown, every other pair of numbers under 1,000 may be drawn."""
    rng = np.random.default_rng([seed, 99]); held, shown = {}, {}
    if tx == "t3":
        o = rng.choice(1000000, 500, replace=False); held["+"] = [(int(q) // 1000, int(q) % 1000) for q in o]; shown["+"] = np.sort(o); return rng, held, shown
    for op in "+-":
        o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = P[:500]; shown[op] = P[2000:]
    return rng, held, shown

def stream(tx, op, shown, G, n, rng):
    """n lines drawn from the pairs shown, with 0 to G letters drawn afresh for every line (skip.py's stream, with the operator given)."""
    if tx == "t3":
        a = rng.integers(1000, size=n); b = rng.integers(1000, size=n); bad = np.isin(1000 * a + b, shown)
        while bad.any(): a = np.where(bad, (a + 1) % 1000, a); bad = np.isin(1000 * a + b, shown)
    else:
        P = np.asarray(shown)[rng.integers(len(shown), size=n)]; a, b = P[:, 0], P[:, 1]
        if op == "-": a, b = np.maximum(a, b), np.minimum(a, b)
    k = rng.integers(G + 1, size=n) if G else np.zeros(n, np.int64)
    let = np.asarray(LETS)[rng.integers(len(LETS), size=(n, max(G, 1)))]; let = np.where(np.arange(max(G, 1))[None, :] < k[:, None], let, -1)
    col = lambda v: np.asarray(v)[:, None]; const = lambda c: np.full((n, 1), SI[ord(c)] if isinstance(c, str) else c); c = a + b if op == "+" else a - b
    if tx == "t3": M = [col(a // 100), col(a // 10 % 10), col(a % 10), const(op), let, col(b // 100), col(b // 10 % 10), col(b % 10), const("="), col(c // 1000), col(c // 100 % 10), col(c // 10 % 10), col(c % 10), const(END)]
    else: M = [col(a // 10), col(a % 10), const(op), let, col(b // 10), col(b % 10), const("="), col(c // 100), col(c // 10 % 10), col(c % 10), const(END)]
    M = np.concatenate(M, 1); return M[M >= 0].astype(np.int32)

def kinds(D, m, with_none=False):
    out = [()] if with_none else []
    for r in range(1, m + 1): out += list(itertools.combinations(range(1, D + 1), r))
    return out
def level(n, u):
    bn = 0; n = int(n)
    while n > 1 and bn < 9: n >>= 1; bn += 1
    return bn * 4 + (0 if u <= 1 else 1 if u <= 2 else 2 if u <= 4 else 3)
def fams(L):
    """Families of trusts. Plain cells' count w (the byte w+1 ahead): family w. Cells keyed on the guess j ahead (j = 2..L), their count w (w < j-1): after those."""
    f = {}; n = L
    for j in range(L, 1, -1):
        for w in range(j - 1): f[(j, w)] = n; n += 1
    return f, max(n, 1)
def birth(L, tp, tg):
    T = np.zeros((MAXF, MAXK, NL)); T[:max(L, 1)] = tp; T[max(L, 1):] = tg; return T
def sg(u): return 1 / (1 + np.exp(-u))

# ------------------------------------------------------------------ the reference: plain Python, cells in dictionaries
class Ref:
    def __init__(s, D=9, m=4, mg=3, look=3, eta=0.1, e=2.0, tp=0.02, tg=0.0, ncap=0.0):
        s.D, s.L, s.eta, s.e = D, look, eta, e; s.ncap = ncap; s.pk = kinds(D, m); s.gk = kinds(D, mg, True); s.fam, _ = fams(look); s.T = birth(look, tp, tg); s.U = np.zeros_like(s.T)
        s.cells = {}; s.h = [END] * (D + 1); s.past = []; s.pend = True; s.esum = 0.0
    def cell(s, fam, k, key, sets): return s.cells.setdefault((fam, k, key), np.zeros((sets, S)))
    def say(s, c, f, k, a, used):
        n = c.sum()
        if n < 1: return
        lv = level(n, (c > 0).sum()); sv = (np.log(c * (s.ncap / n) + s.e) - np.log(s.ncap + S * s.e)) if s.ncap > 0 and n > s.ncap else (np.log(c + s.e) - np.log(n + S * s.e)); a += s.T[f, k, lv] * sv; used.append((f, k, lv, sv))
    @staticmethod
    def share(a):
        p = np.exp(a - a.max()); return p / p.sum()
    def form(s, h):
        L = max(s.L, 1); r = dict(used=[[] for _ in range(L)], p=[None] * L, g={}, F={}); P = [s.cell("P", k, tuple(h[d] for d in ds), L) for k, ds in enumerate(s.pk)]; r["P"] = P
        for j in range(s.L, 1, -1):
            w = j - 1; a = np.zeros(S)
            for k, c in enumerate(P): s.say(c[w], w, k, a, r["used"][w])
            for i in range(s.L, j, -1):
                for q, c in enumerate(r["F"][i]): s.say(c[w], s.fam[(i, w)], q, a, r["used"][w])
            r["g"][j] = int(a.argmax()); r["p"][w] = s.share(a)
            r["F"][j] = [s.cell(("F", j), q, tuple(h[d] for d in ds) + (r["g"][j],), j - 1) for q, ds in enumerate(s.gk)]
        a = np.zeros(S)
        for k, c in enumerate(P): s.say(c[0], 0, k, a, r["used"][0])
        for i in range(s.L, 1, -1):
            for q, c in enumerate(r["F"][i]): s.say(c[0], s.fam[(i, 0)], q, a, r["used"][0])
        r["ans"] = int(a.argmax()); r["p"][0] = s.share(a); return r
    def step(s, y, enter=True):
        """The byte y arrives. enter: whether y will enter the chain when the next byte arrives."""
        r = s.form(s.h); recs = [r] + s.past[::-1]                     # recs[w]: the reading made w bytes ago
        m = -r["p"][0].copy(); m[y] += 1; s.esum += float(m @ m)
        for w, q in enumerate(recs):
            for f, k, lv, sv in q["used"][w]:
                s.T[f, k, lv] += s.eta / np.sqrt(1.0 + s.U[f, k, lv]) * (sv[y] - q["p"][w] @ sv); s.U[f, k, lv] += 1
        for w, q in enumerate(recs):
            for c in q["P"]: c[w, y] += 1
            for i in range(w + 2, s.L + 1):
                for c in q["F"][i]: c[w, y] += 1
        s.past = (s.past + [r])[-(max(s.L, 1) - 1):] if s.L > 1 else []
        s.h = ([0, y] + s.h[1:s.D]) if s.pend else ([0, y] + s.h[2:]); s.pend = bool(enter); return r
    def sizes(s):
        cells = sum(1 for c in s.cells.values() if c[0].sum() > 0); above = int(sum((c > 0).sum() for c in s.cells.values())); return cells, above, int((s.U > 0).sum())

# ------------------------------------------------------------------ the compiled kernel (the same arithmetic; cells addressed directly)
SRC = r'''
#include <math.h>
#include <stdint.h>
#include <string.h>
#define NL 40
#define MAXK 256
#define MAXS 18
#define MAXL 4
#define MAXF 10
typedef struct { int32_t S, D, nP, nG, L; double eta, e, ncap; int32_t *P; int32_t *F[MAXL + 1]; double *T, *U;
                 const int32_t *pd, *pn, *gd, *gn; const int64_t *pbase, *gbase; int32_t fam[MAXL + 1][MAXL]; } St;
typedef struct { int64_t pid[MAXK], fid[MAXL + 1][MAXK]; int lev[MAXF][MAXK]; double s[MAXF][MAXK][MAXS]; double p[MAXL][MAXS]; int g[MAXL + 1], ans; } Rec;
static Rec ring[MAXL], scratch; static int hist[16]; static int64_t tt; static int pend = 1;
static double lt[8192], ln_[65536], cur_e = -1; static int cur_S = -1;
static void tables(double e, int S) { if (e == cur_e && S == cur_S) return; cur_e = e; cur_S = S; for (int c = 0; c < 8192; c++) lt[c] = log(c + e); for (int n = 0; n < 65536; n++) ln_[n] = log(n + S * e); }
static inline int level(int64_t n, int u) { int bn = 0; while (n > 1 && bn < 9) { n >>= 1; bn++; } return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3); }
static inline void say(const St *st, const int32_t *c, int f, int k, Rec *r, double *a) {
    const int S = st->S; int64_t n = 0; int u = 0;
    for (int v = 0; v < S; v++) { n += c[v]; u += c[v] > 0; }
    if (n < 1) { r->lev[f][k] = -1; return; }
    const int lv = level(n, u); r->lev[f][k] = lv;
    const double tk = st->T[((int64_t)f * MAXK + k) * NL + lv]; double *s = r->s[f][k];
    if (st->ncap > 0 && n > st->ncap) {            /* a cell that has counted more than ncap speaks as if it had counted ncap, in the same proportions (nano/sparse, PLAN_5, addition 2) */
        const double sc = st->ncap / n, den = log(st->ncap + S * st->e);
        for (int v = 0; v < S; v++) { const double sv = log(c[v] * sc + st->e) - den; s[v] = sv; a[v] += tk * sv; }
        return;
    }
    const double den = n < 65536 ? ln_[n] : log(n + S * st->e);
    for (int v = 0; v < S; v++) { const double sv = (c[v] < 8192 ? lt[c[v]] : log(c[v] + st->e)) - den; s[v] = sv; a[v] += tk * sv; }
}
static inline int share(int S, const double *a, double *p) {
    int g = 0; double z = 0; for (int v = 1; v < S; v++) if (a[v] > a[g]) g = v;
    for (int v = 0; v < S; v++) { p[v] = exp(a[v] - a[g]); z += p[v]; }
    for (int v = 0; v < S; v++) p[v] /= z;
    return g;
}
static void form(const St *st, const int *h, Rec *r) {
    const int S = st->S, L = st->L, L1 = L > 1 ? L : 1; double a[MAXS]; int64_t gkey[MAXK];
    for (int k = 0; k < st->nP; k++) { int64_t idx = 0, mul = 1; for (int i = 0; i < st->pn[k]; i++) { idx += h[st->pd[k * 4 + i]] * mul; mul *= S; } r->pid[k] = st->pbase[k] + idx; }
    if (L > 1) for (int j = 0; j < st->nG; j++) { int64_t idx = 0, mul = 1; for (int i = 0; i < st->gn[j]; i++) { idx += h[st->gd[j * 4 + i]] * mul; mul *= S; } gkey[j] = st->gbase[j] + idx * S; }
    for (int j = L; j > 1; j--) {
        const int w = j - 1; memset(a, 0, sizeof a);
        for (int k = 0; k < st->nP; k++) say(st, st->P + r->pid[k] * L1 * S + w * S, w, k, r, a);
        for (int i = L; i > j; i--) for (int q = 0; q < st->nG; q++) say(st, st->F[i] + r->fid[i][q] * (i - 1) * S + w * S, st->fam[i][w], q, r, a);
        r->g[j] = share(S, a, r->p[w]);
        for (int q = 0; q < st->nG; q++) r->fid[j][q] = gkey[q] + r->g[j];
    }
    memset(a, 0, sizeof a);
    for (int k = 0; k < st->nP; k++) say(st, st->P + r->pid[k] * L1 * S, 0, k, r, a);
    for (int i = L; i > 1; i--) for (int q = 0; q < st->nG; q++) say(st, st->F[i] + r->fid[i][q] * (i - 1) * S, st->fam[i][0], q, r, a);
    r->ans = share(S, a, r->p[0]);
}
static inline void teach(const St *st, const Rec *r, int f, int nk, const double *p, int y) {
    const int S = st->S;
    for (int k = 0; k < nk; k++) { const int lv = r->lev[f][k]; if (lv < 0) continue;
        const double *s = r->s[f][k]; double g = s[y]; for (int v = 0; v < S; v++) g -= p[v] * s[v];
        const int64_t i = ((int64_t)f * MAXK + k) * NL + lv; st->T[i] += st->eta / sqrt(1.0 + st->U[i]) * g; st->U[i] += 1.0; }
}
void reset(int end) { tt = 0; pend = 1; for (int d = 0; d < 16; d++) hist[d] = end; }
/* read n bytes of the stream: before each, the guesses and the answer; at its arrival, the trusts and the counts; then the chain steps or does not.
   fl[i]: whether the byte x[i] enters the chain when the next byte arrives (0: all enter). Returns the sum of the answering cell's squared mismatches. */
double run(const St *st, const int32_t *x, const uint8_t *fl, int64_t n, double *outp, int32_t *outg) {
    const int S = st->S, L = st->L, L1 = L > 1 ? L : 1; double esum = 0; tables(st->e, S);
    for (int64_t i = 0; i < n; i++, tt++) {
        Rec *rec[MAXL]; const int nr = tt + 1 < L1 ? (int)(tt + 1) : L1; const int y = x[i];
        for (int w = 0; w < nr; w++) rec[w] = &ring[(tt - w) % L1];
        Rec *r0 = rec[0]; form(st, hist, r0);
        if (outp) { for (int w = 0; w < L1; w++) memcpy(outp + (i * L1 + w) * S, r0->p[w], S * sizeof(double)); outg[i * (MAXL + 1)] = r0->ans; for (int j = 2; j <= L; j++) outg[i * (MAXL + 1) + j] = r0->g[j]; }
        for (int v = 0; v < S; v++) { const double m = (v == y) - r0->p[0][v]; esum += m * m; }
        for (int w = 0; w < nr; w++) { teach(st, rec[w], w, st->nP, rec[w]->p[w], y); for (int j = w + 2; j <= L; j++) teach(st, rec[w], st->fam[j][w], st->nG, rec[w]->p[w], y); }
        for (int w = 0; w < nr; w++) {
            for (int k = 0; k < st->nP; k++) st->P[rec[w]->pid[k] * L1 * S + w * S + y]++;
            for (int j = w + 2; j <= L; j++) for (int q = 0; q < st->nG; q++) st->F[j][rec[w]->fid[j][q] * (j - 1) * S + w * S + y]++;
        }
        if (pend) for (int d = st->D; d > 1; d--) hist[d] = hist[d - 1];
        hist[1] = y; pend = fl ? fl[i] : 1;
    }
    return esum;
}
/* the answer and the guesses for a given history (h[1..D]); nothing changes */
void answer(const St *st, const int32_t *h, int32_t *out) {
    int hh[16]; tables(st->e, st->S); for (int d = 0; d <= st->D; d++) hh[d] = h[d];
    form(st, hh, &scratch); out[0] = scratch.ans; for (int j = 2; j <= st->L; j++) out[j] = scratch.g[j];
}
void state(int32_t *h, int32_t *p) { for (int d = 0; d < 16; d++) h[d] = hist[d]; *p = pend; }
/* cells that have counted, counts above zero; and, given a set of byte values (dead[v]), the cells keyed on one of them at a chain position (2..D) */
static void census1(const St *st, const int32_t *C, int stride, int nk, const int32_t *kd, const int32_t *kn, const int64_t *base, int guess, const uint8_t *dead, int64_t *out) {
    const int S = st->S;
    for (int k = 0; k < nk; k++) {
        int64_t cnt = 1; for (int i = 0; i < kn[k]; i++) cnt *= S;
        for (int64_t idx = 0; idx < cnt; idx++) {
            int isdead = 0; int64_t q = idx; for (int i = 0; i < kn[k]; i++) { const int v = q % S; q /= S; if (kd[k * 4 + i] >= 2 && dead[v]) isdead = 1; }
            for (int g = 0; g < (guess ? S : 1); g++) {
                const int32_t *c = C + (guess ? base[k] + idx * S + g : base[k] + idx) * stride; int64_t n = 0, ab = 0;
                for (int v = 0; v < S; v++) n += c[v];
                if (!n) continue;
                for (int v = 0; v < stride; v++) ab += c[v] > 0;
                out[0]++; out[1] += ab; if (isdead) { out[2]++; out[3] += ab; }
            }
        }
    }
}
void census(const St *st, const uint8_t *dead, int64_t *out) {
    const int L1 = st->L > 1 ? st->L : 1; out[0] = out[1] = out[2] = out[3] = 0;
    census1(st, st->P, L1 * st->S, st->nP, st->pd, st->pn, st->pbase, 0, dead, out);
    for (int j = 2; j <= st->L; j++) census1(st, st->F[j], (j - 1) * st->S, st->nG, st->gd, st->gn, st->gbase, 1, dead, out);
}
'''
PI32 = ctypes.POINTER(ctypes.c_int32); PI64 = ctypes.POINTER(ctypes.c_int64); PD = ctypes.POINTER(ctypes.c_double); PU8 = ctypes.POINTER(ctypes.c_uint8)
class St(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int32) for n in ("S", "D", "nP", "nG", "L")] + [("eta", ctypes.c_double), ("e", ctypes.c_double), ("ncap", ctypes.c_double), ("P", PI32), ("F", PI32 * (MAXL + 1))] + [(n, PD) for n in ("T", "U")] + \
               [(n, PI32) for n in ("pd", "pn", "gd", "gn")] + [(n, PI64) for n in ("pbase", "gbase")] + [("fam", (ctypes.c_int32 * MAXL) * (MAXL + 1))]

def kern():
    so = os.path.join(HERE, "kern_u.so"); src = os.path.join(HERE, "kern_u.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); L.run.argtypes = [ctypes.POINTER(St), PI32, PU8, ctypes.c_int64, PD, PI32]; L.run.restype = ctypes.c_double
    L.answer.argtypes = [ctypes.POINTER(St), PI32, PI32]; L.census.argtypes = [ctypes.POINTER(St), PU8, PI64]; L.reset.argtypes = [ctypes.c_int]; L.state.argtypes = [PI32, PI32]
    for f in (L.answer, L.census, L.reset, L.state): f.restype = None
    return L

class Fast:
    def __init__(s, D=9, m=4, mg=3, look=3, eta=0.1, e=2.0, tp=0.02, tg=0.0, ncap=0.0):
        assert S <= MAXS and look <= MAXL
        s.D = D; s.L = look; s.pk = kinds(D, m); s.gk = kinds(D, mg, True); s.K = kern(); assert len(s.pk) <= MAXK and len(s.gk) <= MAXK
        def lay(ks):
            d = np.zeros((len(ks), 4), np.int32); n = np.zeros(len(ks), np.int32); base = np.zeros(len(ks), np.int64); tot = 0
            for k, ds in enumerate(ks): d[k, :len(ds)] = ds; n[k] = len(ds); base[k] = tot; tot += S ** len(ds)
            return d, n, base, tot
        s.pd, s.pn, s.pbase, s.nPc = lay(s.pk); s.gd, s.gn, s.gbase, nk = lay(s.gk); s.gbase *= S; s.nGc = nk * S; L1 = max(look, 1)
        s.P = np.zeros(s.nPc * L1 * S, np.int32); s.F = {j: np.zeros(s.nGc * (j - 1) * S, np.int32) for j in range(2, look + 1)}
        s.T = birth(look, tp, tg); s.U = np.zeros_like(s.T); p = lambda a, t: a.ctypes.data_as(t); fam, s.nfam = fams(look)
        s.st = St(); st = s.st; st.S, st.D, st.nP, st.nG, st.L, st.eta, st.e, st.ncap = S, D, len(s.pk), len(s.gk), look, eta, e, ncap; st.P = p(s.P, PI32); st.T = p(s.T, PD); st.U = p(s.U, PD)
        for j, a in s.F.items(): st.F[j] = p(a, PI32)
        st.pd, st.pn, st.gd, st.gn, st.pbase, st.gbase = p(s.pd, PI32), p(s.pn, PI32), p(s.gd, PI32), p(s.gn, PI32), p(s.pbase, PI64), p(s.gbase, PI64)
        for (j, w), f in fam.items(): st.fam[j][w] = f
        s.fam = fam; s.K.reset(END)
    def run(s, x, fl=None, want=False):
        x = np.ascontiguousarray(x, np.int32); L1 = max(s.L, 1); P = np.zeros((len(x), L1, S)) if want else None; G = np.zeros((len(x), MAXL + 1), np.int32) if want else None
        f = None if fl is None else np.ascontiguousarray(fl, np.uint8)
        e = s.K.run(ctypes.byref(s.st), x.ctypes.data_as(PI32), None if f is None else f.ctypes.data_as(PU8), len(x), P.ctypes.data_as(PD) if want else None, G.ctypes.data_as(PI32) if want else None)
        return (e, P, G) if want else e
    def answer(s, h):
        hh = np.zeros(16, np.int32); hh[:len(h)] = h; out = np.zeros(MAXL + 1, np.int32); s.K.answer(ctypes.byref(s.st), hh.ctypes.data_as(PI32), out.ctypes.data_as(PI32)); return [int(v) for v in out]      # [0] the answer, [j] the guess j ahead
    def state(s):
        h = np.zeros(16, np.int32); p = ctypes.c_int32(0); s.K.state(h.ctypes.data_as(PI32), ctypes.byref(p)); return [int(v) for v in h[:s.D + 1]], bool(p.value)
    def census(s, dead=None):
        d = np.zeros(S, np.uint8) if dead is None else np.ascontiguousarray(dead, np.uint8); out = np.zeros(4, np.int64); s.K.census(ctypes.byref(s.st), d.ctypes.data_as(PU8), out.ctypes.data_as(PI64)); return [int(v) for v in out]
    def sizes(s): c = s.census(); return c[0], c[1], int((s.U > 0).sum())

# ------------------------------------------------------------------ the score: the system writes its own answer bytes; nothing changes
def score(sys_, tx, op, pairs, gap, admit, D=9, past=None):
    """Before the line stand line ends only. The line is read up to the '='; the system then writes the answer's digits and the end mark from its own bytes.
    admit[v]: whether the byte value v enters the chain. Returns whole % (the digits), whole % with the end mark, by place %, and how often the guesses made at the
    '=' are right (the byte j ahead, j = 2..L)."""
    nd = 3 if tx == "g2" else 4; whole = wend = 0; place = np.zeros(nd); L = sys_.L; gr = np.zeros(MAXL + 1)
    for a, b in pairs:
        ln = text(tx, op, a, b, fill(a, b, gap)); eq = ln.index(EQ); h = [0] + [END] * D; pend = True; pre = past() if past else []      # past: a line read before the held-out line (PLAN_4.md); none: line ends only ("empty past")
        def push(c):
            nonlocal h, pend
            h = ([0, c] + h[1:D]) if pend else ([0, c] + h[2:]); pend = bool(admit[c])
        for c in pre + ln[:eq + 1]: push(c)
        ok = True
        for i in range(nd + 1):
            r = sys_.answer(h)
            if i == 0:
                for j in range(2, L + 1): gr[j] += r[j] == ln[eq + j]
            if i < nd: place[i] += r[0] == ln[eq + 1 + i]; ok &= r[0] == ln[eq + 1 + i]
            else: wend += ok and r[0] == END
            push(r[0])
        whole += ok
    n = len(pairs); return 100 * whole / n, 100 * wend / n, (100 * place / n).round(1).tolist(), (100 * gr[2:L + 1] / n).round(1).tolist()

def run(kw):
    t0 = time.process_time(); tx = kw.get("text", "g2"); seed = int(kw.get("seed", 1)); every = int(kw.get("every", 10000)); G = int(kw.get("gap", 8)); gaps = sorted({0, G // 2, G})
    plan = kw.get("plan", "add:200000"); setup(sub=("sub" in plan) and not int(kw.get("nosub", 0))); rng, held, shown = split(tx, seed)
    s = Fast(D=int(kw.get("D", 9)), m=int(kw.get("m", 4)), mg=int(kw.get("mg", 3)), look=int(kw.get("look", 3)), ncap=float(kw.get("ncap", 0)))
    ck = kw.get("clock", "none"); crng = np.random.default_rng([seed, 20261124]); closed_now = np.zeros(S, bool)
    if ck == "oracle": closed_now[LETS] = True
    P = int(kw.get("P", 256)); alpha = float(kw.get("alpha", 2)); eps = float(kw.get("eps", 0.01)); beta = float(kw.get("beta", 0.02)); p0 = float(kw.get("p0", 0.2)); gam = 1 - 1 / float(kw.get("win", 3000))
    freeze = int(kw.get("freeze", 0)); u = np.full(S, np.log(p0 / (1 - p0))); ev = np.zeros((4, S)); tr = dict(b=None, w=None, n=0, zc=[], zo=[])
    def trial_block(xs):
        nonlocal u, ev
        for i in range(0, len(xs), P):
            c = np.clip(sg(u), eps, 1 - eps); closed = crng.random(S) < c; x = xs[i:i + P]; e = s.run(x, ~closed[x])
            if len(x) < P // 4: continue
            r = e / len(x); tr["n"] += 1
            if tr["b"] is None: tr["b"] = r; tr["w"] = 0.05 * r; continue
            z = float(np.clip((tr["b"] - r) / max(tr["w"], 1e-12), -3, 3))
            ev *= gam; ev[0] += z * closed; ev[1] += closed; ev[2] += z * ~closed; ev[3] += ~closed
            u = np.log(p0 / (1 - p0)) + alpha * (ev[0] / (ev[1] + 1) - ev[2] / (ev[3] + 1)) / np.sqrt(1 / (ev[1] + 1) + 1 / (ev[3] + 1))
            tr["w"] += beta * (abs(r - tr["b"]) - tr["w"]); tr["b"] += beta * (r - tr["b"])
    hassub = "-" in shown and ord("-") in SI; lines_shown = {tuple(text(tx, "-", a, b)) for a, b in shown["-"]} if hassub else set()
    clean = [p for p in (held["-"] if hassub else []) if tuple(text(tx, "-", *p)) not in lines_shown]                 # held-out subtractions whose line no shown pair writes (ERRATA_20261120.md)
    out = dict(label=kw.get("label"), args=kw, rows=[], syms=[chr(c) for c in SYMS], clean_pairs=len(clean), kinds=[len(s.pk), len(s.gk)], cells_addressable=int(s.nPc + s.nGc * max(s.L - 1, 0)))
    def reading(pi, name, lines, full_sub):
        if ck == "trial": closed_now[:] = u > 0
        adm = ~closed_now; row = dict(phase=pi, name=name, lines=lines, add={}, add_end={}, add_place={}, add_guess={})
        for g in gaps: row["add"][g], row["add_end"][g], row["add_place"][g], row["add_guess"][g] = score(s, tx, "+", held["+"], g, adm, s.D)
        if npast:                                          # the same score after a read line: mean of npast draws; the line before each held-out line is drawn by chance from the lines of the phase being read
            def past():
                a, b = lastshown[prng.integers(len(lastshown))]; return text(tx, lastop, int(a), int(b), "".join(chr(LET[i]) for i in prng.integers(4, size=prng.integers(G + 1))))
            row["add_past"] = {}; row["add_past_place"] = {}
            for g in gaps:
                q = [score(s, tx, "+", held["+"], g, adm, s.D, past) for _ in range(npast)]; row["add_past"][g] = float(np.mean([x[0] for x in q])); row["add_past_place"][g] = np.mean([x[2] for x in q], 0).round(1).tolist()
        if full_sub:
            row["sub_all"] = {g: score(s, tx, "-", held["-"], g, adm, s.D)[0] for g in gaps}; row["sub_clean"] = {}; row["sub_clean_place"] = {}
            for g in gaps: row["sub_clean"][g], _, row["sub_clean_place"][g], _ = score(s, tx, "-", clean, g, adm, s.D)
        row["closed"] = "".join(chr(SYMS[v]) for v in np.flatnonzero(closed_now)).replace("\n", "N")
        if ck == "trial": row["u"] = [round(float(x), 3) for x in u]; row["periods"] = tr["n"]; row["mismatch_mean"] = tr["b"]; row["ev_counts"] = [[round(float(x), 1) for x in ev[1]], [round(float(x), 1) for x in ev[3]]]
        c = s.census(closed_now); row["cells"], row["counts_above_zero"], row["cells_on_closed"], row["counts_on_closed"] = c; row["trusts_used"] = int((s.U > 0).sum()); row["cpu"] = round(time.process_time() - t0, 1)
        out["rows"].append(row)
        print(name, lines, "never", "/".join(f"{v:.1f}" for v in row["add"].values()), "closed", row["closed"], "after a read line", "/".join(f"{v:.1f}" for v in row.get("add_past", {}).values()), "cells", row["cells"], "on closed", row["cells_on_closed"], "cpu", row["cpu"], file=sys.stderr, flush=True)
    npast = int(kw.get("past", 0)); prng = np.random.default_rng([seed, 777]); lastshown = None; lastop = "+"
    for pi, ph in enumerate(plan.split(",")):
        name, n = ph.split(":"); op = {"add": "+", "sub": "-"}[name]; n = int(n); assert n % every == 0; lastop = op
        lastshown = [(int(q) // 1000, int(q) % 1000) for q in shown[op]] if tx == "t3" else shown[op]
        for b in range(n // every):
            xs = stream(tx, op, shown[op], G, every, rng)
            if ck == "trial" and freeze and (b * every >= freeze or pi > 0): closed_now[:] = u > 0; s.run(xs, ~closed_now[xs])      # freeze=N: after N lines no entry is tried any more
            elif ck == "trial": trial_block(xs)
            else: s.run(xs, ~closed_now[xs])
            reading(pi, name, (b + 1) * every, "-" in held and (pi > 0 or b == n // every - 1) and "sub" in plan)
    r0 = [r for r in out["rows"] if r["phase"] == 0]; out["last3_by_gap"] = {g: float(np.mean([r["add"][g] for r in r0[-3:]])) for g in gaps}; out["last_by_gap"] = dict(r0[-1]["add"])
    out["cpu"] = round(time.process_time() - t0, 1); fam = {v: k for k, v in s.fam.items()}
    out["trust_abs_sum_by_family"] = {(f"plain count {f + 1}" if f < max(s.L, 1) else f"keyed on guess {fam[f][0]} ahead, count {fam[f][1] + 1}"): float(np.abs(s.T[f] * (s.U[f] > 0)).sum()) for f in range(s.nfam)}
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"{kw['label']}.json"), "w")); return out

def check():
    ok_all = True
    for kw, nb, tx, cl in ((dict(D=5, m=2, mg=1, look=3), 1500, "g2", 1), (dict(D=6, m=3, mg=2, look=3), 1500, "g2", 1), (dict(D=6, m=2, mg=2, look=3), 1500, "g2", 0), (dict(D=5, m=2, mg=1, look=0), 1500, "g2", 1),
                           (dict(D=5, m=2, mg=1, look=1), 1000, "g2", 1), (dict(D=6, m=2, mg=1, look=2, tg=0.05, eta=0.03, e=0.5), 1500, "g2", 1), (dict(D=6, m=2, mg=1, look=4), 1500, "t3", 1),
                           (dict(D=9, m=4, mg=3, look=3), 400, "g2", 1), (dict(D=7, m=3, mg=2, look=4), 500, "t3", 1), (dict(D=6, m=3, mg=2, look=3, ncap=4.0), 2500, "g2", 1)):
        setup(); rng, held, shown = split(tx, 7); x = np.concatenate([stream(tx, "+-"[i % 3 == 1 and tx == "g2"], shown["+"], 8, 20, rng) for i in range(nb // 200 + 2)])[:nb]
        fl = (rng.random(nb) < 0.7).astype(np.uint8) if cl else None                    # the chain steps at 70% of the bytes, by chance
        f = Fast(**kw); h = nb // 2; e1, P, G = f.run(x[:h], None if fl is None else fl[:h], True); e2, P2, G2 = f.run(x[h:], None if fl is None else fl[h:], True); P = np.concatenate([P, P2]); G = np.concatenate([G, G2])
        r = Ref(**kw); dp = 0.0; same = True; L = kw["look"]
        for t in range(nb):
            q = r.step(int(x[t]), True if fl is None else bool(fl[t]))
            for w in range(max(L, 1)): dp = max(dp, float(np.abs(q["p"][w] - P[t, w]).max()))
            same &= q["ans"] == G[t, 0] and all(q["g"][j] == G[t, j] for j in range(2, L + 1))
        dT = float(np.abs(r.T - f.T).max()); sz = r.sizes() == f.sizes(); de = abs(r.esum - e1 - e2); fh, fp = f.state(); st_ok = fh[1:] == r.h[1:] and fp == r.pend
        hs = [[0] + [int(v) for v in rng.integers(S, size=kw["D"])] for _ in range(50)]
        ans = all((lambda a, q: a[0] == q["ans"] and all(a[j] == q["g"][j] for j in range(2, L + 1)))(f.answer(hh), r.form(hh)) for hh in hs)
        ok = dp < 1e-9 and dT < 1e-9 and same and sz and ans and de < 1e-7 and st_ok; ok_all &= ok
        print(f"{kw} text {tx}, {'chain stepping at 70% of bytes' if cl else 'chain stepping at every byte'}: {nb} bytes, {len(r.pk)} plain kinds, {len(r.gk)} kinds keyed on each guess, sizes (cells, counts above zero, trusts used) {r.sizes()}: "
              f"compiled against the reference: shares {dp:.1e}, trusts {dT:.1e}, sum of squared mismatch {de:.1e}; answers and guesses the same: {bool(same)}; sizes the same: {sz}; chain the same: {st_ok}; answers without learning the same: {ans}", "PASSED" if ok else "FAILED", flush=True)
    return ok_all

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=") for a in sys.argv[3:]); kw["label"] = sys.argv[2]; o = run(kw); r = o["rows"][-1]
    print(o["label"], "| last3 by gap", " ".join(f"{g}:{v:.1f}" for g, v in o["last3_by_gap"].items()), "|", " ".join(f"{q['phase']}:{q['lines'] // 1000}k " + "/".join(f"{v:.0f}" for v in q["add"].values()) + (f"[{q['closed']}]" if q["closed"] else "") for q in o["rows"]),
          "| place", r["add_place"], "guess", r["add_guess"], "| cells", r["cells"], "counts", r["counts_above_zero"], "on closed", r["cells_on_closed"], "trusts", r["trusts_used"], "|", o["cpu"], "s", flush=True)
