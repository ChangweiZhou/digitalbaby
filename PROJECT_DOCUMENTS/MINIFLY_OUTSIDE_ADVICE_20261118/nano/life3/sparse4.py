#!/usr/bin/env python3
"""PLAN_10.md (the coordinating session's idea): cells that also count what arrives two and three bytes after them; the answering cell forms, before
each byte, a guess for the byte three ahead, then for the byte two ahead, then its answer for the next byte; cells may be keyed on a guess.
    python3 sparse4.py <label> seed=1 [plan=add:40000[,sub:20000]] [m=4] [eta=0.1] [e=2] [T0=0.02] [Tg0=0] [shrink=1] [ncap=0] [rev=0] [every=5000] [ref=1]
    python3 sparse4.py check
Plain cells: as sparse.py (a kind is a set of at most m of the delays 1..9), each with three sets of counts: of the next byte, of the byte after it,
and of the one after that (the text supplies each when it arrives). Before the byte at t:
  1. the plain cells' third counts are read with their own trusts: the guess g3 for the byte at t+2 is the byte of the largest branch;
  2. cells of kind A (a set of at most three delays, or none, together with g3) exist for each combination that has occurred; they hold counts of the
     next byte and of the byte after it. The plain cells' second counts and the A cells' second counts give the guess g2 for the byte at t+1;
  3. cells of kind B (a set of at most three delays, or none, together with g2) hold counts of the next byte. The plain cells' first counts, the A
     cells' first counts and the B cells' counts give the answer for the byte at t.
A guess is the system's own, right or wrong, in learning as in scoring. Trusts: one for each family (which cells, which counts), kind and level of
experience; those of the cells keyed on a guess are born at Tg0. A trust that was used for a guess changes when the byte guessed arrives.
The text, split, score and life are sparse.py's (imported). One learning system at a time in a process (the kernel keeps the last three bytes' cells)."""
import ctypes, json, os, subprocess, sys, time
import numpy as np
import sparse as sp
from sparse import S, END, NL, HP, text, kinds_of, level_np
HERE = os.path.dirname(os.path.abspath(__file__))

SRC = r'''
#include <math.h>
#include <stdint.h>
#define NL 40
#define MAXS 16
#define MAXK 300
#define MAXA 140
#define MAXN (MAXK + 2 * MAXA)
#define HP 16
typedef struct { int64_t S, K, KA, ZTOT, PTOT, END, ncell, cap, step, shrink; const int32_t *nd, *del; const int64_t *off; int32_t *map; float *C, *N, *U; double *T, *Tu; double eta, e, ncap; } sys_t;
typedef struct { double p[3][MAXS]; int n[3]; int tix[3][MAXN]; double s[3][MAXN][MAXS]; int64_t idP[MAXK], idA[MAXA], idB[MAXA]; int g3, g2; } rec_t;
static rec_t ring[3], tmp; static int64_t keyv[MAXK];
static inline int level(float n, float u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* what the cell id says from its counts in `slot` (0: of the next byte, 1: of the one after, 2: of the one after that); returns the trust index or -1 */
static inline int say(const sys_t *q, int64_t id, int slot, int fam, int64_t k, double *s) {
    if (id < 0) return -1; const float n = q->N[id * 3 + slot]; if (n < 1) return -1;
    const int64_t S = q->S; const float *c = q->C + (id * 3 + slot) * S;
    if (q->ncap > 0 && n > q->ncap) { const double sc = q->ncap / n, den = log(q->ncap + S * q->e); for (int v = 0; v < S; v++) s[v] = log(c[v] * sc + q->e) - den; }
    else { const double den = log(n + S * q->e); for (int v = 0; v < S; v++) s[v] = log(c[v] + q->e) - den; }
    return (int)(((fam * (q->K + 1) + k) * NL) + level(n, q->U[id * 3 + slot]));
}
static inline void add(const sys_t *q, rec_t *r, int d, int64_t id, int slot, int fam, int64_t k, double *a) {
    const int i = r->n[d], t = say(q, id, slot, fam, k, r->s[d][i]); if (t < 0) return;
    const double tk = q->T[t]; for (int v = 0; v < q->S; v++) a[v] += tk * r->s[d][i][v]; r->tix[d][i] = t; r->n[d] = i + 1;
}
static inline int soft(const sys_t *q, const double *a, double *p) {
    double mx = -1e300, z = 0; int b = 0;
    for (int v = 0; v < q->S; v++) if (a[v] > mx) { mx = a[v]; b = v; }
    for (int v = 0; v < q->S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
    for (int v = 0; v < q->S; v++) p[v] /= z;
    return b;
}
static inline int64_t cell(sys_t *q, int64_t slot, int make) { int64_t id = q->map[slot]; if (id < 0 && make) { id = q->ncell++; q->map[slot] = (int32_t)id; } return id; }
/* h[-d] is the byte d places back */
static void look(sys_t *q, const int32_t *h, int make, rec_t *r) {
    const int64_t S = q->S, K = q->K, KA = q->KA; double a[MAXS];
    for (int64_t k = 0; k < K; k++) { int64_t key = 0; for (int j = 0; j < q->nd[k]; j++) key = key * S + h[-q->del[k * 4 + j]]; keyv[k] = key; r->idP[k] = cell(q, q->off[k] + key, make); }
    r->n[0] = r->n[1] = r->n[2] = 0;
    for (int v = 0; v < S; v++) a[v] = 0;
    for (int64_t k = 0; k < K; k++) add(q, r, 2, r->idP[k], 2, 2, k, a);
    r->g3 = soft(q, a, r->p[2]);
    for (int64_t k = 0; k <= KA; k++) r->idA[k] = cell(q, q->PTOT + (k < KA ? q->off[k] + keyv[k] : q->ZTOT) * S + r->g3, make);
    for (int v = 0; v < S; v++) a[v] = 0;
    for (int64_t k = 0; k < K; k++) add(q, r, 1, r->idP[k], 1, 1, k, a);
    for (int64_t k = 0; k <= KA; k++) add(q, r, 1, r->idA[k], 1, 4, k, a);
    r->g2 = soft(q, a, r->p[1]);
    for (int64_t k = 0; k <= KA; k++) r->idB[k] = cell(q, q->PTOT + (q->ZTOT + 1) * S + (k < KA ? q->off[k] + keyv[k] : q->ZTOT) * S + r->g2, make);
    for (int v = 0; v < S; v++) a[v] = 0;
    for (int64_t k = 0; k < K; k++) add(q, r, 0, r->idP[k], 0, 0, k, a);
    for (int64_t k = 0; k <= KA; k++) add(q, r, 0, r->idA[k], 0, 3, k, a);
    for (int64_t k = 0; k <= KA; k++) add(q, r, 0, r->idB[k], 0, 5, k, a);
    soft(q, a, r->p[0]);
}
static inline void trust(sys_t *q, const rec_t *r, int d, int y) {
    for (int i = 0; i < r->n[d]; i++) {
        const double *s = r->s[d][i]; double g = s[y]; for (int v = 0; v < q->S; v++) g -= r->p[d][v] * s[v];
        const int t = r->tix[d][i]; q->T[t] += (q->shrink ? q->eta / sqrt(1.0 + q->Tu[t]) : q->eta) * g; q->Tu[t] += 1.0;
    }
}
static inline void count(sys_t *q, int64_t id, int slot, int y) { float *c = q->C + (id * 3 + slot) * q->S; if (c[y] == 0) q->U[id * 3 + slot] += 1; c[y] += 1; q->N[id * 3 + slot] += 1; }
int64_t steps(sys_t *q, int64_t n, const int32_t *x, double *P) {
    const int64_t S = q->S, K = q->K, KA = q->KA;
    for (int64_t t = 0; t < n; t++) {
        if (q->ncell + K + 2 * (KA + 1) > q->cap) return t;
        rec_t *cur = &ring[q->step % 3], *prev = q->step >= 1 ? &ring[(q->step + 2) % 3] : 0, *pp = q->step >= 2 ? &ring[(q->step + 1) % 3] : 0;
        look(q, x + t, 1, cur); const int y = x[t];
        if (P) for (int d = 0; d < 3; d++) for (int v = 0; v < S; v++) P[(t * 3 + d) * S + v] = cur->p[d][v];
        trust(q, cur, 0, y); if (prev) trust(q, prev, 1, y); if (pp) trust(q, pp, 2, y);
        for (int64_t k = 0; k < K; k++) count(q, cur->idP[k], 0, y);
        for (int64_t k = 0; k <= KA; k++) { count(q, cur->idA[k], 0, y); count(q, cur->idB[k], 0, y); }
        if (prev) { for (int64_t k = 0; k < K; k++) count(q, prev->idP[k], 1, y); for (int64_t k = 0; k <= KA; k++) count(q, prev->idA[k], 1, y); }
        if (pp) for (int64_t k = 0; k < K; k++) count(q, pp->idP[k], 2, y);
        q->step++;
    }
    return n;
}
/* out: the byte named at each place; gs: at each place the guesses (three ahead, two ahead) made there */
void answers(sys_t *q, int64_t n, const int32_t *lines, int64_t L, int64_t from, int64_t cnt, int64_t own, int32_t *out, int32_t *gs) {
    int32_t buf[HP + 64];
    for (int64_t i = 0; i < n; i++) {
        for (int j = 0; j < HP; j++) buf[j] = (int32_t)q->END;
        for (int j = 0; j < L; j++) buf[HP + j] = lines[i * L + j];
        for (int64_t c = 0; c < cnt; c++) {
            look(q, buf + HP + from + c, 0, &tmp); int b = 0; for (int v = 1; v < q->S; v++) if (tmp.p[0][v] > tmp.p[0][b]) b = v;
            out[i * cnt + c] = b; gs[(i * cnt + c) * 2] = tmp.g3; gs[(i * cnt + c) * 2 + 1] = tmp.g2; if (own) buf[HP + from + c] = b;
        }
    }
}
'''
class CSys4(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int64) for n in ("S", "K", "KA", "ZTOT", "PTOT", "END", "ncell", "cap", "step", "shrink")] + [(n, ctypes.c_void_p) for n in ("nd", "del_", "off", "map", "C", "N", "U", "T", "Tu")] + [(n, ctypes.c_double) for n in ("eta", "e", "ncap")]

def kern():
    so = os.path.join(HERE, "kern_s4.so"); src = os.path.join(HERE, "kern_s4.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); L.steps.restype = ctypes.c_int64; L.steps.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
    L.answers.restype = None; L.answers.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p] + [ctypes.c_int64] * 4 + [ctypes.c_void_p, ctypes.c_void_p]; return L

def births(K, T0, Tg0):
    T = np.full((6 * (K + 1), NL), float(T0)); T[3 * (K + 1):] = float(Tg0); return T                      # families 0, 1, 2: plain cells' three counts; 3, 4: A cells; 5: B cells

class Sys4:
    FAMS = ("plain, next byte", "plain, one after", "plain, two after", "keyed on the guess three ahead, next byte", "keyed on the guess three ahead, one after", "keyed on the guess two ahead, next byte")
    def __init__(s, D=9, m=4, eta=0.1, e=2.0, T0=0.02, Tg0=0.0, shrink=1, ncap=0.0, cap=1 << 16):
        s.kinds = kinds_of(D, m); s.K = K = len(s.kinds); s.KA = len(kinds_of(D, min(m, 3))); s.S = S; s.eta, s.e, s.shrink, s.ncap = eta, e, int(shrink), float(ncap); assert K <= 300 and s.KA < 140
        s.nd = np.array([len(k) for k in s.kinds], np.int32); s.dl = np.zeros((K, 4), np.int32)
        for i, k in enumerate(s.kinds): s.dl[i, :len(k)] = k
        sz = np.array([S ** len(k) for k in s.kinds], np.int64); s.off = np.concatenate([[0], np.cumsum(sz)[:-1]]).astype(np.int64); s.PTOT = int(sz.sum()); s.ZTOT = int(sz[:s.KA].sum())
        s.map = np.full(s.PTOT + 2 * (s.ZTOT + 1) * S, -1, np.int32); s.cap = cap; s.C = np.zeros((cap * 3, S), np.float32); s.N = np.zeros(cap * 3, np.float32); s.U = np.zeros(cap * 3, np.float32)
        s.T = births(K, T0, Tg0); s.Tu = np.zeros_like(s.T); s.ncell = 0; s.step = 0; s.tail = np.full(HP, END, np.int32); s.L = kern()
    def cs(s): return CSys4(s.S, s.K, s.KA, s.ZTOT, s.PTOT, END, s.ncell, s.cap, s.step, s.shrink, s.nd.ctypes.data, s.dl.ctypes.data, s.off.ctypes.data, s.map.ctypes.data, s.C.ctypes.data, s.N.ctypes.data, s.U.ctypes.data, s.T.ctypes.data, s.Tu.ctypes.data, s.eta, s.e, s.ncap)
    def grow(s):
        s.cap *= 2; C = np.zeros((s.cap * 3, s.S), np.float32); C[:len(s.C)] = s.C; s.C = C
        for nm in "NU": a = np.zeros(s.cap * 3, np.float32); a[:len(getattr(s, nm))] = getattr(s, nm); setattr(s, nm, a)
    def read(s, x, learn=True, want_p=False):
        assert learn; x = np.asarray(x, np.int32).ravel(); buf = np.ascontiguousarray(np.concatenate([s.tail, x])); n = len(x); P = np.zeros((n, 3, s.S)) if want_p else None; done = 0
        while done < n:
            q = s.cs(); done += s.L.steps(ctypes.byref(q), n - done, buf.ctypes.data + 4 * (HP + done), P.ctypes.data + 8 * 3 * s.S * done if want_p else None); s.ncell = q.ncell; s.step = q.step
            if done < n: s.grow()
        s.tail = buf[-HP:].copy(); return P
    def answers(s, lines, own=1, frm=6, cnt=3, want_acc=False, want_g=False):
        lines = np.ascontiguousarray(lines, np.int32); out = np.zeros((len(lines), cnt), np.int32); gs = np.zeros((len(lines), cnt, 2), np.int32); q = s.cs()
        s.L.answers(ctypes.byref(q), len(lines), lines.ctypes.data, lines.shape[1], frm, cnt, own, out.ctypes.data, gs.ctypes.data)
        if want_g: return out, gs
        return (out, np.ones((cnt, s.K, 2))) if want_acc else out
    def size(s):
        N = s.N[:s.ncell * 3].reshape(-1, 3); C = s.C[:s.ncell * 3]; return dict(cells=int((N > 0).any(1).sum()), counts_above_zero=int((C > 0).sum()), trusts_used=int((s.Tu > 0).sum()), trusts=int(s.T.size))

class Ref4:
    """numpy reference: the same arithmetic, dictionaries, one byte at a time."""
    def __init__(s, D=9, m=4, eta=0.1, e=2.0, T0=0.02, Tg0=0.0, shrink=1, ncap=0.0):
        s.kinds = kinds_of(D, m); s.K = len(s.kinds); s.KA = len(kinds_of(D, min(m, 3))); s.S = S; s.eta, s.e, s.shrink, s.ncap = eta, e, int(shrink), float(ncap)
        s.cell = {}; s.C = []; s.N = []; s.U = []; s.T = births(s.K, T0, Tg0); s.Tu = np.zeros_like(s.T); s.hist = [END] * HP; s.past = []
    def get(s, key, make):
        i = s.cell.get(key, -1)
        if i < 0 and make: i = len(s.C); s.cell[key] = i; s.C.append(np.zeros((3, s.S), np.float32)); s.N.append([0.0] * 3); s.U.append([0.0] * 3)
        return i
    def say(s, i, slot, fam, k):
        if i < 0 or s.N[i][slot] < 1: return None
        n = s.N[i][slot]; sc = s.ncap / n if s.ncap > 0 and n > s.ncap else 1.0
        return (fam * (s.K + 1) + k, level_np(n, s.U[i][slot])), np.log(s.C[i][slot].astype(np.float64) * sc + s.e) - np.log(n * sc + s.S * s.e)
    def dist(s, items):
        a = np.zeros(s.S); act = []
        for it in items:
            if it is not None: act.append(it); a += s.T[it[0]] * it[1]
        b = int(a.argmax()); p = np.exp(a - a.max()); return p / p.sum(), act, b
    def look(s, hist, make):
        keys = [tuple(hist[-d] for d in dl) for dl in s.kinds]; idP = [s.get(("P", k) + keys[k], make) for k in range(s.K)]; r = dict(idP=idP, p=[None] * 3, act=[None] * 3)
        r["p"][2], r["act"][2], g3 = s.dist([s.say(idP[k], 2, 2, k) for k in range(s.K)])
        idA = [s.get(("A", k) + (keys[k] if k < s.KA else ()) + (g3,), make) for k in range(s.KA + 1)]
        r["p"][1], r["act"][1], g2 = s.dist([s.say(idP[k], 1, 1, k) for k in range(s.K)] + [s.say(idA[k], 1, 4, k) for k in range(s.KA + 1)])
        idB = [s.get(("B", k) + (keys[k] if k < s.KA else ()) + (g2,), make) for k in range(s.KA + 1)]
        r["p"][0], r["act"][0], b = s.dist([s.say(idP[k], 0, 0, k) for k in range(s.K)] + [s.say(idA[k], 0, 3, k) for k in range(s.KA + 1)] + [s.say(idB[k], 0, 5, k) for k in range(s.KA + 1)])
        r.update(idA=idA, idB=idB, g3=g3, g2=g2, b=b); return r
    def trust(s, r, d, y):
        for ti, sv in r["act"][d]: s.T[ti] += (s.eta / np.sqrt(1.0 + s.Tu[ti]) if s.shrink else s.eta) * (sv[y] - r["p"][d] @ sv); s.Tu[ti] += 1.0
    def count(s, i, slot, y):
        if s.C[i][slot][y] == 0: s.U[i][slot] += 1
        s.C[i][slot][y] += 1; s.N[i][slot] += 1
    def read(s, x, learn=True, want_p=False):
        P = []; hist = list(s.hist)
        for y in np.asarray(x).ravel().tolist():
            cur = s.look(hist, True); P.append(np.array(cur["p"])); prev = s.past[-1] if len(s.past) >= 1 else None; pp = s.past[-2] if len(s.past) >= 2 else None
            s.trust(cur, 0, y)
            if prev: s.trust(prev, 1, y)
            if pp: s.trust(pp, 2, y)
            for i in cur["idP"]: s.count(i, 0, y)
            for i, j in zip(cur["idA"], cur["idB"]): s.count(i, 0, y); s.count(j, 0, y)
            if prev:
                for i in prev["idP"] + prev["idA"]: s.count(i, 1, y)
            if pp:
                for i in pp["idP"]: s.count(i, 2, y)
            s.past = (s.past + [cur])[-2:]; hist.append(y); hist = hist[-HP:]
        s.hist = hist; return np.array(P) if want_p else None
    def answers(s, lines, own=1, frm=6, cnt=3, want_g=False):
        out = np.zeros((len(lines), cnt), np.int32); gs = np.zeros((len(lines), cnt, 2), np.int32)
        for i, L in enumerate(np.asarray(lines).tolist()):
            buf = [END] * HP + L
            for c in range(cnt):
                r = s.look(buf[:HP + frm + c], False); out[i, c] = r["b"]; gs[i, c] = r["g3"], r["g2"]
                if own: buf[HP + frm + c] = r["b"]
        return (out, gs) if want_g else out
    def size(s): return dict(cells=int(sum(max(n) > 0 for n in s.N)), counts_above_zero=int(sum((c > 0).sum() for c in s.C)), trusts_used=int((s.Tu > 0).sum()), trusts=int(s.T.size))

def make(kw, cls=None):
    cls = cls or (Ref4 if int(kw.get("ref", 0)) else Sys4)
    return cls(D=int(kw.get("D", 9)), m=int(kw.get("m", 4)), eta=float(kw.get("eta", 0.1)), e=float(kw.get("e", 2)), T0=float(kw.get("T0", 0.02)), Tg0=float(kw.get("Tg0", 0)), shrink=int(kw.get("shrink", 1)), ncap=float(kw.get("ncap", 0)))

def run(kw):
    t0 = time.process_time(); s = make(kw); o = sp.run(dict(kw, nowrite=1), s); o["args"] = kw; o.update(s.size()); seed = int(kw.get("seed", 1)); rev = int(kw.get("rev", 0)); rng, held, shown = sp.split(seed)
    Ls = np.array([text("+", a, b, rev) for a, b in held["+"]], np.int32); out, gs = s.answers(Ls, 1, 6, 3, want_g=True)                 # the guesses made where the first and the second answer byte are named
    o["guesses_right_add"] = dict(at_first_answer_byte_three_ahead=float(100 * (gs[:, 0, 0] == Ls[:, 8]).mean()), at_first_answer_byte_two_ahead=float(100 * (gs[:, 0, 1] == Ls[:, 7]).mean()), at_second_answer_byte_two_ahead=float(100 * (gs[:, 1, 1] == Ls[:, 8]).mean()))
    top = {}
    for fam in (3, 4, 5):
        T = s.T[fam * (s.K + 1):fam * (s.K + 1) + s.KA + 1]; used = s.Tu[fam * (s.K + 1):fam * (s.K + 1) + s.KA + 1] > 0; mx = np.where(used, T, -np.inf).max(1); order = np.argsort(-mx)[:8]
        top[s.FAMS[fam]] = [dict(delays=list(map(int, s.kinds[k])) if k < s.KA else [], largest_trust=round(float(mx[k]), 3)) for k in order if np.isfinite(mx[k])]
    o["top_trust_add"] = {}; o["top_trust_guess_kinds"] = top; o["trust_sum_by_family"] = {s.FAMS[f]: round(float(np.abs(s.T[f * (s.K + 1):(f + 1) * (s.K + 1)][s.Tu[f * (s.K + 1):(f + 1) * (s.K + 1)] > 0]).sum()), 2) for f in range(6)}
    o["cpu"] = round(time.process_time() - t0, 1); os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(o, open(os.path.join(HERE, "res", f"s_{kw['label']}.json"), "w")); return o

def check():
    for kw in (dict(m=2, D=6), dict(m=3, D=5, shrink=0, eta=0.01, Tg0=0.1), dict(m=2, D=7, ncap=4, rev=1)):
        rng, held, shown = sp.split(1); rev = kw.get("rev", 0); X = np.array([text("+-"[i % 2], *shown["+-"[i % 2]][i % 30], rev) for i in range(150)], np.int32)
        a = make(kw, Sys4); b = make(kw, Ref4); Pa = a.read(X[:70], True, True); Pa = np.concatenate([Pa, a.read(X[70:], True, True)]); Pb = b.read(X, True, True)
        Ls = np.array([text("+", *p, rev) for p in held["+"][:30]] + [text("-", *p, rev) for p in shown["-"][:30]], np.int32); same = all(all((u == v).all() for u, v in zip(a.answers(Ls, own, want_g=True), b.answers(Ls, own, want_g=True))) for own in (0, 1))
        d = max(np.abs(Pa - Pb).max(), np.abs(a.T - b.T).max()); ok = d < 1e-9 and same and a.size() == b.size()
        print(f"{kw}: {X.size} bytes, {a.K} plain kinds, {a.KA + 1} kinds keyed on each guess, {a.size()['cells']} cells that have counted, {a.size()['trusts_used']} trusts used: compiled against numpy (three sets of probabilities, trusts), largest difference {d:.1e}; answers and guesses the same: {same}; sizes the same: {a.size() == b.size()}", "PASSED" if ok else "FAILED")

if __name__ == "__main__":
    if sys.argv[1] == "check": check()
    else:
        kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw); sp.show(o); print("  guesses right:", {k: round(v, 1) for k, v in o["guesses_right_add"].items()}); print("  trust, summed sizes by family:", o["trust_sum_by_family"]); [print("  ", k, [(tuple(x["delays"]), x["largest_trust"]) for x in v[:6]]) for k, v in o["top_trust_guess_kinds"].items()]
