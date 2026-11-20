#!/usr/bin/env python3
"""nano/both, PLAN_1.md: one system with both kinds of cell on two-digit sums. The relay system of loop_small_case/cells2.py (copied unchanged, its
fast kernel) reads the stream beside the cells that count of nano/merge/merge.py (copied unchanged) and is one more sender to the two answering cells
of nano/table/gate.py (its kernel copied here and given one more sender; gate 0 only).
    python3 both.py check
    python3 both.py sums seed=1 arm=alone|joined [lines=400000] [block=50] [floor=0.01] [tag=...]
arm=alone:  nano/table/gate.py's system (must give that line's figures); arm=joined: the same with the relay system as one more sender, and the relay
system's own answers from the same run.
The relay system's message at a byte is the log of its branches cut off below at nothing, with `floor` added, and made to sum to one; its level is its
largest such share in ten bins. For a block of `block` lines the messages come from its strengths at the start of the block; then it learns the block.
It speaks at the bytes it learns (cells2.py teaches it the nine bytes of a line after the first; at a line's first byte it is silent).
Its trust in the first cell (one for each level) and its parts in the second cell (for each delay, level and byte at that delay) are born at nothing
and change by the same rule and step as every other sender's."""
import ctypes, hashlib, itertools, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import merge as G
import cells2 as C
NL = 40; XL = 10; PD = ctypes.POINTER(ctypes.c_double); PF = ctypes.POINTER(ctypes.c_float); PI = ctypes.POINTER(ctypes.c_int32)
SEEDS = (73129, 93683, 116089, 406364, 545353)

SRC = r'''
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#define NL 40
#define XL 10
#define KMAX 400
#define SMAX 65
#define DMAX 16
static inline int level(float n, float u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* nano/table/gate.py's mix_steps2 at gate 0, with one more sender whose message X[t] (S numbers) and level xlev[t] (0..9; below 0: silent) are given.
   First cell: its trust Tx[level]. Second cell: its parts Gx[delay][level][byte at that delay]. Same rule and step as the other senders.
   learn: 1 everything learns; 2 only the trusts and parts learn (nothing is counted). */
void mix_steps3(int64_t n, int K1, int S, const int32_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double *Tu,
                double eta, double e, int learn, int shrink, double *P, int D, const int32_t *cb, double *Gt, double *Gu, double *P2,
                const float *X, const int32_t *xlev, double *Tx, double *Txu, double *Gx, double *Gxu) {
    static double lt[4096], s[KMAX][SMAX]; double a[SMAX], p[SMAX], a2[SMAX], p2[SMAX]; static int lev[KMAX]; static int64_t gix[KMAX][DMAX]; int64_t xix[DMAX];
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int64_t t = 0; t < n; t++) {
        const int32_t *id = ids + t * K1; double mx = -1e300, z = 0; const int xl = (X && xlev[t] >= 0) ? xlev[t] : -1; const float *sx = X ? X + t * S : 0;
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int k = 0; k < K1; k++) {
            lev[k] = -1; if (id[k] < 0 || N[id[k]] < 1) continue;
            const float *c = C + (int64_t)id[k] * S; lev[k] = level(N[id[k]], U[id[k]]); const double tk = T[k * NL + lev[k]], den = log(N[id[k]] + S * e);
            for (int v = 0; v < S; v++) { const double sv = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den; s[k][v] = sv; a[v] += tk * sv; }
        }
        if (xl >= 0) { const double tk = Tx[xl]; for (int v = 0; v < S; v++) a[v] += tk * sx[v]; }
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        for (int v = 0; v < S; v++) a2[v] = a[v];
        for (int k = 0; k < K1; k++) if (lev[k] >= 0) {
            double tk = 0;
            for (int j = 0; j < D; j++) { gix[k][j] = (((int64_t)j * K1 + k) * NL + lev[k]) * S + cb[t * D + j]; tk += Gt[gix[k][j]]; }
            if (tk != 0) for (int v = 0; v < S; v++) a2[v] += tk * s[k][v];
        }
        if (xl >= 0) { double tk = 0; for (int j = 0; j < D; j++) { xix[j] = ((int64_t)j * XL + xl) * S + cb[t * D + j]; tk += Gx[xix[j]]; }
            if (tk != 0) for (int v = 0; v < S; v++) a2[v] += tk * sx[v]; }
        mx = -1e300; z = 0;
        for (int v = 0; v < S; v++) if (a2[v] > mx) mx = a2[v];
        for (int v = 0; v < S; v++) { p2[v] = exp(a2[v] - mx); z += p2[v]; }
        for (int v = 0; v < S; v++) p2[v] /= z;
        if (P2) for (int v = 0; v < S; v++) P2[t * S + v] = p2[v];
        if (learn) {
            for (int k = 0; k < K1; k++) if (lev[k] >= 0) {
                double g = s[k][y[t]], g2 = s[k][y[t]]; for (int v = 0; v < S; v++) { g -= p[v] * s[k][v]; g2 -= p2[v] * s[k][v]; }
                T[k * NL + lev[k]] += (shrink ? eta / sqrt(1.0 + Tu[k * NL + lev[k]]) : eta) * g; Tu[k * NL + lev[k]] += 1.0;
                for (int j = 0; j < D; j++) { const int64_t i = gix[k][j]; Gt[i] += (shrink ? eta / sqrt(1.0 + Gu[i]) : eta) * g2; Gu[i] += 1.0; }
            }
            if (xl >= 0) {
                double g = sx[y[t]], g2 = sx[y[t]]; for (int v = 0; v < S; v++) { g -= p[v] * sx[v]; g2 -= p2[v] * sx[v]; }
                Tx[xl] += (shrink ? eta / sqrt(1.0 + Txu[xl]) : eta) * g; Txu[xl] += 1.0;
                for (int j = 0; j < D; j++) { const int64_t i = xix[j]; Gx[i] += (shrink ? eta / sqrt(1.0 + Gxu[i]) : eta) * g2; Gxu[i] += 1.0; }
            }
            if (learn == 1) for (int k = 0; k < K1; k++) { const int64_t i = id[k] < -1 ? -2 - (int64_t)id[k] : id[k];
                if (i >= 0) { float *c = C + i * S; if (c[y[t]] == 0) U[i] += 1; c[y[t]] += 1; N[i] += 1; } }
        }
    }
}
'''
def kern():
    h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_both_{h}.so"); src = os.path.join(HERE, "kern_both.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd = ctypes.c_int, ctypes.c_double
    L.mix_steps3.argtypes = [ctypes.c_int64, ci, ci, PI, PI, PF, PF, PF, PD, PD, cd, cd, ci, ci, PD, ci, PI, PD, PD, PD, PF, PI, PD, PD, PD, PD]; L.mix_steps3.restype = None; return L
p32 = lambda a: a.ctypes.data_as(PI); pf = lambda a: a.ctypes.data_as(PF); pd = lambda a: a.ctypes.data_as(PD)

class Sys3(G.Sys):
    """nano/merge's system with nano/table/gate.py's second answering cell (gate 0) and one more, given, sender. s.cell: whose answer say() gives.
    s.xq = (X, xlev): the given sender's messages and levels for the bytes about to be read or answered, in order; None: no such sender."""
    def __init__(s, *a, **kw):
        super().__init__(*a, **kw); s.L3 = kern(); assert s.D <= 16
        s.Gt = np.zeros((s.D, s.K1, NL, s.S)); s.Gu = np.zeros_like(s.Gt); s.cell = 2; s.Tx = np.zeros(XL); s.Txu = np.zeros(XL); s.Gx = np.zeros((s.D, XL, s.S)); s.Gxu = np.zeros_like(s.Gx)
        s.xq = None; s.xpos = 0; s.only_trust = False
    def give(s, X, xlev): s.xq = (np.ascontiguousarray(X, np.float32), np.ascontiguousarray(xlev, np.int32)) if X is not None else None; s.xpos = 0
    def _run(s, ctx, y, learn, want_p):
        P = np.zeros((len(ctx), s.S)) if want_p else None
        for a in range(0, len(ctx), 50000):
            idd = s.ids(ctx[a:a + 50000], create=learn and not s.only_trust); yy = np.ascontiguousarray(y[a:a + 50000].astype(np.int32)) if y is not None else np.zeros(len(idd), np.int32)
            cb = np.ascontiguousarray(ctx[a:a + 50000, :s.D].astype(np.int32)); P1 = np.zeros((len(idd), s.S)) if want_p else None; P2 = np.zeros((len(idd), s.S)) if want_p else None
            X = xl = None
            if s.xq is not None: X = s.xq[0][s.xpos + a:s.xpos + a + len(idd)]; xl = s.xq[1][s.xpos + a:s.xpos + a + len(idd)]; assert len(X) == len(idd)
            s.L3.mix_steps3(len(idd), s.K1, s.S, p32(idd), p32(yy), pf(s.C), pf(s.N), pf(s.U), pd(s.T), pd(s.Tu), s.eta, s.e, (2 if s.only_trust else 1) if learn else 0, s.shrink, pd(P1) if want_p else None,
                            s.D, p32(cb), pd(s.Gt), pd(s.Gu), pd(P2) if want_p else None, pf(X) if X is not None else None, p32(xl) if X is not None else None, pd(s.Tx), pd(s.Txu), pd(s.Gx), pd(s.Gxu))
            if want_p: P[a:a + 50000] = P2 if s.cell == 2 else P1
        s.xpos += len(ctx); return P
    def find(s):
        old = {(sl["k"], sl["X"]): p for p, sl in enumerate(s.slots) if sl is not None}; G0, U0 = s.Gt.copy(), s.Gu.copy(); super().find(); s.Gt[:, s.KB:] = 0; s.Gu[:, s.KB:] = 0
        for p, sl in enumerate(s.slots):
            if sl is not None and (sl["k"], sl["X"]) in old: q = old[(sl["k"], sl["X"])]; s.Gt[:, s.KB + p] = G0[:, s.KB + q]; s.Gu[:, s.KB + p] = U0[:, s.KB + q]

# ---------------------------------------------------------------- the reference in numpy
def steps3_np(ids, y, cb, C, N, U, T, Tu, Gt, Gu, eta, e, X, xlev, Tx, Txu, Gx, Gxu, learn=1, shrink=False):
    S = C.shape[1]; n = len(y); P1 = np.zeros((n, S)); P2 = np.zeros((n, S)); D = Gt.shape[0]; st = lambda u: eta / np.sqrt(1.0 + u) if shrink else eta
    for t in range(n):
        a = np.zeros(S); ss = []; gv = [int(cb[t, j]) for j in range(D)]; xl = int(xlev[t]) if X is not None else -1; sx = X[t].astype(np.float64) if xl >= 0 else None
        for k, i in enumerate(ids[t]):
            if i >= 0 and N[i] >= 1:
                l = G.level_np(N[i], U[i]); sv = np.log(C[i].astype(np.float64) + e) - np.log(N[i] + S * e); ss.append((k, l, sv)); a += T[k, l] * sv
        if xl >= 0: a += Tx[xl] * sx
        p = np.exp(a - a.max()); p /= p.sum(); P1[t] = p; a2 = a.copy()
        for k, l, sv in ss: a2 += sum(Gt[j, k, l, gv[j]] for j in range(D)) * sv
        if xl >= 0: a2 += sum(Gx[j, xl, gv[j]] for j in range(D)) * sx
        p2 = np.exp(a2 - a2.max()); p2 /= p2.sum(); P2[t] = p2
        if learn:
            for k, l, sv in ss:
                T[k, l] += st(Tu[k, l]) * (sv[y[t]] - p @ sv); Tu[k, l] += 1.0; g2 = sv[y[t]] - p2 @ sv
                for j in range(D): Gt[j, k, l, gv[j]] += st(Gu[j, k, l, gv[j]]) * g2; Gu[j, k, l, gv[j]] += 1.0
            if xl >= 0:
                Tx[xl] += st(Txu[xl]) * (sx[y[t]] - p @ sx); Txu[xl] += 1.0; g2 = sx[y[t]] - p2 @ sx
                for j in range(D): Gx[j, xl, gv[j]] += st(Gxu[j, xl, gv[j]]) * g2; Gxu[j, xl, gv[j]] += 1.0
            if learn == 1:
                for i in ids[t]:
                    if i < -1: i = -2 - i
                    if i >= 0:
                        if C[i, y[t]] == 0: U[i] += 1
                        C[i, y[t]] += 1; N[i] += 1
    return P1, P2

# ---------------------------------------------------------------- the relay system as a sender
def message(a, floor):
    """Branches (rows) to messages: the log of (the branch cut off below at nothing, plus floor), made to sum to one; and the level."""
    p = np.maximum(a, 0.0) + floor; p /= p.sum(1, keepdims=True); return np.log(p).astype(np.float32), np.minimum((p.max(1) * 10).astype(np.int32), 9).astype(np.int32)

def branches(r, hist):
    """The relay system's branches for many pasts at once (hist: rows of its ten delay lines, newest first); cells2.py's arithmetic."""
    return r.out(r.W1.T[C.OFF + hist].sum(1) + r.b1)

def relay_read(r, lines, tail, block, floor, learn=True):
    """The relay system reads `lines` (n by 10) after the ten bytes `tail`. Returns its messages and levels for every byte of the lines (from its strengths
    at the start of each block of `block` lines; level -1 at a line's first byte, which it is not taught), and the new tail."""
    n = len(lines); flat = np.concatenate([np.asarray(tail, np.int64), np.asarray(lines, np.int64).ravel()]); X = np.zeros((n * 10, 14), np.float32); xl = np.full(n * 10, -1, np.int32)
    idx = np.arange(n * 10); H = flat[(idx[:, None] + 10) - 1 - np.arange(C.D)[None, :]]                      # for target byte i: the ten bytes before it, newest first
    for b0 in range(0, n, block):
        b1 = min(n, b0 + block); sl = slice(b0 * 10, b1 * 10); m, l = message(branches(r, H[sl]), floor); X[sl] = m; xl[sl] = l
        if learn:
            part = np.asarray(lines[b0:b1], np.int32); r.bx = part.ravel(); r.by = np.concatenate([part[:, 1:], np.full((len(part), 1), -1, np.int32)], 1).ravel(); r.flush()
    xl[::10] = -1; return X, xl, flat[-10:]

def new_relay(seed):
    r = C.Sys(seed); r.fast = 3; return r                                                                  # cells2.py's plain system, its own numbers

def new_counting(seed, lines, mode="classes"):
    D = 9; kinds = [c for m in range(5) for c in itertools.combinations(range(1, D + 1), m)]              # nano/merge's run_sums at its defaults (that line's frozen setting)
    return Sys3(14, D, kinds, 0.1, 2.0, 0.02, True, P=32, eJ=0.02, mode=mode, seed=seed, half_at=lines * 7 * 5 // 10, M=16, sweeps=2)

# ---------------------------------------------------------------- the score: never-shown lines, the system's own bytes, after a read line
def past_of(own, prev, t, nd):
    """The nd bytes before byte t of each line (newest first); what reaches before the line is the line `prev` (None: line ends)."""
    h = np.full((len(own), nd), C.END, np.int64)
    for d in range(1, nd + 1):
        if t - d >= 0: h[:, d - 1] = own[:, t - d]
        elif prev is not None: h[:, d - 1] = prev[:, 10 + t - d]
    return h

def answer(cs, r, Ls, prev, floor, who, mute=False):
    """Whole answers in the system's own bytes. who: 'relay' (the relay system alone), 1 or 2 (the first or second answering cell; with the relay system
    as a sender if r is given and not mute)."""
    own = Ls.copy()
    for t in (6, 7, 8):
        if who == "relay": own[:, t] = branches(r, past_of(own, prev, t, C.D)).argmax(1); continue
        if r is not None:
            m, l = message(branches(r, past_of(own, prev, t, C.D)), floor)
            if mute: l[:] = -1
            cs.give(m, l)
        cs.cell = who; own[:, t] = cs.say(past_of(own, prev, t, cs.D)).argmax(1); cs.give(None, None)
    ok = own[:, 6:9] == Ls[:, 6:9]; return float(100 * ok.all(1).mean()), [float(100 * ok[:, i].mean()) for i in range(3)], ok.all(1)

def reading(cs, r, Ls, prevs, floor, arms):
    out = {}
    for name, who, mute in arms:
        e = answer(cs, r, Ls, None, floor, who, mute); a = [answer(cs, r, Ls, pv, floor, who, mute) for pv in prevs]
        out[name] = dict(empty=e[0], after=float(np.mean([q[0] for q in a])), after_place=np.mean([q[1] for q in a], 0).tolist())
    return out

def split(seed):
    rng = np.random.default_rng([seed, 99]); PP = {}
    for op in "+-": o = rng.permutation(10000); PP[op] = [(int(q) // 100, int(q) % 100) for q in o]
    return rng, PP

def run_sums(kw):
    seed = int(kw.get("seed", 1)); lines = int(kw.get("lines", 400000)); arm = kw.get("arm", "joined"); block = int(kw.get("block", 50)); floor = float(kw.get("floor", 0.01)); t0 = time.process_time()
    rng, PP = split(seed); held = PP["+"][:500]; shown = PP["+"][2000:]; TAB = np.array([C.text("+", a, b) for a, b in shown], np.int64); Ls = np.array([C.text("+", a, b) for a, b in held])
    g = np.random.default_rng([seed, 7]); prevs = [TAB[g.integers(8000, size=500)] for _ in range(5)]
    marks = sorted(set([m for m in (5000, 10000, 20000, 40000) if m <= lines] + list(range(50000, lines + 1, 50000)) + [lines]))
    cs = new_counting(seed, lines, kw.get("mode", "classes")) if arm != "relay" else None; r = new_relay(seed) if arm != "alone" else None
    arms = dict(alone=[("cell1", 1, False), ("cell2", 2, False)], relay=[("relay", "relay", False)],
                joined=[("relay", "relay", False), ("cell1", 1, False), ("cell2", 2, False), ("cell1_relay_silent", 1, True), ("cell2_relay_silent", 2, True)])[arm]
    tail9 = np.full(9, C.END, np.int64); tail10 = np.full(10, C.END, np.int64); done = 0; rows = []; t_relay = 0.0
    for mk in marks:
        X = TAB[rng.integers(8000, size=mk - done)]; done = mk
        if r is not None: t1 = time.process_time(); M, xl, tail10 = relay_read(r, X, tail10, block, floor); t_relay += time.process_time() - t1
        if cs is not None:
            x = np.concatenate([tail9, X.ravel()]); cs.give(M, xl) if r is not None else cs.give(None, None); cs.learn(G.ctx_of(x, 9, len(x), 9), x[9:]); cs.give(None, None); tail9 = x[-9:]
        row = dict(lines=mk); row.update(reading(cs, r, Ls, prevs, floor, arms))
        if r is not None and cs is not None:
            row["relay_trust_first_cell"] = cs.Tx.tolist(); row["relay_trust_uses"] = cs.Txu.tolist(); row["relay_parts_sum_abs_by_level"] = np.abs(cs.Gx).sum((0, 2)).tolist()
            row["relay_parts_mean_total_by_level"] = [float((cs.Gx[:, l] * cs.Gxu[:, l]).sum() / max(1.0, cs.Gxu[:, l].sum()) * cs.D) for l in range(XL)]
            row["relay_levels_on_last_block"] = np.bincount(xl[xl >= 0][-45000:], minlength=XL).tolist()
        if cs is not None: row["cells"] = cs.sizes()["cells"]
        rows.append(row); print(f"seed {seed} {arm} {mk}: " + " ".join(f"{k} {v['after']:.1f}" for k, v in row.items() if isinstance(v, dict)), flush=True)
    res = dict(task="sums", args=kw, seed=seed, arm=arm, rows=rows, cpu=round(time.process_time() - t0, 1), cpu_relay=round(t_relay, 1))
    if cs is not None: res["sizes"] = cs.sizes(); res["numbers_of_the_answering_cells"] = dict(first=int(cs.T.size + cs.Tx.size), second=int(cs.Gt.size + cs.Gx.size), second_used=int((cs.Gu > 0).sum() + (cs.Gxu > 0).sum()))
    if r is not None: res["relay_numbers"] = int(r.W1.size + r.b1.size + r.W2.size + r.b2.size + r.R.size)
    return res, cs, r

# ---------------------------------------------------------------- checks
def check():
    ok = True; rng = np.random.default_rng(5); L = kern(); L1 = G.kern()
    for S, shrink, D, withx, learn in ((14, True, 9, True, 1), (14, True, 9, False, 1), (65, False, 6, True, 1), (14, True, 9, True, 2)):
        K1, n, nc = 9, 3000, 400; ids = rng.integers(-1, nc, (n, K1)).astype(np.int32); sil = rng.random((n, K1)) < 0.1; ids[sil & (ids >= 0)] = -2 - ids[sil & (ids >= 0)]; y = rng.integers(0, S, n).astype(np.int32)
        cb = rng.integers(0, S, (n, D)).astype(np.int32); X = rng.normal(0, 1, (n, S)); X = (X - np.log(np.exp(X).sum(1, keepdims=True))).astype(np.float32); xlev = rng.integers(-1, 10, n).astype(np.int32)
        def fresh():
            Cc = np.zeros((nc, S), np.float32); Nn = np.zeros(nc, np.float32)
            if learn == 2: Cc[:] = rng.integers(0, 3, (nc, S)); Nn[:] = Cc.sum(1)
            return [Cc, Nn, (Cc > 0).sum(1).astype(np.float32), np.full((K1, NL), 0.1), np.zeros((K1, NL)), np.zeros((D, K1, NL, S)), np.zeros((D, K1, NL, S)), np.zeros(XL), np.zeros(XL), np.zeros((D, XL, S)), np.zeros((D, XL, S))]
        A = fresh(); B = [q.copy() for q in A]; Cm = [q.copy() for q in A]; P1 = np.zeros((n, S)); P2 = np.zeros((n, S))
        L.mix_steps3(n, K1, S, p32(ids), p32(y), pf(A[0]), pf(A[1]), pf(A[2]), pd(A[3]), pd(A[4]), 0.05, 0.3, learn, int(shrink), pd(P1), D, p32(cb), pd(A[5]), pd(A[6]), pd(P2),
                     pf(X) if withx else None, p32(xlev) if withx else None, pd(A[7]), pd(A[8]), pd(A[9]), pd(A[10]))
        R1, R2 = steps3_np(ids, y, cb, B[0], B[1], B[2], B[3], B[4], B[5], B[6], 0.05, 0.3, X if withx else None, xlev, B[7], B[8], B[9], B[10], learn, shrink)
        d = max([np.abs(P1 - R1).max(), np.abs(P2 - R2).max()] + [np.abs(u - v).max() for u, v in zip(A, B)]); good = d < 1e-9 and np.abs(P2 - P1).max() > 1e-3; extra = ""
        if withx: good &= np.abs(A[7]).max() > 1e-3 and np.abs(A[9]).max() > 1e-3
        if learn == 2: good &= bool((A[0] == Cm[0]).all())
        if not withx and learn == 1:                                                                       # with no given sender: the first cell against nano/merge's own kernel
            Pm = np.zeros((n, S)); L1.mix_steps(n, K1, S, p32(ids), p32(y), pf(Cm[0]), pf(Cm[1]), pf(Cm[2]), pd(Cm[3]), pd(Cm[4]), 0.05, 0.3, 1, int(shrink), pd(Pm))
            d1 = max(np.abs(P1 - Pm).max(), np.abs(A[3] - Cm[3]).max(), np.abs(A[0] - Cm[0]).max()); good &= d1 == 0; extra = f"; the first cell against nano/merge's kernel {d1:.1e}"
        ok &= bool(good); print(f"two answering cells and a given sender ({'given' if withx else 'absent'}), S = {S}, shrinking step = {shrink}, learn = {learn}, {n} bytes: compiled against numpy, largest difference {d:.1e}{extra}", "PASSED" if good else "FAILED")
    # the relay system read in blocks through its fast kernel against cells2.py's numpy loop, and its messages against a plain computation
    lines = np.array([C.text("+", int(a), int(b)) for a, b in rng.integers(0, 100, (300, 2))]); ra = new_relay(3); rb = C.Sys(3); tail = np.full(10, C.END, np.int64)
    X, xl, tail = relay_read(ra, lines, tail, 50, 0.01); dm = 0.0; hist = [C.END] * 10; nlev = 0
    for i, Ln in enumerate(lines.tolist()):
        if i % 50 == 0: snap = C.Sys(3); snap.W1, snap.b1, snap.W2, snap.b2 = rb.W1.copy(), rb.b1.copy(), rb.W2.copy(), rb.b2.copy()
        for t in range(10):
            if t > 0:
                a = snap.fwd(hist)[3]; p = np.maximum(a, 0) + 0.01; p /= p.sum(); dm = max(dm, float(np.abs(np.log(p) - X[i * 10 + t]).max())); nlev += int(min(int(p.max() * 10), 9) != xl[i * 10 + t])
            else: nlev += int(xl[i * 10] != -1)
            hist.insert(0, Ln[t]); hist.pop()
        rb.line(Ln)
    dw = max(np.abs(ra.W1 - rb.W1).max(), np.abs(ra.W2 - rb.W2).max(), np.abs(ra.R - rb.R).max()); good = dw < 1e-9 and dm < 1e-4 and nlev <= 2; ok &= good
    print(f"the relay system, 300 lines in blocks of 50: strengths, fast kernel against cells2.py's numpy loop, largest difference {dw:.1e}; messages against a plain computation from the strengths at the start of the block {dm:.1e} (kept in single precision); levels that differ {nlev}", "PASSED" if good else "FAILED")
    return ok

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=") for a in sys.argv[2:]); res, cs, r = run_sums(kw)
    tag = kw.get("tag") or "sums_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items())); os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(res, open(os.path.join(HERE, "res", tag + ".json"), "w"))
    print(tag, "| cpu", res["cpu"], flush=True)
