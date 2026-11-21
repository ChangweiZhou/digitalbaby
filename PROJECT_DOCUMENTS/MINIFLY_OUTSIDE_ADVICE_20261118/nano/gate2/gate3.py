#!/usr/bin/env python3
"""Line nano/gate2, PLAN_5.md: gate2.py with one addition: a second cell may, while it learns, hear each sender with probability one half.
Everything else is gate2.py (PLAN_1.md to PLAN_4.md): a second answering cell, written from the description in ../table/PLAN_15.md (sections 2 and 4)
and not from its program.  The base system is ../merge/merge.py, imported as it is.
    python3 gate3.py check
    python3 gate3.py run seed=1 [plan=add:40000[,sub:20000]] [mode=classes|none] [nd=2] [tag=...] [nG=2]
The first cell is merge.py's answering cell (the same arithmetic in the same order).  Second cells (nG of them, independent of one another, none
touching the first): cell 0 has for each delay a part of the trust indexed by (kind, level, the byte at that delay); cell 1 is the control, whose
parts read byte value 0 whatever stands there; cells 2 and 3 are the same two, but while learning each hears every sender with probability one half
(and doubles what it hears), and only the senders heard change their parts; when answering they hear all senders.  Scores: never-shown lines, each cell continuing its own answer bytes, after a read line and with
an empty past (loop_small_case/ERRATA_20261120.md, section 3); subtraction on the clean pairs (section 1)."""
import ctypes, hashlib, itertools, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); BUILD = os.path.join(HERE, "build"); os.makedirs(BUILD, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "merge"))
import merge as G
G.HERE = BUILD                                   # merge.py's compiled kernel is built in this line's folder; nothing is written in nano/merge
NL = G.NL; PD, PF, PI = G.PD, G.PF, G.PI; p32, pf, pd = G.p32, G.pf, G.pd

SRC = r'''
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define NL 40
#define KMAX 400
#define SMAX 65
#define GMAX 4
static inline int level(float n, float u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* The first cell: merge.py's mix_steps, line for line.  Second cells g = 0..nG-1: a2(v) = a1(v) + sum_k w_k s_k(v),
   w_k = sum_j Gt[g][k][lev_k][j][b_j], b_j the byte at delay j+1 (or 0 if gconst[g]).  Each entry read changes by its step times the cell's own mismatch. */
static inline uint64_t xs64(uint64_t *x) { *x ^= *x >> 12; *x ^= *x << 25; *x ^= *x >> 27; return *x * 0x2545F4914F6CDD1DULL; }
void mix3_steps(int64_t n, int K1, int S, int D, int nG, const int32_t *ids, const int32_t *y, const int32_t *ctx, float *C, float *N, float *U, double *T, double *Tu,
                double *Gt, double *Gu, const int32_t *gconst, const int32_t *gdrop, uint64_t *rs, double eta, double e, int learn, int shrink, double *P, double *P2) {
    static double lt[4096], s[KMAX][SMAX]; double a[SMAX], p[SMAX], a2[SMAX], p2[GMAX][SMAX]; static int lev[KMAX]; static double tkv[KMAX]; static uint64_t hb[GMAX][(KMAX + 63) / 64];
#define HEARD(g, k) ((hb[g][(k) >> 6] >> ((k) & 63)) & 1)
    const int64_t gk = (int64_t)NL * D * S, gg = (int64_t)K1 * gk;
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int64_t t = 0; t < n; t++) {
        const int32_t *id = ids + t * K1; const int32_t *cx = ctx + t * D; double mx = -1e300, z = 0;
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int k = 0; k < K1; k++) {
            lev[k] = -1; if (id[k] < 0 || N[id[k]] < 1) continue;
            const float *c = C + (int64_t)id[k] * S; lev[k] = level(N[id[k]], U[id[k]]); const double tk = T[k * NL + lev[k]], den = log(N[id[k]] + S * e);
            for (int v = 0; v < S; v++) { const double sv = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den; s[k][v] = sv; a[v] += tk * sv; }
            tkv[k] = tk;
        }
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        for (int g = 0; g < nG; g++) {
            double m2 = -1e300, z2 = 0; const int dr = learn && gdrop[g];
            if (dr) for (int b = 0; b < (K1 + 63) / 64; b++) hb[g][b] = xs64(rs);
            for (int v = 0; v < S; v++) a2[v] = dr ? 0 : a[v];
            for (int k = 0; k < K1; k++) if (lev[k] >= 0) {
                const double *gp = Gt + g * gg + k * gk + (int64_t)lev[k] * D * S; double w = 0;
                if (dr && !HEARD(g, k)) continue;
                for (int j = 0; j < D; j++) w += gp[j * S + (gconst[g] ? 0 : cx[j])];
                if (dr) w = 2.0 * (tkv[k] + w);
                for (int v = 0; v < S; v++) a2[v] += w * s[k][v];
            }
            for (int v = 0; v < S; v++) if (a2[v] > m2) m2 = a2[v];
            for (int v = 0; v < S; v++) { p2[g][v] = exp(a2[v] - m2); z2 += p2[g][v]; }
            for (int v = 0; v < S; v++) p2[g][v] /= z2;
            if (P2) for (int v = 0; v < S; v++) P2[((int64_t)g * n + t) * S + v] = p2[g][v];
        }
        if (learn) {
            for (int k = 0; k < K1; k++) if (lev[k] >= 0) {
                double g = s[k][y[t]]; for (int v = 0; v < S; v++) g -= p[v] * s[k][v];
                T[k * NL + lev[k]] += (shrink ? eta / sqrt(1.0 + Tu[k * NL + lev[k]]) : eta) * g; Tu[k * NL + lev[k]] += 1.0;
            }
            for (int g = 0; g < nG; g++) for (int k = 0; k < K1; k++) if (lev[k] >= 0 && !(gdrop[g] && !HEARD(g, k))) {
                double m = s[k][y[t]]; for (int v = 0; v < S; v++) m -= p2[g][v] * s[k][v];
                const int64_t o = g * gg + k * gk + (int64_t)lev[k] * D * S;
                for (int j = 0; j < D; j++) { const int64_t q = o + j * S + (gconst[g] ? 0 : cx[j]); Gt[q] += (shrink ? eta / sqrt(1.0 + Gu[q]) : eta) * m; Gu[q] += 1.0; }
            }
            for (int k = 0; k < K1; k++) { const int64_t i = id[k] < -1 ? -2 - (int64_t)id[k] : id[k];
                if (i >= 0) { float *c = C + i * S; if (c[y[t]] == 0) U[i] += 1; c[y[t]] += 1; N[i] += 1; } }
        }
    }
}
'''
def kern3():
    h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(BUILD, f"kern_gate3_{h}.so"); src = os.path.join(BUILD, "kern_gate3.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd = ctypes.c_int, ctypes.c_double
    L.mix3_steps.argtypes = [ctypes.c_int64, ci, ci, ci, ci, PI, PI, PI, PF, PF, PF, PD, PD, PD, PD, PI, PI, ctypes.POINTER(ctypes.c_uint64), cd, cd, ci, ci, PD, PD]; L.mix3_steps.restype = None; return L

# ---------------------------------------------------------------- the reference in numpy
M64 = (1 << 64) - 1
def xs64_py(st):
    """The kernel's random stream (xorshift64*), in plain Python.  st: a list holding the state."""
    x = st[0]; x ^= x >> 12; x ^= (x << 25) & M64; x ^= x >> 27; st[0] = x; return (x * 0x2545F4914F6CDD1D) & M64

def steps3_np(ids, y, ctx, C, N, U, T, Tu, Gt, Gu, gconst, gdrop, st, eta, e, learn=True, shrink=False):
    """Gt, Gu: (nG, K1, NL, D, S).  Returns P (n, S) and P2 (nG, n, S)."""
    S = C.shape[1]; nG, D = Gt.shape[0], Gt.shape[3]; P = np.zeros((len(y), S)); P2 = np.zeros((nG, len(y), S)); jj = np.arange(D)
    for t in range(len(y)):
        a = np.zeros(S); ss = []
        for k, i in enumerate(ids[t]):
            if i >= 0 and N[i] >= 1:
                l = G.level_np(N[i], U[i]); sv = np.log(C[i].astype(np.float64) + e) - np.log(N[i] + S * e); ss.append((k, l, sv)); a += T[k, l] * sv
        p = np.exp(a - a.max()); p /= p.sum(); P[t] = p; p2s = []
        for g in range(nG):
            b = np.zeros(D, np.int64) if gconst[g] else ctx[t].astype(np.int64); dr = learn and gdrop[g]; heard = None
            if dr:
                hb = [xs64_py(st) for _ in range((ids.shape[1] + 63) // 64)]; heard = lambda k, hb=hb: (hb[k >> 6] >> (k & 63)) & 1; a2 = np.zeros(S)
                for k, l, sv in ss:
                    if heard(k): a2 += 2.0 * (T[k, l] + Gt[g, k, l, jj, b].sum()) * sv
            else:
                a2 = a.copy()
                for k, l, sv in ss: a2 += Gt[g, k, l, jj, b].sum() * sv
            q = np.exp(a2 - a2.max()); q /= q.sum(); P2[g, t] = q; p2s.append((b, q, heard))
        if learn:
            for k, l, sv in ss: T[k, l] += (eta / np.sqrt(1.0 + Tu[k, l]) if shrink else eta) * (sv[y[t]] - p @ sv); Tu[k, l] += 1.0
            for g, (b, q, heard) in enumerate(p2s):
                for k, l, sv in ss:
                    if heard is not None and not heard(k): continue
                    m = sv[y[t]] - q @ sv; Gt[g, k, l, jj, b] += (eta / np.sqrt(1.0 + Gu[g, k, l, jj, b]) if shrink else eta) * m; Gu[g, k, l, jj, b] += 1.0
            for i in ids[t]:
                if i < -1: i = -2 - i
                if i >= 0:
                    if C[i, y[t]] == 0: U[i] += 1
                    C[i, y[t]] += 1; N[i] += 1
    return P, P2

# ---------------------------------------------------------------- the system: merge.py's, with second cells beside its answering cell
class Sys2(G.Sys):
    def __init__(s, *a, gconst=(0, 1, 0, 1), gdrop=(0, 0, 1, 1), dseed=0, **k):
        super().__init__(*a, **k); s.L2 = kern3(); s.gconst = np.array(gconst, np.int32); s.gdrop = np.array(gdrop, np.int32); s.nG = len(gconst); s.rs = np.array([(0x9E3779B97F4A7C15 ^ (int(dseed) + 1)) & M64], np.uint64)
        s.Gt = np.zeros((s.nG, s.K1, NL, s.D, s.S)); s.Gu = np.zeros((s.nG, s.K1, NL, s.D, s.S)); s.P2 = None
    def _run(s, ctx, y, learn, want_p):
        P = np.zeros((len(ctx), s.S)) if want_p else None; s.P2 = np.zeros((s.nG, len(ctx), s.S)) if want_p else None
        for a in range(0, len(ctx), 50000):
            idd = s.ids(ctx[a:a + 50000], create=learn); yy = np.ascontiguousarray(y[a:a + 50000].astype(np.int32)) if y is not None else np.zeros(len(idd), np.int32)
            cc = np.ascontiguousarray(ctx[a:a + 50000].astype(np.int32)); Pp = np.zeros((len(idd), s.S)) if want_p else None; Pq = np.zeros((s.nG, len(idd), s.S)) if want_p else None
            s.L2.mix3_steps(len(idd), s.K1, s.S, s.D, s.nG, p32(idd), p32(yy), p32(cc), pf(s.C), pf(s.N), pf(s.U), pd(s.T), pd(s.Tu), pd(s.Gt), pd(s.Gu), p32(s.gconst), p32(s.gdrop), s.rs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                            s.eta, s.e, int(learn), s.shrink, pd(Pp) if want_p else None, pd(Pq) if want_p else None)
            if want_p: P[a:a + 50000] = Pp; s.P2[:, a:a + 50000] = Pq
        return P
    def say_all(s, ctx):
        """The probabilities of every answering cell: [first, second cell 0, second cell 1, ...]."""
        P = s._run(ctx, None, False, True); return [P] + [s.P2[g] for g in range(s.nG)]
    def find(s):
        """merge.py's finding of classes; the second cells' parts for a class kind follow the first cell's trust for it."""
        old = {(sl["k"], sl["X"]): p for p, sl in enumerate(s.slots) if sl is not None}; G0, U0 = s.Gt[:, s.KB:].copy(), s.Gu[:, s.KB:].copy()
        super().find(); s.Gt[:, s.KB:] = 0.0; s.Gu[:, s.KB:] = 0.0
        for p, sl in enumerate(s.slots):
            if sl is not None and (sl["k"], sl["X"]) in old: q = old[(sl["k"], sl["X"])]; s.Gt[:, s.KB + p] = G0[:, q]; s.Gu[:, s.KB + p] = U0[:, q]

# ---------------------------------------------------------------- texts, splits, scores
SI, END = G.SI, G.END
def text(op, a, b, nd=2):
    if op == "-" and a < b: a, b = b, a                      # loop_small_case/cells2.py: the larger number first
    return [SI[c] for c in f"{a:0{nd}d}{op}{b:0{nd}d}={(a + b if op == '+' else a - b):0{nd + 1}d}\n".encode()]

def score_cells(s, Ls, prev, nd=2):
    """Ls: (n, W) held-out lines.  prev: None (an empty past) or (n, W) lines standing before them.  Each cell continues its own answer bytes.
    Returns for each cell (whole answers right %, [by place %])."""
    n, W = Ls.shape; D = s.D; a0 = 2 * nd + 2; nc = 1 + s.nG; own = [Ls.copy() for _ in range(nc)]
    for t in range(a0, a0 + nd + 1):
        new = []
        for c in range(nc):
            ctx = np.full((n, D), END, np.int64)
            for d in range(1, D + 1):
                if t - d >= 0: ctx[:, d - 1] = own[c][:, t - d]
                elif prev is not None: ctx[:, d - 1] = prev[:, W + t - d]
            new.append(s.say_all(ctx)[c].argmax(1))
        for c in range(nc): own[c][:, t] = new[c]
    out = []
    for c in range(nc):
        ok = own[c][:, a0:a0 + nd + 1] == Ls[:, a0:a0 + nd + 1]; out.append((float(100 * ok.all(1).mean()), [float(100 * ok[:, i].mean()) for i in range(nd + 1)]))
    return out

CELLS = ("first", "second", "control", "second_half", "control_half")
def reading(s, Ls, prevs, nd=2):
    """prevs: dict name -> list of draws of lines standing before.  Returns {cell: {empty: x, after_<name>: mean, ...}}."""
    r = {c: {} for c in CELLS[:1 + s.nG]}; e = score_cells(s, Ls, None, nd)
    for i, c in enumerate(r): r[c]["empty"] = e[i][0]; r[c]["empty_place"] = e[i][1]
    for name, draws in prevs.items():
        sc = [score_cells(s, Ls, pv, nd) for pv in draws]
        for i, c in enumerate(r): r[c]["after_" + name] = float(np.mean([q[i][0] for q in sc])); r[c]["after_" + name + "_draws"] = [q[i][0] for q in sc]; r[c]["after_" + name + "_place"] = np.mean([q[i][1] for q in sc], 0).tolist()
    return r

def run(kw):
    seed = int(kw.get("seed", 1)); mode = kw.get("mode", "classes"); nd = int(kw.get("nd", 2)); plan = [(p.split(":")[0], int(p.split(":")[1])) for p in kw.get("plan", "add:40000").split(",")]
    nG = int(kw.get("nG", 4)); t0 = time.process_time(); D = 9; rng = np.random.default_rng([seed, 99]); held, TAB, drawer = {}, {}, {}
    if nd == 2:                                               # merge.run_sums' split for addition, which is cells2.py's; the second permutation is cells2.py's split for subtraction
        for op in "+-":
            o = rng.permutation(10000); pairs = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = pairs[:500]; TAB[op] = np.array([text(op, a, b) for a, b in pairs[2000:]], np.int64)
        for op in "+-": drawer[op] = (lambda k, op=op: TAB[op][rng.integers(8000, size=k)])
        shown_sub = {tuple(l) for l in TAB["-"].tolist()}; clean = [pr for pr in held["-"] if tuple(text("-", *pr)) not in shown_sub]
        g7 = np.random.default_rng([seed, 7]); prevs = {"add": [TAB["+"][g7.integers(8000, size=500)] for _ in range(5)]}      # the draws of nano/table/past_cells.py
        if any(n == "sub" for n, _ in plan):
            g8 = np.random.default_rng([seed, 8]); prevs["sub"] = [TAB["-"][g8.integers(8000, size=500)] for _ in range(5)]
            prevs_clean = {k: [pv[:len(clean)] for pv in v] for k, v in prevs.items()}
    else:                                                     # merge.run_sums' split for longer sums (nano/ahead2/ahead3.py's)
        assert all(n == "add" for n, _ in plan); n10 = 10 ** nd; o = rng.permutation(n10 * n10); nh = n10 * n10 // 5; held["+"] = [(int(q) // n10, int(q) % n10) for q in o[:500]]; pool_ = o[nh:]
        tx = lambda q: np.array([text("+", int(v) // n10, int(v) % n10, nd) for v in q], np.int64)
        drawer["+"] = lambda k: tx(pool_[rng.integers(len(pool_), size=k)])
        g7 = np.random.default_rng([seed, 7]); prevs = {"add": [tx(pool_[g7.integers(len(pool_), size=500)]) for _ in range(5)]}
    LsA = np.array([text("+", a, b, nd) for a, b in held["+"]]); LsS = np.array([text("-", a, b) for a, b in clean]) if nd == 2 and "sub" in prevs else None
    kinds = [c for m in range(5) for c in itertools.combinations(range(1, D + 1), m)]
    s = Sys2(14, D, kinds, 0.1, 2.0, 0.02, True, P=int(kw.get("P", 32)), eJ=float(kw.get("eJ", 0.02)), mode=mode, seed=seed, half_at=0, keep=int(kw.get("keep", 0)), gconst=(0, 1, 0, 1)[:nG], gdrop=(0, 0, 1, 1)[:nG], dseed=seed)      # keep=2: nano/merge's setting C (PLAN_2.md, addition 2)
    tag = kw.get("tag") or "run_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items())); os.makedirs(os.path.join(HERE, "res"), exist_ok=True); path = os.path.join(HERE, "res", tag + ".json")
    out = dict(task="sums", args=kw, seed=seed, plan=plan, nd=nd, mode=mode, rows=[], clean_subtractions=len(clean) if LsS is not None else None); tail = np.full(D, END, np.int64); total = 0
    for pi, (name, lines) in enumerate(plan):
        op = "+" if name == "add" else "-"; every = (5000 if lines <= 40000 else 50000) if nd == 2 else 10000
        marks = sorted(set(list(range(every, lines + 1, every)) + [lines] + ([5000, 10000, 20000, 40000] if lines > 40000 else []) + ([1000, 2000] if pi > 0 else []) + ([int(q) for q in kw["marks_extra"].split("+")] if pi == 0 and "marks_extra" in kw else []))); done = 0      # a later task is also read soon after it starts
        for mk in marks:
            X = drawer[op](mk - done).ravel(); x = np.concatenate([tail, X]); s.learn(G.ctx_of(x, D, len(x), D), x[D:]); tail = x[-D:]; done = mk
            row = dict(phase=pi, name=name, lines=mk, lines_in_life=total + mk, on_every=mk % every == 0, add=reading(s, LsA, prevs, nd))
            if LsS is not None: row["sub_clean"] = reading(s, LsS, prevs_clean)
            if int(kw.get("detail", 0)) and s.P:                  # PLAN_2.md, addition 1: the same reading with only the class kinds speaking, and with only the plain kinds
                keep = s.N.copy(); s.N[:s.nbase] = 0; row["add_class_kinds_alone"] = reading(s, LsA, prevs, nd); s.N[:] = keep; s.N[s.nbase:] = 0; row["add_plain_kinds_alone"] = reading(s, LsA, prevs, nd); s.N[:] = keep
            out["findings"] = [dict(bytes_read=f["bytes_read"], kept=[dict(R=r["R"], X=r["X"], classes=r["classes"], members=r["members"], worth=r["worth"]) for r in f["kept"]]) for f in s.finds]
            row.update(s.sizes()); row["second_parts_used"] = [int((s.Gu[g] > 0).sum()) for g in range(s.nG)]; row["cpu"] = round(time.process_time() - t0, 1); out["rows"].append(row)
            out["cpu"] = row["cpu"]; out["cpu_finding"] = round(s.tfind, 1); json.dump(out, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
            print(tag, f"{name} {mk}", " | ".join(f"{c}: after {v['after_add']:.1f} empty {v['empty']:.1f}" for c, v in row["add"].items()),
                  ("| sub clean " + " ".join(f"{c}: {v['after_sub']:.1f}/{v['empty']:.1f}" for c, v in row["sub_clean"].items())) if LsS is not None else "", "| cpu", row["cpu"], flush=True)
        total += lines
    out["second_parts"] = int(s.Gt[0].size); out["class_kinds"] = sum(sl is not None for sl in s.slots); out["finds"] = len(s.finds)
    json.dump(out, open(path + ".tmp", "w")); os.replace(path + ".tmp", path); return out, s

# ---------------------------------------------------------------- checks
def check():
    ok = True; rng = np.random.default_rng(5); L2 = kern3(); L = G.kern(); PU = ctypes.POINTER(ctypes.c_uint64)
    # 1. the compiled loop against numpy, and its first cell against merge.py's compiled loop
    for S, shrink, D, eta in ((14, True, 9, 0.05), (65, False, 6, 0.002)):      # with 70 senders a constant step of 0.05 is unstable in the control cells, and rounding grows to the first place within 1,500 bytes; 0.002 is stable
        K1, n, nc, nG = 70, 1500, 400, 4; ids = rng.integers(-1, nc, (n, K1)).astype(np.int32); sil = rng.random((n, K1)) < 0.1; ids[sil & (ids >= 0)] = -2 - ids[sil & (ids >= 0)]
        y = rng.integers(0, S, n).astype(np.int32); ctx = np.ascontiguousarray(rng.integers(0, S, (n, D)).astype(np.int32)); gc = np.array([0, 1, 0, 1], np.int32); gd = np.array([0, 0, 1, 1], np.int32); rs = np.array([12345], np.uint64); st = [12345]
        def fresh(): return [np.zeros((nc, S), np.float32), np.zeros(nc, np.float32), np.zeros(nc, np.float32), np.full((K1, NL), 0.1), np.zeros((K1, NL))]
        C, N, U, T, Tu = fresh(); Gt = np.zeros((nG, K1, NL, D, S)); Gu = np.zeros_like(Gt); P = np.zeros((n, S)); P2 = np.zeros((nG, n, S))
        L2.mix3_steps(n, K1, S, D, nG, p32(ids), p32(y), p32(ctx), pf(C), pf(N), pf(U), pd(T), pd(Tu), pd(Gt), pd(Gu), p32(gc), p32(gd), rs.ctypes.data_as(PU), eta, 0.3, 1, int(shrink), pd(P), pd(P2))
        Cr, Nr, Ur, Tr, Tur = fresh(); Gtr = np.zeros_like(Gt); Gur = np.zeros_like(Gt); Pr, P2r = steps3_np(ids, y, ctx, Cr, Nr, Ur, Tr, Tur, Gtr, Gur, gc, gd, st, eta, 0.3, True, shrink)
        d = max(np.abs(P - Pr).max(), np.abs(P2 - P2r).max(), np.abs(T - Tr).max(), np.abs(Gt - Gtr).max(), np.abs(Gu - Gur).max(), np.abs(C - Cr).max()); good = d < 1e-9 and int(rs[0]) == st[0] and 0.4 < Gu[2].sum() / Gu[0].sum() < 0.6; ok &= good
        print(f"the first cell and four second cells (two of them hearing half the senders), {K1} senders, S = {S}, D = {D}, shrinking step = {shrink}, step {eta}, {n} bytes: compiled against numpy, largest difference {d:.1e}; the random stream ends in the same state: {int(rs[0]) == st[0]}; uses of parts with half heard over uses with all heard: {Gu[2].sum() / Gu[0].sum():.3f} (parts used: {int((Gu[0] > 0).sum())} and {int((Gu[1] > 0).sum())}; largest part {np.abs(Gt).max():.3f}; the second cell differs from the first by up to {np.abs(P2[0] - P).max():.3f} in probability)", "PASSED" if good else "FAILED")
        Cm, Nm, Um, Tm, Tum = fresh(); Pm = np.zeros((n, S)); L.mix_steps(n, K1, S, p32(ids), p32(y), pf(Cm), pf(Nm), pf(Um), pd(Tm), pd(Tum), eta, 0.3, 1, int(shrink), pd(Pm))
        d = max(np.abs(P - Pm).max(), np.abs(T - Tm).max(), np.abs(C - Cm).max()); good = d == 0; ok &= good
        print(f"  the first cell of this loop against merge.py's compiled loop on the same input: largest difference {d:.1e}", "PASSED (the same to the last bit)" if good else "FAILED")
    # 2. the runner's first cell against merge.run_sums, reading by reading, on a short run with classes; and the parts of class kinds across findings
    kw = dict(seed="1", lines="4000", P="8"); ref, sm = G.run_sums(dict(kw)); out, s = run(dict(seed="1", plan="add:4000", P="8", tag="check_tmp3")); os.remove(os.path.join(HERE, "res", "check_tmp3.json"))
    mine = [r["add"]["first"]["empty"] for r in out["rows"]]; theirs = [r["add"] for r in ref["rows"]]; dT = float(np.abs(s.T - sm.T).max()); dC = float(np.abs(s.C[:s.top] - sm.C[:sm.top]).max()) if s.top == sm.top else float("inf")
    good = mine == theirs and dT == 0 and dC == 0; ok &= good
    print(f"the runner against merge.run_sums, seed 1, 4,000 lines, 8 pairs of classes, {len(s.finds)} findings: first cell's readings {mine} against {theirs}; trusts differ by {dT:.1e}, counts by {dC:.1e}", "PASSED" if good else "FAILED")
    # 3. the parts of a class kind follow the first cell's trust across a finding
    old = {(sl["k"], sl["X"]): p for p, sl in enumerate(s.slots) if sl is not None}; G0 = s.Gt.copy(); T0 = s.T.copy(); s.find(); kept = new = 0; good = True
    for p, sl in enumerate(s.slots):
        if sl is None: good &= not s.Gt[:, s.KB + p].any(); continue
        if (sl["k"], sl["X"]) in old: q = old[(sl["k"], sl["X"])]; kept += 1; good &= np.array_equal(s.Gt[:, s.KB + p], G0[:, s.KB + q]) and np.array_equal(s.T[s.KB + p], T0[s.KB + q])
        else: new += 1; good &= not s.Gt[:, s.KB + p].any() and not s.T[s.KB + p].any()
    good &= np.array_equal(s.Gt[:, :s.KB], G0[:, :s.KB]); ok &= bool(good) and kept > 0
    print(f"a further finding: {kept} pairs kept (their parts and trusts moved together), {new} new (parts and trusts nothing), plain kinds untouched", "PASSED" if good and kept > 0 else "FAILED")
    return ok

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=") for a in sys.argv[2:]); out, s = run(kw); r = out["rows"][-1]
    print("DONE", out["args"], "cpu", out["cpu"], "| parts used", r["second_parts_used"], "of", out["second_parts"], "| cells", r["cells"], flush=True)
