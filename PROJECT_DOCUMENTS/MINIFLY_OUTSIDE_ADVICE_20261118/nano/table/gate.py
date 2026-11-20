#!/usr/bin/env python3
"""PLAN_15.md: a second answering cell whose trusts know what byte stands at each delay. The system is nano/merge/merge.py's, unchanged;
the second cell hears the same senders and the first cell's branches, and holds for each delay a part of the trust indexed by the byte there.
    python3 gate.py check
    python3 gate.py sums  seed=1 [mode=classes|none] [gate=0|1|2] [lines=40000]
    python3 gate.py life  books=new|worn [mode=none|classes] [gate=0|1]
gate=0: one part for each delay, indexed by the byte at that delay.  gate=1 (the control): the same parts and steps, every part reads a constant.
gate=2 (the side arm): one part, indexed by the number (0 to 3) of speaking senders that have counted fewer times and at least twice (PLAN_14.md).
The first cell is not touched: its answers in the same run are the plain system's."""
import ctypes, hashlib, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "merge"))
import merge as G
import score
NL = 40; PD = ctypes.POINTER(ctypes.c_double); PF = ctypes.POINTER(ctypes.c_float); PI = ctypes.POINTER(ctypes.c_int32)
ROOT = os.path.join(HERE, "..", "..", "..", "..")
BOOKS = dict(worn=[("minifly_v148_pretrain_probe_dev_20261025", b) for b in ("bryant-stories", "burgess-busterbrown", "carroll-alice")],
             new=[("minifly_v155_keys_dev_v1_20261026", b) for b in ("chesterton-ball", "austen-emma", "melville-moby_dick")])

SRC = r'''
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#define NL 40
#define KMAX 400
#define SMAX 65
#define DMAX 16
static inline int level(float n, float u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
static int cmpf(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
/* the first cell exactly as nano/merge's mix_steps; the second cell's branches are the first cell's plus, for each speaking sender, (the sum of its
   parts) times the sender's message.  Parts: Gt[part][kind][level][gate value]; gate 0: part j reads the byte at delay j + 1; gate 1: every part reads 0;
   gate 2: one part, reading the number (0..3) of speaking senders that have counted fewer times than this one and at least twice. */
void mix_steps2(int64_t n, int K1, int S, const int32_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double *Tu,
                double eta, double e, int learn, int shrink, double *P, int D, const int32_t *cb, double *Gt, double *Gu, int gate, double *P2, double birth, int rule) {
    static double lt[4096], s[KMAX][SMAX]; double a[SMAX], p[SMAX], a2[SMAX], p2[SMAX]; static int lev[KMAX]; static int64_t gix[KMAX][DMAX]; static float srt[KMAX];
    const int NG = gate == 2 ? 4 : S, np = gate == 2 ? 1 : D;
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int64_t t = 0; t < n; t++) {
        const int32_t *id = ids + t * K1; double mx = -1e300, z = 0; int m = 0;
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int k = 0; k < K1; k++) {
            lev[k] = -1; if (id[k] < 0 || N[id[k]] < 1) continue;
            const float *c = C + (int64_t)id[k] * S; lev[k] = level(N[id[k]], U[id[k]]); const double tk = T[k * NL + lev[k]], den = log(N[id[k]] + S * e);
            for (int v = 0; v < S; v++) { const double sv = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den; s[k][v] = sv; a[v] += tk * sv; }
            if (gate == 2 && N[id[k]] >= 2) srt[m++] = N[id[k]];
        }
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        /* the second cell */
        if (gate == 2) qsort(srt, m, sizeof(float), cmpf);
        for (int v = 0; v < S; v++) a2[v] = a[v];
        for (int k = 0; k < K1; k++) if (lev[k] >= 0) {
            double tk = 0;
            for (int j = 0; j < np; j++) {
                int gv;
                if (gate == 0) gv = cb[t * D + j]; else if (gate == 1) gv = 0;
                else { const float nk = N[id[k]]; int lo = 0, hi = m; while (lo < hi) { const int mid = (lo + hi) / 2; if (srt[mid] < nk) lo = mid + 1; else hi = mid; } gv = lo > 3 ? 3 : lo; }
                gix[k][j] = (((int64_t)j * K1 + k) * NL + lev[k]) * NG + gv; tk += Gt[gix[k][j]];
            }
            if (tk != 0) for (int v = 0; v < S; v++) a2[v] += tk * s[k][v];
        }
        mx = -1e300; z = 0;
        for (int v = 0; v < S; v++) if (a2[v] > mx) mx = a2[v];
        for (int v = 0; v < S; v++) { p2[v] = exp(a2[v] - mx); z += p2[v]; }
        for (int v = 0; v < S; v++) p2[v] /= z;
        if (P2) for (int v = 0; v < S; v++) P2[t * S + v] = p2[v];
        if (learn) {
            for (int k = 0; k < K1; k++) if (lev[k] >= 0) {
                double g = s[k][y[t]], g2 = s[k][y[t]]; for (int v = 0; v < S; v++) { g -= p[v] * s[k][v]; g2 -= p2[v] * s[k][v]; }
                if (rule) { const int q = k * NL + lev[k]; Tu[q] += g * g; if (Tu[q] > 0) T[q] += eta * g / sqrt(Tu[q]);      /* PLAN_22.md: one rule for every text */
                    for (int j = 0; j < np; j++) { const int64_t i = gix[k][j]; Gu[i] += g2 * g2; if (Gu[i] > 0) Gt[i] += eta * g2 / sqrt(Gu[i]); } }
                else {
                T[k * NL + lev[k]] += (shrink ? eta / sqrt(1.0 + Tu[k * NL + lev[k]]) : eta) * g; Tu[k * NL + lev[k]] += 1.0;
                for (int j = 0; j < np; j++) { const int64_t i = gix[k][j]; Gt[i] += (shrink ? eta / sqrt(1.0 + Gu[i]) : eta) * g2; Gu[i] += 1.0; } }
            }
            for (int k = 0; k < K1; k++) { const int64_t i = id[k] < -1 ? -2 - (int64_t)id[k] : id[k];
                if (i >= 0) { if (birth > 0 && N[i] == 0 && p[y[t]] >= birth) continue;      /* PLAN_19.md: a cell is not born at a byte to which the first cell gave at least this probability */
                    float *c = C + i * S; if (c[y[t]] == 0) U[i] += 1; c[y[t]] += 1; N[i] += 1; } }
        }
    }
}
'''
def kern():
    h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_gate_{h}.so"); src = os.path.join(HERE, "kern_gate.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd = ctypes.c_int, ctypes.c_double
    L.mix_steps2.argtypes = [ctypes.c_int64, ci, ci, PI, PI, PF, PF, PF, PD, PD, cd, cd, ci, ci, PD, ci, PI, PD, PD, ci, PD, cd, ci]; L.mix_steps2.restype = None; return L
p32 = lambda a: a.ctypes.data_as(PI); pf = lambda a: a.ctypes.data_as(PF); pd = lambda a: a.ctypes.data_as(PD)

def scramble(a):
    """A fixed scramble of cell addresses (the finishing steps of a common 64-bit mixing function); it knows nothing of kinds, bytes or texts."""
    z = a.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9); z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))

class Sys2(G.Sys):
    """nano/merge's system with the second answering cell. s.cell says whose answer say() gives (1: the first cell, the plain system; 2: the second)."""
    GATE = 0; BLIND = 0; BIRTH = 0.0; RULE = 0; ALPHA = 0.0; NOLEVEL = 0; E = 0.0
    def __init__(s, *a, **kw):
        super().__init__(*a, **kw); s.gate = Sys2.GATE; s.L2 = kern()
        if Sys2.E: s.e = float(Sys2.E)                                       # a side arm of PLAN_22.md: one smoothing for every text
        if Sys2.RULE: s.eta = float(Sys2.ALPHA); s.T[:] = 0.0                # PLAN_22.md: every trust is born at nothing and the step needs no number chosen for a text
        if True: s.NG = 4 if s.gate == 2 else s.S; s.nparts = 1 if s.gate == 2 else s.D; assert s.D <= 16
        s.Gt = np.zeros((s.nparts, s.K1, NL, s.NG)); s.Gu = np.zeros_like(s.Gt); s.cell = 2
    def _run(s, ctx, y, learn, want_p):
        P = np.zeros((len(ctx), s.S)) if want_p else None
        for a in range(0, len(ctx), 50000):
            idd = s.ids(ctx[a:a + 50000], create=learn); yy = np.ascontiguousarray(y[a:a + 50000].astype(np.int32)) if y is not None else np.zeros(len(idd), np.int32)
            cb = np.ascontiguousarray(ctx[a:a + 50000, :s.D].astype(np.int32)); P1 = np.zeros((len(idd), s.S)) if want_p else None; P2 = np.zeros((len(idd), s.S)) if want_p else None
            s.L2.mix_steps2(len(idd), s.K1, s.S, p32(idd), p32(yy), pf(s.C), pf(s.N), pf(s.U), pd(s.T), pd(s.Tu), s.eta, s.e, int(learn), s.shrink, pd(P1) if want_p else None,
                            s.D, p32(cb), pd(s.Gt), pd(s.Gu), s.gate, pd(P2) if want_p else None, float(Sys2.BIRTH), int(Sys2.RULE))
            if want_p: P[a:a + 50000] = P2 if s.cell == 2 else P1
        return P
    def ids(s, ctx, create=False):
        """PLAN_17.md: with BLIND = B > 0, a cell whose address falls in one class of B by a fixed scramble never comes to be (it neither counts nor speaks)."""
        out = super().ids(ctx, create)
        if Sys2.BLIND: out[(out >= 0) & (scramble(out) % np.uint64(Sys2.BLIND) == 0)] = -1
        return out
    def find(s):
        """Classes are found anew by nano/merge's find; the second cell's parts for a class kind follow the first cell's trust: kept where the pair is kept, nothing where it is new."""
        old = {(sl["k"], sl["X"]): p for p, sl in enumerate(s.slots) if sl is not None}; G0, U0 = s.Gt.copy(), s.Gu.copy(); super().find(); s.Gt[:, s.KB:] = 0; s.Gu[:, s.KB:] = 0
        for p, sl in enumerate(s.slots):
            if sl is not None and (sl["k"], sl["X"]) in old: q = old[(sl["k"], sl["X"])]; s.Gt[:, s.KB + p] = G0[:, s.KB + q]; s.Gu[:, s.KB + p] = U0[:, s.KB + q]

# ---------------------------------------------------------------- the reference in numpy
def steps2_np(ids, y, cb, C, N, U, T, Tu, Gt, Gu, eta, e, gate, learn=True, shrink=False, birth=0.0, rule=0):
    S = C.shape[1]; n = len(y); P1 = np.zeros((n, S)); P2 = np.zeros((n, S)); nparts = Gt.shape[0]
    for t in range(n):
        a = np.zeros(S); ss = []
        for k, i in enumerate(ids[t]):
            if i >= 0 and N[i] >= 1:
                l = G.level_np(N[i], U[i]); sv = np.log(C[i].astype(np.float64) + e) - np.log(N[i] + S * e); ss.append((k, l, sv, float(N[i]))); a += T[k, l] * sv
        p = np.exp(a - a.max()); p /= p.sum(); P1[t] = p; a2 = a.copy(); used = []
        for k, l, sv, nk in ss:
            if gate == 0: gv = [int(cb[t, j]) for j in range(nparts)]
            elif gate == 1: gv = [0] * nparts
            else: gv = [min(3, sum(1 for _, _, _, nj in ss if 2 <= nj < nk))]
            a2 += sum(Gt[j, k, l, gv[j]] for j in range(nparts)) * sv; used.append(gv)
        p2 = np.exp(a2 - a2.max()); p2 /= p2.sum(); P2[t] = p2
        if learn:
            for (k, l, sv, nk), gv in zip(ss, used):
                g = sv[y[t]] - p @ sv; g2 = sv[y[t]] - p2 @ sv
                if rule:
                    Tu[k, l] += g * g
                    if Tu[k, l] > 0: T[k, l] += eta * g / np.sqrt(Tu[k, l])
                    for j in range(nparts):
                        Gu[j, k, l, gv[j]] += g2 * g2
                        if Gu[j, k, l, gv[j]] > 0: Gt[j, k, l, gv[j]] += eta * g2 / np.sqrt(Gu[j, k, l, gv[j]])
                    continue
                T[k, l] += (eta / np.sqrt(1.0 + Tu[k, l]) if shrink else eta) * g; Tu[k, l] += 1.0
                for j in range(nparts): Gt[j, k, l, gv[j]] += (eta / np.sqrt(1.0 + Gu[j, k, l, gv[j]]) if shrink else eta) * g2; Gu[j, k, l, gv[j]] += 1.0
            for i in ids[t]:
                if i < -1: i = -2 - i
                if i >= 0:
                    if birth > 0 and N[i] == 0 and p[y[t]] >= birth: continue
                    if C[i, y[t]] == 0: U[i] += 1
                    C[i, y[t]] += 1; N[i] += 1
    return P1, P2

def check():
    ok = True; rng = np.random.default_rng(5); L = kern(); L1 = G.kern()
    for S, shrink, gate, D, birth, rule in ((14, True, 0, 9, 0.0, 0), (65, False, 0, 6, 0.0, 0), (14, True, 1, 9, 0.0, 0), (14, True, 2, 9, 0.0, 0), (65, False, 2, 6, 0.0, 0), (14, True, 0, 9, 0.1, 0), (65, False, 0, 6, 0.02, 0), (14, False, 0, 9, 0.0, 1), (65, False, 0, 6, 0.0, 1)):
        K1, n, nc = 9, 3000, 400; ids = rng.integers(-1, nc, (n, K1)).astype(np.int32); sil = rng.random((n, K1)) < 0.1; ids[sil & (ids >= 0)] = -2 - ids[sil & (ids >= 0)]; y = rng.integers(0, S, n).astype(np.int32)
        cb = rng.integers(0, S, (n, D)).astype(np.int32); NG = 4 if gate == 2 else S; npart = 1 if gate == 2 else D
        def fresh(): return np.zeros((nc, S), np.float32), np.zeros(nc, np.float32), np.zeros(nc, np.float32), np.full((K1, NL), 0.1), np.zeros((K1, NL)), np.zeros((npart, K1, NL, NG)), np.zeros((npart, K1, NL, NG))
        C, N, U, T, Tu, Gt, Gu = fresh(); P1 = np.zeros((n, S)); P2 = np.zeros((n, S))
        L.mix_steps2(n, K1, S, p32(ids), p32(y), pf(C), pf(N), pf(U), pd(T), pd(Tu), 0.05, 0.3, 1, int(shrink), pd(P1), D, p32(cb), pd(Gt), pd(Gu), gate, pd(P2), birth, rule)
        Cb, Nb, Ub, Tb, Tub, Gtb, Gub = fresh(); R1, R2 = steps2_np(ids, y, cb, Cb, Nb, Ub, Tb, Tub, Gtb, Gub, 0.05, 0.3, gate, True, shrink, birth, rule)
        d = max(np.abs(P1 - R1).max(), np.abs(P2 - R2).max(), np.abs(T - Tb).max(), np.abs(Gt - Gtb).max(), np.abs(C - Cb).max(), np.abs(Gu - Gub).max())
        Cc, Nc, Uc, Tc, Tuc, _, _ = fresh(); Pm = np.zeros((n, S)); L1.mix_steps(n, K1, S, p32(ids), p32(y), pf(Cc), pf(Nc), pf(Uc), pd(Tc), pd(Tuc), 0.05, 0.3, 1, int(shrink), pd(Pm))      # the first cell against nano/merge's own kernel
        d1 = max(np.abs(P1 - Pm).max(), np.abs(T - Tc).max(), np.abs(C - Cc).max()) if birth == 0 and rule == 0 else 0.0; fewer = float(Nc.sum() - N.sum()); good = d < 1e-9 and d1 == 0 and np.abs(P2 - P1).max() > 1e-3 and (birth == 0 or fewer > 0); ok &= good
        print(f"two answering cells, S = {S}, shrinking step = {shrink}, gate = {gate}, birth = {birth} ({fewer:.0f} counts fewer than with none), rule = {rule}, {n} bytes: compiled against numpy, largest difference {d:.1e}; the first cell against nano/merge's kernel {d1:.1e}; the cells differ by up to {np.abs(P2 - P1).max():.2f}", "PASSED" if good else "FAILED")
    return ok

# ---------------------------------------------------------------- sums
def run_sums(kw):
    seed = int(kw.get("seed", 1)); D = 9; Sys2.GATE = int(kw.get("gate", 0)); Sys2.BLIND = int(kw.get("blind", 0)); Sys2.BIRTH = float(kw.get("birth", 0)); Sys2.RULE = int(kw.get("rule", 0)); Sys2.ALPHA = float(kw.get("alpha", 0)); Sys2.E = float(kw.get("e", 0)); G.Sys = Sys2; t0 = time.process_time()
    rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); pairs = [(int(q) // 100, int(q) % 100) for q in o]; held = pairs[:500]; shown = pairs[2000:]
    TAB = np.array([G.text(a, b) for a, b in shown], np.int64); Ls = np.array([G.text(a, b) for a, b in held]); g = np.random.default_rng([seed, 7]); prevs = [TAB[g.integers(8000, size=500)] for _ in range(5)]
    def one(s, prev):
        own = Ls.copy()
        for t in (6, 7, 8):
            ctx = np.full((500, D), G.END, np.int64)
            for d in range(1, D + 1):
                if t - d >= 0: ctx[:, d - 1] = own[:, t - d]
                elif prev is not None: ctx[:, d - 1] = prev[:, 10 + t - d]
            own[:, t] = s.say(ctx).argmax(1)
        ok = own[:, 6:9] == Ls[:, 6:9]; return float(100 * ok.all(1).mean()), [float(100 * ok[:, i].mean()) for i in range(3)]
    def both(s):
        r = {}
        for cell in (1, 2):
            s.cell = cell; e = one(s, None); a = [one(s, pv) for pv in prevs]; r[f"cell{cell}"] = dict(empty=e[0], empty_place=e[1], after=float(np.mean([q[0] for q in a])), after_place=np.mean([q[1] for q in a], 0).tolist())
        s.cell = 2; return r
    readings = []; orig = G.sums_score; state = dict(n=0)
    def wrapped(s, held_, D_=9, nd=2):
        state["n"] += 1; readings.append(both(s))                          # one at every reading; nano/merge's looks at the end of the run are dropped below
        return orig(s, held_, D_, nd)
    G.sums_score = wrapped; out, s = G.run_sums(dict(seed=str(seed), lines=kw.get("lines", "40000"), mode=kw.get("mode", "classes"))); G.sums_score = orig
    readings = readings[:len(out["rows"])]; assert len(readings) == len(out["rows"])
    for r, row in zip(readings, out["rows"]): r["lines"] = row["lines"]
    res = dict(task="sums", args=kw, seed=seed, readings=readings, merge_rows=[(r["lines"], r["add"]) for r in out["rows"]], sizes=out["sizes"], parts=int(s.Gt.size), parts_used=int((s.Gu > 0).sum()), largest_part=float(np.abs(s.Gt).max()))
    for cell in ("cell1", "cell2"):
        res[cell] = dict(last_empty=readings[-1][cell]["empty"], last_after=readings[-1][cell]["after"], last3_empty=float(np.mean([r[cell]["empty"] for r in readings[-3:]])), last3_after=float(np.mean([r[cell]["after"] for r in readings[-3:]])),
                         place_after=readings[-1][cell]["after_place"])
    if s.P:
        keep = s.N.copy(); s.N[:s.nbase] = 0; res["class_kinds_alone"] = both(s); s.N[:] = keep
    res["cpu"] = round(time.process_time() - t0, 1); return res

# ---------------------------------------------------------------- prose
def load(books):
    xa, cut, _ = score.load(); vals = score.byte_values(); tab = np.full(256, 1, np.int64); tab[vals] = np.arange(65); parts = [("A", xa)]; replaced = {}
    for folder, b in BOOKS[books]:
        raw = np.frombuffer(open(os.path.join(ROOT, folder, "data", b + ".txt"), "rb").read(), np.uint8); raw = raw[raw != 13]; parts.append((b, tab[raw])); replaced[b] = round(100 * float((~np.isin(raw, vals)).mean()), 2)
    x = np.concatenate([p[1] for p in parts]); segs = {}; off = 0
    for name, p in parts:
        c = cut if name == "A" else int(0.9 * len(p)); segs[name] = (off, off + c, off + len(p)); off += len(p)
    return x, segs, replaced

def run_life(kw):
    books = kw.get("books", "new"); Sys2.GATE = int(kw.get("gate", 0)); Sys2.BLIND = int(kw.get("blind", 0)); Sys2.BIRTH = float(kw.get("birth", 0)); Sys2.RULE = int(kw.get("rule", 0)); Sys2.ALPHA = float(kw.get("alpha", 0)); Sys2.E = float(kw.get("e", 0)); G.Sys = Sys2; t0 = time.process_time(); x, segs, replaced = load(books); s = G.prose_sys(x, dict(mode=kw.get("mode", "none")))
    out = dict(task="prose_life", args=kw, segments={k: list(map(int, v)) for k, v in segs.items()}, replaced_percent=replaced, scores={})
    def sc(lo, hi):
        r = {}
        for cell in (1, 2): s.cell = cell; q = score.score(s.say(G.ctx_of(x, lo, hi, s.D)), x[lo:hi]); r[f"cell{cell}"] = dict(nats=q["nats"], right=q["right"])
        return r
    for name, (a, c, b) in segs.items():
        lo = max(a, s.D)
        for m in range(lo, c, 125000): hi = min(c, m + 125000); s.learn(G.ctx_of(x, m, hi, s.D), x[m:hi])
        out["scores"][name] = sc(c, b)
    a, c, b = segs["A"]; out["scores"]["A_at_the_end"] = sc(c, b)
    out.update(sizes=s.sizes(), parts=int(s.Gt.size), parts_used=int((s.Gu > 0).sum()), largest_part=float(np.abs(s.Gt).max()), cpu=round(time.process_time() - t0, 1)); return out

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=") for a in sys.argv[2:]); out = dict(sums=run_sums, life=run_life)[sys.argv[1]](kw)
    tag = "gate_" + sys.argv[1] + "_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items())); os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", tag + ".json"), "w"))
    if sys.argv[1] == "sums":
        print(tag, "| after a read line, last and mean of last three: first cell", round(out["cell1"]["last_after"], 1), round(out["cell1"]["last3_after"], 1), "second cell", round(out["cell2"]["last_after"], 1), round(out["cell2"]["last3_after"], 1),
              "| empty past: first", round(out["cell1"]["last_empty"], 1), round(out["cell1"]["last3_empty"], 1), "second", round(out["cell2"]["last_empty"], 1), round(out["cell2"]["last3_empty"], 1),
              "| places after a read line: first", [round(v, 1) for v in out["cell1"]["place_after"]], "second", [round(v, 1) for v in out["cell2"]["place_after"]],
              "| class kinds alone:", {k: round(v["after"], 1) for k, v in out.get("class_kinds_alone", {}).items()}, "| merge rows", out["merge_rows"][-3:], "| parts used", out["parts_used"], "of", out["parts"], "largest", round(out["largest_part"], 3), "| cpu", out["cpu"], flush=True)
    else:
        print(tag, "|", {k: (round(v["cell1"]["nats"], 4), round(v["cell2"]["nats"], 4), round(v["cell1"]["nats"] - v["cell2"]["nats"], 4)) for k, v in out["scores"].items()}, "| parts used", out["parts_used"], "of", out["parts"], "largest", round(out["largest_part"], 3), "| cells", out["sizes"]["cells"], "| cpu", out["cpu"], flush=True)
