#!/usr/bin/env python3
"""PLAN_2.md: cells that count, read with a learned trust (the plain system of nano/merge/merge.py, mode none), and READINGS:
a store that has trust is read through other ordered sets of places. Each candidate reading is judged on what the store, read through
those places, foretells of the next block of the stream before that block is counted, less what the plain kind of those places foretells of it
with its own cells, and against the same store with its cells shuffled (PLAN_2.md, addition 1).
Kept readings speak as senders (their trust starts as the store's own kind's) and count into the one store.
    python3 reuse.py check
    python3 reuse.py run seed=1 phases=add2:40000,add3:20000 [mode=reuse|none|chance] [D=9] [P=64] [M=32] [one=1] [block=10000] [jbytes=5000] [inherit=1] [count=1] tag=...
Tasks: add2 `ab+cd=efg`, add3, add4 (needs D=11), add2w `ab++cd==efg` (the same sums, another layout), sub2 `ab-cd=efg` (larger first).
The answering loop and its numpy reference are those of merge.py (source: claude/nano-merge-8mmhfm at e0bc883) with one argument added."""
import ctypes, itertools, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); NL = 40; S = 14
PD = ctypes.POINTER(ctypes.c_double); PF = ctypes.POINTER(ctypes.c_float); PI = ctypes.POINTER(ctypes.c_int32)

SRC = r'''
#include <math.h>
#include <stdint.h>
#define NL 40
#define KMAX 640
#define SMAX 16
static inline int level(float n, float u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* merge.py's answering loop; the one addition: only the first Kc columns of addresses count (Kc = K1: all count) */
void mix_steps(int64_t n, int K1, int Kc, int S, const int32_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double *Tu,
               double eta, double e, int learn, int shrink, double *P) {
    static double lt[4096], s[KMAX][SMAX]; double a[SMAX], p[SMAX]; static int lev[KMAX];
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int64_t t = 0; t < n; t++) {
        const int32_t *id = ids + t * K1; double mx = -1e300, z = 0;
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int k = 0; k < K1; k++) {
            lev[k] = -1; if (id[k] < 0 || N[id[k]] < 1) continue;
            const float *c = C + (int64_t)id[k] * S; lev[k] = level(N[id[k]], U[id[k]]); const double tk = T[k * NL + lev[k]], den = log(N[id[k]] + S * e);
            for (int v = 0; v < S; v++) { const double sv = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den; s[k][v] = sv; a[v] += tk * sv; }
        }
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (learn) {
            for (int k = 0; k < K1; k++) if (lev[k] >= 0) {
                double g = s[k][y[t]]; for (int v = 0; v < S; v++) g -= p[v] * s[k][v];
                T[k * NL + lev[k]] += (shrink ? eta / sqrt(1.0 + Tu[k * NL + lev[k]]) : eta) * g; Tu[k * NL + lev[k]] += 1.0;
            }
            for (int k = 0; k < Kc; k++) { const int64_t i = id[k];
                if (i >= 0) { float *c = C + i * S; if (c[y[t]] == 0) U[i] += 1; c[y[t]] += 1; N[i] += 1; } }
        }
    }
}
/* What a store foretells of a block when read through every ordered set of m of the D places, in the order of itertools.permutations(range(D), m).
   gT[v * ncell + cell]: what the cell says of byte v less the mean of what the store's cells say of v (0 for a cell that has not counted).
   ctx: n x D byte values at delays 1..D. The loop runs over bytes outside and places inside, so that one table of ncell floats is in use at a time. */
void judge(int64_t n, int D, int S, int m, const int32_t *ctx, const int32_t *y, const float *gT, double *out) {
    int64_t nc = 1, np = 1; for (int i = 0; i < m; i++) { nc *= S; np *= D - i; }
    for (int64_t p = 0; p < np; p++) out[p] = 0;
    for (int64_t t = 0; t < n; t++) { const float *gy = gT + y[t] * nc; const int32_t *c = ctx + t * D; int64_t c1[16], c2[16], c3[16]; int p = 0;
        for (int d = 0; d < D; d++) { c1[d] = (int64_t)c[d] * S; c2[d] = c1[d] * S; c3[d] = c2[d] * S; }
        for (int q0 = 0; q0 < D; q0++) for (int q1 = 0; q1 < D; q1++) { if (q1 == q0) continue; const int64_t b1 = c[q0] + c1[q1];
            if (m == 2) { out[p++] += gy[b1]; continue; }
            for (int q2 = 0; q2 < D; q2++) { if (q2 == q0 || q2 == q1) continue; const int64_t b2 = b1 + c2[q2];
                if (m == 3) { out[p++] += gy[b2]; continue; }
                for (int q3 = 0; q3 < D; q3++) { if (q3 == q0 || q3 == q1 || q3 == q2) continue; out[p++] += gy[b2 + c3[q3]]; } } }
    }
}
'''
def kern():
    import hashlib; h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_r_{h}.so"); src = os.path.join(HERE, f"kern_r_{h}.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd = ctypes.c_int, ctypes.c_double
    L.mix_steps.argtypes = [ctypes.c_int64, ci, ci, ci, PI, PI, PF, PF, PF, PD, PD, cd, cd, ci, ci, PD]; L.mix_steps.restype = None
    L.judge.argtypes = [ctypes.c_int64, ci, ci, ci, PI, PI, PF, PD]; L.judge.restype = None; return L
p32 = lambda a: a.ctypes.data_as(PI); pf = lambda a: a.ctypes.data_as(PF); pd = lambda a: a.ctypes.data_as(PD)

# ---------------------------------------------------------------- references in numpy
def level_np(n, u):
    bn = 0; m = int(n)
    while m > 1 and bn < 9: m >>= 1; bn += 1
    return bn * 4 + (0 if u <= 1 else 1 if u <= 2 else 2 if u <= 4 else 3)

def steps_np(ids, y, C, N, U, T, Tu, eta, e, Kc, learn=True, shrink=True):
    P = np.zeros((len(y), S))
    for t in range(len(y)):
        a = np.zeros(S); ss = []
        for k, i in enumerate(ids[t]):
            if i >= 0 and N[i] >= 1:
                l = level_np(N[i], U[i]); sv = np.log(C[i].astype(np.float64) + e) - np.log(N[i] + S * e); ss.append((k, l, sv)); a += T[k, l] * sv
        p = np.exp(a - a.max()); p /= p.sum(); P[t] = p
        if learn:
            for k, l, sv in ss: T[k, l] += (eta / np.sqrt(1.0 + Tu[k, l]) if shrink else eta) * (sv[y[t]] - p @ sv); Tu[k, l] += 1.0
            for i in ids[t][:Kc]:
                if i >= 0:
                    if C[i, y[t]] == 0: U[i] += 1
                    C[i, y[t]] += 1; N[i] += 1
    return P

def judge_np(ctx, y, g, pl):
    m = pl.shape[1]; out = np.zeros(len(pl))
    for p, q in enumerate(pl):
        key = sum(ctx[:, q[i]].astype(np.int64) * S ** i for i in range(m)); out[p] = g[key, y].astype(np.float64).sum()
    return out

def gains_table(C, N, e):
    """For a store: what each counted cell says of each byte, less what a cell that has not counted is taken to say (1/S); 0 for a cell that has not counted."""
    cnt = N >= 1; lp = np.log(C.astype(np.float64) + e) - np.log(N.astype(np.float64) + S * e)[:, None] + np.log(S)
    return np.where(cnt[:, None], lp, 0.0).astype(np.float32), cnt

# ---------------------------------------------------------------- the system
def code(ctx, ds):
    c = np.zeros(len(ctx), np.int64)
    for i, d in enumerate(ds): c += ctx[:, d - 1].astype(np.int64) * (S ** i)
    return c
def ctx_of(x, lo, hi, D): return x[np.arange(lo, hi)[:, None] - np.arange(1, D + 1)[None, :]]

class Sys:
    def __init__(s, D=9, P=32, M=32, block=10000, mode="reuse", seed=0, eta=0.1, e=2.0, born=0.02, inherit=1, count=1, jbytes=5000, one=1):
        s.one = int(one); s.jbytes = int(jbytes); s.D, s.P, s.M, s.block, s.mode, s.eta, s.e, s.inherit, s.count = D, (0 if mode == "none" else P), M, block, mode, eta, e, int(inherit), int(count)
        s.kinds = [c for m in range(5) for c in itertools.combinations(range(1, D + 1), m)]; s.KB = len(s.kinds); s.K1 = s.KB + s.P; assert s.K1 <= 640
        s.off = np.concatenate([[0], np.cumsum([S ** len(k) for k in s.kinds])]).astype(np.int64); s.nbase = int(s.off[-1]); s.L = kern()
        s.C = np.zeros((s.nbase, S), np.float32); s.N = np.zeros(s.nbase, np.float32); s.U = np.zeros(s.nbase, np.float32)
        s.T = np.full((s.K1, NL), float(born)); s.T[s.KB:] = 0.0; s.Tu = np.zeros((s.K1, NL)); s.slots = [None] * s.P; s.saved = {}
        s.pl = {m: np.ascontiguousarray(np.array(list(itertools.permutations(range(D), m)), np.int32)) for m in (2, 3, 4)}
        kix = {k: i for i, k in enumerate(s.kinds)}; s.kidx = {m: np.array([kix[tuple(sorted(int(d) + 1 for d in q))] for q in s.pl[m]]) for m in (2, 3, 4)}     # the plain kind that has a reading's places
        s.read = 0; s.next_find = 0; s.finds = []; s.rng = np.random.default_rng([seed, 20261120, 7]); s.tfind = 0.0
    def ids(s, ctx):
        out = np.full((len(ctx), s.K1), -1, np.int32)
        for k, ds in enumerate(s.kinds): out[:, k] = code(ctx, ds) + s.off[k]
        for p, sl in enumerate(s.slots):
            if sl is not None: out[:, s.KB + p] = code(ctx, sl["pl"]) + s.off[sl["k"]]      # the store of kind k, addressed by the values at the reading's places
        return out
    def _run(s, ctx, y, learn, want_p):
        P = np.zeros((len(ctx), S)) if want_p else None
        for a in range(0, len(ctx), 50000):
            idd = np.ascontiguousarray(s.ids(ctx[a:a + 50000])); yy = np.ascontiguousarray(y[a:a + 50000].astype(np.int32)) if y is not None else np.zeros(len(idd), np.int32)
            Pp = np.zeros((len(idd), S)) if want_p else None
            s.L.mix_steps(len(idd), s.K1, s.K1 if s.count else s.KB, S, p32(idd), p32(yy), pf(s.C), pf(s.N), pf(s.U), pd(s.T), pd(s.Tu), s.eta, s.e, int(learn), 1, pd(Pp) if want_p else None)
            if want_p: P[a:a + 50000] = Pp
        return P
    def learn(s, ctx, y):
        """Reads the bytes y. Before a block is counted the candidate readings are judged on it; the block is then read with the readings
        that stood before, and the readings judged on it stand from its end."""
        a = 0
        while a < len(y):
            if s.P and s.read >= s.next_find and len(y) - a >= s.block // 2:
                nb = min(s.block, len(y) - a); kept, st = s.judge_block(ctx[a:a + nb], y[a:a + nb]); s._run(ctx[a:a + nb], y[a:a + nb], True, False); s.read += nb; a += nb; s.install(kept, st); s.next_find = s.read
            else:
                nb = len(y) - a if not s.P else min(len(y) - a, max(1, s.next_find - s.read) if s.read < s.next_find else len(y) - a)
                s._run(ctx[a:a + nb], y[a:a + nb], True, False); s.read += nb; a += nb
    def say(s, ctx): return s._run(ctx, None, False, True)
    def trust_of_kinds(s):
        u = s.Tu[:s.KB].sum(1); return np.where(u > 0, (s.T[:s.KB] * s.Tu[:s.KB]).sum(1) / np.maximum(u, 1), -np.inf)
    def judge_block(s, ctx, y):
        t0 = time.process_time(); mt = s.trust_of_kinds(); top = [int(k) for k in np.argsort(-mt, kind="stable")[:s.M] if len(s.kinds[k]) >= 2 and np.isfinite(mt[k])]
        cj = ctx[:s.jbytes]; yj = y[:s.jbytes]; c32 = np.ascontiguousarray(cj.astype(np.int32)); y32 = np.ascontiguousarray(yj.astype(np.int32)); cands = []; homes = []; nabove = 0
        idb = s.ids(cj)[:, :s.KB]; Nb = s.N[idb].astype(np.float64); own = np.where(Nb >= 1, np.log(s.C[idb, yj[:, None]].astype(np.float64) + s.e) - np.log(Nb + S * s.e) + np.log(S), 0.0).sum(0)    # what each plain kind's own cells foretell of these bytes
        for k in top:
            m = len(s.kinds[k]); g, cnt = gains_table(s.C[s.off[k]:s.off[k + 1]], s.N[s.off[k]:s.off[k + 1]], s.e)
            if cnt.sum() < 2: continue
            pl = s.pl[m]; out = np.zeros(len(pl)); gT = np.ascontiguousarray(g.T); s.L.judge(len(y32), s.D, S, m, p32(c32), p32(y32), pf(gT), pd(out)); out -= own[s.kidx[m]]
            idx = np.flatnonzero(cnt); gs = g.copy(); gs[idx] = g[idx[s.rng.permutation(len(idx))]]; outs = np.zeros(len(pl)); gT = np.ascontiguousarray(gs.T); s.L.judge(len(y32), s.D, S, m, p32(c32), p32(y32), pf(gT), pd(outs)); outs -= own[s.kidx[m]]
            chance = max(float(outs.max()), 0.0); home = tuple(d - 1 for d in s.kinds[k]); hi = int(np.flatnonzero((pl == np.array(home)).all(1))[0]); homes.append((k, float(own[k]), chance)); out[hi] = -np.inf
            for p in np.flatnonzero(out > chance): cands.append((float(out[p]), k, tuple(int(d) + 1 for d in pl[p]), chance)); nabove += 1
        cands.sort(key=lambda c: -c[0]); kept = []; seen = set()
        for c in cands:                                            # a set of places joins one store only: the one that foretells its new cells best (PLAN_2.md, addition 2)
            if s.one and tuple(sorted(c[2])) in seen: continue
            seen.add(tuple(sorted(c[2]))); kept.append(c)
            if len(kept) == s.P: break
        if s.mode == "chance":                                     # the control: the same stores and the same number of readings, the places drawn by chance
            kept = [(gn, k, tuple(int(d) + 1 for d in s.pl[len(P_)][s.rng.integers(len(s.pl[len(P_)]))]), ch) for gn, k, P_, ch in kept]
            kept = [c for c in kept if c[2] != s.kinds[c[1]]]
        dt = time.process_time() - t0; s.tfind += dt
        return kept, dict(bytes_read=int(s.read), stores_judged=len(top), above_chance=nabove, cpu=round(dt, 2), home=[(list(s.kinds[k]), round(h, 1), round(c, 1)) for k, h, c in homes[:8]])
    def install(s, kept, st):
        for p, sl in enumerate(s.slots):
            if sl is not None: s.saved[(sl["k"], sl["pl"])] = (s.T[s.KB + p].copy(), s.Tu[s.KB + p].copy())
        s.slots = [None] * s.P; s.T[s.KB:] = 0.0; s.Tu[s.KB:] = 0.0
        for p, (gn, k, P_, ch) in enumerate(kept):
            s.slots[p] = dict(k=k, pl=P_)
            if (k, P_) in s.saved: s.T[s.KB + p], s.Tu[s.KB + p] = s.saved[(k, P_)]          # a reading kept again keeps its trust
            elif s.inherit: s.T[s.KB + p] = s.T[k]                                           # a new reading starts with the trust of its store's own kind
        st["kept"] = [dict(store=list(s.kinds[k]), places=list(P_), gain=round(gn, 1), chance=round(ch, 1)) for gn, k, P_, ch in kept]; s.finds.append(st)
    def sizes(s):
        return dict(bytes_read=int(s.read), cells=int((s.N > 0).sum()), counts_above_zero=int((s.C > 0).sum()), readings=sum(sl is not None for sl in s.slots))

# ---------------------------------------------------------------- tasks
SYMS = list(b"0123456789+=\n-"); SI = {c: i for i, c in enumerate(SYMS)}; END = SI[10]
FMT = dict(add2=lambda a, b: f"{a:02d}+{b:02d}={a + b:03d}\n", add3=lambda a, b: f"{a:03d}+{b:03d}={a + b:04d}\n", add4=lambda a, b: f"{a:04d}+{b:04d}={a + b:05d}\n",
           add2w=lambda a, b: f"{a:02d}++{b:02d}=={a + b:03d}\n", sub2=lambda a, b: f"{max(a, b):02d}-{min(a, b):02d}={abs(a - b):03d}\n")
ND = dict(add2=2, add3=3, add4=4, add2w=2, sub2=2)
def text(task, a, b): return [SI[c] for c in FMT[task](a, b).encode()]

class Task:
    """The split and the stream of one task. add2, add2w: the split of loop_small_case/cells2.py as merge.py draws it (500 pairs scored, 8,000 shown);
    sub2: cells2.py's second permutation; add3: nano/ahead2/ahead3.py's split as merge.py draws it; add4: 500 pairs drawn and never shown."""
    def __init__(s, name, seed, phase):
        s.name = name; nd = ND[name]; n10 = 10 ** nd; rng = np.random.default_rng([seed, 99])
        if nd == 2:
            o = rng.permutation(10000); o2 = rng.permutation(10000)
            if name == "sub2": o = o2
            pairs = [(int(q) // 100, int(q) % 100) for q in o]; s.held = pairs[:500]; s.shown = np.array([text(name, a, b) for a, b in pairs[2000:]], np.int64); s.pool = None
        elif nd == 3:
            o = rng.permutation(n10 * n10); s.held = [(int(q) // n10, int(q) % n10) for q in o[:500]]; s.pool = o[n10 * n10 // 5:]
        else:
            h = rng.integers(n10 * n10, size=500); s.held = [(int(q) // n10, int(q) % n10) for q in h]; s.heldset = set(int(q) for q in h); s.pool = None
        s.n10 = n10; s.rng = rng if phase == 0 else np.random.default_rng([seed, 99, phase])          # the first phase draws as merge.py does; a later phase has a stream of its own
        s.Ls = np.array([text(name, a, b) for a, b in s.held]); s.a0 = int(np.flatnonzero(s.Ls[0] == SI[ord("=")])[-1]) + 1; s.alen = nd + 1
    def draw(s, k):
        if ND[s.name] == 2: return s.shown[s.rng.integers(8000, size=k)]
        if ND[s.name] == 3: q = s.pool[s.rng.integers(len(s.pool), size=k)]
        else:
            q = s.rng.integers(s.n10 * s.n10, size=k); bad = np.array([int(v) in s.heldset for v in q])
            while bad.any(): q[bad] = s.rng.integers(s.n10 * s.n10, size=int(bad.sum())); bad = np.array([int(v) in s.heldset for v in q])
        return np.array([text(s.name, int(v) // s.n10, int(v) % s.n10) for v in q], np.int64)
    def score(s, sy):
        """Never-shown lines, the system's own answer bytes, the whole answer."""
        own = s.Ls.copy(); n = len(own); D = sy.D
        for t in range(s.a0, s.a0 + s.alen):
            ctx = np.full((n, D), END, np.int64)
            for d in range(1, D + 1):
                if t - d >= 0: ctx[:, d - 1] = own[:, t - d]
            own[:, t] = sy.say(ctx).argmax(1)
        ok = own[:, s.a0:s.a0 + s.alen] == s.Ls[:, s.a0:s.a0 + s.alen]; return float(100 * ok.all(1).mean()), [float(100 * ok[:, i].mean()) for i in range(s.alen)]

def marks_for(lines):
    return sorted(set([m for m in (1000, 2000, 3000, 5000, 7500) if m <= lines] + list(range(5000, lines + 1, 5000)) + [lines]))

def run(kw):
    seed = int(kw.get("seed", 1)); mode = kw.get("mode", "reuse"); D = int(kw.get("D", 9)); t0 = time.process_time()
    phases = [(p.split(":")[0], int(p.split(":")[1])) for p in kw.get("phases", "add2:40000").split(",")]; tasks = [Task(n, seed, i) for i, (n, _) in enumerate(phases)]
    sy = Sys(D=D, P=int(kw.get("P", 64)), M=int(kw.get("M", 32)), block=int(kw.get("block", 10000)), mode=mode, seed=seed, inherit=int(kw.get("inherit", 1)), count=int(kw.get("count", 1)), jbytes=int(kw.get("jbytes", 5000)), one=int(kw.get("one", 1)))
    tail = np.full(D, END, np.int64); rows = []
    for i, (name, lines) in enumerate(phases):
        tk = tasks[i]; done = 0
        if i > 0: w, pl = tk.score(sy); rows.append(dict(phase=i, task=name, lines=0, score={name: w}, place={name: pl}, **sy.sizes()))       # before any line of the new task
        for mk in marks_for(lines):
            X = tk.draw(mk - done).ravel(); x = np.concatenate([tail, X]); sy.learn(ctx_of(x, D, len(x), D), x[D:]); tail = x[-D:]; done = mk
            sc = {}; pc = {}
            for j in range(i + 1):
                if j == i or mk == lines or mk % 10000 == 0: sc[tasks[j].name + ("" if j == i else f"@{j}")], pc[tasks[j].name + ("" if j == i else f"@{j}")] = tasks[j].score(sy)
            rows.append(dict(phase=i, task=name, lines=mk, score=sc, place=pc, cpu=round(time.process_time() - t0, 1), **sy.sizes()))
    mt = sy.trust_of_kinds(); u = sy.Tu.sum(1); mall = np.where(u > 0, (sy.T * sy.Tu).sum(1) / np.maximum(u, 1), 0.0)
    out = dict(args=kw, seed=seed, mode=mode, phases=phases, rows=rows, sizes=sy.sizes(), cpu=round(time.process_time() - t0, 1), cpu_judging=round(sy.tfind, 1),
               top_kinds=[list(sy.kinds[k]) for k in np.argsort(-mt, kind="stable")[:sy.M]], finds=sy.finds,
               readings_at_end=[dict(store=list(sy.kinds[sl["k"]]), places=list(sl["pl"]), mean_trust=round(float(mall[sy.KB + p]), 4), largest_trust=round(float(sy.T[sy.KB + p].max()), 4), uses=float(u[sy.KB + p])) for p, sl in enumerate(sy.slots) if sl is not None])
    return out, sy

# ---------------------------------------------------------------- checks
def check():
    ok = True; rng = np.random.default_rng(5); L = kern()
    for Kc in (9, 6):
        K1, n, nc = 9, 3000, 400; ids = np.ascontiguousarray(rng.integers(-1, nc, (n, K1)).astype(np.int32)); y = rng.integers(0, S, n).astype(np.int32)
        C = np.zeros((nc, S), np.float32); N = np.zeros(nc, np.float32); U = np.zeros(nc, np.float32); T = np.full((K1, NL), 0.1); Tu = np.zeros((K1, NL)); P = np.zeros((n, S))
        L.mix_steps(n, K1, Kc, S, p32(ids), p32(y), pf(C), pf(N), pf(U), pd(T), pd(Tu), 0.05, 0.3, 1, 1, pd(P))
        C2 = np.zeros((nc, S), np.float32); N2 = np.zeros(nc, np.float32); U2 = np.zeros(nc, np.float32); T2 = np.full((K1, NL), 0.1); Tu2 = np.zeros((K1, NL)); P2 = steps_np(ids, y, C2, N2, U2, T2, Tu2, 0.05, 0.3, Kc)
        d = max(np.abs(P - P2).max(), np.abs(T - T2).max(), np.abs(C - C2).max()); ok &= d < 1e-9; print(f"answering loop, {n} bytes, the first {Kc} of {K1} columns counting: compiled against numpy, largest difference {d:.1e}", "PASSED" if d < 1e-9 else "FAILED")
    for m, D in ((2, 9), (3, 9), (4, 11)):
        n = 700; ctx = np.ascontiguousarray(rng.integers(0, S, (n, D)).astype(np.int32)); y = np.ascontiguousarray(rng.integers(0, S, n).astype(np.int32))
        C = (rng.integers(0, 4, (S ** m, S)) * (rng.random((S ** m, S)) < 0.3)).astype(np.float32); g, cnt = gains_table(C, C.sum(1), 2.0); g = np.ascontiguousarray(g)
        pl = np.array(list(itertools.permutations(range(D), m)), np.int32); out = np.zeros(len(pl)); gT = np.ascontiguousarray(g.T); L.judge(n, D, S, m, p32(ctx), p32(y), pf(gT), pd(out)); ref = judge_np(ctx, y, g, pl)
        d = float(np.abs(out - ref).max()); ok &= d < 1e-6; print(f"judgment of {len(pl)} readings of a store of {m} places over {n} bytes ({D} delays): compiled against numpy, largest difference {d:.1e}", "PASSED" if d < 1e-6 else "FAILED")
    # a reading addresses the store's cell for the values at its places
    sy = Sys(mode="reuse", P=4); k = sy.kinds.index((3, 4, 6, 7)); sy.slots[0] = dict(k=k, pl=(8, 9, 4, 5)); ctx = rng.integers(0, S, (200, 9)); a = sy.ids(ctx)[:, sy.KB]
    ref = np.array([sy.off[k] + c[7] + c[8] * S + c[3] * S * S + c[4] * S ** 3 for c in ctx]); good = bool((a == ref).all()); ok &= good; print("a reading of the store (3, 4, 6, 7) through the places (8, 9, 4, 5): 200 addresses against the definition", "PASSED" if good else "FAILED")
    # the plain system is merge.py's: the same scores on a short run
    import merge
    o1, _ = merge.run_sums(dict(seed="1", lines="5000", mode="none")); o2, _ = run(dict(seed="1", phases="add2:5000", mode="none")); r2 = [r["score"]["add2"] for r in o2["rows"] if r["lines"] == 5000][0]
    good = abs(o1["last"] - r2) < 1e-9; ok &= good; print(f"plain system, two-digit sums, seed 1, 5,000 lines: merge.py {o1['last']}, this program {r2}", "PASSED" if good else "FAILED (the streams are drawn in other portions; see PLAN_2.md)")
    return ok

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=") for a in sys.argv[2:]); out, sy = run(kw); tag = kw["tag"]; os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", tag + ".json"), "w"))
    for i, (name, lines) in enumerate(out["phases"]):
        print(tag, f"| phase {i} {name}:", " ".join(f"{r['lines'] / 1000:g}k:{r['score'][name]:.1f}" for r in out["rows"] if r["phase"] == i), flush=True)
    print(tag, "| end:", out["rows"][-1]["score"], "| cells", out["sizes"]["cells"], "readings", out["sizes"]["readings"], "| cpu", out["cpu"], "judging", out["cpu_judging"], flush=True)
