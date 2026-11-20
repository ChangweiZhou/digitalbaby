#!/usr/bin/env python3
"""PLAN_1.md: cells that count, read with a learned trust; cells of one kind that are refined alike are merged into a class,
and a new kind of cell is keyed on (the class, the refining lines). One program for two-digit sums and for prose.
    python3 merge.py check
    python3 merge.py sums  seed=1 [lines=40000] [P=32] [eJ=0.02] [mode=classes|none|chance|half] [tag=...]
    python3 merge.py dev   [P=32] [eJ=0.02] [mode=...]            prose: taught on the stream before its last twentieth, scored on that twentieth
    python3 merge.py life  [P=32] [eJ=0.02] [mode=...] [scoreA=0]  prose: the test text's stream, then three books, each last tenth scored after its stream
Re-written from loop_small_case/cells2.py (text of sums, split, form of the score), nano/table/trust.py (cells, trust, levels) and
nano/table/life.py (the life); source commit 71172671f. score.py is copied unchanged from nano/table."""
import ctypes, itertools, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); NL = 40
PD = ctypes.POINTER(ctypes.c_double); PF = ctypes.POINTER(ctypes.c_float); PI = ctypes.POINTER(ctypes.c_int32)

SRC = r'''
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define NL 40
#define KMAX 400
#define SMAX 65
static inline int level(float n, float u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* at each byte: every active cell that has counted says s(v) = log((c(v)+e)/(n+S e)); the answering cell's branches compete on the sum of trust times s;
   then each trust that was used changes by its step times (mismatch . s), and the cells count (nano/table/trust.py, with S a parameter) */
void mix_steps(int64_t n, int K1, int S, const int32_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double *Tu,
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
            for (int k = 0; k < K1; k++) { const int64_t i = id[k] < -1 ? -2 - (int64_t)id[k] : id[k];      /* an address below -1: the cell -2-address counts and is silent */
                if (i >= 0) { float *c = C + i * S; if (c[y[t]] == 0) U[i] += 1; c[y[t]] += 1; N[i] += 1; } }
        }
    }
}

/* The search for classes of one pair (X, Z). R-cells i = 0..m-1 with x index xc[i] and z index zc[i]; counts sparse (cptr, sym, cnt), totals n.
   J = sum over R-cells and bytes of c(v) log((C(v) - c(v) + e) / (N - n + S e)), C and N the pooled counts of the cell's class at its z. */
typedef struct { int m, nx, S; const int32_t *xc, *zc, *zptr, *zcell, *xptr, *xcell, *cptr, *sym; const float *cnt, *n; double e;
                 double *Cz, *Nz, *bef, *aft, *dl, *ci; int32_t *tz, *td; char *seen, *ind; int ntd; } W;
static void delta_all(W *w, int x, const int32_t *cls) {
    const int S = w->S; const double e = w->e, lS = log((double)S); w->ntd = 0;
    for (int a = w->xptr[x]; a < w->xptr[x + 1]; a++) {
        const int i = w->xcell[a], z = w->zc[i]; int ntz = 0;
        if (w->zptr[z + 1] - w->zptr[z] < 2) continue;
        for (int q = w->cptr[i]; q < w->cptr[i + 1]; q++) w->ci[w->sym[q]] = w->cnt[q];
        for (int b = w->zptr[z]; b < w->zptr[z + 1]; b++) {
            const int j = w->zcell[b]; if (j == i) continue; const int c = cls[w->xc[j]]; if (c < 0) continue;
            if (!w->seen[c]) { w->seen[c] = 1; w->tz[ntz++] = c; w->Nz[c] = 0; w->bef[c] = 0; w->aft[c] = 0; }
            w->Nz[c] += w->n[j]; for (int q = w->cptr[j]; q < w->cptr[j + 1]; q++) w->Cz[(int64_t)c * S + w->sym[q]] += w->cnt[q];
        }
        for (int b = w->zptr[z]; b < w->zptr[z + 1]; b++) {
            const int j = w->zcell[b]; if (j == i) continue; const int c = cls[w->xc[j]]; if (c < 0) continue;
            const double lb = log(w->Nz[c] - w->n[j] + S * e), la = log(w->Nz[c] + w->n[i] - w->n[j] + S * e); const double *Cc = w->Cz + (int64_t)c * S;
            for (int q = w->cptr[j]; q < w->cptr[j + 1]; q++) { const int v = w->sym[q]; const double cv = w->cnt[q];
                w->bef[c] += cv * (log(Cc[v] - cv + e) - lb); w->aft[c] += cv * (log(Cc[v] + w->ci[v] - cv + e) - la); }
        }
        for (int u = 0; u < ntz; u++) {
            const int c = w->tz[u]; double own = 0; const double lo = log(w->Nz[c] + S * e); const double *Cc = w->Cz + (int64_t)c * S;
            for (int q = w->cptr[i]; q < w->cptr[i + 1]; q++) own += w->cnt[q] * (log(Cc[w->sym[q]] + e) - lo);
            if (!w->ind[c]) { w->ind[c] = 1; w->td[w->ntd++] = c; w->dl[c] = 0; }
            w->dl[c] += w->aft[c] + own - w->bef[c] + w->n[i] * lS; w->seen[c] = 0;
        }
        for (int b = w->zptr[z]; b < w->zptr[z + 1]; b++) {
            const int j = w->zcell[b]; if (j == i) continue; const int c = cls[w->xc[j]]; if (c < 0) continue;
            for (int q = w->cptr[j]; q < w->cptr[j + 1]; q++) w->Cz[(int64_t)c * S + w->sym[q]] = 0;
        }
        for (int q = w->cptr[i]; q < w->cptr[i + 1]; q++) w->ci[w->sym[q]] = 0;
    }
}
static int best(W *w) {                                    /* the class that raises J most (ties to the lowest label); -1 if none raises it */
    int bc = -1; double bd = 1e-9;
    for (int u = 0; u < w->ntd; u++) { const int c = w->td[u]; if (w->dl[c] > bd + 1e-9 || (w->dl[c] > bd - 1e-9 && bc >= 0 && c < bc)) { if (w->dl[c] > 1e-9) { bd = w->dl[c]; bc = c; } } w->ind[c] = 0; }
    return bc;
}
double j_total(int m, int nx, int nz, int S, const int32_t *xc, const int32_t *zptr, const int32_t *zcell, const int32_t *cptr, const int32_t *sym,
               const float *cnt, const float *n, double e, const int32_t *cls) {
    double *Cz = calloc((size_t)nx * S, sizeof(double)), *Nz = calloc(nx, sizeof(double)), J = 0;
    for (int z = 0; z < nz; z++) {
        for (int b = zptr[z]; b < zptr[z + 1]; b++) { const int j = zcell[b], c = cls[xc[j]]; Nz[c] += n[j]; for (int q = cptr[j]; q < cptr[j + 1]; q++) Cz[(int64_t)c * S + sym[q]] += cnt[q]; }
        for (int b = zptr[z]; b < zptr[z + 1]; b++) { const int j = zcell[b], c = cls[xc[j]]; const double lb = log(Nz[c] - n[j] + S * e);
            for (int q = cptr[j]; q < cptr[j + 1]; q++) J += cnt[q] * (log(Cz[(int64_t)c * S + sym[q]] - cnt[q] + e) - lb); }
        for (int b = zptr[z]; b < zptr[z + 1]; b++) { const int j = zcell[b], c = cls[xc[j]]; Nz[c] = 0; for (int q = cptr[j]; q < cptr[j + 1]; q++) Cz[(int64_t)c * S + sym[q]] = 0; }
    }
    free(Cz); free(Nz); return J;
}
void find_classes(int m, int nx, int nz, int S, const int32_t *xc, const int32_t *zc, const int32_t *zptr, const int32_t *zcell, const int32_t *xptr, const int32_t *xcell,
                  const int32_t *cptr, const int32_t *sym, const float *cnt, const float *n, double e, const int32_t *xord, int sweeps, int dissolve, int32_t *cls) {
    W w = { m, nx, S, xc, zc, zptr, zcell, xptr, xcell, cptr, sym, cnt, n, e }; (void)nz;
    w.Cz = calloc((size_t)nx * S, sizeof(double)); w.Nz = calloc(nx, sizeof(double)); w.bef = calloc(nx, sizeof(double)); w.aft = calloc(nx, sizeof(double));
    w.dl = calloc(nx, sizeof(double)); w.ci = calloc(S, sizeof(double)); w.tz = calloc(nx, sizeof(int32_t)); w.td = calloc(nx, sizeof(int32_t)); w.seen = calloc(nx, 1); w.ind = calloc(nx, 1);
    int32_t *size = calloc(nx, sizeof(int32_t));                         /* a class that is founded takes the lowest label that has no member */
#define LOWEST(c) { c = 0; while (size[c]) c++; }
#define SWEEP for (int o = 0; o < nx; o++) { const int x = xord[o], old = cls[x]; size[old]--; cls[x] = -1; delta_all(&w, x, cls); int c = best(&w); \
        if (c < 0) { if (size[old] == 0) c = old; else LOWEST(c) } cls[x] = c; size[c]++; }
    for (int x = 0; x < nx; x++) cls[x] = -1;
    for (int o = 0; o < nx; o++) { const int x = xord[o]; delta_all(&w, x, cls); int c = best(&w); if (c < 0) LOWEST(c) cls[x] = c; size[c]++; }
    for (int sw = 0; sw < sweeps; sw++) SWEEP
    if (dissolve == 2) {  /* PLAN_8.md: a class is split around its worst-foretold member: that member founds a class, every other member chooses between the two; kept only if J rises; up to four times a class; then one more sweep */
        int32_t *save = malloc(nx * sizeof(int32_t)), *ssize = malloc(nx * sizeof(int32_t)), *mem = malloc(nx * sizeof(int32_t)), *labs = malloc(nx * sizeof(int32_t)); int nl = 0;
        double *xn = calloc(nx, sizeof(double)); for (int i = 0; i < m; i++) xn[xc[i]] += n[i];
        for (int c = 0; c < nx; c++) if (size[c] >= 3) labs[nl++] = c;
        for (int u = 0; u < nl; u++) { const int c0 = labs[u];
            for (int round = 0; round < 4 && size[c0] >= 3; round++) {
                const double Jb = j_total(m, nx, nz, S, xc, zptr, zcell, cptr, sym, cnt, n, e, cls); int nm = 0, wm = -1; double worst = 1e300;
                memcpy(save, cls, nx * sizeof(int32_t)); memcpy(ssize, size, nx * sizeof(int32_t));
                for (int o = 0; o < nx; o++) if (cls[xord[o]] == c0) mem[nm++] = xord[o];
                for (int i = 0; i < nm; i++) { const int x = mem[i]; cls[x] = -1; delta_all(&w, x, cls); double d = 0; for (int q = 0; q < w.ntd; q++) { if (w.td[q] == c0) d = w.dl[c0]; w.ind[w.td[q]] = 0; }
                    cls[x] = c0; d /= xn[x]; if (d < worst - 1e-9) { worst = d; wm = x; } }
                int L; size[c0]--; LOWEST(L) cls[wm] = L; size[L] = 1;
                for (int i = 0; i < nm; i++) { const int x = mem[i]; if (x == wm) continue; size[c0]--; cls[x] = -1; delta_all(&w, x, cls); double d0 = 0, dL = 0;
                    for (int q = 0; q < w.ntd; q++) { if (w.td[q] == c0) d0 = w.dl[c0]; if (w.td[q] == L) dL = w.dl[L]; w.ind[w.td[q]] = 0; }
                    const int c = dL > d0 + 1e-9 ? L : c0; cls[x] = c; size[c]++; }
                if (j_total(m, nx, nz, S, xc, zptr, zcell, cptr, sym, cnt, n, e, cls) < Jb + 1e-9) { memcpy(cls, save, nx * sizeof(int32_t)); memcpy(size, ssize, nx * sizeof(int32_t)); break; }
            }
        }
        SWEEP
        free(save); free(ssize); free(mem); free(labs); free(xn);
    }
    if (dissolve == 1) {  /* each class of two or more is dissolved once and its members seated again one by one; kept only if J rises; then one more sweep */
        int32_t *save = malloc(nx * sizeof(int32_t)), *ssize = malloc(nx * sizeof(int32_t)), *mem = malloc(nx * sizeof(int32_t)), *labs = malloc(nx * sizeof(int32_t)); int nl = 0;
        for (int c = 0; c < nx; c++) if (size[c] >= 2) labs[nl++] = c;
        for (int u = 0; u < nl; u++) { const int c0 = labs[u]; if (size[c0] < 2) continue;
            const double Jb = j_total(m, nx, nz, S, xc, zptr, zcell, cptr, sym, cnt, n, e, cls); int nm = 0;
            memcpy(save, cls, nx * sizeof(int32_t)); memcpy(ssize, size, nx * sizeof(int32_t));
            for (int o = 0; o < nx; o++) if (cls[xord[o]] == c0) { mem[nm++] = xord[o]; cls[xord[o]] = -1; }
            size[c0] = 0;
            for (int i = 0; i < nm; i++) { const int x = mem[i]; delta_all(&w, x, cls); int c = best(&w); if (c < 0) LOWEST(c) cls[x] = c; size[c]++; }
            if (j_total(m, nx, nz, S, xc, zptr, zcell, cptr, sym, cnt, n, e, cls) < Jb + 1e-9) { memcpy(cls, save, nx * sizeof(int32_t)); memcpy(size, ssize, nx * sizeof(int32_t)); }
        }
        SWEEP
        free(save); free(ssize); free(mem); free(labs);
    }
    free(w.Cz); free(w.Nz); free(w.bef); free(w.aft); free(w.dl); free(w.ci); free(w.tz); free(w.td); free(w.seen); free(w.ind); free(size);
}
'''
def kern():
    import hashlib; h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_g_{h}.so"); src = os.path.join(HERE, "kern_g.c")      # one .so for each text of the kernel, so that a run in progress keeps its own
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd = ctypes.c_int, ctypes.c_double
    L.mix_steps.argtypes = [ctypes.c_int64, ci, ci, PI, PI, PF, PF, PF, PD, PD, cd, cd, ci, ci, PD]; L.mix_steps.restype = None
    L.j_total.argtypes = [ci, ci, ci, ci, PI, PI, PI, PI, PI, PF, PF, cd, PI]; L.j_total.restype = cd
    L.find_classes.argtypes = [ci, ci, ci, ci, PI, PI, PI, PI, PI, PI, PI, PI, PF, PF, cd, PI, ci, ci, PI]; L.find_classes.restype = None; return L
p32 = lambda a: a.ctypes.data_as(PI); pf = lambda a: a.ctypes.data_as(PF); pd = lambda a: a.ctypes.data_as(PD)

# ---------------------------------------------------------------- references in numpy and plain Python
def level_np(n, u):
    bn = 0; m = int(n)
    while m > 1 and bn < 9: m >>= 1; bn += 1
    return bn * 4 + (0 if u <= 1 else 1 if u <= 2 else 2 if u <= 4 else 3)

def steps_np(ids, y, C, N, U, T, Tu, eta, e, learn=True, shrink=False):
    S = C.shape[1]; P = np.zeros((len(y), S))
    for t in range(len(y)):
        a = np.zeros(S); ss = []
        for k, i in enumerate(ids[t]):
            if i >= 0 and N[i] >= 1:
                l = level_np(N[i], U[i]); sv = np.log(C[i].astype(np.float64) + e) - np.log(N[i] + S * e); ss.append((k, l, sv)); a += T[k, l] * sv
        p = np.exp(a - a.max()); p /= p.sum(); P[t] = p
        if learn:
            for k, l, sv in ss: T[k, l] += (eta / np.sqrt(1.0 + Tu[k, l]) if shrink else eta) * (sv[y[t]] - p @ sv); Tu[k, l] += 1.0
            for i in ids[t]:
                if i < -1: i = -2 - i
                if i >= 0:
                    if C[i, y[t]] == 0: U[i] += 1
                    C[i, y[t]] += 1; N[i] += 1
    return P

def j_py(cells, cls, S, e):
    """cells: list of (x, z, {v: count}). J by its definition, in plain Python."""
    import math; pool = {}
    for x, z, c in cells:
        d = pool.setdefault((cls[x], z), {})
        for v, k in c.items(): d[v] = d.get(v, 0.0) + k
    J = 0.0
    for x, z, c in cells:
        d = pool[(cls[x], z)]; N = sum(d.values()); n = sum(c.values())
        for v, k in c.items(): J += k * (math.log(d[v] - k + e) - math.log(N - n + S * e))
    return J

def find_py(cells, nx, xord, S, e, sweeps, dissolve=0):
    """The search of find_classes by brute force: every move is judged by J computed whole. Labels nx.. stand for 'taken out, alone'."""
    cls = list(range(nx, 2 * nx)); size = [0] * nx; lowest = lambda: next(c for c in range(nx) if size[c] == 0)
    def choose(x):
        base = j_py(cells, cls, S, e); bc, bd = -1, 1e-9
        for c in sorted({cls[q] for q in range(nx) if q != x and cls[q] < nx}):
            old = cls[x]; cls[x] = c; d = j_py(cells, cls, S, e) - base; cls[x] = old
            if d > bd + 1e-9: bd, bc = d, c
        return bc
    def sweep():
        for x in xord:
            old = cls[x]; size[old] -= 1; cls[x] = nx + x; c = choose(x)
            if c < 0: c = old if size[old] == 0 else lowest()
            cls[x] = c; size[c] += 1
    for x in xord:
        c = choose(x)
        if c < 0: c = lowest()
        cls[x] = c; size[c] += 1
    for _ in range(sweeps): sweep()
    if dissolve == 2:
        xn = [0.0] * nx
        for x, z, c in cells: xn[x] += sum(c.values())
        def gain(x, c):                                   # J with x in class c less J with x alone
            old = cls[x]; cls[x] = nx + x; base = j_py(cells, cls, S, e); cls[x] = c; d = j_py(cells, cls, S, e) - base; cls[x] = old; return d
        for c0 in [c for c in range(nx) if size[c] >= 3]:
            for _ in range(4):
                if size[c0] < 3: break
                Jb = j_py(cells, cls, S, e); save, ssize = cls[:], size[:]; mem = [x for x in xord if cls[x] == c0]; w, worst = -1, 1e300
                for x in mem:
                    d = gain(x, c0) / xn[x]
                    if d < worst - 1e-9: worst, w = d, x
                size[c0] -= 1; L = lowest(); cls[w] = L; size[L] = 1
                for x in mem:
                    if x == w: continue
                    size[c0] -= 1; d0 = gain(x, c0) if size[c0] > 0 else 0.0; dL = gain(x, L); c = L if dL > d0 + 1e-9 else c0; cls[x] = c; size[c] += 1
                if j_py(cells, cls, S, e) < Jb + 1e-9: cls[:] = save; size[:] = ssize; break
        sweep()
    if dissolve == 1:
        for c0 in [c for c in range(nx) if size[c] >= 2]:
            if size[c0] < 2: continue
            Jb = j_py(cells, cls, S, e); save, ssize = cls[:], size[:]; mem = [x for x in xord if cls[x] == c0]
            for x in mem: cls[x] = nx + x
            size[c0] = 0
            for x in mem:
                c = choose(x)
                if c < 0: c = lowest()
                cls[x] = c; size[c] += 1
            if j_py(cells, cls, S, e) < Jb + 1e-9: cls[:] = save; size[:] = ssize
        sweep()
    return cls

# ---------------------------------------------------------------- the system
def code(ctx, ds, S):
    c = np.zeros(len(ctx), np.int64)
    for i, d in enumerate(ds): c += ctx[:, d - 1].astype(np.int64) * (S ** i)
    return c

def ctx_of(x, lo, hi, D):
    return x[np.arange(lo, hi)[:, None] - np.arange(1, D + 1)[None, :]]

class Sys:
    def __init__(s, S, D, kinds, eta, e, born, shrink, text=None, P=0, eJ=0.02, M=16, B0=10000, sweeps=2, mode="classes", seed=0, half_at=0, rep=0, dissolve=0, pool=None, practice=0.0, empty=0, keep=0):
        s.S, s.D, s.kinds, s.eta, s.e, s.shrink, s.P, s.eJ, s.M, s.sweeps, s.mode = S, D, [tuple(k) for k in kinds], eta, e, int(shrink), P, eJ, M, sweeps, mode
        s.keep = int(keep); s.rep = int(rep); s.dissolve = int(dissolve); s.practice = float(practice); s.pool = (mode != "half" and not int(empty)) if pool is None else bool(int(pool))
        if mode == "none": s.P = 0
        s.L = kern(); s.KB = len(kinds); s.K1 = s.KB + s.P; assert s.K1 <= 400 and S <= 65
        s.uniq = []; s.off = []; off = 0; full = ctx_of(text, D, len(text), D) if text is not None else None
        for ds in s.kinds:
            if S ** len(ds) <= 1 << 23: s.uniq.append(None); size = S ** len(ds)
            else: u = np.unique(code(full, ds, S)); s.uniq.append(u); size = len(u)
            s.off.append(off); off += size
        s.nbase = off; s.top = off; s.C = np.zeros((off + 1024, S), np.float32); s.N = np.zeros(off + 1024, np.float32); s.U = np.zeros(off + 1024, np.float32)
        s.T = np.full((s.K1, NL), float(born)); s.T[s.KB:] = 0.0; s.Tu = np.zeros((s.K1, NL)); s.slots = [None] * s.P
        s.read = 0; s.next_find = B0 if mode != "half" else half_at; s.finds = []; s.rng = np.random.default_rng([seed, 20261120]); s.tfind = 0.0
    # ---- cell addresses
    def base_ids(s, ctx):
        out = np.empty((len(ctx), s.KB), np.int32)
        for k, ds in enumerate(s.kinds):
            c = code(ctx, ds, s.S)
            if s.uniq[k] is None: out[:, k] = c + s.off[k]
            else:
                u = s.uniq[k]; j = np.searchsorted(u, c); j[j == len(u)] = 0; out[:, k] = np.where(u[j] == c, j + s.off[k], -1)
        return out
    def ensure(s, n):
        if s.top + n > len(s.N):
            cap = int((s.top + n) * 1.5) + 1024
            for a in ("C", "N", "U"):
                old = getattr(s, a); new = np.zeros((cap,) + old.shape[1:], np.float32); new[:len(old)] = old; setattr(s, a, new)
    def class_ids(s, ctx, create):
        out = np.full((len(ctx), s.P), -1, np.int32)
        for p, sl in enumerate(s.slots):
            if sl is None: continue
            c = sl["clsmap"][code(ctx, sl["X"], s.S)]; ok = c >= 0
            if not ok.any(): continue
            key = c[ok].astype(np.int64) * sl["ZS"] + code(ctx[ok], sl["Z"], s.S)
            if create:
                new = np.setdiff1d(key, sl["keys"])
                if len(new):
                    s.ensure(len(new)); ids = np.arange(s.top, s.top + len(new), dtype=np.int32); s.top += len(new)
                    kk = np.concatenate([sl["keys"], new]); ii = np.concatenate([sl["cids"], ids]); o = np.argsort(kk, kind="stable"); sl["keys"], sl["cids"] = kk[o], ii[o]
            if len(sl["keys"]):
                j = np.searchsorted(sl["keys"], key); j[j == len(sl["keys"])] = 0; out[ok, p] = np.where(sl["keys"][j] == key, sl["cids"][j], -1)
        return out
    def ids(s, ctx, create=False):
        b = s.base_ids(ctx)
        if not s.P: return np.ascontiguousarray(b)
        cl = s.class_ids(ctx, create)
        if s.rep or (create and s.practice > 0):                    # replacement: where a class cell stands, the member's own R-cell counts and is silent
            prac = s.rng.random(len(ctx)) < s.practice if not s.rep else None      # practice (PLAN_4.md): only while reading, and only at a share of the bytes
            for p, sl in enumerate(s.slots):
                if sl is not None:
                    r = np.flatnonzero((cl[:, p] >= 0) & prac) if prac is not None else np.flatnonzero(cl[:, p] >= 0); v = b[r, sl["k"]]; b[r, sl["k"]] = np.where(v >= 0, -2 - v, v)
        return np.ascontiguousarray(np.hstack([b, cl]))
    # ---- reading and answering
    def _run(s, ctx, y, learn, want_p):
        P = np.zeros((len(ctx), s.S)) if want_p else None
        for a in range(0, len(ctx), 50000):
            idd = s.ids(ctx[a:a + 50000], create=learn); yy = np.ascontiguousarray(y[a:a + 50000].astype(np.int32)) if y is not None else np.zeros(len(idd), np.int32)
            Pp = np.zeros((len(idd), s.S)) if want_p else None
            s.L.mix_steps(len(idd), s.K1, s.S, p32(idd), p32(yy), pf(s.C), pf(s.N), pf(s.U), pd(s.T), pd(s.Tu), s.eta, s.e, int(learn), s.shrink, pd(Pp) if want_p else None)
            if want_p: P[a:a + 50000] = Pp
        return P
    def learn(s, ctx, y):
        """Reads the bytes y (each with its context); classes are found anew whenever the bytes read have doubled since the last finding."""
        a = 0
        while a < len(y):
            b = len(y) if (s.P == 0 or s.next_find is None) else min(len(y), a + max(0, s.next_find - s.read))
            if b > a: s._run(ctx[a:b], y[a:b], True, False); s.read += b - a; a = b
            if s.P and s.next_find is not None and s.read >= s.next_find:
                s.find(); s.next_find = None if s.mode == "half" else s.next_find * 2
    def say(s, ctx): return s._run(ctx, None, False, True)
    # ---- the search for classes
    def r_cells(s, k):
        """The cells of base kind k that have counted: their addresses and codes."""
        size = (s.S ** len(s.kinds[k])) if s.uniq[k] is None else len(s.uniq[k]); idx = np.flatnonzero(s.N[s.off[k]:s.off[k] + size] > 0)
        return idx + s.off[k], (idx.astype(np.int64) if s.uniq[k] is None else s.uniq[k][idx])
    def split(s, k, X, cod):
        ds = s.kinds[k]; xk = np.zeros(len(cod), np.int64); zk = np.zeros(len(cod), np.int64); ix = iz = 0
        for i, d in enumerate(ds):
            v = cod // (s.S ** i) % s.S
            if d in X: xk += v * (s.S ** ix); ix += 1
            else: zk += v * (s.S ** iz); iz += 1
        return xk, zk
    def search(s, k, X, pre=None):
        ids_, cod = pre if pre is not None else s.r_cells(k); xk, zk = s.split(k, X, cod); ux, xc = np.unique(xk, return_inverse=True); uz, zc = np.unique(zk, return_inverse=True)
        m, nx, nz = len(cod), len(ux), len(uz)
        if nx < 2 or m < 2: return None
        rows = s.C[ids_]; r, v = np.nonzero(rows); cnt = np.ascontiguousarray(rows[r, v]); cptr = np.zeros(m + 1, np.int32); cptr[1:] = np.cumsum(np.bincount(r, minlength=m)); sym = v.astype(np.int32)
        n = np.ascontiguousarray(s.N[ids_]); xc = xc.astype(np.int32); zc = zc.astype(np.int32)
        zcell = np.argsort(zc, kind="stable").astype(np.int32); zptr = np.zeros(nz + 1, np.int32); zptr[1:] = np.cumsum(np.bincount(zc, minlength=nz))
        xcell = np.argsort(xc, kind="stable").astype(np.int32); xptr = np.zeros(nx + 1, np.int32); xptr[1:] = np.cumsum(np.bincount(xc, minlength=nx))
        xn = np.bincount(xc, weights=n, minlength=nx); xord = np.argsort(-xn, kind="stable").astype(np.int32); cls = np.zeros(nx, np.int32)
        s.L.find_classes(m, nx, nz, s.S, p32(xc), p32(zc), p32(zptr), p32(zcell), p32(xptr), p32(xcell), p32(cptr), p32(sym), pf(cnt), pf(n), s.eJ, p32(xord), s.sweeps, s.dissolve, p32(cls))
        jt = lambda c: s.L.j_total(m, nx, nz, s.S, p32(xc), p32(zptr), p32(zcell), p32(cptr), p32(sym), pf(cnt), pf(n), s.eJ, p32(np.ascontiguousarray(c, dtype=np.int32)))
        J = jt(cls); J1 = jt(np.zeros(nx, np.int32)); size = np.bincount(cls, minlength=nx); merged = size[cls] >= 2
        return dict(k=k, X=X, ux=ux, cls=cls, merged=merged, J=J, J1=J1, J0=-float(n.sum()) * np.log(s.S), worth=J - J1, nx=nx, classes=int((size >= 2).sum()), members=int(merged.sum()), tokens=float(n.sum()))
    def trust_of_kinds(s):
        u = s.Tu[:s.KB].sum(1); mt = np.where(u > 0, (s.T[:s.KB] * s.Tu[:s.KB]).sum(1) / np.maximum(u, 1), -np.inf)
        if s.keep == 2 and s.P:                                     # PLAN_6.md, addition 2: the trust of a kind counts the trust of the kinds made from it
            for k, v in s.child_trust(): mt[k] = max(mt[k], 0.0) + v if np.isfinite(mt[k]) else v
        return mt
    def child_trust(s):
        uc = s.Tu[s.KB:s.KB + s.P].sum(1); mc = np.where(uc > 0, (s.T[s.KB:s.KB + s.P] * s.Tu[s.KB:s.KB + s.P]).sum(1) / np.maximum(uc, 1), 0.0)
        return [(sl["k"], max(0.0, float(mc[p]))) for p, sl in enumerate(s.slots) if sl is not None]
    def find(s):
        t0 = time.process_time(); mt = s.trust_of_kinds(); top = [int(k) for k in np.argsort(-mt, kind="stable")[:s.M] if len(s.kinds[k]) >= 2 and np.isfinite(mt[k])]; found = []
        if s.keep == 1: top += sorted({sl["k"] for sl in s.slots if sl is not None} - set(top))          # PLAN_6.md, addition 1: a kind whose classes are in use is searched again
        for k in top:
            pre = s.r_cells(k)
            for nxd in (1, 2):
                for X in itertools.combinations(s.kinds[k], nxd):
                    if len(X) == len(s.kinds[k]): continue
                    r = s.search(k, X, pre)
                    if r is not None and r["members"] >= 2 and r["worth"] > 0: found.append(r)
        found.sort(key=lambda r: -r["worth"]); kept = found[:s.P]; old = {(sl["k"], sl["X"]): p for p, sl in enumerate(s.slots) if sl is not None}
        T0, Tu0 = s.T.copy(), s.Tu.copy(); s.T[s.KB:] = 0.0; s.Tu[s.KB:] = 0.0; s.top = s.nbase; s.slots = [None] * s.P; rec = []
        for p, r in enumerate(kept):
            k, X = r["k"], r["X"]; Z = tuple(d for d in s.kinds[k] if d not in X); cls = r["cls"].copy(); merged = r["merged"]
            if s.mode == "chance": perm = s.rng.permutation(len(cls)); cls = cls[perm]; merged = merged[perm]          # the same sizes, members drawn by chance
            if (k, X) in old: s.T[s.KB + p] = T0[s.KB + old[(k, X)]]; s.Tu[s.KB + p] = Tu0[s.KB + old[(k, X)]]
            clsmap = np.full(s.S ** len(X), -1, np.int32); clsmap[r["ux"][merged]] = cls[merged]; ZS = s.S ** len(Z)
            sl = dict(k=k, X=X, Z=Z, clsmap=clsmap, ZS=ZS, keys=np.zeros(0, np.int64), cids=np.zeros(0, np.int32)); s.slots[p] = sl
            if s.pool:                                              # the class cells start from the pooled counts of their members' R-cells
                ids_, cod = s.r_cells(k); xk, zk = s.split(k, X, cod); c = clsmap[xk]; ok = c >= 0
                key = c[ok].astype(np.int64) * ZS + zk[ok]; uk, inv = np.unique(key, return_inverse=True); s.ensure(len(uk)); o = np.argsort(inv, kind="stable")
                start = np.zeros(len(uk), np.int64); start[1:] = np.cumsum(np.bincount(inv, minlength=len(uk)))[:-1]; pooled = np.add.reduceat(s.C[ids_[ok]][o], start, axis=0)
                cid = np.arange(s.top, s.top + len(uk), dtype=np.int32); s.C[cid] = pooled; s.N[cid] = pooled.sum(1); s.U[cid] = (pooled > 0).sum(1); s.top += len(uk); sl["keys"], sl["cids"] = uk, cid
            rec.append(dict(R=list(s.kinds[k]), X=list(X), Z=list(Z), x_cells=r["nx"], classes=r["classes"], members=r["members"], worth=round(r["worth"], 1), J=round(r["J"], 1), J_one_class=round(r["J1"], 1), J_alone=round(r["J0"], 1),
                            tokens=r["tokens"], members_by_class={int(c): [int(q) for q in r["ux"][merged & (cls == c)]] for c in np.unique(cls[merged])} if r["nx"] <= 400 else None))
        s.C[s.top:] = 0; s.N[s.top:] = 0; s.U[s.top:] = 0; dt = time.process_time() - t0; s.tfind += dt
        s.finds.append(dict(bytes_read=int(s.read), candidates=len(top), pairs_searched=sum(len(list(itertools.combinations(s.kinds[k], 1))) + (len(list(itertools.combinations(s.kinds[k], 2))) if len(s.kinds[k]) > 2 else 0) for k in top),
                            pairs_of_positive_worth=len(found), kept=rec, cpu=round(dt, 1)))
    # ---- sizes
    def withdrawn(s):
        """The addresses of R-cells whose X-cell is in a class of a kept pair (the class cell stands for them)."""
        out = []
        for sl in s.slots:
            if sl is None: continue
            ids_, cod = s.r_cells(sl["k"]); xk, _ = s.split(sl["k"], sl["X"], cod); out.append(ids_[sl["clsmap"][xk] >= 0])
        return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)
    def sizes(s):
        b = int((s.N[:s.nbase] > 0).sum()); c = int((s.N[s.nbase:s.top] > 0).sum()); w = int(len(s.withdrawn())) if s.P else 0
        return dict(bytes_read=int(s.read), base_cells=b, class_cells=c, cells=b + c, cells_with_merged_withdrawn=b + c - w, withdrawn=w,
                    counts_above_zero=int((s.C[:s.top] > 0).sum()), base_counts_above_zero=int((s.C[:s.nbase] > 0).sum()), trusts_used=int((s.Tu > 0).sum()), class_kinds=sum(sl is not None for sl in s.slots))
    def trust_summary(s):
        if not s.P: return {}
        u = s.Tu.sum(1); m = np.where(u > 0, (s.T * s.Tu).sum(1) / np.maximum(u, 1), 0.0); a = np.abs(s.T * (s.Tu > 0))
        return dict(mean_trust_class_kinds=[round(float(x), 4) for x in m[s.KB:]], largest_trust_class_kinds=[round(float(x), 4) for x in s.T[s.KB:].max(1)],
                    share_of_trust_in_class_kinds=round(float(a[s.KB:].sum() / max(a.sum(), 1e-12)), 4), mean_trust_base_top=[round(float(x), 4) for x in np.sort(m[:s.KB])[::-1][:8]])

# ---------------------------------------------------------------- two-digit sums (text and split of loop_small_case/cells2.py)
SYMS = list(b"0123456789+=\n-"); SI = {c: i for i, c in enumerate(SYMS)}; END = SI[10]
def text(a, b, nd=2): return [SI[c] for c in f"{a:0{nd}d}+{b:0{nd}d}={a + b:0{nd + 1}d}\n".encode()]

def sums_score(s, held, D=9, nd=2):
    Ls = np.array([text(a, b, nd) for a, b in held]); own = Ls.copy(); n = len(Ls); a0 = 2 * nd + 2
    for t in range(a0, a0 + nd + 1):
        ctx = np.full((n, D), END, np.int64)
        for d in range(1, D + 1):
            if t - d >= 0: ctx[:, d - 1] = own[:, t - d]
        own[:, t] = s.say(ctx).argmax(1)
    ok = own[:, a0:a0 + nd + 1] == Ls[:, a0:a0 + nd + 1]; return float(100 * ok.all(1).mean()), [float(100 * ok[:, i].mean()) for i in range(nd + 1)]

def carry_purity(rec):
    """For a kept pair whose X is two delays: over the members that are two digits, the share that lies with the majority of its class by 'the digits sum to ten or more'."""
    if rec["members_by_class"] is None or len(rec["X"]) != 2: return None
    tot = maj = 0; cl = []
    for c, mem in rec["members_by_class"].items():
        dd = [(q % 14, q // 14) for q in mem if q % 14 < 10 and q // 14 < 10]
        if len(dd) < 2: continue
        car = sum(a + b >= 10 for a, b in dd); tot += len(dd); maj += max(car, len(dd) - car); cl.append((len(dd), car))
    return dict(digit_members=tot, purity=round(100 * maj / tot, 1) if tot else None, classes=sorted(cl, reverse=True)[:12])

def run_sums(kw):
    seed = int(kw.get("seed", 1)); lines = int(kw.get("lines", 40000)); P = int(kw.get("P", 32)); eJ = float(kw.get("eJ", 0.02)); mode = kw.get("mode", "classes"); t0 = time.process_time(); D = 9
    nd = int(kw.get("nd", 2))
    if nd == 2:
        rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); pairs = [(int(q) // 100, int(q) % 100) for q in o]; held = pairs[:500]; shown = pairs[2000:]; rng.permutation(10000)   # the second draw is cells2.py's split for subtraction
        TAB = np.array([text(a, b) for a, b in shown], np.int64); draw = lambda k: TAB[rng.integers(8000, size=k)]
    else:                                                           # nano/ahead2/ahead3.py's split: the first fifth of a permutation of all pairs never shown, the first 500 scored
        n10 = 10 ** nd; rng = np.random.default_rng([seed, 99]); o = rng.permutation(n10 * n10); nh = n10 * n10 // 5; held = [(int(q) // n10, int(q) % n10) for q in o[:500]]; pool_ = o[nh:]
        def draw(k):
            q = pool_[rng.integers(len(pool_), size=k)]; return np.array([text(int(v) // n10, int(v) % n10, nd) for v in q], np.int64)
    kinds = [c for m in range(5) for c in itertools.combinations(range(1, D + 1), m)]
    s = Sys(14, D, kinds, 0.1, 2.0, 0.02, True, P=P, eJ=eJ, mode=mode, seed=seed, half_at=int(kw.get("half_at", lines * (2 * nd + 3) * 5 // 10)), M=int(kw.get("M", 16)), sweeps=int(kw.get("sweeps", 2)), rep=int(kw.get("rep", 0)), dissolve=int(kw.get("dissolve", 0)), pool=kw.get("pool"), practice=float(kw.get("practice", 0)), empty=int(kw.get("empty", 0)), keep=int(kw.get("keep", 0)))
    every = (5000 if lines <= 40000 else 50000) if nd == 2 else 10000; marks = sorted(set(list(range(every, lines + 1, every)) + [lines] + ([5000, 10000, 20000, 40000] if lines > 40000 else []))); tail = np.full(D, END, np.int64); rows = []; done = 0
    for mk in marks:
        X = draw(mk - done).ravel(); x = np.concatenate([tail, X]); s.learn(ctx_of(x, D, len(x), D), x[D:]); tail = x[-D:]; done = mk
        whole, place = sums_score(s, held, D, nd); row = dict(lines=mk, add=whole, place=place); row.update(s.sizes()); rows.append(row)
    out = dict(task="sums", args=kw, seed=seed, rows=rows, last=rows[-1]["add"], last3=float(np.mean([r["add"] for r in rows if r["lines"] % every == 0 or len(rows) < 3][-3:])), place_last=rows[-1]["place"], sizes=s.sizes(), trust=s.trust_summary())
    if s.P:
        ids_ = s.withdrawn(); keep = s.N[ids_].copy(); s.N[ids_] = 0; w, pl = sums_score(s, held, D, nd); s.N[ids_] = keep; out["with_merged_withdrawn"] = dict(add=w, place=pl)
        out["finds"] = [dict(f, kept=[dict(r, members_by_class=None, carry=carry_purity(r)) for r in f["kept"]]) for f in s.finds[:-1]] + [dict(f, kept=[dict(r, carry=carry_purity(r)) for r in f["kept"]]) for f in s.finds[-1:]]
    if s.P:                                                         # the same with only the class kinds speaking, and with only the plain kinds
        keep = s.N.copy(); s.N[:s.nbase] = 0; out["class_kinds_alone"] = sums_score(s, held, D, nd); s.N[:] = keep; s.N[s.nbase:] = 0; out["plain_kinds_alone"] = sums_score(s, held, D, nd); s.N[:] = keep
    out["cpu"] = round(time.process_time() - t0, 1); out["cpu_finding"] = round(s.tfind, 1); return out, s

# ---------------------------------------------------------------- prose (the life of nano/table/life.py, scored as nano/aheadprose/ahead.py does)
BT = ("bryant-stories.txt", "burgess-busterbrown.txt", "carroll-alice.txt")
def load_all(books=BT):
    import score
    xa, cut, _ = score.load(); vals = score.byte_values(); tab = np.full(256, 1, np.int64); tab[vals] = np.arange(65); parts = [("A", xa)]
    for f in books:
        raw = np.frombuffer(open(os.path.join(HERE, "data", f), "rb").read(), np.uint8); raw = raw[raw != 13]; parts.append((f.split(".")[0], tab[raw]))
    x = np.concatenate([p[1] for p in parts]); segs = {}; off = 0
    for name, p in parts:
        c = cut if name == "A" else int(0.9 * len(p)); segs[name] = (off, off + c, off + len(p)); off += len(p)
    return x, segs

def prose_sys(x, kw):
    K = 6; return Sys(65, K, [tuple(range(1, k + 1)) for k in range(K + 1)], 0.00005, 0.02, 0.25, False, text=x, P=int(kw.get("P", 32)), eJ=float(kw.get("eJ", 0.02)), mode=kw.get("mode", "classes"),
                      half_at=int(kw.get("half_at", 500000)), M=int(kw.get("M", 16)), sweeps=int(kw.get("sweeps", 2)), rep=int(kw.get("rep", 0)), dissolve=int(kw.get("dissolve", 0)), pool=kw.get("pool"), practice=float(kw.get("practice", 0)), empty=int(kw.get("empty", 0)), keep=int(kw.get("keep", 0)))

def prose_scores(s, x, lo, hi):
    import score
    r = score.score(s.say(ctx_of(x, lo, hi, s.D)), x[lo:hi]); out = dict(nats=r["nats"], right=r["right"])
    if s.P:
        ids_ = s.withdrawn(); keep = s.N[ids_].copy(); s.N[ids_] = 0; out["nats_with_merged_withdrawn"] = score.score(s.say(ctx_of(x, lo, hi, s.D)), x[lo:hi])["nats"]; s.N[ids_] = keep
    out.update(s.sizes()); return out

def strip(finds): return [dict(f, kept=[dict(r, members_by_class=None) for r in f["kept"]]) for f in finds[:-1]] + finds[-1:]

def run_dev(kw):
    import score
    x, cut, _ = score.load(); x = x[:cut]; d = cut - cut // 20; t0 = time.process_time(); s = prose_sys(x, kw); s.learn(ctx_of(x, s.D, d, s.D), x[s.D:d])
    out = dict(task="prose_dev", args=kw); out.update(prose_scores(s, x, d, cut)); out.update(trust=s.trust_summary(), finds=strip(s.finds), cpu=round(time.process_time() - t0, 1), cpu_finding=round(s.tfind, 1)); return out, s

def run_life(kw):
    x, segs = load_all(tuple(b + ".txt" for b in kw["books"].split(",")) if "books" in kw else BT); t0 = time.process_time(); s = prose_sys(x, kw); out = dict(task="prose_life", args=kw, segments={k: list(map(int, v)) for k, v in segs.items()}, scores={}, growth=[])
    for name, (a, c, b) in segs.items():
        lo = max(a, s.D)
        for m in range(lo, c, 125000):
            hi = min(c, m + 125000); s.learn(ctx_of(x, m, hi, s.D), x[m:hi]); out["growth"].append(s.sizes())
        if name != "A": out["scores"][name] = prose_scores(s, x, c, b)
    if int(kw.get("scoreA", 0)): a, c, b = segs["A"]; out["scores"]["A_at_the_end"] = prose_scores(s, x, c, b)
    out.update(trust=s.trust_summary(), finds=strip(s.finds), sizes=s.sizes(), cpu=round(time.process_time() - t0, 1), cpu_finding=round(s.tfind, 1)); return out, s

# ---------------------------------------------------------------- checks
def check():
    ok = True; rng = np.random.default_rng(5); L = kern()
    # 1. the compiled answering loop against numpy, on addresses of a small system with classes
    for S, shrink in ((14, True), (65, False)):
        K1, n, nc = 9, 3000, 400; ids = rng.integers(-1, nc, (n, K1)).astype(np.int32); sil = rng.random((n, K1)) < 0.1; ids[sil & (ids >= 0)] = -2 - ids[sil & (ids >= 0)]; y = rng.integers(0, S, n).astype(np.int32)
        C = np.zeros((nc, S), np.float32); N = np.zeros(nc, np.float32); U = np.zeros(nc, np.float32); T = np.full((K1, NL), 0.1); Tu = np.zeros((K1, NL)); P = np.zeros((n, S))
        L.mix_steps(n, K1, S, p32(ids), p32(y), pf(C), pf(N), pf(U), pd(T), pd(Tu), 0.05, 0.3, 1, int(shrink), pd(P))
        C2 = np.zeros((nc, S), np.float32); N2 = np.zeros(nc, np.float32); U2 = np.zeros(nc, np.float32); T2 = np.full((K1, NL), 0.1); Tu2 = np.zeros((K1, NL)); P2 = steps_np(ids, y, C2, N2, U2, T2, Tu2, 0.05, 0.3, True, shrink)
        d = max(np.abs(P - P2).max(), np.abs(T - T2).max(), np.abs(C - C2).max()); ok &= d < 1e-9; print(f"answering loop, S = {S}, shrinking step = {shrink}, {n} bytes: (a tenth of the cells silent and counting) compiled against numpy, largest difference {d:.1e}", "PASSED" if d < 1e-9 else "FAILED")
    # 2. the search for classes against brute force in plain Python, on the counts of a small sums system
    kw = dict(seed="1", lines="300", P="4", mode="none"); _, s = run_sums_small(kw); s.eJ = 0.02; s.sweeps = 2
    def canon(c):
        first = {}; return [first.setdefault(v, i) for i, v in enumerate(c)]
    for k, X, dis in ((s.kinds.index((3, 4, 6, 7)), (3, 6), 0), (s.kinds.index((1, 2, 3)), (2,), 0), (s.kinds.index((2, 4, 5)), (2, 5), 0), (s.kinds.index((3, 4, 6, 7)), (3, 6), 1), (s.kinds.index((2, 4, 5)), (2, 5), 1), (s.kinds.index((1, 2, 3)), (1, 3), 1), (s.kinds.index((3, 4, 6, 7)), (3, 6), 2), (s.kinds.index((2, 4, 5)), (2, 5), 2), (s.kinds.index((1, 2, 3)), (1, 3), 2)):
        s.dissolve = dis; r = s.search(k, X); ids_, cod = s.r_cells(k); xk, zk = s.split(k, X, cod); ux, xc = np.unique(xk, return_inverse=True)
        cells = [(int(xc[i]), int(zk[i]), {int(v): float(s.C[ids_[i], v]) for v in np.flatnonzero(s.C[ids_[i]])}) for i in range(len(cod))]
        xn = np.bincount(xc, weights=s.N[ids_], minlength=len(ux)); xord = [int(q) for q in np.argsort(-xn, kind="stable")]
        ref = find_py(cells, len(ux), xord, s.S, s.eJ, s.sweeps, dis); Jr = j_py(cells, ref, s.S, s.eJ); same = canon(list(r["cls"])) == canon(ref); dj = abs(Jr - r["J"])
        good = same and dj < 1e-6 * max(1, abs(Jr)); ok &= good
        print(f"search for classes{('', ', with dissolving', ', with splitting')[dis]}, kind {s.kinds[k]}, X = {X}: {len(ux)} X-cells, {len(cod)} R-cells, {r['classes']} classes of two or more; compiled against plain Python: the same grouping {same}, J differs by {dj:.1e}", "PASSED" if good else "FAILED")
    # 3. a class cell holds the sum of its members' R-cells, after a finding and after further reading
    out, s = run_sums_small(dict(seed="1", lines="3000", P="8", rep="1")); worst = 0.0; nchk = 0
    for sl in s.slots:
        if sl is None: continue
        ids_, cod = s.r_cells(sl["k"]); xk, zk = s.split(sl["k"], sl["X"], cod); c = sl["clsmap"][xk]; okc = c >= 0; key = c[okc].astype(np.int64) * sl["ZS"] + zk[okc]
        j = np.searchsorted(sl["keys"], key); assert (sl["keys"][j] == key).all(); pooled = np.zeros((len(sl["keys"]), s.S)); np.add.at(pooled, j, s.C[ids_[okc]]); worst = max(worst, float(np.abs(pooled - s.C[sl["cids"]]).max())); nchk += len(sl["keys"])
    ok &= worst == 0 and nchk > 0; print(f"class cells against the sum of their members' cells, {nchk} class cells of {sum(sl is not None for sl in s.slots)} kinds after 3,000 lines with replacement: largest difference {worst}", "PASSED" if worst == 0 and nchk > 0 else "FAILED")
    return ok

def run_sums_small(kw):
    return run_sums(kw)

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=") for a in sys.argv[2:]); out, s = dict(sums=run_sums, dev=run_dev, life=run_life)[sys.argv[1]](kw)
    tag = kw.get("tag") or sys.argv[1] + "_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items())); os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", tag + ".json"), "w"))
    if sys.argv[1] == "sums":
        print(tag, "| last3", round(out["last3"], 1), "last", out["last"], "place", [round(v, 1) for v in out["place_last"]], "|", " ".join(f"{r['lines'] // 1000}k:{r['add']:.1f}" for r in out["rows"]), "| withdrawn:", out.get("with_merged_withdrawn"), "| cells", out["sizes"]["cells"], "after withdrawal", out["sizes"]["cells_with_merged_withdrawn"], "| cpu", out["cpu"], "finding", out["cpu_finding"], flush=True)
    else:
        print(tag, "|", {k: (round(v["nats"], 4), round(v.get("nats_with_merged_withdrawn", 0), 4)) for k, v in out["scores"].items()} if "scores" in out else (round(out["nats"], 4), round(out.get("nats_with_merged_withdrawn", 0), 4)), "| cells", (out.get("sizes") or out)["cells"], "after withdrawal", (out.get("sizes") or out)["cells_with_merged_withdrawn"], "| cpu", out["cpu"], "finding", out["cpu_finding"], flush=True)
