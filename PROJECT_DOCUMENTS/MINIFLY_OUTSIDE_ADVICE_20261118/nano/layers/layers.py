#!/usr/bin/env python3
"""nano/layers: further answering cells above the first, each with families of gated parts of the trust (PLAN_1.md and later plans).
The base systems are imported unchanged: nano/merge/merge.py (sums; kinds of at most four of nine delays, with classes) and nano/table/all.py
(prose; T2 = the table's cells and the cells that step once a word). Their kernels are taken as text and one call is inserted after the first cell
has answered and before anything is learned; nothing flows back, so the first cell is the base system's to the last digit in the same run.

    python3 layers.py check
    python3 layers.py sums seed=1 arms=step1 [lines=40000] [mode=classes]
    python3 layers.py life books=worn|new|edge arms=step1

A further cell c hears the same senders and the branches of one cell below it (the first cell, or another further cell):
    a_c(v) = a_below(v) + sum over speaking senders k of [ sum over the cell's families f of G_f[kind k][level of k][gate value of f] ] * s_k(v)
Every part G is born at nothing and changes by the system's own step times the cell's own mismatch for that sender, s_k(y) - sum_v p_c(v) s_k(v).
A reading cell ('mix') hears the log-probabilities of several cells with one trust for each (born at 1 / their number) and learns the same way.
Families (what the gate value is):
    ("byte", d)          the byte at delay d                                  ("pair", d1, d2)   the two bytes at delays d1 and d2 together
    ("const",)           a constant (the control)                             ("sure", c)        the largest probability of cell c (0: the first), ten bins
    ("own",)             no gate: one number for each cell (a trust the sending cell owns)
    ("slev", q)          the level of counts of the sender of kind q (0 when it is silent)
    ("cls",)             the class, at the class kind with the largest trust, of the bytes that kind reads (systems with classes only)
Options after the tuple, as a dict: kind=0 (the part is not indexed by the kind of sender), level=0 (not by its level of counts)."""
import ctypes, hashlib, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "merge")); sys.path.insert(0, os.path.join(HERE, "..", "table"))
NL = 40; DG = 16; KM = 400
PD = ctypes.POINTER(ctypes.c_double); PF = ctypes.POINTER(ctypes.c_float); PI = ctypes.POINTER(ctypes.c_int32); PL = ctypes.POINTER(ctypes.c_int64)
ROOT = os.path.join(HERE, "..", "..", "..", "..")
BOOKS = dict(worn=[("minifly_v148_pretrain_probe_dev_20261025", b) for b in ("bryant-stories", "burgess-busterbrown", "carroll-alice")],
             new=[("minifly_v155_keys_dev_v1_20261026", b) for b in ("chesterton-ball", "austen-emma", "melville-moby_dick")],
             edge=[("minifly_v149_input_ceiling_dev_20261025", "edgeworth-parents")])

UP = r'''
#include <math.h>
#include <stdint.h>
#include <string.h>
#define NL 40
#define SS 65
#define KM 400
#define MAXC 64
#define MAXF 40
typedef struct { int32_t type, col, nv, usekind, uselev, pad; double *g, *gu; } Fam;
typedef struct { int32_t mix, inp, f0, nf; double *w, *wu; } Cell;
static struct { int nc, F, shrink; double eta; const Cell *cell; const Fam *fam; const int32_t *tv; double *lp; int32_t *arg; } Q;
void up_set(int nc, const Cell *cell, const Fam *fam, const int32_t *tv, int F, double eta, int shrink, double *lp, int32_t *arg) {
    Q.nc = nc; Q.cell = cell; Q.fam = fam; Q.tv = tv; Q.F = F; Q.eta = eta; Q.shrink = shrink; Q.lp = lp; Q.arg = arg;
}
static void finish(const double *a, int S, double *la, double *p) {
    double mx = -1e300, z = 0; for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
    for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
    const double lz = log(z); for (int v = 0; v < S; v++) { p[v] /= z; la[v] = a[v] - mx - lz; }
}
/* called once a byte by the base kernel, after its answering cell has answered (branches a, probabilities p) and before anything is learned.
   lev[k] >= 0: sender k speaks, at that level, with the message s[k * SS + v]; uid[k] its cell. Cell 0 below is the first cell; further cell c is c + 1. */
void up_step(int64_t t, int K1, int S, const int *lev, const double *s, const int64_t *uid, const double *a, const double *p, int y, int learn) {
    static double A[MAXC + 1][SS], LA[MAXC + 1][SS], PM[MAXC + 1]; static double *gp[MAXF][KM], *gup[MAXF][KM]; static int spk[KM]; double pc[SS]; int m = 0;
    if (!Q.nc) return;
    for (int k = 0; k < K1; k++) if (lev[k] >= 0) spk[m++] = k;
    { double mx = 0; int am = 0; for (int v = 0; v < S; v++) { A[0][v] = a[v]; LA[0][v] = log(p[v]); if (p[v] > mx) { mx = p[v]; am = v; } }
      PM[0] = mx; if (Q.lp) { Q.lp[t * (Q.nc + 1)] = LA[0][y]; Q.arg[t * (Q.nc + 1)] = am; } }
    for (int c = 0; c < Q.nc; c++) {
        const Cell *ce = Q.cell + c; double *ac = A[c + 1], *la = LA[c + 1];
        if (ce->mix) {
            for (int v = 0; v < S; v++) ac[v] = 0;
            for (int j = 0; j < ce->nf; j++) { const double w = ce->w[j], *li = LA[ce->f0 + j + 1]; for (int v = 0; v < S; v++) ac[v] += w * li[v]; }
            finish(ac, S, la, pc);
            if (learn) for (int j = 0; j < ce->nf; j++) { const double *li = LA[ce->f0 + j + 1]; double g = li[y]; for (int v = 0; v < S; v++) g -= pc[v] * li[v];
                ce->w[j] += (Q.shrink ? Q.eta / sqrt(1.0 + ce->wu[j]) : Q.eta) * g; ce->wu[j] += 1.0; }
        } else {
            const double *ai = A[ce->inp + 1]; for (int v = 0; v < S; v++) ac[v] = ai[v];
            for (int f = 0; f < ce->nf; f++) {
                const Fam *fa = Q.fam + ce->f0 + f; int gv0 = 0;
                if (fa->type == 0) gv0 = Q.tv[t * Q.F + fa->col];
                else if (fa->type == 1) { gv0 = (int)(PM[fa->col] * fa->nv); if (gv0 > fa->nv - 1) gv0 = fa->nv - 1; }
                else if (fa->type == 3) gv0 = lev[fa->col] + 1;
                for (int i = 0; i < m; i++) { const int k = spk[i]; int64_t ix;
                    if (fa->type == 2) ix = uid[k];
                    else ix = ((int64_t)(fa->usekind ? k : 0) * (fa->uselev ? NL : 1) + (fa->uselev ? lev[k] : 0)) * fa->nv + gv0;
                    gp[f][i] = fa->g + ix; gup[f][i] = fa->gu + ix; }
            }
            for (int i = 0; i < m; i++) { double tk = 0; for (int f = 0; f < ce->nf; f++) tk += *gp[f][i];
                if (tk != 0) { const double *sk = s + (int64_t)spk[i] * SS; for (int v = 0; v < S; v++) ac[v] += tk * sk[v]; } }
            finish(ac, S, la, pc);
            if (learn) for (int i = 0; i < m; i++) { const double *sk = s + (int64_t)spk[i] * SS; double g = sk[y]; for (int v = 0; v < S; v++) g -= pc[v] * sk[v];
                for (int f = 0; f < ce->nf; f++) { *gp[f][i] += (Q.shrink ? Q.eta / sqrt(1.0 + *gup[f][i]) : Q.eta) * g; *gup[f][i] += 1.0; } }
        }
        { double mx = 0; int am = 0; for (int v = 0; v < S; v++) if (pc[v] > mx) { mx = pc[v]; am = v; }
          PM[c + 1] = mx; if (Q.lp) { Q.lp[t * (Q.nc + 1) + c + 1] = la[y]; Q.arg[t * (Q.nc + 1) + c + 1] = am; } }
    }
}
'''
PROTO = "#include <stdint.h>\nvoid up_step(int64_t, int, int, const int *, const double *, const int64_t *, const double *, const double *, int, int);\n"

class CFam(ctypes.Structure): _fields_ = [("type", ctypes.c_int32), ("col", ctypes.c_int32), ("nv", ctypes.c_int32), ("usekind", ctypes.c_int32), ("uselev", ctypes.c_int32), ("pad", ctypes.c_int32), ("g", PD), ("gu", PD)]
class CCell(ctypes.Structure): _fields_ = [("mix", ctypes.c_int32), ("inp", ctypes.c_int32), ("f0", ctypes.c_int32), ("nf", ctypes.c_int32), ("w", PD), ("wu", PD)]

def build(name, base_src):
    full = PROTO + base_src; h = hashlib.md5((full + UP).encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_{name}_{h}.so")
    if not os.path.exists(so):
        b = os.path.join(HERE, f"kern_{name}.c"); u = os.path.join(HERE, "kern_up.c"); open(b, "w").write(full); open(u, "w").write(UP); tmp = so + f".{os.getpid()}"
        subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, b, u, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); L.up_set.argtypes = [ctypes.c_int, ctypes.POINTER(CCell), ctypes.POINTER(CFam), PI, ctypes.c_int, ctypes.c_double, ctypes.c_int, PD, PI]; L.up_set.restype = None; return L

def kern_m():
    import merge as G
    HOOK = "        { static int64_t uid[KMAX]; for (int k = 0; k < K1; k++) uid[k] = id[k]; up_step(t, K1, S, lev, &s[0][0], uid, a, p, y[t], learn); }\n"
    mark = "        if (learn) {\n            for (int k = 0; k < K1; k++) if (lev[k] >= 0) {\n                double g = s[k][y[t]]; for (int v = 0; v < S; v++) g -= p[v] * s[k][v];\n                T[k * NL + lev[k]] += (shrink"
    assert G.SRC.count(mark) == 1 and G.SRC.count("void mix_steps(") == 1
    L = build("lm", G.SRC.replace(mark, HOOK + mark).replace("void mix_steps(", "void lay_steps(")); ci, cd = ctypes.c_int, ctypes.c_double
    L.lay_steps.argtypes = [ctypes.c_int64, ci, ci, PI, PI, PF, PF, PF, PD, PD, cd, cd, ci, ci, PD]; L.lay_steps.restype = None; return L

def kern_a():
    import all as A
    HOOK = "        up_step(t, (int)K1, S, lev, &s[0][0], id, a, p, y[t], learn);\n"; mark = "        if (learn) {\n"
    assert A.SRC.count(mark) == 1 and A.SRC.count("void all_steps(") == 1
    L = build("la", A.SRC.replace(mark, HOOK + mark).replace("void all_steps(", "void lay_steps("))
    L.lay_steps.argtypes = [ctypes.c_int64, ctypes.c_int64, PL, PI, PF, PF, PF, PD, ctypes.c_double, ctypes.c_double, ctypes.c_int, PD, PD, PD, ctypes.c_double, PF, PI]; L.lay_steps.restype = None; return L

# ---------------------------------------------------------------- the stack of further cells
TYPES = dict(byte=0, pair=0, const=0, cls=0, bytes=0, sure=1, own=2, slev=3)
def norm(f):
    """A family as (tuple, options)."""
    if isinstance(f[-1], dict): return tuple(f[:-1]), dict(f[-1])
    return tuple(f), {}

class Stack:
    """cells: a list of dicts. dict(inp=-1 or the index of a cell below, fams=[...]) or dict(mix=(first, number)); each may carry name=..."""
    def __init__(s, K1, S, cells, eta, shrink, ncells=0):
        s.K1, s.S, s.eta, s.shrink, s.cells = K1, S, eta, int(shrink), cells; s.fams = []; s.cols = []; s.G = []; s.Gu = []; s.W = []; s.Wu = []; s.cspec = []
        assert len(cells) <= 64
        for c, ce in enumerate(cells):
            if "mix" in ce:
                f0, nf = ce["mix"]; assert f0 + nf <= c; s.W.append(np.full(nf, 1.0 / nf)); s.Wu.append(np.zeros(nf)); s.cspec.append((1, -1, f0, nf, len(s.W) - 1)); continue
            assert -1 <= ce["inp"] < c and len(ce["fams"]) <= 40; f0 = len(s.fams)
            for f in ce["fams"]:
                tup, opt = norm(f); ty = TYPES[tup[0]]; uk, ul = int(opt.get("kind", 1)), int(opt.get("level", 1)); col = 0
                if ty == 0:
                    if tup not in s.cols: s.cols.append(tup)
                    col = s.cols.index(tup); nv = dict(byte=S, pair=S * S, const=1, cls=S * S + 1, bytes=0)[tup[0]]
                    if tup[0] == "bytes": nv = S ** len(tup[1])
                elif ty == 1: col = tup[1]; nv = 10; assert col <= c
                elif ty == 3: col = tup[1]; nv = NL + 1
                else: nv = 1
                size = max(ncells, 1024) if ty == 2 else (K1 if uk else 1) * (NL if ul else 1) * nv
                s.fams.append(dict(tup=tup, type=ty, col=col, nv=nv, uk=uk, ul=ul, cell=c)); s.G.append(np.zeros(size)); s.Gu.append(np.zeros(size))
            s.cspec.append((0, ce["inp"], f0, len(s.fams) - f0, -1))
        s.nc = len(cells); s.F = max(1, len(s.cols)); s.bind()
    def bind(s):
        s.cf = (CFam * max(1, len(s.fams)))(); s.cc = (CCell * s.nc)()
        for i, f in enumerate(s.fams): s.cf[i] = CFam(f["type"], f["col"], f["nv"], f["uk"], f["ul"], 0, s.G[i].ctypes.data_as(PD), s.Gu[i].ctypes.data_as(PD))
        for i, (mix, inp, f0, nf, w) in enumerate(s.cspec): s.cc[i] = CCell(mix, inp, f0, nf, s.W[w].ctypes.data_as(PD) if mix else None, s.Wu[w].ctypes.data_as(PD) if mix else None)
    def grow(s, ncells):
        ch = False
        for i, f in enumerate(s.fams):
            if f["type"] == 2 and len(s.G[i]) < ncells:
                for A_ in (s.G, s.Gu): new = np.zeros(int(ncells * 1.5) + 1024); new[:len(A_[i])] = A_[i]; A_[i] = new
                ch = True
        if ch: s.bind()
    def tv(s, g, system=None):
        """The gate values that the text alone gives: one column for each such family. g: (n, DG), g[:, d - 1] = the byte at delay d."""
        out = np.zeros((len(g), s.F), np.int32)
        for j, tup in enumerate(s.cols):
            if tup[0] == "byte": out[:, j] = g[:, tup[1] - 1]
            elif tup[0] == "pair": out[:, j] = g[:, tup[1] - 1].astype(np.int64) * s.S + g[:, tup[2] - 1]
            elif tup[0] == "bytes": out[:, j] = sum(g[:, d - 1].astype(np.int64) * s.S ** i for i, d in enumerate(tup[1]))
            elif tup[0] == "cls": out[:, j] = system.cls_value(g)
        return np.ascontiguousarray(out)
    def set(s, L, tv, n, want):
        lp = np.zeros((n, s.nc + 1)) if want else None; arg = np.zeros((n, s.nc + 1), np.int32) if want else None; s._hold = (tv, lp, arg)
        L.up_set(s.nc, s.cc, s.cf, tv.ctypes.data_as(PI), s.F, s.eta, s.shrink, lp.ctypes.data_as(PD) if want else None, arg.ctypes.data_as(PI) if want else None); return lp, arg
    def parts(s):
        """For each cell: the parts it holds and the parts that have been used."""
        out = []
        for c in range(s.nc):
            if s.cspec[c][0]: w = s.cspec[c][4]; out.append((int(s.W[w].size), int((s.Wu[w] > 0).sum())))
            else: ii = [i for i, f in enumerate(s.fams) if f["cell"] == c]; out.append((int(sum(s.G[i].size for i in ii)), int(sum((s.Gu[i] > 0).sum() for i in ii))))
        return out
    def view(s, i):
        f = s.fams[i]; return s.G[i].reshape((s.K1 if f["uk"] else 1, NL if f["ul"] else 1, f["nv"])), s.Gu[i].reshape((s.K1 if f["uk"] else 1, NL if f["ul"] else 1, f["nv"]))

# ---------------------------------------------------------------- the reference in numpy (base loop of nano/merge's form, and the stack)
def level_np(n, u):
    bn = 0; m = int(n)
    while m > 1 and bn < 9: m >>= 1; bn += 1
    return bn * 4 + (0 if u <= 1 else 1 if u <= 2 else 2 if u <= 4 else 3)

def ref_steps(ids, y, tv, C, N, U, T, Tu, st, eta, e, shrink, learn=True):
    """Plain numpy. st: a Stack whose arrays are changed in place. Returns the log-probability of y and the answer, for the first cell and each further cell."""
    S = C.shape[1]; n = len(y); LP = np.zeros((n, st.nc + 1)); AR = np.zeros((n, st.nc + 1), np.int32); step = lambda u: eta / np.sqrt(1.0 + u) if shrink else eta
    for t in range(n):
        a = np.zeros(S); ss = []; lev = -np.ones(ids.shape[1], np.int64)
        for k, i in enumerate(ids[t]):
            if i >= 0 and N[i] >= 1:
                l = level_np(N[i], U[i]); lev[k] = l; sv = np.log(C[i].astype(np.float64) + e) - np.log(N[i] + S * e); ss.append((k, l, sv, int(i))); a += T[k, l] * sv
        p = np.exp(a - a.max()); p /= p.sum(); As = [a]; LAs = [np.log(p)]; PM = [p.max()]; LP[t, 0] = LAs[0][y[t]]; AR[t, 0] = p.argmax()
        for c, (mix, inp, f0, nf, w) in enumerate(st.cspec):
            if mix:
                ac = sum(st.W[w][j] * LAs[f0 + j + 1] for j in range(nf)); pc = np.exp(ac - ac.max()); pc /= pc.sum()
                if learn:
                    for j in range(nf): li = LAs[f0 + j + 1]; st.W[w][j] += step(st.Wu[w][j]) * (li[y[t]] - pc @ li); st.Wu[w][j] += 1
            else:
                ac = As[inp + 1].copy(); used = []
                for k, l, sv, i in ss:
                    ix = []
                    for f in range(f0, f0 + nf):
                        fa = st.fams[f]
                        if fa["type"] == 2: ix.append((f, i)); continue
                        gv = int(tv[t, fa["col"]]) if fa["type"] == 0 else min(fa["nv"] - 1, int(PM[fa["col"]] * fa["nv"])) if fa["type"] == 1 else int(lev[fa["col"]]) + 1
                        ix.append((f, ((k if fa["uk"] else 0) * (NL if fa["ul"] else 1) + (l if fa["ul"] else 0)) * fa["nv"] + gv))
                    ac += sum(st.G[f][j] for f, j in ix) * sv; used.append(ix)
                pc = np.exp(ac - ac.max()); pc /= pc.sum()
                if learn:
                    for (k, l, sv, i), ix in zip(ss, used):
                        g = sv[y[t]] - pc @ sv
                        for f, j in ix: st.G[f][j] += step(st.Gu[f][j]) * g; st.Gu[f][j] += 1
            As.append(ac); LAs.append(ac - ac.max() - np.log(np.exp(ac - ac.max()).sum())); PM.append(pc.max()); LP[t, c + 1] = LAs[-1][y[t]]; AR[t, c + 1] = pc.argmax()
        if learn:
            for k, l, sv, i in ss: T[k, l] += step(Tu[k, l]) * (sv[y[t]] - p @ sv); Tu[k, l] += 1.0
            for i in ids[t]:
                if i < -1: i = -2 - i
                if i >= 0:
                    if C[i, y[t]] == 0: U[i] += 1
                    C[i, y[t]] += 1; N[i] += 1
    return LP, AR

# ---------------------------------------------------------------- arms: the cells of each plan (S byte values, D delay lines, K1 kinds, P class kinds)
def arms(name, S, D, K1, classes=False):
    dl = lambda ds, **o: [("byte", d, o) if o else ("byte", d) for d in ds]; allD = list(range(1, D + 1)); cell = lambda nm, fams, inp=-1: dict(name=nm, inp=inp, fams=fams)
    if name == "second": return [cell("second", dl(allD))]
    if name == "step1":
        return ([cell("second", dl(allD))] + [cell(f"delay{j}", dl([j])) for j in allD] + [cell("delays1-2", dl([1, 2])), cell("delays1-3", dl([1, 2, 3]))] +
                [cell("no-level", dl(allD, level=0)), cell("no-kind", dl(allD, kind=0)), cell("byte-only", dl(allD, kind=0, level=0)), cell("control", [("const",)] * D)])
    if name == "step2":                                              # other gates, alone and beside the delays' parts
        g = [("pair", ("pair", 1, 2)), ("own", ("own",)), ("sure", ("sure", 0))] + ([("class", ("cls",))] if classes else [])
        return ([cell("second", dl(allD))] + [cell(n, [f]) for n, f in g] + [cell("second+" + n, dl(allD) + [f]) for n, f in g] + [cell("second+all", dl(allD) + [f for _, f in g])] +
                [cell("second+pair+sure", dl(allD) + [g[0][1], g[2][1]]), cell("second+pair+own", dl(allD) + [g[0][1], g[1][1]]), cell("second+own+sure", dl(allD) + [g[1][1], g[2][1]])])
    raise KeyError(name)

# ---------------------------------------------------------------- sums: nano/merge's system
def sum_sys():
    import merge as G
    class SumSys(G.Sys):
        def start(s, cells_of):
            s.L2 = kern_m(); s.cells = cells_of(s.S, s.D, s.K1); s.stack = Stack(s.K1, s.S, s.cells, s.eta, s.shrink, len(s.N)); s.clsslot = -1
        def cls_value(s, g):
            if s.clsslot < 0 or s.slots[s.clsslot] is None: return np.zeros(len(g), np.int32)
            sl = s.slots[s.clsslot]; return sl["clsmap"][G.code(g, sl["X"], s.S)] + 1
        def _run(s, ctx, y, learn, want_p):
            n = len(ctx); LP = np.zeros((n, s.stack.nc + 1)) if want_p else None; AR = np.zeros((n, s.stack.nc + 1), np.int32) if want_p else None
            for a in range(0, n, 50000):
                idd = s.ids(ctx[a:a + 50000], create=learn); yy = np.ascontiguousarray(y[a:a + 50000].astype(np.int32)) if y is not None else np.zeros(len(idd), np.int32)
                s.stack.grow(len(s.N)); tv = s.stack.tv(ctx[a:a + 50000], s); lp, ar = s.stack.set(s.L2, tv, len(idd), want_p)
                s.L2.lay_steps(len(idd), s.K1, s.S, G.p32(idd), G.p32(yy), G.pf(s.C), G.pf(s.N), G.pf(s.U), G.pd(s.T), G.pd(s.Tu), s.eta, s.e, int(learn), s.shrink, None)
                if want_p: LP[a:a + 50000] = lp; AR[a:a + 50000] = ar
            return (LP, AR) if want_p else None
        def find(s):
            """Classes are found anew by nano/merge's find. A further cell's parts for a class kind follow the first cell's trust for it: kept where the pair is kept, nothing where
            it is new. Parts owned by class cells, parts gated by a class or by the level of a class kind begin again at nothing."""
            old = {(sl["k"], sl["X"]): p for p, sl in enumerate(s.slots) if sl is not None}; st = s.stack; keep = [(g.copy(), u.copy()) for g, u in zip(st.G, st.Gu)]; super().find()
            mc = [v for _, v in s.child_trust()]; live = [p for p, sl in enumerate(s.slots) if sl is not None]; s.clsslot = live[int(np.argmax(mc))] if live else -1
            for i, f in enumerate(st.fams):
                if f["type"] == 2: st.G[i][s.nbase:] = 0; st.Gu[i][s.nbase:] = 0; continue
                if f["tup"][0] == "cls" or (f["type"] == 3 and f["col"] >= s.KB): st.G[i][:] = 0; st.Gu[i][:] = 0; continue
                if not f["uk"]: continue
                g, u = st.view(i); g0 = keep[i][0].reshape(g.shape); u0 = keep[i][1].reshape(u.shape); g[s.KB:] = 0; u[s.KB:] = 0
                for p, sl in enumerate(s.slots):
                    if sl is not None and (sl["k"], sl["X"]) in old: q = old[(sl["k"], sl["X"])]; g[s.KB + p] = g0[s.KB + q]; u[s.KB + p] = u0[s.KB + q]
    return G, SumSys

def run_sums(kw, cells_of=None):
    import itertools
    G, SumSys = sum_sys(); seed = int(kw.get("seed", 1)); lines = int(kw.get("lines", 40000)); mode = kw.get("mode", "classes"); D = 9; t0 = time.process_time()
    cells_of = cells_of or (lambda S, D_, K1: arms(kw.get("arms", "second"), S, D_, K1, mode == "classes"))
    rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); pairs = [(int(q) // 100, int(q) % 100) for q in o]; held = pairs[:500]; shown = pairs[2000:]; rng.permutation(10000)
    TAB = np.array([G.text(a, b) for a, b in shown], np.int64); draw = lambda k: TAB[rng.integers(8000, size=k)]
    Ls = np.array([G.text(a, b) for a, b in held]); g = np.random.default_rng([seed, 7]); prevs = [TAB[g.integers(8000, size=500)] for _ in range(5)]
    kinds = [c for m in range(5) for c in itertools.combinations(range(1, D + 1), m)]
    s = SumSys(14, D, kinds, 0.1, 2.0, 0.02, True, P=32, eJ=0.02, mode=mode, seed=seed); s.start(cells_of); nc = s.stack.nc + 1
    def one(prev):
        """Whole answers right, %, for every cell at once; each cell goes on from its own answer bytes."""
        own = np.repeat(Ls[None], nc, 0).copy()                    # (cell, line, byte)
        for t in (6, 7, 8):
            ctx = np.full((nc, 500, DG), G.END, np.int64)
            for d in range(1, DG + 1):
                if t - d >= 0: ctx[:, :, d - 1] = own[:, :, t - d]
                elif prev is not None and 10 + t - d >= 0: ctx[:, :, d - 1] = prev[None, :, 10 + t - d]
            if t == 6: ar = s.say(ctx[0])[1]; own[:, :, t] = ar.T
            else: ar = s.say(ctx.reshape(nc * 500, DG))[1].reshape(nc, 500, nc); own[:, :, t] = ar[np.arange(nc), :, np.arange(nc)]
        ok = own[:, :, 6:9] == Ls[None, :, 6:9]; return 100 * ok.all(2).mean(1), 100 * ok.mean(1)
    marks = sorted(set(list(range(5000, lines + 1, 5000 if lines <= 40000 else 50000)) + [lines])); tail = np.full(DG, G.END, np.int64); readings = []; done = 0
    for mk in marks:
        X = draw(mk - done).ravel(); x = np.concatenate([tail, X]); s.learn(G.ctx_of(x, DG, len(x), DG), x[DG:]); tail = x[-DG:]; done = mk
        if mk in marks[-3:]:
            e = one(None); a = [one(pv) for pv in prevs]; readings.append(dict(lines=mk, empty=e[0].tolist(), after=np.mean([q[0] for q in a], 0).tolist(), after_place=np.mean([q[1] for q in a], 0).tolist()))
    names = ["first"] + [c.get("name", f"cell{i}") for i, c in enumerate(s.cells)]; pr = s.stack.parts(); res = dict(task="sums", args=kw, seed=seed, names=names, readings=readings, sizes=s.sizes(), cells={})
    for i, nm in enumerate(names):
        res["cells"][nm] = dict(last_after=readings[-1]["after"][i], last3_after=float(np.mean([r["after"][i] for r in readings])), last_empty=readings[-1]["empty"][i], last3_empty=float(np.mean([r["empty"][i] for r in readings])),
                                place_after=readings[-1]["after_place"][i], parts=pr[i - 1][0] if i else int(s.T.size), parts_used=pr[i - 1][1] if i else int((s.Tu > 0).sum()))
    res["cpu"] = round(time.process_time() - t0, 1); return res, s

# ---------------------------------------------------------------- prose: nano/table/all.py's system (T2 by default)
T2 = ("companions", "relay", "since", "pool")
def prose_sys():
    import all as A
    class ProseSys(A.SysAll):
        def start(s, cells_of):
            assert s.X is None, "the relay system's opinions are not used in this line"; s.L2 = kern_a(); s.S = 65; s.D = s.K; s.cells = cells_of(65, s.K, s.K1); s.stack = Stack(s.K1, 65, s.cells, A.ETA, 0, len(s.N))
            x = s.x; s.g = np.ascontiguousarray(np.stack([np.concatenate([np.zeros(d, x.dtype), x[:-d]]) for d in range(1, DG + 1)], 1).astype(np.int32))
        def go(s, lo, hi, learn, want=False):
            LP = np.zeros((hi - lo, s.stack.nc + 1)) if want else None; AR = np.zeros((hi - lo, s.stack.nc + 1), np.int32) if want else None
            for a in range(lo, hi, 100000):
                b = min(hi, a + 100000); ids = np.ascontiguousarray(s.ids[a - s.K:b - s.K]); y = np.ascontiguousarray(s.x[a:b].astype(np.int32)); tv = s.stack.tv(s.g[a:b], s); lp, ar = s.stack.set(s.L2, tv, b - a, want)
                s.L2.lay_steps(b - a, s.K1, ids.ctypes.data_as(PL), y.ctypes.data_as(PI), s.C.ctypes.data_as(PF), s.N.ctypes.data_as(PF), s.U.ctypes.data_as(PF), s.T.ctypes.data_as(PD), A.ETA, A.E0, int(learn),
                                 None, s.half.ctypes.data_as(PD), s.last.ctypes.data_as(PD), s.clock, None, None)
                if learn: s.clock += b - a
                if want: LP[a - lo:b - lo] = lp; AR[a - lo:b - lo] = ar
            return (LP, AR) if want else None
        def sc(s, lo, hi, learn=False):
            LP, AR = s.go(lo, hi, learn, True); return dict(nats=(-LP.mean(0)).tolist(), right=(100 * (AR == s.x[lo:hi, None]).mean(0)).tolist())
    return A, ProseSys

def load(books):
    import score
    xa, cut, _ = score.load(); vals = score.byte_values(); tab = np.full(256, 1, np.int64); tab[vals] = np.arange(65); parts = [("A", xa)]
    for folder, b in BOOKS[books]:
        raw = np.frombuffer(open(os.path.join(ROOT, folder, "data", b + ".txt"), "rb").read(), np.uint8); raw = raw[raw != 13]; parts.append((b, tab[raw]))
    x = np.concatenate([p[1] for p in parts]); segs = {}; off = 0
    for name, p in parts:
        c = cut if name == "A" else int(0.9 * len(p)); segs[name] = (off, off + c, off + len(p)); off += len(p)
    return x, segs

def run_life(kw, cells_of=None):
    """The stream of the test text, then the books in turn; each last tenth right after its stream, nothing changing; then the test text's last tenth again (what is kept)."""
    A, ProseSys = prose_sys(); x, segs = load(kw.get("books", "worn")); t0 = time.process_time(); w = tuple(v for v in kw.get("without", ",".join(T2)).split(",") if v)
    cells_of = cells_of or (lambda S, D, K1: arms(kw.get("arms", "second"), S, D, K1)); s = ProseSys(x, w); s.start(cells_of); out = dict(task="life", args=kw, kinds=int(s.K1), scores={})
    for name, (a, c, b) in segs.items(): s.go(max(a, 6), c, True); out["scores"][name] = s.sc(c, b)
    a, c, b = segs["A"]; out["scores"]["A_at_the_end"] = s.sc(c, b); pr = s.stack.parts()
    out.update(names=["first"] + [c_.get("name", f"cell{i}") for i, c_ in enumerate(s.cells)], parts=[int(s.T.size)] + [p[0] for p in pr], parts_used=[int((s.T != 0.25).sum())] + [p[1] for p in pr], cells=int((s.N > 0).sum()), cpu=round(time.process_time() - t0, 1)); return out, s

# ---------------------------------------------------------------- checks
def check():
    import merge as G, all as A, score
    ok = True; rng = np.random.default_rng(5); L = kern_m(); L1 = G.kern(); p32, pf, pd = G.p32, G.pf, G.pd
    rich = lambda D: [dict(inp=-1, fams=[("byte", d) for d in range(1, D + 1)]), dict(inp=-1, fams=[("pair", 1, 2)]), dict(inp=-1, fams=[("sure", 0)]), dict(inp=-1, fams=[("own",)]), dict(inp=-1, fams=[("slev", 3)]),
                      dict(inp=0, fams=[("pair", 1, 2), ("own",), ("sure", 1)]), dict(mix=(1, 4)), dict(inp=-1, fams=[("byte", 1, dict(level=0)), ("byte", 2, dict(kind=0)), ("byte", 3, dict(kind=0, level=0)), ("const",)]),
                      dict(inp=5, fams=[("bytes", (2, 5)), ("slev", 1, dict(level=0))])]
    for S, shrink, D in ((14, True, 9), (65, False, 6)):
        K1, n, nc = 9, 2500, 300; ids = rng.integers(-1, nc, (n, K1)).astype(np.int32); sil = rng.random((n, K1)) < 0.1; ids[sil & (ids >= 0)] = -2 - ids[sil & (ids >= 0)]; y = rng.integers(0, S, n).astype(np.int32); g = rng.integers(0, S, (n, DG)).astype(np.int32)
        fresh = lambda: (np.zeros((nc, S), np.float32), np.zeros(nc, np.float32), np.zeros(nc, np.float32), np.full((K1, NL), 0.1), np.zeros((K1, NL)))
        C, N, U, T, Tu = fresh(); st = Stack(K1, S, rich(D), 0.05, shrink, nc); tv = st.tv(g); lp, ar = st.set(L, tv, n, True)
        L.lay_steps(n, K1, S, p32(ids), p32(y), pf(C), pf(N), pf(U), pd(T), pd(Tu), 0.05, 0.3, 1, int(shrink), None)
        Cb, Nb, Ub, Tb, Tub = fresh(); sb = Stack(K1, S, rich(D), 0.05, shrink, nc); LP, AR = ref_steps(ids, y, tv, Cb, Nb, Ub, Tb, Tub, sb, 0.05, 0.3, shrink)
        d = max([np.abs(lp - LP).max(), np.abs(T - Tb).max(), np.abs(C - Cb).max()] + [np.abs(a - b).max() for a, b in zip(st.G + st.Gu + st.W, sb.G + sb.Gu + sb.W)])
        lq, aq = st.set(L, tv, n, True); L.lay_steps(n, K1, S, p32(ids), p32(y), pf(C), pf(N), pf(U), pd(T), pd(Tu), 0.05, 0.3, 0, int(shrink), None)                  # answering only: nothing changes
        LQ, AQ = ref_steps(ids, y, tv, Cb, Nb, Ub, Tb, Tub, sb, 0.05, 0.3, shrink, learn=False); d = max([d, np.abs(lq - LQ).max()] + [np.abs(a - b).max() for a, b in zip(st.G + st.W, sb.G + sb.W)])
        Cc, Nc, Uc, Tc, Tuc = fresh(); Pm = np.zeros((n, S)); L1.mix_steps(n, K1, S, p32(ids), p32(y), pf(Cc), pf(Nc), pf(Uc), pd(Tc), pd(Tuc), 0.05, 0.3, 1, int(shrink), pd(Pm))
        d1 = np.abs(lp[:, 0] - np.log(Pm[np.arange(n), y])).max(); dx = max(np.abs(T - Tc).max(), np.abs(C - Cc).max()); same = (ar == AR).mean(); diff = np.abs(lp[:, 1:] - lp[:, :1]).max(0).min()
        good = d < 1e-9 and d1 < 1e-12 and dx == 0 and same > 0.999 and diff > 1e-3; ok &= good
        print(f"nine further cells on nano/merge's kernel, S = {S}, shrinking step = {shrink}, {n} bytes: compiled against numpy, largest difference {d:.1e}; answers alike {100 * same:.2f}% (ties apart); the first cell against nano/merge's own kernel: trusts and counts {dx:.1e}, log-probabilities {d1:.1e} (numpy's logarithm against C's); every further cell differs from the first by at least {diff:.3f}", "PASSED" if good else "FAILED", flush=True)
    # the prose kernel: T2 on a prefix of the test text; the first cell against all.py's own kernel, the stack against numpy
    A_, ProseSys = prose_sys(); x, cut, _ = score.load(); n = 4000; x = x[:n]; s = ProseSys(x, T2); s.start(lambda S, D, K1: rich(D)); LPc, ARc = s.go(6, n, True, True)
    r = A.SysAll(x, T2); R1 = r.go(6, n, True, True); d1 = np.abs(np.log(R1[np.arange(n - 6), x[6:n]]) - LPc[:, 0]).max(); dx = max(np.abs(r.T - s.T).max(), np.abs(r.C - s.C).max())
    ids = s.ids[:n - 6].astype(np.int32); C = np.zeros_like(s.C); N = np.zeros_like(s.N); U = np.zeros_like(s.U); T = np.full((s.K1 + 1, NL), 0.25); Tu = np.zeros_like(T); sb = Stack(s.K1, 65, rich(6), A.ETA, 0, len(s.N))
    LP, AR = ref_steps(ids, x[6:n].astype(np.int32), sb.tv(s.g[6:n]), C, N, U, T, Tu, sb, A.ETA, A.E0, False)
    d = max([np.abs(LP - LPc).max(), np.abs(T - s.T).max()] + [np.abs(a - b).max() for a, b in zip(s.stack.G + s.stack.W, sb.G + sb.W)]); good = d < 1e-9 and d1 < 1e-12 and dx == 0 and (AR == ARc).mean() > 0.999; ok &= good
    print(f"nine further cells on all.py's kernel (T2), {n - 6} bytes of the test text: compiled against numpy, largest difference {d:.1e}; the first cell against all.py's own kernel: trusts and counts {dx:.1e}, log-probabilities {d1:.1e}", "PASSED" if good else "FAILED", flush=True)
    return ok

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=") for a in sys.argv[2:]); out, _ = dict(sums=run_sums, life=run_life)[sys.argv[1]](kw)
    tag = sys.argv[1] + "_" + "_".join(f"{k}{v.replace(',', '+')}" for k, v in sorted(kw.items())); os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", tag + ".json"), "w"))
    if sys.argv[1] == "sums": print(tag, "| after a read line, last / mean of last three; empty past, last:", {k: (round(v["last_after"], 1), round(v["last3_after"], 1), round(v["last_empty"], 1)) for k, v in out["cells"].items()}, "| cpu", out["cpu"], flush=True)
    else: print(tag, "|", {k: dict(zip(out["names"], [round(q, 4) for q in v["nats"]])) for k, v in out["scores"].items()}, "| cpu", out["cpu"], flush=True)
