#!/usr/bin/env python3
"""nano/lives, PLAN_3.md: remembered trusts. After every N bytes read with learning the second answering cell's trusts and parts are copied and kept (a bookmark).
Every bookmark answers every byte from the same messages; the cell holds a share for the living set and for each bookmark; its answer is the shares' mixture.
    python3 books.py check
    python3 books.py life order=shakespeare,book,py,c,table,xml sys=ZA N=65536 tag=..      (order=seed:73129 draws the order; one name = a single)
The kernel is nano/lives/lives.py's with the additions below (each a replacement asserted to apply once); the living set is not touched by them."""
import ctypes, hashlib, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lives as LV
SU = LV.SU; NL = LV.NL; LMAX = LV.LMAX; PD = LV.PD; PI = LV.PI; rep = LV.rep

BLOCK = r'''
        if (s->bcap > 0 && s->gate) {                    /* PLAN_3: every bookmark answers; the shares' mixture; the shares move */
            const int nb = s->nb, B1 = nb + 1, bc = s->bcap; const double sh = s->share; double mix = 0; int top = 0; double topv = -1;
            s->pb2[0] = p2[y];
            if (nb > 0) {
                static unsigned char mark[SMAX]; static int pos[SMAX]; int tl[SMAX], nt = 0;
                for (int k = 0; k < K; k++) if (lev[k] >= 0) for (int i = 0; i < nv[k]; i++) { const int v = vv[k][i]; if (!mark[v]) { mark[v] = 1; pos[v] = nt; tl[nt++] = v; } }
                if (!mark[y]) { mark[y] = 1; pos[y] = nt; tl[nt++] = y; }
                for (int i = 0; i < nt; i++) { double *r = s->ab + (int64_t)i * bc; for (int b = 0; b < nb; b++) r[b] = 0; }
                for (int k = 0; k < K; k++) if (lev[k] >= 0) {
                    const double *bt = s->BT + (int64_t)(k * NL + lev[k]) * bc; for (int b = 0; b < nb; b++) s->tb[b] = bt[b];
                    for (int j = 0; j < s->J; j++) { const double *bg = s->BG + gix[k][j] * bc; for (int b = 0; b < nb; b++) s->tb[b] += bg[b]; }
                    for (int i = 0; i < nv[k]; i++) { double *r = s->ab + (int64_t)pos[vv[k][i]] * bc; const double d = dv[k][i]; for (int b = 0; b < nb; b++) r[b] += s->tb[b] * d; }
                }
                for (int b = 0; b < nb; b++) s->mxv[b] = nt < S ? 0.0 : -1e300;      /* a value no sender names has the branch sum zero */
                for (int i = 0; i < nt; i++) { const double *r = s->ab + (int64_t)i * bc; for (int b = 0; b < nb; b++) if (r[b] > s->mxv[b]) s->mxv[b] = r[b]; }
                for (int b = 0; b < nb; b++) s->zv[b] = nt < S ? (S - nt) * exp(-s->mxv[b]) : 0.0;
                for (int i = 0; i < nt; i++) { const double *r = s->ab + (int64_t)i * bc; for (int b = 0; b < nb; b++) s->zv[b] += exp(r[b] - s->mxv[b]); }
                { const double *r = s->ab + (int64_t)pos[y] * bc; for (int b = 0; b < nb; b++) s->pb2[b + 1] = exp(r[b] - s->mxv[b]) / s->zv[b]; }
                for (int i = 0; i < nt; i++) mark[tl[i]] = 0;
            }
            for (int b = 0; b < B1; b++) { const double v_ = (1.0 - sh) * s->w2[b] + sh / B1; if (v_ > topv) { topv = v_; top = b; } mix += v_ * s->pb2[b]; s->w2[b] = v_ * s->pb2[b]; }
            for (int b = 0; b < B1; b++) s->w2[b] /= mix;
            if (s->mo2) s->mo2[t - lo] = -log(mix);
            if (s->to2) s->to2[t - lo] = top;
        }
'''
TAKE = r'''
            if (s->bcap > 0) { s->lnow++; if (s->lnow % s->bper == 0 && s->nb < s->bcap) { const int b = s->nb, bc = s->bcap; for (int64_t i = 0; i < (int64_t)K * NL; i++) s->BT[i * bc + b] = s->T[i]; for (int64_t i = 0; i < s->gsize; i++) s->BG[i * bc + b] = s->G[i]; s->w2[b + 1] = 0; s->nb++; } }      /* PLAN_3: a bookmark */
'''
FUNCS = r'''
void sy_books(Sy *s, int64_t bper, int bcap, double share) {
    s->bper = bper; s->bcap = bcap; s->share = share; s->nb = 0; s->lnow = 0;
    s->BT = calloc((size_t)s->K * NL * bcap, sizeof(double)); s->BG = calloc((size_t)s->gsize * bcap, sizeof(double)); s->w2 = calloc(bcap + 1, sizeof(double)); s->pb2 = calloc(bcap + 1, sizeof(double)); s->w2[0] = 1.0;
    s->ab = calloc((size_t)SMAX * bcap, sizeof(double)); s->tb = calloc(bcap, sizeof(double)); s->mxv = calloc(bcap, sizeof(double)); s->zv = calloc(bcap, sizeof(double));
}
void sy_mixout(Sy *s, double *mo2, int32_t *to2) { s->mo2 = mo2; s->to2 = to2; }
int sy_nb(Sy *s) { return s->nb; }
double *sy_w2(Sy *s) { return s->w2; }
'''
SRC = LV.SRC
SRC = rep(SRC, "uint32_t *ptime; double gmin;", "uint32_t *ptime; double gmin; int nb, bcap; int64_t bper, lnow; double share; double *BT, *BG, *w2, *pb2, *ab, *tb, *mxv, *zv, *mo2; int32_t *to2;")
SRC = rep(SRC, "free(s->head); free(s->prev); free(s->ptime); free(s);", "free(s->head); free(s->prev); free(s->ptime); free(s->BT); free(s->BG); free(s->w2); free(s->pb2); free(s->ab); free(s->tb); free(s->mxv); free(s->zv); free(s);")
SRC = rep(SRC, "void sy_gmin(Sy *s, double g) { s->gmin = g; }", "void sy_gmin(Sy *s, double g) { s->gmin = g; }" + FUNCS)
SRC = rep(SRC, "        if (learn) {\n            int nact = 0;", BLOCK.strip("\n") + "\n        if (learn) {\n            int nact = 0;")
SRC = rep(SRC, "/* the place t: the run that ended before it, and the byte x[t] just read */\n", "/* the place t: the run that ended before it, and the byte x[t] just read */\n" + TAKE.strip("\n") + "\n")

def kern():
    h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_books_{h}.so"); src = os.path.join(HERE, "kern_books.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd, cl, vp = ctypes.c_int, ctypes.c_double, ctypes.c_int64, ctypes.c_void_p
    L.sy_new.argtypes = [ci, ci, PI, PI, PI, PD, cd, cd, ci, ci, ci, cd, ci, cl, ci, cd]; L.sy_new.restype = vp; L.sy_free.argtypes = [vp]; L.sy_free.restype = None
    L.sy_run.argtypes = [vp, PI, cl, cl, ci, PD, PD, PD, PD]; L.sy_run.restype = None
    for f in ("sy_cells", "sy_counts", "sy_once", "sy_gsize", "sy_now"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = cl
    for f in ("sy_T", "sy_Tu", "sy_Ta", "sy_G", "sy_Gu", "sy_Ga", "sy_w2"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = PD
    L.sy_gmin.argtypes = [vp, cd]; L.sy_gmin.restype = None; L.sy_books.argtypes = [vp, cl, ci, cd]; L.sy_books.restype = None; L.sy_mixout.argtypes = [vp, PD, PI]; L.sy_mixout.restype = None; L.sy_nb.argtypes = [vp]; L.sy_nb.restype = ci; return L
p32 = LV.p32; pd = LV.pd

class SysB:
    def __init__(s, kinds, x, S=256, eta=0.00005, e=0.02, rule=0, gate=0, J=6, birth=0.0, M=8, age=0, spow=0.0, guard=1e-20, N=0, cap=0):
        s.S, s.kinds, s.K, s.gate, s.J, s.M, s.N = S, kinds, len(kinds), gate, J, M, N; assert S <= 256 and s.K <= 600 and J <= 8; s.L = kern(); s.x = np.ascontiguousarray(x.astype(np.int32)); assert 0 <= s.x.min() and s.x.max() < S
        kt = np.array([k[0] for k in kinds], np.int32); kn = np.array([k[1] for k in kinds], np.int32); kd = np.zeros((s.K, 8), np.int32); kb = np.array([k[3] for k in kinds], np.float64)
        for i, k in enumerate(kinds): kd[i, :len(k[2])] = k[2]
        s.h = s.L.sy_new(S, s.K, p32(kt), p32(kn), p32(np.ascontiguousarray(kd)), pd(kb), eta, e, rule, gate, J, birth, M if any(k[0] == 2 for k in kinds) else 0, len(s.x), age, spow); s.L.sy_gmin(s.h, guard)
        if N > 0: s.L.sy_books(s.h, N, cap or (len(s.x) // N + 2), 1.0 / N)
    def run(s, lo, hi, learn, want_p=False):
        """Losses of the first cell, of the second cell's living set (the plain system), of the second cell's mixture; and the set with the largest share before each byte (0: the living set; b + 1: bookmark b)."""
        n = hi - lo; nl1 = np.zeros(n); nl2 = np.zeros(n); mix = np.zeros(n); top = np.zeros(n, np.int32); P1 = np.zeros((n, s.S)) if want_p else None; P2 = np.zeros((n, s.S)) if want_p else None
        if s.N > 0: s.L.sy_mixout(s.h, pd(mix), p32(top))
        s.L.sy_run(s.h, p32(s.x), lo, hi, int(learn), pd(nl1), pd(nl2), pd(P1), pd(P2)); s.L.sy_mixout(s.h, None, None)
        if s.N == 0: mix = nl2.copy()
        return (nl1, nl2, mix, top, P1, P2) if want_p else (nl1, nl2, mix, top)
    def arr(s, name): return np.ctypeslib.as_array(getattr(s.L, "sy_" + name)(s.h), (s.K, NL)) if name[0] == "T" else np.ctypeslib.as_array(getattr(s.L, "sy_" + name)(s.h), (s.L.sy_gsize(s.h),))
    @property
    def nb(s): return int(s.L.sy_nb(s.h)) if s.N > 0 else 0
    def shares(s): return np.ctypeslib.as_array(s.L.sy_w2(s.h), (s.nb + 1,)).copy() if s.N > 0 else np.ones(1)
    def sizes(s): return dict(cells=int(s.L.sy_cells(s.h)), counts_above_zero=int(s.L.sy_counts(s.h)), kinds=s.K, bookmarks=s.nb, numbers_in_bookmarks=s.nb * (s.K * NL + int(s.L.sy_gsize(s.h))), bytes_read=int(s.L.sy_now(s.h)))

def make(name, x, N=0, S=256, cap=0):
    a = dict(LV.SYSTEMS[name]); kinds = SU.kinds_of(a.pop("kinds"), S, a.pop("recall")); return SysB(kinds, x, S=S, N=N, cap=cap, **a)

# ---------------------------------------------------------------- the reference in plain Python: lives.py's, with the bookmarks and the shares written out
def ref_run(S, kinds, x, lo, hi, state=None, learn=True, N=0, eta=0.00005, e=0.02, rule=0, gate=0, J=6, birth=0.0, M=8, age=0, spow=0.0, guard=1e-20, caps=None, wake=False):
    K = len(kinds); GL = NL if gate == 2 else 1
    if state is None: state = dict(cells={}, T=np.array([[k[3]] * NL for k in kinds], np.float64), Tu=np.zeros((K, NL)), Ta=np.zeros((K, NL)), G=np.zeros((J, K, GL, S)), Gu=np.zeros((J, K, GL, S)), Ga=np.zeros((J, K, GL, S)), places={}, ptime={}, lasts={}, times={}, last={}, now=0, books=[], w2=[1.0], lnow=0, returns=0)
    cells, T, Tu, Ta, G, Gu, Ga, places = (state[k] for k in ("cells", "T", "Tu", "Ta", "G", "Gu", "Ga", "places")); P2 = np.zeros((hi - lo, S)); MIX = np.zeros(hi - lo); xb = lambda q: int(x[q]) if q >= 0 else 0
    step = lambda used, acc: eta if rule == 0 else eta / np.sqrt(1.0 + used) if rule == 1 else eta * sc[0] / np.sqrt(acc); sc = [1.0]
    for t in range(lo, hi):
        tau = state["now"] + (t - lo); y = int(x[t]); a = np.zeros(S); act = []; keys = []; rq = places.get(tuple(int(v) for v in x[t - M:t]), []) if t >= M else []
        for k, (typ, nd, ds, _, _) in enumerate(kinds):
            if typ == 2:
                keys.append(None)
                if nd < len(rq):
                    q = rq[-1 - nd]; L = M
                    while L < LMAX and q - L - 1 >= 0 and x[q - L - 1] == x[t - L - 1]: L += 1
                    c = np.zeros(S); c[int(x[q])] = 1; sv = np.log(c + e) - np.log(1 + S * e); l = SU.agelevel_recall_np(L, M, tau - state["ptime"][q]) if age else SU.reclevel_np(L, M); act.append((k, l, sv, np.log(c + e) - np.log(e))); a += T[k, l] * sv
                continue
            if typ == 3:
                keys.append(None); c = cells.get(keys[nd])
                if c is not None and c.sum() >= 1:
                    c1 = np.zeros(S); c1[state["lasts"][keys[nd]]] = 1; sv = np.log(c1 + e) - np.log(1 + S * e); l = SU.agelevel_last_np(int((c > 0).sum()), tau - state["times"][keys[nd]]) if age else SU.level_np(c.sum(), int((c > 0).sum())); act.append((k, l, sv, np.log(c1 + e) - np.log(e))); a += T[k, l] * sv
                continue
            if typ == 0: key = (k,) + tuple(xb(t - d) for d in ds)
            else:
                if nd not in state["last"]: keys.append(None); continue
                key = (k, min(63, tau - 1 - state["last"][nd]), xb(t - 1))
            keys.append(key); c = cells.get(key)
            if c is not None and c.sum() >= 1:
                n = c.sum(); l = SU.level_np(n, int((c > 0).sum())); sv = np.log(c + e) - np.log(n + S * e); act.append((k, l, sv, np.log(c + e) - np.log(e))); a += T[k, l] * sv
        p = np.exp(a - a.max()); p /= p.sum(); a2 = a.copy()
        for k, l, sv, dv in act: a2 += sum(G[j, k, l if GL > 1 else 0, xb(t - 1 - j)] for j in range(J)) * sv
        p2 = np.exp(a2 - a2.max()); p2 /= p2.sum(); P2[t - lo] = p2
        if N > 0:                                                       # every set's answer to the byte that came; the mixture; the shares
            pb = [p2[y]]
            for BT, BG in state["books"]:
                ab = np.zeros(S)
                for k, l, sv, dv in act: ab += (BT[k, l] + sum(BG[j, k, l if GL > 1 else 0, xb(t - 1 - j)] for j in range(J))) * dv
                q = np.exp(ab - ab.max()); pb.append(q[y] / q.sum())
            B1 = len(pb); v = [(1.0 - 1.0 / N) * w + (1.0 / N) / B1 for w in state["w2"]]; mixp = sum(vi * pi for vi, pi in zip(v, pb)); MIX[t - lo] = mixp; state["w2"] = [vi * pi / mixp for vi, pi in zip(v, pb)]
            top = max(range(B1), key=lambda b: (v[b], -b)); jump = top if (learn and wake and top > 0 and v[top] > 0.5) else 0
        if learn:
            sc[0] = (12.0 / len(act)) ** spow if spow > 0 and act else 1.0
            for k, l, sv, dv in act:
                g = dv[y] - p @ dv; Ta[k, l] += g * g
                if Ta[k, l] > guard or rule != 2: T[k, l] += step(Tu[k, l], Ta[k, l]) * g
                Tu[k, l] += 1; g2 = dv[y] - p2 @ dv
                for j in range(J):
                    i = (j, k, l if GL > 1 else 0, xb(t - 1 - j)); Ga[i] += g2 * g2
                    if Ga[i] > guard or rule != 2: G[i] += step(Gu[i], Ga[i]) * g2
                    Gu[i] += 1
            for k, key in enumerate(keys):
                if key is None: continue
                if key not in cells:
                    if birth > 0 and p[y] >= birth: continue
                    cells[key] = np.zeros(S)
                cells[key][y] += 1; state["lasts"][key] = y; state["times"][key] = tau
                if caps is not None and caps[k] > 0 and cells[key].sum() >= caps[k]: cells[key] = np.floor(cells[key] / 2)      # nano/lives PLAN_5: a cell of a bounded kind halves its counts at the bound
            if any(k[0] == 2 for k in kinds) and t >= M: places.setdefault(tuple(int(v) for v in x[t - M:t]), []).append(t); state["ptime"][t] = tau
            if N > 0:
                state["lnow"] += 1
                if state["lnow"] % N == 0: state["books"].append((T.copy(), G.copy())); state["w2"].append(0.0)
                if jump:                                                     # PLAN_4: the living set is kept as a bookmark and goes back to the bookmark that held more than half of the belief
                    BT, BG = state["books"][jump - 1]; state["books"].append((T.copy(), G.copy())); T[:] = BT; G[:] = BG; state["w2"].append(state["w2"][0]); state["w2"][0] = state["w2"][jump]; state["w2"][jump] = 0.0; state["returns"] += 1
        state["last"][y] = tau
    state["now"] += hi - lo; return P2, MIX, state

m = LV.m
def score(s, lo, hi, learn, marks=None):
    t0 = time.process_time(); n1, n2, mx, top = s.run(lo, hi, learn); r = dict(cell1=m(n1), cell2=m(n2), mix=m(mx), first={str(k): [m(n2[:k]), m(mx[:k])] for k in (1000, 4000, 16000, 64000) if k <= hi - lo}, share_on_living=float((top == 0).mean()), cpu=round(time.process_time() - t0, 1))
    if marks is not None:                                              # bytes of the test at which the largest share was on the living set (index 0) or on a bookmark taken while body i was read (index i + 1)
        r["largest_share_living_then_by_body"] = np.bincount(np.where(top > 0, np.searchsorted(marks, top, side="left") + 1, 0), minlength=len(marks) + 1).tolist()
    if learn: r["sizes_after"] = s.sizes()
    return r

def run_life(kw):
    order = kw["order"]; name = kw.get("sys", "ZA"); tag = kw["tag"]; N = int(kw.get("N", 65536))
    if order.startswith("seed:"): order = [LV.NAMES[i] for i in np.random.default_rng(int(order[5:])).permutation(len(LV.NAMES))]
    else: order = order.split(",")
    tests = list(LV.NAMES) if len(order) > 1 else list(order); x, body, test = LV.layout(order, tests); s = make(name, x, N); tmp = os.path.join(HERE, "tmp", tag); os.makedirs(tmp, exist_ok=True); t0 = time.time(); stages = []; child = None; marks = []
    def stage_scores(i, mk):
        def fn():
            for n in tests:
                for learn in (0, 1): LV.wait(LV.forked(lambda: score(s, test[n][0], test[n][1], bool(learn), mk), os.path.join(tmp, f"s{i}_{n}_{learn}.json")))
            return dict(done=True)
        return fn
    for i, n in enumerate(order):
        lo, hi = body[n]; lo += 6 if i == 0 else 0; c0 = time.process_time(); n1, n2, mx, top = s.run(lo, hi, True); k = (hi - lo) // 10; marks.append(s.nb)      # marks[i]: bookmarks taken up to the end of body i
        stages.append(dict(stream=n, reading=dict(cell1=m(n1), cell2=m(n2), mix=m(mx), first_tenth=[m(n2[:k]), m(mx[:k])], last_tenth=[m(n2[-k:]), m(mx[-k:])]), share_on_living=float((top == 0).mean()), sizes=s.sizes(), cpu_reading=round(time.process_time() - c0, 1)))
        if child is not None: LV.wait(child)
        child = LV.forked(stage_scores(i, np.array(marks)), os.path.join(tmp, f"s{i}_done.json")); print(tag, "read", n, {k_: round(v, 4) for k_, v in stages[-1]["reading"].items() if not k_.endswith("tenth")}, "bookmarks", s.nb, "cells", stages[-1]["sizes"]["cells"], "cpu", stages[-1]["cpu_reading"], flush=True)
    if child is not None: LV.wait(child)
    for i, st in enumerate(stages): st["tests"] = {n: dict(nothing_changing=json.load(open(os.path.join(tmp, f"s{i}_{n}_0.json"))), going_on=json.load(open(os.path.join(tmp, f"s{i}_{n}_1.json")))) for n in tests}
    out = dict(task="books", tag=tag, sys=name, N=N, order=order, stages=stages, bookmarks_by_body=marks, length=int(len(x)), wall=round(time.time() - t0, 1)); json.dump(out, open(os.path.join(HERE, "res", f"books__{tag}.json"), "w")); return out

def check():
    ok = True; rng = np.random.default_rng(7)
    cases = [("shakespeare", "runs+last", dict(gate=1, rule=2, eta=0.03, age=1), 3, 300), ("xml", "runs+last", dict(gate=1, rule=2, eta=0.03, age=1), 3, 250), ("table", "runs", dict(gate=1), 1, 400), ("py", "runs+last", dict(gate=2, rule=1, eta=0.1, age=1), 3, 300),
             ("c", "runs+last+since+pool", dict(gate=1, rule=2, eta=0.03, age=1, spow=0.5), 3, 300)]
    for name, spec, kw, recall, N in cases:
        n = 2500 if "since" not in spec else 1500; x = LV.raw(name)[0][:n]; kinds = SU.kinds_of(spec, 256, recall); cutn = n - 500
        s = SysB(kinds, x, N=N, **kw); a = s.run(6, cutn, True, True); f = s.run(cutn, n, False, True); g = s.run(cutn, n, True, True)
        R2, MA, st = ref_run(256, kinds, x, 6, cutn, None, True, N, **kw); Q2, MF, _ = ref_run(256, kinds, x, cutn, n, st, False, N, **kw); W2, MG, _ = ref_run(256, kinds, x, cutn, n, st, True, N, **kw)
        d = max(np.abs(np.exp(-a[2]) - MA).max(), np.abs(np.exp(-f[2]) - MF).max(), np.abs(np.exp(-g[2]) - MG).max(), np.abs(a[5] - R2).max(), np.abs(f[5] - Q2).max(), np.abs(g[5] - W2).max(), np.abs(s.shares() - np.array(st["w2"])).max())
        p = LV.Sys(kinds, x, **kw); pa = p.run(6, cutn, True); pf = p.run(cutn, n, False); pg = p.run(cutn, n, True); dl = max(np.abs(u[i] - v[i]).max() for u, v in ((a, pa), (f, pf), (g, pg)) for i in (0, 1))
        x2 = x.copy(); n0 = cutn - 300; x2[n0:] = (x2[n0:] + 1 + rng.integers(0, 255, n - n0)) % 256; sa = SysB(kinds, x, N=N, **kw); sb = SysB(kinds, x2, N=N, **kw); u = sa.run(6, n0 + 1, True); v = sb.run(6, n0 + 1, True); blind = bool(np.array_equal(u[2][:-1], v[2][:-1]) and np.array_equal(u[3], v[3]))
        tol = (1e-4 if "since" in spec else 1e-6) if kw.get("rule") == 2 else 1e-8; good = d < tol and dl < tol and blind and s.nb == len(st["books"]) and s.nb >= 4; ok &= good
        print(f"{name} ({n} raw bytes), kinds {spec}, options {kw}, a bookmark every {N} bytes ({s.nb} taken): compiled against the reference, largest difference {d:.1e} (the mixture's share of the byte that came while reading, with nothing changing and going on; the living set's answers; the shares at the end); the living set's losses differ from the plain kernel's by at most {dl:.1e} (the two kernels are compiled separately and round differently in the last digits); no mixture answer before a place, and no largest share up to it, moves when the byte there and all later ones are replaced: {blind}", "PASSED" if good else "FAILED", flush=True)
    x = LV.raw("py")[0][:40000]; s = make("ZA", x, N=10 ** 9); a = s.run(6, 30000, True); f = s.run(30000, 40000, False); same = bool(np.abs(a[2] - a[1]).max() < 1e-12 and np.abs(f[2] - f[1]).max() < 1e-12 and s.nb == 0); ok &= same
    print(f"with no bookmark taken the mixture is the living set's answer: {same}", "PASSED" if same else "FAILED", flush=True)
    s = make("ZA", x, N=4096); s.run(6, 30000, True); p = os.path.join(HERE, "tmp"); os.makedirs(p, exist_ok=True); LV.wait(LV.forked(lambda: score(s, 30000, 35000, True), os.path.join(p, "check_fork_b.json"))); a = s.run(30000, 35000, True)
    s2 = make("ZA", x, N=4096); s2.run(6, 30000, True); b = s2.run(30000, 35000, True); fk = json.load(open(os.path.join(p, "check_fork_b.json"))); same = bool(np.array_equal(a[2], b[2]) and abs(fk["mix"] - float(a[2].mean())) < 1e-12); ok &= same
    print(f"a going-on measurement made in a forked copy leaves the system as it was, and the copy's figure for the mixture is the same ({fk['mix']:.6f}; the living set {fk['cell2']:.6f}; {s.nb} bookmarks): {same}", "PASSED" if same else "FAILED", flush=True)
    return ok

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=", 1) for a in sys.argv[2:]); out = run_life(kw); r = lambda v: None if v is None else round(v, 4)
    for st in out["stages"]: print(out["tag"], f"after {st['stream']}:", " | ".join(f"{n} {r(t['going_on']['cell2'])} -> {r(t['going_on']['mix'])}" for n, t in st["tests"].items()), "(going on: the plain answer -> the mixture)", flush=True)
    e = out["stages"][-1]["tests"]; print(out["tag"], "bookmarks", out["stages"][-1]["sizes"]["bookmarks"], "| mean over the tests at the end, going on: plain", r(float(np.mean([t["going_on"]["cell2"] for t in e.values()]))), "mixture", r(float(np.mean([t["going_on"]["mix"] for t in e.values()]))),
                                   "| nothing changing: plain", r(float(np.mean([t["nothing_changing"]["cell2"] for t in e.values()]))), "mixture", r(float(np.mean([t["nothing_changing"]["mix"] for t in e.values()]))), "| wall", out["wall"], flush=True)
