#!/usr/bin/env python3
"""nano/lives, PLAN_8.md: the set that holds the belief goes on learning. PLAN_3.md's bookmarks and shares, and: at each byte, if the set with the largest
share is a bookmark, that bookmark learns from the byte too (its own answers, its own mismatch, sums of its own); the living set learns as before.
    python3 awake.py check
    python3 awake.py life order=.. sys=ZA|M|MC N=65536 [awake=1] tag=..
The kernel is nano/lives/cellmix.py's (bookmarks, bounded kinds, the cell's mixture) with the additions below; with awake off it is that kernel."""
import ctypes, hashlib, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lives as LV, books as BK, mem as MM, cellmix as CM
SU = LV.SU; NL = LV.NL; LMAX = LV.LMAX; PD = LV.PD; PI = LV.PI; rep = LV.rep; p32 = LV.p32; pd = LV.pd; m = LV.m; SIG = CM.SIG

WAKE = r'''
                if (learn && s->awake) {                         /* PLAN_8: the bookmark that holds the largest share learns from the byte, by its own mismatch */
                    int tp = 0; double tv = -1; for (int b = 0; b < B1; b++) { const double v_ = (1.0 - sh) * s->w2[b] + sh / B1; if (v_ > tv) { tv = v_; tp = b; } }
                    if (tp > 0) {
                        const int b = tp - 1; double a1[SMAX], mx1 = nt < S ? 0.0 : -1e300, z1; int nact = 0;
                        for (int i = 0; i < nt; i++) a1[i] = 0;
                        for (int k = 0; k < K; k++) if (lev[k] >= 0) { nact++; const double t1 = s->BT[(int64_t)(k * NL + lev[k]) * bc + b]; for (int i = 0; i < nv[k]; i++) a1[pos[vv[k][i]]] += t1 * dv[k][i]; }
                        for (int i = 0; i < nt; i++) if (a1[i] > mx1) mx1 = a1[i];
                        z1 = nt < S ? (S - nt) * exp(-mx1) : 0.0; for (int i = 0; i < nt; i++) { a1[i] = exp(a1[i] - mx1); z1 += a1[i]; }
                        for (int i = 0; i < nt; i++) a1[i] /= z1;
                        const double scb = s->spow > 0 && nact > 0 ? pow(12.0 / nact, s->spow) : 1.0;
                        for (int k = 0; k < K; k++) if (lev[k] >= 0) {
                            double g = 0, g2 = 0;
                            for (int i = 0; i < nv[k]; i++) { const int v = vv[k][i]; const double d = dv[k][i]; g += ((v == y) - a1[pos[v]]) * d; g2 += ((v == y) - exp(s->ab[(int64_t)pos[v] * bc + b] - s->mxv[b]) / s->zv[b]) * d; }
                            const int64_t i1 = (int64_t)(k * NL + lev[k]) * bc + b; s->BTa[i1] += g * g; if (s->BTa[i1] > s->gmin || s->rule != 2) s->BT[i1] += step_of(s, s->BTu[i1], s->BTa[i1], scb) * g; s->BTu[i1] += 1.0;
                            for (int j = 0; j < s->J; j++) { const int64_t i2 = gix[k][j] * bc + b; s->BGa[i2] += g2 * g2; if (s->BGa[i2] > s->gmin || s->rule != 2) s->BG[i2] += step_of(s, s->BGu[i2], s->BGa[i2], scb) * g2; s->BGu[i2] += 1.0; }
                        }
                        s->awoke++;
                    }
                }
'''
SRC = CM.SRC
SRC = rep(SRC, "double *BT, *BG, *w2,", "double *BTa, *BGa, *BTu, *BGu; int awake; int64_t awoke; double *BT, *BG, *w2,")
SRC = rep(SRC, "int sy_nb(Sy *s) { return s->nb; }", "int sy_nb(Sy *s) { return s->nb; }\nvoid sy_awake(Sy *s, int on) { s->awake = on; if (on) { s->BTa = calloc((size_t)s->K * NL * s->bcap, sizeof(double)); s->BTu = calloc((size_t)s->K * NL * s->bcap, sizeof(double)); s->BGa = calloc((size_t)s->gsize * s->bcap, sizeof(double)); s->BGu = calloc((size_t)s->gsize * s->bcap, sizeof(double)); } }\nint64_t sy_awoke(Sy *s) { return s->awoke; }")
SRC = rep(SRC, "                for (int i = 0; i < nt; i++) mark[tl[i]] = 0;\n", WAKE.strip("\n") + "\n                for (int i = 0; i < nt; i++) mark[tl[i]] = 0;\n")
SRC = rep(SRC, "for (int64_t i = 0; i < s->gsize; i++) s->BG[i * bc + b] = s->G[i]; s->w2[b + 1] = 0; s->nb++; } }", "for (int64_t i = 0; i < s->gsize; i++) s->BG[i * bc + b] = s->G[i]; if (s->awake) { for (int64_t i = 0; i < (int64_t)K * NL; i++) { s->BTa[i * bc + b] = s->Ta[i]; s->BTu[i * bc + b] = s->Tu[i]; } for (int64_t i = 0; i < s->gsize; i++) { s->BGa[i * bc + b] = s->Ga[i]; s->BGu[i * bc + b] = s->Gu[i]; } } s->w2[b + 1] = 0; s->nb++; } }")

def kern():
    h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_awake_{h}.so"); src = os.path.join(HERE, "kern_awake.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd, cl, vp = ctypes.c_int, ctypes.c_double, ctypes.c_int64, ctypes.c_void_p
    L.sy_new.argtypes = [ci, ci, PI, PI, PI, PD, cd, cd, ci, ci, ci, cd, ci, cl, ci, cd]; L.sy_new.restype = vp; L.sy_free.argtypes = [vp]; L.sy_free.restype = None
    L.sy_run.argtypes = [vp, PI, cl, cl, ci, PD, PD, PD, PD]; L.sy_run.restype = None
    for f in ("sy_cells", "sy_counts", "sy_once", "sy_gsize", "sy_now", "sy_awoke"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = cl
    for f in ("sy_T", "sy_Tu", "sy_Ta", "sy_G", "sy_Gu", "sy_Ga", "sy_w2"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = PD
    L.sy_gmin.argtypes = [vp, cd]; L.sy_gmin.restype = None; L.sy_books.argtypes = [vp, cl, ci, cd]; L.sy_books.restype = None; L.sy_mixout.argtypes = [vp, PD, PI]; L.sy_mixout.restype = None; L.sy_nb.argtypes = [vp]; L.sy_nb.restype = ci
    L.sy_caps.argtypes = [vp, PI]; L.sy_caps.restype = None; L.sy_silent.argtypes = [vp, PI]; L.sy_silent.restype = None; L.sy_cells_of.argtypes = [vp, ci]; L.sy_cells_of.restype = cl; L.sy_awake.argtypes = [vp, ci]; L.sy_awake.restype = None; return L

def kinds_of(name, S=256):
    if name == "ZA": kinds, caps, a = MM.kinds_caps("ZA", S); return kinds, caps, [0] * len(kinds), a
    return CM.kinds_of(name, S)

class SysA(CM.SysC):
    def __init__(s, kinds, caps, silent, x, S=256, eta=0.00005, e=0.02, rule=0, gate=0, J=6, birth=0.0, M=8, age=0, spow=0.0, guard=1e-20, N=0, cap=0, awake=0):
        s.S, s.kinds, s.K, s.gate, s.J, s.M, s.N = S, kinds, len(kinds), gate, J, M, N; assert S <= 256 and s.K <= 600 and J <= 8; s.L = kern(); s.x = np.ascontiguousarray(x.astype(np.int32)); assert 0 <= s.x.min() and s.x.max() < S
        kt = np.array([k[0] for k in kinds], np.int32); kn = np.array([k[1] for k in kinds], np.int32); kd = np.zeros((s.K, 8), np.int32); kb = np.array([k[3] for k in kinds], np.float64)
        for i, k in enumerate(kinds): kd[i, :len(k[2])] = k[2]
        s.h = s.L.sy_new(S, s.K, p32(kt), p32(kn), p32(np.ascontiguousarray(kd)), pd(kb), eta, e, rule, gate, J, birth, M if any(k[0] == 2 for k in kinds) else 0, len(s.x), age, spow); s.L.sy_gmin(s.h, guard); s.L.sy_caps(s.h, p32(np.array(caps, np.int32))); s.L.sy_silent(s.h, p32(np.array(silent, np.int32)))
        if N > 0: s.L.sy_books(s.h, N, cap or (len(s.x) // N + 2), 1.0 / N); s.L.sy_awake(s.h, int(awake))
    def sizes(s): z = MM.SysM.sizes(s); z["bytes_at_which_a_bookmark_learned"] = int(s.L.sy_awoke(s.h)); return z

def make(name, x, N=0, awake=0): kinds, caps, silent, a = kinds_of(name); return SysA(kinds, caps, silent, x, N=N, awake=awake, **a)

# ---------------------------------------------------------------- the reference in plain Python: cellmix.py's, with the bookmarks, the shares and the waking written out
def ref_run(S, kinds, caps, silent, x, lo, hi, state=None, learn=True, N=0, awake=False, eta=0.00005, e=0.02, rule=0, gate=0, J=6, birth=0.0, M=8, age=0, spow=0.0, guard=1e-20):
    K = len(kinds); GL = NL if gate == 2 else 1
    if state is None: state = dict(cells={}, cw={}, T=np.array([[k[3]] * NL for k in kinds], np.float64), Tu=np.zeros((K, NL)), Ta=np.zeros((K, NL)), G=np.zeros((J, K, GL, S)), Gu=np.zeros((J, K, GL, S)), Ga=np.zeros((J, K, GL, S)), places={}, ptime={}, lasts={}, times={}, last={}, now=0, books=[], w2=[1.0], lnow=0, awoke=0)
    cells, cw, T, Tu, Ta, G, Gu, Ga, places = (state[k] for k in ("cells", "cw", "T", "Tu", "Ta", "G", "Gu", "Ga", "places")); P2 = np.zeros((hi - lo, S)); MIX = np.zeros(hi - lo); xb = lambda q: int(x[q]) if q >= 0 else 0; zero = np.zeros(S)
    step = lambda used, acc, scl: eta if rule == 0 else eta / np.sqrt(1.0 + used) if rule == 1 else eta * scl / np.sqrt(acc)
    for t in range(lo, hi):
        tau = state["now"] + (t - lo); y = int(x[t]); a = np.zeros(S); act = []; keys = []; mixes = []; rq = places.get(tuple(int(v) for v in x[t - M:t]), []) if t >= M else []
        for k, (typ, nd, ds, _, _) in enumerate(kinds):
            if typ == 2:
                keys.append(None)
                if nd < len(rq):
                    q = rq[-1 - nd]; L = M
                    while L < LMAX and q - L - 1 >= 0 and x[q - L - 1] == x[t - L - 1]: L += 1
                    c = np.zeros(S); c[int(x[q])] = 1; dv = np.log(c + e) - np.log(e); l = SU.agelevel_recall_np(L, M, tau - state["ptime"][q]) if age else SU.reclevel_np(L, M); act.append((k, l, dv)); a += T[k, l] * dv
                continue
            if typ == 3:
                keys.append(None); c = cells.get(keys[nd])
                if c is not None and c.sum() >= 1:
                    c1 = np.zeros(S); c1[state["lasts"][keys[nd]]] = 1; dv = np.log(c1 + e) - np.log(e); l = SU.agelevel_last_np(int((c > 0).sum()), tau - state["times"][keys[nd]]) if age else SU.level_np(c.sum(), int((c > 0).sum())); act.append((k, l, dv)); a += T[k, l] * dv
                continue
            if typ == 6:
                keys.append(None); kl = keys[nd]; cl = cells.get(kl)
                if cl is not None and cl.sum() >= 1:
                    mem = [cl, cells.get(keys[ds[0]], zero), cells.get(keys[ds[1]], zero)]; w = cw[kl]; wn = [float(w[q]) / (mem[q].sum() + S * e) for q in range(3)]
                    o = wn[0] * (mem[0] + e) + wn[1] * (mem[1] + e) + wn[2] * (mem[2] + e); dv = np.where(cl > 0, np.log(o) - np.log((wn[0] + wn[1] + wn[2]) * e), 0.0); l = SU.level_np(cl.sum(), int((cl > 0).sum())); act.append((k, l, dv)); a += T[k, l] * dv; mixes.append((k, kl, ds))
                continue
            key = (k,) + tuple(xb(t - d) for d in ds); keys.append(key); c = cells.get(key)
            if c is not None and c.sum() >= 1 and not silent[k]:
                l = SU.level_np(c.sum(), int((c > 0).sum())); dv = np.log(c + e) - np.log(e); act.append((k, l, dv)); a += T[k, l] * dv
        p = np.exp(a - a.max()); p /= p.sum(); a2 = a.copy(); gsel = lambda Gx, k, l: sum(Gx[j, k, l if GL > 1 else 0, xb(t - 1 - j)] for j in range(J))
        for k, l, dv in act: a2 += gsel(G, k, l) * dv
        p2 = np.exp(a2 - a2.max()); p2 /= p2.sum(); P2[t - lo] = p2
        if N > 0:
            pb = [p2[y]]; full = []
            for bk in state["books"]:
                ab = np.zeros(S)
                for k, l, dv in act: ab += (bk["T"][k, l] + gsel(bk["G"], k, l)) * dv
                q = np.exp(ab - ab.max()); q /= q.sum(); pb.append(q[y]); full.append(q)
            B1 = len(pb); v = [(1.0 - 1.0 / N) * w + (1.0 / N) / B1 for w in state["w2"]]; mixp = sum(vi * pi for vi, pi in zip(v, pb)); MIX[t - lo] = mixp; top = max(range(B1), key=lambda b: (v[b], -b))
            if learn and awake and top > 0:                             # the bookmark that holds the largest share learns by its own mismatch
                bk = state["books"][top - 1]; a1 = np.zeros(S)
                for k, l, dv in act: a1 += bk["T"][k, l] * dv
                p1b = np.exp(a1 - a1.max()); p1b /= p1b.sum(); p2b = full[top - 1]; scb = (12.0 / len(act)) ** spow if spow > 0 and act else 1.0; upd = []
                for k, l, dv in act: upd.append((k, l, dv[y] - p1b @ dv, dv[y] - p2b @ dv))
                for k, l, g, g2 in upd:
                    bk["Ta"][k, l] += g * g
                    if bk["Ta"][k, l] > guard or rule != 2: bk["T"][k, l] += step(bk["Tu"][k, l], bk["Ta"][k, l], scb) * g
                    bk["Tu"][k, l] += 1
                    for j in range(J):
                        i = (j, k, l if GL > 1 else 0, xb(t - 1 - j)); bk["Ga"][i] += g2 * g2
                        if bk["Ga"][i] > guard or rule != 2: bk["G"][i] += step(bk["Gu"][i], bk["Ga"][i], scb) * g2
                        bk["Gu"][i] += 1
                state["awoke"] += 1
            state["w2"] = [vi * pi / mixp for vi, pi in zip(v, pb)]
        if learn:
            sc = (12.0 / len(act)) ** spow if spow > 0 and act else 1.0
            for k, l, dv in act:
                g = dv[y] - p @ dv; Ta[k, l] += g * g
                if Ta[k, l] > guard or rule != 2: T[k, l] += step(Tu[k, l], Ta[k, l], sc) * g
                Tu[k, l] += 1; g2 = dv[y] - p2 @ dv
                for j in range(J):
                    i = (j, k, l if GL > 1 else 0, xb(t - 1 - j)); Ga[i] += g2 * g2
                    if Ga[i] > guard or rule != 2: G[i] += step(Gu[i], Ga[i], sc) * g2
                    Gu[i] += 1
            for k, kl, ds in mixes:
                mem = [cells[kl], cells.get(keys[ds[0]], zero), cells.get(keys[ds[1]], zero)]; w = cw[kl]; o = [((1.0 - SIG) * float(w[q]) + SIG / 3.0) * (mem[q][y] + e) / (mem[q].sum() + S * e) for q in range(3)]; z = o[0] + o[1] + o[2]; cw[kl] = np.array([o[q] / z for q in range(3)], np.float32)
            for k, key in enumerate(keys):
                if key is None: continue
                if key not in cells:
                    if birth > 0 and p[y] >= birth: continue
                    cells[key] = np.zeros(S); cw[key] = np.full(3, 1.0 / 3.0, np.float32)
                cells[key][y] += 1; state["lasts"][key] = y; state["times"][key] = tau
                if caps[k] > 0 and cells[key].sum() >= caps[k]: cells[key] = np.floor(cells[key] / 2)
            if any(k[0] == 2 for k in kinds) and t >= M: places.setdefault(tuple(int(v) for v in x[t - M:t]), []).append(t); state["ptime"][t] = tau
            if N > 0:
                state["lnow"] += 1
                if state["lnow"] % N == 0: state["books"].append(dict(T=T.copy(), G=G.copy(), Ta=Ta.copy(), Ga=Ga.copy(), Tu=Tu.copy(), Gu=Gu.copy())); state["w2"].append(0.0)
        state["last"][y] = tau
    state["now"] += hi - lo; return P2, MIX, state

def score(s, lo, hi, learn):
    t0 = time.process_time(); n1, n2, mx, top = s.run(lo, hi, learn); r = dict(cell1=m(n1), cell2=m(n2), mix=m(mx), first={str(k): [m(n2[:k]), m(mx[:k])] for k in (1000, 4000, 16000, 64000) if k <= hi - lo}, share_on_living=float((top == 0).mean()), cpu=round(time.process_time() - t0, 1))
    if learn: r["sizes_after"] = s.sizes()
    return r

def run_life(kw):
    order = kw["order"]; name = kw.get("sys", "ZA"); tag = kw["tag"]; N = int(kw.get("N", 65536)); awake = int(kw.get("awake", 1))
    if order.startswith("seed:"): order = [LV.NAMES[i] for i in np.random.default_rng(int(order[5:])).permutation(len(LV.NAMES))]
    else: order = order.split(",")
    tests = list(LV.NAMES) if len(order) > 1 else list(order); x, body, test = LV.layout(order, tests); s = make(name, x, N, awake); tmp = os.path.join(HERE, "tmp", tag); os.makedirs(tmp, exist_ok=True); t0 = time.time(); stages = []; child = None
    def stage_scores(i):
        def fn():
            for n in tests:
                for learn in (0, 1): LV.wait(LV.forked(lambda: score(s, test[n][0], test[n][1], bool(learn)), os.path.join(tmp, f"s{i}_{n}_{learn}.json")))
            return dict(done=True)
        return fn
    for i, n in enumerate(order):
        lo, hi = body[n]; lo += 6 if i == 0 else 0; c0 = time.process_time(); n1, n2, mx, top = s.run(lo, hi, True); k = (hi - lo) // 10
        stages.append(dict(stream=n, reading=dict(cell1=m(n1), cell2=m(n2), mix=m(mx), first_tenth=[m(n2[:k]), m(mx[:k])], last_tenth=[m(n2[-k:]), m(mx[-k:])]), share_on_living=float((top == 0).mean()), sizes=s.sizes(), cpu_reading=round(time.process_time() - c0, 1)))
        if child is not None: LV.wait(child)
        child = LV.forked(stage_scores(i), os.path.join(tmp, f"s{i}_done.json")); print(tag, "read", n, {k_: round(v, 4) for k_, v in stages[-1]["reading"].items() if not k_.endswith("tenth")}, "cells", stages[-1]["sizes"]["cells"], "bookmarks", s.nb, "bytes at which a bookmark learned", stages[-1]["sizes"]["bytes_at_which_a_bookmark_learned"], "cpu", stages[-1]["cpu_reading"], flush=True)
    if child is not None: LV.wait(child)
    for i, st in enumerate(stages): st["tests"] = {n: dict(nothing_changing=json.load(open(os.path.join(tmp, f"s{i}_{n}_0.json"))), going_on=json.load(open(os.path.join(tmp, f"s{i}_{n}_1.json")))) for n in tests}
    out = dict(task="awake", tag=tag, sys=name, N=N, awake=awake, order=order, stages=stages, length=int(len(x)), wall=round(time.time() - t0, 1)); json.dump(out, open(os.path.join(HERE, "res", f"awake__{tag}.json"), "w")); return out

def check():
    ok = True; rng = np.random.default_rng(23)
    pieces = [LV.raw(n)[0][a:a + L] for n, a, L in (("shakespeare", 0, 1500), ("table", 5000, 1200), ("shakespeare", 9000, 1200), ("table", 90000, 1000), ("xml", 0, 1200), ("shakespeare", 30000, 1200), ("table", 200000, 1000))]; x = np.ascontiguousarray(np.concatenate(pieces).astype(np.int32)); n = len(x)
    for sysn, N, extra in (("ZA", 200, {}), ("MC", 250, {}), ("M", 200, dict(rule=1, eta=0.1)), ("MC", 200, dict(rule=0, eta=0.002)), ("ZA", 150, dict(rule=1, eta=0.1))):
        kinds, caps, silent, kw = kinds_of(sysn); kw.update(extra); cutn = n - 800
        s = SysA(kinds, caps, silent, x, N=N, awake=1, **kw); a = s.run(6, cutn, True, True); f = s.run(cutn, n, False, True); g = s.run(cutn, n, True, True)
        R2, MA, st = ref_run(256, kinds, caps, silent, x, 6, cutn, None, True, N, True, **kw); Q2, MF, _ = ref_run(256, kinds, caps, silent, x, cutn, n, st, False, N, True, **kw); W2, MG, _ = ref_run(256, kinds, caps, silent, x, cutn, n, st, True, N, True, **kw)
        d = max(np.abs(a[5] - R2).max(), np.abs(f[5] - Q2).max(), np.abs(g[5] - W2).max(), np.abs(np.exp(-a[2]) - MA).max(), np.abs(np.exp(-f[2]) - MF).max(), np.abs(np.exp(-g[2]) - MG).max(), np.abs(s.shares() - np.array(st["w2"])).max()); sz = s.sizes()
        x2 = x.copy(); n0 = cutn - 300; x2[n0:] = (x2[n0:] + 1 + rng.integers(0, 255, n - n0)) % 256; sa = SysA(kinds, caps, silent, x, N=N, awake=1, **kw); sb = SysA(kinds, caps, silent, x2, N=N, awake=1, **kw); u = sa.run(6, n0 + 1, True); v = sb.run(6, n0 + 1, True); blind = bool(np.array_equal(u[1][:-1], v[1][:-1]) and np.array_equal(u[2][:-1], v[2][:-1]))
        tol = 1e-4 if kw.get("rule") == 2 else 1e-8      # under the rule of the step, relative rounding of small drivers is carried into the trusts (lives.py); here a bookmark's trusts learn by it as well: to 5e-5 with 29 senders; the same logic under the other two rules agrees to 1e-11
        good = d < tol and blind and sz["bytes_at_which_a_bookmark_learned"] == st["awoke"] and st["awoke"] > 200 and sz["bookmarks"] == len(st["books"]); ok &= good
        print(f"{n} bytes of seven pieces from three streams, system {sysn} ({len(kinds)} kinds), options {extra}, a bookmark every {N} bytes: compiled against the reference, largest difference {d:.1e} (the living set's answers and the mixture, while reading, with nothing changing and going on; the shares); bytes at which a bookmark learned {sz['bytes_at_which_a_bookmark_learned']} = {st['awoke']}; bookmarks {sz['bookmarks']}; no answer moves when its own byte and all later ones are replaced: {blind}", "PASSED" if good else "FAILED", flush=True)
    xs = LV.raw("py")[0][:40000]; s = make("MC", xs, 4096, 0); a = s.run(6, 40000, True); kinds, caps, silent, kw = CM.kinds_of("MC"); p = CM.SysC(kinds, caps, silent, xs, N=4096, **kw); b = p.run(6, 40000, True); dl = max(np.abs(a[1] - b[1]).max(), np.abs(a[2] - b[2]).max()); same = dl < 1e-6; ok &= same
    print(f"with the addition off the losses (the living set's and the mixture's) differ from nano/lives/cellmix.py's kernel by at most {dl:.1e} (compiled separately): {same}", "PASSED" if same else "FAILED", flush=True)
    s = make("ZA", xs, 4096, 1); a = s.run(6, 40000, True); p = BK.make("ZA", xs, 4096); b = p.run(6, 40000, True); dl = np.abs(a[1] - b[1]).max(); same = dl < 1e-6; ok &= same
    print(f"with the addition on, the living set's own losses differ from nano/lives/books.py's kernel by at most {dl:.1e} (the living set is untouched): {same}", "PASSED" if same else "FAILED", flush=True)
    return ok

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=", 1) for a in sys.argv[2:]); out = run_life(kw); r = lambda v: None if v is None else round(v, 4)
    for st in out["stages"]: print(out["tag"], f"after {st['stream']}:", " | ".join(f"{n} {r(t['going_on']['cell2'])} -> {r(t['going_on']['mix'])}" for n, t in st["tests"].items()), "(going on: the living set -> the mixture)", flush=True)
    e = out["stages"][-1]["tests"]; print(out["tag"], "bookmarks", out["stages"][-1]["sizes"]["bookmarks"], "| mean over the tests at the end, going on: living", r(float(np.mean([t["going_on"]["cell2"] for t in e.values()]))), "mixture", r(float(np.mean([t["going_on"]["mix"] for t in e.values()]))), "| nothing changing: mixture", r(float(np.mean([t["nothing_changing"]["mix"] for t in e.values()]))), "| wall", out["wall"], flush=True)
