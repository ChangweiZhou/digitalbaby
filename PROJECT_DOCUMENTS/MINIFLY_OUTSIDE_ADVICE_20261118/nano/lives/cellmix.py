#!/usr/bin/env python3
"""nano/lives, PLAN_7.md: a cell's own belief in its memories. A context of 0 to 3 bytes has three cells (PLAN_5.md: one counts for life, one halves its counts
at 512, one at 32). Here the life cell also holds a share for each of the three; a sender says the shares' mixture of the three opinions; after each byte the
shares move by how well each memory foretold it.
    python3 cellmix.py check
    python3 cellmix.py life order=.. sys=M|C|MC tag=..
The kernel is nano/lives/mem.py's with the additions below; with no mixture sender and no silent kind it is that kernel."""
import ctypes, hashlib, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lives as LV, books as BK, mem as MM
SU = LV.SU; NL = LV.NL; LMAX = LV.LMAX; PD = LV.PD; PI = LV.PI; rep = LV.rep; p32 = LV.p32; pd = LV.pd; m = LV.m; SIG = 1.0 / 32

SENDER = r'''
            if (s->ktype[k] == 6) {                                   /* PLAN_7: the mixture of a cell's three memories, by the cell's own shares */
                const int32_t cl = s->hcell[sl[s->knd[k]]]; if (cl < 0 || s->cN[cl] < 1) continue;
                static double cnt[3][SMAX]; int32_t cm[3]; double wn[3]; const float *w = s->cw + 3 * (int64_t)cl; cm[0] = cl; cm[1] = s->hcell[sl[s->kdel[k][0]]]; cm[2] = s->hcell[sl[s->kdel[k][1]]];
                for (int q = 0; q < 3; q++) { const int32_t c = cm[q]; wn[q] = (double)w[q] / ((c >= 0 ? (double)s->cN[c] : 0.0) + S * s->e); if (c >= 0) { const uint8_t *pv = s->pv + s->cptr[c]; const uint32_t *pc = s->pc + s->cptr[c]; for (int i = 0; i < s->cU[c]; i++) cnt[q][pv[i]] = pc[i]; } }
                const double lun = log((wn[0] + wn[1] + wn[2]) * s->e); const int u = s->cU[cl]; lev[k] = level(s->cN[cl], u); nv[k] = u; vv[k] = s->pv + s->cptr[cl]; const double tk = s->T[k * NL + lev[k]];
                for (int i = 0; i < u; i++) { const int v = vv[k][i]; const double d = log(wn[0] * (cnt[0][v] + s->e) + wn[1] * (cnt[1][v] + s->e) + wn[2] * (cnt[2][v] + s->e)) - lun; dv[k][i] = d; a[v] += tk * d; }
                for (int q = 0; q < 3; q++) { const int32_t c = cm[q]; if (c >= 0) { const uint8_t *pv = s->pv + s->cptr[c]; for (int i = 0; i < s->cU[c]; i++) cnt[q][pv[i]] = 0; } }
                continue;
            }
'''
SHARES = r'''
            for (int k = 0; k < K; k++) if (s->ktype[k] == 6 && lev[k] >= 0) {      /* PLAN_7: the cell's shares move by how well each memory foretold the byte */
                const int32_t cl = s->hcell[sl[s->knd[k]]]; float *w = s->cw + 3 * (int64_t)cl; double o[3], z = 0;
                for (int q = 0; q < 3; q++) { const int32_t c = q == 0 ? cl : s->hcell[sl[s->kdel[k][q - 1]]]; double cy = 0, n = 0; if (c >= 0) { n = s->cN[c]; const uint8_t *pv = s->pv + s->cptr[c]; const uint32_t *pc = s->pc + s->cptr[c]; for (int i = 0; i < s->cU[c]; i++) if (pv[i] == y) { cy = pc[i]; break; } }
                    o[q] = ((1.0 - SIG) * (double)w[q] + SIG / 3.0) * (cy + s->e) / (n + S * s->e); z += o[q]; }
                for (int q = 0; q < 3; q++) w[q] = (float)(o[q] / z);
            }
'''
SRC = MM.SRC
SRC = rep(SRC, "#define SMAX 256", "#define SMAX 256\n#define SIG (1.0 / 32)")
SRC = rep(SRC, "int ktype[KMAX], knd[KMAX], kdel[KMAX][8], kcap[KMAX];", "int ktype[KMAX], knd[KMAX], kdel[KMAX][8], kcap[KMAX], ksil[KMAX]; float *cw;")
SRC = rep(SRC, "s->cL = realloc(s->cL, s->ccells); s->cT = realloc(s->cT, s->ccells * 4); }", "s->cL = realloc(s->cL, s->ccells); s->cT = realloc(s->cT, s->ccells * 4); s->cw = realloc(s->cw, s->ccells * 12); }")
SRC = rep(SRC, "int32_t c = (int32_t)s->ncell++; s->cN[c] = 0; s->cU[c] = 0; s->ccap[c] = 0; s->cptr[c] = 0; return c;", "int32_t c = (int32_t)s->ncell++; s->cN[c] = 0; s->cU[c] = 0; s->ccap[c] = 0; s->cptr[c] = 0; s->cw[3 * (int64_t)c] = s->cw[3 * (int64_t)c + 1] = s->cw[3 * (int64_t)c + 2] = (float)(1.0 / 3.0); return c;")
SRC = rep(SRC, "s->cL = malloc(s->ccells); s->cT = malloc(s->ccells * 4);", "s->cL = malloc(s->ccells); s->cT = malloc(s->ccells * 4); s->cw = malloc(s->ccells * 12);")
SRC = rep(SRC, "free(s->cL); free(s->cT);", "free(s->cL); free(s->cT); free(s->cw);")
SRC = rep(SRC, "void sy_caps(Sy *s, const int32_t *caps) {", "void sy_silent(Sy *s, const int32_t *sil) { for (int k = 0; k < s->K; k++) s->ksil[k] = sil[k]; }\nvoid sy_caps(Sy *s, const int32_t *caps) {")
SRC = rep(SRC, "            uint64_t key = 0;\n", SENDER.strip("\n") + "\n            uint64_t key = 0;\n")
SRC = rep(SRC, "keys[k] = key; sl[k] = slot_of(s, key, k); const int32_t c = s->hcell[sl[k]]; if (c < 0 || s->cN[c] < 1) continue;", "keys[k] = key; sl[k] = slot_of(s, key, k); const int32_t c = s->hcell[sl[k]]; if (c < 0 || s->cN[c] < 1 || s->ksil[k]) continue;")
SRC = rep(SRC, "            for (int k = 0; k < K; k++) if (s->ktype[k] < 2 && sl[k] >= 0) {\n", SHARES.strip("\n") + "\n            for (int k = 0; k < K; k++) if (s->ktype[k] < 2 && sl[k] >= 0) {\n")

def kern():
    h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_cellmix_{h}.so"); src = os.path.join(HERE, "kern_cellmix.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd, cl, vp = ctypes.c_int, ctypes.c_double, ctypes.c_int64, ctypes.c_void_p
    L.sy_new.argtypes = [ci, ci, PI, PI, PI, PD, cd, cd, ci, ci, ci, cd, ci, cl, ci, cd]; L.sy_new.restype = vp; L.sy_free.argtypes = [vp]; L.sy_free.restype = None
    L.sy_run.argtypes = [vp, PI, cl, cl, ci, PD, PD, PD, PD]; L.sy_run.restype = None
    for f in ("sy_cells", "sy_counts", "sy_once", "sy_gsize", "sy_now"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = cl
    for f in ("sy_T", "sy_Tu", "sy_Ta", "sy_G", "sy_Gu", "sy_Ga", "sy_w2"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = PD
    L.sy_gmin.argtypes = [vp, cd]; L.sy_gmin.restype = None; L.sy_books.argtypes = [vp, cl, ci, cd]; L.sy_books.restype = None; L.sy_mixout.argtypes = [vp, PD, PI]; L.sy_mixout.restype = None; L.sy_nb.argtypes = [vp]; L.sy_nb.restype = ci
    L.sy_caps.argtypes = [vp, PI]; L.sy_caps.restype = None; L.sy_silent.argtypes = [vp, PI]; L.sy_silent.restype = None; L.sy_clear_kind.argtypes = [vp, ci]; L.sy_clear_kind.restype = None; L.sy_clear_places.argtypes = [vp]; L.sy_clear_places.restype = None; L.sy_cells_of.argtypes = [vp, ci]; L.sy_cells_of.restype = cl; return L

def kinds_of(name, S=256):
    """M: PLAN_5.md's chosen form. C: its bounded kinds silent, and for each run of 0 to 3 bytes a sender of the cell's mixture. MC: all speak.
    A mixture kind is (6, index of the life kind, (index of the kind bounded at 512, index of the kind bounded at 32), trust at birth, name)."""
    kinds, caps, a = MM.kinds_caps("ZAM32x512", S); silent = [0] * len(kinds); idx = {k[4]: i for i, k in enumerate(kinds)}
    if name in ("C", "MC"):
        if name == "C": silent = [1 if k[4].startswith("recent") else 0 for k in kinds]
        for r in range(4): kinds.append((6, idx[f"run{r}"], (idx[f"recent512_run{r}"], idx[f"recent32_run{r}"]), 0.0, f"cellmix_run{r}")); caps.append(0); silent.append(0)
    else: assert name == "M"
    return kinds, caps, silent, a

class SysC(MM.SysM):
    def __init__(s, kinds, caps, silent, x, S=256, eta=0.00005, e=0.02, rule=0, gate=0, J=6, birth=0.0, M=8, age=0, spow=0.0, guard=1e-20, N=0, cap=0):
        s.S, s.kinds, s.K, s.gate, s.J, s.M, s.N = S, kinds, len(kinds), gate, J, M, N; assert S <= 256 and s.K <= 600 and J <= 8; s.L = kern(); s.x = np.ascontiguousarray(x.astype(np.int32)); assert 0 <= s.x.min() and s.x.max() < S
        kt = np.array([k[0] for k in kinds], np.int32); kn = np.array([k[1] for k in kinds], np.int32); kd = np.zeros((s.K, 8), np.int32); kb = np.array([k[3] for k in kinds], np.float64)
        for i, k in enumerate(kinds): kd[i, :len(k[2])] = k[2]
        s.h = s.L.sy_new(S, s.K, p32(kt), p32(kn), p32(np.ascontiguousarray(kd)), pd(kb), eta, e, rule, gate, J, birth, M if any(k[0] == 2 for k in kinds) else 0, len(s.x), age, spow); s.L.sy_gmin(s.h, guard); s.L.sy_caps(s.h, p32(np.array(caps, np.int32))); s.L.sy_silent(s.h, p32(np.array(silent, np.int32)))
        if N > 0: s.L.sy_books(s.h, N, cap or (len(s.x) // N + 2), 1.0 / N)

def make(name, x, N=0): kinds, caps, silent, a = kinds_of(name); return SysC(kinds, caps, silent, x, N=N, **a)

# ---------------------------------------------------------------- the reference in plain Python: books.py's (with bounded kinds), and the cell's shares and mixture written out
def ref_run(S, kinds, caps, silent, x, lo, hi, state=None, learn=True, eta=0.00005, e=0.02, rule=0, gate=0, J=6, birth=0.0, M=8, age=0, spow=0.0, guard=1e-20):
    K = len(kinds); GL = NL if gate == 2 else 1
    if state is None: state = dict(cells={}, cw={}, T=np.array([[k[3]] * NL for k in kinds], np.float64), Tu=np.zeros((K, NL)), Ta=np.zeros((K, NL)), G=np.zeros((J, K, GL, S)), Gu=np.zeros((J, K, GL, S)), Ga=np.zeros((J, K, GL, S)), places={}, ptime={}, lasts={}, times={}, last={}, now=0)
    cells, cw, T, Tu, Ta, G, Gu, Ga, places = (state[k] for k in ("cells", "cw", "T", "Tu", "Ta", "G", "Gu", "Ga", "places")); P1 = np.zeros((hi - lo, S)); P2 = np.zeros((hi - lo, S)); xb = lambda q: int(x[q]) if q >= 0 else 0; zero = np.zeros(S)
    step = lambda used, acc: eta if rule == 0 else eta / np.sqrt(1.0 + used) if rule == 1 else eta * sc[0] / np.sqrt(acc); sc = [1.0]
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
            if typ == 6:                                                # the mixture of the three memories of the context of the life kind nd, by the cell's own shares
                keys.append(None); kl = keys[nd]; cl = cells.get(kl)
                if cl is not None and cl.sum() >= 1:
                    mem = [cl, cells.get(keys[ds[0]], zero), cells.get(keys[ds[1]], zero)]; w = cw[kl]; wn = [float(w[q]) / (mem[q].sum() + S * e) for q in range(3)]
                    o = wn[0] * (mem[0] + e) + wn[1] * (mem[1] + e) + wn[2] * (mem[2] + e); dv = np.where(cl > 0, np.log(o) - np.log((wn[0] + wn[1] + wn[2]) * e), 0.0); l = SU.level_np(cl.sum(), int((cl > 0).sum())); act.append((k, l, dv)); a += T[k, l] * dv; mixes.append((k, kl, ds))
                continue
            key = (k,) + tuple(xb(t - d) for d in ds); keys.append(key); c = cells.get(key)
            if c is not None and c.sum() >= 1 and not silent[k]:
                l = SU.level_np(c.sum(), int((c > 0).sum())); dv = np.log(c + e) - np.log(e); act.append((k, l, dv)); a += T[k, l] * dv
        p = np.exp(a - a.max()); p /= p.sum(); P1[t - lo] = p; a2 = a.copy()
        for k, l, dv in act: a2 += sum(G[j, k, l if GL > 1 else 0, xb(t - 1 - j)] for j in range(J)) * dv
        p2 = np.exp(a2 - a2.max()); p2 /= p2.sum(); P2[t - lo] = p2
        if learn:
            sc[0] = (12.0 / len(act)) ** spow if spow > 0 and act else 1.0
            for k, l, dv in act:
                g = dv[y] - p @ dv; Ta[k, l] += g * g
                if Ta[k, l] > guard or rule != 2: T[k, l] += step(Tu[k, l], Ta[k, l]) * g
                Tu[k, l] += 1; g2 = dv[y] - p2 @ dv
                for j in range(J):
                    i = (j, k, l if GL > 1 else 0, xb(t - 1 - j)); Ga[i] += g2 * g2
                    if Ga[i] > guard or rule != 2: G[i] += step(Gu[i], Ga[i]) * g2
                    Gu[i] += 1
            for k, kl, ds in mixes:                                     # the cell's shares, before anything is counted
                mem = [cells[kl], cells.get(keys[ds[0]], zero), cells.get(keys[ds[1]], zero)]; w = cw[kl]; o = [((1.0 - SIG) * float(w[q]) + SIG / 3.0) * (mem[q][y] + e) / (mem[q].sum() + S * e) for q in range(3)]; z = o[0] + o[1] + o[2]; cw[kl] = np.array([o[q] / z for q in range(3)], np.float32)
            for k, key in enumerate(keys):
                if key is None: continue
                if key not in cells:
                    if birth > 0 and p[y] >= birth: continue
                    cells[key] = np.zeros(S); cw[key] = np.full(3, 1.0 / 3.0, np.float32)
                cells[key][y] += 1; state["lasts"][key] = y; state["times"][key] = tau
                if caps[k] > 0 and cells[key].sum() >= caps[k]: cells[key] = np.floor(cells[key] / 2)
            if any(k[0] == 2 for k in kinds) and t >= M: places.setdefault(tuple(int(v) for v in x[t - M:t]), []).append(t); state["ptime"][t] = tau
        state["last"][y] = tau
    state["now"] += hi - lo; return P1, P2, state

def score(s, lo, hi, learn):
    t0 = time.process_time(); n1, n2, mx, top = s.run(lo, hi, learn); r = dict(cell1=m(n1), cell2=m(n2), mix=m(mx), first={str(k): [m(n2[:k]), m(mx[:k])] for k in (1000, 4000, 16000, 64000) if k <= hi - lo}, cpu=round(time.process_time() - t0, 1))
    if learn: r["sizes_after"] = s.sizes()
    return r

def run_life(kw):
    order = kw["order"]; name = kw.get("sys", "C"); tag = kw["tag"]; N = int(kw.get("N", 0))
    if order.startswith("seed:"): order = [LV.NAMES[i] for i in np.random.default_rng(int(order[5:])).permutation(len(LV.NAMES))]
    else: order = order.split(",")
    tests = list(LV.NAMES) if len(order) > 1 else list(order); x, body, test = LV.layout(order, tests); s = make(name, x, N); tmp = os.path.join(HERE, "tmp", tag); os.makedirs(tmp, exist_ok=True); t0 = time.time(); stages = []; child = None
    def stage_scores(i):
        def fn():
            for n in tests:
                for learn in (0, 1): LV.wait(LV.forked(lambda: score(s, test[n][0], test[n][1], bool(learn)), os.path.join(tmp, f"s{i}_{n}_{learn}.json")))
            return dict(done=True)
        return fn
    for i, n in enumerate(order):
        lo, hi = body[n]; lo += 6 if i == 0 else 0; c0 = time.process_time(); n1, n2, mx, top = s.run(lo, hi, True); k = (hi - lo) // 10
        stages.append(dict(stream=n, reading=dict(cell1=m(n1), cell2=m(n2), mix=m(mx), first_tenth=[m(n2[:k]), m(mx[:k])], last_tenth=[m(n2[-k:]), m(mx[-k:])]), sizes=s.sizes(), cpu_reading=round(time.process_time() - c0, 1)))
        if child is not None: LV.wait(child)
        child = LV.forked(stage_scores(i), os.path.join(tmp, f"s{i}_done.json")); print(tag, "read", n, {k_: round(v, 4) for k_, v in stages[-1]["reading"].items() if not k_.endswith("tenth")}, "cells", stages[-1]["sizes"]["cells"], "cpu", stages[-1]["cpu_reading"], flush=True)
    if child is not None: LV.wait(child)
    for i, st in enumerate(stages): st["tests"] = {n: dict(nothing_changing=json.load(open(os.path.join(tmp, f"s{i}_{n}_0.json"))), going_on=json.load(open(os.path.join(tmp, f"s{i}_{n}_1.json")))) for n in tests}
    out = dict(task="cellmix", tag=tag, sys=name, N=N, order=order, stages=stages, length=int(len(x)), wall=round(time.time() - t0, 1)); json.dump(out, open(os.path.join(HERE, "res", f"cellmix__{tag}.json"), "w")); return out

def check():
    ok = True; rng = np.random.default_rng(19)
    for name, sysn, extra in (("shakespeare", "C", {}), ("table", "MC", {}), ("xml", "C", dict(rule=1, eta=0.1)), ("c", "MC", dict(rule=1, eta=0.1)), ("py", "M", {}), ("elf", "C", dict(rule=0, eta=0.002))):
        n = 2500; x = LV.raw(name)[0][:n] if name != "elf" else np.frombuffer(open(os.path.realpath(sys.executable), "rb").read()[4096:4096 + n], np.uint8).astype(np.int32); kinds, caps, silent, kw = kinds_of(sysn); kw.update(extra); cutn = n - 500
        s = SysC(kinds, caps, silent, x, **kw); a = s.run(6, cutn, True, True); f = s.run(cutn, n, False, True); g = s.run(cutn, n, True, True)
        R1, R2, st = ref_run(256, kinds, caps, silent, x, 6, cutn, None, True, **kw); Q1, Q2, _ = ref_run(256, kinds, caps, silent, x, cutn, n, st, False, **kw); W1, W2, _ = ref_run(256, kinds, caps, silent, x, cutn, n, st, True, **kw)
        d = max(np.abs(a[4] - R1).max(), np.abs(f[4] - Q1).max(), np.abs(g[4] - W1).max(), np.abs(a[5] - R2).max(), np.abs(f[5] - Q2).max(), np.abs(g[5] - W2).max(), np.abs(s.arr("T") - st["T"]).max())
        cells_ref = sum(1 for c in st["cells"].values() if c.sum() > 0); sz = s.sizes(); moved = sum(1 for w in st["cw"].values() if abs(float(w[0]) - 1 / 3) > 0.05)
        x2 = x.copy(); n0 = cutn - 300; x2[n0:] = (x2[n0:] + 1 + rng.integers(0, 255, n - n0)) % 256; sa = SysC(kinds, caps, silent, x, **kw); sb = SysC(kinds, caps, silent, x2, **kw); u = sa.run(6, n0 + 1, True); v = sb.run(6, n0 + 1, True); blind = bool(np.array_equal(u[1][:-1], v[1][:-1]))
        tol = 1e-5 if kw.get("rule") == 2 else 1e-8; good = d < tol and blind and sz["cells"] == cells_ref and (sysn == "M" or moved > 20); ok &= good
        print(f"{name} ({n} raw bytes), system {sysn} ({len(kinds)} kinds, {sum(silent)} silent, {sum(1 for k in kinds if k[0] == 6)} mixture senders), options {extra}: compiled against the reference, largest difference {d:.1e} (both cells' answers while reading, with nothing changing, going on; the trusts); cells {sz['cells']} = {cells_ref}; cells whose shares have moved {moved}; no answer moves when its own byte and all later ones are replaced: {blind}", "PASSED" if good else "FAILED", flush=True)
    x = LV.raw("py")[0][:40000]; s = make("M", x); a = s.run(6, 40000, True); p = MM.make("ZAM32x512", x); b = p.run(6, 40000, True); dl = max(np.abs(a[0] - b[0]).max(), np.abs(a[1] - b[1]).max()); same = dl < 1e-6; ok &= same
    print(f"with no mixture sender and no silent kind the losses differ from nano/lives/mem.py's kernel by at most {dl:.1e} (compiled separately): {same}", "PASSED" if same else "FAILED", flush=True)
    return ok

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=", 1) for a in sys.argv[2:]); out = run_life(kw); r = lambda v: None if v is None else round(v, 4)
    for st in out["stages"]: print(out["tag"], f"after {st['stream']}:", " | ".join(f"{n} {r(t['nothing_changing']['cell2'])} / {r(t['going_on']['cell2'])}" for n, t in st["tests"].items()), "(second cell: nothing changing / going on)", flush=True)
    e = out["stages"][-1]["tests"]; print(out["tag"], "| mean over the tests at the end, going on:", r(float(np.mean([t["going_on"]["cell2"] for t in e.values()]))), "| cells", out["stages"][-1]["sizes"]["cells"], "| wall", out["wall"], flush=True)
