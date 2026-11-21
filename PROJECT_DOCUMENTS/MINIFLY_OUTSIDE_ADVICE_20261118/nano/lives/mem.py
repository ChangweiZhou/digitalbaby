#!/usr/bin/env python3
"""nano/lives, PLAN_5.md: counts in two memories. Beside a cell that counts for life, a cell of a bounded kind on the same context halves its counts whenever
they reach a bound, so that it holds what followed lately. Bounded kinds are senders like any other, born with no trust.
    python3 mem.py check
    python3 mem.py life order=.. sys=ZA|ZAM32|ZAM32x512|ZAM32all [N=0] [clear=short|long|places|all] tag=..
The kernel is nano/lives/books.py's (remembered trusts; N=0 turns them off) with the additions below; kinds without a bound are untouched.
clear= (a diagnostic, the boundary is given): before each new body the counts of the named cells are emptied (short: runs of 0 to 3 bytes; long: runs of 4 to 6;
places: the table of the recalling senders; all: all three)."""
import ctypes, hashlib, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lives as LV, books as BK
SU = LV.SU; NL = LV.NL; PD = LV.PD; PI = LV.PI; rep = LV.rep; p32 = LV.p32; pd = LV.pd; m = LV.m

FUNCS = r'''
static void halve(Sy *s, int32_t c) {      /* PLAN_5: every count of the cell is halved; a count that falls to nothing is dropped */
    uint8_t *pv = s->pv + s->cptr[c]; uint32_t *pc = s->pc + s->cptr[c]; const int u = s->cU[c]; int w = 0; uint32_t n = 0;
    for (int i = 0; i < u; i++) { const uint32_t h = pc[i] >> 1; if (h > 0) { pv[w] = pv[i]; pc[w] = h; w++; n += h; } }
    s->cU[c] = (uint16_t)w; s->cN[c] = n;
}
void sy_caps(Sy *s, const int32_t *caps) { for (int k = 0; k < s->K; k++) s->kcap[k] = caps[k]; }
void sy_clear_kind(Sy *s, int kind) { for (int64_t i = 0; i < s->hcap; i++) if (s->hcell[i] >= 0 && s->hkind[i] == kind) { s->cN[s->hcell[i]] = 0; s->cU[s->hcell[i]] = 0; } }
void sy_clear_places(Sy *s) { memset(s->head, 0xff, sizeof(int32_t) << s->hbits); }
'''
SRC = BK.SRC
SRC = rep(SRC, "int ktype[KMAX], knd[KMAX], kdel[KMAX][8];", "int ktype[KMAX], knd[KMAX], kdel[KMAX][8], kcap[KMAX];")
SRC = rep(SRC, "Sy *sy_new(int S, int K,", FUNCS.strip("\n") + "\nSy *sy_new(int S, int K,")
SRC = rep(SRC, "                count(s, c, y, tau);\n", "                count(s, c, y, tau); if (s->kcap[k] > 0 && s->cN[c] >= (uint32_t)s->kcap[k]) halve(s, c);\n")

def kern():
    h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_mem_{h}.so"); src = os.path.join(HERE, "kern_mem.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd, cl, vp = ctypes.c_int, ctypes.c_double, ctypes.c_int64, ctypes.c_void_p
    L.sy_new.argtypes = [ci, ci, PI, PI, PI, PD, cd, cd, ci, ci, ci, cd, ci, cl, ci, cd]; L.sy_new.restype = vp; L.sy_free.argtypes = [vp]; L.sy_free.restype = None
    L.sy_run.argtypes = [vp, PI, cl, cl, ci, PD, PD, PD, PD]; L.sy_run.restype = None
    for f in ("sy_cells", "sy_counts", "sy_once", "sy_gsize", "sy_now"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = cl
    for f in ("sy_T", "sy_Tu", "sy_Ta", "sy_G", "sy_Gu", "sy_Ga", "sy_w2"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = PD
    L.sy_gmin.argtypes = [vp, cd]; L.sy_gmin.restype = None; L.sy_books.argtypes = [vp, cl, ci, cd]; L.sy_books.restype = None; L.sy_mixout.argtypes = [vp, PD, PI]; L.sy_mixout.restype = None; L.sy_nb.argtypes = [vp]; L.sy_nb.restype = ci
    L.sy_caps.argtypes = [vp, PI]; L.sy_caps.restype = None; L.sy_clear_kind.argtypes = [vp, ci]; L.sy_clear_kind.restype = None; L.sy_clear_places.argtypes = [vp]; L.sy_clear_places.restype = None; L.sy_cells_of.argtypes = [vp, ci]; L.sy_cells_of.restype = cl; return L

# bounded kinds: (bound, the longest run that gets one). A bounded kind is a run kind of its own (its cells are its own), born with no trust.
SYSTEMS = dict(ZA=[], ZAM32=[(32, 3)], ZAM32x512=[(32, 3), (512, 3)], ZAM32all=[(32, 6)], ZAM8x64=[(8, 3), (64, 3)])
def kinds_caps(name, S=256):
    a = dict(LV.SYSTEMS["ZA"]); kinds = SU.kinds_of(a.pop("kinds"), S, a.pop("recall")); caps = [0] * len(kinds)
    for cap, longest in SYSTEMS[name]:
        for k in range(longest + 1): kinds.append((0, k, tuple(range(1, k + 1)), 0.0, f"recent{cap}_run{k}")); caps.append(cap)
    return kinds, caps, a

class SysM(BK.SysB):
    def __init__(s, kinds, caps, x, S=256, eta=0.00005, e=0.02, rule=0, gate=0, J=6, birth=0.0, M=8, age=0, spow=0.0, guard=1e-20, N=0, cap=0):
        s.S, s.kinds, s.K, s.gate, s.J, s.M, s.N = S, kinds, len(kinds), gate, J, M, N; assert S <= 256 and s.K <= 600 and J <= 8; s.L = kern(); s.x = np.ascontiguousarray(x.astype(np.int32)); assert 0 <= s.x.min() and s.x.max() < S
        kt = np.array([k[0] for k in kinds], np.int32); kn = np.array([k[1] for k in kinds], np.int32); kd = np.zeros((s.K, 8), np.int32); kb = np.array([k[3] for k in kinds], np.float64)
        for i, k in enumerate(kinds): kd[i, :len(k[2])] = k[2]
        s.h = s.L.sy_new(S, s.K, p32(kt), p32(kn), p32(np.ascontiguousarray(kd)), pd(kb), eta, e, rule, gate, J, birth, M if any(k[0] == 2 for k in kinds) else 0, len(s.x), age, spow); s.L.sy_gmin(s.h, guard); s.L.sy_caps(s.h, p32(np.array(caps, np.int32)))
        if N > 0: s.L.sy_books(s.h, N, cap or (len(s.x) // N + 2), 1.0 / N)
    def clear(s, what):
        runs = [i for i, k in enumerate(s.kinds) if k[0] == 0 and k[4].startswith("run")]
        for i in runs:
            if (what in ("short", "all") and s.kinds[i][1] <= 3) or (what in ("long", "all") and s.kinds[i][1] >= 4): s.L.sy_clear_kind(s.h, i)
        if what in ("places", "all"): s.L.sy_clear_places(s.h)
    def sizes(s):
        z = BK.SysB.sizes(s); z["cells_of_bounded_kinds"] = int(sum(s.L.sy_cells_of(s.h, i) for i, k in enumerate(s.kinds) if k[4].startswith("recent"))); return z

def make(name, x, N=0): kinds, caps, a = kinds_caps(name); return SysM(kinds, caps, x, N=N, **a)

def score(s, lo, hi, learn):
    t0 = time.process_time(); n1, n2, mx, top = s.run(lo, hi, learn); r = dict(cell1=m(n1), cell2=m(n2), mix=m(mx), first={str(k): [m(n2[:k]), m(mx[:k])] for k in (1000, 4000, 16000, 64000) if k <= hi - lo}, share_on_living=float((top == 0).mean()), cpu=round(time.process_time() - t0, 1))
    if learn: r["sizes_after"] = s.sizes()
    return r

def run_life(kw):
    order = kw["order"]; name = kw.get("sys", "ZA"); tag = kw["tag"]; N = int(kw.get("N", 0)); clear = kw.get("clear")
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
        if i > 0 and clear: s.clear(clear)
        lo, hi = body[n]; lo += 6 if i == 0 else 0; c0 = time.process_time(); n1, n2, mx, top = s.run(lo, hi, True); k = (hi - lo) // 10
        stages.append(dict(stream=n, reading=dict(cell1=m(n1), cell2=m(n2), mix=m(mx), first_tenth=[m(n2[:k]), m(mx[:k])], last_tenth=[m(n2[-k:]), m(mx[-k:])]), sizes=s.sizes(), cpu_reading=round(time.process_time() - c0, 1)))
        if child is not None: LV.wait(child)
        child = LV.forked(stage_scores(i), os.path.join(tmp, f"s{i}_done.json")); print(tag, "read", n, {k_: round(v, 4) for k_, v in stages[-1]["reading"].items() if not k_.endswith("tenth")}, "cells", stages[-1]["sizes"]["cells"], "of bounded kinds", stages[-1]["sizes"]["cells_of_bounded_kinds"], "bookmarks", s.nb, "cpu", stages[-1]["cpu_reading"], flush=True)
    if child is not None: LV.wait(child)
    for i, st in enumerate(stages): st["tests"] = {n: dict(nothing_changing=json.load(open(os.path.join(tmp, f"s{i}_{n}_0.json"))), going_on=json.load(open(os.path.join(tmp, f"s{i}_{n}_1.json")))) for n in tests}
    out = dict(task="mem", tag=tag, sys=name, bounded=SYSTEMS[name], N=N, clear=clear, order=order, stages=stages, length=int(len(x)), wall=round(time.time() - t0, 1)); json.dump(out, open(os.path.join(HERE, "res", f"mem__{tag}.json"), "w")); return out

def check():
    ok = True; rng = np.random.default_rng(13)
    for name, sysn, N, extra in (("shakespeare", "ZAM32x512", 0, {}), ("table", "ZAM32", 300, {}), ("xml", "ZAM8x64", 0, {}), ("c", "ZAM32all", 250, {}), ("py", "ZAM8x64", 0, dict(rule=1, eta=0.1))):
        n = 2500; x = LV.raw(name)[0][:n]; kinds, caps, kw = kinds_caps(sysn); kw.update(extra); cutn = n - 500
        s = SysM(kinds, caps, x, N=N, **kw); a = s.run(6, cutn, True, True); f = s.run(cutn, n, False, True); g = s.run(cutn, n, True, True)
        R2, MA, st = BK.ref_run(256, kinds, x, 6, cutn, None, True, N, caps=caps, **kw); Q2, MF, _ = BK.ref_run(256, kinds, x, cutn, n, st, False, N, caps=caps, **kw); W2, MG, _ = BK.ref_run(256, kinds, x, cutn, n, st, True, N, caps=caps, **kw)
        d = max(np.abs(a[5] - R2).max(), np.abs(f[5] - Q2).max(), np.abs(g[5] - W2).max())
        if N > 0: d = max(d, np.abs(np.exp(-a[2]) - MA).max(), np.abs(np.exp(-g[2]) - MG).max())
        cells_ref = sum(1 for c in st["cells"].values() if c.sum() > 0); counts_ref = sum(int((c > 0).sum()) for c in st["cells"].values()); sz = s.sizes(); halved = sum(1 for key, c in st["cells"].items() if caps[key[0]] > 0 and st["times"][key] >= 0 and c.sum() < 0.75 * caps[key[0]] and c.sum() >= caps[key[0]] // 2 - 8)
        x2 = x.copy(); n0 = cutn - 300; x2[n0:] = (x2[n0:] + 1 + rng.integers(0, 255, n - n0)) % 256; sa = SysM(kinds, caps, x, N=N, **kw); sb = SysM(kinds, caps, x2, N=N, **kw); u = sa.run(6, n0 + 1, True); v = sb.run(6, n0 + 1, True); blind = bool(np.array_equal(u[1][:-1], v[1][:-1]) and np.array_equal(u[2][:-1], v[2][:-1]))
        tol = 1e-5 if kw.get("rule") == 2 else 1e-8      # under the rule of the step, relative rounding of small drivers is carried into the trusts (lives.py); with 25 senders to 6e-6
        good = d < tol and blind and sz["cells"] == cells_ref and sz["counts_above_zero"] == counts_ref; ok &= good
        print(f"{name} ({n} raw bytes), system {sysn} ({len(kinds)} kinds, bounds {sorted(set(c for c in caps if c))}), bookmarks every {N}: compiled against the reference, largest difference {d:.1e} (the second cell's answers while reading, with nothing changing, going on" + ("; the mixture" if N else "") + f"); cells {sz['cells']} = {cells_ref}, counts {sz['counts_above_zero']} = {counts_ref}; no answer moves when its own byte and all later ones are replaced: {blind}", "PASSED" if good else "FAILED", flush=True)
    x = LV.raw("py")[0][:40000]; s = make("ZA", x); a = s.run(6, 40000, True); p = BK.make("ZA", x); b = p.run(6, 40000, True); dl = max(np.abs(a[0] - b[0]).max(), np.abs(a[1] - b[1]).max()); same = dl < 1e-6; ok &= same
    print(f"with no bounded kind the losses differ from nano/lives/books.py's kernel by at most {dl:.1e} (compiled separately): {same}", "PASSED" if same else "FAILED", flush=True)
    s = make("ZA", x); s.run(6, 20000, True); s.clear("all"); a = s.run(20000, 40000, True); f = make("ZA", x); T0 = {k: s.arr(k).copy() for k in ("T",)}; sz = s.sizes()
    s2 = make("ZA", x); s2.run(6, 20000, True); s2.clear("short"); z2 = s2.sizes(); s3 = make("ZA", x); s3.run(6, 20000, True); z3 = s3.sizes(); same = z2["cells"] < z3["cells"] and z2["cells"] > 0; ok &= same
    print(f"emptying the cells of runs of 0 to 3 bytes leaves {z2['cells']} of {z3['cells']} cells with counts: {same}", "PASSED" if same else "FAILED", flush=True)
    return ok

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=", 1) for a in sys.argv[2:]); out = run_life(kw); r = lambda v: None if v is None else round(v, 4)
    for st in out["stages"]: print(out["tag"], f"after {st['stream']}:", " | ".join(f"{n} {r(t['nothing_changing']['cell2'])} / {r(t['going_on']['cell2'])}" + (f" -> {r(t['going_on']['mix'])}" if out["N"] else "") for n, t in st["tests"].items()), "(second cell: nothing changing / going on" + (" -> the mixture going on)" if out["N"] else ")"), flush=True)
    e = out["stages"][-1]["tests"]; print(out["tag"], "| mean over the tests at the end, going on:", r(float(np.mean([t["going_on"]["cell2"] for t in e.values()]))), "mixture", r(float(np.mean([t["going_on"]["mix"] for t in e.values()]))), "| cells", out["stages"][-1]["sizes"]["cells"], "| wall", out["wall"], flush=True)
