#!/usr/bin/env python3
"""nano/lives, PLAN_4.md with its addition 1: return. PLAN_3.md's bookmarks and shares, and: when before a byte a bookmark's share is above one half, then after the
byte has been learned the living set takes that bookmark's trusts and parts, and its share (the living set is not kept: the bookmarks taken at the pace hold its states).
    python3 wake.py check
    python3 wake.py life order=.. sys=ZA|ZAM32|.. N=65536 [wake=1] tag=..
The kernel is nano/lives/mem.py's (remembered trusts and bounded kinds) with the additions below; with wake off it is that kernel."""
import ctypes, hashlib, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lives as LV, books as BK, mem as MM
SU = LV.SU; NL = LV.NL; PD = LV.PD; PI = LV.PI; rep = LV.rep; p32 = LV.p32; pd = LV.pd; m = LV.m

SRC = MM.SRC
SRC = rep(SRC, "int nb, bcap; int64_t bper, lnow;", "int nb, bcap, wake, jtop; int64_t returns; int64_t bper, lnow;")
SRC = rep(SRC, "int sy_nb(Sy *s) { return s->nb; }", "int sy_nb(Sy *s) { return s->nb; }\nvoid sy_wake(Sy *s, int w) { s->wake = w; }\nint64_t sy_returns(Sy *s) { return s->returns; }")
SRC = rep(SRC, "            if (s->to2) s->to2[t - lo] = top;\n", "            if (s->to2) s->to2[t - lo] = top;\n            s->jtop = (learn && s->wake && top > 0 && topv > 0.5) ? top : 0;      /* PLAN_4: a bookmark holds more than half of the belief */\n")
SRC = rep(SRC, "/* PLAN_3: a bookmark */\n", "/* PLAN_3: a bookmark */\n"
          "            if (s->bcap > 0 && s->jtop > 0) {      /* PLAN_4, addition 1: the living set takes the trusts and parts of the bookmark jtop - 1, and its share; the living set is not kept (the bookmarks of the pace hold its states) */\n"
          "                const int b = s->jtop - 1, bc = s->bcap;\n"
          "                for (int64_t i = 0; i < (int64_t)K * NL; i++) s->T[i] = s->BT[i * bc + b];\n"
          "                for (int64_t i = 0; i < s->gsize; i++) s->G[i] = s->BG[i * bc + b];\n"
          "                s->w2[0] += s->w2[b + 1]; s->w2[b + 1] = 0; s->returns++;\n"
          "            }\n"
          "            s->jtop = 0;\n")

def kern():
    h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_wake_{h}.so"); src = os.path.join(HERE, "kern_wake.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd, cl, vp = ctypes.c_int, ctypes.c_double, ctypes.c_int64, ctypes.c_void_p
    L.sy_new.argtypes = [ci, ci, PI, PI, PI, PD, cd, cd, ci, ci, ci, cd, ci, cl, ci, cd]; L.sy_new.restype = vp; L.sy_free.argtypes = [vp]; L.sy_free.restype = None
    L.sy_run.argtypes = [vp, PI, cl, cl, ci, PD, PD, PD, PD]; L.sy_run.restype = None
    for f in ("sy_cells", "sy_counts", "sy_once", "sy_gsize", "sy_now", "sy_returns"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = cl
    for f in ("sy_T", "sy_Tu", "sy_Ta", "sy_G", "sy_Gu", "sy_Ga", "sy_w2"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = PD
    L.sy_gmin.argtypes = [vp, cd]; L.sy_gmin.restype = None; L.sy_books.argtypes = [vp, cl, ci, cd]; L.sy_books.restype = None; L.sy_mixout.argtypes = [vp, PD, PI]; L.sy_mixout.restype = None; L.sy_nb.argtypes = [vp]; L.sy_nb.restype = ci
    L.sy_caps.argtypes = [vp, PI]; L.sy_caps.restype = None; L.sy_clear_kind.argtypes = [vp, ci]; L.sy_clear_kind.restype = None; L.sy_clear_places.argtypes = [vp]; L.sy_clear_places.restype = None; L.sy_cells_of.argtypes = [vp, ci]; L.sy_cells_of.restype = cl
    L.sy_wake.argtypes = [vp, ci]; L.sy_wake.restype = None; return L

class SysW(MM.SysM):
    def __init__(s, kinds, caps, x, S=256, eta=0.00005, e=0.02, rule=0, gate=0, J=6, birth=0.0, M=8, age=0, spow=0.0, guard=1e-20, N=0, cap=0, wake=0):
        s.S, s.kinds, s.K, s.gate, s.J, s.M, s.N = S, kinds, len(kinds), gate, J, M, N; assert S <= 256 and s.K <= 600 and J <= 8; s.L = kern(); s.x = np.ascontiguousarray(x.astype(np.int32)); assert 0 <= s.x.min() and s.x.max() < S
        kt = np.array([k[0] for k in kinds], np.int32); kn = np.array([k[1] for k in kinds], np.int32); kd = np.zeros((s.K, 8), np.int32); kb = np.array([k[3] for k in kinds], np.float64)
        for i, k in enumerate(kinds): kd[i, :len(k[2])] = k[2]
        s.h = s.L.sy_new(S, s.K, p32(kt), p32(kn), p32(np.ascontiguousarray(kd)), pd(kb), eta, e, rule, gate, J, birth, M if any(k[0] == 2 for k in kinds) else 0, len(s.x), age, spow); s.L.sy_gmin(s.h, guard); s.L.sy_caps(s.h, p32(np.array(caps, np.int32)))
        if N > 0: s.L.sy_books(s.h, N, cap or (len(s.x) // N + 2), 1.0 / N); s.L.sy_wake(s.h, int(wake))
    def sizes(s): z = MM.SysM.sizes(s); z["returns"] = int(s.L.sy_returns(s.h)); return z

def make(name, x, N=0, wake=0): kinds, caps, a = MM.kinds_caps(name); return SysW(kinds, caps, x, N=N, wake=wake, **a)

def score(s, lo, hi, learn):
    t0 = time.process_time(); n1, n2, mx, top = s.run(lo, hi, learn); r = dict(cell1=m(n1), cell2=m(n2), mix=m(mx), first={str(k): [m(n2[:k]), m(mx[:k])] for k in (1000, 4000, 16000, 64000) if k <= hi - lo}, share_on_living=float((top == 0).mean()), cpu=round(time.process_time() - t0, 1))
    if learn: r["sizes_after"] = s.sizes()
    return r

def run_life(kw):
    order = kw["order"]; name = kw.get("sys", "ZA"); tag = kw["tag"]; N = int(kw.get("N", 65536)); wake = int(kw.get("wake", 1))
    if order.startswith("seed:"): order = [LV.NAMES[i] for i in np.random.default_rng(int(order[5:])).permutation(len(LV.NAMES))]
    else: order = order.split(",")
    tests = list(LV.NAMES) if len(order) > 1 else list(order); x, body, test = LV.layout(order, tests); s = make(name, x, N, wake); tmp = os.path.join(HERE, "tmp", tag); os.makedirs(tmp, exist_ok=True); t0 = time.time(); stages = []; child = None
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
        child = LV.forked(stage_scores(i), os.path.join(tmp, f"s{i}_done.json")); print(tag, "read", n, {k_: round(v, 4) for k_, v in stages[-1]["reading"].items() if not k_.endswith("tenth")}, "cells", stages[-1]["sizes"]["cells"], "bookmarks", s.nb, "returns", stages[-1]["sizes"]["returns"], "cpu", stages[-1]["cpu_reading"], flush=True)
    if child is not None: LV.wait(child)
    for i, st in enumerate(stages): st["tests"] = {n: dict(nothing_changing=json.load(open(os.path.join(tmp, f"s{i}_{n}_0.json"))), going_on=json.load(open(os.path.join(tmp, f"s{i}_{n}_1.json")))) for n in tests}
    out = dict(task="wake", tag=tag, sys=name, bounded=MM.SYSTEMS[name], N=N, wake=wake, order=order, stages=stages, length=int(len(x)), wall=round(time.time() - t0, 1)); json.dump(out, open(os.path.join(HERE, "res", f"wake__{tag}.json"), "w")); return out

def check():
    ok = True; rng = np.random.default_rng(17)
    # pieces of different kinds one after another, so that old bookmarks come to hold the belief again and returns occur
    pieces = [LV.raw(n)[0][a:a + L] for n, a, L in (("shakespeare", 0, 1500), ("table", 5000, 1200), ("shakespeare", 9000, 1200), ("table", 90000, 1000), ("xml", 0, 1200), ("shakespeare", 30000, 1200), ("table", 200000, 1000))]; x = np.ascontiguousarray(np.concatenate(pieces).astype(np.int32)); n = len(x)
    for sysn, N, extra in (("ZA", 200, {}), ("ZAM32", 150, {}), ("ZA", 200, dict(rule=1, eta=0.1)), ("ZA", 100, dict(rule=0, eta=0.002))):
        kinds, caps, kw = MM.kinds_caps(sysn); kw.update(extra); cutn = n - 800
        s = SysW(kinds, caps, x, N=N, wake=1, **kw); a = s.run(6, cutn, True, True); f = s.run(cutn, n, False, True); g = s.run(cutn, n, True, True)
        R2, MA, st = BK.ref_run(256, kinds, x, 6, cutn, None, True, N, caps=caps, wake=True, **kw); Q2, MF, _ = BK.ref_run(256, kinds, x, cutn, n, st, False, N, caps=caps, wake=True, **kw); W2, MG, _ = BK.ref_run(256, kinds, x, cutn, n, st, True, N, caps=caps, wake=True, **kw)
        d = max(np.abs(a[5] - R2).max(), np.abs(f[5] - Q2).max(), np.abs(g[5] - W2).max(), np.abs(np.exp(-a[2]) - MA).max(), np.abs(np.exp(-f[2]) - MF).max(), np.abs(np.exp(-g[2]) - MG).max(), np.abs(s.shares() - np.array(st["w2"])).max()); sz = s.sizes()
        x2 = x.copy(); n0 = cutn - 300; x2[n0:] = (x2[n0:] + 1 + rng.integers(0, 255, n - n0)) % 256; sa = SysW(kinds, caps, x, N=N, wake=1, **kw); sb = SysW(kinds, caps, x2, N=N, wake=1, **kw); u = sa.run(6, n0 + 1, True); v = sb.run(6, n0 + 1, True); blind = bool(np.array_equal(u[1][:-1], v[1][:-1]) and np.array_equal(u[2][:-1], v[2][:-1]))
        tol = 1e-5 if kw.get("rule") == 2 else 1e-8; good = d < tol and blind and sz["returns"] == st["returns"] and sz["bookmarks"] == len(st["books"]); ok &= good
        print(f"{n} bytes of seven pieces from three streams, system {sysn}, options {extra}, a bookmark every {N} bytes: compiled against the reference, largest difference {d:.1e} (the living set's answers and the mixture, while reading, with nothing changing and going on; the shares); returns {sz['returns']} = {st['returns']}, bookmarks {sz['bookmarks']} = {len(st['books'])}; no answer moves when its own byte and all later ones are replaced: {blind}", "PASSED" if good else "FAILED", flush=True)
    xs = LV.raw("py")[0][:40000]; s = make("ZAM32", xs, 4096, 0); a = s.run(6, 40000, True); p = MM.make("ZAM32", xs, 4096); b = p.run(6, 40000, True); dl = max(np.abs(a[1] - b[1]).max(), np.abs(a[2] - b[2]).max()); same = dl < 1e-6 and s.sizes()["returns"] == 0; ok &= same
    print(f"with the addition switched off the losses differ from nano/lives/mem.py's kernel by at most {dl:.1e} (compiled separately): {same}", "PASSED" if same else "FAILED", flush=True)
    return ok

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=", 1) for a in sys.argv[2:]); out = run_life(kw); r = lambda v: None if v is None else round(v, 4)
    for st in out["stages"]: print(out["tag"], f"after {st['stream']}:", " | ".join(f"{n} {r(t['going_on']['cell2'])} -> {r(t['going_on']['mix'])}" for n, t in st["tests"].items()), "(going on: the living set -> the mixture)", flush=True)
    e = out["stages"][-1]["tests"]; print(out["tag"], "bookmarks", out["stages"][-1]["sizes"]["bookmarks"], "returns", out["stages"][-1]["sizes"]["returns"], "| mean over the tests at the end, going on: mixture", r(float(np.mean([t["going_on"]["mix"] for t in e.values()]))), "| nothing changing:", r(float(np.mean([t["nothing_changing"]["mix"] for t in e.values()]))), "| wall", out["wall"], flush=True)
