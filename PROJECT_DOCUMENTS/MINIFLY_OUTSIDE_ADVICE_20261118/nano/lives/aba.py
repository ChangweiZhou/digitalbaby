#!/usr/bin/env python3
"""nano/lives, PLAN_11.md: addition, then English, then addition again, for cells that count read in four ways.
    python3 aba.py check
    python3 aba.py life seed=1 form=S2|S1|P|F [alpha=0.03] [lines=40000] tag=..
Forms: S2, S1: trusts shared by every cell of a kind and level (the programme's system), with and without the second answering cell. P: every cell holds a
trust of its own (born at the kind's value; it moves, when the cell speaks, by step x driver / root of (1 + the sum of its own squared drivers)). F: every
trust stays where it was born. The kernel is nano/lives/lives.py's with the additions below; with own = 0 it is that kernel."""
import ctypes, hashlib, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lives as LV
SU = LV.SU; NL = LV.NL; PD = LV.PD; PI = LV.PI; rep = LV.rep; p32 = LV.p32; pd = LV.pd; m = LV.m

SRC = LV.SRC
SRC = rep(SRC, "uint8_t *cL; uint32_t *cT; int64_t ccells;", "uint8_t *cL; uint32_t *cT; int64_t ccells; double *cg, *cga; int own; double kb[KMAX];      /* PLAN_11: a trust of its own for every cell, and the sum of its squared drivers */")
SRC = rep(SRC, "s->cL = realloc(s->cL, s->ccells); s->cT = realloc(s->cT, s->ccells * 4); }", "s->cL = realloc(s->cL, s->ccells); s->cT = realloc(s->cT, s->ccells * 4); s->cg = realloc(s->cg, s->ccells * 8); s->cga = realloc(s->cga, s->ccells * 8); }")
SRC = rep(SRC, "s->cL = malloc(s->ccells); s->cT = malloc(s->ccells * 4);", "s->cL = malloc(s->ccells); s->cT = malloc(s->ccells * 4); s->cg = malloc(s->ccells * 8); s->cga = malloc(s->ccells * 8);")
SRC = rep(SRC, "free(s->cL); free(s->cT);", "free(s->cL); free(s->cT); free(s->cg); free(s->cga);")
SRC = rep(SRC, "for (int k = 0; k < K; k++) for (int l = 0; l < NL; l++) s->T[k * NL + l] = kborn[k];", "for (int k = 0; k < K; k++) for (int l = 0; l < NL; l++) s->T[k * NL + l] = kborn[k];\n    for (int k = 0; k < K; k++) s->kb[k] = kborn[k];")
SRC = rep(SRC, "const uint32_t *pc = s->pc + s->cptr[c]; const double tk = s->T[k * NL + lev[k]];", "const uint32_t *pc = s->pc + s->cptr[c]; const double tk = s->own ? s->cg[c] : s->T[k * NL + lev[k]];")
SRC = rep(SRC, "const int64_t i = k * NL + lev[k]; s->Ta[i] += g * g; if (s->Ta[i] > s->gmin || s->rule != 2) s->T[i] += step_of(s, s->Tu[i], s->Ta[i], sc) * g; s->Tu[i] += 1.0;",
          "const int64_t i = k * NL + lev[k];\n                if (s->own && s->ktype[k] < 2) { if (s->own == 1) { const int32_t c = s->hcell[sl[k]]; s->cga[c] += g * g; s->cg[c] += s->eta * g / sqrt(1.0 + s->cga[c]); } }\n                else { s->Ta[i] += g * g; if (s->Ta[i] > s->gmin || s->rule != 2) s->T[i] += step_of(s, s->Tu[i], s->Ta[i], sc) * g; s->Tu[i] += 1.0; }")
SRC = rep(SRC, "c = new_cell(s); s->hkey[sl[k]] = keys[k]; s->hkind[sl[k]] = (uint16_t)k; s->hcell[sl[k]] = c;", "c = new_cell(s); s->hkey[sl[k]] = keys[k]; s->hkind[sl[k]] = (uint16_t)k; s->hcell[sl[k]] = c; s->cg[c] = s->kb[k]; s->cga[c] = 0;")
SRC = rep(SRC, "void sy_gmin(Sy *s, double g) { s->gmin = g; }", "void sy_gmin(Sy *s, double g) { s->gmin = g; }\nvoid sy_own(Sy *s, int own) { s->own = own; }")

def kern():
    h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_own_{h}.so"); src = os.path.join(HERE, "kern_own.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd, cl, vp = ctypes.c_int, ctypes.c_double, ctypes.c_int64, ctypes.c_void_p
    L.sy_new.argtypes = [ci, ci, PI, PI, PI, PD, cd, cd, ci, ci, ci, cd, ci, cl, ci, cd]; L.sy_new.restype = vp; L.sy_free.argtypes = [vp]; L.sy_free.restype = None
    L.sy_run.argtypes = [vp, PI, cl, cl, ci, PD, PD, PD, PD]; L.sy_run.restype = None
    for f in ("sy_cells", "sy_counts", "sy_once", "sy_gsize", "sy_now"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = cl
    for f in ("sy_T", "sy_Tu", "sy_Ta", "sy_G", "sy_Gu", "sy_Ga"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = PD
    L.sy_gmin.argtypes = [vp, cd]; L.sy_gmin.restype = None; L.sy_own.argtypes = [vp, ci]; L.sy_own.restype = None; return L

class SysO(LV.Sys):
    def __init__(s, kinds, x, own=0, S=256, eta=0.00005, e=0.02, rule=0, gate=0, J=6, birth=0.0, M=8, age=0, spow=0.0, guard=1e-20):
        s.S, s.kinds, s.K, s.gate, s.J, s.M = S, kinds, len(kinds), gate, J, M; assert S <= 256 and s.K <= 600 and J <= 8 and not (own and gate); s.L = kern(); s.x = np.ascontiguousarray(x.astype(np.int32)); assert 0 <= s.x.min() and s.x.max() < S
        kt = np.array([k[0] for k in kinds], np.int32); kn = np.array([k[1] for k in kinds], np.int32); kd = np.zeros((s.K, 8), np.int32); kb = np.array([k[3] for k in kinds], np.float64)
        for i, k in enumerate(kinds): kd[i, :len(k[2])] = k[2]
        s.h = s.L.sy_new(S, s.K, p32(kt), p32(kn), p32(np.ascontiguousarray(kd)), pd(kb), eta, e, rule, gate, J, birth, M if any(k[0] == 2 for k in kinds) else 0, len(s.x), age, spow); s.L.sy_gmin(s.h, guard); s.L.sy_own(s.h, own)

FORMS = dict(S2=dict(own=0, gate=1, rule=2), S1=dict(own=0, gate=0, rule=2), P=dict(own=1, gate=0), F=dict(own=2, gate=0))
def make(form, x, alpha=0.03, spec="sums", born=0.02, e=2.0):
    kinds = SU.kinds_of(spec, 256, 0, born); return SysO(kinds, x, e=e, eta=alpha, **FORMS[form])

# ---------------------------------------------------------------- the reference in plain Python (kinds of cells on sets of places only; a trust for each cell, or shared by kind and level)
def ref_run(S, kinds, x, lo, hi, state=None, learn=True, own=0, eta=0.03, e=2.0, rule=2, guard=1e-20):
    K = len(kinds)
    if state is None: state = dict(cells={}, cg={}, cga={}, T=np.array([[k[3]] * NL for k in kinds], np.float64), Tu=np.zeros((K, NL)), Ta=np.zeros((K, NL)))
    cells, cg, cga, T, Tu, Ta = (state[k] for k in ("cells", "cg", "cga", "T", "Tu", "Ta")); P1 = np.zeros((hi - lo, S)); xb = lambda q: int(x[q]) if q >= 0 else 0
    for t in range(lo, hi):
        y = int(x[t]); a = np.zeros(S); act = []; keys = []
        for k, (typ, nd, ds, _, _) in enumerate(kinds):
            assert typ == 0; key = (k,) + tuple(xb(t - d) for d in ds); keys.append(key); c = cells.get(key)
            if c is not None and c.sum() >= 1:
                l = SU.level_np(c.sum(), int((c > 0).sum())); dv = np.log(c + e) - np.log(e); act.append((k, l, key, dv)); a += (cg[key] if own else T[k, l]) * dv
        p = np.exp(a - a.max()); p /= p.sum(); P1[t - lo] = p
        if learn:
            for k, l, key, dv in act:
                g = dv[y] - p @ dv
                if own == 1: cga[key] += g * g; cg[key] += eta * g / np.sqrt(1.0 + cga[key])
                elif own == 0:
                    Ta[k, l] += g * g
                    if rule == 0: T[k, l] += eta * g
                    elif Ta[k, l] > guard: T[k, l] += eta * g / np.sqrt(Ta[k, l])
                    Tu[k, l] += 1
            for k, key in enumerate(keys):
                if key not in cells: cells[key] = np.zeros(S); cg[key] = kinds[k][3]; cga[key] = 0.0
                cells[key][y] += 1
    return P1, state

# ---------------------------------------------------------------- the life: sums, English, sums again
def text(a, b): return list(f"{a:02d}+{b:02d}={a + b:03d}\n".encode())
def build(seed, lines, back=5000, english=400000, D=9):
    """The stream of nano/suite/suite.py's sums (the same split of pairs and the same draws), in the bytes of the text itself; then English; then sums again; then the 500 never-shown lines, each after a read line (five draws)."""
    rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); pairs = [(int(q) // 100, int(q) % 100) for q in o]; held = pairs[:500]; shown = pairs[2000:]; rng.permutation(10000)
    TAB = np.array([text(a, b) for a, b in shown], np.int64); A = np.concatenate([TAB[rng.integers(8000, size=5000)] for _ in range(lines // 5000)]).ravel(); A2 = TAB[np.random.default_rng([seed, 55]).integers(8000, size=back)].ravel()
    Ls = np.array([text(a, b) for a, b in held]); g = np.random.default_rng([seed, 7]); prevs = [TAB[g.integers(8000, size=500)] for _ in range(5)]; ev = np.concatenate([np.hstack([pv, Ls]).ravel() for pv in prevs]); B = LV.raw("shakespeare")[0][:english]
    x = np.concatenate([np.full(D, 10), A, B, A2, ev]).astype(np.int32); a0 = D; b0 = a0 + len(A); c0 = b0 + len(B); e0 = c0 + len(A2); return x, dict(A=(a0, b0), B=(b0, c0), A2=(c0, e0), ev=e0)

def never_shown(s, base, cells=(1,)):
    """500 never-shown sums, each after a read line, five draws: the share whose whole answer the system names in its own bytes, with nothing changing. Changes s.x and the system's clock: call it in a forked copy."""
    truth = s.x[base:].copy().reshape(5, 500, 20); r = {}
    for cell in cells:
        s.x[base:] = truth.ravel(); right = np.zeros((5, 500, 3), bool)
        for d in range(5):
            for i in range(500):
                p0 = base + (d * 500 + i) * 20 + 10
                for j in (6, 7, 8): out = s.run(p0 + j, p0 + j + 1, False, True); v = int(out[1 + cell][0].argmax()); right[d, i, j - 6] = v == truth[d, i, 10 + j]; s.x[p0 + j] = v
        r[f"cell{cell}"] = dict(whole=float(100 * right.all(2).mean()), place=(100 * right.mean((0, 1))).tolist())
    return r

def run_life(kw):
    seed = int(kw.get("seed", 1)); form = kw["form"]; lines = int(kw.get("lines", 40000)); alpha = float(kw.get("alpha", 0.03)); tag = kw["tag"]; x, sp = build(seed, lines); s = make(form, x, alpha); cells = (1, 2) if s.gate else (1,)
    tmp = os.path.join(HERE, "tmp", tag); os.makedirs(tmp, exist_ok=True); t0 = time.time(); rows = []; eng = LV.raw("shakespeare"); probe = None
    def mark(label):
        p = os.path.join(tmp, label.replace(" ", "_") + ".json"); LV.wait(LV.forked(lambda: dict(never_shown(s, sp["ev"], cells), sizes=s.sizes()), p)); r = json.load(open(p)); r["at"] = label; rows.append(r); print(tag, label, {c: round(r[c]["whole"], 1) for c in r if c.startswith("cell")}, "cells", r["sizes"]["cells"], flush=True)
    a = s.run(sp["A"][0], sp["A"][1], True); mark("end of the sums"); nA = m(a[1] if s.gate else a[0][-50000:])
    b = s.run(sp["B"][0], sp["B"][1], True); mark("end of the English"); k = (sp["B"][1] - sp["B"][0]) // 10; nB = [m((b[1] if s.gate else b[0])[:k]), m((b[1] if s.gate else b[0])[-k:])]
    done = sp["A2"][0]
    for n in (500, 1000, 2000, 5000): hi = sp["A2"][0] + n * 10; s.run(done, hi, True); done = hi; mark(f"return {n} lines")
    out = dict(task="aba", tag=tag, seed=seed, form=form, alpha=alpha, lines=lines, rows=rows, nats_a_byte_english_first_and_last_tenth=nB, wall=round(time.time() - t0, 1)); json.dump(out, open(os.path.join(HERE, "res", f"aba__{tag}.json"), "w")); return out

def check():
    ok = True; rng = np.random.default_rng(29); xs, sp = build(1, 5000); eng = LV.raw("shakespeare")[0]
    for name, x, spec, own, extra in (("sums", xs[:2500], "sums", 1, dict(eta=0.1)), ("sums", xs[:2500], "sums", 2, {}), ("English", eng[:2000], "sums", 1, dict(eta=0.03)), ("English", eng[:2500], "runs", 1, dict(eta=0.3, e=0.02)), ("sums", xs[:2500], "sums", 0, dict(eta=0.03))):
        n = len(x); kinds = SU.kinds_of(spec, 256, 0, 0.02); kw = dict(e=2.0, eta=0.03); kw.update(extra); cutn = n - 500
        s = SysO(kinds, x, own=own, rule=2, **kw); a = s.run(9, cutn, True, True); f = s.run(cutn, n, False, True); g = s.run(cutn, n, True, True)
        R, st = ref_run(256, kinds, x, 9, cutn, None, True, own, **kw); Q, _ = ref_run(256, kinds, x, cutn, n, st, False, own, **kw); W, _ = ref_run(256, kinds, x, cutn, n, st, True, own, **kw)
        d = max(np.abs(a[2] - R).max(), np.abs(f[2] - Q).max(), np.abs(g[2] - W).max()); cells_ref = sum(1 for c in st["cells"].values() if c.sum() > 0); sz = s.sizes(); moved = sum(1 for v in st["cg"].values() if abs(v - 0.02) > 1e-3)
        x2 = x.copy(); n0 = cutn - 300; x2[n0:] = (x2[n0:] + 1 + rng.integers(0, 255, n - n0)) % 256; sa = SysO(kinds, x, own=own, rule=2, **kw); sb = SysO(kinds, x2, own=own, rule=2, **kw); u = sa.run(9, n0 + 1, True); v = sb.run(9, n0 + 1, True); blind = bool(np.array_equal(u[0][:-1], v[0][:-1]))
        tol = 1e-8 if own else 1e-4; good = d < tol and blind and sz["cells"] == cells_ref and (own != 1 or moved > 100) and (own != 2 or moved == 0); ok &= good
        print(f"{name} ({n} bytes), kinds {spec} ({len(kinds)}), own = {own} ({'a trust for each cell' if own == 1 else 'trusts never move' if own == 2 else 'trusts shared by kind and level'}), options {kw}: compiled against the reference, largest difference {d:.1e} (answers while reading, with nothing changing, going on); cells {sz['cells']} = {cells_ref}; cells whose own trust has moved {moved}; no answer moves when its own byte and all later ones are replaced: {blind}", "PASSED" if good else "FAILED", flush=True)
    kinds = SU.kinds_of("sums", 256, 0, 0.02); s = SysO(kinds, xs[:20000], own=0, e=2.0, eta=0.03, rule=2, gate=1); a = s.run(9, 20000, True); p = LV.Sys(kinds, xs[:20000], e=2.0, eta=0.03, rule=2, gate=1); b = p.run(9, 20000, True); dl = max(np.abs(a[0] - b[0]).max(), np.abs(a[1] - b[1]).max()); same = dl < 1e-6; ok &= same
    print(f"with own = 0 the losses of both answering cells differ from nano/lives/lives.py's kernel by at most {dl:.1e} (compiled separately): {same}", "PASSED" if same else "FAILED", flush=True)
    x, sp2 = build(1, 5000, back=500, english=3000); s = make("S1", x); s.run(sp2["A"][0], sp2["A"][1], True); p = os.path.join(HERE, "tmp"); os.makedirs(p, exist_ok=True); LV.wait(LV.forked(lambda: never_shown(s, sp2["ev"]), os.path.join(p, "check_aba.json"))); r1 = json.load(open(os.path.join(p, "check_aba.json")))
    LV.wait(LV.forked(lambda: never_shown(s, sp2["ev"]), os.path.join(p, "check_aba2.json"))); r2 = json.load(open(os.path.join(p, "check_aba2.json"))); same = r1 == r2 and 0 < r1["cell1"]["whole"] < 100; ok &= same
    print(f"the score of never-shown sums, made twice in forked copies of one system after 5,000 lines, is the same and leaves the system as it was ({r1['cell1']['whole']:.1f}% whole answers; by place {[round(v, 1) for v in r1['cell1']['place']]}): {same}", "PASSED" if same else "FAILED", flush=True)
    return ok

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=", 1) for a in sys.argv[2:]); out = run_life(kw)
    print(out["tag"], "| never-shown sums, whole answer:", " | ".join(f"{r['at']}: " + "/".join(f"{r[c]['whole']:.1f}" for c in r if c.startswith("cell")) for r in out["rows"]), "| English, nats a byte, first and last tenth:", [round(v, 3) for v in out["nats_a_byte_english_first_and_last_tenth"]], "| wall", out["wall"], flush=True)
