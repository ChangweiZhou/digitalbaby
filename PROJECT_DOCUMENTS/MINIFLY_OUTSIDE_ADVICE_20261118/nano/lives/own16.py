#!/usr/bin/env python3
"""nano/lives, PLAN_16.md: cells that count, every cell with a trust of its own and a step of its own that the cell learns from its own record.
    python3 own16.py check
    python3 own16.py life seed=1 form=A [mu=0.03] [alpha0=0.03] [tau=10000] [lines=40000] tag=..
Form A (own = 3). No number is shared between cells. Each cell holds its trust, its step, a trace of its own recent changes, and a yardstick
for the size of (driver x trace). When the cell speaks and the byte arrives, with g its driver (its message at the byte that came less the
expected message):
    yardstick: v = max(|g h|, v + (step g^2 / tau) (|g h| - v))
    step:      step = step x exp(mu g h / v)          (it grows while the cell's changes keep going one way, and falls while they cancel)
    the answering cell: if the sum over the speaking cells of step g^2 is above one, every speaking cell's step is divided by that sum
                        (form A: the cell keeps the divided step; form B, own = 4: the division holds for this byte only)
    trust:     trust = trust + step g
    trace:     h = h x max(0, 1 - step g^2) + step g
This is the rule published as Autostep (Mahmood, Sutton and others, 2012; from my memory of it, not checked against the paper), with the
squared driver in the place of the squared input. The life and the score are aba.py's (PLAN_11.md); with own = 0, 1 or 2 the kernel is aba.py's."""
import ctypes, hashlib, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import aba as AB, lives as LV
SU = LV.SU; NL = LV.NL; PD = LV.PD; PI = LV.PI; rep = LV.rep; p32 = LV.p32; pd = LV.pd

SRC = AB.SRC
SRC = rep(SRC, "double *cg, *cga; int own; double kb[KMAX];", "double *cg, *cga; int own; double kb[KMAX]; double *ca, *ch, *cv; double mu, tau, alpha0;      /* PLAN_16: a step, a trace and a yardstick for every cell */ ")
SRC = rep(SRC, "s->cg = realloc(s->cg, s->ccells * 8); s->cga = realloc(s->cga, s->ccells * 8); }", "s->cg = realloc(s->cg, s->ccells * 8); s->cga = realloc(s->cga, s->ccells * 8); s->ca = realloc(s->ca, s->ccells * 8); s->ch = realloc(s->ch, s->ccells * 8); s->cv = realloc(s->cv, s->ccells * 8); }")
SRC = rep(SRC, "s->cg = malloc(s->ccells * 8); s->cga = malloc(s->ccells * 8);", "s->cg = malloc(s->ccells * 8); s->cga = malloc(s->ccells * 8); s->ca = malloc(s->ccells * 8); s->ch = malloc(s->ccells * 8); s->cv = malloc(s->ccells * 8);")
SRC = rep(SRC, "free(s->cg); free(s->cga);", "free(s->cg); free(s->cga); free(s->ca); free(s->ch); free(s->cv);")
SRC = rep(SRC, "s->cg[c] = s->kb[k]; s->cga[c] = 0;", "s->cg[c] = s->kb[k]; s->cga[c] = 0; s->ca[c] = s->alpha0; s->ch[c] = 0; s->cv[c] = 0;")
SRC = rep(SRC, "const double sc = s->spow > 0 && nact > 0 ? pow(12.0 / nact, s->spow) : 1.0;", """const double sc = s->spow > 0 && nact > 0 ? pow(12.0 / nact, s->spow) : 1.0;
            if (s->own >= 3) {                          /* PLAN_16: every cell's own step, learned from its own record; the loop below then changes no trust */
                static double gk[KMAX]; double tot = 0;
                for (int k = 0; k < K; k++) if (lev[k] >= 0 && s->ktype[k] < 2) {
                    double g = 0; for (int i = 0; i < nv[k]; i++) { const int v = vv[k][i]; g += ((v == y) - p[v]) * dv[k][i]; }
                    gk[k] = g; const int32_t c = s->hcell[sl[k]]; const double gh = g * s->ch[c], ag = fabs(gh);
                    double v = s->cv[c] + (s->ca[c] * g * g / s->tau) * (ag - s->cv[c]); if (ag > v) v = ag; s->cv[c] = v;
                    if (v > 0) s->ca[c] *= exp(s->mu * gh / v);
                    tot += s->ca[c] * g * g;
                }
                const double M = tot > 1.0 ? tot : 1.0;
                for (int k = 0; k < K; k++) if (lev[k] >= 0 && s->ktype[k] < 2) {
                    const int32_t c = s->hcell[sl[k]]; const double g = gk[k], al = s->ca[c] / M; if (s->own == 3) s->ca[c] = al;      /* own = 4: the division holds for this byte only */
                    s->cg[c] += al * g; double xx = 1.0 - al * g * g; if (xx < 0) xx = 0; s->ch[c] = s->ch[c] * xx + al * g;
                }
            }""")
SRC = rep(SRC, "void sy_own(Sy *s, int own) { s->own = own; }", "void sy_own(Sy *s, int own) { s->own = own; }\nvoid sy_own3(Sy *s, double mu, double tau, double alpha0) { s->mu = mu; s->tau = tau; s->alpha0 = alpha0; }")

def kern():
    h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_own16_{h}.so"); src = os.path.join(HERE, "kern_own16.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd, cl, vp = ctypes.c_int, ctypes.c_double, ctypes.c_int64, ctypes.c_void_p
    L.sy_new.argtypes = [ci, ci, PI, PI, PI, PD, cd, cd, ci, ci, ci, cd, ci, cl, ci, cd]; L.sy_new.restype = vp; L.sy_free.argtypes = [vp]; L.sy_free.restype = None
    L.sy_run.argtypes = [vp, PI, cl, cl, ci, PD, PD, PD, PD]; L.sy_run.restype = None
    for f in ("sy_cells", "sy_counts", "sy_once", "sy_gsize", "sy_now"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = cl
    for f in ("sy_T", "sy_Tu", "sy_Ta", "sy_G", "sy_Gu", "sy_Ga"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = PD
    L.sy_gmin.argtypes = [vp, cd]; L.sy_gmin.restype = None; L.sy_own.argtypes = [vp, ci]; L.sy_own.restype = None; L.sy_own3.argtypes = [vp, cd, cd, cd]; L.sy_own3.restype = None; return L

class Sys16(LV.Sys):
    def __init__(s, kinds, x, own=0, S=256, eta=0.00005, e=0.02, rule=0, gate=0, J=6, birth=0.0, M=8, age=0, spow=0.0, guard=1e-20, mu=0.03, tau=1e4, alpha0=0.03):
        s.S, s.kinds, s.K, s.gate, s.J, s.M = S, kinds, len(kinds), gate, J, M; assert S <= 256 and s.K <= 600 and J <= 8 and not (own and gate); s.L = kern(); s.x = np.ascontiguousarray(x.astype(np.int32)); assert 0 <= s.x.min() and s.x.max() < S
        kt = np.array([k[0] for k in kinds], np.int32); kn = np.array([k[1] for k in kinds], np.int32); kd = np.zeros((s.K, 8), np.int32); kb = np.array([k[3] for k in kinds], np.float64)
        for i, k in enumerate(kinds): kd[i, :len(k[2])] = k[2]
        s.h = s.L.sy_new(S, s.K, p32(kt), p32(kn), p32(np.ascontiguousarray(kd)), pd(kb), eta, e, rule, gate, J, birth, M if any(k[0] == 2 for k in kinds) else 0, len(s.x), age, spow); s.L.sy_gmin(s.h, guard); s.L.sy_own(s.h, own); s.L.sy_own3(s.h, mu, tau, alpha0)

SET = dict(mu=0.03, tau=1e4, alpha0=0.03)
def make16(form, x, alpha=0.03, spec="sums", born=0.02, e=2.0):
    kinds = SU.kinds_of(spec, 256, 0, born)
    if form in ("A", "B"): return Sys16(kinds, x, own=3 if form == "A" else 4, e=e, eta=alpha, gate=0, **SET)
    return Sys16(kinds, x, e=e, eta=alpha, **AB.FORMS[form])

def ref16(S, kinds, x, lo, hi, state=None, learn=True, e=2.0, mu=0.03, tau=1e4, alpha0=0.03, own=3):
    """The reference in plain Python for own = 3 and 4 (kinds on sets of places only)."""
    if state is None: state = dict(cells={}, cg={}, ca={}, ch={}, cv={})
    cells, cg, ca, ch, cv = (state[k] for k in ("cells", "cg", "ca", "ch", "cv")); P1 = np.zeros((hi - lo, S)); xb = lambda q: int(x[q]) if q >= 0 else 0
    for t in range(lo, hi):
        y = int(x[t]); a = np.zeros(S); act = []; keys = []
        for k, (typ, nd, ds, _, _) in enumerate(kinds):
            assert typ == 0; key = (k,) + tuple(xb(t - d) for d in ds); keys.append(key); c = cells.get(key)
            if c is not None and c.sum() >= 1: dv = np.log(c + e) - np.log(e); act.append((key, dv)); a += cg[key] * dv
        p = np.exp(a - a.max()); p /= p.sum(); P1[t - lo] = p
        if learn:
            gs = []; tot = 0.0
            for key, dv in act:
                g = dv[y] - p @ dv; gs.append(g); gh = g * ch[key]; ag = abs(gh); v = cv[key] + (ca[key] * g * g / tau) * (ag - cv[key]); v = max(v, ag); cv[key] = v
                if v > 0: ca[key] *= np.exp(mu * gh / v)
                tot += ca[key] * g * g
            M = max(tot, 1.0)
            for (key, dv), g in zip(act, gs):
                al = ca[key] / M
                if own == 3: ca[key] = al
                cg[key] += al * g; ch[key] = ch[key] * max(0.0, 1 - al * g * g) + al * g
            for k, key in enumerate(keys):
                if key not in cells: cells[key] = np.zeros(S); cg[key] = kinds[k][3]; ca[key] = alpha0; ch[key] = 0.0; cv[key] = 0.0
                cells[key][y] += 1
    return P1, state

def check():
    ok = True; rng = np.random.default_rng(31); xs, sp = AB.build(1, 5000); eng = LV.raw("shakespeare")[0]
    for own, name, x, spec, kw in ((3, "sums", xs[:2500], "sums", dict(mu=0.03)), (3, "sums", xs[:2500], "sums", dict(mu=0.3, alpha0=0.1)), (3, "English", eng[:2000], "sums", dict(mu=0.03)), (3, "English", eng[:2500], "runs", dict(mu=0.1, tau=100.0, e=0.02)),
                                   (4, "sums", xs[:2500], "sums", dict(mu=0.03)), (4, "sums", xs[:2500], "sums", dict(mu=0.3, alpha0=0.1)), (4, "English", eng[:2000], "sums", dict(mu=0.1)), (4, "English", eng[:2500], "runs", dict(mu=0.1, tau=100.0, e=0.02))):
        n = len(x); kinds = SU.kinds_of(spec, 256, 0, 0.02); k2 = dict(e=2.0, mu=0.03, tau=1e4, alpha0=0.03); k2.update(kw); cutn = n - 500; e = k2.pop("e")
        s = Sys16(kinds, x, own=own, e=e, **k2); a = s.run(9, cutn, True, True); f = s.run(cutn, n, False, True); g = s.run(cutn, n, True, True)
        R, st = ref16(256, kinds, x, 9, cutn, None, True, e, own=own, **k2); Q, _ = ref16(256, kinds, x, cutn, n, st, False, e, own=own, **k2); W, _ = ref16(256, kinds, x, cutn, n, st, True, e, own=own, **k2)
        d = max(np.abs(a[2] - R).max(), np.abs(f[2] - Q).max(), np.abs(g[2] - W).max()); steps = np.array(list(st["ca"].values())); moved = int((np.abs(steps - k2["alpha0"]) > 1e-6).sum())
        x2 = x.copy(); n0 = cutn - 300; x2[n0:] = (x2[n0:] + 1 + rng.integers(0, 255, n - n0)) % 256; sa = Sys16(kinds, x, own=own, e=e, **k2); sb = Sys16(kinds, x2, own=own, e=e, **k2); u = sa.run(9, n0 + 1, True); v = sb.run(9, n0 + 1, True); blind = bool(np.array_equal(u[0][:-1], v[0][:-1]))
        good = d < 1e-8 and blind and moved > 100; ok &= good
        print(f"{name} ({n} bytes), kinds {spec} ({len(kinds)}), a step for each cell (form {'A' if own == 3 else 'B'}), options {dict(k2, e=e)}: compiled against the reference, largest difference {d:.1e} (answers while reading, with nothing changing, going on); cells whose step has moved {moved} (steps from {steps.min():.2e} to {steps.max():.2e}); no answer moves when its own byte and all later ones are replaced: {blind}", "PASSED" if good else "FAILED", flush=True)
    for own, form in ((1, "P"), (0, "S1"), (2, "F")):
        kinds = SU.kinds_of("sums", 256, 0, 0.02); kw = dict(AB.FORMS[form]); s = Sys16(kinds, xs[:20000], e=2.0, eta=0.03, **kw); a = s.run(9, 20000, True); p = AB.SysO(kinds, xs[:20000], e=2.0, eta=0.03, **kw); b = p.run(9, 20000, True); dl = float(np.abs(a[0] - b[0]).max()); same = dl < 1e-9; ok &= same
        print(f"form {form}: the losses differ from aba.py's kernel by at most {dl:.1e} (compiled separately): {same}", "PASSED" if same else "FAILED", flush=True)
    return ok

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=", 1) for a in sys.argv[2:])
    for k in SET:
        if k in kw: SET[k] = float(kw.pop(k))
    AB.make = make16; out = AB.run_life(kw); out.update(settings=dict(SET)); json.dump(out, open(os.path.join(HERE, "res", f"aba__{out['tag']}.json"), "w"))
    print(out["tag"], "| never-shown sums, whole answer:", " | ".join(f"{r['at']}: " + "/".join(f"{r[c]['whole']:.1f}" for c in r if c.startswith("cell")) for r in out["rows"]), "| English, nats a byte, first and last tenth:", [round(v, 3) for v in out["nats_a_byte_english_first_and_last_tenth"]], "| settings", SET, "| wall", out["wall"], flush=True)
