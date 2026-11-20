#!/usr/bin/env python3
"""nano/lives (PLAN_1.md): one system reads several byte streams of different kinds one after another. What does it keep?
    python3 lives.py check
    python3 lives.py life order=shakespeare,book,py,c,table,xml sys=ZA|ZPA tag=dev      (order=seed:73129 draws the order; one name = a single)
    python3 lives.py d2 life=<tag>                                                       (PLAN_1.md, diagnostic D2, from the trusts a life saved)
The kernel is nano/suite/suite.py's, changed by the replacements below and by nothing else (each replacement is asserted to apply exactly once):
  1. 256 values: a stream is its raw bytes; a context key holds 8 bits a byte.
  2. A timing kind (bytes since v last came) neither speaks nor counts until v has been met once.
  3. The system has its own time: the number of bytes it has read so far. Ages (of what a cell counted last, of a recalled place) and the timing clocks are
     differences of that time, not of places in the array, so that a stream which lies elsewhere in the array is read as what comes next. Read from the start
     of one array in one piece this is the same as before.
  4. The sums of squared drivers can be read and written.
  5. guard (1e-20 in every run here; 0 gives nano/suite's kernel): under the rule of the step, a trust does not change while the sum of its squared drivers is
     at or below the guard. In nano/suite it changes as soon as that sum is above zero, by the full number of the rule in the direction of the driver; where a
     first driver is a difference of nearly equal numbers, its direction is decided by the machine's rounding (found by this file's check: on raw bytes with the
     timing kinds the compiled kernel and the reference written out in full then part ways). With the guard such a driver moves nothing.
A measurement never changes the life: it is made in a forked copy of the process, which is thrown away."""
import ctypes, hashlib, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "suite"))
import suite as SU, make_streams as MS
NL = SU.NL; LMAX = SU.LMAX; PD = SU.PD; PI = SU.PI; LEAD = 4096; NAMES = MS.NAMES

def rep(t, a, b):
    assert t.count(a) == 1, (a[:70], t.count(a)); return t.replace(a, b)
SRC = SU.SRC
SRC = rep(SRC, "#define SMAX 128", "#define SMAX 256")
SRC = rep(SRC, "<< (7 * i); } }", "<< (8 * i); } }")
SRC = rep(SRC, "int64_t last[SMAX]; int64_t synced;", "int64_t last[SMAX]; int64_t synced; int64_t now; uint32_t *ptime; double gmin;      /* nano/lives: now = bytes read so far; last[] and cT hold times; ptime = the time at which a place was written */")
SRC = rep(SRC, "s->prev = malloc(sizeof(int32_t) * (nmax + 1));", "s->prev = malloc(sizeof(int32_t) * (nmax + 1)); s->ptime = malloc(sizeof(uint32_t) * (nmax + 1)); s->now = 0;")
SRC = rep(SRC, "free(s->head); free(s->prev); free(s);", "free(s->head); free(s->prev); free(s->ptime); free(s);")
SRC = rep(SRC, "    if (s->synced != lo) { for (int v = 0; v < SMAX; v++) s->last[v] = -1; for (int64_t t = 0; t < lo; t++) s->last[x[t]] = t; s->synced = lo; }\n", "")
SRC = rep(SRC, "const int y = x[t]; int64_t rq[8]; int nrq = -1;", "const int y = x[t]; int64_t rq[8]; int nrq = -1; const int64_t tau = s->now + (t - lo);")
SRC = rep(SRC, "agelevel_recall(L, M, t - q)", "agelevel_recall(L, M, tau - (int64_t)s->ptime[q])")
SRC = rep(SRC, "agelevel_last(s->cU[c], t - (int64_t)s->cT[c])", "agelevel_last(s->cU[c], tau - (int64_t)s->cT[c])")
SRC = rep(SRC, "else { const int v = s->knd[k]; const int64_t sn = s->last[v] < 0 ? 63 : (t - 1 - s->last[v] > 63 ? 63 : t - 1 - s->last[v]); key = (uint64_t)sn * 128 + (t >= 1 ? x[t - 1] : 0); }",
           "else { const int v = s->knd[k]; if (s->last[v] < 0) { sl[k] = -1; continue; } const int64_t sn = tau - 1 - s->last[v] > 63 ? 63 : tau - 1 - s->last[v]; key = (uint64_t)sn * 256 + (t >= 1 ? x[t - 1] : 0); }")
SRC = rep(SRC, "for (int k = 0; k < K; k++) if (s->ktype[k] < 2) {\n                int32_t c = s->hcell[sl[k]];", "for (int k = 0; k < K; k++) if (s->ktype[k] < 2 && sl[k] >= 0) {\n                int32_t c = s->hcell[sl[k]];")
SRC = rep(SRC, "{ grow(s); for (int j = 0; j < K; j++) if (s->ktype[j] < 2) sl[j] = slot_of(s, keys[j], j); }", "{ grow(s); for (int j = 0; j < K; j++) if (s->ktype[j] < 2 && sl[j] >= 0) sl[j] = slot_of(s, keys[j], j); }")
SRC = rep(SRC, "for (int j = k + 1; j < K; j++) if (s->ktype[j] < 2) sl[j] = slot_of(s, keys[j], j);", "for (int j = k + 1; j < K; j++) if (s->ktype[j] < 2 && sl[j] >= 0) sl[j] = slot_of(s, keys[j], j);")
SRC = rep(SRC, "count(s, c, y, t);", "count(s, c, y, tau);")
SRC = rep(SRC, "s->prev[t] = s->head[h]; s->head[h] = (int32_t)t; }", "s->prev[t] = s->head[h]; s->head[h] = (int32_t)t; s->ptime[t] = (uint32_t)tau; }")
SRC = rep(SRC, "        s->last[y] = t; s->synced = t + 1;\n    }\n}", "        s->last[y] = tau;\n    }\n    s->now += hi - lo;\n}")
SRC = rep(SRC, "if (s->Ta[i] > 0 || s->rule != 2)", "if (s->Ta[i] > s->gmin || s->rule != 2)")
SRC = rep(SRC, "if (s->Ga[i2] > 0 || s->rule != 2)", "if (s->Ga[i2] > s->gmin || s->rule != 2)")
SRC = rep(SRC, "double *sy_T(Sy *s) { return s->T; }", "void sy_gmin(Sy *s, double g) { s->gmin = g; }  double *sy_Ta(Sy *s) { return s->Ta; }  double *sy_Ga(Sy *s) { return s->Ga; }  int64_t sy_now(Sy *s) { return s->now; }  double *sy_T(Sy *s) { return s->T; }")

def kern():
    h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_lives_{h}.so"); src = os.path.join(HERE, "kern_lives.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd, cl, vp = ctypes.c_int, ctypes.c_double, ctypes.c_int64, ctypes.c_void_p
    L.sy_new.argtypes = [ci, ci, PI, PI, PI, PD, cd, cd, ci, ci, ci, cd, ci, cl, ci, cd]; L.sy_new.restype = vp; L.sy_free.argtypes = [vp]; L.sy_free.restype = None
    L.sy_run.argtypes = [vp, PI, cl, cl, ci, PD, PD, PD, PD]; L.sy_run.restype = None
    for f in ("sy_cells", "sy_counts", "sy_once", "sy_gsize", "sy_now"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = cl
    for f in ("sy_T", "sy_Tu", "sy_Ta", "sy_G", "sy_Gu", "sy_Ga"): getattr(L, f).argtypes = [vp]; getattr(L, f).restype = PD
    L.sy_gmin.argtypes = [vp, cd]; L.sy_gmin.restype = None; return L
p32 = SU.p32; pd = SU.pd

class Sys:
    def __init__(s, kinds, x, S=256, eta=0.00005, e=0.02, rule=0, gate=0, J=6, birth=0.0, M=8, age=0, spow=0.0, guard=1e-20):
        s.S, s.kinds, s.K, s.gate, s.J, s.M = S, kinds, len(kinds), gate, J, M; assert S <= 256 and s.K <= 600 and J <= 8; s.L = kern(); s.x = np.ascontiguousarray(x.astype(np.int32)); assert 0 <= s.x.min() and s.x.max() < S
        kt = np.array([k[0] for k in kinds], np.int32); kn = np.array([k[1] for k in kinds], np.int32); kd = np.zeros((s.K, 8), np.int32); kb = np.array([k[3] for k in kinds], np.float64)
        for i, k in enumerate(kinds): kd[i, :len(k[2])] = k[2]
        s.h = s.L.sy_new(S, s.K, p32(kt), p32(kn), p32(np.ascontiguousarray(kd)), pd(kb), eta, e, rule, gate, J, birth, M if any(k[0] == 2 for k in kinds) else 0, len(s.x), age, spow); s.L.sy_gmin(s.h, guard)
    def run(s, lo, hi, learn, want_p=False):
        n = hi - lo; nl1 = np.zeros(n); nl2 = np.zeros(n) if s.gate else None; P1 = np.zeros((n, s.S)) if want_p else None; P2 = np.zeros((n, s.S)) if want_p and s.gate else None
        s.L.sy_run(s.h, p32(s.x), lo, hi, int(learn), pd(nl1), pd(nl2), pd(P1), pd(P2)); return (nl1, nl2, P1, P2) if want_p else (nl1, nl2)
    def arr(s, name):
        """A view (not a copy) of one of the kernel's arrays of trusts: T, Tu, Ta (kinds x levels); G, Gu, Ga (the second cell's parts)."""
        if name[0] == "T": return np.ctypeslib.as_array(getattr(s.L, "sy_" + name)(s.h), (s.K, NL))
        n = s.L.sy_gsize(s.h); return np.ctypeslib.as_array(getattr(s.L, "sy_" + name)(s.h), (n,)) if n else np.zeros(0)
    def T(s): return s.arr("T").copy()
    def G(s): return s.arr("G").copy()
    def sizes(s): return dict(cells=int(s.L.sy_cells(s.h)), counts_above_zero=int(s.L.sy_counts(s.h)), cells_met_once=int(s.L.sy_once(s.h)), kinds=s.K, trusts_used=int((s.arr("Tu") > 0).sum()), parts_used=int((s.arr("Gu") > 0).sum()), bytes_read=int(s.L.sy_now(s.h)))

# ---------------------------------------------------------------- the reference in plain Python (every cell holds all S counts; the system's own time written out)
def ref_run(S, kinds, x, lo, hi, state=None, learn=True, eta=0.00005, e=0.02, rule=0, gate=0, J=6, birth=0.0, M=8, age=0, spow=0.0, guard=1e-20):
    K = len(kinds); GL = NL if gate == 2 else 1
    if state is None: state = dict(cells={}, T=np.array([[k[3]] * NL for k in kinds], np.float64), Tu=np.zeros((K, NL)), Ta=np.zeros((K, NL)), G=np.zeros((J, K, GL, S)), Gu=np.zeros((J, K, GL, S)), Ga=np.zeros((J, K, GL, S)), places={}, ptime={}, lasts={}, times={}, last={}, now=0)
    cells, T, Tu, Ta, G, Gu, Ga, places = (state[k] for k in ("cells", "T", "Tu", "Ta", "G", "Gu", "Ga", "places")); P1 = np.zeros((hi - lo, S)); P2 = np.zeros((hi - lo, S)); xb = lambda q: int(x[q]) if q >= 0 else 0
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
                if nd not in state["last"]: keys.append(None); continue          # a timing kind is silent until its byte has been met
                key = (k, min(63, tau - 1 - state["last"][nd]), xb(t - 1))
            keys.append(key); c = cells.get(key)
            if c is not None and c.sum() >= 1:
                n = c.sum(); l = SU.level_np(n, int((c > 0).sum())); sv = np.log(c + e) - np.log(n + S * e); act.append((k, l, sv, np.log(c + e) - np.log(e))); a += T[k, l] * sv
        p = np.exp(a - a.max()); p /= p.sum(); P1[t - lo] = p; p2 = p
        if gate:
            a2 = a.copy()
            for k, l, sv, dv in act: a2 += sum(G[j, k, l if GL > 1 else 0, xb(t - 1 - j)] for j in range(J)) * sv
            p2 = np.exp(a2 - a2.max()); p2 /= p2.sum(); P2[t - lo] = p2
        if learn:
            sc[0] = (12.0 / len(act)) ** spow if spow > 0 and act else 1.0
            for k, l, sv, dv in act:                                   # the driver as the kernel forms it: over the values the cell has met, the part common to all values left out
                g = dv[y] - p @ dv; Ta[k, l] += g * g
                if Ta[k, l] > guard or rule != 2: T[k, l] += step(Tu[k, l], Ta[k, l]) * g
                Tu[k, l] += 1
                if gate:
                    g2 = dv[y] - p2 @ dv
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
            if any(k[0] == 2 for k in kinds) and t >= M: places.setdefault(tuple(int(v) for v in x[t - M:t]), []).append(t); state["ptime"][t] = tau
        state["last"][y] = tau
    state["now"] += hi - lo; return P1, P2, state

# ---------------------------------------------------------------- streams as raw bytes, and the array of a life
_RAW = {}
def raw(name):
    """The raw bytes of one of nano/suite's streams (its hash checked against that line's manifest), and the cut."""
    if name not in _RAW:
        b = MS.raw_of(name)[0]; man = json.load(open(os.path.join(HERE, "..", "suite", "manifests", name + ".json"))); assert hashlib.sha256(b).hexdigest() == man["sha256_raw"], name
        _RAW[name] = (np.frombuffer(b, np.uint8).astype(np.int32), man["cut"])
    return _RAW[name]

def layout(order, tests=None):
    """The array: the bodies in the order of the life, one after another; then, for each stream, the last LEAD bytes of its body followed by its test (so that
    whatever the system looks back at from inside a test is that stream's own text). Returns the array and the places of the bodies and the tests."""
    tests = list(tests or NAMES); parts = []; pos = 0; body = {}; test = {}
    for n in order: x, cut = raw(n); parts.append(x[:cut]); body[n] = (pos, pos + cut); pos += cut
    for n in tests: x, cut = raw(n); parts += [x[cut - LEAD:cut], x[cut:]]; test[n] = (pos + LEAD, pos + LEAD + len(x) - cut); pos += LEAD + len(x) - cut
    return np.concatenate(parts), body, test

SYSTEMS = dict(ZA=dict(kinds="runs+last", recall=3, gate=1, rule=2, eta=0.03, age=1), ZPA=dict(kinds="runs+last+since+pool", recall=3, gate=1, rule=2, eta=0.03, age=1, spow=0.5),
               base=dict(kinds="runs", recall=0), Z=dict(kinds="runs+last", recall=3, gate=1, rule=2, eta=0.03))
def make(name, x, S=256, guard=1e-20):
    a = dict(SYSTEMS[name]); kinds = SU.kinds_of(a.pop("kinds"), S, a.pop("recall")); return Sys(kinds, x, S=S, guard=guard, **a)

m = lambda v: None if v is None else float(v.mean())
def forked(fn, path):
    """Runs fn() in a copy of this process and writes what it returns to path; the copy is thrown away. Returns the copy's process id."""
    sys.stdout.flush(); pid = os.fork()
    if pid == 0:
        code = 1
        try:
            r = fn()
            with open(path + ".tmp", "w") as f: json.dump(r, f)
            os.replace(path + ".tmp", path); code = 0
        finally: os._exit(code)
    return pid
def wait(pid):
    _, st = os.waitpid(pid, 0); assert st == 0, f"a forked measurement failed (status {st})"

def score(s, lo, hi, learn):
    t0 = time.process_time(); n1, n2 = s.run(lo, hi, learn); r = dict(cell1=m(n1), cell2=m(n2), first={str(k): [m(n1[:k]), m(n2[:k]) if n2 is not None else None] for k in (1000, 4000, 16000, 64000) if k <= hi - lo}, cpu=round(time.process_time() - t0, 1))
    if learn: r["sizes_after"] = s.sizes()
    return r

def run_life(kw):
    order = kw["order"]; name = kw.get("sys", "ZA"); tag = kw["tag"]
    if order.startswith("seed:"): order = [NAMES[i] for i in np.random.default_rng(int(order[5:])).permutation(len(NAMES))]
    else: order = order.split(",")
    tests = list(NAMES) if len(order) > 1 else list(order); x, body, test = layout(order, tests); s = make(name, x); tmp = os.path.join(HERE, "tmp", tag); os.makedirs(tmp, exist_ok=True); t0 = time.time(); stages = []; saved = {}; child = None
    def stage_scores(i):
        def fn():
            for n in tests:
                for learn in (0, 1): wait(forked(lambda: score(s, test[n][0], test[n][1], bool(learn)), os.path.join(tmp, f"s{i}_{n}_{learn}.json")))
            return dict(done=True)
        return fn
    for i, n in enumerate(order):
        lo, hi = body[n]; lo += 6 if i == 0 else 0; c0 = time.process_time(); n1, n2 = s.run(lo, hi, True); k = (hi - lo) // 10
        stages.append(dict(stream=n, reading=dict(cell1=m(n1), cell2=m(n2), first_tenth=[m(n1[:k]), m(n2[:k])], last_tenth=[m(n1[-k:]), m(n2[-k:])]), sizes=s.sizes(), cpu_reading=round(time.process_time() - c0, 1)))
        saved[n] = (s.T(), s.G())
        if child is not None: wait(child)                                  # at most one set of measurements beside the life
        child = forked(stage_scores(i), os.path.join(tmp, f"s{i}_done.json"))
        if int(kw.get("seq", 0)): wait(child); child = None                 # seq=1: the life waits for its measurements (less memory: no second copy of what the life changes meanwhile)
        print(tag, "read", n, {k_: round(v, 4) for k_, v in stages[-1]["reading"].items() if k_.startswith("cell")}, "cells", stages[-1]["sizes"]["cells"], "cpu", stages[-1]["cpu_reading"], flush=True)
    if child is not None: wait(child)
    np.savez(os.path.join(tmp, "trusts_end.npz"), T=s.T(), G=s.G(), **{f"T_{n}": saved[n][0] for n in order})
    for i, st in enumerate(stages): st["tests"] = {n: dict(nothing_changing=json.load(open(os.path.join(tmp, f"s{i}_{n}_0.json"))), going_on=json.load(open(os.path.join(tmp, f"s{i}_{n}_1.json")))) for n in tests}
    d1 = {}
    if len(order) > 1:                                                     # D1: the system of the end with the trusts and parts of a stream's own stage put back
        for n in order[:-1]:
            def fn(n=n):
                s.arr("T")[:] = saved[n][0]; s.arr("G")[:] = saved[n][1]; return score(s, test[n][0], test[n][1], False)
            wait(forked(fn, os.path.join(tmp, f"d1_{n}.json"))); d1[n] = json.load(open(os.path.join(tmp, f"d1_{n}.json")))
    out = dict(task="life", tag=tag, sys=name, args=SYSTEMS[name], order=order, stages=stages, d1=d1, length=int(len(x)), wall=round(time.time() - t0, 1)); json.dump(out, open(os.path.join(HERE, "res", f"life__{tag}.json"), "w")); return out

def run_d2(kw):
    """D2: for every stream of a life but the last read, a system that reads that stream's body alone, then is given the trusts and parts of the life's end; its test with nothing changing."""
    life = json.load(open(os.path.join(HERE, "res", f"life__{kw['life']}.json"))); z = np.load(os.path.join(HERE, "tmp", kw["life"], "trusts_end.npz")); out = {}
    for n in life["order"][:-1]:
        x, body, test = layout([n], [n]); s = make(life["sys"], x); s.run(body[n][0] + 6, body[n][1], True); p = os.path.join(HERE, "tmp", kw["life"], f"d2_{n}.json")
        wait(forked(lambda: score(s, test[n][0], test[n][1], False), p + ".own")); own = json.load(open(p + ".own"))
        def fn2():
            s.arr("T")[:] = z["T"]; s.arr("G")[:] = z["G"]; return score(s, test[n][0], test[n][1], False)
        wait(forked(fn2, p)); out[n] = dict(own_trusts=own, trusts_of_the_lifes_end=json.load(open(p))); print(kw["life"], "D2", n, "own trusts", round(own["cell1"], 4), round(own["cell2"], 4), "| the life's last trusts", round(out[n]["trusts_of_the_lifes_end"]["cell1"], 4), round(out[n]["trusts_of_the_lifes_end"]["cell2"], 4), flush=True)
    json.dump(out, open(os.path.join(HERE, "res", f"d2__{kw['life']}.json"), "w")); return out

# ---------------------------------------------------------------- the check
def check():
    ok = True; rng = np.random.default_rng(5)
    def stream(name, n):
        if name == "random": return rng.integers(0, 256, n).astype(np.int32)
        if name == "repeats": b = rng.integers(0, 256, 300); return np.concatenate([b[rng.integers(0, 250):][:rng.integers(5, 60)] for _ in range(200)])[:n].astype(np.int32)
        if name == "elf": return np.frombuffer(open(os.path.realpath(sys.executable), "rb").read()[4096:4096 + n], np.uint8).astype(np.int32)
        return raw(name)[0][:n]
    # 1. the compiled kernel against the reference written out in full, raw bytes, 256 values
    cases = [("shakespeare", "runs", dict()), ("py", "runs", dict(gate=1)), ("xml", "runs", dict(gate=2, birth=0.5)), ("c", "runs+since+pool", dict(gate=1)), ("table", "runs", dict(gate=1, recall=1)), ("py", "runs", dict(gate=1, recall=3, M=4)),
             ("book", "runs", dict(rule=2, eta=0.05, gate=1)), ("repeats", "runs", dict(gate=1, recall=3, M=8)), ("c", "runs+last", dict(gate=1, recall=1, rule=2, eta=0.03)), ("table", "runs+last", dict(gate=1, recall=3, rule=2, eta=0.03, age=1)),
             ("repeats", "runs+last", dict(gate=1, recall=3, age=1)), ("xml", "runs+last+since", dict(gate=1, recall=1, rule=2, eta=0.03, age=1, spow=1.0)), ("elf", "runs+last+since+pool", dict(gate=1, recall=3, rule=2, eta=0.03, age=1, spow=0.5, n=1500)),
             ("xml", "runs+last+since+pool", dict(gate=1, recall=3, rule=2, eta=0.03, age=1, spow=0.5, n=1500)), ("random", "runs+last+since", dict(gate=1, recall=3, rule=2, eta=0.03, age=1, n=1200)),
             ("xml", "runs+last+since+pool", dict(gate=1, recall=3, age=1, n=1500)), ("elf", "runs+last+since+pool", dict(gate=1, recall=3, rule=1, eta=0.1, age=1, n=1500)), ("table", "runs+last+since+pool", dict(gate=2, recall=3, rule=1, eta=0.1, age=1, birth=0.5, n=1500))]
    for name, spec, kw in cases:
        kw = dict(kw); n = kw.pop("n", 2500); lo = 6; recall = kw.pop("recall", 0); x = stream(name, n); n = len(x); S = 256; kinds = SU.kinds_of(spec, S, recall); cutn = n - 500
        s = Sys(kinds, x, S=S, **kw); a1, a2, A1, A2 = s.run(lo, cutn, True, True); f1, f2, F1, F2 = s.run(cutn, n, False, True); Tf = s.T(); g1, g2, G1, G2 = s.run(cutn, n, True, True)
        R1, R2, st = ref_run(S, kinds, x, lo, cutn, None, True, **kw); Q1, Q2, _ = ref_run(S, kinds, x, cutn, n, st, False, **kw); Tr = st["T"].copy(); W1, W2, _ = ref_run(S, kinds, x, cutn, n, st, True, **kw)
        d = max(np.abs(A1 - R1).max(), np.abs(F1 - Q1).max(), np.abs(G1 - W1).max(), np.abs(Tf - Tr).max(), np.abs(s.T() - st["T"]).max())
        if kw.get("gate"): d = max(d, np.abs(A2 - R2).max(), np.abs(F2 - Q2).max(), np.abs(G2 - W2).max(), np.abs(s.G() - st["G"].ravel()).max())
        cells_ref = sum(1 for c in st["cells"].values() if c.sum() > 0); counts_ref = sum(int((c > 0).sum()) for c in st["cells"].values()); sz = s.sizes(); silent = sum(1 for k in kinds if k[0] == 1 and k[1] not in st["last"])
        x2 = x.copy(); n0 = cutn - 300; x2[n0:] = (x2[n0:] + 1 + rng.integers(0, S - 1, n - n0)) % S; sa = Sys(kinds, x, S=S, **kw); sb = Sys(kinds, x2, S=S, **kw); _, _, U1, U2 = sa.run(lo, n0 + 1, True, True); _, _, V1, V2 = sb.run(lo, n0 + 1, True, True)
        blind = bool(np.array_equal(U1, V1) and (U2 is None or np.array_equal(U2, V2)))
        # Under the rule of the step a change is the rule's number times the driver over the root of the sum of its squares: where the drivers are themselves small
        # differences of nearly equal numbers, their relative rounding is carried into the trust whatever their size. With a dozen senders that stays under 1e-6
        # (nano/suite's bound); with some two hundred senders that speak (timing and pool kinds) under 1e-4. The same kinds under the other rules agree to 1e-9.
        tol = (1e-4 if "since" in spec else 1e-6) if kw.get("rule") == 2 else 1e-8          # 1e-8: the step 0.1 over the root of the uses, with some two hundred senders, carries rounding to 3e-9
        good = blind and d < tol and sz["cells"] == cells_ref and sz["counts_above_zero"] == counts_ref and sz["bytes_read"] == st["now"]; ok &= good
        print(f"{name} ({n} raw bytes, 256 values), kinds {spec} ({len(kinds)}; timing kinds still silent at the end: {silent}), options {kw}, recall {recall}: compiled against the reference, largest difference {d:.1e} (answers while reading, with nothing changing, going on; trusts; parts); cells {sz['cells']} = {cells_ref}, counts {sz['counts_above_zero']} = {counts_ref}; no answer moves when its own byte and all later ones are replaced: {blind}", "PASSED" if good else "FAILED", flush=True)
    # 2. on nano/suite's mapped streams, without timing kinds: the same answers bit for bit as that line's own kernel
    for name, spec, kw, recall in (("shakespeare", "runs+last", dict(gate=1, rule=2, eta=0.03, age=1), 3), ("table", "runs+last+pool", dict(gate=1, rule=2, eta=0.03, age=1, spow=0.5), 3), ("xml", "runs", dict(gate=2, birth=0.5), 0)):
        xm, cut, S = MS.load(name); n = 60000; xm = xm[:n]; cutn = n - 6000; kinds = SU.kinds_of(spec, S, recall)
        su = SU.Sys(S, kinds, xm, **kw); ua = su.run(6, cutn, True); uf = su.run(cutn, n, False); ug = su.run(cutn, n, True)
        s1 = Sys(kinds, xm, S=S, guard=0.0, **kw); la = s1.run(6, cutn, True); lf = s1.run(cutn, n, False); s2 = Sys(kinds, xm, S=S, guard=0.0, **kw); s2.run(6, cutn, True); lg = s2.run(cutn, n, True)
        same = all(np.array_equal(a, b) for u, l in ((ua, la), (uf, lf), (ug, lg)) for a, b in zip(u, l)); ok &= same
        print(f"{name} (mapped, {S} values, {n} bytes), kinds {spec}, options {kw}, recall {recall}: with the guard at zero, every loss while reading, with nothing changing and going on equal to nano/suite's kernel bit for bit: {same}", "PASSED" if same else "FAILED", flush=True)
    # 3. a test that lies elsewhere in the array, behind a copy of the end of its body, is read as if it followed its body
    for sysname in ("ZA", "ZPA"):
        x0 = raw("xml")[0][:66000]; cutn = 60000; xa = x0; xb_ = np.concatenate([x0[:cutn], np.arange(5000) % 256, x0[cutn - LEAD:cutn], x0[cutn:]]).astype(np.int32); tlo = cutn + 5000 + LEAD; res = []
        for xx, lo in ((xa, cutn), (xb_, tlo)):
            r = []
            for learn in (False, True): s = make(sysname, xx); s.run(6, cutn, True); r.append(s.run(lo, lo + 6000, learn))
            res.append(r)
        same = all(np.array_equal(a, b) for i in (0, 1) for a, b in zip(res[0][i], res[1][i])); ok &= same
        print(f"{sysname}, markup, 60,000 bytes read: the next 6,000 placed elsewhere in the array behind a copy of the last {LEAD} bytes read give the same losses bit for bit as when they follow directly (nothing changing and going on): {same}", "PASSED" if same else "FAILED", flush=True)
    # 4. a measurement in a forked copy leaves the system as it was
    x0 = raw("py")[0][:40000]; s = make("ZA", x0); s.run(6, 30000, True); p = os.path.join(HERE, "tmp"); os.makedirs(p, exist_ok=True); wait(forked(lambda: score(s, 30000, 35000, True), os.path.join(p, "check_fork.json"))); a = s.run(30000, 35000, True)
    s2 = make("ZA", x0); s2.run(6, 30000, True); b = s2.run(30000, 35000, True); fk = json.load(open(os.path.join(p, "check_fork.json"))); same = bool(np.array_equal(a[1], b[1]) and abs(fk["cell2"] - float(a[1].mean())) < 1e-12); ok &= same
    print(f"a going-on measurement made in a forked copy: the system afterwards answers as one that was never measured, and the copy's figure is the same ({fk['cell2']:.6f}): {same}", "PASSED" if same else "FAILED", flush=True)
    return ok

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=", 1) for a in sys.argv[2:])
    if sys.argv[1] == "d2": run_d2(kw); sys.exit(0)
    out = run_life(kw); r = lambda v: None if v is None else round(v, 4)
    for i, st in enumerate(out["stages"]):
        print(out["tag"], f"after {st['stream']}:", " | ".join(f"{n} {r(t['nothing_changing']['cell2'])} / {r(t['going_on']['cell2'])}" for n, t in st["tests"].items()), "(second cell: nothing changing / going on)", flush=True)
    if out["d1"]: print(out["tag"], "D1 (the end's counts with the trusts of the own stage), second cell, nothing changing:", {n: r(v["cell2"]) for n, v in out["d1"].items()}, "| wall", out["wall"], flush=True)
