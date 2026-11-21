#!/usr/bin/env python3
"""nano/lives, PLAN_2.md: situation cells. The system learns, without labels, where it is; each situation keeps a system of its own.
    python3 society.py check
    python3 society.py life order=shakespeare,book,py,c,table,xml sys=ZA W=1024 tag=..      (order=seed:73129 draws the order; one name = a single)
A situation cell counts the bytes read while it is awake (256 counts). The cells hold a belief (shares summing to one); before a byte is read the cell with the
largest share is awake; after the byte b each share becomes ((1 - 1/W) share + 1/(W x cells)) x (count of b + 1/2) / (counts + 128), and the shares are made to
sum to one. After every W bytes: if a cell that knew nothing at the start of the block and counted every byte of it would have been cheaper (sum of minus the log
of its opinions) than every existing cell, a new situation is born with the block's counts and the whole belief; the counts the others took from the block are
taken back. Each situation has a system of nano/lives/lives.py of its own; a byte is answered and counted by the system of the situation that is awake."""
import ctypes, hashlib, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lives as LV
NMAX = 64; PI = LV.PI; PD = LV.PD; PL = ctypes.POINTER(ctypes.c_int64)

SRC = r'''
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define NMAX 64
typedef struct {
    int N; int64_t W; double s;
    double h[NMAX][256], n[NMAX], b[NMAX], cost[NMAX];
    int64_t inblock; int32_t *wlog, *xlog;      /* the block so far: the cell that counted each byte (-1: none), and the byte */
} Det;
Det *det_new(int64_t W) { Det *d = calloc(1, sizeof(Det)); d->W = W; d->s = 1.0 / (double)W; d->N = 1; d->b[0] = 1.0; d->wlog = malloc(4 * W); d->xlog = malloc(4 * W); return d; }
void det_free(Det *d) { free(d->wlog); free(d->xlog); free(d); }
int det_n(Det *d) { return d->N; }
double *det_h(Det *d) { return &d->h[0][0]; }
double *det_b(Det *d) { return d->b; }
/* reads x[lo..hi). awake[t - lo]: the cell with the largest share before byte t is read. born_end[c]: the place just after the block that gave situation c its birth.
   learn = 0: no cell counts and none is born; the belief still moves. */
int det_run(Det *d, const int32_t *x, int64_t lo, int64_t hi, int learn, int32_t *awake, int64_t *born_end) {
    for (int64_t t = lo; t < hi; t++) {
        const int y = x[t]; int w = 0; double z = 0;
        for (int c = 1; c < d->N; c++) if (d->b[c] > d->b[w]) w = c;
        awake[t - lo] = w;
        for (int c = 0; c < d->N; c++) { const double p = (d->h[c][y] + 0.5) / (d->n[c] + 128.0); d->cost[c] -= log(p); d->b[c] = ((1.0 - d->s) * d->b[c] + d->s / d->N) * p; z += d->b[c]; }
        for (int c = 0; c < d->N; c++) d->b[c] /= z;
        if (learn) { d->h[w][y] += 1; d->n[w] += 1; }
        d->wlog[d->inblock] = learn ? w : -1; d->xlog[d->inblock] = y; d->inblock++;
        if (d->inblock == d->W) {
            double fresh = 0, best = 1e300, cnt[256]; memset(cnt, 0, sizeof(cnt));
            for (int64_t i = 0; i < d->W; i++) { const int v = d->xlog[i]; fresh -= log((cnt[v] + 0.5) / ((double)i + 128.0)); cnt[v] += 1; }
            for (int c = 0; c < d->N; c++) if (d->cost[c] < best) best = d->cost[c];
            if (learn && fresh < best && d->N < NMAX) {
                for (int64_t i = 0; i < d->W; i++) if (d->wlog[i] >= 0) { d->h[d->wlog[i]][d->xlog[i]] -= 1; d->n[d->wlog[i]] -= 1; }
                const int c = d->N++; for (int v = 0; v < 256; v++) d->h[c][v] = cnt[v]; d->n[c] = (double)d->W;
                for (int k = 0; k < d->N; k++) d->b[k] = 0; d->b[c] = 1.0; born_end[c] = t + 1;
            }
            memset(d->cost, 0, sizeof(d->cost)); d->inblock = 0;
        }
    }
    return d->N;
}
'''
def kern():
    h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_det_{h}.so"); src = os.path.join(HERE, "kern_det.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); vp, cl, ci = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    L.det_new.argtypes = [cl]; L.det_new.restype = vp; L.det_free.argtypes = [vp]; L.det_free.restype = None; L.det_n.argtypes = [vp]; L.det_n.restype = ci
    L.det_h.argtypes = [vp]; L.det_h.restype = PD; L.det_b.argtypes = [vp]; L.det_b.restype = PD; L.det_run.argtypes = [vp, PI, cl, cl, ci, PI, PL]; L.det_run.restype = ci; return L

class Det:
    def __init__(s, W): s.L = kern(); s.W = W; s.h = s.L.det_new(W); s.born_end = np.full(NMAX, -1, np.int64)
    def run(s, x, lo, hi, learn):
        awake = np.zeros(hi - lo, np.int32); s.L.det_run(s.h, LV.p32(x), lo, hi, int(learn), LV.p32(awake), s.born_end.ctypes.data_as(PL)); return awake
    @property
    def N(s): return int(s.L.det_n(s.h))
    def counts(s): return np.ctypeslib.as_array(s.L.det_h(s.h), (NMAX, 256))[:s.N].copy()
    def belief(s): return np.ctypeslib.as_array(s.L.det_b(s.h), (NMAX,))[:s.N].copy()

class DetRef:
    """The same written out in plain Python."""
    def __init__(s, W): s.W = W; s.h = [np.zeros(256)]; s.b = [1.0]; s.cost = [0.0]; s.log = []; s.born_end = {}
    def run(s, x, lo, hi, learn):
        awake = []
        for t in range(lo, hi):
            y = int(x[t]); N = len(s.h); w = max(range(N), key=lambda c: (s.b[c], -c)); awake.append(w)
            for c in range(N): p = (s.h[c][y] + 0.5) / (s.h[c].sum() + 128.0); s.cost[c] -= np.log(p); s.b[c] = ((1.0 - 1.0 / s.W) * s.b[c] + (1.0 / s.W) / N) * p
            z = sum(s.b); s.b = [v / z for v in s.b]
            if learn: s.h[w][y] += 1
            s.log.append((w if learn else -1, y))
            if len(s.log) == s.W:
                cnt = np.zeros(256); fresh = 0.0
                for i, (_, v) in enumerate(s.log): fresh -= np.log((cnt[v] + 0.5) / (i + 128.0)); cnt[v] += 1
                if learn and fresh < min(s.cost) and N < NMAX:
                    for w_, v in s.log:
                        if w_ >= 0: s.h[w_][v] -= 1
                    s.h.append(cnt); s.b = [0.0] * N + [1.0]; s.cost.append(0.0); s.born_end[N] = t + 1
                s.cost = [0.0] * len(s.h); s.log = []
        return np.array(awake, np.int32)

class Society:
    def __init__(s, sysname, x, W): s.x = np.ascontiguousarray(x.astype(np.int32)); s.sysname = sysname; s.W = W; s.det = Det(W); s.systems = [LV.make(sysname, s.x)]; s.gate = s.systems[0].gate
    def run(s, lo, hi, learn, want_awake=False):
        awake = s.det.run(s.x, lo, hi, learn); n = hi - lo; nl1 = np.zeros(n); nl2 = np.zeros(n); e = np.flatnonzero(np.diff(awake)) + 1
        for a, b in zip([0, *e], [*e, n]):
            c = int(awake[a])
            while len(s.systems) <= c:                                    # a system born with its situation first reads the block that gave it birth
                k = len(s.systems); sy = LV.make(s.sysname, s.x); end = int(s.det.born_end[k]); sy.run(max(end - s.W, 0), end, True); s.systems.append(sy)
            l1, l2 = s.systems[c].run(lo + a, lo + b, learn); nl1[a:b] = l1; nl2[a:b] = l2 if l2 is not None else l1
        return (nl1, nl2, awake) if want_awake else (nl1, nl2)
    def sizes(s):
        z = [sy.sizes() for sy in s.systems]; return dict(situations=len(z), cells=sum(v["cells"] for v in z), counts_above_zero=sum(v["counts_above_zero"] for v in z), cells_by_situation=[v["cells"] for v in z], bytes_by_situation=[v["bytes_read"] for v in z])

m = LV.m
def score(s, lo, hi, learn):
    t0 = time.process_time(); n1, n2, aw = s.run(lo, hi, learn, True); r = dict(cell1=m(n1), cell2=m(n2), first={str(k): [m(n1[:k]), m(n2[:k])] for k in (1000, 4000, 16000, 64000) if k <= hi - lo}, awake=np.bincount(aw, minlength=s.det.N).tolist(), switches=int((np.diff(aw) != 0).sum()), cpu=round(time.process_time() - t0, 1))
    if learn: r["sizes_after"] = s.sizes()
    return r

def run_life(kw):
    order = kw["order"]; name = kw.get("sys", "ZA"); tag = kw["tag"]; W = int(kw.get("W", 1024))
    if order.startswith("seed:"): order = [LV.NAMES[i] for i in np.random.default_rng(int(order[5:])).permutation(len(LV.NAMES))]
    else: order = order.split(",")
    tests = list(LV.NAMES) if len(order) > 1 else list(order); x, body, test = LV.layout(order, tests); s = Society(name, x, W); tmp = os.path.join(HERE, "tmp", tag); os.makedirs(tmp, exist_ok=True); t0 = time.time(); stages = []; child = None
    def stage_scores(i):
        def fn():
            for n in tests:
                for learn in (0, 1): LV.wait(LV.forked(lambda: score(s, test[n][0], test[n][1], bool(learn)), os.path.join(tmp, f"s{i}_{n}_{learn}.json")))
            return dict(done=True)
        return fn
    for i, n in enumerate(order):
        lo, hi = body[n]; lo += 6 if i == 0 else 0; c0 = time.process_time(); n1, n2, aw = s.run(lo, hi, True, True); k = (hi - lo) // 10
        stages.append(dict(stream=n, reading=dict(cell1=m(n1), cell2=m(n2), first_tenth=[m(n1[:k]), m(n2[:k])], last_tenth=[m(n1[-k:]), m(n2[-k:])]), awake=np.bincount(aw, minlength=s.det.N).tolist(), switches=int((np.diff(aw) != 0).sum()), sizes=s.sizes(), cpu_reading=round(time.process_time() - c0, 1)))
        if child is not None: LV.wait(child)
        child = LV.forked(stage_scores(i), os.path.join(tmp, f"s{i}_done.json")); print(tag, "read", n, {k_: round(v, 4) for k_, v in stages[-1]["reading"].items() if k_.startswith("cell")}, "situations", s.det.N, "awake while reading", stages[-1]["awake"], "cells", stages[-1]["sizes"]["cells"], "cpu", stages[-1]["cpu_reading"], flush=True)
    if child is not None: LV.wait(child)
    for i, st in enumerate(stages): st["tests"] = {n: dict(nothing_changing=json.load(open(os.path.join(tmp, f"s{i}_{n}_0.json"))), going_on=json.load(open(os.path.join(tmp, f"s{i}_{n}_1.json")))) for n in tests}
    out = dict(task="society", tag=tag, sys=name, W=W, order=order, stages=stages, length=int(len(x)), wall=round(time.time() - t0, 1)); json.dump(out, open(os.path.join(HERE, "res", f"society__{tag}.json"), "w")); return out

def check():
    ok = True; rng = np.random.default_rng(11)
    pieces = [LV.raw(n)[0][a:a + L] for n, a, L in (("shakespeare", 0, 3000), ("table", 5000, 2500), ("shakespeare", 9000, 2000), ("xml", 0, 3000), ("c", 0, 2500), ("table", 90000, 1500), ("py", 0, 2500), ("shakespeare", 30000, 3000))]; x = np.ascontiguousarray(np.concatenate(pieces).astype(np.int32)); n = len(x)
    for W in (256, 512):
        d = Det(W); a1 = d.run(x, 0, 9000, True); a2 = d.run(x, 9000, 14000, False); a3 = d.run(x, 14000, n, True); r = DetRef(W); b1 = r.run(x, 0, 9000, True); b2 = r.run(x, 9000, 14000, False); b3 = r.run(x, 14000, n, True)
        same = bool(np.array_equal(a1, b1) and np.array_equal(a2, b2) and np.array_equal(a3, b3) and d.N == len(r.h) and np.array_equal(d.counts(), np.array(r.h)) and np.abs(d.belief() - np.array(r.b)).max() < 1e-9 and all(int(d.born_end[c]) == e for c, e in r.born_end.items()))
        x2 = x.copy(); n0 = 12345; x2[n0:] = (x2[n0:] + 1 + rng.integers(0, 255, n - n0)) % 256; d2 = Det(W); c1 = np.concatenate([d2.run(x2, 0, 9000, True), d2.run(x2, 9000, n0 + 1, True)]); d3 = Det(W); c0 = np.concatenate([d3.run(x, 0, 9000, True), d3.run(x, 9000, n0 + 1, True)]); blind = bool(np.array_equal(c0, c1)); ok &= same and blind
        print(f"situation cells, W = {W}, {n} bytes of eight pieces from five streams (read counting, then 5,000 bytes with nothing changing, then counting): compiled against plain Python: awake cell at every byte, counts, belief, births the same: {same}; situations {d.N}, born at {sorted(r.born_end.values())}; the awake cell at a byte does not move when that byte and all later ones are replaced: {blind}", "PASSED" if same and blind else "FAILED", flush=True)
    xs = LV.raw("py")[0][:40000]
    for sysname in ("ZA",):
        s = Society(sysname, xs, 10 ** 9); a = s.run(6, 30000, True); f = s.run(30000, 35000, False); g = s.run(35000, 40000, True); p = LV.make(sysname, xs); pa = p.run(6, 30000, True); pf = p.run(30000, 35000, False); pg = p.run(35000, 40000, True)
        same = all(np.array_equal(u, v) for q, r_ in ((a, pa), (f, pf), (g, pg)) for u, v in zip(q, r_)) and s.det.N == 1; ok &= same
        print(f"{sysname} with a block longer than the stream: one situation, and every loss (reading, nothing changing, going on) equal to the plain system's bit for bit: {same}", "PASSED" if same else "FAILED", flush=True)
    s1 = Society("ZA", x, 512); u = s1.run(6, n, True, True); s2 = Society("ZA", x, 512); v = [s2.run(6, 7000, True, True), s2.run(7000, 7001, True, True), s2.run(7001, n, True, True)]; same = bool(np.array_equal(u[1], np.concatenate([q[1] for q in v])) and np.array_equal(u[2], np.concatenate([q[2] for q in v]))) and s1.det.N > 2; ok &= same
    print(f"a stream read in one piece or in three gives the same losses and awake cells ({s1.det.N} situations, {int((np.diff(u[2]) != 0).sum())} changes of the awake cell): {same}", "PASSED" if same else "FAILED", flush=True)
    return ok

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=", 1) for a in sys.argv[2:]); out = run_life(kw); r = lambda v: None if v is None else round(v, 4)
    for st in out["stages"]: print(out["tag"], f"after {st['stream']}:", " | ".join(f"{n} {r(t['nothing_changing']['cell2'])} / {r(t['going_on']['cell2'])}" for n, t in st["tests"].items()), "(second cell: nothing changing / going on)", flush=True)
    print(out["tag"], "situations", out["stages"][-1]["sizes"]["situations"], "cells", out["stages"][-1]["sizes"]["cells"], "| mean over the tests at the end, going on:", r(float(np.mean([t["going_on"]["cell2"] for t in out["stages"][-1]["tests"].values()]))), "| wall", out["wall"], flush=True)
