#!/usr/bin/env python3
"""PLAN_1.md: a table made of cells, on the text of the test. numpy only for the counting form; kern_t.c for the mismatch form.
    python3 table.py count [K=6] [kn=1] [minuse=1] [d=...]        counting cells; d chosen on the last twentieth of the stream unless given
    python3 table.py mism  [K=6] [eta=...] [readings=1]           cells taught by the mismatch
    python3 table.py check                                        the compiled mismatch loop against numpy on a prefix
What the program computes in one sweep is what the cells hold after reading the stream once: a count does not depend on the order of arrival."""
import ctypes, json, os, subprocess, sys, time
import numpy as np
import score
HERE = os.path.dirname(os.path.abspath(__file__)); S = 65

def ctx_codes(x, k):
    """c[t] = the k bytes ending at t as one number (x[t] lowest), for every t; valid for t >= k - 1."""
    c = np.zeros(len(x), np.int64)
    for i in range(k): c[i:] += x[:len(x) - i] * (S ** i)
    return c

class Counting:
    """The cells after one reading of x[lo:hi]: for each k, the contexts that occurred, and for each its counts by following byte."""
    def __init__(s, x, lo, hi, K, kn=1, minuse=1):
        s.K, s.kn = K, kn; s.tab = []; t = np.arange(max(lo, K - 1), hi - 1); nxt = x[t + 1]; raw = []
        for k in range(K + 1):
            pc, cnt = np.unique(ctx_codes(x, k)[t] * S + nxt, return_counts=True); raw.append((pc, cnt.astype(np.float64)))
        for k in range(K + 1):
            pc, cnt = raw[k]
            if kn and k < K:                              # a shorter cell counts the different longer cells that reported each byte to it
                hp = raw[k + 1][0]; pc, cnt = np.unique((hp // S % (S ** k)) * S + hp % S, return_counts=True); cnt = cnt.astype(np.float64)
            ctx = pc // S; uc, start = np.unique(ctx, return_index=True); n = np.add.reduceat(cnt, start); u = np.diff(np.append(start, len(ctx))).astype(np.float64)
            use = np.add.reduceat(raw[k][1], np.unique(raw[k][0] // S, return_index=True)[1])          # times the cell was active (its own count)
            keep = use >= minuse if k else np.ones(1, bool)
            s.tab.append(dict(pc=pc, cnt=cnt, uc=uc[keep], n=n[keep], u=u[keep], cells=int(keep.sum()), strengths=int((np.isin(ctx, uc[keep])).sum())))
    def probs(s, x, lo, hi, d):
        t = np.arange(lo - 1, hi - 1); P = np.full((len(t), S), 1.0 / S)
        for k in range(s.K + 1):
            T = s.tab[k]; c = ctx_codes(x, k)[t]; j = np.searchsorted(T["uc"], c); j[j == len(T["uc"])] = 0; has = T["uc"][j] == c
            C = np.zeros((len(t), S))
            for v in range(S):
                q = np.searchsorted(T["pc"], c * S + v); q[q == len(T["pc"])] = 0; C[:, v] = np.where(T["pc"][q] == c * S + v, T["cnt"][q], 0.0)
            new = (np.maximum(C - d, 0.0) + (d * T["u"][j])[:, None] * P) / T["n"][j][:, None]; P = np.where(has[:, None], new, P)
        return P / P.sum(1, keepdims=True)

def run_count(kw):
    x, cut, _ = score.load(); K = int(kw.get("K", 6)); kn = int(kw.get("kn", 1)); mu = int(kw.get("minuse", 1)); dev = cut - cut // 20; t0 = time.process_time()
    if "d" in kw: d = float(kw["d"]); grid = {}
    else:
        c = Counting(x, 0, dev, K, kn, mu); grid = {dd: score.score(c.probs(x, dev, cut, dd), x[dev:cut])["nats"] for dd in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}; d = min(grid, key=grid.get)
    c = Counting(x, 0, cut, K, kn, mu); r = score.score(c.probs(x, cut, len(x), d))
    r.update(form="count", K=K, kn=kn, minuse=mu, d=d, d_grid=grid, cells=sum(T["cells"] for T in c.tab), strengths_above_zero=sum(T["strengths"] for T in c.tab), cpu=round(time.process_time() - t0, 1)); return r

SRC = r'''
#include <math.h>
#include <stdint.h>
/* at each byte: the active cells' strengths are summed, the branches compete, and every active cell changes by its step times the mismatch */
void table_steps(int64_t n, int64_t K1, const int64_t *ids, const int32_t *y, double *W, double *uses, double eta, int learn, double *P) {
    double a[65], p[65];
    for (int64_t t = 0; t < n; t++) {
        const int64_t *id = ids + t * K1; double mx = -1e300, z = 0;
        for (int v = 0; v < 65; v++) a[v] = 0;
        for (int64_t k = 0; k < K1; k++) if (id[k] >= 0) { const double *w = W + id[k] * 65; for (int v = 0; v < 65; v++) a[v] += w[v]; }
        for (int v = 0; v < 65; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < 65; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < 65; v++) p[v] /= z;
        if (P) for (int v = 0; v < 65; v++) P[t * 65 + v] = p[v];
        if (learn) for (int64_t k = 0; k < K1; k++) if (id[k] >= 0) {
            double *w = W + id[k] * 65; const double st = eta / sqrt(1.0 + uses[id[k]]);
            for (int v = 0; v < 65; v++) w[v] -= st * p[v];
            w[y[t]] += st; uses[id[k]] += 1.0;
        }
    }
}
'''
def kern():
    so = os.path.join(HERE, "kern_t.so"); src = os.path.join(HERE, "kern_t.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, src, "-lm"])
    L = ctypes.CDLL(so); PD = ctypes.POINTER(ctypes.c_double)
    L.table_steps.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), PD, PD, ctypes.c_double, ctypes.c_int, PD]; L.table_steps.restype = None; return L

def cell_ids(x, lo, hi, K, t_build_hi):
    """For every byte position p in lo..hi-1 (the byte to be named), the cells active before it: one for each k = 0..K, or -1 if that context never occurred in x[:t_build_hi]."""
    tb = np.arange(K - 1, t_build_hi - 1); t = np.arange(lo - 1, hi - 1); ids = np.full((len(t), K + 1), -1, np.int64); off = 0
    for k in range(K + 1):
        cc = ctx_codes(x, k); uc = np.unique(cc[tb]); j = np.searchsorted(uc, cc[t]); j[j == len(uc)] = 0; ids[:, k] = np.where(uc[j] == cc[t], j + off, -1); off += len(uc)
    return ids, off

def steps_np(ids, y, W, uses, eta, learn=True):
    P = np.zeros((len(y), S))
    for t in range(len(y)):
        act = ids[t][ids[t] >= 0]; a = W[act].sum(0); p = np.exp(a - a.max()); p /= p.sum(); P[t] = p
        if learn:
            for c in act:
                st = eta / np.sqrt(1.0 + uses[c]); W[c] -= st * p; W[c, y[t]] += st; uses[c] += 1.0
    return P

def mism(x, lo_b, hi_b, lo_s, hi_s, K, eta, readings=1):
    """Cells read x[lo_b:hi_b] `readings` times, learning from the mismatch; then x[lo_s:hi_s] is scored with learning off."""
    L = kern(); PD = ctypes.POINTER(ctypes.c_double); ids, ncell = cell_ids(x, max(lo_b, K), hi_b, K, hi_b); y = x[max(lo_b, K):hi_b].astype(np.int32)
    W = np.zeros((ncell, S)); uses = np.zeros(ncell); ids = np.ascontiguousarray(ids)
    for _ in range(readings): L.table_steps(len(y), K + 1, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), W.ctypes.data_as(PD), uses.ctypes.data_as(PD), eta, 1, None)
    ids2, _ = cell_ids(x, lo_s, hi_s, K, hi_b); ids2 = np.ascontiguousarray(ids2); y2 = x[lo_s:hi_s].astype(np.int32); P = np.zeros((len(y2), S))
    L.table_steps(len(y2), K + 1, ids2.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), y2.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), W.ctypes.data_as(PD), uses.ctypes.data_as(PD), eta, 0, P.ctypes.data_as(PD))
    return P, ncell, int((uses > 0).sum())

def run_mism(kw):
    x, cut, _ = score.load(); K = int(kw.get("K", 6)); rd = int(kw.get("readings", 1)); dev = cut - cut // 20; t0 = time.process_time()
    if "eta" in kw: eta = float(kw["eta"]); grid = {}
    else: grid = {e: score.score(mism(x, 0, dev, dev, cut, K, e, rd)[0], x[dev:cut])["nats"] for e in (0.125, 0.25, 0.5, 1.0, 2.0)}; eta = min(grid, key=grid.get)
    P, ncell, used = mism(x, 0, cut, cut, len(x), K, eta, rd); r = score.score(P); r.update(form="mismatch", K=K, eta=eta, eta_grid=grid, readings=rd, cells=ncell, cpu=round(time.process_time() - t0, 1)); return r

def check():
    x, cut, _ = score.load(); K = 4; n = 3000; ids, ncell = cell_ids(x, K, n, K, n); y = x[K:n].astype(np.int32); W1 = np.zeros((ncell, S)); u1 = np.zeros(ncell); P1 = steps_np(ids, y, W1, u1, 0.5)
    L = kern(); PD = ctypes.POINTER(ctypes.c_double); W2 = np.zeros((ncell, S)); u2 = np.zeros(ncell); P2 = np.zeros((len(y), S)); ids = np.ascontiguousarray(ids)
    L.table_steps(len(y), K + 1, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), W2.ctypes.data_as(PD), u2.ctypes.data_as(PD), 0.5, 1, P2.ctypes.data_as(PD))
    e = max(np.abs(W1 - W2).max() / np.abs(W1).max(), np.abs(P1 - P2).max()); print(f"mismatch form, {len(y)} bytes, K = {K}: compiled against numpy, largest difference {e:.1e}", "PASSED" if e < 1e-9 else "FAILED")
    c = Counting(x, 0, 200000, 2, kn=0); P = c.probs(x, 200000, 201000, 0.0)            # with no discount the longest cell's relative counts must come out
    t = 200500; k2 = ctx_codes(x, 2)[t - 1]; tt = np.arange(1, 199999); m = ctx_codes(x, 2)[tt] == k2; ref = np.bincount(x[tt + 1][m], minlength=S) / m.sum(); print("counting form: relative counts of one context reproduced:", bool(np.abs(P[t - 200000] - ref).max() < 1e-12))

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:])
    if sys.argv[1] == "check": check()
    else:
        r = run_count(kw) if sys.argv[1] == "count" else run_mism(kw); tag = sys.argv[1] + "_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
        json.dump(r, open(os.path.join(HERE, "res", tag + ".json"), "w")); print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("n", "d_grid", "eta_grid")}, {k: round(v, 4) for k, v in (r.get("d_grid") or r.get("eta_grid") or {}).items()})
