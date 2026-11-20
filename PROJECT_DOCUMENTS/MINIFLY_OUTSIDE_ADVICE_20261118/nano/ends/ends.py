#!/usr/bin/env python3
"""PLAN_1.md of nano/ends: where a segment starts, found by the system from its own surprise.
    python3 ends.py check                       the compiled loop against numpy on a prefix
    python3 ends.py arm=0|G|R|S|H|U [theta=...]   settings chosen on the last twentieth of the stream; then one scoring of the last tenth
Arms: 0 the table's cells (runs of the last 0..6 bytes), read with trust; G with segment cells, starts given (first letter after a byte that is not a letter);
R starts every fifth byte; S starts where the table-only system's surprise at the byte that arrived exceeds theta;
U (PLAN_2) starts where the table-only system's answer was unsure beyond theta before the byte arrived;
H starts fixed by an arbitrary function of the last two bytes, as frequent as S's (reads res/ends_armS.json).
score.py, table.py, words.py are copied unchanged from nano/table; the kernel is trust.py's with one more output (the probability given to the byte that arrived)."""
import ctypes, json, os, subprocess, sys, time
import numpy as np
import score, words
HERE = os.path.dirname(os.path.abspath(__file__)); S = 65; NL = 40; K = 6; MASK = (1 << 62) - 1; ETA0, E0 = 0.00005, 0.02
PD = ctypes.POINTER(ctypes.c_double); PF = ctypes.POINTER(ctypes.c_float); PI = ctypes.POINTER(ctypes.c_int64); PY = ctypes.POINTER(ctypes.c_int32)
THETAS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0); THETAS_U = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25); ETAS = (0.00005, 0.0001, 0.0002, 0.0005); ES = (0.005, 0.02, 0.1)

SRC = r'''
#include <math.h>
#include <stdint.h>
#define S 65
#define NL 40
static inline int level(float n, float u, int flat) {
    if (flat) return 0;
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* trust.py's loop (nano/table) with two more outputs, formed before anything changes: Q[t], the probability given to the byte that arrived (PLAN_1), and H[t], how unsure the answer was (minus the sum of p log p; PLAN_2).
   at each byte: every active cell that has counted says s(v) = log((c(v)+e)/(n+65e)); the answering cell's branches compete on the sum of trust times s;
   then the trust that was used changes by eta times (mismatch . s), and the cells count */
void mix_steps(int64_t n, int64_t K1, const int64_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double eta, double e,
               int learn_c, int learn_t, int flat, double *P, double *Q, double *H) {
    static double lt[4096]; double a[S], p[S], s[32][S]; int lev[32];
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int64_t t = 0; t < n; t++) {
        const int64_t *id = ids + t * K1; double mx = -1e300, z = 0;
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int64_t k = 0; k < K1; k++) {
            lev[k] = -1; if (id[k] < 0 || N[id[k]] < 1) continue;
            const float *c = C + id[k] * S; lev[k] = level(N[id[k]], U[id[k]], flat); const double tk = T[k * NL + lev[k]], den = log(N[id[k]] + S * e);
            for (int v = 0; v < S; v++) { const double sv = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den; s[k][v] = sv; a[v] += tk * sv; }
        }
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (Q) Q[t] = p[y[t]];
        if (H) { double h = 0; for (int v = 0; v < S; v++) if (p[v] > 0) h -= p[v] * log(p[v]); H[t] = h; }
        if (learn_t) for (int64_t k = 0; k < K1; k++) if (lev[k] >= 0) {
            double g = s[k][y[t]]; for (int v = 0; v < S; v++) g -= p[v] * s[k][v];
            T[k * NL + lev[k]] += eta * g;
        }
        if (learn_c) for (int64_t k = 0; k < K1; k++) if (id[k] >= 0) {
            float *c = C + id[k] * S; if (c[y[t]] == 0) U[id[k]] += 1; c[y[t]] += 1; N[id[k]] += 1;
        }
    }
}
'''
def kern():
    so = os.path.join(HERE, "kern_e.so"); src = os.path.join(HERE, "kern_e.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, src, "-lm"])
    L = ctypes.CDLL(so); L.mix_steps.argtypes = [ctypes.c_int64, ctypes.c_int64, PI, PY, PF, PF, PF, PD, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, PD, PD, PD]; L.mix_steps.restype = None; return L

def level_np(n, u):
    bn = 0; m = int(n)
    while m > 1 and bn < 9: m >>= 1; bn += 1
    return bn * 4 + (0 if u <= 1 else 1 if u <= 2 else 2 if u <= 4 else 3)

def steps_np(ids, y, C, N, U, T, eta, e, learn=True):
    """The reference (trust.py's, nano/table): returns the whole answer P at every byte; Q of the kernel must equal P[t, y[t]]."""
    P = np.zeros((len(y), S))
    for t in range(len(y)):
        act = [(k, i) for k, i in enumerate(ids[t]) if i >= 0 and N[i] >= 1]; a = np.zeros(S); ss = []
        for k, i in act:
            l = level_np(N[i], U[i]); sv = np.log(C[i].astype(np.float64) + e) - np.log(N[i] + S * e); ss.append((k, l, sv)); a += T[k, l] * sv
        p = np.exp(a - a.max()); p /= p.sum(); P[t] = p
        if learn:
            for k, l, sv in ss: T[k, l] += eta * (sv[y[t]] - p @ sv)
            for i in ids[t]:
                if i >= 0:
                    if C[i, y[t]] == 0: U[i] += 1
                    C[i, y[t]] += 1; N[i] += 1
    return P

class Sys:
    """The cells (counts) and the answering cell's trust, as trust.py's Sys; go() can also return the probability given to each byte that arrived."""
    def __init__(s, x, keys):
        s.x = x; s.ids, s.ncell, s.kinds, _ = words.make_ids(x, keys, K, len(x), len(x), K, 1); s.K1 = s.ids.shape[1]; s.L = kern()
        s.C = np.zeros((s.ncell, S), np.float32); s.N = np.zeros(s.ncell, np.float32); s.U = np.zeros(s.ncell, np.float32); s.T = np.full((s.K1, NL), 0.25)
    def go(s, lo, hi, eta, e, learn, want_p=False, want_q=False, want_h=False):
        ids = np.ascontiguousarray(s.ids[lo - K:hi - K]); y = np.ascontiguousarray(s.x[lo:hi].astype(np.int32)); P = np.zeros((hi - lo, S)) if want_p else None; Q = np.zeros(hi - lo) if want_q else None; H = np.zeros(hi - lo) if want_h else None
        s.L.mix_steps(hi - lo, s.K1, ids.ctypes.data_as(PI), y.ctypes.data_as(PY), s.C.ctypes.data_as(PF), s.N.ctypes.data_as(PF), s.U.ctypes.data_as(PF), s.T.ctypes.data_as(PD), eta, e, int(learn), int(learn), 0,
                      P.ctypes.data_as(PD) if want_p else None, Q.ctypes.data_as(PD) if want_q else None, H.ctypes.data_as(PD) if want_h else None)
        return Q if want_q else H if want_h else P

def surprise(x, taught_to, hi):
    """sur[t] for every byte t < hi: minus the log of the probability the table-only system gave to byte t when it arrived. The system reads x[K:taught_to] once,
    counts and trust changing after each byte is named (so sur[t] there uses only bytes before t); from taught_to on nothing changes. Bytes before K: no surprise (0)."""
    f = os.path.join(HERE, f"sur_{taught_to}_{hi}.npy")
    if os.path.exists(f): return np.load(f)
    s = Sys(x, []); sur = np.zeros(len(x)); sur[K:taught_to] = -np.log(s.go(K, taught_to, ETA0, E0, True, want_q=True))
    if hi > taught_to: sur[taught_to:hi] = -np.log(s.go(taught_to, hi, ETA0, E0, False, want_q=True))
    np.save(f, sur); return sur

def unsure(x, taught_to, hi):
    """uns[t] for every byte t < hi: how unsure the table-only system's answer for byte t was before byte t arrived (minus the sum of p log p). The system as in surprise()."""
    f = os.path.join(HERE, f"uns_{taught_to}_{hi}.npy")
    if os.path.exists(f): return np.load(f)
    s = Sys(x, []); u = np.zeros(len(x)); u[K:taught_to] = s.go(K, taught_to, ETA0, E0, True, want_h=True)
    if hi > taught_to: u[taught_to:hi] = s.go(taught_to, hi, ETA0, E0, False, want_h=True)
    np.save(f, u); return u

def word_starts(x):
    isl = x >= 13; st = isl.copy(); st[1:] &= ~isl[:-1]; return st

def pair_starts(x, share, lo, hi, seed=0):
    """Starts fixed by an arbitrary function of the last two bytes: each pair of byte values has a number drawn once; a start where it is below q; q set so that the share of starts in x[lo:hi] is nearest to `share`."""
    u = np.random.default_rng(seed).random(S * S); pr = np.zeros(len(x), np.int64); pr[1:] = x[:-1] * S + x[1:]; v = u[pr]; q = float(np.quantile(v[lo:hi], share)); return v < q

def seg_keys(x, start):
    """For every position t (after byte t has been read): the contexts (cur), (prev, cur), (prev2, prev, cur) as numbers below 2**62.
    start[t] true: byte t is the first byte of a new segment. cur = the bytes from the last start to t."""
    n = len(x); A = np.zeros(n, np.int64); B = np.zeros(n, np.int64); Cc = np.zeros(n, np.int64); xs = x.tolist(); st = start.tolist(); cur = prev = prev2 = 0
    for t in range(n):
        if st[t]: prev2 = prev; prev = cur; cur = 0
        cur = (cur * 1000003 + xs[t] + 1) & MASK
        a = (cur * 1442695040888963407 + 1) & MASK; b = (a * 2862933555777941757 + prev * 3202034522624059733 + 2) & MASK
        A[t] = a; B[t] = b; Cc[t] = (b * 3935559000370003845 + prev2 * 2685821657736338717 + 3) & MASK
    return [A, B, Cc]

def agreement(start, x, lo, hi):
    w = word_starts(x)[lo:hi]; f = start[lo:hi]; both = int((w & f).sum())
    return dict(starts=int(f.sum()), word_starts=int(w.sum()), share_of_found_that_are_word_starts=round(both / max(1, int(f.sum())), 4), share_of_word_starts_found=round(both / int(w.sum()), 4),
                mean_bytes_a_segment=round((hi - lo) / max(1, int(f.sum())), 2))

def starts_for(arm, x, taught_to, hi, theta=None, share=None):
    if arm == "G": return word_starts(x)
    if arm == "R": return np.arange(len(x)) % 5 == 0
    if arm == "S": return surprise(x, taught_to, hi) > theta
    if arm == "U": return unsure(x, taught_to, hi) > theta
    if arm == "H": return pair_starts(x, share, K, taught_to)
    raise ValueError(arm)

def one(x, keys, taught_to, lo, hi, eta, e, both=False):
    s = Sys(x, keys); s.go(K, taught_to, eta, e, True); r = score.score(s.go(lo, hi, eta, e, False, True), x[lo:hi])
    if not both: return r["nats"]
    r.update(cells=int((s.N > 0).sum()), counts_above_zero=int((s.C > 0).sum()), cells_by_kind=[int((s.N[np.unique(s.ids[:taught_to - K, k][s.ids[:taught_to - K, k] >= 0])] > 0).sum()) for k in range(s.K1)],
             trusts=int(s.T.size), trust=np.round(s.T, 3).tolist()); return r

def run(kw):
    x, cut, _ = score.load(); arm = kw.get("arm", "0"); dev = cut - cut // 20; t0 = time.process_time(); grid = {}; tgrid = {}; theta = share = None; extra = {}
    if arm == "H":
        rs = json.load(open(os.path.join(HERE, "res", "ends_armS.json"))); share = rs["agreement_stream"]["starts"] / (cut - K); extra["share_matched_to"] = share
    if arm in ("S", "U"):
        if "theta" in kw: theta = float(kw["theta"])
        else:
            for th in (THETAS if arm == "S" else THETAS_U): tgrid[str(th)] = one(x, seg_keys(x, starts_for(arm, x, dev, cut, th)), dev, dev, cut, ETA0, E0)
            theta = float(min(tgrid, key=tgrid.get))
    kd = [] if arm == "0" else seg_keys(x, starts_for(arm, x, dev, cut, theta, share))
    for e in ES:
        for eta in ETAS: grid[f"{eta}/{e}"] = one(x, kd, dev, dev, cut, eta, e)
    eta, e = (float(v) for v in min(grid, key=grid.get).split("/"))
    st = None if arm == "0" else starts_for(arm, x, cut, len(x), theta, share); kf = [] if arm == "0" else seg_keys(x, st)
    r = one(x, kf, cut, cut, len(x), eta, e, both=True)                                         # the one scoring of the last tenth for this arm
    if st is not None: extra.update(agreement_stream=agreement(st, x, K, cut), agreement_last_tenth=agreement(st, x, cut, len(x)))
    r.update(arm=arm, K=K, eta=eta, e=e, theta=theta, theta_grid=tgrid, grid=grid, cpu=round(time.process_time() - t0, 1), **extra); return r

def check():
    x, cut, _ = score.load(); n = 4000; e, eta = 0.05, 0.01; x = x[:n]
    s0 = Sys(x, []); q0 = np.concatenate([s0.go(K, 3000, ETA0, E0, True, want_q=True), s0.go(3000, n, ETA0, E0, False, want_q=True)]); st = np.zeros(n, bool); st[K:] = -np.log(q0) > 2.0
    keys = seg_keys(x, st); s = Sys(x, keys); Q = s.go(K, n, eta, e, True, want_q=True); s2 = Sys(x, keys); P2 = s2.go(K, n, eta, e, True, want_p=True)
    C = np.zeros((s.ncell, S), np.float32); N = np.zeros(s.ncell, np.float32); U = np.zeros(s.ncell, np.float32); T = np.full((s.K1, NL), 0.25); P1 = steps_np(s.ids, x[K:n], C, N, U, T, eta, e)
    d = max(np.abs(P1 - P2).max(), np.abs(T - s.T).max(), np.abs(C - s.C).max(), np.abs(Q - P1[np.arange(n - K), x[K:n]]).max())
    print(f"segments by surprise (theta 2.0, {int(st.sum())} starts), {n - K} bytes, three added kinds: compiled against numpy (whole answer, trust, counts, and the probability of the byte that arrived), largest difference {d:.1e}", "PASSED" if d < 1e-9 else "FAILED")
    C = np.zeros((s0.ncell, S), np.float32); N = np.zeros(s0.ncell, np.float32); U = np.zeros(s0.ncell, np.float32); T = np.full((s0.K1, NL), 0.25)
    Pa = steps_np(s0.ids[:3000 - K], x[K:3000], C, N, U, T, ETA0, E0); Pb = steps_np(s0.ids[3000 - K:], x[3000:n], C, N, U, T, ETA0, E0, learn=False); qn = np.concatenate([Pa, Pb])[np.arange(n - K), x[K:n]]
    s3 = Sys(x, []); h0 = np.concatenate([s3.go(K, 3000, ETA0, E0, True, want_h=True), s3.go(3000, n, ETA0, E0, False, want_h=True)]); Pn = np.concatenate([Pa, Pb]); hn = -(Pn * np.log(Pn)).sum(1)
    dh = np.abs(hn - h0).max(); print(f"how unsure the answer was (same system): compiled against numpy, largest difference {dh:.1e}", "PASSED" if dh < 1e-9 else "FAILED")
    d = np.abs(qn - q0).max(); print(f"the surprise itself (table-only system, taught on bytes {K}..2999, nothing changing after): compiled against numpy, largest difference {d:.1e}", "PASSED" if d < 1e-9 else "FAILED")
    # the keys by eye, and by a slow direct construction
    vals = score.byte_values(); seg = []; cur = b""
    direct = []
    for t in range(n):
        if st[t]: seg.append(cur); cur = b""
        cur += bytes([vals[x[t]]]); direct.append((cur, seg[-1] if seg else b"", seg[-2] if len(seg) > 1 else b""))
    ok = all((direct[i][0] == direct[j][0]) == (keys[0][i] == keys[0][j]) and (direct[i] == direct[j]) == (keys[2][i] == keys[2][j]) for i in range(K, n, 37) for j in range(K, n, 41))
    print("segment keys against a direct construction of (cur) and (prev2, prev, cur) on sampled pairs of positions:", "PASSED" if ok else "FAILED")
    print("a stretch with the starts marked by |:", repr("".join(("|" if st[t] else "") + chr(vals[x[t]]) for t in range(3000, 3160))))

if __name__ == "__main__":
    if sys.argv[1] == "check": check()
    else:
        kw = dict(a.split("=") for a in sys.argv[1:]); r = run(kw); tag = "ends_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
        json.dump(r, open(os.path.join(HERE, "res", tag + ".json"), "w"))
        print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("n", "grid", "trust", "theta_grid")}, "theta:", {k: round(v, 4) for k, v in r["theta_grid"].items()}, "best of grid:", sorted((round(v, 4), k) for k, v in r["grid"].items())[:4], flush=True)
