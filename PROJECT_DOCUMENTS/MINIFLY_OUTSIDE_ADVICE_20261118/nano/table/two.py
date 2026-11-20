#!/usr/bin/env python3
"""PLAN_6.md: cells of two memories. Each cell of the table gets a companion whose counts fade with the time since it was last active.
    python3 two.py T0|T2 H        H = half-life in bytes (0 = no companions; then the figures of PLAN_5.md must come out)
    python3 two.py T0|T2 H return (PLAN_6.md, addition 1) after A and B, the last tenth of A read with everything going on
    python3 two.py check          the compiled loop against numpy on a prefix
The life, the texts and the scoring are those of life.py (PLAN_5.md)."""
import ctypes, json, os, subprocess, sys, time
import numpy as np
import score, table, words, trust, life
HERE = os.path.dirname(os.path.abspath(__file__)); S = 65; NL = 40; ETA, E = 0.00005, 0.02
PD = ctypes.POINTER(ctypes.c_double); PF = ctypes.POINTER(ctypes.c_float); PI = ctypes.POINTER(ctypes.c_int64); PY = ctypes.POINTER(ctypes.c_int32)

SRC = r'''
#include <math.h>
#include <stdint.h>
#define S 65
#define NL 40
static inline int level(double n, int u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* as mix_steps of trust.py; a kind with half[k] > 0 is a kind of companions: before a companion speaks or counts, its counts are multiplied by
   2^(-(time since it was last active)/half[k]); it is silent when its faded total is under one; its level of u is the number of counts of 1/2 or more.
   With learn = 0 nothing changes and the time stands at clock0. */
void two_steps(int64_t n, int64_t K1, const int64_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double eta, double e,
               int learn, double *P, const double *half, double *last, double clock0) {
    static double lt[4096]; double a[S], p[S], s[40][S]; int lev[40];
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int64_t t = 0; t < n; t++) {
        const int64_t *id = ids + t * K1; double mx = -1e300, z = 0; const double clock = clock0 + (learn ? (double)t : 0.0);
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int64_t k = 0; k < K1; k++) {
            lev[k] = -1; const int64_t i = id[k]; if (i < 0) continue;
            float *c = C + i * S;
            if (half[k] > 0) {
                if (N[i] <= 0) continue;
                const double f = exp2(-(clock - last[i]) / half[k]), nn = N[i] * f;
                if (learn) { for (int v = 0; v < S; v++) c[v] = (float)(c[v] * f); N[i] = (float)nn; last[i] = clock; }
                if (nn < 1) continue;
                int u = 0; const double den = log(nn + S * e);
                for (int v = 0; v < S; v++) { const double cv = learn ? (double)c[v] : c[v] * f; if (cv >= 0.5) u++; s[k][v] = log(cv + e) - den; }
                lev[k] = level(nn, u); const double tk = T[k * NL + lev[k]];
                for (int v = 0; v < S; v++) a[v] += tk * s[k][v];
            } else {
                if (N[i] < 1) continue;
                lev[k] = level(N[i], (int)U[i]); const double tk = T[k * NL + lev[k]], den = log(N[i] + S * e);
                for (int v = 0; v < S; v++) { const double sv = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den; s[k][v] = sv; a[v] += tk * sv; }
            }
        }
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (learn) {
            for (int64_t k = 0; k < K1; k++) if (lev[k] >= 0) {
                double g = s[k][y[t]]; for (int v = 0; v < S; v++) g -= p[v] * s[k][v];
                T[k * NL + lev[k]] += eta * g;
            }
            for (int64_t k = 0; k < K1; k++) if (id[k] >= 0) {
                float *c = C + id[k] * S; if (half[k] > 0) last[id[k]] = clock; else if (c[y[t]] == 0) U[id[k]] += 1;
                c[y[t]] += 1; N[id[k]] += 1;
            }
        }
    }
}
'''
def kern():
    so = os.path.join(HERE, "kern_2.so"); src = os.path.join(HERE, "kern_2.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, src, "-lm"])
    L = ctypes.CDLL(so); L.two_steps.argtypes = [ctypes.c_int64, ctypes.c_int64, PI, PY, PF, PF, PF, PD, ctypes.c_double, ctypes.c_double, ctypes.c_int, PD, PD, PD, ctypes.c_double]; L.two_steps.restype = None; return L

def steps_np(ids, y, C, N, U, T, half, last, clock0, eta, e, learn=True):
    """The reference: plain numpy, one byte at a time, in float32 for the counts as the compiled loop."""
    P = np.zeros((len(y), S))
    for t in range(len(y)):
        clock = clock0 + (t if learn else 0); a = np.zeros(S); ss = []
        for k, i in enumerate(ids[t]):
            if i < 0: continue
            if half[k] > 0:
                if N[i] <= 0: continue
                f = 2.0 ** (-(clock - last[i]) / half[k]); nn = float(N[i]) * f
                if learn: C[i] = (C[i].astype(np.float64) * f).astype(np.float32); N[i] = np.float32(nn); last[i] = clock
                if nn < 1: continue
                cv = C[i].astype(np.float64) if learn else C[i].astype(np.float64) * f; u = int((cv >= 0.5).sum()); sv = np.log(cv + e) - np.log(nn + S * e); l = trust.level_np(nn, u)
            else:
                if N[i] < 1: continue
                sv = np.log(C[i].astype(np.float64) + e) - np.log(float(N[i]) + S * e); l = trust.level_np(N[i], U[i])
            ss.append((k, l, sv)); a += T[k, l] * sv
        p = np.exp(a - a.max()); p /= p.sum(); P[t] = p
        if learn:
            for k, l, sv in ss: T[k, l] += eta * (sv[y[t]] - p @ sv)
            for k, i in enumerate(ids[t]):
                if i < 0: continue
                if half[k] > 0: last[i] = clock
                elif C[i, y[t]] == 0: U[i] += 1
                C[i, y[t]] += 1; N[i] += 1
    return P

class Sys2:
    def __init__(s, x, keys, K, H):
        ids, ncell, kinds, _ = words.make_ids(x, keys, K, len(x), len(x), K, 1); ntab = ncell - sum(kinds); tot = ncell
        Hs = [h for h in (H if isinstance(H, (tuple, list)) else (H,)) if h]; tabs = ids[:, :K + 1]            # one kind of companions for each half-life given (PLAN_6.md, addition 2)
        for j, h in enumerate(Hs): ids = np.hstack([ids, np.where(tabs >= 0, tabs + ncell + j * ntab, -1)])
        tot = ncell + len(Hs) * ntab
        s.x, s.K, s.ids, s.K1, s.L = x, K, np.ascontiguousarray(ids), ids.shape[1], kern(); s.half = np.zeros(s.K1)
        for j, h in enumerate(Hs): s.half[s.K1 - (len(Hs) - j) * (K + 1):s.K1 - (len(Hs) - j - 1) * (K + 1)] = h
        s.C = np.zeros((tot, S), np.float32); s.N = np.zeros(tot, np.float32); s.U = np.zeros(tot, np.float32); s.last = np.zeros(tot); s.T = np.full((s.K1, NL), 0.25); s.clock = 0.0
    def go(s, lo, hi, learn, want_p=False):
        ids = np.ascontiguousarray(s.ids[lo - s.K:hi - s.K]); y = np.ascontiguousarray(s.x[lo:hi].astype(np.int32)); P = np.zeros((hi - lo, S)) if want_p else None
        s.L.two_steps(hi - lo, s.K1, ids.ctypes.data_as(PI), y.ctypes.data_as(PY), s.C.ctypes.data_as(PF), s.N.ctypes.data_as(PF), s.U.ctypes.data_as(PF), s.T.ctypes.data_as(PD), ETA, E, int(learn),
                      P.ctypes.data_as(PD) if want_p else None, s.half.ctypes.data_as(PD), s.last.ctypes.data_as(PD), s.clock)
        if learn: s.clock += hi - lo
        return P
    def cells(s): return int((s.N > 0).sum())

def run(name, H, ret=False):
    x, segs, replaced = life.load_all(); t0 = time.process_time(); K = 6; keys = words.word_keys(x) if name == "T2" else []; r = dict(system=name, H=H); a, c, b = segs["A"]
    sy = Sys2(x, keys, K, H); sy.go(K, c, True); r["cells_after_A"] = sy.cells(); r["after_A"] = life.scores(sy, x, segs)
    for k in segs:
        if k != "A": sy.go(segs[k][0], segs[k][1], True)
    r["cells_after_A_and_B"] = sy.cells(); r["after_A_and_B"] = life.scores(sy, x, segs); r["trust_after_A_and_B"] = np.round(sy.T, 3).tolist()
    if ret: on2 = score.score(sy.go(c, b, True, True), x[c:b]); r["A_last_tenth_going_on_after_B"] = dict(nats=on2["nats"], right=on2["right"])      # PLAN_6.md, addition 1: the return to A; changes the system, so it is done last
    s2 = Sys2(x, keys, K, H)
    for k in segs:
        if k != "A": s2.go(segs[k][0], segs[k][1], True)
    if ret: r["cpu"] = round(time.process_time() - t0, 1); return r
    r["B_only"] = life.scores(s2, x, segs); del s2
    s3 = Sys2(x, keys, K, H); s3.go(K, c, True); on = score.score(s3.go(c, b, True, True), x[c:b]); r["A_last_tenth_going_on"] = dict(nats=on["nats"], right=on["right"]); r["cpu"] = round(time.process_time() - t0, 1); return r

def moments(name, H):
    """PLAN_6.md, addition 3: each text's held-out tenth scored right after its own stream; at the end, A again, and A with the companions silenced."""
    x, segs, _ = life.load_all(); K = 6; keys = words.word_keys(x) if name == "T2" else []; r = dict(system=name, H=H); sc = lambda sy, k: score.score(sy.go(segs[k][1], segs[k][2], False, True), x[segs[k][1]:segs[k][2]])["nats"]
    sy = Sys2(x, keys, K, H); r["A_B"] = {}
    for k in segs: sy.go(max(segs[k][0], K), segs[k][1], True); r["A_B"][k] = sc(sy, k)
    r["A_at_end"] = sc(sy, "A"); ncomp = int((sy.half > 0).sum())
    if ncomp: T = sy.T.copy(); sy.T[sy.half > 0] = 0.0; r["A_at_end_companions_silenced"] = sc(sy, "A"); sy.T = T
    s2 = Sys2(x, keys, K, H); r["B_only"] = {}
    for k in segs:
        if k != "A": s2.go(segs[k][0], segs[k][1], True); r["B_only"][k] = sc(s2, k)
    return r

def check():
    x, cut, _ = score.load(); K = 4; n = 4000; x = x[:n]; keys = [k[:n] for k in words.word_keys(x)[:1]]
    for H in (0, 300):
        s = Sys2(x, keys, K, H); P2 = np.vstack([s.go(K, 3000, True, True), s.go(3000, 3500, False, True), s.go(3500, n, True, True)])
        r = Sys2(x, keys, K, H); a1 = (r.ids, None); ETA_, E_ = ETA * 200, E
        # the reference is run with the same constants as the compiled loop
        P1a = steps_np(r.ids[:3000 - K], x[K:3000], r.C, r.N, r.U, r.T, r.half, r.last, 0.0, ETA, E, True); P1b = steps_np(r.ids[3000 - K:3500 - K], x[3000:3500], r.C, r.N, r.U, r.T, r.half, r.last, 3000.0 - K, ETA, E, False)
        P1c = steps_np(r.ids[3500 - K:], x[3500:n], r.C, r.N, r.U, r.T, r.half, r.last, 3000.0 - K, ETA, E, True); P1 = np.vstack([P1a, P1b, P1c])
        d = max(np.abs(P1 - P2).max(), np.abs(r.T - s.T).max(), np.abs(r.C - s.C).max()); print(f"two memories, {n - K} bytes (learn, score, learn), K = {K}, one added kind, H = {H}: compiled against numpy, largest difference {d:.1e}", "PASSED" if d < 1e-6 else "FAILED")

if __name__ == "__main__":
    if sys.argv[1] == "check": check()
    else:
        name = sys.argv[1]; H = tuple(int(h) for h in sys.argv[2].split(",")); H = H[0] if len(H) == 1 else H; tagH = sys.argv[2].replace(",", "+")
        if len(sys.argv) > 3 and sys.argv[3] == "moments":
            r = moments(name, H); json.dump(r, open(os.path.join(HERE, "res", f"two_{name}_H{tagH}_moments.json"), "w")); print(name, "H", tagH, "| A, B in order:", {k[:6]: round(v, 4) for k, v in r["A_B"].items()}, "| B only:", {k[:6]: round(v, 4) for k, v in r["B_only"].items()}, "| A at the end:", round(r["A_at_end"], 4), "silenced:", round(r.get("A_at_end_companions_silenced", float("nan")), 4), flush=True); sys.exit()
        if len(sys.argv) > 3 and sys.argv[3] == "return":
            r = run(name, H, True); json.dump(r, open(os.path.join(HERE, "res", f"two_{name}_H{tagH}_return.json"), "w")); print(name, "H", H, "A after A and B, nothing changing:", round(r["after_A_and_B"]["A"]["nats"], 4), "| the return to A, going on:", round(r["A_last_tenth_going_on_after_B"]["nats"], 4), flush=True); sys.exit()
        r = run(name, H); json.dump(r, open(os.path.join(HERE, "res", f"two_{name}_H{tagH}.json"), "w"))
        f = lambda d: " ".join(f"{k[:6]} {d[k]['nats']:.4f}" for k in d)
        print(name, "H", H, "| after A:", f(r["after_A"]), "| after A and B:", f(r["after_A_and_B"]), "| B only:", f(r["B_only"]), "| A going on:", round(r["A_last_tenth_going_on"]["nats"], 4), "| cells", r["cells_after_A"], r["cells_after_A_and_B"], "| cpu", r["cpu"], flush=True)
