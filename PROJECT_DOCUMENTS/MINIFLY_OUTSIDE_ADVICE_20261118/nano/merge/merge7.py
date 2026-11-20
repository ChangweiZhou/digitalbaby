#!/usr/bin/env python3
"""PLAN_7.md: the trust of a class kind depends on whether the member's own cell speaks, and the trust for "it does not" is taught by practice.
As merge.py, with one change to the answering cell. A class kind has two rows of trust: one used where the R-cell of the same place (the cell its class cell stands for)
has counted and speaks, one used where it has not. While reading, at every byte and for every kind R that speaks and has class kinds made from it speaking beside it,
the answer is formed once more with that R-cell silent and its class kinds on their second row; the mismatch of that answer changes those second rows and nothing else.
    python3 merge7.py check
    python3 merge7.py sums|dev|life  key=value ...      (the arguments of merge.py)"""
import ctypes, hashlib, json, os, subprocess, sys
import numpy as np
import merge
from merge import NL, PD, PF, PI, p32, pf, pd, level_np
HERE = os.path.dirname(os.path.abspath(__file__))

SRC = r'''
#include <math.h>
#include <stdint.h>
#define NL 40
#define KMAX 400
#define SMAX 65
static inline int level(float n, float u) {
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
static inline double step(double eta, double used, int shrink) { return shrink ? eta / sqrt(1.0 + used) : eta; }
void mix_steps_p(int64_t n, int K1, int KB, int S, const int32_t *ids, const int32_t *y, float *C, float *N, float *U, double *T, double *Tu, double *T2, double *Tu2,
                 const int32_t *par, double eta, double e, int learn, int shrink, double *P) {
    static double lt[4096], s[KMAX][SMAX]; double a[SMAX], p[SMAX], a2[SMAX], p2[SMAX]; static int lev[KMAX], use2[KMAX], plist[KMAX]; static char mark[KMAX];
    for (int c = 0; c < 4096; c++) lt[c] = log(c + e);
    for (int64_t t = 0; t < n; t++) {
        const int32_t *id = ids + t * K1; double mx = -1e300, z = 0;
        for (int v = 0; v < S; v++) a[v] = 0;
        for (int k = 0; k < K1; k++) {
            lev[k] = -1; if (id[k] < 0 || N[id[k]] < 1) continue;
            const float *c = C + (int64_t)id[k] * S; lev[k] = level(N[id[k]], U[id[k]]); const double den = log(N[id[k]] + S * e);
            for (int v = 0; v < S; v++) s[k][v] = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + e)) - den;
        }
        for (int k = 0; k < K1; k++) if (lev[k] >= 0) {       /* a class kind is on its second row where the member's own cell does not speak */
            use2[k] = k >= KB && par[k] >= 0 && lev[par[k]] < 0; const double tk = use2[k] ? T2[k * NL + lev[k]] : T[k * NL + lev[k]];
            for (int v = 0; v < S; v++) a[v] += tk * s[k][v];
        }
        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (learn) {
            int np = 0;                                        /* practice: for each speaking R with class kinds beside it, the answer without that R-cell */
            for (int c = KB; c < K1; c++) if (lev[c] >= 0 && par[c] >= 0 && lev[par[c]] >= 0 && !mark[par[c]]) { mark[par[c]] = 1; plist[np++] = par[c]; }
            for (int i = 0; i < np; i++) {
                const int kp = plist[i]; const double tp = T[kp * NL + lev[kp]]; double m2 = -1e300, z2 = 0; mark[kp] = 0;
                for (int v = 0; v < S; v++) a2[v] = a[v] - tp * s[kp][v];
                for (int c = KB; c < K1; c++) if (lev[c] >= 0 && par[c] == kp) { const double d = T2[c * NL + lev[c]] - T[c * NL + lev[c]]; for (int v = 0; v < S; v++) a2[v] += d * s[c][v]; }
                for (int v = 0; v < S; v++) if (a2[v] > m2) m2 = a2[v];
                for (int v = 0; v < S; v++) { p2[v] = exp(a2[v] - m2); z2 += p2[v]; }
                for (int v = 0; v < S; v++) p2[v] /= z2;
                for (int c = KB; c < K1; c++) if (lev[c] >= 0 && par[c] == kp) {
                    double g = s[c][y[t]]; for (int v = 0; v < S; v++) g -= p2[v] * s[c][v];
                    T2[c * NL + lev[c]] += step(eta, Tu2[c * NL + lev[c]], shrink) * g; Tu2[c * NL + lev[c]] += 1.0;
                }
            }
            for (int k = 0; k < K1; k++) if (lev[k] >= 0) {
                double g = s[k][y[t]]; for (int v = 0; v < S; v++) g -= p[v] * s[k][v];
                if (use2[k]) { T2[k * NL + lev[k]] += step(eta, Tu2[k * NL + lev[k]], shrink) * g; Tu2[k * NL + lev[k]] += 1.0; }
                else { T[k * NL + lev[k]] += step(eta, Tu[k * NL + lev[k]], shrink) * g; Tu[k * NL + lev[k]] += 1.0; }
            }
            for (int k = 0; k < K1; k++) if (id[k] >= 0) { float *c = C + (int64_t)id[k] * S; if (c[y[t]] == 0) U[id[k]] += 1; c[y[t]] += 1; N[id[k]] += 1; }
        }
    }
}
'''
def kern():
    h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_p_{h}.so"); src = os.path.join(HERE, "kern_p.c")
    if not os.path.exists(so):
        open(src, "w").write(SRC); tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); ci, cd = ctypes.c_int, ctypes.c_double
    L.mix_steps_p.argtypes = [ctypes.c_int64, ci, ci, ci, PI, PI, PF, PF, PF, PD, PD, PD, PD, PI, cd, cd, ci, ci, PD]; L.mix_steps_p.restype = None; return L

def steps_np(ids, y, C, N, U, T, Tu, T2, Tu2, par, KB, eta, e, learn=True, shrink=False):
    S = C.shape[1]; P = np.zeros((len(y), S)); st = lambda u: eta / np.sqrt(1.0 + u) if shrink else eta; sm = lambda a: np.exp(a - a.max()) / np.exp(a - a.max()).sum()
    for t in range(len(y)):
        act = {}
        for k, i in enumerate(ids[t]):
            if i >= 0 and N[i] >= 1: act[k] = (level_np(N[i], U[i]), np.log(C[i].astype(np.float64) + e) - np.log(N[i] + S * e))
        use2 = {k: (k >= KB and par[k] >= 0 and par[k] not in act) for k in act}; a = np.zeros(S)
        for k, (l, sv) in act.items(): a += (T2[k, l] if use2[k] else T[k, l]) * sv
        p = sm(a); P[t] = p
        if learn:
            parents = []
            for c in range(KB, len(ids[t])):
                if c in act and par[c] >= 0 and par[c] in act and par[c] not in parents: parents.append(par[c])
            for kp in parents:
                a2 = a - T[kp, act[kp][0]] * act[kp][1]; ch = [c for c in range(KB, len(ids[t])) if c in act and par[c] == kp]
                for c in ch: a2 = a2 + (T2[c, act[c][0]] - T[c, act[c][0]]) * act[c][1]
                p2 = sm(a2)
                for c in ch: l, sv = act[c]; T2[c, l] += st(Tu2[c, l]) * (sv[y[t]] - p2 @ sv); Tu2[c, l] += 1.0
            for k, (l, sv) in act.items():
                g = sv[y[t]] - p @ sv
                if use2[k]: T2[k, l] += st(Tu2[k, l]) * g; Tu2[k, l] += 1.0
                else: T[k, l] += st(Tu[k, l]) * g; Tu[k, l] += 1.0
            for i in ids[t]:
                if i >= 0:
                    if C[i, y[t]] == 0: U[i] += 1
                    C[i, y[t]] += 1; N[i] += 1
    return P

class Sys7(merge.Sys):
    def __init__(s, *a, **k):
        super().__init__(*a, **k); s.L7 = kern(); s.T2 = np.zeros((s.K1, NL)); s.Tu2 = np.zeros((s.K1, NL)); s.par = np.full(s.K1, -1, np.int32)
    def _run(s, ctx, y, learn, want_p):
        P = np.zeros((len(ctx), s.S)) if want_p else None
        for a in range(0, len(ctx), 50000):
            idd = s.ids(ctx[a:a + 50000], create=learn); yy = np.ascontiguousarray(y[a:a + 50000].astype(np.int32)) if y is not None else np.zeros(len(idd), np.int32); Pp = np.zeros((len(idd), s.S)) if want_p else None
            s.L7.mix_steps_p(len(idd), s.K1, s.KB, s.S, p32(idd), p32(yy), pf(s.C), pf(s.N), pf(s.U), pd(s.T), pd(s.Tu), pd(s.T2), pd(s.Tu2), p32(s.par), s.eta, s.e, int(learn), s.shrink, pd(Pp) if want_p else None)
            if want_p: P[a:a + 50000] = Pp
        return P
    def find(s):
        old = {(sl["k"], sl["X"]): p for p, sl in enumerate(s.slots) if sl is not None}; T0, U0 = s.T2.copy(), s.Tu2.copy(); super().find(); s.T2[:] = 0.0; s.Tu2[:] = 0.0; s.par[:] = -1
        for p, sl in enumerate(s.slots):
            if sl is None: continue
            s.par[s.KB + p] = sl["k"]
            if (sl["k"], sl["X"]) in old: s.T2[s.KB + p] = T0[s.KB + old[(sl["k"], sl["X"])]]; s.Tu2[s.KB + p] = U0[s.KB + old[(sl["k"], sl["X"])]]
    def trust_summary(s):
        z = super().trust_summary()
        if s.P: z["second_row_largest"] = [round(float(x), 3) for x in s.T2[s.KB:].max(1)]; z["second_row_uses"] = [int(x) for x in s.Tu2[s.KB:].sum(1)]; z["first_row_largest"] = [round(float(x), 3) for x in s.T[s.KB:].max(1)]
        return z

def check():
    rng = np.random.default_rng(7); L = kern(); ok = True
    for S, shrink in ((14, True), (65, False)):
        K1, KB, n, nc = 12, 7, 3000, 300; ids = rng.integers(-1, nc, (n, K1)).astype(np.int32); ids[rng.random((n, K1)) < 0.3] = -1; y = rng.integers(0, S, n).astype(np.int32); par = np.full(K1, -1, np.int32); par[KB:] = rng.integers(0, KB, K1 - KB)
        mk = lambda: (np.zeros((nc, S), np.float32), np.zeros(nc, np.float32), np.zeros(nc, np.float32), np.full((K1, NL), 0.1), np.zeros((K1, NL)), np.full((K1, NL), 0.05), np.zeros((K1, NL)))
        C, N, U, T, Tu, T2, Tu2 = mk(); P = np.zeros((n, S)); L.mix_steps_p(n, K1, KB, S, p32(ids), p32(y), pf(C), pf(N), pf(U), pd(T), pd(Tu), pd(T2), pd(Tu2), p32(par), 0.05, 0.3, 1, int(shrink), pd(P))
        Cb, Nb, Ub, Tb, Tub, T2b, Tu2b = mk(); Pb = steps_np(ids, y, Cb, Nb, Ub, Tb, Tub, T2b, Tu2b, par, KB, 0.05, 0.3, True, shrink)
        d = max(np.abs(P - Pb).max(), np.abs(T - Tb).max(), np.abs(T2 - T2b).max(), np.abs(C - Cb).max(), np.abs(Tu2 - Tu2b).max()); used = int((Tu2 > 0).sum())
        good = d < 1e-9 and used > 0; ok &= good; print(f"answering loop with two rows of trust and practice, S = {S}, shrinking step = {shrink}, {n} bytes ({used} second-row trusts used): compiled against numpy, largest difference {d:.1e}", "PASSED" if good else "FAILED")
    return ok

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    merge.Sys = Sys7; kw = dict(a.split("=") for a in sys.argv[2:]); out, s = dict(sums=merge.run_sums, dev=merge.run_dev, life=merge.run_life)[sys.argv[1]](kw); tag = kw["tag"]; json.dump(out, open(os.path.join(HERE, "res", tag + ".json"), "w"))
    if sys.argv[1] == "sums": print(tag, "| last3", round(out["last3"], 1), "last", out["last"], "place", [round(v, 1) for v in out["place_last"]], "|", " ".join(f"{r['lines'] // 1000}k:{r['add']:.1f}" for r in out["rows"]), "| class kinds alone", out.get("class_kinds_alone"), "| cells", out["sizes"]["cells"], "| cpu", out["cpu"], "finding", out["cpu_finding"], flush=True)
    else: print(tag, "|", {k: round(v["nats"], 4) for k, v in out["scores"].items()} if "scores" in out else round(out["nats"], 4), "| cells", (out.get("sizes") or out)["cells"], "| cpu", out["cpu"], "finding", out["cpu_finding"], flush=True)
