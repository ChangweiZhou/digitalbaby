#!/usr/bin/env python3
"""PLAN_16.md: the second answering cell of PLAN_15.md (one part of the trust for each of the last six bytes, indexed by the byte there, born at nothing)
added to this line's own prose systems: T2 (the table's cells and the cells that step once a word) and the kinds of PLAN_10.md together.
    python3 gate_all.py check
    python3 gate_all.py text [without=companions,relay,since,pool] [gate=0|1]     the test text: last tenth with nothing changing, and going on
    python3 gate_all.py life books=new|worn [without=...] [gate=0|1]              the stream of the test text, then three books, each last tenth right after its stream
The first cell is all.py's, untouched: its answers in the same run are that system's."""
import ctypes, json, os, subprocess, sys, time
import numpy as np
import score, words, place, relay, all as A
HERE = os.path.dirname(os.path.abspath(__file__)); S = 65; NL = 40; ETA, E0, D = 0.00005, 0.02, 6
PD = ctypes.POINTER(ctypes.c_double); PF = ctypes.POINTER(ctypes.c_float); PI = ctypes.POINTER(ctypes.c_int64); PY = ctypes.POINTER(ctypes.c_int32)
ROOT = os.path.join(HERE, "..", "..", "..", "..")
BOOKS = dict(worn=[("minifly_v148_pretrain_probe_dev_20261025", b) for b in ("bryant-stories", "burgess-busterbrown", "carroll-alice")],
             new=[("minifly_v155_keys_dev_v1_20261026", b) for b in ("chesterton-ball", "austen-emma", "melville-moby_dick")])

SRC = A.SRC.replace("void all_steps(", "void gate_steps(").replace("const float *X, const int32_t *xlev) {", "const float *X, const int32_t *xlev, const int32_t *cb, int D, double *Gt, int gate, double *P2) {")
SRC = SRC.replace("double a[S], p[S], s[192][S]; int lev[192];", "double a[S], p[S], a2[S], p2[S], s[192][S]; int lev[192]; int64_t gix[193][16];")
# the second cell: after the first cell's answer is formed, before anything is learned
SRC = SRC.replace("        if (learn) {\n", """        for (int v = 0; v < S; v++) a2[v] = a[v];
        for (int64_t k = 0; k <= K1; k++) {
            const int lk = k < K1 ? lev[k] : (X ? xlev[t] : -1); if (lk < 0) continue;
            const double *sk = k < K1 ? s[k] : 0; const float *sx = X ? X + t * S : 0; double tk = 0;
            for (int j = 0; j < D; j++) { gix[k][j] = (((int64_t)j * (K1 + 1) + k) * NL + lk) * S + (gate ? 0 : cb[t * D + j]); tk += Gt[gix[k][j]]; }
            if (tk != 0) { if (sk) for (int v = 0; v < S; v++) a2[v] += tk * sk[v]; else for (int v = 0; v < S; v++) a2[v] += tk * sx[v]; }
        }
        { double m2 = -1e300, z2 = 0; for (int v = 0; v < S; v++) if (a2[v] > m2) m2 = a2[v];
          for (int v = 0; v < S; v++) { p2[v] = exp(a2[v] - m2); z2 += p2[v]; }
          for (int v = 0; v < S; v++) p2[v] /= z2; }
        if (P2) for (int v = 0; v < S; v++) P2[t * S + v] = p2[v];
        if (learn) {
            for (int64_t k = 0; k <= K1; k++) {
                const int lk = k < K1 ? lev[k] : (X ? xlev[t] : -1); if (lk < 0) continue;
                double g2;
                if (k < K1) { g2 = s[k][y[t]]; for (int v = 0; v < S; v++) g2 -= p2[v] * s[k][v]; }
                else { const float *sx = X + t * S; g2 = sx[y[t]]; for (int v = 0; v < S; v++) g2 -= p2[v] * sx[v]; }
                for (int j = 0; j < D; j++) Gt[gix[k][j]] += eta * g2;
            }
""", 1)
assert "gate_steps" in SRC and SRC.count("a2[v] = a[v]") == 1 and SRC.count("Gt[gix[k][j]] += eta * g2") == 1

def kern():
    so = os.path.join(HERE, "kern_ga.so"); src = os.path.join(HERE, "kern_ga.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, src, "-lm"])
    L = ctypes.CDLL(so); cd, ci = ctypes.c_double, ctypes.c_int
    L.gate_steps.argtypes = [ctypes.c_int64, ctypes.c_int64, PI, PY, PF, PF, PF, PD, cd, cd, ci, PD, PD, PD, cd, PF, PY, PY, ci, PD, ci, PD]; L.gate_steps.restype = None; return L

class SysGate(A.SysAll):
    """all.py's system with the second answering cell. go() returns (first cell's probabilities, second cell's) when asked."""
    def __init__(s, x, without=(), gate=0):
        super().__init__(x, without); s.L2 = kern(); s.gate = gate; s.Gt = np.zeros((D, s.K1 + 1, NL, S))
        s.cb = np.ascontiguousarray(np.stack([np.concatenate([np.zeros(d, x.dtype), x[:-d]]) for d in range(1, D + 1)], 1).astype(np.int32))      # the bytes at delays 1..6
    def go(s, lo, hi, learn, want_p=False):
        ids = np.ascontiguousarray(s.ids[lo - s.K:hi - s.K]); y = np.ascontiguousarray(s.x[lo:hi].astype(np.int32)); P = np.zeros((hi - lo, S)) if want_p else None; P2 = np.zeros((hi - lo, S)) if want_p else None
        X = np.ascontiguousarray(s.X[lo:hi]) if s.X is not None else None; xl = np.ascontiguousarray(s.xlev[lo:hi]) if s.X is not None else None; cb = np.ascontiguousarray(s.cb[lo:hi])
        s.L2.gate_steps(hi - lo, s.K1, ids.ctypes.data_as(PI), y.ctypes.data_as(PY), s.C.ctypes.data_as(PF), s.N.ctypes.data_as(PF), s.U.ctypes.data_as(PF), s.T.ctypes.data_as(PD), ETA, E0, int(learn),
                        P.ctypes.data_as(PD) if want_p else None, s.half.ctypes.data_as(PD), s.last.ctypes.data_as(PD), s.clock, X.ctypes.data_as(PF) if X is not None else None, xl.ctypes.data_as(PY) if X is not None else None,
                        cb.ctypes.data_as(PY), D, s.Gt.ctypes.data_as(PD), s.gate, P2.ctypes.data_as(PD) if want_p else None)
        if learn: s.clock += hi - lo
        return (P, P2) if want_p else None
    def state(s): return [a.copy() for a in (s.C, s.N, s.U, s.last, s.T, s.Gt)] + [s.clock]
    def put(s, st): s.C[:], s.N[:], s.U[:], s.last[:], s.T[:], s.Gt[:] = st[:6]; s.clock = st[6]

def both(s, lo, hi, learn=False, y=None):
    P1, P2 = s.go(lo, hi, learn, True); a = score.score(P1, s.x[lo:hi]); b = score.score(P2, s.x[lo:hi]); return dict(cell1=dict(nats=a["nats"], right=a["right"]), cell2=dict(nats=b["nats"], right=b["right"]))

def run_text(kw):
    x, cut, _ = score.load(); t0 = time.process_time(); w = tuple(v for v in kw.get("without", "").split(",") if v); s = SysGate(x, w, int(kw.get("gate", 0))); s.go(6, cut, True)
    r = dict(task="text", args=kw, kinds=int(s.K1), frozen=both(s, cut, len(x))); r["going_on"] = both(s, cut, len(x), True); r.update(cells=int((s.N > 0).sum()), parts=int(s.Gt.size), parts_used=int((s.Gt != 0).sum()), largest_part=float(np.abs(s.Gt).max()), cpu=round(time.process_time() - t0, 1)); return r

def load(books):
    xa, cut, _ = score.load(); vals = score.byte_values(); tab = np.full(256, 1, np.int64); tab[vals] = np.arange(S); parts = [("A", xa)]
    for folder, b in BOOKS[books]:
        raw = np.frombuffer(open(os.path.join(ROOT, folder, "data", b + ".txt"), "rb").read(), np.uint8); raw = raw[raw != 13]; parts.append((b, tab[raw]))
    x = np.concatenate([p[1] for p in parts]); segs = {}; off = 0
    for name, p in parts:
        c = cut if name == "A" else int(0.9 * len(p)); segs[name] = (off, off + c, off + len(p)); off += len(p)
    return x, segs

def run_life(kw):
    x, segs = load(kw.get("books", "new")); t0 = time.process_time(); w = tuple(v for v in kw.get("without", "").split(",") if v); assert "relay" in w, "the relay system's opinions exist for the test text only"
    s = SysGate(x, w, int(kw.get("gate", 0))); out = dict(task="life", args=kw, kinds=int(s.K1), scores={})
    for name, (a, c, b) in segs.items(): s.go(max(a, 6), c, True); out["scores"][name] = both(s, c, b)
    a, c, b = segs["A"]; out["scores"]["A_at_the_end"] = both(s, c, b); out["scores"]["A_on_the_return_going_on"] = both(s, c, b, True)
    out.update(cells=int((s.N > 0).sum()), parts=int(s.Gt.size), parts_used=int((s.Gt != 0).sum()), largest_part=float(np.abs(s.Gt).max()), cpu=round(time.process_time() - t0, 1)); return out

def check():
    """The compiled loop against numpy on a prefix (T2 with companions and a made-up given sender), both cells; and the first cell against all.py's own kernel."""
    x, cut, _ = score.load(); n = 3000; x = x[:n]; g = np.random.default_rng(0); s = SysGate(x, ("relay", "since", "pool"), 0); X = g.normal(0, 1, (n, S)); s.X = (X - np.log(np.exp(X).sum(1, keepdims=True))).astype(np.float32); s.xlev = g.integers(0, 10, n).astype(np.int32)
    s.T = np.full((s.K1 + 1, NL), 0.25); P1, P2 = s.go(6, n, True, True); Q1, Q2 = s.go(6, n, False, True)
    r = A.SysAll(x, ("relay", "since", "pool")); r.X, r.xlev = s.X, s.xlev; r.T = np.full((r.K1 + 1, NL), 0.25); R1 = r.go(6, n, True, True); d1 = max(np.abs(R1 - P1).max(), np.abs(r.T - s.T).max(), np.abs(r.C - s.C).max())
    # numpy: both cells, with fading companions
    K1 = s.K1; ids = s.ids; C = np.zeros_like(s.C, dtype=np.float64); N = np.zeros(len(s.N)); U = np.zeros(len(s.N)); last = np.zeros(len(s.N)); T = np.full((K1 + 1, NL), 0.25); Gt = np.zeros_like(s.Gt); half = s.half; M1 = np.zeros((n - 6, S)); M2 = np.zeros((n - 6, S))
    def lvl(nn, u):
        bn = 0; m = int(nn)
        while m > 1 and bn < 9: m >>= 1; bn += 1
        return bn * 4 + (0 if u <= 1 else 1 if u <= 2 else 2 if u <= 4 else 3)
    for t in range(n - 6):
        clock = float(t); a = np.zeros(S); ss = []; yy = int(x[6 + t])
        for k in range(K1):
            i = ids[t, k]
            if i < 0: continue
            if half[k] > 0:
                if N[i] <= 0: continue
                f = 2.0 ** (-(clock - last[i]) / half[k]); C[i] = np.float32(C[i] * f).astype(np.float64); N[i] = np.float64(np.float32(N[i] * f)); last[i] = clock; nn = N[i]
                if nn < 1: continue
                sv = np.log(C[i] + E0) - np.log(nn + S * E0); l = lvl(nn, int((C[i] >= 0.5).sum()))
            else:
                if N[i] < 1: continue
                sv = np.log(C[i] + E0) - np.log(N[i] + S * E0); l = lvl(N[i], int(U[i]))
            ss.append((k, l, sv)); a += T[k, l] * sv
        sx = s.X[6 + t].astype(np.float64); xl = int(s.xlev[6 + t]); a += T[K1, xl] * sx; ss.append((K1, xl, sx)); p = np.exp(a - a.max()); p /= p.sum(); M1[t] = p; a2 = a.copy(); cbt = s.cb[6 + t]
        for k, l, sv in ss: a2 += sum(Gt[j, k, l, cbt[j]] for j in range(D)) * sv
        p2 = np.exp(a2 - a2.max()); p2 /= p2.sum(); M2[t] = p2
        for k, l, sv in ss:
            g2 = sv[yy] - p2 @ sv
            for j in range(D): Gt[j, k, l, cbt[j]] += ETA * g2
        T[K1, xl] += ETA * (sx[yy] - p @ sx)
        for k, l, sv in ss[:-1]: T[k, l] += ETA * (sv[yy] - p @ sv)
        for k in range(K1):
            i = ids[t, k]
            if i >= 0:
                if half[k] > 0: last[i] = clock
                elif C[i, yy] == 0: U[i] += 1
                C[i, yy] = np.float64(np.float32(C[i, yy] + 1)); N[i] = np.float64(np.float32(N[i] + 1))
    d = max(np.abs(M1 - P1).max(), np.abs(M2 - P2).max(), np.abs(Gt - s.Gt).max(), np.abs(T - s.T).max()); good = d < 1e-6 and d1 == 0 and np.abs(P2 - P1).max() > 1e-4 and np.abs(Q2 - Q1).max() > 1e-4
    print(f"two answering cells on all.py's system (T2 with companions and a given sender), {n - 6} bytes: compiled against numpy, largest difference {d:.1e}; the first cell against all.py's kernel {d1:.1e}; the cells differ by up to {np.abs(P2 - P1).max():.3f}", "PASSED" if good else "FAILED"); return good

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=") for a in sys.argv[2:]); out = dict(text=run_text, life=run_life)[sys.argv[1]](kw); tag = "gateall_" + sys.argv[1] + "_" + "_".join(f"{k}{v.replace(',', '+')}" for k, v in sorted(kw.items()))
    json.dump(out, open(os.path.join(HERE, "res", tag + ".json"), "w")); f = lambda d: (round(d["cell1"]["nats"], 4), round(d["cell2"]["nats"], 4), round(d["cell1"]["nats"] - d["cell2"]["nats"], 4))
    if sys.argv[1] == "text": print(tag, "| kinds", out["kinds"], "| last tenth, nothing changing (first cell, second, gain):", f(out["frozen"]), "| going on:", f(out["going_on"]), "| parts used", out["parts_used"], "of", out["parts"], "| cells", out["cells"], "| cpu", out["cpu"], flush=True)
    else: print(tag, "| kinds", out["kinds"], "|", {k: f(v) for k, v in out["scores"].items()}, "| parts used", out["parts_used"], "of", out["parts"], "| cells", out["cells"], "| cpu", out["cpu"], flush=True)
