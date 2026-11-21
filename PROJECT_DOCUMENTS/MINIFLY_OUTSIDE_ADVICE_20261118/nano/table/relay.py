#!/usr/bin/env python3
"""PLAN_7.md: do the relay's cells add anything to the cells that count? The relay system of nano/loop is read as one more sender with its own trust.
    python3 relay.py arm=0|1|2 [H=1024] [seed=1] [block=512]     0: T2 alone; 1: with the relay system born at nothing; 2: born at the earlier size
    python3 relay.py check                                       the compiled loop against numpy on a prefix, with a made-up sender
The relay system (nano/loop/loop.py, unchanged) reads the stream once, at the same time as the cells that count. Its answers for a block of bytes are
computed from its strengths at the start of the block; then it learns the block."""
import ctypes, json, os, subprocess, sys, time
import numpy as np
import score, table, words, trust
HERE = os.path.dirname(os.path.abspath(__file__)); S = 65; NL = 40; ETA, E0 = 0.00005, 0.02
sys.path.insert(0, os.path.join(HERE, "..", "loop"))
PD = ctypes.POINTER(ctypes.c_double); PF = ctypes.POINTER(ctypes.c_float); PI = ctypes.POINTER(ctypes.c_int64); PY = ctypes.POINTER(ctypes.c_int32)

SRC = trust.SRC.replace("void mix_steps(", "void mix_steps_x(").replace("int learn_c, int learn_t, int flat, double *P, double *Tu, int shrink) {", "int learn_c, int learn_t, int flat, double *P, double *Tu, int shrink, const float *X, const int32_t *xlev) {")
SRC = SRC.replace("        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];", "        if (X) { const float *sx = X + t * S; const double tk = T[K1 * NL + xlev[t]]; for (int v = 0; v < S; v++) a[v] += tk * sx[v]; }      /* one more sender: its message is given */\n        for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];")
SRC = SRC.replace("        if (learn_c) for", "        if (learn_t && X) { const float *sx = X + t * S; double g = sx[y[t]]; for (int v = 0; v < S; v++) g -= p[v] * sx[v]; T[K1 * NL + xlev[t]] += eta * g; }\n        if (learn_c) for")
assert SRC.count("const float *sx") == 2 and "mix_steps_x" in SRC

def kern():
    so = os.path.join(HERE, "kern_r.so"); src = os.path.join(HERE, "kern_r.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, src, "-lm"])
    L = ctypes.CDLL(so); L.mix_steps_x.argtypes = [ctypes.c_int64, ctypes.c_int64, PI, PY, PF, PF, PF, PD, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, PD, PD, ctypes.c_int, PF, PY]; L.mix_steps_x.restype = None; return L

class SysX:
    """trust.Sys with one more sender whose message X[t] (logs of probabilities) and level xlev[t] are given for every byte."""
    def __init__(s, x, keys, K, X=None, xlev=None):
        s.x, s.K = x, K; s.ids, s.ncell, s.kinds, _ = words.make_ids(x, keys, K, len(x), len(x), K, 1); s.K1 = s.ids.shape[1]; s.L = kern(); s.X, s.xlev = X, xlev
        s.C = np.zeros((s.ncell, S), np.float32); s.N = np.zeros(s.ncell, np.float32); s.U = np.zeros(s.ncell, np.float32); s.T = np.full((s.K1 + 1, NL), 0.25); s.Tu = np.zeros((s.K1 + 1, NL))
    def go(s, lo, hi, learn, want_p=False):
        ids = np.ascontiguousarray(s.ids[lo - s.K:hi - s.K]); y = np.ascontiguousarray(s.x[lo:hi].astype(np.int32)); P = np.zeros((hi - lo, S)) if want_p else None
        X = np.ascontiguousarray(s.X[lo:hi]) if s.X is not None else None; xl = np.ascontiguousarray(s.xlev[lo:hi]) if s.X is not None else None
        s.L.mix_steps_x(hi - lo, s.K1, ids.ctypes.data_as(PI), y.ctypes.data_as(PY), s.C.ctypes.data_as(PF), s.N.ctypes.data_as(PF), s.U.ctypes.data_as(PF), s.T.ctypes.data_as(PD), ETA, E0, int(learn), int(learn), 0,
                          P.ctypes.data_as(PD) if want_p else None, s.Tu.ctypes.data_as(PD), 0, X.ctypes.data_as(PF) if X is not None else None, xl.ctypes.data_as(PY) if X is not None else None)
        return P

def opinions(rscale, H=1024, seed=1, block=512):
    """The relay system reads the stream once; returns the logs of its probabilities for every byte of the text (stream: from its strengths at the start
    of each block; last tenth: from its strengths at the end of the stream), and its level of confidence (largest probability in ten bins)."""
    import loop as LP
    x, cut, S_, pad = LP.load(); assert S_ == S; D = 10; s = LP.Sys(seed, H, D, S, pad, eta1=1.0, eta2=0.25, lam=1e-5, rscale=float(rscale), soft=1); s.tied = int(rscale == 0)
    X = np.full((len(x), S), -np.log(S), np.float32); s.read(x[:D - 1], x[1:D])                       # bytes 1..D-1 are learned with no answer kept
    def answer(lo, hi):
        for i in range(lo, hi, 256):
            a = s.branches(LP.contexts(x, i, min(i + 256, hi), D)[0]); a = a - a.max(1, keepdims=True); X[i:min(i + 256, hi)] = a - np.log(np.exp(a).sum(1, keepdims=True))
    for lo in range(D, cut, block):
        hi = min(lo + block, cut); answer(lo, hi); s.read(x[lo - 1:hi - 1], x[lo:hi])
    stream_nats = float(-X[np.arange(D, cut), x[D:cut]].mean()); answer(cut, len(x)); y = x[cut:]; lp = X[np.arange(cut, len(x)), y]
    alone = dict(nats=float(-lp.mean()), right=float(100 * (X[cut:].argmax(1) == y).mean()), stream_nats_as_read=stream_nats)
    xlev = np.minimum((np.exp(X.max(1)) * 10).astype(np.int32), 9).astype(np.int32); return X, xlev, alone

def run(kw):
    x, cut, _ = score.load(); arm = int(kw.get("arm", 0)); t0 = time.process_time(); keys = words.word_keys(x); r = dict(arm=arm); X = xlev = None
    if arm: X, xlev, alone = opinions(0 if arm == 1 else 1, int(kw.get("H", 1024)), int(kw.get("seed", 1)), int(kw.get("block", 512))); r["relay_alone"] = alone; r["relay_cpu"] = round(time.process_time() - t0, 1)
    s = SysX(x, keys, 6, X, xlev); s.go(6, cut, True); sc = score.score(s.go(cut, len(x), False, True)); r.update(nats=sc["nats"], right=sc["right"], trust_of_the_relay_by_confidence=np.round(s.T[-1, :10], 3).tolist() if arm else None, cpu=round(time.process_time() - t0, 1)); return r

def check():
    x, cut, _ = score.load(); K = 4; n = 4000; x = x[:n]; keys = [k[:n] for k in words.word_keys(x)[:1]]; g = np.random.default_rng(0); X = g.normal(0, 1, (n, S)); X = (X - np.log(np.exp(X).sum(1, keepdims=True))).astype(np.float32); xlev = g.integers(0, 10, n).astype(np.int32)
    s = SysX(x, keys, K, X, xlev); P2 = s.go(K, n, True, True); ids = s.ids; C = np.zeros((s.ncell, S), np.float32); N = np.zeros(s.ncell, np.float32); U = np.zeros(s.ncell, np.float32); T = np.full((s.K1 + 1, NL), 0.25); P1 = np.zeros((n - K, S))
    for t in range(n - K):                                                                            # the reference: trust.steps_np's arithmetic with the given sender added
        act = [(k, i) for k, i in enumerate(ids[t]) if i >= 0 and N[i] >= 1]; a = np.zeros(S); ss = []; y = x[K + t]
        for k, i in act:
            l = trust.level_np(N[i], U[i]); sv = np.log(C[i].astype(np.float64) + E0) - np.log(N[i] + S * E0); ss.append((k, l, sv)); a += T[k, l] * sv
        sx = X[K + t].astype(np.float64); a += T[s.K1, xlev[K + t]] * sx; p = np.exp(a - a.max()); p /= p.sum(); P1[t] = p
        for k, l, sv in ss: T[k, l] += ETA * (sv[y] - p @ sv)
        T[s.K1, xlev[K + t]] += ETA * (sx[y] - p @ sx)
        for i in ids[t]:
            if i >= 0:
                if C[i, y] == 0: U[i] += 1
                C[i, y] += 1; N[i] += 1
    d = max(np.abs(P1 - P2).max(), np.abs(T - s.T).max(), np.abs(C - s.C).max()); print(f"trust form with one given sender, {n - K} bytes, K = {K}: compiled against numpy, largest difference {d:.1e}", "PASSED" if d < 1e-9 else "FAILED")

if __name__ == "__main__":
    if sys.argv[1] == "check": check()
    else:
        kw = dict(a.split("=") for a in sys.argv[1:]); r = run(kw); tag = "relay_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items())); json.dump(r, open(os.path.join(HERE, "res", tag + ".json"), "w")); print(tag, r, flush=True)
