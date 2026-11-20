#!/usr/bin/env python3
"""nano/loop (roadmap steps L1 and L2): the loop of loop_small_case on the test text of nano/data, with the answering cell in two forms.
    python3 loop.py run <label> [arm=P|LS] [H=256] [D=10] [eta1=1] [eta2=1/16] [rscale=1] [passes=8] [seed=1] [lam=1e-5] [numpy=1]
    python3 loop.py tables
The system is cells2.Sys of loop_small_case (plain: logistic cells, every line read, one step of relay), written again here with the number of byte
values and of delay lines as arguments, and with the strengths held by line. step() is the numpy reference; kernp.c is the compiled inner loop and
check.py compares the two.
  arm=LS: mismatch = (1 for the byte that arrived, 0 for the others) - a, a = the branch outputs.
  arm=P : the branches compete: p = exp(a) / sum(exp(a)); mismatch = the same indicator - p. Everything else is identical.
  rscale: the size of the return strengths at birth, as a multiple of the earlier size (uniform in -1..1); 0 is 'born at nothing'.
The first nine tenths of the text are the stream, read `passes` times one byte at a time; the last tenth is never read by the learner. After each
reading, with learning off, every byte of the last tenth is scored from the D bytes before it: loss in nats and bits, share named right."""
import ctypes, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); TEXT = os.path.join(HERE, "..", "data", "tinyshakespeare.txt")
SO = os.path.join(HERE, "kernp.so"); SRC = os.path.join(HERE, "kernp.c"); FLAGS = ["-O3", "-march=native", "-ffast-math", "-shared", "-fPIC"]
I64, F64, PD, PI, P32 = ctypes.c_int64, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32)

def load():
    raw = np.frombuffer(open(TEXT, "rb").read(), np.uint8); vals = np.unique(raw); idx = np.zeros(256, np.int32); idx[vals] = np.arange(len(vals))
    x = idx[raw]; cut = len(x) * 9 // 10; return x, cut, len(vals), int(idx[10])        # the whole text, where the never-read tenth starts, byte values, the line feed

def sg(u): return 1 / (1 + np.exp(-u))

class PNet(ctypes.Structure):
    _fields_ = [(k, I64) for k in "H NL S D soft right".split()] + [(k, F64) for k in "eta1 eta2 lam loss".split()] + [(k, PD) for k in "W1t b1 W2 b2 Rt".split()] + [("hist", PI)]
_lib = None
def lib():
    global _lib
    if _lib is None:
        if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
            tmp = f"{SO}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", *FLAGS, "-o", tmp, SRC, "-lm"]); os.replace(tmp, SO)
        _lib = ctypes.CDLL(SO); _lib.p_steps.restype = I64; _lib.p_steps.argtypes = [ctypes.POINTER(PNet), I64, P32, P32]
    return _lib

class Sys:
    def __init__(s, seed, H, D, S, pad, eta1=1.0, eta2=1 / 16, lam=1e-5, rscale=1.0, soft=0):
        wr = np.random.default_rng([seed, 20261119]); s.H, s.D, s.S, s.NL = H, D, S, D * S; s.eta1, s.eta2, s.lam, s.soft = eta1, eta2, lam, soft
        s.W1t = np.ascontiguousarray(wr.uniform(-.5, .5, (H, s.NL)).T); s.b1 = np.zeros(H)          # births as in cells2.Sys: strengths from lines uniform in -1/2..1/2
        s.Rt = np.ascontiguousarray(wr.uniform(-1, 1, (H, S)).T) * rscale                              # return strengths uniform in -1..1, times rscale
        s.W2 = np.zeros((S, H)); s.b2 = np.zeros(S); s.OFF = np.arange(D) * S; s.hist = np.full(D, pad, np.int64); s.loss = 0.0; s.right = 0
    def branches(s, hist):
        """hist: positions by D (newest first) -> the answering cell's branch outputs a, for many positions at once."""
        return sg(s.W1t[hist + s.OFF].sum(1) + s.b1) @ s.W2.T + s.b2
    def step(s, x, y):
        """The numpy reference: byte x arrives on the lines, the next byte y is named and then arrives."""
        s.hist[1:] = s.hist[:-1].copy(); s.hist[0] = x; act = s.OFF + s.hist
        h = sg(s.W1t[act].sum(0) + s.b1); a = s.W2 @ h + s.b2; s.right += int(a.argmax() == y)
        if s.soft: p = np.exp(a - a.max()); p /= p.sum(); m = -p; s.loss -= np.log(p[y])
        else: q = np.maximum(a, 1e-3); s.loss -= np.log(q[y] / q.sum()); m = -a
        m[y] += 1; g = s.eta2 / (h @ h + 1)
        d = s.eta1 * (m @ s.Rt) * h * (1 - h)            # what a cell receives over its return strengths, times its own sensitivity
        d2 = np.outer(g * m, h); s.Rt += d2; s.W2 += d2; s.b2 += g * m; s.W1t[act] += d; s.b1 += d
        s.W2 *= 1 - s.lam; s.Rt *= 1 - s.lam
    def read(s, xs, ys, fast=True):
        if not fast:
            for x, y in zip(xs.tolist(), ys.tolist()): s.step(x, y)
            return
        n = PNet(); n.H, n.NL, n.S, n.D, n.soft, n.right = s.H, s.NL, s.S, s.D, s.soft, 0; n.eta1, n.eta2, n.lam, n.loss = s.eta1, s.eta2, s.lam, 0.0
        for k in ("W1t", "b1", "W2", "b2", "Rt"):
            a = getattr(s, k); assert a.dtype == np.float64 and a.flags.c_contiguous; setattr(n, k, a.ctypes.data_as(PD))
        n.hist = s.hist.ctypes.data_as(PI); xs = np.ascontiguousarray(xs, np.int32); ys = np.ascontiguousarray(ys, np.int32)
        lib().p_steps(ctypes.byref(n), len(xs), xs.ctypes.data_as(P32), ys.ctypes.data_as(P32)); s.loss += n.loss; s.right += n.right

def contexts(x, lo, hi, D):
    """For every position t in lo..hi-1: the D bytes before it, newest first (lo >= D)."""
    return np.stack([x[lo - 1 - d: hi - 1 - d] for d in range(D)], 1), x[lo:hi]

def score(s, x, cut, soft, chunk=None):
    hist, y = contexts(x, cut, len(x), s.D); right = 0; nats = 0.0; chunk = chunk or max(64, 2_000_000 // (s.H * s.D))
    for i in range(0, len(y), chunk):
        a = s.branches(hist[i:i + chunk]); yy = y[i:i + chunk]; right += int((a.argmax(1) == yy).sum())
        if soft: a = a - a.max(1, keepdims=True); lp = a[np.arange(len(yy)), yy] - np.log(np.exp(a).sum(1))
        else: p = np.maximum(a, 1e-3); lp = np.log(p[np.arange(len(yy)), yy] / p.sum(1))               # crude: cut off below at 1e-3, made to sum to one
        nats -= float(lp.sum())
    return dict(nats=nats / len(y), bits=nats / len(y) / np.log(2), right=100 * right / len(y))

def tables(x, cut, S, kmax=6):
    """Unsmoothed tables of the last k bytes, counted on the stream; the longest context seen in the stream decides (as loop_small_case/prose.py).
    Loss: the counts of that context as shares, cut off below at 1e-3 and made to sum to one (crude, as for arm LS)."""
    train = x[:cut]; counts = []
    for k in range(0, kmax + 1):
        key = np.zeros(cut - k, np.int64)
        for d in range(k): key = key * S + train[d: cut - k + d]
        uk, inv = np.unique(key, return_inverse=True); c = np.zeros((len(uk), S), np.int32); np.add.at(c, (inv, train[k:]), 1); counts.append((uk, c))
    out = {}; y = x[cut:]; n = len(y)
    keys = []
    for k in range(0, kmax + 1):
        key = np.zeros(n, np.int64)
        for d in range(k): key = key * S + x[cut - k + d: len(x) - k + d]
        uk, c = counts[k]; pos = np.searchsorted(uk, key); pos[pos == len(uk)] = 0; keys.append((pos, uk[pos] == key))
    for K in range(1, kmax + 1):
        row = np.zeros((n, S)); done = np.zeros(n, bool)
        for k in range(K, -1, -1):
            pos, seen = keys[k]; take = seen & ~done; row[take] = counts[k][1][pos[take]]; done |= take
        p = row / row.sum(1, keepdims=True); q = np.maximum(p, 1e-3); q /= q.sum(1, keepdims=True); nats = float(-np.log(q[np.arange(n), y]).mean())
        out[K] = dict(right=float(100 * (row.argmax(1) == y).mean()), nats=nats, bits=nats / np.log(2), contexts=int(len(counts[K][0])))
    return out

def run(kw):
    t0 = time.process_time(); x, cut, S, pad = load(); D = int(kw.get("D", 10)); H = int(kw.get("H", 256)); soft = int(kw.get("arm", "P") == "P"); passes = int(kw.get("passes", 8))
    s = Sys(int(kw.get("seed", 1)), H, D, S, pad, eta1=float(eval(kw.get("eta1", "1"))), eta2=float(eval(kw.get("eta2", "1/16"))), lam=float(kw.get("lam", 1e-5)), rscale=float(kw.get("rscale", 1)), soft=soft)
    out = dict(label=kw["label"], args=kw, S=S, stream=cut, never_read=len(x) - cut, numbers_that_learn=dict(strengths_from_lines=H * D * S, thresholds=H, answering_cell=S * H + S, return_strengths=S * H), rows=[])
    xs = x[:cut - 1]; ys = x[1:cut]; fast = not int(kw.get("numpy", 0))
    for p in range(passes):
        s.loss = 0.0; s.right = 0; s.read(xs, ys, fast); r = score(s, x, cut, soft)
        if not np.isfinite(s.W2).all() or not np.isfinite(s.W1t).all(): r = dict(nats=float("inf"), bits=float("inf"), right=0.0)
        r.update(passes=p + 1, stream_nats=s.loss / len(xs), stream_right=100 * s.right / len(xs), cpu=round(time.process_time() - t0, 1)); out["rows"].append(r)
        json.dump(out, open(os.path.join(HERE, "res", f"{kw['label']}.json"), "w"))
        if not np.isfinite(r["nats"]): break
    out["return_and_forward_cosine"] = float((s.Rt * s.W2).sum() / (np.linalg.norm(s.Rt) * np.linalg.norm(s.W2) + 1e-300)); out["cpu"] = round(time.process_time() - t0, 1)
    json.dump(out, open(os.path.join(HERE, "res", f"{kw['label']}.json"), "w")); return out

if __name__ == "__main__":
    if sys.argv[1] == "tables":
        x, cut, S, pad = load(); t = tables(x, cut, S); f = np.bincount(x[:cut], minlength=S); p = f / f.sum(); y = x[cut:]
        base = dict(most_frequent_right=float(100 * (y == f.argmax()).mean()), frequencies_nats=float(-np.log(p[y]).mean()), uniform_nats=float(np.log(S)))
        print(f"stream {cut} bytes, never read {len(x) - cut} bytes, {S} byte values; most frequent byte {base['most_frequent_right']:.2f}%; byte frequencies {base['frequencies_nats']:.4f} nats; uniform {base['uniform_nats']:.4f} nats")
        for k, v in t.items(): print(f"table of the last {k} bytes, unsmoothed: {v['nats']:.4f} nats {v['bits']:.4f} bits (crude), {v['right']:.2f}% right, {v['contexts']} contexts in the stream")
        json.dump(dict(tables=t, **base), open(os.path.join(HERE, "res", "tables.json"), "w"))
    else:
        kw = dict(a.split("=") for a in sys.argv[3:]); kw["label"] = sys.argv[2]; o = run(kw)
        print(o["label"], "| " + " ".join(f"{r['passes']}: {r['nats']:.4f}n {r['right']:.2f}%" for r in o["rows"]), f"| cosine {o['return_and_forward_cosine']:.3f}", o["cpu"], "s", flush=True)
