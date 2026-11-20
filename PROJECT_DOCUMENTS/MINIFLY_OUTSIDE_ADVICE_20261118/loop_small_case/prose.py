#!/usr/bin/env python3
"""PLAN_10.md, task A: the next byte of real prose, by the system of cells2.py (compiled kernel) beside tables of the last k bytes.
    python3 prose.py <label> [file=<path>] [arm=relay|comparator|fixed|none] [H=256] [D=10] [eta1=1/4] [passes=8] [seed=1]
    python3 prose.py tables [file=<path>]
The first nine tenths of the text are the stream, read `passes` times; the last tenth is never read and is what the score is taken on."""
import json, os, re, sys, time
import numpy as np
import cells2
HERE = os.path.dirname(os.path.abspath(__file__)); KEEP = "abcdefghijklmnopqrstuvwxyz .,;:!?'\"-"
DEFAULT = os.path.join(HERE, "..", "..", "..", "minifly_v148_pretrain_probe_dev_20261025", "data", "bryant-stories.txt")

def load(path):
    t = re.sub(r"\s+", " ", open(path, "rb").read().decode("latin-1").lower()); idx = {c: i for i, c in enumerate(KEEP)}
    x = np.array([idx.get(c, len(KEEP)) for c in t], np.int32); cut = len(x) * 9 // 10; return x[:cut], x[cut:], len(KEEP) + 1

def setup(S, D):
    cells2.S = S; cells2.D = D; cells2.NL = D * S; cells2.OFF = np.arange(D) * S; cells2.END = KEEP.index(" ")

def windows(x, D, pad):
    """For every position t of x but the last: the D bytes ending at t, newest first; before the start, the pad byte."""
    z = np.concatenate([np.full(D - 1, pad, np.int32), x]); return np.stack([z[D - 1 - d: len(z) - 1 - d] for d in range(D)], 1), x[1:]

def score(branches, test, D, pad, S):
    """branches(hist) -> the answering cell's branches for many positions at once."""
    hist, y = windows(test, D, pad); right = 0; bits = 0.0
    for i in range(0, len(y), 4000):
        a = branches(hist[i:i + 4000]); yy = y[i:i + 4000]; right += int((a.argmax(1) == yy).sum())
        p = np.maximum(a, 1e-3); p /= p.sum(1, keepdims=True); bits += float(-np.log2(p[np.arange(len(yy)), yy]).sum())
    return 100 * right / len(y), bits / len(y)

def tables(train, test, S, kmax=6):
    out = {}; counts = []
    for k in range(0, kmax + 1):                         # counts of the next byte after each context of k bytes
        key = np.zeros(len(train) - k, np.int64)
        for d in range(k): key = key * S + train[d: len(train) - k + d]
        c = {}; nxt = train[k:]
        for kk, n in zip(key.tolist(), nxt.tolist()): c.setdefault(kk, np.zeros(S, np.int32))[n] += 1
        counts.append(c)
    for K in range(1, kmax + 1):
        right = 0; n = 0
        for t in range(kmax, len(test) - 1):
            for k in range(K, -1, -1):                   # the longest context seen in the stream decides
                key = 0
                for d in range(k): key = key * S + int(test[t - k + 1 + d])
                c = counts[k].get(key)
                if c is not None: break
            right += int(c.argmax()) == int(test[t + 1]); n += 1
        out[K] = 100 * right / n
    return out

def run(kw):
    t0 = time.process_time(); train, test, S = load(kw.get("file", DEFAULT)); D = int(kw.get("D", 10)); setup(S, D); arm = kw.get("arm", "relay"); passes = int(kw.get("passes", 8))
    H = int(kw.get("H", 256)); eta1 = float(eval(kw.get("eta1", "1/4"))); eta2 = float(eval(kw.get("eta2", "1/16"))); lam = float(kw.get("lam", 1e-5)); pad = cells2.END; out = dict(label=kw["label"], args=kw, rows=[], S=S, stream=len(train), never_read=len(test))
    if arm == "none":                                    # the answering cell reads the lines themselves
        W = np.zeros((S, D * S)); b = np.zeros(S); hist, y = windows(train, D, pad); act = hist + cells2.OFF; g = eta2 / (D + 1)
        for p in range(passes):
            for t in range(len(y)):
                m = -(W[:, act[t]].sum(1) + b); m[y[t]] += 1; W[:, act[t]] += g * m[:, None]; b += g * m
            acc, bits = score(lambda h: W.T[h + cells2.OFF].sum(1) + b, test, D, pad, S); out["rows"].append(dict(passes=p + 1, right=acc, bits=bits))
    else:
        s = cells2.Sys(int(kw.get("seed", 1)), H=H, eta1=0.0 if arm == "fixed" else eta1, eta2=eta2, lam=lam); s.fast = int(kw.get("fast", 3)); s.hist = [pad] * D
        if arm == "comparator": s.R[:] = 0.0             # return strengths that start at nothing equal the forward strengths from then on
        xs = train[:-1]; ys = train[1:]
        for p in range(passes):
            s.bx = xs; s.by = ys; s.flush()
            acc, bits = score(lambda h: s.out(s.W1.T[h + cells2.OFF].sum(1) + s.b1), test, D, pad, S); out["rows"].append(dict(passes=p + 1, right=acc, bits=bits))
        out["return_and_forward_cosine"] = float((s.R * s.W2.T).sum() / (np.linalg.norm(s.R) * np.linalg.norm(s.W2) + 1e-300))
    out["cpu"] = round(time.process_time() - t0, 1); os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"prose_{kw['label']}.json"), "w")); return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]
    if sys.argv[1] == "tables":
        train, test, S = load(kw.get("file", DEFAULT)); t = tables(train, test, S); freq = np.bincount(train, minlength=S)
        print(f"stream {len(train)} bytes, never read {len(test)} bytes, {S} byte values; the most frequent byte: {100 * (test == freq.argmax()).mean():.1f}%")
        print("table of the last k bytes (longest context seen decides), next byte right on the text never read: " + ", ".join(f"k={k}: {v:.1f}%" for k, v in t.items()))
        json.dump(dict(tables=t, most_frequent=float(100 * (test == freq.argmax()).mean())), open(os.path.join(HERE, "res", "prose_tables.json"), "w"))
    else:
        o = run(kw); print(o["label"], "| " + " ".join(f"{r['passes']}: {r['right']:.1f}% {r['bits']:.2f}b" for r in o["rows"]), f"| cosine {o.get('return_and_forward_cosine', float('nan')):.3f}", o["cpu"], "s")
