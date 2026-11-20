#!/usr/bin/env python3
"""PLAN_7.md (the coordinator's rule): a byte that leaves the expectation of the next byte unchanged does not enter the chain.
Beside the relay system a table of cells that count: a cell for each run of the last 1 to 4 bytes that has occurred, counting what follows it.
When the byte v arrives after the bytes c, the cell of the run (c, v) and the cell of the run c are compared, at the longest length (c, v) of 4, 3
or 2 bytes at which both have counted at least nmin times: half the sum of the differences of their relative counts (the total variation). Under
theta, v is passed over. No byte is named. The counts used for a block of 2,000 lines are those at the block's start (the table is numpy, by blocks).

    python3 expect.py crit [seed] [text]                 the criterion alone: the share of bytes passed over, by kind of byte
    python3 expect.py run <label> seed=1 theta=0.1 [text=g2 lines=1200000 every=100000 nmin=20]     the system of skip.py with this clock (compiled kernel)
"""
import json, os, sys, time
import numpy as np
import skip
K = 4
class Table:
    def __init__(s, nmin=20, theta=0.1): s.S = skip.S; s.C = [np.zeros((s.S ** L, s.S), np.float64) for L in range(K + 1)]; s.nmin, s.theta = nmin, theta
    def code(s, xs, pre, L, shift=0):
        """For each byte i of the block, the code of the run of L bytes ending at byte i - shift (pre: the K bytes before the block)."""
        x = np.concatenate([pre, xs]).astype(np.int64); n = len(xs); return sum(x[K - shift - j:K - shift - j + n] * s.S ** j for j in range(L)) if L else np.zeros(n, np.int64)
    def enters(s, xs, pre):
        """For each byte of the block: does it enter the chain (True) or is it passed over; and the total variation used (nan where no length had counts)."""
        tv = np.full(len(xs), np.nan)
        for L in (2, 3, 4):                               # the run (c, v) of L bytes ending at v, and the run c of L - 1 bytes ending one byte earlier
            A = s.C[L][s.code(xs, pre, L)]; B = s.C[L - 1][s.code(xs, pre, L - 1, 1)]; na = A.sum(1); nb = B.sum(1); ok = (na >= s.nmin) & (nb >= s.nmin)
            t = 0.5 * np.abs(A / np.maximum(na, 1)[:, None] - B / np.maximum(nb, 1)[:, None]).sum(1); tv = np.where(ok, t, tv)
        return ~(tv < s.theta), tv
    def count(s, xs, ys, pre):
        m = ys >= 0
        for L in range(1, K + 1): np.add.at(s.C[L], (s.code(xs, pre, L)[m], ys[m]), 1.0)

KINDS = ["a tens", "a units", "+", "letter", "b tens", "b units", "=", "c hundreds", "c tens", "c units", "end mark"]
def kind_of(xs):
    """The kind of each byte itself (g2)."""
    out = np.zeros(len(xs), np.int64); st = 0
    for i, x in enumerate(xs):
        if x == skip.END: out[i] = 10; st = 0; continue
        if st == 0: out[i] = 0; st = 1
        elif st == 1: out[i] = 1; st = 2
        elif st == 2: out[i] = 2; st = 3
        elif st == 3 and x in skip.LETS: out[i] = 3
        elif st == 3: out[i] = 4; st = 4
        elif st == 4: out[i] = 5; st = 5
        elif st == 5: out[i] = 6; st = 6
        else: out[i] = st + 1; st += 1
    return out

def pairs(seed):
    rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; return rng, P[:500], P[2000:]

def crit(seed, tx="g2"):
    out = {}
    for theta in (0.05, 0.1, 0.2):
        rng, held, shown = pairs(seed); T = Table(theta=theta); pre = np.full(K, skip.END); done = 0; res = {}
        for target in (10000, 100000, 400000):
            while done < target:
                xs, ys = skip.stream(tx, shown, 8, 2000, rng); T.count(xs, ys, pre); pre = xs[-K:]; done += 2000
            xs, ys = skip.stream(tx, shown, 8, 3000, np.random.default_rng([seed, 556])); ent, tv = T.enters(xs, np.full(K, skip.END)); kd = kind_of(xs)
            sh = [float((~ent)[kd == k].mean()) for k in range(len(KINDS))]; res[target] = sh
            print(f"theta {theta:4.2f} after {target:6d} lines, share passed over: " + "  ".join(f"{k}: {v:.2f}" for k, v in zip(KINDS, sh)))
        out[str(theta)] = res
    json.dump(dict(kinds=KINDS, share_passed_over=out), open(os.path.join(skip.HERE, "res", f"p7_crit_{tx}_s{seed}.json"), "w"))

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); tx = kw.get("text", "g2"); lines = int(kw.get("lines", 1200000)); every = int(kw.get("every", 100000)); rng, held, shown = pairs(seed)
    s = skip.Sk(seed); s.fast = 1; T = Table(nmin=int(kw.get("nmin", 20)), theta=float(kw.get("theta", 0.1))); pre = np.full(K, skip.END); done = 0; out = dict(label=kw["label"], args=kw, rows=[]); tot = np.zeros(len(KINDS)); psd = np.zeros(len(KINDS))
    def score(pp, gap):
        hists = []; want = []; own = []
        for k, (a, b) in enumerate(pp):
            L = skip.text(tx, a, b, skip.fill(a, b, gap)); ent, _ = T.enters(np.array(L), pre); eq = L.index(skip.EQ); hist = list(s.hist); pend = bool(s.pend)
            for t in range(len(L) - 1):
                p = hist[0]
                if pend: hist = [L[t], p] + hist[1:-1]
                else: hist[0] = L[t]
                pend = bool(ent[t])
                if t >= eq and L[t + 1] != skip.END: hists.append(list(hist)); want.append(L[t + 1]); own.append((k, t - eq))
        hists = np.array(hists); o = skip.sg(s.W1.T[skip.OFF + hists].sum(1) + s.b1) @ s.W2.T + s.b2; r = o.argmax(1) == np.array(want); ok = np.ones(len(pp), bool); pl = np.zeros(3)
        for (k, j), rr in zip(own, r): ok[k] &= bool(rr); pl[j] += rr
        return float(100 * ok.mean()), [int(round(100 * x / len(pp))) for x in pl]
    while done < lines:
        xs, ys = skip.stream(tx, shown, 8, 2000, rng); ent, _ = T.enters(xs, pre); s.run(xs, ys, ent); T.count(xs, ys, pre); kd = kind_of(xs); tot += np.bincount(kd, minlength=len(KINDS)); psd += np.bincount(kd, weights=~ent, minlength=len(KINDS)); pre = xs[-K:]; done += 2000
        if done % every == 0:
            row = dict(lines=done, never_by_gap={}, place={}, share_passed_over=dict(zip(KINDS, (psd / np.maximum(tot, 1)).round(3).tolist()))); tot[:] = 0; psd[:] = 0
            for g in (0, 4, 8): row["never_by_gap"][g], row["place"][g] = score(held, g)
            out["rows"].append(row)
    out["last3_by_gap"] = {g: float(np.mean([r["never_by_gap"][g] for r in out["rows"][-3:]])) for g in (0, 4, 8)}; out["cpu"] = round(time.process_time() - t0, 1)
    json.dump(out, open(os.path.join(skip.HERE, "res", f"{kw['label']}.json"), "w")); r = out["rows"][-1]
    print(out["label"], "| last3 by gap", out["last3_by_gap"], "|", " ".join(f"{q['lines']//1000}k " + "/".join(f"{v:.0f}" for v in q["never_by_gap"].values()) for q in out["rows"]), "| place", r["place"], "| share passed over", r["share_passed_over"], "|", out["cpu"], "s")

if __name__ == "__main__":
    if sys.argv[1] == "crit": crit(int(sys.argv[2]) if len(sys.argv) > 2 else 1, sys.argv[3] if len(sys.argv) > 3 else "g2")
    else: kw = dict(a.split("=") for a in sys.argv[3:]); kw["label"] = sys.argv[2]; run(kw)
