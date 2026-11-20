#!/usr/bin/env python3
"""PLAN_4.md (line C, step C5): the plain system (form 1 of joint.py, which reproduces loop_small_case/cells2.py) with the rule by which a strength
is changed replaced, one rule at a time; the edge stays one-sided.
    python3 joint5.py <label> seed=1 [ans=1|2|3|4|5] [p=x] [cel=1|3|4] [q=x] [lines=400000] [every=50000] [fast=1] [H=256]
    The answering cell's strengths A[v, j] (m: mismatch of branch v; h: output of cell j; g = eta2 / (h.h + 1)):
      ans=1: A += g m h                     (the present rule)
      ans=2: A += p m h                     (no division; p is the fixed step)
      ans=3: A += p sign(m h)               (a change of fixed size p in the direction of the product)
      ans=5: A += p sign(m (h - 1/2))       (an addition of mine: the same with the sender's output taken about the middle of its range)
      ans=4: A += g m h (|A| + p) / (1/64 + p)      (in proportion to the strength's own size; 1/64 is the mean |A| of the plain system)
    The return strength R[j, v] receives the same change as A[v, j] in every arm; the constants b2 change by g m in every arm; fading as before.
    The cells in between (sig_j = (R[j] . m) h_j (1 - h_j); the strengths W1[j, l] from the ten active lines and the threshold b1[j]):
      cel=1: W1 += eta1 sig; b1 += eta1 sig         (the present rule)
      cel=3: W1 += q sign(sig); b1 += q sign(sig)   (a change of fixed size q)
      cel=4: W1 += eta1 sig (|W1| + q) / (TYPW + q); b1 += eta1 sig      (in proportion to the strength's own size)
    fast=1: the inner loop runs compiled (kern5.c through fast5.py); the numpy loop here is the reference (check_joint5.py)."""
import json, os, sys, time
import numpy as np
import joint
from joint import HERE, S, D, OFF, text, sg
TYP = 1 / 64; TYPW = 0.25

class Sys5(joint.Sys):
    def __init__(s, seed, ans=1, p=0.0, cel=1, q=0.0, **kw):
        super().__init__(seed, form=1, **kw); s.ans, s.p, s.cel, s.q = ans, p, cel, q
    def learn(s, y):
        act = OFF + np.asarray(s.hist); h = sg(s.W1[:, act].sum(1) + s.b1); a = s.A @ h + s.b2; m = -a; m[y] += 1; g = s.eta2 / (h @ h + 1)
        sig = (s.R @ m) * h * (1 - h)
        if s.ans == 1: d2 = np.outer(g * m, h)
        elif s.ans == 2: d2 = np.outer(s.p * m, h)
        elif s.ans == 3: d2 = s.p * np.sign(np.outer(m, h))
        elif s.ans == 5: d2 = s.p * np.sign(np.outer(m, h - .5))
        else: d2 = np.outer(g * m, h) * (np.abs(s.A) + s.p) / (TYP + s.p)
        if s.cel == 1: d = s.eta1 * sig; s.W1[:, act] += np.repeat(d[:, None], D, 1)
        elif s.cel == 3: d = s.q * np.sign(sig); s.W1[:, act] += np.repeat(d[:, None], D, 1)
        else: d = s.eta1 * sig; s.W1[:, act] += d[:, None] * (np.abs(s.W1[:, act]) + s.q) / (TYPW + s.q)
        s.R += d2.T; s.b1 += d; s.A += d2
        s.b2 += g * m; s.A *= 1 - s.lam; s.R *= 1 - s.lam

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng = np.random.default_rng([seed, 99]); held, shown = {}, {}
    for op in "+-":
        o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = P[:500]; shown[op] = P[2000:]
    s = Sys5(seed, ans=int(kw.get("ans", 1)), p=float(eval(kw.get("p", "0"))), cel=int(kw.get("cel", 1)), q=float(eval(kw.get("q", "0"))), H=int(kw.get("H", 256)))
    n = int(kw.get("lines", 400000)); every = int(kw.get("every", 50000)); fast = int(kw.get("fast", 0)); out = dict(label=kw.get("label"), args=kw, rows=[])
    TAB = np.array([text("+", a, b) for a, b in shown["+"]], np.int32); it = 0
    while it < n:
        nxt = min((it // every + 1) * every, n) - 1; X = TAB[rng.integers(8000, size=nxt - it + 1)]
        xs = X.ravel(); ys = np.concatenate([X[:, 1:], np.full((len(X), 1), -1, np.int32)], 1).ravel(); it = nxt + 1
        if fast: import fast5; fast5.run(s, xs, ys)
        else: s.steps(xs, ys)
        if (nxt + 1) % every == 0: out["rows"].append(dict(lines=nxt + 1, add=s.score(held["+"])))
    out["last3"] = float(np.mean([r["add"] for r in out["rows"][-3:]])) if out["rows"] else None
    out["finite"] = bool(np.isfinite(s.A).all() and np.isfinite(s.W1).all()); aw = np.abs(s.A)
    out["m"] = dict(end=s.score(held["+"]), largest=float(aw.max()), mean_abs_A=float(aw.mean()), share_below_hundredth=float((aw < .01 * aw.max()).mean()), mean_abs_W1=float(np.abs(s.W1).mean()),
                    cos_R_w=float((s.R.T * s.A).sum() / (np.linalg.norm(s.R) * np.linalg.norm(s.A) + 1e-300)))
    out["cpu"] = round(time.process_time() - t0, 1); run.last = s
    if not kw.get("nowrite"):
        os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"s_{kw['label']}.json"), "w"))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw); m = o["m"]
    print(o["label"], "last3", round(o["last3"], 1), "| " + " ".join(f"{r['lines']//1000}k {r['add']:.0f}" for r in o["rows"]),
          f"| end {m['end']:.1f} largest {m['largest']:.3g} mean |A| {m['mean_abs_A']:.3g} below a hundredth {100*m['share_below_hundredth']:.1f}% mean |W1| {m['mean_abs_W1']:.3f} cos {m['cos_R_w']:.3f} finite {o['finite']}", o["cpu"], "s")
