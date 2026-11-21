#!/usr/bin/env python3
"""nano/lives, PLAN_11.md, learner 3: the relay network (nano/loop/loop.py's system, which is loop_small_case/cells2.py's with the alphabet as an argument)
through the same life: sums, English, sums again. One alphabet for both texts.
    python3 relay_aba.py seed=1 form=R|R0 [lines=400000] tag=..
R: as it is. R0: the fading of the answering strengths and of the return strengths is switched off from the English on.
With form R0 the return is also read by a copy in which the cells in between are held still (only the answering cell learns): a diagnostic."""
import copy, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "loop"))
import aba as AB, loop as LP

def run(kw):
    seed = int(kw.get("seed", 1)); form = kw["form"]; lines = int(kw.get("lines", 400000)); tag = kw["tag"]; D = 10; t0 = time.time()
    x, sp = AB.build(seed, lines); vals = np.unique(x); idx = np.zeros(256, np.int64); idx[vals] = np.arange(len(vals)); xm = idx[x].astype(np.int32); S = len(vals)
    s = LP.Sys(seed, 256, D, S, int(idx[10]), eta1=1.0, eta2=1 / 16, lam=1e-5, rscale=1.0, soft=0)      # the step 1 is the one that gives the recorded level on sums (PLAN_11.md, section 4a); ev = xm[sp["ev"]:].reshape(5, 500, 20); rows = []
    def never_shown(sy):
        out = []
        for d in range(5):
            prev = ev[d, :, :10]; Ls = ev[d, :, 10:]; ok = np.ones(500, bool)
            for t in (5, 6, 7):
                hist = np.zeros((500, D), np.int64); hist[:, :t + 1] = Ls[:, t::-1]; hist[:, t + 1:] = prev[:, ::-1][:, :D - t - 1]; ok &= sy.branches(hist).argmax(1) == Ls[:, t + 1]
            out.append(100 * ok.mean())
        return float(np.mean(out))
    def read(sy, lo, hi): sy.read(xm[lo - 1:hi - 1], xm[lo:hi])
    def mark(label, sy=None): v = never_shown(sy or s); rows.append(dict(at=label, whole=v)); print(tag, label, round(v, 1), flush=True)
    a0, b0 = sp["A"]; done = a0
    for n in sorted({40000, lines}):
        if n <= lines: read(s, done, a0 + n * 10); done = a0 + n * 10; mark(f"sums {n} lines")
    rows[-1]["at"] = "end of the sums"
    if form == "R0": s.lam = 0.0
    r0 = s.right; read(s, sp["B"][0], sp["B"][1]); english_right = 100 * (s.right - r0) / (sp["B"][1] - sp["B"][0]); mark("end of the English")
    held = copy.deepcopy(s) if form == "R0" else None; done = sp["A2"][0]
    for n in (500, 1000, 2000, 5000): hi = sp["A2"][0] + n * 10; read(s, done, hi); done = hi; mark(f"return {n} lines")
    if held is not None:
        held.eta1 = 0.0; done = sp["A2"][0]
        for n in (500, 1000, 2000, 5000): hi = sp["A2"][0] + n * 10; read(held, done, hi); done = hi; mark(f"return {n} lines, the cells in between held still", held)
    out = dict(task="relay_aba", tag=tag, seed=seed, form=form, lines=lines, values=S, rows=rows, english_bytes_named_right_while_reading=english_right, wall=round(time.time() - t0, 1)); json.dump(out, open(os.path.join(HERE, "res", f"relay_aba__{tag}.json"), "w")); return out

if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[1:]); out = run(kw); print(out["tag"], "| never-shown sums, whole answer:", " | ".join(f"{r['at']}: {r['whole']:.1f}" for r in out["rows"]), "| wall", out["wall"], flush=True)
