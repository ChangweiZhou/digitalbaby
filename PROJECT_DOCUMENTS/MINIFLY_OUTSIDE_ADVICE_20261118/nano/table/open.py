#!/usr/bin/env python3
"""PLAN_18.md: is the fall with long reading what a closed stream teaches? The system is nano/merge's with the two answering cells of PLAN_15.md, unchanged.
    python3 open.py seed=73129 [mode=none|classes] [lines=400000] [q=0.01] [start=40000]
q: from line `start` on, each line is with chance q a pair that is neither among the 8,000 shown pairs nor among the 500 scored ones (drawn from the 1,500 others).
   With q=0 the stream, the system and the readings are gate.py's to the digit.
At every reading, beside the usual scores (never-shown additions, the system's own answer bytes, after a read line and with an empty past, both cells), one more
score with nothing learned and nothing kept: every cell that has met three or more different followers is made to say of the followers it has not met what it
says on average of those it has met (a diagnostic for the account of PLAN_18.md, not a mechanism)."""
import itertools, json, os, sys, time
import numpy as np
import gate
from gate import G, Sys2
HERE = os.path.dirname(os.path.abspath(__file__)); D = 9

def run(kw):
    seed = int(kw.get("seed", 1)); lines = int(kw.get("lines", 400000)); mode = kw.get("mode", "none"); q = float(kw.get("q", 0)); start = int(kw.get("start", 40000)); t0 = time.process_time()
    Sys2.GATE = 0; Sys2.BLIND = 0; G.Sys = Sys2
    rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); pairs = [(int(v) // 100, int(v) % 100) for v in o]; held = pairs[:500]; rest = pairs[500:2000]; shown = pairs[2000:]; rng.permutation(10000)
    TAB = np.array([G.text(a, b) for a, b in shown], np.int64); NEW = np.array([G.text(a, b) for a, b in rest], np.int64); Ls = np.array([G.text(a, b) for a, b in held])
    g = np.random.default_rng([seed, 7]); prevs = [TAB[g.integers(8000, size=500)] for _ in range(5)]; rt = np.random.default_rng([seed, 31])
    kinds = [c for m in range(5) for c in itertools.combinations(range(1, D + 1), m)]
    s = G.Sys(14, D, kinds, 0.1, 2.0, 0.02, True, P=32, eJ=0.02, mode=mode, seed=seed, half_at=lines * 7 * 5 // 10, M=16, sweeps=2, rep=0, dissolve=0, pool=None, practice=0.0, empty=0)
    def one(prev):
        own = Ls.copy()
        for t in (6, 7, 8):
            ctx = np.full((500, D), G.END, np.int64)
            for d in range(1, D + 1):
                if t - d >= 0: ctx[:, d - 1] = own[:, t - d]
                elif prev is not None: ctx[:, d - 1] = prev[:, 10 + t - d]
            own[:, t] = s.say(ctx).argmax(1)
        ok = own[:, 6:9] == Ls[:, 6:9]; return float(100 * ok.all(1).mean()), [float(100 * ok[:, i].mean()) for i in range(3)]
    def both():
        r = {}
        for cell in (1, 2):
            s.cell = cell; e = one(None); a = [one(pv) for pv in prevs]; r[f"cell{cell}"] = dict(empty=e[0], after=float(np.mean([v[0] for v in a])), after_place=np.mean([v[1] for v in a], 0).tolist())
        s.cell = 2; return r
    def amnesty():
        """The same scores with every cell of three or more met followers saying of the unmet ones what it says on average of the met ones; then everything is put back."""
        top = s.top; C = s.C[:top]; N = s.N[:top]; keepC = C.copy(); keepN = N.copy(); m = np.flatnonzero(s.U[:top] >= 3); Cm = C[m]; seen = Cm > 0
        mean = Cm.sum(1) / np.maximum(seen.sum(1), 1); Cm = np.where(seen, Cm, mean[:, None]).astype(np.float32); C[m] = Cm; N[m] = Cm.sum(1); r = both(); C[:] = keepC; N[:] = keepN; return r
    every = 50000 if lines > 40000 else 5000; marks = sorted(set(list(range(every, lines + 1, every)) + [lines] + ([5000, 10000, 20000, 40000] if lines > 40000 else []))); tail = np.full(D, G.END, np.int64); done = 0; readings = []; novel = 0
    for mk in marks:
        k = mk - done; X = TAB[rng.integers(8000, size=k)]
        if q > 0:
            m = (np.arange(done, mk) >= start) & (rt.random(k) < q); X[m] = NEW[rt.integers(len(NEW), size=int(m.sum()))]; novel += int(m.sum())
        x = np.concatenate([tail, X.ravel()]); s.learn(G.ctx_of(x, D, len(x), D), x[D:]); tail = x[-D:]; done = mk
        r = both(); r["amnesty"] = amnesty(); r["lines"] = mk; r["novel_lines_so_far"] = novel; readings.append(r)
    out = dict(task="open", args=kw, seed=seed, readings=readings, sizes=s.sizes(), novel_lines=novel, cpu=round(time.process_time() - t0, 1))
    for cell in ("cell1", "cell2"):
        at40 = next(r for r in readings if r["lines"] == min(40000, lines))
        out[cell] = dict(at_40000=at40[cell]["after"], last3_after=float(np.mean([r[cell]["after"] for r in readings[-3:]])), last_after=readings[-1][cell]["after"],
                         amnesty_at_40000=at40["amnesty"][cell]["after"], amnesty_last3_after=float(np.mean([r["amnesty"][cell]["after"] for r in readings[-3:]])))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:]); out = run(kw); tag = "open_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items())); json.dump(out, open(os.path.join(HERE, "res", tag + ".json"), "w"))
    print(tag, "| novel lines", out["novel_lines"], "| first cell: at 40,000", round(out["cell1"]["at_40000"], 1), "last three", round(out["cell1"]["last3_after"], 1), "; with the diagnostic", round(out["cell1"]["amnesty_at_40000"], 1), round(out["cell1"]["amnesty_last3_after"], 1),
          "| second cell: at 40,000", round(out["cell2"]["at_40000"], 1), "last three", round(out["cell2"]["last3_after"], 1), "; with the diagnostic", round(out["cell2"]["amnesty_at_40000"], 1), round(out["cell2"]["amnesty_last3_after"], 1),
          "| readings (lines: first, second | diagnostic first, second):", "  ".join(f"{r['lines'] // 1000}k {r['cell1']['after']:.1f} {r['cell2']['after']:.1f} | {r['amnesty']['cell1']['after']:.1f} {r['amnesty']['cell2']['after']:.1f}" for r in out["readings"]), "| cpu", out["cpu"], flush=True)
