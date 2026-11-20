# Fourth look: the relay system of loop_small_case/cells2.py (unchanged), addition; the never-shown sums scored as cells2.py scores them (the bytes before the line are line ends)
# and scored after a line the system has read (the true bytes before the answer byte in both, as cells2.py does).  No mechanism is changed.
import sys, json, numpy as np
import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '.')); OUT = os.path.join(HERE, "res"); os.makedirs(OUT, exist_ok=True)
import cells2 as C
seed = int(sys.argv[1]); lines = int(sys.argv[2]); kw = dict(label="past", seed=str(seed), plan=f"add:{lines}", fast="3", nowrite="1", every=str(lines // 8))
out = C.run(kw); s = C.run.last; s.flush()
rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held = P[:500]; shown = P[2000:]
TAB = np.array([C.text("+", a, b) for a, b in shown]); Ls = np.array([C.text("+", a, b) for a, b in held]); W1T = s.W1.T; D = C.D
def score(prev):
    ok = np.ones(500, bool); pl = []
    for t in (5, 6, 7):
        hist = np.full((500, D), C.END); hist[:, :t + 1] = Ls[:, t::-1]
        if prev is not None: hist[:, t + 1:] = prev[:, ::-1][:, :D - t - 1]
        r = s.out(W1T[C.OFF + hist].sum(1) + s.b1).argmax(1) == Ls[:, t + 1]; ok &= r; pl.append(float(100 * r.mean()))
    return float(100 * ok.mean()), pl
g = np.random.default_rng([seed, 7]); e = score(None); p = [score(TAB[g.integers(8000, size=500)]) for _ in range(5)]
res = dict(seed=seed, lines=lines, rows=[(r["lines"], r["add"]) for r in out["rows"]], empty_past=e, after_a_read_line=p, after_mean=float(np.mean([q[0] for q in p])), cpu=out["cpu"])
print(f"relay system, seed {seed}, {lines} lines | cells2's own readings {[(r['lines'] // 1000, r['add']) for r in out['rows']]} | empty past {e[0]:.1f} {['%.1f' % v for v in e[1]]} | after a read line (five draws) {['%.1f' % q[0] for q in p]} mean {res['after_mean']:.2f} places {['%.1f' % v for v in np.mean([q[1] for q in p], 0)]} | cpu {out['cpu']}", flush=True)
json.dump(res, open(os.path.join(OUT, f"past_seed{seed}_{lines}.json"), "w"))
