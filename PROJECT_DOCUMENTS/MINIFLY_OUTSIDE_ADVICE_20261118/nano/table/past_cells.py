# A second look on the same system (nano/merge, frozen setting, 40,000 lines): the never-shown sums scored as every line so far scores them (an empty past: the bytes before the
# line are line ends) and scored after a line that the system has read (the past it always has when it reads).  The system's own answer bytes in both.  No mechanism is changed.
import sys, json, itertools, numpy as np
import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '../merge')); OUT = os.path.join(HERE, "res"); os.makedirs(OUT, exist_ok=True)
import merge as G
seed = int(sys.argv[1]); mode = sys.argv[2] if len(sys.argv) > 2 else "classes"; lines = 40000; D = 9
out, s = G.run_sums(dict(seed=str(seed), lines=str(lines), mode=mode))
rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); pairs = [(int(q) // 100, int(q) % 100) for q in o]; held = pairs[:500]; shown = pairs[2000:]
TAB = np.array([G.text(a, b) for a, b in shown], np.int64); Ls = np.array([G.text(a, b) for a, b in held])
def score(prev):
    """prev: None (empty past) or an array (500, 10) of lines read before each held line."""
    own = Ls.copy(); n = len(Ls)
    for t in (6, 7, 8):
        ctx = np.full((n, D), G.END, np.int64)
        for d in range(1, D + 1):
            if t - d >= 0: ctx[:, d - 1] = own[:, t - d]
            elif prev is not None: ctx[:, d - 1] = prev[:, 10 + t - d]
        own[:, t] = s.say(ctx).argmax(1)
    ok = own[:, 6:9] == Ls[:, 6:9]; return float(100 * ok.all(1).mean()), [float(100 * ok[:, i].mean()) for i in range(3)]
g = np.random.default_rng([seed, 7]); prevs = [TAB[g.integers(8000, size=500)] for _ in range(5)]
def both(name):
    e = score(None); p = [score(pv) for pv in prevs]; res[name] = dict(empty_past=e, after_a_read_line=p, after_mean=float(np.mean([q[0] for q in p])), after_place_mean=np.mean([q[1] for q in p], 0).tolist())
    print(f"seed {seed} {mode} | {name}: empty past {e[0]:.1f} {['%.1f' % v for v in e[1]]} | after a read line (five draws) {['%.1f' % q[0] for q in p]} mean {res[name]['after_mean']:.2f} places {['%.1f' % v for v in res[name]['after_place_mean']]}", flush=True)
res = dict(seed=seed, mode=mode, merge_last=out["last"], merge_last3=out["last3"]); both("all kinds")
if s.P:
    keep = s.N.copy(); s.N[:s.nbase] = 0; both("class kinds alone"); s.N[:] = keep; s.N[s.nbase:] = 0; both("plain kinds alone"); s.N[:] = keep
json.dump(res, open(os.path.join(OUT, f"past_cells_seed{seed}_{mode}.json"), "w"))
