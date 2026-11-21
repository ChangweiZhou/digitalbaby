# A look on a system that exists (nano/merge, frozen setting, development seed 1, 40,000 lines): how sure is the answering cell where it is wrong on never-shown sums,
# and how sure is it on the lines it reads (met before or not)?  No mechanism is changed.
import sys, json, itertools, numpy as np
import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '../merge')); OUT = os.path.join(HERE, "res"); os.makedirs(OUT, exist_ok=True)
import merge as G
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1; lines = 40000; D = 9
rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); pairs = [(int(q) // 100, int(q) % 100) for q in o]; held = pairs[:500]; shown = pairs[2000:]; rng.permutation(10000)
TAB = np.array([G.text(a, b) for a, b in shown], np.int64)
kinds = [c for m in range(5) for c in itertools.combinations(range(1, D + 1), m)]
s = G.Sys(14, D, kinds, 0.1, 2.0, 0.02, True, P=32, eJ=0.02, mode="classes", seed=seed, half_at=lines * 7 * 5 // 10, M=16, sweeps=2)
tail = np.full(D, G.END, np.int64); done = 0; seen = np.zeros(8000, np.int64); rec = []          # rec: (lines read, met before, position, max p, p of the true byte, right)
for mk in range(5000, lines + 1, 5000):
    idx = rng.integers(8000, size=mk - done); X = TAB[idx]
    # read line by line in blocks of 500 lines so that the answer the system would give is looked at before the block is counted
    for a in range(0, len(idx), 250):
        blk = X[a:a + 250]; ii = idx[a:a + 250]; x = np.concatenate([tail, blk.ravel()]); ctx = G.ctx_of(x, D, len(x), D); P = s.say(ctx).reshape(len(blk), 10, 14)
        met = seen[ii] > 0
        for pos in (6, 7, 8):
            pm = P[:, pos].max(1); py = P[np.arange(len(blk)), pos, blk[:, pos]]; ok = P[:, pos].argmax(1) == blk[:, pos]
            rec.append(np.stack([np.full(len(blk), done + a), met, np.full(len(blk), pos), pm, py, ok], 1))
        s.learn(ctx, x[D:]); tail = x[-D:]; np.add.at(seen, ii, 1)
    done = mk
    print(mk, G.sums_score(s, held, D, 2), flush=True)
rec = np.concatenate(rec); pass
# never-shown sums at the end: the system's own answer bytes
Ls = np.array([G.text(a, b) for a, b in held]); own = Ls.copy(); out = []
for t in (6, 7, 8):
    ctx = np.full((500, D), G.END, np.int64)
    for d in range(1, D + 1):
        if t - d >= 0: ctx[:, d - 1] = own[:, t - d]
    P = s.say(ctx); own[:, t] = P.argmax(1); out.append(np.stack([np.full(500, t), P.max(1), P[np.arange(500), Ls[:, t]], own[:, t] == Ls[:, t]], 1))
out = np.concatenate(out); pass
bins = [0, .5, .6, .7, .8, .9, .95, .99, 1.01]
def table(name, pm, ok):
    print(name, "n", len(pm), "right %", round(100 * ok.mean(), 1))
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pm >= lo) & (pm < hi)
        if m.sum(): print(f"   max p in [{lo}, {hi}): {int(m.sum()):6d} moments, wrong {int((~ok[m].astype(bool)).sum()):5d}")
for t in (6, 7, 8): m = out[:, 0] == t; table(f"never-shown, at the end, answer byte {t - 5}", out[m, 1], out[m, 3])
late = rec[rec[:, 0] >= 20000]
for met in (0, 1):
    for t in (6, 7, 8): m = (late[:, 1] == met) & (late[:, 2] == t); table(f"lines read after 20,000, {'met before' if met else 'first meeting'}, answer byte {t - 5}", late[m, 3], late[m, 5]); print("      mean of 1 - p(true):", round(float((1 - late[m, 4]).mean()), 4), " sum:", round(float((1 - late[m, 4]).sum()), 1))
