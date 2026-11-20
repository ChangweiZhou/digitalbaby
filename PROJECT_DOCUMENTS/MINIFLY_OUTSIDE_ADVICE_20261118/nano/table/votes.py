# Third look, same system (nano/merge frozen setting, seed given, 40,000 lines): at the tens byte of never-shown sums answered after a read line, which senders carry the vote?
# For each sender k: trust times (its message for the byte the system gave minus its message for the true byte).  No mechanism is changed.
import sys, json, itertools, numpy as np
import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '../merge')); OUT = os.path.join(HERE, "res"); os.makedirs(OUT, exist_ok=True)
import merge as G
seed = int(sys.argv[1]); lines = 40000; D = 9
out, s = G.run_sums(dict(seed=str(seed), lines=str(lines), mode="classes"))
rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); pairs = [(int(q) // 100, int(q) % 100) for q in o]; held = pairs[:500]; shown = pairs[2000:]
TAB = np.array([G.text(a, b) for a, b in shown], np.int64); Ls = np.array([G.text(a, b) for a, b in held]); g = np.random.default_rng([seed, 7]); prev = TAB[g.integers(8000, size=500)]
own = Ls.copy(); n = 500; names = [str(k) for k in s.kinds] + [f"class{tuple(sl['X'])}|{tuple(sl['Z'])} of {s.kinds[sl['k']]}" if sl is not None else "-" for sl in s.slots]
def senders(ctx):
    ids = s.ids(ctx); K1 = ids.shape[1]; Sm = np.zeros((len(ctx), K1, s.S)); Tr = np.zeros((len(ctx), K1)); act = np.zeros((len(ctx), K1), bool); Nn = np.zeros((len(ctx), K1)); Uu = np.zeros((len(ctx), K1))
    for i in range(len(ctx)):
        for k in range(K1):
            c = ids[i, k]
            if c >= 0 and s.N[c] >= 1:
                l = G.level_np(s.N[c], s.U[c]); Sm[i, k] = np.log(s.C[c].astype(np.float64) + s.e) - np.log(s.N[c] + s.S * s.e); Tr[i, k] = s.T[k, l]; act[i, k] = True; Nn[i, k] = s.N[c]; Uu[i, k] = s.U[c]
    return Sm, Tr, act, Nn, Uu
for t in (6, 7):
    ctx = np.full((n, D), G.END, np.int64)
    for d in range(1, D + 1): ctx[:, d - 1] = own[:, t - d] if t - d >= 0 else prev[:, 10 + t - d]
    P = s.say(ctx); own[:, t] = P.argmax(1); Sm, Tr, act, Nn, Uu = senders(ctx); a = (Tr[:, :, None] * Sm).sum(1); assert (a.argmax(1) == own[:, t]).all()
    wrong = np.flatnonzero(own[:, t] != Ls[:, t]); print(f"\nseed {seed}, answer byte {t - 5}: {len(wrong)} wrong of {n} (after a read line)")
    if not len(wrong): continue
    # for the wrong moments: each sender's vote for the given byte against the true byte
    vote = np.array([Tr[i] * (Sm[i, :, own[i, t]] - Sm[i, :, Ls[i, t]]) for i in wrong]); tot = vote.sum(0); KB = s.KB
    print("  sum of votes for the wrong byte against the true one, over the wrong moments: plain kinds %.1f (those reaching the line before, delay 8 or 9 at this byte: %.1f), class kinds %.1f" % (
        tot[:KB].sum(), sum(tot[k] for k in range(KB) if any(d > t + 1 for d in s.kinds[k])), tot[KB:].sum()))
    order = np.argsort(-tot)
    print("  the ten kinds that vote most for the wrong byte:"); [print(f"     {names[k]:40s} vote {tot[k]:7.2f}  speaking at {int(act[wrong, k].sum())} of {len(wrong)}, mean trust {Tr[wrong, k][act[wrong, k]].mean():.3f}, median counts {np.median(Nn[wrong, k][act[wrong, k]]):.0f}, median followers {np.median(Uu[wrong, k][act[wrong, k]]):.0f}") for k in order[:10]]
    print("  the ten kinds that vote most for the true byte:"); [print(f"     {names[k]:40s} vote {tot[k]:7.2f}  speaking at {int(act[wrong, k].sum())} of {len(wrong)}, mean trust {Tr[wrong, k][act[wrong, k]].mean():.3f}, median counts {np.median(Nn[wrong, k][act[wrong, k]]):.0f}, median followers {np.median(Uu[wrong, k][act[wrong, k]]):.0f}") for k in order[::-1][:10]]
    # by number of delays and by whether the kind reaches the line before
    for m in range(5):
        ks = [k for k in range(KB) if len(s.kinds[k]) == m]; print(f"  plain kinds of {m} delays: {len(ks)} kinds, vote {tot[ks].sum():.1f}")
    ok = np.flatnonzero(own[:, t] == Ls[:, t]); 
    # the margin at right moments: how much do class kinds and plain kinds each give
    second = np.array([np.partition(a[i], -2)[-2] for i in ok]); print(f"  right moments: {len(ok)}; median margin of the answer over the next byte {np.median(a[ok, own[ok, t]] - second):.2f}")
