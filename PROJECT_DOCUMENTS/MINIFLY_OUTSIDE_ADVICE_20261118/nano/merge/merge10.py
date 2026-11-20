#!/usr/bin/env python3
"""PLAN_10.md: kinds that the system makes while reading. As setting C2 (merge6.py with keep=2), with one addition: at each finding, a kind of largest trust
is refined by one more delay. Which refinements are made is decided on the system's own last W bytes: the worth of (R with one more delay) is how well its cells
foretell those bytes, each left out of its own cell, less the same for R. The G of largest positive worth are made (or kept, with their cells and trusts, if they are
there already); a new one starts with no cells and no trust. A made kind is a kind like any other: it speaks, it can be refined again, and classes are sought in it.
    python3 merge10.py check
    python3 merge10.py sums|dev|life  key=value ...      (the arguments of merge.py; keep=2 is set; G=16)"""
import json, os, sys, time
import numpy as np
import merge, merge6
from merge import code
from merge6 import loo
HERE = os.path.dirname(os.path.abspath(__file__))

class Sys10(merge6.Sys6):
    G = 16
    def __init__(s, *a, **k):
        super().__init__(*a, **k)
        if not s.P: s.G = 0
        s.GOFF = s.K1; s.K1 += s.G; assert s.K1 <= 400
        s.T = np.vstack([s.T, np.zeros((s.G, merge.NL))]); s.Tu = np.vstack([s.Tu, np.zeros((s.G, merge.NL))]); s.base_set = set(s.kinds); s.nbase0 = s.nbase
        s.kinds = list(s.kinds) + [()] * (s.K1 - s.KB)             # a kind's number is its column; the columns of class kinds hold no delays
        s.grown = [None] * s.G; s.finds_g = []
    # ---- addresses
    def base_ids(s, ctx):
        out = np.empty((len(ctx), s.KB), np.int32)
        for k in range(s.KB):
            c = code(ctx, s.kinds[k], s.S)
            if s.uniq[k] is None: out[:, k] = c + s.off[k]
            else:
                u = s.uniq[k]; j = np.searchsorted(u, c); j[j == len(u)] = 0; out[:, k] = np.where(u[j] == c, j + s.off[k], -1)
        return out
    def ids(s, ctx, create=False):
        a = super().ids(ctx, create)
        if not s.G: return a
        if a.shape[1] < s.GOFF: a = np.hstack([a, np.full((len(ctx), s.GOFF - a.shape[1]), -1, np.int32)])
        out = np.full((len(ctx), s.G), -1, np.int32)
        for g, sl in enumerate(s.grown):
            if sl is None: continue
            key = code(ctx, sl["ds"], s.S)
            if create:
                new = np.setdiff1d(key, sl["keys"])
                if len(new):
                    s.ensure(len(new)); idn = np.arange(s.top, s.top + len(new), dtype=np.int32); s.top += len(new)
                    kk = np.concatenate([sl["keys"], new]); ii = np.concatenate([sl["cids"], idn]); o = np.argsort(kk, kind="stable"); sl["keys"], sl["cids"] = kk[o], ii[o]
            if len(sl["keys"]):
                j = np.searchsorted(sl["keys"], key); j[j == len(sl["keys"])] = 0; out[:, g] = np.where(sl["keys"][j] == key, sl["cids"][j], -1)
        return np.ascontiguousarray(np.hstack([a, out]))
    def r_cells(s, k):
        if k < s.KB: return super().r_cells(k)
        sl = s.grown[k - s.GOFF]; m = s.N[sl["cids"]] > 0; return sl["cids"][m].astype(np.int64), sl["keys"][m]
    # ---- trust of every kind that holds delays (plain or made), with the trust of the kinds made from it
    def trust_of_kinds(s):
        u = s.Tu.sum(1); mt = np.where(u > 0, (s.T * s.Tu).sum(1) / np.maximum(u, 1), -np.inf); mt[s.KB:s.GOFF] = -np.inf
        for g, sl in enumerate(s.grown):
            if sl is None: mt[s.GOFF + g] = -np.inf
        if s.keep == 2 and s.P:
            for k, v in s.child_trust(): mt[k] = max(mt[k], 0.0) + v if np.isfinite(mt[k]) else v
        return mt
    GC = 0; GR = 0
    def more_candidates(s, top):                                   # PLAN_10.md, addition 1: the made kinds are candidates for the search for classes beside the 16 by trust
        return top + [s.GOFF + g for g, sl in enumerate(s.grown) if sl is not None and s.GOFF + g not in top and len(sl["keys"])] if s.GC else top
    # ---- the finding: first the kinds to make, then the classes (in plain and made kinds alike), then the kinds keyed on two classes
    def find(s):
        t0 = time.process_time(); rec = []
        if s.G:
            buf, y = s.buf_ctx, s.buf_y; mt = s.trust_of_kinds(); top = [int(k) for k in np.argsort(-mt, kind="stable")[:s.TOP] if np.isfinite(mt[k]) and len(s.kinds[k]) >= 1]; cand = {}
            base_loo = {}
            for k in top:
                R = s.kinds[k]
                if R not in base_loo: base_loo[R] = loo(code(buf, R, s.S), y, s.S, s.eJ)
                for d in range(1, s.D + 1):
                    if d in R: continue
                    Rp = tuple(sorted(R + (d,)))
                    if Rp in s.base_set or Rp in cand: continue
                    cand[Rp] = loo(code(buf, Rp, s.S), y, s.S, s.eJ) - base_loo[R]
            best = [Rp for Rp, w in sorted(cand.items(), key=lambda q: -q[1]) if w > 0][:s.G]; old = {sl["ds"]: g for g, sl in enumerate(s.grown) if sl is not None}
            if s.GR:                                                # control (PLAN_10.md, addition 2): as many kinds as worth would make, but which ones is drawn by chance; one made earlier is kept
                n = len(best); have = list(old)[:n]; pool = [Rp for Rp in cand if Rp not in have]; best = have + [pool[i] for i in s.rng.permutation(len(pool))[:n - len(have)]]
            stay = {old[Rp] for Rp in best if Rp in old}; gone = [g for g in range(s.G) if s.grown[g] is not None and g not in stay]; free = [g for g in range(s.G) if g not in stay]
            for g in gone:                                          # a kind that is not kept goes, with the class kinds found in it
                kcol = s.GOFF + g; s.grown[g] = None; s.kinds[kcol] = (); s.T[kcol] = 0.0; s.Tu[kcol] = 0.0
                for p, sl in enumerate(s.slots):
                    if sl is not None and sl["k"] == kcol: s.slots[p] = None; s.T[s.KB + p] = 0.0; s.Tu[s.KB + p] = 0.0
                for j, sl in enumerate(s.slots2):
                    if sl is not None and kcol in (sl["np"][0], sl["nq"][0]): s.slots2[j] = None; s.T[s.KB + s.P + j] = 0.0; s.Tu[s.KB + s.P + j] = 0.0
            for Rp in best:
                if Rp not in old: g = free.pop(0); s.grown[g] = dict(ds=Rp, keys=np.zeros(0, np.int64), cids=np.zeros(0, np.int32), born=int(s.read)); s.kinds[s.GOFF + g] = Rp
            data = [(sl, s.C[sl["cids"]].copy(), s.N[sl["cids"]].copy(), s.U[sl["cids"]].copy()) for sl in s.grown if sl is not None]; pos = s.nbase0      # the cells of the made kinds are moved below the class cells, which are made anew
            s.ensure(sum(len(d[2]) for d in data) + 16)
            for sl, C_, N_, U_ in data:
                n = len(N_); s.C[pos:pos + n] = C_; s.N[pos:pos + n] = N_; s.U[pos:pos + n] = U_; sl["cids"] = np.arange(pos, pos + n, dtype=np.int32); pos += n
            s.nbase = pos
            rec = [dict(delays=list(sl["ds"]), worth=round(cand.get(sl["ds"], 0.0), 1), born_at=sl["born"], cells=int((s.N[sl["cids"]] > 0).sum())) for sl in s.grown if sl is not None]
            s.finds_g.append(dict(bytes_read=int(s.read), candidates=len(cand), of_positive_worth=sum(w > 0 for w in cand.values()), made=rec, cpu=round(time.process_time() - t0, 1))); s.tfind += time.process_time() - t0
        super().find()
    def sizes(s):
        z = super().sizes(); z["made_kinds"] = sum(sl is not None for sl in s.grown); z["made_cells"] = int(sum((s.N[sl["cids"]] > 0).sum() for sl in s.grown if sl is not None)); return z
    def trust_summary(s):
        z = super().trust_summary()
        if s.P:
            a = np.abs(s.T * (s.Tu > 0)); t = max(a.sum(), 1e-12); z["share_of_trust_in_class_kinds"] = round(float(a[s.KB:s.KB + s.P].sum() / t), 4); z["share_of_trust_in_two_class_kinds"] = round(float(a[s.KB + s.P:s.GOFF].sum() / t), 4)
            z["share_of_trust_in_made_kinds"] = round(float(a[s.GOFF:].sum() / t), 4); z["made_kinds"] = [dict(delays=list(sl["ds"]), largest_trust=round(float(s.T[s.GOFF + g].max()), 3)) for g, sl in enumerate(s.grown) if sl is not None]
            z["largest_trust_two_class_kinds"] = [round(float(x), 3) for x in s.T[s.KB + s.P:s.GOFF].max(1)]
        return z

def check():
    """The cells of a made kind against a count of the stream by hand, from the byte at which the kind was made; and its addresses."""
    import itertools
    merge.Sys = Sys10; D = 9; seed = 1; rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); pairs = [(int(q) // 100, int(q) % 100) for q in o]; shown = pairs[2000:]; rng.permutation(10000)
    TAB = np.array([merge.text(a, b) for a, b in shown], np.int64); kinds = [c for m in range(5) for c in itertools.combinations(range(1, D + 1), m)]
    s = Sys10(14, D, kinds, 0.1, 2.0, 0.02, True, P=32, eJ=0.02, mode="classes", seed=seed, keep=2); X = TAB[rng.integers(8000, size=6000)].ravel(); x = np.concatenate([np.full(D, merge.END, np.int64), X])
    s.learn(merge.ctx_of(x, D, len(x), D), x[D:]); ok = True; n = sum(sl is not None for sl in s.grown); bad = tot = 0
    for g, sl in enumerate(s.grown):
        if sl is None: continue
        ref = {}
        for t in range(D + sl["born"], len(x)):                    # byte number b of the stream is x[D + b]
            key = sum(int(x[t - d]) * 14 ** i for i, d in enumerate(sl["ds"])); ref.setdefault(key, {}); ref[key][int(x[t])] = ref[key].get(int(x[t]), 0) + 1
        have = {int(k): {int(v): int(c) for v, c in enumerate(s.C[i]) if c > 0} for k, i in zip(sl["keys"], sl["cids"]) if s.N[i] > 0}; tot += len(ref); bad += have != ref
    good = n > 0 and bad == 0; ok &= good; print(f"cells of the kinds made while reading, 6,000 lines of two-digit sums: {n} kinds, {tot} cells counted again by hand from the byte each kind was made at; kinds that differ: {bad}", "PASSED" if good else "FAILED")
    ctx = merge.ctx_of(x, len(x) - 500, len(x), D); idd = s.ids(ctx); bad = 0
    for g, sl in enumerate(s.grown):
        if sl is None: continue
        for r in range(len(ctx)):
            key = sum(int(ctx[r, d - 1]) * 14 ** i for i, d in enumerate(sl["ds"])); pos = np.flatnonzero(sl["keys"] == key); bad += int(idd[r, s.GOFF + g]) != (int(sl["cids"][pos[0]]) if len(pos) else -1)
    ok &= bad == 0; print(f"addresses of those cells on 500 contexts: {bad} differ from the address worked out by hand", "PASSED" if bad == 0 else "FAILED")
    made_in = sum(sl is not None and sl["k"] >= s.GOFF for sl in s.slots); print(f"(of the {sum(sl is not None for sl in s.slots)} kept pairs, {made_in} are inside a made kind; made kinds: {[sl['ds'] for sl in s.grown if sl is not None]})")
    return ok

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    merge.Sys = Sys10; kw = dict(a.split("=") for a in sys.argv[2:]); kw.setdefault("keep", "2"); Sys10.G = int(kw.get("G", 16)); Sys10.GC = int(kw.get("gc", 0)); Sys10.GR = int(kw.get("gchance", 0)); merge6.Sys6.P2 = int(kw.get("P2", 32)); merge6.Sys6.W = int(kw.get("W", 50000))
    out, s = dict(sums=merge.run_sums, dev=merge.run_dev, life=merge.run_life)[sys.argv[1]](kw); out["finds_two"] = s.finds2; out["finds_made"] = s.finds_g; tag = kw["tag"]
    if sys.argv[1] == "sums":
        nd = int(kw.get("nd", 2)); seed = int(kw.get("seed", 1)); n10 = 10 ** nd; o = np.random.default_rng([seed, 99]).permutation(n10 * n10); held = [(int(q) // n10, int(q) % n10) for q in o[:500]]; out["two_class_kinds_alone"], out["two_class_cells_alone"] = merge6.two_alone(s, held, nd)
    json.dump(out, open(os.path.join(HERE, "res", tag + ".json"), "w")); z = out.get("sizes") or out
    if sys.argv[1] == "sums": print(tag, "| last3", round(out["last3"], 1), "last", out["last"], "place", [round(v, 1) for v in out["place_last"]], "|", " ".join(f"{r['lines'] // 1000}k:{r['add']:.1f}" for r in out["rows"]), "| two-class kinds alone", out["two_class_kinds_alone"], "| cells", z["cells"], "| made kinds", z["made_kinds"], "cells", z["made_cells"], "| trust in made, class, two-class", out["trust"].get("share_of_trust_in_made_kinds"), out["trust"].get("share_of_trust_in_class_kinds"), out["trust"].get("share_of_trust_in_two_class_kinds"), "| cpu", out["cpu"], "finding", out["cpu_finding"], flush=True)
    else: print(tag, "|", {k: round(v["nats"], 4) for k, v in out["scores"].items()} if "scores" in out else round(out["nats"], 4), "| cells", z["cells"], "| made kinds", z["made_kinds"], "cells", z["made_cells"], "| trust in made", out["trust"].get("share_of_trust_in_made_kinds"), "| cpu", out["cpu"], "finding", out["cpu_finding"], flush=True)
