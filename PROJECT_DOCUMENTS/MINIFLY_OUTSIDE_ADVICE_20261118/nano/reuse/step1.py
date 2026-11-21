#!/usr/bin/env python3
"""PLAN_1.md: in a system that has read two-digit sums with classes (merge.py, frozen setting), which kinds of cell say the same?
A kind is a table from the values of its key places (in an order) to its cells' most-counted follower; two tables are compared on their common values.
    python3 step1.py check
    python3 step1.py [seed=1] [lines=40000]"""
import itertools, json, os, sys
import numpy as np
import merge
HERE = os.path.dirname(os.path.abspath(__file__)); S = 14; MINC = 2

def compare(a, oa, b, ob, perm):
    """a, b: arrays of most-counted followers, shape (S,)*m; oa, ob: which cells enter. A's place i is matched with B's place perm[i]."""
    bp = b.transpose(perm); c = oa & ob.transpose(perm); n = int(c.sum())
    if n == 0: return 0, 0.0, 0.0
    x, y = a[c], bp[c]; fa = np.bincount(x, minlength=S) / n; fb = np.bincount(y, minlength=S) / n
    return n, float((x == y).mean()), float(fa @ fb)

def compare_py(a, oa, b, ob, perm):
    m = a.ndim; n = same = 0; ca = {}; cb = {}
    for key in itertools.product(range(S), repeat=m):
        kb = [0] * m
        for i in range(m): kb[perm[i]] = key[i]
        kb = tuple(kb)
        if oa[key] and ob[kb]:
            n += 1; same += int(a[key] == b[kb]); ca[a[key]] = ca.get(a[key], 0) + 1; cb[b[kb]] = cb.get(b[kb], 0) + 1
    return n, (same / n if n else 0.0), (sum(ca[v] * cb.get(v, 0) for v in ca) / n / n if n else 0.0)

def check():
    rng = np.random.default_rng(4); ok = True
    for m in (2, 3):
        for _ in range(4):
            a = rng.integers(0, 5, (S,) * m); b = rng.integers(0, 5, (S,) * m); oa = rng.random((S,) * m) < 0.6; ob = rng.random((S,) * m) < 0.6; perm = tuple(rng.permutation(m))
            r1 = compare(a, oa, b, ob, perm); r2 = compare_py(a, oa, b, ob, perm); good = r1[0] == r2[0] and abs(r1[1] - r2[1]) < 1e-12 and abs(r1[2] - r2[2]) < 1e-12; ok &= good
            print(f"comparison of two tables of {m} places under the matching {perm}: numpy {r1[0]} common, {r1[1]:.4f}, chance {r1[2]:.4f}; plain Python {r2[0]}, {r2[1]:.4f}, {r2[2]:.4f}", "PASSED" if good else "FAILED")
    return ok

def tables(s):
    """Every plain kind with 1 to 4 delays, and every slice of a class kind, as (name, places, arg, counts)."""
    out = []
    for k, ds in enumerate(s.kinds):
        m = len(ds)
        if m == 0: continue
        C = s.C[s.off[k]:s.off[k] + S ** m]; N = s.N[s.off[k]:s.off[k] + S ** m]
        out.append(dict(kind="plain", k=k, places=ds, arg=C.argmax(1).reshape((S,) * m, order="F"), n=N.reshape((S,) * m, order="F")))
    for p, sl in enumerate(s.slots):
        if sl is None: continue
        m = len(sl["Z"]); cl = sl["keys"] // sl["ZS"]; z = sl["keys"] % sl["ZS"]
        for c in np.unique(cl):
            arg = np.zeros(S ** m, np.int64); n = np.zeros(S ** m, np.float32); q = cl == c; arg[z[q]] = s.C[sl["cids"][q]].argmax(1); n[z[q]] = s.N[sl["cids"][q]]
            members = [int(v) for v in np.flatnonzero(sl["clsmap"] == c)]
            out.append(dict(kind="slice", slot=p, R=s.kinds[sl["k"]], X=sl["X"], places=sl["Z"], cls=int(c), members=members, arg=arg.reshape((S,) * m, order="F"), n=n.reshape((S,) * m, order="F"), trust=float(s.T[s.KB + p].max())))
    return out

def name(t):
    if t["kind"] == "plain": return "plain" + str(tuple(t["places"]))
    return f"class kind X={tuple(t['X'])} Z={tuple(t['places'])} at class {t['cls']} ({len(t['members'])} members)"

def main(kw):
    kw = dict(kw); kw.setdefault("seed", "1"); kw.setdefault("lines", "40000"); out, s = merge.run_sums(kw); tabs = tables(s); mt = s.trust_of_kinds()
    top = [int(k) for k in np.argsort(-mt, kind="stable")[:16] if len(s.kinds[k]) >= 2]; plain = [t for t in tabs if t["kind"] == "plain"]; slices = [t for t in tabs if t["kind"] == "slice"]
    print("score of the run (never-shown sums, last three readings):", round(out["last3"], 1), "| the 16 kinds of largest trust with two delays or more:", [s.kinds[k] for k in top])
    recs = []
    def one(A, B, digits):
        best = None
        for perm in itertools.permutations(range(len(A["places"]))):
            if A is B and perm == tuple(range(len(perm))): continue
            oa = A["n"] >= MINC; ob = B["n"] >= MINC
            if digits: oa = oa & (A["arg"] < 10); ob = ob & (B["arg"] < 10)
            n, ag, ch = compare(A["arg"], oa, B["arg"], ob, perm)
            if n >= 50 and (best is None or ag - ch > best["agree"] - best["chance"]): best = dict(a=name(A), b=name(B), same=A is B, matching=[int(B["places"][j]) for j in perm], a_places=[int(d) for d in A["places"]], common=n, agree=round(ag, 4), chance=round(ch, 4), digits=digits)
        return best
    for digits in (False, True):
        for A in [t for t in plain if t["k"] in top]:
            for B in plain:
                if len(B["places"]) == len(A["places"]):
                    r = one(A, B, digits)
                    if r: r["type"] = "plain-plain"; recs.append(r)
        for A in slices:
            for B in plain + slices:
                if len(B["places"]) == len(A["places"]) and (B["kind"] == "plain" or (B["slot"], B["cls"]) > (A["slot"], A["cls"])):
                    r = one(A, B, digits)
                    if r: r["type"] = "slice-" + B["kind"]; recs.append(r)
    json.dump(dict(args=kw, score=out["last3"], top=[list(s.kinds[k]) for k in top], records=recs, kept=[dict(R=r["R"], X=r["X"], Z=r["Z"], classes=r["classes"]) for r in out["finds"][-1]["kept"]]), open(os.path.join(HERE, "res", "step1_seed%s.json" % kw["seed"]), "w"))
    for digits in (False, True):
        print("\n==== cells whose most-counted follower is a digit ====" if digits else "\n==== all cells with at least two counts ====")
        for typ, self_ in (("plain-plain", True), ("plain-plain", False), ("slice-plain", False), ("slice-slice", False)):
            rr = [r for r in recs if r["digits"] == digits and r["type"] == typ and r["same"] == self_]; ex = np.array([r["agree"] - r["chance"] for r in rr])
            if not len(rr): continue
            print(f"\n{typ}{' (a kind against itself with places exchanged)' if self_ else ''}: {len(rr)} pairs with at least 50 common values at their best matching; excess over chance above 0.2: {int((ex > 0.2).sum())} ({100 * (ex > 0.2).mean():.1f}%); agreement at least 0.9 with chance under 0.5: {sum(r['agree'] >= 0.9 and r['chance'] < 0.5 for r in rr)}; median excess {np.median(ex):.3f}")
            for r in sorted(rr, key=lambda r: -(r["agree"] - r["chance"]))[:12]:
                print(f"   {r['a']} places {r['a_places']} ~ {r['b']} places {r['matching']}: {r['common']} common, agree {r['agree']:.3f}, chance {r['chance']:.3f}")
    # the cases named in PLAN_1.md
    print("\n==== the named cases ====")
    P47 = next(t for t in plain if tuple(t["places"]) == (4, 7))
    for t in slices:
        if tuple(t["X"]) == (3, 6) and tuple(t["places"]) == (4, 7):
            dd = [(q % 14, q // 14) for q in t["members"] if q % 14 < 10 and q // 14 < 10]; car = sum(a + b >= 10 for a, b in dd)
            for perm in ((0, 1),):
                n, ag, ch = compare(t["arg"], (t["n"] >= MINC) & (t["arg"] < 10), P47["arg"], (P47["n"] >= MINC) & (P47["arg"] < 10), perm)
                print(f"S1/S2: {name(t)} [{car} of its {len(dd)} pairs of digits carry] against plain (4, 7): {n} common, agree {ag:.3f}, chance {ch:.3f}")
    for ds, perm in (((3, 4, 6, 7), (2, 3, 0, 1)), ((4, 7), (1, 0)), ((3, 6), (1, 0))):
        t = next(t for t in plain if tuple(t["places"]) == ds); o = (t["n"] >= MINC) & (t["arg"] < 10); n, ag, ch = compare(t["arg"], o, t["arg"], o, perm)
        print(f"S4: plain {ds} against itself at places {[ds[j] for j in perm]} (digit followers): {n} common, agree {ag:.3f}, chance {ch:.3f}; trust rank {int(np.argsort(-mt).tolist().index(s.kinds.index(ds))) + 1}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check": sys.exit(0 if check() else 1)
    main(dict(a.split("=") for a in sys.argv[1:]))
