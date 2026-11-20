#!/usr/bin/env python3
"""PLAN_6.md: cells keyed on two classes. As merge.py's frozen setting, with one addition: the class of an X-cell is used as a line.
For two kept pairs p and q whose X's share no delay, a kind is keyed on (the class of p's X-cell, the class of q's X-cell, the delays of p's Z that are not in q's X).
Which of these kinds are made is decided on the system's own last W bytes: a candidate's worth is how well its cells foretell those bytes, each byte left out of its own cell,
less the better of the two class kinds it is made from, measured on the same bytes. The P2 of largest positive worth are made; their cells start empty and their trusts at nothing.
    python3 merge6.py check
    python3 merge6.py sums|dev|life  key=value ...      (the arguments of merge.py; P2=32 W=50000)"""
import json, os, sys, time
import numpy as np
import merge
from merge import code
HERE = os.path.dirname(os.path.abspath(__file__)); NC = 4226                     # more than any number of classes of an X of two delays

def loo(key, y, S, e):
    """Cells keyed on `key` counting the bytes y: the sum over the bytes of the log of what the byte's cell says of it with that byte left out."""
    uk, inv = np.unique(key, return_inverse=True); n = np.bincount(inv); kv, c = np.unique(inv.astype(np.int64) * S + y, return_counts=True); nn = n[kv // S]
    return float((c * (np.log(c - 1 + e) - np.log(nn - 1 + S * e))).sum())

def loo_py(key, y, S, e):
    import math; cells = {}
    for k, v in zip(key, y): d = cells.setdefault(int(k), {}); d[int(v)] = d.get(int(v), 0) + 1
    return sum(c * (math.log(c - 1 + e) - math.log(sum(d.values()) - 1 + S * e)) for d in cells.values() for c in d.values())

class Sys6(merge.Sys):
    P2 = 32; W = 50000; TOP = 16
    def __init__(s, *a, **k):
        super().__init__(*a, **k)
        if not s.P: s.P2 = 0
        s.K1 = s.KB + s.P + s.P2; assert s.K1 <= 400
        s.T = np.vstack([s.T, np.zeros((s.P2, merge.NL))]); s.Tu = np.vstack([s.Tu, np.zeros((s.P2, merge.NL))]); s.slots2 = [None] * s.P2
        s.buf_ctx = np.zeros((0, s.D), np.int64); s.buf_y = np.zeros(0, np.int64); s.finds2 = []
    def _run(s, ctx, y, learn, want_p):
        r = super()._run(ctx, y, learn, want_p)
        if learn and s.P2: s.buf_ctx = np.concatenate([s.buf_ctx, ctx])[-s.W:]; s.buf_y = np.concatenate([s.buf_y, np.asarray(y, np.int64)])[-s.W:]
        return r
    def key2(s, sl, ctx):
        cp = sl["mp"][code(ctx, sl["Xp"], s.S)]; cq = sl["mq"][code(ctx, sl["Xq"], s.S)]; ok = (cp >= 0) & (cq >= 0)
        return ok, (cp[ok].astype(np.int64) * NC + cq[ok]) * sl["ZS"] + code(ctx[ok], sl["Zr"], s.S)
    def ids(s, ctx, create=False):
        a = super().ids(ctx, create)
        if not s.P2: return a
        out = np.full((len(ctx), s.P2), -1, np.int32)
        for j, sl in enumerate(s.slots2):
            if sl is None: continue
            ok, key = s.key2(sl, ctx)
            if not ok.any(): continue
            if create:
                new = np.setdiff1d(key, sl["keys"])
                if len(new):
                    s.ensure(len(new)); idn = np.arange(s.top, s.top + len(new), dtype=np.int32); s.top += len(new)
                    kk = np.concatenate([sl["keys"], new]); ii = np.concatenate([sl["cids"], idn]); o = np.argsort(kk, kind="stable"); sl["keys"], sl["cids"] = kk[o], ii[o]
            if len(sl["keys"]):
                jj = np.searchsorted(sl["keys"], key); jj[jj == len(sl["keys"])] = 0; out[ok, j] = np.where(sl["keys"][jj] == key, sl["cids"][jj], -1)
        return np.ascontiguousarray(np.hstack([a, out]))
    def find(s):
        name = lambda sl: (sl["k"], sl["X"]); old = {(sl["np"], sl["nq"], sl["Zr"]): j for j, sl in enumerate(s.slots2) if sl is not None}; T0, Tu0 = s.T.copy(), s.Tu.copy()
        super().find()
        if not s.P2: return
        t0 = time.process_time(); s.T[s.KB + s.P:] = 0.0; s.Tu[s.KB + s.P:] = 0.0; s.slots2 = [None] * s.P2; buf, y = s.buf_ctx, s.buf_y; first = [(p, sl) for p, sl in enumerate(s.slots) if sl is not None]
        cls = {p: sl["clsmap"][code(buf, sl["X"], s.S)] for p, sl in first}; u = s.Tu[s.KB:s.KB + s.P].sum(1); mt = np.where(u > 0, (s.T[s.KB:s.KB + s.P] * s.Tu[s.KB:s.KB + s.P]).sum(1) / np.maximum(u, 1), -np.inf)
        top = [int(p) for p in np.argsort(-mt, kind="stable")[:s.TOP] if s.slots[p] is not None]; cand = {}; tried = 0
        for p in top:
            sp = s.slots[p]
            for q, sq in first:
                if q == p or set(sq["X"]) & set(sp["X"]): continue
                Zr = tuple(d for d in sp["Z"] if d not in sq["X"]); sig = (frozenset((name(sp), name(sq))), Zr) if not Zr else (name(sp), name(sq), Zr)
                if sig in cand: continue
                ok = (cls[p] >= 0) & (cls[q] >= 0)
                if ok.sum() < 2: continue
                b = buf[ok]; yy = y[ok]; tried += 1
                kpq = (cls[p][ok].astype(np.int64) * NC + cls[q][ok]) * (s.S ** len(Zr)) + code(b, Zr, s.S)
                kp = cls[p][ok].astype(np.int64) * sp["ZS"] + code(b, sp["Z"], s.S); kq = cls[q][ok].astype(np.int64) * sq["ZS"] + code(b, sq["Z"], s.S)
                w = loo(kpq, yy, s.S, s.eJ) - max(loo(kp, yy, s.S, s.eJ), loo(kq, yy, s.S, s.eJ)); cand[sig] = (w, p, q, Zr, int(ok.sum()))
        best = sorted((v for v in cand.values() if v[0] > 0), key=lambda v: -v[0])[:s.P2]; rec = []
        for j, (w, p, q, Zr, nok) in enumerate(best):
            sp, sq = s.slots[p], s.slots[q]; sl = dict(np=name(sp), nq=name(sq), Xp=sp["X"], Xq=sq["X"], mp=sp["clsmap"], mq=sq["clsmap"], Zr=Zr, ZS=s.S ** len(Zr), keys=np.zeros(0, np.int64), cids=np.zeros(0, np.int32)); s.slots2[j] = sl
            if (sl["np"], sl["nq"], Zr) in old: jo = old[(sl["np"], sl["nq"], Zr)]; s.T[s.KB + s.P + j] = T0[s.KB + s.P + jo]; s.Tu[s.KB + s.P + j] = Tu0[s.KB + s.P + jo]
            rec.append(dict(p=dict(R=list(s.kinds[sp["k"]]), X=list(sp["X"])), q=dict(R=list(s.kinds[sq["k"]]), X=list(sq["X"])), rest=list(Zr), worth=round(w, 1), bytes=nok))
        dt = time.process_time() - t0; s.tfind += dt; s.finds2.append(dict(bytes_read=int(s.read), buffer=int(len(y)), candidates=tried, of_positive_worth=sum(v[0] > 0 for v in cand.values()), kept=rec, cpu=round(dt, 1)))
    def child_trust(s):
        out = super().child_trust(); a = s.KB + s.P; u = s.Tu[a:].sum(1); m = np.where(u > 0, (s.T[a:] * s.Tu[a:]).sum(1) / np.maximum(u, 1), 0.0)
        for j, sl in enumerate(s.slots2):                           # a kind keyed on two classes is a child of both kinds its classes were found in
            if sl is not None: out += [(sl["np"][0], max(0.0, float(m[j]))), (sl["nq"][0], max(0.0, float(m[j])))]
        return out
    def sizes(s):
        z = super().sizes(); z["two_class_kinds"] = sum(sl is not None for sl in s.slots2); z["two_class_cells"] = int(sum((s.N[sl["cids"]] > 0).sum() for sl in s.slots2 if sl is not None)); return z
    def trust_summary(s):
        if not s.P: return {}
        u = s.Tu.sum(1); m = np.where(u > 0, (s.T * s.Tu).sum(1) / np.maximum(u, 1), 0.0); a = np.abs(s.T * (s.Tu > 0)); z = max(a.sum(), 1e-12)
        return dict(share_of_trust_in_class_kinds=round(float(a[s.KB:s.KB + s.P].sum() / z), 4), share_of_trust_in_two_class_kinds=round(float(a[s.KB + s.P:].sum() / z), 4),
                    largest_trust_two_class_kinds=[round(float(x), 3) for x in s.T[s.KB + s.P:].max(1)], mean_trust_two_class_kinds=[round(float(x), 4) for x in m[s.KB + s.P:]])

def two_alone(s, held, nd):
    """The never-shown score with only the kinds keyed on two classes speaking; with only those and the class kinds."""
    keep = s.N.copy(); cid = np.concatenate([sl["cids"] for sl in s.slots2 if sl is not None] + [np.zeros(0, np.int32)]); m = np.zeros(len(s.N), bool); m[cid] = True
    s.N[~m] = 0; a = merge.sums_score(s, held, 9, nd); s.N[:] = keep; return a, int((keep[cid] > 0).sum())

def check():
    rng = np.random.default_rng(4); ok = True; key = rng.integers(0, 40, 3000).astype(np.int64) * 977; y = (key // 977 % 5 + (rng.random(3000) < 0.2) * rng.integers(0, 14, 3000)) % 14
    a = loo(key, y, 14, 0.02); b = loo_py(key, y, 14, 0.02); good = abs(a - b) < 1e-8 * abs(b); ok &= good; print(f"worth of a kind on 3,000 bytes: numpy {a:.6f} against plain Python {b:.6f}", "PASSED" if good else "FAILED")
    merge.Sys = Sys6; out, s = merge.run_sums(dict(seed="1", lines="6000", P="32")); D = 9; n2 = sum(sl is not None for sl in s.slots2)
    rs = np.random.default_rng([1, 99]); o = rs.permutation(10000); held = [(int(q) // 100, int(q) % 100) for q in o][:300]; Ls = np.array([merge.text(a, b) for a, b in held]); bad = tot = 0
    for t in (6, 7, 8):
        ctx = np.full((len(Ls), D), merge.END, np.int64)
        for d in range(1, D + 1):
            if t - d >= 0: ctx[:, d - 1] = Ls[:, t - d]
        idd = s.ids(ctx)
        for j, sl in enumerate(s.slots2):
            if sl is None: continue
            for r in range(len(ctx)):                                         # the address by hand, one context at a time
                cp = sl["mp"][sum(int(ctx[r, d - 1]) * 14 ** i for i, d in enumerate(sl["Xp"]))]; cq = sl["mq"][sum(int(ctx[r, d - 1]) * 14 ** i for i, d in enumerate(sl["Xq"]))]; want = -1
                if cp >= 0 and cq >= 0:
                    k = (int(cp) * NC + int(cq)) * sl["ZS"] + sum(int(ctx[r, d - 1]) * 14 ** i for i, d in enumerate(sl["Zr"])); pos = np.flatnonzero(sl["keys"] == k); want = int(sl["cids"][pos[0]]) if len(pos) else -1
                tot += 1; bad += int(idd[r, s.KB + s.P + j]) != want
    good = bad == 0 and n2 > 0 and tot > 0; ok &= good; print(f"addresses of the cells keyed on two classes, {n2} kinds, {tot} contexts: {bad} differ from the address worked out by hand", "PASSED" if good else "FAILED")
    return ok

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    merge.Sys = Sys6; kw = dict(a.split("=") for a in sys.argv[2:]); Sys6.P2 = int(kw.get("P2", 32)); Sys6.W = int(kw.get("W", 50000)); out, s = dict(sums=merge.run_sums, dev=merge.run_dev, life=merge.run_life)[sys.argv[1]](kw)
    out["finds_two"] = s.finds2; tag = kw["tag"]
    if sys.argv[1] == "sums":
        nd = int(kw.get("nd", 2)); seed = int(kw.get("seed", 1)); n10 = 10 ** nd; o = np.random.default_rng([seed, 99]).permutation(n10 * n10); held = [(int(q) // n10, int(q) % n10) for q in o[:500]]; out["two_class_kinds_alone"], out["two_class_cells_alone"] = two_alone(s, held, nd)
    json.dump(out, open(os.path.join(HERE, "res", tag + ".json"), "w")); f2 = s.finds2[-1] if s.finds2 else {}
    if sys.argv[1] == "sums": print(tag, "| last3", round(out["last3"], 1), "last", out["last"], "place", [round(v, 1) for v in out["place_last"]], "|", " ".join(f"{r['lines'] // 1000}k:{r['add']:.1f}" for r in out["rows"]), "| class kinds alone", out.get("class_kinds_alone"), "| two-class kinds alone", out["two_class_kinds_alone"], "cells", out["two_class_cells_alone"], "| cells", out["sizes"]["cells"], "| two-class kinds", out["sizes"].get("two_class_kinds"), "| trust", out["trust"].get("share_of_trust_in_two_class_kinds"), "| cpu", out["cpu"], "finding", out["cpu_finding"], flush=True)
    else: print(tag, "|", {k: round(v["nats"], 4) for k, v in out["scores"].items()} if "scores" in out else round(out["nats"], 4), "| cells", (out.get("sizes") or out)["cells"], "| two-class kinds", (out.get("sizes") or out).get("two_class_kinds"), "cells", (out.get("sizes") or out).get("two_class_cells"), "| trust", out["trust"].get("share_of_trust_in_two_class_kinds"), "| cpu", out["cpu"], "finding", out["cpu_finding"], flush=True)
