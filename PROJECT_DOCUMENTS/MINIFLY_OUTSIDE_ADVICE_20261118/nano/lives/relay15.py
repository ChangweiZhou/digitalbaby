#!/usr/bin/env python3
"""nano/lives, PLAN_15.md: the life of PLAN_13.md (three returns) with rehearsal from a small store of episodes.
    python3 relay15.py check
    python3 relay15.py life seed=1 form=R|Rgtqr rho=0.01 [store=1000] [stretch=30] [theta=4] [lines=400000] tag=..
The store: as the stream is read it is cut into consecutive episodes of ten bytes plus `stretch` bytes; the system keeps `store` of them, so chosen
that every episode read so far has the same chance of being among them (when the store is full, the n-th episode replaces a kept one, drawn
evenly, with chance store / n). No kind of text is named and no line is aligned. rho: bytes rehearsed for each byte read (0: none, and the run is
relay13.py's). After every stretch / rho bytes of the stream one kept episode, drawn evenly, is read again with learning on: its first ten bytes
fill the delay line without learning, the rest are learned; afterwards the delay line of the stream is put back.
The learner is relay12.py's, unchanged. The marks are relay13.py's."""
import copy, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "loop"))
import aba as AB, loop as LP, relay12 as R12

class Reader:
    """Reads a stream into a network, with rehearsal of the stream's own past."""
    def __init__(s, net, xm, first, rho, stretch, seed, store=1000):
        s.net, s.xm, s.first, s.rho, s.L = net, xm, first, rho, stretch; s.K = int(round(stretch / rho)) if rho else 0; s.since = 0; s.rng = np.random.default_rng([seed, 2026]); s.rehearsed = 0; s.right_rehearsing = 0; s.starts = []
        s.M = store; s.kept = []; s.offered = 0; s.E = net.D + stretch; s.next = first             # the store: the places where the kept episodes begin (their bytes are the system's to keep: store x (10 + stretch) bytes)
    def offer(s, now):
        """Every episode that has been read whole by now is offered to the store once."""
        while s.next + s.E <= now:
            s.offered += 1
            if len(s.kept) < s.M: s.kept.append(s.next)
            elif s.rng.random() < s.M / s.offered: s.kept[int(s.rng.integers(s.M))] = s.next
            s.next += s.E
    def rehearse(s, now):
        """One kept episode, drawn evenly from the store."""
        D = s.net.D
        if not s.kept: return
        r = s.kept[int(s.rng.integers(len(s.kept)))] + D; keep = s.net.hist.copy(); r0 = s.net.right
        xs = s.xm[r - D:r + s.L - 1]; ys = s.xm[r - D + 1:r + s.L].copy(); ys[:D - 1] = -1                 # the first D - 1 steps only fill the delay line; then L bytes are learned
        s.net.read(xs, ys); s.net.hist[:] = keep; s.rehearsed += s.L; s.right_rehearsing += s.net.right - r0; s.starts.append(r)
    def read(s, lo, hi):
        if not s.K: s.net.read(s.xm[lo - 1:hi - 1], s.xm[lo:hi]); return
        p = lo
        while p < hi:
            q = min(hi, p + s.K - s.since); s.net.read(s.xm[p - 1:q - 1], s.xm[p:q]); s.since += q - p; p = q; s.offer(p)
            if s.since >= s.K: s.rehearse(p); s.since = 0

def run(kw):
    seed = int(kw.get("seed", 1)); form = kw["form"]; lines = int(kw.get("lines", 400000)); theta = float(kw.get("theta", 4.0)); rho = float(kw.get("rho", 0.0)); stretch = int(kw.get("stretch", 30)); tag = kw["tag"]; D = 10; t0 = time.time()
    x, sp = AB.build(seed, lines); eng = AB.LV.raw("shakespeare")[0]; held_eng = eng[400000:420000]; vals = np.unique(x); idx = np.zeros(256, np.int64); idx[vals] = np.arange(len(vals)); S = len(vals)
    rg = np.random.default_rng([seed, 99]); o = rg.permutation(10000); pairs = [(int(q) // 100, int(q) % 100) for q in o]; TAB = np.array([AB.text(a, b) for a, b in pairs[2000:]], np.int64)
    B2 = eng[420000:760000]; left_out = int((~np.isin(B2, vals)).sum()); B2 = B2[np.isin(B2, vals)]; B3 = eng[760000:1100000]; assert np.isin(B3, vals).all() and np.isin(held_eng, vals).all()
    more = [B2, TAB[np.random.default_rng([seed, 56]).integers(8000, size=5000)].ravel(), B3, TAB[np.random.default_rng([seed, 57]).integers(8000, size=5000)].ravel()]
    head = x[:sp["A2"][1]]; xm = idx[np.concatenate([head] + more)].astype(np.int32); cuts = np.cumsum([len(head)] + [len(m) for m in more])            # the stream of relay13.py
    spans = {"sums": sp["A"], "English 1": sp["B"], "return 1": sp["A2"], "English 2": (int(cuts[0]), int(cuts[1])), "return 2": (int(cuts[1]), int(cuts[2])), "English 3": (int(cuts[2]), int(cuts[3])), "return 3": (int(cuts[3]), int(cuts[4]))}
    s = R12.Sys(seed, 256, D, S, int(idx[10]), eta1=1.0, eta2=1 / 16, lam=1e-5, rscale=1.0, soft=0, theta=theta, **R12.switches(form)); rd = Reader(s, xm, sp["A"][0], rho, stretch, seed, int(kw.get("store", 1000)))
    ev = idx[x[sp["ev"]:]].reshape(5, 500, 20); rows = []; digits = np.array([int(idx[v]) for v in range(48, 58)]); he = idx[held_eng].astype(np.int64)
    HS = []; SH = []; Lsh = idx[TAB[:500]]
    for d in range(5):
        prev = ev[d, :, :10]; Ls = ev[d, :, 10:]
        for t in (5, 6, 7):
            for L, out in ((Ls, HS), (Lsh, SH)): hist = np.zeros((500, D), np.int64); hist[:, :t + 1] = L[:, t::-1]; hist[:, t + 1:] = prev[:, ::-1][:, :D - t - 1]; out.append((hist, L[:, t + 1]))
    HE = np.stack([he[D - 1 - j: len(he) - 1 - j] for j in range(D)], 1); YE = he[D:]; HSall = np.concatenate([h for h, _ in HS]); YS = np.concatenate([y for _, y in HS]); others = np.setdiff1d(np.arange(S), digits)
    def whole(sy, among=None, states=None):
        ok = np.ones((5, 500), bool)
        for i, (hist, y) in enumerate(states or HS):
            a = sy.branches(hist)
            if among is not None: b = np.full_like(a, -np.inf); b[:, among] = a[:, among]; a = b
            ok[i // 3] &= a.argmax(1) == y
        return float(100 * ok.mean())
    def english(sy): return float(100 * (sy.branches(HE).argmax(1) == YE).mean())
    def mark(label, sy=None): sy = sy or s; r = dict(at=label, whole=whole(sy), among_digits=whole(sy, digits), english=english(sy), shown_whole=whole(sy, None, SH)); rows.append(r); print(tag, label, {k: round(v, 1) for k, v in r.items() if k != "at"}, flush=True)
    def voices(sy):
        a = sy.branches(HSall); right = a[np.arange(len(YS)), YS]; o_ = a[:, others]; top = o_.argmax(1); loud = o_[np.arange(len(YS)), top]
        return dict(right_digit_output=float(right.mean()), loudest_other_branch_output=float(loud.mean()), share_of_states_where_another_branch_is_loudest=float((loud > a[:, digits].max(1)).mean()))
    def where(a, b):
        """Of the stretches rehearsed while the stream stood between a and b: how many began in the sums (first phase or a return) and how many in a piece of English."""
        st = np.array(rd.starts[a:b], np.int64); eng_ = np.zeros(len(st), bool)
        for k in (1, 2, 3): lo, hi = spans[f"English {k}"]; eng_ |= (st >= lo) & (st < hi)
        return dict(stretches=int(len(st)), began_in_sums=int((~eng_).sum()), began_in_english=int(eng_.sum()))
    def where_kept():
        st = np.array(rd.kept, np.int64); eng_ = np.zeros(len(st), bool)
        for k in (1, 2, 3): lo, hi = spans[f"English {k}"]; eng_ |= (st >= lo) & (st < hi)
        return dict(kept=int(len(st)), of_sums=int((~eng_).sum()), of_english=int(eng_.sum()))
    rd.read(*spans["sums"]); mark("end of the sums"); notes = dict(end_of_the_sums=dict(voices=voices(s))); right = {}
    for k in (1, 2, 3):
        lo, hi = spans[f"English {k}"]; r0 = s.right - rd.right_rehearsing; n0 = len(rd.starts); rd.read(lo, hi); right[f"English {k}"] = 100 * (s.right - rd.right_rehearsing - r0) / (hi - lo); mark(f"end of English {k}"); notes[f"English {k}"] = dict(voices=voices(s), rehearsed=where(n0, len(rd.starts)))
        lo, hi = spans[f"return {k}"]; done = lo; n0 = len(rd.starts)
        for n in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000): rd.read(done, lo + n * 10); done = lo + n * 10; mark(f"return {k}, {n} lines")
        notes[f"return {k}"] = dict(rehearsed=where(n0, len(rd.starts)))
    out = dict(task="relay15", tag=tag, seed=seed, form=form, theta=theta, rho=rho, stretch=stretch, store=rd.M, episodes_offered=rd.offered, episodes_kept_at_the_end=where_kept(), lines=lines, values=S, bytes_left_out_of_english_2=left_out, bytes_read=int(cuts[4] - sp["A"][0]), bytes_rehearsed=int(rd.rehearsed), rows=rows, notes=notes, english_bytes_named_right_while_reading=right, wall=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(HERE, "res", f"relay15__{tag}.json"), "w")); return out

def check():
    """The reader: with no rehearsal it is a plain reading; with rehearsal the stream's own delay line is as if nothing had been rehearsed, every stretch lies wholly in what was read before it, and the bytes rehearsed are the stated share."""
    bad = 0; x, sp = AB.build(1, 5000, back=500, english=3000); vals = np.unique(x); idx = np.zeros(256, np.int64); idx[vals] = np.arange(len(vals)); xm = idx[x].astype(np.int32); S = len(vals); lo = sp["A"][0]; hi = lo + 30000
    mk = lambda: R12.Sys(3, 48, 10, S, int(idx[10]), eta1=1.0, eta2=1 / 16, lam=1e-5, rscale=1.0, soft=0, theta=4.0, **R12.switches("Rgtqr"))
    A = mk(); A.read(xm[lo - 1:hi - 1], xm[lo:hi]); B = mk(); Reader(B, xm, lo, 0.0, 30, 1).read(lo, hi); same = all((getattr(A, k) == getattr(B, k)).all() for k in ("W1t", "W2", "b2", "Rt")); bad += not same
    print(f"with no rehearsal the reader gives the plain reading bit for bit: {same} {'PASSED' if same else 'FAILED'}")
    C = mk(); rd = Reader(C, xm, lo, 0.05, 30, 1, store=100); cutp = lo + 17000; rd.read(lo, cutp); n1 = len(rd.starts); ok1 = all(lo + 10 <= r and r + 30 <= cutp for r in rd.starts); rd.read(cutp, hi); st = np.array(rd.starts)
    E = mk(); E.read(xm[lo - 1:hi - 1], xm[lo:hi]); ok2 = bool((C.hist == E.hist).all()); share = rd.rehearsed / (hi - lo); ok3 = abs(share - 0.05) < 0.002 and len(st) == (hi - lo) // 600; differs = not (C.W2 == E.W2).all()
    ok4 = len(rd.kept) == 100 and rd.offered == (hi - lo) // 40 and all((k - lo) % 40 == 0 and k + 40 <= hi for k in rd.kept) and len(set(rd.kept)) == 100
    ok = ok1 and ok2 and ok3 and ok4 and differs and all(lo + 10 <= r for r in st); bad += not ok
    print(f"with rehearsal at 0.05 in two calls and a store of 100 episodes: {len(st)} rehearsals ({n1} in the first call), each of an episode that had been read whole: {ok1}; the stream's delay line at the end is that of a plain reading: {ok2}; bytes rehearsed over bytes read {share:.4f}; the store holds 100 different episodes of the {rd.offered} offered: {ok4}; the strengths differ from a plain reading: {differs} {'PASSED' if ok else 'FAILED'}")
    # the store is an even sample: over many seeds, how often each tenth of the stream is among the kept
    cnt = np.zeros(10)
    for sd in range(200):
        r2 = Reader(mk(), xm, lo, 0.0, 30, sd, store=50); r2.K = 0; r2.next = lo; r2.offer(lo + 40000); cnt += np.histogram(np.array(r2.kept), bins=10, range=(lo, lo + 40000))[0]
    even = bool(np.abs(cnt / cnt.sum() - 0.1).max() < 0.01); bad += not even
    print(f"the store over 200 draws: the share of kept episodes from each tenth of the stream is between {cnt.min() / cnt.sum():.3f} and {cnt.max() / cnt.sum():.3f} (0.100 if even): {even} {'PASSED' if even else 'FAILED'}")
    # a rehearsed stretch teaches exactly its L bytes: one stretch by the reader against the same bytes read by hand into a copy
    F = mk(); F.read(xm[lo - 1:lo + 4999], xm[lo:lo + 5000]); G = copy.deepcopy(F); rd = Reader(F, xm, lo, 1.0, 30, 7); rd.offer(lo + 5000); rd.rehearse(lo + 5000); r = rd.starts[0]
    G.hist[:] = xm[r - 11:r - 1][::-1]; G.read(xm[r - 1:r + 29], xm[r:r + 30])      # the delay line as it stands before the byte at r - 1 arrives; ok = bool(np.allclose(F.W2, G.W2, rtol=0, atol=1e-12)) and bool(np.allclose(F.W1t, G.W1t, rtol=0, atol=1e-12)); bad += not ok
    print(f"one stretch read by the reader equals the same thirty bytes read after the ten bytes before them were put on the delay line by hand: {ok} {'PASSED' if ok else 'FAILED'}")
    print("CHECK", "FAILED" if bad else "PASSED"); return not bad

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=", 1) for a in sys.argv[2:]); out = run(kw); g = {r["at"]: r for r in out["rows"]}
    print(out["tag"], "| never-shown sums, whole answer (among digit branches; held-out English): " + " | ".join(f"{k}: {g[k]['whole']:.1f} ({g[k]['among_digits']:.1f}; {g[k]['english']:.1f})" for k in ["end of the sums"] + [f"{a}{b}" for n in (1, 2, 3) for a, b in ((f"end of English {n}", ""), (f"return {n}, ", "100 lines"), (f"return {n}, ", "5000 lines"))]), "| rehearsed", out["bytes_rehearsed"], "of", out["bytes_read"], "| wall", out["wall"], flush=True)
