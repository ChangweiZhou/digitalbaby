#!/usr/bin/env python3
"""nano/lives, PLAN_13.md: the life of PLAN_12.md with three returns in place of one. Sums, English, sums, English, sums, English, sums.
    python3 relay13.py life seed=1 form=R|Rgtqr [theta=4] [lines=400000] tag=..
The network, its switches and its kernel are relay12.py's (nothing is changed in the learner). The stream up to the end of the first return is
relay12.py's byte for byte, so the marks up to there must repeat PLAN_12.md's. The second and third pieces of English are the bytes 420,000 to
760,000 and 760,000 to 1,100,000 of tiny Shakespeare (one byte, a '$', whose value the alphabet of the life lacks, is left out of the second);
the bytes 400,000 to 420,000 are never read and are the held-out English. Each return is 5,000 fresh draws of the shown pairs."""
import copy, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "loop"))
import aba as AB, loop as LP, relay12 as R12

def run(kw):
    seed = int(kw.get("seed", 1)); form = kw["form"]; lines = int(kw.get("lines", 400000)); theta = float(kw.get("theta", 4.0)); tag = kw["tag"]; D = 10; t0 = time.time()
    x, sp = AB.build(seed, lines); eng = AB.LV.raw("shakespeare")[0]; held_eng = eng[400000:420000]; vals = np.unique(x); idx = np.zeros(256, np.int64); idx[vals] = np.arange(len(vals)); S = len(vals)
    rg = np.random.default_rng([seed, 99]); o = rg.permutation(10000); pairs = [(int(q) // 100, int(q) % 100) for q in o]; TAB = np.array([AB.text(a, b) for a, b in pairs[2000:]], np.int64)
    B2 = eng[420000:760000]; left_out = int((~np.isin(B2, vals)).sum()); B2 = B2[np.isin(B2, vals)]; B3 = eng[760000:1100000]; assert np.isin(B3, vals).all() and np.isin(held_eng, vals).all()
    more = [B2, TAB[np.random.default_rng([seed, 56]).integers(8000, size=5000)].ravel(), B3, TAB[np.random.default_rng([seed, 57]).integers(8000, size=5000)].ravel()]
    head = x[:sp["A2"][1]]; xm = idx[np.concatenate([head] + more)].astype(np.int32); cuts = np.cumsum([len(head)] + [len(m) for m in more])            # relay12's stream to the end of the first return, then English 2, return 2, English 3, return 3
    spans = {"sums": sp["A"], "English 1": sp["B"], "return 1": sp["A2"], "English 2": (int(cuts[0]), int(cuts[1])), "return 2": (int(cuts[1]), int(cuts[2])), "English 3": (int(cuts[2]), int(cuts[3])), "return 3": (int(cuts[3]), int(cuts[4]))}
    s = R12.Sys(seed, 256, D, S, int(idx[10]), eta1=1.0, eta2=1 / 16, lam=1e-5, rscale=1.0, soft=0, theta=theta, **R12.switches(form)); ev = idx[x[sp["ev"]:]].reshape(5, 500, 20); rows = []; digits = np.array([int(idx[v]) for v in range(48, 58)]); he = idx[held_eng].astype(np.int64)
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
    def read(sy, lo, hi): sy.read(xm[lo - 1:hi - 1], xm[lo:hi])
    def mark(label, sy=None): sy = sy or s; r = dict(at=label, whole=whole(sy), among_digits=whole(sy, digits), english=english(sy), shown_whole=whole(sy, None, SH)); rows.append(r); print(tag, label, {k: round(v, 1) for k, v in r.items() if k != "at"}, flush=True)
    def acts(sy, hist): return LP.sg(sy.W1t[hist + sy.OFF].sum(1) + sy.b1)
    def voices(sy):
        a = sy.branches(HSall); right = a[np.arange(len(YS)), YS]; o_ = a[:, others]; top = o_.argmax(1); loud = o_[np.arange(len(YS)), top]
        return dict(right_digit_output=float(right.mean()), loudest_other_branch_output=float(loud.mean()), share_of_states_where_another_branch_is_loudest=float((loud > a[:, digits].max(1)).mean()))
    def code(sy, lo, hi):
        q = np.linspace(lo + D, hi - 1, 5000).astype(np.int64); HB = np.stack([xm[q - 1 - j] for j in range(D)], 1).astype(np.int64); hs = acts(sy, HSall); hb = acts(sy, HB); quiet = hb.mean(0) < 0.05
        fs = (hs > 0.5).mean(0) >= 0.01; fe = (hb > 0.5).mean(0) >= 0.01                                    # a cell of the sums / of the English: above one half in at least one state in a hundred
        return dict(cells_above_half_in_a_state_sums=float((hs > 0.5).sum(1).mean()), cells_above_half_in_a_state_english=float((hb > 0.5).sum(1).mean()), cells_quiet_through_this_english=int(quiet.sum()), share_of_the_sums_activity_on_cells_quiet_through_this_english=float(hs[:, quiet].sum() / hs.sum()),
                    cells_of_the_sums=int(fs.sum()), cells_of_the_english=int(fe.sum()), cells_of_both=int((fs & fe).sum()), share_of_the_english_activity_on_cells_of_the_sums=float(hb[:, fs].sum() / hb.sum()), share_of_the_sums_activity_on_cells_of_the_english=float(hs[:, fe].sum() / hs.sum()),
                    share_of_english_states_with_a_line_feed_in_the_last_ten=float((HB == int(idx[10])).any(1).mean()))
    read(s, *spans["sums"]); mark("end of the sums"); notes = {}; right = {}
    for k in (1, 2, 3):
        kept = dict(W2=s.W2.copy(), b2=s.b2.copy()); lo, hi = spans[f"English {k}"]; r0 = s.right; read(s, lo, hi); right[f"English {k}"] = 100 * (s.right - r0) / (hi - lo); mark(f"end of English {k}")
        c = copy.deepcopy(s); c.W2[...] = kept["W2"]; c.b2[...] = kept["b2"]; mark(f"end of English {k}, the answering strengths of before it put back", c); notes[f"English {k}"] = dict(voices=voices(s), code=code(s, lo, hi), digit_strengths_change_over_size_before=float(np.linalg.norm(s.W2[digits] - kept["W2"][digits]) / np.linalg.norm(kept["W2"][digits])))
        lo, hi = spans[f"return {k}"]; done = lo
        for n in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000): read(s, done, lo + n * 10); done = lo + n * 10; mark(f"return {k}, {n} lines")
    out = dict(task="relay13", tag=tag, seed=seed, form=form, theta=theta, lines=lines, values=S, bytes_left_out_of_english_2=left_out, rows=rows, notes=notes, english_bytes_named_right_while_reading=right, wall=round(time.time() - t0, 1)); json.dump(out, open(os.path.join(HERE, "res", f"relay13__{tag}.json"), "w")); return out

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True); kw = dict(a.split("=", 1) for a in sys.argv[2:]); out = run(kw); g = {r["at"]: r for r in out["rows"]}
    print(out["tag"], "| never-shown sums, whole answer (among digit branches; held-out English): " + " | ".join(f"{k}: {g[k]['whole']:.1f} ({g[k]['among_digits']:.1f}; {g[k]['english']:.1f})" for k in ["end of the sums"] + [f"{a}{b}" for n in (1, 2, 3) for a, b in ((f"end of English {n}", ""), (f"return {n}, ", "100 lines"), (f"return {n}, ", "5000 lines"))]), "| wall", out["wall"], flush=True)
