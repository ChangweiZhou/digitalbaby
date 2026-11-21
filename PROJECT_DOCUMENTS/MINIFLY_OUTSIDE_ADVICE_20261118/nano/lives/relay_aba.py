#!/usr/bin/env python3
"""nano/lives, PLAN_11.md, learner 3: the relay network (nano/loop/loop.py's system, which is loop_small_case/cells2.py's with the alphabet as an argument)
through the same life: sums, English, sums again. One alphabet for both texts.
    python3 relay_aba.py seed=1 form=R|R0 [lines=400000] tag=..
R: as it is. R0: the fading of the answering strengths and of the return strengths is switched off from the English on.
With form R0 the return is also read by a copy in which the cells in between are held still (only the answering cell learns): a diagnostic.
Additions of PLAN_11.md, section 4b (both forms): the first 5,000 lines are also read by a copy whose cells in between stay as they were born; after the
English, parts of the end of the sums are put back in copies and scored; how many cells are active in a state of the sums and of the English."""
import copy, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "loop"))
import aba as AB, loop as LP

def run(kw):
    seed = int(kw.get("seed", 1)); form = kw["form"]; lines = int(kw.get("lines", 400000)); tag = kw["tag"]; D = 10; t0 = time.time()
    x, sp = AB.build(seed, lines); vals = np.unique(x); idx = np.zeros(256, np.int64); idx[vals] = np.arange(len(vals)); xm = idx[x].astype(np.int32); S = len(vals)
    s = LP.Sys(seed, 256, D, S, int(idx[10]), eta1=1.0, eta2=1 / 16, lam=1e-5, rscale=1.0, soft=0); ev = xm[sp["ev"]:].reshape(5, 500, 20); rows = []      # the step 1 is the one that gives the recorded level on sums (PLAN_11.md, section 4a)
    def never_shown(sy):
        out = []
        for d in range(5):
            prev = ev[d, :, :10]; Ls = ev[d, :, 10:]; ok = np.ones(500, bool)
            for t in (5, 6, 7):
                hist = np.zeros((500, D), np.int64); hist[:, :t + 1] = Ls[:, t::-1]; hist[:, t + 1:] = prev[:, ::-1][:, :D - t - 1]; ok &= sy.branches(hist).argmax(1) == Ls[:, t + 1]
            out.append(100 * ok.mean())
        return float(np.mean(out))
    def read(sy, lo, hi): sy.read(xm[lo - 1:hi - 1], xm[lo:hi])
    def mark(label, sy=None): v = never_shown(sy or s); rows.append(dict(at=label, whole=v)); print(tag, label, round(v, 1), flush=True)
    a0, b0 = sp["A"]; done = a0
    born = copy.deepcopy(s); born.eta1 = 0.0                                             # addition 3: a network whose cells in between stay as born; only its answering cell learns
    for n in (500, 1000, 2000, 5000): read(born, done, a0 + n * 10); done = a0 + n * 10; mark(f"first {n} lines, the cells in between as born and held still", born)
    done = a0
    for n in sorted({500, 1000, 2000, 5000, 40000, lines}):
        if n <= lines: read(s, done, a0 + n * 10); done = a0 + n * 10; mark(f"sums {n} lines")
    rows[-1]["at"] = "end of the sums"
    # the states in which the never-shown answers are named (the true bytes before them), and 5,000 states of the English, as the lines that are active
    HS = []
    for d in range(5):
        prev = ev[d, :, :10]; Ls = ev[d, :, 10:]
        for t in (5, 6, 7): hist = np.zeros((500, D), np.int64); hist[:, :t + 1] = Ls[:, t::-1]; hist[:, t + 1:] = prev[:, ::-1][:, :D - t - 1]; HS.append(hist)
    HS = np.concatenate(HS); q = np.linspace(sp["B"][0] + D, sp["B"][1] - 1, 5000).astype(np.int64); HE = np.stack([xm[q - 1 - j] for j in range(D)], 1).astype(np.int64)
    def acts(sy, hist): return LP.sg(sy.W1t[hist + sy.OFF].sum(1) + sy.b1)
    def code(sy):
        hs = acts(sy, HS); he = acts(sy, HE); quiet = he.mean(0) < 0.05
        return dict(mean_activity_sums=float(hs.mean()), mean_activity_english=float(he.mean()), cells_above_half_in_a_state_sums=float((hs > 0.5).sum(1).mean()), cells_above_half_in_a_state_english=float((he > 0.5).sum(1).mean()),
                    share_of_activities_under_005_sums=float((hs < 0.05).mean()), share_of_activities_under_005_english=float((he < 0.05).mean()), cells_quiet_through_the_english=int(quiet.sum()),
                    share_of_the_sums_activity_on_cells_quiet_through_the_english=float(hs[:, quiet].sum() / hs.sum()))
    kept = dict(W1t=s.W1t.copy(), b1=s.b1.copy(), W2=s.W2.copy(), b2=s.b2.copy()); hs_a = acts(s, HS); codes = dict(end_of_the_sums=code(s)); digits = np.array([int(idx[v]) for v in range(48, 58)])
    if form == "R0": s.lam = 0.0
    r0 = s.right; read(s, sp["B"][0], sp["B"][1]); english_right = 100 * (s.right - r0) / (sp["B"][1] - sp["B"][0]); mark("end of the English")
    def with_parts(names):
        c = copy.deepcopy(s)
        for k in names: getattr(c, k)[...] = kept[k]
        return c
    mark("end of the English, the answering strengths of the end of the sums put back", with_parts(("W2", "b2")))            # addition 2
    mark("end of the English, the cells in between of the end of the sums put back", with_parts(("W1t", "b1")))
    mark("end of the English, the answering strengths and the cells' thresholds of the end of the sums put back", with_parts(("W2", "b2", "b1")))
    codes["end_of_the_english"] = code(s); codes["mean_change_of_a_cell_activity_in_the_sums_states_through_the_english"] = float(np.abs(acts(s, HS) - hs_a).mean())      # addition 4
    codes["answering_strengths_of_the_digit_branches_size_after_over_before"] = float(np.linalg.norm(s.W2[digits]) / np.linalg.norm(kept["W2"][digits])); codes["answering_strengths_of_the_digit_branches_change_over_size_before"] = float(np.linalg.norm(s.W2[digits] - kept["W2"][digits]) / np.linalg.norm(kept["W2"][digits]))
    print(tag, "code", json.dumps({k: (round(v, 3) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()}) for k, v in codes.items()}), flush=True)
    held = copy.deepcopy(s) if form == "R0" else None; done = sp["A2"][0]
    for n in (500, 1000, 2000, 5000): hi = sp["A2"][0] + n * 10; read(s, done, hi); done = hi; mark(f"return {n} lines")
    if held is not None:
        held.eta1 = 0.0; done = sp["A2"][0]
        for n in (500, 1000, 2000, 5000): hi = sp["A2"][0] + n * 10; read(held, done, hi); done = hi; mark(f"return {n} lines, the cells in between held still", held)
    out = dict(task="relay_aba", tag=tag, seed=seed, form=form, lines=lines, values=S, rows=rows, code=codes, english_bytes_named_right_while_reading=english_right, wall=round(time.time() - t0, 1)); json.dump(out, open(os.path.join(HERE, "res", f"relay_aba__{tag}.json"), "w")); return out

if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[1:]); out = run(kw); print(out["tag"], "| never-shown sums, whole answer:", " | ".join(f"{r['at']}: {r['whole']:.1f}" for r in out["rows"]), "| wall", out["wall"], flush=True)
