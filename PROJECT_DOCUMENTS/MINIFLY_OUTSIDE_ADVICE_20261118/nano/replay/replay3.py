#!/usr/bin/env python3
"""nano/replay, PLAN_3.md: replay.py's system on a harsh change of task that brings no new line. First addition, `aa+bb=ccc`; then subtraction
written with the bytes of addition, `aa=bb+ccc` for aa - bb = ccc, larger number first (the text of nano/keep's PLAN_8.md, test 3, read on
claude/nano-keep-8mmhfm at 6f2f61e and written here anew).
    python3 replay3.py <label> seed=1 [who=P|R|O|Z] [r=1] [K=8] [plan=add:400000,swp:200000] [every=50000] [fast=1]
The relay system, the counting cells and the replay are those of replay.py, imported unchanged. Scores: never-shown addition (500 pairs, as
before); the second task on the 500 never-shown pairs as cells2.py would score it ("mixed": a never-shown pair (a, b) with a < b writes the line
of (b, a), which may be shown), and on the never-shown pairs whose written line no shown pair writes ("clean" among the 500, "clean, wide" among
all 2,000 never shown)."""
import json, os, random, sys, time
import numpy as np
import replay
from replay import HERE, Sys, Counting, text, sg, SI, END, D, OFF, PLUS, EQ

def swp(a, b):
    if a < b: a, b = b, a
    return [SI[c] for c in f"{a:02d}={b:02d}+{a - b:03d}\n".encode()]
def kind(line):
    """(well formed, correct, task) of an extra line: 'add' for aa+bb=ccc, 'swp' for aa=bb+ccc."""
    if len(line) != 10 or line[9] != END or any(not 0 <= line[i] <= 9 for i in (0, 1, 3, 4, 6, 7, 8)): return False, False, None
    a, b, c = line[0] * 10 + line[1], line[3] * 10 + line[4], line[6] * 100 + line[7] * 10 + line[8]
    if line[2] == PLUS and line[5] == EQ: return True, a + b == c, "add"
    if line[2] == EQ and line[5] == PLUS: return True, a >= b and a - b == c, "swp"
    return False, False, None
def score(s, Ls):
    """Whole answer right (the three bytes of ccc) on the lines Ls; the arithmetic of replay.Sys.score."""
    Ls = np.array(Ls); ok = np.ones(len(Ls), bool); W1T = s.W1.T
    for t in (5, 6, 7):
        hist = np.full((len(Ls), D), END); hist[:, :t + 1] = Ls[:, t::-1]; ok &= (sg(W1T[OFF + hist].sum(1) + s.b1) @ s.W2.T + s.b2).argmax(1) == Ls[:, t + 1]
    return float(100 * ok.sum() / len(Ls))

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng = np.random.default_rng([seed, 99]); PP = {}
    for op in "+-": o = rng.permutation(10000); PP[op] = [(int(q) // 100, int(q) % 100) for q in o]          # the two splits of cells2.py, in its order
    WR = {"add": lambda a, b: text("+", a, b), "swp": swp}; SPL = {"add": PP["+"], "swp": PP["-"]}
    TAB = {k: np.array([WR[k](a, b) for a, b in SPL[k][2000:]], np.int32) for k in WR}; shown_lines = set(map(tuple, TAB["swp"].tolist()))
    held = {"add": [WR["add"](a, b) for a, b in SPL["add"][:500]], "swp": [swp(a, b) for a, b in SPL["swp"][:500]]}
    clean = [L for L in held["swp"] if tuple(L) not in shown_lines]; clean_wide = [swp(a, b) for a, b in SPL["swp"][:2000] if tuple(swp(a, b)) not in shown_lines]
    assert not set(map(tuple, held["add"])) & set(map(tuple, TAB["add"].tolist()))
    s = Sys(seed); who = kw.get("who", "P"); r = float(eval(kw.get("r", "1"))); K = int(kw.get("K", 8)); cells = Counting(K); rnd = random.Random(seed * 1000 + 7)
    every = int(kw.get("every", 50000)); fast = int(kw.get("fast", 0)); out = dict(label=kw.get("label"), args=kw, rows=[], n_clean=len(clean), n_clean_wide=len(clean_wide)); OLD = TAB["add"].tolist(); t_rep = 0.0
    def go(xs, ys):
        if fast: import fastr; fastr.run(s, xs, ys)
        else: s.steps(xs, ys)
    for pi, ph in enumerate(kw.get("plan", "add:400000,swp:200000").split(",")):
        name, n = ph.split(":"); n = int(n); it = 0
        if who == "Z" and pi > 0: cells.empty()
        while it < n:
            nxt = min((it // every + 1) * every, n) - 1; X = TAB[name][rng.integers(8000, size=nxt - it + 1)]; it = nxt + 1; st = dict(extra=0, x_add=0, x_swp=0, well_formed=0, correct=0)
            if who == "P": lines = X
            else:
                t1 = time.process_time(); lines = []
                for L in X.tolist():
                    lines.append(L)
                    if who != "O": cells.count(L)
                    for _ in range(int(r) + (rnd.random() < r - int(r))):
                        E = OLD[rnd.randrange(8000)] if who == "O" else cells.replay(rnd); wf, ok, tk = kind(E); st["extra"] += 1; st["well_formed"] += wf; st["correct"] += ok; st["x_add"] += tk == "add"; st["x_swp"] += tk == "swp"
                        if wf: lines.append(E)
                lines = np.array(lines, np.int32); t_rep += time.process_time() - t1
            xs = lines.ravel(); ys = np.concatenate([lines[:, 1:], np.full((len(lines), 1), -1, np.int32)], 1).ravel(); go(xs, ys)
            out["rows"].append(dict(phase=pi, name=name, lines=nxt + 1, add=score(s, held["add"]), swp=score(s, held["swp"]), swp_clean=score(s, clean), swp_clean_wide=score(s, clean_wide), **st))
    ph = lambda i: [x for x in out["rows"] if x["phase"] == i]; nc, nz = cells.size(); e = ph(1)[-1] if ph(1) else {}
    out.update(add_after_first=float(np.mean([x["add"] for x in ph(0)[-3:]])), swp_after_first=ph(0)[-1]["swp_clean_wide"], add_after_second=e.get("add"), swp_after_second=e.get("swp"), swp_clean_after_second=e.get("swp_clean"),
               swp_clean_wide_after_second=e.get("swp_clean_wide"), cells=nc, counts_above_zero=nz, cpu=round(time.process_time() - t0, 1), cpu_counting_and_replay=round(t_rep, 1)); run.last = s
    if not kw.get("nowrite"):
        os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"h_{kw['label']}.json"), "w"))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw)
    print(o["label"], f"addition after its phase {o['add_after_first']:.1f}; at the end: addition {o['add_after_second']}, second task mixed {o['swp_after_second']}, clean wide {o['swp_clean_wide_after_second']:.1f} ({o['n_clean_wide']} pairs) |",
          " ".join(f"{x['phase']}:{x['lines']//1000}k {x['add']:.0f}/{x['swp_clean_wide']:.0f}" for x in o["rows"]), "| second phase, extra lines addition/second task/correct of all:",
          " ".join(f"{x['x_add']}/{x['x_swp']}/{x['correct']}of{x['extra']}" for x in o["rows"] if x["phase"] == 1), f"| cells {o['cells']} | cpu {o['cpu']} s")
