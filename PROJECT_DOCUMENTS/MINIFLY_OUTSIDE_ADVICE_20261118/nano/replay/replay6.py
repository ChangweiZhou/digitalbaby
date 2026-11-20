#!/usr/bin/env python3
"""nano/replay, PLAN_6.md: replay with the line chosen by the relay system's own mismatch. A repeat, with this line's own program, of a rule
tried by the line nano/life3 (its prose read on claude/nano-life3-8mmhfm at b96cd74; its program not read).
    python3 replay6.py <label> seed=1 [who=R|M] [n=4] [plan=add:400000,sub:200000 | add:400000,swp:200000] [every=50000] [chunk=500] [fast=1]
    who=R: arm R of replay.py (one line replayed from the counting cells after each real line, K = 8).
    who=M: after each real line n lines are drawn from the counting cells exactly as R draws one; the relay system learns only from the one on
           which its own squared mismatch, summed over the fourteen branches and the nine bytes of the line it would learn from, is largest.
           Nothing names a task. The mismatch is taken with the line read alone after a newline, and with the relay system's strengths as they
           were at the start of the chunk of `chunk` real lines (at most `chunk` real lines old).  With fast=1 the judging runs compiled too (kernm.c through fastm.py); the numpy function mismatch() here is its reference.
Texts: add `aa+bb=ccc`; sub `aa-bb=ccc` (cells2.py's); swp `aa=bb+ccc` for aa - bb = ccc (replay3.py's). The relay system, the counting cells and
the replay are replay.py's, imported unchanged."""
import json, os, random, sys, time
import numpy as np
import replay, replay3
from replay import HERE, Sys, Counting, text, sg, S, END, D, OFF

def mismatch(s, L):
    """For each line of L (n by 10): the sum over its nine learned bytes and the fourteen branches of the squared mismatch."""
    L = np.asarray(L); out = np.zeros(len(L)); W1T = s.W1.T; rows = np.arange(len(L))
    for t in range(9):
        hist = np.full((len(L), D), END); hist[:, :t + 1] = L[:, t::-1]; m = -(sg(W1T[OFF + hist].sum(1) + s.b1) @ s.W2.T + s.b2); m[rows, L[:, t + 1]] += 1; out += (m * m).sum(1)
    return out

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng = np.random.default_rng([seed, 99]); PP = {}
    for op in "+-": o = rng.permutation(10000); PP[op] = [(int(x) // 100, int(x) % 100) for x in o]
    WR = dict(add=lambda a, b: text("+", a, b), sub=lambda a, b: text("-", a, b), swp=replay3.swp); SPL = dict(add=PP["+"], sub=PP["-"], swp=PP["-"])
    plan = [(p.split(":")[0], int(p.split(":")[1])) for p in kw.get("plan", "add:400000,sub:200000").split(",")]; first, second = plan[0][0], plan[-1][0]
    TAB = {k: np.array([WR[k](a, b) for a, b in SPL[k][2000:]], np.int32) for k in (first, second)}; shown2 = set(map(tuple, TAB[second].tolist()))
    held1 = [WR[first](a, b) for a, b in SPL[first][:500]]; mixed2 = [WR[second](a, b) for a, b in SPL[second][:500]]; clean2 = [WR[second](a, b) for a, b in SPL[second][:2000] if tuple(WR[second](a, b)) not in shown2]
    s = Sys(seed); who = kw.get("who", "R"); n_c = int(kw.get("n", 4)); cells = Counting(int(kw.get("K", 8))); rnd = random.Random(seed * 1000 + 7)
    every = int(kw.get("every", 50000)); chunk = int(kw.get("chunk", 500)); fast = int(kw.get("fast", 0)); out = dict(label=kw.get("label"), args=kw, rows=[], n_clean=len(clean2)); t_rep = t_judge = 0.0; opi = 2
    def judge(s_, flat):
        import fastm; return fastm.judge(s_, flat, END)
    def go(lines):
        lines = np.array(lines, np.int32); xs = lines.ravel(); ys = np.concatenate([lines[:, 1:], np.full((len(lines), 1), -1, np.int32)], 1).ravel()
        if fast: import fastr; fastr.run(s, xs, ys)
        else: s.steps(xs, ys)
    isfirst = (lambda L: bool(L[2] == int(TAB[first][0][2])))                 # the third byte of a line tells the two texts apart; used for the report only
    for pi, (name, n) in enumerate(plan):
        it = 0
        while it < n:
            nxt = min((it // every + 1) * every, n) - 1; X = TAB[name][rng.integers(8000, size=nxt - it + 1)].tolist(); it = nxt + 1; st = dict(learned=0, learned_first=0, drawn=0, drawn_first=0, mm_chosen=0.0, mm_drawn=0.0)
            for c0 in range(0, len(X), chunk):
                t1 = time.process_time(); part = X[c0:c0 + chunk]; cand = []
                for Lr in part:
                    cells.count(Lr); rnd.random()                # replay.py draws once here for the fractional part of r; kept so that who=R is its arm R to the last draw
                    cand.append([cells.replay(rnd) for _ in range(n_c if who == "M" else 1)])
                t_rep += time.process_time() - t1; lines = []
                if who == "M":
                    t1 = time.process_time(); flat = [E for cs in cand for E in cs]; mm = (judge(s, flat) if fast else mismatch(s, flat)).reshape(len(part), n_c); pick = mm.argmax(1); t_judge += time.process_time() - t1
                    st["mm_chosen"] += float(mm[np.arange(len(part)), pick].sum()); st["mm_drawn"] += float(mm.sum())
                else: pick = np.zeros(len(part), int)
                for Lr, cs, k in zip(part, cand, pick):
                    lines.append(Lr); lines.append(cs[k]); st["learned"] += 1; st["learned_first"] += int(isfirst(cs[k])); st["drawn"] += len(cs); st["drawn_first"] += int(sum(isfirst(E) for E in cs))
                go(lines)
            out["rows"].append(dict(phase=pi, name=name, lines=nxt + 1, first=replay3.score(s, held1), second_mixed=replay3.score(s, mixed2), second_clean=replay3.score(s, clean2), **st))
    ph = lambda i: [x for x in out["rows"] if x["phase"] == i]; nc, nz = cells.size(); e = ph(1)[-1]
    out.update(first_after_first=float(np.mean([x["first"] for x in ph(0)[-3:]])), first_at_end=e["first"], second_clean_at_end=e["second_clean"], second_mixed_at_end=e["second_mixed"], cells=nc, counts_above_zero=nz,
               cpu=round(time.process_time() - t0, 1), cpu_counting_and_replay=round(t_rep, 1), cpu_judging=round(t_judge, 1)); run.last = s
    if not kw.get("nowrite"):
        os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"m_{kw['label']}.json"), "w"))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw)
    print(o["label"], f"first task after its phase {o['first_after_first']:.1f}; at the end: first task {o['first_at_end']:.1f}, second task clean {o['second_clean_at_end']:.1f} (mixed {o['second_mixed_at_end']:.1f}) |",
          " ".join(f"{x['phase']}:{x['lines']//1000}k {x['first']:.0f}/{x['second_clean']:.0f}" for x in o["rows"]), "| second phase, share of first-task lines among drawn / learned, %:",
          " ".join(f"{100*x['drawn_first']/x['drawn']:.0f}/{100*x['learned_first']/x['learned']:.0f}" for x in o["rows"] if x["phase"] == 1), f"| cpu {o['cpu']} s (replay {o['cpu_counting_and_replay']}, judging {o['cpu_judging']})")
