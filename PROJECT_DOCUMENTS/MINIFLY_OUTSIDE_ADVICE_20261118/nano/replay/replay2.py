#!/usr/bin/env python3
"""nano/replay, PLAN_2.md: replay.py's system on a change of task that brings no new line: first the sums whose first number is below 50 ("lo"),
then the sums whose first number is 50 or above ("hi"). Every line is an addition; every byte value and every position occurs in both tasks.
    python3 replay2.py <label> seed=1 [who=P|R|O|Z] [r=1] [K=8] [plan=lo:400000,hi:200000] [every=50000] [fast=1]
The relay system, the counting cells and the replay are those of replay.py, imported unchanged. The split is cells2.py's (2,000 of the 10,000
pairs never shown); the never-shown pairs of each task are all of them that belong to it (about 1,000), not 500."""
import json, os, random, sys, time
import numpy as np
import replay
from replay import HERE, Sys, Counting, kind, text
TASK = dict(lo=lambda a, b: a < 50, hi=lambda a, b: a >= 50)

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]
    held = {k: [p for p in P[:2000] if f(*p)] for k, f in TASK.items()}; TAB = {k: np.array([text("+", a, b) for a, b in P[2000:] if f(a, b)], np.int32) for k, f in TASK.items()}
    s = Sys(seed); who = kw.get("who", "P"); r = float(eval(kw.get("r", "1"))); K = int(kw.get("K", 8)); cells = Counting(K); rnd = random.Random(seed * 1000 + 7)
    every = int(kw.get("every", 50000)); fast = int(kw.get("fast", 0)); out = dict(label=kw.get("label"), args=kw, rows=[], shown={k: len(v) for k, v in TAB.items()}, never_shown={k: len(v) for k, v in held.items()})
    plan = [(ph.split(":")[0], int(ph.split(":")[1])) for ph in kw.get("plan", "lo:400000,hi:200000").split(",")]; first, second = plan[0][0], plan[-1][0]; OLD = TAB[first].tolist(); t_rep = 0.0
    def go(xs, ys):
        if fast: import fastr; fastr.run(s, xs, ys)
        else: s.steps(xs, ys)
    for pi, (name, n) in enumerate(plan):
        it = 0
        if who == "Z" and pi > 0: cells.empty()
        while it < n:
            nxt = min((it // every + 1) * every, n) - 1; X = TAB[name][rng.integers(len(TAB[name]), size=nxt - it + 1)]; it = nxt + 1; st = dict(extra=0, x_first=0, x_second=0, well_formed=0, correct=0)
            if who == "P": lines = X
            else:
                t1 = time.process_time(); lines = []
                for L in X.tolist():
                    lines.append(L)
                    if who != "O": cells.count(L)
                    for _ in range(int(r) + (rnd.random() < r - int(r))):
                        E = OLD[rnd.randrange(len(OLD))] if who == "O" else cells.replay(rnd); wf, ok, op = kind(E); st["extra"] += 1; st["well_formed"] += wf; st["correct"] += ok
                        if wf: lines.append(E); a = E[0] * 10 + E[1]; st["x_first"] += TASK[first](a, 0); st["x_second"] += TASK[second](a, 0)
                lines = np.array(lines, np.int32); t_rep += time.process_time() - t1
            xs = lines.ravel(); ys = np.concatenate([lines[:, 1:], np.full((len(lines), 1), -1, np.int32)], 1).ravel(); go(xs, ys)
            out["rows"].append(dict(phase=pi, name=name, lines=nxt + 1, first=s.score("+", held[first]), second=s.score("+", held[second]), **st))
    ph = lambda i: [x for x in out["rows"] if x["phase"] == i]; nc, nz = cells.size()
    out.update(first_after_first=float(np.mean([x["first"] for x in ph(0)[-3:]])), second_after_first=float(np.mean([x["second"] for x in ph(0)[-3:]])), first_after_second=ph(1)[-1]["first"], second_after_second=ph(1)[-1]["second"],
               first_last3_second=float(np.mean([x["first"] for x in ph(1)[-3:]])), second_last3_second=float(np.mean([x["second"] for x in ph(1)[-3:]])), cells=nc, counts_above_zero=nz, cpu=round(time.process_time() - t0, 1), cpu_counting_and_replay=round(t_rep, 1))
    run.last = s
    if not kw.get("nowrite"):
        os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"q_{kw['label']}.json"), "w"))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw)
    print(o["label"], f"after the first phase: first task {o['first_after_first']:.1f}, second task (not yet taught) {o['second_after_first']:.1f}; after the second phase: first task {o['first_after_second']:.1f}, second task {o['second_after_second']:.1f} |",
          " ".join(f"{x['phase']}:{x['lines']//1000}k {x['first']:.0f}/{x['second']:.0f}" for x in o["rows"]), "| second phase, extra lines of the first task / of the second / correct of all:",
          " ".join(f"{x['x_first']}/{x['x_second']}/{x['correct']}of{x['extra']}" for x in o["rows"] if x["phase"] == 1), f"| cells {o['cells']} | cpu {o['cpu']} s ({o['cpu_counting_and_replay']} s counting and replay)")
