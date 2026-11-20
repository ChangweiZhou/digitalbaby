#!/usr/bin/env python3
"""Checks of replay.py.  (1) The relay system's compiled loop against its numpy loop, same seed and lines, in every arm, both phases, every strength
compared (the manner of loop_small_case/check_fast.py).  (2) The plain arm against loop_small_case/cells2.py (numpy against numpy, two phases).
(3) The counting cells: with K = 8 every replayed line is a line that was counted, and lines come back in proportion to how often they were counted;
with K = 6 some are not lines that were counted.      python3 check_replay.py"""
import os, random, sys, time
import numpy as np
import replay
NAMES = ("W1", "b1", "W2", "b2", "R")
def one(kw, fast):
    t = time.process_time(); o = replay.run(dict(kw, label="chk", nowrite="1", fast=fast)); return replay.run.last, o, time.process_time() - t
def diff(a, b): return max(np.abs(getattr(a, k) - getattr(b, k)).max() / (np.abs(getattr(a, k)).max() + 1e-300) for k in NAMES)
if __name__ == "__main__":
    bad = 0; base = dict(seed="3", plan="add:200,sub:100", every="50")
    for kw in (dict(who="P"), dict(who="R", r="1"), dict(who="R", r="1/4"), dict(who="R", r="2"), dict(who="O", r="1"), dict(who="Z", r="1"), dict(who="R", r="1", K="6")):
        (a, oa, ta), (b, ob, tb) = one(dict(base, **kw), "0"), one(dict(base, **kw), "1"); w = diff(a, b); ex = sum(x["extra"] for x in oa["rows"]); same = [x["extra"] for x in oa["rows"]] == [x["extra"] for x in ob["rows"]]; bad += not (w < 1e-6 and same)
        print(f"{' '.join(f'{k}={v}' for k, v in kw.items()):16s}: largest difference over largest strength {w:.1e}; extra lines {ex} (correct {sum(x['correct'] for x in oa['rows'])}, addition {sum(x['x_add'] for x in oa['rows'])}, "
              f"subtraction {sum(x['x_sub'] for x in oa['rows'])}); numpy {ta:4.1f} s, compiled {tb:4.1f} s")
    sys.path.insert(0, os.path.join(replay.HERE, "..", "..", "loop_small_case")); import cells2
    cells2.run(dict(seed="3", plan="add:200,sub:100", every="50", label="chk", nowrite="1")); c = cells2.run.last; a, oa, _ = one(dict(base, who="P"), "0"); w = diff(a, c); bad += not w < 1e-6
    print(f"the plain arm (numpy) against cells2.py (numpy), add:200,sub:100: {w:.1e}")
    rng = np.random.default_rng(5); lines = [replay.text("+-"[int(rng.integers(2))], int(rng.integers(100)), int(rng.integers(100))) for _ in range(3000)]; seen = {}
    for L in lines: seen[tuple(L)] = seen.get(tuple(L), 0) + 1
    for K in (10, 8, 6):
        cells = replay.Counting(K); rnd = random.Random(1); got = {}
        for L in lines: cells.count(L)
        for _ in range(30000): R = tuple(cells.replay(rnd)); got[R] = got.get(R, 0) + 1
        inside = sum(v for k, v in got.items() if k in seen); corr = sum(v for k, v in got.items() if replay.kind(list(k))[1]); wf = sum(v for k, v in got.items() if replay.kind(list(k))[0])
        dev = max(abs(got.get(k, 0) / 30000 - v / 3000) for k, v in seen.items())
        print(f"K={K:2d}: of 30,000 replayed lines {inside} are lines that were counted, {wf} well formed, {corr} correct; largest gap between a line's share of the replays and its share of the counted lines {dev:.4f}; cells {cells.size()[0]}, counts above zero {cells.size()[1]}")
        if K >= 8: bad += inside != 30000 or dev > 0.003
    print("CHECK", "FAILED" if bad else "PASSED")
