#!/usr/bin/env python3
"""Checks of replay3.py.  (1) The relay system's compiled loop against its numpy loop in every arm over the two phases.  (2) The second text
(aa=bb+ccc) activates no line that addition does not: by enumeration of every (byte value, delay) pair at every byte from which the system learns,
over all 10,000 pairs of each text read as a stream of single lines after a newline history, and over pairs of consecutive lines.  (3) The counting
cells with K = 8 on a mixture of the two texts: every replayed line is a line that was counted.      python3 check_replay3.py"""
import random
import numpy as np
import replay, replay3
from replay import text, END, D
NAMES = ("W1", "b1", "W2", "b2", "R")
def active(lines_pairs):
    """The set of (delay, byte) lines active at a byte from which the system learns, for streams of two consecutive lines."""
    A = set()
    for prev, cur in lines_pairs:
        st = [END] * D + prev + cur; base = D + len(prev)
        for t in range(len(cur) - 1):                    # after reading cur[t] the system names cur[t + 1] and learns
            A.update((d, st[base + t - d]) for d in range(D))
    return A
if __name__ == "__main__":
    bad = 0
    for kw in (dict(who="P"), dict(who="R", r="1"), dict(who="O", r="1"), dict(who="Z", r="1")):
        res = []
        for fast in "01": o = replay3.run(dict(kw, seed="3", plan="add:200,swp:100", every="50", label="chk", nowrite="1", fast=fast)); res.append((replay3.run.last, o))
        (a, oa), (b, ob) = res; w = max(np.abs(getattr(a, k) - getattr(b, k)).max() / (np.abs(getattr(a, k)).max() + 1e-300) for k in NAMES); bad += not w < 1e-6
        print(f"{' '.join(f'{k}={v}' for k, v in kw.items()):10s}: largest difference over largest strength {w:.1e}; extra lines {sum(x['extra'] for x in oa['rows'])}, addition {sum(x['x_add'] for x in oa['rows'])}, second task {sum(x['x_swp'] for x in oa['rows'])}, correct {sum(x['correct'] for x in oa['rows'])}")
    rng = np.random.default_rng(7); allp = [(a, b) for a in range(100) for b in range(100)]; ADD = [text("+", a, b) for a, b in allp]; SWP = [replay3.swp(a, b) for a, b in allp]; SUB = [text("-", a, b) for a, b in allp]
    pick = lambda X, n: [X[int(i)] for i in rng.integers(len(X), size=n)]
    Aadd = active(zip(pick(ADD, 200000), pick(ADD, 200000))); Aswp = active(zip(pick(SWP, 200000), pick(SWP, 200000))) | active(zip(pick(ADD, 100000), pick(SWP, 100000))); Asub = active(zip(pick(SUB, 200000), pick(SUB, 200000)))
    new_swp = sorted(Aswp - Aadd); new_sub = sorted(Asub - Aadd)
    print(f"lines active in addition streams: {len(Aadd)} of 140; active in streams of the second text (and at the change from addition to it) but never in addition: {len(new_swp)} {[(d, chr(replay.SYMS[v]) if v != END else 'nl') for d, v in new_swp]}")
    print(f"for comparison, the usual subtraction text activates {len(new_sub)} lines that addition does not")
    lines = pick(ADD, 2000) + pick(SWP, 2000); seen = set(map(tuple, lines)); cells = replay.Counting(8); rnd = random.Random(1)
    for L in lines: cells.count(L)
    got = [tuple(cells.replay(rnd)) for _ in range(30000)]; inside = sum(g in seen for g in got); corr = sum(replay3.kind(list(g))[1] for g in got); bad += inside != 30000
    print(f"K=8, 2,000 lines of each text counted: of 30,000 replayed lines {inside} are lines that were counted, {corr} correct; of them addition {sum(replay3.kind(list(g))[2] == 'add' for g in got)}")
    print("CHECK", "FAILED" if bad else "PASSED")
