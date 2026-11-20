#!/usr/bin/env python3
"""Checks of replay6.py.  (1) The relay system's compiled loop against its numpy loop, arms R and M, both lives.  (2) who=R is arm R of replay.py
(sub) and of replay3.py (swp): the same strengths after the same life.  (3) The mismatch used for choosing is the mismatch the relay system has
when it reads that line alone after a newline: compared with the numpy learning loop's own mismatch on such a line.      python3 check_replay6.py"""
import numpy as np
import replay, replay3, replay6
from replay import sg, OFF, D, END
NAMES = ("W1", "b1", "W2", "b2", "R")
def diff(a, b): return max(np.abs(getattr(a, k) - getattr(b, k)).max() / (np.abs(getattr(a, k)).max() + 1e-300) for k in NAMES)
if __name__ == "__main__":
    bad = 0
    for second in ("sub", "swp"):
        base = dict(seed="3", plan=f"add:600,{second}:300", every="100", chunk="50", label="chk", nowrite="1")
        for who in "RM":
            res = []
            for fast in "01": o = replay6.run(dict(base, who=who, fast=fast)); res.append((replay6.run.last, o))
            (a, oa), (b, ob) = res; w = diff(a, b); same = [x["learned_first"] for x in oa["rows"]] == [x["learned_first"] for x in ob["rows"]]; bad += not (w < 1e-6 and same)
            print(f"add then {second}, who={who}: largest difference over largest strength {w:.1e}; the same lines chosen: {same}; lines learned from {sum(x['learned'] for x in oa['rows'])} of {sum(x['drawn'] for x in oa['rows'])} drawn")
        replay6.run(dict(base, who="R", fast="0")); a = replay6.run.last
        if second == "sub": replay.run(dict(seed="3", plan="add:600,sub:300", every="100", who="R", r="1", label="chk", nowrite="1", fast="0")); b = replay.run.last
        else: replay3.run(dict(seed="3", plan="add:600,swp:300", every="100", who="R", r="1", label="chk", nowrite="1", fast="0")); b = replay3.run.last
        w = diff(a, b); bad += not w < 1e-6; print(f"add then {second}, who=R against the earlier program's arm R (numpy, the same life): {w:.1e}")
    s = replay6.run.last; L = replay.text("+", 37, 58); tot = 0.0; hist = [END] * D
    for t in range(9):
        hist.insert(0, L[t]); hist.pop(); act = OFF + np.asarray(hist); h = sg(s.W1[:, act].sum(1) + s.b1); m = -(s.W2 @ h + s.b2); m[L[t + 1]] += 1; tot += float(m @ m)
    got = float(replay6.mismatch(s, [L])[0]); bad += abs(got - tot) > 1e-9 * max(1, tot); print(f"the mismatch used for choosing against the learning loop's own on one line read after a newline: {got:.6f} and {tot:.6f}")
    import fastm, json
    rng = np.random.default_rng(2); Ls = [replay.text("+-"[int(rng.integers(2))], int(rng.integers(100)), int(rng.integers(100))) for _ in range(2000)] + [replay3.swp(int(rng.integers(100)), int(rng.integers(100))) for _ in range(1000)]
    a1 = replay6.mismatch(s, Ls); a2 = fastm.judge(s, Ls, END); w = float(np.abs(a1 - a2).max() / np.abs(a1).max()); bad += not w < 1e-9; print(f"the compiled judging against the numpy function on 3,000 lines: largest difference over largest value {w:.1e}; the same line would be chosen among every four: {bool((a1.reshape(-1, 4).argmax(1) == a2.reshape(-1, 4).argmax(1)).all())}")
    o = replay6.run(dict(seed="3", plan="add:600,sub:300", every="100", chunk="50", who="M", fast="1", label="chk", nowrite="1")); json.dumps(o); print("a result of arm M can be written as JSON")
    sx = replay6.run.last; held = [replay.text("+", int(a_), int(b_)) for a_, b_ in rng.integers(100, size=(200, 2))]; e1 = replay3.score(sx, held); e2 = replay6.score_after(sx, held, np.full((7, 10), END), np.random.default_rng(0)); bad += abs(e1 - e2) > 1e-9
    print(f"the score after a line made of line ends only is the empty-past score: {e2:.2f} and {e1:.2f}")
    prev = replay.text("+", 12, 34); L = held[0]; hist = [END] * D
    for x in prev: hist.insert(0, x); hist.pop()
    for x in L[:8]: hist.insert(0, x); hist.pop()
    P = np.array([prev]); Lr = np.array([L]); h2 = [Lr[0, 7 - d] if 7 - d >= 0 else P[0, 10 + 7 - d] for d in range(D)]; bad += [int(v) for v in h2] != hist; print(f"the delays at the last answer byte of a line read after another line are those of the stream: {[int(v) for v in h2] == hist}")
    o = replay6.run(dict(seed="3", plan="add:600,sub:300", every="100", chunk="50", who="P", fast="0", label="chk", nowrite="1")); a = replay6.run.last; replay.run(dict(seed="3", plan="add:600,sub:300", every="100", who="P", label="chk", nowrite="1", fast="0")); w = diff(a, replay.run.last); bad += not w < 1e-6
    print(f"who=P against replay.py's plain arm (numpy, the same life): {w:.1e}")
    print("CHECK", "FAILED" if bad else "PASSED")
