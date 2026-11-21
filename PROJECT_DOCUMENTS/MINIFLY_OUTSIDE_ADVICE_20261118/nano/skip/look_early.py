#!/usr/bin/env python3
"""PLAN_3.md: where the early evidence for passing over a letter comes from. The system reads N lines of g2 under a given clock (compiled kernel);
then 3,000 fresh lines are read with nothing changing (numpy), and the answering cell's squared mismatch (summed over its branches) is averaged for
each kind of byte to be named.   python3 look_early.py [seed]"""
import json, os, sys
import numpy as np
import skip
from skip import SI, S, END, EQ, LETS, sg

KINDS = ["second digit of aa", "+", "first byte after +", "byte after a letter", "second digit of bb", "=", "c hundreds", "c tens", "c units", "end mark"]
def kinds(xs):
    """For each byte of a stream of g2 lines, the kind of the byte that follows it (the one to be named), or -1 at a line's end."""
    out = np.full(len(xs), -1); pos = 0; seen_plus = False; nb = 0; after_eq = -1
    for i, x in enumerate(xs):
        if x == END: pos = 0; seen_plus = False; nb = 0; after_eq = -1; continue
        if not seen_plus:
            if x == SI[ord("+")]: seen_plus = True; out[i] = 2
            else: out[i] = 0 if pos == 0 else 1
        elif after_eq >= 0: out[i] = 7 + after_eq; after_eq += 1
        elif x == EQ: out[i] = 6; after_eq = 0
        elif x in LETS: out[i] = 3
        else: nb += 1; out[i] = 4 if nb == 1 else 5
        pos += 1
    return out

def look(seed, clock, N):
    kw = dict(label="look", nowrite="1", text="g2", seed=str(seed), lines=str(N), every=str(N), fast="1", clock="none" if clock.startswith("only:") else clock)
    passed = np.zeros(S, bool)
    if clock == "oracle": passed[LETS] = True
    if clock.startswith("only:"):                      # one byte value passed over by the program (a measurement, not a candidate): run by hand through the given flags
        passed[SI[ord(clock[5:])]] = True
        rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; shown = P[2000:]
        s = skip.Sk(seed); s.fast = 1; xs, ys = skip.stream("g2", shown, 8, N, rng); s.run(xs, ys, ~passed[xs])
    else:
        skip.run(kw); s = skip.run.last; rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; shown = P[2000:]
    xs, ys = skip.stream("g2", shown, 8, 3000, np.random.default_rng([seed, 555])); kd = kinds(xs); tot = np.zeros(len(KINDS)); cnt = np.zeros(len(KINDS)); hist = list(s.hist); pend = bool(s.pend)
    for i in range(len(xs)):
        p = hist[0]; x = int(xs[i])
        if pend: hist = [x, p] + hist[1:-1]
        else: hist[0] = x
        pend = not passed[x]
        if ys[i] < 0: continue
        h, a = s.fwd(hist); m = -a; m[ys[i]] += 1; tot[kd[i]] += float(m @ m); cnt[kd[i]] += 1
    return (tot / 3000).tolist(), (tot / np.maximum(cnt, 1)).tolist()                # a line's mean total for each kind; the mean for a byte of the kind

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1; out = {}
    for N in (10000, 30000, 100000):
        for clock in ("none", "only:w", "oracle"):
            line, byte = look(seed, clock, N); out[f"{clock}_{N}"] = dict(by_line=line, by_byte=byte, total_by_line=float(sum(line)))
            print(f"{N:7d} lines  {clock:7s} a line's squared mismatch {sum(line):.3f} | " + "  ".join(f"{k}: {v:.3f}" for k, v in zip(KINDS, line)))
    out["kinds"] = KINDS; json.dump(out, open(os.path.join(skip.HERE, "res", f"p3_look_early_s{seed}.json"), "w"))
