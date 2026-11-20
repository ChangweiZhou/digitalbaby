#!/usr/bin/env python3
"""PLAN_12.md: the control `reserve=128` of cells2.py, with the old line that is given one time in `reh` during the second phase drawn from a store of N
addition lines that the system kept: the last N lines read before the change of task (N*10 bytes). N=0: fresh lines, which must give cells2.py's figures.
    python3 store.py <label> seed=.. N=.. [reh=32] [first=400000] [second=200000]
The loop is cells2.run's for this one case (same generator, same order of draws), so that the first phase is the same stream."""
import json, os, sys, time
import numpy as np
import cells2
from cells2 import Sys, text
kw = dict(a.split("=") for a in sys.argv[2:]); label = sys.argv[1]; seed = int(kw["seed"]); N = int(kw.get("N", 0)); reh = int(kw.get("reh", 32)); n0 = int(kw.get("first", 400000)); n1 = int(kw.get("second", 200000))
t0 = time.process_time(); rng = np.random.default_rng([seed, 99]); held, shown = {}, {}; every = 50000; nres = 128
for op in "+-":
    o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = P[:500]; shown[op] = P[2000:]
s = Sys(seed, H=256 + nres); s.live = np.arange(s.H) < s.H - nres; s.free = s.live.astype(float); s.fast = 1
TAB = {op: np.array([text(op, a, b) for a, b in shown[op]], np.int32) for op in "+-"}; TL = {op: TAB[op].tolist() for op in "+-"}; rows = []; last = None
key = lambda p: (max(p), min(p)); S = {key(p) for p in shown["-"]}; clean = [p for p in held["-"] if key(p) not in S]
def read(ph, lines): rows.append(dict(phase=ph, lines=lines, add=s.score("+", held["+"]), sub=s.score("-", held["-"]), sub_clean=s.score("-", clean)))
for it in range(0, n0, every):
    idx = rng.integers(8000, size=every); X = TAB["+"][idx]; s.bx = X.ravel(); s.by = np.concatenate([X[:, 1:], np.full((len(X), 1), -1, np.int32)], 1).ravel(); s.flush(); read(0, it + every); last = idx
kept = [int(i) for i in last[-N:]] if N else []                      # the store: the last N lines read (indices into the shown addition pairs)
s.free = (~(np.arange(s.H) < s.H - nres)).astype(float); s.live = np.ones(s.H, bool); s.late = True
for it in range(n1):
    old = reh and rng.integers(reh) == 0
    if old and N: L = TL["+"][kept[rng.integers(N)]]
    else: L = TL["+" if old else "-"][rng.integers(8000)]
    s.line(L)
    if (it + 1) % every == 0: read(1, it + 1)
stored = [shown["+"][i] for i in sorted(set(kept))]
out = dict(label=label, args=kw, rows=rows, store_lines=N, store_bytes=10 * N, distinct_stored_sums=len(stored), add_on_stored_sums=(s.score("+", stored) if stored else None),
           add_on_500_shown_sums=s.score("+", shown["+"][:500]), n_clean=len(clean), cpu=round(time.process_time() - t0, 1))
json.dump(out, open(os.path.join(cells2.HERE, "res", f"c2_{label}.json"), "w")); e = rows[-1]
print(label, "end: addition never shown", e["add"], "| on the stored sums", out["add_on_stored_sums"], "| on 500 shown sums", out["add_on_500_shown_sums"], "| subtraction", e["sub"], "clean", round(e["sub_clean"], 1), out["cpu"], "s")
