#!/usr/bin/env python3
"""A look inside (no plan of its own; belongs to PLAN_3.md and PLAN_7.md): where on the last tenth the loss of T2 lies, by the kind of byte to be named.
    python3 where.py"""
import json, os
import numpy as np
import score, words, trust
HERE = os.path.dirname(os.path.abspath(__file__))
x, cut, _ = score.load(); s = trust.Sys(x, words.word_keys(x), 6); s.go(6, cut, 0.00005, 0.02, True); P = s.go(cut, len(x), 0.00005, 0.02, False, True)
y = x[cut:]; prev = x[cut - 1:-1]; nll = -np.log(P[np.arange(len(y)), y]); right = P.argmax(1) == y; let = y >= 13; plet = prev >= 13
pos = np.zeros(len(y), int)                      # place of a letter in its word: 1, 2, 3, ...
for i in range(len(y)): pos[i] = (pos[i - 1] + 1 if i and let[i - 1] else 1) if let[i] else 0
since_nl = np.zeros(len(y), int); c = 0
for i in range(len(y)): since_nl[i] = c; c = 0 if y[i] == 0 else c + 1
groups = [("a line end", y == 0), ("a space", y == 1), ("another byte that is not a letter", (~let) & (y > 1)), ("the first letter of a word", pos == 1), ("the second letter", pos == 2), ("the third letter", pos == 3), ("a later letter", pos >= 4)]
out = {}; print(f"{'the byte to be named is':36s} share of bytes  mean nats  share of the loss  named right")
for name, m in groups:
    out[name] = dict(share=float(m.mean()), nats=float(nll[m].mean()), loss_share=float(nll[m].sum() / nll.sum()), right=float(right[m].mean())); print(f"{name:36s} {100*m.mean():10.1f}%  {nll[m].mean():9.3f}  {100*nll[m].sum()/nll.sum():14.1f}%  {100*right[m].mean():9.1f}%")
m = (y <= 1) & plet; ends = m; print("at the end of a word (a letter before, a space or line end to be named): line end named as line end", round(100 * float((P[m & (y == 0)].argmax(1) == 0).mean()), 1), "% ; mean nats when it is a line end", round(float(nll[m & (y == 0)].mean()), 3), "; when it is a space", round(float(nll[m & (y == 1)].mean()), 3))
print("total", round(float(nll.mean()), 4)); out["total"] = float(nll.mean()); json.dump(out, open(os.path.join(HERE, "res", "where_T2.json"), "w"))
