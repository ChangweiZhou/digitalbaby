#!/usr/bin/env python3
"""What the cells in between have formed (text d2, PLAN_2.md): read from saved strengths, nothing is run.

    python3 look_inside.py res/<label>.npz [...]

At the byte "=" the first number's tens digit is 5 bytes back, its units digit 4, the second number's tens digit 2 and its units
digit 1; one byte later each is one further back, and so on. For every cell in between and each of these delays the ten strengths
from the digit lines are set against the digits 0 to 9: their correlation r and their slope. A cell is called ordered at a delay
when |r| is at least 0.9. Laid down by the program: nothing; a strength at birth is drawn at random, for which |r| >= 0.9 occurs
in about 1 cell in 2,700 at one delay (the script counts it at birth as well, from the same generator as the runner).
"""
import sys

import numpy as np

S = 13
dig = np.arange(10.0); dc = dig - dig.mean()


def order(W1, k):
    w = W1[:, k * S: k * S + 10].astype(np.float64); wc = w - w.mean(1, keepdims=True)
    return (wc @ dc) / (np.linalg.norm(wc, axis=1) * np.linalg.norm(dc) + 1e-30), (wc @ dc) / (dc @ dc)


def look(W1, name):
    H = W1.shape[0]
    print(f"{name}: {H} cells in between")
    for p, what in ((0, "at \"=\", before the hundreds digit is named"), (1, "one byte later, before the tens digit"), (2, "two bytes later, before the units digit")):
        at, au, bt, bu = 5 + p, 4 + p, 2 + p, 1 + p
        r = {k: order(W1, k) for k in (at, au, bt, bu)}
        o = {k: np.abs(r[k][0]) >= 0.9 for k in r}
        tens = o[at] & o[bt] & (np.sign(r[at][0]) == np.sign(r[bt][0])); units = o[au] & o[bu] & (np.sign(r[au][0]) == np.sign(r[bu][0]))
        allfour = tens & units & (np.sign(r[at][0]) == np.sign(r[au][0]))
        line = (f"   {what}: cells ordered at one of the four delays {int((o[at] | o[au] | o[bt] | o[bu]).sum())}; at both tens digits with one sign {int(tens.sum())}; "
                f"at both units digits with one sign {int(units.sum())}; at all four with one sign {int(allfour.sum())}")
        if tens.any():
            q = np.sort(r[bt][1][tens] / r[at][1][tens]); line += f"; slope of the second tens digit over the first, median {np.median(q):.2f} (from {q[0]:.2f} to {q[-1]:.2f})"
        if allfour.any():
            q = np.sort(r[at][1][allfour] / r[au][1][allfour]); line += f"; slope of a tens digit over a units digit in the cells ordered at all four, median {np.median(q):.1f} (from {q[0]:.1f} to {q[-1]:.1f})"
        print(line)


for path in sys.argv[1:]:
    z = np.load(path)
    look(z["W1"], path)
    wr = np.random.default_rng(0)
    look(wr.uniform(-0.5, 0.5, z["W1"].shape), "   for comparison, strengths drawn at random as at birth")
