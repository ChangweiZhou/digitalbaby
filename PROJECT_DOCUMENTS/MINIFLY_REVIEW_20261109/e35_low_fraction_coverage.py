"""Operand coverage of the taught pairs at low taught fractions (E35, proposed fractions 0.03 and 0.05).

For each fraction, seed and task: how many of the 89 operand values never appear in the first or
the second operand slot among the taught pairs, and how many are missing from a slot in all four tasks.
Run: python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e35_low_fraction_coverage.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import split, DEV

N = 89
for frac in (0.03, 0.05, 0.10):
    for seed in (73129, 93683, 116089):
        taught = {t[0]: split(t, frac, seed)[0] for t in DEV}
        per = [(name, len(tp), N - len({p[0] for p in tp}), N - len({p[1] for p in tp})) for name, tp in taught.items()]
        never = [sum(all(v not in {p[s] for p in tp} for tp in taught.values()) for v in range(N)) for s in (0, 1)]
        print(f"f={frac} seed={seed}: (task, taught pairs, values never first, never second) {per}; missing in every task: first {never[0]}, second {never[1]}")
