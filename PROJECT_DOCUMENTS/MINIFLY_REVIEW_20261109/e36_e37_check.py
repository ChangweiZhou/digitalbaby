"""Independent recomputation of headline figures of E36 and E37 from the workers' committed weights (15 November 2026).

Read-only: forward passes with the numpy reader; nothing trains.
  e36  the wrap score at row 1,024 (mean of add and sub on the pairs that need the wrap at 89, whole answers, no
       letters) for the nine reminders and six fresh references
  e37  the fresh networks' held-out score (mean of add and sub at 12 letters, 400 items, the life receipts' panels)
       at the committed rows: family F3 at row 4,096 and family F1 at row 6,144

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e36_e37_check.py [e36|e37]
"""
import os, sys
import numpy as np
which = sys.argv[1] if len(sys.argv) > 1 else "both"
sys.argv = [sys.argv[0], "Z"]                       # keep e36_e38_brief_checks from running its own parts on import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import load_weights, panel, split, forward, exact_teacher_forced, DEV, LIFE, BOS, P
from e36_e38_brief_checks import wrap_split, F3

SEEDS = (73129, 93683, 116089)
if which in ("e36", "both"):
    ref = "origin/claude/vibrant-allen-8hziln"
    for kind, arms in (("reminder", "SBK"), ("reference", "SB")):
        for arm in arms:
            out = []
            for s in SEEDS:
                W = load_weights(P + f"e36/e36_{kind}_{arm}_{s}_row1024.pt", ref)[0]
                out.append((wrap_split(W, DEV[0])[1] + wrap_split(W, DEV[1])[1]) / 2)
            print(f"E36 {kind} {arm}: wrap score at row 1,024 = " + ", ".join(f"{v:.1f}" for v in out), flush=True)
if which in ("e37", "both"):
    ref = "origin/claude/magical-pascal-sflem4"
    def held12(W, tasks, s):
        acc = []
        for t in tasks[:2]:                                                       # add and sub
            items = panel(t, split(t, .25, s)[1], s, (0, 8, 12, 20, 40), 400)[12]
            x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64); tgt = np.array([a for _, a in items], dtype=np.int64)
            acc.append(100 * float(exact_teacher_forced(forward(W, x)[3], tgt).mean()))
        return sum(acc) / 2
    for fam, tasks, row in (("F3", F3, 4096), ("F1", LIFE, 6144)):
        for arm in "SB":
            out = [held12(load_weights(P + f"e37/e37_{fam}_{arm}_{s}_row{row}.pt", ref)[0], tasks, s) for s in SEEDS]
            print(f"E37 fresh {fam} {arm} at row {row}: " + ", ".join(f"{v:.1f}" for v in out), flush=True)
