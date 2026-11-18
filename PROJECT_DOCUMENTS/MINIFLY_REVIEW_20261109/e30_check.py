"""Independent check of E30's U and H on E28's twelve births (numpy forward; no training), completing the seeds E30 could not read."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from np_gru import load_weights, comp_panels, forward, panel, split, DEV, BOS, exact_teacher_forced
REF = 'origin/claude/charming-cori-ljzwz1'; P28 = 'MOD97_LOCAL_RULE_20261106/exp45/e28/'
SEEDS = (73129, 93683, 116089)
def U(W, items):
    x = np.array([[BOS] + p + t[:-1] for p, t in items], dtype=np.int64)
    Z = forward(W, x, want_logits=False)[0][:, 1:, :]
    return 100 * float(((Z >= 0.1) & (Z <= 0.9)).mean())
def acc(W, items):
    x = np.array([[BOS] + p + t[:-1] for p, t in items], dtype=np.int64); tgt = np.array([t for _, t in items], dtype=np.int64)
    return 100 * float(exact_teacher_forced(forward(W, x)[3], tgt).mean())
print('arm seed | U plus-minus | acc add89 at 12 / at 40 letters | H')
for arm in ('S-all', 'B-all', 'S-bal', 'B-bal'):
    for s in SEEDS:
        W, _ = load_weights(P28 + f'e28_dev_{arm}_{s}.pt', ref=REF)
        comp, _ = comp_panels(s); pm = comp['+-'] + comp['-+']
        pk = panel(DEV[0], split(DEV[0], .5, s)[1], s, (12, 40), 300)
        a12, a40 = acc(W, pk[12]), acc(W, pk[40])
        print(f'{arm} {s} | {U(W, pm):6.2f} | {a12:5.1f} / {a40:5.1f} | {a12 - a40:6.2f}')
