"""Post hoc, unregistered. On saved births (numpy forward, no training): what the update gate does across the letters between the operands.
keep(unit, item) = product of z over the letter positions = the share of the unit's state that survives the gap (h' = (1-z) n + z h).
U_letters = % of z in [0.1, 0.9] at letter positions; edge = % in (0.9, 0.98): outside E30's band yet too leaky to hold 40 steps (0.9**40 = 0.015)."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from np_gru import load_weights, forward, panel, split, DEV, BOS, exact_teacher_forced, roles_of
REF = 'origin/codex/minifly-consolidated-20261023'; P = 'MOD97_LOCAL_RULE_20261106/exp45/'
SEEDS = (73129, 93683, 116089)
ARMS = [('S-all', 'e28/e28_dev_S-all_{s}.pt'), ('S-bal', 'e28/e28_dev_S-bal_{s}.pt'), ('B-all', 'e28/e28_dev_B-all_{s}.pt'),
        ('B-bal', 'e28/e28_dev_B-bal_{s}.pt'), ('S-ans', 'e26/e26_dev_S_{s}.pt'), ('B-ans', 'e26/e26_dev_B_{s}.pt')]
print('arm    seed   | acc12  acc40 | U_add89(E30 def, k12) | at 40 letters: U_letters  edge(.9-.98)  z>=.98 | units keeping >= half their state across 12 / 40 letters (median item)')
for arm, pat in ARMS:
    for s in SEEDS:
        W, _ = load_weights(P + pat.format(s=s), ref=REF)
        pk = panel(DEV[0], split(DEV[0], .5, s)[1], s, (12, 40), 300)
        out = {}
        for k in (12, 40):
            items = pk[k]
            x = np.array([[BOS] + p + t[:-1] for p, t in items], dtype=np.int64); tgt = np.array([t for _, t in items], dtype=np.int64)
            Z, _, _, lg = forward(W, x)
            letter = np.array([[r == 'letter' for r in roles_of(row)] for row in x])
            assert (letter.sum(1) == k).all()
            zl = Z[letter]                                              # (N*k, units)
            keep = np.where(letter[..., None], Z.astype(np.float64), 1.0).prod(1)   # (N, units)
            out[k] = dict(acc=100 * float(exact_teacher_forced(lg, tgt).mean()), U=100 * float(((Z[:, 1:] >= .1) & (Z[:, 1:] <= .9)).mean()),
                          Ul=100 * float(((zl >= .1) & (zl <= .9)).mean()), edge=100 * float(((zl > .9) & (zl < .98)).mean()),
                          hi=100 * float((zl >= .98).mean()), keepers=int((np.median(keep, 0) >= .5).sum()))
        a, b = out[12], out[40]
        print(f"{arm:6s} {s:6d} | {a['acc']:5.1f}  {b['acc']:5.1f} | {a['U']:6.2f} | {b['Ul']:6.2f}  {b['edge']:6.2f}  {b['hi']:6.2f} | {a['keepers']:3d} / {b['keepers']:3d}")
