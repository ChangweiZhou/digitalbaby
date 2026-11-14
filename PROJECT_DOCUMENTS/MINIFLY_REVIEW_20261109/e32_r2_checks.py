"""Checks for the audit of revision 2 of the E32 brief. Post hoc, unregistered; numpy forward on saved weights; nothing is trained.

A. Keeping units by the brief's definition (product of the update gate over the k letter positions >= 0.5 in at least half the items),
   beside the median form used in retention.py, at 12, 20 and 40 letters; and how many of them also write before the gap
   (z <= 0.5 at some position before the first letter, in at least half the items).
B. The floor at each saved row: U (E30), keeping units, and at the letters the shares of gate values above 0.9, 0.944 and 0.983.
C. The state the push meets at update 4,096: which side of one half each unit's gate is on, by position kind.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from np_gru import load_weights, forward, panel, split, DEV, BOS, exact_teacher_forced, roles_of
REF = 'origin/codex/minifly-consolidated-20261023'; P = 'MOD97_LOCAL_RULE_20261106/exp45/'
SEEDS = (73129, 93683, 116089)
FINALS = [('S-all', 'e28/e28_dev_S-all_{s}.pt'), ('S-bal', 'e28/e28_dev_S-bal_{s}.pt'), ('B-all', 'e28/e28_dev_B-all_{s}.pt'),
          ('B-bal', 'e28/e28_dev_B-bal_{s}.pt'), ('S-ans', 'e26/e26_dev_S_{s}.pt'), ('B-ans', 'e26/e26_dev_B_{s}.pt')]
ROWS = (4096, 8192, 12288, 16384, 20480, 24576)

def panels(s):
    pairs = split(DEV[0], .5, s)[1]
    pk = panel(DEV[0], pairs, s, (12, 40), 300)            # E30's call: the 40-letter letters depend on ks
    pk[20] = panel(DEV[0], pairs, s, (20,), 300)[20]       # same pairs, letters drawn by a call of its own
    return pk

def run(W, items):
    x = np.array([[BOS] + p + t[:-1] for p, t in items], dtype=np.int64); tgt = np.array([t for _, t in items], dtype=np.int64)
    Z, _, _, lg = forward(W, x)
    roles = np.array([roles_of(r) for r in x])
    return x, Z, roles, 100 * float(exact_teacher_forced(lg, tgt).mean())

def keeping(Z, roles):
    letter = roles == 'letter'
    keep = np.where(letter[..., None], Z.astype(np.float64), 1.0).prod(1)             # (items, units)
    brief = (keep >= .5).sum(0) >= keep.shape[0] / 2                                  # the brief's words
    med = np.median(keep, 0) >= .5                                                    # retention.py
    first = letter.argmax(1)                                                          # index of the first letter (same in every item)
    assert (first == first[0]).all()
    pre = Z[:, 1:first[0], :]                                                         # inputs: task code, a1, a2, operator
    writes = (pre.min(1) <= .5).sum(0) >= Z.shape[0] / 2
    return brief, med, writes

if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'ABC'
    if 'A' in part:
        print('A. final files | accuracy at 12 / 20 / 40 | keeping units, brief (median form) at 12, 20, 40 | of the brief\'s keepers at 12: also write before the gap')
        for arm, pat in FINALS:
            for s in SEEDS:
                W, _ = load_weights(P + pat.format(s=s), ref=REF); pk = panels(s); acc, kb, km = {}, {}, {}
                for k in (12, 20, 40):
                    _, Z, roles, acc[k] = run(W, pk[k]); b, m_, w = keeping(Z, roles); kb[k], km[k] = int(b.sum()), int(m_.sum())
                    if k == 12: kw = int((b & w).sum())
                print(f"  {arm:6s} {s:6d} | {acc[12]:5.1f} {acc[20]:5.1f} {acc[40]:5.1f} | {kb[12]:3d} ({km[12]:3d})  {kb[20]:3d} ({km[20]:3d})  {kb[40]:3d} ({km[40]:3d}) | {kw:3d}")
    if 'B' in part:
        print('B. the floor by row | U (E30, 12 letters) | accuracy 12 / 20 / 40 | keeping units 12 / 20 / 40 | letter gates (12-letter panel): % above 0.9, 0.944, 0.983')
        for s in SEEDS:
            pk = panels(s)
            for row in ROWS:
                f = f'e28/e28_dev_S-all_{s}.pt' if row == 24576 else f'e28/e28_dev_S-all_{s}_row{row}.pt'
                W, _ = load_weights(P + f, ref=REF); acc, kb = {}, {}
                for k in (12, 20, 40):
                    _, Z, roles, acc[k] = run(W, pk[k]); kb[k] = int(keeping(Z, roles)[0].sum())
                    if k == 12:
                        U = 100 * float(((Z[:, 1:] >= .1) & (Z[:, 1:] <= .9)).mean()); zl = Z[roles == 'letter']
                        sh = [100 * float((zl > t).mean()) for t in (.9, .944, .983)]
                print(f"  S-all {s:6d} row {row:5d} | {U:6.2f} | {acc[12]:5.1f} {acc[20]:5.1f} {acc[40]:5.1f} | {kb[12]:3d} {kb[20]:3d} {kb[40]:3d} | {sh[0]:5.2f} {sh[1]:5.2f} {sh[2]:5.2f}")
    if 'C' in part:
        print('C. the floor at update 4,096 (where the push is switched on), 12-letter panel. A unit is "up" on a set of positions if z > 0.5 on at least 90% of them, "down" if on at most 10%.')
        print('   seed | gates inside 0.01-0.99 | at letters: up / down / split | by letter identity (mean z per letter): up on all 26 / down on all 26 / mixed | up at letters AND up before the gap (never writes) | in at least half the items every letter gate is above 0.5: at 12 / 40 letters, and of those at 12 that also write')
        for s in SEEDS:
            W, _ = load_weights(P + f'e28/e28_dev_S-all_{s}_row4096.pt', ref=REF); pk = panels(s)
            x, Z, roles, _ = run(W, pk[12]); letter = roles == 'letter'
            band = 100 * float(((Z[:, 1:] >= .01) & (Z[:, 1:] <= .99)).mean())
            up = (Z[letter] > .5).mean(0); n_up, n_dn = int((up >= .9).sum()), int((up <= .1).sum())
            byl = np.stack([Z[letter & (x == c)].mean(0) for c in range(65, 91)])                 # (26, units)
            a26, d26 = int((byl > .5).all(0).sum()), int((byl < .5).all(0).sum())
            first = letter.argmax(1)[0]; pre_up = (Z[:, 1:first, :] > .5).mean((0, 1))
            dead = int(((up >= .9) & (pre_up >= .9)).sum())
            allup12 = (np.where(letter[..., None], Z, 1.0) > .5).all(1)                           # (items, units)
            c12 = allup12.sum(0) >= 150; w = keeping(Z, roles)[2]
            _, Z40, r40, _ = run(W, pk[40]); c40 = (np.where((r40 == 'letter')[..., None], Z40, 1.0) > .5).all(1).sum(0) >= 150
            print(f"   {s} | {band:6.2f}% | {n_up} / {n_dn} / {256 - n_up - n_dn} | {a26} / {d26} / {256 - a26 - d26} | {dead} | {int(c12.sum())} / {int(c40.sum())}, {int((c12 & w).sum())}")
