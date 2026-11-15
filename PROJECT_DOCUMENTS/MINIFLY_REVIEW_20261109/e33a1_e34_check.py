"""Independent recomputation of the E33 addendum 1 and E34 stage 1 tables (15 November 2026).

Read-only: forward passes on the worker's committed weights with the numpy reader; nothing trains.  For each birth it
prints exact answers on the mod-89 addition panel at 12, 20 and 40 letters for the final file and rows 16,384 and
20,480, and their mean (the briefs' endpoint).  Earlier rows are not committed and cannot be recomputed.

Run from the repository root:
  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e33a1_e34_check.py [e34|e33]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import load_weights
from e32_r2_checks import panels, run

W_REF = 'origin/claude/vibrant-curie-ivmrt1'; P = 'MOD97_LOCAL_RULE_20261106/exp45/'
E34 = [(f'e34/e34_dev_{arm}_73129', 73129) for arm in ('T050', 'T100', 'T200')]
E33 = [(f'e33/e33_dev_{arm}_{s}', s) for s in (93683, 116089) for arm in ('A+D', 'A+OS')]
which = sys.argv[1] if len(sys.argv) > 1 else 'both'
for stem, seed in (E34 if which == 'e34' else E33 if which == 'e33' else E34 + E33):
    pk = panels(seed); rows = {}
    for tag, suffix in (('16384', '_row16384.pt'), ('20480', '_row20480.pt'), ('final', '.pt')):
        W, _ = load_weights(P + stem + suffix, ref=W_REF)
        rows[tag] = [run(W, pk[k])[3] for k in (12, 20, 40)]
    mean = np.mean(list(rows.values()), 0)
    print(f"{stem}: endpoint 12/20/40 = {mean[0]:.2f} / {mean[1]:.2f} / {mean[2]:.2f} | " + " | ".join(f"{t} {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}" for t, v in rows.items()), flush=True)
