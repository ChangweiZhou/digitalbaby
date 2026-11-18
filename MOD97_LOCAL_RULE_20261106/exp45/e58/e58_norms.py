"""E58: norm of the tensors outside the byte table (all together) relative to the lived network, and the norm of the change relative to the
lived norm, at rows 64, 1024 and 8192, for the three free arms of Part A. Printed from the receipts.  python3 exp45/e58/e58_norms.py"""
import json
from pathlib import Path
D = Path(__file__).resolve().parent
for rule in "SB":
    for arm in ("free", "free-wd0", "free-anchor"):
        for s in (73129, 93683, 116089):
            r = json.loads((D / f"e58_{arm}_{rule}_f10_{s}.json").read_text())
            rw = {x["updates"]: x for x in r["rows"]}
            tot = lambda u, k: sum(v[k] ** 2 for v in rw[u]["norms"].values()) ** .5
            n0 = tot(0, "norm")
            print(rule, f"{arm:12s}", s, " ".join(f"row {u}: norm {tot(u, 'norm') / n0:.3f} change {tot(u, 'diff_from_lived') / n0:.3f}" for u in (64, 1024, 8192)))
