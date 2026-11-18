"""E53: the brief's early report: loss, A, R, ring_magnitudes and gates at given rows of a run, printed from its receipt.
    python3 exp45/e53/e53_early.py <arm> <rule> <row> [<row> ...]"""
import json, sys
from pathlib import Path
arm, rule, rows = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
r = json.loads((Path(__file__).resolve().parent / f"e53_32_{arm}_{rule}_73129.json").read_text())
c = {x["step"]: x for x in r["curve"]}
for u in rows:
    if u not in c:
        print(arm, rule, u, "not yet in the receipt"); continue
    x = c[u]; g = x["gates"]
    print(f"{arm} {rule} {u}: loss {'-' if x['loss'] is None else format(x['loss'], '.3f')} A {x['A']:.1f} R {x['R']:.1f} finite {x['finite_losses']} | ring_magnitudes "
          + " ".join(f"{k} {v:.3f}" for k, v in x["ring_magnitudes"].items())
          + " | gates extreme r/z % " + " ".join(f"{k} {100 * g[k]['reset_share_extreme']:.1f}/{100 * g[k]['update_share_extreme']:.1f}" for k in ("interface", "ring1", "ring2", "ring3"))
          + " | mean|h| " + " ".join(f"{k} {g[k]['mean_abs_state']:.3f}" for k in ("interface", "ring1", "ring2", "ring3")))
