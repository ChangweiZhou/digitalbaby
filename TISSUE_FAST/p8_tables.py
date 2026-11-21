"""Tables of plan 8: S(d,0) and the share of active cells at every distance in window A2, activity by stretch, cost."""
import json
import numpy as np
from pathlib import Path
from depth_read import depth
RUNS = Path(__file__).resolve().parent / "TISSUE_EVENT" / "runs"
LEVEL = 0.1103
out = []
for form in ("first", "coincidence"):
    out += [f"\n### {form}\n", "| Arm | S(d,0) for d = 1 ... 11 | Active share at reading steps, d = 1 ... 11 | Farthest d at level | Connections at the end | Active at stretch ends, min / max | ms a byte | H8 met |", "|---|---|---|---|---|---|---|---|"]
    for arm, name in [("F32 (plan 6)", f"P6_{form}_F32"), ("B075 (plan 7)", f"P7_{form}_B075"), ("B10", f"P8_{form}_B10"), ("B125", f"P8_{form}_B125"), ("B075i", f"P8_{form}_B075i"), ("B10i", f"P8_{form}_B10i"), ("B15 (plan 7)", f"P7_{form}_B15")]:
        f = RUNS / name / "receipt.json"
        if not f.exists():
            out.append(f"| {arm} | not run |"); continue
        r = json.loads(f.read_text())
        dp = depth(name)
        z = np.load(RUNS / name / "A2.npz"); X, dist = z["x"], z["distance"]
        shares = " ".join(f"{float((X[:, dist == d] > 0).mean()):.2f}" if (dist == d).any() else "–" for d in range(1, 12))
        far = max([d for d, (s, c) in dp.items() if s is not None and s >= LEVEL], default=0)
        cs = {c["bytes"][1] + 1: c["mean_connections"] for c in r["cpu_by_stretch"]}
        act = [c["active_end"] for c in r["cpu_by_stretch"]]
        ok = far >= 8 and dp[1][0] >= LEVEL and max(act) <= 1536
        out.append(f"| {arm} | {' '.join('–' if s is None else f'{s:.3f}' for d, (s, c) in dp.items())} | {shares} | {far} | {cs[16384]:.1f} | {min(act)} / {max(act)} | {1000 * r['cpu_per_byte_core_only']:.2f} | {'yes' if ok else 'no'} |")
print("\n".join(out))
