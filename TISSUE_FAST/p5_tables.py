"""Tables of plan 5: S(d,0) at every distance (depth_read.py), the farthest distance at level, activity by stretch, cost."""
import json
from pathlib import Path
from depth_read import depth
RUNS = Path(__file__).resolve().parent / "TISSUE_EVENT" / "runs"
LEVEL = 0.1103
out = []
for form in ("first", "coincidence"):
    out += [f"\n### {form}\n", "| Arm | S(d,0) in A2 for d = 1 ... 11 | Farthest d at level | Connections at the end | Last 4,096 bytes | Active cells at stretch ends, min / max | ms a byte core | H5 met |", "|---|---|---|---|---|---|---|---|"]
    base_far = None
    for arm, name in [("N75T (plan 3)", f"P3_{form}_N75T"), ("F", f"P5_{form}_F"), ("G", f"P5_{form}_G"), ("L", f"P5_{form}_L"), ("FL", f"P5_{form}_FL")]:
        f = RUNS / name / "receipt.json"
        if not f.exists():
            out.append(f"| {arm} | not run | | | | | | |"); continue
        r = json.loads(f.read_text())
        dp = depth(name)
        far = max([d for d, (s, c) in dp.items() if s is not None and s >= LEVEL], default=0)
        if base_far is None: base_far = far
        cs = {c["bytes"][1] + 1: c["mean_connections"] for c in r["cpu_by_stretch"]}
        act = [c.get("active_end") for c in r["cpu_by_stretch"] if c.get("active_end") is not None]
        s1 = dp[1][0]
        ok = (arm != "N75T (plan 3)") and far > base_far and act and max(act) < 1024 and s1 >= LEVEL
        out.append(f"| {arm} | {' '.join('–' if s is None else f'{s:.3f}' for d, (s, c) in dp.items())} | {far} | {cs[16384]:.1f} | {100 * (cs[16384] - cs[12288]) / cs[12288]:+.1f}% | "
                   f"{(str(min(act)) + ' / ' + str(max(act))) if act else 'not recorded'} | {1000 * r['cpu_per_byte_core_only']:.2f} | {'yes' if ok else ('–' if arm.startswith('N75T') else 'no')} |")
print("\n".join(out))
