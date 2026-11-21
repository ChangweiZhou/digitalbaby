"""Tables of plan 6 (and plan 5's F beside them)."""
import json
from pathlib import Path
from depth_read import depth
RUNS = Path(__file__).resolve().parent / "TISSUE_EVENT" / "runs"
LEVEL = 0.1103
out = []
for form in ("first", "coincidence"):
    out += [f"\n### {form}\n", "| Arm | S(d,0) in A2 for d = 1 ... 11 | Farthest d at level | Connections a cell by 4,096 bytes | Change over the last 4,096 (Flong: 8,192) bytes | Fewest connections into a non-feedback cell at the end | Active at stretch ends, min / max | ms a byte | H6 met |", "|---|---|---|---|---|---|---|---|---|"]
    arms = [("F (plan 5)", f"P5_{form}_F"), ("F16", f"P6_{form}_F16"), ("F32", f"P6_{form}_F32"), ("F8t2", f"P6_{form}_F8t2")] + ([("Flong", f"P6_{form}_Flong")] if form == "coincidence" else [])
    for arm, name in arms:
        f = RUNS / name / "receipt.json"
        if not f.exists():
            out.append(f"| {arm} | not run |"); continue
        r = json.loads(f.read_text())
        dp = depth(name)
        far = max([d for d, (s, c) in dp.items() if s is not None and s >= LEVEL], default=0)
        cs = {c["bytes"][1] + 1: c["mean_connections"] for c in r["cpu_by_stretch"]}
        nb = r["stream"]["length"]; back = 8192 if arm == "Flong" else 4096
        ch = (cs[nb] - cs[nb - back]) / cs[nb - back]
        act = [c.get("active_end") for c in r["cpu_by_stretch"] if c.get("active_end") is not None]
        mn = r["counts"][-1]["per_cell_min_nonfb"] if r["counts"] else None
        ok = form == "coincidence" and far >= 8 and max(act) <= 1024 and dp[1][0] >= LEVEL
        traj = " / ".join(f"{cs[b]:.1f}" for b in sorted(cs) if b % 4096 == 0)
        out.append(f"| {arm} | {' '.join('–' if s is None else f'{s:.3f}' for d, (s, c) in dp.items())} | {far} | {traj} | {100 * ch:+.1f}% | {mn} | {min(act)} / {max(act)} | {1000 * r['cpu_per_byte_core_only']:.2f} | {'yes' if ok else 'no'} |")
print("\n".join(out))
