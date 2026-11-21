"""One table for any runs: S(d,0) at d = 1 ... 11 (depth_read.py), the farthest d at level, connections, active cells, cost.
Usage: python3 tab.py <run name or glob> ..."""
import sys, json, glob
from pathlib import Path
from depth_read import depth
RUNS = Path(__file__).resolve().parent / "TISSUE_EVENT" / "runs"
LEVEL = 0.1103
names = []
for a in sys.argv[1:]:
    names += sorted(p.parent.name for p in RUNS.glob(a + "/receipt.json")) or [a]
print("| Run | S(d,0), d = 1 ... 11 | Farthest d at level | Connections at the end | Active at stretch ends, min / max | ms a byte (with records) |")
print("|---|---|---|---|---|---|")
for n in names:
    f = RUNS / n / "receipt.json"
    if not f.exists():
        print(f"| {n} | not run | | | | |"); continue
    r = json.loads(f.read_text()); dp = depth(n)
    far = max([d for d, (s, c) in dp.items() if s is not None and s >= LEVEL], default=0)
    cs = {c["bytes"][1] + 1: c["mean_connections"] for c in r["cpu_by_stretch"]}
    act = [c.get("active_end") for c in r["cpu_by_stretch"] if c.get("active_end") is not None]
    print(f"| {n} | {' '.join('–' if s is None else f'{s:.3f}' for d, (s, c) in dp.items())} | {far} | {cs[max(cs)]:.1f} | "
          f"{(str(min(act)) + ' / ' + str(max(act))) if act else '–'} | {1000 * r['cpu_per_byte_with_records']:.2f} |")
