"""Tables of plan 1 from the receipts (no reading of its own beyond E63's step1_read output stored in each receipt)."""
import json, sys
from pathlib import Path
RUNS = Path(__file__).resolve().parent / "TISSUE_EVENT" / "runs"
LEVEL = 0.1103
arms = ["R", "T128", "W", "W32", "W128"]
out = []
for form in ("first", "coincidence"):
    out.append(f"\n### {form}\n")
    out.append("| Arm | Connections a cell at bytes 1,024 / 4,096 / 8,192 / 12,288 / 16,384 | Change over the last 4,096 bytes | Made | Given up by strength | Withdrawn by use | ms a byte, core / with records | Emission share | S(1,0), S(2,0), S(4,0) in A2 | Informative share d=1/2/4 in A2 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        f = RUNS / f"P1_{form}_{arm}" / "receipt.json"
        if not f.exists():
            out.append(f"| {arm} | not run | | | | | | | | |"); continue
        r = json.loads(f.read_text())
        cs = {c["bytes"][1] + 1: c["mean_connections"] for c in r["cpu_by_stretch"]}
        pts = [cs.get(b) for b in (1024, 4096, 8192, 12288, 16384)]
        ch = (pts[-1] - pts[-2]) / pts[-2] if pts[-2] else float("nan")
        m = r.get("measures", {}).get("A2")
        if m:
            sc = [m["scores"][str(d)]["by_lag"]["0"]["S"] if m["scores"].get(str(d)) and m["scores"][str(d)]["by_lag"].get("0") else None for d in (1, 2, 4)]
            sh = [m["shares"][str(d)]["share"] if m["shares"].get(str(d)) else None for d in (1, 2, 4)]
        else:
            sc = sh = [None] * 3
        fmt = lambda v, p=3: "–" if v is None else f"{v:.{p}f}"
        gone_strength = r["given_up_total"] - r["withdrawn_by_use_total"]
        out.append(f"| {arm} | {' / '.join(fmt(p, 1) for p in pts)} | {100 * ch:+.1f}% | {r['made_total']:,} | {gone_strength:,} | {r['withdrawn_by_use_total']:,} | "
                   f"{1000 * r['cpu_per_byte_core_only']:.3f} / {1000 * r['cpu_per_byte_with_records']:.3f} | {r['emissions']['share']:.4f} | "
                   f"{', '.join(fmt(v, 4) for v in sc)} | {' / '.join(fmt(v) for v in sh)} |")
print("\n".join(out))
