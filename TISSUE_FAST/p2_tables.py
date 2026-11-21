"""Tables of plan 2 from the receipts, with plan 1's R and W of each form beside them."""
import json
from pathlib import Path
RUNS = Path(__file__).resolve().parent / "TISSUE_EVENT" / "runs"
def row(name, label):
    f = RUNS / name / "receipt.json"
    if not f.exists():
        return f"| {label} | not run | | | | | | |"
    r = json.loads(f.read_text())
    cs = {c["bytes"][1] + 1: c["mean_connections"] for c in r["cpu_by_stretch"]}
    pts = [cs.get(b) for b in (1024, 4096, 8192, 12288, 16384)]
    ch = (pts[-1] - pts[-2]) / pts[-2]
    m = r.get("measures", {}).get("A2", {})
    def S(d):
        try: return f'{m["scores"][str(d)]["by_lag"]["0"]["S"]:.3f}'
        except Exception: return "–"
    sw = r["switches"]
    tr = f'{sw.get("trial_withdrawn", 0):,} / {sw.get("trial_kept", 0):,}' if sw.get("turnover") else "–"
    ok = pts[-1] < 47 and abs(ch) < 0.05 and m and m["scores"]["1"]["by_lag"]["0"]["S"] >= 0.1103
    return (f"| {label} | {' / '.join(f'{p:.1f}' for p in pts)} | {100 * ch:+.1f}% | {tr} | {r['withdrawn_by_use_total']:,} | "
            f"{1000 * r['cpu_per_byte_core_only']:.2f} / {1000 * r['cpu_per_byte_with_records']:.2f} | {S(1)}, {S(2)}, {S(4)} | {'yes' if ok else 'no'} |")
out = []
for form in ("first", "coincidence"):
    out += [f"\n### {form}\n", "| Setting (making period, trial, threshold) | Connections a cell at 1,024 / 4,096 / 8,192 / 12,288 / 16,384 bytes | Last 4,096 bytes | Trials withdrawn / kept | Withdrawn by use | ms a byte, core / with records | S(1,0), S(2,0), S(4,0) in A2 | H2 met |", "|---|---|---|---|---|---|---|---|"]
    out.append(row(f"P1_{form}_R", "plan 1 R (E63's making)"))
    out.append(row(f"P1_{form}_W", "plan 1 W (E63's making, withdrawal by use)"))
    for P in (512, 2048):
        for T in (512, 2048):
            for th in ("0.25", "1"):
                out.append(row(f"P2_{form}_P{P}_T{T}_th{th}", f"{P}, {T}, {'1/4' if th == '0.25' else '1'}"))
print("\n".join(out))
