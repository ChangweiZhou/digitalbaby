"""Tables of plan 3 from the receipts, with plan 1's R and plan 2's turnover setting of each form on seed 73129."""
import json
from pathlib import Path
RUNS = Path(__file__).resolve().parent / "TISSUE_EVENT" / "runs"
TT = {"first": "P2_first_P2048_T2048_th0.25", "coincidence": "P2_coincidence_P2048_T512_th0.25"}
out = []
for form in ("first", "coincidence"):
    R = json.loads((RUNS / f"P1_{form}_R" / "receipt.json").read_text())
    rs1 = R["measures"]["A2"]["scores"]["1"]["by_lag"]["0"]["S"]
    out += [f"\n### {form}\n", "| Arm | Connections a cell at 1,024 / 4,096 / 8,192 / 12,288 / 16,384 | Last 4,096 bytes | Given up / withdrawn by use | ms a byte core / with records | Cells sending a step | S(1,0), S(2,0), S(4,0), S(8,0) in A2 | Informative share d = 1 / 2 / 4 | H3 met |", "|---|---|---|---|---|---|---|---|---|"]
    for name, label in [(f"P1_{form}_R", "R (plan 1)"), (TT[form], "turnover alone (plan 2)")] + [(f"P3_{form}_{a}", a) for a in ("N75", "N38", "N75T", "N38T")]:
        r = json.loads((RUNS / name / "receipt.json").read_text())
        cs = {c["bytes"][1] + 1: c["mean_connections"] for c in r["cpu_by_stretch"]}
        pts = [cs[b] for b in (1024, 4096, 8192, 12288, 16384)]
        ch = (pts[-1] - pts[-2]) / pts[-2]
        m = r["measures"]["A2"]
        def S(d):
            v = m["scores"].get(str(d), {}).get("by_lag", {}).get("0")
            return f"{v['S']:.3f}" if v else "–"
        def sh(d):
            v = m["shares"].get(str(d)); return f"{v['share']:.2f}" if v else "–"
        s1 = m["scores"]["1"]["by_lag"]["0"]["S"]
        ok = name.startswith("P3_") and not name.endswith("T") and pts[-1] < 47 and abs(ch) < 0.05 and s1 >= 0.1103 and s1 >= rs1
        out.append(f"| {label} | {' / '.join(f'{p:.1f}' for p in pts)} | {100 * ch:+.1f}% | {r['given_up_total']:,} / {r['withdrawn_by_use_total']:,} | "
                   f"{1000 * r['cpu_per_byte_core_only']:.2f} / {1000 * r['cpu_per_byte_with_records']:.2f} | {r['mean_cells_sending_a_step']:.1f} | "
                   f"{S(1)}, {S(2)}, {S(4)}, {S(8)} | {sh(1)} / {sh(2)} / {sh(4)} | {'yes' if ok else ('–' if not name.startswith('P3_') or name.endswith('T') else 'no')} |")
print("\n".join(out))
