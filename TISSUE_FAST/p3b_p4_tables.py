"""Tables of plans 3b and 4 from the receipts and window A2."""
import json
import numpy as np
from pathlib import Path
RUNS = Path(__file__).resolve().parent / "TISSUE_EVENT" / "runs"
def get(name):
    r = json.loads((RUNS / name / "receipt.json").read_text())
    cs = {c["bytes"][1] + 1: c["mean_connections"] for c in r["cpu_by_stretch"]}
    m = r["measures"]["A2"]
    S = lambda d: m["scores"][str(d)]["by_lag"]["0"]["S"] if m["scores"].get(str(d), {}).get("by_lag", {}).get("0") else float("nan")
    th = np.load(RUNS / name / "A2.npz")["theta_mean"]
    return dict(end=cs[16384], ch=(cs[16384] - cs[12288]) / cs[12288], s1=S(1), s2=S(2), s4=S(4), ms=1000 * r["cpu_per_byte_core_only"],
                snd=r["mean_cells_sending_a_step"], thpos=float((th > 0).mean()), thmean=float(th.mean()))
out = ["## Plan 3b", ""]
cases = {"N75": [], "N75T": []}
for form in ("first", "coincidence"):
    out += [f"### {form}", "", "| Seed | Arm | Connections at the end | Last 4,096 bytes | S(1,0), S(2,0), S(4,0) | R: S(1,0) | ms a byte | Holds |", "|---|---|---|---|---|---|---|---|"]
    for seed in (73129, 93683, 116089):
        R = get(f"P1_{form}_R") if seed == 73129 else get(f"P2B_{form}_R_s{seed}")
        a = get(f"P3_{form}_N75") if seed == 73129 else get(f"P3B_{form}_N75_s{seed}")
        b = get(f"P3_{form}_N75T") if seed == 73129 else get(f"P3B_{form}_N75T_s{seed}")
        oka = a["end"] < 47 and abs(a["ch"]) < 0.05 and a["s1"] >= 0.1103 and a["s1"] >= R["s1"]
        okb = b["s1"] >= a["s1"] and b["s2"] >= a["s2"]
        if seed != 73129: cases["N75"].append(oka); cases["N75T"].append(okb)
        tag = " (chosen on this seed)" if seed == 73129 else ""
        for nm, x, ok in (("N75", a, oka), ("N75T", b, okb)):
            out.append(f"| {seed} | {nm} | {x['end']:.1f} | {100 * x['ch']:+.1f}% | {x['s1']:.3f}, {x['s2']:.3f}, {x['s4']:.3f} | {R['s1']:.3f} | {x['ms']:.2f} | {'yes' if ok else 'no'}{tag} |")
    out.append("")
out.append(f"New-seed cases holding: N75 {sum(cases['N75'])} of 4, N75T {sum(cases['N75T'])} of 4")
out += ["", "## Plan 4", "", "| Form | a* | Connections at the end | Last 4,096 bytes | Cells sending a step | Cells with mean threshold above 0 in A2 | Mean threshold in A2 | S(1,0), S(2,0), S(4,0) | ms a byte |", "|---|---|---|---|---|---|---|---|---|"]
for form in ("first", "coincidence"):
    for lab, name in (("1/8 (plan 3)", f"P3_{form}_N75T"), ("1/32", f"P4_{form}_N75T_astar1_32"), ("1/128", f"P4_{form}_N75T_astar1_128")):
        x = get(name)
        out.append(f"| {form} | {lab} | {x['end']:.1f} | {100 * x['ch']:+.1f}% | {x['snd']:.1f} | {x['thpos']:.3f} | {x['thmean']:.4f} | {x['s1']:.3f}, {x['s2']:.3f}, {x['s4']:.3f} | {x['ms']:.2f} |")
print("\n".join(out))
