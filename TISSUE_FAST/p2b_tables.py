"""Tables of plan 2b from the receipts (seed 73129 rows are plan 1 and plan 2's runs)."""
import json
from pathlib import Path
RUNS = Path(__file__).resolve().parent / "TISSUE_EVENT" / "runs"
def get(name):
    f = RUNS / name / "receipt.json"
    if not f.exists(): return None
    r = json.loads(f.read_text())
    cs = {c["bytes"][1] + 1: c["mean_connections"] for c in r["cpu_by_stretch"]}
    m = r["measures"]["A2"]
    S = lambda d: m["scores"][str(d)]["by_lag"]["0"]["S"]
    return dict(end=cs[16384], ch=(cs[16384] - cs[12288]) / cs[12288], s1=S(1), s2=S(2), s4=S(4), ms=1000 * r["cpu_per_byte_core_only"])
sets = {"first": ["P2048_T512_th0.25", "P2048_T2048_th0.25"], "coincidence": ["P2048_T512_th1", "P2048_T512_th0.25"]}
out = []; cases = []
for form, arms in sets.items():
    out += [f"\n### {form}\n", "| Seed | Setting | Connections a cell at the end | Last 4,096 bytes | S(1,0), S(2,0), S(4,0) in A2 | R on that seed: connections, S(1,0) | ms a byte (core) | Holds |", "|---|---|---|---|---|---|---|---|"]
    for seed in (73129, 93683, 116089):
        R = get(f"P1_{form}_R") if seed == 73129 else get(f"P2B_{form}_R_s{seed}")
        for arm in arms:
            a = get(f"P2_{form}_{arm}") if seed == 73129 else get(f"P2B_{form}_{arm}_s{seed}")
            ok = a["end"] < 47 and abs(a["ch"]) < 0.05 and a["s1"] >= 0.1103 and a["s1"] >= R["s1"]
            if seed != 73129: cases.append(ok)
            out.append(f"| {seed} | {arm} | {a['end']:.1f} | {100 * a['ch']:+.1f}% | {a['s1']:.3f}, {a['s2']:.3f}, {a['s4']:.3f} | {R['end']:.1f}, {R['s1']:.3f} | {a['ms']:.2f} | {'yes' if ok else 'no'}{' (chosen on this seed)' if seed == 73129 else ''} |")
out.append(f"\nNew-seed cases holding: {sum(cases)} of {len(cases)}")
print("\n".join(out))
