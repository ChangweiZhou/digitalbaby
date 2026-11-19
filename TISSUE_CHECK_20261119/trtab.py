"""Table of own_trace.py results: last quarter per form (i)-(iv) for D and Q, levels, largest weight-row norm (8,192-byte steps)."""
import json, os
O = "/home/user/tissue_check/out/"
for t in ("A40", "L80", "L128", "A4"):
    for s in (73129, 93683, 116089):
        R = {}; lq = None
        for fm in ("13", "24"):
            f = O + f"trace_{t}_{s}_f{fm}.json"
            if os.path.exists(f): d = json.load(open(f)); R.update(d["rules"]); lq = d["levels_last_quarter"]
        if not R: continue
        row = []
        for form in ("(i) x", "(ii) x s2", "(iii) x s2 s4", "(iv) x s2 s4 s8"):
            row.append(" | ".join(f"{100*R[r+' '+form]['last_quarter']:5.2f}" if (r+' '+form) in R else "  -  " for r in ("D", "Q")))
        nm = max((v["norms"][-1]["max"] for v in R.values()))
        print(f"{t:4s} {s}: " + "   ".join(row) + f"   refs {100*lq['ref_1']:.2f} {100*lq['ref_2']:.2f} {100*lq['ref_3']:.2f} end {100*lq['end mark']:.2f}  max row norm {nm:.3g}")
