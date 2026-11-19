# Builds TABLE.md from out/*.json (window A2). Marks: * S(d,1) > 0.090; + S(d,2) > c0(lag 2) + 0.02; # S(d,0) > 0.1103.
import json, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
order = [l.split()[0] for f in ("batch_s73129.txt", "batch_best_seeds.txt") for l in open(os.path.join(HERE, f))]
rows = []
for lab in order:
    d = json.load(open(os.path.join(HERE, "out", lab + ".json"))); a = d["A2"]; c0 = a["c0"]
    f = lambda v: "-" if v is None else f"{v:.3f}"
    def cell(v, mark, thr):
        return "-" if v is None else f"{v:.3f}" + (mark if v > thr else "")
    S = lambda dd, L, mark, thr: cell(a["S"][dd][L], mark, thr)
    rows.append(f"| {lab} | {d['args'].get('seed', '73129')} | {d['conns_per_cell']:.1f} | {d['active_per_step']:.0f} | "
                + " / ".join(f(a["share"][x]) for x in "124") + " | "
                + " / ".join(S(x, "0", "#", 0.1103) for x in "124") + " | "
                + " / ".join(S(x, "1", "*", 0.090) for x in "124") + " | "
                + " / ".join(S(x, "2", "+", c0["2"] + 0.02) for x in "124") + " | "
                + " / ".join(f(c0.get(k)) for k in "012") + f" | {d['ms_per_byte']:.2f} |")
head = ["# Slow variables of one cell: does the previous byte reach the reading step? (19 November 2026)", "",
        "**The values of tau_h, beta, tau_u and rho below are a first trial, not a choice of the definition.**", "",
        "Window A2 (the last 4,096 bytes of 16,384), every tissue with `gain=1 astar=1/32 floor=1/128`. Labels: `fixed` = rule=fixed (strengths never change), "
        "`coinc` = rule=coincidence; `A_h<tau_h>_b<beta>` = adaptation, `B_u<tau_u>_r<rho>` = depression of sending (`r1o4` is rho = 1/4); `base` = neither.", "",
        "Columns: connections a cell; cells sending a step (mean number with output not zero); share neither silent nor constant at d = 1 / 2 / 4; "
        "S(d,0), S(d,1), S(d,2) at d = 1 / 2 / 4; c0 at lags 0 / 1 / 2 (what the present byte alone gives); ms a byte. "
        "Marks: `#` S(d,0) above 0.1103; `*` S(d,1) above 0.090; `+` S(d,2) above its c0 plus 0.02.", "",
        "| tissue | seed | conns/cell | sending/step | share d1/d2/d4 | S(d,0) | S(d,1) | S(d,2) | c0 lag 0/1/2 | ms/byte |",
        "|---|---|---|---|---|---|---|---|---|---|"]
open(os.path.join(HERE, "TABLE.md"), "w").write("\n".join(head + rows) + "\n")
print("\n".join(head + rows))
