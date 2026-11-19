# Builds TABLE.md from out/*.json (window A2) and diag/*.json. Marks: * S(d,1) > 0.090; + S(d,2) > c0(lag 2) + 0.02; # S(d,0) > 0.1103.
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
f = lambda v: "-" if v is None else f"{v:.3f}"
def cell(v, mark, thr):
    return "-" if v is None else f"{v:.3f}" + (mark if v > thr else "")
def row(lab):
    d = json.load(open(os.path.join(HERE, "out", lab + ".json"))); a = d["A2"]; c0 = a["c0"]
    S = lambda dd, L, mark, thr: cell(a["S"][dd][L], mark, thr)
    dg = os.path.join(HERE, "diag", lab + ".json")
    div = " / ".join(f"{json.load(open(dg))['divergence'][n]:.1e}" for n in ("16", "64", "512")) if os.path.exists(dg) else "not measured"
    return (f"| {lab} | {d['args'].get('seed', '73129')} | {d['conns_per_cell']:.1f} | {d['active_per_step']:.0f} | "
            + " / ".join(f(a["share"][x]) for x in "124") + " | " + " / ".join(S(x, "0", "#", 0.1103) for x in "124") + " | "
            + " / ".join(S(x, "1", "*", 0.090) for x in "124") + " | " + " / ".join(S(x, "2", "+", c0["2"] + 0.02) for x in "124") + " | "
            + " / ".join(f(c0.get(k)) for k in "012") + f" | {div} |")
cols = ["| tissue | seed | conns/cell | sending/step | share d1/d2/d4 | S(d,0) | S(d,1) | S(d,2) | c0 lag 0/1/2 | divergence after 16 / 64 / 512 bytes |",
        "|---|---|---|---|---|---|---|---|---|---|"]
labels = lambda fn: [l.split()[0] for l in open(os.path.join(HERE, fn))]
out = ["# Slow variables of one cell: does the previous byte reach the reading step? (19 November 2026)", "",
       "**The values of tau_h, beta, tau_u and rho_u below are a first trial, not a choice of the definition.**", "",
       "Window A2 (the last 4,096 bytes of 16,384). Labels: `A_h<tau_h>_b<beta>` = adaptation; `B_u<tau_u>_r<rho_u>` = depression of sending (`r1o4` is 1/4); `base` = neither. "
       "Columns: connections a cell; cells sending a step (mean number with output not zero); share neither silent nor constant at d = 1 / 2 / 4; S(d,0), S(d,1), S(d,2) at d = 1 / 2 / 4; "
       "c0 at lags 0 / 1 / 2 (what the present byte alone gives); the largest difference of outputs between two copies of the tissue, the second with one cell's potential changed by 1e-9 "
       "(`diag.py`). Marks: `#` S(d,0) above 0.1103; `*` S(d,1) above 0.090; `+` S(d,2) above its c0 plus 0.02.", "",
       "## 1. The base of the reviewer's message of 05:49: `rule=fixed gain=2 rho=1`, tau 32 and 128 steps (seed 73129)", ""] + cols + [row(l) for l in labels("batch_ratio_s73129.txt")]
out += ["", "## 2. The same base, slower variables: tau 512 and 2048 steps (16 and 64 bytes; seed 73129)", ""] + cols + [row(l) for l in labels("batch_ratio_slow_s73129.txt")]
out += ["", "## 3. The earlier base, `gain=1 astar=1/32 floor=1/128` (the reviewer later found that this control makes cells burst on their own; kept for the record)", "",
        "In these runs the depression option's parameter was named `rho` on the command line (see `batch_s73129.txt`); it is now `rho_u`, and `rho` is the ratio control's. "
        "`fixed` = rule=fixed, `coinc` = rule=coincidence.", ""] + cols + [row(l) for l in labels("batch_s73129.txt") + labels("batch_best_seeds.txt")]
open(os.path.join(HERE, "TABLE.md"), "w").write("\n".join(out) + "\n"); print("\n".join(out[:30]))
