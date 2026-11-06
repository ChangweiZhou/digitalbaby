"""Summarize runs/*.json into markdown tables and a compact JSON."""
import json
import re
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
ARMS = ("bp", "bp_v181", "bpw01", "bpw03", "bpw10", "dfa", "dfa_fix", "dfa_v181", "dfa_lowlr", "dfa_outwd", "sign", "signw10")
SEEDS = (73129, 93683, 116089)
LABEL = {"bp": "Backprop, Adam lr .003",
         "bp_v181": "Backprop, V181 settings (lr .001, wd 1e-4, dropout .1)",
         "dfa": "Local rule (DFA), Adam lr .003",
         "dfa_fix": "Local rule (DFA), frozen embedding, lr .003",
         "dfa_v181": "Local rule (DFA), V181 settings",
         "dfa_lowlr": "Local rule (DFA), lr .001, no decay",
         "dfa_outwd": "Local rule (DFA), lr .003, decay 1e-4 on output layer only",
         "sign": "Sign-symmetric feedback, lr .003",
         "bpw01": "Backprop, AdamW lr .001, decoupled wd 0.1",
         "bpw03": "Backprop, AdamW lr .001, decoupled wd 0.3",
         "bpw10": "Backprop, AdamW lr .001, decoupled wd 1.0",
         "signw10": "Sign-symmetric feedback, AdamW lr .001, decoupled wd 1.0"}


def pct(a, b):
    return f"{100 * a / b:.1f}%"


def first_step(curve, key, frac):
    for r in curve:
        if r[key]["exact"] / r[key]["n"] >= frac:
            return r["step"]
    return "—"


def load():
    out = {}
    for p in sorted(RUNS.glob("*.json")):
        m = re.match(r"([a-z0-9_]+?)_(\d+)(?:_f(\d+))?\.json$", p.name)
        if not m or p.name.startswith("SUMMARY") or p.name.startswith("QUALIFY"):
            continue
        arm, seed, f = m.group(1), int(m.group(2)), int(m.group(3) or 50)
        out[(arm, seed, f)] = json.loads(p.read_text())
    return out


def table(data, fracs):
    hdr = ("Rule", "Taught %", "Seed", "Updates", "Taught exact", "Held final", "Held peak (step)", "Held at monitor-best step",
           "Both orders final", "Carry final", "First ≥50%", "First ≥80%", "CPU min")
    md = "| " + " | ".join(hdr) + " |\n|" + "|".join("---" for _ in hdr) + "|\n"
    compact = {}
    for f in fracs:
        for arm in ARMS:
            for seed in SEEDS:
                d = data.get((arm, seed, f))
                if not d:
                    continue
                c = d["curve"]
                fin = c[-1]
                s, t = fin["sealed"], fin["taught"]
                best = max(c, key=lambda r: r["sealed"]["exact"])
                mb = max(c, key=lambda r: r["monitor"]["exact"])
                md += "| " + " | ".join(map(str, (
                    LABEL[arm], f, seed, fin["step"], pct(t["exact"], t["n"]), pct(s["exact"], s["n"]),
                    f"{pct(best['sealed']['exact'], best['sealed']['n'])} ({best['step']})",
                    f"{pct(mb['sealed']['exact'], mb['sealed']['n'])} ({mb['step']})",
                    pct(s["both_orders"], s["both_order_n"]), pct(*s["decimal_carry"]),
                    first_step(c, "sealed", .5), first_step(c, "sealed", .8), f"{fin['cpu'] / 60:.1f}"))) + " |\n"
                compact[f"{arm}_{seed}_f{f}"] = dict(
                    step=fin["step"], taught_n=t["n"], final_sealed=s["exact"], sealed_n=s["n"],
                    peak_sealed=[best["sealed"]["exact"], best["step"]], monitor_best=[mb["sealed"]["exact"], mb["step"]],
                    final_taught=t["exact"], sealed_curve=[(r["step"], r["sealed"]["exact"]) for r in c],
                    taught_curve=[(r["step"], r["taught"]["exact"]) for r in c])
    return md, compact


if __name__ == "__main__":
    data = load()
    fracs = sorted({f for _, _, f in data}, reverse=True)
    md, compact = table(data, fracs)
    print(md)
    (RUNS / "SUMMARY.md").write_text(md)
    (RUNS / "SUMMARY.json").write_text(json.dumps(compact, indent=1))
