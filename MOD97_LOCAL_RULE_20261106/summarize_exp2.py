"""Summarize exp2 births and life runs against predictions P1-P5."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "exp2"
ARMS = ("M", "S", "H", "C", "D")
LABEL = {"M": "M mixed birth, sign rule", "S": "S single-task (add97) birth, sign rule", "H": "H shuffled mixed birth, sign rule",
         "C": "C random birth, sign rule", "D": "D random birth, backprop (ceiling)"}
SEEDS = (73129, 93683, 116089)
TASKS = ("add73", "sub73", "dec")
pct = lambda a, b: 100 * a / b


def main():
    lines = ["## Development births (sealed held on each development task, final)\n", "| Birth | Seed | " + " | ".join(f"{op}{m}" for m in (83, 89, 97) for op in ("add", "sub")) + " |", "|---|---|" + "---|" * 6]
    for b in ("mixed", "single", "shuffled"):
        for s in SEEDS:
            p = OUT / f"dev_{b}_{s}.json"
            if p.exists():
                c = json.loads(p.read_text())["curve"][-1]["sealed"]
                lines.append(f"| {b} | {s} | " + " | ".join(f"{pct(*c[f'{op}{m}']):.1f}%" for m in (83, 89, 97) for op in ("add", "sub")) + " |")
    table = {}
    for task in TASKS:
        lines += [f"\n## Life task {task}\n"]
        for frac in (10, 25):
            lines += [f"\n### {frac}% of pairs taught\n", "| Arm | Sealed held final (3 seeds) | Peak | At monitor-best | Wrap subset | Carry subset | Taught final | Both orders | Before life |", "|---|---|---|---|---|---|---|---|---|"]
            sc = None
            for arm in ARMS:
                cols = {k: [] for k in ("fin", "pk", "mb", "wr", "ca", "tt", "bo", "bf")}
                for s in SEEDS:
                    p = OUT / f"life_{task}_f{frac}_{arm}_{s}.json"
                    if not p.exists():
                        continue
                    d = json.loads(p.read_text()); c = d["curve"]; f = c[-1]; n = f["sealed"]["n"]
                    sc = sc or d["shortcuts"]
                    cols["fin"].append(pct(f["sealed"]["exact"], n))
                    cols["pk"].append(pct(max(r["sealed"]["exact"] for r in c), n))
                    mbest = max(c, key=lambda r: r["monitor"]["exact"]); cols["mb"].append(pct(mbest["sealed"]["exact"], n))
                    cols["wr"].append(pct(*f["sealed"]["wrap"])); cols["ca"].append(pct(*f["sealed"]["carry"]))
                    cols["tt"].append(pct(f["taught"]["exact"], f["taught"]["n"]))
                    cols["bo"].append(pct(f["sealed"]["both_orders"], f["sealed"]["both_order_n"]))
                    cols["bf"].append(pct(d["sealed_before"]["exact"], n))
                    table[(task, frac, arm, s)] = dict(final=cols["fin"][-1], peak=cols["pk"][-1], wrap=cols["wr"][-1], taught=cols["tt"][-1])
                if cols["fin"]:
                    j = lambda v: " · ".join(f"{x:.1f}" for x in v)
                    lines.append(f"| {LABEL[arm]} | " + " | ".join(j(cols[k]) for k in ("fin", "pk", "mb", "wr", "ca", "tt", "bo", "bf")) + " |")
            if sc:
                n = sc["n"]
                lines.append("\nShortcut ceilings on the sealed panel: " + ", ".join(f"{k} {pct(v, n):.1f}%" for k, v in sc.items() if k != "n"))
    lines += ["\n## Predictions (sealed final)\n"]
    def margins(task, frac, a, b):
        return [round(table[(task, frac, a, s)]["final"] - table[(task, frac, b, s)]["final"], 1) for s in SEEDS if (task, frac, a, s) in table and (task, frac, b, s) in table]
    for frac in (10, 25):
        lines.append(f"- P1 at {frac}%: M−S on sub73 {margins('sub73', frac, 'M', 'S')} (need ≥10); M−S on add73 {margins('add73', frac, 'M', 'S')} (need within 5)")
        lines.append(f"- P2 at {frac}%: M−C on add73 {margins('add73', frac, 'M', 'C')}, on sub73 {margins('sub73', frac, 'M', 'C')} (need ≥15 every seed)")
        lines.append(f"- P4 at {frac}%: H−C on add73 {margins('add73', frac, 'H', 'C')}, sub73 {margins('sub73', frac, 'H', 'C')}, dec {margins('dec', frac, 'H', 'C')} (expect ≤0)")
        lines.append(f"- P5 at {frac}%: D−M on add73 {margins('add73', frac, 'D', 'M')}, sub73 {margins('sub73', frac, 'D', 'M')}, dec {margins('dec', frac, 'D', 'M')} (≤10 at 25%, <0 at 10%)")
    lines.append(f"- P3 at 25%: M−C on dec {margins('dec', 25, 'M', 'C')} (need ≥10); M absolute {[round(table[('dec', 25, 'M', s)]['final'], 1) for s in SEEDS if ('dec', 25, 'M', s) in table]} (need ≥60)")
    md = "\n".join(lines)
    (OUT / "SUMMARY.md").write_text(md)
    (OUT / "SUMMARY.json").write_text(json.dumps({"|".join(map(str, k)): v for k, v in table.items()}, indent=1))
    print(md)


if __name__ == "__main__":
    main()
