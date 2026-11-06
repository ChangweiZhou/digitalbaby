"""Summarize exp1/life_*.json and exp1/dev_*.json against the pre-stated predictions."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "exp1"
ARMS = ("A1", "A1bp", "A2", "B", "B2", "C", "C1", "D", "E", "E0")
LABEL = {"A1": "A1 aligned birth, frozen trunk, sign rule", "A2": "A2 aligned birth, warm start, sign rule",
         "A1bp": "A1bp aligned birth, frozen trunk, backprop on top", "C1": "C1 random birth, frozen trunk, sign rule",
         "B": "B shuffled birth, frozen trunk, sign rule", "B2": "B2 shuffled birth, warm start, sign rule",
         "C": "C random birth, sign rule",
         "D": "D random birth, backprop (ceiling)", "E": "E aligned birth, three-factor rule, decay 1.0",
         "E0": "E0 aligned birth, three-factor rule, no decay"}
SEEDS = (73129, 93683, 116089)
pct = lambda a, b: 100 * a / b


def main():
    lines = ["## Development births\n", "| Birth | Seed | Taught exact | Sealed held (mod-97 addition) |", "|---|---|---|---|"]
    for b in ("aligned", "shuffled"):
        for s in SEEDS:
            p = OUT / f"dev_{b}_{s}.json"
            if p.exists():
                c = json.loads(p.read_text())["curve"][-1]
                lines.append(f"| {b} | {s} | {pct(c['taught']['exact'], c['taught']['n']):.1f}% | {pct(c['sealed']['exact'], c['sealed']['n']):.1f}% |")
    table = {}
    for task in ("sub97", "add89"):
        lines += [f"\n## Life task {task}\n"]
        for frac in (10, 25, 50):
            lines += [f"\n### {frac}% of pairs taught\n", "| Arm | Sealed held final (3 seeds) | Sealed held peak | At monitor-best | Wrap subset final | Taught final | Both orders final | Before life |", "|---|---|---|---|---|---|---|---|"]
            sc = None
            for arm in ARMS:
                fin, pk, mb, tt, bo, bf, wr = [], [], [], [], [], [], []
                for s in SEEDS:
                    p = OUT / f"life_{task}_f{frac}_{arm}_{s}.json"
                    if not p.exists():
                        continue
                    d = json.loads(p.read_text()); c = d["curve"]; f = c[-1]
                    sc = sc or d["shortcuts"]
                    n = f["sealed"]["n"]
                    fin.append(pct(f["sealed"]["exact"], n))
                    pk.append(pct(max(r["sealed"]["exact"] for r in c), n))
                    m = max(c, key=lambda r: r["monitor"]["exact"]); mb.append(pct(m["sealed"]["exact"], n))
                    tt.append(pct(f["taught"]["exact"], f["taught"]["n"]))
                    bo.append(pct(f["sealed"]["both_orders"], f["sealed"]["both_order_n"]))
                    bf.append(pct(d["sealed_before"]["exact"], n))
                    wr.append(pct(*f["sealed"]["wrap"]))
                    table[(task, frac, arm, s)] = dict(final=fin[-1], peak=pk[-1], monitor_best=mb[-1], taught=tt[-1], wrap=wr[-1])
                if fin:
                    j = lambda v: " · ".join(f"{x:.1f}" for x in v)
                    lines.append(f"| {LABEL[arm]} | {j(fin)} | {j(pk)} | {j(mb)} | {j(wr)} | {j(tt)} | {j(bo)} | {j(bf)} |")
            if sc:
                n = sc["n"]
                lines.append("\nShortcut ceilings on the sealed panel: " + ", ".join(f"{k} {pct(v, n):.1f}%" for k, v in sc.items() if k != "n"))
    # decision rule: A beats B and C by >= 10 points, sealed final, 3/3 seeds, at 25% taught
    lines += ["\n## Decision rule (sealed final, 25% taught, every seed)\n"]
    for task in ("sub97", "add89"):
        for A in ("A1", "A2"):
            margins = []
            for s in SEEDS:
                a = table.get((task, 25, A, s)); b = table.get((task, 25, "B", s)); c = table.get((task, 25, "C", s))
                if a and b and c:
                    margins.append((round(a["final"] - b["final"], 1), round(a["final"] - c["final"], 1)))
            ok = bool(margins) and all(x >= 10 and y >= 10 for x, y in margins)
            lines.append(f"- {task} {A}: margins over B / over C per seed = {margins} → {'ADVANCE' if ok else 'not met'}")
    md = "\n".join(lines)
    (OUT / "SUMMARY.md").write_text(md)
    (OUT / "SUMMARY.json").write_text(json.dumps({"|".join(map(str, k)): v for k, v in table.items()}, indent=1))
    print(md)


if __name__ == "__main__":
    main()
