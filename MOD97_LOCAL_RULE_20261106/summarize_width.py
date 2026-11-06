"""Width test: mixed vs single births at hidden 128 (Experiment 2), 256 and 512."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "exp2"
SEEDS = (73129, 93683, 116089)
pct = lambda a, b: 100 * a / b
tag = lambda h: "" if h == 128 else f"_h{h}"


def main():
    lines = ["## Development births by width: sealed held on own tasks (final)\n",
             "| Width | Birth | add83 | sub83 | add89 | sub89 | add97 | sub97 |", "|---|---|---|---|---|---|---|---|"]
    for h in (128, 256, 512):
        for b in ("mixed", "single"):
            cells = {k: [] for k in ("add83", "sub83", "add89", "sub89", "add97", "sub97")}
            for s in SEEDS:
                p = OUT / f"dev_{b}{tag(h)}_{s}.json"
                if p.exists():
                    c = json.loads(p.read_text())["curve"][-1]["sealed"]
                    for k in cells:
                        cells[k].append(pct(*c[k]))
            if cells["add97"]:
                lines.append(f"| {h} | {b} | " + " | ".join(" · ".join(f"{x:.0f}" for x in cells[k]) for k in cells) + " |")
    lines += ["\n## Life: sealed held exact, final (3 seeds), sign rule from each birth\n",
              "| Width | Arm | add73 10% | add73 25% | sub73 10% | sub73 25% | wrap add73 25% | wrap sub73 25% |", "|---|---|---|---|---|---|---|---|"]
    res = {}
    for h in (128, 256, 512):
        for arm in ("M", "S", "C"):
            row = []
            for task in ("add73", "sub73"):
                for f in (10, 25):
                    v = []
                    for s in SEEDS:
                        p = OUT / f"life_{task}_f{f}_{arm}{tag(h)}_{s}.json"
                        if p.exists():
                            d = json.loads(p.read_text()); fin = d["curve"][-1]["sealed"]
                            v.append(pct(fin["exact"], fin["n"])); res[(h, arm, task, f, s)] = (pct(fin["exact"], fin["n"]), pct(*fin["wrap"]))
                    row.append(" · ".join(f"{x:.0f}" for x in v) if v else "")
            wr = []
            for task in ("add73", "sub73"):
                wr.append(" · ".join(f"{res[(h, arm, task, 25, s)][1]:.0f}" for s in SEEDS if (h, arm, task, 25, s) in res))
            if any(row):
                lines.append(f"| {h} | {arm} | " + " | ".join(row + wr) + " |")
    lines += ["\n## Mixed minus single, per seed (sealed final)\n"]
    for h in (128, 256, 512):
        for task in ("add73", "sub73"):
            for f in (10, 25):
                m = [round(res[(h, 'M', task, f, s)][0] - res[(h, 'S', task, f, s)][0], 1) for s in SEEDS if (h, 'M', task, f, s) in res and (h, 'S', task, f, s) in res]
                if m:
                    lines.append(f"- width {h}, {task}, {f}%: {m}")
    md = "\n".join(lines)
    (OUT / "WIDTH_SUMMARY.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
