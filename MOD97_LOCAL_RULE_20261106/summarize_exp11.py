"""Summary tables for E11 (exp11_cell.py). Writes exp45/SUMMARY_E11.md."""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "exp45"
SEEDS = (73129, 93683, 116089)
KS = (0, 8, 12, 20)
L97 = ("add97", "sub97", "max97", "min97")
D89 = ("add89", "sub89", "max89", "min89")


def pct(d):
    return f"{100 * d['exact'] / d['n']:.0f}"


def main():
    lines = ["# E11 summary: diagonal memory cell with a time-local trace in the E8 learner", ""]
    lines += ["## Births (backprop through time, 24,576 updates; exact %, four mod-89 operations)", "",
              "| Birth | Seed | k=0 | k=4 | k=8 | k=12 | k=20 | k=12 silenced | cpu s |", "|---|---|---|---|---|---|---|---|---|"]
    for b in "MN":
        for s in SEEDS:
            f = OUT / f"e11_dev_{b}_{s}.json"
            if not f.exists():
                continue
            d = json.load(open(f)); last = d["curve"][-1]
            row = [" ".join(pct(last["sealed"][n][str(k)]) for n in D89) for k in (0, 4, 8, 12, 20)]
            sil = " ".join(pct(last["silenced_k12"][n]) for n in D89) if "silenced_k12" in last else "-"
            lines.append(f"| {b} | {s} | " + " | ".join(row) + f" | {sil} | {last['cpu']:.0f} |")
    lines += ["", "Each cell: add sub max min.", ""]
    for arm in ("MT", "MS", "MD", "NT", "NS", "ND"):
        files = [(s, OUT / f"e11_life_{arm}_{s}.json") for s in SEEDS]
        files = [(s, f) for s, f in files if f.exists()]
        if not files:
            continue
        lines += [f"## Life {arm}", "", "| Seed | After | add97 (k 0/8/12/20) | sub97 | max97 | min97 | dev mod-89 @12 | silenced @12 (97 ops) | cpu s |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for s, f in files:
            d = json.load(open(f))
            for st in d["stages"]:
                sc = st["scores"]
                cells = ["/".join(pct(sc[n][str(k)]) for k in KS) for n in L97]
                dev = " ".join(pct(sc[n]["12"]) for n in D89)
                sil = " ".join(pct(sc["silenced_k12"][n]) for n in L97)
                lines.append(f"| {s} | {st['task'] or 'birth'} | " + " | ".join(cells) + f" | {dev} | {sil} | {st.get('cpu', 0):.0f} |")
        lines.append("")
    # headline: k=12 after own stage, and after the fourth stage
    lines += ["## Headline: k = 12 exact % on each mod-97 operation after its own stage → after the fourth stage", "",
              "| Arm | Seed | add97 | sub97 | max97 | min97 | mean after own | mean final | k=20 final mean |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in ("MT", "MS", "MD", "NT", "NS", "ND"):
        for s in SEEDS:
            f = OUT / f"e11_life_{arm}_{s}.json"
            if not f.exists():
                continue
            d = json.load(open(f)); st = d["stages"]
            if len(st) < 5:
                continue
            own = [100 * st[i + 1]["scores"][n]["12"]["exact"] / 400 for i, n in enumerate(L97)]
            fin = [100 * st[4]["scores"][n]["12"]["exact"] / 400 for n in L97]
            k20 = [100 * st[4]["scores"][n]["20"]["exact"] / 400 for n in L97]
            lines.append(f"| {arm} | {s} | " + " | ".join(f"{o:.0f}→{v:.0f}" for o, v in zip(own, fin)) + f" | {sum(own)/4:.0f} | {sum(fin)/4:.0f} | {sum(k20)/4:.0f} |")
    (OUT / "SUMMARY_E11.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
