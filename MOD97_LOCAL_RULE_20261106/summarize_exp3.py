"""Experiment 3 summary: held exact by distractor run length k, per architecture and arm."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "exp3"
SEEDS = (73129, 93683, 116089)
ARCHS = ("win", "rnn", "gru")
KS = ("0", "2", "4", "6", "8", "10", "12", "14", "16", "20")
LABEL = {"W": "warm, sign rule, no credit through time", "R": "random, sign rule, no credit through time", "B": "random, backprop with BPTT",
         "T": "warm, true gradient, no credit through time", "F": "warm, sign rule with credit through time", "G": "warm, backprop with BPTT",
         "E": "warm, sign rule with eligibility traces (local in time)", "ER": "random, sign rule with eligibility traces"}
pct = lambda v: 100 * v[0] / v[1]


def row(cells):
    return " | ".join(cells)


def main():
    lines = ["## Development births: sealed held exact by k on mod 97 (final), three seeds\n",
             "| Arch | params | " + " | ".join(f"k={k}" for k in KS) + " |", "|---|---|" + "---|" * len(KS)]
    for arch in ARCHS:
        per = {k: [] for k in KS}
        params = None
        for s in SEEDS:
            p = OUT / f"dev_{arch}_{s}.json"
            if p.exists():
                d = json.loads(p.read_text()); params = d["params"]
                sc = d["curve"][-1]["sealed"]["97"]
                for k in KS:
                    per[k].append(pct(sc[k]))
        if params:
            lines.append(f"| {arch} | {params:,} | " + row(" · ".join(f"{x:.0f}" for x in per[k]) for k in KS) + " |")
    lines += ["\n## Life on held-out mod 73, 25% taught: sealed held exact by k (final), three seeds\n",
              "| Arch | Arm | " + " | ".join(f"k={k}" for k in KS) + " |", "|---|---|" + "---|" * len(KS)]
    summary = {}
    for arch in ARCHS:
        for arm in ("W", "E", "T", "F", "G", "R", "ER", "B"):
            per = {k: [] for k in KS}
            for s in SEEDS:
                p = OUT / f"life_{arch}_{arm}_{s}.json"
                if p.exists():
                    d = json.loads(p.read_text()); sc = d["curve"][-1]["sealed"]
                    for k in KS:
                        per[k].append(pct(sc[k]))
                    summary[(arch, arm, s)] = {k: pct(sc[k]) for k in KS}
            if per["0"]:
                lines.append(f"| {arch} | {LABEL[arm]} | " + row(" · ".join(f"{x:.0f}" for x in per[k]) for k in KS) + " |")
    p = next(OUT.glob("life_*.json"), None)
    if p:
        sc = json.loads(p.read_text())["shortcuts"]
        lines.append("\nShortcut floors on the sealed panel: " + ", ".join(f"{k} {100 * v / sc['n']:.1f}%" for k, v in sc.items() if k != "n"))
    lines += ["\n## Predictions\n"]
    def mean_k(arch, arm, ks):
        vals = [summary[(arch, arm, s)][k] for s in SEEDS for k in ks if (arch, arm, s) in summary]
        return sum(vals) / len(vals) if vals else None
    f = lambda v: "n/a" if v is None else f"{v:.0f}%"
    lines.append(f"- P1 window net at k≥6 (first operand out of window): dev-birth life {f(mean_k('win','W',('6','8','10','12')))}, backprop life {f(mean_k('win','B',('6','8','10','12')))} (expect floor ≈ no-wrap half or below)")
    lines.append(f"- P2 after development, held at k≤12: gru W {f(mean_k('gru','W',('0','2','4','6','8','10','12')))} (need ≥80 at birth; see dev table), rnn at k≥10 {f(mean_k('rnn','W',('10','12')))} (expect <50)")
    lines.append(f"- P3 within-step local rule: gru warm k≤12 {f(mean_k('gru','W',('0','2','4','6','8','10','12')))} (need ≥70); gru random k≥6 {f(mean_k('gru','R',('6','8','10','12')))} (expect <30); rnn random k≥6 {f(mean_k('rnn','R',('6','8','10','12')))}")
    lines.append(f"- P4 extrapolation k 14-20: gru W {f(mean_k('gru','W',('14','16','20')))} (need ≥50), gru B {f(mean_k('gru','B',('14','16','20')))}, rnn W {f(mean_k('rnn','W',('14','16','20')))}, win W {f(mean_k('win','W',('14','16','20')))}")
    md = "\n".join(lines)
    (OUT / "SUMMARY.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
