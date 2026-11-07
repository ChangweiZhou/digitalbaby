"""Summarize E4 (retention) and E5 (composition) against E45_PREDICTIONS.md."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "exp45"
SEEDS = (73129, 93683, 116089)
E4_LIFE = ("add97", "sub97", "add73", "sub73")
E4_DEV = ("add83", "sub83", "add89", "sub89")
E4_ARMS = ("A", "A0", "B", "C", "Cs", "D")
E4_LABEL = {"A": "A shared path alone, decay 1.0", "A0": "A0 shared path alone, decay 0.1", "B": "B + episodic store, no replay",
            "C": "C + store + self-generated replay", "Cs": "C′ + store + replay with random answers", "D": "D backprop, joint ground truth (ceiling)"}
E5_ARMS = ("Z", "DIR", "SCR", "SCRsh", "DIRbp", "SCRbp", "SCRr", "SCRrbp")
E5_LABEL = {"Z": "Z zero-shot (no life)", "DIR": "DIR direct, sign rule", "SCR": "SCR scratchpad, sign rule", "SCRsh": "SCR-sh scratchpad, shuffled finals",
            "DIRbp": "DIR-bp direct, backprop", "SCRbp": "SCR-bp scratchpad, backprop", "SCRr": "SCR-R scratchpad, sign rule, replay of single ops", "SCRrbp": "SCR-R-bp scratchpad, backprop, replay of single ops"}
pct = lambda d, k="exact": 100 * d[k] / d["n"]


def e4():
    lines = ["## E4 retention: sealed held exact (%) on each life task after each stage, three seeds\n"]
    table = {}
    for arm in E4_ARMS:
        lines.append(f"\n### {E4_LABEL[arm]}\n")
        lines.append("| After stage | " + " | ".join(E4_LIFE) + " | dev tasks (mean) | T1 taught |")
        lines.append("|---|" + "---|" * (len(E4_LIFE) + 2))
        for si in range(0, 5):
            cells = []
            for n in E4_LIFE:
                v = []
                for s in SEEDS:
                    p = OUT / f"e4_life_{arm}_{s}.json"
                    if p.exists():
                        d = json.loads(p.read_text())
                        if si < len(d["stages"]):
                            v.append(pct(d["stages"][si]["scores"][n]["sealed"]))
                            table[(arm, s, si, n)] = v[-1]
                cells.append(" · ".join(f"{x:.0f}" for x in v))
            dev, t1 = [], []
            for s in SEEDS:
                p = OUT / f"e4_life_{arm}_{s}.json"
                if p.exists():
                    d = json.loads(p.read_text())
                    if si < len(d["stages"]):
                        sc = d["stages"][si]["scores"]
                        dev.append(sum(pct(sc[n]["sealed"]) for n in E4_DEV) / len(E4_DEV))
                        t1.append(pct(sc["add97"]["taught"]))
            if any(cells):
                label = "birth" if si == 0 else f"{si} ({E4_LIFE[si - 1]})"
                lines.append(f"| {label} | " + " | ".join(cells) + " | " + " · ".join(f"{x:.0f}" for x in dev) + " | " + " · ".join(f"{x:.0f}" for x in t1) + " |")
    lines.append("\n### E4 predictions (sealed T1 = add97 after stage 4 unless stated)\n")
    g = lambda arm, si, n: [round(table[(arm, s, si, n)]) for s in SEEDS if (arm, s, si, n) in table]
    lines.append(f"- P4.1 A: T1 after stage 1 {g('A', 1, 'add97')} (≥70), after stage 4 {g('A', 4, 'add97')} (<30)")
    lines.append(f"- P4.2 A0: T1 after stage 4 {g('A0', 4, 'add97')} (<50, above A)")
    lines.append(f"- P4.3 B: T1 sealed after stage 4 {g('B', 4, 'add97')} (within 10 of A)")
    lines.append(f"- P4.4 C: after stage 4, T1 {g('C', 4, 'add97')}, T2 {g('C', 4, 'sub97')}, T3 {g('C', 4, 'add73')} (≥70), T4 {g('C', 4, 'sub73')} (≥70)")
    lines.append(f"- P4.5 C′: T1 after stage 4 {g('Cs', 4, 'add97')} (tracks A)")
    lines.append(f"- P4.6 D: after stage 4 T1..T4 {[g('D', 4, n) for n in E4_LIFE]} (≥85)")
    return "\n".join(lines)


def e5():
    lines = ["\n## E5 composition: final-answer exact (%) on held two-step panels at the end of life, three seeds\n",
             "| Arm | pp within-type | mm within-type | pm cross-type | mp cross-type | intermediate ok (cross, mean) | single ops after life (add97 · sub97, mean over seeds) |", "|---|---|---|---|---|---|---|"]
    table = {}
    for arm in E5_ARMS:
        cols = {t: [] for t in ("pp", "mm", "pm", "mp")}
        inter, single = [], []
        for s in SEEDS:
            p = OUT / f"e5_life_{arm}_{s}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text()); last = d["curve"][-1]
            for t in cols:
                cols[t].append(pct(last["compose"][t], "final")); table[(arm, s, t)] = cols[t][-1]
            if last["compose"]["pm"]["intermediate"] is not None:
                inter.append((pct(last["compose"]["pm"], "intermediate") + pct(last["compose"]["mp"], "intermediate")) / 2)
            single.append((pct(last["single"]["add97"]), pct(last["single"]["sub97"])))
        if cols["pp"]:
            j = lambda v: " · ".join(f"{x:.0f}" for x in v)
            sm = f"{sum(a for a, _ in single) / len(single):.0f} · {sum(b for _, b in single) / len(single):.0f}"
            lines.append(f"| {E5_LABEL[arm]} | {j(cols['pp'])} | {j(cols['mm'])} | {j(cols['pm'])} | {j(cols['mp'])} | {j(inter) if inter else '—'} | {sm} |")
    p = next(OUT.glob("e5_life_DIR_*.json"), None)
    if p:
        sc = json.loads(p.read_text())["shortcuts"]
        lines.append("\nShortcut floors (percent of panel): " + "; ".join(f"{t}: " + ", ".join(f"{k} {100 * v / sc[t]['n']:.1f}" for k, v in sc[t].items() if k != "n") for t in sc))
    g = lambda arm, t: [round(table[(arm, s, t)]) for s in SEEDS if (arm, s, t) in table]
    lines += ["\n### E5 predictions\n",
              f"- P5.1 Z under 5 everywhere: pp {g('Z', 'pp')}, pm {g('Z', 'pm')}",
              f"- P5.2 DIR cross-type <30: pm {g('DIR', 'pm')}, mp {g('DIR', 'mp')}; within-type ≥60: pp {g('DIR', 'pp')}, mm {g('DIR', 'mm')}",
              f"- P5.3 SCR cross-type ≥70: pm {g('SCR', 'pm')}, mp {g('SCR', 'mp')}; within-type ≥80: pp {g('SCR', 'pp')}, mm {g('SCR', 'mm')}",
              f"- P5.4 SCR-sh under 10: pm {g('SCRsh', 'pm')}, pp {g('SCRsh', 'pp')}",
              f"- ceilings: DIR-bp pm {g('DIRbp', 'pm')}, SCR-bp pm {g('SCRbp', 'pm')}",
              f"- E5-R: SCR-R cross-type ≥70: pm {g('SCRr', 'pm')}, mp {g('SCRr', 'mp')}; within: pp {g('SCRr', 'pp')}, mm {g('SCRr', 'mm')}; backprop with replay pm {g('SCRrbp', 'pm')}, mp {g('SCRrbp', 'mp')}"]
    return "\n".join(lines)


if __name__ == "__main__":
    md = e4() + "\n" + e5()
    (OUT / "SUMMARY.md").write_text(md)
    print(md)


def e5c(tag=""):
    ARMS = ("SCR", "SCRc", "SCRcsh", "SCRcbp")
    LABEL = {"SCR": "SCR plain scratchpad, sign rule (left-noise development)", "SCRc": "SCRc canonical scratchpad, sign rule",
             "SCRcsh": "SCRc-sh canonical, shuffled finals", "SCRcbp": "SCRc-bp canonical, backprop"}
    lines = [f"\n## E5-C canonical scratchpad{tag}: final-answer exact (%) on held panels at the end of life, three seeds\n",
             "| Arm | pp within | mm within | pm cross | mp cross | step prompts right, cross (step 1 · step 2) | intermediate right, cross | single ops after life |", "|---|---|---|---|---|---|---|---|"]
    table = {}
    for arm in ARMS:
        cols = {t: [] for t in ("pp", "mm", "pm", "mp")}; s1, s2, inter, single = [], [], [], []
        for s in SEEDS:
            p = OUT / f"e5c_life_{arm}{tag}_{s}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text()); last = d["curve"][-1]
            for t in cols:
                cols[t].append(pct(last["compose"][t], "final")); table[(arm, s, t)] = cols[t][-1]
            cr = [last["compose"]["pm"], last["compose"]["mp"]]
            inter.append(sum(pct(c, "intermediate") for c in cr) / 2)
            if "step1_prompt" in cr[0]:
                s1.append(sum(pct(c, "step1_prompt") for c in cr) / 2); s2.append(sum(pct(c, "step2_prompt") for c in cr) / 2)
            single.append((pct(last["single"]["add97"]) + pct(last["single"]["sub97"])) / 2)
        if cols["pp"]:
            j = lambda v: " · ".join(f"{x:.0f}" for x in v)
            lines.append(f"| {LABEL[arm]} | {j(cols['pp'])} | {j(cols['mm'])} | {j(cols['pm'])} | {j(cols['mp'])} | {(j(s1) + ' / ' + j(s2)) if s1 else '—'} | {j(inter)} | {j(single)} |")
    g = lambda arm, t: [round(table[(arm, s, t)]) for s in SEEDS if (arm, s, t) in table]
    lines += ["\n### E5-C predictions\n",
              f"- P5C.1 SCRc cross-type ≥70: pm {g('SCRc', 'pm')}, mp {g('SCRc', 'mp')}",
              f"- P5C.3 SCRc-sh <10: pm {g('SCRcsh', 'pm')}",
              f"- P5C.4 SCR ≤60 cross-type: pm {g('SCR', 'pm')}, mp {g('SCR', 'mp')}",
              f"- ceiling SCRc-bp: pm {g('SCRcbp', 'pm')}, mp {g('SCRcbp', 'mp')}"]
    return "\n".join(lines)


if __name__ == "__main__":
    md = e5c() + "\n" + e5c("_w18")
    (OUT / "SUMMARY_E5C.md").write_text(md)
    print(md)
