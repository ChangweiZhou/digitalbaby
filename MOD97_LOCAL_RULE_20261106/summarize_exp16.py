"""Tables for E16 from exp45/e16_life_{T,D}_{seed}.json -> exp45/SUMMARY_E16.md."""
import json
import os
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "exp45"
SEEDS = tuple(int(x) for x in os.environ.get("E16_SEEDS", "73129,93683,116089").split(","))   # E16-F: E16_SEEDS=200003,300007,400009
SUMMARY = OUT / os.environ.get("E16_SUMMARY", "SUMMARY_E16.md")
F1 = ("add97", "sub97", "max97", "min97"); F3 = ("add73", "sub73", "max73", "min73"); DEV = ("add89", "sub89", "max89", "min89")
HELD = ("+-", "-+", "><", "<>")
pct = lambda d: 100 * d["exact"] / d["n"]
pctf = lambda d: 100 * d["final"] / d["n"]


ARMS = ("T", "D", "M", "T_oracle", "T_balanced", "T_all_balanced", "T_share", "T_res2048", "T_res4096", "T_res8192", "T_res16384", "T_tot24576", "T_tot65536")
LABEL = {"T": "T learner (uniform replay)", "D": "D twin (joint ground truth)", "M": "M backprop on T's batches", "T_oracle": "T on the twin's stream",
         "T_balanced": "T balanced per family (1,024/block)", "T_all_balanced": "T balanced, every entry", "T_share": "T earlier families 50% of batch",
         "T_res2048": "T reservoir 2,048/family", "T_res4096": "T reservoir 4,096/family", "T_res8192": "T reservoir 8,192/family", "T_res16384": "T reservoir 16,384/family", "T_tot24576": "T one reservoir 24,576, no labels", "T_tot65536": "T one reservoir 65,536, no labels"}


def load(arm, seed):
    p = OUT / f"e16_life_{arm}_{seed}.json"
    return json.load(open(p)) if p.exists() else None


def fam_mean(sc, fam, k=12):
    return sum(pct(sc[n][str(k)] if str(k) in sc[n] else sc[n][k]) for n in fam) / len(fam)


def K(sc, n, k):
    d = sc[n]; return d[str(k)] if str(k) in d else d[k]


def main():
    lines = ["# E16 summary: the integrated toy learner\n", "Scores are exact answers (%) on sealed panels; composition rows give final-answer accuracy. Blocks: F1.1–F1.2 mod-97 operations with distractors, F2.1–F2.2 taught operator pairs, F3.1–F3.2 mod-73 operations.\n"]
    runs = {(a, s): load(a, s) for a in ARMS for s in SEEDS}
    runs = {k: v for k, v in runs.items() if v is not None}
    # per-block family means
    lines += ["## Family means by block (k = 12 for operations; held-pair final-answer mean for composition)\n",
              "| Arm | Seed | Block | F1 mean | F1 add | F1 sub | F3 mean | Held comp | +- | -+ | >< | <> | within | dev89 mean |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (a, s), d in runs.items():
        for st in d["stages"]:
            sc = st["scores"]; c = sc["compose"]
            lines.append(f"| {a} | {s} | {st['task'] or 'birth'} | {fam_mean(sc, F1):.1f} | {pct(K(sc,'add97',12)):.1f} | {pct(K(sc,'sub97',12)):.1f} | {fam_mean(sc, F3):.1f} | "
                         f"{sum(pctf(c[h]) for h in HELD)/4:.1f} | " + " | ".join(f"{pctf(c[h]):.1f}" for h in HELD) + f" | {pctf(c['within']):.1f} | {fam_mean(sc, DEV):.1f} |")
    # final table: T vs D and the 80% gate
    lines += ["\n## After the last block: learner against twin (gate: T ≥ 80% of D on every family)\n",
              "| Seed | Arm: family | arm | D | arm/D | F1 after block 2 (T) | T k=20 | T silenced k=12 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in SEEDS:
        dd = runs.get(("D", s))
        if not dd or len(dd["stages"]) < 7:
            continue
        for a in ARMS:
            t = runs.get((a, s))
            if a == "D" or not t or len(t["stages"]) < 7:
                continue
            st, sd = t["stages"][-1]["scores"], dd["stages"][-1]["scores"]; s2 = t["stages"][2]["scores"]
            rows = [("F1 ops k=12", fam_mean(st, F1), fam_mean(sd, F1), fam_mean(s2, F1), sum(pct(K(st, n, 20)) for n in F1) / 4, sum(pct(st["silenced_k12"][n]) for n in F1) / 4),
                    ("F3 ops k=12", fam_mean(st, F3), fam_mean(sd, F3), None, sum(pct(K(st, n, 20)) for n in F3) / 4, sum(pct(st["silenced_k12"][n]) for n in F3) / 4),
                    ("held comp final", sum(pctf(st["compose"][h]) for h in HELD) / 4, sum(pctf(sd["compose"][h]) for h in HELD) / 4, None, None, sum(pctf(st["silenced_compose"][h]) for h in HELD) / 4)]
            for name, tv, dv, b2, k20, sil in rows:
                lines.append(f"| {s} | {LABEL.get(a, a)}: {name} | {tv:.1f} | {dv:.1f} | {tv/dv:.2f} | {'' if b2 is None else f'{b2:.1f}'} | {'' if k20 is None else f'{k20:.1f}'} | {sil:.1f} |")
    # E16-D: retention across arms
    lines += ["\n## Retention across arms: what each family keeps after the stream moves on (block 6 vs the family's own last block)\n",
              "| Arm | Seed | F1 k=12 after 2 → 6 | F1 add/sub after 6 | Held comp after 4 → 6 | +- / -+ after 6 | within after 4 → 6 | F3 after 6 |", "|---|---|---|---|---|---|---|---|"]
    for (a, s), d in runs.items():
        if len(d["stages"]) < 7:
            continue
        s2, s4, s6 = (d["stages"][i]["scores"] for i in (2, 4, 6))
        hc = lambda sc: sum(pctf(sc["compose"][h]) for h in HELD) / 4
        lines.append(f"| {LABEL.get(a, a)} | {s} | {fam_mean(s2, F1):.1f} → {fam_mean(s6, F1):.1f} | {pct(K(s6,'add97',12)):.0f}/{pct(K(s6,'sub97',12)):.0f} | {hc(s4):.1f} → {hc(s6):.1f} | "
                     f"{pctf(s6['compose']['+-']):.0f}/{pctf(s6['compose']['-+']):.0f} | {pctf(s4['compose']['within']):.0f} → {pctf(s6['compose']['within']):.0f} | {fam_mean(s6, F3):.1f} |")
    # per-operation detail at the end
    lines += ["\n## Per-operation exact % at k = 0 / 8 / 12 / 20 after the last block\n", "| Arm | Seed | " + " | ".join(F1 + F3) + " |", "|---|---|" + "---|" * 8]
    for (a, s), d in runs.items():
        if len(d["stages"]) >= 7:
            sc = d["stages"][-1]["scores"]
            lines.append(f"| {a} | {s} | " + " | ".join("/".join(f"{pct(K(sc, n, k)):.0f}" for k in (0, 8, 12, 20)) for n in F1 + F3) + " |")
    if runs:
        sc0 = next(iter(runs.values()))["stages"][0]
        lines += ["\n## Shortcut ceilings on the held pairs (final answer from a visible-only rule, %)\n", "| Pair | ignore o1, use a | ignore o1, use b | ignore o2 | answer = c |", "|---|---:|---:|---:|---:|"]
        for h, v in sc0["shortcut_ceilings"].items():
            lines.append(f"| {h} | " + " | ".join(f"{100*v[k]:.1f}" for k in ("ignore_o1_use_a", "ignore_o1_use_b", "ignore_o2", "answer_is_c")) + " |")
    else:
        lines.append(f"\nNo receipts found for seeds {SEEDS}.")
    SUMMARY.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
