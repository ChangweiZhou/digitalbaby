"""E37 tables: life minus fresh at equal items of the family, per rule, seed and comparison (the brief's fixed reading).

    python3 e37_report.py      -> exp45/e37/E37_TABLE.json and E37_TABLE.md (read-only on the receipts)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E37, E26, E31 = HERE / "exp45/e37", HERE / "exp45/e26", HERE / "exp45/e31"
SEEDS = (73129, 93683, 116089)
COMPS = [("F3 at ~131,000 items", "F3", 5, 2048, 73), ("F3 at ~262,000 items", "F3", 6, 4096, 73), ("F1 at 393,216 items", "F1", 1, 6144, 97)]
LIVES = {"S": "S", "B": "B", "K": "S"}          # life arm -> fresh rule it is read against


def life(arm, seed):
    f = E31 / f"e31_life_K_{seed}.json" if arm == "K" else E26 / f"e26_life_{arm}_{seed}.json"
    return json.loads(f.read_text())["stages"]


def pct(sc):
    return 100 * sc["exact"] / sc["n"]


def verdict(ds):
    if sum(d >= 15 for d in ds) >= 2:
        return "saving"
    if sum(d <= -15 for d in ds) >= 2:
        return "cost"
    if sum(abs(d) < 15 for d in ds) >= 2:
        return "none"
    return "mixed"


def main():
    rows, verdicts = [], {}
    for name, part, stage, row, mod in COMPS:
        for arm, rule in LIVES.items():
            ds = []
            for seed in SEEDS:
                st = life(arm, seed)[stage]
                ls = {o: pct(st["scores"][f"{o}{mod}"]["12"]) for o in ("add", "sub", "max", "min")}
                rec = json.loads((E37 / f"e37_{part}_{rule}_{seed}.json").read_text())
                assert rec["complete"] is True
                cr = next(c for c in rec["curve"] if c["step"] == row)
                fs = {o: pct(cr["scores_k12"][f"{o}{mod}"]) for o in ("add", "sub", "max", "min")}
                lm, fm = (ls["add"] + ls["sub"]) / 2, (fs["add"] + fs["sub"]) / 2
                ds.append(lm - fm)
                f3 = st["exposure_rows_tokens"].get(f"joint/{part}", [0])[0] + (life(arm, seed)[5]["exposure_rows_tokens"]["joint/F3"][0] if stage == 6 else 0)
                rows.append(dict(comparison=name, life_arm=arm, fresh_rule=rule, seed=seed, life_stage=stage, life_items_of_family=f3,
                                 fresh_row=row, fresh_items=row * 64, life=lm, fresh=fm, difference=lm - fm,
                                 life_add_sub_max_min=ls, fresh_add_sub_max_min=fs))
            verdicts[f"{name} | life {arm} vs fresh {rule}"] = dict(differences=ds, verdict=verdict(ds))
    curves = {f"{p}_{r}_{s}": [dict(step=c["step"], mean=c["mean_add_sub"], **{k: v["exact"] for k, v in c["scores_k12"].items()})
                               for c in json.loads((E37 / f"e37_{p}_{r}_{s}.json").read_text())["curve"]]
              for p in ("F3", "F1") for r in ("S", "B") for s in SEEDS}
    cpu = {f"{p}_{r}_{s}": json.loads((E37 / f"e37_{p}_{r}_{s}.json").read_text())["cpu_total"] for p in ("F3", "F1") for r in ("S", "B") for s in SEEDS}
    (E37 / "E37_TABLE.json").write_text(json.dumps(dict(rows=rows, verdicts=verdicts, fresh_curves=curves, cpu_seconds=cpu), indent=1))
    md = ["| Comparison | Life | Fresh rule | Seed | Life items of family | Fresh items | Life | Fresh | Life − fresh | Life max · min | Fresh max · min |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['comparison']} | {r['life_arm']} | {r['fresh_rule']} | {r['seed']} | {r['life_items_of_family']:,} | {r['fresh_items']:,} | "
                  f"{r['life']:.1f} | {r['fresh']:.1f} | **{r['difference']:+.1f}** | {r['life_add_sub_max_min']['max']:.1f} · {r['life_add_sub_max_min']['min']:.1f} | "
                  f"{r['fresh_add_sub_max_min']['max']:.1f} · {r['fresh_add_sub_max_min']['min']:.1f} |")
    md += ["", "| Comparison and rule | Differences (73129, 93683, 116089) | Verdict |", "|---|---|---|"]
    for k, v in verdicts.items():
        md.append(f"| {k} | {', '.join(f'{d:+.1f}' for d in v['differences'])} | **{v['verdict']}** |")
    md += ["", "Fresh curves (mean of add and sub at k = 12, %):", "", "| Run | " + " | ".join("row" for _ in range(3)) + " |", "|---|---|---|---|"]
    for k, c in curves.items():
        md.append(f"| {k} | " + " | ".join(f"{x['step']}: {x['mean']:.1f}" for x in c) + " |")
    (E37 / "E37_TABLE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
