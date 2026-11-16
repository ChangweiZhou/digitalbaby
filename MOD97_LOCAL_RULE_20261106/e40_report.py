"""E40 report: reads exp45/e40/e40_*.json, writes exp45/e40/E40_TABLE.json and exp45/e40/E40_REPORT_TABLES.md. The verdict is the brief's
(gain in the reduction score at row 8,192, lived minus fresh under the same rule, K against fresh S); nothing else is read."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / "exp45" / "e40"
SEEDS = (73129, 93683, 116089)
ROWS = (0, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
RUNS = [("lived", "S"), ("lived", "B"), ("lived", "K"), ("fresh", "S"), ("fresh", "B")]
FRESH = {"S": "S", "B": "B", "K": "S"}
F4N = ("add83", "sub83", "max83", "min83")
E6 = ("add89", "sub89", "add97", "sub97", "add73", "sub73")


def pct(d):
    return 100.0 * d["exact"] / d["n"]


def flat(row):
    s = row["scores"]
    f = {}
    for n in F4N:
        f[f"h_{n}"] = pct(s["held_k12"][n]); f[f"t_{n}"] = pct(s["taught_k12"][n])
    for n in ("add83", "sub83"):
        f[f"{n}_nored"] = pct(s["held_plain_k0"][n]["no_reduction"]); f[f"{n}_red"] = pct(s["held_plain_k0"][n]["reduction"])
    for n in E6:
        f[n] = pct(s["earlier_k12"][n])
    sm = s["summary"]
    f.update(held=sm["held_add_sub_mean"], taught=sm["taught_add_sub_mean"], red=sm["reduction_score"], nored=sm["no_reduction_score"])
    return f


def cat(d):
    return "saving" if d >= 15 else "cost" if d <= -15 else "none"


def main():
    F = {}
    for k, r in RUNS:
        for s in SEEDS:
            rec = json.loads((D / f"e40_{k}_{r}_{s}.json").read_text())
            assert rec["complete"] and [x["updates"] for x in rec["rows"]] == list(ROWS)
            F[(k, r, s)] = {x["updates"]: flat(x) for x in rec["rows"]}
            for u in ROWS:
                F[(k, r, s)][u]["gain"] = F[(k, r, s)][u]["red"] - F[(k, r, s)][0]["red"]
    verdict = {}
    for rule in ("S", "B", "K"):
        per = {}
        for s in SEEDS:
            L, Fr = F[("lived", rule, s)], F[("fresh", FRESH[rule], s)]
            per[s] = {u: dict(lived_gain=L[u]["gain"], fresh_gain=Fr[u]["gain"], diff=L[u]["gain"] - Fr[u]["gain"]) for u in (2048, 4096, 8192)}
            per[s]["category_8192"] = cat(per[s][8192]["diff"])
        cats = [per[s]["category_8192"] for s in SEEDS]
        v = next((c for c in ("saving", "cost", "none") if cats.count(c) >= 2), "mixed")
        verdict[rule] = dict(fresh=FRESH[rule], seeds=per, categories=cats, verdict=v)

    def first90(key):
        return next((u for u in ROWS if F[key][u]["taught"] > 90), None)

    facts = {}
    for key in F:
        c = F[key]; f9 = first90(key)
        best = max(ROWS, key=lambda u: (c[u]["red"], -u))
        facts["_".join(map(str, key))] = dict(first_taught_over_90=f9, held_at_that_row=c[f9]["held"] if f9 is not None else None,
                                              held_0=c[0]["held"], held_8192=c[8192]["held"], nored_0=c[0]["nored"], nored_8192=c[8192]["nored"],
                                              red_8192=c[8192]["red"], best_red=c[best]["red"], best_red_row=best,
                                              earlier_fall_8192={n: c[0][n] - c[8192][n] for n in E6})
    # predictions
    fr = [k for k in F if k[0] == "fresh"]
    p1_mean = all(F[k][u]["held"] <= 30 for k in fr for u in ROWS)
    p1_each = all(F[k][u]["h_add83"] <= 30 and F[k][u]["h_sub83"] <= 30 for k in fr for u in ROWS)
    max_fresh = max(max(F[k][u]["h_add83"], F[k][u]["h_sub83"]) for k in fr for u in ROWS)
    max_fresh_mean = max(F[k][u]["held"] for k in fr for u in ROWS)
    p2 = sum(F[("lived", "S", s)][8192]["held"] >= 60 for s in SEEDS)
    p4 = {}
    for rule in ("S", "B", "K"):
        n = 0
        for s in SEEDS:
            a, b = first90(("lived", rule, s)), first90(("fresh", FRESH[rule], s))
            n += a is not None and (b is None or a < b)
        p4[rule] = n
    p6 = sum(F[("lived", "S", s)][8192]["red"] <= F[("lived", "S", s)][8192]["nored"] - 20 for s in SEEDS)
    p7 = max(F[k][8192]["gain"] for k in fr)
    preds = {
        "P40.1": dict(chance="80%", text="Every fresh run is at or below 30 on held-out add and sub at every row",
                      held=p1_mean and p1_each, detail=f"mean of add and sub: max {max_fresh_mean:.1f}; add and sub each: max {max_fresh:.1f}"),
        "P40.2": dict(chance="45%", text="Lived sweep runs at or above 60 on the held-out mean at 12 letters at row 8,192 in two of three seeds",
                      held=p2 >= 2, detail=f"{p2}/3"),
        "P40.3": dict(chance="55%", text="A saving, on the gain in the reduction score, under each of the three rules",
                      held=all(verdict[r]["verdict"] == "saving" for r in verdict), detail=", ".join(f"{r} {verdict[r]['verdict']}" for r in verdict)),
        "P40.4": dict(chance="65%", text="Lived runs pass 90 on taught pairs at an earlier row than their fresh references, two of three seeds per rule",
                      held=all(v >= 2 for v in p4.values()), detail=", ".join(f"{r} {v}/3" for r, v in p4.items())),
        "P40.5": dict(chance="20%", text="A verdict that differs between the sweep and backpropagation",
                      held=verdict["S"]["verdict"] != verdict["B"]["verdict"], detail=f"S {verdict['S']['verdict']}, B {verdict['B']['verdict']}"),
        "P40.6": dict(chance="50%", text="Row 8,192: lived sweep reduction pairs at least 20 below their no-reduction pairs in two of three seeds",
                      held=p6 >= 2, detail=f"{p6}/3"),
        "P40.7": dict(chance="75%", text="Every fresh run's gain at row 8,192 is at most 25", held=p7 <= 25, detail=f"largest {p7:.1f}"),
    }
    (D / "E40_TABLE.json").write_text(json.dumps(dict(verdict=verdict, facts=facts, predictions=preds,
                                                      curves={"_".join(map(str, k)): v for k, v in F.items()}), indent=1, default=str))
    # markdown
    L = ["### Curves (whole answers right with the end symbol, %)\n",
         "Score 1 held-out at 12 letters (300 each); score 2 taught at 12 letters (300 each); score 3 every held-out add/sub pair at 0 letters, "
         "without / with the reduction at 83; score 4 earlier families at 12 letters (reported, not read). `held` = mean of held-out add and sub; "
         "`red` = reduction score (mean of add and sub reduction pairs); `gain` = red minus the run's own red at row 0.\n"]
    c1 = ["h_add83", "h_sub83", "h_max83", "h_min83"]; c2 = ["t_add83", "t_sub83", "t_max83", "t_min83"]
    c3 = ["add83_nored", "add83_red", "sub83_nored", "sub83_red"]; c4 = list(E6)
    hdr = ("| row | held add | sub | max | min | taught add | sub | max | min | add no-red | add red | sub no-red | sub red | "
           + " | ".join(c4) + " | held | red | gain |")
    for k, r in RUNS:
        for s in SEEDS:
            L += [f"\n**{k} {r} {s}**\n", hdr, "|" + "---|" * (1 + 12 + 6 + 3)]
            for u in ROWS:
                f = F[(k, r, s)][u]
                L.append(f"| {u} | " + " | ".join(f"{f[c]:.1f}" for c in c1 + c2 + c3 + c4) + f" | {f['held']:.1f} | {f['red']:.1f} | {f['gain']:+.1f} |")
    L += ["\n### Gains and verdict per rule\n",
          "| Rule | Fresh | Seed | gain lived @2048 | fresh | diff | gain lived @4096 | fresh | diff | **gain lived @8192** | **fresh** | **diff** | category |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rule, v in verdict.items():
        for s in SEEDS:
            p = v["seeds"][s]
            L.append(f"| {rule} | {v['fresh']} | {s} | " + " | ".join(f"{p[u]['lived_gain']:+.1f} | {p[u]['fresh_gain']:+.1f} | {p[u]['diff']:+.1f}" for u in (2048, 4096, 8192))
                     + f" | {p['category_8192']} |")
    L += ["\n| Rule | categories at 8,192 (73129, 93683, 116089) | **Verdict** |", "|---|---|---|"]
    for rule, v in verdict.items():
        L.append(f"| {rule} | {', '.join(v['categories'])} | **{v['verdict']}** |")
    L += ["\n### Facts (not marks)\n",
          "| Run | first row taught > 90 | held there | held @0 | held @8192 | no-red @0 | no-red @8192 | red @8192 | best red (row) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, r in RUNS:
        for s in SEEDS:
            x = facts[f"{k}_{r}_{s}"]
            f9 = "taught never passed 90" if x["first_taught_over_90"] is None else x["first_taught_over_90"]
            hh = "—" if x["held_at_that_row"] is None else f"{x['held_at_that_row']:.1f}"
            L.append(f"| {k} {r} {s} | {f9} | {hh} | {x['held_0']:.1f} | {x['held_8192']:.1f} | {x['nored_0']:.1f} | {x['nored_8192']:.1f} | {x['red_8192']:.1f} | {x['best_red']:.1f} ({x['best_red_row']}) |")
    L += ["\nLived runs' held-out level at row 8,192 beside E35's fresh figure of 19.6 (arm G, backpropagation, mod 89, f = 0.10, a full birth, seed 73129): "
          + "; ".join(f"{r} {s} {facts[f'lived_{r}_{s}']['held_8192']:.1f}" for r in ("S", "B", "K") for s in SEEDS) + ".\n",
          "\nFall of the earlier families, row 0 minus row 8,192 (points):\n",
          "| Run | " + " | ".join(E6) + " |", "|---|" + "---|" * len(E6)]
    for k, r in RUNS:
        for s in SEEDS:
            x = facts[f"{k}_{r}_{s}"]["earlier_fall_8192"]
            L.append(f"| {k} {r} {s} | " + " | ".join(f"{x[n]:.1f}" for n in E6) + " |")
    L += ["\n### Predictions\n", "| | Prediction | Chance | Outcome | Count |", "|---|---|---|---|---|"]
    for k_, v in preds.items():
        L.append(f"| {k_} | {v['text']} | {v['chance']} | {'held' if v['held'] else 'failed'} | {v['detail']} |")
    (D / "E40_REPORT_TABLES.md").write_text("\n".join(L) + "\n")
    print(json.dumps({r: (v["verdict"], v["categories"]) for r, v in verdict.items()}), json.dumps({k_: v["held"] for k_, v in preds.items()}))


if __name__ == "__main__":
    main()
