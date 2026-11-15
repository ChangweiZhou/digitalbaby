"""E36 report: reads exp45/e36/e36_*.json and the check receipts, writes exp45/e36/E36_TABLE.json and the body tables of E36_REPORT.md
(exp45/e36/E36_REPORT_TABLES.md). The readings are the brief's table, tried in order; nothing else is read into the numbers."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / "exp45" / "e36"
SEEDS = (73129, 93683, 116089)
ROWS = (0, 16, 32, 64, 128, 256, 512, 1024)
RUNS = [("reminder", "S"), ("reminder", "B"), ("reminder", "K"), ("reference", "S"), ("reference", "B")]
REF = {"S": "S", "B": "B", "K": "S"}


def pct(d):
    return 100.0 * d["exact"] / d["n"]


def load():
    out = {}
    for k, r in RUNS:
        for s in SEEDS:
            rec = json.loads((D / f"e36_{k}_{r}_{s}.json").read_text())
            assert rec["complete"] and [x["updates"] for x in rec["rows"]] == list(ROWS)
            out[(k, r, s)] = {x["updates"]: x for x in rec["rows"]}
    return out


def flat(row):
    sc = row["scores"]
    f = {n: pct(v) for n, v in sc["birth_held_k12"].items()}
    for op in ("add89", "sub89"):
        for w in ("wrap", "nowrap"):
            f[f"{op}_{w}"] = pct(sc["birth_all_pairs_k0"][op][w])
    f.update({n: pct(v) for n, v in sc["life_families"].items()})
    f.update(held=sc["summary"]["held_add_sub_mean"], wrap=sc["summary"]["wrap_score"], nowrap=sc["summary"]["nowrap_score"],
             restored=sc["summary"]["restored"])
    return f


def main():
    runs = load()
    F = {key: {u: flat(r) for u, r in rows.items()} for key, rows in runs.items()}
    readings = {}
    for arm in ("S", "B", "K"):
        per = {}
        for s in SEEDS:
            a, ref = F[("reminder", arm, s)], F[("reference", REF[arm], s)]
            dw = a[1024]["wrap"] - ref[1024]["wrap"]
            per[s] = dict(held_256=a[256]["held"], restored_256=a[256]["restored"], wrap_1024=a[1024]["wrap"], ref_wrap_1024=ref[1024]["wrap"],
                          wrap_minus_ref_1024=dw, p1=a[256]["restored"], p2=dw >= 30, p3=abs(dw) < 15,
                          first_restored_row=next((u for u in ROWS if a[u]["restored"]), None))
        counts = {p: sum(per[s][p] for s in SEEDS) for p in ("p1", "p2", "p3")}
        reading = next((i for i, p in ((1, "p1"), (2, "p2"), (3, "p3")) if counts[p] >= 2), 4)
        readings[arm] = dict(seeds=per, counts=counts, row=reading, reference=REF[arm])
    # predictions
    S = readings["S"]["seeds"]
    by = lambda arm, lim: sum(any(F[("reminder", arm, s)][u]["restored"] for u in ROWS if u <= lim) for s in SEEDS)
    fall = lambda arm, s, n, u: F[("reminder", arm, s)][0][n] - F[("reminder", arm, s)][u][n]
    p5 = {s: dict(add97=fall("S", s, "add97", 1024), sub97=fall("S", s, "sub97", 1024)) for s in SEEDS}
    preds = {
        "P36.1": dict(text="Sweep reminders (arm S): restored by row 256 in two of three seeds", seeds_restored_by_256=by("S", 256), held=by("S", 256) >= 2),
        "P36.2": dict(text="Sweep reminders (arm S): restored by row 1,024 in two of three seeds", seeds_restored_by_1024=by("S", 1024), held=by("S", 1024) >= 2),
        "P36.3": dict(text="Every arm: wrap score >= 30 above its reference's at row 1,024 in two of three seeds",
                      per_arm={a: readings[a]["counts"]["p2"] for a in readings}, held=all(readings[a]["counts"]["p2"] >= 2 for a in readings)),
        "P36.4": dict(text="The table row differs between the sweep and backpropagation", rows=dict(S=readings["S"]["row"], B=readings["B"]["row"]),
                      held=readings["S"]["row"] != readings["B"]["row"]),
        "P36.5": dict(text="Arm S: at row 1,024 add97 and sub97 (k 12) each fallen >= 30 points from row 0, in two of three seeds", falls=p5,
                      seeds=sum(v["add97"] >= 30 and v["sub97"] >= 30 for v in p5.values()),
                      held=sum(v["add97"] >= 30 and v["sub97"] >= 30 for v in p5.values()) >= 2),
    }
    (D / "E36_TABLE.json").write_text(json.dumps(dict(readings=readings, predictions=preds,
                                                      curves={f"{k}_{r}_{s}": F[(k, r, s)] for k, r, s in F}), indent=1, default=str))
    # markdown
    L = []
    cols = ["add89", "sub89", "max89", "min89", "add89_wrap", "add89_nowrap", "sub89_wrap", "sub89_nowrap", "add97", "sub97", "add73", "sub73", "within"]
    L.append("### Curves (exact whole answers with the end symbol, %)\n")
    L.append("Group 1: birth family held out, k 12 (300 each). Group 2: every add/sub mod-89 pair at k 0, wrap / no-wrap. Group 3: the life's families, k 12 (400 each), and `compose[\"within\"]`. `held` = mean of held-out add89 and sub89 (\"restored\" if >= 85); `wrap` = mean of the add and sub wrap scores.\n")
    for k, r in RUNS:
        for s in SEEDS:
            L.append(f"\n**{k} {r} {s}**\n")
            L.append("| row | " + " | ".join(cols) + " | held | wrap | no-wrap |")
            L.append("|" + "---|" * (len(cols) + 4))
            for u in ROWS:
                f = F[(k, r, s)][u]
                L.append(f"| {u} | " + " | ".join(f"{f[c]:.1f}" for c in cols) + f" | {f['held']:.1f}{' R' if f['restored'] else ''} | {f['wrap']:.1f} | {f['nowrap']:.1f} |")
    L.append("\n### Table row per arm\n")
    L.append("| Arm | Reference | Seed | held @256 | restored @256 | first restored row | wrap @1024 | ref wrap @1024 | difference | pattern 1 | 2 | 3 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a, v in readings.items():
        for s in SEEDS:
            p = v["seeds"][s]
            L.append(f"| {a} | {v['reference']} | {s} | {p['held_256']:.1f} | {p['restored_256']} | {p['first_restored_row']} | {p['wrap_1024']:.1f} | {p['ref_wrap_1024']:.1f} | {p['wrap_minus_ref_1024']:+.1f} | {p['p1']} | {p['p2']} | {p['p3']} |")
    L.append("\n| Arm | seeds meeting pattern 1 / 2 / 3 | Table row |")
    L.append("|---|---|---|")
    for a, v in readings.items():
        L.append(f"| {a} | {v['counts']['p1']} / {v['counts']['p2']} / {v['counts']['p3']} | {v['row']} |")
    L.append("\n### Fall of the life's families (row 0 minus row; points)\n")
    L.append("| Arm | Seed | row | add97 | sub97 | add73 | sub73 | within |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a in ("S", "B", "K"):
        for s in SEEDS:
            for u in (256, 1024):
                L.append(f"| {a} | {s} | {u} | " + " | ".join(f"{fall(a, s, n, u):.1f}" for n in ("add97", "sub97", "add73", "sub73", "within")) + " |")
    L.append("\n### Paired difference, sweep minus backpropagation reminder (points)\n")
    L.append("| Seed | row | held | wrap | no-wrap |")
    L.append("|---|---|---|---|---|")
    for s in SEEDS:
        for u in ROWS:
            a, b = F[("reminder", "S", s)][u], F[("reminder", "B", s)][u]
            L.append(f"| {s} | {u} | {a['held'] - b['held']:+.1f} | {a['wrap'] - b['wrap']:+.1f} | {a['nowrap'] - b['nowrap']:+.1f} |")
    L.append("\n### Predictions\n")
    L.append("| | Prediction | Chance | Outcome | Count |")
    L.append("|---|---|---|---|---|")
    ch = {"P36.1": "30%", "P36.2": "60%", "P36.3": "70%", "P36.4": "15%", "P36.5": "55%"}
    cnt = {"P36.1": f"{preds['P36.1']['seeds_restored_by_256']}/3", "P36.2": f"{preds['P36.2']['seeds_restored_by_1024']}/3",
           "P36.3": ", ".join(f"{a} {c}/3" for a, c in preds["P36.3"]["per_arm"].items()), "P36.4": f"S row {preds['P36.4']['rows']['S']}, B row {preds['P36.4']['rows']['B']}",
           "P36.5": f"{preds['P36.5']['seeds']}/3"}
    for k_, v in preds.items():
        L.append(f"| {k_} | {v['text']} | {ch[k_]} | {'held' if v['held'] else 'failed'} | {cnt[k_]} |")
    (D / "E36_REPORT_TABLES.md").write_text("\n".join(L) + "\n")
    print(json.dumps({a: (v["row"], v["counts"]) for a, v in readings.items()}), json.dumps({k_: v["held"] for k_, v in preds.items()}))


if __name__ == "__main__":
    main()
