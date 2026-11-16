"""E43 reader: curves, the condition, the reading per rule, the facts and P43.1-P43.9, from the receipts only (no reading beyond the brief's).

    python3 e43_read.py      # writes exp45/e43/E43_READ.json and E43_READ_TABLES.md
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / "exp45" / "e43"; CK = D / "checks"
ARMS = ("conflict", "conflict-decay", "unrelated", "unrelated-decay")
RULES, SEEDS = ("S", "B"), (73129, 93683, 116089)
pct = lambda d: 100 * d["exact"] / d["n"]


def view(r):
    s1, s2, s3 = r["score1_first_family_all_pairs_k0"], r["score2_first_family_heldout_k12"], r["score3_new_task"]
    v = dict(updates=r["updates"], reduction=s1["reduction_score"], red_add=s1["add89"]["reduction"]["pct"], red_sub=s1["sub89"]["reduction"]["pct"],
             other=(s1["add89"]["other"]["pct"] + s1["sub89"]["other"]["pct"]) / 2, max=s1["max89"]["all"]["pct"], min=s1["min89"]["all"]["pct"],
             held={n: pct(x) for n, x in s2.items()}, new={n: pct(x) for n, x in s3.items()}, state=r["score6_state_change"]["ratio"],
             move={g: (x["rel_diff"], x["cosine"]) for g, x in r["score4_movement"]["groups"].items()})
    return v


def load():
    runs = {}
    for a in ARMS:
        for r in RULES:
            for s in SEEDS:
                rec = json.loads((D / f"e43_{a}_{r}_{s}.json").read_text()); assert rec.get("complete"), (a, r, s)
                rows = [view(x) for x in rec["rows"]]
                ext = D / f"e43_{a}_{r}_{s}_ext16384.json"
                if ext.exists():
                    e = json.loads(ext.read_text()); assert e.get("complete"); rows += [view(x) for x in e["rows"]]
                runs[a, r, s] = {v["updates"]: v for v in rows}
    return runs


def share(c, u, key="reduction"):
    return 100 * c[u][key] / c[0][key] if c[0][key] else None


def main():
    runs = load()
    decay = {(r, s): json.loads((CK / f"e43_check_score5_decay_{r}_{s}.json").read_text())["result"] for r in RULES for s in SEEDS}
    dec_red = {(r, s): {int(t): v["score1"]["reduction_score"] for t, v in d["rows"].items()} for (r, s), d in decay.items()}
    cond, reading, ud, facts = {}, {}, {}, {}
    for r in RULES:
        rowsel = None; tried = {}
        for u in (4096, 8192, 16384):
            if all(u in runs["unrelated", r, s] for s in SEEDS):
                taught = [runs["unrelated", r, s][u]["new"]["taught"] for s in SEEDS]
                held = [runs["unrelated", r, s][u]["new"]["held"] for s in SEEDS]
                ok = sum(t >= 80 for t in taught) >= 2
                tried[u] = dict(taught=taught, held=held, holds=ok)
                if ok and rowsel is None:
                    rowsel = u
        cond[r] = dict(rows=tried, reading_row=rowsel)
        if rowsel is None:
            reading[r] = dict(reading="the new task was not learned; nothing is drawn about retention", shares=None)
        else:
            sh = [share(runs["unrelated", r, s], rowsel) for s in SEEDS]
            kept, lost = sum(x >= 80 for x in sh), sum(x <= 20 for x in sh)
            reading[r] = dict(row=rowsel, shares=sh, reading="Kept" if kept >= 2 else "Lost" if lost >= 2 else "Partly kept")
        # unrelated-decay against decay alone (points of reduction score) at 256, 1,024, 2,048
        diffs = {s: {u: runs["unrelated-decay", r, s][u]["reduction"] - dec_red[r, s][u] for u in (256, 1024, 2048)} for s in SEEDS}
        up = sum(d[1024] >= 15 for d in diffs.values()) >= 2
        flat = sum(all(abs(x) < 15 for x in d.values()) for d in diffs.values()) >= 2
        ud[r] = dict(diffs=diffs, decay_alone={s: dec_red[r, s] for s in SEEDS},
                     reading="held up" if up else "not held up" if flat else "neither: differences given",
                     note="under the standing decay")
        f = {}
        for a in ARMS:
            for s in SEEDS:
                c = runs[a, r, s]; rr = reading[r].get("row") or 4096
                f[f"{a}_{s}"] = dict(share_reduction={u: share(c, u) for u in (64, 256, 1024, rr)},
                                     share_other=share(c, rr, "other"), share_max=share(c, rr, "max"), share_min=share(c, rr, "min"),
                                     movement_4096={g: dict(rel_diff=v[0], cosine=v[1]) for g, v in c[4096]["move"].items()},
                                     state_change_4096=c[4096]["state"], reduction_1024=c[1024]["reduction"])
        facts[r] = f
    check = lambda n: json.loads((CK / f"e43_check_{n}.json").read_text())["result"]["pass"]
    two = lambda xs: sum(bool(x) for x in xs) >= 2
    P = {}
    P["P43.1"] = (check("1_conflict_is_e38"), "check 1 passes under both rules", "85%")
    P["P43.2"] = (all(two(runs["unrelated", r, s][4096]["new"]["held"] >= 80 for s in SEEDS) for r in RULES),
                  "the unrelated task generalises (held-out >= 80 at 4,096, two of three) under each rule", "60%")
    P["P43.3"] = (reading["S"]["reading"], "unrelated, sweep: kept 25 / partly 45 / lost 30", "")
    P["P43.4"] = (all(two(runs["unrelated", r, s][1024]["reduction"] - runs["conflict", r, s][1024]["reduction"] >= 30 for s in SEEDS) for r in RULES),
                  "unrelated's reduction score >= 30 above conflict's at 1,024, two of three, each rule", "75%")
    P["P43.5"] = (ud["S"]["reading"] == "held up", "unrelated-decay >= 15 above decay alone at 1,024, two of three, sweep", "30%")
    P["P43.9"] = (all(cond[r]["rows"].get(4096, {}).get("holds") for r in RULES), "the condition holds at 4,096 under each rule", "75%")
    P["P43.6"] = (reading["S"]["reading"] == reading["B"]["reading"], "the reading is the same under both rules", "75%")
    rrS = reading["S"].get("row") or 4096
    P["P43.7"] = (two(share(runs["unrelated", "S", s], rrS, "max") > share(runs["unrelated", "S", s], rrS) and
                      share(runs["unrelated", "S", s], rrS, "min") > share(runs["unrelated", "S", s], rrS) for s in SEEDS),
                  "under unrelated, max and min keep a larger share than the reduction pairs, two of three, sweep", "60%")
    P["P43.8"] = (all(two(runs["unrelated", r, s][4096]["move"]["recurrent matrix"][0] < runs["conflict", r, s][4096]["move"]["recurrent matrix"][0]
                          for s in SEEDS) for r in RULES),
                  "the recurrent matrix moves less under unrelated than conflict at 4,096 (relative norm of the difference), two of three, each rule", "60%")
    out = dict(condition=cond, reading=reading, unrelated_decay_vs_decay_alone=ud, facts=facts,
               predictions={k: dict(outcome=v[0], what=v[1], chance=v[2]) for k, v in P.items()},
               curves={f"{a}_{r}_{s}": list(c.values()) for (a, r, s), c in runs.items()})
    (D / "E43_READ.json").write_text(json.dumps(out, indent=1))
    f1 = lambda x: f"{x:.1f}"
    L = ["# E43 read tables, from the receipts", "",
         "Score 1: reduction = mean of add89 and sub89 on their reduction pairs (all pairs, k = 0); other = the same on the pairs without reduction; "
         "max, min on all 7,921 pairs. Score 2: held-out k = 12, 300 each. Score 3: the new task (unrelated: taught / held-out panels, 300 each; "
         "conflict: add97 / sub97 held-out k = 12, 400 each). Score 6: mean |change| of the state at '=' over the mean |state| at row 0. "
         "Score 4: recurrent matrix (weight_hh and bias_hh) relative norm of the difference from birth / cosine. Percent exact.", ""]
    for r in RULES:
        for s in SEEDS:
            L += [f"## rule {r}, seed {s}", "", "| arm | row | reduction | other | max | min | held add/sub/max/min | new | state | recurrent rel / cos |",
                  "|---|---:|---:|---:|---:|---:|---|---|---:|---|"]
            for a in ARMS:
                for u, v in runs[a, r, s].items():
                    L.append(f"| {a} | {u} | {f1(v['reduction'])} | {f1(v['other'])} | {f1(v['max'])} | {f1(v['min'])} | "
                             + "/".join(f1(x) for x in v["held"].values()) + " | " + "/".join(f1(x) for x in v["new"].values())
                             + f" | {v['state']:.3f} | {v['move']['recurrent matrix'][0]:.3f} / {v['move']['recurrent matrix'][1]:.3f} |")
            L.append("")
        L += [f"## decay alone, rule {r} (reduction score, points)", "", "| seed | " + " | ".join(str(t) for t in (0, 64, 256, 1024, 2048, 4096)) + " |", "|---|" + "---:|" * 6]
        for s in SEEDS:
            L.append(f"| {s} | " + " | ".join(f1(dec_red[r, s][t]) for t in (0, 64, 256, 1024, 2048, 4096)) + " |")
        L.append("")
    (D / "E43_READ_TABLES.md").write_text("\n".join(L))
    print(json.dumps(dict(condition=cond, reading=reading, ud=ud, predictions=out["predictions"]), indent=1))


if __name__ == "__main__":
    main()
