"""E38 reader: curves, marks at row 12,288, readings per rule, facts and predictions, from the life receipts only.

    python3 e38_read.py [seed]       # writes exp45/e38/E38_READ.json and E38_READ_TABLES.md
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / "exp45" / "e38"
ARMS = ("zero", "none", "anchor", "anchor10", "rehearse")
RULES = ("S", "B")
END = 12288


def pct(d):
    return 100 * d["exact"] / d["n"]


def row_view(r):
    h, s2, nw = r["birth_heldout_k12"], r["birth_allpairs_k0"], r["new_heldout_k12"]
    return dict(updates=r["updates"], held_add89=pct(h["add89"]), held_sub89=pct(h["sub89"]), held_max89=pct(h["max89"]), held_min89=pct(h["min89"]),
                held_addsub=(pct(h["add89"]) + pct(h["sub89"])) / 2,
                wrap_add=s2["add89"]["wrap"]["pct"], wrap_sub=s2["sub89"]["wrap"]["pct"], nowrap_add=s2["add89"]["nowrap"]["pct"],
                nowrap_sub=s2["sub89"]["nowrap"]["pct"], wrap=(s2["add89"]["wrap"]["pct"] + s2["sub89"]["wrap"]["pct"]) / 2,
                nowrap=(s2["add89"]["nowrap"]["pct"] + s2["sub89"]["nowrap"]["pct"]) / 2,
                new_add97=pct(nw["add97"]), new_sub97=pct(nw["sub97"]), new=(pct(nw["add97"]) + pct(nw["sub97"])) / 2)


def marks(v):
    keeps = v["held_addsub"] >= 80 and v["wrap"] >= 70
    loses = v["wrap"] <= 20
    return dict(keeps=keeps, loses=loses, family="keeps" if keeps else "loses" if loses else "intermediate", acquires=v["new"] >= 85)


def first_below(curve, key, thr):
    return next((v["updates"] for v in curve if v[key] < thr), None)


def main(seed):
    lives, curves, mk, facts = {}, {}, {}, {}
    for a in ARMS:
        for r in RULES:
            rec = json.loads((D / f"e38_{a}_{r}_{seed}.json").read_text())
            assert rec.get("complete"), f"{a} {r}: receipt not complete"
            lives[a, r] = rec
            curves[a, r] = [row_view(x) for x in rec["rows"]]
            end = curves[a, r][-1]; assert end["updates"] == END
            mk[a, r] = dict(marks(end), held_addsub=end["held_addsub"], wrap=end["wrap"], new=end["new"])
            facts[a, r] = dict(first_held_below_80=first_below(curves[a, r], "held_addsub", 80), first_wrap_below_50=first_below(curves[a, r], "wrap", 50),
                               expo=[s.get("exposure_rows_tokens") for s in rec["stages"][1:]],
                               birth_checks=rec["birth_checks"], held_pairs_presented=rec["held_pairs_presented"],
                               decay_target=rec["stages"][0]["settings"]["decay_target"], decay_strength=rec["stages"][0]["settings"]["decay_strength"],
                               stored_birth_values=rec["stages"][0]["settings"]["stored_birth_values"],
                               groups_wd=[g["weight_decay"] for g in rec["stages"][0]["settings"]["optimizer_groups_read_back"]],
                               cpu_total=rec["complete"]["cpu_total"], cpu_scoring=rec["complete"]["cpu_scoring"])
    readings = {}
    for r in RULES:
        rd = dict(zero_loses=mk["zero", r]["loses"], applies=[], notes=[])
        if not mk["zero", r]["loses"]:
            rd["notes"].append("zero does not lose the family by the mark: readings 1 to 5 not read")
        else:
            n = mk["none", r]["family"]
            if n == "keeps":
                rd["applies"].append(1)
            elif n == "loses":
                rd["applies"].append(2)
            else:
                rd["notes"].append("none: intermediate (rows 1, 2 not read)")
            kept = False
            for a in ("anchor", "anchor10"):
                m = mk[a, r]
                if m["keeps"] and m["acquires"]:
                    rd["applies"].append(f"3 ({a})"); kept = True
                elif m["keeps"]:
                    rd["applies"].append(f"4 ({a})"); kept = True
                else:
                    rd["notes"].append(f"{a}: family {m['family']}, acquires {m['acquires']}")
            if not kept:
                rd["applies"].append(5)
        rd["applies"].append(6)
        readings[r] = rd
    # rehearsal's cost to the new family: per update (row 12288 vs zero's row 12288) and per F1 item (vs zero's row 6144, the same F1 count)
    cost = {}
    for r in RULES:
        f1_items = sum(e.get("joint/F1", [0])[0] for e in facts["rehearse", r]["expo"])
        birth_items = sum(e.get("joint/other", [0])[0] for e in facts["rehearse", r]["expo"])
        z = {v["updates"]: v for v in curves["zero", r]}
        cost[r] = dict(rehearse_new_12288=mk["rehearse", r]["new"], zero_new_12288=z[END]["new"], zero_new_6144=z[6144]["new"],
                       per_update=mk["rehearse", r]["new"] - z[END]["new"], per_F1_item=mk["rehearse", r]["new"] - z[6144]["new"],
                       rehearse_F1_items=f1_items, rehearse_birth_items=birth_items, zero_F1_items_at_6144=393216,
                       per_block=[dict(F1=e.get("joint/F1", [0])[0], birth=e.get("joint/other", [0])[0]) for e in facts["rehearse", r]["expo"]])
    paired = {a: {k: mk[a, "S"][k] - mk[a, "B"][k] for k in ("held_addsub", "wrap", "new")} for a in ARMS}
    S = lambda a: mk[a, "S"]
    rset = lambda r: set(map(str, readings[r]["applies"]))
    preds = {
        "P38.1": (all(json.loads((D / f"e38_check1_{r}_{seed}.json").read_text())["result"]["pass"] for r in RULES), "check 1 passes for both rules", 90),
        "P38.2": (S("none")["keeps"], "none keeps (sweep)", 15),
        "P38.3": (S("anchor")["keeps"], "anchor keeps (sweep)", 25),
        "P38.4": (S("anchor10")["keeps"], "anchor10 keeps (sweep)", 55),
        "P38.5": (S("anchor10")["acquires"], "anchor10 acquires (sweep)", 55),
        "P38.6": (any(S(a)["keeps"] and S(a)["acquires"] for a in ("anchor", "anchor10")), "some anchor keeps and acquires (sweep)", 35),
        "P38.7": (S("rehearse")["keeps"] and S("rehearse")["acquires"], "rehearse keeps and acquires (sweep)", 70),
        "P38.8": (rset("S") != rset("B"), "a reading differs between the sweep and backpropagation", 15),
    }
    out = dict(seed=seed, marks={f"{a}_{r}": v for (a, r), v in mk.items()}, readings=readings, facts={f"{a}_{r}": v for (a, r), v in facts.items()},
               rehearsal_cost=cost, paired_S_minus_B_12288=paired, predictions={k: dict(outcome=v[0], what=v[1], chance=v[2]) for k, v in preds.items()},
               curves={f"{a}_{r}": v for (a, r), v in curves.items()})
    (D / "E38_READ.json").write_text(json.dumps(out, indent=1))
    f = lambda x: f"{x:.1f}"
    L = ["# E38 read tables (seed %d), from the life receipts" % seed, "",
         "Columns: held = held-out k=12 (300 pairs each) add89/sub89 and their mean, max89, min89; wrap/nowrap = score 2 (all 7,921 pairs, k=0), "
         "mean of add89 and sub89; new = held-out k=12 (400 each) add97/sub97 and their mean. Percent exact (answer bytes + EOS).", ""]
    for a in ARMS:
        for r in RULES:
            L += [f"## {a}, rule {r}", "", "| row | add89 | sub89 | held mean | max89 | min89 | wrap add | wrap sub | wrap | no-wrap | add97 | sub97 | new |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for v in curves[a, r]:
                L.append("| " + " | ".join([str(v["updates"])] + [f(v[k]) for k in ("held_add89", "held_sub89", "held_addsub", "held_max89", "held_min89", "wrap_add",
                                                                                       "wrap_sub", "wrap", "nowrap", "new_add97", "new_sub97", "new")]) + " |")
            L.append("")
    (D / "E38_READ_TABLES.md").write_text("\n".join(L))
    print(json.dumps(dict(marks=out["marks"], readings=readings, cost=cost, paired=paired, predictions=out["predictions"],
                          facts={k: {x: v[x] for x in ("first_held_below_80", "first_wrap_below_50")} for k, v in out["facts"].items()}), indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 73129)
