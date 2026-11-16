"""E43 addendum 2 reader: curves, readings (the addendum's tables only) and P43.15-P43.18, from the receipts.

    python3 e43_addendum2_read.py      # writes exp45/e43/addendum2/E43_ADDENDUM2_READ.json and E43_ADDENDUM2_TABLES.md
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / "exp45" / "e43" / "addendum2"; D43 = HERE / "exp45" / "e43"
RULES, SEEDS, ROWS = ("S", "B"), (73129, 93683, 116089), (0, 64, 256, 1024, 2048, 4096)
REH = {"unrelated-rehearse": "unrelated", "unrelated-rehearse-decay": "unrelated-decay"}
pc = lambda d: 100 * d["exact"] / d["n"]


def view(r, batch=64):
    s1 = r["score1_first_family_all_pairs_k0"]
    return dict(updates=r["updates"], reduction=s1["reduction_score"], other=(s1["add89"]["other"]["pct"] + s1["sub89"]["other"]["pct"]) / 2,
                max=s1["max89"]["all"]["pct"], min=s1["min89"]["all"]["pct"], held={n: pc(v) for n, v in r["score2_first_family_heldout_k12"].items()},
                taught=pc(r["score3_new_task"]["taught"]), heldnew=pc(r["score3_new_task"]["held"]), state=r["score6_state_change"]["ratio"],
                unrelated_items=r.get("items_so_far", {}).get("unrelated", r["updates"] * batch),
                first_items=r.get("items_so_far", {}).get("first_family", 0),
                recurrent=r["score4_movement"]["groups"]["recurrent matrix"])


def load(path):
    x = json.loads(path.read_text()); assert x.get("complete"), path
    return {v["updates"]: view(v) for v in x["rows"]}


def first80(c):
    return next(((u, c[u]["unrelated_items"]) for u in ROWS if c[u]["taught"] >= 80), (None, None))


def main():
    reh, base = {}, {}
    for a, b in REH.items():
        for r in RULES:
            for s in SEEDS:
                reh[a, r, s] = load(D / f"e43_{a}_{r}_{s}.json")
                base[b, r, s] = load(D43 / f"e43_{b}_{r}_{s}.json")
    readings = {}
    for a, b in REH.items():
        for r in RULES:
            per = {}
            for s in SEEDS:
                c = reh[a, r, s]; sh = 100 * c[4096]["reduction"] / c[0]["reduction"]
                u_r, i_r = first80(c); u_b, i_b = first80(base[b, r, s])
                per[s] = dict(share_4096=sh, taught_4096=c[4096]["taught"], heldnew_4096=c[4096]["heldnew"],
                              first80_update=u_r, first80_unrelated_items=i_r, base_arm=b, base_first80_update=u_b, base_first80_items=i_b,
                              items_4096=dict(unrelated=c[4096]["unrelated_items"], first_family=c[4096]["first_items"]),
                              share={u: 100 * c[u]["reduction"] / c[0]["reduction"] for u in ROWS})
            n = lambda f: sum(f(v) for v in per.values())
            if n(lambda v: v["share_4096"] >= 80 and v["taught_4096"] >= 80) >= 2:
                rd = "kept at no cost"
            elif n(lambda v: v["share_4096"] >= 80 and v["taught_4096"] < 80) >= 2:
                rd = "kept at a cost"
            elif n(lambda v: v["share_4096"] <= 20) >= 2:
                rd = "not kept"
            else:
                rd = "figures given"
            readings[f"{a}_{r}"] = dict(reading=rd, per_seed=per)
    e42 = {}
    for cfg in ("F10", "P8"):
        for r in RULES:
            c = load(D / f"e43_unrelated-e42{cfg}_{r}_73129.json")
            x = json.loads((D / f"e43_unrelated-e42{cfg}_{r}_73129.json").read_text())
            sh = 100 * c[4096]["reduction"] / c[0]["reduction"]
            cond = c[4096]["taught"] >= 80
            rd = ("Kept" if sh >= 80 else "Lost" if sh <= 20 else "Partly kept") if cond else "condition not met at 4,096"
            e42[f"{cfg}_{r}"] = dict(reading=rd, condition=cond, share_4096=sh, taught_4096=c[4096]["taught"], heldnew_4096=c[4096]["heldnew"],
                                     share={u: 100 * c[u]["reduction"] / c[0]["reduction"] for u in ROWS},
                                     row0_check=x["settings"]["row0_against_e42_final_row"], curve=c)
    two = lambda xs: sum(bool(v) for v in xs) >= 2
    P = {"P43.15": (readings["unrelated-rehearse_S"]["reading"] == "kept at no cost", "unrelated-rehearse, sweep: kept at no cost", "60%"),
         "P43.16": (readings["unrelated-rehearse-decay_S"]["reading"] == "kept at no cost", "unrelated-rehearse-decay, sweep: kept at no cost", "65%"),
         "P43.17": (all(two(v["first80_update"] is not None and v["base_first80_update"] is not None and v["first80_update"] <= 2 * v["base_first80_update"]
                            for v in readings[f"unrelated-rehearse_{r}"]["per_seed"].values()) for r in RULES),
                    "with rehearsal the taught panel reaches 80% no more than twice as many updates later than without (rehearse vs unrelated), two of three, each rule", "60%"),
         "P43.18": (all(v["reading"] == "Lost" for v in e42.values()), "E42's F10 and P8: lost in all four runs", "85%")}
    P17_decay = all(two(v["first80_update"] is not None and v["base_first80_update"] is not None and v["first80_update"] <= 2 * v["base_first80_update"]
                        for v in readings[f"unrelated-rehearse-decay_{r}"]["per_seed"].values()) for r in RULES)
    out = dict(readings=readings, e42=e42, P43_17_for_the_decay_pair=P17_decay,
               predictions={k: dict(outcome=v[0], what=v[1], chance=v[2]) for k, v in P.items()},
               curves={f"{a}_{r}_{s}": list(c.values()) for (a, r, s), c in reh.items()})
    (D / "E43_ADDENDUM2_READ.json").write_text(json.dumps(out, indent=1, default=str))
    f = lambda x: f"{x:.1f}"
    L = ["# E43 addendum 2 tables, from the receipts", "",
         "Cells: reduction / pairs without reduction / max / min (score 1, all pairs, k = 0) · unrelated task taught / held-out · items so far "
         "(unrelated / first family). Percent exact.", ""]
    hdr = "| run | " + " | ".join(str(u) for u in ROWS) + " |"
    for a in REH:
        L += [f"## {a}", "", hdr, "|---|" + "---|" * 6]
        for r in RULES:
            for s in SEEDS:
                c = reh[a, r, s]
                L.append(f"| {r} {s} | " + " | ".join(f"{f(c[u]['reduction'])}/{f(c[u]['other'])}/{f(c[u]['max'])}/{f(c[u]['min'])} · {f(c[u]['taught'])}/"
                                                         f"{f(c[u]['heldnew'])} · {c[u]['unrelated_items']}/{c[u]['first_items']}" for u in ROWS) + " |")
        L.append("")
    L += ["## E42's networks (unrelated arm, no decay, seed 73129)", "", hdr, "|---|" + "---|" * 6]
    for k, v in e42.items():
        c = v["curve"]
        L.append(f"| {k} | " + " | ".join(f"{f(c[u]['reduction'])}/{f(c[u]['other'])}/{f(c[u]['max'])}/{f(c[u]['min'])} · {f(c[u]['taught'])}/{f(c[u]['heldnew'])}"
                                          for u in ROWS) + " |")
    (D / "E43_ADDENDUM2_TABLES.md").write_text("\n".join(L) + "\n")
    print(json.dumps(dict(readings={k: v["reading"] for k, v in readings.items()}, e42={k: (v["reading"], round(v["share_4096"], 1), v["taught_4096"]) for k, v in e42.items()},
                          P43_17_decay=P17_decay, predictions=out["predictions"]), indent=1, default=str))


if __name__ == "__main__":
    main()
