"""E51: the readings of Part A and Part B from the receipts, as the brief fixes them (revision 2). Writes exp45/e51/E51_READ_A.json and
E51_TABLES_A.md (part a) or exp45/e51/partB/E51_READ_B.json and E51_TABLES_B.md (part b). Reads receipts only; no torch.

    python3 e51_read.py a <seed>
    python3 e51_read.py b <seed>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e51"
E43 = HERE / "exp45" / "e43"
TASKS, ARMS, RULES = ("unrelated", "conflict"), ("open", "rehearse", "keyed-free", "keyed"), ("S", "B")


def first_learned(curve):
    """The first scored row at which the score is at 80% and stays there at every later row; None if there is none."""
    for i, (u, v) in enumerate(curve):
        if all(w >= 80 for _, w in curve[i:]):
            return u
    return None


def part_a(seed):
    read, md = {}, [f"# E51 Part A, seed {seed}: every run at every row", "",
                    "Columns: updates; the first family's reduction score (%); max and min on all pairs (%); held-out add, sub, max, min at 12 letters (%); "
                    "share of the first family's answers equal to row 0's (all pairs without letters, %); the new task's score on its never-taught items (%); "
                    "the same with the reserve cells held shut (%); reserve cells open by more than 0.1; largest size of a reserve cell's state on the first family's panel.", ""]
    for task in TASKS:
        for rule in RULES:
            for arm in ARMS:
                f = OUT / f"e51_{task}_{arm}_{rule}_{seed}.json"; r = json.loads(f.read_text()); assert r["complete"], f
                rows = r["rows"]; r0, rl = rows[0], rows[-1]; assert rl["updates"] == 4096
                red0, red = r0["score1_first_family_all_pairs_k0"]["reduction_score"], rl["score1_first_family_all_pairs_k0"]["reduction_score"]
                curve = [(x["updates"], x["score3_never_taught"]["pct"]) for x in rows]
                fl = first_learned(curve); shut = rl["score3_never_taught_reserve_shut"]["pct"]
                red64 = [x for x in rows if x["updates"] == 64][0]["score1_first_family_all_pairs_k0"]["reduction_score"]
                cl = rl["wrong_answers_on_reduction_pairs"]; cls = {c: cl["add89"][c] + cl["sub89"][c] for c in cl["add89"]}
                read[f"{task} {rule} {arm}"] = dict(receipt=f.name, reduction_row0=red0, reduction_64=red64, reduction_4096=red, kept=abs(red - red0) <= 2,
                                                    share_equal_row0_4096=rl["share_of_answers_equal_to_row0"]["all_pairs_k0"], new_curve=curve, learned=fl is not None,
                                                    first_row_learned=fl, new_4096=curve[-1][1], new_4096_reserve_shut=shut, learned_in_the_reserve=fl is not None and shut < 20,
                                                    wrong_answers_on_reduction_pairs_4096=cls, reserve_4096=rl["reserve"], tensors_sha256_4096=rl["tensors_sha256"],
                                                    cpu_total=r["complete"]["cpu_total"])
                md += [f"## {task}, rule {rule}, arm `{arm}` ({f.name})", "", "| updates | reduction | max | min | held add | held sub | held max | held min | equal to row 0 | new | new, reserve shut | reserve open | reserve state |",
                       "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
                for x in rows:
                    s = x["score1_first_family_all_pairs_k0"]; h = x["score2_first_family_heldout_k12"]
                    md.append(f"| {x['updates']} | {s['reduction_score']:.1f} | {s['max89']['all']['pct']:.1f} | {s['min89']['all']['pct']:.1f} | "
                              + " | ".join(f"{100 * h[n]['exact'] / h[n]['n']:.1f}" for n in ("add89", "sub89", "max89", "min89"))
                              + f" | {100 * x['share_of_answers_equal_to_row0']['all_pairs_k0']:.2f} | {x['score3_never_taught']['pct']:.1f} | {x['score3_never_taught_reserve_shut']['pct']:.1f} | "
                              f"{x['reserve']['reserve_cells_open_by_more_than_0p1']} | {x['reserve']['largest_reserve_state_on_first_family_panel']:.3g} |")
                md.append("")
    cost = {}
    for task in TASKS:
        for rule in RULES:
            k = read[f"{task} {rule} keyed"]["first_row_learned"]; kf = read[f"{task} {rule} keyed-free"]["first_row_learned"]; o = read[f"{task} {rule} open"]["first_row_learned"]
            cost[f"{task} {rule}"] = dict(keyed=k, keyed_free=kf, open=o, cost_of_the_rule=(k / kf) if k and kf else None, cost_of_the_device=(k / o) if k and o else None)
    summ = ["# E51 Part A, seed " + str(seed) + ": the readings", "", "| task | rule | arm | reduction row 0 | at 64 | at 4,096 | kept | new at 4,096 | first row learned | new, reserve shut | learned in the reserve |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in read.items():
        t, ru, a = k.split(" ")
        summ.append(f"| {t} | {ru} | `{a}` | {v['reduction_row0']:.1f} | {v['reduction_64']:.1f} | {v['reduction_4096']:.1f} | {'yes' if v['kept'] else 'no'} | {v['new_4096']:.1f} | "
                    f"{v['first_row_learned'] if v['learned'] else 'not learned'} | {v['new_4096_reserve_shut']:.1f} | {'yes' if v['learned_in_the_reserve'] else 'no' if v['learned'] else '-'} |")
    summ += ["", "| task | rule | first row learned: `keyed` | `keyed-free` | `open` | cost of the rule (ratio) | cost of the device (ratio) |", "|---|---|---|---|---|---|---|"]
    for k, v in cost.items():
        t, ru = k.split(" ")
        f_ = lambda z: "-" if z is None else f"{z:g}"
        summ.append(f"| {t} | {ru} | {f_(v['keyed'])} | {f_(v['keyed_free'])} | {f_(v['open'])} | {f_(v['cost_of_the_rule'])} | {f_(v['cost_of_the_device'])} |")
    summ += ["", "Wrong answers on the reduction pairs at update 4,096 (add and sub together, 7,832 pairs), by class:", "",
             "| task | rule | arm | end symbol alone | byte of 128 or above | right under modulus 97 | differs by 1 or 10 | other two digits | anything else |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in read.items():
        t, ru, a = k.split(" ")
        summ.append(f"| {t} | {ru} | `{a}` | " + " | ".join(str(n) for n in v["wrong_answers_on_reduction_pairs_4096"].values()) + " |")
    (OUT / "E51_READ_A.json").write_text(json.dumps(dict(experiment="E51", part="A", seed=seed, runs=read, cost=cost), indent=1))
    (OUT / "E51_TABLES_A.md").write_text("\n".join(summ + [""] + md) + "\n")
    print("\n".join(summ))


def part_b(seed):
    B = OUT / "partB"; read = dict(B1={}, B2={}, B3={}); md = [f"# E51 Part B, seed {seed}: the readings", ""]
    # ---- B1
    md += ["## B1. The share of rehearsal", "", "| rule | share | reduction row 0 | at 4,096 | % of row 0 | kept (within 5 points) | lowest probe (%) | unrelated held at 80%: first probe row | unrelated held at 4,096 | source |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for rule in RULES:
        entries = []
        ref0 = json.loads((E43 / f"e43_unrelated_{rule}_{seed}.json").read_text())
        ref2 = json.loads((E43 / "addendum2" / f"e43_unrelated-rehearse_{rule}_{seed}.json").read_text())
        for name, rec, src in (("1/2 (the second addendum)", ref2, ref2.get("tag", "addendum 2")), ):
            a, z = rec["rows"][0], rec["rows"][-1]
            entries.append((2, name, a, z, None, src))
        for k in (4, 8, 16, 32):
            rec = json.loads((B / f"e51_B1_share_{k}_{rule}_{seed}.json").read_text()); assert rec["complete"]
            entries.append((k, f"1/{k}", rec["rows"][0], rec["rows"][-1], rec["probe"], rec["tag"]))
        entries.append((None, "0 (E43 `unrelated`)", ref0["rows"][0], ref0["rows"][-1], None, f"e43_unrelated_{rule}_{seed}"))
        kept = {}
        for k, name, a, z, probe, src in entries:
            r0 = a["score1_first_family_all_pairs_k0"]["reduction_score"]; r4 = z["score1_first_family_all_pairs_k0"]["reduction_score"]; assert z["updates"] == 4096
            low = min(p["first_family_pct"] for p in probe) if probe else None
            f80 = next((p["updates"] for p in probe if p["unrelated_held_pct"] >= 80), None) if probe else None
            un = 100 * z["score3_new_task"]["held"]["exact"] / z["score3_new_task"]["held"]["n"]
            kept[k] = abs(r4 - r0) <= 5
            read["B1"][f"{rule} {name}"] = dict(reduction_row0=r0, reduction_4096=r4, pct_of_row0=100 * r4 / r0, kept=kept[k], lowest_probe=low, unrelated_80_first_probe_row=f80, unrelated_held_4096=un)
            md.append(f"| {rule} | {name} | {r0:.1f} | {r4:.1f} | {100 * r4 / r0:.1f} | {'yes' if kept[k] else 'no'} | {'-' if low is None else f'{low:.1f}'} | {'-' if f80 is None else f80} | {un:.1f} | {src} |")
        least = None
        for k in (2, 4, 8, 16, 32):                        # the smallest share that is kept with every larger share also kept
            if kept[k]:
                least = k
            else:
                break
        read["B1"][f"{rule} least share that keeps"] = None if least is None else f"1/{least}"
        md.append(f"| {rule} | **the least share that keeps** | | | | {'none' if least is None else f'1/{least}'} | | | | |")
    # ---- B2
    md += ["", "## B2. Rehearsal at intervals", ""]
    for rule in RULES:
        rec = json.loads((B / f"e51_B2_intervals_{rule}_{seed}.json").read_text()); assert rec["complete"]
        cy = rec["cycles"]; c1, c8 = cy[0], cy[-1]
        shortens = c8["review_updates"] <= c1["review_updates"] / 2
        more = c8["probe_at_end_of_gap"]["first_family_pct"] - c1["probe_at_end_of_gap"]["first_family_pct"] >= 20
        s0, s1 = rec["rows"][0]["score1_first_family_all_pairs_k0"]["reduction_score"], rec["rows"][-1]["score1_first_family_all_pairs_k0"]["reduction_score"]
        un = lambda r: 100 * r["score3_new_task"]["held"]["exact"] / r["score3_new_task"]["held"]["n"]
        sentence = f"with rehearsal at intervals the first family {'becomes' if shortens else 'does not become'} cheaper to restore, and {'more' if more else 'no more'} of it survives a gap"
        read["B2"][rule] = dict(cycles=cy, review_shortens=shortens, more_is_left=more, sentence=sentence, probe_first_family_pct_at_update_0=rec["settings"]["probe_first_family_pct_at_update_0"],
                                review_goal_pct=rec["settings"]["review_goal_pct"], reduction_row0=s0, reduction_end=s1, unrelated_held_end=un(rec["rows"][-1]), updates=rec["complete"]["updates"])
        md += [f"Rule {rule} ({rec['tag']}): probe at update 0 {rec['settings']['probe_first_family_pct_at_update_0']:.2f}%, the review's goal {rec['settings']['review_goal_pct']:.2f}%; "
               f"E43's reduction score {s0:.1f} at update 0 and {s1:.1f} at the end (update {rec['complete']['updates']}); the unrelated task's held-out score at the end {un(rec['rows'][-1]):.1f}%.", "",
               "| cycle | probe at the end of the gap: first family (%) | unrelated held (%) | review updates | goal reached | probe at the end of the review: first family (%) | unrelated held (%) |", "|---|---|---|---|---|---|---|"]
        for c in cy:
            md.append(f"| {c['cycle']} | {c['probe_at_end_of_gap']['first_family_pct']:.2f} | {c['probe_at_end_of_gap']['unrelated_held_pct']:.1f} | {c['review_updates']} | {'yes' if c['review_reached_goal'] else 'no'} | "
                      f"{c['probe_at_end_of_review']['first_family_pct']:.2f} | {c['probe_at_end_of_review']['unrelated_held_pct']:.1f} |")
        md += ["", f"The review shortens: {'yes' if shortens else 'no'}. More is left: {'yes' if more else 'no'}. Rule {rule}: {sentence}.", ""]
    # ---- B3
    md += ["## B3. Smaller steps on the new task", "", "| rule | rate | updates | first probe row with the unrelated task at 80% | first-family probe there (%) | last probe row with the first family at half of update 0 or more | unrelated held there (%) | reduction at the end | unrelated held at the end |",
           "|---|---|---|---|---|---|---|---|---|"]
    for rule in RULES:
        at80 = {}
        for key in ("0.001", "0.0003", "0.0001"):
            rec = json.loads((B / f"e51_B3_steps_{key}_{rule}_{seed}.json").read_text()); assert rec["complete"]
            pr = rec["probe"]; p0 = pr[0]["first_family_pct"]
            f80 = next((p for p in pr if p["unrelated_held_pct"] >= 80), None)
            half = [p for p in pr if p["first_family_pct"] >= p0 / 2]; lh = half[-1] if half else None
            z = rec["rows"][-1]
            at80[key] = f80["first_family_pct"] if f80 else None
            read["B3"][f"{rule} {key}"] = dict(updates=rec["complete"]["updates"], probe_row0=p0, unrelated_80_first_probe_row=f80["updates"] if f80 else None, first_family_probe_there=at80[key],
                                               last_probe_row_first_family_at_half=lh["updates"] if lh else None, unrelated_held_there=lh["unrelated_held_pct"] if lh else None,
                                               reduction_end=z["score1_first_family_all_pairs_k0"]["reduction_score"], unrelated_held_end=100 * z["score3_new_task"]["held"]["exact"] / 300)
            v = read["B3"][f"{rule} {key}"]
            g = lambda q, f="{:.1f}": "-" if q is None else f.format(q)
            md.append(f"| {rule} | {key} | {v['updates']} | {g(v['unrelated_80_first_probe_row'], '{}')} | {g(v['first_family_probe_there'], '{:.2f}')} | {g(v['last_probe_row_first_family_at_half'], '{}')} | "
                      f"{g(v['unrelated_held_there'])} | {v['reduction_end']:.1f} | {v['unrelated_held_end']:.1f} |")
        base = at80["0.001"]
        more = any(at80[k] is not None and base is not None and at80[k] - base >= 20 for k in ("0.0003", "0.0001"))
        read["B3"][f"{rule} smaller steps leave more"] = more
        md.append(f"| {rule} | **smaller steps leave more** | {'yes' if more else 'no'} | | | | | | |")
    (B / "E51_READ_B.json").write_text(json.dumps(dict(experiment="E51", part="B", seed=seed, **read), indent=1))
    (B / "E51_TABLES_B.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    {"a": part_a, "b": part_b}[sys.argv[1]](int(sys.argv[2]))
