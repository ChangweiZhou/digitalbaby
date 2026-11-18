"""E51, Part B on seeds 93683 and 116089 (the coordinator's decision 3 of 18 November 2026: B1 and B2 only, the runner and checks as they
are). Two additions that exp51_partb.py does not cover for these seeds; it is not edited, and no torch is needed.

    forecast <seed>   check 11 for a new seed. exp51_partb.py's check 11 reads Part A's `open` receipt of the seed, which exists for
                      seed 73129 only; here the CPU-seconds an update are read from the seed's own B3 runs at rate 0.001 (the runs of
                      check 9, E43's `unrelated` arm reproduced), and the forecast covers B1 (four shares) and B2 (at its upper limit).
                      Writes exp45/e51/partB/checks/e51_check_11_cost_from_check9_<seed>.json.
    read <seed>...    the readings of B1 and B2 as the brief fixes them, for each seed named, from the receipts; seed 73129's are read
                      the same way for the side-by-side table. Writes exp45/e51/partB/E51_READ_B_SEEDS.json and E51_TABLES_B_SEEDS.md.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = HERE / "exp45" / "e51" / "partB"
E43 = HERE / "exp45" / "e43"
RULES, SHARES = ("S", "B"), (4, 8, 16, 32)
GAP, REVIEW_MAX, CYCLES = 256, 1024, 8


def forecast(seed):
    per = {}
    for rule in RULES:
        c = json.loads((B / f"e51_B3_steps_0.001_{rule}_{seed}.json").read_text())["complete"]
        per[rule] = c["cpu_updates"] / c["updates"]
    e43row, probe = 22.0, 1.5
    runs = {}
    for rule in RULES:
        for k in SHARES:
            runs[f"B1 k={k} {rule}"] = 4096 * per[rule] + 6 * e43row + 29 * probe
        runs[f"B2 {rule}"] = CYCLES * (GAP + REVIEW_MAX) * per[rule] + 2 * e43row + CYCLES * (4 + 64) * probe
    out = dict(experiment="E51", kind="check 11 (new seed)", seed=seed, cpu_seconds_an_update=per,
               source="the cpu_updates of the seed's B3 runs at rate 0.001 (check 9)", forecast_cpu_seconds=runs, runs=len(runs),
               forecast_cpu_hours=sum(runs.values()) / 3600, wall_hours_at_three_processes=sum(runs.values()) / 3600 / 3,
               note="B2 at its upper limit (every review 1,024 updates); the same arithmetic as exp51_partb.py's check 11", ok=True)
    p = B / "checks" / f"e51_check_11_cost_from_check9_{seed}.json"
    assert not p.exists(), p
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "forecast_cpu_seconds"}))


def red(row):
    return row["score1_first_family_all_pairs_k0"]["reduction_score"]


def held(row):
    return 100 * row["score3_new_task"]["held"]["exact"] / row["score3_new_task"]["held"]["n"]


def read_b1(seed, rule):
    res = {}
    ref2 = json.loads((E43 / "addendum2" / f"e43_unrelated-rehearse_{rule}_{seed}.json").read_text())
    ref0 = json.loads((E43 / f"e43_unrelated_{rule}_{seed}.json").read_text())
    entries = [(2, ref2, None)] + [(k, json.loads((B / f"e51_B1_share_{k}_{rule}_{seed}.json").read_text()), True) for k in SHARES] + [(0, ref0, None)]
    for k, rec, own in entries:
        assert rec["complete"] and rec["rows"][-1]["updates"] == 4096
        p = rec.get("probe") if own else None
        r0, r4 = red(rec["rows"][0]), red(rec["rows"][-1])
        res[k] = dict(reduction_row0=r0, reduction_4096=r4, pct_of_row0=100 * r4 / r0, kept=abs(r4 - r0) <= 5,
                      lowest_probe=min(x["first_family_pct"] for x in p) if p else None,
                      unrelated_80_first_probe_row=next((x["updates"] for x in p if x["unrelated_held_pct"] >= 80), None) if p else None,
                      unrelated_held_4096=held(rec["rows"][-1]))
    least = None
    for k in (2, 4, 8, 16, 32):
        if res[k]["kept"]:
            least = k
        else:
            break
    return res, least


def read_b2(seed, rule):
    rec = json.loads((B / f"e51_B2_intervals_{rule}_{seed}.json").read_text()); assert rec["complete"]
    cy = rec["cycles"]; c1, c8 = cy[0], cy[-1]
    shortens = c8["review_updates"] <= c1["review_updates"] / 2
    more = c8["probe_at_end_of_gap"]["first_family_pct"] - c1["probe_at_end_of_gap"]["first_family_pct"] >= 20
    return dict(cycles=[dict(cycle=c["cycle"], gap_end_first_family_pct=c["probe_at_end_of_gap"]["first_family_pct"],
                             gap_end_unrelated_held_pct=c["probe_at_end_of_gap"]["unrelated_held_pct"], review_updates=c["review_updates"],
                             review_reached_goal=c["review_reached_goal"], review_end_first_family_pct=c["probe_at_end_of_review"]["first_family_pct"],
                             review_end_unrelated_held_pct=c["probe_at_end_of_review"]["unrelated_held_pct"]) for c in cy],
                probe_at_update_0=rec["settings"]["probe_first_family_pct_at_update_0"], goal=rec["settings"]["review_goal_pct"],
                review_shortens=shortens, more_is_left=more, updates=rec["complete"]["updates"], reduction_row0=red(rec["rows"][0]),
                reduction_end=red(rec["rows"][-1]), unrelated_held_end=held(rec["rows"][-1]),
                sentence=f"with rehearsal at intervals the first family {'becomes' if shortens else 'does not become'} cheaper to restore, and "
                         f"{'more' if more else 'no more'} of it survives a gap")


def read(seeds):
    out, md = {}, ["# E51 Part B, B1 and B2 on three seeds", "", "Read by `e51_partb_seeds.py read " + " ".join(map(str, seeds)) + "` from the receipts.", ""]
    md += ["## B1. The share of rehearsal", "", "Reduction score at update 4,096 as a percentage of row 0 (kept: within 5 points of row 0); lowest probe; first probe row with the unrelated task's held score at 80%.", "",
           "| seed | rule | row 0 | 1/2 (addendum) | 1/4 | 1/8 | 1/16 | 1/32 | none (E43) | least share that keeps | PB.1 (1/4 keeps) | PB.2 (1/32 does not keep) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for seed in seeds:
        out[seed] = dict(B1={}, B2={})
        for rule in RULES:
            res, least = read_b1(seed, rule)
            out[seed]["B1"][rule] = dict(shares={str(k): v for k, v in res.items()}, least_share_that_keeps=None if least is None else f"1/{least}",
                                         PB1_quarter_keeps=res[4]["kept"], PB2_thirty_second_does_not_keep=not res[32]["kept"])
            cell = lambda k: f"{res[k]['reduction_4096']:.1f} ({res[k]['pct_of_row0']:.1f}%{', kept' if res[k]['kept'] else ''})"
            md.append(f"| {seed} | {rule} | {res[4]['reduction_row0']:.1f} | " + " | ".join(cell(k) for k in (2, 4, 8, 16, 32, 0))
                      + f" | {'none' if least is None else f'1/{least}'} | {'held' if res[4]['kept'] else 'failed'} | {'held' if not res[32]['kept'] else 'failed'} |")
    md += ["", "| seed | rule | share | lowest probe (%) | unrelated held at 80%: first probe row | unrelated held at 4,096 (%) |", "|---|---|---|---|---|---|"]
    for seed in seeds:
        for rule in RULES:
            for k in SHARES:
                v = out[seed]["B1"][rule]["shares"][str(k)]
                md.append(f"| {seed} | {rule} | 1/{k} | {v['lowest_probe']:.2f} | {v['unrelated_80_first_probe_row']} | {v['unrelated_held_4096']:.1f} |")
    md += ["", "## B2. Rehearsal at intervals", "", "For each cycle: the probe's first-family share at the end of the gap (%) / the review's updates.", "",
           "| seed | rule | probe at 0 | " + " | ".join(f"cycle {c}" for c in range(1, 9)) + " | review shortens | more is left | updates | reduction at 0 / end | unrelated held at end |",
           "|---|---|---|" + "---|" * 8 + "---|---|---|---|---|"]
    for seed in seeds:
        for rule in RULES:
            r = read_b2(seed, rule); out[seed]["B2"][rule] = r
            md.append(f"| {seed} | {rule} | {r['probe_at_update_0']:.2f} | " + " | ".join(f"{c['gap_end_first_family_pct']:.2f} / {c['review_updates']}" for c in r["cycles"])
                      + f" | {'yes' if r['review_shortens'] else 'no'} | {'yes' if r['more_is_left'] else 'no'} | {r['updates']} | {r['reduction_row0']:.1f} / {r['reduction_end']:.1f} | {r['unrelated_held_end']:.1f} |")
    md += ["", "The brief's sentence for each seed and rule:", ""]
    for seed in seeds:
        for rule in RULES:
            md.append(f"- seed {seed}, rule {rule}: {out[seed]['B2'][rule]['sentence']}.")
    md += ["", "Every cycle's review reached its goal: " + ", ".join(f"seed {s} {r}: {all(c['review_reached_goal'] for c in out[s]['B2'][r]['cycles'])}" for s in seeds for r in RULES) + "."]
    (B / "E51_READ_B_SEEDS.json").write_text(json.dumps(dict(experiment="E51", part="B (B1, B2)", seeds=seeds, readings=out), indent=1))
    (B / "E51_TABLES_B_SEEDS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    if sys.argv[1] == "forecast":
        forecast(int(sys.argv[2]))
    elif sys.argv[1] == "read":
        read([int(s) for s in sys.argv[2:]])
    else:
        raise SystemExit(__doc__)
