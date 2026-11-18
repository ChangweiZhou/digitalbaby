"""E55, first part: the readings fixed by the brief (revision 4, "How the result is stated"), from the receipts only; no torch.

    python3 e55_read.py <blind|shut|all>     writes exp45/e55/E55_READ_<arm>.json and E55_TABLES_<arm>.md
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
D55, D51 = HERE / "exp45" / "e55", HERE / "exp45" / "e51"
SEED, TASKS, RULES = 73129, ("unrelated", "conflict"), ("S", "B")
N0, NH = 31684, 1200


def red(r):
    return r["score1_first_family_all_pairs_k0"]["reduction_score"]


def d_counts(r):
    pp = r["share_of_answers_equal_to_row0"]["per_panel"]
    k0 = sum(v for n, v in pp.items() if n.startswith("k0_")); k12 = sum(v for n, v in pp.items() if n.startswith("k12_"))
    return N0 - k0, NH - k12


def first_learned(rows):
    for i, r in enumerate(rows):
        if all(x["score3_never_taught"]["pct"] >= 80 for x in rows[i:]):
            return r["updates"]
    return None


def words(arm, rows):
    r0 = red(rows[0])
    ds = [d_counts(r) for r in rows]
    exact = all(a == 0 and b == 0 for a, b in ds)
    first_fail = None
    for r, (a, b) in zip(rows, ds):
        conds = []
        if a > 31:
            conds.append(f"d(g) = {a} > 31")
        if b > 1:
            conds.append(f"{b} held-out answers differ (> 1)")
        if abs(red(r) - r0) > 0.5:
            conds.append(f"reduction score {red(r):.2f} is more than 0.5 from row 0's {r0:.2f}")
        if conds:
            first_fail = dict(row=r["updates"], conditions=conds, gates_on_first_family=r["gates_on_first_family"]); break
    if exact:
        w = "kept exactly"
    elif first_fail is None:
        w = "kept"
    elif red(rows[-1]) < r0 / 2:
        w = "lost"
    else:
        w = "partly kept"
    d40 = [r["first_family_40_letters"]["differ_from_row0"] for r in rows]
    if arm == "blind" and w in ("kept exactly", "kept") and max(d40) > 3:
        w40 = "kept on items of up to 22 positions and not on items of 50"
    else:
        w40 = None
    if w40:                                          # the brief's wording for such a run, alone (the reviewer's verification, finding 4)
        w = w40
    return dict(word=w, word_40_letters=w40, largest_d=max(a for a, _ in ds), largest_d_pct=100 * max(a for a, _ in ds) / N0,
                largest_heldout_differ=max(b for _, b in ds), reduction_row0=r0, reduction_last=red(rows[-1]), first_failure_of_kept=first_fail,
                differ_40_letters_by_row={r["updates"]: x for r, x in zip(rows, d40)},
                right_40_letters_by_row={r["updates"]: r["first_family_40_letters"]["exact"] for r in rows},
                reserve_state_40_letters_by_row={r["updates"]: r["gates_on_first_family_40_letters"]["largest_reserve_state"] for r in rows},
                largest_opening_40_letters_by_row={r["updates"]: r["gates_on_first_family_40_letters"]["largest_opening"] for r in rows})


def read_run(task, arm, rule):
    f = D55 / f"e55_{task}_{arm}_{rule}_{SEED}.json"; rec = json.loads(f.read_text()); assert rec["complete"], f
    rows = rec["rows"]; assert rows[-1]["updates"] == 16384
    fl = first_learned(rows); last = rows[-1]
    out = dict(receipt=f.name, **words(arm, rows), learned_from_row=fl, never_taught_last=last["score3_never_taught"]["pct"],
               never_taught_shut_last=last["score3_never_taught_reserve_shut"]["pct"],
               largest_reserve_state_first_family=max(r["gates_on_first_family"]["largest_reserve_state"] for r in rows),
               largest_output_change=max(r["first_family_largest_change_of_an_output"] for r in rows),
               eigenvalue_by_row={r["updates"]: r["reserve_map_largest_eigenvalue"] for r in rows},
               cells_open_at_some_position_new_task_last=last["gates_on_new_task"]["cells_open_at_some_position"],
               cpu_total=rec["complete"]["cpu_total"])
    return out, rows


def e51_rows(task, arm, rule):
    f = D51 / f"e51_{task}_{arm}_{rule}_{SEED}.json"
    return json.loads(f.read_text())["rows"]


def table(task, arm, rule, rows):
    md = [f"### {task}, rule {rule}, arm `{arm}` (`e55_{task}_{arm}_{rule}_{SEED}.json`)", "",
          "| updates | reduction | d(g) of 31,684 | held-out differ of 1,200 | 40 letters differ of 300 | never-taught | never-taught, reserve shut | new task: cells open at some position | new task: open by > 0.1 in the mean | new task: largest 1 − z | first family: largest 1 − z | first family: cells open at some position | first family: largest reserve state | first family: largest added unit | largest output change | eigenvalue | 40 letters: right of 300 | 40 letters: largest reserve state | 40 letters: largest 1 − z |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a, b = d_counts(r); gn, gf = r["gates_on_new_task"], r["gates_on_first_family"]
        md.append(f"| {r['updates']} | {red(r):.2f} | {a} | {b} | {r['first_family_40_letters']['differ_from_row0']} | {r['score3_never_taught']['pct']:.1f} | "
                  f"{r['score3_never_taught_reserve_shut']['pct']:.1f} | {gn['cells_open_at_some_position']} | {gn['cells_open_by_more_than_0p1']} | {gn['largest_opening']:.3g} | "
                  f"{gf['largest_opening']:.3g} | {gf['cells_open_at_some_position']} | {gf['largest_reserve_state']:.3g} | {gf['largest_added_unit_at_answer_positions']:.3g} | "
                  f"{r['first_family_largest_change_of_an_output']:.3g} | {r['reserve_map_largest_eigenvalue']:.3f} | {r['first_family_40_letters']['exact']} | "
                  f"{r['gates_on_first_family_40_letters']['largest_reserve_state']:.3g} | {r['gates_on_first_family_40_letters']['largest_opening']:.3g} |")
    md += ["", f"E51's rows of the same task and rule (reduction; never-taught; never-taught with the reserve shut):", "",
           "| updates | `keyed` | `open` | `rehearse` |", "|---|---|---|---|"]
    ref = {a: {x["updates"]: x for x in e51_rows(task, a, rule)} for a in ("keyed", "open", "rehearse")}
    for u in sorted(ref["keyed"]):
        md.append(f"| {u} | " + " | ".join(f"{red(ref[a][u]):.1f}; {ref[a][u]['score3_never_taught']['pct']:.1f}; {ref[a][u]['score3_never_taught_reserve_shut']['pct']:.1f}" for a in ("keyed", "open", "rehearse")) + " |")
    return md


def main(which):
    arms = ("blind", "shut") if which == "all" else (which,)
    read, md = {}, [f"# E55, first part, seed {SEED}: the runs of the arm(s) {', '.join(arms)}", "", "Read by `python3 e55_read.py " + which + "` from the receipts.", ""]
    summ = ["| arm | task | rule | first family | largest d(g) | largest held-out differ | reduction row 0 / 16,384 | learned from row | never-taught at 16,384 | with the reserve shut | largest reserve state on the first family | largest output change | 40 letters differ at 16,384 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    body = []
    for arm in arms:
        for task in TASKS:
            for rule in RULES:
                v, rows = read_run(task, arm, rule); read[f"{arm} {task} {rule}"] = v
                summ.append(f"| `{arm}` | {task} | {rule} | {v['word']} | {v['largest_d']} | {v['largest_heldout_differ']} | "
                            f"{v['reduction_row0']:.2f} / {v['reduction_last']:.2f} | {v['learned_from_row'] if v['learned_from_row'] else 'not learned'} | {v['never_taught_last']:.1f} | "
                            f"{v['never_taught_shut_last']:.1f} | {v['largest_reserve_state_first_family']:.3g} | {v['largest_output_change']:.3g} | {v['differ_40_letters_by_row'][16384]} |")
                body += table(task, arm, rule, rows) + [""]
    notes = ["", "Notes (from the reviewer's verification of the arm `blind`, `E55_BLIND_VERIFICATION_20261118.md` at 77e65eba, findings 2, 3 and 5):", "",
             "- The eigenvalue column is the largest magnitude of an eigenvalue of 0.5 I + 0.25 Wn, Wn the candidate rows of the cells in reserve from the cells in reserve. It is above 1 at row 16,384 in all four runs of the arm `blind`. The construction leaves the weights among the cells in reserve free, so that nothing bounds the state of the reserve on longer items or after more updates; *kept exactly* is an observation on items of up to 22 positions.",
             "- The 40-letter counts, the small states and the output changes depend on the arithmetic as well as on the weights: at row 4,096 of `unrelated` under back-propagation, 45 answers at 40 letters differ in the receipt, 15 in double precision and 142 in single precision in the reviewer's numpy. The outcomes of the brief's rules are the same in each.",
             "- The outputs move by more than the smallest lead without a changed answer on the scored panels (largest change 1.69e-3 against a smallest lead of 1.78e-4 under the sweep; 0.200 against 2.33e-2 under back-propagation), so that *kept exactly* is not the consequence of a bound."]
    (D55 / f"E55_READ_{which}.json").write_text(json.dumps(dict(experiment="E55", part="first", seed=SEED, runs=read), indent=1))
    (D55 / f"E55_TABLES_{which}.md").write_text("\n".join(md + summ + notes + [""] + body) + "\n")
    print("\n".join(summ))
    for k, v in read.items():
        print(k, "first failure of kept:", v["first_failure_of_kept"] and {kk: vv for kk, vv in v["first_failure_of_kept"].items() if kk != "gates_on_first_family"})


if __name__ == "__main__":
    main(sys.argv[1])
