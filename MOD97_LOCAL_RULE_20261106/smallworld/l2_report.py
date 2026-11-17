"""E44, job L2 (roster note OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md, revision 2): the tables of the trial's report and of the
final report, from the receipts, the event files and the list files of the lives. Nothing here trains.

    python3 l2_report.py scores      # every finished life: l2_score.quantities at 256 visits and, for a long life, at 1,024
                                     #   -> exp45/e44/L2/scores/<stem>_V<visits>.json (a file that exists is not written again)
    python3 l2_report.py trial       # world 1's two lives -> exp45/e44/L2/trial/ (E44_L2_TRIAL.json, E44_L2_TRIAL_TABLES.md, the curves as .csv)
    python3 l2_report.py final       # all lives and the probe -> exp45/e44/L2/E44_L2.json, E44_L2_TABLES.md, curves and norms as .csv
"""
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MDIR = HERE.parent
ROOT = MDIR.parent
sys.path.insert(0, str(HERE))
import l2_score as L                  # noqa: E402
import l2_compare as C                # noqa: E402
import l2_lives as LV                 # noqa: E402

L2 = MDIR / "exp45" / "e44" / "L2"
LIVES, PROBE, SCORES, TRIAL, NORMS = L2 / "lives", L2 / "probe", L2 / "scores", L2 / "trial", L2 / "norms"
REF = "origin/claude/vibrant-cerf-sje726"
RDIR = "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/"
ARMS = ("naive", "replay", "naive-wd0", "replay-wd0")
ABS = ("1 visit", "2 to 4", "5 to 32", "33 or more", "5 or more")


def stem(arm, w, visits):
    return LV.stem(arm.split("-")[0], w, visits, arm.endswith("wd0"))


def life_of(arm, w):
    """The life that gives arm's figures in world w: the long life for naive and replay in worlds 1 to 4, otherwise the 256-visit life."""
    return stem(arm, w, 1024 if (w <= 4 and not arm.endswith("wd0")) else 256)


def receipt(s):
    return json.loads((LIVES / f"{s}.json").read_text())


def finished(s):
    return (LIVES / f"{s}.json").exists()


def f3(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def sgn(x, d=4):
    return "—" if x is None else f"{x:+.{d}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def score(s, V):
    SCORES.mkdir(parents=True, exist_ok=True)
    p = SCORES / f"{s}_V{V}.json"
    if p.exists():
        return json.loads(p.read_text())
    w = int(s.split("_w")[1].split("_")[0])
    r = L.quantities(L.load(LIVES / f"{s}.npz"), L.load(LIVES / f"{s}_list.npz"), w, V)
    r["life"] = s
    p.write_text(json.dumps(r, indent=1))
    return r


def scores():
    for j in LV.roster():
        s = j.get("stem")
        if s and finished(s):
            score(s, 256)
            if "_v1024" in s:
                score(s, 1024)
            print("scored", s, flush=True)


# ---------------------------------------------------------------- pieces shared by the two reports
def blocks(q, size=64):
    V = q["visits"]; rows = []
    for b in range(0, V, size):
        vs = range(b, min(V, b + size))
        m = lambda xs, k=None: (lambda ys: float(np.mean(ys)) if ys else None)([xs[v] for v in vs if k is None or q["kind"][v] == k])
        rows.append(dict(visits=f"{b} to {vs[-1]}", all=m(q["score_by_age"]), blind=m(q["blind_by_age"]), returns=m(q["score_by_age"], "return"),
                         continuations=m(q["score_by_age"], "continuation"), first=m(q["score_by_age"], "first"),
                         n_return=sum(q["kind"][v] == "return" for v in vs), n_cont=sum(q["kind"][v] == "continuation" for v in vs),
                         n_first=sum(q["kind"][v] == "first" for v in vs)))
    return rows


def curves_csv(path, qs):
    """One row a visit: the kind of visit, and for each life its score by age on reduction items and the blind predictor's."""
    assert not Path(path).exists(), f"refusing an existing path: {path}"
    names = list(qs)
    V = max(q["visits"] for q in qs.values())
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["visit"] + [f"{n}:{c}" for n in names for c in ("kind", "family", "score", "blind")])
        for v in range(V):
            row = [v]
            for n in names:
                q = qs[n]
                if v < q["visits"]:
                    fam = next((r["family"] for r in q["first_visits"] if r["visit"] == v), None)
                    row += [q["kind"][v], "" if fam is None else fam, f"{q['score_by_age'][v]:.6f}", f"{q['blind_by_age'][v]:.6f}"]
                else:
                    row += ["", "", "", ""]
            wr.writerow(row)


def keys_check(td, s, w, V):
    p = LIVES / f"{s}_list.npz"
    a = [sys.executable, "-I", str(Path(td) / "e44_l2_checks.py"), "K", str(p), str(w), str(V), "8", "none"]
    run = subprocess.run(a, capture_output=True, text=True, cwd=td)
    assert run.returncode == 0, run.stderr[-500:]
    return dict(command=f"python3 -I e44_l2_checks.py K <{p.name}> {w} {V} 8 none", output=run.stdout.strip().splitlines(),
                keys_equal_as_sets="keys equal as sets: True" in run.stdout, items_in_reviewers_order="in the reviewer's order: True" in run.stdout)


def reviewer_dir(td):
    names = subprocess.run(["git", "ls-tree", "--name-only", REF, RDIR], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    hs = {}
    for n in names:
        if n.endswith(".py"):
            b = subprocess.run(["git", "show", f"{REF}:{n}"], cwd=ROOT, capture_output=True, check=True).stdout
            (Path(td) / Path(n).name).write_bytes(b); hs[Path(n).name] = hashlib.sha256(b).hexdigest()
    return {k: hs[k] for k in ("e44_l2_checks.py", "e44_world_reader.py", "np_gru.py")}


def life_facts(s):
    r = receipt(s)
    losses = [v["loss_mean"] for v in r["by_visit"]]
    return dict(life=s, command=r["provenance"]["command"], commit=r["provenance"]["git"]["commit"], host=r["provenance"]["host"],
                env=r["provenance"]["env"], sw_run_sha256=r["provenance"]["sources_sha256"]["smallworld/sw_run.py"],
                settings={k: r["settings"][k] for k in ("arm", "world", "visits", "rho", "wd", "list_every", "decay_distances", "visit_updates")},
                learner=dict(file=r["learner"]["file"], seed=r["learner"]["seed"], file_sha256=r["learner"]["file_sha256"]),
                optimizer_weight_decay=r["optimizer"]["weight_decay"], first_group_weight_decay_at_end=r["optimizer"]["groups"][0]["weight_decay"],
                cpu_hours=r["cpu"]["total"] / 3600, cpu_seconds_per_visit=r["cpu"]["mean_per_visit"], list_cpu_hours=r["cpu"]["list_total"] / 3600,
                peak_rss_mib=r["cpu"]["peak_rss_kib"] / 1024, losses_finite=all(math.isfinite(x) for x in losses), loss_first_visit=losses[0], loss_last_visit=losses[-1],
                loss_max=max(losses), resumes=[dict(done_visits=x["done_visits"], wall_utc=x["wall_utc"]) for x in r["resumes"]],
                ledger=dict(stored_numbers=r["ledger"]["stored_numbers"], buffer=r["ledger"]["buffer"], replayed_per_update=r["ledger"]["replayed_per_update"],
                            new_learned_per_update=r["ledger"]["new_learned_per_update_once_replaying"], updates=r["ledger"]["updates"]),
                counters=r["counters"], events=dict(file=r["events"]["file"], entries=r["events"]["entries"], arrays_sha256=r["events"]["arrays_sha256"],
                                                    file_sha256=r["events"]["file_sha256"]),
                list=dict(file=r["list"]["file"], entries=r["list"]["entries"], arrays_sha256=r["list"]["arrays_sha256"], file_sha256=r["list"]["file_sha256"],
                          items_sha256=r["list"]["items_sha256"], families_taken=len(r["list"]["families_taken"])),
                final=r["final"], stream_sha256=r["stream_sha256"], learned_sha256=r["learned_sha256"],
                states=[dict(visits=x["visits"], file_sha256=x["file_sha256"], weights_tensors_sha256=x["weights_tensors_sha256"]) for x in r["states"]])


def md_quantities(names, qs):
    """The tables of D4 to D10, D14 and D15 for the named lives (each a quantities dict), as markdown."""
    o = []
    o.append(table(["Life", "Visits read", "Families met", "First / continuation / return visits", "Late level (D4)", "Blind predictor's", "Late level: returns",
                    "continuations", "first visits", "Score trend, returns (D5)", "… all visits (not used)", "Trend of learning (D7)", "… ln n in the joint fit",
                    "Median first-visit level (D6)", "… without the first family"],
                   [[n, q["visits"], q["families_met"], f"{q['kinds']['first']} / {q['kinds']['continuation']} / {q['kinds']['return']}", f3(q["late_level"]),
                     f3(q["blind_late_level"]), f3(q["late_level_by_kind"]["return"]), f3(q["late_level_by_kind"]["continuation"]), f3(q["late_level_by_kind"]["first"]),
                     sgn(q["score_trend"]), sgn(q["score_trend_all_visits"]), sgn(q["trend_of_learning"]),
                     sgn(q["trend_of_learning_joint_fit"]["ln_n"]) if q["trend_of_learning_joint_fit"] else "—", f3(q["median_first_visit_level"]),
                     f3(q["median_first_visit_level_families_2_on"])] for n, q in zip(names, qs)]))
    o.append("\n**Kept at return, from the stream (D8), and relearning at return (D10).** Share right on the reduction items of the visit's first update "
             "(answered before any update of the visit); the family's figure over the first 16 updates and over the last 64 updates of the visit; the last 64 "
             "updates of the family's previous visit. The class \"5 or more\" pools the two classes above it (recommendation 4).\n")
    o.append(table(["Life", "Absence", "Updates since", "Decay constants", "Visits", "Items", "Right", "Kept (D8)", "First 16 updates", "Last 64 updates", "Previous visit, last 64"],
                   [[n, a, (lambda r: f"{r['updates'][0]:,}" + ("+" if r["updates"][1] is None else f" to {r['updates'][1]:,}"))(q["absence_classes_restated"][a]),
                     (lambda r: f"{r['decay_constants'][0]:.2f}" + ("+" if r["decay_constants"][1] is None else f" to {r['decay_constants'][1]:.2f}"))(q["absence_classes_restated"][a]),
                     k["visits"], k["items"], k["right"], f3(k["share"]), f3(k["first16"]), f3(k["last_quarter"]), f3(k["previous_last_quarter"])]
                    for n, q in zip(names, qs) for a in ABS for k in [q["kept_at_return"][a]]]))
    o.append("\n**Kept on the list (D9).** A family's share right on its 200 list items at the end of visit v divided by its share at the end of its most recent "
             "visit ℓ < v, by v − ℓ; a family whose share at ℓ is 0 is left out and counted. Beside it the mean share right itself.\n")
    o.append(table(["Life", "Visits since the family's last visit", "Points", "Left out", "Mean kept", "Median kept", "Mean share right"],
                   [[n, a, k["points"], k["left_out"], f3(k["mean"]), f3(k["median"]), f3(k["mean_share_right"])]
                    for n, q in zip(names, qs) for a in ABS for k in [q["kept_on_list"][a]]]))
    o.append("\n**Does a memory become more lasting as a family is met again? (D15.)** Kept at return (D8) and the family's figure over the first 16 updates of the "
             "visit (D10), by the number of earlier visits to the family within each class of absence. Each cell: kept (items) / first 16 updates (visits).\n")
    o.append(table(["Life", "Absence"] + [f"{e[0]} earlier" for e in L.EARLIER],
                   [[n, a] + [(lambda k: "—" if not k["visits"] else f"{f3(k['share'])} ({k['items']}) / {f3(k['first16'])} ({k['visits']})")(q["kept_at_return_by_earlier_visits"][a][e[0]])
                              for e in L.EARLIER] for n, q in zip(names, qs) for a in ABS[:4]]))
    o.append("\n**What the wrong answers are (D14).** Shares of all answers, classes tried in the order of the columns. \"Stream\": the reduction items of the first "
             "update of a visit to a family met before (the family taught last is the family of the visit before; at an absence of 1 it is the item's own family, so "
             "that class is empty). \"List\": a family's 200 list items at the end of a visit to another family (the family taught last is that visit's). "
             "\"Chance\": the modulus of family 55, which no life is scheduled to meet, tried in the place of the family taught last.\n")
    short = ["right", "no reduction", "last family's modulus", "an earlier family's", "89", "an operand", "never-met modulus", "another two-digit number", "not two digits and end"]
    for part, key in (("Stream", "wrong_answers_stream"), ("List", "wrong_answers_list")):
        o.append(table(["Life", part, "Absence", "Answers"] + short + ["Chance"],
                       [[n, part, a, t["answers"]] + [f3(t["shares"][c]) for c in L.ANSWER_CLASSES] + [f3(t["chance_never_met_modulus_in_place_of_the_last"])]
                        for n, q in zip(names, qs) for a in ABS[:4] for t in [q[key][a]]]))
        o.append("")
    return "\n".join(o)


# ---------------------------------------------------------------- the trial
def trial():
    TRIAL.mkdir(parents=True, exist_ok=True)
    jp, mp = TRIAL / "E44_L2_TRIAL.json", TRIAL / "E44_L2_TRIAL_TABLES.md"
    assert not jp.exists() and not mp.exists(), "refusing existing trial files"
    lives = {"naive": stem("naive", 1, 1024), "replay": stem("replay", 1, 1024)}
    facts = {a: life_facts(s) for a, s in lives.items()}
    q = {(a, V): score(s, V) for a, s in lives.items() for V in (256, 1024)}
    with tempfile.TemporaryDirectory() as td:
        rv = reviewer_dir(td)
        keys = {a: keys_check(td, s, 1, 1024) for a, s in lives.items()}
    cmp_ = {f"{a}, {V} visits": C.compare(LIVES / f"{s}.npz", LIVES / f"{s}_list.npz", 1, V) for a, s in lives.items() for V in (256, 1024)}
    audit = subprocess.run(["git", "show", f"{REF}:{RDIR}AUDIT_E44_L2_ROSTER_NOTE_20261117.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    want_items = "48e6d710e93cbe6777cc2c0cb51c611949fbf1309e17c66c911c04487b7282fc"; want_keys = "e7cdf44f0f9485ff9a64824d3544e11361d5b451309651ba3ad2c927b089bfe7"
    assert want_items in audit and want_keys in audit
    checks = {a: dict(list_items_sha256_equal_to_audit=facts[a]["list"]["items_sha256"] == want_items, families_taken=facts[a]["list"]["families_taken"],
                      keys_digest_equal_to_audit=any(want_keys in ln for ln in keys[a]["output"]) and keys[a]["keys_equal_as_sets"],
                      reviewers_part_K=keys[a]) for a in lives}
    # the stop conditions
    norms_r = json.loads((NORMS / f"{lives['replay']}_norms.json").read_text())
    store64 = norms_r["visits"].get("64", {}).get("store")
    late = {a: float(np.mean(q[a, 1024]["score_by_age"][512:1024])) for a in lives}
    c7 = json.loads((L2 / "checks" / "E44_L2_CHECKS.json").read_text())["checks"]["7"]
    per_visit = {a: facts[a]["cpu_hours"] / 1024 for a in lives}                       # the list's cost included
    forecast = 4 * 1024 * (per_visit["naive"] + per_visit["replay"]) + 4 * 256 * (per_visit["naive"] + per_visit["replay"]) + 8 * 256 * (per_visit["naive"] + per_visit["replay"])
    stops = dict(
        loss_not_finite=dict(holds=not all(facts[a]["losses_finite"] for a in lives), losses={a: dict(first=facts[a]["loss_first_visit"], last=facts[a]["loss_last_visit"],
                                                                                                     largest=facts[a]["loss_max"]) for a in lives}),
        buffer_below_capacity_after_visit_64=dict(holds=not (store64 is not None and store64["items"] == 55209 and facts["replay"]["ledger"]["buffer"]["items_at_end"] == 55209),
                                                  store_in_the_state_after_64_visits=store64, at_end=facts["replay"]["ledger"]["buffer"],
                                                  first_replay_update=facts["replay"]["counters"]["first_replay_update"]),
        score_below_0_05_over_visits_512_to_1023_in_both_arms=dict(holds=all(x < 0.05 for x in late.values()), mean_score_by_age_512_to_1023=late),
        forecast_above_75_cpu_hours=dict(holds=forecast > 75, forecast_cpu_hours=forecast, cpu_hours_per_life_of_1024={a: facts[a]["cpu_hours"] for a in lives},
                                         forecast_before_the_trial=c7["forecast_release_1_cpu_hours"],
                                         note="32 lives at each arm's measured CPU-hours a visit in the trial, the list included; the decay-0 arms at their partners' cost"))
    any_stop = any(s["holds"] for s in stops.values())
    med = {a: dict(all_families=q[a, 1024]["median_first_visit_level"], without_the_first=q[a, 1024]["median_first_visit_level_families_2_on"],
                   at_256_visits=q[a, 256]["median_first_visit_level"], families=q[a, 1024]["families_met"]) for a in lives}
    out = dict(experiment="E44", job="L2, release 1: the trial (world 1, 1,024 visits, naive and replay)", lives=facts, checks=checks,
               reviewer=dict(ref=REF, commit=subprocess.run(["git", "rev-parse", REF], cwd=ROOT, capture_output=True, text=True).stdout.strip(), files_sha256=rv),
               scorer_against_part_J=cmp_, stop_conditions=stops, any_stop_condition_holds=any_stop, median_first_visit_level=med,
               replay_median_below_0_20=med["replay"]["all_families"] < 0.20,
               quantities={f"{a}, {V} visits": {k: v for k, v in q[a, V].items() if k not in ("score_by_age", "blind_by_age", "kind", "later_visits")}
                           for a in lives for V in (256, 1024)},
               blocks={a: blocks(q[a, 1024]) for a in lives})
    jp.write_text(json.dumps(out, indent=1))
    curves_csv(TRIAL / "E44_L2_TRIAL_score_by_age.csv", {a: q[a, 1024] for a in lives})
    with open(TRIAL / "E44_L2_TRIAL_first_visits.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["n", "family", "modulus", "taught_pairs", "visit", "naive_level", "replay_level", "naive_whole_visit", "replay_whole_visit"])
        for a, b in zip(q["naive", 1024]["first_visits"], q["replay", 1024]["first_visits"]):
            wr.writerow([a["n"], a["family"], a["p"], a["taught_pairs"], a["visit"], f"{a['level']:.6f}", f"{b['level']:.6f}", f"{a['whole']:.6f}", f"{b['whole']:.6f}"])
    # the tables
    o = ["# E44, job L2, release 1: the trial's tables (world 1, 1,024 visits, `naive` and `replay`)", "",
         "Written by `smallworld/l2_report.py trial` from the two lives' receipts, event files and list files; every figure is in `E44_L2_TRIAL.json` beside this file.", "",
         "## The two lives", "",
         table(["Arm", "Life", "Commit", "CPU-hours", "of which the list", "CPU-seconds a visit", "Peak memory, MiB", "Resumed", "Store: items at the end / capacity",
                "Replayed items an update", "Decay"],
               [[a, f"`{s}`", facts[a]["commit"][:8], f3(facts[a]["cpu_hours"], 2), f3(facts[a]["list_cpu_hours"], 2), f3(facts[a]["cpu_seconds_per_visit"], 2),
                 f3(facts[a]["peak_rss_mib"], 0), len(facts[a]["resumes"]),
                 f"{facts[a]['ledger']['buffer']['items_at_end']:,} / {facts[a]['ledger']['buffer']['capacity_items']:,}", facts[a]["ledger"]["replayed_per_update"],
                 facts[a]["optimizer_weight_decay"]] for a, s in lives.items()]), "",
         "Host of both: " + json.dumps(facts["naive"]["host"]) + "; threads " + json.dumps(facts["naive"]["env"]) + ".", "",
         "Commands:", ""] + [f"- `{facts[a]['command']}`" for a in lives] + ["",
         "## Checks on the files", "",
         table(["Arm", "List items digest equals the audit's (52 families)", "Row keys equal the reviewer's reference (part K; 5,031 keys, 1,238,400 items)", "Items in the reviewer's order",
                "Scorer equals part J at 256 visits", "at 1,024 visits"],
               [[a, checks[a]["list_items_sha256_equal_to_audit"], checks[a]["keys_digest_equal_to_audit"], keys[a]["items_in_reviewers_order"],
                 cmp_[f"{a}, 256 visits"]["all_equal"], cmp_[f"{a}, 1024 visits"]["all_equal"]] for a in lives]), "",
         "## The stop conditions", "",
         table(["Condition", "Holds", "Figures"],
               [["a loss that is not finite", stops["loss_not_finite"]["holds"],
                 "; ".join(f"{a}: mean loss of the first visit {v['first']:.4f}, of the last {v['last']:.4f}, largest {v['largest']:.4f}" for a, v in stops["loss_not_finite"]["losses"].items())],
                ["the buffer below capacity after visit 64", stops["buffer_below_capacity_after_visit_64"]["holds"],
                 f"the store in the state after 64 visits: {store64}; at the end {facts['replay']['ledger']['buffer']['items_at_end']:,} items; first replayed update {facts['replay']['counters']['first_replay_update']}"],
                ["score by age on reduction pairs below 0.05 over visits 512 to 1,023 in both arms", stops["score_below_0_05_over_visits_512_to_1023_in_both_arms"]["holds"],
                 "; ".join(f"{a} {v:.3f}" for a, v in late.items())],
                ["a forecast for release 1 above 75 CPU-hours", stops["forecast_above_75_cpu_hours"]["holds"], f"{forecast:.1f} CPU-hours (before the trial: {c7['forecast_release_1_cpu_hours']:.1f})"]]), "",
         f"**Any stop condition holds: {any_stop}.**", "",
         "## The first-visit level of `replay` beside `naive`'s (change 8)", "",
         table(["Arm", "Families met", "Median first-visit level (D6), all families", "without the first family", "over the families met in the first 256 visits"],
               [[a, med[a]["families"], f3(med[a]["all_families"]), f3(med[a]["without_the_first"]), f3(med[a]["at_256_visits"])] for a in lives]), "",
         f"`replay`'s median is {'below' if out['replay_median_below_0_20'] else 'not below'} 0.20.", "",
         "## Score by age on reduction items (D2), in blocks of 64 visits", "",
         "Every visit is in `E44_L2_TRIAL_score_by_age.csv`. \"Returns\" are visits to a family met before after an absence of two or more visits.", "",
         table(["Visits", "First / continuation / return", "naive: all", "naive: returns", "naive: blind", "replay: all", "replay: returns", "replay: blind"],
               [[x["visits"], f"{x['n_first']} / {x['n_cont']} / {x['n_return']}", f3(x["all"]), f3(x["returns"]), f3(x["blind"]), f3(y["all"]), f3(y["returns"]), f3(y["blind"])]
                for x, y in zip(out["blocks"]["naive"], out["blocks"]["replay"])]), "",
         "## The quantities D4 to D10, D14 and D15", "",
         md_quantities([f"{a}, {V}" for V in (256, 1024) for a in lives], [q[a, V] for V in (256, 1024) for a in lives])]
    mp.write_text("\n".join(o) + "\n")
    print("trial tables written; any stop condition holds:", any_stop, "; replay median first-visit level", med["replay"]["all_families"], "naive", med["naive"]["all_families"])


if __name__ == "__main__":
    {"scores": scores, "trial": trial}[sys.argv[1]]()
