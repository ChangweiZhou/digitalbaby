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


# ---------------------------------------------------------------- the final report's tables
PREDICTIONS = (("PL2.1", "`naive` at 1,024 visits is *stuck*", 0.45), ("PL2.2", "`replay` is *higher* than `naive` in late level at 1,024 visits", 0.75),
               ("PL2.3", "`replay` is *higher* than `naive` in kept at return after long absences, at 256 visits", 0.85),
               ("PL2.4", "`naive-wd0` against `naive` at 256 visits: *not distinguished* in late level", 0.60),
               ("PL2.5", "`naive-wd0` is *higher* than `naive` in kept at return after long absences, at 256 visits", 0.30),
               ("PL2.6", "`replay` at 1,024 visits is *still learning*", 0.55),
               ("PL2.7", "The trend of learning under `naive` at 1,024 visits has an interval above zero", 0.25),
               ("PL2.8", "The comparison of `naive` and `replay` closes at 1,024 visits", 0.70),
               ("PL2.9", "`replay-wd0` against `replay` at 256 visits is *not distinguished* in late level (the reviewer's)", 0.60))


def q_of(arm, w, V):
    s = life_of(arm, w)
    return score(s, V) if finished(s) else None


def worlds_of(V):
    return range(1, 5) if V == 1024 else range(1, 9)


def per_world(arm, V, f):
    return {w: f(q) for w in worlds_of(V) for q in [q_of(arm, w, V)] if q is not None}


def diff(a, b, V, f):
    """D11: per world the difference f(a) - f(b), its mean and 90% interval over worlds, and the word."""
    xa, xb = per_world(a, V, f), per_world(b, V, f)
    ws = [w for w in xa if w in xb and xa[w] is not None and xb[w] is not None]
    iv = L.interval90([xa[w] - xb[w] for w in ws])
    return dict(worlds=ws, per_world={w: xa[w] - xb[w] for w in ws}, interval=iv, word=verdict(iv))


def verdict(iv):
    return L.verdict(iv)


def trend_state(arm, V):
    """D13: stuck, still learning (and declining) from D5 and D7 over worlds."""
    i5 = L.interval90(list(per_world(arm, V, lambda q: q["score_trend"]).values()))
    i7 = L.interval90(list(per_world(arm, V, lambda q: q["trend_of_learning"]).values()))
    up = lambda iv: iv["low"] is not None and iv["low"] > 0
    dn = lambda iv: iv["high"] is not None and iv["high"] < 0
    if i5["low"] is None or i7["low"] is None:
        word = "not read"
    else:
        word = "still learning" if (up(i5) or up(i7)) else "stuck"
    return dict(score_trend=i5, trend_of_learning=i7, word=word, score_trend_declining=dn(i5), trend_of_learning_declining=dn(i7))


KEPT = lambda name: (lambda q: q["kept_at_return"][name]["share"])


def pooled_d15(arm, V):
    """D15 pooled over the worlds of a life length: kept (right / items) and the mean first-16 figure over the returns of each cell."""
    out = {}
    for name, lo, hi in L.CLASSES[1:4]:
        out[name] = {}
        for en, elo, ehi in L.EARLIER:
            n = k = 0; f16 = []; visits = 0
            for w in worlds_of(V):
                q = q_of(arm, w, V)
                if q is None:
                    continue
                for r in q["later_visits"]:
                    if lo <= r["absence"] <= hi and elo <= r["earlier_visits"] <= ehi:
                        n += r["n"]; k += r["right"]; visits += 1
                        if r["first16"] is not None:
                            f16.append(r["first16"])
            out[name][en] = dict(returns=visits, items=n, right=k, kept=(k / n) if n else None, first16=float(np.mean(f16)) if f16 else None)
    return out


def pooled_wrong(arm, V, key, names=("2 to 4", "5 to 32", "33 or more")):
    c = np.zeros(len(L.ANSWER_CLASSES)); ctrl = 0.0; n = 0
    for w in worlds_of(V):
        q = q_of(arm, w, V)
        if q is None:
            continue
        for name in names:
            t = q[key][name]
            if t["answers"]:
                c += np.array([t["shares"][x] * t["answers"] for x in L.ANSWER_CLASSES]); n += t["answers"]
                ctrl += t["chance_never_met_modulus_in_place_of_the_last"] * t["answers"]
    return dict(answers=int(n), shares={x: (float(c[i] / n) if n else None) for i, x in enumerate(L.ANSWER_CLASSES)}, chance=(ctrl / n if n else None))


def probe_table():
    rows = []
    for j in LV.roster():
        if "needs" not in j:
            continue
        p = PROBE / f"{j['id']}.npz"
        if not p.exists():
            continue
        s, rest = j["id"].split("_br")
        age = int(rest.split("_")[0]); w = int(s.split("_w")[1].split("_")[0]); arm = s.split("_")[1]
        r = L.probe(L.load(p), w)
        rows.append(dict(life=s, arm=arm, world=w, age=age, families=[x["family"] for x in r], levels=[x["level"] for x in r],
                         mean=float(np.mean([x["level"] for x in r]))))
    return rows


def norms_table():
    out = {}
    for p in sorted(NORMS.glob("*_norms.json")):
        d = json.loads(p.read_text())
        out[d["life"]] = {int(v): dict(total=float(np.sqrt(sum(x * x for k, x in r["norms"].items() if not k.startswith("book.")))),
                                       rnn_hh=r["norms"]["rnn.weight_hh_l0"], rnn_ih=r["norms"]["rnn.weight_ih_l0"], l3=r["norms"]["l3.weight"])
                          for v, r in d["visits"].items()}
    return out


def final():
    jp, mp = L2 / "E44_L2.json", L2 / "E44_L2_TABLES.md"
    assert not jp.exists() and not mp.exists(), "refusing existing final files"
    lives = [j["stem"] for j in LV.roster() if "stem" in j]
    missing = [s for s in lives if not finished(s)]
    facts = {s: life_facts(s) for s in lives if finished(s)}
    scores()
    with tempfile.TemporaryDirectory() as td:
        rv = reviewer_dir(td)
        keys = {s: keys_check(td, s, int(s.split("_w")[1].split("_")[0]), 1024 if "_v1024" in s else 256) for s in facts}
    res = dict(missing=missing, lives=facts, part_K=keys, reviewer_files_sha256=rv)
    # per-life quantities
    perlife = {}
    for arm in ARMS:
        for V in (256, 1024):
            if V == 1024 and arm.endswith("wd0"):
                continue
            for w in worlds_of(V):
                q = q_of(arm, w, V)
                if q is not None:
                    perlife[f"{arm}|{w}|{V}"] = {k: v for k, v in q.items() if k not in ("score_by_age", "blind_by_age", "kind", "later_visits")}
    res["per_life"] = perlife
    # statements
    st = {}
    for arm, V in (("naive", 1024), ("replay", 1024)) + tuple((a, 256) for a in ARMS):
        st[f"{arm} at {V}"] = trend_state(arm, V)
    res["stuck_or_still_learning"] = st
    comps = {}
    for a, b in (("replay", "naive"), ("naive-wd0", "naive"), ("replay-wd0", "replay")):
        for V in (256, 1024):
            if V == 1024 and (a.endswith("wd0") or b.endswith("wd0")):
                continue
            comps[f"{a} - {b} at {V}"] = {"late level (D4)": diff(a, b, V, lambda q: q["late_level"]),
                                          "kept at return, 5 to 32 (D8)": diff(a, b, V, KEPT("5 to 32")),
                                          "kept at return, 33 or more (D8)": diff(a, b, V, KEPT("33 or more")),
                                          "kept at return, 5 or more (D8)": diff(a, b, V, KEPT("5 or more")),
                                          "score trend (D5)": diff(a, b, V, lambda q: q["score_trend"]),
                                          "trend of learning (D7)": diff(a, b, V, lambda q: q["trend_of_learning"])}
    res["between_arms"] = comps
    c256, c1024 = comps.get("replay - naive at 256"), comps.get("replay - naive at 1024")
    closed = None
    if c256 and c1024:
        s256, s1024 = c256["late level (D4)"]["word"], c1024["late level (D4)"]["word"]
        opp = {"higher": "lower", "lower": "higher"}
        closed = (s256 == s1024 and s256 in ("higher", "lower") and c1024["score trend (D5)"]["word"] != opp[s256]
                  and c1024["trend of learning (D7)"]["word"] != opp[s256])
    res["closed_D12"] = closed
    res["d15"] = {f"{a} at {V}": pooled_d15(a, V) for a in ARMS for V in (256, 1024) if not (V == 1024 and a.endswith("wd0"))}
    res["d14_pooled_returns_2_or_more"] = {f"{a} at {V}, {k}": pooled_wrong(a, V, k) for a in ("naive", "replay") for V in (256, 1024)
                                           for k in ("wrong_answers_stream", "wrong_answers_list")}
    res["probe"] = probe_table()
    res["norms"] = norms_table()
    # predictions
    out = {}
    out["PL2.1"] = st["naive at 1024"]["word"] == "stuck"
    out["PL2.2"] = c1024 is not None and c1024["late level (D4)"]["word"] == "higher"
    out["PL2.3"] = c256 is not None and c256["kept at return, 33 or more (D8)"]["word"] == "higher"
    out["PL2.4"] = comps["naive-wd0 - naive at 256"]["late level (D4)"]["word"] == "not distinguished"
    out["PL2.5"] = comps["naive-wd0 - naive at 256"]["kept at return, 33 or more (D8)"]["word"] == "higher"
    out["PL2.6"] = st["replay at 1024"]["word"] == "still learning"
    iv = st["naive at 1024"]["trend_of_learning"]
    out["PL2.7"] = iv["low"] is not None and iv["low"] > 0
    out["PL2.8"] = bool(closed)
    out["PL2.9"] = comps["replay-wd0 - replay at 256"]["late level (D4)"]["word"] == "not distinguished"
    unread = {"PL2.1": st["naive at 1024"]["word"], "PL2.6": st["replay at 1024"]["word"],
              "PL2.2": c1024["late level (D4)"]["word"] if c1024 else "not read",
              "PL2.3": c256["kept at return, 33 or more (D8)"]["word"] if c256 else "not read",
              "PL2.4": comps["naive-wd0 - naive at 256"]["late level (D4)"]["word"], "PL2.5": comps["naive-wd0 - naive at 256"]["kept at return, 33 or more (D8)"]["word"],
              "PL2.9": comps["replay-wd0 - replay at 256"]["late level (D4)"]["word"], "PL2.7": "read" if iv["low"] is not None else "not read",
              "PL2.8": "read" if closed is not None else "not read"}
    res["predictions"] = {k: dict(statement=t, chance=c, holds=None if unread[k] == "not read" else out[k], read_as=unread[k]) for k, t, c in PREDICTIONS}
    nl = per_world("naive", 1024, lambda q: q["late_level"])
    rm = per_world("replay", 256, lambda q: q["median_first_visit_level"])
    w3 = res["d14_pooled_returns_2_or_more"]["naive at 1024, wrong_answers_stream"]
    res["reviewer_expectations"] = {
        "(i) naive's late level at 1,024 visits between 0.40 and 0.55 (55%)": dict(per_world=nl, mean=float(np.mean(list(nl.values()))) if nl else None,
                                                                                   holds=bool(nl) and 0.40 <= float(np.mean(list(nl.values()))) <= 0.55),
        "(ii) replay's median first-visit level below 0.20 in at least three of eight worlds at 256 visits (50%)": dict(per_world=rm, below=sum(x < 0.20 for x in rm.values()),
                                                                                                                        holds=sum(x < 0.20 for x in rm.values()) >= 3),
        "(iii) under naive at returns after two or more visits, more than a tenth of answers are the right answer under the modulus of the family taught last (75%)":
            dict(share=w3["shares"][L.ANSWER_CLASSES[2]], answers=w3["answers"], holds=(w3["shares"][L.ANSWER_CLASSES[2]] or 0) > 0.10)}
    jp.write_text(json.dumps(res, indent=1))
    # curves
    with open(L2 / "E44_L2_score_by_age_blocks.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["arm", "world", "visits", "block", "all", "returns", "continuations", "first", "blind", "n_return", "n_cont", "n_first"])
        for arm in ARMS:
            for w in range(1, 9):
                s = life_of(arm, w)
                if not finished(s):
                    continue
                V = 1024 if "_v1024" in s else 256
                for b in blocks(score(s, V), 32):
                    wr.writerow([arm, w, V, b["visits"], b["all"], b["returns"], b["continuations"], b["first"], b["blind"], b["n_return"], b["n_cont"], b["n_first"]])
    with open(L2 / "E44_L2_norms.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["life", "visits", "norm_all_but_byte_table", "rnn_weight_hh", "rnn_weight_ih", "l3_weight"])
        for life, d in res["norms"].items():
            for v in sorted(d):
                wr.writerow([life, v, d[v]["total"], d[v]["rnn_hh"], d[v]["rnn_ih"], d[v]["l3"]])
    # markdown
    o = ["# E44, job L2, release 1: the tables of the final report", "",
         "Written by `smallworld/l2_report.py final`; every figure is in `E44_L2.json` beside this file. Lives missing: " + (", ".join(missing) or "none") + ".", ""]
    o += ["## The lives", "", table(["Life", "Learner seed", "CPU-hours", "Resumes (from visit)", "Store at end", "Decay", "List keys equal the reviewer's (part K)"],
            [[f"`{s}`", f["learner"]["seed"], f3(f["cpu_hours"], 2), ", ".join(str(r["done_visits"]) for r in f["resumes"]) or "—",
              f"{f['ledger']['buffer']['items_at_end']:,}", f["optimizer_weight_decay"], keys[s]["keys_equal_as_sets"] and keys[s]["items_in_reviewers_order"]] for s, f in facts.items()]), ""]
    o += ["## Stuck or still learning (D13)", "", table(["Arm", "Visits", "Worlds", "Score trend D5: mean [90% interval]", "Trend of learning D7: mean [90% interval]", "Statement", "Declining"],
            [[k.split(" at ")[0], k.split(" at ")[1], v["score_trend"]["n"], f"{sgn(v['score_trend']['mean'])} [{sgn(v['score_trend']['low'])}, {sgn(v['score_trend']['high'])}]",
              f"{sgn(v['trend_of_learning']['mean'])} [{sgn(v['trend_of_learning']['low'])}, {sgn(v['trend_of_learning']['high'])}]", v["word"],
              ", ".join(x for x, y in (("D5", v["score_trend_declining"]), ("D7", v["trend_of_learning_declining"])) if y) or "—"] for k, v in st.items()]), ""]
    o += ["## Between arms (D11)", "", table(["Comparison", "Quantity", "Worlds", "Per world", "Mean [90% interval]", "Statement"],
            [[k, qn, len(d["worlds"]), " ".join(f"{d['per_world'][w]:+.3f}" for w in d["worlds"]), f"{sgn(d['interval']['mean'], 3)} [{sgn(d['interval']['low'], 3)}, {sgn(d['interval']['high'], 3)}]", d["word"]]
             for k, c in comps.items() for qn, d in c.items()]), "", f"**Closed (D12):** {closed}.", ""]
    o += ["## Predictions", "", table(["", "Prediction", "Chance", "Holds"], [[k, v["statement"], f"{v['chance']:.0%}", v["holds"]] for k, v in res["predictions"].items()]), ""]
    o += ["**The reviewer's expectations.**", ""] + [f"- {k}: {json.dumps({a: b for a, b in v.items() if a != 'per_world'})}; per world {json.dumps(v.get('per_world', {}))}" for k, v in res["reviewer_expectations"].items()] + [""]
    o += ["## Per world", "", table(["Arm", "World", "Visits", "Families", "Late level D4", "returns", "continuations", "first", "D5", "D7", "Median D6", "Kept 5 to 32", "Kept 33+", "Kept 5+", "List kept 5+ (D9)"],
            [[k.split("|")[0], k.split("|")[1], k.split("|")[2], q["families_met"], f3(q["late_level"]), f3(q["late_level_by_kind"]["return"]), f3(q["late_level_by_kind"]["continuation"]),
              f3(q["late_level_by_kind"]["first"]), sgn(q["score_trend"]), sgn(q["trend_of_learning"]), f3(q["median_first_visit_level"]),
              f3(q["kept_at_return"]["5 to 32"]["share"]), f3(q["kept_at_return"]["33 or more"]["share"]), f3(q["kept_at_return"]["5 or more"]["share"]),
              f3(q["kept_on_list"]["5 or more"]["mean"])] for k, q in perlife.items()]), ""]
    o += ["## D15, pooled over worlds", "", "Each cell: kept at return (items) / mean share right over the first 16 updates of the return (returns).", "",
          table(["Arm", "Visits", "Absence"] + [f"{e[0]} earlier" for e in L.EARLIER],
                [[k.split(" at ")[0], k.split(" at ")[1], a] + [(lambda c: "—" if not c["returns"] else f"{f3(c['kept'])} ({c['items']}) / {f3(c['first16'])} ({c['returns']})")(cells[e[0]]) for e in L.EARLIER]
                 for k, d in res["d15"].items() for a, cells in d.items()]), ""]
    short = ["right", "no reduction", "last family's modulus", "an earlier family's", "89", "an operand", "never-met modulus", "another two-digit number", "not two digits and end"]
    o += ["## D14, pooled over worlds and over absences of 2 visits or more", "", table(["Arm, visits, part", "Answers"] + short + ["Chance"],
            [[k, t["answers"]] + [f3(t["shares"][c]) for c in L.ANSWER_CLASSES] + [f3(t["chance"])] for k, t in res["d14_pooled_returns_2_or_more"].items()]), ""]
    o += ["## The learning probe", "", "Each probe family's first-visit level (D6), the branch visiting families 52, 53, 54 and 55 in that order.", "",
          table(["Arm", "World", "Age (visits)", "Levels of 52, 53, 54, 55", "Mean"], [[r["arm"], r["world"], r["age"], " ".join(f"{x:.3f}" for x in r["levels"]), f3(r["mean"])]
                                                                                for r in sorted(res["probe"], key=lambda r: (r["arm"], r["age"], r["world"]))]), ""]
    ages = {}
    for r in res["probe"]:
        ages.setdefault((r["arm"], r["age"]), []).append(r["mean"])
    o += [table(["Arm", "Age", "Worlds", "Mean over worlds of the mean over the four families"], [[a, g, len(v), f3(float(np.mean(v)))] for (a, g), v in sorted(ages.items())]), ""]
    o += ["## The sizes of the weights", "", "Euclidean norm of all weights but the byte table, and of the GRU's recurrent matrix, at the visits named (0: the born weights), mean over the worlds.", ""]
    rows = []
    for arm, tag, vs in (("naive", "_v256_wd0", (0, 64, 128, 192, 256)), ("replay", "_v256_rho0.5_wd0", (0, 64, 128, 192, 256)),
                         ("naive", "_v1024", (0, 64, 256, 512, 1024)), ("replay", "_v1024_rho0.5", (0, 64, 256, 512, 1024)),
                         ("naive", "_v256", (0, 64, 128, 192, 256)), ("replay", "_v256_rho0.5", (0, 64, 128, 192, 256))):
        ls = [d for life, d in res["norms"].items() if life.startswith(f"lw_{arm}_w") and life.endswith(tag)]
        if ls:
            rows.append([arm + tag.replace("_rho0.5", ""), len(ls)] + [(lambda xs: f"{np.mean([x[0] for x in xs]):.1f} / {np.mean([x[1] for x in xs]):.1f}" if xs else "—")(
                [(d[v]["total"], d[v]["rnn_hh"]) for d in ls if v in d]) for v in vs] + [", ".join(map(str, vs))])
    o += [table(["Lives", "Worlds", "1st", "2nd", "3rd", "4th", "5th", "Visits of the columns"], rows), ""]
    mp.write_text("\n".join(o) + "\n")
    print("final tables written; missing:", missing)


if __name__ == "__main__":
    {"scores": scores, "trial": trial, "final": final}[sys.argv[1]]()
