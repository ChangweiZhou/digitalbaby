"""Sizing sweep, stage 1: the admission record and the forecast for the 84 fits (brief OPUS_TASK_TARGET_SIZING_SWEEP_20261111.md rev. 3).

Runs in this process: default parity against TARGET_STEP0_CHECK.json (seeds 1, 2, 3), the modulus fixtures, the frozen taught panels'
reproduction, the scoring-code fixtures (oracle and ten-byte-suffix responders, world 104, every setting and path), the probe-protocol
fixtures on the real GRU (world 104), the line-use distances in the streams with lines (world 104). Read from runs/: the cross-cut
fixtures, the cost profile, the shape verification (each run as its own process, see the report).

    python3 sz_stage1.py core       the checks run in this process; writes runs/stage1_core.json
    python3 sz_stage1.py record     assembles SIZING_STAGE1_RECORD.json from runs/
No roster world is trained on; no fit is run; nothing is concluded about any learner or rule.
"""
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sz_check as CK              # noqa: E402
import sz_config as CF             # noqa: E402
import sz_learner as L             # noqa: E402
import sz_modfix as MF             # noqa: E402
import sz_panels as P              # noqa: E402
import sz_probe_fixtures as PF     # noqa: E402
import sz_score as SC              # noqa: E402
import sz_stream as G              # noqa: E402

RUNS = HERE / "runs"
W = CF.PROFILE_WORLD
PATHS = dict(A_ordinary=("A", "ordinary"), B_ordinary=("B", "ordinary"), B_wider=("B", "wider"), B_extra=("B", "extra"))


def timed(fn, *a):
    t0 = time.process_time(); r = fn(*a); return r, round(time.process_time() - t0, 1)


def parity():
    out = CK.parity(CF.default())
    mine = json.loads(json.dumps(out, indent=1, default=str))
    filed = json.loads((HERE.parent / "TARGET_STEP0_CHECK.json").read_text())
    diffs = CK.compare(mine, filed, renamed={"sz_oracle": "target_oracle", "sz_solver": "target_solver"})
    sizes_equal = filed["sizes"] == json.loads(json.dumps(CF.default()))
    return dict(seeds=[1, 2, 3], all_checks_passed=out["passed"], values_on_file=sum(1 for _ in CK._leaves(filed)), differences=len(diffs),
                first_differences=diffs[:5], default_configuration_equals_step0_sizes=sizes_equal,
                allowed_renaming="check 2 names the files it read: sz_oracle/sz_solver in place of target_oracle/target_solver",
                passed=out["passed"] and not diffs and sizes_equal)


def panels_frozen():
    d = json.loads(P.OUT.read_text()); bad = []
    for key, e in d["panels"].items():
        seed, M = map(int, key.split("/"))
        items, _, _ = P.build_panel(CF.make(M, 4), seed)
        if G.sha([[it["prompt"], it["truth"]] for it in items]) != e["sha256"] or items != e["items"]:
            bad.append(key)
    return dict(file="TAUGHT_PANELS.json", panels=len(d["panels"]), items_per_panel=len(next(iter(d["panels"].values()))["items"]),
                reproduced=not bad, not_reproduced=bad, sha256={k: v["sha256"] for k, v in d["panels"].items()},
                identical_at_4_32_256_presentations=all(v["identical_at_4_32_256_presentations"] for v in d["panels"].values()),
                passed=not bad and d["passed"])


def scoring_fixtures():
    rows = {}; ok = True
    for M, Pn in CF.settings():
        cfg = CF.make(M, Pn)
        B = G.build(cfg, W, streams_wanted=("ordinary", "wider", "extra"))
        taught, _ = P.load(W, M)
        for name, (arm, stream) in PATHS.items():
            ev, cuts, end = SC.plan(B, stream, arm, taught)
            refs = SC.references(B, stream, arm, ev)
            res = {}
            for rn, cls in (("oracle", SC.OracleResponder), ("suffix", SC.SuffixResponder)):
                R = SC.readings(B, ev, SC.run(cls(B, stream, arm), ev, end), refs, stream == "wider")
                fs = R["first_encounter_summary"]
                res[rn] = dict(first_right=sum(v["right"] for v in fs.values()), first_n=sum(v["n"] for v in fs.values()),
                               twins_both_right=sum(t[f]["both_right"] for t in R["twins"].values() for f in t),
                               twins=sum(t[f]["renderings"] for t in R["twins"].values() for f in t),
                               distinct_pairs_right_A=sum(R["twins"]["A"][f]["distinct_pairs_right"] for f in R["twins"]["A"]),
                               taught_right=sum(v["right"] for t in R["taught"].values() for v in t.values()),
                               taught_n=sum(v["n"] for t in R["taught"].values() for v in t.values()), cls=SC.classify(R)[0])
            o, s = res["oracle"], res["suffix"]
            good = (o["first_right"] == o["first_n"] and o["twins_both_right"] == o["twins"] and o["taught_right"] == o["taught_n"]
                    and s["twins_both_right"] == 0)
            ok &= good
            rows[f"{M}/{Pn}/{name}"] = dict(passed=good, **res)
    return dict(world=W, rows=rows, passed=ok,
                condition="the oracle responder scores every first-encounter probe, both members of every twin and every taught-panel item; "
                          "the ten-byte-suffix responder scores no twin; both through sz_score's own schedule, references and readings")


def line_distances():
    out = {}
    for Pn in (4, 256):
        cfg = CF.make(97, Pn)
        B = G.build(cfg, W, streams_wanted=("wider",))
        taught, _ = P.load(W, 97)
        ev, cuts, end = SC.plan(B, "wider", "B", taught)
        grid = sorted(set(range(L.CHUNK, end, L.CHUNK)) | set(cuts) | {end})
        refs = SC.references(B, "wider", "B", ev, grid=grid)
        d = {"first_encounter": [], "twin_member": []}
        for e, rr in zip(ev, refs):
            for m, r in zip(e.get("meta", [dict(kind="first")]), rr):
                if r["line_distance"] and m["kind"] in ("first", "twin"):
                    d["first_encounter" if m["kind"] == "first" else "twin_member"].append(r["line_distance"])
        q = lambda v: [sorted(v)[int(f * (len(v) - 1))] for f in (0, .1, .5, .9, 1)] if v else None
        out[f"97/{Pn}"] = {k: dict(n=len(v), **{f"{which}_bytes_quantiles_0_10_50_90_100": q([x[which]["bytes"] for x in v]) for which in ("context_line", "cue_line")},
                                   **{f"{which}_cuts_between_quantiles_0_10_50_90_100": q([x[which]["cuts_between"] for x in v]) for which in ("context_line", "cue_line")})
                           for k, v in d.items()}
    return out


def forecast(gen, paths, sizes):
    fits = []
    for M, Pn in CF.settings():
        sz = sizes[f"{M}/{Pn}"]
        for name, (arm, stream) in PATHS.items():
            if name == "B_extra" and (M, Pn) != (97, 4):
                continue
            pr = paths[name]; s = sz["paths"][name]
            for w in CF.ROSTER_WORLDS:
                train = s["updates"] * pr["cpu_per_update"]
                score = (s["first_encounter_probes"] * pr["cpu_first_encounter_probe"] + 2 * pr["cpu_panel_A_twins_and_taught"] + pr["cpu_panel_I_twins"]
                         + (pr["cpu_references_full_stream"] + pr["cpu_readings"]) * s["bytes"] / pr["bytes"])
                genc = pr["cpu_generation_one_stream"] * sz["episodes"] / pr["episodes"]
                fits.append(dict(arm=arm, stream=stream, modulus=M, presentations=Pn, world=w, bytes=s["bytes"], target_positions=s["targets"],
                                 updates=s["updates"], probe_cuts=s["probe_cuts"], cpu_generation=round(genc, 1), cpu_training=round(train, 1),
                                 cpu_scoring=round(score, 1), cpu_total=round(genc + train + score, 1)))
    checks = [dict(modulus=M, presentations=Pn, world=w, cpu_checks=gen[f"{M}/{Pn}"]["cpu_checks"] + gen[f"{M}/{Pn}"]["cpu_generation"])
              for M, Pn in CF.settings() for w in CF.ROSTER_WORLDS]
    tot = lambda k: round(sum(f[k] for f in fits), 0)
    by = {}
    for f in fits:
        k = f"{f['arm']}_{f['stream']}"
        b = by.setdefault(k, dict(fits=0, bytes=0, target_positions=0, updates=0, cpu_training=0.0, cpu_scoring=0.0, cpu_generation=0.0))
        for x in ("bytes", "target_positions", "updates", "cpu_training", "cpu_scoring", "cpu_generation"):
            b[x] += f[x]
        b["fits"] += 1
    for b in by.values():
        for x in ("cpu_training", "cpu_scoring", "cpu_generation"):
            b[x] = round(b[x])
    cpu_all = tot("cpu_total") + sum(c["cpu_checks"] for c in checks)
    return dict(fits=len(fits), per_fit=fits, by_path=by, learner_free_checks_per_setting_world=checks,
                totals=dict(bytes=tot("bytes"), target_positions=tot("target_positions"), updates=tot("updates"), cpu_generation=tot("cpu_generation"),
                            cpu_checks=round(sum(c["cpu_checks"] for c in checks)), cpu_training=tot("cpu_training"), cpu_scoring=tot("cpu_scoring"),
                            cpu_all=round(cpu_all), cpu_hours=round(cpu_all / 3600, 2), wall_hours_at_three_processes=round(cpu_all / 3600 / 3, 2),
                            budget_cpu_hours=12, within_budget=cpu_all / 3600 <= 12),
                largest_fit=max(fits, key=lambda f: f["cpu_total"]),
                basis="sizes from world 104's own streams at every setting (other worlds differ only in filler bytes, so in bytes by well under 1%); "
                      "CPU per update, per probe and per panel from the four 256-presentation paths profiled on world 104 (500 updates each, run "
                      "beside at most two other processes); checks and generation from world 104 at every setting")


def core():
    T.set_num_threads(1)
    t_all = time.process_time()
    rec = dict(brief="OPUS_TASK_TARGET_SIZING_SWEEP_20261111.md, revision 3", stage=1,
               source_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
               host=dict(python=platform.python_version(), torch=T.__version__, machine=platform.machine(), cores=len(__import__("os").sched_getaffinity(0)),
                         threads_per_process=1),
               note="Stage 1 only. No roster world was trained on and no fit was run. World 104 is used for profiling and fixtures only; "
                    "every learner here was discarded. Nothing is concluded about any learner or rule.")
    rec["1_default_parity"], c1 = timed(parity)
    rec["2_modulus_fixtures"], c2 = timed(MF.run)
    rec["3_taught_panels"], c3 = timed(panels_frozen)
    rec["admission_scoring_fixtures"], c4 = timed(scoring_fixtures)
    rec["4_probe_protocol"], c5 = timed(PF.run)
    rec["line_use_distances_world104"], c6 = timed(line_distances)
    rec["cpu_seconds_this_process"] = dict(parity=c1, modulus=c2, panels=c3, scoring_fixtures=c4, probe_fixtures=c5, line_distances=c6,
                                           total=round(time.process_time() - t_all, 1))
    (RUNS / "stage1_core.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def record():
    rec = json.loads((RUNS / "stage1_core.json").read_text())
    cc = {D: json.loads((RUNS / f"crosscut_D{D}.json").read_text()) for D in (60, 600)}
    diag = {f.stem.replace("crosscut_diag_", ""): json.loads(f.read_text()) for f in sorted(RUNS.glob("crosscut_diag_*.json"))}
    rec["5_cross_cut_dependency"] = dict(registered_D60=cc[60], D600=cc[600], passed=cc[60]["passed"],
                                         diagnostics_after_the_failure=diag,
                                         sanity_D4_300_chunks=(RUNS / "sanity_crosscut_D4.txt").read_text().strip())
    gen = json.loads((RUNS / "profile_gen.json").read_text())
    paths = {n: json.loads((RUNS / f"profile_path_{n}.json").read_text()) for n in PATHS}
    sizes = json.loads((RUNS / "profile_sizes.json").read_text())
    shape = json.loads((RUNS / "verify_shape.json").read_text())
    rec["learner_shape_vs_E20B"] = shape
    rec["6_cost_profile"] = dict(world=W, generation_and_checks=gen, paths=paths, sizes=sizes,
                                 checks_pass_at_every_setting_world104=all(not v["checks_failed"] for k, v in gen.items() if isinstance(v, dict) and "checks_failed" in v))
    rec["forecast_84_fits"] = forecast(gen, paths, sizes)
    other = dict(crosscut_D60=cc[60]["cpu_seconds"], crosscut_D600=cc[600]["cpu_seconds"], **{f"crosscut_diag_{k}": v["cpu_seconds"] for k, v in diag.items()},
                 profile_gen=gen.get("cpu_seconds_this_process"), **{f"profile_{n}": paths[n].get("cpu_seconds_this_process") for n in PATHS},
                 profile_sizes=sizes.get("cpu_seconds_this_process"))
    rec["cpu_seconds_other_processes"] = other
    dev = json.loads((RUNS / "dev_runs_cpu.json").read_text())
    rec["cpu_seconds_development_runs"] = dev
    rec["cpu_seconds_stage1_total"] = round(rec["cpu_seconds_this_process"]["total"] + sum(v for v in other.values() if v)
                                            + sum(dev["runs"].values()), 1)
    rec["cpu_hours_stage1_total"] = round(rec["cpu_seconds_stage1_total"] / 3600, 2)
    rec["dry_runs_of_the_fit_driver"] = dict(files=sorted(f.name for f in (RUNS / "dry_world104").glob("*.json")),
                                             note="world 104, 97/4, arm B wider and arm A ordinary: run only to show the driver works end to end "
                                                  "and refuses to overwrite; not fits, not readings, not to be read as results")
    rec["admission"] = {
        "1 copies reproduce TARGET_STEP0_CHECK.json (seeds 1-3, every value)": rec["1_default_parity"]["passed"],
        "2 modulus fixtures, addition and subtraction at 97, 31, 11; wrong modulus caught; configuration reach": rec["2_modulus_fixtures"]["passed"],
        "3 frozen taught panels reproduce": rec["3_taught_panels"]["passed"],
        "  scoring code: oracle responder scores every probe, suffix responder no twin (world 104, 9 settings, 4 paths)": rec["admission_scoring_fixtures"]["passed"],
        "4 probe protocol on the real GRU (F1-F4)": rec["4_probe_protocol"]["passed"],
        "5 dependency across a cut learnable (D = 60 criterion)": rec["5_cross_cut_dependency"]["passed"],
        "  step 0 learner-free checks at all 9 settings, world 104": rec["6_cost_profile"]["checks_pass_at_every_setting_world104"],
        "  learner = E20-B G1 in shape and initial weights": shape["passed"],
        "6 forecast within 12 CPU-hours": rec["forecast_84_fits"]["totals"]["within_budget"],
    }
    rec["admitted_for_stage_2"] = all(rec["admission"].values())
    (HERE / "SIZING_STAGE1_RECORD.json").write_text(json.dumps(rec, indent=1, default=str))
    for k, v in rec["admission"].items():
        print(("PASS " if v else "FAIL ") + k)
    print("forecast:", json.dumps(rec["forecast_84_fits"]["totals"]))
    return rec


if __name__ == "__main__":
    if sys.argv[1:2] == ["core"]:
        core()
    else:
        sys.exit(0 if record()["admitted_for_stage_2"] else 1)
