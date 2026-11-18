"""E44, job L2, the sweep's long lives: the six checks before the lives (section 5 of the reviewer's instructions
WORKER_E44_L2_SWEEP_LONG_LIVES.md of 17 November 2026).

    python3 l2_sweep_checks.py report     # reads the check lives -> exp45/e44/L2/sweep/checks/E44_L2_SWEEP_CHECKS.json (refused if it exists)

  1  sw_run.py has the SHA-256 that the main worker's check receipt records (exp45/e44/L2/checks/E44_L2_CHECKS.json, sources_sha256);
     the scorer's hash and the commit this work starts from are recorded
  2  world 1's eight-visit calibration life under `sweep` (in turn, 256 updates, the defaults of the two new arguments) gives on this host
     the hashes of E44's receipt exp45/e44/F1/lw_sweep_w1_v8_turn.json
  3  `sweep+replay` with rho = 0 equals `sweep` on world 1 for one visit
  4  resume: a `sweep+replay` life of world 1 stopped at visit 4 of 8 and resumed gives the uninterrupted hashes and event file
  5  the three births under the sweep: file hashes equal their E41 receipts; A and R through the runner equal the receipts' final rows
  6  cost: CPU-seconds a visit for each arm, and a forecast for the eight lives
Nothing here trains.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MDIR = HERE.parent
ROOT = MDIR.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(MDIR))
import sw_run as R                    # noqa: E402

SW = MDIR / "exp45" / "e44" / "L2" / "sweep"
CHK = SW / "checks"
F1 = MDIR / "exp45" / "e44" / "F1"
MAIN_CHECKS = MDIR / "exp45" / "e44" / "L2" / "checks" / "E44_L2_CHECKS.json"
SEEDS = (73129, 93683, 116089)
START_COMMIT = "b218f351"             # the main worker's branch claude/e44-l2-first-long-lives as this work found it


def rec(d, stem):
    return json.loads((CHK / d / f"{stem}.json").read_text())


def npz(p):
    f = np.load(p)
    return {k: f[k] for k in f.files}


def hashes(r):
    return dict(list_arrays=r["list"]["arrays_sha256"], events_arrays=r["events"]["arrays_sha256"], weights=r["final"]["weights_tensors_sha256"],
                optimizer=r["final"]["optimizer_sha256"], learned=r["learned_sha256"], stream=r["stream_sha256"])


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def check1():
    main = json.loads(MAIN_CHECKS.read_text())["sources_sha256"]
    now = R.source_hashes()
    start = git("rev-parse", START_COMMIT)
    at_start = lambda rel: R.hashlib.sha256(subprocess.run(["git", "show", f"{start}:{rel}"], cwd=ROOT, capture_output=True, check=True).stdout).hexdigest()
    files = {f"smallworld/{n}": R.sha_file(HERE / n) for n in ("l2_score.py", "sw_score.py", "l2_driver.py", "sw_run.py")}
    same_as_start = {k: v == at_start(f"MOD97_LOCAL_RULE_20261106/{k}") for k, v in files.items()}
    return dict(passed=now == main and all(same_as_start.values()), sw_run_sha256=now["smallworld/sw_run.py"],
                main_workers_receipt=dict(file=str(MAIN_CHECKS.relative_to(MDIR)), file_sha256=R.sha_file(MAIN_CHECKS), sources_sha256=main),
                sources_sha256_now=now, every_source_equal=now == main,
                scorer=dict(file="smallworld/l2_score.py", sha256=files["smallworld/l2_score.py"], commit_started_from=start,
                            branch_started_from="claude/e44-l2-first-long-lives"),
                unchanged_files_sha256=files, equal_to_the_commit_started_from=same_as_start,
                note="The runner's twelve source files have the hashes of the main worker's check receipt. The scorer (l2_score.py) was "
                     "extended by the main worker after that receipt; it is used as it stands in the commit this work starts from.")


def check2():
    new = rec("c2_calib", "lw_sweep_w1_v8_turn")
    e44 = json.loads((F1 / "lw_sweep_w1_v8_turn.json").read_text())
    hn, he = hashes(new), hashes(e44)
    ea, eb = npz(CHK / "c2_calib" / "lw_sweep_w1_v8_turn.npz"), npz(F1 / "lw_sweep_w1_v8_turn.npz")
    la, lb = npz(CHK / "c2_calib" / "lw_sweep_w1_v8_turn_list.npz"), npz(F1 / "lw_sweep_w1_v8_turn_list.npz")
    ev_equal = sorted(ea) == sorted(eb) and all(ea[k].shape == eb[k].shape and (ea[k] == eb[k]).all() for k in ea)
    li_equal = sorted(la) == sorted(lb) and all(la[k].shape == lb[k].shape and (la[k] == lb[k]).all() for k in la)
    differ = [k for k in hn if hn[k] != he[k]]
    return dict(passed=hn == he and ev_equal, this_host=dict(command=new["provenance"]["command"], hashes=hn, host=new["provenance"]["host"],
                                                             settings=dict(list_every=new["settings"]["list_every"], decay_distances=new["settings"]["decay_distances"])),
                E44_committed=dict(receipt="exp45/e44/F1/lw_sweep_w1_v8_turn.json", hashes=he, host=e44["provenance"]["host"]),
                hashes_that_differ=differ, event_columns_equal=ev_equal, list_columns_equal=li_equal,
                events_right_this_host=int(ea["correct"].sum()), events_right_E44=int(eb["correct"].sum()), entries=int(len(ea["correct"])),
                note="World 1, 8 visits in turn at 256 updates a visit, `sweep`, from the birth of seed 73129 under the sweep, with the defaults of "
                     "--list-every and --decay-distances. Compared with E44's committed receipt: the list's arrays, the events' arrays, the final "
                     "weights, the optimizer state, the learned batches and the stream.")


def check3():
    a, b = rec("c3_sweep", "lw_sweep_w1_v1"), rec("c3_sweep_replay_rho0", "lw_sweep+replay_w1_v1_rho0")
    ha, hb = hashes(a), hashes(b)
    ea, eb = npz(CHK / "c3_sweep" / "lw_sweep_w1_v1.npz"), npz(CHK / "c3_sweep_replay_rho0" / "lw_sweep+replay_w1_v1_rho0.npz")
    ev_equal = all((ea[k] == eb[k]).all() for k in ea)
    return dict(passed=ha == hb and ev_equal and a["local"] and b["local"] and b["learner"]["rule"] == "S",
                sweep=dict(command=a["provenance"]["command"], hashes=ha, local=a["local"]),
                sweep_replay_rho0=dict(command=b["provenance"]["command"], hashes=hb, local=b["local"], replay_items_per_update=b["replay_items_per_update"],
                                       learner_rule=b["learner"]["rule"], ledger_buffer=b["ledger"]["buffer"]),
                event_columns_equal=ev_equal,
                note="`sweep+replay` with rho = 0 loads the sweep's birth and learns from the 48 new items of every update: the weights, the "
                     "optimizer state, the event file's arrays, the learned batches and the list are those of `sweep`.")


def check4():
    stem = "lw_sweep+replay_w1_v8_rho0.5"
    a, b = rec("c4_whole", stem), rec("c4_cut", stem)
    ha, hb = hashes(a), hashes(b)
    la, lb = npz(CHK / "c4_whole" / f"{stem}_list.npz"), npz(CHK / "c4_cut" / f"{stem}_list.npz")
    ea, eb = npz(CHK / "c4_whole" / f"{stem}.npz"), npz(CHK / "c4_cut" / f"{stem}.npz")
    ev_equal, li_equal = all((ea[k] == eb[k]).all() for k in ea), all((la[k] == lb[k]).all() for k in la)
    buf_ok = all(r["ledger"]["buffer"]["capacity_items"] == 55209 and r["ledger"]["buffer"]["max_items"] <= 55209 for r in (a, b))
    return dict(passed=ha == hb and ev_equal and li_equal and buf_ok and len(b["resumes"]) == 1 and b["resumes"][0]["done_visits"] == 4 and not a["resumes"],
                whole=dict(command=a["provenance"]["command"], hashes=ha, ledger_buffer=a["ledger"]["buffer"], counters=a["counters"]),
                stopped_and_resumed=dict(command=b["provenance"]["command"], resumes=b["resumes"], hashes=hb, ledger_buffer=b["ledger"]["buffer"],
                                         counters=b["counters"]),
                list_columns_equal=li_equal, event_columns_equal=ev_equal, store_capacity_is_55209_items_and_not_exceeded=buf_ok,
                note="A `sweep+replay` life of world 1 (rho 0.5, decay 0.25), 8 visits, --list-every 2 --decay-distances 1,2,4: whole, and stopped "
                     "after visit 4 (--stop-after 4) then continued from its state with `life --resume`. Equal: the final weights, the optimizer "
                     "state, the event file's arrays, the learned batches, the stream and the list file's arrays.")


def check5():
    import arch26
    E = arch26.E
    out, ok = {}, True
    for seed in SEEDS:
        p = MDIR / "exp45" / "e41" / f"e41_S_h256_f50_{seed}.pt"
        r = json.loads(p.with_suffix(".json").read_text())
        last = r["curve"][-1]
        m, info = R.load_learner(p, 256, 128, "sweep")
        m2, info2 = R.load_learner(p, 256, 128, "sweep+replay")
        rit = R.e41_r_items(0.5, seed, info["reduction_sample_sha256"])
        g = R.answer_all(m, rit)
        rc = {"add89": sum(x == it[4] for x, it in zip(g[:1000], rit[:1000])), "sub89": sum(x == it[4] for x, it in zip(g[1000:], rit[1000:]))}
        ac = {}
        for t in E.DEV:
            held = E.split(t, 0.5, seed)[1]
            items = E.panel(t, held, seed, ks=(12,), n=300)[12]
            gg = R.answer_all(m, [(0, 0, 0, pr, y) for pr, y in items])
            ac[E.NAME(*t)] = sum(x == y for x, (pr, y) in zip(gg, items))
        d = dict(seed=seed, rule=r["rule"], local_in_birth=r["local"], file_sha256=R.sha_file(p), receipt_weights_sha256=r["weights_sha256"],
                 tensors_sha256=info["tensors_sha256"], receipt_tensors_sha256=r["tensors_sha256"],
                 receipt_final_row=dict(step=last["step"], A=last["A"], R=last["R"], A_counts=last["A_counts"], R_counts=last["R_counts"]),
                 runner=dict(cls="sw_net.SWNet", R_counts=rc, A_counts=ac, R=100 * (rc["add89"] + rc["sub89"]) / 2000,
                             A=100 * (ac["add89"] + ac["sub89"]) / 600),
                 sweep_replay_arm_loads_it=info2["tensors_sha256"] == info["tensors_sha256"])
        d["file_hash_equal"] = d["file_sha256"] == d["receipt_weights_sha256"]
        d["runner_equal"] = (d["tensors_sha256"] == d["receipt_tensors_sha256"] and rc == last["R_counts"]
                             and all(ac[k] == last["A_counts"][k] for k in ac) and abs(d["runner"]["A"] - last["A"]) < 1e-9 and abs(d["runner"]["R"] - last["R"]) < 1e-9)
        ok &= d["file_hash_equal"] and d["runner_equal"] and d["sweep_replay_arm_loads_it"] and r["rule"] == "S" and r["local"] is True
        out[p.name] = d
    return dict(passed=bool(ok), births=out,
                note="E41's half-width births under the sweep at taught fraction 0.50 (H = 256, U = 128), seeds 73129, 93683 and 116089. The file's "
                     "SHA-256 and the tensors hash equal the E41 receipt's. Through the runner (sw_net.SWNet(256, 128) loaded by sw_run.load_learner, "
                     "answered by sw_net.answer): R on E41's two reduction samples of 1,000 held-out pairs each, and A on E41's panels of 300 "
                     "held-out items of add89 and of sub89 (A is their mean, as the receipt's), equal the receipt's final row. A_counts of max89 "
                     "and min89 through the runner are recorded where the panel gives them.")


def check6():
    arms = {}
    for name, d, stem in (("sweep", "c6_sweep", "lw_sweep_w1_v4"), ("sweep+replay", "c6_sweep_replay", "lw_sweep+replay_w1_v4_rho0.5")):
        r = rec(d, stem)
        arms[name] = dict(cpu_seconds_per_visit=r["cpu"]["per_visit"], mean=r["cpu"]["mean_per_visit"], per_update=r["cpu"]["mean_per_visit"] / 256,
                          list_cpu=r["cpu"]["list_total"], list_entries=r["list"]["entries"], host=r["provenance"]["host"], command=r["provenance"]["command"])
    c2 = rec("c2_calib", "lw_sweep_w1_v8_turn")
    per_item = c2["cpu"]["list_total"] / c2["list"]["entries"]
    main = json.loads(MAIN_CHECKS.read_text())["checks"]["7"]
    items = 1238400                                              # the audit's section 7: world 1, 1,024 visits, --list-every 8 --decay-distances none
    life = {a: 1024 * arms[a]["mean"] / 3600 + items * per_item / 3600 for a in arms}
    total = 4 * sum(life.values())
    return dict(passed=True, arms=arms, list_cpu_seconds_per_item=per_item, list_items_of_a_life=items,
                life_of_1024_visits_cpu_hours=life, forecast_eight_lives_cpu_hours=total,
                forecast_wall_hours_three_at_once=total / 3,
                main_workers_check_7=dict(naive_mean_cpu_seconds_per_visit=main["arms"]["naive"]["mean"], replay_mean=main["arms"]["replay"]["mean"],
                                          life_of_1024_visits_cpu_hours=main["life_of_1024_visits_cpu_hours"]),
                note="Each arm: world 1, 4 visits with the lives' settings (--list-every 8 --decay-distances none), one process alone on the host. "
                     "The list's cost is forecast from the seconds an item of check 2's life (run beside two other processes) and the item count of "
                     "world 1 in the audit's section 7. The store is not full in 4 visits. The lives run three at once, where a visit takes longer "
                     "than alone; the forecast is of CPU time measured alone and is a lower bound of the CPU time of lives run three at once.")


def report():
    to = CHK / "E44_L2_SWEEP_CHECKS.json"
    R.refuse_existing(to)
    out = dict(experiment="E44", job="L2, the sweep's long lives", instructions="PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/"
               "WORKER_E44_L2_SWEEP_LONG_LIVES.md on claude/vibrant-cerf-sje726", instructions_commit=git("rev-parse", "origin/claude/vibrant-cerf-sje726"),
               host=R.host(), git=R.git_state(), sources_sha256=R.source_hashes(),
               checks={"1": check1(), "2": check2(), "3": check3(), "4": check4(), "5": check5(), "6": check6()})
    out["all_passed"] = all(c["passed"] for c in out["checks"].values())
    R.write_json(to, out)
    for k, v in out["checks"].items():
        print(k, "passed" if v["passed"] else "FAILED")
    print("all passed:", out["all_passed"])


if __name__ == "__main__":
    {"report": report}[sys.argv[1]]()
