"""E60 (brief TASK_E60_..._20261118.md, revision 3, on claude/vibrant-cerf-sje726 at 9bb87033): the six checks before the four lives
(section 5 of the brief).

    python3 l2_san_checks.py report     # reads the check lives -> exp45/e44/L2/sweep_after_naive/checks/E60_CHECKS.json (refused if it exists)

The check lives (exp45/e44/L2/sweep_after_naive/checks/; their jobs in check_jobs.json, run by l2_driver.py, except c5_cut, run by
l2_san_cut.sh, and the refused start, run by hand with its output in c2_refused.txt):
  c2_w1 .. c2_w4   one visit of `sweep-after-naive` in each world with the birth of rule B, the command of section 4 with 1 visit
  c2_refused.txt   the same command for world 1 with the sweep's birth e41_S_h256_f50_73129.pt (the runner's refusal)
  c3_calib         world 1, 8 visits in turn, 256 updates a visit, the defaults of --list-every and --decay-distances
  c5_whole         world 1, 8 visits, --list-every 8 --decay-distances none --save-states 4,8, whole (run alone on the host: check 6)
  c5_cut           the same, killed with signal 9 after the line of visit 5 and before that of visit 6, resumed from the state after 4 visits
The main worker's lives of `naive` are read from origin/claude/e44-l2-first-long-lives with `git show`. Nothing here trains.
"""
import hashlib
import io
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

SAN = MDIR / "exp45" / "e44" / "L2" / "sweep_after_naive"
CHK = SAN / "checks"
SWEEP = MDIR / "exp45" / "e44" / "L2" / "sweep"
W1 = "origin/claude/e44-l2-first-long-lives"
SEEDS = {1: 73129, 2: 93683, 3: 116089, 4: 73129}


def git(*a, text=True):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=text, check=True).stdout


def npz_bytes(b):
    f = np.load(io.BytesIO(b))
    return {k: f[k] for k in f.files}


def npz(p):
    f = np.load(p)
    return {k: f[k] for k in f.files}


def rec(d, stem):
    return json.loads((CHK / d / f"{stem}.json").read_text())


def hashes(r):
    return dict(list_arrays=r["list"]["arrays_sha256"], events_arrays=r["events"]["arrays_sha256"], weights=r["final"]["weights_tensors_sha256"],
                optimizer=r["final"]["optimizer_sha256"], learned=r["learned_sha256"], stream=r["stream_sha256"])


def naive(w):
    base = f"MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w{w}_v1024"
    r = json.loads(git("show", f"{W1}:{base}.json"))
    return r, base


def check1():
    prev = json.loads((SWEEP / "checks" / "E44_L2_SWEEP_CHECKS.json").read_text())
    want = dict(prev["sources_sha256"])
    now = R.source_hashes()
    unchanged = prev["checks"]["1"]["unchanged_files_sha256"]
    now_unchanged = {k: R.sha_file(MDIR / k) for k in unchanged}
    new = {f"smallworld/{p.name}": R.sha_file(p) for p in sorted(HERE.glob("l2_san*"))}
    return dict(passed=now == want and now_unchanged == unchanged, sources_sha256_recorded=want, sources_sha256_now=now,
                unchanged_files_recorded=unchanged, unchanged_files_now=now_unchanged, new_scripts_sha256=new,
                recorded_in="exp45/e44/L2/sweep/checks/E44_L2_SWEEP_CHECKS.json")


def check2():
    out, ok = {}, True
    for w in (1, 2, 3, 4):
        r = rec(f"c2_w{w}", f"lw_sweep-after-naive_w{w}_v1")
        n, base = naive(w)
        eq = r["learner"] == n["learner"]
        diff = sorted(k for k in set(r["learner"]) | set(n["learner"]) if r["learner"].get(k) != n["learner"].get(k))
        out[f"world {w}"] = dict(command=r["provenance"]["command"], learner=r["learner"], naive_receipt=f"{W1}:{base}.json",
                                 learner_equal_key_for_key=eq, keys_that_differ=diff, local=r["local"], arm=r["settings"]["arm"])
        ok &= eq and r["local"] is True and r["settings"]["arm"] == "sweep-after-naive"
    txt = (CHK / "c2_refused.txt").read_text()
    refused = "AssertionError: arm sweep-after-naive lives from E41's rule B, not S" in txt and txt.strip().endswith("exit 1")
    out["refusal of the sweep's birth (world 1)"] = dict(refused=refused, text=txt.strip().splitlines()[-2:],
                                                          command="python3 sw_run.py life sweep-after-naive 1 1 ... --learner <abs. path of e41_S_h256_f50_73129.pt> "
                                                                  "... --out <abs. path of checks/c2_refused>")
    return dict(passed=bool(ok and refused), worlds=out, note="The runner does not compare the birth's seed with the world; each long life's "
                "receipt is compared in the same way with that of `naive` in the report.")


def check3():
    r = rec("c3_calib", "lw_sweep-after-naive_w1_v8_turn")
    e = json.loads((MDIR / "exp45/e44/F1/lw_sweep-after-naive_w1_v8_turn.json").read_text())
    hr, he = hashes(r), hashes(e)
    ea, eb = npz(CHK / "c3_calib" / "lw_sweep-after-naive_w1_v8_turn.npz"), npz(MDIR / "exp45/e44/F1/lw_sweep-after-naive_w1_v8_turn.npz")
    ev_equal = sorted(ea) == sorted(eb) and all((ea[k] == eb[k]).all() for k in ea)
    return dict(passed=hr == he, this_host=dict(command=r["provenance"]["command"], hashes=hr, settings=dict(list_every=r["settings"]["list_every"],
                decay_distances=r["settings"]["decay_distances"])), E44_receipt=dict(file="exp45/e44/F1/lw_sweep-after-naive_w1_v8_turn.json", hashes=he),
                hashes_that_differ=[k for k in hr if hr[k] != he[k]], event_columns_equal=ev_equal,
                right_this_host=int(ea["correct"].sum()), right_E44=int(eb["correct"].sum()))


def check4():
    out, ok = {}, True
    cols = ("task", "a", "b", "g1", "g2", "g3", "correct")
    for w in (1, 2, 3, 4):
        r = rec(f"c2_w{w}", f"lw_sweep-after-naive_w{w}_v1")
        la = npz(CHK / f"c2_w{w}" / f"lw_sweep-after-naive_w{w}_v1_list.npz")
        n, base = naive(w)
        lb = npz_bytes(git("show", f"{W1}:{base}_list.npz", text=False))
        sa = (la["at_visit"] == 65535) & (la["source"] == 0); sb = (lb["at_visit"] == 65535) & (lb["source"] == 0)
        rows_equal = int(sa.sum()) == 2000 and int(sb.sum()) == 2000 and all((la[c][sa] == lb[c][sb]).all() for c in cols)
        st = r["by_visit"][0]["stream_sha256"] == n["by_visit"][0]["stream_sha256"]
        out[f"world {w}"] = dict(rows=int(sa.sum()), rows_naive=int(sb.sum()), rows_equal=rows_equal, right_before_visit_0=int(la["correct"][sa].sum()),
                                 right_before_visit_0_naive=int(lb["correct"][sb].sum()), stream_sha256_visit_0=r["by_visit"][0]["stream_sha256"],
                                 stream_sha256_visit_0_equal=st)
        ok &= rows_equal and st
    return dict(passed=bool(ok), worlds=out, columns=list(cols))


def check5():
    stem = "lw_sweep-after-naive_w1_v8"
    a, b = rec("c5_whole", stem), rec("c5_cut", stem)
    ha, hb = hashes(a), hashes(b)
    ha["events_file_sha256"], hb["events_file_sha256"] = a["events"]["file_sha256"], b["events"]["file_sha256"]
    ha["list_file_sha256"], hb["list_file_sha256"] = a["list"]["file_sha256"], b["list"]["file_sha256"]
    kill = (CHK / "c5_cut_kill.txt").read_text().strip().splitlines()
    valid = not any("not valid" in x for x in kill) and any("last visit line printed: visit 5" in x for x in kill)
    return dict(passed=ha == hb and valid and len(b["resumes"]) == 1 and b["resumes"][0]["done_visits"] == 4 and not a["resumes"],
                whole=dict(command=a["provenance"]["command"], hashes=ha), killed_and_resumed=dict(command=b["provenance"]["command"], resumes=b["resumes"],
                hashes=hb, kill_record=kill), script="smallworld/l2_san_cut.sh")


def check6():
    a = rec("c5_whole", "lw_sweep-after-naive_w1_v8")
    prev = json.loads((SWEEP / "checks" / "E44_L2_SWEEP_CHECKS.json").read_text())["checks"]["6"]
    per = a["cpu"]["mean_per_visit"]
    sweep_lives = [json.loads((SWEEP / "lives" / f"lw_sweep_w{w}_v1024.json").read_text())["cpu"]["total"] / 3600 for w in (1, 2, 3, 4)]
    ratio = per / prev["arms"]["sweep"]["mean"]
    forecast = sum(sweep_lives) * ratio
    return dict(passed=forecast <= 19, cpu_seconds_per_visit=a["cpu"]["per_visit"], mean_per_visit=per, host=a["provenance"]["host"],
                sweep_mean_per_visit_E44_L2_sweep_check_6=prev["arms"]["sweep"]["mean"], ratio=ratio,
                sweep_lives_cpu_hours=sweep_lives, forecast_four_lives_cpu_hours=forecast, limit=19,
                note="c5_whole ran alone on the host (8 visits of world 1; the list answered at visit 8 only). The forecast scales the CPU-hours of "
                     "the four lives of the arm `sweep` (same worlds, settings and list) by the ratio of the CPU-seconds a visit of this arm to that "
                     "of `sweep` measured alone in the earlier check 6.")


def report():
    to = CHK / "E60_CHECKS.json"
    R.refuse_existing(to)
    out = dict(experiment="E60", brief="PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/TASK_E60_THE_SWEEPS_LONG_LIFE_FROM_BACK_PROPAGATIONS_BIRTH_20261118.md",
               brief_revision=3, brief_commit="9bb87033", main_worker_ref=W1, main_worker_commit=git("rev-parse", W1).strip(),
               host=R.host(), git=R.git_state(),
               checks={"1": check1(), "2": check2(), "3": check3(), "4": check4(), "5": check5(), "6": check6()})
    out["all_passed"] = all(c["passed"] for c in out["checks"].values())
    R.write_json(to, out)
    for k, v in out["checks"].items():
        print(k, "passed" if v["passed"] else "FAILED")
    c = out["checks"]
    print("forecast", round(c["6"]["forecast_four_lives_cpu_hours"], 2), "per visit", round(c["6"]["mean_per_visit"], 2), "differ3", c["3"]["hashes_that_differ"])


if __name__ == "__main__":
    {"report": report}[sys.argv[1]]()
