"""E39 job L1 (brief OPUS_TASK_E39_SMALL_WORLD_PROTOCOL_L0_L1_20261115.md, revision 2): calibration.

Worlds 1 to 8, lives of 64 visits, arms naive and sweep, at one rung of the ladder:

    R0        default settings (H = U = 64, visits of 256 updates)
    vu512     visits of 512 updates                       (if part 1 fails or the median of part 2 is below 0.10)
    h96vu512  H = U = 96 with visits of 512 updates       (the next rung up)
    vu128     visits of 128 updates                       (if the median of part 2 is above 0.90)

The gate (under naive, in at least six of eight worlds): (1) both tasks of the first group reach the 0.8 criterion within the life;
(2) the median first-visit level of the tasks met after the first group is between 0.10 and 0.90; and the measured cost puts a life of
1,024 visits within 6 CPU-hours.

    python3 sw_l1.py run <rung>      # the 16 lives, at most three processes at a time, into exp45/e39/
    python3 sw_l1.py table <rung>    # scores every life (sw_score.py) and evaluates the gate -> exp45/e39/E39_L1_<rung>.json
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MDIR = HERE.parent
sys.path.insert(0, str(HERE))
import sw_world as W                  # noqa: E402
import sw_score as S                  # noqa: E402

OUT = MDIR / "exp45" / "e39"
WORLDS = range(1, 9)
ARMS = ("naive", "sweep")
VISITS = 64
RUNGS = {"R0": dict(H=64, visit_updates=256), "vu512": dict(H=64, visit_updates=512), "h96vu512": dict(H=96, visit_updates=512),
         "vu128": dict(H=64, visit_updates=128)}
LOW, HIGH, NEED_WORLDS, BOUND_HOURS = 0.10, 0.90, 6, 6.0


def stem(arm, world, rung):
    r = RUNGS[rung]; tag = []
    if r["H"] != 64:
        tag += [f"h{r['H']}", f"u{r['H']}"]
    if r["visit_updates"] != 256:
        tag.append(f"vu{r['visit_updates']}")
    return f"sw_{arm}_w{world}_v{VISITS}" + ("_" + "_".join(tag) if tag else "")


def args(arm, world, rung):
    r = RUNGS[rung]
    a = [arm, str(world), str(VISITS)]
    if r["H"] != 64:
        a += ["--H", str(r["H"])]
    if r["visit_updates"] != 256:
        a += ["--visit-updates", str(r["visit_updates"])]
    return a


def run(rung):
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    jobs = [(arm, w) for w in WORLDS for arm in ARMS]
    running = []
    while jobs or running:
        while jobs and len(running) < 3:
            arm, w = jobs.pop(0)
            log = OUT / "logs" / f"{stem(arm, w, rung)}.log"
            assert not log.exists(), log
            c = [sys.executable, str(HERE / "sw_run.py"), "life", *args(arm, w, rung), "--out", str(OUT)]
            running.append((subprocess.Popen(c, stdout=open(log, "w"), stderr=subprocess.STDOUT), c, log))
        time.sleep(2)
        for r in list(running):
            if r[0].poll() is not None:
                running.remove(r)
                assert r[0].returncode == 0, f"failed: {' '.join(r[1])} (log {r[2]})"
                print("done", r[2].name, flush=True)


def table(rung):
    path = OUT / f"E39_L1_{rung}.json"
    assert not path.exists(), path
    lives, gate_rows = {}, []
    for w in WORLDS:
        sym = S.symmetric_tasks(w)
        world = W.World(w, RUNGS[rung]["visit_updates"])
        gen = W.digest(world, VISITS)
        tasks = {t["id"]: dict(symbol=chr(t["symbol"]), op=t["op"][0] if t["op"][0] != "lin" else f"{t['op'][1][0]:+d}a{t['op'][1][1]:+d}b", p=t["p"])
                 for t in world.tasks}
        for arm in ARMS:
            s = stem(arm, w, rung)
            rec = json.loads((OUT / f"{s}.json").read_text())
            f = np.load(OUT / f"{s}.npz"); ev = {k: f[k] for k in f.files}
            tp = OUT / f"{s}.score.json"
            sc = S.score(ev, VISITS, sym)
            if not tp.exists():
                tp.write_text(json.dumps(sc, indent=1))
            fv = sc["first_visit"]
            first = [d for d in fv if d["n"] <= 2]; after = [d["level"] for d in fv if d["n"] >= 3]
            cost = {d["n"]: d["cost"] for d in sc["cost"]}
            p1 = cost.get(1) is not None and cost.get(2) is not None
            med = float(np.median(after)) if after else None
            p2 = med is not None and LOW <= med <= HIGH
            fg_eov = [dict(visit=e["visit"], task=e["task"], level=e["level"]) for e in sc["end_of_visit"] if e["n"] <= 2]
            lives[s] = dict(arm=arm, world=w, receipt=f"{s}.json", events=f"{s}.npz", score_table=tp.name,
                            stream_sha256=rec["stream_sha256"], generator_digest=gen, digest_equal=rec["stream_sha256"] == gen,
                            final=rec["final"], cpu_per_visit=rec["cpu"]["mean_per_visit"], forecast_1024_hours=rec["cpu"]["mean_per_visit"] * 1024 / 3600,
                            groups_met=sum(x["new"] for x in rec["schedule"]), tasks_met=len(fv),
                            part1=p1, first_group_cost=[cost.get(1), cost.get(2)], median_after_first_group=med, part2=p2,
                            mean_first_group_level=float(np.mean([d["level"] for d in first])), mean_after_first_group=float(np.mean(after)) if after else None,
                            mean_score=sc["mean_score"], late_level=sc["late_level"], score_trend=sc["score_trend"], learning_trend=sc["learning_trend"],
                            cost_trend=sc["cost_trend"], kept_pooled=sc["kept_pooled"], symmetric=sc["symmetric"],
                            first_visit=[dict(d, **tasks[d["task"]]) for d in fv], first_group_end_of_visit=fg_eov,
                            score_by_age=sc["score_by_age"], blind_by_age=sc["blind_by_age"])
            if arm == "naive":
                gate_rows.append(dict(world=w, part1=p1, median=med, part2=p2, both=p1 and p2))
    naive_cost = max(v["forecast_1024_hours"] for v in lives.values() if v["arm"] == "naive")
    n_both = sum(r["both"] for r in gate_rows)
    gate = dict(rows=gate_rows, worlds_passing=n_both, part1_worlds=sum(r["part1"] for r in gate_rows),
                part2_worlds=sum(r["part2"] for r in gate_rows), below_low=sum(r["median"] is not None and r["median"] < LOW for r in gate_rows),
                above_high=sum(r["median"] is not None and r["median"] > HIGH for r in gate_rows),
                forecast_1024_hours_naive_max=naive_cost, cost_ok=naive_cost <= BOUND_HOURS,
                passed=n_both >= NEED_WORLDS and naive_cost <= BOUND_HOURS)
    out = dict(experiment="E39", job="L1", rung=rung, settings=RUNGS[rung], visits=VISITS, worlds=list(WORLDS), arms=list(ARMS), gate=gate,
               lives=lives, sources_sha256={p.name: __import__("hashlib").sha256(p.read_bytes()).hexdigest()
                                            for p in sorted(HERE.glob("sw_*.py"))})
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(gate, indent=1))
    return out


if __name__ == "__main__":
    {"run": run, "table": table}[sys.argv[1]](sys.argv[2])
