"""E44 (brief OPUS_TASK_E44_LONG_LIFE_WORLD_IN_THE_FULL_SCALE_FORMAT_20261116.md, revision 2), job F1: the calibration.

    python3 lw_f1.py run <set>      # sets: gate (48 lives), san (24, sweep-after-naive), wd0 (24, naive with decay 0 in the life),
                                    #   step32 / stepts05 / step512 (16 each, the named further steps), lap2 <visit length> (16 branches)
    python3 lw_f1.py table          # every life scored -> exp45/e44/F1/E44_F1.json, E44_F1_TABLES.md and CSVs (refused if they exist)

Every life: worlds 1 to 8, eight new families in turn (--in-turn, 8 visits), seed 73129's half-width E41 births (naive and
sweep-after-naive: B; sweep: S), learning rate 0.001 and decay 0.25 unless the set says otherwise, a state after 8 visits only. At most
three processes at a time, one thread each.

The gate (per visit length, under naive): in at least six of eight worlds the median over the second to the eighth family taken (families
1 to 7) of the first-visit level on reduction pairs (last quarter of the first visit) is between 0.20 and 0.90; and a life of 1,024
visits at that length is forecast within 6 CPU-hours. The shortest length that passes is adopted.
"""
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MDIR = HERE.parent
sys.path.insert(0, str(HERE))
import lw_world as LW                 # noqa: E402
import sw_score as S                  # noqa: E402

OUT = MDIR / "exp45" / "e44" / "F1"
LAP = OUT / "lap2"
BIRTHS = {"B": MDIR / "exp45" / "e41" / "e41_B_h256_f50_73129.pt", "S": MDIR / "exp45" / "e41" / "e41_S_h256_f50_73129.pt"}
RULE = {"naive": "B", "sweep": "S", "sweep-after-naive": "B"}
WORLDS, LENGTHS, VISITS = range(1, 9), (64, 128, 256), 8
LO, HI, NEED, BOUND_H = 0.20, 0.90, 6, 6.0
SETS = {
    "gate": [(arm, vu, []) for vu in (256, 128, 64) for arm in ("naive", "sweep")],
    "san": [("sweep-after-naive", vu, []) for vu in (256, 128, 64)],
    "wd0": [("naive", vu, ["--wd", "0"]) for vu in (256, 128, 64)],
    "step32": [(arm, 32, []) for arm in ("naive", "sweep")],
    "stepts05": [(arm, 64, ["--taught-share", "0.05"]) for arm in ("naive", "sweep")],
    "step512": [(arm, 512, []) for arm in ("naive", "sweep")],
}


def stem(arm, w, vu, extra):
    tags = []
    if vu != 256:
        tags.append(f"vu{vu}")
    if "--wd" in extra:
        tags.append(f"wd{float(extra[extra.index('--wd') + 1]):g}")
    if "--taught-share" in extra:
        tags.append(f"ts{float(extra[extra.index('--taught-share') + 1]):g}")
    return f"lw_{arm}_w{w}_v{VISITS}_" + "_".join(tags + ["turn"])


def jobs(name, vu_lap=None):
    if name == "lap2":
        out = []
        for w in WORLDS:
            for arm in ("naive", "sweep"):
                st = OUT / f"{stem(arm, w, vu_lap, [])}_state_v8.pt"
                assert st.exists(), st
                out.append((["branch", str(st), "--groups", ",".join(str(g) for g in range(8)), "--out", str(LAP)], LAP / "logs" / f"{st.stem}_lap2.log"))
        return out
    out = []
    for arm, vu, extra in SETS[name]:
        for w in WORLDS:
            s = stem(arm, w, vu, extra)
            assert not (OUT / f"{s}.json").exists(), s
            out.append((["life", arm, str(w), str(VISITS), "--world", "lw", "--learner", str(BIRTHS[RULE[arm]]), "--in-turn", "--visit-updates", str(vu),
                         "--save-states", str(VISITS), "--out", str(OUT)] + extra, OUT / "logs" / f"{s}.log"))
    return out


def run(name, vu_lap=None):
    todo = jobs(name, vu_lap)
    running = []
    while todo or running:
        while todo and len(running) < 3:
            args, log = todo.pop(0)
            log.parent.mkdir(parents=True, exist_ok=True)
            assert not log.exists(), log
            running.append((subprocess.Popen([sys.executable, str(HERE / "sw_run.py")] + args, stdout=open(log, "w"), stderr=subprocess.STDOUT), args, log))
        time.sleep(2)
        for r in list(running):
            if r[0].poll() is not None:
                running.remove(r)
                assert r[0].returncode == 0, f"failed: {r[1]} (log {r[2]})"
                print("done", r[2].name, flush=True)


# ---------------------------------------------------------------- scoring
def score_life(rec, d=OUT):
    s = rec["settings"]; vu = s["visit_updates"]
    w = LW.World(s["world"], vu, s["p_new"], s["taught_share"], s["lowest"])
    ev = S.load(d / rec["events"]["file"]); la = S.load(d / rec["list"]["file"])
    visits = [(r["visit"], r["group"]) for r in rec["by_visit"]]
    first = S.lw_first_visits(ev, w, vu, visits)
    kept = S.lw_kept(la)
    return dict(life=rec["events"]["file"][:-4], arm=s["arm"], world=s["world"], visit_updates=vu, wd=s["wd"], taught_share=s["taught_share"],
                first=first, gate_median=S.lw_gate_median(first), trend=S.lw_trend(first), by_visit=S.lw_by_visit(ev, visits), kept=kept,
                cpu_per_visit=rec["cpu"]["per_visit"], list_cpu_per_visit=rec["cpu"]["list_per_visit"], cpu_total=rec["cpu"]["total"],
                peak_rss_kib=rec["cpu"]["peak_rss_kib"], stream_sha256=rec["stream_sha256"], learner_sha256=rec["learner"]["file_sha256"])


def gate(lives, vu, wd=0.25, ts=0.10):
    nv = [x for x in lives if x["arm"] == "naive" and x["visit_updates"] == vu and x["wd"] == wd and x["taught_share"] == ts]
    if len(nv) != 8:
        return None
    meds = {x["world"]: x["gate_median"] for x in sorted(nv, key=lambda x: x["world"])}
    inside = [w for w, m in meds.items() if LO <= m <= HI]
    above = [w for w, m in meds.items() if m > HI]; below = [w for w, m in meds.items() if m < LO]
    per_visit = float(np.mean([c for x in nv for c in x["cpu_per_visit"]]))
    fc = 1024 * per_visit / 3600
    return dict(visit_updates=vu, medians=meds, worlds_inside=len(inside), worlds_above=len(above), worlds_below=len(below),
                cpu_per_visit_naive=per_visit, forecast_1024_hours=fc, passes=len(inside) >= NEED and fc <= BOUND_H)


def fmt(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def table():
    jp, mp = OUT / "E44_F1.json", OUT / "E44_F1_TABLES.md"
    assert not jp.exists() and not mp.exists()
    recs = [json.loads(p.read_text()) for p in sorted(OUT.glob("lw_*_turn.json"))]
    lives = [score_life(r) for r in recs]
    gates = {vu: gate(lives, vu) for vu in LENGTHS}
    passing = [vu for vu in LENGTHS if gates[vu] and gates[vu]["passes"]]
    adopted = min(passing) if passing else None
    further = None
    if adopted is None:
        g64, g256 = gates[64], gates[256]
        if g64["worlds_above"] >= NEED:
            further = "ceiling at 64: the same at 32 updates a visit, then at a taught share of 0.05 with 64 updates (the first that passes)"
        elif g256["worlds_below"] >= NEED:
            further = "floor at 256: the same at 512 updates a visit"
        else:
            further = "no length passes and neither case holds: stop and report the three medians of each world"
    steps = {}
    for key, vu, ts in (("step32", 32, 0.10), ("stepts05", 64, 0.05), ("step512", 512, 0.10)):
        g = gate(lives, vu, ts=ts)
        if g:
            steps[key] = g
    laps = []
    for p in sorted(LAP.glob("*_br8_*.json")) if LAP.exists() else []:
        r = json.loads(p.read_text())
        s = r["settings"]; vu = s["visit_updates"]
        w = LW.World(s["world"], vu, s["p_new"], s["taught_share"], s["lowest"])
        trunk = next(x for x in lives if x["life"] == Path(r["branch"]["from_state"]["path"]).name.replace("_state_v8.pt", ""))
        ev = S.load(LAP / r["events"]["file"]); la = S.load(LAP / r["list"]["file"])
        visits = [(x["visit"], x["group"]) for x in r["by_visit"]]
        rows = S.lw_second_lap(ev, la, w, vu, trunk["first"], visits)
        for row in rows:
            k = trunk["kept"][row["family"]]
            row["list_share_end_of_own_visit"] = k["own_share"]
            row["list_share_end_of_lap1"] = next((x["share"] for x in k["later"] if x["at_visit"] == VISITS - 1), k["own_share"] if k["own_visit"] == VISITS - 1 else None)
        laps.append(dict(branch=r["events"]["file"][:-4], arm=s["arm"], world=s["world"], visit_updates=vu, rows=rows,
                         birth_family_list=[x for x in r["list_summary"] if x["family"] == -1], cpu_per_visit=r["cpu"]["per_visit"],
                         stream_sha256=r["stream_sha256"]))
    out = dict(experiment="E44", job="F1", brief="revision 2 (cc00760d)", lives=lives, gates={str(k): v for k, v in gates.items()}, adopted=adopted,
               further_step=further, further_gates=steps, second_lap=laps)
    jp.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(dict(gates=gates, adopted=adopted, further=further, steps=steps), indent=1, default=str))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        table()
