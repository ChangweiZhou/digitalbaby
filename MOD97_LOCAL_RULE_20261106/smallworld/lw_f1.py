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
    if adopted is None and None in gates.values():
        further = "incomplete: not every gate life has a receipt"
    elif adopted is None:
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


# ---------------------------------------------------------------- tables (from E44_F1.json)
SIZE_BINS = (("264 or fewer", 0, 264), ("265 to 688", 265, 688), ("689 or more", 689, 10 ** 9))


def setname(x):
    if x["arm"] == "sweep-after-naive":
        return "san"
    if x["wd"] == 0:
        return "wd0"
    if x["taught_share"] != 0.10:
        return f"ts{x['taught_share']:g}"
    return "gate"


def groups_of(lives):
    out = {}
    for x in lives:
        out.setdefault((setname(x), x["arm"], x["visit_updates"]), []).append(x)
    return dict(sorted(out.items(), key=lambda kv: (["gate", "san", "wd0"].index(kv[0][0]) if kv[0][0] in ("gate", "san", "wd0") else 9,
                                                   kv[0][2], kv[0][1])))


def joint_fit(rows):
    """Least squares of the family's figure on ln n and ln(taught pairs) together, with an intercept; rows (n, taught, figure)."""
    X = np.array([[1.0, math.log(n), math.log(tp)] for n, tp, _ in rows]); y = np.array([f for _, _, f in rows])
    if len(rows) < 4:
        return None
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return dict(intercept=float(b[0]), per_ln_n=float(b[1]), per_ln_taught_pairs=float(b[2]), points=len(rows))


def P(x, d=3):
    return "–" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def md():
    J = json.loads((OUT / "E44_F1.json").read_text())
    lives = J["lives"]
    G = groups_of(lives)
    L = ["# E44 job F1: the calibration's tables", "",
         "Machine record `E44_F1.json`; receipts, event files and lists beside it. Levels are shares right over asked items; the **family's "
         "figure** is the mean of add and sub on asked reduction pairs; **first-visit level** reads the last quarter of the family's first "
         "visit (16, 32, 64 updates at visits of 64, 128, 256). Families are numbered as in the file (0 to 7); a calibration life takes them "
         "in turn, so family f is the (f + 1)-th taken.", ""]
    # the gate
    L += ["## The gate (under `naive`): the median over families 1 to 7 of the first-visit level on reduction pairs", "",
          "| Visit length | " + " | ".join(f"World {w}" for w in WORLDS) + " | Worlds in 0.20–0.90 | above | below | CPU-s a visit | Forecast, 1,024 visits | Passes |",
          "|---|" + "---|" * 8 + "---|---|---|---|---|---|"]
    for vu, g in list(J["gates"].items()) + [(k, v) for k, v in J["further_gates"].items()]:
        if not g:
            continue
        L.append(f"| {g['visit_updates']}{'' if vu in ('64', '128', '256') else ' (' + vu + ')'} | " + " | ".join(P(g["medians"][str(w)] if str(w) in g["medians"] else g["medians"].get(w)) for w in WORLDS)
                 + f" | {g['worlds_inside']} | {g['worlds_above']} | {g['worlds_below']} | {g['cpu_per_visit_naive']:.1f} | {g['forecast_1024_hours']:.2f} h | {'yes' if g['passes'] else 'no'} |")
    L += ["", f"Adopted visit length: **{J['adopted'] if J['adopted'] else 'none'}**." + (f" Further step: {J['further_step']}." if J["further_step"] else ""), ""]
    # medians of every set
    L += ["## The same median in every set", "", "| Set | Arm | Visit length | " + " | ".join(f"W{w}" for w in WORLDS) + " | Mean | Trend over n = 2..8: mean slope, worlds positive |",
          "|---|---|---|" + "---|" * 8 + "---|---|"]
    for (st, arm, vu), xs in G.items():
        xs = sorted(xs, key=lambda x: x["world"])
        tr = [x["trend"] for x in xs if x["trend"] is not None]
        L.append(f"| {st} | {arm} | {vu} | " + " | ".join(P(x["gate_median"]) for x in xs) + f" | {P(float(np.mean([x['gate_median'] for x in xs])))} | "
                 f"{P(float(np.mean(tr)))}, {sum(t > 0 for t in tr)} of {len(tr)} |")
    # by family: level on reduction pairs, without, max, min, three windows (pooled means over worlds)
    L += ["", "## First-visit levels by window, pooled over worlds (mean over the 8 worlds of families 1 to 7)", "",
          "| Set | Arm | Visit length | Window | Reduction (figure) | Without reduction | Max | Min |", "|---|---|---|---|---|---|---|---|"]
    for (st, arm, vu), xs in G.items():
        for wname in ("first16", "whole", "last_quarter"):
            rows = [r[wname] for x in xs for r in x["first"] if r["n"] >= 2]
            mean = lambda key, sub=None: float(np.mean([(r[key] if sub is None else r[key]["share"]) for r in rows if (r[key] if sub is None else r[key]["share"]) is not None]))
            L.append(f"| {st} | {arm} | {vu} | {wname.replace('_', ' ')} | {P(mean('figure'))} | {P(mean('none_figure'))} | {P(mean('max', 1))} | {P(mean('min', 1))} |")
    # against size and place
    L += ["", "## The family's figure (first-visit level, last quarter) against its taught pairs and its place in turn, pooled over worlds", "",
          "Families 1 to 7 of every world (56 a cell row); the joint fit is the least-squares fit of the figure on ln n and ln(taught pairs) "
          "together, over n = 2 to 8.", "",
          "| Set | Arm | Visit length | " + " | ".join(f"{b[0]} pairs" for b in SIZE_BINS) + " | Joint fit: per ln n | per ln(taught pairs) |",
          "|---|---|---|" + "---|" * 3 + "---|---|"]
    for (st, arm, vu), xs in G.items():
        rs = [r for x in xs for r in x["first"] if r["n"] >= 2]
        cells = []
        for name, lo, hi in SIZE_BINS:
            v = [r["last_quarter"]["figure"] for r in rs if lo <= r["taught_pairs"] <= hi]
            cells.append(f"{P(float(np.mean(v)))} ({len(v)})" if v else "–")
        jf = joint_fit([(r["n"], r["taught_pairs"], r["last_quarter"]["figure"]) for r in rs])
        L.append(f"| {st} | {arm} | {vu} | " + " | ".join(cells) + f" | {P(jf['per_ln_n'] if jf else None)} | {P(jf['per_ln_taught_pairs'] if jf else None)} |")
    L += ["", "| Set | Arm | Visit length | " + " | ".join(f"n = {n} (family {n - 1})" for n in range(1, 9)) + " |", "|---|---|---|" + "---|" * 8]
    for (st, arm, vu), xs in G.items():
        L.append(f"| {st} | {arm} | {vu} | " + " | ".join(P(float(np.mean([r["last_quarter"]["figure"] for x in xs for r in x["first"] if r["n"] == n]))) for n in range(1, 9)) + " |")
    # score by age
    L += ["", "## Score by age (share right over all asked items of the visit, mean over worlds) and the blind predictor's", "",
          "| Set | Arm | Visit length | " + " | ".join(f"Visit {v}" for v in range(8)) + " | Blind, mean over visits |", "|---|---|---|" + "---|" * 8 + "---|"]
    for (st, arm, vu), xs in G.items():
        L.append(f"| {st} | {arm} | {vu} | " + " | ".join(P(float(np.mean([x["by_visit"][v]["score"] for x in xs]))) for v in range(8))
                 + f" | {P(float(np.mean([b['blind'] for x in xs for b in x['by_visit']])))} |")
    # kept
    L += ["", "## What is kept (the list of item 5): share at a later visit divided by the share at the end of the family's own visit, with decay alone beside it", "",
          "Median over the families of the eight worlds (families 0 to 6 at one visit since, fewer as the distance grows); the families with a "
          "share of 0 at the end of their own visit are left out (count in brackets). Decay alone: the same list answered by the weights at "
          "the end of the family's own visit, every parameter but the byte table multiplied by (1 − lr × decay)^u.", "",
          "| Set | Arm | Visit length | Visits since: 1 | 2 | 3 | 4 | 7 | Decay alone: 1 | 2 | 3 | 4 | 7 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (st, arm, vu), xs in G.items():
        cells, dcells = [], []
        for d in (1, 2, 3, 4, 7):
            ks, ds, zero = [], [], 0
            for x in xs:
                for f, k in x["kept"].items():
                    if int(f) < 0:
                        continue
                    if not k["own_share"]:
                        zero += any(r["visits_since"] == d for r in k["later"]); continue
                    for r in k["later"]:
                        if r["visits_since"] == d:
                            ks.append(r["kept"]); ds.append(r["decay_alone_kept"])
            cells.append(f"{P(float(np.median(ks)))} ({len(ks)}{', ' + str(zero) + ' out' if zero else ''})" if ks else "–")
            dcells.append(P(float(np.median([v for v in ds if v is not None]))) if ks else "–")
        L.append(f"| {st} | {arm} | {vu} | " + " | ".join(cells) + " | " + " | ".join(dcells) + " |")
    L += ["", "**The birth family** (mod 89, E41's R items, 2,000): share right before visit 0 and at the end of visits 0, 1, 3 and 7 (mean over worlds), "
          "with decay alone of the born weights beside it.", "",
          "| Set | Arm | Visit length | Before | After visit 0 | 1 | 3 | 7 | Decay alone after visit 0 | 1 | 3 | 7 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (st, arm, vu), xs in G.items():
        b = [x["kept"]["-1"] if "-1" in x["kept"] else x["kept"][-1] for x in xs]
        at = lambda k, key: P(float(np.mean([next(r[key] for r in bb["later"] if r["at_visit"] == k) for bb in b])))
        L.append(f"| {st} | {arm} | {vu} | {P(float(np.mean([bb['own_share'] for bb in b])))} | " + " | ".join(at(k, "share") for k in (0, 1, 3, 7)) + " | "
                 + " | ".join(at(k, "decay_alone_share") for k in (0, 1, 3, 7)) + " |")
    # second lap
    if J["second_lap"]:
        L += ["", "## The second lap (at the adopted length): each family's return", "",
              "Pooled share right on the family's asked reduction items (add and sub) in the return's first update and its first 16 updates; the "
              "family's figure over the last quarter of the return beside that of its first visit; its list share at the end of its own visit, at "
              "the end of the first lap (visit 7) and at the end of the return. Means over the eight worlds.", "",
              "| Arm | Family | First update | First 16 updates | Last quarter (return) | Last quarter (first visit) | List: own visit | end of lap 1 | end of return |",
              "|---|---|---|---|---|---|---|---|---|"]
        for arm in ("naive", "sweep"):
            ls = [l for l in J["second_lap"] if l["arm"] == arm]
            for f in range(8):
                rows = [next(r for r in l["rows"] if r["family"] == f) for l in ls]
                m = lambda fn: P(float(np.mean([v for v in (fn(r) for r in rows) if v is not None]))) if rows else "–"
                L.append(f"| {arm} | {f} | {m(lambda r: r['first_update']['share'])} | {m(lambda r: r['first16']['share'])} | {m(lambda r: r['last_quarter']['figure'])} | "
                         f"{m(lambda r: r['first_visit_last_quarter_figure'])} | {m(lambda r: r['list_share_end_of_own_visit'])} | {m(lambda r: r['list_share_end_of_lap1'])} | "
                         f"{m(lambda r: r['list_share_end_of_return'])} |")
    # cost
    L += ["", "## CPU-seconds per visit (updates with their asking; the list apart)", "", "| Set | Arm | Visit length | Per visit, mean | List per visit, mean | Forecast, 1,024 visits | Peak memory |",
          "|---|---|---|---|---|---|---|"]
    for (st, arm, vu), xs in G.items():
        pv = float(np.mean([c for x in xs for c in x["cpu_per_visit"]])); lv = float(np.mean([c for x in xs for c in x["list_cpu_per_visit"]]))
        L.append(f"| {st} | {arm} | {vu} | {pv:.2f} | {lv:.2f} | {1024 * pv / 3600:.2f} h | {max(x['peak_rss_kib'] for x in xs) / 1024:.0f} MiB |")
    (OUT / "E44_F1_TABLES.md").write_text("\n".join(L) + "\n")
    # CSVs
    import csv
    with open(OUT / "E44_F1_first_visit.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["set", "arm", "visit_updates", "world", "n", "family", "modulus", "taught_pairs", "window", "add_reduction", "add_reduction_n", "sub_reduction",
                     "sub_reduction_n", "figure", "add_none", "sub_none", "none_figure", "max", "max_n", "min", "min_n"])
        for (st, arm, vu), xs in G.items():
            for x in sorted(xs, key=lambda x: x["world"]):
                for r in x["first"]:
                    for wname in ("last_quarter", "whole", "first16"):
                        d = r[wname]
                        wr.writerow([st, arm, vu, x["world"], r["n"], r["family"], r["p"], r["taught_pairs"], wname, d["add_reduction"]["share"], d["add_reduction"]["n"],
                                     d["sub_reduction"]["share"], d["sub_reduction"]["n"], d["figure"], d["add_none"]["share"], d["sub_none"]["share"], d["none_figure"],
                                     d["max"]["share"], d["max"]["n"], d["min"]["share"], d["min"]["n"]])
    with open(OUT / "E44_F1_kept.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["set", "arm", "visit_updates", "world", "family", "modulus", "own_visit", "own_share", "at_visit", "visits_since", "share", "kept",
                     "decay_updates", "decay_alone_share", "decay_alone_kept"])
        for (st, arm, vu), xs in G.items():
            for x in sorted(xs, key=lambda x: x["world"]):
                w = LW.World(x["world"])
                for f, k in sorted(x["kept"].items(), key=lambda kv: int(kv[0])):
                    p_ = 89 if int(f) < 0 else w.groups[int(f)][0]["p"]
                    for r in k["later"]:
                        wr.writerow([st, arm, vu, x["world"], f, p_, k["own_visit"], k["own_share"], r["at_visit"], r["visits_since"], r["share"], r["kept"],
                                     r["decay_updates"], r["decay_alone_share"], r["decay_alone_kept"]])
    with open(OUT / "E44_F1_score_by_age.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["set", "arm", "visit_updates", "world", "visit", "family", "score", "blind"])
        for (st, arm, vu), xs in G.items():
            for x in sorted(xs, key=lambda x: x["world"]):
                for b in x["by_visit"]:
                    wr.writerow([st, arm, vu, x["world"], b["visit"], b["family"], b["score"], b["blind"]])
    if J["second_lap"]:
        with open(LAP / "E44_F1_second_lap.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["arm", "visit_updates", "world", "visit", "family", "modulus", "first_update_share", "first_update_n", "first16_share", "first16_n",
                         "last_quarter_figure", "first_visit_last_quarter_figure", "list_end_of_own_visit", "list_end_of_lap1", "list_end_of_return"])
            for l in J["second_lap"]:
                for r in l["rows"]:
                    wr.writerow([l["arm"], l["visit_updates"], l["world"], r["visit"], r["family"], r["p"], r["first_update"]["share"], r["first_update"]["n"],
                                 r["first16"]["share"], r["first16"]["n"], r["last_quarter"]["figure"], r["first_visit_last_quarter_figure"],
                                 r["list_share_end_of_own_visit"], r["list_share_end_of_lap1"], r["list_share_end_of_return"]])
    print("tables written")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif sys.argv[1] == "md":
        md()
    else:
        table()
