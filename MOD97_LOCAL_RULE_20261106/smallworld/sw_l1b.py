"""E39 revision 5, job L1b: the life with a birth.

Worlds 1 to 8, 64 visits, three arms (naive, sweep: births of their own; sweep-after-naive: the sweep's life from naive's state at the end
of its birth), at the size L1a's rule gives and visits of 512 updates; birth on groups 0 and 1, complete when three of its four tasks are
at the criterion, cap 32,768 updates. The named further steps (one, then stop): visits of 256 updates ("vu256", if the median of part (2)
is above 0.90) or a birth on groups 0 to 3, complete at six of eight, cap 65,536 ("birth0123", if below 0.10 with births complete).

    python3 sw_l1b.py run <H> [main|vu256|birth0123]     # the 24 lives (and, for main, check 12's stopped copy) -> exp45/e39/
    python3 sw_l1b.py table <H> [main|vu256|birth0123]   # score every life with its birth groups; the gate -> exp45/e39/L1b/E39_L1B_<setting>.json
    python3 sw_l1b.py report <H> <setting> [...]         # -> exp45/e39/L1b/E39_L1B_TABLES.md and two CSVs
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

E = MDIR / "exp45" / "e39"
L1B = E / "L1b"
WORLDS = range(1, 9)
ARMS = ("naive", "sweep", "sweep-after-naive")
VISITS = 64
SETTINGS = {"main": dict(vu=512, groups=(0, 1), cap=32768, need=3), "vu256": dict(vu=256, groups=(0, 1), cap=32768, need=3),
            "birth0123": dict(vu=512, groups=(0, 1, 2, 3), cap=65536, need=6)}
LOW, HIGH, NEED_WORLDS, BOUND_HOURS = 0.10, 0.90, 6, 6.0


def stem(arm, w, H, st):
    s = SETTINGS[st]
    tag = (f"_h{H}_u{H}" if H != 64 else "") + (f"_vu{s['vu']}" if s["vu"] != 256 else "") + "_birth" + "".join(str(g) for g in s["groups"])
    return f"sw_{arm}_w{w}_v{VISITS}{tag}"


def life_args(arm, w, H, st):
    s = SETTINGS[st]
    a = [arm, str(w), str(VISITS), "--H", str(H), "--visit-updates", str(s["vu"]), "--birth-groups", ",".join(str(g) for g in s["groups"]),
         "--birth-cap", str(s["cap"]), "--birth-need", str(s["need"])]
    return a


def run(H, st="main"):
    (E / "logs").mkdir(parents=True, exist_ok=True); L1B.mkdir(parents=True, exist_ok=True)
    py = [sys.executable, str(HERE / "sw_run.py"), "life"]
    jobs = []
    for w in WORLDS:
        for arm in ("naive", "sweep"):
            jobs.append((stem(arm, w, H, st), py + life_args(arm, w, H, st) + ["--out", str(E)], None))
    for w in WORLDS:
        v0 = E / f"{stem('naive', w, H, st)}_state_v0.pt"
        jobs.append((stem("sweep-after-naive", w, H, st), py + ["--birth-from", str(v0), "--out", str(E)], v0))
    if st == "main":
        c12 = L1B / "c12"; c12.mkdir(exist_ok=True)
        s1 = stem("naive", 1, H, st)
        stop_log = E / "logs" / f"c12_stop_{s1}.log"
        jobs.append((f"c12_stop_{s1}", py + life_args("naive", 1, H, st) + ["--out", str(c12), "--stop-after", "4"], None))
        jobs.append((f"c12_resume_{s1}", py + ["--resume", str(c12 / f"{s1}_state_v4.pt"), "--out", str(c12)], stop_log))
    running, done = [], set()

    def ready(dep):
        if dep is None:
            return True
        if dep.suffix == ".log":                                     # the stopped run must have finished
            return dep.exists() and "stopped after" in dep.read_text()
        return dep.exists()
    while jobs or running:
        while len(running) < 3:
            pick = next((j for j in jobs if ready(j[2])), None)
            if pick is None:
                break
            jobs.remove(pick)
            name, c, _ = pick
            log = E / "logs" / (f"{name}.log" if name.startswith("c12") else f"{name}.log")
            assert not log.exists(), log
            running.append((subprocess.Popen(c, stdout=open(log, "w"), stderr=subprocess.STDOUT), c, log))
        time.sleep(5)
        for r in list(running):
            if r[0].poll() is not None:
                running.remove(r)
                assert r[0].returncode == 0, f"failed: {' '.join(r[1])} (log {r[2]})"
                print("done", r[2].name, flush=True)


def kind(t):
    k, c = t["op"]
    return k if k != "lin" else ("lin c1=c2" if c[0] == c[1] else "lin c1!=c2")


def table(H, st="main"):
    path = L1B / f"E39_L1B_{st}.json"
    assert not path.exists(), path
    s = SETTINGS[st]
    lives, gate_rows = {}, []
    for w in WORLDS:
        world = W.World(w, s["vu"])
        sym = S.symmetric_tasks(w)
        bt = sorted(t["id"] for g in s["groups"] for t in world.groups[g])
        gen = W.digest(world, VISITS)
        sched = world.schedule(VISITS)
        first_later = next(x["visit"] for x in sched if x["group"] >= len(s["groups"]))
        for arm in ARMS:
            sm = stem(arm, w, H, st)
            rec = json.loads((E / f"{sm}.json").read_text())
            f = np.load(E / f"{sm}.npz"); ev = {k: f[k] for k in f.files}
            sc = S.score(ev, VISITS, sym, bt)
            tp = E / f"{sm}.score.json"
            if not tp.exists():
                tp.write_text(json.dumps(sc, indent=1))
            fv = sc["first_visit"]
            lev = [d["level"] for d in fv]
            med = float(np.median(lev)) if lev else None
            by_kind = {}
            for d in fv:
                by_kind.setdefault(kind(world.tasks[d["task"]]), []).append(d["level"])
            lin = [d["level"] for d in fv if world.tasks[d["task"]]["op"][0] == "lin"]
            eov = {(e["task"], e["visit"]): e["level"] for e in sc["end_of_visit"]}
            before_later = {}
            for t in bt:
                vs = [v for (tt, v) in eov if tt == t and v < first_later]
                before_later[t] = dict(visit=max(vs), level=eov[(t, max(vs))])
            b = rec["birth"]
            forecast = (b["cpu"] + 1024 * rec["cpu"]["mean_per_visit"]) / 3600
            lives[sm] = dict(arm=arm, world=w, receipt=f"{sm}.json", events=f"{sm}.npz", score_table=tp.name, birth_tasks=bt,
                             birth=dict(updates=b["updates"], complete=b["complete"], end_shares=b["end_shares"], checks=len(b["checks"]), cpu=b["cpu"],
                                        inherited=rec.get("birth_from") is not None),
                             stream_sha256=rec["stream_sha256"], generator_digest=gen, digest_equal=rec["stream_sha256"] == gen,
                             final=rec["final"], cpu_per_visit=rec["cpu"]["mean_per_visit"], forecast_1024_hours=forecast,
                             peak_rss_kib=rec["cpu"].get("peak_rss_kib"), first_later_group_visit=first_later, birth_tasks_before_first_later=before_later,
                             tasks_after_birth=len(fv), median_first_visit=med, part2=med is not None and LOW <= med <= HIGH,
                             mean_first_visit=float(np.mean(lev)) if lev else None, mean_first_visit_linear=float(np.mean(lin)) if lin else None,
                             first_visit_by_kind={k: dict(n=len(v), mean=float(np.mean(v))) for k, v in sorted(by_kind.items())},
                             reached={k: sum(1 for d in sc["cost"] if kind(world.tasks[d["task"]]) == k and d["cost"] is not None) for k in by_kind},
                             mean_score=sc["mean_score"], late_level=sc["late_level"], score_trend=sc["score_trend"], learning_trend=sc["learning_trend"],
                             cost_trend=sc["cost_trend"], kept_pooled=sc["kept_pooled"], symmetric=sc["symmetric"],
                             first_visit=[dict(d, op=kind(world.tasks[d["task"]]), p=world.tasks[d["task"]]["p"]) for d in fv],
                             birth_tasks_end_of_visit=[dict(task=t, visit=v, level=l) for (t, v), l in sorted(eov.items()) if t in bt],
                             score_by_age=sc["score_by_age"], blind_by_age=sc["blind_by_age"], birth_scores=sc["birth"])
            if arm == "naive":
                gate_rows.append(dict(world=w, part1=b["complete"], median=med, part2=lives[sm]["part2"], both=b["complete"] and lives[sm]["part2"],
                                      forecast_1024_hours=forecast))
    fc = max(r["forecast_1024_hours"] for r in gate_rows)
    n_both = sum(r["both"] for r in gate_rows)
    gate = dict(rows=gate_rows, worlds_passing=n_both, part1_worlds=sum(r["part1"] for r in gate_rows), part2_worlds=sum(r["part2"] for r in gate_rows),
                below_low=sum(r["median"] is not None and r["median"] < LOW for r in gate_rows),
                above_high=sum(r["median"] is not None and r["median"] > HIGH for r in gate_rows),
                forecast_1024_hours_naive_max=fc, cost_ok=fc <= BOUND_HOURS, passed=n_both >= NEED_WORLDS and fc <= BOUND_HOURS)
    if gate["passed"]:
        nxt = "the gate passes: this setting with its birth is the protocol's size; L2 follows by roster note"
    elif gate["part1_worlds"] < NEED_WORLDS:
        nxt = "births are not complete in six worlds: stop and report"
    elif gate["part2_worlds"] < NEED_WORLDS and gate["above_high"] > gate["below_low"]:
        nxt = "the median of (2) is above 0.90: the further step is the same lives with visits of 256 updates (vu256)"
    elif gate["part2_worlds"] < NEED_WORLDS and gate["below_low"] >= gate["above_high"]:
        nxt = "the median of (2) is below 0.10 with births complete: the further step is a birth on groups 0 to 3 (six of eight, cap 65,536)"
    elif n_both < NEED_WORLDS:
        nxt = "fewer than six worlds pass both parts although each part holds in six: no further step is named; stop and report"
    else:
        nxt = "only the cost bound fails: no further step is named (revision 5: report the figure; decided with the reviewer)"
    gate["next"] = nxt
    out = dict(experiment="E39", job="L1b", setting=st, H=H, settings=s, visits=VISITS, worlds=list(WORLDS), arms=list(ARMS), gate=gate, lives=lives)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(gate, indent=1))
    return out


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def report(H, settings):
    """E39_L1B_TABLES.md and E39_L1B_score_by_age.csv / E39_L1B_first_visit.csv from the per-setting files."""
    import csv
    md, sba, fvr = ["# E39 job L1b: tables", "", "Generated by `smallworld/sw_l1b.py report` from `L1b/E39_L1B_<setting>.json` and the score tables "
                    "(scored with the birth groups). Every task's first-visit level and cost: `E39_L1B_first_visit.csv`; score by age and the blind "
                    "predictor's: `E39_L1B_score_by_age.csv`.", ""], [], []
    for st in settings:
        T = json.loads((L1B / f"E39_L1B_{st}.json").read_text()); L = T["lives"]; g = T["gate"]
        key = {(v["arm"], v["world"]): v for v in L.values()}
        md += [f"## Setting {st}: H = U = {H}, {T['settings']}", "", "### The gate (naive)", "",
               "| World | Birth: updates, complete | (1) | Tasks after the birth | (2) median first-visit level | (2) | Both | Forecast, 1,024 visits with birth (CPU-h) |",
               "|---|---|---|---|---|---|---|---|"]
        for r in g["rows"]:
            v = key[("naive", r["world"])]
            md.append(f"| {r['world']} | {v['birth']['updates']:,}, {'yes' if v['birth']['complete'] else 'no'} | {'yes' if r['part1'] else 'no'} | "
                      f"{v['tasks_after_birth']} | {f(r['median'])} | {'yes' if r['part2'] else 'no'} | {'yes' if r['both'] else 'no'} | {r['forecast_1024_hours']:.2f} |")
        md += ["", f"Worlds passing both: **{g['worlds_passing']} of 8** (needed 6); births complete in {g['part1_worlds']}; part (2) in {g['part2_worlds']}; "
                   f"median below 0.10 in {g['below_low']}, above 0.90 in {g['above_high']}; largest forecast {g['forecast_1024_hours_naive_max']:.2f} CPU-hours (bound 6). "
                   f"**Gate {'passes' if g['passed'] else 'fails'}.** Next: {g['next']}.", ""]
        md += ["### Births, per world and arm", "", "| Arm | World | Updates | Complete | Share right at the last check, birth tasks in id order |", "|---|---|---|---|---|"]
        for arm in ("naive", "sweep", "sweep-after-naive"):
            for w in WORLDS:
                v = key[(arm, w)]; b = v["birth"]
                md.append(f"| {arm} | {w} | {b['updates']:,} | {'yes' if b['complete'] else 'no'}{' (naive birth)' if b['inherited'] else ''} | "
                          + ", ".join(f(b["end_shares"][str(t)] if str(t) in b["end_shares"] else b["end_shares"].get(t), 2) for t in v["birth_tasks"]) + " |")
        md += ["", "### Beside the gate: the first later group, and the birth tasks' levels at their last visit before it (−1 = the birth)", "",
               "| World | First later group met at visit | Birth tasks: last visit before it, level |", "|---|---|---|"]
        for w in WORLDS:
            v = key[("naive", w)]
            md.append(f"| {w} | {v['first_later_group_visit']} | " + "; ".join(f"task {t}: v{x['visit']}, {f(x['level'], 2)}" for t, x in v["birth_tasks_before_first_later"].items()) + " |")
        md += ["", "### Per life", "",
               "| Arm | World | Mean score | Blind | Late level | Score trend | Trend of learning (n ≥ 1) | Mean first-visit level | Linear | CPU-s per visit | Digest = generator's |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in ("naive", "sweep", "sweep-after-naive"):
            for w in WORLDS:
                v = key[(arm, w)]
                bl = np.mean([x for x in v["blind_by_age"] if x is not None])
                md.append(f"| {arm} | {w} | {pct(v['mean_score'])} | {pct(bl)} | {pct(v['late_level'])} | {f(v['score_trend'])} | {f(v['learning_trend'])} | "
                          f"{f(v['mean_first_visit'])} | {f(v['mean_first_visit_linear'])} | {v['cpu_per_visit']:.2f} | {'yes' if v['digest_equal'] else 'NO'} |")
        md += ["", "### First-visit level by kind of operation (tasks after the birth; mean, tasks; pooled over worlds) and tasks reaching 0.8 within the life", "",
               "| Arm | lin c1≠c2 | lin c1=c2 | max | min | Reached 0.8: lin c1≠c2 / c1=c2 / max / min |", "|---|---|---|---|---|---|"]
        for arm in ("naive", "sweep", "sweep-after-naive"):
            pool, reach = {}, {}
            for w in WORLDS:
                v = key[(arm, w)]
                for d in v["first_visit"]:
                    pool.setdefault(d["op"], []).append(d["level"])
                for k, n in v["reached"].items():
                    reach[k] = reach.get(k, 0) + n
            ks = ("lin c1!=c2", "lin c1=c2", "max", "min")
            md.append(f"| {arm} | " + " | ".join(f"{f(np.mean(pool[k]))} ({len(pool[k])})" if k in pool else "–" for k in ks) + " | "
                      + " / ".join(f"{reach.get(k, 0)} of {len(pool.get(k, []))}" for k in ks) + " |")
        md += ["", "### Birth tasks: the mean of each one's last 32 asked items at the end of the birth (v−1) and of each of its visits", "", "| Arm | World | Levels |", "|---|---|---|"]
        for arm in ("naive", "sweep", "sweep-after-naive"):
            for w in WORLDS:
                v = key[(arm, w)]; by = {}
                for e in v["birth_tasks_end_of_visit"]:
                    by.setdefault(e["task"], []).append(f"v{e['visit']}: {f(e['level'], 2)}")
                md.append(f"| {arm} | {w} | " + " · ".join(f"task {t}: " + ", ".join(x) for t, x in by.items()) + " |")
        md += ["", "### Kept at return, the visit's first update, pooled by absence class (returns: all / wrap / no wrap, percent)", "",
               "| Arm | World | Short | Medium | Long | Short, first 32 | Medium, first 32 | Long, first 32 |", "|---|---|---|---|---|---|---|---|"]
        for arm in ("naive", "sweep", "sweep-after-naive"):
            for w in WORLDS:
                kp = key[(arm, w)]["kept_pooled"]
                cells = [f"{kp[c]['returns']}: {pct(kp[c]['first_update']['share'])} / {pct(kp[c]['first_update_wrap']['share'])} / {pct(kp[c]['first_update_nowrap']['share'])}"
                         for c in ("short", "medium", "long")] + [pct(kp[c]["first32"]["share"]) for c in ("short", "medium", "long")]
                md.append(f"| {arm} | {w} | " + " | ".join(cells) + " |")
        md += ["", "### Symmetric operations by `mirror` (percent right, items)", "",
               "| Arm | World | Symmetric, mirror = 1 | mirror = 0 | Non-symmetric | Blind, mirror = 1 |", "|---|---|---|---|---|---|"]
        for arm in ("naive", "sweep", "sweep-after-naive"):
            for w in WORLDS:
                sy = key[(arm, w)]["symmetric"]
                md.append(f"| {arm} | {w} | {pct(sy['mirror']['share'])} ({sy['mirror']['n']:,}) | {pct(sy['no_mirror']['share'])} ({sy['no_mirror']['n']:,}) | "
                          f"{pct(sy['non_symmetric']['share'])} ({sy['non_symmetric']['n']:,}) | {pct(sy['blind_mirror']['share'])} |")
        md += ["", "### Loss when the stream leaves a task, restoration at return (pooled over eight worlds; birth tasks' first return is from the birth)", "",
               "| Arm | Class | Returns | Level before | Kept, first update | Relearning reached in the visit | Median taught items to 0.8 within the visit |",
               "|---|---|---|---|---|---|---|"]
        for arm in ("naive", "sweep", "sweep-after-naive"):
            for c in ("short", "medium", "long"):
                rows = []
                for w in WORLDS:
                    sc = json.loads((E / key[(arm, w)]["score_table"]).read_text())
                    rows += [(k, x) for k, x in zip(sc["kept"], sc["relearn"]) if k["cls"] == c]
                if not rows:
                    md.append(f"| {arm} | {c} | 0 | – | – | – | – |"); continue
                bef = [k["before"] for k, _ in rows if k["before"] is not None]
                kr = sum(k["first_update"]["right"] for k, _ in rows); kn = sum(k["first_update"]["n"] for k, _ in rows)
                rl = [x["items"] for _, x in rows if x["items"] is not None]
                md.append(f"| {arm} | {c} | {len(rows)} | {pct(np.mean(bef)) if bef else '–'} | {pct(kr / kn) if kn else '–'} | {len(rl)} of {len(rows)} | "
                          f"{f(float(np.median(rl)), 0) if rl else '–'} |")
        md += ["", "### First-visit level by order of meeting (tasks after the birth, n from 1; mean, tasks; pooled over worlds)", "",
               "| Arm | n = 1–4 | n = 5–10 | n = 11–20 | n ≥ 21 |", "|---|---|---|---|---|"]
        for arm in ("naive", "sweep", "sweep-after-naive"):
            bins = {k: [] for k in ("a", "b", "c", "d")}
            for w in WORLDS:
                for d in key[(arm, w)]["first_visit"]:
                    bins["a" if d["n"] <= 4 else "b" if d["n"] <= 10 else "c" if d["n"] <= 20 else "d"].append(d["level"])
            md.append(f"| {arm} | " + " | ".join(f"{f(np.mean(x))} ({len(x)})" if x else "–" for x in bins.values()) + " |")
        md.append("")
        for arm in ("naive", "sweep", "sweep-after-naive"):
            for w in WORLDS:
                v = key[(arm, w)]
                for d in v["first_visit"]:
                    fvr.append(dict(setting=st, H=H, arm=arm, world=w, n=d["n"], task=d["task"], op=d["op"], p=d["p"], first_visit=d["visit"],
                                    level=d["level"], first_visit_cost=d["cost"]))
                for vv, (a_, b_) in enumerate(zip(v["score_by_age"], v["blind_by_age"])):
                    sba.append(dict(setting=st, H=H, arm=arm, world=w, visit=vv, score=a_, blind=b_))
    for name, rows in (("E39_L1B_first_visit.csv", fvr), ("E39_L1B_score_by_age.csv", sba)):
        with open(L1B / name, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    (L1B / "E39_L1B_TABLES.md").write_text("\n".join(md) + "\n")
    print("tables written for", settings)


if __name__ == "__main__":
    cmd, H = sys.argv[1], int(sys.argv[2])
    if cmd == "report":
        report(H, sys.argv[3:] or ["main"])
    else:
        st = sys.argv[3] if len(sys.argv) > 3 else "main"
        {"run": run, "table": table}[cmd](H, st)
