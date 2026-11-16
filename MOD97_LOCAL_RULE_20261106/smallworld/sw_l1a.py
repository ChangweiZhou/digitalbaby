"""E39 revision 5, job L1a: how many updates does the small network need for one linear task alone?

Tasks by rule: in each band of modulus (8-13, 18-24, 30-37), the task of lowest id whose operation is linear with c1 != c2, and the task of
lowest id whose operation is linear with c1 = c2; worlds 1 and 2. Sizes H = U = 64, 96 and 192 (revision 5); arms naive and sweep; 16,384
updates; default rate and decay. 72 runs (sw_run.py solo), at most three processes at a time, the largest size first.

The size rule (fixed in the brief): a size qualifies if, under naive, at least six of the eight tasks of the two upper bands reach the
criterion within 8,192 updates. Use the smallest qualifying size of 64, 96 and 192, with visits of 512 updates. If none qualifies, stop
and report.

    python3 sw_l1a.py tasks          # the rule's tasks for worlds 1 and 2
    python3 sw_l1a.py run            # the 72 runs -> exp45/e39/L1a/
    python3 sw_l1a.py table          # -> exp45/e39/L1a/E39_L1A.json and E39_L1A_TABLE.md (the size rule's verdict)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sw_world as W                  # noqa: E402

OUT = HERE.parent / "exp45" / "e39" / "L1a"
BANDS = (("8-13", 8, 13), ("18-24", 18, 24), ("30-37", 30, 37))
UPPER = ("18-24", "30-37")
WORLDS, SIZES, ARMS, UPDATES = (1, 2), (64, 96, 192), ("naive", "sweep"), 16384
WITHIN, NEED = 8192, 6


def kind(t):
    k, c = t["op"]
    return None if k != "lin" else ("c1=c2" if c[0] == c[1] else "c1!=c2")


def describe(t):
    k, c = t["op"]
    return k if k != "lin" else f"{c[0]:+d}a{c[1]:+d}b".replace("+1a", "a").replace("-1a", "-a").replace("+1b", "+b").replace("-1b", "-b")


def rule_tasks(world):
    w = W.World(world); out = []
    for band, lo, hi in BANDS:
        for kd in ("c1!=c2", "c1=c2"):
            t = min((t for t in w.tasks if lo <= t["p"] <= hi and kind(t) == kd), key=lambda t: t["id"])
            out.append(dict(world=world, band=band, kind=kd, task=t["id"], op=describe(t), p=t["p"], taught_pairs=len(t["taught"])))
    return out


def stem(arm, world, task, H):
    return f"sw_solo_{arm}_w{world}_t{task}_h{H}"


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(arm, d["world"], d["task"], H) for H in sorted(SIZES, reverse=True) for d in rule_tasks(1) + rule_tasks(2) for arm in ARMS]
    assert len(jobs) == 72
    running = []
    while jobs or running:
        while jobs and len(running) < 3:
            arm, w, t, H = jobs.pop(0)
            log = OUT / f"{stem(arm, w, t, H)}.log"
            assert not log.exists(), log
            c = [sys.executable, str(HERE / "sw_run.py"), "solo", arm, str(w), str(t), "--H", str(H), "--updates", str(UPDATES), "--out", str(OUT)]
            running.append((subprocess.Popen(c, stdout=open(log, "w"), stderr=subprocess.STDOUT), c, log))
        time.sleep(2)
        for r in list(running):
            if r[0].poll() is not None:
                running.remove(r)
                assert r[0].returncode == 0, f"failed: {' '.join(r[1])} (log {r[2]})"
                print("done", r[2].name, flush=True)


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def table():
    jpath, mpath = OUT / "E39_L1A.json", OUT / "E39_L1A_TABLE.md"
    assert not jpath.exists() and not mpath.exists()
    tasks = rule_tasks(1) + rule_tasks(2)
    rows = []
    for H in SIZES:
        for arm in ARMS:
            for d in tasks:
                r = json.loads((OUT / f"{stem(arm, d['world'], d['task'], H)}.json").read_text())
                q = r["quantities"]
                rows.append(dict(d, H=H, arm=arm, updates_to_criterion=q["updates_to_criterion"], updates_to_half=q["updates_to_half"],
                                 window32=q["window32"], last256=q["last256"], loss_last64=q["loss_last64"], cpu=r["cpu"]["total"],
                                 receipt=f"{stem(arm, d['world'], d['task'], H)}.json", events_sha256=r["events"]["arrays_sha256"],
                                 stream_first256_sha256=r["stream_first256_sha256"]))
    verdict = {}
    for H in SIZES:
        up = [x for x in rows if x["H"] == H and x["arm"] == "naive" and x["band"] in UPPER]
        ok = [x for x in up if x["updates_to_criterion"] is not None and x["updates_to_criterion"] <= WITHIN]
        verdict[H] = dict(upper_band_tasks=len(up), within_8192=len(ok), qualifies=len(ok) >= NEED,
                          tasks=[(x["world"], x["task"], x["updates_to_criterion"]) for x in up])
    q = [H for H in SIZES if verdict[H]["qualifies"]]
    adopted = min(q) if q else None
    out = dict(experiment="E39", job="L1a", brief="revision 5 (8183db86)", tasks=tasks, rows=rows, size_rule=verdict,
               adopted_size=adopted, decision=(f"H = U = {adopted} with visits of 512 updates for L1b" if adopted else
                                               "no size qualifies: stop and report (the brief: the task space or the learner's settings are next, "
                                               "a decision for the reviewer and the coordinator)"),
               cpu_total_hours=sum(x["cpu"] for x in rows) / 3600)
    jpath.write_text(json.dumps(out, indent=1))
    md = ["# E39 job L1a: one linear task alone (revision 5)", "",
          "16,384 updates from the network's initial values (`T.manual_seed(S)`, then the constructor), the solo stream "
          "`Random(\"world|S|solo|<task>\")`; default rate and decay. **Updates to criterion**: the first check (every 64 updates) at which the "
          "share right over the task's asked items of the last 64 updates (1,024 items) is at least 0.8; \"–\" = not reached in 16,384. "
          "Machine record: `E39_L1A.json`; receipts and event files beside it.", "",
          "## The tasks (the brief's rule)", "", "| World | Band | Kind | Task | Operation | p | Taught pairs |", "|---|---|---|---|---|---|---|"]
    md += [f"| {d['world']} | {d['band']} | {d['kind']} | {d['task']} | {d['op']} | {d['p']} | {d['taught_pairs']} |" for d in tasks]
    md += ["", "## Updates to criterion, by size and arm", ""]
    for H in SIZES:
        for arm in ARMS:
            md += [f"### H = U = {H}, `{arm}`", "",
                   "| World | Band | Task | Taught pairs | Updates to 0.8 | Updates to 0.5 | 32-item window ≥ 0.8: update (seen) | Last 256: all / wrap / no wrap | Loss, last 64 updates |",
                   "|---|---|---|---|---|---|---|---|---|"]
            for x in [x for x in rows if x["H"] == H and x["arm"] == arm]:
                w32 = "–" if not x["window32"] else f"{x['window32']['update']} ({x['window32']['seen']:,})"
                l = x["last256"]
                md.append(f"| {x['world']} | {x['band']} | {x['task']} ({x['op']} mod {x['p']}) | {x['taught_pairs']} | {x['updates_to_criterion'] or '–'} | "
                          f"{x['updates_to_half'] or '–'} | {w32} | {f(l['all']['share'])} / {f(l['wrap']['share'])} ({l['wrap']['n']}) / "
                          f"{f(l['nowrap']['share'])} ({l['nowrap']['n']}) | {x['loss_last64']:.4f} |")
            md.append("")
    md += ["## The size rule", "", "A size qualifies if, under `naive`, at least six of the eight upper-band tasks (bands 18–24 and 30–37, both worlds) "
           "reach the criterion within 8,192 updates.", "", "| Size | Upper-band tasks within 8,192 updates | Qualifies |", "|---|---|---|"]
    for H in SIZES:
        v = verdict[H]
        md.append(f"| {H} | {v['within_8192']} of {v['upper_band_tasks']} | {'yes' if v['qualifies'] else 'no'} |")
    md += ["", f"**{out['decision']}.**", "", f"CPU: {out['cpu_total_hours']:.2f} CPU-hours for the 72 runs.", ""]
    mpath.write_text("\n".join(md))
    print(json.dumps(dict(size_rule=verdict, adopted=adopted), indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tasks":
        for w in WORLDS:
            for d in rule_tasks(w):
                print(d)
    else:
        {"run": run, "table": table}[cmd]()
