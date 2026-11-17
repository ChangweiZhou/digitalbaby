"""Computations behind E44_GATE_FOR_THE_SWEEP_20261117.md: the gate of E44's calibration (job F1), which the brief asked of the arm
`naive` only, computed for the arm `sweep` from the calibration's committed event files; the level within a first visit by quarter
of the visit, for both arms; and what the calibration's files hold for an arm with a store.

Read-only.  Nothing is trained, no file of another branch is executed, and no file is written: files of other refs are read with
`git show` and listed with `git ls-tree`.  The functions that read an event file and define the gate's figure are those of the
audit's script, e44_l2_checks.py (beside this file), reused by import; that file is not edited.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e44_gate_sweep.py [G Q S B]

  G  the gate's figure for `naive`, `sweep` and the third arm of the calibration, `sweep-after-naive`, at 64, 128 and 256 updates
     a visit in worlds 1 to 8, by three routes: (1) the audit's (every answer marked again against the reviewer's own stream),
     (2) from the event file's own columns alone (the audit's l2_quantities), (3) the worker's committed table; the band, the
     worlds inside it, the cost condition, the shortest length that passes for each arm; the difference between the arms
  Q  at 256 updates a visit: the family's figure in each quarter and each eighth of a first visit, per world and pooled; the rise
     from each quarter to the next; the difference between the arms in each quarter
  S  every receipt of the calibration read for a store; the lives at 128 updates a visit, which taught the number of new items
     that an arm with a store teaches in a visit of 256; files of an arm with a store on other fetched refs
  B  the births: the calibration used one seed's births in every world and the long lives use three; accuracy R at birth of the
     six births; the calibration life of worlds 2 and 3 under `naive`, run again by the worker of the first long lives with the
     world's own birth, beside the calibration's

Every part reads the 72 event files of the three arms first (about 65 s); the whole takes about 75 s.
"""
import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True                              # importing the audit's script then writes no cache file: this script writes nothing
sys.path.insert(0, str(HERE))
import e44_l2_checks as C                                   # noqa: E402  (the audit's script: git_npz, stem, marked_events, figure, gate_table, l2_quantities)

D, M, F1 = C.D, C.M, C.F1                                   # the default branch; MOD97_LOCAL_RULE_20261106/; .../exp45/e44/F1/
E44 = M + "exp45/e44/"
L2_REF = "origin/claude/e44-l2-first-long-lives"            # the branch of the worker of the first long lives (read with git show only)
BAND = (0.20, 0.90)                                         # the brief: "between 0.20 and 0.90"
MIN_WORLDS, MAX_HOURS = 6, 6.0                              # "in at least six of eight worlds"; "forecast within 6 CPU-hours"
LENGTHS, WORLDS, FAMILIES = (64, 128, 256), range(1, 9), range(1, 8)        # families 1 to 7: "the second to the eighth family taken"
ARMS = ("naive", "sweep", "sweep-after-naive")
T90_7 = 1.894579                                            # Student's t, 95th percentile, 7 degrees of freedom (90% interval over 8 worlds)


def git_lines(*args):
    r = subprocess.run(["git", *args], capture_output=True, cwd=str(HERE))
    assert r.returncode == 0, (args, r.stderr[:200])
    return r.stdout.decode().splitlines()


def tree(ref, path):
    """The files under a path of a ref (paths from the repository's root)."""
    return git_lines("ls-tree", "-r", "--name-only", "--full-tree", ref, "--", path)


def inside(m):
    return BAND[0] <= m <= BAND[1]


def interval(xs):
    """Mean of eight per-world values and the half-width of its 90% interval (Student's t, 7 degrees of freedom)."""
    xs = np.asarray(xs, float)
    assert len(xs) == 8
    return float(xs.mean()), float(T90_7 * xs.std(ddof=1) / np.sqrt(8))


def fmt_iv(xs):
    m, h = interval(xs)
    return f"{m:+.3f} (90% interval {m - h:+.3f} to {m + h:+.3f}; {sum(x > 0 for x in xs)} of 8 worlds above zero)"


# ================================================================ the figures of every life
_lives = {}


def lives():
    """For every arm, visit length and world: `fig`, the family's figure of the eight first visits by the audit's route (for
    `naive` and `sweep` this is the audit's own table, gate_table; for the third arm the same two calls); `alone`, the same from
    the event file's columns alone (the audit's l2_quantities, definition D6); `me`, the marked answers, kept at 256 updates."""
    if _lives:
        return _lives
    G = C.gate_table({w: C.World(w) for w in WORLDS})                       # naive and sweep: exactly what the audit's part C reads
    for arm in ARMS:
        for vu in LENGTHS:
            for w in WORLDS:
                ev = C.git_npz(F1 + C.stem(arm, w, vu) + ".npz")
                if arm in ("naive", "sweep"):
                    fig, me = G[arm, vu, w]["fig"], G[arm, vu, w]["me"]
                else:                                                         # the lines of gate_table, for an arm it does not read
                    me = C.marked_events(C.World(w), vu, ev, list(range(8)))
                    q = vu // 4
                    fig = [C.figure(me, v, vu - q, vu)[0] for v in range(8)]
                    me = me if vu == 256 else None
                r = C.l2_quantities(ev, 8, visit_updates=vu)
                assert r["kind"] == ["first"] * 8 and [x["family"] for x in r["first_visits"]] == list(range(8))
                _lives[arm, vu, w] = dict(fig=fig, me=me, alone=[x["level"] for x in r["first_visits"]], answers=len(ev["task"]))
    return _lives


def forecast_hours(arm, vu):
    """The brief's second condition: CPU-seconds a visit (the receipts' cpu.mean_per_visit, the list timed apart), mean over the
    eight worlds, times 1,024 visits, in hours.  As the audit's part C."""
    cpu = C.tmean([C.git_json(F1 + C.stem(arm, w, vu) + ".json")["cpu"]["mean_per_visit"] for w in WORLDS])
    return cpu, cpu * 1024 / 3600


# ================================================================ G. the gate
def part_g():
    t0 = time.time()
    L = lives()
    print("G. The gate's figure, from the calibration's event files on", D, "at", git_lines("rev-parse", "--short", D)[0])
    print(f"   {sum(v['answers'] for v in L.values()):,} answers of {len(L)} lives read; for every one the task, the reduction mark and right or wrong"
          " equal the reviewer's own stream (asserted in marked_events)")
    # route 2: the event file alone; route 3: the worker's committed table
    d2 = max(abs(a - b) for v in L.values() for a, b in zip(v["fig"], v["alone"]))
    rows = list(csv.DictReader(io.StringIO(C.git_bytes(F1 + "E44_F1_first_visit.csv").decode())))
    d3, n3, dw, nw = 0.0, 0, 0.0, 0
    for r in rows:
        key = (r["arm"], int(r["visit_updates"]), int(r["world"]))
        if r["set"] not in ("gate", "san") or key not in L:
            continue
        f = int(r["family"])
        if r["window"] == "last_quarter":
            d3 = max(d3, abs(float(r["figure"]) - L[key]["fig"][f])); n3 += 1
        elif L[key]["me"] is not None:                                        # the other two windows, where the marked answers are kept
            lo, hi = (0, 256) if r["window"] == "whole" else (0, 16)
            dw = max(dw, abs(float(r["figure"]) - C.figure(L[key]["me"], f, lo, hi)[0])); nw += 1
    assert n3 == len(L) * 8 and nw == 3 * 8 * 8 * 2
    print(f"   route 2 (the event file's own columns, l2_quantities): largest difference from route 1 over {len(L) * 8} family figures: {d2:.1e}")
    print(f"   route 3 (the worker's table E44_F1_first_visit.csv): largest difference over {n3} last-quarter figures: {d3:.1e};"
          f" over {nw} figures of the whole visit and of its first 16 updates at 256: {dw:.1e}")
    assert d2 < 1e-12 and d3 < 1e-12 and dw < 1e-12
    n_items = [C.figure(L[arm, 256, w]["me"], f, 192, 256)[2] for arm in ("naive",) for w in WORLDS for f in range(8)]
    print(f"   asked reduction items of add and sub in the last quarter of a first visit of 256 updates: {min(n_items)} to {max(n_items)}, median {int(np.median(n_items))}")
    print(f"   the band: {BAND[0]:.2f} to {BAND[1]:.2f}, both ends counted inside; a length passes if at least {MIN_WORLDS} of 8 worlds are inside"
          f" and 1,024 visits are forecast within {MAX_HOURS:.0f} CPU-hours")
    print("   arm, updates a visit: the median over families 1 to 7 of the family's figure, worlds 1 to 8 | mean of the eight | inside, below, above |"
          " CPU-s a visit, forecast | both conditions met")
    med, passes = {}, {}
    for arm in ARMS:
        for vu in LENGTHS:
            m = [float(np.median(L[arm, vu, w]["fig"][1:])) for w in WORLDS]
            cpu, hours = forecast_hours(arm, vu)
            n_in, lo, hi = sum(inside(x) for x in m), sum(x < BAND[0] for x in m), sum(x > BAND[1] for x in m)
            med[arm, vu], passes[arm, vu] = m, n_in >= MIN_WORLDS and hours <= MAX_HOURS
            print(f"   {arm:17s} {vu:3d}: {' '.join(f'{x:.3f}' for x in m)} | {np.mean(m):.3f} | {n_in}, {lo}, {hi} | {cpu:.2f} s, {hours:.2f} h |"
                  f" {'yes' if passes[arm, vu] else 'no'}")
    table = {64: "0.059 0.112 0.068 0.065 0.068 0.053 0.084 0.133", 128: "0.169 0.434 0.169 0.322 0.213 0.096 0.195 0.301", 256: "0.368 0.566 0.301 0.457 0.491 0.368 0.436 0.636"}
    for vu, row in table.items():
        assert " ".join(f"{x:.3f}" for x in med["naive", vu]) == row
    print("   the `naive` rows equal the report's table and the audit's section 2.1 to the third decimal (asserted)")
    for arm in ARMS:
        ok = [vu for vu in LENGTHS if passes[arm, vu]]
        print(f"   {arm}: lengths at which both conditions are met: {ok or 'none'}; the shortest: {ok[0] if ok else 'none'}")
    print("   the worlds inside the band, by number:")
    for arm in ARMS:
        print(f"      {arm:17s} " + "; ".join(f"{vu}: {[w for w, x in zip(WORLDS, med[arm, vu]) if inside(x)] or 'none'}" for vu in LENGTHS))
    print("   the medians of world 6 and world 3 under `sweep-after-naive` at 256, to five decimals:",
          f"{med['sweep-after-naive', 256][5]:.5f}, {med['sweep-after-naive', 256][2]:.5f}")
    print("   differences of the medians, worlds 1 to 8, and their mean:")
    for a, b in (("sweep", "naive"), ("sweep-after-naive", "naive"), ("sweep", "sweep-after-naive")):
        for vu in LENGTHS:
            d = [x - y for x, y in zip(med[a, vu], med[b, vu])]
            print(f"      {a} minus {b}, {vu:3d}: {' '.join(f'{x:+.3f}' for x in d)} | mean {fmt_iv(d)}")
    print("   the same figure with the first family counted too (families 0 to 7), and the mean in place of the median, at 256:")
    for arm in ARMS:
        figs = [L[arm, 256, w]["fig"] for w in WORLDS]
        print(f"      {arm:17s} median of 8 families: {' '.join(f'{np.median(x):.3f}' for x in figs)} | mean of families 1 to 7: "
              f"{' '.join(f'{np.mean(x[1:]):.3f}' for x in figs)}")
    print("   the learners (the receipts' learner entry; one birth file an arm in all eight worlds and all three lengths):")
    for arm in ARMS:
        ln = {json.dumps([r["learner"]["file"], r["learner"]["rule"], r["learner"]["seed"], r["learner"]["final_row"]["R"], r["local"], r["replay"]])
              for r in (C.git_json(F1 + C.stem(arm, w, vu) + ".json") for w in WORLDS for vu in LENGTHS)}
        assert len(ln) == 1
        f, rule, seed, R, local, replay = json.loads(ln.pop())
        print(f"      {arm:17s} birth {f} (rule {rule}, seed {seed}, accuracy R at birth {R:.2f}%); the life learns by {'the sweep' if local else 'back-propagation'}; store: {replay}")
    print(f"   part G: {time.time() - t0:.0f} s")
    return med


# ================================================================ Q. the level within a first visit
def part_q():
    t0 = time.time()
    L = lives()
    print("Q. The level within a first visit of 256 updates: the family's figure (mean of add's and sub's share right on asked reduction items)")
    print("   over each quarter of the visit's updates (updates 0-63, 64-127, 128-191, 192-255); families 1 to 7, as the gate counts them")
    quart = {}                                                                # arm -> array [world, family 0..7, quarter]
    for arm in ARMS:
        quart[arm] = np.array([[[C.figure(L[arm, 256, w]["me"], f, 64 * k, 64 * k + 64)[0] for k in range(4)] for f in range(8)] for w in WORLDS])
        assert np.isfinite(quart[arm]).all()
        assert all(abs(quart[arm][w - 1, f, 3] - L[arm, 256, w]["fig"][f]) < 1e-12 for w in WORLDS for f in range(8))     # the last quarter is the gate's figure
        for w in WORLDS:                                                      # the quarters and the eighths divide the visit: right answers and items add up
            for f in range(8):
                me = L[arm, 256, w]["me"]
                _, sh, n = C.figure(me, f, 0, 256)
                for step in (64, 32):
                    parts = [C.figure(me, f, lo, lo + step)[1:] for lo in range(0, 256, step)]
                    assert sum(k for _, k in parts) == n and round(sum(x * k for x, k in parts)) == round(sh * n)
    for arm in ARMS:
        q = quart[arm][:, 1:, :]                                              # families 1 to 7
        print(f"   {arm}: world | mean over families 1 to 7, quarters 1 2 3 4 | rise 1 to 2, 2 to 3, 3 to 4 | median over families 1 to 7, quarters 1 2 3 4")
        for i, w in enumerate(WORLDS):
            mn, md = q[i].mean(0), np.median(q[i], 0)
            print(f"      {w}  | {' '.join(f'{x:.3f}' for x in mn)} | {' '.join(f'{x:+.3f}' for x in np.diff(mn))} | {' '.join(f'{x:.3f}' for x in md)}")
        pooled = q.reshape(-1, 4)
        cnt = np.array([[C.figure(L[arm, 256, w]["me"], f, 64 * k, 64 * k + 64)[1:] for k in range(4)] for w in WORLDS for f in FAMILIES])      # share, n
        share = (cnt[:, :, 0] * cnt[:, :, 1]).sum(0) / cnt[:, :, 1].sum(0)
        print(f"      pooled, mean of the 56 first visits | {' '.join(f'{x:.3f}' for x in pooled.mean(0))} | {' '.join(f'{x:+.3f}' for x in np.diff(pooled.mean(0)))} |"
              f" median of the 56: {' '.join(f'{x:.3f}' for x in np.median(pooled, 0))}")
        print(f"      pooled, share right over all reduction items of the 56 first visits ({int(cnt[:, 0, 1].sum()):,} to {int(cnt[:, 3, 1].sum()):,} items a quarter): "
              f"{' '.join(f'{x:.3f}' for x in share)}")
        for k, name in ((0, "quarter 1 to 2"), (1, "quarter 2 to 3"), (2, "quarter 3 to 4")):
            per_world = (q[:, :, k + 1] - q[:, :, k]).mean(1)
            up = int((pooled[:, k + 1] > pooled[:, k]).sum())
            print(f"      rise {name}: mean over worlds {fmt_iv(per_world)}; first visits in which the later quarter is higher: {up} of 56")
        print(f"      the third quarter's level as a share of the last quarter's (pooled means): {pooled[:, 2].mean() / pooled[:, 3].mean():.3f};"
              f" family 0 (not counted by the gate), mean of 8 worlds by quarter: {' '.join(f'{x:.3f}' for x in quart[arm][:, 0, :].mean(0))}")
    print("   by eighth of the visit (32 updates each), pooled mean of the 56 first visits:")
    eighth = {}
    for arm in ARMS:
        e = np.array([[[C.figure(L[arm, 256, w]["me"], f, 32 * k, 32 * k + 32)[0] for k in range(8)] for f in FAMILIES] for w in WORLDS])
        eighth[arm] = e
        print(f"      {arm:17s} {' '.join(f'{x:.3f}' for x in e.reshape(-1, 8).mean(0))} | rise from the seventh to the last eighth, mean over worlds {fmt_iv((e[:, :, 7] - e[:, :, 6]).mean(1))}")
    print("      sweep minus naive, by eighth:              "
          + " ".join(f"{x:+.3f}" for x in (eighth["sweep"] - eighth["naive"]).reshape(-1, 8).mean(0)))
    print("   the end of the visit, per first visit and then the mean over families 1 to 7 in each world: the rise from the seventh to the last eighth,")
    print("   and the least-squares slope of the level over the last four eighths (per eighth, that is per 32 updates):")
    end = {}
    for arm in ARMS:
        e = eighth[arm]
        x = np.arange(4) - 1.5
        end[arm] = ((e[:, :, 7] - e[:, :, 6]).mean(1), ((e[:, :, 4:] * x).sum(2) / (x * x).sum()).mean(1))
        print(f"      {arm:17s} slope over the last four eighths {fmt_iv(end[arm][1])}")
    for a, b in (("sweep", "naive"), ("sweep-after-naive", "naive")):
        print(f"      {a} minus {b}: rise from the seventh to the last eighth {fmt_iv(end[a][0] - end[b][0])}")
        print(f"      {a} minus {b}: slope over the last four eighths {fmt_iv(end[a][1] - end[b][1])}")
    steps = np.diff(eighth["sweep"] - eighth["naive"], axis=2).mean(1)      # [world, step]: sweep's rise minus naive's, from each eighth to the next
    print("      sweep's rise minus naive's from each eighth to the next (mean over worlds, half-width of the 90% interval, worlds above zero): "
          + "; ".join(f"{k + 1} to {k + 2}: {interval(steps[:, k])[0]:+.3f} +/- {interval(steps[:, k])[1]:.3f}, {int((steps[:, k] > 0).sum())}" for k in range(7)))
    print("   the difference between the arms in each quarter (mean over families 1 to 7 in each world, then over the eight worlds):")
    for a, b in (("sweep", "naive"), ("sweep-after-naive", "naive")):
        d = quart[a][:, 1:, :].mean(1) - quart[b][:, 1:, :].mean(1)          # [world, quarter]
        for k in range(4):
            print(f"      {a} minus {b}, quarter {k + 1}: {fmt_iv(d[:, k])}")
        rise = (quart[a][:, 1:, 3] - quart[a][:, 1:, 2]).mean(1) - (quart[b][:, 1:, 3] - quart[b][:, 1:, 2]).mean(1)
        print(f"      {a}'s rise from quarter 3 to 4 minus {b}'s: {fmt_iv(rise)}")
        half = (quart[a][:, 1:, 3] - quart[a][:, 1:, 1]).mean(1) - (quart[b][:, 1:, 3] - quart[b][:, 1:, 1]).mean(1)
        print(f"      {a}'s rise from quarter 2 to 4 minus {b}'s: {fmt_iv(half)}")
    # where in its visit `sweep` reaches the level `naive` ends with
    nv, sw = eighth["naive"].reshape(-1, 8).mean(0), eighth["sweep"].reshape(-1, 8).mean(0)
    print(f"   `naive`'s pooled level over its last eighth is {nv[7]:.3f}; `sweep`'s pooled level first reaches it in eighth {int(np.argmax(sw >= nv[7])) + 1} of its visit"
          f" (updates {32 * int(np.argmax(sw >= nv[7]))} to {32 * int(np.argmax(sw >= nv[7])) + 31})")
    print(f"   part Q: {time.time() - t0:.0f} s")
    return quart


# ================================================================ S. an arm with a store
def part_s(med=None):
    t0 = time.time()
    print("S. What the calibration's files hold for an arm with a store")
    files = tree(D, E44)
    rec = []
    for p in files:
        if p.endswith(".json"):
            r = C.git_json(p)
            if isinstance(r, dict) and "settings" in r and "ledger" in r:
                rec.append((p[len(E44):], r))
    by = {}
    for p, r in rec:
        top = p.split("/")[0] + ("/lap2" if "/lap2/" in p else "")
        k = (top, r["settings"]["arm"], bool(r["replay"]), bool(r["ledger"]["buffer"]["used"]), r["settings"]["rho"], r["ledger"]["replayed_per_update"], r["counters"]["replayed_items"])
        by[k] = by.get(k, 0) + 1
    print(f"   {len(files)} files under {E44} on {D}; {len(rec)} of them are receipts of lives or branches")
    print("   folder, arm | receipts | replay, store used, rho, items replayed an update, items replayed in the life")
    for k in sorted(by, key=str):
        print(f"      {k[0]:11s} {k[1]:17s} | {by[k]:2d} | {k[2]}, {k[3]}, {k[4]}, {k[5]}, {k[6]}")
    n_store = sum(v for k, v in by.items() if k[2] or k[3] or k[5] or k[6])
    assert sum(v for k, v in by.items() if k[0] == "F1") == 96 and sum(v for k, v in by.items() if k[0] == "F1/lap2") == 16      # the calibration and its second lap
    print(f"   receipts of a life with a store: {n_store}; file names under {E44} that contain 'replay': {sum('replay' in p for p in files)}")
    assert n_store == 0
    # the nearest thing the calibration holds
    if med is None:
        L = lives()
        med = {(arm, vu): [float(np.median(L[arm, vu, w]["fig"][1:])) for w in WORLDS] for arm in ARMS for vu in LENGTHS}
    print("   new taught items in a visit: without a store 48 an update; with a store at rho = 0.5, 24 an update (the audit's section 4.2):"
          f" 256 x 24 = {256 * 24:,} = 128 x 48 = {128 * 48:,}")
    print("   the calibration's lives at 128 updates a visit taught that number of new items in half the updates and with no stored items between them:")
    for arm in ("naive", "sweep"):
        m = med[arm, 128]
        print(f"      {arm:5s} 128: medians {' '.join(f'{x:.3f}' for x in m)} | inside the band {sum(inside(x) for x in m)} of 8")
    # other fetched refs
    print("   files of E44 on other fetched refs whose name has an arm with a store (read-only listing; these are not the calibration's):")
    refs = git_lines("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin")
    n_sr = 0
    for ref in refs:
        names = [p for p in tree(ref, E44) if "replay" in p.split("/")[-1] and p.endswith(".npz") and not p.endswith("_list.npz")]
        n_sr += sum("sweep+replay" in p for p in names)
        if names:
            print(f"      {ref} at {git_lines('rev-parse', '--short', ref)[0]}: {len(names)} event files: " + ", ".join(p[len(E44):] for p in names))
    print(f"      {len(refs)} refs read; event files of `sweep+replay` among them: {n_sr}")
    # the one first visit of a later family under an arm with a store, in a check life of the worker of the first long lives
    a, b = E44 + "L2/checks/c7_replay/lw_replay_w5_v4_rho0.5", E44 + "L2/checks/c7_naive/lw_naive_w5_v4"
    if C.exists(a + ".npz", L2_REF) and C.exists(b + ".npz", L2_REF):
        sched = C.World(5).schedule(4)
        print(f"   check lives of world 5, 4 visits by the schedule (families {sched}), on {L2_REF}: visit 1 is family 1's first visit")
        for name, p in (("replay", a), ("naive", b)):
            ev, r = C.git_npz(p + ".npz", L2_REF), C.git_json(p + ".json", L2_REF)
            me = C.marked_events(C.World(5), 256, ev, sched)
            q = [C.figure(me, 1, 64 * k, 64 * k + 64)[0] for k in range(4)]
            lv = C.l2_quantities(ev, 4)["first_visits"]
            assert lv[1]["visit"] == 1 and abs(lv[1]["level"] - q[3]) < 1e-12
            print(f"      {name:6s} (birth {r['learner']['file'].split('/')[-1]}, rho {r['settings']['rho']}, wd {r['settings']['wd']}; items replayed in the life {r['counters']['replayed_items']:,}):"
                  f" family 1's figure by quarter of its first visit {' '.join(f'{x:.3f}' for x in q)}; family 0's last quarter {C.figure(me, 0, 192, 256)[0]:.3f}")
        Lv = lives()
        print("      the calibration's figure of the same family (world 5, family 1, last quarter; birth of seed 73129): "
              + "; ".join(f"{arm} at {vu}: {Lv[arm, vu, 5]['fig'][1]:.3f}" for arm, vu in (("naive", 256), ("naive", 128), ("sweep", 256), ("sweep", 128))))
        print("      one family in one world: not the gate's figure, which is a median over seven families in each of eight worlds")
    print(f"   part S: {time.time() - t0:.0f} s")


# ================================================================ B. the births of the long lives
def part_b():
    t0 = time.time()
    L = lives()
    print("B. The births: the calibration used the births of seed 73129 in every world; the long lives use, in world w, the births of seed")
    print(f"   {C.SEEDS}[(w - 1) mod 3] (the brief; the helper's instructions: worlds 1 and 4 seed 73129, world 2 seed 93683, world 3 seed 116089)")
    print("   accuracy R at birth (the last row of E41's receipts on the default branch), half width, taught fraction 0.50:")
    for seed in C.SEEDS:
        row = []
        for rule in ("B", "S"):
            rec = C.git_json(M + f"exp45/e41/e41_{rule}_h256_f50_{seed}.json")
            assert (rec["H"], rec["U"]) == (256, 128) and rec["complete"]
            row.append(f"rule {rule} {rec['curve'][-1]['R']:.2f}%")
        print(f"      seed {seed}: " + ", ".join(row))
    print(f"   the calibration life of worlds 2 and 3 under `naive` at 256 updates a visit, run again by the worker of the first long lives with the")
    print(f"   birth of the world's own seed ({L2_REF} at {git_lines('rev-parse', '--short', L2_REF)[0]}, L2/births/); no such life exists under `sweep`:")
    for w in (2, 3):
        p = E44 + f"L2/births/lw_naive_w{w}_v8_turn"
        if not C.exists(p + ".npz", L2_REF):
            print(f"      world {w}: not on the ref"); continue
        ev, rec = C.git_npz(p + ".npz", L2_REF), C.git_json(p + ".json", L2_REF)
        st = rec["settings"]
        assert (st["arm"], st["world"], st["visits"], st["visit_updates"], st["in_turn"], st["wd"], st["lr"]) == ("naive", w, 8, 256, True, 0.25, 0.001) and rec["complete"]
        me = C.marked_events(C.World(w), 256, ev, list(range(8)))             # the same stream as the calibration's life (asserted item by item)
        fig, cal = [C.figure(me, v, 192, 256)[0] for v in range(8)], L["naive", 256, w]["fig"]
        d = [a - b for a, b in zip(fig, cal)]
        print(f"      world {w}, birth {st['learner']['file'].split('/')[-1]}: median over families 1 to 7 {np.median(fig[1:]):.3f} (the calibration's, seed 73129: {np.median(cal[1:]):.3f};"
              f" difference {np.median(fig[1:]) - np.median(cal[1:]):+.3f});"
              f" family by family, this birth minus the calibration's: {' '.join(f'{x:+.3f}' for x in d)}")
    # is there a calibration life of `sweep` anywhere but in the calibration's own folder?
    refs = git_lines("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin")
    births = [p[len(E44):] for p in tree(L2_REF, E44 + "L2/births/") if p.endswith(".npz") and not p.endswith("_list.npz")]
    other = sorted({p[len(E44):] for ref in refs for p in tree(ref, E44)
                    if p.split("/")[-1].startswith("lw_sweep") and p.endswith("_turn.npz") and not p.startswith(F1)})
    print(f"   event files under L2/births/ on {L2_REF}: {births}")
    print(f"   event files of an in-turn life of a sweep arm outside {F1} on any of the {len(refs)} fetched refs: {other or 'none'}")
    print(f"   part B: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if a in ("G", "Q", "S", "B")] or ["G", "Q", "S", "B"]
    T0 = time.time()
    med = part_g() if "G" in want else None
    if "Q" in want:
        part_q()
    if "S" in want:
        part_s(med)
    if "B" in want:
        part_b()
    print(f"done in {time.time() - T0:.0f} s")
