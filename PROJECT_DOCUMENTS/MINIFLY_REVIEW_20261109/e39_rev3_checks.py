"""Checks behind the audit of revision 3 of the E39 brief (jobs L1a and L1b) at a3275395 (16 November 2026).

Read-only: the reviewer's second implementation (e39_world.py, beside this file), arithmetic, and the worker's score files
read with `git show` from origin/claude/gallant-edison-p0qa68.  Nothing trains and no harness code is run.

  A  L1a's selection rule applied to world 1 (and to the other worlds): the six tasks, and what they are beside the 192
     linear tasks met in L1
  B  the birth groups (groups 0 and 1) of worlds 1 to 8
  C  what L1's own lives say about single linear tasks: which reached the criterion, by modulus band; best level by band
  D  how often the 32-item criterion is met at a true level below 0.8, in a solo run and at a birth's checks
  E  the two numbers 8,192 and 32,768, and what a median over six allows
  F  the two new streams from the brief's text: digests, and one reading that gives other items
  G  the schedule of a 64-visit life with a birth: later tasks per world, visits to the birth groups
  H  cost from L1's measured CPU-seconds

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_rev3_checks.py [A B C D E F G H]
"""
import json, os, subprocess, sys
from math import comb
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e39_world as W

WORKER = "origin/claude/gallant-edison-p0qa68"
E39 = "MOD97_LOCAL_RULE_20261106/exp45/e39/"
TAGS = {"R0": "", "vu512": "_vu512", "h96vu512": "_h96_u96_vu512"}
BANDS = [(8, 13), (18, 24), (30, 37)]
want = set(sys.argv[1:]) or set("ABCDEFGH")
worlds = {s: W.SpecWorld(s) for s in range(1, 9)}


def name(op):
    return f"{op[1]:+d}a{op[2]:+d}b" if op[0] == "lin" else op[0]


def band5(p):
    return "8-13" if p <= 13 else "14-17" if p <= 17 else "18-24" if p <= 24 else "25-29" if p <= 29 else "30-37"


def l1a_tasks(w):
    out = []
    for lo, hi in BANDS:
        for same in (False, True):
            ids = [i for i, (op, p) in enumerate(w.tasks) if op[0] == "lin" and lo <= p <= hi and ((op[1] == op[2]) == same)]
            out.append((lo, hi, same, ids[0], len(ids)))
    return out


def tail(n, k, q):
    return sum(comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1))


if "A" in want:
    print("A. L1a's rule on world 1: in each band, the lowest id with c1 != c2 and the lowest id with c1 = c2 (all 540 tasks)")
    w = worlds[1]
    for lo, hi, same, i, n in l1a_tasks(w):
        op, p = w.tasks[i]
        pairs = [(a, b) for a in range(p) for b in range(p)]
        wrap = np.mean([W.needs_wrap(op, p, a, b) for a, b in pairs])
        print(f"   band {lo:2d}-{hi:2d}, c1 {'= ' if same else '!='} c2: task {i:3d}  {name(op):7s} mod {p:2d}  group {i // 2:2d}"
              f"{' (a birth task of world 1)' if i < 4 else '':27s} taught pairs {len(w.taught[(op, p)]):3d}, asked {len(w.asked[(op, p)]):3d}; "
              f"pairs needing the reduction {wrap:.2f}; candidates {n}")
    ids = [i for _, _, _, i, _ in l1a_tasks(w)]
    print(f"   six distinct tasks: {len(set(ids)) == 6}; ids {sorted(ids)}; none is a probe task (id >= 500): {all(i < 500 for i in ids)}")
    print("   of the 192 linear tasks met in L1, 48 have c1 = c2 (a quarter); of L1a's six, three (a half)")
    print("   the same rule in the other worlds (ids):", {s: sorted(i for _, _, _, i, _ in l1a_tasks(worlds[s])) for s in range(2, 9)})

if "B" in want:
    print("B. the birth groups: tasks 0 to 3 of each world (operation, modulus, taught pairs)")
    for s in range(1, 9):
        w = worlds[s]
        row = [f"{name(op)} mod {p} ({len(w.taught[(op, p)])})" for op, p in w.tasks[:4]]
        lin = sum(op[0] == "lin" for op, p in w.tasks[:4]); small = sum(p <= 13 for op, p in w.tasks[:4])
        sym = sum(op[0] == "lin" and op[1] == op[2] for op, p in w.tasks[:4])
        print(f"   world {s}: {'; '.join(row)} | linear {lin}, of them c1 = c2 {sym}; modulus 13 or less {small}")
    print("   worlds with at least one birth task of modulus 13 or less:", [s for s in range(1, 9) if any(p <= 13 for op, p in worlds[s].tasks[:4])])

if "C" in want:
    print("C. L1's lives (the worker's score files): linear tasks that reached the 0.8 criterion within the 64-visit life")
    rows = {}
    for rung, tag in TAGS.items():
        for arm in ("naive", "sweep"):
            for s in range(1, 9):
                d = json.loads(subprocess.check_output(["git", "show", f"{WORKER}:{E39}sw_{arm}_w{s}_v64{tag}.score.json"]))
                best = {}
                for e in d["end_of_visit"]:
                    best[e["task"]] = max(best.get(e["task"], 0.0), e["level"])
                for c in d["cost"]:
                    op, p = worlds[s].tasks[c["task"]]
                    if op[0] == "lin":
                        rows.setdefault((rung, arm), []).append((op[1] == op[2], band5(p), c["cost"], best[c["task"]]))
    for rung in TAGS:
        r = rows[(rung, "naive")]
        line = []
        for b in ("8-13", "14-17", "18-24", "25-29", "30-37"):
            sel = [x for x in r if x[1] == b]
            line.append(f"{b}: {sum(x[2] is not None for x in sel)} of {len(sel)}")
        print(f"   {rung:9s} naive, by modulus band: " + "; ".join(line) + f" | all {sum(x[2] is not None for x in r)} of {len(r)}")
    r = rows[("h96vu512", "naive")]
    for b in ("8-13", "14-17", "18-24", "25-29", "30-37"):
        x = np.array([v[3] for v in r if v[1] == b])
        print(f"   h96vu512 naive, band {b:5s}: best end-of-visit level over the life: mean {x.mean():.2f}, median {np.median(x):.2f}, share at 0.5 or more {np.mean(x >= 0.5):.2f}")
    for rung in ("vu512", "h96vu512"):
        rc = sorted(x[2] for x in rows[(rung, "naive")] if x[2] is not None)
        q = np.percentile(rc, [25, 50, 75])
        print(f"   {rung} naive: taught items of the task seen when it reached, {len(rc)} tasks: quartiles {[int(v) for v in q]}, largest {max(rc):,}"
              f" | as updates alone (48 a update) {[round(v / 48) for v in q]}, largest {round(max(rc) / 48):,}"
              f" | as updates of a birth on four tasks (12 a update) {[round(v / 12) for v in q]}, largest {round(max(rc) / 12):,}")
    print("   these are the tasks that reached; the others are censored by the length of the life")

if "D" in want:
    print("D. the 32-item criterion at a true level below 0.8 (0.8 of 32 needs 26 right)")
    print("   true level | one window | solo: some window in 1,000 updates (about 500 separate windows) | birth: some check in 1,024 updates (16 checks; the other three tasks sure)")
    for q in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        p = tail(32, 26, q)
        print(f"      {q:.2f}    |   {p:.4f}   |   {1 - (1 - p) ** 500:.3f}   |   {1 - (1 - p) ** 16:.3f}")
    print("   with every asked item of the task over the last 64 updates (solo 1,024 items; birth about 256 a task), share right at least 0.8:")
    for q in (0.65, 0.70, 0.75):
        print(f"      {q:.2f}    |   n = 256: {tail(256, 205, q):.5f}   |   n = 1,024: {tail(1024, 820, q):.1e}")

if "E" in want:
    print("E. the two numbers")
    print(f"   taught items of one task: alone, 8,192 updates x 48 = {8192 * 48:,}; in a birth on four tasks, 32,768 updates x 12 = {32768 * 12:,}")
    print(f"   a full L1a run: 16,384 x 48 = {16384 * 48:,} taught items")
    print("   the median of six is the mean of the third and fourth: with the fourth 'not reached' (above 16,384) it is above 8,192 whatever the third is,")
    print("   so the rule is computable in every case; a size qualifies with up to two tasks 'not reached' and the fourth as slow as 16,384 - third")
    print("   a size at which half the tasks need more than 8,192 updates alone qualifies; a birth needs all four of its tasks inside the same allowance")

if "F" in want:
    print("F. the two new streams, from the brief's text; first 256 updates; the world digest's bytes per item, taught then asked")
    w = worlds[1]
    for _, _, _, i, _ in l1a_tasks(w):
        print(f"   solo, world 1, task {i:3d}: {w.stream_digest(w.solo_batches(i, 256))}")
    for s in (1, 2, 3):
        print(f"   birth, world {s}: {worlds[s].stream_digest(worlds[s].birth_batches(256))}")

    def solo_with_task_draw(w, task_id, updates):
        """The reading a runner gets by calling visit_batches with a one-task group: randrange(1) is drawn for the task."""
        r = Random(f"world|{w.seed}|solo|{task_id}"); t = w.tasks[task_id]
        for u in range(updates):
            items = []
            for i in range(W.TAUGHT + W.ASKED):
                r.randrange(1)
                items.append(w._draw(r, t, i < W.TAUGHT))
            yield 0, u, items[:W.TAUGHT], items[W.TAUGHT:]
    a, b = w.stream_digest(w.solo_batches(2, 256)), w.stream_digest(solo_with_task_draw(w, 2, 256))
    print(f"   solo, task 2, if a draw IS made for the task (randrange(1)): {b[:16]}... equal to the text's reading: {a == b}")
    x, y = Random("x"), Random("x"); x.randrange(1)
    print("   randrange(1) consumes generator state:", x.random() != y.random())

if "G" in want:
    print("G. a 64-visit life after a birth on groups 0 and 1")
    for s in range(1, 9):
        sc = worlds[s].schedule(64)
        first = {}
        for v, g in enumerate(sc):
            first.setdefault(g, v)
        later = [g for g in first if g > 1]
        print(f"   world {s}: groups met {len(first)}; group 1 first scheduled at visit {first.get(1)}; visits to groups 0 and 1: {sum(g <= 1 for g in sc)} of 64; "
              f"groups first met in the life {len(later)} ({2 * len(later)} tasks, of them linear {sum(op[0] == 'lin' for g in later for op, p in worlds[s].groups[g])})")
    print("   visit 0 is group 0 in every world: with the birth counted as visit -1 its absence is 1 (short)")

if "H" in want:
    print("H. cost, from L1's CPU-seconds per visit (the worker's report): 64 units 6.66 naive, 7.35 sweep per 512 updates; 96 units 8.97, 9.88")
    per = {(64, "naive"): 6.66 / 512, (64, "sweep"): 7.35 / 512, (96, "naive"): 8.97 / 512, (96, "sweep"): 9.88 / 512}
    tot = sum(6 * 16384 * v for v in per.values())
    print(f"   L1a: 16,384 updates take {', '.join(f'{k[0]} {k[1]} {16384 * v / 60:.1f} min' for k, v in per.items())}; 24 runs {tot / 3600:.2f} CPU-hours")
    for size in (64, 96):
        life = 8 * 64 * 512 * (per[(size, "naive")] + per[(size, "sweep")])
        birth = 8 * 32768 * (per[(size, "naive")] + per[(size, "sweep")])
        print(f"   L1b at {size} units: sixteen lives {life / 3600:.2f} CPU-hours; sixteen births at the cap {birth / 3600:.2f}; together {(life + birth) / 3600:.2f}")
        print(f"      a 1,024-visit life under naive with a birth at the cap: {(1024 * 512 + 32768) * per[(size, 'naive')] / 3600:.2f} CPU-hours; one birth at the cap {32768 * per[(size, 'naive')] / 60:.1f} min")
