"""Checks behind AUDIT_E44_L2_ROSTER_NOTE_20261117.md: the roster note for the first long lives of E44
(OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md, revision 1, on the default branch at 30628a4c), the change to the runner
that the note describes, and E44's calibration (job F1, merged at 08ef17e2).

Read-only.  Nothing is trained and no file of another branch is executed: files are read with `git show`, the world is the
reviewer's own second implementation (e44_world_reader.py), the network is the reviewer's numpy reader (np_gru.py).

    python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e44_l2_checks.py [A B C D E F G H]        (from the repository root)

  A  the schedules of worlds 1 to 8: families met, returns by length of absence, the asked reduction items a return's first
     update holds; the life that the note's check (b) names
  B  the runner at 39e4e594 (the version the calibration ran): the line that refuses the roster's `replay` arm; a model of which
     list rows the runner writes, tested against the committed list files; the rows and their cost in a life of 1,024 visits
     under the present runner and under the two arguments the note describes; which of the rows the note describes can be read
     as "decay alone beside what is kept"; digests of the row keys that an implementation of the change must produce
  C  the calibration's gate, from the 48 event files, with every answer re-marked against the reviewer's own stream
  D  the second lap, from the 16 branch files
  E  the eleven predictions P44.1 to P44.11
  F  the numpy reader on saved weights: the list at the end of world 1's calibration life, item by item, and decay alone of the
     born weights
  G  the learners: E41's half-width births of the three seeds (committed files, hashes, accuracy R by the numpy reader), and
     the width rule from E41's receipts
  H  the roster's cost: the note's forecast, the list's share of it, the buffer against the taught pairs a life meets
"""
import hashlib
import io
import json
import math
import pickle
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from random import Random

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e44_world_reader import World, needs_wrap, answer as op_answer        # noqa: E402  (the reviewer's generator)
from np_gru import sigmoid, split as e26_split, DEV                       # noqa: E402

D = "origin/codex/minifly-consolidated-20261023"
M = "MOD97_LOCAL_RULE_20261106/"
F1 = M + "exp45/e44/F1/"
RUNNER_AT = "39e4e594"                    # the runner's commit in every calibration receipt
EOS, BOS, GEN_END = 256, 257, 65535
BEFORE, BIRTH_FAMILY = 65535, -1
SEEDS = (73129, 93683, 116089)
L2_K, L2_DIST = 8, (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
T0 = time.time()


def git_bytes(path, ref=D):
    r = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, cwd=str(HERE))
    assert r.returncode == 0, (ref, path, r.stderr[:200])
    return r.stdout


def git_json(path, ref=D):
    return json.loads(git_bytes(path, ref))


def git_npz(path, ref=D):
    f = np.load(io.BytesIO(git_bytes(path, ref)))
    return {k: f[k] for k in f.files}


def exists(path, ref=D):
    return subprocess.run(["git", "cat-file", "-e", f"{ref}:{path}"], capture_output=True, cwd=str(HERE)).returncode == 0


def stem(arm, w, vu=256, wd0=False, visits=8):
    return f"lw_{arm}_w{w}_v{visits}" + ("" if vu == 256 else f"_vu{vu}") + ("_wd0" if wd0 else "") + "_turn"


def tmean(xs):
    return float(np.mean(xs))


def t_interval(xs, level=0.90):
    """Mean and half-width of the two-sided t interval across worlds (the note's "90% interval (t over worlds)")."""
    tq = {3: 2.3534, 7: 1.8946, 2: 2.9200, 1: 6.3138, 4: 2.1318, 5: 2.0150, 6: 1.9432}[len(xs) - 1]
    return tmean(xs), tq * float(np.std(xs, ddof=1)) / math.sqrt(len(xs))


# ================================================================ A. the schedules
def first_update_reduction_items(world, v, family):
    """The asked add and sub items that need the reduction in the first update of visit v (they are answered before any update
    of the visit): the items on which the note reads "kept at return"."""
    taught, asked = next(world.visit(v, family))
    return sum(1 for i, a, b, _, _ in asked if world.wrap(i, a, b))


def part_a():
    print("A. The schedules of worlds 1 to 8 (the reviewer's generator; its digests equal the coordinator's, audit of the brief, section 7)")
    print("   world  families met at 256 / 1,024 visits | returns in 1,024 visits, of them after more than 8 visits | returns by absence"
          " 1 / 2-4 / 5-32 / 33+ at 256 visits | at 1,024 visits")
    worlds = {w: World(w) for w in range(1, 9)}
    for w, W in worlds.items():
        out = []
        for V in (256, 1024):
            rows = W.absences(V)
            ret = [r for r in rows if r[3] is not None]
            cls = [sum(1 for r in ret if lo <= r[3] <= hi) for lo, hi in ((1, 1), (2, 4), (5, 32), (33, 10 ** 9))]
            out.append((len(set(r[1] for r in rows)), len(ret), sum(1 for r in ret if r[3] > 8), cls))
        print(f"   {w}      {out[0][0]:2d} / {out[1][0]:2d} | {out[1][1]} returns, {out[1][2]} after more than 8 | {out[0][3]} | {out[1][3]}")
    print("   the note: \"41 to 52 families\" at 1,024 visits, \"9 to 17\" at 256, \"440 to 497 of 972 to 983\" returns after more than 8 visits")
    print("   asked reduction items of add and sub in the first update of a return (on which the note reads \"kept at return\"), life of 256 visits:")
    print("   world  returns after 5-32 visits: count, items | after 33 or more: count, items | items per return, mean")
    for w, W in worlds.items():
        rows = W.absences(256)
        n = {c: [0, 0] for c in ("medium", "long", "all")}
        for v, f, new, d in rows:
            if d is None:
                continue
            k = first_update_reduction_items(W, v, f)
            n["all"][0] += 1; n["all"][1] += k
            c = "medium" if 5 <= d <= 32 else "long" if d >= 33 else None
            if c:
                n[c][0] += 1; n[c][1] += k
        print(f"   {w}      {n['medium'][0]:3d} returns, {n['medium'][1]:3d} items | {n['long'][0]:3d} returns, {n['long'][1]:3d} items | {n['all'][1] / n['all'][0]:.2f}")
    print("   the life the note's check (b) names (world 1, 16 visits, by the schedule): families of the 16 visits:", worlds[1].schedule(16))
    print("   16-visit lives by the schedule, worlds 1 to 8: number of families met:", [len(set(W.schedule(16))) for W in worlds.values()])
    for w, V in ((5, 24), (2, 32), (5, 32)):
        s = worlds[w].schedule(V)
        print(f"   world {w}, {V} visits: {len(set(s))} families; visits by family {s}")
    return worlds


# ================================================================ B. the runner's list rows
def rows_present(sched, V, vu=256):
    """The keys (at_visit, source, family, decay_updates, weights_from_visit, items) of the list rows that sw_run.py at 39e4e594
    writes for a life, in its order (Life.answer_list_before, then Life.answer_list after every visit)."""
    out = [(BEFORE, 0, BIRTH_FAMILY, 0, BEFORE, 2000)]
    out += [(k, 1, BIRTH_FAMILY, (k + 1) * vu, BEFORE, 2000) for k in range(V)]
    taken = []
    for v, g in enumerate(sched[:V]):
        first = g not in taken
        if first:
            taken.append(g)
        out += [(v, 0, f, 0, v, 200) for f in taken]
        out.append((v, 0, BIRTH_FAMILY, 0, v, 2000))
        if first:
            out += [(k, 1, g, (k - v) * vu, v, 200) for k in range(v + 1, V)]
    return out


def rows_note(sched, V, K, dist, vu=256, birth_live="every", born_decay="every"):
    """The same under the two arguments as the note words them.  The note does not say what becomes of the birth family's 2,000
    items; the two readings are arguments: birth_live "every" (after every visit, as now) or "K" (as the other families);
    born_decay "every" (the born weights' decay alone for every visit, as now) or "dist" (at the distances).
    `--list-every K` is read as: after K, 2K, ... visits and after the last."""
    at_k = lambda v: (v + 1) % K == 0 or v == V - 1
    out = [(BEFORE, 0, BIRTH_FAMILY, 0, BEFORE, 2000)]
    ks = range(V) if born_decay == "every" or dist is None else [d - 1 for d in dist if d - 1 < V]
    out += [(k, 1, BIRTH_FAMILY, (k + 1) * vu, BEFORE, 2000) for k in ks]
    taken = []
    for v, g in enumerate(sched[:V]):
        first = g not in taken
        if first:
            taken.append(g)
        out += [(v, 0, f, 0, v, 200) for f in taken if at_k(v) or f == g]
        if birth_live == "every" or at_k(v):
            out.append((v, 0, BIRTH_FAMILY, 0, v, 2000))
        if first:
            ks = range(v + 1, V) if dist is None else [v + d for d in dist if v + d < V]
            out += [(k, 1, g, (k - v) * vu, v, 200) for k in ks]
    return out


def keys_digest(rows):
    """SHA-256 over the row keys in the runner's order, one line a key: at_visit,source,family,decay_updates,weights_from_visit,items."""
    return hashlib.sha256("".join(",".join(str(x) for x in r) + "\n" for r in rows).encode()).hexdigest()


def keys_of_list(la):
    """The row keys of a committed list file, in file order, with the number of items of each run of equal keys."""
    cols = [np.asarray(la[k]).astype(np.int64) for k in ("at_visit", "source", "family", "decay_updates", "weights_from_visit")]
    key = np.stack(cols, 1)
    cut = np.nonzero(np.any(key[1:] != key[:-1], axis=1))[0] + 1
    starts = np.concatenate([[0], cut]); ends = np.concatenate([cut, [len(key)]])
    return [tuple(int(x) for x in key[s]) + (int(e - s),) for s, e in zip(starts, ends)]


def part_b(worlds):
    print("B. The runner at 39e4e594 and the change the note describes")
    src = git_bytes(M + "smallworld/sw_run.py", RUNNER_AT).decode()
    assert hashlib.sha256(src.encode()).hexdigest() == hashlib.sha256(git_bytes(M + "smallworld/sw_run.py")).hexdigest()
    print("   sw_run.py on the default branch at 7cabb13a is byte-identical to its version at 39e4e594: the change exists only in the note's words")
    line = next(l for l in src.splitlines() if l.strip().startswith("want = {"))
    print("   load_learner, the line that maps an arm to the rule of the birth it may load:")
    print("      " + line.strip())
    assert '"replay"' not in line and "'replay'" not in line
    print("      -> `life replay ... --learner F` raises KeyError('replay') there: the roster's arms `replay` and `replay-wd0` cannot run with this runner")
    # the model of the rows against committed list files
    n_ok = 0
    for arm in ("naive", "sweep"):
        for w in range(1, 9):
            for vu in (64, 128, 256):
                rec = git_json(F1 + stem(arm, w, vu) + ".json")
                if (arm, vu) == ("naive", 256) or w == 1:
                    la = git_npz(F1 + stem(arm, w, vu) + "_list.npz")
                    assert keys_of_list(la) == rows_present(list(range(8)), 8, vu), (arm, w, vu)
                    n_ok += 1
                assert rec["list"]["entries"] == sum(r[-1] for r in rows_present(list(range(8)), 8, vu))
                assert rec["provenance"]["sources_sha256"]["smallworld/sw_run.py"] == hashlib.sha256(src.encode()).hexdigest()
                assert not rec["provenance"]["git"]["sources_modified_vs_commit"]
    print(f"   the row model (rows_present) gives the keys of {n_ok} committed list files exactly, in order, and the entry count of all 48 gate receipts (46,800);")
    print("   every one of the 48 receipts records the sha256 of sw_run.py as it is at 39e4e594, with no source modified")
    # cost of an answered list item, from the calibration receipts at 256 updates a visit
    per_item, per_update = [], []
    for w in range(1, 9):
        rec = git_json(F1 + stem("naive", w) + ".json")
        per_item.append(rec["cpu"]["list_total"] / rec["list"]["entries"]); per_update.append(rec["cpu"]["mean_per_visit"] / 256)
    ci, cu = tmean(per_item), tmean(per_update)
    print(f"   calibration, naive at 256 updates a visit, worlds 1 to 8: {cu:.4f} CPU-seconds an update (the list apart), {1000 * ci:.3f} ms an answered list item")
    life = 1024 * 256 * cu
    print(f"   a life of 1,024 visits without the list: {life:,.0f} CPU-seconds = {life / 3600:.2f} CPU-hours")
    print("   answered list items in a life of 1,024 visits, and their cost as a share of that life:")
    print("   world | present runner: all (decay alone of families; birth family) | the note's change, birth family as now | birth family by the same two arguments")
    for w in range(1, 5):
        s = worlds[w].schedule(1024)
        p = rows_present(s, 1024)
        tot = sum(r[-1] for r in p); dec = sum(r[-1] for r in p if r[1] == 1 and r[2] >= 0); bf = sum(r[-1] for r in p if r[2] == BIRTH_FAMILY)
        a = sum(r[-1] for r in rows_note(s, 1024, L2_K, L2_DIST)); b = sum(r[-1] for r in rows_note(s, 1024, L2_K, L2_DIST, birth_live="K", born_decay="dist"))
        print(f"   {w}     | {tot:,} ({dec:,}; {bf:,}) = {100 * tot * ci / life:.0f}% | {a:,} = {100 * a * ci / life:.1f}% | {b:,} = {100 * b * ci / life:.1f}%")
    s1 = worlds[1].schedule(1024)
    first = {}
    for v, g in enumerate(s1):
        first.setdefault(g, v)
    print(f"   the note: \"about 10 million answered items per family for decay alone\"; world 1: at most {200 * 1023:,} a family (family 0), "
          f"{sum(200 * (1023 - v) for v in first.values()):,} over its 52 families")
    # can the rows the note describes be read as "decay alone beside what is kept"?
    print("   decay alone at the note's distances, counted from a family's first visit: at how many (family, distance) pairs")
    print("   (1) the family has not been visited again within the distance, and (2) the live network answers the family at that same visit")
    for V, ws in ((1024, range(1, 5)), (256, range(1, 9))):
        print(f"   lives of {V} visits, worlds {ws[0]} to {ws[-1]} pooled; distance: pairs | (1) | (1) and (2)")
        for d in L2_DIST:
            if d >= V:
                continue
            n = c1 = c12 = 0
            for w in ws:
                s = worlds[w].schedule(V); first = {}
                for v, g in enumerate(s):
                    first.setdefault(g, v)
                for g, v in first.items():
                    k = v + d
                    if k >= V:
                        continue
                    n += 1
                    clean = g not in s[v + 1:k + 1]
                    live = (k + 1) % L2_K == 0 or k == V - 1 or s[k] == g
                    c1 += clean; c12 += clean and live
            print(f"      {d:4d}: {n:3d} | {c1:3d} | {c12:3d}")
    # what a return could be read on instead: the family's 200 list items just before its return
    print("   returns in a life, with the visit just before each: the 200 list items of the returning family, answered there, would read")
    print("   \"kept at return\" on 200 fixed items a return in place of the 3 to 4 stream items of its first update:")
    for V, ws in ((256, range(1, 9)), (1024, range(1, 5))):
        for w in ws:
            rows = worlds[w].absences(V)
            lo = sum(1 for r in rows if r[3] is not None and r[3] >= 33); me = sum(1 for r in rows if r[3] is not None and 5 <= r[3] <= 32)
            if w in (1, 3):
                print(f"      world {w}, {V} visits: absences of 5 to 32 visits {me} returns = {200 * me:,} list items; of 33 or more {lo} returns = {200 * lo:,} list items")
    # digests an implementation can be compared with
    print("   digests of the row keys (keys_digest: one line a key, `at_visit,source,family,decay_updates,weights_from_visit,items`, in the runner's order):")
    for label, w, V, K, dist, in_turn in (("the calibration life (defaults)", 1, 8, 1, None, True),
                                          ("the note's check (b): world 1, 16 visits, defaults", 1, 16, 1, None, False),
                                          ("the note's check (b): world 1, 16 visits, K=4, distances 1,2,4,8", 1, 16, 4, (1, 2, 4, 8), False),
                                          ("proposed: world 5, 32 visits, defaults", 5, 32, 1, None, False),
                                          ("proposed: world 5, 32 visits, K=4, distances 1,2,4,8", 5, 32, 4, (1, 2, 4, 8), False)):
        s = list(range(V)) if in_turn else worlds[w].schedule(V)
        r = rows_note(s, V, K, dist, birth_live="K", born_decay="dist")
        fam_rows = sum(1 for x in r if x[1] == 0 and x[2] >= 0); skipped = sum(1 for x in rows_present(s, V) if x[1] == 0 and x[2] >= 0) - fam_rows
        print(f"      {label}: {len(r)} rows, {sum(x[-1] for x in r):,} items, {len(set(s))} families; live family rows {fam_rows}, {skipped} fewer than the defaults; {keys_digest(r)[:16]}")
    assert rows_note(list(range(8)), 8, 1, None) == rows_present(list(range(8)), 8)
    s = worlds[1].schedule(16)
    live = lambda rr: [x for x in rr if x[1] == 0 and x[2] >= 0]
    assert live(rows_note(s, 16, 4, (1, 2, 4, 8))) == live(rows_present(s, 16))
    print("      in the life of check (b) every live family row of the defaults is also written with K=4 (one family, visited at every visit):")
    print("      the argument `--list-every` changes no family's rows there, so the check cannot fail on it")
    return ci, cu


# ================================================================ C. the calibration's gate
def marked_events(W, vu, ev, groups, branch_from=None):
    """Every asked item of a life re-marked from the reviewer's own stream: (visit, update in visit, family, operation index,
    needs the reduction, right).  Asserts that the event file's task, wrap and correct columns equal the reviewer's."""
    W.visit_updates = vu
    out = []
    gi = 0
    it = W.branch(branch_from, groups) if branch_from is not None else ((v, f, t, a) for v, f in enumerate(groups) for t, a in W.visit(v, f))
    task, cor, wrap = ev["task"], ev["correct"], ev["wrap"]
    g = np.stack([ev["g1"], ev["g2"], ev["g3"]], 1).astype(np.int64)
    upd0, last_v, u_in = None, None, 0
    for v, f, taught, asked in it:
        if v != last_v:
            last_v, u_in = v, 0
        for i, a, b, prompt, target in asked:
            right = int(list(g[gi]) == target)
            wr = int(W.wrap(i, a, b))
            assert task[gi] == i and wrap[gi] == wr and cor[gi] == right and ev["visit"][gi] == v, (gi, i, int(task[gi]))
            out.append((v, u_in, f, i % 4, wr, right))
            gi += 1
        u_in += 1
    assert gi == len(task)
    return np.array(out, dtype=np.int64)


def figure(me, v, lo, hi):
    """The family's figure of visit v over updates lo .. hi - 1 of the visit: the mean of add's and sub's share right on asked
    reduction items.  Also the pooled share and its count."""
    m = (me[:, 0] == v) & (me[:, 1] >= lo) & (me[:, 1] < hi) & (me[:, 4] == 1)
    sh = [me[m & (me[:, 3] == j), 5].mean() for j in (0, 1)]
    return float(np.mean(sh)), float(me[m, 5].mean()), int(m.sum())


_gate_cache = {}


def gate_table(worlds):
    if _gate_cache:
        return _gate_cache
    for arm in ("naive", "sweep"):
        for vu in (64, 128, 256):
            for w in range(1, 9):
                ev = git_npz(F1 + stem(arm, w, vu) + ".npz")
                me = marked_events(World(w), vu, ev, list(range(8)))
                q = vu // 4
                _gate_cache[arm, vu, w] = dict(fig=[figure(me, v, vu - q, vu)[0] for v in range(8)], me=me if vu == 256 else None)
    return _gate_cache


def part_c(worlds):
    print("C. The gate, recomputed from the 48 event files (naive and sweep, worlds 1 to 8, visits of 64, 128, 256 updates)")
    G = gate_table(worlds)
    print(f"   every asked item of the 48 lives ({sum(8 * vu * 16 for vu in (64, 128, 256)) * 16:,} answers) re-marked against the reviewer's stream: task, reduction mark and right/wrong all equal the files'")
    print("   the brief: \"In at least six of eight worlds, the median over the second to the eighth family taken ... of the first-visit level on")
    print("   reduction pairs is between 0.20 and 0.90; and a life of 1,024 visits at that length is forecast within 6 CPU-hours. Use the shortest visit length that passes.\"")
    res = {}
    for arm in ("naive", "sweep"):
        for vu in (64, 128, 256):
            med = [float(np.median(G[arm, vu, w]["fig"][1:])) for w in range(1, 9)]
            inside = sum(0.20 <= m <= 0.90 for m in med); below = sum(m < 0.20 for m in med); above = sum(m > 0.90 for m in med)
            cpu = tmean([git_json(F1 + stem(arm, w, vu) + ".json")["cpu"]["mean_per_visit"] for w in range(1, 9)])
            res[arm, vu] = med
            print(f"   {arm:5s} {vu:3d}: medians {' '.join(f'{m:.3f}' for m in med)} | inside {inside}, below {below}, above {above} | "
                  f"{cpu:.2f} CPU-s a visit, 1,024 visits {cpu * 1024 / 3600:.2f} h" + (f" | passes: {'yes' if inside >= 6 and cpu * 1024 / 3600 <= 6 else 'no'}" if arm == "naive" else ""))
    table = {64: "0.059 0.112 0.068 0.065 0.068 0.053 0.084 0.133", 128: "0.169 0.434 0.169 0.322 0.213 0.096 0.195 0.301", 256: "0.368 0.566 0.301 0.457 0.491 0.368 0.436 0.636"}
    for vu, row in table.items():
        assert " ".join(f"{m:.3f}" for m in res["naive", vu]) == row, (vu, res["naive", vu])
    print("   the naive rows equal the report's table to the third decimal; 256 is the shortest length that passes (64: 0 of 8 inside; 128: 4 of 8)")
    return res


# ================================================================ D. the second lap
def part_d(worlds):
    print("D. The second lap (16 branch files: from the state after 8 visits, families 0 to 7 again, 256 updates a visit)")
    G = gate_table(worlds)
    print("   arm    family | first update | first 16 updates | last quarter (return) | last quarter (first visit) | list at the end of the return   (means over the eight worlds)")
    tab = {}
    for arm in ("naive", "sweep"):
        acc = {f: [] for f in range(8)}
        for w in range(1, 9):
            st = f"lap2/lw_{arm}_w{w}_v8_turn_br8_g0-1-2-3-4-5-6-7"
            ev = git_npz(F1 + st + ".npz"); la = git_npz(F1 + st + "_list.npz")
            me = marked_events(World(w), 256, ev, list(range(8)), branch_from=8)
            for k in range(8):
                v = 8 + k
                m1 = (me[:, 0] == v) & (me[:, 1] < 1) & (me[:, 4] == 1); m16 = (me[:, 0] == v) & (me[:, 1] < 16) & (me[:, 4] == 1)
                ls = (la["family"] == k) & (la["source"] == 0) & (la["at_visit"] == v)
                acc[k].append((me[m1, 5].sum(), m1.sum(), me[m16, 5].sum(), m16.sum(), figure(me, v, 192, 256)[0], G[arm, 256, w]["fig"][k], la["correct"][ls].mean()))
        for k in range(8):
            a = np.array(acc[k], dtype=float)
            fu = np.nanmean(np.where(a[:, 1] > 0, a[:, 0] / np.maximum(a[:, 1], 1), np.nan)); f16 = (a[:, 2] / a[:, 3]).mean()
            tab[arm, k] = (fu, f16, a[:, 4].mean(), a[:, 5].mean(), a[:, 6].mean())
            print(f"   {arm:5s}  {k}      | {fu:.3f} ({int(a[:, 1].sum())} items in 8 worlds) | {f16:.3f} | {a[:, 4].mean():.3f} | {a[:, 5].mean():.3f} | {a[:, 6].mean():.3f}")
    for arm in ("naive", "sweep"):
        pooled1 = np.mean([tab[arm, k][0] for k in range(8)]); p16 = np.mean([tab[arm, k][1] for k in range(8)])
        print(f"   {arm}: mean over families: first update {pooled1:.3f}, first 16 updates {p16:.3f}, last quarter of the return {np.mean([tab[arm, k][2] for k in range(8)]):.3f}"
              f" against {np.mean([tab[arm, k][3] for k in range(8)]):.3f} at the first visit")
    want = {("naive", 0): (0.014, 0.102, 0.672, 0.616, 0.636), ("naive", 7): (0.000, 0.049, 0.434, 0.332, 0.399), ("sweep", 2): (0.050, 0.127, 0.763, 0.730, 0.722)}
    for k, v in want.items():
        assert all(abs(x - y) < 0.0006 for x, y in zip(tab[k], v)), (k, tab[k])
    print("   the rows checked against the report's table (naive 0, naive 7, sweep 2) agree to the third decimal")


# ================================================================ E. the predictions
def part_e(worlds, res):
    print("E. The eleven predictions, scored from the files")
    G = gate_table(worlds)
    nv = {vu: res["naive", vu] for vu in (64, 128, 256)}
    sw = {vu: res["sweep", vu] for vu in (64, 128, 256)}
    ck = git_json(M + "exp45/e44/checks/E44_CHECKS.json")["before_calibration"]
    out = {}
    c1 = ck["check1"]
    out["P44.1"] = ("the reviewer's implementation gives the six digests of check 1 at the first attempt",
                    f"holds (check receipt: the world file identical to the specification {c1['spec_identical']}, its output equal to the committed one "
                    f"{c1['output_equals_OUTPUT_LW_WORLD_MAC']}, check passed {c1['passed']}; the reviewer's digests are those of the audit of the brief, section 7)")
    c4 = ck["check4"]
    out["P44.2"] = ("check 4: the worker's row-0 scores agree with the coordinator's to 2 points",
                    f"{'holds' if c4['passed'] and c4['largest_difference_points'] <= 2 else 'fails'} (check receipt: largest difference {c4['largest_difference_points']} points over {c4['lines_committed']} lines)")
    passes = {vu: sum(0.20 <= m <= 0.90 for m in nv[vu]) >= 6 for vu in (64, 128, 256)}
    out["P44.3"] = ("the gate passes at some visit length", f"{'holds' if any(passes.values()) else 'fails'} (64: {passes[64]}, 128: {passes[128]}, 256: {passes[256]})")
    out["P44.4"] = ("the shortest passing length is 64", "fails" if not passes[64] else "holds")
    out["P44.5"] = ("at 256 the median is above 0.90 in six of eight worlds", f"{'holds' if sum(m > 0.90 for m in nv[256]) >= 6 else 'fails'} ({sum(m > 0.90 for m in nv[256])} of 8)")
    out["P44.6"] = ("at 64 it is below 0.20 in six of eight worlds", f"{'holds' if sum(m < 0.20 for m in nv[64]) >= 6 else 'fails'} ({sum(m < 0.20 for m in nv[64])} of 8)")
    out["P44.10"] = ("at 64 it is above 0.90 in six of eight worlds", f"{'holds' if sum(m > 0.90 for m in nv[64]) >= 6 else 'fails'} ({sum(m > 0.90 for m in nv[64])} of 8)")
    out["P44.11"] = ("at 256 it is below 0.20 in six of eight worlds", f"{'holds' if sum(m < 0.20 for m in nv[256]) >= 6 else 'fails'} ({sum(m < 0.20 for m in nv[256])} of 8)")
    up = [np.mean(G["naive", 128, w]["fig"][5:8]) > np.mean(G["naive", 128, w]["fig"][1:4]) for w in range(1, 9)]
    out["P44.7"] = ("naive at 128: families 5 to 7 above families 1 to 3 in six of eight worlds", f"{'holds' if sum(up) >= 6 else 'fails'} ({sum(up)} of 8)")
    near = [abs(a - b) <= 0.15 for a, b in zip(sw[256], nv[256])]
    out["P44.8"] = ("the sweep's median within 0.15 of naive's at 256 in six of eight worlds", f"{'holds' if sum(near) >= 6 else 'fails'} ({sum(near)} of 8; differences {' '.join(f'{a - b:+.3f}' for a, b in zip(sw[256], nv[256]))})")
    kept_med = []
    for w in range(1, 9):
        la = git_npz(F1 + stem("naive", w) + "_list.npz")
        sh = lambda f, v: la["correct"][(la["family"] == f) & (la["source"] == 0) & (la["at_visit"] == v)].mean()
        ratios = [sh(f, f + 1) / sh(f, f) for f in range(7) if sh(f, f) > 0]
        kept_med.append(float(np.median(ratios)))
    out["P44.9"] = ("naive at 256: median over families 0 to 6 of the list share one visit later over the share at the end of the own visit is 0.20 or less, in six of eight worlds",
                    f"{'holds' if sum(k <= 0.20 for k in kept_med) >= 6 else 'fails'} ({sum(k <= 0.20 for k in kept_med)} of 8; medians {' '.join(f'{k:.3f}' for k in kept_med)})")
    for k in ("P44.1", "P44.2", "P44.3", "P44.4", "P44.5", "P44.6", "P44.10", "P44.11", "P44.7", "P44.8", "P44.9"):
        print(f"   {k:6s} {out[k][0]}: {out[k][1]}")
    return ck


# ================================================================ F. the numpy reader on saved weights
class _Tolerant(pickle.Unpickler):
    """Reads a torch.save file's pickle without torch and without executing anything from it: tensors become ("T", storage, offset,
    shape); every other class becomes an inert record of its arguments."""

    def find_class(self, mod, name):
        import collections
        if mod == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if mod == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, stride, *a, **k: ("T", storage, offset, tuple(size))
        if mod == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *a, **k: data
        if mod == "torch" and name.endswith("Storage"):
            return name
        if mod == "_codecs" and name == "encode":
            return lambda s, enc="latin1": s.encode(enc)
        return lambda *a, **k: ("OBJ", mod, name)

    def persistent_load(self, pid):
        return (pid[1], pid[2])


def state_weights(blob):
    """The weights (float32) in the `model` entry of a life state written by sw_run.py; also the entry's cfg."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]; root = pk[:-len("data.pkl")]
    s = _Tolerant(z.open(pk)).load()
    Wt = {}
    for k, (_, (stype, key), offset, shape) in s["model"].items():
        assert stype == "FloatStorage"
        n = int(np.prod(shape))
        Wt[k] = np.frombuffer(z.read(root + "data/" + key), dtype="<f4")[offset:offset + n].reshape(shape).copy()
    return Wt, s


def plain_weights(blob):
    z = zipfile.ZipFile(io.BytesIO(blob))
    pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]; root = pk[:-len("data.pkl")]
    s = _Tolerant(z.open(pk)).load()
    return {k: np.frombuffer(z.read(root + "data/" + key), dtype="<f4")[offset:offset + int(np.prod(shape))].reshape(shape).copy()
            for k, (_, (stype, key), offset, shape) in s.items()}


def tensors_sha(Wt):
    """sw_run.tensors_sha (exp20_reference's): names, dtypes, shapes and values in sorted order."""
    h = hashlib.sha256()
    for k in sorted(Wt):
        v = np.ascontiguousarray(Wt[k])
        h.update(k.encode()); h.update(b"torch.float32"); h.update(str(tuple(int(x) for x in v.shape)).encode()); h.update(v.tobytes())
    return h.hexdigest()


def greedy(Wt, prompts, n=3, chunk=500):
    """Greedy generation of n bytes for prompts of one length, by the numpy reader (torch's GRU convention), state carried forward."""
    out = []
    Whh, bhh = Wt["rnn.weight_hh_l0"], Wt["rnn.bias_hh_l0"]; U = Whh.shape[1]
    z1_tok = np.tanh(Wt["book.weight"] @ Wt["l1.weight"].T + Wt["l1.bias"])
    gi_tok = z1_tok @ Wt["rnn.weight_ih_l0"].T + Wt["rnn.bias_ih_l0"]; l2_tok = z1_tok @ Wt["l2.weight"].T + Wt["l2.bias"]
    for c in range(0, len(prompts), chunk):
        x = np.array([[BOS] + list(p) for p in prompts[c:c + chunk]], dtype=np.int64)
        h = np.zeros((len(x), U), np.float32)

        def stepf(tok, h):
            gi = gi_tok[tok]; gh = h @ Whh.T + bhh
            r = sigmoid(gi[:, :U] + gh[:, :U]); z = sigmoid(gi[:, U:2 * U] + gh[:, U:2 * U]); nn = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:])
            h = (1 - z) * nn + z * h
            return h, (np.tanh(l2_tok[tok] + h @ Wt["lm.weight"].T + Wt["lm.bias"]) @ Wt["l3.weight"].T + Wt["l3.bias"]).argmax(1)
        for t in range(x.shape[1]):
            h, nxt = stepf(x[:, t], h)
        gen = [nxt]
        for _ in range(n - 1):
            h, nxt = stepf(nxt, h)
            gen.append(nxt)
        out += np.stack(gen, 1).tolist()
    return out


def gen_cols(g):
    out, ended = [], False
    for x in g:
        out.append(GEN_END if ended else int(x)); ended = ended or x == EOS
    return out


def r_items(seed, fraction=0.5):
    """E41's accuracy-R items: 1,000 held-out reduction pairs each of add and sub mod 89 (codes p and q), no letters."""
    out = []
    for j, t in enumerate(DEV[:2]):
        held = e26_split(t, fraction, seed)[1]
        pairs = [(a, b) for a, b in held if (a + b >= 89 if t[0] == "add" else a - b < 0)]
        smp = Random(f"e41|{seed}|{round(fraction * 100)}|{t[0]}").sample(sorted(pairs), 1000)
        sha = hashlib.sha256(json.dumps([list(q) for q in smp]).encode()).hexdigest()
        for a, b in smp:
            prompt = list(t[2]) + list(f"{a:02d}".encode()) + [ord("+" if t[0] == "add" else "-")] + list(f"{b:02d}".encode()) + [ord("=")]
            out.append((65534 + j, a, b, prompt, list(f"{op_answer(t[0], 89, a, b):02d}".encode()) + [EOS], sha))
    return out


def decayed(Wt, factor):
    return {k: (v.copy() if k.startswith("book.") else (v * np.float32(factor)).astype(np.float32)) for k, v in Wt.items()}


def part_f(worlds):
    print("F. The numpy reader on saved weights: world 1, naive, 256 updates a visit, 8 visits in turn")
    st = stem("naive", 1)
    rec = git_json(F1 + st + ".json"); la = git_npz(F1 + st + "_list.npz")
    blob = git_bytes(F1 + st + "_state_v8.pt")
    assert hashlib.sha256(blob).hexdigest() == rec["states"][0]["file_sha256"]
    Wt, s = state_weights(blob)
    assert tensors_sha(Wt) == rec["final"]["weights_tensors_sha256"]
    print(f"   state after 8 visits: file hash and tensors hash equal the receipt's ({rec['final']['weights_tensors_sha256'][:16]}); {sum(v.size for v in Wt.values()):,} parameters")
    W = worlds[1]
    n_all = n_same = n_right = 0
    for f in range(8):
        items = W.probe_list([f])
        m = (la["family"] == f) & (la["source"] == 0) & (la["at_visit"] == 7)
        assert m.sum() == 200 and list(la["task"][m]) == [it[0] for it in items] and list(la["a"][m]) == [it[1] for it in items] and list(la["b"][m]) == [it[2] for it in items]
        gen = greedy(Wt, [it[3] for it in items])
        mine = np.array([gen_cols(g) for g in gen]); theirs = np.stack([la["g1"][m], la["g2"][m], la["g3"][m]], 1).astype(np.int64)
        right = np.array([int(g == it[4]) for g, it in zip(gen, items)])
        assert (right == la["correct"][m]).all()
        n_all += 200; n_same += int((mine == theirs).all(1).sum()); n_right += int(right.sum())
        print(f"   family {f} (modulus {W.moduli[f]}): list share at the end of the life {right.mean():.3f}; generated bytes equal the file's on {int((mine == theirs).all(1).sum())} of 200 items")
    assert n_same == n_all
    ri = r_items(73129)
    assert {it[0]: it[5] for it in ri} == {65534: rec["learner"]["reduction_sample_sha256"]["add89"], 65535: rec["learner"]["reduction_sample_sha256"]["sub89"]}
    m = (la["family"] == BIRTH_FAMILY) & (la["source"] == 0) & (la["at_visit"] == 7)
    gen = greedy(Wt, [it[3] for it in ri]); right = np.array([int(g == it[4]) for g, it in zip(gen, ri)])
    assert (right == la["correct"][m]).all() and (np.array([gen_cols(g) for g in gen]) == np.stack([la["g1"][m], la["g2"][m], la["g3"][m]], 1)).all()
    print(f"   the birth family's 2,000 R items at the end of the life: {right.mean():.4f}, item by item as the file ({n_all + 2000:,} items compared in all, every generated byte equal)")
    # the born weights and decay alone
    born = plain_weights(git_bytes(M + rec["learner"]["file"]))
    assert tensors_sha(born) == rec["learner"]["tensors_sha256"]
    gen = greedy(born, [it[3] for it in ri]); right0 = np.array([int(g == it[4]) for g, it in zip(gen, ri)])
    m0 = (la["family"] == BIRTH_FAMILY) & (la["source"] == 0) & (la["at_visit"] == BEFORE)
    assert (right0 == la["correct"][m0]).all()
    print(f"   the born network (E41's B birth, half width, seed 73129) before visit 0: R {right0.mean():.4f} = the receipt's final row ({rec['learner']['final_row']['R']}%)")
    line = []
    for k in range(8):
        u = 256 * (k + 1)
        gen = greedy(decayed(born, 0.99975 ** u), [it[3] for it in ri]); r = np.array([int(g == it[4]) for g, it in zip(gen, ri)])
        mk = (la["family"] == BIRTH_FAMILY) & (la["source"] == 1) & (la["at_visit"] == k)
        assert (la["decay_updates"][mk] == u).all()
        line.append((u, r.mean(), la["correct"][mk].mean(), int((r != la["correct"][mk]).sum())))
    print("   decay alone of the born weights (every parameter but the byte table times 0.99975^u): u, the reader's share, the file's share, items that differ")
    print("      " + "; ".join(f"{u}: {a:.4f}, {b:.4f}, {d}" for u, a, b, d in line))
    assert all(d <= 2 for *_, d in line)
    return rec


# ================================================================ G. the learners
def e41_threshold(curve, key="R", level=80.0):
    """First row at which the score is at least `level` and stays so for that row and the next two (E41's rule of three rows)."""
    v = [r[key] for r in curve]
    for i in range(len(v) - 2):
        if v[i] >= level and v[i + 1] >= level and v[i + 2] >= level:
            return curve[i]["step"]
    return None


def part_g():
    print("G. The learners of release 1: E41's half-width births at f = 0.50, and the width rule")
    for seed in SEEDS:
        for rule in ("B", "S"):
            p = M + f"exp45/e41/e41_{rule}_h256_f50_{seed}"
            if not exists(p + ".pt") or not exists(p + ".json"):
                print(f"   seed {seed} rule {rule}: NOT on the default branch"); continue
            rec = git_json(p + ".json"); blob = git_bytes(p + ".pt")
            Wt = plain_weights(blob)
            ok = hashlib.sha256(blob).hexdigest() == rec["weights_sha256"] and tensors_sha(Wt) == rec["tensors_sha256"]
            ri = r_items(seed)
            gen = greedy(Wt, [it[3] for it in ri]); right = np.array([int(g == it[4]) for g, it in zip(gen, ri)])
            cnt = {"add89": int(right[:1000].sum()), "sub89": int(right[1000:].sum())}
            last = rec["curve"][-1]
            same = cnt == last["R_counts"] and {it[0]: it[5] for it in ri} == {65534: rec["reduction_sample_sha256"]["add89"], 65535: rec["reduction_sample_sha256"]["sub89"]}
            print(f"   seed {seed} rule {rule}: committed, complete {rec['complete']}, H {rec['H']}, U {rec['U']}, dose {rec['dose']}; file and tensors hashes equal the receipt's: {ok}; "
                  f"R by the numpy reader {cnt} = the receipt's final row {last['R_counts']}: {same}")
            assert ok and same and rec["complete"] and (rec["H"], rec["U"]) == (256, 128)
    print("   the width rule (brief): the half width if it reaches 80% on R at f = 0.50 with the fewest weight-items in at least two of three seeds under backpropagation")
    print("   seed    width H: updates to 80% on R (three rows), parameters, weight-items = updates x 64 items x parameters")
    for seed in SEEDS:
        row = []
        for H in (256, 512, 1024):
            p = M + f"exp45/e41/e41_B_h{H}_f50_{seed}.json"
            if not exists(p):
                row.append((H, None, None, None)); continue
            rec = git_json(p)
            u = e41_threshold(rec["curve"]); P = rec.get("parameters") or rec.get("params")
            row.append((H, u, P, None if u is None or P is None else u * 64 * P))
        best = min((r for r in row if r[3] is not None), key=lambda r: r[3])[0] if any(r[3] is not None for r in row) else None
        print(f"   {seed}  " + " | ".join(f"H {H}: {u} updates, {P} parameters, {('%.3g' % wi) if wi else 'n/a'}" for H, u, P, wi in row) + f" | cheapest: H {best}")


# ================================================================ H. the roster's cost and the buffer
def part_h(worlds, ci, cu):
    print("H. The roster's cost and the replay store")
    life = lambda V: V * 256 * cu / 3600
    runs = (("trial", 2, 1024), ("decay", 16, 256), ("long", 6, 1024), ("short", 8, 256))
    tot = 0
    for name, n, V in runs:
        h = n * life(V) * 1.1; tot += h
        print(f"   {name:5s}: {n:2d} lives of {V:4d} visits: {h:.1f} CPU-hours with the note's 10% for the buffer")
    print(f"   total {tot:.1f} CPU-hours (the note: about 59); the calibration's figure leaves the list out")
    s = worlds[1].schedule(1024)
    for label, kw in (("birth family answered after every visit and its decay alone for every visit, as now", {}),
                      ("birth family by the same two arguments", dict(birth_live="K", born_decay="dist"))):
        n1024 = sum(r[-1] for r in rows_note(s, 1024, L2_K, L2_DIST, **kw)); n256 = sum(r[-1] for r in rows_note(s[:256], 256, L2_K, L2_DIST, **kw))
        add = (8 * n1024 + 24 * n256) * ci / 3600
        print(f"   the list under the note's arguments, {label}: {n1024:,} items a long life, {n256:,} a short one: {add:.1f} CPU-hours over the 32 lives ({100 * add / tot:.1f}%)")
    P = 317457
    cap = (4 * P) // 23
    print(f"   the store: 4 x {P:,} = {4 * P:,} bytes; {cap:,} items of at most 23 bytes (the note: 55,209)")
    print("   world | distinct taught pairs of the families met in 256 / 1,024 visits | new taught items offered to the store in 256 / 1,024 visits")
    for w in range(1, 9):
        W = worlds[w]
        d = []
        for V in (256, 1024):
            fam = set(W.schedule(V)); d.append(sum(len(W.taught[4 * f + j]) for f in fam for j in range(4)))
        print(f"   {w}     | {d[0]:,} / {d[1]:,} | {256 * 256 * 48:,} / {1024 * 256 * 48:,}")
    print(f"   an item's mean length: code 2, operands 4, symbol 1, '=' 1, answer 3 and 6 letters on average = 17 bytes; {cap:,} items then hold about {17 * cap:,} bytes")
    print("   new taught items a visit: naive 256 x 48 = 12,288; replay at rho = 0.5: 256 x 24 = 6,144, the number the calibration taught at 128 updates a visit")


if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if a in tuple("ABCDEFGH")] or list("ABCDEFGH")
    worlds = part_a() if "A" in want else {w: World(w) for w in range(1, 9)}
    ci = cu = None
    if "B" in want:
        ci, cu = part_b(worlds)
    res = part_c(worlds) if ("C" in want or "E" in want) else None
    if "D" in want:
        part_d(worlds)
    if "E" in want:
        part_e(worlds, res)
    if "F" in want:
        part_f(worlds)
    if "G" in want:
        part_g()
    if "H" in want:
        if ci is None:
            per_item, per_update = [], []
            for w in range(1, 9):
                rec = git_json(F1 + stem("naive", w) + ".json")
                per_item.append(rec["cpu"]["list_total"] / rec["list"]["entries"]); per_update.append(rec["cpu"]["mean_per_visit"] / 256)
            ci, cu = tmean(per_item), tmean(per_update)
        part_h(worlds, ci, cu)
    print(f"done in {time.time() - T0:.0f} s")
