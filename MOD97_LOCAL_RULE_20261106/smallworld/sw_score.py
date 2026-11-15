"""E39 (brief OPUS_TASK_E39_SMALL_WORLD_PROTOCOL_L0_L1_20261115.md, revision 2): the scorer. Reads an event file, never weights.

Throughout: a mean of the last 32 asked items exists only once the task has 32 (WINDOW); the 0.8 criterion is at least 26 right of 32;
tasks are numbered by first visit, then id (n = 1, 2, ...); "taught items" is `seen`. Only the first `visits` visits are read.

  score_by_age, blind_by_age   share of asked items right in each visit; the blind predictor's share
  first_visit                  per task: n, task, its first visit, level = mean of its last 32 asked items in that visit, cost = its `seen`
                               at the first asked item of that visit at which the mean of its last 32 (within the visit) is >= 0.8 (None:
                               not reached in the first visit)
  learning_trend               least-squares slope of first-visit level on ln(n), n >= 3 (the one compared)
  cost                         per task: `seen` at the first asked item at which the mean of its last 32 asked items (over the whole life)
                               is >= 0.8; None if not reached, with the mean of its last 32 (reported, not compared)
  cost_trend                   least-squares slope of ln(cost) on ln(n) over tasks that reached, n >= 2 (reported, not compared)
  kept / kept_pooled           per return of a task (a visit after its first): the share right among its asked items in the visit's first
                               update (answered before any update of the visit), split by wrap, and the same over its first 32 items of the
                               visit; the level before (mean of its last 32 in its previous visit); pooled over the items of all returns in
                               each absence class (short 1-4, medium 5-32, long 33+)
  relearn                      per return: `seen` within the visit (at the item, minus at the task's first item of the visit) at the first
                               asked item at which the mean of the visit's own last 32 is >= 0.8; None if not reached in the visit
  symmetric                    score on asked items of symmetric operations (c1 = c2, max, min), split by `mirror`
  late_level                   mean of score_by_age over the last quarter of the visits (the last max(1, visits // 4))
  score_trend                  least-squares slope of score_by_age on ln(visit + 1) over the second half (visits // 2 .. visits - 1)
  end_of_visit                 per task and visit: the mean of its last 32 asked items in that visit (the first group's is reported in L1)

Which tasks are symmetric is a property of the world (sw_world.World(seed).tasks), not of the life; it is the scorer's only other input.

    python3 sw_score.py <events.npz> --world S [--visits V] [--to TABLE.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WINDOW, LEVEL = 32, 0.8
NEED = math.ceil(LEVEL * WINDOW - 1e-9)          # 26 right of 32
CLASSES = (("short", 1, 4), ("medium", 5, 32), ("long", 33, 10 ** 9))


def absence_class(d):
    return next(name for name, lo, hi in CLASSES if lo <= d <= hi)


def symmetric_tasks(world_seed):
    import sw_world as W
    w = W.World(world_seed)
    return {t["id"] for t in w.tasks if t["op"][0] != "lin" or t["op"][1][0] == t["op"][1][1]}


def slope(x, y):
    """Least-squares slope; None with fewer than two points or no spread in x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.all(x == x[0]):
        return None
    xm = x - x.mean()
    return float((xm * (y - y.mean())).sum() / (xm * xm).sum())


def reach(c, sn):
    """`seen` at the first item at which the last WINDOW items hold at least NEED right; None if never (or fewer than WINDOW items)."""
    if len(c) < WINDOW:
        return None
    cs = np.concatenate([[0], np.cumsum(c)])
    run = cs[WINDOW:] - cs[:-WINDOW]                               # run[i] = right among items i .. i + WINDOW - 1
    hit = np.nonzero(run >= NEED)[0]
    return int(sn[hit[0] + WINDOW - 1]) if len(hit) else None


def last_mean(c):
    return float(c[-WINDOW:].mean()) if len(c) >= WINDOW else None


def frac(c):
    return dict(right=int(c.sum()), n=int(len(c)), share=float(c.mean()) if len(c) else None)


def score(ev, visits, sym):
    keep = np.asarray(ev["visit"]) < visits
    col = {k: np.asarray(ev[k])[keep].astype(np.int64) for k in ("update", "visit", "task", "correct", "blind", "wrap", "mirror", "seen")}
    upd, vis, task, cor, blind, wrap, mir, seen = (col[k] for k in ("update", "visit", "task", "correct", "blind", "wrap", "mirror", "seen"))
    out = dict(visits=visits, entries=int(len(cor)), window=WINDOW, level=LEVEL)
    sba, bba = [], []
    first_update = {}
    for v in range(visits):
        s = vis == v
        sba.append(float(cor[s].mean()) if s.any() else None); bba.append(float(blind[s].mean()) if s.any() else None)
        if s.any():
            first_update[v] = int(upd[s].min())
    out["score_by_age"], out["blind_by_age"] = sba, bba
    firstv = {}
    for t, v in zip(task.tolist(), vis.tolist()):
        firstv.setdefault(t, v)
    order = sorted(firstv, key=lambda t: (firstv[t], t))
    first_visit, cost, kept, relearn, eov = [], [], [], [], []
    for n, t in enumerate(order, 1):
        m = task == t
        c, sn, vt, wt, ut = cor[m], seen[m], vis[m], wrap[m], upd[m]
        f = vt == firstv[t]
        first_visit.append(dict(n=n, task=int(t), visit=int(firstv[t]), level=last_mean(c[f]), cost=reach(c[f], sn[f])))
        r = reach(c, sn)
        cost.append(dict(n=n, task=int(t), cost=r, level=None if r is not None else last_mean(c)))
        vs = sorted(set(vt.tolist()))
        for v in vs:
            eov.append(dict(task=int(t), n=n, visit=int(v), level=last_mean(c[vt == v]), items=int((vt == v).sum())))
        for prev, v in zip(vs, vs[1:]):
            mv = vt == v
            cv, sv, wv, uv = c[mv], sn[mv], wt[mv], ut[mv]
            fu = uv == first_update[v]                                   # the visit's first update: before any update of the visit
            head = np.arange(len(cv)) < WINDOW
            d = v - prev
            kept.append(dict(task=int(t), n=n, visit=int(v), absence=int(d), cls=absence_class(d), before=last_mean(c[vt == prev]),
                             first_update=frac(cv[fu]), first_update_wrap=frac(cv[fu & (wv == 1)]), first_update_nowrap=frac(cv[fu & (wv == 0)]),
                             first32=frac(cv[head]), first32_wrap=frac(cv[head & (wv == 1)]), first32_nowrap=frac(cv[head & (wv == 0)])))
            rr = reach(cv, sv)
            relearn.append(dict(task=int(t), n=n, visit=int(v), absence=int(d), cls=absence_class(d), items=None if rr is None else int(rr - sv[0])))
    out.update(first_visit=first_visit, cost=cost, kept=kept, relearn=relearn, end_of_visit=eov)
    pooled = {}
    for name, _, _ in CLASSES:
        ks = [k for k in kept if k["cls"] == name]
        pooled[name] = dict(returns=len(ks))
        for part in ("first_update", "first_update_wrap", "first_update_nowrap", "first32", "first32_wrap", "first32_nowrap"):
            r_ = sum(k[part]["right"] for k in ks); n_ = sum(k[part]["n"] for k in ks)
            pooled[name][part] = dict(right=r_, n=n_, share=r_ / n_ if n_ else None)
        rl = [x["items"] for x in relearn if x["cls"] == name]
        pooled[name]["relearn_reached"] = sum(x is not None for x in rl)
        pooled[name]["relearn_median_items"] = float(np.median([x for x in rl if x is not None])) if any(x is not None for x in rl) else None
    out["kept_pooled"] = pooled
    s_ = np.isin(task, sorted(sym))
    out["symmetric"] = dict(all=frac(cor[s_]), mirror=frac(cor[s_ & (mir == 1)]), no_mirror=frac(cor[s_ & (mir == 0)]),
                            non_symmetric=frac(cor[~s_]), blind_mirror=frac(blind[s_ & (mir == 1)]))
    q = visits - max(1, visits // 4)
    tail = [x for x in sba[q:] if x is not None]
    out["late_level"] = float(np.mean(tail)) if tail else None
    half = [(v, sba[v]) for v in range(visits // 2, visits) if sba[v] is not None]
    out["score_trend"] = slope([math.log(v + 1) for v, _ in half], [s for _, s in half])
    fv = [(d["n"], d["level"]) for d in first_visit if d["n"] >= 3 and d["level"] is not None]
    out["learning_trend"] = slope([math.log(n) for n, _ in fv], [l for _, l in fv])
    rc = [(d["n"], d["cost"]) for d in cost if d["cost"] and d["n"] >= 2]
    out["cost_trend"] = slope([math.log(n) for n, _ in rc], [math.log(c) for _, c in rc])
    out["mean_score"] = float(cor[vis < visits].mean()) if len(cor) else None
    return out


def load(path):
    f = np.load(path)
    return {k: f[k] for k in f.files}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("events"); ap.add_argument("--world", type=int, required=True); ap.add_argument("--visits", type=int); ap.add_argument("--to")
    a = ap.parse_args(argv)
    ev = load(a.events)
    V = a.visits if a.visits is not None else int(ev["visit"].max()) + 1
    r = score(ev, V, symmetric_tasks(a.world))
    if a.to:
        assert not Path(a.to).exists(), f"refusing an existing path: {a.to}"
        Path(a.to).write_text(json.dumps(r, indent=1))
    print(json.dumps({k: r[k] for k in ("visits", "entries", "mean_score", "late_level", "score_trend", "learning_trend", "cost_trend")}))
    return r


if __name__ == "__main__":
    main(sys.argv[1:])
