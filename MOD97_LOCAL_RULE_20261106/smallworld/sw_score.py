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
BIRTH_VISIT = 65535                              # E39 revision 4: the `visit` value of a birth's asked items
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


def score(ev, visits, sym, birth_tasks=None):
    """birth_tasks (E39 revision 4): the tasks trained in a birth before visit 0. Their rows (visit = BIRTH_VISIT) are in no visit. They are
    left out of first_visit, cost and both trends; n counts from the first task that is not a birth task, and the trend of learning is
    fitted from n = 1. The birth counts as a visit before visit 0 (visit -1): every life visit of a birth task is a return, the first with
    absence v + 1 and, as its level before, the mean of the task's last 32 asked items in the birth. Without birth_tasks: unchanged."""
    born_set = set(birth_tasks or ())
    vis_all = np.asarray(ev["visit"])
    bm = vis_all == BIRTH_VISIT
    b_task, b_cor = np.asarray(ev["task"])[bm].astype(np.int64), np.asarray(ev["correct"])[bm].astype(np.int64)
    keep = vis_all < visits
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
    order = sorted((t for t in firstv if t not in born_set), key=lambda t: (firstv[t], t))
    born = sorted(born_set)
    first_visit, cost, kept, relearn, eov = [], [], [], [], []
    for n, t in [(None, t) for t in born] + list(enumerate(order, 1)):
        m = task == t
        c, sn, vt, wt, ut = cor[m], seen[m], vis[m], wrap[m], upd[m]
        vs = sorted(set(vt.tolist()))
        if n is None:                                                    # a birth task: the birth is visit -1
            cb = b_cor[b_task == t]
            eov.append(dict(task=int(t), n=None, visit=-1, level=last_mean(cb), items=int(len(cb))))
        else:
            f = vt == firstv[t]
            first_visit.append(dict(n=n, task=int(t), visit=int(firstv[t]), level=last_mean(c[f]), cost=reach(c[f], sn[f])))
            r = reach(c, sn)
            cost.append(dict(n=n, task=int(t), cost=r, level=None if r is not None else last_mean(c)))
        for v in vs:
            eov.append(dict(task=int(t), n=n, visit=int(v), level=last_mean(c[vt == v]), items=int((vt == v).sum())))
        if n is None:
            vs = [-1] + vs
        for prev, v in zip(vs, vs[1:]):
            mv = vt == v
            cv, sv, wv, uv = c[mv], sn[mv], wt[mv], ut[mv]
            fu = uv == first_update[v]                                   # the visit's first update: before any update of the visit
            head = np.arange(len(cv)) < WINDOW
            d = v - prev
            kept.append(dict(task=int(t), n=n, visit=int(v), absence=int(d), cls=absence_class(d),
                             before=last_mean(c[vt == prev]) if prev >= 0 else last_mean(b_cor[b_task == t]),
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
    fv = [(d["n"], d["level"]) for d in first_visit if d["n"] >= (1 if born_set else 3) and d["level"] is not None]
    out["learning_trend"] = slope([math.log(n) for n, _ in fv], [l for _, l in fv])
    rc = [(d["n"], d["cost"]) for d in cost if d["cost"] and d["n"] >= 2]
    out["cost_trend"] = slope([math.log(n) for n, _ in rc], [math.log(c) for _, c in rc])
    out["mean_score"] = float(cor[vis < visits].mean()) if len(cor) else None
    if born_set:
        out["birth"] = dict(tasks=born, entries=int(bm.sum()),
                            per_task={int(t): dict(n=int((b_task == t).sum()), share=float(b_cor[b_task == t].mean()) if (b_task == t).any() else None,
                                                   last32=last_mean(b_cor[b_task == t])) for t in born})
    return out


def criterion_checks(ev, tasks, start=0, end=None, every=64, rows=None):
    """E39 revision 4: a task is at the criterion at a check, made every `every` updates, if the share right over all of its asked items
    answered in the last `every` updates is at least 0.8. Check k (k = 1, 2, ...) is made after start + every * k updates and reads the
    asked items of updates start + every * (k - 1) .. start + every * k - 1. Returns [(updates since start, {task: (right, n)})]."""
    upd = np.asarray(ev["update"]).astype(np.int64); task = np.asarray(ev["task"]).astype(np.int64); cor = np.asarray(ev["correct"]).astype(np.int64)
    if rows is not None:
        upd, task, cor = upd[rows], task[rows], cor[rows]
    end = int(upd.max()) + 1 if end is None else end
    out = []
    for k in range(1, (end - start) // every + 1):
        lo, hi = start + every * (k - 1), start + every * k
        sel = (upd >= lo) & (upd < hi)
        out.append((every * k, {int(t): (int(cor[sel & (task == t)].sum()), int((sel & (task == t)).sum())) for t in tasks}))
    return out


def first_at(checks, task, level):
    """The updates at the first check at which the task's share is at least `level`; None if never."""
    for u, d in checks:
        r, n = d[task]
        if n and r / n >= level - 1e-12:
            return u
    return None


def solo_quantities(ev, tid, losses=None):
    """L1a's recorded quantities for a solo event file (one task, visit 0 throughout)."""
    ck = criterion_checks(ev, [tid])
    cor = np.asarray(ev["correct"]).astype(np.int64); sn = np.asarray(ev["seen"]).astype(np.int64)
    upd = np.asarray(ev["update"]).astype(np.int64); wrap = np.asarray(ev["wrap"]).astype(np.int64)
    w32 = None
    if len(cor) >= WINDOW:
        cs = np.concatenate([[0], np.cumsum(cor)]); run = cs[WINDOW:] - cs[:-WINDOW]
        hit = np.nonzero(run >= NEED)[0]
        if len(hit):
            i = hit[0] + WINDOW - 1
            w32 = dict(seen=int(sn[i]), update=int(upd[i]))
    tail = slice(len(cor) - 256, len(cor))
    ct, wt = cor[tail], wrap[tail]
    return dict(updates=int(upd.max()) + 1, checks=[(u, d[tid][0], d[tid][1]) for u, d in ck],
                updates_to_criterion=first_at(ck, tid, LEVEL), updates_to_half=first_at(ck, tid, 0.5), window32=w32,
                last256=dict(all=frac(ct), wrap=frac(ct[wt == 1]), nowrap=frac(ct[wt == 0])),
                loss_last64=float(np.mean(losses[-64:])) if losses is not None else None)


# ---------------------------------------------------------------- E44: the long-life world (lw_world), from the event file and the list
LW_OPS = ("add", "sub", "max", "min")           # task 4 g + j is operation j of family g
LW_FIRST = 16                                   # "over its first 16 updates"
BEFORE_LIFE, BIRTH_FAMILY = 65535, -1           # the list's at_visit before visit 0; the list's family number of the birth family


def share(cor, sel):
    n = int(sel.sum()); r = int(cor[sel].sum())
    return dict(right=r, n=n, share=(r / n) if n else None)


def mean2(a, b):
    return None if a is None or b is None else (a + b) / 2


def visit_range(ev, v):
    """(first update, last update + 1) of visit v: every update of a visit asks items of the visit's family."""
    u = np.asarray(ev["update"])[np.asarray(ev["visit"]) == v]
    return int(u.min()), int(u.max()) + 1


def lw_levels(ev, v, g, lo, hi):
    """Shares right of family g's asked items of visit v in updates lo .. hi - 1: add and sub on reduction pairs (`wrap`) and on pairs
    without, max and min; the family's figure is the mean of add and sub on reduction pairs (none_figure: the same without reduction)."""
    u, vis, task = np.asarray(ev["update"]), np.asarray(ev["visit"]), np.asarray(ev["task"])
    cor, wrap = np.asarray(ev["correct"]).astype(np.int64), np.asarray(ev["wrap"])
    w = (vis == v) & (u >= lo) & (u < hi)
    d = {}
    for j, op in enumerate(LW_OPS):
        t = w & (task == 4 * g + j)
        if op in ("add", "sub"):
            d[f"{op}_reduction"] = share(cor, t & (wrap == 1)); d[f"{op}_none"] = share(cor, t & (wrap == 0))
        else:
            d[op] = share(cor, t)
    d["figure"] = mean2(d["add_reduction"]["share"], d["sub_reduction"]["share"])
    d["none_figure"] = mean2(d["add_none"]["share"], d["sub_none"]["share"])
    red = (w & (wrap == 1) & ((task == 4 * g) | (task == 4 * g + 1)))
    d["reduction_pooled"] = share(cor, red)
    return d


def lw_first_visits(ev, world, visit_updates, visits):
    """E44's scorer item 3, per family in the order first met (n = 1, 2, ...): the first-visit level on reduction pairs over the last
    quarter of its first visit (the last 16, 32, 64, 8 and 128 updates at visits of 64, 128, 256, 32 and 512), with the same over the whole
    first visit and over its first 16 updates. `visits`: [(visit, family)] in order."""
    out, q = [], visit_updates // 4
    assert q * 4 == visit_updates
    for v, g in visits:
        if any(r["family"] == g for r in out):
            continue
        u0, u1 = visit_range(ev, v)
        assert u1 - u0 == visit_updates, (v, u0, u1)
        t = world.groups[g][0]
        out.append(dict(n=len(out) + 1, family=g, visit=v, p=t["p"], taught_pairs=len(t["taught"]),
                        last_quarter=lw_levels(ev, v, g, u1 - q, u1), whole=lw_levels(ev, v, g, u0, u1),
                        first16=lw_levels(ev, v, g, u0, u0 + LW_FIRST), window=dict(last_quarter_updates=q, first_updates=LW_FIRST)))
    return out


def lw_gate_median(first, ns=range(2, 9)):
    """The median over the second to the eighth family taken (n = 2 .. 8) of the first-visit level on reduction pairs (last quarter)."""
    xs = [r["last_quarter"]["figure"] for r in first if r["n"] in ns]
    return float(np.median(xs)) if xs and all(x is not None for x in xs) else None


def lw_trend(first, ns=range(2, 9)):
    """The least-squares slope of the family's figure (last quarter) against ln n over n = 2 .. 8, the first family taken being n = 1."""
    rs = [r for r in first if r["n"] in ns]
    return slope([math.log(r["n"]) for r in rs], [r["last_quarter"]["figure"] for r in rs])


def lw_by_visit(ev, visits):
    """Score by age (share right over all asked items of each visit) and the blind predictor's."""
    vis = np.asarray(ev["visit"]); cor = np.asarray(ev["correct"]).astype(np.int64); bl = np.asarray(ev["blind"]).astype(np.int64)
    return [dict(visit=v, family=g, score=share(cor, vis == v)["share"], blind=share(bl, vis == v)["share"]) for v, g in visits]


def lw_kept(la):
    """What is kept, from the list: for each family (and the birth family, -1) its share right on its list items at the end of its own
    first visit (the birth family: before visit 0), and at the end of every later visit: the share, the share divided by its own, and
    decay alone beside it (source 1: the weights at the end of its own visit decayed by the updates since)."""
    at, src, fam, cor = (np.asarray(la[k]) for k in ("at_visit", "source", "family", "correct"))
    du = np.asarray(la["decay_updates"])
    cor = cor.astype(np.int64)
    out = {}
    for f in sorted(set(int(x) for x in fam)):
        m0 = (fam == f) & (src == 0)
        own_v = BEFORE_LIFE if f == BIRTH_FAMILY else int(min(at[m0]))
        own = share(cor, m0 & (at == own_v))["share"]
        rows = []
        for k in sorted(set(int(x) for x in at[m0 & (at != own_v)])):
            if f != BIRTH_FAMILY and k <= own_v:
                continue
            s0 = share(cor, m0 & (at == k))["share"]
            md = (fam == f) & (src == 1) & (at == k)
            sd = share(cor, md)["share"] if md.any() else None
            rows.append(dict(at_visit=k, visits_since=(k + 1) if f == BIRTH_FAMILY else k - own_v, share=s0,
                             kept=(s0 / own) if own else None, decay_alone_share=sd, decay_alone_kept=(sd / own) if (own and sd is not None) else None,
                             decay_updates=int(du[md][0]) if md.any() else None))
        out[f] = dict(own_visit=own_v, own_share=own, later=rows)
    return out


def lw_second_lap(ev, la, world, visit_updates, first, visits):
    """The second lap (a branch over families taken before): per return, the family's share right on its reduction items (add and sub
    pooled) in the return's first update and its first 16 updates; the family's figure over the last quarter of the return beside that of
    its first visit; and its list share at the end of the return beside that at the end of its own first visit (from the trunk's list)."""
    q = visit_updates // 4
    own = {r["family"]: r for r in first}
    at, src, fam, cor = (np.asarray(la[k]) for k in ("at_visit", "source", "family", "correct"))
    cor = cor.astype(np.int64)
    out = []
    for v, g in visits:
        u0, u1 = visit_range(ev, v)
        m = (fam == g) & (src == 0) & (at == v)
        out.append(dict(visit=v, family=g, p=world.groups[g][0]["p"], first_update=lw_levels(ev, v, g, u0, u0 + 1)["reduction_pooled"],
                        first16=lw_levels(ev, v, g, u0, u0 + LW_FIRST)["reduction_pooled"], last_quarter=lw_levels(ev, v, g, u1 - q, u1),
                        first_visit_last_quarter_figure=own[g]["last_quarter"]["figure"] if g in own else None,
                        list_share_end_of_return=share(cor, m)["share"]))
    return out


def load(path):
    f = np.load(path)
    return {k: f[k] for k in f.files}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("events"); ap.add_argument("--world", type=int, required=True); ap.add_argument("--visits", type=int); ap.add_argument("--to")
    ap.add_argument("--birth-groups")
    a = ap.parse_args(argv)
    ev = load(a.events)
    life_vis = ev["visit"][ev["visit"] != BIRTH_VISIT]
    V = a.visits if a.visits is not None else int(life_vis.max()) + 1
    bt = None
    if a.birth_groups:
        import sw_world as W
        w = W.World(a.world)
        bt = sorted(t["id"] for g in a.birth_groups.split(",") for t in w.groups[int(g)])
    r = score(ev, V, symmetric_tasks(a.world), bt)
    if a.to:
        assert not Path(a.to).exists(), f"refusing an existing path: {a.to}"
        Path(a.to).write_text(json.dumps(r, indent=1))
    print(json.dumps({k: r[k] for k in ("visits", "entries", "mean_score", "late_level", "score_trend", "learning_trend", "cost_trend")}))
    return r


if __name__ == "__main__":
    main(sys.argv[1:])
