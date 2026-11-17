"""E44, job L2 (roster note OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md, revision 2): the scorer's additions for the recorded
quantities D1 to D14 (the reviewer's audit, section 5). A new file beside sw_score.py, which is not edited; D6 is sw_score.lw_first_visits.

    python3 l2_score.py life <events .npz> <list .npz> <world> <visits read> [--to OUT.json]
    python3 l2_score.py probe <branch events .npz> <world> [--to OUT.json]

Everything is read from the event file, except D9 and the list's part of D14, which read the list file, and the operands of the items of
D14's stream part, which the event file does not hold: they are the asked items of the first update of the visit, regenerated from the
world (lw_world.World.visit_batches) and checked against the event file's task and reduction mark, item by item.

  D1   a reduction item: an asked item of add or sub (task % 4 in 0, 1) whose `wrap` is 1
  D2   score by age on reduction items: per visit, the share right over its reduction items, add and sub pooled; the blind predictor's
  D3   kinds of visit: first; continuation (absence 1); return (absence 2 or more)
  D4   late level: the mean of D2 over the last max(1, V // 4) visits; also by kind of visit
  D5   score trend: the least-squares slope of D2 on ln(v + 1) over the returns among the visits V // 2 .. V - 1; over all visits of that
       half beside it (the memo's definition; not used)
  D6   first-visit level of a family: the mean of add's and sub's share right on reduction items over the last 64 updates of its first visit
  D7   trend of learning: the slope of D6 on ln n over all families met, n = 1 for the first met; beside it the coefficient of ln n in the
       joint least-squares fit with the logarithm of the number of taught pairs (recommendation 5)
  D8   kept at return, from the stream: the reduction items of the first update of a continuation or a return; right and count pooled over
       the visits of an absence class (1 visit; 2 to 4; 5 to 32; 33 or more; and, for recommendation 4, 5 or more)
  D9   kept on the list: for a family f answered by the network at the end of visit v, with l < v its most recent visit: its share right
       at v divided by its share at the end of l; by v - l in D8's classes, mean and median; a share of 0 at l is left out and counted
  D10  relearning at return, by absence class: the mean over the visits of the family's figure (as D6) over the first 16 updates and the
       last 64 updates of the visit, beside the last 64 updates of the family's previous visit
  D14  what the wrong answers are: for D8's items and for the list's rows of D9, by absence class, the share of answers in each class,
       tried in this order: right; the sum or difference without the reduction; the right answer under the modulus of the family taught
       last; under the modulus of an earlier family of the life; under 89; one of the operands; (a modulus the life never meets, a control
       tried last;) another two-digit number; not two digits and the end symbol. The chance line: the never-met modulus tried in the place
       of the family taught last. The family taught last: for a stream item of visit v, the family of visit v - 1; for a list row at
       the end of visit v, the family of visit v. The never-met modulus: family 55's, a probe family that no life is scheduled to visit.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lw_world as LW                 # noqa: E402
import sw_score as S                  # noqa: E402

EOS = LW.EOS
CLASSES = (("1 visit", 1, 1), ("2 to 4", 2, 4), ("5 to 32", 5, 32), ("33 or more", 33, 10 ** 9), ("5 or more", 5, 10 ** 9))
ANSWER_CLASSES = ("right", "the sum or difference without the reduction", "the answer under the modulus of the family taught last",
                  "the answer under the modulus of an earlier family", "the answer under 89, the birth family's modulus", "one of the operands",
                  "the answer under a modulus the life never meets (control)", "another two-digit number", "not two digits and the end symbol")
CONTROL_FAMILY = 55
T95 = {1: 6.3138, 2: 2.9200, 3: 2.3534, 4: 2.1318, 5: 2.0150, 6: 1.9432, 7: 1.8946, 8: 1.8595, 9: 1.8331, 10: 1.8125, 11: 1.7959, 12: 1.7823,
       13: 1.7709, 14: 1.7613, 15: 1.7531}      # the 0.95 quantile of Student's t: a two-sided 90% interval


def load(p):
    f = np.load(p)
    return {k: f[k] for k in f.files}


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.all(x == x[0]):
        return None
    xm = x - x.mean()
    return float((xm * (y - y.mean())).sum() / (xm * xm).sum())


def interval90(xs):
    """Mean and two-sided 90% interval (Student's t over the values); None for fewer than two values."""
    xs = [float(x) for x in xs if x is not None]
    if len(xs) < 2:
        return dict(n=len(xs), mean=xs[0] if xs else None, low=None, high=None, values=xs)
    m, sd = float(np.mean(xs)), float(np.std(xs, ddof=1))
    h = T95[len(xs) - 1] * sd / math.sqrt(len(xs))
    return dict(n=len(xs), mean=m, sd=sd, low=m - h, high=m + h, values=xs)


def verdict(iv, up="higher", down="lower"):
    if iv["low"] is None:
        return "not read"
    return up if iv["low"] > 0 else down if iv["high"] < 0 else "not distinguished"


def answer_class(op, p, a, b, g, p_last, p_others, p_control, control_first=False):
    raw = a + b if op == "add" else a - b
    if not (48 <= g[0] <= 57 and 48 <= g[1] <= 57 and g[2] == EOS):
        return 8
    x = (g[0] - 48) * 10 + (g[1] - 48)
    if x == raw % p:
        return 0
    if x == abs(raw) or x == raw % 100:
        return 1
    if control_first:
        return 6 if x == raw % p_control else 7
    if p_last is not None and x == raw % p_last:
        return 2
    if any(x == raw % q for q in p_others):
        return 3
    if x == raw % 89:
        return 4
    if x in (a, b):
        return 5
    if x == raw % p_control:
        return 6
    return 7


def class_table(counts, ctrl):
    n = int(sum(counts))
    return dict(answers=n, shares={ANSWER_CLASSES[i]: (counts[i] / n if n else None) for i in range(len(ANSWER_CLASSES))},
                chance_never_met_modulus_in_place_of_the_last=(ctrl / n if n else None))


def quantities(ev, la, world_seed, V, visit_updates=256):
    w = LW.World(world_seed, visit_updates)
    sched = w.schedule(V)
    mod = [g[0]["p"] for g in w.groups]
    keep = ev["visit"] < V
    vis, upd, task = ev["visit"][keep].astype(np.int64), ev["update"][keep].astype(np.int64), ev["task"][keep].astype(np.int64)
    cor, blind = ev["correct"][keep].astype(np.int64), ev["blind"][keep].astype(np.int64)
    gen = np.stack([ev[k][keep].astype(np.int64) for k in ("g1", "g2", "g3")], 1)
    op = task % 4
    red = (ev["wrap"][keep] == 1) & (op < 2)                                                      # D1
    per = 16 * visit_updates
    assert len(vis) == V * per and (vis == np.repeat(np.arange(V), per)).all(), "the event file is not V visits of 16 asked items an update"
    assert (upd == np.repeat(np.arange(V * visit_updates), 16)).all()
    fam = task // 4
    for s in sched:
        assert (fam[s["visit"] * per:(s["visit"] + 1) * per] == s["group"]).all(), "the event file's families are not the schedule's"

    def sl(v, lo=0, hi=visit_updates):
        return slice(v * per + 16 * lo, v * per + 16 * hi)

    def pooled(v, lo=0, hi=visit_updates, col=cor):
        m = red[sl(v, lo, hi)]
        return float(col[sl(v, lo, hi)][m].mean()) if m.any() else None

    def figure(v, lo, hi):                                                                         # as D6: the mean of add's and sub's share
        m, c, o = red[sl(v, lo, hi)], cor[sl(v, lo, hi)], op[sl(v, lo, hi)]
        sh = [float(c[m & (o == j)].mean()) if (m & (o == j)).any() else None for j in (0, 1)]
        return None if None in sh else (sh[0] + sh[1]) / 2

    kind = ["first" if s["new"] else "continuation" if s["absence"] == 1 else "return" for s in sched]                         # D3
    sba = [pooled(v) for v in range(V)]; bba = [pooled(v, col=blind) for v in range(V)]                                        # D2
    q0 = V - max(1, V // 4)
    late = lambda xs, k=None: (lambda ys: float(np.mean(ys)) if ys else None)([xs[v] for v in range(q0, V) if k is None or kind[v] == k])
    half = range(V // 2, V)
    ret = [v for v in half if kind[v] == "return"]
    # D6, D7
    fv = S.lw_first_visits({k: ev[k][keep] for k in ("visit", "update", "task", "correct", "wrap")}, w, visit_updates, [(s["visit"], s["group"]) for s in sched])
    firsts = [dict(n=r["n"], family=r["family"], visit=r["visit"], p=r["p"], taught_pairs=r["taught_pairs"], level=r["last_quarter"]["figure"],
                   whole=r["whole"]["figure"], first16=r["first16"]["figure"]) for r in fv]
    for r in firsts:
        assert abs(r["level"] - figure(r["visit"], visit_updates - visit_updates // 4, visit_updates)) < 1e-12
    ln_n = [math.log(r["n"]) for r in firsts]; lev = [r["level"] for r in firsts]
    joint = None
    if len(firsts) >= 4:
        X = np.stack([np.ones(len(firsts)), ln_n, [math.log(r["taught_pairs"]) for r in firsts]], 1)
        co = np.linalg.lstsq(X, np.asarray(lev), rcond=None)[0]
        joint = dict(ln_n=float(co[1]), ln_taught_pairs=float(co[2]), intercept=float(co[0]))
    out = dict(world=world_seed, visits=V, visit_updates=visit_updates, entries=int(len(vis)), reduction_items=int(red.sum()),
               families_met=len(firsts), kinds={k: kind.count(k) for k in ("first", "continuation", "return")},
               score_by_age=sba, blind_by_age=bba, kind=kind,
               late_level=late(sba), blind_late_level=late(bba), late_level_by_kind={k: late(sba, k) for k in ("first", "continuation", "return")},
               late_visits_by_kind={k: sum(1 for v in range(q0, V) if kind[v] == k) for k in ("first", "continuation", "return")},
               score_trend=slope([math.log(v + 1) for v in ret], [sba[v] for v in ret]), score_trend_returns=len(ret),
               score_trend_all_visits=slope([math.log(v + 1) for v in half], [sba[v] for v in half]),
               first_visits=firsts, median_first_visit_level=float(np.median(lev)),
               median_first_visit_level_families_2_on=float(np.median(lev[1:])) if len(lev) > 1 else None,
               trend_of_learning=slope(ln_n, lev), trend_of_learning_joint_fit=joint)
    # D8, D10 and the stream part of D14
    later, last = [], {}
    control = mod[CONTROL_FAMILY]
    for s in sched:
        v, g = s["visit"], s["group"]
        if not s["new"]:
            m1 = red[sl(v, 0, 1)]
            taught, asked = next(iter(w.visit_batches(v, g)))
            assert [it[0] for it in asked] == task[sl(v, 0, 1)].tolist(), "the regenerated first update differs from the event file"
            assert [LW.needs_wrap(w.tasks[it[0]]["op"], w.tasks[it[0]]["p"], it[1], it[2]) for it in asked] == (ev["wrap"][keep][sl(v, 0, 1)] == 1).tolist()
            met = [x["group"] for x in sched[:v]]
            others = sorted({mod[h] for h in met if h not in (g, met[-1])})
            cc, ctrl = [0] * len(ANSWER_CLASSES), 0
            for i, it in enumerate(asked):
                if m1[i]:
                    t = w.tasks[it[0]]; gg = tuple(int(x) for x in gen[sl(v, 0, 1)][i])
                    c = answer_class(t["op"], t["p"], it[1], it[2], gg, mod[met[-1]] if met[-1] != g else None, others, control)
                    assert (c == 0) == bool(cor[sl(v, 0, 1)][i]), "an answer's class disagrees with the event file's `correct`"
                    cc[c] += 1
                    ctrl += answer_class(t["op"], t["p"], it[1], it[2], gg, None, others, control, control_first=True) == 6
            later.append(dict(visit=v, family=g, absence=s["absence"], right=int(cor[sl(v, 0, 1)][m1].sum()), n=int(m1.sum()),
                              first16=figure(v, 0, 16), last_quarter=figure(v, visit_updates - visit_updates // 4, visit_updates),
                              previous_last_quarter=figure(last[g], visit_updates - visit_updates // 4, visit_updates), classes=cc, control=ctrl))
        last[g] = v
    mean_ = lambda xs: (lambda ys: float(np.mean(ys)) if ys else None)([x for x in xs if x is not None])
    kept, wrong = {}, {}
    for name, lo, hi in CLASSES:
        rs = [r for r in later if lo <= r["absence"] <= hi]
        n, k = sum(r["n"] for r in rs), sum(r["right"] for r in rs)
        kept[name] = dict(visits=len(rs), items=n, right=k, share=(k / n) if n else None, first16=mean_([r["first16"] for r in rs]),
                          last_quarter=mean_([r["last_quarter"] for r in rs]), previous_last_quarter=mean_([r["previous_last_quarter"] for r in rs]))
        wrong[name] = class_table([sum(r["classes"][i] for r in rs) for i in range(len(ANSWER_CLASSES))], sum(r["control"] for r in rs))
    out.update(kept_at_return=kept, wrong_answers_stream=wrong,
               absence_classes_restated={name: dict(visits=[lo, None if hi > 10 ** 6 else hi], updates=[lo * visit_updates, None if hi > 10 ** 6 else hi * visit_updates],
                                                    decay_constants=[lo * visit_updates / 4000, None if hi > 10 ** 6 else hi * visit_updates / 4000])
                                         for name, lo, hi in CLASSES},
               later_visits=[{k: r[k] for k in ("visit", "family", "absence", "right", "n", "first16", "last_quarter", "previous_last_quarter")} for r in later])
    if la is not None:
        out.update(list_quantities(la, w, sched, V, mod, control))
    return out


def list_quantities(la, w, sched, V, mod, control):
    """D9 and the list's part of D14, from the list file's rows answered by the network (source 0) at visits below V."""
    at, src, fam = la["at_visit"].astype(np.int64), la["source"], la["family"].astype(np.int64)
    sel = (src == 0) & (fam >= 0) & (at < V)
    at, fam = at[sel], fam[sel]
    cor = la["correct"][sel].astype(np.int64); task = la["task"][sel].astype(np.int64); a, b = la["a"][sel].astype(np.int64), la["b"][sel].astype(np.int64)
    gen = np.stack([la[k][sel].astype(np.int64) for k in ("g1", "g2", "g3")], 1)
    key = np.stack([at, fam], 1)
    cut = np.nonzero(np.any(key[1:] != key[:-1], axis=1))[0] + 1
    rows = {}
    for s_, e_ in zip(np.concatenate([[0], cut]), np.concatenate([cut, [len(key)]])):
        rows[int(key[s_, 0]), int(key[s_, 1])] = (int(s_), int(e_))
    share = {k: float(cor[s_:e_].mean()) for k, (s_, e_) in rows.items()}
    acc = {name: dict(points=0, left_out=0, ratios=[], shares=[], classes=[0] * len(ANSWER_CLASSES), control=0) for name, _, _ in CLASSES}
    last, met = {}, []
    for s in sched:
        v, g = s["visit"], s["group"]
        last[g] = v
        if g not in met:
            met.append(g)
        for f, l in last.items():
            if l == v or (v, f) not in rows:
                continue
            assert (l, f) in rows, "a family was not answered at the end of its own visit"
            others = sorted({mod[h] for h in met if h not in (f, g)})
            s_, e_ = rows[v, f]
            cc, ctrl = [0] * len(ANSWER_CLASSES), 0
            for i in range(s_, e_):
                o = "add" if task[i] % 4 == 0 else "sub"; gg = (int(gen[i, 0]), int(gen[i, 1]), int(gen[i, 2]))
                c = answer_class(o, mod[f], int(a[i]), int(b[i]), gg, mod[g], others, control)
                assert (c == 0) == bool(cor[i])
                cc[c] += 1
                ctrl += answer_class(o, mod[f], int(a[i]), int(b[i]), gg, None, others, control, control_first=True) == 6
            for name, lo, hi in CLASSES:
                if lo <= v - l <= hi:
                    o_ = acc[name]
                    o_["shares"].append(share[v, f])
                    for i in range(len(ANSWER_CLASSES)):
                        o_["classes"][i] += cc[i]
                    o_["control"] += ctrl
                    if share[l, f] == 0:
                        o_["left_out"] += 1
                    else:
                        o_["points"] += 1; o_["ratios"].append(share[v, f] / share[l, f])
    kept = {k: dict(points=o["points"], left_out=o["left_out"], mean=float(np.mean(o["ratios"])) if o["ratios"] else None,
                    median=float(np.median(o["ratios"])) if o["ratios"] else None,
                    mean_share_right=float(np.mean(o["shares"])) if o["shares"] else None) for k, o in acc.items()}
    return dict(kept_on_list=kept, wrong_answers_list={k: class_table(o["classes"], o["control"]) for k, o in acc.items()},
                list_rows_read=len(rows))


def probe(ev, world_seed, visit_updates=256):
    """The learning probe: a branch of four visits over the probe families; each family's first-visit level (D6) from the branch's events."""
    w = LW.World(world_seed, visit_updates)
    vs = sorted(set(int(x) for x in ev["visit"]))
    out = []
    for v in vs:
        g = int(ev["task"][ev["visit"] == v][0]) // 4
        u0, u1 = S.visit_range(ev, v)
        q = visit_updates // 4
        out.append(dict(visit=v, family=g, p=w.groups[g][0]["p"], taught_pairs=len(w.groups[g][0]["taught"]),
                        level=S.lw_levels(ev, v, g, u1 - q, u1)["figure"], whole=S.lw_levels(ev, v, g, u0, u1)["figure"],
                        first16=S.lw_levels(ev, v, g, u0, u0 + 16)["figure"]))
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("life", "probe")); ap.add_argument("pos", nargs="+"); ap.add_argument("--to")
    a = ap.parse_args(argv)
    if a.cmd == "life":
        r = quantities(load(a.pos[0]), load(a.pos[1]) if a.pos[1] != "-" else None, int(a.pos[2]), int(a.pos[3]))
    else:
        r = probe(load(a.pos[0]), int(a.pos[1]))
    if a.to:
        assert not Path(a.to).exists(), f"refusing an existing path: {a.to}"
        Path(a.to).write_text(json.dumps(r, indent=1))
    else:
        print(json.dumps(r, indent=1)[:3000])
    return r


if __name__ == "__main__":
    main(sys.argv[1:])
