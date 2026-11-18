"""E60 (brief TASK_E60_THE_SWEEPS_LONG_LIFE_FROM_BACK_PROPAGATIONS_BIRTH_20261118.md, revision 3, on claude/vibrant-cerf-sje726 at 9bb87033,
section 7): the scores of the four lives of `sweep-after-naive` and the fixed reading. It imports l2_score and uses its `quantities`,
`interval90` and `verdict` only. Nothing here trains.

    python3 l2_san_report.py scores     # each finished life: l2_score.quantities at V = 256 and 1,024
                                        #   -> exp45/e44/L2/sweep_after_naive/scores/<stem>_V<V>.json (a file that exists is not written again)
    python3 l2_san_report.py final [name] # -> exp45/e44/L2/sweep_after_naive/final/ : E60.json, E60_TABLES.md, curves_w<w>.csv (refused if it exists)

The figures of `naive` and `sweep` are those of the score files of the helper's branch at 738e2d61 (sweep/main/lw_naive_w<w>_v1024_V<V>.json
and sweep/scores/lw_sweep_w<w>_v1024_V<V>.json), read with `git show 738e2d61:<path>`. The receipts of `naive` (learner, stream digests) are
read from origin/claude/e44-l2-first-long-lives.

Pairs (each a difference of per-world values, unrounded): R = `sweep-after-naive` - `naive`; B = `sweep` - `sweep-after-naive`;
RB = `sweep` - `naive`. Two readings: over worlds 1 to 4 (interval90 on exactly four values) and over worlds 1 to 3 (on exactly three).
Words: lower, higher, not distinguished, not read.
"""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MDIR = HERE.parent
ROOT = MDIR.parent
sys.path.insert(0, str(HERE))
import l2_score as L                  # noqa: E402

SAN = MDIR / "exp45" / "e44" / "L2" / "sweep_after_naive"
LIVES, SCORES = SAN / "lives", SAN / "scores"
PREV = "738e2d61"
W1 = "origin/claude/e44-l2-first-long-lives"
SW_REL = "MOD97_LOCAL_RULE_20261106/exp45/e44/L2/sweep/"
SEEDS = {1: 73129, 2: 93683, 3: 116089, 4: 73129}
WORLDS = (1, 2, 3, 4)
READINGS = (("over worlds 1 to 4", (1, 2, 3, 4)), ("over worlds 1 to 3", (1, 2, 3)))
AGES = (("early", "over the first 64 visits"), ("256", "at 256 visits"), ("1024", "at 1,024 visits"))
SHARED = "Worlds 1 and 4 start from the same births (seed 73129), so that the four values are not those of four independent learners."


def git(*a, text=True):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=text, check=True).stdout


def stem(w):
    return f"lw_sweep-after-naive_w{w}_v1024"


def scores():
    SCORES.mkdir(parents=True, exist_ok=True)
    for w in WORLDS:
        s = stem(w)
        if not (LIVES / f"{s}.json").exists():
            continue
        for V in (256, 1024):
            p = SCORES / f"{s}_V{V}.json"
            if p.exists():
                continue
            r = L.quantities(L.load(LIVES / f"{s}.npz"), L.load(LIVES / f"{s}_list.npz"), w, V)
            r["life"] = s
            p.write_text(json.dumps(r, indent=1))
            print("scored", s, V, flush=True)


_cache = {}


def q(arm, w, V):
    key = (arm, w, V)
    if key not in _cache:
        if arm == "sweep-after-naive":
            p = SCORES / f"{stem(w)}_V{V}.json"
            _cache[key] = json.loads(p.read_text()) if p.exists() else None
        else:
            rel = SW_REL + (f"main/lw_naive_w{w}_v1024_V{V}.json" if arm == "naive" else f"scores/lw_sweep_w{w}_v1024_V{V}.json")
            _cache[key] = json.loads(git("show", f"{PREV}:{rel}"))
    return _cache[key]


def level(arm, w, age):
    if age == "early":
        x = q(arm, w, 1024)
        return None if x is None else float(np.mean(x["score_by_age"][:64]))
    x = q(arm, w, int(age))
    return None if x is None else x["late_level"]


def pair(name, w, age):
    a, b = {"R": ("sweep-after-naive", "naive"), "B": ("sweep", "sweep-after-naive"), "RB": ("sweep", "naive")}[name]
    x, y = level(a, w, age), level(b, w, age)
    return None if x is None or y is None else x - y


def reading(vals, ws):
    """interval90 and verdict on exactly the values of the worlds ws; `not read` if one is missing."""
    v = [vals[w] for w in ws]
    if any(x is None for x in v):
        return dict(mean=None, low=None, high=None, word="not read", n=sum(x is not None for x in v))
    iv = L.interval90(v)
    return dict(mean=iv["mean"], low=iv["low"], high=iv["high"], sd=iv["sd"], word=L.verdict(iv), n=len(v))


def f(x, d=3):
    return "—" if x is None else f"{x:+.{d}f}"


def iv_txt(r, d=3):
    return "not read" if r["mean"] is None else f"{f(r['mean'], d)} [{f(r['low'], d)}, {f(r['high'], d)}], {r['word']}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def d13(ws):
    t5 = [q("sweep-after-naive", w, 1024)["score_trend"] if q("sweep-after-naive", w, 1024) else None for w in WORLDS]
    t7 = [q("sweep-after-naive", w, 1024)["trend_of_learning"] if q("sweep-after-naive", w, 1024) else None for w in WORLDS]
    r5, r7 = reading(dict(zip(WORLDS, t5)), ws), reading(dict(zip(WORLDS, t7)), ws)
    if "not read" in (r5["word"], r7["word"]):
        st = "not read"
    else:
        st = "still learning" if (r5["low"] > 0 or r7["low"] > 0) else "stuck"
    dec = [n for n, r in (("score trend", r5), ("trend of learning", r7)) if r["high"] is not None and r["high"] < 0]
    return dict(score_trend=r5, trend_of_learning=r7, statement=st, declining=dec)


def final(name="final"):
    out = SAN / name
    assert not out.exists(), f"refusing an existing path: {out}"
    have = [w for w in WORLDS if q("sweep-after-naive", w, 1024) is not None]
    J = dict(brief="TASK_E60_..._20261118.md revision 3 (9bb87033)", worlds_scored=have, naive_and_sweep_from=f"{PREV} (sweep/main, sweep/scores)",
             scorer_sha256=hashlib.sha256((HERE / "l2_score.py").read_bytes()).hexdigest(), shared_births=SHARED)
    md = ["# E60: the long life under the sweep from back-propagation's birth: tables", "",
          f"Written by `smallworld/l2_san_report.py final`. Lives of `sweep-after-naive` scored: worlds {', '.join(map(str, have))}. The figures of `naive` "
          f"and `sweep` are the score files of `{PREV}`. Each difference is computed from unrounded per-world values; each figure is rounded once. "
          "Intervals: two-sided 90%, Student's t over the worlds of the reading (`l2_score.interval90`), read with `l2_score.verdict`.", "",
          f"**Limit beside every reading over worlds 1 to 4:** {SHARED}", ""]

    # levels
    rows = []
    lev = {}
    for arm in ("naive", "sweep-after-naive", "sweep"):
        for a, label in AGES:
            vals = {w: level(arm, w, a) for w in WORLDS}
            lev[f"{arm}|{a}"] = vals
            rows.append([f"`{arm}`", label] + [("—" if vals[w] is None else f"{vals[w]:.3f}") for w in WORLDS])
    md += ["## Levels", "", "Early level: the mean of D2 over visits 0 to 63 (from the score file at V = 1,024). Late level: D4 at V = 256 and 1,024.", "",
           table(["Arm", "Age", "World 1", "World 2", "World 3", "World 4"], rows), ""]
    J["levels"] = lev

    # the nine cells
    cells = {}
    rows = []
    for name, label in (("R", "R: `sweep-after-naive` − `naive` (same birth; rules differ)"), ("B", "B: `sweep` − `sweep-after-naive` (same rule; births differ)"),
                        ("RB", "RB: `sweep` − `naive` (both differ)")):
        for a, alabel in AGES:
            vals = {w: pair(name, w, a) for w in WORLDS}
            rd = {rn: reading(vals, ws) for rn, ws in READINGS}
            cells[f"{name}|{a}"] = dict(per_world=vals, **rd)
            rows.append([label, alabel] + [f(vals[w], 4) for w in WORLDS] + [iv_txt(rd[rn]) for rn, _ in READINGS])
    md += ["## The nine cells (section 7 of the brief)", "", f"Each cell: the four per-world differences, then each reading's mean, 90% interval and word. {SHARED}", "",
           table(["Pair", "Age", "World 1", "World 2", "World 3", "World 4", "Over worlds 1 to 4", "Over worlds 1 to 3"], rows), ""]
    add = []
    for a, _ in AGES:
        for w in WORLDS:
            r_, b_, rb_ = cells[f"R|{a}"]["per_world"][w], cells[f"B|{a}"]["per_world"][w], cells[f"RB|{a}"]["per_world"][w]
            if None not in (r_, b_, rb_):
                add.append(abs(rb_ - (r_ + b_)))
    J["RB_equals_R_plus_B_largest_deviation"] = max(add) if add else None
    md += [f"RB = R + B in every world and age: the largest deviation is {max(add) if add else '—'} (rounding of floating point).", ""]
    J["cells"] = cells

    # pair R: D5, D13, the change
    extra = {}
    t5 = {w: (q("sweep-after-naive", w, 1024)["score_trend"] - q("naive", w, 1024)["score_trend"]) if w in have else None for w in WORLDS}
    ch = {w: (pair("R", w, "1024") - pair("R", w, "early")) if w in have else None for w in WORLDS}
    rows = [["difference of D5 (score trend) at 1,024 visits"] + [f(t5[w], 4) for w in WORLDS] + [iv_txt(reading(t5, ws), 4) for _, ws in READINGS],
            ["change: R at 1,024 visits less R over the first 64"] + [f(ch[w], 4) for w in WORLDS] + [iv_txt(reading(ch, ws), 4) for _, ws in READINGS]]
    extra["score_trend_difference"] = dict(per_world=t5, **{rn: reading(t5, ws) for rn, ws in READINGS})
    extra["change"] = dict(per_world=ch, **{rn: reading(ch, ws) for rn, ws in READINGS})
    md += ["## Pair R: the score trend, the change, and D13", "", table(["Quantity", "World 1", "World 2", "World 3", "World 4", "Over worlds 1 to 4", "Over worlds 1 to 3"], rows), ""]
    rows = []
    for rn, ws in READINGS:
        d = d13(ws); extra[f"D13 {rn}"] = d
        rows.append([rn, iv_txt(d["score_trend"], 4), iv_txt(d["trend_of_learning"], 4), d["statement"], ", ".join(d["declining"]) or "none"])
    md += ["D13 for `sweep-after-naive` at 1,024 visits: *still learning* if the interval of D5 or of D7 lies wholly above zero, *stuck* otherwise; "
           "*declining* for an interval wholly below zero.", "", table(["Reading", "Score trend (D5)", "Trend of learning (D7)", "Statement", "Declining"], rows), ""]
    J["pair_R"] = extra

    # the three sentences
    sent = {}
    md += ["## The three fixed sentences", ""]
    for rn, ws in READINGS:
        k = len(ws); wd = {4: "four", 3: "three"}[k]
        c = {a: cells[f"R|{a}"][rn] for a, _ in AGES}
        w_ = lambda r: {"lower": "lower than", "higher": "higher than", "not distinguished": "not distinguished from", "not read": "not read against"}[r["word"]]
        s1 = (f"From the same birth, the life under the sweep is {w_(c['early'])} the life under back-propagation over the first 64 visits "
              f"({iv_txt(c['early'])}), {w_(c['256'])} it at 256 visits ({iv_txt(c['256'])}) and {w_(c['1024'])} it at 1,024 visits ({iv_txt(c['1024'])}).")
        kinds = {}
        for w in ws:
            e, l_ = pair("R", w, "early"), pair("R", w, "1024")
            if e is None or l_ is None:
                kinds[w] = "not read"
            elif (e > 0) != (l_ > 0) and e != 0 and l_ != 0:
                kinds[w] = "changed sign"
            elif abs(l_) < abs(e):
                kinds[w] = "nearer to zero"
            else:
                kinds[w] = "not nearer to zero"
        cr = reading(ch, ws)
        kk, jj, mm = (sum(v == x for v in kinds.values()) for x in ("nearer to zero", "changed sign", "not nearer to zero"))
        s2 = (f"In {kk} of the {wd} worlds the difference `sweep-after-naive` − `naive` is nearer to zero at 1,024 visits than over the first 64 visits, "
              f"in {jj} it changed sign, and in {mm} it is not nearer to zero; over these worlds the change (the difference at 1,024 visits less the difference "
              f"over the first 64 visits) is {cr['word']} ({iv_txt(cr, 4)}).")
        smaller = [w for w in ws if None not in (pair("R", w, "early"), pair("R", w, "1024")) and pair("R", w, "early") < pair("R", w, "1024") < 0]
        overtaking = c["early"]["word"] == "lower" and c["1024"]["word"] == "higher"
        s2b = ("Overtaking is said in this reading." if overtaking else "Overtaking is not said in this reading (it needs pair R *lower* over the first 64 visits and *higher* at 1,024 visits).")
        if smaller:
            s2b += " " + " ".join(f"The deficit of `sweep-after-naive` became smaller in world {w}." for w in smaller)
        rb, r_, b_ = cells["RB|1024"][rn], cells["R|1024"][rn], cells["B|1024"][rn]
        s3 = (f"Of the difference between the arms `sweep` and `naive` at 1,024 visits (RB, {iv_txt(rb)}), the part with the birth held equal is R ({iv_txt(r_)}) "
              f"and the part with the rule held equal is B ({iv_txt(b_)}).")
        dist = lambda r: r["word"] in ("lower", "higher")
        if dist(rb) and r_["word"] == rb["word"] and not dist(b_):
            assign = "RB is assigned to the rule."
        elif dist(rb) and b_["word"] == rb["word"] and not dist(r_):
            assign = "RB is assigned to the birth."
        else:
            fails = []
            if not dist(rb):
                fails.append("RB is not distinguished")
            else:
                if r_["word"] != rb["word"]:
                    fails.append(f"R is {r_['word']}, not {rb['word']} as RB (needed for the rule)")
                if dist(b_):
                    fails.append(f"B is {b_['word']}, not undistinguished (needed for the rule)")
                if b_["word"] != rb["word"]:
                    fails.append(f"B is {b_['word']}, not {rb['word']} as RB (needed for the birth)")
                if dist(r_):
                    fails.append(f"R is {r_['word']}, not undistinguished (needed for the birth)")
            assign = "Nothing is assigned: " + "; ".join(fails) + "."
        sent[rn] = dict(sentence_1=s1, sentence_2=s2, world_kinds=kinds, overtaking=overtaking, smaller_deficit=smaller, sentence_2_note=s2b,
                        sentence_3=s3, assignment=assign)
        md += [f"**{rn[0].upper() + rn[1:]}**" + (f" ({SHARED})" if k == 4 else "") + ":", "",
               f"1. {s1}", f"2. {s2} By world: " + "; ".join(f"world {w}: {kinds[w]}" for w in ws) + f". {s2b}", f"3. {s3} {assign}", ""]
    J["sentences"] = sent

    # births, stream digests, learners
    rows, births = [], {}
    for seed in sorted(set(SEEDS.values())):
        for rule in ("B", "S"):
            r = json.loads((MDIR / "exp45" / "e41" / f"e41_{rule}_h256_f50_{seed}.json").read_text())
            last = r["curve"][-1]
            births[f"{rule}_{seed}"] = dict(A=last["A"], R=last["R"])
            rows.append([seed, rule, f"{last['A']:.2f}", f"{last['R']:.2f}"])
    md += ["## Scores at birth", "", "The receipts' final rows (E41): A and R in per cent.", "", table(["Seed", "Rule of the birth", "A", "R"], rows), ""]
    J["births"] = births
    rows, dig = [], {}
    for w in have:
        mine = json.loads((LIVES / f"{stem(w)}.json").read_text())
        nv = json.loads(git("show", f"{W1}:MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w{w}_v1024.json"))
        eq = sum(a["stream_sha256"] == b["stream_sha256"] for a, b in zip(mine["by_visit"], nv["by_visit"]))
        dig[w] = dict(visits=len(mine["by_visit"]), equal=eq, learner_equal=mine["learner"] == nv["learner"],
                      learner_keys_that_differ=sorted(k for k in set(mine["learner"]) | set(nv["learner"]) if mine["learner"].get(k) != nv["learner"].get(k)),
                      stream_sha256=mine["stream_sha256"], stream_sha256_naive=nv["stream_sha256"], resumes=mine["resumes"], cpu_hours=mine["cpu"]["total"] / 3600,
                      events_file_sha256=mine["events"]["file_sha256"], list_file_sha256=mine["list"]["file_sha256"])
        rows.append([w, f"{eq} of {len(mine['by_visit'])}", mine["stream_sha256"] == nv["stream_sha256"], mine["learner"] == nv["learner"],
                     len(mine["resumes"]), f"{mine['cpu']['total'] / 3600:.2f}"])
    md += ["## The new lives against `naive`", "", "Per-visit stream digests equal to those of `naive`'s receipt of the same world; the whole-life stream digest; "
           "the receipt's `learner` object equal key for key; resumptions; CPU-hours.", "",
           table(["World", "Visits with equal stream digests", "Whole stream equal", "Learner equal", "Resumes", "CPU-hours"], rows), ""]
    J["against_naive"] = dig

    # curves
    out.mkdir(parents=True)
    md += ["## The curves of score by age", "", "Mean of D2 over blocks of 64 visits (per visit in `curves_w<world>.csv`).", ""]
    for w in have:
        arms = ("naive", "sweep-after-naive", "sweep")
        sba = {a: q(a, w, 1024)["score_by_age"] for a in arms}
        rows = [[f"{64 * i} to {64 * i + 63}"] + [f"{np.mean(sba[a][64 * i:64 * i + 64]):.3f}" for a in arms] for i in range(16)]
        md += [f"**World {w}.**", "", table(["Visits", "`naive`", "`sweep-after-naive`", "`sweep`"], rows), ""]
        with open(out / f"curves_w{w}.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["visit", "kind"] + [f"{a}:score" for a in arms])
            for v in range(1024):
                wr.writerow([v, q("naive", w, 1024)["kind"][v]] + [f"{sba[a][v]:.6f}" for a in arms])
    (out / "E60.json").write_text(json.dumps(J, indent=1))
    (out / "E60_TABLES.md").write_text("\n".join(md))
    print("written", out)


if __name__ == "__main__":
    if sys.argv[1] == "final":
        final(sys.argv[2] if len(sys.argv) > 2 else "final")
    else:
        {"scores": scores}[sys.argv[1]]()
