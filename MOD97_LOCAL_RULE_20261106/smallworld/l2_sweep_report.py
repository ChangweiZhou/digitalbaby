"""E44, job L2, the sweep's long lives (the reviewer's instructions WORKER_E44_L2_SWEEP_LONG_LIVES.md of 17 November 2026, section 6):
the recorded quantities D1 to D15 of the eight lives, computed with the main worker's scorer (l2_score.py, unchanged), and the comparison
between the two learning rules. Nothing here trains.

    python3 l2_sweep_report.py scores         # every finished life of the sweep: l2_score.quantities at 256 and at 1,024 visits
                                              #   -> exp45/e44/L2/sweep/scores/<stem>_V<visits>.json (a file that exists is not written again)
    python3 l2_sweep_report.py main           # the main worker's finished lives under `naive` and `replay` in worlds 1 to 4, read from its
                                              #   branch (MAIN_REF) with `git show` and scored with the same scorer
                                              #   -> exp45/e44/L2/sweep/main/<stem>_V<visits>.json, with the commit and the files' hashes
    python3 l2_sweep_report.py tables <name>  # -> exp45/e44/L2/sweep/<name>/ : E44_L2_SWEEP.json, E44_L2_SWEEP_TABLES.md, curves_w<w>.csv
                                              #   (refused if the directory exists); worlds whose four lives are not all scored are named

Terms. The sweep: the learning rule that passes the error backward by the signs of the weights (the runner's arms `sweep` and
`sweep+replay`). Back-propagation: the exact gradient (arms `naive` and `replay`). The pairs compared: `sweep` against `naive`, and
`sweep+replay` against `replay` (both with rho = 0.5). A difference is always the sweep's figure minus back-propagation's.
"""
import csv
import hashlib
import io
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
import l2_sweep as SWP                # noqa: E402

SW = SWP.SW
LIVES, SCORES, MAIN = SW / "lives", SW / "scores", SW / "main"
MAIN_REF = "origin/claude/e44-l2-first-long-lives"
MAIN_LIVES = "MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/"
WORLDS = (1, 2, 3, 4)
ARMS = ("sweep", "naive", "sweep+replay", "replay")
PAIRS = (("sweep", "naive"), ("sweep+replay", "replay"))
ABS = ("1 visit", "2 to 4", "5 to 32", "33 or more", "5 or more")
VS = (256, 1024)


def stem(arm, w):
    return f"lw_{arm}_w{w}_v1024" + ("_rho0.5" if "replay" in arm else "")


def f3(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def sgn(x, d=4):
    return "—" if x is None else f"{x:+.{d}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def git(*a, text=True):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=text, check=True).stdout


# ---------------------------------------------------------------- scoring
def scores():
    SCORES.mkdir(parents=True, exist_ok=True)
    for arm in ("sweep", "sweep+replay"):
        for w in WORLDS:
            s = stem(arm, w)
            if not (LIVES / f"{s}.json").exists():
                continue
            for V in VS:
                p = SCORES / f"{s}_V{V}.json"
                if p.exists():
                    continue
                r = L.quantities(L.load(LIVES / f"{s}.npz"), L.load(LIVES / f"{s}_list.npz"), w, V)
                r["life"] = s
                p.write_text(json.dumps(r, indent=1))
                print("scored", s, V, flush=True)


def main_scores():
    """The main worker's lives, from its branch: fetched, read with `git show`, scored with the scorer of this checkout."""
    subprocess.run(["git", "fetch", "-q", "origin", "+refs/heads/claude/e44-l2-first-long-lives:refs/remotes/" + MAIN_REF], cwd=ROOT, check=True)
    commit = git("rev-parse", MAIN_REF).strip()
    names = set(git("ls-tree", "--name-only", MAIN_REF, MAIN_LIVES).split())
    MAIN.mkdir(parents=True, exist_ok=True)
    for arm in ("naive", "replay"):
        for w in WORLDS:
            s = stem(arm, w)
            need = [MAIN_LIVES + f"{s}{x}" for x in (".json", ".npz", "_list.npz")]
            if not all(n in names for n in need) or all((MAIN / f"{s}_V{V}.json").exists() for V in VS):
                continue
            blobs = [git("show", f"{MAIN_REF}:{n}", text=False) for n in need]
            rec = json.loads(blobs[0])
            assert rec["complete"] and rec["events"]["file_sha256"] == hashlib.sha256(blobs[1]).hexdigest(), s
            assert rec["list"]["file_sha256"] == hashlib.sha256(blobs[2]).hexdigest(), s
            ev = (lambda f: {k: f[k] for k in f.files})(np.load(io.BytesIO(blobs[1])))
            la = (lambda f: {k: f[k] for k in f.files})(np.load(io.BytesIO(blobs[2])))
            for V in VS:
                r = L.quantities(ev, la, w, V)
                r["life"] = s
                r["read_from"] = dict(branch="claude/e44-l2-first-long-lives", commit=commit, receipt=need[0], receipt_sha256=hashlib.sha256(blobs[0]).hexdigest(),
                                      events_file_sha256=rec["events"]["file_sha256"], list_file_sha256=rec["list"]["file_sha256"],
                                      settings={k: rec["settings"][k] for k in ("arm", "world", "visits", "rho", "H", "U", "visit_updates", "lr", "wd", "list_every",
                                                                                 "decay_distances")},
                                      learner=dict(file=rec["learner"]["file"], rule=rec["learner"]["rule"], seed=rec["learner"]["seed"],
                                                   final_row=rec["learner"]["final_row"]),
                                      cpu_total=rec["cpu"]["total"], resumes=len(rec["resumes"]), ledger_buffer=rec["ledger"]["buffer"],
                                      host=rec["provenance"]["host"], sources_sha256=rec["provenance"]["sources_sha256"],
                                      birth_family_before_visit_0=rec["list_summary"][0])
                (MAIN / f"{s}_V{V}.json").write_text(json.dumps(r, indent=1))
                print("scored (main worker's)", s, V, "at", commit[:8], flush=True)


def q_of(arm, w, V):
    p = (SCORES if "sweep" in arm else MAIN) / f"{stem(arm, w)}_V{V}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- the quantities of a comparison
def kept(q, cls):
    return q["kept_at_return"][cls]["share"]


QUANT = (("late level (D4)", lambda q: q["late_level"]),
         ("kept at return, absences of 5 to 32 visits (D8)", lambda q: kept(q, "5 to 32")),
         ("kept at return, absences of 33 visits or more (D8)", lambda q: kept(q, "33 or more")),
         ("score trend (D5)", lambda q: q["score_trend"]),
         ("trend of learning (D7)", lambda q: q["trend_of_learning"]))


def diff(a, b):
    return None if a is None or b is None else a - b


def compare(x, y, worlds, V):
    """x minus y, per world and over the worlds (D11), for the five quantities."""
    out = {}
    for name, fn in QUANT:
        per = {w: dict(x=fn(q_of(x, w, V)), y=fn(q_of(y, w, V))) for w in worlds}
        for w in per:
            per[w]["difference"] = diff(per[w]["x"], per[w]["y"])
        iv = L.interval90([per[w]["difference"] for w in worlds])
        out[name] = dict(per_world=per, interval=iv, verdict=L.verdict(iv))
    return out


def closed(c256, c1024):
    """D12 for a pair: the late-level difference distinguished on the same side at 256 and 1,024 visits, and at 1,024 visits neither the
    difference in D5 nor in D7 distinguished on the opposite side."""
    a, b = c256["late level (D4)"]["verdict"], c1024["late level (D4)"]["verdict"]
    if a != b or a not in ("higher", "lower"):
        return dict(closed=False, why=f"late-level difference: {a} at 256 visits, {b} at 1,024")
    opp = "lower" if a == "higher" else "higher"
    t5, t7 = c1024["score trend (D5)"]["verdict"], c1024["trend of learning (D7)"]["verdict"]
    return dict(closed=t5 != opp and t7 != opp, why=f"late-level difference {a} at both lengths; at 1,024 visits the difference in the score trend is {t5} and in "
                                                    f"the trend of learning {t7}")


def stuck(arm, worlds, V):
    t5 = L.interval90([q_of(arm, w, V)["score_trend"] for w in worlds])
    t7 = L.interval90([q_of(arm, w, V)["trend_of_learning"] for w in worlds])
    word = lambda iv: "not read" if iv["low"] is None else "improving" if iv["low"] > 0 else "declining" if iv["high"] < 0 else "not found"
    w5, w7 = word(t5), word(t7)
    return dict(score_trend=t5, score_trend_word=w5, trend_of_learning=t7, trend_of_learning_word=w7,
                statement="not read" if "not read" in (w5, w7) else "still learning" if "improving" in (w5, w7) else "stuck")


def sentence(c):
    lv, tr = c["late level (D4)"]["verdict"], c["score trend (D5)"]["verdict"]
    return dict(late_level={"higher": "ahead", "lower": "behind", "not distinguished": "level", "not read": "not read"}[lv],
                trend={"higher": "steeper", "lower": "flatter", "not distinguished": "the same", "not read": "not read"}[tr])


def blocks(q, size=64):
    V = q["visits"]
    return [float(np.mean(q["score_by_age"][b:min(V, b + size)])) for b in range(0, V, size)]


def pooled_classes(qs, part):
    """D14 pooled over lives: the answers of each class summed over the lives (counts recovered from the scorer's shares and totals)."""
    out = {}
    for cls in ABS:
        n = sum(q[part][cls]["answers"] for q in qs)
        tot = {k: sum(round(q[part][cls]["shares"][k] * q[part][cls]["answers"]) for q in qs if q[part][cls]["answers"]) for k in L.ANSWER_CLASSES}
        ctrl = sum(round(q[part][cls]["chance_never_met_modulus_in_place_of_the_last"] * q[part][cls]["answers"]) for q in qs if q[part][cls]["answers"])
        out[cls] = dict(answers=n, shares={k: (v / n if n else None) for k, v in tot.items()}, chance=(ctrl / n if n else None))
    return out


def pooled_d15(qs):
    """D15 pooled over lives, from the scorer's `later_visits`: by absence class and number of earlier visits."""
    out = {}
    for cname, lo, hi in L.CLASSES[1:4]:
        out[cname] = {}
        for en, elo, ehi in L.EARLIER:
            rs = [r for q in qs for r in q["later_visits"] if lo <= r["absence"] <= hi and elo <= r["earlier_visits"] <= ehi]
            n, k = sum(r["n"] for r in rs), sum(r["right"] for r in rs)
            f16 = [r["first16"] for r in rs if r["first16"] is not None]
            out[cname][en] = dict(returns=len(rs), items=n, right=k, share=(k / n if n else None), first16=float(np.mean(f16)) if f16 else None)
    return out


def receipt(arm, w):
    if "sweep" in arm:
        return json.loads((LIVES / f"{stem(arm, w)}.json").read_text())
    return None


# ---------------------------------------------------------------- the tables
def tables(name):
    out_dir = SW / name
    assert not out_dir.exists(), f"refusing an existing path: {out_dir}"
    have = {(a, w) for a in ARMS for w in WORLDS if all(q_of(a, w, V) is not None for V in VS)}
    own = [w for w in WORLDS if ("sweep", w) in have or ("sweep+replay", w) in have]
    pair_worlds = {f"{x} against {y}": [w for w in WORLDS if (x, w) in have and (y, w) in have] for x, y in PAIRS}
    store_worlds = [w for w in WORLDS if ("sweep", w) in have and ("sweep+replay", w) in have]
    main_commit = sorted({q_of(a, w, 256)["read_from"]["commit"] for a in ("naive", "replay") for w in WORLDS if (a, w) in have})
    md = [f"# E44, job L2, the sweep's long lives: tables ({name})", "",
          "Written by `smallworld/l2_sweep_report.py tables " + name + "` from the score files under `exp45/e44/L2/sweep/scores/` (the sweep's lives) "
          "and `exp45/e44/L2/sweep/main/` (the main worker's lives under back-propagation, read from `claude/e44-l2-first-long-lives` at "
          + (", ".join(c[:8] for c in main_commit) or "no commit yet") + "). Every figure is the scorer's (`smallworld/l2_score.py`, unchanged). "
          "A difference is the sweep's figure minus back-propagation's. Intervals are two-sided 90% intervals (Student's t over worlds).", "",
          "Lives scored: " + "; ".join(f"`{a}` worlds {', '.join(str(w) for w in WORLDS if (a, w) in have) or 'none'}" for a in ARMS) + ".", ""]
    J = dict(name=name, lives_scored={a: [w for w in WORLDS if (a, w) in have] for a in ARMS}, main_workers_commit=main_commit,
             scorer_sha256=hashlib.sha256((HERE / "l2_score.py").read_bytes()).hexdigest())

    # births
    rows, births = [], {}
    for seed in SWP.SEEDS:
        for rule in ("B", "S"):
            r = json.loads((MDIR / "exp45" / "e41" / f"e41_{rule}_h256_f50_{seed}.json").read_text())
            last = r["curve"][-1]
            births[f"{rule}_{seed}"] = dict(A=last["A"], R=last["R"], updates=last["step"], receipt=f"exp45/e41/e41_{rule}_h256_f50_{seed}.json")
            rows.append([seed, {"B": "back-propagation", "S": "the sweep"}[rule], f3(last["A"], 2), f3(last["R"], 2), last["step"]])
    md += ["## The births", "", "The two rules start from different births: each life starts from E41's half-width birth (H = 256, U = 128, taught "
           "fraction 0.50) made under its own rule. A: per cent right on E41's held-out panel of add89 and sub89; R: per cent right on E41's 2,000 "
           "held-out reduction pairs (the receipts' final rows). Worlds 1 and 4 use seed 73129, world 2 seed 93683, world 3 seed 116089.", "",
           table(["Seed", "Rule of the birth", "A", "R", "Updates"], rows), ""]
    J["births"] = births

    # per life
    md += ["## The quantities of each life (D2 to D7)", "",
           "Late level: the mean score on reduction items over the last quarter of the visits (D4), beside the blind predictor's and the level by "
           "kind of visit. Score trend: the slope of the score on ln(visit + 1) over the returns of the second half (D5); beside it the slope over "
           "all visits of that half (not used). First-visit level: D6, the median over the families met. Trend of learning: the slope of D6 on ln n "
           "(D7), beside the coefficient of ln n in the joint fit with the logarithm of the taught pairs.", ""]
    rows, per_life = [], {}
    for V in VS:
        for a in ARMS:
            for w in WORLDS:
                if (a, w) not in have:
                    continue
                q = q_of(a, w, V)
                jf = q["trend_of_learning_joint_fit"]
                per_life[f"{a}|{w}|{V}"] = {k: q[k] for k in ("late_level", "blind_late_level", "late_level_by_kind", "score_trend", "score_trend_returns",
                                                               "score_trend_all_visits", "median_first_visit_level", "trend_of_learning",
                                                               "trend_of_learning_joint_fit", "families_met", "kinds")}
                rows.append([V, f"`{a}`", w, q["families_met"], f3(q["late_level"]), f3(q["blind_late_level"]), f3(q["late_level_by_kind"]["first"]),
                             f3(q["late_level_by_kind"]["continuation"]), f3(q["late_level_by_kind"]["return"]), sgn(q["score_trend"]),
                             q["score_trend_returns"], sgn(q["score_trend_all_visits"]), f3(q["median_first_visit_level"]), sgn(q["trend_of_learning"]),
                             sgn(jf["ln_n"]) if jf else "—"])
    md += [table(["Visits", "Arm", "World", "Families", "Late level", "Blind", "First visits", "Continuations", "Returns", "Score trend", "Returns in it",
                  "Slope, all visits", "Median first-visit level", "Trend of learning", "Joint fit, ln n"], rows), ""]
    J["per_life"] = per_life

    # D8, D9, D10
    md += ["## Kept at return (D8), kept on the list (D9) and relearning at return (D10)", "",
           "D8: the share right on the reduction items of the first update of a later visit, pooled over the visits of an absence class (right / "
           "items). D9: the family's share right on its 200 list items divided by its share at the end of its most recent visit; mean and median "
           "over the points, and the points left out because the earlier share was 0. D10: the family's figure over the first 16 updates and "
           "over the last 64 updates of the visit, beside the last 64 updates of the family's previous visit, averaged over the visits.", ""]
    for V in VS:
        rows = []
        for a in ARMS:
            for w in WORLDS:
                if (a, w) not in have:
                    continue
                q = q_of(a, w, V)
                for c in ABS:
                    k, li = q["kept_at_return"][c], q["kept_on_list"][c]
                    rows.append([f"`{a}`", w, c, k["visits"], f"{f3(k['share'])} ({k['right']}/{k['items']})", f3(k["first16"]), f3(k["last_quarter"]),
                                 f3(k["previous_last_quarter"]), li["points"], li["left_out"], f3(li["mean"]), f3(li["median"]), f3(li["mean_share_right"])])
        md += [f"**Up to {V:,} visits.**", "",
               table(["Arm", "World", "Absence (visits)", "Visits", "D8 kept at return", "D10 first 16 updates", "D10 last 64", "D10 previous visit's last 64",
                      "D9 points", "D9 left out", "D9 mean", "D9 median", "List share right"], rows), ""]

    # D14
    md += ["## What the wrong answers are (D14)", "", "Pooled over the worlds scored for the arm. Stream: the reduction items of the first update of a "
           "later visit. List: the list's rows of D9. Classes tried in the scorer's order; the chance line is the never-met modulus (family 55's) "
           "tried in the place of the family taught last.", ""]
    d14 = {}
    short = ("right", "no reduction", "modulus taught last", "earlier modulus", "89", "an operand", "never-met (control)", "other two digits", "not two digits")
    for V in VS:
        for part, label in (("wrong_answers_stream", "stream"), ("wrong_answers_list", "list")):
            rows = []
            for a in ARMS:
                qs = [q_of(a, w, V) for w in WORLDS if (a, w) in have]
                if not qs:
                    continue
                pc = pooled_classes(qs, part); d14[f"{a}|{V}|{label}"] = pc
                for c in ABS[:4]:
                    rows.append([f"`{a}`", c, pc[c]["answers"]] + [f3(pc[c]["shares"][k]) for k in L.ANSWER_CLASSES] + [f3(pc[c]["chance"])])
            md += [f"**Up to {V:,} visits, {label}.**", "", table(["Arm", "Absence", "Answers", *short, "Chance line"], rows), ""]
    J["wrong_answers"] = d14

    # D15
    md += ["## Kept at return by the number of earlier visits to the family (D15)", "", "Pooled over the worlds scored for the arm. In each cell: "
           "kept at return (right / items; returns), and after the semicolon the mean share right over the first 16 updates of the return.", ""]
    d15 = {}
    for V in VS:
        rows = []
        for a in ARMS:
            qs = [q_of(a, w, V) for w in WORLDS if (a, w) in have]
            if not qs:
                continue
            p = pooled_d15(qs); d15[f"{a}|{V}"] = p
            for c in p:
                rows.append([f"`{a}`", c] + [("—" if not x["returns"] else f"{f3(x['share'])} ({x['right']}/{x['items']}; {x['returns']}); {f3(x['first16'])}")
                                              for x in p[c].values()])
        md += [f"**Up to {V:,} visits.**", "", table(["Arm", "Absence (visits)"] + [f"{en} earlier" for en, _, _ in L.EARLIER], rows), ""]
    J["kept_by_earlier_visits"] = d15

    # stuck
    md += ["## Stuck or still learning (D13)", "", "Over the worlds scored for the arm: the mean and 90% interval of the score trend (D5) and of the "
           "trend of learning (D7). *Stuck*: neither interval lies above zero. *Declining*: an interval wholly below zero.", ""]
    rows, st = [], {}
    for V in VS:
        for a in ARMS:
            ws = [w for w in WORLDS if (a, w) in have]
            if len(ws) < 2:
                continue
            s = stuck(a, ws, V); st[f"{a}|{V}"] = dict(worlds=ws, **s)
            iv = lambda i: f"{sgn(i['mean'])} [{sgn(i['low'])}, {sgn(i['high'])}]"
            rows.append([V, f"`{a}`", ", ".join(map(str, ws)), iv(s["score_trend"]), s["score_trend_word"], iv(s["trend_of_learning"]),
                         s["trend_of_learning_word"], s["statement"]])
    md += [table(["Visits", "Arm", "Worlds", "Score trend", "", "Trend of learning", "", "Statement"], rows) if rows else "Fewer than two worlds are scored.", ""]
    J["stuck"] = st

    # between the rules
    md += ["## Between the rules (D11, D12)", "", "Per world: the sweep's figure, back-propagation's, and the difference. Over the worlds: the mean "
           "difference with its 90% interval, and *higher*, *lower* or *not distinguished*.", ""]
    between = {}
    comps = [(x, y, pair_worlds[f"{x} against {y}"]) for x, y in PAIRS] + [("sweep+replay", "sweep", store_worlds)]
    for x, y, ws in comps:
        key = f"{x} against {y}"
        if not ws:
            md += [f"**`{x}` against `{y}`**: no world has both lives scored yet.", ""]; continue
        cs = {V: compare(x, y, ws, V) for V in VS}
        between[key] = dict(worlds=ws, at={str(V): cs[V] for V in VS})
        rows = []
        for V in VS:
            for qn, _ in QUANT:
                c = cs[V][qn]; iv = c["interval"]
                cells = [f"{f3(c['per_world'][w]['x'], 4)} − {f3(c['per_world'][w]['y'], 4)} = {sgn(c['per_world'][w]['difference'])}" if w in ws else "—" for w in WORLDS]
                rows.append([V, qn, *cells, sgn(iv["mean"]), "—" if iv["low"] is None else f"[{sgn(iv['low'])}, {sgn(iv['high'])}]", c["verdict"]])
        md += [f"**`{x}` against `{y}`** (worlds {', '.join(map(str, ws))}).", "",
               table(["Visits", "Quantity", *[f"World {w}" for w in WORLDS], "Mean difference", "90% interval", ""], rows), ""]
        if (x, y) in PAIRS:
            cl, se = closed(cs[256], cs[1024]), sentence(cs[1024])
            between[key].update(closed=cl, sentence=se, sentence_256=sentence(cs[256]))
            md += [f"Closed (D12): **{'closed' if cl['closed'] else 'open'} at 1,024 visits** ({cl['why']}). The fixed sentence: over 1,024 visits, "
                   f"with {'the store' if 'replay' in x else 'no store'}, the sweep is **{se['late_level']}** in late level, and its score trend over the "
                   f"second half is **{se['trend']}** than back-propagation's" + (" (as back-propagation's)" if se["trend"] == "the same" else "") + ". "
                   f"At 256 visits the sweep is {sentence(cs[256])['late_level']} in late level. " + ("" if len(ws) == 4 else f"**Only {len(ws)} of the four worlds "
                   "are scored; the sentence is provisional.**"), ""]
    J["between"] = between

    # curves
    md += ["## The curves of score by age", "", "Mean score on reduction items over blocks of 64 visits, for the four arms in each world (the "
           "per-visit curves are in `curves_w<world>.csv`). The last column of each pair marks the blocks in which the sweep's mean is at or above "
           "back-propagation's.", ""]
    out_dir.mkdir(parents=True)
    curves = {}
    for w in WORLDS:
        arms = [a for a in ARMS if (a, w) in have]
        if not arms:
            continue
        qs = {a: q_of(a, w, 1024) for a in arms}
        bl = {a: blocks(qs[a]) for a in arms}; curves[str(w)] = bl
        head = ["Visits"] + [f"`{a}`" for a in arms] + [f"`{x}` ≥ `{y}`" for x, y in PAIRS if x in arms and y in arms]
        rows = []
        for i in range(16):
            rows.append([f"{64 * i} to {64 * i + 63}"] + [f3(bl[a][i]) for a in arms]
                        + [("yes" if bl[x][i] >= bl[y][i] else "") for x, y in PAIRS if x in arms and y in arms])
        md += [f"**World {w}.**", "", table(head, rows), ""]
        with open(out_dir / f"curves_w{w}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["visit", "kind", "family"] + [f"{a}:score" for a in arms] + ["blind"])
            q0 = qs[arms[0]]
            w_ = L.LW.World(w, 256); sched = w_.schedule(1024)
            for v in range(1024):
                wr.writerow([v, q0["kind"][v], sched[v]["group"]] + [f"{qs[a]['score_by_age'][v]:.6f}" for a in arms] + [f"{q0['blind_by_age'][v]:.6f}"])
    J["curves_blocks_of_64"] = curves

    # cost and ledger of the sweep's lives
    rows, led = [], {}
    for a in ("sweep", "sweep+replay"):
        for w in WORLDS:
            if not (LIVES / f"{stem(a, w)}.json").exists():
                continue
            r = receipt(a, w); b = r["ledger"]["buffer"]
            led[f"{a}|{w}"] = dict(cpu_hours=r["cpu"]["total"] / 3600, resumes=len(r["resumes"]), buffer=b, learner=r["learner"]["file"],
                                   birth_family_before_visit_0=r["list_summary"][0], events_file_sha256=r["events"]["file_sha256"],
                                   list_file_sha256=r["list"]["file_sha256"], final=r["final"], host=r["provenance"]["host"],
                                   stored_numbers=r["ledger"]["stored_numbers"], replayed_per_update=r["ledger"]["replayed_per_update"])
            rows.append([f"`{a}`", w, r["learner"]["seed"], f3(r["list_summary"][0]["share"]), f3(r["cpu"]["total"] / 3600, 2), len(r["resumes"]),
                         b["capacity_items"], b["max_items"], r["ledger"]["replayed_per_update"], r["events"]["file_sha256"][:12]])
    md += ["## The sweep's lives: cost and ledger", "", "From the receipts under `exp45/e44/L2/sweep/lives/`. Share right at birth: the birth family's "
           "2,000 reduction items answered before visit 0. Stored numbers beyond the weights (317,457) and the optimizer's moments (634,914): the "
           "store of `sweep+replay`, 55,209 items of 23 bytes (1,269,828 bytes).", "",
           table(["Arm", "World", "Birth seed", "Share right at birth (R items)", "CPU-hours", "Resumes", "Store capacity (items)", "Store maximum", "Replayed an update",
                  "Event file SHA-256"], rows), ""]
    J["ledger"] = led
    (out_dir / "E44_L2_SWEEP.json").write_text(json.dumps(J, indent=1))
    (out_dir / "E44_L2_SWEEP_TABLES.md").write_text("\n".join(md))
    print("written", out_dir)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "tables":
        tables(sys.argv[2])
    else:
        {"scores": scores, "main": main_scores}[c]()
