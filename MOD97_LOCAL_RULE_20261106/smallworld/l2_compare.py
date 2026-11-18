"""E44, job L2: the worker's scorer (l2_score.quantities) against the reviewer's part J (e44_l2_checks.l2_quantities, kept_on_list), on one
life, quantity by quantity (the note's revision 2: "compared on the trial's two lives with the reviewer's part J").

    python3 l2_compare.py <events .npz> <list .npz> <world> <visits read> [--to OUT.json]

The reviewer's files are read from git (origin/claude/vibrant-cerf-sje726) into a temporary directory and imported from there; their
hashes are recorded. Compared: D2 (every visit, both columns), D4 (with the kinds of visit), D5 (both slopes), D6 (every family), D7, D8 and
D10 (the four absence classes: visits, items, share, the three figures), D9 (points, left out, mean, median). D14 has no counterpart in
part J; l2_score asserts for every answer that its class is `right` exactly when the file's `correct` is 1.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import l2_score as L                  # noqa: E402

REF = "origin/claude/vibrant-cerf-sje726"
RDIR = "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/"


def close(a, b, tol=1e-12):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return abs(a - b) <= tol


def compare(ev_path, list_path, world, V):
    ev, la = L.load(ev_path), L.load(list_path)
    mine = L.quantities(ev, la, world, V)
    with tempfile.TemporaryDirectory() as td:
        names = subprocess.run(["git", "ls-tree", "--name-only", REF, RDIR], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        hs = {}
        for n in names:
            if n.endswith(".py"):
                b = subprocess.run(["git", "show", f"{REF}:{n}"], cwd=ROOT, capture_output=True, check=True).stdout
                (Path(td) / Path(n).name).write_bytes(b)
                if Path(n).name in ("e44_l2_checks.py", "e44_world_reader.py", "np_gru.py"):
                    hs[Path(n).name] = hashlib.sha256(b).hexdigest()
        sys.path.insert(0, td)
        import e44_l2_checks as J
        theirs = J.l2_quantities(ev, V)
        sched = [s["group"] if isinstance(s, dict) else s for s in J.World(world).schedule(V)]
        kl = J.kept_on_list({k: v[la["at_visit"] < V] if True else v for k, v in la.items()}, sched)
        sys.path.remove(td)
    res = {}
    res["D2 score by age, every visit"] = all(close(a, b) for a, b in zip(mine["score_by_age"], theirs["score_by_age"])) and len(theirs["score_by_age"]) == V
    res["D2 the blind predictor's, every visit"] = all(close(a, b) for a, b in zip(mine["blind_by_age"], theirs["blind_by_age"]))
    res["D3 kinds of visit"] = mine["kind"] == theirs["kind"]
    res["D4 late level"] = close(mine["late_level"], theirs["late_level"]) and close(mine["blind_late_level"], theirs["blind_late"])
    res["D4 late level by kind"] = all(close(mine["late_level_by_kind"][k], theirs["late_level_by_kind"][k]) for k in ("first", "continuation", "return"))
    res["D5 score trend (returns)"] = close(mine["score_trend"], theirs["score_trend"])
    res["D5 score trend (all visits of the half)"] = close(mine["score_trend_all_visits"], theirs["score_trend_all"])
    res["D6 first-visit levels, every family"] = len(mine["first_visits"]) == len(theirs["first_visits"]) and all(
        a["family"] == b["family"] and a["visit"] == b["visit"] and close(a["level"], b["level"]) for a, b in zip(mine["first_visits"], theirs["first_visits"]))
    res["D7 trend of learning"] = close(mine["trend_of_learning"], theirs["trend_of_learning"])
    ok8 = ok10 = ok9 = True
    for name in ("1 visit", "2 to 4", "5 to 32", "33 or more"):
        a, b = mine["kept_at_return"][name], theirs["kept_at_return"][name]
        ok8 &= a["visits"] == b["visits"] and a["items"] == b["items"] and close(a["share"], b["share"])
        ok10 &= all(close(a[k], b[k]) for k in ("first16", "last_quarter", "previous_last_quarter"))
        a, b = mine["kept_on_list"][name], kl[name]
        ok9 &= a["points"] == b["points"] and a["left_out"] == b["left_out"] and close(a["mean"], b["mean"]) and close(a["median"], b["median"])
    res["D8 kept at return (visits, items, share), four classes"] = bool(ok8)
    res["D10 relearning at return (three figures), four classes"] = bool(ok10)
    res["D9 kept on the list (points, left out, mean, median), four classes"] = bool(ok9)
    return dict(events=str(ev_path), list=str(list_path), world=world, visits=V, reviewer=dict(ref=REF, commit=subprocess.run(
        ["git", "rev-parse", REF], cwd=ROOT, capture_output=True, text=True).stdout.strip(), files_sha256=hs), equal=res, all_equal=all(res.values()),
        figures=dict(late_level=mine["late_level"], score_trend=mine["score_trend"], trend_of_learning=mine["trend_of_learning"],
                     families=mine["families_met"], kept_at_return={k: v["share"] for k, v in mine["kept_at_return"].items()},
                     kept_on_list={k: v["mean"] for k, v in mine["kept_on_list"].items()}))


if __name__ == "__main__":
    r = compare(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    if "--to" in sys.argv:
        p = Path(sys.argv[sys.argv.index("--to") + 1])
        assert not p.exists(), f"refusing an existing path: {p}"
        p.write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
