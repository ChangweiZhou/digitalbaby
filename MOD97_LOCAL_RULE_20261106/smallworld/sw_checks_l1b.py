"""E39 revision 5, job L1b: checks 11 to 15 (check 10 is exp45/e39/L1b/E39_CHECK10.json).

    python3 sw_checks_l1b.py c13                 # the scorer with birth groups (no lives needed)
    python3 sw_checks_l1b.py assemble <H>        # checks 11-15 on the L1b lives at size H -> exp45/e39/L1b/E39_L1B_CHECKS.json

11  the stream digest of the 64 visits after a birth equals sw_world.digest of that world at that visit length (every L1b life)
12  a life with a birth stopped at the state of visit 4 and resumed (L1b/c12/) gives the hashes and event file of the uninterrupted one
13  hand-made file with two birth tasks and two later tasks (every quantity, hand-computed); the reviewer's fixture (its assertions);
    without --birth-groups the scorer reproduces all 48 L1 score tables
14  the event files the reviewer recomputes from (72 solo runs, world 1's births), listed with hashes and this runner's own values
15  sweep-after-naive's state at visit 0 has the weights and optimizer-state hashes of naive's state at the end of its birth
"""
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MDIR = HERE.parent
ROOT = MDIR.parent
sys.path.insert(0, str(HERE))
import sw_world as W                  # noqa: E402
import sw_score as S                  # noqa: E402

E = MDIR / "exp45" / "e39"
L1B = E / "L1b"
REVIEWER = ("ae61164e", "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_world.py")
BV = S.BIRTH_VISIT


def close(a, b, tol=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


# ---------------------------------------------------------------- check 13
def hand_birth():
    """Birth tasks A = 0, B = 1; later tasks C = 2, D = 3. 8 updates a visit (and in the birth), 8 asked items per task per update,
    24 taught per task per update. Birth: A right from update 4, B never. Visit 0 (A, B): A wrong in the first update only, B right in
    the last two updates. Visit 1 (C, D): C right from update 2, D from update 5. Visit 2 (A, B): A always, B never. Visit 3 (C, D): C
    always, D on items k < 4. wrap = k % 2; mirror (A only, the symmetric task) for k < 2; blind for C's item k = 0."""
    A, B, C, D = 0, 1, 2, 3
    ev = {k: [] for k in ("update", "visit", "task", "correct", "blind", "wrap", "mirror", "seen", "replayed")}

    def add(u, v, t, right_fn, seen):
        for k in range(8):
            ev["update"].append(u); ev["visit"].append(v); ev["task"].append(t); ev["correct"].append(int(right_fn(k)))
            ev["blind"].append(int(t == C and k == 0)); ev["wrap"].append(k % 2); ev["mirror"].append(int(t == A and k < 2))
            ev["seen"].append(seen); ev["replayed"].append(0)
    for u in range(8):
        add(u, BV, A, lambda k: u >= 4, 24 * u); add(u, BV, B, lambda k: False, 24 * u)
    for v in range(4):
        for i in range(8):
            u = 8 + 8 * v + i
            if v == 0:
                add(u, v, A, lambda k: i > 0, 24 * (8 + i)); add(u, v, B, lambda k: i >= 6, 24 * (8 + i))
            elif v == 1:
                add(u, v, C, lambda k: i >= 2, 24 * i); add(u, v, D, lambda k: i >= 5, 24 * i)
            elif v == 2:
                add(u, v, A, lambda k: True, 24 * (16 + i)); add(u, v, B, lambda k: False, 24 * (16 + i))
            else:
                add(u, v, C, lambda k: True, 24 * (8 + i)); add(u, v, D, lambda k: k < 4, 24 * (8 + i))
    ev = {k: np.array(x) for k, x in ev.items()}
    hand = dict(
        score_by_age=[72 / 128, 72 / 128, 64 / 128, 96 / 128], blind_by_age=[0.0, 8 / 128, 0.0, 8 / 128],
        first_visit=[dict(n=1, task=C, visit=1, level=1.0, cost=120), dict(n=2, task=D, visit=1, level=0.75, cost=None)],
        cost=[dict(n=1, task=C, cost=120, level=None), dict(n=2, task=D, cost=192, level=None)],
        learning_trend=(0.75 - 1.0) / math.log(2), cost_trend=None,
        kept={(A, 0): (1, 1.0, 0, 8, 0, 4, 0, 4, 24, 32), (A, 2): (2, 1.0, 8, 8, 4, 4, 4, 4, 32, 32), (B, 0): (1, 0.0, 0, 8, 0, 4, 0, 4, 0, 32),
              (B, 2): (2, 0.5, 0, 8, 0, 4, 0, 4, 0, 32), (C, 3): (2, 1.0, 8, 8, 4, 4, 4, 4, 32, 32), (D, 3): (2, 0.75, 4, 8, 2, 4, 2, 4, 16, 32)},
        relearn={(A, 0): 96, (A, 2): 72, (B, 0): None, (B, 2): None, (C, 3): 72, (D, 3): None},
        pooled_short_first_update=(20, 48), pooled_short_relearn=(3, 72.0),
        symmetric=dict(all=(120, 128), mirror=(30, 32), no_mirror=(90, 96), non_symmetric=(184, 384)),
        late_level=0.75, score_trend=(96 / 128 - 64 / 128) / (math.log(4) - math.log(3)), mean_score=304 / 512,
        end_of_visit={(A, -1): 1.0, (A, 0): 1.0, (A, 2): 1.0, (B, -1): 0.0, (B, 0): 0.5, (B, 2): 0.0, (C, 1): 1.0, (C, 3): 1.0, (D, 1): 0.75, (D, 3): 0.5},
        birth={A: (64, 0.5, 1.0), B: (64, 0.0, 0.0)})
    return ev, {A}, [A, B], hand


def check_hand_birth():
    ev, sym, bt, h = hand_birth()
    r = S.score(ev, 4, sym, bt)
    f = []
    for v in range(4):
        if not close(r["score_by_age"][v], h["score_by_age"][v]): f.append(("score_by_age", v))
        if not close(r["blind_by_age"][v], h["blind_by_age"][v]): f.append(("blind_by_age", v))
    if r["first_visit"] != h["first_visit"]: f.append(("first_visit", r["first_visit"]))
    if r["cost"] != h["cost"]: f.append(("cost", r["cost"]))
    for q in ("learning_trend", "cost_trend", "late_level", "score_trend", "mean_score"):
        if not close(r[q], h[q], 1e-12): f.append((q, r[q], h[q]))
    if len(r["kept"]) != 6: f.append(("kept count", len(r["kept"])))
    for k in r["kept"]:
        got = (k["absence"], k["before"], k["first_update"]["right"], k["first_update"]["n"], k["first_update_wrap"]["right"], k["first_update_wrap"]["n"],
               k["first_update_nowrap"]["right"], k["first_update_nowrap"]["n"], k["first32"]["right"], k["first32"]["n"])
        if got != h["kept"][(k["task"], k["visit"])]: f.append(("kept", k["task"], k["visit"], got))
    for x in r["relearn"]:
        if x["items"] != h["relearn"][(x["task"], x["visit"])]: f.append(("relearn", x))
    p = r["kept_pooled"]["short"]
    if (p["first_update"]["right"], p["first_update"]["n"]) != h["pooled_short_first_update"]: f.append(("pooled", p["first_update"]))
    if (p["relearn_reached"], p["relearn_median_items"]) != h["pooled_short_relearn"]: f.append(("pooled relearn", p))
    if r["kept_pooled"]["medium"]["returns"] or r["kept_pooled"]["long"]["returns"]: f.append(("pooled classes",))
    for part, val in h["symmetric"].items():
        if (r["symmetric"][part]["right"], r["symmetric"][part]["n"]) != val: f.append(("symmetric", part))
    eov = {(x["task"], x["visit"]): x["level"] for x in r["end_of_visit"]}
    if eov.keys() != h["end_of_visit"].keys() or any(not close(eov[k], v) for k, v in h["end_of_visit"].items()): f.append(("end_of_visit", eov))
    for t, (n, sh, l32) in h["birth"].items():
        b = r["birth"]["per_task"][t]
        if (b["n"], b["share"], b["last32"]) != (n, sh, l32): f.append(("birth", t, b))
    return f


def reviewer_module():
    tmp = Path(tempfile.mkdtemp(prefix="e39_reviewer_"))
    src = subprocess.run(["git", "show", f"{REVIEWER[0]}:{REVIEWER[1]}"], cwd=ROOT, capture_output=True, check=True).stdout
    p = tmp / "e39_world.py"; p.write_bytes(src)
    spec = importlib.util.spec_from_file_location("e39_world_reviewer", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod, dict(commit=REVIEWER[0], path=REVIEWER[1], sha256=hashlib.sha256(src).hexdigest(), file=str(p))


def check_reviewer_fixture(rv):
    """The reviewer's fixture (e39_world.hand_made_birth), rebuilt from its own lines with a zero `mirror` column, scored by this scorer;
    its assertions applied to this scorer's output."""
    BIRTH = rv.BIRTH_VISIT
    ev = dict(update=[], visit=[], task=[], correct=[], blind=[], wrap=[], seen=[])

    def add(u, v, t, right, seen):
        for j in range(4):
            ev["update"].append(u); ev["visit"].append(v); ev["task"].append(t); ev["correct"].append(int(right))
            ev["blind"].append(0); ev["wrap"].append(j % 2); ev["seen"].append(seen)
    for u in range(8):
        for t in (0, 1):
            add(u, BIRTH, t, t == 0, u * 24)
    for u in range(48):
        v, k = u // 16, u % 16
        if v in (0, 2):
            add(8 + u, v, 0, k != 0, (8 + (u if v == 0 else u - 16)) * 24)
            add(8 + u, v, 1, False, (8 + (u if v == 0 else u - 16)) * 24)
        else:
            add(8 + u, v, 2, k >= 6, k * 24)
            add(8 + u, v, 3, k >= 12, k * 24)
    ev = {k: np.array(x) for k, x in ev.items()}; ev["mirror"] = np.zeros_like(ev["task"]); ev["replayed"] = np.zeros_like(ev["task"])
    r = S.score(ev, 3, set(), [0, 1])
    f = []
    if [(d["n"], d["task"]) for d in r["first_visit"]] != [(1, 2), (2, 3)]: f.append("first_visit order")
    if not (r["first_visit"][0]["level"] == 1.0 and r["first_visit"][0]["cost"] == 288): f.append("first_visit task 2")
    if not (r["first_visit"][1]["level"] == 16 / 32 and r["first_visit"][1]["cost"] is None): f.append("first_visit task 3")
    if not close(r["learning_trend"], (0.5 - 1.0) / math.log(2)): f.append("learning_trend")
    if [(k["visit"], k["absence"], k["first_update"]["share"]) for k in r["kept"] if k["task"] == 0] != [(0, 1, 0.0), (2, 2, 0.0)]: f.append("kept task 0")
    if [(k["visit"], k["absence"]) for k in r["kept"] if k["task"] == 1] != [(0, 1), (2, 2)]: f.append("kept task 1")
    if [k for k in r["kept"] if k["task"] in (2, 3)]: f.append("kept later tasks")
    if not close(r["score_by_age"][0], 60 / 128): f.append("score_by_age")
    r0 = S.score(ev, 3, set())
    if [d["task"] for d in r0["first_visit"]] != [0, 1, 2, 3]: f.append("without birth tasks")
    selftest = subprocess.run([sys.executable, "-I", rv.__file__, "spec", "scorer"], capture_output=True, text=True)
    return f, selftest.returncode, selftest.stdout.strip().splitlines()[-1:]


def check_l1_unchanged():
    n = bad = 0
    for tp in sorted(E.glob("sw_*_v64*.score.json")):
        st = tp.name[:-len(".score.json")]
        if "birth" in st:
            continue
        w = int(re.search(r"_w(\d+)_", st).group(1))
        f = np.load(E / f"{st}.npz"); ev = {k: f[k] for k in f.files}
        n += 1; bad += json.loads(json.dumps(S.score(ev, 64, S.symmetric_tasks(w)))) != json.loads(tp.read_text())
    return n, bad


def c13():
    rv, info = reviewer_module()
    mine = check_hand_birth()
    rf, rc, rl = check_reviewer_fixture(rv)
    n, bad = check_l1_unchanged()
    return dict(passed=not mine and not rf and rc == 0 and n == 48 and bad == 0, hand_made_failures=[str(x) for x in mine],
                reviewer_fixture_failures=rf, reviewer_selftest=dict(returncode=rc, last_line=rl), reviewer=info,
                l1_score_tables_rescored=n, l1_score_tables_differing=bad,
                note="Hand-made file: birth tasks 0 and 1 (birth of 8 updates), later tasks 2 and 3, visits 0-3; values computed by hand in the "
                     "docstring of hand_birth. The reviewer's fixture is rebuilt from its own lines (a zero mirror column added) and its "
                     "assertions applied to this scorer.")


# ---------------------------------------------------------------- checks 11, 12, 14, 15
def l1b_stem(arm, w, H):
    tag = (f"_h{H}_u{H}" if H != 64 else "") + "_vu512_birth01"
    return f"sw_{arm}_w{w}_v64{tag}"


def rec(p):
    return json.loads(Path(p).read_text())


def assemble(H):
    path = L1B / "E39_L1B_CHECKS.json"
    assert not path.exists(), path
    out = {}
    # 11
    rows, ok = [], True
    for w in range(1, 9):
        g = W.digest(W.World(w, 512), 64)
        for arm in ("naive", "sweep", "sweep-after-naive"):
            r = rec(E / f"{l1b_stem(arm, w, H)}.json")
            rows.append(dict(world=w, arm=arm, stream_sha256=r["stream_sha256"], generator=g, equal=r["stream_sha256"] == g)); ok &= rows[-1]["equal"]
    out["11_life_after_birth"] = dict(passed=ok, lives=rows)
    # 12
    st = l1b_stem("naive", 1, H)
    a, b = rec(E / f"{st}.json"), rec(L1B / "c12" / f"{st}.json")
    ea, eb = np.load(E / f"{st}.npz"), np.load(L1B / "c12" / f"{st}.npz")
    same = sorted(ea.files) == sorted(eb.files) and all(np.array_equal(ea[k], eb[k]) for k in ea.files)
    v4a = [x for x in a["states"] if x["visits"] == 4][0]; v4b = [x for x in b["states"] if x["visits"] == 4][0]
    good = a["final"] == b["final"] and same and a["events"]["arrays_sha256"] == b["events"]["arrays_sha256"] and a["stream_sha256"] == b["stream_sha256"] \
        and a["learned_sha256"] == b["learned_sha256"] and len(b["resumes"]) == 1 and b["resumes"][0]["done_visits"] == 4 \
        and a["birth"]["stream_sha256"] == b["birth"]["stream_sha256"] and a["birth"]["updates"] == b["birth"]["updates"] \
        and (v4a["weights_tensors_sha256"], v4a["optimizer_sha256"]) == (v4b["weights_tensors_sha256"], v4b["optimizer_sha256"])
    out["12_birth_and_resume"] = dict(passed=good, life=st, uninterrupted=a["final"], resumed=b["final"], events_equal=same,
                                      events_sha=(a["events"]["arrays_sha256"], b["events"]["arrays_sha256"]), resumes=b["resumes"],
                                      birth_updates=(a["birth"]["updates"], b["birth"]["updates"]), state_v4_equal=(v4a["weights_tensors_sha256"] == v4b["weights_tensors_sha256"]),
                                      note="c12: a separate run of the same life (its own birth), stopped after the state of visit 4, then "
                                           "resumed from that state to 64 visits; compared with the uninterrupted L1b life")
    # 13
    out["13_scorer_with_birth"] = c13()
    # 14
    solo = sorted((E / "L1a").glob("sw_solo_*.npz"))
    births = [E / f"{l1b_stem(arm, 1, H)}.npz" for arm in ("naive", "sweep", "sweep-after-naive")]
    l1a = rec(E / "L1a" / "E39_L1A.json")
    out["14_files_for_the_reviewer"] = dict(
        passed=len(solo) == 72 and all(p.exists() for p in births), note="The reviewer recomputes; this check does not hold the runs.",
        solo_event_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in solo},
        birth_event_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in births},
        worker_values=dict(solo={x["receipt"]: dict(updates_to_criterion=x["updates_to_criterion"], updates_to_half=x["updates_to_half"]) for x in l1a["rows"]},
                           births={p.name: dict(birth_updates=rec(p.with_suffix(".json"))["birth"]["updates"],
                                                complete=rec(p.with_suffix(".json"))["birth"]["complete"]) for p in births}),
        definitions="updates to criterion: the first check k (after 64k updates) at which the share right over the task's asked items of updates "
                    "64(k-1) .. 64k-1 is at least 0.8 (sw_score.criterion_checks); the birth ends at the first such check at which three of its four "
                    "tasks are at the criterion; its last update is the birth's update count minus one; birth rows have visit 65535 and updates 0 .. B-1")
    # 15
    rows, ok = [], True
    for w in range(1, 9):
        n_, s_ = rec(E / f"{l1b_stem('naive', w, H)}.json"), rec(E / f"{l1b_stem('sweep-after-naive', w, H)}.json")
        a0 = [x for x in n_["states"] if x["visits"] == 0][0]; b0 = [x for x in s_["states"] if x["visits"] == 0][0]
        good = (a0["weights_tensors_sha256"], a0["optimizer_sha256"]) == (b0["weights_tensors_sha256"], b0["optimizer_sha256"]) \
            and s_["birth_from"]["weights_tensors_sha256"] == a0["weights_tensors_sha256"]
        rows.append(dict(world=w, naive_v0=(a0["weights_tensors_sha256"], a0["optimizer_sha256"]), san_v0=(b0["weights_tensors_sha256"], b0["optimizer_sha256"]), equal=good))
        ok &= good
    out["15_third_arm_start"] = dict(passed=ok, worlds=rows)
    res = dict(experiment="E39", job="L1b checks 11-15", brief="revision 5 (8183db86)", H=H, checks=out, all_passed=all(c["passed"] for c in out.values()),
               check10="E39_CHECK10.json (passed)", sources_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob("sw_*.py"))})
    path.write_text(json.dumps(res, indent=1))
    for k, c in out.items():
        print(k, "PASS" if c["passed"] else "FAIL")


if __name__ == "__main__":
    if sys.argv[1] == "c13":
        r = c13(); print(json.dumps({k: v for k, v in r.items() if k != "reviewer"}, indent=1))
    else:
        assemble(int(sys.argv[2]))
