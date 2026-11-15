"""E39 job L0: the nine checks of the brief (OPUS_TASK_E39_SMALL_WORLD_PROTOCOL_L0_L1_20261115.md, revision 2).

    python3 sw_checks.py lives       # the check lives, at most three processes at a time, into exp45/e39/L0/lives/
    python3 sw_checks.py assemble    # every check -> exp45/e39/L0/E39_L0_CHECKS.json (refused if it exists)

The reviewer's second implementation is read from git (REVIEWER_COMMIT:REVIEWER_PATH) into a temporary directory and imported from there;
its hash is recorded. Nothing in this file trains except the check lives (through sw_run.py) and the in-process 64-update runs of
checks 3 and 6.
"""
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from random import Random

import numpy as np
import torch as T

HERE = Path(__file__).resolve().parent
MDIR = HERE.parent
ROOT = MDIR.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(MDIR))
import sw_world as W                  # noqa: E402
import sw_net as N                    # noqa: E402
import sw_run as R                    # noqa: E402
import sw_score as S                  # noqa: E402
import arch20                         # noqa: E402
import arch26                         # noqa: E402
import exp45 as X                     # noqa: E402
from torch.nn.functional import cross_entropy   # noqa: E402

L0 = MDIR / "exp45" / "e39" / "L0"
LIVES = L0 / "lives"
SPEC = ROOT / "PROJECT_DOCUMENTS" / "COORD_CHECKS_20261115" / "sw_world_spec.py"
DEFAULT_BRANCH = "origin/codex/minifly-consolidated-20261023"
REVIEWER_COMMIT, REVIEWER_PATH = "b622775f", "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_world.py"
AUDIT_DIGESTS = {   # the brief's check 1 (prefixes) and in full from OUTPUT_SW_WORLD_MAC.txt and the audit's R1
    1: ("05e4888194e7b2fcbe83d7aba98a9e31cb2ca0c7d327c977daa12490e3c898b3", "ef5afac92b5ef98d7e3d63eb7bf53ca10198cb0811ab2662188f553a0d6272f2"),
    2: ("3d7ba088a0b76b0278571d67cba55b0b1ba9d3233b10a82cfac310b6ecc2a998", "983ff313b0fb88a4ccc6bc01d609733ed8d5f8981f9e56fe256e24ffaad44ce2"),
    3: ("ea204b199ba7ef450189571d881dae7b87980d8528f0fbde7a1a5a751adb99e1", "dfc3270f241638e0bed82db04da8daaf3692c1bd2eb6aba2439b827c258a0efc")}
BRIEF_PREFIXES = {1: ("05e48881", "ef5afac9"), 2: ("3d7ba088", "983ff313"), 3: ("ea204b19", "dfc3270f")}
MAC_FIGURES = {1: (189, (273, 279, 283), 758), 2: (211, (289, 264, 260), 777), 3: (192, (309, 260, 263), 922)}

JOBS = [  # (directory, arguments of sw_run.py life)
    ("c4", ["naive", "1", "4"]), ("c4", ["sweep", "1", "4"]), ("c4", ["replay", "1", "4", "--rho", "0.5"]),
    ("c4", ["sweep+replay", "1", "4", "--rho", "0.5"]),
    ("c78", ["naive", "1", "16"]), ("c78", ["sweep", "1", "16"]),
    ("c9_full", ["naive", "1", "8"]), ("c9_full", ["replay", "1", "8", "--rho", "0.5"]),
    ("c9_split", ["naive", "1", "8", "--stop-after", "4"]), ("c9_split", ["replay", "1", "8", "--rho", "0.5", "--stop-after", "4"]),
    ("c9_again", ["naive", "1", "8"]),
]
RESUMES = [("c9_split", "sw_naive_w1_v8_state_v4.pt"), ("c9_split", "sw_replay_w1_v8_rho0.5_state_v4.pt")]


def pool(cmds, n=3):
    running, done = [], []
    cmds = list(cmds)
    while cmds or running:
        while cmds and len(running) < n:
            c, log = cmds.pop(0)
            running.append((subprocess.Popen(c, stdout=open(log, "w"), stderr=subprocess.STDOUT), c, log))
        time.sleep(1)
        for r in list(running):
            if r[0].poll() is not None:
                running.remove(r); done.append((r[1], r[0].returncode, r[2]))
                assert r[0].returncode == 0, f"failed: {' '.join(r[1])} (log {r[2]})"
    return done


def lives():
    LIVES.mkdir(parents=True, exist_ok=True)
    py = [sys.executable, str(HERE / "sw_run.py"), "life"]
    cmds = []
    for d, args in JOBS:
        (LIVES / d).mkdir(exist_ok=True)
        cmds.append((py + args + ["--out", str(LIVES / d)], str(LIVES / d / ("log_" + "_".join(a.lstrip("-") for a in args) + ".txt"))))
    pool(cmds)
    pool([(py + ["--resume", str(LIVES / d / s), "--out", str(LIVES / d)], str(LIVES / d / f"log_resume_{s}.txt")) for d, s in RESUMES])
    print("check lives done")


def rec(d, stem):
    return json.loads((LIVES / d / f"{stem}.json").read_text())


def reviewer_module():
    tmp = Path(tempfile.mkdtemp(prefix="e39_reviewer_"))
    src = subprocess.run(["git", "show", f"{REVIEWER_COMMIT}:{REVIEWER_PATH}"], cwd=ROOT, capture_output=True, check=True).stdout
    p = tmp / "e39_world.py"; p.write_bytes(src)
    spec = importlib.util.spec_from_file_location("e39_world_reviewer", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod, dict(commit=REVIEWER_COMMIT, path=REVIEWER_PATH, sha256=hashlib.sha256(src).hexdigest(), file=str(p))


# ---------------------------------------------------------------- check 1
def check1(rv):
    here = (HERE / "sw_world.py").read_bytes()
    spec_wt = SPEC.read_bytes()
    spec_d = subprocess.run(["git", "show", f"{DEFAULT_BRANCH}:PROJECT_DOCUMENTS/COORD_CHECKS_20261115/sw_world_spec.py"], cwd=ROOT,
                            capture_output=True, check=True).stdout
    rows, ok = [], here == spec_wt == spec_d
    for s in (1, 2, 3):
        w = W.World(s)
        d16, d64 = W.digest(w, 16), W.digest(w, 64)
        r16, r64 = rv.SpecWorld(s).digest(16), rv.SpecWorld(s).digest(64)
        sch = w.schedule(1024); ab = [x["absence"] for x in sch if x["absence"] is not None]
        fig = (sum(x["new"] for x in sch), tuple(sum(1 for d in ab if lo <= d <= hi) for lo, hi in ((1, 4), (5, 32), (33, 10 ** 9))), max(ab))
        good = (d16, d64) == AUDIT_DIGESTS[s] and d16.startswith(BRIEF_PREFIXES[s][0]) and d64.startswith(BRIEF_PREFIXES[s][1]) \
            and (r16, r64) == (d16, d64) and fig == MAC_FIGURES[s]
        ok &= good
        rows.append(dict(world=s, digest16=d16, digest64=d64, audit=AUDIT_DIGESTS[s], reviewer_reimplementation_here=(r16, r64),
                         figures_1024=dict(new_groups=fig[0], returns=fig[1], longest_absence=fig[2]), mac_figures=MAC_FIGURES[s], equal=good))
    # probe reserve: no 1,024-visit life in worlds 1-8 meets 250 groups; the schedule never names a probe group
    probe = []
    for s in range(1, 9):
        sch = W.World(s).schedule(1024); mx = max(x["group"] for x in sch)
        probe.append(dict(world=s, groups_met=sum(x["new"] for x in sch), largest_group=mx)); ok &= mx < len(W.World(s).groups) - W.PROBE_GROUPS
    # the runner's branch drawing (sw_run.batches_with) reproduces the file's visit batches under the file's generator
    w = W.World(1); sch = w.schedule(16); same = True
    for s in sch[:3] + sch[-2:]:
        a = list(w.visit_batches(s["visit"], s["group"])); b = list(R.batches_with(w, Random(f"world|1|visit|{s['visit']}"), s["group"]))
        same &= a == b
    ok &= same
    return dict(passed=bool(ok), file_identical=dict(working_tree_spec=here == spec_wt, default_branch_spec=here == spec_d,
                                                     sha256=hashlib.sha256(here).hexdigest()),
                worlds=rows, probe_reserve=probe, branch_drawing_equals_file=same,
                note="Both digest columns are the audit's in full (prefixes as in the brief); the reviewer's SpecWorld, run on this host, "
                     "gives the same six; the 1,024-visit figures equal OUTPUT_SW_WORLD_MAC.txt.")


# ---------------------------------------------------------------- check 2
def check2():
    ok, rows = True, []
    for s in range(1, 9):
        w = W.World(s)
        disj = all(set(t["taught"]).isdisjoint(t["asked"]) for t in w.tasks)
        cover = all(set(t["taught"]) | set(t["asked"]) == {(a, b) for a in range(t["p"]) for b in range(t["p"])} for t in w.tasks)
        sizes = all(len(t["taught"]) == round(t["p"] ** 2 / 2) for t in w.tasks)
        rows.append(dict(world=s, disjoint=disj, cover_all_pairs=cover, taught_is_round_half=sizes)); ok &= disj and cover and sizes
    # the runner's assertion is live: a batch with one asked item in place of a taught one is refused
    life = R.Life("naive", 1, 1)
    taught, asked = next(life.world.visit_batches(0, 0))
    bad = [asked[0]] + taught[1:]
    try:
        life.update(0, bad, asked, False, hashlib.sha256(), hashlib.sha256()); refused = False
    except AssertionError:
        refused = True
    ok &= refused
    verified = {}
    for d, stem in (("c78", "sw_naive_w1_v16"), ("c78", "sw_sweep_w1_v16"), ("c9_full", "sw_replay_w1_v8_rho0.5"), ("c4", "sw_sweep+replay_w1_v4_rho0.5")):
        r = rec(d, stem); verified[stem] = dict(learned_items_verified=r["counters"]["learned_items_verified"], updates=r["final"]["updates"])
        ok &= r["counters"]["learned_items_verified"] == 48 * r["final"]["updates"]
    return dict(passed=bool(ok), worlds=rows, runner_refuses_an_asked_item=refused, runner_assertions=verified)


# ---------------------------------------------------------------- check 3
def check3():
    ok, rows = True, []
    for arm in ("naive", "sweep"):
        life = R.Life(arm, 1, 1)
        hs, hl = hashlib.sha256(), hashlib.sha256()
        for i, (taught, asked) in enumerate(life.world.visit_batches(0, 0)):
            if i >= 64:
                break
            if i in (0, 1, 31, 63):
                before = (R.tensors_sha(life.m.state_dict()), R.opt_sha(life.opt))
                N.answer(life.m, [p for _, _, _, p, _ in asked])
                after = (R.tensors_sha(life.m.state_dict()), R.opt_sha(life.opt))
                life.update(0, taught, asked, True, hs, hl)            # the runner's own assertion, too
                moved = (R.tensors_sha(life.m.state_dict()), R.opt_sha(life.opt))
                rows.append(dict(arm=arm, update=i, weights_before=before[0], weights_after_answering=after[0], optimizer_before=before[1],
                                 optimizer_after_answering=after[1], unchanged=before == after, changed_by_the_update=moved[0] != after[0] and moved[1] != after[1]))
                ok &= before == after and moved[0] != after[0] and moved[1] != after[1]
            else:
                life.update(0, taught, asked, False, hs, hl)
    lives_ = {}
    for d, stem in (("c78", "sw_naive_w1_v16"), ("c78", "sw_sweep_w1_v16"), ("c9_full", "sw_replay_w1_v8_rho0.5")):
        r = rec(d, stem); lives_[stem] = r["counters"]["asking_checks"]; ok &= r["counters"]["asking_checks"] == r["final"]["visits"]
    return dict(passed=bool(ok), in_process=rows, runner_asserted_at_first_update_of_every_visit=lives_)


# ---------------------------------------------------------------- check 4
def check4():
    stems = ["sw_naive_w1_v4", "sw_sweep_w1_v4", "sw_replay_w1_v4_rho0.5", "sw_sweep+replay_w1_v4_rho0.5"]
    rs = {s: rec("c4", s) for s in stems}
    gen = W.digest(W.World(1), 4)
    st = {s: r["stream_sha256"] for s, r in rs.items()}
    by_visit = {s: [v["stream_sha256"] for v in r["by_visit"]] for s, r in rs.items()}
    learned = {s: r["learned_sha256"] for s, r in rs.items()}
    ok = len(set(st.values())) == 1 and gen in st.values() and len({tuple(v) for v in by_visit.values()}) == 1
    ok &= learned[stems[0]] == learned[stems[1]] and learned[stems[2]] == learned[stems[3]] and learned[stems[0]] != learned[stems[2]]
    ev = {s: np.load(LIVES / "c4" / f"{s}.npz") for s in stems}
    cols = ("update", "visit", "task", "blind", "wrap", "mirror", "seen")
    same_cols = all(np.array_equal(ev[s][k], ev[stems[0]][k]) for s in stems for k in cols)
    ok &= same_cols
    return dict(passed=bool(ok), stream_sha256=st, generator_digest_4_visits=gen, learned_sha256=learned, arm_independent_event_columns_equal=same_cols,
                note="The stream digest is sw_world.digest's over the 64 stream items of every update; the learned digest is over what each arm "
                     "learned from (equal for naive and sweep, and for the two replay arms; different between them, as it must be).")


# ---------------------------------------------------------------- check 5
def check5():
    m = N.new_net(1)
    taught, _ = next(W.World(1).visit_batches(0, 0))
    x, y, mask = X.batch_tensor([(p, t) for _, _, _, p, t in taught])
    ref = arch20.forward(m, x, False, True)[0]                       # nn.GRU on the same parameters (arch20._forward_rnn)
    out = {}
    for local in (False, True):
        lg = arch26.forward(m, x, local, True)[0]
        out[local] = float((lg - ref).abs().max().detach())

    def grads(fn):
        m.zero_grad()
        cross_entropy(fn()[mask], y[mask]).backward()
        return {n: p.grad.detach().clone() for n, p in m.named_parameters()}
    g_ref = grads(lambda: arch20.forward(m, x, False, True)[0])
    g_bp = grads(lambda: arch26.forward(m, x, False, True)[0])
    g_sw = grads(lambda: arch26.forward(m, x, True, True)[0])
    rows, ok = [], max(out.values()) <= 1e-6
    for n in g_ref:
        scale = float(g_ref[n].abs().max())
        e_bp = float((g_bp[n] - g_ref[n]).abs().max()); e_sw = float((g_sw[n] - g_ref[n]).abs().max())
        rows.append(dict(param=n, ref_max=scale, bp_diff=e_bp, bp_rel=e_bp / scale if scale else 0.0, sweep_diff=e_sw, sweep_rel=e_sw / scale if scale else 0.0))
        ok &= e_bp <= 1e-5 * scale
    differ = [r["param"] for r in rows if r["sweep_rel"] > 1e-3]
    same_sweep = [r["param"] for r in rows if r["sweep_rel"] <= 1e-5]
    ok &= len(differ) >= len(rows) - 2 and set(same_sweep) <= {"l3.weight", "l3.bias"}
    return dict(passed=bool(ok), logits_max_abs_diff=dict(backprop=out[False], sweep_forward=out[True]), tolerance_logits=1e-6, tolerance_grad=1e-5,
                gradients=rows, sweep_differs_on=differ, sweep_equal_on=same_sweep,
                note="Under the sweep the output layer's own gradients are exact (its weight gradient is g^T x); every map below it receives "
                     "the error through weight signs, so every other tensor's gradient differs.")


# ---------------------------------------------------------------- check 6
def check6():
    out, ok = {}, True
    a, b = R.Life("naive", 1, 1), R.Life("replay", 1, 1, rho=0.0)
    s0 = a.world.schedule(1)[0]
    a.visit(s0, a.world.visit_batches(0, 0), max_updates=64); b.visit(s0, b.world.visit_batches(0, 0), max_updates=64)
    wa, wb = R.tensors_sha(a.m.state_dict()), R.tensors_sha(b.m.state_dict())
    out["rho0_vs_naive_64_updates"] = dict(naive=wa, replay_rho0=wb, equal=wa == wb, optimizer_equal=R.opt_sha(a.opt) == R.opt_sha(b.opt),
                                           buffer_items=len(b.buf), first_replay_update=b.counters["first_replay_update"])
    ok &= wa == wb and R.opt_sha(a.opt) == R.opt_sha(b.opt)
    life = R.Life("replay", 1, 1, rho=0.5)
    seen_batches, orig = [], R.N.step

    def spy(m, opt, items, local):
        seen_batches.append(list(items)); return orig(m, opt, items, local)
    R.N.step = spy
    per = []
    try:
        for i, (taught, asked) in enumerate(life.world.visit_batches(0, 0)):
            if i >= 64:
                break
            snap = {id(it) for it in life.buf}; n_before = len(life.buf)
            _, r = life.update(0, taught, asked, False, hashlib.sha256(), hashlib.sha256())
            new = [(p, y) for _, _, _, p, y in taught]; got = seen_batches[-1]
            if n_before >= 24:
                good = r == 24 and got[:24] == new[:24] and all(id(it) in snap for it in got[24:])
            else:
                good = r == 0 and got == new
            per.append(dict(update=i, buffer_before=n_before, replayed=r, good=good))
    finally:
        R.N.step = orig
    ok &= all(p["good"] for p in per)
    out["rho0.5_first_64_updates"] = dict(replayed_per_update=[p["replayed"] for p in per], buffer_before=[p["buffer_before"] for p in per],
                                          all_as_specified=all(p["good"] for p in per),
                                          rule="24 replayed (drawn from the buffer as it was before the update's offers) and the first 24 new "
                                               "taught items once the buffer holds 24; all 48 new before")
    caps = {}
    for d, stem in (("c9_full", "sw_replay_w1_v8_rho0.5"), ("c4", "sw_sweep+replay_w1_v4_rho0.5")):
        L = rec(d, stem)["ledger"]["buffer"]; r_ = rec(d, stem)
        caps[stem] = dict(L, replayed_items=r_["counters"]["replayed_items"], updates=r_["final"]["updates"],
                          first_replay_update=r_["counters"]["first_replay_update"])
        ok &= L["max_bytes"] <= L["capacity_bytes"] and L["max_items"] == L["capacity_items"] == 14033 and L["offered"] > L["capacity_items"]
        ok &= r_["counters"]["replayed_items"] == 24 * (r_["final"]["updates"] - 1) and r_["counters"]["first_replay_update"] == 1
    out["capacity"] = caps
    out["capacity_rule"] = "capacity (4 * 52,625) // 15 = 14,033 items, 210,500 bytes; asserted at every update of every replay life"
    return dict(passed=bool(ok), **out)


# ---------------------------------------------------------------- check 7: hand-made files
def hand_two_tasks():
    """Two tasks (3 symmetric, 7 not), one group visited at 0, 1, 6, 45; 8 updates a visit, 8 asked items per task per update."""
    A, B = 3, 7
    ev = {k: [] for k in ("update", "visit", "task", "correct", "blind", "wrap", "mirror", "seen", "replayed")}
    j = 0
    for v in (0, 1, 6, 45):
        for i in range(8):
            u = 8 * v + i
            for t in (A, B):
                for k in range(8):
                    if t == A:
                        right = {0: i >= 3, 1: (i > 0 or k in (1, 3, 5, 6, 7)), 6: i >= 2, 45: False}[v]
                    else:
                        right = {0: False, 1: i >= 1, 6: True, 45: True}[v]
                    ev["update"].append(u); ev["visit"].append(v); ev["task"].append(t); ev["correct"].append(int(right))
                    ev["blind"].append(int(t == A and k == 0 and j > 0)); ev["wrap"].append(k % 2)
                    ev["mirror"].append(int(t == A and k < 2)); ev["seen"].append(24 * j); ev["replayed"].append(0)
            j += 1
    ev = {k: np.array(x) for k, x in ev.items()}
    hand = dict(
        score_by_age={0: 40 / 128, 1: 117 / 128, 6: 112 / 128, 45: 64 / 128}, blind_by_age={0: 7 / 128, 1: 8 / 128, 6: 8 / 128, 45: 8 / 128},
        first_visit=[dict(n=1, task=A, visit=0, level=1.0, cost=144), dict(n=2, task=B, visit=0, level=0.0, cost=None)],
        cost=[dict(n=1, task=A, cost=144, level=None), dict(n=2, task=B, cost=288, level=None)],
        kept={(A, 1): (5, 8, 4, 4, 1, 4, 29, 32, 1.0), (A, 6): (0, 8, 0, 4, 0, 4, 16, 32, 1.0), (A, 45): (0, 8, 0, 4, 0, 4, 0, 32, 1.0),
              (B, 1): (0, 8, 0, 4, 0, 4, 24, 32, 0.0), (B, 6): (8, 8, 4, 4, 4, 4, 32, 32, 1.0), (B, 45): (8, 8, 4, 4, 4, 4, 32, 32, 1.0)},
        relearn={(A, 1): 72, (A, 6): 120, (A, 45): None, (B, 1): 96, (B, 6): 72, (B, 45): 72},
        pooled=dict(short=dict(first_update=(5, 16), first_update_wrap=(4, 8), first_update_nowrap=(1, 8), first32=(53, 64), first32_wrap=(28, 32),
                               first32_nowrap=(25, 32), relearn_reached=2, relearn_median_items=84.0),
                    medium=dict(first_update=(8, 16), first_update_wrap=(4, 8), first_update_nowrap=(4, 8), first32=(48, 64), first32_wrap=(24, 32),
                                first32_nowrap=(24, 32), relearn_reached=2, relearn_median_items=96.0),
                    long=dict(first_update=(8, 16), first_update_wrap=(4, 8), first_update_nowrap=(4, 8), first32=(32, 64), first32_wrap=(16, 32),
                              first32_nowrap=(16, 32), relearn_reached=1, relearn_median_items=72.0)),
        symmetric=dict(all=(149, 256), mirror=(37, 64), no_mirror=(112, 192), non_symmetric=(184, 256), blind_mirror=(31, 64)),
        late_level=0.5, score_trend=None, learning_trend=None, cost_trend=None, mean_score=333 / 512,
        end_of_visit={(A, 0): 1.0, (A, 1): 1.0, (A, 6): 1.0, (A, 45): 0.0, (B, 0): 0.0, (B, 1): 1.0, (B, 6): 1.0, (B, 45): 1.0})
    return ev, 46, {A}, hand


def hand_four_tasks():
    """Four tasks met one per visit at 0-3 and returning at 4-7; 4 updates a visit, 8 asked items an update, 24 taught per update.
    First visit: the last r right; return: the first w wrong."""
    r_, w_ = [8, 16, 20, 24], [0, 4, 2, 8]
    ev = {k: [] for k in ("update", "visit", "task", "correct", "blind", "wrap", "mirror", "seen", "replayed")}
    for v in range(8):
        t = v % 4
        for i in range(4):
            for k in range(8):
                idx = 8 * i + k
                right = idx >= 32 - r_[t] if v < 4 else idx >= w_[t]
                ev["update"].append(4 * v + i); ev["visit"].append(v); ev["task"].append(t); ev["correct"].append(int(right))
                ev["blind"].append(0); ev["wrap"].append(0); ev["mirror"].append(0); ev["seen"].append(24 * (i + (4 if v >= 4 else 0))); ev["replayed"].append(0)
    ev = {k: np.array(x) for k, x in ev.items()}
    sba = [0.25, 0.5, 0.625, 0.75, 1.0, 28 / 32, 30 / 32, 24 / 32]
    hand = dict(score_by_age=dict(enumerate(sba)),
                first_visit=[dict(n=t + 1, task=t, visit=t, level=r_[t] / 32, cost=None) for t in range(4)],
                cost=[dict(n=1, task=0, cost=144, level=None), dict(n=2, task=1, cost=120, level=None), dict(n=3, task=2, cost=96, level=None),
                      dict(n=4, task=3, cost=None, level=0.75)],
                relearn={(0, 4): 72, (1, 5): 72, (2, 6): 72, (3, 7): None},
                kept_first_update={(0, 4): (8, 8), (1, 5): (4, 8), (2, 6): (6, 8), (3, 7): (0, 8)},
                learning_trend=(0.75 - 0.625) / (math.log(4) - math.log(3)),
                cost_trend=(math.log(96) - math.log(120)) / (math.log(3) - math.log(2)),
                score_trend=float(np.polyfit(np.log([5, 6, 7, 8]), sba[4:], 1)[0]),
                late_level=(30 / 32 + 24 / 32) / 2)
    return ev, 8, set(), hand


def close(a, b, tol=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


def check_hand():
    fails = []
    ev, V, sym, h = hand_two_tasks()
    r = S.score(ev, V, sym)
    for v, x in h["score_by_age"].items():
        if not close(r["score_by_age"][v], x): fails.append(("two", "score_by_age", v))
    for v, x in h["blind_by_age"].items():
        if not close(r["blind_by_age"][v], x): fails.append(("two", "blind_by_age", v))
    if [x for i, x in enumerate(r["score_by_age"]) if i not in h["score_by_age"]] != [None] * (V - 4): fails.append(("two", "empty visits"))
    if r["first_visit"] != h["first_visit"]: fails.append(("two", "first_visit", r["first_visit"]))
    if r["cost"] != h["cost"]: fails.append(("two", "cost", r["cost"]))
    for k in r["kept"]:
        e = h["kept"][(k["task"], k["visit"])]
        got = (k["first_update"]["right"], k["first_update"]["n"], k["first_update_wrap"]["right"], k["first_update_wrap"]["n"],
               k["first_update_nowrap"]["right"], k["first_update_nowrap"]["n"], k["first32"]["right"], k["first32"]["n"], k["before"])
        if got != e: fails.append(("two", "kept", k["task"], k["visit"], got, e))
    if len(r["kept"]) != 6: fails.append(("two", "kept count"))
    for x in r["relearn"]:
        if x["items"] != h["relearn"][(x["task"], x["visit"])]: fails.append(("two", "relearn", x))
    for cls, e in h["pooled"].items():
        p = r["kept_pooled"][cls]
        for part, val in e.items():
            got = p[part] if part.startswith("relearn") else (p[part]["right"], p[part]["n"])
            if got != val: fails.append(("two", "pooled", cls, part, got, val))
    for part, val in h["symmetric"].items():
        got = (r["symmetric"][part]["right"], r["symmetric"][part]["n"])
        if got != val: fails.append(("two", "symmetric", part, got, val))
    for q in ("late_level", "score_trend", "learning_trend", "cost_trend", "mean_score"):
        if not close(r[q], h[q]): fails.append(("two", q, r[q], h[q]))
    for x in r["end_of_visit"]:
        if not close(x["level"], h["end_of_visit"][(x["task"], x["visit"])]): fails.append(("two", "end_of_visit", x))
    ev, V, sym, h = hand_four_tasks()
    r = S.score(ev, V, sym)
    for v, x in h["score_by_age"].items():
        if not close(r["score_by_age"][v], x): fails.append(("four", "score_by_age", v))
    if r["first_visit"] != h["first_visit"]: fails.append(("four", "first_visit", r["first_visit"]))
    if r["cost"] != h["cost"]: fails.append(("four", "cost", r["cost"]))
    for x in r["relearn"]:
        if x["items"] != h["relearn"][(x["task"], x["visit"])]: fails.append(("four", "relearn", x))
    for k in r["kept"]:
        if (k["first_update"]["right"], k["first_update"]["n"]) != h["kept_first_update"][(k["task"], k["visit"])]: fails.append(("four", "kept", k))
    for q in ("learning_trend", "cost_trend", "score_trend", "late_level"):
        if not close(r[q], h[q], 1e-9): fails.append(("four", q, r[q], h[q]))
    return fails, dict(two_tasks=dict(learning_trend=None, cost_trend=None, score_trend=None),
                       four_tasks=dict(learning_trend=h["learning_trend"], cost_trend=h["cost_trend"], score_trend=h["score_trend"]))


def compare_reviewer(rv, path, world, visits):
    """The reviewer's numpy scorer and this one on the same event file, quantity by quantity."""
    f = np.load(path); ev = {k: f[k] for k in f.files}
    mine = S.score(ev, visits, S.symmetric_tasks(world))
    theirs = rv.score(ev, visits)
    diffs = {}

    def nn(x):
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
    for q in ("score_by_age", "blind_by_age"):
        diffs[q] = int(sum(not close(a, nn(b)) for a, b in zip(mine[q], theirs[q].tolist())))
    diffs["first_visit"] = int(sum((a["n"], a["task"], a["cost"]) != (b["n"], b["task"], b["cost"]) or not close(a["level"], b["level"])
                                   for a, b in zip(mine["first_visit"], theirs["first_visit"]))) + abs(len(mine["first_visit"]) - len(theirs["first_visit"]))
    diffs["cost"] = int(sum((a["n"], a["task"], a["cost"]) != (b["n"], b["task"], b["cost"]) or not close(a["level"], b["level"])
                            for a, b in zip(mine["cost"], theirs["cost"])))
    tk = {(k["task"], k["visit"]): k for k in theirs["kept"]}
    kd = 0
    for k in mine["kept"]:
        o = tk[(k["task"], k["visit"])]
        pairs = [("first_update", "kept"), ("first_update_wrap", "kept_wrap"), ("first_update_nowrap", "kept_nowrap"), ("first32", "kept32"),
                 ("first32_wrap", "kept32_wrap"), ("first32_nowrap", "kept32_nowrap")]
        kd += int(any(not close(k[a]["share"], o[b]) for a, b in pairs) or k["absence"] != o["absence"])
    diffs["kept"] = kd + abs(len(mine["kept"]) - len(theirs["kept"]))
    tr = {(x["task"], x["visit"]): x["items"] for x in theirs["relearn"]}
    diffs["relearn"] = int(sum(x["items"] != tr[(x["task"], x["visit"])] for x in mine["relearn"]))
    for q in ("late_level", "score_trend", "learning_trend", "cost_trend"):
        diffs[q] = 0 if close(mine[q], nn(theirs[q]), 1e-9) else 1
    return dict(file=Path(path).name, visits=visits, entries=mine["entries"], disagreements=diffs, all_agree=all(v == 0 for v in diffs.values()),
                compared=dict(returns=len(mine["kept"]), tasks=len(mine["first_visit"]),
                              values=dict(late_level=mine["late_level"], score_trend=mine["score_trend"], learning_trend=mine["learning_trend"],
                                          cost_trend=mine["cost_trend"])))


def check7(rv, rv_info):
    fails, trends = check_hand()
    self_test = subprocess.run([sys.executable, "-I", rv_info["file"], "text", "scorer"], capture_output=True, text=True)
    ok = not fails and self_test.returncode == 0
    comp = [compare_reviewer(rv, LIVES / "c78" / f"{s}.npz", 1, 16) for s in ("sw_naive_w1_v16", "sw_sweep_w1_v16")]
    ok &= all(c["all_agree"] for c in comp)
    rebuild = {}
    for s in ("sw_naive_w1_v16", "sw_sweep_w1_v16"):
        p = subprocess.run([sys.executable, "-I", rv_info["file"], "spec", "events", str(LIVES / "c78" / f"{s}.npz"), "1"], capture_output=True, text=True)
        lines = [l.strip() for l in p.stdout.splitlines() if l.strip()]
        rebuild[s] = dict(returncode=p.returncode, output=lines)
        ok &= p.returncode == 0 and len(lines) == 7 and all(l.endswith("equal") and "DIFFERENT" not in l for l in lines)
    return dict(passed=bool(ok), hand_made=dict(failures=[str(f) for f in fails], files=["two tasks, visits 0, 1, 6, 45 (every quantity)",
                                                                                    "four tasks met one per visit (the three trends, cost not reached)"],
                                                trends=trends),
                reviewer=dict(rv_info, self_test=self_test.stdout.strip()), scorer_agreement=comp, reviewer_rebuild_of_event_file=rebuild)


# ---------------------------------------------------------------- check 8
def check8():
    rows = {}
    for d, stem in (("c78", "sw_naive_w1_v16"), ("c78", "sw_sweep_w1_v16"), ("c9_full", "sw_replay_w1_v8_rho0.5"), ("c4", "sw_sweep+replay_w1_v4_rho0.5")):
        r = rec(d, stem); pv = r["cpu"]["per_visit"]
        rows[stem] = dict(visits=len(pv), cpu_per_visit_mean=float(np.mean(pv)), min=float(np.min(pv)), max=float(np.max(pv)),
                          forecast_1024_visits_cpu_hours=float(np.mean(pv)) * 1024 / 3600, total_cpu=r["cpu"]["total"])
    ok = all(v["forecast_1024_visits_cpu_hours"] <= 6 for k, v in rows.items() if "naive" in k or "sweep_w1" in k)
    return dict(passed=bool(ok), lives=rows, host=R.host(), bound_cpu_hours=6,
                note="CPU-seconds per visit (time.process_time per process, one thread) at the default size, 256 updates a visit; the check "
                     "lives ran three processes at a time on a 4-core container. The forecast is the mean times 1,024.")


# ---------------------------------------------------------------- check 9
def check9():
    out, ok = {}, True
    for stem in ("sw_naive_w1_v8", "sw_replay_w1_v8_rho0.5"):
        a, b = rec("c9_full", stem), rec("c9_split", stem)
        ea, eb = np.load(LIVES / "c9_full" / f"{stem}.npz"), np.load(LIVES / "c9_split" / f"{stem}.npz")
        same_ev = all(np.array_equal(ea[k], eb[k]) for k in ea.files) and sorted(ea.files) == sorted(eb.files)
        row = dict(uninterrupted=a["final"], resumed=b["final"], events_sha_uninterrupted=a["events"]["arrays_sha256"],
                   events_sha_resumed=b["events"]["arrays_sha256"], events_equal=same_ev, resumes=b["resumes"],
                   stream_equal=a["stream_sha256"] == b["stream_sha256"], learned_equal=a["learned_sha256"] == b["learned_sha256"],
                   buffer_equal=a["ledger"]["buffer"] == b["ledger"]["buffer"])
        good = a["final"]["weights_tensors_sha256"] == b["final"]["weights_tensors_sha256"] and a["final"]["optimizer_sha256"] == b["final"]["optimizer_sha256"] \
            and same_ev and a["events"]["arrays_sha256"] == b["events"]["arrays_sha256"] and row["stream_equal"] and row["learned_equal"] \
            and len(b["resumes"]) == 1 and b["resumes"][0]["done_visits"] == 4 and row["buffer_equal"]
        row["passed"] = good; ok &= good; out[stem] = row
    a, c = rec("c9_full", "sw_naive_w1_v8"), rec("c9_again", "sw_naive_w1_v8")
    twice = a["final"] == c["final"] and a["events"]["arrays_sha256"] == c["events"]["arrays_sha256"] and a["stream_sha256"] == c["stream_sha256"]
    ok &= twice
    # prefixes: the state of visit 4 of the 8-visit lives equals the end of the 4-visit lives of check 4
    pref = {}
    for arm, s8, s4 in (("naive", "sw_naive_w1_v8", "sw_naive_w1_v4"), ("replay", "sw_replay_w1_v8_rho0.5", "sw_replay_w1_v4_rho0.5")):
        st = [x for x in rec("c9_full", s8)["states"] if x["visits"] == 4][0]
        f4 = rec("c4", s4)["final"]
        pref[arm] = st["weights_tensors_sha256"] == f4["weights_tensors_sha256"] and st["optimizer_sha256"] == f4["optimizer_sha256"]
    ok &= all(pref.values())
    return dict(passed=bool(ok), resume=out, run_twice=dict(equal=twice, first=a["final"], second=c["final"]), prefix_state4_equals_life4=pref)


def assemble():
    path = L0 / "E39_L0_CHECKS.json"
    R.refuse_existing(path)
    rv, rv_info = reviewer_module()
    t0 = time.process_time()
    checks = {}
    for name, fn in (("1_generator", lambda: check1(rv)), ("2_asked_never_taught", check2), ("3_asking_does_not_teach", check3),
                     ("4_same_stream", check4), ("5_forward_and_rules", check5), ("6_replay", check6), ("7_scorer", lambda: check7(rv, rv_info)),
                     ("8_cost", check8), ("9_resume", check9)):
        c0 = time.process_time()
        checks[name] = fn(); checks[name]["cpu"] = time.process_time() - c0
        print(name, "PASS" if checks[name]["passed"] else "FAIL", flush=True)
    out = dict(experiment="E39", job="L0", brief="OPUS_TASK_E39_SMALL_WORLD_PROTOCOL_L0_L1_20261115.md, revision 2", checks=checks,
               all_passed=all(c["passed"] for c in checks.values()), host=R.host(), git=R.git_state(), sources_sha256=R.source_hashes(),
               checks_source_sha256=R.sha_file(Path(__file__)), cpu=time.process_time() - t0,
               lives=sorted(str(p.relative_to(L0)) for p in LIVES.rglob("*.json")))
    R.write_json(path, out)
    print("all passed" if out["all_passed"] else "SOME CHECK FAILED", flush=True)


if __name__ == "__main__":
    {"lives": lives, "assemble": assemble}[sys.argv[1]]()
