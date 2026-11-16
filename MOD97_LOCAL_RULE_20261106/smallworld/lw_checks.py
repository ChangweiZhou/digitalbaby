"""E44 (brief OPUS_TASK_E44_LONG_LIFE_WORLD_IN_THE_FULL_SCALE_FORMAT_20261116.md, revision 2): the eight checks before the calibration.

    python3 lw_checks.py lives      # the check lives (a 4-visit small-world life; an 8-visit lw life whole, and stopped at 4 then resumed),
                                    #   at most three processes, into exp45/e44/checks/lives/
    python3 lw_checks.py pre        # checks 1 to 7 as far as they need no calibration life -> exp45/e44/checks/E44_CHECKS_PRE.json
    python3 lw_checks.py post       # after the calibration: check 1's 48 receipts, check 6's same stream, check 7 on a calibration life,
                                    #   check 8 -> exp45/e44/checks/E44_CHECKS.json (with the pre-calibration parts)

The reviewer's files are read from git (REVIEWER_COMMIT) into a temporary directory and run from there with python3 -I; their hashes are
recorded. The coordinator's row-0 script is run from the default branch's tree as it stands in this checkout (its hash recorded).
Nothing here trains except the check lives (through sw_run.py) and the in-process asking check.
"""
import ast
import hashlib
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from random import Random

import numpy as np
import torch as T

HERE = Path(__file__).resolve().parent
MDIR = HERE.parent
ROOT = MDIR.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(MDIR))
import lw_world as LW                 # noqa: E402
import sw_world as W                  # noqa: E402
import sw_net as N                    # noqa: E402
import sw_run as R                    # noqa: E402
import sw_score as S                  # noqa: E402

OUT = MDIR / "exp45" / "e44"
CHK = OUT / "checks"
LIVES = CHK / "lives"
F1 = OUT / "F1"
K = ROOT / "PROJECT_DOCUMENTS" / "COORD_CHECKS_20261116"
SPEC, WORLD_OUT, ROW0, ROW0_OUT = K / "lw_world_spec.py", K / "OUTPUT_LW_WORLD_MAC.txt", K / "lw_row0_check.py", K / "OUTPUT_LW_ROW0_MAC.txt"
REVIEWER_COMMIT = "dc05d575"
RDIR = "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/"
AUDIT = RDIR + "AUDIT_E44_LONG_LIFE_WORLD_BRIEF_20261116.md"
BIRTHS = {"B": MDIR / "exp45" / "e41" / "e41_B_h256_f50_73129.pt", "S": MDIR / "exp45" / "e41" / "e41_S_h256_f50_73129.pt"}
LENGTHS = (64, 128, 256)
SP = [sys.executable, str(HERE / "sw_run.py")]

# the check lives: (directory, sw_run.py arguments)
JOBS = [("c5", ["life", "naive", "1", "4"]),
        ("c6_full", ["life", "naive", "1", "8", "--world", "lw", "--learner", str(BIRTHS["B"]), "--in-turn", "--visit-updates", "64",
                     "--save-states", "4,8"]),
        ("c6_split", ["life", "naive", "1", "8", "--world", "lw", "--learner", str(BIRTHS["B"]), "--in-turn", "--visit-updates", "64",
                      "--save-states", "4,8", "--stop-after", "4"])]
C6 = "lw_naive_w1_v8_vu64_turn"


def git_show(path, ref=REVIEWER_COMMIT):
    return subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def pool(cmds, n=3):
    running, cmds = [], list(cmds)
    while cmds or running:
        while cmds and len(running) < n:
            d, args, log = cmds.pop(0)
            Path(d).mkdir(parents=True, exist_ok=True)
            running.append((subprocess.Popen(SP + args + ["--out", str(d)], stdout=open(log, "w"), stderr=subprocess.STDOUT), args, log))
        for r in list(running):
            if r[0].poll() is not None:
                running.remove(r)
                assert r[0].returncode == 0, f"failed: {r[1]} (log {r[2]})"
                print("done", r[2], flush=True)
        __import__("time").sleep(1)


def lives():
    pool([(LIVES / d, a, LIVES / d / f"log_{i}.txt") for i, (d, a) in enumerate(JOBS)])
    st = LIVES / "c6_split" / f"{C6}_state_v4.pt"
    pool([(LIVES / "c6_split", ["life", "--resume", str(st)], LIVES / "c6_split" / "log_resume.txt")])


# ---------------------------------------------------------------- the audit's section 7, parsed from the audit itself
def audit_tables():
    txt = git_show(AUDIT).decode()
    sec = txt.split("## 7. Figures for the worker to compare")[1].split("## 8.")[0]
    rows = {}
    for ln in sec.splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) >= 5 and cells[0].isdigit() and all(len(c) == 16 for c in cells[1:5]):
            rows.setdefault(int(cells[0]), []).append(cells[1:])
    t1 = {w: dict(zip(("in_turn_64", "in_turn_128", "in_turn_256", "schedule_1024", "tasks"), r[0])) for w, r in rows.items()}
    t2 = {w: dict(zip(("lap2_64", "lap2_128", "lap2_256", "list_0_7"), r[1])) for w, r in rows.items()}
    fx = {}
    for vu in (64, 256):
        ln = next(l for l in sec.splitlines() if l.startswith(f"| {vu} |") and "/" in l)
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        fx[vu] = dict(columns=[x.strip() for x in cells[1].split(",")], medians=[float(x) for x in cells[2].split("/")])
    return t1, t2, fx, sha(txt.encode())


def schedule_sha(w, visits):
    return sha(bytes(s["group"] for s in w.schedule(visits)))


def tasks_sha(w):
    """The audit's: for each task in id order, its two code bytes, the modulus, the symbol, the number of taught pairs as two bytes, then
    every taught pair and every asked pair as the bytes a, b in the split's order."""
    h = hashlib.sha256()
    for t in w.tasks:
        h.update(bytes(t["code"]) + bytes([t["p"], t["symbol"], len(t["taught"]) // 256, len(t["taught"]) % 256]))
        for a, b in t["taught"] + t["asked"]:
            h.update(bytes([a, b]))
    return h.hexdigest()


def branch_digest(world, n0, groups):
    h = hashlib.sha256()
    for k, g in enumerate(groups):
        for taught, asked in R.batches_with(world, Random(f"world|{world.seed}|branch|{n0}|{k}"), g):
            for it in taught + asked:
                h.update(R.stream_bytes(*it))
    return h.hexdigest()


# ---------------------------------------------------------------- check 1
def check1_pre():
    t1, t2, _, audit_sha = audit_tables()
    out = dict(spec_identical=SPEC.read_bytes() == (HERE / "lw_world.py").read_bytes(), lw_world_sha256=sha((HERE / "lw_world.py").read_bytes()),
               spec_sha256=sha(SPEC.read_bytes()))
    mine = "".join(subprocess.run([sys.executable, "-I", str(HERE / "lw_world.py"), str(s)], capture_output=True, text=True, check=True).stdout
                   for s in (1, 2, 3))
    out["output_equals_OUTPUT_LW_WORLD_MAC"] = mine == WORLD_OUT.read_text()
    with tempfile.TemporaryDirectory() as td:
        rp = Path(td) / "e44_world_reader.py"; rp.write_bytes(git_show(RDIR + "e44_world_reader.py"))
        theirs = subprocess.run([sys.executable, "-I", str(rp), "1", "2", "3"], capture_output=True, text=True, check=True, cwd=td).stdout
        out["reviewer_reader"] = dict(commit=REVIEWER_COMMIT, path=RDIR + "e44_world_reader.py", sha256=sha(rp.read_bytes()),
                                      output_equals_OUTPUT_LW_WORLD_MAC=theirs == WORLD_OUT.read_text())
    full, cur = {}, None
    for ln in WORLD_OUT.read_text().splitlines():
        if ln.startswith("world "):
            cur = int(ln.split()[1].rstrip(":")); full[cur] = {}
        elif "digest of the first 4 visits:" in ln:
            full[cur]["first4"] = ln.split(": ")[1].strip()
        elif "digest of 8 visits in turn at 64 updates:" in ln:
            full[cur]["in_turn_64"] = ln.split(": ")[1].strip()
    rows, ok = {}, out["spec_identical"] and out["output_equals_OUTPUT_LW_WORLD_MAC"] and out["reviewer_reader"]["output_equals_OUTPUT_LW_WORLD_MAC"]
    for s in range(1, 9):
        r = {}
        for vu in LENGTHS:
            r[f"in_turn_{vu}"] = LW.digest(LW.World(s, visit_updates=vu), 8, in_turn=True)
            r[f"lap2_{vu}"] = branch_digest(LW.World(s, visit_updates=vu), 8, list(range(8)))
        w = LW.World(s)
        r["schedule_1024"] = schedule_sha(w, 1024)
        r["tasks"] = tasks_sha(w)
        r["list_0_7"] = R.list_digest([it for f in range(8) for it in R.probe_items(w, f)])
        if s <= 3:
            r["first4"] = LW.digest(w, 4)
            r["full_digests_equal_output"] = r["first4"] == full[s]["first4"] and r["in_turn_64"] == full[s]["in_turn_64"]
            ok &= r["full_digests_equal_output"]
        want = dict(t1[s], **t2[s])
        r["equal_to_audit"] = {k: r[k][:16] == v for k, v in want.items()}
        ok &= all(r["equal_to_audit"].values())
        rows[s] = r
    out.update(passed=bool(ok), worlds=rows, audit=dict(commit=REVIEWER_COMMIT, path=AUDIT, sha256=audit_sha),
               note="The 48 receipts' stream digests are compared after the calibration (check 1, post).")
    return out


def check1_post(receipts):
    t1, t2, _, _ = audit_tables()
    rows, ok = [], True
    for r in receipts:
        w, vu = r["settings"]["world"], r["settings"]["visit_updates"]
        want = t1[w][f"in_turn_{vu}"]
        e = r["stream_sha256"][:16] == want and r["stream_sha256"] == LW.digest(LW.World(w, visit_updates=vu), 8, in_turn=True)
        rows.append(dict(life=r["events"]["file"], world=w, visit_updates=vu, stream_sha256=r["stream_sha256"], audit=want, equal=e))
        ok &= e
    return dict(passed=bool(ok and len(rows) == 48), lives=len(rows), rows=rows)


def check1_lap2():
    """The second lap's receipts: stream digests and list digests against the audit's section 7 at the adopted length."""
    _, t2, _, _ = audit_tables()
    rows, ok = [], True
    for p in sorted((F1 / "lap2").glob("*_br8_*.json")):
        r = json.loads(p.read_text()); w, vu = r["settings"]["world"], r["settings"]["visit_updates"]
        e = r["stream_sha256"][:16] == t2[w][f"lap2_{vu}"] and r["list"]["items_sha256"][:16] == t2[w]["list_0_7"]
        rows.append(dict(branch=p.name, world=w, visit_updates=vu, stream_sha256=r["stream_sha256"], audit=t2[w][f"lap2_{vu}"],
                         list_items_sha256=r["list"]["items_sha256"], audit_list=t2[w]["list_0_7"], equal=e))
        ok &= e
    return dict(passed=bool(ok and len(rows) == 16), branches=len(rows), rows=rows)


# ---------------------------------------------------------------- check 2
def check2():
    out, ok = {}, True
    used = {73, 83, 89, 97} | set(range(8, 38))
    for s in range(1, 9):
        w = LW.World(s)
        mods = sorted({t["p"] for t in w.tasks}); code_bytes = sorted({b for t in w.tasks for b in t["code"]})
        dis = all(set(t["taught"]).isdisjoint(t["asked"]) for t in w.tasks) and all(len(t["taught"]) + len(t["asked"]) == t["p"] ** 2 for t in w.tasks)
        good = dis and not (set(mods) & used) and min(code_bytes) >= 128 and len({t["code"] for t in w.tasks}) == len(w.tasks)
        out[s] = dict(moduli=len(mods), moduli_shared_with_used=sorted(set(mods) & used), code_bytes_min=min(code_bytes), code_bytes_max=max(code_bytes),
                      codes_distinct=len({t["code"] for t in w.tasks}) == len(w.tasks), taught_asked_disjoint_and_complete=dis, passed=good)
        ok &= good
    sw_codes = sorted({b for t in W.World(1).tasks for b in t["code"]})
    import arch26
    dev_codes = sorted({b for t in arch26.E.DEV for b in t[2]})
    ok &= max(sw_codes) < 128 and max(dev_codes) < 128
    # the runner's own assertion: an asked pair offered as taught is refused
    life = R.Life("naive", 1, 1, H=256, U=128, visit_updates=64, p_new=LW.P_NEW, world_kind="lw", learner=str(BIRTHS["B"]), in_turn=True)
    taught, asked = next(life.world.visit_batches(0, 0))
    bad = list(taught); bad[0] = asked[0]
    import hashlib as _h
    try:
        life.update(0, bad, asked, False, _h.sha256(), _h.sha256()); refused = False
    except AssertionError:
        refused = True
    ok &= refused
    lives_ = {}
    for d in ("c6_full", "c6_split"):
        r = json.loads((LIVES / d / f"{C6}.json").read_text())
        lives_[d] = dict(learned_items_verified=r["counters"]["learned_items_verified"], updates=r["final"]["updates"],
                         all_verified=r["counters"]["learned_items_verified"] == 48 * r["final"]["updates"])
        ok &= lives_[d]["all_verified"]
    return dict(passed=bool(ok), worlds=out, small_world_code_bytes_max=max(sw_codes), birth_family_code_bytes=dev_codes,
                asked_pair_offered_as_taught_refused=refused, runner_verified_every_learned_item=lives_,
                note="Moduli used before: 73, 83, 89, 97 (the harness's families) and 8 to 37 (the small world). The file's own assertions run "
                     "when it is run as a script (check 1 runs it so for worlds 1 to 3).")


# ---------------------------------------------------------------- check 3
E41_SIDE = r'''
import json, sys, hashlib
sys.path.insert(0, sys.argv[1])
import exp41_scale as X
import torch as T
X.arch26.install()
out = {}
for rule, path in (("B", sys.argv[2]), ("S", sys.argv[3])):
    rec = json.load(open(path[:-3] + ".json"))
    m = X.NetW(256, 128); m.load_state_dict(T.load(path))
    A, Rs, TA = X.panels(0.5, 73129)
    sc = X.score_all(m, A, Rs, TA)
    last = rec["curve"][-1]
    out[rule] = dict(tensors_sha256=X.R_.tensors_sha(m.state_dict()), receipt_tensors_sha256=rec["tensors_sha256"], A=sc["A"], R=sc["R"],
                     A_counts=sc["A_counts"], R_counts=sc["R_counts"], receipt_A=last["A"], receipt_R=last["R"], receipt_A_counts=last["A_counts"],
                     receipt_R_counts=last["R_counts"], cls=type(m).__module__ + "." + type(m).__name__)
print(json.dumps(out))
'''


def check3():
    e41 = json.loads(subprocess.run([sys.executable, "-c", E41_SIDE, str(MDIR), str(BIRTHS["B"]), str(BIRTHS["S"])], capture_output=True, text=True,
                                    check=True, cwd=str(MDIR)).stdout)
    ok = True
    import arch26
    E = arch26.E
    for rule, path in BIRTHS.items():
        d = e41[rule]
        d["e41_side_equal"] = d["tensors_sha256"] == d["receipt_tensors_sha256"] and d["A_counts"] == d["receipt_A_counts"] and d["R_counts"] == d["receipt_R_counts"]
        arm = "naive" if rule == "B" else "sweep"
        m, info = R.load_learner(path, 256, 128, arm)
        # the runner's network and answering: R on E41's two samples, A on E41's held-out panel at 12 letters
        rit = R.e41_r_items(0.5, 73129, info["reduction_sample_sha256"])
        g = R.answer_all(m, rit)
        rc = {"add89": sum(x == it[4] for x, it in zip(g[:1000], rit[:1000])), "sub89": sum(x == it[4] for x, it in zip(g[1000:], rit[1000:]))}
        ac = {}
        for t in E.DEV:
            held = E.split(t, 0.5, 73129)[1]
            items = E.panel(t, held, 73129, ks=(12,), n=300)[12]
            gg = R.answer_all(m, [(0, 0, 0, p, y) for p, y in items])
            ac[E.NAME(*t)] = sum(x == y for x, (p, y) in zip(gg, items))
        d["runner"] = dict(cls="sw_net.SWNet", tensors_sha256=info["tensors_sha256"], R_counts=rc, A_counts=ac,
                           R=100 * (rc["add89"] + rc["sub89"]) / 2000, A=100 * (ac["add89"] + ac["sub89"]) / 600)
        d["runner_equal"] = info["tensors_sha256"] == d["receipt_tensors_sha256"] and rc == d["receipt_R_counts"] and ac == d["receipt_A_counts"]
        ok &= d["e41_side_equal"] and d["runner_equal"]
    return dict(passed=bool(ok), births=e41,
                note="E41's side: exp41_scale.NetW(256, 128) with the file, in a process that imports exp41_scale first; A and R by E41's own "
                     "panels and scorer (exp41_scale.score_all). The runner's side: sw_net.SWNet(256, 128) with the same file (the runner's "
                     "learner, sw_run.load_learner), answered by sw_net.answer on the same items.")


# ---------------------------------------------------------------- check 4
def row0_runner(rule, world_seed, fams=(0, 1, 2)):
    """lw_row0_check.py's sample and items, answered by the runner's network and sw_net.answer."""
    m, _ = R.load_learner(BIRTHS[rule], 256, 128, "naive" if rule == "B" else "sweep")
    w = LW.World(world_seed); rng = Random(0)
    def items(t, pairs):
        return [(t["id"], a, b, list(t["code"] + f"{a:02d}".encode() + bytes([t["symbol"]]) + f"{b:02d}".encode() + b"="),
                 list(f"{LW.apply(t['op'], t['p'], a, b):02d}".encode()) + [LW.EOS]) for a, b in pairs]
    def pct(its):
        g = R.answer_all(m, its); return 100 * sum(x == it[4] for x, it in zip(g, its)) / len(its)
    lines = []
    for g in fams:
        row = []
        for t in w.groups[g]:
            asked = rng.sample(t["asked"], min(600, len(t["asked"])))
            if t["op"] in ("add", "sub"):
                red = [pr for pr in asked if LW.needs_wrap(t["op"], t["p"], *pr)]; non = [pr for pr in asked if not LW.needs_wrap(t["op"], t["p"], *pr)]
                row.append((t["op"], pct(items(t, red)), pct(items(t, non))))
            else:
                row.append((t["op"], pct(items(t, asked)), None))
        lines.append((g, w.groups[g][0]["p"], row))
    return lines


def parse_row0(text):
    out = {}
    for ln in text.splitlines():
        if not ln.startswith("born "):
            continue
        rule, ws, fam = ln.split()[1], int(ln.split("world ")[1].split(",")[0]), int(ln.split("family ")[1].split(" ")[0])
        vals = {}
        for part in ln.split(": ", 1)[1].split("; "):
            op, rest = part.split(": ", 1)
            if "reduction" in rest:
                r_, n_ = rest.replace("reduction ", "").replace("none ", "").split(", ")
                vals[op] = (float(r_), float(n_))
            else:
                vals[op] = (float(rest), None)
        out[(rule, ws, fam)] = vals
    return out


def check4():
    committed = parse_row0(ROW0_OUT.read_text())
    here = {}
    for rule, ws in (("S", 1), ("B", 1), ("B", 2)):
        txt = subprocess.run([sys.executable, "-I", str(ROW0), rule, str(ws)], capture_output=True, text=True, check=True, cwd=str(ROOT)).stdout
        here.update(parse_row0(txt))
    rows, ok, worst = [], True, 0.0
    for rule in ("S", "B"):
        for ws in (1, 2):
            for g, p, row in row0_runner(rule, ws):
                for op, x, y in row:
                    key = (rule, ws, g)
                    c = committed.get(key, {}).get(op); h = here.get(key, {}).get(op)
                    r = dict(rule=rule, world=ws, family=g, p=p, op=op, runner=(round(x, 2), None if y is None else round(y, 2)),
                             committed=c, coordinator_script_here=h)
                    if c is not None:
                        d = max(abs(x - c[0]), 0 if y is None else abs(y - c[1]))
                        r["max_difference_to_committed"] = round(d, 3); worst = max(worst, d); ok &= d <= 2.0
                    rows.append(r)
    exact_here = all(here[k] == v for k, v in committed.items())
    return dict(passed=bool(ok and len(committed) == 9), lines_committed=len(committed), largest_difference_points=round(worst, 3),
                coordinator_script_on_this_host_equals_committed=exact_here, coordinator_script_sha256=sha(ROW0.read_bytes()), rows=rows,
                note="The runner's scores: sw_run.load_learner's network and sw_net.answer on lw_row0_check.py's items (Random(0), 600 asked "
                     "pairs a task, no letters). The committed output has nine lines (S on world 1; B on worlds 1 and 2); the runner's S on "
                     "world 2 is tabled and compared with nothing.")


# ---------------------------------------------------------------- check 5
def check5():
    l0 = MDIR / "exp45" / "e39" / "L0" / "lives" / "c4"
    a, b = json.loads((l0 / "sw_naive_w1_v4.json").read_text()), json.loads((LIVES / "c5" / "sw_naive_w1_v4.json").read_text())
    ea, eb = np.load(l0 / "sw_naive_w1_v4.npz"), np.load(LIVES / "c5" / "sw_naive_w1_v4.npz")
    nine = [k for k, _, _ in R.EVENT_COLUMNS][:9]
    cols = {k: bool(np.array_equal(ea[k], eb[k])) for k in nine}
    ok = a["stream_sha256"] == b["stream_sha256"] == W.digest(W.World(1), 4) and all(cols.values()) and sorted(ea.files) == sorted(nine) \
        and sorted(eb.files) == sorted(nine + ["g1", "g2", "g3"])
    right = eb["correct"] == 1
    digits = lambda x: (x >= 48) & (x <= 57)
    consistent = bool(np.all(digits(eb["g1"][right]) & digits(eb["g2"][right]) & (eb["g3"][right] == 256)))
    ended = bool(np.all((eb["g1"] != 256) | ((eb["g2"] == 65535) & (eb["g3"] == 65535))) and np.all((eb["g2"] != 256) | (eb["g3"] == 65535)))
    ok &= consistent and ended
    return dict(passed=bool(ok), stream_sha256_L0=a["stream_sha256"], stream_sha256_now=b["stream_sha256"], nine_columns_equal=cols,
                final_weights_equal_L0=a["final"]["weights_tensors_sha256"] == b["final"]["weights_tensors_sha256"],
                columns_now=sorted(eb.files), right_items_have_two_digits_and_the_end=consistent, generation_ended_after_the_end_symbol=ended)


# ---------------------------------------------------------------- check 6
def check6_pre():
    rows, ok = [], True
    for arm in ("naive", "sweep"):
        life = R.Life(arm, 1, 1, H=256, U=128, visit_updates=64, p_new=LW.P_NEW, world_kind="lw",
                      learner=str(BIRTHS["B" if arm == "naive" else "S"]), in_turn=True)
        hs, hl = hashlib.sha256(), hashlib.sha256()
        for i, (taught, asked) in enumerate(life.world.visit_batches(0, 0)):
            if i in (0, 1, 31, 63):
                before = (R.tensors_sha(life.m.state_dict()), R.opt_sha(life.opt))
                N.answer(life.m, [p for _, _, _, p, _ in asked])
                after = (R.tensors_sha(life.m.state_dict()), R.opt_sha(life.opt))
                life.update(0, taught, asked, True, hs, hl)
                moved = (R.tensors_sha(life.m.state_dict()), R.opt_sha(life.opt))
                good = before == after and moved[0] != after[0]
                rows.append(dict(arm=arm, update=i, unchanged_by_answering=before == after, changed_by_the_update=moved[0] != after[0]))
                ok &= good
            else:
                life.update(0, taught, asked, False, hs, hl)
    a, b = json.loads((LIVES / "c6_full" / f"{C6}.json").read_text()), json.loads((LIVES / "c6_split" / f"{C6}.json").read_text())
    ea, eb = np.load(LIVES / "c6_full" / f"{C6}.npz"), np.load(LIVES / "c6_split" / f"{C6}.npz")
    la, lb = np.load(LIVES / "c6_full" / f"{C6}_list.npz"), np.load(LIVES / "c6_split" / f"{C6}_list.npz")
    same_ev = sorted(ea.files) == sorted(eb.files) and all(np.array_equal(ea[k], eb[k]) for k in ea.files)
    same_list = sorted(la.files) == sorted(lb.files) and all(np.array_equal(la[k], lb[k]) for k in la.files)
    st4 = [x for x in a["states"] if x["visits"] == 4][0]
    resume = dict(final_equal=a["final"] == b["final"], events_equal=same_ev, list_equal=same_list, stream_equal=a["stream_sha256"] == b["stream_sha256"],
                  learned_equal=a["learned_sha256"] == b["learned_sha256"], resumes=b["resumes"],
                  state4_equal=st4["weights_tensors_sha256"] == T.load(LIVES / "c6_split" / f"{C6}_state_v4.pt", weights_only=False)["weights_tensors_sha256"])
    ok &= all(v for k, v in resume.items() if k != "resumes") and len(b["resumes"]) == 1 and b["resumes"][0]["done_visits"] == 4
    ok &= a["counters"]["asking_checks"] == 8 and b["counters"]["asking_checks"] == 8
    return dict(passed=bool(ok), asking_in_process=rows, asking_checks_in_lives={"c6_full": a["counters"]["asking_checks"], "c6_split": b["counters"]["asking_checks"]},
                resume=resume, note="Same stream for both arms is read on the calibration's receipts (post).")


def check6_post(receipts):
    by = {}
    for r in receipts:
        by.setdefault((r["settings"]["world"], r["settings"]["visit_updates"]), {})[r["settings"]["arm"]] = r
    rows, ok = [], True
    for (w, vu), d in sorted(by.items()):
        n, s_ = d["naive"], d["sweep"]
        e = n["stream_sha256"] == s_["stream_sha256"] and [v["stream_sha256"] for v in n["by_visit"]] == [v["stream_sha256"] for v in s_["by_visit"]] \
            and n["learned_sha256"] == s_["learned_sha256"]
        ea, eb = np.load(F1 / n["events"]["file"]), np.load(F1 / s_["events"]["file"])
        cols = all(np.array_equal(ea[k], eb[k]) for k in ("update", "visit", "task", "wrap", "mirror", "seen"))
        rows.append(dict(world=w, visit_updates=vu, stream_equal=e, arm_independent_columns_equal=cols)); ok &= e and cols
    return dict(passed=bool(ok and len(rows) == 24), pairs=rows)


# ---------------------------------------------------------------- check 7
def hand_events(w, visits, right):
    ev = dict(update=[], visit=[], task=[], correct=[], wrap=[], mirror=[])
    u = 0
    for s in w.schedule(visits, in_turn=True):
        for taught, asked in w.visit_batches(s["visit"], s["group"]):
            for i, a, b, p, y in asked:
                t = w.tasks[i]
                ev["update"].append(u); ev["visit"].append(s["visit"]); ev["task"].append(i); ev["correct"].append(int(right(u, i, a, b)))
                ev["wrap"].append(int(LW.needs_wrap(t["op"], t["p"], a, b))); ev["mirror"].append(w.mirror(t, a, b))
            u += 1
    return {k: np.array(x) for k, x in ev.items()}


def reviewer_first_visit_levels():
    """The reviewer's scorer for this quantity: first_visit_levels from e44_brief_checks.py (part I), taken from its source by ast."""
    src = git_show(RDIR + "e44_brief_checks.py").decode()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "first_visit_levels")
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "e44_brief_checks.py", "exec"), ns)
    return ns["first_visit_levels"], sha(src.encode())


def check7_pre():
    _, _, fx, _ = audit_tables()
    rule = lambda u, i, a, b: (7 * a + 3 * b + u + i) % 4 == 0
    rows, ok = {}, True
    for vu in (64, 256):
        w = LW.World(1, visit_updates=vu); ev = hand_events(w, 8, rule)
        fv = S.lw_first_visits(ev, w, vu, [(v, v) for v in range(8)])
        last64 = []
        for r in fv:
            u0, u1 = S.visit_range(ev, r["visit"]); last64.append(S.lw_levels(ev, r["visit"], r["family"], u1 - 64, u1)["figure"])
        med = [round(statistics.median(last64[1:]), 4), round(S.lw_gate_median(fv), 4),
               round(statistics.median([r["whole"]["figure"] for r in fv if r["n"] >= 2]), 4)]
        cols = [sha(bytes(int(x) for x in ev[k]))[:16] for k in ("correct", "wrap", "mirror")]
        good = med == fx[vu]["medians"] and cols == fx[vu]["columns"]
        # the hand computation: the window's items counted directly from the stream, without the scorer
        hand = []
        for f in range(8):
            q = vu // 4; lo, hi = (f + 1) * vu - q, (f + 1) * vu
            sh = []
            for j in (0, 1):
                n = r_ = 0
                for u_, v_, t_, c_, wr in zip(ev["update"], ev["visit"], ev["task"], ev["correct"], ev["wrap"]):
                    if v_ == f and lo <= u_ < hi and t_ == 4 * f + j and wr == 1:
                        n += 1; r_ += int(c_)
                sh.append(r_ / n)
            hand.append((sh[0] + sh[1]) / 2)
        hand_equal = all(abs(h - r["last_quarter"]["figure"]) < 1e-12 for h, r in zip(hand, fv))
        good &= hand_equal
        rows[vu] = dict(medians_last64_lastquarter_whole=med, audit=fx[vu]["medians"], columns=cols, audit_columns=fx[vu]["columns"],
                        per_family_last_quarter=[dict(family=r["family"], add=r["last_quarter"]["add_reduction"], sub=r["last_quarter"]["sub_reduction"],
                                                      figure=r["last_quarter"]["figure"]) for r in fv],
                        hand_computed_equal=hand_equal, passed=good)
        ok &= good
    return dict(passed=bool(ok), fixture=rows)


def check7_post(life_npz, vu):
    f, src_sha = reviewer_first_visit_levels()
    ev = {k: v for k, v in np.load(life_npz).items()}
    w = LW.World(int(Path(life_npz).name.split("_w")[1].split("_")[0]), visit_updates=vu)
    mine = S.lw_first_visits(ev, w, vu, [(v, v) for v in range(8)])
    out, ok = {}, True
    for name, window in (("last_quarter", vu // 4), ("whole", None)):
        theirs = f(ev, w, list(range(8)), window)
        rows = []
        for r in mine:
            a, b, m, n = theirs[r["family"]]
            mm = r[name]
            e = abs(a - mm["add_reduction"]["share"]) < 1e-12 and abs(b - mm["sub_reduction"]["share"]) < 1e-12 and abs(m - mm["figure"]) < 1e-12 \
                and n == mm["add_reduction"]["n"] + mm["sub_reduction"]["n"]
            rows.append(dict(family=r["family"], reviewer=(a, b, m, n), mine=(mm["add_reduction"]["share"], mm["sub_reduction"]["share"], mm["figure"],
                                                                               mm["add_reduction"]["n"] + mm["sub_reduction"]["n"]), equal=e))
            ok &= e
        out[name] = rows
    return dict(passed=bool(ok), life=Path(life_npz).name, reviewer_source_sha256=src_sha, rows=out)


# ---------------------------------------------------------------- check 8
def check8(receipts):
    out = {}
    for vu in LENGTHS:
        d = {}
        for arm in ("naive", "sweep"):
            rs = [r for r in receipts if r["settings"]["visit_updates"] == vu and r["settings"]["arm"] == arm]
            per = [x for r in rs for x in r["cpu"]["per_visit"]]
            lst = [x for r in rs for x in r["cpu"]["list_per_visit"]]
            d[arm] = dict(lives=len(rs), cpu_per_visit_mean=float(np.mean(per)), cpu_per_visit_max=float(np.max(per)),
                          cpu_per_update=float(np.mean(per)) / vu, forecast_1024_visits_hours=1024 * float(np.mean(per)) / 3600,
                          list_cpu_per_visit_mean=float(np.mean(lst)),
                          peak_rss_kib=max(r["cpu"]["peak_rss_kib"] for r in rs))
        out[vu] = d
    return dict(passed=True, by_visit_length=out,
                note="CPU-seconds per visit: the visit's updates with the answering of its asked items (sw_run Life.visit), one thread; the "
                     "list answered at the end of every visit is timed apart (list_cpu). The forecast is 1,024 times the mean per visit.")


def write_new(path, obj):
    assert not Path(path).exists(), f"refusing an existing path: {path}"
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


def pre():
    res = {}
    for name, fn in (("check1", check1_pre), ("check2", check2), ("check3", check3), ("check4", check4), ("check5", check5),
                     ("check6", check6_pre), ("check7", check7_pre)):
        res[name] = fn()
        print(name, "PASS" if res[name]["passed"] else "FAIL", flush=True)
    srcs = {p.name: sha(p.read_bytes()) for p in sorted(HERE.glob("*.py")) if p.name.startswith(("sw_", "lw_"))}
    write_new(CHK / "E44_CHECKS_PRE.json", dict(experiment="E44", stage="checks before the calibration", brief="revision 2 (cc00760d)", checks=res,
                                               all_passed=all(r["passed"] for r in res.values()), sources_sha256=srcs))


def gate_receipts():
    rs = [json.loads(p.read_text()) for p in sorted(F1.glob("lw_*_turn.json"))]
    return [r for r in rs if r["settings"]["arm"] in ("naive", "sweep") and r["settings"]["wd"] == 0.25 and r["settings"]["visit_updates"] in LENGTHS
            and r["settings"]["taught_share"] == 0.10 and r["kind"] == "life"]


def post():
    pre_ = json.loads((CHK / "E44_CHECKS_PRE.json").read_text())
    rs = gate_receipts()
    res = dict(check1_receipts=check1_post(rs), check1_second_lap=check1_lap2(), check6_same_stream=check6_post(rs),
               check7_calibration_life=check7_post(F1 / "lw_naive_w1_v8_vu128_turn.npz", 128), check8=check8(rs))
    for k, v in res.items():
        print(k, "PASS" if v["passed"] else "FAIL", flush=True)
    allp = pre_["all_passed"] and all(v["passed"] for v in res.values())
    srcs = {p.name: sha(p.read_bytes()) for p in sorted(HERE.glob("*.py")) if p.name.startswith(("sw_", "lw_"))}
    write_new(CHK / "E44_CHECKS.json", dict(experiment="E44", brief="revision 2 (cc00760d)", before_calibration=pre_["checks"], after_calibration=res,
                                           all_passed=allp, sources_sha256=srcs))


if __name__ == "__main__":
    {"lives": lives, "pre": pre, "post": post}[sys.argv[1]]()
