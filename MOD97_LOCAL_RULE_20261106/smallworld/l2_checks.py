"""E44, job L2 (roster note OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md, revision 2): the checks of the runner change, (a) to (d),
the note's checks 1 to 7, and the two 8-visit lives of the other two births (the audit's recommendation 3).

    python3 l2_checks.py jobs       # writes exp45/e44/L2/checks/check_jobs.json (the check lives; run by l2_driver.py)
    python3 l2_checks.py report     # reads the check lives -> exp45/e44/L2/checks/E44_L2_CHECKS.json (refused if it exists)

The check lives (all through sw_run.py; directories under exp45/e44/L2/checks/ unless said):
  a_base       world 1, 8 visits in turn, 256 updates, defaults, under the runner at 39e4e594 (run before the change was made)
  a_new        the same under the changed runner
  b_default, b_k4_d1248, b_k4_none      world 5, 32 visits by the schedule, naive, the birth of seed 93683: defaults; --list-every 4
               --decay-distances 1,2,4,8; --list-every 4 --decay-distances none
  c3_naive, c3_replay_rho0              world 1, one visit: naive; replay with rho = 0
  c5_whole, c5_cut                      world 1, 8 visits, replay, rho 0.5, decay 0, --list-every 2 --decay-distances 1,2,4: whole; stopped
               after 4 visits and resumed from its state
  c7_<arm>     world 5, 4 visits, --list-every 8 --decay-distances none, one process alone on the host: the four arms' cost
  ../births/   world 2 with the birth of seed 93683 and world 3 with the birth of seed 116089: 8 visits in turn at 256 updates, naive

The reviewer's files (e44_l2_checks.py with the .py files beside it, which it imports) are read from git (REVIEWER_REF) into a temporary directory
and run from there with python3 -I; their hashes are recorded. Nothing here trains.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch as T

HERE = Path(__file__).resolve().parent
MDIR = HERE.parent
ROOT = MDIR.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(MDIR))
import lw_world as LW                 # noqa: E402
import sw_run as R                    # noqa: E402
import sw_score as S                  # noqa: E402

L2 = MDIR / "exp45" / "e44" / "L2"
CHK = L2 / "checks"
F1 = MDIR / "exp45" / "e44" / "F1"
REVIEWER_REF = "origin/claude/vibrant-cerf-sje726"
RDIR = "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/"
AUDIT = RDIR + "AUDIT_E44_L2_ROSTER_NOTE_20261117.md"
SEEDS = (73129, 93683, 116089)
BEFORE, BIRTH_FAMILY = 65535, -1


def birth(seed, rule="B"):
    return MDIR / "exp45" / "e41" / f"e41_{rule}_h256_f50_{seed}.pt"


def world_seed(w):
    return SEEDS[(w - 1) % 3]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git_show(path, ref=REVIEWER_REF):
    return subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


def git_rev(ref):
    return subprocess.run(["git", "rev-parse", ref], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def rec(d, stem):
    return json.loads((CHK / d / f"{stem}.json").read_text())


def npz(p):
    f = np.load(p)
    return {k: f[k] for k in f.files}


def hashes(r):
    return dict(list_arrays=r["list"]["arrays_sha256"], events_arrays=r["events"]["arrays_sha256"], weights=r["final"]["weights_tensors_sha256"],
                optimizer=r["final"]["optimizer_sha256"], learned=r["learned_sha256"], stream=r["stream_sha256"])


# ---------------------------------------------------------------- the check lives
def jobs():
    lw = lambda s: ["--world", "lw", "--learner", str(birth(s))]
    c5 = ["life", "replay", "1", "8", "--rho", "0.5", "--wd", "0", *lw(73129), "--list-every", "2", "--decay-distances", "1,2,4"]
    b = ["life", "naive", "5", "32", *lw(93683), "--save-states", "32"]
    js = [dict(id="a_new", out=str(CHK / "a_new"), args=["life", "naive", "1", "8", *lw(73129), "--in-turn", "--visit-updates", "256", "--save-states", "8"]),
          dict(id="b_default", out=str(CHK / "b_default"), args=b),
          dict(id="b_k4_d1248", out=str(CHK / "b_k4_d1248"), args=b + ["--list-every", "4", "--decay-distances", "1,2,4,8"]),
          dict(id="b_k4_none", out=str(CHK / "b_k4_none"), args=b + ["--list-every", "4", "--decay-distances", "none"]),
          dict(id="c3_naive", out=str(CHK / "c3_naive"), args=["life", "naive", "1", "1", *lw(73129)]),
          dict(id="c3_replay_rho0", out=str(CHK / "c3_replay_rho0"), args=["life", "replay", "1", "1", "--rho", "0", *lw(73129)]),
          dict(id="c5_whole", out=str(CHK / "c5_whole"), args=c5),
          dict(id="c5_cut", out=str(CHK / "c5_cut"), args=c5 + ["--stop-after", "4"]),
          dict(id="c5_resumed", out=str(CHK / "c5_cut"), after=["c5_cut"],
               args=["life", "--resume", str(CHK / "c5_cut" / "lw_replay_w1_v8_rho0.5_wd0_state_v4.pt")]),
          dict(id="birth_w2_93683", out=str(L2 / "births"),
               args=["life", "naive", "2", "8", *lw(93683), "--in-turn", "--visit-updates", "256", "--save-states", "8"]),
          dict(id="birth_w3_116089", out=str(L2 / "births"),
               args=["life", "naive", "3", "8", *lw(116089), "--in-turn", "--visit-updates", "256", "--save-states", "8"])]
    cost = ["--list-every", "8", "--decay-distances", "none", "--save-states", "4"]
    for name, a in (("naive", ["life", "naive", "5", "4"]), ("replay", ["life", "replay", "5", "4", "--rho", "0.5"]),
                    ("naive_wd0", ["life", "naive", "5", "4", "--wd", "0"]), ("replay_wd0", ["life", "replay", "5", "4", "--rho", "0.5", "--wd", "0"])):
        js.append(dict(id=f"c7_{name}", out=str(CHK / f"c7_{name}"), alone=True, args=a + lw(93683) + cost))
    CHK.mkdir(parents=True, exist_ok=True)
    (CHK / "check_jobs.json").write_text(json.dumps(js, indent=1))


# ---------------------------------------------------------------- the list's rows
def list_rows(la):
    """key (at_visit, source, family, decay_updates, weights_from_visit) -> the row's items as arrays (task, a, b, g1, g2, g3, correct)."""
    cols = [np.asarray(la[k]).astype(np.int64) for k in ("at_visit", "source", "family", "decay_updates", "weights_from_visit")]
    key = np.stack(cols, 1)
    cut = np.nonzero(np.any(key[1:] != key[:-1], axis=1))[0] + 1
    out = {}
    for s, e in zip(np.concatenate([[0], cut]), np.concatenate([cut, [len(key)]])):
        k = tuple(int(x) for x in key[s])
        assert k not in out, f"a key occurs twice: {k}"
        out[k] = np.stack([np.asarray(la[c][s:e]).astype(np.int64) for c in ("task", "a", "b", "g1", "g2", "g3", "correct")], 1)
    return out


def keys_digest(rows):
    """The audit's (section 7): SHA-256 over the sorted lines at_visit,source,family,decay_updates,weights_from_visit,items."""
    lines = sorted(k + (len(v),) for k, v in rows.items())
    return hashlib.sha256("".join(",".join(str(x) for x in r) + "\n" for r in lines).encode()).hexdigest()


def audit_digests():
    """The row-key digests and the list-item digests of the audit's section 7, parsed from the audit itself."""
    txt = git_show(AUDIT).decode()
    sec = txt.split("## 7. Figures for the worker to compare")[1].split("## 8.")[0]
    keys, items = {}, {}
    for ln in sec.splitlines():
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) == 4 and len(c[3]) == 64:
            keys[c[0]] = dict(keys=int(c[1].replace(",", "")), items=int(c[2].replace(",", "")), digest=c[3])
        if len(c) == 3 and c[0].isdigit():
            for V, cell in ((256, c[1]), (1024, c[2])):
                if cell:
                    n, d = cell.split(",")
                    items[(int(c[0]), V)] = dict(families=int(n.split()[0]), digest=d.strip())
    return keys, items, sha(txt.encode())


def reviewer_k(td, path, world, visits, K, dist, in_turn=False):
    a = [sys.executable, "-I", str(Path(td) / "e44_l2_checks.py"), "K", str(path), str(world), str(visits), str(K), dist] + (["in-turn"] if in_turn else [])
    run = subprocess.run(a, capture_output=True, text=True, cwd=td)
    assert run.returncode == 0, f"the reviewer's part K did not run: {run.stderr[-600:]}"
    out = run.stdout
    return dict(command="python3 -I e44_l2_checks.py K " + " ".join(a[4:]), output=out.strip().splitlines(),
                keys_equal_as_sets="keys equal as sets: True" in out, items_in_reviewers_order="in the reviewer's order: True" in out)


# ---------------------------------------------------------------- the checks of the change
def check_a():
    base, new = rec("a_base", "lw_naive_w1_v8_turn"), rec("a_new", "lw_naive_w1_v8_turn")
    e44 = json.loads((F1 / "lw_naive_w1_v8_turn.json").read_text())
    hb, hn, he = hashes(base), hashes(new), hashes(e44)
    return dict(passed=hb == hn, this_host_reproduces_E44s_committed_receipt=hb == he, equal_to_committed=hn == he,
                runner_at_39e4e594=dict(sw_run_sha256=base["provenance"]["sources_sha256"]["smallworld/sw_run.py"], hashes=hb,
                                        equals_E44_receipts_runner=base["provenance"]["sources_sha256"]["smallworld/sw_run.py"] == e44["provenance"]["sources_sha256"]["smallworld/sw_run.py"]),
                changed_runner=dict(sw_run_sha256=new["provenance"]["sources_sha256"]["smallworld/sw_run.py"], hashes=hn, settings=new["settings"]),
                E44_committed=dict(receipt="exp45/e44/F1/lw_naive_w1_v8_turn.json", hashes=he),
                list_file_bytes_equal=(CHK / "a_base" / "lw_naive_w1_v8_turn_list.npz").read_bytes() == (CHK / "a_new" / "lw_naive_w1_v8_turn_list.npz").read_bytes())


def check_bc(td, keys):
    names = dict(b_default=("world 5, 32 visits by the schedule, defaults", "1", "all"),
                 b_k4_d1248=("world 5, 32 visits, `--list-every 4 --decay-distances 1,2,4,8`", "4", "1,2,4,8"),
                 b_k4_none=("world 5, 32 visits, `--list-every 4 --decay-distances none`", "4", "none"))
    stem = "lw_naive_w5_v32"
    rows, out, ok = {}, {}, True
    for d, (label, K, dist) in names.items():
        r = rec(d, stem); la = npz(CHK / d / f"{stem}_list.npz")
        rows[d] = list_rows(la)
        dg = keys_digest(rows[d]); want = keys[label]
        k = reviewer_k(td, CHK / d / f"{stem}_list.npz", 5, 32, K, dist)
        out[d] = dict(command=r["provenance"]["command"], settings=dict(list_every=r["settings"]["list_every"], decay_distances=r["settings"]["decay_distances"]),
                      keys=len(rows[d]), items=int(sum(len(v) for v in rows[d].values())), keys_digest=dg, audit=want,
                      equal_to_audit=dg == want["digest"] and len(rows[d]) == want["keys"] and sum(len(v) for v in rows[d].values()) == want["items"],
                      reviewers_part_K=k, hashes=hashes(r), birth_file_sha256=r["learner"]["file_sha256"], birth_seed=r["learner"]["seed"],
                      list_cpu=r["cpu"]["list_total"], cpu_total=r["cpu"]["total"])
        ok &= out[d]["equal_to_audit"] and k["keys_equal_as_sets"] and k["items_in_reviewers_order"]
    pairs = {}
    for x, y in (("b_default", "b_k4_d1248"), ("b_default", "b_k4_none"), ("b_k4_d1248", "b_k4_none")):
        common = sorted(set(rows[x]) & set(rows[y]))
        same = [k for k in common if rows[x][k].shape == rows[y][k].shape and (rows[x][k] == rows[y][k]).all()]
        pairs[f"{x} and {y}"] = dict(rows_in_both=len(common), rows_equal_item_by_item=len(same), items_compared=int(sum(len(rows[x][k]) for k in common)),
                                     all_equal=len(same) == len(common),
                                     second_is_subset_of_first=set(rows[y]) <= set(rows[x]))
        ok &= len(same) == len(common) and len(common) > 0
    b = dict(passed=bool(ok), runs=out, rows_in_two_runs=pairs,
             network_family_rows=dict(defaults=sum(1 for k in rows["b_default"] if k[1] == 0 and k[2] >= 0),
                                      list_every_4=sum(1 for k in rows["b_k4_none"] if k[1] == 0 and k[2] >= 0)))
    hs = [{k: v for k, v in out[d]["hashes"].items() if k != "list_arrays"} for d in names]
    c = dict(passed=hs[0] == hs[1] == hs[2], compared=["events_arrays", "weights", "optimizer", "learned", "stream"], hashes={d: out[d]["hashes"] for d in names},
             event_files_bytes_equal=len({(CHK / d / f"{stem}.npz").read_bytes() for d in names}) == 1,
             note="The three runs of (b) differ only in which list rows they answer; the event file, the final weights, the optimizer state and the "
                  "learned batches are the same in all three.")
    return b, c


def check_d():
    out, ok = {}, True
    for d, stem, want in (("b_default", "lw_naive_w5_v32", (1, None)), ("b_k4_d1248", "lw_naive_w5_v32", (4, [1, 2, 4, 8])),
                          ("b_k4_none", "lw_naive_w5_v32", (4, [])), ("c5_whole", "lw_replay_w1_v8_rho0.5_wd0", (2, [1, 2, 4])),
                          ("c5_cut", "lw_replay_w1_v8_rho0.5_wd0", (2, [1, 2, 4]))):
        r = rec(d, stem)
        got = (r["settings"]["list_every"], r["settings"]["decay_distances"])
        states = {}
        for p in sorted((CHK / d).glob(f"{stem}_state_v*.pt")):
            s = T.load(p, weights_only=False)
            states[p.name] = [s["cfg"].get("list_every"), s["cfg"].get("decay_distances")]
        good = got == want and all(tuple(v) == want for v in states.values()) and len(states) > 0
        out[d] = dict(receipt_settings=list(got), states=states, resumes=len(r["resumes"]), passed=good)
        ok &= good
    return dict(passed=bool(ok), lives=out,
                note="c5_cut was stopped after 4 visits and continued with `life --resume`, which takes no --list-every or --decay-distances: its "
                     "receipt and its states after the resume carry the settings of the state; its list file equals the whole life's (check 5).")


# ---------------------------------------------------------------- the note's checks
def check1(items_audit):
    import lw_checks as LC
    pre = LC.check1_pre()
    e44 = json.loads((MDIR / "exp45" / "e44" / "checks" / "E44_CHECKS.json").read_text())
    old = e44["before_calibration"]["check1"]["worlds"]
    same = all(pre["worlds"][s][k] == old[str(s)][k] for s in range(1, 9) for k in ("schedule_1024", "tasks", "list_0_7", "in_turn_256"))
    lists, ok = {}, True
    for (w, V), want in sorted(items_audit.items()):
        wd = LW.World(w); fams = []
        for s in wd.schedule(V):
            if s["group"] not in fams:
                fams.append(s["group"])
        dg = R.list_digest([it for f in fams for it in R.probe_items(wd, f)])
        lists[f"world {w}, {V} visits"] = dict(families=len(fams), items_sha256=dg, audit=want, equal=dg == want["digest"] and len(fams) == want["families"])
        ok &= lists[f"world {w}, {V} visits"]["equal"]
    src_now = R.source_hashes()
    src_e44 = json.loads((F1 / "lw_naive_w1_v8_turn.json").read_text())["provenance"]["sources_sha256"]
    return dict(passed=bool(pre["passed"] and same and ok), e44_check1_pre=pre, equal_to_E44_CHECKS_json=same, list_items_of_the_roster=lists,
                sources_changed_since_the_calibration=sorted(k for k in src_now if src_now[k] != src_e44.get(k)),
                note="E44's check 1 (lw_checks.check1_pre) run again with the changed runner: the generator against its specification, the "
                     "reviewer's reader, and the earlier audit's digests of the schedules of 1,024 visits and of the tasks of worlds 1 to 8. "
                     "Added: the digest of the list's items of each roster life (families in the order first met) against the audit's section 7.")


E41_SIDE = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import exp41_scale as X
import torch as T
X.arch26.install()
out = {}
for path in sys.argv[2:]:
    rec = json.load(open(path[:-3] + ".json"))
    m = X.NetW(256, 128); m.load_state_dict(T.load(path))
    A, Rs, TA = X.panels(rec["fraction"], rec["seed"])
    sc = X.score_all(m, A, Rs, TA)
    last = rec["curve"][-1]
    out[path] = dict(rule=rec["rule"], seed=rec["seed"], tensors_sha256=X.R_.tensors_sha(m.state_dict()), receipt_tensors_sha256=rec["tensors_sha256"],
                     A=sc["A"], R=sc["R"], A_counts=sc["A_counts"], R_counts=sc["R_counts"], receipt_A=last["A"], receipt_R=last["R"],
                     receipt_A_counts=last["A_counts"], receipt_R_counts=last["R_counts"], cls=type(m).__module__ + "." + type(m).__name__)
print(json.dumps(out))
'''


def check2():
    import arch26
    E = arch26.E
    paths = [birth(s, rule) for s in (93683, 116089) for rule in ("B", "S")]
    e41 = json.loads(subprocess.run([sys.executable, "-c", E41_SIDE, str(MDIR)] + [str(p) for p in paths], capture_output=True, text=True,
                                    check=True, cwd=str(MDIR)).stdout)
    out, ok = {}, True
    for p in paths:
        d = e41[str(p)]; seed = d["seed"]
        r = json.loads(p.with_suffix(".json").read_text())
        d["file_sha256"] = R.sha_file(p); d["receipt_weights_sha256"] = r["weights_sha256"]
        d["e41_side_equal"] = d["tensors_sha256"] == d["receipt_tensors_sha256"] and d["A_counts"] == d["receipt_A_counts"] and d["R_counts"] == d["receipt_R_counts"]
        m, info = R.load_learner(p, 256, 128, "naive" if d["rule"] == "B" else "sweep")
        rit = R.e41_r_items(0.5, seed, info["reduction_sample_sha256"])
        g = R.answer_all(m, rit)
        rc = {"add89": sum(x == it[4] for x, it in zip(g[:1000], rit[:1000])), "sub89": sum(x == it[4] for x, it in zip(g[1000:], rit[1000:]))}
        ac = {}
        for t in E.DEV:
            held = E.split(t, 0.5, seed)[1]
            items = E.panel(t, held, seed, ks=(12,), n=300)[12]
            gg = R.answer_all(m, [(0, 0, 0, pr, y) for pr, y in items])
            ac[E.NAME(*t)] = sum(x == y for x, (pr, y) in zip(gg, items))
        d["runner"] = dict(cls="sw_net.SWNet", tensors_sha256=info["tensors_sha256"], R_counts=rc, A_counts=ac,
                           R=100 * (rc["add89"] + rc["sub89"]) / 2000, A=100 * (ac["add89"] + ac["sub89"]) / 600)
        d["runner_equal"] = info["tensors_sha256"] == d["receipt_tensors_sha256"] and rc == d["receipt_R_counts"] and ac == d["receipt_A_counts"]
        d["file_hash_equal"] = d["file_sha256"] == d["receipt_weights_sha256"]
        if d["rule"] == "B":                                      # the replay arm loads the same birth (the second runner change)
            m2, info2 = R.load_learner(p, 256, 128, "replay")
            d["replay_arm_loads_it"] = info2["tensors_sha256"] == info["tensors_sha256"]
            ok &= d["replay_arm_loads_it"]
        ok &= d["e41_side_equal"] and d["runner_equal"] and d["file_hash_equal"]
        out[p.name] = d
    return dict(passed=bool(ok), births=out,
                note="As E44's check 3, for the half-width births of seeds 93683 and 116089 (rules B and S; the roster uses B). E41's side: "
                     "exp41_scale.NetW(256, 128) with the file, A and R by E41's own panels and scorer. The runner's side: sw_net.SWNet(256, 128) "
                     "through sw_run.load_learner, answered by sw_net.answer on the same items.")


def check3():
    a, b = rec("c3_naive", "lw_naive_w1_v1"), rec("c3_replay_rho0", "lw_replay_w1_v1_rho0")
    ha, hb = hashes(a), hashes(b)
    ea, eb = npz(CHK / "c3_naive" / "lw_naive_w1_v1.npz"), npz(CHK / "c3_replay_rho0" / "lw_replay_w1_v1_rho0.npz")
    ev_equal = all((ea[k] == eb[k]).all() for k in ea)
    return dict(passed=ha == hb and ev_equal, naive=dict(command=a["provenance"]["command"], hashes=ha),
                replay_rho0=dict(command=b["provenance"]["command"], hashes=hb, replay_items_per_update=b["replay_items_per_update"],
                                 learner_rule=b["learner"]["rule"], ledger_buffer=b["ledger"]["buffer"]),
                event_columns_equal=ev_equal,
                note="`replay` with rho = 0 loads backpropagation's birth (the second runner change) and learns from the 48 new items of every "
                     "update: the weights, the optimizer state, the event file's arrays, the learned batches and the list are those of `naive`.")


def check4():
    life = R.Life("replay", 1, 1, rho=0.5, H=256, U=128, visit_updates=256, p_new=LW.P_NEW, world_kind="lw", learner=str(birth(73129)))
    cap = life.capacity()
    out, ok = dict(at_the_start=dict(parameters=cap[0], capacity_items=cap[1], capacity_bytes=cap[2])), cap == (317457, 55209, 1269828)
    lives = {}
    for d, stem in (("c5_whole", "lw_replay_w1_v8_rho0.5_wd0"), ("c5_cut", "lw_replay_w1_v8_rho0.5_wd0"), ("c7_replay", "lw_replay_w5_v4_rho0.5"),
                    ("c7_replay_wd0", "lw_replay_w5_v4_rho0.5_wd0")):
        b = rec(d, stem)["ledger"]["buffer"]
        lives[d] = b
        ok &= b["capacity_items"] == 55209 and b["max_items"] <= 55209 and b["max_bytes"] <= b["capacity_bytes"]
    return dict(passed=bool(ok), **out, check_lives=lives,
                note="The runner does not print the capacity; it is read here from Life.capacity() of a `replay` life at its start, and from every "
                     "replay receipt's ledger (capacity_items, max_items). The runner asserts at every update that neither the count nor the "
                     "bytes exceed the capacity. The roster's replay lives are read the same way in the report.")


def check5():
    a, b = rec("c5_whole", "lw_replay_w1_v8_rho0.5_wd0"), rec("c5_cut", "lw_replay_w1_v8_rho0.5_wd0")
    ha, hb = hashes(a), hashes(b)
    la, lb = npz(CHK / "c5_whole" / "lw_replay_w1_v8_rho0.5_wd0_list.npz"), npz(CHK / "c5_cut" / "lw_replay_w1_v8_rho0.5_wd0_list.npz")
    ea, eb = npz(CHK / "c5_whole" / "lw_replay_w1_v8_rho0.5_wd0.npz"), npz(CHK / "c5_cut" / "lw_replay_w1_v8_rho0.5_wd0.npz")
    return dict(passed=ha == hb and len(b["resumes"]) == 1 and b["resumes"][0]["done_visits"] == 4 and not a["resumes"],
                whole=dict(command=a["provenance"]["command"], hashes=ha, ledger_buffer=a["ledger"]["buffer"], counters=a["counters"]),
                stopped_and_resumed=dict(command=b["provenance"]["command"], resumes=b["resumes"], hashes=hb, ledger_buffer=b["ledger"]["buffer"],
                                         counters=b["counters"]),
                list_columns_equal=all((la[k] == lb[k]).all() for k in la), event_columns_equal=all((ea[k] == eb[k]).all() for k in ea),
                list_keys=len(list_rows(la)),
                note="A `replay` life without decay of world 1, 8 visits, --list-every 2 --decay-distances 1,2,4: whole, and stopped after visit 4 "
                     "(--stop-after 4) then continued from its state. Equal: the final weights, the optimizer state, the event file's arrays, the "
                     "learned batches, the stream and the list file's arrays (change 4 d).")


def check6():
    out, ok = {}, True
    for arm, rho, wd in (("naive", None, 0.0), ("replay", 0.5, 0.0), ("naive", None, 0.25), ("replay", 0.5, 0.25)):
        life = R.Life(arm, 5, 4, rho=rho, H=256, U=128, visit_updates=256, p_new=LW.P_NEW, wd=wd, world_kind="lw", learner=str(birth(93683)))
        d = "c7_" + arm + ("_wd0" if wd == 0 else "")
        stem = f"lw_{arm}_w5_v4" + ("_rho0.5" if rho else "") + ("_wd0" if wd == 0 else "")
        r = rec(d, stem)
        g0 = life.opt.param_groups[0]["weight_decay"]; g_end = r["optimizer"]["groups"][0]["weight_decay"]
        good = g0 == wd and g_end == wd and r["optimizer"]["weight_decay"] == wd and r["settings"]["wd"] == wd and r["optimizer"]["groups"][1]["weight_decay"] == 0.0
        out[d] = dict(first_group_at_start=g0, first_group_at_end=g_end, receipt_optimizer_weight_decay=r["optimizer"]["weight_decay"],
                      receipt_settings_wd=r["settings"]["wd"], byte_table_group=r["optimizer"]["groups"][1]["weight_decay"], stem=stem, passed=good)
        ok &= good
    return dict(passed=bool(ok), arms=out,
                note="At the start: the first parameter group of a fresh optimizer of the arm (sw_net.make_opt through Life). At the end: the "
                     "receipt's optimizer.groups, read from the optimizer after the last update. The receipt's fields optimizer.weight_decay and "
                     "settings.wd record the arm's decay.")


def check7(keys):
    arms = {}
    for name, stem in (("naive", "lw_naive_w5_v4"), ("replay", "lw_replay_w5_v4_rho0.5"), ("naive-wd0", "lw_naive_w5_v4_wd0"),
                       ("replay-wd0", "lw_replay_w5_v4_rho0.5_wd0")):
        r = rec("c7_" + name.replace("-", "_"), stem)
        arms[name] = dict(cpu_seconds_per_visit=r["cpu"]["per_visit"], mean=r["cpu"]["mean_per_visit"], per_update=r["cpu"]["mean_per_visit"] / 256,
                          list_cpu=r["cpu"]["list_total"], list_entries=r["list"]["entries"], host=r["provenance"]["host"])
    b = rec("b_default", "lw_naive_w5_v32")
    per_item = b["cpu"]["list_total"] / b["list"]["entries"]
    trial_items = keys["the trial: world 1, 1,024 visits, `--list-every 8 --decay-distances none`"]["items"]
    short_items = keys["world 1, 256 visits, `--list-every 8 --decay-distances none`"]["items"]
    h = lambda arm, visits, lives: lives * visits * arms[arm]["mean"] / 3600
    rows = [("the trial", "naive", 1024, 1), ("the trial", "replay", 1024, 1), ("long", "naive", 1024, 3), ("long", "replay", 1024, 3),
            ("short", "naive", 256, 4), ("short", "replay", 256, 4), ("decay", "naive-wd0", 256, 8), ("decay", "replay-wd0", 256, 8)]
    fc = [dict(run=n, arm=a, visits=v, lives=k, cpu_hours=h(a, v, k)) for n, a, v, k in rows]
    list_h = (8 * trial_items + 24 * short_items) * per_item / 3600
    total = sum(x["cpu_hours"] for x in fc) + list_h
    return dict(passed=total <= 75, arms=arms, list_cpu_seconds_per_item=per_item, forecast=fc,
                list_cpu_hours=list_h, forecast_release_1_cpu_hours=total, stop_condition_cpu_hours=75,
                life_of_1024_visits_cpu_hours={a: 1024 * arms[a]["mean"] / 3600 + trial_items * per_item / 3600 for a in ("naive", "replay")},
                note="Each arm: world 5, 4 visits (families 0, 1, 0, 1), one process alone on the host. The list's cost is forecast from the "
                     "seconds an item of check (b)'s default run and the item counts of world 1 in the audit's section 7 (worlds differ a "
                     "little). The store is not full in 4 visits; a draw from it and an insertion cost the same when it is.")


def births():
    out, ok = {}, True
    e44 = {2: 0.566, 3: 0.301}
    for w, seed in ((2, 93683), (3, 116089)):
        stem = f"lw_naive_w{w}_v8_turn"
        r = json.loads((L2 / "births" / f"{stem}.json").read_text())
        ev = npz(L2 / "births" / f"{stem}.npz")
        first = S.lw_first_visits(ev, LW.World(w), 256, [(v, v) for v in range(8)])
        med = S.lw_gate_median(first)
        out[f"world {w}"] = dict(birth_seed=r["learner"]["seed"], birth_file_sha256=r["learner"]["file_sha256"], command=r["provenance"]["command"],
                                 first_visit_levels=[x["last_quarter"]["figure"] for x in first], median_families_2_to_8=med,
                                 calibration_median_with_seed_73129=e44[w], at_or_above_0_20=med >= 0.20, events_arrays_sha256=r["events"]["arrays_sha256"],
                                 cpu_total=r["cpu"]["total"])
        assert r["learner"]["seed"] == seed
        ok &= med >= 0.20
    return dict(passed=bool(ok), lives=out,
                note="The audit's recommendation 3: one 8-visit life in turn at 256 updates under `naive` in worlds 2 and 3 with the births of "
                     "seeds 93683 and 116089. The median is the gate's: over the second to the eighth family taken, of the first-visit level on "
                     "reduction pairs (sw_score.lw_first_visits, lw_gate_median). Below 0.20 the worker stops and reports.")


def report():
    path = CHK / "E44_L2_CHECKS.json"
    assert not path.exists(), f"refusing an existing path: {path}"
    keys, items, audit_sha = audit_digests()
    with tempfile.TemporaryDirectory() as td:
        rv = {}
        names = subprocess.run(["git", "ls-tree", "--name-only", REVIEWER_REF, RDIR], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        for f in sorted(Path(n).name for n in names if n.endswith(".py")):          # the script imports its neighbours (np_gru, compare_pt_tensors, ...)
            b = git_show(RDIR + f); (Path(td) / f).write_bytes(b)
            if f in ("e44_l2_checks.py", "e44_world_reader.py", "np_gru.py", "compare_pt_tensors.py"):
                rv[f] = sha(b)
        a = check_a()
        a["reviewers_part_K"] = reviewer_k(td, CHK / "a_new" / "lw_naive_w1_v8_turn_list.npz", 1, 8, "1", "all", in_turn=True)
        a["keys_digest"] = keys_digest(list_rows(npz(CHK / "a_new" / "lw_naive_w1_v8_turn_list.npz")))
        a["keys_digest_equal_to_audit"] = a["keys_digest"] == keys["the calibration life: world 1, 8 visits in turn, defaults"]["digest"]
        a["passed"] = bool(a["passed"] and a["keys_digest_equal_to_audit"])
        b, c = check_bc(td, keys)
    out = dict(experiment="E44", job="L2, release 1: the checks before the lives", note_revision=2,
               reviewer=dict(ref=REVIEWER_REF, commit=git_rev(REVIEWER_REF), audit=AUDIT, audit_sha256=audit_sha, files_sha256=rv),
               host=R.host(), git=R.git_state(), sources_sha256=R.source_hashes(),
               change=dict(a=a, b=b, c=c, d=check_d()),
               checks={"1": check1(items), "2": check2(), "3": check3(), "4": check4(), "5": check5(), "6": check6(), "7": check7(keys)},
               other_births=births())
    out["all_passed"] = bool(all(x["passed"] for x in out["change"].values()) and all(x["passed"] for x in out["checks"].values())
                             and out["other_births"]["passed"])
    R.write_json(path, out)
    for k, v in list(out["change"].items()) + list(out["checks"].items()) + [("other births", out["other_births"])]:
        print(k, "PASS" if v["passed"] else "FAIL")
    print("all passed:", out["all_passed"])


if __name__ == "__main__":
    {"jobs": jobs, "report": report}[sys.argv[1]]()
