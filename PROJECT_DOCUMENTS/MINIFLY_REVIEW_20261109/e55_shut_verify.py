#!/usr/bin/env python3
"""Verification of the four runs of the arm `shut` of E55, first part (seed 73129), and of the worker's final report, against the brief
TASK_E55_THE_FIRST_FAMILY_KEPT_WITHOUT_A_LINE_OF_THE_PROGRAM_THAT_LOOKS_AT_THE_ITEM_20261118.md (revision 4) and the coordinator's decision 14.

The receipts, logs, weight files and check receipts are read at E51's worker's branch at commit 340a0009 and at that branch's head 05d223ff
(the two are shown to hold the same blob for each of them); the report, the tables, the reading and the list of committed files are read at the
head, which carries the worker's corrections.

Read-only. Nothing is trained, the training library is not used, and no file of the worker's branch is executed: every file is read with
`git show` into memory, one weight file at a time (the script stops reading weight files if less than 0.9 GB of disk would remain). The forward
pass, the generation of answers, the items, the streams, the digests and the hashes are those of the reviewer's verification of E51's trial
(e51_trial_verify.py) as imported by the verification of the arm `blind` (e55_blind_verify.py), both beside this file. The script writes no file.

    sections  0  the two commits: what was added or changed between them, and the identity of every receipt, log and weight file
              1  the receipts of the four runs of `shut`: completeness, settings, scored rows, losses, data digests from the script's own
                 streams, provenance, logs, CPU-hours
              2  the start `shut`, built again in numpy from E51's keyed start, against the saved start and check 1's receipt
              3  every figure of E55_TABLES_all.md and E55_READ_all.json that concerns `shut`, the figures of the report and those of the
                 worker's message, recomputed from the receipts' raw counts
              4  the saved networks of rows 64, 4,096 and 16,384 of each run of `shut` (the plain forward pass in numpy)
              5  the fixed wording and the predictions, run by run
              6  the runner as text
              7  the table asked for: the first family at 22 and at 50 positions, the new task as run and with the reserve held shut, the
                 gates of the reserve on both kinds of item, by scored row
              8  the nine corrections of message 9 at the head; what the report says remains supplied by the program; decision 14
              9  form: unfilled marks and names of language models in the committed text files of exp45/e55 and the two scripts

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e55_shut_verify.py
"""
import hashlib, importlib.util, json, math, os, re, shutil, subprocess, sys, time
import numpy as np

T0 = time.time()
sys.dont_write_bytecode = True
A = "340a00098895431d8c6ef63068e809c21e404709"          # the commit named in the task
H = "05d223ffabf84e8ca0b964723cbf3bf7f4f6e860"          # the head of the worker's branch, with the corrections
M = "MOD97_LOCAL_RULE_20261106/"
E55 = M + "exp45/e55/"
E51 = M + "exp45/e51/"
REVIEW = "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/"
assert os.path.isdir(".git") or os.path.isfile(".git"), "run from the repository root"

_argv = sys.argv; sys.argv = [sys.argv[0]]
sys.path.insert(0, REVIEW)
_spec = importlib.util.spec_from_file_location("e55_blind_verify", os.path.join(REVIEW, "e55_blind_verify.py"))
B = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(B)        # its sections run only as a program
sys.argv = _argv
V = B.V

SEED, BOS, EOS = V.SEED, V.BOS, V.EOS
H0, U0, H1, U1 = V.H0, V.U0, V.H1, V.U1
TASKS, RULES = ("unrelated", "conflict"), ("S", "B")
RUNS = [(t, r) for t in TASKS for r in RULES]
ROWS55, KEEP = B.ROWS55, B.KEEP
MATS, OLD = V.MATS, V.OLD
Z = np.arange(U1 + U0, 2 * U1)                          # rows 384 to 511: the update gates of the cells in reserve
ADDED = np.arange(H0, H1)
SHUTBLOCKS = {"ih": Z, "hh": Z, "lm": ADDED, "l2": ADDED}
RULEROWS = {"ih": np.concatenate([OLD, Z]), "hh": np.concatenate([OLD, Z]), "lm": np.arange(H1), "l2": np.arange(H1)}
S1, HELD, held, LEN40, SAMPLE = B.S1, B.HELD, B.held, B.LEN40, B.SAMPLE
N0 = sum(len(S1[n][0]) for n in S1)


def say(*a):
    print(*a, flush=True)


def show(path, ref=H):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def jshow(path, ref=H):
    return json.loads(show(path, ref))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def blob(ref, path):
    r = subprocess.run(["git", "rev-parse", f"{ref}:{path}"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def lsn(ref, *paths):
    return subprocess.run(["git", "ls-tree", "-r", "--name-only", ref, "--", *paths], capture_output=True, text=True, check=True).stdout.split()


def load_weights(path, ref=H):
    free = shutil.disk_usage("/home/user").free
    if free < 0.9e9 + 20e6:
        raise SystemExit(f"stopped before reading {path}: {free / 1e9:.2f} GB free")
    b = show(path, ref)
    W = {n.lstrip("/"): np.frombuffer(raw, dtype=V.DTYPE[st]).reshape(shape).copy() for n, (st, shape, raw) in V.tensors(b).items()}
    return W, sha(b)


def away(D, Vm):
    return D - (D @ Vm) @ Vm.T


def block(W, mn, rows):
    return B.block(W, mn, rows)


def clock(label, t0):
    say(f"[section {label}: {time.time() - t0:.0f} s]\n")


def gates(net, items, **key):
    """The runner's `gates_on`, written again, with more figures: over the cells in reserve, under teacher forcing.
    largest 1 - z; cells whose mean of 1 - z over positions and items exceeds 0.1; cells for which the mean over items of the largest 1 - z
    over the item's positions exceeds 0.1 (the brief's *open at some position*) and 0.5; the largest state; the largest added unit at the
    answer positions; the share of (item, position) pairs at which at least one cell in reserve has 1 - z above 0.5; the share of
    (item, position, cell) triples with 1 - z above 0.5; the number of cells with 1 - z above 0.5 at some position of some item; and the
    largest state of a cell in reserve at each position (for items of one length)."""
    tot = np.zeros(U1 - U0); maxsum = np.zeros(U1 - U0); n = ni = 0
    big = st = add = 0.0; pos_any = pos_n = trip = 0; ever = np.zeros(U1 - U0, bool); prof = None
    for idx in V.groups_of(items):
        x = np.array([[BOS] + items[i][0] + items[i][1][:-1] for i in idx])
        k = net.run(x, keep=True, **key)[1]
        op = 1.0 - k["z"][:, :, U0:].astype(np.float64)
        tot += op.sum((0, 1)); n += op.shape[0] * op.shape[1]; maxsum += op.max(1).sum(0); ni += op.shape[0]
        big = max(big, float(op.max())); h = np.abs(k["h"][:, :, U0:]); st = max(st, float(h.max()))
        add = max(add, float(np.abs(k["z2"][:, :, H0:]).max()))
        o5 = op > 0.5; pos_any += int(o5.any(2).sum()); pos_n += op.shape[0] * op.shape[1]; trip += int(o5.sum()); ever |= o5.any((0, 1))
        p = h.max((0, 2)); prof = p if prof is None or len(p) != len(prof) else np.maximum(prof, p)
    mo, mm = tot / n, maxsum / ni
    return dict(largest_opening=big, open_mean_0p1=int((mo > 0.1).sum()), open_some=int((mm > 0.1).sum()), open_some_0p5=int((mm > 0.5).sum()),
                largest_state=st, largest_added=add, share_pos_any_0p5=pos_any / pos_n, share_triples_0p5=trip / (pos_n * (U1 - U0)),
                cells_ever_0p5=int(ever.sum()), profile=prof)


def first_learned(curve):
    return B.first_learned(curve)


# ---------------------------------------------------------------- receipts
REC = {(t, r): jshow(f"{E55}e55_{t}_shut_{r}_{SEED}.json") for t, r in RUNS}
row = lambda rec, g: [x for x in rec["rows"] if x["updates"] == g][0]


def recompute(rec):
    out = B.recompute(rec)
    for o, x in zip(out, rec["rows"]):
        s1, s2 = x["score1_first_family_all_pairs_k0"], x["score2_first_family_heldout_k12"]
        o["right0"] = sum(s1[n][p]["exact"] for n in ("add89", "sub89") for p in ("reduction", "other")) + sum(s1[n]["all"]["exact"] for n in ("max89", "min89"))
        o["right12"] = sum(s2[n]["exact"] for n in s2)
        o["new_05"] = x["gates_on_new_task"]["cells_open_by_more_than_0p5_at_some_position"]
        o["ff_open05"] = x["gates_on_first_family"]["cells_open_by_more_than_0p5_at_some_position"]
        o["ff40_open"] = x["gates_on_first_family_40_letters"]["cells_open_at_some_position"]
    return out


MINE = {k: recompute(v) for k, v in REC.items()}


def words(rows):
    """The brief's words for the first family in one run (the 40-letter clause applies to the arm `blind` only)."""
    r0 = rows[0]["reduction"]
    if all(x["d"] == 0 and x["held"] == 0 for x in rows):
        return "kept exactly"
    if all(x["d"] <= 31 and x["held"] <= 1 and abs(x["reduction"] - r0) <= 0.5 for x in rows):
        return "kept"
    return "lost" if rows[-1]["reduction"] < r0 / 2 else "partly kept"


def first_fail(rows):
    r0 = rows[0]["reduction"]
    for x in rows:
        c = []
        if x["d"] > 31:
            c.append(("d", x["d"]))
        if x["held"] > 1:
            c.append(("held", x["held"]))
        if abs(x["reduction"] - r0) > 0.5:
            c.append(("reduction", round(x["reduction"] - r0, 2)))
        if c:
            return x["updates"], c
    return None, []


# ================================================================ section 0
def section0():
    t0 = time.time(); say("===== SECTION 0. The two commits =====")
    paths = [E55, M + "exp55_reserve.py", M + "e55_read.py"]
    la, lh = lsn(A, *paths), lsn(H, *paths)
    say(f"0.1 the head of the worker's branch {subprocess.run(['git', 'for-each-ref', '--format=%(objectname:short=8)', 'refs/remotes/origin/*/e51-reserve-017dSEpW'], capture_output=True, text=True).stdout.split()}; "
        f"commits after {A[:8]}: {subprocess.run(['git', 'log', '--format=%h %s', f'{A}..{H}'], capture_output=True, text=True).stdout.strip().splitlines()}")
    say(f"0.2 files at {A[:8]} {len(la)}, at {H[:8]} {len(lh)}; added: {sorted(set(lh) - set(la))}; removed: {sorted(set(la) - set(lh))}")
    changed = [p for p in la if p in lh and blob(A, p) != blob(H, p)]
    say(f"0.3 paths present at both whose blob differs: {changed}")
    kinds = {"receipts of the eight runs": [p for p in la if re.search(r"e55_(unrelated|conflict)_(blind|shut)_[SB]_73129\.json$", p)],
             "check receipts and dry receipts": [p for p in la if "/checks/" in p and p.endswith(".json")],
             "weight files": [p for p in la if p.endswith(".pt")], "logs": [p for p in la if "/logs/" in p],
             "files of generated bytes": [p for p in la if "_reduction_bytes_" in p and "/checks/" not in p],
             "start receipts": [p for p in la if re.search(r"e55_start_\w+\.json$", p)]}
    for k, v in kinds.items():
        same = sum(blob(A, p) == blob(H, p) for p in v)
        say(f"0.4 {k}: {len(v)} paths, the same blob at both commits {same}")
    say(f"0.5 exp55_reserve.py the same blob at both: {blob(A, M + 'exp55_reserve.py') == blob(H, M + 'exp55_reserve.py')}; e55_read.py the same: {blob(A, M + 'e55_read.py') == blob(H, M + 'e55_read.py')}")
    listed = show(E55 + "E55_COMMITTED_FILES.txt").decode().split()
    lst = lsn(H, E55, M + "exp55_reserve.py", M + "e55_read.py")
    say(f"0.6 E55_COMMITTED_FILES.txt at the head lists {len(listed)} paths; git ls-tree at the head gives {len(lst)}; equal as sets: {set(listed) == set(lst)}; "
        f"listed but absent {sorted(set(listed) - set(lst))}; present but not listed {sorted(set(lst) - set(listed))}")
    clock(0, t0)
    return lh


# ================================================================ section 1
def section1(files):
    t0 = time.time(); say("===== SECTION 1. The receipts of the four runs of `shut` =====")
    say(f"1.0 receipts of `shut` runs: {sum(bool(re.search(r'e55_(unrelated|conflict)_shut_[SB]_73129[.]json$', f)) for f in files)}; weight files of `shut` runs "
        f"{sum(bool(re.search(r'_shut_[SB]_73129_row\d+[.]pt$', f)) for f in files)}; files under an `interrupted` folder {sum('/interrupted/' in f for f in files)}")
    src = {n: show(M + n) for n in ("exp55_reserve.py", "exp51_reserve.py", "e55_read.py")}
    say(f"1.1 SHA-256 at the head: exp55_reserve.py {sha(src['exp55_reserve.py'])[:16]}; exp51_reserve.py {sha(src['exp51_reserve.py'])[:16]} (the brief's 84d0dad7: "
        f"{sha(src['exp51_reserve.py']) == '84d0dad7f6e5eeb15c4518a8c591eb5fc043d72b17926913a3b3fde690d1f34a'})")
    ks = {r: jshow(f"{E51}e51_start_keyed_{r}_{SEED}.json") for r in RULES}
    st = {r: jshow(f"{E55}e55_start_shut_{r}_{SEED}.json") for r in RULES}
    cache = {}
    for (task, rule), rec in REC.items():
        s, c = rec["settings"], rec["complete"]; p = s["provenance"]
        losses = [x["loss_since_last_row"] for x in rec["rows"][1:]]
        ok = dict(
            complete=bool(c) and c["updates"] == 16384 and c["optimizer_steps"] == [16384] and c["items"] == {task: 16384 * 64, "first_family": 0},
            rows=[x["updates"] for x in rec["rows"]] == list(ROWS55) and s["rows"] == list(ROWS55) and s["committed_rows"] == list(KEEP),
            losses_finite=all(isinstance(v, float) and math.isfinite(v) for v in losses),
            arm_task_rule_seed=(s["experiment"], s["part"], s["arm"], s["start_form"], s["task"], s["rule"], s["seed"], s["local"], s["updates"]) == ("E55", "first", "shut", "shut", task, rule, SEED, rule == "S", 16384),
            forward=(s["keyed_forward"] is False and s["forward"] == "exp51_reserve.fwd with KEY['in_use'] = None (no added line)" and s["rule_applied"] is True and s["restore_book"] is True),
            gate_bias=s["gate_bias"] == 10.0,
            rule_rows="update-gate rows of the reserve (384-511) and rows 256-511 of lm and l2" in s["rule_rows"],
            start=(s["start"]["start_file"] == f"e51_start_keyed_{rule}_{SEED}.pt" and s["start"]["start_sha256"] == ks[rule]["weights_sha256"] and s["start"]["start_tensors_sha256"] == ks[rule]["tensors_sha256"]
                   and s["start_e55"]["form"] == "shut" and s["start_e55"]["sha256"] == st[rule]["sha256"] and s["start_e55"]["tensors_sha256"] == st[rule]["tensors_sha256"] == s["start_tensors_sha256"]),
            optimizer=all(g["weight_decay"] == 0.0 and g["lr"] == 0.001 and g["betas"] == [0.9, 0.98] for g in s["optimizer_groups_read_back"] + c["optimizer_groups_read_back"]) and s["decay"] == 0.0 and s["batch"] == 64,
            stream=s["stream"] == "the new task's stream alone",
            end=c["data_digest"] == rec["rows"][-1]["data_digest"] and c["tensors_sha256"] == rec["rows"][-1]["tensors_sha256"],
            weights_named=[x.get("weights") for x in rec["rows"] if x.get("weights")] == [f"e55_{task}_shut_{rule}_{SEED}_row{g}.pt" for g in KEEP],
            git_clean=p["git"]["sources_modified_vs_commit"] == [] and p["git"]["e51_sources_modified_vs_commit"] == [] and p["git"]["e55_sources_modified_vs_commit"] == [],
        )
        bad = []
        for grp in ("sources_sha256", "e51_sources_sha256", "e55_sources_sha256"):
            for f, h in p[grp].items():
                if f not in cache:
                    cache[f] = sha(show(M + f))
                if cache[f] != h:
                    bad.append(f)
        ok["source_hashes"] = not bad
        say(f"1.2 {task} {rule}: " + ", ".join(f"{k} {v}" for k, v in ok.items()) + f"; sources hashed {sum(len(p[g]) for g in ('sources_sha256', 'e51_sources_sha256', 'e55_sources_sha256'))}, "
            f"differing {bad}; commit {p['git']['commit'][:8]}; command '{p['command']}'; torch {p['host']['torch']}, threads {p['host']['torch_threads']}; cpu {c['cpu_total']:.0f} s; "
            f"loss at row 16,384 {losses[-1]:.4g}, largest mean loss of an interval {max(losses):.4g}")
    cm = sorted({rec["settings"]["provenance"]["git"]["commit"] for rec in REC.values()})
    anc = {c[:8]: subprocess.run(["git", "merge-base", "--is-ancestor", c, H]).returncode == 0 for c in cm}
    same = {c[:8]: blob(c, M + "exp55_reserve.py") == blob(H, M + "exp55_reserve.py") for c in cm}
    by = {f"{t} {r}": REC[(t, r)]["settings"]["provenance"]["git"]["commit"][:8] for t, r in RUNS}
    say(f"1.3 commits recorded by the runs of `shut` {by}; ancestors of the head {anc}; exp55_reserve.py the same blob there {same}")
    allcpu = {f"{a} {t} {r}": jshow(f"{E55}e55_{t}_{a}_{r}_{SEED}.json")["complete"]["cpu_total"] for a in ("blind", "shut") for t, r in RUNS}
    say(f"1.4 CPU-seconds of the eight runs {', '.join(f'{k} {v:.0f}' for k, v in allcpu.items())}; smallest {min(allcpu.values()):.0f}, largest {max(allcpu.values()):.0f}; "
        f"sum {sum(allcpu.values()) / 3600:.3f} CPU-hours; the four of `shut` {sum(v for k, v in allcpu.items() if k.startswith('shut')) / 3600:.3f}")
    c3 = jshow(f"{E55}checks/e55_check_3_{SEED}.json")
    say(f"1.5 check 3's forecast: {c3['forecast_cpu_hours_eight_runs']:.2f} CPU-hours for the eight runs, that is {3600 * c3['forecast_cpu_hours_eight_runs'] / 8:.0f} CPU-seconds a run; "
        f"the forecast of one arm {c3['forecast_cpu_hours_blind']:.2f} CPU-hours ({3600 * c3['forecast_cpu_hours_blind'] / 4:.0f} s a run, {3600 * c3['forecast_cpu_hours_blind'] / 8:.0f} s if divided by eight)")
    for task in TASKS:
        nxt = V.source(task, False); dg = V.Digest(); mine = {0: dg.value()}
        for g in range(1, 16385):
            dg.add([nxt()[1] for _ in range(64)])
            if g in ROWS55:
                mine[g] = dg.value()
        say(f"1.6 data digests of `{task}` from the script's own stream at the twelve scored rows equal to the receipts of `shut`: "
            f"{[all(row(REC[(task, r)], g)['data_digest'] == mine[g] for g in ROWS55) for r in RULES]} (S, B)")
    for f in [f for f in files if re.search(r"logs/e55_\w+_shut_[SB]_73129\.\d+\.log$", f)]:
        txt = show(f).decode(); tag = re.search(r"(e55_\w+_shut_[SB]_73129)", f).group(1)
        rows_p = [l for l in txt.splitlines() if l.startswith(tag + " row")]
        bad = [w for w in ("nan", "inf ", "Traceback", "Error:", "FAILED", "Killed") if w in txt]
        warn = sorted({l.strip()[:60] for l in txt.splitlines() if "Warning" in l})
        say(f"1.7 {f.split('e55/')[1]}: rows printed {len(rows_p)}, line of completion {(tag + ' complete') in txt}, words of failure {bad}, warnings {warn}")
    for n in (1, 2, 3):
        ck = jshow(f"{E55}checks/e55_check_{n}_{SEED}.json"); say(f"1.8 check {n}: ok {ck['ok']}")
    clock(1, t0)
    return src


# ================================================================ section 2
def section2():
    t0 = time.time(); say("===== SECTION 2. The start `shut` =====")
    say(f"2.0 free disk before the first weight file {shutil.disk_usage('/home/user').free / 1e9:.2f} GB")
    ck1 = jshow(f"{E55}checks/e55_check_1_{SEED}.json"); out = {}
    for rule in RULES:
        Wk, shk = load_weights(f"{E51}e51_start_keyed_{rule}_{SEED}.pt")
        D, _ = load_weights(f"{E51}e51_start_keyed_{rule}_{SEED}_directions.pt")
        dirs = {m: D[f"directions/{m}"] for m in MATS}; in_use = D["in_use"].astype(bool)
        Ws, shs = load_weights(f"{E55}e55_start_shut_{rule}_{SEED}.pt"); srec = jshow(f"{E55}e55_start_shut_{rule}_{SEED}.json")
        say(f"2.1 rule {rule}: keyed start file hash equal to E51's receipt {shk == jshow(f'{E51}e51_start_keyed_{rule}_{SEED}.json')['weights_sha256']}; start shut: file hash equal "
            f"to its receipt {shs == srec['sha256']}, tensors hash equal {V.tensors_sha(Ws) == srec['tensors_sha256']}; gate bias recorded {srec['gate_bias']}")
        rb = {"rnn.weight_ih_l0": Z, "rnn.bias_ih_l0": Z, "rnn.weight_hh_l0": Z, "rnn.bias_hh_l0": Z, "lm.weight": ADDED, "lm.bias": ADDED, "l2.weight": ADDED, "l2.bias": ADDED}
        outside, changed = True, {}
        for k in Wk:
            diff = (Ws[k] != Wk[k]).reshape(len(Wk[k]), -1).any(1); ins = np.zeros(len(Wk[k]), bool)
            if k in rb:
                ins[rb[k]] = True
            outside &= not diff[~ins].any()
            if diff.any():
                changed[k] = int(diff.sum())
        zeros = not Ws["rnn.weight_hh_l0"][Z][:, :U0].any() and not Ws["rnn.bias_hh_l0"][Z].any() and not Ws["lm.weight"][ADDED][:, :U0].any() and not Ws["lm.bias"][ADDED].any()
        say(f"2.2 (a) every entry outside the four blocks bit-equal to the keyed start {bool(outside)}; rows that differ, by tensor {changed}; entries of `zero_blocks` exactly zero {zeros}")
        rem = {}
        for mn in MATS:
            Dm = block(Ws, mn, SHUTBLOCKS[mn])
            if mn == "ih":
                Dm[:, -1] -= 10.0
            rem[mn] = float(np.abs(Dm @ dirs[mn]).max())
        say("2.3 (b) largest |D @ Vm| at the four blocks (10 taken from the bias of the gate rows of the input map): " + ", ".join(f"{m} {v:.1e}" for m, v in rem.items())
            + "; check 1's receipt: " + ", ".join(f"{m} {v:.1e}" for m, v in ck1["starts"][f"shut {rule}"]["b_largest_D_at_Vm"].items()))
        Wn = {k: v.copy() for k, v in Wk.items()}
        for mn, rows in SHUTBLOCKS.items():
            wk, bk = MATS[mn]; Dm = away(block(Wn, mn, rows), dirs[mn]); Wn[wk][rows] = Dm[:, :-1].astype(np.float32); Wn[bk][rows] = Dm[:, -1].astype(np.float32)
        Wn["rnn.weight_hh_l0"][np.ix_(Z, np.arange(U0))] = 0; Wn["rnn.bias_hh_l0"][Z] = 0; Wn["lm.weight"][np.ix_(ADDED, np.arange(U0))] = 0; Wn["lm.bias"][ADDED] = 0
        Wn["rnn.bias_ih_l0"][Z] = Wn["rnn.bias_ih_l0"][Z] + np.float32(10.0)
        nd = sum(int((Wn[k] != Ws[k]).sum()) for k in Ws)
        say(f"2.4 the start built again in numpy (projection in double precision, stored in single, 10 added in single): entries that differ from the saved start {nd}; "
            f"bias of the gate rows of the reserve in the input map, smallest {Ws['rnn.bias_ih_l0'][Z].min():.4f}, largest {Ws['rnn.bias_ih_l0'][Z].max():.4f}")
        nb, nk = V.Net(Ws), V.Net(Wk)
        gh, gs = gates(nb, held), gates(nb, SAMPLE)
        dlg = max(float(np.abs(V.answer_logits(nb, its) - V.answer_logits(nk, its, in_use=in_use)).max()) for its in (held, SAMPLE))
        fb = {t: gates(nb, (lambda s: [s()[1] for _ in range(64)])(V.source(t, False)))["largest_state"] for t in TASKS}
        gens, gk = V.score_first_family(nb, S1, HELD)[1], V.score_first_family(nk, S1, HELD, in_use=in_use)[1]
        ndiff = sum(a != b for n in gens for a, b in zip(gens[n], gk[n]))
        r = ck1["starts"][f"shut {rule}"]
        say(f"2.5 (c) on the held-out pairs and check 1's 2,000 items: largest 1 - z {max(gh['largest_opening'], gs['largest_opening']):.4e} (receipt "
            f"{max(r['c_gates_on_heldout_k12']['largest_opening'], r['c_gates_on_sample_2000']['largest_opening']):.4e}); largest state {max(gh['largest_state'], gs['largest_state']):.3e} "
            f"(receipt {max(r['c_gates_on_heldout_k12']['largest_reserve_state'], r['c_gates_on_sample_2000']['largest_reserve_state']):.3e}); largest added unit "
            f"{max(gh['largest_added'], gs['largest_added']):.3e} (receipt {max(r['c_gates_on_heldout_k12']['largest_added_unit_at_answer_positions'], r['c_gates_on_sample_2000']['largest_added_unit_at_answer_positions']):.3e}); "
            f"largest output difference from the keyed start under the keyed forward {dlg:.3e} (receipt {r['c_largest_output_difference_from_keyed_start']:.3e}); first batch of each new task, largest state "
            + ", ".join(f"{t} {v:.2e}" for t, v in fb.items()) + " (receipt " + ", ".join(f"{t} {v:.2e}" for t, v in r["c_largest_reserve_state_first_batch_of_new_task"].items()) + ")")
        say(f"2.6 (d) of the {sum(len(v) for v in gens.values())} generated answers, those that differ from the keyed start's under the keyed forward: {ndiff} (receipt {r['d_answers_differing_from_keyed_start']})")
        gshut = V.score_first_family(nb, S1, HELD, shut_all=True)[1]
        nds = sum(a != b for n in gens for a, b in zip(gshut[n], gk[n]))
        say(f"2.7 the same start with the reserve held shut on every item: answers that differ from the keyed start's under the keyed forward {nds}")
        nb32 = V.Net(Ws, np.float32)
        out[rule] = dict(W=Ws, dirs=dirs, in_use=in_use, gens=gens, gshut=gshut, logits=V.answer_logits(nb, held), g40=V.generate(nb, LEN40),
                         gens32=V.score_first_family(nb32, S1, HELD)[1], g4032=V.generate(nb32, LEN40), eig=B.radius(Ws))
        out[rule]["fig"] = dict(ff=gh, ff40=gates(nb, LEN40), new={t: gates(nb, newpanel(t)) for t in TASKS}, right40=V.exact(out[rule]["g40"], LEN40))
        del Wk, Wn, D
    clock(2, t0)
    return out


PAN = {t: V.new_panels(t) for t in TASKS}


def newpanel(task):
    return [it for n in (("add97", "sub97") if task == "conflict" else ("held",)) for it in PAN[task][n]]


# ================================================================ section 3
FIGS = {"n": 0, "diffs": []}


def cmp(label, rep, mine):
    FIGS["n"] += 1
    if isinstance(mine, (tuple, list, dict)):
        ok = rep == mine
    elif isinstance(mine, (bool, str)) or mine is None:
        ok = str(rep).strip() == str(mine)
    elif isinstance(rep, (int, float)) and not isinstance(rep, bool):
        ok = abs(rep - mine) <= 1e-9 * max(1.0, abs(mine))
    else:
        ok = B.agrees(str(rep).replace(",", ""), mine)
    if not ok:
        FIGS["diffs"].append(f"{label}: reported {rep!r}, recomputed {mine!r}")
    return ok


COLS19 = ["updates", "reduction", "d", "held", "d40", "nt", "ns", "new_open", "new_mean", "new_1z", "ff_1z", "ff_open", "ff_state", "ff_added", "out", "eig", "ff40_exact", "ff40_state", "ff40_1z"]


def summary_cells(rows, w):
    fl = first_learned([(x["updates"], x["nt"]) for x in rows])
    return [w, max(x["d"] for x in rows), max(x["held"] for x in rows), rows[0]["reduction"], rows[-1]["reduction"], str(fl) if fl else "not learned",
            rows[-1]["nt"], rows[-1]["ns"], max(x["ff_state"] for x in rows), max(x["out"] for x in rows), rows[-1]["d40"]]


def section3():
    t0 = time.time(); say("===== SECTION 3. The worker's figures recomputed from the receipts =====")
    mine = MINE
    blindrec = {(t, r): B.recompute(jshow(f"{E55}e55_{t}_blind_{r}_{SEED}.json")) for t, r in RUNS}
    md = show(f"{E55}E55_TABLES_all.md").decode(); rep = show(f"{E55}E55_REPORT.md").decode(); rd = jshow(f"{E55}E55_READ_all.json")["runs"]
    n_before = FIGS["n"]
    for sec in re.split(r"^### ", md, flags=re.M)[1:]:
        m = re.match(r"(\w+), rule (\w), arm `shut`", sec)
        if not m:
            continue
        task, rule = m.group(1), m.group(2)
        lines = [l for l in sec.splitlines() if re.match(r"^\| \d", l)]
        for l, x in zip(lines[:12], mine[(task, rule)]):
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            assert int(cells[0]) == x["updates"] and len(cells) == 19
            for c, col in zip(cells[1:], COLS19[1:]):
                cmp(f"tables {task} {rule} row {x['updates']} {col}", c, x[col])
        ref = {a: B.recompute(jshow(f"{E51}e51_{task}_{a}_{rule}_{SEED}.json")) for a in ("keyed", "open", "rehearse")}
        for l in lines[12:]:
            cells = [c.strip() for c in l.strip().strip("|").split("|")]; g = int(cells[0])
            for c, a in zip(cells[1:], ("keyed", "open", "rehearse")):
                x = [y for y in ref[a] if y["updates"] == g][0]
                for v, col in zip(c.split(";"), ("reduction", "nt", "ns")):
                    cmp(f"tables {task} {rule} E51 {a} row {g} {col}", v.strip(), x[col])
    say(f"3.1 E55_TABLES_all.md, the four tables of `shut` (twelve rows by eighteen figures) and E51's rows beside them: {FIGS['n'] - n_before} figures compared")
    # the summary tables: E55_TABLES_all.md and section 4 of the report (all eight rows)
    for name, text in (("tables", md), ("report", rep)):
        n_before = FIGS["n"]
        for l in text.splitlines():
            m = re.match(r"^\| `(blind|shut)` \| (\w+) \| (\w) \|", l)
            if not m or l.count("|") < 14:
                continue
            arm, task, rule = m.groups(); rows = (mine if arm == "shut" else blindrec)[(task, rule)]
            w = words(rows)
            if arm == "blind" and w == "kept exactly" and max(x["d40"] for x in rows) > 3:
                w = "kept on items of up to 22 positions and not on items of 50"
            cells = [c.strip() for c in l.strip().strip("|").split("|")][3:]
            r0, rl = cells[3].split("/")
            for c, v in zip([cells[0], cells[1], cells[2], r0, rl] + cells[4:], summary_cells(rows, w)):
                cmp(f"{name} summary {arm} {task} {rule}", c, v)
        say(f"3.2 summary table of the eight runs in {name}: {FIGS['n'] - n_before} figures or words compared")
    # the report's table of the largest state on the first family's held-out items at rows 0, 1,024, 4,096, 16,384
    n_before = FIGS["n"]
    for l in rep.splitlines():
        m = re.match(r"^\| (blind|shut) \| (\w+) \| (\w) \| 0: ", l)
        if m:
            arm, task, rule = m.groups(); rows = (mine if arm == "shut" else blindrec)[(task, rule)]
            for g, v in re.findall(r"(\d+): ([0-9.e-]+)", l):
                cmp(f"report state table {arm} {task} {rule} row {g}", v, [x for x in rows if x["updates"] == int(g)][0]["ff_state"])
    # section 4.1 of the report (the arm blind by row)
    qmap = {"largest reserve state, 40 letters": "ff40_state", "eigenvalue of 0.5 I + 0.25 Wn": "eig", "right answers of 300, 40 letters": "ff40_exact", "largest 1 − z, 40 letters": "ff40_1z"}
    for l in rep.splitlines():
        m = re.match(r"^\| (unrelated|conflict) \| (\w) \| ([^|]+) \|", l)
        if m and m.group(3).strip() in qmap:
            task, rule, q = m.group(1), m.group(2), qmap[m.group(3).strip()]
            cells = [c.strip() for c in l.strip().strip("|").split("|")][3:]
            for c, x in zip(cells, blindrec[(task, rule)]):
                cmp(f"report 4.1 {task} {rule} {q} row {x['updates']}", c, x[q])
    say(f"3.3 the report's table of states by row and its table of section 4.1: {FIGS['n'] - n_before} figures compared")
    # E55_READ_all.json, the entries of `shut`, leaf by leaf
    n_before = FIGS["n"]
    for (task, rule) in RUNS:
        v = rd[f"shut {task} {rule}"]; rows = mine[(task, rule)]; rec = REC[(task, rule)]
        fl = first_learned([(x["updates"], x["nt"]) for x in rows]); ffr, ffc = first_fail(rows)
        L = f"READ shut {task} {rule}"
        cmp(L + " word", v["word"], words(rows)); cmp(L + " word_40_letters", v["word_40_letters"], None)
        cmp(L + " largest_d", v["largest_d"], max(x["d"] for x in rows)); cmp(L + " largest_d_pct", v["largest_d_pct"], 100 * max(x["d"] for x in rows) / N0)
        cmp(L + " largest_heldout_differ", v["largest_heldout_differ"], max(x["held"] for x in rows))
        cmp(L + " reduction_row0", v["reduction_row0"], rows[0]["reduction"]); cmp(L + " reduction_last", v["reduction_last"], rows[-1]["reduction"])
        f = v["first_failure_of_kept"]; cmp(L + " first failure row", f["row"], ffr)
        nums = sorted(int(n) for c in f["conditions"] for n in re.findall(r"(?<![\d.])(\d+)(?![\d.])", c.split(">")[0]))
        cmp(L + " first failure conditions (count and numbers)", (len(f["conditions"]), nums), (len(ffc), sorted(int(abs(n)) for _, n in ffc)))
        for k, val in f["gates_on_first_family"].items():
            cmp(L + f" first failure gates {k}", val, row(rec, ffr)["gates_on_first_family"][k])
        for key, col in (("differ_40_letters_by_row", "d40"), ("right_40_letters_by_row", "ff40_exact"), ("reserve_state_40_letters_by_row", "ff40_state"),
                         ("largest_opening_40_letters_by_row", "ff40_1z"), ("eigenvalue_by_row", "eig")):
            for x in rows:
                cmp(L + f" {key} {x['updates']}", v[key][str(x["updates"])], x[col])
        cmp(L + " learned_from_row", v["learned_from_row"], fl); cmp(L + " never_taught_last", v["never_taught_last"], rows[-1]["nt"])
        cmp(L + " never_taught_shut_last", v["never_taught_shut_last"], rows[-1]["ns"])
        cmp(L + " largest_reserve_state_first_family", v["largest_reserve_state_first_family"], max(x["ff_state"] for x in rows))
        cmp(L + " largest_output_change", v["largest_output_change"], max(x["out"] for x in rows))
        cmp(L + " cells_open_at_some_position_new_task_last", v["cells_open_at_some_position_new_task_last"], rows[-1]["new_open"])
        cmp(L + " cpu_total", v["cpu_total"], rec["complete"]["cpu_total"])
    say(f"3.4 E55_READ_all.json, the four entries of `shut`: {FIGS['n'] - n_before} values compared")
    # the report's prose and small tables (sections 3, 5, 6, 8, 9)
    n_before = FIGS["n"]
    g = lambda t, r: mine[(t, r)]
    at = lambda t, r, u, c: [x for x in g(t, r) if x["updates"] == u][0][c]
    mx = lambda t, r, c: max(x[c] for x in g(t, r))
    ck1, ck2, ck3 = (jshow(f"{E55}checks/e55_check_{n}_{SEED}.json") for n in (1, 2, 3))
    st1 = ck1["starts"]; mxp = lambda s, k: max(s["c_gates_on_heldout_k12"][k], s["c_gates_on_sample_2000"][k])
    P = [("3 (b) D@Vm at most", "1.9e-7", max(v for s in st1.values() for v in s["b_largest_D_at_Vm"].values()))]
    for r, a, b_, c_, d_ in (("S", "2.4e-7", "2.4e-7", "1.1e-5", "0.153"), ("B", "3.5e-7", "3.3e-7", "7.6e-6", "0.146")):
        s = st1[f"blind {r}"]
        P += [(f"3 (c) blind {r} state", a, mxp(s, "largest_reserve_state")), (f"3 (c) blind {r} added", b_, mxp(s, "largest_added_unit_at_answer_positions")),
              (f"3 (c) blind {r} output difference", c_, s["c_largest_output_difference_from_keyed_start"]), (f"3 (c) blind {r} first batch unrelated", d_, s["c_largest_reserve_state_first_batch_of_new_task"]["unrelated"])]
    P += [("3 (c) blind S first batch conflict", "0.070", st1["blind S"]["c_largest_reserve_state_first_batch_of_new_task"]["conflict"]),
          ("3 (c) blind B first batch conflict", "0.087", st1["blind B"]["c_largest_reserve_state_first_batch_of_new_task"]["conflict"])]
    for r, a, b_, c_ in (("S", "5.6e-4", "2.9e-4", "2.4e-3"), ("B", "5.9e-4", "3.6e-4", "2.5e-3")):
        s = st1[f"shut {r}"]
        P += [(f"3 (c) shut {r} 1 - z", "4.542e-5", mxp(s, "largest_opening")), (f"3 (c) shut {r} state", a, mxp(s, "largest_reserve_state")),
              (f"3 (c) shut {r} added", b_, mxp(s, "largest_added_unit_at_answer_positions")), (f"3 (c) shut {r} output difference", c_, s["c_largest_output_difference_from_keyed_start"])]
    for f, v in (("blind S", 0), ("blind B", 0), ("shut S", 1), ("shut B", 0)):
        P.append((f"3 (d) {f}", str(v), st1[f]["d_answers_differing_from_keyed_start"]))
    R2 = ck2["runs"]
    P += [("3, 2 old rows' change @ Vm at most", "6.7e-9", max(v for x in R2.values() for v in x["a_old_rows_change_at_Vm"].values())),
          ("3, 2 blind D@Vm at most", "4.3e-9", max(v for k, x in R2.items() if k.startswith("blind") for v in x["d_blind_rows_at_Vm"].values())),
          ("3, 2 blind state at most", "4.2e-7", max(x[p]["largest_reserve_state"] for k, x in R2.items() if k.startswith("blind") for p in ("gates_on_heldout_k12", "gates_on_sample_2000"))),
          ("3, 2 blind output change at most", "1.9e-5", max(x["largest_change_of_an_output"] for k, x in R2.items() if k.startswith("blind"))),
          ("3, 2 blind input change from", "0.14", min(x["input_map_change_on_new_task_bytes"]["largest"] for k, x in R2.items() if k.startswith("blind"))),
          ("3, 2 blind input change to", "0.20", max(x["input_map_change_on_new_task_bytes"]["largest"] for k, x in R2.items() if k.startswith("blind"))),
          ("3, 2 shut 1 - z", "4.54e-5", max(x[p]["largest_opening"] for k, x in R2.items() if k.startswith("shut") for p in ("gates_on_heldout_k12", "gates_on_sample_2000"))),
          ("3, 2 shut state at most", "8.1e-4", max(x[p]["largest_reserve_state"] for k, x in R2.items() if k.startswith("shut") for p in ("gates_on_heldout_k12", "gates_on_sample_2000"))),
          ("3, 2 shut output change from", "0.008", min(x["largest_change_of_an_output"] for k, x in R2.items() if k.startswith("shut"))),
          ("3, 2 shut output change to", "0.026", max(x["largest_change_of_an_output"] for k, x in R2.items() if k.startswith("shut"))),
          ("3, 2 shut input change from", "0.079", min(x["input_map_change_on_new_task_bytes"]["largest"] for k, x in R2.items() if k.startswith("shut"))),
          ("3, 2 shut input change to", "0.171", max(x["input_map_change_on_new_task_bytes"]["largest"] for k, x in R2.items() if k.startswith("shut")))]
    for k, v in (("shut S unrelated", "3.8e-9"), ("shut S conflict", "2.0e-10"), ("shut B unrelated", "4.6e-8"), ("shut B conflict", "5.5e-9")):
        P.append((f"3, 2 median gradient {k}", v, R2[k]["c_gradient_first_update"]["median"]))
    cpu = [jshow(f"{E55}e55_{t}_{a}_{r}_{SEED}.json")["complete"]["cpu_total"] for a in ("blind", "shut") for t, r in RUNS]
    P += [("3.3 forecast", "6.1", ck3["forecast_cpu_hours_eight_runs"]), ("3.3 took", "6.15", sum(cpu) / 3600),
          ("9 costs, smallest", "2536", min(cpu)), ("9 costs, largest", "2992", max(cpu)), ("9 forecast a run", "1370", 3600 * ck3["forecast_cpu_hours_eight_runs"] / 8)]
    ff1z = [x["ff_1z"] for k in RUNS for x in g(*k)]
    P += [("5 shut 1 - z on the first family, lowest", "4.5e-5", min(ff1z)), ("5 shut 1 - z on the first family, highest", "6.5e-5", max(ff1z)),
          ("5 shut no cell opens on the first family", "0", max(x["ff_open"] for k in RUNS for x in g(*k)))]
    t, r = "unrelated", "S"
    P += [("5 unrelated S never-taught", "72.7", at(t, r, 16384, "nt")), ("5 unrelated S open 16384", "126", at(t, r, 16384, "new_open")), ("5 unrelated S open 8192", "125", at(t, r, 8192, "new_open")),
          ("5 unrelated S open 4096", "16", at(t, r, 4096, "new_open")), ("5 unrelated S 1-z 256", "0.002", at(t, r, 256, "new_1z")), ("5 unrelated S 1-z 1024", "0.34", at(t, r, 1024, "new_1z")),
          ("5 unrelated S 1-z 4096", "1.0", at(t, r, 4096, "new_1z")), ("5 unrelated S 1-z 8192", "1.0", at(t, r, 8192, "new_1z")), ("5 unrelated S 1-z 16384", "1.0", at(t, r, 16384, "new_1z")),
          ("5 unrelated S largest d", "135", mx(t, r, "d")), ("5 unrelated S largest d %", "0.43", 100 * mx(t, r, "d") / N0), ("5 unrelated S held", "17", mx(t, r, "held")),
          ("5 unrelated S reduction last", "92.56", at(t, r, 16384, "reduction")), ("5 unrelated S reduction row 0", "92.40", at(t, r, 0, "reduction")),
          ("5 unrelated S first failure row", "64", first_fail(g(t, r))[0]), ("5 unrelated S d(64)", "42", at(t, r, 64, "d")), ("5 unrelated S held(64)", "2", at(t, r, 64, "held")),
          ("5 unrelated S 1-z at 64", "4.6e-5", at(t, r, 64, "ff_1z")), ("5 unrelated S open at 64", "0", at(t, r, 64, "ff_open")), ("5 unrelated S state at 64", "1.0e-3", at(t, r, 64, "ff_state")),
          ("5 unrelated S added at 64", "0.0086", at(t, r, 64, "ff_added"))]
    t, r = "unrelated", "B"
    P += [("5 unrelated B learned", "4096", str(first_learned([(x["updates"], x["nt"]) for x in g(t, r)]))), ("5 unrelated B never-taught", "99.7", at(t, r, 16384, "nt")),
          ("5 unrelated B %", "0.11", 100 * mx(t, r, "d") / N0), ("5 unrelated B largest d", "34", mx(t, r, "d")), ("5 unrelated B held", "1", mx(t, r, "held")),
          ("5 unrelated B reduction last", "95.29", at(t, r, 16384, "reduction")), ("5 unrelated B reduction 0", "95.25", at(t, r, 0, "reduction")),
          ("5 unrelated B first failure", "1024", first_fail(g(t, r))[0]), ("5 unrelated B d(1024)", "34", at(t, r, 1024, "d")), ("5 unrelated B 1-z at 1024", "4.6e-5", at(t, r, 1024, "ff_1z")),
          ("5 unrelated B state at 1024", "1.0e-3", at(t, r, 1024, "ff_state")), ("5 unrelated B added at 1024", "0.044", at(t, r, 1024, "ff_added")),
          ("5 unrelated B cells open 16384", "128", at(t, r, 16384, "new_open")), ("5 unrelated B shut", "0.0", at(t, r, 16384, "ns"))]
    t, r = "conflict", "S"
    P += [("5 conflict S never-taught", "19.8", at(t, r, 16384, "nt")), ("5 conflict S cells open at any row", "0", mx(t, r, "new_open")), ("5 conflict S new 1-z below 0.001", "True", str(mx(t, r, "new_1z") < 0.001)),
          ("5 conflict S largest d", "925", mx(t, r, "d")), ("5 conflict S %", "2.92", 100 * mx(t, r, "d") / N0), ("5 conflict S held", "72", mx(t, r, "held")),
          ("5 conflict S reduction last", "93.00", at(t, r, 16384, "reduction")), ("5 conflict S first failure", "32", first_fail(g(t, r))[0]), ("5 conflict S d(32)", "37", at(t, r, 32, "d")),
          ("5 conflict S held(32)", "2", at(t, r, 32, "held")), ("5 conflict S 1-z at 32", "4.5e-5", at(t, r, 32, "ff_1z")), ("5 conflict S state at 32", "9.7e-4", at(t, r, 32, "ff_state")),
          ("5 conflict S added at 32", "0.0048", at(t, r, 32, "ff_added"))]
    t, r = "conflict", "B"
    P += [("5 conflict B learned", "8192", str(first_learned([(x["updates"], x["nt"]) for x in g(t, r)]))), ("5 conflict B never-taught", "92.9", at(t, r, 16384, "nt")),
          ("5 conflict B %", "0.77", 100 * mx(t, r, "d") / N0), ("5 conflict B largest d", "244", mx(t, r, "d")), ("5 conflict B held", "30", mx(t, r, "held")),
          ("5 conflict B reduction last", "95.26", at(t, r, 16384, "reduction")), ("5 conflict B first failure", "32", first_fail(g(t, r))[0]), ("5 conflict B held(32)", "3", at(t, r, 32, "held")),
          ("5 conflict B 1-z at 32", "4.5e-5", at(t, r, 32, "ff_1z")), ("5 conflict B state at 32", "9.7e-4", at(t, r, 32, "ff_state")), ("5 conflict B added at 32", "0.0040", at(t, r, 32, "ff_added")),
          ("5 conflict B cells open 16384", "58", at(t, r, 16384, "new_open")), ("5 conflict B shut", "34.4", at(t, r, 16384, "ns"))]
    dev = max(abs(x["reduction"] - g(*k)[0]["reduction"]) for k in RUNS for x in g(*k))
    devk = max(RUNS, key=lambda k: max(abs(x["reduction"] - g(*k)[0]["reduction"]) for x in g(*k)))
    P += [("8 P55.4b largest deviation", "1.43", dev), ("8 P55.4b where", "conflict S", f"{devk[0]} {devk[1]}"), ("8 P55.5 largest", "6.5e-5", max(ff1z)),
          ("8 P55.6 unrelated B", "128", at("unrelated", "B", 16384, "new_open")), ("8 P55.6 conflict B", "58", at("conflict", "B", 16384, "new_open")),
          ("8 P55.2 S", "72.7", at("unrelated", "S", 16384, "nt")), ("8 P55.3 S", "19.8", at("conflict", "S", 16384, "nt"))]
    # section 6 of the report: the two arms side by side, and the rows of learning beside E51's
    e51l = {(t, r, a): first_learned([(x["updates"], x["nt"]) for x in B.recompute(jshow(f"{E51}e51_{t}_{a}_{r}_{SEED}.json"))]) for t, r in RUNS for a in ("keyed", "open")}
    for l in rep.splitlines():
        m = re.match(r"^\| (unrelated|conflict) \| (\w) \| (kept|partly)", l)
        if m:
            t, r = m.group(1), m.group(2); cells = [c.strip() for c in l.strip().strip("|").split("|")][2:]
            exp = []
            for arm, rows in (("blind", blindrec[(t, r)]), ("shut", g(t, r))):
                w = words(rows)
                if arm == "blind" and w == "kept exactly" and max(x["d40"] for x in rows) > 3:
                    w = "kept on items of up to 22 positions and not on items of 50"
                fl = first_learned([(x["updates"], x["nt"]) for x in rows])
                exp += [f"{w} ({max(x['d'] for x in rows)})", f"{fl:,}" if fl else f"not learned ({rows[-1]['nt']:.1f})", f"{rows[-1]['ns']:.1f}"]
            for c, e in zip(cells, exp):
                cmp(f"report 6 side by side {t} {r}", c, e)
        m = re.match(r"^\| `(blind|shut)`, `(\w+)`, (\w) \| ([\d,]+) \| ([\d,]+) \| ([\d,]+) \|", l)
        if m:
            arm, t, r = m.group(1), m.group(2), m.group(3); rows = (blindrec if arm == "blind" else mine)[(t, r)]
            cmp(f"report 6 learned {arm} {t} {r}", m.group(4), f"{first_learned([(x['updates'], x['nt']) for x in rows]):,}")
            cmp(f"report 6 E51 keyed {t} {r}", m.group(5), f"{e51l[(t, r, 'keyed')]:,}"); cmp(f"report 6 E51 open {t} {r}", m.group(6), f"{e51l[(t, r, 'open')]:,}")
    P += [("6 shut lost", "244", mx("conflict", "B", "d")), ("6 shut lost %", "0.77", 100 * mx("conflict", "B", "d") / N0), ("6 shut held lost", "30", mx("conflict", "B", "held"))]
    for label, r_, v in P:
        cmp("report " + label, r_, v)
    say(f"3.5 the report's prose figures and its tables of section 6: {FIGS['n'] - n_before} figures or words compared")
    # the worker's message
    n_before = FIGS["n"]
    Q = [("eight runs finished at 16,384 updates", "8", sum(jshow(f"{E55}e55_{t}_{a}_{r}_{SEED}.json")["complete"]["updates"] == 16384 for a in ("blind", "shut") for t, r in RUNS)),
         ("CPU-hours", "6.15", sum(cpu) / 3600)]
    bw = {k: words(v) for k, v in blindrec.items()}
    Q += [("blind: kept exactly in all four (by the definition)", "True", str(all(w == "kept exactly" for w in bw.values()))),
          ("blind unrelated learned S", "8192", str(first_learned([(x["updates"], x["nt"]) for x in blindrec[("unrelated", "S")]]))),
          ("blind unrelated learned B", "2048", str(first_learned([(x["updates"], x["nt"]) for x in blindrec[("unrelated", "B")]]))),
          ("blind unrelated shut S", "0.0", blindrec[("unrelated", "S")][-1]["ns"]), ("blind unrelated shut B", "0.0", blindrec[("unrelated", "B")][-1]["ns"]),
          ("blind unrelated B 40 letters differ", "278", blindrec[("unrelated", "B")][-1]["d40"]),
          ("blind conflict S", "69.4", blindrec[("conflict", "S")][-1]["nt"]), ("blind conflict B", "63.2", blindrec[("conflict", "B")][-1]["nt"]),
          ("blind conflict S shut", "34.9", blindrec[("conflict", "S")][-1]["ns"]), ("blind conflict B shut", "28.9", blindrec[("conflict", "B")][-1]["ns"]),
          ("shut: all four partly kept", "True", str(all(words(g(*k)) == "partly kept" for k in RUNS))), ("shut 1 - z at most", "6.5e-5", max(ff1z)),
          ("unrelated S", "72.7", at("unrelated", "S", 16384, "nt")), ("unrelated S cells", "126", at("unrelated", "S", 16384, "new_open")),
          ("unrelated S d", "135", mx("unrelated", "S", "d")), ("unrelated S %", "0.43", 100 * mx("unrelated", "S", "d") / N0), ("unrelated S held", "17", mx("unrelated", "S", "held")),
          ("unrelated B learned", "4096", str(first_learned([(x["updates"], x["nt"]) for x in g("unrelated", "B")]))), ("unrelated B d", "34", mx("unrelated", "B", "d")),
          ("unrelated B %", "0.11", 100 * mx("unrelated", "B", "d") / N0), ("unrelated B held", "1", mx("unrelated", "B", "held")), ("unrelated B shut", "0.0", at("unrelated", "B", 16384, "ns")),
          ("conflict S", "19.8", at("conflict", "S", 16384, "nt")), ("conflict S no cell ever opened", "0", mx("conflict", "S", "new_open")),
          ("conflict S d", "925", mx("conflict", "S", "d")), ("conflict S %", "2.92", 100 * mx("conflict", "S", "d") / N0), ("conflict S held", "72", mx("conflict", "S", "held")),
          ("conflict B learned", "8192", str(first_learned([(x["updates"], x["nt"]) for x in g("conflict", "B")]))), ("conflict B d", "244", mx("conflict", "B", "d")),
          ("conflict B %", "0.77", 100 * mx("conflict", "B", "d") / N0), ("conflict B held", "30", mx("conflict", "B", "held")), ("conflict B shut", "34.4", at("conflict", "B", 16384, "ns")),
          ("conflict B cells", "58", at("conflict", "B", 16384, "new_open"))]
    for (t, r), v in zip(RUNS, ("64", "1024", "32", "32")):
        Q.append((f"kept first fails {t} {r}", v, str(first_fail(g(t, r))[0])))
    Q.append(("reduction within", "1.43", dev))
    for label, r_, v in Q:
        cmp("message " + label, r_, v)
    say(f"3.6 the worker's message: {FIGS['n'] - n_before} figures or words compared (its predictions are compared in section 5)")
    clock(3, t0)
    return blindrec


# ================================================================ section 4
def section4(start):
    t0 = time.time(); say("===== SECTION 4. From the saved weights of `shut` (the plain forward pass in numpy, double precision unless said) =====")
    W4 = {}
    for (task, rule) in RUNS:
        rec = REC[(task, rule)]; st = start[rule]; dirs = st["dirs"]; W0 = st["W"]
        for g in KEEP:
            r = row(rec, g); x = [y for y in MINE[(task, rule)] if y["updates"] == g][0]
            W, sh = load_weights(f"{E55}{r['weights']}", A)
            hashes = (sh == r["weights_sha256"], V.tensors_sha(W) == r["tensors_sha256"])
            net = V.Net(W)
            res, gens = V.score_first_family(net, S1, HELD)
            d = sum(a != b for n in gens if n.startswith("k0_") for a, b in zip(gens[n], st["gens"][n]))
            hd = sum(a != b for n in gens if n.startswith("k12_") for a, b in zip(gens[n], st["gens"][n]))
            right0 = sum(V.exact(gens["k0_" + n], S1[n][0]) for n in S1); right12 = sum(V.exact(gens["k12_" + n], HELD[n]) for n in HELD)
            same = V.same_counts(res, V.receipt_first_family(r))
            rb = ""
            if g in (64, 4096):
                fb = jshow(f"{E55}e55_{task}_shut_{rule}_{SEED}_reduction_bytes_row{g}.json")
                rb = f"; reduction pairs' bytes equal to the file of row {g}: {all([q for q, f in zip(gens['k0_' + n], S1[n][1]) if f] == fb[n] for n in ('add89', 'sub89'))}"
            say(f"4.1 {task} {rule} row {g}: file and tensors hashes equal to the receipt {hashes}; answers that differ from the start's: {d} of 31,684 (receipt {x['d']}), {hd} of 1,200 "
                f"(receipt {x['held']}); right answers {right0} of 31,684 and {right12} of 1,200 (receipt {x['right0']}, {x['right12']}); every count of scores 1 and 2 reproduced {same}{rb}")
            gsh = V.score_first_family(net, S1, HELD, shut_all=True)[1]
            ds = sum(a != b for n in gsh if n.startswith("k0_") for a, b in zip(gsh[n], st["gshut"][n])); hs = sum(a != b for n in gsh if n.startswith("k12_") for a, b in zip(gsh[n], st["gshut"][n]))
            dsp = sum(a != b for n in gsh if n.startswith("k0_") for a, b in zip(gsh[n], st["gens"][n])); hsp = sum(a != b for n in gsh if n.startswith("k12_") for a, b in zip(gsh[n], st["gens"][n]))
            say(f"4.2 {task} {rule} row {g}: with the cells in reserve held shut on every item, the first family's answers that differ from the start's with the reserve held shut: "
                f"{ds} of 31,684 and {hs} of 1,200; from the start's as run: {dsp} and {hsp}")
            ff, ffs = gates(net, held), None
            ff40 = gates(net, LEN40); new = gates(net, newpanel(task)); news = gates(net, newpanel(task), shut_all=True)
            g40 = V.generate(net, LEN40); d40 = sum(a != b for a, b in zip(g40, st["g40"])); r40 = V.exact(g40, LEN40)
            say(f"4.3 {task} {rule} row {g}: first family's held-out pairs (22 positions): largest 1 - z {ff['largest_opening']:.3e} (receipt {x['ff_1z']:.3e}); cells open at some position "
                f"{ff['open_some']} (receipt {x['ff_open']}), by more than 0.5 {ff['open_some_0p5']}; share of positions with a gate open by more than 0.5 {ff['share_pos_any_0p5']:.4f}; largest state "
                f"{ff['largest_state']:.3e} (receipt {x['ff_state']:.3e}), at positions 1, 6, 11, 16, 21: " + ", ".join(f"{ff['profile'][q]:.1e}" for q in (1, 6, 11, 16, 21))
                + f"; largest added unit {ff['largest_added']:.3e} (receipt {x['ff_added']:.3e})")
            say(f"4.4 {task} {rule} row {g}: 40-letter panel (50 positions): answers that differ from the start's {d40} (receipt {x['d40']}), right {r40} of 300 (receipt {x['ff40_exact']}); "
                f"largest 1 - z {ff40['largest_opening']:.3e} (receipt {x['ff40_1z']:.3e}); share of positions open by more than 0.5 {ff40['share_pos_any_0p5']:.4f}; largest state {ff40['largest_state']:.3e} "
                f"(receipt {x['ff40_state']:.3e}), at positions 11, 21, 31, 41, 49: " + ", ".join(f"{ff40['profile'][q]:.1e}" for q in (11, 21, 31, 41, 49)))
            s3 = V.score_new(net, PAN[task]); s3s = V.score_new(net, PAN[task], shut_all=True)
            ps = ("add97", "sub97") if task == "conflict" else ("held",)
            nt = 100 * sum(s3[p]["exact"] for p in ps) / sum(s3[p]["n"] for p in ps); ns = 100 * sum(s3s[p]["exact"] for p in ps) / sum(s3s[p]["n"] for p in ps)
            okn = r["score3_new_task"] == {k: dict(n=v["n"], exact=v["exact"], final=v["final"]) for k, v in s3.items()}
            oks = r["score3_new_task_reserve_shut"] == {k: dict(n=v["n"], exact=v["exact"], final=v["final"]) for k, v in s3s.items()}
            say(f"4.5 {task} {rule} row {g}: never-taught score {nt:.1f} (receipt {x['nt']:.1f}), with the reserve held shut {ns:.1f} (receipt {x['ns']:.1f}); every count of score 3 reproduced: "
                f"as run {okn}, shut {oks}")
            say(f"4.6 {task} {rule} row {g}: the new task's never-taught items: largest 1 - z {new['largest_opening']:.3f} (receipt {x['new_1z']:.3f}); cells open at some position {new['open_some']} "
                f"(receipt {x['new_open']}), by more than 0.5 {new['open_some_0p5']} (receipt {x['new_05']}), by more than 0.1 in the mean {new['open_mean_0p1']} (receipt {x['new_mean']}); "
                f"share of positions at which a gate of the reserve is open by more than 0.5 {new['share_pos_any_0p5']:.4f}, of (position, cell) pairs {new['share_triples_0p5']:.4f}; cells open by more than 0.5 "
                f"at some position of some item {new['cells_ever_0p5']}; largest state {new['largest_state']:.3f}; with the reserve held shut, largest 1 - z {news['largest_opening']:.1f}")
            lg = V.answer_logits(net, held); dout = float(np.abs(lg - st["logits"]).max())
            say(f"4.7 {task} {rule} row {g}: largest change of an output on the held-out pairs {dout:.3e} (receipt {x['out']:.3e}); eigenvalue {B.radius(W):.4f} (receipt {x['eig']:.4f}); "
                f"smallest lead {B.lead(lg):.2e}")
            rem = {mn: float(np.abs((block(W, mn, RULEROWS[mn]) - block(W0, mn, RULEROWS[mn])) @ dirs[mn]).max()) for mn in MATS}
            remz = {mn: float(np.abs((block(W, mn, SHUTBLOCKS[mn]) - block(W0, mn, SHUTBLOCKS[mn])) @ dirs[mn]).max()) for mn in MATS}
            zeros = not W["rnn.weight_hh_l0"][Z][:, :U0].any() and not W["rnn.bias_hh_l0"][Z].any() and not W["lm.weight"][ADDED][:, :U0].any() and not W["lm.bias"][ADDED].any()
            fixed = all(np.array_equal(W[k], W0[k]) for k in ("l1.weight", "l1.bias", "l3.bias")) and np.array_equal(W["l3.weight"][:, :H0], W0["l3.weight"][:, :H0]) \
                and np.array_equal(W["book.weight"][st["in_use"]], W0["book.weight"][st["in_use"]])
            free = {"reset and candidate rows of the reserve": float(np.abs(np.concatenate([W["rnn.weight_hh_l0"][U0:U1] - W0["rnn.weight_hh_l0"][U0:U1], W["rnn.weight_hh_l0"][2 * U1 + U0:] - W0["rnn.weight_hh_l0"][2 * U1 + U0:]])).max()),
                    "old rows from cells in reserve (recurrent matrix)": float(np.abs(W["rnn.weight_hh_l0"][OLD][:, U0:] - W0["rnn.weight_hh_l0"][OLD][:, U0:]).max()),
                    "lm from cells in reserve": float(np.abs(W["lm.weight"][:, U0:] - W0["lm.weight"][:, U0:]).max()),
                    "gate rows of the reserve from cells in reserve": float(np.abs(W["rnn.weight_hh_l0"][Z][:, U0:] - W0["rnn.weight_hh_l0"][Z][:, U0:]).max())}
            say(f"4.8 {task} {rule} row {g}: the rows the rule projects (old rows and gate rows of the reserve; all rows of lm and l2), largest |(row - start row) @ Vm|: "
                + ", ".join(f"{m} {v:.1e}" for m, v in rem.items()) + "; of these, the gate rows of the reserve and rows 256-511: " + ", ".join(f"{m} {v:.1e}" for m, v in remz.items())
                + f"; `zero_blocks` entries exactly zero {zeros}; l1, l3's old columns and bias, the byte table's rows in use bit-equal to the start {fixed}; largest change of free weights: "
                + ", ".join(f"{k} {v:.3f}" for k, v in free.items()))
            n32 = V.Net(W, np.float32); g32 = V.score_first_family(n32, S1, HELD)[1]
            d32 = sum(a != b for n in g32 if n.startswith("k0_") for a, b in zip(g32[n], st["gens32"][n])); h32 = sum(a != b for n in g32 if n.startswith("k12_") for a, b in zip(g32[n], st["gens32"][n]))
            g4032 = V.generate(n32, LEN40); d4032 = sum(a != b for a, b in zip(g4032, st["g4032"])); f32 = gates(n32, held)
            say(f"4.9 {task} {rule} row {g}, single precision: answers that differ from the start's {d32} of 31,684 and {h32} of 1,200; 40 letters: differ {d4032}, right {V.exact(g4032, LEN40)}; "
                f"largest 1 - z on the held-out pairs {f32['largest_opening']:.3e}, largest state {f32['largest_state']:.3e}")
            W4[(task, rule, g)] = dict(d=d, held=hd, right0=right0, right12=right12, d40=d40, r40=r40, nt=nt, ns=ns, ff=ff, ff40=ff40, new=new, ds=ds, hs=hs, d32=d32, h32=h32, d4032=d4032)
            del W, net, n32
    say(f"4.10 free disk after the weight files {shutil.disk_usage('/home/user').free / 1e9:.2f} GB")
    clock(4, t0)
    return W4


# ================================================================ section 5
def section5(blindrec):
    t0 = time.time(); say("===== SECTION 5. The fixed wording and the predictions =====")
    lf = {}; n_before = FIGS["n"]
    for (t, r), rows in MINE.items():
        w = words(rows); fl = first_learned([(x["updates"], x["nt"]) for x in rows]); lf[(t, r)] = fl; ffr, ffc = first_fail(rows)
        opened = [x["updates"] for x in rows if x["new_open"] > 0]
        say(f"5.1 {t} {r}: word {w}; largest d(g) {max(x['d'] for x in rows)} ({100 * max(x['d'] for x in rows) / N0:.2f}% of 31,684); largest held-out {max(x['held'] for x in rows)}; "
            f"reduction row 0 {rows[0]['reduction']:.2f}, last {rows[-1]['reduction']:.2f}, largest deviation {max(abs(x['reduction'] - rows[0]['reduction']) for x in rows):.2f}; "
            f"first failure of *kept* at row {ffr}: {ffc}; new task {'learned from row ' + str(fl) if fl else 'not learned'} ({rows[-1]['nt']:.1f} at 16,384; reserve shut {rows[-1]['ns']:.1f}); "
            f"rows with a cell open at some position on the new task {opened}; the addition 'learned without the cells in reserve' applies: "
            f"{bool(fl) and (rows[-1]['ns'] >= 80 or rows[-1]['new_open'] == 0)}; 'learning did not open the cells in reserve' applies: {(not fl) and not opened}")
        say(f"5.2 {t} {r}, by row (updates: largest 1 - z / cells open at some position on the new task's never-taught items): " + "; ".join(f"{x['updates']}: {x['new_1z']:.2g} / {x['new_open']}" for x in rows))
    sh = MINE; allrows = [x for rows in sh.values() for x in rows]
    lb = {k: first_learned([(x["updates"], x["nt"]) for x in v]) for k, v in blindrec.items()}
    bw = {k: words(v) for k, v in blindrec.items()}
    P = {"P55.1": all(jshow(f"{E55}checks/e55_check_{n}_{SEED}.json")["ok"] for n in (1, 2, 3)),
         "P55.2": all(lf[("unrelated", r)] for r in RULES), "P55.2b": bool(lf[("unrelated", "B")]), "P55.3": all(lf[("conflict", r)] for r in RULES),
         "P55.4": all(words(v) in ("kept exactly", "kept") for v in sh.values()),
         "P55.4b": all(abs(x["reduction"] - rows[0]["reduction"]) <= 2 for rows in sh.values() for x in rows),
         "P55.5": all(x["ff_1z"] < 1e-3 for x in allrows),
         "P55.6": all(sh[k][-1]["new_open"] >= 32 for k in RUNS if lf[k]),
         "P55.8": all(lf[("unrelated", r)] <= 4096 for r in RULES if lf[("unrelated", r)]),
         "P55.9": all(w == "kept exactly" for w in bw.values()), "P55.10": all(x["d"] <= 31 for v in blindrec.values() for x in v),
         "P55.11": all(x["ff_state"] < 1e-4 for v in blindrec.values() for x in v), "P55.12": all(lb[("unrelated", r)] for r in RULES), "P55.13": not any(lb[("conflict", r)] for r in RULES)}
    held_ = [k for k, v in P.items() if v]; failed = [k for k, v in P.items() if not v]
    say(f"5.3 predictions held {held_}; failed {failed}")
    claim_held = ["P55.1", "P55.2b", "P55.4b", "P55.5", "P55.6", "P55.8", "P55.9", "P55.10", "P55.12", "P55.13"]; claim_failed = ["P55.2", "P55.3", "P55.4", "P55.11"]
    for k in claim_held:
        cmp(f"message prediction {k} held", "True", str(P[k]))
    for k in claim_failed:
        cmp(f"message prediction {k} failed", "False", str(P[k]))
    rep = show(f"{E55}E55_REPORT.md").decode()
    for l in rep.splitlines():
        m = re.match(r"^\| (P55\.\w+) \|.*\| (Held|Failed)\b", l)
        if m:
            cmp(f"report prediction {m.group(1)}", m.group(2), "Held" if P[m.group(1)] else "Failed")
    say(f"5.3b predictions of the message and outcomes of the report's table compared: {FIGS['n'] - n_before}")
    say(f"5.4 P55.6 counts at row 16,384 where the task is learned: " + ", ".join(f"{k[0]} {k[1]} {sh[k][-1]['new_open']}" for k in RUNS if lf[k])
        + f"; P55.5 largest 1 - z on the held-out pairs over all rows and runs {max(x['ff_1z'] for x in allrows):.3e}")
    say(f"5.5 in all {FIGS['n']} figures or words compared (sections 3 and 5), {len(FIGS['diffs'])} differ:")
    for d in FIGS["diffs"]:
        say("      " + d)
    clock(5, t0)


# ================================================================ section 6
def section6(src):
    t0 = time.time(); say("===== SECTION 6. The runner as text =====")
    r51 = src["exp51_reserve.py"].decode().splitlines(); r55 = src["exp55_reserve.py"].decode().splitlines()
    a = next(i for i, l in enumerate(r51) if l.startswith("def fwd(")); b = next(i for i, l in enumerate(r51) if l.startswith("def install("))
    say(f"6.1 exp51_reserve.fwd (lines {a + 1} to {b}): lines that index the item x: {[(i + 1, l.strip()) for i, l in enumerate(r51[a:b], a) if re.search(r'\bx\[', l)]}")
    say(f"6.2 exp55_reserve.py, every line that sets or reads KEY or calls install: {[(i + 1, l.strip()) for i, l in enumerate(r55) if 'KEY[' in l or 'install(' in l]}")
    rb = next(i for i, l in enumerate(r55) if l.startswith("class Rule55"))
    say(f"6.3 Rule55, the rows of the arm shut: {[l.strip() for l in r55[rb:rb + 12] if 'cells = ' in l or 'units = ' in l]}; after every update in the arm shut: "
        f"{[l.strip() for l in r55[rb:rb + 40] if 'zero_blocks(self.m, Z_RES)' in l]}; the start shut: {[l.strip() for l in r55 if 'GATE_BIAS' in l and '+=' in l]}")
    clock(6, t0)


# ================================================================ section 7
def section7(start, W4):
    t0 = time.time(); say("===== SECTION 7. The table of the first family at 22 and 50 positions, the new task, and the gates, by scored row =====")
    say("7.0 columns: row | answers that differ of 31,684 / of 1,200 (22 positions) | right of 31,684 / of 1,200 | 40 letters (50 positions): differ / right of 300 | never-taught as run / reserve shut "
        "| new task: cells open at some position / by more than 0.5 | first family: cells open at some position; and, where weights are saved, the share of positions with a gate of the reserve open by more than 0.5 "
        "on the new task's items / on the held-out pairs / on the 40-letter items")
    for (t, r), rows in MINE.items():
        for x in rows:
            w = W4.get((t, r, x["updates"]))
            if x["updates"] == 0:
                f = start[r]["fig"]
                extra = f" | shares {f['new'][t]['share_pos_any_0p5']:.4f} / {f['ff']['share_pos_any_0p5']:.4f} / {f['ff40']['share_pos_any_0p5']:.4f} (the start, from its weights)"
            elif w:
                extra = f" | shares {w['new']['share_pos_any_0p5']:.4f} / {w['ff']['share_pos_any_0p5']:.4f} / {w['ff40']['share_pos_any_0p5']:.4f} (weights)"
            else:
                extra = ""
            say(f"7.1 {t} {r} {x['updates']}: {x['d']} / {x['held']} | {x['right0']} / {x['right12']} | {x['d40']} / {x['ff40_exact']} | {x['nt']:.1f} / {x['ns']:.1f} | {x['new_open']} / {x['new_05']} | {x['ff_open']}{extra}")
    for (t, r), rows in MINE.items():
        say(f"7.2 {t} {r}: right answers at 50 positions, row 0 {rows[0]['ff40_exact']}, smallest over the rows {min(x['ff40_exact'] for x in rows)} (row {min(rows, key=lambda x: x['ff40_exact'])['updates']}), "
            f"row 16,384 {rows[-1]['ff40_exact']}; largest 1 - z on the 40-letter items over the rows {max(x['ff40_1z'] for x in rows):.2e}; cells open at some position there, largest over the rows {max(x['ff40_open'] for x in rows)}; "
            f"right answers at 22 positions (31,684 + 1,200) row 0 {rows[0]['right0'] + rows[0]['right12']}, row 16,384 {rows[-1]['right0'] + rows[-1]['right12']}")
    clock(7, t0)


# ================================================================ section 8
def section8():
    t0 = time.time(); say("===== SECTION 8. The nine corrections at the head; what remains supplied; decision 14 =====")
    rep = show(f"{E55}E55_REPORT.md").decode(); tb = show(f"{E55}E55_TABLES_blind.md").decode(); ta = show(f"{E55}E55_TABLES_all.md").decode()
    rb_ = jshow(f"{E55}E55_READ_blind.json")["runs"]; ra = jshow(f"{E55}E55_READ_all.json")["runs"]
    has = lambda txt, *ss: all(s in txt for s in ss)
    c = {1: dict(report=has(rep, "274 of 300", "21 of 300", "0.9998", "0.997", "lost in this run"), tables=has(tb, "40 letters: right of 300", "40 letters: largest 1 − z") and has(ta, "40 letters: right of 300"),
                 read=all("right_40_letters_by_row" in v and "largest_opening_40_letters_by_row" in v for v in rb_.values())),
         2: dict(report=has(rep, "1.098, 1.434, 1.058, 1.197", "nothing bounds the state", "observation on items of up to 22 positions") and rep.count("largest reserve state, 40 letters") == 4,
                 tables=has(tb, "nothing bounds the state", "eigenvalue", "40 letters: largest reserve state"), read=all("reserve_state_40_letters_by_row" in v for v in rb_.values())),
         3: dict(report=has(rep, "45 changed answers", "15 in double precision", "142 in single precision", "outcomes of the brief's rules are the same"), tables=has(tb, "15 in double precision", "142 in single precision")),
         4: dict(report="kept exactly (kept" not in rep and "| kept on items of up to 22 positions and not on items of 50 |" in rep, tables="kept exactly (kept" not in tb + ta,
                 read=rb_["blind unrelated B"]["word"] == ra["blind unrelated B"]["word"] == "kept on items of up to 22 positions and not on items of 50"),
         5: dict(report=has(rep, "1.69e-3", "1.78e-4", "0.200", "2.33e-2", "not the consequence of a bound"), tables=has(tb, "1.69e-3", "1.78e-4", "0.200", "2.33e-2")),
         6: dict(report=bool(re.search(r"\| P55\.1 \|[^\n]*\| Held", rep))),
         7: dict(report=has(rep, "34.5 points", "34.4 points", "8,192 against 1,024", "2,048 against 512")),
         8: dict(report=has(rep, "2abb0c3f", "description in place of a command", "4e2a72c2")),
         9: dict(report=has(rep, "The record of the bytes in use", "One use of the keyed forward pass"))}
    for k, v in c.items():
        say(f"8.{k} correction {k}: " + ", ".join(f"{a} {b}" for a, b in v.items()))
    sec7 = rep.split("## 7.")[1].split("## 8.")[0]
    say(f"8.10 the report's section 7 names: directions computed once {'directions in use are computed once' in sec7}; rows named {'rows that the rule protects are named' in sec7}; "
        f"reserve put in {'put in by the program at the change of task' in sec7}; bias of 10 {'bias of 10' in sec7}; change of task known {'change of task is known' in sec7}; "
        f"the two parts {('record of the bytes in use' in sec7) and ('keyed forward pass' in sec7)}")
    say(f"8.11 in the report: 'requirement A' {len(re.findall('[Rr]equirement A', rep))} times; 'decision 14' {rep.count('decision 14')} times; 'one seed' {rep.count('one seed')} times; "
        f"'22 positions' {rep.count('22 positions')} times; the word 'shut' in the definition of the arm `shut` (section 1) mentions lm and l2: "
        f"{'lm' in [l for l in rep.splitlines() if l.startswith('- *The arm `shut`*')][0]}")
    say(f"8.12 the report's provenance paragraph names the commit recorded by `conflict` under back-propagation of the arm `shut` (77d92a72): {'77d92a72' in rep}; "
        f"mentions of '40 letters' or '50 positions' in the report's paragraphs on the arm `shut` (section 5): {sum(rep.split('**Arm `shut`.**')[1].split('## 6.')[0].count(w) for w in ('40 letters', '50 positions'))}")
    clock(8, t0)


# ================================================================ section 9
def section9():
    t0 = time.time(); say("===== SECTION 9. Form =====")
    files = lsn(H, E55)
    texts = [f for f in files if not f.endswith(".pt")] + [M + "exp55_reserve.py", M + "e55_read.py"]
    body = {f: show(f).decode(errors="replace") for f in texts}
    marks = ["@" * 2, "TO" + "DO", "TB" + "D", "X" * 2]
    for m in marks:
        where = {f.split("/")[-1]: t.count(m) for f, t in body.items() if m in t}
        say(f"9.1 mark {m!r}: {sum(t.count(m) for t in body.values())} occurrences in {len(texts)} text files; by file {where}")
    rep = body[E55 + "E55_REPORT.md"]; dpart = rep.split("```diff")[1].split("```")[0] if "```diff" in rep else ""
    say(f"9.2 in E55_REPORT.md, occurrences of the first mark inside its unified difference (section 10, hunk headers): {dpart.count(marks[0])} of {rep.count(marks[0])}")
    msg = subprocess.run(["git", "log", "-1", "--format=%B", H], capture_output=True, text=True, check=True).stdout
    wl = {w.lower() for w in re.findall(r"Acting-Agent:\s*(\w+):", msg)} | {w.lower() for w in re.findall(r"^(\w+)-Session:", msg, flags=re.M)}
    extra = [w.lower() for w in os.environ.get("E55_EXTRA_WORDS", "").split(",") if w.strip()]
    allw = sorted(wl | set(extra))
    tot = exempt = 0; kinds = {}
    for f, t in body.items():
        lt = t.lower()
        for w in allw:
            for mm in re.finditer(re.escape(w), lt):
                tot += 1
                before, after = lt[max(0, mm.start() - 1):mm.start()], lt[mm.end():mm.end() + 8]
                if before == "/" or after.startswith("/") or after.startswith("-session") or after.startswith(":session"):
                    exempt += 1
                else:
                    k = f.split("/")[-1]; kinds[k] = kinds.get(k, 0) + 1
    say(f"9.3 search for {len(wl)} word(s) from the head commit's trailers and {len(extra)} from E55_EXTRA_WORDS, case-insensitive, in {len(texts)} text files: {tot} occurrences, "
        f"{exempt} of them in a branch name or a trailer; the others by file {kinds}")
    clock(9, t0)


if __name__ == "__main__":
    say(f"E55, arm shut and the final report, verification at {A[:8]} and {H[:8]}; numpy {np.__version__}")
    files = section0()
    src = section1(files)
    start = section2()
    blindrec = section3()
    W4 = section4(start)
    section5(blindrec)
    section6(src)
    section7(start, W4)
    section8()
    section9()
    say(f"running time {time.time() - T0:.0f} s")
