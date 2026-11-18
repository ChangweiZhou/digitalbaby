#!/usr/bin/env python3
"""Verification of the four runs of the arm `blind` of E55, first part (seed 73129), on E51's worker's branch at commit
fb0ec89aa896c597431135b864d7d3543e70f213, against the brief
TASK_E55_THE_FIRST_FAMILY_KEPT_WITHOUT_A_LINE_OF_THE_PROGRAM_THAT_LOOKS_AT_THE_ITEM_20261118.md (revision 4).

Read-only. Nothing is trained, the training library is not used, and no file of the worker's branch is executed: every receipt,
log, source text and weight file is read with `git show` into memory, one weight file at a time (the clone is partial, so a file
read for the first time is fetched into the repository's object store; the script stops reading weight files if less than 0.9 GB
of disk would remain). The forward pass, the generation of answers, the items, the streams and the hashes are those of the
reviewer's verification of E51's trial (e51_trial_verify.py beside this file, imported as `V`), which writes them in numpy from
the brief of E51 and a reading of the sources. The script writes no file.

    sections  1  the receipts of the four runs: completeness, settings, scored rows, data digests (recomputed from the reader's own
                 streams), the hashes of the runner and of the files it takes from, the logs
              2  the start `blind`: built again from E51's keyed start, against the saved start and the receipts of check 1
              3  every figure of the worker's table (E55_TABLES_blind.md), of E55_READ_blind.json and of the worker's message,
                 recomputed from the receipts with this script's own code, side by side
              4  the same from the weights at rows 64, 4,096 and 16,384 of each run (the plain forward pass in numpy)
              5  the fixed wording and the predictions, run by run
              6  the runner as text
              7  figures used in the reading (decision 14)
              8  form: unfilled marks in the committed text files of exp45/e55 and the two scripts; a search, in the same files, for the
                 word that the trailers of the worker's commit give as the acting agent's name and for any words passed, separated by
                 commas, in the environment variable E55_EXTRA_WORDS (no such word is written in this file)

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e55_blind_verify.py
"""
import hashlib, importlib.util, json, math, os, re, shutil, subprocess, sys, time
from random import Random
import numpy as np

T0 = time.time()
sys.dont_write_bytecode = True
REF = "fb0ec89aa896c597431135b864d7d3543e70f213"
M = "MOD97_LOCAL_RULE_20261106/"
E55 = M + "exp45/e55/"
E51 = M + "exp45/e51/"
REVIEW = "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/"
assert os.path.isdir(".git") or os.path.isfile(".git"), "run from the repository root"

_argv = sys.argv; sys.argv = [sys.argv[0]]
_spec = importlib.util.spec_from_file_location("e51_trial_verify", os.path.join(REVIEW, "e51_trial_verify.py"))
sys.path.insert(0, REVIEW)
V = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V)
sys.argv = _argv

SEED, BOS, EOS = V.SEED, V.BOS, V.EOS
H0, U0, H1, U1 = V.H0, V.U0, V.H1, V.U1
TASKS, RULES = ("unrelated", "conflict"), ("S", "B")
RUNS = [(t, r) for t in TASKS for r in RULES]
ROWS55 = (0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
KEEP = (64, 4096, 16384)
RES, OLD = V.RES, V.OLD
ADDED = np.arange(H0, H1)
MATS = V.MATS
OLDSETS = {"ih": OLD, "hh": OLD, "lm": np.arange(H0), "l2": np.arange(H0)}
BLINDSETS = {"ih": RES, "hh": RES, "lm": ADDED, "l2": ADDED}
FIG = {}                                     # every figure printed, by key


def say(*a):
    print(*a, flush=True)


def keep(key, value):
    FIG[key] = value
    return value


def show(path, ref=REF):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def jshow(path):
    return json.loads(show(path))


def disk_ok():
    free = shutil.disk_usage("/home/user").free
    return free, free > 0.9e9 + 20e6


def load_weights(path):
    free, ok = disk_ok()
    if not ok:
        raise SystemExit(f"stopped before reading {path}: {free / 1e9:.2f} GB free")
    blob = show(path)
    W = {n.lstrip("/"): np.frombuffer(raw, dtype=V.DTYPE[st]).reshape(shape).copy() for n, (st, shape, raw) in V.tensors(blob).items()}
    return W, sha(blob)


def away(D, Vm):
    return D - (D @ Vm) @ Vm.T


def block(W, mn, rows):
    wk, bk = MATS[mn]
    return np.concatenate([W[wk][rows], W[bk][rows][:, None]], 1).astype(np.float64)


def radius(W):
    Wn = W["rnn.weight_hh_l0"][2 * U1 + U0:3 * U1, U0:U1].astype(np.float64)
    return float(np.abs(np.linalg.eigvals(0.5 * np.eye(U1 - U0) + 0.25 * Wn)).max())


def states_by_position(net, items, **key):
    """Teacher forcing: the largest magnitude of a state of a cell in reserve over all items and positions, and by position for items of one length."""
    big, prof = 0.0, None
    for idx in V.groups_of(items):
        x = np.array([[BOS] + items[i][0] + items[i][1][:-1] for i in idx])
        k = net.run(x, keep=True, **key)[1]; h = np.abs(k["h"][:, :, U0:])
        big = max(big, float(h.max())); p = h.max((0, 2))
        prof = p if prof is None or len(p) != len(prof) else np.maximum(prof, p)
    return big, prof


def gate_figs(net, items, **key):
    """The runner's `gates_on` as read: over cells in reserve, teacher forcing: largest 1 - z; the largest state; the largest added unit at the answer positions."""
    big_open, big_h, big_z2 = 0.0, 0.0, 0.0
    for idx in V.groups_of(items):
        x = np.array([[BOS] + items[i][0] + items[i][1][:-1] for i in idx])
        k = net.run(x, keep=True, **key)[1]
        big_open = max(big_open, float((1 - k["z"][:, :, U0:]).max())); big_h = max(big_h, float(np.abs(k["h"][:, :, U0:]).max()))
        big_z2 = max(big_z2, float(np.abs(k["z2"][:, :, H0:]).max()))
    return dict(largest_opening=big_open, largest_state=big_h, largest_added=big_z2)


def lead(lg):
    """The smallest lead of the largest output over the second largest, over items and answer positions."""
    top2 = np.sort(lg, axis=-1)[..., -2:]
    return float((top2[..., 1] - top2[..., 0]).min())


def first_learned(curve):
    for i in range(len(curve)):
        if all(v >= 80 for _, v in curve[i:]):
            return curve[i][0]
    return None


def clock(label, t0):
    say(f"[section {label}: {time.time() - t0:.0f} s]\n")


# ---------------------------------------------------------------- items
S1 = V.first_family_items(); HELD = V.held12(); held = [it for n in HELD for it in HELD[n]]
ADD = V.DEV[0]
LEN40 = V.panel(ADD, V.split(ADD, .5, SEED)[1], SEED, (12, 40), 300)[40]      # the runner's `len40`: E.panel(..., ks=(12, 40), n=300)[40]
SAMPLE = V.check1_sample(S1)
REC = {(t, r): jshow(f"{E55}e55_{t}_blind_{r}_{SEED}.json") for t, r in RUNS}
row = lambda rec, g: [x for x in rec["rows"] if x["updates"] == g][0]


# ================================================================ section 1
def section1():
    t0 = time.time(); say("===== SECTION 1. The receipts of the four runs =====")
    files = subprocess.run(["git", "ls-tree", "-r", "--name-only", REF, "--", E55], capture_output=True, text=True, check=True).stdout.split()
    say(f"1.0 files under exp45/e55 at {REF[:8]}: {keep('n_files', len(files))}; weight files {sum(f.endswith('.pt') for f in files)}; receipts of the arm blind's runs "
        f"{sum(bool(re.search(r'e55_(unrelated|conflict)_blind_[SB]_73129[.]json$', f)) for f in files)}")
    src = {}
    for name in ("exp55_reserve.py", "exp51_reserve.py", "e55_read.py"):
        src[name] = show(M + name)
    say(f"1.1 SHA-256 at {REF[:8]}: exp55_reserve.py {sha(src['exp55_reserve.py'])}; exp51_reserve.py {sha(src['exp51_reserve.py'])} "
        f"(the brief's: 84d0dad7...: {sha(src['exp51_reserve.py']) == '84d0dad7f6e5eeb15c4518a8c591eb5fc043d72b17926913a3b3fde690d1f34a'})")
    ksS = jshow(f"{E51}e51_start_keyed_S_{SEED}.json"); ksB = jshow(f"{E51}e51_start_keyed_B_{SEED}.json"); ks = {"S": ksS, "B": ksB}
    stS = {r: jshow(f"{E55}e55_start_blind_{r}_{SEED}.json") for r in RULES}
    srcsha_cache = {}
    for (task, rule), rec in REC.items():
        s, c = rec["settings"], rec["complete"]; p = s["provenance"]
        ups = [x["updates"] for x in rec["rows"]]
        losses = [x["loss_since_last_row"] for x in rec["rows"][1:]]
        ok = dict(
            complete=bool(c) and c["updates"] == 16384 and c["optimizer_steps"] == [16384] and c["items"] == {task: 16384 * 64, "first_family": 0},
            rows=ups == list(ROWS55) and s["rows"] == list(ROWS55) and s["committed_rows"] == list(KEEP),
            losses_finite=all(isinstance(v, float) and math.isfinite(v) for v in losses),
            arm_task_rule_seed=(s["experiment"], s["part"], s["arm"], s["task"], s["rule"], s["seed"], s["local"], s["updates"]) == ("E55", "first", "blind", task, rule, SEED, rule == "S", 16384),
            forward=(s["keyed_forward"] is False and s["forward"] == "exp51_reserve.fwd with KEY['in_use'] = None (no added line)" and s["rule_applied"] is True and s["restore_book"] is True and s["gate_bias"] is None),
            start=(s["start"]["start_file"] == f"e51_start_keyed_{rule}_{SEED}.pt" and s["start"]["start_sha256"] == ks[rule]["weights_sha256"] and s["start"]["start_tensors_sha256"] == ks[rule]["tensors_sha256"]
                   and s["start_e55"]["sha256"] == stS[rule]["sha256"] and s["start_e55"]["tensors_sha256"] == stS[rule]["tensors_sha256"] == s["start_tensors_sha256"] and s["start_e55"]["form"] == "blind"),
            optimizer=all(g["weight_decay"] == 0.0 and g["lr"] == 0.001 and g["betas"] == [0.9, 0.98] for g in s["optimizer_groups_read_back"]) and s["decay"] == 0.0 and s["batch"] == 64,
            stream=s["stream"] == "the new task's stream alone",
            digest_end=c["data_digest"] == rec["rows"][-1]["data_digest"] and c["tensors_sha256"] == rec["rows"][-1]["tensors_sha256"],
            weights_named=[x.get("weights") for x in rec["rows"] if x.get("weights")] == [f"e55_{task}_blind_{rule}_{SEED}_row{g}.pt" for g in KEEP],
            git_clean=p["git"]["sources_modified_vs_commit"] == [] and p["git"]["e51_sources_modified_vs_commit"] == [] and p["git"]["e55_sources_modified_vs_commit"] == [],
        )
        # the hashes of every source the receipt names, against the files at REF
        bad = []
        for grp in ("sources_sha256", "e51_sources_sha256", "e55_sources_sha256"):
            for f, h in p[grp].items():
                if f not in srcsha_cache:
                    srcsha_cache[f] = sha(show(M + f))
                if srcsha_cache[f] != h:
                    bad.append(f)
        ok["source_hashes"] = not bad
        keep(f"receipt_ok_{task}_{rule}", ok)
        say(f"1.2 {task} {rule}: " + ", ".join(f"{k} {v}" for k, v in ok.items()) + f"; updates {c['updates']}, optimizer steps {c['optimizer_steps']}, items {c['items']}; sources hashed {sum(len(p[g]) for g in ('sources_sha256', 'e51_sources_sha256', 'e55_sources_sha256'))}, differing {bad}; "
            f"runtime commit {p['git']['commit'][:8]}; command '{p['command']}'; torch {p['host']['torch']}, threads {p['host']['torch_threads']}; cpu total {c['cpu_total']:.0f} s; "
            f"loss at row 16,384 {losses[-1]:.4g}")
    cm = {rec["settings"]["provenance"]["git"]["commit"] for rec in REC.values()}
    anc = all(subprocess.run(["git", "merge-base", "--is-ancestor", c, REF]).returncode == 0 for c in cm)
    same = all(subprocess.run(["git", "rev-parse", f"{c}:{M}exp55_reserve.py"], capture_output=True, text=True).stdout == subprocess.run(["git", "rev-parse", f"{REF}:{M}exp55_reserve.py"], capture_output=True, text=True).stdout for c in cm)
    say(f"1.3 the commits recorded in the runs' provenance {sorted(c[:8] for c in cm)} are ancestors of {REF[:8]}: {anc}; exp55_reserve.py is the same blob there: {same}; CPU-hours of the four runs together {keep('cpu_hours', sum(r['complete']['cpu_total'] for r in REC.values()) / 3600):.2f}")
    # data digests from the reader's own streams, at every scored row, against the receipts and against E51's `keyed` (rows up to 4,096)
    for task in TASKS:
        nxt = V.source(task, False); dg = V.Digest(); mine = {0: dg.value()}
        for g in range(1, 16385):
            dg.add([nxt()[1] for _ in range(64)])
            if g in ROWS55:
                mine[g] = dg.value()
        res = {}
        for rule in RULES:
            e51 = jshow(f"{E51}e51_{task}_keyed_{rule}_{SEED}.json")
            res[rule] = dict(receipt=all(row(REC[(task, rule)], g)["data_digest"] == mine[g] for g in ROWS55),
                             e51_keyed=all(row(e51, g)["data_digest"] == mine[g] for g in ROWS55 if g <= 4096))
        keep(f"digest_{task}", res)
        say(f"1.4 data digests of `{task}` from the reader's own stream at the twelve scored rows: equal to the receipts {[res[r]['receipt'] for r in RULES]} (S, B); "
            f"at the ten rows up to 4,096 equal to E51's `keyed` {[res[r]['e51_keyed'] for r in RULES]}; digest at 16,384 {mine[16384]['sha256'][:16]}")
    # the logs
    logs = [f for f in files if "/logs/" in f]
    for f in logs:
        txt = show(f).decode()
        if "/aborted/" in f:
            say(f"1.5 {f.split('e55/')[1]}: {txt.strip()!r}")
    runlogs = [f for f in logs if re.search(r"logs/e55_\w+_blind_[SB]_73129\.\d+\.log$", f)]
    for f in runlogs:
        txt = show(f).decode(); tag = re.search(r"(e55_\w+_blind_[SB]_73129)", f).group(1)
        lines = [l for l in txt.splitlines() if l.startswith(tag + " row")]
        bad = [w for w in ("nan", "inf ", "Traceback", "Error", "FAILED") if w in txt]
        say(f"1.6 {f.split('e55/')[1]}: rows printed {len(lines)}, 'complete' line {(tag + ' complete') in txt}, words of failure {bad}")
    for f in logs:
        if re.search(r"logs/check\d", f):
            txt = show(f).decode(); say(f"1.7 {f.split('e55/')[1]}: last line {txt.strip().splitlines()[-1]!r}")
    for n in (1, 2, 3):
        ck = jshow(f"{E55}checks/e55_check_{n}_{SEED}.json"); keep(f"check{n}_ok", ck["ok"])
        say(f"1.8 receipt of check {n}: ok {ck['ok']}; source hash of exp55_reserve.py {ck['provenance']['e55_sources_sha256']['exp55_reserve.py'] == sha(src['exp55_reserve.py'])}; "
            f"commit {ck['provenance']['git']['commit'][:8]}, exp55_reserve.py untracked then: {ck['provenance']['git']['e55_sources_modified_vs_commit']}")
    for r in RULES:
        say(f"1.9 start receipt blind {r}: source hash of exp55_reserve.py equal to the file at {REF[:8]}: {stS[r]['provenance']['e55_sources_sha256']['exp55_reserve.py'] == sha(src['exp55_reserve.py'])}")
    dry = [f for f in files if f.endswith("_dry64.json")]
    for f in dry:
        d = jshow(f); say(f"1.10 {f.split('e55/')[1]}: command recorded {d['settings']['provenance']['command']!r}")
    clock(1, t0)
    return src


# ================================================================ section 2
def section2():
    t0 = time.time(); say("===== SECTION 2. The start `blind` =====")
    free, _ = disk_ok(); say(f"2.0 free disk before the first weight file {free / 1e9:.2f} GB")
    ck1 = jshow(f"{E55}checks/e55_check_1_{SEED}.json")
    out = {}
    for rule in RULES:
        Wk, shk = load_weights(f"{E51}e51_start_keyed_{rule}_{SEED}.pt")
        D, _ = load_weights(f"{E51}e51_start_keyed_{rule}_{SEED}_directions.pt")
        dirs = {m: D[f"directions/{m}"] for m in MATS}; in_use = D["in_use"].astype(bool)
        Wb, shb = load_weights(f"{E55}e55_start_blind_{rule}_{SEED}.pt"); srec = jshow(f"{E55}e55_start_blind_{rule}_{SEED}.json")
        say(f"2.1 rule {rule}: keyed start file hash equal to E51's receipt {shk == jshow(f'{E51}e51_start_keyed_{rule}_{SEED}.json')['weights_sha256']}; "
            f"start blind: file hash equal to its receipt {shb == srec['sha256']}, hash of its tensors equal to its receipt {V.tensors_sha(Wb) == srec['tensors_sha256']}; "
            f"directions {[dirs[m].shape[1] for m in MATS]}, bytes in use {int(in_use.sum())}")
        # (a) bit-equal outside the four blocks; which rows differ
        rows_block = {"rnn.weight_ih_l0": RES, "rnn.bias_ih_l0": RES, "rnn.weight_hh_l0": RES, "rnn.bias_hh_l0": RES, "lm.weight": ADDED, "lm.bias": ADDED, "l2.weight": ADDED, "l2.bias": ADDED}
        outside_equal, rows_changed = True, {}
        for k in Wk:
            diff = (Wb[k] != Wk[k]).reshape(len(Wk[k]), -1).any(1)
            inside = np.zeros(len(Wk[k]), bool)
            if k in rows_block:
                inside[rows_block[k]] = True
            outside_equal &= not diff[~inside].any()
            if diff.any():
                rows_changed[k] = int(diff.sum())
        zeros = (not Wb["rnn.weight_hh_l0"][RES][:, :U0].any() and not Wb["rnn.bias_hh_l0"][RES].any() and not Wb["lm.weight"][ADDED][:, :U0].any() and not Wb["lm.bias"][ADDED].any())
        say(f"2.2 (a) every entry outside the four blocks bit-equal to E51's keyed start: {keep(f'start_outside_equal_{rule}', bool(outside_equal))}; rows that differ by tensor {rows_changed}; "
            f"the entries of `zero_blocks` exactly zero: {zeros}")
        # (b) the blocks against the directions in use; the same blocks in the keyed start
        rem = {mn: float(np.abs(block(Wb, mn, BLINDSETS[mn]) @ dirs[mn]).max()) for mn in MATS}
        remk = {mn: float(np.abs(block(Wk, mn, BLINDSETS[mn]) @ dirs[mn]).max()) for mn in MATS}
        keep(f"start_remainder_{rule}", rem)
        say(f"2.3 (b) largest |D @ Vm| at the four blocks of the start blind: " + ", ".join(f"{m} {v:.1e}" for m, v in rem.items()) + "; check 1's receipt: "
            + ", ".join(f"{m} {v:.1e}" for m, v in ck1["starts"][f"blind {rule}"]["b_largest_D_at_Vm"].items()) + "; the same blocks in the keyed start: " + ", ".join(f"{m} {v:.2f}" for m, v in remk.items()))
        # the start built again in numpy from the keyed start, against the saved start
        Wn = {k: v.copy() for k, v in Wk.items()}
        for mn, rows in BLINDSETS.items():
            wk, bk = MATS[mn]; Dm = away(block(Wn, mn, rows), dirs[mn]); Wn[wk][rows] = Dm[:, :-1].astype(np.float32); Wn[bk][rows] = Dm[:, -1].astype(np.float32)
        Wn["rnn.weight_hh_l0"][np.ix_(RES, np.arange(U0))] = 0; Wn["rnn.bias_hh_l0"][RES] = 0; Wn["lm.weight"][np.ix_(ADDED, np.arange(U0))] = 0; Wn["lm.bias"][ADDED] = 0
        nd = sum(int((Wn[k] != Wb[k]).sum()) for k in Wb); md = max(float(np.abs(Wn[k].astype(np.float64) - Wb[k]).max()) for k in Wb)
        say(f"2.4 the start built again in numpy (projection in double precision, stored in single): entries that differ from the saved start {keep(f'start_rebuild_entries_differ_{rule}', nd)}, largest difference {md:.1e}")
        # (c) and (d)
        nb, nk = V.Net(Wb), V.Net(Wk)
        f_h, f_s = gate_figs(nb, held), gate_figs(nb, SAMPLE)
        dlg = max(float(np.abs(V.answer_logits(nb, its) - V.answer_logits(nk, its, in_use=in_use)).max()) for its in (held, SAMPLE))
        fb = {}
        for task in TASKS:
            st = V.source(task, False); items = [st()[1] for _ in range(64)]
            fb[task] = gate_figs(nb, items)["largest_state"]
        gb = V.score_first_family(nb, S1, HELD)[1]; gk = V.score_first_family(nk, S1, HELD, in_use=in_use)[1]
        ndiff = sum(a != b for n in gb for a, b in zip(gb[n], gk[n]))
        r = ck1["starts"][f"blind {rule}"]
        say(f"2.5 (c) on the held-out pairs and on check 1's 2,000 items: largest state of a cell in reserve {max(f_h['largest_state'], f_s['largest_state']):.3e} (receipt "
            f"{max(r['c_gates_on_heldout_k12']['largest_reserve_state'], r['c_gates_on_sample_2000']['largest_reserve_state']):.3e}; limit 1e-5); largest added unit at the answer positions "
            f"{max(f_h['largest_added'], f_s['largest_added']):.3e} (receipt {max(r['c_gates_on_heldout_k12']['largest_added_unit_at_answer_positions'], r['c_gates_on_sample_2000']['largest_added_unit_at_answer_positions']):.3e}); "
            f"largest 1 - z {max(f_h['largest_opening'], f_s['largest_opening']):.6f}; largest difference of an output from the keyed start under the keyed forward {dlg:.2e} (receipt {r['c_largest_output_difference_from_keyed_start']:.2e}; limit 1e-4)")
        say(f"2.6 (c) first batch of each new task's stream: largest state of a cell in reserve " + ", ".join(f"{t} {v:.4f}" for t, v in fb.items()) + " (receipt "
            + ", ".join(f"{t} {v:.4f}" for t, v in r["c_largest_reserve_state_first_batch_of_new_task"].items()) + "; limit above 1e-3)")
        say(f"2.7 (d) generated answers of the first family (31,684 without letters and 1,200 held-out) that differ from the keyed start's under the keyed forward: "
            f"{keep(f'start_answers_differ_{rule}', ndiff)} of {sum(len(v) for v in gb.values())} (receipt {r['d_answers_differing_from_keyed_start']} of {r['d_answers']})")
        e0 = radius(Wb); say(f"2.8 eigenvalue of the reserve's map at the start: {keep(f'eig0_{rule}', e0):.4f}; check 1 receipt ok {r['ok']}")
        say(f"2.10 smallest lead of the largest output over the second at an answer position of the 1,200 held-out pairs at the start: {keep(f'lead0_{rule}', lead(V.answer_logits(nb, held))):.2e}")
        nb32 = V.Net(Wb, np.float32)
        out[rule] = dict(W=Wb, dirs=dirs, in_use=in_use, gens=gb, logits=V.answer_logits(nb, held), g40=V.generate(nb, LEN40),
                         gens32=V.score_first_family(nb32, S1, HELD)[1], logits32=V.answer_logits(nb32, held), g4032=V.generate(nb32, LEN40))
        f32 = gate_figs(nb32, held)
        say(f"2.9 the same start in single precision (numpy): largest state of a cell in reserve on the held-out pairs {f32['largest_state']:.3e}, largest added unit {f32['largest_added']:.3e}")
        del Wk, Wn
    clock(2, t0)
    return out


# ================================================================ section 3
def num(s):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


def unit(s):
    """Half a unit of the last printed digit."""
    s = s.strip(); m = re.match(r"^-?(\d+)(?:\.(\d+))?(?:e([+-]\d+))?$", s)
    dec = len(m.group(2) or ""); ex = int(m.group(3) or 0)
    return 0.5 * 10 ** (ex - dec)


def agrees(rep, mine):
    if isinstance(mine, str) or mine is None:
        return str(rep).strip() == str(mine)
    v = num(rep)
    if v is None:
        return False
    return abs(v - mine) <= unit(rep) * (1 + 1e-9) + 1e-12


def recompute(rec):
    """Every figure of a run's table, from the receipt's raw counts, by this script's code."""
    out = []
    for x in rec["rows"]:
        s1 = x["score1_first_family_all_pairs_k0"]
        red = (100 * s1["add89"]["reduction"]["exact"] / s1["add89"]["reduction"]["n"] + 100 * s1["sub89"]["reduction"]["exact"] / s1["sub89"]["reduction"]["n"]) / 2
        pp = x["share_of_answers_equal_to_row0"]["per_panel"]
        d = sum(7921 - pp["k0_" + n] for n in ("add89", "sub89", "max89", "min89")); hd = sum(300 - pp["k12_" + n] for n in ("add89", "sub89", "max89", "min89"))
        task = "conflict" if "add97" in x["score3_new_task"] else "unrelated"
        ps = ("add97", "sub97") if task == "conflict" else ("held",)
        nt = 100 * sum(x["score3_new_task"][p]["exact"] for p in ps) / sum(x["score3_new_task"][p]["n"] for p in ps)
        ns = 100 * sum(x["score3_new_task_reserve_shut"][p]["exact"] for p in ps) / sum(x["score3_new_task_reserve_shut"][p]["n"] for p in ps)
        gn, gf = x.get("gates_on_new_task", {}), x.get("gates_on_first_family", {})
        g40 = x.get("gates_on_first_family_40_letters", {}); f40 = x.get("first_family_40_letters", {})
        out.append(dict(updates=x["updates"], reduction=red, d=d, held=hd, d40=f40.get("differ_from_row0"), nt=nt, ns=ns,
                        new_open=gn.get("cells_open_at_some_position"), new_mean=gn.get("cells_open_by_more_than_0p1"), new_1z=gn.get("largest_opening"),
                        ff_1z=gf.get("largest_opening"), ff_open=gf.get("cells_open_at_some_position"), ff_state=gf.get("largest_reserve_state"), ff_added=gf.get("largest_added_unit_at_answer_positions"),
                        out=x.get("first_family_largest_change_of_an_output"), eig=x.get("reserve_map_largest_eigenvalue"),
                        ff40_state=g40.get("largest_reserve_state"), ff40_1z=g40.get("largest_opening"), ff40_exact=f40.get("exact"), loss=x["loss_since_last_row"]))
    return out


COLS = ["updates", "reduction", "d", "held", "d40", "nt", "ns", "new_open", "new_mean", "new_1z", "ff_1z", "ff_open", "ff_state", "ff_added", "out", "eig"]


def words(rows):
    r0 = rows[0]["reduction"]
    exact = all(x["d"] == 0 and x["held"] == 0 for x in rows)
    kept = all(x["d"] <= 31 and x["held"] <= 1 and abs(x["reduction"] - r0) <= 0.5 for x in rows)
    w = "kept exactly" if exact else "kept" if kept else "lost" if rows[-1]["reduction"] < r0 / 2 else "partly kept"
    w40 = "kept on items of up to 22 positions and not on items of 50" if w in ("kept exactly", "kept") and max(x["d40"] for x in rows) > 3 else None
    return w, w40


def section3():
    t0 = time.time(); say("===== SECTION 3. The worker's figures recomputed from the receipts =====")
    md = show(f"{E55}E55_TABLES_blind.md").decode(); rd = jshow(f"{E55}E55_READ_blind.json")["runs"]
    mine = {k: recompute(rec) for k, rec in REC.items()}
    total, diffs = 0, []
    # the per-run tables
    secs = re.split(r"^### ", md, flags=re.M)[1:]
    for sec in secs:
        m = re.match(r"(\w+), rule (\w), arm `blind`", sec); task, rule = m.group(1), m.group(2)
        lines = [l for l in sec.splitlines() if l.startswith("| ") and re.match(r"^\| \d", l)]
        e55lines = lines[:12]; e51lines = lines[12:]
        for l, x in zip(e55lines, mine[(task, rule)]):
            cells = [c.strip() for c in l.strip("|").split("|")]
            assert int(cells[0]) == x["updates"]
            for c, col in zip(cells[1:], COLS[1:]):
                total += 1
                if not agrees(c, x[col]):
                    diffs.append(f"{task} {rule} row {x['updates']} {col}: reported {c}, recomputed {x[col]}")
        # E51's reference rows from E51's receipts
        ref = {a: recompute(jshow(f"{E51}e51_{task}_{a}_{rule}_{SEED}.json")) for a in ("keyed", "open", "rehearse")}
        for l in e51lines:
            cells = [c.strip() for c in l.strip("|").split("|")]; g = int(cells[0])
            for c, a in zip(cells[1:], ("keyed", "open", "rehearse")):
                x = [y for y in ref[a] if y["updates"] == g][0]
                for v, col in zip(c.split(";"), ("reduction", "nt", "ns")):
                    total += 1
                    if not agrees(v, x[col]):
                        diffs.append(f"{task} {rule} E51 `{a}` row {g} {col}: reported {v.strip()}, recomputed {x[col]}")
    say(f"3.1 per-run tables (the four runs' twelve rows, fifteen figures each) and E51's reference rows: {total} figures compared")
    # the summary table
    summ = [l for l in md.splitlines() if l.startswith("| `blind` |")]
    n_summ = 0
    for l in summ:
        cells = [c.strip() for c in l.strip("|").split("|")]
        task, rule = cells[1], cells[2]; rows = mine[(task, rule)]; w, w40 = words(rows)
        fl = first_learned([(x["updates"], x["nt"]) for x in rows])
        red0, redl = cells[6].split("/")
        vals = [(cells[3], f"{w}" + (f" ({w40})" if w40 else "")), (cells[4], max(x["d"] for x in rows)), (cells[5], max(x["held"] for x in rows)),
                (red0, rows[0]["reduction"]), (redl, rows[-1]["reduction"]), (cells[7], str(fl) if fl else "not learned"), (cells[8], rows[-1]["nt"]), (cells[9], rows[-1]["ns"]),
                (cells[10], max(x["ff_state"] for x in rows)), (cells[11], max(x["out"] for x in rows)), (cells[12], rows[-1]["d40"])]
        for rep, v in vals:
            total += 1; n_summ += 1
            if not agrees(rep, v):
                diffs.append(f"summary {task} {rule}: reported {rep!r}, recomputed {v!r}")
    say(f"3.2 summary table: {n_summ} figures or words compared")
    # E55_READ_blind.json
    n_read = 0
    for k, v in rd.items():
        _, task, rule = k.split(); rows = mine[(task, rule)]; w, w40 = words(rows)
        fl = first_learned([(x["updates"], x["nt"]) for x in rows])
        checks = [(v["word"], w), (v["word_40_letters"], w40), (v["largest_d"], max(x["d"] for x in rows)), (v["largest_heldout_differ"], max(x["held"] for x in rows)),
                  (v["learned_from_row"], fl), (v["never_taught_last"], rows[-1]["nt"]), (v["never_taught_shut_last"], rows[-1]["ns"]),
                  (v["largest_reserve_state_first_family"], max(x["ff_state"] for x in rows)), (v["largest_output_change"], max(x["out"] for x in rows)),
                  (v["reduction_row0"], rows[0]["reduction"]), (v["reduction_last"], rows[-1]["reduction"]),
                  (v["differ_40_letters_by_row"], {str(x["updates"]): x["d40"] for x in rows}), (v["eigenvalue_by_row"], {str(x["updates"]): x["eig"] for x in rows}),
                  (v["first_failure_of_kept"], None)]
        for a, b in checks:
            total += 1; n_read += 1
            same = (abs(a - b) < 1e-9 * max(1, abs(b))) if isinstance(a, float) and isinstance(b, float) else a == b
            if not same:
                diffs.append(f"E55_READ_blind.json {k}: reported {a!r}, recomputed {b!r}")
    say(f"3.3 E55_READ_blind.json: {n_read} values compared")
    # the worker's message
    g = lambda t, r: mine[(t, r)]
    mx = lambda t, r, c: max(x[c] for x in g(t, r))
    at = lambda t, r, u, c: [x for x in g(t, r) if x["updates"] == u][0][c]
    fl = lambda t, r: first_learned([(x["updates"], x["nt"]) for x in g(t, r)])
    msg = [("runs complete, updates", "16384", min(REC[k]["complete"]["updates"] for k in REC))]
    for t, r, ff, lf, nt, ns in (("unrelated", "S", "1.3e-5", "8192", "99.0", "0.0"), ("unrelated", "B", "1.2e-3", "2048", "100.0", "0.0"),
                                 ("conflict", "S", "3.3e-6", None, "69.4", "34.9"), ("conflict", "B", "3.0e-5", None, "63.2", "28.9")):
        w, w40 = words(g(t, r))
        msg += [(f"{t} {r} word", "kept exactly", w), (f"{t} {r} largest d(g)", "0", mx(t, r, "d")), (f"{t} {r} largest held-out differ", "0", mx(t, r, "held")),
                (f"{t} {r} largest reserve state on the first family", ff, mx(t, r, "ff_state")), (f"{t} {r} learned from row", lf or "not learned", str(fl(t, r)) if fl(t, r) else "not learned"),
                (f"{t} {r} never-taught at 16,384", nt, at(t, r, 16384, "nt")), (f"{t} {r} with the reserve shut", ns, at(t, r, 16384, "ns"))]
    for u, v in ((2048, "21"), (4096, "45"), (8192, "194"), (16384, "278")):
        msg.append((f"unrelated B 40 letters differ at row {u}", v, at("unrelated", "B", u, "d40")))
    msg.append(("unrelated B 40-letter wording", "kept on items of up to 22 positions and not on items of 50", words(g("unrelated", "B"))[1]))
    for r, v in (("S", "92.40"), ("B", "95.25")):
        msg.append((f"reduction score unchanged at every row, rule {r}", v, g("unrelated", r)[0]["reduction"]))
        msg.append((f"reduction score the same at all rows of both tasks, rule {r}", "True", str(all(x["reduction"] == g("unrelated", r)[0]["reduction"] for t in TASKS for x in g(t, r)))))
    for (t, r), v in zip(RUNS, ("1.7e-3", "0.20", "9.9e-5", "3.5e-4")):
        msg.append((f"{t} {r} largest output change on held-out pairs", v, mx(t, r, "out")))
    msg.append(("eigenvalue at row 0", "0.605", g("unrelated", "S")[0]["eig"]))
    for (t, r), v in zip(RUNS, ("1.10", "1.43", "1.06", "1.20")):
        msg.append((f"{t} {r} eigenvalue at row 16,384", v, at(t, r, 16384, "eig")))
    n_msg = 0
    for label, rep, v in msg:
        total += 1; n_msg += 1
        if not agrees(rep, v):
            diffs.append(f"message, {label}: reported {rep!r}, recomputed {v!r}")
    say(f"3.4 the worker's message: {n_msg} figures or words compared")
    keep("figures_compared", total); keep("figures_differ", len(diffs))
    say(f"3.5 in all {total} figures compared, {len(diffs)} differ:")
    for d in diffs:
        say("      " + d)
    # the figures the table leaves out, from the receipts
    for (t, r), rows in mine.items():
        say(f"3.6 {t} {r}: largest state of a cell in reserve on the 300 pairs at 40 letters, by row: " + ", ".join(f"{x['updates']}: {x['ff40_state']:.2g}" for x in rows)
            + f"; largest 1 - z there at 16,384 {rows[-1]['ff40_1z']:.3f}; right answers at 40 letters at row 0 / 16,384: {rows[0]['ff40_exact']} / {rows[-1]['ff40_exact']} of 300")
        say(f"3.7 {t} {r}: largest output change on the held-out pairs by row: " + ", ".join(f"{x['updates']}: {x['out']:.2g}" for x in rows) + "; mean loss since the last row at 16,384 "
            f"{rows[-1]['loss']:.3g}; first row with a never-taught score of 80 or more {next((x['updates'] for x in rows if x['nt'] >= 80), None)}")
        keep(f"rows_{t}_{r}", rows)
    clock(3, t0)
    return mine


# ================================================================ section 4
def section4(start, mine):
    t0 = time.time(); say("===== SECTION 4. From the weights (the plain forward pass in numpy, double precision) =====")
    PAN = {t: V.new_panels(t) for t in TASKS}
    # the never-taught panels are not taught
    u_taught, u_held = V.u_split(); hp = {(it[1][0], it[1][1]) for it in PAN["unrelated"]["held"]}
    c_taught = {(t[0], a, b) for t in V.LIFE for a, b in V.split(t, .25, SEED)[0]}
    cp = set()
    for n, its in PAN["conflict"].items():
        op = n[:3]
        for p, a in its:
            s = bytes(p).decode(); x, y = int(s[1:3]), int(s[-3:-1]); cp.add((op, x, y))
    say(f"4.0 never-taught panels: unrelated, {len(PAN['unrelated']['held'])} items, pairs among the taught pairs {len(hp & set(u_taught))}; conflict, "
        f"{sum(len(v) for v in PAN['conflict'].values())} items ({list(PAN['conflict'])}), pairs among the taught quarter {len(cp & c_taught)}")
    for (task, rule) in RUNS:
        rec = REC[(task, rule)]; st = start[rule]; dirs = st["dirs"]; W0 = st["W"]
        for g in KEEP:
            r = row(rec, g)
            W, sh = load_weights(f"{E55}{r['weights']}")
            hashes = (sh == r["weights_sha256"], V.tensors_sha(W) == r["tensors_sha256"])
            net = V.Net(W)
            res, gens = V.score_first_family(net, S1, HELD)
            d = sum(a != b for n in gens if n.startswith("k0_") for a, b in zip(gens[n], st["gens"][n]))
            hd = sum(a != b for n in gens if n.startswith("k12_") for a, b in zip(gens[n], st["gens"][n]))
            red_ok = V.same_counts(res, V.receipt_first_family(r))
            # the generated bytes of the reduction pairs against the files of rows 64 and 4,096
            rb = ""
            if g in (64, 4096):
                fb = jshow(f"{E55}e55_{task}_blind_{rule}_{SEED}_reduction_bytes_row{g}.json")
                mineb = {n: [x for x, f in zip(gens["k0_" + n], S1[n][1]) if f] for n in ("add89", "sub89")}
                rb = f"; reduction pairs' bytes equal to the file of row {g}, item for item: {all(mineb[n] == fb[n] for n in mineb)}"
            lg = V.answer_logits(net, held); dout = float(np.abs(lg - st["logits"]).max()); ld = lead(lg)
            sbig, prof = states_by_position(net, held)
            g40 = V.generate(net, LEN40); d40 = sum(a != b for a, b in zip(g40, st["g40"]))
            s40, prof40 = states_by_position(net, LEN40)
            eig = radius(W)
            blind = {mn: float(np.abs(block(W, mn, BLINDSETS[mn]) @ dirs[mn]).max()) for mn in MATS}
            oldm = {mn: float(np.abs((block(W, mn, OLDSETS[mn]) - block(W0, mn, OLDSETS[mn])) @ dirs[mn]).max()) for mn in MATS}
            zeros = (not W["rnn.weight_hh_l0"][RES][:, :U0].any() and not W["rnn.bias_hh_l0"][RES].any() and not W["lm.weight"][ADDED][:, :U0].any() and not W["lm.bias"][ADDED].any())
            fixed = all(np.array_equal(W[k], W0[k]) for k in ("l1.weight", "l1.bias", "l3.bias")) and np.array_equal(W["l3.weight"][:, :H0], W0["l3.weight"][:, :H0]) \
                and np.array_equal(W["book.weight"][st["in_use"]], W0["book.weight"][st["in_use"]])
            x = [y for y in mine[(task, rule)] if y["updates"] == g][0]
            s3 = V.score_new(net, PAN[task]); s3s = V.score_new(net, PAN[task], shut_all=True)
            ps = ("add97", "sub97") if task == "conflict" else ("held",)
            nt = 100 * sum(s3[p]["exact"] for p in ps) / sum(s3[p]["n"] for p in ps); ns = 100 * sum(s3s[p]["exact"] for p in ps) / sum(s3s[p]["n"] for p in ps)
            rec_new = r["score3_new_task"] == {k: dict(n=v["n"], exact=v["exact"], final=v["final"]) for k, v in s3.items()}
            rec_shut = r["score3_new_task_reserve_shut"] == {k: dict(n=v["n"], exact=v["exact"], final=v["final"]) for k, v in s3s.items()}
            # single precision (numpy float32), the precision in which the runner computed its figures
            n32 = V.Net(W, np.float32); gens32 = V.score_first_family(n32, S1, HELD)[1]
            d32 = sum(a != b for n in gens32 if n.startswith("k0_") for a, b in zip(gens32[n], st["gens32"][n]))
            hd32 = sum(a != b for n in gens32 if n.startswith("k12_") for a, b in zip(gens32[n], st["gens32"][n]))
            dout32 = float(np.abs(V.answer_logits(n32, held) - st["logits32"]).max()); sbig32 = states_by_position(n32, held)[0]
            g4032 = V.generate(n32, LEN40); d4032 = sum(a != b for a, b in zip(g4032, st["g4032"])); s4032 = states_by_position(n32, LEN40)[0]
            keep(f"w_{task}_{rule}_{g}", dict(lead=ld, d=d, held=hd, out=dout, state=sbig, d40=d40, state40=s40, eig=eig, blind=blind, old=oldm, nt=nt, ns=ns,
                                             d32=d32, held32=hd32, out32=dout32, state32=sbig32, d40_32=d4032, state40_32=s4032))
            say(f"4.1 {task} {rule} row {g}: file and tensors hashes equal to the receipt {hashes}; first family's answers that differ from the start's (numpy): {d} of 31,684 (receipt {x['d']}), "
                f"held-out {hd} of 1,200 (receipt {x['held']}); every count of the receipt's scores 1 and 2 reproduced {red_ok}{rb}")
            say(f"4.2 {task} {rule} row {g}: largest change of an output on the held-out pairs {dout:.3e} (receipt {x['out']:.3e}); largest state of a cell in reserve on the held-out pairs "
                f"{sbig:.3e} (receipt {x['ff_state']:.3e}), at positions 1, 6, 11, 16, 21: " + ", ".join(f"{prof[t]:.1e}" for t in (1, 6, 11, 16, 21))
                + f"; eigenvalue {eig:.4f} (receipt {x['eig']:.4f}); smallest lead of the largest output over the second on the held-out pairs {ld:.2e}")
            say(f"4.3 {task} {rule} row {g}: 40-letter panel (300 held-out pairs of addition, 50 positions under teacher forcing): answers that differ from the start's {d40} (receipt {x['d40']}); "
                f"right answers {V.exact(g40, LEN40)} (receipt {x['ff40_exact']}); largest state of a cell in reserve {s40:.3e} (receipt {x['ff40_state']:.3e}), at positions 11, 21, 31, 41, 49: "
                + ", ".join(f"{prof40[t]:.1e}" for t in (11, 21, 31, 41, 49)))
            say(f"4.3b {task} {rule} row {g}, single precision: answers that differ from the start's {d32} of 31,684 and {hd32} of 1,200; largest change of an output {dout32:.3e}; "
                f"largest state on the held-out pairs {sbig32:.3e}; 40 letters: answers that differ {d4032}, largest state {s4032:.3e}")
            say(f"4.4 {task} {rule} row {g}: rows kept blind, largest |row @ Vm| (whole row with its bias): " + ", ".join(f"{m} {v:.1e}" for m, v in blind.items())
                + "; old rows, largest |(row - start row) @ Vm|: " + ", ".join(f"{m} {v:.1e}" for m, v in oldm.items()) + f"; `zero_blocks` entries exactly zero {zeros}; "
                f"l1, l3's old columns and bias, the byte table's rows in use bit-equal to the start {fixed}")
            say(f"4.5 {task} {rule} row {g}: new task's never-taught score {nt:.1f} (receipt {x['nt']:.1f}), with the reserve shut {ns:.1f} (receipt {x['ns']:.1f}); every count of the "
                f"receipt's score 3 reproduced: as run {rec_new}, shut {rec_shut}")
            del W, net, n32
    say(f"4.6 free disk after the weight files {disk_ok()[0] / 1e9:.2f} GB")
    clock(4, t0)


# ================================================================ section 5
def section5(mine):
    t0 = time.time(); say("===== SECTION 5. The fixed wording and the predictions =====")
    lf = {}
    for (t, r), rows in mine.items():
        w, w40 = words(rows); fl = first_learned([(x["updates"], x["nt"]) for x in rows]); lf[(t, r)] = fl
        first40 = next((x["updates"] for x in rows if x["d40"] > 3), None)
        say(f"5.1 {t} {r}: d(g) 0 at every row {all(x['d'] == 0 for x in rows)}; held-out 0 at every row {all(x['held'] == 0 for x in rows)}; reduction score at every row equal to row 0's "
            f"{all(x['reduction'] == rows[0]['reduction'] for x in rows)}; word: {w}; more than 3 of the 300 answers at 40 letters differ at some row: {max(x['d40'] for x in rows) > 3} "
            f"(first at row {first40}); wording required: {w40 or w}; new task: {'learned from row ' + str(fl) if fl else 'not learned'}, "
            f"never-taught {rows[-1]['nt']:.1f} as run and {rows[-1]['ns']:.1f} with the reserve shut at 16,384, difference {rows[-1]['nt'] - rows[-1]['ns']:.1f} points")
        say(f"5.1b {t} {r}, by row (updates: never-taught / with the reserve shut / 40-letter answers that differ / largest state on the held-out pairs): "
            + "; ".join(f"{x['updates']}: {x['nt']:.1f} / {x['ns']:.1f} / {x['d40']} / {x['ff_state']:.2e}" for x in rows))
        if t == "unrelated" and fl:
            e51 = {a: first_learned([(x["updates"], x["nt"]) for x in recompute(jshow(f"{E51}e51_{t}_{a}_{r}_{SEED}.json"))]) for a in ("keyed", "open")}
            say(f"5.2 {t} {r}: learned from row {fl}; E51's `keyed` from row {e51['keyed']}, `open` from row {e51['open']}; ratio to `keyed` {fl / e51['keyed']:.0f} "
                f"({int(round(math.log2(fl / e51['keyed'])))} scored rows of doubling)")
    allrows = [x for rows in mine.values() for x in rows]
    p9 = all(words(rows)[0] == "kept exactly" for rows in mine.values())
    p10 = all(x["d"] <= 31 for x in allrows)
    big = {k: max(x["ff_state"] for x in rows) for k, rows in mine.items()}
    p11 = all(v < 1e-4 for v in big.values())
    first11 = {k: next((x["updates"] for x in rows if x["ff_state"] >= 1e-4), None) for k, rows in mine.items()}
    p12 = all(lf[("unrelated", r)] for r in RULES)
    p13 = not any(lf[("conflict", r)] for r in RULES)
    p1 = all(jshow(f"{E55}checks/e55_check_{n}_{SEED}.json")["ok"] for n in (1, 2, 3))
    for k, v in (("P55.1", p1), ("P55.9", p9), ("P55.10", p10), ("P55.11", p11), ("P55.12", p12), ("P55.13", p13)):
        keep(k, v)
    say(f"5.3 P55.1 (checks 1 to 3 pass at the first attempt; one receipt each, ok): {p1}")
    say(f"5.4 P55.9 (kept exactly in all four runs, by the definition): {p9}")
    say(f"5.5 P55.10 (d(g) at most 31 at every row of all four runs): {p10}; largest d(g) {max(x['d'] for x in allrows)}")
    say(f"5.6 P55.11 (largest state below 1e-4 at every row of every run): {p11}; largest by run " + ", ".join(f"{k[0]} {k[1]} {v:.2e}" for k, v in big.items())
        + "; first row at or above 1e-4 " + ", ".join(f"{k[0]} {k[1]} {v}" for k, v in first11.items()))
    say(f"5.7 P55.12 (unrelated learned under both rules): {p12}; P55.13 (conflict learned under neither rule): {p13}")
    clock(5, t0)


# ================================================================ section 6
def section6(src):
    t0 = time.time(); say("===== SECTION 6. The runner as text =====")
    r51 = src["exp51_reserve.py"].decode().splitlines(); r55 = src["exp55_reserve.py"].decode().splitlines()
    a = next(i for i, l in enumerate(r51) if l.startswith("def fwd(")); b = next(i for i, l in enumerate(r51) if l.startswith("def install("))
    xs = [(i + 1, l.strip()) for i, l in enumerate(r51[a:b], a) if re.search(r"\bx\[", l)]
    say(f"6.1 exp51_reserve.fwd (lines {a + 1} to {b}): lines that index the item x: {xs}; the enclosing condition of the first: "
        f"{[l.strip() for l in r51[a:b] if 'KEY[\"in_use\"] is not None' in l]}")
    keys = [(i + 1, l.strip()) for i, l in enumerate(r55) if "KEY[" in l or "install(" in l]
    say(f"6.2 exp55_reserve.py, every line that sets or reads KEY or calls install: {keys}")
    say(f"6.3 in exp55_reserve.py the words 'in_use' occur on lines {[i + 1 for i, l in enumerate(r55) if 'in_use' in l]}")
    rb = next(i for i, l in enumerate(r55) if l.startswith("class Rule55")); re_ = next(i for i, l in enumerate(r55) if l.startswith("# ----") and i > rb)
    say(f"6.4 Rule55 (lines {rb + 1} to {re_}); E51's Rule.after restores: {[l.strip() for l in r51 if 'copy_(p0)' in l or 'snap[\"l3\"]' in l or 'snap[\"book\"]' in l]}")
    say(f"6.5 the arguments of `fwd` in the update (exp51_reserve.step): {[l.strip() for l in r51 if 'logits = fwd(' in l]}")
    say(f"6.6 the stream assertions in the run loop (read the bytes of the batch, change nothing): {[(i + 1, l.strip()) for i, l in enumerate(r55) if 'assert all(b' in l]}")
    clock(6, t0)


# ================================================================ section 7
def section7(mine):
    t0 = time.time(); say("===== SECTION 7. Figures used in the reading =====")
    for (t, r), rows in mine.items():
        e = rows[-1]["eig"]
        say(f"7.1 {t} {r}: eigenvalue at 16,384 {e:.3f}; to the power 21 {e ** 21:.3g}, to the power 49 {e ** 49:.3g}; largest over the rows {max(x['eig'] for x in rows):.3f}; "
            f"rows with eigenvalue above 1: {[x['updates'] for x in rows if x['eig'] > 1]}")
    clock(7, t0)


# ================================================================ section 8
def section8():
    t0 = time.time(); say("===== SECTION 8. Form =====")
    files = subprocess.run(["git", "ls-tree", "-r", "--name-only", REF, "--", E55], capture_output=True, text=True, check=True).stdout.split()
    texts = [f for f in files if not f.endswith(".pt")] + [M + "exp55_reserve.py", M + "e55_read.py"]
    body = {f: show(f).decode() for f in texts}
    marks = {m: sum(t.count(m) for t in body.values()) for m in ("@@", "TODO", "TBD", "XX")}
    where = {m: sorted({f.split("/")[-1] for f, t in body.items() if m in t}) for m in marks}
    say(f"8.1 text files searched {len(texts)}; unfilled marks: " + ", ".join(f"{m!r} {marks[m]}" for m in marks) + f"; files where a mark occurs {where}")
    tbl = body[E55 + "E55_TABLES_blind.md"]
    say(f"8.2 E55_TABLES_blind.md: unfilled marks {sum(tbl.count(m) for m in marks)}; lines {len(tbl.splitlines())}")
    # words taken from the trailers of the worker's commit (the agent's name before ':session'), and words from an optional environment variable
    msg = subprocess.run(["git", "log", "-1", "--format=%B", REF], capture_output=True, text=True, check=True).stdout
    words = {w.lower() for w in re.findall(r"Acting-Agent:\s*(\w+):", msg)} | {w.lower() for w in re.findall(r"^(\w+)-Session:", msg, flags=re.M)}
    extra = [w.lower() for w in os.environ.get("E55_EXTRA_WORDS", "").split(",") if w.strip()]
    allw = sorted(words | set(extra))
    hits = {f: sum(len(re.findall(re.escape(w), t.lower())) for w in allw) for f, t in body.items()}
    kinds = {}
    for f, n in hits.items():
        if n:
            k = "receipt (json)" if f.endswith(".json") else "log" if f.endswith(".log") else "table (md)" if f.endswith(".md") else "script (py)"
            kinds[k] = kinds.get(k, 0) + n
    say(f"8.3 search for {len(words)} word(s) taken from the commit's trailers and {len(extra)} word(s) from E55_EXTRA_WORDS, case-insensitive, in the {len(texts)} text files: "
        f"occurrences {sum(hits.values())}; by kind of place {kinds}")
    clock(8, t0)


if __name__ == "__main__":
    say(f"E55, arm blind, verification at {REF}; numpy {np.__version__}")
    src = section1()
    start = section2()
    mine = section3()
    section4(start, mine)
    section5(mine)
    section6(src)
    section7(mine)
    section8()
    say(f"running time {time.time() - T0:.0f} s")
