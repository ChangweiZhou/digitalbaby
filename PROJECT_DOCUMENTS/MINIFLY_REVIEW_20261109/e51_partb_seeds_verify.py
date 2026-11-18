"""Verification of E51, Part B (B1 and B2) on seeds 93683 and 116089, and of section 9 of E51_REPORT.md, at the worker's commit
2abb0c3f (the head of the worker's E51 branch when it was fetched).

Read-only.  Nothing is trained, torch is not used, and no code of another branch is run: every file is read with `git show` into
memory, and nothing is written.  The functions of the verification of seed 73129 (`e51_partb_verify.py` beside this file) and,
through it, of Part A (`e51_trial_verify.py`: items, streams, the digest of the batches, the reader of weight files, the forward
pass and the generation of answers) are imported and pointed at commit 2abb0c3f and at each seed in turn.

  1  what changed between f3bba94e (the commit verified for seed 73129) and 2abb0c3f: files added, modified and removed; lines
     removed from the report; the report's sections 1 to 8 byte for byte; every file of seed 73129; the runner's hash
  2  the receipts of the two new seeds: completeness and settings of the 28 run receipts (B1 x 8, B2 x 2, B3 at 0.001 x 2 and the
     two runs of check 8, per seed); the data digests at every scored row recomputed from the reader's own streams; the probe
     items; B2's cycles from the trace of probes; every cell of E51_TABLES_B_SEEDS.md, every value of E51_READ_B_SEEDS.json and
     every figure of section 9 of the report recomputed and compared (all three seeds)
  3  the predictions PB.1 to PB.4 on each seed by the brief's fixed rules; the three-seed statements of section 9; the exact
     margins and the other readings of "within 5 points"
  4  the checks 8 to 11 of each new seed against the receipts they name and the reader's own forward pass on the births; the
     order of events from the logs and the commits; the six aborted starts of seed 93683
  5  the weights: the SHA-256 of every weight file of the new seeds against its receipt; from the final weights of B1 at k = 4
     (seed 93683, rule B) and B1 at k = 4 (seed 116089, rule S): the first family's counts, the new task's panels, the last probe

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e51_partb_seeds_verify.py [1 2 3 4 5]
"""
import hashlib, json, os, re, subprocess, sys, time
from random import Random

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WANT = sys.argv[1:] or list("12345")
sys.argv = sys.argv[:1]                       # the imported scripts read their own argument lists at import
import numpy as np                            # noqa: E402
import e51_partb_verify as M                  # noqa: E402  (its parts run only under __main__)

V = M.V
REF = "2abb0c3f8362d115c6465712b747dfe573fe94a6"   # the worker's commit under verification
OLD = "f3bba94edc212fa6a13ad3b4bb2a1b1d459fef59"   # the commit whose Part B on seed 73129 was verified
RUNNER_VERIFIED = "608f6b4c"                     # the commit at which exp51_partb.py was verified with seed 73129
DREF = V.DREF
P, B = M.P, M.B
SEEDS = (93683, 116089)
ALL = (73129, 93683, 116089)
RULES, KS, E43ROWS = M.RULES, M.KS, M.E43ROWS
diffs = []


def show(path, ref=REF):
    return V.show(path, ref)


def js(path, ref=REF):
    return json.loads(show(path, ref))


# point the imported functions at the commit under verification
M.WREF = REF; M.show = show; M.js = js


def use(seed):
    """Point the imported functions at a seed (they read the module-level SEED at call time)."""
    V.SEED = seed; M.SEED = seed


def note(ok, what):
    if not ok:
        diffs.append(what)
    return "agrees" if ok else "DIFFERS"


def clock(label, t0):
    print(f"[part {label}: {time.time() - t0:.0f} s]\n", flush=True)


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout


def report():
    return show(f"{P}e51/E51_REPORT.md").decode()


# ================================================================ part 1: what changed since f3bba94e
def part1():
    t0 = time.time(); print("===== PART 1. What changed between f3bba94e and 2abb0c3f =====")
    print(f"1.1 f3bba94e is an ancestor of 2abb0c3f: {subprocess.run(['git', 'merge-base', '--is-ancestor', OLD, REF]).returncode == 0}; "
          f"commits between them: {len(git('rev-list', f'{OLD}..{REF}').split())}")
    ns = [l.split("\t") for l in git("diff", "--name-status", OLD, REF).splitlines()]
    kinds = {}
    for k, *_ in ns:
        kinds[k] = kinds.get(k, 0) + 1
    print(f"1.2 files by status (A added, M modified, D deleted, R renamed): {kinds}; modified: {[p for k, p, *_ in ns if k != 'A']}")
    s73 = [p for k, p, *_ in ns if "73129" in p]
    print(f"1.3 files whose name contains 73129 among the changed files: {len(s73)} {note(not s73, 'a file of seed 73129 changed')}")
    num = [l.split("\t") for l in git("diff", "--numstat", OLD, REF).splitlines()]
    removed = sum(int(a[1]) for a in num if a[1] != "-")
    print(f"1.4 lines removed over all text files: {removed}; files with lines removed: {[a[2] for a in num if a[1] not in ('-', '0')]}")
    old, new = show(f"{P}e51/E51_REPORT.md", OLD).decode(), report()
    print(f"1.5 the report: {len(old.splitlines())} lines at f3bba94e, {len(new.splitlines())} at 2abb0c3f; the old report is a prefix of the new one byte for byte: "
          f"{note(new.startswith(old), 'report sections 1 to 8 changed')}; added after it: {len(new[len(old):].splitlines())} lines, beginning "
          f"{new[len(old):].strip().splitlines()[0][:60]!r}")
    # the files of seed 73129 (receipts, weights, logs, the references, the tables) by blob identity
    tree_old = {l.split()[3]: l.split()[2] for l in git("ls-tree", "-r", OLD, "--", P).splitlines()}
    tree_new = {l.split()[3]: l.split()[2] for l in git("ls-tree", "-r", REF, "--", P).splitlines()}
    f73 = [p for p in tree_old if "73129" in p]
    same = sum(tree_new.get(p) == tree_old[p] for p in f73)
    other = [p for p in tree_old if p.startswith(B) and "73129" not in p and tree_new.get(p) != tree_old[p]]
    print(f"1.6 files of seed 73129 under exp45/ at f3bba94e: {len(f73)}; with the same blob at 2abb0c3f: {same} {note(same == len(f73), 'seed 73129 files')}; "
          f"other files of partB/ changed: {other or 'none'} (E51_TABLES_B.md and E51_READ_B.json included)")
    for f in ("MOD97_LOCAL_RULE_20261106/exp51_partb.py", "MOD97_LOCAL_RULE_20261106/e51_read.py"):
        a, b = hashlib.sha256(show(f, RUNNER_VERIFIED)).hexdigest(), hashlib.sha256(show(f)).hexdigest()
        print(f"1.7 {f.split('/')[-1]}: SHA-256 at 608f6b4c {a[:16]}, at 2abb0c3f {b[:16]}: {note(a == b, 'runner changed ' + f)}")
    clock(1, t0)


# ================================================================ part 2: the receipts, the tables, the readings, section 9
def b2_digest(rec):
    unrel = V.u_stream(); fam = V.family_stream(); dg = V.Digest()
    for c in rec["cycles"]:
        for _ in range(256):
            dg.add([next(unrel) for _ in range(64)])
        for _ in range(c["review_updates"]):
            dg.add([next(fam) for _ in range(64)])
    return dg.value()


def receipts(seed, shas):
    use(seed); bad = []
    for k in list(KS) + ["k2addendum"]:
        dig, counts = M.digests_of(M.b1_source(k), set(E43ROWS), 4096)
        for rule in RULES:
            path = f"{B}{'check8/' if k == 'k2addendum' else ''}e51_B1_share_{k}_{rule}_{seed}.json"; rec = js(path)
            shas.add(rec["settings"]["provenance"]["e51_sources_sha256"]["exp51_partb.py"])
            pb = M.check_receipt(rec, f"e51_B1_share_{k}_{rule}_{seed}", dict(set="B1", k=str(k), rule=rule, updates=4096), E43ROWS, 4096, f"e51_B1_share_{k}_{rule}_{seed}_row4096.pt")
            lr = {g["lr"] for g in rec["settings"]["optimizer_groups_read_back"] + rec["complete"]["optimizer_groups_read_back"]}
            prow = [p["updates"] for p in rec["probe"]] == sorted(set(range(0, 1025, 64)) | set(range(1024, 4097, 256)))
            dok = all(V.row(rec, g)["data_digest"] == dig[g] for g in E43ROWS) and rec["complete"]["data_digest"] == dig[4096]
            cok = rec["complete"]["items"] == counts
            ff = counts["first_family"] / 4096 / 64
            print(f"    {seed} B1 {k:>10} {rule}: problems {pb or 'none'}; lr {sorted(lr)}; probe rows as the brief {prow}; digests at the six rows equal to the reader's "
                  f"{note(dok, f'digest B1 {k} {rule} {seed}')}; item counts equal {note(cok, f'items B1 {k} {rule} {seed}')}; share of first-family items {ff:.4f}")
            if pb or not prow or lr != {0.001}:
                bad.append(f"settings B1 {k} {rule} {seed}: {pb}")
    for rule in RULES:
        rec = M.rec_b2(rule); s = rec["settings"]; pb = []
        shas.add(s["provenance"]["e51_sources_sha256"]["exp51_partb.py"])
        for kk, v in dict(set="B2", rule=rule, gap=256, review_max=1024, cycles=8, review_probe_every=16, decay=0.0, seed=seed, batch=64).items():
            if s.get(kk) != v:
                pb.append(kk)
        if s["start"]["birth"] != f"exp45/e26/e26_dev_{rule}_{seed}.pt" or rec["complete"]["optimizer_steps"] != [rec["complete"]["updates"]]:
            pb.append("birth or one optimizer")
        if any(g["weight_decay"] != 0.0 or g["lr"] != 0.001 for g in s["optimizer_groups_read_back"] + rec["complete"]["optimizer_groups_read_back"]):
            pb.append("lr/decay")
        if s["local"] != (rule == "S") or rec["complete"]["weights"] != f"e51_B2_intervals_{rule}_{seed}_end.pt":
            pb.append("local or weights")
        if s["provenance"]["git"]["sources_modified_vs_commit"] or s["provenance"]["git"]["e51_sources_modified_vs_commit"]:
            pb.append("sources modified")
        n = rec["complete"]["updates"]; tot = 8 * 256 + sum(c["review_updates"] for c in rec["cycles"])
        dg = b2_digest(rec); dok = rec["complete"]["data_digest"] == dg and rec["rows"][-1]["data_digest"] == dg and rec["rows"][0]["updates"] == 0
        print(f"    {seed} B2 {rule}: problems {pb or 'none'}; {len(rec['cycles'])} cycles; updates {n} (8 x 256 + reviews = {tot}); E43's rows at {[r['updates'] for r in rec['rows']]}; "
              f"digest of all {dg['batches']} batches equal to the reader's {note(dok and n == tot, f'digest B2 {rule} {seed}')}")
        if pb:
            bad.append(f"settings B2 {rule} {seed}: {pb}")
    st = V.u_stream(); dig, _ = M.digests_of(lambda: ("unrelated", next(st)), {0, 64, 4096}, 4096)
    for rule in RULES:
        rec = M.rec_b3("0.001", rule)
        shas.add(rec["settings"]["provenance"]["e51_sources_sha256"]["exp51_partb.py"])
        pb = M.check_receipt(rec, f"e51_B3_steps_0.001_{rule}_{seed}", dict(set="B3", learning_rate=0.001, rule=rule, updates=4096), [0, 64, 4096], 4096, f"e51_B3_steps_0.001_{rule}_{seed}_row4096.pt")
        prow = [p["updates"] for p in rec["probe"]] == sorted(set(range(0, 257, 16)) | set(range(256, 4097, 64)))
        dok = all(V.row(rec, g)["data_digest"] == dig[g] for g in (0, 64, 4096))
        print(f"    {seed} B3 0.001 {rule} (the run of check 9): problems {pb or 'none'}; probe rows as the brief {prow}; digests equal to the reader's {note(dok, f'digest B3 {rule} {seed}')}")
        if pb or not prow:
            bad.append(f"settings B3 {rule} {seed}: {pb}")
    S1 = V.first_family_items(); rng = Random(f"e51|{seed}|probe")
    mine = {n: rng.sample([i for i, f in enumerate(S1[n][1]) if f], 200) for n in ("add89", "sub89")}
    allp = [M.rec_b1(k, r)["settings"]["probe_items_first_family"] for k in KS for r in RULES] + [M.rec_b2(r)["settings"]["probe_items_first_family"] for r in RULES] \
        + [M.rec_b3("0.001", r)["settings"]["probe_items_first_family"] for r in RULES]
    print(f"    {seed}: the probe's 400 items drawn by the reader equal the indices in all 12 receipts: {note(all(p == mine for p in allp), f'probe items {seed}')}")
    for rule in RULES:
        rec, d = M.read_b2(rule)
        same = all(rc["review_updates"] == c["review"] and rc["probe_at_end_of_gap"]["first_family_pct"] == c["gap_ff"] and rc["probe_at_end_of_review"]["first_family_pct"] == c["rev_ff"]
                   and rc["review_reached_goal"] == c["reached"] and c["first_reach"] == (c["review"] or None) and c["gap_probes_ok"] for rc, c in zip(rec["cycles"], d["cycles"]))
        g_ok = abs(rec["settings"]["probe_first_family_pct_at_update_0"] - d["p0"]) < 1e-12 and abs(rec["settings"]["review_goal_pct"] - d["goal"]) < 1e-9
        print(f"    {seed} B2 {rule}: the eight cycle records equal the trace of probes; every review stopped at the first probe at or above the goal; four probes in each gap: "
              f"{note(same, f'B2 cycles vs trace {rule} {seed}')}; probe at 0 and goal in the settings equal the trace's {note(g_ok, f'B2 goal {rule} {seed}')}")
    diffs.extend(bad)


def readings(seed):
    """The reader's own readings of one seed, in the structure of E51_READ_B_SEEDS.json (B2 from the trace of probes)."""
    use(seed); out = dict(B1={}, B2={}); extra = {}
    for rule in RULES:
        o, least = M.read_b1(rule); extra[("B1", rule)] = (o, least)
        out["B1"][rule] = dict(shares={str(k): dict(reduction_row0=d["r0"], reduction_4096=d["r4"], pct_of_row0=d["pct"], kept=d["kept"], lowest_probe=d["low"],
                                                    unrelated_80_first_probe_row=d["f80"], unrelated_held_4096=d["held4096"]) for k, d in o.items()},
                               least_share_that_keeps=None if least is None else f"1/{least}", PB1_quarter_keeps=o[4]["kept"], PB2_thirty_second_does_not_keep=not o[32]["kept"])
    for rule in RULES:
        rec, d = M.read_b2(rule); extra[("B2", rule)] = d
        out["B2"][rule] = dict(cycles=[dict(cycle=c["cycle"], gap_end_first_family_pct=c["gap_ff"], gap_end_unrelated_held_pct=c["gap_un"], review_updates=c["review"],
                                            review_reached_goal=c["reached"], review_end_first_family_pct=c["rev_ff"], review_end_unrelated_held_pct=c["rev_un"]) for c in d["cycles"]],
                               probe_at_update_0=d["p0"], goal=d["goal"], review_shortens=d["shortens"], more_is_left=d["more"], updates=d["end"],
                               reduction_row0=d["red0"], reduction_end=d["red_end"], unrelated_held_end=d["held_end"],
                               sentence=f"with rehearsal at intervals the first family {'becomes' if d['shortens'] else 'does not become'} cheaper to restore, and "
                                        f"{'more' if d['more'] else 'no more'} of it survives a gap")
    return out, extra


def compare(a, b, path, bad, n):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                bad.append((path + "/" + k, "missing on one side")); continue
            compare(a[k], b[k], path + "/" + k, bad, n)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            bad.append((path, "length"))
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", bad, n)
    else:
        n[0] += 1
        if isinstance(a, float) or isinstance(b, float):
            if a is None or b is None or abs(a - b) > 1e-9:
                bad.append((path, a, b))
        elif a != b:
            bad.append((path, a, b))


def md_rows(R):
    """The rows of the three-seed tables, formatted from the reader's readings (the same columns as E51_TABLES_B_SEEDS.md)."""
    b1, b1b, b2 = [], [], []
    for seed in ALL:
        for rule in RULES:
            r = R[seed]["B1"][rule]; sh = r["shares"]
            cell = lambda k: f"{sh[k]['reduction_4096']:.1f} ({sh[k]['pct_of_row0']:.1f}%{', kept' if sh[k]['kept'] else ''})"
            b1.append([str(seed), rule, f"{sh['4']['reduction_row0']:.1f}"] + [cell(k) for k in ("2", "4", "8", "16", "32", "0")]
                      + [r["least_share_that_keeps"] or "none", "held" if r["PB1_quarter_keeps"] else "failed", "held" if r["PB2_thirty_second_does_not_keep"] else "failed"])
            for k in KS:
                v = sh[str(k)]
                b1b.append([str(seed), rule, f"1/{k}", f"{v['lowest_probe']:.2f}", str(v["unrelated_80_first_probe_row"]), f"{v['unrelated_held_4096']:.1f}"])
            q = R[seed]["B2"][rule]
            b2.append([str(seed), rule, f"{q['probe_at_update_0']:.2f}"] + [f"{c['gap_end_first_family_pct']:.2f} / {c['review_updates']}" for c in q["cycles"]]
                      + ["yes" if q["review_shortens"] else "no", "yes" if q["more_is_left"] else "no", str(q["updates"]), f"{q['reduction_row0']:.1f} / {q['reduction_end']:.1f}",
                         f"{q['unrelated_held_end']:.1f}"])
    return b1, b1b, b2


def cells(text, first):
    return [[x.strip() for x in l.strip().strip("|").split("|")] for l in text.splitlines() if re.match(rf"\| ({first}) \|", l)]


def part2():
    t0 = time.time(); print("===== PART 2. The receipts of the new seeds; the tables, the readings and section 9 =====")
    print("2.1 completeness, settings, data digests from the reader's own streams, probe items, B2 cycles from the trace")
    shas = set()
    for seed in SEEDS:
        receipts(seed, shas)
    runner = hashlib.sha256(show("MOD97_LOCAL_RULE_20261106/exp51_partb.py")).hexdigest()
    print(f"    the runner's hash recorded in the 28 receipts: {sorted(shas)}; equal to exp51_partb.py at 2abb0c3f: {note(shas == {runner}, 'runner hash in receipts')}")
    R, X = {}, {}
    for seed in ALL:
        R[seed], X[seed] = readings(seed)
    globals()["_R"], globals()["_X"] = R, X
    # the readings file
    theirs = js(f"{B}E51_READ_B_SEEDS.json"); bad, n = [], [0]
    top_ok = theirs["experiment"] == "E51" and theirs["seeds"] == list(ALL)
    compare({str(s): v for s, v in R.items()}, theirs["readings"], "", bad, n)
    print(f"2.2 E51_READ_B_SEEDS.json: seeds {theirs['seeds']}; {n[0]} values (sentences included) against the reader's: {note(not bad and top_ok, 'readings file')}"
          + (f"; differences {bad[:12]}" if bad else ""))
    # the tables file
    tables = show(f"{B}E51_TABLES_B_SEEDS.md").decode(); b1, b1b, b2 = md_rows(R)
    t1 = [r for r in cells(tables, r"\d{5,6}") if len(r) == 12]; t1b = [r for r in cells(tables, r"\d{5,6}") if len(r) == 6]
    t2 = [r for r in cells(tables, r"\d{5,6}") if len(r) == 16]
    for name, a, m in (("B1 main", t1, b1), ("B1 probe", t1b, b1b), ("B2", t2, b2)):
        badr = [(x, y) for x, y in zip(a, m) if x != y]
        print(f"2.3 E51_TABLES_B_SEEDS.md, {name}: {len(a)} rows, {sum(map(len, a))} cells, against the reader's {len(m)} rows: {note(not badr and len(a) == len(m), 'tables ' + name)}")
        for x, y in badr:
            print("     worker:", x, "\n     reader:", y)
    sent = [l[2:] for l in tables.splitlines() if l.startswith("- seed ")]
    mine_s = [f"seed {s}, rule {r}: {R[s]['B2'][r]['sentence']}." for s in ALL for r in RULES]
    print(f"    the six sentences of B2: {note(sent == mine_s, 'tables sentences')}")
    allgoal = all(c["review_reached_goal"] for s in ALL for r in RULES for c in R[s]["B2"][r]["cycles"])
    print(f"    every review reached its goal (all 48 cycles): reader {allgoal}; tables say True for all six: {note(allgoal and tables.count(': True') == 6, 'reached goal')}")
    # section 9 of the report
    rep = report(); sec = rep[rep.index("## 9. Part B, B1 and B2"):]
    r1 = [r for r in cells(sec, r"\d{5,6}") if len(r) == 12]; r2 = [r for r in cells(sec, r"\d{5,6}") if len(r) == 16]
    b2c = [x[:13] + [x[13].replace(",", "")] + x[14:] for x in r2]
    print(f"2.4 section 9, B1 table: {len(r1)} rows: {note(r1 == b1, 'section 9 B1 table')}; B2 table: {len(r2)} rows (thousands separators removed): {note(b2c == b2, 'section 9 B2 table')}")
    for x, y in [(x, y) for x, y in zip(b2c, b2) if x != y] + [(x, y) for x, y in zip(r1, b1) if x != y]:
        print("     report:", x, "\n     reader:", y)
    ss = re.findall(r"- Seed (\d+), (S|B): (.*?)\n", sec)
    print(f"    the four sentences of section 9.3: {note([(int(a), b, c) for a, b, c in ss] == [(s, r, R[s]['B2'][r]['sentence'] + '.') for s in SEEDS for r in RULES], 'section 9 sentences')}")
    # the prose figures of section 9
    d = lambda s, r, k: X[s][("B1", r)][0][k]["r4"] - X[s][("B1", r)][0][k]["r0"]
    q = [("1/4: 3.728 below (93683, S)", -d(93683, "S", 4), 3.728), ("1/4: 4.980 below (93683, B)", -d(93683, "B", 4), 4.980),
         ("1/4: 7.240 below (116089, S)", -d(116089, "S", 4), 7.240), ("1/4: 9.551 below (116089, B)", -d(116089, "B", 4), 9.551),
         ("1/2: 3.498 below (116089, S)", -d(116089, "S", 2), 3.498), ("1/2: 4.699 above (116089, B)", d(116089, "B", 2), 4.699)]
    print("2.5 section 9.2, the exact distances from row 0 (points)")
    for t, got, said in q:
        print(f"    {t}: receipts {got:.6f}: {note(round(got, 3) == said, 'section 9: ' + t)}")
    held = [X[s][("B1", r)][0][k]["held4096"] for s in ALL for r in RULES for k in (2, 4, 8, 16, 32, 0)]
    held_b1 = [X[s][("B1", r)][0][k]["held4096"] for s in ALL for r in RULES for k in KS]
    print(f"    'the unrelated held score at 4,096 is between 98.7% and 100% in every run of the three seeds': B1's 24 runs {min(held_b1):.2f} to {max(held_b1):.2f}; "
          f"with the addendum and E43 (36 runs) {min(held):.2f} to {max(held):.2f}: {note(round(min(held_b1), 1) >= 98.7, 'held range')}")
    # CPU of the new seeds
    print("2.6 section 9.1, CPU-hours from the receipts' cpu_total, and the check receipts' figures")
    for seed in SEEDS:
        use(seed)
        runs = sum(M.rec_b1(k, r)["complete"]["cpu_total"] for k in KS for r in RULES) + sum(M.rec_b2(r)["complete"]["cpu_total"] for r in RULES)
        chk = sum(js(f"{B}check8/e51_B1_share_k2addendum_{r}_{seed}.json")["complete"]["cpu_total"] + M.rec_b3("0.001", r)["complete"]["cpu_total"] for r in RULES)
        f11 = js(f"{B}checks/e51_check_11_cost_from_check9_{seed}.json")["forecast_cpu_hours"]
        c10 = js(f"{B}checks/e51_check_10_probe_{seed}.json")["result"]
        h8 = js(f"{B}checks/e51_check_8_b1_is_addendum_{seed}.json")["result"]; h9 = js(f"{B}checks/e51_check_9_b3_is_e43_{seed}.json")["result"]
        sec_has = all(x in sec for x in [h8[r]["tensors_sha256_4096_here"][:8] for r in RULES] + [h9[r]["tensors_sha256_4096_here"][:8] for r in RULES]
                      + [f"{c10[r]['probe_exact']}/400 ({r})" for r in RULES] + [f"{f11:.2f} CPU-hours"])
        print(f"    {seed}: B1 and B2 runs {runs / 3600:.3f} CPU-hours (report 1.47); checks 8 and 9 {chk / 3600:.3f} (report 0.61); forecast {f11:.4f}; "
              f"check 10 {c10['S']['probe_exact']}/{c10['B']['probe_exact']}; the hashes, counts and forecast of 9.1 are those of the receipts: {note(sec_has, f'section 9.1 {seed}')}")
        note(round(runs / 3600, 2) == 1.47 and round(chk / 3600, 2) == 0.61, f"section 9.1 CPU {seed}")
    clock(2, t0)


# ================================================================ part 3: the predictions
def part3():
    t0 = time.time(); print("===== PART 3. PB.1 to PB.4 on each seed, by the brief's fixed rules; the three-seed statements =====")
    if "_X" not in globals():
        R, X = {}, {}
        for seed in ALL:
            R[seed], X[seed] = readings(seed)
    X = globals().get("_X") or X
    for seed in ALL:
        b1 = {r: X[seed][("B1", r)][0] for r in RULES}; b2 = {r: X[seed][("B2", r)] for r in RULES}
        k4 = {r: b1[r][4]["kept"] for r in RULES}; k32 = {r: b1[r][32]["kept"] for r in RULES}
        print(f"--- seed {seed}")
        print(f"  PB.1 (1/4 keeps, both rules): {'held' if all(k4.values()) else 'failed'}; " + "; ".join(
            f"{r}: row 0 {b1[r][4]['r0']:.4f}, 4,096 {b1[r][4]['r4']:.4f}, difference {b1[r][4]['r4'] - b1[r][4]['r0']:+.4f} ({b1[r][4]['pct']:.3f}% of row 0), kept {k4[r]}" for r in RULES))
        print(f"  PB.2 (1/32 does not keep, either rule): {'held' if not any(k32.values()) else 'failed'}; " + "; ".join(
            f"{r}: difference {b1[r][32]['r4'] - b1[r][32]['r0']:+.3f} ({b1[r][32]['pct']:.1f}%)" for r in RULES))
        print(f"  PB.3 (the review shortens under B): {'held' if b2['B']['shortens'] else 'failed'}; " + "; ".join(
            f"{r}: cycle 1 {b2[r]['cycles'][0]['review']}, cycle 8 {b2[r]['cycles'][7]['review']} (at most {b2[r]['cycles'][0]['review'] / 2:g}): {b2[r]['shortens']}" for r in RULES))
        print(f"  PB.4 (more is left under B): {'held' if b2['B']['more'] else 'failed'}; " + "; ".join(
            f"{r}: cycle 1 {b2[r]['cycles'][0]['gap_ff']:.2f}, cycle 8 {b2[r]['cycles'][7]['gap_ff']:.2f} (difference {b2[r]['cycles'][7]['gap_ff'] - b2[r]['cycles'][0]['gap_ff']:.2f}; 20 needed): {b2[r]['more']}" for r in RULES))
    print("3.1 the report's three-seed statements")
    least = {s: {r: X[s][("B1", r)][1] for r in RULES} for s in ALL}
    print(f"    least share that keeps: {least} (report: 1/2, 1/4, 1/2 by seed, both rules: {note([least[s]['S'] for s in ALL] == [2, 4, 2] and all(least[s]['S'] == least[s]['B'] for s in ALL), 'least share')})")
    more = {(s, r): X[s][("B2", r)]["more"] for s in ALL for r in RULES}; short = {(s, r): X[s][("B2", r)]["shortens"] for s in ALL for r in RULES}
    print(f"    more is left on all six seed-and-rule pairs: {all(more.values())} {note(all(more.values()), 'more on six')}; the smallest difference "
          f"{min(X[s][('B2', r)]['cycles'][7]['gap_ff'] - X[s][('B2', r)]['cycles'][0]['gap_ff'] for s in ALL for r in RULES):.2f} points")
    print(f"    the review shortens: S on {sum(short[(s, 'S')] for s in ALL)} seeds, B on {sum(short[(s, 'B')] for s in ALL)} seed(s) ({[s for s in ALL if short[(s, 'B')]]}): "
          f"{note(sum(short[(s, 'S')] for s in ALL) == 3 and sum(short[(s, 'B')] for s in ALL) == 1, 'shortens statement')}")
    print("3.2 the other readings of 'within 5 points of row 0', at every share, and the least share under each")
    readings_ = {"absolute difference <= 5 (the brief's, the worker's)": lambda r0, r4: abs(r4 - r0) <= 5,
                 "strict: absolute difference < 5": lambda r0, r4: abs(r4 - r0) < 5,
                 "loss only: r0 - r4 <= 5 (a rise counts as kept)": lambda r0, r4: r0 - r4 <= 5,
                 "on the printed figures (one decimal)": lambda r0, r4: abs(round(r4, 1) - round(r0, 1)) <= 5 + 1e-9,
                 "as a percentage of row 0: |100 r4 / r0 - 100| <= 5": lambda r0, r4: abs(100 * r4 / r0 - 100) <= 5,
                 "percentage, loss only: 100 r4 / r0 >= 95": lambda r0, r4: 100 * r4 / r0 >= 95}
    for name, f in readings_.items():
        out = []
        for s in ALL:
            for r in RULES:
                o = X[s][("B1", r)][0]; ls = None
                for k in (2, 4, 8, 16, 32):
                    if f(o[k]["r0"], o[k]["r4"]):
                        ls = k
                    else:
                        break
                out.append(f"{s}{r} 1/4 {'kept' if f(o[4]['r0'], o[4]['r4']) else 'not'}, least {'none' if ls is None else '1/' + str(ls)}")
        print(f"    {name}: " + "; ".join(out))
    o = X[93683][("B1", "B")][0][4]; o2 = X[116089][("B1", "B")][0][2]
    print(f"3.3 93683 B at 1/4: row 0 {o['r0']!r}, 4,096 {o['r4']!r}; difference {o['r4'] - o['r0']:.10f}; percentage {100 * o['r4'] / o['r0']:.6f}%; "
          f"116089 B at 1/2: difference {o2['r4'] - o2['r0']:+.10f}, percentage {100 * o2['r4'] / o2['r0']:.6f}%")
    for s, r in ((93683, "B"), (116089, "B")):
        use(s); rec = M.rec_b1(4, r) if s == 93683 else M.ref_add2(r)
        for g in (0, 4096):
            sc = V.row(rec, g)["score1_first_family_all_pairs_k0"]
            print(f"    {s} {r} {'1/4' if s == 93683 else '1/2'} row {g}: reduction pairs right add {sc['add89']['reduction']['exact']}/{sc['add89']['reduction']['n']}, "
                  f"sub {sc['sub89']['reduction']['exact']}/{sc['sub89']['reduction']['n']}, score {sc['reduction_score']!r}")
    clock(3, t0)


# ================================================================ part 4: the checks of the new seeds; the aborted starts
def part4():
    t0 = time.time(); print("===== PART 4. The checks 8 to 11 of the new seeds; the order of events; the aborted starts =====")
    S1 = V.first_family_items()
    for seed in SEEDS:
        use(seed); print(f"--- seed {seed}")
        c8 = js(f"{B}checks/e51_check_8_b1_is_addendum_{seed}.json")
        for r in RULES:
            mine = js(f"{B}check8/e51_B1_share_k2addendum_{r}_{seed}.json"); ref = js(f"{P}e43/addendum2/e43_unrelated-rehearse_{r}_{seed}.json", DREF)
            a, b = V.row(mine, 64), V.row(ref, 64)
            s12 = a["score1_first_family_all_pairs_k0"] == b["score1_first_family_all_pairs_k0"] and a["score2_first_family_heldout_k12"] == b["score2_first_family_heldout_k12"]
            th = mine["complete"]["tensors_sha256"] == ref["complete"]["tensors_sha256"] == c8["result"][r]["tensors_sha256_4096_here"]
            dd = mine["complete"]["data_digest"] == ref["complete"]["data_digest"]
            allr = all(V.row(mine, g)["score1_first_family_all_pairs_k0"] == V.row(ref, g)["score1_first_family_all_pairs_k0"] for g in E43ROWS)
            print(f"  8 {r}: receipt ok {c8['result'][r]['ok']}; against the addendum's receipt of seed {seed} on the default branch: scores 1 and 2 at row 64 equal {s12}; "
                  f"tensors hash at 4,096 equal {th} ({mine['complete']['tensors_sha256'][:8]}); data digest equal {dd}; score 1 equal at all six rows {allr}: {note(s12 and th and dd, f'check 8 {seed} {r}')}")
        c9 = js(f"{B}checks/e51_check_9_b3_is_e43_{seed}.json")
        for r in RULES:
            mine = M.rec_b3("0.001", r); ref = js(f"{P}e43/e43_unrelated_{r}_{seed}.json", DREF)
            eq = all(V.row(mine, g)["score1_first_family_all_pairs_k0"] == V.row(ref, g)["score1_first_family_all_pairs_k0"]
                     and V.row(mine, g)["score2_first_family_heldout_k12"] == V.row(ref, g)["score2_first_family_heldout_k12"] for g in (64, 4096))
            th = mine["complete"]["tensors_sha256"] == ref["complete"]["tensors_sha256"] == c9["result"][r]["tensors_sha256_4096_here"]
            dd = all(V.row(mine, g)["data_digest"] == V.row(ref, g)["data_digest"] for g in (0, 64, 4096))
            lr = [g["lr"] for g in mine["complete"]["optimizer_groups_read_back"]]
            print(f"  9 {r}: receipt ok {c9['result'][r]['ok']}; against E43's receipt on the default branch: counts equal at rows 64 and 4,096 {eq}; tensors hash equal {th} "
                  f"({mine['complete']['tensors_sha256'][:8]}); data digests equal {dd}; rates read back {lr}: {note(eq and th, f'check 9 {seed} {r}')}")
        c10 = js(f"{B}checks/e51_check_10_probe_{seed}.json")
        rng = Random(f"e51|{seed}|probe"); idx = {n: rng.sample([i for i, f in enumerate(S1[n][1]) if f], 200) for n in ("add89", "sub89")}
        for r in RULES:
            W, sha = V.load(f"{P}e26/e26_dev_{r}_{seed}.pt", REF); net = V.Net(W); st = M.rec_b1(4, r)["settings"]["start"]
            n_ok = full = 0
            for n in ("add89", "sub89"):
                outs = V.generate(net, S1[n][0]); ok = [o == t for o, (_, t) in zip(outs, S1[n][0])]
                n_ok += sum(ok[i] for i in idx[n]); full += sum(o for o, f in zip(ok, S1[n][1]) if f)
            e = V.row(js(f"{P}e43/e43_unrelated_{r}_{seed}.json", DREF), 0)["score1_first_family_all_pairs_k0"]
            e43red = e["add89"]["reduction"]["exact"] + e["sub89"]["reduction"]["exact"]
            p0 = [M.rec_b1(k, r)["probe"][0]["first_family_exact"] for k in KS] + [M.rec_b2(r)["probe"][0]["first_family_exact"], M.rec_b3("0.001", r)["probe"][0]["first_family_exact"]]
            print(f"  10 {r}: birth file hash equal to the receipts' {sha == st['birth_sha256']}, tensors hash {V.tensors_sha(W) == st['birth_tensors_sha256']}; the reader's forward pass answers "
                  f"{n_ok} of the 400 probe items (receipt {c10['result'][r]['probe_exact']}, score-1 generation {c10['result'][r]['same_items_in_score1_exact']}): "
                  f"{note(n_ok == c10['result'][r]['probe_exact'] == c10['result'][r]['same_items_in_score1_exact'], f'check 10 {seed} {r}')}; probe at update 0 in the six runs of the rule (B1 x 4, B2, B3) {p0}; "
                  f"all reduction pairs by the reader {full}, E43's receipt at row 0 {e43red}: {note(full == e43red, f'E43 row 0 {seed} {r}')}")
        c11 = js(f"{B}checks/e51_check_11_cost_from_check9_{seed}.json")
        per = {r: M.rec_b3("0.001", r)["complete"]["cpu_updates"] / 4096 for r in RULES}
        runs = {}
        for r in RULES:
            for k in KS:
                runs[f"B1 k={k} {r}"] = 4096 * per[r] + 6 * 22.0 + 29 * 1.5
            runs[f"B2 {r}"] = 8 * (256 + 1024) * per[r] + 2 * 22.0 + 8 * (4 + 64) * 1.5
        same = per == c11["cpu_seconds_an_update"] and all(abs(runs[k] - c11["forecast_cpu_seconds"][k]) < 1e-6 for k in runs)
        spent = sum(M.rec_b1(k, r)["complete"]["cpu_total"] for k in KS for r in RULES) + sum(M.rec_b2(r)["complete"]["cpu_total"] for r in RULES)
        partA = {"S": 0.0934, "B": 0.0881}
        altf = sum((4096 * partA[r] + 6 * 22.0 + 29 * 1.5) * 4 + 8 * 1280 * partA[r] + 44 + 8 * 68 * 1.5 for r in RULES) / 3600
        print(f"  11: CPU-seconds an update from the seed's check-9 runs {per}; the receipt's forecast recomputed by the runner's formula for the ten runs of B1 and B2: "
              f"{sum(runs.values()) / 3600:.4f} CPU-hours (receipt {c11['forecast_cpu_hours']:.4f}): {note(same, f'check 11 {seed}')}; the same formula with Part A's rates of seed 73129 "
              f"(0.0934, 0.0881): {altf:.4f}; spent by the ten runs {spent / 3600:.4f}")
    # order of events
    print("4.1 the order of events: the log stamps (start of each process, UTC) and the commits recorded in the check receipts")
    import datetime as dt
    logs = git("ls-tree", "-r", "--name-only", REF, "--", B + "logs").split()
    st = sorted((int(l.rsplit(".", 2)[-2]), l.split("/")[-1].rsplit(".", 2)[0]) for l in logs if ("93683" in l or "116089" in l))
    for t, n in st:
        print(f"    {dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime('%H:%M:%S')}  {n}")
    for seed in SEEDS:
        for name in ("10_probe", "8_b1_is_addendum", "9_b3_is_e43"):
            c = js(f"{B}checks/e51_check_{name}_{seed}.json")["provenance"]["git"]["commit"]
            print(f"    check {name.split('_')[0]} of {seed}: written at commit {c[:8]} (committed {git('show', '-s', '--format=%ci', c).strip()})")
        added = git("log", "--diff-filter=A", "--format=%h %ci", REF, "--", f"{B}checks/e51_check_11_cost_from_check9_{seed}.json").strip()
        print(f"    check 11 of {seed}: the receipt carries no time or commit; first committed in {added}")
    # the aborted starts
    print("4.2 the six aborted starts of seed 93683")
    ab = [("e51_B2_intervals_S_93683", 1794971004), ("e51_B2_intervals_B_93683", 1794971004), ("e51_B1_share_4_S_93683", 1794971004),
          ("e51_B1_share_4_B_93683", 1794971020), ("e51_B1_share_8_S_93683", 1794971020), ("e51_B1_share_8_B_93683", 1794971020)]
    tree = git("ls-tree", "-r", "--name-only", REF, "--", B).split()
    for tag, t in ab:
        lg = [l for l in logs if l.split("/")[-1].startswith(tag + ".")]
        first = show(f"{B}logs/{tag}.{t}.log").decode(); second = [l for l in lg if not l.endswith(f".{t}.log")]
        sec_txt = show(second[0]).decode() if len(second) == 1 else ""
        rec = js(f"{B}{tag}.json")
        files = [f for f in tree if f.split("/")[-1].startswith(tag) and not f.endswith(".log")]
        # the second log against the receipt: every probe line and the final cpu
        pl = re.findall(rf"{tag} probe (\d+) first family ([\d.]+)", sec_txt)
        pr = {p["updates"]: p["first_family_pct"] for p in rec["probe"]} if "cycles" not in rec else None
        if pr is not None:
            log_ok = len(pl) == len(pr) and all(abs(float(f) - pr[int(g)]) < 1e-9 for g, f in pl)
            cpu = re.findall(r"cpu (\d+)s", sec_txt)
            log_ok = log_ok and bool(cpu) and abs(int(cpu[-1]) - rec["complete"]["cpu_total"]) < 1
        else:
            log_ok = "complete" in sec_txt and all(str(c["review_updates"]) in sec_txt for c in rec["cycles"])
        print(f"    {tag}: aborted log {len(first.splitlines())} line(s) ({first.strip().splitlines()[-1]!r}); second log {second[0].split('/')[-1] if second else None} "
              f"({len(sec_txt.splitlines())} lines); files of the run besides logs: {[f.split('/')[-1] for f in files]}; the receipt's figures are the second log's: {note(bool(log_ok), 'aborted ' + tag)}")
    hist = git("log", "--format=%h", f"{OLD}..{REF}", "--", *[f"{B}{tag}.json" for tag, _ in ab]).split()
    print(f"    commits that touch the six receipts: {hist} (the first after the second starts at 03:11:44: "
          f"{git('show', '-s', '--format=%ci', hist[-1]).strip() if hist else None})")
    for tag, _ in ab:
        first_c = git("log", "--diff-filter=A", "--format=%h", REF, "--", f"{B}{tag}.json").strip()
        r0 = js(f"{B}{tag}.json", first_c); rf = js(f"{B}{tag}.json")
        pre = rf["probe"][:len(r0["probe"])] == r0["probe"]
        print(f"    {tag}: first committed at {first_c} with {len(r0['probe'])} probes, a prefix of the final receipt's: {note(pre, 'partial receipt ' + tag)}")
    clock(4, t0)


# ================================================================ part 5: the weights
def part5():
    t0 = time.time(); print("===== PART 5. The weights =====")
    files = [f for f in git("ls-tree", "-r", "--name-only", REF, "--", B).split() if f.endswith(".pt") and ("93683" in f or "116089" in f)]
    ok = 0
    for f in files:
        jf = f.rsplit("_row", 1)[0] + ".json" if "_row" in f else f.rsplit("_end", 1)[0] + ".json"
        got = hashlib.sha256(show(f)).hexdigest(); ok += got == js(jf)["complete"]["weights_sha256"]
        if got != js(jf)["complete"]["weights_sha256"]:
            diffs.append("weights sha " + f)
    print(f"5.1 the SHA-256 of the {len(files)} weight files of the new seeds against their receipts: {ok} equal")
    S1 = V.first_family_items()
    for seed, k, rule in ((93683, 4, "B"), (116089, 4, "S")):
        use(seed); HELD = V.held12(); pan = {"held": V.u_panel("held"), "taught": V.u_panel("taught")}
        rng = Random(f"e51|{seed}|probe"); idx = {n: rng.sample([i for i, f in enumerate(S1[n][1]) if f], 200) for n in ("add89", "sub89")}
        rec = M.rec_b1(k, rule); t1 = time.time()
        W, sha = V.load(f"{B}e51_B1_share_{k}_{rule}_{seed}_row4096.pt", REF); net = V.Net(W); x = V.row(rec, 4096)
        r, gens = V.score_first_family(net, S1, HELD); a = V.score_new(net, pan)
        mine_probe = sum(gens["k0_" + n][i] == S1[n][0][i][1] for n in idx for i in idx[n]); lastp = rec["probe"][-1]
        same = V.same_counts(r, V.receipt_first_family(x))
        print(f"5.2 B1 k = {k}, seed {seed}, rule {rule}, row 4,096: file hash equal {sha == rec['complete']['weights_sha256']}; tensors hash equal {V.tensors_sha(W) == x['tensors_sha256']}")
        print(f"     first family (reader): {V.fmt_ff(r)}; equal to the receipt: {note(same, f'weights ff {seed}')}")
        print(f"     receipt:               {V.fmt_ff(V.receipt_first_family(x))}")
        print(f"     new task: {V.fmt_new(a)} (held-out {100 * a['held']['exact'] / 300:.1f}%); equal to the receipt's score 3: {note(a == x['score3_new_task'], f'weights new {seed}')}; "
              f"last probe (row {lastp['updates']}): {mine_probe} of 400 (receipt {lastp['first_family_exact']}), held {a['held']['exact']} (receipt {lastp['unrelated_held_exact']}): "
              f"{note(mine_probe == lastp['first_family_exact'] and a['held']['exact'] == lastp['unrelated_held_exact'], f'weights probe {seed}')} [{time.time() - t1:.0f} s]")
    clock(5, t0)


PARTS = {"1": part1, "2": part2, "3": part3, "4": part4, "5": part5}

if __name__ == "__main__":
    for p in WANT:
        PARTS[p]()
    print("Figures that differ:", diffs if diffs else "none")
