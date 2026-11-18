"""Verification of the trial of E51, Part B (how much rehearsal keeps the first family; seed 73129; the worker's branch
claude/e51-reserve-017dSEpW at 608f6b4c).

Read-only.  Nothing is trained, torch is not used, and no code of another branch is run: every receipt and weight file is
read with `git show` into memory, and nothing is written.  Items, streams, the digest of the batches, the forward pass and
the generation of answers are imported from the verification of Part A (`e51_trial_verify.py` beside this file), which wrote
them from the brief's text; that script's reader of weight files (`compare_pt_tensors.tensors`) is used through it.

  1  the receipts: that each of the eighteen run receipts of Part B (sixteen runs and the two runs of check 8) is complete and
     records the settings its name says; the data digest of every scored row recomputed from the reader's own streams (this
     tests the share k, the gap and review order of B2 and the stream of B3); every cell of E51_TABLES_B.md, every value of
     E51_READ_B.json and every figure of sections 5 and 6 of E51_REPORT.md recomputed and compared
  2  the five predictions PB.1 to PB.5 scored by the brief's fixed rules, and the figures behind the report's wording
  3  the checks 8 to 11: what each receipt records, compared with the receipts it names (read here directly), with the
     reader's own probe items, and with the probe at update 0 from E26's births by the reader's forward pass
  4  the weights: the SHA-256 of every weight file of Part B against its receipt; from the final weights of two runs (B1 at
     k = 8 under rule B, B3 at rate 0.0003 under rule S) the first family's counts, the new task's panels and the probe
     recomputed with the reader's forward pass, and the hash of the tensors

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e51_partb_verify.py [1 2 3 4]
"""
import hashlib, json, os, re, subprocess, sys, time
from random import Random

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
want = sys.argv[1:] or list("1234")
sys.argv = sys.argv[:1]                       # the imported script reads its own argument list at import
import numpy as np
import e51_trial_verify as V                  # noqa: E402  (its parts run only under __main__)

WREF = "608f6b4c"                             # origin/claude/e51-reserve-017dSEpW, the commit under verification
DREF = V.DREF                                 # the default branch at 7cabb13a (E43's receipts and the second addendum's, as published)
P = "MOD97_LOCAL_RULE_20261106/exp45/"
B = P + "e51/partB/"
SEED = V.SEED
RULES = ("S", "B")
KS = (4, 8, 16, 32)
RATES = {"0.001": (0.001, 4096), "0.0003": (0.0003, 13824), "0.0001": (0.0001, 40960)}
E43ROWS = [0, 64, 256, 1024, 2048, 4096]
diffs = []                                    # every figure that differs, for the summary


def show(path, ref=WREF):
    return V.show(path, ref)


def js(path, ref=WREF):
    return json.loads(show(path, ref))


def note(ok, what):
    if not ok:
        diffs.append(what)
    return "agrees" if ok else "DIFFERS"


def clock(label, t0):
    print(f"[part {label}: {time.time() - t0:.0f} s]\n", flush=True)


def held_pct(r):
    return 100 * r["score3_new_task"]["held"]["exact"] / r["score3_new_task"]["held"]["n"]


def red(r):
    return r["score1_first_family_all_pairs_k0"]["reduction_score"]


def rec_b1(k, rule):
    return js(f"{B}e51_B1_share_{k}_{rule}_{SEED}.json")


def rec_b2(rule):
    return js(f"{B}e51_B2_intervals_{rule}_{SEED}.json")


def rec_b3(key, rule):
    return js(f"{B}e51_B3_steps_{key}_{rule}_{SEED}.json")


def ref_e43(rule):
    return js(f"{P}e43/e43_unrelated_{rule}_{SEED}.json")


def ref_add2(rule):
    return js(f"{P}e43/addendum2/e43_unrelated-rehearse_{rule}_{SEED}.json")


# ---------------------------------------------------------------- the readings, from the brief's text
def read_b1(rule):
    """For each share: reduction score at rows 0 and 4,096, as % of row 0, kept (within 5 points), lowest probe, the first probe row
    with the unrelated held-out score at 80% (and, for comparison, the first such row after which it stays at 80%), held at 4,096."""
    out = {}
    srcs = [(2, ref_add2(rule))] + [(k, rec_b1(k, rule)) for k in KS] + [(0, ref_e43(rule))]
    for k, rec in srcs:
        a, z = V.row(rec, 0), V.row(rec, 4096); pr = rec.get("probe") if k in KS else None
        r0, r4 = red(a), red(z)
        d = dict(r0=r0, r4=r4, pct=100 * r4 / r0, drop=r0 - r4, kept=abs(r4 - r0) <= 5, kept_pct_reading=abs(100 * r4 / r0 - 100) <= 5,
                 low=min(p["first_family_pct"] for p in pr) if pr else None,
                 f80=next((p["updates"] for p in pr if p["unrelated_held_pct"] >= 80), None) if pr else None,
                 f80_stays=V.first_learned([(p["updates"], p["unrelated_held_pct"]) for p in pr]) if pr else None,
                 held4096=held_pct(z), held_rows={g: held_pct(V.row(rec, g)) for g in E43ROWS},
                 red_rows={g: red(V.row(rec, g)) for g in E43ROWS})
        out[k] = d
    least = None
    for k in (2, 4, 8, 16, 32):
        if out[k]["kept"]:
            least = k
        else:
            break
    return out, least


def read_b2(rule):
    rec = rec_b2(rule); cy = rec["cycles"]; tr = rec["probe"]
    p0 = tr[0]["first_family_pct"]; goal = 0.9 * p0
    # each cycle again from the trace of probes: the probe at the end of the gap, the review's length, the probe at its end
    mine = []
    for c in range(1, 9):
        gp = [p for p in tr if p.get("cycle") == c and p["phase"] == "gap"]; rv = [p for p in tr if p.get("cycle") == c and p["phase"] == "review"]
        end_gap = gp[-1]; n = rv[-1]["updates_in_phase"] if rv else 0; end_rev = rv[-1] if rv else end_gap
        first_reach = next((p["updates_in_phase"] for p in rv if p["first_family_pct"] >= goal), None)
        mine.append(dict(cycle=c, gap_ff=end_gap["first_family_pct"], gap_un=end_gap["unrelated_held_pct"], gap_at=end_gap["updates"], review=n,
                         reached=end_rev["first_family_pct"] >= goal, first_reach=first_reach, rev_ff=end_rev["first_family_pct"], rev_un=end_rev["unrelated_held_pct"],
                         end=end_rev["updates"], gap_probes_ok=[p["updates_in_phase"] for p in gp] == [64, 128, 192, 256]))
    shortens = mine[7]["review"] <= mine[0]["review"] / 2
    more = mine[7]["gap_ff"] - mine[0]["gap_ff"] >= 20
    return rec, dict(p0=p0, goal=goal, cycles=mine, shortens=shortens, more=more, red0=red(rec["rows"][0]), red_end=red(rec["rows"][-1]),
                     held_end=held_pct(rec["rows"][-1]), held_end_n=rec["rows"][-1]["score3_new_task"]["held"]["exact"], end=rec["complete"]["updates"])


def read_b3(rule):
    out = {}
    for key in RATES:
        rec = rec_b3(key, rule); pr = rec["probe"]; p0 = pr[0]["first_family_pct"]
        f80 = next((p for p in pr if p["unrelated_held_pct"] >= 80), None)
        half = [p for p in pr if p["first_family_pct"] >= p0 / 2]
        # the second reading of "the last probe row at which the share is still at least half": the row before the first drop below half
        before = None
        for p in pr:
            if p["first_family_pct"] < p0 / 2:
                break
            before = p
        z = rec["rows"][-1]
        out[key] = dict(updates=rec["complete"]["updates"], p0=p0, f80=f80["updates"] if f80 else None, ff_at80=f80["first_family_pct"] if f80 else None,
                        last_half=half[-1]["updates"] if half else None, held_there=half[-1]["unrelated_held_pct"] if half else None,
                        before_drop=before["updates"] if before else None, held_before_drop=before["unrelated_held_pct"] if before else None,
                        red_end=red(z), held_end=held_pct(z), probe64=next(p["first_family_pct"] for p in pr if p["updates"] == 64),
                        curve=[(p["updates"], p["first_family_pct"], p["unrelated_held_pct"]) for p in pr])
    more = any(out[k]["ff_at80"] is not None and out["0.001"]["ff_at80"] is not None and out[k]["ff_at80"] - out["0.001"]["ff_at80"] >= 20 for k in ("0.0003", "0.0001"))
    return out, more


# ================================================================ part 1: the receipts
def b1_source(k):
    """The items of B1 in order, as the brief gives them: the first family's taught half when the choice says so, else the unrelated stream."""
    unrel = V.u_stream(); fam = V.family_stream()
    if k == "k2addendum":
        rj = Random(f"e43|{SEED}|joint")
        return lambda: ("unrelated", next(unrel)) if rj.randrange(2) == 0 else ("first_family", next(fam))
    rj = Random(f"e51|{SEED}|joint|{k}")
    return lambda: ("first_family", next(fam)) if rj.randrange(k) == 0 else ("unrelated", next(unrel))


def digests_of(nxt, rows, dose):
    dg = V.Digest(); out = {0: dg.value()}; counts = {"unrelated": 0, "first_family": 0}
    for g in range(1, dose + 1):
        batch = []
        for _ in range(64):
            kind, it = nxt(); counts[kind] += 1; batch.append(it)
        dg.add(batch)
        if g in rows:
            out[g] = dg.value()
    return out, counts


def check_receipt(rec, tag, want_settings, rows, dose, wfile):
    s, c = rec["settings"], rec["complete"]
    probs = []
    if rec["tag"] != tag:
        probs.append("tag")
    if not c or c["updates"] != dose:
        probs.append("not complete")
    for k, v in want_settings.items():
        if s.get(k) != v:
            probs.append(f"settings {k}={s.get(k)!r}, expected {v!r}")
    if s["seed"] != SEED or s["decay"] != 0.0 or s["batch"] != 64:
        probs.append("seed/decay/batch")
    groups = s["optimizer_groups_read_back"] + c["optimizer_groups_read_back"]
    if any(g["weight_decay"] != 0.0 for g in groups):
        probs.append("weight decay")
    if s["local"] != (s["rule"] == "S"):
        probs.append("rule/local")
    if s["start"]["birth"] != f"exp45/e26/e26_dev_{s['rule']}_{SEED}.pt":
        probs.append("birth")
    if [r["updates"] for r in rec["rows"]] != rows:
        probs.append(f"rows {[r['updates'] for r in rec['rows']]}")
    if c.get("weights") != wfile:
        probs.append("weights file")
    if c.get("optimizer_steps") != [dose]:
        probs.append(f"optimizer steps {c.get('optimizer_steps')}")
    if s["provenance"]["git"]["sources_modified_vs_commit"] or s["provenance"]["git"]["e51_sources_modified_vs_commit"]:
        probs.append("sources modified")
    return probs


def part1():
    t0 = time.time(); print("===== PART 1. The receipts of Part B =====")
    runner_sha = hashlib.sha256(show("MOD97_LOCAL_RULE_20261106/exp51_partb.py")).hexdigest()
    print(f"1.0 SHA-256 of exp51_partb.py at {WREF}: {runner_sha}")
    # E43's and the second addendum's receipts on the worker's branch against the default branch
    for path in [f"{P}e43/e43_unrelated_{r}_{SEED}.json" for r in RULES] + [f"{P}e43/addendum2/e43_unrelated-rehearse_{r}_{SEED}.json" for r in RULES]:
        a, b = hashlib.sha256(show(path)).hexdigest(), hashlib.sha256(show(path, DREF)).hexdigest()
        print(f"1.0 {path.split('/')[-1]}: the worker's copy is the default branch's: {note(a == b, 'reference receipt ' + path)}")

    print("1.1 completeness, settings and the data digests recomputed from the reader's own streams")
    shas = {}
    for k in list(KS) + ["k2addendum"]:
        dig, counts = digests_of(b1_source(k), set(E43ROWS), 4096)
        for rule in RULES:
            path = f"{B}{'check8/' if k == 'k2addendum' else ''}e51_B1_share_{k}_{rule}_{SEED}.json"; rec = js(path)
            shas[path] = rec["settings"]["provenance"]["e51_sources_sha256"]["exp51_partb.py"]
            pb = check_receipt(rec, f"e51_B1_share_{k}_{rule}_{SEED}", dict(set="B1", k=str(k), rule=rule, updates=4096), E43ROWS, 4096, f"e51_B1_share_{k}_{rule}_{SEED}_row4096.pt")
            lr = {g["lr"] for g in rec["settings"]["optimizer_groups_read_back"]}
            prow = [p["updates"] for p in rec["probe"]] == sorted(set(range(0, 1025, 64)) | set(range(1024, 4097, 256)))
            dok = all(V.row(rec, g)["data_digest"] == dig[g] for g in E43ROWS) and rec["complete"]["data_digest"] == dig[4096]
            cok = rec["complete"]["items"] == counts
            ff = counts["first_family"] / (counts["first_family"] + counts["unrelated"])
            print(f"    B1 {k:>10} {rule}: problems {pb or 'none'}; lr {lr}; probe rows as the brief: {prow}; digests at the six rows equal to the reader's: {note(dok, f'digest B1 {k} {rule}')}; "
                  f"items {counts} equal: {note(cok, f'items B1 {k} {rule}')}; share of first-family items {ff:.4f} (1/{1 / ff:.2f})")
            if pb or not prow or lr != {0.001}:
                diffs.append(f"settings B1 {k} {rule}: {pb}")
    # B2: the gap and the review batches in the order the receipt's cycles give
    for rule in RULES:
        rec = rec_b2(rule); unrel = V.u_stream(); fam = V.family_stream(); dg = V.Digest(); g = 0
        for c in rec["cycles"]:
            for _ in range(256):
                dg.add([next(unrel) for _ in range(64)]); g += 1
            for _ in range(c["review_updates"]):
                dg.add([next(fam) for _ in range(64)]); g += 1
        s = rec["settings"]; pb = []
        for kk, v in dict(set="B2", rule=rule, gap=256, review_max=1024, cycles=8, review_probe_every=16, decay=0.0, seed=SEED, batch=64).items():
            if s.get(kk) != v:
                pb.append(kk)
        if s["start"]["birth"] != f"exp45/e26/e26_dev_{rule}_{SEED}.pt" or rec["complete"]["optimizer_steps"] != [rec["complete"]["updates"]]:
            pb.append("birth or one optimizer")
        if any(gg["weight_decay"] != 0.0 or gg["lr"] != 0.001 for gg in s["optimizer_groups_read_back"] + rec["complete"]["optimizer_groups_read_back"]):
            pb.append("lr/decay")
        dok = rec["complete"]["data_digest"] == dg.value() and rec["rows"][-1]["data_digest"] == dg.value()
        print(f"    B2 {rule}: problems {pb or 'none'}; {len(rec['cycles'])} cycles; updates {rec['complete']['updates']} (= 8 x 256 + reviews {sum(c['review_updates'] for c in rec['cycles'])}: "
              f"{8 * 256 + sum(c['review_updates'] for c in rec['cycles'])}); E43's rows at {[r['updates'] for r in rec['rows']]}; digest of all {g} batches equal to the reader's: {note(dok, f'digest B2 {rule}')}")
        if pb:
            diffs.append(f"settings B2 {rule}: {pb}")
    # B3: the unrelated stream alone, one pass to 40,960 updates
    rows_all = {0, 64, 4096, 13824, 40960}
    st = V.u_stream()
    dig, _ = digests_of(lambda: ("unrelated", next(st)), rows_all, 40960)
    for key, (lr, dose) in RATES.items():
        for rule in RULES:
            rec = rec_b3(key, rule)
            want_rows = sorted({0, 64, 4096, dose})
            pb = check_receipt(rec, f"e51_B3_steps_{key}_{rule}_{SEED}", dict(set="B3", learning_rate=lr, rule=rule, updates=dose), want_rows, dose, f"e51_B3_steps_{key}_{rule}_{SEED}_row{dose}.pt")
            lrs = [g["lr"] for g in rec["settings"]["optimizer_groups_read_back"] + rec["complete"]["optimizer_groups_read_back"]]
            if any(x != lr for x in lrs):
                pb.append(f"lr read back {lrs}")
            prow = [p["updates"] for p in rec["probe"]] == sorted(set(range(0, 257, 16)) | set(range(256, dose + 1, 64)))
            dok = all(V.row(rec, g)["data_digest"] == dig[g] for g in want_rows)
            print(f"    B3 {key:>6} {rule}: problems {pb or 'none'}; rates read back {sorted(set(lrs))}; probe rows as the brief: {prow}; digests equal to the reader's: {note(dok, f'digest B3 {key} {rule}')}")
            if pb or not prow:
                diffs.append(f"settings B3 {key} {rule}: {pb}")
            shas[key + rule] = rec["settings"]["provenance"]["e51_sources_sha256"]["exp51_partb.py"]
    for rule in RULES:
        shas["B2" + rule] = rec_b2(rule)["settings"]["provenance"]["e51_sources_sha256"]["exp51_partb.py"]
    print(f"1.1 the runner's hash recorded in the receipts: {sorted(set(shas.values()))}; equal to the file at {WREF}: {set(shas.values()) == {runner_sha}}")
    # probe items, the same in every receipt
    S1 = V.first_family_items(); rng = Random(f"e51|{SEED}|probe")
    mine = {n: rng.sample([i for i, f in enumerate(S1[n][1]) if f], 200) for n in ("add89", "sub89")}
    allp = [rec_b1(k, r)["settings"]["probe_items_first_family"] for k in KS for r in RULES] + [rec_b2(r)["settings"]["probe_items_first_family"] for r in RULES] \
        + [rec_b3(k, r)["settings"]["probe_items_first_family"] for k in RATES for r in RULES]
    print(f"1.2 the probe's 400 items drawn by the reader (Random('e51|{SEED}|probe').sample, add89 then sub89, from the reduction pairs in order) equal to the indices in all 16 receipts: "
          f"{note(all(p == mine for p in allp), 'probe items')}")

    # ---- the tables and the readings file
    tables = show(f"{B}E51_TABLES_B.md").decode(); readj = js(f"{B}E51_READ_B.json"); report = show(f"{P}e51/E51_REPORT.md").decode()
    my_md = []; my_read = {"B1": {}, "B2": {}, "B3": {}}
    names = {2: "1/2 (the second addendum)", 0: "0 (E43 `unrelated`)"}
    b1 = {}
    for rule in RULES:
        o, least = read_b1(rule); b1[rule] = (o, least)
        for k in (2, 4, 8, 16, 32, 0):
            d = o[k]; nm = names.get(k, f"1/{k}")
            my_md.append(f"| {rule} | {nm} | {d['r0']:.1f} | {d['r4']:.1f} | {d['pct']:.1f} | {'yes' if d['kept'] else 'no'} | {'-' if d['low'] is None else format(d['low'], '.1f')} | "
                         f"{'-' if d['f80'] is None else d['f80']} | {d['held4096']:.1f} |")
            my_read["B1"][f"{rule} {nm}"] = dict(reduction_row0=d["r0"], reduction_4096=d["r4"], pct_of_row0=d["pct"], kept=d["kept"], lowest_probe=d["low"],
                                                 unrelated_80_first_probe_row=d["f80"], unrelated_held_4096=d["held4096"])
        my_read["B1"][f"{rule} least share that keeps"] = f"1/{least}" if least else None
    b2 = {}
    for rule in RULES:
        rec, d = read_b2(rule); b2[rule] = d
        for c in d["cycles"]:
            my_md.append(f"| {c['cycle']} | {c['gap_ff']:.2f} | {c['gap_un']:.1f} | {c['review']} | {'yes' if c['reached'] else 'no'} | {c['rev_ff']:.2f} | {c['rev_un']:.1f} |")
        my_read["B2"][rule] = dict(review_shortens=d["shortens"], more_is_left=d["more"], probe_first_family_pct_at_update_0=d["p0"], review_goal_pct=d["goal"],
                                   reduction_row0=d["red0"], reduction_end=d["red_end"], unrelated_held_end=d["held_end"], updates=d["end"],
                                   cycles=[dict(cycle=c["cycle"], updates_at_end_of_gap=c["gap_at"], probe_at_end_of_gap=dict(first_family_pct=c["gap_ff"], unrelated_held_pct=c["gap_un"]),
                                                review_updates=c["review"], review_reached_goal=c["reached"], probe_at_end_of_review=dict(first_family_pct=c["rev_ff"], unrelated_held_pct=c["rev_un"]),
                                                updates_at_end_of_review=c["end"]) for c in d["cycles"]])
        # the receipt's own cycle records against the trace of probes
        same = all(rc["review_updates"] == c["review"] and rc["probe_at_end_of_gap"]["first_family_pct"] == c["gap_ff"] and rc["probe_at_end_of_review"]["first_family_pct"] == c["rev_ff"]
                   and rc["review_reached_goal"] == c["reached"] and c["first_reach"] == (c["review"] or None) and c["gap_probes_ok"] for rc, c in zip(rec["cycles"], d["cycles"]))
        print(f"1.3 B2 {rule}: the receipt's eight cycle records equal the trace of probes, every review stopped at the first probe at or above the goal, four probes in every gap: "
              f"{note(same, f'B2 cycles vs trace {rule}')}")
    b3 = {}
    for rule in RULES:
        o, more = read_b3(rule); b3[rule] = (o, more)
        for key, d in o.items():
            f = lambda q, s="{:.1f}": "-" if q is None else s.format(q)
            my_md.append(f"| {rule} | {key} | {d['updates']} | {f(d['f80'], '{}')} | {f(d['ff_at80'], '{:.2f}')} | {f(d['last_half'], '{}')} | {f(d['held_there'])} | {d['red_end']:.1f} | {d['held_end']:.1f} |")
            my_read["B3"][f"{rule} {key}"] = dict(updates=d["updates"], probe_row0=d["p0"], unrelated_80_first_probe_row=d["f80"], first_family_probe_there=d["ff_at80"],
                                                  last_probe_row_first_family_at_half=d["last_half"], unrelated_held_there=d["held_there"], reduction_end=d["red_end"], unrelated_held_end=d["held_end"])
        my_read["B3"][f"{rule} smaller steps leave more"] = more
    # compare the table cells (the worker's rows without the last column 'source' for B1)
    theirs = []
    for line in tables.splitlines():
        if line.startswith("| S |") or line.startswith("| B |") or re.match(r"\| \d \|", line):
            if "**" in line:
                continue
            cells = [x.strip() for x in line.strip("|").split("|")]
            if len(cells) == 10:
                cells = cells[:9]
            theirs.append([c.replace("the second addendum", "the second addendum") for c in cells])
    mine_cells = [[x.strip() for x in l.strip("|").split("|")] for l in my_md]
    nc = sum(len(r) for r in mine_cells); bad = [(a, b) for a, b in zip(theirs, mine_cells) if a != b]
    print(f"1.4 E51_TABLES_B.md: {len(theirs)} rows of figures ({nc} cells) against the reader's {len(mine_cells)} rows: {note(not bad and len(theirs) == len(mine_cells), 'tables')}")
    for a, b in bad:
        print("    worker:", a, "\n    reader:", b)
    for rule in RULES:
        lst = b1[rule][1]; m = re.search(rf"\| {rule} \| \*\*the least share that keeps\*\* \| \| \| \| (\S+) \|", tables)
        print(f"    least share that keeps, {rule}: worker {m.group(1)}, reader 1/{lst}: {note(m.group(1) == f'1/{lst}', 'least share ' + rule)}")
        m = re.search(rf"\| {rule} \| \*\*smaller steps leave more\*\* \| (\S+) \|", tables)
        print(f"    smaller steps leave more, {rule}: worker {m.group(1)}, reader {'yes' if b3[rule][1] else 'no'}: {note(m.group(1) == ('yes' if b3[rule][1] else 'no'), 'B3 reading ' + rule)}")
        m = re.search(rf"Rule {rule} \(e51_B2.*?probe at update 0 ([\d.]+)%, the review's goal ([\d.]+)%; E43's reduction score ([\d.]+) at update 0 and ([\d.]+) at the end \(update (\d+)\); "
                      rf"the unrelated task's held-out score at the end ([\d.]+)%", tables)
        d = b2[rule]; vals = (f"{d['p0']:.2f}", f"{d['goal']:.2f}", f"{d['red0']:.1f}", f"{d['red_end']:.1f}", str(d['end']), f"{d['held_end']:.1f}")
        print(f"    B2 {rule} header line: worker {m.groups()}, reader {vals}: {note(m.groups() == vals, 'B2 header ' + rule)}")
        m = re.search(rf"The review shortens: (\w+)\. More is left: (\w+)\. Rule {rule}: (.*?)\.\n", tables)
        sent = f"with rehearsal at intervals the first family {'becomes' if d['shortens'] else 'does not become'} cheaper to restore, and {'more' if d['more'] else 'no more'} of it survives a gap"
        print(f"    B2 {rule} readings: worker {m.group(1)}, {m.group(2)}; reader {d['shortens']}, {d['more']}; sentence equal: {note(m.group(3) == sent and (m.group(1) == 'yes') == d['shortens'] and (m.group(2) == 'yes') == d['more'], 'B2 sentence ' + rule)}")

    # the readings file, value by value
    nval, bad = 0, []

    def cmp(a, b, path):
        nonlocal nval
        if isinstance(a, dict):
            for kk in set(a) | set(b):
                if kk == "sentence":
                    continue
                if kk not in a or kk not in b:
                    if kk not in ("sentence",):
                        bad.append((path + "/" + kk, "missing"))
                    continue
                cmp(a[kk], b[kk], path + "/" + kk)
        elif isinstance(a, list):
            if len(a) != len(b):
                bad.append((path, "length"))
            for i, (x, y) in enumerate(zip(a, b)):
                cmp(x, y, f"{path}[{i}]")
        else:
            nval += 1
            if isinstance(a, float) or isinstance(b, float):
                if a is None or b is None or abs(a - b) > 1e-9:
                    bad.append((path, a, b))
            elif a != b:
                bad.append((path, a, b))
    theirs_r = {k: readj[k] for k in ("B1", "B2", "B3")}
    for rule in RULES:
        theirs_r["B2"][rule].pop("sentence", None)
    cmp(theirs_r, my_read, "")
    print(f"1.5 E51_READ_B.json: {nval} values compared with the reader's: {note(not bad, 'readings file')}" + (f"; differences {bad[:10]}" if bad else ""))

    # ---- the report's own figures (sections 5 and 6)
    print("1.6 the report's tables of sections 6 (B1, B2, B3) cell by cell")
    sec = report[report.index("## 6. Part B: results"):report.index("## 7.")]
    b1t = [[x.strip().replace("\u2013", "-") for x in l.strip("|").split("|")] for l in sec[:sec.index("### B2")].splitlines() if l.startswith("| S |") or l.startswith("| B |")]
    mine_b1 = [[x.strip() for x in l.strip("|").split("|")] for l in my_md[:12]]
    for r in mine_b1:
        r[1] = r[1].replace("1/2 (the second addendum)", "1/2 (second addendum)")
    badb1 = [(a, b) for a, b in zip(b1t, mine_b1) if a != b]
    print(f"    B1: {len(b1t)} rows: {note(not badb1 and len(b1t) == 12, 'report B1 table')}")
    for a, b in badb1:
        print("     report:", a, "\n     reader:", b)
    b2t = [[x.strip() for x in l.strip("|").split("|")] for l in sec[sec.index("### B2"):sec.index("### B3")].splitlines() if re.match(r"\| \d \|", l)]
    mine_b2 = [[str(c), f"{b2['S']['cycles'][c - 1]['gap_ff']:.2f}", str(b2['S']['cycles'][c - 1]['review']), f"{b2['B']['cycles'][c - 1]['gap_ff']:.2f}", str(b2['B']['cycles'][c - 1]['review'])] for c in range(1, 9)]
    print(f"    B2: {len(b2t)} rows: {note(b2t == mine_b2, 'report B2 table')}")
    b3t = [[x.strip() for x in l.strip("|").split("|")] for l in sec[sec.index("### B3"):sec.index("### The predictions")].splitlines() if l.startswith("| S |") or l.startswith("| B |")]
    mine_b3 = []
    for rule in RULES:
        for key, d in b3[rule][0].items():
            f = lambda q, s="{:.1f}": "-" if q is None else s.format(q)
            mine_b3.append([rule, key, f"{d['updates']:,}", f(d['f80'], '{:,}'), f(d['ff_at80'], '{:.2f}'), f(d['last_half'], '{}'), f(d['held_there']), f"{d['red_end']:.1f}", f"{d['held_end']:.1f}"])
    badb3 = [(a, b) for a, b in zip(b3t, mine_b3) if a != b]
    print(f"    B3: {len(b3t)} rows: {note(not badb3 and len(b3t) == 6, 'report B3 table')}")
    for a, b in badb3:
        print("     report:", a, "\n     reader:", b)

    print("1.7 the report's figures in its prose (section 6), each against the reader's")
    S, Bb = b2["S"], b2["B"]
    claims = [
        ("B1: a share of 1/4 leaves the reduction score 7.6 points below row 0 (S)", round(b1['S'][0][4]['drop'], 1) == 7.6,
         f"{b1['S'][0][4]['drop']:.3f} (96.565 - 89.019; 7.6 is the difference of the rounded figures 96.6 and 89.0)"),
        ("B1: a share of 1/4 leaves the reduction score 9.2 points below row 0 (B)", round(b1['B'][0][4]['drop'], 1) == 9.2, f"{b1['B'][0][4]['drop']:.3f}"),
        ("B1: what is left falls steadily as the share falls (2, 4, 8, 16, 32, 0)", all(all(b1[r][0][a]['r4'] > b1[r][0][b]['r4'] for a, b in zip((2, 4, 8, 16, 32), (4, 8, 16, 32, 0))) for r in RULES),
         "; ".join(f"{r}: " + ", ".join(f"{b1[r][0][k]['r4']:.1f}" for k in (2, 4, 8, 16, 32, 0)) for r in RULES)),
        ("B2: probe at update 0 and goal, S 95.25% / 85.73%", (f"{S['p0']:.2f}", f"{S['goal']:.2f}") == ("95.25", "85.73"), f"{S['p0']:.4f} / {S['goal']:.4f}"),
        ("B2: probe at update 0 and goal, B 98.25% / 88.43%", (f"{Bb['p0']:.2f}", f"{Bb['goal']:.3f}") == ("98.25", "88.425"), f"{Bb['p0']:.4f} / {Bb['goal']:.4f} (the tables print 88.42; both are roundings of 88.425)"),
        ("B2: every review reached its goal within 1,024", all(c['reached'] and c['review'] <= 1024 for r in RULES for c in b2[r]['cycles']), "all eight in both rules"),
        ("B2: the unrelated held score at the end of each gap is 98 to 100% from cycle 2 on; cycle 1: S 6.0%, B 98.0%",
         all(98 <= c['gap_un'] <= 100 for r in RULES for c in b2[r]['cycles'][1:]) and (round(S['cycles'][0]['gap_un'], 1), round(Bb['cycles'][0]['gap_un'], 1)) == (6.0, 98.0),
         "; ".join(f"{r}: " + ", ".join(f"{c['gap_un']:.1f}" for c in b2[r]['cycles']) for r in RULES)),
        ("B2: at the end of the reviews of cycles 1 to 3 it is 0 to 35% (S) and 0.3 to 70% (B); from cycle 4 on 88 to 100%",
         all(0 <= c['rev_un'] <= 35.5 for c in S['cycles'][:3]) and all(0.3 <= c['rev_un'] <= 70 for c in Bb['cycles'][:3]) and all(88 <= c['rev_un'] <= 100 for r in RULES for c in b2[r]['cycles'][3:]),
         "; ".join(f"{r}: " + ", ".join(f"{c['rev_un']:.1f}" for c in b2[r]['cycles']) for r in RULES)),
        ("B2: at the end of the run 4,616 updates under S and 4,448 under B", (S['end'], Bb['end']) == (4616, 4448), f"{S['end']} (S) and {Bb['end']} (B)"),
        ("B2: the reduction score at the end 87.9 (S) and 91.9 (B)", (round(S['red_end'], 1), round(Bb['red_end'], 1)) == (87.9, 91.9), f"{S['red_end']:.2f}, {Bb['red_end']:.2f}"),
        ("B2: the unrelated held score at the end 299/300 (S) and 294/300 (B)", (S['held_end_n'], Bb['held_end_n']) == (299, 294), f"{S['held_end_n']}, {Bb['held_end_n']}"),
        ("B2 S: cycle 8 review 32 updates against cycle 1's 80; more left 58.25 against 0.00", (S['cycles'][7]['review'], S['cycles'][0]['review'], S['cycles'][7]['gap_ff'], S['cycles'][0]['gap_ff']) == (32, 80, 58.25, 0.0), ""),
        ("B2 B: 48 updates against 80; more left 72.50 against 0.00", (Bb['cycles'][7]['review'], Bb['cycles'][0]['review'], Bb['cycles'][7]['gap_ff'], Bb['cycles'][0]['gap_ff']) == (48, 80, 72.5, 0.0), ""),
        ("B2 S: what is left after a gap in cycles 5 to 8 is 74.5, 65.0, 54.3, 58.3", [round(c['gap_ff'], 1) for c in S['cycles'][4:]] == [74.5, 65.0, 54.2, 58.2] or [round(c['gap_ff'] + 1e-9, 1) for c in S['cycles'][4:]] == [74.5, 65.0, 54.3, 58.3],
         ", ".join(f"{c['gap_ff']:.2f}" for c in S['cycles'][4:])),
        ("B3: the probe at update 64 under S: 0.0% at 0.001, 21.5% at 0.0003, 62.0% at 0.0001", [b3['S'][0][k]['probe64'] for k in RATES] == [0.0, 21.5, 62.0], ", ".join(f"{b3['S'][0][k]['probe64']}" for k in RATES)),
        ("B3: by the time the new task is at 80% the probe is at 0 to 0.25% at every rate", all(0 <= b3[r][0][k]['ff_at80'] <= 0.25 for r in RULES for k in RATES), ""),
    ]
    for text, ok, got in claims:
        print(f"    {note(ok, 'report: ' + text)}: {text}" + (f" | receipts: {got}" if got else ""))
    # section 5's figures of the checks are tested in part 3
    globals()["_B1"], globals()["_B2"], globals()["_B3"] = b1, b2, b3
    clock(1, t0)


# ================================================================ part 2: the predictions
def part2():
    t0 = time.time(); print("===== PART 2. The predictions of Part B, by the brief's fixed rules =====")
    b1 = {r: read_b1(r) for r in RULES}; b2 = {r: read_b2(r)[1] for r in RULES}; b3 = {r: read_b3(r) for r in RULES}
    kept4 = {r: b1[r][0][4]["kept"] for r in RULES}; kept32 = {r: b1[r][0][32]["kept"] for r in RULES}
    out = [
        ("PB.1", "A share of a quarter (k = 4) keeps the first family, under both rules", all(kept4.values()),
         "; ".join(f"{r}: row 0 {b1[r][0][4]['r0']:.2f}, row 4,096 {b1[r][0][4]['r4']:.2f}, drop {b1[r][0][4]['drop']:.2f} points ({b1[r][0][4]['pct']:.1f}% of row 0); kept {kept4[r]}" for r in RULES)),
        ("PB.2", "A share of 1/32 does not keep it, under either rule", not any(kept32.values()),
         "; ".join(f"{r}: drop {b1[r][0][32]['drop']:.2f} points ({b1[r][0][32]['pct']:.1f}% of row 0); kept {kept32[r]}" for r in RULES)),
        ("PB.3", "The review shortens, under back-propagation", b2["B"]["shortens"],
         "; ".join(f"{r}: cycle 1 {b2[r]['cycles'][0]['review']}, cycle 8 {b2[r]['cycles'][7]['review']} (shortens if at most {b2[r]['cycles'][0]['review'] / 2:g}): {b2[r]['shortens']}" for r in RULES)),
        ("PB.4", "More is left after a gap at cycle 8, under back-propagation", b2["B"]["more"],
         "; ".join(f"{r}: cycle 1 {b2[r]['cycles'][0]['gap_ff']:.2f}, cycle 8 {b2[r]['cycles'][7]['gap_ff']:.2f} (difference {b2[r]['cycles'][7]['gap_ff'] - b2[r]['cycles'][0]['gap_ff']:.2f}; at least 20 needed): {b2[r]['more']}" for r in RULES)),
        ("PB.5", "Smaller steps do not leave more, under either rule", not any(b3[r][1] for r in RULES),
         "; ".join(f"{r}: probe at the first row with the unrelated task at 80%: " + ", ".join(f"{k} {b3[r][0][k]['ff_at80']:.2f}" for k in RATES) + f"; smaller steps leave more: {b3[r][1]}" for r in RULES)),
    ]
    report_says = {"PB.1": False, "PB.2": True, "PB.3": False, "PB.4": True, "PB.5": True}
    for pid, text, held, fig in out:
        print(f"{pid} \"{text}\": {'held' if held else 'failed'} (report: {'held' if report_says[pid] else 'failed'}: {note(held == report_says[pid], 'prediction ' + pid)})\n     {fig}")
    print("2.1 the other reading of 'within 5 points' (the reduction score at 4,096 as a percentage of row 0 within 5 of 100) gives the same verdict at every share: "
          + str(all(b1[r][0][k]["kept"] == b1[r][0][k]["kept_pct_reading"] for r in RULES for k in (2, 4, 8, 16, 32, 0))))
    print("2.2 B1 beside the readings: E43's score rows, reduction score / unrelated held-out score (%)")
    for r in RULES:
        for k in (2, 4, 8, 16, 32, 0):
            d = b1[r][0][k]
            print(f"    {r} k={k:>2}: " + "  ".join(f"{g}: {d['red_rows'][g]:.1f}/{d['held_rows'][g]:.1f}" for g in E43ROWS)
                  + (f" | probe: first row at 80% {d['f80']}, first row from which it stays at 80% {d['f80_stays']}, lowest first-family probe {d['low']:.2f}" if k in KS else ""))
    print("2.3 B2: the unrelated task's held-out score at the start of each gap (the end of the previous review; cycle 1: update 0) and at its end, with the first-family probe at the end of the gap")
    for r in RULES:
        cy = b2[r]["cycles"]
        print(f"    {r}: " + "  ".join(f"c{c['cycle']}: {(0.0 if c['cycle'] == 1 else cy[c['cycle'] - 2]['rev_un']):.1f}->{c['gap_un']:.1f} ff {c['gap_ff']:.2f}" for c in cy))
    print("2.4 B3: the second reading of 'the last probe row at which the share is still at least half' (the row before the first drop below half), and the probe curve up to 1,024")
    for r in RULES:
        for k in RATES:
            d = b3[r][0][k]
            print(f"    {r} {k}: last row at half or more {d['last_half']} (held {d['held_there']:.1f}); row before the first drop {d['before_drop']} (held {d['held_before_drop']:.1f}); "
                  + " ".join(f"{u}:{f:.1f}/{h:.0f}" for u, f, h in d["curve"] if u <= 1024 and (u <= 256 or u % 128 == 0)))
    clock(2, t0)


# ================================================================ part 3: the checks 8 to 11
def part3():
    t0 = time.time(); print("===== PART 3. The checks of Part B =====")
    c8 = js(f"{B}checks/e51_check_8_b1_is_addendum_{SEED}.json")
    print(f"3.8 check 8 receipt: ok {c8['ok']}; written by `{c8['provenance']['command']}`")
    for r in RULES:
        mine = js(f"{B}check8/e51_B1_share_k2addendum_{r}_{SEED}.json"); ref = js(f"{P}e43/addendum2/e43_unrelated-rehearse_{r}_{SEED}.json", DREF)
        a, b = V.row(mine, 64), V.row(ref, 64)
        s1 = a["score1_first_family_all_pairs_k0"] == b["score1_first_family_all_pairs_k0"]; s2 = a["score2_first_family_heldout_k12"] == b["score2_first_family_heldout_k12"]
        th = mine["complete"]["tensors_sha256"] == ref["complete"]["tensors_sha256"]; dd = mine["complete"]["data_digest"] == ref["complete"]["data_digest"]
        allrows = all(V.row(mine, g)["score1_first_family_all_pairs_k0"] == V.row(ref, g)["score1_first_family_all_pairs_k0"] for g in E43ROWS)
        print(f"    {r}: against the addendum's receipt on the default branch: scores 1 and 2 at row 64 equal {s1 and s2}; tensors hash at 4,096 equal {th} ({mine['complete']['tensors_sha256'][:8]}); "
              f"data digest equal {dd}; score 1 equal at all six rows {allrows}; the receipt's own figures {c8['result'][r]}")
        note(s1 and s2 and th, f"check 8 {r}")
    c9 = js(f"{B}checks/e51_check_9_b3_is_e43_{SEED}.json")
    print(f"3.9 check 9 receipt: ok {c9['ok']}")
    for r in RULES:
        mine = rec_b3("0.001", r); ref = js(f"{P}e43/e43_unrelated_{r}_{SEED}.json", DREF)
        eq = {g: V.row(mine, g)["score1_first_family_all_pairs_k0"] == V.row(ref, g)["score1_first_family_all_pairs_k0"] and V.row(mine, g)["score2_first_family_heldout_k12"] == V.row(ref, g)["score2_first_family_heldout_k12"] for g in (64, 4096)}
        th = mine["complete"]["tensors_sha256"] == ref["complete"]["tensors_sha256"]; dd = all(V.row(mine, g)["data_digest"] == V.row(ref, g)["data_digest"] for g in (0, 64, 4096))
        lr = [g["lr"] for g in mine["complete"]["optimizer_groups_read_back"]]
        print(f"    {r}: against E43's receipt on the default branch: counts equal at rows 64 and 4,096 {eq}; tensors hash equal {th} ({mine['complete']['tensors_sha256'][:8]}); data digests equal {dd}; rates read back {lr}")
        note(all(eq.values()) and th, f"check 9 {r}")
    c10 = js(f"{B}checks/e51_check_10_probe_{SEED}.json")
    print(f"3.10 check 10 receipt: ok {c10['ok']}; {c10['result']}")
    S1 = V.first_family_items(); rng = Random(f"e51|{SEED}|probe")
    idx = {n: rng.sample([i for i, f in enumerate(S1[n][1]) if f], 200) for n in ("add89", "sub89")}
    for r in RULES:
        W, sha = V.load(f"{P}e26/e26_dev_{r}_{SEED}.pt", WREF); net = V.Net(W)
        bsha = js(f"{B}e51_B1_share_4_{r}_{SEED}.json")["settings"]["start"]
        n_ok = 0; full = 0
        for n in ("add89", "sub89"):
            outs = V.generate(net, S1[n][0]); ok = [o == t for o, (_, t) in zip(outs, S1[n][0])]
            n_ok += sum(ok[i] for i in idx[n]); full += sum(o for o, f in zip(ok, S1[n][1]) if f)
        e43r = V.row(js(f"{P}e43/e43_unrelated_{r}_{SEED}.json", DREF), 0)["score1_first_family_all_pairs_k0"]
        e43red = e43r["add89"]["reduction"]["exact"] + e43r["sub89"]["reduction"]["exact"]
        probe0 = [rec_b1(4, r)["probe"][0]["first_family_exact"], rec_b2(r)["probe"][0]["first_family_exact"]] + [rec_b3(k, r)["probe"][0]["first_family_exact"] for k in RATES]
        print(f"    {r}: E26's birth (file SHA-256 equal to the receipts' {sha == bsha['birth_sha256']}; tensors hash equal {V.tensors_sha(W) == bsha['birth_tensors_sha256']}): "
              f"the reader's forward answers {n_ok} of the 400 probe items (receipt {c10['result'][r]['probe_exact']}: {note(n_ok == c10['result'][r]['probe_exact'], 'check 10 ' + r)}); "
              f"the probe at update 0 in the five runs of the rule: {probe0}; all reduction pairs by the reader {full}, by E43's receipt at row 0 {e43red} ({note(full == e43red, 'E43 row 0 ' + r)})")
    c11 = js(f"{B}checks/e51_check_11_cost_{SEED}.json")["result"]
    per = c11["cpu_seconds_an_update"]
    pa = {r: js(f"{P}e51/e51_unrelated_open_{r}_{SEED}.json")["complete"] for r in RULES}
    per_mine = {r: pa[r]["cpu_updates"] / pa[r]["updates"] for r in RULES}
    tot = sum(c11["forecast_cpu_seconds"].values()) / 3600
    runs = [rec_b1(k, r) for k in KS for r in RULES] + [rec_b2(r) for r in RULES] + [rec_b3(k, r) for k in RATES for r in RULES]
    spent = sum(x["complete"]["cpu_total"] for x in runs) / 3600
    spent8 = sum(js(f"{B}check8/e51_B1_share_k2addendum_{r}_{SEED}.json")["complete"]["cpu_total"] for r in RULES) / 3600
    per_b = {r: [x["complete"]["cpu_updates"] / x["complete"]["updates"] for x in runs if x["settings"]["rule"] == r] for r in RULES}
    print(f"3.11 check 11: CPU-seconds an update from Part A's `open` runs {per} (reader from those receipts {per_mine}: {note(per == per_mine, 'check 11 rate')}); forecast {tot:.3f} CPU-hours "
          f"for {c11['runs']} runs (the receipt {c11['forecast_cpu_hours']:.3f}); {c11['note']}; the per-row and per-probe costs are constants of the runner (22.0 s and 1.5 s)")
    print(f"     spent, from the sixteen receipts' cpu_total: {spent:.3f} CPU-hours ; with the two runs of check 8: {spent + spent8:.3f} (report 4.95, which is this sum: {note(round(spent + spent8, 2) == 4.95, 'Part B CPU-hours')}); "
          f"measured CPU-seconds an update in Part B: S {min(per_b['S']):.3f} to {max(per_b['S']):.3f}, B {min(per_b['B']):.3f} to {max(per_b['B']):.3f}")
    # order of events, from the receipts' commits and logs
    order = []
    for name in ["checks/e51_check_10_probe", "checks/e51_check_8_b1_is_addendum", "checks/e51_check_9_b3_is_e43", "checks/e51_check_11_cost"]:
        order.append((name, js(f"{B}{name}_{SEED}.json")["provenance"]["git"]["commit"][:8]))
    print(f"3.12 the commit recorded by each check receipt at its writing: {order}")
    logs = subprocess.run(["git", "ls-tree", "-r", "--name-only", WREF, "--", B + "logs"], capture_output=True, text=True).stdout.split()
    stamps = sorted((int(l.rsplit(".", 2)[-2]), l.split("/")[-1].rsplit(".", 2)[0]) for l in logs)
    print("     the logs' start stamps (seconds), in order: " + "; ".join(f"{t}: {n}" for t, n in stamps))
    clock(3, t0)


# ================================================================ part 4: the weights
def part4():
    t0 = time.time(); print("===== PART 4. The weights =====")
    files = subprocess.run(["git", "ls-tree", "-r", "--name-only", WREF, "--", B], capture_output=True, text=True).stdout.split()
    pts = [f for f in files if f.endswith(".pt")]; ok = 0
    for f in pts:
        jf = f.rsplit("_row", 1)[0] + ".json" if "_row" in f else f.rsplit("_end", 1)[0] + ".json"
        want_sha = js(jf)["complete"]["weights_sha256"]; got = hashlib.sha256(show(f)).hexdigest()
        ok += got == want_sha
        if got != want_sha:
            diffs.append("weights sha " + f)
    print(f"4.1 the SHA-256 of {len(pts)} weight files of Part B against their receipts: {ok} equal")
    S1 = V.first_family_items(); HELD = V.held12(); pan = {"held": V.u_panel("held"), "taught": V.u_panel("taught")}
    rng = Random(f"e51|{SEED}|probe"); idx = {n: rng.sample([i for i, f in enumerate(S1[n][1]) if f], 200) for n in ("add89", "sub89")}
    for label, rec, wpath in (("B1 k = 8, rule B", rec_b1(8, "B"), f"{B}e51_B1_share_8_B_{SEED}_row4096.pt"),
                              ("B3 rate 0.0003, rule S", rec_b3("0.0003", "S"), f"{B}e51_B3_steps_0.0003_S_{SEED}_row13824.pt")):
        t1 = time.time(); W, sha = V.load(wpath, WREF); net = V.Net(W); x = rec["rows"][-1]
        r, gens = V.score_first_family(net, S1, HELD); a = V.score_new(net, pan)
        mine_probe = sum(gens["k0_" + n][i] == S1[n][0][i][1] for n in idx for i in idx[n]); lastp = rec["probe"][-1]
        same = V.same_counts(r, V.receipt_first_family(x))
        print(f"4.2 {label}, row {x['updates']}: tensors hash equal to the receipt {V.tensors_sha(W) == x['tensors_sha256']}; first family {V.fmt_ff(r)}; equal to the receipt: {note(same, 'weights ff ' + label)}")
        print(f"     receipt: {V.fmt_ff(V.receipt_first_family(x))}")
        print(f"     new task: {V.fmt_new(a)} (held-out {100 * a['held']['exact'] / 300:.1f}%); equal to the receipt's score 3 {x['score3_new_task']}: {note(a == x['score3_new_task'], 'weights new ' + label)}; "
              f"probe at row {lastp['updates']}: {mine_probe} of 400 first-family items (receipt {lastp['first_family_exact']}), held {a['held']['exact']} (receipt {lastp['unrelated_held_exact']}); "
              f"{note(mine_probe == lastp['first_family_exact'] and a['held']['exact'] == lastp['unrelated_held_exact'], 'weights probe ' + label)} [{time.time() - t1:.0f} s]")
    clock(4, t0)


PARTS = {"1": part1, "2": part2, "3": part3, "4": part4}

if __name__ == "__main__":
    for p in want:
        PARTS[p]()
    print("Figures that differ:", diffs if diffs else "none")
