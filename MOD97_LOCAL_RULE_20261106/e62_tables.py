"""E62: the tables, readings and outcomes of the predictions fixed in the brief (revision 2, "How the result is stated"), from the receipts.

    python3 e62_tables.py        # writes exp45/e62/E62_TABLES.md and exp45/e62/E62_TABLES.json from the complete receipts present

Definitions (the brief's): a threshold is reached at the first of three consecutive scored rows at or above it; the first single row at
or above it is given in brackets. A level at an update is the mean of the three scored rows that end there. The full run of a rule is its
run with N = 255. An N is as fast as the full run if it reaches 80 on R in at most 1.25 times the full run's updates; slower if it reaches
80 on R later than that within 49,152 updates; otherwise not reached. The record (N = 10 from the neighbourhood, E50's stage 3) and the
reference of decision 31 (E42's G under the exact gradient) are read from the default branch with git show.
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e62"
QUAL = HERE / "exp45" / "e62_qual"
SEED = 73129
DOSE = 49152
GRID = (2, 4, 8, 16, 32, 64, 128, 255)
D_REF = "origin/codex/minifly-consolidated-20261023"
RULE_NAME = {"S": "the sweep", "B": "the exact gradient"}


def git_json(path):
    return json.loads(subprocess.run(["git", "show", f"{D_REF}:MOD97_LOCAL_RULE_20261106/{path}"], cwd=HERE, capture_output=True, check=True).stdout)


def reach(curve, key, th=80):
    v = [(r["step"], r[key]) for r in curve]
    three = next((v[i][0] for i in range(len(v) - 2) if all(x >= th for _, x in v[i:i + 3])), None)
    single = next((s for s, x in v if x >= th), None)
    return three, single


def level(curve, key, u):
    by = {r["step"]: r[key] for r in curve}
    return sum(by[u - 256 * i] for i in range(3)) / 3


def summary(rec, name):
    c = rec["curve"]
    s = dict(receipt=name, N=rec["N"], choice=rec.get("choice", "near"), rule=rec["rule"], R80=reach(c, "R"), A80=reach(c, "A"),
             levels={str(u): {k: level(c, k, u) for k in ("A", "R", "taught")} for u in (24576, 49152)},
             weights_taking_part=rec["counts"]["weights_taking_part"], graph=rec.get("graph"), complete=rec.get("complete"),
             tensors_sha256=rec.get("tensors_sha256"), resumes=rec.get("resumes", []),
             cpu_hours=(rec["cpu_updates"] + rec["cpu_scoring"]) / 3600 if rec.get("complete") else None)
    u = s["R80"][0]
    if u is not None:
        row = next(r for r in c if r["step"] == u)
        s["at_R80"] = dict(multiply_adds=row["multiply_adds"], cpu_updates=row["cpu_updates"])
    return s


def load():
    runs = {}
    for N in GRID + (10,):
        for rule in "SB":
            p = OUT / f"e62_N{N}_all_{rule}_{SEED}.json"
            if p.exists():
                r = json.loads(p.read_text())
                if r.get("complete"):
                    runs[(N, "all", rule)] = summary(r, p.name)
    rec = {}
    for rule in "SB":
        r = git_json(f"exp45/e50/e50_C46-one_{rule}_73129.json")
        r["N"], r["choice"] = 10, "near"
        rec[rule] = summary(r, f"{D_REF}:MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C46-one_{rule}_73129.json")
    g = git_json("exp45/e42/e42_G_B_73129.json")
    ref = dict(receipt=f"{D_REF}:MOD97_LOCAL_RULE_20261106/exp45/e42/e42_G_B_73129.json", R80=reach(g["curve"], "R"), A80=reach(g["curve"], "A"))
    return runs, rec, ref


def f80(t):
    return "not reached" if t[0] is None else f"{t[0]:,} [{t[1]:,}]"


def fmt(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def reading(u, full):
    if full is None:
        return None
    if u is not None and u <= 1.25 * full:
        return "as fast"
    if u is not None:
        return "slower"
    return "not reached"


def main():
    runs, rec, ref = load()
    out = dict(seed=SEED, runs={f"{k[0]} {k[1]} {k[2]}": v for k, v in runs.items()}, record=rec, reference=ref, readings={}, sentences={}, predictions={})
    md = ["# E62 tables (seed 73129)", "",
          "Written by `python3 e62_tables.py` from the receipts under `exp45/e62/` and, for the record and the reference, from the default branch "
          f"(`{D_REF}`) with `git show`. Updates to a threshold: the first of three consecutive scored rows at or above it, with the first single "
          "row in brackets. Levels are means of three scored rows. All scores are percentages.", "",
          "## The table of the runs", "",
          "| N | Connections from | Exact gradient: updates to 80 on R | Sweep: updates to 80 on R | Ratio sweep / exact gradient | Exact gradient: to 80 on A | Sweep: to 80 on A | "
          "Read by no other cell | Every cell from every cell within | Weights taking part |", "|---|---|---|---|---|---|---|---|---|---|"]
    rows = [(N, "all") for N in GRID if N < 10] + [(10, "near"), (10, "all")] + [(N, "all") for N in GRID if N > 10]
    get = lambda N, ch, rule: rec[rule] if ch == "near" else runs.get((N, ch, rule))
    for N, ch in rows:
        b, s = get(N, ch, "B"), get(N, ch, "S")
        if b is None and s is None:
            continue
        ratio = s["R80"][0] / b["R80"][0] if b and s and b["R80"][0] and s["R80"][0] else None
        g = (b or s)["graph"] or {}
        md.append(f"| {N} | {'the neighbourhood (the record)' if ch == 'near' else 'all cells'} | {f80(b['R80']) if b else '—'} | {f80(s['R80']) if s else '—'} | "
                  f"{fmt(ratio)} | {f80(b['A80']) if b else '—'} | {f80(s['A80']) if s else '—'} | {g.get('read_by_no_other_cell', '0 (record)')} | "
                  f"{g.get('every_cell_from_every_cell_within', '5 (record)')} | {(b or s)['weights_taking_part']:,} |")
    md += ["", "## Levels, cost to 80 on R, and resumptions", "",
           "| N | From | Rule | A, R, T at 24,576 | A, R, T at 49,152 | Multiply-adds to 80 on R | CPU-seconds of updates to 80 on R | CPU-hours of the run | Resumed at |",
           "|---|---|---|---|---|---|---|---|---|"]
    for N, ch in rows:
        for rule in "BS":
            x = get(N, ch, rule)
            if x is None:
                continue
            lv = lambda u: ", ".join(fmt(x["levels"][str(u)][k]) for k in ("A", "R", "taught"))
            a = x.get("at_R80")
            md.append(f"| {N} | {'near' if ch == 'near' else 'all'} | {rule} | {lv(24576)} | {lv(49152)} | {'—' if a is None else f'{a['multiply_adds']:,}'} | "
                      f"{'—' if a is None else f'{a['cpu_updates']:.0f}'} | {fmt(x['cpu_hours'])} | {[r['update'] for r in x['resumes']] or 'never'} |")
    md += ["", "## Reached within k steps (least / mean / most), from the receipts", "", "| N | From | 1 | 2 | 3 | 4 | 6 | 8 | Strongly connected parts (largest) |", "|---|---|---|---|---|---|---|---|---|"]
    for N, ch in rows:
        x = get(N, ch, "S") or get(N, ch, "B")
        if x is None or not x["graph"]:
            continue
        g = x["graph"]
        md.append(f"| {N} | {ch} | " + " | ".join(f"{v['least']} / {v['mean']:.1f} / {v['most']}" for v in g["reached_within"].values())
                  + f" | {g['strongly_connected_parts']} ({g['largest_part']}) |")
    # ---------------------------------------------------------------- readings
    md += ["", "## Readings", ""]
    for rule in "BS":
        full = runs.get((255, "all", rule))
        if full is None:
            continue
        fu = full["R80"][0]
        grid = {N: runs.get((N, "all", rule)) for N in GRID}
        if fu is None:
            lv = {N: x["levels"]["49152"]["R"] for N, x in grid.items() if x}
            sent = (f"On seed 73129 {RULE_NAME[rule]} did not reach 80 on R within 49,152 updates with every cell reading every other; its levels of R at update "
                    f"49,152 were, for N = " + ", ".join(str(N) for N in lv) + ": " + ", ".join(f"{v:.2f}" for v in lv.values()) + ".")
            out["readings"][rule] = dict(full_run_reached=False, levels_R_49152=lv)
        else:
            rd = {N: reading(x["R80"][0], fu) for N, x in grid.items() if x}
            ups = {N: x["R80"][0] for N, x in grid.items() if x}
            fast = [N for N in GRID if N < 255 and rd.get(N) == "as fast"]
            nstar = min(fast) if fast else None
            larger = [N for N in GRID if nstar is not None and nstar < N < 255]
            all_larger = all(rd.get(N) == "as fast" for N in larger)
            reached_any = [N for N in GRID if N < 255 and ups.get(N) is not None]
            sent = (f"On seed 73129, with connections drawn from all cells, {RULE_NAME[rule]} reached 80 on R after u updates with N connections a cell, for N = "
                    + ", ".join(str(N) for N in ups) + ": " + ", ".join("not reached" if v is None else f"{v:,}" for v in ups.values()) + "; ")
            if nstar is not None:
                sent += (f"the smallest N of the grid of powers of two that was as fast as the network in which every cell reads every other was {nstar}, "
                         f"and every larger N of the grid was {'also' if all_larger else 'not in every case'} as fast.")
            else:
                sent += ("no N below 255 was as fast as the network in which every cell reads every other; the smallest N that reached 80 at all was "
                         f"{min(reached_any) if reached_any else 'none'}.")
            out["readings"][rule] = dict(full_run_updates=fu, limit=1.25 * fu, words=rd, updates=ups, smallest_as_fast=nstar, every_larger_as_fast=all_larger)
        out["sentences"][f"grid {rule}"] = sent
        md += [f"- {sent}"]
        t = runs.get((10, "all", rule))
        if t:
            ru = rec[rule]["R80"][0]
            s2 = (f"With 10 connections from all cells {RULE_NAME[rule]} reached 80 on R after {'—' if t['R80'][0] is None else f'{t['R80'][0]:,}'} updates, against "
                  f"{ru:,} ({RULE_NAME[rule]}) with 10 connections from the neighbourhood.")
            if t["R80"][0] is None:
                s2 = s2.replace("reached 80 on R after — updates", "did not reach 80 on R within 49,152 updates")
            out["sentences"][f"ten {rule}"] = s2
            md += [f"- {s2}"]
    both = {N: (runs[(N, "all", "S")]["R80"][0], runs[(N, "all", "B")]["R80"][0]) for N in GRID + (10,)
            if (N, "all", "S") in runs and (N, "all", "B") in runs and runs[(N, "all", "S")]["R80"][0] and runs[(N, "all", "B")]["R80"][0]}
    if both:
        s3 = ("The ratio of the sweep's updates to the exact gradient's was " + "; ".join(f"{s / b:.2f} at N = {N}" for N, (s, b) in both.items())
              + f", beside the record's {rec['S']['R80'][0] / rec['B']['R80'][0]:.2f} (N = 10 from the neighbourhood).")
        out["sentences"]["ratio"] = s3; md += [f"- {s3}"]
    md += ["", f"Beside the record: decision 31's reference (E42's `G` under the exact gradient, `{ref['receipt']}`) reached 80 on R after {f80(ref['R80'])} updates "
           f"and on A after {f80(ref['A80'])}.", ""]
    # ---------------------------------------------------------------- predictions
    P = out["predictions"]
    have = lambda rule, Ns: all((N, "all", rule) in runs for N in Ns)
    r80 = lambda N, rule: runs[(N, "all", rule)]["R80"][0]
    chk = [QUAL / f"check{k}_{SEED}.json" for k in (1, 2, 3, 5)]
    if all(p.exists() for p in chk):
        P["P62.1"] = dict(holds=all(json.loads(p.read_text())["pass"] for p in chk), note="each check was run once")
    if have("B", (4, 8, 16, 32, 64, 128, 255)):
        P["P62.2"] = dict(holds=all(r80(N, "B") is not None for N in (4, 8, 16, 32, 64, 128, 255)), updates={N: r80(N, "B") for N in (4, 8, 16, 32, 64, 128, 255)})
    if have("B", (2,)):
        P["P62.3"] = dict(holds=r80(2, "B") is not None, updates=r80(2, "B"))
    if have("S", (16, 32, 64, 128, 255)):
        P["P62.4"] = dict(holds=all(r80(N, "S") is not None for N in (16, 32, 64, 128, 255)), updates={N: r80(N, "S") for N in (16, 32, 64, 128, 255)})
    if have("S", (2,)):
        P["P62.5"] = dict(holds=r80(2, "S") is None, updates=r80(2, "S"))
    if have("S", (255,)) and have("B", (255,)):
        P["P62.6"] = dict(holds=all(r80(255, r) is not None and r80(255, r) < rec[r]["R80"][0] for r in "SB"),
                          updates={r: (r80(255, r), rec[r]["R80"][0]) for r in "SB"})
    if have("S", (10,)) and have("B", (10,)):
        P["P62.7"] = dict(holds=all(r80(10, r) is not None and 0.75 * rec[r]["R80"][0] <= r80(10, r) <= 1.25 * rec[r]["R80"][0] for r in "SB"),
                          ratios={r: None if r80(10, r) is None else r80(10, r) / rec[r]["R80"][0] for r in "SB"})
    if have("S", GRID) and have("B", GRID):
        bothg = {N: (r80(N, "S"), r80(N, "B")) for N in GRID if r80(N, "S") and r80(N, "B")}
        P["P62.8"] = dict(holds=all(s > b for s, b in bothg.values()), pairs=bothg)
    if out["readings"].get("B", {}).get("words") and 32 in out["readings"]["B"]["words"]:
        P["P62.9"] = dict(holds=out["readings"]["B"]["words"][32] == "as fast", word=out["readings"]["B"]["words"][32])
    if P:
        md += ["## Predictions", "", "| | Holds | Figures |", "|---|---|---|"] + [
            f"| {k} | {'yes' if v['holds'] else 'no'} | {json.dumps({a: b for a, b in v.items() if a != 'holds'})} |" for k, v in P.items()] + [""]
    (OUT / "E62_TABLES.json").write_text(json.dumps(out, indent=1, default=str))
    (OUT / "E62_TABLES.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
