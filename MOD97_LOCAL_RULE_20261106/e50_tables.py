"""E50 tables (brief OPUS_TASK_E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE_20261116.md, revision 2): thresholds in updates and
in multiply-adds (two counts), the levels, the four operations and the two answer bytes apart, the cosines, the figure "by what a cell
knows", and the descriptions fixed beforehand: `C-dense` against E42's `F10`; `C16-in` and `C46-one` against `C-dense` of the same rule,
with E49's `in16` against `F10` beside them; and the brief's rules for stages 2 and 3. Computed from the receipts only (E49's forms,
e49_tables.py at a36de661, written again here for this job's settings and its two doses); kept apart from exp50_code.py so that the
runner stays at its commit during the runs.

    python3 e50_tables.py [--out DIR] [--tag NAME] [--partial]

Reads DIR/e50_<setting>_<rule>_73129.json (complete runs; with --partial also runs in progress, marked) and DIR/reference/
e42_F10_<rule>_73129.json and e49_in16_<rule>_73129.json (copied from a36de661; DIR/reference/SOURCE.txt). Writes
DIR/E50_TABLES_<tag>.md and DIR/E50_SUMMARY_<tag>.json. A run of 49,152 updates is read twice: within its first 24,576 updates (the
dose of `C-dense`, of `F10` and of E49) and at its end. No reading beyond the brief's forms.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e50"
SEED = 73129
FIRST, ROW, BATCH = 24576, 256, 64
THRESHOLDS = (50, 80, 90)
SETTINGS = ("C-dense", "C16-in", "C46-one")
MA = {"F10": 930560, "in16": 549632, "C-dense": 926464, "C16-in": 545536, "C46-one": 568576}          # the first count (E49's)
MA2 = {"C-dense": 285184, "C16-in": 277000, "C46-one": 277504}                                           # products with a turned-on value
OPS = ("add89", "sub89", "max89", "min89")
OPN = {"add89": "add", "sub89": "sub", "max89": "max", "min89": "min"}
MAPS = ("rnn.weight_ih_l0", "lm.weight", "rnn.weight_hh_l0")


def rows_of(rec):
    """The scored rows with the four operations as shares, also for E42's and E49's receipts (whose A_counts hold all four)."""
    out = []
    for r in rec["curve"]:
        q = dict(r)
        q.update({OPN[k]: 100 * r["A_counts"][k] / 300 for k in OPS})
        out.append(q)
    return out


def first_row(curve, key, th, version):
    v = [(r["step"], r[key]) for r in curve]
    for i, (u, a) in enumerate(v):
        if version == "single" and a >= th:
            return i
        if version == "three" and i + 2 < len(v) and all(x >= th for _, x in v[i:i + 3]):
            return i
    return None


def mean_rows(curve, key, end_step, n=3):
    rows = [r for r in curve if r["step"] <= end_step][-n:]
    return sum(r[key] for r in rows) / len(rows)


def summarize(rec, setting, upto=None):
    """One run read within its first `upto` updates (all of it when upto is None)."""
    curve = rows_of(rec)
    if upto is not None:
        curve = [r for r in curve if r["step"] <= upto]
    last = curve[-1]["step"]; ma = MA[setting]
    out = dict(setting=setting, rule=rec["rule"], seed=rec["seed"], dose=last, run_dose=rec["dose"], complete=bool(rec.get("complete")),
               multiply_adds_per_position=ma, multiply_adds_second_count=MA2.get(setting), thresholds={})
    for key in ("A", "R") + tuple(OPN.values()):
        for th in (THRESHOLDS if key in "AR" else (80,)):
            for ver in ("single", "three"):
                i = first_row(curve, key, th, ver)
                r = curve[i] if i is not None else None
                out["thresholds"][f"{key}{th}_{ver}"] = None if r is None else dict(
                    updates=r["step"], multiply_adds=ma * BATCH * r["step"],
                    multiply_adds_second_count=None if setting not in MA2 else MA2[setting] * BATCH * r["step"])
    for key in ("A", "R"):
        best = max(curve, key=lambda r: r[key])
        out[f"max_{key}"] = dict(value=best[key], row=best["step"])
    keys = ["A", "R", "len40", "len12", "taught"] + list(OPN.values()) + (["R_first_byte", "R_second_byte"] if "R_first_byte" in curve[0] else [])
    out["levels"] = {k: mean_rows(curve, k, last) for k in keys}
    out["R_last3_minus_2048_earlier"] = out["levels"]["R"] - mean_rows(curve, "R", last - 2048)
    out["final"] = {k: curve[-1][k] for k in keys}
    out["digest_4096"] = next((r["data_digest"]["sha256"] for r in curve if r["step"] == 4096), None)
    out["limit_every_row"] = all(all(r["limit"].values()) for r in curve)
    if "kept_mean_magnitude" in curve[0]:
        out["magnitude_start"], out["magnitude_end"] = curve[0]["kept_mean_magnitude"], curve[-1]["kept_mean_magnitude"]
        out["masks_every_row"] = all(all(all(v.values()) for v in r["masks"].values()) for r in curve)
    if "fixed" in curve[0]:
        out["code_fixed_every_row"] = all(all(r["fixed"].values()) for r in curve)
        out["knows_start"], out["knows_end"] = curve[0]["by_what_a_cell_knows"], curve[-1]["by_what_a_cell_knows"]
        out["input_size_start"], out["input_size_end"] = curve[0]["input_size"], curve[-1]["input_size"]
    out["cosine_start"] = rec.get("cosine_first_batch_start")
    out["cosine_end"] = rec.get("cosine_first_batch_end") if last == rec["dose"] else (rec.get("first_dose") or {}).get("cosine_first_batch")
    out["cpu_updates"], out["cpu_scoring"] = curve[-1].get("cpu_updates"), curve[-1].get("cpu_scoring")
    out["tensors_sha256"] = rec.get("tensors_sha256") if last == rec["dose"] else (rec.get("first_dose") or {}).get("tensors_sha256")
    out["weights_sha256"] = rec.get("weights_sha256") if last == rec["dose"] else (rec.get("first_dose") or {}).get("weights_sha256")
    out["byte_table_and_l1_bit_equal_to_start"] = rec.get("byte_table_and_l1_bit_equal_to_start")
    out["curve"] = [dict(step=r["step"], A=r["A"], R=r["R"], loss=r["loss"], len40=r["len40"], **{k: r[k] for k in OPN.values()},
                         **({"R_first_byte": r["R_first_byte"], "R_second_byte": r["R_second_byte"]} if "R_first_byte" in r else {})) for r in curve]
    return out


def describe(xr, xc, Dr, Dc):
    """E41's table of four cases (as in e42_tables.py and e49_tables.py), the reference against the candidate."""
    if xr is not None and xc is not None:
        q = xc / xr
        return "fewer" if q < 2 / 3 else "more" if q > 1.5 else "about the same"
    if xr is not None:
        return "more" if Dc / xr >= 1.5 else "not decided"
    if xc is not None:
        return "fewer" if xc / Dr <= 2 / 3 else "not decided"
    return "neither reaches"


MA_WORDS = {"fewer": "falls", "more": "rises", "about the same": "stays", "not decided": "not decided", "neither reaches": "neither reaches"}


def thr_desc(sr, sc, key, ver, unit):
    t = f"{key}_{ver}"
    cr, cc = sr["thresholds"][t], sc["thresholds"][t]
    if unit == "updates":
        mr = mc = 1
    elif unit == "multiply-adds":
        mr, mc = MA[sr["setting"]] * BATCH, MA[sc["setting"]] * BATCH
    else:                                                   # the second count: only between settings of this job
        mr, mc = MA2[sr["setting"]] * BATCH, MA2[sc["setting"]] * BATCH
    Dr = (sr["dose"] - (512 if ver == "three" else 0)) * mr
    Dc = (sc["dose"] - (512 if ver == "three" else 0)) * mc
    xr = cr["updates"] * mr if cr else None
    xc = cc["updates"] * mc if cc else None
    d = describe(xr, xc, Dr, Dc)
    return (d if unit == "updates" else MA_WORDS[d]), xr, xc


def level_desc(sr, sc, k):
    d = sc["levels"][k] - sr["levels"][k]
    return ("lower" if d <= -15 else "higher" if d >= 15 else "about the same"), sr["levels"][k], sc["levels"][k]


def load(out_dir, partial):
    """S[(name, rule)]: `F10`, `in16`, `C-dense`; and for a run of 49,152 updates two entries, `<setting> @24,576` and `<setting> @end`."""
    S = {}; notes = []
    for p in sorted(Path(out_dir).glob(f"e50_*_*_{SEED}.json")):
        if "_dry" in p.name or p.name.endswith("_answers.json"):
            continue
        r = json.loads(p.read_text())
        if r.get("experiment") != "E50":
            continue
        if not r.get("complete"):
            if not partial:
                continue
            notes.append(f"`{r['setting']}` {r['rule']} is in progress at {r['curve'][-1]['step']:,} of {r['dose']:,} updates")
        s = r["setting"]
        if r["dose"] > FIRST:
            if r["curve"][-1]["step"] >= FIRST:
                S[(f"{s} @24,576", r["rule"])] = summarize(r, s, FIRST)
            S[(f"{s} @end", r["rule"])] = summarize(r, s)
        else:
            S[(s, r["rule"])] = summarize(r, s)
    for rule in "SB":
        for name, f in (("F10", f"e42_F10_{rule}_{SEED}.json"), ("in16", f"e49_in16_{rule}_{SEED}.json")):
            r = json.loads((Path(out_dir) / "reference" / f).read_text())
            assert r["complete"] and r["dose"] == FIRST and r.get("config") == name
            S[(name, rule)] = summarize(r, name)
    return S, notes


def fmt_u(c):
    return "not reached" if c is None else f"{c['updates']:,}"


def fmt_ma(c, k="multiply_adds"):
    return "not reached" if c is None or c[k] is None else f"{c[k]:.3g}"


def stage_rules(S):
    """The brief's rules. Stage 2 (`C16-in`) under each rule under which `C-dense` reaches 80 on R within its dose (three-row version);
    stage 3 (`C46-one`) under each rule under which `C16-in` does not reach 80 on R within its dose of 49,152 updates."""
    out = {}
    for rule in "SB":
        d = dict(stage2=None, stage3=None)
        if ("C-dense", rule) in S and S[("C-dense", rule)]["complete"]:
            c = S[("C-dense", rule)]["thresholds"]["R80_three"]
            d["stage2"] = dict(run=c is not None, reason=f"`C-dense` reaches 80 on R at {c['updates']:,} updates" if c else "`C-dense` does not reach 80 on R within its dose")
        if (f"C16-in @end", rule) in S and S[("C16-in @end", rule)]["complete"]:
            c = S[("C16-in @end", rule)]["thresholds"]["R80_three"]
            d["stage3"] = dict(run=c is None, reason="`C16-in` does not reach 80 on R within its dose" if c is None else f"`C16-in` reaches 80 on R at {c['updates']:,} updates")
        out[rule] = d
    return out


def write(out_dir, tag, partial):
    S, notes = load(out_dir, partial)
    names = ["F10", "in16", "C-dense", "C16-in @24,576", "C16-in @end", "C46-one @24,576", "C46-one @end"]
    order = [(s, r) for r in "BS" for s in names if (s, r) in S]
    mine = [k for k in order if k[0] not in ("F10", "in16")]
    L = [f"Receipts read from `exp45/e50/` (E50) and `exp45/e50/reference/` (E42's `F10` and E49's `in16`, seed {SEED}, copied from a36de661). "
         f"Rules: B is backpropagation, S the sweep. A run of 49,152 updates appears twice: read within its first 24,576 updates (`@24,576`) "
         f"and at its end (`@end`). Thresholds are in the three-row version (the first of three consecutive scored rows at or above the "
         f"threshold), with the single-row version in brackets.\n"]
    if notes:
        L.append("**Runs in progress (their rows are provisional):** " + "; ".join(notes) + ".\n")
    L.append("### Checks inside the runs: batch digest at row 4,096 (E49's check 4), the masks (E49's check 3), E42's limit, the code (check 2)\n")
    L.append("| Setting | Rule | digest at 4,096 (first 16) | masks held at every row | E42's limit held at every row | code table, byte table and `l1` unchanged at every row |\n|---|---|---|---|---|---|")
    for k in order:
        v = S[k]
        L.append(f"| {k[0]} | {k[1]} | {(v['digest_4096'] or '')[:16]} | {v.get('masks_every_row', '(E42 receipt)')} | {v['limit_every_row']} | {v.get('code_fixed_every_row', '(learned code)')} |")
    L.append("")
    L.append("### Per run: updates to the thresholds on A and R\n")
    L.append("| Setting | Rule | A50 | A80 | A90 | R50 | R80 | R90 | max A (row) | max R (row) | CPU-h updates + scoring |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in order:
        v = S[k]; t = v["thresholds"]
        cells = [f"{fmt_u(t[f'{key}{th}_three'])} [{fmt_u(t[f'{key}{th}_single'])}]" for key in "AR" for th in THRESHOLDS]
        L.append(f"| {k[0]} | {k[1]} | " + " | ".join(cells) + f" | {v['max_A']['value']:.2f} ({v['max_A']['row']:,}) | {v['max_R']['value']:.2f} "
                 f"({v['max_R']['row']:,}) | {v['cpu_updates'] / 3600:.2f} + {v['cpu_scoring'] / 3600:.2f} |")
    L.append("")
    L.append("### Per run: the four operations apart (the panel behind A: untaught pairs at 12 letters, 300 items each): updates to 80, and the level (mean of the last three rows)\n")
    L.append("| Setting | Rule | max: to 80 | min: to 80 | add: to 80 | sub: to 80 | levels max, min, add, sub |\n|---|---|---|---|---|---|---|")
    for k in order:
        v = S[k]; t = v["thresholds"]; lv = v["levels"]
        L.append(f"| {k[0]} | {k[1]} | " + " | ".join(f"{fmt_u(t[f'{o}80_three'])} [{fmt_u(t[f'{o}80_single'])}]" for o in ("max", "min", "add", "sub"))
                 + f" | {lv['max']:.1f}, {lv['min']:.1f}, {lv['add']:.1f}, {lv['sub']:.1f} |")
    L.append("")
    L.append("### Per run: multiply-adds to the thresholds, three-row version. First count: weights that take part x 64 items x updates (E49's count; for this job without `l1`). Second count: products with a turned-on value x 64 x updates (this job's settings only)\n")
    L.append("| Setting | Rule | first count a position | A80 | R80 | second count a position | A80 | R80 |\n|---|---|---|---|---|---|---|---|")
    for k in order:
        v = S[k]; t = v["thresholds"]; s2 = v["multiply_adds_second_count"]
        L.append(f"| {k[0]} | {k[1]} | {v['multiply_adds_per_position']:,} | {fmt_ma(t['A80_three'])} | {fmt_ma(t['R80_three'])} | "
                 f"{'' if s2 is None else f'{s2:,}'} | {fmt_ma(t['A80_three'], 'multiply_adds_second_count') if s2 else ''} | {fmt_ma(t['R80_three'], 'multiply_adds_second_count') if s2 else ''} |")
    L.append("")
    L.append("### Per run: the levels (mean of the last three scored rows), and R at the end against 2,048 updates earlier\n")
    L.append("| Setting | Rule | A | R | length at 40 | length at 12 | taught | R: first answer byte right | R: second answer byte right | R: last three minus the three ending 2,048 earlier |\n|---|---|---|---|---|---|---|---|---|---|")
    for k in order:
        v = S[k]; lv = v["levels"]; d = v["R_last3_minus_2048_earlier"]
        by = (f"{lv['R_first_byte']:.2f} | {lv['R_second_byte']:.2f}") if "R_first_byte" in lv else "(not recorded) | (not recorded)"
        L.append(f"| {k[0]} | {k[1]} | {lv['A']:.2f} | {lv['R']:.2f} | {lv['len40']:.2f} | {lv['len12']:.2f} | {lv['taught']:.2f} | {by} | "
                 f"{d:+.2f}{', still rising' if d >= 5 and v['thresholds']['R80_three'] is None else ''} |")
    L.append("")
    L.append("### Per run: A, R every 2,048 updates, and for this job's runs the share of R's items with the first and with the second answer byte right\n")
    last = max(S[k]["dose"] for k in order)
    steps = list(range(2048, last + 1, 2048))
    L.append("| Setting | Rule | " + " | ".join(f"{u:,}" for u in steps) + " |\n|---|---|" + "---|" * len(steps))
    for k in order:
        if k[0].endswith("@24,576"):
            continue
        rows = {r["step"]: r for r in S[k]["curve"]}
        cell = lambda r: f"{r['A']:.1f}, {r['R']:.1f}" + (f" ({r['R_first_byte']:.0f}, {r['R_second_byte']:.0f})" if "R_first_byte" in r else "")
        L.append(f"| {k[0].replace(' @end', '')} | {k[1]} | " + " | ".join(cell(rows[u]) if u in rows else "" for u in steps) + " |")
    L.append("")
    L.append("### Per run: max, min, add, sub at 12 letters every 4,096 updates\n")
    steps4 = list(range(4096, last + 1, 4096))
    L.append("| Setting | Rule | " + " | ".join(f"{u:,}" for u in steps4) + " |\n|---|---|" + "---|" * len(steps4))
    for k in order:
        if k[0].endswith("@24,576"):
            continue
        rows = {r["step"]: r for r in S[k]["curve"]}
        L.append(f"| {k[0].replace(' @end', '')} | {k[1]} | " + " | ".join(f"{rows[u]['max']:.0f}, {rows[u]['min']:.0f}, {rows[u]['add']:.0f}, {rows[u]['sub']:.0f}" if u in rows else "" for u in steps4) + " |")
    L.append("")
    L.append("### Per run: mean training loss of the 256 updates before the row\n")
    lsteps = [u for u in (256, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 32768, 40960, 49152) if u <= last]
    L.append("| Setting | Rule | " + " | ".join(f"{u:,}" for u in lsteps) + " |\n|---|---|" + "---|" * len(lsteps))
    for k in order:
        if k[0].endswith("@24,576"):
            continue
        rows = {r["step"]: r for r in S[k]["curve"]}
        L.append(f"| {k[0].replace(' @end', '')} | {k[1]} | " + " | ".join(f"{rows[u]['loss']:.3f}" if u in rows else "" for u in lsteps) + " |")
    L.append("")
    L.append("### This job's runs: cosine between the sweep's and the exact gradient of each map (first batch; at the start and at the end of the reading), the mean magnitude of the kept weights, and a cell's input for a needed byte (root mean square, without the bias)\n")
    L.append("| Setting | Rule | input map: cosine start, end | lm: cosine start, end | recurrent: cosine start, end | input map: magnitude start, end | lm: magnitude start, end | recurrent: magnitude start, end | input for a needed byte: start, end |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    f3 = lambda c, k_: "n/a" if not c else f"{c[k_]:.3f}"
    for k in mine:
        v = S[k]; cs, ce = v["cosine_start"], v["cosine_end"]; ms, me = v["magnitude_start"], v["magnitude_end"]
        L.append(f"| {k[0]} | {k[1]} | " + " | ".join(f"{f3(cs, mp)}, {f3(ce, mp)}" for mp in MAPS) + " | "
                 + " | ".join(f"{ms[mp]:.4f}, {me[mp]:.4f}" for mp in MAPS)
                 + f" | {v['input_size_start']['needed_rms']:.3f}, {v['input_size_end']['needed_rms']:.3f} |")
    L.append("")
    L.append("### This job's runs: by what a cell knows. Mean magnitude of the kept recurrent weights into the cells that, under 16 values a cell, have input for 0 to 2, for 3 or 4 and for 5 or more of the ten digits (95, 131 and 30 cells); neighbours only [with the cell's reading of itself]. The grouping means nothing in `C-dense` and `C46-one`. The figure describes the size of weights and is not read as use\n")
    L.append("| Setting | Rule | 0 to 2 digits: start, end | 3 or 4: start, end | 5 or more: start, end |\n|---|---|---|---|---|")
    for k in mine:
        v = S[k]; a, b = v["knows_start"], v["knows_end"]
        L.append(f"| {k[0]} | {k[1]} | " + " | ".join(f"{a[g]['neighbours_only']:.4f}, {b[g]['neighbours_only']:.4f} [{a[g]['with_self']:.4f}, {b[g]['with_self']:.4f}]"
                                                      for g in ("digits_0_to_2", "digits_3_to_4", "digits_5_or_more")) + " |")
    L.append("")
    L.append("### Descriptions fixed beforehand. * where the single-row version describes the cell differently\n")
    L.append("| Candidate | Reference | Rule | Measure | Description | Reference | Candidate | Ratio |\n|---|---|---|---|---|---|---|---|")
    desc = {}
    pairs = [("C-dense", "F10"), ("in16", "F10"), ("C16-in @24,576", "C-dense"), ("C16-in @end", "C-dense"), ("C46-one @24,576", "C-dense"),
             ("C46-one @end", "C-dense"), ("C46-one @end", "C16-in @end")]
    for rule in "BS":
        for c, rf in pairs:
            if (c, rule) not in S or (rf, rule) not in S:
                continue
            sr, sc = S[(rf, rule)], S[(c, rule)]
            units = ("updates", "multiply-adds") + (("multiply-adds, second count",) if rf != "F10" else ())
            for key in ("A80", "R80"):
                for unit in units:
                    d3, xr, xc = thr_desc(sr, sc, key, "three", unit)
                    d1 = thr_desc(sr, sc, key, "single", unit)[0]
                    desc[f"{c} against {rf} {rule} {key} {unit}"] = dict(three=d3, single=d1, reference=xr, candidate=xc)
                    f = (lambda z: "not reached" if z is None else f"{z:,}") if unit == "updates" else (lambda z: "not reached" if z is None else f"{z:.3g}")
                    L.append(f"| {c} | {rf} | {rule} | {key} {unit} | {d3}{'*' if d1 != d3 else ''} | {f(xr)} | {f(xc)} | {'' if xr is None or xc is None else f'{xc / xr:.2f}'} |")
            for kk in ("A", "R", "len40"):
                d, a, b = level_desc(sr, sc, kk)
                desc[f"{c} against {rf} {rule} level {kk}"] = dict(description=d, reference=a, candidate=b)
                L.append(f"| {c} | {rf} | {rule} | level of {kk if kk != 'len40' else 'the length test at 40'} | {d} | {a:.2f} | {b:.2f} | |")
    L.append("")
    L.append("### The brief's rules for stages 2 and 3\n")
    st = stage_rules(S)
    for rule in "BS":
        for stg, nm in (("stage2", "stage 2 (`C16-in`)"), ("stage3", "stage 3 (`C46-one`)")):
            c = st[rule][stg]
            L.append(f"- {rule}, {nm}: " + ("not yet decided (the run it depends on is not complete)" if c is None else (("**runs**: " if c["run"] else "does not run: ") + c["reason"])))
    L.append("")
    (Path(out_dir) / f"E50_TABLES_{tag}.md").write_text("\n".join(L) + "\n")
    js = dict(runs={f"{k[0]} {k[1]}": v for k, v in S.items()}, descriptions=desc, stages=st, in_progress=notes)
    (Path(out_dir) / f"E50_SUMMARY_{tag}.json").write_text(json.dumps(js, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    write(Path(opt("--out", str(OUT))), opt("--tag", str(SEED)), "--partial" in a)
