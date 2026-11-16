"""E49 tables (brief OPUS_TASK_E49_A_LIMITED_INTERFACE_20261116.md, revision 2): thresholds in updates and multiply-adds, the three levels,
the cosines, the fixed descriptions against E42's F10 of seed 73129 under the same rule, and the brief's condition for stage 2. Computed
from the receipts only; kept apart from exp49_interface.py so the runner stays at its commit during the runs.

    python3 e49_tables.py [--out DIR] [--tag NAME] [--f10-seeds DIR]

Reads DIR/e49_<setting>_<rule>_73129.json (complete full runs) and DIR/reference_e42/e42_F10_<rule>_73129.json (E42's, copied from
7fbdbdc2). With --f10-seeds, also E42's e42_F10_<rule>_<seed>.json of the other seeds, when complete, for the brief's seed comparison.
Writes DIR/E49_TABLES_<tag>.md and DIR/E49_SUMMARY_<tag>.json. No reading beyond the brief's forms.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e49"
SEED = 73129
DOSE, ROW = 24576, 256
THRESHOLDS = (50, 80, 90)
SETTINGS = ("I64", "I16", "in16", "out8")
MA = {"F10": 930560, "I64": 471808, "I16": 422656, "in16": 549632, "out8": 803584}
BATCH = 64
MAPS = ("rnn.weight_ih_l0", "lm.weight", "rnn.weight_hh_l0")
MAP_NAME = {"rnn.weight_ih_l0": "input map", "lm.weight": "lm", "rnn.weight_hh_l0": "recurrent (E42's)"}


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


def summarize(rec, setting):
    curve = rec["curve"]; ma = MA[setting]
    out = dict(setting=setting, rule=rec["rule"], seed=rec["seed"], dose=rec["dose"], multiply_adds_per_position=ma, thresholds={})
    for key in ("A", "R"):
        for th in THRESHOLDS:
            for ver in ("single", "three"):
                i = first_row(curve, key, th, ver)
                r = curve[i] if i is not None else None
                out["thresholds"][f"{key}{th}_{ver}"] = None if r is None else dict(updates=r["step"], multiply_adds=ma * BATCH * r["step"])
        best = max(curve, key=lambda r: r[key])
        out[f"max_{key}"] = dict(value=best[key], row=best["step"])
    last = curve[-1]["step"]
    out["levels"] = {k: mean_rows(curve, k, last) for k in ("A", "R", "len40", "len12", "taught")}
    out["R_last3_minus_2048_earlier"] = out["levels"]["R"] - mean_rows(curve, "R", last - 2048)
    out["final"] = {k: curve[-1][k] for k in ("A", "R", "taught", "len12", "len40")}
    out["digest_4096"] = next((r["data_digest"]["sha256"] for r in curve if r["step"] == 4096), None)
    out["cosine_start"], out["cosine_end"] = rec.get("cosine_first_batch_start"), rec.get("cosine_first_batch_end")
    out["cpu_updates"], out["cpu_scoring"] = rec["cpu_updates"], rec["cpu_scoring"]
    out["tensors_sha256"] = rec.get("tensors_sha256")
    if "kept_mean_magnitude" in curve[0]:
        out["magnitude_start"] = curve[0]["kept_mean_magnitude"]; out["magnitude_end"] = curve[-1]["kept_mean_magnitude"]
        out["masks_every_row"] = all(all(all(v.values()) for v in r["masks"].values()) for r in curve)
    out["limit_every_row"] = all(all(r["limit"].values()) for r in curve)
    return out


def describe(xr, xc, Dr, Dc):
    """E41's table of four cases (as in e42_tables.py), the reference against the candidate."""
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
    mr, mc = (1, 1) if unit == "updates" else (MA[sr["setting"]] * BATCH, MA[sc["setting"]] * BATCH)
    Dr = (sr["dose"] - (512 if ver == "three" else 0)) * mr
    Dc = (sc["dose"] - (512 if ver == "three" else 0)) * mc
    xr = cr["updates"] * mr if cr else None
    xc = cc["updates"] * mc if cc else None
    d = describe(xr, xc, Dr, Dc)
    return (d if unit == "updates" else MA_WORDS[d]), xr, xc


def level_desc(sr, sc, k):
    d = sc["levels"][k] - sr["levels"][k]
    return ("lower" if d <= -15 else "higher" if d >= 15 else "about the same"), sr["levels"][k], sc["levels"][k]


def load(out_dir):
    S = {}
    for p in sorted(Path(out_dir).glob(f"e49_*_*_{SEED}.json")):
        if "_dry" in p.name or p.name.endswith("_answers.json"):
            continue
        r = json.loads(p.read_text())
        if r.get("experiment") == "E49" and r.get("complete") and r["dose"] == DOSE:
            S[(r["setting"], r["rule"])] = summarize(r, r["setting"])
    for rule in "SB":
        p = Path(out_dir) / "reference_e42" / f"e42_F10_{rule}_{SEED}.json"
        r = json.loads(p.read_text())
        assert r["config"] == "F10" and r["complete"] and r["dose"] == DOSE
        S[("F10", rule)] = summarize(r, "F10")
    return S


def load_f10_seeds(d):
    out = {}
    if d is None:
        return out
    for p in sorted(Path(d).glob("e42_F10_*_*.json")):
        if "_dry" in p.name or p.name.endswith(("_structure.json", "_answers.json")):
            continue
        r = json.loads(p.read_text())
        if r.get("complete") and r.get("config") == "F10" and r["dose"] == DOSE:
            out[(r["rule"], r["seed"])] = summarize(r, "F10")
    return out


def fmt_u(c):
    return "not reached" if c is None else f"{c['updates']:,}"


def fmt_ma(c):
    return "not reached" if c is None else f"{c['multiply_adds']:.3g}"


def stage2_condition(S, rule):
    """Stage 2 under a rule if I16 does not reach 80 on R within the dose, or its updates to 80 on R are 'rises' (more) against F10, or it
    is 'lower' than F10 on a level. The three-row version."""
    if ("I16", rule) not in S:
        return None
    sr, sc = S[("F10", rule)], S[("I16", rule)]
    reasons = []
    if sc["thresholds"]["R80_three"] is None:
        reasons.append("does not reach 80 on R within the dose")
    d, _, _ = thr_desc(sr, sc, "R80", "three", "updates")
    if d == "more":
        reasons.append("updates to 80 on R: more (rises) against F10")
    for k in ("A", "R", "len40"):
        if level_desc(sr, sc, k)[0] == "lower":
            reasons.append(f"lower than F10 on the level of {k}")
    return dict(run_stage2=bool(reasons), reasons=reasons)


def write(out_dir, tag, f10_dir):
    S = load(out_dir)
    F = load_f10_seeds(f10_dir)
    od = Path(out_dir).resolve()
    od = od.relative_to(HERE) if od.is_relative_to(HERE) else od
    L = [f"Receipts read from `{od}` (E49) and `reference_e42/` (E42's `F10`, seed {SEED}, from 7fbdbdc2). "
         f"Complete E49 runs: {sorted(k for k in S if k[0] != 'F10')}.\n"]
    L.append("### Batch digest at row 4,096 (check 4) and the masks at every scored row (check 3)\n")
    L.append("| Setting | Rule | digest at 4,096 (first 16) | masks held at every row | E42's limit held at every row |\n|---|---|---|---|---|")
    for (s, r), v in sorted(S.items()):
        L.append(f"| {s} | {r} | {(v['digest_4096'] or '')[:16]} | {v.get('masks_every_row', '(E42 receipt)')} | {v['limit_every_row']} |")
    L.append("")
    L.append("### Per run: updates to the thresholds, three-row version [single-row in brackets]\n")
    L.append("| Setting | Rule | A50 | A80 | A90 | R50 | R80 | R90 | max A (row) | max R (row) | final A, R, taught | CPU-h updates + scoring |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = [("F10", r) for r in "SB"] + [(s, r) for s in SETTINGS for r in "SB"]
    for k in order:
        if k not in S:
            continue
        v = S[k]; t = v["thresholds"]
        cells = [f"{fmt_u(t[f'{key}{th}_three'])} [{fmt_u(t[f'{key}{th}_single'])}]" for key in "AR" for th in THRESHOLDS]
        L.append(f"| {k[0]} | {k[1]} | " + " | ".join(cells) + f" | {v['max_A']['value']:.2f} ({v['max_A']['row']:,}) | {v['max_R']['value']:.2f} "
                 f"({v['max_R']['row']:,}) | {v['final']['A']:.2f}, {v['final']['R']:.2f}, {v['final']['taught']:.2f} | "
                 f"{v['cpu_updates'] / 3600:.2f} + {v['cpu_scoring'] / 3600:.2f} |")
    L.append("")
    L.append("### Per run: multiply-adds to the thresholds (multiply-adds a position x 64 items x updates), three-row version [single-row]\n")
    L.append("| Setting | Rule | multiply-adds a position | A50 | A80 | A90 | R50 | R80 | R90 |\n|---|---|---|---|---|---|---|---|---|")
    for k in order:
        if k not in S:
            continue
        v = S[k]; t = v["thresholds"]
        cells = [f"{fmt_ma(t[f'{key}{th}_three'])} [{fmt_ma(t[f'{key}{th}_single'])}]" for key in "AR" for th in THRESHOLDS]
        L.append(f"| {k[0]} | {k[1]} | {MA[k[0]]:,} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### Per run: the three levels (mean of the last three scored rows), the length test at 12, and R at the end against 2,048 updates earlier\n")
    L.append("| Setting | Rule | A | R | length at 40 | length at 12 | taught | R: last three minus the three ending 2,048 earlier |\n|---|---|---|---|---|---|---|---|")
    for k in order:
        if k not in S:
            continue
        v = S[k]; lv = v["levels"]
        L.append(f"| {k[0]} | {k[1]} | {lv['A']:.2f} | {lv['R']:.2f} | {lv['len40']:.2f} | {lv['len12']:.2f} | {lv['taught']:.2f} | "
                 f"{v['R_last3_minus_2048_earlier']:+.2f} |")
    L.append("")
    L.append("### Per run: cosine between the sweep's and the exact gradient of each masked map (first batch; at the start and at the end), "
             "and the mean magnitude of the kept weights (start, end)\n")
    L.append("| Setting | Rule | input map: cosine start, end | lm: cosine start, end | recurrent (E42's): start, end | input map: magnitude start, end | lm: magnitude start, end |")
    L.append("|---|---|---|---|---|---|---|")
    for k in order:
        if k not in S or k[0] == "F10":
            continue
        v = S[k]; cs, ce = v["cosine_start"], v["cosine_end"]; ms, me = v["magnitude_start"], v["magnitude_end"]
        L.append(f"| {k[0]} | {k[1]} | {cs['rnn.weight_ih_l0']:.3f}, {ce['rnn.weight_ih_l0']:.3f} | {cs['lm.weight']:.3f}, {ce['lm.weight']:.3f} | "
                 f"{cs['rnn.weight_hh_l0']:.3f}, {ce['rnn.weight_hh_l0']:.3f} | {ms['rnn.weight_ih_l0']:.4f}, {me['rnn.weight_ih_l0']:.4f} | "
                 f"{ms['lm.weight']:.4f}, {me['lm.weight']:.4f} |")
    L.append("")
    L.append("### Descriptions against E42's `F10` of seed 73129 under the same rule (fixed beforehand). Three-row version; "
             "* where the single-row version describes the cell differently\n")
    L.append("| Setting | Rule | Measure | Description | F10 | Setting |\n|---|---|---|---|---|---|")
    desc = {}
    for s in SETTINGS:
        for rule in "SB":
            if (s, rule) not in S:
                continue
            sr, sc = S[("F10", rule)], S[(s, rule)]
            for key in ("A80", "R80"):
                for unit in ("updates", "multiply-adds"):
                    d3, xr, xc = thr_desc(sr, sc, key, "three", unit)
                    d1 = thr_desc(sr, sc, key, "single", unit)[0]
                    desc[f"{s} {rule} {key} {unit}"] = dict(three=d3, single=d1, reference=xr, setting=xc)
                    f = (lambda z: "not reached" if z is None else f"{z:,}") if unit == "updates" else (lambda z: "not reached" if z is None else f"{z:.3g}")
                    L.append(f"| {s} | {rule} | {key} {unit} | {d3}{'*' if d1 != d3 else ''} | {f(xr)} | {f(xc)} |")
            for k in ("A", "R", "len40"):
                d, a, b = level_desc(sr, sc, k)
                desc[f"{s} {rule} level {k}"] = dict(description=d, reference=a, setting=b)
                L.append(f"| {s} | {rule} | level of {k if k != 'len40' else 'the length test at 40'} | {d} | {a:.2f} | {b:.2f} |")
    L.append("")
    L.append("### The brief's condition for stage 2 (I16 under a rule)\n")
    st2 = {rule: stage2_condition(S, rule) for rule in "SB"}
    for rule in "SB":
        c = st2[rule]
        L.append(f"- {rule}: " + ("I16 not complete" if c is None else ("**stage 2 runs**: " + "; ".join(c["reasons"]) if c["run_stage2"] else "stage 2 does not run")))
    L.append("")
    L.append("### What a seed alone changes\n")
    if F:
        L.append("| F10 rule | seed | A80 | R80 | levels A, R, length at 40 |\n|---|---|---|---|---|")
        for (rule, sd), v in sorted(F.items()):
            L.append(f"| {rule} | {sd} | {fmt_u(v['thresholds']['A80_three'])} | {fmt_u(v['thresholds']['R80_three'])} | "
                     f"{v['levels']['A']:.2f}, {v['levels']['R']:.2f}, {v['levels']['len40']:.2f} |")
    else:
        L.append("E42's `F10` of seeds 93683 and 116089 is not on file complete. The nearest figure (the brief's) is `G` under the sweep, whose "
                 "three seeds reach 80 on R at 4,864 to 6,656 updates (a ratio of 1.37).")
    L.append("")
    out_md = Path(out_dir) / f"E49_TABLES_{tag}.md"
    out_md.write_text("\n".join(L) + "\n")
    js = dict(runs={f"{k[0]} {k[1]}": v for k, v in S.items()}, descriptions=desc, stage2=st2,
              f10_other_seeds={f"{k[0]} {k[1]}": v for k, v in F.items()})
    (Path(out_dir) / f"E49_SUMMARY_{tag}.json").write_text(json.dumps(js, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    write(Path(opt("--out", str(OUT))), opt("--tag", str(SEED)), opt("--f10-seeds", None))
