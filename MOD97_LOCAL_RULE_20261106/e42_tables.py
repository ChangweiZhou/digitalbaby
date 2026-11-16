"""E42 tables (brief OPUS_TASK_E42_CELLS_WITH_LIMITED_LOCAL_CONNECTIONS_AND_PERIODS_20261116.md, revision 2): thresholds, the fixed
descriptions and the structure tables, computed from the receipts and structure files only. Kept apart from exp42_cells.py so that the runner
stays at its commit during the runs.

    python3 e42_tables.py [--out DIR] [--seeds 73129,93683,116089] [--tag NAME]

Writes DIR/E42_TABLES_<tag>.md and DIR/E42_SUMMARY_<tag>.json. No reading beyond the brief's description.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e42"
SEEDS = (73129, 93683, 116089)
DOSE, ROW = 24576, 256
THRESHOLDS = (50, 80, 90)
STAGE1 = ("G", "F10", "A10", "P8")
STAGE2 = ("N5", "N20", "W10", "P4", "P16", "P8-last", "P8-source")
CONFIG_ORDER = STAGE1 + STAGE2
COMPARISONS = [("F10", "G"), ("A10", "G"), ("P8", "G"), ("A10", "F10"), ("P8", "A10"),
               ("N5", "A10"), ("N20", "A10"), ("W10", "A10"), ("P4", "P8"), ("P16", "P8"), ("P8-last", "P8"), ("P8-source", "P8")]
E26_TENSORS = {("S", 73129): "d906412631d18946560ce5cd17eeec29634e11263f0d82416cac44c8bd7af053",
               ("B", 73129): "efbdf87a6e1e2aeb8298a39f0c8968a331966e7e65a1286c2ec45f70ac441144"}


def first_row(curve, key, th, version):
    v = [(r["step"], r[key]) for r in curve]
    for i, (u, a) in enumerate(v):
        if version == "single" and a >= th:
            return i
        if version == "three" and i + 2 < len(v) and all(x >= th for _, x in v[i:i + 3]):
            return i
    return None


def summarize(rec):
    curve = rec["curve"]
    out = dict(config=rec["config"], rule=rec["rule"], seed=rec["seed"], dose=rec["dose"], weight_count=rec["weight_count"], thresholds={},
               theta=rec.get("theta"))
    for key in ("A", "R"):
        for th in THRESHOLDS:
            for ver in ("single", "three"):
                i = first_row(curve, key, th, ver)
                r = curve[i] if i is not None else None
                out["thresholds"][f"{key}{th}_{ver}"] = None if r is None else dict(
                    updates=r["step"], weight_items=r["weight_items"], cpu_seconds=r["cpu_updates"], taught=r["taught"])
        best = max(curve, key=lambda r: r[key])
        out[f"max_{key}"] = dict(value=best[key], row=best["step"])
    last3 = curve[-3:]
    out["len40_last3"] = sum(r["len40"] for r in last3) / 3
    out["len12_last3"] = sum(r["len12"] for r in last3) / 3
    out["final"] = {k: curve[-1][k] for k in ("A", "R", "taught", "len12", "len40")}
    out["replaced_total"] = sum(r["replaced_since_previous_row"] for r in curve)
    out["digest_4096"] = next((r["data_digest"]["sha256"] for r in curve if r["step"] == 4096), None)
    out["cosine_start"], out["cosine_end"] = rec.get("cosine_first_batch_start"), rec.get("cosine_first_batch_end")
    out["cpu_updates"], out["cpu_scoring"] = rec["cpu_updates"], rec["cpu_scoring"]
    out["tensors_sha256"] = rec["tensors_sha256"]
    out["limit_every_row"] = all(all(r["limit"].values()) for r in curve)
    return out


def describe(xr, xc, Dr, Dc):
    """E41's table of four cases, reference (narrower) against candidate (wider); falls = fewer, rises = more, stays = about the same."""
    if xr is not None and xc is not None:
        q = xc / xr
        return "fewer" if q < 2 / 3 else "more" if q > 1.5 else "about the same"
    if xr is not None:
        return "more" if Dc / xr >= 1.5 else "not decided"
    if xc is not None:
        return "fewer" if xc / Dr <= 2 / 3 else "not decided"
    return "neither reaches"


def thr_desc(sr, sc, key, ver, unit):
    t = f"{key}_{ver}"
    cr, cc = sr["thresholds"][t], sc["thresholds"][t]
    mr, mc = (1, 1) if unit == "updates" else (sr["weight_count"] * 64, sc["weight_count"] * 64)
    Dr = (sr["dose"] - (512 if ver == "three" else 0)) * mr
    Dc = (sc["dose"] - (512 if ver == "three" else 0)) * mc
    xr = cr["updates"] * mr if cr else None
    xc = cc["updates"] * mc if cc else None
    return describe(xr, xc, Dr, Dc), xr, xc


def len_desc(sr, sc):
    d = sc["len40_last3"] - sr["len40_last3"]
    return ("lower" if d <= -15 else "higher" if d >= 15 else "about the same"), sr["len40_last3"], sc["len40_last3"]


def cell(ds):
    if len(ds) < 3:
        return "(seeds missing)"
    return next((d for d in dict.fromkeys(ds) if ds.count(d) >= 2), "mixed")


def load(out_dir):
    S = {}
    for p in sorted(Path(out_dir).glob("e42_*_*_*.json")):
        if "_dry" in p.name or p.name.endswith("_structure.json") or p.name.endswith("_answers.json"):
            continue
        r = json.loads(p.read_text())
        if r.get("experiment") == "E42" and r.get("complete"):
            S[(r["config"], r["rule"], r["seed"])] = summarize(r)
    return S


def structure_tables(out_dir, seeds):
    rows = []
    for p in sorted(Path(out_dir).glob("e42_*_structure.json")):
        if "_dry" in p.name:
            continue
        d = json.loads(p.read_text())
        if d["seed"] not in seeds:
            continue
        st, en = d["start"], d["end"]
        per = st["periods"]
        n_conn = sum(len(c) for c in en["connections"])
        never = sum(1 for i, cs in enumerate(en["connections"]) for j, mk in zip(cs, en["made"][i]) if mk == 0 and j in st["connections"][i])
        pairs = {}
        for tag, s_ in (("start", st), ("end", en)):
            cnt = {}
            for i, cs in enumerate(s_["connections"]):
                for j in cs:
                    cnt[(per[i], per[j])] = cnt.get((per[i], per[j]), 0) + 1
            pairs[tag] = {f"{a}<-{b}": v / n_conn for (a, b), v in sorted(cnt.items())}
        rows.append(dict(config=d["config"], rule=d["rule"], seed=d["seed"], connections=n_conn, never_replaced_share=never / n_conn,
                         by_period=pairs))
    return rows


def fmt_u(c):
    return "not reached" if c is None else f"{c['updates']:,}"


def write(out_dir, seeds, tag):
    S = load(out_dir)
    L = [f"Receipts read from `{Path(out_dir).resolve().relative_to(HERE)}`; seeds requested {', '.join(map(str, seeds))}; complete runs {len(S)}.\n"]
    L.append("### Batch digest at row 4,096 (check 2) and the limit at every scored row (check 4)\n")
    L.append("| Seed | runs | digests equal | digest (first 16) | limit held at every row |\n|---|---|---|---|---|")
    for sd in seeds:
        ks = [k for k in S if k[2] == sd]
        if ks:
            ds = {S[k]["digest_4096"] for k in ks}
            L.append(f"| {sd} | {len(ks)} | {'yes' if len(ds) == 1 else 'NO'} | {', '.join(d[:16] for d in ds if d)} | "
                     f"{'yes' if all(S[k]['limit_every_row'] for k in ks) else 'NO'} |")
    L.append("")
    L.append("### Check 11: G's final tensors_sha256 beside E26's receipt (reported, not a failure)\n")
    L.append("| Rule | Seed | G | E26 | equal |\n|---|---|---|---|---|")
    for sd in seeds:
        for rule in "SB":
            k = ("G", rule, sd)
            if k in S:
                e26 = E26_TENSORS.get((rule, sd))
                if e26 is None:
                    p = HERE / "exp45" / "e26" / f"e26_dev_{rule}_{sd}.json"
                    e26 = json.loads(p.read_text())["tensors_sha256"] if p.exists() else None
                L.append(f"| {rule} | {sd} | {S[k]['tensors_sha256'][:16]} | {(e26 or '')[:16]} | {S[k]['tensors_sha256'] == e26} |")
    L.append("")
    for sd in seeds:
        L.append(f"### Seed {sd}: per run (three-row version; single-row in brackets)\n")
        L.append("| Config | Rule | " + " | ".join(f"A{t}" for t in THRESHOLDS) + " | " + " | ".join(f"R{t}" for t in THRESHOLDS)
                 + " | A80 weight-items | max A (row) | max R (row) | final A, R, taught | len 12, 40 (last three rows) | replaced | cosine start, end | CPU-h updates + scoring |")
        L.append("|" + "---|" * (2 + 2 * len(THRESHOLDS) + 8))
        for cfg in CONFIG_ORDER:
            for rule in "SB":
                k = (cfg, rule, sd)
                if k not in S:
                    continue
                s = S[k]; cells = []
                for key in ("A", "R"):
                    for th in THRESHOLDS:
                        cells.append(f"{fmt_u(s['thresholds'][f'{key}{th}_three'])} [{fmt_u(s['thresholds'][f'{key}{th}_single'])}]")
                c80 = s["thresholds"]["A80_three"]
                cs, ce = s["cosine_start"] or {}, s["cosine_end"] or {}
                L.append(f"| {cfg} | {rule} | " + " | ".join(cells)
                         + f" | {'' if c80 is None else format(c80['weight_items'], '.3g')}"
                         + f" | {s['max_A']['value']:.2f} ({s['max_A']['row']:,}) | {s['max_R']['value']:.2f} ({s['max_R']['row']:,})"
                         + f" | {s['final']['A']:.2f}, {s['final']['R']:.2f}, {s['final']['taught']:.2f}"
                         + f" | {s['len12_last3']:.2f}, {s['len40_last3']:.2f} | {s['replaced_total']:,}"
                         + f" | {cs.get('recurrent_matrix', float('nan')):.3f}, {ce.get('recurrent_matrix', float('nan')):.3f}"
                         + f" | {s['cpu_updates'] / 3600:.2f} + {s['cpu_scoring'] / 3600:.2f} |")
        L.append("")
    # descriptions
    L.append("### Descriptions (fixed beforehand). Updates and weight-items to the 80 threshold, three-row version (* where the single-row "
             "version describes the cell differently); the length test at 40 letters, mean of the last three rows\n")
    L.append("| Candidate | Reference | Rule | Measure | Cell | Per seed: description (reference, candidate) |\n|---|---|---|---|---|---|")
    comps = [(c, r, rule) for c, r in COMPARISONS for rule in ("SB" if c in STAGE1 and r in STAGE1 else "S")]
    comps += [(f"{c}", f"{c}", "S vs B") for c in STAGE1]
    for cand, ref, rule in comps:
        for meas in ("A80 updates", "A80 weight-items", "R80 updates", "R80 weight-items", "length at 40"):
            per, per1 = [], []
            for sd in seeds:
                if rule == "S vs B":
                    kc, kr = (cand, "S", sd), (ref, "B", sd)
                else:
                    kc, kr = (cand, rule, sd), (ref, rule, sd)
                if kc not in S or kr not in S:
                    continue
                if meas == "length at 40":
                    d, a, b = len_desc(S[kr], S[kc]); per.append((sd, d, f"{a:.2f}", f"{b:.2f}")); per1.append(d)
                else:
                    key, unit = meas.split()[0], ("updates" if "updates" in meas else "weight-items")
                    d, a, b = thr_desc(S[kr], S[kc], key, "three", unit)
                    d1, _, _ = thr_desc(S[kr], S[kc], key, "single", unit)
                    f_ = (lambda v: "nr" if v is None else (f"{v:,}" if unit == "updates" else f"{v:.3g}"))
                    per.append((sd, d, f_(a), f_(b))); per1.append(d1)
            if not per:
                continue
            ds = [p[1] for p in per]
            c3 = cell(ds); c1 = cell(per1)
            mark = "*" if (meas != "length at 40" and (c1 != c3 or any(a != b for a, b in zip(ds, per1)))) else ""
            L.append(f"| {cand} | {ref} | {rule} | {meas} | {c3}{mark} | " + "; ".join(f"{s_}: {d} ({a}, {b})" for s_, d, a, b in per) + " |")
    L.append("")
    # structure
    st = structure_tables(out_dir, seeds)
    if st:
        L.append("### Structure: share of connections never replaced; share of connections by the reader's and the source's period (reader<-source)\n")
        L.append("| Config | Rule | Seed | never replaced | by period at the start | by period at the end |\n|---|---|---|---|---|---|")
        for r in st:
            f2 = lambda d: ", ".join(f"{k} {v:.3f}" for k, v in d.items())
            L.append(f"| {r['config']} | {r['rule']} | {r['seed']} | {r['never_replaced_share']:.3f} | {f2(r['by_period']['start'])} | {f2(r['by_period']['end'])} |")
        L.append("")
    Path(out_dir, f"E42_TABLES_{tag}.md").write_text("\n".join(L) + "\n")
    Path(out_dir, f"E42_SUMMARY_{tag}.json").write_text(json.dumps(dict(runs={f"{k[0]} {k[1]} {k[2]}": v for k, v in sorted(S.items())},
                                                                        structure=st), indent=1))
    return S


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda n, d: (a[a.index(n) + 1] if n in a else d)
    write(Path(opt("--out", str(OUT))), [int(s) for s in opt("--seeds", ",".join(map(str, SEEDS))).split(",")], opt("--tag", "all"))
