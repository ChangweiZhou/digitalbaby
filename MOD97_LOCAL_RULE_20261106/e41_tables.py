"""E41 thresholds and descriptions from the receipts (brief OPUS_TASK_E41_..._20261116.md, revision 2, "What is recorded and tabled" and
"How the result is stated"). Reads exp45/e41/e41_<rule>_h<H>_f<fp>_<seed>.json (complete, full-dose receipts only); imports nothing from the
harness, so it can run anywhere with Python 3.

    python3 e41_tables.py [--out exp45/e41] [--seeds 73129,93683,116089] [--tag all]
        -> <out>/E41_TABLES_<tag>.md and <out>/E41_SUMMARY_<tag>.json

Three-row version: the first row of the first three consecutive scored rows all at or above the threshold (recordable up to dose - 512).
Single-row version: the first scored row at or above it. Descriptions use the three-row version; a cell is marked * where the single-row
version describes it differently (in the cell or in any seed).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDTHS = {256: 128, 512: 256, 1024: 512}
FRACS = {10: 0.10, 25: 0.25, 50: 0.50}
SEEDS = (73129, 93683, 116089)
THRESHOLDS = (50, 80, 90)


def first_row(curve, key, th, version):
    v = [(r["step"], r[key]) for r in curve]
    for i, (u, a) in enumerate(v):
        if version == "single" and a >= th:
            return i
        if version == "three" and i + 2 < len(v) and all(x >= th for _, x in v[i:i + 3]):
            return i
    return None


def summarize(rec):
    curve = rec["curve"]; out = dict(rule=rec["rule"], H=rec["H"], fraction=rec["fraction"], seed=rec["seed"], dose=rec["dose"],
                                     params=rec["model"]["params"], thresholds={})
    for key in ("A", "R"):
        for th in THRESHOLDS:
            for ver in ("single", "three"):
                i = first_row(curve, key, th, ver)
                r = curve[i] if i is not None else None
                out["thresholds"][f"{key}{th}_{ver}"] = None if r is None else dict(
                    updates=r["step"], weight_items=r["weight_items"], cpu_seconds=r["cpu_updates"], taught=r["taught"])
        best = max(curve, key=lambda r: r[key])
        out[f"max_{key}"] = dict(value=best[key], row=best["step"])
        lv = [r[key] for r in curve if r["step"] in (7680, 7936, 8192)]
        out[f"mean_{key}_7680_8192"] = sum(lv) / len(lv) if len(lv) == 3 else None
    out["digest_4096"] = next((r["data_digest"]["sha256"] for r in curve if r["step"] == 4096), None)
    out["final"] = dict(A=curve[-1]["A"], R=curve[-1]["R"], taught=curve[-1]["taught"])
    out["cpu_updates"], out["cpu_scoring"] = rec["cpu_updates"], rec["cpu_scoring"]
    out["processes_beside"] = rec.get("processes_beside")
    return out


def describe_step(xn, xw, Dn, Dw):
    if xn is not None and xw is not None:
        q = xw / xn
        return "falls" if q < 2 / 3 else "rises" if q > 1.5 else "stays"
    if xn is not None:
        return "rises" if Dw / xn >= 1.5 else "not decided"
    if xw is not None:
        return "falls" if xw / Dn <= 2 / 3 else "not decided"
    return "neither reaches"


def step_description(sn, sw, key, ver, unit):
    t = f"{key}_{ver}"
    cn, cw = sn["thresholds"][t], sw["thresholds"][t]
    mul_n, mul_w = (1, 1) if unit == "updates" else (sn["params"], sw["params"])
    Dn = (sn["dose"] - (512 if ver == "three" else 0)) * mul_n
    Dw = (sw["dose"] - (512 if ver == "three" else 0)) * mul_w
    xn = cn["updates"] * mul_n if cn else None
    xw = cw["updates"] * mul_w if cw else None
    return describe_step(xn, xw, Dn, Dw), xn, xw


def tables(out_dir):
    recs = {}
    for p in sorted(Path(out_dir).glob("e41_*_h*_f*_*.json")):
        if "_dry" in p.name:
            continue
        r = json.loads(p.read_text())
        if r.get("complete"):
            recs[(r["rule"], r["H"], round(r["fraction"] * 100), r["seed"])] = summarize(r)
    return recs


def fnum(v):
    return "" if v is None else f"{v:.2f}"


def fmt(c, unit):
    if c is None:
        return "not reached"
    return f"{c['updates']:,}" if unit == "updates" else f"{c['weight_items']:.3g}"


def write_tables(out_dir, seeds, md_path, json_path):
    S = tables(out_dir)
    lines = []
    have = sorted({k[3] for k in S})
    lines.append(f"Receipts read from `{out_dir}`; seeds present: {', '.join(map(str, have))}; runs: {len(S)}.\n")
    # data digests
    lines.append("### Batch digest at row 4,096 (check 2)\n")
    lines.append("| Seed | f | digests equal across the six runs | digest (first 16) |\n|---|---|---|---|")
    for seed in seeds:
        for fp in FRACS:
            ds = {S[k]["digest_4096"] for k in S if k[2] == fp and k[3] == seed}
            n = sum(1 for k in S if k[2] == fp and k[3] == seed)
            if n:
                lines.append(f"| {seed} | {fp / 100:.2f} | {'yes' if len(ds) == 1 else 'NO'} ({n} runs) | {', '.join(d[:16] for d in ds if d)} |")
    lines.append("")
    # per-run table
    for seed in seeds:
        lines.append(f"### Seed {seed}: updates and computation to threshold (three-row; single-row in brackets; * where they differ in description)\n")
        for key in ("A", "R"):
            lines.append(f"**Accuracy {key}**\n")
            lines.append("| Rule | f | H | " + " | ".join(f"{key}{th}: updates [single]" for th in THRESHOLDS)
                         + f" | {key}80 weight-items | {key}80 CPU-s | taught at {key}80 | max {key} (row) | mean {key} rows 7,680-8,192 |")
            lines.append("|" + "---|" * (3 + len(THRESHOLDS) + 5))
            for rule in "SB":
                for fp in FRACS:
                    for H in WIDTHS:
                        k = (rule, H, fp, seed)
                        if k not in S:
                            continue
                        s = S[k]; cells = []
                        for th in THRESHOLDS:
                            c3, c1 = s["thresholds"][f"{key}{th}_three"], s["thresholds"][f"{key}{th}_single"]
                            cells.append(f"{fmt(c3, 'updates')} [{fmt(c1, 'updates')}]")
                        c = s["thresholds"][f"{key}80_three"]
                        lines.append(f"| {rule} | {fp / 100:.2f} | {H} | " + " | ".join(cells)
                                     + f" | {fmt(c, 'wi')} | {'' if c is None else format(c['cpu_seconds'], '.0f')}"
                                     + f" | {'' if c is None else format(c['taught'], '.1f')}"
                                     + f" | {s['max_' + key]['value']:.2f} ({s['max_' + key]['row']:,}) | {fnum(s['mean_' + key + '_7680_8192'])} |")
            lines.append("")
    # descriptions per seed and three-seed cells
    for ths in ((80,), (50, 90)):
        lines.append(f"### Descriptions by width step, threshold{'s' if len(ths) > 1 else ''} {', '.join(map(str, ths))} (three-row version; "
                     "per-seed descriptions with the seeds' values; * where the single-row version describes the cell differently)\n")
        lines.append("| Threshold | Measure | Unit | Rule | f | Step | Cell | Per seed (description: narrower, wider) |\n|---|---|---|---|---|---|---|---|")
        for th in ths:
            for key in ("R", "A"):
                for unit in ("updates", "weight-items"):
                    for rule in "SB":
                        for fp in FRACS:
                            for hn, hw in ((256, 512), (512, 1024)):
                                per = []; per1 = []
                                for seed in seeds:
                                    kn, kw = (rule, hn, fp, seed), (rule, hw, fp, seed)
                                    if kn in S and kw in S:
                                        d, xn, xw = step_description(S[kn], S[kw], f"{key}{th}", "three", unit)
                                        d1, _, _ = step_description(S[kn], S[kw], f"{key}{th}", "single", unit)
                                        f_ = (lambda v: "nr" if v is None else (f"{v:,}" if unit == "updates" else f"{v:.3g}"))
                                        per.append((seed, d, f_(xn), f_(xw))); per1.append(d1)
                                if not per:
                                    continue
                                ds = [p[1] for p in per]
                                cell = next((d for d in set(ds) if ds.count(d) >= 2), "mixed") if len(per) == 3 else "(seeds missing)"
                                ds1 = per1
                                cell1 = next((d for d in set(ds1) if ds1.count(d) >= 2), "mixed") if len(per1) == 3 else None
                                mark = "*" if (len(per) == 3 and cell1 != cell) or any(a != b for a, b in zip(ds, ds1)) else ""
                                lines.append(f"| {th} | {key} | {unit} | {rule} | {fp / 100:.2f} | {hn}→{hw} | {cell}{mark} | "
                                             + "; ".join(f"{s_}: {d} ({a}, {b})" for s_, d, a, b in per) + " |")
        lines.append("")
    Path(md_path).write_text("\n".join(lines) + "\n")
    Path(json_path).write_text(json.dumps({f"{k[0]} h{k[1]} f{k[2]} {k[3]}": v for k, v in sorted(S.items())}, indent=1))
    return S




def main(argv):
    a = argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    out = Path(opt("--out", str(HERE / "exp45" / "e41")))
    seeds = [int(s) for s in opt("--seeds", ",".join(map(str, SEEDS))).split(",")]
    tag = opt("--tag", "all")
    write_tables(out, seeds, out / f"E41_TABLES_{tag}.md", out / f"E41_SUMMARY_{tag}.json")


if __name__ == "__main__":
    main(sys.argv)
