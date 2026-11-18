"""E46 tables, descriptions and the sentence per rule, from the receipts (brief OPUS_TASK_E46_..._20261116.md, revision 2, "How the result is
stated"). Reads exp45/e46/e46_<S|B>_wd<000|005|025>_<seed>.json (complete, full-dose receipts only). Python 3 only, no harness import.

    python3 e46_tables.py [--out exp45/e46]   -> <out>/E46_TABLES.md, <out>/E46_SUMMARY.json

Thresholds: E41's single-row and three-row versions (three-row carries the descriptions). Updates to 80: E41's table of four cases, ratios
below 2/3 falls, above 1.5 rises, otherwise stays; D = dose (single-row) or dose - 512 (three-row). Levels: mean over rows 24,064, 24,320 and
24,576; lower or higher at a difference of 15 points or more, otherwise about the same. A cell carries the description two of three seeds share,
otherwise mixed.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (73129, 93683, 116089)
DECAYS = {"000": 0.0, "005": 0.05, "025": 0.25}
THRESHOLDS = (50, 80, 90)
END = (24064, 24320, 24576)
MID = (11776, 12032, 12288)


def first_row(curve, key, th, version):
    v = [(r["step"], r[key]) for r in curve]
    for i, (u, a) in enumerate(v):
        if version == "single" and a >= th:
            return u
        if version == "three" and i + 2 < len(v) and all(x >= th for _, x in v[i:i + 3]):
            return u
    return None


def mean_at(curve, key, steps):
    v = [r[key] for r in curve if r["step"] in steps]
    return sum(v) / len(v) if len(v) == len(steps) else None


def summarize(r):
    c = r["curve"]
    s = dict(rule=r["rule"], decay=r["decay"], seed=r["seed"], dose=r["dose"], thresholds={})
    for key in ("A", "R"):
        for th in THRESHOLDS:
            for ver in ("single", "three"):
                s["thresholds"][f"{key}{th}_{ver}"] = first_row(c, key, th, ver)
    for key in ("A", "R", "len40", "len12", "taught"):
        s[f"end_{key}"] = mean_at(c, key, END)
        s[f"mid_{key}"] = mean_at(c, key, MID)
    s["len40_last8"] = sum(r["len40"] for r in c[-8:]) / 8
    s["norms_end"] = c[-1]["norms"]
    s["norms_start"] = c[0]["norms"]
    s["digest_4096"] = next(r["data_digest"]["sha256"] for r in c if r["step"] == 4096)
    s["groups_start_end"] = [g["weight_decay"] for g in r["optimizer_groups_at_start"]], [g["weight_decay"] for g in r["optimizer_groups_at_end"]]
    s["tensors_sha256"] = r["tensors_sha256"]
    s["cpu_updates"], s["cpu_scoring"], s["processes_beside"] = r["cpu_updates"], r["cpu_scoring"], r.get("processes_beside")
    return s


def thr_desc(xr, xn, Dr, Dn):
    """E41's four cases with the reference (decay 0.25) in the narrower run's place and the new decay in the wider run's place."""
    if xr is not None and xn is not None:
        q = xn / xr
        return "falls" if q < 2 / 3 else "rises" if q > 1.5 else "stays"
    if xr is not None:
        return "rises" if Dn / xr >= 1.5 else "not decided"
    if xn is not None:
        return "falls" if xn / Dr <= 2 / 3 else "not decided"
    return "neither reaches"


def lvl_desc(ref, new):
    d = new - ref
    return "lower" if d <= -15 else "higher" if d >= 15 else "about the same"


def cell(ds):
    return next((d for d in dict.fromkeys(ds) if ds.count(d) >= 2), "mixed") if len(ds) == 3 else "(seeds missing)"


def main(argv):
    a = argv[1:]
    out = Path(a[a.index("--out") + 1]) if "--out" in a else HERE / "exp45" / "e46"
    S = {}
    for p in sorted(out.glob("e46_*_wd*_*.json")):
        if "_dry" in p.name:
            continue
        r = json.loads(p.read_text())
        if r.get("complete"):
            S[(r["rule"], f"{round(r['decay'] * 100):03d}", r["seed"])] = summarize(r)
    L = [f"Receipts: `{out}`; complete runs: {len(S)} of 18.\n"]
    f = lambda v: "nr" if v is None else f"{v:,}"
    g = lambda v: "" if v is None else f"{v:.2f}"
    L.append("### Batch digest at row 4,096 (check 3)\n\n| Seed | digests equal across the six runs | digest |\n|---|---|---|")
    for s in SEEDS:
        ds = {S[k]["digest_4096"] for k in S if k[2] == s}
        L.append(f"| {s} | {'yes' if len(ds) == 1 else 'NO'} ({sum(1 for k in S if k[2] == s)} runs) | {', '.join(d[:16] for d in ds)} |")
    L.append("\n### Per run: updates to threshold (three-row [single-row]); levels; norms at the end\n")
    L.append("| Rule | Decay | Seed | " + " | ".join(f"{k}{t}" for k in "AR" for t in THRESHOLDS)
             + " | end A | end R | end len40 | len40 last 8 | mid A | mid R | mid len40 | end taught | recurrent norm (start → end) | groups' decay (start; end) |")
    L.append("|" + "---|" * 19)
    for rule in "SB":
        for wdt in ("025", "005", "000"):
            for s in SEEDS:
                k = (rule, wdt, s)
                if k not in S:
                    continue
                x = S[k]; t = x["thresholds"]
                L.append(f"| {rule} | {DECAYS[wdt]} | {s} | " + " | ".join(f"{f(t[f'{kk}{th}_three'])} [{f(t[f'{kk}{th}_single'])}]" for kk in "AR" for th in THRESHOLDS)
                         + f" | {g(x['end_A'])} | {g(x['end_R'])} | {g(x['end_len40'])} | {g(x['len40_last8'])} | {g(x['mid_A'])} | {g(x['mid_R'])} | {g(x['mid_len40'])}"
                         + f" | {g(x['end_taught'])} | {x['norms_start']['recurrent']:.2f} → {x['norms_end']['recurrent']:.2f} | {x['groups_start_end'][0]}; {x['groups_start_end'][1]} |")
    L.append("\n### Norms of the seven weight groups at row 24,576\n")
    grp = ("book", "l1", "input_map", "recurrent", "lm", "l2", "l3")
    L.append("| Rule | Decay | Seed | " + " | ".join(grp) + " |\n|" + "---|" * (3 + len(grp)))
    for rule in "SB":
        for wdt in ("025", "005", "000"):
            for s in SEEDS:
                if (rule, wdt, s) in S:
                    n = S[(rule, wdt, s)]["norms_end"]
                    L.append(f"| {rule} | {DECAYS[wdt]} | {s} | " + " | ".join(f"{n[q]:.2f}" for q in grp) + " |")
    # descriptions
    D = {}
    L.append("\n### Descriptions against decay 0.25 (per seed: description (reference, new); * where the single-row version differs)\n")
    L.append("| Rule | Decay | Measure | Cell | Per seed |\n|---|---|---|---|---|")
    for rule in "SB":
        for wdt in ("000", "005"):
            for meas in ("updates to 80 on R", "updates to 80 on A", "level R", "level A", "level len40"):
                per, per1 = [], []
                for s in SEEDS:
                    kr, kn = (rule, "025", s), (rule, wdt, s)
                    if kr not in S or kn not in S:
                        continue
                    xr, xn = S[kr], S[kn]
                    if meas.startswith("updates"):
                        key = meas[-1]
                        a3, b3 = xr["thresholds"][f"{key}80_three"], xn["thresholds"][f"{key}80_three"]
                        a1, b1 = xr["thresholds"][f"{key}80_single"], xn["thresholds"][f"{key}80_single"]
                        d3 = thr_desc(a3, b3, xr["dose"] - 512, xn["dose"] - 512); d1 = thr_desc(a1, b1, xr["dose"], xn["dose"])
                        per.append((s, d3, f(a3), f(b3))); per1.append(d1)
                    else:
                        key = meas.split()[1]
                        a_, b_ = xr[f"end_{key}"], xn[f"end_{key}"]
                        dd = lvl_desc(a_, b_); per.append((s, dd, g(a_), g(b_))); per1.append(dd)
                ds = [p[1] for p in per]; c3 = cell(ds); c1 = cell(per1)
                mark = "*" if (len(per) == 3 and c1 != c3) or any(x != y for x, y in zip(ds, per1)) else ""
                D[(rule, wdt, meas)] = c3
                L.append(f"| {rule} | {DECAYS[wdt]} | {meas} | {c3}{mark} | " + "; ".join(f"{s}: {d} ({x}, {y})" for s, d, x, y in per) + " |")
    # the sentence
    L.append("\n### The sentence per rule (decay 0 against 0.25; three cells: updates to 80 on R, level R, level at 40 letters)\n")
    sentences = {}
    for rule, name in (("S", "the sweep"), ("B", "backpropagation")):
        cells = {m: D.get((rule, "000", m)) for m in ("updates to 80 on R", "level R", "level len40")}
        if any(v in (None, "(seeds missing)") for v in cells.values()):
            sent = "not yet formed (seeds missing)"
        else:
            need = [m for m, v in cells.items() if v in ("rises", "lower")]
            if need:
                sent = "needs decay for: " + ", ".join(need)
            elif cells["updates to 80 on R"] in ("falls", "stays") and all(cells[m] in ("about the same", "higher") for m in ("level R", "level len40")):
                sent = "does not need decay"
            else:
                sent = "not decided"
        sentences[rule] = dict(sentence=sent, cells=cells)
        L.append(f"- **{name}:** learning the first family {sent}. Cells: " + "; ".join(f"{m}: {v}" for m, v in cells.items()) + ".")
    (out / "E46_TABLES.md").write_text("\n".join(L) + "\n")
    (out / "E46_SUMMARY.json").write_text(json.dumps(dict(runs={f"{k[0]} wd{k[1]} {k[2]}": v for k, v in sorted(S.items())},
                                                          cells={f"{k[0]} wd{k[1]} {k[2]}": v for k, v in D.items()}, sentences=sentences), indent=1))


if __name__ == "__main__":
    main(sys.argv)
