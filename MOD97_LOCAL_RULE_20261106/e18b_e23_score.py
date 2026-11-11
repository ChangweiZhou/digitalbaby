"""E18b / E18c / E23 scorer (11 Nov 2026), written before any full birth. Reads receipts only; writes exp45/E18B_E23_TABLES.md and
exp45/E18B_E23_VERDICTS.json. A cell whose receipt fails validation is listed and not scored. Definitions (plan E18B_E18C_E23_PLAN_20261111.md):
births: main head, whole answers (exact), k = 12; hard mean = (add89 + sub89) / 2, max/min mean = (max89 + min89) / 2, at the final checkpoint;
checkpoint mean = mean of the hard mean over the six checkpoints. Lives: k = 12, F1 = mean of add97, sub97, max97, min97; block i = stage i.
Verdicts need all three seeds validated; "in >= 2 seeds" counts seeds."""
import hashlib, json, os, sys
from pathlib import Path
OUT = Path(os.environ.get("E18B_SCORE_OUT", Path(__file__).resolve().parent / "exp45"))
DRY = os.environ.get("E18B_SCORE_DRY")              # tests of this scorer on dry runs only: "<dev updates>,<stage updates>"
DUP, SUP = (int(x) for x in DRY.split(",")) if DRY else (24576, 6144)
SEEDS = (73129, 93683, 116089)
DV = ("add89", "sub89", "max89", "min89"); F1 = ("add97", "sub97", "max97", "min97"); F3 = ("add73", "sub73", "max73", "min73")
pc = lambda d: 100 * d["exact"] / d["n"]
BIRTHS = {   # cell: (stem in exp45, expected receipt config)
    "B2": ("e17_dev_bank2_nbd_lh_bT_tall_e18b", dict(heads="local", traceup="all", loss="answer")),
    "B2-all": ("e17_dev_bank2_nbd_lh_bT_tall_lossall", dict(heads="local", traceup="all", loss="all")),
    "B3": ("e17_dev_bank2_nbd_lh_bT_e18b", dict(heads="local", traceup="none", loss="answer")),
    "B3-all": ("e17_dev_bank2_nbd_lh_bT_lossall", dict(heads="local", traceup="none", loss="all")),
    "R2": ("e17_dev_bank2_nbd_rh_bT_tall", dict(heads="relay", traceup="all", loss="answer")),
    "H0": ("e17_dev_bank2_nbd_lh_bT_tall_h1all", dict(heads="local", traceup="all", loss="answer", head1loss="all")),
    "N1": ("e17_dev_bank2_nbd_lh_bT_tall_h1all_fh1", dict(heads="local", traceup="all", loss="answer", head1loss="all", farhead=1, farctrl="none")),
    "F4": ("e17_dev_bank2_nbd_lh_bT_tall_h1all_fh4", dict(heads="local", traceup="all", loss="answer", head1loss="all", farhead=4, farctrl="none")),
    "F4s": ("e17_dev_bank2_nbd_lh_bT_tall_h1all_fh4s", dict(heads="local", traceup="all", loss="answer", head1loss="all", farhead=4, farctrl="shuffle")),
}
LIVES = {"L-B2": ("e17_life_T_bank2_nbd_lh_tall_bT_raw65536_e18c", "local", "B2"), "L-R2": ("e17_life_T_bank2_nbd_rh_tall_bT_raw65536_e18c", "relay", "R2")}
COMMON = dict(arch="bank2", bookdecay="off", birthrule="T", feedback="sign", projfb="raw", fixedcodes="off", decay=0.25, updates=DUP)
problems, B, L = [], {}, {}
dry_stem = lambda st: st.replace("_e18b", "") + f"_dry{DUP}" + ("_e18b" if st.endswith("_e18b") else "")


def k12(row, name, key="sealed"):
    d = row[key][name]
    return pc(d["12"] if "12" in d else d[12])


for cell, (stem, want) in BIRTHS.items():
    for s in SEEDS:
        sm = dry_stem(stem) if DRY else stem
        p = OUT / f"{sm}_{s}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text()); bad = []
        exp = dict(COMMON, **want)
        for k, v in exp.items():
            got = r.get(k, {"farhead": 0, "farctrl": "none", "head1loss": "answer"}.get(k))
            if got != v: bad.append(f"{k}={got!r} (want {v!r})")
        if not r.get("complete") or r["curve"][-1]["step"] != DUP or len(r["curve"]) != (6 if not DRY else 1): bad.append("not complete / final step")
        w = OUT / f"{sm}_{s}.pt"
        if not w.exists() or hashlib.sha256(w.read_bytes()).hexdigest() != r.get("weights_sha256"): bad.append("weights missing or hash differs")
        if bad:
            problems.append(f"{cell} {s}: {bad}"); continue
        c = r["curve"]; f = c[-1]
        hard = lambda row: (k12(row, "add89") + k12(row, "sub89")) / 2
        B[(cell, s)] = dict(hard=hard(f), maxmin=(k12(f, "max89") + k12(f, "min89")) / 2, ckpt_mean=sum(hard(x) for x in c) / len(c),
                            curve=[round(hard(x), 1) for x in c], cpu=f["cpu"], ops={n: k12(f, n) for n in DV},
                            head1_hard=((k12(f, "add89", "head1") + k12(f, "sub89", "head1")) / 2) if "head1" in f else None,
                            book_shift=f.get("book_shift_by_role"), census=r.get("credit_census_first_update", {}).get("by_role") if isinstance(r.get("credit_census_first_update"), dict) else None,
                            far=f.get("far_head_k12"), silenced=f.get("silenced_k12"))

for cell, (stem, heads, bcell) in LIVES.items():
    for s in SEEDS:
        p = OUT / f"{stem + (f'_dry{SUP}' if DRY else '')}_{s}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text()); st = d["stages"]; bad = []
        se = [x["settings"] for x in st if "settings" in x][-1]
        if [x["stage"] for x in st] != list(range(7)): bad.append("stage sequence")
        want = dict(arch="bank2", bookdecay="off", heads=heads, birthrule="T", traceup="all", feedback="sign", replay16="raw", cap=65536, outtag="_e18c", loss="answer", stage_updates=SUP, freeze="none")
        bad += [f"setting {k}={se.get(k)!r}" for k, v in want.items() if se.get(k) != v]
        # the life ran in the work folder under the birth's unsuffixed name: bind it by hash to the same-host birth filed in exp45
        bstem = dry_stem(BIRTHS[bcell][0]) if DRY else BIRTHS[bcell][0]
        bf = OUT / f"{bstem}_{s}.pt"
        if not bf.exists() or se.get("birth_sha256") != hashlib.sha256(bf.read_bytes()).hexdigest(): bad.append(f"birth hash differs from {bf.name}")
        cpl = d.get("complete") or {}
        w = OUT / cpl.get("weights", "missing")
        if cpl.get("stages") != 7 or not w.exists() or hashlib.sha256(w.read_bytes()).hexdigest() != cpl.get("weights_sha256"): bad.append("completion record or weight hash")
        if bad:
            problems.append(f"{cell} {s}: {bad}"); continue
        fam = lambda i, names: sum(pc(st[i]["scores"][n]["12"]) for n in names) / len(names)
        comp = {n: 100 * v["final"] / v["n"] for n, v in st[6]["scores"]["compose"].items()}
        L[(cell, s)] = dict(F1=[round(fam(i, F1), 1) for i in range(7)], F3_6=fam(6, F3), F1_6=fam(6, F1), F1_ret=(fam(6, F1) / fam(2, F1) if fam(2, F1) else None),
                            compose_final_6=comp, cpu=st[-1].get("cpu"))

ok = lambda cells, kind="b": all((c, s) in (B if kind == "b" else L) for c in cells for s in SEEDS)
count = lambda f: sum(1 for s in SEEDS if f(s))
V = {}
b = lambda c, s, k="hard": B[(c, s)][k]
if ok(("B3-all",)):
    shifts = lambda s: B[("B3-all", s)]["book_shift"] or {}
    V["P18b.1 B3-all operator and letter embeddings moved >= 0.2 row norm, all seeds"] = count(lambda s: all(shifts(s).get(k, -1) >= 0.2 for k in ("operator", "distractor letter"))) == 3
    V["P18b.2 B3-all max/min >= 66 and hard >= 10, >= 2 seeds"] = count(lambda s: b("B3-all", s, "maxmin") >= 66 and b("B3-all", s) >= 10) >= 2
if ok(("B3-all", "B2-all")):
    V["P18b.3 B3-all hard >= 20 below B2-all, all seeds"] = count(lambda s: b("B2-all", s) - b("B3-all", s) >= 20) == 3
if ok(("B2-all", "B2")):
    V["P18b.4 |B2-all - B2| hard <= 15, >= 2 seeds"] = count(lambda s: abs(b("B2-all", s) - b("B2", s)) <= 15) >= 2
if ok(("R2", "B2")):
    V["PC.1 R2 hard >= B2 + 5, >= 2 seeds"] = count(lambda s: b("R2", s) - b("B2", s) >= 5) >= 2
    V["PC.2 R2 checkpoint-mean hard >= B2 + 5, >= 2 seeds"] = count(lambda s: b("R2", s, "ckpt_mean") - b("B2", s, "ckpt_mean") >= 5) >= 2
    V["PC.3 R2 birth CPU-s within 15% of B2, all seeds"] = count(lambda s: abs(b("R2", s, "cpu") / b("B2", s, "cpu") - 1) <= 0.15) == 3
    V["PC.5 R2 first-head hard >= B2's + 5, >= 2 seeds"] = count(lambda s: b("R2", s, "head1_hard") - b("B2", s, "head1_hard") >= 5) >= 2
if ok(("L-B2", "L-R2"), "l"):
    V["PC.4 R2-life F1 after block 6 >= B2-life + 3, >= 2 seeds"] = count(lambda s: L[("L-R2", s)]["F1_6"] - L[("L-B2", s)]["F1_6"] >= 3) >= 2
if ok(("F4", "N1")):
    V["P23.1 F4 hard >= N1 + 5, >= 2 seeds"] = count(lambda s: b("F4", s) - b("N1", s) >= 5) >= 2
if ok(("F4", "F4s")):
    V["P23.2 F4 hard >= F4s + 5, >= 2 seeds"] = count(lambda s: b("F4", s) - b("F4s", s) >= 5) >= 2
if ok(("N1", "H0")):
    V["P23.3 N1 hard >= H0 - 5, >= 2 seeds"] = count(lambda s: b("N1", s) - b("H0", s) >= -5) >= 2
if ok(("H0", "B2")):
    V["P23.4 H0 hard >= B2 (same host) - 5, >= 2 seeds"] = count(lambda s: b("H0", s) - b("B2", s) >= -5) >= 2
if ok(("F4",)):
    def far_gap(s):
        fr = B[("F4", s)]["far"]; h = sum(v["prompt"][0] for v in fr.values()); n = sum(v["prompt"][1] for v in fr.values()); bs = sum(v["structural_baseline"][0] for v in fr.values())
        return 100 * (h - bs) / n
    V["P23.5 F4 far-head prompt top-1 >= structural baseline + 3, all seeds"] = count(lambda s: far_gap(s) >= 3) == 3

lines = ["# E18b / E18c / E23 tables (written by e18b_e23_score.py from receipts)", "", "## Births, final checkpoint, k = 12, main head", "",
         "| Cell | Seed | add89 | sub89 | max89 | min89 | Hard | Max/min | Ckpt-mean hard | Hard curve (6 ckpts) | First head hard | CPU-s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
for (c, s), r in B.items():
    o = r["ops"]; h1 = "—" if r["head1_hard"] is None else f"{r['head1_hard']:.1f}"
    lines.append(f"| {c} | {s} | {o['add89']:.1f} | {o['sub89']:.1f} | {o['max89']:.1f} | {o['min89']:.1f} | **{r['hard']:.1f}** | {r['maxmin']:.1f} | {r['ckpt_mean']:.1f} | {' / '.join(map(str, r['curve']))} | {h1} | {r['cpu']:,.0f} |")
lines += ["", "## Embedding movement by role at the final checkpoint", ""]
for (c, s), r in B.items():
    lines.append(f"- {c} {s}: {json.dumps(r['book_shift'])}; census {json.dumps(r['census'])}")
far_rows = [(c, s, r) for (c, s), r in B.items() if r["far"]]
if far_rows:
    lines += ["", "## Far head (diagnostic; teacher forced, prompt-byte targets, k = 12 panels pooled)", "", "| Cell | Seed | Top-1 prompt | Structural baseline | Top-1 digit targets |", "|---|---:|---:|---:|---:|"]
    for c, s, r in far_rows:
        fr = r["far"]; n = sum(v["prompt"][1] for v in fr.values())
        lines.append(f"| {c} | {s} | {100 * sum(v['prompt'][0] for v in fr.values()) / n:.1f} | {100 * sum(v['structural_baseline'][0] for v in fr.values()) / n:.1f} | {100 * sum(v['digit'][0] for v in fr.values()) / max(1, sum(v['digit'][1] for v in fr.values())):.1f} |")
if L:
    lines += ["", "## Lives (k = 12)", "", "| Cell | Seed | F1 by block 0-6 | F1 after 6 / after 2 | F3 after 6 | Compositions after 6 (final answer) | CPU-s |", "|---|---:|---|---:|---:|---|---:|"]
    for (c, s), r in L.items():
        lines.append(f"| {c} | {s} | {' / '.join(map(str, r['F1']))} | {r['F1_ret']:.2f} | {r['F3_6']:.1f} | {', '.join(f'{k} {v:.0f}' for k, v in r['compose_final_6'].items())} | {r['cpu']:,.0f} |")
lines += ["", "## Verdicts", ""] + [f"- {k}: **{'holds' if v else 'misses'}**" for k, v in V.items()] + ["", "## Validation problems", ""] + ([f"- {p}" for p in problems] or ["- none"])
(OUT / "E18B_E23_TABLES.md").write_text("\n".join(lines) + "\n")
(OUT / "E18B_E23_VERDICTS.json").write_text(json.dumps(dict(verdicts={k: ("holds" if v else "misses") for k, v in V.items()}, problems=problems,
                                                          births={f"{c} {s}": r for (c, s), r in B.items()}, lives={f"{c} {s}": r for (c, s), r in L.items()}), indent=1))
print("\n".join(lines[-(len(V) + 6):]))
sys.exit(0 if not problems else 1)
