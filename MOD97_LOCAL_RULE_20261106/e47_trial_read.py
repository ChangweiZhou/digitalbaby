"""E47 trial tables (brief OPUS_TASK_E47_LOCALITY_AT_THE_INPUT_AND_THE_OUTPUT_20261116.md, revision 2), from the receipts only: thresholds
(E42's definitions, copied), E41's table of four cases, the length test, the use of the far cells by the brief's fixed forms, the cosines, and
the predictions P47.1 to P47.7, P47.10 and P47.11. Kept apart from exp47_local.py so that the runner stays at its commit during the runs.

    python3 e47_trial_read.py            writes exp45/e47/E47_TRIAL_TABLES.md and exp45/e47/E47_TRIAL_SUMMARY.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e47"
QUAL = HERE / "exp45" / "e47_qual"
REF = QUAL / "e42_reference"
SEED = 73129
DOSE = 24576
THRESHOLDS = (50, 80, 90)
RUNS = [("one", "S"), ("one", "B"), ("place", "S"), ("place", "B")]
MACS = {"F10": 930560, "G": 1118720}     # multiply-adds a position (E47 brief's cost table); E42's P8 has none stated there


def first_row(curve, key, th, version):                              # E42's e42_tables.first_row
    v = [(r["step"], r[key]) for r in curve]
    for i, (u, a) in enumerate(v):
        if version == "single" and a >= th:
            return i
        if version == "three" and i + 2 < len(v) and all(x >= th for _, x in v[i:i + 3]):
            return i
    return None


def describe(xr, xc, Dr, Dc):                                         # E42's e42_tables.describe (E41's table of four cases)
    if xr is not None and xc is not None:
        q = xc / xr
        return "fewer" if q < 2 / 3 else "more" if q > 1.5 else "about the same"
    if xr is not None:
        return "more" if Dc / xr >= 1.5 else "not decided"
    if xc is not None:
        return "fewer" if xc / Dr <= 2 / 3 else "not decided"
    return "neither reaches"


def thresholds(curve):
    out = {}
    for key in ("A", "R"):
        for th in THRESHOLDS:
            for ver in ("single", "three"):
                i = first_row(curve, key, th, ver)
                out[f"{key}{th}_{ver}"] = None if i is None else curve[i]["step"]
    return out


def use_form(fall_off_A, fall_off_R, fall_mean_A, fall_mean_R):
    """The brief's forms, the first that applies."""
    if fall_mean_A >= 15 or fall_mean_R >= 15:
        return "used"
    if fall_off_A < 5 and fall_off_R < 5:
        return "not used"
    if (fall_off_A >= 15 or fall_off_R >= 15) and fall_mean_A < 5 and fall_mean_R < 5:
        return "a constant is used"
    return "partly"


def summarize(rec, macs):
    c = rec["curve"]
    s = dict(dose=rec["dose"], macs=macs, thresholds=thresholds(c), final={k: c[-1][k] for k in ("A", "R", "taught", "len12", "len40")},
             len40_last3=sum(r["len40"] for r in c[-3:]) / 3, len12_last3=sum(r["len12"] for r in c[-3:]) / 3,
             max_R=max(r["R"] for r in c), max_A=max(r["A"] for r in c),
             cpu_updates=rec.get("cpu_updates"), cpu_scoring=rec.get("cpu_scoring"),
             digest_4096=next((r["data_digest"]["sha256"] for r in c if r["step"] == 4096), None),
             cosine_start=rec.get("cosine_first_batch_start"), cosine_end=rec.get("cosine_first_batch_end"),
             tensors_sha256=rec.get("tensors_sha256"), complete=rec.get("complete"))
    last = c[-1]
    if last.get("far_off"):
        fo = {k: last[k] - last["far_off"][k] for k in ("A", "R", "len12", "len40")}
        fm = {k: last[k] - last["far_mean"][k] for k in ("A", "R", "len12", "len40")}
        s["fall_far_off"], s["fall_far_mean"] = fo, fm
        s["use"] = use_form(fo["A"], fo["R"], fm["A"], fm["R"])
        s["diag_final"] = dict(off={k: last["far_off"][k] for k in ("A", "R", "len12", "len40")},
                               mean={k: last["far_mean"][k] for k in ("A", "R", "len12", "len40")})
        s["magnitudes_start"], s["magnitudes_end"] = c[0]["magnitudes"], last["magnitudes"]
        s["cpu_per_update"] = rec.get("cpu_per_update")
        s["cpu_per_row"] = (rec["cpu_scoring"] / len(c)) if rec.get("cpu_scoring") else None
    return s


def compare(sr, sc, key="R80_three"):
    xr, xc = sr["thresholds"][key], sc["thresholds"][key]
    D = DOSE - 512
    upd = describe(xr, xc, D, D)
    ma = None
    if sr["macs"] and sc["macs"]:
        mr, mc = sr["macs"], sc["macs"]
        ma = describe(None if xr is None else xr * mr, None if xc is None else xc * mc, D * mr, D * mc)
    d40 = sc["len40_last3"] - sr["len40_last3"]
    return dict(reference=xr, candidate=xc, updates=upd, multiply_adds=ma, len40_diff=d40,
                length_test="lower" if d40 <= -15 else "higher" if d40 >= 15 else "within 15 points")


def main():
    runs = {}
    for arm, rule in RUNS:
        p = OUT / f"e47_32_{arm}_{rule}_{SEED}.json"
        if p.exists():
            rec = json.loads(p.read_text())
            runs[f"{arm} {rule}"] = summarize(rec, rec["multiply_adds_with_periods"])
    refs = {}
    for cfg in ("F10", "P8", "G"):
        for rule in "SB":
            p = REF / f"e42_{cfg}_{rule}_{SEED}.json"
            if p.exists():
                refs[f"{cfg} {rule}"] = summarize(json.loads(p.read_text()), MACS.get(cfg))
    comps = {}
    for name, s in runs.items():
        arm, rule = name.split()
        for ref in (["F10", "G"] + (["P8"] if arm != "one" else [])):
            if f"{ref} {rule}" in refs:
                for key in ("A80_three", "R80_three"):
                    comps[f"{name} vs E42 {ref} {rule}: {key}"] = compare(refs[f"{ref} {rule}"], s, key)
    for rule in "SB":
        if f"one {rule}" in runs and f"place {rule}" in runs:
            for key in ("A80_three", "R80_three"):
                comps[f"place {rule} vs one {rule}: {key}"] = compare(runs[f"one {rule}"], runs[f"place {rule}"], key)
    # predictions
    pred = {}
    c1 = json.loads((QUAL / "check1.json").read_text()) if (QUAL / "check1.json").exists() else None
    c2 = json.loads((QUAL / "check2.json").read_text()) if (QUAL / "check2.json").exists() else None
    if c1 and c2:
        pred["P47.1"] = dict(text="Stage 0, checks 1 and 2 pass under both rules", outcome=bool(c1["pass"] and c2["pass"]))
    g = lambda n: runs.get(n)
    if g("place B"):
        pred["P47.2"] = dict(text="B place reaches 80 on R (three rows)", outcome=g("place B")["thresholds"]["R80_three"] is not None,
                             single=g("place B")["thresholds"]["R80_single"])
    if g("place S"):
        pred["P47.3"] = dict(text="S place reaches 80 on R (three rows)", outcome=g("place S")["thresholds"]["R80_three"] is not None,
                             single=g("place S")["thresholds"]["R80_single"])
    if g("one B"):
        pred["P47.4"] = dict(text="B one: far cells used", outcome=g("one B")["use"] == "used", form=g("one B")["use"])
    if g("one S"):
        pred["P47.5"] = dict(text="S one: far cells used", outcome=g("one S")["use"] == "used", form=g("one S")["use"])
    if len(runs) == 4:
        pred["P47.10"] = dict(text="in at least one run: a constant is used", outcome=any(s["use"] == "a constant is used" for s in runs.values()),
                              forms={k: s["use"] for k, s in runs.items()})
        k = "place B vs one B: R80_three"
        pred["P47.11"] = dict(text="B: place reaches 80 on R in about the same updates as one", outcome=comps[k]["updates"] == "about the same",
                              description=comps[k])
    for n in ("place S", "place B"):
        if g(n) and g(n)["cosine_end"]:
            ce = g(n)["cosine_end"]
            pred.setdefault("P47.6", dict(text="place: cosine at the end lower for far cells than for interface cells by 0.1 or more", runs={}))
            # the cosine is between the sweep's and the exact gradient on the same weights, so each place run gives one reading
            pred["P47.6"]["runs"][n] = dict(interface=ce["interface"], far=ce["far"], difference=ce["interface"] - ce["far"],
                                            holds=ce["interface"] - ce["far"] >= 0.1)
    if "P47.6" in pred:
        pred["P47.6"]["outcome"] = all(v["holds"] for v in pred["P47.6"]["runs"].values())
    if g("one B") and "F10 B" in refs:
        k = "one B vs E42 F10 B: R80_three"
        pred["P47.7"] = dict(text="B one: updates to 80 on R about the same as or more than E42's F10 (not fewer)",
                             outcome=comps[k]["updates"] in ("about the same", "more"), description=comps[k])
    summary = dict(runs=runs, e42_references=refs, comparisons=comps, predictions=pred)
    (OUT / "E47_TRIAL_SUMMARY.json").write_text(json.dumps(summary, indent=1))
    L = ["# E47 trial tables (computed from the receipts by e47_trial_read.py)", ""]
    L.append("| Run | A80 (3 rows) | R50 | R80 (3 rows) | R90 | final A | final R | len12 | len40 (last 3) | use of far cells | CPU s/update |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for n, s in list(runs.items()) + [(f"E42 {k}", v) for k, v in refs.items()]:
        t = s["thresholds"]
        L.append(f"| {n} | {t['A80_three']} | {t['R50_three']} | {t['R80_three']} | {t['R90_three']} | {s['final']['A']:.1f} | {s['final']['R']:.1f} | "
                 f"{s['final']['len12']:.1f} | {s['len40_last3']:.1f} | {s.get('use', '—')} | "
                 f"{(s['cpu_per_update'] if s.get('cpu_per_update') else (s['cpu_updates'] or 0) / DOSE):.3f} |")
    L += ["", "## Scores at the end, plain and with the far cells held", "",
          "| Run | A plain / zero / mean | R plain / zero / mean | len12 plain / zero / mean | len40 plain / zero / mean | form |", "|---|---|---|---|---|---|"]
    for n, s in runs.items():
        f, d = s["final"], s["diag_final"]
        L.append(f"| {n} | " + " | ".join(f"{f[k]:.1f} / {d['off'][k]:.1f} / {d['mean'][k]:.1f}" for k in ("A", "R", "len12", "len40")) + f" | {s['use']} |")
    L += ["", "## Comparisons (E41's four cases on updates; lengths: mean of the last three rows at 40 letters)", "",
          "| Comparison | reference | candidate | updates | multiply-adds | len40 difference |", "|---|---|---|---|---|---|"]
    for k, c in comps.items():
        L.append(f"| {k} | {c['reference']} | {c['candidate']} | {c['updates']} | {c['multiply_adds'] or '—'} | {c['len40_diff']:+.1f} ({c['length_test']}) |")
    L += ["", "## Cosines between the sweep's and the exact gradient of the recurrent weights (first batch)", "",
          "| Run | start: interface / far / all | end: interface / far / all | far non-zero share start B / S | end B / S |", "|---|---|---|---|---|"]
    for n, s in runs.items():
        a, b = s["cosine_start"], s["cosine_end"]
        L.append(f"| {n} | {a['interface']:.3f} / {a['far']:.3f} / {a['all']:.3f} | {b['interface']:.3f} / {b['far']:.3f} / {b['all']:.3f} | "
                 f"{a['far_nonzero_share_B']:.3f} / {a['far_nonzero_share_S']:.3f} | {b['far_nonzero_share_B']:.3f} / {b['far_nonzero_share_S']:.3f} |")
    L += ["", "## Mean magnitude of the recurrent weights", "", "| Run | start interface / far / scale | end interface / far / scale |", "|---|---|---|"]
    for n, s in runs.items():
        a, b = s["magnitudes_start"], s["magnitudes_end"]
        L.append(f"| {n} | {a['interface']:.4f} / {a['far']:.4f} / {a['scale']:.4f} | {b['interface']:.4f} / {b['far']:.4f} / {b['scale']:.4f} |")
    L += ["", "## Predictions", ""]
    for k, v in pred.items():
        L.append(f"- **{k}** {v['text']}: **{'yes' if v['outcome'] else 'no'}**")
    (OUT / "E47_TRIAL_TABLES.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
