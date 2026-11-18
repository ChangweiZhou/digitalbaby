"""E54: the tables and the readings fixed in the brief (revision 3, "How the result is stated"), from the receipts under exp45/e54/.

    python3 e54_tables.py            # writes exp45/e54/E54_TABLES.json and exp45/e54/E54_TABLES.md from the complete receipts present

Definitions (the brief's). A scored row is the set of scores taken every 256 updates. Updates to 80 on a score: the update of the first of
three consecutive scored rows at or above 80. Level at an update: the mean of a score over the three scored rows that end at that update.
Last rise of R: the mean of R over the nine scored rows that end at the last update, less the mean over the nine rows that end 8,192
updates earlier; the rise before it: that second mean less the mean over the nine rows that end 16,384 updates before the last update.
A curve has flattened if both rises are smaller than 3 points in size; with a rise of minus 3 points or less it fell.
`describe` is E42's (e42_tables.py lines 63 to 72 on claude/gallant-dijkstra-i9zgsr), copied as text.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e54"
SEED = 73129
DOSE = 49152
CONFIGS = ("G", "F10", "P8")
FRACS = (25, 15, 10)
HALF = {("F10", "S", "A"): 2.36, ("F10", "B", "A"): 1.70, ("P8", "S", "A"): 1.36, ("P8", "B", "A"): 1.25,     # the brief, seed 73129, f = 0.50
        ("F10", "S", "R"): 2.27, ("F10", "B", "R"): 1.48, ("P8", "S", "R"): 1.27, ("P8", "B", "R"): 1.12}
MARGIN = {(25, "S"): 10, (25, "B"): 10, (10, "S"): 10, (10, "B"): 20, (15, "S"): 20, (15, "B"): 20}
MARGIN_SB = {25: 10, 15: 20, 10: 20}
SPREAD = {(10, "B"): "the dense network's level of R differs between the three seeds by 14.2 points at update 12,288 at the present width "
                     "and by 12.7 to 20.8 points at the half width (E41), and seed 73129 is its lowest seed",
          (15, "S"): "no spread between seeds is on record at this fraction", (15, "B"): "no spread between seeds is on record at this fraction"}


def describe(xr, xc, Dr, Dc):
    if xr is not None and xc is not None:
        q = xc / xr
        return "fewer" if q < 2 / 3 else "more" if q > 1.5 else "about the same"
    if xr is not None:
        return "more" if Dc / xr >= 1.5 else "not decided"
    if xc is not None:
        return "fewer" if xc / Dr <= 2 / 3 else "not decided"
    return "neither reaches"


def col(curve, key):
    if key in ("max", "min"):
        return {r["step"]: 100 * r["A_counts"][key + "89"] / 300 for r in curve}
    return {r["step"]: r[key] for r in curve}


def to80(curve, key):
    v = [(r["step"], r[key]) for r in curve]
    for i in range(len(v) - 2):
        if all(x >= 80 for _, x in v[i:i + 3]):
            return v[i][0]
    return None


def mean_rows(c, end, n):
    return sum(c[end - 256 * i] for i in range(n)) / n


def summary(rec):
    curve = rec["curve"]; end = rec["dose"]
    s = dict(receipt=rec["_name"], updates=end, tensors_sha256=rec["tensors_sha256"], to80={k: to80(curve, k) for k in ("A", "R")}, levels={})
    marks = (12288, 24576, 49152) + ((122880,) if end == 122880 else ())
    keys = ("A", "R", "taught") + (("len40", "max", "min") if "len40" in curve[0] else ())
    for k in keys:
        c = col(curve, k)
        s["levels"][k] = {str(u): mean_rows(c, u, 3) for u in marks if u <= end}
    c = col(curve, "R")
    m0, m1, m2 = mean_rows(c, end, 9), mean_rows(c, end - 8192, 9), mean_rows(c, end - 16384, 9)
    s["last_rise"], s["rise_before"] = m0 - m1, m1 - m2
    s["flattened"] = abs(s["last_rise"]) < 3 and abs(s["rise_before"]) < 3
    s["fell"] = s["last_rise"] <= -3 or s["rise_before"] <= -3
    s["end_level_R"] = mean_rows(c, end, 3)
    s["generalises_from_the_fraction"] = s["to80"]["R"] is not None
    s["cpu_hours"] = (rec["cpu_updates"] + rec["cpu_scoring"] + rec.get("cpu_replacement", 0.0)) / 3600
    if rec.get("changing"):
        ev = [r["replaced_since_previous_row"] for r in curve[1:]]
        total = 256 * rec["N"]
        s["replaced_share_first16"] = 100 * sum(ev[:16]) / (16 * total); s["replaced_share_last16"] = 100 * sum(ev[-16:]) / (16 * total)
        s["resumed_at"] = [x["at_update"] for x in rec["checkpointing"]["resumed"]]
    elif "checkpointing" in rec:
        s["resumed_at"] = [x["at_update"] for x in rec["checkpointing"]["resumed"]]
    return s


def end_reading(d, margin, low_flat_for_better, low_flat_for_worse, both_flat):
    if d >= margin and low_flat_for_better:
        return "generalises better"
    if d <= -margin and low_flat_for_worse:
        return "generalises worse"
    if abs(d) < margin and both_flat:
        return "not distinguished"
    return "not decided within the run"


def load():
    runs = {}
    for fp in FRACS:
        for cfg in CONFIGS:
            for rule in "SB":
                p = OUT / f"e54_{cfg}_{rule}_f{fp}_{SEED}.json"
                if p.exists():
                    r = json.loads(p.read_text())
                    if r["complete"]:
                        r["_name"] = p.name; runs[(fp, cfg, rule)] = r
    p4 = {}
    for rule in "SB":
        p = OUT / f"e41_{rule}_h256_f10_dry122880_{SEED}.json"
        if p.exists():
            r = json.loads(p.read_text())
            if r["complete"]:
                r["_name"] = p.name; p4[rule] = r
    return runs, p4


def fmt(x, d=1):
    return "—" if x is None else f"{x:,}" if isinstance(x, int) else f"{x:.{d}f}"


def flat_word(s):
    return "flattened" if s["flattened"] else "fell" if s["fell"] else "not flattened"


def main():
    runs, p4 = load()
    S = {k: summary(r) for k, r in runs.items()}
    out = dict(seed=SEED, runs={f"f{fp} {cfg} {rule}": s for (fp, cfg, rule), s in S.items()}, against_G={}, sweep_against_backprop={}, part4={},
               predictions={})
    md = ["# E54 tables (seed 73129)", "", "Written by `python3 e54_tables.py` from the receipts under `exp45/e54/`. Levels are means of three scored rows; "
          "rises are differences of means of nine scored rows; all scores are percentages of items answered correctly.", ""]
    for fp in FRACS:
        have = [k for k in S if k[0] == fp]
        if not have:
            continue
        md += [f"## Taught fraction 0.{fp}", "", "| Network | Rule | Updates to 80, A | Updates to 80, R | A at 12,288 / 24,576 / 49,152 | R at 12,288 / 24,576 / 49,152 | "
               "Length test (40 letters) | max | min | Rise before, last rise of R | Curve of R | CPU-hours | Resumed at |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for rule in "SB":
            for cfg in CONFIGS:
                s = S.get((fp, cfg, rule))
                if s is None:
                    continue
                lv = lambda k: " / ".join(fmt(s["levels"][k][str(u)]) for u in (12288, 24576, 49152))
                md.append(f"| `{cfg}` | {rule} | {fmt(s['to80']['A'])} | {fmt(s['to80']['R'])} | {lv('A')} | {lv('R')} | {lv('len40')} | {lv('max')} | {lv('min')} | "
                          f"{s['rise_before']:+.1f}, {s['last_rise']:+.1f} | {flat_word(s)} | {s['cpu_hours']:.2f} | {s.get('resumed_at') or 'never'} |")
        md.append("")
        rep = [(rule, S[(fp, "P8", rule)]) for rule in "SB" if (fp, "P8", rule) in S]
        if rep:
            md += ["Share of the connections of `P8` replaced at an event: " + "; ".join(
                f"rule {rule}: {s['replaced_share_first16']:.1f}% over the first sixteen events and {s['replaced_share_last16']:.1f}% over the last sixteen" for rule, s in rep) + ".", ""]
        md += ["| Rule | Network against `G` | Updates to 80 on A: word (ratio; ratio at a half) | Updates to 80 on R: word (ratio; ratio at a half) | R at 49,152: network, `G`, difference | Margin | "
               "Rises of the network; of `G` | End result |", "|---|---|---|---|---|---|---|---|"]
        for rule in "SB":
            g = S.get((fp, "G", rule))
            for cfg in ("F10", "P8"):
                s = S.get((fp, cfg, rule))
                if s is None or g is None:
                    continue
                e = dict(margin=MARGIN[(fp, rule)])
                cells = []
                for key in ("A", "R"):
                    xr, xc = g["to80"][key], s["to80"][key]
                    w = describe(xr, xc, DOSE - 512, DOSE - 512)
                    ratio = xc / xr if xr and xc else None
                    e[f"to80_{key}"] = dict(word=w, network=xc, G=xr, ratio=ratio, ratio_at_half=HALF[(cfg, rule, key)])
                    if ratio is not None and ratio < HALF[(cfg, rule, key)]:
                        e[f"to80_{key}"]["note"] = "is slowed less by the smaller fraction"
                    cells.append(f"{w} ({fmt(ratio, 2)}; {HALF[(cfg, rule, key)]:.2f})")
                d = s["end_level_R"] - g["end_level_R"]
                word = end_reading(d, e["margin"], g["flattened"], s["flattened"], g["flattened"] and s["flattened"])
                e.update(level_network=s["end_level_R"], level_G=g["end_level_R"], difference=d, reading=word,
                         rises_network=[s["rise_before"], s["last_rise"]], rises_G=[g["rise_before"], g["last_rise"]],
                         flattened_network=s["flattened"], flattened_G=g["flattened"], network_generalises_from_the_fraction=s["generalises_from_the_fraction"],
                         G_generalises_from_the_fraction=g["generalises_from_the_fraction"], spread_between_seeds=SPREAD.get((fp, rule)))
                pre = ""
                if s["flattened"] and g["flattened"] and not s["generalises_from_the_fraction"] and not g["generalises_from_the_fraction"]:
                    pre = "neither generalises from this fraction; "; e["neither_generalises_from_this_fraction"] = True
                if s["generalises_from_the_fraction"] and not g["generalises_from_the_fraction"] and word != "generalises better":
                    e["fixed_sentence"] = (f"reaches 80 on R at this fraction; the dense network has flattened at {g['end_level_R']:.1f}; the difference of {d:.1f} points is "
                                           f"inside the margin of {e['margin']}") if g["flattened"] else \
                        "reaches 80 on R at this fraction within the run, and the dense network does not; whether the dense network would reach 80 with more updates is not decided"
                out["against_G"][f"f{fp} {cfg} {rule}"] = e
                md.append(f"| {rule} | `{cfg}` | {cells[0]} | {cells[1]} | {s['end_level_R']:.1f}, {g['end_level_R']:.1f}, {d:+.1f} | {e['margin']} | "
                          f"{s['rise_before']:+.1f}, {s['last_rise']:+.1f} ({flat_word(s)}); {g['rise_before']:+.1f}, {g['last_rise']:+.1f} ({flat_word(g)}) | {pre}{word} |")
        md += ["", "| Network | R at 49,152: sweep, back-propagation, difference | Margin | Rises under the sweep; under back-propagation | The sweep against back-propagation |", "|---|---|---|---|---|"]
        for cfg in CONFIGS:
            a, b = S.get((fp, cfg, "S")), S.get((fp, cfg, "B"))
            if a is None or b is None:
                continue
            d = a["end_level_R"] - b["end_level_R"]; mg = MARGIN_SB[fp]
            word = end_reading(d, mg, b["flattened"], a["flattened"], a["flattened"] and b["flattened"])
            out["sweep_against_backprop"][f"f{fp} {cfg}"] = dict(level_S=a["end_level_R"], level_B=b["end_level_R"], difference=d, margin=mg, reading=word,
                                                                 rises_S=[a["rise_before"], a["last_rise"]], rises_B=[b["rise_before"], b["last_rise"]],
                                                                 flattened_S=a["flattened"], flattened_B=b["flattened"])
            md.append(f"| `{cfg}` | {a['end_level_R']:.1f}, {b['end_level_R']:.1f}, {d:+.1f} | {mg} | {a['rise_before']:+.1f}, {a['last_rise']:+.1f} ({flat_word(a)}); "
                      f"{b['rise_before']:+.1f}, {b['last_rise']:+.1f} ({flat_word(b)}) | {word} |")
        md.append("")
    if p4:
        md += ["## Part 4: the dense network at the half width, taught fraction 0.10, 122,880 updates", "",
               "| Rule | Updates to 80, R | Updates to 80, A | A at 12,288 / 24,576 / 49,152 / 122,880 | R at 12,288 / 24,576 / 49,152 / 122,880 | Rise before, last rise of R | Curve of R | CPU-hours |",
               "|---|---|---|---|---|---|---|---|"]
        for rule, r in p4.items():
            s = summary(r); out["part4"][rule] = s
            lv = lambda k: " / ".join(fmt(s["levels"][k][str(u)]) for u in (12288, 24576, 49152, 122880))
            md.append(f"| {rule} | {fmt(s['to80']['R'])} | {fmt(s['to80']['A'])} | {lv('A')} | {lv('R')} | {s['rise_before']:+.1f}, {s['last_rise']:+.1f} | {flat_word(s)} | {s['cpu_hours']:.2f} |")
        md.append("")
    # ---------------------------------------------------------------- the predictions
    P = out["predictions"]; ag = out["against_G"]
    L = lambda fp, cfg, rule, k, u: S[(fp, cfg, rule)]["levels"][k][str(u)]
    full = lambda fp: all((fp, c, r) in S for c in CONFIGS for r in "SB")
    if full(25):
        P["P54.1"] = dict(holds=all(ag[f"f25 F10 {r}"]["to80_A"]["word"] == "more" for r in "SB"), words={r: ag[f"f25 F10 {r}"]["to80_A"]["word"] for r in "SB"})
        x = S[(25, "P8", "S")]["to80"]["R"]
        P["P54.2"] = dict(holds=x is not None and x <= 24064, updates_to_80_R=x)
    if full(10):
        d = {f"{c} {r}": (ag[f"f10 {c} {r}"]["difference"], MARGIN[(10, r)]) for c in ("F10", "P8") for r in "SB"}
        P["P54.3"] = dict(holds=all(v < mg for v, mg in d.values()), differences_and_margins=d)
        g = {c: L(10, c, "B", "R", 49152) - L(10, c, "B", "R", 24576) for c in CONFIGS}
        P["P54.4"] = dict(holds=any(v >= 10 for v in g.values()), gains=g)
        a = {c: L(10, c, "S", "A", 49152) for c in CONFIGS}
        P["P54.5"] = dict(holds=all(v <= 30 for v in a.values()), levels_A=a)
    if (15, "G", "B") in S:
        P["P54.6"] = dict(holds=S[(15, "G", "B")]["to80"]["R"] is not None, updates_to_80_R=S[(15, "G", "B")]["to80"]["R"])
    if (15, "G", "S") in S:
        P["P54.7"] = dict(holds=S[(15, "G", "S")]["to80"]["R"] is None, updates_to_80_R=S[(15, "G", "S")]["to80"]["R"])
    if all(full(fp) for fp in FRACS):
        better = [k for k, e in ag.items() if e["reading"] == "generalises better"]
        P["P54.8"] = dict(holds=not better, generalises_better=better)
    if "B" in out["part4"]:
        P["P54.9"] = dict(holds=out["part4"]["B"]["to80"]["R"] is not None, updates_to_80_R=out["part4"]["B"]["to80"]["R"])
    if "S" in out["part4"]:
        P["P54.10"] = dict(holds=out["part4"]["S"]["end_level_R"] < 20, level_R=out["part4"]["S"]["end_level_R"])
    if P:
        md += ["## Predictions", "", "| | Holds | Figures |", "|---|---|---|"] + [
            f"| {k} | {'yes' if v['holds'] else 'no'} | {json.dumps({a: b for a, b in v.items() if a != 'holds'})} |" for k, v in P.items()] + [""]
    (OUT / "E54_TABLES.json").write_text(json.dumps(out, indent=1))
    (OUT / "E54_TABLES.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
