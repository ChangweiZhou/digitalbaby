"""E59 tables (brief TASK_E59_..._20261117.md, revision 2): the readings in the words fixed by the brief, from the receipts only.

    python3 e59_tables.py [--out DIR] [--tag NAME]

Reads DIR/e59_<start>_n<n>_<rule>_<seed>.json (complete runs; a run in progress is listed and not read). Writes DIR/E59_TABLES_<tag>.md and
DIR/E59_SUMMARY_<tag>.json. Definitions, all from the brief:
  updates to 80   the first row r at which the held panel at 12 fillers is at or above 240 of 300 and is so at every row up to r + 64;
                  none if no such r <= 8,128 (also computed on the held panel at 0 fillers)
  level           the mean over the nine rows from update 7,936 to 8,192
  flattened       the mean of the held panel over rows 7,296 to 8,192 is less than 3 points above its mean over rows 5,248 to 6,144
  seed's word     describe() of e42_tables.py with the second start as the reference and L = 8,128 on both sides, as the brief states it
  comparison      fewer / more / about the same / not decided from the three seeds' words
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e59"
SEEDS = (73129, 93683, 116089)
STARTS = ("fresh", "born", "lived", "born-shuffled")
NS = (2048, 512, 256)
RULES = ("S", "B")
L = 8128
PCT = lambda c: 100.0 * c["exact"] / c["n"]


def load(d):
    runs, prog = {}, []
    for p in sorted(Path(d).glob("e59_*_n*_*_*.json")):
        r = json.loads(p.read_text())
        if r.get("experiment") != "E59" or r.get("updates") != 8192:
            continue
        k = (r["start"], r["n_taught"], r["rule"], r["seed"])
        if r.get("complete"):
            runs[k] = r
        else:
            prog.append(f"{p.name} at update {r['rows'][-1]['updates']:,}" + (f" ({r['stopped']})" if r.get("stopped") else ""))
    return runs, prog


def to80(rows, key):
    v = [(x["updates"], x[key]["exact"] >= 0.8 * x[key]["n"]) for x in rows]
    for i, (u, ok) in enumerate(v):
        if u > L:
            break
        if ok and all(o for w, o in v[i:] if w <= u + 64):
            return u
    return None


def mean_rows(rows, key, a, b):
    xs = [PCT(x[key]) for x in rows if a <= x["updates"] <= b]
    return sum(xs) / len(xs), len(xs)


def summarize(r):
    rows = r["rows"]
    lv = {k: mean_rows(rows, k, 7936, 8192)[0] for k in ("held12", "held0", "taught")}
    late, n1 = mean_rows(rows, "held12", 7296, 8192); early, n2 = mean_rows(rows, "held12", 5248, 6144)
    assert n1 == n2 == 29 and len(rows) == 273
    third = next((x["updates"] for x in rows if x["held12"]["third_is_end"] >= 240), None)
    out = dict(to80=to80(rows, "held12"), to80_held0=to80(rows, "held0"), levels=lv, flattened=late - early < 3, late_minus_early=late - early,
               first_row_third_byte_end_240=third, finite=r["complete"]["all_losses_finite"] and all(x["finite_losses"] for x in rows),
               taught_n=rows[-1]["taught"]["n"], cpu=r["complete"]["cpu_total"], tensors_sha256=r["complete"]["tensors_sha256"],
               weights_sha256=r["complete"]["weights_sha256"], weights=r["complete"]["weights"],
               held12_last=rows[-1]["held12"], earlier_row0=rows[0].get("earlier_panels_k12"), earlier_last=rows[-1].get("earlier_panels_k12"),
               digests={x["updates"]: x["data_digest"]["sha256"] for x in rows}, check4=r.get("check4"), check4_pass=r["complete"].get("check4_pass"))
    if r["n_taught"] == 256:
        u = [x["held12"]["unseen"] for x in rows if 7936 <= x["updates"]]; s = [x["held12"]["seen"] for x in rows if 7936 <= x["updates"]]
        out["held12_parts_level"] = dict(unseen_n=u[0]["n"], unseen=sum(100 * c["exact"] / max(c["n"], 1) for c in u) / len(u),
                                         seen_n=s[0]["n"], seen=sum(100 * c["exact"] / c["n"] for c in s) / len(s))
    return out


def word(a, b):
    """The seed's word for a first start (updates a) against a second (updates b)."""
    if a is not None and b is not None:
        if a / b < 2 / 3 and b - a >= 48:
            return "fewer"
        if a / b > 1.5 and a - b >= 48:
            return "more"
        return "about the same"
    if a is not None:
        return "fewer" if a / L <= 2 / 3 else "not decided"
    if b is not None:
        return "more" if L / b >= 1.5 else "not decided"
    return "neither reaches"


def combine(ws):
    c = {w: ws.count(w) for w in set(ws)}
    if len(ws) < 3:
        return "incomplete"
    if c.get("fewer", 0) >= 2 and "more" not in ws:
        return "fewer"
    if c.get("more", 0) >= 2 and "fewer" not in ws:
        return "more"
    if c.get("about the same", 0) >= 2:
        return "about the same"
    return "not decided"


def compare(S, x, y, key="to80"):
    """x, y: functions seed -> run key. Returns the three seeds' values and words, and the comparison's word."""
    per = []
    for sd in SEEDS:
        kx, ky = x(sd), y(sd)
        if kx in S and ky in S:
            a, b = S[kx][key], S[ky][key]
            per.append(dict(seed=sd, first=a, second=b, word=word(a, b)))
    return dict(seeds=per, word=combine([p["word"] for p in per]))


def saves(S, start, n, rule, sd, other="fresh"):
    """The five conditions of "saves pairs against fresh" for one seed; returns the conditions and the outcome."""
    t, f = S.get((start, n, rule, sd)), S.get((other, n, rule, sd))
    if not t or not f:
        return None
    c = dict(i=t["to80"] is not None and t["levels"]["held12"] >= 80, ii=f["to80"] is None, iii=f["flattened"], iv=f["levels"]["taught"] >= 95,
             v=f["levels"]["held12"] <= t["levels"]["held12"] - 10)
    return dict(conditions=c, saves=all(c.values()), trained_level=t["levels"]["held12"], fresh_level=f["levels"]["held12"])


def pairs_reading(S, start, rule):
    out = {}
    for n in NS:
        per = {sd: saves(S, start, n, rule, sd) for sd in SEEDS}
        if any(v is None for v in per.values()):
            out[n] = dict(per_seed=per, sentence="incomplete"); continue
        yes = [sd for sd, v in per.items() if v["saves"]]
        licensed = False
        if len(yes) >= 2:
            third = [sd for sd in SEEDS if sd not in yes]
            licensed = all(per[sd]["fresh_level"] <= per[sd]["trained_level"] for sd in third)
        i_ii = [sd for sd, v in per.items() if v["conditions"]["i"] and v["conditions"]["ii"]]
        s = None
        if licensed:
            s = "saves pairs (the brief's sentence is licensed at this n)"
        elif len(i_ii) >= 2 and any(not per[sd]["conditions"]["iii"] for sd in i_ii):
            s = "reaches 80 from n pairs within the run, and the fresh network does not; whether the fresh network would reach 80 with more updates is not decided"
        elif len(i_ii) >= 2 and any(not per[sd]["conditions"]["iv"] for sd in i_ii):
            s = "the fresh network has not learned its taught pairs within the run; the difference is one of presentations, and the number of pairs is not compared"
        out[n] = dict(per_seed=per, licensed=licensed, sentence=s)
    every256 = all(sum(1 for sd in SEEDS if (st, 256, rule, sd) in S and S[(st, 256, rule, sd)]["to80"] is not None) >= 2 for st in ("fresh", "born", "lived"))
    out["every_start_reaches_80_from_256_on_two_seeds"] = every256
    return out


def fmt(v):
    return "–" if v is None else f"{v:,}"


def write(d, tag):
    runs, prog = load(d)
    S = {k: summarize(r) for k, r in runs.items()}
    Lines = [f"# E59 tables ({tag})", "", f"Complete runs read: {len(S)} of 60." + (" In progress or stopped: " + "; ".join(prog) + "." if prog else ""), ""]
    Lines += ["## Every run", "", "Updates to 80 on the held panel at 12 and at 0 fillers; levels (mean of the nine rows 7,936 to 8,192) of the held panels and of the taught panel; "
              "whether the held curve has flattened (late minus early mean); the first row at which the third answer byte is the end symbol on 240 of 300 held items; CPU-seconds.", "",
              "| Start | n | Rule | Seed | to 80 (12) | to 80 (0) | held 12 | held 0 | taught (items) | flattened (late − early) | third byte end | CPU s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for st in STARTS:
        for n in NS:
            for rule in RULES:
                for sd in SEEDS:
                    k = (st, n, rule, sd)
                    if k not in S:
                        continue
                    v = S[k]; lv = v["levels"]
                    Lines.append(f"| {st} | {n} | {rule} | {sd} | {fmt(v['to80'])} | {fmt(v['to80_held0'])} | {lv['held12']:.1f} | {lv['held0']:.1f} | "
                                 f"{lv['taught']:.1f} ({v['taught_n']}) | {v['flattened']} ({v['late_minus_early']:+.1f}) | {fmt(v['first_row_third_byte_end_240'])} | {v['cpu']:.0f} |")
    Lines.append("")
    p256 = [(k, v["held12_parts_level"]) for k, v in S.items() if "held12_parts_level" in v]
    if p256:
        Lines += ["### 256 pairs: the held panel's level in two parts (items with a symbol in a place no taught pair has it; the others)", "",
                  "| Start | Rule | Seed | unseen-place items (n): level | other items (n): level |", "|---|---|---|---|---|"]
        for (st, n, rule, sd), q in sorted(p256, key=lambda z: (STARTS.index(z[0][0]), z[0][2], z[0][3])):
            Lines.append(f"| {st} | {rule} | {sd} | ({q['unseen_n']}) {q['unseen']:.1f} | ({q['seen_n']}) {q['seen']:.1f} |")
        Lines.append("")
    # comparisons
    R = dict(presentations={}, pairs={}, source={}, rules={}, b2={})
    Lines += ["## Presentations: updates to 80 on the held panel at 12 fillers (the brief's words)", "",
              "| Rule | n | Comparison | 73129 | 93683 | 116089 | Word |", "|---|---|---|---|---|---|---|"]
    cmp_cell = lambda p: f"{fmt(p['first'])} vs {fmt(p['second'])}: {p['word']}"
    def row(label, rule, n, c):
        cells = {p["seed"]: cmp_cell(p) for p in c["seeds"]}
        Lines.append(f"| {rule} | {n} | {label} | " + " | ".join(cells.get(sd, "") for sd in SEEDS) + f" | **{c['word']}** |")
    for rule in RULES:
        for n in NS:
            for a, b in (("born", "fresh"), ("lived", "fresh"), ("lived", "born")):
                c = compare(S, lambda sd: (a, n, rule, sd), lambda sd: (b, n, rule, sd))
                R["presentations"][f"{a} against {b}, {rule}, {n}"] = c; row(f"{a} against {b}", rule, n, c)
        for a, b in (("born", "born-shuffled"), ("born-shuffled", "fresh")):
            c = compare(S, lambda sd: (a, 2048, rule, sd), lambda sd: (b, 2048, rule, sd))
            R["presentations"][f"{a} against {b}, {rule}, 2048"] = c; row(f"{a} against {b}", rule, 2048, c)
    Lines.append("")
    # the order of the three starts
    Lines += ["Where *fewer* holds for lived against born and for born against fresh and not for lived against fresh, no order is drawn:", ""]
    for rule in RULES:
        for n in NS:
            g = lambda a, b: R["presentations"].get(f"{a} against {b}, {rule}, {n}", {}).get("word")
            if g("lived", "born") == "fewer" and g("born", "fresh") == "fewer" and g("lived", "fresh") != "fewer":
                Lines.append(f"- {rule}, {n}: the three comparisons do not form an order; none is drawn.")
    Lines.append("")
    # source and fillers
    Lines += ["## The source of a saving (2,048 pairs)", ""]
    for rule in RULES:
        bf, bbs, bsf = (R["presentations"][f"{x}, {rule}, 2048"]["word"] for x in ("born against fresh", "born against born-shuffled", "born-shuffled against fresh"))
        if bf == "fewer" and bbs == "fewer":
            src = "ascribed to what the first family taught"
        elif bf == "fewer" and bsf == "fewer" and bbs == "about the same":
            src = "ascribed to the sizes of the weights that training leaves"
        elif bf == "fewer":
            src = "no source named"
        else:
            src = "no saving of born against fresh to ascribe"
        R["source"][rule] = dict(born_against_fresh=bf, born_against_shuffled=bbs, shuffled_against_fresh=bsf, source=src)
        Lines.append(f"- {rule}: born against fresh *{bf}*; born against born-shuffled *{bbs}*; born-shuffled against fresh *{bsf}*: {src}.")
    Lines += ["", "Updates to 80 at 12 and at 0 fillers, and the first row with the third byte the end symbol on 240 held items, 2,048 pairs:", "",
              "| Start | Rule | Seed | to 80 at 12 | to 80 at 0 | third byte end |", "|---|---|---|---|---|---|"]
    for st in STARTS:
        for rule in RULES:
            for sd in SEEDS:
                v = S.get((st, 2048, rule, sd))
                if v:
                    Lines.append(f"| {st} | {rule} | {sd} | {fmt(v['to80'])} | {fmt(v['to80_held0'])} | {fmt(v['first_row_third_byte_end_240'])} |")
    Lines.append("")
    # pairs
    Lines += ["## Pairs (the five conditions of \"saves pairs against fresh\")", ""]
    for st in ("born", "lived"):
        for rule in RULES:
            pr = pairs_reading(S, st, rule); R["pairs"][f"{st} {rule}"] = pr
            for n in NS:
                q = pr[n]
                if q.get("sentence") == "incomplete":
                    Lines.append(f"- {st}, {rule}, {n}: incomplete"); continue
                per = "; ".join(f"{sd}: " + " ".join(f"({k}) {'yes' if ok else 'no'}" for k, ok in v["conditions"].items()) +
                                f" [held level {v['trained_level']:.1f} against fresh {v['fresh_level']:.1f}]" for sd, v in q["per_seed"].items())
                Lines.append(f"- {st}, {rule}, {n}: {q['sentence'] or 'no sentence of the brief applies'}. Seeds: {per}")
            Lines.append(f"- {st}, {rule}: every start (fresh, born, lived) reaches 80 from 256 pairs on at least two seeds: {pr['every_start_reaches_80_from_256_on_two_seeds']}"
                         + ("; the brief's sentence: \"every start generalises from the smallest number of pairs tested; no difference in the number of pairs is shown, and none is excluded below 256\"" if pr['every_start_reaches_80_from_256_on_two_seeds'] else ""))
            # the reverse case: fresh in the trained start's place
            for n in NS:
                per = {sd: saves(S, "fresh", n, rule, sd, other=st) for sd in SEEDS}
                if any(v is None for v in per.values()):
                    continue
                yes = [sd for sd, v in per.items() if v["saves"]]
                lic = len(yes) >= 2 and all(per[sd]["fresh_level"] <= per[sd]["trained_level"] for sd in SEEDS if sd not in yes)
                R["pairs"][f"fresh against {st} {rule} {n}"] = dict(per_seed=per, licensed=lic)
                if any(v["conditions"]["i"] and v["conditions"]["ii"] for v in per.values()):
                    Lines.append(f"- reverse case, fresh in the place of {st}, {rule}, {n}: licensed {lic}; seeds: " + "; ".join(
                        f"{sd}: " + " ".join(f"({k}) {'yes' if ok else 'no'}" for k, ok in v["conditions"].items()) +
                        f" [fresh {v['trained_level']:.1f} against {st} {v['fresh_level']:.1f}]" for sd, v in per.items()))
    Lines.append("")
    # rules
    Lines += ["## The two rules: the sweep against back-propagation, same start and n", "", "| Start | n | 73129 | 93683 | 116089 | Word |", "|---|---|---|---|---|---|"]
    for st in STARTS:
        for n in (NS if st != "born-shuffled" else (2048,)):
            c = compare(S, lambda sd: (st, n, "S", sd), lambda sd: (st, n, "B", sd)); R["rules"][f"{st} {n}"] = c
            Lines.append(f"| {st} | {n} | " + " | ".join({p['seed']: cmp_cell(p) for p in c['seeds']}.get(sd, "") for sd in SEEDS) + f" | **{c['word']}** |")
    Lines += ["", "Requirement B2's case: born and lived under the sweep against fresh under back-propagation.", "", "| Start | n | 73129 | 93683 | 116089 | Word |", "|---|---|---|---|---|---|"]
    for st in ("born", "lived"):
        for n in NS:
            c = compare(S, lambda sd: (st, n, "S", sd), lambda sd: ("fresh", n, "B", sd)); R["b2"][f"{st} S against fresh B, {n}"] = c
            Lines.append(f"| {st} | {n} | " + " | ".join({p['seed']: cmp_cell(p) for p in c['seeds']}.get(sd, "") for sd in SEEDS) + f" | **{c['word']}** |")
    Lines.append("")
    # the explanation under test
    Lines += ["## The explanation under test (the brief's criteria)", ""]
    ex = {}
    for rule in RULES:
        P_ = R["presentations"]
        any_fewer = any(P_[f"{a} against fresh, {rule}, {n}"]["word"] == "fewer" for a in ("born", "lived") for n in NS)
        at2048 = [P_[f"{a} against fresh, {rule}, 2048"]["word"] for a in ("born", "lived")]
        untested = any(w in ("not decided", "neither reaches") for w in at2048)
        sup_words = P_[f"born against fresh, {rule}, 2048"]["word"] == "fewer" and P_[f"born against born-shuffled, {rule}, 2048"]["word"] == "fewer"
        ex[rule] = dict(any_fewer_any_n=any_fewer, words_at_2048=at2048, untested=untested, support_words_hold=sup_words,
                        refutation_conditions_hold=(not any_fewer) and all(w in ("about the same", "more") for w in at2048))
        Lines.append(f"- {rule}: some comparison of born or lived against fresh is *fewer* at some n: {any_fewer}; the two comparisons at 2,048: {at2048}; "
                     f"the refutation's conditions hold under this rule: {ex[rule]['refutation_conditions_hold']}; the support's first two conditions hold: {sup_words}.")
    ref_both = all(ex[r]["refutation_conditions_hold"] for r in RULES)
    Lines.append(f"- Refuted (the conditions under both rules): {ref_both}. Supported (fewer than fresh and than born-shuffled at 2,048 under both rules, with the filler condition): "
                 f"{all(ex[r]['support_words_hold'] for r in RULES)}.")
    R["explanation"] = dict(per_rule=ex, refuted=ref_both, supported=all(ex[r]["support_words_hold"] for r in RULES))
    Lines.append("")
    # updates to 80 at 0 and at 12 fillers at 256 pairs (where a saving is found)
    Lines += ["Updates to 80 at 12 and at 0 fillers, 256 pairs:", "", "| Start | Rule | Seed | to 80 at 12 | to 80 at 0 |", "|---|---|---|---|---|"]
    for st in ("fresh", "born", "lived"):
        for rule in RULES:
            for sd in SEEDS:
                v = S.get((st, 256, rule, sd))
                if v:
                    Lines.append(f"| {st} | {rule} | {sd} | {fmt(v['to80'])} | {fmt(v['to80_held0'])} |")
    Lines.append("")
    # earlier panels, given and not interpreted
    Lines += ["## The earlier families' panels at row 0 and at the last row (right of 300 at 12 letters; given, not interpreted)", "",
              "| Start | n | Rule | Seed | " + " | ".join(f"{e} (0 → end)" for e in ("add89", "sub89", "max89", "min89", "add97", "sub97", "add73", "sub73")) + " |",
              "|---|---|---|---|" + "---|" * 8]
    for st in STARTS:
        for n in NS:
            for rule in RULES:
                for sd in SEEDS:
                    v = S.get((st, n, rule, sd))
                    if v and v["earlier_row0"]:
                        Lines.append(f"| {st} | {n} | {rule} | {sd} | " + " | ".join(f"{v['earlier_row0'][e]['exact']} → {v['earlier_last'][e]['exact']}"
                                                                             for e in ("add89", "sub89", "max89", "min89", "add97", "sub97", "add73", "sub73")) + " |")
    Lines.append("")
    tot = sum(v["cpu"] for v in S.values())
    R["cpu_total_runs_h"] = tot / 3600
    Lines += [f"CPU of the {len(S)} runs, scoring included: {tot / 3600:.2f} hours ({min(v['cpu'] for v in S.values()):.0f} to {max(v['cpu'] for v in S.values()):.0f} seconds a run).", ""]
    # check 5
    bad = []
    for sd in SEEDS:
        for n in NS:
            ks = [k for k in S if k[1] == n and k[3] == sd]
            for k in ks[1:]:
                if S[k]["digests"] != S[ks[0]]["digests"]:
                    bad.append(f"{k} against {ks[0]}")
    R["check5"] = dict(differences=bad, pass_=not bad)
    Lines += ["## Check 5: the same batches", "", ("At every row the data digest is equal across the complete runs of a seed and a number of pairs." if not bad else "Differences: " + "; ".join(bad)), ""]
    c4 = {f"{k[2]} {k[3]}": S[k]["check4_pass"] for k in S if k[0] == "born" and k[1] == 2048}
    Lines += [f"Check 4 (born, 2,048 pairs, rows 64 to 4,096 against E43): {c4}.", ""]
    fin = [k for k, v in S.items() if not v["finite"]]
    Lines += [f"Runs with a loss that is not finite: {fin or 'none'}.", ""]
    (Path(d) / f"E59_TABLES_{tag}.md").write_text("\n".join(Lines) + "\n")
    js = dict(runs={" ".join(map(str, k)): {kk: vv for kk, vv in v.items() if kk != "digests"} for k, v in S.items()}, readings=R, in_progress=prog)
    (Path(d) / f"E59_SUMMARY_{tag}.json").write_text(json.dumps(js, indent=1, default=str))
    print("\n".join(Lines))


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    write(Path(opt("--out", str(OUT))), opt("--tag", "test"))
