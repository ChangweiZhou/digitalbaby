"""E57 tables (brief TASK_E57_THE_SWEEP_WITH_A_SCALE_FOR_EACH_RECEIVING_UNIT_20261117.md, revision 5): the readings fixed by the brief, from
the receipts only, and the summaries of checks 4 and 6.

    python3 e57_tables.py [--tag NAME] [--checks]

Reads exp45/e57/e57_*_73129.json; the references are read with git show: E42's G and F10 under both rules at 44f91edc
(exp45/e42/e42_<G|F10>_<S|B>_73129.json) and E49's in16 under both rules at a36de661 (exp45/e49/e49_in16_<S|B>_73129.json).
Writes exp45/e57/E57_TABLES_<tag>.md and E57_SUMMARY_<tag>.json; with --checks also exp45/e57_qual/check4_73129.json and check6_73129.json.

`describe` is copied from e42_tables.py (origin/claude/gallant-dijkstra-i9zgsr at 44f91edc, lines 63 to 72; that file's SHA-256 is
87d2befa333151739622738953d5a72a881909b7ca6779b3834ad87b1931551c).
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e57"
QUAL = HERE / "exp45" / "e57_qual"
HALF, ROWK = 24576, ("loss", "A_counts", "R_counts", "taught_counts", "length_counts", "data_digest")
SCALES = ("matrix", "row", "rowcells", "rowcol")
TENSORS = ("book.weight", "l1.weight", "rnn.weight_ih_l0", "rnn.weight_hh_l0", "lm.weight", "l2.weight")
TNAME = {"book.weight": "byte table", "l1.weight": "l1", "rnn.weight_ih_l0": "input map", "rnn.weight_hh_l0": "recurrent map",
         "lm.weight": "lm", "l2.weight": "l2"}
CHECK2 = {"matrix": (0.367, 0.298, 0.603, 0.651, 0.606, 0.643), "row": (0.697, 0.624, 0.676, 0.762, 0.694, 0.710)}
B_END = "707c85693a98b6b1017ba94abb0754b12758650d8a0d6562d44a28363584ac3e"


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


def show(ref, path):
    return json.loads(subprocess.run(["git", "show", f"{ref}:MOD97_LOCAL_RULE_20261106/{path}"], cwd=HERE, capture_output=True, text=True,
                                     check=True).stdout)


def to80(curve, key):
    for i in range(len(curve) - 2):
        if all(r[key] >= 80 for r in curve[i:i + 3]):
            return curve[i]["step"]
    return None


def summarize(rec, upto=None):
    c = [r for r in rec["curve"] if upto is None or r["step"] <= upto]
    lv = {k: sum(r[k] for r in c[-3:]) / 3 for k in ("A", "R", "taught", "len12", "len40")}
    a, r = to80(c, "A"), to80(c, "R")
    learns = a is not None and r is not None and lv["A"] >= 80 and lv["R"] >= 80
    word = "learns the family" if learns else ("partly" if lv["taught"] >= 50 else "not learned")
    rise = lv["R"] - sum(x["R"] for x in [y for y in c if y["step"] <= c[-1]["step"] - 2048][-3:]) / 3
    return dict(updates=c[-1]["step"], A80=a, R80=r, T80=to80(c, "taught"), levels=lv, word=word, R_rise_2048=rise,
                loss_last=c[-1]["loss"], masks_every_row=all(all(all(v.values()) for v in x["masks"].values()) for x in c))


def name_of(rec):
    return f"{rec['setting']} {rec['scale']} {rec['rule']}" + (" cd0" if rec.get("code_decay") == 0 else "")


def load():
    runs = {}
    for p in sorted(OUT.glob("e57_*_73129.json")):
        r = json.loads(p.read_text())
        if r.get("experiment") == "E57" and r.get("complete"):
            runs[name_of(r)] = r
    refs = {"E42 G S": show("44f91edc", "exp45/e42/e42_G_S_73129.json"), "E42 G B": show("44f91edc", "exp45/e42/e42_G_B_73129.json"),
            "E42 F10 S": show("44f91edc", "exp45/e42/e42_F10_S_73129.json"), "E42 F10 B": show("44f91edc", "exp45/e42/e42_F10_B_73129.json"),
            "E49 in16 S": show("a36de661", "exp45/e49/e49_in16_S_73129.json"), "E49 in16 B": show("a36de661", "exp45/e49/e49_in16_B_73129.json")}
    return runs, refs


def fmt(x):
    return "not reached" if x is None else f"{x:,}"


def write(tag, checks):
    runs, refs = load()
    S = {k: summarize(v) for k, v in runs.items()}
    H = {k: summarize(v, HALF) for k, v in runs.items() if v["dose"] > HALF}
    RS = {k: summarize(v) for k, v in refs.items()}
    L = [f"# E57 tables ({tag})", "", f"Complete runs: {len(S)} of 9: {', '.join(S)}.", ""]
    L += ["## Every run, beside its references", "",
          "Updates to 80 (first of three consecutive scored rows at or above 80); levels (mean of the last three scored rows); the word for the run.", "",
          "| Run | Updates | A to 80 | R to 80 | T to 80 | Level A | Level R | Level T | Level at 40 letters | R: rise over the last 2,048 | Word | Masks at every row |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def row(n, s):
        lv = s["levels"]
        L.append(f"| {n} | {s['updates']:,} | {fmt(s['A80'])} | {fmt(s['R80'])} | {fmt(s['T80'])} | {lv['A']:.2f} | {lv['R']:.2f} | {lv['taught']:.2f} | "
                 f"{lv['len40']:.2f} | {s['R_rise_2048']:+.2f} | {s['word']} | {s['masks_every_row']} |")
    for k in S:
        if k in H:
            row(f"{k} (first 24,576)", H[k])
        row(k, S[k])
    for k, s in RS.items():
        row(k + " (reference)", s)
    L.append("")
    # describe
    D = {}
    L += ["## Updates to 80 in the words of `describe` (reference, candidate)", "", "| Candidate | Reference | Score | Reference's updates | Candidate's | Word |", "|---|---|---|---|---|---|"]
    def desc(cand, cs, refname, rs, key, Dc):
        w = describe(rs[key], cs[key], refs[refname]["dose"] - 512, Dc)
        D[f"{cand} against {refname} {key}"] = w
        L.append(f"| {cand} | {refname} | {key} | {fmt(rs[key])} | {fmt(cs[key])} | {w} |")
    for k, s in S.items():
        net = k.split()[0]
        rlist = {"in16": ["E49 in16 B"], "F10": ["E42 F10 S", "E42 F10 B"], "G": ["E42 G S", "E42 G B"]}[net]
        if k == "in16 none B":
            rlist = ["E49 in16 B"]
        for rn in rlist:
            for key in ("A80", "R80"):
                desc(k, s, rn, RS[rn], key, runs[k]["dose"] - 512)
                if k in H:
                    desc(k + " (first 24,576)", H[k], rn, RS[rn], key, HALF - 512)
    L.append("")
    seed_range = {}
    for k in S:
        if k.startswith(("F10", "G ")):
            net = k.split()[0]
            ref = RS[f"E42 {net} S"]
            seed_range[k] = {key: (None if S[k][key] is None else S[k][key] - ref[key]) for key in ("A80", "R80")}
    L += ["Differences from E42's run under the sweep (updates to 80; within the range of E42's three seeds if 768 or less in size): " +
          "; ".join(f"{k}: A {v['A80']:+,}, R {v['R80']:+,}" if None not in v.values() else f"{k}: {v}" for k, v in seed_range.items()), ""]
    # cosines
    L += ["## Cosines and sizes during the runs", "",
          "For each reading: the mean over 16 batches of the cosine between the sweep's gradient under each scale and the exact gradient, for six tensors, "
          "and the two sizes (`l1_weight_mean_abs`, `code_mean_abs`).", ""]
    for k, r in runs.items():
        L += [f"### {k}", "", "| Update | " + " | ".join(f"{s}: " + ", ".join(TNAME[t] for t in TENSORS) for s in SCALES) + " | l1 mean abs | code mean abs | loss of the row |",
              "|---|" + "---|" * len(SCALES) + "---|---|---|"]
        lossat = {x["step"]: x["loss"] for x in r["curve"]}
        for c in r["cosines"]:
            lv = lossat.get(c["update"])
            L.append(f"| {c['update']:,} | " + " | ".join(", ".join(f"{c['cosines'][s][t]:.3f}" for t in TENSORS) for s in SCALES) +
                     f" | {c['l1_weight_mean_abs']:.4f} | {c['code_mean_abs']:.4f} | {'' if lv is None else f'{lv:.3f}'} |")
        L.append("")
    # the cosine of l1 under matrix below 0.5
    def first_below(r):
        for c in r["cosines"]:
            if c["cosines"]["matrix"]["l1.weight"] < 0.5:
                return c["update"], next((x["loss"] for x in r["curve"] if x["step"] == c["update"]), None)
        return None, None
    fb = {}
    for k in ("in16 matrix S", "in16 none B"):
        if k in runs:
            fb[k] = first_below(runs[k])
    L += ["## The first reading at which the cosine of l1 under the scale `matrix` is below 0.5", ""]
    for k, (u, ls) in fb.items():
        L.append(f"- {k}: update {fmt(u)}, mean training loss of that scored row {ls if ls is None else f'{ls:.3f}'}.")
    if len(fb) == 2 and None not in (fb["in16 matrix S"][0], fb["in16 none B"][0]):
        earlier = fb["in16 none B"][0] - fb["in16 matrix S"][0] >= 4096
        sent = ("the cosine fell earlier in the run that fails" if earlier else
                "the cosine of l1 under the scale `matrix` does not separate the run that fails from the run that learns")
        L.append(f"- The brief's sentence: {sent}.")
        fb["sentence"] = sent
    L.append("")
    # the explanation
    W = {k: S[k]["word"] for k in S}
    def w(k):
        return W.get(k)
    out = None
    if all(w(k) for k in ("in16 matrix S", "in16 row S", "in16 rowcells S")):
        if w("in16 matrix S") == "learns the family":
            out = "E49's run was late and had not failed. No word is given to the explanation; the scales are compared by their updates to 80."
        elif w("in16 rowcells S") == "learns the family":
            out = "Part 1 is supported: a scale for each gate of each cell was sufficient." + (" The run with `row` does not learn it: the scales at the other maps took the gain away." if w("in16 row S") != "learns the family" else "")
        elif w("in16 row S") == "learns the family":
            out = ("Part 1 alone is not supported. A scale for each receiving unit at every map was sufficient; the runs do not say which of the scales at "
                   "`lm`, `l3`, `l2` and `l1` is needed beside that of the cells.")
        elif w("in16 rowcells S") == "not learned" and w("in16 row S") == "not learned":
            out = "The explanation is not supported." + (" The run with `rowcol` learns the family." if w("in16 rowcol S") == "learns the family" else "")
        else:
            out = "Any other outcome: the figures, without either word."
    L += ["## The explanation under test (words over 49,152 updates)", "", f"Words: {W}.", "", f"Reading: {out}", ""]
    # the code not decayed
    cd = None
    if "in16 matrix S cd0" in S and "in16 matrix S" in S:
        wd, wn = w("in16 matrix S"), w("in16 matrix S cd0")
        if wd == "not learned":
            cd = {"learns the family": "On this seed, the sweep with the scale `matrix` learns the family in the network with a limited input when l1 is not decayed, so that the decay of l1 was necessary for the failure of the run with a decayed code.",
                  "partly": f"On this seed, the network without the decay of l1 reaches A {S['in16 matrix S cd0']['levels']['A']:.2f}, R {S['in16 matrix S cd0']['levels']['R']:.2f} and T {S['in16 matrix S cd0']['levels']['taught']:.2f} and does not learn the family.",
                  "not learned": "On this seed, the network fails without the decay of l1 as well."}[wn]
        else:
            cd = "The run with a decayed code has another word; the levels and the updates to 80 of the two runs are given and no sentence on the decay."
        a, b = runs["in16 matrix S"]["cosines"], runs["in16 matrix S cd0"]["cosines"]
        L += ["## The code not decayed, beside the run with a decayed code", "", f"Reading: {cd}", "",
              "| Update | l1 mean abs (decayed) | l1 mean abs (not decayed) | code mean abs (decayed) | code mean abs (not decayed) |", "|---|---|---|---|---|"]
        for x, y in zip(a, b):
            L.append(f"| {x['update']:,} | {x['l1_weight_mean_abs']:.4f} | {y['l1_weight_mean_abs']:.4f} | {x['code_mean_abs']:.4f} | {y['code_mean_abs']:.4f} |")
        L.append("")
    js = dict(runs={k: S[k] for k in S}, first_24576={k: H[k] for k in H}, references=RS, describe=D, seed_range=seed_range, first_below=fb,
              words=W, explanation=out, code_not_decayed=cd)
    (OUT / f"E57_TABLES_{tag}.md").write_text("\n".join(L) + "\n")
    (OUT / f"E57_SUMMARY_{tag}.json").write_text(json.dumps(js, indent=1, default=str))
    print("\n".join(L[:60]))
    if checks:
        c4 = dict(runs={k: dict(rows=len(r["curve"]), masks_true_at_every_row=S[k]["masks_every_row"]) for k, r in runs.items()})
        c4["pass"] = len(runs) == 9 and all(v["masks_true_at_every_row"] for v in c4["runs"].values())
        (QUAL / "check4_73129.json").write_text(json.dumps(c4, indent=1))
        m, b = runs["in16 matrix S"], runs["in16 none B"]
        mine = [r for r in m["curve"] if r["step"] <= HALF]; ref = refs["E49 in16 S"]["curve"]
        diffs = [(x["step"], k) for x, y in zip(mine, ref) for k in ROWK if x[k] != y[k]]
        cos = {}
        for name, r in (("in16 matrix S", m), ("in16 none B", b)):
            c = next(x for x in r["cosines"] if x["update"] == HALF)
            exp = CHECK2 if name == "in16 matrix S" else {"matrix": (0.442, 0.309, 0.523, 0.582, 0.700, 0.706), "row": (0.405, 0.363, 0.567, 0.631, 0.700, 0.707)}
            cos[name] = {s: dict(got=[c["cosines"][s][t] for t in TENSORS], expected=list(v),
                                 max_diff=max(abs(c["cosines"][s][t] - e) for t, e in zip(TENSORS, v))) for s, v in exp.items()}
        c6 = dict(matrix_rows_to_24576=dict(rows=len(mine), rows_ref=len([r for r in ref if r["step"] <= HALF]), differences=diffs[:10]),
                  B_tensors_sha256=b["tensors_sha256"], B_expected=B_END, cosines_at_24576=cos)
        c6["pass"] = not diffs and len(mine) == 97 and b["tensors_sha256"] == B_END and all(v["max_diff"] <= 0.002 for d in cos.values() for v in d.values())
        (QUAL / "check6_73129.json").write_text(json.dumps(c6, indent=1))
        print("check 4:", c4["pass"], "check 6:", c6["pass"])


if __name__ == "__main__":
    a = sys.argv[1:]
    write(a[a.index("--tag") + 1] if "--tag" in a else "73129", "--checks" in a)
