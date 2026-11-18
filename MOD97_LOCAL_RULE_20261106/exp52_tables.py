"""E52: the tables and the readings fixed beforehand in the brief (revision 2), from the receipts of exp52_rate.py. Read-only; standard library.

    python3 exp52_tables.py [--out DIR]      # writes E52_TABLES.md and E52_SUMMARY.json in DIR (default exp45/e52)

Updates to 80 on a score: the first scored row from which three scored rows in a row are at or above 80% (rows every 256 updates). A run that
does not reach 80 has no count; its last recordable count is its dose less 512 (for a diverged run, the dose is still the planned one: it is
counted as not reached).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RATES = (0.0003, 0.001, 0.003, 0.01)
E41_DOSE = {256: 40960, 512: 12288}
WIDTH_NAME = {256: "half", 512: "present"}
GROUPS = ("book", "l1", "input_map", "recurrent", "lm", "l2", "l3")
ROW = 256


def first_row(curve, key, th=80):          # exp41_scale.first_row, version "three"
    v = [(r["step"], r[key]) for r in curve]
    for i in range(len(v)):
        if i + 2 < len(v) and all(x >= th for _, x in v[i:i + 3]):
            return i
    return None


def describe_step(xn, xw, Dn, Dw):         # exp41_scale.describe_step, verbatim
    if xn is not None and xw is not None:
        q = xw / xn
        return "falls" if q < 2 / 3 else "rises" if q > 1.5 else "stays"
    if xn is not None:
        return "rises" if Dw / xn >= 1.5 else "not decided"
    if xw is not None:
        return "falls" if xw / Dn <= 2 / 3 else "not decided"
    return "neither reaches"


WORD = {"falls": "faster", "rises": "slower", "stays": "the same", "not decided": "not decided", "neither reaches": "neither reaches"}


def mean(v):
    return sum(v) / len(v)


def rms_change(curve, key, n=16):
    v = [r[key] for r in curve[-n:]]
    return (sum((b - a) ** 2 for a, b in zip(v, v[1:])) / (len(v) - 1)) ** .5


def summarize(rec):
    c = rec["curve"]; H = rec["H"]
    s = dict(rule=rec["rule"], H=H, fp=round(rec["fraction"] * 100), rate=rec["rate"], decay=rec["decay"], product=rec["product_control"],
             dose=rec["dose"], diverged=rec["diverged"], diverged_at_row=rec.get("diverged_at_row"), rows=len(c),
             last_recordable=rec["dose"] - 512, tensors_sha256=rec["tensors_sha256"], cpu_updates=rec["cpu_updates"], cpu_scoring=rec["cpu_scoring"],
             len12_is_A_add=rec["length_test_12_items_equal_A_add89"])
    for key in ("A", "R"):
        i = first_row(c, key)
        s[f"{key}80"] = None if i is None else c[i]["step"]
        s[f"{key}80_norms"] = None if i is None else c[i]["norms"]
        s[f"{key}_final"] = mean([r[key] for r in c[-3:]])
        s[f"{key}_rms16"] = rms_change(c, key)
    at41 = [r for r in c if r["step"] <= E41_DOSE[H]]
    s["at_e41_dose"] = None
    if rec["dose"] != E41_DOSE[H] and len(at41) >= 3 and at41[-1]["step"] == E41_DOSE[H]:
        s["at_e41_dose"] = dict(A=mean([r["A"] for r in at41[-3:]]), R=mean([r["R"] for r in at41[-3:]]), len40_last=at41[-1]["len40"],
                                len40_mean3=mean([r["len40"] for r in at41[-3:]]), norms=at41[-1]["norms"])
    s["len40_last"] = c[-1]["len40"]; s["len40_mean3"] = mean([r["len40"] for r in c[-3:]])
    s["len40_range_last12"] = [min(r["len40"] for r in c[-12:]), max(r["len40"] for r in c[-12:])]
    s["taught_final"] = mean([r["taught"] for r in c[-3:]])
    s["norms_final"] = c[-1]["norms"]
    return s


def load(out):
    S = {}
    for p in sorted(Path(out).glob("e52_*.json")):
        r = json.loads(p.read_text())
        if r.get("complete"):
            s = summarize(r); s["receipt"] = p.name
            S[(s["rule"], s["H"], s["fp"], s["rate"], s["product"])] = s
    return S


def best_rate(S, rule, H, fp, key):
    """(named best rate or None, tied rates, count used) among the main runs of the case."""
    got = {rate: S[(rule, H, fp, rate, False)][f"{key}80"] for rate in RATES if (rule, H, fp, rate, False) in S}
    got = {k: v for k, v in got.items() if v is not None}
    if not got:
        return None, [], None
    lo = min(got.values()); tied = sorted(k for k, v in got.items() if v - lo <= ROW)
    if len(tied) == 1:
        return tied[0], tied, got[tied[0]]
    if 0.001 in tied:
        return 0.001, tied, got[0.001]
    return None, tied, lo


def compare(S, rule, H, fp, rate, key, product=False):
    a, b = S.get((rule, H, fp, 0.001, False)), S.get((rule, H, fp, rate, product))
    if a is None or b is None:
        return None
    return WORD[describe_step(a[f"{key}80"], b[f"{key}80"], a["last_recordable"], b["last_recordable"])]


def gains(S, rule, H, fp):
    """The rule gains from its own rate: faster at some rate on R and not slower at that rate on A; where no rate reaches 80 on R, faster on A."""
    rates = [r for r in RATES if r != 0.001 and (rule, H, fp, r, False) in S]
    reached_R = any(S[(rule, H, fp, r, False)]["R80"] is not None for r in RATES if (rule, H, fp, r, False) in S)
    at = []
    for r in rates:
        if reached_R:
            if compare(S, rule, H, fp, r, "R") == "faster" and compare(S, rule, H, fp, r, "A") != "slower":
                at.append(r)
        elif compare(S, rule, H, fp, r, "A") == "faster":
            at.append(r)
    return dict(gains=bool(at), at_rates=at, read_on="R and A" if reached_R else "A (no rate reaches 80 on R)")


def lag(S, H, fp, key):
    out = {}
    s1, b1 = S.get(("S", H, fp, 0.001, False)), S.get(("B", H, fp, 0.001, False))
    xs, xb = (s1 or {}).get(f"{key}80"), (b1 or {}).get(f"{key}80")
    if xs and xb:
        out["at_0.001"] = dict(value=xs / xb, text=f"{xs / xb:.2f}")
    elif xb and s1:
        out["at_0.001"] = dict(lower=s1["last_recordable"] / xb, text=f"above {s1['last_recordable'] / xb:.2f} (the sweep not reached)")
    elif xs and b1:
        out["at_0.001"] = dict(upper=xs / b1["last_recordable"], text=f"below {xs / b1['last_recordable']:.2f} (back-propagation not reached)")
    else:
        out["at_0.001"] = dict(text="not formed")
    bs = best_rate(S, "S", H, fp, key); bb = best_rate(S, "B", H, fp, key)
    longest = lambda rule: max(S[k]["last_recordable"] for k in S if k[:3] == (rule, H, fp) and not k[4])
    ys, yb = bs[2], bb[2]
    name = lambda b: ("tie of " + ", ".join(f"{r:g}" for r in b[1])) if b[0] is None and b[1] else (f"{b[0]:g}" if b[0] is not None else "none")
    out["sweep_best"], out["backprop_best"] = name(bs), name(bb)
    if ys and yb:
        out["own"] = dict(value=ys / yb, text=f"{ys / yb:.2f}")
    elif yb:
        out["own"] = dict(lower=longest("S") / yb, text=f"above {longest('S') / yb:.2f} (the sweep reached at no rate; bound from its longest run)")
    elif ys:
        out["own"] = dict(upper=ys / longest("B"), text=f"below {ys / longest('B'):.2f} (back-propagation reached at no rate; bound from its longest run)")
    else:
        out["own"] = dict(text="not formed")
    a, o = out["at_0.001"], out["own"]
    cls = "not formed" if "text" in a and len(a) == 1 or len(o) == 1 else "not decided"
    if "value" in a and "value" in o:
        q = o["value"] / a["value"]
        cls = ("gone" if o["value"] <= 1.0 else "smaller") if q < 2 / 3 else "larger" if q > 1.5 else "the same within the scatter of one seed"
    elif "lower" in a and "value" in o:                       # lag at 0.001 is above a bound
        if o["value"] < 2 / 3 * a["lower"]:
            cls = "gone" if o["value"] <= 1.0 else "smaller"
    elif "value" in a and "lower" in o:
        if o["lower"] > 1.5 * a["value"]:
            cls = "larger"
    out["class"] = cls
    return out


def fmt(x):
    return "–" if x is None else f"{x:,}"


def nrm(n):
    return "–" if n is None else " / ".join(f"{n[g]:.1f}" for g in GROUPS)


def write(out):
    S = load(out); L = []
    cases = [(H, fp) for H in (256, 512) for fp in (50, 25)]
    L.append(f"Receipts read: {len(S)} complete runs in `{Path(out).relative_to(HERE)}`. Seed 93683. Written by `exp52_tables.py`.\n")
    L.append("## Table 1. Every run\n")
    L.append("Updates to 80 by the three-row rule; “–” is not reached within the dose. Final levels are means of the last three rows. "
             "The length test at 40 letters is given at the last row, as the mean of the last three rows, and with its range over the last twelve rows; "
             "one row of it is a weak figure. RMS is the root mean square change of R from row to row over the last 16 rows.\n")
    L.append("| Width, f | Rule | Rate | Decay | Dose | Updates to 80, A | Updates to 80, R | Final A | Final R | Final taught | Length 40: last, mean of 3 (range of 12) | RMS of R | Diverged | Receipt |")
    L.append("|" + "---|" * 14)
    order = sorted(S, key=lambda k: (k[1], -k[2], k[0] != "S", k[3], k[4]))
    for k in order:
        s = S[k]
        L.append(f"| {WIDTH_NAME[s['H']]}, {s['fp'] / 100:.2f} | {s['rule']} | {s['rate']:g}{' (product control)' if s['product'] else ''} | {s['decay']:.4g} | {s['dose']:,} | "
                 f"{fmt(s['A80'])} | {fmt(s['R80'])} | {s['A_final']:.2f} | {s['R_final']:.2f} | {s['taught_final']:.2f} | "
                 f"{s['len40_last']:.1f}, {s['len40_mean3']:.1f} ({s['len40_range_last12'][0]:.1f}–{s['len40_range_last12'][1]:.1f}) | {s['R_rms16']:.2f} | "
                 f"{'yes, row ' + str(s['diverged_at_row']) if s['diverged'] else 'no'} | `{s['receipt']}` |")
    L.append("")
    L.append("## Table 2. Runs at 0.0003 at E41's dose (their dose is 10/3 of E41's)\n")
    L.append("| Width, f | Rule | Control | Final A at E41's dose | Final R at E41's dose | Length 40 there: last row, mean of 3 | Final A, R at the end of the run |")
    L.append("|---|---|---|---|---|---|---|")
    for k in order:
        s = S[k]
        if s["at_e41_dose"]:
            a = s["at_e41_dose"]
            L.append(f"| {WIDTH_NAME[s['H']]}, {s['fp'] / 100:.2f} | {s['rule']} | {'product' if s['product'] else 'no'} | {a['A']:.2f} | {a['R']:.2f} | {a['len40_last']:.1f}, {a['len40_mean3']:.1f} | {s['A_final']:.2f}, {s['R_final']:.2f} |")
    L.append("")
    L.append("## Table 3. The seven norms (book / l1 / input map / recurrent / lm / l2 / l3)\n")
    L.append("| Width, f | Rule | Rate | At 80 on A | At 80 on R | At the end | At E41's dose (0.0003 only) |")
    L.append("|---|---|---|---|---|---|---|")
    for k in order:
        s = S[k]
        L.append(f"| {WIDTH_NAME[s['H']]}, {s['fp'] / 100:.2f} | {s['rule']} | {s['rate']:g}{' (product control)' if s['product'] else ''} | {nrm(s['A80_norms'])} | {nrm(s['R80_norms'])} | "
                 f"{nrm(s['norms_final'])} | {nrm(s['at_e41_dose']['norms']) if s['at_e41_dose'] else ''} |")
    L.append("")
    L.append("## Table 4. Readings for each case (a rule, a width and a taught fraction)\n")
    L.append("| Width, f | Rule | Best rate on R | Best rate on A | " + " | ".join(f"{r:g} against 0.001: R, A" for r in RATES if r != 0.001) + " | Gains from its own rate |")
    L.append("|" + "---|" * (4 + 3 + 1))
    J = dict(cases={}, lags={}, controls={})
    for H, fp in cases:
        for rule in "SB":
            if not any(k[:3] == (rule, H, fp) for k in S):
                continue
            br, ba = best_rate(S, rule, H, fp, "R"), best_rate(S, rule, H, fp, "A")
            nm = lambda b: "not reached" if not b[1] else (f"{b[0]:g}" if b[0] is not None else "tie of " + ", ".join(f"{r:g}" for r in b[1])) + (
                f" (tied with {', '.join(f'{r:g}' for r in b[1] if r != b[0])})" if b[0] is not None and len(b[1]) > 1 else "")
            cmp_ = {r: (compare(S, rule, H, fp, r, "R"), compare(S, rule, H, fp, r, "A")) for r in RATES if r != 0.001}
            g = gains(S, rule, H, fp)
            L.append(f"| {WIDTH_NAME[H]}, {fp / 100:.2f} | {rule} | {nm(br)} | {nm(ba)} | "
                     + " | ".join("not run" if c[0] is None else f"{c[0]}, {c[1]}" for c in cmp_.values())
                     + f" | {'yes, at ' + ', '.join(f'{r:g}' for r in g['at_rates']) if g['gains'] else 'no'} (read on {g['read_on']}) |")
            J["cases"][f"{rule} h{H} f{fp}"] = dict(best_R=dict(named=br[0], tied=br[1], count=br[2]), best_A=dict(named=ba[0], tied=ba[1], count=ba[2]),
                                                    against_0001={f"{r:g}": c for r, c in cmp_.items()}, gains=g)
    L.append("")
    L.append("## Table 5. The lag (the sweep's updates to 80 over back-propagation's) at each width and fraction\n")
    L.append("| Width, f | Score | Lag at 0.001 | Sweep's best rate | Back-propagation's best rate | Lag at the rules' own best rates | Reading |")
    L.append("|---|---|---|---|---|---|---|")
    for H, fp in cases:
        for key in ("R", "A"):
            if ("S", H, fp, 0.001, False) not in S or ("B", H, fp, 0.001, False) not in S:
                continue
            g = lag(S, H, fp, key)
            L.append(f"| {WIDTH_NAME[H]}, {fp / 100:.2f} | {key} | {g['at_0.001']['text']} | {g['sweep_best']} | {g['backprop_best']} | {g['own']['text']} | {g['class']} |")
            J["lags"][f"h{H} f{fp} {key}"] = g
    L.append("")
    L.append("## Table 6. The control (rate × decay held at 0.00025) beside the main runs: half width, half of the pairs\n")
    L.append("| Rule | Rate | Decay | Updates to 80, A | Updates to 80, R | Class against 0.001: R, A | Final A | Final R | Length 40: last, mean of 3 | Norms at the end |")
    L.append("|" + "---|" * 10)
    for rule in "SB":
        for rate in (0.0003, 0.003):
            for product in (False, True):
                s = S.get((rule, 256, 50, rate, product))
                if s is None:
                    continue
                cR, cA = compare(S, rule, 256, 50, rate, "R", product), compare(S, rule, 256, 50, rate, "A", product)
                L.append(f"| {rule} | {rate:g} | {s['decay']:.4g}{' (control)' if product else ' (main)'} | {fmt(s['A80'])} | {fmt(s['R80'])} | {cR}, {cA} | {s['A_final']:.2f} | {s['R_final']:.2f} | "
                         f"{s['len40_last']:.1f}, {s['len40_mean3']:.1f} | {nrm(s['norms_final'])} |")
                J["controls"][f"{rule} {rate:g} {'product' if product else 'main'}"] = dict(R=cR, A=cA)
    L.append("")
    need = []
    for H, fp in cases:
        for rule in "SB":
            if (rule, H, fp, 0.003, False) not in S:
                continue
            br = best_rate(S, rule, H, fp, "R"); b = br if br[1] else best_rate(S, rule, H, fp, "A")
            if b[0] == 0.003 and not S[(rule, H, fp, 0.003, False)]["diverged"]:
                need.append(f"{rule} {H} {fp}")
    J["further_runs_called_for"] = need
    L.append(f"Further runs at 0.01 called for by the rule (best rate 0.003 on R, or on A where the case is not reached on R; a tie is not a best rate): "
             f"{', '.join(need) if need else 'none'}.\n")
    # predictions (the reviewer's, on file before any run)
    P = {}
    cases_hf = [(H, fp) for H in (256, 512) for fp in (50, 25)]
    have = lambda rule, H, fp, rate: (rule, H, fp, rate, False) in S
    if all(have("B", H, fp, r) for H, fp in cases_hf for r in (0.0003, 0.001, 0.003)):
        ok = all((lambda b: bool(b[1]) and any(r in (0.001, 0.003) for r in b[1]))(best_rate(S, "B", H, fp, "R")) for H, fp in cases_hf)
        P["P52.1"] = dict(statement="back-propagation's best rate on R is 0.001 or 0.003, or a tie that includes one of them, at all four", outcome=ok,
                          detail={f"h{H} f{fp}": best_rate(S, "B", H, fp, "R")[1] for H, fp in cases_hf})
        g = {f"h{H} f{fp}": gains(S, "B", H, fp)["gains"] for H, fp in cases_hf}
        P["P52.5"] = dict(statement="back-propagation gains from its own rate at none of the four", outcome=not any(g.values()), detail=g)
    if all(have("S", H, fp, r) for H, fp in cases_hf for r in (0.0003, 0.001, 0.003)):
        g = {f"h{H} f{fp}": gains(S, "S", H, fp)["gains"] for H, fp in cases_hf}
        P["P52.2"] = dict(statement="the sweep gains from its own rate at one or more of the four", outcome=any(g.values()), detail=g)
        fin = {f"{k[3]:g}": S[k]["R_final"] for k in S if k[:3] == ("S", 512, 25) and not k[4]}
        P["P52.3"] = dict(statement="present width, f 0.25: the sweep's mean R over the last three rows is at least 85 at some rate", outcome=max(fin.values()) >= 85, detail=fin)
        lg = lag(S, 256, 25, "R")
        P["P52.4"] = dict(statement="half width, f 0.25: the lag on R at the rules' own rates is smaller than at 0.001", outcome=lg["class"] in ("smaller", "gone"),
                          detail=dict(at_0001=lg["at_0.001"]["text"], own=lg["own"]["text"], reading=lg["class"]))
        a, b = S[("S", 256, 25, 0.0003, False)]["R_rms16"], S[("S", 256, 25, 0.001, False)]["R_rms16"]
        P["P52.6"] = dict(statement="half width, f 0.25: the sweep's R is steadier at 0.0003 than at 0.001 (RMS change over the last 16 rows)", outcome=a < b,
                          detail={"0.0003": a, "0.001": b})
    J["predictions"] = P
    if P:
        L.append("## Table 7. The predictions\n")
        L.append("| | Statement | Outcome | Figures |\n|---|---|---|---|")
        for k in sorted(P):
            L.append(f"| {k} | {P[k]['statement']} | {'true' if P[k]['outcome'] else 'false'} | {json.dumps(P[k]['detail'])} |")
        L.append("")
    J["runs"] = {f"{k[0]} h{k[1]} f{k[2]} lr{k[3]:g}{' product' if k[4] else ''}": v for k, v in S.items()}
    (Path(out) / "E52_TABLES.md").write_text("\n".join(L) + "\n")
    (Path(out) / "E52_SUMMARY.json").write_text(json.dumps(J, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv[1:]
    write(Path(a[a.index("--out") + 1]) if "--out" in a else HERE / "exp45" / "e52")
