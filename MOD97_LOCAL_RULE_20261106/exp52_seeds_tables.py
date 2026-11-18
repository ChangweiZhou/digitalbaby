"""E52, the other two seeds (decision 19): readings at rates 0.001 (E41's receipts, exp45/e41/) and 0.003 (exp45/e52_seeds/), with the
margin of every reading in scored rows, and the same readings for seed 93683 restricted to these two rates (exp45/e52/). Read-only;
standard library.

    python3 exp52_seeds_tables.py      # writes exp45/e52_seeds/E52_SEEDS_TABLES.md and E52_SEEDS_SUMMARY.json

The rules are the brief's (as in exp52_tables.py): updates to 80 by the three-row rule; best rate = fewest updates, counts within one
row (256) tied, a tie that includes 0.001 given to 0.001; faster / the same / slower by exp41_scale.describe_step with the run at 0.001
in the narrower run's place; the rule gains if faster on R at 0.003 and not slower on A (on A alone if neither rate reaches 80 on R);
the lag at its own rates is smaller (larger) below 2/3 (above 1.5) of the lag at 0.001, gone if smaller and at most 1.0.

Margin of a reading: the fewest whole rows by which one count to 80 (of one run, on one score) must move for the reading to change.
A count c moves to c + 256 k or c - 256 k (not below 256, the first row after update 0); a count beyond the run's last recordable row (dose - 512) becomes 'not
reached'; a count that is not reached becomes reached at the last recordable row less 256 (k - 1). Every reading of the width and
fraction is recomputed from the moved counts, best rates included. 'none within 200' when no move of up to 200 rows changes it.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
E41 = HERE / "exp45" / "e41"
E52 = HERE / "exp45" / "e52"
SEEDS_DIR = HERE / "exp45" / "e52_seeds"
ROW = 256
RATES = (0.001, 0.003)
CASES = ((256, 50), (256, 25), (512, 25))
WN = {256: "half", 512: "present"}
KMAX = 200


def first_row(curve, key, th=80):
    v = [r[key] for r in curve]
    for i in range(len(v) - 2):
        if all(x >= th for x in v[i:i + 3]):
            return curve[i]["step"]
    return None


def describe_step(xn, xw, Dn, Dw):          # exp41_scale.describe_step, verbatim
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


def receipt(seed, rule, H, fp, rate):
    if rate == 0.001:
        p = (E52 / f"e52_{rule}_h{H}_f{fp}_lr0.001_93683.json") if seed == 93683 else (E41 / f"e41_{rule}_h{H}_f{fp}_{seed}.json")
    else:
        p = (E52 if seed == 93683 else SEEDS_DIR) / f"e52_{rule}_h{H}_f{fp}_lr0.003_{seed}.json"
    if not p.exists():
        return None, p
    r = json.loads(p.read_text())
    return (r if r.get("complete") else None), p


def summarize(r, p):
    c = r["curve"]
    s = dict(receipt=str(p.relative_to(HERE)), dose=r["dose"], D=r["dose"] - 512, tensors_sha256=r.get("tensors_sha256"),
             A80=first_row(c, "A"), R80=first_row(c, "R"), A_final=mean([x["A"] for x in c[-3:]]), R_final=mean([x["R"] for x in c[-3:]]),
             taught_final=mean([x["taught"] for x in c[-3:]]), resumed=[x["at_update"] for x in r.get("resumed", [])])
    if "len40" in c[-1]:
        s.update(len40_last=c[-1]["len40"], len40_mean3=mean([x["len40"] for x in c[-3:]]),
                 len40_range12=[min(x["len40"] for x in c[-12:]), max(x["len40"] for x in c[-12:])], norms_final=c[-1]["norms"])
    return s


# ---------------------------------------------------------------- readings from a table of counts
def best(counts, D, rule, key):
    got = {rate: counts[(rule, rate, key)] for rate in RATES if counts[(rule, rate, key)] is not None}
    if not got:
        return None, [], None
    lo = min(got.values()); tied = sorted(k for k, v in got.items() if v - lo <= ROW)
    if len(tied) == 1:
        return tied[0], tied, got[tied[0]]
    if 0.001 in tied:
        return 0.001, tied, got[0.001]
    return None, tied, lo


def cls(counts, D, rule, key):
    return WORD[describe_step(counts[(rule, 0.001, key)], counts[(rule, 0.003, key)], D[(rule, 0.001)], D[(rule, 0.003)])]


def gains(counts, D, rule):
    reached_R = any(counts[(rule, r, "R")] is not None for r in RATES)
    if reached_R:
        return cls(counts, D, rule, "R") == "faster" and cls(counts, D, rule, "A") != "slower"
    return cls(counts, D, rule, "A") == "faster"


def lag(counts, D, key):
    xs, xb = counts[("S", 0.001, key)], counts[("B", 0.001, key)]
    if xs and xb:
        a = ("value", xs / xb)
    elif xb:
        a = ("lower", D[("S", 0.001)] / xb)
    elif xs:
        a = ("upper", xs / D[("B", 0.001)])
    else:
        a = ("none", None)
    bs, bb = best(counts, D, "S", key), best(counts, D, "B", key)
    ys, yb = bs[2], bb[2]
    if ys and yb:
        o = ("value", ys / yb)
    elif yb:
        o = ("lower", max(D[("S", r)] for r in RATES) / yb)
    elif ys:
        o = ("upper", ys / max(D[("B", r)] for r in RATES))
    else:
        o = ("none", None)
    c = "not formed" if a[0] == "none" or o[0] == "none" else "not decided"
    if a[0] == "value" and o[0] == "value":
        q = o[1] / a[1]
        c = ("gone" if o[1] <= 1.0 else "smaller") if q < 2 / 3 else "larger" if q > 1.5 else "the same within the scatter of one seed"
    elif a[0] == "lower" and o[0] == "value" and o[1] < 2 / 3 * a[1]:
        c = "gone" if o[1] <= 1.0 else "smaller"
    elif a[0] == "value" and o[0] == "lower" and o[1] > 1.5 * a[1]:
        c = "larger"
    return dict(at_0001=a, own=o, sweep_best=bs, backprop_best=bb, reading=c)


def readings(counts, D):
    out = {}
    for rule in "SB":
        for key in "RA":
            b = best(counts, D, rule, key)
            out[f"{rule} best {key}"] = "not reached" if not b[1] else (f"{b[0]:g}" if b[0] is not None else "tie " + ",".join(f"{r:g}" for r in b[1]))
            out[f"{rule} 0.003 vs 0.001 {key}"] = cls(counts, D, rule, key)
        out[f"{rule} gains"] = "yes" if gains(counts, D, rule) else "no"
    for key in "RA":
        out[f"lag {key}"] = lag(counts, D, key)["reading"]
    return out


def moves(c, Dmax, k):
    res = []
    if c is None:
        v = Dmax - ROW * (k - 1)
        if v >= ROW:
            res.append(v)
    else:
        for v in (c + ROW * k, c - ROW * k):
            if v < ROW:
                continue
            res.append(None if v > Dmax else v)
    return res


def margins(counts, D):
    base = readings(counts, D)
    m = {k: None for k in base}
    for k in range(1, KMAX + 1):
        for var in counts:
            for v in moves(counts[var], D[var[:2]], k):
                c2 = dict(counts); c2[var] = v
                r2 = readings(c2, D)
                for name in base:
                    if m[name] is None and r2[name] != base[name]:
                        m[name] = dict(rows=k, move=f"{var[0]} at {var[1]:g} on {var[2]}: {counts[var]} -> {v}", to=r2[name])
        if all(v is not None for v in m.values()):
            break
    return base, m


def fmt(x):
    return "–" if x is None else f"{x:,}"


def lagtext(t):
    kind, v = t
    import math       # a lower bound is rounded down and an upper bound up, so that the printed bound stays true
    return {"value": lambda: f"{v:.2f}", "lower": lambda: f"above {math.floor(v * 100) / 100:.2f}",
            "upper": lambda: f"below {math.ceil(v * 100) / 100:.2f}", "none": lambda: "not formed"}[kind]()


def main():
    L = ["# E52, the other two seeds: tables and readings at rates 0.001 and 0.003\n",
         "Written by `exp52_seeds_tables.py` from the receipts named in Table 1. Rate 0.001: E41's receipts for seeds 73129 and 116089 "
         "(`exp45/e41/`), E52's for seed 93683 (`exp45/e52/`, equal to E41's by check 1b). Rate 0.003: `exp45/e52_seeds/` for seeds 73129 "
         "and 116089, `exp45/e52/` for seed 93683. Seed 93683's readings here use only the rates 0.001 and 0.003; its full readings, with "
         "0.0003 and 0.01, are in `exp45/e52/E52_TABLES.md`.\n"]
    J = dict(runs={}, readings={}, missing=[])
    L.append("## Table 1. Every run\n")
    L.append("| Seed | Width, f | Rule | Rate | Dose | Updates to 80, A | Updates to 80, R | Final A | Final R | Final taught | Length 40: last, mean of 3 (range of 12) | Norms at the end (book / l1 / input / recurrent / lm / l2 / l3) | Resumed at | Receipt |")
    L.append("|" + "---|" * 14)
    S = {}
    for seed in (73129, 116089, 93683):
        for H, fp in CASES:
            for rule in "SB":
                for rate in RATES:
                    r, p = receipt(seed, rule, H, fp, rate)
                    if r is None:
                        J["missing"].append(str(p.relative_to(HERE))); continue
                    s = summarize(r, p); S[(seed, H, fp, rule, rate)] = s
                    J["runs"][f"{seed} {rule} h{H} f{fp} lr{rate:g}"] = s
                    ln = f"{s['len40_last']:.1f}, {s['len40_mean3']:.1f} ({s['len40_range12'][0]:.1f}–{s['len40_range12'][1]:.1f})" if "len40_last" in s else "not scored (E41)"
                    nm = " / ".join(f"{s['norms_final'][g]:.1f}" for g in ("book", "l1", "input_map", "recurrent", "lm", "l2", "l3")) if "norms_final" in s else "not recorded (E41)"
                    L.append(f"| {seed} | {WN[H]}, {fp / 100:.2f} | {rule} | {rate:g} | {s['dose']:,} | {fmt(s['A80'])} | {fmt(s['R80'])} | {s['A_final']:.2f} | "
                             f"{s['R_final']:.2f} | {s['taught_final']:.2f} | {ln} | {nm} | {', '.join(map(str, s['resumed'])) or '–'} | `{s['receipt']}` |")
    L.append("")
    L.append("## Table 2. Readings, with the margin of each in rows\n")
    L.append("Each cell: the reading, then in brackets the fewest rows by which one count would have to move to change it, and the reading it would become.\n")
    L.append("| Seed | Width, f | Best rate on R: S; B | Best rate on A: S; B | 0.003 against 0.001, S: R; A | 0.003 against 0.001, B: R; A | S gains | B gains | Lag on R: at 0.001 → own rates | Lag on A: at 0.001 → own rates |")
    L.append("|" + "---|" * 10)
    for seed in (73129, 116089, 93683):
        for H, fp in CASES:
            keys = [(rule, rate) for rule in "SB" for rate in RATES]
            if not all((seed, H, fp, rule, rate) in S for rule, rate in keys):
                continue
            counts = {(rule, rate, key): S[(seed, H, fp, rule, rate)][f"{key}80"] for rule, rate in keys for key in "RA"}
            D = {(rule, rate): S[(seed, H, fp, rule, rate)]["D"] for rule, rate in keys}
            base, mg = margins(counts, D)
            lg = {key: lag(counts, D, key) for key in "RA"}
            cell = lambda name: f"{base[name]} ({'none within 200' if mg[name] is None else str(mg[name]['rows']) + ' → ' + mg[name]['to']})"
            L.append(f"| {seed} | {WN[H]}, {fp / 100:.2f} | {cell('S best R')}; {cell('B best R')} | {cell('S best A')}; {cell('B best A')} | "
                     f"{cell('S 0.003 vs 0.001 R')}; {cell('S 0.003 vs 0.001 A')} | {cell('B 0.003 vs 0.001 R')}; {cell('B 0.003 vs 0.001 A')} | "
                     f"{cell('S gains')} | {cell('B gains')} | "
                     f"{lagtext(lg['R']['at_0001'])} → {lagtext(lg['R']['own'])}: {cell('lag R')} | {lagtext(lg['A']['at_0001'])} → {lagtext(lg['A']['own'])}: {cell('lag A')} |")
            J["readings"][f"{seed} h{H} f{fp}"] = dict(readings=base, margins=mg, counts={f"{k[0]} {k[1]:g} {k[2]}": v for k, v in counts.items()},
                                                       lag={k: dict(at_0001=v["at_0001"], own=v["own"], reading=v["reading"]) for k, v in lg.items()})
    L.append("")
    L.append("## Table 3. Three seeds side by side (readings only)\n")
    L.append("| Width, f | Reading | 73129 | 116089 | 93683 |\n|---|---|---|---|---|")
    for H, fp in CASES:
        for name in ("S gains", "B gains", "lag R", "lag A"):
            row = []
            for seed in (73129, 116089, 93683):
                x = J["readings"].get(f"{seed} h{H} f{fp}")
                row.append("–" if x is None else x["readings"][name])
            L.append(f"| {WN[H]}, {fp / 100:.2f} | {name} | " + " | ".join(row) + " |")
    L.append("")
    if J["missing"]:
        L.append("Receipts missing or incomplete: " + ", ".join(f"`{m}`" for m in J["missing"]) + "\n")
    (SEEDS_DIR / "E52_SEEDS_TABLES.md").write_text("\n".join(L) + "\n")
    (SEEDS_DIR / "E52_SEEDS_SUMMARY.json").write_text(json.dumps(J, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
