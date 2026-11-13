"""E27 scorer, written and fixture-tested before the run (brief revision 2, step 4). Reads only exp45/E27_MEASURE_<seed>.json.

Judges P27.1 to P27.5 exactly as registered in E27_PREDICTIONS.md, at the places its amendment names, and computes supplements S1 to S6
beside them. "Holds" means in at least two of three seeds. Per seed a prediction holds, misses, is void (P27.2 only: either usable <= 0), or
is not judgeable (a file it needs is absent). Overall: holds if at least two seeds hold; misses if two holding seeds are no longer possible;
otherwise not judgeable.

Operationalisations the registered words leave open are NOT chosen here: where the amendment says "near 1" (S1, S3) both readings are
reported, inside the S4 null's range and within 0.8 to 1.25, and a disagreement between them is reported as such.

  python3 e27_score.py                 scores seeds 73129 93683 116089, writes exp45/E27_SCORE.json
  python3 e27_score.py fixtures        fabricated inputs, a hold and a miss of each prediction; writes exp45/E27_SCORER_FIXTURES.json"""
import sys, json, copy, pathlib

SEEDS = (73129, 93683, 116089)
SB_ANS = "two-bank born by the sweep, answer targets"
COLLAPSED = "two-bank born by the sweep, every byte (collapsed)"
BALANCED = "two-bank born by the sweep, balanced"
BP_EVERY = "two-bank born by backprop, every byte"
SWEEP_LIVED = ("sweep -> sweep", "backprop -> sweep")
BACKPROP_LIVED = ("sweep -> backprop", "backprop -> backprop")
WITHIN_BIRTH = (("sweep -> sweep", "sweep -> backprop"), ("backprop -> sweep", "backprop -> backprop"))   # (sweep-lived, backprop-lived)


def overall(per_seed):
    v = list(per_seed.values()); h = v.count("holds"); open_ = v.count("not judgeable") + (len(SEEDS) - len(v))
    if h >= 2:
        return "holds"
    if h + open_ < 2:
        nv = v.count("void")
        return "misses" if v.count("misses") >= 2 or nv == 0 else f"does not hold (void in {nv} seed{'s' if nv > 1 else ''})"
    return "not judgeable"


def get(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def p1(m):
    a = get(m, "A", SB_ANS, "all_parameters", "amplification")
    return ("not judgeable", None) if a is None else ("holds" if a >= 1.5 else "misses", dict(amplification=a))


def p2(m):
    s, e = get(m, "A", SB_ANS, "all_parameters", "sign", "usable"), get(m, "A", SB_ANS, "all_parameters", "exact", "usable")
    if s is None or e is None:
        return "not judgeable", None
    if s <= 0 or e <= 0:
        return "void", dict(sign=s, exact=e)
    return ("holds" if s <= 0.5 * e else "misses"), dict(sign=s, exact=e, ratio=s / e)


def p3(m):
    c, b = get(m, "A", COLLAPSED, "all_parameters", "sign", "flood"), get(m, "A", BALANCED, "all_parameters", "sign", "flood")
    if c is None or b is None:
        return "not judgeable", None
    return ("holds" if c >= 2 * b else "misses"), dict(collapsed=c, balanced=b, ratio=c / b)


def p4_arm(m, arm):
    u, s = get(m, "B", arm, "untaught_plus_minus", "all_parameters"), get(m, "B", arm, "single_operations", "all_parameters")
    return None if u is None or s is None else dict(untaught=u, single=s, gap=s - u, shows=(u <= s - 0.10))


def p4(m):
    arms = {a: p4_arm(m, a) for a in SWEEP_LIVED}
    if any(v is None for v in arms.values()):
        return "not judgeable", dict(arms=arms, partial="one arm only" if any(v is not None for v in arms.values()) else "no arm")
    return ("holds" if all(v["shows"] for v in arms.values()) else "misses"), dict(arms=arms)


def p5(m):
    comps = {}
    for sw, bp in WITHIN_BIRTH:
        us, ub = get(m, "B", sw, "untaught_plus_minus", "all_parameters"), get(m, "B", bp, "untaught_plus_minus", "all_parameters")
        comps[f"{sw} vs {bp}"] = None if us is None or ub is None else dict(sweep_lived=us, backprop_lived=ub, difference=ub - us, shows=(us <= ub - 0.05))
    if any(v is None for v in comps.values()):
        return "not judgeable", dict(comparisons=comps)
    return ("holds" if all(v["shows"] for v in comps.values()) else "misses"), dict(comparisons=comps)


PREDICTIONS = dict(P27_1=p1, P27_2=p2, P27_3=p3, P27_4=p4, P27_5=p5)


def near1(amp, null):
    if amp is None:
        return None
    return dict(value=amp, inside_null=(None if not null else null["min"] <= amp <= null["max"]), within_0_8_to_1_25=(0.8 <= amp <= 1.25))


def supplements(ms):
    S = {}
    # S1: all parameters except the top layer, on the sweep-born answer-target file; >= 1.5 in two seeds reads as flood present
    s1 = {s: get(m, "A", SB_ANS, "all_parameters_except_top", "amplification") for s, m in ms.items()}
    n = sum(1 for v in s1.values() if v is not None and v >= 1.5)
    S["S1"] = dict(except_top_amplification=s1, seeds_at_least_1_5=n,
                   reading=("flood present at those weights" if n >= 2 else "flood not present below the top layer at those weights" if sum(v is not None for v in s1.values()) - n >= 2 else "not judgeable"),
                   near_1={s: near1(v, get(ms[s], "A", SB_ANS, "null", "all_parameters_except_top")) for s, v in s1.items()},
                   per_tensor={s: {k: t.get("amplification") for k, t in (get(m, "A", SB_ANS, "per_tensor") or {}).items()} for s, m in ms.items()})
    # S2, S4, S5 for every A file
    S["S2"], S["S4"], S["S5"] = {}, {}, {}
    for s, m in ms.items():
        for label, a in (m.get("A") or {}).items():
            for sname in ("all_parameters", "all_parameters_except_top"):
                x = a.get(sname, {}); nl = get(a, "null", sname)
                S["S2"].setdefault(label, {}).setdefault(sname, {})[s] = dict(letters=x.get("amplification_letters"), structural=x.get("amplification_structural"))
                S["S4"].setdefault(label, {}).setdefault(sname, {})[s] = dict(amplification=x.get("amplification"), null=(None if not nl else [nl["min"], nl["max"]]),
                                                                              inside_null=(None if not nl or x.get("amplification") is None else nl["min"] <= x["amplification"] <= nl["max"]),
                                                                              letters_inside_null=(None if not nl or x.get("amplification_letters") is None else nl["min"] <= x["amplification_letters"] <= nl["max"]))
                S["S5"].setdefault(label, {}).setdefault(sname, {})[s] = dict(usable_exact=get(x, "exact", "usable"), usable_sign=get(x, "sign", "usable"), **(x.get("usable_baselines") or {}))
    # S3: the file born by backprop under every-byte targets
    S["S3"] = {sname: {s: near1(get(m, "A", BP_EVERY, sname, "amplification"), get(m, "A", BP_EVERY, "null", sname)) for s, m in ms.items()} for sname in ("all_parameters", "all_parameters_except_top")}
    S["S3"]["letters_all_parameters_except_top"] = {s: near1(get(m, "A", BP_EVERY, "all_parameters_except_top", "amplification_letters"), get(m, "A", BP_EVERY, "null", "all_parameters_except_top")) for s, m in ms.items()}
    # S6: untaught minus taught (same format), per arm; the reading needs it more negative on sweep-lived than on backprop-lived files
    s6 = {}
    for s, m in ms.items():
        c = {}
        for arm in SWEEP_LIVED + BACKPROP_LIVED:
            u, t = get(m, "B", arm, "untaught_plus_minus", "all_parameters"), get(m, "B", arm, "taught_pairs_new_instances", "all_parameters")
            c[arm] = None if u is None or t is None else u - t
        within = {f"{sw} vs {bp}": (None if c[sw] is None or c[bp] is None else dict(sweep_lived=c[sw], backprop_lived=c[bp], sweep_lived_gap_larger=c[sw] < c[bp])) for sw, bp in WITHIN_BIRTH}
        sw_ = [c[a] for a in SWEEP_LIVED if c[a] is not None]; bp_ = [c[a] for a in BACKPROP_LIVED if c[a] is not None]
        across = None
        if sw_ and bp_:
            across = dict(sweep_lived_mean=sum(sw_) / len(sw_), backprop_lived_mean=sum(bp_) / len(bp_), arms_present=[a for a in c if c[a] is not None],
                          sweep_lived_gap_larger=sum(sw_) / len(sw_) < sum(bp_) / len(bp_),
                          note=("across births: the within-birth form needs files that are absent" if any(v is None for v in within.values()) else "all four arms"))
        s6[s] = dict(untaught_minus_taught=c, within_birth=within, available_arms=across)
    S["S6"] = s6
    return S


def readings(V, S):
    out = []
    a, b = V["P27_1"]["overall"], V["P27_2"]["overall"]
    if a == "holds" and b == "holds":
        out.append("P27.1 and P27.2 hold: flood real at the point where a sweep-born network would meet every-byte targets.")
    elif a == "misses" and b == "misses":
        s1n = [v for v in S["S1"]["near_1"].values() if v]; s3n = [v for v in S["S3"]["all_parameters_except_top"].values() if v]
        def two(xs, k):
            return sum(1 for v in xs if v[k]) >= 2
        ops = {k: two(s1n, k) and two(s3n, k) for k in ("inside_null", "within_0_8_to_1_25")}
        if all(ops.values()):
            out.append("P27.1 and P27.2 miss, S1 and S3 near 1 under both operationalisations: no amplification at these saved weights.")
        elif not any(ops.values()):
            out.append("P27.1 and P27.2 miss, but S1 or S3 is not near 1 under either operationalisation: registered figures and supplements disagree; both reported.")
        else:
            out.append(f"P27.1 and P27.2 miss; whether S1 and S3 are 'near 1' depends on the operationalisation ({ops}); the ambiguity is reported, not resolved.")
    else:
        out.append(f"P27.1 {a}, P27.2 {b}: reported as they stand; they do not agree" + ("" if a != b else "") + ".")
    if (a == "holds") != (S["S1"]["seeds_at_least_1_5"] >= 2):
        out.append(f"P27.1 ({a}) and S1 ({S['S1']['reading']}) disagree; both reported.")
    out.append(f"P27.3 {V['P27_3']['overall']}: judged and reported; nothing is read from it (amendment).")
    out.append(f"P27.4 {V['P27_4']['overall']}, P27.5 {V['P27_5']['overall']}: neither triggers the composition reading; composition is read from S6 only (amendment).")
    return out


def score(ms):
    V = {}
    for name, f in PREDICTIONS.items():
        per, fig = {}, {}
        for s, m in ms.items():
            per[s], fig[s] = f(m)
        V[name] = dict(per_seed=per, figures=fig, overall=overall(per))
    S = supplements(ms)
    return dict(seeds=sorted(ms), absent={s: m.get("absent", []) for s, m in ms.items()}, predictions=V, supplements=S, readings=readings(V, S))


# ---------------------------------------------------------------- fixtures: fabricated inputs
def fab(amp=1.0, ex_top=1.0, us_s=0.5, us_e=0.6, col=1.0, bal=1.0, single=0.7, untaught=0.65, taught=0.66, life_shift=0.0, arms=("sweep -> sweep", "backprop -> backprop", "sweep -> backprop", "backprop -> sweep")):
    null = dict(min=0.97, max=1.03, mean=1.0)
    def af(a, e, fs=1.0):
        return dict(all_parameters=dict(amplification=a, amplification_letters=a, amplification_structural=1.0, exact=dict(flood=1.0, usable=us_e), sign=dict(flood=fs, usable=us_s), usable_baselines=dict(sign_answer_vs_exact_answer=0.8, sign_every_vs_sign_answer=0.9)),
                    all_parameters_except_top=dict(amplification=e, amplification_letters=e, amplification_structural=1.0, exact=dict(flood=1.0, usable=us_e), sign=dict(flood=1.0, usable=us_s)),
                    per_tensor={"l1.weight": dict(amplification=e), "l3.weight": dict(amplification=1.0)}, null=dict(all_parameters=null, all_parameters_except_top=null))
    A = {SB_ANS: af(amp, ex_top), COLLAPSED: af(1.0, 1.0, col), BALANCED: af(1.0, 1.0, bal), BP_EVERY: af(1.0, 1.0)}
    B = {}
    for arm in arms:
        sweep = arm.endswith("sweep")
        B[arm] = dict(single_operations=dict(all_parameters=single), untaught_plus_minus=dict(all_parameters=untaught - (life_shift if sweep else 0.0)),
                      taught_pairs_new_instances=dict(all_parameters=taught))
    return dict(A=A, B=B, absent=[])


def fixtures():
    cases = {
        "P27_1 holds": (dict(amp=1.6), "P27_1", "holds"), "P27_1 misses": (dict(amp=1.49), "P27_1", "misses"),
        "P27_2 holds": (dict(us_s=0.29, us_e=0.6), "P27_2", "holds"), "P27_2 misses": (dict(us_s=0.31, us_e=0.6), "P27_2", "misses"),
        "P27_2 void": (dict(us_s=-0.1, us_e=0.6), "P27_2", "does not hold (void in 3 seeds)"),
        "P27_3 holds": (dict(col=2.0, bal=1.0), "P27_3", "holds"), "P27_3 misses": (dict(col=1.9, bal=1.0), "P27_3", "misses"),
        "P27_4 holds": (dict(single=0.7, untaught=0.59), "P27_4", "holds"), "P27_4 misses": (dict(single=0.7, untaught=0.61), "P27_4", "misses"),
        "P27_4 one arm absent": (dict(single=0.7, untaught=0.5, arms=("sweep -> backprop", "backprop -> sweep")), "P27_4", "not judgeable"),
        "P27_5 holds": (dict(life_shift=0.05), "P27_5", "holds"), "P27_5 misses": (dict(life_shift=0.04), "P27_5", "misses"),
        "P27_5 lives absent": (dict(life_shift=0.2, arms=("sweep -> backprop", "backprop -> sweep")), "P27_5", "not judgeable"),
    }
    out, ok = {}, True
    for name, (kw, pred, want) in cases.items():
        r = score({s: fab(**kw) for s in SEEDS}); got = r["predictions"][pred]["overall"]
        out[name] = dict(want=want, got=got, per_seed=r["predictions"][pred]["per_seed"], ok=(got == want)); ok = ok and got == want
    # two of three seeds is enough, one is not
    m = {73129: fab(amp=1.6), 93683: fab(amp=1.6), 116089: fab(amp=1.0)}; g = score(m)["predictions"]["P27_1"]["overall"]
    out["P27_1 two of three"] = dict(want="holds", got=g, ok=g == "holds"); ok = ok and g == "holds"
    m = {73129: fab(amp=1.6), 93683: fab(amp=1.0), 116089: fab(amp=1.0)}; g = score(m)["predictions"]["P27_1"]["overall"]
    out["P27_1 one of three"] = dict(want="misses", got=g, ok=g == "misses"); ok = ok and g == "misses"
    # P27.4 needs both sweep-lived arms in a seed: one arm showing it is not enough
    m = fab(single=0.7, untaught=0.59); m["B"]["sweep -> sweep"]["untaught_plus_minus"]["all_parameters"] = 0.65
    g = score({s: copy.deepcopy(m) for s in SEEDS})["predictions"]["P27_4"]["overall"]
    out["P27_4 one arm shows it"] = dict(want="misses", got=g, ok=g == "misses"); ok = ok and g == "misses"
    # P27.5 needs both within-birth comparisons
    m = fab(life_shift=0.0); m["B"]["sweep -> sweep"]["untaught_plus_minus"]["all_parameters"] = 0.5
    g = score({s: copy.deepcopy(m) for s in SEEDS})["predictions"]["P27_5"]["overall"]
    out["P27_5 one comparison shows it"] = dict(want="misses", got=g, ok=g == "misses"); ok = ok and g == "misses"
    # supplements: S1 reading, S4 null membership, S6 direction
    r = score({s: fab(ex_top=1.6) for s in SEEDS}); out["S1 present"] = dict(got=r["supplements"]["S1"]["reading"], ok=r["supplements"]["S1"]["reading"] == "flood present at those weights")
    r2 = score({s: fab(ex_top=1.1) for s in SEEDS}); out["S1 absent"] = dict(got=r2["supplements"]["S1"]["reading"], ok=r2["supplements"]["S1"]["reading"].startswith("flood not present"))
    out["S4 inside / outside"] = dict(ok=(r2["supplements"]["S4"][SB_ANS]["all_parameters"][73129]["inside_null"] is True and score({s: fab(amp=1.2) for s in SEEDS})["supplements"]["S4"][SB_ANS]["all_parameters"][73129]["inside_null"] is False))
    r3 = score({s: fab(life_shift=0.03) for s in SEEDS}); out["S6 sweep-lived gap larger"] = dict(ok=all(v["sweep_lived_gap_larger"] for v in r3["supplements"]["S6"][73129]["within_birth"].values()))
    r4 = score({s: fab(life_shift=-0.03) for s in SEEDS}); out["S6 sweep-lived gap smaller"] = dict(ok=not any(v["sweep_lived_gap_larger"] for v in r4["supplements"]["S6"][73129]["within_birth"].values()))
    # the no-amplification reading fires only when both operationalisations agree
    r5 = score({s: fab(amp=1.0, us_s=0.5, us_e=0.6) for s in SEEDS}); out["reading: no amplification"] = dict(got=r5["readings"][0], ok=r5["readings"][0].startswith("P27.1 and P27.2 miss, S1 and S3 near 1"))
    r6 = score({s: fab(amp=1.0, ex_top=1.1, us_s=0.5, us_e=0.6) for s in SEEDS}); out["reading: near-1 ambiguous"] = dict(got=r6["readings"][0], ok="depends on the operationalisation" in r6["readings"][0])
    for k in ("S1 present", "S1 absent", "S4 inside / outside", "S6 sweep-lived gap larger", "S6 sweep-lived gap smaller", "reading: no amplification", "reading: near-1 ambiguous"):
        ok = ok and out[k]["ok"]
    out["pass"] = ok
    pathlib.Path("exp45/E27_SCORER_FIXTURES.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v.get("ok") if isinstance(v, dict) else v for k, v in out.items()}, indent=1))
    return ok


def main():
    ms = {}
    for s in SEEDS:
        p = pathlib.Path(f"exp45/E27_MEASURE_{s}.json")
        if p.exists():
            ms[s] = json.loads(p.read_text())
    r = score(ms); r["read"] = [f"exp45/E27_MEASURE_{s}.json" for s in ms]
    pathlib.Path("exp45/E27_SCORE.json").write_text(json.dumps(r, indent=1, default=str))
    for k, v in r["predictions"].items():
        print(k, v["overall"], v["per_seed"])
    print("S1", r["supplements"]["S1"]["reading"]); [print(" -", x) for x in r["readings"]]


if __name__ == "__main__":
    sys.exit(0 if fixtures() else 1) if sys.argv[1:] == ["fixtures"] else main()
