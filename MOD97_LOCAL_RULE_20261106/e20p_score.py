"""E20-P: the seven predictions of E20P_PREDICTIONS.md judged from exp45/E20P_e16_life_D_cm6_<seed>.json. Writes exp45/E20P_VERDICTS.json and a table."""
import json, pathlib
SEEDS = (73129, 93683, 116089); PAIRS = ("+-", "-+", "><", "<>"); PM = ("+-", "-+")
R = {s: json.loads(pathlib.Path(f"exp45/E20P_e16_life_D_cm6_{s}.json").read_text()) for s in SEEDS}
pct = lambda k, n: round(100 * k / n, 1) if n else None
def pool(s, key, pairs=PAIRS, panel="twins"):
    k = sum(R[s]["probe1"][f"{panel} {p}"][key]["k"] for p in pairs); n = sum(R[s]["probe1"][f"{panel} {p}"][key]["n"] for p in pairs)
    return k, n
V = {}; rows = {}
for s in SEEDS:
    r = R[s]; p1, p2, p3 = r["probe1"], r["probe2"], r["probe3"]; d = {}
    ft = [p1[f"twins {p}"]["failed_twins"] for p in PAIRS]
    d["failed_twins"] = sum(f["n"] for f in ft); d["failed_with_wrong_intermediate"] = sum(f["an_intermediate_wrong"] for f in ft)
    d["failed_would_pass_with_true_intermediate"] = sum(f["would_pass_with_true_intermediate"] for f in ft)
    d["P1_pct"] = pct(d["failed_with_wrong_intermediate"], d["failed_twins"])
    k, n = pool(s, "ending_given_true_intermediate_o2_in_view"); d["P2_pct"] = pct(k, n); d["P2_n"] = n
    k2, n2 = pool(s, "ending_given_true_intermediate_o2_out_of_view"); d["P3_out_pct"] = pct(k2, n2); d["P3_n"] = n2; d["P3_gap"] = round(d["P2_pct"] - d["P3_out_pct"], 1)
    k, n = pool(s, "ending_follows_own_wrong_intermediate"); d["P4_pct"] = pct(k, n); d["P4_n"] = n
    t = p3["twin_items"]; d["P5_pct"] = pct(t["a"]["both"], t["a"]["n"]); d["P5_shuffled"] = pct(t["a_shuffled_labels"]["both"], t["a"]["n"]); d["P5_untrained"] = pct(t["a_untrained_net"]["both"], t["a"]["n"])
    d["controls_in_order"] = d["P5_shuffled"] < 5 and d["P5_untrained"] < 10
    d["readout"] = {k_: pct(t[k_]["both"], t[k_]["n"]) for k_ in ("a", "b", "c", "i")}; d["readout_untrained"] = {k_: pct(t[k_ + "_untrained_net"]["both"], t[k_]["n"]) for k_ in ("a", "b", "c", "i")}
    x = p3["first_operand_read_out_by_intermediate_written"]
    ro, nro = x["read out, intermediate right"] + x["read out, intermediate wrong"], x["not read out, intermediate right"] + x["not read out, intermediate wrong"]
    d["P6_inter_right_where_read_out"] = pct(x["read out, intermediate right"], ro); d["P6_inter_right_where_not"] = pct(x["not read out, intermediate right"], nro)
    d["P6_gap"] = None if None in (d["P6_inter_right_where_read_out"], d["P6_inter_right_where_not"]) else round(d["P6_inter_right_where_read_out"] - d["P6_inter_right_where_not"], 1)
    d["P6_n"] = (ro, nro)
    none = sum(p2[p]["none"]["both_correct"] for p in PAIRS); suf = sum(p2[p]["suffix"]["both_correct"] for p in PAIRS); eq = sum(p2[p]["eq"]["both_correct"] for p in PAIRS)
    d["twins_both"] = dict(ordinary=none, reset_before_suffix=suf, reset_before_eq=eq, of=sum(p2[p]["none"]["pairs"] for p in PAIRS))
    d["P7_ratio"] = round(suf / none, 3) if none else None
    by = {}
    for p in PAIRS:
        for st, v in p2[p]["suffix"]["by_steps_operand_still_in_window"].items():
            b = by.setdefault(int(st), [0, 0]); b[0] += v["both_correct"]; b[1] += v["pairs"]
    d["reset_before_suffix_by_steps"] = {st: f"{by[st][0]}/{by[st][1]}" for st in sorted(by)}
    # descriptive: per pair group
    for name, prs in (("plus_minus", PM), ("comparison", ("><", "<>"))):
        g = {}
        for key in ("intermediate", "whole", "ending_given_true_intermediate", "whole_given_intermediate_right"):
            k, n = pool(s, key, prs); g[key] = pct(k, n)
        g["both_intermediate"] = pct(sum(p1[f"twins {p}"]["both"]["intermediate"] for p in prs), sum(p1[f"twins {p}"]["both"]["pairs"] for p in prs))
        g["both_whole"] = pct(sum(p1[f"twins {p}"]["both"]["whole"] for p in prs), sum(p1[f"twins {p}"]["both"]["pairs"] for p in prs))
        g["both_with_true_intermediate"] = pct(sum(p1[f"twins {p}"]["both"]["ending_given_true_intermediate"] for p in prs), sum(p1[f"twins {p}"]["both"]["pairs"] for p in prs))
        gl = {}
        for key in ("intermediate", "whole", "ending_given_true_intermediate"):
            k, n = pool(s, key, prs, "lead"); gl[key] = pct(k, n)
        d[name] = dict(twins=g, lead=gl)
    d["taught_pairs_hidden"] = {k_: {key: pct(p1[k_][key]["k"], p1[k_][key]["n"]) for key in ("intermediate", "whole", "ending_given_true_intermediate")} for k_ in ("within_hidden", "taught_hidden")}
    rows[s] = d
def verdict(name, test, void=False):
    hits = [bool(test(rows[s])) for s in SEEDS]
    V[name] = dict(per_seed=hits, verdict="void" if void else ("holds" if sum(hits) >= 2 else "misses"))
ok = all(rows[s]["controls_in_order"] for s in SEEDS)
verdict("P-PR.1", lambda d: d["P1_pct"] >= 70)
verdict("P-PR.2", lambda d: d["P2_pct"] >= 85)
verdict("P-PR.3", lambda d: d["P3_gap"] >= 10)
verdict("P-PR.4", lambda d: d["P4_pct"] >= 60)
verdict("P-PR.5", lambda d: d["P5_pct"] >= 50, void=not ok)
verdict("P-PR.6", lambda d: d["P6_gap"] is not None and d["P6_gap"] >= 20, void=not ok)
verdict("P-PR.7", lambda d: d["P7_ratio"] is not None and d["P7_ratio"] <= .5)
out = dict(verdicts=V, per_seed=rows, script_sha256={s: R[s]["script_sha256"] for s in SEEDS}, weights_sha256={s: R[s]["weights_sha256"] for s in SEEDS})
pathlib.Path("exp45/E20P_VERDICTS.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
