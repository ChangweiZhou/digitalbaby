"""E20-P2: the four predictions of E20P2_PREDICTIONS.md judged from exp45/E20P2_e16_life_D_cm6_<seed>.json. Writes exp45/E20P2_VERDICTS.json."""
import json, pathlib
SEEDS = (73129, 93683, 116089); NEAR = ("first_operand_one_digit_off", "second_operand_one_digit_off", "off_by_one", "one_digit_right")
R = {s: json.loads(pathlib.Path(f"exp45/E20P2_e16_life_D_cm6_{s}.json").read_text()) for s in SEEDS}
pct = lambda k, n: round(100 * k / n, 1) if n else None
rows, V = {}, {}
for s in SEEDS:
    tx, op = R[s]["taxonomy"], R[s]["operators"]; d = {}
    f = op["trained"]["first_operator"]; u = op["untrained_net"]["first_operator"]; sh = op["trained"]["first_operator_shuffled_labels"]
    d["first_operator"] = {k: pct(f[k]["k"], f[k]["n"]) for k in f}; d["first_operator_n"] = {k: f[k]["n"] for k in f}
    d["first_operator_untrained"] = {k: pct(u[k]["k"], u[k]["n"]) for k in u}
    d["first_operator_shuffled"] = {k: pct(sh[k]["k"], sh[k]["n"]) for k in sh}
    d["second_operator"] = {k: pct(v["k"], v["n"]) for k, v in op["trained"]["second_operator"].items()}
    d["second_operator_untrained"] = {k: pct(v["k"], v["n"]) for k, v in op["untrained_net"]["second_operator"].items()}
    d["first_operator_by_intermediate"] = op["trained"]["first_operator_read_out_by_intermediate_written"]
    d["control_in_order"] = max(d["first_operator_shuffled"].values()) <= 35
    for name in ("twins plus_minus", "lead plus_minus", "twins comparison", "lead comparison"):
        t = tx[name]; n = t["wrong_readable"]
        d[name] = dict(items=t["items"], intermediate_right=t["intermediate_right"], unreadable=t["unreadable"], wrong_readable=n,
                       share={k: pct(v, n) for k, v in t["counts"].items()}, uniform_baseline_share={k: pct(v, n) for k, v in t["uniform_baseline_counts"].items()},
                       near_miss_share=pct(sum(t["counts"][k] for k in NEAR), n), near_miss_baseline=pct(sum(t["uniform_baseline_counts"][k] for k in NEAR), n),
                       largest=max(t["counts"], key=t["counts"].get) if n else None)
    rows[s] = d
def verdict(name, test, void=False):
    hits = [bool(test(rows[s])) for s in SEEDS]; V[name] = dict(per_seed=hits, verdict="void" if void else ("holds" if sum(hits) >= 2 else "misses"))
verdict("P-P2.1", lambda d: d["first_operator"]["twins_operator_out_of_view"] >= 90 and d["first_operator"]["test16"] >= 90, void=not all(rows[s]["control_in_order"] for s in SEEDS))
verdict("P-P2.2", lambda d: d["twins plus_minus"]["share"]["other_operator"] >= 25)
verdict("P-P2.3", lambda d: d["twins plus_minus"]["near_miss_share"] >= 50)
verdict("P-P2.4", lambda d: d["twins plus_minus"]["largest"] == d["lead plus_minus"]["largest"])
out = dict(verdicts=V, per_seed=rows); pathlib.Path("exp45/E20P2_VERDICTS.json").write_text(json.dumps(out, indent=1))
for k, v in V.items(): print(k, v)
for s in SEEDS:
    d = rows[s]; print("\n==", s)
    for k in ("first_operator", "first_operator_n", "first_operator_untrained", "first_operator_shuffled", "second_operator", "second_operator_untrained", "first_operator_by_intermediate"):
        print(" ", k, d[k])
    for name in ("twins plus_minus", "lead plus_minus", "twins comparison", "lead comparison"):
        t = d[name]; print(" ", name, {k: t[k] for k in ("items", "intermediate_right", "unreadable", "wrong_readable", "near_miss_share", "near_miss_baseline", "largest")})
        print("     share   ", t["share"]); print("     baseline", t["uniform_baseline_share"])
