"""E20-P2: follow-up to E20-P on the same three checkpoints (review 9.94). Scoring only. Exploratory: it was designed after E20-P's results were seen.
  A. What are the wrong intermediates? Each readable wrong intermediate is put in the first category it fits, in a fixed order, and the shares are
     set beside what a uniformly random wrong two-digit value would give for the same items.
  B. Is the first operator, which is out of view in most twins, still in the state at `=`? A four-way ridge readout fitted on fresh items of ALL
     sixteen operator pairs, so that the first operator cannot be read off the second (among the four held pairs each determines the other).

  E16_COMPGAP=6 python3 e20_probe2.py fixtures <seed>
  E16_COMPGAP=6 python3 e20_probe2.py run <weights.pt> <seed>      writes exp45/E20P2_<stem>.json"""
import os, sys, json, pathlib
from random import Random
import torch as T
import e20_probe as P
X, E = P.X, P.E
OPS = "+-><"; NFIT, NTEST = 400, 100          # fresh items per operator pair for fitting and for the sixteen-pair test; fixed before any trained checkpoint was probed
ORDER = ("other_operator", "operands_swapped", "another_modulus", "copied_operand", "first_operand_one_digit_off", "second_operand_one_digit_off",
         "off_by_one", "one_digit_right", "other")


def category_sets(o1, a, b, i):
    op = X.SYM_OP[o1]; other = {"add": "sub", "sub": "add", "max": "min", "min": "max"}[op]
    near = lambda v: [v2 for v2 in range(97) if v2 != v and (v2 // 10 == v // 10 or v2 % 10 == v % 10)]
    return [("other_operator", {E.apply(other, 97, a, b)}), ("operands_swapped", {E.apply(op, 97, b, a)}),
            ("another_modulus", {E.apply(op, 89, a, b), E.apply(op, 73, a, b)} if op in ("add", "sub") else set()), ("copied_operand", {a, b}),
            ("first_operand_one_digit_off", {E.apply(op, 97, a2, b) for a2 in near(a)}), ("second_operand_one_digit_off", {E.apply(op, 97, a, b2) for b2 in near(b)}),
            ("off_by_one", {(i + 1) % 97, (i - 1) % 97}), ("one_digit_right", {v for v in range(100) if v // 10 == i // 10 or v % 10 == i % 10})]


def classify(o1, a, b, i, w):
    assert w != i
    for name, vals in category_sets(o1, a, b, i):
        if w in vals:
            return name
    return "other"


def baseline(o1, a, b, i):
    """Share of each category if the wrong value were uniform over the 99 other two-digit values."""
    c = {k: 0 for k in ORDER}
    for w in range(100):
        if w != i:
            c[classify(o1, a, b, i, w)] += 1
    return {k: v / 99 for k, v in c.items()}


def taxonomy(items, rows):
    """rows: P.probe_intermediate's rows for these items (the written answer is rows[k]['out'])."""
    counts = {k: 0 for k in ORDER}; base = {k: 0.0 for k in ORDER}; n_wrong = unreadable = right = 0
    for (p, t), r in zip(items, rows):
        d = P.parse(p); i = int(bytes(t[:2])); o = r["out"]
        if len(o) < 3 or not all(48 <= ch <= 57 for ch in o[:2]) or o[2] != ord(";"):
            unreadable += 1; continue
        w = int(bytes(o[:2]))
        if w == i:
            right += 1; continue
        n_wrong += 1; counts[classify(d["o1"], d["a"], d["b"], i, w)] += 1
        for k, v in baseline(d["o1"], d["a"], d["b"], i).items():
            base[k] += v
    return dict(items=len(items), whole_right=sum(bool(r["whole"]) for r in rows), intermediate_right=right, unreadable=unreadable, wrong_readable=n_wrong, counts=counts,
                uniform_baseline_counts={k: round(v, 1) for k, v in base.items()})


# ---------------------------------------------------------------- B: operator readout
def fresh16(seed, all_taught, panel_triples):
    g = Random(f"probe2|{seed}"); fit, test = [], []
    def letters():
        while True:
            a, b = (bytes(g.choice(E.DISTRACT) for _ in range(g.randint(0, X.COMPGAP16))) for _ in range(2))
            if len(a) + len(b) >= 3:
                return a, b
    for o1, o2 in X.PAIRS:
        seen = set(); got = []
        while len(got) < NFIT + NTEST:
            tr = (g.randrange(97), g.randrange(97), g.randrange(97))
            if tr in seen or tr in all_taught or (o1 + o2, tr) in panel_triples:
                continue
            seen.add(tr); got.append(X.render_pair(o1, o2, *tr, inside=letters()))
        fit += got[:NFIT]; test += got[NFIT:]
    return fit, test


def ridge_classes(Ftr, Fte, ytr, yte, k):
    mu, sd = Ftr.mean(0), Ftr.std(0).clamp_min(1e-8)
    A = T.cat([(Ftr - mu) / sd, T.ones(len(Ftr), 1, dtype=T.double)], 1); B = T.cat([(Fte - mu) / sd, T.ones(len(Fte), 1, dtype=T.double)], 1)
    Wt = T.linalg.solve(A.T @ A + P.LAMBDA * T.eye(A.shape[1], dtype=T.double), A.T @ T.nn.functional.one_hot(T.tensor(ytr), k).double())
    return ((B @ Wt).argmax(1) == T.tensor(yte)).tolist()


def op_labels(items):
    ps = [P.parse(p) for p, _ in items]
    return [OPS.index(d["o1"]) for d in ps], [OPS.index(d["o2"]) for d in ps]


def operator_readout(feat, seed, fit, test, twin_items, twin_rows=None):
    """feat: items -> feature matrix. Accuracy of the first and second operator on the sixteen-pair test and on the twin items; shuffled-label control."""
    Ff, Ft, Fw = feat(fit), feat(test), feat(twin_items)
    (f1, f2), (t1, t2), (w1, w2) = op_labels(fit), op_labels(test), op_labels(twin_items)
    perm = list(range(len(fit))); Random(f"shuffle2|{seed}").shuffle(perm)
    held = [(P.parse(p)["o1"], P.parse(p)["o2"]) in X.HELD_PAIRS for p, _ in test]
    out_of_view = [len(p) - 10 >= 4 for p, _ in twin_items]      # the first operator is at prompt index 3; the last ten bytes start at len - 10
    h1t, h1w = ridge_classes(Ff, Ft, f1, t1, 4), ridge_classes(Ff, Fw, f1, w1, 4)
    res = dict(first_operator=dict(test16=P.rate(h1t), test16_held_pairs=P.rate(h for h, x in zip(h1t, held) if x), test16_taught_pairs=P.rate(h for h, x in zip(h1t, held) if not x),
                                   twins=P.rate(h1w), twins_operator_out_of_view=P.rate(h for h, x in zip(h1w, out_of_view) if x),
                                   twins_operator_in_view=P.rate(h for h, x in zip(h1w, out_of_view) if not x)),
               second_operator=dict(test16=P.rate(ridge_classes(Ff, Ft, f2, t2, 4)), twins=P.rate(ridge_classes(Ff, Fw, f2, w2, 4))),
               first_operator_shuffled_labels=dict(test16=P.rate(ridge_classes(Ff, Ft, [f1[j] for j in perm], t1, 4)), twins=P.rate(ridge_classes(Ff, Fw, [f1[j] for j in perm], w1, 4))),
               fitted_on=len(fit))
    if twin_rows is not None:
        res["first_operator_read_out_by_intermediate_written"] = {
            "read out, intermediate right": sum(h and r["inter"] for h, r in zip(h1w, twin_rows)), "read out, intermediate wrong": sum(h and not r["inter"] for h, r in zip(h1w, twin_rows)),
            "not read out, intermediate right": sum((not h) and r["inter"] for h, r in zip(h1w, twin_rows)), "not read out, intermediate wrong": sum((not h) and not r["inter"] for h, r in zip(h1w, twin_rows))}
    return res


def probe2_all(m, m0, seed, mem, all_taught, panel_triples, n_twins=None):
    out = dict(taxonomy={}, operators={}); groups = dict(plus_minus=("+-", "-+"), comparison=("><", "<>")); tw_items, tw_rows = [], []
    for gname, pairs in groups.items():
        for panel in ("twins", "lead"):
            items = [it for p in pairs for xy in mem[panel][p][:n_twins] for it in xy]; rows = P.probe_intermediate(m, items)
            out["taxonomy"][f"{panel} {gname}"] = taxonomy(items, rows)
            if panel == "twins":
                tw_items += items; tw_rows += rows
    fit, test = fresh16(seed, all_taught, panel_triples)
    if n_twins is not None:
        fit = [it for k in range(16) for it in fit[k * NFIT:k * NFIT + 60]]; test = [it for k in range(16) for it in test[k * NTEST:k * NTEST + 20]]
    out["operators"]["trained"] = operator_readout(lambda its: P.state_after_eq(m, [p for p, _ in its]), seed, fit, test, tw_items, tw_rows)
    if m0 is not None:
        u = operator_readout(lambda its: P.state_after_eq(m0, [p for p, _ in its]), seed, fit, test, tw_items)
        out["operators"]["untrained_net"] = dict(first_operator=u["first_operator"], second_operator=u["second_operator"])
    return out


def fixtures(seed):
    out = {}; mem, sha, all_taught, panel_triples = P.build(seed)
    # 1. the categories, on one hand-worked item: 60 + 50 = 13 (mod 97)
    hand = {10: "other_operator", 21: "another_modulus", 37: "another_modulus", 60: "copied_operand", 50: "copied_operand", 14: "first_operand_one_digit_off",
            12: "off_by_one", 93: "one_digit_right", 47: "other"}
    got = {w: classify("+", 60, 50, 13, w) for w in hand}
    sub = classify("-", 60, 50, 10, 87)                # 50 - 60 = 87 (mod 97): the operands swapped
    mx = classify(">", 60, 50, 60, 50)                 # the other operand of a maximum is the minimum: other operator
    b = baseline("+", 60, 50, 13)
    out["1_categories"] = dict(expected=hand, got=got, subtraction_swapped=sub, maximum_other_operand=mx, baseline_sums_to_one=abs(sum(b.values()) - 1) < 1e-9, baseline=b)
    ok1 = got == hand and sub == "operands_swapped" and mx == "other_operator" and out["1_categories"]["baseline_sums_to_one"]
    # a stub responder that always answers with the other operator must be counted there, whole
    tw = [it for xy in mem["twins"]["+-"][:20] for it in xy]
    def other(prompts, cap):
        res = []
        for p in prompts:
            d = P.parse(p if p[-1] == ord("=") else p[:-3]); iw = E.apply("sub", 97, d["a"], d["b"])
            res.append((P.digits(iw) + [ord(";")] + P.digits(E.apply(X.SYM_OP[d["o2"]], 97, iw, d["c"])) + [P.EOS])[:cap] if p[-1] == ord("=") else P.digits(0) + [P.EOS])
        return res
    tx = taxonomy(tw, P.probe_intermediate(None, tw, other)); out["1_stub_other_operator"] = tx
    ok1 = ok1 and tx["counts"]["other_operator"] == tx["wrong_readable"] and tx["wrong_readable"] + tx["intermediate_right"] == len(tw)
    # 2. the operator readout on synthetic features: the first operator present; only the second present, fitted on sixteen pairs and fitted on the held four
    fit, test = fresh16(seed, all_taught, panel_triples); g = T.Generator().manual_seed(seed)
    def synth(which):
        def feat(items):
            f = T.randn(len(items), 32, generator=g, dtype=T.double) * .1
            for k, (p, _) in enumerate(items):
                f[k, OPS.index(P.parse(p)[which])] += 1
            return f
        return feat
    present = operator_readout(synth("o1"), seed, fit, test, tw)["first_operator"]
    second_only = operator_readout(synth("o2"), seed, fit, test, tw)["first_operator"]
    held_fit = [it for it in fit if (P.parse(it[0])["o1"], P.parse(it[0])["o2"]) in X.HELD_PAIRS]
    confounded = operator_readout(synth("o2"), seed, held_fit, test, tw)["first_operator"]
    out["2_operator_readout"] = dict(first_operator_present=present, only_second_present_fit_on_sixteen=second_only, only_second_present_fit_on_held_four=confounded)
    ok2 = present["test16"]["k"] == present["test16"]["n"] and present["twins"]["k"] == present["twins"]["n"] \
        and second_only["test16"]["k"] <= .4 * second_only["test16"]["n"] and second_only["twins"]["k"] <= .6 * second_only["twins"]["n"] \
        and confounded["twins"]["k"] == confounded["twins"]["n"] and confounded["test16"]["k"] <= .5 * confounded["test16"]["n"]
    pairs_fit = sorted({P.parse(p)["o1"] + P.parse(p)["o2"] for p, _ in fit})
    shared = len({tuple(p) for p, _ in fit} & {tuple(p) for p, _ in test})
    out["2_fresh_items"] = dict(fit=len(fit), test=len(test), operator_pairs=len(pairs_fit), prompts_shared_between_fit_and_test=shared,
                                taught_triples=sum((P.parse(p)["a"], P.parse(p)["b"], P.parse(p)["c"]) in all_taught for p, _ in fit + test))
    ok2 = ok2 and len(pairs_fit) == 16 and shared == 0 and out["2_fresh_items"]["taught_triples"] == 0
    # 3. the run path on untrained nets at a small size
    pa = probe2_all(E.new_net(seed), E.new_net(seed + 1), seed, mem, all_taught, panel_triples, n_twins=20)
    out["3_run_path_on_untrained_net"] = dict(twins_plus_minus=pa["taxonomy"]["twins plus_minus"], first_operator=pa["operators"]["trained"]["first_operator"], second_operator=pa["operators"]["trained"]["second_operator"])
    ok3 = pa["taxonomy"]["twins plus_minus"]["intermediate_right"] <= 2
    out["pass"] = dict(categories=ok1, operator_readout=ok2, run_path=ok3)
    pathlib.Path(f"exp45/E20P2_FIXTURES_{seed}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out["pass"])); sys.exit(0 if all(out["pass"].values()) else 1)


def run(weights, seed):
    import time
    wp = pathlib.Path(weights); mem, sha, all_taught, panel_triples = P.build(seed)
    rec = json.loads(wp.with_suffix(".json").read_text())
    assert isinstance(rec.get("complete"), dict) and rec["complete"]["weights_sha256"] == X.sha_file(wp), "the weights file is not the one named in the life's completion record"
    assert rec["stages"][0]["settings"]["memory_panels"]["sha256"] == sha, "the memory panels built here are not the ones the life scored"
    m = E.new_net(seed); m.load_state_dict(T.load(wp)); m.eval(); t0 = time.process_time()
    out = dict(weights=wp.name, weights_sha256=X.sha_file(wp), seed=seed, panel_sha256=sha, lambda_=P.LAMBDA, fit_per_pair=NFIT, test_per_pair=NTEST, category_order=ORDER,
               script_sha256=X.sha_file(__file__), probe_script_sha256=X.sha_file(P.__file__), **probe2_all(m, E.new_net(seed), seed, mem, all_taught, panel_triples))
    life = rec["stages"][-1]["scores"]["memory"]["twins"]     # the items this script scored must be the life's: same item accuracy on the twins
    for gname, pairs in (("plus_minus", ("+-", "-+")), ("comparison", ("><", "<>"))):
        tx = out["taxonomy"][f"twins {gname}"]
        assert tx["items"] == 2 * sum(life[p]["pairs"] for p in pairs) and tx["whole_right"] == sum(life[p]["items_correct"] for p in pairs), f"twins {gname}: this script and the life disagree"
    out["cpu_seconds"] = round(time.process_time() - t0, 1)
    pathlib.Path(f"exp45/E20P2_{wp.stem}.json").write_text(json.dumps(out, indent=1)); print("wrote", f"exp45/E20P2_{wp.stem}.json", out["cpu_seconds"], "cpu-s")


if __name__ == "__main__":
    T.set_num_threads(1)
    if sys.argv[1] == "fixtures": fixtures(int(sys.argv[2]))
    else: run(sys.argv[2], int(sys.argv[3]))
