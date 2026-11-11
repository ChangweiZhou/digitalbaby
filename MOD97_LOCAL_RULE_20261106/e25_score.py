"""E25: the five predictions of E25_PREDICTIONS.md judged from exp45/E25_ALIGNMENT_<seed>.json. Writes exp45/E25_VERDICTS.json and prints tables."""
import json, pathlib
SEEDS = (73129, 93683, 116089)
R = {s: json.loads(pathlib.Path(f"exp45/E25_ALIGNMENT_{s}.json").read_text())["cells"] for s in SEEDS}
r2 = lambda v: None if v is None else round(v, 3)
def st(s, net, which, panel, key, back):
    d = R[s][f"{net} {which} {panel} answer"]["states"][key].get(str(back))
    return d["mean_cosine"] if d else None
def par(s, net, which, panel, obj, name):
    return R[s][f"{net} {which} {panel} {obj}"]["parameters"][name]["cosine"]
OPS = "operations_mod97"; out = dict(verdicts={}, tables={})
def verdict(name, test, detail):
    hits = [bool(test(s)) for s in SEEDS]; out["verdicts"][name] = dict(per_seed=hits, verdict="holds" if sum(hits) >= 2 else "misses", detail=[detail(s) for s in SEEDS])
g = lambda s, back: st(s, "G1", "life", OPS, "h", back); b1 = lambda s, back: st(s, "G0", "life", OPS, "h1", back); b2 = lambda s, back: st(s, "G0", "life", OPS, "h2", back)
lc = lambda s, back: st(s, "G2", "life", OPS, "c", back); lh = lambda s, back: st(s, "G2", "life", OPS, "h", back)
verdict("P25.1 GRU at 12 back <= half its value at =", lambda s: g(s, 12) <= .5 * g(s, 0), lambda s: (r2(g(s, 0)), r2(g(s, 12))))
verdict("P25.2 first bank at 12 back >= 0.8 of its value at =", lambda s: b1(s, 12) >= .8 * b1(s, 0), lambda s: (r2(b1(s, 0)), r2(b1(s, 12))))
verdict("P25.3 GRU below first bank at 12 back", lambda s: g(s, 12) < b1(s, 12), lambda s: (r2(g(s, 12)), r2(b1(s, 12))))
verdict("P25.4 LSTM cell state above GRU state by >= 0.10 at 12 back", lambda s: lc(s, 12) - g(s, 12) >= .10, lambda s: (r2(lc(s, 12)), r2(g(s, 12))))
verdict("P25.5 l1.weight: GRU below two-bank by >= 0.10", lambda s: par(s, "G0", "life", OPS, "answer", "l1.weight") - par(s, "G1", "life", OPS, "answer", "l1.weight") >= .10,
        lambda s: (r2(par(s, "G1", "life", OPS, "answer", "l1.weight")), r2(par(s, "G0", "life", OPS, "answer", "l1.weight"))))
out["lstm_clearly_better"] = [lc(s, 12) - g(s, 12) >= .15 for s in SEEDS]
# tables
BACKS = (0, 2, 4, 8, 12, 16)
T1 = {}
for which, panel in (("life", OPS), ("life", "compositions"), ("birth", "operations_mod89")):
    for label, net, key in (("GRU h", "G1", "h"), ("LSTM c", "G2", "c"), ("LSTM h", "G2", "h"), ("bank 1", "G0", "h1"), ("bank 2", "G0", "h2")):
        T1[f"{which} {panel} | {label}"] = {str(bk): [r2(st(s, net, which, panel, key, bk)) for s in SEEDS] for bk in BACKS}
out["tables"]["state_error_cosine_by_steps_back"] = T1
T2 = {}
for net, names in (("G1", ("l3.weight", "lm.weight", "l2.weight", "rnn.weight_hh_l0", "rnn.weight_ih_l0", "l1.weight", "book.weight")),
                   ("G2", ("l3.weight", "lm.weight", "l2.weight", "rnn.weight_hh_l0", "rnn.weight_ih_l0", "l1.weight", "book.weight")),
                   ("G0", ("l3.weight", "lm2.weight", "lm.weight", "l2.weight", "lc2.weight", "lw2.weight", "lp2.weight", "lc.weight", "lw.weight", "lp.weight", "l1.weight", "book.weight"))):
    for which, panel in (("life", OPS), ("life", "compositions"), ("birth", "operations_mod89")):
        for obj in ("answer", "all"):
            T2[f"{net} {which} {panel} {obj}"] = dict(all_parameters=[r2(R[s][f"{net} {which} {panel} {obj}"]["all_parameters"]) for s in SEEDS], **{n: [r2(par(s, net, which, panel, obj, n)) for s in SEEDS] for n in names})
out["tables"]["parameter_cosine"] = T2
pathlib.Path("exp45/E25_VERDICTS.json").write_text(json.dumps(out, indent=1))
for k, v in out["verdicts"].items(): print(k, "|", v["verdict"], v["per_seed"], v["detail"])
print("LSTM clearly better (>= 0.15 in all three):", out["lstm_clearly_better"])
print("\nstate-error cosine by steps back (three seeds):")
for k, v in T1.items(): print(" ", k.ljust(44), "  ".join(f"{bk}:{v[str(bk)]}" for bk in BACKS))
print("\nparameter cosines, end of life, mod-97 operations:")
for net in ("G1", "G2", "G0"):
    for obj in ("answer", "all"):
        d = T2[f"{net} life {OPS} {obj}"]; print(" ", net, obj.ljust(6), "  ".join(f"{n.replace('.weight','').replace('rnn.weight_','W')}:{v}" for n, v in d.items()))
