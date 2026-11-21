# Fifth look: the relay system of cells2.py through the life add 400,000 lines, then subtract 200,000 lines; never-shown additions and subtractions scored with an empty past and
# after a read line of each kind.  For subtraction the clean score as well (held-out pairs whose line no shown pair writes; ERRATA_20261120.md).  No mechanism is changed.
import sys, json, numpy as np
import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '.')); OUT = os.path.join(HERE, "res"); os.makedirs(OUT, exist_ok=True)
import cells2 as C
seed = int(sys.argv[1]); D = C.D
rng = np.random.default_rng([seed, 99]); held, shown = {}, {}
for op in "+-":
    o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = P[:500]; shown[op] = P[2000:]
TAB = {op: np.array([C.text(op, a, b) for a, b in shown[op]]) for op in "+-"}; shown_sub = {tuple(r) for r in TAB["-"].tolist()}
clean = np.array([tuple(C.text("-", a, b)) not in shown_sub for a, b in held["-"]])
def score(s, op, prev, mask=None):
    Ls = np.array([C.text(op, a, b) for a, b in held[op]]); W1T = s.W1.T; ok = np.ones(500, bool)
    for t in (5, 6, 7):
        hist = np.full((500, D), C.END); hist[:, :t + 1] = Ls[:, t::-1]
        if prev is not None: hist[:, t + 1:] = prev[:, ::-1][:, :D - t - 1]
        ok &= s.out(W1T[C.OFF + hist].sum(1) + s.b1).argmax(1) == Ls[:, t + 1]
    return float(100 * (ok if mask is None else ok[mask]).mean())
res = dict(seed=seed, clean_subtractions=int(clean.sum()))
for plan, tag in (("add:400000", "after addition"), ("add:400000,sub:200000", "after addition, then subtraction")):
    out = C.run(dict(label="past_life", seed=str(seed), plan=plan, fast="3", nowrite="1", every="200000")); s = C.run.last; s.flush(); g = np.random.default_rng([seed, 7]); r = {}
    for op in "+-":
        r[op] = dict(empty=score(s, op, None), after_add_line=float(np.mean([score(s, op, TAB["+"][g.integers(8000, size=500)]) for _ in range(5)])), after_sub_line=float(np.mean([score(s, op, TAB["-"][g.integers(8000, size=500)]) for _ in range(5)])))
    r["- clean"] = dict(empty=score(s, "-", None, clean), after_add_line=float(np.mean([score(s, "-", TAB["+"][g.integers(8000, size=500)], clean) for _ in range(5)])), after_sub_line=float(np.mean([score(s, "-", TAB["-"][g.integers(8000, size=500)], clean) for _ in range(5)])))
    res[tag] = r; print(f"relay, seed {seed}, {tag}: " + " | ".join(f"{k}: empty {v['empty']:.1f}, after an addition line {v['after_add_line']:.1f}, after a subtraction line {v['after_sub_line']:.1f}" for k, v in r.items()), f"| cells2 rows {[(x['phase'], x['lines'] // 1000, x['add'], x['sub']) for x in out['rows']]}", flush=True)
json.dump(res, open(os.path.join(OUT, f"past_life_seed{seed}.json"), "w"))
