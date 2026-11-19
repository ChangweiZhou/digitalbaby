"""The yes/no question on EXACT inputs, written again (task of 12:52 UTC): output units, one per byte value, on one-hots of the present
byte (inputs 1), the present and previous byte (2) or three bytes (3), plus a constant 1. Rules (as stated in the main worker's
message and yn.py's head comment; readings in RESULTS.md section 13):
  B (eps 1/16): the emitter e (largest activity, ties to the lower value; with chance eps a value seen so far, at random) moves by
     (ok - r_e) * (1/4)/n * U, r_e its own running share of yes, rate 1/64, updated after the step; n = number of active inputs.
  D (eta): every unit moves by eta (target - a) U / n (|U|^2 = n).
  Y (choice eps, up, base own, decay s|c, s0): the emitter as B with step st = s0 (1/4)/n [/ sqrt(1 + m_e/16) for decay s, m_e the
     number of the unit's earlier emissions]; on "no", every other unit (all 255) adds up * s0 * (1/4)/n * U.
Random choices: one generator for all rules, drawn in the order of the rules at each byte (as yn.py), seeded by gen_seed (default
the life seed); 'sep' gives each rule a generator of its own, default_rng([gen_seed, rule index]).
Scores: windows of 4,096; last quarter from (3 NB)//4; levels from score_emissions.references.

    python3 own_yn.py <label> seed=.. nbytes=.. inputs=1|2|3 rules=B:1/16,D:1/64,Y:eps:1/256:own:s:1 (A:eps:kappa too) [gen=<int>] [sep=1]
           [trace=1: Y and B weight norms every 4,096 bytes]
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, "/home/user/prep_deps")
import life_stream as LS, score_emissions as SE

label = sys.argv[1]; kw = dict(a.split("=") for a in sys.argv[2:]); t0 = time.time()
seed = int(kw["seed"]); NB = int(kw["nbytes"]); h = int(kw["inputs"]); gseed = int(kw.get("gen", seed)); sep = kw.get("sep") == "1"
trace = kw.get("trace") == "1"
life = LS.life(seed); b = np.asarray(life["bytes"][:NB], dtype=np.int64)
F = 256 * h + 1
rules = []
for r in kw["rules"].split(","):
    f = r.split(":")
    if f[0] == "A": rules.append(("A", float(eval(f[1])), float(eval(f[2]))))          # A:eps:kappa, only to keep the draw order
    elif f[0] == "B": rules.append(("B", float(eval(f[1]))))
    elif f[0] == "D": rules.append(("D", float(eval(f[1]))))
    elif f[0] == "Y":
        assert f[1] == "eps" and f[3] == "own", "only choice eps with base own is written here"
        rules.append(("Y", float(eval(f[2])), f[4], float(eval(f[5]))))
C = len(rules)
W = np.zeros((C, 256, F)); r = np.zeros((C, 256)); m = np.zeros((C, 256)); right = np.zeros((C, NB), bool); seen = np.zeros(256, bool)
gens = [np.random.default_rng([gseed, c]) for c in range(C)] if sep else [np.random.default_rng(gseed)] * C
norms = {c: [] for c in range(C)}
for t in range(NB - 1):
    idx = np.array([256 * j + b[t - j] for j in range(h) if t - j >= 0] + [F - 1]); n = len(idx)
    seen[b[t]] = True; nxt = b[t + 1]
    for c, rule in enumerate(rules):
        a = W[c][:, idx].sum(1)
        e = int(np.argmax(a))
        if (rule[0] in ("B", "Y") or (rule[0] == "A" and rule[1] > 0)) and gens[c].random() < (rule[1] if rule[0] == "A" else 1 / 16):
            s = np.flatnonzero(seen); e = int(s[gens[c].integers(0, len(s))])
        ok = e == nxt; right[c, t + 1] = ok
        if rule[0] == "A":
            W[c, e, idx] += (1.0 if ok else -rule[2]) * 0.25 / n
        elif rule[0] == "D":
            tg = np.zeros(256); tg[nxt] = 1.0
            W[c][:, idx] += (rule[1] * (tg - a) / n)[:, None]
        else:
            if rule[0] == "B": step = 0.25 / n
            else:
                _, up, decay, s0 = rule
                step = s0 * 0.25 / n / (np.sqrt(1.0 + m[c, e] / 16.0) if decay == "s" else 1.0); m[c, e] += 1
            W[c, e, idx] += (float(ok) - r[c, e]) * step
            r[c, e] += (float(ok) - r[c, e]) / 64
            if rule[0] == "Y" and up and not ok:
                inc = up * rule[3] * 0.25 / n
                W[c][:, idx] += inc; W[c, e, idx] -= inc          # every unit but the emitter
    if trace and (t + 2) % 4096 == 0:
        for c in range(C):
            nv = np.linalg.norm(W[c], axis=1); norms[c].append(dict(byte=t + 2, max=float(nv.max()), median=float(np.median(nv)),
                                                                    mean_w=float(W[c].mean()), max_abs_w=float(np.abs(W[c]).max())))
refs = SE.references(life["bytes"][:NB])
wins = [(a_, min(a_ + 4096, NB)) for a_ in range(0, NB, 4096)]
wsh = lambda ok: [float(ok[max(a_, 1):e_].mean()) for a_, e_ in wins]
lv = {k: wsh(refs[k] == b) for k in ("ref_1", "ref_2", "ref_3")}; lv["end mark"] = [float((b[a_:e_] == 10).mean()) for a_, e_ in wins]
q0 = (3 * NB) // 4
res = {}
for c, rule in enumerate(rules):
    nm = {"A": lambda: f"A eps={rule[1]:g} kappa={rule[2]:g}", "B": lambda: "B eps=0.0625", "D": lambda: f"D eta={rule[1]:g}", "Y": lambda: f"Y eps up={rule[1]:g} base=own decay={rule[2]} s0={rule[3]:g}"}[rule[0]]()
    w = wsh(right[c])
    res[nm] = dict(windows=w, last_window=w[-1], last_quarter=float(right[c, q0:].mean()),
                   first_window_reaching={k: next((wins[i][1] for i in range(len(w)) if w[i] >= l[i]), None) for k, l in lv.items()})
    if trace: res[nm]["norms"] = norms[c]
out = dict(label=label, seed=seed, bytes=NB, inputs=h, gen=gseed, sep=sep, rules=res, levels=lv,
           last_quarter_refs={k: float((refs[k] == b)[q0:].mean()) for k in ("ref_1", "ref_2", "ref_3")}, seconds=time.time() - t0)
os.makedirs("/home/user/tissue_check/out/yn", exist_ok=True)
json.dump(out, open(f"/home/user/tissue_check/out/yn/{label}.json", "w"), indent=0)
print(label, {k: round(100 * v["last_quarter"], 2) for k, v in res.items()}, f"{time.time() - t0:.0f} s", flush=True)
