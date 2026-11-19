"""An output end written from E55's worker's README_YESNO.md / README_REACH3.md, on the outputs recorded by my_tissue.c.

Each of the 256 output lines' drivers is a linear unit whose inputs are the cells near the output lines (set (a): the drivers and the
members of a driver's neighbourhood, enlarged lists, feedback cells excluded) that lie in its own neighbourhood (the driver included), plus
a constant 1; strengths start at 0. At position t (outputs at the reading step of byte t) every unit's activity is computed, the most
active emits (ties to the lowest byte value), and the emission is right if it equals byte t+1. Rules:
  A (kappa 1/8; eps 0 or 1/16): the emitter moves by +eta u after yes, -kappa eta u after no;
  B (eps 1/16): the emitter moves by eta (r - rbar_e) u, r = 1 after yes, 0 after no, rbar_e its own running share of yes (rate 1/64,
    starting at 0, updated after the step);
  D (eta 1/64 or 1/16): every unit moves by eta (target - activity) u / |u|^2 (u with the constant 1).
For A and B, eta = (1/4) / (number of the unit's cell inputs + 1). With eps the emitter is, with that chance, a unit drawn at random among
the byte values seen so far (numpy default_rng(seed), drawn in the order of the configurations, as the reading of measure.py needs).
Inputs: run (each cell's running mean and deviation: r = 1/min(n, 1024), m += r (x - m), s2 += r ((x - m)^2 - s2), with the new m; the
input is (x - m) / sqrt(s2) after this byte's update, 0 where s2 = 0); clip8 (run clipped at +-8; its own generator default_rng(seed + 1)
so that run and raw keep their draws); raw (the outputs as they are).
Scores: share right in windows of 4,096 positions, the last quarter (positions from 3(NB-1)/4 + 1 on), and ref_1, ref_2, the end mark's
share from the scorer (deps/PREP/score_emissions.py).

    python3 own_end.py <rec file (float32, every byte)> <seed> <nbytes> <tag> [A4]
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "/home/user/tissue_deps"); sys.path.insert(0, "/home/user/tissue_check"); sys.path.insert(0, "/home/user/prep_deps")
import tissue_step1 as ts, my_tissue as mt
import life_stream as LS, score_emissions as SE

recf, seed, NB, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
FR, RE = (1.0, 0.25) if (len(sys.argv) > 5 and sys.argv[5] == "A4") else (1 / 8, 0.5)   # A4: every eligible cell long, reach 1/4
t0 = time.time()
Bt = ts.birth(seed); _, members, _ = mt.long_axons(Bt, seed, FR, RE)
b, _ = ts.stream(seed, NB); b = b.astype(np.int64)
L = LS.life(seed); assert np.array_equal(L["bytes"][:NB], b), "the scorer's life stream differs from the tissue's stream"
isfb = np.asarray(Bt["kind"]) == ts.FBK
drivers = list(Bt["driver"])
near = set(drivers)
for i in drivers: near.update(int(m) for m in members[i] if m < ts.N)
near = np.array(sorted(i for i in near if not isfb[i])); K = len(near); col = {c: k for k, c in enumerate(near)}
own = [sorted({col[c] for c in [i] + [int(m) for m in members[i] if m < ts.N] if c in col}) for i in drivers]
G = max(map(len, own)); IDX = np.full((256, G), K); P = np.array([len(o) + 1.0 for o in own])
for v in range(256): IDX[v, :len(own[v])] = own[v]
R = np.fromfile(recf, dtype=np.float32).reshape(-1, ts.N)[:NB][:, near].astype(np.float64)
M = NB - 1; y = b[1:]; nf = (3 * M) // 4; q0 = nf + 1
RULES = [("A", 0.0, 1 / 8), ("A", 1 / 16, 1 / 8), ("B", 1 / 16, 0.0), ("D", 0.0, 1 / 64), ("D", 0.0, 1 / 16)]
CFG = [(p, r) for p in ("run", "raw", "clip8") for r in RULES]; C = len(CFG)
W = np.zeros((C, 256, G + 1)); rbar = np.zeros((C, 256))
rng = {"run": np.random.default_rng(seed), "raw": None, "clip8": np.random.default_rng(seed + 1)}; rng["raw"] = rng["run"]
right = np.zeros((C, NB), bool); seen = np.zeros(256, bool); m = np.zeros(K); s2 = np.zeros(K)
for t in range(M):
    x = R[t]; r = 1.0 / min(t + 1, 1024); seen[b[t]] = True
    m += r * (x - m); s2 += r * ((x - m) ** 2 - s2); sd = np.sqrt(s2)
    u = np.where(sd > 0, (x - m) / np.where(sd > 0, sd, 1), 0.0)
    Us = {}
    for nm, z in (("run", u), ("raw", x), ("clip8", np.clip(u, -8, 8))):
        U = np.ones((256, G + 1)); U[:, :-1] = np.append(z, 0.0)[IDX]; Us[nm] = U
    for c, (prep, (rule, eps, par)) in enumerate(CFG):
        U = Us[prep]; a = (W[c] * U).sum(1); e = int(a.argmax())
        if eps and rng[prep].random() < eps:
            sv = np.flatnonzero(seen); e = int(sv[rng[prep].integers(0, len(sv))])
        ok = e == y[t]; right[c, t + 1] = ok
        if rule == "A":
            W[c, e] += (1.0 if ok else -par) * (0.25 / P[e]) * U[e]
        elif rule == "B":
            W[c, e] += (float(ok) - rbar[c, e]) * (0.25 / P[e]) * U[e]; rbar[c, e] += (float(ok) - rbar[c, e]) / 64
        else:
            tg = np.zeros(256); tg[y[t]] = 1; W[c] += (par * (tg - a) / (U * U).sum(1))[:, None] * U
refs = SE.references(L["bytes"][:NB])
wins = [(a_, min(a_ + 4096, NB)) for a_ in range(0, NB, 4096)]
wsh = lambda ok: [float(ok[max(a_, 1):e_].mean()) for a_, e_ in wins]
lv = {"end mark": [float((b[a_:e_] == 10).mean()) for a_, e_ in wins], "ref_1": wsh(refs["ref_1"] == b), "ref_2": wsh(refs["ref_2"] == b)}
lq = {"end mark": float((b[q0:] == 10).mean()), "ref_1": float((refs["ref_1"] == b)[q0:].mean()), "ref_2": float((refs["ref_2"] == b)[q0:].mean())}
res = {}
for c, (prep, (rule, eps, par)) in enumerate(CFG):
    nm = f"{prep} {rule}" + (f" eps={eps:g}" if rule != "D" else "") + (f" kappa={par:g}" if rule == "A" else "") + (f" eta={par:g}" if rule == "D" else "")
    w = wsh(right[c])
    res[nm] = dict(windows=w, last_quarter=float(right[c, q0:].mean()),
                   first_window_reaching={k: next((wins[i][1] for i in range(len(w)) if w[i] >= l[i]), None) for k, l in lv.items()})
out = dict(tag=tag, seed=seed, bytes=NB, cells_a=K, inputs_per_driver=dict(min=int(P.min() - 1), median=float(np.median(P) - 1), max=int(P.max() - 1)),
           rules=res, levels=lv, levels_last_quarter=lq, seconds=time.time() - t0)
json.dump(out, open(f"/home/user/tissue_check/out/own_end_{tag}.json", "w"), indent=0)
print(f"== {tag}: set a {K} cells; inputs a driver {out['inputs_per_driver']}; last quarter ref_1 {100*lq['ref_1']:.2f} ref_2 {100*lq['ref_2']:.2f} end mark {100*lq['end mark']:.2f}")
for k, v in res.items():
    print(f"   {k:28s} last quarter {100 * v['last_quarter']:5.2f}  first reaching {v['first_window_reaching']}")
