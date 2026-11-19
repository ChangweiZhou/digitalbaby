"""Memory at the output unit's own connections (reviewer's task of 13:25 UTC). Each input x of an output unit keeps slow copies of it,
s_k(t) = s_k(t-1) + (x(t) - s_k(t-1)) / k, updated once a byte with the present x(t) before the unit reads (s_k = 0 before byte 0).
The unit's input vector is [x, s_2, s_4, s_8] cut to the form asked, plus a constant 1; each component is standardised with its own
running mean and deviation (measure.py's 'run': r = 1/min(t+1, 1024), m updated first, s2 with the new m, standardised after this
byte's update) and clipped at +-8. Inputs of unit v: the set-(a) cells of its driver's enlarged neighbourhood, the driver included.
Rules: D (eta 1/64): every unit W[v] += eta (target - a_v) U_v / |U_v|^2. Q (eta 1/16): the emitter (largest activity, ties to the
lower value; with chance 1/16 a seen value at random) W[e] += eta (ok - a_e) U_e / |U_e|^2; on yes every other unit
W[v] += eta (0 - a_v) U_v / |U_v|^2. Forms (i) x, (ii) x s_2, (iii) x s_2 s_4, (iv) x s_2 s_4 s_8. Q with form f has its own
generator default_rng([seed, f]).
Scores as own_end.py: windows of 4,096, last quarter from 3(NB-1)/4 + 1; levels ref_1, ref_2, ref_3 and the end mark from
score_emissions.references. Weight-row norms at the end and every 8,192 bytes.

    python3 own_trace.py <rec float32 every byte> <seed> <nbytes> <tag> <A4|L8> [forms=1234]
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "/home/user/tissue_deps"); sys.path.insert(0, "/home/user/tissue_check"); sys.path.insert(0, "/home/user/prep_deps")
import tissue_step1 as ts, my_tissue as mt
import life_stream as LS, score_emissions as SE

recf, seed, NB, tag, kind = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
forms = [int(c) for c in (sys.argv[6].split("=")[1] if len(sys.argv) > 6 else "1234")]
t0 = time.time()
FR, RE = (1.0, 0.25) if kind == "A4" else (1 / 8, 0.5)
Bt = ts.birth(seed); _, members, _ = mt.long_axons(Bt, seed, FR, RE)
b, _ = ts.stream(seed, NB); b = b.astype(np.int64)
L = LS.life(seed); assert np.array_equal(L["bytes"][:NB], b)
isfb = np.asarray(Bt["kind"]) == ts.FBK; drivers = list(Bt["driver"])
near = set(drivers)
for i in drivers: near.update(int(m) for m in members[i] if m < ts.N)
near = np.array(sorted(i for i in near if not isfb[i])); K = len(near); col = {c: k for k, c in enumerate(near)}
own = [sorted({col[c] for c in [i] + [int(m) for m in members[i] if m < ts.N] if c in col}) for i in drivers]
G = max(map(len, own)); IDX = np.full((256, G), K)
for v in range(256): IDX[v, :len(own[v])] = own[v]
R = np.fromfile(recf, dtype=np.float32).reshape(-1, ts.N)[:NB][:, near].astype(np.float64)
M = NB - 1; y = b[1:]; q0 = (3 * M) // 4 + 1
KS = [1, 2, 4, 8]                                       # x = s_1
CFG = [(rule, f) for f in forms for rule in ("D", "Q")]; C = len(CFG)
W = [np.zeros((256, f * G + 1)) for _, f in CFG]
gen = [np.random.default_rng([seed, f]) for _, f in CFG]          # Q's generator keyed by the form, so forms may run apart
right = np.zeros((C, NB), bool); seen = np.zeros(256, bool)
S = np.zeros((4, K)); m = np.zeros((4, K)); s2 = np.zeros((4, K))
norms = {c: [] for c in range(C)}
for t in range(M):
    x = R[t]; seen[b[t]] = True; r = 1.0 / min(t + 1, 1024)
    for j, k in enumerate(KS): S[j] += (x - S[j]) / k
    m += r * (S - m); s2 += r * ((S - m) ** 2 - s2); sd = np.sqrt(s2)
    Z = np.clip(np.where(sd > 0, (S - m) / np.where(sd > 0, sd, 1), 0.0), -8, 8)
    Zp = np.concatenate([Z, np.zeros((4, 1))], axis=1)                       # padding column for units with fewer inputs
    Uall = Zp[:, IDX]                                                          # (4, 256, G)
    for c, (rule, f) in enumerate(CFG):
        U = np.ones((256, f * G + 1)); U[:, :-1] = Uall[:f].transpose(1, 0, 2).reshape(256, f * G)
        n2 = (U * U).sum(1); a = (W[c] * U).sum(1)
        if rule == "D":
            tg = np.zeros(256); tg[y[t]] = 1.0
            right[c, t + 1] = int(a.argmax()) == y[t]
            W[c] += ((1 / 64) * (tg - a) / n2)[:, None] * U
        else:
            e = int(a.argmax())
            if gen[c].random() < 1 / 16:
                sv = np.flatnonzero(seen); e = int(sv[gen[c].integers(0, len(sv))])
            ok = e == y[t]; right[c, t + 1] = ok
            if ok:
                dq = (1 / 16) * (0.0 - a) / n2; dq[e] = (1 / 16) * (1.0 - a[e]) / n2[e]; W[c] += dq[:, None] * U
            else:
                W[c][e] += (1 / 16) * (0.0 - a[e]) / n2[e] * U[e]
    if (t + 2) % 8192 == 0:
        for c in range(C):
            nv = np.linalg.norm(W[c], axis=1); norms[c].append(dict(byte=t + 2, max=float(nv.max()), median=float(np.median(nv))))
refs = SE.references(L["bytes"][:NB])
wins = [(a_, min(a_ + 4096, NB)) for a_ in range(0, NB, 4096)]
wsh = lambda ok: [float(ok[max(a_, 1):e_].mean()) for a_, e_ in wins]
lq = {k: float((refs[k] == b)[q0:].mean()) for k in ("ref_1", "ref_2", "ref_3")}; lq["end mark"] = float((b[q0:] == 10).mean())
res = {}
for c, (rule, f) in enumerate(CFG):
    nm = f"{rule} {'(i) x' if f == 1 else '(ii) x s2' if f == 2 else '(iii) x s2 s4' if f == 3 else '(iv) x s2 s4 s8'}"
    res[nm] = dict(windows=wsh(right[c]), last_quarter=float(right[c, q0:].mean()), norms=norms[c])
out = dict(tag=tag, seed=seed, bytes=NB, kind=kind, cells_a=K, inputs_per_driver=dict(min=int(min(map(len, own))), median=float(np.median(list(map(len, own)))), max=G),
           rules=res, levels_last_quarter=lq, seconds=time.time() - t0)
json.dump(out, open(f"/home/user/tissue_check/out/trace_{tag}_f{''.join(map(str, forms))}.json", "w"), indent=0)
print(f"== {tag}: refs last quarter " + " ".join(f"{k} {100 * v:.2f}" for k, v in lq.items()) + f"; {time.time() - t0:.0f} s")
for k, v in res.items(): print(f"   {k:20s} last quarter {100 * v['last_quarter']:5.2f}  largest row norm {v['norms'][-1]['max']:.3g}")
