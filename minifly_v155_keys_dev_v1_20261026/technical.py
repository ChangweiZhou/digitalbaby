"""V155 technical checks (C1, MiniFly environment). Writes results/technical/CHECKS.json.
Run through launch.py:  python launch.py run --stage technical -- /tmp/venv_n/bin/python technical.py

1  Keys.code equals codes.codes (and so MiniFly's encode, by C0's parity) on cached Thursday positions
2  L-COMP: one update equals the formula computed independently; column sums and non-negativity hold
3  L-PRED: one update (forward and retrograde terms, W) equals the formula computed independently; sums, signs, clip
4  the shared key data: Keys.BT products equal a freshly built matrix after updates
5  determinism: two L-PRED runs give identical keys and W
6  the design 12.10 fixture: L-PRED raises the same-future code-overlap index over FROZEN by >= 0.01 on both streams
7  cost per position for both rules on 2 KB of Thursday, and projections for C2 and C3
"""
import json
import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cache  # noqa: E402  (MiniFly front end for the synthetic fixture; V153 states)
import codes  # noqa: E402
import rules  # noqa: E402

CACHE = HERE / "cache"
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
SET = json.loads((HERE / "SETTINGS.json").read_text())["rules"]
out, records = [], 0


def check(name, ok, detail=None):
    out.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name, "" if detail is None else detail, flush=True)


birth = codes.load_birth(CACHE / "birth.npz")
z = np.load(CACHE / "thursday_A.npz")
P, lines = z["P_train"], z["n_train"]

# 1 code parity
K = rules.Keys(birth)
C = codes.codes(P[:400], birth["B"], birth["pn"], birth["side"], birth["k"])
check("Keys.code equals codes.codes on 400 cached positions",
      all(np.array_equal(K.code(K.q(P[i])), C.indices[C.indptr[i]:C.indptr[i + 1]]) for i in range(400)))

# 2 L-COMP one step against an independent dense computation
Bd = birth["B"].toarray()
mask = Bd > 0
cj = Bd.sum(0)
ne = lambda i: C.indptr[i + 1] > C.indptr[i]                                      # noqa: E731
t = next(i for i in range(5, 398) if ne(i) and ne(i + 1) and ne(i - 3))
q = K.q(P[t])
act = K.code(q)
exp = Bd.copy()
for j in act:
    S = mask[:, j]
    if q[S].sum() > 0:
        exp[S, j] += 0.03 * (cj[j] * q[S] / q[S].sum() - exp[S, j])
rules.comp_step(K, q, 0.03)
got = K.csr().toarray()
check("L-COMP one update equals the formula", np.allclose(got, exp, rtol=0, atol=1e-12),
      float(np.abs(got - exp).max()))
check("L-COMP keeps column sums and signs", np.allclose(got.sum(0), cj, rtol=1e-12, atol=1e-12) and got.min() >= 0
      and not got[~mask].any())
records += 1

# 3 L-PRED one step (forward + retrograde + W) against an independent dense computation
K = rules.Keys(birth)
pr = rules.Pred(K, 0.03, SET["margin"], 7)
rng = np.random.default_rng(3)
pr.W[:] = rng.uniform(-0.3, 0.3, pr.W.shape).astype(np.float32)
t0, t1 = t, t + 1
pr.seen = [t0 - 3]                                  # one earlier position to replay as the negative
Q = P[:, K.pn].astype(np.float64)
a = K.code(Q[t0]); b = K.code(Q[t1]); bn = K.code(Q[t0 - 3])
W0 = pr.W.astype(np.float64).copy()
exp = Bd.copy()
pairs = []
s = W0[np.ix_(b, a)].mean()
if s < SET["margin"]:
    pairs.append((b, Q[t1], 1.0))
s = W0[np.ix_(bn, a)].mean() if len(bn) else None
if len(bn) and s > -SET["margin"]:
    pairs.append((bn, Q[t0 - 3], -1.0))
touched = set()
for bb, qq, g in pairs:
    fwd = W0[np.ix_(bb, a)].mean(1)
    back = W0[np.ix_(bb, a)].mean(0)
    for units, qin, pred in ((bb, qq, fwd), (a, Q[t0], back)):
        for u, j in enumerate(units):
            S = mask[:, j]
            if qin[S].sum() > 0:
                exp[S, j] += 0.03 * cj[j] * g * pred[u] * qin[S] / qin[S].sum()
            touched.add(j)
for j in touched:
    col = np.maximum(exp[:, j], 0) * mask[:, j]
    exp[:, j] = col * cj[j] / col.sum() if col.sum() > 0 else Bd[:, j]
Wexp = W0.copy()
for bb, _, g in pairs:
    Wexp[np.ix_(bb, a)] = np.clip(Wexp[np.ix_(bb, a)] + 0.03 * g, -1, 1)
pr.rng = np.random.default_rng(0)                   # the only seen position is drawn whatever the seed
pr.step(Q[t0], Q[t1], Q, t0)
got = K.csr().toarray()
check("L-PRED one update (forward, retrograde, rescaling) equals the formula", len(pairs) == 2 and
      np.allclose(got, exp, rtol=0, atol=1e-10), {"pairs": len(pairs), "max_abs_diff": float(np.abs(got - exp).max())})
check("L-PRED W update equals the formula and stays in [-1, 1]",
      np.allclose(pr.W, Wexp, atol=1e-6) and np.abs(pr.W).max() <= 1, float(np.abs(pr.W - Wexp).max()))
check("L-PRED keeps column sums, signs and wiring", np.allclose(got.sum(0), cj, rtol=1e-9, atol=1e-12) and got.min() >= 0
      and not got[~mask].any())
records += 1

# 4 shared data
qq = K.q(P[t + 7])
check("Keys.BT products track the updated keys", np.allclose(K.BT @ qq, K.csr().T @ qq, atol=1e-12))

# 5 determinism (30 records, 1 pass, L-PRED at the middle rate)
sub = int(lines[:30].sum())
r1 = rules.run("L-PRED", birth, P[:sub], lines[:30], 0.03, 1, SET)
r2 = rules.run("L-PRED", birth, P[:sub], lines[:30], 0.03, 1, SET)
check("L-PRED is deterministic", np.array_equal(r1[1][0].data, r2[1][0].data) and np.array_equal(r1[1][1].W, r2[1][1].W),
      r1[2])
records += 60

# 6 fixture: identical futures, identical pasts (design 12.10)
S1, S2, F1, F2 = b"abcdefgh", b"ijklmnop", b"qrstuvwx", b"yzABCDEF"


def stream(seed, n_lines=300, length=40):
    g = np.random.default_rng(seed)
    out_lines = []
    for _ in range(n_lines):
        cur = (S1 + S2)[g.integers(16)]
        line = [cur]
        while len(line) < length:
            if cur in S1:
                cur = F1[g.integers(8)]
            elif cur in S2:
                cur = F2[g.integers(8)]
            else:
                cur = (S1 + S2)[g.integers(16)]
            line.append(cur)
        out_lines.append(bytes(line))
    return out_lines


def overlap_index(X, last, g):
    rows = {kk: np.flatnonzero(np.isin(last, list(sset))) for kk, sset in (("S1", S1), ("S2", S2))}
    allr = np.concatenate([rows["S1"], rows["S2"]])
    setof = np.where(np.isin(last, list(S1)), 1, 2)
    vals = {"within": [], "between": []}
    while min(len(v) for v in vals.values()) < 2000:
        i, j = g.choice(allr, 2, replace=False)
        if last[i] == last[j]:
            continue
        key = "within" if setof[i] == setof[j] else "between"
        if len(vals[key]) < 2000:
            ov = len(np.intersect1d(X.indices[X.indptr[i]:X.indptr[i + 1]], X.indices[X.indptr[j]:X.indptr[j + 1]]))
            vals[key].append(ov / 184)
    return float(np.mean(vals["within"]) - np.mean(vals["between"]))


fx = {}
mem = cache.states.new_memory(cache.CELL)
for seed in (0, 1):
    recs = stream(seed)
    Ps, last = [], []
    for r in recs:
        ps, _ = cache.record_traces(mem, r)
        Ps += ps
        last += [0] + list(r)              # byte before each position (0 = line start)
    Ps, last = np.array(Ps), np.array(last)
    nl = np.array([len(r) + 1 for r in recs])
    row = {}
    for name in ("FROZEN", "L-COMP", "L-PRED"):
        if name == "FROZEN":
            B = birth["B"]
        else:
            snaps, _, cnt = rules.run(name, birth, Ps, nl, 0.03, 2, SET, checkpoints=(2,))
            B = snaps[2]
            records += 2 * len(recs)
        X = codes.codes(Ps, B, birth["pn"], birth["side"], birth["k"])
        row[name] = overlap_index(X, last, np.random.default_rng(11 + seed))
    fx[seed] = row
check("fixture: L-PRED raises the same-future overlap index over FROZEN by >= 0.01 on both streams",
      all(fx[s]["L-PRED"] - fx[s]["FROZEN"] >= 0.01 for s in fx), fx)

# 7 cost on 2 KB of Thursday (about 35 records), one pass each
n2 = int(np.searchsorted(np.cumsum(lines), 2048)) + 1
sub = int(lines[:n2].sum())
cost = {}
for name in ("L-COMP", "L-PRED"):
    t_ = time.process_time()
    rules.run(name, birth, P[:sub], lines[:n2], 0.03, 1, SET)
    cost[name] = (time.process_time() - t_) / sub
    records += n2
pos_A = len(P)
proj = {name: {"cpu_s_per_pass_40KB": round(cost[name] * pos_A, 1)} for name in cost}
check("cost measured", True, {"cpu_ms_per_position": {k: round(v * 1e3, 3) for k, v in cost.items()},
                              "positions_taught_thursday": pos_A, "projection": proj})

d = HERE / "results" / STAGE
d.mkdir(parents=True, exist_ok=True)
(d / "CHECKS.json").write_text(json.dumps({"checks": out, "learner_records": records}, indent=1))
print(sum(o["pass"] for o in out), "/", len(out), "pass; learner records", records)
import runtime  # noqa: E402
runtime.write_closure()
sys.exit(0 if all(o["pass"] for o in out) else 1)
