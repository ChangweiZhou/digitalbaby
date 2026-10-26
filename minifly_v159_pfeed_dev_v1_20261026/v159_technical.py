"""V159 C1 technical (design sections 2 and 6). Writes results/technical/CHECKS.json.
Run through launch.py:  python launch.py run --stage technical -- /tmp/venv_n/bin/python v159_technical.py [--variant]

checks  delta-rule step equals the formula; the feedback trace holds only earlier predictions (no look-ahead); determinism
X1      same-future index of r on V155's fixture streams (pass >= 0.5 on both)
X2      PF-READ and SHUFFLED on the class-transfer task, readout as redesigned in section 6 (pass: PF-READ >= 0.80 on
        unseen members on both seeds, FROZEN (testbed attempt 2) and SHUFFLED <= 0.65)
--variant runs the design's one allowed variant (gamma 0.5) for X2 only, if the first X2 failed.
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
import runtime  # noqa: E402  (V159's own; before the V155 modules)
import v159_streams as ST  # noqa: E402
import v159_readout as RD  # noqa: E402
import v159_pf as PFM  # noqa: E402
import v159_testbed as TB  # noqa: E402  (birth keys, SHA-checked; X2 scoring helpers)
from v159_streams import codes  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
SET = ST.SET
VARIANT = "--variant" in sys.argv
GAMMA = SET["pf"]["variant_gamma"] if VARIANT else 0.0
out, records = [], 0


def check(name, ok, detail=None):
    out.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name, "" if detail is None else detail, flush=True)


def cosine_index(Rv, last, seed):
    """V155's pair sampling (seeds 11 + stream seed), with the cosine of r in place of code overlap / 184."""
    g = np.random.default_rng(SET["x2"].get("pair_seed_base", 11) + seed)
    rows = np.flatnonzero(np.isin(last, list(ST.S1 + ST.S2)))
    setof = np.where(np.isin(last, list(ST.S1)), 1, 2)
    if hasattr(Rv, "tocsr"):                       # binary KC codes: cosine from sparse rows
        Xc = Rv.tocsr()
        nrm = np.sqrt(np.maximum(np.asarray(Xc.multiply(Xc).sum(1)).ravel(), 1e-12))
        sim = lambda i, j: float(Xc[i].multiply(Xc[j]).sum() / (nrm[i] * nrm[j]))      # noqa: E731
    else:
        U = Rv / np.maximum(np.linalg.norm(Rv, axis=1, keepdims=True), 1e-12)
        sim = lambda i, j: float(U[i] @ U[j])      # noqa: E731
    vals = {"within": [], "between": []}
    while min(len(v) for v in vals.values()) < 2000:
        i, j = g.choice(rows, 2, replace=False)
        if last[i] == last[j]:
            continue
        key = "within" if setof[i] == setof[j] else "between"
        if len(vals[key]) < 2000:
            vals[key].append(sim(i, j))
    return float(np.mean(vals["within"]) - np.mean(vals["between"]))


bd = TB.birth()
n_kc = bd["B"].shape[1]
t_start = time.process_time()

if not VARIANT:
    # checks on a small class-transfer stream
    d = ST.transfer_set(0)["train"]
    X = codes.codes(d["P"], bd["B"], bd["pn"], bd["side"], bd["k"])
    A, T = PFM.actives(X), PFM.targets(d["y"])
    pf = PFM.PF(n_kc, eta=0.1)
    rng = np.random.default_rng(1)
    pf.M[:] = rng.normal(0, 0.1, pf.M.shape)
    t = next(i for i in range(5, 200) if len(A[i]))
    x = np.zeros(n_kc + 1)
    x[A[t]] = 1.0 / len(A[t])
    x[-1] = 1.0
    exp = pf.M + 0.1 * np.outer(T[t] - pf.M @ x, x)
    pf.step(A[t], T[t])
    check("delta-rule step equals M + eta (target - M x) x^T", np.allclose(pf.M, exp, atol=1e-12),
          float(np.abs(pf.M - exp).max()))
    pf2 = PFM.PF(n_kc, eta=0.1).learn(X[:400], d["y"][:400], np.array([400]), 1)
    Rv = pf2.predict(X[:400])
    check("batch prediction equals the per-position prediction", np.allclose(Rv[37], pf2.predict_one(A[37]), atol=1e-12))
    # no look-ahead: changing everything after position t leaves z_t unchanged
    n = d["n"][:3]
    Z = RD.past_trace(Rv[:n.sum()], n, SET["pf"]["fade"])
    R2 = Rv[:n.sum()].copy()
    R2[20:] = rng.normal(size=R2[20:].shape)
    Z2 = RD.past_trace(R2, n, SET["pf"]["fade"])
    check("feedback trace at t uses only predictions before t", np.array_equal(Z[:21], Z2[:21]) and not np.array_equal(Z[21:], Z2[21:]))
    check("feedback trace resets at each line start", not Z[0].any() and not Z[n[0]].any())
    a = PFM.PF(n_kc).learn(X[:300], d["y"][:300], np.array([300]), 2).M
    b = PFM.PF(n_kc).learn(X[:300], d["y"][:300], np.array([300]), 2).M
    check("learning is deterministic", np.array_equal(a, b))
    records += 2

    # X1: same-future index of r on V155's fixture streams
    x1 = {}
    for seed in SET["x1"]["seeds"]:
        P, y, n = ST.traces(f"x1_{seed}", ST.samefuture_stream(seed))
        X = codes.codes(P, bd["B"], bd["pn"], bd["side"], bd["k"])
        last = ST.previous_bytes(y, n, 1)
        pf = PFM.PF(n_kc, eta=SET["pf"]["eta"]).learn(X, y, n, SET["pf"]["passes"])
        records += SET["pf"]["passes"] * len(n)
        x1[seed] = {"FROZEN_codes": cosine_index(X, last, seed),
                    "PF_r": cosine_index(pf.predict(X), last, seed)}
    check("X1: same-future index of r >= 0.5 on both streams",
          all(v["PF_r"] >= SET["x1"]["pass_index"] for v in x1.values()), x1)

# X2: PF-READ and SHUFFLED
tb = json.loads((HERE / "results" / "testbed" / "TESTBED_X2_ATTEMPT2.json").read_text())
x2s = SET["x2"]
x2 = {}
for seed in x2s["seeds"]:
    data = ST.transfer_set(seed)
    Xs = {p: codes.codes(data[p]["P"], bd["B"], bd["pn"], bd["side"], bd["k"]) for p in data}
    pf = PFM.PF(n_kc, eta=SET["pf"]["eta"], gamma=GAMMA).learn(Xs["train"], data["train"]["y"], data["train"]["n"],
                                                              SET["pf"]["passes"])
    records += SET["pf"]["passes"] * len(data["train"]["n"])
    Rv = {p: pf.predict(Xs[p]) for p in data}
    perm = {p: np.random.default_rng(SET["pf"]["shuffle_seed"]).permutation(len(Rv[p])) for p in data}
    row = {}
    for arm, src in (("PF-READ", Rv), ("SHUFFLED", {p: Rv[p][perm[p]] for p in data})):
        E = {p: RD.past_trace(src[p], data[p]["n"], SET["pf"]["fade"]) for p in data}
        scale = RD.block_scale(Xs["train"], E["train"])
        F = {p: RD.features(Xs[p], E[p], scale) for p in data}
        cls, theta, info = RD.r1(F["train"], data["train"]["y"], F["calib"], data["calib"]["y"], x2s["r1_grid"],
                                 x2s["maxiter"])
        prev2 = ST.previous_bytes(data["test"]["y"], data["test"]["n"], 2)
        sc = RD.g_score(cls, theta, F["test"], data["test"]["y"], prev2, ST.G,
                        lambda b: (b in ST.S1[:4]) or (b in ST.S2[:4]))
        row[arm] = {**sc, **info, "block_scale": scale}
    row["FROZEN_testbed"] = tb["x2"][str(seed)]["FROZEN"]["unseen"]
    row["ORACLE_testbed"] = tb["x2"][str(seed)]["ORACLE"]["unseen"]
    x2[seed] = row
    print("X2 seed", seed, {a: (round(row[a]["unseen"]["accuracy"], 3), round(row[a]["seen"]["accuracy"], 3), row[a]["lam"])
                            for a in ("PF-READ", "SHUFFLED")}, flush=True)
x2_pass = all(r["PF-READ"]["unseen"]["accuracy"] >= x2s["pf_min"] and r["SHUFFLED"]["unseen"]["accuracy"] <= x2s["frozen_max"]
              and r["FROZEN_testbed"]["accuracy"] <= x2s["frozen_max"] for r in x2.values())
check(f"X2{' (variant gamma 0.5)' if VARIANT else ''}: PF-READ >= 0.80 on unseen members, FROZEN and SHUFFLED <= 0.65, "
      "both seeds", x2_pass, {s: {a: x2[s][a]["unseen"]["accuracy"] for a in ("PF-READ", "SHUFFLED")} for s in x2})

d = HERE / "results" / STAGE
d.mkdir(parents=True, exist_ok=True)
name = "CHECKS_VARIANT.json" if VARIANT else "CHECKS.json"
(d / name).write_text(json.dumps({"checks": out, "x2": x2, "x2_pass": bool(x2_pass), "gamma": GAMMA,
                                  "learner_records": records, "cpu_s": round(time.process_time() - t_start, 1)}, indent=1))
print(sum(o["pass"] for o in out), "/", len(out), "pass; learner records", records)
runtime.write_closure()
sys.exit(0 if all(o["pass"] for o in out) else 1)
