"""V158 C0 testbed (design sections 2 and 7), evaluator-side, before any V158 learner code.
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_n/bin/python testbed.py
On both fixture streams: the overlap index of FROZEN (birth keys), ORACLE (keys built from the set means) and ORACLE-CODE
(FROZEN codes with a fixed shared unit set swapped in per set). Writes results/testbed/TESTBED.json.
"""
import json
import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime  # noqa: E402  (V158's own, imported before V155 modules)
import samefuture as fixture  # noqa: E402  (named samefuture: MiniFly has its own "fixture" module)
from samefuture import codes  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
OR = json.loads((HERE / "SETTINGS.json").read_text())["oracle"]


def oracle_units(side, seed):
    g = np.random.default_rng(seed)
    n = OR["units_per_set_per_side"]
    out = {1: [], 2: []}
    for s in (0, 1):
        ix = g.permutation(np.flatnonzero(side == s))
        out[1].append(ix[:n])
        out[2].append(ix[n:2 * n])
    return {k: np.concatenate(v) for k, v in out.items()}


def oracle_keys(bd, P, last, units):
    B = bd["B"].toarray()
    Q = P[:, bd["pn"]]
    for k, sset in ((1, fixture.S1), (2, fixture.S2)):
        qbar = Q[np.isin(last, list(sset))].mean(0)
        B[:, units[k]] = (OR["column_sum"] * qbar / qbar.sum())[:, None]
    return sparse.csr_matrix(B)


def oracle_code(X, R, last, units, side, k):
    """FROZEN codes; at positions ending in a set byte, the lowest-ranked active units on each side are replaced by
    that set's fixed units (46 per side), keeping 92 active per side."""
    rows = []
    for i in range(X.shape[0]):
        act = X.indices[X.indptr[i]:X.indptr[i + 1]]
        set_k = 1 if last[i] in fixture.S1 else 2 if last[i] in fixture.S2 else 0
        if set_k:
            new = []
            for s in (0, 1):
                a_s = act[side[act] == s]
                u_s = units[set_k][side[units[set_k]] == s]
                keep = [u for u in a_s[np.argsort(-R[i, a_s], kind="stable")] if u not in set(u_s)]
                keep = keep[:max(0, len(a_s) - len(u_s))]
                new.append(np.concatenate([np.array(keep, dtype=np.int64), u_s]))
            act = np.sort(np.concatenate(new))
        rows.append(act)
    indptr = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
    return sparse.csr_matrix((np.ones(indptr[-1]), np.concatenate(rows), indptr), shape=X.shape)


def main():
    t0 = time.process_time()
    bd = fixture.birth()
    units = oracle_units(bd["side"], OR["seed"])
    res = {"stage": STAGE, "settings": {"oracle": OR, "fixture": fixture.SET}, "streams": {}, "learner_records": 0}
    for seed in fixture.SET["seeds"]:
        P, last, n = fixture.traces(seed)
        X = codes.codes(P, bd["B"], bd["pn"], bd["side"], bd["k"])
        R = codes.raw(P, bd["B"], bd["pn"])
        Xo = codes.codes(P, oracle_keys(bd, P, last, units), bd["pn"], bd["side"], bd["k"])
        Xc = oracle_code(X, R, last, units, bd["side"], bd["k"])
        row = {"positions": len(P), "FROZEN": fixture.index(X, last, seed), "ORACLE": fixture.index(Xo, last, seed),
               "ORACLE-CODE": fixture.index(Xc, last, seed),
               "oracle_units_active_in_set_contexts": float(np.mean([len(np.intersect1d(
                   Xo.indices[Xo.indptr[i]:Xo.indptr[i + 1]], units[1 if last[i] in fixture.S1 else 2]))
                   for i in np.flatnonzero(np.isin(last, list(fixture.S1 + fixture.S2)))]))}
        res["streams"][str(seed)] = row
        print(seed, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    S = res["streams"].values()
    checks = {"floor_frozen_at_most": all(r["FROZEN"] <= OR["floor"] for r in S),
              "measurement_oracle_code": all(r["ORACLE-CODE"] - r["FROZEN"] >= OR["ceiling"] for r in S),
              "ceiling_oracle_keys": all(r["ORACLE"] - r["FROZEN"] >= OR["ceiling"] for r in S)}
    if not checks["measurement_oracle_code"] or not checks["floor_frozen_at_most"]:
        verdict = "INVALID"
    elif checks["ceiling_oracle_keys"]:
        verdict = "CONTINUE"
    else:
        verdict = "CANNOT_TEST_CLAIM_AS_SPECIFIED"
    res.update({"checks": checks, "verdict": verdict, "cpu_s": round(time.process_time() - t0, 1)})
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "TESTBED.json").write_text(json.dumps(res, indent=1))
    print(verdict, checks)
    runtime.write_closure()


if __name__ == "__main__":
    main()
