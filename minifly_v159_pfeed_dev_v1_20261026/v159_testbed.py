"""V159 C0 testbed (design section 2), evaluator-side, before any predictive-layer code.
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_n/bin/python v159_testbed.py
  X2 ceiling/floor: FROZEN and ORACLE readouts on the class-transfer streams (both seeds)
  X4: EVEN keys (equal synapse count per receptor channel) scored by R1 on Thursday against V155's FROZEN
Writes results/testbed/TESTBED.json.
"""
import hashlib
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
import runtime  # noqa: E402  (V159's own; imported before the V155 modules)
import v159_streams as ST  # noqa: E402
import v159_readout as RD  # noqa: E402
from v159_streams import codes  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
SET = ST.SET
V155 = ST.V155


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def birth():
    path = V155 / "cache" / "birth.npz"
    rec = json.loads((V155 / "results" / "testbed" / "cache_thursday_A.json").read_text())
    if sha(path) != rec["birth_sha256"]:
        raise SystemExit("V155 birth.npz differs from its recorded SHA-256")
    return codes.load_birth(path)


def oracle_extra(y, n):
    """Trace (same fade, reset per line) of the true class of each byte read: 2 channels."""
    E = np.zeros((len(y), 2))
    for t, b in enumerate(y):
        c = ST.byte_class(int(b)) if b != ST.EOS else 0
        if c:
            E[t, c - 1] = 1.0
    return RD.past_trace(E, n, SET["pf"]["fade"])


def x2_arm(bd, data, extra_fn, seed):
    x2 = SET["x2"]
    Xs = {p: codes.codes(data[p]["P"], bd["B"], bd["pn"], bd["side"], bd["k"]) for p in ("train", "calib", "test")}
    Es = {p: None if extra_fn is None else extra_fn(data[p]["y"], data[p]["n"]) for p in Xs}
    scale = RD.block_scale(Xs["train"], Es["train"]) if extra_fn is not None and x2.get("block_normalise") else 1.0
    Fs = {p: RD.features(Xs[p], Es[p], scale) for p in Xs}
    cls, theta, info = RD.r1(Fs["train"], data["train"]["y"], Fs["calib"], data["calib"]["y"], x2["r1_grid"], x2["maxiter"])
    prev2 = ST.previous_bytes(data["test"]["y"], data["test"]["n"], 2)
    score = RD.g_score(cls, theta, Fs["test"], data["test"]["y"], prev2, ST.G,
                       lambda b: (b in ST.S1[:4]) or (b in ST.S2[:4]))
    return {**score, **info, "block_scale": scale}


def even_keys(bd, seed):
    """Each synapse keeps its KC and weight; sources reassigned so every receptor channel has the same number of
    synapses (+-1), a projection neuron of that channel chosen uniformly."""
    C = bd["B"].tocoo()
    g = np.random.default_rng(seed)
    pn = bd["pn"]
    nnz, nch = C.nnz, 88
    counts = np.full(nch, nnz // nch) + (np.arange(nch) < nnz % nch)
    chans = g.permutation(np.repeat(np.arange(nch), counts))
    by_ch = [np.flatnonzero(pn == c) for c in range(nch)]
    rows = np.array([by_ch[c][g.integers(len(by_ch[c]))] for c in chans])
    B = sparse.csr_matrix((C.data, (rows, C.col)), shape=C.shape)
    return B, {"synapses": int(nnz), "distinct_after_merge": int(B.nnz), "per_channel": [int(counts.min()), int(counts.max())]}


def main():
    only_x2 = "--only-x2" in sys.argv          # C0 attempt 2 (design section 6): X4 already scored in attempt 1
    t0 = time.process_time()
    bd = birth()
    res = {"stage": STAGE, "settings": SET, "x2": {}, "learner_records": 0}
    for seed in SET["x2"]["seeds"]:
        data = ST.transfer_set(seed)
        row = {"positions": {p: int(len(data[p]["y"])) for p in data},
               "FROZEN": x2_arm(bd, data, None, seed), "ORACLE": x2_arm(bd, data, oracle_extra, seed)}
        res["x2"][str(seed)] = row
        print("X2 seed", seed, {a: (round(row[a]["unseen"]["accuracy"], 3), round(row[a]["seen"]["accuracy"], 3))
                                for a in ("FROZEN", "ORACLE")}, flush=True)
    x2 = SET["x2"]
    rows = res["x2"].values()
    res["x2_checks"] = {"ceiling_oracle_unseen": all(r["ORACLE"]["unseen"]["accuracy"] >= x2["oracle_min"] for r in rows),
                        "floor_frozen_unseen": all(r["FROZEN"]["unseen"]["accuracy"] <= x2["frozen_max"] for r in rows)}
    res["x2_verdict"] = "CONTINUE" if all(res["x2_checks"].values()) else "CANNOT_TEST_CLAIM_AS_SPECIFIED"

    # X4: EVEN keys on Thursday (V155 cache, SHA-verified), against V155's recorded FROZEN lines
    if only_x2:
        return finish(res, t0)
    rec = json.loads((V155 / "results" / "testbed" / "cache_thursday_A.json").read_text())
    cpath = V155 / "cache" / "thursday_A.npz"
    if sha(cpath) != rec["cache_sha256"]:
        raise SystemExit("V155 Thursday cache differs from its recorded SHA-256")
    z = np.load(cpath)
    v155 = json.loads((V155 / "results" / "testbed" / "SCORES_thursday.json").read_text())
    B, info = even_keys(bd, SET["x4"]["seed"])
    X = {k: codes.codes(z[f"P_{k}"], B, bd["pn"], bd["side"], bd["k"]) for k in ("train", "calib", "held")}
    cls, theta, r1info = RD.r1(RD.features(X["train"]), z["y_train"], RD.features(X["calib"]), z["y_calib"],
                               SET["x3"]["r1_grid_A"], SET["x3"]["maxiter"])
    P = RD.R.predict_r1(theta, RD.features(X["held"]), cls.K)
    bits = cls.bits(P, z["y_held"])
    lines = RD.line_sums(bits, z["n_held"])
    sys.path.insert(1, str(V155))
    import summarize as v155_summarize  # noqa: E402  (V155's line bootstrap)
    f_lines = v155["arms"]["FROZEN"]["lines"]
    res["x4"] = {"EVEN_held_bpb": float(bits.mean()), "FROZEN_held_bpb_v155": v155["arms"]["FROZEN"]["held_bpb"],
                 "difference": float(bits.mean()) - v155["arms"]["FROZEN"]["held_bpb"],
                 "bootstrap_sd": v155_summarize.boot_sd(lines, f_lines, z["n_held"]),
                 "usage_unused_kcs_train": int((np.asarray(X["train"].sum(0)).ravel() == 0).sum()),
                 "keys": info, **r1info}
    print("X4", {k: res["x4"][k] for k in ("EVEN_held_bpb", "difference", "bootstrap_sd")}, flush=True)
    finish(res, t0)


def finish(res, t0):
    res["cpu_s"] = round(time.process_time() - t0, 1)
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    name = "TESTBED_X2_ATTEMPT2.json" if "--only-x2" in sys.argv else "TESTBED.json"
    (d / name).write_text(json.dumps(res, indent=1))
    print("X2 verdict", res["x2_verdict"], res["x2_checks"])
    runtime.write_closure()


if __name__ == "__main__":
    main()
