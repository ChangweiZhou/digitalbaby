"""V162 C2 dev (design section 3): text carry-over, Thursday (A) -> Edgeworth (B).
Run through launch.py:  python launch.py run --stage dev -- /tmp/venv_n/bin/python v162_dev.py
Predictive layer learned on A's taught text and frozen; on B: S (R1 on the KC code), G (R1 on the centred trace, fade 0.3),
familiarity from B's taught text; GATED and GATED-SHUF with (beta, tau, a_low) or "S alone" chosen on B's calibration bits.
Pass: GATED beats S on B held-out by >= 0.02 bits and >= 3 line-bootstrap SDs, and GATED-SHUF does not.
Writes results/dev/RESULTS.json.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime  # noqa: E402
import v162_lib as V  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
V155 = V.ST.V155


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def b_traces():
    """Edgeworth (role B: 10/5/10 KB) traces, cached in this folder."""
    path = V.ST.CACHE / "edgeworth_B.npz"
    if path.exists():
        return dict(np.load(path))
    import corpus as v155_corpus  # noqa: E402  (V155: pinned texts and V148 split rules)
    sp = v155_corpus.splits("edgeworth", "B")
    mem = V.ST.v155_cache.states.new_memory(V.ST.v155_cache.CELL)
    out = {}
    for k in ("train", "calib", "held"):
        P, y = [], []
        for r in sp[k]:
            ps, _ = V.ST.v155_cache.record_traces(mem, r)
            P += ps
            y += list(r) + [V.ST.EOS]
        out[f"P_{k}"], out[f"y_{k}"], out[f"n_{k}"] = np.array(P), np.array(y), np.array([len(r) + 1 for r in sp[k]])
    V.ST.CACHE.mkdir(exist_ok=True)
    np.savez(path, **out)
    return out


def combos(b):
    yield {"name": "S alone"}
    for beta in b["beta_grid"]:
        for tp in b["tau_percentiles"]:
            for al in (b["a_low_grid"] if tp is not None else [1.0]):
                yield {"beta": beta, "tau_pct": tp, "a_low": al}


def main():
    t0 = time.process_time()
    b = V.SET["b"]
    m = V.SET["mech"]
    bd = V.TB.birth()
    n_kc = bd["B"].shape[1]
    codes = V.L.codes
    # A: Thursday taught text (V155 cache, SHA-verified) -> predictive layer
    rec = json.loads((V155 / "results" / "testbed" / "cache_thursday_A.json").read_text())
    apath = V155 / "cache" / "thursday_A.npz"
    if sha(apath) != rec["cache_sha256"]:
        raise SystemExit("V155 Thursday cache differs from its recorded SHA-256")
    za = np.load(apath)
    XA = codes.codes(za["P_train"], bd["B"], bd["pn"], bd["side"], bd["k"])
    s = V.ST.SET["pf"]
    pf = V.PFM.PF(n_kc, eta=s["eta"]).learn(XA, za["y_train"], za["n_train"], s["passes"])
    records = s["passes"] * len(za["n_train"])
    # B: Edgeworth
    zb = b_traces()
    sp = ("train", "calib", "held")
    X = {k: codes.codes(zb[f"P_{k}"], bd["B"], bd["pn"], bd["side"], bd["k"]) for k in sp}
    y = {k: zb[f"y_{k}"] for k in sp}
    n = {k: zb[f"n_{k}"] for k in sp}
    F = {k: V.RD.features(X[k]) for k in sp}
    cls, theta, iS = V.RD.r1(F["train"], y["train"], F["calib"], y["calib"], b["s_grid"], b["maxiter"])
    PS = {k: V.RD.R.predict_r1(theta, F[k], cls.K) for k in ("calib", "held")}
    E = V.G161.evidence(X["train"], y["train"], cls)
    fam = {k: V.G161.familiarity(X[k], PS[k].argmax(1), E)[0] for k in ("calib", "held")}
    nz = np.asarray(X["calib"].sum(1)).ravel() > 0
    Rv = {k: pf.predict(X[k]) for k in sp}
    perm_seed = s["shuffle_seed"]
    res = {"stage": STAGE, "settings": V.SET, "S_fit": {k: iS[k] for k in ("lam", "calib_bpb", "lam_at_grid_edge")},
           "positions": {k: int(len(y[k])) for k in sp}, "arms": {}}
    S_bits = cls.bits(PS["held"], y["held"])
    S_lines = V.RD.line_sums(S_bits, n["held"])
    res["arms"]["S"] = {"held_bpb": float(S_bits.mean()), "calib_bpb": float(cls.bits(PS["calib"], y["calib"]).mean())}
    for arm, shuffle in (("GATED", False), ("GATED-SHUF", True)):
        Vv = {k: (Rv[k][np.random.default_rng(perm_seed).permutation(len(Rv[k]))] if shuffle else Rv[k]) for k in sp}
        cG, PGd, iG = V.trace_expert(Vv, n, y, ("train", "calib", "held"), m["fade"], b["g_grid"], b["maxiter"])
        assert list(cG.seen) == list(cls.seen)
        grid = []
        for c in combos(b):
            if "name" in c:
                Pc = PS["calib"]
            else:
                tau = None if c["tau_pct"] is None else float(np.percentile(fam["calib"][nz], c["tau_pct"]))
                c["tau"] = tau
                Pc = V.soft_gate(PS["calib"], PGd["calib"], fam["calib"], tau, c["a_low"], c["beta"])
            c["calib_bpb"] = float(cls.bits(Pc, y["calib"]).mean())
            grid.append(c)
        best = min(grid, key=lambda c: c["calib_bpb"])
        if "name" in best:
            Ph = PS["held"]
        else:
            Ph = V.soft_gate(PS["held"], PGd["held"], fam["held"], best["tau"], best["a_low"], best["beta"])
        bits = cls.bits(Ph, y["held"])
        lines = V.RD.line_sums(bits, n["held"])
        gain = float(S_bits.mean() - bits.mean())
        sd = V.boot_sd(S_lines, lines, n["held"])
        res["arms"][arm] = {"chosen": best, "held_bpb": float(bits.mean()), "gain_over_S": gain, "bootstrap_sd": sd,
                            "G_fit": {k: iG[k] for k in ("lam", "calib_bpb", "lam_at_grid_edge")},
                            "G_alone_held_bpb": float(cG.bits(PGd["held"], y["held"]).mean()), "grid": grid}
        print(arm, "chosen", {k: best.get(k) for k in ("name", "beta", "tau_pct", "a_low", "calib_bpb")},
              "held", round(bits.mean(), 4), "gain", round(gain, 4), "sd", round(sd, 4), flush=True)
    bm = b
    g, h = res["arms"]["GATED"], res["arms"]["GATED-SHUF"]
    passes = lambda a: a["gain_over_S"] >= bm["min_gain_bits"] and a["gain_over_S"] >= bm["min_sd"] * a["bootstrap_sd"]  # noqa: E731
    res["pass_b"] = bool(passes(g) and not passes(h))
    res.update({"learner_records": records, "cpu_s": round(time.process_time() - t0, 1)})
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "RESULTS.json").write_text(json.dumps(res, indent=1))
    print("S held", round(res["arms"]["S"]["held_bpb"], 4), "pass (b):", res["pass_b"])
    runtime.write_closure()


if __name__ == "__main__":
    main()
