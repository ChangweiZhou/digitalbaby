"""V163 C1 technical (design section 4), lane L only (lane C was CANNOT_TEST at C0, so COLA is not built).
Run through launch.py:  python launch.py run --stage technical -- /tmp/venv_t/bin/python v163_technical.py
Checks: the autograd update equals the explicit three-factor formula (float64); no gradient from a probe reaches W;
determinism on 50 lines; after one pass the layer separates positives from negatives on A's calibration (>= 0.6); cost.
Writes results/technical/CHECKS.json.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v163_lib as V  # noqa: E402
import v163_ssl as SSL  # noqa: E402

import torch  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def check(out, name, ok, detail=None):
    out.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name, "" if detail is None else detail, flush=True)


def main():
    t0 = time.process_time()
    out, records = [], 0
    A = V.load_codes()["A"]
    # 1. autograd update = explicit local formula, float64, with a non-zero prediction matrix
    W, b = V.init_H()
    m = SSL.CLAPP(W, b, dtype=torch.float64)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        m.B.copy_(torch.randn(m.B.shape, generator=g, dtype=torch.float64) * 0.02)
        m.b.copy_(torch.randn(m.b.shape, generator=g, dtype=torch.float64) * 0.1)
    P = torch.from_numpy(SSL.pairs(A["train"]))
    t = P[torch.randperm(len(P), generator=g)[:64]]
    neg = torch.randint(0, len(A["train"]["y"]), (64,), generator=g)
    m.zero_grad()
    m.loss(A["train"]["idx"], t, neg).backward()
    dW, db, dB = SSL.explicit_grads(m, A["train"]["idx"], t, neg)
    err = max(float((m.W.grad - dW).abs().max()), float((m.b.grad - db).abs().max()), float((m.B.grad - dB).abs().max()))
    scale = float(m.W.grad.abs().max())
    check(out, "autograd update equals the explicit three-factor formula (float64, <= 1e-10)", err <= 1e-10,
          {"max_abs_diff": err, "max_abs_grad_W": scale})
    # 2. a probe fitted on the layer's output sends no gradient to W
    Wc = W.clone()
    Bd = V.load_codes()["B"]
    feats = {k: V.h_of(Bd[k]["idx"][:2000], W, b) for k in V.SPLITS}
    ys = {k: Bd[k]["y"][:2000] for k in V.SPLITS}
    V.SET["probe"]["max_epochs"], keep = 1, V.SET["probe"]["max_epochs"]
    V.fit_probe("dense", feats, ys, [2000], d_in=W.shape[1])
    V.SET["probe"]["max_epochs"] = keep
    check(out, "probe training leaves W untouched and the features carry no gradient",
          torch.equal(W, Wc) and not any(f.requires_grad for f in feats.values()))
    # 3. determinism on 50 lines, one pass
    m1, _, i1 = SSL.train(A, 1, lines=50)
    m2, _, i2 = SSL.train(A, 1, lines=50)
    records += i1["records"] + i2["records"]
    check(out, "one pass on 50 lines is deterministic", torch.equal(m1.W, m2.W) and torch.equal(m1.B, m2.B))
    # 4. the rule learns its own objective: one full pass
    m3, _, i3 = SSL.train(A, 1)
    records += i3["records"]
    sep0 = SSL.separation(SSL.CLAPP(*V.init_H()), A["calib"])
    sep = SSL.separation(m3, A["calib"])
    check(out, "after one pass, positive beats negative on >= 60% of A's calibration pairs",
          sep >= V.SET["rules"]["clapp_sep_min"], {"separation": sep, "at_initialisation_B0": sep0,
                                                    "loss": i3["loss_per_pass"], "cpu_s_per_pass": i3["cpu_s_per_pass"]})
    with torch.no_grad():
        H = m3.h(A["calib"]["idx"])
    diag = {"dead_units_calib": int((H.max(0).values == 0).sum()), "mean_active_units": float((H > 0).float().sum(1).mean())}
    print(diag)
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "CHECKS.json").write_text(json.dumps({"checks": out, "diagnostics_one_pass": diag, "learner_records": records,
                                              "cpu_s": round(time.process_time() - t0, 1)}, indent=1))
    print(sum(o["pass"] for o in out), "/", len(out), "pass; learner records", records)
    import runtime
    runtime.write_closure()
    sys.exit(0 if all(o["pass"] for o in out) else 1)


if __name__ == "__main__":
    main()
