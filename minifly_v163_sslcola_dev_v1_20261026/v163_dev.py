"""V163 C2 dev, lane L (design sections 2 and 4, amendment 1).
Run through launch.py:  python launch.py run --stage dev -- /tmp/venv_t/bin/python v163_dev.py
SSL-H learned on A's taught text from RAND-H's initialisation (8 passes if the recorded caps allow, else 4, design section 6);
at each checkpoint a fresh carry-over probe (seeds 0, 1, 2) is fitted on B's taught text; the checkpoint with the best mean
B calibration bits is chosen and B's held-out lines are scored once. Compared with RAND-H and BP-H from TESTBED_R2 (same
seeds, same probe). Pass: SSL-H beats RAND-H by >= 0.03 bits and >= 3 line-bootstrap SDs.
Non-gating: phi, the same-text reading, a linear carry-over probe, the alignment between the local update and the
back-propagated next-byte gradient, dead units, effective rank. Writes results/dev/RESULTS.json.
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
from v163_testbed_r2 import mean_fit  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def budget():
    auth = json.loads((HERE / "AUTHORIZATION.json").read_text())
    used = sum(json.loads(x).get("learner_records", 0) for x in (HERE / "runlog" / "ATTEMPTS.jsonl").read_text().splitlines())
    s = V.SET["lane_L"]["ssl"]
    full = used + s["passes"] * 705 <= auth["caps"]["max_records"]
    return (s["passes"], s["checkpoints"]) if full else (s["passes_default_caps"], s["checkpoints_default_caps"])


def eff_rank(H):
    X = H - H.mean(0)
    s = torch.linalg.svdvals(X.double())
    p = s ** 2 / (s ** 2).sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def alignment(W, b, Bm, A, cls, top):
    """Cosine between the local rule's gradient on W and the back-propagated next-byte gradient on W (through a probe
    fitted on this layer), both on the same 2,048 calibration positions; only rows of W that receive input."""
    idx, y = A["calib"]["idx"], A["calib"]["y"]
    pos = torch.arange(min(2048, len(y) - 1))
    Wp, bp = torch.nn.Parameter(W.clone()), torch.nn.Parameter(b.clone())
    top.eval()
    h = torch.relu(F.embedding_bag(idx[pos], Wp, mode="sum", padding_idx=V.PAD) + bp)
    F.cross_entropy(top(h), torch.from_numpy(cls.encode(y[pos.numpy()]))).backward()
    g_bp = Wp.grad.clone()
    m = SSL.CLAPP(W, b)
    with torch.no_grad():
        m.B.copy_(Bm)
    split = A["calib"]
    t = torch.from_numpy(SSL.pairs(split))[:2048]
    neg = torch.randint(0, len(split["y"]), (len(t),), generator=torch.Generator().manual_seed(5))
    return m, g_bp, t, neg


def main():
    t0 = time.process_time()
    data = V.load_codes()
    A, B = data["A"], data["B"]
    passes, cps = budget()
    m, snaps, info = SSL.train(A, passes, checkpoints=cps)
    print("SSL trained", passes, "passes; loss", [round(x, 4) for x in info["loss_per_pass"]], flush=True)
    r2 = json.loads((HERE / "results" / "testbed" / "TESTBED_R2.json").read_text())["carry_over_3seed"]
    rows = {}
    for p in cps:
        W, b, _ = snaps[p]
        feats = {k: V.h_of(B[k]["idx"], W, b) for k in V.SPLITS}
        r = mean_fit("dense", feats, B, d_in=W.shape[1])
        r["mean_calib_bpb"] = float(np.mean([s["calib_bpb"] for s in r["per_seed"]]))
        rows[p] = r
        print("checkpoint", p, "calib", round(r["mean_calib_bpb"], 4), flush=True)
    best = min(cps, key=lambda p: rows[p]["mean_calib_bpb"])
    ssl = rows[best]
    nB = B["held"]["n"]
    gain = r2["RAND-H"]["held_bpb"] - ssl["held_bpb"]
    sd = V.boot_sd(r2["RAND-H"]["lines"], ssl["lines"], nB)
    bp_gain = r2["RAND-H"]["held_bpb"] - r2["BP-H"]["held_bpb"]
    ru = V.SET["rules"]
    passed = gain >= ru["pass_L_bits"] and gain >= ru["pass_L_sd"] * sd
    # non-gating readings
    W, b, Bm = snaps[best]
    featsB = {k: V.h_of(B[k]["idx"], W, b) for k in V.SPLITS}
    lin = V.fit_probe("dense", featsB, {k: B[k]["y"] for k in V.SPLITS}, nB, hidden=0, d_in=W.shape[1])
    featsA = {k: V.h_of(A[k]["idx"], W, b) for k in V.SPLITS}
    clsA = V.R.Classes(A["train"]["y"])
    torch.manual_seed(V.SET["probe"]["seed"])
    top = V.Probe("dense", W.shape[1], V.SET["probe"]["hidden"], clsA.K, V.SET["probe"]["dropout"])
    tinfo = V.train_loop(top, featsA["train"], torch.from_numpy(clsA.encode(A["train"]["y"])),
                         lambda q: float(clsA.bits(V.probs(q, featsA["calib"]), A["calib"]["y"]).mean()))
    same_bits = clsA.bits(V.probs(top, featsA["held"]), A["held"]["y"])
    mm, g_bp, t, neg = alignment(W, b, Bm, A, clsA, top)
    mm.zero_grad()
    mm.loss(A["calib"]["idx"], t, neg).backward()
    rows_in = g_bp.abs().sum(1) > 0
    a, c = -g_bp[rows_in].flatten(), -mm.W.grad[rows_in].flatten()
    cos = float(a @ c / (a.norm() * c.norm()))
    HB = featsB["held"]
    res = {"stage": STAGE, "passes": passes, "checkpoints": cps, "ssl_training": info,
           "checkpoint_rows": {str(p): rows[p] for p in cps}, "chosen_checkpoint": best,
           "carry_over_held_bpb": {"SSL-H": ssl["held_bpb"], "RAND-H": r2["RAND-H"]["held_bpb"],
                                   "BP-H": r2["BP-H"]["held_bpb"], "TOP": r2["TOP"]["held_bpb"],
                                   "V162_R1_on_KC": 3.3202},
           "gain_over_RAND-H": gain, "bootstrap_sd": sd, "phi": gain / bp_gain if bp_gain else None, "pass_L": bool(passed),
           "non_gating": {"carry_over_linear_probe_seed0": lin["held_bpb"],
                          "same_text_probe_seed0": {"held_bpb": float(same_bits.mean()), **tinfo},
                          "alignment_cosine_local_vs_bp": cos,
                          "separation_A_calib": SSL.separation(mm, A["calib"]),
                          "separation_B_calib": SSL.separation(mm, B["calib"]),
                          "dead_units_B_held": int((HB.max(0).values == 0).sum()),
                          "mean_active_units_B_held": float((HB > 0).float().sum(1).mean()),
                          "effective_rank_B_held": {"SSL-H": eff_rank(HB),
                                                    "RAND-H": eff_rank(V.h_of(B["held"]["idx"], *V.init_H()))}},
           "learner_records": info["records"], "cpu_s": round(time.process_time() - t0, 1)}
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "RESULTS.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("chosen_checkpoint", "carry_over_held_bpb", "gain_over_RAND-H", "bootstrap_sd",
                                          "phi", "pass_L")}, indent=1))
    print(json.dumps(res["non_gating"], indent=1, default=str)[:1500])
    import runtime
    runtime.write_closure()


if __name__ == "__main__":
    main()
