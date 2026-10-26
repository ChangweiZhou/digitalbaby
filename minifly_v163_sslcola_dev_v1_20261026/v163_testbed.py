"""V163 C0 testbed, step 2 (torch environment), evaluator-side, before any learner code (design section 4). No learner records.
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_t/bin/python v163_testbed.py
Three parts run as child processes in parallel (one thread each) and are merged here:
  P  parity (the probe on the 88-channel trace reproduces T3) and TOP (the probe on the KC code), both readings, seed-1 repeat
  L  RAND-H and BP-H, both readings, seed-1 repeat of RAND-H on carry-over; linear probes
  C  lane C gain, RAND-R, TBPTT-1 and BPTT, both readings
Writes results/testbed/TESTBED.json; arrays for later cycles go to cache/.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v163_lib as V  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
HC = V.ROOT / "minifly_v155_headroom_check_20261026" / "results" / "main"
CACHE = HERE / "cache"


def ys(T):
    return {k: T[k]["y"] for k in V.SPLITS}


def both(name, feats, data, d_in, out, linear=True, seed=None):
    """Probe on frozen features for both readings: A (same text) and B (carry-over). feats[t][split] tensors."""
    for t, reading in (("A", "same_text"), ("B", "carry_over")):
        r = V.fit_probe("dense", feats[t], ys(data[t]), data[t]["held"]["n"], seed=seed, d_in=d_in)
        out[f"{name}|{reading}"] = r
        print(name, reading, "held", round(r["held_bpb"], 4), "epoch", r["best_epoch"], "cpu", r["cpu_s"], flush=True)
        if linear:
            r = V.fit_probe("dense", feats[t], ys(data[t]), data[t]["held"]["n"], hidden=0, d_in=d_in)
            out[f"{name}|{reading}|linear"] = r


def part_P(data):
    out = {}
    z = np.load(HC / "states.npz")
    ref = json.loads((HC / "REFERENCE.json").read_text())["T3"]
    X = {k: torch.from_numpy(np.asarray(z[f"rho_{k}"], np.float32)) for k in V.SPLITS}
    r = V.fit_probe("dense", X, {k: z[f"y_{k}"] for k in V.SPLITS}, z["held_line_len"],
                    hidden=ref["chosen"]["H"], d_in=88)
    out["T3_parity"] = {**r, "recorded_held_bpb": ref["held_bpb"]}
    print("T3 parity", r["held_bpb"], "recorded", ref["held_bpb"], flush=True)
    for t, reading in (("A", "same_text"), ("B", "carry_over")):
        X = {k: data[t][k]["idx"] for k in V.SPLITS}
        for hidden in (None, 0):
            r = V.fit_probe("kc", X, ys(data[t]), data[t]["held"]["n"], hidden=hidden)
            out[f"TOP|{reading}" + ("|linear" if hidden == 0 else "")] = r
            print("TOP", reading, hidden, "held", round(r["held_bpb"], 4), "epoch", r["best_epoch"], "cpu", r["cpu_s"],
                  flush=True)
    X = {k: data["B"][k]["idx"] for k in V.SPLITS}
    out["TOP|carry_over|seed1"] = V.fit_probe("kc", X, ys(data["B"]), data["B"]["held"]["n"], seed=1)
    return out


def part_L(data):
    out = {}
    W, b = V.init_H()
    feats = {t: {k: V.h_of(data[t][k]["idx"], W, b) for k in V.SPLITS} for t in ("A", "B")}
    torch.save({"W": W, "b": b}, CACHE / "rand_h.pt")
    both("RAND-H", feats, data, W.shape[1], out)
    out["RAND-H|carry_over|seed1"] = V.fit_probe("dense", feats["B"], ys(data["B"]), data["B"]["held"]["n"], seed=1,
                                                 d_in=W.shape[1])
    Wb, bb, info = V.train_bph(data["A"])
    out["BP-H|training"] = info
    torch.save({"W": Wb, "b": bb}, CACHE / "bp_h.pt")
    print("BP-H trained", info["best_epoch"], info["calib_bpb"], flush=True)
    feats = {t: {k: V.h_of(data[t][k]["idx"], Wb, bb) for k in V.SPLITS} for t in ("A", "B")}
    both("BP-H", feats, data, Wb.shape[1], out)
    return out


def part_C(data):
    out = {}
    g, ginfo = V.choose_gain(data["A"])
    out["gain"] = ginfo
    print("gain", ginfo, flush=True)
    for rule in ("rand", "tbptt1", "bptt"):
        if rule == "rand":
            cls = V.R.Classes(data["A"]["train"]["y"])
            net = V.RNN(cls.K, g)
            out["RAND-R|training"] = {"rule": "none (frozen at initialisation)"}
        else:
            net, info = V.train_rnn(data["A"], g, rule)
            out[f"{rule.upper()}|training"] = info
            print(rule, "trained", info["best_epoch"], info["calib_bpb"], info["cpu_s"], flush=True)
        name = {"rand": "RAND-R", "tbptt1": "TBPTT1", "bptt": "BPTT"}[rule]
        torch.save(net.state_dict(), CACHE / f"rnn_{name}.pt")
        feats = {t: {k: V.rnn_states(net, data[t][k]) for k in V.SPLITS} for t in ("A", "B")}
        both(name, feats, data, net.Wrr.shape[0], out, linear=False)
    out["g"] = g
    return out


def diff(res, a, b, reading, n):
    x, y = res[f"{a}|{reading}"], res[f"{b}|{reading}"]
    return {"a_minus_b": x["held_bpb"] - y["held_bpb"], "boot_sd": V.boot_sd(x["lines"], y["lines"], n)}


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--part":
        part = sys.argv[2]
        data = V.load_codes()
        res = {"P": part_P, "L": part_L, "C": part_C}[part](data)
        (CACHE / f"c0_part_{part}.json").write_text(json.dumps(res, indent=1))
        return
    t0 = time.time()
    procs = {p: subprocess.Popen([sys.executable, __file__, "--part", p], stdout=open(CACHE / f"c0_{p}.log", "w"),
                                 stderr=subprocess.STDOUT) for p in ("P", "L", "C")}
    codes = {p: q.wait() for p, q in procs.items()}
    if any(codes.values()):
        raise SystemExit(f"a part failed: {codes}")
    res = {}
    for p in ("P", "L", "C"):
        res.update(json.loads((CACHE / f"c0_part_{p}.json").read_text()))
    data = V.load_codes()
    nA, nB = data["A"]["held"]["n"], data["B"]["held"]["n"]
    ru = V.SET["rules"]
    d = {"BP-H_vs_RAND-H|carry_over": diff(res, "RAND-H", "BP-H", "carry_over", nB),
         "BP-H_vs_RAND-H|same_text": diff(res, "RAND-H", "BP-H", "same_text", nA),
         "TOP_vs_RAND-H|carry_over": diff(res, "RAND-H", "TOP", "carry_over", nB),
         "RAND-R_minus_BPTT|same_text": diff(res, "RAND-R", "BPTT", "same_text", nA),
         "TBPTT1_minus_BPTT|same_text": diff(res, "TBPTT1", "BPTT", "same_text", nA),
         "TBPTT1_minus_BPTT|carry_over": diff(res, "TBPTT1", "BPTT", "carry_over", nB)}
    seed_moves = {a: abs(res[f"{a}|carry_over|seed1"]["held_bpb"] - res[f"{a}|carry_over"]["held_bpb"])
                  for a in ("TOP", "RAND-H")}
    chosen = [k for k in res if isinstance(res[k], dict) and "cut_off_by_budget" in res[k]]
    checks = {
        "parity_T3": abs(res["T3_parity"]["held_bpb"] - res["T3_parity"]["recorded_held_bpb"]) <= ru["parity_tol"],
        "L_ceiling": d["BP-H_vs_RAND-H|carry_over"]["a_minus_b"] >= ru["ceiling_L"],
        "L_noise": all(v <= ru["noise_L"] for v in seed_moves.values()),
        "C_reference_learns": d["RAND-R_minus_BPTT|same_text"]["a_minus_b"] >= ru["learns_C"],
        "C_ceiling": d["TBPTT1_minus_BPTT|same_text"]["a_minus_b"] >= ru["ceiling_C"],
    }
    export = json.loads((HERE / "results" / STAGE / "EXPORT.json").read_text()) if (
        HERE / "results" / STAGE / "EXPORT.json").exists() else {"all_pass": None}
    checks["codes_export"] = bool(export["all_pass"])
    if not (checks["parity_T3"] and checks["codes_export"]):
        vl = vc = "INVALID"
    else:
        vl = "INVALID_NOISE" if not checks["L_noise"] else ("CONTINUE" if checks["L_ceiling"] else "CANNOT_TEST_CLAIM_AS_SPECIFIED")
        vc = ("REFERENCE_NOT_READY" if not checks["C_reference_learns"]
              else "CONTINUE" if checks["C_ceiling"] else "CANNOT_TEST_CLAIM_AS_SPECIFIED")
    summary = {k: round(v["held_bpb"], 4) for k, v in res.items() if isinstance(v, dict) and "held_bpb" in v}
    out = {"stage": STAGE, "settings": V.SET, "checks": checks, "verdict_lane_L": vl, "verdict_lane_C": vc,
           "differences": d, "seed_moves_carry_over": seed_moves, "held_bpb": summary,
           "cut_off_by_budget": [k for k in chosen if res[k]["cut_off_by_budget"]],
           "results": res, "learner_records": 0, "wall_s": round(time.time() - t0, 1), "torch": torch.__version__}
    dd = HERE / "results" / STAGE
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "TESTBED.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"checks": checks, "L": vl, "C": vc, "held_bpb": summary, "differences": d,
                      "seed_moves": seed_moves}, indent=1))
    import runtime
    runtime.write_closure()


if __name__ == "__main__":
    main()
