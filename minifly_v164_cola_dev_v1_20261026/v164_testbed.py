"""V164 C0 testbed (design section 3): references only, before COLA code. No learner records.
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_t/bin/python v164_testbed.py
Per task (S-COPY, S-SUM, TEXT; one child process each): gain, RAND-R, TBPTT-1, BPTT, test bits from the trained readout.
A synthetic task qualifies if BPTT beats TBPTT-1 by >= 0.3 bits and >= 3 line-bootstrap SDs.
Text parity: the first epochs reproduce V163's TBPTT-1 and BPTT calibration curves and V163's gain.
Writes results/testbed/TESTBED.json.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v164_lib as L  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
CACHE = HERE / "cache"


def run_task(name):
    t, tr, ca, te, sc, iv, me, pa = next(x for x in L.settings() if x[0] == name)
    g, ginfo = L.gain(tr, iv)
    out = {"gain": ginfo, "arms": {}}
    for rule in ("rand", "tbptt1", "bptt"):
        net, info = L.train(tr, ca, sc, g, rule, iv, me, pa)
        sums, counts = L.evaluate(net, te, sc)
        out["arms"][rule] = {**info, "test_bpb": float(sums.sum() / counts.sum()), "line_sums": sums.tolist(),
                             "counts": counts.tolist()}
        print(name, rule, round(out["arms"][rule]["test_bpb"], 4), info["best_epoch"], info["cpu_s"], flush=True)
    return out


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--task":
        (CACHE / f"c0_{sys.argv[2]}.json").write_text(json.dumps(run_task(sys.argv[2])))
        return
    t0 = time.time()
    CACHE.mkdir(exist_ok=True)
    names = ["S-COPY", "S-SUM", "TEXT"]
    procs = {n: subprocess.Popen([sys.executable, __file__, "--task", n], stdout=open(CACHE / f"c0_{n}.log", "w"),
                                 stderr=subprocess.STDOUT) for n in names}
    if any(p.wait() for p in procs.values()):
        raise SystemExit("a task failed")
    ru = L.SET["rules"]
    res, qual = {}, {}
    import numpy as np
    for n in names:
        r = json.loads((CACHE / f"c0_{n}.json").read_text())
        a = r["arms"]
        c = np.array(a["bptt"]["counts"])
        d = a["tbptt1"]["test_bpb"] - a["bptt"]["test_bpb"]
        sd = L.boot_sd(np.array(a["tbptt1"]["line_sums"]), np.array(a["bptt"]["line_sums"]), c)
        r["tbptt1_minus_bptt"] = {"diff": d, "boot_sd": sd}
        if n != "TEXT":
            qual[n] = bool(d >= ru["qualify_bits"] and d >= ru["qualify_sd"] * sd)
        res[n] = r
    v163 = json.loads((L.V163 / "results" / "testbed" / "TESTBED.json").read_text())["results"]
    k = len(res["TEXT"]["arms"]["tbptt1"]["calib_curve"])
    parity = {"tbptt1_curve": res["TEXT"]["arms"]["tbptt1"]["calib_curve"] == v163["TBPTT1|training"]["calib_curve"][:k],
              "bptt_curve": res["TEXT"]["arms"]["bptt"]["calib_curve"]
              == v163["BPTT|training"]["calib_curve"][:len(res["TEXT"]["arms"]["bptt"]["calib_curve"])],
              "gain": abs(res["TEXT"]["gain"]["g"] - v163["g"]) < 1e-9}
    out = {"stage": STAGE, "settings": L.SET, "qualifies": qual, "text_parity_v163": parity,
           "verdict": ("INVALID" if not all(parity.values()) else
                       "CONTINUE" if any(qual.values()) else "CANNOT_TEST_CLAIM_AS_SPECIFIED"),
           "summary": {n: {r: round(res[n]["arms"][r]["test_bpb"], 4) for r in res[n]["arms"]} for n in names},
           "results": res, "learner_records": 0, "wall_s": round(time.time() - t0, 1)}
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "TESTBED.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("qualifies", "text_parity_v163", "verdict", "summary")}, indent=1))
    print({n: res[n]["tbptt1_minus_bptt"] for n in names}, {n: res[n]["gain"]["g"] for n in names})
    import runtime
    runtime.write_closure()


if __name__ == "__main__":
    main()
