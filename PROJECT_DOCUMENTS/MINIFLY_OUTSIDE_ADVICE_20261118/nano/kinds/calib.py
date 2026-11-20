#!/usr/bin/env python3
"""nano/kinds, PLAN_3.md: the constant k of the common signal, set on the development seed so that the mean multiplier over the run is 1.
    python3 calib.py t1|t2 <beta> <p>      runs with k = 1, then with k over the mean multiplier found, and once more if the mean is not within 0.02 of 1.
Each run is written to res/ (labels q3d_<testbed>_b<1/beta>_p<p>_r<round>_1); one line a setting is appended to calib_<testbed>.txt."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    tb, beta, p = sys.argv[1], sys.argv[2], sys.argv[3]; k = 1.0; log = []
    for rnd in range(3):
        lab = f"q3d_{tb}_b{beta.split('/')[1]}_p{p}_r{rnd}_1"; kw = dict(label=lab, seed="1", sig=beta, sigp=p, sigk=repr(k))
        if tb == "t1":
            import hold; o = hold.run(dict(kw, text="m1", gap="8", eta1="1", cross="1", ret="exact", lines="960000", every="20000", fast="1"))
            rows = {r["lines"]: r["shown"] for r in o["rows"]}; sc = sum(rows[960000 - i * 80000] for i in range(3)) / 3
        else:
            import cells2; o = cells2.run(dict(kw, fast="3")); sc = o["last3_add_phase0"]
        log.append(f"round {rnd}: k {k:.4f} mean multiplier {o['sig_mean']:.4f} a/A at the end {o['sig_ratio_end']:.3f} mean size {o['sig_size_mean']:.3f} score {sc:.1f}")
        done = abs(o["sig_mean"] - 1) <= 0.02
        if done and rnd > 0: break
        if done and rnd == 0: break
        k = k / o["sig_mean"]
    open(os.path.join(HERE, f"calib_{tb}.txt"), "a").write(f"beta={beta} p={p} final k={k!r} | " + " | ".join(log) + "\n")
