"""Part (a), coincidence tissue: does E63's own program amplify a change of one unit in the last place?

Runs E63's program (TISSUE_STEP1/tissue_step1.py, unchanged) twice on seed 73129 for 1,600 steps; in the second copy every strength
of a connection between cells is moved up by one unit in the last place (numpy.nextafter) after step 4. Prints the largest difference
of the outputs and thresholds every 100 steps, and whether connections given up and made differ. Writes qual/nudge_test.json.
    python3 nudge_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "TISSUE_STEP1"))
import tissue_step1 as ts  # noqa: E402

st, _ = ts.stream(73129, 60)
R = ts.Tissue(73129, "coincidence"); R2 = ts.Tissue(73129, "coincidence"); R.audit = []; R2.audit = []
rows = []
for k in range(1600):
    b = int(st[k // 32]); R.step(b); R2.step(b)
    if k == 4:
        n = R2.n; c = np.nonzero(R2.snd[:n] < ts.N)[0]; R2.w[c] = np.nextafter(R2.w[c], np.inf)
    if k % 100 == 99:
        rows.append(dict(step=k, largest_output_difference=float(np.abs(R.x - R2.x).max()),
                         largest_threshold_difference=float(np.abs(R.th - R2.th).max())))
        print(rows[-1], flush=True)
g1 = [(int(a), int(s), int(r)) for a, ss, ww, rr in R.audit for s, r in zip(ss, rr)]
g2 = [(int(a), int(s), int(r)) for a, ss, ww, rr in R2.audit for s, r in zip(ss, rr)]
out = dict(rows=rows, given_up=[len(g1), len(g2)], given_up_identical=g1 == g2, made=[R.made, R2.made])
(HERE / "qual" / "nudge_test.json").write_text(json.dumps(out, indent=1))
print(out["given_up"], out["given_up_identical"], out["made"])
