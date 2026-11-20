#!/usr/bin/env python3
"""nano/ref2, step 2: the per-byte losses of the coordinator's systems of cells on the last tenth, made by running nano/table's own program
(gate_all.SysGate) unchanged, as the coordinator's task describes: read the stream, then score the last tenth with nothing changing.
    python3 cells.py T2      # without companions, relay, since, pool   (reported: first cell 1.4973, second cell 1.4820)
    python3 cells.py big     # without relay                            (reported: 1.4740 and 1.4444); 4 to 7 GB: only alone
Writes res/cells_<name>_cell1.f32, res/cells_<name>_cell2.f32 (float32, 111,540 numbers: minus the log of the probability of the true byte)
and res/cells_<name>.json. Nothing in nano/table is changed (its compiled kernels are built there if missing; they are not tracked)."""
import json, os, resource, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); TABLE = os.path.join(HERE, "..", "table"); sys.path.insert(0, TABLE); os.chdir(TABLE)
import score, gate_all
name = sys.argv[1]; without = dict(T2=("companions", "relay", "since", "pool"), big=("relay",))[name]
t0 = time.process_time(); x, cut, _ = score.load(); s = gate_all.SysGate(x, without, 0); s.go(6, cut, True); P1, P2 = s.go(cut, len(x), False, True); y = x[cut:]
out = dict(name=name, without=list(without), gate=0, n=int(len(y)), kinds=int(s.K1))
for tag, P in (("cell1", P1), ("cell2", P2)):
    with np.errstate(divide="ignore"): nll = -np.log(P[np.arange(len(y)), y])
    nll.astype(np.float32).tofile(os.path.join(HERE, "res", f"cells_{name}_{tag}.f32")); sc = score.score(P, y); e = score.score(P[::8], y[::8])
    out[tag] = dict(all=sc, every_eighth=e, mean_of_kept_file=float(nll.astype(np.float32).astype(np.float64).mean()))
out["cpu"] = round(time.process_time() - t0, 1); out["peak_memory_GB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
json.dump(out, open(os.path.join(HERE, "res", f"cells_{name}.json"), "w"), indent=1); print(json.dumps(out))
