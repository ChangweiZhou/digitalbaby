"""S(d, 0) at d = 1, 2, 4, 8, 10 of the reference N0 run (seed 73129) and of runs with one cell's potential changed by 1e-12 before
step 0; the window is the last 4,096 bytes; distances from each run's own final connections.

    python3 spread.py <tag> [<tag> ...]
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/user/tissue_deps"); sys.path.insert(0, "/home/user/tissue_check")
import tissue_step1 as ts, readings, my_tissue as mt
seed, nb = 73129, 16384; lo = nb - 4096
b, _ = ts.stream(seed, nb); B = ts.birth(seed)
res = {}
for tag in sys.argv[1:]:
    rec = np.fromfile(f"out/{tag}.rec", dtype=np.float32).reshape(-1, ts.N)[-4096:]
    conns = [(int(r.split()[0]), int(r.split()[1])) for r in open(f"out/{tag}.conn").read().split("\n") if r]
    dist = mt.distances(B, conns, b[lo:nb])
    x = np.where(rec < 0.01, 0.0, rec).astype(np.float64)
    r = readings.read_window(x, b[lo:nb], dist, seed, stream=b, start=lo, distances=(1, 2, 4, 8, 10), lags=(0,))
    st = json.load(open(f"out/{tag}.stats"))
    res[tag] = dict(S={d: r["scores"][d]["by_lag"][0]["S"] for d in (1, 2, 4, 8, 10)}, connections=st["connections"], made=st["made"], withdrawn=st["withdrawn"])
    print(tag, st["connections"], {d: round(v, 3) for d, v in res[tag]["S"].items()}, flush=True)
json.dump(res, open("out/spread_" + "_".join(sys.argv[1:]) + ".json", "w"), indent=1)
