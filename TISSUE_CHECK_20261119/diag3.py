"""Part 3: where the present byte comes from (seed 112290): distance of the cells near the output lines; the share of the drivers' and of
the near cells' connections from long-axon cells; S(lag 0) and S(lag 1) read by the tool from the long-axon cells alone and from the other
cells (all cells of the tissue, and within the near group).

    python3 diag3.py <rec file> <rec dtype f4|f8> <rows offset of the window> <conn file> <seed> <window end>
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/user/tissue_deps"); sys.path.insert(0, "/home/user/tissue_check")
import tissue_step1 as ts, readings, my_tissue as mt
recf, dt, off, connf, seed, e = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
b, _ = ts.stream(seed, e); B = ts.birth(seed); long, mem_long, _ = mt.long_axons(B, seed); L = set(long)
X = np.fromfile(recf, dtype=np.float64 if dt == "f8" else np.float32).reshape(-1, ts.N)[off:off + 4096]
lo = e - 4096; wb = b[lo:e]; x = np.where(X < 0.01, 0.0, X).astype(np.float64)
conns = [(int(r.split()[0]), int(r.split()[1])) for r in open(connf).read().split("\n") if r]
dist = mt.distances(B, conns, wb)
isfb = np.asarray(B["kind"]) == ts.FBK; drivers = [int(i) for i in np.where(np.asarray(B["kind"]) == ts.OUTE)[0]]
near = np.zeros(ts.N, bool)
for i in drivers:
    near[i] = True
    for m in mem_long[i]:
        if m < ts.N: near[m] = True
near &= ~isfb
src = {i: [] for i in range(ts.N)}
for i, s in conns: src[i].append(s)
share = lambda cells: (lambda c: sum(s in L for s in c) / max(1, len(c)))([s for i in cells for s in src[i]])
isL = np.zeros(ts.N, bool); isL[long] = True
def S(mask, lags=(0, 1)):
    r = readings.read_window(x, wb, np.where(mask, 1, 0).astype(np.int16), seed, stream=b, start=lo, distances=(1,), lags=lags)
    return {f"L{l}": round(r["scores"][1]["by_lag"][l]["S"], 3) for l in lags} | dict(cells=int(mask.sum()))
out = dict(near_cells=int(near.sum()), near_by_distance={str(int(d)): int(((dist == d) & near).sum()) for d in sorted(set(dist[near].tolist()))},
           near_within_4=int(((dist > 0) & (dist <= 4) & near).sum()),
           drivers_connections_from_long_share=share(drivers), near_connections_from_long_share=share(np.where(near)[0]),
           long_cells=len(long), long_by_distance={str(int(d)): int(((dist == d) & isL).sum()) for d in sorted(set(dist[isL].tolist()))},
           read_long_all=S(isL & ~isfb), read_other_all=S(~isL & ~isfb), read_long_near=S(isL & near), read_other_near=S(~isL & near))
print(json.dumps(out))
