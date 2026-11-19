"""Part 1: a long life recorded in windows of 4,096 bytes ending at 16,384, 32,768, 49,152 and 65,536; for each window, with the
connections at its end: the nine entries and lags 0, 1, 2 near the output lines (enlarged lists and birth()'s 1/8 lists), with levels (a)
and (b) as in L8read.py. Also part 2's description of the cells near the output lines.

    python3 LLread.py <prefix> <seed> [ends]
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/user/tissue_deps"); sys.path.insert(0, "/home/user/tissue_check")
import tissue_step1 as ts, readings, my_tissue as mt
pre, seed = sys.argv[1], int(sys.argv[2])
ends = [int(e) for e in (sys.argv[3].split(",") if len(sys.argv) > 3 else "16384,32768,49152,65536".split(","))]
b, _ = ts.stream(seed, max(ends)); B = ts.birth(seed); long, mem_long, _ = mt.long_axons(B, seed)
rec = np.fromfile(f"out/{pre}.rec", dtype=np.float32).reshape(-1, ts.N)
isfb = np.asarray(B["kind"]) == ts.FBK; drivers = [int(i) for i in np.where(np.asarray(B["kind"]) == ts.OUTE)[0]]
def group(members):
    near = np.zeros(ts.N, bool)
    for i in drivers:
        near[i] = True
        for m in members[i]:
            if m < ts.N: near[m] = True
    return near & ~isfb
groups = dict(enlarged=group(mem_long), within_1_8=group(B["members"]))
def judge(r, d, L):
    s = r["scores"][d]["by_lag"][L]; c = r["c0"][L]; cS, cse = (c["S"], c["se"]) if isinstance(c, dict) else (c, 0.0)
    lev = 0.1103 if L == 0 else cS + 0.02; hi = 0.1103 if L == 0 else cS + 2 * cse
    return dict(S=s["S"], se=s["se"], level=lev, a=s["S"] > lev, b=s["S"] - 2 * s["se"] > hi + 0.02)
out = {}
for w, e in enumerate(ends):
    lo = e - 4096; X = rec[w * 4096:(w + 1) * 4096]; wb = b[lo:e]
    conns = [(int(r.split()[0]), int(r.split()[1])) for r in open(f"out/{pre}.conn_{e}").read().split("\n") if r]
    dist = mt.distances(B, conns, wb)
    x = np.where(X < 0.01, 0.0, X).astype(np.float64)
    r = readings.read_window(x, wb, dist, seed, stream=b, start=lo, distances=(1, 2, 4), lags=(0, 1, 2))
    row = dict(nine={f"d{d} L{L}": judge(r, d, L) for d in (1, 2, 4) for L in (0, 1, 2)}, near={}, describe={})
    src = {i: [] for i in range(ts.N)}
    for i, s_ in conns: src[i].append(s_)
    dc = [s_ for i in drivers for s_ in src[i]]
    row["describe"]["drivers_connections"] = len(dc)
    row["describe"]["drivers_connections_from_long_cells_share"] = sum(s_ in set(long) for s_ in dc) / max(1, len(dc))
    for gname, g in groups.items():
        rn = readings.read_window(x, wb, np.where(g, 1, 0).astype(np.int16), seed, stream=b, start=lo, distances=(1,), lags=(0, 1, 2))
        row["near"][gname] = dict(cells=int(g.sum()), **{f"L{L}": judge(rn, 1, L) for L in (0, 1, 2)})
        cells = np.where(g)[0]
        act = cells[(x[:, cells] > 0.01).sum(0) >= 32]
        pref = set()
        for c in act:
            means = {v: x[wb == v, c].mean() for v in set(wb.tolist())}
            pref.add(max(means, key=means.get))
        row["describe"][gname] = dict(cells=int(g.sum()), by_distance={str(int(d)): int(((dist == d) & g).sum()) for d in sorted(set(dist[g].tolist()))},
                                      not_silent=int(len(act)), byte_values_preferred=len(pref), byte_values_in_window=len(set(wb.tolist())),
                                      share_largest_output_above_half=float((X[:, cells].max(0) > 0.5).mean()))
    out[str(e)] = row
    n = row["near"]
    print(pre, e, "nine", "".join("." if v["a"] and v["b"] else ("b" if v["a"] else "A") for v in row["nine"].values()),
          "| near enlarged", " ".join(f"{n['enlarged'][f'L{L}']['S']:.3f}" for L in (0, 1, 2)),
          "| near 1/8", " ".join(f"{n['within_1_8'][f'L{L}']['S']:.3f}" for L in (0, 1, 2)), flush=True)
json.dump(out, open(f"out/{pre}_LL.json", "w"), indent=1, default=float)
