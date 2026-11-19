"""Part 4: why the output end does not learn on L8-256, seed 93683, against seed 73129.

For each seed, on the outputs at the reading step (65,536 bytes; fitting part = first three quarters, scored = last quarter):
  own   each byte v's unit fitted on its own driver's inputs (set (a) cells in the driver's enlarged neighbourhood; standardised over the
        fitting part; ridge, constant chosen from 10^-6..10^2 x n on the last fifth of the fitting part), as E55's description says;
  oh1   the same unit on a one-hot of the present byte (what the present byte alone gives);
  oh2   on one-hots of the present and the previous byte.
Then, for each next byte value v, the recall (share of the scored positions with next byte v named right) under each reader, and for the
driver of v: how well its own inputs tell the present byte, as the balanced accuracy of a nearest-centroid reader of the present byte
(the readings tool's classifier on the driver's inputs alone) and the share of the predecessors of v (the present bytes after which v
comes, weighted by count) that are the preferred byte (largest mean output) of at least one of its input cells.

    python3 why93683.py <rec> <seed> <tag>
"""
import json, sys
import numpy as np
sys.path.insert(0, "/home/user/tissue_deps"); sys.path.insert(0, "/home/user/tissue_check")
import tissue_step1 as ts, my_tissue as mt, readings

recf, seed, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
NB = 65536
B = ts.birth(seed); _, members, _ = mt.long_axons(B, seed)
b, _ = ts.stream(seed, NB); b = b.astype(np.int64)
isfb = np.asarray(B["kind"]) == ts.FBK; drivers = list(B["driver"])
near = set(drivers)
for i in drivers: near.update(int(m) for m in members[i] if m < ts.N)
near = np.array(sorted(i for i in near if not isfb[i])); col = {c: k for k, c in enumerate(near)}
own = [sorted({col[c] for c in [i] + [int(m) for m in members[i] if m < ts.N] if c in col}) for i in drivers]
X = np.fromfile(recf, dtype=np.float32).reshape(-1, ts.N)[:NB][:, near].astype(np.float64)
M = NB - 1; y = b[1:]; nf = (3 * M) // 4; nin = (4 * nf) // 5
X = X[:M]; sd = X[:nf].std(0); X = np.where(sd > 0, X / np.where(sd > 0, sd, 1), 0.0)
Y = np.zeros((M, 256)); Y[np.arange(M), y] = 1

def unit_scores(F):
    """F(v) -> feature matrix for unit v; returns the scored-quarter scores of all 256 units with the constant chosen on the inner split."""
    def pred(n, lam):
        out = np.zeros((M, 256))
        for v in range(256):
            Xv = F(v); mu = Xv[:n].mean(0); my = Y[:n, v].mean(); Xc = Xv[:n] - mu
            G = Xc.T @ Xc; G[np.diag_indices_from(G)] += lam * n
            out[:, v] = (Xv - mu) @ np.linalg.solve(G, Xc.T @ (Y[:n, v] - my)) + my
        return out
    inner = {e: float((pred(nin, 10.0 ** e)[nin:nf].argmax(1) == y[nin:nf]).mean()) for e in range(-6, 3)}
    be = max(inner, key=lambda e: (inner[e], e)); return pred(nf, 10.0 ** be)[nf:], 10.0 ** be

OH1 = np.zeros((M, 256)); OH1[np.arange(M), b[:M]] = 1
OH2 = np.zeros((M, 512)); OH2[np.arange(M), b[:M]] = 1; OH2[np.arange(1, M), 256 + b[:M - 1][:M - 1]] = 1
res = {}
scores = {}
for name, F in (("own", lambda v: X[:, own[v]]), ("oh1", lambda v: OH1), ("oh2", lambda v: OH2)):
    s, lam = unit_scores(F); scores[name] = s.argmax(1)
    res[name] = dict(share=float((scores[name] == y[nf:]).mean()), ridge=lam)
yt = y[nf:]; vals = sorted(set(yt.tolist()))
per_v = {}
pt = b[nf:M]                                                  # present byte at the scored positions
for v in vals:
    m = yt == v
    pred_v = b[:M][y == v]; cnt = {c: int((pred_v == c).sum()) for c in set(pred_v.tolist())}
    # the driver of v: its input cells' preferred bytes (largest mean output over the fitting part)
    Xo = X[:nf][:, own[v]]; pres = b[:nf]
    pref = set()
    for k in range(Xo.shape[1]):
        if Xo[:, k].std() > 0:
            means = {c: Xo[pres == c, k].mean() for c in set(pres.tolist())}; pref.add(max(means, key=means.get))
    cover = sum(n for c, n in cnt.items() if c in pref) / sum(cnt.values())
    per_v[str(v)] = dict(count=int(m.sum()), recall={k: float((scores[k][m] == v).mean()) for k in scores}, inputs=len(own[v]),
                         predecessors=len(cnt), predecessors_covered_share=cover)
# how well each driver's inputs tell the present byte (the readings tool's classifier on the last quarter window)
lo = NB - 4096
Xw = np.fromfile(recf, dtype=np.float32).reshape(-1, ts.N)[lo:NB][:, near].astype(np.float64)
Xw = np.where(Xw < 0.01, 0.0, Xw)
pres_acc = {}
for v in vals:
    d = np.zeros(len(near), np.int16); d[own[v]] = 1
    r = readings.read_window(Xw, b[lo:NB], d, seed, stream=b, start=lo, distances=(1,), lags=(0, 1))
    pres_acc[str(v)] = {f"L{L}": r["scores"][1]["by_lag"][L]["S"] for L in (0, 1)}
for v in vals: per_v[str(v)]["driver_reads_present_byte"] = pres_acc[str(v)]
out = dict(seed=seed, readers=res, per_value=per_v,
           mean_driver_reads_present_byte=float(np.mean([pres_acc[str(v)]["L0"] for v in vals])),
           mean_driver_reads_previous_byte=float(np.mean([pres_acc[str(v)]["L1"] for v in vals])),
           mean_predecessors_covered=float(np.average([per_v[str(v)]["predecessors_covered_share"] for v in vals], weights=[per_v[str(v)]["count"] for v in vals])))
json.dump(out, open(f"/home/user/tissue_check/out/why_{tag}.json", "w"), indent=1)
print(tag, {k: round(100 * v["share"], 2) for k, v in res.items()}, "driver reads present byte (mean S)", round(out["mean_driver_reads_present_byte"], 3),
      "previous", round(out["mean_driver_reads_previous_byte"], 3), "predecessors covered", round(out["mean_predecessors_covered"], 3))
