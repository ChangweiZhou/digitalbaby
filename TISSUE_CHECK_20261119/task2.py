"""Task 2/3: two copies of one tissue (seed 73129), one cell's potential (cell 100) changed by 1e-9 before the first step of byte B.
For each B: the largest difference of any cell's output at the reading step of byte B+63 and of byte B+511 (64 and 512 bytes after
the change), and over all reading steps up to them; connections made and withdrawn in B..B+64 and B..B+512 (the reference copy);
the final connection lists (equal or not); and S(d, 0) at d = 1, 2, 4, 8, 10 over the last 4,096 bytes for both copies.

    python3 task2.py <T2|T3> <N0|N256>
"""
import glob, json, sys
import numpy as np
sys.path.insert(0, "/home/user/tissue_deps"); sys.path.insert(0, "/home/user/tissue_check")
import tissue_step1 as ts, readings, my_tissue as mt
pre, kind = sys.argv[1], sys.argv[2]
seed, nb = 73129, 16384; lo = nb - 4096
b, _ = ts.stream(seed, nb); B0 = ts.birth(seed)
ref = f"out/{pre}_{kind}_ref"
ref_lo = 0 if pre == "T2" else 4096
R = np.fromfile(ref + ".rec", dtype=np.float64).reshape(-1, ts.N)
cnt = np.fromfile(ref + ".counts", dtype=np.int64).reshape(-1, 2)          # made, withdrawn after each byte
conn = lambda p: sorted(tuple(r.split()[:2]) for r in open(p + ".conn").read().split("\n") if r)
def S0(p, rec):
    conns = [(int(a), int(c)) for a, c in conn(p)]
    dist = mt.distances(B0, conns, b[lo:nb]); x = np.where(rec[-4096:] < 0.01, 0.0, rec[-4096:]).astype(np.float64)
    r = readings.read_window(x, b[lo:nb], dist, seed, stream=b, start=lo, distances=(1, 2, 4, 8, 10), lags=(0,))
    return {d: (round(r["scores"][d]["by_lag"][0]["S"], 3) if r["scores"][d]["by_lag"][0] else None) for d in (1, 2, 4, 8, 10)}
out = dict(reference=dict(S0=S0(ref, R), made=int(cnt[-1, 0]), withdrawn=int(cnt[-1, 1])), rows={})
rc = conn(ref)
for p in sorted(glob.glob(f"out/{pre}_{kind}_B*.stats"), key=lambda s: int(s.split("_B")[1][:-6])):
    B = int(p.split("_B")[1][:-6]); q = p[:-6]
    X = np.fromfile(q + ".rec", dtype=np.float64).reshape(-1, ts.N)          # bytes B .. nb
    row = {}
    for n in (64, 512):
        e = min(B + n, nb)
        d_at = float(np.abs(X[e - 1 - B] - R[e - 1 - ref_lo]).max())
        d_upto = float(np.abs(X[:e - B] - R[B - ref_lo:e - ref_lo]).max())
        m0, w0 = (cnt[B - 1] if B > 0 else (0, 0)); m1, w1 = cnt[e - 1]
        row[str(n)] = dict(last_byte=e - 1, diff_at=d_at, diff_max_upto=d_upto, made=int(m1 - m0), withdrawn=int(w1 - w0))
    dend = float(np.abs(X[-1] - R[-1]).max())
    st = json.load(open(q + ".stats"))
    Xw = X if len(X) >= 4096 else np.concatenate([R[lo - ref_lo:B - ref_lo], X])   # before byte B the two copies are the same
    row.update(diff_at_end=dend, connections_equal_at_end=conn(q) == rc, connections=st["connections"], S0=S0(q, Xw))
    out["rows"][str(B)] = row
    print(kind, "B", B, {n: (f"{row[n]['diff_at']:.2e}", row[n]["made"], row[n]["withdrawn"]) for n in ("64", "512")}, "end", f"{dend:.2e}",
          "same conns" if row["connections_equal_at_end"] else "different conns", row["S0"], flush=True)
print(kind, "reference", out["reference"])
json.dump(out, open(f"out/{pre}_{kind}.json", "w"), indent=1)
