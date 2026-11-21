"""Test 1: is the byte carried L regions deep? Random independent bytes; for every region d, the byte presented `lag` steps earlier
is decoded from the region's code; the best lag and its score are reported beside a shuffled-label null (the largest of 20)."""
import sys, time
import numpy as np
from net import Tissue, decode
args = dict(a.split("=") for a in sys.argv[1:])
T = int(args.get("T", 6000)); L = int(args.get("L", 24))
kw = {k: (float(v) if "." in v else int(v)) for k, v in args.items() if k not in ("T",)}
t0 = time.time()
net = Tissue(**kw)
rng = np.random.default_rng(99)
stream = rng.integers(0, net.n_values, T)
codes = np.zeros((T, net.N), np.uint8)
for t in range(T):
    codes[t] = net.step(stream[t])
run_s = time.time() - t0
burn = 500
rows = []
for d in range(net.L):
    cols = slice(d * net.R, (d + 1) * net.R)
    best = (0.0, -1)
    for lag in range(max(0, d - 2), d + 6):
        tt = np.arange(burn + lag, T)
        s = decode(codes[tt, cols].astype(float), stream[tt - lag])
        best = max(best, (s, lag))
    s, lag = best
    tt = np.arange(burn + lag, T)
    lab = stream[tt - lag]
    null = max(decode(codes[tt, cols].astype(float), np.random.default_rng(k).permutation(lab)) for k in range(20))
    active = codes[burn:, cols].sum(1).mean()
    rows.append((d, lag, s, null, active))
print(f"cells {net.N}, regions {net.L}, steps {T}, run {run_s:.1f} s, chance {1 / net.n_values:.3f}")
print("region | best lag | score | null (max of 20) | cells firing a step")
for d, lag, s, null, a in rows:
    print(f"{d:3d} | {lag:3d} | {s:.3f} | {null:.3f} | {a:.1f}{'' if s > null else '   <= null'}")
