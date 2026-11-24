"""Log-scale linear regression of log-perplexity on network size and data (24 Nov 2026; owner's request).
Data: datascale_results_{a,b,c,d}.jsonl (t26 learned-temperature readout, nats per byte on 10,000 never-read bytes, seed 1).
N = context cells = regions (16) x columns (cC) x cells per column (M); D = bytes read. Fits, per stream, by least squares:
  (1) L = a + b*log2(N) + c*log2(D)                         - b, c: change per doubling of size / of data
  (2) L = a + b*log2(N) + c*log2(D) + d*log2(N)*log2(D)      - d: whether size helps more when there is more data
  (3) log2(L) = a + b*log2(N) + c*log2(D)                    - power-law form: L ~ N^b * D^c (exponents)
Nets in the fit: M=16, M=64, M=128 (cC=64) and cC=128 M=64 (pure size). cC=128 M=32 rc_all=1 differs in kind (rule cells
gate all readouts), so it is shown against the fit (residual), not fitted."""
import json, numpy as np
SIZE = {"M16": 16 * 64 * 16, "M64": 16 * 64 * 64, "M128": 16 * 64 * 128, "E6": 16 * 128 * 64, "E4": 16 * 128 * 32}
rows = []
for p in "abcd":
    for l in open(f"datascale_results_{p}.jsonl"):
        j = json.loads(l); net, st, _ = j["name"].rsplit("_", 2); rows.append((st, net, SIZE[net], j["T"], j["nats"]["temp"]))
def fit(X, y):
    w, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ w
    return w, 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean())), np.sqrt((r @ r) / len(y))
for st in ("moby", "code"):
    R = [r for r in rows if r[0] == st and r[1] != "E4"]
    n = np.log2([r[2] for r in R]); d = np.log2([r[3] for r in R]); y = np.array([r[4] for r in R]); one = np.ones_like(n)
    print(f"\n== {st}: {len(R)} points (4 nets x 3 data sizes) ==")
    w, r2, rm = fit(np.c_[one, n, d], y)
    print(f"(1) L = {w[0]:.3f} {w[1]:+.4f}*log2(N) {w[2]:+.4f}*log2(D)    R2 {r2:.3f}, rms residual {rm:.4f}")
    print(f"    per doubling: size {w[1]:+.4f} nats/byte, data {w[2]:+.4f} nats/byte; data/size ratio {w[2] / w[1]:.1f}")
    nc, dc = n - n.mean(), d - d.mean()
    w2, r22, rm2 = fit(np.c_[one, nc, dc, nc * dc], y)
    print(f"(2) centred: L = {w2[0]:.3f} {w2[1]:+.4f}*n {w2[2]:+.4f}*d {w2[3]:+.4f}*n*d    R2 {r22:.3f}, rms {rm2:.4f}")
    for D in (16000, 64000, 256000):
        print(f"    size effect per doubling at D={D // 1000}k: {w2[1] + w2[3] * (np.log2(D) - d.mean()):+.4f}")
    w3, r23, rm3 = fit(np.c_[one, n, d], np.log2(y))
    print(f"(3) power law: L ~ N^{w3[1]:+.4f} * D^{w3[2]:+.4f}    R2 {r23:.3f}")
    E = [r for r in rows if r[0] == st and r[1] == "E4"]
    res = [r[4] - (w[0] + w[1] * np.log2(r[2]) + w[2] * np.log2(r[3])) for r in E]
    print("    cC=128 M=32 rc_all=1 vs fit (1), residual by D: " + ", ".join(f"{r[3] // 1000}k {x:+.3f}" for r, x in zip(E, res)))
