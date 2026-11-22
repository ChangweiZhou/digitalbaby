"""On text: is the previous byte decodable from a region's cells (order-2 code) beyond what its columns (the present byte) give?
Last third of T bytes; decode the previous byte among the 8 most frequent values, for the present bytes ' ' and 'e'."""
import sys, importlib
import numpy as np
from net import decode
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
T = a.pop("T", 6000); Tissue = importlib.import_module("net%d" % a.pop("net", 5)).Tissue
raw = np.frombuffer(open("../PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8)[:T].astype(int)
net = Tissue(**a); X = []; P = []
for t in range(T):
    c, cl = net.step(raw[t])
    if t >= 2 * T // 3: X.append(c[0].reshape(-1).copy()); P.append(net.predicted)
X = np.array(X); prev = raw[2 * T // 3 - 1:T - 1]; cur = raw[2 * T // 3:T]
for ch in b" e":
    m = cur == ch; vals, cnt = np.unique(prev[m], return_counts=True); top = vals[np.argsort(-cnt)[:8]]; mm = m & np.isin(prev, top)
    print(f"present {chr(ch)!r}: previous byte from region-0 cells {decode(X[mm], prev[mm]):.2f} (chance {1/len(top):.2f}; n={mm.sum()}, distinct codes {len({x.tobytes() for x in X[mm]})})")
print(f"predicted share (last third) {np.mean(P):.2f}")
