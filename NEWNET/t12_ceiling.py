"""Diagnostic on text: the online answer right share, and the best linear readout of the cells' states fitted offline (ridge, bytes
0-12,000 to fit, 12,000-16,000 to score), for net8 with the given parameters. The offline readout is a measuring instrument only."""
import sys, numpy as np
from net8 import Tissue
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
raw = np.frombuffer(open('../PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt', 'rb').read(), np.uint8)[:16001].astype(int)
kw = dict(n_values=128, tau_max=150, center=1000); kw.update(a); net = Tissue(**kw); Y = []; R = []
for t in range(16000):
    net.step(int(raw[t])); Y.append(net.y.copy()); R.append(net.answer() == raw[t + 1])
Y = np.array(Y); X = np.c_[Y, np.ones(16000)]; T = np.eye(128)[raw[1:]]
Wr = np.linalg.solve(X[:12000].T @ X[:12000] + np.eye(X.shape[1]), X[:12000].T @ T[:12000])
print(f"{' '.join(f'{k}={v}' for k, v in a.items())}: online {np.mean(R[12000:]):.3f} | best linear readout of the state {(np.argmax(X[12000:] @ Wr, 1) == raw[12001:]).mean():.3f}")
