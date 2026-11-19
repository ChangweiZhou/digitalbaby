# Why the supervised one-pass rule (D of yesno3.py) stays below the fitted reader on the same inputs (each driver's own neighbourhood).
# The tissue is run once and its set-(a) outputs kept. Driver v's inputs: its own cells (as yesno3.py) and a constant 1. Target 1 for the
# next byte, 0 otherwise; the emitted byte is the largest output; scored on the last quarter of the positions (as reach.py).
#   (1) D one pass in order through all positions, step eta (target - activity) u / |u|^2, inputs standardised by a running mean and
#       deviation (yesno3.py's "run"), eta 1/256, 1/64, 1/16: share on the last quarter while learning.
#   (2) the same with inputs standardised by the mean and deviation of the fitting part (fixed; not available to a cell in advance).
#   (3) the same rule with fixed standardisation, several passes over the fitting part (1, 2, 8; eta 1/64), then frozen and scored on the last
#       quarter.
#   (4) the fitted ridge on the same inputs (reach_nbhd.py's own-neighbourhood reader) for reference, with the ridge constant chosen
#       as there.
# usage: lms_diag.py <label> [tissue options] seed=.. nbytes=..   -> reach/lms_<label>.json
import sys, os, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tis
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
T = tis.make(kw); NB = T['NB']; N = T['N']; ts = tis.ts; B = T['B']; b = T['b'].astype(np.int64); isfb = B['kind'] == ts.FBK
drivers = list(B['driver']); near = set(drivers)
for i in drivers: near.update(int(m) for m in B['members'][i] if m < N)
near = np.array(sorted(i for i in near if not isfb[i])); col = {c: k for k, c in enumerate(near)}; K = len(near)
mc = np.zeros((256, K))
for v, i in enumerate(drivers):
    for c in [i] + [int(m) for m in B['members'][i] if m < N]:
        if c in col: mc[v, col[c]] = 1
X = np.zeros((NB, K), np.float32)
for lo in range(0, NB, 4096):
    hi = min(NB, lo + 4096); rec = np.zeros((hi - lo, N), np.float32)
    T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(T['b'].ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
    X[lo:hi] = rec[:, near]
X = X[:NB - 1].astype(np.float64); y = b[1:]; M = len(y); nf = (3 * M) // 4
G_ = int(mc.sum(1).max()); IDX = np.full((256, G_), K, np.int64)              # each driver's own cells, padded with a zero column
for v in range(256):
    cs = np.flatnonzero(mc[v]); IDX[v, :len(cs)] = cs
# running standardisation (as yesno3.py)
Xr = np.zeros_like(X); m = np.zeros(K); s2 = np.zeros(K)
for t in range(M):
    r = 1.0 / min(t + 1, 1024); m += r * (X[t] - m); s2 += r * ((X[t] - m) ** 2 - s2); sd = np.sqrt(s2)
    Xr[t] = np.where(sd > 0, (X[t] - m) / np.where(sd > 0, sd, 1), 0.0)
mu = X[:nf].mean(0); sdf = X[:nf].std(0); Xf = np.where(sdf > 0, (X - mu) / np.where(sdf > 0, sdf, 1), 0.0)
ETAS = (1 / 256, 1 / 64, 1 / 16)
def inp(z):
    return np.concatenate([np.append(z, 0.0)[IDX], np.ones((256, 1))], 1)
def lms(Z, eta, order, W=None, record=True):
    W = np.zeros((256, G_ + 1)) if W is None else W; ok = np.zeros(M, bool)
    for t in order:
        U = inp(Z[t]); a = (W * U).sum(1)
        if record: ok[t] = a.argmax() == y[t]
        tg = np.zeros(256); tg[y[t]] = 1; W += (eta * (tg - a) / (U * U).sum(1))[:, None] * U
    return W, ok
def score(Z, W):
    ok = [((W * inp(Z[t])).sum(1).argmax() == y[t]) for t in range(nf, M)]
    return float(np.mean(ok))
res = {}
for nm, Z in (('one pass, running standardisation', Xr), ('one pass, fixed standardisation', Xf)):
    for eta in ETAS:
        _, ok = lms(Z, eta, range(M)); res[f'{nm}, eta {eta:g}'] = float(ok[nf:].mean())
for eta in (1 / 64,):
    W = None
    for p in range(1, 9):
        W, _ = lms(Xf, eta, range(nf), W, record=False)
        if p in (1, 2, 8): res[f'{p} passes over the fitting part, fixed standardisation, frozen, eta {eta:g}'] = score(Xf, W)
# fitted ridge, per driver, own neighbourhood (as reach_nbhd.py)
nin = (4 * nf) // 5; Y = np.zeros((M, 256)); Y[np.arange(M), y] = 1
def ridge(n, lam):
    out = np.zeros((M, 256))
    for v in range(256):
        cs = np.flatnonzero(mc[v]); Xa = Xf[:n][:, cs]; mu_ = Xa.mean(0); my = Y[:n, v].mean(); Xc = Xa - mu_
        G = Xc.T @ Xc; G[np.diag_indices_from(G)] += lam * n; w = np.linalg.solve(G, Xc.T @ (Y[:n, v] - my)); out[:, v] = (Xf[:, cs] - mu_) @ w + my
    return out
inner = {e: float((ridge(nin, 10.0 ** e)[nin:nf].argmax(1) == y[nin:nf]).mean()) for e in range(-6, 3)}
be = max(inner, key=lambda e: (inner[e], e)); res['fitted ridge, own neighbourhood'] = float((ridge(nf, 10.0 ** be)[nf:].argmax(1) == y[nf:]).mean())
res['end mark share'] = float((y[nf:] == 10).mean())
json.dump(dict(label=label, args=kw, bytes=NB, results=res, ridge=10.0 ** be, seconds=time.time() - t0), open(os.path.join(HERE, 'reach', f'lms_{label}.json'), 'w'), indent=1)
print(f'== {label} ({time.time() - t0:.0f} s): share right on the last quarter, %')
for k, v in res.items(): print(f'   {k:80s} {100 * v:6.2f}')
sys.stdout.flush()
