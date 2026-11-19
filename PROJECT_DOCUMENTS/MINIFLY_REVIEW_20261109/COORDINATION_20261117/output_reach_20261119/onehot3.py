# The reader of reach.py fed with the last three bytes (three one-hots, 768 features: the present byte, the previous byte and the one before),
# on the same fitting and scored positions as reach.py. No tissue is run.   usage: onehot3.py <seed> <nbytes>   -> reach/onehot3_<seed>_<nbytes>.json
import sys, os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'deps', 'PREP'))
import life_stream as LS, score_emissions as SE
seed, NB = int(sys.argv[1]), int(sys.argv[2])
L = LS.life(seed); b = L['bytes'][:NB].astype(np.int64)
tt = np.arange(NB - 1); y = b[1:]; nf = (3 * len(tt)) // 4; nin = (4 * nf) // 5
Y = np.zeros((len(tt), 256)); Y[np.arange(len(tt)), y] = 1.0
X = np.zeros((NB - 1, 768)); X[tt, b[tt]] = 1; X[tt[1:], 256 + b[tt[1:] - 1]] = 1; X[tt[2:], 512 + b[tt[2:] - 2]] = 1
def solve(Xa, Ya, lam):
    mu = Xa.mean(0); my = Ya.mean(0); Xc = Xa - mu; G = Xc.T @ Xc; G[np.diag_indices_from(G)] += lam * len(Xa)
    W = np.linalg.solve(G, Xc.T @ (Ya - my)); return lambda Z: (Z - mu) @ W + my
inner = {e: float((solve(X[:nin], Y[:nin], 10.0 ** e)(X[nin:nf]).argmax(1) == y[nin:nf]).mean()) for e in range(-6, 5)}
best = max(inner, key=lambda e: (inner[e], e)); g = solve(X[:nf], Y[:nf], 10.0 ** best)(X[nf:]).argmax(1)
pos = tt[nf:] + 1; right = g == b[pos]
K = SE.kinds(L); km = {'all': np.ones(len(pos), bool)}
for kn, m in K.items():
    if m[:NB][pos].any(): km[kn] = m[:NB][pos]
r12 = np.zeros(len(pos), bool)
for kn in km:
    if kn.startswith('R'): r12 |= km[kn]
km['rest (not R1, R2)'] = ~r12
out = dict(seed=seed, bytes=NB, features=768, ridge=10.0 ** best, inner_check=inner, shares={k: dict(n=int(m.sum()), right=float(right[m].mean())) for k, m in km.items()})
os.makedirs(os.path.join(HERE, 'reach'), exist_ok=True); json.dump(out, open(os.path.join(HERE, 'reach', f'onehot3_{seed}_{NB}.json'), 'w'), indent=1)
print(seed, NB, ' '.join(f"{k.split(' (')[0]} {100 * v['right']:.2f}" for k, v in out['shares'].items()), f"ridge {10.0 ** best:.0e}")
