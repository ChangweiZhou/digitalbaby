"""Diagnostic for the context stream: at each position of the shared middle, is the branch decodable from the cells of each region,
and how many distinct codes does a (position, branch) get? Last third of the run. No output end."""
import sys
import numpy as np
import importlib
from net import decode
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
T, G, NB = a.pop("T", 6000), a.pop("G", 2), a.pop("NB", 2)
Tissue = importlib.import_module("net%d" % a.pop("net", 3)).Tissue; net = Tissue(**a); rng = np.random.default_rng(a.get("seed", 1) + 1000); V = net.n_values
u = rng.choice(V, G + 2 * NB, replace=False); mid = list(u[:G]); starts, ends = u[G:G + NB], u[G + NB:]
s, br, pos = [], [], []
while len(s) < T:
    b = int(rng.integers(NB)); s += [starts[b]] + mid + [ends[b]]; br += [b] * (G + 2); pos += list(range(G + 2))
s, br, pos = np.array(s[:T]), np.array(br[:T]), np.array(pos[:T])
keep = []; pred = []; sur = []
for t in range(T):
    cells, cols = net.step(s[t]); pred.append(net.predicted); sur.append(getattr(net, 'surprise', np.zeros(net.L)).copy())
    if t >= 2 * T // 3: keep.append(cells.reshape(net.L, -1).copy())
K = np.array(keep); bk, pk = br[2 * T // 3:], pos[2 * T // 3:]
print(f"G={G} NB={NB} predicted share (last third) {np.mean(pred[2*T//3:]):.2f}")
Su = np.array(sur)[2 * T // 3:]
print(" surprise share by position (regions 0 / last): " + " ".join(f"{Su[pk == p, 0].mean():.2f}/{Su[pk == p, -1].mean():.2f}" for p in range(G + 2)))
for p in range(1, G + 1):
    m = pk == p
    row = [decode(K[m, d], bk[m]) for d in range(net.L)]
    nd = [len({K[i, d].tobytes() for i in np.where(m & (bk == 0))[0]}) for d in (0, net.L - 1)]
    print(f" middle byte {p}: branch score by region " + " ".join(f"{x:.2f}" for x in row) + f" | distinct codes branch 0, region 0/last: {nd}")
