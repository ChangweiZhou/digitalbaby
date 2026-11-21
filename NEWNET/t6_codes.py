"""How codes of the context stream change over a life: for each sixth of the run, the share of (position, branch) occurrences
whose code at a region has never been seen before for that (position, branch), and the number of distinct codes so far."""
import sys, importlib
import numpy as np
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
T, G, NB = a.pop("T", 6000), a.pop("G", 2), a.pop("NB", 2)
Tissue = importlib.import_module("net%d" % a.pop("net", 4)).Tissue; net = Tissue(**a)
rng = np.random.default_rng(a.get("seed", 1) + 1000); V = net.n_values
u = rng.choice(V, G + 2 * NB, replace=False); mid = list(u[:G]); starts, ends = u[G:G + NB], u[G + NB:]
s, key = [], []
while len(s) < T:
    b = int(rng.integers(NB)); s += [starts[b]] + mid + [ends[b]]; key += [(p, b) for p in range(G + 2)]
seen = [dict() for _ in range(net.L)]; new = np.zeros((T, net.L), bool); pr = np.zeros(T)
for t in range(T):
    cells, _ = net.step(s[t]); pr[t] = net.predicted
    X = cells.reshape(net.L, -1)
    for d in range(net.L):
        c = X[d].tobytes(); S_ = seen[d].setdefault(key[t], set())
        new[t, d] = c not in S_; S_.add(c)
six = T // 6
for d in (0, net.L // 2, net.L - 1):
    print(f"region {d:2d}: new-code share by sixth " + " ".join(f"{new[i*six:(i+1)*six, d].mean():.3f}" for i in range(6))
          + f" | distinct codes per (position, branch), mean {np.mean([len(v) for v in seen[d].values()]):.1f}")
print("predicted share by sixth " + " ".join(f"{pr[i*six:(i+1)*six].mean():.2f}" for i in range(6)))
