"""Test 2 on the minicolumn tissue (net3): the output end learns from feedback to give the next byte.
Output end as in t3_output.py: one cell a byte value, FAN_OUT connections from random cells of the last region, teach or yesno,
each output cell's total strength held at 1 (subtractive), babbling when it has nothing to go on.
Streams: 'cycle' (P distinct bytes repeated); 'context' (sequences A B C D and E B C F in random order: after C the next byte is
known only from two bytes back); 'random' (the control). Right share in quarters; for 'context', also at the decisive steps
(the byte after C) in the last quarter."""
import sys
import numpy as np
import importlib

def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
T, P, mode, kind = a.pop("T", 6000), a.pop("P", 7), a.pop("mode", "teach"), a.pop("stream", "cycle")
eta, eps, fan_out = a.pop("eta_out", 0.1), a.pop("eps", 0.1), a.pop("fan_out", 64)
G, NB, out_all = a.pop("G", 2), a.pop("NB", 2), a.pop("out_all", 0)   # out_all=1: the output end reads every region                    # context stream: G shared middle bytes, NB branches
Tissue = importlib.import_module("net%d" % a.pop("net", 3)).Tissue; net = Tissue(**a); rng = np.random.default_rng(a.get("seed", 1) + 1000); V = net.n_values
if kind == "cycle":
    pat = rng.choice(V, P, replace=False); stream = np.array([pat[t % P] for t in range(T + 1)])
elif kind == "context":
    u = rng.choice(V, G + 2 * NB, replace=False); mid = list(u[:G]); starts, ends = u[G:G + NB], u[G + NB:]
    Cc = mid[-1]; s = []
    while len(s) < T + 1:
        b = rng.integers(NB); s += [starts[b]] + mid + [ends[b]]
    stream = np.array(s[: T + 1])
else:
    stream = rng.integers(0, V, T + 1)
ncell = net.C * net.M * (net.L if out_all else 1)
fan_out = fan_out or ncell
osrc = rng.choice(ncell, (V, fan_out)) if fan_out < ncell else np.tile(np.arange(ncell), (V, 1)); ow = np.zeros((V, fan_out))
right = np.zeros(T, bool)
for t in range(T):
    cells, cols = net.step(stream[t])
    x = (cells if out_all else cells[-1]).reshape(-1); pre = x[osrc]
    O = (ow * pre).sum(1)
    gave = int(np.argmax(O)) if O.max() > 0 else int(rng.integers(0, V))
    if mode == "yesno" and rng.random() < eps: gave = int(rng.integers(0, V))
    nxt = stream[t + 1]; right[t] = gave == nxt
    if mode == "teach":
        tgt = np.zeros(V); tgt[nxt] = 1.0; g = np.zeros(V); g[gave] = 1.0
        ow += eta * pre * (tgt - g)[:, None]
    else:
        ow[gave] += eta * pre[gave] * (1.0 if gave == nxt else -1.0)
    np.clip(ow, 0.0, 4.0, out=ow); ow += (1.0 - ow.sum(1, keepdims=True)) / fan_out; np.clip(ow, 0.0, 4.0, out=ow)
q = T // 4
extra = ""
if kind == "context":
    dec = np.array([t for t in range(3 * q, T) if stream[t] == Cc])
    ph = np.array([t % (G + 2) for t in range(T)])   # position in the sequence (sequences have equal length)
    extra = " | by position " + " ".join(f"{right[3*q:][ph[3*q:]==p].mean():.2f}" for p in range(G + 2))
    extra += f" | at the byte after C (last quarter): {right[dec].mean():.3f} (1/NB={1/NB:.3f} without context; G={G} NB={NB})"
print(f"{mode:5s} {kind:7s} regions={net.L}: right share by quarter " + " ".join(f"{right[i*q:(i+1)*q].mean():.3f}" for i in range(4)) + extra)
