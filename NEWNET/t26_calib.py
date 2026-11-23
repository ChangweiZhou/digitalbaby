"""A probability readout for the bits (23 November 2026). As t25_bits.py (read T bytes, learning on; score NT never-read bytes from the
middle, learning off), with two readouts learned online while reading (they do not change the net's own learning):
  temp: p ~ exp(beta * O + c_v), O the net's summed outputs; beta and the per-byte biases c_v learn by the log-loss gradient
        (a gain and a bias for each output cell: local);
  soft: a softmax readout over the net's active cells (context cells, holders' expansion cells, and net12's lag inputs when present):
        logits z_v = c_v + sum over active cells i of w[v, i]; p = softmax(z) (divisive normalisation across the output cells, as
        lateral inhibition); w[v, i] += eta (y_v - p_v) for each active i: each synapse by its own output cell's error and its
        presynaptic activity (the delta rule of a single layer; no error is carried back).
Reports bits per byte (no mixing with uniform except 1e-6 for safety) and right share of each readout, next to the clipped-and-
normalised outputs (t25's, mixed with 1% uniform) and the back-off count table.
Usage: python3 t26_calib.py stream=moby|code T=16000 NT=10000 net=10|12 ...net options"""
import sys, glob, importlib, json, numpy as np
from collections import Counter, defaultdict
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
kind, T, NT, netn = a.pop("stream", "moby"), a.pop("T", 16000), a.pop("NT", 10000), a.pop("net", 10)
eta_s, eta_t = a.pop("eta_s", 1.0), a.pop("eta_t", 0.01)
root = "/home/user/digitalbaby/"
if kind == "moby": raw = open(root + "minifly_v155_keys_dev_v1_20261026/data/melville-moby_dick.txt", "rb").read()
else:
    fs = sorted(f for f in glob.glob(root + "**/*.py", recursive=True) if "/NEWNET/" not in f and "/.git/" not in f)
    raw = b"".join(open(f, "rb").read() for f in fs)
s = np.minimum(np.frombuffer(raw, np.uint8).astype(int), 127); mid = len(s) // 2; test = s[mid:mid + NT + 1]
net = importlib.import_module("net%d" % netn).Tissue(**a); V = net.V
nctx = net.L * net.nc; nh = sum(h.gran for h in net.holds); nlag = getattr(net, "nin", 0) if getattr(net, "lagro", 0) else 0
NIN = nctx + nh + nlag
Wsoft = np.zeros((V, NIN)); csoft = np.zeros(V); beta = 1.0; ctemp = np.zeros(V)
def active():
    idx = [np.flatnonzero(net.xc.reshape(-1))]
    o = nctx
    for h in net.holds: idx.append(o + np.flatnonzero(h.z)); o += h.gran
    if nlag: idx.append(o + net._act())
    return np.concatenate(idx)
def sm(z): z = z - z.max(); e = np.exp(z); return e / e.sum()
def readouts():
    O = net._osum(); act = active()
    pt = sm(beta * O + ctemp); ps = sm(csoft + Wsoft[:, act].sum(1))
    Oc = np.maximum(O, 0); pc = Oc / Oc.sum() if Oc.sum() > 0 else np.ones(V) / V
    return O, act, pt, ps, 0.99 * pc + 0.01 / V
for t in range(T):
    net.step(int(s[t])); net.answer()
    if t + 1 < T:
        y = int(s[t + 1]); O, act, pt, ps, _ = readouts(); oh = np.zeros(V); oh[y] = 1.0
        g = oh - pt; beta += eta_t * float(g @ O); ctemp += eta_t * g                         # temp: gain and biases
        g = oh - ps; csoft += eta_s * g; Wsoft[:, act] += eta_s * g[:, None] / max(len(act), 1)   # soft: delta rule (step shared by the active cells)
B = defaultdict(float); R = defaultdict(int)
for t in range(NT):
    net.step(int(test[t]), False); y = int(test[t + 1]); O, act, pt, ps, pc = readouts()
    for k, p in (("clip", pc), ("temp", pt), ("soft", ps)): B[k] += -np.log2(p[y] + 1e-6); R[k] += int(np.argmax(p)) == y
K = 4; tab = [defaultdict(Counter) for _ in range(K + 1)]
for t in range(K, T - 1):
    for k in range(K + 1): tab[k][tuple(s[t - k + 1:t + 1]) if k else ()][s[t + 1]] += 1
def ppm(ctx):
    p = np.ones(128) / 128
    for k in range(K + 1):
        c = tab[k].get(tuple(ctx[len(ctx) - k:]) if k else ())
        if c:
            n = sum(c.values()); u = len(c); lam = n / (n + u); q = np.zeros(128)
            for b, m in c.items(): q[b] = m / n
            p = lam * q + (1 - lam) * p
    return p
tb = float(np.mean([-np.log2(ppm(list(test[max(0, t - K + 1):t + 1]))[test[t + 1]]) for t in range(K, NT)]))
print(json.dumps(dict(stream=kind, T=T, net=netn, opts=a, eta_s=eta_s, eta_t=eta_t, beta=round(beta, 3),
                      bits={k: round(v / NT, 3) for k, v in B.items()}, right={k: round(v / NT, 3) for k, v in R.items()}, table_bits=round(tb, 3))), flush=True)
