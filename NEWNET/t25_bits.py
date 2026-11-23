"""Bits per byte (log-perplexity) on never-read text (23 November 2026). As t17_general.py (stream=moby|code: read T bytes from the
start, learning on; then NT bytes from the middle, learning off), but scores -log2 p(next byte): p from the outputs, max(O, 0)
normalised, mixed with uniform over 128 at eps (eps=0.01 and the eps best for the net on this test, the latter is optimistic).
Reference: count tables of the last k bytes (k=0..4) fitted on the same T bytes, with interpolated back-off (Witten-Bell style).
Usage: python3 t25_bits.py stream=moby|code T=16000 NT=10000 net=10|12 ...net options"""
import sys, glob, importlib, numpy as np, json
from collections import Counter, defaultdict
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
kind, T, NT, netn = a.pop("stream", "moby"), a.pop("T", 16000), a.pop("NT", 10000), a.pop("net", 10)
root = "/home/user/digitalbaby/"
if kind == "moby": raw = open(root + "minifly_v155_keys_dev_v1_20261026/data/melville-moby_dick.txt", "rb").read()
else:
    fs = sorted(f for f in glob.glob(root + "**/*.py", recursive=True) if "/NEWNET/" not in f and "/.git/" not in f)
    raw = b"".join(open(f, "rb").read() for f in fs)
s = np.minimum(np.frombuffer(raw, np.uint8).astype(int), 127); mid = len(s) // 2; test = s[mid:mid + NT + 1]
net = importlib.import_module("net%d" % netn).Tissue(**a)
for t in range(T): net.step(int(s[t])); net.answer()
P = np.zeros((NT, 128)); y = test[1:NT + 1]; right = 0
for t in range(NT):
    net.step(int(test[t]), False); O = np.maximum(net._osum(), 0); P[t] = O / O.sum() if O.sum() > 0 else 1 / 128
    right += net.answer() == test[t + 1]
pt = P[np.arange(NT), y]
bits = {e: float(np.mean(-np.log2((1 - e) * pt + e / 128))) for e in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)}
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
tb = float(np.mean([-np.log2(ppm(list(test[max(0, t - K + 1):t + 1]))[y[t]]) for t in range(K, NT)]))
print(json.dumps(dict(stream=kind, T=T, net=netn, opts=a, right=round(right / NT, 3), bits_eps001=round(bits[0.01], 3),
                      bits_best=round(min(bits.values()), 3), eps_best=min(bits, key=bits.get), table_bits=round(tb, 3))), flush=True)
