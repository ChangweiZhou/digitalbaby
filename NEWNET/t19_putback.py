"""Where the first text is lost through the second (life=enzh or zhen as in t13_life8.py, net10): after the second text, put back
one part as it was after the first, test the first text (learning off), and take the part back out. Parts: the context cells'
connections (sources, strengths, use, reliability: what recycling of cells changes); the context compartments' output strengths; the
holders' output strengths. Usage: python3 t19_putback.py life=enzh|zhen [TA TB NT seed ...net10 options]"""
import sys, copy, numpy as np
from net10 import Tissue
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
life, TA, TB, NT = a.pop("life", "zhen"), a.pop("TA", 16000), a.pop("TB", 16000), a.pop("NT", 10000)
net = Tissue(n_values=256, **a)
here = "/home/user/digitalbaby/"
EN = np.frombuffer(open(here + "PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8).astype(int)
ZH = np.frombuffer(open(here + "NEWNET/data/hongloumeng.txt", "rb").read(), np.uint8).astype(int)
en_r, en_t, zh_r, zh_t = EN, EN[600000:600000 + NT + 1], ZH[100000:], ZH[1500000:1500000 + NT + 1]
(Ar, At, Br) = (en_r, en_t, zh_r) if life == "enzh" else (zh_r, zh_t, en_r)
def run(s, learn):
    r = np.zeros(len(s) - 1, bool)
    for t in range(len(s) - 1): net.step(int(s[t]), learn); r[t] = net.answer() == s[t + 1]
    return r.mean()
CTX = ("rsrc", "rw", "gsrc", "use", "rel")
def snap():
    return {"ctx": {k: getattr(net.ctx, k).copy() for k in CTX}, "Ac": net.Ac.copy(), "hA": [h.A.copy() for h in net.holds]}
def put(part, st):
    if part == "ctx":
        for k, v in st["ctx"].items(): getattr(net.ctx, k)[...] = v
        if hasattr(net.ctx, "reindex"): net.ctx.reindex()
    elif part == "Ac": net.Ac[...] = st["Ac"]
    elif part == "hA":
        for h, v in zip(net.holds, st["hA"]): h.A[...] = v
run(Ar[:TA + 1], True); a1 = run(At, False); s1 = snap()
run(Br[:TB + 1], True); a2 = run(At, False); s2 = snap()
res = {}
for part in ("ctx", "Ac", "hA"):
    put(part, s1); res[part] = run(At, False); put(part, s2)
put("Ac", s1); put("hA", s1); res["both outputs"] = run(At, False); put("Ac", s2); put("hA", s2)
print(f"life={life} {' '.join(f'{k}={v}' for k, v in a.items())}: first text after it {a1:.3f}, after the second {a2:.3f}; with put back: "
      + ", ".join(f"{k} {v:.3f}" for k, v in res.items()))
