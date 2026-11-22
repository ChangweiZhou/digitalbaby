"""A -> B -> A in one life for net8 (nothing reset, no task signal). life=ctx: A = context stream on lowercase letters (G middle bytes,
two branches), B = tinyshakespeare; life=texts: A = tinyshakespeare, B = Emma (from byte 50,000), then A again where it stopped.
Tests with learning off at the end of each phase: the A test (ctx: right share at the byte after the gap over 100 fresh sequences;
texts: right share on 2,000 bytes of A never read, from 600,000) and the B test (2,000 bytes of B never read, from 600,000).
Control: TA=0 (B only). Usage: python3 t13_life8.py [life=ctx|texts TA TB TR G seed ...net8 parameters]"""
import sys, numpy as np
from net8 import Tissue
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
life, G = a.pop("life", "ctx"), a.pop("G", 8)
rho, K, EP = a.pop("rho", 0.0), a.pop("K", 1000), 40   # rehearsal: a store of K episodes of 40 bytes (each read episode has the same chance to be kept); rho bytes rehearsed a byte read
TA, TB, TR = a.pop("TA", 4000 if life == "ctx" else 16000), a.pop("TB", 16000), a.pop("TR", 2000)
kw = dict(n_values=128, N=1024, tau_max=150, center=1000, fb="none", gran=8192, out_var=1000); kw.update(a)
net = Tissue(**kw); rng = np.random.default_rng(kw.get("seed", 1) + 1000)
here = "/home/user/digitalbaby/"
SH = np.frombuffer(open(here + "PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8).astype(int)
EM = np.frombuffer(open(here + "minifly_v155_keys_dev_v1_20261026/data/austen-emma.txt", "rb").read(), np.uint8).astype(int)
u = rng.choice(np.arange(97, 123), G + 4, replace=False); mid = list(u[:G]); st, en = u[G:G + 2], u[G + 2:]
def seqs(n):
    s, dec = [], []
    for _ in range(n):
        b = rng.integers(2); s += [st[b]] + mid + [en[b]]; dec.append(len(s) - 2)
    return np.array(s), np.array(dec)
store, buf, seen, nread = [], [], 0, 0
srng = np.random.default_rng(kw.get("seed", 1) + 77); every = int(round((EP - 10) / rho)) if rho else 0
STATE = ("h", "y", "z", "O", "byte", "prev_byte", "yprev", "hprev_byte", "m")
def rehearse():
    ep = store[int(srng.integers(len(store)))]
    keep = {k: (getattr(net, k).copy() if hasattr(getattr(net, k), "copy") else getattr(net, k)) for k in STATE if hasattr(net, k)}
    net.byte = None                                                    # nothing is learned from the stream's last answer here
    for t in range(EP):
        net.step(int(ep[t]), t >= 10)                                  # the first ten bytes set up the context; the rest are learned
    for k, v in keep.items(): setattr(net, k, v)
def run(stream, learn):
    global buf, seen, nread
    r = np.zeros(len(stream) - 1, bool)
    for t in range(len(stream) - 1):
        if learn and rho:
            buf.append(int(stream[t]))
            if len(buf) == EP:
                seen += 1
                if len(store) < K: store.append(np.array(buf))
                elif srng.random() < K / seen: store[int(srng.integers(K))] = np.array(buf)
                buf = []
            nread += 1
            if store and nread % every == 0: rehearse()
        net.step(int(stream[t]), learn); r[t] = net.answer() == stream[t + 1]
    return r
if life == "ctx":
    def test_a(): s, d = seqs(100); return run(s, False)[d].mean()
    A1 = lambda n: run(seqs(n // (G + 2))[0], True); B1 = SH; Bt = SH[600000:602001]
else:
    def test_a(): return run(SH[600000:602001], False).mean()
    A1 = None; B1 = EM[50000:]; Bt = EM[600000:602001]
test = lambda: (test_a(), run(Bt, False).mean())
if TA: A1(TA) if life == "ctx" else run(SH[:TA + 1], True)
m1 = test()
run(B1[:TB + 1], True); m2 = test()
A1(TR) if life == "ctx" else run(SH[TA:TA + TR + 1], True)
m3 = test()
print(f"life={life} {' '.join(f'{k}={v}' for k, v in a.items())} TA={TA} TB={TB}: tests A / B   after A {m1[0]:.3f} / {m1[1]:.3f}   after B {m2[0]:.3f} / {m2[1]:.3f}   after the return {m3[0]:.3f} / {m3[1]:.3f}")
