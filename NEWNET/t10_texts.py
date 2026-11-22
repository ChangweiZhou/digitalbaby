"""Two real texts in one life: A (tinyshakespeare, from its start) -> B (Austen's Emma, from byte 50,000) -> A again (continuing where A
stopped). Nothing reset, no task signal. Tests with learning off, at each mark, on 2,000 bytes of each text never read (A from
600,000, B from 600,000). Output end as t8_life.py (rule=gated default). Reported: next-byte right share.
Usage: python3 t10_texts.py [TA=8000 TB=8000 TR=2000 rule=gated eta_out=0.2 seed=1 ...net6 parameters]"""
import sys, importlib
import numpy as np

def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
TA, TB, TR = a.pop("TA", 8000), a.pop("TB", 8000), a.pop("TR", 2000)
rule, eta = a.pop("rule", "gated"), a.pop("eta_out", 0.2)
step, floor = a.pop("step", "const"), a.pop("floor", 0.0)   # step=count: each output connection steps 1/n (n: its own updates), at least floor
rho, K, EP = a.pop("rho", 0.0), a.pop("K", 1000), 40     # rehearsal: rho bytes rehearsed for each byte read, from a store of K episodes of 40 bytes
kw = dict(C=128, lines_in=16, n_values=128, fast=1, o_L=8, h_L=0, h_burst=1, recall=1, lam=0); kw.update(a)
net = importlib.import_module("net6").Tissue(**kw); V = net.n_values
rng = np.random.default_rng(kw.get("seed", 1) + 1000)
here = "/home/user/digitalbaby/"
A = np.frombuffer(open(here + "PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8).astype(np.int64)
B = np.frombuffer(open(here + "minifly_v155_keys_dev_v1_20261026/data/austen-emma.txt", "rb").read(), np.uint8).astype(np.int64)
N = net.C * net.M * net.L; ow = np.zeros((V, N)); nup = np.zeros(N); osl = np.zeros((V, N))   # osl: the slow connections (step=both)
store, seen, buf = [], 0, []
srng = np.random.default_rng(kw.get("seed", 1) + 77); every = int(round((EP - 10) / rho)) if rho else 0; n_read = 0
ACT = ("cells", "cols", "hold")
def rehearse():
    ep = store[int(srng.integers(len(store)))]
    st = [{k: getattr(p, k).copy() for k in ACT} for p in net.parts]
    for t in range(EP - 1):
        lrn = t >= 9                                       # the first ten bytes set up the context, the rest are learned
        cells, _ = net.step(int(ep[t]), lrn)
        if lrn:
            act = np.flatnonzero(cells.reshape(-1)); O = ow[:, act].sum(1); nxt = int(ep[t + 1])
            tgt = np.zeros(V); tgt[nxt] = 1.0; err = tgt - O
            if rule == "gated": err = np.where((tgt > 0) | (O > 0), err, 0.0)
            if len(act): ow[:, act] += (eta / len(act) * err)[:, None]
    for p, d in zip(net.parts, st):
        for k, v in d.items(): setattr(p, k, v)
def keep(x):                                               # the store: every episode read so far has the same chance to be in it
    global seen, buf
    buf.append(x)
    if len(buf) == EP:
        seen += 1
        if len(store) < K: store.append(np.array(buf))
        elif srng.random() < K / seen: store[int(srng.integers(K))] = np.array(buf)
        buf = []
def run(stream, learn):
    global n_read
    right = np.zeros(len(stream) - 1, bool)
    for t in range(len(stream) - 1):
        if learn and rho:
            keep(int(stream[t])); n_read += 1
            if store and n_read % every == 0: rehearse()
        cells, _ = net.step(int(stream[t]), learn)
        act = np.flatnonzero(cells.reshape(-1)); O = ow[:, act].sum(1) + (osl[:, act].sum(1) if step == "both" else 0)
        gave = int(np.argmax(O)) if O.max() > 0 else int(rng.integers(V)); nxt = int(stream[t + 1]); right[t] = gave == nxt
        if learn and len(act):
            tgt = np.zeros(V); tgt[nxt] = 1.0; err = tgt - O
            if rule == "gated": err = np.where((tgt > 0) | (O > 0), err, 0.0)
            if step == "both":                                 # fast (constant step) and slow (1/n) connections, one error
                nup[act] += 1
                ow[:, act] += (eta / len(act) * err)[:, None]
                osl[:, act] += err[:, None] * (1.0 / nup[act] / len(act))[None, :]
            elif step == "count":
                nup[act] += 1; lr = np.maximum(1.0 / nup[act], floor)
                ow[:, act] += err[:, None] * (lr / len(act))[None, :]
            else:
                ow[:, act] += (eta / len(act) * err)[:, None]
    return right
test = lambda: (run(A[600000:602001], False).mean(), run(B[600000:602001], False).mean())
q = lambda r: " ".join(f"{r[i*len(r)//4:(i+1)*len(r)//4].mean():.3f}" for i in range(4))
rA = run(A[:TA + 1], True); m1 = test()
import copy
def snap(): return [copy.deepcopy({k: v for k, v in p.__dict__.items() if isinstance(v, np.ndarray)}) for p in net.parts], (ow.copy(), nup.copy(), osl.copy())
def load(st):
    for p, d in zip(net.parts, st[0]):
        for k, v in d.items(): getattr(p, k)[...] = v
    ow[...] = st[1][0]; nup[...] = st[1][1]; osl[...] = st[1][2]
sA = snap()
rB = run(B[50000:50000 + TB + 1], True); m2 = test()
sB = snap()
load((sB[0], sA[1])); pb_out = test()          # tissue after B, output end of after A
load((sA[0], sB[1])); pb_tis = test()          # tissue of after A, output end after B
load(sB)
rR = run(A[TA:TA + TR + 1], True); m3 = test()
print(f"step={step} floor={floor} rho={rho} rule={rule} seed={kw.get('seed', 1)} o_L={kw['o_L']} h_L={kw['h_L']}  (learning: A by quarter {q(rA)} | B {q(rB)} | return {q(rR)})")
print(f" tests, learning off, A (Shakespeare) / B (Emma):  after A {m1[0]:.3f} / {m1[1]:.3f}   after B {m2[0]:.3f} / {m2[1]:.3f}   after the return {m3[0]:.3f} / {m3[1]:.3f}")
print(f" put back (A / B tests): output end of after A on the tissue after B {pb_out[0]:.3f} / {pb_out[1]:.3f};   tissue of after A with the output end after B {pb_tis[0]:.3f} / {pb_tis[1]:.3f}")
