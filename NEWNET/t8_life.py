"""A -> B -> A in one life, nothing reset, no task signal. A: the context stream (G middle bytes, NB branches) on byte values text never
uses (0-9, 11-31); B: tinyshakespeare from its start. Tests with learning off (tissue and output end): A = right share at the byte after
the gap over 100 sequences; text = right share over 2,000 bytes never read (from byte 600,000). Marks: before B, after B, after the return.
The return is learned as usual and scored in blocks of 20 sequences.

Output end (reads every cell of the tissue): rule=delta: every output cell moves its output towards its target (0 for all but the
right byte); rule=gated: the right byte's cell moves towards 1, another cell only if its output is above 0 (it spoke wrongly) -- a cell
that is silent and not the answer is not taught (PLAN_12's letter r in nano/lives).
Usage: python3 t8_life.py [TA=4000 TB=8000 TR=2000 G=8 NB=2 rule=delta eta_out=0.2 seed=1 net=6 ...tissue parameters]"""
import sys, importlib
import numpy as np

def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
TA, TB, TR, G, NB = a.pop("TA", 4000), a.pop("TB", 8000), a.pop("TR", 2000), a.pop("G", 8), a.pop("NB", 2)
rule, eta, netn = a.pop("rule", "delta"), a.pop("eta_out", 0.2), a.pop("net", 6)
alpha = a.pop("alpha", "free")              # free: A on byte values text never uses; letters: A on lowercase letters (text uses them too)
kw = dict(C=128, lines_in=16, n_values=128); kw.update(a)
if netn == 6: kw.setdefault("fast", 1); kw.setdefault("h_L", 6); kw.setdefault("h_burst", 1)
net = importlib.import_module("net%d" % netn).Tissue(**kw); V = net.n_values
rng = np.random.default_rng(kw.get("seed", 1) + 1000)
raw = np.frombuffer(open("../PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8).astype(np.int64)
free = [v for v in range(32) if v != 10 and v not in set(raw.tolist())]
u = rng.choice(free if alpha == "free" else list(range(97, 123)), G + 2 * NB, replace=False); mid = list(u[:G]); starts, ends = u[G:G + NB], u[G + NB:]
def a_seqs(n):
    s, dec = [], []
    for _ in range(n):
        b = rng.integers(NB); s += [starts[b]] + mid + [ends[b]]; dec.append(len(s) - 2)     # index of the last middle byte
    return np.array(s), np.array(dec)

N = net.C * net.M * net.L; ow = np.zeros((V, N))
def run(stream, learn, score_at=None):
    """Read stream; the output end answers the next byte at every step; returns right per step."""
    right = np.zeros(len(stream) - 1, bool)
    for t in range(len(stream) - 1):
        cells, _ = net.step(int(stream[t]), learn)
        act = np.flatnonzero(cells.reshape(-1)); O = ow[:, act].sum(1)
        gave = int(np.argmax(O)) if O.max() > 0 else int(rng.integers(V)); nxt = int(stream[t + 1]); right[t] = gave == nxt
        if learn and len(act):
            tgt = np.zeros(V); tgt[nxt] = 1.0; err = tgt - O
            if rule == "gated": err = np.where((tgt > 0) | (O > 0), err, 0.0)
            ow[:, act] += (eta / len(act) * err)[:, None]
    return right

def test_a():
    s, dec = a_seqs(100); return run(s, False)[dec].mean()
def test_text():
    return run(raw[600000:602001], False).mean()

out = []
sa, da = a_seqs(TA // (G + 2)); r = run(sa, True)
out.append(f"A learned: last quarter at the byte after the gap {r[da[3 * len(da) // 4:]].mean():.3f}")
m0 = (test_a(), test_text())
ow_before = ow.copy()
from net import decode
def tissue_a():                                   # is the branch separable in the tissue's cells at the gap (no output end)?
    s, dec = a_seqs(200); br = np.array([0 if x == starts[0] else 1 for x in s[dec - G]]); X = []
    for t in range(len(s)):
        c, _ = net.step(int(s[t]), False)
        X.append(c.reshape(-1).copy())
    X = np.array(X)[dec]; return decode(X, br)
t0 = tissue_a()
r = run(raw[:TB + 1], True); q = TB // 4
out.append(f"B learned: text right share by quarter " + " ".join(f"{r[i*q:(i+1)*q].mean():.3f}" for i in range(4)))
m1 = (test_a(), test_text())
t1 = tissue_a(); ow_after = ow.copy(); ow[:] = ow_before; m1b = test_a(); ow[:] = ow_after
sr, dr = a_seqs(TR // (G + 2)); r = run(sr, True)
blocks = [r[dr[i:i + 20]].mean() for i in range(0, len(dr), 20)]
m2 = (test_a(), test_text())
print(f"rule={rule} eta_out={eta} seed={kw.get('seed', 1)} G={G}")
for o in out: print(" " + o)
print(f" tests (learning off): A at the gap / text   before B {m0[0]:.3f} / {m0[1]:.3f}   after B {m1[0]:.3f} / {m1[1]:.3f}   after the return {m2[0]:.3f} / {m2[1]:.3f}")
print(" return to A, blocks of 20 sequences: " + " ".join(f"{x:.2f}" for x in blocks))
print(f" put back: A after B with the output end of before B {m1b:.3f} | branch decodable from the tissue's cells at the gap: before B {t0:.2f}, after B {t1:.2f}")
