"""IDEA3 copy of t22_rules.py: the net is always net12_cf (net=12 only; gate_in=all|match|match+pos as a net option)."""
"""Has a rule been learned? First encounters in one life (22 November 2026).
Stream: lines "w=W\n", w = 3 symbols; the rule sets W: copy (W = w) or rev (W = w reversed). Rules in one life: copy for lines
[0, S1), rev for [S1, S2), copy again from S2 (A -> B -> A). Symbols: 93 (printable ASCII but "=", in a fixed shuffled order); the alphabet starts
with 8 and gains one every GROW lines; a line holds the newest symbol with chance 0.3 (so that new symbols are met).
Every byte is predicted before it is read (prequential, the life's own answers, learning on); only answer bytes (the 3 symbols of W) of
items met for the first time under the rule in force are scored. Each such byte is put in one layer:
  a - storage could know it: the shortest stretch of the past that fixes it (from the symbol it repeats to the byte before it) has been
      read before, followed by this byte;
  b - not so, but the symbol has been an answer byte at least 5 times before;
  n - not so, and the symbol has been an answer byte only 1-4 times before;
  c - the symbol has never been an answer byte before (a new symbol for the answer).
Reported: right share and bits (p from the outputs: max(O,0) normalised, mixed 0.99/0.01 with uniform over 128) in each layer and phase,
and the right share in windows of 250 lines after each change of rule. Baseline (net=store): storage with the same window, an n-gram
table of every context up to 8 bytes with back-off to the longest one seen.
Usage: python3 t22_rules.py net=10|12|store [seed lines S1 S2 GROW ...net options] [struct=K8|N1]"""
import sys, json, time, numpy as np
from collections import defaultdict
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=", 1) for x in sys.argv[1:])}
netn, sname, seed = a.pop("net", 10), a.pop("struct", "K8"), a.pop("seed", 1)
NL, S1, S2, GROW, out = a.pop("lines", 24000), a.pop("S1", 12000), a.pop("S2", 18000), a.pop("GROW", 250), a.pop("out", None)
STRUCT = dict(K8=dict(L=16, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, c_th_on=0.9, c_th_match=0.75),
              N1=dict(L=8, cC=64, M=32, joint=1, mesh=2, mesh_base=0, mesh_dag=2, c_k_win=4))
rng = np.random.default_rng(seed + 4000)
SYM = [int(c) for c in np.random.default_rng(7).permutation([c for c in range(33, 127) if c != 61])]   # 93 printable symbols, not "="

class Store:                                                                       # the storage baseline: back-off n-gram table, orders 1..8
    def __init__(s, K=8): s.K, s.t, s.h = K, defaultdict(lambda: defaultdict(int)), []
    def probs(s):
        for k in range(min(s.K, len(s.h)), 0, -1):
            c = s.t.get(tuple(s.h[-k:]))
            if c:
                p = np.zeros(128); [p.__setitem__(b, n) for b, n in c.items()]; return p
        return np.ones(128)
    def step(s, b, learn=True):
        if learn:
            for k in range(1, min(s.K, len(s.h)) + 1): s.t[tuple(s.h[-k:])][b] += 1
        s.h = (s.h + [b])[-s.K:]
if netn == "store": net = Store()
else:
    import importlib
    opts = dict(STRUCT[sname]); opts.update(a); net = importlib.import_module("net12_cf").Tissue(seed=seed, n_values=128, **opts)
def probs():
    if netn == "store": O = net.probs()
    else: O = net._osum()
    O = np.maximum(O, 0); return O / O.sum() if O.sum() > 0 else np.ones(128) / 128

seen_items, answered, past = set(), defaultdict(int), set()                                  # items met per rule; symbols given as answers; stretches read
hist = []                                                                          # the bytes read, for the stretches (layer a only)
R = defaultdict(lambda: [0, 0, 0.0])                                               # (phase, layer) -> [n, right, bits]
win = defaultdict(lambda: [0, 0])                                                  # (change, window) -> [n, right]
t0 = time.time()
for ln in range(NL):
    phase = 0 if ln < S1 else 1 if ln < S2 else 2; rule = "rev" if phase == 1 else "copy"
    n_al = min(len(SYM), 8 + ln // GROW); al = SYM[:n_al]
    w = [al[int(rng.integers(n_al))] for _ in range(3)]
    if rng.random() < 0.3: w[int(rng.integers(3))] = al[-1]
    W = w if rule == "copy" else w[::-1]
    line = w + [61] + W + [10]; first = (rule, tuple(w)) not in seen_items; seen_items.add((rule, tuple(w)))
    for i, b in enumerate(line):
        if 4 <= i <= 6:                                                            # an answer byte: score before reading it
            if first:
                p = probs(); right = int(np.argmax(p)) == b if netn == "store" else net.answer() == b
                src = (i - 4) if rule == "copy" else (6 - i)                       # the position in w of the symbol it repeats
                d = i - src                                                        # how far back it is
                stretch = tuple(hist[-d:]) + (b,)
                lay = "a" if stretch in past else "c" if not answered[b] else "n" if answered[b] < 5 else "b"
                pb = 0.99 * p[b] + 0.01 / 128
                r = R[(phase, lay)]; r[0] += 1; r[1] += right; r[2] += -np.log2(pb)
                if phase: k = (ln - (S1 if phase == 1 else S2)) // 250; ww = win[(phase, k)]; ww[0] += 1; ww[1] += right
            elif netn != "store": net.answer()
            answered[b] += 1
        elif netn != "store": net.answer()
        hist.append(b)
        for d in range(1, 8):
            if len(hist) > d: past.add(tuple(hist[-d - 1:]))
        net.step(b, True)
    if (ln + 1) % 3000 == 0:
        print(f"line {ln + 1} {time.time() - t0:.0f}s " + " ".join(f"p{ph}{l}:{v[1] / max(v[0], 1):.2f}({v[0]})" for (ph, l), v in sorted(R.items())), flush=True)
res = dict(net=netn, struct=sname if netn != "store" else None, seed=seed, opts=a, lines=NL, S1=S1, S2=S2, GROW=GROW, s=round(time.time() - t0),
           layers={f"{ph}{l}": dict(n=v[0], right=round(v[1] / max(v[0], 1), 3), bits=round(v[2] / max(v[0], 1), 2)) for (ph, l), v in sorted(R.items())},
           after_change={f"{ph}": [round(win[(ph, k)][1] / max(win[(ph, k)][0], 1), 2) for k in range(12) if win[(ph, k)][0]] for ph in (1, 2)})
print(json.dumps(res), flush=True)
if out: open(out, "a").write(json.dumps(res) + "\n")
