"""Learning set (Harlow 1949) in one life (23 November 2026): does the network learn to learn?
One continuous life, nothing reset: 20 rules, each never seen before, one after another (BL lines each), then a revisit block of 5
earlier rules (RV lines each). Lines "w D W\n" (no spaces, as t23): w = 3 symbols of the 40-symbol alphabet of t23 (all 40 from the
start), D = the rule's delimiter, W = the rule's answer.
Rule family (fixed, listed by `python3 t28_learnset.py list=1`; see LEARNSET_PLAN.md):
  P  (6): W = w permuted by one of the 6 permutations of 3 positions (W[i] = w[pi[i]]);
  M  (6): W[i] = SIGMA(w[pi[i]]), SIGMA one fixed random derangement of the 40 symbols (the same for all M rules);
  C  (8): W = w permuted by pi, with a constant symbol k inserted at position q (4 answer bytes); 4 positions x 2 permutations.
cue=1: each rule its own new delimiter (20 bytes never used otherwise); cue=0: all rules use "=".
Scores (prequential: every byte predicted before it is read, learning on; the life's own answers, teacher-forced):
  - per rule block: first-met answer bytes (items (rule, w) met for the first time), right share per line;
      L80 / L50 = lines (within the block) until the right share over the last 100 lines first reaches 0.8 / 0.5 (window end;
      >= 100; None = not reached);  f20 / f100 / f500 = right share in the block's first 20 / 100 / 500 lines;  last500;
      bins = right share in 50-line bins;
  - revisit blocks: the same measures (savings against the rule's first block);
  - probes with learning off after the 20 rules and at the end: 30 fresh items of every rule (right share of the answer bytes);
  - bits of the whole life (as t23: p = max(O, 0) normalised, mixed 0.99/0.01 with uniform) and its parts (input / answer bytes).
Nets: net=10|12 (net%d.Tissue), net=12rc (net12_rc.Tissue), net=store (back-off table of t23, orders 1-8).
Usage: python3 t28_learnset.py net=10|12|12rc|store [struct=K8 cue=0|1 seed=1 BL=1500 RV=500 ...net options] [out=FILE]"""
import sys, json, time, numpy as np
from collections import defaultdict
from itertools import permutations
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=", 1) for x in sys.argv[1:])}
netn, sname, seed, cue = str(a.pop("net", 12)), a.pop("struct", "K8"), a.pop("seed", 1), a.pop("cue", 1)
BL, RV, NPR, out, LIST = a.pop("BL", 1500), a.pop("RV", 500), a.pop("NPR", 30), a.pop("out", None), a.pop("list", 0)
STRUCT = dict(K8=dict(L=16, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, c_th_on=0.9, c_th_match=0.75))

# ---- the rule family and its order (fixed; independent of seed and cue) ----
SYM = [int(c) for c in np.random.default_rng(7).permutation([c for c in range(33, 127) if chr(c) not in "=~+"])][:40]   # as t23
fr = np.random.default_rng(28)
while True:                                                                         # SIGMA: a random derangement of the 40 symbols
    pm = fr.permutation(40)
    if not np.any(pm == np.arange(40)): break
SIGMA = {SYM[i]: SYM[int(pm[i])] for i in range(40)}
PERMS = list(permutations(range(3)))
RULES = [dict(kind="P", pi=p) for p in PERMS] + [dict(kind="M", pi=p) for p in PERMS]
for q in range(4):
    for p in fr.choice(6, 2, replace=False): RULES.append(dict(kind="C", pi=PERMS[int(p)], q=q, k=SYM[int(fr.integers(40))]))
FREE = [c for c in range(33, 127) if c not in SYM and c != 61]
DELS = [int(x) for x in fr.choice(FREE, 20, replace=False)]
ORDER = [int(x) for x in fr.permutation(20)]                                         # rule r of the life = RULES[ORDER[r]]
LIFE = [RULES[i] for i in ORDER]
for r, R_ in enumerate(LIFE): R_["delim"] = DELS[r] if cue else 61; R_["idx"] = r + 1
REVISIT = [1, 5, 9, 13, 17]                                                         # life rule numbers revisited at the end
def name(R_):
    s = R_["kind"] + "".join(map(str, R_["pi"]))
    return s + (f"_q{R_['q']}k{chr(R_['k'])}" if R_["kind"] == "C" else "")
def answer(R_, w):
    W = [w[j] for j in R_["pi"]]
    if R_["kind"] == "M": W = [SIGMA[x] for x in W]
    if R_["kind"] == "C": W = W[:R_["q"]] + [R_["k"]] + W[R_["q"]:]
    return W
if LIST:
    for R_ in LIFE: print(R_["idx"], name(R_), "delim", chr(R_["delim"]))
    print("SIGMA", "".join(chr(SYM[i]) for i in range(40)), "->", "".join(chr(SIGMA[SYM[i]]) for i in range(40))); sys.exit()

rng, prng = np.random.default_rng(seed + 4000), np.random.default_rng(seed + 6000)
class Store:
    def __init__(s, K=8): s.K, s.t, s.h = K, defaultdict(lambda: defaultdict(int)), []
    def probs(s):
        for k in range(min(s.K, len(s.h)), 0, -1):
            c = s.t.get(tuple(s.h[-k:]))
            if c:
                p = np.zeros(128)
                for b, n in c.items(): p[b] = n
                return p
        return np.ones(128)
    def step(s, b, learn=True):
        if learn:
            for k in range(1, min(s.K, len(s.h)) + 1): s.t[tuple(s.h[-k:])][b] += 1
        s.h = (s.h + [b])[-s.K:]
if netn == "store": net = Store()
else:
    import importlib
    opts = dict(STRUCT[sname]); opts.update(a)
    net = importlib.import_module("net12_rc" if netn == "12rc" else "net" + netn).Tissue(seed=seed, n_values=128, **opts)
def probs():
    O = net.probs() if netn == "store" else net._osum()
    O = np.maximum(O, 0); return O / O.sum() if O.sum() > 0 else np.ones(128) / 128
def guess(): return int(np.argmax(probs())) if netn == "store" else net.answer()
def word(r): return [SYM[int(r.integers(40))] for _ in range(3)]

BITS, NB = defaultdict(float), defaultdict(int); t0 = time.time()
def read(b, kind):
    p = probs(); g = guess(); pb = 0.99 * p[b] + 0.01 / 128
    BITS[kind] += -np.log2(pb); NB[kind] += 1; net.step(b, True); return g == b
def probe(tag):
    r = {}
    for R_ in LIFE:
        ok = n = 0
        for _ in range(NPR):
            w = word(prng)
            for b in w + [R_["delim"]]: net.step(b, False)
            for b in answer(R_, w): ok += guess() == b; n += 1; net.step(b, False)
            net.step(10, False)
        r[R_["idx"]] = round(ok / n, 3)
    print(f"probe {tag}: " + json.dumps(r), flush=True); return r
def crit(ok, nb, th):
    co, cn = np.cumsum(ok), np.cumsum(nb)
    for k in range(99, len(ok)):
        s = (co[k] - (co[k - 100] if k >= 100 else 0)) / max(cn[k] - (cn[k - 100] if k >= 100 else 0), 1)
        if s >= th: return k + 1
    return None
seen = set(); blocks = []
def block(R_, n, tag):
    ok, nb = np.zeros(n), np.zeros(n)
    for k in range(n):
        w = word(rng); first = (R_["idx"], tuple(w)) not in seen; seen.add((R_["idx"], tuple(w)))
        for b in w + [R_["delim"]]: read(b, "input")
        for b in answer(R_, w):
            g = read(b, "answer")
            if first: ok[k] += g; nb[k] += 1
        read(10, "input")
    sh = lambda lo, hi: round(float(ok[lo:hi].sum() / max(nb[lo:hi].sum(), 1)), 3)
    d = dict(tag=tag, rule=R_["idx"], name=name(R_), kind=R_["kind"], n=n, first_met=int(nb.sum()),
             L80=crit(ok, nb, 0.8), L50=crit(ok, nb, 0.5), f20=sh(0, 20), f100=sh(0, 100), f500=sh(0, 500), last500=sh(n - 500, n),
             all=sh(0, n), bins=[sh(i, i + 50) for i in range(0, n, 50)])
    blocks.append(d)
    print(f"{tag} rule {R_['idx']} {d['name']} ({time.time() - t0:.0f}s): L80 {d['L80']} L50 {d['L50']} f20 {d['f20']} f100 {d['f100']}"
          f" f500 {d['f500']} last500 {d['last500']}", flush=True)
for R_ in LIFE: block(R_, BL, "learn")
pr20 = probe("after20")
for i in REVISIT: block(LIFE[i - 1], RV, "revisit")
prend = probe("end")
res = dict(net=netn, struct=sname if netn != "store" else None, cue=cue, seed=seed, opts=a, BL=BL, RV=RV, s=round(time.time() - t0),
           rules=[dict(idx=R_["idx"], name=name(R_), kind=R_["kind"], delim=chr(R_["delim"])) for R_ in LIFE],
           bits=dict(total=round(sum(BITS.values())), **{k: round(v) for k, v in BITS.items()}, per_byte={k: round(BITS[k] / NB[k], 3) for k in BITS}),
           blocks=blocks, probe_after20=pr20, probe_end=prend)
print(json.dumps(res), flush=True)
if out: open(out, "a").write(json.dumps(res) + "\n")
