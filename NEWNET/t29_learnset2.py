"""Learning set 2 (23 November 2026): learning to learn, or reuse of familiar (position, lag) pairs? (plan: LEARNSET2_PLAN.md)
Lines "w D W\n" (no spaces): w = 4 symbols of t23/t28's 40-symbol alphabet, D = the rule's delimiter, W = 4 answer bytes.
A position rule f: W[i] = w[f(i)]. When W[i] is predicted, the last byte read is 4+i, so w[j] is 4+i-j back (lag 4+i-j, 1..7 < W=8);
the delimiter is i back, so (answer position i, source lag 4+i-j) is the rule's pair at position i.
Two halves of the 16 pairs (each answer position has 2 sources in each half):
  A: f(i) = i + c_i mod 4, c_i in {0, 1}  -> lags {3, 4, 7}        B: c_i in {2, 3} -> lags {1, 2, 5, 6}
  A rule of a half is a bit pattern b (c_i = offset[b_i]); 16 per half; its complement (all bits flipped) shares no pair with it.
Test rules (all four cyclic rotations; pairwise disjoint lag sets):
  X1 = A0000 copy (W = w0w1w2w3), X2 = A1111 rot1 (w1w2w3w0), Y1 = B0000 rot2 (w2w3w0w1), Y2 = B1111 rot3 (w3w0w1w2).
History H_A / H_B: the same 8 non-constant bit patterns (4 complementary pairs, fixed order) in half A / B: each pair of the half used 4x.
Lives (nothing reset within a life):
  life=A : H_A (8 rules), then Y1 (new pairs), X1 (familiar pairs), Y2 (new), X2 (familiar)
  life=B : H_B (8 rules), then X1 (new), Y1 (familiar), X2 (new), Y2 (familiar)
  life=RA / RB: 8 blocks of random answers (format only; own delimiters as H), then the test rules in the order of A / B
  life=fresh test=X1|X2|Y1|Y2: that test rule only, as the first rule of a new net
Each rule (and random block) has its own delimiter: history slot k -> DH[k], test rule -> DT[rule] (the same in every life).
Scores as t28 (prequential, learning on, own answers teacher-forced): per block first-met answer bytes; L80 / L50 (lines until the right
share over the last 100 lines >= 0.8 / 0.5; L80s: the same over the last 20 lines, >= 20), err80 (wrong first-met answer bytes before
L80s, or in the whole block if not reached), f20 / f100 / f500, last500, 50-line bins; probes with learning off after the history and
at the end (30 fresh items per rule); bits of the life (as t23/t28). Each block records its pairs and how often the life used them before.
Nets: net=13 (net13.Tissue; default rc=64 = net12_rc), net=12 (net12.Tissue, give lagro=1 relay=1), net=10 (K8), net=store.
Usage: python3 t29_learnset2.py net=13|12|10|store life=A|B|RA|RB|fresh [test=X1] [seed=1 BL=1500 struct=K8 ...net options] [out=FILE]
       python3 t29_learnset2.py list=1"""
import sys, json, time, numpy as np
from collections import defaultdict
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=", 1) for x in sys.argv[1:])}
netn, sname, seed, life, test = str(a.pop("net", 13)), a.pop("struct", "K8"), a.pop("seed", 1), str(a.pop("life", "A")), a.pop("test", None)
BL, NPR, out, LIST = a.pop("BL", 1500), a.pop("NPR", 30), a.pop("out", None), a.pop("list", 0)
STRUCT = dict(K8=dict(L=16, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, c_th_on=0.9, c_th_match=0.75))

# ---- rules (fixed; independent of seed and net) ----
SYM = [int(c) for c in np.random.default_rng(7).permutation([c for c in range(33, 127) if chr(c) not in "=~+"])][:40]   # as t23/t28
OFF = dict(A=(0, 1), B=(2, 3))
def mk(half, bits): return dict(name=f"{half}{''.join(map(str, bits))}", half=half, f=tuple((i + OFF[half][b]) % 4 for i, b in enumerate(bits)))
fr = np.random.default_rng(29)
cand = [tuple(int(x) for x in f"{p:04b}") for p in range(1, 8)]                     # b0 = 0, not 0000: 7 patterns, one per complementary pair
pick = [cand[int(i)] for i in fr.choice(7, 4, replace=False)]
HB = [p for q in pick for p in (q, tuple(1 - x for x in q))]
HB = [HB[int(i)] for i in fr.permutation(8)]                                          # the 8 history bit patterns, in life order
TEST = dict(X1=mk("A", (0, 0, 0, 0)), X2=mk("A", (1, 1, 1, 1)), Y1=mk("B", (0, 0, 0, 0)), Y2=mk("B", (1, 1, 1, 1)))
for k, R_ in TEST.items(): R_["tname"] = k
FREE = [c for c in range(33, 127) if c not in SYM and c != 61]
DLS = [int(x) for x in fr.choice(FREE, 12, replace=False)]
DH, DT = DLS[:8], dict(zip(("X1", "X2", "Y1", "Y2"), DLS[8:]))
ORDER = dict(A=("Y1", "X1", "Y2", "X2"), B=("X1", "Y1", "X2", "Y2"))
def pairs(R_): return [(i, 4 + i - j) for i, j in enumerate(R_["f"])]
def build(life):
    L = []
    if life in ("A", "B"): L += [dict(mk(life, b), kind="H", slot=k + 1, delim=DH[k]) for k, b in enumerate(HB)]
    if life in ("RA", "RB"): L += [dict(name="rand", half=None, f=None, kind="R", slot=k + 1, delim=DH[k]) for k in range(8)]
    if life == "fresh": order = (test,)
    else: order = ORDER[life[-1]]
    for t in order:
        R_ = dict(TEST[t], kind="T", slot=len(L) + 1, delim=DT[t])
        if life in ("A", "B"): R_["cond"] = "fam" if R_["half"] == life else "new"
        else: R_["cond"] = {"RA": "rnd", "RB": "rnd", "fresh": "fresh"}[life]
        L.append(R_)
    used = defaultdict(int)                                                            # prior uses of each pair within this life
    for R_ in L:
        if R_["f"] is None: continue
        R_["pairs"] = pairs(R_); R_["prior"] = [used[p] for p in R_["pairs"]]
        for p in R_["pairs"]: used[p] += 1
        if R_["kind"] == "T": assert (min(R_["prior"]) > 0) if R_["cond"] == "fam" else (max(R_["prior"]) == 0), R_
    return L
def answer(R_, w, r):
    if R_["f"] is None: return [SYM[int(r.integers(40))] for _ in range(4)]
    return [w[j] for j in R_["f"]]
if LIST:
    for lf in ("A", "B", "RA", "RB"):
        print(f"life {lf}:")
        for R_ in build(lf):
            src = "random" if R_["f"] is None else "".join(f"w{j}" for j in R_["f"])
            print(f"  {R_['slot']:2d} {R_['kind']} {R_.get('tname', '')} {R_['name']} W={src} delim {chr(R_['delim'])} {R_.get('cond', '')}"
                  f" pairs(i,lag) {R_.get('pairs', '')} prior uses {R_.get('prior', '')}")
    sys.exit()

LIFE = build(life)
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
    net = importlib.import_module("net" + netn).Tissue(seed=seed, n_values=128, **opts)
def probs():
    O = net.probs() if netn == "store" else net._osum()
    O = np.maximum(O, 0); return O / O.sum() if O.sum() > 0 else np.ones(128) / 128
def guess(): return int(np.argmax(probs())) if netn == "store" else net.answer()
def word(r): return [SYM[int(r.integers(40))] for _ in range(4)]

BITS, NB = defaultdict(float), defaultdict(int); t0 = time.time()
def read(b, kind):
    p = probs(); g = guess(); pb = 0.99 * p[b] + 0.01 / 128
    BITS[kind] += -np.log2(pb); NB[kind] += 1; net.step(b, True); return g == b
def probe(tag, rules):
    r = {}
    for R_ in rules:
        if R_["f"] is None: continue
        ok = n = 0
        for _ in range(NPR):
            w = word(prng)
            for b in w + [R_["delim"]]: net.step(b, False)
            for b in answer(R_, w, prng): ok += guess() == b; n += 1; net.step(b, False)
            net.step(10, False)
        r[R_["slot"]] = round(ok / n, 3)
    print(f"probe {tag}: " + json.dumps(r), flush=True); return r
def crit(ok, nb, th, win=100):
    co, cn = np.cumsum(ok), np.cumsum(nb)
    for k in range(win - 1, len(ok)):
        s = (co[k] - (co[k - win] if k >= win else 0)) / max(cn[k] - (cn[k - win] if k >= win else 0), 1)
        if s >= th: return k + 1
    return None
seen = set(); blocks = []
def block(R_, n):
    ok, nb = np.zeros(n), np.zeros(n)
    for k in range(n):
        w = word(rng); first = (R_["slot"], tuple(w)) not in seen; seen.add((R_["slot"], tuple(w)))
        for b in w + [R_["delim"]]: read(b, "input")
        for b in answer(R_, w, rng):
            g = read(b, "answer")
            if first: ok[k] += g; nb[k] += 1
        read(10, "input")
    sh = lambda lo, hi: round(float(ok[lo:hi].sum() / max(nb[lo:hi].sum(), 1)), 3)
    d = dict(slot=R_["slot"], name=R_["name"], kind=R_["kind"], test=R_.get("tname"), cond=R_.get("cond"), pairs=R_.get("pairs"),
             prior=R_.get("prior"), n=n, first_met=int(nb.sum()), L80=crit(ok, nb, 0.8), L50=crit(ok, nb, 0.5), L80s=crit(ok, nb, 0.8, 20),
             err80=int((nb - ok)[:crit(ok, nb, 0.8, 20) or n].sum()), f20=sh(0, 20),
             f100=sh(0, 100), f500=sh(0, 500), last500=sh(n - 500, n), all=sh(0, n), bins=[sh(i, i + 50) for i in range(0, n, 50)])
    blocks.append(d)
    print(f"slot {R_['slot']} {R_['kind']} {R_.get('tname') or ''} {R_['name']} {R_.get('cond') or ''} ({time.time() - t0:.0f}s): L80 {d['L80']}"
          f" L50 {d['L50']} f20 {d['f20']} f100 {d['f100']} f500 {d['f500']} last500 {d['last500']}", flush=True)
prh = None
for R_ in LIFE:
    if R_["kind"] == "T" and prh is None and life != "fresh": prh = probe("after_history", [x for x in LIFE if x["kind"] == "H"])
    block(R_, BL)
prend = probe("end", LIFE)
res = dict(net=netn, struct=sname if netn != "store" else None, life=life, test=test, seed=seed, opts=a, BL=BL, s=round(time.time() - t0),
           bits=dict(total=round(sum(BITS.values())), **{k: round(v) for k, v in BITS.items()}, per_byte={k: round(BITS[k] / NB[k], 3) for k in BITS}),
           blocks=blocks, probe_after_history=prh, probe_end=prend)
print(json.dumps(res), flush=True)
if out: open(out, "a").write(json.dumps(res) + "\n")
