"""Idea 8 (23 November 2026): is the blocked schedule of t23_life_rules.py the E2 problem? The same amounts of each rule and the same
text as t23's life (copy N0+N4 = 9,000 lines, rev N1+N5 = 9,000, succ N3 = 6,000, 40,000 bytes of tinyshakespeare from its start), but
the 24,000 rule lines are in one random order from the start (the three rules interleaved line by line) and the text comes in NB=8
blocks of 5,000 consecutive bytes, block k read after line round((k+0.5)*24000/8) (spread through the life). Same items, alphabet
growth (8 symbols, one more every 150 lines), delimiters (cue), probes (every PROBE lines, learning off: 60 fresh items of each rule,
1,000 never-read text bytes from byte 600,000) and bits as t23_life_rules.py. First-met answer bytes by rule and layer (a/n/c/b, as
t23) and their right share per rule in windows of 1,000 lines.
Usage: python3 t23_mixed.py net=10|12|store [struct=K8|N1 cue=0|1 seed NB ...net options] [out=FILE]"""
import sys, json, time, numpy as np
from collections import defaultdict
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=", 1) for x in sys.argv[1:])}
netn, sname, seed, cue = a.pop("net", 10), a.pop("struct", "K8"), a.pop("seed", 1), a.pop("cue", 1)
N0, N1, TXT, N3, N4, N5 = (a.pop(k, d) for k, d in (("N0", 6000), ("N1", 6000), ("TXT", 40000), ("N3", 6000), ("N4", 3000), ("N5", 3000)))
PROBE, out, NBLK = a.pop("PROBE", 1000), a.pop("out", None), a.pop("NB", 8)
STRUCT = dict(K8=dict(L=16, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, c_th_on=0.9, c_th_match=0.75),
              N1=dict(L=8, cC=64, M=32, joint=1, mesh=2, mesh_base=0, mesh_dag=2, c_k_win=4))
rng, prng = np.random.default_rng(seed + 4000), np.random.default_rng(seed + 6000)
SYM = [int(c) for c in np.random.default_rng(7).permutation([c for c in range(33, 127) if chr(c) not in "=~+"])][:40]
SUCC = {SYM[i]: SYM[(i + 1) % 40] for i in range(40)}
DEL = dict(copy=61, rev=126 if cue else 61, succ=43 if cue else 61)
SH = np.frombuffer(open("/home/user/digitalbaby/PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8).astype(int)
SH_T = SH[600000:601001]

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
    opts = dict(STRUCT[sname]); opts.update(a); net = importlib.import_module("net%d" % netn).Tissue(seed=seed, n_values=128, **opts)
def probs():
    O = net.probs() if netn == "store" else net._osum()
    O = np.maximum(O, 0); return O / O.sum() if O.sum() > 0 else np.ones(128) / 128
def guess(): return int(np.argmax(probs())) if netn == "store" else net.answer()

def item(rule, n_al, r):
    al = SYM[:n_al]; w = [al[int(r.integers(n_al))] for _ in range(3)]
    if r is rng and r.random() < 0.3: w[int(r.integers(3))] = al[-1]
    W = w if rule == "copy" else w[::-1] if rule == "rev" else [SUCC[x] for x in w]
    return w, W

seen, answered, past, hist = set(), defaultdict(int), set(), []
R = defaultdict(lambda: [0, 0, 0.0]); WIN = defaultdict(lambda: [0, 0]); BITS = defaultdict(float); NB = defaultdict(int)
probes = []; t0 = time.time()
def read(b, kind, learn=True, score=None):
    """predict b, then read it; score: None, or (phase, layer) for a first-met answer byte"""
    p = probs(); g = guess(); pb = 0.99 * p[b] + 0.01 / 128
    if learn: BITS[kind] += -np.log2(pb); NB[kind] += 1
    hist.append(b)
    for d in range(1, 8):
        if len(hist) > d: past.add(tuple(hist[-d - 1:]))
    net.step(b, learn); return g == b, pb
def probe(ln, n_al):
    r = {}
    for rule in ("copy", "rev", "succ"):
        ok = n = 0
        for _ in range(60):
            w, W = item(rule, n_al, prng)
            for b in w + [DEL[rule]]: net.step(b, False)
            for b in W:
                ok += guess() == b; n += 1; net.step(b, False)
            net.step(10, False)
        r[rule] = round(ok / n, 3)
    ok = 0
    for t in range(len(SH_T) - 1):
        net.step(int(SH_T[t]), False); ok += guess() == SH_T[t + 1]
    r["text"] = round(ok / (len(SH_T) - 1), 3); net.step(10, False)
    probes.append(dict(line=ln, **r)); print(json.dumps(probes[-1]), flush=True)

srng = np.random.default_rng(seed + 5000)
rules = ["copy"] * (N0 + N4) + ["rev"] * (N1 + N5) + ["succ"] * N3; srng.shuffle(rules); NL = len(rules)
BL = TXT // NBLK; at = {int(round((k + 0.5) * NL / NBLK)): k for k in range(NBLK)}             # line after which block k is read
ln = 0; tpos = 0; txt_done = []
def text_block(k):
    global tpos
    for t in range(BL): read(int(SH[tpos + t]), "text")
    tpos += BL; txt_done.append(ln)
for k, rule in enumerate(rules):
    if k in at: text_block(at[k])
    n_al = min(40, 8 + ln // 150); w, W = item(rule, n_al, rng)
    first = (rule, tuple(w)) not in seen; seen.add((rule, tuple(w)))
    for b in w + [DEL[rule]]: read(b, "input")
    for i, b in enumerate(W):
        if first:
            d = 4 if rule != "rev" else 2 + 2 * i
            stretch = tuple(hist[-d:]) + (b,) if rule != "succ" else None
            lay = "a" if stretch is not None and stretch in past else "c" if not answered[b] else "n" if answered[b] < 5 else "b"
            ok, pb = read(b, "answer")
            rr = R[(rule, lay)]; rr[0] += 1; rr[1] += ok; rr[2] += -np.log2(pb)
            ww = WIN[(rule, k // 1000)]; ww[0] += 1; ww[1] += ok
        else: read(b, "answer")
        answered[b] += 1
    read(10, "input"); ln += 1
    if ln % PROBE == 0: probe(ln, n_al)
    if ln % 6000 == 0: print(f"line {ln}, {time.time() - t0:.0f}s: " + " ".join(f"{r_}_{l}:{v[1] / max(v[0], 1):.2f}({v[0]})" for (r_, l), v in sorted(R.items())), flush=True)
if NL in at: text_block(at[NL])
res = dict(script="t23_mixed", net=netn, struct=sname if netn != "store" else None, cue=cue, seed=seed, opts=a, NB=NBLK, text_after_lines=txt_done,
           lines={r: rules.count(r) for r in ("copy", "rev", "succ")}, s=round(time.time() - t0),
           bits=dict(total=round(sum(BITS.values())), **{k: round(v) for k, v in BITS.items()}, per_byte={k: round(BITS[k] / NB[k], 3) for k in BITS}),
           first_met={f"{r_}_{l}": dict(n=v[0], right=round(v[1] / max(v[0], 1), 3), bits=round(v[2] / max(v[0], 1), 2)) for (r_, l), v in sorted(R.items())},
           windows={r: [round(WIN[(r, k)][1] / max(WIN[(r, k)][0], 1), 2) for k in range(NL // 1000 + 1) if WIN[(r, k)][0]] for r in ("copy", "rev", "succ")},
           probes=probes)
print(json.dumps(res), flush=True)
if out: open(out, "a").write(json.dumps(res) + "\n")
