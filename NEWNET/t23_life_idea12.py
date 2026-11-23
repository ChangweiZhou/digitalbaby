"""t23_life_rules.py with an episode store and replay (idea 12, 23 Nov 2026); net = net12_rep. Extra options:
  rho=R (one stored episode replayed, learning on, every 1/R lines; text: 8 bytes count as a line), tgt=0|1 (random / drawn in
  proportion to the episode's last error^2), plus net12_rep's slow slowB sk ftau. Store: 500 episodes, reservoir; an episode is a rule
  line (8 bytes; error = mean 1 - p over its 3 answer bytes) or 40 text bytes (error over bytes 8..39). Replayed bytes are not scored.
The original docstring follows.
E2 and E3 in one life (22 November 2026): several rules and a text in one stream, nothing reset.
Lines "w D W\n", w = 3 symbols of a 40-symbol alphabet (printable, not "= ~ +"; it starts with 8 and gains one every 150 lines):
  copy: W = w;  rev: W = w reversed;  succ: W = each symbol's successor in a fixed cyclic order of the 40 (a mapping to be learned).
cue=1: each rule has its own delimiter (copy "=", rev "~", succ "+"); cue=0: all use "=" (the rule must be found from the answers).
Phases (one life): P0 copy (N0 lines), P1 rev (N1), P2 text (TXT bytes of tinyshakespeare from its start), P3 succ (N3), P4 copy (N4),
P5 rev (N5) - each earlier rule returns after others.
Scores (all prequential: every byte predicted before it is read, learning on; the life's own answers):
  - E3: answer bytes of items met for the first time under their rule, by layer (a: storage could know it - the stretch from the
    repeated symbol to the byte before has been read before followed by this byte; n: the symbol has been an answer 1-4 times;
    c: never an answer before; b: the rest), per phase;
  - savings: right share on first-met answer bytes in windows of 250 lines from the start of each phase (P4 against P0, P5 against P1);
  - E2: every PROBE lines, probes with learning off: 60 fresh items of each rule (teacher-forced right share of the answer bytes;
    with cue=0 all use "=") and 1,000 never-read bytes of the text (from byte 600,000) (right share);
  - one number for both: the bits of the whole life (all bytes read; p = max(O, 0) normalised, mixed 0.99/0.01 with uniform), and
    its parts (answer bytes, text bytes).
Baseline net=store: the back-off n-gram table of t22 (orders 1-8).
Usage: python3 t23_life_rules.py net=10|12|store [struct=K8|N1 cue=0|1 seed ...net options] [out=FILE]"""
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
PROBE, out = a.pop("PROBE", 1000), a.pop("out", None)
rho, tgt = a.pop("rho", 0.0), a.pop("tgt", 0); netn = 12
ST, SE, SI = [], [], []; rsr = np.random.default_rng(seed + 9000); nseen = [0]; nrep = defaultdict(int); tick = [0]
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
    opts = dict(STRUCT[sname]); opts.update(a); net = importlib.import_module("net12_rep").Tissue(seed=seed, n_values=128, **opts)
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
def p_of(b):
    return 0.99 * probs()[b] + 0.01 / 128
def keep(ep, e, kind):
    """offer an episode to the store (reservoir of 500)"""
    nseen[0] += 1
    if len(ST) < 500: ST.append(ep); SE.append(e); SI.append(kind)
    elif rsr.random() < 500 / nseen[0]: k = int(rsr.integers(500)); ST[k] = ep; SE[k] = e; SI[k] = kind
def clock():
    """one line's tick: replay one stored episode every 1/rho ticks"""
    if not rho: return
    tick[0] += 1
    if tick[0] % int(round(1 / rho)) or not ST: return
    if tgt: w = np.array(SE) ** 2 + 1e-4; k = int(rsr.choice(len(ST), p=w / w.sum()))
    else: k = int(rsr.integers(len(ST)))
    ep, sc = ST[k], (range(4, 7) if SI[k] != "text" else range(8, 40)); e = 0.0; net.replaying = True
    for i, b in enumerate(ep):
        if i in sc: e += 1 - p_of(b)
        net.step(b, True)
    net.replaying = False; SE[k] = e / len(sc); nrep[SI[k]] += 1
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

phases = [("copy", N0), ("rev", N1), ("text", TXT), ("succ", N3), ("copy", N4), ("rev", N5)]
ln = 0
for ph, (rule, n) in enumerate(phases):
    if rule == "text":
        te = 0.0
        for t in range(n):
            if t % 8 == 0: clock()
            if t % 40 >= 8: te += 1 - p_of(int(SH[t]))
            read(int(SH[t]), "text")
            if rho and t % 40 == 39: keep([int(x) for x in SH[t - 39:t + 1]], te / 32, "text"); te = 0.0
            if (t + 1) % (PROBE * 8) == 0: probe(ln + (t + 1) // 8, min(40, 8 + ln // 150))
        ln += n // 8; continue                                                     # the text counts as n/8 lines (for the probes' clock)
    for k in range(n):
        n_al = min(40, 8 + ln // 150); w, W = item(rule, n_al, rng)
        first = (rule, tuple(w)) not in seen; seen.add((rule, tuple(w))); clock(); le = 0.0
        for b in w + [DEL[rule]]: read(b, "input")
        for i, b in enumerate(W):
            if rho: le += 1 - p_of(b)
            if first:
                d = 4 if rule != "rev" else 2 + 2 * i                              # how far back the symbol it comes from is (rev: 2, 4, 6)
                stretch = tuple(hist[-d:]) + (b,) if rule != "succ" else None
                lay = "a" if stretch is not None and stretch in past else "c" if not answered[b] else "n" if answered[b] < 5 else "b"
                ok, pb = read(b, "answer")
                rr = R[(ph, rule, lay)]; rr[0] += 1; rr[1] += ok; rr[2] += -np.log2(pb)
                ww = WIN[(ph, k // 250)]; ww[0] += 1; ww[1] += ok
            else: read(b, "answer")
            answered[b] += 1
        read(10, "input"); ln += 1
        if rho: keep(w + [DEL[rule]] + W + [10], le / 3, rule)
        if ln % PROBE == 0: probe(ln, n_al)
    print(f"replayed in phase {ph}: {dict(nrep)}", flush=True); nrep.clear()
    print(f"phase {ph} {rule} done at line {ln}, {time.time() - t0:.0f}s: " + " ".join(f"{l}:{v[1] / max(v[0], 1):.2f}({v[0]})" for (p_, r_, l), v in sorted(R.items()) if p_ == ph), flush=True)
res = dict(script="t23_life_idea12", rho=rho, tgt=tgt, net=netn, struct=sname if netn != "store" else None, cue=cue, seed=seed, opts=a, phases=[f"{r}:{n}" for r, n in phases], s=round(time.time() - t0),
           bits=dict(total=round(sum(BITS.values())), **{k: round(v) for k, v in BITS.items()}, per_byte={k: round(BITS[k] / NB[k], 3) for k in BITS}),
           first_met={f"P{p_}_{r_}_{l}": dict(n=v[0], right=round(v[1] / max(v[0], 1), 3), bits=round(v[2] / max(v[0], 1), 2)) for (p_, r_, l), v in sorted(R.items())},
           windows={f"P{p_}": [round(WIN[(p_, k)][1] / max(WIN[(p_, k)][0], 1), 2) for k in range(24) if WIN[(p_, k)][0]] for p_ in range(len(phases)) if phases[p_][0] != "text"},
           probes=probes)
print(json.dumps(res), flush=True)
if out: open(out, "a").write(json.dumps(res) + "\n")
