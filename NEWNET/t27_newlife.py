"""A new life for rule cells (23 November 2026): the t23 measurement design on rules and delimiters never used in any design.
Lines "w D W\n", w = 3 symbols of a 40-symbol alphabet (printable, not ": | # % @"; a different draw from t23's; it starts with 8 and
gains one every 150 lines). Five rules:
  copy: W = w0 w1 w2;  swap: W = w1 w0 w2;  rot: W = w1 w2 w0;  skip: W = w0 w2 w0;
  map: W = each symbol's partner in a fixed random pairing of the 40 (20 pairs; a mapping to be learned, different from succ).
cue=1: each rule has its own delimiter (copy ":", swap "|", rot "#", skip "%", map "@"); cue=0: all use ":".
Phases (one life, ORDER below, 3,000 lines a rule block): rot, map, text (TXT bytes of tinyshakespeare from byte 300,000), swap, skip,
rot, map, copy, swap - rot, map and swap return after others.
Scores as t23_life_rules.py (all prequential, learning on, the life's own answers):
  - E3: answer bytes of items met for the first time under their rule, by layer (a: the stretch from the source symbol to the byte
    before has been read before followed by this byte (not for map); n: the symbol has been an answer 1-4 times; c: never an answer
    before; b: the rest), per phase;
  - savings: right share on first-met answer bytes in windows of 250 lines from the start of each phase;
  - E2: every PROBE lines, probes with learning off: 60 fresh items of each rule (teacher-forced right share of the answer bytes;
    with cue=0 all use ":") and 1,000 never-read bytes of the text (from byte 600,000) (right share);
  - total bits of the life (all bytes read; p = max(O, 0) normalised, mixed 0.99/0.01 with uniform) and its parts.
Baseline net=store: the back-off n-gram table (orders 1-8). net=12 imports net12_rc when an rc= option is given (rule cells), else net12.
Usage: python3 t27_newlife.py net=10|12|store [struct=K8 cue=0|1 seed ...net options] [out=FILE]"""
import sys, json, time, numpy as np
import resume
from collections import defaultdict
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=", 1) for x in sys.argv[1:])}
netn, sname, seed, cue = a.pop("net", 10), a.pop("struct", "K8"), a.pop("seed", 1), a.pop("cue", 1)
NB_, TXT, TXT0 = a.pop("NB", 3000), a.pop("TXT", 40000), a.pop("TXT0", 300000)
PROBE, out = a.pop("PROBE", 1000), a.pop("out", None)
ORDER = ["rot", "map", "text", "swap", "skip", "rot", "map", "copy", "swap"]
STRUCT = dict(K8=dict(L=16, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, c_th_on=0.9, c_th_match=0.75))
rng, prng = np.random.default_rng(seed + 4000), np.random.default_rng(seed + 6000)
SYM = [int(c) for c in np.random.default_rng(27).permutation([c for c in range(33, 127) if chr(c) not in ":|#%@"])][:40]
_p = np.random.default_rng(127).permutation(40)
MAP = {}
for i in range(0, 40, 2): MAP[SYM[_p[i]]] = SYM[_p[i + 1]]; MAP[SYM[_p[i + 1]]] = SYM[_p[i]]
RULES = ("copy", "swap", "rot", "skip", "map")
SRC = dict(copy=(0, 1, 2), swap=(1, 0, 2), rot=(1, 2, 0), skip=(0, 2, 0))           # answer byte i = w[SRC[i]]
DEL = {r: (ord(c) if cue else 58) for r, c in zip(RULES, ":|#%@")}
SH = np.frombuffer(open("/home/user/digitalbaby/PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8).astype(int)
SH_L, SH_T = SH[TXT0:TXT0 + TXT], SH[600000:601001]

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
    mod = "net12_rc" if netn == 12 and ("rc" in a or "ag" in a) else "net%d" % netn
    opts = dict(STRUCT[sname]); opts.update(a); net = importlib.import_module(mod).Tissue(seed=seed, n_values=128, **opts)
def probs():
    O = net.probs() if netn == "store" else net._osum()
    O = np.maximum(O, 0); return O / O.sum() if O.sum() > 0 else np.ones(128) / 128
def guess(): return int(np.argmax(probs())) if netn == "store" else net.answer()

def item(rule, n_al, r):
    al = SYM[:n_al]; w = [al[int(r.integers(n_al))] for _ in range(3)]
    if r is rng and r.random() < 0.3: w[int(r.integers(3))] = al[-1]
    W = [MAP[x] for x in w] if rule == "map" else [w[k] for k in SRC[rule]]
    return w, W

seen, answered, past, hist = set(), defaultdict(int), set(), []
def _z3(): return [0, 0, 0.0]
def _z2(): return [0, 0]
R = defaultdict(_z3); WIN = defaultdict(_z2); BITS = defaultdict(float); NB = defaultdict(int)   # module-level factories: picklable (resume.py)
probes = []; t0 = time.time()
def read(b, kind, learn=True):
    p = probs(); g = guess(); pb = 0.99 * p[b] + 0.01 / 128
    if learn: BITS[kind] += -np.log2(pb); NB[kind] += 1
    hist.append(b)
    for d in range(1, 8):
        if len(hist) > d: past.add(tuple(hist[-d - 1:]))
    net.step(b, learn); return g == b, pb
def probe(ln, n_al):
    r = {}
    for rule in RULES:
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

phases = [(r, TXT if r == "text" else NB_) for r in ORDER]
ln = 0; PH0 = K0 = 0
STATE = ("net", "rng", "prng", "seen", "answered", "past", "hist", "R", "WIN", "BITS", "NB", "probes", "ln")
_s = resume.load()                                                                 # a state saved before a container restart
if _s: globals().update({k: _s[k] for k in STATE}); PH0, K0 = _s["ph"], _s["k"]; t0 = time.time() - _s["elapsed"]
def _save(ph, k, force=False): resume.save(dict({n: globals()[n] for n in STATE}, ph=ph, k=k, elapsed=time.time() - t0), f"phase {ph} step {k}", force)
for ph, (rule, n) in enumerate(phases):
    if ph < PH0: continue
    k0 = K0 if ph == PH0 else 0
    if rule == "text":
        for t in range(k0, n):
            read(int(SH_L[t]), "text")
            if (t + 1) % (PROBE * 8) == 0: probe(ln + (t + 1) // 8, min(40, 8 + ln // 150))
            _save(ph, t + 1)
        ln += n // 8; _save(ph + 1, 0, True); continue                                                     # the text counts as n/8 lines (for the probes' clock)
    for k in range(k0, n):
        n_al = min(40, 8 + ln // 150); w, W = item(rule, n_al, rng)
        first = (rule, tuple(w)) not in seen; seen.add((rule, tuple(w)))
        for b in w + [DEL[rule]]: read(b, "input")
        for i, b in enumerate(W):
            if first:
                stretch = None
                if rule != "map":
                    d = 4 + i - SRC[rule][i]                                       # how far back the source symbol is
                    stretch = tuple(hist[-d:]) + (b,)
                lay = "a" if stretch is not None and stretch in past else "c" if not answered[b] else "n" if answered[b] < 5 else "b"
                ok, pb = read(b, "answer")
                rr = R[(ph, rule, lay)]; rr[0] += 1; rr[1] += ok; rr[2] += -np.log2(pb)
                ww = WIN[(ph, k // 250)]; ww[0] += 1; ww[1] += ok
            else: read(b, "answer")
            answered[b] += 1
        read(10, "input"); ln += 1
        if ln % PROBE == 0: probe(ln, n_al)
        _save(ph, k + 1)
    print(f"phase {ph} {rule} done at line {ln}, {time.time() - t0:.0f}s: " + " ".join(f"{l}:{v[1] / max(v[0], 1):.2f}({v[0]})" for (p_, r_, l), v in sorted(R.items()) if p_ == ph), flush=True)
    _save(ph + 1, 0, True)
res = dict(net=netn, struct=sname if netn != "store" else None, cue=cue, seed=seed, opts=a, phases=[f"{r}:{n}" for r, n in phases], s=round(time.time() - t0),
           bits=dict(total=round(sum(BITS.values())), **{k: round(v) for k, v in BITS.items()}, per_byte={k: round(BITS[k] / NB[k], 3) for k in BITS}),
           first_met={f"P{p_}_{r_}_{l}": dict(n=v[0], right=round(v[1] / max(v[0], 1), 3), bits=round(v[2] / max(v[0], 1), 2)) for (p_, r_, l), v in sorted(R.items())},
           windows={f"P{p_}": [round(WIN[(p_, k)][1] / max(WIN[(p_, k)][0], 1), 2) for k in range(24) if WIN[(p_, k)][0]] for p_ in range(len(phases)) if phases[p_][0] != "text"},
           probes=probes, nrecruit=getattr(net, "nrecruit", None))
print(json.dumps(res), flush=True)
resume.done()
if out: open(out, "a").write(json.dumps(res) + "\n")
