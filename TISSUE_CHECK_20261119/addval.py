"""Addition with the byte's VALUE as input (reviewer's task of 15:25 UTC; Mr. Xu: a cell takes the byte itself, a level). My own code.
Text: lines "a+b=c\\n", a, b in 0..maxn; `held` of the pairs never shown (fixed per seed); order and the cells' random choices from
default_rng([seed, g]). One answering cell (rule Q, eta 1/16, eps 1/16 while learning) reads: the present byte and delay cells d_1..d_D
(one-hot each: the line's form), a constant, and the cells of the system chosen by `arch`:
  V3  (step 1) two holding cells, A takes the byte VALUE present one byte before '+', B the one before '=' (gated by a byte); counting
      cells with drive A + B (byte values, so '7' counts 55) and a threshold at every level 0..510 (nothing about digits laid down),
      in three position sets (present '=', 1 back '=', 2 back '='): 3 x 511 cells
  P   (step 2) a pool of N cells, nothing laid down about holding or place: each cell adds the VALUES of the bytes at two random delays
      i < j (0..D), is tuned to one total theta (it fires when its drive equals theta; theta is its own drive at the byte it was made, a
      level it has seen) and is gated by one random (delay, byte value) entry (the byte at that delay must equal that value; value
      drawn among the values seen). Local rule (keep / withdraw): every `per` lines the answering cell withdraws its connections from the
      quarter of the pool cells with the weakest connection (norm of its weights from the cell) and those cells make new random choices
      (their connection starts at 0). keep=rate: the strength is weighed by the square root of the cell's firings per line since it
      was last judged (a sum cell fires for several pairs, a table cell for one). A cell needs its two delayed inputs, its gate, its tuned level; the answering cell needs the
      strength of its own connections.
Score every `every` lines with learning frozen, no random choice: each pair's line fed byte by byte after an end of line; right if
every byte after '=' is named; also per answer place counted from the right (units, tens, hundreds) on the never-shown pairs.
    python3 addval.py <label> seed=.. gen=.. arch=V3|P lines=.. [N=..] [D=5] [per=2000] [maxn=9] [held=20] [every=5000] [neval=..]
"""
import json, os, sys, time
import numpy as np

lab = sys.argv[1]; kw = dict(a.split("=") for a in sys.argv[2:]); t0 = time.time()
seed = int(kw["seed"]); g = int(kw.get("gen", 0)); arch = kw["arch"]; NL = int(kw["lines"]); every = int(kw.get("every", 5000))
N = int(kw.get("N", 0)); D = int(kw.get("D", 5)); per = int(kw.get("per", 2000)); maxn = int(kw.get("maxn", 9)); nheld = int(kw.get("held", 20))
eta = 1 / 16; eps = 1 / 16
pairs = [(a, b) for a in range(maxn + 1) for b in range(maxn + 1)]
hi = np.random.default_rng(seed).choice(len(pairs), nheld, replace=False); held = [pairs[i] for i in sorted(hi)]
hs = set(held); shown = [p for p in pairs if p not in hs]
rng = np.random.default_rng([seed, g])
evr = np.random.default_rng([seed, 99])
neval = int(kw.get("neval", 0))
ev_shown = shown if not neval or neval >= len(shown) else [shown[i] for i in evr.choice(len(shown), neval, replace=False)]
ev_held = held if not neval or neval >= len(held) else [held[i] for i in evr.choice(len(held), neval, replace=False)]
line = lambda a, b: list(f"{a}+{b}={a + b}\n".encode())
EQ, PLUS, NLc = ord("="), ord("+"), ord("\n")
SYMS = sorted(set(b"0123456789+=\n")); SI = {v: k for k, v in enumerate(SYMS)}; NS = len(SYMS)
if arch == "P":
    def new_cell():
        i, j = sorted(rng.choice(D + 1, 2, replace=False)); return [int(i), int(j), int(rng.integers(0, D + 1)), int(SYMS[rng.integers(0, NS)]), None]
    pool = [new_cell() for _ in range(N)]
    PI = np.array([c[0] for c in pool]); PJ = np.array([c[1] for c in pool]); PK = np.array([c[2] for c in pool]); PV = np.array([c[3] for c in pool])
    PT = np.full(N, -1)


class State:
    def __init__(s): s.d = [NLc] * (D + 1); s.A = 0; s.B = 0

    def step(s, x):
        s.d = [x] + s.d[:-1]
        if x == PLUS and chr(s.d[1]).isdigit(): s.A = s.d[1]
        if x == EQ and chr(s.d[1]).isdigit(): s.B = s.d[1]

    def features(s):
        f = []
        for k in range(D + 1):
            v = np.zeros(NS); v[SI[s.d[k]]] = 1.0; f.append(v)
        if arch == "V3":
            cnt = (s.A + s.B >= np.arange(511)).astype(float)
            for pos in (0, 1, 2): f.append(cnt * float(s.d[pos] == EQ))
        if arch == "P":
            dv = np.array(s.d); f.append(((dv[PI] + dv[PJ] == PT) & (dv[PK] == PV)).astype(float))
        f.append(np.ones(1))
        return np.concatenate(f)


st = State(); base = (D + 1) * NS; FIRED = np.zeros(N)
F = len(st.features()); W = np.zeros((NS, F)); seen = np.zeros(NS, bool)
if arch == "P":
    for j, c in enumerate(pool): c[4] = st.d[c[0]] + st.d[c[1]]; PT[j] = c[4]


def evaluate(plist):
    right = []; places = {"units": [], "tens": [], "hundreds": []}
    for a, b in plist:
        s = State(); s.step(NLc); L = line(a, b); q = L.index(EQ); ok = True; ans = L[q + 1:-1]
        for i, x in enumerate(L[:-1]):
            s.step(x)
            if i >= q:
                nm = SYMS[int((W @ s.features()).argmax())]; r = nm == L[i + 1]; ok &= r
                k = i + 1 - (q + 1)                                   # index in the answer (len(ans) = end of line)
                if k < len(ans): places[["units", "tens", "hundreds"][len(ans) - 1 - k]].append(r)
        right.append(ok)
    return float(np.mean(right)), {k: (float(np.mean(v)) if v else None) for k, v in places.items()}


curve = []
for li in range(1, NL + 1):
    a, b = shown[rng.integers(0, len(shown))]; L = line(a, b)
    for x, nxt in zip(L, L[1:] + [None]):
        st.step(x); seen[SI[x]] = True
        if nxt is None: break
        u = st.features(); A = W @ u; e = int(A.argmax())
        if arch == "P": FIRED += u[base:base + N]
        if rng.random() < eps: sv = np.flatnonzero(seen); e = int(sv[rng.integers(0, len(sv))])
        ok = SYMS[e] == nxt; W[e] += eta * (float(ok) - A[e]) / float(u @ u) * u
    if arch == "P" and li % per == 0:
        strength = np.linalg.norm(W[:, base:base + N], axis=0)
        if kw.get("keep") == "rate":                                   # weigh by how often the cell fired since it was last judged
            strength = strength * np.sqrt(FIRED / per); FIRED[:] = 0
        for j in np.argsort(strength, kind="stable")[: N // 4]:
            pool[j] = new_cell(); pool[j][4] = st.d[pool[j][0]] + st.d[pool[j][1]]; W[:, base + j] = 0.0
            PI[j], PJ[j], PK[j], PV[j], PT[j] = pool[j]
    if li % every == 0:
        s_, ps = evaluate(ev_shown); h_, ph = evaluate(ev_held)
        curve.append(dict(lines=li, shown=s_, never_shown=h_, never_shown_places=ph, shown_places=ps))
out = dict(label=lab, args=kw, features=F, cells=dict(V3=D + 2 + 3 * 511, P=D + N).get(arch), held=len(held), curve=curve,
           max_row_norm=float(np.linalg.norm(W, axis=1).max()), seconds=time.time() - t0)
if arch == "P":
    strength = np.linalg.norm(W[:, base:base + N], axis=0); top = np.argsort(-strength)[:12]
    out["strongest_cells"] = [dict(delays=[pool[j][0], pool[j][1]], gate=[pool[j][2], chr(pool[j][3])], theta=pool[j][4], strength=float(strength[j])) for j in top]
os.makedirs("/home/user/tissue_check/out/addv", exist_ok=True)
json.dump(out, open(f"/home/user/tissue_check/out/addv/{lab}.json", "w"), indent=0)
c = curve[-1]
print(lab, f"at {c['lines']}: shown {100*c['shown']:.0f} never shown {100*c['never_shown']:.0f} places {c['never_shown_places']} | {time.time() - t0:.0f} s", flush=True)
