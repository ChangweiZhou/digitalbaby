"""Learning addition from yes or no (reviewer's task of about 15:35 UTC, after Mr. Xu's words of 15:09-15:12 UTC). My own code, written
from the description; readings in RESULTS.md section 19.
Text: lines "a+b=c\\n", a, b in 0..9; 20 of the 100 pairs never shown (fixed per seed: default_rng(seed)); the shown lines in random order
(default_rng([seed, g])). Symbols '0'-'9', '+', '=', '\\n' (13 byte values). One answering cell (outpart.py's single cell: it names its
best value, with chance 1/16 while learning a seen value at random; told yes or no; rule Q: W[e] += eta (ok - a_e) U / n, eta 1/16).
Score every `every` lines, learning frozen, no random choice: for each of the 100 pairs, the line fed byte by byte after an end of line;
the line is right if every byte after '=' (the answer digits and the end of line) is named. Shown and never-shown pairs apart.
What the answering cell reads (arch):
  A0  the present byte (one-hot) and its slow copies s_2, s_4, s_8 (scaled by sqrt(2k - 1)) -- the reviewer's baseline
  A1  the present byte and delay cells d_1..d_5 (each passes the byte it held to the next at the next byte: the byte 1..5 back)
  A2  A1 and P conjunction cells: each, at birth, gets two (delay, byte value) entries at random (delays 0..5), sensitivity 1 each, and a
      threshold 2: it fires only when both are present (a cell answering to two bytes together); fixed after birth
  A5  A1 and the holding cells A, B read by 100 pair cells (one for each A = x and B = y together), in the three position sets
      (300 cells): a table of pairs
  A7  A1, the holding cells A, B, and P cells with ADDITIVE but unordered random sensitivities to A and B (uA, uB in [0, 1)) and a
      random threshold in [0, 2), by position cue (3P cells): a control for the order of the magnitude code
  A8  A7 with a three-factor rule in the additive cells: at an answer position, after the yes or no, every additive cell that fired
      moves its sensitivity to the held A and B by +r8 on yes, -r8 on no (clipped to [0, 1]); r8 = 1/64. The cell needs: whether it
      fired, which A and B it holds, and the common yes or no.
  A3  A1 and a magnitude path: two holding cells, A takes the digit value present one byte before '+', B the one before '=' (each holds
      it until replaced: a cell gated by a byte); counting cells: each receives from A and B through 9 + 9 fixed 'at least k'
      sensitivities (cell k of A's code is on if A >= k), so its drive is A + B, and it fires if the drive >= its threshold s
      (s = 0..18) AND its position cue is present (three sets: present '=', byte 1 back '=', byte 2 back '='); 57 cells
    python3 addsys.py <label> seed=.. gen=.. arch=A0|A1|A2|A3 lines=.. [P=2000] [every=2000]
"""
import json, sys, time
import numpy as np

lab = sys.argv[1]; kw = dict(a.split("=") for a in sys.argv[2:]); t0 = time.time()
seed = int(kw["seed"]); g = int(kw.get("gen", 0)); arch = kw["arch"]; NL = int(kw["lines"]); P = int(kw.get("P", 2000)); every = int(kw.get("every", 2000))
eta = 1 / 16; eps = 1 / 16; R8 = float(eval(kw.get("r8", "1/64")))
SYM = {**{str(i): i for i in range(10)}, "+": 10, "=": 11, "\n": 12}; NS = 13
pairs = [(a, b) for a in range(10) for b in range(10)]
held = set(map(tuple, np.array(pairs)[np.random.default_rng(seed).choice(100, 20, replace=False)].tolist()))
shown = [p for p in pairs if p not in held]
rng = np.random.default_rng([seed, g])
line = lambda a, b: [SYM[c] for c in f"{a}+{b}={a + b}\n"]
D = 5
if arch == "A2":
    CJ = np.array([[rng.integers(0, D + 1), rng.integers(0, NS), rng.integers(0, D + 1), rng.integers(0, NS)] for _ in range(P)])
    same = CJ[:, 0] == CJ[:, 2]
    CJ[same, 2] = (CJ[same, 2] + 1) % (D + 1)                       # two different delays


if arch in ("A7", "A8"):
    UA = rng.random((P, 10)); UB = rng.random((P, 10)); TH7 = 2 * rng.random(P)


class State:
    def __init__(s):
        s.d = [SYM["\n"]] * (D + 1)                                   # d[0] present, d[k] k back
        s.c = {k: np.zeros(NS) for k in (2, 4, 8)}; s.A = 0; s.B = 0

    def step(s, x):
        s.d = [x] + s.d[:-1]
        oh = np.zeros(NS); oh[x] = 1.0
        for k in s.c: s.c[k] += (oh - s.c[k]) / k
        if x == SYM["+"] and s.d[1] < 10: s.A = s.d[1]
        if x == SYM["="] and s.d[1] < 10: s.B = s.d[1]

    def features(s):
        oh = np.zeros(NS); oh[s.d[0]] = 1.0
        f = [oh]
        if arch == "A0":
            f += [np.sqrt(2 * k - 1) * s.c[k] for k in (2, 4, 8)]
        else:
            for k in range(1, D + 1):
                v = np.zeros(NS); v[s.d[k]] = 1.0; f.append(v)
        if arch == "A2":
            dv = np.array(s.d)
            f.append(((dv[CJ[:, 0]] == CJ[:, 1]) & (dv[CJ[:, 2]] == CJ[:, 3])).astype(float))
        if arch == "A5":                                              # 100 pair cells on the holding cells (A = x and B = y), per position cue
            pc = np.zeros(100); pc[10 * s.A + s.B] = 1.0
            for pos in (0, 1, 2):
                f.append(pc * float(s.d[pos] == SYM["="]))
        if arch in ("A7", "A8"):                                              # P additive but unordered cells: random uA, uB in [0, 1), threshold
            dr = UA[:, s.A] + UB[:, s.B]                                # random in [0, 2); fire if drive >= threshold, by position cue
            fc = (dr >= TH7).astype(float)
            for pos in (0, 1, 2):
                f.append(fc * float(s.d[pos] == SYM["="]))
        if arch == "A3":
            tot = s.A + s.B; cnt = (tot >= np.arange(19)).astype(float)
            for pos in (0, 1, 2):
                f.append(cnt * float(s.d[pos] == SYM["="]))
        f.append(np.ones(1))
        return np.concatenate(f)


st = State(); F = len(st.features()); W = np.zeros((NS, F)); seen = np.zeros(NS, bool)


def name(u): return int((W @ u).argmax())


def evaluate():
    res = {}
    for a, b in pairs:
        s = State(); s.step(SYM["\n"]); L = line(a, b); ok = True
        for i, x in enumerate(L[:-1]):
            s.step(x)
            if i >= L.index(SYM["="]):
                global st
                old, st = st, s
                ok &= name(s.features()) == L[i + 1]
                st = old
        res[(a, b)] = ok
    return float(np.mean([res[p] for p in shown])), float(np.mean([res[p] for p in held]))


curve = []; right_train = []; nb = 0
for li in range(1, NL + 1):
    a, b = shown[rng.integers(0, len(shown))]
    for x, nxt in zip(line(a, b), line(a, b)[1:] + [None]):
        st.step(x); seen[x] = True
        if nxt is None: break
        u = st.features(); A = W @ u; e = int(A.argmax())
        if rng.random() < eps: sv = np.flatnonzero(seen); e = int(sv[rng.integers(0, len(sv))])
        ok = e == nxt; W[e] += eta * (float(ok) - A[e]) / float(u @ u) * u; nb += 1
        if arch == "A8" and st.d[0] < 10 and SYM["="] in st.d[:3] or arch == "A8" and st.d[0] == SYM["="]:
            fire = (UA[:, st.A] + UB[:, st.B]) >= TH7
            dlt = R8 if ok else -R8
            UA[fire, st.A] = np.clip(UA[fire, st.A] + dlt, 0, 1); UB[fire, st.B] = np.clip(UB[fire, st.B] + dlt, 0, 1)
    if li % every == 0:
        s_, h_ = evaluate(); curve.append(dict(lines=li, shown=s_, never_shown=h_))
out = dict(label=lab, args=kw, features=F, cells=dict(A0=0, A1=D, A2=D + P, A3=D + 2 + 18 + 57, A5=D + 2 + 300, A7=D + 2 + 3 * P, A8=D + 2 + 3 * P).get(arch), held=sorted(held), curve=curve,
           max_row_norm=float(np.linalg.norm(W, axis=1).max()), seconds=time.time() - t0)
import os; os.makedirs("/home/user/tissue_check/out/add", exist_ok=True)
json.dump(out, open(f"/home/user/tissue_check/out/add/{lab}.json", "w"), indent=0)
print(lab, "shown / never shown at", " ".join(f"{c['lines']}: {100*c['shown']:.0f}/{100*c['never_shown']:.0f}" for c in curve[::max(1, len(curve) // 6)] + curve[-1:]), f"| {time.time() - t0:.0f} s", flush=True)
