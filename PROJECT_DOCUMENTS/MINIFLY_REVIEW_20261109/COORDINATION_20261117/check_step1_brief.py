"""Check of the brief TASK_STEP_1_THE_TISSUE_READS_THE_STREAM_20261118.md: it recomputes every number of the brief that can be
computed without running the tissue.

Run from the repository's root:   python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/check_step1_brief.py
Standard library only; it reads no file and writes no file. About one minute.

Part A: the stream (section 4 of the brief): its lengths, digests, byte frequencies and chance levels.
Part B: the birth of the tissue (section 5.3 of the brief) for the three seeds: counts of cells by kind and by distance, digests.
Part C: the worked example of section 13 of revision 4, in three variants (as written; first form with erratum E4; arm K).
Part D: single-step tests of the sign of the two strength rules.
Part E: the countermodel C0 (the present byte alone) in the windows W0 and W3, with the classifier of section 7 of the brief.
Part F: time constants, trace shares and the giving-up delay of a connection from a line.
Part G: the cost arithmetic and the sizes of the files saved.
"""
import hashlib
import json
import math
from random import Random

SEEDS = (73129, 93683, 116089)

# =============================================================================================================== A. the stream
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
END = 10                                   # the end mark: the byte 10 (line feed) in place of the record's symbol 256
KMAX = 12
LENGTH = 262144                            # bytes in the stream of step 1
WIN = 8192                                 # bytes in a recording window
WINDOWS = {"W0": 0, "W1": 32768 - WIN, "W2": 131072 - WIN, "W3": LENGTH - WIN}
LAGS = (0, 1, 2, 4, 8)


def apply(op, m, a, b):
    return {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]


def render(task, a, b, k, rng):           # the record's render (exp11_cell.render), with the end mark as a byte
    op, m, code = task
    d = [rng.choice(DISTRACT) for _ in range(k)]
    prompt = list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")]
    return prompt + list(f"{apply(op, m, a, b):02d}".encode()) + [END]


def split(task, frac, seed):               # the record's split (exp11_cell.split)
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]).shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def items(seed):                           # the record's stream order (exp11_cell.stream), item by item
    rng = Random(seed + 77)
    flat = [(t, pr) for t in DEV for pr in split(t, .5, seed)[0]]
    passes = 0
    while True:
        order = list(range(len(flat))); rng.shuffle(order); passes += 1
        for i in order:
            t, (a, b) = flat[i]
            k = rng.randint(0, KMAX)
            yield render(t, a, b, k, rng), t, a, b, k, passes


def stream(seed, n=LENGTH):
    out, its = [], []
    for byts, t, a, b, k, p in items(seed):
        its.append((t[0], a, b, k, len(out)))
        out.extend(byts)
        if len(out) >= n:
            return out[:n], its, p


print("=" * 110)
print("A. THE STREAM")
for t in DEV:
    tau, rest = split(t, .5, 73129)
    print(f"A.1 task {t[0]} (task byte {t[2][0]}, sign byte {ord(SYM[t[0]])}): pairs {89 * 89}, in the stream {len(tau)}, never in it {len(rest)}")
n_taught = sum(len(split(t, .5, s)[0]) for t in DEV for s in (73129,))
print(f"A.2 pairs in the stream: {n_taught} of {4 * 89 * 89}; never shown: {4 * 89 * 89 - n_taught}")
print(f"A.3 an item has 10 + k bytes, k uniform on 0..12: mean length {10 + KMAX / 2:.1f} bytes; shortest 10, longest 22")
STREAMS = {}
for s in SEEDS:
    st, its, passes = stream(s)
    STREAMS[s] = st
    vals = sorted(set(st))
    ok_pairs = all((a, b) in set(split([t for t in DEV if t[0] == op][0], .5, s)[0]) for op, a, b, k, pos in its[:2000])
    print(f"A.4 seed {s}: SHA-256 of the {LENGTH} bytes {hashlib.sha256(bytes(st)).hexdigest()}")
    print(f"    items begun: {len(its)} (passes through the taught pairs begun: {passes}); distinct byte values: {len(vals)}; "
          f"first 48 bytes: {bytes(st[:48])!r}")
    print(f"    the first 2,000 items are all taught pairs: {ok_pairs}; mean item length in the stream {LENGTH / len(its):.2f} bytes")
print(f"A.5 the byte values of the stream: {bytes(sorted(set(STREAMS[73129])))!r}")
s = 73129
st = STREAMS[s]
for wn, w0 in WINDOWS.items():
    seg = st[w0:w0 + WIN]
    cnt = {}
    for b in seg:
        cnt[b] = cnt.get(b, 0) + 1
    mx = max(cnt.items(), key=lambda kv: kv[1])
    mn = min(cnt.items(), key=lambda kv: kv[1])
    print(f"A.6 seed {s} window {wn} (bytes {w0} to {w0 + WIN - 1}): distinct values {len(cnt)}; balanced-accuracy chance 1/{len(cnt)} = "
          f"{1 / len(cnt):.4f}; most frequent byte {bytes([mx[0]])!r} share {mx[1] / WIN:.4f} (the majority rate); "
          f"rarest {bytes([mn[0]])!r} count {mn[1]}")
cnt = {}
for b in st:
    cnt[b] = cnt.get(b, 0) + 1
print("A.7 shares of the byte values over the whole stream (seed 73129), rarest five and commonest five: "
      + ", ".join(f"{bytes([b])!r} {c / LENGTH:.4f}" for b, c in sorted(cnt.items(), key=lambda kv: kv[1])[:5]) + " ... "
      + ", ".join(f"{bytes([b])!r} {c / LENGTH:.4f}" for b, c in sorted(cnt.items(), key=lambda kv: kv[1])[-5:]))
print(f"A.8 recording windows: " + "; ".join(f"{k} bytes {v} to {v + WIN - 1}" for k, v in WINDOWS.items())
      + f"; steps of the stream {LENGTH * 32:,}")

# =============================================================================================================== B. the birth
N, NFB, NIN, NOUT, R2 = 2048, 64, 256, 256, (1 / 8) ** 2


def birth(seed):
    rng = Random(seed)
    P = [(rng.random(), rng.random()) for _ in range(N)]
    IN_Y = [rng.random() for _ in range(NIN)]
    OUT_Y = [rng.random() for _ in range(NOUT)]
    FB = rng.sample(range(N), NFB)
    YES, NO = FB[:32], FB[32:]
    KU = [rng.random() for _ in range(N)]
    fbset = set(FB)
    taken = set(FB)
    driver = []
    for v in range(NOUT):
        best = min((i for i in range(N) if i not in taken), key=lambda i: ((1 - P[i][0]) ** 2 + (P[i][1] - OUT_Y[v]) ** 2, i))
        driver.append(best); taken.add(best)
    outset = set(driver)
    nb = {}
    for i in range(N):
        if i in fbset:
            continue
        xi, yi = P[i]
        mem = [j for j in range(N) if j != i and (P[j][0] - xi) ** 2 + (P[j][1] - yi) ** 2 <= R2]
        mem += [N + v for v in range(NIN) if xi ** 2 + (yi - IN_Y[v]) ** 2 <= R2]
        nb[i] = mem
    src = {}
    for i in range(N):
        if i in fbset:
            src[i] = []
            continue
        src[i] = rng.sample(nb[i], min(8, len(nb[i])))
    inhib = [(KU[i] < 0.25) and i not in outset and i not in fbset for i in range(N)]
    return dict(P=P, IN_Y=IN_Y, OUT_Y=OUT_Y, FB=FB, YES=YES, NO=NO, driver=driver, nb=nb, src=src, inhib=inhib, fbset=fbset, outset=outset)


def distances(B):
    tg = {i: [] for i in range(N)}
    dist = [None] * N
    for i in range(N):
        for j in B["src"][i]:
            if j < N:
                tg[j].append(i)
            else:
                dist[i] = 1
    frontier = [i for i in range(N) if dist[i] == 1]
    d = 1
    while frontier:
        nxt = []
        for j in frontier:
            for i in tg[j]:
                if dist[i] is None:
                    dist[i] = d + 1; nxt.append(i)
        frontier, d = nxt, d + 1
    return dist


print("=" * 110)
print("B. THE BIRTH (first form, 64 feedback cells with no sources, erratum E4)")
BIRTHS = {}
for s in SEEDS:
    B = birth(s)
    BIRTHS[s] = B
    dist = distances(B)
    enc = [i for i in range(N) if i not in B["fbset"] and any(m >= N for m in B["nb"][i])]
    inner = [i for i in range(N) if i not in B["fbset"] and i not in B["outset"] and i not in set(enc)]
    sizes = [len(B["nb"][i]) for i in B["nb"]]
    cell_sizes = [sum(1 for m in B["nb"][i] if m < N) for i in B["nb"]]
    line_src = sum(1 for i in range(N) for j in B["src"][i] if j >= N)
    enc_draws = sum(len(B["src"][i]) for i in enc)
    hist = {}
    for d in dist:
        hist[d] = hist.get(d, 0) + 1
    xo = min(B["P"][i][0] for i in B["driver"])
    digest = hashlib.sha256(json.dumps([sorted(B["src"][i]) for i in range(N)]).encode()).hexdigest()[:16]
    kdig = hashlib.sha256(json.dumps([B["FB"], B["driver"], B["inhib"]]).encode()).hexdigest()[:16]
    pdig = hashlib.sha256(json.dumps(B["P"]).encode()).hexdigest()[:16]
    print(f"B.1 seed {s}: digest of coordinates {pdig}; of the sources {digest}; of feedback cells, drivers and kinds {kdig}")
    print(f"    cells: output-end {len(B['outset'])}, feedback {len(B['fbset'])} (yes {len(B['YES'])}, no {len(B['NO'])}), encoder {len(enc)}, "
          f"inner {len(inner)}; inhibitory {sum(B['inhib'])}")
    print(f"    neighbourhood members (cells and lines) smallest {min(sizes)}, mean {sum(sizes) / len(sizes):.1f}, largest {max(sizes)}; "
          f"cells only: mean {sum(cell_sizes) / len(cell_sizes):.1f}; cells with fewer than 8 members: {sum(1 for z in sizes if z < 8)}")
    print(f"    starting connections {sum(len(v) for v in B['src'].values())}; from lines {line_src}, share of lines in the encoder cells' draws "
          f"{line_src / enc_draws:.3f}; smallest x of an output-end cell {xo:.4f}")
    print(f"    cells by distance from the input lines (d: count): "
          + ", ".join(f"{d}: {hist[d]}" for d in sorted(k for k in hist if k is not None)) + f"; not reached: {hist.get(None, 0)}")
    print(f"    cells recorded (d = 1, 2, 4): {hist.get(1, 0)} + {hist.get(2, 0)} + {hist.get(4, 0)} = {hist.get(1, 0) + hist.get(2, 0) + hist.get(4, 0)}")
    outd = [dist[i] for i in B["driver"]]
    print(f"    output-end cells at distance {min(d for d in outd if d)} to {max(d for d in outd if d)} from the input lines "
          f"(not reached: {sum(1 for d in outd if d is None)})")
    fbt = {i for i in range(N) for j in B["src"][i] if j in B["fbset"]}
    print(f"    cells taking a feedback cell as a source at birth: {len(fbt)}; of them output-end {len(fbt & B['outset'])} "
          f"(taking a yes-cell {len({i for i in B['outset'] if set(B['src'][i]) & set(B['YES'])})}, a no-cell "
          f"{len({i for i in B['outset'] if set(B['src'][i]) & set(B['NO'])})}), encoder {len(fbt & set(enc))}; connections from feedback cells "
          f"{sum(1 for i in range(N) for j in B['src'][i] if j in B['fbset'])}")

# =============================================================================================================== C. worked example
ETA, TAU_V, TAU, ETA_T, ASTAR = 1 / 8, 2, 32, 1 / 128, 1 / 8


def example(variant):
    """variant 'r4': revision 4 as written. 'E4': first form with erratum E4 (F has its line only, theta 0, formulas 7, 9, 12 off at F).
    'K': arm K (E4; F inhibitory by E3; the coincidence rule at the inner cell I, theta-bar starting at 0)."""
    ext = {"s1": lambda k: 0.1, "h": lambda k: 0.25, "c": lambda k: 0.125, "q": lambda k: 0.25, "g": lambda k: 0.125,
           "u": lambda k: 0.5 if k < 16 else 0.125, "f": lambda k: 0.25,
           "L3": lambda k: 1.0, "L5": lambda k: 0.0, "NO": lambda k: 1.0 if k == 0 else 0.0}
    sign = {"s1": 1, "h": -1, "c": 1, "q": 1, "g": -1, "u": 1, "f": -1, "L3": 1, "L5": 1, "NO": 1, "E": 1, "F": -1 if variant == "K" else 1}
    fixed = {"L3", "L5", "NO"}
    cells = {
        "F": dict(src={"NO": [1.0, 0.0], **({} if variant in ("E4", "K") else {"s1": [0.25, 0.10]})}, v=0.02, p=0.02,
                  th=0.0 if variant in ("E4", "K") else 0.05, xprev=0.0, plastic=variant == "r4"),
        "E": dict(src={"L3": [0.125, 0.02], "L5": [0.125, 0.40], "h": [0.25, 0.25], "c": [0.5, 0.125], "F": [0.25, 0.0]},
                  v=0.04, p=0.05, th=0.02, xprev=0.03, plastic=True),
        "I": dict(src={"E": [0.5, 0.06], "q": [0.25, 0.25], "g": [0.25, 0.125]}, v=0.10, p=0.12, th=0.05, xprev=0.06, plastic=True,
                  rule="K" if variant == "K" else "F", vb=0.0),
        "O": dict(src={"u": [0.75, 0.50], "f": [0.25, 0.25], "F": [0.25, 0.0]}, v=0.70, p=0.3125, th=0.10, xprev=0.60, plastic=True),
    }
    rows = {}
    for k in range(32):
        z = {n: f(k) for n, f in ext.items()}
        z.update({n: c["xprev"] for n, c in cells.items()})
        new = {}
        for n, c in cells.items():                                   # operations 2 to 4
            A = sum(sign[j] * w * z[j] for j, (w, e) in c["src"].items())
            m = A - c["p"]
            v = (1 - 1 / TAU_V) * c["v"] + (1 / TAU_V) * A
            x = min(1.0, max(0.0, v - c["th"]))
            new[n] = (A, m, v, x)
        for n, c in cells.items():                                   # operation 5: formula 7 (or the coincidence rule) and formula 8
            A, m, v, x = new[n]
            if not c["plastic"]:
                continue
            for j, we in c["src"].items():
                if j in fixed:
                    continue
                if c.get("rule") == "K":
                    dw = ETA * we[1] * c["xprev"] * (c["xprev"] - c["vb"])
                else:
                    dw = -ETA * sign[j] * m * c["xprev"] * we[1]
                we[0] = min(1.0, max(0.0, we[0] + dw))
        for n, c in cells.items():                                   # operation 6: traces, p, theta, theta-bar
            A, m, v, x = new[n]
            for j, we in c["src"].items():
                we[1] = (1 - 1 / TAU) * we[1] + (1 / TAU) * z[j]
            c["p"] = (1 - 1 / TAU) * c["p"] + (1 / TAU) * A
            if not (n == "F" and variant in ("E4", "K")):
                c["th"] = max(0.0, c["th"] + ETA_T * (x - ASTAR))
            if c.get("rule") == "K":
                c["vb"] = (1 - 1 / TAU) * c["vb"] + (1 / TAU) * x * x / ASTAR
            c["v"], c["xprev"] = v, x
        rows[k] = {n: new[n] for n in cells}
    fin = {n: {j: round(we[0], 5) for j, we in c["src"].items() if j not in fixed} for n, c in cells.items()}
    return rows, fin


print("=" * 110)
print("C. THE WORKED EXAMPLE OF SECTION 13 (formulas 1 to 12 in the order of sentence 118; ages start at 0, so nothing is given up or made)")
PUBLISHED = {0: (0.4725, 0.1250, 0.0750, 0.0625, 0.0462, -0.0737, 0.0231, 0.3125, 0.0000, 0.4062),
             1: (0.2210, 0.2431, 0.1908, 0.1433, 0.0625, -0.0552, 0.0186, 0.4306, 0.1181, 0.3662),
             2: (0.0959, 0.1801, 0.1218, 0.1518, 0.1030, -0.0130, 0.0370, 0.3659, 0.0497, 0.3131),
             3: (0.0340, 0.1487, 0.0866, 0.1402, 0.1072, -0.0083, 0.0486, 0.3339, 0.0161, 0.2700),
             8: (0.0000, 0.1243, 0.0521, 0.1061, 0.0851, -0.0275, 0.0427, 0.3099, -0.0076, 0.2026),
             16: (0.0000, 0.1239, 0.0401, 0.1055, 0.0842, -0.0221, 0.0446, 0.0304, -0.2855, 0.0561),
             24: (0.0000, 0.1236, 0.0308, 0.1064, 0.0847, -0.0167, 0.0499, 0.0315, -0.2205, 0.0000),
             31: (0.0000, 0.1234, 0.0245, 0.1072, 0.0851, -0.0129, 0.0544, 0.0315, -0.1766, 0.0000)}
for variant in ("r4", "E4", "K"):
    rows, fin = example(variant)
    print(f"C.{['r4', 'E4', 'K'].index(variant) + 1} variant {variant}: step | F x | E A, m, x | I A, m, x | O A, m, x")
    worst = 0.0
    for k in PUBLISHED:
        r = rows[k]
        vals = (r["F"][3], r["E"][0], r["E"][1], r["E"][3], r["I"][0], r["I"][1], r["I"][3], r["O"][0], r["O"][1], r["O"][3])
        if variant == "r4":
            worst = max(worst, max(abs(round(a, 4) - b) for a, b in zip(vals, PUBLISHED[k])))
        print(f"    {k:2d} | " + " ".join(f"{v:.4f}" for v in vals))
    print(f"    final strengths: {json.dumps(fin)}")
    if variant == "r4":
        print(f"    largest difference from the published table after rounding to four decimals: {worst:.4f} "
              f"(0.0001 is a rounding of the last place)")

# =============================================================================================================== D. sign tests
print("=" * 110)
print("D. ONE-STEP TESTS OF THE SIGN OF THE TWO STRENGTH RULES (w = 0.5, e = 0.2, x(k-1) = 0.3)")
for sig in (1, -1):
    for m in (0.1, -0.1):
        print(f"D.1 formula 7: sigma {sig:+d}, m {m:+.1f}: delta w = {-ETA * sig * m * 0.3 * 0.2:+.6f}")
for vb in (0.1, 0.5):
    print(f"D.2 coincidence rule: theta-bar {vb}: delta w = {ETA * 0.2 * 0.3 * (0.3 - vb):+.6f} (the same for both kinds of sending cell)")
print(f"D.3 with x(k-1) = 0 both rules give 0; with e = 0 both give 0")

# =============================================================================================================== E. countermodel C0
print("=" * 110)
print("E. THE COUNTERMODEL C0: THE CLASSIFIER OF SECTION 7 FITTED TO A ONE-HOT CODE OF THE PRESENT BYTE ALONE")
FOLDS, GAP = 4, 16


def c0(st, w0, L):
    idx = [t for t in range(w0, w0 + WIN) if t - L >= 0]
    stretch = WIN // FOLDS
    feats = sorted(set(st[w0:w0 + WIN]))
    right, tot = {}, {}
    for f in range(FOLDS):
        lo, hi = w0 + f * stretch, w0 + (f + 1) * stretch
        test = [t for t in idx if lo <= t < hi]
        fit = [t for t in idx if not (lo - GAP <= t < hi + GAP)]
        n = len(fit)
        mu = {b: sum(1 for t in fit if st[t] == b) / n for b in feats}
        sd = {b: math.sqrt(mu[b] * (1 - mu[b])) for b in feats}
        use = [b for b in feats if sd[b] > 1e-9]
        zval = {b: {g: ((1.0 if g == b else 0.0) - mu[g]) / sd[g] for g in use} for b in feats}
        classes = sorted(set(st[t - L] for t in fit))
        cen = {}
        for c in classes:
            members = [t for t in fit if st[t - L] == c]
            cnt = {}
            for t in members:
                cnt[st[t]] = cnt.get(st[t], 0) + 1
            cen[c] = {g: sum(cnt[b] * zval[b][g] for b in cnt) / len(members) for g in use}
        pred = {}
        for b in feats:
            pred[b] = min(classes, key=lambda c: (sum((zval[b][g] - cen[c][g]) ** 2 for g in use), c))
        for t in test:
            lab = st[t - L]
            tot[lab] = tot.get(lab, 0) + 1
            right[lab] = right.get(lab, 0) + (pred[st[t]] == lab)
    return sum(right[c] / tot[c] for c in tot) / len(tot), len(tot)


for s in SEEDS:
    st = STREAMS[s]
    for wn in ("W0", "W3"):
        res = [c0(st, WINDOWS[wn], L) for L in LAGS]
        print(f"E.1 seed {s} {wn}: C0 by lag " + ", ".join(f"L={L}: {r[0]:.4f}" for L, r in zip(LAGS, res))
              + f"; classes {res[0][1]}, chance {1 / res[0][1]:.4f}")

# =============================================================================================================== F. time constants
print("=" * 110)
print("F. TIME CONSTANTS AND SHARES")
for n in (9, 32, 64, 128, 256):
    print(f"F.1 (31/32)^{n} = {(31 / 32) ** n:.4f}")
print(f"F.2 lags in steps: " + ", ".join(f"L={L}: {32 * L} steps" for L in LAGS) + "; state v keeps (1/2)^32 = "
      f"{0.5 ** 32:.2e} of a byte one window back")
u = 1 - (127 / 128) ** 32
k = math.ceil(math.log(u * 128) / -math.log(127 / 128))
print(f"F.3 running average u-bar of a line after one window of 32 steps on, from 0: {u:.4f}; steps of silence until it falls below 1/128: {k} "
      f"({k / 32:.1f} bytes)")
uu = 1.0 / 16
k2 = math.ceil(math.log(uu * 128) / -math.log(127 / 128))
print(f"F.4 from a running average of 1/16 (a line active once in 16 bytes, at its peak somewhat higher): {k2} steps ({k2 / 32:.1f} bytes) of silence")
st = STREAMS[73129]
for b in (ord("A"), ord("p"), ord("9"), END):
    pos = [t for t, x in enumerate(st) if x == b]
    gaps = [q - p for p, q in zip(pos, pos[1:])]
    print(f"F.5 byte {bytes([b])!r}: share {len(pos) / LENGTH:.4f}, mean gap {sum(gaps) / len(gaps):.1f} bytes, share of gaps longer than "
          f"{math.ceil(k / 32)} bytes {sum(1 for g in gaps if g > math.ceil(k / 32)) / len(gaps):.3f}")
print(f"F.6 first possible making and giving up at step P_s = 128 = byte 4 of life; P_s periods in the stream: {LENGTH * 32 // 128:,}")

# =============================================================================================================== G. cost and files
print("=" * 110)
print("G. COST AND FILES")
STEPS = LENGTH * 32
OPS_CONN_PLASTIC, OPS_CONN_FROZEN, OPS_CELL, CALLS, CALL_US, NS_PER_OP = 20, 4, 25, 80, 3.0, 1.5
for cbar in (8, 12, 32, 90):
    opsp = N * (cbar * OPS_CONN_PLASTIC + OPS_CELL)
    opsf = N * (cbar * OPS_CONN_FROZEN + OPS_CELL)
    tp = CALLS * CALL_US * 1e-6 + opsp * NS_PER_OP * 1e-9
    tf = (CALLS / 2) * CALL_US * 1e-6 + opsf * NS_PER_OP * 1e-9
    print(f"G.1 mean connections {cbar:3d}: plastic tissue {opsp:,} operations a step, {opsp * 32:,} a byte, {tp * 1e3:.2f} ms a step, "
          f"{tp * 32 * 1e3:.1f} ms a byte, {tp * STEPS / 3600:.2f} CPU-hours for the stream; frozen tissue {opsf * 32:,} a byte, "
          f"{tf * 32 * 1e3:.1f} ms a byte, {tf * STEPS / 3600:.2f} CPU-hours")
    if cbar == 12:
        r1_low = 6 * tp * STEPS / 3600 + 3 * tf * STEPS / 3600
    if cbar == 32:
        r1_high = 6 * tp * STEPS / 3600 + 3 * tf * STEPS / 3600
print(f"G.2 round 1 (six plastic and three frozen tissues): {r1_low:.1f} CPU-hours at a mean of 12 connections, {r1_high:.1f} at 32; "
      f"ceiling 40 CPU-hours")
print(f"G.3 total operations of round 1 at a mean of 12 connections: "
      f"{(6 * N * (12 * OPS_CONN_PLASTIC + OPS_CELL) + 3 * N * (12 * OPS_CONN_FROZEN + OPS_CELL)) * STEPS:.3e}")
rec = max(sum(1 for d in distances(BIRTHS[s]) if d in (1, 2, 4)) for s in SEEDS)
per_win = WIN * rec * 2
print(f"G.4 records: {len(WINDOWS)} windows x {WIN} bytes x the cells at d = 1, 2, 4 at the window's start ({rec} at birth in the "
      f"largest of the three draws) x 2 bytes (float16): {per_win / 1e6:.1f} MB a window, {len(WINDOWS) * per_win / 1e6:.1f} MB a tissue, "
      f"{9 * len(WINDOWS) * per_win / 1e6:.0f} MB for round 1 before compression, if the counts stay as at birth; "
      f"upper bound if every cell were at one of these distances {9 * len(WINDOWS) * WIN * N * 2 / 1e6:.0f} MB")
summ = 9 * len(WINDOWS) * N * 4 * 8 + 9 * LENGTH * 2
print(f"G.5 per-cell summaries (4 numbers a cell a window, float64) and emissions (int16 a byte): {summ / 1e6:.1f} MB for round 1")
for cbar in (12, 90):
    nc = N * cbar
    ck = nc * (4 * 8 + 3 * 4) + N * 12 * 8
    print(f"G.6 saved state at a mean of {cbar} connections: {nc:,} connections, {ck / 1e6:.2f} MB a tissue")
