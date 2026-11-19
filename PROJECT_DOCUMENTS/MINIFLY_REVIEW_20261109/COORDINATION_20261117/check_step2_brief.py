"""Check of the brief TASK_STEP_2_THE_TWO_ENDS_CORRECT_THEMSELVES_20261119.md: it recomputes every number of the brief that can be
computed without running a tissue.

Run from the repository's root:   python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/check_step2_brief.py
Standard library only; it reads no file and writes no file. About one minute.

The stream and the birth are copied from check_step1_brief.py (revision 3) without change, so that the two scripts agree.

Part A: the stream of steps 1 and 2 (positions 0 to 393,215), its classes of positions and their shares.
Part B: what a table of the last one, two and three bytes, fitted on the step-1 positions, scores for the next byte on the step-2
        positions, by class; the frequency emitter; which classes can be known from the last few bytes; constant emitters.
Part C: revision 5's worked example, part 1, with a "no" (as published) and with a "yes" (the check of what a yes moves).
Part D: the birth of form (b): the 64 emission feedback cells and their emission lines, for the three seeds.
Part E: the comparator requested of the coordinator: its arithmetic.
Part F: the cost of step 2 at the rates of the brief of step 1, and the bytes of each tissue.
"""
import hashlib
import json
from random import Random

SEEDS = (73129, 93683, 116089)
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
END, KMAX = 10, 12
L1 = 262144                                   # the step-1 stream: positions 0 to 262,143
L2 = 131072                                   # the step-2 stream: positions 262,144 to 393,215
LTOT = L1 + L2
STRETCH = 16384                               # a stretch of the step-2 measure
CLASSES = ["TASK", "A1", "A2", "SIGN", "LET", "B1", "B2", "EQ", "R1", "R2", "END"]


def apply(op, m, a, b):
    return {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]


def render(task, a, b, k, rng):              # as check_step1_brief.py, with the class of every byte
    op, m, code = task
    d = [rng.choice(DISTRACT) for _ in range(k)]
    byts = list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")] \
        + list(f"{apply(op, m, a, b):02d}".encode()) + [END]
    cls = ["TASK", "A1", "A2", "SIGN"] + ["LET"] * k + ["B1", "B2", "EQ", "R1", "R2", "END"]
    return byts, cls


def split(task, frac, seed):
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]).shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def stream(seed, n):
    rng = Random(seed + 77)
    flat = [(t, pr) for t in DEV for pr in split(t, .5, seed)[0]]
    out, cl = [], []
    while True:
        order = list(range(len(flat))); rng.shuffle(order)
        for i in order:
            t, (a, b) = flat[i]
            k = rng.randint(0, KMAX)
            byts, cls = render(t, a, b, k, rng)
            out.extend(byts); cl.extend(cls)
            if len(out) >= n:
                return out[:n], cl[:n]


print("=" * 110)
print("A. THE STREAM OF STEPS 1 AND 2")
ST, CL = {}, {}
for s in SEEDS:
    st, cl = stream(s, LTOT)
    ST[s], CL[s] = st, cl
    print(f"A.1 seed {s}: SHA-256 of positions 0 to {L1 - 1} {hashlib.sha256(bytes(st[:L1])).hexdigest()[:16]} (step 1, as in the brief "
          f"of step 1); of positions {L1} to {LTOT - 1} {hashlib.sha256(bytes(st[L1:])).hexdigest()}")
    print(f"    first 32 bytes of step 2: {bytes(st[L1:L1 + 32])!r}; class of position {L1}: {cl[L1]}")
s = 73129
n2 = {c: sum(1 for t in range(L1, LTOT) if CL[s][t] == c) for c in CLASSES}
print("A.2 seed 73129, step-2 positions by class of the byte that arrives (count, share): "
      + ", ".join(f"{c} {n2[c]} ({n2[c] / L2:.4f})" for c in CLASSES))
last = range(LTOT - STRETCH, LTOT)
print("A.3 seed 73129, the last stretch (positions 376,832 to 393,215), by class: "
      + ", ".join(f"{c} {sum(1 for t in last if CL[s][t] == c)}" for c in CLASSES))

# ============================================================================================================ B. tables
print("=" * 110)
print("B. TABLES OF THE LAST n BYTES, FITTED ON POSITIONS 0 TO 262,143, SCORED ON THE STEP-2 POSITIONS (share right, by class)")


def table(st, n):
    cnt = {}
    for t in range(n, L1):
        key = tuple(st[t - n:t])
        d = cnt.setdefault(key, {})
        d[st[t]] = d.get(st[t], 0) + 1
    return {k: min(d.items(), key=lambda kv: (-kv[1], kv[0]))[0] for k, d in cnt.items()}


KNOW = {}
for s in SEEDS:
    st, cl = ST[s], CL[s]
    freq = {}
    for b in st[:L1]:
        freq[b] = freq.get(b, 0) + 1
    top = min(freq.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    tabs = {n: table(st, n) for n in (1, 2, 3)}
    rows = {}
    for name, pred in [("frequency emitter", lambda t: top)] + \
            [(f"table of the last {n}", (lambda n: lambda t: tabs[n].get(tuple(st[t - n:t]), top))(n)) for n in (1, 2, 3)]:
        for rng_name, rng_ in (("step 2", range(L1, LTOT)), ("last stretch", range(LTOT - STRETCH, LTOT))):
            r, tot = {}, {}
            for t in rng_:
                c = cl[t]
                tot[c] = tot.get(c, 0) + 1
                r[c] = r.get(c, 0) + (pred(t) == st[t])
            rows[(name, rng_name)] = {c: r[c] / tot[c] for c in CLASSES}
    print(f"B.1 seed {s}: the most frequent byte of positions 0 to {L1 - 1} is {bytes([top])!r} (share {freq[top] / L1:.4f})")
    print("    " + "class".ljust(36) + "".join(c.rjust(7) for c in CLASSES))
    for (name, rng_name), v in rows.items():
        print("    " + f"{name} ({rng_name})".ljust(36) + "".join(f"{v[c]:7.3f}" for c in CLASSES))
    KNOW[s] = [c for c in CLASSES if rows[("table of the last 3", "step 2")][c] >= 0.9]
    print(f"    classes where the table of the last 3 bytes scores at least 0.9 on the step-2 positions: {KNOW[s]}")
    allr = rows[("table of the last 3", "step 2")]
    print(f"    table of the last 3, step 2, all positions: "
          f"{sum(allr[c] * sum(1 for t in range(L1, LTOT) if cl[t] == c) for c in CLASSES) / L2:.4f}")
    # constant emitters
    best = 0
    for b in range(256):
        hits = 0
        for c in KNOW[s]:
            pos = [t for t in range(LTOT - STRETCH, LTOT) if cl[t] == c]
            if sum(1 for t in pos if st[t] == b) / len(pos) >= 0.5:
                hits += 1
        best = max(best, hits)
    print(f"    a constant emitter (one byte value at every position) reaches 0.5 in at most {best} of the {len(KNOW[s])} classes "
          f"in the last stretch")
    # what each class's next byte is under the table of the last 3
    if s == SEEDS[0]:
        for c in KNOW[s]:
            vals = {}
            for t in range(L1, LTOT):
                if cl[t] == c:
                    vals[st[t]] = vals.get(st[t], 0) + 1
            print(f"    class {c}: byte values that arrive there: "
                  + ", ".join(f"{bytes([b])!r} {v}" for b, v in sorted(vals.items(), key=lambda kv: -kv[1])[:6]))
print(f"B.2 the classes that can be known from the last few bytes are the same on the three seeds: "
      f"{all(KNOW[s] == KNOW[SEEDS[0]] for s in SEEDS)} ({KNOW[SEEDS[0]]})")
nk = [sum(1 for t in range(LTOT - STRETCH, LTOT) if CL[s][t] in KNOW[s]) for s in SEEDS]
print(f"B.3 positions of those classes in the last stretch: {nk} (seeds in order); of each class about "
      f"{[round(sum(1 for t in range(LTOT - STRETCH, LTOT) if CL[73129][t] == c)) for c in KNOW[73129]]} (seed 73129)")
for p_ in (0.5, 0.3, 0.1):
    n = min(sum(1 for t in range(LTOT - STRETCH, LTOT) if CL[s][t] == c) for s in SEEDS for c in KNOW[s])
    print(f"B.4 standard error of a share {p_} over the smallest such class ({n} positions): {(p_ * (1 - p_) / n) ** 0.5:.4f}")

# ============================================================================================================ C. worked example
print("=" * 110)
print("C. REVISION 5, SECTION 13, PART 1, WITH A NO (AS PUBLISHED) AND WITH A YES")
ETA, TAU_V, TAU, ETA_T, ASTAR = 1 / 8, 2, 32, 1 / 128, 1 / 8


def example(answer):
    """answer 'no': F is a no-cell (excitatory), the no line active at step 0. answer 'yes': F is a yes-cell (inhibitory), the yes line
    active at step 0, O is the cell of the byte that was emitted and arrived; every other value as published."""
    ext = {"h": lambda k: 0.25, "c": lambda k: 0.125, "q": lambda k: 0.25, "g": lambda k: 0.125,
           "u": lambda k: 0.5 if k < 16 else 0.125, "f": lambda k: 0.25,
           "L3": lambda k: 1.0, "L5": lambda k: 0.0, "FL": lambda k: 1.0 if k == 0 else 0.0}
    sign = {"h": -1, "c": 1, "q": 1, "g": -1, "u": 1, "f": -1, "L3": 1, "L5": 1, "FL": 1, "E": 1, "F": 1 if answer == "no" else -1}
    fixed = {"L3", "L5", "FL"}
    cells = {
        "F": dict(src={"FL": [1.0, 0.0]}, v=0.0, p=0.02, th=0.0, xprev=0.0, plastic=False),
        "E": dict(src={"L3": [0.125, 0.02], "L5": [0.125, 0.40], "h": [0.25, 0.25], "c": [0.5, 0.125], "F": [0.25, 0.0]},
                  v=0.04, p=0.05, th=0.02, xprev=0.03, plastic=True),
        "I": dict(src={"E": [0.5, 0.06], "q": [0.25, 0.25], "g": [0.25, 0.125]}, v=0.10, p=0.12, th=0.05, xprev=0.06, plastic=True),
        "O": dict(src={"u": [0.75, 0.50], "f": [0.25, 0.25], "F": [0.25, 0.0]}, v=0.70, p=0.3125, th=0.10, xprev=0.60, plastic=True),
    }
    rows, du = {}, []
    for k in range(32):
        z = {n: f(k) for n, f in ext.items()}
        z.update({n: c["xprev"] for n, c in cells.items()})
        new = {}
        for n, c in cells.items():
            A = sum(sign[j] * w * z[j] for j, (w, e) in c["src"].items())
            m = A - c["p"]
            v = (1 - 1 / TAU_V) * c["v"] + (1 / TAU_V) * A
            new[n] = (A, m, v, min(1.0, max(0.0, v - c["th"])))
        u0 = cells["O"]["src"]["u"][0]
        for n, c in cells.items():
            A, m, v, x = new[n]
            if not c["plastic"]:
                continue
            for j, we in c["src"].items():
                if j in fixed:
                    continue
                we[0] = min(1.0, max(0.0, we[0] - ETA * sign[j] * m * c["xprev"] * we[1]))
        du.append(cells["O"]["src"]["u"][0] - u0)
        for n, c in cells.items():
            A, m, v, x = new[n]
            for j, we in c["src"].items():
                we[1] = (1 - 1 / TAU) * we[1] + (1 / TAU) * z[j]
            c["p"] = (1 - 1 / TAU) * c["p"] + (1 / TAU) * A
            if n != "F":
                c["th"] = max(0.0, c["th"] + ETA_T * (x - ASTAR))
            c["v"], c["xprev"] = v, x
        rows[k] = new
    fin = {n: {j: round(we[0], 4) for j, we in c["src"].items() if j not in fixed} for n, c in cells.items()}
    return rows, fin, du


for answer in ("no", "yes"):
    rows, fin, du = example(answer)
    print(f"C.{1 if answer == 'no' else 2} {answer}: step | F x | E A, m, x | I A, m, x | O A, m, x")
    for k in (0, 1, 2, 3, 8, 16, 24, 31):
        r = rows[k]
        vals = (r["F"][3], r["E"][0], r["E"][1], r["E"][3], r["I"][0], r["I"][1], r["I"][3], r["O"][0], r["O"][1], r["O"][3])
        print(f"    {k:2d} | " + " ".join(f"{v:+.4f}" if v < 0 else f"{v:.4f}" for v in vals))
    print(f"    change of O's strength from u at steps 1, 2 and 3: {du[1]:+.4f}, {du[2]:+.4f}, {du[3]:+.4f}; over the window "
          f"{sum(du):+.4f}; final strengths {json.dumps(fin)}")

# ============================================================================================================ D. form (b) birth
print("=" * 110)
print("D. THE BIRTH OF FORM (b): 64 EMISSION FEEDBACK CELLS, DRAWN BY Random(seed + 700000) AFTER THE BIRTH OF STEP 1")
N = 2048
for s in SEEDS:
    rb = Random(s)
    P = [(rb.random(), rb.random()) for _ in range(N)]
    IN_Y = [rb.random() for _ in range(256)]
    OUT_Y = [rb.random() for _ in range(256)]
    FB = rb.sample(range(N), 64)
    taken = set(FB)
    driver = []
    for v in range(256):
        best = min((i for i in range(N) if i not in taken), key=lambda i: ((1 - P[i][0]) ** 2 + (P[i][1] - OUT_Y[v]) ** 2, i))
        driver.append(best); taken.add(best)
    re_ = Random(s + 700000)
    eligible = sorted(set(range(N)) - set(FB) - set(driver))
    EM = re_.sample(eligible, 64)
    lines = [sorted(re_.sample(range(256), 8)) for _ in EM]
    dg = hashlib.sha256(json.dumps([EM, lines]).encode()).hexdigest()[:16]
    xs = sorted(P[i][0] for i in EM)
    cover = len({v for ls in lines for v in ls})
    print(f"D.1 seed {s}: eligible cells {len(eligible)}; digest of [emission cells, their lines] {dg}; first five cells {EM[:5]} "
          f"with lines {lines[0]}; x of the cells from {xs[0]:.3f} to {xs[-1]:.3f}; byte values with at least one emission line "
          f"{cover} of 256")

# ============================================================================================================ E. comparator
print("=" * 110)
print("E. THE COMPARATOR REQUESTED (a GRU of 128 cells, byte embedding of 32, readout to 256; one update per item)")
H, EMB, V = 128, 32, 256
fwd = 3 * H * (EMB + H) + H * V
print(f"E.1 multiply-adds a byte: forward {fwd:,}; with the backward pass (taken as twice the forward) {3 * fwd:,}")
items = [0, 0]
for s in (73129,):
    items = [sum(1 for t in range(L1) if CL[s][t] == "END"), sum(1 for t in range(L1, LTOT) if CL[s][t] == "END")]
print(f"E.2 seed 73129: items ended in positions 0 to {L1 - 1}: {items[0]:,}; in the step-2 positions: {items[1]:,}; "
      f"multiply-adds of training over all {LTOT:,} bytes: {3 * fwd * LTOT:.3e}")
w = 3 * H * (EMB + H) + 3 * 2 * H + V * EMB + H * V + V
print(f"E.3 its weights: {w:,}")

# ============================================================================================================ F. cost
print("=" * 110)
print("F. COST OF STEP 2 AT THE RATES OF THE BRIEF OF STEP 1 (to be replaced by stage A's measured rate)")
bytes_a = 3 * 2 * L2
bytes_b = L1 + 2 * L2
print(f"F.1 bytes: form (a), told and never told, three seeds: {bytes_a:,}; form (b), its step-1 phase and told and never told, "
      f"seed 73129: {bytes_b:,}; in all {bytes_a + bytes_b:,}")
for ms in (49.1, 88.4, 209.9):
    print(f"F.2 at {ms} ms a byte (step-1 brief, section 19, G.1: 12, 32 and 93.8 connections a cell with makings): "
          f"form (a) {bytes_a * ms / 1000 / 3600:.1f} CPU-hours, form (b) {bytes_b * ms / 1000 / 3600:.1f}, in all "
          f"{(bytes_a + bytes_b) * ms / 1000 / 3600:.1f}")
print(f"F.3 opening 1 (128 feedback cells; a new birth with its own step-1 phase, told and never told) on one seed: {bytes_b:,} bytes, "
      f"{bytes_b * 49.1 / 3.6e6:.1f} to {bytes_b * 209.9 / 3.6e6:.1f} CPU-hours")
print(f"F.4 records: emissions (int16) {2 * L2 / 1e6:.2f} MB a tissue of step 2; the step-2 positions of the eight step-2 tissues "
      f"{8 * 2 * L2 / 1e6:.1f} MB")
