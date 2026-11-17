import hashlib, json, statistics, subprocess, sys
from array import array
from random import Random
D, W, M = "origin/codex/minifly-consolidated-20261023", "origin/claude/gallant-dijkstra-i9zgsr", "MOD97_LOCAL_RULE_20261106"
SEEDS = (73129, 93683, 116089)
show = lambda ref, path: json.loads(subprocess.run(["git", "show", f"{ref}:{M}/exp45/{path}"], capture_output=True, check=True).stdout)
e41 = lambda rule, H, fp, s: show(D, f"e41/e41_{rule}_h{H}_f{fp}_{s}.json")
e42 = lambda cfg, rule, s: show(W, f"e42/e42_{cfg}_{rule}_{s}.json")
part = sys.argv[1] if len(sys.argv) > 1 else "all"

# ---- the task, copied from exp11_cell.py lines 224-265 (random.Random only)
EOS, BOS, BATCH = 256, 257, 64
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
LET = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ap = lambda op, m, a, b: {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]
def render(task, a, b, k, rng):
    op, m, code = task
    d = [rng.choice(LET) for _ in range(k)]
    return list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")], list(f"{ap(op, m, a, b):02d}".encode()) + [EOS]
def split(task, frac, seed):
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]).shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]
def batches(f, seed):
    rng = Random(seed + 77); flat = [(t, pr) for t in DEV for pr in split(t, f, seed)[0]]
    def stream():
        while True:
            order = list(range(len(flat))); rng.shuffle(order)
            for i in order:
                t, (a, b) = flat[i]
                yield render(t, a, b, rng.randint(0, 12), rng)
    st = stream()
    while True:
        yield [next(st) for _ in range(BATCH)]
def digest(f, seed, n):          # exp26_candidate.Digest over exp45.batch_tensor (int64 x, y; bool mask), without torch
    h = hashlib.sha256(); bt = batches(f, seed)
    for _ in range(n):
        items = next(bt); rows = [p + t for p, t in items]; L = max(map(len, rows)) + 1
        data = [[BOS] + r + [BOS] * (L - 1 - len(r)) for r in rows]
        x = b"".join(array("q", r[:-1]).tobytes() for r in data); y = b"".join(array("q", r[1:]).tobytes() for r in data)
        mask = b"".join(bytes(1 if len(p) <= j < len(p) + len(t) else 0 for j in range(L - 1)) for p, t in items)
        for t in (x, y, mask):
            h.update(str((len(rows), L - 1)).encode()); h.update(t)
    return h.hexdigest()

if part in ("all", "length"):
    for seed in SEEDS:
        lp = Random(seed + 3).sample(split(DEV[0], .5, seed)[1], 300)                      # exp42_cells.py line 230 with exp11_cell.panel
        for f in (.25, .10):
            taught = set(split(DEV[0], f, seed)[0])
            a_add = Random(seed + 3).sample(split(DEV[0], f, seed)[1], 300)                # the add panel behind A at f
            print(f"length panel, seed {seed}, f {f}: {sum(p in taught for p in lp)} of 300 pairs are taught; it shares {len(set(lp) & set(a_add))} pairs with the add panel behind A")
    t25, t50 = set(split(DEV[0], .25, 73129)[0]), set(split(DEV[0], .5, 73129)[0])
    print(f"seed 73129, addition: {len(t25)} pairs taught at a quarter, of which {len(t25 & t50)} are taught at a half")
    for rule in "SB":
        r = e41(rule, 512, 10, 73129)["curve"][-1]
        print(f"E41 dense, present width, f 0.10, {rule}, last row: addition right on {r['taught_counts']['add89']} of 300 taught and {r['A_counts']['add89']} of 300 untaught pairs at 12 letters")
if part in ("all", "stream"):
    for fp, f in ((25, .25), (10, .10)):
        want = next(r["data_digest"]["sha256"] for r in e41("S", 512, fp, 73129)["curve"] if r["step"] == 4096)
        print(f"stream digest at 4,096, f {f}: {digest(f, 73129, 4096)[:16]}; E41's receipt {want[:16]}")

def to80(c, key):
    v = [r[key] for r in c]
    return next((c[i]["step"] for i in range(len(v) - 2) if min(v[i:i + 3]) >= 80), None)
def m3(c, key, at):
    rows = [r[key] for r in c if at - 512 <= r["step"] <= at]; assert len(rows) == 3
    return sum(rows) / 3
if part in ("all", "e42"):
    for cfg in ("G", "F10", "P8"):
        for rule in "SB":
            cs = [e42(cfg, rule, s) for s in SEEDS]
            print(cfg, rule, "to 80 on A", [to80(r["curve"], "A") for r in cs], "on R", [to80(r["curve"], "R") for r in cs],
                  "A at end", [round(m3(r["curve"], "A", 24576), 1) for r in cs], "R at end", [round(m3(r["curve"], "R", 24576), 1) for r in cs],
                  "rise of R over the last 8,192", [round(m3(r["curve"], "R", 24576) - m3(r["curve"], "R", 16384), 1) for r in cs],
                  "theta", cs[0]["theta"], "CPU-h", [round((r["cpu_updates"] + r["cpu_scoring"] + r["cpu_replacement"]) / 3600, 2) for r in cs])
if part in ("all", "e41"):
    for H, fp in ((512, 25), (512, 10), (256, 25), (256, 10)):
        for rule in "SB":
            cs = [e41(rule, H, fp, s) for s in SEEDS]; d = cs[0]["dose"]; R = [m3(r["curve"], "R", d) for r in cs]
            print(f"E41 H={H} f=0.{fp} {rule} at {d}: A", [round(m3(r["curve"], "A", d), 1) for r in cs], "R", [round(x, 1) for x in R],
                  "range of R between seeds", round(max(R) - min(R), 1), "taught", [round(m3(r["curve"], "taught", d), 1) for r in cs],
                  "rise of R over the last 2,048", [round(m3(r["curve"], "R", d) - m3(r["curve"], "R", d - 2048), 1) for r in cs],
                  "over the last 8,192", [round(m3(r["curve"], "R", d) - m3(r["curve"], "R", d - 8192), 1) for r in cs],
                  "to 80 on A", [to80(r["curve"], "A") for r in cs], "on R", [to80(r["curve"], "R") for r in cs],
                  *((f"R at 12,288 and 24,576", [(round(m3(r["curve"], "R", 12288), 1), round(m3(r["curve"], "R", 24576), 1)) for r in cs]) if d > 24576 else ()))
if part in ("all", "same"):
    for rule in "SB":
        for s in SEEDS:
            a = {r["step"]: r for r in e41(rule, 512, 50, s)["curve"]}; b = {r["step"]: r for r in e42("G", rule, s)["curve"]}
            keys = ("loss", "A_counts", "R_counts", "taught_counts", "data_digest")
            print(f"G of E42 against E41 H=512 f=0.50, {rule} {s}: rows equal in loss, counts and digest:", sum(all(a[u][k] == b[u][k] for k in keys) for u in a), "of", len(a))
if part in ("all", "spread"):
    for cfg in ("F10", "P8"):
        for rule in "SB":
            g = [e42("G", rule, s)["curve"] for s in SEEDS]; c = [e42(cfg, rule, s)["curve"] for s in SEEDS]
            print(f"E42 f=0.50, level of R of {cfg} minus G under {rule}, by seed:",
                  {u: [round(m3(a, "R", u) - m3(b, "R", u), 1) for a, b in zip(c, g)] for u in (4096, 6144, 8192, 24576)})
    mr = lambda c, at: m3(c, "R", round(at / 256) * 256)
    for s in SEEDS:
        c = e41("B", 256, 10, s)["curve"]
        print(f"dense, half width, f=0.10, B, seed {s}: points of R by which a learner k times slower would be behind at 40,960:",
              {k: round(mr(c, 40960) - mr(c, 40960 / k), 1) for k in (1.5, 2.0)})
    def scatter(c):               # sd of the differences between successive rows over the last 2,048 updates, over root 2
        r = [x["R"] for x in c[-9:]]
        return round(statistics.pstdev([b - a for a, b in zip(r, r[1:])]) / 2 ** .5, 1)
    for H, fp in ((512, 10), (256, 10), (512, 25)):
        print(f"row-to-row scatter of R, dense, H={H} f=0.{fp}:", {rule: [scatter(e41(rule, H, fp, s)["curve"]) for s in SEEDS] for rule in "SB"})
    print("E41 S H=512 f=0.25 seed 73129, R of the last four rows:", [r["R"] for r in e41("S", 512, 25, 73129)["curve"][-4:]])
    for rule in "SB":
        r = e41(rule, 256, 10, 73129)
        print(f"dense, half width, {rule}: CPU-seconds an update {r['cpu_updates'] / r['dose']:.4f}, a scored row {r['cpu_scoring'] / len(r['curve']):.2f}")
    r = e42("P8", "S", 73129)["curve"]
    print("P8 S 73129, share of the 2,560 connections replaced an event: first 16 events", round(sum(x["replaced_since_previous_row"] for x in r[1:17]) / 16 / 2560, 3),
          "last 16", round(sum(x["replaced_since_previous_row"] for x in r[-16:]) / 16 / 2560, 3))
    print("length_counts['12'] == A_counts['add89'] at every row, E42 receipts:",
          sum(all(x["length_counts"]["12"] == x["A_counts"]["add89"] for x in e42(c, ru, s)["curve"]) for c in ("G", "F10", "P8") for ru in "SB" for s in SEEDS), "of 18")
