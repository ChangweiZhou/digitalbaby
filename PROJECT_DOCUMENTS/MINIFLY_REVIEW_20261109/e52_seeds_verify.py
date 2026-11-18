"""Verification of E52's other two seeds (the coordinator's decision 19; the worker's commit 5cb22315).

Read-only. Nothing is trained, the training library is not used, and no file of the worker's branch is executed: every file is read
with `git show` into memory and listed with `git ls-tree -r --name-only`; object identifiers are read with `git rev-parse <ref>:<path>`.
Weight files are read one at a time into memory (never extracted to disk) with an unpickler that refuses every class except those
that describe a tensor (copied from e52_verify.py, beside this file). The script writes no file.

Sections (the numbering of the verification document E52_SEEDS_VERIFICATION_20261118.md):
  1  inventory, completeness, settings, identities with E41's runs, logs, launch log, processes at once, cost, files changed
  2  every figure of E52_SEEDS_TABLES.md, of the report and of the worker's message, recomputed and set beside the reported figure
  3  the readings in the brief's fixed words seed by seed, the margins in rows, the explanation under test, seed 93683 at four rates
  4  a sample of two weight files: hashes, norms, and the final row's counts recomputed with a numpy forward
  5  form: unfilled marks, names of language models (counted, never written), commit cadence, the readings script fixed before the runs
  6  the count of figures compared and of those that differ, with every difference
The sections are printed in the order 1, 3, 2, 4, 5, 6, because the comparisons of section 2 use the readings of section 3.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e52_seeds_verify.py
"""
import collections, hashlib, io, json, math, os, pickle, re, subprocess, sys, time, zipfile
from datetime import datetime, timezone
from random import Random
import numpy as np

T0 = time.time()
sys.dont_write_bytecode = True

WREF = "5cb2231504590a365c0ec96dcd76a67070a5a302"   # the worker's commit of the report on the two seeds
BASE = "2d36fcfa2ea9860aa77f85ccf9e6d22556d2c744"   # the worker's commit before decision 19 (the first seed's corrected report)
PRE = "cb5cacfcb01384073c9c1a6e6071e73614a2b8a2"    # the commit of the first seed's verification
DREF = "7cabb13adc38753483d34dd5472fc3c84c4de675"   # the default branch at the first seed's source commit
TABS_COMMIT = "6e826b99a1f83a41a4b04fc60605ca6c237b82cb"   # the commit that added exp52_seeds_tables.py
DEC = "0cf288cea3050aa608784433f1b50f46f9065894"    # the default branch at decision 19
DEC_FILE = "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/COORDINATOR_DECISIONS_20261118.md"
M = "MOD97_LOCAL_RULE_20261106/"
NEW = M + "exp45/e52_seeds/"
E52 = M + "exp45/e52/"
E41 = M + "exp45/e41/"
REPORT = M + "E52_SEEDS_REPORT.md"
SEEDS = (73129, 116089, 93683)
NEWSEEDS = (73129, 116089)
SETTINGS = ((256, 50), (256, 25), (512, 25))
WIDTH = {256: "half", 512: "present"}
U_OF = {256: 128, 512: 256}
DOSE = {256: 40960, 512: 12288}
ROW = 256
KMAX = 200
GROUPS = ("book", "l1", "input_map", "recurrent", "lm", "l2", "l3")
GROUP_TENSORS = {"book": ("book.weight",), "l1": ("l1.weight", "l1.bias"), "input_map": ("rnn.weight_ih_l0", "rnn.bias_ih_l0"),
                 "recurrent": ("rnn.weight_hh_l0", "rnn.bias_hh_l0"), "lm": ("lm.weight", "lm.bias"), "l2": ("l2.weight", "l2.bias"),
                 "l3": ("l3.weight", "l3.bias")}
MIN_FREE = 0.9e9
SAME = "the same within the scatter of one seed"
COMP = []      # (section, label, reported, recomputed, equal)


def clock(label):
    print(f"[{label}: {time.time() - T0:.1f} s]", flush=True)


def cmp(sec, label, reported, recomputed, equal=None):
    eq = (str(reported) == str(recomputed)) if equal is None else bool(equal)
    COMP.append((sec, label, str(reported), str(recomputed), eq))
    return eq


# ---------------------------------------------------------------- reading through git
def show(path, ref=WREF):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def jshow(path, ref=WREF):
    return json.loads(show(path, ref))


def names(d, ref=WREF):
    args = ["git", "ls-tree", "-r", "--name-only", "-z", ref] + (["--", d] if d else [])
    out = subprocess.run(args, capture_output=True, check=True).stdout.decode("utf-8", "surrogateescape")
    return [x for x in out.split("\0") if x]


def oids(ref, paths):
    """The object identifier of each path at ref, read from the trees by `git rev-parse` (no blob is read)."""
    out = {}
    paths = list(paths)
    for i in range(0, len(paths), 1500):
        chunk = paths[i:i + 1500]
        r = subprocess.run(["git", "rev-parse"] + [f"{ref}:{p}" for p in chunk], capture_output=True, text=True)
        lines = r.stdout.split("\n")
        if r.returncode == 0 and len([x for x in lines if x]) == len(chunk):
            out.update(zip(chunk, lines))
        else:
            for p in chunk:
                q = subprocess.run(["git", "rev-parse", f"{ref}:{p}"], capture_output=True, text=True)
                out[p] = q.stdout.strip() if q.returncode == 0 else None
    return out


def git_text(args):
    return subprocess.run(["git"] + args, capture_output=True, check=True, text=True).stdout


sha = lambda b: hashlib.sha256(b).hexdigest()
fu = lambda v: "–" if v is None else f"{v:,}"
utc = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
hms = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M:%S")


def free_bytes():
    s = os.statvfs("/home/user")
    return s.f_bavail * s.f_frsize


# ---------------------------------------------------------------- weight files (copied from e52_verify.py)
class _Unpickler(pickle.Unpickler):
    """Refuses every class except those that describe a tensor: nothing from the file is executed."""
    def find_class(self, mod, name):
        if mod == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if mod == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, stride, *a, **k: ("T", storage, offset, tuple(size), tuple(stride))
        if mod == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *a, **k: data
        if mod == "torch" and name.endswith("Storage"):
            return name
        raise pickle.UnpicklingError(f"refused {mod}.{name}")

    def persistent_load(self, pid):
        return (pid[1], pid[2])


DTYPE = {"FloatStorage": "<f4", "DoubleStorage": "<f8", "LongStorage": "<i8", "BoolStorage": "?"}


def load_pt(blob):
    z = zipfile.ZipFile(io.BytesIO(blob))
    pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    root = pk[:-len("data.pkl")]
    obj = _Unpickler(z.open(pk)).load()
    W = {}
    for k, o in obj.items():
        _, (stype, key), offset, shape, stride = o
        n = int(np.prod(shape)) if shape else 1
        dt = np.dtype(DTYPE[stype])
        raw = z.read(root + "data/" + key)[offset * dt.itemsize:(offset + n) * dt.itemsize]
        a = np.frombuffer(raw, dtype=dt).reshape(shape).copy()
        exp, acc = [], 1
        for s in reversed(shape):
            exp.append(acc); acc *= s
        assert tuple(reversed(exp)) == tuple(stride), (k, shape, stride)
        W[k] = a
    return W


def tensors_sha(W):
    h = hashlib.sha256()
    for k in sorted(W):
        v = W[k]
        h.update(k.encode()); h.update({np.float32: b"torch.float32", np.float64: b"torch.float64", np.int64: b"torch.int64"}[v.dtype.type])
        h.update(str(tuple(v.shape)).encode()); h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- the task and the forward (copied from e52_verify.py)
EOS, BOS = 256, 257
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def apply(op, m, a, b):
    return {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]


def render(task, a, b, k, rng):
    op, m, code = task
    d = [rng.choice(DISTRACT) for _ in range(k)]
    return list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")], list(f"{apply(op, m, a, b):02d}".encode()) + [EOS]


def split(task, frac, seed):
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]).shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def panel(task, pairs, seed, ks, n):
    rng = Random(seed + 3); sub = rng.sample(pairs, n)
    return {k: [render(task, a, b, k, rng) for a, b in sub] for k in ks}


def reduction_sample(t, held, f, seed):
    pairs = [(a, b) for a, b in held if (a + b >= 89 if t[0] == "add" else a - b < 0)]
    return Random(f"e41|{seed}|{round(f * 100)}|{t[0]}").sample(sorted(pairs), 1000)


def panels(f, seed):
    th = {t: split(t, f, seed) for t in DEV}
    A = {f"{t[0]}{t[1]}": panel(t, th[t][1], seed, (12,), 300)[12] for t in DEV}
    TA = {f"{t[0]}{t[1]}": panel(t, th[t][0], seed + 1, (12,), 300)[12] for t in DEV}
    Rs, dig = {}, {}
    for t in DEV[:2]:
        s = reduction_sample(t, th[t][1], f, seed)
        Rs[f"{t[0]}{t[1]}"] = [render(t, a, b, 0, Random(0)) for a, b in s]
        dig[f"{t[0]}{t[1]}"] = sha(json.dumps([list(p) for p in s]).encode())
    LP = panel(DEV[0], th[DEV[0]][1], seed, (12, 40), 300)
    return A, Rs, TA, LP, dig, th


sig = lambda v: 1.0 / (1.0 + np.exp(-v))


class Net:
    def __init__(self, W):
        self.w = {k: np.asarray(v, np.float32) for k, v in W.items()}

    def embed(self, x):
        w = self.w
        z1 = np.tanh(w["book.weight"][x] @ w["l1.weight"].T + w["l1.bias"])
        return z1, z1 @ w["rnn.weight_ih_l0"].T + w["rnn.bias_ih_l0"]

    def step(self, h, z1, gi):
        w = self.w
        gh = h @ w["rnn.weight_hh_l0"].T + w["rnn.bias_hh_l0"]
        n = h.shape[1]
        r = sig(gi[:, :n] + gh[:, :n]); z = sig(gi[:, n:2 * n] + gh[:, n:2 * n]); c = np.tanh(gi[:, 2 * n:] + r * gh[:, 2 * n:])
        h = (1 - z) * c + z * h
        z2 = np.tanh(z1 @ w["l2.weight"].T + w["l2.bias"] + h @ w["lm.weight"].T + w["lm.bias"])
        return h, z2 @ w["l3.weight"].T + w["l3.bias"]

    def respond(self, prompts, cap=4):
        out = [None] * len(prompts); groups = {}
        for i, p in enumerate(prompts):
            groups.setdefault(len(p), []).append(i)
        for _, ids in groups.items():
            x = np.array([[BOS, *prompts[i]] for i in ids])
            z1, gi = self.embed(x)
            h = np.zeros((len(ids), self.w["rnn.weight_hh_l0"].shape[1]), np.float32)
            for t in range(x.shape[1]):
                h, lg = self.step(h, z1[:, t], gi[:, t])
            gen = [[] for _ in ids]; done = [False] * len(ids)
            for _ in range(cap):
                yy = lg.argmax(1)
                for q in range(len(ids)):
                    if not done[q]:
                        gen[q].append(int(yy[q])); done[q] = done[q] or int(yy[q]) == EOS
                if all(done):
                    break
                z1n, gin = self.embed(yy[:, None])
                h, lg = self.step(h, z1n[:, 0], gin[:, 0])
            for q, i in enumerate(ids):
                out[i] = gen[q]
        return out

    def exact(self, items):
        outs = self.respond([p for p, _ in items])
        return sum(o == t for o, (_, t) in zip(outs, items))


# ---------------------------------------------------------------- scores from raw counts, and the brief's readings
def scores(x):
    """A, R and taught from the raw counts of a row (add and sub only; A and taught of 300 items each, R of 1,000 each)."""
    A = 100 * (x["A_counts"]["add89"] + x["A_counts"]["sub89"]) / 600
    R = 100 * (x["R_counts"]["add89"] + x["R_counts"]["sub89"]) / 2000
    T = 100 * (x["taught_counts"]["add89"] + x["taught_counts"]["sub89"]) / 600
    L = 100 * x["length_counts"]["40"] / 300 if "length_counts" in x else None
    return A, R, T, L


def to80(vals, steps):
    for i in range(len(vals) - 2):
        if vals[i] >= 80 and vals[i + 1] >= 80 and vals[i + 2] >= 80:
            return steps[i]
    return None


def mean(v):
    return sum(v) / len(v)


def summarize(r, path):
    c = r["curve"]
    steps = [x["step"] for x in c]
    sc = [scores(x) for x in c]
    s = dict(path=path, dose=r["dose"], last=r["dose"] - 512, rows=len(c), steps=steps, r=r)
    s["A"], s["R"], s["T"], s["L"] = [v[0] for v in sc], [v[1] for v in sc], [v[2] for v in sc], [v[3] for v in sc]
    s["fields_equal"] = all(abs(v[0] - x["A"]) < 1e-9 and abs(v[1] - x["R"]) < 1e-9 and abs(v[2] - x["taught"]) < 1e-9 and
                            (v[3] is None or abs(v[3] - x["len40"]) < 1e-9) for v, x in zip(sc, c))
    s["A80"], s["R80"] = to80(s["A"], steps), to80(s["R"], steps)
    s["A_final"], s["R_final"], s["T_final"] = mean(s["A"][-3:]), mean(s["R"][-3:]), mean(s["T"][-3:])
    s["has_len"] = all(v is not None for v in s["L"])
    if s["has_len"]:
        s["len_last"], s["len_mean3"] = s["L"][-1], mean(s["L"][-3:])
        s["len_lo"], s["len_hi"] = min(s["L"][-12:]), max(s["L"][-12:])
    s["norms"] = c[-1].get("norms")
    return s


def klass(x1, x2, D1, D2):
    """exp41_scale.describe_step as read, with the run at 0.001 in the place of the narrower run (x1, D1)."""
    if x1 is not None and x2 is not None:
        q = x2 / x1
        return "faster" if q < 2 / 3 else "slower" if q > 1.5 else "the same"
    if x1 is not None:
        return "slower" if D2 / x1 >= 1.5 else "not decided"
    if x2 is not None:
        return "faster" if x2 / D1 <= 2 / 3 else "not decided"
    return "neither reaches"


RATES2 = (0.001, 0.003)


def best(C, D, rule, key, rates=RATES2):
    got = {r: C[(rule, r, key)] for r in rates if C.get((rule, r, key)) is not None}
    if not got:
        return dict(word="not reached", count=None)
    lo = min(got.values()); tied = sorted(r for r, v in got.items() if v - lo <= ROW)
    if len(tied) == 1:
        return dict(word=f"{tied[0]:g}", count=got[tied[0]])
    if 0.001 in tied:
        return dict(word="0.001", count=got[0.001])
    return dict(word="tie of " + ", ".join(f"{r:g}" for r in tied), count=lo)


def vs(C, D, rule, key, rate=0.003):
    return klass(C[(rule, 0.001, key)], C[(rule, rate, key)], D, D)


def gains(C, D, rule, rates=RATES2):
    others = [r for r in rates if r != 0.001 and (rule, r, "R") in C]
    if any(C.get((rule, r, "R")) is not None for r in rates):
        return any(vs(C, D, rule, "R", r) == "faster" and vs(C, D, rule, "A", r) != "slower" for r in others)
    return any(vs(C, D, rule, "A", r) == "faster" for r in others)


def interval(xs, xb, D):
    """The lag as an interval [lo, hi] from two counts; a missing count gives a bound from the last recordable count D."""
    if xs is not None and xb is not None:
        return (xs / xb, xs / xb, "value")
    if xb is not None:
        return (D / xb, math.inf, "lower")
    if xs is not None:
        return (0.0, xs / D, "upper")
    return None


def lag(C, D, key, rates=RATES2):
    at = interval(C[("S", 0.001, key)], C[("B", 0.001, key)], D)
    bs, bb = best(C, D, "S", key, rates), best(C, D, "B", key, rates)
    own = interval(bs["count"], bb["count"], D)
    if at is None or own is None:
        cls = "not formed"
    else:
        qlo = own[0] / at[1] if at[1] != math.inf else 0.0
        qhi = own[1] / at[0] if at[0] > 0 else math.inf
        if qhi < 2 / 3:
            cls = "gone" if own[1] <= 1.0 else "smaller"
        elif qlo > 1.5:
            cls = "larger"
        elif qlo >= 2 / 3 and qhi <= 1.5:
            cls = SAME
        else:
            cls = "not decided"
    return dict(at=at, own=own, cls=cls, sbest=bs, bbest=bb)


def lagtext(iv):
    lo, hi, kind = iv
    if kind == "value":
        return f"{lo:.2f}"
    if kind == "lower":
        return f"above {math.floor(lo * 100) / 100:.2f}"
    return f"below {math.ceil(hi * 100) / 100:.2f}"


def readings(C, D, rates=RATES2):
    out = {}
    for rule in "SB":
        for key in "RA":
            out[f"best {key} {rule}"] = best(C, D, rule, key, rates)["word"]
            out[f"vs {rule} {key}"] = vs(C, D, rule, key)
        out[f"gains {rule}"] = "yes" if gains(C, D, rule, rates) else "no"
    for key in "RA":
        out[f"lag {key}"] = lag(C, D, key, rates)["cls"]
    return out


def margins(C, D):
    """The fewest rows by which one count to 80 must move to change each reading, and every reading it can then become.
    A count c moves to c +- 256 k (never below 256); past the last recordable row it becomes 'not reached'; a count that is not
    reached becomes reached at the last recordable row less 256 (k - 1). Every reading is recomputed from the moved counts."""
    base = readings(C, D)
    out = {n: None for n in base}
    for k in range(1, KMAX + 1):
        hits = collections.defaultdict(set)
        for var, c in C.items():
            if c is None:
                cand = [D - ROW * (k - 1)] if D - ROW * (k - 1) >= ROW else []
            else:
                cand = [None if v > D else v for v in (c + ROW * k, c - ROW * k) if v >= ROW]
            for v in cand:
                C2 = dict(C); C2[var] = v
                r2 = readings(C2, D)
                for n in base:
                    if out[n] is None and r2[n] != base[n]:
                        hits[n].add(r2[n])
        for n, s in hits.items():
            out[n] = (k, s)
        if all(v is not None for v in out.values()):
            break
    return base, out


# ---------------------------------------------------------------- markdown tables
def md_tables(text):
    out, cur = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            cur.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif cur:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return [(t[0], t[2:]) for t in out]


def find_table(tables, header_start):
    return [t for t in tables if " | ".join(t[0]).startswith(header_start)][0]


NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def nums(s):
    return [x.replace(",", "") for x in NUM.findall(s)]


# ================================================================ 1
def section1(ctx):
    print("================ 1  inventory, completeness, settings, identities, logs, cost, files changed")
    allf = names(NEW)
    rec = sorted(n for n in allf if n.endswith(".json") and n.split("/")[-1].startswith("e52_"))
    pts = sorted(n for n in allf if n.endswith(".pt"))
    logs = sorted(n for n in allf if "/logs/" in n)
    other = sorted(set(allf) - set(rec) - set(pts) - set(logs))
    print(f"  files in exp45/e52_seeds: {len(allf)} (receipts {len(rec)}, weight files {len(pts)}, logs {len(logs)}, other {len(other)}: "
          f"{[o.split('/')[-1] for o in other]}); checkpoints committed: {sum(n.endswith('.ckpt') for n in allf)}; "
          f"files in an interrupted/ folder: {sum('/interrupted/' in n for n in allf)}")
    ctx["inv"] = dict(n=len(allf), rec=len(rec), pts=len(pts), logs=len(logs), other=other)

    srcnames = ("exp52_resume.py", "exp52_rate.py", "exp46_decay.py", "exp41_scale.py", "arch26.py", "exp26_candidate.py", "arch20.py",
                "exp20_reference.py", "exp16_integrated.py", "arch17.py", "exp11_cell.py", "exp45.py", "exp3_state.py", "mod97_local.py")
    srcsha = {s: sha(show(M + s)) for s in srcnames}
    pre_ids, w_ids = oids(PRE, [M + s for s in srcnames]), oids(WREF, [M + s for s in srcnames])
    print(f"  the 14 sources the runs record are the same blobs at the worker's commit as at the first seed's verified commit {PRE[:8]}: "
          f"{all(pre_ids[M + s] == w_ids[M + s] for s in srcnames)}")
    ctx["src_same_pre"] = all(pre_ids[M + s] == w_ids[M + s] for s in srcnames)
    ctx["srcsha"] = srcsha

    # the runs: 0.003 from exp45/e52_seeds, 0.001 from E41's receipts; seed 93683 from exp45/e52 (all four rates)
    RUNS, RAW = {}, {}
    for seed in NEWSEEDS:
        for H, fp in SETTINGS:
            for rule in "SB":
                p3 = f"{NEW}e52_{rule}_h{H}_f{fp}_lr0.003_{seed}.json"
                p1 = f"{E41}e41_{rule}_h{H}_f{fp}_{seed}.json"
                RAW[(seed, rule, H, fp, 0.003)] = jshow(p3); RAW[(seed, rule, H, fp, 0.001)] = jshow(p1)
                RUNS[(seed, rule, H, fp, 0.003)] = summarize(RAW[(seed, rule, H, fp, 0.003)], p3)
                RUNS[(seed, rule, H, fp, 0.001)] = summarize(RAW[(seed, rule, H, fp, 0.001)], p1)
    for H in (256, 512):
        for fp in (25, 50):
            for rule in "SB":
                for rate in (0.0003, 0.001, 0.003, 0.01):
                    p = f"{E52}e52_{rule}_h{H}_f{fp}_lr{rate:g}_93683.json"
                    if p in ctx.setdefault("e52names", set(names(E52))):
                        RAW[(93683, rule, H, fp, rate)] = jshow(p)
                        RUNS[(93683, rule, H, fp, rate)] = summarize(RAW[(93683, rule, H, fp, rate)], p)
    ctx["RUNS"], ctx["RAW"] = RUNS, RAW
    print(f"  receipts read: {len([k for k in RUNS if k[0] != 93683 and k[4] == 0.003])} new, "
          f"{len([k for k in RUNS if k[0] != 93683 and k[4] == 0.001])} of E41 (seeds 73129, 116089), {len([k for k in RUNS if k[0] == 93683])} of E52's first seed")
    print(f"  A, R, taught and the length test recomputed from the raw counts equal the receipts' own fields at every row: "
          f"{all(s['fields_equal'] for s in RUNS.values())}")

    bad = collections.defaultdict(list); ntests = 0
    e41_ids_w = oids(WREF, [RUNS[k]["path"] for k in RUNS if k[4] == 0.001 and k[0] != 93683])
    e41_ids_d = oids(DREF, list(e41_ids_w))
    for seed in NEWSEEDS:
        for H, fp in SETTINGS:
            for rule in "SB":
                k = (seed, rule, H, fp, 0.003)
                r, c = RAW[k], RAW[k]["curve"]
                e = RAW[(seed, rule, H, fp, 0.001)]
                dose = DOSE[H]
                rest = [p for p in r["model"]["params_by_tensor"] if not p.startswith("book.")]
                g = [dict(weight_decay=0.25, lr=0.003, betas=[0.9, 0.98], params=rest), dict(weight_decay=0.0, lr=0.003, betas=[0.9, 0.98], params=["book.weight"])]
                tests = {
                    "complete": r["complete"] is True, "not diverged": r["diverged"] is False, "dose = E41's": r["dose"] == dose == e["dose"] == r["e41_dose"],
                    "updates_done = dose": r["updates_done"] == dose, "optimizer steps = dose": r["optimizer_steps"] == [dose],
                    "rows 0..dose every 256": [x["step"] for x in c] == list(range(0, dose + 1, ROW)) and r["row_every"] == ROW,
                    "finite_losses every row": all(x["finite_losses"] is True for x in c),
                    "losses finite": all(x["loss"] is None or (isinstance(x["loss"], float) and math.isfinite(x["loss"])) for x in c),
                    "rule": r["rule"] == rule and r["local"] == (rule == "S"), "seed": r["seed"] == seed, "fraction": r["fraction"] == fp / 100,
                    "width": r["H"] == H and r["U"] == U_OF[H] and r["model"]["H"] == H and r["model"]["U"] == U_OF[H],
                    "params": r["model"]["params"] == {256: 317457, 512: 1124113}[H] == e["model"]["params"],
                    "batch 64, kmax 12": r["batch"] == 64 and r["kmax"] == 12 and e["batch"] == 64 and e["kmax"] == 12,
                    "rate 0.003": r["rate"] == 0.003, "decay 0.25": r["decay"] == 0.25, "no product control": r["product_control"] is False,
                    "optimizer record": r["optimizer"]["name"] == "AdamW" and r["optimizer"]["lr"] == 0.003 and r["optimizer"]["weight_decay"] == 0.25
                                        and r["optimizer"]["betas"] == [0.9, 0.98] and r["optimizer"]["groups"] == g,
                    "groups expected": r["optimizer_groups_expected"] == g, "groups at start": r["optimizer_groups_at_start"] == g,
                    "groups at end": r["optimizer_groups_at_end"] == g, "groups read back": r["optimizer_read_back"]["groups"] == g,
                    "update and forward as E41's": r["update"] == e["update"] and r["model"]["forward"] == e["model"]["forward"] == "arch26.forward",
                    "data stream as E41's": r["data"].startswith(e["data"]),
                    "R sample = E41's": r["reduction_sample_sha256"] == e["reduction_sample_sha256"],
                    "length test drawn from the run's own split": "split(('add', 89, b'p'), f, seed)[1]" in r["length_test"],
                    "length 12 items = A add items; none taught": r["length_test_12_items_equal_A_add89"] is True and r["length_test_items_in_taught"] == 0,
                    "length 12 count = A add count every row": all(x["length_counts"]["12"] == x["A_counts"]["add89"] for x in c),
                    "norms at every row": all(set(x["norms"]) == set(GROUPS) for x in c),
                    "digest 4096 = E41 (flag)": r["digest_4096_equals_e41"] is True,
                    "digest 4096 = E41 (recomputed)": next(x["data_digest"]["sha256"] for x in c if x["step"] == 4096) == next(
                        x["data_digest"]["sha256"] for x in e["curve"] if x["step"] == 4096) == r["e41_digest_4096"],
                    "digest = E41 at every row": [x["data_digest"] for x in c] == [x["data_digest"] for x in e["curve"]],
                    "final digest": r["data_digest"] == c[-1]["data_digest"] and r["data_digest"]["batches"] == dose,
                    "row 0 counts = E41's row 0": all(c[0][f] == e["curve"][0][f] for f in ("A_counts", "R_counts", "taught_counts")),
                    "no resumption record": "resumed" not in r and not any("resum" in kk for kk in r),
                    "runner exp52_resume.py": r["provenance"]["runner"].startswith("exp52_resume.py") and r["provenance"]["runner_sha256"] == srcsha["exp52_resume.py"]
                                              and "exp52_resume.py run" in r["provenance"]["command"] and "--resume" not in r["provenance"]["command"],
                    "checkpoint every 4096": r["checkpoint_every"] == 4096,
                    "sources sha = committed files": all(srcsha.get(s) == v for s, v in r["provenance"]["sources_sha256"].items())
                                                     and set(r["provenance"]["sources_sha256"]) == set(srcnames),
                    "no source modified vs commit": r["provenance"]["git"]["sources_modified_vs_commit"] == [],
                    "exp46_decay copy from E46's commit": r["provenance"]["exp46_decay_copy"]["from_commit"] == "b76f5979",
                    "host": r["provenance"]["host"]["torch"] == "2.15.0+cpu" and r["provenance"]["host"]["torch_threads"] == 1
                            and r["provenance"]["host"]["python"] == "3.13.18" and r["provenance"]["host"]["cores"] == 4,
                    "environment": r["provenance"]["environment_at_start"] == {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                    "weights named": r["weights"] == f"e52_{rule}_h{H}_f{fp}_lr0.003_{seed}.pt" and NEW + r["weights"] in allf,
                    "E41 receipt complete, same blob as on the default branch": e["complete"] is True and e41_ids_w[RUNS[(seed, rule, H, fp, 0.001)]["path"]]
                                                                                  == e41_ids_d[RUNS[(seed, rule, H, fp, 0.001)]["path"]],
                    "E41 at 0.001 and decay 0.25": e["optimizer"]["lr"] == 0.001 and e["optimizer"]["weight_decay"] == 0.25,
                }
                ntests = len(tests)
                for t, ok in tests.items():
                    if not ok:
                        bad[t].append(f"{seed} {rule} {WIDTH[H]} f{fp}")
    ctx["bad"] = dict(bad); ctx["ntests"] = ntests
    print(f"  the 12 new receipts against {ntests} tests each (settings of the brief and of decision 19, completeness, identities with E41): "
          f"failures {dict(bad) if bad else 'none'}")
    # stream identity within a seed and fraction, across rules and widths (new runs and E41's)
    by = collections.defaultdict(lambda: collections.defaultdict(set))
    for k, s in RUNS.items():
        if k[0] in NEWSEEDS:
            for x in s["r"]["curve"]:
                by[(k[0], k[3])][x["step"]].add(x["data_digest"]["sha256"])
    ok_stream = {f"{sd} f{fp}": all(len(v) == 1 for v in d.values()) for (sd, fp), d in sorted(by.items())}
    print(f"  batch digests identical at every row across rules, widths and both rates within each seed and fraction: {ok_stream}")
    ctx["ok_stream"] = all(ok_stream.values())
    st = collections.defaultdict(set)
    for k, s in RUNS.items():
        if k[0] in NEWSEEDS and k[4] == 0.003:
            st[(k[0], k[2])].add(json.dumps(s["r"]["curve"][0]["norms"], sort_keys=True))
    print(f"  the same start network within each seed and width (row-0 norms identical across rules and fractions): {all(len(v) == 1 for v in st.values())}")
    ctx["start_ok"] = all(len(v) == 1 for v in st.values())
    # seed 93683: E52's runs at 0.001 equal E41's
    eq = []
    for H, fp in SETTINGS:
        for rule in "SB":
            a, b = RAW[(93683, rule, H, fp, 0.001)], jshow(f"{E41}e41_{rule}_h{H}_f{fp}_93683.json")
            eq.append(a["tensors_sha256"] == b["tensors_sha256"] and all(x[f] == y[f] for x, y in zip(a["curve"], b["curve"])
                                                                         for f in ("A_counts", "R_counts", "taught_counts", "data_digest")))
    print(f"  seed 93683: E52's runs at 0.001 end on E41's tensors_sha256 and equal E41's counts and digests at every row: {sum(eq)} of {len(eq)}")
    ctx["e93_eq"] = (sum(eq), len(eq))
    e41_no = all("length_counts" not in x and "norms" not in x for k, s in RUNS.items() if k[0] in NEWSEEDS and k[4] == 0.001 for x in s["r"]["curve"])
    print(f"  E41's receipts of seeds 73129 and 116089 record neither the length test nor the norms at any row: {e41_no}")
    ctx["e41_no"] = e41_no

    # logs: every row printed once, equal to the receipt at the printed precision
    pat = re.compile(r"^e52 ([SB]) h(\d+) f(\d+) lr([\d.]+) (\d+) (\d+): A ([\d.]+) R ([\d.]+) taught ([\d.]+) len40 ([\d.]+) rec-norm ([\d.]+) cpu (\d+)\+(\d+)s$")
    logok, lognote = 0, []
    for seed in NEWSEEDS:
        for H, fp in SETTINGS:
            for rule in "SB":
                k = (seed, rule, H, fp, 0.003); r = RAW[k]
                lines = show(f"{NEW}logs/e52_{rule}_h{H}_f{fp}_lr0.003_{seed}.log").decode().splitlines()
                rows = [m.groups() for m in map(pat.match, lines) if m]
                steps = [int(g[5]) for g in rows]
                other_lines = [l for l in lines if not pat.match(l)]
                byst = {x["step"]: x for x in r["curve"]}
                same = all(f"{byst[int(g[5])]['A']:.2f}" == g[6] and f"{byst[int(g[5])]['R']:.2f}" == g[7] and f"{byst[int(g[5])]['taught']:.2f}" == g[8]
                           and f"{byst[int(g[5])]['len40']:.1f}" == g[9] and f"{byst[int(g[5])]['norms']['recurrent']:.2f}" == g[10] for g in rows)
                comp = [l for l in other_lines if l.startswith("e52 complete")]
                good = (steps == list(range(0, DOSE[H] + 1, ROW)) and same and len(comp) == 1 and r["tensors_sha256"].startswith(comp[0].split()[3])
                        and not any("resum" in l.lower() for l in lines))
                logok += good
                if not good or len(other_lines) != 1:
                    lognote.append((seed, rule, H, fp, len(rows), other_lines[:3]))
    print(f"  logs: every scored row printed exactly once, from 0 to the dose, equal to the receipt at the printed precision, one 'complete' line "
          f"with the receipt's tensors_sha256, no line about resuming: {logok} of 12; other lines: {lognote if lognote else 'none'}")
    ctx["logok"] = logok

    # the launch log and processes at once
    ll = show(f"{NEW}logs/LAUNCHES.txt").decode().splitlines()
    starts = {l.split()[2]: utc(l.split()[0]) for l in ll if l.split()[1] == "start"}
    exits = [(l.split()[3], l.split()[2], utc(l.split()[0])) for l in ll if l.split()[1] == "exit"]
    resumes = [l for l in ll if "resume" in l or "interrupted" in l]
    head = [l for l in ll if "driver3 begin" in l][0].split("HEAD ")[1].split(";")[0]
    print(f"  launch log: {len(starts)} starts, {len(exits)} exits with codes {collections.Counter(e[1] for e in exits)}, resumptions or moves to interrupted/: {len(resumes)}; "
          f"first line {hms(utc(ll[0].split()[0]))}, last line {hms(utc(ll[-1].split()[0]))}; driver HEAD {head[:8]}")
    print(f"  the exit lines carry the times at which the driver waited on each process in launch order ({sorted(set(hms(e[2]) for e in exits))}), not the times at which the runs ended")
    ctx["first_start"] = min(starts.values())
    det = []
    for seed in NEWSEEDS:
        for H, fp in SETTINGS:
            for rule in "SB":
                n = f"e52_{rule}_h{H}_f{fp}_lr0.003_{seed}"; s = RUNS[(seed, rule, H, fp, 0.003)]
                for key in "AR":
                    if s[key + "80"] is not None:
                        x = next(x for x in s["r"]["curve"] if x["step"] == s[key + "80"] + 512)
                        det.append((starts[n] + x["wall"], f"{n} on {key}"))
    ctx["det"] = min(det)
    iv = []
    for seed in NEWSEEDS:
        for H, fp in SETTINGS:
            for rule in "SB":
                n = f"e52_{rule}_h{H}_f{fp}_lr0.003_{seed}"; r = RAW[(seed, rule, H, fp, 0.003)]
                iv.append((starts[n], starts[n] + r["wall"], n))
    ev = sorted([(a, 1) for a, b, n in iv] + [(b, -1) for a, b, n in iv])
    cur = mx = 0
    for _, d in ev:
        cur += d; mx = max(mx, cur)
    last_end = max(b for a, b, n in iv)
    print(f"  processes at once, from the launch times and each receipt's wall clock: at most {mx}; the last run ended at about {hms(last_end)} "
          f"({max(iv, key=lambda t: t[1])[2]})")
    commits = collections.Counter(RAW[(sd, ru, H, fp, 0.003)]["provenance"]["git"]["commit"][:8] for sd in NEWSEEDS for H, fp in SETTINGS for ru in "SB")
    print(f"  commits recorded in the 12 receipts (the HEAD when each run started): {dict(commits)}")
    ctx["launch"] = dict(starts=len(starts), exits=len(exits), codes=collections.Counter(e[1] for e in exits), resumes=len(resumes),
                         first=hms(utc(ll[0].split()[0])), last=hms(utc(ll[-1].split()[0])), head=head, mx=mx, commits=commits)

    # cost
    cpu = {k: RAW[k]["cpu_updates"] + RAW[k]["cpu_scoring"] for k in RAW if k[0] in NEWSEEDS and k[4] == 0.003}
    tot = sum(cpu.values()) / 3600
    upd = sum(RAW[k]["cpu_updates"] for k in cpu) / 3600
    print(f"  CPU: {tot:.3f} CPU-hours for the 12 runs ({upd:.3f} updates, {tot - upd:.3f} scoring); by width: "
          f"half {sum(v for k, v in cpu.items() if k[2] == 256) / 3600:.2f}, present {sum(v for k, v in cpu.items() if k[2] == 512) / 3600:.2f}")
    ctx["cpu"] = tot

    # files changed between the worker's commit before decision 19 and the report's commit
    allb, allw = names("", BASE), names("", WREF)
    common = sorted(set(allb) & set(allw))
    ib, iw = oids(BASE, common), oids(WREF, common)
    changed = [p for p in common if ib[p] != iw[p]]
    added = sorted(set(allw) - set(allb)); removed = sorted(set(allb) - set(allw))
    print(f"  files between {BASE[:8]} (before decision 19) and {WREF[:8]}: {len(allb)} and {len(allw)}; added {len(added)}, removed {len(removed)}, "
          f"changed {len(changed)} {changed if changed else ''}")
    added_out = [a for a in added if not a.startswith(NEW)]
    print(f"  added outside exp45/e52_seeds: {added_out}")
    ctx["files"] = dict(added=len(added), removed=len(removed), changed=changed, added_out=added_out)
    clock("section 1")


# ================================================================ 2
def counts_of(RUNS, seed, H, fp, rates=RATES2):
    C = {}
    for rule in "SB":
        for rate in rates:
            for key in "RA":
                s = RUNS.get((seed, rule, H, fp, rate))
                if s is not None:
                    C[(rule, rate, key)] = s[key + "80"]
    return C, DOSE[H] - 512


def section2(ctx):
    print("================ 2  figures of E52_SEEDS_TABLES.md, of the report and of the message, recomputed")
    RUNS = ctx["RUNS"]
    rep = show(REPORT).decode(); ctx["report"] = rep
    tabs = show(NEW + "E52_SEEDS_TABLES.md").decode()
    TR, TT = md_tables(rep), md_tables(tabs)

    RD = ctx["RD"]      # the readings and margins of section 3

    def key_of(row):
        return (int(row[0]), {"half": 256, "present": 512}[row[1].split(",")[0]], round(float(row[1].split(",")[1]) * 100))

    # Table 1: every run (report section 3 and tables Table 1)
    def table1(src, T):
        hdr, rows = find_table(T, "Seed | Width, f | Rule | Rate | Dose")
        n = 0
        for row in rows:
            seed, H, fp = key_of(row); rule = row[2]; rate = float(row[3])
            s = RUNS[(seed, rule, H, fp, rate)]
            lab = f"{seed} {rule} {WIDTH[H]} f{fp} {rate:g}"
            mine = {4: f"{s['dose']:,}", 5: fu(s["A80"]), 6: fu(s["R80"]), 7: f"{s['A_final']:.2f}", 8: f"{s['R_final']:.2f}", 9: f"{s['T_final']:.2f}"}
            for j, v in mine.items():
                cmp(src, f"{lab}: {hdr[j]}", row[j], v, nums(row[j]) == nums(v) and (row[j] == "–") == (v == "–")); n += 1
            if s["has_len"]:
                mv = [f"{s['len_last']:.1f}", f"{s['len_mean3']:.1f}", f"{s['len_lo']:.1f}", f"{s['len_hi']:.1f}"]
                rv = nums(row[10])
                for i, nm in enumerate(("length last", "length mean of 3", "length low of 12", "length high of 12")):
                    cmp(src, f"{lab}: {nm}", rv[i] if i < len(rv) else "missing", mv[i]); n += 1
                rv = [x.strip() for x in row[11].split("/")]
                for i, gname in enumerate(GROUPS):
                    cmp(src, f"{lab}: norm {gname}", rv[i] if i < len(rv) else "missing", f"{s['norms'][gname]:.1f}"); n += 1
            else:
                cmp(src, f"{lab}: length test", row[10], "not scored (E41)", row[10] == "not scored (E41)" and not s["has_len"]); n += 1
                cmp(src, f"{lab}: norms", row[11], "not recorded (E41)", row[11] == "not recorded (E41)" and s["norms"] is None); n += 1
            res = s["r"].get("resumed")
            cmp(src, f"{lab}: resumed at", row[12], "–" if not res else str(res)); n += 1
            cmp(src, f"{lab}: receipt", row[13].strip("`"), s["path"][len(M):]); n += 1
        print(f"  {src}: {len(rows)} rows, {n} figures compared")
        return len(rows)
    ctx["t1rows"] = table1("2 tables, Table 1", TT)
    table1("2 report, section 3", TR)

    # Table 2: readings with margins (report section 4 and tables Table 2)
    cell_re = re.compile(r"^(.*?) \((none within 200|(\d+) → (.*))\)$")

    def cmp_reading(src, lab, text, seed, H, fp, name):
        rd = RD[(seed, H, fp)]
        m = cell_re.match(text.strip())
        word, kk, to = (m.group(1), m.group(3), m.group(4)) if m else (text, None, None)
        mine_word = rd["base"][name]
        mw = {"yes": "yes", "no": "no"}.get(mine_word, mine_word)
        cmp(src, f"{lab}: reading", word, mw)
        mg = rd["mg"][name]
        cmp(src, f"{lab}: margin in rows", "none within 200" if kk is None else kk, "none within 200" if mg is None else str(mg[0]))
        if mg is not None and to is not None:
            cmp(src, f"{lab}: reading it would become", to, " or ".join(sorted(mg[1])), to in mg[1])
        elif mg is None and kk is None:
            cmp(src, f"{lab}: reading it would become", "–", "–")
    lag_re = re.compile(r"^(.*?) → (.*?): (.*?) \((\d+|none within 200)(?: → (.*))?\)$")

    def cmp_lag(src, lab, text, seed, H, fp, key):
        rd = RD[(seed, H, fp)]; L = rd["lagR"] if key == "R" else rd["lagA"]
        m = lag_re.match(text.strip())
        cmp(src, f"{lab}: lag at 0.001", m.group(1), lagtext(L["at"]))
        cmp(src, f"{lab}: lag at own rates", m.group(2), lagtext(L["own"]))
        cmp(src, f"{lab}: reading", m.group(3), L["cls"])
        mg = rd["mg"][f"lag {key}"]
        cmp(src, f"{lab}: margin in rows", m.group(4), "none within 200" if mg is None else str(mg[0]))
        if mg is not None:
            cmp(src, f"{lab}: reading it would become", m.group(5), " or ".join(sorted(mg[1])), m.group(5) in mg[1])

    def table2(src, T):
        hdr, rows = find_table(T, "Seed | Width, f | Best rate on R")
        before = len(COMP)
        for row in rows:
            seed, H, fp = key_of(row)
            lab0 = f"{seed} {WIDTH[H]} f{fp}"
            for j, (n1, n2) in ((2, ("best R S", "best R B")), (3, ("best A S", "best A B")), (4, ("vs S R", "vs S A")), (5, ("vs B R", "vs B A"))):
                parts = row[j].split("; ")
                cmp_reading(src, f"{lab0} {hdr[j]} (first)", parts[0], seed, H, fp, n1)
                cmp_reading(src, f"{lab0} {hdr[j]} (second)", parts[1], seed, H, fp, n2)
            cmp_reading(src, f"{lab0} S gains", row[6], seed, H, fp, "gains S")
            cmp_reading(src, f"{lab0} B gains", row[7], seed, H, fp, "gains B")
            cmp_lag(src, f"{lab0} lag on R", row[8], seed, H, fp, "R")
            cmp_lag(src, f"{lab0} lag on A", row[9], seed, H, fp, "A")
        print(f"  {src}: {len(rows)} rows, {len(COMP) - before} figures compared")
    table2("2 tables, Table 2", TT)
    table2("2 report, section 4", TR)

    # Table 3 of the tables file and the report's table of section 5
    rmap = {"S gains": "gains S", "B gains": "gains B", "lag R": "lag R", "lag A": "lag A", "The sweep gains": "gains S",
            "Back-propagation gains": "gains B", "Lag on R": "lag R", "Lag on A": "lag A"}
    hdr, rows = find_table(TT, "Width, f | Reading | 73129")
    for row in rows:
        H = {"half": 256, "present": 512}[row[0].split(",")[0]]; fp = round(float(row[0].split(",")[1]) * 100)
        for j, seed in ((2, 73129), (3, 116089), (4, 93683)):
            cmp("2 tables, Table 3", f"{row[0]} {row[1]} {seed}", row[j], RD[(seed, H, fp)]["base"][rmap[row[1]]])
    hdr, rows = find_table(TR, "Width, f | Reading | 73129")
    sec5 = re.compile(r"^(?:(.*?) → (.*?), )?(.*?) \((\d+|none within 200)\)$")
    for row in rows:
        H = {"half": 256, "present": 512}[row[0].split(",")[0]]; fp = round(float(row[0].split(",")[1]) * 100)
        name = rmap[row[1]]
        lab0 = f"{row[0]} {row[1]}"
        cls_of = {}
        for j, seed in ((2, 73129), (3, 116089), (4, 93683)):
            m = sec5.match(row[j]); rd = RD[(seed, H, fp)]
            mine = rd["base"][name]; cls_of[seed] = mine
            word = m.group(3)
            cmp("2 report, section 5 table", f"{lab0} {seed}: reading", word, mine, word == mine or (word == "the same" and mine == SAME))
            mg = rd["mg"][name]
            cmp("2 report, section 5 table", f"{lab0} {seed}: margin", m.group(4), "none within 200" if mg is None else str(mg[0]))
            if m.group(1):
                L = rd["lagR"] if name == "lag R" else rd["lagA"]
                cmp("2 report, section 5 table", f"{lab0} {seed}: lag at 0.001", m.group(1), lagtext(L["at"]))
                cmp("2 report, section 5 table", f"{lab0} {seed}: lag at own rates", m.group(2), lagtext(L["own"]))
        cnt = row[5]
        if name.startswith("gains"):
            cmp("2 report, section 5 table", f"{lab0}: count", cnt, f"{sum(v == 'yes' for v in cls_of.values())} of 3")
        else:
            cat = cnt.split(",")[0]
            sets = {"the same": {SAME}, "smaller": {"smaller"}, "smaller or gone": {"smaller", "gone"}}[cat]
            cmp("2 report, section 5 table", f"{lab0}: count ({cat})", cnt, f"{cat}, {sum(v in sets for v in cls_of.values())} of 3")
    sentences(ctx)
    message(ctx)
    clock("section 2")


def sentences(ctx):
    RUNS, RD, RAW = ctx["RUNS"], ctx["RD"], ctx["RAW"]
    c = lambda label, reported, mine, eq=None: cmp("2 report, sentences", label, reported, mine, eq)
    L = ctx["launch"]
    new = [k for k in RAW if k[0] in NEWSEEDS and k[4] == 0.003]
    c("1 runs complete / diverged / with a resumption record", "12 / 0 / 0",
      f"{sum(RAW[k]['complete'] for k in new)} / {sum(RAW[k]['diverged'] for k in new)} / {sum('resumed' in RAW[k] for k in new)}")
    c("1 runs made at commit (the driver's HEAD in the launch log)", "3abe1674", L["head"][:8])
    h = RAW[new[0]]["provenance"]["host"]
    c("1 host: cores, Python, torch", "4 3.13.18 2.15.0+cpu", f"{h['cores']} {h['python']} {h['torch']}",
      all(RAW[k]["provenance"]["host"]["cores"] == 4 and RAW[k]["provenance"]["host"]["python"] == "3.13.18" for k in new))
    c("1 at most three processes at once", "3", str(L["mx"]), L["mx"] <= 3)
    c("1 sha256 of exp52_resume.py (prefix)", "6531525f8bffec5f", ctx["srcsha"]["exp52_resume.py"][:16])
    c("1 exp52_resume.py unchanged since the first seed (same blob as at cb5cacfc)", "unchanged", "unchanged" if ctx["src_same_pre"] else "changed")
    c("1 checkpoint every (updates)", "4,096", f"{RAW[new[0]]['checkpoint_every']:,}", all(RAW[k]["checkpoint_every"] == 4096 for k in new))
    c("1 sha256 of exp52_drive3.py (prefix)", "1401ba576d687ab1", sha(show(M + "exp52_drive3.py"))[:16])
    c("1 sha256 of exp52_seeds_tables.py (prefix)", "115e1fda949da2d7", sha(show(M + "exp52_seeds_tables.py"))[:16])
    c("1 no existing file was modified (files changed since 2d36fcfa)", "0", str(len(ctx["files"]["changed"]) + ctx["files"]["removed"]))
    doses = {H: sorted({RAW[k]["dose"] for k in new if k[2] == H}) for H in (256, 512)}
    c("1 dose at half width / present width (every run)", "40,960 / 12,288", " / ".join(", ".join(f"{d:,}" for d in doses[H]) for H in (256, 512)))
    c("1 number of runs", "12", str(len(new)))
    c("1 E41's receipts record neither the length test nor the norms", "true", str(ctx["e41_no"]).lower())
    c("1 on seed 93683 exp52_rate.py reproduced E41's runs (of the three settings, both rules)", "6 of 6", f"{ctx['e93_eq'][0]} of {ctx['e93_eq'][1]}")
    c("1 launch log: starts, exits with code 0, resumptions", "12 12 0", f"{L['starts']} {L['codes'].get('0', 0)} {L['resumes']}")
    c("1 launch log: first and last line", "14:24:54 16:30:14", f"{L['first']} {L['last']}")
    c("1 CPU-hours for the 12 runs", "5.94", f"{ctx['cpu']:.2f}")
    q = ctx["q93"]
    c("1 settings that qualify on seed 93683 by the brief's rule (four rates)", "half f50, half f25, present f25", ", ".join(q))
    m93 = ctx["m93"]
    r4 = ctx["r4"]
    c("1 seed 93683 at four rates: half f50 (S gains, B gains); half f25 (S gains, lag R); present f25 (S gains, lag R, lag A)",
      "yes yes; yes smaller; yes smaller smaller", f"{r4[(256, 50)]['gains S']} {r4[(256, 50)]['gains B']}; {r4[(256, 25)]['gains S']} {r4[(256, 25)]['lag R']}; "
      f"{r4[(512, 25)]['gains S']} {r4[(512, 25)]['lag R']} {r4[(512, 25)]['lag A']}")
    c("5 seed 93683: readings at two rates equal those at four rates (differences)", "0", str(len(ctx["diff4"])))
    c("2 seed 93683 margins of the readings the verification names (lag R present f25, B gains present f25, lag R half f25, B gains half f25)",
      "1 1 2 3", " ".join(str(x) for x in m93[:4]))
    c("2 seed 93683 margins of the sweep's gains (half f50, half f25, present f25)", "6 19 10", " ".join(str(x) for x in m93[4:]))
    rng = {k: RUNS[k]["len_hi"] - RUNS[k]["len_lo"] for k in new}
    kmax = max(rng, key=rng.get)
    c("3 range of the length test over the last twelve rows, the 12 new runs: smallest and largest", "8.7 21.7", f"{min(rng.values()):.1f} {max(rng.values()):.1f}")
    c("3 the run with the largest range", "73129 S half f25 (69.3 to 91.0)", f"{kmax[0]} {kmax[1]} {WIDTH[kmax[2]]} f{kmax[3]} ({RUNS[kmax]['len_lo']:.1f} to {RUNS[kmax]['len_hi']:.1f})")
    rng18 = {k: RUNS[k]["len_hi"] - RUNS[k]["len_lo"] for k in RUNS if RUNS[k]["has_len"] and k[0] in SEEDS and k[4] in RATES2 and (k[2], k[3]) in SETTINGS}
    print(f"  section 3 of the report: over all {len(rng18)} runs of its table with a length test (seed 93683 included) the range is "
          f"{min(rng18.values()):.1f} to {max(rng18.values()):.1f} points (smallest: {min(rng18, key=rng18.get)})")
    ctx["rng18"] = (min(rng18.values()), min(rng18, key=rng18.get), len(rng18))
    # section 5
    a, b = RD[(73129, 512, 25)], RD[(116089, 512, 25)]
    c("5 lag on R at own rates, present f25, seeds 73129 and 116089", "0.87 1.00", f"{a['lagR']['own'][0]:.2f} {b['lagR']['own'][0]:.2f}")
    c("5 seed 73129 B present f25 R at 0.001 and 0.003; best", "7,680 7,424 0.001", f"{fu(a['C'][('B', 0.001, 'R')])} {fu(a['C'][('B', 0.003, 'R')])} {a['base']['best R B']}")
    c("5 seed 73129 B present f25 A at 0.003 against 0.001; reading", "8,192 4,352 slower", f"{fu(a['C'][('B', 0.003, 'A')])} {fu(a['C'][('B', 0.001, 'A')])} {a['base']['vs B A']}")
    c("5 seed 116089 present f25 at 0.003: S and B on R, S and B on A", "4,608 4,608 4,096 4,096",
      f"{fu(b['C'][('S', 0.003, 'R')])} {fu(b['C'][('B', 0.003, 'R')])} {fu(b['C'][('S', 0.003, 'A')])} {fu(b['C'][('B', 0.003, 'A')])}")
    c("5 the sweep does not reach 80 on R at 0.001 at present f25 (73129, 116089)", "not reached not reached",
      f"{fu(a['C'][('S', 0.001, 'R')])} {fu(b['C'][('S', 0.001, 'R')])}".replace("–", "not reached"))
    c("5 seed 116089 'gone' (R and A) one row from 'smaller'", "1 smaller 1 smaller",
      f"{b['mg']['lag R'][0]} {'/'.join(sorted(b['mg']['lag R'][1]))} {b['mg']['lag A'][0]} {'/'.join(sorted(b['mg']['lag A'][1]))}")
    c("5 seed 73129 'gone' four rows from 'not decided' on R, two from 'smaller' on A", "4 not decided 2 smaller",
      f"{a['mg']['lag R'][0]} {'/'.join(sorted(a['mg']['lag R'][1]))} {a['mg']['lag A'][0]} {'/'.join(sorted(a['mg']['lag A'][1]))}")
    for (H, fp), words in (((256, 50), {73129: (SAME, SAME), 116089: (SAME, SAME), 93683: (SAME, SAME)}),
                           ((256, 25), {93683: ("smaller", SAME), 116089: ("smaller", "smaller"), 73129: (SAME, SAME)}),
                           ((512, 25), {73129: ("gone", "gone"), 116089: ("gone", "gone"), 93683: ("smaller", "smaller")})):
        for sd, (wr, wa) in words.items():
            rd = RD[(sd, H, fp)]
            c(f"5 the brief's sentence, {WIDTH[H]} f{fp}, seed {sd}: lag on R, on A", f"{wr}; {wa}", f"{rd['lagR']['cls']}; {rd['lagA']['cls']}")
    d = RD[(73129, 256, 25)]
    c("5 seed 73129 half f25: B gains; lag on R at 0.001 and own; margin to 'larger'", "yes 2.12 2.90 3 larger",
      f"{d['base']['gains B']} {d['lagR']['at'][0]:.2f} {d['lagR']['own'][0]:.2f} {d['mg']['lag R'][0]} {'/'.join(sorted(d['mg']['lag R'][1]))}")
    # section 6
    ex = ctx["expl"]
    c("6 seed 73129: the sweep gains at", "half f25, present f25", ", ".join(ex[73129]["sg"]))
    c("6 seed 73129 present f25: B gains; lag (R, A)", "no gone gone", f"{RD[(73129, 512, 25)]['base']['gains B']} {RD[(73129, 512, 25)]['lagR']['cls']} {RD[(73129, 512, 25)]['lagA']['cls']}")
    c("6 seed 73129 half f25: B gains; lag (R, A)", f"yes {SAME} {SAME}", f"{d['base']['gains B']} {d['lagR']['cls']} {d['lagA']['cls']}")
    c("6 seed 116089: the sweep gains at", "half f50, half f25, present f25", ", ".join(ex[116089]["sg"]))
    c("6 seed 116089 half f25 and present f25: B gains; lag on R", "no smaller no gone",
      f"{RD[(116089, 256, 25)]['base']['gains B']} {RD[(116089, 256, 25)]['lagR']['cls']} {RD[(116089, 512, 25)]['base']['gains B']} {RD[(116089, 512, 25)]['lagR']['cls']}")
    for sd in SEEDS:
        c(f"6 the explanation refuted by the brief's test, seed {sd} (three settings, two rates)", "not refuted", "refuted" if ex[sd]["refuted"] else "not refuted")
    c("6 seed 93683 half f25: lag on R; margin to 'the same'", "smaller 2", f"{RD[(93683, 256, 25)]['lagR']['cls']} {RD[(93683, 256, 25)]['mg']['lag R'][0]}",
      RD[(93683, 256, 25)]['lagR']['cls'] == "smaller" and RD[(93683, 256, 25)]['mg']['lag R'][0] == 2 and SAME in RD[(93683, 256, 25)]['mg']['lag R'][1])
    c("6 E41's lags on R at 0.001, half f25, seeds 73129, 93683, 116089", "2.12 3.31 3.31",
      " ".join(f"{RD[(sd, 256, 25)]['lagR']['at'][0]:.2f}" for sd in (73129, 93683, 116089)))
    # section 7
    c("7 seed 73129 S half f50 final R at 0.003 and 0.001", "85.05 93.82", f"{RUNS[(73129, 'S', 256, 50, 0.003)]['R_final']:.2f} {RUNS[(73129, 'S', 256, 50, 0.001)]['R_final']:.2f}")
    c("7 seed 73129 B present f25 final R at 0.003 and 0.001", "81.57 83.68", f"{RUNS[(73129, 'B', 512, 25, 0.003)]['R_final']:.2f} {RUNS[(73129, 'B', 512, 25, 0.001)]['R_final']:.2f}")
    c("8 files: 12 runs, 24 receipt and weight files", "24", str(ctx["inv"]["rec"] + ctx["inv"]["pts"]))
    c("8 new sources (exp52_drive3.py, exp52_seeds_tables.py) and the report outside exp45/e52_seeds", "3", str(len(ctx["files"]["added_out"])))


def message(ctx):
    RD, RAW, L = ctx["RD"], ctx["RAW"], ctx["launch"]
    c = lambda label, reported, mine, eq=None: cmp("2 message", label, reported, mine, eq)
    new = [k for k in RAW if k[0] in NEWSEEDS and k[4] == 0.003]
    c("runs at 0.003; exits with code 0; diverged; resumed", "12 12 0 0", f"{len(new)} {L['codes'].get('0', 0)} {sum(RAW[k]['diverged'] for k in new)} {sum('resumed' in RAW[k] for k in new) + L['resumes']}")
    c("CPU-hours", "5.94", f"{ctx['cpu']:.2f}")
    c("assertions held: groups, digest at 4,096, length test at 12 letters (failures)", "0",
      str(sum(len(v) for t, v in ctx["bad"].items() if t.startswith(("groups", "digest 4096", "length 12")))))
    m93 = ctx["m93"]
    c("seed 93683 reproduces the reviewer's figures", "1 1 2 3 6 19 10", " ".join(str(x) for x in m93))
    for (H, fp), tags in (((256, 50), ("the same", "the same", "the same")), ((256, 25), ("the same", "smaller", "smaller")), ((512, 25), ("gone", "gone", "smaller"))):
        got = []
        for sd, t in zip((73129, 116089, 93683), tags):
            rd = RD[(sd, H, fp)]
            got.append(rd["lagR"]["cls"].replace(SAME, "the same"))
        c(f"lag on R at own rates, {WIDTH[H]} f{fp}, seeds 73129 / 116089 / 93683: words", " / ".join(tags), " / ".join(got))
        c(f"lag on R, {WIDTH[H]} f{fp}: margins", {(256, 50): "9 5 7", (256, 25): "3 10 2", (512, 25): "4 1 1"}[(H, fp)],
          " ".join(str(RD[(sd, H, fp)]["mg"]["lag R"][0]) for sd in (73129, 116089, 93683)))
    c("lag on R values: 73129 half f25, 116089 half f25", "2.12→2.90 3.31→1.43",
      " ".join(f"{RD[(sd, 256, 25)]['lagR']['at'][0]:.2f}→{RD[(sd, 256, 25)]['lagR']['own'][0]:.2f}" for sd in (73129, 116089)))
    c("lag on R values: 73129 present f25, 116089 present f25", "≥1.53→0.87 ≥2.00→1.00",
      " ".join(f"≥{math.floor(RD[(sd, 512, 25)]['lagR']['at'][0] * 100) / 100:.2f}→{RD[(sd, 512, 25)]['lagR']['own'][0]:.2f}" for sd in (73129, 116089)))
    sg = [(sd, H, fp) for sd in SEEDS for H, fp in SETTINGS if RD[(sd, H, fp)]["base"]["gains S"] == "yes"]
    c("the sweep gains in ... of 9 seed-settings", "8", str(len(sg)))
    c("the sweep does not gain at half f50 on seed 73129; margin", "no 4", f"{RD[(73129, 256, 50)]['base']['gains S']} {RD[(73129, 256, 50)]['mg']['gains S'][0]}")
    c("B gains at half f50 on all three seeds", "yes yes yes", " ".join(RD[(sd, 256, 50)]["base"]["gains B"] for sd in (73129, 116089, 93683)))
    c("B gains at half f25 only on 73129; margin", "yes no no 14", " ".join(RD[(sd, 256, 25)]["base"]["gains B"] for sd in (73129, 116089, 93683)) + f" {RD[(73129, 256, 25)]['mg']['gains B'][0]}")
    c("B never gains at present f25", "no no no", " ".join(RD[(sd, 512, 25)]["base"]["gains B"] for sd in (73129, 116089, 93683)))
    c("the explanation not refuted on any seed", "0 refuted", f"{sum(ctx['expl'][sd]['refuted'] for sd in SEEDS)} refuted")


# ================================================================ 3
def section3(ctx):
    print("================ 3  the readings in the brief's words, seed by seed")
    RUNS = ctx["RUNS"]
    RD = {}
    for seed in SEEDS:
        for H, fp in SETTINGS:
            C, D = counts_of(RUNS, seed, H, fp)
            base, mg = margins(C, D)
            RD[(seed, H, fp)] = dict(C=C, D=D, base=base, mg=mg, lagR=lag(C, D, "R"), lagA=lag(C, D, "A"))
    ctx["RD"] = RD
    print("  counts to 80 (R; A) at 0.001 and 0.003, and the readings (margin in rows; readings it can become at that margin):")
    for seed in SEEDS:
        for H, fp in SETTINGS:
            rd = RD[(seed, H, fp)]; C = rd["C"]
            cs = "; ".join(f"{ru} {r:g}: R {fu(C[(ru, r, 'R')])}, A {fu(C[(ru, r, 'A')])}" for ru in "SB" for r in RATES2)
            print(f"    {seed} {WIDTH[H]} f{fp} (last recordable {rd['D']:,}): {cs}")
            for n, v in rd["base"].items():
                mg = rd["mg"][n]
                print(f"        {n}: {v} ({'none within 200' if mg is None else str(mg[0]) + ' → ' + ' or '.join(sorted(mg[1]))})")
            for key in "RA":
                L = rd["lagR"] if key == "R" else rd["lagA"]
                at, own = L["at"], L["own"]
                print(f"        lag on {key}: at 0.001 {lagtext(at)} (exact {at[0]:.4f}{'' if at[2] == 'value' else ', a ' + at[2] + ' bound'}); at own rates {lagtext(own)} "
                      f"({own[0]:.4f}; S at {L['sbest']['word']}, B at {L['bbest']['word']}); two thirds of the lag at 0.001 {2 / 3 * at[0]:.4f}; reading {L['cls']}")
    # the brief's sentence, seed by seed
    print("  the brief's sentence ('at this width and fraction the sweep's lag behind back-propagation is [...] when each rule is run at its own best"
          " learning rate among those tried', here 0.001 and 0.003), seed by seed, on R and on A:")
    for H, fp in SETTINGS:
        print(f"    {WIDTH[H]} f{fp}: " + "; ".join(f"{sd}: R {RD[(sd, H, fp)]['lagR']['cls']}, A {RD[(sd, H, fp)]['lagA']['cls']}" for sd in SEEDS))
    # readings on one or two rows
    thin = [(sd, H, fp, n, v, RD[(sd, H, fp)]["mg"][n]) for sd in SEEDS for H, fp in SETTINGS for n, v in RD[(sd, H, fp)]["base"].items()
            if RD[(sd, H, fp)]["mg"][n] is not None and RD[(sd, H, fp)]["mg"][n][0] <= 2]
    print(f"  readings with a margin of one or two rows: {len(thin)} of {sum(len(RD[k]['base']) for k in RD)}")
    for sd, H, fp, n, v, mg in thin:
        print(f"    {sd} {WIDTH[H]} f{fp} {n}: {v} ({mg[0]} → {' or '.join(sorted(mg[1]))})")
    ctx["thin"] = thin
    key_thin = [(sd, H, fp, n) for sd, H, fp, n, v, mg in thin if n.startswith(("gains", "lag"))]
    print(f"  of which gains or lags: {len(key_thin)}: {[f'{sd} {WIDTH[H]} f{fp} {n}' for sd, H, fp, n in key_thin]}")
    # the explanation under test
    expl = {}
    for sd in SEEDS:
        sg = [(H, fp) for H, fp in SETTINGS if RD[(sd, H, fp)]["base"]["gains S"] == "yes"]
        cond = {(H, fp): (RD[(sd, H, fp)]["base"]["gains B"], RD[(sd, H, fp)]["lagR"]["cls"], RD[(sd, H, fp)]["lagA"]["cls"]) for H, fp in sg}
        small = ("smaller", "gone")
        ref_R = (not sg) or all(b == "yes" and lr not in small for b, lr, la in cond.values())
        ref_A = (not sg) or all(b == "yes" and la not in small for b, lr, la in cond.values())
        ref_both = (not sg) or all(b == "yes" and lr not in small and la not in small for b, lr, la in cond.values())
        saving = [(H, fp) for (H, fp), (b, lr, la) in cond.items() if not (b == "yes" and lr not in small)]
        sav_m = {f"{WIDTH[H]} f{fp}": (RD[(sd, H, fp)]["mg"]["gains S"][0], RD[(sd, H, fp)]["mg"]["gains B"][0] if RD[(sd, H, fp)]["mg"]["gains B"] else None,
                                       RD[(sd, H, fp)]["mg"]["lag R"][0]) for H, fp in saving}
        expl[sd] = dict(sg=[f"{WIDTH[H]} f{fp}" for H, fp in sg], refuted=ref_R and ref_A, ref_R=ref_R, ref_A=ref_A, ref_both=ref_both)
        print(f"  explanation under test, seed {sd}: the sweep gains at {expl[sd]['sg']}; there (B gains, lag on R, lag on A): "
              f"{ {f'{WIDTH[H]} f{fp}': v for (H, fp), v in cond.items()} }; refuted with the lag read on R: {ref_R}, on A: {ref_A}; "
              f"settings that keep it unrefuted on R, with the margins of the sweep's gain, of back-propagation's reading and of the lag on R: {sav_m}")
    ctx["expl"] = expl

    # the margin of the test itself: the fewest rows by which one count, in one setting, must move for the verdict to change
    def refuted(per, score):
        sg = [st for st, r in per.items() if r["gains S"] == "yes"]
        small = ("smaller", "gone")
        ok = lambda r: r["gains B"] == "yes" and all(r["lag " + k] not in small for k in score)
        return (not sg) or all(ok(per[st]) for st in sg)
    EM = {}
    for sd in SEEDS:
        for score in ("R", "RA"):
            per = {st: RD[(sd,) + st]["base"] for st in SETTINGS}
            base = refuted(per, score); found = None
            for k in range(1, KMAX + 1):
                for st in SETTINGS:
                    C, D = RD[(sd,) + st]["C"], RD[(sd,) + st]["D"]
                    for var, cnt in C.items():
                        cand = ([D - ROW * (k - 1)] if D - ROW * (k - 1) >= ROW else []) if cnt is None else \
                               [None if v > D else v for v in (cnt + ROW * k, cnt - ROW * k) if v >= ROW]
                        for v in cand:
                            C2 = dict(C); C2[var] = v
                            per2 = dict(per); per2[st] = readings(C2, D)
                            if refuted(per2, score) != base and found is None:
                                found = (k, f"{WIDTH[st[0]]} f{st[1]} {var[0]} {var[1]:g} {var[2]} {fu(cnt)} -> {fu(v)}")
                if found:
                    break
            EM[(sd, score)] = (base, found)
            print(f"  the test of the explanation, seed {sd}, lag read on {'R' if score == 'R' else 'R and A (refuted only if neither is smaller)'}: "
                  f"{'refuted' if base else 'not refuted'}; margin: {'none within 200 (no single count changes it)' if found is None else f'{found[0]} rows ({found[1]})'}")
    ctx["EM"] = EM
    cls = collections.Counter(RD[(sd, H, fp)]["lag" + k]["cls"] for sd in SEEDS for H, fp in SETTINGS for k in "RA")
    print(f"  the 18 lag readings (three seeds, three settings, R and A): {dict(cls)}")
    ctx["cls18"] = cls
    own = {f"{WIDTH[H]} f{fp} {k}": [f"{RD[(sd, H, fp)]['lag' + k]['own'][0]:.2f}" for sd in SEEDS] for H, fp in SETTINGS for k in "RA"}
    print(f"  the lag at the rules' own rates, seeds 73129, 116089, 93683: {own}")
    lower = []
    for sd in SEEDS:
        for H, fp in SETTINGS:
            for ru in "SB":
                a, b = RUNS[(sd, ru, H, fp, 0.003)], RUNS[(sd, ru, H, fp, 0.001)]
                if a["R_final"] < b["R_final"]:
                    lower.append(f"{sd} {ru} {WIDTH[H]} f{fp}: {a['R_final']:.2f} < {b['R_final']:.2f}")
    print(f"  runs at 0.003 whose final R (mean of the last three rows) is below the run at 0.001 of the same rule, setting and seed: {len(lower)} of 18: {lower}")
    ctx["lower"] = lower
    hl = {sd: (RD[(sd, 256, 25)]["base"]["gains S"], RD[(sd, 256, 25)]["base"]["gains B"], RD[(sd, 256, 25)]["lagR"]["cls"], RD[(sd, 256, 25)]["lagA"]["cls"]) for sd in SEEDS}
    print(f"  at the case the explanation names (half width, f = 0.25), (S gains, B gains, lag on R, lag on A): {hl}")
    # seed 93683 at four rates against two
    R4 = (0.0003, 0.001, 0.003, 0.01)
    diff4, q93 = [], []
    for H in (256, 512):
        for fp in (50, 25):
            C4, D = counts_of(RUNS, 93683, H, fp, rates=R4)
            r4 = {f"gains {ru}": "yes" if gains(C4, D, ru, R4) else "no" for ru in "SB"}
            r4.update({f"lag {k}": lag(C4, D, k, R4)["cls"] for k in "RA"})
            r4.update({f"best {k} {ru}": best(C4, D, ru, k, R4)["word"] for k in "RA" for ru in "SB"})
            if r4["gains S"] == "yes" or r4["gains B"] == "yes" or r4["lag R"] != SAME or r4["lag A"] != SAME:
                q93.append(f"{WIDTH[H]} f{fp}")
            ctx.setdefault("r4", {})[(H, fp)] = r4
            if (H, fp) in SETTINGS:
                r2 = RD[(93683, H, fp)]["base"]
                for n in r4:
                    if r4[n] != r2[n]:
                        diff4.append(f"{WIDTH[H]} f{fp} {n}: four rates {r4[n]}, two rates {r2[n]}")
            print(f"  seed 93683 {WIDTH[H]} f{fp} at four rates: gains S {r4['gains S']}, gains B {r4['gains B']}, lag R {r4['lag R']}, lag A {r4['lag A']}, "
                  f"best R S {r4['best R S']}, B {r4['best R B']}")
    print(f"  seed 93683: readings at the two rates that differ from the four-rate readings in the three settings: {diff4 if diff4 else 'none'}")
    print(f"  settings that qualify for the other seeds by the brief's rule on seed 93683 (either rule gains, or the lag changes class on R or A): {q93}")
    ctx["q93"], ctx["diff4"] = q93, diff4
    # the seed-93683 margins the verification named, under the worker's definition of a move (also moving unreached counts)
    r = lambda H, fp, n: RD[(93683, H, fp)]["mg"][n][0]
    ctx["m93"] = [r(512, 25, "lag R"), r(512, 25, "gains B"), r(256, 25, "lag R"), r(256, 25, "gains B"), r(256, 50, "gains S"), r(256, 25, "gains S"), r(512, 25, "gains S")]
    print(f"  seed 93683 margins (lag R present f25, B gains present f25, lag R half f25, B gains half f25; S gains half f50, half f25, present f25): {ctx['m93']}")
    clock("section 3")


# ================================================================ 4
def section4(ctx):
    print("================ 4  a sample of two weight files")
    RAW = ctx["RAW"]
    sample = [(73129, "S", 512, 25), (116089, "B", 512, 25)]
    out = []
    for sd, ru, H, fp in sample:
        fb = free_bytes()
        print(f"  free space on /home/user before reading: {fb / 1e9:.2f} GB")
        if fb < MIN_FREE:
            print("  stopped: less than 0.9 GB free"); break
        r = RAW[(sd, ru, H, fp, 0.003)]
        blob = show(NEW + r["weights"])
        W = load_pt(blob)
        ok_file, ok_t = sha(blob) == r["weights_sha256"], tensors_sha(W) == r["tensors_sha256"]
        nm = {g: float(sum(float((W[t].astype(np.float64) ** 2).sum()) for t in GROUP_TENSORS[g]) ** .5) for g in GROUPS}
        last = r["curve"][-1]
        dn = max(abs(nm[g] - last["norms"][g]) / last["norms"][g] for g in GROUPS)
        A, Rs, TA, LP, dig, th = panels(fp / 100, sd)
        net = Net(W)
        mine = dict(A={n: net.exact(it) for n, it in A.items()}, R={n: net.exact(it) for n, it in Rs.items()},
                    TA={n: net.exact(it) for n, it in TA.items()}, L={str(k): net.exact(it) for k, it in LP.items()})
        same = (mine["A"] == last["A_counts"], mine["R"] == last["R_counts"], mine["TA"] == last["taught_counts"], mine["L"] == last["length_counts"])
        print(f"  {sd} {ru} {WIDTH[H]} f{fp} 0.003 ({len(blob) / 1e6:.1f} MB): file sha256 as recorded {ok_file}; tensors_sha256 as recorded {ok_t}; "
              f"norms equal to the last row's (largest relative difference {dn:.1e}); R sample digest recomputed = receipt {dig == r['reduction_sample_sha256']}; "
              f"final-row counts A, R, taught, length recomputed by the numpy forward equal the receipt's: {same}")
        out.append((ok_file and ok_t, dn, all(same), dig == r["reduction_sample_sha256"]))
        del blob, W, net
    print(f"  free space on /home/user after: {free_bytes() / 1e9:.2f} GB")
    ctx["weights"] = out
    clock("section 4")


# ================================================================ 5
def section5(ctx):
    print("================ 5  form")
    files = [n for n in names(NEW) if not n.endswith(".pt")] + [REPORT, M + "exp52_drive3.py", M + "exp52_seeds_tables.py"]
    trailer = git_text(["log", "-1", "--format=%B", WREF])
    word = re.search(r"Acting-Agent:\s*([A-Za-z]+):", trailer).group(1)
    marks = {"@@": re.compile(r"@@"), "TODO": re.compile(r"TODO"), "TBD": re.compile(r"TBD"), "XX": re.compile(r"XX")}
    found = collections.Counter(); where = collections.defaultdict(list); mname = {}
    for f in files:
        t = show(f).decode("utf-8", "replace")
        for k, p in marks.items():
            n = len(p.findall(t))
            if n:
                found[k] += n; where[k].append(f.split("/")[-1])
        n = len(re.findall(re.escape(word), t, re.I))
        if n:
            mname[f.split("/")[-1]] = n
    print(f"  text files searched: {len(files)} (every committed file of exp45/e52_seeds but the weights, the report, the two new sources)")
    print(f"  unfilled marks: {dict(found) if found else 'none'} {dict(where) if where else ''}")
    by_kind = collections.Counter("report" if k == REPORT.split("/")[-1] else "sources" if k.endswith(".py") else "receipts" if k.endswith(".json") else "other" for k in mname)
    print(f"  the agent word of the commit trailer (not printed; {len(word)} letters): occurrences by file kind {dict(by_kind)}, "
          f"total {sum(mname.values())} in {len(mname)} files; in the report: {mname.get(REPORT.split('/')[-1], 0)}")
    rep = ctx["report"]
    in_rep = [m.start() for m in re.finditer(re.escape(word), rep, re.I)]
    ctxs = collections.Counter("branch name" if "/" in rep[max(0, i - 12):i] or rep[max(0, i - 1):i] == "`" else "other" for i in in_rep)
    print(f"  places in the report: {dict(ctxs)}")
    msgs = git_text(["log", "--format=%B%x00", f"{BASE}..{WREF}"]).split("\x00")
    msgs = [m for m in msgs if m.strip()]
    print(f"  commit messages since {BASE[:8]}: {len(msgs)}; with the word (in the trailer): {sum(bool(re.search(re.escape(word), m, re.I)) for m in msgs)}")
    ctx["form"] = dict(marks=dict(found), mname=mname, files=len(files), commits=len(msgs), rep=len(in_rep))
    # cadence of the commits during the runs
    ts = [int(x) for x in git_text(["log", "--format=%ct", f"{BASE}..{WREF}"]).split()]
    ts.sort()
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    print(f"  commits during the runs: {len(ts)}, from {hms(ts[0])} to {hms(ts[-1])}; largest gap between commits {max(gaps) / 60:.1f} minutes")
    ctx["gap"] = max(gaps) / 60
    # the readings script fixed before the runs
    same = oids(TABS_COMMIT, [M + "exp52_seeds_tables.py"])[M + "exp52_seeds_tables.py"] == oids(WREF, [M + "exp52_seeds_tables.py"])[M + "exp52_seeds_tables.py"]
    t_tabs = int(git_text(["log", "-1", "--format=%ct", TABS_COMMIT]).strip())
    det = ctx["det"]
    print(f"  exp52_seeds_tables.py committed at {hms(t_tabs)} ({TABS_COMMIT[:8]}), {t_tabs - utc_first(ctx):.0f} s after the first run's start at {ctx['launch']['first']}; "
          f"the earliest count to 80 of any new run was settled (its third row at or above 80 written) at about {hms(det[0])} ({det[1]}); unchanged since: {same}")
    ctx["tabs_fixed"] = same
    # decision 19 as quoted
    dec = show(DEC_FILE, DEC).decode()
    q = re.search(r'> "(.*?)"', ctx["report"], re.S).group(1)
    print(f"  the report's quotation of decision 19 is a verbatim part of the coordinator's message at {DEC[:8]}: {q in dec}")
    ctx["quote_ok"] = q in dec
    clock("section 5")


def utc_first(ctx):
    return ctx["first_start"]


# ================================================================ 6
def section6(ctx):
    print("================ 6  summary of the comparisons")
    by = collections.Counter(s for s, *_ in COMP); bad = [x for x in COMP if not x[4]]
    for s in sorted(by):
        print(f"  {s}: {by[s]} compared, {sum(1 for x in bad if x[0] == s)} differ")
    print(f"  figures compared: {len(COMP)}; agree: {len(COMP) - len(bad)}; differ: {len(bad)}")
    for s, lab, rep, mine, _ in bad:
        print(f"    DIFFERS [{s}] {lab}: reported {rep!r}, recomputed {mine!r}")
    # equal comparisons whose target was one of several possible readings
    multi = [x for x in COMP if "would become" in x[1] and " or " in x[3]]
    print(f"  margins at which a move can lead to more than one reading (the reported one is among them): {len(multi)}")
    for s, lab, rep, mine, eq in multi:
        print(f"    [{s}] {lab}: reported {rep!r}; possible {mine!r}")


def main():
    ctx = {}
    section1(ctx)
    section3(ctx)
    section2(ctx)
    section4(ctx)
    section5(ctx)
    section6(ctx)
    print(f"total time {time.time() - T0:.1f} s")


if __name__ == "__main__":
    main()
