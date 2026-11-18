"""Verification of the final report of E54 (the limited and the periodic network with fewer pairs taught; the worker's commit 87787a54).

Read-only. Nothing is trained, the training library is not used, and no file of the worker's branch is executed: every file is read with
`git show` into memory and listed with `git ls-tree -r --name-only`. Weight files are read one at a time into memory (never extracted to
disk) with an unpickler that refuses every class except those that describe a tensor; reading stops when less than 1.0 GB of disk would
remain. The script writes no file.

Sections (the numbering of the verification document):
  1  inventory, roster, completeness, settings and identities between runs
  2  every figure of the report's tables, of E54_TABLES.md and of the report's sentences, recomputed and set beside the reported figure
  3  the readings in the brief's fixed words, their distance from the rule's boundaries, the predictions, the record the report quotes
  4  the resumptions and check 6
  5  checks 1 to 5
  6  the weights: file and tensor hashes, the structure of F10 and P8, the forward recomputed with numpy
  7  the runner as text
  8  form: placeholders, the list of committed files, names of language models (counted, never written), the brief's revision

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I <path>/e54_verify.py
"""
import collections, hashlib, io, json, math, os, pickle, re, subprocess, sys, time, zipfile
from array import array
from datetime import datetime, timezone
from random import Random
import numpy as np

T0 = time.time()
sys.dont_write_bytecode = True

WREF = "87787a54c50eda2051630a805c09b74c98196065"      # the worker's commit of the report (verified)
WTAB = "e718483e"                                      # the commit whose file list the report's Appendix B gives
DREF = "7cabb13adc38753483d34dd5472fc3c84c4de675"      # the default branch at the worker's source commit
EREF = "44f91edcbccd0aa8b4e6033ab0bacd4a0b768bf6"      # E42's worker's commit named by the brief
BREF = "dc27dbb5"                                      # the reviewer's commit at which the report says the brief was read
M = "MOD97_LOCAL_RULE_20261106/"
E54 = M + "exp45/e54/"
QUAL = E54 + "qual/"
BRIEF = "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/TASK_E54_THE_LIMITED_AND_THE_PERIODIC_NETWORK_WITH_FEWER_PAIRS_20261117.md"
SEED = 73129
DOSE = 49152
P4DOSE = 122880
CONFIGS = ("G", "F10", "P8")
FRACS = (25, 15, 10)
MARGIN = {(25, "S"): 10, (25, "B"): 10, (15, "S"): 20, (15, "B"): 20, (10, "S"): 10, (10, "B"): 20}      # the brief, "How the result is stated"
MARGIN_SB = {25: 10, 15: 20, 10: 20}
EXPECT = {"G": dict(N=255, period_set="1", variant="step", wide_neighbourhood=False, changing=False, theta=None, weight_count=1124113),
          "F10": dict(N=10, period_set="1", variant="step", wide_neighbourhood=False, changing=False, theta=None, weight_count=935953),
          "P8": dict(N=10, period_set="8", variant="step", wide_neighbourhood=False, changing=True, theta=0.5, weight_count=936465)}
G0 = ["l1.weight", "l1.bias", "rnn.weight_ih_l0", "rnn.weight_hh_l0", "rnn.bias_ih_l0", "rnn.bias_hh_l0", "lm.weight", "lm.bias", "l2.weight", "l2.bias",
      "l3.weight", "l3.bias"]
MODEL_KEYS = ("book.weight", "l1.weight", "l1.bias", "rnn.weight_ih_l0", "rnn.weight_hh_l0", "rnn.bias_ih_l0", "rnn.bias_hh_l0", "lm.weight", "lm.bias",
              "l2.weight", "l2.bias", "l3.weight", "l3.bias")
MIN_FREE = 1.0e9
COMP = []                                              # (section, label, reported, recomputed, equal)


def clock(label):
    print(f"[{label}: {time.time() - T0:.1f} s]", flush=True)


def cmp(sec, label, reported, recomputed, equal=None):
    norm = lambda s: str(s).replace("−", "-").replace("—", "—").strip()
    eq = (norm(reported) == norm(recomputed)) if equal is None else bool(equal)
    COMP.append((sec, label, norm(reported), norm(recomputed), eq))
    return eq


# ---------------------------------------------------------------- reading through git
def show(path, ref=WREF):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def names(d, ref=WREF):
    return subprocess.run(["git", "ls-tree", "-r", "--name-only", ref, "--", d], capture_output=True, check=True, text=True).stdout.split()


def blob_id(path, ref):
    r = subprocess.run(["git", "rev-parse", "--verify", "-q", f"{ref}:{path}"], capture_output=True, text=True)
    return r.stdout.strip() or None


def jshow(path, ref=WREF):
    return json.loads(show(path, ref))


sha = lambda b: hashlib.sha256(b).hexdigest()
utc = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
hms = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M:%S")


def free_bytes():
    s = os.statvfs("/home/user")
    return s.f_bavail * s.f_frsize


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


DTYPE = {"FloatStorage": ("<f4", "torch.float32"), "DoubleStorage": ("<f8", "torch.float64"), "LongStorage": ("<i8", "torch.int64"),
         "BoolStorage": ("?", "torch.bool")}


def load_pt(blob):
    """{name: (numpy array, torch dtype string)}."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    root = pk[:-len("data.pkl")]
    obj = _Unpickler(z.open(pk)).load()
    W = {}
    for k, o in obj.items():
        _, (stype, key), offset, shape, stride = o
        n = int(np.prod(shape)) if shape else 1
        dt = np.dtype(DTYPE[stype][0])
        raw = z.read(root + "data/" + key)[offset * dt.itemsize:(offset + n) * dt.itemsize]
        a = np.frombuffer(raw, dtype=dt).reshape(shape).copy()
        exp, acc = [], 1
        for s in reversed(shape):
            exp.append(acc); acc *= s
        assert tuple(reversed(exp)) == tuple(stride), (k, shape, stride)
        W[k] = (a, DTYPE[stype][1])
    return W


def tensors_sha(W, keys):
    """exp20_reference.tensors_sha as read: per name in sorted order, the name, torch's dtype string, the shape, the bytes."""
    h = hashlib.sha256()
    for k in sorted(keys):
        v, dts = W[k]
        h.update(k.encode()); h.update(dts.encode()); h.update(str(tuple(v.shape)).encode()); h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


def tsha(v, dts):
    """exp42_cells.tsha as read: sha256 of the dtype string, the shape and the bytes."""
    return sha(dts.encode() + str(tuple(v.shape)).encode() + np.ascontiguousarray(v).tobytes())


# ---------------------------------------------------------------- the task (exp11_cell, exp41_scale, exp42_cells, exp45, exp26_candidate, as read)
EOS, BOS = 256, 257
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
NAME = lambda t: f"{t[0]}{t[1]}"
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
    """exp42_cells.panels(seed, f)[:3] and exp54_fraction.py's length panel."""
    th = {t: split(t, f, seed) for t in DEV}
    A = {NAME(t): panel(t, th[t][1], seed, (12,), 300)[12] for t in DEV}
    TA = {NAME(t): panel(t, th[t][0], seed + 1, (12,), 300)[12] for t in DEV}
    Rs, dig = {}, {}
    for t in DEV[:2]:
        s = reduction_sample(t, th[t][1], f, seed)
        Rs[NAME(t)] = [render(t, a, b, 0, Random(0)) for a, b in s]
        dig[NAME(t)] = sha(json.dumps([list(p) for p in s]).encode())
    LP = panel(DEV[0], split(DEV[0], f, seed)[1], seed, (12, 40), 300)
    return A, Rs, TA, LP, dig, th


def item_key(p):
    return p[0], int(bytes(p[1:3])), int(bytes(p[-3:-1]))


def stream_digest(f, seed, rows):
    """exp11_cell.batches over exp26_candidate.Digest of exp45.batch_tensor, without torch. Returns {row: sha256}; asserts every item taught."""
    th = {t: split(t, f, seed) for t in DEV}
    tk = {(t[2][0], a, b) for t in DEV for a, b in th[t][0]}
    rng = Random(seed + 77); flat = [(t, pr) for t in DEV for pr in th[t][0]]

    def st():
        while True:
            order = list(range(len(flat))); rng.shuffle(order)
            for i in order:
                t, (a, b) = flat[i]
                yield render(t, a, b, rng.randint(0, 12), rng)
    s = st(); h = hashlib.sha256(); out = {}; bad = 0
    for u in range(1, max(rows) + 1):
        items = [next(s) for _ in range(64)]
        bad += sum(item_key(p) not in tk for p, _ in items)
        rws = [p + t for p, t in items]; L = max(map(len, rws)) + 1
        data = [[BOS] + r + [BOS] * (L - 1 - len(r)) for r in rws]
        x = b"".join(array("q", r[:-1]).tobytes() for r in data); y = b"".join(array("q", r[1:]).tobytes() for r in data)
        mk = b"".join(bytes(1 if len(p) <= j < len(p) + len(t) else 0 for j in range(L - 1)) for p, t in items)
        for tt in (x, y, mk):
            h.update(str((len(rws), L - 1)).encode()); h.update(tt)
        if u in rows:
            out[u] = h.copy().hexdigest()
    return out, bad


# ---------------------------------------------------------------- the cell layer's geometry (e42_cells.py, as read)
U, SIDE, RADIUS = 256, 16, 3
PERIOD_SETS = {"1": {1: 256}, "8": {1: 128, 2: 64, 4: 32, 8: 32}}


def neighbourhood(i):
    r, c = divmod(i, SIDE)
    return sorted({((r + dr) % SIDE) * SIDE + (c + dc) % SIDE for dr in range(-RADIUS, RADIUS + 1) for dc in range(-RADIUS, RADIUS + 1)} - {i})


def draws(seed, N, period_set):
    order = list(range(U)); Random(f"e42|periods|{seed}").shuffle(order)
    p, k = [0] * U, 0
    for per, n in PERIOD_SETS[period_set].items():
        for i in order[k:k + n]:
            p[i] = per
        k += n
    rp = Random(f"e42|phases|{seed}"); f = [rp.randrange(p[i]) for i in range(U)]
    rc = Random(f"e42|connections|{seed}")
    A = [[j for j in range(U) if j != i] if N == U - 1 else sorted(rc.sample(sorted(neighbourhood(i)), N)) for i in range(U)]
    return p, f, A


# ---------------------------------------------------------------- the forward in numpy (arch26.forward, e42_cells.forward, exp11_cell.respond, as read)
sig = lambda v: 1.0 / (1.0 + np.exp(-v))


class Net:
    """G and F10: arch26.forward with the recurrent matrix times the mask (the mask is all ones for G). P8: e42_cells.forward, variant `step`."""

    def __init__(self, W, periodic=False):
        self.w = {k: np.asarray(v[0], np.float32) for k, v in W.items() if k in MODEL_KEYS}
        self.periodic = periodic
        if "mask" in W:
            m3 = np.tile(W["mask"][0], (3, 1)).astype(np.float32)
            self.w["rnn.weight_hh_l0"] = self.w["rnn.weight_hh_l0"] * m3
        if periodic:
            p, f = W["periods"][0], W["phases"][0]
            self.p, self.f, self.K = p, f, int(p.max())
            kmask = (np.arange(self.K)[:, None] < p[None, :]).astype(np.float32)
            fixed = kmask / p[None, :].astype(np.float32)
            c = np.where((p == 1)[None, :], fixed, W["a"][0].astype(np.float32) * kmask)
            self.c3 = np.tile(c, (1, 3)).astype(np.float32)

    def logits_last(self, x):
        w = self.w
        z1 = np.tanh(w["book.weight"][x] @ w["l1.weight"].T + w["l1.bias"])
        gi = z1 @ w["rnn.weight_ih_l0"].T + w["rnn.bias_ih_l0"]
        B, L = x.shape; n = w["rnn.weight_hh_l0"].shape[1]
        h = np.zeros((B, n), np.float32); hs = []; ghh = []
        for t in range(L):
            gh = h @ w["rnn.weight_hh_l0"].T + w["rnn.bias_hh_l0"]
            if self.periodic:
                ghh.append(gh)
                kk = min(self.K, t + 1)
                G = sum(self.c3[k] * gi[:, t - k] for k in range(kk)); H = sum(self.c3[k] * ghh[t - k] for k in range(kk))
            else:
                G, H = gi[:, t], gh
            r = sig(G[:, :n] + H[:, :n]); z = sig(G[:, n:2 * n] + H[:, n:2 * n]); c = np.tanh(G[:, 2 * n:] + r * H[:, 2 * n:])
            hn = (1 - z) * c + z * h
            if self.periodic:
                upd = (t >= self.f) & ((t - self.f) % self.p == 0)
                h = np.where(upd[None, :], hn, h).astype(np.float32)
            else:
                h = hn
            hs.append(h)
        z2 = np.tanh(z1[:, -1] @ w["l2.weight"].T + w["l2.bias"] + hs[-1] @ w["lm.weight"].T + w["lm.bias"])
        return z2 @ w["l3.weight"].T + w["l3.bias"]

    def respond(self, prompts, cap=4):
        out = [None] * len(prompts); groups = {}
        for i, p in enumerate(prompts):
            groups.setdefault(len(p), []).append(i)
        for _, ids in groups.items():
            x = np.array([[BOS, *prompts[i]] for i in ids])
            gen = [[] for _ in ids]; done = [False] * len(ids)
            for _ in range(cap):
                yy = self.logits_last(x).argmax(1)
                for q in range(len(ids)):
                    if not done[q]:
                        gen[q].append(int(yy[q])); done[q] = done[q] or int(yy[q]) == EOS
                x = np.concatenate([x, yy[:, None]], 1)
                if all(done):
                    break
            for q, i in enumerate(ids):
                out[i] = gen[q]
        return out

    def exact(self, items):
        outs = self.respond([p for p, _ in items])
        return sum(o == t for o, (_, t) in zip(outs, items))


# ---------------------------------------------------------------- the readings, written here from the brief's words
def to80(curve, key):
    v = [(r["step"], r[key]) for r in curve]
    for i in range(len(v) - 2):
        if v[i][1] >= 80 and v[i + 1][1] >= 80 and v[i + 2][1] >= 80:
            return v[i][0]
    return None


def colv(curve, key):
    if key in ("max", "min"):
        return {r["step"]: 100 * r["A_counts"][key + "89"] / 300 for r in curve}
    return {r["step"]: r[key] for r in curve}


def mrows(c, end, n):
    return sum(c[end - 256 * i] for i in range(n)) / n


def summ(r):
    c = r["curve"]; end = r["dose"]
    s = dict(dose=end, to80={k: to80(c, k) for k in ("A", "R")}, lv={})
    marks = [u for u in (12288, 24576, 49152, 122880) if u <= end]
    for k in ("A", "R", "taught") + (("len40", "max", "min") if "len40" in c[0] else ()):
        cc = colv(c, k); s["lv"][k] = {u: mrows(cc, u, 3) for u in marks}
    cR = colv(c, "R")
    m0, m1, m2 = mrows(cR, end, 9), mrows(cR, end - 8192, 9), mrows(cR, end - 16384, 9)
    s["last"], s["before"] = m0 - m1, m1 - m2
    s["flat"] = abs(s["last"]) < 3 and abs(s["before"]) < 3
    s["fell"] = s["last"] <= -3 or s["before"] <= -3
    s["endR"] = mrows(cR, end, 3)
    s["gen"] = s["to80"]["R"] is not None
    s["cpu"] = (r["cpu_updates"] + r["cpu_scoring"] + r.get("cpu_replacement", 0.0)) / 3600
    s["resumed"] = [x["at_update"] for x in r.get("checkpointing", {}).get("resumed", [])]
    if r.get("changing"):
        ev = [x["replaced_since_previous_row"] for x in c[1:]]
        tot = 256 * r["N"]
        s["rep16"] = (100 * sum(ev[:16]) / (16 * tot), 100 * sum(ev[-16:]) / (16 * tot))
    return s


def word_to80(xg, xn, last=DOSE - 512):
    """The brief: both reach -> ratio below 2/3 fewer, above 1.5 more, else about the same; only G -> more if G's <= 32,256, else not decided;
    only the network -> fewer if its <= 32,256, else not decided; neither -> neither reaches."""
    if xg is not None and xn is not None:
        q = xn / xg
        return "fewer" if q < 2 / 3 else "more" if q > 1.5 else "about the same"
    if xg is not None:
        return "more" if xg <= 32256 else "not decided"
    if xn is not None:
        return "fewer" if xn <= 32256 else "not decided"
    return "neither reaches"


def end_word(d, margin, flat_ref, flat_net):
    """The brief: better if d >= margin and the reference (G, or back-propagation) has flattened; worse if d <= -margin and the network's own
    curve has flattened; not distinguished if |d| < margin and both have flattened; otherwise not decided within the run."""
    if d >= margin and flat_ref:
        return "generalises better"
    if d <= -margin and flat_net:
        return "generalises worse"
    if abs(d) < margin and flat_ref and flat_net:
        return "not distinguished"
    return "not decided within the run"


f1 = lambda x: f"{x:.1f}"
fs = lambda x: f"{x:+.1f}"
fu = lambda x: "—" if x is None else f"{x:,}"
flw = lambda s: "flattened" if s["flat"] else "fell" if s["fell"] else "not flattened"


def md_tables(text):
    out, cur = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            cur.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif cur:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return [t for t in out if len(t) > 2]


def sections(text, pat):
    """{heading: text until the next heading of the same pattern}."""
    parts = re.split(pat, text)
    return {parts[i].strip(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


# ================================================================ 1
def section1(ctx):
    print("================ 1  inventory, roster, completeness, settings, identities")
    allf = names(E54)
    ctx["allf"] = allf
    top = [x for x in allf if x.count("/") == M.count("/") + 2]
    js = [x for x in top if x.endswith(".json")]; pts = [x for x in top if x.endswith(".pt")]
    logs = [x for x in allf if "/logs/" in x]; qf = [x for x in allf if "/qual/" in x]
    print(f"  files under exp45/e54 at {WREF[:8]}: {len(allf)} (top level {len(top)}: {len(js)} JSON, {len(pts)} weight files, "
          f"{len([x for x in top if x.endswith('.diff')])} diff, {len([x for x in top if x.endswith('.md')])} tables file); logs {len(logs)}; qual {len(qf)}")
    print(f"  folders 'ckpt' or 'interrupted' committed: {[x for x in allf if '/ckpt/' in x or '/interrupted/' in x]}")
    R = {}
    for fp in FRACS:
        for cfg in CONFIGS:
            for rule in "SB":
                p = f"{E54}e54_{cfg}_{rule}_f{fp}_{SEED}.json"
                R[(fp, cfg, rule)] = jshow(p)
    P4 = {rule: jshow(f"{E54}e41_{rule}_h256_f10_dry{P4DOSE}_{SEED}.json") for rule in "SB"}
    ctx["R"], ctx["P4"] = R, P4
    receipts = sorted([x.split("/")[-1] for x in js if re.search(r"/e(54|41)_.*_73129\.json$", x) and "_structure" not in x])
    print(f"  receipts of runs: {len(receipts)}; the roster's 18 + 2 present: {len(R) == 18 and len(P4) == 2}")
    # sources at the worker's commit
    srcnames = set()
    for r in list(R.values()) + list(P4.values()):
        srcnames |= set(r["provenance"]["sources_sha256"])
    SRC = {s: show(M + s) for s in sorted(srcnames)}
    ctx["SRC"] = SRC
    srcsha = {s: sha(b) for s, b in SRC.items()}
    same_d = {s: blob_id(M + s, WREF) == blob_id(M + s, DREF) for s in SRC}
    print(f"  source files recorded in the receipts: {len(SRC)}; same blob at the worker's commit as on the default branch at {DREF[:8]}: "
          f"{sum(same_d.values())} (not: {[s for s, v in same_d.items() if not v]})")
    ck = show(M + "exp42_ckpt.py", EREF)
    print(f"  exp42_ckpt.py at E42's commit {EREF[:8]}: {len(ck)} bytes; exp42_cells.py same blob on E42's branch and the default branch: "
          f"{blob_id(M + 'exp42_cells.py', EREF) == blob_id(M + 'exp42_cells.py', DREF)}")
    # source blobs at every commit of the worker's branch
    commits = subprocess.run(["git", "rev-list", f"{DREF}..{WREF}"], capture_output=True, text=True).stdout.split()
    changed = collections.Counter()
    for c in commits:
        for s in SRC:
            if blob_id(M + s, c) not in (blob_id(M + s, WREF), None):
                changed[s] += 1
    first = {s: [c[:8] for c in commits if blob_id(M + s, c) is None] for s in ("exp54_fraction.py",)}
    print(f"  commits of the worker's branch: {len(commits)}; source files whose blob differs at some commit from the final one: {dict(changed) or 'none'}; "
          f"commits without exp54_fraction.py (before it was added): {len(first['exp54_fraction.py'])}")
    ctx["commits"] = commits
    # settings and completeness
    tests = collections.Counter(); fails = []
    for (fp, cfg, rule), r in R.items():
        ex = EXPECT[cfg]; c = r["curve"]
        cmd = f"python3 exp54_fraction.py run {cfg} {rule} {SEED} --fraction {fp}" + (" --theta 0.5" if cfg == "P8" else "")
        g0 = G0 + (["a"] if cfg == "P8" else [])
        opt = dict(name="AdamW", lr=0.001, weight_decay=0.25, betas=[0.9, 0.98],
                   groups=[dict(weight_decay=0.25, lr=0.001, betas=[0.9, 0.98], params=g0), dict(weight_decay=0.0, lr=0.001, betas=[0.9, 0.98], params=["book.weight"])])
        pv = r["provenance"]
        t = dict(
            experiment=r["experiment"] == "E54", complete=r["complete"] is True, fraction=r["fraction"] == fp / 100, seed=r["seed"] == SEED,
            dose=r["dose"] == DOSE, config=r["config"] == cfg, rule=r["rule"] == rule, local=r["local"] == (rule == "S"),
            settings=all(r[k] == v for k, v in ex.items()), batch=r["batch"] == 64 and r["kmax"] == 12 and r["row_every"] == 256,
            rows=[x["step"] for x in c] == list(range(0, DOSE + 1, 256)),
            finite=all(x["finite_losses"] for x in c[1:]) and all(isinstance(x["loss"], float) and math.isfinite(x["loss"]) for x in c[1:]),
            limit=all(all(x["limit"].values()) for x in c),
            check5=all(x["length_counts"]["12"] == x["A_counts"]["add89"] for x in c),
            lp_flag=r["length_panel_is_addition_panel"] is True,
            optimizer=r["optimizer"] == opt and r["optimizer_read_back"] == opt,
            steps=all(v == DOSE for v in r["optimizer_steps"]),
            digest=r["data_digest"] == c[-1]["data_digest"] and r["data_digest"]["batches"] == DOSE,
            command=pv["command"] == cmd and all(x["cmd"] == cmd for x in r["checkpointing"]["resumed"]),
            wrapper=r["checkpointing"]["wrapper"] == "exp54_fraction.py" and r["checkpointing"]["wrapper_sha256"] == srcsha["exp54_fraction.py"] and r["checkpointing"]["every"] == 1024,
            sources=all(srcsha[s] == v for s, v in pv["sources_sha256"].items()) and {"exp54_fraction.py", "exp41_scale.py", "exp42_cells.py", "e42_cells.py"} <= set(pv["sources_sha256"]),
            unmodified=pv["git"]["sources_modified_vs_commit"] == [],
            host=pv["host"]["torch"] == "2.15.0+cpu" and pv["host"]["torch_threads"] == 1 and pv["environment_at_start"] == {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            hash12288="12288" in r.get("tensors_sha256_at", {}),
            data=f"split(t, {fp / 100}, seed)" in r["data"],
            added=r["added_parameters"] == ({"a": [8, 256], "Wx": None} if cfg == "P8" else {"a": None, "Wx": None}),
            beside=r["processes_beside"] == "2")
        for k, v in t.items():
            tests[k] += bool(v)
            if not v:
                fails.append((fp, cfg, rule, k))
    print(f"  {len(R)} receipts of parts 1 to 3, {len(t)} tests each; passed per test: {dict(tests)}")
    print(f"  failures: {fails or 'none'}")
    ctx["fails1"] = fails
    commits_at_start = collections.Counter(r["provenance"]["git"]["commit"][:8] for r in R.values())
    print(f"  commits recorded at the start of the 18 runs: {dict(commits_at_start)}")
    # part 4
    t4 = {}
    for rule, r in P4.items():
        c = r["curve"]; pv = r["provenance"]
        t4[rule] = dict(experiment=r["experiment"] == "E41", complete=r["complete"] is True, H=r["H"] == 256 and r["U"] == 128, fraction=r["fraction"] == 0.1,
                        seed=r["seed"] == SEED, dose=r["dose"] == P4DOSE, rule=r["rule"] == rule and r["local"] == (rule == "S"),
                        rows=[x["step"] for x in c] == list(range(0, P4DOSE + 1, 256)),
                        finite=all(x["finite_losses"] for x in c[1:]) and all(math.isfinite(x["loss"]) for x in c[1:]),
                        steps=all(v == P4DOSE for v in r["optimizer_steps"]),
                        command=pv["command"] == f"python3 exp41_scale.py run {rule} 256 10 {SEED} --dose {P4DOSE} --out exp45/e54",
                        sources=all(srcsha[s] == v for s, v in pv["sources_sha256"].items()) and "exp41_scale.py" in pv["sources_sha256"],
                        exp41_unchanged=blob_id(M + "exp41_scale.py", WREF) == blob_id(M + "exp41_scale.py", DREF),
                        host=pv["host"]["torch"] == "2.15.0+cpu" and pv["environment_at_start"] == {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                        digest=r["data_digest"] == c[-1]["data_digest"])
        print(f"  part 4 {rule}: {sum(t4[rule].values())} of {len(t4[rule])} tests pass; failing: {[k for k, v in t4[rule].items() if not v] or 'none'}; rows {len(c)}")
    # identities between runs
    for fp in FRACS:
        runs = [R[(fp, c, r)] for c in CONFIGS for r in "SB"]
        same = all(all(runs[0]["curve"][i]["data_digest"] == x["curve"][i]["data_digest"] for x in runs) for i in range(len(runs[0]["curve"])))
        print(f"  f = 0.{fp}: batch digest equal in all six runs at every one of the {len(runs[0]['curve'])} rows: {same}")
    p4same = all(P4[r]["curve"][i]["data_digest"] == R[(10, "G", "S")]["curve"][i]["data_digest"] for r in "SB" for i in range(193))
    print(f"  part 4 (both rules): batch digest at rows 0 to 49,152 equal to that of the runs of parts 1 to 3 at f = 0.10: {p4same}")
    for cfg in CONFIGS:
        st = {json.dumps(R[(fp, cfg, r)]["structure_start_sha256"]) for fp in FRACS for r in "SB"}
        row0 = {fp: len({json.dumps({k: R[(fp, cfg, r)]["curve"][0][k] for k in ("A_counts", "R_counts", "taught_counts", "length_counts")}) for r in "SB"}) for fp in FRACS}
        print(f"  {cfg}: one start structure (mask, made) in all six runs: {len(st) == 1}; row 0 equal under S and B at each fraction: {all(v == 1 for v in row0.values())}")
    # the panels and the taught sets, regenerated
    ctx["PAN"] = {}
    for fp in (50,) + FRACS:
        f = fp / 100
        A, Rs, TA, LP, dig, th = panels(f, SEED)
        tk = {(t[2][0], a, b) for t in DEV for a, b in th[t][0]}
        ok_un = all(item_key(p) not in tk for pan in list(A.values()) + list(Rs.values()) + list(LP.values()) for p, _ in pan)
        ok_ta = all(item_key(p) in tk for pan in TA.values() for p, _ in pan)
        oldlp = Random(SEED + 3).sample(split(DEV[0], .5, SEED)[1], 300)
        n_old = sum(pr in set(th[DEV[0]][0]) for pr in oldlp)
        ctx["PAN"][fp] = (A, Rs, TA, LP, dig)
        print(f"  f = 0.{fp}: taught pairs per operation {len(th[DEV[0]][0])}; no taught pair in the panels of A, R and the length test: {ok_un}; "
              f"taught panel all taught: {ok_ta}; length panel at 12 letters = addition panel behind A: {LP[12] == A['add89']}; "
              f"E42's length panel (pairs held out at a half) holds {n_old} taught pairs of 300")
    for fp in (25, 10):
        e41 = jshow(f"{M}exp45/e41/e41_S_h512_f{fp}_{SEED}.json", DREF)
        print(f"  R sample at f = 0.{fp} regenerated: digests equal to E41's receipt: {ctx['PAN'][fp][4] == e41['reduction_sample_sha256']}")
    print(f"  part 4 R sample digests equal to the regenerated ones at f = 0.10: {all(P4[r]['reduction_sample_sha256'] == ctx['PAN'][10][4] for r in 'SB')}")
    tsets = {fp: set(split(DEV[0], fp / 100, SEED)[0]) for fp in (50, 25, 15, 10)}
    print("  taught sets of addition are not nested across fractions (exp11_cell.split seeds its shuffle with the fraction): "
          + "; ".join(f"of {len(tsets[b])} taught at 0.{b}, {len(tsets[a] & tsets[b])} taught at 0.{a}" for a, b in ((25, 15), (25, 10), (15, 10), (50, 25))))
    # the stream regenerated without torch
    for fp in FRACS:
        dg, bad = stream_digest(fp / 100, SEED, {256, 4096})
        rr = R[(fp, "G", "S")]["curve"]
        print(f"  stream at f = 0.{fp} regenerated: digest at row 256 equal {dg[256] == rr[1]['data_digest']['sha256']}, at row 4,096 equal "
              f"{dg[4096] == rr[16]['data_digest']['sha256']} ({dg[4096][:16]}); untaught items in 4,096 batches: {bad}")
    ctx["counts"] = dict(files=len(allf), top=len(top), json=len(js), pt=len(pts), logs=len(logs), qual=len(qf))
    clock("section 1")


# ================================================================ 2
def section2(ctx):
    print("================ 2  every figure, recomputed")
    R, P4 = ctx["R"], ctx["P4"]
    S = {k: summ(r) for k, r in R.items()}; S4 = {r: summ(x) for r, x in P4.items()}
    ctx["S"], ctx["S4"] = S, S4
    # E42's ratios at a half, seed 73129
    e42 = {(c, r): summ(dict(jshow(f"{M}exp45/e42/e42_{c}_{r}_{SEED}.json", EREF), dose=24576)) for c in CONFIGS for r in "SB"}
    HALF = {(c, r, k): e42[(c, r)]["to80"][k] / e42[("G", r)]["to80"][k] for c in ("F10", "P8") for r in "SB" for k in "AR"}
    ctx["HALF"] = HALF; ctx["e42"] = e42
    print("  E42 at a half, seed 73129, ratios to G: " + ", ".join(f"{c} {r} {k} {v:.2f}" for (c, r, k), v in HALF.items()))
    rep = show(M + "E54_REPORT.md").decode(); ctx["report"] = rep
    tab = show(E54 + "E54_TABLES.md").decode(); ctx["tables"] = tab
    rsec = sections(rep, r"\n### (.*)\n"); tsec = sections(tab, r"\n## (.*)\n")

    def run_row(fp, cfg, rule):
        s = S[(fp, cfg, rule)]
        lv = lambda k: " / ".join(f1(s["lv"][k][u]) for u in (12288, 24576, 49152))
        return [f"`{cfg}`", rule, fu(s["to80"]["A"]), fu(s["to80"]["R"]), lv("A"), lv("R"), lv("len40"), lv("max"), lv("min"),
                f"{fs(s['before'])}, {fs(s['last'])}", flw(s), f"{s['cpu']:.2f}", str(s["resumed"]) if s["resumed"] else "never"]

    def vs_row(fp, cfg, rule):
        s, g = S[(fp, cfg, rule)], S[(fp, "G", rule)]
        cells = []
        for k in "AR":
            xg, xn = g["to80"][k], s["to80"][k]
            ratio = xn / xg if xg and xn else None
            cells.append(f"{word_to80(xg, xn)} ({'—' if ratio is None else f'{ratio:.2f}'}; {HALF[(cfg, rule, k)]:.2f})")
        d = s["endR"] - g["endR"]; mg = MARGIN[(fp, rule)]
        w = end_word(d, mg, g["flat"], s["flat"])
        pre = "neither generalises from this fraction; " if (s["flat"] and g["flat"] and not s["gen"] and not g["gen"]) else ""
        return [rule, f"`{cfg}`", cells[0], cells[1], f"{f1(s['endR'])}, {f1(g['endR'])}, {fs(d)}", str(mg),
                f"{fs(s['before'])}, {fs(s['last'])} ({flw(s)}); {fs(g['before'])}, {fs(g['last'])} ({flw(g)})", pre + w]

    def sb_row(fp, cfg):
        a, b = S[(fp, cfg, "S")], S[(fp, cfg, "B")]
        d = a["endR"] - b["endR"]; mg = MARGIN_SB[fp]
        return [f"`{cfg}`", f"{f1(a['endR'])}, {f1(b['endR'])}, {fs(d)}", str(mg),
                f"{fs(a['before'])}, {fs(a['last'])} ({flw(a)}); {fs(b['before'])}, {fs(b['last'])} ({flw(b)})", end_word(d, mg, b["flat"], a["flat"])]

    def p4_row(rule):
        s = S4[rule]
        lv = lambda k: " / ".join(f1(s["lv"][k][u]) for u in (12288, 24576, 49152, 122880))
        return [rule, fu(s["to80"]["R"]), fu(s["to80"]["A"]), lv("A"), lv("R"), f"{fs(s['before'])}, {fs(s['last'])}", flw(s), f"{s['cpu']:.2f}"]

    def compare_tables(where, secs, key_of):
        n = bad = 0
        for fp in FRACS:
            txt = secs[key_of(fp)]; tb = md_tables(txt)
            assert len(tb) == 3, (where, fp, len(tb))
            mine = [[run_row(fp, c, r) for r in "SB" for c in CONFIGS], [vs_row(fp, c, r) for r in "SB" for c in ("F10", "P8")], [sb_row(fp, c) for c in CONFIGS]]
            for ti, (t, mt) in enumerate(zip(tb, mine)):
                body = t[2:]
                assert len(body) == len(mt), (where, fp, ti)
                for row, mrow in zip(body, mt):
                    for j, (a, b) in enumerate(zip(row, mrow)):
                        n += 1
                        if not cmp(f"2 {where} table", f"f{fp} table {ti + 1} row {row[0]} {row[1]} col {t[0][j]}", a, b):
                            bad += 1
            # the replacement sentence
            m = re.search(r"rule S: ([\d.]+)% over the first sixteen events and ([\d.]+)% over the last sixteen; rule B: ([\d.]+)% over the first sixteen events and ([\d.]+)%", txt)
            mine_r = [S[(fp, "P8", "S")]["rep16"][0], S[(fp, "P8", "S")]["rep16"][1], S[(fp, "P8", "B")]["rep16"][0], S[(fp, "P8", "B")]["rep16"][1]]
            for lab, a, b in zip(("S first 16", "S last 16", "B first 16", "B last 16"), m.groups(), mine_r):
                n += 1; bad += not cmp(f"2 {where} table", f"f{fp} replaced share {lab}", a, f1(b))
        p4t = md_tables(secs[[k for k in secs if k.startswith("Part 4")][0]])[0]
        for row in p4t[2:]:
            for j, (a, b) in enumerate(zip(row, p4_row(row[0]))):
                n += 1; bad += not cmp(f"2 {where} table", f"part 4 {row[0]} col {p4t[0][j]}", a, b)
        print(f"  {where}: {n} cells compared, {bad} differ")
        return n

    ctx["ncells_report"] = compare_tables("report", rsec, lambda fp: f"Taught fraction 0.{fp}")
    ctx["ncells_tables"] = compare_tables("E54_TABLES.md", tsec, lambda fp: f"Taught fraction 0.{fp}")
    # E54_TABLES.json against the recomputed figures at full precision
    TJ = jshow(E54 + "E54_TABLES.json")
    nj = bj = 0
    for (fp, cfg, rule), s in S.items():
        t = TJ["runs"][f"f{fp} {cfg} {rule}"]
        pairs = [(t["to80"]["A"], s["to80"]["A"]), (t["to80"]["R"], s["to80"]["R"]), (t["last_rise"], s["last"]), (t["rise_before"], s["before"]),
                 (t["flattened"], s["flat"]), (t["end_level_R"], s["endR"]), (t["cpu_hours"], s["cpu"])]
        pairs += [(t["levels"][k][str(u)], s["lv"][k][u]) for k in ("A", "R", "len40", "max", "min") for u in (12288, 24576, 49152)]
        for a, b in pairs:
            nj += 1
            ok = (a == b) if not isinstance(a, float) else abs(a - b) < 1e-9
            bj += not cmp("2 E54_TABLES.json", f"f{fp} {cfg} {rule}", a, b, ok)
    print(f"  E54_TABLES.json: {nj} figures of the runs compared at full precision, {bj} differ")
    # the report's sentences: (label, exact text that must stand in the report, reported value, recomputed value)
    g = lambda fp, c, r: S[(fp, c, r)]
    d = lambda fp, c, r: g(fp, c, r)["endR"] - g(fp, "G", r)["endR"]
    dsb = lambda fp, c: g(fp, c, "S")["endR"] - g(fp, c, "B")["endR"]
    rat = lambda fp, c, r, k: g(fp, c, r)["to80"][k] / g(fp, "G", r)["to80"][k]
    rises = lambda fp: [x for c in CONFIGS for r in "SB" for x in (g(fp, c, r)["last"], g(fp, c, r)["before"])]
    mm = lambda fp, keys=("max", "min"): [g(fp, c, r)["lv"][k][u] for c in CONFIGS for r in "SB" for k in keys for u in (12288, 24576, 49152)]
    cpu18 = sum(s["cpu"] for s in S.values()); cpu4 = sum(s["cpu"] for s in S4.values())
    upd_rate = {k: R[k]["cpu_updates"] / DOSE for k in R}; row_rate = {k: R[k]["cpu_scoring"] / 193 for k in R}
    gf = [k for k in R if k[1] != "P8"]; pk = [k for k in R if k[1] == "P8"]
    c4 = jshow(QUAL + "check4.json"); c3p = jshow(QUAL + "check1_73129.json"); c6 = jshow(QUAL + "check6.json")
    c1 = jshow(QUAL + "check1.json")
    p4B = P4["B"]["curve"]; cRB = colv(p4B, "R")
    top = max(p4B, key=lambda x: x["R"])
    above = [x["step"] for x in p4B if x["R"] >= 80]
    sent = [
        ("2", "resumed G S f25", "G S and F10 S at f = 0.25 at update 28,672", "28,672", f"{R[(25, 'G', 'S')]['checkpointing']['resumed'][0]['at_update']:,}"),
        ("2", "resumed F10 S f25", "G S and F10 S at f = 0.25 at update 28,672", "28,672", f"{R[(25, 'F10', 'S')]['checkpointing']['resumed'][0]['at_update']:,}"),
        ("2", "resumed G B f25", "G B at f = 0.25 at 30,720", "30,720", f"{R[(25, 'G', 'B')]['checkpointing']['resumed'][0]['at_update']:,}"),
        ("2", "resumed G S f15", "G S at f = 0.15 at 25,600", "25,600", f"{R[(15, 'G', 'S')]['checkpointing']['resumed'][0]['at_update']:,}"),
        ("2", "resumed G B f15", "G B at f = 0.15 at 2,048", "2,048", f"{R[(15, 'G', 'B')]['checkpointing']['resumed'][0]['at_update']:,}"),
        ("2", "resumed F10 S f15", "F10 S at f = 0.15 at 1,024", "1,024", f"{R[(15, 'F10', 'S')]['checkpointing']['resumed'][0]['at_update']:,}"),
        ("2", "number of resumed runs (listed)", "The receipts record the resumptions under `checkpointing.resumed`:", "6", str(sum(bool(s["resumed"]) for s in S.values()))),
        ("2", "saved state every", "writes every 1,024 updates", "1024", str(R[(25, "G", "S")]["checkpointing"]["every"])),
        ("4", "check 1 F10 tensors", "837dadc1", "837dadc1", c1["cases"]["F10 S f50 dry2048"]["tensors_sha256"]["new"][:8]),
        ("4", "check 1 P8 tensors", "1b78f306", "1b78f306", c1["cases"]["P8 S f50 dry1024"]["tensors_sha256"]["new"][:8]),
        ("4", "check 1 P8 replaced", "0, 0, 173, 49 and 208", "[0, 0, 173, 49, 208]", str(c1["cases"]["P8 S f50 dry1024"]["replaced"])),
        ("4", "check 3 pre-condition CPU-seconds", "675 CPU-seconds", "675", f"{c3p['cpu']:.0f}"),
        ("4", "check 3 rows to 12,288", "all 49 scored rows", "49", str(sum(1 for x in R[(25, 'G', 'S')]["curve"] if x["step"] <= 12288))),
        ("4", "check 3 part 4 rows", "all 161 rows to update 40,960", "161", str(sum(1 for x in P4["S"]["curve"] if x["step"] <= 40960))),
        ("4", "check 4 G S rate", "0.135 and 0.126", "0.135", f"{c4['rates']['G S']['cpu_s_per_update']:.3f}"),
        ("4", "check 4 G B rate", "0.135 and 0.126", "0.126", f"{c4['rates']['G B']['cpu_s_per_update']:.3f}"),
        ("4", "check 4 F10 S rate", "0.136 for `F10`", "0.136", f"{c4['rates']['F10 S']['cpu_s_per_update']:.3f}"),
        ("4", "check 4 P8 S rate", "0.181 for `P8`", "0.181", f"{c4['rates']['P8 S']['cpu_s_per_update']:.3f}"),
        ("4", "check 4 row rates", "7.3 to 11.9 CPU-seconds a scored row", "7.3 to 11.9",
         f"{min(v['cpu_s_per_scored_row'] for v in c4['rates'].values()):.1f} to {max(v['cpu_s_per_scored_row'] for v in c4['rates'].values()):.1f}"),
        ("4", "check 4 forecast per part", "15.1 CPU-hours for each of parts 1 to 3", "15.1", f"{c4['forecast_cpu_h_per_part_1_to_3']:.1f}"),
        ("4", "check 4 forecast job", "49.3 for the job", "49.3", f"{c4['forecast_cpu_h_job']:.1f}"),
        ("4", "check 6 tensors", "0cd4ca8e", "0cd4ca8e", c6["tensors_sha256"]["resumed"][:8]),
        ("4", "check 6 rows", "the same nine rows", "9", str(c6["rows"])),
        ("4", "cost of the 18 runs", "used 37.7 CPU-hours", "37.7", f1(cpu18)),
        ("4", "cost of part 4", "part 4 used 3.4 CPU-hours", "3.4", f1(cpu4)),
        ("4", "cost of the job", "about 41.7 CPU-hours", "41.7", f1(cpu18 + cpu4 + 0.6)),
        ("4", "rate an update G and F10", "0.100 to 0.120 CPU-seconds an update for `G` and `F10`", "0.100 to 0.120",
         f"{min(upd_rate[k] for k in gf):.3f} to {max(upd_rate[k] for k in gf):.3f}"),
        ("4", "rate an update P8", "0.144 to 0.162 for `P8`", "0.144 to 0.162", f"{min(upd_rate[k] for k in pk):.3f} to {max(upd_rate[k] for k in pk):.3f}"),
        ("4", "rate a row G and F10", "5.6 to 6.4 and 9.3 to 10.3", "5.6 to 6.4", f"{min(row_rate[k] for k in gf):.1f} to {max(row_rate[k] for k in gf):.1f}"),
        ("4", "rate a row P8", "5.6 to 6.4 and 9.3 to 10.3", "9.3 to 10.3", f"{min(row_rate[k] for k in pk):.1f} to {max(row_rate[k] for k in pk):.1f}"),
        ("6", "f25 ratios F10 A S, B", "ratios 3.65 under the sweep and 4.29 under back-propagation", "3.65, 4.29", f"{rat(25, 'F10', 'S', 'A'):.2f}, {rat(25, 'F10', 'B', 'A'):.2f}"),
        ("6", "f25 ratios F10 R S, B", "on R (3.40 and 2.67", "3.40, 2.67", f"{rat(25, 'F10', 'S', 'R'):.2f}, {rat(25, 'F10', 'B', 'R'):.2f}"),
        ("6", "f25 ratios P8 S A, R", "(1.21 and 1.13, against 1.36 and 1.27 at a half)", "1.21, 1.13", f"{rat(25, 'P8', 'S', 'A'):.2f}, {rat(25, 'P8', 'S', 'R'):.2f}"),
        ("6", "f25 ratios P8 B A, R", "*more* on A (1.94, against 1.25) and *about the same* on R (1.33, against 1.12)", "1.94, 1.33",
         f"{rat(25, 'P8', 'B', 'A'):.2f}, {rat(25, 'P8', 'B', 'R'):.2f}"),
        ("6", "half ratios", "against 2.36 and 1.70 at a half", "2.36, 1.70, 2.27, 1.48, 1.36, 1.27, 1.25, 1.12",
         ", ".join(f"{HALF[k]:.2f}" for k in (("F10", "S", "A"), ("F10", "B", "A"), ("F10", "S", "R"), ("F10", "B", "R"), ("P8", "S", "A"), ("P8", "S", "R"), ("P8", "B", "A"), ("P8", "B", "R")))),
        ("6", "f25 S P8 vs G", "R at update 49,152 is 90.7 for `P8` and 96.0 for `G`, a difference of −5.3", "90.7, 96.0, -5.3",
         f"{f1(g(25, 'P8', 'S')['endR'])}, {f1(g(25, 'G', 'S')['endR'])}, {fs(d(25, 'P8', 'S'))}"),
        ("6", "f25 S F10 vs G", "R is 82.8 against 96.0 (−13.2 points)", "82.8, 96.0, -13.2", f"{f1(g(25, 'F10', 'S')['endR'])}, {f1(g(25, 'G', 'S')['endR'])}, {fs(d(25, 'F10', 'S'))}"),
        ("6", "f25 B vs G", "`F10` 94.6 against 95.7 (−1.1) and `P8` 94.0 against 95.7 (−1.7)", "94.6, 95.7, -1.1, 94.0, -1.7",
         f"{f1(g(25, 'F10', 'B')['endR'])}, {f1(g(25, 'G', 'B')['endR'])}, {fs(d(25, 'F10', 'B'))}, {f1(g(25, 'P8', 'B')['endR'])}, {fs(d(25, 'P8', 'B'))}"),
        ("6", "f25 S vs B", "(90.7 against 94.0, −3.3 points, margin 10", "-3.3", fs(dsb(25, "P8"))),
        ("6", "f25 S vs B G, F10", "For `G` (+0.3 points) and for `F10` (−11.8 points)", "+0.3, -11.8", f"{fs(dsb(25, 'G'))}, {fs(dsb(25, 'F10'))}"),
        ("6", "f25 max and min range", "answered at 89.8 to 99.2%", "89.8 to 99.2", f"{f1(min(mm(25)))} to {f1(max(mm(25)))}"),
        ("6", "E42 replaced share", "were 7.6% and 19.8%", "7.6, 19.8", "{:.1f}, {:.1f}".format(*e42[("P8", "S")]["rep16"])),
        ("6", "f15 P8 B on A", "reaches 80, at update 47,872", "47,872", fu(g(15, "P8", "B")["to80"]["A"])),
        ("6", "f15 last rises", "every last rise is between +2.1 and +7.8", "+2.1 to +7.8",
         f"{fs(min(g(15, c, r)['last'] for c in CONFIGS for r in 'SB'))} to {fs(max(g(15, c, r)['last'] for c in CONFIGS for r in 'SB'))}"),
        ("6", "f15 rises before", "every rise before it between +3.7 and +8.0", "+3.7 to +8.0",
         f"{fs(min(g(15, c, r)['before'] for c in CONFIGS for r in 'SB'))} to {fs(max(g(15, c, r)['before'] for c in CONFIGS for r in 'SB'))}"),
        ("6", "f15 vs G", "under the sweep `F10` 29.7 against `G` 46.6 (−16.9) and `P8` 56.6 against 46.6 (+10.0); under back-propagation `F10` 56.8 against 73.0 (−16.2) and `P8` 72.9 against 73.0 (−0.1)",
         "-16.9, +10.0, -16.2, -0.1", ", ".join(fs(d(15, c, r)) for r in "SB" for c in ("F10", "P8"))),
        ("6", "f15 S vs B", "(by 26.4 points for `G`, 27.1 for `F10` and 16.4 for `P8`)", "26.4, 27.1, 16.4", ", ".join(f1(-dsb(15, c)) for c in CONFIGS)),
        ("6", "f10 rises", "every rise is between −0.1 and +2.8 points", "-0.1 to +2.8", f"{fs(min(rises(10)))} to {fs(max(rises(10)))}"),
        ("6", "f10 vs G", "(9.3 against 15.1, −5.7)", "-5.7, +4.0, -7.3, +4.5", ", ".join(fs(d(10, c, r)) for r in "SB" for c in ("F10", "P8"))),
        ("6", "f10 S vs B", "`G` 15.1 against 24.6 (−9.5), `F10` 9.3 against 17.3 (−8.0), `P8` 19.1 against 29.1 (−10.0)", "-9.5, -8.0, -10.0", ", ".join(fs(dsb(10, c)) for c in CONFIGS)),
        ("6", "f10 max and min range", "answered at 68.7 to 92.7%", "68.7 to 92.7", f"{f1(min(mm(10)))} to {f1(max(mm(10)))}"),
        ("6", "f10 A at most", "stays at or below 33.3", "33.3", f1(max(mm(10, ("A",))))),
        ("6", "part 4 B R levels", "rises from 26.8 at update 12,288 to 49.9 at update 49,152 and 75.0 at update 122,880", "26.8, 49.9, 75.0",
         ", ".join(f1(S4["B"]["lv"]["R"][u]) for u in (12288, 49152, 122880))),
        ("6", "part 4 B highest single row", "one single scored row reaches 83.25 (update 116,992)", "83.25 at 116,992", f"{top['R']:.2f} at {top['step']:,}"),
        ("6", "part 4 B rows at or above 80", "one single scored row reaches 83.25", "1", str(len(above))),
        ("6", "part 4 B rises", "The rise before the last rise is +2.5 points and the last rise −0.7 points", "+2.5, -0.7", f"{fs(S4['B']['before'])}, {fs(S4['B']['last'])}"),
        ("6", "part 4 B A to 80, level", "Its score A reaches 80 at update 112,384 and ends at a level of 79.8", "112,384, 79.8", f"{fu(S4['B']['to80']['A'])}, {f1(S4['B']['lv']['A'][122880])}"),
        ("6", "part 4 B nine-row means", "42.5 at update 40,960", "42.5, 61.8, 69.0, 73.8, 76.3, 75.5",
         ", ".join(f1(mrows(cRB, u, 9)) for u in (40960, 73728, 90112, 106496, 114688, 122880))),
        ("6", "part 4 S R levels", "7.0, 8.1, 10.1 and 13.4", "7.0, 8.1, 10.1, 13.4", ", ".join(f1(S4["S"]["lv"]["R"][u]) for u in (12288, 24576, 49152, 122880))),
        ("6", "part 4 S rises", "with rises +1.4 and −0.5", "+1.4, -0.5", f"{fs(S4['S']['before'])}, {fs(S4['S']['last'])}"),
        ("7", "P54.4 gains", "gains 5.1 (`G`), 6.6 (`F10`), 7.9 (`P8`)", "5.1, 6.6, 7.9",
         ", ".join(f1(g(10, c, "B")["lv"]["R"][49152] - g(10, c, "B")["lv"]["R"][24576]) for c in CONFIGS)),
        ("7", "P54.5 levels", "| 15.2, 8.5, 17.4 |", "15.2, 8.5, 17.4", ", ".join(f1(g(10, c, "S")["lv"]["A"][49152]) for c in CONFIGS)),
        ("7", "P54.2", "| 13,568 |", "13,568", fu(g(25, "P8", "S")["to80"]["R"])),
        ("7", "P54.6 level", "does not reach 80; level 73.0", "73.0", f1(g(15, "G", "B")["endR"])),
        ("7", "P54.7 level", "| level 46.6 |", "46.6", f1(g(15, "G", "S")["endR"])),
        ("7", "P54.9 level", "level 75.0 at update 122,880, flattened", "75.0", f1(S4["B"]["endR"])),
        ("7", "P54.10 level", "| level 13.4 |", "13.4", f1(S4["S"]["endR"])),
    ]
    print(f"  rates a scored row, G and F10, unrounded: lowest {min(row_rate[k] for k in gf):.5f} ({min(gf, key=lambda k: row_rate[k])}), highest {max(row_rate[k] for k in gf):.4f}")
    e42d = jshow(f"{M}exp45/e42/e42_P8_S_{SEED}.json", EREF)["dose"]
    c10 = jshow(f"{M}exp45/e42_qual/check10_73129.json", EREF)
    print(f"  E42's figures 7.6% and 19.8%: from E42's receipt of P8 under the sweep, seed 73129, a run of {e42d:,} updates ({e42d // 256} events), over its first and its last "
          f"sixteen events; E42's check 10 (A10, theta 0.5, the first 4,096 updates) gives a mean share of {100 * c10['runs']['0.5']['mean_share']:.1f}% over its "
          f"{len(c10['runs']['0.5']['events'])} events")
    cmp("2 report 6", "E42's 7.6% and 19.8% described as 'over its first 4,096 updates'", "first 4,096 updates", f"first and last sixteen events of {e42d:,} updates", False)
    nf = 0
    for sec, lab, snippet, a, b in sent:
        present = snippet in rep
        if not present:
            print(f"  NOTE: the text quoted for '{lab}' was not found in the report: {snippet!r}")
        nf += 1; cmp(f"2 report {sec}", lab, a, b, (a.replace("−", "-") == b) and present)
    print(f"  sentences of the report: {nf} figures compared")
    # the predictions table
    pt = [t for t in md_tables(rep) if t[0][0] == "" and t[0][1].startswith("Statement")][0]
    out = predictions(ctx)
    for row in pt[2:]:
        cmp("2 report 7 table", f"{row[0]} outcome", row[3], "holds" if out[row[0]] else "does not hold")
    TJ = jshow(E54 + "E54_TABLES.json")
    for k, v in out.items():
        cmp("2 E54_TABLES.json", f"{k} holds", TJ["predictions"][k]["holds"], v)
        print(f"  {k}: {'holds' if v else 'does not hold'}")
    # the costs that the report gives as measured
    chk = 0.0; parts = {}
    for p in ("check1_new/e54_F10_S_f50_dry2048_73129.json", "check1_new/e54_P8_S_f50_dry1024_73129.json", "check1_e42/e42_F10_S_dry2048_73129.json",
              "check1_e42/e42_P8_S_dry1024_73129.json", "check6_whole/e54_P8_S_f25_dry2048_73129.json", "check6_resumed/e54_P8_S_f25_dry2048_73129.json"):
        r = jshow(QUAL + p); v = r["cpu_updates"] + r["cpu_scoring"] + r.get("cpu_replacement", 0.0)
        parts[p.split("/")[0]] = parts.get(p.split("/")[0], 0) + v; chk += v
    chk += c3p["cpu"]
    print(f"  CPU of the checks' dry runs from their receipts: {chk / 3600:.2f} CPU-hours ({ {k: round(v) for k, v in parts.items()} } CPU-seconds, check 3's "
          f"pre-condition {c3p['cpu']:.0f}); the report's 0.6 is the brief's estimate")
    ctx["chk_cpu"] = chk / 3600; ctx["cpu18"] = cpu18; ctx["cpu4"] = cpu4
    clock("section 2")


def predictions(ctx):
    S, S4 = ctx["S"], ctx["S4"]
    g = lambda fp, c, r: S[(fp, c, r)]
    d = lambda fp, c, r: g(fp, c, r)["endR"] - g(fp, "G", r)["endR"]
    out = {}
    out["P54.1"] = all(word_to80(g(25, "G", r)["to80"]["A"], g(25, "F10", r)["to80"]["A"]) == "more" for r in "SB")
    x = g(25, "P8", "S")["to80"]["R"]; out["P54.2"] = x is not None and x <= 24064
    out["P54.3"] = all(d(10, c, r) < MARGIN[(10, r)] for c in ("F10", "P8") for r in "SB")
    out["P54.4"] = any(g(10, c, "B")["lv"]["R"][49152] - g(10, c, "B")["lv"]["R"][24576] >= 10 for c in CONFIGS)
    out["P54.5"] = all(g(10, c, "S")["lv"]["A"][49152] <= 30 for c in CONFIGS)
    out["P54.6"] = g(15, "G", "B")["to80"]["R"] is not None
    out["P54.7"] = g(15, "G", "S")["to80"]["R"] is None
    out["P54.8"] = not any(end_word(d(fp, c, r), MARGIN[(fp, r)], g(fp, "G", r)["flat"], g(fp, c, r)["flat"]) == "generalises better"
                           for fp in FRACS for c in ("F10", "P8") for r in "SB")
    out["P54.9"] = S4["B"]["to80"]["R"] is not None
    out["P54.10"] = S4["S"]["endR"] < 20
    return out


# ================================================================ 3
def section3(ctx):
    print("================ 3  readings, boundaries, licences, the record quoted")
    S, S4 = ctx["S"], ctx["S4"]
    print("  rises of R to two decimals (rise before, last rise), the curve's word, and the distance of the larger rise in size from 3 points:")
    for fp in FRACS:
        for r in "SB":
            for c in CONFIGS:
                s = S[(fp, c, r)]; big = max(abs(s["before"]), abs(s["last"]))
                print(f"    f0.{fp} {r} {c:3s}: {s['before']:+.2f}, {s['last']:+.2f}  {flw(s):13s} larger rise {big:.2f} ({3 - big:+.2f} from 3, that is {9 * (3 - big):+.1f} "
                      f"points of R in one scored row); end level of R {s['endR']:.2f}")
    for r in "SB":
        s = S4[r]; print(f"    part 4 {r}: {s['before']:+.2f}, {s['last']:+.2f}  {flw(s)}; end level {s['endR']:.2f}")
    print("  readings against G (difference of end levels of R, margin, the reading, what would change it):")
    readings = {}
    for fp in FRACS:
        for r in "SB":
            gg = S[(fp, "G", r)]
            for c in ("F10", "P8"):
                s = S[(fp, c, r)]; dd = s["endR"] - gg["endR"]; mg = MARGIN[(fp, r)]
                w = end_word(dd, mg, gg["flat"], s["flat"]); readings[(fp, c, r)] = w
                why = []
                if not gg["flat"]:
                    why.append("G not flattened")
                if not s["flat"]:
                    why.append(f"{c} not flattened")
                print(f"    f0.{fp} {r} {c} vs G: {dd:+.2f} (margin {mg}; {abs(dd) / mg:.2f} of it) -> {w}" + (f" [{'; '.join(why)}]" if why else "")
                      + f"; if both had flattened: {end_word(dd, mg, True, True)}")
    ctx["readings"] = readings
    print("  the sweep against back-propagation:")
    for fp in FRACS:
        for c in CONFIGS:
            a, b = S[(fp, c, "S")], S[(fp, c, "B")]; dd = a["endR"] - b["endR"]; mg = MARGIN_SB[fp]
            print(f"    f0.{fp} {c}: {dd:+.2f} (margin {mg}) -> {end_word(dd, mg, b['flat'], a['flat'])}; if both had flattened: {end_word(dd, mg, True, True)}")
    nb = [k for k, w in readings.items() if w == "generalises better"]
    nd = [k for k, w in readings.items() if w.startswith("not decided")]
    print(f"  'generalises better' anywhere: {nb or 'no'}; readings against G not decided: {[f'f0.{a} {b} {c}' for a, b, c in nd]}")
    reach = [(fp, c, r) for fp in FRACS for c in ("F10", "P8") for r in "SB" if S[(fp, c, r)]["gen"] and not S[(fp, "G", r)]["gen"]]
    print(f"  a network that reaches 80 on R where G does not: {reach or 'none'}")
    for fp in FRACS:
        print(f"  f0.{fp}: generalises from the fraction (R reaches 80): " + ", ".join(f"{c} {r} {S[(fp, c, r)]['gen']}" for r in "SB" for c in CONFIGS))
    slow = [(fp, c, r, k) for fp in FRACS for c in ("F10", "P8") for r in "SB" for k in "AR"
            if S[(fp, c, r)]["to80"][k] and S[(fp, "G", r)]["to80"][k] and S[(fp, c, r)]["to80"][k] / S[(fp, "G", r)]["to80"][k] < ctx["HALF"][(c, r, k)]]
    print(f"  ratios smaller at the fraction than at a half (\"is slowed less by the smaller fraction\"): {slow}")
    # the one-row sensitivity of updates to 80 near the bands
    print("  updates to 80 at f = 0.25: ratio, and the ratios one row (256 updates) either way of the network's count:")
    for r in "SB":
        for c in ("F10", "P8"):
            for k in "AR":
                xg, xn = S[(25, "G", r)]["to80"][k], S[(25, c, r)]["to80"][k]
                print(f"    {c} {r} {k}: {xn}/{xg} = {xn / xg:.3f} ({word_to80(xg, xn)}); with {xn - 256}: {(xn - 256) / xg:.3f}, with {xn + 256}: {(xn + 256) / xg:.3f}; "
                      f"with G's one row earlier/later: {xn / (xg - 256):.3f}/{xn / (xg + 256):.3f}")
    # the flattening rule applied to part 4's back-propagation curve at earlier ends
    for rule in "SB":
        c = ctx["P4"][rule]["curve"]; cR = colv(c, "R")
        res = []
        for end in range(16384 + 2048, P4DOSE + 1, 256):
            m0, m1, m2 = mrows(cR, end, 9), mrows(cR, end - 8192, 9), mrows(cR, end - 16384, 9)
            if abs(m0 - m1) < 3 and abs(m1 - m2) < 3:
                later = max(mrows(cR, u, 3) for u in range(end, P4DOSE + 1, 256)) - mrows(cR, end, 3)
                res.append((end, later))
        e49 = [x for x in res if x[0] == 49152]
        big5 = [x for x in res if x[1] >= 5]
        print(f"  part 4 {rule}: earliest end said 'flattened' {res[0][0]:,}; ends said 'flattened' at or before 49,152: {sum(1 for e, _ in res if e <= 49152)}")
        if big5:
            print(f"  part 4 {rule}: ends said 'flattened' and followed by a later rise of 5 points or more lie between {big5[0][0]:,} and {big5[-1][0]:,}; "
                  f"those at or after 40,960: {[(e, round(l, 1)) for e, l in big5 if e >= 40960][:12]}")
        c49 = (mrows(cR, 49152, 9) - mrows(cR, 40960, 9), mrows(cR, 40960, 9) - mrows(cR, 32768, 9))
        print(f"  part 4 {rule}: rises if the run had ended at 49,152: before {c49[1]:+.2f}, last {c49[0]:+.2f}; ends (every 256 updates from 18,432) at which the rule "
              f"says 'flattened': {len(res)}; of these, followed by a rise of the level of R of 5 points or more later in the run: {sum(1 for _, l in res if l >= 5)}"
              + (f"; the largest such later rise {max(l for _, l in res):.1f}" if res else ""))
    # the present-width dense runs against part 4 at matched updates
    for u in (12288, 24576, 49152):
        print(f"  level of R at {u:,} under back-propagation, f = 0.10: present width (G, this job) {S[(10, 'G', 'B')]['lv']['R'][u]:.1f}; half width (part 4) {S4['B']['lv']['R'][u]:.1f}")
    # the record quoted: spreads between seeds of the dense network (E41)
    lv = lambda rule, H, fp, seed, u: mrows(colv(jshow(f"{M}exp45/e41/e41_{rule}_h{H}_f{fp}_{seed}.json", DREF)["curve"], "R"), u, 3)
    seeds = (73129, 93683, 116089)
    sp = {}
    for rule, H, fp, u in (("S", 512, 25, 12288), ("B", 512, 25, 12288), ("S", 512, 10, 12288), ("B", 512, 10, 12288),
                           ("B", 256, 10, 12288), ("B", 256, 10, 24576), ("B", 256, 10, 40960)):
        v = [lv(rule, H, fp, s, u) for s in seeds]
        sp[(rule, H, fp, u)] = (max(v) - min(v), v.index(min(v)) == 0, v)
        print(f"  E41 spread of the level of R over the three seeds, {rule} h{H} f0.{fp} at {u:,}: {max(v) - min(v):.1f} (levels {', '.join(f'{x:.1f}' for x in v)}; "
              f"seed 73129 lowest: {v.index(min(v)) == 0})")
    ctx["spread"] = sp
    clock("section 3")


# ================================================================ 4
def section4(ctx):
    print("================ 4  resumptions and check 6")
    R = ctx["R"]
    drv = show(E54 + "logs/driver.log").decode().splitlines()
    launch = collections.defaultdict(list)
    for l in drv:
        m = re.match(r"(\S+Z) launched (\S+) ", l)
        if m:
            launch[m.group(2)].append(utc(m.group(1)))
    restarts = [(utc(l[:20]), l[21:]) for l in drv if "restart" in l]
    run = set(); mx = 0; order = []; passt = {}
    for l in drv:
        m = re.match(r"(\S+Z) (launched|ended) (\S+)", l)
        if m:
            if m.group(2) == "launched":
                run.add(m.group(3)); order.append((m.group(1), m.group(3)))
            else:
                run.discard(m.group(3))
            mx = max(mx, len(run))
        if "restart" in l:
            run = set()
        m = re.match(r"(\S+Z) check (\S+ .*?) -> (\S+) PASS", l)
        if m:
            passt.setdefault(m.group(3), m.group(1))
    firsts = {}
    for t, n in order:
        firsts.setdefault(n, t)
    f15 = min(t for n, t in firsts.items() if "_f15" in n or "_f10" in n or "part4" in n)
    print(f"  driver log: at most {mx} runs alive at once; check 3 passed at f = 0.25 at {passt['check3_G_B_f25'][11:19]} (B) and {passt['check3_G_S_f25'][11:19]} (S); "
          f"first full comparisons of check 3 at f = 0.25 at {passt['check3_G_B_f25_partial'][11:19]} (B) and {passt['check3_G_S_f25_partial'][11:19]} (S); "
          f"first launch of parts 2 to 4 at {f15[11:19]}; first launches in part 1: {[n for _, n in order[:3]]}; in part 3, in order: "
          f"{[nm for nm, tt in sorted(firsts.items(), key=lambda x: x[1]) if '_f10' in nm]}")
    print(f"  driver log: {len(drv)} lines; lines written by hand about restarts: {[(hms(t), s[:60]) for t, s in restarts]}")
    total_dup = 0; ctx["dup"] = {}
    kill_lb = collections.defaultdict(list)
    for (fp, cfg, rule), r in sorted(R.items()):
        name = f"e54_{cfg}_{rule}_f{fp}"
        log = show(f"{E54}logs/{name}.log").decode().splitlines()
        res = r["checkpointing"]["resumed"]
        rl = [i for i, l in enumerate(log) if " resume " in l]
        rows = lambda ls: {int(re.search(r" (\d+): A", l).group(1)): l.split(": ", 1)[1].split(" cpu ")[0] for l in ls if re.search(r" \d+: A ", l)}
        cpu = lambda ls: {int(re.search(r" (\d+): A", l).group(1)): sum(map(int, re.search(r"cpu (\d+)\+(\d+)s", l).groups())) for l in ls if re.search(r" \d+: A ", l)}
        if not res:
            assert not rl, name
            continue
        a, b = log[:rl[0]], log[rl[0] + 1:]
        ra, rb = rows(a), rows(b)
        dup = sorted(set(ra) & set(rb))
        eq = all(ra[s] == rb[s] for s in dup)
        u0 = res[0]["at_update"]
        # the receipt's rows after the resumption equal the log's
        cur = {x["step"]: x for x in r["curve"]}
        logrec = all(abs(cur[s]["A"] - float(re.search(r"A ([\d.]+)", rb[s]).group(1))) < 0.006 and abs(cur[s]["R"] - float(re.search(r"R ([\d.]+)", rb[s]).group(1))) < 0.006 for s in rb)
        # data digests after the resumption against the uninterrupted P8 run of the same fraction (same stream)
        ref = {x["step"]: x["data_digest"] for x in R[(fp, "P8", "S")]["curve"]}
        dg = all(x["data_digest"] == ref[x["step"]] for x in r["curve"] if x["step"] > u0)
        nafter = sum(1 for x in r["curve"] if x["step"] > u0)
        last_before = max(ra); lb = launch[name][0] + cpu(a)[last_before]
        kill_lb[round(launch[name][1])].append(lb)
        print(f"  {name}: resumed at {u0:,} ({res[0]['when']}; replay {res[0]['cpu_replay']:.1f} CPU-s; command the original: {res[0]['cmd'] == r['provenance']['command']}); "
              f"rows printed before the stop after the saved state: {[s for s in sorted(ra) if s > u0]}; printed again after the resumption and equal: {dup} {eq}; "
              f"lost CPU-seconds between the saved state and the last printed row {cpu(a)[last_before] - cpu(a)[u0]}; "
              f"receipt rows after the resumption {nafter}, batch digests equal to the uninterrupted P8 S run's: {dg}; log rows equal to the receipt's: {logrec}; "
              f"process alive at least until {hms(lb)} (launch {hms(launch[name][0])} plus the CPU-seconds printed at row {last_before:,})")
        total_dup += len(dup); ctx["dup"][name] = (u0, dup, eq); ctx.setdefault("lost", []).append(cpu(a)[last_before] - cpu(a)[u0])
    ratio = max((x["cpu_updates"] + x["cpu_scoring"] + x["cpu_replacement"]) / x["wall"] for r in R.values() for x in r["curve"][1:])
    print(f"  CPU-seconds at a row divided by the wall-clock seconds at that row, largest over every row of the 18 receipts: {ratio:.3f} "
          f"(below 1, so CPU-seconds since a launch are a lower bound of the time elapsed since it)")
    for t, lbs in sorted(kill_lb.items()):
        print(f"  relaunch at {hms(t)}: the interrupted processes were alive at least until {hms(max(lbs))}")
    ctx["kill"] = {hms(t): hms(max(lbs)) for t, lbs in kill_lb.items()}
    lost = sum(ctx["lost"]); replay = sum(r["checkpointing"]["resumed"][0]["cpu_replay"] for r in R.values() if r["checkpointing"]["resumed"])
    print(f"  CPU not counted in the receipts: work between the saved state and the last printed row before the stop {lost} CPU-seconds (at least; the work after "
          f"the last printed row is not known), replays of the stream {replay:.0f} CPU-seconds")
    cmp("2 report 2", "first restart 'at about 00:24 UTC'", "00:24", "after 00:24:14, before 00:27:01", True)
    cmp("2 report 2", "second restart 'at about 03:46 UTC'", "03:46", "after 03:48:01, before 03:48:37", False)
    tot = ctx["cpu18"] + ctx["cpu4"] + ctx["chk_cpu"] + (lost + replay) / 3600
    print(f"  the job's CPU-hours with the checks' dry runs measured and the lost work and replays counted: {tot:.2f} (report 41.7)")
    cmp("2 report 4", "the checks' CPU-hours ('about 0.6')", "0.6", f"{ctx['chk_cpu']:.2f}", False)
    cmp("2 report 4", "the job's CPU-hours with the checks measured and the lost work and replays counted", "41.7", f"{tot:.1f}", False)
    print(f"  rows printed both before the stop and after the resumption: {total_dup}, all equal at the printed precision: {all(v[2] for v in ctx['dup'].values())}")
    nres = [k for k, r in R.items() if r["checkpointing"]["resumed"]]
    print(f"  runs resumed: {len(nres)}: {sorted(nres)}; configurations: {sorted({k[1] for k in nres})}; rules: {sorted({k[2] for k in nres})}; fractions: {sorted({k[0] for k in nres})}")
    print(f"  part 4: no folder 'interrupted' committed, driver state part4_interrupted = {jshow(E54 + 'driver_state.json')['part4_interrupted']}, part 4 logs: "
          f"{[x.split('/')[-1] for x in ctx['allf'] if '/logs/part4' in x]}")
    # check 6
    a = jshow(QUAL + "check6_whole/e54_P8_S_f25_dry2048_73129.json"); b = jshow(QUAL + "check6_resumed/e54_P8_S_f25_dry2048_73129.json")
    VOL = ("wall", "cpu_updates", "cpu_scoring", "cpu_replacement")
    strip = lambda c: [{k: v for k, v in x.items() if k not in VOL} for x in c]
    full = strip([x for x in R[(25, "P8", "S")]["curve"] if x["step"] <= 2048])
    print(f"  check 6: tensors equal {a['tensors_sha256'] == b['tensors_sha256']} ({a['tensors_sha256'][:8]}); digest equal {a['data_digest'] == b['data_digest']}; "
          f"rows equal except times {strip(a['curve']) == strip(b['curve'])} ({len(a['curve'])} rows); resumed at {[x['at_update'] for x in b['checkpointing']['resumed']]} / "
          f"{[x['at_update'] for x in a['checkpointing']['resumed']]}; structure at the end equal {a['structure_end_sha256'] == b['structure_end_sha256']}; "
          f"added parameter a equal {a['added_parameters_sha256'] == b['added_parameters_sha256']}")
    print(f"  check 6 receipts' rows 0 to 2,048 equal to the full run P8 S f0.25 (all fields except times): {strip(a['curve']) == full}; replaced per row "
          f"{[x['replaced_since_previous_row'] for x in a['curve']]}; events after the resumption with replacements: {sum(1 for x in b['curve'] if x['step'] > 1024 and x['replaced_since_previous_row'] > 0)}")
    print(f"  check 6 settings: config {b['config']}, rule {b['rule']}, fraction {b['fraction']}, theta {b['theta']}, dose {b['dose']}, wrapper hash = the runner's "
          f"{b['checkpointing']['wrapper_sha256'] == sha(ctx['SRC']['exp54_fraction.py'])}")
    clock("section 4")


# ================================================================ 5
def section5(ctx):
    print("================ 5  checks 1 to 5")
    R, P4 = ctx["R"], ctx["P4"]
    keys = ("loss", "A_counts", "R_counts", "taught_counts", "data_digest", "length_counts", "replaced_since_previous_row")
    for cfg, dose, wp in (("F10", 2048, "exp45/e42_qual/check8_73129_runs/e42_F10_S_dry2048_73129.json"), ("P8", 1024, "exp45/e42_qual/check7_73129_runs/rep0/e42_P8_S_dry1024_73129.json")):
        new = jshow(f"{QUAL}check1_new/e54_{cfg}_S_f50_dry{dose}_{SEED}.json"); host = jshow(f"{QUAL}check1_e42/e42_{cfg}_S_dry{dose}_{SEED}.json"); e42 = jshow(M + wp, EREF)
        rows = lambda r: [{k: x[k] for k in keys} for x in r["curve"]]
        print(f"  check 1 {cfg}: tensors new/host/E42 {new['tensors_sha256'][:8]}/{host['tensors_sha256'][:8]}/{e42['tensors_sha256'][:8]} equal "
              f"{new['tensors_sha256'] == host['tensors_sha256'] == e42['tensors_sha256']}; mask and made at start and end equal "
              f"{new['structure_start_sha256'] == host['structure_start_sha256'] == e42['structure_start_sha256'] and new['structure_end_sha256'] == host['structure_end_sha256'] == e42['structure_end_sha256']}; "
              f"all {len(new['curve'])} rows equal in {len(keys)} fields {rows(new) == rows(host) == rows(e42)}; replaced {[x['replaced_since_previous_row'] for x in new['curve']]}; "
              f"new runner fraction {new['fraction']}, theta {new['theta']}; E42 host receipt's command: {host['provenance']['command']}")
    c3p = jshow(QUAL + "check1_73129.json")
    srcl = ctx["SRC"]["exp42_cells.py"].decode()
    want = dict(re.findall(r'"([SB])": "([0-9a-f]{64})"', srcl[srcl.index("DRY2048 = {"):srcl.index("DRY2048 = {") + 300]))
    print(f"  check 3's pre-condition: S {c3p['runs']['S']['tensors_sha256'][:8]} = exp42_cells.DRY2048 {c3p['runs']['S']['tensors_sha256'] == want['S']}; "
          f"B {c3p['runs']['B']['tensors_sha256'][:8]} = {c3p['runs']['B']['tensors_sha256'] == want['B']}; digest at 2,048 {c3p['runs']['S']['digest_2048'][:16]}; pass {c3p['pass']}")
    dg50, _ = stream_digest(0.5, SEED, {2048})
    print(f"  stream at f = 0.50 regenerated: digest at 2,048 equal to the pre-condition's {dg50[2048] == c3p['runs']['S']['digest_2048']}; "
          f"equal to the check 1 F10 receipt's {dg50[2048] == jshow(QUAL + 'check1_new/e54_F10_S_f50_dry2048_73129.json')['data_digest']['sha256']}")
    for fp in FRACS:
        c2 = jshow(f"{QUAL}check2_f{fp}.json")
        print(f"  check 2 f0.{fp}: recorded {c2['digest'][:16]} ({c2['read_from']}); every run's row 4,096: "
              f"{len({R[(fp, c, r)]['curve'][16]['data_digest']['sha256'] for c in CONFIGS for r in 'SB'})} distinct value, equal {all(R[(fp, c, r)]['curve'][16]['data_digest']['sha256'] == c2['digest'] for c in CONFIGS for r in 'SB')}")
    K3 = ("loss", "A_counts", "R_counts", "taught_counts", "data_digest")
    for fp in (25, 10):
        for rule in "SB":
            e41 = jshow(f"{M}exp45/e41/e41_{rule}_h512_f{fp}_{SEED}.json", DREF)
            same_blob = blob_id(f"{M}exp45/e41/e41_{rule}_h512_f{fp}_{SEED}.json", WREF) == blob_id(f"{M}exp45/e41/e41_{rule}_h512_f{fp}_{SEED}.json", DREF)
            by = {x["step"]: x for x in e41["curve"]}
            mine = [x for x in R[(fp, "G", rule)]["curve"] if x["step"] <= 12288]
            diff = [(x["step"], k) for x in mine for k in K3 if x[k] != by[x["step"]][k]]
            more = all(x[k] == by[x["step"]][k] for x in mine for k in ("A", "R", "taught"))
            h = R[(fp, "G", rule)]["tensors_sha256_at"]["12288"]
            c3 = jshow(f"{QUAL}check3_G_{rule}_f{fp}.json"); r256 = jshow(f"{QUAL}check3_G_{rule}_f{fp}_row256.json")
            print(f"  check 3 G {rule} f0.{fp}: {len(mine)} rows compared in {len(K3)} fields, differences {len(diff)}; scores A, R, taught equal {more}; "
                  f"hash at 12,288 = E41's {h == e41['tensors_sha256']}; E41 receipt same blob at both commits {same_blob}; recorded pass {c3['pass']} "
                  f"({c3['rows_compared']} rows); row-256 file compares the printed line to two decimals: {r256['pass']}")
    for rule in "SB":
        e41 = jshow(f"{M}exp45/e41/e41_{rule}_h256_f10_{SEED}.json", DREF); by = {x["step"]: x for x in e41["curve"]}
        mine = [x for x in P4[rule]["curve"] if x["step"] <= 40960]
        diff = [(x["step"], k) for x in mine for k in K3 if x[k] != by[x["step"]][k]]
        print(f"  check 3 part 4 {rule}: {len(mine)} rows compared, differences {len(diff)}; E41's final hash {e41['tensors_sha256'][:8]} (no hash at 40,960 in part 4's receipt); "
              f"recorded pass {jshow(f'{QUAL}check3_part4_{rule}.json')['pass']}")
    # check 4
    c4 = jshow(QUAL + "check4.json")
    e42cpu = {(c, r): (lambda x: (x["cpu_updates"] + x["cpu_scoring"] + x.get("cpu_replacement", 0)) / 3600)(jshow(f"{M}exp45/e42/e42_{c}_{r}_{SEED}.json", EREF)) for c in CONFIGS for r in "SB"}
    print("  E42's CPU-hours at 24,576 updates, seed 73129: " + ", ".join(f"{c} {r} {v:.2f}" for (c, r), v in e42cpu.items()))
    fF = c4["forecast_cpu_h_per_run"]["F10 S"] * e42cpu[("F10", "B")] / e42cpu[("F10", "S")]
    fP = c4["forecast_cpu_h_per_run"]["P8 S"] * e42cpu[("P8", "B")] / e42cpu[("P8", "S")]
    part = sum(v for k, v in c4["forecast_cpu_h_per_run"].items() if k not in ("F10 B", "P8 B")) + fF + fP
    job = 3 * part + c4["part4_cpu_h_from_e41_receipts"] + 0.6
    print(f"  check 4: forecast F10 B as recorded {c4['forecast_cpu_h_per_run']['F10 B']:.3f} (ratio used {0.86 / 0.82:.3f}); with E42's own F10 B/F10 S ratio "
          f"{e42cpu[('F10', 'B')] / e42cpu[('F10', 'S')]:.3f}: {fF:.3f}; P8 B {c4['forecast_cpu_h_per_run']['P8 B']:.3f} against {fP:.3f}; part {c4['forecast_cpu_h_per_part_1_to_3']:.2f} against "
          f"{part:.2f}; job {c4['forecast_cpu_h_job']:.2f} against {job:.2f}; below 51 either way: {job <= 51}")
    ctx["c4fix"] = (c4["forecast_cpu_h_job"], job, e42cpu)
    cmp("5 check 4", "forecast of the job with E42's own ratio of F10 under the two rules", f"{c4['forecast_cpu_h_job']:.1f}", f"{job:.1f}", False)
    # check 5
    n5 = sum(len(r["curve"]) for r in R.values())
    print(f"  check 5: rows with length_counts['12'] = A_counts['add89']: {sum(x['length_counts']['12'] == x['A_counts']['add89'] for r in R.values() for x in r['curve'])} of {n5}; "
          f"flag in every receipt: {all(r['length_panel_is_addition_panel'] for r in R.values())}")
    clock("section 5")


# ================================================================ 6
def section6(ctx):
    print("================ 6  the weights")
    R, P4 = ctx["R"], ctx["P4"]
    print(f"  free disk at the start of the section: {free_bytes() / 1e9:.2f} GB")
    done = []; skipped = []
    order = [((fp, c, r), R[(fp, c, r)], f"{E54}e54_{c}_{r}_f{fp}_{SEED}") for fp in FRACS for c in CONFIGS for r in "SB"] + \
            [(("p4", "H", r), P4[r], f"{E54}e41_{r}_h256_f10_dry{P4DOSE}_{SEED}") for r in "SB"]
    p8draw = draws(SEED, 10, "8"); f10draw = draws(SEED, 10, "1")
    dgw = lambda v: sha(json.dumps(v).encode())[:16]
    srcl = show(M + "e42_cells.py").decode()
    want = re.search(r"73129: \(\"([0-9a-f ]+)\", \"([0-9a-f ]+)\", \"([0-9a-f ]+)\"\)", srcl).groups()
    print(f"  e42_cells.py's own digests for seed 73129 reproduced: periods {' '.join(dgw(draws(SEED, 10, ps)[0]) for ps in ('8',)) in want[0]}, "
          f"phases {dgw(draws(SEED, 10, '8')[1]) in want[1]}, connections N=10 {dgw(f10draw[2]) in want[2]}")
    ctx["fwd"] = {}
    for key, r, stem in order:
        if free_bytes() < MIN_FREE + 8e6:
            skipped.append(stem.split("/")[-1]); continue
        blob = show(stem + ".pt"); W = load_pt(blob)
        ok_file = sha(blob) == r["weights_sha256"]; ok_t = tensors_sha(W, [k for k in W if k in MODEL_KEYS]) == r["tensors_sha256"]
        line = f"  {stem.split('/')[-1]}: {len(blob):,} bytes; file sha256 as recorded {ok_file}; tensors_sha256 as recorded {ok_t}"
        if key[0] != "p4":
            cfg = key[1]
            ok_s = tsha(*W["mask"]) == r["structure_end_sha256"]["mask"] and tsha(*W["made"]) == r["structure_end_sha256"]["made"]
            small = load_pt(show(stem + "_recurrent.pt"))
            ok_small = set(small) <= set(W) and all(small[k][0].tobytes() == W[k][0].tobytes() for k in small)
            st = jshow(stem + "_structure.json")
            mask = W["mask"][0]; whh = W["rnn.weight_hh_l0"][0]; m3 = np.tile(mask, (3, 1))
            outside = int(np.count_nonzero(whh[~m3]))
            conn = [[j for j in range(U) if mask[i, j] and j != i] for i in range(U)]
            ok_struct = st["end"]["connections"] == conn and st["end"]["made"] == [[int(W["made"][0][i, j]) for j in cs] for i, cs in enumerate(conn)]
            pw, fw = W["periods"][0].tolist(), W["phases"][0].tolist()
            line += f"; structure hashes {ok_s}; _recurrent.pt tensors equal {ok_small}; _structure.json end = weights {ok_struct}; non-zero weights outside the mask {outside}"
            if cfg == "G":
                line += f"; mask all ones {bool(mask.all())}"
            if cfg == "F10":
                line += (f"; mask = drawn connections {conn == f10draw[2]}; start = end in structure.json {st['start']['connections'] == conn}; "
                         f"periods all 1 {set(pw) == {1}}; made all 0 {int(W['made'][0].max()) == 0}")
            if cfg == "P8":
                made = W["made"][0]
                nnb = sum(j not in set(neighbourhood(i)) for i in range(U) for j in conn[i])
                cnt = all(len(c) == 10 for c in conn) and bool(mask.diagonal().all())
                mvals = sorted({int(made[i, j]) for i in range(U) for j in conn[i]})
                nnew = sum(int(made[i, j]) > 0 for i in range(U) for j in conn[i]); nrep = sum(x["replaced_since_previous_row"] for x in r["curve"])
                a = W["a"][0]; p = np.array(pw); kmask = (np.arange(8)[:, None] < p[None, :])
                line += (f"; periods and phases = drawn {pw == p8draw[0] and fw == p8draw[1]} (period counts {dict(sorted(collections.Counter(pw).items()))}); "
                         f"10 connections and itself in every row {cnt}; connections outside the neighbourhood {nnb}; start = drawn {st['start']['connections'] == p8draw[2]}; "
                         f"connections made after the start {nnew} of 2,560 (replacements recorded {nrep}); made values multiples of 256 "
                         f"{all(v % 256 == 0 for v in mvals)}, largest {max(mvals)}; a outside each cell's period zero {bool((a[~kmask] == 0).all())}; "
                         f"a hash as recorded {tsha(*W['a']) == r['added_parameters_sha256']['a']}")
        done.append((stem.split("/")[-1], ok_file and ok_t))
        print(line)
        # the forward reproduced for four runs
        if key in ((25, "G", "S"), (10, "G", "B"), (25, "F10", "S"), (10, "P8", "B")):
            fp = key[0]; A, Rs, TA, LP, _ = ctx["PAN"][fp]
            net = Net(W, periodic=(key[1] == "P8"))
            t = time.time()
            got = dict(A_counts={k: net.exact(v) for k, v in A.items()}, R_counts={k: net.exact(v) for k, v in Rs.items()},
                       taught_counts={k: net.exact(v) for k, v in TA.items()}, length_counts={str(k): net.exact(v) for k, v in LP.items()})
            last = r["curve"][-1]
            eq = {k: got[k] == last[k] for k in got}
            ctx["fwd"][key] = eq
            print(f"    forward in numpy, {key[1]} {key[2]} f0.{fp}: final row reproduced {eq} ({time.time() - t:.1f} s); A {got['A_counts']}, R {got['R_counts']}")
        del W, blob
    print(f"  weight files read: {len(done)} of {len(order)}; file and tensor hashes as recorded: {sum(ok for _, ok in done)}; not read for want of disk: {skipped or 'none'}")
    print(f"  free disk at the end of the section: {free_bytes() / 1e9:.2f} GB")
    ctx["wdone"] = (len(done), sum(ok for _, ok in done), skipped)
    clock("section 6")


# ================================================================ 7
def apply_diff(old, diff):
    """Applies a unified diff to `old` (a list of lines) and returns the new lines; asserts every context and removed line."""
    lines = diff.splitlines(); out = []; i = 0; k = 0
    while k < len(lines) and not lines[k].startswith("@" * 2):
        k += 1
    while k < len(lines):
        m = re.match("@" * 2 + r" -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? " + "@" * 2, lines[k]); k += 1
        start = int(m.group(1)) - 1
        out += old[i:start]; i = start
        while k < len(lines) and not lines[k].startswith("@" * 2):
            l = lines[k]; k += 1
            if l.startswith(" ") or l == "":
                assert old[i] == l[1:], (i, old[i], l); out.append(old[i]); i += 1
            elif l.startswith("-"):
                assert old[i] == l[1:], (i, old[i], l); i += 1
            elif l.startswith("+"):
                out.append(l[1:])
    return out + old[i:]


def section7(ctx):
    print("================ 7  the runner as text")
    new = ctx["SRC"]["exp54_fraction.py"].decode(); old = show(M + "exp42_ckpt.py", EREF).decode()
    dfile = show(E54 + "exp54_fraction_vs_exp42_ckpt.diff").decode()
    rebuilt = apply_diff(old.split("\n"), dfile)
    print(f"  the committed .diff applied to exp42_ckpt.py at {EREF[:8]} gives exp54_fraction.py exactly: {rebuilt == new.split(chr(10))}")
    rep = ctx["report"]
    app = rep[rep.index("```diff") + 8:rep.index("```", rep.index("```diff") + 8)]
    body = lambda t: [l for l in t.splitlines() if not l.startswith(("---", "+++"))]
    print(f"  the report's Appendix A equals the committed .diff apart from the two header lines: {body(app) == body(dfile)}")
    chk = {
        "1 --fraction required, divided by 100": '"--fraction" in a' in new and "fp / 100" in new,
        "1 the stream at f": "R.E.batches([(t, R.E.split(t, f, seed)[0]) for t in R.E.DEV], seed, R.E.KMAX[\"M\"])" in new,
        "2 panels at f, first three": "A, Rs, TA = R.panels(seed, f)[:3]" in new,
        "2 length panel from pairs held out at f": "R.E.panel(R.LEN_TASK, R.E.split(R.LEN_TASK, f, seed)[1], seed, ks=(12, 40), n=300)" in new,
        "2 the three assertions before the first update or the replay": new.index("assert LP[12] == A[") < new.index("if not resume:") and "assert item_key(p) not in tk" in new,
        "3 assert_taught in the loop and the replay": new.count("assert_taught(items, tk)") == 2,
        "3 imported from exp41_scale after exp42_cells": new.index("import exp42_cells as R") < new.index("from exp41_scale import"),
        "4 49,152 updates, receipt name, folder": "DOSE = 49152" in new and 'f"e54_{config}_{rule}_f{fp}{tag}_{seed}.json"' in new and 'CKDIR = OUT / "ckpt"' in new,
        "4 E54 and fraction recorded; fraction compared on resuming": 'experiment="E54", kind="birth", fraction=f' in new and 'rec["fraction"]) == (config, rule, seed, dose, f)' in new,
        "5 wrapper and hashes": 'wrapper="exp54_fraction.py"' in new and '("exp54_fraction.py", "exp41_scale.py")' in new,
        "6 hash at 12,288 after the row, before writing and saving": new.index('["12288"] = R.R_.tensors_sha') < new.index("if u % 4096 == 0 or u == dose") and new.index("score_row(u); losses = []") < new.index('["12288"] = R.R_.tensors_sha'),
        "7 self-test removed, --stop-after kept": "def selftest" not in new and '"--stop-after"' in new,
        "8 names e54": "E54_BESIDE" in new and "e42 " not in re.sub(r'"e42\|replace', "", new),
        "9 nothing replaced but R.install": not re.search(r"\b(R|L42|E|R\.E|R\.L42)\.[A-Za-z_]+\s*=[^=]", new),
        "check 5 assertion at every row": 'row["length_counts"]["12"] == row["A_counts"]["add89"]' in new,
        "configuration enters only through R.CONFIGS, R.build and theta": new.count("config") > 0 and "if config ==" not in new,
        "no source hash is recorded on resuming": 'rec["checkpointing"]["resumed"].append(dict(at_update=u0, cpu_replay' in new and "sources_sha256" not in new[new.index("assert dg.value() == st[\"digest\"]"):new.index("curve = rec[\"curve\"]")],
        "saved state loaded with weights_only=False, no hash of it": "T.load(ckp, weights_only=False)" in new and "sha" not in new[new.index("def checkpoint(u)"):new.index("if u0 == 0")],
        "replacement generator state saved and restored": "rr=rr.getstate()" in new and "rr.setstate(st[\"rr\"])" in new,
        "replay digest asserted before the training continues": new.index('assert dg.value() == st["digest"]') < new.index("for u in range(u0 + 1, dose + 1)"),
    }
    for k, v in chk.items():
        print(f"  {k}: {v}")
    ctx["runner"] = chk
    clock("section 7")


# ================================================================ 8
def section8(ctx):
    print("================ 8  form")
    rep = ctx["report"]
    ph = {f"mark {i + 1}": rep.count(p) for i, p in enumerate(("@" * 2, "TO" + "DO", "T" + "BD", "X" * 2))}
    a0 = rep.index("```diff"); a1 = rep.index("```", a0 + 8)
    hunks = rep[a0:a1].count("\n" + "@" * 2 + " -")
    outside = (rep[:a0] + rep[a1:]).count("@" * 2)
    print(f"  placeholder marks in the report (a doubled at-sign, the to-do mark, the to-be-decided mark, a doubled capital X): {ph}; the doubled at-signs "
          f"inside Appendix A's diff (two per hunk header, {hunks} hunks): {rep[a0:a1].count('@' * 2)}; outside it: {outside}")
    # names of language models: the words are taken from the repository's own branch and file names, never written here
    refs = subprocess.run(["git", "for-each-ref", "--format=%(refname:short) %(objectname)", "refs/remotes/origin"], capture_output=True, text=True).stdout.splitlines()
    w1 = [l.split()[0].split("/")[1] for l in refs if l.split()[1] == WREF][0]
    w2 = [l.split()[0].split("/")[1] for l in refs if l.split()[0].endswith("minifly-consolidated-20261023")][0]
    w3 = re.search(r"brief ([A-Z]+)_TASK_E42", show(M + "exp42_cells.py").decode()).group(1)
    w4 = [l.split()[0].split("/")[2].split("-")[0] for l in refs if l.split()[0].split("/")[1] == w2 and l.split()[0].endswith("audit-20261025")][0]
    memo = [x for x in subprocess.run(["git", "ls-tree", "--name-only", "HEAD", "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/"], capture_output=True, text=True).stdout.split()
            if x.split("/")[-1].startswith("MEMO_TO_") and x.endswith("_WORKER_20261109.md")]
    w5 = memo[0].split("/")[-1][len("MEMO_TO_"):-len("_WORKER_20261109.md")]
    words = [w1, w2, w3, w4, w5]
    print(f"  words searched (from the worker's branch name, the default branch's name, the prefix of E42's brief file, another branch's name and a memo's "
          f"file name): {len(words)}, of lengths {[len(w) for w in words]}")
    targets = [M + "E54_REPORT.md"] + [M + s for s in ("exp54_fraction.py", "e54_checks.py", "e54_driver.py", "e54_tables.py")] + [x for x in ctx["allf"] if not x.endswith(".pt")]
    places = collections.Counter(); files_with = collections.Counter(); detail = collections.Counter()
    for t in targets:
        txt = show(t).decode(errors="replace")
        kind = ("the report" if t.endswith("E54_REPORT.md") else "scripts" if t.endswith(".py") else "logs" if "/logs/" in t
                else "the runner's diff" if t.endswith(".diff") else "receipts and check files" if t.endswith(".json") else "tables")
        hit = False
        for i, w in enumerate(words):
            for m in re.finditer(re.escape(w), txt, re.I):
                places[(kind, i + 1)] += 1; hit = True
                ctxt = txt[max(0, m.start() - 40):m.end() + 40]
                where = ("a branch name" if (txt[m.end():m.end() + 1] == "/" or "origin/" in ctxt or "branch" in ctxt) else
                         "a brief's file name" if "_TASK_" in ctxt else "other")
                detail[(kind, where)] += 1
        files_with[kind] += hit
    print(f"  occurrences by kind of file and word number: {dict(places)}")
    print(f"  occurrences by kind of file and kind of place: {dict(detail)}; files with any: {dict(files_with)}")
    ctx["names"] = (dict(places), dict(detail), dict(files_with))
    msgs = subprocess.run(["git", "log", "--format=%B%x00", f"{DREF}..{WREF}"], capture_output=True, text=True).stdout.split("\0")[:-1]
    nm = sum(bool(re.search(re.escape(w1), m_, re.I)) for m_ in msgs)
    print(f"  commit messages of the worker's branch: {len(msgs)}; containing word 1: {nm}; occurrences per message: "
          f"{sorted({len(re.findall(re.escape(w1), m_, re.I)) for m_ in msgs})} (in the two trailer lines)")
    ctx["msgs"] = (len(msgs), nm)
    head = show(BRIEF, "HEAD"); rb = show(BRIEF, BREF)
    print(f"  the brief on the checked-out branch equals the brief at {BREF} named by the report: {head == rb}; revision 3 in its title: {b'revision 3' in head[:200]}")
    # Appendix B against the tree at e718483e and at the report's commit
    app = rep[rep.index("## Appendix B"):]
    lst = re.findall(r"^(\d+) (\S+)$", app, re.M)
    tree = names(M[:-1], WTAB)
    rel = lambda x: x[len(M):]
    inlist = {p for _, p in lst}
    expect = {rel(x) for x in tree if rel(x).startswith("exp45/e54/") or rel(x) in ("exp54_fraction.py", "e54_checks.py", "e54_driver.py", "e54_tables.py")}
    sizes_ok = 0; sizes_bad = []
    for sz, p in lst:
        n = len(show(M + p, WTAB))
        if n == int(sz):
            sizes_ok += 1
        else:
            sizes_bad.append((p, sz, n))
    now = {rel(x) for x in ctx["allf"]} | {"exp54_fraction.py", "e54_checks.py", "e54_driver.py", "e54_tables.py"}
    print(f"  Appendix B: {len(lst)} files listed; equal to the tree at {WTAB} {inlist == expect}; sizes equal {sizes_ok} (differ: {sizes_bad or 'none'}); "
          f"files of exp45/e54 and the scripts at the report's commit not in the list: {sorted(now - inlist) or 'none'}")
    ctx["appB"] = (len(lst), inlist == expect, sizes_ok, sizes_bad)
    # terms
    for term in ("checker gate", "official seal", "E26's hashes", "replay", "saved state", "dry run", "event", "theta", "the harness", "evidence level"):
        print(f"  term {term!r}: occurrences in the report {rep.count(term)}")
    print(f"  the report names numpy 2.5.5: {'numpy 2.5.5' in rep}; a numpy version in any receipt: "
          f"{any('numpy' in json.dumps(r['provenance']) for r in ctx['R'].values())}")
    clock("section 8")


def summary():
    print("================ comparisons")
    by = collections.Counter(c[0] for c in COMP); bad = [c for c in COMP if not c[4]]
    for s in sorted(by):
        print(f"  {s}: {by[s]} compared, {sum(1 for c in COMP if c[0] == s and not c[4])} differ")
    print(f"  total: {len(COMP)} compared, {len(COMP) - len(bad)} agree, {len(bad)} differ")
    print("  every comparison of the report's sentences:")
    for c in COMP:
        if c[0].startswith("2 report ") and "table" not in c[0]:
            print(f"    [{c[0]}] {c[1]}: reported {c[2]!r}, recomputed {c[3]!r}, {'agree' if c[4] else 'DIFFER'}")
    for c in bad:
        print(f"  DIFFERS [{c[0]}] {c[1]}: reported {c[2]!r}, recomputed {c[3]!r}")


def main():
    ctx = {}
    for f in (section1, section2, section3, section4, section5, section6, section7, section8):
        f(ctx)
    summary()
    print(f"running time: {time.time() - T0:.1f} s")


if __name__ == "__main__":
    main()
