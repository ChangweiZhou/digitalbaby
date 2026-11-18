"""Verification of the final report of E52 (the two learning rules under other learning rates; the worker's commit cb5cacfc).

Read-only. Nothing is trained, the training library is not used, and no file of the worker's branch is executed: every file is read
with `git show` into memory and listed with `git ls-tree -r --name-only`. Weight files are read one at a time into memory (never
extracted to disk) with an unpickler that refuses every class except those that describe a tensor. The script writes nothing
except, optionally, its own printed output when the caller redirects it.

Sections (the numbering of the verification document):
  1  inventory, roster, completeness, settings and identities between runs
  2  every figure of the report and of E52_TABLES.md recomputed from the receipts and set beside the reported figure
  3  the readings in the brief's fixed words, the further runs, the predictions, the explanation under test, the rules after the trial
  4  the resumptions, the interrupted runs, check 4 and its failed first attempt, the number of processes at once
  5  checks 1a, 1b, 2 and 3
  6  the weights: file and tensor hashes, the final norms, the forward recomputed with numpy for three runs
  7  the runner as text (the lines on which the document relies)
  8  form: placeholders, the list of committed files, names of language models (counted, never written), the brief's revision

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I <path>/e52_verify.py
"""
import collections, hashlib, io, json, os, pickle, re, subprocess, sys, time, zipfile
from datetime import datetime, timezone
from random import Random
import numpy as np

T0 = time.time()
sys.dont_write_bytecode = True

WREF = "cb5cacfcb01384073c9c1a6e6071e73614a2b8a2"      # the worker's commit of the report (verified)
DREF = "7cabb13adc38753483d34dd5472fc3c84c4de675"      # the default branch at the worker's source commit
W46C = "b76f5979"                                      # E46's commit named in the receipts for the copy of exp46_decay.py
M = "MOD97_LOCAL_RULE_20261106/"
E52 = M + "exp45/e52/"
QUAL = M + "exp45/e52_qual/"
E41 = M + "exp45/e41/"
BRIEF = "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/TASK_E52_THE_TWO_RULES_UNDER_OTHER_LEARNING_RATES_20261117.md"
BRIEF_REV2 = "12168f12"
SEED = 93683
RATES = (0.0003, 0.001, 0.003, 0.01)
WIDTH = {256: "half", 512: "present"}
U_OF = {256: 128, 512: 256}
E41_DOSE = {256: 40960, 512: 12288}
LONG_DOSE = {256: 136704, 512: 40960}
GROUPS = ("book", "l1", "input_map", "recurrent", "lm", "l2", "l3")
GROUP_TENSORS = {"book": ("book.weight",), "l1": ("l1.weight", "l1.bias"), "input_map": ("rnn.weight_ih_l0", "rnn.bias_ih_l0"),
                 "recurrent": ("rnn.weight_hh_l0", "rnn.bias_hh_l0"), "lm": ("lm.weight", "lm.bias"), "l2": ("l2.weight", "l2.bias"),
                 "l3": ("l3.weight", "l3.bias")}
MIN_FREE = 1.0e9                                       # stop reading weight files when less than 1.0 GB would remain
COMP = []                                              # (section, label, reported, recomputed, equal)


def clock(label):
    print(f"[{label}: {time.time() - T0:.1f} s]", flush=True)


def cmp(sec, label, reported, recomputed, equal=None):
    eq = (str(reported) == str(recomputed)) if equal is None else bool(equal)
    COMP.append((sec, label, str(reported), str(recomputed), eq))
    return eq


# ---------------------------------------------------------------- reading through git
def show(path, ref=WREF):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def names(d, ref=WREF):
    return subprocess.run(["git", "ls-tree", "-r", "--name-only", ref, "--", d], capture_output=True, check=True, text=True).stdout.split()


def blob_id(path, ref):
    out = subprocess.run(["git", "ls-tree", ref, "--", path], capture_output=True, check=True, text=True).stdout.split()
    return out[2] if len(out) >= 3 else None


def jshow(path, ref=WREF):
    return json.loads(show(path, ref))


sha = lambda b: hashlib.sha256(b).hexdigest()
fu = lambda v: "–" if v is None else f"{v:,}"
utc = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
hm = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M:%S")


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
    """exp20_reference.tensors_sha as read: per name in sorted order, the name, torch's dtype string, the shape, the bytes."""
    h = hashlib.sha256()
    for k in sorted(W):
        v = W[k]
        h.update(k.encode()); h.update({np.float32: b"torch.float32", np.float64: b"torch.float64", np.int64: b"torch.int64"}[v.dtype.type])
        h.update(str(tuple(v.shape)).encode()); h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


def free_bytes():
    s = os.statvfs("/home/user")
    return s.f_bavail * s.f_frsize


# ---------------------------------------------------------------- the task and the forward (exp11_cell, exp41_scale, arch26, as read)
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


# ---------------------------------------------------------------- the readings, written here from the brief's words
def to80(curve, key):
    """E41's three-row rule: the first scored row from which three scored rows in a row are at or above 80."""
    v = [r[key] for r in curve]
    for i in range(len(v) - 2):
        if v[i] >= 80 and v[i + 1] >= 80 and v[i + 2] >= 80:
            return curve[i]["step"]
    return None


def mean(v):
    return sum(v) / len(v)


def rms(vals):
    return (sum((b - a) ** 2 for a, b in zip(vals, vals[1:])) / (len(vals) - 1)) ** .5


def summ(r):
    c = r["curve"]
    s = dict(rule=r["rule"], H=r["H"], fp=round(r["fraction"] * 100), rate=r["rate"], decay=r["decay"], product=r["product_control"],
             dose=r["dose"], last=r["dose"] - 512, rows=len(c))
    for k in ("A", "R"):
        s[k + "80"] = to80(c, k)
        s[k + "80_row"] = next((x for x in c if x["step"] == s[k + "80"]), None)
        s[k + "_final"] = mean([x[k] for x in c[-3:]])
    s["taught_final"] = mean([x["taught"] for x in c[-3:]])
    s["len40_last"] = c[-1]["len40"]; s["len40_mean3"] = mean([x["len40"] for x in c[-3:]])
    s["len40_lo"], s["len40_hi"] = min(x["len40"] for x in c[-12:]), max(x["len40"] for x in c[-12:])
    s["rms16"] = rms([x["R"] for x in c[-16:]])
    s["rms17"] = rms([x["R"] for x in c[-17:]])
    s["norms_end"] = c[-1]["norms"]
    at = [x for x in c if x["step"] <= E41_DOSE[r["H"]]]
    s["at41"] = None
    if r["dose"] != E41_DOSE[r["H"]]:
        s["at41"] = dict(A=mean([x["A"] for x in at[-3:]]), R=mean([x["R"] for x in at[-3:]]), len40_last=at[-1]["len40"],
                         len40_mean3=mean([x["len40"] for x in at[-3:]]), norms=at[-1]["norms"], rms16=rms([x["R"] for x in at[-16:]]))
    return s


def klass(x1, x2, D1, D2):
    """E41's rule for two runs (exp41_scale.describe_step as read): x1 the count at 0.001, x2 the count at the other rate."""
    if x1 is not None and x2 is not None:
        q = x2 / x1
        return "faster" if q < 2 / 3 else "slower" if q > 1.5 else "the same"
    if x1 is not None:
        return "slower" if D2 / x1 >= 1.5 else "not decided"
    if x2 is not None:
        return "faster" if x2 / D1 <= 2 / 3 else "not decided"
    return "neither reaches"


def best(S, rule, H, fp, key, rates=RATES):
    got = {r: S[(rule, H, fp, r, False)][key + "80"] for r in rates if (rule, H, fp, r, False) in S}
    got = {r: v for r, v in got.items() if v is not None}
    if not got:
        return dict(named=None, tied=[], count=None, text="not reached")
    lo = min(got.values()); tied = sorted(r for r, v in got.items() if v - lo <= 256)
    if len(tied) == 1:
        return dict(named=tied[0], tied=tied, count=got[tied[0]], text=f"{tied[0]:g}")
    if 0.001 in tied:
        return dict(named=0.001, tied=tied, count=got[0.001], text="0.001 (tied with " + ", ".join(f"{r:g}" for r in tied if r != 0.001) + ")")
    return dict(named=None, tied=tied, count=lo, text="tie of " + ", ".join(f"{r:g}" for r in tied))


def vs(S, rule, H, fp, rate, key, product=False):
    a, b = S.get((rule, H, fp, 0.001, False)), S.get((rule, H, fp, rate, product))
    if a is None or b is None:
        return "not run"
    return klass(a[key + "80"], b[key + "80"], a["last"], b["last"])


def gains(S, rule, H, fp):
    reachedR = any(S[k]["R80"] is not None for k in S if k[:3] == (rule, H, fp) and not k[4])
    at = []
    for r in RATES:
        if r == 0.001 or (rule, H, fp, r, False) not in S:
            continue
        if reachedR:
            if vs(S, rule, H, fp, r, "R") == "faster" and vs(S, rule, H, fp, r, "A") != "slower":
                at.append(r)
        elif vs(S, rule, H, fp, r, "A") == "faster":
            at.append(r)
    return at


def lag(S, H, fp, key):
    s1, b1 = S[("S", H, fp, 0.001, False)], S[("B", H, fp, 0.001, False)]
    xs, xb = s1[key + "80"], b1[key + "80"]
    if xs and xb:
        a = dict(kind="value", v=xs / xb, nums=(xs, xb))
    elif xb:
        a = dict(kind="lower", v=s1["last"] / xb, nums=(s1["last"], xb), granular=(s1["last"] + 256) / xb)
    elif xs:
        a = dict(kind="upper", v=xs / b1["last"], nums=(xs, b1["last"]))
    else:
        a = dict(kind="none")
    bs, bb = best(S, "S", H, fp, key), best(S, "B", H, fp, key)
    o = dict(kind="value", v=bs["count"] / bb["count"], nums=(bs["count"], bb["count"])) if bs["count"] and bb["count"] else dict(kind="none")
    cls = "not formed"
    if a["kind"] == "value" and o["kind"] == "value":
        q = o["v"] / a["v"]
        cls = ("gone" if o["v"] <= 1.0 else "smaller") if q < 2 / 3 else "larger" if q > 1.5 else "the same within the scatter of one seed"
    elif a["kind"] == "lower" and o["kind"] == "value":
        cls = ("gone" if o["v"] <= 1.0 else "smaller") if o["v"] < 2 / 3 * a["v"] else "not decided"
    return dict(at=a, own=o, cls=cls, sbest=bs, bbest=bb)


# ---------------------------------------------------------------- the report's tables
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


def cell_equal(rep, mine):
    """Numbers are compared as written (commas removed); a cell without numbers is compared as words."""
    nr, nm = [x.replace(",", "") for x in NUM.findall(rep)], [x.replace(",", "") for x in NUM.findall(mine)]
    if nr or nm:
        return nr == nm
    norm = lambda s: re.sub(r"[*`]", "", s).strip().lower()
    return norm(rep) == norm(mine)


def rate_text(s):
    return f"{s['rate']:g}" + (" (control)" if s["product"] else "")


# ================================================================ 1
def section1(ctx):
    print("================ 1  inventory, roster, completeness, settings and identities")
    allf = names(E52); qf = names(QUAL)
    ctx["allf"], ctx["qf"] = allf, qf
    rec_names = [n for n in allf if re.fullmatch(re.escape(E52) + r"e52_[SB]_h\d+_f\d+_lr[\d.]+(_product)?_93683\.json", n)]
    pts = [n for n in allf if n.endswith(".pt")]
    logs = [n for n in allf if "/logs/" in n]
    intr = [n for n in allf if "/interrupted/" in n]
    tabs = [n for n in allf if n.endswith(("E52_TABLES.md", "E52_SUMMARY.json"))]
    srcs = [M + s for s in ("exp52_rate.py", "exp52_resume.py", "exp52_drive.py", "exp52_drive2.py", "exp52_tables.py", "exp46_decay.py")]
    on_default = [s for s in srcs if blob_id(s, DREF)]
    total = len(allf) + len(qf) + len(srcs) + 1
    print(f"  files in exp45/e52: {len(allf)} (receipts {len(rec_names)}, weights {len(pts)}, logs {len(logs)}, interrupted {len(intr)}, tables {len(tabs)}); "
          f"exp45/e52_qual: {len(qf)}; sources {len(srcs)} (on the default branch already: {len(on_default)}); the report 1; total {total}")
    ctx["counts"] = dict(receipts=len(rec_names), pts=len(pts), logs=len(logs), intr=len(intr), qual=len(qf), total=total, tabs=len(tabs))

    R = {}
    for n in rec_names:
        r = jshow(n)
        R[(r["rule"], r["H"], round(r["fraction"] * 100), r["rate"], r["product_control"])] = (n, r)
    ctx["R"] = R
    S = {k: summ(r) for k, (n, r) in R.items()}
    ctx["S"] = S

    # roster: 24 main, 4 controls, further runs by the brief's rule computed from the three main rates
    main = {(rule, H, fp, rate, False) for rule in "SB" for H in (256, 512) for fp in (25, 50) for rate in (0.0003, 0.001, 0.003)}
    ctrl = {(rule, 256, 50, rate, True) for rule in "SB" for rate in (0.0003, 0.003)}
    further = set()
    for rule in "SB":
        for H in (256, 512):
            for fp in (25, 50):
                b = best(S, rule, H, fp, "R", rates=(0.0003, 0.001, 0.003))
                if not b["tied"]:
                    b = best(S, rule, H, fp, "A", rates=(0.0003, 0.001, 0.003))
                if b["named"] == 0.003:
                    further.add((rule, H, fp, 0.01, False))
    ctx["further"] = further
    expected = main | ctrl | further
    print(f"  roster: main {len(main)}, controls {len(ctrl)}, further at 0.01 by the brief's rule {len(further)} "
          f"({', '.join(f'{k[0]} {WIDTH[k[1]]} f{k[2]}' for k in sorted(further))}); expected {len(expected)}, receipts {len(R)}, "
          f"missing {sorted(expected - set(R))}, extra {sorted(set(R) - expected)}")
    ctx["roster_ok"] = expected == set(R)

    # per receipt
    srcsha = {s: sha(show(M + s)) for s in ("exp52_rate.py", "exp52_resume.py", "exp46_decay.py", "exp41_scale.py", "arch26.py", "exp26_candidate.py",
                                           "arch20.py", "exp20_reference.py", "exp16_integrated.py", "arch17.py", "exp11_cell.py", "exp45.py",
                                           "exp3_state.py", "mod97_local.py")}
    ctx["srcsha"] = srcsha
    e41 = {}
    for rule in "SB":
        for H in (256, 512):
            for fp in (25, 50):
                e41[(rule, H, fp)] = jshow(f"{E41}e41_{rule}_h{H}_f{fp}_{SEED}.json")
    ctx["e41"] = e41
    bad = collections.defaultdict(list)
    for k, (n, r) in sorted(R.items()):
        rule, H, fp, rate, prod = k
        c = r["curve"]; dose = LONG_DOSE[H] if rate == 0.0003 else E41_DOSE[H]
        wd = 0.25 * 0.001 / rate if prod else 0.25
        steps_ok = [x["step"] for x in c] == list(range(0, dose + 1, 256))
        rest = [p for p in r["model"]["params_by_tensor"] if not p.startswith("book.")]
        g_exp = [dict(weight_decay=wd, lr=rate, betas=[0.9, 0.98], params=rest), dict(weight_decay=0.0, lr=rate, betas=[0.9, 0.98], params=["book.weight"])]
        tests = {
            "complete": r["complete"] is True, "not diverged": r["diverged"] is False, "dose": r["dose"] == dose, "updates_done": r["updates_done"] == dose,
            "rows 0..dose by 256": steps_ok, "finite losses every row": all(x["finite_losses"] for x in c),
            "losses finite numbers": all(x["loss"] is None or (x["loss"] == x["loss"] and abs(x["loss"]) < 1e30) for x in c),
            "optimizer steps = dose": r["optimizer_steps"] == [dose],
            "rule/local": r["local"] == (rule == "S"), "U": r["U"] == U_OF[H], "seed": r["seed"] == SEED, "fraction": r["fraction"] == fp / 100,
            "rate": r["rate"] == rate, "decay = expression": r["decay"] == wd, "product flag": r["product_control"] == prod,
            "optimizer record": r["optimizer"]["lr"] == rate and r["optimizer"]["weight_decay"] == wd and r["optimizer"]["betas"] == [0.9, 0.98],
            "groups expected (mine)": r["optimizer_groups_expected"] == g_exp, "groups at start": r["optimizer_groups_at_start"] == g_exp,
            "groups at end": r["optimizer_groups_at_end"] == g_exp, "groups read back": r["optimizer_read_back"]["groups"] == g_exp
                                                                                            and r["optimizer"]["groups"] == g_exp,
            "params": r["model"]["params"] == {256: 317457, 512: 1124113}[H], "data stream text": "split(t, f, seed)[0]" in r["data"],
            "runner sha": r["provenance"]["runner_sha256"] == srcsha["exp52_resume.py" if rate in (0.0003, 0.01) else "exp52_rate.py"],
            "sources sha = committed files": all(srcsha[s] == v for s, v in r["provenance"]["sources_sha256"].items()),
            "exp41_scale among sources": "exp41_scale.py" in r["provenance"]["sources_sha256"],
            "host": r["provenance"]["host"]["torch"] == "2.15.0+cpu" and r["provenance"]["host"]["torch_threads"] == 1,
            "env": r["provenance"]["environment_at_start"] == {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            "digest 4096 = E41 (flag)": r.get("digest_4096_equals_e41") is True,
            "digest 4096 = E41 (mine)": next(x["data_digest"]["sha256"] for x in c if x["step"] == 4096) == next(
                x["data_digest"]["sha256"] for x in e41[(rule, H, fp)]["curve"] if x["step"] == 4096) == r["e41_digest_4096"],
            "final digest batches": r["data_digest"]["batches"] == dose and r["data_digest"] == c[-1]["data_digest"],
            "length items = A add (flag)": r["length_test_12_items_equal_A_add89"] is True and r["length_test_items_in_taught"] == 0,
            "length 12 count = A add count every row": all(x["length_counts"]["12"] == x["A_counts"]["add89"] for x in c),
            "panel sample digest = E41": r["reduction_sample_sha256"] == e41[(rule, H, fp)]["reduction_sample_sha256"],
            "rates of runner per brief": (r["provenance"]["runner"].startswith("exp52_resume.py") == (rate in (0.0003, 0.01))),
        }
        for t, ok in tests.items():
            if not ok:
                bad[t].append(n.split("/")[-1])
    ctx["bad"] = bad
    print(f"  receipts checked: {len(R)}; tests per receipt: {len(tests)}; failures: {dict(bad) if bad else 'none'}")
    print(f"  runner per rate: exp52_rate.py at 0.001 and 0.003 ({sum(1 for k in R if k[3] in (0.001, 0.003))} runs), exp52_resume.py at 0.0003 and 0.01 "
          f"({sum(1 for k in R if k[3] in (0.0003, 0.01))} runs)")
    print(f"  sha256 of the committed exp52_rate.py {srcsha['exp52_rate.py'][:16]}, exp52_resume.py {srcsha['exp52_resume.py'][:16]}, "
          f"exp46_decay.py {srcsha['exp46_decay.py'][:16]}")
    ctx["e46_same"] = blob_id(M + "exp46_decay.py", WREF) == blob_id(M + "exp46_decay.py", W46C)
    srcs_same = {s: blob_id(M + s, WREF) == blob_id(M + s, DREF) for s in ("exp41_scale.py", "arch26.py", "exp26_candidate.py", "arch20.py",
                 "exp20_reference.py", "exp16_integrated.py", "arch17.py", "exp11_cell.py", "exp45.py", "exp3_state.py", "mod97_local.py")}
    print(f"  exp46_decay.py on the worker's commit is the same blob as on E46's commit {W46C}: {ctx['e46_same']}; "
          f"the 11 other sources the runs read are the same blobs as on the default branch at {DREF[:8]}: {all(srcs_same.values())}")
    ctx["srcs_same"] = all(srcs_same.values())

    # identities between runs
    by_f = collections.defaultdict(dict)
    for k, (n, r) in R.items():
        for x in r["curve"]:
            by_f[k[2]].setdefault(x["step"], set()).add(x["data_digest"]["sha256"])
    dig_ok = {fp: all(len(v) == 1 for v in d.values()) for fp, d in by_f.items()}
    ctx["dig_ok"] = dig_ok
    nrows = {fp: len(d) for fp, d in by_f.items()}
    print(f"  batch digests: at every scored row, every run of the same fraction (both rules, both widths, every rate, the controls) has "
          f"the same digest: f25 {dig_ok[25]} ({nrows[25]} distinct rows), f50 {dig_ok[50]} ({nrows[50]} distinct rows)")
    e41d = all(x["data_digest"] == y["data_digest"] for k, (n, r) in R.items() for x, y in zip(r["curve"], e41[k[:3]]["curve"]))
    print(f"  digests equal to E41's receipt of the same case at every row E41 has: {e41d}")
    ctx["e41d"] = e41d
    start = collections.defaultdict(set); row0 = collections.defaultdict(set)
    for k, (n, r) in R.items():
        start[k[1]].add(json.dumps(r["curve"][0]["norms"], sort_keys=True))
        row0[(k[1], k[2])].add(json.dumps([r["curve"][0][c] for c in ("A_counts", "R_counts", "taught_counts", "length_counts")]))
    ctx["start_ok"] = all(len(v) == 1 for v in start.values()) and all(len(v) == 1 for v in row0.values())
    print(f"  the same start network at each width (row-0 norms identical across rules, fractions and rates): "
          f"{all(len(v) == 1 for v in start.values())}; row-0 counts identical within width and fraction: {all(len(v) == 1 for v in row0.values())}")
    pan = collections.defaultdict(set)
    for k, (n, r) in R.items():
        pan[k[2]].add(json.dumps(r["reduction_sample_sha256"], sort_keys=True))
    print(f"  R samples identical within a fraction: {all(len(v) == 1 for v in pan.values())}")
    clock("section 1")


# ================================================================ 2
def section2(ctx):
    print("================ 2  figures of the report and of E52_TABLES.md, recomputed")
    S, R = ctx["S"], ctx["R"]
    rep = show(M + "E52_REPORT.md").decode(); ctx["report"] = rep
    tabs_md = show(E52 + "E52_TABLES.md").decode()
    T = md_tables(rep)

    def mine_row(s):
        return [f"{WIDTH[s['H']]}, {s['fp'] / 100:.2f}", s["rule"], rate_text(s), f"{s['decay']:.4g}", f"{s['dose']:,}", fu(s["A80"]), fu(s["R80"]),
                f"{s['A_final']:.2f}", f"{s['R_final']:.2f}", f"{s['taught_final']:.2f}",
                f"{s['len40_last']:.1f}, {s['len40_mean3']:.1f} ({s['len40_lo']:.1f}–{s['len40_hi']:.1f})", f"{s['rms16']:.2f}"]

    key_of = lambda row: (row[1], {"half": 256, "present": 512}[row[0].split(",")[0]], round(float(row[0].split(",")[1]) * 100),
                          float(row[2].split()[0]), "control" in row[2])
    # report table 1
    hdr, rows = find_table(T, "Width, f | Rule | Rate | Decay")
    n1 = 0
    for row in rows:
        s = S[key_of(row)]; mr = mine_row(s)
        for j in range(2, len(hdr)):
            cmp("2 report table 1", f"{row[0]} {row[1]} {row[2]}: {hdr[j]}", row[j], mr[j], cell_equal(row[j], mr[j])); n1 += 1
    print(f"  report table 1: {len(rows)} rows, {n1} cells compared")
    # E52_TABLES.md table 1
    TT = md_tables(tabs_md)
    hdr, rows = find_table(TT, "Width, f | Rule | Rate | Decay")
    n2 = 0
    for row in rows:
        row = [row[0], row[1], row[2].replace("product control", "control")] + row[3:]
        s = S[key_of(row)]; mr = mine_row(s) + ["no" if not R[key_of(row)][1]["diverged"] else "yes", f"`{R[key_of(row)][0].split('/')[-1]}`"]
        for j in range(2, len(hdr)):
            cmp("2 tables 1", f"{row[0]} {row[1]} {row[2]}: {hdr[j]}", row[j], mr[j], cell_equal(row[j], mr[j]) if j < 12 else row[j] == mr[j]); n2 += 1
    print(f"  E52_TABLES.md table 1: {len(rows)} rows, {n2} cells compared")
    # E52_TABLES.md table 2 (0.0003 at E41's dose)
    hdr, rows = find_table(TT, "Width, f | Rule | Control | Final A at E41")
    for row in rows:
        k = (row[1], {"half": 256, "present": 512}[row[0].split(",")[0]], round(float(row[0].split(",")[1]) * 100), 0.0003, row[2] == "product")
        a = S[k]["at41"]
        mr = [f"{a['A']:.2f}", f"{a['R']:.2f}", f"{a['len40_last']:.1f}, {a['len40_mean3']:.1f}", f"{S[k]['A_final']:.2f}, {S[k]['R_final']:.2f}"]
        for j in range(3, 7):
            cmp("2 tables 2", f"{row[0]} {row[1]} {row[2]}: {hdr[j]}", row[j], mr[j - 3], cell_equal(row[j], mr[j - 3]))
    # table 3 (norms)
    nrm = lambda n: "–" if n is None else " / ".join(f"{n[g]:.1f}" for g in GROUPS)
    hdr, rows = find_table(TT, "Width, f | Rule | Rate | At 80 on A")
    n3 = 0
    for row in rows:
        row2 = [row[0], row[1], row[2].replace("product control", "control")]
        s = S[key_of(row2)]
        mr = [nrm(s["A80_row"]["norms"] if s["A80_row"] else None), nrm(s["R80_row"]["norms"] if s["R80_row"] else None), nrm(s["norms_end"]),
              nrm(s["at41"]["norms"]) if s["at41"] else ""]
        for j in range(3, 7):
            cmp("2 tables 3", f"{row[0]} {row[1]} {row[2]}: {hdr[j]}", row[j], mr[j - 3], cell_equal(row[j], mr[j - 3])); n3 += 1
    print(f"  E52_TABLES.md tables 2 and 3 compared ({n3} norm cells)")

    # readings tables (report 4.2 and tables 4)
    def reading_row(rule, H, fp):
        g = gains(S, rule, H, fp)
        return [best(S, rule, H, fp, "R")["text"], best(S, rule, H, fp, "A")["text"]] + \
               [("not run" if vs(S, rule, H, fp, r, "R") == "not run" else f"{vs(S, rule, H, fp, r, 'R')}, {vs(S, rule, H, fp, r, 'A')}") for r in (0.0003, 0.003, 0.01)] + \
               ["yes, at " + ", ".join(f"{r:g}" for r in g) if g else "no"]
    for src, TX in (("2 report 4.2", T), ("2 tables 4", TT)):
        hdr, rows = find_table(TX, "Width, f | Rule | Best rate on R")
        for row in rows:
            H = {"half": 256, "present": 512}[row[0].split(",")[0]]; fp = round(float(row[0].split(",")[1]) * 100)
            mr = reading_row(row[1], H, fp)
            for j in range(2, 8):
                rp = re.sub(r"\s*\(read on .*\)$", "", row[j]).replace("tie with", "tied with")
                cmp(src, f"{row[0]} {row[1]}: {hdr[j]}", row[j], mr[j - 2], cell_equal(rp, mr[j - 2]) and
                    (re.sub(r"[\d.,() ]", "", rp).lower() == re.sub(r"[\d.,() ]", "", mr[j - 2]).lower()))
    # lag tables (report 4.3 and tables 5)
    for src, TX in (("2 report 4.3", T), ("2 tables 5", TT)):
        hdr, rows = find_table(TX, "Width, f | Score | Lag at 0.001")
        for row in rows:
            H = {"half": 256, "present": 512}[row[0].split(",")[0]]; fp = round(float(row[0].split(",")[1]) * 100)
            L = lag(S, H, fp, row[1])
            a = L["at"]; o = L["own"]
            nums_at = [f"{a['v']:.2f}"] + ([f"{a['nums'][0]:,}", f"{a['nums'][1]:,}"] if "(" in row[2] and NUM.findall(row[2].split("(", 1)[1]) else [])
            nums_own = [f"{o['v']:.2f}"] + ([f"{o['nums'][0]:,}", f"{o['nums'][1]:,}"] if "(" in row[5] else [])
            sb = L["sbest"]; bb = L["bbest"]
            mine = [" ".join(nums_at) + (" above" if a["kind"] == "lower" else ""), f"{sb['named']:g}" + (" tie 0.003" if len(sb["tied"]) > 1 else ""),
                    f"{bb['named']:g}", " ".join(nums_own), L["cls"]]
            repc = [row[2], row[3], row[4], row[5], row[6]]
            for j, (rc, mc) in enumerate(zip(repc, mine)):
                ok = cell_equal(rc, mc)
                if j == 0:
                    ok = ok and (("above" in rc) == (a["kind"] == "lower"))
                if j == 1:
                    ok = [x for x in NUM.findall(rc)][0] == f"{sb['named']:g}" and (("tie" in rc) == (len(sb["tied"]) > 1) or src == "2 tables 5")
                if j == 4:
                    ok = re.sub(r"[*]", "", rc).strip() == L["cls"]
                cmp(src, f"{row[0]} {row[1]}: {hdr[j + 2]}", rc, mc, ok)
    # control table (report 4.5 and tables 6)
    for src, TX in (("2 report 4.5", T), ("2 tables 6", TT)):
        hdr, rows = find_table(TX, "Rule | Rate | Decay | Updates to 80, A")
        for row in rows:
            prod = "control" in row[2]; k = (row[0], 256, 50, float(row[1]), prod); s = S[k]
            mr = [f"{s['decay']:.4g}" + (" (control)" if prod else " (main)"), fu(s["A80"]), fu(s["R80"]),
                  f"{vs(S, row[0], 256, 50, float(row[1]), 'R', prod)}, {vs(S, row[0], 256, 50, float(row[1]), 'A', prod)}",
                  f"{s['A_final']:.2f}", f"{s['R_final']:.2f}", f"{s['len40_last']:.1f}, {s['len40_mean3']:.1f}", nrm(s["norms_end"])]
            for j in range(2, 10):
                rc = row[j]
                if j == 2:
                    ok = cell_equal(rc, mr[0]) or (float(NUM.findall(rc)[0]) == round(s["decay"], 4) or NUM.findall(rc)[0] == f"{s['decay']:.4g}")
                elif j == 5:
                    ok = rc.replace("*", "").strip() == mr[3]
                else:
                    ok = cell_equal(rc, mr[j - 2])
                cmp(src, f"{row[0]} {row[1]} {row[2]}: {hdr[j]}", rc, mr[j - 2], ok)
    # tables 7 (predictions) is compared in section 3; the report's sentences:
    sent(ctx)
    clock("section 2")


def sent(ctx):
    """The figures in the report's sentences, each beside its recomputation."""
    S, R, rep = ctx["S"], ctx["R"], ctx["report"]
    g = lambda *k: S[k]
    c = lambda label, reported, mine, eq=None: cmp("2 sentences", label, reported, mine, eq)
    c("1 parameters, present width", "1,124,113", f"{R[('S', 512, 25, 0.001, False)][1]['model']['params']:,}")
    c("1 parameters, half width", "317,457", f"{R[('S', 256, 25, 0.001, False)][1]['model']['params']:,}")
    c("2 runs in all (24 + 4 + 7)", "35", str(len(R)))
    # 4.1
    rng12 = {k: s["len40_hi"] - s["len40_lo"] for k, s in S.items()}
    kmax = max(rng12, key=rng12.get)
    c("4.1 range of the length test over the last twelve rows, S half f0.50 0.01 ('reaches 30 points')", "30",
      f"{rng12[('S', 256, 50, 0.01, False)]:.1f}", abs(rng12[("S", 256, 50, 0.01, False)] - 30) < 0.5)
    print(f"  largest range of the length test over the last twelve rows: {rng12[kmax]:.1f} points ({kmax}, {S[kmax]['len40_lo']:.1f} to {S[kmax]['len40_hi']:.1f}); runs with a range of 30 points or more: "
          f"{sum(v >= 30 for v in rng12.values())}; of 15 or more: {sum(v >= 15 for v in rng12.values())} of {len(rng12)}")
    ctx["rng12"] = rng12
    a = S[("S", 512, 25, 0.0003, False)]
    c("4.1 S present f0.25 0.0003: R at 12,288", "54.80", f"{a['at41']['R']:.2f}")
    c("4.1 S present f0.25 0.0003: R at 40,960", "72.53", f"{a['R_final']:.2f}")
    below = []
    for k, s in S.items():
        if k[3] == 0.0003:
            k1 = (k[0], k[1], k[2], 0.001, False)
            for sc in ("A", "R"):
                if not (s["at41"][sc] < s[sc + "_final"]):
                    below.append((k, sc, "not below its own final"))
                if not (s["at41"][sc] < S[k1][sc + "_final"]):
                    below.append((k, sc, f"{s['at41'][sc]:.2f} not below 0.001's {S[k1][sc + '_final']:.2f}"))
    c("4.1 every run at 0.0003 at E41's dose below its own final level and below the run at 0.001, on A and R (exceptions)", "0", str(len(below)))
    ctx["below"] = below
    print(f"  exceptions to 'every run at 0.0003 below ... at E41's dose': {below if below else 'none'}")
    for rate, v in zip((0.0003, 0.001, 0.003), ("19.5", "21.5", "24.6")):
        c(f"4.1 S half f0.50 {rate:g}: recurrent norm at 80 on R", v, f"{S[('S', 256, 50, rate, False)]['R80_row']['norms']['recurrent']:.1f}")
    for rate, v in zip(RATES, ("23.0", "29.0", "40.4", "78.3")):
        c(f"4.1 S present f0.25 {rate:g}: recurrent norm at the end", v, f"{S[('S', 512, 25, rate, False)]['norms_end']['recurrent']:.1f}")
    exc = []
    for rule in "SB":
        for H in (256, 512):
            for fp in (25, 50):
                got = [(r, S[(rule, H, fp, r, False)]["R80_row"]["norms"]) for r in RATES
                       if (rule, H, fp, r, False) in S and S[(rule, H, fp, r, False)]["R80_row"]]
                for gname in ("input_map", "recurrent"):
                    v = [n[gname] for _, n in got]
                    if any(b <= a for a, b in zip(v, v[1:])):
                        exc.append(f"{rule} {WIDTH[H]} f{fp} {gname}: " + ", ".join(f"{r:g} {n[gname]:.1f}" for r, n in got))
    ctx["norm_exc"] = exc
    c("4.1 'within each case ... the norms of the input map and of the recurrent map at 80 on R grow with the rate' (cases that do not)", "0", str(len(exc)))
    print(f"  cases where a norm at 80 on R does not grow with the rate: {exc if exc else 'none'}")
    # 4.3 sentences
    L = lag(S, 512, 25, "R")
    c("4.3 present f0.25 R: two thirds of the bound ('1.81')", "1.81", f"{2 / 3 * L['at']['v']:.2f}")
    c("4.3 present f0.25 R: the bound ('above 2.71')", "2.71", f"{L['at']['v']:.4f}", f"{L['at']['v']:.2f}" == "2.71")
    LA = lag(S, 512, 25, "A")
    c("4.3 present f0.25 A: lag at 0.001", "2.79", f"{LA['at']['v']:.2f}"); c("4.3 present f0.25 A: lag at own rates", "1.33", f"{LA['own']['v']:.2f}")
    L5 = lag(S, 512, 50, "R")
    c("4.3 present f0.50: ratio of the lags ('1.45')", "1.45", f"{L5['own']['v'] / L5['at']['v']:.2f}")
    c("4.4 smallest lag at the rules' own rates", "1.33", f"{min(lag(S, H, fp, k)['own']['v'] for H in (256, 512) for fp in (25, 50) for k in 'RA'):.2f}")
    # 4.5 sentences
    c("4.5 S 0.0003 control: final R", "64.60", f"{S[('S', 256, 50, 0.0003, True)]['R_final']:.2f}")
    c("4.5 S 0.0003 main: final R", "92.37", f"{S[('S', 256, 50, 0.0003, False)]['R_final']:.2f}")
    for lab, v, k, gname in (("S 0.003 control input", "50.7", ("S", 256, 50, 0.003, True), "input_map"), ("S 0.003 control recurrent", "36.4", ("S", 256, 50, 0.003, True), "recurrent"),
                             ("S 0.003 main input", "33.5", ("S", 256, 50, 0.003, False), "input_map"), ("S 0.003 main recurrent", "22.9", ("S", 256, 50, 0.003, False), "recurrent"),
                             ("S 0.0003 control input", "11.9", ("S", 256, 50, 0.0003, True), "input_map"), ("S 0.0003 control recurrent", "13.2", ("S", 256, 50, 0.0003, True), "recurrent"),
                             ("S 0.0003 main input", "17.1", ("S", 256, 50, 0.0003, False), "input_map"), ("S 0.0003 main recurrent", "17.3", ("S", 256, 50, 0.0003, False), "recurrent")):
        c(f"4.5 {lab} norm at the end", v, f"{S[k]['norms_end'][gname]:.1f}")
    c("4.5 S control 0.003 R80 against main ('5,632 against 7,424')", "5,632 / 7,424", f"{fu(S[('S', 256, 50, 0.003, True)]['R80'])} / {fu(S[('S', 256, 50, 0.003, False)]['R80'])}")
    c("4.5 ratio of those counts within the band (two thirds to 1.5)", "within", "within" if 2 / 3 <= 5632 / 7424 <= 1.5 else "outside")
    c("4.5 B control 0.003 against 0.001 ('6,912 and 5,376 against 8,704 and 7,424')", "6,912 5,376 8,704 7,424",
      f"{fu(S[('B', 256, 50, 0.003, True)]['R80'])} {fu(S[('B', 256, 50, 0.003, True)]['A80'])} {fu(S[('B', 256, 50, 0.001, False)]['R80'])} {fu(S[('B', 256, 50, 0.001, False)]['A80'])}")
    # 6 predictions (figures)
    c("6 P52.3 S present f0.25 final R at 0.003 / 0.0003 / 0.001 / 0.01", "95.05 72.53 78.78 21.13",
      " ".join(f"{S[('S', 512, 25, r, False)]['R_final']:.2f}" for r in (0.003, 0.0003, 0.001, 0.01)))
    L = lag(S, 256, 25, "R")
    c("6 P52.4 lag on R at own rates; two thirds of 3.31", "2.05 2.21", f"{L['own']['v']:.2f} {2 / 3 * L['at']['v']:.2f}")
    c("6 P52.5 B half f0.50 at 0.003 against 0.001: R and A counts", "4,864 8,704 4,864 7,424",
      f"{fu(S[('B', 256, 50, 0.003, False)]['R80'])} {fu(S[('B', 256, 50, 0.001, False)]['R80'])} {fu(S[('B', 256, 50, 0.003, False)]['A80'])} {fu(S[('B', 256, 50, 0.001, False)]['A80'])}")
    c("6 P52.6 RMS change of R over the last 16 rows, S half f0.25, 0.0003 and 0.001", "1.77 2.45",
      f"{S[('S', 256, 25, 0.0003, False)]['rms16']:.2f} {S[('S', 256, 25, 0.001, False)]['rms16']:.2f}")
    # 7
    c("7 B half f0.50: final R at 0.003 and 0.001; RMS at 0.003", "91.25 93.97 3.97",
      f"{S[('B', 256, 50, 0.003, False)]['R_final']:.2f} {S[('B', 256, 50, 0.001, False)]['R_final']:.2f} {S[('B', 256, 50, 0.003, False)]['rms16']:.2f}")
    d = S[('B', 256, 50, 0.001, False)]['R_final'] - S[('B', 256, 50, 0.003, False)]['R_final']
    print(f"  section 7: the difference of final R is {d:.2f} points, {d / S[('B', 256, 50, 0.003, False)]['rms16']:.2f} of the RMS change 3.97 the report sets beside it")
    ctx["sec7_diff"] = d
    nbest3 = sum(1 for rule in "SB" for H in (256, 512) for fp in (25, 50) if best(S, rule, H, fp, "R")["named"] == 0.003)
    c("7 cases whose best rate on R is 0.003", "7", str(nbest3))
    hi = [f"{rule} {WIDTH[H]} f{fp}" for rule in "SB" for H in (256, 512) for fp in (25, 50)
          if max(RATES, key=lambda r: S[(rule, H, fp, r, False)]["R_final"] if (rule, H, fp, r, False) in S else -1) == 0.0003]
    print(f"  cases in which the run at 0.0003 ends highest on R: {len(hi)} ({', '.join(hi)})")
    ctx["hi0003"] = hi
    # 9 counts
    cn = ctx["counts"]
    c("9 files added by E52", "153", str(cn["total"])); c("9 receipts and weights", "70", str(cn["receipts"] + cn["pts"]))
    c("9 logs", "39", str(cn["logs"])); c("9 interrupted files", "11", str(cn["intr"]))
    for s, v in (("exp52_rate.py", "f633f6f4f3759b36"), ("exp52_resume.py", "6531525f8bffec5f"), ("exp52_drive.py", "50865187911102cc"),
                 ("exp52_drive2.py", "6c0ff96a1d7aa5a8"), ("exp52_tables.py", "78b83f736d2a6ba5"), ("exp46_decay.py", "d7e98b3001629589")):
        c(f"9 sha256 of {s}", v, sha(show(M + s))[:16])


# ================================================================ 3
def section3(ctx):
    print("================ 3  readings in the brief's words")
    S = ctx["S"]; rep = ctx["report"]
    print("  case (rule, width, fraction): best rate on R; on A; class of 0.0003 / 0.003 / 0.01 against 0.001 (R, A); gains at")
    G = {}
    for H in (256, 512):
        for fp in (50, 25):
            for rule in "SB":
                G[(rule, H, fp)] = gains(S, rule, H, fp)
                cls = " | ".join(f"{vs(S, rule, H, fp, r, 'R')}, {vs(S, rule, H, fp, r, 'A')}" for r in (0.0003, 0.003, 0.01))
                print(f"    {rule} {WIDTH[H]} f{fp}: {best(S, rule, H, fp, 'R')['text']}; {best(S, rule, H, fp, 'A')['text']}; {cls}; "
                      f"{', '.join(f'{r:g}' for r in G[(rule, H, fp)]) or 'none'}")
    ctx["G"] = G
    print("  counts to 80 (R; A) by rate, and the ratio to the count at 0.001:")
    for H in (256, 512):
        for fp in (50, 25):
            for rule in "SB":
                parts = []
                for r in RATES:
                    k = (rule, H, fp, r, False)
                    if k in S:
                        s, s1 = S[k], S[(rule, H, fp, 0.001, False)]
                        q = lambda key: f"{s[key + '80'] / s1[key + '80']:.3f}" if s[key + "80"] and s1[key + "80"] else "–"
                        parts.append(f"{r:g}: {fu(s['R80'])} ({q('R')}); {fu(s['A80'])} ({q('A')})")
                print(f"    {rule} {WIDTH[H]} f{fp}: " + " | ".join(parts))
    print("  lags (sweep over back-propagation), at 0.001 and at each rule's best rate, with the ratio of the two and the reading:")
    LAGS = {}
    for H in (256, 512):
        for fp in (50, 25):
            for key in "RA":
                L = lag(S, H, fp, key); LAGS[(H, fp, key)] = L
                a = L["at"]; o = L["own"]
                at = (f"{a['v']:.4f} ({a['nums'][0]:,}/{a['nums'][1]:,})" if a["kind"] == "value" else
                      f"above {a['v']:.4f} (bound {a['nums'][0]:,}/{a['nums'][1]:,}; with the next row, {a['nums'][0] + 256:,}, {a['granular']:.4f})")
                print(f"    {WIDTH[H]} f{fp} {key}: at 0.001 {at}; own {o['v']:.4f} ({o['nums'][0]:,}/{o['nums'][1]:,}; S at {L['sbest']['text']}, "
                      f"B at {L['bbest']['text']}); own/at0.001 {o['v'] / a['v']:.3f}; two thirds of the lag at 0.001 {2 / 3 * a['v']:.4f}; reading: {L['cls']}")
    ctx["LAGS"] = LAGS
    print("  E41's lag at 0.001 on the three seeds (from E41's receipts on the worker's commit; '>' marks a bound from the last recordable count):")
    for H in (256, 512):
        for fp in (50, 25):
            for key in "RA":
                vals = []
                for sd in (73129, 93683, 116089):
                    rs, rb = jshow(f"{E41}e41_S_h{H}_f{fp}_{sd}.json"), jshow(f"{E41}e41_B_h{H}_f{fp}_{sd}.json")
                    xs, xb = to80(rs["curve"], key), to80(rb["curve"], key)
                    vals.append(f"{xs / xb:.2f}" if xs and xb else (f">{(rs['dose'] - 512) / xb:.2f}" if xb else "not formed"))
                print(f"    {WIDTH[H]} f{fp} {key}: seeds 73129, 93683, 116089: {', '.join(vals)}")
    # further runs and the best rate after them
    print(f"  further runs called for (best on R among the three main rates is 0.003): {len(ctx['further'])}; "
          f"none diverged: {all(not ctx['R'][k][1]['diverged'] for k in ctx['further'])}")
    after = {k[:3]: best(S, k[0], k[1], k[2], "R")["text"] for k in ctx["further"]}
    print(f"  best rate on R once 0.01 is included, in those cases: {sorted(set(after.values()))} (no case at 'at or above 0.01': {all(v != '0.01' for v in after.values())})")
    slower001 = {k: vs(S, k[0], k[1], k[2], 0.01, "R") for k in ctx["further"]}
    print(f"  class of 0.01 against 0.001 on R: {collections.Counter(slower001.values())}")
    v003 = {k: (S[k]["R80"], S[(k[0], k[1], k[2], 0.003, False)]["R80"]) for k in ctx["further"]}
    print(f"  no run at 0.01 faster than its run at 0.003 on R: {all(a is None or a > b for a, b in v003.values())}")
    # predictions
    P = {}
    P["P52.1"] = all(any(r in (0.001, 0.003) for r in best(S, "B", H, fp, "R")["tied"]) for H in (256, 512) for fp in (25, 50))
    P["P52.2"] = any(G[("S", H, fp)] for H in (256, 512) for fp in (25, 50))
    P["P52.3"] = max(S[("S", 512, 25, r, False)]["R_final"] for r in RATES) >= 85
    P["P52.4"] = LAGS[(256, 25, "R")]["cls"] in ("smaller", "gone")
    P["P52.5"] = not any(G[("B", H, fp)] for H in (256, 512) for fp in (25, 50))
    P["P52.6"] = S[("S", 256, 25, 0.0003, False)]["rms16"] < S[("S", 256, 25, 0.001, False)]["rms16"]
    hdr, rows = find_table(md_tables(rep), " | Statement | Chance | Outcome")
    for row in rows:
        occurred = row[3].replace("*", "").startswith("Occurred")
        cmp("3 predictions", f"{row[0]} outcome", "occurred" if occurred else "did not occur", "occurred" if P[row[0]] else "did not occur")
    TT = md_tables(show(E52 + "E52_TABLES.md").decode())
    hdr, rows = find_table(TT, " | Statement | Outcome")
    for row in rows:
        cmp("3 tables 7", f"{row[0]} outcome", row[2], "true" if P[row[0]] else "false")
    s0, s1 = S[("S", 256, 25, 0.0003, False)], S[("S", 256, 25, 0.001, False)]
    print(f"  predictions: {P}")
    print(f"  P52.6 in detail: RMS change of R over the last 16 rows (15 changes): 0.0003 {s0['rms16']:.3f}, 0.001 {s1['rms16']:.3f}; "
          f"over the last 17 rows (16 changes): {s0['rms17']:.3f}, {s1['rms17']:.3f}; at 0.0003 over the 16 rows ending at E41's dose 40,960: {s0['at41']['rms16']:.3f}; "
          f"the last 16 rows lie at {s0['dose'] - 15 * 256:,} to {s0['dose']:,} updates (0.0003) and {s1['dose'] - 15 * 256:,} to {s1['dose']:,} (0.001)")
    ctx["P"] = P
    # explanation under test
    sg = [(H, fp) for H in (256, 512) for fp in (25, 50) if G[("S", H, fp)]]
    cond = {(H, fp): (bool(G[("B", H, fp)]), LAGS[(H, fp, "R")]["cls"], LAGS[(H, fp, "A")]["cls"]) for H, fp in sg}
    refuted = (not sg) or all(b and lr not in ("smaller", "gone") and la not in ("smaller", "gone") for b, lr, la in cond.values())
    print(f"  explanation under test: the sweep gains at {[(WIDTH[H], fp) for H, fp in sg]}; there (back-propagation gains, lag reading on R, on A): {cond}; refuted: {refuted}")
    ctx["refuted"] = refuted
    # after the trial
    other = [(H, fp) for H in (256, 512) for fp in (50, 25) if G[("S", H, fp)] or G[("B", H, fp)]
             or LAGS[(H, fp, "R")]["cls"] != "the same within the scatter of one seed" or LAGS[(H, fp, "A")]["cls"] != "the same within the scatter of one seed"]
    print(f"  widths and fractions for the other seeds (either rule gains, or the lag changes class on R or on A): {[(WIDTH[H], fp) for H, fp in other]}")
    ll = sorted({r for fp in (25, 50) for r in G[("S", 256, fp)]})
    print(f"  the sweep gains at the half width: {bool(ll)}, at the rate(s) {ll}: the brief's condition for asking again for the long lives is met on this seed")
    ctx["other"], ctx["ll"] = other, ll
    # how far each reading is from changing: the fewest rows (of 256 updates) by which one count to 80 must move to change it
    def flip(fn, keys):
        base = fn(S); out = None
        for k, sc in keys:
            x = S[k][sc + "80"]
            if x is None:
                continue
            for n in range(1, 41):
                hit = None
                for sign in (-1, 1):
                    nx = x + sign * 256 * n
                    if nx < 256:
                        continue
                    S2 = dict(S); S2[k] = dict(S[k]); S2[k][sc + "80"] = nx
                    if fn(S2) != base:
                        hit = (n, k, sc, nx, fn(S2)); break
                if hit:
                    if out is None or hit[0] < out[0]:
                        out = hit
                    break
        return base, out
    print("  margins: the fewest rows by which a single count to 80 must move to change a reading (count, its new value, the new reading):")
    MG = {}
    for H in (256, 512):
        for fp in (50, 25):
            for rule in "SB":
                keys = [(k, sc) for k in S if k[:3] == (rule, H, fp) and not k[4] for sc in "RA"]
                base, out = flip(lambda SS, rule=rule, H=H, fp=fp: tuple(gains(SS, rule, H, fp)), keys)
                MG[("gains", rule, H, fp)] = out
                print(f"    gains {rule} {WIDTH[H]} f{fp} (now {list(base) or 'none'}): " + (f"{out[0]} row(s): {out[1][0]} {out[1][3]:g} on {out[2]} to {out[3]:,} gives {list(out[4]) or 'none'}" if out else "more than 40 rows"))
            for key in "RA":
                keys = [(k, key) for k in S if k[1:3] == (H, fp) and not k[4]]
                base, out = flip(lambda SS, H=H, fp=fp, key=key: lag(SS, H, fp, key)["cls"], keys)
                MG[("lag", H, fp, key)] = out
                print(f"    lag {WIDTH[H]} f{fp} {key} (now {base}): " + (f"{out[0]} row(s): {out[1][0]} {out[1][3]:g} on {out[2]} to {out[3]:,} gives {out[4]}" if out else "more than 40 rows"))
    ctx["MG"] = MG
    for rate in (0.0003, 0.003):
        sc, bc = S[("S", 256, 50, rate, True)], S[("B", 256, 50, rate, True)]
        q = lambda key: f"{sc[key + '80'] / bc[key + '80']:.2f}" if sc[key + "80"] and bc[key + "80"] else "not formed"
        print(f"  the ratio of the sweep's count to back-propagation's within the product control at {rate:g} (no reading is fixed for it): "
              f"R {fu(sc['R80'])}/{fu(bc['R80'])} = {q('R')}; A {fu(sc['A80'])}/{fu(bc['A80'])} = {q('A')}")
    clock("section 3")


# ================================================================ 4
ROWKEYS = ("A_counts", "R_counts", "taught_counts", "length_counts", "norms", "data_digest", "loss", "finite_losses")
PAT = re.compile(r" (\d+): A ([\d.]+) R ([\d.]+) taught ([\d.]+) len40 ([\d.]+) rec-norm ([\d.]+) cpu (\d+)\+(\d+)s")


def section4(ctx):
    print("================ 4  resumptions, interrupted runs, check 4, processes at once")
    R, S = ctx["R"], ctx["S"]
    res = {k: r.get("resumed") for k, (n, r) in R.items() if r.get("resumed")}
    print(f"  receipts with a 'resumed' record: {len(res)} of {len(R)}")
    for k, lst in sorted(res.items()):
        n, r = R[k]
        for e in lst:
            psame = e["provenance"]["sources_sha256"] == r["provenance"]["sources_sha256"] and e["provenance"]["runner_sha256"] == r["provenance"]["runner_sha256"]
            print(f"    {n.split('/')[-1]}: resumed at update {e['at_update']:,} at {e['at']}, replay {e['replay_wall']:.1f} s wall; "
                  f"commit at start {r['provenance']['git']['commit'][:8]}, at resumption {e['provenance']['git']['commit'][:8]}; sources and runner unchanged: {psame}")
    ctx["res"] = res
    # the rows after the resumption against an uninterrupted run of the same fraction and dose
    for k in sorted(res):
        n, r = R[k]; at = res[k][0]["at_update"]
        ref = [kk for kk in R if kk[2] == k[2] and kk[3] == 0.0003 and kk not in res]
        refk = sorted(ref)[0]; rr = {x["step"]: x["data_digest"] for x in R[refk][1]["curve"]}
        after = [x for x in r["curve"] if x["step"] > at]
        eq = all(x["data_digest"] == rr[x["step"]] for x in after)
        print(f"    {k[0]} f{k[2]} {k[3]:g}{' product' if k[4] else ''}: {len(after)} rows after the resumption; batch digests equal to those of the "
              f"uninterrupted run {refk[0]} h{refk[1]} f{refk[2]} {refk[3]:g} at every one of them: {eq}")
    # the logs: rows printed before the restart after the checkpoint, and the same rows printed after the resumption
    dup = {}
    for k in sorted(res):
        n, r = R[k]
        L = show(E52 + "logs/" + n.split("/")[-1].replace(".json", ".log")).decode().splitlines()
        i = [j for j, l in enumerate(L) if "resumed at" in l][0]
        first = {int(m.group(1)): m.groups()[1:6] for l in L[:i] for m in [PAT.search(l)] if m}
        second = {int(m.group(1)): m.groups()[1:6] for l in L[i:] for m in [PAT.search(l)] if m}
        d = sorted(set(first) & set(second))
        cpu_before = [m for l in L[:i] for m in [PAT.search(l)] if m][-1]
        cpu_ck = next(x for x in r["curve"] if x["step"] == res[k][0]["at_update"])
        rep_cpu = int(cpu_before.group(7)) + int(cpu_before.group(8)) - cpu_ck["cpu_updates"] - cpu_ck["cpu_scoring"]
        dup[k] = (len(d), d[0], d[-1], all(first[s] == second[s] for s in d), rep_cpu, res[k][0]["replay_cpu"])
        print(f"    log of {n.split('/')[-1]}: {len(d)} rows ({d[0]:,} to {d[-1]:,}) printed both before the restart and after the resumption; "
              f"identical at the printed precision (A, R, taught, length 40, recurrent norm): {dup[k][3]}; CPU of the repeated updates and rows about {rep_cpu:.0f} s, "
              f"replay {res[k][0]['replay_cpu']:.0f} s")
    ctx["dup"] = dup
    # interrupted runs against the runs started again
    names_i = [x for x in ctx["allf"] if "/interrupted/" in x]
    print("  interrupted runs (partial receipts and logs in interrupted/):")
    icpu_rec = icpu_log = 0.0; ires = []
    for x in sorted(names_i):
        base = x.split("/")[-1]; stem = base.split(".json")[0].split(".log")[0]
        m = re.match(r"e52_([SB])_h(\d+)_f(\d+)_lr([\d.]+)(_product)?_93683", stem)
        k = (m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4)), bool(m.group(5)))
        fin = {y["step"]: y for y in R[k][1]["curve"]}
        if ".json." in base:
            p = jshow(x); c = p["curve"]
            eq = all({kk: y[kk] for kk in ROWKEYS} == {kk: fin[y["step"]][kk] for kk in ROWKEYS} for y in c)
            icpu_rec += c[-1]["cpu_updates"] + c[-1]["cpu_scoring"]
            ires.append((stem, "receipt", len(c), c[-1]["step"], eq))
            print(f"    {base}: partial receipt, {len(c)} rows to {c[-1]['step']:,}, runner {p['provenance']['runner']}; every row equal to the restarted run's "
                  f"(counts, norms, digest, loss): {eq}; CPU {c[-1]['cpu_updates'] + c[-1]['cpu_scoring']:.0f} s")
        else:
            L = show(x).decode().splitlines()
            ms = [PAT.search(l) for l in L]; ms = [m for m in ms if m]
            eq = all((f"{fin[int(m.group(1))]['A']:.2f}", f"{fin[int(m.group(1))]['R']:.2f}", f"{fin[int(m.group(1))]['taught']:.2f}",
                      f"{fin[int(m.group(1))]['len40']:.1f}", f"{fin[int(m.group(1))]['norms']['recurrent']:.2f}") == m.groups()[1:6] for m in ms)
            icpu_log += int(ms[-1].group(7)) + int(ms[-1].group(8))
            ires.append((stem, "log", len(ms), int(ms[-1].group(1)), eq))
            print(f"    {base}: log, {len(ms)} rows to {int(ms[-1].group(1)):,}; every printed row equal to the restarted run's: {eq}; CPU at its last row "
                  f"{int(ms[-1].group(7)) + int(ms[-1].group(8))} s")
    ctx["ires"] = ires
    ctx["icpu"] = (icpu_rec / 3600, icpu_log / 3600)
    print(f"  CPU of the interrupted runs: {icpu_rec / 3600:.2f} h from the five partial receipts; {icpu_log / 3600:.2f} h from the six logs (to the last printed row)")
    # check 4
    q = jshow(QUAL + "check4.json")
    print(f"  check4.json: pass {q['pass']}; per rule: " + "; ".join(
        f"{r}: killed exit {q[r]['killed_exit_code']}, resumed at {q[r]['resumed']}, resume exit {q[r]['resume_exit_code']}, tensors i/ii/iii equal "
        f"{len(set(q[r]['tensors'].values())) == 1} ({q[r]['tensors']['i'][:8]}), rows {q[r]['n_rows']}, steps {q[r]['optimizer_steps']}" for r in "SB"))
    c4 = {}
    for r in "SB":
        runs = {v: jshow(f"{QUAL}check4_runs/{r}/{v}/e52_{r}_h256_f25_lr0.003_dry3072_93683.json") for v in ("i", "ii", "iii")}
        rows = lambda x: [{kk: y[kk] for kk in ROWKEYS} | dict(step=y["step"]) for y in x["curve"]]
        c4[r] = dict(t=len({x["tensors_sha256"] for x in runs.values()}) == 1, rows=rows(runs["i"]) == rows(runs["ii"]) == rows(runs["iii"]),
                     runners=[runs[v]["provenance"]["runner"].split()[0] for v in ("i", "ii", "iii")], ck=runs["iii"].get("checkpoint_every"),
                     resumed=[e["at_update"] for e in runs["iii"].get("resumed", [])], n=len(runs["i"]["curve"]),
                     cpu=sum(x["cpu_updates"] + x["cpu_scoring"] for x in runs.values()))
        print(f"    recomputed from the run receipts, rule {r}: one tensors_sha256 for i, ii, iii {c4[r]['t']}; all {c4[r]['n']} rows equal (counts, norms, digest, loss) "
              f"{c4[r]['rows']}; runners {c4[r]['runners']}; checkpoint every {c4[r]['ck']}; iii resumed at {c4[r]['resumed']}; CPU in the three receipts {c4[r]['cpu']:.0f} s")
    ctx["c4"] = c4; ctx["c4cpu"] = q["cpu"]
    print(f"  check4.json records {q['cpu']:.0f} CPU-s (the parent process only); the six run receipts record {sum(v['cpu'] for v in c4.values()):.0f} CPU-s")
    fa = [x for x in ctx["qf"] if "attempt1_failed_runs" in x]
    fr = {x: jshow(x) for x in fa}
    good = {r: jshow(f"{QUAL}check4_runs/S/{r}/e52_S_h256_f25_lr0.003_dry3072_93683.json") for r in ("i", "ii")}
    same = all(fr[x]["tensors_sha256"] == good[x.split("/")[-2]]["tensors_sha256"] for x in fa)
    flog = show(QUAL + "logs/check4_attempt1_failed_20261118T013037Z.log").decode()
    print(f"  failed first attempt: {len(fa)} receipts ({', '.join('/'.join(x.split('/')[-3:-1]) for x in fa)}), equal in tensors_sha256 to the second attempt's: {same}; "
          f"the log ends in FileNotFoundError for the killed run's receipt (iii): {'FileNotFoundError' in flog and '/iii/' in flog}; "
          f"CPU of the failed attempt's receipts {sum(x['cpu_updates'] + x['cpu_scoring'] for x in fr.values()):.0f} s")
    ctx["fa"] = (len(fa), same)
    # processes at once, from LAUNCHES.txt and the receipts' wall clock
    L = show(E52 + "logs/LAUNCHES.txt").decode().splitlines()
    starts = {}
    iv = []
    restarts = [utc(l.split()[0]) for l in L if "worker restart" in l]
    for l in L:
        m = re.match(r"(\S+) start (\(resume\) )?(e52_\S+)", l)
        if m:
            starts.setdefault(m.group(3), []).append((utc(m.group(1)), bool(m.group(2))))
    for nm, lst in starts.items():
        m = re.match(r"e52_([SB])_h(\d+)_f(\d+)_lr([\d.]+)(_product)?_93683", nm)
        k = (m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4)), bool(m.group(5)))
        r = R[k][1]
        fresh = [t for t, rs in lst if not rs]; resumes = [t for t, rs in lst if rs]
        last_fresh = fresh[-1]
        for t in fresh[:-1]:
            iv.append((t, min(x for x in restarts + [utc("2026-11-18T00:44:00Z")] if x > t) if any(x > t for x in restarts) else t, nm + " (interrupted)"))
        if resumes:
            e = r["resumed"][0]; ckw = next(x["wall"] for x in r["curve"] if x["step"] == e["at_update"])
            killed = min(x for x in restarts if x > last_fresh)
            iv.append((last_fresh, killed, nm + " (before the restart)"))
            iv.append((resumes[-1], resumes[-1] + e["replay_wall"] + r["wall"] - ckw, nm + " (resumed)"))
        else:
            iv.append((last_fresh, last_fresh + r["wall"], nm))
    # the restart of 00:44 is not stamped in the launch log; the interrupted runs' partial work bounds it
    c4s = [utc(l.split()[0]) for l in L if l.split()[1:3] == ["start", "check4"]]
    iv.append((c4s[-1], c4s[-1] + q["wall"], "check 4"))
    a1 = c4s[0] + sum(fr[x]["wall"] for x in fa)
    iv.append((c4s[0], a1, "check 4, attempt 1"))
    print(f"  check 4's first attempt: started {hm(c4s[0])}; its two completed runs took {sum(fr[x]['wall'] for x in fa):.0f} s of wall clock, so it ended about {hm(a1)}; "
          f"the launch log notes its failure at {[l for l in L if 'attempt 1 failed' in l][0].split()[0][11:19]}")
    ev = sorted([(a, 1, n) for a, b, n in iv] + [(b, -1, n) for a, b, n in iv])
    cur = mx = 0; at = None; live = set()
    for t, d, n in ev:
        cur += d
        live.add(n) if d > 0 else live.discard(n)
        if cur > mx:
            mx, at, who = cur, t, sorted(live)
    print(f"  processes at once, from the launch times and the receipts' wall clock (interrupted runs counted to the next restart): at most {mx} "
          f"(first reached at {hm(at)}: {who})")
    ctx["maxproc"] = mx
    # the first restart (not stamped in the launch log): the killed runs' logs bound it from below
    kl = []
    for nm in ("e52_S_h256_f25_lr0.003_93683", "e52_B_h256_f50_lr0.003_product_93683", "e52_S_h256_f50_lr0.003_product_93683"):
        lg = [x for x in ctx["allf"] if "/interrupted/" in x and nm + ".log" in x][0]
        ms = [m for m in map(PAT.search, show(lg).decode().splitlines()) if m]
        t0 = [t for t, rs in starts[nm] if not rs][0]
        kl.append(t0 + int(ms[-1].group(7)) + int(ms[-1].group(8)))
    print(f"  the first restart: the three killed runs had used CPU up to {hm(max(kl))} at the latest (start plus CPU of the last printed row); the driver began again at "
          f"{[l for l in L if 'driver begin' in l][1].split()[0][11:19]}")
    ctx["restart1"] = max(kl)
    # departure 4: how far the sweep's run at 0.0003, present width, a quarter of the pairs, had come when the further runs of that case began
    nm = "e52_S_h512_f25_lr0.0003_93683"; t0 = [t for t, rs in starts[nm] if not rs][-1]
    tf = min(t for n2 in ("e52_B_h512_f25_lr0.01_93683",) for t, rs in starts[n2])
    cv = R[("S", 512, 25, 0.0003, False)][1]["curve"]
    past = max(x["step"] for x in cv if t0 + x["wall"] <= tf)
    best_after = [x["step"] for x in cv if x["step"] >= past]
    print(f"  departure 4: when the run at 0.01 of back-propagation, present width, f = 0.25, started ({hm(tf)}), the sweep's run at 0.0003 of that case had "
          f"passed update {past:,} (rows by wall clock); its count to 80 on R and on A, if any, could then be no lower than {past - 512:,}")
    ctx["dep4"] = past
    s01 = [t for t, rs in starts["e52_S_h512_f25_lr0.01_93683"]][0]
    print(f"  the sweep's run at 0.01 of that case started at {hm(s01)}; the sweep's run at 0.0003 of that case ended about {hm(t0 + cv[-1]['wall'])} "
          f"(start plus its wall clock); the launch log records the decision for both further runs of present width, f = 0.25, at "
          f"{[l for l in L if 'begin further B 512 25 S 512 25' in l][0].split()[0][11:19]}")
    lg = subprocess.run(["git", "log", "--format=%h %ad %s", "--date=format-local:%Y-%m-%d %H:%M", f"{DREF}..{WREF}"], capture_output=True, text=True,
                        env=dict(os.environ, TZ="UTC")).stdout.splitlines()
    rc = [l for l in lg if l.split(" ", 3)[3].startswith("E52: exp52_resume.py")]
    print(f"  the commit that added the resuming runner: {rc[0][:25]} (UTC)")
    clock("section 4")


# ================================================================ 5
def section5(ctx):
    print("================ 5  checks 1a, 1b, 2, 3; costs")
    R, S, e41 = ctx["R"], ctx["S"], ctx["e41"]
    src41 = show(M + "exp41_scale.py").decode()
    dry = dict(re.findall(r'"([SB])": "([0-9a-f]{8})"', re.search(r"DRY2048 = \{[^}]*\}", src41).group(0)))
    d2048 = re.search(r'DIGEST_2048 = "([0-9a-f]+)"', src41).group(1)
    q1 = jshow(QUAL + "check1a.json")
    for r in "SB":
        run = jshow(f"{QUAL}check1a_runs/e52_{r}_h512_f50_lr0.001_dry2048_73129.json")
        ok = run["tensors_sha256"].startswith(dry[r]) and run["data_digest"]["sha256"].startswith(d2048) and run["tensors_sha256"] == q1[r]["tensors_sha256"]
        cmp("5 check 1a", f"{r}: dry run of 2,048 updates ends on E41's DRY2048 prefix, digest prefix {d2048}", "PASS", "PASS" if ok else "FAIL")
        print(f"  check 1a {r}: run receipt tensors {run['tensors_sha256'][:8]} (E41's constant {dry[r]}), digest {run['data_digest']['sha256'][:16]}, rate {run['rate']}, decay {run['decay']}")
    ok_all = True
    for rule in "SB":
        for H in (256, 512):
            for fp in (25, 50):
                r = R[(rule, H, fp, 0.001, False)][1]; e = e41[(rule, H, fp)]
                okk = (r["tensors_sha256"] == e["tensors_sha256"] and len(r["curve"]) == len(e["curve"])
                       and all(x["step"] == y["step"] and all(x[c] == y[c] for c in ("A_counts", "R_counts", "taught_counts", "data_digest", "loss"))
                               for x, y in zip(r["curve"], e["curve"])))
                ok_all = ok_all and okk
                print(f"  check 1b {rule} {WIDTH[H]} f{fp}: tensors {r['tensors_sha256'][:8]} = E41 {e['tensors_sha256'][:8]}; {len(r['curve'])} rows; "
                      f"counts, digests and losses equal at every row: {okk}")
    cmp("5 check 1b", "eight runs at 0.001 equal E41's (final tensors; A, R, taught counts, digests and losses at every row)", "PASS", "PASS" if ok_all else "FAIL")
    q2 = jshow(QUAL + "check2.json")
    lo = min(v["decay_recovered"] for rr in q2["rates"].values() for v in rr["a"]["main (0.25)"]["per_tensor"].values() if "decay_recovered" in v)
    hi = max(v["decay_recovered"] for rr in q2["rates"].values() for v in rr["a"]["main (0.25)"]["per_tensor"].values() if "decay_recovered" in v)
    cmp("5 check 2", "(a) decay recovered for d = 0.25, range", "0.24994–0.25000", f"{lo:.5f}–{hi:.5f}")
    for rate, rep in (("0.0003", "0.83327–0.83328"), ("0.003", "0.083326–0.083328")):
        v = [x["decay_recovered"] for x in q2["rates"][rate]["a"]["product (0.25 * 0.001 / rate)"]["per_tensor"].values() if "decay_recovered" in x]
        cmp("5 check 2", f"(a) decay recovered, product control at {rate}", rep, f"{min(v):.5f}–{max(v):.5f}" if rate == "0.0003" else f"{min(v):.6f}–{max(v):.6f}")
    bt = all(x["bit_equal"] for rr in q2["rates"].values() for a in rr["a"].values() for kk, x in a["per_tensor"].items() if kk == "book.weight")
    cmp("5 check 2", "(a) byte table bit-equal", "yes", "yes" if bt else "no")
    for rate, rep in (("0.0003", "0.300000"), ("0.003", "3.00000"), ("0.01", "10.0000")):
        v = [x["ratio_of_change"] for x in q2["rates"][rate]["b"].values()]
        cmp("5 check 2", f"(b) ratio of change at {rate} (min, max)", rep, f"{min(v):.6g} {max(v):.6g}",
            all(abs(x - float(rep)) / float(rep) < 1e-5 for x in v))
    mres = max(x["max_residual_rel_to_max_change"] for rr in q2["rates"].values() for x in rr["b"].values())
    cmp("5 check 2", "(b) largest residual, share of the largest change", "0.1%", f"{100 * mres:.3f}%", round(100 * mres, 1) == 0.1)
    q3 = jshow(QUAL + "check3.json")
    pu = {k: v["cpu_per_update"] for k, v in q3["runs"].items()}; pr = {k: v["cpu_per_scored_row"] for k, v in q3["runs"].items()}
    cmp("5 check 3", "CPU-s per update B, S half; B, S present", "0.038 0.042 0.108 0.118", f"{pu['B h256']:.3f} {pu['S h256']:.3f} {pu['B h512']:.3f} {pu['S h512']:.3f}")
    cmp("5 check 3", "CPU-s per scored row half; present", "2.2–2.5 7.2–7.3", f"{min(pr['B h256'], pr['S h256']):.1f}–{max(pr['B h256'], pr['S h256']):.1f} "
        f"{min(pr['B h512'], pr['S h512']):.1f}–{max(pr['B h512'], pr['S h512']):.1f}")
    tu = ts = 0.0
    for k in R:
        if k[3] == 0.01:
            continue
        d = R[k][1]["dose"]; key = f"{k[0]} h{k[1]}"
        tu += d * pu[key]; ts += (d // 256 + 1) * pr[key]
    cmp("5 check 3", "forecast for the 28 runs (CPU-h), recomputed from the per-update figures", "27.1", f"{(tu + ts) / 3600:.1f}")
    print(f"  the brief's condition on the forecast: 26 CPU-hours plus a quarter = {26 * 1.25} CPU-hours; check3.json records limit {q3['forecast']['limit_cpu_h']}")
    main28 = [k for k in R if k[3] != 0.01]; f7 = [k for k in R if k[3] == 0.01]
    cu = sum(R[k][1]["cpu_updates"] for k in main28) / 3600; cs = sum(R[k][1]["cpu_scoring"] for k in main28) / 3600
    fu_ = sum(R[k][1]["cpu_updates"] for k in f7) / 3600; fs = sum(R[k][1]["cpu_scoring"] for k in f7) / 3600
    cmp("5 cost", "the 28 runs: total (updates, scoring)", "27.88 (23.25, 4.63)", f"{cu + cs:.2f} ({cu:.2f}, {cs:.2f})")
    cmp("5 cost", "the 7 further runs: total (updates, scoring)", "3.73 (3.09, 0.63)", f"{fu_ + fs:.2f} ({fu_:.2f}, {fs:.2f})",
        f"{fu_ + fs:.2f}" == "3.73" and f"{fu_:.2f}" == "3.09" and abs(fs - 0.63) < 0.006)
    print(f"  further runs' scoring CPU-h unrounded: {fs:.4f}")
    cmp("5 cost", "partial runs killed by the restarts (CPU-h)", "1.45", f"{ctx['icpu'][0]:.2f} (receipts) / {ctx['icpu'][1]:.2f} (logs)", f"{ctx['icpu'][0]:.2f}" == "1.45")
    qc = sum(jshow(QUAL + f"check{n}.json")["cpu"] for n in ("1a", "2", "3", "4")) / 3600
    cmp("5 cost", "checks 1a, 2, 3 and 4 (CPU-h, parent processes)", "0.40", f"{qc:.2f}")
    print(f"  CPU of check 4's six run receipts, which include the child processes that check4.json does not count: {sum(v['cpu'] for v in ctx['c4'].values()) / 3600:.2f} h")
    ctx["cost"] = dict(cu=cu, cs=cs, fu=fu_, fs=fs, qc=qc, forecast=(tu + ts) / 3600)
    clock("section 5")


# ================================================================ 6
def section6(ctx):
    print("================ 6  weights")
    R = ctx["R"]
    order = sorted(R, key=lambda k: (k[1], k[0], k[2], k[3], k[4]))
    done, skipped = [], []
    normdiff = 0.0
    Wkeep = {}
    keep = {("S", 256, 25, 0.0003, False), ("B", 256, 25, 0.0003, False), ("S", 512, 25, 0.003, False)}
    for k in order:
        n, r = R[k]
        if free_bytes() < MIN_FREE + 10e6:
            skipped.append(k); continue
        blob = show(E52 + r["weights"])
        W = load_pt(blob)
        ok = sha(blob) == r["weights_sha256"] and tensors_sha(W) == r["tensors_sha256"]
        nm = {g: float(sum(float((W[t].astype(np.float64) ** 2).sum()) for t in GROUP_TENSORS[g]) ** .5) for g in GROUPS}
        dn = max(abs(nm[g] - r["curve"][-1]["norms"][g]) / r["curve"][-1]["norms"][g] for g in GROUPS)
        normdiff = max(normdiff, dn)
        done.append((k, ok, len(blob)))
        if k in keep:
            Wkeep[k] = W
    print(f"  weight files read: {len(done)} of {len(R)}; file sha256 and tensors_sha256 as recorded: {sum(ok for _, ok, _ in done)}; "
          f"by rule and width: {dict(collections.Counter((k[0], k[1]) for k, _, _ in done))}; total {sum(b for _, _, b in done) / 1e6:.1f} MB; skipped {skipped}")
    print(f"  the seven norms recomputed from each file equal the last row's norms of its receipt: largest relative difference {normdiff:.1e}")
    ctx["wdone"] = (len(done), sum(ok for _, ok, _ in done), normdiff, skipped)
    print(f"  disk free after reading the weight files: {free_bytes() / 1e9:.2f} GB")
    # E52's runs at 0.001 against E41's weight files, tensor by tensor
    eq41 = 0; n41 = 0
    for rule in "SB":
        for H in (256, 512):
            for fp in (25, 50):
                if free_bytes() < MIN_FREE + 10e6:
                    continue
                a = load_pt(show(E52 + R[(rule, H, fp, 0.001, False)][1]["weights"])); b = load_pt(show(f"{E41}e41_{rule}_h{H}_f{fp}_{SEED}.pt"))
                n41 += 1; eq41 += sorted(a) == sorted(b) and all(np.array_equal(a[t], b[t]) for t in a)
    print(f"  E52's final weights at 0.001 equal E41's committed weights of the same case, tensor by tensor: {eq41} of {n41}")
    ctx["eq41"] = (eq41, n41)
    # the forward with numpy
    fwd = {}
    for k, W in sorted(Wkeep.items()):
        A, Rs, TA, LP, dig, th = panels(k[2] / 100, SEED)
        r = R[k][1]; last = r["curve"][-1]
        net = Net(W)
        mine = dict(A={n: net.exact(it) for n, it in A.items()}, R={n: net.exact(it) for n, it in Rs.items()},
                    TA={n: net.exact(it) for n, it in TA.items()}, L={str(kk): net.exact(it) for kk, it in LP.items()})
        same = (mine["A"] == last["A_counts"], mine["R"] == last["R_counts"], mine["TA"] == last["taught_counts"], mine["L"] == last["length_counts"])
        diffs = {"A": {n: mine["A"][n] - last["A_counts"][n] for n in mine["A"]}, "R": {n: mine["R"][n] - last["R_counts"][n] for n in mine["R"]},
                 "taught": {n: mine["TA"][n] - last["taught_counts"][n] for n in mine["TA"]}, "length": {n: mine["L"][n] - last["length_counts"][n] for n in mine["L"]}}
        fwd[k] = (same, diffs, dig == r["reduction_sample_sha256"], LP[12] == A["add89"])
        heldA = all((a, b) in set(th[DEV[0]][1]) for p, _ in LP[40] for a, b in [(int(bytes(p[1:3])), int(bytes(p[-3:-1])))])
        print(f"  forward {k[0]} {WIDTH[k[1]]} f{k[2]} {k[3]:g}{' (resumed run)' if R[k][1].get('resumed') else ''}: final row counts A, R, taught, length equal to the receipt: {same}; "
              f"differences {diffs}; R sample digest recomputed = receipt: {fwd[k][2]}; length items at 12 = A's add items: {fwd[k][3]}; length items at 40 untaught: {heldA}")
    ctx["fwd"] = fwd
    clock("section 6")


# ================================================================ 7
def section7(ctx):
    print("================ 7  the runner as text")
    a = show(M + "exp52_rate.py").decode(); b = show(M + "exp52_resume.py").decode()
    pats = {
        "every group's lr set to the rate": (a, r"for g in opt\.param_groups:\n\s+g\[\"lr\"\] = rate"),
        "only group 0's decay changed, only with --product": (a, r"if product:\n\s+opt\.param_groups\[0\]\[\"weight_decay\"\] = wd"),
        "decay of the control computed by the brief's expression": (a, r"0\.25 \* 0\.001 / rate if product else C\.WD"),
        "make_opt of exp26_candidate first (asserts the standing groups)": (a, r"opt = C\.make_opt\(m\)"),
        "groups asserted at the start and at the end": (a, r"assert R_\.opt_groups\(m, opt\) == expected_groups.*\n[\s\S]*assert g_end == g_exp"),
        "A and R from E41's panels of never-taught pairs": (a, r"A, Rs, TA = S41\.panels\(f, seed\)"),
        "length test from the run's own held-out split": (a, r"E\.split\(LEN_TASK, f, seed\)\[1\]"),
        "length test asserted free of taught pairs": (a, r"assert len_in_taught == 0"),
        "every batch asserted taught": (a, r"S41\.assert_taught\(items, tk\)"),
        "diverged row stops the run": (a, r"if not row\[\"finite_losses\"\]:[\s\S]{0,250}break"),
        "resume: checkpoint holds model, optimizer state, receipt, CPU counters, wall": (b, r"dict\(update=u, model=m\.state_dict\(\), opt=opt\.state_dict\(\), rec=rec, cpu_up=cpu_up, cpu_sc=cpu_sc, wall=wall\)"),
        "resume: stream redrawn and digested from the start": (b, r"for _ in range\(start\):\n\s+items = next\(bt\); S41\.assert_taught\(items, tk\); dg\.add\(items\)"),
        "resume: digest asserted before the states are loaded": (b, r"assert dg\.value\(\) == rec\[\"curve\"\]\[-1\]\[\"data_digest\"\][^\n]*\n\s+m\.load_state_dict\(ck\[\"model\"\]\); opt\.load_state_dict\(ck\[\"opt\"\]\)"),
        "resume: loading with weights_only=False (pickle of the checkpoint)": (b, r"T\.load\(cp, weights_only=False\)"),
        "checkpoint every 4,096 updates by default": (b, r"CKPT = 4096"),
    }
    found = {k: bool(re.search(p, s)) for k, (s, p) in pats.items()}
    for k, v in found.items():
        print(f"  {k}: {v}")
    absent = {"a hash of the checkpoint file is recorded or checked": bool(re.search(r"sha256\(.*ckpt|ckpt.*sha", b)),
              "a random-number state is saved in the checkpoint": bool(re.search(r"get_rng_state|getstate\(\)", b))}
    for k, v in absent.items():
        print(f"  {k}: {v}")
    step = show(M + "exp26_candidate.py").decode()
    st = re.search(r"def step\(m, opt, items, local\):[\s\S]*?return float", step).group(0)
    print(f"  exp26_candidate.step uses no random numbers (no rand, dropout or generator in its text): {not re.search(r'rand|dropout|Generator', st)}")
    ctx["runner"] = (found, absent)
    clock("section 7")


# ================================================================ 8
def section8(ctx):
    print("================ 8  form")
    rep = ctx["report"]
    ph = {f"mark {i + 1}": rep.count(p) for i, p in enumerate(("@" * 2, "TO" + "DO", "T" + "BD", "X" * 2))}
    print(f"  placeholder marks in the report (a doubled at-sign, the to-do mark, the to-be-decided mark, a doubled capital X): {ph}")
    # names of language models: the words are taken from the repository's own branch names and file names, never written here
    refs = subprocess.run(["git", "for-each-ref", "--format=%(refname:short) %(objectname)", "refs/remotes/origin"], capture_output=True, text=True).stdout.splitlines()
    w1 = [l.split()[0].split("/")[1] for l in refs if l.split()[1] == WREF][0]
    w2 = [l.split()[0].split("/")[1] for l in refs if l.split()[0].endswith("minifly-consolidated-20261023")][0]
    w3 = re.search(r"brief ([A-Z]+)_TASK_E41", show(M + "exp41_scale.py").decode()).group(1)
    words = [w1, w2, w3]
    print(f"  words searched (taken from the worker's branch name, the default branch's name and the prefix of E41's brief file): {len(words)}, of lengths {[len(w) for w in words]}")
    places = collections.Counter(); files_with = collections.Counter()
    targets = [M + "E52_REPORT.md"] + [M + s for s in ("exp52_rate.py", "exp52_resume.py", "exp52_drive.py", "exp52_drive2.py", "exp52_tables.py", "exp46_decay.py")] \
        + [x for x in ctx["allf"] if not x.endswith(".pt")] + ctx["qf"]
    detail = collections.Counter()
    for t in targets:
        txt = show(t).decode(errors="replace")
        for i, w in enumerate(words):
            n = len(re.findall(w, txt, re.I))
            if n:
                kind = ("the report" if t.endswith("E52_REPORT.md") else "scripts" if t.endswith(".py") else "receipts and check files" if ".json" in t
                        else "logs" if "/logs/" in t or ".log" in t else "tables")
                places[(kind, i + 1)] += n; files_with[kind] += 1
                if t.endswith(".py") or t.endswith(".md"):
                    for m in re.finditer(w, txt, re.I):
                        ctxt = txt[max(0, m.start() - 30):m.end() + 30]
                        where = ("a branch name" if "origin/" in ctxt or "/" in txt[m.end():m.end() + 1] else "a commit trailer" if "Acting-Agent" in ctxt
                                 else "a brief's file name" if "_TASK_" in ctxt else "other")
                        detail[(t.split("/")[-1], i + 1, where)] += 1
    print(f"  occurrences by kind of file and word number: {dict(places)}; files: {dict(files_with)}")
    print(f"  in the report and the scripts, by place: {dict(detail)}")
    ctx["names"] = (dict(places), dict(detail))
    msgs = subprocess.run(["git", "log", "--format=%B%x00", f"{DREF}..{WREF}"], capture_output=True, text=True).stdout.split("\0")[:-1]
    print(f"  commit messages from the source commit to the report: {len(msgs)}; containing word 1: {sum(bool(re.search(w1, m_, re.I)) for m_ in msgs)}, "
          f"all of them in the line 'Acting-Agent: <word 1>:<session>': {all(re.search(r'Acting-Agent: ' + w1 + ':', m_, re.I) for m_ in msgs if re.search(w1, m_, re.I))}")
    # the brief's revision
    head = show(BRIEF, "HEAD"); r2 = show(BRIEF, BRIEF_REV2)
    print(f"  the brief on the checked-out branch equals revision 2 ({BRIEF_REV2}): {head == r2}; the report names a commit of the brief: "
          f"{bool(re.search(r'brief[^.]*[0-9a-f]{8}', rep[:800]))}")
    clock("section 8")


def summary():
    print("================ comparisons")
    by = collections.Counter(c[0] for c in COMP); bad = [c for c in COMP if not c[4]]
    for s in sorted(by):
        print(f"  {s}: {by[s]} compared, {sum(1 for c in COMP if c[0] == s and not c[4])} differ")
    print(f"  total: {len(COMP)} compared, {len(COMP) - len(bad)} agree, {len(bad)} differ")
    print("  every comparison outside the table cells (the table cells are counted above; each differing cell would be listed below):")
    for c in COMP:
        if not c[0].startswith(("2 report table", "2 tables", "2 report 4.")):
            print(f"    [{c[0]}] {c[1]}: reported {c[2]!r}, recomputed {c[3]!r}, {'agree' if c[4] else 'DIFFER'}")
    for c in bad:
        print(f"  DIFFERS [{c[0]}] {c[1]}: reported {c[2]!r}, recomputed {c[3]!r}")


def main():
    ctx = {}
    section1(ctx); section2(ctx); section3(ctx); section4(ctx); section5(ctx); section6(ctx); section7(ctx); section8(ctx)
    summary()
    print(f"running time: {time.time() - T0:.1f} s")


if __name__ == "__main__":
    main()
