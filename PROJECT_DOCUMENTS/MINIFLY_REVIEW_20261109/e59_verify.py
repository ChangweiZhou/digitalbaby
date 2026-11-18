"""Verification of the final report of E59 (the unrelated task learned from a fresh, a born and a lived network; the worker's
commit 4594ffc9). Read-only: nothing is trained, torch is not used, and no file of the worker's branch is executed. Every file of
the worker's branch is read with `git show` into memory and listed with `git ls-tree -r --name-only`; single files of the default
branch at the brief's base (7cabb13a) are read in the same way. The script writes no file.

Sections (the numbering of the verification document):
  1  inventory and receipts: the roster of 60 runs, completeness, settings, identities between runs, logs and the queue
  2  every figure of the report recomputed from the receipts and set beside the reported figure
  3  the readings in the brief's fixed words, the pairs, the source, the explanation under test, the predictions
  4  the five checks: what each receipt records, and the parts recomputed here (stream, panels, digests, pairs, check 3, check 4)
  5  the weights: starts (born, lived, born-shuffled), the 20 committed final files, the forward recomputed with numpy
  6  the runner as text (the lines on which the document relies)
  7  form: placeholders, the list of committed files, names of language models (held encoded), the dry runs' weights
  8  the newer head of the worker's branch (the 40 further weight files and any change of the report)

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e59_verify.py
"""
import codecs, collections, hashlib, io, json, math, os, pickle, re, subprocess, sys, time, zipfile
from random import Random
import numpy as np

T0 = time.time()
sys.dont_write_bytecode = True

WREF = "4594ffc927ff5aab92aa172349e5d5d5ea3f4562"     # the worker's commit of the report (verified)
NEWREF = "55d480c712df9aa64a2c6559575b74a2f94cd43b"                                 # the worker's head after the last fetch (section 8); set below
DREF = "7cabb13adc38753483d34dd5472fc3c84c4de675"     # the default branch at the brief's base
M = "MOD97_LOCAL_RULE_20261106/"
E59 = M + "exp45/e59/"
SEEDS = (73129, 93683, 116089)
STARTS = ("fresh", "born", "lived", "born-shuffled")
NS = (2048, 512, 256)
RULES = ("S", "B")
ROWS = [0] + list(range(16, 513, 16)) + list(range(544, 8193, 32))
L80 = 8128
EOS, BOS = 256, 257
DTYPE = {"FloatStorage": "<f4", "DoubleStorage": "<f8", "LongStorage": "<i8", "BoolStorage": "?"}
COMP = []                                              # (section, label, reported string, recomputed string)


def clock(label):
    print(f"[{label}: {time.time() - T0:.1f} s]", flush=True)


def cmp(sec, label, reported, recomputed):
    COMP.append((sec, label, str(reported), str(recomputed)))


# ---------------------------------------------------------------- reading through git
def show(path, ref=WREF):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def exists(path, ref):
    return subprocess.run(["git", "cat-file", "-e", f"{ref}:{path}"], capture_output=True).returncode == 0


def names(d, ref=WREF):
    return subprocess.run(["git", "ls-tree", "-r", "--name-only", ref, "--", d], capture_output=True, check=True, text=True).stdout.split()


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


sha = lambda b: hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- the task (e43_task.py and the runner, as read)
SYM = list(range(128, 192)); FILL = list(range(192, 224)); CODE, ASK = 224, 225


def split(seed):
    pairs = [(a, b) for a in SYM for b in SYM]
    Random(f"e43|{seed}|split").shuffle(pairs)
    return pairs[:2048], pairs[2048:]


def render(a, b, k, rng):
    return [CODE, a, b] + [rng.choice(FILL) for _ in range(k)] + [ASK], [a, b, EOS]


def stream(seed, n):
    taught = split(seed)[0][:n]; rng = Random(f"e43|{seed}|stream")
    while True:
        order = list(range(len(taught))); rng.shuffle(order)
        for i in order:
            yield render(*taught[i], rng.randint(0, 12), rng)


def panel(seed, which, k=12):
    taught, held = split(seed); rng = Random(f"e43|{seed}|panel|{which}")
    return [render(a, b, k, rng) for a, b in rng.sample(taught if which == "taught" else held, 300)]


def taught_panel(seed, n):
    taught = split(seed)[0][:n]; rng = Random(f"e43|{seed}|panel|taught")
    return [render(a, b, 12, rng) for a, b in rng.sample(taught, min(300, n))]


def batch_tensor(items):
    """exp45.batch_tensor, as read: rows padded with the start symbol; x, y and the answer mask."""
    rows = [p + t for p, t in items]
    Lr = max(len(r) for r in rows) + 1
    data = np.full((len(rows), Lr), BOS, np.int64)
    for i, r in enumerate(rows):
        data[i, 1:len(r) + 1] = r
    x, y = data[:, :-1], data[:, 1:]
    mask = np.zeros(y.shape, bool)
    for i, (p, t) in enumerate(items):
        mask[i, len(p):len(p) + len(t)] = True
    return x, y, mask


# ---------------------------------------------------------------- the network's forward (arch26.forward and exp11_cell.respond, as read)
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

    def respond(self, prompts, cap=6):
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
                        gen[q].append(int(yy[q]))
                        done[q] = done[q] or int(yy[q]) == EOS
                if all(done):
                    break
                z1n, gin = self.embed(yy[:, None])
                h, lg = self.step(h, z1n[:, 0], gin[:, 0])
            for q, i in enumerate(ids):
                out[i] = gen[q]
        return out

    def counts(self, items):
        outs = self.respond([p for p, _ in items])
        b = lambda j: sum(len(o) > j and o[j] == t[j] for o, (_, t) in zip(outs, items))
        form = sum(len(o) == 3 and 128 <= o[0] <= 191 and 128 <= o[1] <= 191 and o[2] == EOS for o in outs)
        return dict(exact=sum(o == t for o, (_, t) in zip(outs, items)), first=b(0), second=b(1), third=b(2), form=form)


# the earlier families' panels (exp16_integrated.panels with E16_ARCH=bank2, as copied in the reviewer's npnet.py from exp11_cell.py)
OPS = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
LIFE = [("add", 97, b"r"), ("sub", 97, b"t")]
F3 = [("add", 73, b"s"), ("sub", 73, b"u")]
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def fam_render(task, a, b, k, rng):
    op, m, code = task
    val = {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]
    d = [rng.choice(DISTRACT) for _ in range(k)]
    return list(code) + list(f"{a:02d}".encode()) + [ord(OPS[op])] + d + list(f"{b:02d}".encode()) + [ord("=")], list(f"{val:02d}".encode()) + [EOS]


def fam_split(task, frac, seed):
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]).shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def fam_panel(task, pairs, seed, ks, n):
    rng = Random(seed + 3); sub = rng.sample(pairs, n)
    return {k: [fam_render(task, a, b, k, rng) for a, b in sub] for k in ks}


def earlier_panels(seed):
    P = {f"{t[0]}{t[1]}": fam_panel(t, fam_split(t, .25, seed)[1], seed, (0, 8, 12, 20, 40), 400)[12] for t in LIFE + F3}
    P.update({f"{t[0]}{t[1]}": fam_panel(t, fam_split(t, .5, seed)[1], seed, (12,), 300)[12] for t in DEV})
    return P


# ---------------------------------------------------------------- the readings, written here from the brief's words
def pct(c):
    return 100.0 * c["exact"] / c["n"]


def to80(rows, key):
    for i, r in enumerate(rows):
        u = r["updates"]
        if u > L80:
            return None
        span = [x for x in rows if u <= x["updates"] <= u + 64]
        if all(x[key]["exact"] >= 0.8 * x[key]["n"] for x in span):
            return u
    return None


def mean_over(rows, key, a, b, part=None):
    v = [(100.0 * (x[key][part]["exact"] / x[key][part]["n"]) if part else pct(x[key])) for x in rows if a <= x["updates"] <= b]
    return sum(v) / len(v), len(v)


def summarize(r):
    rows = r["rows"]
    late, n1 = mean_over(rows, "held12", 7296, 8192); early, n2 = mean_over(rows, "held12", 5248, 6144)
    out = dict(to80=to80(rows, "held12"), to80_0=to80(rows, "held0"),
               lv={k: mean_over(rows, k, 7936, 8192)[0] for k in ("held12", "held0", "taught")}, nlv=mean_over(rows, "held12", 7936, 8192)[1],
               late=late, early=early, n_late=n1, n_early=n2, flat=(late - early) < 3,
               third=next((x["updates"] for x in rows if x["held12"]["third_is_end"] >= 240), None),
               taught_n=rows[-1]["taught"]["n"], cpu=r["complete"]["cpu_total"])
    if r["n_taught"] == 256:
        out["parts"] = {p: (rows[-1]["held12"][p]["n"], mean_over(rows, "held12", 7936, 8192, p)[0]) for p in ("unseen", "seen")}
    return out


def word(a, b):
    if a is not None and b is not None:
        if a / b < 2 / 3 and b - a >= 48:
            return "fewer"
        if a / b > 1.5 and a - b >= 48:
            return "more"
        return "about the same"
    if a is not None:
        return "fewer" if a / L80 <= 2 / 3 else "not decided"
    if b is not None:
        return "more" if L80 / b >= 1.5 else "not decided"
    return "neither reaches"


def combine(ws):
    c = collections.Counter(ws)
    if c["fewer"] >= 2 and "more" not in ws:
        return "fewer"
    if c["more"] >= 2 and "fewer" not in ws:
        return "more"
    if c["about the same"] >= 2:
        return "about the same"
    return "not decided"


fu = lambda v: "–" if v is None else f"{v:,}"


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


def find_table(tables, header_start, k=0):
    hits = [t for t in tables if " | ".join(t[0]).startswith(header_start)]
    return hits[k]


# ================================================================ 1
def section1():
    print("================ 1  inventory and receipts")
    allf = names(E59)
    print(f"  files under exp45/e59 at {WREF[:8]}: {len(allf)}")
    kinds = collections.Counter(("receipt" if re.fullmatch(r"e59_[a-z-]+_n\d+_[SB]_\d+\.json", p[len(E59):]) else
                                 "final weights" if re.fullmatch(r"e59_[a-z-]+_n\d+_[SB]_\d+_final\.pt", p[len(E59):]) else p[len(E59):].split("/")[0] if "/" in p[len(E59):] else p[len(E59):])
                                for p in allf)
    print("  by kind:", dict(kinds))
    roster = [(s, n, r, sd) for s in STARTS for n in (NS if s != "born-shuffled" else (2048,)) for r in RULES for sd in SEEDS]
    want = {f"{E59}e59_{s}_n{n}_{r}_{sd}.json" for s, n, r, sd in roster}
    have = {p for p in allf if re.fullmatch(re.escape(E59) + r"e59_[a-z-]+_n\d+_[SB]_\d+\.json", p)}
    print(f"  roster of the brief: {len(roster)} runs; receipts present {len(want & have)}; missing {sorted(want - have)}; not in the roster {sorted(have - want)}")
    wfiles = {p for p in allf if p.endswith("_final.pt") and "/dry/" not in p}
    wantw = {f"{E59}e59_{s}_n{n}_{r}_73129_final.pt" for s, n, r, sd in roster if sd == 73129}
    print(f"  final weights of seed 73129 committed: {len(wfiles & wantw)} of {len(wantw)}; other final weight files beside the receipts: {sorted(wfiles - wantw)}")
    print(f"  interrupted/ present: {any('/interrupted/' in p for p in allf)}; STOP file present: {any(p.endswith('/STOP') for p in allf)}; "
          f"rows_uncommitted/ outside dry/: {[p for p in allf if '/rows_uncommitted/' in p and '/dry/' not in p]}")
    R = {}
    for k in roster:
        R[k] = json.loads(show(f"{E59}e59_{k[0]}_n{k[1]}_{k[2]}_{k[3]}.json"))
    # completeness and settings
    bad = collections.defaultdict(list)
    runner_sha = sha(show(M + "exp59_unrelated_starts.py"))
    own = ("exp59_unrelated_starts.py", "exp43_unrelated.py", "e43_task.py", "e40_new_family.py", "exp38_erase.py", "arch31_workerB.py")
    own_sha = {f: sha(show(M + f)) for f in own}
    own_sha_d = {f: sha(show(M + f, DREF)) for f in own if f != "exp59_unrelated_starts.py"}
    print(f"  the runner at {WREF[:8]}: sha256 {runner_sha}")
    print("  the files the runner takes from, sha256 at the worker's commit equal to the default branch's at 7cabb13a:",
          {f: own_sha[f] == own_sha_d[f] for f in own_sha_d})
    srcs0 = R[roster[0]]["provenance"]["sources_sha256"]
    other = sorted(set(srcs0) - set(own))
    other_eq = {f: sha(show(M + f)) == srcs0[f] and sha(show(M + f, DREF)) == srcs0[f] for f in other}
    print(f"  the other sources recorded ({len(other)}): sha256 in the receipts equal to the files at the worker's commit and at 7cabb13a: {other_eq}")
    groups0 = R[roster[0]]["settings"]["optimizer_groups_read_back"]
    rest = ["l1.weight", "l1.bias", "rnn.weight_ih_l0", "rnn.weight_hh_l0", "rnn.bias_ih_l0", "rnn.bias_hh_l0", "lm.weight", "lm.bias", "l2.weight", "l2.bias", "l3.weight", "l3.bias"]
    exp_groups = [dict(weight_decay=0.25, lr=0.001, betas=[0.9, 0.98], params=rest), dict(weight_decay=0.0, lr=0.001, betas=[0.9, 0.98], params=["book.weight"])]
    print(f"  optimizer groups of the first receipt equal the brief's (AdamW, rate 0.001, decay 0.25 on every tensor but the byte table; betas as read): {groups0 == exp_groups}")
    commits = collections.Counter()
    for k, r in R.items():
        s, n, rule, sd = k
        rows = r["rows"]
        chk = {
            "header": (r["experiment"], r["kind"], r["start"], r["n_taught"], r["rule"], r["seed"], r["updates"], r["local"]) == ("E59", "run", s, n, rule, sd, 8192, rule == "S"),
            "rows": [x["updates"] for x in rows] == ROWS and r["settings"]["rows"] == ROWS,
            "complete": bool(r.get("complete")) and r["complete"]["updates"] == 8192 and r["complete"]["optimizer_steps"] == [8192] and "stopped" not in r,
            "finite": r["complete"]["all_losses_finite"] and all(x["finite_losses"] for x in rows) and rows[0]["loss"] is None
                      and all(isinstance(x["loss"], float) and math.isfinite(x["loss"]) for x in rows[1:]),
            "last row = complete": rows[-1]["tensors_sha256"] == r["complete"]["tensors_sha256"] and rows[-1]["data_digest"] == r["complete"]["data_digest"]
                                   and r["complete"]["data_digest"]["batches"] == 8192,
            "row 0 = start": rows[0]["tensors_sha256"] == r["settings"]["start_record"]["tensors_sha256"],
            "groups": r["settings"]["optimizer_groups_read_back"] == exp_groups and r["complete"]["optimizer_groups_read_back"] == exp_groups,
            "sources": all(r["provenance"]["sources_sha256"].get(f) == own_sha[f] for f in own) and r["provenance"]["sources_sha256"] == srcs0,
            "unmodified": r["provenance"]["git"]["sources_modified_vs_commit"] == [],
            "versions": r["provenance"]["versions"] == dict(python="3.13.18", torch="2.15.0+cpu", numpy="2.5.5", torch_threads=1, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
            "batch": r["settings"]["batch"] == 64,
            "panels n": all(x["held12"]["n"] == 300 and x["held0"]["n"] == 300 and x["taught"]["n"] == min(300, n) for x in rows),
            "unseen items": r["settings"]["panels"]["held12_unseen_items"] == ({73129: 11, 93683: 6, 116089: 2}[sd] if n == 256 else 0),
            "earlier panels at rows 0 and 8192 only": [x["updates"] for x in rows if "earlier_panels_k12" in x] == [0, 8192],
            "earlier items": r["settings"]["earlier_training_items"] == {"fresh": 0, "born": 1572864, "lived": 3932160, "born-shuffled": 1572864}[s],
            "weights file": r["complete"]["weights"] == (f"e59_{s}_n{n}_{rule}_{sd}_final.pt" if sd == 73129 else f"rows_uncommitted/e59_{s}_n{n}_{rule}_{sd}_final.pt"),
            "check4 field": ("check4" in r) == (s == "born" and n == 2048),
        }
        commits[r["provenance"]["git"]["commit"]] += 1
        for c, ok in chk.items():
            if not ok:
                bad[c].append(k)
    print(f"  receipts checked: {len(R)}; per test, receipts that fail it: " + "; ".join(f"{c}: {len(bad[c])}" for c in
          ("header", "rows", "complete", "finite", "last row = complete", "row 0 = start", "groups", "sources", "unmodified", "versions", "batch", "panels n",
           "unseen items", "earlier panels at rows 0 and 8192 only", "earlier items", "weights file", "check4 field")))
    for c, v in bad.items():
        if v:
            print(f"    failing {c}: {v}")
    print(f"  commits recorded in the receipts: {len(commits)} different commits; the runner's sha256 is the same in all 60 (test 'sources' above)")
    # start records
    E40 = {(r, sd): json.loads(show(f"{M}exp45/e40/e40_fresh_{r}_{sd}.json", DREF))["start"]["tensors_sha256"] for r in RULES for sd in SEEDS}
    E43 = {(r, sd): json.loads(show(f"{M}exp45/e43/e43_unrelated-decay_{r}_{sd}.json", DREF)) for r in RULES for sd in SEEDS}
    L26 = {(r, sd): json.loads(show(f"{M}exp45/e26/e26_life_{r}_{sd}.json", DREF))["complete"] for r in RULES for sd in SEEDS}
    D26 = {(r, sd): json.loads(show(f"{M}exp45/e26/e26_dev_{r}_{sd}.json", DREF)) for r in RULES for sd in SEEDS}
    st_ok = collections.Counter(); st_bad = []
    for (s, n, rule, sd), r in R.items():
        sr = r["settings"]["start_record"]
        if s == "fresh":
            ok = sr["tensors_sha256"] == E40[(rule, sd)] and sr["equals_e40"] and sr["tensors_sha256"] == E40[("S", sd)]
        elif s == "lived":
            ok = (sr["start"] == f"exp45/e26/e26_life_{rule}_{sd}.pt" and sr["weights_sha256"] == L26[(rule, sd)]["weights_sha256"]
                  and sr["tensors_sha256"] == L26[(rule, sd)]["tensors_sha256"] and sr["tensors_sha_matches_receipt"])
        else:
            ok = (sr["start"] == f"exp45/e26/e26_dev_{rule}_{sd}.pt" and sr["weights_sha256"] == D26[(rule, sd)]["weights_sha256"]
                  and sr["e43_birth_tensors_sha256"] == E43[(rule, sd)]["settings"]["birth_tensors_sha256"] and sr["equals_e43_birth"]
                  and (sr["tensors_sha256"] == sr["e43_birth_tensors_sha256"] if s == "born" else
                       sr["born_tensors_sha256"] == sr["e43_birth_tensors_sha256"] and sr["sorted_values_equal_born"] and sr["byte_table_unchanged"] and sr["tensors_reordered"] == 12))
        st_ok[s] += ok
        if not ok:
            st_bad.append((s, n, rule, sd))
    print(f"  start records against E40's fresh receipts, E26's birth and life receipts and E43's birth hash (7cabb13a): agree {dict(st_ok)}; disagree {st_bad}")
    same_start = all(len({R[(s, n, rule, sd)]["rows"][0]["tensors_sha256"] for n in (NS if s != "born-shuffled" else (2048,))}) == 1 for s in STARTS for rule in RULES for sd in SEEDS)
    fresh_rules = all(R[("fresh", 2048, "S", sd)]["rows"][0]["tensors_sha256"] == R[("fresh", 2048, "B", sd)]["rows"][0]["tensors_sha256"] for sd in SEEDS)
    print(f"  row 0 weights equal across the numbers of pairs for each start, rule and seed: {same_start}; fresh equal under the two rules: {fresh_rules}")
    # identities between runs: the same batches (check 5), the earlier panels at row 0
    c5 = []
    for sd in SEEDS:
        for n in NS:
            ks = [k for k in R if k[1] == n and k[3] == sd]
            ref = [x["data_digest"] for x in R[ks[0]]["rows"]]
            c5.append((sd, n, len(ks), all([x["data_digest"] for x in R[k]["rows"]] == ref for k in ks)))
    print("  data digest equal at every row across all runs of a seed and a number of pairs (both rules, all starts): " + "; ".join(f"{sd} n{n}: {m} runs {ok}" for sd, n, m, ok in c5))
    e0 = all(len({json.dumps(R[(s, n, rule, sd)]["rows"][0]["earlier_panels_k12"], sort_keys=True) for n in (NS if s != "born-shuffled" else (2048,))}) == 1 for s in STARTS for rule in RULES for sd in SEEDS)
    print(f"  earlier panels at row 0 equal across the numbers of pairs: {e0}")
    # logs and the queue
    logs = [p for p in allf if "/logs/" in p]
    lg_bad = []
    for s, n, rule, sd in roster:
        t = show(f"{E59}logs/e59_{s}_n{n}_{rule}_{sd}.log").decode()
        rows_l = re.findall(r" row (\d+): ", t)
        if not (rows_l == [str(u) for u in ROWS] and t.count("e59 complete") == 1 and not re.search(r"Traceback|Error|STOP|Killed|interrupt", t, re.I)):
            lg_bad.append((s, n, rule, sd))
    q = show(f"{E59}logs/queue.log").decode().splitlines()
    starts_ = [l for l in q if " start " in l and " pid " in l]; ends = [l for l in q if " end " in l and "exit=" in l]
    exits = collections.Counter(re.search(r"exit=(-?\d+)", l).group(1) for l in ends)
    live, peak = set(), 0
    for l in q:
        m_ = re.match(r"\S+ (start|end) (\S+) n(\d+) ([SB]) (\d+)", l)
        if m_:
            key = m_.groups()[1:]
            (live.add if m_.group(1) == "start" else live.discard)(key); peak = max(peak, len(live))
    spans = {}
    for sd in SEEDS:
        a = next(l for l in q if f"queue start seed {sd}" in l)[:20]; b = next(l for l in q if f"queue end seed {sd}" in l)[:20]
        spans[sd] = (a, b)
    outs = {p.rsplit("/", 1)[1]: len(show(p)) for p in logs if p.endswith(".out")}
    print(f"  logs: {len(logs)} files; run logs whose rows are not exactly the 273 rows once, or that lack one completion line, or that mention an error, a stop or an interruption: {lg_bad}")
    print(f"  queue.log: {len(starts_)} starts, {len(ends)} ends, exit codes {dict(exits)}; most processes at once {peak}; queue spans {spans}; sizes of the .out files {outs}")
    for sd in SEEDS:
        ql = [l for l in q if l.endswith(f" {sd}") or f" {sd} " in l]
        st = [l for l in ql if " start " in l and " pid " in l]
        en = [l for l in ql if " end " in l and "exit=" in l]
        print(f"    seed {sd}: first three starts {[l[20:].split(' pid')[0] for l in st[:3]]} at {st[0][:20]}; fourth start at {st[3][:20]}; first end at {en[0][:20]}")
    return R, allf, E43, D26, L26


# ================================================================ 2
def section2(R, report):
    print("\n================ 2  every figure: recomputed beside reported")
    S = {k: summarize(r) for k, r in R.items()}
    assert all(v["n_late"] == v["n_early"] == 29 and v["nlv"] == 9 for v in S.values())
    tabs = md_tables(report)
    # every run
    hdr, rows = find_table(tabs, "Start | n | Rule | Seed | to 80 (12)")
    for c in rows:
        k = (c[0], int(c[1]), c[2], int(c[3])); v = S[k]
        mine = [fu(v["to80"]), fu(v["to80_0"]), f"{v['lv']['held12']:.1f}", f"{v['lv']['held0']:.1f}", f"{v['lv']['taught']:.1f} ({v['taught_n']})",
                f"{v['flat']} ({v['late'] - v['early']:+.1f})", fu(v["third"]), f"{v['cpu']:.0f}"]
        for j, lab in enumerate(hdr[4:]):
            cmp("2 every run", f"{k} {lab}", c[4 + j], mine[j])
    print(f"  table 'Every run': {len(rows)} rows")
    hdr, rows = find_table(tabs, "Start | Rule | Seed | unseen-place")
    for c in rows:
        k = (c[0], 256, c[1], int(c[2])); p = S[k]["parts"]
        cmp("2 256 in two parts", f"{k} unseen", c[3], f"({p['unseen'][0]}) {p['unseen'][1]:.1f}")
        cmp("2 256 in two parts", f"{k} other", c[4], f"({p['seen'][0]}) {p['seen'][1]:.1f}")
    print(f"  table '256 pairs in two parts': {len(rows)} rows")
    W = {}
    hdr, rows = find_table(tabs, "Rule | n | Comparison")
    for c in rows:
        rule, n = c[0], int(c[1]); a, b = c[2].split(" against ")
        per = []
        for j, sd in enumerate(SEEDS):
            x, y = S[(a, n, rule, sd)]["to80"], S[(b, n, rule, sd)]["to80"]; w = word(x, y); per.append(w)
            cmp("2 presentations", f"{rule} {n} {a} against {b} {sd}", c[3 + j], f"{fu(x)} vs {fu(y)}: {w}")
        W[(rule, n, a, b)] = combine(per)
        cmp("2 presentations", f"{rule} {n} {a} against {b} word", c[6], f"**{combine(per)}**")
    print(f"  table 'Presentations': {len(rows)} rows")
    hdr, rows = find_table(tabs, "Start | Rule | Seed | to 80 at 12 | to 80 at 0 | third")
    for c in rows:
        k = (c[0], 2048, c[1], int(c[2])); v = S[k]
        for lab, rp, mv in (("to 80 at 12", c[3], fu(v["to80"])), ("to 80 at 0", c[4], fu(v["to80_0"])), ("third byte end", c[5], fu(v["third"]))):
            cmp("2 source table", f"{k} {lab}", rp, mv)
    hdr, rows = [t for t in tabs if t[0] == ["Start", "Rule", "Seed", "to 80 at 12", "to 80 at 0"]][0]
    for c in rows:
        k = (c[0], 256, c[1], int(c[2])); v = S[k]
        cmp("2 256 table", f"{k} to 80 at 12", c[3], fu(v["to80"])); cmp("2 256 table", f"{k} to 80 at 0", c[4], fu(v["to80_0"]))
    two = [t for t in tabs if t[0][:2] == ["Start", "n"] and len(t[0]) == 6]
    for (hdr, rows), label in zip(two, ("two rules", "B2")):
        for c in rows:
            st, n = c[0], int(c[1]); per = []
            for j, sd in enumerate(SEEDS):
                x = S[(st, n, "S", sd)]["to80"]; y = S[("fresh" if label == "B2" else st, n, "B", sd)]["to80"]; w = word(x, y); per.append(w)
                cmp(f"2 {label}", f"{st} {n} {sd}", c[2 + j], f"{fu(x)} vs {fu(y)}: {w}")
            W[(label, st, n)] = combine(per)
            cmp(f"2 {label}", f"{st} {n} word", c[5], f"**{combine(per)}**")
        print(f"  table '{label}': {len(rows)} rows")
    hdr, rows = find_table(tabs, "Start | n | Rule | Seed | add89")
    EARL = ("add89", "sub89", "max89", "min89", "add97", "sub97", "add73", "sub73")
    for c in rows:
        k = (c[0], int(c[1]), c[2], int(c[3])); r = R[k]["rows"]
        for j, e in enumerate(EARL):
            cmp("2 earlier panels", f"{k} {e}", c[4 + j], f"{r[0]['earlier_panels_k12'][e]['exact']} → {r[-1]['earlier_panels_k12'][e]['exact']}")
    nE = {e: R[("born", 2048, "S", 73129)]["rows"][0]["earlier_panels_k12"][e]["n"] for e in EARL}
    print(f"  table 'earlier panels': {len(rows)} rows; panel sizes {nE}")
    allzero = all(R[k]["rows"][-1]["earlier_panels_k12"][e]["exact"] == 0 for k in R for e in EARL)
    print(f"  every run at 0 on every earlier panel at its last row: {allzero}")
    th = {st: sorted({S[k]["third"] for k in S if k[0] == st}) for st in STARTS}
    print(f"  first row at which the third byte is the end symbol on 240 held items, values taken, by start: {th}")
    # costs and sentences
    cpu = [v["cpu"] for v in S.values()]
    cmp("2 sentences", "CPU-hours of the 60 runs", "18.53", f"{sum(cpu) / 3600:.2f}")
    cmp("2 sentences", "least CPU-seconds of a run", "1,031", f"{min(cpu):,.0f}")
    cmp("2 sentences", "most CPU-seconds of a run", "1,242", f"{max(cpu):,.0f}")
    upd = [R[k]["complete"]["cpu_updates"] / 8192 for k in R]; scr = [R[k]["complete"]["cpu_scoring"] / 273 for k in R]
    print(f"  CPU-seconds an update {min(upd):.4f} to {max(upd):.4f}; scoring a row (with the two special rows) {min(scr):.2f} to {max(scr):.2f}; "
          f"CPU-hours a run {min(cpu) / 3600:.3f} to {max(cpu) / 3600:.3f} (the brief's forecast: about 0.35 a run, 21 for the 60)")
    b2048 = {r: [S[("born", 2048, r, sd)]["to80"] for sd in SEEDS] for r in RULES}
    cmp("2 sentences", "born, 2,048, B: range of updates to 80", "208 to 240", f"{min(b2048['B'])} to {max(b2048['B'])}")
    cmp("2 sentences", "born, 2,048, S: range of updates to 80", "320 to 400", f"{min(b2048['S'])} to {max(b2048['S'])}")
    third64 = [R[("born", 2048, r, 73129)]["rows"][4]["held12"]["third_is_end"] for r in RULES]
    cmp("2 sentences", "born 2,048 seed 73129, third byte end at row 64 (S, B)", "300 and 282", f"{third64[0]} and {third64[1]}")
    nreach = {(st, r): sum(S[(st, 256, r, sd)]["to80"] is not None for sd in SEEDS) for st in ("fresh", "born", "lived") for r in RULES}
    print(f"  seeds reaching 80 with 256 pairs: {nreach}")
    for (st, r), v in nreach.items():
        cmp("2 sentences", f"{st}, 256, {r}: seeds reaching 80", {("lived", "S"): "1"}.get((st, r), "3"), str(v))
    fresh_all = all(S[("fresh", n, r, sd)]["to80"] is not None for n in NS for r in RULES for sd in SEEDS)
    cmp("2 sentences", "fresh reaches 80 in every run", "True", str(fresh_all))
    flat_lived_S256 = [S[("lived", 256, "S", sd)]["flat"] for sd in SEEDS]
    cmp("2 sentences", "lived, S, 256: flattened on no seed", "[False, False, False]", str(flat_lived_S256))
    return S, W


# ================================================================ 3
def section3(S, W, R):
    print("\n================ 3  the readings in the brief's words")
    P = lambda rule, n, a, b: W[(rule, n, a, b)]
    for rule in RULES:
        print(f"  {rule}: " + "; ".join(f"n {n}: born/fresh {P(rule, n, 'born', 'fresh')}, lived/fresh {P(rule, n, 'lived', 'fresh')}, lived/born {P(rule, n, 'lived', 'born')}" for n in NS)
              + f"; 2,048: born/born-shuffled {P(rule, 2048, 'born', 'born-shuffled')}, born-shuffled/fresh {P(rule, 2048, 'born-shuffled', 'fresh')}")
    noorder = [(r, n) for r in RULES for n in NS if P(r, n, "lived", "born") == "fewer" and P(r, n, "born", "fresh") == "fewer" and P(r, n, "lived", "fresh") != "fewer"]
    cmp("3 readings", "cases without an order of the three starts", "none", "none" if not noorder else str(noorder))
    # ratios and scatter between seeds
    print("  ratios of updates to 80 (first start over second), per seed, where both reach 80:")
    for rule in RULES:
        for n in NS:
            for a, b in (("born", "fresh"), ("lived", "fresh"), ("lived", "born")) + ((("born", "born-shuffled"), ("born-shuffled", "fresh")) if n == 2048 else ()):
                rs = [S[(a, n, rule, sd)]["to80"] / S[(b, n, rule, sd)]["to80"] if S[(a, n, rule, sd)]["to80"] and S[(b, n, rule, sd)]["to80"] else None for sd in SEEDS]
                print(f"    {rule} {n:4d} {a:13s} / {b:13s}: " + ", ".join("–" if x is None else f"{x:.2f}" for x in rs))
    for rule in RULES:
        for n in NS:
            for st in ("fresh", "born", "lived"):
                v = [S[(st, n, rule, sd)]["to80"] for sd in SEEDS]
                if all(v):
                    print(f"    spread between seeds, {rule} {n:4d} {st:6s}: {min(v):,} to {max(v):,} (largest over smallest {max(v) / min(v):.2f})")
    # pairs
    def saves(t, f):
        c = dict(i=t["to80"] is not None and t["lv"]["held12"] >= 80, ii=f["to80"] is None, iii=f["flat"], iv=f["lv"]["taught"] >= 95, v=f["lv"]["held12"] <= t["lv"]["held12"] - 10)
        return c, all(c.values())
    lic = {}; ii_fail = 0; cases = 0
    for st in ("born", "lived"):
        for rule in RULES:
            for n in NS:
                per = {sd: saves(S[(st, n, rule, sd)], S[("fresh", n, rule, sd)]) for sd in SEEDS}
                yes = [sd for sd in SEEDS if per[sd][1]]
                lic[(st, rule, n)] = len(yes) >= 2 and all(S[("fresh", n, rule, sd)]["lv"]["held12"] <= S[(st, n, rule, sd)]["lv"]["held12"] for sd in SEEDS if sd not in yes)
                ii_fail += sum(not per[sd][0]["ii"] for sd in SEEDS); cases += 3
                i_ii = [sd for sd in SEEDS if per[sd][0]["i"] and per[sd][0]["ii"]]
                if i_ii:
                    print(f"    (i) and (ii) hold for {st} {rule} {n} on {i_ii}")
    cmp("3 pairs", "sentence on pairs licensed (any start, rule, n)", "False", str(any(lic.values())))
    cmp("3 pairs", "cases (start, rule, n, seed) in which (ii) fails", f"{cases} of {cases}", f"{ii_fail} of {cases}")
    rev = {}
    for st in ("born", "lived"):
        for rule in RULES:
            for n in NS:
                per = {sd: saves(S[("fresh", n, rule, sd)], S[(st, n, rule, sd)]) for sd in SEEDS}
                yes = [sd for sd in SEEDS if per[sd][1]]
                rev[(st, rule, n)] = len(yes) >= 2
                i_ii = [sd for sd in SEEDS if per[sd][0]["i"] and per[sd][0]["ii"]]
                if i_ii:
                    print(f"    reverse case: fresh in the place of {st}, {rule}, {n}: (i) and (ii) hold on {i_ii}; conditions per seed "
                          + "; ".join(f"{sd}: " + "".join(k for k, ok in per[sd][0].items() if ok) for sd in SEEDS))
    cmp("3 pairs", "reverse sentence licensed (any)", "False", str(any(rev.values())))
    every256 = {r: all(sum(S[(st, 256, r, sd)]["to80"] is not None for sd in SEEDS) >= 2 for st in ("fresh", "born", "lived")) for r in RULES}
    cmp("3 pairs", "B: every start reaches 80 from 256 pairs on at least two seeds", "True", str(every256["B"]))
    cmp("3 pairs", "S: every start reaches 80 from 256 pairs on at least two seeds", "False", str(every256["S"]))
    # the source
    for rule in RULES:
        bf, bbs, bsf = P(rule, 2048, "born", "fresh"), P(rule, 2048, "born", "born-shuffled"), P(rule, 2048, "born-shuffled", "fresh")
        src = ("what the first family taught" if bf == "fewer" and bbs == "fewer" else "the sizes of the weights" if bf == "fewer" and bsf == "fewer" and bbs == "about the same"
               else "no source named")
        cmp("3 source", f"{rule}: born/fresh, born/born-shuffled, born-shuffled/fresh at 2,048; source",
            {"S": "more, about the same, about the same; no source named", "B": "about the same, about the same, about the same; no source named"}[rule], f"{bf}, {bbs}, {bsf}; {src}")
    # fillers at 256 pairs under back-propagation (given here, no reading of the brief applies)
    for st in ("born", "lived"):
        d12 = [S[("fresh", 256, "B", sd)]["to80"] - S[(st, 256, "B", sd)]["to80"] for sd in SEEDS]
        d0 = [S[("fresh", 256, "B", sd)]["to80_0"] - S[(st, 256, "B", sd)]["to80_0"] for sd in SEEDS]
        print(f"  256 pairs, B: updates to 80 of fresh less those of {st}: at 12 fillers {d12}; at 0 fillers {d0}")
    # the explanation under test
    ex = {}
    for rule in RULES:
        anyf = any(P(rule, n, a, "fresh") == "fewer" for a in ("born", "lived") for n in NS)
        at = [P(rule, 2048, a, "fresh") for a in ("born", "lived")]
        ex[rule] = dict(any_fewer=anyf, at2048=at, refute=(not anyf) and all(w in ("about the same", "more") for w in at),
                        untested=any(w in ("not decided", "neither reaches") for w in at), support2=P(rule, 2048, "born", "fresh") == "fewer" and P(rule, 2048, "born", "born-shuffled") == "fewer")
        print(f"  explanation, {rule}: {ex[rule]}")
    cmp("3 explanation", "supported", "False", str(all(ex[r]["support2"] for r in RULES)))
    cmp("3 explanation", "refuted", "False", str(all(ex[r]["refute"] for r in RULES)))
    cmp("3 explanation", "refutation's conditions under S; under B", "True; False", f"{ex['S']['refute']}; {ex['B']['refute']}")
    cmp("3 explanation", "untested under a rule", "False", str(any(ex[r]["untested"] for r in RULES)))
    # predictions
    fresh256 = {r: sum(S[("fresh", 256, r, sd)]["to80"] is not None for sd in SEEDS) for r in RULES}
    p6 = [word(S[("fresh", 2048, "S", sd)]["to80"], S[("fresh", 2048, "B", sd)]["to80"]) for sd in SEEDS]
    pred = {
        "P59.2": all(P(r, 2048, "born", "fresh") == "fewer" for r in RULES),
        "P59.3": all(P(r, 2048, "lived", "born") == "about the same" for r in RULES),
        "P59.4": all(fresh256[r] <= 1 for r in RULES),
        "P59.5": any(lic.values()),
        "P59.6": p6.count("more") >= 2,
        "P59.7": all(P(r, 2048, "born", "born-shuffled") == "fewer" for r in RULES),
        "P59.8": all(P(r, 2048, "born-shuffled", "fresh") == "about the same" for r in RULES),
        "P59.9": W[("B2", "lived", 2048)] == "fewer",
    }
    rep = {"P59.2": False, "P59.3": False, "P59.4": False, "P59.5": False, "P59.6": False, "P59.7": False, "P59.8": True, "P59.9": False}
    for k in pred:
        cmp("3 predictions", k, rep[k], pred[k])
    print(f"  P59.4: fresh reaches 80 with 256 pairs on {fresh256} seeds; P59.6: the seeds' words {p6}; P59.9: {W[('B2', 'lived', 2048)]}")
    return ex


# ================================================================ 4
def section4(R, E43):
    print("\n================ 4  the checks")
    c1 = json.loads(show(E59 + "checks/e59_check1.json")); c2 = json.loads(show(E59 + "checks/e59_check2.json"))
    c3 = json.loads(show(E59 + "checks/e59_check3.json")); c3a = json.loads(show(E59 + "checks/e59_check3_addendum_forecast.json"))
    brief = open("PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/TASK_E59_THE_UNRELATED_TASK_FROM_A_FRESH_A_BORN_AND_A_LIVED_NETWORK_20261117.md").read()
    ref = {}
    for m_ in re.finditer(r"n = ([\d,]+), row ([\d,]+)\s+([0-9a-f]{64})\s+([0-9a-f]{64})\s+([0-9a-f]{64})", brief):
        ref[(int(m_.group(1).replace(",", "")), int(m_.group(2).replace(",", "")))] = m_.groups()[2:]
    print(f"  the brief's reference digests read from its text: {sorted(ref)}")
    print(f"  check 1 receipt: pass {c1['pass']}; per seed {[c1['seeds'][str(sd)]['pass'] for sd in SEEDS]}; commit {c1['provenance']['git']['commit'][:8]}, modified {c1['provenance']['git']['sources_modified_vs_commit']}; "
          f"runner sha256 recorded {c1['provenance']['sources_sha256']['exp59_unrelated_starts.py'][:16]}")
    # the stream, the panels, the digests, the pairs: recomputed here
    t1 = time.time(); res = {}
    for sd in SEEDS:
        taught, held = split(sd)
        h = hashlib.sha256(); st = stream(sd, 2048)
        for _ in range(64 * 64):
            p, y = next(st); h.update(bytes([len(p)]) + bytes(p) + bytes(v % 256 for v in y))
        a_ok = h.hexdigest() == c1["seeds"][str(sd)]["a"]["e43_task_digest"] and h.hexdigest().startswith({73129: "34eefb4442b48399", 93683: "153cafdcda7379d0", 116089: "821bcb2a9ca6dd54"}[sd])
        disjoint = set(taught).isdisjoint(held)
        hp = panel(sd, "held"); held_in_held = all((p[1], p[2]) in set(held) for p, _ in hp)
        tp = {n: taught_panel(sd, n) for n in NS}
        tp_ok = tp[2048] == panel(sd, "taught") and all(len({(p[1], p[2]) for p, _ in tp[n]}) == min(300, n) and all((p[1], p[2]) in set(taught[:n]) for p, _ in tp[n]) for n in NS)
        for n in NS:
            st = stream(sd, n); hh = hashlib.sha256(); seen = set(); dig = {}
            for g in range(1, 8193):
                items = [next(st) for _ in range(64)]
                seen |= {(p[1], p[2]) for p, _ in items}
                assert all(b >= 128 for p, _ in items for b in p)
                for t in batch_tensor(items):
                    hh.update(str(tuple(t.shape)).encode()); hh.update(np.ascontiguousarray(t).tobytes())
                if g in ROWS:
                    dig[g] = hh.hexdigest()
            rec_rows = {x["updates"]: x["data_digest"]["sha256"] for x in R[("fresh", n, "S", sd)]["rows"] if x["updates"] > 0}
            i = SEEDS.index(sd)
            r64 = E43[("S", sd)]["rows"][1]["data_digest"]["sha256"] if n == 2048 else ref[(n, 64)][i]
            res[(sd, n)] = dict(rows_equal=all(dig[g] == rec_rows[g] for g in dig), n_rows=len(dig), row64_ref=dig[64] == r64, row8192_ref=dig[8192] == ref[(n, 8192)][i],
                                pairs=seen == set(taught[:n]), receipt64=c1["seeds"][str(sd)]["c"][str(n)]["row64"] == dig[64], receipt8192=c1["seeds"][str(sd)]["c"][str(n)]["row8192"] == dig[8192])
        first = {a for a, _ in taught[:256]}; second = {b for _, b in taught[:256]}
        unseen = sum(p[1] not in first or p[2] not in second for p, _ in hp)
        cnt1 = collections.Counter(a for a, _ in taught[:512]); cnt2 = collections.Counter(b for _, b in taught[:512])
        occ256 = [collections.Counter(a for a, _ in taught[:256]).get(s, 0) for s in SYM] + [collections.Counter(b for _, b in taught[:256]).get(s, 0) for s in SYM]
        print(f"  seed {sd}: (a) 64 batches of the stream with 2,048 pairs, by the lines of e43_task.digest, equal to the receipt and beginning as the brief: {a_ok}; "
              f"taught and held disjoint {disjoint}; held panel from held pairs {held_in_held}; taught panels as the brief (2,048: E43's) {tp_ok}")
        print(f"     unseen-place items at 256 pairs: {unseen}; never first {sorted(set(SYM) - first)}, never second {sorted(set(SYM) - second)}; "
              f"at 512 pairs least count in a place {min(min(cnt1.get(s, 0) for s in SYM), min(cnt2.get(s, 0) for s in SYM))}; at 256 counts in a place {min(occ256)} to {max(occ256)}")
        for n in NS:
            print(f"     n {n}: data digest recomputed with numpy at all {res[(sd, n)]['n_rows']} rows equal to the receipts {res[(sd, n)]['rows_equal']}; row 64 equal to the brief's (E43's for 2,048) "
                  f"{res[(sd, n)]['row64_ref']}; row 8,192 equal to the brief's {res[(sd, n)]['row8192_ref']}; equal to check 1's receipt {res[(sd, n)]['receipt64'] and res[(sd, n)]['receipt8192']}; "
                  f"pairs presented = taught[:n] {res[(sd, n)]['pairs']}")
    print(f"  (stream and digests recomputed in {time.time() - t1:.0f} s)")
    cmp("4 check 1", "unseen-place items at 256 pairs (three seeds)", "11, 6 and 2",
        ", ".join(str(R[("fresh", 256, "S", sd)]["settings"]["panels"]["held12_unseen_items"]) for sd in SEEDS[:2]) + " and " + str(R[("fresh", 256, "S", SEEDS[2])]["settings"]["panels"]["held12_unseen_items"]))
    # check 2
    print(f"  check 2 receipt: pass {c2['pass']}; per seed {[c2['seeds'][str(sd)]['pass'] for sd in SEEDS]}; byte table rows equal {[all(c2['seeds'][str(sd)]['byte_table_rows_128_to_225_equal_to_fresh_S'].values()) for sd in SEEDS]}; "
          f"born and lived 0 at row 0 {[c2['seeds'][str(sd)]['born_and_lived_zero_at_row0'] for sd in SEEDS]}; tensors reordered {sorted({v['info'].get('tensors_reordered') for sd in SEEDS for v in c2['seeds'][str(sd)]['starts'].values() if 'tensors_reordered' in v['info']})}")
    agree2 = all(c2["seeds"][str(sd)]["starts"][f"{s} {r}"]["info"]["tensors_sha256"] == R[(s, 2048, r, sd)]["settings"]["start_record"]["tensors_sha256"] for s in STARTS for r in RULES for sd in SEEDS)
    zero_runs = {s: all(R[(s, n, r, sd)]["rows"][0][p]["exact"] == 0 for n in (NS if s != "born-shuffled" else (2048,)) for r in RULES for sd in SEEDS for p in ("held12", "held0", "taught")) for s in STARTS}
    print(f"  check 2's start hashes equal to the runs' start records: {agree2}; every run at 0 on the three panels at row 0, by start: {zero_runs}")
    cmp("4 check 2", "tensors reordered in born-shuffled", "12", str(R[("born-shuffled", 2048, "S", 73129)]["settings"]["start_record"]["tensors_reordered"]))
    # check 3
    d64 = json.loads(show(E59 + "dry/e59_fresh_n256_S_73129_dry64.json")); d256 = json.loads(show(E59 + "dry/e59_fresh_n256_S_73129_dry256.json"))
    c = d64["complete"]; pu = c["cpu_updates"] / 64; pr = c["cpu_scoring"] / len(d64["rows"])
    f64 = 60 * (8192 * pu + 273 * pr) / 3600
    cmp("4 check 3", "the receipt's forecast (CPU-hours)", "29.6", f"{f64:.1f}")
    cmp("4 check 3", "the receipt's forecast as recorded", "29.6", f"{c3['forecast_60_runs_cpu_h']:.1f}")
    print(f"  runner sha256 recorded by checks 2 and 3: {c2['provenance']['sources_sha256']['exp59_unrelated_starts.py'][:16]}, {c3['provenance']['sources_sha256']['exp59_unrelated_starts.py'][:16]}; "
          f"their commits {c2['provenance']['git']['commit'][:8]}, {c3['provenance']['git']['commit'][:8]}; modified {c2['provenance']['git']['sources_modified_vs_commit']}, {c3['provenance']['git']['sources_modified_vs_commit']}")
    print(f"  check 3 receipt: pass {c3['pass']}; per update {c3['cpu_per_update']:.4f} (recomputed {pu:.4f}); per row {c3['cpu_per_row']:.3f} (recomputed {pr:.3f}) over {len(d64['rows'])} rows of the dry run")
    cr = [x["cpu"] for x in d256["rows"]]; c = d256["complete"]; pu2 = c3a["cpu_per_update"]
    ordr = [cr[i] - cr[i - 1] - 16 * pu2 for i in range(1, len(cr) - 1)]
    row0 = cr[0]; last = cr[-1] - cr[-2] - 16 * pu2
    f256 = 60 * (8192 * pu2 + 271 * np.mean(ordr) + row0 + last) / 3600
    fmax = 60 * (8192 * pu2 + 271 * max(ordr) + row0 + last) / 3600
    print(f"  addendum, recomputed from the 256-update dry run's rows (each row's CPU less 16 updates at {pu2:.4f}): ordinary rows {len(ordr)}, mean {np.mean(ordr):.3f}, "
          f"least {min(ordr):.3f}, most {max(ordr):.3f}; row 0 {row0:.2f} (receipt {c3a['special_rows_cpu']['row0']:.2f}); last {last:.2f} (receipt {c3a['special_rows_cpu']['last']:.2f}); "
          f"per update in the receipt {pu2:.4f}, from the dry run's complete record {c['cpu_updates'] / 256:.4f}")
    cmp("4 check 3", "the addendum's forecast (CPU-hours)", "18.4", f"{f256:.1f}")
    cmp("4 check 3", "the addendum's forecast with every ordinary row at its dearest", "19.4", f"{fmax:.1f}")
    other = (c1["cpu"] + c2["cpu"] + c3["cpu"] + d256["complete"]["cpu_total"]) / 3600
    cmp("4 check 3", "CPU-hours of the checks and the two dry runs", "0.07", f"{other:.2f}")
    print(f"  CPU of check 1 {c1['cpu']:.0f} s, check 2 {c2['cpu']:.0f} s, check 3 (with the 64-update dry run) {c3['cpu']:.0f} s, 256-update dry run {d256['complete']['cpu_total']:.0f} s")
    # check 4, recomputed from E43's receipts
    c4 = {}
    for r in RULES:
        for sd in SEEDS:
            e = {x["updates"]: x for x in E43[(r, sd)]["rows"]}; mine = {x["updates"]: x for x in R[("born", 2048, r, sd)]["rows"]}
            res4 = {u: (mine[u]["data_digest"] == e[u]["data_digest"], mine[u]["held12"]["exact"] == e[u]["score3_new_task"]["held"]["exact"],
                        mine[u]["taught"]["exact"] == e[u]["score3_new_task"]["taught"]["exact"], mine[u]["tensors_sha256"] == e[u]["tensors_sha256"]) for u in (64, 256, 1024, 2048, 4096)}
            c4[(r, sd)] = all(all(v) for v in res4.values())
            rec0 = mine[0]["tensors_sha256"] == e[0]["tensors_sha256"]
            print(f"  check 4, {r} {sd}: the four quantities equal at rows 64, 256, 1,024, 2,048, 4,096 {c4[(r, sd)]}; row 0 weights equal {rec0}; receipt's own field "
                  f"{R[('born', 2048, r, sd)]['check4']} -> pass {R[('born', 2048, r, sd)]['complete']['check4_pass']}; "
                  f"held/taught at 256 {mine[256]['held12']['exact']}/{mine[256]['taught']['exact']} (E43 {e[256]['score3_new_task']['held']['exact']}/{e[256]['score3_new_task']['taught']['exact']})")
    cmp("4 check 4", "equal bit for bit on the three seeds under both rules", "True", str(all(c4.values())))
    print(f"  check 5 has no receipt of its own among {names(E59 + 'checks/')}")
    return c1, c2, c3, c3a


# ================================================================ 5
def section5(R, D26, L26):
    print("\n================ 5  the weights")
    BORN = {}
    for r in RULES:
        for sd in SEEDS:
            blob = show(f"{M}exp45/e26/e26_dev_{r}_{sd}.pt", DREF); W = load_pt(blob)
            BORN[(r, sd)] = W
            sr = R[("born", 2048, r, sd)]["settings"]["start_record"]
            ok = sha(blob) == sr["weights_sha256"] == D26[(r, sd)]["weights_sha256"] and tensors_sha(W) == sr["tensors_sha256"]
            # the shuffled network, from the brief's definition
            Wsh = {}
            for name, t in W.items():
                if name == "book.weight":
                    Wsh[name] = t; continue
                perm = list(range(t.size)); Random(f"e59|{sd}|shuffle|{name}").shuffle(perm)
                Wsh[name] = t.reshape(-1)[np.array(perm)].reshape(t.shape)
            ssr = R[("born-shuffled", 2048, r, sd)]
            sh_ok = tensors_sha(Wsh) == ssr["settings"]["start_record"]["tensors_sha256"] == ssr["rows"][0]["tensors_sha256"]
            vals = all(np.array_equal(np.sort(W[k].ravel()), np.sort(Wsh[k].ravel())) for k in W)
            moved = sum(not np.array_equal(W[k], Wsh[k]) for k in W if k != "book.weight")
            blob = show(f"{M}exp45/e26/e26_life_{r}_{sd}.pt", DREF); WL = load_pt(blob)
            lr_ = R[("lived", 2048, r, sd)]["settings"]["start_record"]
            l_ok = sha(blob) == lr_["weights_sha256"] == L26[(r, sd)]["weights_sha256"] and tensors_sha(WL) == lr_["tensors_sha256"] == L26[(r, sd)]["tensors_sha256"]
            book = np.array_equal(W["book.weight"][128:226], WL["book.weight"][128:226])
            print(f"  {r} {sd}: born file sha256 and tensors as recorded {ok}; shuffled network rebuilt from the brief's definition has the recorded tensors_sha256 {sh_ok}, "
                  f"the same values in each tensor {vals}, tensors moved {moved} of {len(W) - 1}; lived file and tensors as recorded {l_ok}; byte table rows 128 to 225 equal born/lived {book}; "
                  f"parameters {sum(v.size for v in W.values()):,}")
    rows_eq = all(np.array_equal(BORN[("S", sd)]["book.weight"][128:226], BORN[("B", sd)]["book.weight"][128:226]) for sd in SEEDS)
    print(f"  byte table rows 128 to 225 equal under the two rules (born): {rows_eq}")
    # the committed final weights of seed 73129
    t1 = time.time(); fin = {}; okc = 0
    for (s, n, r, sd), rec in R.items():
        if sd != 73129:
            continue
        blob = show(f"{E59}e59_{s}_n{n}_{r}_{sd}_final.pt"); W = load_pt(blob)
        ok = sha(blob) == rec["complete"]["weights_sha256"] and tensors_sha(W) == rec["complete"]["tensors_sha256"] == rec["rows"][-1]["tensors_sha256"]
        okc += ok
        if (s, n) in (("fresh", 256), ("born", 256)):
            fin[(s, n, r)] = W
    cmp("5 weights", "final files of seed 73129 with the sha256 and tensors_sha256 of their receipts", "20", str(okc))
    print(f"  ({time.time() - t1:.0f} s)")
    # the forward in numpy: final scores of four runs, and the born networks at row 0
    for (s, n, r), W in fin.items():
        net = Net(W); sd = 73129; last = R[(s, n, r, sd)]["rows"][-1]
        h12 = net.counts(panel(sd, "held")); h0 = net.counts(panel(sd, "held", k=0)); tp = net.counts(taught_panel(sd, n))
        mine = (h12["exact"], h12["first"], h12["second"], h12["third"], h0["exact"], tp["exact"])
        rec_ = (last["held12"]["exact"], last["held12"]["first_byte"], last["held12"]["second_byte"], last["held12"]["third_is_end"], last["held0"]["exact"], last["taught"]["exact"])
        cmp("5 forward", f"{s} n{n} {r} 73129 final: held12 exact/first/second/third, held0, taught", str(rec_), str(mine))
    for r in RULES:
        W = BORN[(r, 73129)]; net = Net(W); row0 = R[("born", 2048, r, 73129)]["rows"][0]
        h12 = net.counts(panel(73129, "held"))
        cmp("5 forward", f"born {r} 73129 row 0: held12 exact/first/second/third", str((row0["held12"]["exact"], row0["held12"]["first_byte"], row0["held12"]["second_byte"], row0["held12"]["third_is_end"])),
            str((h12["exact"], h12["first"], h12["second"], h12["third"])))
        EP = earlier_panels(73129)
        mine = {e: net.counts(EP[e])["exact"] for e in ("add89", "sub89", "max89", "min89", "add97", "sub97", "add73", "sub73")}
        cmp("5 forward", f"born {r} 73129 row 0: earlier panels", str({e: v["exact"] for e, v in row0["earlier_panels_k12"].items()}), str(mine))
    # the brief's form count at row 64 (E43's arm without decay) beside this job's count (a reproduction of E43's arm with decay)
    for r in RULES:
        p = f"{M}exp45/e43/e43_unrelated_{r}_73129_row64.pt"
        if exists(p, DREF):
            net = Net(load_pt(show(p, DREF))); c = net.counts(panel(73129, "held"))
            print(f"  E43's arm without decay, {r}, 73129, row 64 (numpy): form 'symbol, symbol, end' {c['form']}; third byte the end symbol {c['third']}; first byte right {c['first']}; "
                  f"this job's born n2048 {r} 73129 row 64 (E43's arm with decay): third byte the end symbol {R[('born', 2048, r, 73129)]['rows'][4]['held12']['third_is_end']}, "
                  f"first byte right {R[('born', 2048, r, 73129)]['rows'][4]['held12']['first_byte']}")
        else:
            print(f"  {p} not on 7cabb13a")


# ================================================================ 6
def section6():
    print("\n================ 6  the runner as text (lines the document cites)")
    src = show(M + "exp59_unrelated_starts.py").decode().splitlines()
    pats = {"rows": r"^ROWS = ", "stream": r"def stream\(seed, n\)", "taught panel": r"rng\.sample\(taught, min\(300, n\)\)", "shuffle": r"Random\(f\"e59\|\{seed\}\|shuffle\|\{name\}\"\)",
            "optimizer": r"opt = C\.make_opt\(m\)", "update": r"v = C\.step\(m, opt, batch, local\)", "local": r"local = C\.ARMS\[rule\]", "held12 scorer": r"outs = E\.respond\(m, \[p for p, _ in self\.held12\]",
            "earlier panels": r"if g in \(0, updates\):", "forecast": r"forecast = 60 \* \(UPDATES \* per_update \+ len\(ROWS\) \* per_row\)", "per row": r"per_row = cs\[\"cpu_scoring\"\] / len\(rows\)",
            "check4 compare": r"tensors_sha256=row\[\"tensors_sha256\"\] == e\[\"tensors_sha256\"\]", "weights kept": r"keep = seed == 73129 and updates == UPDATES",
            "lived assert": r"assert info\[\"tensors_sha_matches_receipt\"\]", "born admit": r"E38\.admit\(bfile, rule, seed\)", "install": r"arch26\.install\(\)"}
    for k, p in pats.items():
        ln = [i + 1 for i, l in enumerate(src) if re.search(p, l)]
        print(f"  {k}: line(s) {ln}")
    print(f"  lines {len(src)}; calls of a random generator in training: {[i + 1 for i, l in enumerate(src) if 'manual_seed' in l]}")


# ================================================================ 7
NAMES_ROT13 = ["pynhqr", "bchf", "fbaarg", "unvxh", "tcg", "pbqrk", "trzvav", "yynzn", "zvfgeny", "snoyr"]


def section7(report, allf):
    print("\n================ 7  form")
    for pat in ("@@", "TODO", "TBD", "XX", "FIXME", "<seed>", "???", "<rule>", "<start>"):
        print(f"  '{pat}' in the report: {report.count(pat)}")
    pats = [re.compile(r"\b" + codecs.decode(e, "rot13") + r"\b", re.I) for e in NAMES_ROT13]
    files = [p for p in allf if not p.endswith(".pt")] + [M + "exp59_unrelated_starts.py", M + "e59_tables.py"]
    hits = collections.defaultdict(collections.Counter)
    for p in files:
        txt = show(p).decode("utf-8", "replace")
        for i, rx in enumerate(pats):
            for m_ in rx.finditer(txt):
                before = txt[max(0, m_.start() - 12):m_.start()]
                after = txt[m_.end():m_.end() + 1]
                kind = ("in a branch name" if after == "/" or before.endswith("origin/") else "in a link to a session" if "ai/" in before or "code/" in before else "elsewhere")
                hits[p[len(M):]][f"<name {i + 1}> {kind}"] += 1
    print(f"  text files searched: {len(files)}; files with a name of a language model (held encoded; shown as <name k>): {len(hits)}; occurrences {sum(sum(v.values()) for v in hits.values())}")
    agg = collections.Counter()
    for p, v in hits.items():
        agg.update(v)
    print(f"  occurrences by name and kind of place: {dict(agg)}")
    kinds = {("the report" if p.endswith("E59_REPORT.md") else "table or summary" if "E59_" in p else "receipt" if p.endswith(".json") else "log" if "/logs/" in p else "the runner" if "exp59" in p else "script"): sum(v.values()) for p, v in hits.items()}
    print(f"  occurrences by file: {kinds}")
    log = subprocess.run(["git", "log", "--format=%H%x00%B%x01", NEWREF or WREF, "--", E59, M + "exp59_unrelated_starts.py", M + "e59_tables.py"], capture_output=True, text=True, check=True).stdout
    msgs = [m_.split("\x00", 1) for m_ in log.split("\x01") if "\x00" in m_]
    in_tr = out_tr = 0
    for h, body in msgs:
        for line in body.splitlines():
            n_ = sum(len(rx.findall(line)) for rx in pats)
            if re.match(r"^\s*[A-Za-z-]+: ", line) and ("Acting-Agent" in line or "Session" in line):
                in_tr += n_
            else:
                out_tr += n_
    print(f"  commit messages of the commits that touch E59's files (up to {(NEWREF or WREF)[:8]}): {len(msgs)}; names in trailer lines {in_tr}; names elsewhere in the messages {out_tr}")
    dec = [p for p in names("PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117", "16b0e494f2a9b9f410b01e9bd0d227aae07c7d54") if p.endswith("COORDINATOR_DECISIONS_20261118.md")]
    txt = show(dec[0], "16b0e494f2a9b9f410b01e9bd0d227aae07c7d54").decode() if dec else ""
    d2 = bool(re.search(r"^\| 2 \|.*E59", txt, re.M))
    print(f"  the coordinator's decisions of 18 November on the default branch's head 16b0e494: present {bool(dec)}; its decision 2 names E59: {d2}")
    # the list of committed files in the report
    listed = re.findall(r"^- `([^`]+)`", report.split("## Committed files of E59")[1], re.M)
    tree = subprocess.run(["git", "ls-tree", "-r", "--name-only", "d9e785d9"], capture_output=True, text=True, check=True).stdout.split()
    e59_tree = [p for p in tree if p.startswith(E59) or p in (M + "exp59_unrelated_starts.py", M + "e59_tables.py")]
    not_e59 = [p for p in listed if not (p.startswith(E59) or p in (M + "exp59_unrelated_starts.py", M + "e59_tables.py"))]
    print(f"  files listed in the report: {len(listed)} (stated count '(179 files.)' present: {'(179 files.)' in report}); E59 files at d9e785d9: {len(e59_tree)}; "
          f"listed and E59's {len(set(listed) & set(e59_tree))}; E59's and not listed {sorted(set(e59_tree) - set(listed))}; listed and not E59's {len(not_e59)}")
    for p in not_e59:
        print(f"    listed, not a file of E59: {p}")
    cmp("7 form", "files listed as committed files of E59", "179", str(len(listed)))
    cmp("7 form", "of which files of E59", "179", str(len(listed) - len(not_e59)))
    e59_head = [p for p in names(E59)] + [M + "exp59_unrelated_starts.py", M + "e59_tables.py"]
    print(f"  E59 files at the report's commit {WREF[:8]}: {len(e59_head)} (the report and the two later summaries beside the {len(e59_tree)} of d9e785d9: {sorted(set(e59_head) - set(e59_tree))})")
    # the dry runs' weights
    dry = [p for p in allf if "/dry/" in p and p.endswith(".pt")]
    first = subprocess.run(["git", "ls-tree", "-r", "--name-only", "349aacb0", "--", E59 + "dry"], capture_output=True, text=True).stdout.split()
    parent = subprocess.run(["git", "ls-tree", "-r", "--name-only", "349aacb0^", "--", E59 + "dry"], capture_output=True, text=True).stdout.split()
    okd = []
    for p in dry:
        tag = p.rsplit("/", 1)[1].replace("_final.pt", "")
        rec = json.loads(show(f"{E59}dry/{tag}.json")); okd.append(sha(show(p)) == rec["complete"]["weights_sha256"])
    print(f"  dry runs' weights committed at {WREF[:8]}: {[p[len(E59):] for p in dry]}; at 349aacb0: {[p[len(E59):] for p in first if p.endswith('.pt')]}; at its parent: {[p for p in parent if p.endswith('.pt')]}; "
          f"sha256 equal to the dry receipts {okd}")
    # the brief as read at 39057293
    bpath = "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/TASK_E59_THE_UNRELATED_TASK_FROM_A_FRESH_A_BORN_AND_A_LIVED_NETWORK_20261117.md"
    print(f"  the brief at 39057293 equal to the brief on the reviewer's checked-out branch: {sha(show(bpath, '39057293')) == sha(open(bpath, 'rb').read())}")
    tab = show(E59 + 'E59_TABLES_all.md').decode().splitlines()
    rl = [l for l in report.splitlines() if l.startswith('| ') and not l.startswith('|---')]
    miss = [l for l in rl if l not in tab]
    print(f"  the report's table rows found verbatim in the worker's E59_TABLES_all.md: {len(rl) - len(miss)} of {len(rl)}; the others (headers and predictions): {[l[:60] for l in miss]}")


# ================================================================ 8
def section8(R):
    print("\n================ 8  the newer head of the worker's branch")
    if not NEWREF:
        print("  not set"); return
    print(f"  head {NEWREF}")
    allf = names(E59, NEWREF)
    want = {f"e59_{k[0]}_n{k[1]}_{k[2]}_{k[3]}_final.pt": k for k in R if k[3] != 73129}
    here = {p.rsplit("/", 1)[1]: p for p in allf if p.endswith("_final.pt")}
    found = [f for f in want if f in here]
    print(f"  files under exp45/e59: {len(allf)}; of the 40 further final weight files present: {len(found)}; their folders {sorted({here[f][len(E59):].rsplit('/', 1)[0] if '/' in here[f][len(E59):] else '.' for f in found})}")
    ok = 0; spot = {}
    for f in sorted(found):
        blob = show(here[f], NEWREF); k = want[f]; rec = R[k]
        W = load_pt(blob)
        good = sha(blob) == rec["complete"]["weights_sha256"] and tensors_sha(W) == rec["complete"]["tensors_sha256"]
        ok += good
        if not good:
            print(f"    differs: {f}")
        if k in (("fresh", 256, "S", 93683), ("born", 256, "B", 116089), ("lived", 256, "S", 116089), ("fresh", 256, "B", 116089)):
            spot[k] = W
    if found:
        cmp("8 newer head", "further final files with the sha256 and tensors_sha256 of their receipts", "40", str(ok))
    for k, W in spot.items():
        s, n, r, sd = k; net = Net(W); last = R[k]["rows"][-1]
        h12 = net.counts(panel(sd, "held")); h0 = net.counts(panel(sd, "held", k=0)); tp = net.counts(taught_panel(sd, n))
        cmp("8 forward", f"{s} n{n} {r} {sd} final: held12 exact/first/second/third, held0, taught",
            str((last["held12"]["exact"], last["held12"]["first_byte"], last["held12"]["second_byte"], last["held12"]["third_is_end"], last["held0"]["exact"], last["taught"]["exact"])),
            str((h12["exact"], h12["first"], h12["second"], h12["third"], h0["exact"], tp["exact"])))
    # the receipts and the report at the newer head
    same_receipts = all(sha(show(f"{E59}e59_{k[0]}_n{k[1]}_{k[2]}_{k[3]}.json", NEWREF)) == sha(show(f"{E59}e59_{k[0]}_n{k[1]}_{k[2]}_{k[3]}.json")) for k in R)
    print(f"  the 60 receipts unchanged at the newer head: {same_receipts}")
    old = show(E59 + "E59_REPORT.md").decode().splitlines(); new = show(E59 + "E59_REPORT.md", NEWREF).decode().splitlines()
    import difflib
    d = [l for l in difflib.unified_diff(old, new, lineterm="", n=0) if not l.startswith(("---", "+++"))]
    print(f"  the report changed: {old != new}; changed lines: {sum(1 for l in d if l[:1] in '+-')}")
    for l in d:
        print("    " + l[:400])
    newf = sorted(set(allf) - set(names(E59)))
    print(f"  files added under exp45/e59 since {WREF[:8]}: {len(newf)}; not final weights: {[p[len(E59):] for p in newf if not p.endswith('_final.pt')]}")


if __name__ == "__main__":
    report = show(E59 + "E59_REPORT.md").decode()
    R, allf, E43, D26, L26 = section1(); clock("1")
    S, W = section2(R, report); clock("2")
    section3(S, W, R); clock("3")
    section4(R, E43); clock("4")
    section5(R, D26, L26); clock("5")
    section6(); clock("6")
    section7(report, allf); clock("7")
    section8(R); clock("8")
    print("\n================ every figure compared: [section] label: recomputed (reported, where it differs)")
    for sec, label, rp, v in COMP:
        print(f"  [{sec}] {label}: {v}" + ("" if rp == v else f"   (reported {rp!r})"))
    print("\n================ summary: reported against recomputed")
    diff = [c for c in COMP if c[2] != c[3]]
    bysec = collections.Counter(c[0] for c in COMP)
    print(f"figures compared: {len(COMP)} ({dict(bysec)}); agree: {len(COMP) - len(diff)}; differ: {len(diff)}")
    for sec, label, rp, v in diff:
        print(f"  [{sec}] {label}: reported {rp!r}, recomputed {v!r}")
    print(f"\nrunning time: {time.time() - T0:.1f} s")
