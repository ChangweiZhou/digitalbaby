"""Verification of the final report of E58 (a new family learned in a gated term against free learning; the worker's commit
e4c9bc0b). Read-only: nothing is trained, torch is not used, and no file of the worker's branch is executed. Every file of the
worker's branch is read with `git show` into memory and listed with `git ls-tree -r --name-only`; single files of the default
branch (7cabb13a) and of E48's worker's commit (2d9d1726) are read in the same way. The script writes no file.

Sections (the numbering of the verification document):
  1  inventory: the files of exp45/e58, the roster of 72 runs, the committed sizes against the report's list
  2  every receipt: completeness, settings, identities between runs, the restarted runs
  3  every figure of the report recomputed from the receipts and set beside the reported figure
  4  the eight checks: what each receipt records, and the parts recomputed here (streams, digests, pairs, check 1, 2, 6, 7)
  5  the weights: the six lived networks, the 42 weight files, the 48 term files, the norms; the forward recomputed with numpy
  6  the runner and the imported sources as text
  7  form: placeholders and names of language models in the files of exp45/e58 (the names are held in encoded form)

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e58_verify.py
"""
import codecs, hashlib, json, os, re, subprocess, sys, time
from random import Random
import numpy as np

T0 = time.time()
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.getcwd(), "PROJECT_DOCUMENTS", "MINIFLY_REVIEW_20261109"))
from compare_pt_tensors import tensors      # the reviewer's reader of torch.save files (no torch, no code from the file)

WREF = "e4c9bc0bd4a73bd2e160d8e98aa5f6051eb0a4dd"     # the worker's head
DREF = "7cabb13adc38753483d34dd5472fc3c84c4de675"     # the default branch at the brief's base
W8REF = "2d9d1726fe52056d999363c222445fda54513bfb"    # E48's worker's head
M = "MOD97_LOCAL_RULE_20261106/"
E58 = M + "exp45/e58/"
SEEDS = (73129, 93683, 116089)
RULES = ("S", "B")
ROWS = (0, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192)
LATE, MID = (4096, 5120, 6144, 7168, 8192), (512, 1024)
FT = {"05": 0.05, "10": 0.10, "20": 0.20}
NT = {"05": 344, "10": 689, "20": 1378}
PART_A = ("free", "free-again", "free-wd0", "free-anchor", "term", "term-wd0")
PART_B = ("free", "free-wd0", "term")
LIFE4, FIRST2 = ("add97", "sub97", "add73", "sub73"), ("add89", "sub89")
EOS, BOS = 256, 257
DTYPE = {"FloatStorage": "<f4", "DoubleStorage": "<f8", "LongStorage": "<i8", "BoolStorage": "?"}
COMP = []          # (section, label, reported, recomputed, decimals)


def clock(label):
    print(f"[{label}: {time.time() - T0:.1f} s]", flush=True)


# ---------------------------------------------------------------- reading through git
def show(path, ref=WREF):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def names(d, ref=WREF):
    return subprocess.run(["git", "ls-tree", "-r", "--name-only", ref, "--", d], capture_output=True, check=True, text=True).stdout.split()


def load_pt(path, ref=WREF):
    blob = show(path, ref)
    W = {n.lstrip("/"): np.frombuffer(raw, dtype=DTYPE[st]).reshape(shape).copy() for n, (st, shape, raw) in tensors(blob).items()}
    return W, hashlib.sha256(blob).hexdigest(), len(blob)


def tensors_sha(W):
    """exp20_reference.tensors_sha as read: per name in sorted order, the name, torch's dtype string, the shape, the bytes."""
    h = hashlib.sha256()
    for k in sorted(W):
        v = W[k]
        h.update(k.encode()); h.update(b"torch.float32" if v.dtype == np.float32 else b"torch.float64")
        h.update(str(tuple(v.shape)).encode()); h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


_rec = {}


def rec(arm, rule, ft, seed):
    k = (arm, rule, ft, seed)
    if k not in _rec:
        _rec[k] = json.loads(show(f"{E58}e58_{arm}_{rule}_f{ft}_{seed}.json"))
    return _rec[k]


def rows(r):
    return {x["updates"]: x for x in r["rows"]}


def red(x):
    return x["scores"]["summary"]["reduction_score"]


def com(x):
    return x["common"]["common_score"]


def pct(d):
    return 100.0 * d["exact"] / d["n"]


def mean(v):
    return sum(v) / len(v)


def life4(x):
    return mean([pct(x["scores"]["earlier_k12"][n]) for n in LIFE4])


def first2(x):
    return mean([pct(x["scores"]["earlier_k12"][n]) for n in FIRST2])


def runs_all():
    out = [(a, r, "10", s) for a in PART_A for r in RULES for s in SEEDS]
    out += [(a, r, f, s) for a in PART_B for r in RULES for f in ("05", "20") for s in SEEDS]
    return out


def cmp(sec, label, reported, value, nd=1):
    COMP.append((sec, label, reported, value, nd))


def agrees(reported, value, nd):
    if isinstance(reported, str) or isinstance(value, str):
        return str(reported) == str(value)
    return round(value + 0.0, nd) == round(reported, nd) or abs(round(value, nd) - reported) < 0.5 * 10 ** -nd


def fnum(s):
    return float(s.replace("−", "-").replace("+", "").replace(",", ""))


# ---------------------------------------------------------------- items, pairs, streams (from a reading of exp11_cell, exp16_integrated, exp45, e40_new_family)
F4 = [("add", 83, b"e"), ("sub", 83, b"f"), ("max", 83, b"g"), ("min", 83, b"h")]
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def apply(op, m, a, b):
    return {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]


def render(task, a, b, k, rng):
    op, m, code = task
    d = [rng.choice(DISTRACT) for _ in range(k)]
    return list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")], \
        list(f"{apply(op, m, a, b):02d}".encode()) + [EOS]


def split(task, frac, seed):
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]).shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def nested(t, f, seed):
    taught, rest = split(t, 0.10, seed)
    pairs = taught + rest
    k = round(f * len(pairs))
    return pairs[:k], pairs[k:]


def needs_reduction(op, a, b):
    return a + b >= 83 if op == "add" else a - b < 0


def common_pairs(t, seed):
    t05 = nested(t, 0.05, seed)[0]
    first = {a for a, _ in t05}; second = {b for _, b in t05}
    return [(a, b) for a, b in nested(t, 0.20, seed)[1] if needs_reduction(t[0], a, b) and a in first and b in second]


def reduction_pairs(t, f, seed):
    return [(a, b) for a, b in nested(t, f, seed)[1] if needs_reduction(t[0], a, b)]


def entry_stream(entries, seed):
    rng = Random(seed + 77)
    while True:
        order = list(range(len(entries))); rng.shuffle(order)
        for i in order:
            t, a, b = entries[i]
            yield render(t, a, b, rng.randint(0, 12), rng)


def digest_add(h, items):
    """exp26_candidate.Digest.add as read: x, y and the answer mask of exp45.batch_tensor, each as its shape string and its bytes."""
    rws = [p + t for p, t in items]; L = max(len(r) for r in rws) + 1
    data = np.full((len(rws), L), BOS, dtype=np.int64)
    for i, r in enumerate(rws):
        data[i, 1:len(r) + 1] = r
    x, y = np.ascontiguousarray(data[:, :-1]), np.ascontiguousarray(data[:, 1:])
    mask = np.zeros(y.shape, bool)
    for i, (p, t) in enumerate(items):
        mask[i, len(p):len(p) + len(t)] = True
    for a in (x, y, mask):
        h.update(str(tuple(a.shape)).encode()); h.update(a.tobytes())


def walk_stream(ft, seed, offset):
    """The stream of a run (offset 140; 141 for free-again), 8,192 batches of 64: the digest at every row and what was presented."""
    f = FT[ft]
    taught = {t[0]: nested(t, f, seed)[0] for t in F4}
    entries = [(t, a, b) for t in F4 for a, b in taught[t[0]]]
    tset = {(t[0], a, b) for t in F4 for a, b in taught[t[0]]}
    st = entry_stream(entries, seed + offset); h = hashlib.sha256(); dig = {}; seen = set(); untaught = 0; u = 0
    for r in ROWS:
        while u < r:
            items = [next(st) for _ in range(64)]
            for p, _ in items:
                key = ({101: "add", 102: "sub", 103: "max", 104: "min"}[p[0]], int(bytes(p[1:3])), int(bytes(p[-3:-1])))
                seen.add(key); untaught += key not in tset
            digest_add(h, items); u += 1
        dig[r] = h.copy().hexdigest()
    return dig, untaught, len(tset - seen), len(seen - tset)


# ---------------------------------------------------------------- the network (arch26's GRU, with E48's gated term), forward only
def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Net:
    """byte table (258 x 8) -> l1 (8 -> 512) with tanh = z1; GRU of 256 cells in torch's form (rows r, z, n); readout
    z2 = tanh(l2 z1 + lm h), then l3. With a term: for a gated item, up_i (down_i z1) is added to the input map and
    up_h (down_h h) to the recurrent map, and the four code rows of the byte table are the term file's."""

    def __init__(self, W, term=None, dtype=np.float32):
        W = {k: np.asarray(v, dtype) for k, v in W.items()}
        book = W["book.weight"].copy()
        if term is not None:
            book[101:105] = np.asarray(term["code_rows"], dtype)
        z1 = np.tanh(book @ W["l1.weight"].T + W["l1.bias"])
        self.gi = z1 @ W["rnn.weight_ih_l0"].T + W["rnn.bias_ih_l0"]
        if term is not None:
            self.gi = self.gi + (z1 @ np.asarray(term["down_i"], dtype).T) @ np.asarray(term["up_i"], dtype).T
            self.dhT, self.uhT = np.asarray(term["down_h"], dtype).T.copy(), np.asarray(term["up_h"], dtype).T.copy()
        self.term = term is not None
        self.a2 = z1 @ W["l2.weight"].T + W["l2.bias"]
        self.WhT, self.bh = W["rnn.weight_hh_l0"].T.copy(), W["rnn.bias_hh_l0"]
        self.lmT, self.lmb = W["lm.weight"].T.copy(), W["lm.bias"]
        self.l3T, self.l3b = W["l3.weight"].T.copy(), W["l3.bias"]
        self.dtype = dtype

    def step(self, h, byte, gate):
        gi = self.gi[byte]; gh = h @ self.WhT + self.bh
        if self.term and gate:
            gh = gh + (h @ self.dhT) @ self.uhT
        U = 256
        r = sigmoid(gi[:, :U] + gh[:, :U]); z = sigmoid(gi[:, U:2 * U] + gh[:, U:2 * U])
        n = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:])
        return (1 - z) * n + z * h

    def generate(self, prompts, cap=6):
        """exp11_cell.respond as read: [start] + prompt, then the byte of the largest output appended, up to `cap` bytes or the end
        symbol; the state is carried from step to step. All prompts here have the same length and the same kind of code byte."""
        x = np.array([[BOS] + p for p in prompts]); gate = bool(101 <= x[0, 1] <= 104)
        assert all(bool(101 <= v <= 104) == gate for v in x[:, 1])
        h = np.zeros((len(x), 256), self.dtype)
        for t in range(x.shape[1]):
            h = self.step(h, x[:, t], gate)
        last = x[:, -1]; gen = [[] for _ in prompts]; done = np.zeros(len(x), bool)
        for _ in range(cap):
            z2 = np.tanh(self.a2[last] + h @ self.lmT + self.lmb)
            y = (z2 @ self.l3T + self.l3b).argmax(1)
            for q in np.nonzero(~done)[0]:
                gen[q].append(int(y[q]))
            done |= y == EOS
            if done.all():
                break
            h = self.step(h, y, gate); last = y
        return gen


def exact(net, items):
    outs = net.generate([p for p, _ in items])
    return sum(o == t for o, (_, t) in zip(outs, items))


# ================================================================ 1  inventory
def section1():
    print("\n================ 1  inventory of exp45/e58 at the worker's head")
    allf = names(E58)
    print("files under exp45/e58:", len(allf))
    want = {f"{E58}e58_{a}_{r}_f{f}_{s}.json" for a, r, f, s in runs_all()}
    have = {p for p in allf if re.fullmatch(re.escape(E58) + r"e58_[a-z0-9-]+_[SB]_f\d\d_\d+\.json", p)}
    print("roster of the brief: runs", len(want), "| receipts present", len(have), "| missing", sorted(want - have), "| not in roster", sorted(have - want))
    wts = {p for p in allf if p.endswith("_row8192.pt") and "/e58_free" in p}
    want_w = {f"{E58}e58_{a}_{r}_f{f}_{s}_row8192.pt" for a, r, f, s in runs_all() if a in ("free", "free-wd0", "free-anchor")}
    print("weight files of free, free-wd0, free-anchor: expected", len(want_w), "present", len(wts & want_w), "extra", sorted(wts - want_w))
    tf = {p for p in allf if re.search(r"/e58_term[a-z0-9-]*_[SB]_f\d\d_\d+_row(1024|8192)\.pt$", p)}
    want_t = {f"{E58}e58_{a}_{r}_f{f}_{s}_row{u}.pt" for a, r, f, s in runs_all() if a.startswith("term") for u in (1024, 8192)}
    print("term files at rows 1,024 and 8,192: expected", len(want_t), "present", len(tf & want_t), "extra", sorted(tf - want_t))
    pts = [p for p in allf if ".pt" in p]
    print("all files with .pt in the name:", len(pts), "| of them in interrupted/:", len([p for p in pts if "/interrupted/" in p]))
    print("free-again weight files (the brief: receipt only):", len([p for p in pts if "free-again" in p]))
    cats = {"qual": [p for p in allf if "/qual/" in p], "logs": [p for p in allf if "/logs/" in p],
            "interrupted": [p for p in allf if "/interrupted/" in p], "queues": [p for p in allf if "/queues/" in p]}
    for k, v in cats.items():
        print(f"  {k}: {len(v)}")
    other = [p for p in allf if p not in want and p not in want_w and p not in want_t and not any(p in v for v in cats.values())]
    print("  other files:", [p[len(E58):] for p in other])
    # sizes, from the blobs themselves, against the report's list
    report = show(E58 + "E58_REPORT.md").decode()
    listed = {m.group(1): int(m.group(2).replace(",", "")) for m in re.finditer(r"- `([^`]+)` \(([\d,]+) bytes\)", report)}
    sizes = {}
    for p in allf:
        sizes[p] = len(show(p))
    for s in ("E58_COPIED_SOURCES.txt", "e48_gated.py", "exp48_gated_run.py", "exp58_new_family.py"):
        sizes[M + s] = len(show(M + s))
    bad = []
    for k, v in listed.items():
        full = (M + k)
        if full not in sizes:
            bad.append((k, "not on the branch"))
        elif sizes[full] != v:
            bad.append((k, v, sizes[full]))
    print("report's list: entries with a size", len(listed), "| size differences or missing:", bad)
    unlisted = [p[len(M):] for p in sizes if p[len(M):] not in listed and not p.endswith("E58_REPORT.md")]
    print("files of exp45/e58 (and the four sources) not in the report's list:", unlisted)
    tot = sum(v for p, v in sizes.items() if p.startswith(E58) and not p.endswith("E58_REPORT.md"))
    print(f"total of exp45/e58 without the report: {tot:,} bytes = {tot / 1e6:.1f} MB (10^6) = {tot / 2**20:.1f} MiB")
    cmp("8", "total size of exp45/e58 without the report, MB (10^6 bytes)", 202.0, tot / 1e6, 1)
    counts = dict(receipts=len(have), weights=len(wts), term_files=len(tf), qual=len(cats["qual"]), logs=len(cats["logs"]),
                  interrupted=len(cats["interrupted"]))
    for k, v in (("receipts", 72), ("weights", 42), ("term_files", 48), ("qual", 18), ("logs", 73), ("interrupted", 13)):
        cmp("8", f"number of files: {k}", v, counts[k], 0)
    return report, allf


# ================================================================ 2  receipts: completeness, settings, identities
def section2(allf):
    print("\n================ 2  receipts: completeness, settings, identities between runs")
    runner_sha = hashlib.sha256(show(M + "exp58_new_family.py")).hexdigest()
    src_sha = {s: hashlib.sha256(show(M + s)).hexdigest() for s in ("exp48_gated_run.py", "e48_gated.py", "e40_new_family.py")}
    print("SHA-256 of the runner at the head:", runner_sha)
    print("SHA-256 of the three imported files at the head:", src_sha)
    w8 = {s: hashlib.sha256(show(M + s, W8REF)).hexdigest() for s in ("exp48_gated_run.py", "e48_gated.py")}
    print("the two copied files equal E48's worker's files byte for byte:", {s: show(M + s) == show(M + s, W8REF) for s in w8})
    spec = show("PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e48_gated_spec.py", DREF)
    print("e48_gated.py equals the default branch's e48_gated_spec.py:", spec == show(M + "e48_gated.py"))
    lib = ("arch26.py", "exp26_candidate.py", "arch20.py", "exp20_reference.py", "exp16_integrated.py", "arch17.py", "exp11_cell.py",
           "exp45.py", "exp3_state.py", "mod97_local.py", "e40_new_family.py", "e36_reminder.py", "arch31_workerB.py", "exp31_switch_workerB.py")
    same_lib = {s: show(M + s) == show(M + s, DREF) for s in lib}
    print("imported library files equal at the worker's head and at the default branch:", all(same_lib.values()), [s for s, v in same_lib.items() if not v])
    lib_sha = {s: hashlib.sha256(show(M + s)).hexdigest() for s in lib}
    life = {}
    for r in RULES:
        for s in SEEDS:
            life[(r, s)] = json.loads(show(f"{M}exp45/e26/e26_life_{r}_{s}.json", DREF))["complete"]
    problems = []
    ex_groups = {
        "free": [(0.25, "rest"), (0.0, ["book.weight"])], "free-again": [(0.25, "rest"), (0.0, ["book.weight"])],
        "free-wd0": [(0.0, "rest"), (0.0, ["book.weight"])], "free-anchor": [(0.0, "rest"), (0.0, ["book.weight"])],
        "term": [(0.25, ["term.down_i", "term.up_i", "term.down_h", "term.up_h"]), (0.0, ["book.weight"])],
        "term-wd0": [(0.0, ["term.down_i", "term.up_i", "term.down_h", "term.up_h"]), (0.0, ["book.weight"])]}
    REST = ["l1.weight", "l1.bias", "rnn.weight_ih_l0", "rnn.weight_hh_l0", "rnn.bias_ih_l0", "rnn.bias_hh_l0", "lm.weight", "lm.bias",
            "l2.weight", "l2.bias", "l3.weight", "l3.bias"]
    red_n = {(ft, s): {t[0]: len(reduction_pairs(t, FT[ft], s)) for t in F4[:2]} for ft in FT for s in SEEDS}
    comm = {s: {f"{t[0]}83": len(common_pairs(t, s)) for t in F4[:2]} for s in SEEDS}
    nchecked = 0
    for a, r_, ft, s in runs_all():
        R_ = rec(a, r_, ft, s); rw = R_["rows"]; p = R_["provenance"]
        def need(c, msg):
            if not c:
                problems.append(f"{a} {r_} f{ft} {s}: {msg}")
        need(R_["complete"] is True, "not complete")
        need((R_["arm"], R_["rule"], R_["seed"], R_["fraction"], R_["updates"]) == (a, r_, s, FT[ft], 8192), "settings differ from the name")
        need(tuple(R_["rows_planned"]) == ROWS and tuple(x["updates"] for x in rw) == ROWS, "rows are not the brief's")
        need(rw[-1]["optimizer_steps"] == [8192], "optimizer steps at the last row")
        need(all(x["data_digest"]["batches"] == x["updates"] for x in rw), "batches counted")
        need(all(x["presented"] == dict(items=64 * x["updates"], held_out_presented=0, not_taught_presented=0, earlier_code_items=0) for x in rw), "presented")
        need(all(x["finite_losses"] for x in rw), "losses not finite")
        need(R_["start"]["start"] == f"exp45/e26/e26_life_{r_}_{s}.pt", "start file")
        need(R_["start"]["tensors_sha256"] == life[(r_, s)]["tensors_sha256"] == R_["start"]["receipt_tensors_sha256"], "start tensors")
        need(R_["start"]["weights_sha256"] == life[(r_, s)]["weights_sha256"], "start file hash")
        need(rw[0]["tensors_sha256"] == life[(r_, s)]["tensors_sha256"], "row 0 tensors are not the lived network's")
        fa = R_["fraction_assertions"]
        need(fa["taught_pairs"] == {o: NT[ft] for o in ("add", "sub", "max", "min")} and fa["stream_entries"] == 4 * NT[ft]
             and fa["untaught_plain_items"] == {"add83": 6889 - NT[ft], "sub83": 6889 - NT[ft]} and fa["d_checked"] == (ft == "10"), "fraction assertions")
        need(R_["common_pairs"] == comm[s], "common pairs")
        need(all(x["common"]["counts"]["add83"]["n"] == comm[s]["add83"] and x["common"]["counts"]["sub83"]["n"] == comm[s]["sub83"] for x in rw), "common counts n")
        need(all(x["scores"]["held_plain_k0"]["add83"]["reduction"]["n"] == red_n[(ft, s)]["add"] and
                 x["scores"]["held_plain_k0"]["sub83"]["reduction"]["n"] == red_n[(ft, s)]["sub"] for x in rw), "reduction panel not at the run's fraction")
        need(all(x["scores"]["held_plain_k0"][n]["reduction"]["n"] + x["scores"]["held_plain_k0"][n]["no_reduction"]["n"] == 6889 - NT[ft]
                 for x in rw for n in ("add83", "sub83")), "untaught plain panel size")
        g = R_["optimizer"]["read_back_groups"]
        eg = ex_groups[a]
        need(len(g) == 2 and all(abs(gg["lr"] - 0.001) < 1e-12 and gg["betas"] == [0.9, 0.98] for gg in g), "lr or betas")
        need(all(gg["weight_decay"] == e[0] and gg["params"] == (REST if e[1] == "rest" else e[1]) for gg, e in zip(g, eg)), "optimizer groups")
        need(("anchor" in R_["update"]) == (a == "free-anchor"), "update text")
        need(("seed + 141" in R_["data"]) == (a == "free-again"), "data text")
        need(p["e58_sources_sha256"] == dict(src_sha, **{"exp58_new_family.py": runner_sha}) or
             (p["e58_sources_sha256"]["exp58_new_family.py"] == runner_sha and all(p["e58_sources_sha256"][k] == v for k, v in src_sha.items())), "source hashes")
        need(all(p["sources_sha256"][k] == lib_sha[k] for k in p["sources_sha256"]), "library hashes")
        need(p["versions"] == {"python": "3.13.18", "torch": "2.15.0+cpu", "numpy": "2.5.5"}, "versions")
        need(p["host"]["torch_threads"] == 1 and p["environment_at_start"].get("OMP_NUM_THREADS") == "1", "threads")
        need(p["git"]["sources_modified_vs_commit"] == [], "sources modified against the commit")
        if a != "free-again":
            need(all(x.get("digest_equals_check5") is True for x in rw), "digest_equals_check5")
        else:
            need(all("digest_equals_check5" not in x for x in rw), "free-again asserts digests")
        if a.startswith("term"):
            need(R_["rank"] == 16 and R_["numbers_that_may_change"] == 16 * 512 + 768 * 16 + 16 * 256 + 768 * 16 + 32, "rank or size of the term")
            need(R_["trainable"] == ["book.weight", "term.down_i", "term.up_i", "term.down_h", "term.up_h"], "trainable")
            need(all(x["term_file"]["file"] == f"e58_{a}_{r_}_f{ft}_{s}_row{x['updates']}.pt" for x in rw if x["updates"] in (1024, 8192))
                 and all("term_file" not in x for x in rw if x["updates"] not in (1024, 8192)), "term files")
            need(all(v == 0.0 for x in rw for v in [n["diff_from_lived"] for n in x["norms"].values()]), "a lived tensor outside the byte table moved")
            need(all(x["tensors_sha256"] != life[(r_, s)]["tensors_sha256"] for x in rw[1:]), "tensors unchanged (code rows should move)")
            need((r_ == "S") == all(x["sweep_cosines"] is not None for x in rw), "sweep cosines")
            need(all("scores_gate_forced_off" not in x and "check2" not in x and "check4" not in x for x in rw), "fields E48's check2/check4 or forced-off scores")
            need(R_["gate_calls"]["gate_on"] < R_["gate_calls"]["calls"], "gate calls")
        else:
            need(all("weights" not in x for x in rw[:-1]) and ((rw[-1].get("weights") == f"e58_{a}_{r_}_f{ft}_{s}_row8192.pt") == (a != "free-again")), "weight files")
            need("gate_calls" not in R_, "gate in a free arm")
        nchecked += 1
    print("receipts read:", nchecked, "| departures from the brief's settings or from the identities above:", len(problems))
    for q in problems:
        print("   ", q)
    # identities between runs
    print("-- identities between runs")
    same_dig, diff_again, same_row0 = 0, 0, 0; n_dig = n_again = n_row0 = 0
    for s in SEEDS:
        for ft in FT:
            arms = PART_A if ft == "10" else PART_B
            ref = [x["data_digest"]["sha256"] for x in rec("free", "S", ft, s)["rows"]]
            for a in arms:
                for r_ in RULES:
                    d = [x["data_digest"]["sha256"] for x in rec(a, r_, ft, s)["rows"]]
                    if a == "free-again":
                        n_again += 1; diff_again += all(u != v for u, v in zip(d[1:], ref[1:])) and d[0] == ref[0]
                    else:
                        n_dig += 1; same_dig += d == ref
            for r_ in RULES:
                s0 = rec("free", r_, ft, s)["rows"][0]["scores"]
                for a in arms:
                    n_row0 += 1; same_row0 += rec(a, r_, ft, s)["rows"][0]["scores"] == s0
    print(f"batch digests at every row equal to free's of the same seed and fraction (both rules): {same_dig} of {n_dig} runs")
    print(f"free-again's digests differ from free's at every row after 0: {diff_again} of {n_again} runs")
    print(f"row 0 scores equal to free's of the same seed, rule and fraction: {same_row0} of {n_row0} runs")
    # the restarted runs
    print("-- the restarted runs")
    for p in [q for q in allf if "/interrupted/" in q and ".json." in q]:
        I = json.loads(show(p)); C = rec(I["arm"], I["rule"], {0.05: "05", 0.1: "10", 0.2: "20"}[I["fraction"]], I["seed"])
        cr = rows(C)
        same = all(x["scores"] == cr[x["updates"]]["scores"] and x["tensors_sha256"] == cr[x["updates"]]["tensors_sha256"]
                   and x["data_digest"] == cr[x["updates"]]["data_digest"] and x["common"] == cr[x["updates"]]["common"] for x in I["rows"])
        print(f"  {p.split('/')[-1]}: complete {I['complete']}, last row {I['rows'][-1]['updates']}, rows {len(I['rows'])}; completed receipt: "
              f"complete {C['complete']}, {len(C['rows'])} rows, cpu at row 0 {C['rows'][0]['cpu']:.0f} s, cpu_total {C['cpu_total']:.0f} s; "
              f"every shared row equal (scores, common, tensors, digest): {same}")
    pI, sI = load_pt(E58 + "interrupted/e58_term-wd0_S_f10_73129_row1024.pt.20261118T001933")[:2]
    pC = load_pt(E58 + "e58_term-wd0_S_f10_73129_row1024.pt")[0]
    print("  interrupted term file of term-wd0 S 73129 at row 1,024 against the completed run's: tensors equal",
          all(np.array_equal(pI[k], pC[k]) for k in pC) and set(pI) == set(pC))
    rr = {"free-again_S_f10_73129": 1024, "free-again_B_f10_73129": 256, "term-wd0_S_f10_73129": 7168, "free_S_f05_93683": 3072,
          "free_B_f05_93683": 4096, "free-wd0_S_f05_93683": 3072}
    for k, v in rr.items():
        p = [q for q in allf if f"/interrupted/e58_{k}.json." in q][0]
        cmp("2", f"row reached by the interrupted run {k}", v, json.loads(show(p))["rows"][-1]["updates"], 0)
    return life


# ================================================================ 3  figures
def figures(r):
    rw = rows(r)
    hi = max((u for u in rw if u >= 64), key=lambda u: red(rw[u]))
    return dict(rmid=mean([red(rw[u]) for u in MID]), rlate=mean([red(rw[u]) for u in LATE]), hi=red(rw[hi]), hi_row=hi,
                d84=red(rw[8192]) - red(rw[4096]), clate=mean([com(rw[u]) for u in LATE]), taught=rw[8192]["scores"]["summary"]["taught_add_sub_mean"],
                r1024=red(rw[1024]), r8192=red(rw[8192]))


def F(a, r_, ft, s, k="rlate"):
    return figures(rec(a, r_, ft, s))[k]


def verdict(x, y):
    d = [p - q for p, q in zip(x, y)]
    if sum(min(p, q) > 95 for p, q in zip(x, y)) >= 2:
        return "at the ceiling", d
    up = sum(v >= 5 for v in d); down = sum(v <= -5 for v in d)
    if up >= 2 and down == 0:
        return "above", d
    if down >= 2 and up == 0:
        return "below", d
    return "not distinguished", d


def V(a, b, r_, ft="10", k="rlate", fa=None, fb=None, ra=None, rb=None):
    x = [F(a, ra or r_, fa or ft, s, k) for s in SEEDS]; y = [F(b, rb or r_, fb or ft, s, k) for s in SEEDS]
    v, d = verdict(x, y)
    return v, d, x, y


def near(d, edges=(5.0, 2.0, -5.0, -2.0), eps=0.15):
    return [round(v, 3) for v in d if any(abs(v - e) < eps for e in edges)]


def section3(report):
    print("\n================ 3  figures of the report recomputed from the receipts")
    rep = report.replace("−", "-")
    # ---- 4.1 Part A table
    print("-- Part A: R-late (R-mid), highest row from 64 (row), row 8192 - row 4096, R-late common, taught at 8192")
    tab = re.findall(r"^\| ([SB]) \| ([a-z0-9-]+) \| ([\d.]+) \(([\d.]+)\) \| ([\d.]+) \(([\d.]+)\) \| ([\d.]+) \(([\d.]+)\) \|$", rep, re.M)
    for r_, a, *v in tab:
        for i, s in enumerate(SEEDS):
            g = figures(rec(a, r_, "10", s))
            cmp("4.1", f"{r_} {a} {s} R-late", float(v[2 * i]), g["rlate"]); cmp("4.1", f"{r_} {a} {s} R-mid", float(v[2 * i + 1]), g["rmid"])
    print("rows of the Part A table read from the report:", len(tab))
    for r_ in RULES:
        for a in PART_A:
            for s in SEEDS:
                g = figures(rec(a, r_, "10", s))
                print(f"  {r_} {a:11s} {s:6d} R-late {g['rlate']:7.3f} R-mid {g['rmid']:7.3f} highest {g['hi']:5.1f} ({g['hi_row']}) "
                      f"8192-4096 {g['d84']:+5.1f} common R-late {g['clate']:6.2f} taught@8192 {g['taught']:5.1f}")
    # ---- 4.2 N
    print("-- the scatter of one arm: N = |R-late(free-again) - R-late(free)|, and the same on the common pairs")
    rN = {"S": (4.2, 2.6, 0.0), "B": (0.4, 1.2, 2.6)}
    for r_ in RULES:
        n = [abs(F("free-again", r_, "10", s) - F("free", r_, "10", s)) for s in SEEDS]
        nc = [abs(F("free-again", r_, "10", s, "clate") - F("free", r_, "10", s, "clate")) for s in SEEDS]
        nm = [abs(F("free-again", r_, "10", s, "rmid") - F("free", r_, "10", s, "rmid")) for s in SEEDS]
        print(f"  {r_}: N {', '.join(f'{v:.3f}' for v in n)} | on the common pairs {', '.join(f'{v:.3f}' for v in nc)} | on R-mid "
              f"{', '.join(f'{v:.2f}' for v in nm)}")
        for i, s in enumerate(SEEDS):
            cmp("4.2", f"N {r_} {s}", rN[r_][i], n[i])
    # ---- 4.3 point 1
    print("-- point 1: term against free; leads on R-late and R-mid; movements between R-mid and R-late")
    p1 = {"S": ((13.0, 9.3, 3.7), (7.8, 6.4, 3.7)), "B": ((11.3, 6.6, 9.8), (8.6, 6.1, 7.9))}
    for r_ in RULES:
        v, d, x, y = V("term", "free", r_)
        vm, dm, _, _ = V("term", "free", r_, k="rmid")
        print(f"  {r_}: term against free {v}; R-late lead {', '.join(f'{u:+.3f}' for u in d)}; R-mid lead {', '.join(f'{u:+.3f}' for u in dm)}")
        cmp("4.3", f"point 1 {r_} verdict", "above", v)
        for i, s in enumerate(SEEDS):
            cmp("4.3", f"point 1 {r_} lead on R-late {s}", p1[r_][0][i], d[i]); cmp("4.3", f"point 1 {r_} lead on R-mid {s}", p1[r_][1][i], dm[i])
            t = (F("term", r_, "10", s, "rmid"), F("term", r_, "10", s)); f_ = (F("free", r_, "10", s, "rmid"), F("free", r_, "10", s))
            print(f"     {s}: term {t[0]:.3f} -> {t[1]:.3f} ({t[1] - t[0]:+.3f}); free {f_[0]:.3f} -> {f_[1]:.3f} ({f_[1] - f_[0]:+.3f}); "
                  f"lead R-late minus lead R-mid {d[i] - dm[i]:+.3f}")
        if r_ == "B":
            for i, (u, w) in enumerate(((91.9, 94.2), (97.4, 98.5), (98.6, 99.0))):
                cmp("4.3", f"point 1 B term R-mid {SEEDS[i]}", u, F("term", "B", "10", SEEDS[i], "rmid"))
                cmp("4.3", f"point 1 B term R-late {SEEDS[i]}", w, F("term", "B", "10", SEEDS[i]))
    dd = (F("term", "S", "10", 116089) - F("free", "S", "10", 116089)) - (F("term", "S", "10", 116089, "rmid") - F("free", "S", "10", 116089, "rmid"))
    cmp("4.3", "point 1 S 116089: lead on R-late against lead on R-mid", "equal", "equal" if abs(dd) < 1e-9 else ("larger" if dd > 0 else "smaller") + f" by {abs(dd):.3f}")
    cmp("4.3", "point 1 S term fall on 116089", -0.5, F("term", "S", "10", 116089) - F("term", "S", "10", 116089, "rmid"))
    # ---- point 2
    print("-- point 2: L, A, P and a = A / L")
    sp = re.findall(r"^\| ([SB]) \| (\d+) \| ([+-][\d.]+) \| ([+-][\d.]+) \| ([+-][\d.]+) \| ([\d.]+|\(L < 5\)) \|$", rep, re.M)
    for r_, s, L, A, P, a in sp:
        s = int(s); t, fr, an = F("term", r_, "10", s), F("free", r_, "10", s), F("free-anchor", r_, "10", s)
        print(f"  {r_} {s}: L {t - fr:+.3f} A {an - fr:+.3f} P {t - an:+.3f} a {((an - fr) / (t - fr)):.3f}")
        cmp("4.3", f"point 2 {r_} {s} L", fnum(L), t - fr); cmp("4.3", f"point 2 {r_} {s} A", fnum(A), an - fr); cmp("4.3", f"point 2 {r_} {s} P", fnum(P), t - an)
        if a.startswith("("):
            cmp("4.3", f"point 2 {r_} {s} L below 5", "yes", "yes" if t - fr < 5 else "no")
        else:
            cmp("4.3", f"point 2 {r_} {s} a", float(a), (an - fr) / (t - fr), 2)
    for r_ in RULES:
        aa = [(F("free-anchor", r_, "10", s) - F("free", r_, "10", s)) / (F("term", r_, "10", s) - F("free", r_, "10", s))
              for s in SEEDS if F("term", r_, "10", s) - F("free", r_, "10", s) >= 5]
        cls = ["decay" if v >= 0.75 else "place" if v <= 0.25 else "both" for v in aa]
        out = cls[0] if len(set(cls)) == 1 and len(aa) >= 2 else "the seeds disagree"
        print(f"  {r_}: shares on the seeds with L >= 5: {[round(v, 3) for v in aa]} -> {out}")
        cmp("4.3", f"point 2 {r_} reading", "the seeds disagree", out)
    # ---- point 3, 4
    print("-- points 3 and 4: verdicts with the unrounded differences (values within 0.15 of a threshold listed)")
    pr = {("term-wd0", "term"): {"S": ("not distinguished", (-0.4, -1.7, -0.3)), "B": ("at the ceiling", (0.4, -0.9, -0.3))},
          ("free-wd0", "free-anchor"): {"S": ("not distinguished", (-7.9, -1.5, -2.3)), "B": ("not distinguished", (-1.9, -3.9, -0.9))},
          ("term-wd0", "free-wd0"): {"S": ("above", (14.8, 8.2, 3.4)), "B": ("above", (11.1, 6.0, 5.1))},
          ("term", "free-anchor"): {"S": ("above", (7.3, 8.4, 1.4)), "B": ("not distinguished", (8.8, 3.0, 4.4))},
          ("free-anchor", "free"): {"S": ("not distinguished", (5.7, 0.9, 2.2)), "B": ("not distinguished", (2.4, 3.6, 5.4))},
          ("free-wd0", "free"): {"S": ("not distinguished", (-2.2, -0.5, -0.0)), "B": ("not distinguished", (0.5, -0.3, 4.5))}}
    for (a, b), byr in pr.items():
        for r_ in RULES:
            v, d, x, y = V(a, b, r_)
            print(f"  {r_} {a} against {b}: {v}; {', '.join(f'{u:+.3f}' for u in d)}; near a threshold {near(d)}")
            cmp("4.3", f"{r_} {a} against {b}: verdict", byr[r_][0], v)
            if not (a == "term-wd0" and b == "term" and r_ == "B"):
                for i, s in enumerate(SEEDS):
                    cmp("4.3", f"{r_} {a} against {b} {s}", byr[r_][1][i], d[i])
    for i, (u, w) in enumerate(((94.6, 94.2), (97.6, 98.5), (98.7, 99.0))):
        s = SEEDS[i]
        cmp("4.3", f"B ceiling term-wd0 {s}", u, F("term-wd0", "B", "10", s)); cmp("4.3", f"B ceiling term {s}", w, F("term", "B", "10", s))
        cmp("4.3", f"B ceiling 100 - term-wd0 {s}", round(100 - u, 1), 100 - F("term-wd0", "B", "10", s))
        cmp("4.3", f"B ceiling 100 - term {s}", round(100 - w, 1), 100 - F("term", "B", "10", s))
    # ---- point 5
    print("-- point 5: the life's four panels, the first family's two, the norms (free arms of Part A)")
    for r_, v0, v16 in (("S", 86.6, 60.9), ("B", 90.5, 56.6)):
        rw = rows(rec("free", r_, "10", 73129))
        n0 = sum(v["norm"] ** 2 for v in rw[0]["norms"].values()) ** .5; n16 = sum(v["norm"] ** 2 for v in rw[16]["norms"].values()) ** .5
        cmp("4.3", f"point 5 {r_} 73129 life's four at row 0", v0, life4(rw[0])); cmp("4.3", f"point 5 {r_} 73129 free life's four at row 16", v16, life4(rw[16]))
        print(f"  {r_} 73129: row 0 {life4(rw[0]):.3f}, free row 16 {life4(rw[16]):.3f}, norm ratio at row 16 {n16 / n0:.5f}; "
              f"free-wd0 row 16 {life4(rows(rec('free-wd0', r_, '10', 73129))[16]):.2f}; free-anchor row 16 {life4(rows(rec('free-anchor', r_, '10', 73129))[16]):.2f}; "
              f"first family's two at row 0 {first2(rw[0]):.2f}")
        cmp("4.3", f"point 5 {r_} 73129 norm at row 16 within 0.2% of row 0", "yes", "yes" if abs(n16 / n0 - 1) < 0.002 else "no")
    t5 = re.findall(r"^  \| ([SB]) \| ([\d.]+), ([\d.]+), ([\d.]+) \| ([\d.]+), ([\d.]+), ([\d.]+) \| ([\d.]+), ([\d.]+), ([\d.]+) \|$", rep, re.M)
    for r_, *v in t5:
        for j, a in enumerate(("free", "free-wd0", "free-anchor")):
            for i, s in enumerate(SEEDS):
                cmp("4.3", f"point 5 {r_} {a} {s} life's four at row 1024", float(v[3 * j + i]), life4(rows(rec(a, r_, "10", s))[1024]))
    print("  rows of the row-1,024 table read from the report:", len(t5))
    for r_ in RULES:
        print(f"  {r_} life's four at row 1,024 (free, free-wd0, free-anchor by seed):",
              [[round(life4(rows(rec(a, r_, "10", s))[1024]), 2) for s in SEEDS] for a in ("free", "free-wd0", "free-anchor")])
    nt = re.findall(r"^  \| `(free[a-z0-9-]*)` \| ([\d.]+) to ([\d.]+) \| ([\d.]+) to ([\d.]+) \|$", rep, re.M)
    def tot(x, k):
        return sum(v[k] ** 2 for v in x["norms"].values()) ** .5
    for a, s1, s2, b1, b2 in nt:
        for r_, lo, hi in (("S", s1, s2), ("B", b1, b2)):
            v = [tot(rows(rec(a, r_, "10", s))[8192], "norm") / tot(rows(rec(a, r_, "10", s))[0], "norm") for s in SEEDS]
            cmp("4.3", f"norm ratio at 8192 {a} {r_} lowest", float(lo), min(v), 2); cmp("4.3", f"norm ratio at 8192 {a} {r_} highest", float(hi), max(v), 2)
            print(f"  norm at 8192 relative to the lived norm, {a} {r_}: {[round(u, 4) for u in v]}")
    for a, lo, hi in (("free", 0.61, 0.76), ("free-wd0", 0.82, 0.96), ("free-anchor", 0.36, 0.52)):
        v = [tot(rows(rec(a, r_, "10", s))[8192], "diff_from_lived") / tot(rows(rec(a, r_, "10", s))[0], "norm") for r_ in RULES for s in SEEDS]
        print(f"  change at 8192 relative to the lived norm, {a}: {[round(u, 4) for u in v]}")
        cmp("4.3", f"change at 8192 {a} lowest", lo, min(v), 2); cmp("4.3", f"change at 8192 {a} highest", hi, max(v), 2)
    keep = {}
    for r_ in RULES:
        for s in SEEDS:
            v = []
            for a in ("free", "free-wd0", "free-anchor"):
                rw = rows(rec(a, r_, "10", s))
                v.append((mean([life4(rw[u]) for u in LATE]), F(a, r_, "10", s), a))
            keep[(r_, s)] = v
            top = max(v)
            print(f"  {r_} {s}: life's four over the late rows / R-late: " + "; ".join(f"{a} {l:.2f} / {rl:.2f}" for l, rl, a in v)
                  + f" | keeps most: {top[2]}, highest R-late: {max(v, key=lambda q: q[1])[2]}")
    an = [v[2][0] for v in keep.values()]; fr = [v[0][0] for v in keep.values()]
    cmp("4.3", "point 5 free-anchor keeps most, lowest", 42.4, min(an)); cmp("4.3", "point 5 free-anchor keeps most, highest", 52.5, max(an))
    cmp("4.3", "point 5 free, lowest", 22.2, min(fr)); cmp("4.3", "point 5 free, highest", 41.9, max(fr))
    cmp("4.3", "point 5 free-anchor keeps most on all six", "yes", "yes" if all(max(v)[2] == "free-anchor" for v in keep.values()) else "no")
    cmp("4.3", "point 5 free-anchor has the highest R-late on all six", "yes", "yes" if all(max(v, key=lambda q: q[1])[2] == "free-anchor" for v in keep.values()) else "no")
    # ---- Part B
    print("-- Part B: R-late on the common pairs")
    t51 = re.findall(r"^\| ([SB]) \| ([a-z0-9-]+) \| (0\.\d\d) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$", rep, re.M)
    for r_, a, f, *v in t51:
        ft = f[2:]
        for i, s in enumerate(SEEDS):
            cmp("5.1", f"{r_} {a} f{ft} {s} common R-late", float(v[i]), F(a, r_, ft, s, "clate"))
    print("  rows of the common-pairs table read from the report:", len(t51))
    for r_ in RULES:
        for ft in FT:
            for a in PART_B:
                print(f"  {r_} f{ft} {a:9s} R-late {', '.join(f'{F(a, r_, ft, s):6.2f}' for s in SEEDS)} | common {', '.join(f'{F(a, r_, ft, s, 'clate'):6.2f}' for s in SEEDS)}"
                      f" | R-mid {', '.join(f'{F(a, r_, ft, s, 'rmid'):5.1f}' for s in SEEDS)} | taught@8192 {', '.join(f'{F(a, r_, ft, s, 'taught'):5.1f}' for s in SEEDS)}")
    tb = [F(a, r_, ft, s, "taught") for a in PART_B for r_ in RULES for ft in FT for s in SEEDS]
    cmp("5", "taught panel at 8192 in Part B, lowest", 95.8, min(tb)); cmp("5", "taught panel at 8192 in Part B, highest", 100.0, max(tb))
    tb2 = [F(a, r_, ft, s, "taught") for a in PART_B for r_ in RULES for ft in ("05", "20") for s in SEEDS]
    print(f"  taught panel at 8192: all Part B tables (f 0.05, 0.10, 0.20) {min(tb):.2f} to {max(tb):.2f}; the 36 runs of Part B proper {min(tb2):.2f} to {max(tb2):.2f}")
    fr05 = [(r_, s, F("free", r_, "05", s, "r1024"), F("free", r_, "05", s, "r8192"), F("free", r_, "05", s, "d84")) for r_ in RULES for s in SEEDS]
    print("  free at f 0.05 (rule, seed, row 1024, row 8192, 8192-4096):", [(a, b, round(c, 2), round(d, 2), round(e, 2)) for a, b, c, d, e in fr05])
    cmp("5.1", "free f0.05 row 1024 lowest", 70.9, min(v[2] for v in fr05)); cmp("5.1", "free f0.05 row 1024 highest", 91.1, max(v[2] for v in fr05))
    cmp("5.1", "free f0.05 row 8192 lowest", 36.6, min(v[3] for v in fr05)); cmp("5.1", "free f0.05 row 8192 highest", 78.4, max(v[3] for v in fr05))
    dS = [v[4] for v in fr05 if v[0] == "S"]; dB = [v[4] for v in fr05 if v[0] == "B"]
    cmp("5.1", "free f0.05 S 8192-4096 smallest fall", -14.5, max(dS)); cmp("5.1", "free f0.05 S 8192-4096 largest fall", -19.1, min(dS))
    cmp("5.1", "free f0.05 B 8192-4096 smallest fall", -4.2, max(dB)); cmp("5.1", "free f0.05 B 8192-4096 largest fall", -10.9, min(dB))
    w05 = [F("free-wd0", r_, "05", s, "r8192") for r_ in RULES for s in SEEDS]; t05 = [F("term", r_, "05", s, "r8192") for r_ in RULES for s in SEEDS]
    cmp("5.1", "free-wd0 f0.05 row 8192 lowest", 60.3, min(w05)); cmp("5.1", "free-wd0 f0.05 row 8192 highest", 89.5, max(w05))
    cmp("5.1", "term f0.05 row 8192 lowest", 68.7, min(t05)); cmp("5.1", "term f0.05 row 8192 highest", 92.5, max(t05))
    falls = {(r_, s): F("term", r_, "05", s, "r8192") - F("term", r_, "05", s, "r1024") for r_ in RULES for s in SEEDS}
    k = min(falls, key=falls.get)
    print("  term at f 0.05, row 8192 minus row 1024:", {f"{a} {b}": round(v, 2) for (a, b), v in falls.items()}, "largest fall:", k)
    cmp("5.1", "term f0.05 largest fall from 1024 to 8192: where", "S 116089", f"{k[0]} {k[1]}")
    cmp("5.1", "term f0.05 S 116089 row 1024", 81.6, F("term", "S", "05", 116089, "r1024")); cmp("5.1", "term f0.05 S 116089 row 8192", 68.7, F("term", "S", "05", 116089, "r8192"))
    print("  free-wd0 at f 0.05, 8192-4096:", [round(F("free-wd0", r_, "05", s, "d84"), 2) for r_ in RULES for s in SEEDS],
          "| term at f 0.05, 8192-4096:", [round(F("term", r_, "05", s, "d84"), 2) for r_ in RULES for s in SEEDS])
    # 5.2
    print("-- Part B: term against free and free-wd0 at each fraction (R-late of the reduction score)")
    t52 = re.findall(r"^\| ([SB]) \| (0\.\d\d) \| (.+?) \| (.+?) \|$", rep, re.M)
    for r_, f, c1, c2 in t52:
        ft = f[2:]
        for b, c in (("free", c1), ("free-wd0", c2)):
            v, d, x, y = V("term", b, r_, ft)
            rv = c.split(" (")[0].strip()
            cmp("5.2", f"{r_} f{ft} term against {b}: verdict", rv, v)
            nums = re.findall(r"[+-]?[\d.]+", c)
            if nums and rv != "at the ceiling":
                for i, s in enumerate(SEEDS):
                    cmp("5.2", f"{r_} f{ft} term against {b} {s}", fnum(nums[i]), d[i])
            print(f"  {r_} f{ft} term against {b}: {v}; {', '.join(f'{u:+.3f}' for u in d)}; lower figure above 95 on {sum(min(p, q) > 95 for p, q in zip(x, y))} seeds; near a threshold {near(d)}")
    for b, figs in (("free", ((97.7, 92.3), (99.5, 97.1), (99.8, 95.3))), ("free-wd0", ((97.7, 94.1), (99.5, 96.6), (99.8, 96.7)))):
        for i, (u, w) in enumerate(figs):
            s = SEEDS[i]
            cmp("5.2", f"B f0.20 ceiling: term {s}", u, F("term", "B", "20", s)); cmp("5.2", f"B f0.20 ceiling: {b} {s}", w, F(b, "B", "20", s))
    # 5.3
    print("-- Part B: comparisons between fractions on R-late of the common pairs")
    rep53 = {("S", "S"): {("05", "10"): ((36.5, 34.6, 26.4), (-6.3, 1.2, 9.6), "no"), ("10", "20"): ((12.0, 7.4, 9.4), (0.4, -0.2, 6.4), "no"),
                          ("05", "20"): ((48.5, 41.9, 35.7), (6.8, 10.6, 19.5), "no")},
             ("B", "B"): {("05", "10"): ((27.7, 21.6, 13.9), (-4.7, 0.1, -0.2), "sentence"), ("10", "20"): ((9.2, 5.4, 6.1), (-0.1, -1.4, -2.2), "ceiling"),
                          ("05", "20"): ((36.9, 26.9, 20.0), (5.6, 5.4, 2.9), "no")},
             ("S", "B"): {("05", "10"): (None, (-1.8, 7.9, 19.6), "no"), ("10", "20"): (None, (2.1, 2.5, 9.6), "no"), ("05", "20"): (None, (8.5, 13.3, 22.6), "no")}}
    for (ra, rb), d0 in rep53.items():
        for (f, g), (ra_rep, def_rep, out_rep) in d0.items():
            va, da, _, _ = V("free", "free", rb, k="clate", fa=g, fb=f)
            tf = [F("term", ra, f, s, "clate") for s in SEEDS]
            best = [max(F("free", rb, g, s, "clate"), F("free-wd0", rb, g, s, "clate")) for s in SEEDS]
            which = ["free" if F("free", rb, g, s, "clate") >= F("free-wd0", rb, g, s, "clate") else "free-wd0" for s in SEEDS]
            de = [b - t for t, b in zip(tf, best)]
            cb = sum(v <= 2 for v in de) >= 2 and all(v < 5 for v in de)
            ceil = sum(min(t, b) > 95 for t, b in zip(tf, best)) >= 2
            out = "no-a" if va != "above" else "ceiling" if ceil else "sentence" if cb else "no"
            print(f"  term {ra} at {f} against free/free-wd0 {rb} at {g}: (a) free {rb} {g} against {f}: {va} ({', '.join(f'{u:+.3f}' for u in da)}); "
                  f"deficit {', '.join(f'{u:+.3f}' for u in de)} (higher arm {which}); (b) {cb}; lower figure above 95 on "
                  f"{sum(min(t, b) > 95 for t, b in zip(tf, best))} seeds -> {out}; near a threshold {near(de)}")
            if ra == "S" and rb == "B":
                vs, ds, _, _ = V("free", "free", "S", k="clate", fa=g, fb=f)
                print(f"     (a) read under the sweep instead: {vs} ({', '.join(f'{u:+.3f}' for u in ds)})")
            if ra_rep:
                for i, s in enumerate(SEEDS):
                    cmp("5.3", f"{ra}/{rb} ({f},{g}) (a) {s}", ra_rep[i], da[i])
            cmp("5.3", f"{ra}/{rb} ({f},{g}) (a) verdict", "above", va)
            for i, s in enumerate(SEEDS):
                cmp("5.3", f"{ra}/{rb} ({f},{g}) deficit {s}", def_rep[i], de[i])
            cmp("5.3", f"{ra}/{rb} ({f},{g}) outcome", out_rep, out)
    # 5.4
    print("-- Part B: first row at which the reduction score is 80 or more, 90 or more; presentations of each taught pair")
    fr = {}
    for r_ in RULES:
        for ft in FT:
            for a in PART_B:
                for s in SEEDS:
                    rw = rec(a, r_, ft, s)["rows"]
                    f80 = next((x["updates"] for x in rw if red(x) >= 80), None); f90 = next((x["updates"] for x in rw if red(x) >= 90), None)
                    fr[(r_, ft, a, s)] = (f80, f90)
    for r_ in RULES:
        for ft in FT:
            print(f"  {r_} f{ft}: " + "; ".join(f"{a} " + ",".join(f"{fr[(r_, ft, a, s)][0]}/{fr[(r_, ft, a, s)][1]}" for s in SEEDS) for a in PART_B))
    def rng_(r_, arms):
        v = [fr[(r_, ft, a, s)][0] for ft in FT for a in arms for s in SEEDS]
        return min(x for x in v if x is not None), max(x for x in v if x is not None), [k for k in fr if k[0] == r_ and k[2] in arms and fr[k][0] is None]
    lo, hi, never = rng_("S", ("term",)); cmp("5.4", "S term first 80, earliest", 256, lo, 0); cmp("5.4", "S term first 80, latest", 512, hi, 0)
    lo, hi, never = rng_("S", ("free", "free-wd0")); cmp("5.4", "S free arms first 80, earliest", 64, lo, 0); cmp("5.4", "S free arms first 80, latest", 512, hi, 0)
    cmp("5.4", "S free arms never at 80", "free f05 116089", "; ".join(f"{k[2]} f{k[1]} {k[3]}" for k in never))
    lo, hi, never = rng_("B", ("term",)); cmp("5.4", "B term first 80, earliest", 64, lo, 0); cmp("5.4", "B term first 80, latest", 256, hi, 0)
    lo, hi, never = rng_("B", ("free", "free-wd0")); cmp("5.4", "B free arms first 80, earliest", 32, lo, 0); cmp("5.4", "B free arms first 80, latest", 128, hi, 0)
    slower80 = {"no later": [], "later": []}; slower90 = {"term earlier": [], "term later or equal": [], "neither": []}
    for r_ in RULES:
        for ft in FT:
            for s in SEEDS:
                t80, t90 = fr[(r_, ft, "term", s)]
                for a in ("free", "free-wd0"):
                    a80, a90 = fr[(r_, ft, a, s)]
                    (slower80["later"] if (a80 is not None and t80 > a80) else slower80["no later"]).append(f"{r_} f{ft} {s} vs {a}: term {t80}, {a} {a80}")
                    if t90 is None and a90 is None:
                        slower90["neither"].append(f"{r_} f{ft} {s} vs {a}")
                    elif a90 is None or (t90 is not None and t90 < a90):
                        slower90["term earlier"].append(f"{r_} f{ft} {s} vs {a}: term {t90}, {a} {a90}")
                    else:
                        slower90["term later or equal"].append(f"{r_} f{ft} {s} vs {a}: term {t90}, {a} {a90}")
    print("  level 80: cases (rule, fraction, seed, free arm) in which term is NOT later than the free arm:", slower80["no later"])
    print(f"  level 80: term later than the free arm in {len(slower80['later'])} of 36 cases")
    print("  level 90: term reaches it earlier than the free arm (or the free arm never does):", len(slower90["term earlier"]), slower90["term earlier"])
    print("  level 90: term later or equal:", len(slower90["term later or equal"]), "| neither reaches 90:", len(slower90["neither"]))
    cmp("5.4", "term slower in presented items at every fraction (level 80, against free and free-wd0)", "yes", "yes" if not slower80["no later"] else "no")
    # presentations each at row 8192
    for ft in FT:
        print(f"  presentations of each taught pair at row 8192, f 0.{ft}: {8192 * 64 / (4 * NT[ft]):.2f}")
    # ---- predictions
    print("-- predictions")
    d7 = {r_: [life4(rows(rec("free-anchor", r_, "10", s))[1024]) - life4(rows(rec("free", r_, "10", s))[1024]) for s in SEEDS] for r_ in RULES}
    for r_, rv in (("S", (2.2, 5.6, 3.5)), ("B", (-0.8, 1.2, 0.4))):
        for i, s in enumerate(SEEDS):
            cmp("6", f"P58.7 {r_} {s}", rv[i], d7[r_][i])
    print("  P58.7 differences (free-anchor minus free, life's four at row 1024):", {r_: [round(v, 3) for v in d7[r_]] for r_ in RULES})
    print("  P58.4 free-wd0 against free:", [V("free-wd0", "free", r_)[0] for r_ in RULES], "| P58.6 free-anchor against free:", [V("free-anchor", "free", r_)[0] for r_ in RULES])
    print("  P58.5 term-wd0 against term:", [V("term-wd0", "term", r_)[0] for r_ in RULES])
    print("  P58.8 term against free at f 0.05:", [V("term", "free", r_, "05")[0] for r_ in RULES])
    print("  P58.10 term against free at f 0.20 under B:", V("term", "free", "B", "20")[0])
    # ---- cost
    print("-- cost (check 7 and the total)")
    c7 = json.loads(show(E58 + "qual/E58_check7.json"))["result"]
    tot = [c7["runs"][k]["cpu_total"] for k in c7["runs"]]; pu = [c7["runs"][k]["cpu_seconds_an_update"] for k in c7["runs"]]
    pr_ = [c7["runs"][k]["cpu_seconds_a_scored_row"] for k in c7["runs"]]
    own = {f"{a}_{r_}": rec(a, r_, "10", 73129) for a in PART_A for r_ in RULES}
    same = all(abs(c7["runs"][k]["cpu_total"] - own[k]["cpu_total"]) < 1e-6 for k in own)
    print(f"  check 7's twelve runs equal the receipts' cpu_total: {same}; cpu_total {min(tot):.1f} to {max(tot):.1f}; per update {min(pu):.4f} to {max(pu):.4f}; per row {min(pr_):.2f} to {max(pr_):.2f}")
    cmp("3", "check 7 run cost lowest, CPU-s", 881, min(tot), 0); cmp("3", "check 7 run cost highest, CPU-s", 1465, max(tot), 0)
    cmp("3", "check 7 per update lowest", 0.077, min(pu), 3); cmp("3", "check 7 per update highest", 0.137, max(pu), 3)
    cmp("3", "check 7 per row lowest", 15.6, min(pr_)); cmp("3", "check 7 per row highest", 22.7, max(pr_))
    fc = c7["forecast"]
    part_a = 3 * sum(own[k]["cpu_total"] for k in own); part_b = 6 * sum(own[f"{a}_{r_}"]["cpu_total"] for a in PART_B for r_ in RULES)
    print(f"  forecast recomputed: Part A {part_a / 3600:.2f} + Part B {part_b / 3600:.2f} = {(part_a + part_b) / 3600:.2f} CPU-hours; receipt {fc['total_cpu_hours']:.2f}")
    cmp("3", "check 7 forecast, CPU-hours", 24.1, (part_a + part_b) / 3600)
    allc = sum(rec(*k)["cpu_total"] for k in runs_all())
    cmp("3", "CPU-hours of the 72 completed runs", 25.4, allc / 3600)
    print(f"  the 72 completed runs: {allc / 3600:.2f} CPU-hours; Part A {sum(rec(*k)['cpu_total'] for k in runs_all() if k[2] == '10') / 3600:.2f}, "
          f"Part B {sum(rec(*k)['cpu_total'] for k in runs_all() if k[2] != '10') / 3600:.2f}")
    return fr


# ================================================================ 4  the checks
def section4(life):
    print("\n================ 4  the eight checks")
    Q = E58 + "qual/"
    c = {n: json.loads(show(f"{Q}E58_{n}.json")) for n in ("check1", "check2", "check3_S", "check3_B", "check4_S", "check4_B", "check5", "check6",
                                                             "check6_part_a", "check6_seed73129_only", "check7", "check8_S", "check8_B")}
    for n, v in c.items():
        print(f"  {n}: ok {v['result'].get('ok')}; command {v['command'].split(' ', 1)[1]}; cpu {v['cpu_seconds']:.1f} s; runner hash "
              f"{v['provenance']['e58_sources_sha256']['exp58_new_family.py'][:12]}")
    # check 1: against E40's receipts, all three seeds
    print("-- check 1 recomputed: free at f 0.10 against E40's receipts at E40's rows (digest, every count, tensors), every seed and rule")
    for r_ in RULES:
        for s in SEEDS:
            w = {x["updates"]: x for x in json.loads(show(f"{M}exp45/e40/e40_lived_{r_}_{s}.json", DREF))["rows"]}
            mine = rows(rec("free", r_, "10", s))
            ok = {u: (mine[u]["data_digest"] == w[u]["data_digest"], mine[u]["scores"] == w[u]["scores"], mine[u]["tensors_sha256"] == w[u]["tensors_sha256"])
                  for u in (0, 64, 128, 256, 512, 1024, 2048, 4096, 8192)}
            print(f"  {r_} {s}: all equal at rows 0..8192: {all(all(v) for v in ok.values())}; rows {sorted(ok)}")
            if s == 73129:
                cmp("3", f"check 1 {r_} reduction score at 8192 (E40 = E58)", round(w[8192]["scores"]["summary"]["reduction_score"], 1), red(mine[8192]))
    print("  check 1's receipt compares rows:", c["check1"]["result"]["rows_compared"])
    # check 2: against E48's receipts, all three seeds
    print("-- check 2 recomputed: term at f 0.10 against E48's rank-16 receipts at rows 0..1024 (every count, digest, earlier answers), every seed and rule")
    for r_ in RULES:
        for s in SEEDS:
            w = {x["updates"]: x for x in json.loads(show(f"{M}exp45/e48/e48_{r_}_{s}_r16.json", W8REF))["rows"]}
            mine = rows(rec("term", r_, "10", s))
            ok = all(mine[u]["scores"] == w[u]["scores"] and mine[u]["data_digest"] == w[u]["data_digest"]
                     and mine[u]["earlier_answers_sha256"] == w[u]["earlier_answers_sha256"] for u in (0, 16, 32, 64, 128, 256, 512, 1024))
            a, _, _ = load_pt(f"{E58}e58_term_{r_}_f10_{s}_row1024.pt"); b, _, _ = load_pt(f"{M}exp45/e48/e48_{r_}_{s}_r16_row1024.pt", W8REF)
            tf = set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)
            print(f"  {r_} {s}: rows equal {ok}; term file at row 1,024 bit-equal to E48's: {tf}")
    # check 3, 4, 8 from the receipts; the reference figures recomputed from the lived weights
    print("-- checks 3, 4, 8: figures in the receipts; the deviation the standing decay would cause, recomputed from the lived weights")
    for r_ in RULES:
        W, _, _ = load_pt(f"{M}exp45/e26/e26_life_{r_}_73129.pt", DREF)
        per = {k: float(np.abs(0.00025 * W[k].astype(np.float64)).max()) for k in W if k != "book.weight"}
        r3 = c[f"check3_{r_}"]["result"]; r4 = c[f"check4_{r_}"]["result"]; r8 = c[f"check8_{r_}"]["result"]
        same3 = all(abs(per[k] - r3["tensors"][k]["deviation_if_no_decay"]) < 1e-12 for k in per)
        same4 = all(abs(per[k] - r4["a"]["tensors"][k]["deviation_of_the_standing_decay"]) < 1e-12 for k in per)
        print(f"  {r_}: check 3 worst {r3['worst']:.3e}, smallest deviation without decay {r3['smallest_deviation_if_no_decay']:.3e} (recomputed {min(per.values()):.3e}; "
              f"every tensor equal {same3}); all tensors changed by the update {all(v['changed_by_the_update'] for v in r3['tensors'].values())}; byte table bit-equal {r3['byte_table_bit_equal']}, "
              f"changed by the update {r3['byte_table_changed_by_the_update']}")
        print(f"     check 4 (a) worst {r4['a']['worst']:.3e} (standing-decay reference equal {same4}, smallest {min(per.values()):.3e}); byte table {r4['a']['byte_table_bit_equal']}; "
              f"(b) {r4['b']['all_unchanged']}; (c) {[g['weight_decay'] for g in r4['c']['optimizer']]}")
        print(f"     check 8: down of term-wd0 bit-equal to start {r8['down_wd0_bit_equal_to_start']}; down of term from 0.99975 x start "
              f"{max(r8['down_term_max_abs_deviation_from_0_99975_start'].values()):.2e}; first-update gradient of down {r8['first_update_gradients']['term']['down_i']}, "
              f"{r8['first_update_gradients']['term']['down_h']}; bit-equal in the two arms {r8['bit_equal_in_the_two_arms']}; decay {[g['weight_decay'] for g in r8['optimizer_term']]}/"
              f"{[g['weight_decay'] for g in r8['optimizer_term_wd0']]}")
        cmp("3", f"check 3 worst {r_}", {"S": 4.1e-8, "B": 7.7e-8}[r_] * 1e8, r3["worst"] * 1e8)
        cmp("3", f"check 3 without decay {r_} (two significant figures, x 1e-6 or 1e-5)", {"S": 6.4, "B": 2.0}[r_], r3["smallest_deviation_if_no_decay"] * {"S": 1e6, "B": 1e5}[r_])
        cmp("3", f"check 4 worst {r_} (x 1e-8)", {"S": 5.7, "B": 11.0}[r_], r4["a"]["worst"] * 1e8, {"S": 1, "B": 0}[r_])
        cmp("3", f"check 8 worst {r_} (x 1e-9)", 3.0, max(r8["down_term_max_abs_deviation_from_0_99975_start"].values()) * 1e9)
        # the term's down factors at the start: E48's Term(seed, 16) draws them; the term file at row 1024 of E48 is compared above
    # check 5
    print("-- check 5 recomputed: the fractions, the pairs, the streams and their digests")
    for s in SEEDS:
        for t in F4:
            a, b, cc = (nested(t, f, s)[0] for f in (0.05, 0.10, 0.20))
            assert a == b[:344] and b == cc[:689] and (len(a), len(b), len(cc)) == (344, 689, 1378) and b == split(t, 0.10, s)[0]
    print("  nested 344 within 689 within 1,378 for every seed and operation, and 689 = exp11_cell.split at 0.10: True")
    cp = {s: {t[0]: len(common_pairs(t, s)) for t in F4[:2]} for s in SEEDS}
    print("  common pairs (add, sub) by seed:", cp, "| receipt of check 5:", c["check5"]["result"]["common_pairs"])
    for s, (x, y) in zip(SEEDS, ((2491, 2671), (2556, 2560), (2718, 2629))):
        cmp("3", f"common pairs add {s}", x, cp[s]["add"], 0); cmp("3", f"common pairs sub {s}", y, cp[s]["sub"], 0)
    for s in SEEDS:
        print(f"  reduction pairs untaught (add, sub) at f 0.05/0.10/0.20, seed {s}:", [tuple(len(reduction_pairs(t, FT[ft], s)) for t in F4[:2]) for ft in FT])
    # the common pairs are reduction pairs untaught at every fraction
    for s in SEEDS:
        for t in F4[:2]:
            cpl = set(common_pairs(t, s))
            assert all(cpl <= set(reduction_pairs(t, FT[ft], s)) for ft in FT)
    print("  the common pairs are untaught reduction pairs at all three fractions: True")
    t0 = time.time(); bad = []
    c5 = c["check5"]["result"]["digests"]
    for s in SEEDS:
        for ft in FT:
            dig, unt, never, notaught = walk_stream(ft, s, 140)
            arms = PART_A if ft == "10" else PART_B
            eq_c5 = all(dig[u] == c5[ft][str(s)][str(u)] for u in ROWS)
            eq_runs = all(rows(rec(a, r_, ft, s))[u]["data_digest"]["sha256"] == dig[u] for a in arms if a != "free-again" for r_ in RULES for u in ROWS)
            msg = f"  f {ft} seed {s}: own digests equal check 5's {eq_c5}, equal every run's at every row {eq_runs}; untaught presented {unt}; taught never presented {never}"
            if ft == "10":
                eq40 = all(json.loads(show(f"{M}exp45/e40/e40_lived_{r_}_{s}.json", DREF))["rows"][i]["data_digest"]["sha256"] == dig[u]
                           for r_ in RULES for i, u in enumerate((0, 64, 128, 256, 512, 1024, 2048, 4096, 8192)))
                dga, unta, nevera, _ = walk_stream("10", s, 141)
                eqa = all(rows(rec("free-again", r_, "10", s))[u]["data_digest"]["sha256"] == dga[u] for r_ in RULES for u in ROWS)
                msg += f"; equal E40's at E40's rows {eq40}; free-again's stream (seed + 141): digests equal both receipts {eqa}, untaught {unta}, never presented {nevera}"
            print(msg, flush=True)
            if not (eq_c5 and eq_runs and unt == 0 and never == 0):
                bad.append((ft, s))
    print(f"  streams walked in {time.time() - t0:.0f} s; departures: {bad}")
    # check 6
    print("-- check 6 recomputed from the 24 receipts of term and term-wd0")
    n = 0; okall = True; dig_by = {}
    for a, r_, ft, s in [k for k in runs_all() if k[0].startswith("term")]:
        R_ = rec(a, r_, ft, s); rw = R_["rows"]; d0 = rw[0]["earlier_answers_sha256"]["all"]; c0 = rw[0]["check3"]["earlier_counts"]
        ok = all(x["earlier_answers_sha256"]["all"] == d0 and x["check3"]["earlier_digest_equals_row0"] and x["check3"]["earlier_counts"] == c0
                 and x["check3"]["lived_tensors_sha256_codes_reset"] == life[(r_, s)]["tensors_sha256"] and x["check3"]["equals_life_end"] for x in rw)
        okall &= ok; n += 1
        dig_by.setdefault((r_, s), set()).add(d0)
        e40c = {k: v["exact"] for k, v in json.loads(show(f"{M}exp45/e40/e40_lived_{r_}_{s}.json", DREF))["rows"][0]["scores"]["earlier_k12"].items()}
        okall &= c0 == e40c
    print(f"  runs {n}; at every row: earlier digest = row 0's, six counts = row 0's = E40's row 0, lived hash with codes reset = the life's receipt: {okall}")
    print("  one earlier-answers digest for all term runs of a seed and rule (fractions and arms):", all(len(v) == 1 for v in dig_by.values()))
    print("  runs in check 6's receipts: full", c["check6"]["result"]["n"], "| part A", c["check6_part_a"]["result"]["n"], "| seed 73129 only", c["check6_seed73129_only"]["result"]["n"])
    cmp("3", "check 6 runs", 24, c["check6"]["result"]["n"], 0); cmp("3", "check 6 part A runs", 12, c["check6_part_a"]["result"]["n"], 0)
    cmp("3", "check 6 seed 73129 runs", 4, c["check6_seed73129_only"]["result"]["n"], 0)


# ================================================================ 5  the weights
def section5(life):
    print("\n================ 5  the weights")
    lived = {}
    for r_ in RULES:
        for s in SEEDS:
            W, sha, _ = load_pt(f"{M}exp45/e26/e26_life_{r_}_{s}.pt", DREF)
            lived[(r_, s)] = W
            print(f"  lived {r_} {s}: file hash = life receipt {sha == life[(r_, s)]['weights_sha256']}; tensors hash = life receipt {tensors_sha(W) == life[(r_, s)]['tensors_sha256']}; "
                  f"shapes {sorted((k, v.shape) for k, v in W.items() if k in ('book.weight', 'l1.weight', 'rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'l3.weight'))}")
    used = sorted(set([101, 102, 103, 104] + list(range(48, 58)) + [43, 45, 60, 62, 61] + list(range(65, 91))))
    print("-- the 42 weight files of free, free-wd0, free-anchor (row 8,192)")
    worst = 0.0; rowsets = {}; nbad = []
    for a, r_, ft, s in [k for k in runs_all() if k[0] in ("free", "free-wd0", "free-anchor")]:
        p = f"{E58}e58_{a}_{r_}_f{ft}_{s}_row8192.pt"
        W, sha, _ = load_pt(p); x = rows(rec(a, r_, ft, s))[8192]; L = lived[(r_, s)]
        ok = sha == x["weights_sha256"] and tensors_sha(W) == x["tensors_sha256"]
        for k, v in x["norms"].items():
            nn_ = float(np.linalg.norm(W[k].astype(np.float64))); dd = float(np.linalg.norm(W[k].astype(np.float64) - L[k].astype(np.float64)))
            worst = max(worst, abs(nn_ - v["norm"]) / max(v["norm"], 1e-12), abs(dd - v["diff_from_lived"]) / max(v["diff_from_lived"], 1e-12))
        ch = tuple(np.nonzero((W["book.weight"] != L["book.weight"]).any(1))[0].tolist())
        rowsets.setdefault(ch, []).append(f"{a} {r_} f{ft} {s}")
        if not ok:
            nbad.append(p)
        if s == 73129 and ft == "10":
            n0 = np.sqrt(sum(float(np.sum(L[k].astype(np.float64) ** 2)) for k in L if k != "book.weight"))
            n8 = np.sqrt(sum(float(np.sum(W[k].astype(np.float64) ** 2)) for k in W if k != "book.weight"))
            d8 = np.sqrt(sum(float(np.sum((W[k].astype(np.float64) - L[k].astype(np.float64)) ** 2)) for k in W if k != "book.weight"))
            print(f"  {a} {r_} 73129 from the weights: norm {n8 / n0:.4f} of the lived norm, change {d8 / n0:.4f}")
    print(f"  file hash and tensors hash equal the receipts: {42 - len(nbad)} of 42 {nbad}")
    print(f"  largest relative difference between the norms in the receipts and the norms of the weight files: {worst:.2e}")
    for ch, who in rowsets.items():
        print(f"  rows of the byte table changed: {len(ch)} rows {list(ch)} in {len(who)} files; all within the bytes of the new family's inputs: {set(ch) <= set(used)}; "
              f"bytes of the inputs not changed: {sorted(set(used) - set(ch))}")
    print("-- the 48 term files")
    bad = []; ranks = set(); shapes = set(); code_changed = set(); worst_n = 0.0
    for a, r_, ft, s in [k for k in runs_all() if k[0].startswith("term")]:
        for u in (1024, 8192):
            p = f"{E58}e58_{a}_{r_}_f{ft}_{s}_row{u}.pt"
            Tm, sha, nb = load_pt(p); x = rows(rec(a, r_, ft, s))[u]
            ok = sha == x["term_file"]["sha256"] and tensors_sha(Tm) == x["term_file"]["tensors_sha256"] and nb == x["term_file"]["bytes"]
            ok &= tensors_sha({"code_rows": Tm["code_rows"]}) == x["code_rows_tensors_sha256"]
            for k in ("down_i", "up_i", "down_h", "up_h"):
                worst_n = max(worst_n, abs(float(np.linalg.norm(Tm[k].astype(np.float64))) - x["term_norms"][k]) / x["term_norms"][k])
            shapes.add(tuple(sorted((k, v.shape, str(v.dtype)) for k, v in Tm.items())))
            ranks.add((np.linalg.matrix_rank(Tm["up_i"].astype(np.float64) @ Tm["down_i"].astype(np.float64)),
                       np.linalg.matrix_rank(Tm["up_h"].astype(np.float64) @ Tm["down_h"].astype(np.float64))))
            code_changed.add(tuple(bool((Tm["code_rows"][i] != lived[(r_, s)]["book.weight"][101 + i]).any()) for i in range(4)))
            if not ok:
                bad.append(p)
    print(f"  file hash, tensors hash, bytes and code-row hash equal the receipts: {48 - len(bad)} of 48 {bad}")
    print(f"  names, shapes and types: {shapes}")
    print(f"  ranks of up_i down_i and of up_h down_h: {ranks}; code rows differ from the lived rows (e, f, g, h): {code_changed}")
    print(f"  largest relative difference of the term norms against the receipts: {worst_n:.2e}")
    print("  products as large as the matrices they are added to (seed 73129, row 8,192, f 0.10):")
    for a in ("term", "term-wd0"):
        for r_ in RULES:
            Tm = load_pt(f"{E58}e58_{a}_{r_}_f10_73129_row8192.pt")[0]; L = lived[(r_, 73129)]
            pi = np.linalg.norm(Tm["up_i"].astype(np.float64) @ Tm["down_i"].astype(np.float64)); ph = np.linalg.norm(Tm["up_h"].astype(np.float64) @ Tm["down_h"].astype(np.float64))
            print(f"    {a} {r_}: |up_i down_i| {pi:.3f} against |W_ih| {np.linalg.norm(L['rnn.weight_ih_l0']):.3f}; |up_h down_h| {ph:.3f} against |W_hh| {np.linalg.norm(L['rnn.weight_hh_l0']):.3f}")
    return lived


def section5b(lived):
    print("\n================ 5b  the forward recomputed with numpy (reduction panel and common pairs)")
    jobs = [("lived (row 0)", "free", "S", "10", 73129, 0), ("term", "term", "S", "10", 73129, 1024), ("term", "term", "S", "10", 73129, 8192),
            ("term", "term", "B", "10", 73129, 8192), ("free", "free", "S", "10", 73129, 8192), ("free", "free", "B", "10", 73129, 8192),
            ("free", "free", "S", "05", 73129, 8192), ("term", "term", "S", "05", 73129, 8192)]
    for label, a, r_, ft, s, u in jobs:
        t0 = time.time()
        L = lived[(r_, s)]
        if u == 0:
            W, term = L, None
        elif a.startswith("term"):
            W, term = L, load_pt(f"{E58}e58_{a}_{r_}_f{ft}_{s}_row{u}.pt")[0]
        else:
            W, term = load_pt(f"{E58}e58_{a}_{r_}_f{ft}_{s}_row{u}.pt")[0], None
        x = rows(rec(a, r_, ft, s))[u]
        out = []
        for dt in (np.float32, np.float64):
            net = Net(W, term, dt)
            got = {}
            for t in F4[:2]:
                items = [render(t, p, q, 0, Random(0)) for p, q in reduction_pairs(t, FT[ft], s)]
                citems = [render(t, p, q, 0, Random(0)) for p, q in common_pairs(t, s)]
                got[t[0]] = (exact(net, items), len(items), exact(net, citems), len(citems))
            out.append(got)
        want = {o: (x["scores"]["held_plain_k0"][f"{o}83"]["reduction"]["exact"], x["scores"]["held_plain_k0"][f"{o}83"]["reduction"]["n"],
                    x["common"]["counts"][f"{o}83"]["exact"], x["common"]["counts"][f"{o}83"]["n"]) for o in ("add", "sub")}
        sc = lambda g: (100 * g["add"][0] / g["add"][1] + 100 * g["sub"][0] / g["sub"][1]) / 2
        print(f"  {label}: {a} {r_} f{ft} {s} row {u}: receipt (exact, n, common exact, common n) {want}; float32 {out[0]}; float64 {out[1]}; "
              f"reduction score receipt {sc(want):.3f}, float32 {sc(out[0]):.3f}, float64 {sc(out[1]):.3f}; {time.time() - t0:.0f} s", flush=True)
        cmp("5", f"forward {a} {r_} f{ft} {s} row {u}: reduction score (float32)", round(sc(want), 3), sc(out[0]), 3)


# ================================================================ 6  the runner as text
def section6():
    print("\n================ 6  the runner and the imported sources as text")
    src = show(M + "exp58_new_family.py").decode().splitlines()
    keys = ["E40.split = lambda t, seed: nested(t, f, seed)", "taught, rest = E.split(t, 0.10, seed)", "k = round(f * len(pairs))",
            "wd = C.WD if arm in (\"free\", \"free-again\") else 0.0", "p.sub_(p - p0[n], alpha=ALPHA)", "G.freeze(m); opt = G.make_opt(m, term, wd=0.25 if arm == \"term\" else 0.0)",
            "st = X16.entry_stream(entries, seed + 141)", "a in first and b in second", "nested(t, 0.20, seed)[1] if E40.needs_reduction",
            "E.render(t, a, b, 0, Random(0))", "X48.install()", "arch26.install()", "anchor(m, p0)", "set_fraction(f)", "fraction_assertions(f, seed, entries, sc, pr)",
            "assert row[\"check3\"][\"earlier_digest_equals_row0\"]", "assert row[\"check3\"][\"equals_life_end\"]", "assert all(row[\"e40_compare\"].values())",
            "assert row[\"digest_equals_check5\"]", "s, ans = X48.score(sc, m)", "s = sc(m)", "cs = common_score(m, com)"]
    for k in keys:
        hits = [i + 1 for i, l in enumerate(src) if k in l]
        print(f"  line(s) {hits}: {k}")
    gate = show(M + "exp48_gated_run.py").decode()
    i = gate.index("def gate_of"); print("  the gate (exp48_gated_run.gate_of):", " | ".join(l.strip() for l in gate[i:i + 260].splitlines()[:4]))
    print("  order of building in run(): set_fraction before Scorer, entries, batches, Presented:",
          min(i for i, l in enumerate(src) if "set_fraction(f)" in l and "def" not in l) < min(i for i, l in enumerate(src) if "sc = E40.Scorer(seed)" in l))
    e40 = show(M + "e40_new_family.py").decode()
    print("  e40_new_family calls split by its module-level name in entries, held_plain, Scorer, Presented:",
          [ln.strip() for ln in e40.splitlines() if "split(t, seed)" in ln])


# ================================================================ 7  form
def section7(report, allf):
    print("\n================ 7  form")
    for pat in ("@@", "TODO", "TBD", "XX", "FIXME", "<seed>", "???"):
        print(f"  '{pat}' in the report: {report.count(pat)}")
    enc = ["pynhqr", "bchf", "fbaarg", "unvxh", "tcg", "pbqrk", "trzvav", "yynzn", "zvfgeny"]
    pats = [re.compile(r"\b" + codecs.decode(e, "rot13") + r"\b", re.I) for e in enc]
    hits = {}
    files = [p for p in allf if ".pt" not in p] + [M + s for s in ("E58_COPIED_SOURCES.txt", "e48_gated.py", "exp48_gated_run.py", "exp58_new_family.py")]
    for p in files:
        txt = show(p).decode("utf-8", "replace")
        for i, rx in enumerate(pats):
            for m_ in rx.finditer(txt):
                ctx = txt[max(0, m_.start() - 25):m_.end() + 25].replace("\n", " ")
                for j, rx2 in enumerate(pats):
                    ctx = rx2.sub(f"<name {j + 1}>", ctx)
                hits.setdefault(p[len(M):], []).append(ctx)
    print(f"  text files searched: {len(files)}; files with a name of a language model (names held encoded, shown as <name k>): {len(hits)}")
    for p, v in hits.items():
        print(f"    {p}: {len(v)} occurrence(s); first: ...{v[0]}...")


if __name__ == "__main__":
    report, allf = section1(); clock("1")
    life = section2(allf); clock("2")
    section3(report); clock("3")
    section4(life); clock("4")
    lived = section5(life); clock("5")
    section5b(lived); clock("5b")
    section6(); clock("6")
    section7(report, allf); clock("7")
    print("\n================ summary: reported against recomputed")
    diff = []
    for sec, label, rp, v, nd in COMP:
        ok = agrees(rp, v, nd)
        if not ok:
            diff.append((sec, label, rp, v, nd))
    print(f"figures compared: {len(COMP)}; agree at the report's precision: {len(COMP) - len(diff)}; differ: {len(diff)}")
    for sec, label, rp, v, nd in diff:
        if isinstance(v, str) or isinstance(rp, str):
            print(f"  [{sec}] {label}: reported {rp!r}, recomputed {v!r}")
        else:
            print(f"  [{sec}] {label}: reported {rp}, recomputed {v:.4f} (rounded {round(v, nd)}), difference {v - rp:+.4f}")
    print(f"\nrunning time: {time.time() - T0:.1f} s")
