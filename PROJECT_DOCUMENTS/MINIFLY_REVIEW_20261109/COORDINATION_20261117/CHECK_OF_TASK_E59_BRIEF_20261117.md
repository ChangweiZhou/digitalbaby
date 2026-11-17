# Check of the brief for E59 before its release (17 November 2026)

**State of this document: in progress (started 22:22 UTC). While this line stands, the check is not finished and the verdict is provisional.**

Checked: `TASK_E59_A_TASK_OF_ANOTHER_KIND_FROM_A_FRESH_A_BORN_AND_A_LIVED_NETWORK_20261117.md` (revision 1, commit 1ed3d41c), against the default branch `origin/codex/minifly-consolidated-20261023` at 7cabb13a. Read-only; nothing was trained.

## Working notes (to be replaced by the final text)

Provisional verdict: release after the listed changes.

Found so far (each recomputed; scripts in the scratch folder, to be appended):

1. `exp26_candidate.admit_e26` called directly FAILS on the default branch: the births' receipts hold `exp16_integrated.py` at f4559971..., the default branch has it at c9fcedcf.... E43 admits through `exp38_erase.admit`, which carries the one recorded exception. The brief names the wrong function twice (runner; check 3).
2. Taught panel: `e43_task.panel` cannot be used for n < 2,048 (it calls `split(seed)` itself), and sampling 300 of 256 raises ValueError. Construction that works for all n and is E43's panel at 2,048 (verified, three seeds): `rng = Random(f"e43|{seed}|panel|taught"); [render(a, b, 12, rng) for a, b in rng.sample(taught[:n], min(300, n))]`.
3. Chances: exact 1.56% (not 1.8%) for 256 and 0.018% (not 0.03%) for 512; the brief's figures are exp(-4) and exp(-8). Realised: at n = 256, symbols absent from the first/second place: 2/1 (73129), 1/1 (93683), 0/1 (116089); held-panel items with such a symbol: 11, 6, 2 of 300 (ceilings 289, 294, 298). At 512 none absent.
4. E40: lived 77.0 to 94.5 (brief: "77 to 95"; 94.48 rounds to 94), fresh 6.0 to 11.8. Cost 974 s is the largest of E40's twelve (851 to 974). E43's own runs: 0.100 to 0.110 CPU-s an update.
5. Rows: 89 confirmed; windows of "has flattened" are 8 rows each. "Three consecutive rows" spans 64 updates up to a first row of 960, 160 at 992, 256 from 1,024; the last row at which a crossing counts is 7,936. Fresh networks on a family task are below 10% until update 1,024 (E40, E41), so the fresh side is likely to cross in the coarse part and the born side in the fine part.
6. E42's `describe` is used with the run length reduced by two rows for the three-row criterion (`thr_desc`, lines 79-80); the brief says 8,192 on both sides; it should be 7,936. The two-of-three rule is undefined when a seed's word is "not decided" or "neither reaches"; "otherwise about the same" then mislabels unmeasured cases.
7. The byte table's rows 128 to 225 are bit-equal in fresh (read from E40's fresh runs), born S/B and lived S/B, for each seed. The other maps differ in size: l1.weight rms 0.20 fresh, 0.05 born; recurrent 0.038 against 0.059-0.071; l3.weight 0.029 against 0.061-0.080; l3.bias 0.029 against 0.24-0.42 (seed 73129).
8. "Has flattened" licenses the sentence on pairs when a fresh network is flat at chance without having learned the taught pairs. On E41's 23 fresh curves below 80 at 8,192 the rule calls 9 flattened, all with the taught panel at 96.8% or more; one rose by 5 points or more later, none by 10.
9. `.gitignore` at the root of the default branch excludes `*.pt`: final weights need `git add -f`. `refuse_existing` refuses a restart while a partial receipt exists.
10. B2 compares the sweep with back-propagation; the brief fixes no statement of the two rules against each other except P59.6. Nearest record for P59.6 (E41, fresh, half of the pairs, present width): ratio sweep/back-propagation 1.10, 1.19, 1.24.
11. Check 1 holds in principle: nothing between updates in E43's runner touches the optimizer, the model or a generator used in training; training draws no random numbers. E51's check 6 reproduced E43's row 64 under both rules in another container.

12. The born and lived networks give no answer of the task's form before any update (0 to 36 of 300 held items have three bytes with the end symbol third); E43's networks after 64 updates give the form on 300 and 283 of 300 and the first symbol on 3 and 15. Counts by place at every row would show where a fresh network loses its time.
13. The two-of-three rule can give "born fewer than fresh", "lived fewer than born", "lived about the same as fresh" (example in the script, part votes).
14. Reference batch digests in the receipts' own form were computed in numpy and reproduce E43's at five rows on three seeds; values for 512 and 256 pairs are in the script's output (part digest).

## Appendix (working copy of the script)

```python
"""Check of the E59 brief (17 November 2026). Read-only: every file is read with `git show`; no code of another branch is executed; nothing
is trained; torch is not used. Standard library, and numpy for the parts digest, weights and format.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I check_e59.py <part> ...      (from anywhere inside the repository)
    parts: figures task votes digest weights format flat

Lines marked "copied" are copied from the named file at origin/codex/minifly-consolidated-20261023 (7cabb13a)."""
import collections, hashlib, io, json, pickle, subprocess, sys, zipfile
from fractions import Fraction
from math import comb
from pathlib import Path
from random import Random

D = "origin/codex/minifly-consolidated-20261023"; M = "MOD97_LOCAL_RULE_20261106"
TOP = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent)).stdout.strip() or "."
SEEDS = (73129, 93683, 116089)
mean = lambda v: sum(v) / len(v)


def blob(path, ref=D):
    r = subprocess.run(["git", "-C", TOP, "show", f"{ref}:{M}/{path}"], capture_output=True)
    assert r.returncode == 0, (ref, path)
    return r.stdout


def js(path, ref=D):
    return json.loads(blob(path, ref))


# ---------------------------------------------------------------- the task (copied from e43_task.py)
SYM = list(range(128, 192)); FILL = list(range(192, 224)); CODE, ASK = 224, 225; EOS, BOS = 256, 257; KMAX, TAUGHT = 12, 2048


def split(seed):
    pairs = [(a, b) for a in SYM for b in SYM]
    Random(f"e43|{seed}|split").shuffle(pairs)
    return pairs[:TAUGHT], pairs[TAUGHT:]


def render(a, b, k, rng):
    return [CODE, a, b] + [rng.choice(FILL) for _ in range(k)] + [ASK], [a, b, EOS]


def panel(seed, which, n=300, k=12):
    taught, held = split(seed); rng = Random(f"e43|{seed}|panel|{which}")
    return [render(a, b, k, rng) for a, b in rng.sample(taught if which == "taught" else held, n)]


def stream_n(seed, n):                               # e43_task.stream's text with taught[:n]; n = 2,048 is e43_task.stream
    taught = split(seed)[0][:n]; rng = Random(f"e43|{seed}|stream")
    while True:
        order = list(range(len(taught))); rng.shuffle(order)
        for i in order:
            yield render(*taught[i], rng.randint(0, KMAX), rng)


def spec_digest(st, updates, batch=64):              # e43_task.digest's text, on any stream
    h = hashlib.sha256()
    for _ in range(updates):
        for p, y in [next(st) for _ in range(batch)]:
            h.update(bytes([len(p)]) + bytes(p) + bytes(v % 256 for v in y))
    return h.hexdigest()


def taught_panel_n(seed, n, k=12):                   # the construction proposed in the repair of defect 6
    pool = split(seed)[0][:n]; rng = Random(f"e43|{seed}|panel|taught")
    return [render(a, b, k, rng) for a, b in rng.sample(pool, min(300, n))]


# ---------------------------------------------------------------- weights without torch (reader copied from compare_pt_tensors.py)
class _Unpickler(pickle.Unpickler):                  # refuses every class but those that describe a tensor: nothing from a file is executed
    def find_class(self, mod, name):
        if mod == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if mod == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, stride, *a, **k: ("T", storage, offset, tuple(size))
        if mod == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *a, **k: data
        if mod == "torch" and name.endswith("Storage"):
            return name
        raise pickle.UnpicklingError(f"refused {mod}.{name}")

    def persistent_load(self, pid):
        return (pid[1], pid[2])


def weights(path):
    import numpy as np
    z = zipfile.ZipFile(io.BytesIO(blob(path))); pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]; root = pk[:-len("data.pkl")]; out = {}
    for name, o in _Unpickler(z.open(pk)).load().items():
        _, (stype, key), offset, shape = o
        assert stype == "FloatStorage"
        n = 1
        for s in shape:
            n *= s
        out[name] = np.frombuffer(z.read(root + "data/" + key)[offset * 4:(offset + n) * 4], dtype="<f4").reshape(shape).copy()
    return out


def gru(w, x, all_steps=False):                      # torch's GRU as arch26.forward writes it out; returns the logits
    import numpy as np
    sg = lambda v: 1.0 / (1.0 + np.exp(-v))
    z1 = np.tanh(w["book.weight"][x] @ w["l1.weight"].T + w["l1.bias"])
    gi = z1 @ w["rnn.weight_ih_l0"].T + w["rnn.bias_ih_l0"]; Whh, bhh = w["rnn.weight_hh_l0"], w["rnn.bias_hh_l0"]; U = Whh.shape[1]
    h = np.zeros((x.shape[0], U), dtype=np.float32); Hs = []
    for t in range(x.shape[1]):
        gh = h @ Whh.T + bhh
        r = sg(gi[:, t, :U] + gh[:, :U]); z = sg(gi[:, t, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
        h = (1 - z) * n + z * h; Hs.append(h)
    if all_steps:
        Hs = np.stack(Hs, 1)
        return np.tanh(z1 @ w["l2.weight"].T + w["l2.bias"] + Hs @ w["lm.weight"].T + w["lm.bias"]) @ w["l3.weight"].T + w["l3.bias"]
    return np.tanh(z1[:, -1] @ w["l2.weight"].T + w["l2.bias"] + h @ w["lm.weight"].T + w["lm.bias"]) @ w["l3.weight"].T + w["l3.bias"]


# ---------------------------------------------------------------- parts
def figures():
    print("E43: right of 300 on the taught and on the held panel, by row")
    R = {}
    for arm in ("unrelated-decay", "unrelated"):
        for rule in ("B", "S"):
            for seed in SEEDS:
                j = js(f"exp45/e43/e43_{arm}_{rule}_{seed}.json")
                R[(arm, rule, seed)] = {r["updates"]: (r["score3_new_task"]["taught"]["exact"], r["score3_new_task"]["held"]["exact"]) for r in j["rows"]}
                c = j["complete"]
                print(f"  {arm:15s} {rule} {seed}: decay {j['settings']['decay']} rows {sorted(R[(arm, rule, seed)])} counts {[R[(arm, rule, seed)][u] for u in sorted(R[(arm, rule, seed)])]} "
                      f"cpu per update {c['cpu_updates'] / 4096:.4f}")
    dmax = max(abs(a - b) for (arm, rule, seed), v in R.items() if arm == "unrelated" for u in v for a, b in zip(v[u], R[("unrelated-decay", rule, seed)][u]))
    print("  largest difference between the arm with decay and the arm without, any row and panel:", dmax, "items")
    ext = subprocess.run(["git", "-C", TOP, "ls-tree", "-r", "--name-only", D, "--", f"{M}/exp45/e43/"], capture_output=True, text=True).stdout
    print("  receipts of an extension beyond 4,096 updates on file:", [l for l in ext.splitlines() if "_ext" in l])
    print("E40: reduction score at update 8,192, and cost")
    for kind in ("lived", "fresh"):
        vals, cost = [], []
        for rule in ("S", "B"):
            for seed in SEEDS:
                j = js(f"exp45/e40/e40_{kind}_{rule}_{seed}.json")
                assert j["rows"][-1]["updates"] == 8192
                vals.append(j["rows"][-1]["scores"]["summary"]["reduction_score"]); cost.append(j["cpu_updates"])
                if kind == "fresh":
                    assert "tensors_sha256" in j["start"]
        print(f"  {kind}: {min(vals):.2f} to {max(vals):.2f}; CPU-seconds of the updates {min(cost):.0f} to {max(cost):.0f} ({min(cost) / 8192:.4f} to {max(cost) / 8192:.4f} an update)")
    print("  fresh start: one tensors_sha256 for both rules of a seed:", all(js(f"exp45/e40/e40_fresh_S_{s}.json")["start"]["tensors_sha256"] == js(f"exp45/e40/e40_fresh_B_{s}.json")["start"]["tensors_sha256"] for s in SEEDS))
    print("The starts on the default branch")
    for rule in ("S", "B"):
        for seed in SEEDS:
            b = js(f"exp45/e26/e26_dev_{rule}_{seed}.json"); l = js(f"exp45/e26/e26_life_{rule}_{seed}.json")
            srcs = b["provenance"]["sources_sha256"]
            diff = sorted(k for k in srcs if hashlib.sha256(blob(k)).hexdigest() != srcs[k])
            okb = hashlib.sha256(blob(f"exp45/e26/e26_dev_{rule}_{seed}.pt")).hexdigest() == b["weights_sha256"]
            okl = hashlib.sha256(blob(f"exp45/e26/e26_life_{rule}_{seed}.pt")).hexdigest() == l["complete"]["weights_sha256"]
            e43 = js(f"exp45/e43/e43_unrelated-decay_{rule}_{seed}.json"); e40 = js(f"exp45/e40/e40_lived_{rule}_{seed}.json")
            print(f"  {rule} {seed}: birth file = receipt {okb}; life file = receipt {okl}; birth updates {b['optimizer_steps']}, life updates {l['complete']['optimizer_steps']}; parameters {b['model']['params']}; "
                  f"sources of the birth that differ on the default branch {diff}; E43's birth_tensors_sha256 = birth receipt {e43['settings']['birth_tensors_sha256'] == b['tensors_sha256']}; "
                  f"E40's lived start = life receipt {e40['start']['tensors_sha256'] == l['complete']['tensors_sha256']}")
    print("  E43's admission:", js("exp45/e43/e43_unrelated-decay_S_73129.json")["settings"]["birth_admission"])


def task():
    print("chance that a given symbol is absent from a given place of n pairs drawn without replacement from 4,096")
    for n in (256, 512):
        p = float(Fraction(comb(4096 - 64, n), comb(4096, n)))
        print(f"  n = {n}: {100 * p:.4f}% (expected absent symbols in a place {64 * p:.3f}); exp(-64 n / 4096) = {100 * 2.718281828459045 ** (-64 * n / 4096):.4f}%")
    for seed in SEEDS:
        taught, held = split(seed); hp = panel(seed, "held"); hpairs = [(p[1], p[2]) for p, _ in hp]
        assert set(taught).isdisjoint(held) and all(pr in set(held) for pr in hpairs) and len(set(hpairs)) == 300
        assert taught_panel_n(seed, 2048) == panel(seed, "taught")
        print(f"seed {seed}: e43_task.digest(seed, 64) {spec_digest(stream_n(seed, 2048), 64)}; proposed taught panel at 2,048 is E43's: True")
        for n in (2048, 512, 256):
            sub = taught[:n]; c1 = collections.Counter(a for a, _ in sub); c2 = collections.Counter(b for _, b in sub)
            m1 = sorted(set(SYM) - set(c1)); m2 = sorted(set(SYM) - set(c2))
            bad = sum((a in m1) or (b in m2) for a, b in hpairs)
            try:
                Random(f"e43|{seed}|panel|taught").sample(sub, 300); samp = "possible"
            except ValueError as e:
                samp = f"ValueError ({e})"
            tp = taught_panel_n(seed, n)
            print(f"  n = {n}: symbols absent from the first place {m1}, from the second {m2}; a symbol occurs in a place {min(min(c1[s], c2[s]) for s in SYM)} to {max(max(c1.values()), max(c2.values()))} times; "
                  f"shown once or twice in a place: {sum(0 < c1[s] <= 2 for s in SYM)} + {sum(0 < c2[s] <= 2 for s in SYM)}")
            print(f"           held-panel items with a symbol in a place where no taught pair has it: {bad} of 300; sampling 300 as e43_task.panel does: {samp}; proposed taught panel: {len(tp)} items, "
                  f"{len({(p[1], p[2]) for p, _ in tp})} distinct pairs, all taught: {all((p[1], p[2]) in set(sub) for p, _ in tp)}; e43_task.digest form, first 64 batches: {spec_digest(stream_n(seed, n), 64)}")
    rows = list(range(0, 1025, 32)) + list(range(1152, 8193, 128))
    print("rows of the brief:", len(rows), "| in 7,296..8,192:", sum(7296 <= r <= 8192 for r in rows), "| in 5,248..6,144:", sum(5248 <= r <= 6144 for r in rows), "| last three:", rows[-3:])
    for first in (960, 992, 1024, 7936, 8064):
        tri = rows[rows.index(first):rows.index(first) + 3]
        print(f"  three consecutive rows from {first}: {tri}" + (f", a span of {tri[-1] - tri[0]} updates" if len(tri) == 3 else ": fewer than three rows remain"))
    new = list(range(0, 513, 16)) + list(range(544, 8193, 32))
    print("rows of the repair (every 16 to 512, every 32 after):", len(new))


def votes():
    """The brief's rule on two of three seeds, applied to one invented set of updates to 80 (fresh, born, lived)."""
    data = {1: (1000, 600, 840), 2: (1000, 600, 360), 3: (1000, 1400, 840)}

    def word(first, second):
        q = [data[s][first] / data[s][second] for s in data]
        lo, hi = sum(v < 2 / 3 for v in q), sum(v > 1.5 for v in q)
        w = "fewer" if lo >= 2 and hi == 0 else "more" if hi >= 2 and lo == 0 else "about the same"
        return w, [round(v, 2) for v in q]
    print("updates to 80 (fresh, born, lived) by seed:", data)
    print("  born against fresh:", word(1, 0)); print("  lived against born:", word(2, 1)); print("  lived against fresh:", word(2, 0))


def digest():
    """exp26_candidate.Digest over exp45.batch_tensor's tensors, written in numpy from their text; tested against E43's receipts."""
    import numpy as np

    def run(seed, n, rows):
        h = hashlib.sha256(); st = stream_n(seed, n); out = {}
        for u in range(1, max(rows) + 1):
            items = [next(st) for _ in range(64)]
            L = max(len(p) + len(t) for p, t in items) + 1
            data = np.full((64, L), BOS, dtype=np.int64)
            for i, (p, t) in enumerate(items):
                data[i, 1:len(p) + len(t) + 1] = p + t
            x, y = data[:, :-1], data[:, 1:]; mask = np.zeros(y.shape, dtype=np.bool_)
            for i, (p, t) in enumerate(items):
                mask[i, len(p):len(p) + len(t)] = True
            for t_ in (x, y, mask):
                h.update(str(tuple(int(s) for s in t_.shape)).encode()); h.update(np.ascontiguousarray(t_).tobytes())
            if u in rows:
                out[u] = h.copy().hexdigest()
        return out
    for seed in SEEDS:
        want = {r["updates"]: r["data_digest"]["sha256"] for r in js(f"exp45/e43/e43_unrelated-decay_S_{seed}.json")["rows"] if r["updates"]}
        got = run(seed, 2048, set(want) | {8192})
        print(f"seed {seed}: n = 2,048 equals E43's receipt at rows {sorted(want)}: {all(got[u] == want[u] for u in want)}; row 8,192: {got[8192]}")
        for n in (512, 256):
            g = run(seed, n, {64, 8192}); print(f"   n = {n}: row 64 {g[64]}; row 8,192 {g[8192]}")


def weights_part():
    import numpy as np
    rms = lambda a: float(np.sqrt(np.mean(np.square(a.astype(np.float64)))))
    TASKB = list(range(128, 226)); FAM1 = sorted(set(b"0123456789+-><=pqvwABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    W = {}
    print("the byte table's rows 128 to 225; 'fresh' is read from E40's fresh runs (after 8,192 updates on another task; after 64 for seed 73129)")
    for seed in SEEDS:
        files = [(k, r, f) for r in ("S", "B") for k, f in (("fresh8192", f"exp45/e40/e40_fresh_{r}_{seed}_row8192.pt"), ("born", f"exp45/e26/e26_dev_{r}_{seed}.pt"), ("lived", f"exp45/e26/e26_life_{r}_{seed}.pt"))]
        if seed == 73129:
            files += [("fresh64", r, f"exp45/e40/e40_fresh_{r}_{seed}_row64.pt") for r in ("S", "B")]
        ref = None; eq = []
        for k, r, f in files:
            W[(k, r, seed)] = weights(f); rows = W[(k, r, seed)]["book.weight"][TASKB]
            ref = rows if ref is None else ref; eq.append(bool(np.array_equal(rows, ref)))
        print(f"  seed {seed}: {len(files)} files, rows bit-equal in all: {all(eq)}; rms of these rows {rms(ref):.3f}")
        for r in ("S", "B"):
            print(f"     rule {r}: rms of the rows of the first family's bytes: born {rms(W[('born', r, seed)]['book.weight'][FAM1]):.3f}, lived {rms(W[('lived', r, seed)]['book.weight'][FAM1]):.3f}")
    cols = [("fresh64", "S"), ("fresh64", "B"), ("born", "S"), ("born", "B"), ("lived", "S"), ("lived", "B")]
    print("root mean square of each tensor, seed 73129; columns:", cols)
    for n in W[("born", "S", 73129)]:
        print(f"  {n:18s} " + " ".join(f"{rms(W[(k, r, 73129)][n]):8.4f}" for k, r in cols))
    print("  parameters:", sum(v.size for v in W[("born", "S", 73129)].values()))
    print("range over the three seeds of the ratio born/initial and lived/initial, the initial rms taken as torch's default (uniform on +-1/sqrt(fan-in))")
    init = {"l1.weight": 8 ** -0.5 / 3 ** 0.5, "rnn.weight_ih_l0": (1 / 16) / 3 ** 0.5, "rnn.weight_hh_l0": (1 / 16) / 3 ** 0.5, "lm.weight": (1 / 16) / 3 ** 0.5,
            "l2.weight": 512 ** -0.5 / 3 ** 0.5, "l3.weight": 512 ** -0.5 / 3 ** 0.5, "l3.bias": 512 ** -0.5 / 3 ** 0.5}
    for n, v0 in init.items():
        q = [rms(W[(k, r, s)][n]) / v0 for k in ("born", "lived") for r in ("S", "B") for s in SEEDS]
        print(f"  {n:18s} initial {v0:.4f}; ratio {min(q):.2f} to {max(q):.2f}")
    print("loss on the answer bytes of the first 64 items of E43's stream, before any update of the task (ln 257 = 5.55)")
    for seed in SEEDS:
        st = stream_n(seed, 2048); items = [next(st) for _ in range(64)]; line = []
        for k, r in (("born", "S"), ("born", "B"), ("lived", "S"), ("lived", "B")):
            ls = []
            for p, t in items:
                lg = gru(W[(k, r, seed)], np.array([[BOS] + p + t[:-1]]), all_steps=True)[0, -3:].astype(np.float64)
                lg -= lg.max(1, keepdims=True); lp = lg - np.log(np.exp(lg).sum(1, keepdims=True)); ls.append(-mean([lp[j, t[j]] for j in range(3)]))
            line.append(f"{k} {r} {mean(ls):.2f}")
        print(f"  seed {seed}: " + "; ".join(line))


def format_part():
    import numpy as np

    def respond(w, items, cap=6):                    # greedy decoding as exp11_cell.respond
        x = np.array([[BOS] + p for p, _ in items]); gen = [[] for _ in items]; done = [False] * len(items)
        for _ in range(cap):
            y = gru(w, x).argmax(1)
            for q in range(len(items)):
                if not done[q]:
                    gen[q].append(int(y[q])); done[q] = int(y[q]) == EOS
            x = np.concatenate([x, y[:, None]], 1)
            if all(done):
                break
        return gen
    kind = lambda b: "digit" if 48 <= b <= 57 else "end" if b == EOS else "symbol" if 128 <= b <= 191 else "other"
    for seed in SEEDS:
        items = panel(seed, "held")
        files = [("born S", f"exp45/e26/e26_dev_S_{seed}.pt"), ("born B", f"exp45/e26/e26_dev_B_{seed}.pt"), ("lived S", f"exp45/e26/e26_life_S_{seed}.pt"), ("lived B", f"exp45/e26/e26_life_B_{seed}.pt")]
        if seed == 73129:
            files += [("E43 unrelated S, row 64", "exp45/e43/e43_unrelated_S_73129_row64.pt"), ("E43 unrelated B, row 64", "exp45/e43/e43_unrelated_B_73129_row64.pt")]
        for name, f in files:
            outs = respond(weights(f), items)
            print(f"seed {seed} {name:24s}: whole answer right {sum(o == t for o, (_, t) in zip(outs, items)):3d}; three bytes with the end symbol third {sum(len(o) == 3 and o[2] == EOS for o in outs):3d}; "
                  f"right by place {[sum(len(o) > j and o[j] == t[j] for o, (_, t) in zip(outs, items)) for j in range(3)]}; commonest forms {collections.Counter(tuple(kind(b) for b in o) for o in outs).most_common(2)}")


def flat():
    """The brief's rule 'has flattened' at update 8,192, on E41's curves of fresh networks on the first family (held-out pairs at 12 letters)."""
    n = fl = w5 = w10 = 0
    for h in (512, 256):
        for f in (10, 25, 50):
            for rule in ("S", "B"):
                for seed in SEEDS:
                    c = js(f"exp45/e41/e41_{rule}_h{h}_f{f}_{seed}.json")["curve"]; v = {r["step"]: r["A"] for r in c}; t = {r["step"]: r["taught"] for r in c}
                    if any(all(v[s + 256 * i] >= 80 for i in range(3)) for s in sorted(v) if s <= 8192 - 512):
                        continue
                    late = mean([v[s] for s in v if 7296 <= s <= 8192]); early = mean([v[s] for s in v if 5248 <= s <= 6144])
                    l8 = mean([v[s] for s in (7680, 7936, 8192)]); le = mean([v[s] for s in sorted(v)[-3:]]); isf = late - early < 3
                    n += 1; fl += isf; w5 += isf and le - l8 >= 5; w10 += isf and le - l8 >= 10
                    print(f"  width {h} fraction 0.{f} {rule} {seed}: windows {early:5.1f} -> {late:5.1f}; flattened {str(isf):5s}; level at 8,192 {l8:5.1f}, taught panel {t[8192]:5.1f}; level at {max(v)} {le:5.1f}")
    print(f"curves below 80 at 8,192: {n}; called flattened: {fl}; of these higher later by 5 points or more: {w5}; by 10 or more: {w10}")
    print("fresh networks, sweep against back-propagation, updates to 80 on held-out pairs at 12 letters (first of three rows), E41 at the present width")
    for f in (50, 25):
        for seed in SEEDS:
            x = {}
            for rule in ("S", "B"):
                v = [(r["step"], r["A"]) for r in js(f"exp45/e41/e41_{rule}_h512_f{f}_{seed}.json")["curve"]]
                x[rule] = next((v[i][0] for i in range(len(v) - 2) if all(a >= 80 for _, a in v[i:i + 3])), None)
            print(f"  fraction 0.{f} seed {seed}: sweep {x['S']}, back-propagation {x['B']}, ratio {None if None in x.values() else round(x['S'] / x['B'], 2)}")
    print("fresh networks on a family: the taught panel in the first rows")
    for rule in ("S", "B"):
        for seed in SEEDS:
            j = js(f"exp45/e40/e40_fresh_{rule}_{seed}.json")
            print(f"  E40 fresh {rule} {seed}: (update, taught panel) {[(r['updates'], round(r['scores']['summary']['taught_add_sub_mean'], 1)) for r in j['rows']]}")


if __name__ == "__main__":
    P = dict(figures=figures, task=task, votes=votes, digest=digest, weights=weights_part, format=format_part, flat=flat)
    for a in sys.argv[1:] or list(P):
        print(f"==== {a}"); P[a]()
```
