"""E40 runner (brief OPUS_TASK_E40_NEW_FAMILY_FROM_A_TENTH_OF_PAIRS_20261115.md, revision 2): does a learner that has lived generalise a
new family (F4, mod 83, codes e f g h) from a tenth of its pairs, where a fresh one memorises? Built on a copy of e36_reminder.py (the
start from a life-end file or a fresh net, the three update paths and E36's checks 3 and 4 are kept as they were). A new file; every
existing source stays as it is.

Runs (8,192 updates each, fresh exp26_candidate.make_opt in every run, the same batches for every run of a seed):
    lived S   from exp45/e26/e26_life_S_<seed>.pt, the sweep: exp26_candidate.step(m, opt, items, True)
    lived B   from exp45/e26/e26_life_B_<seed>.pt, backprop: exp26_candidate.step(m, opt, items, False)
    lived K   from exp45/e31/e31_life_K_<seed>.pt with Bhh from e31_life_K_<seed>_Bhh.pt: step_K (E36's)
    fresh S, B   arch26.new_net(seed), untrained, sweep / backprop
Data: batches of 64 consecutive items of X16.entry_stream([("op", t, a, b) for t in F4 for a, b in split(t, .10, seed)[0]], seed + 140).
Rows: 0, 64, 128, 256, 512, 1024, 2048, 4096, 8192 updates; scores and weights at every row.

    python3 e40_new_family.py run <lived|fresh> <S|B|K> <seed> [--updates N] [--out DIR]
    python3 e40_new_family.py check <1|3|4|5|7|8> [seed] [--out DIR]     (check 2 is read from run receipts: `check 2 <dir>`)
"""
import json
import sys
import time
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C          # noqa: E402  (imports exp20_reference, which applies the environment gate)
import torch as T                    # noqa: E402
import arch26                        # noqa: E402
import arch31_workerB as A31         # noqa: E402

E, X, X16, R = C.E, C.X, C.X16, C.R
OUT = HERE / "exp45" / "e40"
E26, E31 = HERE / "exp45" / "e26", HERE / "exp45" / "e31"
ROWS = (0, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
DECAY = C.LR * C.WD                  # 0.00025
SEEDS = (73129, 93683, 116089)
OWN = ("e40_new_family.py", "e36_reminder.py", "arch31_workerB.py", "exp31_switch_workerB.py")
F4 = [("add", 83, b"e"), ("sub", 83, b"f"), ("max", 83, b"g"), ("min", 83, b"h")]
F4N = [f"{t[0]}83" for t in F4]
EARLIER6 = ("add89", "sub89", "add97", "sub97", "add73", "sub73")
FRAC = .10

# ---------------------------------------------------------------- the family, asserted new
_EARLIER = list(E.DEV) + list(E.LIFE) + list(X16.F1) + list(X16.F3)
assert not {t[2] for t in F4} & ({t[2] for t in _EARLIER} | {b"c"}), "a code of F4 is used by an earlier family"
assert 83 not in {t[1] for t in _EARLIER}, "modulus 83 is used by an earlier family"
assert all(len(t[2]) == 1 and 97 <= t[2][0] <= 122 for t in F4)


def provenance(cmd):
    p = C.provenance(cmd)
    p["e40_sources_sha256"] = {s: X16.sha_file(HERE / s) for s in OWN}
    return p


# ---------------------------------------------------------------- data and panels
def split(t, seed):
    return E.split(t, FRAC, seed)


def entries(seed):
    return [("op", t, a, b) for t in F4 for a, b in split(t, seed)[0]]


def batches(seed):
    st = X16.entry_stream(entries(seed), seed + 140)
    while True:
        yield [next(st) for _ in range(X.BATCH)]


def needs_reduction(op, a, b):
    return a + b >= 83 if op == "add" else a - b < 0


def held_plain(t, seed):
    """Score 3: every held-out pair of t rendered with k = 0, split by whether its answer needs the reduction at 83."""
    red, nored = [], []
    for a, b in split(t, seed)[1]:
        (red if needs_reduction(t[0], a, b) else nored).append(E.render(t, a, b, 0, Random(0)))
    return dict(reduction=red, no_reduction=nored)


class Scorer:
    def __init__(self, seed):
        self.held = {f"{t[0]}83": E.panel(t, split(t, seed)[1], seed, ks=(12,), n=300)[12] for t in F4}
        self.taught = {f"{t[0]}83": E.panel(t, split(t, seed)[0], seed + 1, ks=(12,), n=300)[12] for t in F4}
        self.plain = {f"{t[0]}83": held_plain(t, seed) for t in F4[:2]}
        self.P, _ = X16.panels(seed)

    def __call__(self, m):
        s = X16.score
        held = {n: s(m, it, None) for n, it in self.held.items()}
        taught = {n: s(m, it, None) for n, it in self.taught.items()}
        plain = {n: {w: s(m, it, None) for w, it in v.items()} for n, v in self.plain.items()}
        earlier = {n: s(m, self.P[n][12], None) for n in EARLIER6}
        pct = lambda d: 100.0 * d["exact"] / d["n"]
        summary = dict(held_add_sub_mean=(pct(held["add83"]) + pct(held["sub83"])) / 2,
                       taught_add_sub_mean=(pct(taught["add83"]) + pct(taught["sub83"])) / 2,
                       reduction_score=(pct(plain["add83"]["reduction"]) + pct(plain["sub83"]["reduction"])) / 2,
                       no_reduction_score=(pct(plain["add83"]["no_reduction"]) + pct(plain["sub83"]["no_reduction"])) / 2)
        return dict(held_k12=held, taught_k12=taught, held_plain_k0=plain, earlier_k12=earlier, summary=summary)


# ---------------------------------------------------------------- starts (as e36_reminder.start; "lived" for "reminder", "fresh" for "reference")
def start(kind, rule, seed):
    m = arch26.new_net(seed)
    if kind == "fresh":
        assert rule in ("S", "B")
        return m, None, dict(start="arch26.new_net(seed), untrained", tensors_sha256=R.tensors_sha(m.state_dict()))
    if rule in ("S", "B"):
        f = E26 / f"e26_life_{rule}_{seed}.pt"
    else:
        f = E31 / f"e31_life_K_{seed}.pt"
    rec = json.loads(f.with_suffix(".json").read_text())
    sha = X16.sha_file(f)
    assert sha == rec["complete"]["weights_sha256"], f"{f.name}: file hash is not the receipt's"
    m.load_state_dict(T.load(f))
    tsha = R.tensors_sha(m.state_dict())
    info = dict(start=str(f.relative_to(HERE)), weights_sha256=sha, tensors_sha256=tsha, receipt_tensors_sha256=rec["complete"]["tensors_sha256"],
                tensors_sha_matches_receipt=tsha == rec["complete"]["tensors_sha256"])
    B = None
    if rule == "K":
        bf = E31 / f"e31_life_K_{seed}_Bhh.pt"
        B = T.load(bf)["Bhh"].detach().clone()
        bsha = R.tensors_sha({"Bhh": B})
        info.update(Bhh_file=str(bf.relative_to(HERE)), Bhh_file_sha256=X16.sha_file(bf), Bhh_tensors_sha256=bsha,
                    receipt_Bhh_final_tensors_sha256=rec["e31"]["Bhh_final_tensors_sha256"], Bhh_matches_receipt=bsha == rec["e31"]["Bhh_final_tensors_sha256"])
    return m, B, info


# ---------------------------------------------------------------- the update (as e36_reminder)
def step_K(m, opt, items, B):
    """exp26_candidate.step's text with arch31_workerB.forward(switch="K", Bhh=B); then Bhh's rule (exp31_switch_workerB lines 74-77, 196-202)."""
    x, y, mask = X.batch_tensor(items)
    logits = A31.forward(m, x, True, True, switch="K", Bhh=B)[0]
    loss = E.cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    W0 = m.rnn.weight_hh_l0.detach().clone()
    opt.step()
    W1 = m.rnn.weight_hh_l0.detach().clone()
    B = (1 - DECAY) * B + (W1 - (1 - DECAY) * W0)
    return float(loss.detach()), B


class Learner:
    def __init__(self, m, rule, B=None):
        self.m, self.rule, self.B = m, rule, B
        self.opt = C.make_opt(m)

    def update(self, items):
        if self.rule == "K":
            loss, self.B = step_K(self.m, self.opt, items, self.B)
            return loss
        return C.step(self.m, self.opt, items, self.rule == "S")


def bhh_measures(B, W):
    W = W.detach()
    return dict(Bhh_tensors_sha256=R.tensors_sha({"Bhh": B}),
                cos_Bhh_W=float(T.nn.functional.cosine_similarity(B.flatten().double(), W.flatten().double(), dim=0)),
                rel_diff_Bhh_W=float((B.double() - W.double()).norm() / W.double().norm()))


# ---------------------------------------------------------------- check 5, per item: what was presented
CODE = {t[2][0]: t for t in F4}


def parse(prompt):
    """(task, a, b) of an F4 item rendered by exp11_cell.render: code, a (2 digits), symbol, letters, b (2 digits), '='."""
    t = CODE[prompt[0]]
    assert prompt[-1] == ord("=") and prompt[3] == ord(E.SYM[t[0]])
    a = int(bytes(prompt[1:3])); b = int(bytes(prompt[-3:-1]))
    assert all(65 <= c <= 90 for c in prompt[4:-3])
    return t, a, b


class Presented:
    def __init__(self, seed):
        self.taught = {(t[0], a, b) for t in F4 for a, b in split(t, seed)[0]}
        self.held = {(t[0], a, b) for t in F4 for a, b in split(t, seed)[1]}
        self.n = self.held_hits = self.not_taught = 0
        self.earlier_codes = 0
        self.codes_earlier = {t[2][0] for t in _EARLIER} | {ord("c")}

    def add(self, items):
        for p, ans in items:
            self.earlier_codes += p[0] in self.codes_earlier
            t, a, b = parse(p)
            k = (t[0], a, b)
            self.n += 1; self.held_hits += k in self.held; self.not_taught += k not in self.taught
            assert ans == list(f"{E.apply(t[0], 83, a, b):02d}".encode()) + [E.EOS]

    def value(self):
        return dict(items=self.n, held_out_presented=self.held_hits, not_taught_presented=self.not_taught, earlier_code_items=self.earlier_codes)


# ---------------------------------------------------------------- one run
def run(kind, rule, seed, updates, out, cmd):
    arch26.install()
    out = Path(out)
    name = f"e40_{kind}_{rule}_{seed}"
    rows = [r for r in ROWS if r <= updates]
    if updates not in rows:
        rows.append(updates)
    rec_path = out / f"{name}.json"
    wpaths = {r: out / f"{name}_row{r}.pt" for r in rows}
    R.refuse_existing(rec_path, *wpaths.values(), *[str(p) + ".tmp" for p in wpaths.values()])
    out.mkdir(parents=True, exist_ok=True)
    T.manual_seed(seed)
    m, B, info = start(kind, rule, seed)
    L = Learner(m, rule, B)
    sc = Scorer(seed)
    bt = batches(seed)
    dg = C.Digest(); pr = Presented(seed)
    rec = dict(experiment="E40", kind=kind, rule=rule, seed=seed, updates=updates, rows_planned=rows, start=info,
               book_rows_efgh_tensors_sha256=R.tensors_sha({"efgh": m.book.weight.detach()[[101, 102, 103, 104]]}),
               optimizer=C.opt_record(m, L.opt), data="64 consecutive items of X16.entry_stream(F4 taught tenth, seed + 140)",
               update={"S": "exp26_candidate.step(local=True)", "B": "exp26_candidate.step(local=False)",
                       "K": "e40_new_family.step_K (= e36_reminder.step_K; arch31_workerB.forward switch=K, Bhh rule d=0.00025)"}[rule],
               provenance=provenance(cmd), rows=[], complete=False)
    t0 = time.process_time(); cpu_score = 0.0; losses = []; u = 0
    for r in rows:
        while u < r:
            items = next(bt); dg.add(items); pr.add(items); losses.append(L.update(items)); u += 1
        cb = time.process_time(); s = sc(m); cpu_score += time.process_time() - cb
        wsha = R.save_weights(m, wpaths[r])
        row = dict(updates=r, scores=s, loss_since_last_row=(sum(losses) / len(losses)) if losses else None,
                   finite_losses=all(v == v and abs(v) != float("inf") for v in losses), data_digest=dg.value(), presented=pr.value(),
                   weights=wpaths[r].name, weights_sha256=wsha, tensors_sha256=R.tensors_sha(m.state_dict()),
                   optimizer_steps=R.opt_steps(L.opt), cpu=time.process_time() - t0, cpu_scoring_so_far=cpu_score)
        if rule == "K":
            row["Bhh"] = bhh_measures(L.B, m.rnn.weight_hh_l0)
        losses = []
        rec["rows"].append(row)
        sm = s["summary"]
        print(name, r, f"held {sm['held_add_sub_mean']:.1f} taught {sm['taught_add_sub_mean']:.1f} red {sm['reduction_score']:.1f} nored {sm['no_reduction_score']:.1f}",
              f"cpu {row['cpu']:.0f}s", flush=True)
        rec_path.write_text(json.dumps(rec, indent=1))
    assert pr.held_hits == 0 and pr.not_taught == 0 and pr.earlier_codes == 0, pr.value()
    rec.update(complete=True, cpu_total=time.process_time() - t0, cpu_scoring=cpu_score, cpu_updates=time.process_time() - t0 - cpu_score)
    rec_path.write_text(json.dumps(rec, indent=1))
    print(name, "complete", flush=True)


# ---------------------------------------------------------------- checks
def check1():
    """Row 0 of each lived run has the life-end file's tensors_sha256."""
    arch26.install()
    res = {}
    for seed in SEEDS:
        for rule in ("S", "B", "K"):
            _, _, info = start("lived", rule, seed)
            res[f"{rule}_{seed}"] = dict(tensors_sha256=info["tensors_sha256"], receipt=info["receipt_tensors_sha256"], ok=info["tensors_sha_matches_receipt"])
    return dict(rows=res, ok=all(v["ok"] for v in res.values()))


def check2(d):
    """The batch digest is equal across every run of a seed at every row (read from the run receipts in d)."""
    res = {}
    for seed in SEEDS:
        recs = sorted(Path(d).glob(f"e40_*_{seed}.json"))
        if not recs:
            continue
        dig = {}
        for p in recs:
            r = json.loads(p.read_text())
            for row in r["rows"]:
                dig.setdefault(row["updates"], {})[p.stem] = row["data_digest"]["sha256"]
        res[seed] = dict(runs=[p.stem for p in recs], rows={u: dict(distinct=len(set(v.values())), runs=len(v)) for u, v in dig.items()},
                         ok=all(len(set(v.values())) == 1 and len(v) == len(recs) for v in dig.values()))
    return dict(seeds=res, ok=bool(res) and all(v["ok"] for v in res.values()))


def maxrel(a, b):                    # exp31_switch_workerB.maxrel
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def grads(m, x, y, mask, **kw):      # exp31_switch_workerB.grads, against arch31_workerB
    m.zero_grad()
    logits, _ = A31.forward(m, x, True, True, **kw)
    E.cross_entropy(logits[mask], y[mask]).backward()
    return {n: p.grad.detach().clone() for n, p in m.named_parameters()}


def check4(seed):
    """Arm K, as E36's check 3 (r2), on E40's first batches."""
    arch26.install()
    res = {}
    m, B, info = start("lived", "K", seed)
    res["Bhh_sha"] = dict(loaded=info["Bhh_tensors_sha256"], receipt=info["receipt_Bhh_final_tensors_sha256"], ok=info["Bhh_matches_receipt"])
    first = next(batches(seed)); x, y, mask = X.batch_tensor(first)
    gH = grads(m, x, y, mask, switch="H")
    gKW = grads(m, x, y, mask, switch="K", Bhh=m.rnn.weight_hh_l0.detach().clone())
    rel = {n: maxrel(gKW[n], gH[n]) for n in gH}
    res["K_equals_H_when_Bhh_is_W"] = dict(grads_max_rel=rel, worst=max(rel.values()), ok=max(rel.values()) <= 1e-6)
    gK = grads(m, x, y, mask, switch="K", Bhh=B)
    gN = grads(m, x, y, mask, switch=None)
    rel2 = {n: maxrel(gK[n], gN[n]) for n in gN}
    res["K_loaded_differs_from_None"] = dict(grads_max_rel=rel2, worst=max(rel2.values()), ok=max(rel2.values()) > 1e-3)
    m.zero_grad()
    L = Learner(m, "K", B)
    rows = []
    bt = batches(seed)
    for u in range(1, 101):
        items = next(bt)
        Bold = L.B.clone(); W0 = m.rnn.weight_hh_l0.detach().clone()
        L.update(items)
        W1 = m.rnn.weight_hh_l0.detach().clone(); Bnew = L.B
        if u in (1, 10, 100):
            contr = float(((Bnew - W1) - (1 - DECAY) * (Bold - W0)).abs().max())
            ratio = float((Bnew - W1).norm() / (Bold - W0).norm())
            rows.append(dict(update=u, contraction_max_abs=contr, norm_ratio=ratio, expected_ratio=1 - DECAY, **bhh_measures(Bnew, W1)))
    res["contraction"] = dict(rows=rows, ok=all(r["contraction_max_abs"] <= 1e-7 for r in rows),
                              norm_ratio_within_1e_5=all(abs(r["norm_ratio"] - (1 - DECAY)) <= 1e-5 for r in rows))
    res["ok"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
    return res


def check3():
    """E36's check 4: from arch26.new_net(73129) with make_opt, 2,048 updates on the birth's stream reproduce E26's dry births (S and B)."""
    arch26.install()
    seed = 73129; res = {}
    for rule, f in (("S", "e26_dev_S_dry2048_73129.json"), ("B", "e26_dev_B_dry2048_73129.json")):
        want = json.loads((HERE / "exp45" / "e26_qual" / f).read_text())["tensors_sha256"]
        T.manual_seed(seed)
        m = arch26.new_net(seed); L = Learner(m, rule)
        bt = E.batches([(t, E.split(t, .5, seed)[0]) for t in E.DEV], seed, 12)
        t0 = time.process_time()
        for _ in range(2048):
            L.update(next(bt))
        got = R.tensors_sha(m.state_dict())
        res[rule] = dict(got=got, want=want, ok=got == want, cpu=time.process_time() - t0)
        print("check3", rule, got[:8], want[:8], got == want, flush=True)
    res["ok"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
    return res


def check5():
    """No presented item is a held-out (task, a, b); no code of F4 in any earlier family. Over the first 8,192 batches of every seed
    (the whole stream of every run: the batches are the same for every run of a seed, check 2)."""
    res = {}
    for seed in SEEDS:
        pr = Presented(seed); bt = batches(seed)
        for _ in range(8192):
            pr.add(next(bt))
        tp = {t[0]: len(split(t, seed)[0]) for t in F4}
        res[seed] = dict(pr.value(), taught_pairs=tp, ok=pr.held_hits == 0 and pr.not_taught == 0 and pr.earlier_codes == 0)
    codes = dict(F4=[t[2].decode() for t in F4], earlier=sorted({t[2].decode() for t in _EARLIER} | {"c"}), moduli_earlier=sorted({t[1] for t in _EARLIER}))
    return dict(seeds=res, codes=codes, ok=all(v["ok"] for v in res.values()))


def check7():
    """At row 0 the book rows of e f g h are bit-equal in all five networks of a seed; every score of a fresh net is 0."""
    arch26.install()
    res = {}
    for seed in SEEDS:
        rows = {}
        for kind, rule in (("lived", "S"), ("lived", "B"), ("lived", "K"), ("fresh", "S"), ("fresh", "B")):
            m, _, _ = start(kind, rule, seed)
            rows[f"{kind}_{rule}"] = m.book.weight.detach()[[101, 102, 103, 104]].clone()
        base = rows["fresh_S"]
        eq = {k: bool(T.equal(v, base)) for k, v in rows.items()}
        sc = Scorer(seed)
        m = arch26.new_net(seed)          # fresh S and fresh B are the same network before any update
        s = sc(m)
        exacts = [v["exact"] for v in s["held_k12"].values()] + [v["exact"] for v in s["taught_k12"].values()] + \
                 [w["exact"] for v in s["held_plain_k0"].values() for w in v.values()] + [v["exact"] for v in s["earlier_k12"].values()]
        res[seed] = dict(efgh_equal_to_fresh=eq, fresh_scores_exact=exacts, ok=all(eq.values()) and not any(exacts))
        print("check7", seed, res[seed]["ok"], flush=True)
    return dict(seeds=res, ok=all(v["ok"] for v in res.values()))


# The reviewer's row-0 table (AUDIT_E40_NEW_FAMILY_FROM_A_TENTH_20261115.md, section 3.1, on claude/vibrant-cerf-sje726 at 089409fe):
# held-out at 12 letters add, sub, max, min | taught at 12 letters add, sub, max, min | held-out no letters: add no-red, add red, sub no-red, sub red
AUDIT_ROW0 = {
    ("S", 73129): (17.0, 29.7, 52.3, 34.0, 19.3, 29.0, 48.7, 36.0, 16.7, 12.2, 0.0, 0.0),
    ("S", 93683): (11.3, 19.3, 31.0, 63.3, 7.7, 23.3, 24.3, 60.0, 12.5, 3.9, 2.9, 0.4),
    ("S", 116089): (46.7, 16.0, 85.3, 99.0, 45.7, 14.3, 86.7, 97.0, 88.8, 0.0, 1.6, 0.3),
    ("B", 73129): (37.3, 48.3, 41.3, 42.3, 31.7, 50.3, 43.3, 38.3, 72.6, 0.0, 52.9, 8.8),
    ("B", 93683): (13.3, 47.0, 95.0, 97.3, 9.7, 50.0, 93.0, 98.3, 1.0, 0.8, 97.6, 0.1),
    ("B", 116089): (46.0, 57.3, 95.0, 99.3, 46.0, 60.7, 96.0, 99.0, 89.8, 0.1, 92.1, 23.1),
    ("K", 73129): (30.0, 42.3, 53.0, 37.0, 33.0, 48.0, 48.3, 40.0, 57.0, 14.2, 14.5, 3.1),
    ("K", 93683): (10.3, 36.7, 79.7, 81.7, 14.3, 42.3, 78.7, 85.3, 6.0, 4.0, 48.0, 1.4),
    ("K", 116089): (55.0, 26.7, 26.0, 94.3, 55.0, 26.0, 30.7, 95.0, 35.8, 14.3, 0.0, 0.0),
}


def row0_vector(s):
    pct = lambda d: 100.0 * d["exact"] / d["n"]
    return tuple([pct(s["held_k12"][n]) for n in F4N] + [pct(s["taught_k12"][n]) for n in F4N] +
                 [pct(s["held_plain_k0"][n][w]) for n in ("add83", "sub83") for w in ("no_reduction", "reduction")])


def check8():
    """Scores 1, 2, 3 at row 0 of the nine lived runs equal the reviewer's table to one decimal."""
    arch26.install()
    res = {}
    for seed in SEEDS:
        sc = Scorer(seed)
        for rule in ("S", "B", "K"):
            m, _, _ = start("lived", rule, seed)
            got = row0_vector(sc(m))
            want = AUDIT_ROW0[(rule, seed)]
            r1 = tuple(round(v, 1) for v in got)
            diff = [i for i, (g, w) in enumerate(zip(r1, want)) if abs(g - w) > 1e-9]
            res[f"{rule}_{seed}"] = dict(got=got, got_1dp=r1, audit=want, differing_columns=diff, ok=not diff)
            print("check8", rule, seed, "ok" if not diff else f"DIFFER {diff} got {r1} audit {want}", flush=True)
    return dict(rows=res, columns="held12 add sub max min | taught12 add sub max min | plain add no-red, add red, sub no-red, sub red",
                ok=all(v["ok"] for v in res.values()))


if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])

    def opt_(flag, default=None):
        return a[a.index(flag) + 1] if flag in a else default

    out = Path(opt_("--out", str(OUT)))
    if a[0] == "run":
        assert a[1] in ("lived", "fresh") and a[2] in ("S", "B", "K")
        run(a[1], a[2], int(a[3]), int(opt_("--updates", "8192")), out, cmd)
    elif a[0] == "check":
        t0 = time.process_time()
        part = a[1]
        fns = {"1": (check1, "check1"), "3": (check3, "check3"), "5": (check5, "check5"), "7": (check7, "check7"), "8": (check8, "check8")}
        if part == "2":
            res = check2(a[2]); name = f"check2_{Path(a[2]).name}"
        elif part == "4":
            res = check4(int(a[2])); name = f"check4_{a[2]}"
        elif part in fns:
            res = fns[part][0](); name = fns[part][1]
        else:
            raise SystemExit(__doc__)
        p = out / "checks" / f"E40_{name}.json"
        R.refuse_existing(p); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(check=name, command=cmd, provenance=provenance(cmd), cpu_seconds=time.process_time() - t0, result=res), indent=1, default=str))
        print(name, "pass" if res.get("ok") else "FAIL", f"cpu {time.process_time() - t0:.0f}s", flush=True)
    else:
        raise SystemExit(__doc__)
