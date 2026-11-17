"""E58 runner (brief TASK_E58_A_GATED_TERM_AGAINST_FREE_LEARNING_DECAY_AND_NUMBER_OF_PAIRS_20261117.md, revision 2): E40's new family
(mod 83, codes e f g h) learned freely or in E48's gated term of rank 16, from a taught fraction f of its pairs, for 8,192 updates.
A new file; every existing source stays as it is. e40_new_family.py (the start from a lived network, the stream, the Scorer, the record
of presented items) and exp48_gated_run.py with e48_gated.py (the gated forward and its update; byte-for-byte copies from
origin/claude/sweet-turing-lwi3no, see E58_COPIED_SOURCES.txt) are imported, not rewritten.

Arms:
    free         every weight; AdamW, decay 0.25, byte table 0 (E40's `lived` run again, with more rows)
    free-again   as free, the stream X16.entry_stream(e40_new_family.entries(seed), seed + 141)
    free-wd0     every weight; decay 0 on every tensor
    free-anchor  every weight; the optimizer of free-wd0; after each update p.sub_(p - p0, alpha=0.00025) on every tensor but the byte table
    term         E48's gated term of rank 16 and the four code rows; decay 0.25 on the term's factors
    term-wd0     as term; decay 0

    python3 exp58_new_family.py run <arm> <S|B> <05|10|20> <seed> [--updates N] [--out DIR]
    python3 exp58_new_family.py check 5                       the fractions (no network); writes the batch digests every run asserts
    python3 exp58_new_family.py check <3|4|8> <S|B>           seed 73129, one update
    python3 exp58_new_family.py check <1|2|6|7>               read from the run receipts in exp45/e58
"""
import json
import subprocess
import sys
import time
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C          # noqa: E402  (imports exp20_reference, which applies the environment gate)
import torch as T                    # noqa: E402
import numpy                         # noqa: E402
import arch26                        # noqa: E402
import e40_new_family as E40         # noqa: E402
import exp48_gated_run as X48        # noqa: E402
import e48_gated as G                # noqa: E402

E, X, X16, R = C.E, C.X, C.X16, C.R
OUT = HERE / "exp45" / "e58"
QUAL = OUT / "qual"
ROWS = (0, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192)
SEEDS = E40.SEEDS
F4 = E40.F4
EARLIER6 = E40.EARLIER6
FREE_ARMS = ("free", "free-again", "free-wd0", "free-anchor")
TERM_ARMS = ("term", "term-wd0")
FRACTIONS = {"05": 0.05, "10": 0.10, "20": 0.20}
TAUGHT_N = {0.05: 344, 0.10: 689, 0.20: 1378}
ALPHA = 0.00025                      # the decay toward the start, an update
RANK = 16
TERM_FILE_ROWS = (1024, 8192)
WEIGHT_ARMS = ("free", "free-wd0", "free-anchor")
OWN = ("exp58_new_family.py", "exp48_gated_run.py", "e48_gated.py", "e40_new_family.py")
EXPECTED_SHA = {"exp48_gated_run.py": "f4f03b01ad5dba650f1d8304ad8547baeacf8279126332cc466e2c66d8d131ac",
                "e48_gated.py": "ab324e9ce4f55b01fa8d8f802b4a1622990c73a70d974727a1f19916e2ccb1a0",
                "e40_new_family.py": "aa11c27f1588abbac6659afc124980b88994f6fbae375ffc742f0d31dbdda9cf"}
W8 = "origin/claude/sweet-turing-lwi3no"


def provenance(cmd):
    p = E40.provenance(cmd)
    p["e58_sources_sha256"] = {s: X16.sha_file(HERE / s) for s in OWN}
    for s, want in EXPECTED_SHA.items():
        assert p["e58_sources_sha256"][s] == want, f"{s} is not the file the brief names"
    p["versions"] = dict(python=sys.version.split()[0], torch=T.__version__, numpy=numpy.__version__)
    return p


# ---------------------------------------------------------------- the fraction
def nested(t, f, seed):
    taught, rest = E.split(t, 0.10, seed)     # E40's shuffled list of the 6,889 pairs, cut at 689
    pairs = taught + rest
    k = round(f * len(pairs))
    return pairs[:k], pairs[k:]


def set_fraction(f):
    """The one replacement of e40_new_family.split, before the stream, the Scorer and the record of presented items are built."""
    assert f in TAUGHT_N
    E40.split = lambda t, seed: nested(t, f, seed)


def fraction_assertions(f, seed, entries, sc, pr):
    """(a) to (d) of the brief's "The fraction"; returns what was counted."""
    taught = {t[0]: nested(t, f, seed)[0] for t in F4}
    n = {op: len(v) for op, v in taught.items()}
    assert all(v == TAUGHT_N[f] for v in n.values()), n                                              # (a)
    assert all(len(E40.split(t, seed)[0]) == TAUGHT_N[f] for t in F4)
    assert entries == [("op", t, a, b) for t in F4 for a, b in taught[t[0]]], "the stream's entries are not the taught pairs"   # (b)
    assert pr.taught == {(t[0], a, b) for t in F4 for a, b in taught[t[0]]}
    plain = {nm: len(sc.plain[nm]["reduction"]) + len(sc.plain[nm]["no_reduction"]) for nm in ("add83", "sub83")}
    assert all(v == 6889 - TAUGHT_N[f] for v in plain.values()), plain                               # (c)
    if f == 0.10:
        assert all(taught[t[0]] == E.split(t, 0.10, seed)[0] for t in F4)                            # (d)
    return dict(taught_pairs=n, stream_entries=len(entries), untaught_plain_items=plain, d_checked=f == 0.10)


# ---------------------------------------------------------------- the common pairs
def common_pairs(t, seed):
    """Pairs of add or sub that need the reduction, are untaught at f = 0.20, and whose two operand values each occur in the same
    position in at least one pair taught at f = 0.05."""
    t05 = nested(t, 0.05, seed)[0]
    first = {a for a, _ in t05}; second = {b for _, b in t05}
    return [(a, b) for a, b in nested(t, 0.20, seed)[1] if E40.needs_reduction(t[0], a, b) and a in first and b in second]


def common_items(seed):
    return {f"{t[0]}83": [E.render(t, a, b, 0, Random(0)) for a, b in common_pairs(t, seed)] for t in F4[:2]}


def common_score(m, items):
    c = {n: X16.score(m, it, None) for n, it in items.items()}
    return dict(counts=c, common_score=sum(100.0 * v["exact"] / v["n"] for v in c.values()) / len(c))


# ---------------------------------------------------------------- the free arms
def free_opt(m, arm):
    wd = C.WD if arm in ("free", "free-again") else 0.0
    return X16.adamw16(m, C.LR, wd, ())


def lived_copy(m):
    """A copy of every tensor outside the byte table."""
    assert {n for n, _ in m.named_parameters()} == set(m.state_dict())
    return {n: p.detach().clone() for n, p in m.named_parameters() if n != "book.weight"}


def anchor(m, p0):
    """The decay toward the start: every tensor outside the byte table moves toward its lived value by 0.00025 of its distance."""
    with T.no_grad():
        for n, p in m.named_parameters():
            if n != "book.weight":
                p.sub_(p - p0[n], alpha=ALPHA)


def norms(m, p0):
    return {n: dict(norm=float(p.detach().double().norm()), diff_from_lived=float((p.detach().double() - p0[n].double()).norm()))
            for n, p in m.named_parameters() if n != "book.weight"}


def term_opt_record(m, term, opt):
    names = {id(m.book.weight): "book.weight"}
    names.update({id(getattr(term, k)): f"term.{k}" for k in ("down_i", "up_i", "down_h", "up_h")})
    return [dict(weight_decay=g["weight_decay"], lr=g["lr"], betas=list(g["betas"]), params=[names[id(p)] for p in g["params"]]) for g in opt.param_groups]


def digests_expected(ftag, seed):
    d = json.loads((QUAL / "E58_check5.json").read_text())["result"]["digests"][ftag][str(seed)]
    return {int(k): v for k, v in d.items()}


def e40_receipt(rule, seed):
    return {r["updates"]: r for r in json.loads((HERE / "exp45" / "e40" / f"e40_lived_{rule}_{seed}.json").read_text())["rows"]}


# ---------------------------------------------------------------- one run
def run(arm, rule, ftag, seed, updates, out, cmd):
    assert arm in FREE_ARMS + TERM_ARMS and rule in ("S", "B") and seed in SEEDS
    f = FRACTIONS[ftag]
    is_term = arm in TERM_ARMS
    if is_term:
        X48.install()
    else:
        arch26.install()
    out = Path(out)
    name = f"e58_{arm}_{rule}_f{ftag}_{seed}"
    rows = [r for r in ROWS if r <= updates]
    if updates not in rows:
        rows.append(updates)
    rec_path = out / f"{name}.json"
    files = {}
    if is_term:
        files = {r: out / f"{name}_row{r}.pt" for r in rows if r in TERM_FILE_ROWS}
    elif arm in WEIGHT_ARMS:
        files = {rows[-1]: out / f"{name}_row{rows[-1]}.pt"}
    R.refuse_existing(rec_path, str(rec_path) + ".tmp", *files.values(), *[str(p) + ".tmp" for p in files.values()])
    out.mkdir(parents=True, exist_ok=True)
    local = rule == "S"
    set_fraction(f)
    T.manual_seed(seed)
    m, _, info = E40.start("lived", rule, seed)
    assert info["tensors_sha_matches_receipt"]
    p0 = lived_copy(m)
    start_codes = m.book.weight.detach()[list(G.CODES)].clone()
    if is_term:
        term = G.Term(seed, RANK); X48.STATE["term"] = term; X48.STATE["force_off"] = False
        G.freeze(m); opt = G.make_opt(m, term, wd=0.25 if arm == "term" else 0.0)
        opt_rec = term_opt_record(m, term, opt)
        update = f"exp48_gated_run.step (e48_gated.forward with the gate; keep_code_rows; local={local})"
    else:
        term = None
        opt = free_opt(m, arm)
        opt_rec = R.opt_groups(m, opt)
        update = f"exp26_candidate.step(local={local})" + ("; then exp58_new_family.anchor (p.sub_(p - p0, alpha=0.00025), byte table excluded)" if arm == "free-anchor" else "")
    sc = E40.Scorer(seed)
    entries = E40.entries(seed)
    if arm == "free-again":
        def batches():
            st = X16.entry_stream(entries, seed + 141)
            while True:
                yield [next(st) for _ in range(X.BATCH)]
        bt = batches()
        data = "64 consecutive items of X16.entry_stream(e40_new_family.entries(seed), seed + 141)"
    else:
        bt = E40.batches(seed)
        data = "e40_new_family.batches(seed) on the nested taught pairs: 64 consecutive items of X16.entry_stream(taught pairs of F4, seed + 140)"
    dg = C.Digest(); pr = E40.Presented(seed)
    fr = fraction_assertions(f, seed, entries, sc, pr)
    com = common_items(seed)
    want_digest = None if arm == "free-again" else digests_expected(ftag, seed)
    e40 = e40_receipt(rule, seed) if (arm == "free" and ftag == "10") else None
    rec = dict(experiment="E58", arm=arm, rule=rule, seed=seed, fraction=f, updates=updates, rows_planned=rows, start=info,
               fraction_assertions=fr, common_pairs={n: len(v) for n, v in com.items()},
               book_rows_efgh_tensors_sha256=R.tensors_sha({"efgh": m.book.weight.detach()[[101, 102, 103, 104]]}),
               optimizer=dict(read_back_groups=opt_rec), data=data, update=update,
               provenance=provenance(cmd), rows=[], complete=False)
    if is_term:
        rec.update(rank=RANK, numbers_that_may_change=term.count(), gate="on iff x[:, 1] in 101..104, asserted equal for all rows of a call",
                   trainable=[n for n, q in m.named_parameters() if q.requires_grad] + ["term.down_i", "term.up_i", "term.down_h", "term.up_h"])
    t0 = time.process_time(); cpu_score = 0.0; losses = []; u = 0; grads = []
    row0_digest = None
    for r in rows:
        while u < r:
            items = next(bt); dg.add(items); pr.add(items)
            if is_term:
                losses.append(X48.step(m, term, opt, items, local, grads if u < 2 else None))
            else:
                losses.append(C.step(m, opt, items, local))
                if arm == "free-anchor":
                    anchor(m, p0)
            u += 1
        cb = time.process_time()
        if is_term:
            s, ans = X48.score(sc, m)
            ed = X48.earlier_digests(ans)
        else:
            s = sc(m)
        cs = common_score(m, com)
        cpu_score += time.process_time() - cb
        row = dict(updates=r, scores=s, common=cs, loss_since_last_row=(sum(losses) / len(losses)) if losses else None,
                   finite_losses=all(v == v and abs(v) != float("inf") for v in losses), data_digest=dg.value(), presented=pr.value(),
                   tensors_sha256=R.tensors_sha(m.state_dict()), norms=norms(m, p0), optimizer_steps=R.opt_steps(opt))
        if want_digest is not None:
            row["digest_equals_check5"] = dg.value()["sha256"] == want_digest[r]
            assert row["digest_equals_check5"], f"{name}: the batch digest at row {r} is not check 5's"
        if is_term:
            if r == 0:
                row0_digest = ed["all"]
            lived_sha = X48.lived_sha_with_codes_reset(m, start_codes)
            row.update(earlier_answers_sha256=ed,
                       check3=dict(earlier_digest_equals_row0=ed["all"] == row0_digest,
                                   earlier_counts={n: s["earlier_k12"][n]["exact"] for n in EARLIER6},
                                   lived_tensors_sha256_codes_reset=lived_sha, equals_life_end=lived_sha == info["receipt_tensors_sha256"]),
                       code_rows_tensors_sha256=R.tensors_sha({"code_rows": m.book.weight.detach()[list(G.CODES)]}),
                       term_norms={k: float(getattr(term, k).detach().norm()) for k in ("down_i", "up_i", "down_h", "up_h")},
                       sweep_cosines=X48.cosines(term) if local else None)
            if r in files:
                fsha, tsha = X48.save_term(m, term, files[r])
                row["term_file"] = dict(file=files[r].name, sha256=fsha, tensors_sha256=tsha, bytes=files[r].stat().st_size)
            if len(grads) >= 2:
                rec["grads_updates_1_2"] = grads[:2]
        elif r in files:
            row["weights"] = files[r].name; row["weights_sha256"] = R.save_weights(m, files[r])
        if e40 is not None and r in e40:
            w = e40[r]
            row["e40_compare"] = dict(digest_equal=row["data_digest"] == w["data_digest"], scores_equal=s == w["scores"],
                                      tensors_equal=row["tensors_sha256"] == w["tensors_sha256"])
        row.update(cpu=time.process_time() - t0, cpu_scoring_so_far=cpu_score)
        losses = []
        rec["rows"].append(row)
        sm = s["summary"]
        print(name, r, f"held {sm['held_add_sub_mean']:.1f} taught {sm['taught_add_sub_mean']:.1f} red {sm['reduction_score']:.1f} common {cs['common_score']:.1f}",
              f"cpu {row['cpu']:.0f}s", flush=True)
        X16.write_json(rec_path, rec)
        if is_term:      # check 6: asserted at every row; a failure stops the run
            assert row["check3"]["earlier_digest_equals_row0"], f"{name}: the earlier families' answers changed at row {r}"
            assert row["check3"]["equals_life_end"], f"{name}: a lived tensor changed at row {r}"
        if "e40_compare" in row:   # check 1: a difference stops the run
            assert all(row["e40_compare"].values()), f"{name}: row {r} is not E40's: {row['e40_compare']}"
    assert pr.held_hits == 0 and pr.not_taught == 0 and pr.earlier_codes == 0, pr.value()
    if is_term:
        rec["gate_calls"] = dict(calls=X48.STATE["calls"], gate_on=X48.STATE["on_calls"])
    rec.update(complete=True, cpu_total=time.process_time() - t0, cpu_scoring=cpu_score, cpu_updates=time.process_time() - t0 - cpu_score)
    X16.write_json(rec_path, rec)
    print(name, "complete", flush=True)


# ---------------------------------------------------------------- checks before the runs
def maxdev(a, b):
    return float((a.double() - b.double()).abs().max())


def check3(rule, seed=73129):
    """free-wd0 differs from free in the decay only: after the first update, tensor(free-wd0) - tensor(free) = 0.00025 x start."""
    arch26.install(); set_fraction(0.10)
    local = rule == "S"
    items = next(E40.batches(seed))
    nets = {}
    for arm in ("free", "free-wd0"):
        T.manual_seed(seed)
        m, _, info = E40.start("lived", rule, seed)
        start = {n: p.detach().clone() for n, p in m.named_parameters()}
        opt = free_opt(m, arm)
        C.step(m, opt, items, local)
        nets[arm] = (m, opt, start)
    (ma, oa, sa), (mb, ob, sb) = nets["free"], nets["free-wd0"]
    assert all(T.equal(sa[n], sb[n]) for n in sa)
    pa, pb = dict(ma.named_parameters()), dict(mb.named_parameters())
    per = {}
    for n in sa:
        if n == "book.weight":
            continue
        d = pb[n].detach().double() - pa[n].detach().double()
        per[n] = dict(max_abs_deviation=float((d - ALPHA * sa[n].double()).abs().max()),
                      deviation_if_no_decay=float((ALPHA * sa[n].double()).abs().max()),
                      changed_by_the_update=not T.equal(pa[n].detach(), sa[n]))
    groups = R.opt_groups(mb, ob)
    res = dict(rule=rule, seed=seed, tensors=per, worst=max(v["max_abs_deviation"] for v in per.values()),
               smallest_deviation_if_no_decay=min(v["deviation_if_no_decay"] for v in per.values()),
               byte_table_bit_equal=bool(T.equal(pa["book.weight"].detach(), pb["book.weight"].detach())),
               byte_table_changed_by_the_update=not T.equal(pa["book.weight"].detach(), sa["book.weight"]),
               optimizer_free_wd0=groups, optimizer_free=R.opt_groups(ma, oa))
    res["ok"] = (res["worst"] < 1e-6 and res["byte_table_bit_equal"] and all(g["weight_decay"] == 0 for g in groups)
                 and all(v["changed_by_the_update"] for v in per.values()))
    return res


def check4(rule, seed=73129):
    """free-anchor: (a) on a doubled copy each tensor is left at 1.99975 x lived, the byte table bit-equal; (b) on the lived network
    nothing changes; (c) the optimizer read back has decay 0 in every group."""
    arch26.install()
    m, _, info = E40.start("lived", rule, seed)
    p0 = lived_copy(m)
    full0 = {n: p.detach().clone() for n, p in m.named_parameters()}
    anchor(m, p0)
    b = {n: bool(T.equal(p.detach(), full0[n])) for n, p in m.named_parameters()}
    with T.no_grad():
        for n, p in m.named_parameters():
            if n != "book.weight":
                p.mul_(2.0)
    assert all(T.equal(p.detach(), 2 * p0[n]) for n, p in m.named_parameters() if n != "book.weight")
    anchor(m, p0)
    a = {n: dict(max_abs_deviation=maxdev(p.detach(), 1.99975 * p0[n].double()),
                 deviation_of_the_standing_decay=float((ALPHA * p0[n].double()).abs().max()))
         for n, p in m.named_parameters() if n != "book.weight"}
    book_equal = bool(T.equal(m.book.weight.detach(), full0["book.weight"]))
    m2, _, _ = E40.start("lived", rule, seed)
    opt = free_opt(m2, "free-anchor")
    groups = R.opt_groups(m2, opt)
    res = dict(rule=rule, seed=seed, a=dict(tensors=a, worst=max(v["max_abs_deviation"] for v in a.values()),
                                            smallest_deviation_of_the_standing_decay=min(v["deviation_of_the_standing_decay"] for v in a.values()),
                                            byte_table_bit_equal=book_equal),
               b=dict(unchanged=b, all_unchanged=all(b.values())), c=dict(optimizer=groups))
    res["ok"] = res["a"]["worst"] < 1e-6 and book_equal and all(b.values()) and all(g["weight_decay"] == 0 for g in groups)
    return res


def check8(rule, seed=73129):
    """term-wd0 differs from term in the decay only, after the first update."""
    X48.install(); set_fraction(0.10)
    local = rule == "S"
    items = next(E40.batches(seed))
    got = {}
    for arm in ("term", "term-wd0"):
        T.manual_seed(seed)
        m, _, info = E40.start("lived", rule, seed)
        term = G.Term(seed, RANK); X48.STATE["term"] = term; X48.STATE["force_off"] = False
        start = X48.term_tensors(m, term)
        G.freeze(m); opt = G.make_opt(m, term, wd=0.25 if arm == "term" else 0.0)
        grads = []
        X48.step(m, term, opt, items, local, grads)
        got[arm] = dict(start=start, after=X48.term_tensors(m, term), opt=term_opt_record(m, term, opt), grads=grads[0])
    a, b = got["term"], got["term-wd0"]
    assert all(T.equal(a["start"][k], b["start"][k]) for k in a["start"])
    res = dict(rule=rule, seed=seed,
               down_wd0_bit_equal_to_start={k: bool(T.equal(b["after"][k], b["start"][k])) for k in ("down_i", "down_h")},
               down_term_max_abs_deviation_from_0_99975_start={k: maxdev(a["after"][k], 0.99975 * a["start"][k].double()) for k in ("down_i", "down_h")},
               down_term_differs_from_start={k: not T.equal(a["after"][k], a["start"][k]) for k in ("down_i", "down_h")},
               bit_equal_in_the_two_arms={k: bool(T.equal(a["after"][k], b["after"][k])) for k in ("up_i", "up_h", "code_rows")},
               changed_by_the_update={k: not T.equal(a["after"][k], a["start"][k]) for k in ("up_i", "up_h", "code_rows")},
               first_update_gradients=dict(term=a["grads"], term_wd0=b["grads"]),
               optimizer_term=a["opt"], optimizer_term_wd0=b["opt"])
    res["ok"] = (all(res["down_wd0_bit_equal_to_start"].values()) and all(v < 1e-7 for v in res["down_term_max_abs_deviation_from_0_99975_start"].values())
                 and all(res["down_term_differs_from_start"].values()) and all(res["bit_equal_in_the_two_arms"].values())
                 and all(res["changed_by_the_update"].values())
                 and [g["weight_decay"] for g in b["opt"]] == [0.0, 0.0] and [g["weight_decay"] for g in a["opt"]] == [0.25, 0.0])
    return res


def check5():
    """The fractions, without a network."""
    res = dict(fractions={}, digests={}, common_pairs={})
    ok = True
    e40_rows = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
    for seed in SEEDS:
        res["common_pairs"][str(seed)] = {f"{t[0]}83": len(common_pairs(t, seed)) for t in F4[:2]}
        for t in F4:
            a, b, c = (nested(t, f, seed)[0] for f in (0.05, 0.10, 0.20))
            nest = a == b[:344] and b == c[:689] and set(a) < set(b) < set(c) and (len(a), len(b), len(c)) == (344, 689, 1378)
            assert nest, (seed, t)
    for ftag, f in FRACTIONS.items():
        res["fractions"][ftag] = {}; res["digests"][ftag] = {}
        for seed in SEEDS:
            set_fraction(f)
            sc = E40.Scorer(seed); entries = E40.entries(seed); pr = E40.Presented(seed)
            fr = fraction_assertions(f, seed, entries, sc, pr)
            taught = {t[0]: set(nested(t, f, seed)[0]) for t in F4}            # by nested directly, not through e40_new_family.split
            bad_untaught = 0; bad_taught = 0; n_untaught = 0; n_taught = 0
            panels_untaught = [it for v in sc.held.values() for it in v] + [it for v in sc.plain.values() for w in v.values() for it in w]
            for p, _ in panels_untaught:
                t, a, b = E40.parse(p); n_untaught += 1; bad_untaught += (a, b) in taught[t[0]]
            for v in sc.taught.values():
                for p, _ in v:
                    t, a, b = E40.parse(p); n_taught += 1; bad_taught += (a, b) not in taught[t[0]]
            bt = E40.batches(seed); dg = C.Digest(); seen = set(); dig = {}; u = 0
            for r in ROWS:
                while u < r:
                    items = next(bt); dg.add(items); pr.add(items)
                    for p, _ in items:
                        t, a, b = E40.parse(p); seen.add((t[0], a, b))
                    u += 1
                dig[str(r)] = dg.value()["sha256"]
            v = pr.value()
            all_taught = {(op, a, b) for op, s_ in taught.items() for a, b in s_}
            row = dict(fr, untaught_panel_items=n_untaught, untaught_panel_items_with_a_taught_pair=bad_untaught,
                       taught_panel_items=n_taught, taught_panel_items_with_an_untaught_pair=bad_taught, presented=v,
                       taught_pairs_never_presented=len(all_taught - seen), presented_pairs_not_taught=len(seen - all_taught))
            if f == 0.10:
                row["digests_equal_e40"] = {}
                for rule in ("S", "B"):
                    w = e40_receipt(rule, seed)
                    row["digests_equal_e40"][rule] = all(dig[str(r)] == w[r]["data_digest"]["sha256"] for r in e40_rows)
            row["ok"] = (bad_untaught == 0 and bad_taught == 0 and v["held_out_presented"] == 0 and v["not_taught_presented"] == 0
                         and v["earlier_code_items"] == 0 and v["items"] == 8192 * 64 and row["taught_pairs_never_presented"] == 0
                         and row["presented_pairs_not_taught"] == 0 and all(row.get("digests_equal_e40", {"x": True}).values()))
            ok = ok and row["ok"]
            res["fractions"][ftag][str(seed)] = row; res["digests"][ftag][str(seed)] = dig
            print("check5", ftag, seed, row["ok"], flush=True)
    res["nested_344_in_689_in_1378"] = True
    res["ok"] = ok
    return res


# ---------------------------------------------------------------- checks read from the runs
def load(arm, rule, ftag, seed, d=OUT):
    return json.loads((Path(d) / f"e58_{arm}_{rule}_f{ftag}_{seed}.json").read_text())


def git_show(path):
    return subprocess.run(["git", "show", f"{W8}:MOD97_LOCAL_RULE_20261106/{path}"], cwd=HERE, capture_output=True, check=True).stdout


def check1(seed=73129):
    """free is E40's run: at each of E40's rows the batch digest, every count of every score and tensors_sha256 are E40's."""
    res = {}
    for rule in ("S", "B"):
        rows = {r["updates"]: r for r in load("free", rule, "10", seed)["rows"]}
        w = e40_receipt(rule, seed)
        per = {}
        for u in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
            if u in rows:
                per[u] = dict(digest_equal=rows[u]["data_digest"] == w[u]["data_digest"], scores_equal=rows[u]["scores"] == w[u]["scores"],
                              tensors_equal=rows[u]["tensors_sha256"] == w[u]["tensors_sha256"],
                              loss_equal=rows[u]["loss_since_last_row"] == w[u]["loss_since_last_row"] if u not in (64, 4096, 8192) else None)
        res[rule] = dict(run=f"e58_free_{rule}_f10_{seed}", against=f"exp45/e40/e40_lived_{rule}_{seed}.json", rows=per,
                         ok=bool(per) and all(v["digest_equal"] and v["scores_equal"] and v["tensors_equal"] for v in per.values()))
    return dict(seed=seed, rules=res, rows_compared={k: sorted(v["rows"]) for k, v in res.items()}, ok=all(v["ok"] for v in res.values()))


def check2(seed=73129):
    """term is E48's run: at each of E48's rows every count of every score, the batch digest and the digests of the earlier families'
    answers are E48's; the five tensors of the term at row 1,024 are bit-equal to E48's term file."""
    import io
    res = {}
    for rule in ("S", "B"):
        rows = {r["updates"]: r for r in load("term", rule, "10", seed)["rows"]}
        w = {r["updates"]: r for r in json.loads(git_show(f"exp45/e48/e48_{rule}_{seed}_r16.json"))["rows"]}
        per = {}
        for u in (0, 16, 32, 64, 128, 256, 512, 1024):
            if u in rows:
                per[u] = dict(scores_equal=rows[u]["scores"] == w[u]["scores"], digest_equal=rows[u]["data_digest"] == w[u]["data_digest"],
                              earlier_answers_equal=rows[u]["earlier_answers_sha256"] == w[u]["earlier_answers_sha256"])
        tf = None
        if 1024 in rows:
            theirs = T.load(io.BytesIO(git_show(f"exp45/e48/e48_{rule}_{seed}_r16_row1024.pt")))
            mine = T.load(OUT / rows[1024]["term_file"]["file"])
            tf = dict(tensors={k: bool(T.equal(mine[k], theirs[k])) for k in ("up_i", "down_i", "up_h", "down_h", "code_rows")},
                      keys_equal=sorted(mine) == sorted(theirs),
                      tensors_sha256_equal=rows[1024]["term_file"]["tensors_sha256"] == w[1024]["term_file"]["tensors_sha256"])
        res[rule] = dict(run=f"e58_term_{rule}_f10_{seed}", against=f"{W8}:MOD97_LOCAL_RULE_20261106/exp45/e48/e48_{rule}_{seed}_r16.json",
                         rows=per, term_file_row_1024=tf,
                         ok=bool(per) and all(all(v.values()) for v in per.values())
                         and (tf is None or (all(tf["tensors"].values()) and tf["keys_equal"] and tf["tensors_sha256_equal"])))
    return dict(seed=seed, rules=res, rows_compared={k: sorted(v["rows"]) for k, v in res.items()},
                term_file_compared={k: v["term_file_row_1024"] is not None for k, v in res.items()}, ok=all(v["ok"] for v in res.values()))


def check6():
    """The earlier families are unchanged in term and term-wd0, at every row of every such run found."""
    res = {}
    for p in sorted(OUT.glob("e58_term*_f*_*.json")):
        r = json.loads(p.read_text())
        d0 = r["rows"][0]["earlier_answers_sha256"]["all"]
        res[p.stem] = dict(complete=r["complete"], rows=len(r["rows"]),
                           earlier_digest_equals_row0=all(x["earlier_answers_sha256"]["all"] == d0 and x["check3"]["earlier_digest_equals_row0"] for x in r["rows"]),
                           lived_tensors_equal_life_end=all(x["check3"]["equals_life_end"] and
                                                            x["check3"]["lived_tensors_sha256_codes_reset"] == r["start"]["receipt_tensors_sha256"] for x in r["rows"]))
        res[p.stem]["ok"] = res[p.stem]["complete"] and res[p.stem]["earlier_digest_equals_row0"] and res[p.stem]["lived_tensors_equal_life_end"]
    return dict(runs=res, n=len(res), ok=bool(res) and all(v["ok"] for v in res.values()))


def check7(seed=73129):
    """Cost: CPU-seconds an update and a scored row for each arm from Part A's runs of one seed, and a forecast for the whole job."""
    res = {}; per_run = {}
    for arm in FREE_ARMS + TERM_ARMS:
        for rule in ("S", "B"):
            r = load(arm, rule, "10", seed)
            assert r["complete"]
            res[f"{arm}_{rule}"] = dict(cpu_updates=r["cpu_updates"], cpu_scoring=r["cpu_scoring"], cpu_total=r["cpu_total"],
                                        cpu_seconds_an_update=r["cpu_updates"] / r["updates"], cpu_seconds_a_scored_row=r["cpu_scoring"] / len(r["rows"]))
            per_run[(arm, rule)] = r["cpu_total"]
    part_a = 3 * sum(per_run.values())
    part_b = 3 * 2 * sum(per_run[(arm, rule)] for arm in ("free", "free-wd0", "term") for rule in ("S", "B"))
    fc = dict(part_a_cpu_hours=part_a / 3600, part_b_cpu_hours=part_b / 3600, total_cpu_hours=(part_a + part_b) / 3600,
              basis="each run of the other seeds and fractions is taken to cost what the run of the same arm and rule cost on this seed at f = 0.10")
    return dict(seed=seed, runs=res, forecast=fc, limit_cpu_hours=35, ok=fc["total_cpu_hours"] <= 35)


if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])

    def opt_(flag, default=None):
        return a[a.index(flag) + 1] if flag in a else default

    if a[0] == "run":
        run(a[1], a[2], a[3], int(a[4]), int(opt_("--updates", "8192")), Path(opt_("--out", str(OUT))), cmd)
    elif a[0] == "check":
        t0 = time.process_time(); part = a[1]
        if part in ("3", "4", "8"):
            res = {"3": check3, "4": check4, "8": check8}[part](a[2]); name = f"check{part}_{a[2]}"
        elif part in ("1", "2", "5", "6", "7"):
            res = {"1": check1, "2": check2, "5": check5, "6": check6, "7": check7}[part](); name = f"check{part}" + (f"_{a[2]}" if len(a) > 2 else "")
        else:
            raise SystemExit(__doc__)
        p = QUAL / f"E58_{name}.json"
        R.refuse_existing(p); p.parent.mkdir(parents=True, exist_ok=True)
        X16.write_json(p, dict(check=name, command=cmd, provenance=provenance(cmd), cpu_seconds=time.process_time() - t0, result=res))
        print(name, "pass" if res.get("ok") else "FAIL", f"cpu {time.process_time() - t0:.0f}s", flush=True)
    else:
        raise SystemExit(__doc__)
