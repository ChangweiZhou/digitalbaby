"""E59 runner (brief TASK_E59_THE_UNRELATED_TASK_FROM_A_FRESH_A_BORN_AND_A_LIVED_NETWORK_20261117.md, revision 2, on
claude/vibrant-cerf-sje726 at 39057293): E43's unrelated task learned from four starts (fresh, born, lived, born-shuffled) with
2,048, 512 or 256 taught pairs, under the sweep (S) and back-propagation (B), 8,192 updates.

A new file; no existing source is edited. It takes, unchanged: the update exp26_candidate.step; E43's task file e43_task.py (split,
render, panel, digest), its held panel and its assertion that every prompt byte of a batch is 128 or above; E40's loading of a fresh and
of a lived network (e40_new_family.start); and E43's admission of the born networks (exp38_erase.admit, which calls
exp26_candidate.admit_e26 with the one recorded exception for exp16_integrated.py). Scores use X16.score's call (exp11_cell.respond with
X16.CAP); the per-byte counts are taken from the same answers.

    python3 exp59_unrelated_starts.py run <fresh|born|lived|born-shuffled> <n> <S|B> <seed> [--updates N] [--out DIR]
    python3 exp59_unrelated_starts.py check <1|2|3> [--out DIR]

Rows: update 0, every 16 updates to 512, every 32 updates from 544 to 8,192 (273 rows for a full run).
Check 4 (born, 2,048 pairs, every seed) is made inside the run at rows 64, 256, 1,024, 2,048 and 4,096 against E43's receipt
e43_unrelated-decay_<rule>_<seed>.json; a difference writes exp45/e59/STOP and stops the run, and every other run stops at its next row.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp38_erase as E38            # noqa: E402  (imports exp26_candidate, which applies the environment gate; not edited)
import exp43_unrelated as E43        # noqa: E402  (not edited; named for its provenance and its admission line)
import e43_task as TK                # noqa: E402  (not edited)
import e40_new_family as E40         # noqa: E402  (not edited; start("fresh" | "lived", rule, seed))

C, R, X16, E, X, arch26, T = E38.C, E38.R, E38.X16, E38.E, E38.X, E38.arch26, E38.T
T.set_num_threads(1)

SEEDS = (73129, 93683, 116089)
STARTS = ("fresh", "born", "lived", "born-shuffled")
NS = (2048, 512, 256)
RULES = ("S", "B")
UPDATES = 8192
ROWS = [0] + list(range(16, 513, 16)) + list(range(544, UPDATES + 1, 32))
assert len(ROWS) == 273
OUT = X.OUT / "e59"
E43DIR, E40DIR, E26DIR = X.OUT / "e43", X.OUT / "e40", X.OUT / "e26"
CHECK4_ROWS = (64, 256, 1024, 2048, 4096)
OWN = ("exp59_unrelated_starts.py", "exp43_unrelated.py", "e43_task.py", "e40_new_family.py", "exp38_erase.py", "arch31_workerB.py")
EARLIER = ("add89", "sub89", "max89", "min89", "add97", "sub97", "add73", "sub73")   # E43's score 2 and E40's six earlier panels, add89/sub89 once
LIFE_ITEMS, BIRTH_ITEMS = 61440 * 64, 24576 * 64
# the brief's values for check 1 (c), in the order 73129, 93683, 116089
DIGEST_REF = {
    (2048, 8192): ("bbea8212a5c8f1d57cc462f4c5a9992e77a8013ea6692cb06ac801a6f78a5b67", "9c4e76b18ad899a8a463adb2d4b2b9a06ab1210ee54010c8606134afa333c71f",
                   "8760e2485a81c98136b9f06f98a331ceb8581fee2b2f791be70aa3ba3495b70f"),
    (512, 64): ("4dd6f47484f72cfefc5138cc4298e402a234cb745eac606cb379541c63e762ac", "05d69b502bca7e4dc268ba38599609bbf3605e15cff549a7324b28eec2e58e98",
                "8312268753e6452949ab5dae9929de0ba6d3149440e56ba99b58e5e173da16c5"),
    (512, 8192): ("db66d453988b1810e56d00b3195a2166ba6f6d19b9e8b0c5b069abcfbd570d7e", "d18a695db39003851631b5fdc23054e893072663d0dae5422a0fe0fab35963a1",
                  "4c5baac78565f0ee022730a119a3227246a1a750110f8a043060cfd17e22fb25"),
    (256, 64): ("1ad9aaaf93b0ea7d10bce78230ce810a563dcfe6b456dc205d15afaa32dc6694", "f9782a97b7e5f72e105ff95d7e57ac8306743f514e8936be9762735ee68a899b",
                "a49e561f0d6f5a8f143e063b72982d4bd7667e19c6c01a64951d193088f8cf0a"),
    (256, 8192): ("e61f65514d935cf788d3382b430a3e2a637eb1eab6534b9388fee599a2edc49e", "a717d2cb247d57fad3c6bc4e511d7d1a096182795826ca4b2ad6dab2fca6d543",
                  "cee2a5968c59c4b2af3a9850735b2181cd32dd81df1f6b97f734008bcd43e31e"),
}
TASK_DIGEST64 = {73129: "34eefb4442b48399", 93683: "153cafdcda7379d0", 116089: "821bcb2a9ca6dd54"}


# ---------------------------------------------------------------- the task with n taught pairs
def taught_set(seed, n):
    taught, held = TK.split(seed)
    return taught[:n], held


def stream(seed, n):
    """The text of e43_task.stream with taught[:n] in place of taught."""
    taught, _ = taught_set(seed, n); rng = Random(f"e43|{seed}|stream")
    while True:
        order = list(range(len(taught))); rng.shuffle(order)
        for i in order:
            yield TK.render(*taught[i], rng.randint(0, TK.KMAX), rng)


def batches(seed, n, batch=64):
    st = stream(seed, n)
    while True:
        yield [next(st) for _ in range(batch)]


def taught_panel(seed, n):
    taught, _ = taught_set(seed, n); rng = Random(f"e43|{seed}|panel|taught")
    return [TK.render(a, b, 12, rng) for a, b in rng.sample(taught, min(300, n))]


def unseen_flags(seed, n, items):
    """For each item: True if its first symbol never stands first, or its second never second, in the taught pairs."""
    taught, _ = taught_set(seed, n)
    first, second = {a for a, _ in taught}, {b for _, b in taught}
    return [p[1] not in first or p[2] not in second for p, _ in items]


# ---------------------------------------------------------------- starts
def shuffle_in_place(m, seed):
    with T.no_grad():
        for name, t in m.named_parameters():
            if name == "book.weight":
                continue
            perm = list(range(t.numel())); Random(f"e59|{seed}|shuffle|{name}").shuffle(perm)
            t.copy_(t.flatten()[T.tensor(perm)].view_as(t))


def make_start(start, rule, seed):
    """Returns the network and a record of where it came from. Asserts what the brief asks of each start."""
    if start == "fresh":
        m, _, info = E40.start("fresh", rule, seed)
        ref = json.loads((E40DIR / f"e40_fresh_{rule}_{seed}.json").read_text())["start"]["tensors_sha256"]
        info.update(e40_receipt_start_tensors_sha256=ref, equals_e40=info["tensors_sha256"] == ref)
        assert info["equals_e40"], "fresh: tensors_sha256 differs from E40's fresh start"
        return m, info
    if start == "lived":
        m, _, info = E40.start("lived", rule, seed)
        assert info["tensors_sha_matches_receipt"], "lived: tensors_sha256 differs from the life's receipt"
        return m, info
    bfile = C.birth_path(E26DIR, rule, seed)
    m, birth_sha, admission = E38.admit(bfile, rule, seed)
    m.load_state_dict(T.load(bfile))
    tsha = R.tensors_sha(m.state_dict())
    ref = json.loads((E43DIR / f"e43_unrelated-decay_{rule}_{seed}.json").read_text())["settings"]["birth_tensors_sha256"]
    info = dict(start=str(bfile.relative_to(HERE)), weights_sha256=birth_sha, admission=admission, tensors_sha256=tsha,
                e43_birth_tensors_sha256=ref, equals_e43_birth=tsha == ref)
    assert info["equals_e43_birth"], "born: tensors_sha256 differs from E43's birth"
    if start == "born-shuffled":
        before = {k: v.detach().clone() for k, v in m.named_parameters()}
        shuffle_in_place(m, seed)
        after = dict(m.named_parameters())
        same_values = all(T.equal(T.sort(before[k].flatten())[0], T.sort(after[k].detach().flatten())[0]) for k in before)
        book_same = T.equal(before["book.weight"], after["book.weight"].detach())
        moved = sum(not T.equal(before[k], after[k].detach()) for k in before if k != "book.weight")
        assert same_values and book_same, "born-shuffled: values or byte table changed"
        info.update(shuffled=True, sorted_values_equal_born=same_values, byte_table_unchanged=book_same, tensors_reordered=moved,
                    tensors_sha256=R.tensors_sha(m.state_dict()), born_tensors_sha256=tsha)
    return m, info


# ---------------------------------------------------------------- scoring
class Scorer:
    def __init__(self, seed, n):
        self.held12 = TK.panel(seed, "held")
        self.held0 = TK.panel(seed, "held", k=0)
        self.taught = taught_panel(seed, n)
        self.unseen = unseen_flags(seed, n, self.held12)
        self.P = None

    def __call__(self, m):
        outs = E.respond(m, [p for p, _ in self.held12], None, cap=X16.CAP)
        ok = [o == t for o, (_, t) in zip(outs, self.held12)]
        b = lambda i: sum(len(o) > i and o[i] == t[i] for o, (_, t) in zip(outs, self.held12))
        held12 = dict(n=len(ok), exact=sum(ok), first_byte=b(0), second_byte=b(1), third_is_end=b(2),
                      unseen=dict(n=sum(self.unseen), exact=sum(k for k, u in zip(ok, self.unseen) if u)),
                      seen=dict(n=sum(not u for u in self.unseen), exact=sum(k for k, u in zip(ok, self.unseen) if not u)))
        return dict(held12=held12, held0=X16.score(m, self.held0, None), taught=X16.score(m, self.taught, None))

    def earlier(self, m, seed):
        if self.P is None:
            self.P, _ = X16.panels(seed)
        return {nm: X16.score(m, self.P[nm][12], None) for nm in EARLIER}


# ---------------------------------------------------------------- provenance
def provenance(cmd):
    p = C.provenance(cmd)
    p["sources_sha256"] = {s: X16.sha_file(HERE / s) for s in sorted(set(C.SOURCES) | set(OWN))}
    import subprocess
    p["git"]["sources_modified_vs_commit"] = subprocess.run(["git", "status", "--porcelain", "--", *sorted(set(C.SOURCES) | set(OWN))], cwd=HERE,
                                                            capture_output=True, text=True).stdout.strip().splitlines()
    import numpy
    p["versions"] = dict(python=sys.version.split()[0], torch=T.__version__, numpy=numpy.__version__, torch_threads=T.get_num_threads(),
                         OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"), MKL_NUM_THREADS=os.environ.get("MKL_NUM_THREADS"))
    return p


# ---------------------------------------------------------------- one run
def tag_of(start, n, rule, seed, updates):
    return f"e59_{start}_n{n}_{rule}_{seed}" + ("" if updates == UPDATES else f"_dry{updates}")


def run(start, n, rule, seed, out, cmd, updates=UPDATES):
    assert start in STARTS and n in NS and rule in RULES and seed in SEEDS
    local = C.ARMS[rule]
    arch26.install()
    out = Path(out); tag = tag_of(start, n, rule, seed, updates)
    path = out / f"{tag}.json"
    keep = seed == 73129 and updates == UPDATES
    wdir = out if keep else out / "rows_uncommitted"
    wpath = wdir / f"{tag}_final.pt"
    R.refuse_existing(path, wpath)
    out.mkdir(parents=True, exist_ok=True); wdir.mkdir(parents=True, exist_ok=True)
    stop = OUT / "STOP"
    assert not stop.exists(), f"{stop} exists: {stop.read_text()}"
    m, info = make_start(start, rule, seed)
    opt = C.make_opt(m)
    groups = R.opt_groups(m, opt)
    sc = Scorer(seed, n)
    ref43 = None
    if start == "born" and n == 2048 and updates == UPDATES:
        ref43 = {r["updates"]: r for r in json.loads((E43DIR / f"e43_unrelated-decay_{rule}_{seed}.json").read_text())["rows"]}
    bt = batches(seed, n); dg = C.Digest()
    rows_plan = [r for r in ROWS if r <= updates]
    rec = dict(experiment="E59", kind="run", start=start, n_taught=n, rule=rule, local=local, seed=seed, updates=updates,
               settings=dict(start_record=info, optimizer_groups_read_back=groups, optimizer="exp26_candidate.make_opt (AdamW, lr 0.001, weight decay 0.25 on every tensor but the byte table)",
                             update="exp26_candidate.step(m, opt, batch, local)", batch=X.BATCH, rows=rows_plan,
                             stream="the text of e43_task.stream with taught[:n] in place of taught; generator Random(f'e43|{seed}|stream')",
                             panels=dict(held12="e43_task.panel(seed, 'held')", held0="e43_task.panel(seed, 'held', k=0)",
                                         taught="Random(f'e43|{seed}|panel|taught').sample(taught[:n], min(300, n)), rendered at 12 fillers",
                                         held12_unseen_items=sum(sc.unseen)),
                             check4_against=None if ref43 is None else f"exp45/e43/e43_unrelated-decay_{rule}_{seed}.json",
                             earlier_training_items=dict(fresh=0, born=BIRTH_ITEMS, lived=LIFE_ITEMS, **{"born-shuffled": BIRTH_ITEMS})[start],
                             weights_file=str(wpath.relative_to(out))),
               provenance=provenance(cmd), rows=[], complete=None)
    t0 = time.process_time(); cpu_sc = 0.0; losses = []; finite = True; check4 = {}

    def write():
        tmp = path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(rec, indent=1)); os.replace(tmp, path)

    def score_row(g):
        nonlocal cpu_sc, losses
        if stop.exists():
            rec["stopped"] = f"STOP file at row {g}: {stop.read_text()}"; write()
            raise SystemExit(f"{tag}: stopped at row {g} by {stop}")
        cb = time.process_time()
        s = sc(m)
        row = dict(updates=g, data_digest=dg.value(), tensors_sha256=R.tensors_sha(m.state_dict()),
                   loss=(sum(losses) / len(losses)) if losses else None, finite_losses=all(v == v and abs(v) != float("inf") for v in losses), **s)
        if g in (0, updates):
            row["earlier_panels_k12"] = sc.earlier(m, seed)
        cpu_sc += time.process_time() - cb
        row["cpu"] = time.process_time() - t0
        rec["rows"].append(row); losses = []
        if ref43 is not None and g in CHECK4_ROWS:
            e = ref43[g]
            c4 = dict(data_digest=row["data_digest"] == e["data_digest"],
                      held=row["held12"]["exact"] == e["score3_new_task"]["held"]["exact"],
                      taught=row["taught"]["exact"] == e["score3_new_task"]["taught"]["exact"],
                      tensors_sha256=row["tensors_sha256"] == e["tensors_sha256"])
            check4[str(g)] = c4; rec["check4"] = check4
            if not all(c4.values()):
                stop.write_text(f"{tag} check 4 failed at row {g}: {c4}\n"); write()
                raise SystemExit(f"{tag}: check 4 failed at row {g}: {c4}")
        h = row["held12"]
        print(f"{tag} row {g}: held12 {h['exact']} (bytes {h['first_byte']} {h['second_byte']} {h['third_is_end']}) held0 {row['held0']['exact']} "
              f"taught {row['taught']['exact']}/{row['taught']['n']} loss {row['loss'] if row['loss'] is None else round(row['loss'], 4)} "
              f"cpu {row['cpu']:.0f}s" + (f" check4 {check4[str(g)]}" if str(g) in check4 else ""), flush=True)

    score_row(0); write()
    g = 0
    while g < updates:
        batch = next(bt)
        assert all(b >= 128 for p, _ in batch for b in p), "a prompt byte below 128"      # E43's check 4, task side
        dg.add(batch)
        v = C.step(m, opt, batch, local)
        finite = finite and v == v and abs(v) != float("inf")
        losses.append(v)
        g += 1
        if g in rows_plan:
            score_row(g)
            if len(rec["rows"]) % 8 == 0 or g == updates:
                write()
    sha = R.save_weights(m, wpath)
    rec["complete"] = dict(updates=g, optimizer_steps=R.opt_steps(opt), optimizer_groups_read_back=R.opt_groups(m, opt), data_digest=dg.value(),
                           tensors_sha256=R.tensors_sha(m.state_dict()), weights=str(wpath.relative_to(out)), weights_sha256=sha,
                           all_losses_finite=finite, cpu_total=time.process_time() - t0, cpu_scoring=cpu_sc, cpu_updates=time.process_time() - t0 - cpu_sc,
                           check4_pass=None if ref43 is None else all(all(v.values()) for v in check4.values()))
    write()
    print(f"e59 complete {tag} tensors {rec['complete']['tensors_sha256'][:16]} cpu {rec['complete']['cpu_total']:.0f}s", flush=True)
    return rec


# ---------------------------------------------------------------- checks before the runs
def check1():
    """The stream and the panels; nothing is trained."""
    out = dict(seeds={})
    ok = True
    for i, seed in enumerate(SEEDS):
        r = {}
        # (a) the digest of the first 64 batches of the runner's stream with n = 2,048, by the lines of e43_task.digest
        h = hashlib.sha256(); bt = batches(seed, 2048)
        for _ in range(64):
            for p, y in next(bt):
                h.update(bytes([len(p)]) + bytes(p) + bytes(v % 256 for v in y))
        mine, ref = h.hexdigest(), TK.digest(seed, 64)
        r["a"] = dict(runner=mine, e43_task_digest=ref, equal=mine == ref, begins_as_brief=ref.startswith(TASK_DIGEST64[seed]))
        # (b) the panels
        r["b"] = dict(held_equal=TK.panel(seed, "held") == Scorer(seed, 2048).held12, taught_2048_equal=taught_panel(seed, 2048) == TK.panel(seed, "taught"))
        # (c) and (d)
        e43row64 = json.loads((E43DIR / f"e43_unrelated-decay_S_{seed}.json").read_text())["rows"][1]
        assert e43row64["updates"] == 64
        r["c"], r["d"], r["unseen"] = {}, {}, {}
        for n in NS:
            taught, _ = taught_set(seed, n)
            bt = batches(seed, n); dg = C.Digest(); seen = set()
            for k in range(1, 8193):
                b = next(bt); dg.add(b); seen |= {(p[1], p[2]) for p, _ in b}
                if k == 64:
                    d64 = dg.value()["sha256"]
            d8192 = dg.value()["sha256"]
            ref64 = e43row64["data_digest"]["sha256"] if n == 2048 else DIGEST_REF[(n, 64)][i]
            r["c"][str(n)] = dict(row64=d64, row64_reference=ref64, row64_equal=d64 == ref64, row8192=d8192,
                                  row8192_reference=DIGEST_REF[(n, 8192)][i] if (n, 8192) in DIGEST_REF else None,
                                  row8192_equal=d8192 == DIGEST_REF[(n, 8192)][i] if (n, 8192) in DIGEST_REF else None)
            r["d"][str(n)] = dict(pairs_presented=len(seen), taught=len(taught), equal=seen == set(taught))
            fl = unseen_flags(seed, n, TK.panel(seed, "held"))
            first, second = {a for a, _ in taught}, {b for _, b in taught}
            r["unseen"][str(n)] = dict(held_items_with_unseen_place=sum(fl), symbols_never_first=sorted(set(TK.SYM) - first),
                                       symbols_never_second=sorted(set(TK.SYM) - second))
        r["pass"] = r["a"]["equal"] and r["a"]["begins_as_brief"] and all(r["b"].values()) and all(v["row64_equal"] and v["row8192_equal"] is not False
                                                                                                  and v["row8192_equal"] is not None for v in r["c"].values()) \
            and all(v["equal"] for v in r["d"].values())
        ok = ok and r["pass"]; out["seeds"][str(seed)] = r
        print(f"check 1 seed {seed}: pass {r['pass']}; unseen at 256 {r['unseen']['256']}", flush=True)
    out["note"] = "row 8,192 with n = 2,048 has no other reference than the brief's; row 64 with n = 2,048 is E43's receipt (rule S, row 64)"
    out["pass"] = ok
    return out


def check2():
    """The starts: the assertions of make_start for every start, rule and seed; the byte table's rows 128 to 225 equal in every start under
    both rules; the counts of every start at row 0 on the two held panels and the taught panel (each n)."""
    arch26.install()
    out = dict(seeds={}); ok = True
    for seed in SEEDS:
        rows = {}; recs = {}
        for rule in RULES:
            for start in STARTS:
                m, info = make_start(start, rule, seed)
                rows[(start, rule)] = m.book.weight.detach()[128:226].clone()
                cnt = {}
                for n in NS:
                    s = Scorer(seed, n)(m)
                    cnt[str(n)] = dict(held12=s["held12"]["exact"], held0=s["held0"]["exact"], taught=s["taught"]["exact"], taught_n=s["taught"]["n"])
                recs[f"{start} {rule}"] = dict(info=info, counts_row0=cnt)
                print(f"check 2 seed {seed} {start} {rule}: {cnt}", flush=True)
        ref = rows[("fresh", "S")]
        book_equal = {f"{k[0]} {k[1]}": bool(T.equal(v, ref)) for k, v in rows.items()}
        zero = all(all(c[k] == 0 for c in recs[f"{s} {r}"]["counts_row0"].values() for k in ("held12", "held0", "taught")) for s in ("born", "lived") for r in RULES)
        r = dict(starts=recs, byte_table_rows_128_to_225_equal_to_fresh_S=book_equal, born_and_lived_zero_at_row0=zero)
        r["pass"] = all(book_equal.values()) and zero
        ok = ok and r["pass"]; out["seeds"][str(seed)] = r
    out["pass"] = ok
    return out


def check3(cmd):
    """Cost: 64 updates of fresh with 256 pairs under the sweep into exp45/e59/dry/; CPU of an update and of a row; a forecast for 60 runs."""
    rec = run("fresh", 256, "S", 73129, OUT / "dry", cmd, updates=64)
    rows = rec["rows"]; cs = rec["complete"]
    per_row = cs["cpu_scoring"] / len(rows); per_update = cs["cpu_updates"] / 64
    forecast = 60 * (UPDATES * per_update + len(ROWS) * per_row) / 3600
    return dict(cpu_per_update=per_update, cpu_per_row=per_row, rows_per_run=len(ROWS), forecast_60_runs_cpu_h=forecast,
                forecast_per_run_cpu_h=forecast / 60, limit_cpu_h=28, **{"pass": forecast <= 28})


def main(argv):
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *argv[1:]])
    a = argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    if a[0] == "run":
        run(a[1], int(a[2]), a[3], int(a[4]), Path(opt("--out", str(OUT))), cmd, int(opt("--updates", UPDATES)))
    elif a[0] == "check":
        q = OUT / "checks"; q.mkdir(parents=True, exist_ok=True)
        p = q / f"e59_check{a[1]}.json"
        R.refuse_existing(p)
        t0 = time.process_time(); w0 = time.time()
        res = {"1": check1, "2": check2}[a[1]]() if a[1] in ("1", "2") else check3(cmd)
        res.update(check=a[1], cpu=time.process_time() - t0, wall=time.time() - w0, provenance=provenance(cmd))
        p.write_text(json.dumps(res, indent=1))
        print(f"e59 check {a[1]}:", "PASS" if res["pass"] else "FAIL", flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
