"""E37 runner (brief OPUS_TASK_E37_FRESH_REFERENCE_FOR_A_LIFE_FAMILY_20261115.md, revision 2): fresh networks taught one life family alone,
on the lives' own streams, as the reference for "does a learner that has lived need fewer items for a new family?".

A fresh arch26.new_net(seed), the update exp26_candidate.step (local=True: the sweep, S; local=False: backpropagation, B), the optimizer
exp26_candidate.make_opt(m). Batches are 64 consecutive items of the stream:

    part F3: X16.entry_stream(X16.family_entries("F3", seed), seed + 105), 4,096 updates, rows 1,024, 2,048, 4,096
    part F1: X16.entry_stream(X16.family_entries("F1", seed), seed + 101), 6,144 updates, rows 2,048, 4,096, 6,144

Score at each row: X16.panels(seed)[0] at k = 12 by X16.score, the family's add, sub, max, min (add and sub read; max and min tabled only).

    python3 exp37_fresh_family.py run   <F3|F1> <S|B> <seed> [--updates N] [--out DIR]   # N for the dry runs (check 4) only
    python3 exp37_fresh_family.py check data            [--out DIR]   # checks 1 and 2 (no training)
    python3 exp37_fresh_family.py check path <S|B>      [--out DIR]   # check 3 (2,048 updates of the E26 dry birth's stream, seed 73129)

Nothing existing is edited; exp26_candidate.step and make_opt are called, not patched. Every output path must not exist.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C         # noqa: E402  (imports exp20_reference: environment gate, E16_ variables set as E26 ran)

R, X16, E, X, arch26, T = C.R, C.X16, C.E, C.X, C.arch26, C.T

SEEDS = (73129, 93683, 116089)
PARTS = {"F3": dict(si=5, updates=4096, rows=(1024, 2048, 4096), mod=73),
         "F1": dict(si=1, updates=6144, rows=(2048, 4096, 6144), mod=97)}
OPS = ("add", "sub", "max", "min")
OUT = HERE / "exp45" / "e37"
E26 = HERE / "exp45" / "e26"
E31 = HERE / "exp45" / "e31"
E26Q = HERE / "exp45" / "e26_qual"


def provenance(cmd):
    p = C.provenance(cmd)
    p["sources_sha256"]["exp37_fresh_family.py"] = X16.sha_file(HERE / "exp37_fresh_family.py")
    return p


def fresh_stream(part, seed):
    return X16.entry_stream(X16.family_entries(part, seed), seed + 100 + PARTS[part]["si"])


def batches_of(stream):
    while True:
        yield [next(stream) for _ in range(X.BATCH)]


def life_receipt(arm, seed):
    return json.loads(((E31 / f"e31_life_K_{seed}.json") if arm == "K" else (E26 / f"e26_life_{arm}_{seed}.json")).read_text())


# ---------------------------------------------------------------- the runs
def run(part, rule, seed, out, cmd, updates=None):
    arch26.install()
    local = C.ARMS[rule]
    cfg = PARTS[part]
    full = updates is None
    updates = cfg["updates"] if full else updates
    rows = cfg["rows"] if full else (updates,)
    stem = f"e37_{part}_{rule}{'' if full else f'_dry{updates}'}_{seed}"
    out = Path(out); rec_path = out / f"{stem}.json"
    wpaths = {r: out / f"{stem}_row{r}.pt" for r in rows}
    R.refuse_existing(rec_path, *wpaths.values(), *(str(p) + ".tmp" for p in wpaths.values()))
    out.mkdir(parents=True, exist_ok=True)
    m = arch26.new_net(seed)
    opt = C.make_opt(m)
    P = X16.panels(seed)[0]
    pan = {f"{o}{cfg['mod']}": P[f"{o}{cfg['mod']}"][12] for o in OPS}
    bt = batches_of(fresh_stream(part, seed))
    dg = C.Digest()
    rec = dict(experiment="E37", kind=part, rule=rule, local=local, net="G1", seed=seed, updates=updates, rows=list(rows), batch=X.BATCH,
               stream=f"X16.entry_stream(X16.family_entries('{part}', {seed}), {seed} + {100 + cfg['si']})",
               panel=f"X16.panels({seed})[0][op{cfg['mod']}][12], X16.score", model=arch26.describe(m), optimizer=C.opt_record(m, opt),
               update="exp26_candidate.step (forward(local), answer-masked cross entropy, zero_grad, backward, AdamW step)",
               installed=dict(E_forward=f"{E.forward.__module__}.{E.forward.__name__}"), provenance=provenance(cmd), complete=False, curve=[])
    t0 = time.process_time(); cpu_score = 0.0; losses = []
    for u in range(1, updates + 1):
        items = next(bt); dg.add(items); losses.append(C.step(m, opt, items, local))
        if u in rows:
            cb = time.process_time()
            sc = {n: X16.score(m, it) for n, it in pan.items()}
            cpu_score += time.process_time() - cb
            wsha = R.save_weights(m, wpaths[u])
            a, s = (100 * sc[f"{o}{cfg['mod']}"]["exact"] / sc[f"{o}{cfg['mod']}"]["n"] for o in ("add", "sub"))
            row = dict(step=u, items=u * X.BATCH, loss_last512=sum(losses[-512:]) / min(512, len(losses)),
                       finite_losses=all(v == v and abs(v) != float("inf") for v in losses), scores_k12=sc, mean_add_sub=(a + s) / 2,
                       cpu=time.process_time() - t0, cpu_scoring_so_far=cpu_score, optimizer_steps=R.opt_steps(opt), data_digest=dg.value(),
                       weights=wpaths[u].name, weights_sha256=wsha, tensors_sha256=R.tensors_sha(m.state_dict()))
            rec["curve"].append(row)
            print("e37", part, rule, seed, u, {n: v["exact"] for n, v in sc.items()}, f"mean {row['mean_add_sub']:.2f}", f"cpu {row['cpu']:.0f}s", flush=True)
            rec_path.write_text(json.dumps(rec, indent=1))
    rec.update(complete=True, data_digest=dg.value(), cpu_total=time.process_time() - t0, cpu_scoring=cpu_score,
               cpu_updates=time.process_time() - t0 - cpu_score, optimizer_read_back=C.opt_record(m, opt))
    if part == "F1" and full:            # check 1, again on the run itself
        want = life_receipt("S", seed)["stages"][1]["data_digest"]
        rec["check1_life_block1_digest"] = dict(life=want, run=dg.value(), equal=want == dg.value())
    rec_path.write_text(json.dumps(rec, indent=1))
    print("e37", part, rule, seed, "complete", f"cpu {rec['cpu_total']:.0f}s", flush=True)


# ---------------------------------------------------------------- checks 1 and 2
def tagged(stream, fam):
    while True:
        yield fam, next(stream)


def check_data(out, cmd):
    """1: part F1's 6,144 batches hash to stages[1].data_digest of the S, B and K lives. 2: the F3 counts per block from the receipts beside
    the rows' items, and (beyond the brief) a rebuild of each life's joint stream: the F3 items it drew, in order, are a prefix of part F3's."""
    path = Path(out) / "E37_CHECK_DATA.json"; R.refuse_existing(path); Path(out).mkdir(parents=True, exist_ok=True)
    res = dict(experiment="E37", check="data (checks 1 and 2)", provenance=provenance(cmd), seeds={})
    for seed in SEEDS:
        r = {}
        dg = C.Digest(); bt = batches_of(fresh_stream("F1", seed))
        for _ in range(6144):
            dg.add(next(bt))
        r["check1"] = {arm: dict(life=life_receipt(arm, seed)["stages"][1]["data_digest"], fresh=dg.value()) for arm in ("S", "B", "K")}
        for v in r["check1"].values():
            v["equal"] = v["life"] == v["fresh"]
        rec = {arm: life_receipt(arm, seed)["stages"] for arm in ("S", "B", "K")}
        b5 = {arm: st[5]["exposure_rows_tokens"]["joint/F3"][0] for arm, st in rec.items()}
        b6 = {arm: st[6]["exposure_rows_tokens"]["joint/F3"][0] for arm, st in rec.items()}
        r["check2_receipts"] = dict(block5=b5, block5_vs=131072, block5_plus_block6={a: b5[a] + b6[a] for a in b5}, block5_plus_block6_vs=262144)
        # rebuild of the lives' joint streams (exp26_candidate.life lines 241-245), each item tagged with its family
        entries = {f: X16.family_entries(f, seed) for f in ("F1", "F2", "F3")}
        seen, f3_items, counts = [], [], []
        for si, (fam, block) in enumerate(X16.SCHEDULE, 1):
            if block == 1:
                seen.append(tagged(X16.entry_stream(entries[fam], seed + 100 + si), fam))
            joint = X16.joint_stream(seen, seed + si)
            n = 0
            for _ in range(E.STAGE_UPDATES * X.BATCH):
                f, it = next(joint)
                if f == "F3":
                    f3_items.append(it); n += 1
            counts.append(n)
        fs = fresh_stream("F3", seed)
        fresh = [next(fs) for _ in range(len(f3_items))]
        r["check2_rebuild"] = dict(f3_per_block=counts, block5_equals_receipts=all(counts[4] == v for v in b5.values()),
                                   block6_equals_receipts=all(counts[5] == v for v in b6.values()),
                                   life_f3_items=len(f3_items), prefix_of_part_F3=fresh == f3_items)
        res["seeds"][seed] = r
        print("e37 check data", seed, json.dumps(dict(c1=[v["equal"] for v in r["check1"].values()], c2=r["check2_receipts"], rb=r["check2_rebuild"])), flush=True)
    res["ok"] = all(all(v["equal"] for v in r["check1"].values()) and r["check2_rebuild"]["prefix_of_part_F3"] and r["check2_rebuild"]["block5_equals_receipts"]
                    and r["check2_rebuild"]["block6_equals_receipts"] for r in res["seeds"].values())
    path.write_text(json.dumps(res, indent=1))
    print("e37 check data ok:", res["ok"], flush=True)


# ---------------------------------------------------------------- check 3
def check_path(rule, out, cmd):
    """2,048 updates from arch26.new_net(73129) with make_opt on the E26 birth's own stream give the dry birth's tensors_sha256."""
    arch26.install()
    seed = 73129
    path = Path(out) / f"E37_CHECK_PATH_{rule}.json"; R.refuse_existing(path); Path(out).mkdir(parents=True, exist_ok=True)
    want = json.loads((E26Q / f"e26_dev_{rule}_dry2048_{seed}.json").read_text())
    m = arch26.new_net(seed); opt = C.make_opt(m)
    bt = E.batches([(t, E.split(t, .5, seed)[0]) for t in E.DEV], seed, 12)
    dg = C.Digest(); t0 = time.process_time()
    for _ in range(2048):
        items = next(bt); dg.add(items); C.step(m, opt, items, C.ARMS[rule])
    got = R.tensors_sha(m.state_dict())
    res = dict(experiment="E37", check=f"path (check 3), rule {rule}", provenance=provenance(cmd), reference=str((E26Q / f"e26_dev_{rule}_dry2048_{seed}.json").relative_to(HERE)),
               tensors_sha256=dict(reference=want["tensors_sha256"], run=got), data_digest=dict(reference=want["data_digest"], run=dg.value()),
               cpu=time.process_time() - t0)
    res["ok"] = got == want["tensors_sha256"] and dg.value() == want["data_digest"]
    path.write_text(json.dumps(res, indent=1))
    print("e37 check path", rule, "ok:", res["ok"], got[:16], want["tensors_sha256"][:16], f"cpu {res['cpu']:.0f}s", flush=True)


def main(argv):
    out = OUT
    if "--out" in argv:
        i = argv.index("--out"); out = Path(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    upd = None
    if "--updates" in argv:
        i = argv.index("--updates"); upd = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    cmd = " ".join(["python3"] + sys.argv)
    if argv[0] == "run":
        part, rule, seed = argv[1], argv[2], int(argv[3])
        assert part in PARTS and rule in C.ARMS and seed in SEEDS
        run(part, rule, seed, out, cmd, upd)
    elif argv[0] == "check" and argv[1] == "data":
        check_data(out, cmd)
    elif argv[0] == "check" and argv[1] == "path":
        check_path(argv[2], out, cmd)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
