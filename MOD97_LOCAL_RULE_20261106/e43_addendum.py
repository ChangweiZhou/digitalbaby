"""E43 addendum 1 (OPUS_TASK_E43_IS_A_FAMILY_LOST_TO_UNRELATED_LEARNING_20261116.md, addendum of 16 November): a reminder after the unrelated
task, and one arm under plain gradient steps. A new file; nothing existing is edited, called beyond what it names, or patched.

    reminder   from exp45/e43/e43_unrelated_<rule>_<seed>_row4096.pt (checked against its receipt), 1,024 updates on the birth family:
               e36_reminder's reminder path with the start file replaced: E36's data (e36_reminder.batches(seed): exp11_cell.batches(birth
               taught half, seed + 36, 12)), E36's scorer (e36_reminder.Scorer), the rule of the run that made the file
               (exp26_candidate.step, local = rule S), a fresh exp26_candidate.make_opt(m), T.manual_seed(seed) as E36.
               Rows 0, 16, 32, 64, 128, 256, 1,024; weights committed at 128 and 1,024.
    plain      unrelated-plain: exp43_unrelated's unrelated arm (e43_task stream, no decay, its rows and scores) with
               torch.optim.SGD(m.parameters(), lr=LR), no momentum, in place of AdamW; the step is exp26_candidate.step's text with that
               optimizer (exp38_erase.step, no pull). Weights committed at 64 and 4,096.

    python3 e43_addendum.py reminder <S|B> <seed>
    python3 e43_addendum.py plain <S|B> <seed> <lr>
    python3 e43_addendum.py check                 # the reminder's checks (row 0; batch digest at 1,024 against E36)
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp43_unrelated as U          # noqa: E402
import e36_reminder as E36           # noqa: E402

E38, C, R, X16, E, X, arch26, T, TK = U.E38, U.C, U.R, U.X16, U.E, U.X, U.arch26, U.T, U.TK
OUT = U.OUT43 / "addendum1"
RROWS = (0, 16, 32, 64, 128, 256, 1024)
RCOMMIT = (128, 1024)
PCOMMIT = (64, 4096)
SEEDS = (73129, 93683, 116089)
OWN = ("e43_addendum.py", "exp43_unrelated.py", "e43_task.py", "e36_reminder.py", "exp38_erase.py")


def provenance(cmd):
    p = U.provenance(cmd)
    p["addendum_sources_sha256"] = {s: X16.sha_file(HERE / s) for s in OWN}
    import subprocess
    p["git"]["addendum_sources_modified_vs_commit"] = subprocess.run(["git", "status", "--porcelain", "--", *OWN], cwd=HERE, capture_output=True,
                                                                     text=True).stdout.strip().splitlines()
    return p


def reminder(rule, seed, cmd):
    arch26.install()
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "rows_uncommitted").mkdir(exist_ok=True)
    name = f"e43_reminder_{rule}_{seed}"; rec_path = OUT / f"{name}.json"; R.refuse_existing(rec_path)
    T.manual_seed(seed)
    src = U.OUT43 / f"e43_unrelated_{rule}_{seed}_row4096.pt"
    srec = json.loads((U.OUT43 / f"e43_unrelated_{rule}_{seed}.json").read_text())
    r4096 = next(r for r in srec["rows"] if r["updates"] == 4096)
    sha = X16.sha_file(src); assert sha == r4096["weights_sha256"], "start file hash is not the receipt's"
    m = arch26.new_net(seed); m.load_state_dict(T.load(src))
    tsha = R.tensors_sha(m.state_dict()); assert tsha == r4096["tensors_sha256"]
    opt = C.make_opt(m)
    sc = E36.Scorer(seed); bt = E36.batches(seed); dg = C.Digest()
    rec = dict(experiment="E43 addendum 1", kind="reminder", rule=rule, seed=seed, updates=1024, rows_planned=list(RROWS),
               start=dict(file=str(src.relative_to(HERE)), weights_sha256=sha, tensors_sha256=tsha, receipt_row4096=dict(weights_sha256=r4096["weights_sha256"],
                          tensors_sha256=r4096["tensors_sha256"])),
               optimizer=C.opt_record(m, opt), data="e36_reminder.batches(seed) = exp11_cell.batches(birth taught half, seed + 36, 12)",
               update=f"exp26_candidate.step(local={rule == 'S'})", provenance=provenance(cmd), rows=[], complete=False)
    t0 = time.process_time(); cpu_score = 0.0; losses = []; u = 0
    for r in RROWS:
        while u < r:
            items = next(bt); dg.add(items); losses.append(C.step(m, opt, items, rule == "S")); u += 1
        cb = time.process_time(); s = sc(m); cpu_score += time.process_time() - cb
        wp = (OUT if r in RCOMMIT else OUT / "rows_uncommitted") / f"{name}_row{r}.pt"
        row = dict(updates=r, scores=s, loss_since_last_row=(sum(losses) / len(losses)) if losses else None,
                   finite_losses=all(v == v and abs(v) != float("inf") for v in losses), data_digest=dg.value(),
                   weights=str(wp.relative_to(OUT)), weights_sha256=R.save_weights(m, wp), tensors_sha256=R.tensors_sha(m.state_dict()),
                   optimizer_steps=R.opt_steps(opt), cpu=time.process_time() - t0, cpu_scoring_so_far=cpu_score)
        losses = []; rec["rows"].append(row)
        sm = s["summary"]
        print(name, r, f"held {sm['held_add_sub_mean']:.1f} wrap {sm['wrap_score']:.1f} nowrap {sm['nowrap_score']:.1f}", f"cpu {row['cpu']:.0f}s", flush=True)
        rec_path.write_text(json.dumps(rec, indent=1))
    rec.update(complete=True, cpu_total=time.process_time() - t0, cpu_scoring=cpu_score)
    rec_path.write_text(json.dumps(rec, indent=1))
    print(name, "complete", flush=True)


def plain(rule, seed, lr, cmd, updates=4096):
    """exp43_unrelated.run's unrelated arm, copied, with SGD(lr) in place of AdamW. updates = 16384: the brief's condition failed at 4,096, so the
    run is repeated from the birth to 16,384 (rows 8,192 and 16,384 added); its rows to 4,096 must equal the 4,096 run's hashes (checked by `check16`)."""
    local = C.ARMS[rule]
    arch26.install()
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "rows_uncommitted").mkdir(exist_ok=True)
    tag = f"e43_unrelated-plain_lr{lr:g}_{rule}_{seed}" + ("" if updates == 4096 else f"_to{updates}"); path = OUT / f"{tag}.json"; R.refuse_existing(path)
    bfile = C.birth_path(E38.E26DIR, rule, seed); m, birth_sha, admission = E38.admit(bfile, rule, seed)
    bsd = T.load(bfile); m.load_state_dict(bsd); birth_tsha = R.tensors_sha(m.state_dict())
    opt = T.optim.SGD(m.parameters(), lr=lr)
    groups = [dict(lr=g["lr"], momentum=g["momentum"], weight_decay=g["weight_decay"], dampening=g["dampening"], nesterov=g["nesterov"],
                   n_params=len(g["params"])) for g in opt.param_groups]
    P, comp = X16.panels(seed)
    held = {n: P[n][12] for n in ("add89", "sub89", "max89", "min89")}
    s1 = U.first_family_items(); s2_items = [it for n in held for it in held[n]]
    bt = TK.batches(seed, X.BATCH)
    new_panels = {"held": TK.panel(seed, "held"), "taught": TK.panel(seed, "taught")}
    dg = C.Digest(); h0 = U.eq_states(m, s2_items); h0_mean = float(h0.abs().mean())
    t0 = time.process_time(); cpu_score = [0.0]; rows = []; losses = []

    def score_row(g):
        cb = time.process_time()
        r = dict(updates=g, data_digest=dg.value(), loss_since_last_row=(sum(losses) / len(losses)) if losses else None,
                 finite_losses=all(v == v and abs(v) != float("inf") for v in losses))
        r["score1_first_family_all_pairs_k0"] = U.score1(m, s1)
        r["score2_first_family_heldout_k12"] = {n: X16.score(m, it, None) for n, it in held.items()}
        r["score3_new_task"] = {n: X16.score(m, it, None) for n, it in new_panels.items()}
        r["score4_movement"] = U.movement(m, bsd)
        h = U.eq_states(m, s2_items)
        r["score6_state_change"] = dict(mean_abs_change=float((h - h0).abs().mean()), mean_abs_state_row0=h0_mean, ratio=float((h - h0).abs().mean()) / h0_mean)
        cpu_score[0] += time.process_time() - cb
        r["tensors_sha256"] = R.tensors_sha(m.state_dict())
        wp = (OUT if g in PCOMMIT or g == 16384 else OUT / "rows_uncommitted") / f"{tag}_row{g}.pt"
        r["weights"] = str(wp.relative_to(OUT)); r["weights_sha256"] = R.save_weights(m, wp)
        r["cpu"] = time.process_time() - t0
        rows.append(r); losses.clear()
        s = r["score1_first_family_all_pairs_k0"]; pc = lambda d: round(100 * d["exact"] / d["n"], 1)
        print(tag, "row", g, "loss", r["loss_since_last_row"], "reduction", round(s["reduction_score"], 1), "other", round(s["add89"]["other"]["pct"], 1),
              round(s["sub89"]["other"]["pct"], 1), "max/min", round(s["max89"]["all"]["pct"], 1), round(s["min89"]["all"]["pct"], 1),
              "new", {n: pc(v) for n, v in r["score3_new_task"].items()}, f"cpu {r['cpu']:.0f}s", flush=True)

    settings = dict(experiment="E43 addendum 1", arm="unrelated-plain", rule=rule, local=local, seed=seed, lr=lr, optimizer="torch.optim.SGD, no momentum",
                    optimizer_groups_read_back=groups, decay=0.0, birth=str(bfile.relative_to(HERE)), birth_sha256=birth_sha,
                    birth_tensors_sha256=birth_tsha, birth_admission=admission, rows=list(U.ROWS) + ([8192, 16384] if updates == 16384 else []), committed_rows=list(PCOMMIT) + ([16384] if updates == 16384 else []),
                    task_file_sha256=X16.sha_file(HERE / "e43_task.py"), provenance=provenance(cmd))

    def write(complete=None):
        path.write_text(json.dumps(dict(experiment="E43 addendum 1", kind="run", arm="unrelated-plain", rule=rule, seed=seed, lr=lr, settings=settings,
                                        rows=rows, complete=complete), indent=1))

    score_row(0); g = 0
    while g < updates:
        batch = next(bt)
        assert all(b >= 128 for p, _ in batch for b in p)
        dg.add(batch)
        losses.append(E38.step(m, opt, batch, local))
        g += 1
        if g in U.ROWS or g in (8192, 16384):
            score_row(g); write()
    write(dict(updates=g, tensors_sha256=R.tensors_sha(m.state_dict()), data_digest=dg.value(), cpu_total=time.process_time() - t0, cpu_scoring=cpu_score[0]))
    print(tag, "complete", flush=True)


def check(cmd):
    """Reminder checks: row 0 = the start file's tensors_sha256 and E43's row-4,096 first-family scores; batch digest at 1,024 = E36's."""
    res = {}
    for rule in ("S", "B"):
        for seed in SEEDS:
            me = json.loads((OUT / f"e43_reminder_{rule}_{seed}.json").read_text())
            e43 = next(r for r in json.loads((U.OUT43 / f"e43_unrelated_{rule}_{seed}.json").read_text())["rows"] if r["updates"] == 4096)
            e36 = next(r for r in json.loads((E36.OUT / f"e36_reference_{rule}_{seed}.json").read_text())["rows"] if r["updates"] == 1024)
            r0 = me["rows"][0]; s = r0["scores"]; s1 = e43["score1_first_family_all_pairs_k0"]
            d = dict(row0_tensors=r0["tensors_sha256"] == e43["tensors_sha256"],
                     row0_heldout=s["birth_held_k12"] == e43["score2_first_family_heldout_k12"],
                     row0_allpairs=all(s["birth_all_pairs_k0"][n]["wrap"]["exact"] == s1[n]["reduction"]["exact"] and
                                       s["birth_all_pairs_k0"][n]["nowrap"]["exact"] == s1[n]["other"]["exact"] for n in ("add89", "sub89")),
                     digest_1024=me["rows"][-1]["data_digest"] == e36["data_digest"])
            res[f"{rule}_{seed}"] = d
    res["pass"] = all(all(v.values()) for k, v in res.items())
    p = OUT / "e43_addendum_reminder_checks.json"; R.refuse_existing(p)
    p.write_text(json.dumps(dict(command=cmd, provenance=provenance(cmd), result=res), indent=1))
    print("reminder checks", "pass" if res["pass"] else "FAIL", res, flush=True)


def check16(cmd):
    """The 16,384 reruns: rows 0 to 4,096 have the same tensors_sha256 and batch digests as the 4,096 runs."""
    res = {}
    for f in sorted(OUT.glob("e43_unrelated-plain_lr*_to16384.json")):
        long = {r["updates"]: r for r in json.loads(f.read_text())["rows"]}
        short = {r["updates"]: r for r in json.loads((OUT / f.name.replace("_to16384", "")).read_text())["rows"]}
        res[f.stem] = {u: short[u]["tensors_sha256"] == long[u]["tensors_sha256"] and short[u]["data_digest"] == long[u]["data_digest"] for u in short}
    res["pass"] = bool(res) and all(all(v.values()) for v in res.values())
    p = OUT / "e43_addendum_plain16384_checks.json"; R.refuse_existing(p)
    p.write_text(json.dumps(dict(command=cmd, provenance=provenance(cmd), result=res), indent=1))
    print("check16", "pass" if res["pass"] else "FAIL", res, flush=True)


if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])
    if a[0] == "reminder":
        reminder(a[1], int(a[2]), cmd)
    elif a[0] == "plain":
        plain(a[1], int(a[2]), float(a[3]), cmd, int(a[4]) if len(a) > 4 else 4096)
    elif a[0] == "check":
        check(cmd)
    elif a[0] == "check16":
        check16(cmd)
    else:
        raise SystemExit(__doc__)
