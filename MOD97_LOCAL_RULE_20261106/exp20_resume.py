"""E20-B: the birth and life loops of exp20_reference.py, made resumable after a container restart (11 Nov 2026, after the restart that
killed the three G0 lives of stage 2 at F2.2).

exp20_reference.py (qualified at commit 0e131a29) is imported unchanged and every helper of it is reused: the environment check, the
admission of births (admit_e20), the taught panel, the optimizer, the receipt fields. Only the two loops are written again here, line for
line as in exp20_reference.dev / life, with three additions:

  1. A checkpoint every --ckpt-every updates (default 512) and after every scoring: weights, the AdamW state (moments and step counts),
     the receipt so far, the losses and exposure counts of the block in progress, CPU so far, and the position (stage, update).
     Written to <output>.ckpt by a temporary file and a rename, so a kill during the write leaves the previous checkpoint intact.
  2. On start, if <output>.ckpt exists, the run resumes from it. The data streams are Python generators and cannot be saved; they are
     deterministic, so they are rebuilt exactly as the loop builds them and advanced by drawing and discarding as many batches as were
     already used (pure Python, no network). The network then sees exactly the batches it would have seen.
  3. The receipt records every resume (position, time, CPU spent advancing the streams) and this file's hash. The checkpoint is deleted
     only after the completion record naming the final weights hash has been written.

Claim tested before use (exp45/e20b_resume_check/): a run killed with SIGKILL at random moments and resumed ends with the same scores,
losses, exposure counts, optimizer step counts and final tensors as the qualified uninterrupted run.

    python3 exp20_resume.py dev  <net> <seed> [--out DIR] [--ckpt-every N]
    python3 exp20_resume.py life <net> <seed> [--out DIR] [--ckpt-every N]
"""
import os
import sys
import time
from pathlib import Path

import exp20_reference as R                 # environment check, harness import, helpers (unchanged, qualified)
from exp20_reference import X16, E, X, arch20, T, json

HERE = Path(__file__).resolve().parent
RESUME_SHA = X16.sha_file(HERE / "exp20_resume.py")


def save_ckpt(ck, payload):
    tmp = Path(str(ck) + ".tmp")
    T.save(payload, tmp)
    os.replace(tmp, ck)


def load_ckpt(ck, kind, net, seed):
    c = T.load(ck, weights_only=False)
    want = dict(kind=kind, net=net, seed=seed, dev_updates=E.DEV_DOSE, stage_updates=E.STAGE_UPDATES,
                sources_sha256=R.source_hashes(), resume_sha256=RESUME_SHA)
    bad = {k: (c["meta"].get(k), v) for k, v in want.items() if c["meta"].get(k) != v}
    assert not bad, f"{ck.name}: checkpoint is not for this run {list(bad)}"
    return c


def meta(kind, net, seed):
    return dict(kind=kind, net=net, seed=seed, dev_updates=E.DEV_DOSE, stage_updates=E.STAGE_UPDATES,
                sources_sha256=R.source_hashes(), resume_sha256=RESUME_SHA)


def resume_note(c, ff_cpu):
    return dict(at=c["pos"], wall_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), fast_forward_cpu=ff_cpu)


# ---------------------------------------------------------------- birth (exp20_reference.dev, resumable)
def dev(net, seed, out, cmd, every):
    arch20.install(net)
    path = R.birth_path(out, net, seed); rec_path = path.with_suffix(".json"); ck = path.with_suffix(".ckpt")
    R.refuse_existing(path, str(path) + ".tmp")
    Path(out).mkdir(parents=True, exist_ok=True)
    m = E.new_net(seed)
    desc = arch20.describe(m)
    book0 = m.book.weight.detach().clone()
    opt = R.make_opt(m)
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    pan = {E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV}
    bt = E.batches(tp, seed, E.KMAX["M"])
    if ck.exists():
        c = load_ckpt(ck, "birth", net, seed)
        m.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); book0 = c["book0"]
        rec, census, losses, done, cpu0 = c["rec"], c["census"], c["losses"], c["pos"]["update"], c["cpu"]
        f0 = time.process_time()
        for _ in range(done):
            next(bt)
        rec["resumes"].append(resume_note(c, time.process_time() - f0)); print("e20 dev", net, seed, "resumed at update", done, flush=True)
    else:
        R.refuse_existing(rec_path, ck)
        rec = dict(experiment="E20-B", kind="birth", net=net, seed=seed, kmax=E.KMAX["M"], dev_updates=E.DEV_DOSE, block=X.BLOCK, batch=X.BATCH,
                   optimizer=dict(name="AdamW", lr=R.LR, weight_decay=R.WD, betas=list(R.BETAS), groups=R.opt_groups(m, opt)),
                   protocol="exp16_integrated.develop17 plain path (backprop through time, answer loss, four mod-89 operations, k 0..12)",
                   model=desc, installed=dict(E_forward=f"{E.forward.__module__}.{E.forward.__name__}", E_new_net_class=desc["cls"]),
                   provenance=dict(R.provenance(cmd), resume_runner="exp20_resume.py", resume_runner_sha256=RESUME_SHA, ckpt_every=every),
                   complete=False, curve=[], resumes=[])
        census, losses, done, cpu0 = None, [], 0, 0.0
    t0 = time.process_time()
    cpu = lambda: cpu0 + time.process_time() - t0
    def checkpoint(upd):
        save_ckpt(ck, dict(meta=meta("birth", net, seed), pos=dict(update=upd), model=m.state_dict(), opt=opt.state_dict(), book0=book0,
                           rec=rec, census=census, losses=losses, cpu=cpu()))
    curve = rec["curve"]; row = curve[-1] if curve else None
    for upd in range(done, E.DEV_DOSE):
        if upd % X.BLOCK == 0:
            losses = []
        items = next(bt); losses.append(E.update(m, opt, items, "D"))
        if census is None:
            census = X16.zero_grad_report(m, X.batch_tensor(items)[0]); print("e20 dev credit census:", census["by_role"], flush=True)
        stop = upd + 1
        if stop % X.BLOCK == 0 and (stop % 4096 == 0 or stop == E.DEV_DOSE):
            row = dict(step=stop, loss=sum(losses) / len(losses), cpu=cpu(),
                       sealed={n: {k: E.score(m, it) for k, it in pk.items()} for n, pk in pan.items()}, book_shift_by_role=X16.book_shift(m, book0))
            curve.append(row)
            print("e20 dev", net, seed, stop, {n: [v[k]["exact"] for k in (0, 4, 8, 12, 20)] for n, v in row["sealed"].items()}, f"cpu {row['cpu']:.0f}s", flush=True)
            rec_path.write_text(json.dumps(dict(rec, credit_census_first_update=census), indent=1))
            checkpoint(stop)
        elif stop % every == 0:
            checkpoint(stop)
    assert row is not None, "the birth loop works in whole blocks of 512 updates: E11_DEV_UPDATES must be a positive multiple of 512"
    row["silenced_k12"] = {n: E.score(m, pk[12], silence=True) for n, pk in pan.items()}
    wsha = R.save_weights(m, path)
    rec.update(credit_census_first_update=census, optimizer_steps=R.opt_steps(opt), weights=path.name, weights_sha256=wsha,
               tensors_sha256=R.tensors_sha(m.state_dict()), cpu_total=cpu(), complete=True)
    rec_path.write_text(json.dumps(rec, indent=1))
    ck.unlink(missing_ok=True)
    print("e20 dev", net, seed, "complete", wsha[:16], flush=True)


# ---------------------------------------------------------------- life (exp20_reference.life, arm D, resumable)
def life(net, seed, out, cmd, every):
    arch20.install(net)
    path = Path(out) / f"e20_life_{net}{R.dose_tag_life()}_{seed}.json"; ck = path.with_suffix(".ckpt")
    R.refuse_existing(path.with_suffix(".pt"), str(path.with_suffix(".pt")) + ".tmp")
    if not ck.exists():
        R.refuse_existing(path)
    Path(out).mkdir(parents=True, exist_ok=True)
    bfile = R.birth_path(out, net, seed); m, birth_sha, admission = R.admit_e20(bfile, net, seed)
    brec = json.loads(bfile.with_suffix(".json").read_text())
    rsha = brec.get("provenance", {}).get("resume_runner_sha256")
    assert rsha in (None, RESUME_SHA), f"{bfile.name}: born by another version of exp20_resume.py"
    m.load_state_dict(T.load(bfile))
    desc = arch20.describe(m)
    opt = R.make_opt(m)
    P, comp = X16.panels(seed)
    SP, strict_counts = X16.strict_panels(seed)
    taught = R.taught_panel(seed)
    tcheck = R.taught_panel_check(seed, P, comp, taught)
    assert tcheck["taught_in_F2_entries"] == len(taught) and tcheck["within_in_F2_entries"] == 0 and not any(tcheck["held_in_F2_entries"].values())
    f1c, f3c = {t[2][0] for t in X16.F1}, {t[2][0] for t in X16.F3}
    code_of = lambda p_: next((b_ for b_ in p_ if not 65 <= b_ <= 90), p_[0])
    fam_of = lambda p_: "F2" if code_of(p_) == ord("c") else "F1" if code_of(p_) in f1c else "F3" if code_of(p_) in f3c else "other"
    entries = {f: X16.family_entries(f, seed) for f in ("F1", "F2", "F3")}
    taught_keys = {tuple(p) for p, _ in taught}
    unseen_keys = {tuple(p) for n, v in comp.items() for p, _ in v}
    score = X16.score

    def evaluate():
        res = {n: {k: score(m, it, None) for k, it in pk.items()} for n, pk in P.items()}
        res["compose"] = {n: score(m, it, None) for n, it in comp.items()}
        res["silenced_k12"] = {n: score(m, pk[12], None, silence=True) for n, pk in P.items() if not n.endswith("89")}
        res["silenced_compose"] = {n: score(m, it, None, silence=True) for n, it in comp.items() if n != "within"}
        res["strict_k12"] = {n: score(m, it, None) for n, it in SP.items()}
        res["taught"] = score(m, taught, None)
        return res

    t0 = time.process_time()
    if ck.exists():
        c = load_ckpt(ck, "life", net, seed)
        assert c["birth_sha256"] == birth_sha, "checkpoint was made from another birth"
        m.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        stages, cpu0, cpu_score, losses, expo, presented = c["stages"], c["cpu"], c["cpu_score"], c["losses"], c["expo"], c["presented"]
        presented["taught_panel_distinct_presented"] = set(map(tuple, presented["taught_panel_distinct_presented"]))
        si0, u0 = c["pos"]["stage"], c["pos"]["update"]
        resumed = c
    else:
        cpu0 = 0.0
        c0 = time.process_time(); scores0 = evaluate(); cpu_score = [time.process_time() - c0]
        settings = dict(net=net, model=desc, installed=dict(E_forward=f"{E.forward.__module__}.{E.forward.__name__}", E_new_net_class=desc["cls"]),
                        optimizer=dict(name="AdamW", lr=R.LR, weight_decay=R.WD, betas=list(R.BETAS), groups=R.opt_groups(m, opt)),
                        stage_updates=E.STAGE_UPDATES, dev_updates=E.DEV_DOSE, batch=X.BATCH, arch16=X16.ARCH16, bookdecay=X16.BOOKDECAY16, compgap=X16.COMPGAP16,
                        store=False, panel_sha256=X16.panel_sha(P, comp, SP), strict_panel_counts=strict_counts, taught_panel=tcheck,
                        birth=str(bfile), birth_sha256=birth_sha, birth_admission=admission, protocol="exp16_integrated.life arm D (joint stream of families admitted so far, backprop, decay 0.3)",
                        provenance=dict(R.provenance(cmd), resume_runner="exp20_resume.py", resume_runner_sha256=RESUME_SHA, ckpt_every=every), resumes=[])
        stages = [dict(stage=0, task=None, scores=scores0, shortcut_ceilings=X16.shortcut_ceilings(comp), entries={f: len(v) for f, v in entries.items()}, settings=settings)]
        presented = dict(taught_panel_items_presented=0, taught_panel_distinct_presented=set(), within_or_held_presented=0)
        si0, u0, losses, expo, resumed = 1, 0, [], {}, None
    cpu = lambda: cpu0 + time.process_time() - t0

    def write(extra=None):
        path.write_text(json.dumps(dict(experiment="E20-B", kind="life", net=net, arm="D", seed=seed, schedule=X16.SCHEDULE, stages=stages, **(extra or {})), indent=1))

    def checkpoint(si, u):
        save_ckpt(ck, dict(meta=meta("life", net, seed), pos=dict(stage=si, update=u), birth_sha256=birth_sha, model=m.state_dict(), opt=opt.state_dict(),
                           stages=stages, cpu=cpu(), cpu_score=cpu_score, losses=losses, expo=expo,
                           presented=dict(presented, taught_panel_distinct_presented=sorted(presented["taught_panel_distinct_presented"]))))

    if resumed is None:
        checkpoint(1, 0)
    # rebuild the streams exactly as the loop builds them; advance them past every batch already used
    seen_streams = []; f0 = time.process_time()
    for si, (fam, block) in enumerate(X16.SCHEDULE, 1):
        if si > si0:
            break
        if block == 1:
            seen_streams.append(X16.entry_stream(entries[fam], seed + 100 + si))
        joint = X16.joint_stream(seen_streams, seed + si)
        for _ in range(E.STAGE_UPDATES if si < si0 else u0):
            for _ in range(X.BATCH):
                next(joint)
    if resumed is not None:
        stages[0]["settings"]["resumes"].append(resume_note(resumed, time.process_time() - f0))
        print("e20 life", net, seed, "resumed at", resumed["pos"], flush=True)
    for si, (fam, block) in enumerate(X16.SCHEDULE, 1):
        if si < si0:
            continue
        if si > si0:
            if block == 1:
                seen_streams.append(X16.entry_stream(entries[fam], seed + 100 + si))
            joint = X16.joint_stream(seen_streams, seed + si)
            losses, expo = [], {}
        for u in range(u0 if si == si0 else 0, E.STAGE_UPDATES):
            batch = [next(joint) for _ in range(X.BATCH)]
            for p_, t_ in batch:
                e_ = expo.setdefault(f"joint/{fam_of(p_)}", [0, 0]); e_[0] += 1; e_[1] += len(t_)
                k_ = tuple(p_)
                if k_ in taught_keys:
                    presented["taught_panel_items_presented"] += 1; presented["taught_panel_distinct_presented"].add(k_)
                if k_ in unseen_keys:
                    presented["within_or_held_presented"] += 1
            losses.append(E.update(m, opt, batch, "D", None))
            if si == 1 and u == 0:
                stages[0]["settings"]["zero_grad_first_update"] = X16.zero_grad_report(m, X.batch_tensor(batch)[0])
                print("e20 zero-grad check:", stages[0]["settings"]["zero_grad_first_update"]["by_role"], flush=True)
            if (u + 1) % every == 0 and u + 1 < E.STAGE_UPDATES:
                checkpoint(si, u + 1)
        cb = cpu(); sc = evaluate(); ca = cpu(); cpu_score[0] += ca - cb
        stages.append(dict(stage=si, task=f"{fam}.{block}", loss=sum(losses[-512:]) / min(512, len(losses)), cpu=cb, scores=sc,
                           cpu_scoring_so_far=cpu_score[0], cpu_updates_so_far=ca - cpu_score[0], optimizer_steps=R.opt_steps(opt),
                           exposure_rows_tokens=expo, exposure_note="[rows, bytes under the loss mask]; loss=answer",
                           exposure_panels=dict(presented, taught_panel_distinct_presented=len(presented["taught_panel_distinct_presented"]))))
        print("e20", net, seed, "after", f"{fam}.{block}",
              "F1@k", {n: [sc[n][k]["exact"] for k in X16.KS] for n in sc if n.endswith("97")},
              "comp.exact", {n: v["exact"] for n, v in sc["compose"].items()}, "taught", sc["taught"]["exact"],
              f"cpu {stages[-1]['cpu']:.0f}s", flush=True)
        write()
        losses, expo = [], {}
        if si < len(X16.SCHEDULE):
            checkpoint(si + 1, 0)
            nfam, nblock = X16.SCHEDULE[si]
            if nblock == 1:
                seen_streams.append(X16.entry_stream(entries[nfam], seed + 100 + si + 1))
            joint = X16.joint_stream(seen_streams, seed + si + 1)
            si0, u0 = si + 1, 0        # the next stage starts at update 0 with the streams just built
    assert presented["within_or_held_presented"] == 0, "a within or held-pair panel item was presented in training"
    wsha = R.save_weights(m, path.with_suffix(".pt"))
    write(dict(complete=dict(stages=len(stages), weights=path.with_suffix(".pt").name, weights_sha256=wsha,
                             tensors_sha256=R.tensors_sha(m.state_dict()), optimizer_steps=R.opt_steps(opt),
                             cpu_total=cpu(), cpu_scoring=cpu_score[0])))
    ck.unlink(missing_ok=True)
    print("e20 life", net, seed, "complete", wsha[:16], flush=True)


if __name__ == "__main__":
    cmd = " ".join([f"{k}={v}" for k, v in sorted(R.ENV_AT_START.items()) if k.startswith(("E16_", "E11_"))] + ["python3"] + sys.argv)
    a = sys.argv[1:]
    opt_ = lambda flag, default=None: a[a.index(flag) + 1] if flag in a else default
    outdir = opt_("--out", str(X16.OUT)); every = int(opt_("--ckpt-every", "512"))
    assert every > 0
    if a[0] == "dev":
        dev(a[1], int(a[2]), outdir, cmd, every)
    elif a[0] == "life":
        life(a[1], int(a[2]), outdir, cmd, every)
    else:
        raise SystemExit(__doc__)
