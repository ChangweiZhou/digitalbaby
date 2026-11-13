"""E26 / E26-X: the life loop of exp26_candidate.life, made resumable after a container restart (13 Nov 2026, after the restart that killed
three E26-X crossed lives; the owner's instruction "Do it").

exp26_candidate.py (frozen at 360e1b8d) is imported unchanged and its helpers are reused: the environment gate (through exp20_reference),
arch26.install, admit_e26, make_opt, opt_record, step, Digest, provenance, the panels and the scorer. Only the life loop is written again
here, line for line as in exp26_candidate.life, with the birth and the update named separately as in exp26x_cross.py, and three additions:

  1. A checkpoint every --ckpt-every updates (default 512), at the start of every stage and after every scoring: weights, the AdamW state
     (moments and step counts), the stages written so far, the losses of the stage in progress, CPU so far and the position (stage, update).
     Written to <output>.ckpt through a temporary file and a rename, so a kill during the write leaves the previous checkpoint intact.
  2. On start, if <output>.ckpt exists, the run resumes from it. The data streams are deterministic Python generators and cannot be saved;
     they are rebuilt exactly as the loop builds them and advanced by drawing the batches already used. Those batches are passed again
     through the bookkeeping that does not touch the network (the batch digest, exposure counts, taught-panel and held-item counts), so
     the receipt's digests and counts are those of the uninterrupted run. The network sees exactly the batches it would have seen.
  3. The receipt records every resume (position, time, CPU spent advancing the streams) and this file's hash. The checkpoint is deleted
     only after the completion record naming the final weights hash has been written.

Qualified before use (exp45/e26x/resume_check/): runs killed with SIGKILL at random moments and resumed end with the same scores, losses,
digests, exposure counts, optimizer steps and final tensors as exp26x_cross.py's uninterrupted pilot runs.

    python3 exp26_resume.py life <S|B birth> <S|B update> <seed> [--out DIR] [--ckpt-every N]

Output: <out>/e26x_life_<birth><update><DEVTAG><DRYTAG>_<seed>.json (+ .pt, and .ckpt while running). Births are read from exp45/e26.
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C                   # noqa: E402  (frozen; environment gate through exp20_reference)
from exp26_candidate import R, X16, E, X, T, json, arch26   # noqa: E402

RESUME_SHA = X16.sha_file(HERE / "exp26_resume.py")
BIRTHS = HERE / "exp45" / "e26"


def save_ckpt(ck, payload):
    tmp = Path(str(ck) + ".tmp")
    T.save(payload, tmp)
    os.replace(tmp, ck)


def meta(b, u, seed):
    return dict(kind="life", birth_arm=b, update_arm=u, seed=seed, dev_updates=E.DEV_DOSE, stage_updates=E.STAGE_UPDATES,
                sources_sha256=C.source_hashes(), resume_sha256=RESUME_SHA)


def life(b, u, seed, out, cmd, every):
    assert b in C.ARMS and u in C.ARMS
    local = C.ARMS[u]
    arch26.install()
    path = Path(out) / f"e26x_life_{b}{u}{E.DEVTAG}{E.DRYTAG}_{seed}.json"; ck = path.with_suffix(".ckpt")
    R.refuse_existing(path.with_suffix(".pt"), str(path.with_suffix(".pt")) + ".tmp")
    if not ck.exists():
        R.refuse_existing(path)
    Path(out).mkdir(parents=True, exist_ok=True)
    bfile = C.birth_path(BIRTHS, b, seed); m, birth_sha, admission = C.admit_e26(bfile, b, seed)
    m.load_state_dict(T.load(bfile))
    desc = arch26.describe(m)
    opt = C.make_opt(m)
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
    presented = dict(taught_panel_items_presented=0, taught_panel_distinct_presented=set(), within_or_held_presented=0)
    dg = C.Digest()
    score = X16.score

    def evaluate():
        res = {n: {k: score(m, it, None) for k, it in pk.items()} for n, pk in P.items()}
        res["compose"] = {n: score(m, it, None) for n, it in comp.items()}
        res["silenced_k12"] = {n: score(m, pk[12], None, silence=True) for n, pk in P.items() if not n.endswith("89")}
        res["silenced_compose"] = {n: score(m, it, None, silence=True) for n, it in comp.items() if n != "within"}
        res["strict_k12"] = {n: score(m, it, None) for n, it in SP.items()}
        res["taught"] = score(m, taught, None)
        return res

    def account(batch, expo):          # the bookkeeping of exp26_candidate.life that does not touch the network
        for p_, t_ in batch:
            e_ = expo.setdefault(f"joint/{fam_of(p_)}", [0, 0]); e_[0] += 1; e_[1] += len(t_)
            k_ = tuple(p_)
            if k_ in taught_keys:
                presented["taught_panel_items_presented"] += 1; presented["taught_panel_distinct_presented"].add(k_)
            if k_ in unseen_keys:
                presented["within_or_held_presented"] += 1
        dg.add(batch)

    t0 = time.process_time()
    if ck.exists():
        c = T.load(ck, weights_only=False)
        bad = [k for k, v in meta(b, u, seed).items() if c["meta"].get(k) != v]
        assert not bad, f"{ck.name}: checkpoint is not for this run {bad}"
        assert c["birth_sha256"] == birth_sha, "checkpoint was made from another birth"
        m.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        stages, cpu0, cpu_score, losses = c["stages"], c["cpu"], c["cpu_score"], c["losses"]
        si0, u0 = c["pos"]["stage"], c["pos"]["update"]
        resumed = c
    else:
        cpu0 = 0.0
        c0 = time.process_time(); scores0 = evaluate(); cpu_score = [time.process_time() - c0]
        prov = dict(C.provenance(cmd), crossover=dict(birth_arm=b, update_arm=u, update_local=local, runner="exp26_resume.py", runner_sha256=RESUME_SHA),
                    resume_runner="exp26_resume.py", resume_runner_sha256=RESUME_SHA, ckpt_every=every)
        settings = dict(arm=u, local=local, net="G1", model=desc, installed=dict(E_forward=f"{E.forward.__module__}.{E.forward.__name__}", E_new_net_class=desc["cls"]),
                        optimizer=C.opt_record(m, opt), stage_updates=E.STAGE_UPDATES, dev_updates=E.DEV_DOSE, batch=X.BATCH, arch16=X16.ARCH16,
                        bookdecay=X16.BOOKDECAY16, compgap=X16.COMPGAP16, store=False, panel_sha256=X16.panel_sha(P, comp, SP), strict_panel_counts=strict_counts,
                        taught_panel=tcheck, birth=str(bfile), birth_sha256=birth_sha, birth_admission=admission,
                        update="exp26_candidate.step (the same loop as the birth)",
                        protocol="exp16_integrated.life arm D's protocol (joint stream of families admitted so far), as exp20_reference.life; decay 0.25",
                        provenance=prov, resumes=[])
        stages = [dict(stage=0, task=None, scores=scores0, shortcut_ceilings=X16.shortcut_ceilings(comp), entries={f: len(v) for f, v in entries.items()}, settings=settings)]
        si0, u0, losses, resumed = 1, 0, [], None
    cpu = lambda: cpu0 + time.process_time() - t0

    def write(extra=None):
        path.write_text(json.dumps(dict(experiment="E26", kind="life", arm=u, local=local, net="G1", seed=seed, schedule=X16.SCHEDULE, stages=stages, **(extra or {})), indent=1))

    def checkpoint(si, uu):
        save_ckpt(ck, dict(meta=meta(b, u, seed), pos=dict(stage=si, update=uu), birth_sha256=birth_sha, model=m.state_dict(), opt=opt.state_dict(),
                           stages=stages, cpu=cpu(), cpu_score=cpu_score, losses=losses))

    if resumed is None:
        checkpoint(1, 0)
    # rebuild the streams exactly as the loop builds them; pass every batch already used through the bookkeeping again
    seen_streams = []; f0 = time.process_time(); expo = {}
    for si, (fam, block) in enumerate(X16.SCHEDULE, 1):
        if si > si0:
            break
        if block == 1:
            seen_streams.append(X16.entry_stream(entries[fam], seed + 100 + si))
        joint = X16.joint_stream(seen_streams, seed + si)
        expo = {}
        for _ in range(E.STAGE_UPDATES if si < si0 else u0):
            account([next(joint) for _ in range(X.BATCH)], expo)
    if resumed is not None:
        stages[0]["settings"]["resumes"].append(dict(at=resumed["pos"], wall_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                     fast_forward_cpu=time.process_time() - f0))
        print("e26x life", b, u, seed, "resumed at", resumed["pos"], flush=True)
    si = si0
    while si <= len(X16.SCHEDULE):
        fam, block = X16.SCHEDULE[si - 1]
        for uu in range(u0, E.STAGE_UPDATES):
            batch = [next(joint) for _ in range(X.BATCH)]
            account(batch, expo)
            losses.append(C.step(m, opt, batch, local))
            if si == 1 and uu == 0:
                stages[0]["settings"]["zero_grad_first_update"] = X16.zero_grad_report(m, X.batch_tensor(batch)[0])
                print("e26 zero-grad check:", stages[0]["settings"]["zero_grad_first_update"]["by_role"], flush=True)
            if (uu + 1) % every == 0 and uu + 1 < E.STAGE_UPDATES:
                checkpoint(si, uu + 1)
        cb = cpu(); sc = evaluate(); ca = cpu(); cpu_score[0] += ca - cb
        stages.append(dict(stage=si, task=f"{fam}.{block}", loss=sum(losses[-512:]) / min(512, len(losses)),
                           finite_losses=all(v == v and abs(v) != float("inf") for v in losses), cpu=cb, scores=sc,
                           cpu_scoring_so_far=cpu_score[0], cpu_updates_so_far=ca - cpu_score[0], optimizer_steps=R.opt_steps(opt),
                           data_digest=dg.value(), exposure_rows_tokens=expo, exposure_note="[rows, bytes under the loss mask]; loss=answer",
                           exposure_panels=dict(presented, taught_panel_distinct_presented=len(presented["taught_panel_distinct_presented"]))))
        print("e26", u, seed, "after", f"{fam}.{block}",
              "F1@k", {n: [sc[n][k]["exact"] for k in X16.KS] for n in sc if n.endswith("97")},
              "comp.exact", {n: v["exact"] for n, v in sc["compose"].items()}, "taught", sc["taught"]["exact"],
              f"cpu {stages[-1]['cpu']:.0f}s", flush=True)
        write()
        losses, expo, u0 = [], {}, 0
        si += 1
        if si <= len(X16.SCHEDULE):
            checkpoint(si, 0)
            nfam, nblock = X16.SCHEDULE[si - 1]
            if nblock == 1:
                seen_streams.append(X16.entry_stream(entries[nfam], seed + 100 + si))
            joint = X16.joint_stream(seen_streams, seed + si)
    assert presented["within_or_held_presented"] == 0, "a within or held-pair panel item was presented in training"
    wsha = R.save_weights(m, path.with_suffix(".pt"))
    write(dict(complete=dict(stages=len(stages), weights=path.with_suffix(".pt").name, weights_sha256=wsha,
                             tensors_sha256=R.tensors_sha(m.state_dict()), optimizer_steps=R.opt_steps(opt),
                             optimizer_read_back=C.opt_record(m, opt), data_digest=dg.value(),
                             cpu_total=cpu(), cpu_scoring=cpu_score[0], cpu_updates=cpu() - cpu_score[0])))
    ck.unlink(missing_ok=True)
    print("e26x life", b, u, seed, "complete", wsha[:16], flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])
    opt_ = lambda flag, default=None: a[a.index(flag) + 1] if flag in a else default
    outdir = opt_("--out", str(HERE / "exp45" / "e26x")); every = int(opt_("--ckpt-every", "512"))
    assert a[0] == "life" and every > 0, __doc__
    life(a[1], a[2], int(a[3]), outdir, cmd, every)
