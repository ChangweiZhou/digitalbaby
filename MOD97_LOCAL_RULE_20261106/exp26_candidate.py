"""E26 runner (brief OPUS_TASK_E26_GRU_UNDER_THE_SWEEP_20261111.md, revision 2): E20-B's GRU born and living under the sign sweep (arm S)
beside backprop (arm B), on E20-B's protocol, with the recurrent step written out (arch26) in both arms.

One training loop serves both arms, for births and lives; the arms share every line but the `local` flag (the brief's "The update, exactly"):

    logits = forward(m, x, local=<True for arm S, False for arm B>)     # credit through the whole item in both arms
    loss   = cross_entropy(logits[answer_mask], y[answer_mask])
    opt.zero_grad(); loss.backward(); opt.step()

exp11_cell.update is never called. AdamW, rate 0.001, betas (0.9, 0.98), weight decay 0.25 in both arms, the embedding table without decay
(exp16_integrated.adamw16; the groups are read back from the optimizer and recorded). No trace routine, no scan, no local head, no store.

    python3 exp26_candidate.py dev  <S|B> <seed> [--out DIR]
    python3 exp26_candidate.py life <S|B> <seed> [--out DIR]          # admits DIR/e26_dev_<arm>[dose]_<seed>.pt
    python3 exp26_candidate.py check <part> [seed] [--qual DIR]       # stage 1 checks; parts: step paths onestep scorer arch binding
                                                                      #   data (after the reduced runs), assemble -> DIR/E26_CHECK.json

Dose: E11_DEV_UPDATES and E11_STAGE_UPDATES, read by exp11_cell (full dose when unset). No other E16_ or E11_ variable may be set (E16_RESUME
included): exp20_reference refuses them at import. Every output path must not exist. Every receipt records its command, environment, host
(machine, platform, Python, torch, threads) and the hashes of the source files it ran, and a digest of every batch it was fed.
"""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp20_reference as R          # noqa: E402  (environment gate, harness imports, provenance helpers; not edited)

import torch as T                    # noqa: E402

X16, E, X, arch20 = R.X16, R.E, R.X, R.arch20
import arch26                        # noqa: E402

LR, BETAS = R.LR, R.BETAS
WD = 0.25
ARMS = {"S": True, "B": False}       # arm -> local (the sweep's way back) flag
SOURCES = ("arch26.py", "exp26_candidate.py", "arch20.py", "exp20_reference.py", "exp16_integrated.py", "arch17.py", "exp11_cell.py",
           "exp45.py", "exp3_state.py", "mod97_local.py")


# ---------------------------------------------------------------- provenance
def source_hashes():
    return {s: X16.sha_file(HERE / s) for s in SOURCES}


def git_state():
    import subprocess
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCES], cwd=HERE, capture_output=True, text=True).stdout.strip()
        return dict(commit=head, sources_modified_vs_commit=dirty.splitlines())
    except Exception as e:      # noqa: BLE001
        return dict(error=str(e))


def provenance(cmd):
    return dict(command=cmd, environment_at_start=R.ENV_AT_START, environment_set_by_runner=R.REQUIRED_ENV, host=R.host(), git=git_state(),
                sources_sha256=source_hashes())


# ---------------------------------------------------------------- optimizer (check 4): read back from the groups
def expected_optimizer(m):
    rest = [n for n, _ in m.named_parameters() if not n.startswith("book.")]
    return [dict(weight_decay=WD, lr=LR, betas=list(BETAS), params=rest), dict(weight_decay=0.0, lr=LR, betas=list(BETAS), params=["book.weight"])]


def make_opt(m):
    opt = X16.adamw16(m, LR, WD, ())
    g = R.opt_groups(m, opt)
    assert g == expected_optimizer(m), f"optimizer groups are not the expected ones: {g}"
    return opt


def opt_record(m, opt):
    return dict(name="AdamW", lr=LR, weight_decay=WD, betas=list(BETAS), groups=R.opt_groups(m, opt))


# ---------------------------------------------------------------- the update, exactly (one loop, both arms)
def step(m, opt, items, local):
    x, y, mask = X.batch_tensor(items)
    logits = arch26.forward(m, x, local, True)[0]
    loss = E.cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach())


# ---------------------------------------------------------------- data digest (check 3)
class Digest:
    """A running SHA-256 over every batch fed to the learner, as the tensors the loop builds (x, y, answer mask), in order."""

    def __init__(self):
        self.h, self.n = hashlib.sha256(), 0

    def add(self, items):
        x, y, mask = X.batch_tensor(items)
        for t in (x, y, mask):
            self.h.update(str(tuple(t.shape)).encode()); self.h.update(t.contiguous().numpy().tobytes())
        self.n += 1

    def value(self):
        return dict(batches=self.n, sha256=self.h.copy().hexdigest())


# ---------------------------------------------------------------- birth (develop17's plain path, as exp20_reference.dev)
def birth_path(out, arm, seed):
    return Path(out) / f"e26_dev_{arm}{E.DEVTAG}_{seed}.pt"


def life_path(out, arm, seed):
    return Path(out) / f"e26_life_{arm}{E.DEVTAG}{E.DRYTAG}_{seed}.json"


def dev(arm, seed, out, cmd):
    local = ARMS[arm]
    arch26.install()
    path = birth_path(out, arm, seed); rec_path = path.with_suffix(".json")
    R.refuse_existing(path, rec_path, str(path) + ".tmp")
    Path(out).mkdir(parents=True, exist_ok=True)
    m = E.new_net(seed)
    desc = arch26.describe(m)
    book0 = m.book.weight.detach().clone()
    opt = make_opt(m)
    census = None
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    pan = {E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV}
    bt = E.batches(tp, seed, E.KMAX["M"])
    dg = Digest()
    rec = dict(experiment="E26", kind="birth", arm=arm, local=local, net="G1", seed=seed, kmax=E.KMAX["M"], dev_updates=E.DEV_DOSE, block=X.BLOCK,
               batch=X.BATCH, optimizer=opt_record(m, opt),
               update="exp26_candidate.step: forward(local), answer-masked cross entropy, zero_grad, backward, AdamW step (credit through the whole item)",
               protocol="exp16_integrated.develop17 plain path (answer loss, four mod-89 operations, k 0..12), as exp20_reference.dev",
               model=desc, installed=dict(E_forward=f"{E.forward.__module__}.{E.forward.__name__}", E_new_net_class=desc["cls"]),
               provenance=provenance(cmd), complete=False, curve=[])
    t0 = time.process_time(); cpu_score = 0.0; curve = rec["curve"]; row = None
    for stop in range(X.BLOCK, E.DEV_DOSE + 1, X.BLOCK):
        losses = []
        for _ in range(X.BLOCK):
            items = next(bt); dg.add(items); losses.append(step(m, opt, items, local))
            if census is None:
                census = X16.zero_grad_report(m, X.batch_tensor(items)[0]); print("e26 dev credit census:", census["by_role"], flush=True)
        if stop % 4096 == 0 or stop == E.DEV_DOSE:
            cb = time.process_time()
            sealed = {n: {k: E.score(m, it) for k, it in pk.items()} for n, pk in pan.items()}
            cpu_score += time.process_time() - cb
            row = dict(step=stop, loss=sum(losses) / len(losses), finite_losses=all(map(lambda v: v == v and abs(v) != float("inf"), losses)),
                       cpu=time.process_time() - t0, cpu_scoring_so_far=cpu_score, cpu_updates_so_far=time.process_time() - t0 - cpu_score,
                       sealed=sealed, book_shift_by_role=X16.book_shift(m, book0), data_digest=dg.value())
            curve.append(row)
            print("e26 dev", arm, seed, stop, {n: [v[k]["exact"] for k in (0, 4, 8, 12, 20)] for n, v in row["sealed"].items()}, f"cpu {row['cpu']:.0f}s", flush=True)
            rec_path.write_text(json.dumps(dict(rec, credit_census_first_update=census), indent=1))
    assert row is not None, "the birth loop works in whole blocks of 512 updates: E11_DEV_UPDATES must be a positive multiple of 512"
    cb = time.process_time()
    row["silenced_k12"] = {n: E.score(m, pk[12], silence=True) for n, pk in pan.items()}
    cpu_score += time.process_time() - cb
    wsha = R.save_weights(m, path)
    rec.update(credit_census_first_update=census, optimizer_steps=R.opt_steps(opt), optimizer_read_back=opt_record(m, opt), data_digest=dg.value(),
               weights=path.name, weights_sha256=wsha, tensors_sha256=R.tensors_sha(m.state_dict()), cpu_total=time.process_time() - t0,
               cpu_scoring=cpu_score, cpu_updates=time.process_time() - t0 - cpu_score, complete=True)
    rec_path.write_text(json.dumps(rec, indent=1))
    print("e26 dev", arm, seed, "complete", wsha[:16], flush=True)


# ---------------------------------------------------------------- admission (check 6)
def admit_e26(bfile, arm, seed):
    """Admit an E26 birth for a life of `arm` and `seed`, from its complete receipt. Raises AssertionError on any mismatch."""
    bfile = Path(bfile); rp = bfile.with_suffix(".json")
    assert bfile.exists() and rp.exists(), f"{bfile.name}: birth or its receipt is missing"
    r = json.loads(rp.read_text())
    assert r.get("experiment") == "E26" and r.get("kind") == "birth", f"{bfile.name}: not an E26 birth receipt"
    assert r.get("complete") is True, f"{bfile.name}: receipt is not complete"
    fseed = bfile.stem.rsplit("_", 1)[1]
    assert str(r.get("seed")) == fseed == str(seed), f"{bfile.name}: receipt seed {r.get('seed')}, file-name seed {fseed}, life seed {seed} must agree"
    assert r.get("arm") == arm and r.get("local") == ARMS[arm] and bfile.stem.startswith(f"e26_dev_{arm}"), \
        f"{bfile.name}: receipt arm {r.get('arm')} (local {r.get('local')}), life arm {arm}"
    assert r.get("dev_updates") == E.DEV_DOSE and r["curve"] and r["curve"][-1]["step"] == E.DEV_DOSE, f"{bfile.name}: birth dose {r.get('dev_updates')}, expected {E.DEV_DOSE}"
    m = E.new_net(seed)
    exp_opt = dict(name="AdamW", lr=LR, weight_decay=WD, betas=list(BETAS), groups=expected_optimizer(m))
    assert r.get("optimizer") == exp_opt and r.get("optimizer_read_back") == exp_opt, f"{bfile.name}: optimizer settings differ"
    assert r["model"]["cls"] == arch26.describe(m)["cls"] and r["model"]["params"] == arch26.describe(m)["params"], f"{bfile.name}: network class or size differs"
    srcs = r.get("provenance", {}).get("sources_sha256", {})
    now = source_hashes()
    assert srcs == now, f"{bfile.name}: made by other sources: {sorted(k for k in set(now) | set(srcs) if srcs.get(k) != now.get(k))}"
    sha = X16.sha_file(bfile)
    assert r.get("weights_sha256") == sha, f"{bfile.name}: weights hash {sha[:12]} is not the receipt's {str(r.get('weights_sha256'))[:12]}"
    return m, sha, "E26 receipt: arm, seed (receipt, file name, life), dose, optimizer, sources and weights hash checked"


# ---------------------------------------------------------------- life (exp16_integrated.life arm D's protocol, as exp20_reference.life)
def life(arm, seed, out, cmd):
    local = ARMS[arm]
    arch26.install()
    path = life_path(out, arm, seed)
    R.refuse_existing(path, path.with_suffix(".pt"), str(path.with_suffix(".pt")) + ".tmp")
    Path(out).mkdir(parents=True, exist_ok=True)
    bfile = birth_path(out, arm, seed); m, birth_sha, admission = admit_e26(bfile, arm, seed)
    m.load_state_dict(T.load(bfile))
    desc = arch26.describe(m)
    opt = make_opt(m)
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
    seen_streams = []
    dg = Digest()
    t0 = time.process_time(); stages = []
    score = X16.score

    def evaluate():
        res = {n: {k: score(m, it, None) for k, it in pk.items()} for n, pk in P.items()}
        res["compose"] = {n: score(m, it, None) for n, it in comp.items()}
        res["silenced_k12"] = {n: score(m, pk[12], None, silence=True) for n, pk in P.items() if not n.endswith("89")}
        res["silenced_compose"] = {n: score(m, it, None, silence=True) for n, it in comp.items() if n != "within"}
        res["strict_k12"] = {n: score(m, it, None) for n, it in SP.items()}
        res["taught"] = score(m, taught, None)
        return res

    c0 = time.process_time(); scores0 = evaluate(); cpu_score = [time.process_time() - c0]
    settings = dict(arm=arm, local=local, net="G1", model=desc, installed=dict(E_forward=f"{E.forward.__module__}.{E.forward.__name__}", E_new_net_class=desc["cls"]),
                    optimizer=opt_record(m, opt), stage_updates=E.STAGE_UPDATES, dev_updates=E.DEV_DOSE, batch=X.BATCH, arch16=X16.ARCH16,
                    bookdecay=X16.BOOKDECAY16, compgap=X16.COMPGAP16, store=False, panel_sha256=X16.panel_sha(P, comp, SP), strict_panel_counts=strict_counts,
                    taught_panel=tcheck, birth=str(bfile), birth_sha256=birth_sha, birth_admission=admission,
                    update="exp26_candidate.step (the same loop as the birth)",
                    protocol="exp16_integrated.life arm D's protocol (joint stream of families admitted so far), as exp20_reference.life; decay 0.25",
                    provenance=provenance(cmd))
    stages.append(dict(stage=0, task=None, scores=scores0, shortcut_ceilings=X16.shortcut_ceilings(comp), entries={f: len(v) for f, v in entries.items()}, settings=settings))
    for si, (fam, block) in enumerate(X16.SCHEDULE, 1):
        if block == 1:
            seen_streams.append(X16.entry_stream(entries[fam], seed + 100 + si))
        joint = X16.joint_stream(seen_streams, seed + si)
        losses, expo = [], {}
        for u in range(E.STAGE_UPDATES):
            batch = [next(joint) for _ in range(X.BATCH)]
            for p_, t_ in batch:
                e_ = expo.setdefault(f"joint/{fam_of(p_)}", [0, 0]); e_[0] += 1; e_[1] += len(t_)
                k_ = tuple(p_)
                if k_ in taught_keys:
                    presented["taught_panel_items_presented"] += 1; presented["taught_panel_distinct_presented"].add(k_)
                if k_ in unseen_keys:
                    presented["within_or_held_presented"] += 1
            dg.add(batch)
            losses.append(step(m, opt, batch, local))
            if si == 1 and u == 0:
                stages[0]["settings"]["zero_grad_first_update"] = X16.zero_grad_report(m, X.batch_tensor(batch)[0])
                print("e26 zero-grad check:", stages[0]["settings"]["zero_grad_first_update"]["by_role"], flush=True)
        cb = time.process_time(); sc = evaluate(); ca = time.process_time(); cpu_score[0] += ca - cb
        stages.append(dict(stage=si, task=f"{fam}.{block}", loss=sum(losses[-512:]) / min(512, len(losses)),
                           finite_losses=all(v == v and abs(v) != float("inf") for v in losses), cpu=cb - t0, scores=sc,
                           cpu_scoring_so_far=cpu_score[0], cpu_updates_so_far=ca - t0 - cpu_score[0], optimizer_steps=R.opt_steps(opt),
                           data_digest=dg.value(), exposure_rows_tokens=expo, exposure_note="[rows, bytes under the loss mask]; loss=answer",
                           exposure_panels=dict(presented, taught_panel_distinct_presented=len(presented["taught_panel_distinct_presented"]))))
        print("e26", arm, seed, "after", f"{fam}.{block}",
              "F1@k", {n: [sc[n][k]["exact"] for k in X16.KS] for n in sc if n.endswith("97")},
              "comp.exact", {n: v["exact"] for n, v in sc["compose"].items()}, "taught", sc["taught"]["exact"],
              f"cpu {stages[-1]['cpu']:.0f}s", flush=True)
        path.write_text(json.dumps(dict(experiment="E26", kind="life", arm=arm, local=local, net="G1", seed=seed, schedule=X16.SCHEDULE, stages=stages), indent=1))
    assert presented["within_or_held_presented"] == 0, "a within or held-pair panel item was presented in training"
    wsha = R.save_weights(m, path.with_suffix(".pt"))
    path.write_text(json.dumps(dict(experiment="E26", kind="life", arm=arm, local=local, net="G1", seed=seed, schedule=X16.SCHEDULE, stages=stages,
                                    complete=dict(stages=len(stages), weights=path.with_suffix(".pt").name, weights_sha256=wsha,
                                                  tensors_sha256=R.tensors_sha(m.state_dict()), optimizer_steps=R.opt_steps(opt),
                                                  optimizer_read_back=opt_record(m, opt), data_digest=dg.value(),
                                                  cpu_total=time.process_time() - t0, cpu_scoring=cpu_score[0],
                                                  cpu_updates=time.process_time() - t0 - cpu_score[0])), indent=1))
    print("e26 life", arm, seed, "complete", wsha[:16], flush=True)



# ================================================================ stage 1 checks (the brief's checks 1 to 7)
SEEDS = (73129, 93683, 116089)
E20B = X.OUT / "e20b"


def e25():
    import e25_alignment                 # the coordinator's reference for the written-out step and an independent gradient (not edited)
    return e25_alignment


def load_e20b(seed, which):
    """E20-B's saved G1 file (which = birth | life), its hash checked against its own receipt; or an untrained G1 (which = untrained)."""
    m = arch20.new_net(seed, "G1")
    if which == "untrained":
        return m, None
    f = E20B / f"e20_{'dev' if which == 'birth' else 'life'}_G1_{seed}.pt"
    rec = json.loads(f.with_suffix(".json").read_text())
    want = rec["complete"]["weights_sha256"] if which == "life" else rec["weights_sha256"]
    sha = X16.sha_file(f)
    assert sha == want, f"{f.name}: the weights are not the ones its receipt names"
    m.load_state_dict(T.load(f))
    return m, dict(file=str(f.relative_to(HERE)), sha256=sha)


def check_batches(seed):
    """mixed: as E25's fixtures (64 mod-97 operations at k = 12 and 64 new instances of taught pairs); birth: the first birth batch."""
    P, comp = X16.panels(seed)
    mixed = P["add97"][12][:32] + P["sub97"][12][:32] + comp["within"][:64]
    birth = next(E.batches([(t, E.split(t, .5, seed)[0]) for t in E.DEV], seed, E.KMAX["M"]))
    return dict(mixed=mixed, birth=birth)


def grads_of(m, fwd, x, y, mask):
    m.zero_grad()
    logits = fwd(m, x)
    loss = E.cross_entropy(logits[mask], y[mask]); loss.backward()
    return logits.detach(), {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in m.named_parameters()}


def maxrel(ga, gb):
    return max(float((ga[n] - gb[n]).abs().max()) / max(float(ga[n].abs().max()), 1e-12) for n in ga)


def check_step():
    """Check 1. The written-out forward against arch20's library forward, in every logit (1e-5), and the exact path's gradients against the
    library's (1e-4 relative), on an untrained network and on E20-B's saved G1 birth and life files of all three seeds. A version with two
    gates' recurrent weights swapped must fail. Also: bitwise against e25_alignment.forward_rnn / grads (the coordinator's reference)."""
    ref = e25(); out = {}; ok = True
    for seed in SEEDS:
        for which in ("untrained", "birth", "life"):
            m, src = load_e20b(seed, which)
            for bname, items in check_batches(seed).items():
                x, y, mask = X.batch_tensor(items)
                lo_lib, g_lib = grads_of(m, lambda m_, x_: arch20._forward_rnn(m_, x_, False)[0], x, y, mask)
                lo_mine, g_mine = grads_of(m, lambda m_, x_: arch26.forward(m_, x_, False, True)[0], x, y, mask)
                lo_e25, g_e25 = grads_of(m, lambda m_, x_: ref.forward_rnn(m_, x_, False)[0], x, y, mask)
                _, g_e25g, _, _ = ref.grads(m, x, y, mask, False)
                w = m.rnn.weight_hh_l0.data.clone(); n_ = m.units
                m.rnn.weight_hh_l0.data = T.cat([w[n_:2 * n_], w[:n_], w[2 * n_:]])
                with T.no_grad():
                    swapped = float((arch26.forward(m, x, False, True)[0] - lo_lib).abs().max())
                m.rnn.weight_hh_l0.data = w
                r = dict(source=src, batch=bname, items=len(items), width=x.shape[1],
                         logits_max_abs_diff_vs_library=float((lo_mine - lo_lib).abs().max()),
                         grads_max_rel_diff_vs_library=maxrel(g_lib, g_mine),
                         two_gates_swapped_logits_max_abs_diff_vs_library=swapped,
                         bitwise_equal_to_e25_forward_rnn=bool(T.equal(lo_mine, lo_e25)) and all(T.equal(g_mine[k], g_e25[k]) for k in g_mine),
                         bitwise_equal_to_e25_grads=all(T.equal(g_mine[k], g_e25g[k]) for k in g_mine))
                d_ = (lo_mine - lo_lib).abs(); i_ = int(d_.argmax()); v_ = float(lo_lib.flatten()[i_])
                r["diagnostics_not_part_of_the_criterion"] = dict(
                    logit_at_largest_difference=v_, float32_spacing_at_that_logit=float(T.finfo(T.float32).eps) * 2.0 ** math.floor(math.log2(abs(v_))) if v_ else None,
                    largest_abs_logit=float(lo_lib.abs().max()),
                    answer_positions_with_a_different_argmax=int((lo_mine.argmax(-1) != lo_lib.argmax(-1))[mask].sum()), answer_positions=int(mask.sum()))
                r["pass"] = r["logits_max_abs_diff_vs_library"] <= 1e-5 and r["grads_max_rel_diff_vs_library"] <= 1e-4 and swapped > 1e-5
                r["swapped_version_fails"] = swapped > 1e-5
                ok = ok and r["pass"]
                out[f"{seed} {which} {bname}"] = r
    coord = {}
    for seed in SEEDS:
        f = X.OUT / f"E25_FIXTURES_{seed}.json"
        if f.exists():
            g = json.loads(f.read_text())["G1"]
            coord[str(seed)] = {k: g[k] for k in ("outputs_max_difference", "gradients_max_relative_difference",
                                                  "written_out_step_with_two_gates_swapped_differs_from_the_library_by", "pass")}
    return dict(cases=out, coordinator_e25_fixtures_untrained_for_comparison=coord, tolerances=dict(logits_abs=1e-5, grads_rel=1e-4, swapped_must_exceed=1e-5),
                **{"pass": ok})


def active_rows(x, mask):
    last = [int(mask[i].nonzero().max()) for i in range(x.shape[0])]
    return sorted({int(x[i, j]) for i in range(x.shape[0]) for j in range(last[i] + 1)} - {E.BOS})


def check_paths():
    """Check 2. Same forward values under both paths; under the sweep the top layer's own gradient equals the exact one and the recurrent
    matrix's does not; every parameter and every active embedding row receives a finite, non-zero gradient under both (BOS exempt)."""
    out = {}; ok = True
    for seed in SEEDS:
        for which in ("untrained", "birth"):
            m, src = load_e20b(seed, which)
            for bname, items in check_batches(seed).items():
                x, y, mask = X.batch_tensor(items)
                lo_b, g_b = grads_of(m, lambda m_, x_: arch26.forward(m_, x_, False, True)[0], x, y, mask)
                lo_s, g_s = grads_of(m, lambda m_, x_: arch26.forward(m_, x_, True, True)[0], x, y, mask)
                act = active_rows(x, mask)
                per = {}
                for arm, g in (("S", g_s), ("B", g_b)):
                    per[arm] = dict(params_without_finite_nonzero_grad=[n for n, v in g.items() if v is None or not bool(T.isfinite(v).all()) or not bool(v.any())],
                                    active_rows=len(act), active_rows_without_grad=[r_ for r_ in act if not bool(g["book.weight"][r_].any())],
                                    bos_row_grad_zero=not bool(g["book.weight"][E.BOS].any()))
                r = dict(source=src, batch=bname,
                         forward_bitwise_equal=bool(T.equal(lo_s, lo_b)),
                         top_layer_grad_max_abs_diff={n: float((g_s[n] - g_b[n]).abs().max()) for n in ("l3.weight", "l3.bias")},
                         recurrent_grad_max_abs_diff={n: float((g_s[n] - g_b[n]).abs().max()) for n in ("rnn.weight_hh_l0", "rnn.bias_hh_l0")},
                         below_top_grad_differs={n: float((g_s[n] - g_b[n]).abs().max()) > 0 for n in g_b if not n.startswith("l3.")},
                         per_arm=per)
                r["pass"] = (r["forward_bitwise_equal"] and all(v <= 1e-7 for v in r["top_layer_grad_max_abs_diff"].values())
                             and r["recurrent_grad_max_abs_diff"]["rnn.weight_hh_l0"] > 1e-6
                             and all(not v["params_without_finite_nonzero_grad"] and not v["active_rows_without_grad"] and v["bos_row_grad_zero"] for v in per.values()))
                ok = ok and r["pass"]
                out[f"{seed} {which} {bname}"] = r
    return dict(cases=out, tolerances=dict(top_layer_abs=1e-7, recurrent_must_exceed=1e-6, note="1e-7 and 1e-6 as in e25_alignment.fixtures"), **{"pass": ok})


def tensor_cmp(sa, sb):
    return {k: dict(equal=bool(T.equal(sa[k], sb[k])), max_abs_diff=float((sa[k] - sb[k]).abs().max())) for k in sa}


def check_onestep():
    """Check 2a. From one saved state (E20-B's G1 birth file of seed 73129) and one batch (that seed's first birth batch): one step of the
    production loop in each arm, against the same AdamW given the gradient computed independently by e25_alignment.grads with the same flag.
    Two starting states: a fresh optimizer, and after two production steps of that arm (non-zero moments). The two arms' weights after the step
    must differ below the top layer. Fixtures: the harness's rule update E.update(..., "T") must fail; E.update(..., "D") is reported."""
    import copy
    ref = e25(); seed = 73129
    arch26.install()
    base, src = load_e20b(seed, "birth")
    bt = E.batches([(t, E.split(t, .5, seed)[0]) for t in E.DEV], seed, E.KMAX["M"])
    warm = [next(bt), next(bt)]; items = next(bt)
    x, y, mask = X.batch_tensor(items)

    def start(arm, warmed):
        m = copy.deepcopy(base); opt = make_opt(m)
        if warmed:
            for w in warm:
                step(m, opt, w, ARMS[arm])
        return m, opt

    def independent(m, opt, local):
        _, gp, _, _ = ref.grads(m, x, y, mask, local)
        opt.zero_grad()
        for n, p_ in m.named_parameters():
            p_.grad = gp[n].clone()
        opt.step()

    out = dict(saved_state=src, batch="first birth batch of seed 73129 (after two warm-up batches for the warmed case)", cases={})
    ok = True
    for warmed in (False, True):
        after = {}
        for arm, local in ARMS.items():
            m1, o1 = start(arm, warmed); m2, o2 = copy.deepcopy((m1, o1))
            step(m1, o1, items, local); independent(m2, o2, local)
            cmp_ = tensor_cmp(m1.state_dict(), m2.state_dict())
            after[arm] = m1.state_dict()
            c = dict(production_equals_independent_tensor_for_tensor=all(v["equal"] for v in cmp_.values()),
                     max_abs_diff=max(v["max_abs_diff"] for v in cmp_.values()), per_tensor=cmp_, optimizer_steps=R.opt_steps(o1),
                     optimizer_read_back=opt_record(m1, o1))
            ok = ok and c["production_equals_independent_tensor_for_tensor"] and all(g["weight_decay"] == (0.0 if g["params"] == ["book.weight"] else WD) for g in c["optimizer_read_back"]["groups"])
            out["cases"][f"{'after two warm-up steps' if warmed else 'fresh optimizer'} arm {arm}"] = c
        d = tensor_cmp(after["S"], after["B"])
        below = {k: not v["equal"] for k, v in d.items() if not k.startswith("l3.")}
        out["cases"][f"{'after two warm-up steps' if warmed else 'fresh optimizer'} arms S and B after the step"] = dict(
            per_tensor_max_abs_diff={k: v["max_abs_diff"] for k, v in d.items()}, every_tensor_below_top_differs=all(below.values()))
        ok = ok and all(below.values())
    # fixtures: the harness's rule update must fail this check
    fx = {}
    m, o = start("S", False); m_ref, o_ref = copy.deepcopy((m, o)); independent(m_ref, o_ref, True)
    try:
        E.update(m, o, items, "T"); fx["E.update(T) as installed"] = "ran (not expected)"
    except AssertionError as e:
        fx["E.update(T) as installed"] = f"refused: {e}"
    real = E.trace_grads
    try:
        E.trace_grads = E.trace_grads_scan = lambda m_, it_: None
        m, o = start("S", False); E.update(m, o, items, "T")
        cmp_ = tensor_cmp(m.state_dict(), m_ref.state_dict())
        fx["E.update(T) with the trace routine disabled, against arm S's independent step"] = dict(
            equal=all(v["equal"] for v in cmp_.values()), max_abs_diff=max(v["max_abs_diff"] for v in cmp_.values()),
            tensors_differing=sorted(k for k, v in cmp_.items() if not v["equal"]))
    finally:
        E.trace_grads = E.trace_grads_scan = real
    fx["harness_rule_update_fails_the_check"] = fx["E.update(T) as installed"].startswith("refused") and \
        not fx["E.update(T) with the trace routine disabled, against arm S's independent step"]["equal"]
    m, o = start("B", False); m2, o2 = copy.deepcopy((m, o)); E.update(m, o, items, "D"); step(m2, o2, items, False)
    fx["E.update(D), reported only: equals arm B's production step"] = all(v["equal"] for v in tensor_cmp(m.state_dict(), m2.state_dict()).values())
    out["fixtures"] = fx
    ok = ok and fx["harness_rule_update_fails_the_check"]
    out["pass"] = ok
    return out


def check_scorer(seed):
    """Check 2b. arch26.forward has the harness's signature and is what exp16_integrated.score runs, silence passed through; the three stub
    fixtures of E20-B's stage 1; with silence the readout sees a zero state, as in arch20; and E20-B's saved end-of-life G1 file of the seed,
    loaded into the written-out network and scored by the production scorer, against the counts in its own receipt."""
    import inspect
    out = {}
    sig = list(inspect.signature(arch26.forward).parameters.items())
    out["signature"] = [f"{n}={v.default}" if v.default is not inspect._empty else n for n, v in sig]
    out["signature_ok"] = out["signature"] == ["m", "x", "local", "through_time", "keep=False", "silence=False"]
    fxt = R.scoring_fixtures(seed)                    # stub responder; installs G0 for its own birth-scorer test
    out["stub_fixtures"] = dict(fixtures=fxt["fixtures"], fixtures_ok=fxt["fixtures_ok"], life_scorer_cap=fxt["life_scorer_cap"])
    arch26.install()
    m, src = load_e20b(seed, "life")
    out["source"] = src
    out["installed_forward"] = f"{E.forward.__module__}.{E.forward.__name__}"
    P, comp = X16.panels(seed)
    calls = []
    real = E.forward
    try:
        E.forward = lambda m_, x_, local, through_time, keep=False, silence=False: (calls.append((local, through_time, silence)), real(m_, x_, local, through_time, keep, silence))[1]
        X16.score(m, comp["+-"][:4]); X16.score(m, comp["+-"][:4], None, silence=True)
    finally:
        E.forward = real
    out["scorer_calls_arch26_forward"] = dict(calls=len(calls), flags_seen=sorted(set(calls)),
                                              ok=(False, True, False) in calls and (False, True, True) in calls and E.forward is arch26.forward)
    x = X.batch_tensor(comp["within"][:64])[0]
    with T.no_grad():
        lo_s = arch26.forward(m, x, False, True, silence=True)[0]
        lo_lib = arch20._forward_rnn(m, x, True)[0]
        z1 = T.tanh(m.l1(m.book(x))); manual = m.l3(T.tanh(m.l2(z1) + m.lm.bias))
        lo_on = arch26.forward(m, x, False, True)[0]
    out["silence"] = dict(vs_arch20_silenced_max_abs_diff=float((lo_s - lo_lib).abs().max()), vs_zero_state_readout_max_abs_diff=float((lo_s - manual).abs().max()),
                          silenced_differs_from_unsilenced=float((lo_s - lo_on).abs().max()))
    out["silence"]["ok"] = out["silence"]["vs_arch20_silenced_max_abs_diff"] <= 1e-5 and out["silence"]["vs_zero_state_readout_max_abs_diff"] <= 1e-5 \
        and out["silence"]["silenced_differs_from_unsilenced"] > 1e-3
    # the receipt's own counts: every block the life scored at its last checkpoint
    SP, _ = X16.strict_panels(seed); taught = R.taught_panel(seed)
    rec = json.loads((E20B / f"e20_life_G1_{seed}.json").read_text())["stages"][-1]["scores"]
    c0 = time.process_time(); score = X16.score
    mine = {n: {k: score(m, it, None) for k, it in pk.items()} for n, pk in P.items()}
    mine["compose"] = {n: score(m, it, None) for n, it in comp.items()}
    mine["silenced_k12"] = {n: score(m, pk[12], None, silence=True) for n, pk in P.items() if not n.endswith("89")}
    mine["silenced_compose"] = {n: score(m, it, None, silence=True) for n, it in comp.items() if n != "within"}
    mine["strict_k12"] = {n: score(m, it, None) for n, it in SP.items()}
    mine["taught"] = score(m, taught, None)
    out["cpu_scoring_one_checkpoint"] = time.process_time() - c0
    mine = json.loads(json.dumps(mine))              # the receipt's form (string keys for k)
    diffs = []

    def walk(a, b, path):
        if isinstance(b, dict) and "exact" in b:
            for f_ in ("n", "exact", "final"):
                if a.get(f_) != b.get(f_):
                    diffs.append(dict(block=path, field=f_, written_out=a.get(f_), receipt=b.get(f_), difference=(a.get(f_) or 0) - (b.get(f_) or 0)))
            return 1
        return sum(walk(a.get(k, {}), v, f"{path}/{k}") for k, v in b.items())
    blocks = walk(mine, rec, "stage6")
    req = ("compose", "silenced_k12", "silenced_compose")
    out["receipt_comparison"] = dict(blocks_compared=blocks, differences=diffs,
                                     required_blocks_equal=not [d for d in diffs if d["block"].split("/")[1] in req],
                                     all_blocks_equal=not diffs, compose_counts={n: mine["compose"][n]["exact"] for n in mine["compose"]})
    out["pass"] = out["signature_ok"] and out["stub_fixtures"]["fixtures_ok"] and out["scorer_calls_arch26_forward"]["ok"] and out["silence"]["ok"] \
        and out["receipt_comparison"]["required_blocks_equal"]
    return out


def check_arch():
    """Check 5. Causality, rows, reset for the written-out network (E20-B stage 1, amendment 11), under both paths' forward."""
    seed = 73129; arch26.install()
    m = E.new_net(seed); items = R.mixed_items(seed)
    x, y, mask = X.batch_tensor(items); lens = [len(p) + len(t) for p, t in items]
    out = dict(batch=dict(rows=x.shape[0], width=x.shape[1], row_lengths=sorted(set(lens))))
    f = lambda xx, local=False: arch26.forward(m, xx, local, True)[0]
    with T.no_grad():
        lg = f(x); lgs = f(x, True)
    out["forward_equal_under_both_paths"] = bool(T.equal(lg, lgs))
    out["shapes"] = dict(x=list(x.shape), logits=list(lg.shape), ok=list(lg.shape) == [x.shape[0], x.shape[1], 257])
    g = T.Generator().manual_seed(11); worst, n = 0.0, 0
    for i in range(x.shape[0]):
        for j in (2, lens[i] // 2, lens[i] - 1):
            x2 = x.clone(); x2[i, j + 1:] = T.randint(0, 256, (x.shape[1] - j - 1,), generator=g)
            with T.no_grad():
                l2 = f(x2)
            worst = max(worst, float((l2[i, :j + 1] - lg[i, :j + 1]).abs().max())); n += 1
    out["causality"] = dict(cases=n, max_abs_diff_earlier_logits=worst, exact=worst == 0.0)
    x3 = x.clone(); x3[1:] = T.randint(0, 256, x3[1:].shape, generator=g); perm = T.randperm(x.shape[0], generator=g)
    with T.no_grad():
        l3 = f(x3); l4 = f(x[perm])
    out["rows"] = dict(other_rows_changed_max_abs_diff=float((l3[0] - lg[0]).abs().max()), reordered_max_abs_diff=float((l4 - lg[perm]).abs().max()))
    out["rows"]["exact"] = out["rows"]["other_rows_changed_max_abs_diff"] == 0.0 and out["rows"]["reordered_max_abs_diff"] == 0.0
    with T.no_grad():
        f(x3); l5 = f(x)
        alone = max(float((f(x[i:i + 1, :lens[i]])[0] - lg[i, :lens[i]]).abs().max()) for i in range(x.shape[0]))
    out["state_reset"] = dict(module_buffers=[n_ for n_, _ in m.named_buffers()], same_batch_after_another_max_abs_diff=float((l5 - lg).abs().max()),
                              each_row_alone_vs_in_batch_max_abs_diff=alone, tolerance_for_row_alone=1e-5)
    out["state_reset"]["ok"] = not out["state_reset"]["module_buffers"] and out["state_reset"]["same_batch_after_another_max_abs_diff"] == 0.0 and alone <= 1e-5
    out["model"] = {k: v for k, v in arch26.describe(m).items() if k != "params_by_tensor"}
    out["same_initial_values_as_arch20_G1"] = all(T.equal(a, b) for a, b in zip(m.state_dict().values(), arch20.new_net(seed, "G1").state_dict().values()))
    out["bos_eos"] = dict(vocab=m.book.num_embeddings, padding_idx=m.book.padding_idx, BOS=E.BOS, EOS=E.EOS, outputs=m.l3.out_features,
                          ok=m.book.num_embeddings == 258 and m.book.padding_idx == E.BOS and m.l3.out_features == 257)
    opt = make_opt(m)
    out["optimizer_read_back"] = [dict(weight_decay=g_["weight_decay"], n_params=len(g_["params"]), names=g_["params"] if len(g_["params"]) < 4 else g_["params"][:3] + ["..."])
                                  for g_ in R.opt_groups(m, opt)]
    out["pass"] = (out["forward_equal_under_both_paths"] and out["shapes"]["ok"] and out["causality"]["exact"] and out["rows"]["exact"]
                   and out["state_reset"]["ok"] and out["same_initial_values_as_arch20_G1"] and out["bos_eos"]["ok"] and out["model"]["params"] == 1124113)
    return out


def fake_birth(td, arm, seed, **over):
    """A birth file and complete receipt as `dev` writes them (untrained weights), for the admission fixtures."""
    m = E.new_net(seed); opt = make_opt(m)
    f = Path(td) / f"e26_dev_{arm}{E.DEVTAG}_{seed}.pt"
    wsha = R.save_weights(m, f)
    rec = dict(experiment="E26", kind="birth", arm=arm, local=ARMS[arm], net="G1", seed=seed, dev_updates=E.DEV_DOSE, curve=[dict(step=E.DEV_DOSE)],
               optimizer=opt_record(m, opt), optimizer_read_back=opt_record(m, opt), model=arch26.describe(m),
               provenance=dict(sources_sha256=source_hashes()), weights=f.name, weights_sha256=wsha, complete=True)
    rec.update(over)
    f.with_suffix(".json").write_text(json.dumps(rec))
    return f, rec


def check_binding():
    """Check 6. A life refuses a birth of the other arm, of another seed, with a changed weights file, or made by other sources; no existing
    file is overwritten."""
    import shutil, tempfile
    arch26.install(); seed = 73129; other = 93683; res = {}

    def attempt(f, arm, s):
        try:
            return "admitted: " + admit_e26(f, arm, s)[2]
        except AssertionError as e:
            return f"refused: {e}"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for arm in ARMS:
            (td / arm).mkdir()
        own, rec = fake_birth(td / "S", "S", seed)
        res["own_birth_own_arm_own_seed"] = attempt(own, "S", seed)
        b_own, _ = fake_birth(td / "B", "B", seed)
        res["arm_B_birth_presented_to_arm_S_life"] = attempt(b_own, "S", seed)
        ren = td / "ren" / own.name; ren.parent.mkdir(); shutil.copy(b_own, ren); shutil.copy(b_own.with_suffix(".json"), ren.with_suffix(".json"))
        res["arm_B_birth_renamed_as_arm_S"] = attempt(ren, "S", seed)
        res["birth_presented_to_another_seeds_life"] = attempt(own, "S", other)
        cp = td / "cp" / f"e26_dev_S{E.DEVTAG}_{other}.pt"; cp.parent.mkdir(); shutil.copy(own, cp); shutil.copy(own.with_suffix(".json"), cp.with_suffix(".json"))
        res["birth_copied_under_another_seed_for_that_seed"] = attempt(cp, "S", other)
        ch = td / "chg" / own.name; ch.parent.mkdir()
        sd = T.load(own); sd["l3.bias"] = sd["l3.bias"].clone(); sd["l3.bias"][0] += 1e-3; T.save(sd, ch); shutil.copy(own.with_suffix(".json"), ch.with_suffix(".json"))
        res["changed_weights_file"] = attempt(ch, "S", seed)
        srcs = dict(rec["provenance"]["sources_sha256"]); srcs["arch26.py"] = "0" * 64
        os_ = td / "src" / own.name; os_.parent.mkdir(); shutil.copy(own, os_)
        os_.with_suffix(".json").write_text(json.dumps(dict(rec, provenance=dict(sources_sha256=srcs)))); res["birth_made_by_other_sources"] = attempt(os_, "S", seed)
        dec = td / "dec" / own.name; dec.parent.mkdir(); shutil.copy(own, dec)
        bad_opt = json.loads(json.dumps(rec["optimizer"])); bad_opt["groups"][0]["weight_decay"] = 0.3
        dec.with_suffix(".json").write_text(json.dumps(dict(rec, optimizer=bad_opt, optimizer_read_back=bad_opt))); res["birth_with_decay_0.3"] = attempt(dec, "S", seed)
        inc = td / "inc" / own.name; inc.parent.mkdir(); shutil.copy(own, inc)
        inc.with_suffix(".json").write_text(json.dumps(dict(rec, complete=False))); res["incomplete_receipt"] = attempt(inc, "S", seed)
        e20 = td / "e20" / own.name; e20.parent.mkdir(); shutil.copy(E20B / f"e20_dev_G1_{seed}.pt", e20)
        shutil.copy(E20B / f"e20_dev_G1_{seed}.json", e20.with_suffix(".json")); res["E20-B_G1_birth_receipt_presented"] = attempt(e20, "S", seed)
        for what, p_ in (("birth_output_exists", own), ("life_output_exists", td / "S" / f"e26_life_S{E.DEVTAG}{E.DRYTAG}_{seed}.json")):
            p_.touch()
            try:
                R.refuse_existing(p_); res[what] = "not refused"
            except AssertionError as e:
                res[what] = f"refused: {e}"
        # the runner's own entry points refuse an existing output
        for what, fn in (("dev() with its birth file present", lambda: dev("S", seed, td / "S", "fixture")),
                         ("life() with its receipt present", lambda: life("S", seed, td / "S", "fixture"))):
            try:
                fn(); res[what] = "not refused"
            except AssertionError as e:
                res[what] = f"refused: {e}"
        res = {k: v.replace(str(td), "<tmp>") for k, v in res.items()}
    res["pass"] = res["own_birth_own_arm_own_seed"].startswith("admitted") and all(v.startswith("refused") for k, v in res.items() if k not in ("own_birth_own_arm_own_seed", "pass"))
    return res


def check_data(qual):
    """Checks 3, 4 and 7 from the reduced-dose receipts of seed 73129 in `qual`: the two arms were fed the same batches in the same order (a
    digest of every batch, birth and life); both record AdamW with decay 0.25 except the embedding table (read back from the groups); both runs
    complete; update CPU-seconds per update and scoring CPU-seconds per checkpoint, and a forecast for stages 2 and 3."""
    seed = 73129; recs = {}
    for arm in ARMS:
        b = json.loads((qual / f"e26_dev_{arm}{E.DEVTAG}_{seed}.json").read_text())
        l_ = json.loads((qual / f"e26_life_{arm}{E.DEVTAG}{E.DRYTAG}_{seed}.json").read_text())
        recs[arm] = (b, l_)
    (bS, lS), (bB, lB) = recs["S"], recs["B"]
    d3 = dict(birth_rows=[(r["step"], r["data_digest"]) for r in bS["curve"]] == [(r["step"], r["data_digest"]) for r in bB["curve"]],
              birth_final=bS["data_digest"] == bB["data_digest"],
              life_stages=[s.get("data_digest") for s in lS["stages"]] == [s.get("data_digest") for s in lB["stages"]],
              life_final=lS["complete"]["data_digest"] == lB["complete"]["data_digest"],
              birth_digest=bS["data_digest"], life_digest=lS["complete"]["data_digest"])
    d3["pass"] = d3["birth_rows"] and d3["birth_final"] and d3["life_stages"] and d3["life_final"]
    d4 = {}
    for arm, (b, l_) in recs.items():
        m = arch26.new_net(seed); want = expected_optimizer(m)
        for kind, gr in (("birth", b["optimizer_read_back"]["groups"]), ("life", l_["complete"]["optimizer_read_back"]["groups"]),
                         ("life_settings", l_["stages"][0]["settings"]["optimizer"]["groups"])):
            d4[f"{arm} {kind}"] = dict(groups=[dict(weight_decay=g["weight_decay"], n_params=len(g["params"]), embedding_only=g["params"] == ["book.weight"]) for g in gr],
                                       ok=gr == want)
    d4["pass"] = all(v["ok"] for v in d4.values())
    d7 = {}
    for arm, (b, l_) in recs.items():
        nb, nl = b["dev_updates"], l_["stages"][0]["settings"]["stage_updates"] * 6
        nrow = len(b["curve"])
        d7[arm] = dict(birth=dict(complete=b["complete"], updates=nb, optimizer_steps=b["optimizer_steps"], weights_sha256=b["weights_sha256"],
                                  finite_losses=all(r["finite_losses"] for r in b["curve"]),
                                  cpu_updates=b["cpu_updates"], cpu_per_update=b["cpu_updates"] / nb, cpu_scoring=b["cpu_scoring"], scored_rows=nrow,
                                  cpu_scoring_per_row=b["curve"][-1]["cpu_scoring_so_far"] / nrow,
                                  cpu_scoring_silenced_end=b["cpu_scoring"] - b["curve"][-1]["cpu_scoring_so_far"], curve_last_row=b["curve"][-1]["sealed"]),
                       life=dict(complete=bool(l_.get("complete")), stages=len(l_["stages"]), updates=nl, optimizer_steps=l_["complete"]["optimizer_steps"],
                                 weights_sha256=l_["complete"]["weights_sha256"], finite_losses=all(s.get("finite_losses", True) for s in l_["stages"]),
                                 cpu_updates=l_["complete"]["cpu_updates"], cpu_per_update=l_["complete"]["cpu_updates"] / nl,
                                 cpu_scoring=l_["complete"]["cpu_scoring"], cpu_scoring_per_checkpoint=l_["complete"]["cpu_scoring"] / 7,
                                 end=dict(compose={n: v["exact"] for n, v in l_["stages"][-1]["scores"]["compose"].items()}, taught=l_["stages"][-1]["scores"]["taught"]["exact"])))
        bsc = d7[arm]["birth"]["cpu_scoring_per_row"] * 6 + d7[arm]["birth"]["cpu_scoring_silenced_end"]
        bu = d7[arm]["birth"]["cpu_per_update"] * 24576 + bsc
        lu = d7[arm]["life"]["cpu_per_update"] * 36864 + d7[arm]["life"]["cpu_scoring_per_checkpoint"] * 7
        d7[arm]["forecast_full_dose"] = dict(birth_cpu_s=round(bu), birth_updates_cpu_s=round(d7[arm]["birth"]["cpu_per_update"] * 24576),
                                             birth_scoring_cpu_s=round(bsc),
                                             life_cpu_s=round(lu), life_updates_cpu_s=round(d7[arm]["life"]["cpu_per_update"] * 36864),
                                             life_scoring_cpu_s=round(d7[arm]["life"]["cpu_scoring_per_checkpoint"] * 7))
    d7["forecast_stage2_six_births_cpu_s"] = 3 * (d7["S"]["forecast_full_dose"]["birth_cpu_s"] + d7["B"]["forecast_full_dose"]["birth_cpu_s"])
    d7["forecast_stage3_six_lives_cpu_s"] = 3 * (d7["S"]["forecast_full_dose"]["life_cpu_s"] + d7["B"]["forecast_full_dose"]["life_cpu_s"])
    d7["pass"] = all(d7[a]["birth"]["complete"] and d7[a]["life"]["complete"] and d7[a]["life"]["stages"] == 7 and d7[a]["birth"]["finite_losses"]
                     and d7[a]["life"]["finite_losses"] for a in ARMS)
    return dict(check3_data=d3, check4_optimizer=d4, check7_reduced_runs=d7, **{"pass": d3["pass"] and d4["pass"] and d7["pass"]})


def run_check(part, args, qual, cmd):
    qual.mkdir(parents=True, exist_ok=True)
    t0 = time.process_time()
    if part == "assemble":
        parts = {}
        for f in sorted(qual.glob("check_*.json")):
            parts[f.stem[len("check_"):]] = json.loads(f.read_text())
        need = ["step", "paths", "onestep", "scorer_73129", "scorer_93683", "scorer_116089", "arch", "binding", "data"]
        summary = {k: (parts[k]["result"].get("pass") if k in parts else "NOT RUN") for k in need}
        out = dict(experiment="E26", stage=1, summary=summary, all_pass=all(v is True for v in summary.values()), parts=parts,
                   assembled_by=dict(command=cmd, provenance=provenance(cmd)))
        (qual / "E26_CHECK.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(dict(summary=summary, all_pass=out["all_pass"]), indent=1)); return
    fn = dict(step=check_step, paths=check_paths, onestep=check_onestep, arch=check_arch, binding=check_binding)
    if part in fn:
        res, name = fn[part](), part
    elif part == "scorer":
        res, name = check_scorer(int(args[0])), f"scorer_{int(args[0])}"
    elif part == "data":
        res, name = check_data(qual), "data"
    else:
        raise SystemExit(f"unknown check part {part}")
    rec = dict(part=name, command=cmd, provenance=provenance(cmd), e25_alignment_sha256=X16.sha_file(HERE / "e25_alignment.py"),
               cpu_seconds=time.process_time() - t0, result=res)
    (qual / f"check_{name}.json").write_text(json.dumps(rec, indent=1))
    print(name, "pass" if res.get("pass") else "FAIL", f"cpu {rec['cpu_seconds']:.0f}s", flush=True)

if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])

    def opt_(flag, default=None):
        return a[a.index(flag) + 1] if flag in a else default

    outdir = opt_("--out", str(X16.OUT / "e26"))
    if a[0] == "dev":
        assert a[1] in ARMS; dev(a[1], int(a[2]), outdir, cmd)
    elif a[0] == "life":
        assert a[1] in ARMS; life(a[1], int(a[2]), outdir, cmd)
    elif a[0] == "check":
        run_check(a[1], [v for v in a[2:] if not v.startswith("--") and v != opt_("--qual")], Path(opt_("--qual", str(X16.OUT / "e26_qual"))), cmd)
    else:
        raise SystemExit(__doc__)
