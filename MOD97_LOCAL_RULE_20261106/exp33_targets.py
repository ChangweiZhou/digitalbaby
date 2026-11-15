"""E33 runner (brief OPUS_TASK_E33_WHICH_TARGETS_BREAK_EXTRAPOLATION_20261115.md, revision 2, commit 214169f8): which non-answer targets stop the
sweep from extrapolating.

E28's birth loop (exp28_everybyte.dev, copied here, not called or patched) with one change, the loss. masks, FINE, roles_of and the role
diagnostics are imported from exp28_everybyte unchanged; exp28_everybyte.dev, .step and .loss_of are not called or patched (loss_of is called
only by the checks, as the reference the brief names).

    python3 exp33_targets.py dev <arm> <seed> [--out DIR]       arm: A A+D A+O A+S A+DO A+DS A+OS A+DOS
    python3 exp33_targets.py checks <seed>                      checks 1 (loss on three batches; first-batch gradients) and 2 -> exp45/e33/E33_CHECKS.json

The loss (the brief's code):

    x, y, amask, real, fine = exp28_everybyte.masks(items)
    logits = arch26.forward(m, x, True, True)[0]
    other  = real & ~amask
    cls    = {D: fine == distractor, O: fine == operand, S: code | operator | equals}
    sel    = OR of cls[c] for c in arm_classes
    loss   = 0.5 * CE(logits[amask], y[amask])  [+ 0.5 * CE(logits[sel], y[sel], sum) / other.sum()  when arm_classes is not empty]

Each selected position keeps exactly its S-bal weight 0.5 / other.sum(); unselected positions are dropped; nothing is rescaled.
Dose: E11_DEV_UPDATES (full dose when unset). Every output path must not exist.
"""
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp28_everybyte as E28        # noqa: E402  (imports exp26_candidate -> exp20_reference: environment gate; not edited)

import torch as T                    # noqa: E402
from torch.nn.functional import cross_entropy  # noqa: E402

C, R, X16, E, X, arch26 = E28.C, E28.R, E28.X16, E28.E, E28.X, E28.arch26
masks, FINE, roles_of = E28.masks, E28.FINE, E28.roles_of
ARMS = {"A": "", "A+D": "D", "A+O": "O", "A+S": "S", "A+DO": "DO", "A+DS": "DS", "A+OS": "OS", "A+DOS": "DOS"}
SOURCES = E28.SOURCES + ("exp33_targets.py",)
E28_DIR = X.OUT / "e28"
OUT_DEFAULT = X16.OUT / "e33"
F = FINE.index


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
                interpreter=E28.environment(), sources_sha256=source_hashes())


# ---------------------------------------------------------------- the loss
def classes(fine):
    return {"D": fine == F("distractor"),
            "O": fine == F("operand"),
            "S": (fine == F("code")) | (fine == F("operator")) | (fine == F("equals"))}


def selection(fine, real, amask, arm_classes):
    other = real & ~amask
    cls = classes(fine)
    sel = T.zeros_like(other)
    for c in arm_classes:
        sel |= cls[c]
    assert not (sel & ~other).any() and T.equal(cls["D"] | cls["O"] | cls["S"], other)
    return other, sel


def loss33(logits, y, amask, real, fine, arm_classes):
    other, sel = selection(fine, real, amask, arm_classes)
    loss = 0.5 * cross_entropy(logits[amask], y[amask])
    if arm_classes:
        loss = loss + 0.5 * cross_entropy(logits[sel], y[sel], reduction="sum") / other.sum()
    return loss


def step(m, opt, items, arm_classes, roles_acc=None):
    x, y, amask, real, fine = masks(items)
    logits = arch26.forward(m, x, True, True)[0]
    loss = loss33(logits, y, amask, real, fine, arm_classes)
    if roles_acc is not None:            # diagnostics on the same logits, detached, for every role selected or not (as E28)
        with T.no_grad():
            ce = cross_entropy(logits.detach()[real], y[real], reduction="none").double()
            fr = fine[real]
            for i, n in enumerate(FINE):
                s_ = fr == i
                roles_acc[n][0] += float(ce[s_].sum()); roles_acc[n][1] += int(s_.sum())
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach())


# ---------------------------------------------------------------- files
def safe(arm):
    return arm


def birth_path(out, arm, seed):
    return Path(out) / f"e33_dev_{safe(arm)}{E.DEVTAG}_{seed}.pt"


def row_path(out, arm, seed, stop):
    return Path(out) / f"e33_dev_{safe(arm)}{E.DEVTAG}_{seed}_row{stop}.pt"


def e28_reference(seed):
    """S-bal's receipt of this seed (full dose) and, for a dry dose, E28's dry receipt at the same dose if there is one."""
    full = json.loads((E28_DIR / f"e28_dev_S-bal_{seed}.json").read_text())
    rows = {r["step"]: dict(data_digest=r["data_digest"], tensors_sha256=r["tensors_sha256"]) for r in full["curve"]}
    dry = X.OUT / "e28_qual" / "dry" / f"e28_dev_S-bal_dry{E.DEV_DOSE}_{seed}.json"
    if E.DEVTAG and dry.exists():
        d = json.loads(dry.read_text())
        rows = {r["step"]: dict(data_digest=r["data_digest"], tensors_sha256=r["tensors_sha256"]) for r in d["curve"]}
        return rows, str(dry.relative_to(HERE))
    return rows, str((E28_DIR / f"e28_dev_S-bal_{seed}.json").relative_to(HERE))


# ---------------------------------------------------------------- birth
def dev(arm, seed, out, cmd):
    arm_classes = ARMS[arm]
    local = True
    arch26.install()
    path = birth_path(out, arm, seed); rec_path = path.with_suffix(".json")
    R.refuse_existing(path, rec_path, str(path) + ".tmp", *[row_path(out, arm, seed, s) for s in range(4096, E.DEV_DOSE, 4096)])
    Path(out).mkdir(parents=True, exist_ok=True)
    ref, ref_file = e28_reference(seed)
    m = E.new_net(seed)
    desc = arch26.describe(m)
    book0 = m.book.weight.detach().clone()
    opt = C.make_opt(m)
    census = None
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    pan = {E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV}
    bt = E.batches(tp, seed, E.KMAX["M"])
    dg = C.Digest()
    rec = dict(experiment="E33", kind="birth", arm=arm, arm_classes=arm_classes, local=local, net="G1", seed=seed, kmax=E.KMAX["M"],
               dev_updates=E.DEV_DOSE, block=X.BLOCK, batch=X.BATCH, optimizer=C.opt_record(m, opt),
               update="exp33_targets.step: forward(local=True), E33 loss for arm_classes, zero_grad, backward, AdamW step (credit through the whole item)",
               loss_definition=("0.5 * mean CE over answer positions + 0.5 * (sum of CE over the selected non-answer positions) / (count of all "
                                "non-answer real positions); selected = union of classes in arm_classes (D distractor, O operand, S code|operator|equals); "
                                "unselected positions dropped, nothing rescaled"),
               roles=dict(fine=list(FINE), fine_to_coarse=E28.FINE_TO_ROLE, classes=dict(D=["distractor"], O=["operand"], S=["code", "operator", "equals"]),
                          identification="exp28_everybyte.masks / roles_of, imported unchanged"),
               protocol="exp28_everybyte.dev's loop (E28 S arm), loss changed only",
               reference=dict(file=ref_file, checked="data_digest at every saved row (asserted); tensors_sha256 compared and recorded"
                              + (" and asserted equal at every row for A+DOS (brief check 1, applied when the dry hash matched)" if arm == "A+DOS" else "")),
               model=desc, installed=dict(E_forward=f"{E.forward.__module__}.{E.forward.__name__}", E_new_net_class=desc["cls"]),
               provenance=provenance(cmd), complete=False, curve=[], role_curve=[], finite=dict(all_finite=True, updates_covered=0, first_nonfinite_update=None))
    t0 = time.process_time(); cpu_score = 0.0; cpu_save = 0.0; curve = rec["curve"]; row = None; nupd = 0
    for stop in range(X.BLOCK, E.DEV_DOSE + 1, X.BLOCK):
        losses = []; acc = E28.new_roles()
        for _ in range(X.BLOCK):
            items = next(bt); dg.add(items)
            v = step(m, opt, items, arm_classes, acc); losses.append(v); nupd += 1
            if not math.isfinite(v) and rec["finite"]["all_finite"]:
                rec["finite"].update(all_finite=False, first_nonfinite_update=nupd)
            rec["finite"]["updates_covered"] = nupd
            if census is None:
                census = X16.zero_grad_report(m, X.batch_tensor(items)[0]); print("e33 dev credit census:", census["by_role"], flush=True)
        rec["role_curve"].append(dict(block_end=stop, train_loss=sum(losses) / len(losses), **E28.summarize_roles(acc)))
        if stop % 4096 == 0 or stop == E.DEV_DOSE:
            cb = time.process_time()
            sealed = {n: {k: E.score(m, it) for k, it in pk.items()} for n, pk in pan.items()}
            cpu_score += time.process_time() - cb
            d_now = dg.value()
            assert stop in ref, f"the reference {ref_file} has no row at {stop}"
            assert d_now == ref[stop]["data_digest"], f"data digest at row {stop} is not E28's: {d_now} vs {ref[stop]['data_digest']}"
            cs = time.process_time()
            if stop == E.DEV_DOSE:
                wf, wsha = path, None
            else:
                wf = row_path(out, arm, seed, stop); wsha = R.save_weights(m, wf)
            cpu_save += time.process_time() - cs
            tsha = R.tensors_sha(m.state_dict())
            same = tsha == ref[stop]["tensors_sha256"]
            hard = {n: v[12]["exact"] for n, v in sealed.items()}
            row = dict(step=stop, loss=sum(losses) / len(losses), finite_losses=all(math.isfinite(v) for v in losses),
                       finite_cumulative=dict(rec["finite"]), cpu=time.process_time() - t0, cpu_scoring_so_far=cpu_score,
                       cpu_updates_so_far=time.process_time() - t0 - cpu_score - cpu_save, sealed=sealed,
                       hard_mean_add89_sub89_k12=(hard["add89"] + hard["sub89"]) / 2 / 300 * 100,
                       book_shift_by_role=X16.book_shift(m, book0), data_digest=d_now,
                       data_digest_check=f"equal to {ref_file} at this row: asserted",
                       weights=wf.name, weights_sha256=wsha, tensors_sha256=tsha,
                       tensors_sha256_reference=ref[stop]["tensors_sha256"], tensors_equal_to_reference=same)
            curve.append(row)
            print("e33 dev", arm, seed, stop, {n: [v[k]["exact"] for k in (0, 4, 8, 12, 20)] for n, v in row["sealed"].items()},
                  f"hard {row['hard_mean_add89_sub89_k12']:.1f} cpu {row['cpu']:.0f}s tensors {'=' if same else '!='} S-bal", flush=True)
            X16.write_json(rec_path, dict(rec, credit_census_first_update=census))
            if arm == "A+DOS" and not same:
                rec["stopped"] = f"tensors_sha256 at row {stop} differs from S-bal's ({tsha} vs {ref[stop]['tensors_sha256']}): stopped (brief check 1)"
                X16.write_json(rec_path, dict(rec, credit_census_first_update=census))
                raise SystemExit(rec["stopped"])
    assert row is not None, "E11_DEV_UPDATES must be a positive multiple of 512"
    cb = time.process_time()
    row["silenced_k12"] = {n: E.score(m, pk[12], silence=True) for n, pk in pan.items()}
    cpu_score += time.process_time() - cb
    wsha = R.save_weights(m, path)
    row["weights_sha256"] = wsha
    fin = dg.value()
    rec.update(credit_census_first_update=census, optimizer_steps=R.opt_steps(opt), optimizer_read_back=C.opt_record(m, opt), data_digest=fin,
               weights=path.name, weights_sha256=wsha, tensors_sha256=R.tensors_sha(m.state_dict()),
               row_weights={str(r["step"]): dict(file=r["weights"], sha256=r["weights_sha256"]) for r in curve},
               cpu_total=time.process_time() - t0, cpu_scoring=cpu_score, cpu_saving_rows=cpu_save,
               cpu_updates=time.process_time() - t0 - cpu_score - cpu_save, complete=True)
    rec["cpu_per_update"] = rec["cpu_updates"] / E.DEV_DOSE
    X16.write_json(rec_path, rec)
    print("e33 dev", arm, seed, "complete", wsha[:16], flush=True)


# ---------------------------------------------------------------- checks 1 (loss and first-batch gradients) and 2
def first_batches(seed, n):
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    bt = E.batches(tp, seed, E.KMAX["M"])
    return [next(bt) for _ in range(n)]


def rel_close(a, b, tol):
    a = a.detach().double(); b = b.detach().double()
    scale = float(b.abs().max())
    d = float((a - b).abs().max())
    return d, scale, (d <= tol * scale) if scale > 0 else d == 0


def checks(seed, cmd):
    arch26.install()
    out = dict(experiment="E33", kind="checks", seed=seed, provenance=provenance(cmd), tolerance="largest |difference| <= tol * largest |entry| of the reference")
    batches = first_batches(seed, 3)
    # check 1a: A+DOS's loss against exp28_everybyte.loss_of(..., "balanced") on the first three batches, A+DOS updates between them
    m = E.new_net(seed); opt = C.make_opt(m); rows = []
    for i, items in enumerate(batches):
        x, y, amask, real, fine = masks(items)
        logits = arch26.forward(m, x, True, True)[0]
        mine = loss33(logits, y, amask, real, fine, "DOS")
        refl = E28.loss_of(logits, y, amask, real, "balanced")
        d = abs(float(mine) - float(refl))
        rows.append(dict(batch=i + 1, e33=float(mine), e28_balanced=float(refl), abs_diff=d, rel_diff=d / abs(float(refl)),
                         bit_equal=bool(T.equal(mine.detach(), refl.detach())), within_1e6_rel=d <= 1e-6 * abs(float(refl))))
        opt.zero_grad(); mine.backward(); opt.step()
    out["check1_loss_three_batches"] = dict(rows=rows, passed=all(r["within_1e6_rel"] for r in rows))
    # check 1b: first-batch gradients of A+DOS against S-bal's loss (parameters and logits), each from a fresh net of this seed
    grads = {}
    for name, fn in (("e33_A+DOS", lambda lg, y, a, r, f: loss33(lg, y, a, r, f, "DOS")),
                     ("e28_balanced", lambda lg, y, a, r, f: E28.loss_of(lg, y, a, r, "balanced"))):
        m = E.new_net(seed)
        x, y, amask, real, fine = masks(batches[0])
        logits = arch26.forward(m, x, True, True)[0]; logits.retain_grad()
        fn(logits, y, amask, real, fine).backward()
        grads[name] = dict(logits=logits.grad.clone(), **{n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
    g1 = []
    for k in grads["e28_balanced"]:
        d, s, ok = rel_close(grads["e33_A+DOS"][k], grads["e28_balanced"][k], 1e-5)
        g1.append(dict(tensor=k, max_abs_diff=d, max_abs_entry=s, bit_equal=bool(T.equal(grads["e33_A+DOS"][k], grads["e28_balanced"][k])), agrees=ok))
    out["check1_first_batch_gradients"] = dict(rows=g1, passed=all(r["agrees"] for r in g1), all_bit_equal=all(r["bit_equal"] for r in g1))
    # check 2: for each arm, on the first batch, the logit gradient is exactly zero outside amask | sel and non-zero somewhere in each selected class
    c2 = {}
    x, y, amask, real, fine = masks(batches[0])
    for arm, ac in ARMS.items():
        m = E.new_net(seed)
        logits = arch26.forward(m, x, True, True)[0]; logits.retain_grad()
        loss33(logits, y, amask, real, fine, ac).backward()
        g = logits.grad
        other, sel = selection(fine, real, amask, ac)
        carry = (g != 0).any(-1)                       # (N, L): any non-zero entry at the position
        outside = ~(amask | sel)
        cls = classes(fine)
        by_fine = {n: int((sel & (fine == F(n))).sum()) for n in FINE}
        nz_class = {c: bool((carry & cls[c]).any()) for c in ac}
        c2[arm] = dict(arm_classes=ac, zero_outside_amask_or_sel=bool((g[outside] == 0).all()),
                       positions_outside_with_nonzero_gradient=int((carry & outside).sum()),
                       nonzero_somewhere_in_each_selected_class=nz_class, answer_positions_with_gradient=int((carry & amask).sum()),
                       answer_positions=int(amask.sum()), selected_positions_by_fine_role=by_fine, selected_total=int(sel.sum()),
                       non_answer_total=int(other.sum()),
                       passed=bool((g[outside] == 0).all()) and all(nz_class.values()))
    out["check2_selected_positions_only"] = dict(arms=c2, passed=all(v["passed"] for v in c2.values()))
    p = OUT_DEFAULT / "E33_CHECKS.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    X16.write_json(p, out)
    print(json.dumps({k: v.get("passed") for k, v in out.items() if isinstance(v, dict) and "passed" in v}),
          "grad bit-equal:", out["check1_first_batch_gradients"]["all_bit_equal"], flush=True)


if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])
    outdir = a[a.index("--out") + 1] if "--out" in a else str(OUT_DEFAULT)
    if a and a[0] == "dev":
        assert a[1] in ARMS, f"arm must be one of {list(ARMS)}"
        dev(a[1], int(a[2]), outdir, cmd)
    elif a and a[0] == "checks":
        checks(int(a[1]), cmd)
    else:
        raise SystemExit(__doc__)
