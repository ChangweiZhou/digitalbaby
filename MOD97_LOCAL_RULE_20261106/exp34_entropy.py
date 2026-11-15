"""E34 runner (brief OPUS_TASK_E34_ENTROPY_WEIGHTED_ERROR_20261115.md, revision 2, commit 95d57fd1): sweep births with each byte's error scaled
by the network's own confidence.

E33's birth loop (exp33_targets.dev, itself E28's; copied here, not called or patched) with the loss below. masks, FINE and the role
diagnostics are imported from exp28_everybyte unchanged; no existing dev, step or loss function is called by a birth (loss_of is called only
by the checks, as the reference the brief names).

    python3 exp34_entropy.py dev <arm> <seed> [--out DIR]     arm: U (tau inf) T050 (0.5) T100 (1.0) T200 (2.0)
                                                              addendum 1: BU (tau inf, local False: dry check against B-all), B050 (0.5, local False)
    python3 exp34_entropy.py checks <seed>                    checks 1 (loss, first-batch gradients), 2 and 3 -> exp45/e34/E34_CHECKS.json

The loss (the brief's code):

    x, y, amask, real, fine = exp28_everybyte.masks(items)       # amask and fine: diagnostics only
    logits = arch26.forward(m, x, True, True)[0]
    ce = cross_entropy(logits[real], y[real], reduction="none")
    with T.no_grad():
        lp = log_softmax(logits[real], -1); Hh = -(lp.exp() * lp).sum(-1); w = T.exp(-Hh / tau)
        assert T.isfinite(w).all() and w.sum() > 0
    loss = (w * ce).sum() / w.sum()

Weight diagnostics per 512-update block and per fine role, from the same detached Hh, w and ce: sum of Hh, sum of w, sum of w * ce, count.
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
from torch.nn.functional import cross_entropy, log_softmax  # noqa: E402

C, R, X16, E, X, arch26 = E28.C, E28.R, E28.X16, E28.E, E28.X, E28.arch26
masks, FINE = E28.masks, E28.FINE
ARMS = {"U": float("inf"), "T050": 0.5, "T100": 1.0, "T200": 2.0, "BU": float("inf"), "B050": 0.5}
BACKPROP_ARMS = ("BU", "B050")       # addendum 1: the only difference is local = False in the forward (full backpropagation through the item)
SOURCES = E28.SOURCES + ("exp34_entropy.py",)
E28_DIR = X.OUT / "e28"
OUT_DEFAULT = X16.OUT / "e34"


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
def gain(logits_real, tau):
    with T.no_grad():
        lp = log_softmax(logits_real, -1)
        Hh = -(lp.exp() * lp).sum(-1)
        w = T.exp(-Hh / tau)
        assert T.isfinite(w).all() and w.sum() > 0
    return Hh, w


def loss34(logits, y, real, tau):
    ce = cross_entropy(logits[real], y[real], reduction="none")
    Hh, w = gain(logits[real], tau)
    loss = (w * ce).sum() / w.sum()
    return loss, ce, Hh, w


def new_wdiag():
    return {n: [0.0, 0.0, 0.0, 0] for n in FINE}       # sum Hh, sum w, sum w*ce, count


def step(m, opt, items, tau, roles_acc, wdiag, local=True):
    x, y, amask, real, fine = masks(items)
    logits = arch26.forward(m, x, local, True)[0]
    loss, ce, Hh, w = loss34(logits, y, real, tau)
    with T.no_grad():                    # diagnostics on the same logits, detached: cannot reach the update
        ce_d = cross_entropy(logits.detach()[real], y[real], reduction="none").double()
        fr = fine[real]
        Hd, wd, wce = Hh.double(), w.double(), (w.double() * ce.detach().double())
        for i, n in enumerate(FINE):
            s_ = fr == i
            roles_acc[n][0] += float(ce_d[s_].sum()); roles_acc[n][1] += int(s_.sum())
            a = wdiag[n]; a[0] += float(Hd[s_].sum()); a[1] += float(wd[s_].sum()); a[2] += float(wce[s_].sum()); a[3] += int(s_.sum())
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach())


ANSWER = ("answer_digit", "eos")


def summarize_wdiag(acc):
    tot_wce = sum(v[2] for v in acc.values())
    by = {n: dict(mean_entropy=(h / c if c else None), mean_weight=(w / c if c else None), share_weighted_error=(wce / tot_wce if tot_wce else None),
                  sum_entropy=h, sum_weight=w, sum_weighted_ce=wce, positions=c) for n, (h, w, wce, c) in acc.items()}
    aw = sum(acc[n][1] for n in ANSWER); ac = sum(acc[n][3] for n in ANSWER)
    dw, dc = acc["distractor"][1], acc["distractor"][3]
    ratio = (dw / dc) / (aw / ac) if dc and ac and aw else None
    return dict(by_fine_role=by, distractor_to_answer_weight_ratio=ratio,
                ratio_definition="mean weight of distractor positions / mean weight of answer positions (answer digits and EOS)")


# ---------------------------------------------------------------- files
def birth_path(out, arm, seed):
    return Path(out) / f"e34_dev_{arm}{E.DEVTAG}_{seed}.pt"


def row_path(out, arm, seed, stop):
    return Path(out) / f"e34_dev_{arm}{E.DEVTAG}_{seed}_row{stop}.pt"


def e28_reference(seed, ref_arm="S-all"):
    if E.DEVTAG:
        p = X.OUT / "e28_qual" / "dry" / f"e28_dev_{ref_arm}_dry{E.DEV_DOSE}_{seed}.json"
    else:
        p = E28_DIR / f"e28_dev_{ref_arm}_{seed}.json"
    d = json.loads(p.read_text())
    return {r["step"]: dict(data_digest=r["data_digest"], tensors_sha256=r["tensors_sha256"]) for r in d["curve"]}, str(p.relative_to(HERE))


# ---------------------------------------------------------------- birth
def dev(arm, seed, out, cmd):
    tau = ARMS[arm]
    local = arm not in BACKPROP_ARMS
    arch26.install()
    path = birth_path(out, arm, seed); rec_path = path.with_suffix(".json")
    R.refuse_existing(path, rec_path, str(path) + ".tmp", *[row_path(out, arm, seed, s) for s in range(4096, E.DEV_DOSE, 4096)])
    Path(out).mkdir(parents=True, exist_ok=True)
    ref, ref_file = e28_reference(seed, "S-all" if local else "B-all")
    m = E.new_net(seed)
    desc = arch26.describe(m)
    book0 = m.book.weight.detach().clone()
    opt = C.make_opt(m)
    census = None
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    pan = {E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV}
    bt = E.batches(tp, seed, E.KMAX["M"])
    dg = C.Digest()
    rec = dict(experiment="E34", kind="birth", arm=arm, tau=("inf" if math.isinf(tau) else tau), local=local, net="G1", seed=seed, kmax=E.KMAX["M"],
               dev_updates=E.DEV_DOSE, block=X.BLOCK, batch=X.BATCH, optimizer=C.opt_record(m, opt),
               update=f"exp34_entropy.step: forward(local={local}), entropy-weighted CE over every real byte, zero_grad, backward, AdamW step",
               loss_definition=("sum over real positions of w * CE / sum of w, w = exp(-H / tau) with H the predictive entropy in nats, computed under "
                                "no_grad (no gradient through w); no answer mask or role label enters the loss"),
               roles=dict(fine=list(FINE), fine_to_coarse=E28.FINE_TO_ROLE, identification="exp28_everybyte.masks / roles_of, imported unchanged; diagnostics only"),
               protocol="exp33_targets.dev's loop (E28's S arm), loss changed only",
               reference=dict(file=ref_file, checked="data_digest at every saved row (asserted); tensors_sha256 compared and recorded"),
               model=desc, installed=dict(E_forward=f"{E.forward.__module__}.{E.forward.__name__}", E_new_net_class=desc["cls"]),
               provenance=provenance(cmd), complete=False, curve=[], role_curve=[], finite=dict(all_finite=True, updates_covered=0, first_nonfinite_update=None))
    t0 = time.process_time(); cpu_score = 0.0; cpu_save = 0.0; curve = rec["curve"]; row = None; nupd = 0
    for stop in range(X.BLOCK, E.DEV_DOSE + 1, X.BLOCK):
        losses = []; acc = E28.new_roles(); wacc = new_wdiag()
        for _ in range(X.BLOCK):
            items = next(bt); dg.add(items)
            v = step(m, opt, items, tau, acc, wacc, local); losses.append(v); nupd += 1
            if not math.isfinite(v) and rec["finite"]["all_finite"]:
                rec["finite"].update(all_finite=False, first_nonfinite_update=nupd)
            rec["finite"]["updates_covered"] = nupd
            if census is None:
                census = X16.zero_grad_report(m, X.batch_tensor(items)[0]); print("e34 dev credit census:", census["by_role"], flush=True)
        rec["role_curve"].append(dict(block_end=stop, train_loss=sum(losses) / len(losses), finite_losses=all(math.isfinite(v) for v in losses),
                                      **E28.summarize_roles(acc), weights=summarize_wdiag(wacc)))
        if stop % 4096 == 0 or stop == E.DEV_DOSE:
            cb = time.process_time()
            sealed = {n: {k: E.score(m, it) for k, it in pk.items()} for n, pk in pan.items()}
            cpu_score += time.process_time() - cb
            d_now = dg.value()
            assert stop in ref, f"the reference {ref_file} has no row at {stop}"
            assert d_now == ref[stop]["data_digest"], f"data digest at row {stop} is not E28 S-all's: {d_now} vs {ref[stop]['data_digest']}"
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
            print("e34 dev", arm, seed, stop, {n: [v[k]["exact"] for k in (0, 4, 8, 12, 20)] for n, v in row["sealed"].items()},
                  f"hard {row['hard_mean_add89_sub89_k12']:.1f} cpu {row['cpu']:.0f}s tensors {'=' if same else '!='} {ref_file.split("/")[-1]}", flush=True)
            X16.write_json(rec_path, dict(rec, credit_census_first_update=census))
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
    print("e34 dev", arm, seed, "complete", wsha[:16], flush=True)


# ---------------------------------------------------------------- checks 1, 2, 3
def first_batches(seed, n):
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    bt = E.batches(tp, seed, E.KMAX["M"])
    return [next(bt) for _ in range(n)]


def rel_close(a, b, tol):
    a = a.detach().double(); b = b.detach().double()
    scale = float(b.abs().max())
    d = float((a - b).abs().max())
    return d, scale, (d <= tol * scale) if scale > 0 else d == 0


def param_grads(m):
    return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def checks(seed, cmd):
    arch26.install()
    out = dict(experiment="E34", kind="checks", seed=seed, provenance=provenance(cmd),
               tolerance="agrees: largest |difference| <= 1e-5 * largest |entry| of the reference tensor")
    batches = first_batches(seed, 3)
    # check 1a: U's loss against exp28_everybyte.loss_of(..., "all") on the first three batches, U updates between them
    m = E.new_net(seed); opt = C.make_opt(m); rows = []
    for i, items in enumerate(batches):
        x, y, amask, real, fine = masks(items)
        logits = arch26.forward(m, x, True, True)[0]
        mine = loss34(logits, y, real, ARMS["U"])[0]
        refl = E28.loss_of(logits, y, amask, real, "all")
        a_, b_ = float(mine.detach()), float(refl.detach())
        d = abs(a_ - b_)
        rows.append(dict(batch=i + 1, e34_U=a_, e28_all=b_, abs_diff=d, rel_diff=d / abs(b_), bit_equal=bool(T.equal(mine.detach(), refl.detach())),
                         within_1e6_rel=d <= 1e-6 * abs(b_)))
        opt.zero_grad(); mine.backward(); opt.step()
    out["check1_loss_three_batches"] = dict(rows=rows, passed=all(r["within_1e6_rel"] for r in rows))
    # check 1b: first-batch gradients (logits and parameters), U against S-all's loss, each from a fresh net
    x, y, amask, real, fine = masks(batches[0])
    G = {}
    for name in ("U", "all"):
        m = E.new_net(seed)
        logits = arch26.forward(m, x, True, True)[0]; logits.retain_grad()
        (loss34(logits, y, real, ARMS["U"])[0] if name == "U" else E28.loss_of(logits, y, amask, real, "all")).backward()
        G[name] = dict(logits=logits.grad.clone(), **param_grads(m))
    g1 = []
    for k in G["all"]:
        d, s, ok = rel_close(G["U"][k], G["all"][k], 1e-5)
        g1.append(dict(tensor=k, max_abs_diff=d, max_abs_entry=s, bit_equal=bool(T.equal(G["U"][k], G["all"][k])), agrees=ok))
    out["check1_first_batch_gradients"] = dict(rows=g1, passed=all(r["agrees"] for r in g1), all_bit_equal=all(r["bit_equal"] for r in g1))
    # check 2: logit gradient equals (softmax - onehot) * w / w.sum() at real positions, with w recomputed independently; exactly zero elsewhere
    c2 = {}
    for arm in ("T050", "T100", "T200"):
        tau = ARMS[arm]
        m = E.new_net(seed)
        logits = arch26.forward(m, x, True, True)[0]; logits.retain_grad()
        loss34(logits, y, real, tau)[0].backward()
        g = logits.grad
        L = logits.detach().double()[real]
        p = T.softmax(L, -1)
        H = -(p * T.log(p.clamp_min(1e-300))).sum(-1)
        w = T.exp(-H / tau)
        oh = T.zeros_like(p); oh[T.arange(len(oh)), y[real]] = 1.0
        want = (p - oh) * (w / w.sum())[:, None]
        d, s, ok = rel_close(g[real], want, 1e-5)
        zero = bool((g[~real] == 0).all())
        c2[arm] = dict(tau=tau, max_abs_diff=d, max_abs_entry=s, agrees_at_real=ok, zero_at_non_real=zero, non_real_positions=int((~real).sum()),
                       real_positions=int(real.sum()), passed=ok and zero)
    out["check2_gain_is_what_the_code_says"] = dict(arms=c2, passed=all(v["passed"] for v in c2.values()),
                                                    independent_w="float64 softmax of the detached logits, H = -sum p log p, w = exp(-H / tau)")
    # check 3: no gradient through the gain, tau 1: the same loss with w replaced by a constant tensor of the same values
    G3 = {}
    for name in ("rule", "constant"):
        m = E.new_net(seed)
        logits = arch26.forward(m, x, True, True)[0]
        if name == "rule":
            loss = loss34(logits, y, real, 1.0)[0]
        else:
            ce = cross_entropy(logits[real], y[real], reduction="none")
            wc = T.tensor(gain(logits.detach()[real], 1.0)[1].tolist(), dtype=ce.dtype)      # a fresh constant tensor, no graph
            loss = (wc * ce).sum() / wc.sum()
        loss.backward()
        G3[name] = param_grads(m)
    g3 = []
    for k in G3["constant"]:
        d, s, ok = rel_close(G3["rule"][k], G3["constant"][k], 1e-5)
        g3.append(dict(tensor=k, max_abs_diff=d, max_abs_entry=s, bit_equal=bool(T.equal(G3["rule"][k], G3["constant"][k])), agrees=ok))
    out["check3_no_gradient_through_gain"] = dict(rows=g3, passed=all(r["agrees"] for r in g3), all_bit_equal=all(r["bit_equal"] for r in g3))
    p_ = OUT_DEFAULT / "E34_CHECKS.json"
    p_.parent.mkdir(parents=True, exist_ok=True)
    X16.write_json(p_, out)
    print(json.dumps({k: v.get("passed") for k, v in out.items() if isinstance(v, dict) and "passed" in v}),
          "1b bit-equal:", out["check1_first_batch_gradients"]["all_bit_equal"], "3 bit-equal:", out["check3_no_gradient_through_gain"]["all_bit_equal"], flush=True)
    return out


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
