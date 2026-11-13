"""E28 runner (brief OPUS_TASK_E28_GRU_EVERY_BYTE_20261113.md, revision 2, commit 50bd275a): E26's GRU born under every-byte targets.

The condition differs from E26's birth in the loss mask alone. The loop is new (it records what E26's could not); the model, the data, the
optimizer groups and the scorer are exp26_candidate's and arch26's, imported unedited. exp26_candidate.dev and exp26_candidate.step are
NOT called or patched here: they write experiment "E26", an "answer-masked" update string and e26_dev_ names.

    python3 exp28_everybyte.py dev <arm> <seed> [--out DIR]       arm: S-ans B-ans (host check) | S-all B-all S-bal B-bal

The update, one loop for every arm (local = True for S, False for B):

    x, y, amask = exp45.batch_tensor(items)
    logits      = arch26.forward(m, x, local, True)[0]                       # credit through the whole item, as E26
    real        = y != BOS                                                   # padding is BOS in y; no real target is BOS (asserted)
    answer   : loss = cross_entropy(logits[amask], y[amask])                 # E26's loss, for the host check only
    all      : loss = cross_entropy(logits[real],  y[real])
    balanced : loss = 0.5 * cross_entropy(logits[amask], y[amask]) + 0.5 * cross_entropy(logits[real & ~amask], y[real & ~amask])
    opt.zero_grad(); loss.backward(); opt.step()                             # AdamW 0.001, (0.9, 0.98), decay 0.25, embedding without decay

Target roles (loss by role; identified from the item's own layout, exp11_cell.render, and checked against the bytes):
    an item is  prompt = code a1 a2 op d_1..d_k b1 b2 '='   and   target = c1 c2 EOS,  k = len(prompt) - 7;
    y at position j of a row is byte j of prompt+target (the input there is the byte before it, or BOS at j = 0). Roles:
      distractor   d_1..d_k                                   (asserted upper-case letters)
      operand      a1 a2 b1 b2                                (asserted digits)
      structural   code, op, '='                              (asserted: a DEV task code, the task's operator, '=')
      answer       c1 c2 EOS = exactly the answer mask exp45.batch_tensor returns (asserted)
    The brief lists "end" among the structural bytes; the end byte (EOS) is inside the answer mask E26 trained on and the balanced loss uses,
    so it is counted under answer here, and the task code (not named in the brief) under structural. Per-role sums and counts are also kept
    for each finer role (code, operator, equals, EOS, answer digits), so any other grouping can be formed from the receipt.

Dose: E11_DEV_UPDATES (full dose when unset), as E26. Every output path must not exist.
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
import exp26_candidate as C          # noqa: E402  (imports exp20_reference: environment gate; not edited)

import torch as T                    # noqa: E402
from torch.nn.functional import cross_entropy  # noqa: E402

R, X16, E, X, arch20 = C.R, C.X16, C.E, C.X, C.arch20
import arch26                        # noqa: E402

LR, BETAS, WD = C.LR, C.BETAS, C.WD
assert WD == 0.25
ARMS = {"S-ans": (True, "answer"), "B-ans": (False, "answer"), "S-all": (True, "all"), "B-all": (False, "all"),
        "S-bal": (True, "balanced"), "B-bal": (False, "balanced")}
TARGETS = ("answer", "all", "balanced")
SOURCES = C.SOURCES + ("exp28_everybyte.py",)
E26_DIR = X.OUT / "e26"
ROLES = ("distractor", "operand", "structural", "answer")
FINE = ("distractor", "operand", "code", "operator", "equals", "answer_digit", "eos")
FINE_TO_ROLE = dict(distractor="distractor", operand="operand", code="structural", operator="structural", equals="structural",
                    answer_digit="answer", eos="answer")
DEV_CODES = {t[2][0] for t in E.DEV}
SYM_OF_CODE = {t[2][0]: ord(E.SYM[t[0]]) for t in E.DEV}


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


def environment():
    import importlib.metadata as md
    pk = {}
    for d in ("torch", "numpy"):
        try:
            pk[d] = md.version(d)
        except Exception:       # noqa: BLE001
            pk[d] = None
    return dict(python_executable=sys.executable, python_version_full=sys.version, packages=pk)


def provenance(cmd):
    return dict(command=cmd, environment_at_start=R.ENV_AT_START, environment_set_by_runner=R.REQUIRED_ENV, host=R.host(), git=git_state(),
                interpreter=environment(), sources_sha256=source_hashes())


# ---------------------------------------------------------------- masks and roles
def roles_of(items, y):
    """A tensor like y: -1 at padding, else the index in FINE of each target's role. Asserts the layout of every item."""
    fine = T.full(y.shape, -1, dtype=T.long)
    for i, (p, t) in enumerate(items):
        k = len(p) - 7
        assert k >= 0 and len(t) == 3 and t[-1] == E.EOS, f"item {i}: not the E26 item layout"
        lab = (["code", "operand", "operand", "operator"] + ["distractor"] * k + ["operand", "operand", "equals"] + ["answer_digit", "answer_digit", "eos"])
        r = p + t
        assert len(lab) == len(r)
        assert r[0] in DEV_CODES and r[3] == SYM_OF_CODE[r[0]] and r[6 + k] == ord("="), f"item {i}: code, operator or '=' not where expected"
        assert all(48 <= r[j] <= 57 for j in (1, 2, 4 + k, 5 + k, 7 + k, 8 + k)), f"item {i}: digits not where expected"
        assert all(65 <= r[j] <= 90 for j in range(4, 4 + k)), f"item {i}: distractors not upper-case letters"
        fine[i, :len(r)] = T.tensor([FINE.index(v) for v in lab])
    return fine


def masks(items):
    x, y, amask = X.batch_tensor(items)
    real = y != E.BOS
    fine = roles_of(items, y)
    assert T.equal(real, fine >= 0), "the real positions (y != BOS) are not exactly the bytes of prompt and answer"
    ans = (fine == FINE.index("answer_digit")) | (fine == FINE.index("eos"))
    assert T.equal(ans, amask), "the answer role is not the answer mask"
    return x, y, amask, real, fine


def loss_of(logits, y, amask, real, targets):
    if targets == "answer":
        return cross_entropy(logits[amask], y[amask])
    if targets == "all":
        return cross_entropy(logits[real], y[real])
    if targets == "balanced":
        other = real & ~amask
        return 0.5 * cross_entropy(logits[amask], y[amask]) + 0.5 * cross_entropy(logits[other], y[other])
    raise ValueError(targets)


# ---------------------------------------------------------------- the update (one loop, every arm)
def step(m, opt, items, local, targets, roles_acc=None):
    x, y, amask, real, fine = masks(items)
    logits = arch26.forward(m, x, local, True)[0]
    loss = loss_of(logits, y, amask, real, targets)
    if roles_acc is not None:            # diagnostics on the same logits, detached: cannot reach the update
        with T.no_grad():
            ce = cross_entropy(logits.detach()[real], y[real], reduction="none").double()
            fr = fine[real]
            for i, n in enumerate(FINE):
                sel = fr == i
                roles_acc[n][0] += float(ce[sel].sum()); roles_acc[n][1] += int(sel.sum())
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach())


def new_roles():
    return {n: [0.0, 0] for n in FINE}


def summarize_roles(acc):
    fine = {n: dict(mean=(s / c if c else None), positions=c) for n, (s, c) in acc.items()}
    coarse = {}
    for r in ROLES:
        s = sum(acc[n][0] for n in FINE if FINE_TO_ROLE[n] == r); c = sum(acc[n][1] for n in FINE if FINE_TO_ROLE[n] == r)
        coarse[r] = dict(mean=(s / c if c else None), positions=c)
    return dict(by_role=coarse, by_fine_role=fine)


# ---------------------------------------------------------------- files
def birth_path(out, arm, seed):
    return Path(out) / f"e28_dev_{arm}{E.DEVTAG}_{seed}.pt"


def row_path(out, arm, seed, stop):
    return Path(out) / f"e28_dev_{arm}{E.DEVTAG}_{seed}_row{stop}.pt"


def e26_rows(seed):
    """E26's data digests by row, from both arms' receipts of this seed (they must agree)."""
    out = {}
    for a in ("S", "B"):
        r = json.loads((E26_DIR / f"e26_dev_{a}_{seed}.json").read_text())
        for row in r["curve"]:
            v = out.setdefault(row["step"], row["data_digest"])
            assert v == row["data_digest"], f"E26's two arms disagree at row {row['step']} of seed {seed}"
    return out


# ---------------------------------------------------------------- birth
def dev(arm, seed, out, cmd):
    local, targets = ARMS[arm]
    arch26.install()
    path = birth_path(out, arm, seed); rec_path = path.with_suffix(".json")
    R.refuse_existing(path, rec_path, str(path) + ".tmp", *[row_path(out, arm, seed, s) for s in range(4096, E.DEV_DOSE, 4096)])
    Path(out).mkdir(parents=True, exist_ok=True)
    ref_digest = e26_rows(seed)
    m = E.new_net(seed)
    desc = arch26.describe(m)
    book0 = m.book.weight.detach().clone()
    opt = C.make_opt(m)
    census = None
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    pan = {E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV}
    bt = E.batches(tp, seed, E.KMAX["M"])
    dg = C.Digest()
    rec = dict(experiment="E28", kind="birth", arm=arm, targets=targets, local=local, net="G1", seed=seed, kmax=E.KMAX["M"], dev_updates=E.DEV_DOSE,
               block=X.BLOCK, batch=X.BATCH, optimizer=C.opt_record(m, opt),
               update=f"exp28_everybyte.step: forward(local), {targets}-target cross entropy, zero_grad, backward, AdamW step (credit through the whole item)",
               loss_definition=dict(answer="CE over the answer mask (E26's loss)", all="CE over y != BOS (every real byte)",
                                    balanced="0.5 * mean CE over answer positions + 0.5 * mean CE over the other real positions")[targets],
               roles=dict(coarse=list(ROLES), fine=list(FINE), fine_to_coarse=FINE_TO_ROLE,
                          identification="from the item's layout (exp11_cell.render): code a1 a2 op d_1..d_k b1 b2 '=' | c1 c2 EOS, k = len(prompt) - 7; "
                                         "byte classes asserted on every item; the answer role equals the answer mask (asserted); EOS counted under answer, "
                                         "the task code under structural"),
               protocol="exp16_integrated.develop17 plain path (four mod-89 operations, k 0..12), as exp26_candidate.dev, loss mask changed only",
               model=desc, installed=dict(E_forward=f"{E.forward.__module__}.{E.forward.__name__}", E_new_net_class=desc["cls"]),
               provenance=provenance(cmd), complete=False, curve=[], role_curve=[], finite=dict(all_finite=True, updates_covered=0, first_nonfinite_update=None))
    t0 = time.process_time(); cpu_score = 0.0; cpu_save = 0.0; curve = rec["curve"]; row = None; nupd = 0
    for stop in range(X.BLOCK, E.DEV_DOSE + 1, X.BLOCK):
        losses = []; acc = new_roles()
        for _ in range(X.BLOCK):
            items = next(bt); dg.add(items)
            v = step(m, opt, items, local, targets, acc); losses.append(v); nupd += 1
            if not math.isfinite(v) and rec["finite"]["all_finite"]:
                rec["finite"].update(all_finite=False, first_nonfinite_update=nupd)
            rec["finite"]["updates_covered"] = nupd
            if census is None:
                census = X16.zero_grad_report(m, X.batch_tensor(items)[0]); print("e28 dev credit census:", census["by_role"], flush=True)
        rec["role_curve"].append(dict(block_end=stop, train_loss=sum(losses) / len(losses), **summarize_roles(acc)))
        if stop % 4096 == 0 or stop == E.DEV_DOSE:
            cb = time.process_time()
            sealed = {n: {k: E.score(m, it) for k, it in pk.items()} for n, pk in pan.items()}
            cpu_score += time.process_time() - cb
            d_now = dg.value()
            if stop in ref_digest:
                assert d_now == ref_digest[stop], f"data digest at row {stop} is not E26's for seed {seed}: {d_now} vs {ref_digest[stop]}"
                dcheck = "equal to E26's receipt (both arms) at this row: asserted"
            else:
                dcheck = f"not asserted: E26's receipts have no row at step {stop} (reduced dose)"
            cs = time.process_time()
            if stop == E.DEV_DOSE:
                wf, wsha = path, None             # the final row's weights are the birth file, written below
            else:
                wf = row_path(out, arm, seed, stop); wsha = R.save_weights(m, wf)
            cpu_save += time.process_time() - cs
            hard = {n: v[12]["exact"] for n, v in sealed.items()}
            row = dict(step=stop, loss=sum(losses) / len(losses), finite_losses=all(math.isfinite(v) for v in losses),
                       finite_cumulative=dict(rec["finite"]), cpu=time.process_time() - t0, cpu_scoring_so_far=cpu_score,
                       cpu_updates_so_far=time.process_time() - t0 - cpu_score - cpu_save, sealed=sealed,
                       hard_mean_add89_sub89_k12=(hard["add89"] + hard["sub89"]) / 2 / 300 * 100,
                       book_shift_by_role=X16.book_shift(m, book0), data_digest=d_now, data_digest_check=dcheck,
                       weights=wf.name, weights_sha256=wsha, tensors_sha256=R.tensors_sha(m.state_dict()))
            curve.append(row)
            print("e28 dev", arm, seed, stop, {n: [v[k]["exact"] for k in (0, 4, 8, 12, 20)] for n, v in row["sealed"].items()},
                  f"hard {row['hard_mean_add89_sub89_k12']:.1f} cpu {row['cpu']:.0f}s", flush=True)
            X16.write_json(rec_path, dict(rec, credit_census_first_update=census))
    assert row is not None, "the birth loop works in whole blocks of 512 updates: E11_DEV_UPDATES must be a positive multiple of 512"
    cb = time.process_time()
    row["silenced_k12"] = {n: E.score(m, pk[12], silence=True) for n, pk in pan.items()}
    cpu_score += time.process_time() - cb
    wsha = R.save_weights(m, path)
    row["weights_sha256"] = wsha
    fin = dg.value()
    if E.DEV_DOSE in ref_digest:
        assert fin == ref_digest[E.DEV_DOSE]
    rec.update(credit_census_first_update=census, optimizer_steps=R.opt_steps(opt), optimizer_read_back=C.opt_record(m, opt), data_digest=fin,
               weights=path.name, weights_sha256=wsha, tensors_sha256=R.tensors_sha(m.state_dict()),
               row_weights={str(r["step"]): dict(file=r["weights"], sha256=r["weights_sha256"]) for r in curve},
               cpu_total=time.process_time() - t0, cpu_scoring=cpu_score, cpu_saving_rows=cpu_save,
               cpu_updates=time.process_time() - t0 - cpu_score - cpu_save, complete=True)
    rec["cpu_per_update"] = rec["cpu_updates"] / E.DEV_DOSE
    X16.write_json(rec_path, rec)
    print("e28 dev", arm, seed, "complete", wsha[:16], flush=True)


if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])
    outdir = a[a.index("--out") + 1] if "--out" in a else str(X16.OUT / "e28")
    if a and a[0] == "dev":
        assert a[1] in ARMS, f"arm must be one of {list(ARMS)}"
        dev(a[1], int(a[2]), outdir, cmd)
    else:
        raise SystemExit(__doc__)
