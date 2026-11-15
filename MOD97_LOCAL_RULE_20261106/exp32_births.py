"""E32 runner, main worker (brief OPUS_TASK_E32_EVERY_BYTE_BIRTH_SWITCHES_20261114.md, revision 4): E28's S-all birth (every byte a target,
the sweep, 24,576 updates) with one addition per arm. A new file; nothing existing is edited; E31's files are not imported.

    python3 exp32_births.py dev <arm> <seed> [--out DIR]      arm: floor G H K P4a P4b P4c P4z     (P4z: lam = 0.0, computed and added; check 6)
    python3 exp32_births.py check [--out DIR]                 checks 3 to 6 on seed 73129 -> E32_CHECK_MAIN.json
    python3 exp32_births.py calib [--out DIR]                 check 7, P4: the nine lam values without training -> E32_LAMBDA_MAIN.json
    python3 exp32_births.py watch-ceiling <seed> [--out DIR]  check 2: compares each row the E28 runner writes for B-all with E28's receipt

The update, one loop (E28's S-all loop: exp28_everybyte.masks and loss_of, imported unedited; the same data stream, optimizer and scorer):

    x, y, amask, real, fine = exp28_everybyte.masks(items)
    logits, out = forward(m, x, switch, Bhh)                    # this file's forward: arch26.forward's values, the arm's way back
    loss  = cross_entropy(logits[real], y[real])                # every real byte a target
    total = loss + p4_term(...)  for P4 arms from update 4,097   # lam registered in E32_PREDICTIONS.md / E32_LAMBDA_MAIN.json
    opt.zero_grad(); total.backward(); opt.step()
    K: Bhh <- (1 - 0.00025) Bhh + (W1 - (1 - 0.00025) W0)       # W0, W1: weight_hh_l0 before and after the step

Arms (the way back; forward values are arch26.forward's in every arm):
    floor  the sweep everywhere (exp3_state.lin(., ., True)); must reproduce E28's S-all
    G      weight_ih_l0 and weight_hh_l0: gx = g[..., :512] @ W[:512] + g[..., 512:] @ S[512:], two products added, S = sign(W) * mean|W|
           (whole matrix); every other map the sweep
    H      weight_hh_l0: gx = g @ W; every other map the sweep
    K      weight_hh_l0: gx = g @ Bhh, Bhh (768 x 256) from U(-1/16, 1/16) by torch.Generator().manual_seed(seed + 31); every other map the sweep
    P4*    the sweep everywhere; from update 4,097 the term
               hp[:, t] = 0 at t = 0, else the state after step t-1 (the state step t reads)
               msk  = (x != BOS) & (y != BOS)
               a    = lin(z1.detach(), ih, False)[..., 256:512] + lin(hp.detach(), hh, False)[..., 256:512]
               term = lam * (relu(4.59512 - |a|) ** 2)[msk].sum() / (msk.sum() * 256)
           lam = ratio * (mean task norm) / (mean term norm at lam 1), ratio 0.25 / 0.05 / 0.01 (P4a / P4b / P4c), the norms taken jointly
           over rows 256-511 of weight_ih_l0 and weight_hh_l0 and the same entries of both biases, the task norm as the sweep computes it,
           the means over batches 4,097 to 4,224 at the weights after update 4,096. P4z: lam = 0.0, the term computed and added.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp28_everybyte as B28        # noqa: E402  (imports exp26_candidate and exp20_reference: environment gate; not edited)

import torch as T                    # noqa: E402
from torch.nn.functional import cross_entropy, linear  # noqa: E402

C, R, X16, E, X = B28.C, B28.R, B28.X16, B28.E, B28.X
import arch26                        # noqa: E402
from exp3_state import lin           # noqa: E402

OUT = X.OUT / "e32"
E28_DIR = X.OUT / "e28"
E28_DRY = X.OUT / "e28_qual" / "dry"
SEEDS = (73129, 93683, 116089)
ARMS = ("floor", "G", "H", "K", "P4a", "P4b", "P4c", "P4z")
RATIO = {"P4a": 0.25, "P4b": 0.05, "P4c": 0.01, "P4z": None}
GATE_ROWS = 512
Z = slice(256, 512)
CLIP = 4.59512                       # logit(0.99)
DECAY = C.LR * C.WD                  # 0.00025
P4_START = 4096                      # the term is added from update 4,097
CAL_BATCHES = 128                    # batches 4,097 to 4,224
SOURCES = B28.SOURCES + ("exp32_births.py",)
TOL = 1e-5                           # the brief's tolerance: max |a - b| <= 1e-5 * max |reference|


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_hashes():
    return {s: sha_file(HERE / s) for s in SOURCES}


def provenance(cmd):
    return dict(command=cmd, environment_at_start=R.ENV_AT_START, environment_set_by_runner=R.REQUIRED_ENV, host=R.host(), git=B28.git_state(),
                interpreter=B28.environment(), sources_sha256=source_hashes())


# ================================================================ the forward pass with the arm's way back
class SwitchLinear(T.autograd.Function):
    """y = x W^T + b. Backward: the arm's gx; gw = g^T x and gb = sum g, as exp3_state.SignLinear computes them."""

    @staticmethod
    def forward(ctx, x, weight, bias, mode, B):
        ctx.save_for_backward(x, weight, B if B is not None else weight)
        ctx.mode = mode
        return linear(x, weight, bias)

    @staticmethod
    def backward(ctx, g):
        x, weight, B = ctx.saved_tensors
        if ctx.mode == "G":
            S = weight.sign() * weight.abs().mean()
            gx = g[..., :GATE_ROWS] @ weight[:GATE_ROWS] + g[..., GATE_ROWS:] @ S[GATE_ROWS:]
        elif ctx.mode == "H":
            gx = g @ weight
        elif ctx.mode == "K":
            gx = g @ B
        else:
            raise AssertionError(ctx.mode)
        gw = g.reshape(-1, g.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        gb = g.reshape(-1, g.shape[-1]).sum(0)
        return gx, gw, gb, None, None


def smap(x, layer, mode, B=None):
    if mode is None:
        return lin(x, layer, True)
    return SwitchLinear.apply(x, layer.weight, layer.bias, mode, B)


def maps(arm):
    """(input-map mode, recurrent-map mode) of an arm; None = the sweep."""
    return {"G": ("G", "G"), "H": (None, "H"), "K": (None, "K")}.get(arm, (None, None))


def forward(m, x, arm="floor", Bhh=None, keep=False):
    """arch26.forward(m, x, True, True) with the arm's way back. Returns logits and dict(z1, hp (the state each step reads), h, z, gh)."""
    assert getattr(m, "kind", None) == "gru"
    ih_mode, hh_mode = maps(arm)
    assert (hh_mode == "K") == (Bhh is not None)
    u = m.book(x)
    z1 = T.tanh(lin(u, m.l1, True))
    B, L = x.shape
    ih = arch26.Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0)
    hh = arch26.Mat(m.rnn.weight_hh_l0, m.rnn.bias_hh_l0)
    gi = smap(z1, ih, ih_mode)
    h = T.zeros(B, m.units)
    hs, hps, zs, ghs = [], [], [], []
    for t in range(L):
        hp = h
        gh = smap(hp, hh, hh_mode, Bhh)
        if keep and gh.requires_grad:
            gh.retain_grad()
        ghs.append(gh)
        i_r, i_z, i_n = gi[:, t].chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        r = T.sigmoid(i_r + h_r)
        z = T.sigmoid(i_z + h_z)
        nn_ = T.tanh(i_n + r * h_n)
        h = (1 - z) * nn_ + z * hp
        hs.append(h); hps.append(hp); zs.append(z)
    Hs = T.stack(hs, 1)
    z2 = T.tanh(lin(z1, m.l2, True) + lin(Hs, m.lm, True))
    return lin(z2, m.l3, True), dict(z1=z1, h=hs, hp=hps, z=zs, gh=ghs)


# ================================================================ arm P4: the term
def p4_term(m, x, y, out, lam):
    z1 = out["z1"].detach()
    hp = T.stack(out["hp"], 1).detach()                       # hp[:, t] = the state step t reads (zeros at t = 0)
    ih = arch26.Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0)
    hh = arch26.Mat(m.rnn.weight_hh_l0, m.rnn.bias_hh_l0)
    a = lin(z1, ih, False)[..., 256:512] + lin(hp, hh, False)[..., 256:512]
    msk = (x != X.BOS) & (y != X.BOS)
    term = lam * (T.relu(CLIP - a.abs()) ** 2)[msk].sum() / (msk.sum() * 256)
    return term, dict(a=a, msk=msk, z1=z1, hp=hp)


def gate_params(m):
    return [m.rnn.weight_ih_l0, m.rnn.weight_hh_l0, m.rnn.bias_ih_l0, m.rnn.bias_hh_l0]


def joint_norm(grads):
    return float(T.sqrt(sum((g[Z].double() ** 2).sum() for g in grads)))


def norms_on(m, items):
    """(task norm as the sweep computes it, term norm at lam = 1), each one joint norm over the update-gate rows; no update is made."""
    x, y, amask, real, fine = B28.masks(items)
    logits, out = forward(m, x, "floor")
    task = B28.loss_of(logits, y, amask, real, "all")
    P = gate_params(m)
    gt = T.autograd.grad(task, P, retain_graph=True)
    term, _ = p4_term(m, x, y, out, 1.0)
    gg = T.autograd.grad(term, P)
    return joint_norm(gt), joint_norm(gg)


# ================================================================ data
def stream(seed):
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    return E.batches(tp, seed, E.KMAX["M"])


def e28_rows(seed, arm="S-all"):
    r = json.loads((E28_DIR / f"e28_dev_{arm}_{seed}.json").read_text())
    return {c["step"]: c for c in r["curve"]}, r


def calibration_batches(seed):
    """Batches 4,097 to 4,224 of the birth stream; the stream's digest after 4,096 batches must be E28's at row 4,096."""
    bt = stream(seed); dg = C.Digest()
    for _ in range(P4_START):
        dg.add(next(bt))
    rows, _ = e28_rows(seed)
    assert dg.value() == rows[P4_START]["data_digest"], "the stream before batch 4,097 is not E28's"
    return [next(bt) for _ in range(CAL_BATCHES)]


def calibrate(m, batches):
    pairs = [norms_on(m, b) for b in batches]
    mt = sum(p[0] for p in pairs) / len(pairs); mg = sum(p[1] for p in pairs) / len(pairs)
    return dict(mean_task_norm=mt, mean_term_norm_at_lam1=mg, first_batch_task_norm=pairs[0][0], first_batch_term_norm_at_lam1=pairs[0][1],
                task_norm_min=min(p[0] for p in pairs), task_norm_max=max(p[0] for p in pairs),
                lam={a: r * mt / mg for a, r in RATIO.items() if r is not None})


def load_row(seed, stop):
    m = E.new_net(seed)
    f = E28_DIR / f"e28_dev_S-all_{seed}_row{stop}.pt"
    rows, _ = e28_rows(seed)
    assert X16.sha_file(f) == rows[stop]["weights_sha256"], f"{f.name}: weights hash differs from E28's receipt"
    m.load_state_dict(T.load(f))
    assert R.tensors_sha(m.state_dict()) == rows[stop]["tensors_sha256"]
    return m


# ================================================================ arm K
class Feedback:
    def __init__(self, seed, shape=(768, 256)):
        g = T.Generator().manual_seed(seed + 31)
        self.B = T.empty(*shape).uniform_(-1 / 16, 1 / 16, generator=g)
        self.B0_sha = R.tensors_sha({"Bhh": self.B})
        self.steps = 0

    def after_step(self, W0, W1):
        self.B = (1 - DECAY) * self.B + (W1 - (1 - DECAY) * W0)
        self.steps += 1

    def measures(self, W):
        W = W.detach().double(); Bd = self.B.double()
        return dict(cos_Bhh_W=float((Bd * W).sum() / (Bd.norm() * W.norm())), rel_diff_Bhh_W=float((Bd - W).norm() / W.norm()),
                    feedback_updates=self.steps, Bhh_tensors_sha256=R.tensors_sha({"Bhh": self.B}))


# ================================================================ the birth
def registered_lambda(seed, arm):
    rec = json.loads((OUT / "E32_LAMBDA_MAIN.json").read_text())
    return rec["seeds"][str(seed)]["lam"][arm]


def birth_path(out, arm, seed):
    return Path(out) / f"e32_dev_{arm}{E.DEVTAG}_{seed}.pt"


def row_path(out, arm, seed, stop):
    return Path(out) / f"e32_dev_{arm}{E.DEVTAG}_{seed}_row{stop}.pt"


def dev(arm, seed, out, cmd):
    assert arm in ARMS
    arch26.install()
    out = Path(out)
    path = birth_path(out, arm, seed); rec_path = path.with_suffix(".json")
    R.refuse_existing(path, rec_path, str(path) + ".tmp", *[row_path(out, arm, seed, s) for s in range(4096, E.DEV_DOSE, 4096)])
    out.mkdir(parents=True, exist_ok=True)
    e28, e28rec = e28_rows(seed)
    full = E.DEV_DOSE == 24576
    m = E.new_net(seed)
    desc = arch26.describe(m)
    book0 = m.book.weight.detach().clone()
    opt = C.make_opt(m)
    pan = {E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV}
    bt = stream(seed)
    dg = C.Digest()
    fb = Feedback(seed) if arm == "K" else None
    lam = None; lam_rec = None
    if arm.startswith("P4"):
        if arm == "P4z":
            lam, lam_rec = 0.0, dict(lam=0.0, note="check 6: lam = 0.0, the term computed and added, not skipped")
        else:
            lam_rec = dict(registered=registered_lambda(seed, arm))
    rec = dict(experiment="E32", kind="birth", arm=arm, worker="main", targets="all", local=True, net="G1", seed=seed, kmax=E.KMAX["M"],
               dev_updates=E.DEV_DOSE, block=X.BLOCK, batch=X.BATCH, optimizer=C.opt_record(m, opt),
               update="exp32_births.dev: E28's S-all loop (every real byte a target, the sweep) with the arm's addition",
               arm_rule=dict(maps=maps(arm), K=dict(Bhh_seed=seed + 31, Bhh_start_tensors_sha256=fb.B0_sha, decay=DECAY) if fb else None,
                             P4=dict(ratio=RATIO.get(arm), start_after_update=P4_START, clip=CLIP, lam=lam_rec) if arm.startswith("P4") else None),
               model=desc, provenance=provenance(cmd), complete=False, curve=[],
               finite=dict(all_finite=True, updates_covered=0, first_nonfinite_update=None))
    t0 = time.process_time(); cpu_score = 0.0; cpu_save = 0.0; cpu_diag = 0.0; curve = rec["curve"]; row = None; nupd = 0
    first_loss = None
    for stop in range(X.BLOCK, E.DEV_DOSE + 1, X.BLOCK):
        losses = []
        for j in range(X.BLOCK):
            if arm.startswith("P4") and arm != "P4z" and nupd == P4_START and lam is None:
                cd = time.process_time()
                cal = calibrate(m, calibration_batches(seed))
                lam = cal["lam"][arm]
                reg = lam_rec["registered"]
                assert abs(lam - reg) <= 1e-12 * abs(reg), f"lam {lam!r} is not the registered {reg!r}"
                lam_rec.update(computed_in_run=lam, calibration=cal)
                cpu_diag += time.process_time() - cd
                print("e32", arm, seed, "lam", lam, "registered", reg, flush=True)
            items = next(bt); dg.add(items)
            x, y, amask, real, fine = B28.masks(items)
            logits, outd = forward(m, x, arm, fb.B if fb else None)
            loss = B28.loss_of(logits, y, amask, real, "all")
            total = loss
            if arm.startswith("P4") and nupd >= P4_START:
                term, _ = p4_term(m, x, y, outd, lam)
                total = loss + term
            opt.zero_grad()
            total.backward()
            if fb is not None:
                W0 = m.rnn.weight_hh_l0.detach().clone()
            opt.step()
            if fb is not None:
                fb.after_step(W0, m.rnn.weight_hh_l0.detach().clone())
            v = float(loss.detach()); losses.append(v); nupd += 1
            if first_loss is None:
                first_loss = v
            if not math.isfinite(v) and rec["finite"]["all_finite"]:
                rec["finite"].update(all_finite=False, first_nonfinite_update=nupd)
            rec["finite"]["updates_covered"] = nupd
        if stop % 4096 == 0 or stop == E.DEV_DOSE:
            cb = time.process_time()
            sealed = {n: {k: E.score(m, it) for k, it in pk.items()} for n, pk in pan.items()}
            cpu_score += time.process_time() - cb
            d_now = dg.value()
            if stop in e28:
                assert d_now == e28[stop]["data_digest"], f"data digest at row {stop} is not E28's"
                dcheck = "equal to E28 S-all's receipt at this row: asserted"
            else:
                dcheck = f"E28 has no row at {stop}"
            cs = time.process_time()
            if stop == E.DEV_DOSE:
                wf, wsha = path, None
            else:
                wf = row_path(out, arm, seed, stop); wsha = R.save_weights(m, wf)
            cpu_save += time.process_time() - cs
            tsha = R.tensors_sha(m.state_dict())
            e32 = {}
            if fb is not None:
                e32["K"] = fb.measures(m.rnn.weight_hh_l0)
            if arm.startswith("P4"):
                cd = time.process_time()
                nt, ng = norms_on(m, items)
                cpu_diag += time.process_time() - cd
                e32["P4_norms_on_last_batch_of_block"] = dict(task_norm=nt, term_norm_at_lam1=ng, term_norm_at_lam=(lam or 0.0) * ng)
            if arm.startswith("P4") and stop == P4_START and stop in e28:
                assert tsha == e28[stop]["tensors_sha256"], "a P4 birth's row 4,096 is not E28's"
                e32["row_4096_equals_E28"] = True
            hard = {n: v[12]["exact"] for n, v in sealed.items()}
            row = dict(step=stop, loss=sum(losses) / len(losses), finite_losses=all(math.isfinite(v) for v in losses),
                       cpu=time.process_time() - t0, cpu_scoring_so_far=cpu_score, sealed=sealed,
                       hard_mean_add89_sub89_k12=(hard["add89"] + hard["sub89"]) / 2 / 300 * 100, book_shift_by_role=X16.book_shift(m, book0),
                       data_digest=d_now, data_digest_check=dcheck, weights=wf.name, weights_sha256=wsha, tensors_sha256=tsha,
                       e28_row_tensors_sha256=e28[stop]["tensors_sha256"] if stop in e28 else None, e32=e32)
            curve.append(row)
            print("e32 dev", arm, seed, stop, f"hard {row['hard_mean_add89_sub89_k12']:.1f}", tsha[:12],
                  "E28" if stop in e28 and tsha == e28[stop]["tensors_sha256"] else "", e32.get("K", {}).get("cos_Bhh_W", ""),
                  f"cpu {row['cpu']:.0f}s", flush=True)
            X16.write_json(rec_path, dict(rec, first_batch_task_loss=first_loss))
    wsha = R.save_weights(m, path)
    row["weights_sha256"] = wsha
    rec.update(first_batch_task_loss=first_loss, optimizer_steps=R.opt_steps(opt), optimizer_read_back=C.opt_record(m, opt), data_digest=dg.value(),
               weights=path.name, weights_sha256=wsha, tensors_sha256=R.tensors_sha(m.state_dict()),
               row_weights={str(r["step"]): dict(file=r["weights"], sha256=r["weights_sha256"]) for r in curve},
               cpu_total=time.process_time() - t0, cpu_scoring=cpu_score, cpu_saving_rows=cpu_save, cpu_diagnostics=cpu_diag,
               cpu_updates=time.process_time() - t0 - cpu_score - cpu_save - cpu_diag, complete=True)
    if fb is not None:
        T.save({"Bhh": fb.B}, path.with_name(path.stem + "_Bhh.pt"))
        rec["Bhh_final_file"] = path.stem + "_Bhh.pt"
    X16.write_json(rec_path, rec)
    print("e32 dev", arm, seed, "complete", rec["tensors_sha256"][:16], flush=True)


# ================================================================ checks 3 to 6 (seed 73129) and check 7's P4 values
def maxdiff(a, b):
    """max |a - b| / max |b|: the brief's tolerance is 1e-5."""
    return float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30))


def all_grads(m, x, y, amask, real, arm, Bhh=None):
    m.zero_grad()
    logits, out = forward(m, x, arm, Bhh)
    B28.loss_of(logits, y, amask, real, "all").backward()
    return {n: p.grad.detach().clone() for n, p in m.named_parameters()}


def checks(out):
    seed = 73129
    arch26.install()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = out / "E32_CHECK_MAIN.json"
    R.refuse_existing(p)
    bt = stream(seed)
    first = [next(bt) for _ in range(100)]
    m0 = E.new_net(seed)
    x, y, amask, real, fine = B28.masks(first[0])
    res = dict(experiment="E32", kind="checks", worker="main", seed=seed, host=R.host(), interpreter=B28.environment(), sources_sha256=source_hashes(),
               git=B28.git_state())
    # check 3: first-batch task loss of every arm equals the floor's (and E28's forward) bit for bit
    ref = B28.loss_of(arch26.forward(m0, x, True, True)[0], y, amask, real, "all")
    losses = {}
    for arm in ("floor", "G", "H", "K", "P4a"):
        Bhh = Feedback(seed).B if arm == "K" else None
        lo = B28.loss_of(forward(m0, x, arm, Bhh)[0], y, amask, real, "all")
        losses[arm] = dict(loss=float(lo), bit_equal_to_arch26_forward=bool(T.equal(lo, ref)))
    res["check3_first_batch_loss"] = dict(arch26_forward=float(ref), arms=losses, ok=all(v["bit_equal_to_arch26_forward"] for v in losses.values()),
                                          note="data digests are asserted against E28's at every saved row inside every birth")
    # floor gradients equal arch26.forward(local=True)'s
    m0.zero_grad(); B28.loss_of(arch26.forward(m0, x, True, True)[0], y, amask, real, "all").backward()
    g26 = {n: p_.grad.detach().clone() for n, p_ in m0.named_parameters()}
    gfl = all_grads(m0, x, y, amask, real, "floor")
    res["floor_grads_equal_arch26"] = dict(bit_equal=all(T.equal(gfl[n], g26[n]) for n in g26))
    # check 4: G and H, the error sent back across the recurrent map (and G's input map) on one batch
    m = load_row(seed, 4096)                                    # trained weights, so the sweep and the exact way back differ
    W = m.rnn.weight_hh_l0.detach(); S = W.sign() * W.abs().mean()
    c4 = {}
    for arm in ("G", "H"):
        m.zero_grad()
        logits, o = forward(m, x, arm, keep=True)
        B28.loss_of(logits, y, amask, real, "all").backward()
        rows = []
        for t in (1, x.shape[1] // 2, x.shape[1] - 1):
            g = o["gh"][t].grad.detach()
            hp = o["hp"][t].detach().requires_grad_()
            gx, = T.autograd.grad(smap(hp, arch26.Mat(m.rnn.weight_hh_l0, m.rnn.bias_hh_l0), maps(arm)[1]), hp, g)
            if arm == "G":
                pinned = g[:, :512] @ W[:512] + g[:, 512:] @ S[512:]
                hand = T.einsum("bo,oi->bi", g, T.cat([W[:512], S[512:]], 0))            # second form: one stacked matrix, einsum
            else:
                pinned = g @ W
                hand = T.einsum("bo,oi->bi", g, W)
            rows.append(dict(t=t, pinned_max_rel=maxdiff(gx, pinned), bit_equal_pinned=bool(T.equal(gx, pinned)), hand_max_rel=maxdiff(gx, hand),
                             vs_sweep_max_rel=maxdiff(gx, g @ S)))
        c4[arm] = dict(steps=rows, ok=all(r["bit_equal_pinned"] and r["hand_max_rel"] <= TOL for r in rows))
    Wi = m.rnn.weight_ih_l0.detach(); Si = Wi.sign() * Wi.abs().mean()
    z1 = forward(m, x, "floor")[1]["z1"].detach().requires_grad_()
    gi = smap(z1, arch26.Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0), "G")
    gg = T.randn(gi.shape, generator=T.Generator().manual_seed(1))
    gx, = T.autograd.grad(gi, z1, gg)
    c4["G_input_map"] = dict(pinned_bit_equal=bool(T.equal(gx, gg[..., :512] @ Wi[:512] + gg[..., 512:] @ Si[512:])),
                             hand_max_rel=maxdiff(gx, T.einsum("blo,oi->bli", gg, T.cat([Wi[:512], Si[512:]], 0))), note="random g, real z1")
    res["check4_G_H"] = c4
    # check 5: K
    m = E.new_net(seed); opt = C.make_opt(m); fb = Feedback(seed); rows = []
    for u, items in enumerate(first, 1):
        xx, yy, aa, rr, _ = B28.masks(items)
        Bold = fb.B.clone(); W0 = m.rnn.weight_hh_l0.detach().clone()
        opt.zero_grad(); B28.loss_of(forward(m, xx, "K", fb.B)[0], yy, aa, rr, "all").backward(); opt.step()
        W1 = m.rnn.weight_hh_l0.detach().clone(); fb.after_step(W0, W1)
        if u in (1, 10, 100):
            rows.append(dict(update=u, identity_max_abs=float((fb.B - ((1 - DECAY) * Bold + (W1 - (1 - DECAY) * W0))).abs().max()),
                             contraction_max_abs=float(((fb.B - W1) - (1 - DECAY) * (Bold - W0)).abs().max()), **fb.measures(W1)))
    m = load_row(seed, 4096)
    gH = all_grads(m, x, y, amask, real, "H")
    gK = all_grads(m, x, y, amask, real, "K", m.rnn.weight_hh_l0.detach().clone())
    rel = {n: maxdiff(gK[n], gH[n]) for n in gH}
    res["check5_K"] = dict(update_identity=rows, identity_ok=all(r["identity_max_abs"] <= 1e-7 and r["contraction_max_abs"] <= 1e-7 for r in rows),
                           K_equals_H_when_Bhh_is_W=dict(max_rel=rel, ok=max(rel.values()) <= TOL))
    # check 6: P4
    m = load_row(seed, 4096)
    logits, o = forward(m, x, "floor")
    term, aux = p4_term(m, x, y, o, 1.0)
    zf = T.stack(o["z"], 1).detach()
    zt = T.sigmoid(aux["a"]).detach()
    msk = aux["msk"]
    zlate = T.cat([zf[:, :1], zf[:, :-1]], 1)                    # the forward's gate one step late: must not agree
    res_z = dict(max_rel_all_positions=maxdiff(zt, zf), identical_all_positions=bool(T.equal(zt, zf)),
                 max_abs_masked=float((zt - zf)[msk].abs().max()), identical_masked=bool(T.equal(zt[msk], zf[msk])),
                 one_step_late_max_abs=float((zt - zlate)[msk].abs().max()))
    m.zero_grad(); term.backward()
    outside = {}
    for n, p_ in m.named_parameters():
        g = p_.grad if p_.grad is not None else T.zeros_like(p_)
        if n in ("rnn.weight_ih_l0", "rnn.weight_hh_l0", "rnn.bias_ih_l0", "rnn.bias_hh_l0"):
            g = g.clone(); g[Z] = 0
        outside[n] = float(g.abs().max())
    a = aux["a"].detach().double()
    N = int(msk.sum()) * 256
    da = (-2.0 * T.relu(CLIP - a.abs()) * a.sign() / N) * msk.unsqueeze(-1).double()      # d term / d a  (lam = 1)
    da2 = da.reshape(-1, 256)
    hand = {"rnn.weight_ih_l0": da2.T @ aux["z1"].reshape(-1, aux["z1"].shape[-1]).double(), "rnn.weight_hh_l0": da2.T @ aux["hp"].reshape(-1, 256).double(),
            "rnn.bias_ih_l0": da2.sum(0), "rnn.bias_hh_l0": da2.sum(0)}
    pr = dict(m.named_parameters())
    two = {n: maxdiff(pr[n].grad[Z], h) for n, h in hand.items()}
    res["check6_P4"] = dict(gate_agreement=dict(res_z, ok=res_z["max_rel_all_positions"] <= TOL),
                            term_grad_outside_update_gate_rows=dict(max_abs=outside, ok=all(v == 0.0 for v in outside.values())),
                            term_grad_two_factor=dict(max_rel=two, ok=max(two.values()) <= TOL,
                                                      form="d/dW = sum over masked items, positions of input * (-2 relu(C - |a|) sign(a)) / (masked positions * 256)"),
                            lam0_birth="exp32_births.py dev P4z 73129 at E11_DEV_UPDATES=8192: rows 4,096 and 8,192 against E28's")
    oks = [res["check3_first_batch_loss"]["ok"], res["floor_grads_equal_arch26"]["bit_equal"], c4["G"]["ok"], c4["H"]["ok"], c4["G_input_map"]["pinned_bit_equal"],
           res["check5_K"]["identity_ok"], res["check5_K"]["K_equals_H_when_Bhh_is_W"]["ok"], res["check6_P4"]["gate_agreement"]["ok"],
           res["check6_P4"]["term_grad_outside_update_gate_rows"]["ok"], res["check6_P4"]["term_grad_two_factor"]["ok"]]
    res["pass"] = all(oks)
    X16.write_json(p, res)
    print("e32 checks pass:", res["pass"], "->", p, flush=True)


REVIEWER = {73129: (0.036740, 0.890970), 93683: (0.077113, 0.836491), 116089: (0.093320, 0.849241)}   # batch 4,097 alone (brief, check 7)


def calib(out):
    arch26.install()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = out / "E32_LAMBDA_MAIN.json"
    R.refuse_existing(p)
    res = dict(experiment="E32", kind="P4 strengths (check 7), no training", worker="main", host=R.host(), interpreter=B28.environment(),
               sources_sha256=source_hashes(), definition="lam = ratio * mean task norm / mean term norm at lam 1, over batches 4,097-4,224 at E28 S-all row 4,096; "
               "one joint norm over rows 256-511 of weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0; task norm as the sweep computes it", seeds={})
    for seed in SEEDS:
        t0 = time.process_time()
        m = load_row(seed, 4096)
        cal = calibrate(m, calibration_batches(seed))
        rt, rg = REVIEWER[seed]
        cal["reviewer_batch_4097"] = dict(task_norm=rt, term_norm_at_lam1=rg, rel_diff_task=abs(cal["first_batch_task_norm"] - rt) / rt,
                                          rel_diff_term=abs(cal["first_batch_term_norm_at_lam1"] - rg) / rg)
        cal["cpu"] = time.process_time() - t0
        res["seeds"][str(seed)] = cal
        print("e32 calib", seed, {k: v for k, v in cal.items() if k != "lam"}, cal["lam"], flush=True)
    X16.write_json(p, res)


def watch_ceiling(seed, out):
    """Check 2: poll the E28 runner's B-all receipt in DIR/ceiling; compare every row's tensors_sha256 with E28's as it is written; on a
    mismatch, kill the run (by its command line) and record it."""
    import subprocess
    rec = Path(out) / "ceiling" / f"e28_dev_B-all_{seed}.json"
    ref, _ = e28_rows(seed, "B-all")
    seen = {}
    log = Path(out) / "ceiling" / f"watch_B-all_{seed}.json"
    while True:
        if rec.exists():
            try:
                r = json.loads(rec.read_text())
            except ValueError:
                time.sleep(20); continue
            for c in r["curve"]:
                s = c["step"]
                if s in seen:
                    continue
                ok = c["tensors_sha256"] == ref[s]["tensors_sha256"]
                seen[s] = dict(tensors_sha256=c["tensors_sha256"], e28=ref[s]["tensors_sha256"], equal=ok, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                log.write_text(json.dumps(seen, indent=1))
                print("e32 ceiling", seed, s, "equal" if ok else "DIFFERENT", flush=True)
                if not ok:
                    subprocess.run(["pkill", "-f", f"exp28_everybyte.py dev B-all {seed}"])
                    raise SystemExit(f"ceiling {seed} row {s} differs from E28's: stopped")
            if r.get("complete"):
                return
        time.sleep(20)


if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])
    outdir = Path(a[a.index("--out") + 1]) if "--out" in a else OUT
    if a and a[0] == "dev":
        dev(a[1], int(a[2]), outdir, cmd)
    elif a and a[0] == "check":
        checks(outdir)
    elif a and a[0] == "calib":
        calib(outdir)
    elif a and a[0] == "watch-ceiling":
        watch_ceiling(int(a[1]), outdir)
    else:
        raise SystemExit(__doc__)
