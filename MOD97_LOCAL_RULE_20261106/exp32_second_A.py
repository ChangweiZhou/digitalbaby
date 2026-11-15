"""E32 check 7, the second worker's independent implementation (brief OPUS_TASK_E32_EVERY_BYTE_BIRTH_SWITCHES_20261114.md, revision 4).

Written from the brief alone, without reading the main worker's exp32_births.py, and without importing E31's files (arch31.py,
exp31_switch.py): the switched forward is written again here. Imports only the unedited harness (exp26_candidate, arch26, exp3_state, ...).

    python3 exp32_second_A.py dry <floor|G|H|K> <seed>      a birth at E11_DEV_UPDATES (2,048 for check 7) of E28's S-all with one switch;
                                                            writes exp45/e32/second/e32a_dev_<arm>_dry<dose>_<seed>.json (+ .pt)
    python3 exp32_second_A.py lam                           P4's strengths for the nine seed-and-strength pairs, no training;
                                                            writes exp45/e32/second/E32_LAMBDA_SECOND_A.json

The birth loop is E28's S-all (exp28_everybyte.dev with arm S-all), written again: arch26.install(); E.new_net(seed); exp26_candidate.make_opt;
batches E.batches([(t, E.split(t, .5, seed)[0]) for t in E.DEV], seed, E.KMAX["M"]); loss = cross_entropy(logits[y != BOS], y[y != BOS]);
zero_grad, backward, AdamW step. With no switch ("floor") it must give exp45/e28_qual/dry/e28_dev_S-all_dry2048_73129.json's tensors_sha256.

Switches, as the brief pins them (S = sign(W) * W.abs().mean() over the whole matrix; weight and bias gradients the floor's, i.e.
exp3_state.SignLinear's expressions g^T x and sum g):
    G   weight_ih_l0 and weight_hh_l0: gx = g[..., :512] @ W[:512] + g[..., 512:] @ S[512:]   (two products added)
    H   weight_hh_l0: gx = g @ W
    K   weight_hh_l0: gx = g @ Bhh; Bhh0 = torch.empty(768, 256).uniform_(-1/16, 1/16, generator=torch.Generator().manual_seed(seed + 31));
        after each step Bhh = (1 - 0.00025) * Bhh + (W1 - (1 - 0.00025) * W0)

P4's lam (brief, arms P4): at the frozen row-4,096 weights of E28's S-all birth (exp45/e28/e28_dev_S-all_<seed>_row4096.pt, its hash checked
against the receipt), over batches 4,097 to 4,224 of the birth stream (the stream advanced past 4,096 batches; its digest checked against
E28's row 4,096): the mean over the 128 batches of the task gradient norm as the sweep computes it (arch26.forward(m, x, True, True), every-byte
loss), and the mean of the term's gradient norm at lam = 1, each one joint norm over rows 256-511 of weight_ih_l0, weight_hh_l0, bias_ih_l0,
bias_hh_l0; lam = strength * task / term for strengths 0.25, 0.05, 0.01. The term exactly as the brief writes it:
    msk = (x != BOS) & (y != BOS); a = lin(z1.detach(), ih, False)[..., 256:512] + lin(hp.detach(), hh, False)[..., 256:512]
    term = (relu(4.59512 - a.abs()) ** 2)[msk].sum() / (msk.sum() * 256)
with hp[:, t] the state step t reads (zeros at t = 0, else the state after step t - 1).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C          # noqa: E402  (exp20_reference's environment gate; nothing edited)
import torch as T                    # noqa: E402
from torch.nn.functional import cross_entropy, linear  # noqa: E402

R, X16, E, X = C.R, C.X16, C.E, C.X
import arch26                        # noqa: E402
from exp3_state import lin           # noqa: E402

OUT = HERE / "exp45" / "e32" / "second"
E28 = HERE / "exp45" / "e28"
SEEDS = (73129, 93683, 116089)
STRENGTHS = {"P4a": 0.25, "P4b": 0.05, "P4c": 0.01}
CLOGIT = 4.59512
DECAY = 0.00025
ZR = slice(256, 512)
ROWS_OF = ("rnn.weight_ih_l0", "rnn.weight_hh_l0", "rnn.bias_ih_l0", "rnn.bias_hh_l0")
REFERENCE_4097 = {73129: (0.036740, 0.890970), 93683: (0.077113, 0.836491), 116089: (0.093320, 0.849241)}   # the reviewer's numpy reader


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Switched(T.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, mode, B):
        ctx.save_for_backward(x, weight, B if B is not None else weight)
        ctx.mode = mode
        return linear(x, weight, bias)

    @staticmethod
    def backward(ctx, g):
        x, W, B = ctx.saved_tensors
        if ctx.mode == "G":
            S = W.sign() * W.abs().mean()
            gx = g[..., :512] @ W[:512] + g[..., 512:] @ S[512:]
        elif ctx.mode == "H":
            gx = g @ W
        else:
            gx = g @ B
        g2 = g.reshape(-1, g.shape[-1])
        return gx, g2.T @ x.reshape(-1, x.shape[-1]), g2.sum(0), None, None


def mapped(x, weight, bias, mode, B=None):
    if mode is None:
        return lin(x, arch26.Mat(weight, bias), True)
    return Switched.apply(x, weight, bias, mode, B)


def fwd(m, x, arm, Bhh=None):
    """arch26.forward(m, x, True, True) with the arm's switch; returns logits, z1 and the states each step read (hp)."""
    u = m.book(x)
    z1 = T.tanh(lin(u, m.l1, True))
    Bn, L = x.shape
    gi = mapped(z1, m.rnn.weight_ih_l0, m.rnn.bias_ih_l0, "G" if arm == "G" else None)
    hmode = arm if arm in ("G", "H", "K") else None
    h = T.zeros(Bn, m.units); hs, hps = [], []
    for t in range(L):
        hp = h
        gh = mapped(hp, m.rnn.weight_hh_l0, m.rnn.bias_hh_l0, hmode, Bhh)
        i_r, i_z, i_n = gi[:, t].chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        r = T.sigmoid(i_r + h_r); z = T.sigmoid(i_z + h_z); n_ = T.tanh(i_n + r * h_n)
        h = (1 - z) * n_ + z * hp
        hs.append(h); hps.append(hp)
    z2 = T.tanh(lin(z1, m.l2, True) + lin(T.stack(hs, 1), m.lm, True))
    return lin(z2, m.l3, True), z1, T.stack(hps, 1)


def birth_stream(seed):
    return E.batches([(t, E.split(t, .5, seed)[0]) for t in E.DEV], seed, E.KMAX["M"])


def dry(arm, seed):
    assert arm in ("floor", "G", "H", "K")
    arch26.install()
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"e32a_dev_{arm}{E.DEVTAG}_{seed}"
    R.refuse_existing(stem.with_suffix(".json"), stem.with_suffix(".pt"))
    m = E.new_net(seed); opt = C.make_opt(m); bt = birth_stream(seed); dg = C.Digest()
    Bhh = T.empty(768, 256).uniform_(-1 / 16, 1 / 16, generator=T.Generator().manual_seed(seed + 31)) if arm == "K" else None
    ident = {}; t0 = time.process_time(); first_loss = None
    for n in range(1, E.DEV_DOSE + 1):
        items = next(bt); dg.add(items)
        x, y, _ = X.batch_tensor(items); real = y != E.BOS
        logits = fwd(m, x, arm, Bhh)[0]
        loss = cross_entropy(logits[real], y[real])
        if first_loss is None:
            first_loss = float(loss.detach())
        opt.zero_grad(); loss.backward()
        if arm == "K":
            W0 = m.rnn.weight_hh_l0.detach().clone(); B0 = Bhh.clone()
        opt.step()
        if arm == "K":
            W1 = m.rnn.weight_hh_l0.detach().clone()
            Bhh = (1 - DECAY) * Bhh + (W1 - (1 - DECAY) * W0)
            if n in (1, 10, 100):
                ident[str(n)] = float(((Bhh - W1) - (1 - DECAY) * (B0 - W0)).abs().max())
    wsha = R.save_weights(m, stem.with_suffix(".pt"))
    rec = dict(experiment="E32", role="check 7, second worker (A)", arm=arm, seed=seed, dev_updates=E.DEV_DOSE, first_batch_loss=first_loss,
               tensors_sha256=R.tensors_sha(m.state_dict()), weights_sha256=wsha, data_digest=dg.value(), cpu_seconds=time.process_time() - t0,
               K_identity_max_abs=ident or None,
               K_final=(dict(cosine=float((Bhh * m.rnn.weight_hh_l0.detach()).sum() / (Bhh.norm() * m.rnn.weight_hh_l0.detach().norm())),
                             rel_diff=float((Bhh - m.rnn.weight_hh_l0.detach()).norm() / m.rnn.weight_hh_l0.detach().norm())) if arm == "K" else None),
               script_sha256=sha(__file__), torch=T.__version__, threads=T.get_num_threads())
    stem.with_suffix(".json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


def term_of(m, x, y, z1, hp):
    msk = (x != X.BOS) & (y != X.BOS)
    ih, hh = arch26.Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0), arch26.Mat(m.rnn.weight_hh_l0, m.rnn.bias_hh_l0)
    a = lin(z1.detach(), ih, False)[..., 256:512] + lin(hp.detach(), hh, False)[..., 256:512]
    return (T.relu(CLOGIT - a.abs()) ** 2)[msk].sum() / (msk.sum() * 256)


def rows_norm(m, loss):
    ps = [dict(m.named_parameters())[n] for n in ROWS_OF]
    gs = T.autograd.grad(loss, ps, allow_unused=True)
    return float(T.cat([(g if g is not None else T.zeros_like(p))[ZR].flatten() for g, p in zip(gs, ps)]).norm())


def lam():
    arch26.install()
    out = dict(role="check 7, second worker (A)", script_sha256=sha(__file__), torch=T.__version__, seeds={})
    for seed in SEEDS:
        f = E28 / f"e28_dev_S-all_{seed}_row4096.pt"; rec = json.loads((E28 / f"e28_dev_S-all_{seed}.json").read_text())
        assert rec["row_weights"]["4096"]["sha256"] == sha(f), f"{f.name}: not the receipt's row file"
        m = E.new_net(seed); m.load_state_dict(T.load(f))
        row = next(r for r in rec["curve"] if r["step"] == 4096)
        assert R.tensors_sha(m.state_dict()) == row["tensors_sha256"]
        bt = birth_stream(seed); dg = C.Digest()
        for _ in range(4096):
            dg.add(next(bt))
        assert dg.value() == row["data_digest"], f"stream after 4,096 batches is not E28's for seed {seed}"
        tasks, terms = [], []
        for _ in range(128):
            items = next(bt)
            x, y, _ = X.batch_tensor(items); real = y != E.BOS
            logits, z1, hp = fwd(m, x, "floor")
            tasks.append(rows_norm(m, cross_entropy(logits[real], y[real])))
            terms.append(rows_norm(m, term_of(m, x, y, z1, hp)))
        mt, mm = sum(tasks) / 128, sum(terms) / 128
        ref = REFERENCE_4097[seed]
        out["seeds"][str(seed)] = dict(row4096_weights_sha256=sha(f), row4096_tensors_sha256=row["tensors_sha256"], data_digest_after_4096=dg.value(),
                                       mean_task_norm=mt, mean_term_norm_at_lam_1=mm, task_norm_spread=[min(tasks), max(tasks)],
                                       lam={k: s * mt / mm for k, s in STRENGTHS.items()},
                                       batch_4097=dict(task_norm=tasks[0], term_norm=terms[0], reviewer=dict(task_norm=ref[0], term_norm=ref[1]),
                                                       rel_diff=[abs(tasks[0] - ref[0]) / ref[0], abs(terms[0] - ref[1]) / ref[1]]))
        print(seed, out["seeds"][str(seed)]["lam"], out["seeds"][str(seed)]["batch_4097"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E32_LAMBDA_SECOND_A.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    if a[0] == "dry":
        dry(a[1], int(a[2]))
    elif a[0] == "lam":
        lam()
    else:
        raise SystemExit(__doc__)
