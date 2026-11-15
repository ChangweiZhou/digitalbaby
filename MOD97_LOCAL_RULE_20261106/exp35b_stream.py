"""E35-B stage 1a (brief OPUS_TASK_E35B_INNATE_CIRCUITS_INVENTORY_YI_20261115.md, revision 2): E26's GRU (arch26) plus inventory 乙's
fixed module, born from the byte stream with no parsing supplied, at taught fraction f.

Arms: G (the GRU alone: arch26.forward's lines, nothing added), M (plus the module), X (plus the scrambled module), F (M with the direct
read-out's logits for bytes '0'..'9' removed at every step). The loop is E26's birth loop (exp26_candidate.dev, lines 120-170) written anew
with the taught fraction, the arm's network and the brief's two panels; exp26_candidate.dev and .step are NOT called or patched.

    python3 exp35b_stream.py dev <G|M|X|F> <frac> <seed> [--out DIR]     # dose: E11_DEV_UPDATES (full 24,576 when unset)
    python3 exp35b_stream.py check <1|2|3|4|5|all> [--out DIR]

The module's wiring (the brief's code):
    c = content(Hs); g = sigmoid(wg Hs); o = softmax(wo Hs); q = sigmoid(wq Hs)
    r = module_path(c, g, o)            (exp35b_parsed.module_path: registers from the neutral code, the four fixed operations)
    d = dec(r); base = l3(z2) (digits zeroed for F); logits = base + q0 d_tens + q1 d_units
New parameters (wa, ws, wg, wo, wq, wd1, wd2) are created after arch26.new_net(seed) under T.manual_seed(seed + 35), with torch's defaults.
"""
import hashlib
import inspect
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C          # noqa: E402  (imports exp20_reference: environment gate; not edited; dev/step not called)

import torch as T                    # noqa: E402
from torch import nn                 # noqa: E402

R, X16, E, X = C.R, C.X16, C.E, C.X
import arch26                        # noqa: E402
import exp28_everybyte as E28        # noqa: E402  (roles_of only)
import exp35b_parsed as P            # noqa: E402  (the fixed primitives, the scrambled maps, module_path)
from exp3_state import lin           # noqa: E402

LR, WD = C.LR, C.WD
assert LR == .001 and WD == .25
ARMS = ("G", "M", "X", "F")
SOURCES = C.SOURCES + ("exp28_everybyte.py", "exp35b_parsed.py", "exp35b_stream.py")
KS_HELD = (0, 4, 8, 12, 20, 40)
ROWS = (16384, 20480, 24576)
OUT = X.OUT / "e35b"
NEW_PARAMS = ("wa", "ws", "wg", "wo", "wq", "wd1", "wd2")
DIGITS = slice(48, 58)


# ---------------------------------------------------------------- the network
def new_net(seed, arm):
    m = arch26.new_net(seed)
    m.e35_arm = arm
    m.e35_off = False
    if arm == "G":
        return m
    T.manual_seed(seed + 35)
    m.wa, m.ws = nn.Linear(256, 8), nn.Linear(256, 1)            # content
    m.wg = nn.Linear(256, 2)
    m.wo = nn.Linear(256, 4)
    m.wq = nn.Linear(256, 2)
    m.wd1, m.wd2 = nn.Linear(16, 128), nn.Linear(128, 514)       # dec
    return m


def content(m, Hs):
    ang = m.wa(Hs)
    return P.Code(P=T.stack([ang.cos(), ang.sin()], -1), s=m.ws(Hs).squeeze(-1))


def forward(m, x, local, through_time, keep=False, silence=False):
    """arch26.forward's lines (local=False, credit through time), then the module terms for arms M, X and F."""
    assert local is False and through_time is True, "E35-B runs local=False, credit through the whole item"
    assert getattr(m, "kind", None) == "gru"
    u = m.book(x)
    z1 = T.tanh(lin(u, m.l1, local))
    B, L = x.shape
    gi = lin(z1, arch26.Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0), local)
    hh = arch26.Mat(m.rnn.weight_hh_l0, m.rnn.bias_hh_l0)
    h = T.zeros(B, m.units)
    hs = []
    for t in range(L):
        hp = h if through_time else h.detach()
        gh = lin(hp, hh, local)
        i_r, i_z, i_n = gi[:, t].chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        r = T.sigmoid(i_r + h_r)
        z = T.sigmoid(i_z + h_z)
        nn_ = T.tanh(i_n + r * h_n)
        h = (1 - z) * nn_ + z * hp
        if keep and h.requires_grad:
            h.retain_grad()
        hs.append(h)
    Hs = T.stack(hs, 1)
    if silence:
        Hs = T.zeros_like(Hs)
    z2 = T.tanh(lin(z1, m.l2, local) + lin(Hs, m.lm, local))
    base = lin(z2, m.l3, local)
    arm = getattr(m, "e35_arm", "G")
    if arm == "G":
        return base, dict(z1=z1, h=hs)
    if arm == "F":
        base = T.cat([base[..., :48], T.zeros_like(base[..., DIGITS]), base[..., 58:]], -1)
    if m.e35_off:
        return base, dict(z1=z1, h=hs)
    c = content(m, Hs)
    g = T.sigmoid(m.wg(Hs))
    o = T.softmax(m.wo(Hs), -1)
    q = T.sigmoid(m.wq(Hs))
    rr = P.module_path(c, g, o, P.OP_SCR if arm == "X" else P.OP)
    d = m.wd2(T.tanh(m.wd1(rr.reshape(B, L, 16)))).reshape(B, L, 2, 257)
    logits = base + q[..., 0:1] * d[:, :, 0] + q[..., 1:2] * d[:, :, 1]
    return logits, dict(z1=z1, h=hs, g=g.detach(), o=o.detach(), q=q.detach())


@contextmanager
def module_off(m):
    old = m.e35_off
    m.e35_off = True
    try:
        yield
    finally:
        m.e35_off = old


def install():
    arch26.install()
    E.forward = forward


def opt_record(m, opt):
    return dict(name="AdamW", lr=LR, weight_decay=WD, betas=list(C.BETAS), groups=R.opt_groups(m, opt))


def make_opt(m):
    opt = X16.adamw16(m, LR, WD, ())
    g = R.opt_groups(m, opt)
    rest = [n for n, _ in m.named_parameters() if not n.startswith("book.")]
    assert g == [dict(weight_decay=WD, lr=LR, betas=list(C.BETAS), params=rest), dict(weight_decay=0.0, lr=LR, betas=list(C.BETAS), params=["book.weight"])], g
    return opt


# ---------------------------------------------------------------- data and panels
def frac_tag(f):
    return f"f{round(f * 100)}"


def taught_and_panels(f, seed, e26_panels=False):
    tp = [(t, E.split(t, f, seed)[0]) for t in E.DEV]
    if e26_panels:          # check 1: E26's own panels
        held = {E.NAME(*t): E.panel(t, E.split(t, f, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV}
        return tp, held, None
    held = {E.NAME(*t): E.panel(t, E.split(t, f, seed)[1], seed, ks=KS_HELD, n=300) for t in E.DEV}
    taught = {E.NAME(*t): E.panel(t, E.split(t, f, seed)[0], seed + 1, ks=(12,), n=300) for t in E.DEV}
    return tp, held, taught


def exposure_check(tp, held, taught):
    """A held-out panel item's operand pair is never a taught pair of that task (each task has its own split)."""
    out = {}
    for (t, pairs), n in zip(tp, [E.NAME(*t) for t in E.DEV]):
        ps = set(pairs)
        hp = {(int(bytes(p[1:3])), int(bytes(p[-3:-1]))) for k, it in held[n].items() for p, _ in it}
        tpp = {(int(bytes(p[1:3])), int(bytes(p[-3:-1]))) for k, it in taught[n].items() for p, _ in it}
        out[n] = dict(held_panel_pairs_in_taught=len(hp & ps), taught_panel_pairs_in_taught=len(tpp & ps), taught_panel_pairs=len(tpp))
        assert not (hp & ps) and tpp <= ps
    return out


# ---------------------------------------------------------------- diagnostics: gates by the role of the byte just read
IN_ROLES = ("bos",) + E28.FINE


def diagnostics(items, y, aux):
    fine = E28.roles_of(items, y)                                   # role of the target byte at each position
    inr = T.full_like(fine, -1)
    inr[:, 0] = 0                                                   # input at position 0 is BOS
    inr[:, 1:] = T.where(fine[:, :-1] >= 0, fine[:, :-1] + 1, -1)  # input at j is the target byte of j-1
    out = {}
    for i, name in enumerate(IN_ROLES):
        sel = inr == i
        if not bool(sel.any()):
            continue
        out[name] = dict(n=int(sel.sum()), gA=float(aux["g"][..., 0][sel].mean()), gB=float(aux["g"][..., 1][sel].mean()),
                         q_tens=float(aux["q"][..., 0][sel].mean()), q_units=float(aux["q"][..., 1][sel].mean()),
                         op_weights=[float(v) for v in aux["o"][sel].mean(0)])
    eq = inr == IN_ROLES.index("equals")
    out["at_equals_max_op_weight"] = float(aux["o"][eq].max(-1).values.mean())
    out["at_equals_argmax_op_count"] = [int(v) for v in T.bincount(aux["o"][eq].argmax(-1), minlength=4)]
    return out


# ---------------------------------------------------------------- the update (E26's step, the arm's forward)
def step(m, opt, items):
    x, y, mask = X.batch_tensor(items)
    logits, aux = forward(m, x, False, True)
    loss = E.cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach()), aux, y


def provenance(cmd):
    return dict(command=cmd, environment_at_start=R.ENV_AT_START, environment_set_by_runner=R.REQUIRED_ENV, host=R.host(),
                git=C.git_state(), sources_sha256={s: X16.sha_file(HERE / s) for s in SOURCES})


def describe(m):
    d = arch26.describe(m)
    d["arm"] = m.e35_arm
    d["new_params"] = {n: sum(p.numel() for p in getattr(m, n).parameters()) for n in NEW_PARAMS if hasattr(m, n)}
    d["forward"] = "exp35b_stream.forward (arch26.forward's lines + inventory 乙 module)"
    return d


def dev(arm, f, seed, out, cmd, e26_panels=False):
    assert arm in ARMS
    install()
    tag = f"e35b_dev_{arm}{E.DEVTAG}_{frac_tag(f)}_{seed}"
    rec_path = Path(out) / f"{tag}.json"
    R.refuse_existing(rec_path)
    Path(out).mkdir(parents=True, exist_ok=True)
    m = new_net(seed, arm)
    desc = describe(m)
    book0 = m.book.weight.detach().clone()
    opt = make_opt(m)
    census = None
    tp, held, taught = taught_and_panels(f, seed, e26_panels)
    expo = exposure_check(tp, held, taught) if taught else None
    bt = E.batches(tp, seed, E.KMAX["M"])
    dg = C.Digest()
    rec = dict(experiment="E35B", kind="birth", arm=arm, fraction=f, local=False, net="G1+inventory_yi" if arm != "G" else "G1", seed=seed,
               kmax=E.KMAX["M"], dev_updates=E.DEV_DOSE, block=X.BLOCK, batch=X.BATCH, optimizer=opt_record(m, opt),
               update="exp35b_stream.step: forward (the arm's network), answer-masked cross entropy, zero_grad, backward, AdamW step",
               protocol="exp26_candidate.dev's birth loop with taught fraction f; held panel ks (0,4,8,12,20,40) n=300; taught panel seed+1 k=12",
               e26_panels=e26_panels, model=desc, scrambled=P.SCR_INFO if arm == "X" else None, exposure=expo,
               installed=dict(E_forward=f"{E.forward.__module__}.{E.forward.__name__}"), provenance=provenance(cmd), complete=False,
               curve=[], diagnostics=[], saved_rows={})
    t0 = time.process_time(); cpu_score = 0.0; curve = rec["curve"]; row = None
    for stop in range(X.BLOCK, E.DEV_DOSE + 1, X.BLOCK):
        losses = []
        for i in range(X.BLOCK):
            items = next(bt); dg.add(items)
            lv, aux, y = step(m, opt, items)
            losses.append(lv)
            if census is None:
                census = X16.zero_grad_report(m, X.batch_tensor(items)[0]); print("e35b dev credit census:", census["by_role"], flush=True)
            if i == X.BLOCK - 1 and arm != "G":
                rec["diagnostics"].append(dict(step=stop, **diagnostics(items, y, aux)))
        if stop % 4096 == 0 or stop == E.DEV_DOSE or stop in ROWS:
            cb = time.process_time()
            sealed = {n: {k: E.score(m, it) for k, it in pk.items()} for n, pk in held.items()}
            row = dict(step=stop, loss=sum(losses) / len(losses), finite_losses=all(math.isfinite(v) for v in losses), sealed=sealed)
            if stop in ROWS and taught is not None:
                row["taught_k12"] = {n: E.score(m, pk[12]) for n, pk in taught.items()}
                if arm != "G":
                    with module_off(m):
                        row["module_off"] = {n: {k: E.score(m, pk[k]) for k in (12, 40)} for n, pk in held.items()}
                wp = Path(out) / f"{tag}_row{stop}.pt"
                rec["saved_rows"][str(stop)] = dict(file=wp.name, sha256=R.save_weights(m, wp), tensors_sha256=R.tensors_sha(m.state_dict()))
            cpu_score += time.process_time() - cb
            row.update(cpu=time.process_time() - t0, cpu_scoring_so_far=cpu_score, cpu_updates_so_far=time.process_time() - t0 - cpu_score,
                       book_shift_by_role=X16.book_shift(m, book0), data_digest=dg.value())
            curve.append(row)
            print("e35b dev", arm, f, seed, stop, {n: [v[k]["exact"] for k in v] for n, v in row["sealed"].items()},
                  ("taught", {n: v["exact"] for n, v in row["taught_k12"].items()}) if "taught_k12" in row else "",
                  ("off", {n: [v[k]["exact"] for k in v] for n, v in row["module_off"].items()}) if "module_off" in row else "",
                  f"loss {row['loss']:.4f} cpu {row['cpu']:.0f}s", flush=True)
            rec_path.write_text(json.dumps(dict(rec, credit_census_first_update=census), indent=1))
    assert row is not None
    cb = time.process_time()
    row["silenced_k12"] = {n: E.score(m, pk[12], silence=True) for n, pk in held.items()}
    cpu_score += time.process_time() - cb
    rec.update(credit_census_first_update=census, optimizer_steps=R.opt_steps(opt), optimizer_read_back=opt_record(m, opt), data_digest=dg.value(),
               tensors_sha256=R.tensors_sha(m.state_dict()), cpu_total=time.process_time() - t0, cpu_scoring=cpu_score,
               cpu_updates=time.process_time() - t0 - cpu_score, complete=True)
    rec_path.write_text(json.dumps(rec, indent=1))
    print("e35b dev", arm, f, seed, "complete", rec["tensors_sha256"][:16], flush=True)
    return rec


# ================================================================ checks
FREQS = (1, 2, 3, 5, 8, 13, 21, 34)


def exact_codes(mod=89):
    v = T.arange(mod, dtype=T.float64)
    ang = 2 * math.pi * T.tensor(FREQS, dtype=T.float64)[None, :] * v[:, None] / mod
    return P.Code(P=T.stack([ang.cos(), ang.sin()], -1), s=10 * v)


def decode(Pout, codes):
    d = ((Pout[:, None] - codes.P[None]) ** 2).sum((-1, -2))
    return d.argmin(1)


TRUTH = [lambda a, b: (a + b) % 89, lambda a, b: (a - b) % 89, T.maximum, T.minimum]


def check2():
    codes = exact_codes()
    a = T.arange(89).repeat_interleave(89); b = T.arange(89).repeat(89)
    A, Bq = P.Code(codes.P[a], codes.s[a]), P.Code(codes.P[b], codes.s[b])
    ops = {}
    for j, name in enumerate(P.OPNAMES):
        ops[name] = int((decode(P.OP[j](A, Bq), codes) == TRUTH[j](a, b)).sum())
    g = T.Generator().manual_seed(2)
    ra, rb = P.random_unit_codes(10000, g).double(), P.random_unit_codes(10000, g).double()
    err = float((P.mul(ra, P.div(rb, ra)) - rb).abs().max())
    ra32, rb32 = ra.float(), rb.float()
    err32 = float((P.mul(ra32, P.div(rb32, ra32)) - rb32).abs().max())
    out = dict(frequencies=FREQS, scalar="10 v", decoded_correct_of_7921=ops, mul_div_inverse_max_abs_err_float64=err,
               mul_div_inverse_max_abs_err_float32=err32, scrambled=P.SCR_INFO)
    out["pass"] = all(v == 7921 for v in ops.values()) and err32 < 1e-5
    # 2b: the whole register / operation / result path with ideal interfaces, no training
    res = {}
    for gap in (0, 12, 40):
        t1, t2 = 2, 2 + 1 + gap
        L = t2 + 3
        N = a.numel()
        gen = T.Generator().manual_seed(100 + gap)
        junk = P.random_unit_codes(N * L, gen).reshape(N, L, 8, 2).double()
        cP = junk.clone(); cs = T.rand(N, L, generator=gen, dtype=T.float64) * 900
        cP[:, t1] = codes.P[a]; cs[:, t1] = codes.s[a]
        cP[:, t2] = codes.P[b]; cs[:, t2] = codes.s[b]
        gt = T.zeros(N, L, 2, dtype=T.float64); gt[:, t1, 0] = 1; gt[:, t2, 1] = 1
        for j, name in enumerate(P.OPNAMES):
            o = T.zeros(N, L, 4, dtype=T.float64); o[..., j] = 1
            r = P.module_path(P.Code(cP, cs), gt, o)
            res[f"{name}_gap{gap}"] = int((decode(r[:, -1], codes) == TRUTH[j](a, b)).sum())
    out["check_2b_decoded_correct_of_7921"] = res
    out["check_2b_pass"] = all(v == 7921 for v in res.values())
    return out


def check1(out_dir):
    """A dry birth of arm G at f = 0.5 with E26's panels, 2,048 updates: tensors_sha256 must equal E26's arm B dry receipt."""
    assert E.DEV_DOSE == 2048, "run check 1 with E11_DEV_UPDATES=2048"
    ref = json.loads((X.OUT / "e26_qual" / "e26_dev_B_dry2048_73129.json").read_text())["tensors_sha256"]
    rec = dev("G", 0.5, 73129, out_dir, "check 1", e26_panels=True)
    return dict(reference="exp45/e26_qual/e26_dev_B_dry2048_73129.json", reference_tensors_sha256=ref, ours=rec["tensors_sha256"],
                equal=rec["tensors_sha256"] == ref, cpu_total=rec["cpu_total"])


def check345():
    install()
    out = {}
    seed = 73129
    g = new_net(seed, "G"); gsd = g.state_dict()
    for arm in ("M", "X", "F"):
        m = new_net(seed, arm)
        sd = m.state_dict()
        out[f"check3_{arm}_shared_equal"] = all(T.equal(sd[k], gsd[k]) for k in gsd)
        out[f"check3_{arm}_extra_params"] = sorted(set(sd) - set(gsd))
    mx = new_net(seed, "M"); xx = new_net(seed, "X")
    out["check3_M_X_all_equal"] = all(T.equal(mx.state_dict()[k], xx.state_dict()[k]) for k in mx.state_dict())
    # check 4: module_off logits of M equal a G holding the same shared parameters (untrained, and after random perturbation of all params)
    tp, held, _ = taught_and_panels(0.25, seed)
    items = held["add89"][12][:32] + held["max89"][40][:32]
    x, y, mask = X.batch_tensor(items)
    res4 = {}
    for name, perturb in (("initial", False), ("perturbed", True)):
        m = new_net(seed, "M")
        if perturb:
            gen = T.Generator().manual_seed(4)
            with T.no_grad():
                for p in m.parameters():
                    p.add_(T.randn(p.shape, generator=gen) * 0.05)
        gg = new_net(seed, "G"); gg.load_state_dict({k: v for k, v in m.state_dict().items() if k in gg.state_dict()})
        with T.no_grad(), module_off(m):
            lm_ = forward(m, x, False, True)[0]
        with T.no_grad():
            lg_ = forward(gg, x, False, True)[0]
            lon = forward(m, x, False, True)[0]
        res4[name] = dict(equal=bool(T.equal(lm_, lg_)), max_abs=float((lm_ - lg_).abs().max()), module_on_differs=float((lon - lg_).abs().max()))
    out["check4"] = res4
    # check 5: no parsing leaks
    sig = list(inspect.signature(forward).parameters)
    out["check5_forward_signature"] = sig
    out["check5_pass"] = sig == ["m", "x", "local", "through_time", "keep", "silence"] and x.dtype == T.long
    src = inspect.getsource(forward)
    out["check5_forward_mentions"] = {w: (w in src) for w in ("mask", "roles_of", "position", "arange")}
    out["check5_pass"] = out["check5_pass"] and not any(out["check5_forward_mentions"].values())
    # params and the path's numerical identity of arch26.forward for arm G
    with T.no_grad():
        a26 = arch26.forward(g, x, False, True)[0]; o26 = forward(g, x, False, True)[0]
    out["G_forward_equals_arch26_forward"] = bool(T.equal(a26, o26))
    out["params"] = {arm: sum(p.numel() for p in new_net(seed, arm).parameters()) for arm in ("G", "M")}
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    out = Path(args[args.index("--out") + 1]) if "--out" in args else OUT
    cmd = " ".join(["python3", "exp35b_stream.py", *args])
    if args[0] == "dev":
        dev(args[1], float(args[2]), int(args[3]), out, cmd)
    elif args[0] == "check":
        part = args[1]
        out.mkdir(parents=True, exist_ok=True)
        if part == "1":
            res = check1(out)
        elif part == "2":
            res = check2()
        elif part == "345":
            res = check345()
        res = dict(check=part, result=res, provenance=provenance(cmd))
        (out / f"E35B_CHECK_{part}.json").write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps(res["result"], indent=1, default=str))
