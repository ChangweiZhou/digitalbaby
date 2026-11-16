"""E35-A stage 1a (brief OPUS_TASK_E35A_INNATE_CIRCUITS_INVENTORY_JIA_20261115.md, revision 2): inventory 甲 from the byte stream.

The controller is E26's GRU (arch26.new_net, arch26.forward, local=False throughout). Arms:

    G  the GRU alone: arch26.forward's logits unchanged (E26's arm B)
    M  the GRU plus the fixed module (the four mod-89 operations on distributions over 89 positions, exp10_ring's with MOD = 89)
    X  the GRU plus the scrambled module (the same wiring and learned parameters; each operation a fixed random table, manual_seed(35))
    F  arm M with l3's logits for the bytes '0'..'9' removed at every step (run only after outcome row 3)

The module's wiring is the brief's, as code (module_path is the lines from `A, Bq = neutral` to `r = T.stack(r, 1)`; the operations are
applied to the flattened (B*L, 89) register stacks, in row chunks, which the brief allows). No parsing is supplied: every gate is computed
from the GRU's own states, and the forward takes the byte tensor only (check 5).

The birth loop is exp26_candidate.dev's (lines 120-170) written out again here with the taught fraction, the arm's network and the brief's
two panels; exp26_candidate.dev and .step are not called or patched. Optimizer: exp16_integrated.adamw16(m, LR, WD, ()) with E26's
LR, betas and WD = 0.25 (the new parameters join the decayed group; the groups are read back and recorded).

    python3 exp35a_stream.py dev <G|M|X|F> <10|25|50> <seed> [--e26-panels] [--out DIR]
    python3 exp35a_stream.py check <ops|path|start|ablation|noparse> [--qual DIR]

Dose: E11_DEV_UPDATES (full dose 24,576 when unset), as E26.
"""
import ast
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C          # noqa: E402  (imports exp20_reference: environment gate; constants, Digest, provenance helpers only)
import exp28_everybyte as EB         # noqa: E402  (roles_of, FINE)

import torch as T                    # noqa: E402

R, X16, E, X, arch26 = C.R, C.X16, C.E, C.X, C.arch26
from exp3_state import lin           # noqa: E402

LR, BETAS, WD = C.LR, C.BETAS, C.WD
assert WD == 0.25 and LR == 0.001
SOURCES = C.SOURCES + ("exp28_everybyte.py", "exp35a_stream.py")
OUT = X.OUT / "e35a"
SAVE_ROWS = (16384, 20480, 24576)
KS_HELD = (0, 4, 8, 12, 20, 40)
CHUNK = 4096

# ---------------------------------------------------------------- inventory 甲: the fixed operations (the brief's lines)
IDX = T.arange(89)
def op_add(a, b): return T.einsum("bi,bki->bk", a, b[:, (IDX[:, None] - IDX[None, :]) % 89])
def op_sub(a, b): return T.einsum("bi,bki->bk", a, b[:, (IDX[None, :] - IDX[:, None]) % 89])
def op_max(a, b): F = a.cumsum(1) * b.cumsum(1); return T.cat([F[:, :1], F[:, 1:] - F[:, :-1]], 1)
def op_min(a, b): S = a.flip(1).cumsum(1).flip(1) * b.flip(1).cumsum(1).flip(1); return T.cat([S[:, :-1] - S[:, 1:], S[:, -1:]], 1)
OP = [op_add, op_sub, op_max, op_min]


def table_op(tab):
    flat = tab.reshape(-1)
    def f(a, b):
        outer = (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)
        return T.zeros(len(a), 89, dtype=a.dtype).index_add(1, flat, outer)
    return f


TABLES = T.randint(0, 89, (4, 89, 89), generator=T.Generator().manual_seed(35))
TABLES_SHA = hashlib.sha256(TABLES.contiguous().numpy().tobytes()).hexdigest()
OP_X = [table_op(TABLES[j]) for j in range(4)]
OPS_OF_ARM = {"M": OP, "F": OP, "X": OP_X}


def _rows(f, a, b):
    """f applied to (N, 89) rows in chunks of CHUNK rows (row-wise operations: the same values as one call)."""
    if len(a) <= CHUNK:
        return f(a, b)
    return T.cat([f(a[i:i + CHUNK], b[i:i + CHUNK]) for i in range(0, len(a), CHUNK)], 0)


def module_path(c, g, o, ops=OP):
    """The brief's lines from `A, Bq = neutral` to `r = T.stack(r, 1)`. c: (B, L, 89) codes; g: (B, L, 2) write gates; o: (B, L, 4)
    operation weights. Returns r: (B, L, 89). The registers are run step by step; the operations are applied to every step's (A, Bq)."""
    B, L, N = c.shape
    A = T.full((B, N), 1 / N, dtype=c.dtype)
    Bq = T.full((B, N), 1 / N, dtype=c.dtype)
    As, Bs = [], []
    for t in range(L):
        A = (1 - g[:, t, 0:1]) * A + g[:, t, 0:1] * c[:, t]
        Bq = (1 - g[:, t, 1:2]) * Bq + g[:, t, 1:2] * c[:, t]
        As.append(A); Bs.append(Bq)
    Af, Bf, of = T.stack(As, 1).reshape(B * L, N), T.stack(Bs, 1).reshape(B * L, N), o.reshape(B * L, 4)
    r = sum(of[:, j:j + 1] * _rows(ops[j], Af, Bf) for j in range(4))
    return r.reshape(B, L, N)


# ---------------------------------------------------------------- the network
def new_net(seed, arm):
    m = arch26.new_net(seed)
    m.e35_arm, m.module_off = arm, False
    if arm != "G":
        T.manual_seed(seed + 35)
        m.wc = T.nn.Linear(256, 89)
        m.wg = T.nn.Linear(256, 2)
        m.wo = T.nn.Linear(256, 4)
        m.wq = T.nn.Linear(256, 2)
        m.wd = T.nn.Linear(89, 514)
    return m


def forward(m, x, local, through_time, keep=False, silence=False):
    """exp11_cell's forward signature. x: the byte tensor (B, L); nothing else about the item is seen."""
    base, it = arch26.forward(m, x, local, through_time, keep, silence)
    arm = m.e35_arm
    if arm == "G":
        return base, it
    if arm == "F":
        base = T.cat([base[..., :48], T.zeros_like(base[..., 48:58]), base[..., 58:]], -1)
    if m.module_off:
        return base, it
    Hs = T.stack(it["h"], 1)
    if silence:
        Hs = T.zeros_like(Hs)
    B, L = x.shape
    c = T.softmax(lin(Hs, m.wc, local), -1)
    g = T.sigmoid(lin(Hs, m.wg, local))
    o = T.softmax(lin(Hs, m.wo, local), -1)
    q = T.sigmoid(lin(Hs, m.wq, local))
    r = module_path(c, g, o, OPS_OF_ARM[arm])
    d = lin(r, m.wd, local).reshape(B, L, 2, 257)
    logits = base + q[..., 0:1] * d[:, :, 0] + q[..., 1:2] * d[:, :, 1]
    it.update(g=g, o=o, q=q)
    return logits, it


def install():
    arch26.install()
    E.forward = forward


def describe(m):
    d = arch26.describe(m)
    d.update(arm=m.e35_arm, params=sum(p.numel() for p in m.parameters()), params_by_tensor={n: p.numel() for n, p in m.named_parameters()},
             forward=f"{forward.__module__}.{forward.__name__}", tables_sha256=TABLES_SHA if m.e35_arm == "X" else None)
    return d


def make_opt(m):
    opt = X16.adamw16(m, LR, WD, ())
    g = R.opt_groups(m, opt)
    rest = [n for n, _ in m.named_parameters() if not n.startswith("book.")]
    assert g == [dict(weight_decay=WD, lr=LR, betas=list(BETAS), params=rest), dict(weight_decay=0.0, lr=LR, betas=list(BETAS), params=["book.weight"])], g
    return opt


def opt_record(m, opt):
    return dict(name="AdamW", builder="exp16_integrated.adamw16(m, LR, WD, ())", lr=LR, weight_decay=WD, betas=list(BETAS), groups=R.opt_groups(m, opt))


def provenance(cmd):
    return dict(command=cmd, environment_at_start=R.ENV_AT_START, environment_set_by_runner=R.REQUIRED_ENV, host=R.host(), git=C.git_state(),
                sources_sha256={s: X16.sha_file(HERE / s) for s in SOURCES})


# ---------------------------------------------------------------- the update (E26's, written out)
def step(m, opt, items):
    x, y, mask = X.batch_tensor(items)
    logits, it = forward(m, x, False, True)
    loss = E.cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach()), it, y


@T.no_grad()
def gate_diag(items, it, y):
    """On the training batch, detached: mean of each write gate and of q at each fine role of the byte just read (the state at step t has
    read x[t] = the target of step t-1; step 0 reads BOS), and the mean largest operation weight at the step that reads '='."""
    fine_y = EB.roles_of(items, y)
    fine_x = T.full_like(fine_y, -1); fine_x[:, 1:] = fine_y[:, :-1]
    names = list(EB.FINE)
    out = {}
    for name, v in (("gA", it["g"][..., 0]), ("gB", it["g"][..., 1]), ("q_tens", it["q"][..., 0]), ("q_units", it["q"][..., 1])):
        out[name] = {}
        for j, rn in enumerate(names):
            sel = fine_x == j
            if bool(sel.any()):
                out[name][rn] = float(v[sel].mean())
    eq = fine_x == names.index("equals")
    omax = it["o"].max(-1)
    out["op_max_weight_at_equals"] = float(omax.values[eq].mean())
    out["op_argmax_hist_at_equals"] = T.bincount(omax.indices[eq], minlength=4).tolist()
    return out


# ---------------------------------------------------------------- birth
def score_all(m, pan, ks):
    return {n: {k: E.score(m, pk[k]) for k in ks} for n, pk in pan.items()}


def dev(arm, fpct, seed, out, cmd, e26_panels=False):
    install()
    frac = {10: .10, 25: .25, 50: .50}[fpct]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    stem = f"e35a_dev_{arm}_f{fpct}{'_e26panels' if e26_panels else ''}{E.DEVTAG}_{seed}"
    rec_path = out / f"{stem}.json"
    R.refuse_existing(rec_path, *(out / f"{stem}_row{r_}.pt" for r_ in SAVE_ROWS), out / f"{stem}.pt")
    m = new_net(seed, arm)
    desc = describe(m)
    book0 = m.book.weight.detach().clone()
    opt = make_opt(m)
    census = None
    splits = {E.NAME(*t): E.split(t, frac, seed) for t in E.DEV}
    tp = [(t, splits[E.NAME(*t)][0]) for t in E.DEV]
    if e26_panels:
        held = {E.NAME(*t): E.panel(t, splits[E.NAME(*t)][1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV}; ks_held = (0, 4, 8, 12, 20)
        taught = {}
    else:
        held = {E.NAME(*t): E.panel(t, splits[E.NAME(*t)][1], seed, ks=KS_HELD, n=300) for t in E.DEV}; ks_held = KS_HELD
        taught = {E.NAME(*t): E.panel(t, splits[E.NAME(*t)][0], seed + 1, ks=(12,), n=300) for t in E.DEV}
    # exposure: no held-out item's operand pair is in its task's taught set
    exposure = {n: len(set(splits[n][0]) & {(int(bytes(p[1:3])), int(bytes(p[-3:-1]))) for p, _ in held[n][12]}) for n in held}
    assert not any(exposure.values()), exposure
    bt = E.batches(tp, seed, E.KMAX["M"])
    dg = C.Digest()
    rec = dict(experiment="E35A", kind="birth", arm=arm, fraction=frac, local=False, net="G1" + ("" if arm == "G" else "+module_jia"), seed=seed,
               kmax=E.KMAX["M"], dev_updates=E.DEV_DOSE, block=X.BLOCK, batch=X.BATCH, optimizer=opt_record(m, opt),
               update="exp35a_stream.step: forward (local=False), answer-masked cross entropy, zero_grad, backward, AdamW step",
               protocol="exp26_candidate.dev's birth loop with the taught fraction, the arm's network and the E35-A panels",
               panels=dict(held=dict(ks=list(ks_held), n=300, seed=seed), taught=None if e26_panels else dict(ks=[12], n=300, seed=seed + 1)),
               split_counts={n: dict(taught=len(v[0]), held=len(v[1])) for n, v in splits.items()}, held_pairs_in_taught_set=exposure,
               model=desc, installed=dict(E_forward=f"{E.forward.__module__}.{E.forward.__name__}"), tables_sha256=TABLES_SHA,
               provenance=provenance(cmd), complete=False, curve=[], gate_diagnostics=[])
    t0 = time.process_time(); cpu_score = 0.0; curve = rec["curve"]; row = None; saved = {}
    for stop in range(X.BLOCK, E.DEV_DOSE + 1, X.BLOCK):
        losses = []
        for u in range(X.BLOCK):
            items = next(bt); dg.add(items)
            loss, it, y = step(m, opt, items); losses.append(loss)
            if census is None:
                census = X16.zero_grad_report(m, X.batch_tensor(items)[0]); print("e35a dev credit census:", census["by_role"], flush=True)
        if arm != "G":
            rec["gate_diagnostics"].append(dict(step=stop, **gate_diag(items, it, y)))
        if stop % 4096 == 0 or stop == E.DEV_DOSE:
            cb = time.process_time()
            row = dict(step=stop, loss=sum(losses) / len(losses), finite_losses=all(v == v and abs(v) != float("inf") for v in losses))
            row["sealed"] = score_all(m, held, ks_held)
            if taught:
                row["taught"] = score_all(m, taught, (12,))
            if arm != "G" and (stop in SAVE_ROWS or stop == E.DEV_DOSE):
                m.module_off = True
                row["module_off"] = score_all(m, held, (12, 40) if not e26_panels else (12,))
                m.module_off = False
            cpu_score += time.process_time() - cb
            row.update(cpu=time.process_time() - t0, cpu_scoring_so_far=cpu_score, cpu_updates_so_far=time.process_time() - t0 - cpu_score,
                       book_shift_by_role=X16.book_shift(m, book0), data_digest=dg.value())
            curve.append(row)
            if stop in SAVE_ROWS or stop == E.DEV_DOSE:
                p_ = out / f"{stem}_row{stop}.pt"
                saved[str(stop)] = dict(file=p_.name, weights_sha256=R.save_weights(m, p_), tensors_sha256=R.tensors_sha(m.state_dict()))
            print("e35a dev", arm, fpct, seed, stop, f"loss {row['loss']:.4f}", "held12", {n: v[12]["exact"] for n, v in row["sealed"].items()},
                  "taught12", {n: v[12]["exact"] for n, v in row.get("taught", {}).items()},
                  "off12", {n: v[12]["exact"] for n, v in row.get("module_off", {}).items()}, f"cpu {row['cpu']:.0f}s", flush=True)
            rec_path.write_text(json.dumps(dict(rec, credit_census_first_update=census, saved_rows=saved), indent=1))
    assert row is not None, "E11_DEV_UPDATES must be a positive multiple of 512"
    cb = time.process_time()
    row["silenced_k12"] = {n: E.score(m, pk[12], silence=True) for n, pk in held.items()}
    cpu_score += time.process_time() - cb
    rec.update(credit_census_first_update=census, saved_rows=saved, optimizer_steps=R.opt_steps(opt), optimizer_read_back=opt_record(m, opt),
               data_digest=dg.value(), tensors_sha256=R.tensors_sha(m.state_dict()), cpu_total=time.process_time() - t0, cpu_scoring=cpu_score,
               cpu_updates=time.process_time() - t0 - cpu_score, complete=True)
    rec_path.write_text(json.dumps(rec, indent=1))
    print("e35a dev", arm, fpct, seed, "complete", rec["tensors_sha256"][:16], f"cpu {rec['cpu_total']:.0f}s", flush=True)


# ---------------------------------------------------------------- checks 2-5
def onehot_pairs():
    pa, pb = T.meshgrid(IDX, IDX, indexing="ij"); pa, pb = pa.reshape(-1), pb.reshape(-1)
    return pa, pb, T.eye(89)[pa], T.eye(89)[pb]


def want_of(pa, pb):
    return {"add": (pa + pb) % 89, "sub": (pa - pb) % 89, "max": T.maximum(pa, pb), "min": T.minimum(pa, pb)}


def check_ops():
    pa, pb, A, B = onehot_pairs(); want = want_of(pa, pb); res = {}
    for n, f in zip(("add", "sub", "max", "min"), OP):
        out = f(A, B)
        res[n] = dict(argmax_exact=int((out.argmax(1) == want[n]).sum()), n=len(pa), min_mass_on_answer=float(out.gather(1, want[n][:, None]).min()),
                      max_mass_off_answer=float((out.sum(1) - out.gather(1, want[n][:, None])[:, 0]).abs().max()))
    true_add = table_op((IDX[:, None] + IDX[None, :]) % 89)
    gen = T.Generator().manual_seed(7)
    Pa, Pb = T.rand(512, 89, generator=gen).softmax(1), T.rand(512, 89, generator=gen).softmax(1)
    res["table_true_add_vs_op_add"] = dict(onehot_max_abs=float((true_add(A, B) - op_add(A, B)).abs().max()),
                                           random_dists_max_abs=float((true_add(Pa, Pb) - op_add(Pa, Pb)).abs().max()))
    tabs = T.randint(0, 89, (4, 89, 89), generator=T.Generator().manual_seed(35))
    res["random_tables"] = dict(sha256=hashlib.sha256(tabs.contiguous().numpy().tobytes()).hexdigest(), equals_runner_tables=bool(T.equal(tabs, TABLES)),
                                per_table_sha256=[hashlib.sha256(tabs[j].contiguous().numpy().tobytes()).hexdigest() for j in range(4)],
                                agreement_with_true_op_pct={n: float(100 * (OP_X[j](A, B).argmax(1) == want[n]).float().mean()) for j, n in enumerate(("add", "sub", "max", "min"))})
    ok = all(res[n]["argmax_exact"] == len(pa) and res[n]["min_mass_on_answer"] > 1 - 1e-5 for n in ("add", "sub", "max", "min"))
    ok = ok and res["table_true_add_vs_op_add"]["onehot_max_abs"] < 1e-6 and res["table_true_add_vs_op_add"]["random_dists_max_abs"] < 1e-6
    ok = ok and res["random_tables"]["equals_runner_tables"]
    res["pass"] = bool(ok)
    return res


def check_path():
    """2b: module_path with ideal interfaces on all 7,921 pairs, each operation, gaps 0, 12, 40; and dec with wd set by hand."""
    pa, pb, Aoh, Boh = onehot_pairs(); want = want_of(pa, pb); N = len(pa)
    wd = T.nn.Linear(89, 514)
    with T.no_grad():
        wd.weight.zero_(); wd.bias.zero_()
        for v in range(89):
            wd.weight[48 + v // 10, v] = 1.0
            wd.weight[257 + 48 + v % 10, v] = 1.0
    res = {}; ok = True
    for j, n in enumerate(("add", "sub", "max", "min")):
        for gap in (0, 12, 40):
            L = gap + 3                    # step 0: write A; step gap+1: write B; step gap+2: hold (no write); the last step is read
            s1, s2 = 0, gap + 1
            c = T.full((N, L, 89), 1 / 89)
            c[:, s1] = Aoh; c[:, s2] = Boh
            g = T.zeros(N, L, 2); g[:, s1, 0] = 1; g[:, s2, 1] = 1
            o = T.zeros(N, L, 4); o[..., j] = 1
            with T.no_grad():
                r = module_path(c, g, o)
                last = r[:, -1]
                d = wd(last).reshape(N, 2, 257)
            mass = last.gather(1, want[n][:, None])[:, 0]
            tens = d[:, 0].argmax(1); units = d[:, 1].argmax(1)
            dig_ok = (tens == 48 + want[n] // 10) & (units == 48 + want[n] % 10)
            e = dict(argmax_exact=int((last.argmax(1) == want[n]).sum()), n=N, min_mass_on_answer=float(mass.min()), dec_two_bytes_exact=int(dig_ok.sum()))
            res[f"{n}_gap{gap}"] = e
            ok = ok and e["argmax_exact"] == N and e["min_mass_on_answer"] > 1 - 1e-5 and e["dec_two_bytes_exact"] == N
    res["pass"] = bool(ok)
    return res


def check_start(seed=73129):
    install()
    g = new_net(seed, "G").state_dict(); res = {}
    for arm in ("M", "X", "F"):
        sd = new_net(seed, arm).state_dict()
        res[arm] = dict(G_params=len(g), all_equal=all(T.equal(g[k], sd[k]) for k in g), extra=[k for k in sd if k not in g])
    res["M_X_new_params_equal"] = all(T.equal(new_net(seed, "M").state_dict()[k], new_net(seed, "X").state_dict()[k]) for k in res["M"]["extra"])
    res["pass"] = all(res[a]["all_equal"] for a in ("M", "X", "F")) and res["M_X_new_params_equal"]
    return res


def check_ablation(seed=73129):
    install()
    tp = [(t, E.split(t, .25, seed)[0]) for t in E.DEV]
    bt = E.batches(tp, seed, E.KMAX["M"])
    res = {}
    for arm in ("M", "X"):
        m = new_net(seed, arm); opt = make_opt(m)
        for _ in range(8):
            step(m, opt, next(bt))
        gnet = new_net(seed, "G")
        gnet.load_state_dict({k: v for k, v in m.state_dict().items() if k in gnet.state_dict()})
        x = X.batch_tensor(next(bt))[0]
        with T.no_grad():
            m.module_off = True; off = forward(m, x, False, True)[0]; m.module_off = False
            on = forward(m, x, False, True)[0]
            gl = forward(gnet, x, False, True)[0]
        res[arm] = dict(module_off_equals_G_exactly=bool(T.equal(off, gl)), max_abs=float((off - gl).abs().max()), module_on_differs=bool(not T.equal(on, gl)))
    res["pass"] = all(res[a]["module_off_equals_G_exactly"] and res[a]["module_on_differs"] for a in ("M", "X"))
    return res


def check_noparse():
    sig = list(inspect.signature(forward).parameters)
    tree = ast.parse(inspect.getsource(forward) + "\n" + inspect.getsource(module_path))
    names = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)})
    banned = [n for n in names if any(s in n.lower() for s in ("mask", "role", "pos", "fine", "items", "prompt"))]
    install()
    m = new_net(73129, "M")
    x = X.batch_tensor([E.render(E.DEV[0], 3, 5, 4, __import__("random").Random(0))])[0]
    with T.no_grad():
        out = forward(m, x, False, True)[0]
    res = dict(signature=sig, signature_is_harness=sig == ["m", "x", "local", "through_time", "keep", "silence"], names_in_forward_and_module_path=names,
               banned_names_found=banned, only_tensor_argument="x (B, L) long byte tensor", logits_shape=list(out.shape))
    res["pass"] = res["signature_is_harness"] and not banned
    return res


def run_check(part, qual):
    qual = Path(qual); qual.mkdir(parents=True, exist_ok=True)
    fn = dict(ops=check_ops, path=check_path, start=check_start, ablation=check_ablation, noparse=check_noparse)[part]
    t0 = time.process_time(); r = fn(); r["cpu"] = time.process_time() - t0
    r["provenance"] = provenance(" ".join(sys.argv))
    p = qual / f"check_{part}.json"
    p.write_text(json.dumps(r, indent=1))
    print(part, "pass" if r["pass"] else "FAIL", json.dumps({k: v for k, v in r.items() if k != "provenance"})[:1500], flush=True)
    assert r["pass"], f"check {part} failed"


if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])

    def opt_(flag, default=None):
        return a[a.index(flag) + 1] if flag in a else default

    if a[0] == "dev":
        assert a[1] in ("G", "M", "X", "F")
        dev(a[1], int(a[2]), int(a[3]), opt_("--out", str(OUT)), cmd, e26_panels="--e26-panels" in a)
    elif a[0] == "check":
        run_check(a[1], opt_("--qual", str(OUT / "checks")))
    else:
        raise SystemExit(__doc__)
