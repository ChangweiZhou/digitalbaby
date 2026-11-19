"""E57 runner (brief TASK_E57_THE_SWEEP_WITH_A_SCALE_FOR_EACH_RECEIVING_UNIT_20261117.md, revision 5, on claude/vibrant-cerf-sje726 at
2b85dd8b): the sweep with a scale for each receiving unit, in the network with a limited input (E49's in16), in the network with limited
connections (E42's F10) and in the dense network (E42's G); and, in in16, the sweep with the scale `matrix` when the code (l1) is not
decayed.

A new file; no existing source is edited. exp49_interface.py and e49_interface.py are byte-for-byte copies from E49's worker at a36de661
(their SHA-256 is asserted on import; E49_COPIES_FOR_E57.txt beside them). The run is a copy of the text of exp49_interface.run (lines
164-243) with the changes the brief lists (point 4) and saving and resuming (point 7); the report shows the difference.

    python3 exp57_scale.py run <in16|F10|G> <matrix|row|rowcells|rowcol|none> <S|B> 73129 [--dose N] [--code-decay 0] [--out DIR]
                               [--ckpt N] [--resume] [--die-after N]
    python3 exp57_scale.py check <1|2|3|5|7|8> [--qual DIR]
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C26        # noqa: E402  first harness import (environment gate)

import torch as T                    # noqa: E402
from torch.nn.functional import linear  # noqa: E402

COPIED = {"exp49_interface.py": "cdf93ce167d562c8a3765c1ea4f98f74623469492324a85aa1eb34216b7ccbf3",
          "e49_interface.py": "92f968e092f2d27eaff4ee7e93ecbbb82b053cc88b6a2d3f967adadf7e3791f1"}
for _f, _h in COPIED.items():
    assert hashlib.sha256((HERE / _f).read_bytes()).hexdigest() == _h, f"{_f} is not E49's copy at a36de661"

import e42_cells as L42              # noqa: E402
import e49_interface as I49          # noqa: E402
import exp42_cells as R42            # noqa: E402
import exp49_interface as R49        # noqa: E402

R_, X16, E, X = C26.R, C26.X16, C26.E, C26.X
T.set_num_threads(1)

U, D = I49.U, I49.D
SEED = 73129
DOSE = R42.DOSE                      # 24,576
ROW = R42.ROW                        # 256
RULES = R42.RULES                    # S: the sweep (local=True); B: back-propagation
OUT = X.OUT / "e57"
QUAL = X.OUT / "e57_qual"
NETWORKS = {"in16": (16, 256), "F10": (512, 256), "G": (512, 256)}
MA_EXPECTED = {"in16": 549632, "F10": 930560, "G": 1118720}
SOURCES = ("exp57_scale.py",) + R49.SOURCES
SCALES = ("matrix", "row", "rowcells", "rowcol")
COS_TENSORS = ("book.weight", "l1.weight", "rnn.weight_ih_l0", "rnn.weight_hh_l0", "lm.weight", "l2.weight")

SCALE_AT = {"matrix": ("matrix", "matrix"), "row": ("row", "row"), "rowcells": ("row", "matrix"),
            "rowcol": ("rowcol", "matrix"), "exact": ("exact", "exact"), "none": ("matrix", "matrix")}
# the scale of a run: (form at the input map and at the recurrent map, form at l1, l2, lm and l3)


# ---------------------------------------------------------------- point 1: the crossing
def scale_magnitudes(W, maskf, form):     # W detached; maskf of W's shape, ones where the map is dense
    A = W.abs() * maskf
    if form == "exact":                   # check 2 only: the sweep is then back-propagation
        return A
    mean = A.sum() / maskf.sum()
    if form == "matrix":
        return mean
    row = A.sum(1, keepdim=True) / maskf.sum(1, keepdim=True).clamp(min=1)
    if form == "row":
        return row
    assert form == "rowcol", form
    col = A.sum(0, keepdim=True) / maskf.sum(0, keepdim=True).clamp(min=1)
    return row * col / mean


def cross(x, W, b, maskf, local, S):
    return L42.MaskedSign.apply(x, W, b, maskf, S) if local else linear(x, W * maskf, b)


_ONES = {}


def ONES(W):
    k = tuple(W.shape)
    if k not in _ONES:
        _ONES[k] = T.ones(k)
    return _ONES[k]


# ---------------------------------------------------------------- point 2: the forward
def forward(m, layer, iface, x, local, scale="matrix", silence=False):
    assert layer.K == 1
    cells, others = SCALE_AT[scale]
    mag = lambda W, mk, form: scale_magnitudes(W.detach(), mk, form) if local else None

    def dense(v, lay):                    # l1, l2, l3: a mask of ones
        mk = ONES(lay.weight)
        return cross(v, lay.weight, lay.bias, mk, local, mag(lay.weight, mk, others))
    z1 = T.tanh(dense(m.book(x), m.l1))
    B, L = x.shape
    Wi, bi, mi = m.rnn.weight_ih_l0, m.rnn.bias_ih_l0, iface.mask_in.repeat(3, 1).float()
    gi = cross(z1, Wi, bi, mi, local, mag(Wi, mi, cells))
    Wh, bh, mh = m.rnn.weight_hh_l0, m.rnn.bias_hh_l0, layer.mask3().float()      # mh is all ones for G
    Sh = mag(Wh, mh, cells)
    h = T.zeros(B, I49.U); hs = []
    for t in range(L):
        gh = cross(h, Wh, bh, mh, local, Sh)      # never L42.rec: for G it calls SignLinear and ignores the scale
        i_r, i_z, i_n = gi[:, t].chunk(3, 1); h_r, h_z, h_n = gh.chunk(3, 1)
        r = T.sigmoid(i_r + h_r); z = T.sigmoid(i_z + h_z); n = T.tanh(i_n + r * h_n)
        h = (1 - z) * n + z * h
        hs.append(h)
    Hs = T.stack(hs, 1)
    if silence:
        Hs = T.zeros_like(Hs)
    mo = iface.mask_out.float()
    out = cross(Hs, m.lm.weight, m.lm.bias, mo, local, mag(m.lm.weight, mo, others))
    z2 = T.tanh(dense(z1, m.l2) + out)
    return dense(z2, m.l3), hs


# ---------------------------------------------------------------- point 3: the networks
def build(network, seed):
    if network in ("in16", "F10"):
        setting = "in16" if network == "in16" else "F10"
        return R49.build(setting, seed)
    assert network == "G"
    m, layer = R42.build("G", seed)
    before = R_.tensors_sha(m.state_dict())
    iface = I49.Interface(m, seed, 512, 256)
    assert R_.tensors_sha(m.state_dict()) == before, "G: the interface changed the network"
    assert bool(layer.mask3().all()), "G: the recurrent mask is not full"
    return m, layer, iface


def install(layer, iface):
    """E49's install with the new forward. Scoring calls the forward with local=False, so the scale plays no part in it."""
    import arch26
    arch26.install()

    def fwd(m, x, local, through_time, keep=False, silence=False):
        lg, hs = forward(m, layer, iface, x, local, silence=silence)
        return lg, dict(h=hs)
    E.forward = fwd


def step(m, layer, iface, opt, items, local, scale):
    """exp26_candidate.step's text with E57's forward and the scale."""
    x, y, mask = X.batch_tensor(items)
    logits = forward(m, layer, iface, x, local, scale)[0]
    loss = E.cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach())


def no_code_decay(m, opt):
    dec = [g for g in opt.param_groups if g["weight_decay"] == C26.WD]
    nod = [g for g in opt.param_groups if g["weight_decay"] == 0.0]
    assert len(dec) == 1 and len(nod) == 1
    for p in (m.l1.weight, m.l1.bias):
        k = [i for i, q in enumerate(dec[0]["params"]) if q is p]
        assert len(k) == 1
        nod[0]["params"].append(dec[0]["params"].pop(k[0]))


# ---------------------------------------------------------------- point 5: cosines and sizes during the run
def cosine_batches(seed):
    """The second to the seventeenth batch of the stream, from a second generator (the run's own stream is not advanced)."""
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    g = E.batches(tp, seed, E.KMAX["M"])
    next(g)
    return [next(g) for _ in range(16)]


def grads_of(m, layer, iface, x, y, mask, local, scale):
    for q in m.parameters():
        q.grad = None
    lg = forward(m, layer, iface, x, local, scale)[0]
    E.cross_entropy(lg[mask], y[mask]).backward()
    g = {n: (q.grad.detach().clone() if q.grad is not None else T.zeros_like(q)) for n, q in m.named_parameters()}
    for q in m.parameters():
        q.grad = None
    return g


def cos_d(a, b):
    return float(T.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), 0))


def cosine_set(m, layer, iface, batches, scales=SCALES, tensors=COS_TENSORS):
    """For each scale, the mean over the batches of the cosine between the sweep's gradient under that scale and the exact gradient."""
    acc = {s: {n: 0.0 for n in tensors} for s in scales}
    for items in batches:
        x, y, mask = X.batch_tensor(items)
        ex = grads_of(m, layer, iface, x, y, mask, False, "none")
        for s in scales:
            sw = grads_of(m, layer, iface, x, y, mask, True, s)
            for n in tensors:
                acc[s][n] += cos_d(sw[n], ex[n])
    return {s: {n: v / len(batches) for n, v in d.items()} for s, d in acc.items()}


@T.no_grad()
def sizes(m, batches):
    """l1_weight_mean_abs, and code_mean_abs over the positions of the batches whose byte is not the padding symbol 257."""
    tot, cnt = 0.0, 0
    for items in batches:
        x, _, _ = X.batch_tensor(items)
        keep = x != 257
        z = T.tanh(linear(m.book(x), m.l1.weight, m.l1.bias))
        tot += float(z.abs()[keep].double().sum()); cnt += int(keep.sum()) * z.shape[-1]
    return dict(l1_weight_mean_abs=float(m.l1.weight.detach().abs().mean()), code_mean_abs=tot / cnt)


def opt_state_digest(opt):
    h = hashlib.sha256()
    for st in opt.state.values():
        for k in sorted(st):
            v = st[k]
            h.update(k.encode()); h.update(v.detach().contiguous().numpy().tobytes() if T.is_tensor(v) else repr(v).encode())
    return h.hexdigest()


# ---------------------------------------------------------------- receipts and paths
def receipt_path(out, network, scale, rule, seed, dose, code_decay):
    tag = ("_cd0" if code_decay == 0 else "") + ("" if dose == DOSE else f"_u{dose}")
    return Path(out) / f"e57_{network}_{scale}_{rule}{tag}_{seed}.json"


def versions():
    import numpy
    return dict(python=sys.version.split()[0], torch=T.__version__, numpy=numpy.__version__, torch_threads=T.get_num_threads(),
                OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"), MKL_NUM_THREADS=os.environ.get("MKL_NUM_THREADS"))


def provenance(cmd):
    p = C26.provenance(cmd)
    p["sources_sha256"] = {s: X16.sha_file(HERE / s) for s in SOURCES}
    p["runner_sha256"] = X16.sha_file(HERE / "exp57_scale.py")
    p["versions"] = versions()
    return p


# ---------------------------------------------------------------- point 4: the run (exp49_interface.run with the brief's changes)
def run(network, scale, rule, seed, out, cmd, dose=None, code_decay=None, ckpt=4096, resume=False, die_after=None):
    assert (scale == "none") == (rule == "B")
    assert code_decay is None or (code_decay == 0 and network == "in16" and scale == "matrix" and rule == "S")
    assert ckpt == 0 or ckpt % 2048 == 0, "--ckpt must be 0 or a multiple of 2,048"
    k_in, k_out = NETWORKS[network]
    local = RULES[rule]
    dose = dose or DOSE
    assert dose % ROW == 0
    rp = receipt_path(out, network, scale, rule, seed, dose, code_decay)
    paths = dict(receipt=rp, start=rp.with_name(rp.stem + "_start.pt"), weights=rp.with_suffix(".pt"),
                 answers=rp.with_name(rp.stem + "_answers.json"))
    cp = rp.with_suffix(".ckpt")
    saved = T.load(cp, weights_only=False) if (resume and cp.exists()) else None
    if saved is None:
        R_.refuse_existing(*paths.values(), str(paths["weights"]) + ".tmp", str(paths["start"]) + ".tmp", cp)
    else:
        R_.refuse_existing(paths["weights"], paths["answers"], str(paths["weights"]) + ".tmp")
    Path(out).mkdir(parents=True, exist_ok=True)
    m, layer, iface = build(network, seed)
    rd = R49.readers(iface)
    print(f"e57 {network} {scale}{' cd0' if code_decay == 0 else ''} {rule} {seed}: least cells reading an input value {rd['input_value_min']}, "
          f"least readout units reading a cell {rd['cell_min']}", flush=True)
    if rd["input_value_min"] == 0 or rd["cell_min"] == 0:
        raise SystemExit(f"e57 {network} {seed}: an input value or a cell is read by nobody; stopped as the brief requires")
    install(layer, iface)
    opt = R42.make_opt(m, layer)
    if code_decay == 0:
        no_code_decay(m, opt)
    n_rec = int(layer.mask3().sum())
    cnt = iface.counts(m, n_rec)
    assert cnt["multiply_adds"] == MA_EXPECTED[network], (network, cnt)
    ma = cnt["multiply_adds"]
    A, Rs, TA, LP = R42.panels(seed)
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    bt = E.batches(tp, seed, E.KMAX["M"]); dg = C26.Digest()
    cb = cosine_batches(seed)
    if saved is None:
        start = R49.weights_blob(m, layer, iface)
        R49.save_pt(start, paths["start"])
        rec = dict(experiment="E57", kind="birth", setting=network, config=network, scale=scale,
                   scale_at=dict(recurrent_layer=SCALE_AT[scale][0], other_maps=SCALE_AT[scale][1]), rule=rule, local=local, seed=seed,
                   dose=dose, row_every=ROW, code_decay=0 if code_decay == 0 else C26.WD,
                   k_in=k_in, k_out=k_out, cells=U, N=layer.N, batch=X.BATCH, kmax=E.KMAX["M"], counts=cnt, multiply_adds_per_position=ma,
                   params_module=sum(q.numel() for q in m.parameters()), readers=rd, draw_digests=R49.draw_digests(seed, k_in, k_out),
                   start_scale=dict(input=(D / k_in) ** 0.5 if k_in < D else 1.0, lm=(U / k_out) ** 0.5 if k_out < U else 1.0),
                   optimizer=R42.opt_record(m, layer, opt),
                   update="exp26_candidate.step's text with exp57_scale.forward(m, layer, iface, x, local, scale) in place of arch26.forward",
                   data="exp11_cell.batches([(t, split(t, .5, seed)[0]) for t in DEV], seed, 12)",
                   forward="exp57_scale.forward(m, layer, iface, x, local, scale)",
                   start_file=paths["start"].name, start_tensors_sha256=R_.tensors_sha(m.state_dict()),
                   mask_sha256=dict(recurrent=R42.tsha(layer.mask), input=R42.tsha(iface.mask_in), readout=R42.tsha(iface.mask_out)),
                   ckpt_every=ckpt, provenance=provenance(cmd), complete=False, curve=[], cosines=[], resumes=[])
        cpu_up = cpu_sc = cpu_cos = 0.0; wall0 = 0.0; u0 = 0
    else:
        u0 = saved["updates"]
        for _ in range(u0):
            dg.add(next(bt))
        rec = saved["receipt"]
        assert dg.value() == rec["curve"][-1]["data_digest"], "the stream drawn again is not the one the run saw"
        sd0 = {k: v for k, v in T.load(paths["start"], weights_only=False).items() if k in m.state_dict()}
        assert R_.tensors_sha(sd0) == rec["start_tensors_sha256"], "the starting file is not the run's"
        m.load_state_dict(saved["model"]); opt.load_state_dict(saved["opt"])
        assert R42.opt_record(m, layer, opt) == rec["optimizer"], "the optimizer's groups differ from the receipt's"
        assert not layer.added_parameters()
        cpu_up, cpu_sc, cpu_cos, wall0 = saved["cpu_updates"], saved["cpu_scoring"], saved["cpu_cosines"], saved["wall"]
        rec["resumes"].append(dict(update=u0, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), cmd=cmd))
    curve = rec["curve"]; w0 = time.time() - wall0; losses = []

    def score_row(u):
        nonlocal cpu_sc
        c = time.process_time()
        row = dict(step=u, **R42.score_all(m, A, Rs, TA, LP))
        cpu_sc += time.process_time() - c
        lim = R42.check_limit(m, layer)
        assert all(lim.values()), f"E42's check 4 failed at row {u}: {lim}"
        mk = R49.check_masks(m, layer, iface, opt)
        assert all(all(v.values()) for v in mk.values()), f"check 4 (E49's check 3) failed at row {u}: {mk}"
        row.update(limit=lim, masks=mk, kept_mean_magnitude=R49.magnitudes(m, layer, iface), cpu_updates=cpu_up, cpu_scoring=cpu_sc,
                   cpu_cosines=cpu_cos, multiply_adds=ma * X.BATCH * u, data_digest=dg.value(),
                   loss=(sum(losses) / len(losses)) if losses else None, finite_losses=all(v == v and abs(v) != float("inf") for v in losses),
                   wall=time.time() - w0)
        curve.append(row)
        print(f"e57 {network} {scale}{' cd0' if code_decay == 0 else ''} {rule} {seed} {u}: A {row['A']:.2f} R {row['R']:.2f} "
              f"taught {row['taught']:.2f} len12 {row['len12']:.1f} len40 {row['len40']:.1f} cpu {cpu_up:.0f}+{cpu_sc:.0f}+{cpu_cos:.0f}s",
              flush=True)

    def cos_row(u):
        nonlocal cpu_cos
        c = time.process_time()
        check = len(rec["cosines"]) < 2
        if check:
            b_sd, b_opt = R_.tensors_sha(m.state_dict()), opt_state_digest(opt)
        cs = cosine_set(m, layer, iface, cb)
        sz = sizes(m, cb)
        if check:
            assert R_.tensors_sha(m.state_dict()) == b_sd and opt_state_digest(opt) == b_opt, "a set of cosines changed the network"
        cpu_cos += time.process_time() - c
        rec["cosines"].append(dict(update=u, cosines=cs, **sz, cpu=time.process_time() - c))
        curve[-1]["cpu_cosines"] = cpu_cos

    def save_state(u):
        tmp = Path(str(cp) + ".tmp")
        T.save(dict(updates=u, model=m.state_dict(), opt=opt.state_dict(), receipt=rec, cpu_updates=cpu_up, cpu_scoring=cpu_sc,
                    cpu_cosines=cpu_cos, wall=time.time() - w0), tmp)
        os.replace(tmp, cp)

    if saved is None:
        score_row(0)
        cos_row(0)
        rp.write_text(json.dumps(rec, indent=1))
    for u in range(u0 + 1, dose + 1):
        items = next(bt)
        dg.add(items)
        c = time.process_time(); losses.append(step(m, layer, iface, opt, items, local, scale)); cpu_up += time.process_time() - c
        if u % ROW == 0:
            score_row(u); losses = []
            if u % 2048 == 0:
                cos_row(u)
            if u % 2048 == 0 or u == dose:
                rp.write_text(json.dumps(rec, indent=1))
            if ckpt and u % ckpt == 0 and u < dose:
                save_state(u)
        if die_after is not None and u == die_after:
            os._exit(9)
    sd = m.state_dict()
    R49.save_pt(R49.weights_blob(m, layer, iface), paths["weights"])
    paths["answers"].write_text(json.dumps({str(k): E.respond(m, [p for p, _ in it]) for k, it in LP.items()}))
    rec.update(complete=True, weights=paths["weights"].name, weights_sha256=X16.sha_file(paths["weights"]),
               start_sha256=X16.sha_file(paths["start"]), tensors_sha256=R_.tensors_sha(sd), answers_file=paths["answers"].name,
               optimizer_steps=R_.opt_steps(opt), optimizer_read_back=R42.opt_record(m, layer, opt), data_digest=dg.value(),
               cpu_updates=cpu_up, cpu_scoring=cpu_sc, cpu_cosines=cpu_cos, wall=time.time() - w0,
               processes_beside=os.environ.get("E57_BESIDE", "not recorded"))
    rp.write_text(json.dumps(rec, indent=1))
    if cp.exists():
        cp.unlink()
    print("e57 complete", rp.name, rec["tensors_sha256"][:16], flush=True)
    return rec


# ---------------------------------------------------------------- checks
def gitjson(ref, path):
    import subprocess
    return json.loads(subprocess.run(["git", "show", f"{ref}:MOD97_LOCAL_RULE_20261106/{path}"], cwd=HERE, capture_output=True, text=True,
                                     check=True).stdout)


ROW_KEYS = ("loss", "A_counts", "R_counts", "taught_counts", "length_counts", "data_digest")


def rows_equal(mine, ref, upto):
    a = [r for r in mine["curve"] if r["step"] <= upto]; b = [r for r in ref["curve"] if r["step"] <= upto]
    diffs = [dict(step=x["step"], key=k, mine=x[k], ref=y[k]) for x, y in zip(a, b) for k in ROW_KEYS if x[k] != y[k]]
    dig = len(a) == len(b) and all(x["data_digest"] == y["data_digest"] for x, y in zip(a, b))
    return dict(rows=len(a), rows_ref=len(b), equal=len(a) == len(b) and not diffs, digests_equal=dig, first_differences=diffs[:6])


def first_batch():
    tp = [(t, E.split(t, .5, SEED)[0]) for t in E.DEV]
    return X.batch_tensor(next(E.batches(tp, SEED, E.KMAX["M"])))


def all_grads(m, fwd, x, y, mask):
    for q in m.parameters():
        q.grad = None
    lg = fwd(x)
    E.cross_entropy(lg[mask], y[mask]).backward()
    g = {n: q.grad.detach().clone() for n, q in m.named_parameters() if q.grad is not None}
    for q in m.parameters():
        q.grad = None
    return lg.detach(), g


def check1(qual, cmd):
    x, y, mask = first_batch()
    out = dict(a={}, b={})
    exact_a = True; worst_a = 0.0
    for net in ("G", "F10", "in16"):
        old42 = ("R42.forward", lambda m, l, f, loc: R42.forward(m, l, x, loc)[0])
        old49 = ("I49.forward", lambda m, l, f, loc: I49.forward(m, l, f, x, loc)[0])
        olds = {"G": (old42,), "F10": (old42, old49), "in16": (old49,)}[net]
        for oname, old in olds:
            for rule, loc in RULES.items():
                m, layer, iface = build(net, SEED)
                ln, gn = all_grads(m, lambda xx: forward(m, layer, iface, xx, loc, "matrix")[0], x, y, mask)
                lo, go = all_grads(m, lambda xx: old(m, layer, iface, loc), x, y, mask)
                eq = bool(T.equal(ln, lo)) and sorted(gn) == sorted(go) and all(T.equal(gn[k], go[k]) for k in gn)
                rel = {k: float((gn[k] - go[k]).abs().max()) / max(float(go[k].abs().max()), 1e-30) for k in gn if not T.equal(gn[k], go[k])}
                lrel = float((ln - lo).abs().max()) / max(float(lo.abs().max()), 1e-30)
                exact_a = exact_a and eq; worst_a = max([worst_a, lrel] + list(rel.values()))
                out["a"][f"{net} {rule} vs {oname}"] = dict(equal=eq, logits_rel=lrel, grads_rel_where_unequal=rel)
                print(f"check 1a {net} {rule} vs {oname}: equal {eq} {rel if rel else ''}", flush=True)
    q = Path(qual)
    runs = {}
    runs["G"] = run("G", "matrix", "S", SEED, q, cmd, dose=2048, ckpt=0)
    runs["F10"] = run("F10", "matrix", "S", SEED, q, cmd, dose=2048, ckpt=0)
    runs["in16_256"] = run("in16", "matrix", "S", SEED, q, cmd, dose=256, ckpt=0)
    runs["in16_2048"] = run("in16", "matrix", "S", SEED, q / "in16_2048", cmd, dose=2048, ckpt=0)
    g_ref = gitjson("44f91edc", "exp45/e42_qual/check1_73129_runs/e42_G_S_dry2048_73129.json")
    in16_ref = gitjson("a36de661", "exp45/e49/e49_in16_S_73129.json")
    b = out["b"]
    b["G_2048"] = dict(tensors_sha256=runs["G"]["tensors_sha256"], expected=R42.DRY2048["S"], equal=runs["G"]["tensors_sha256"] == R42.DRY2048["S"],
                       rows=rows_equal(runs["G"], g_ref, 2048))
    b["F10_2048"] = dict(start=runs["F10"]["start_tensors_sha256"], start_expected="dede71d76cb8a9e6af09aa5394eb90c69a45177f6d3dc1861cce0c3b7e35a561",
                         end=runs["F10"]["tensors_sha256"], end_expected="837dadc12e8b1bd702c5d602b80e3fdca82eceda478dd110b480aff57815ea75")
    b["F10_2048"]["equal"] = b["F10_2048"]["start"] == b["F10_2048"]["start_expected"] and b["F10_2048"]["end"] == b["F10_2048"]["end_expected"]
    b["in16_256"] = dict(start=runs["in16_256"]["start_tensors_sha256"], start_expected="657e185828eb7a4c6e91a3c3a0bae71d6d0c8d38cd84840a66c3924834ff45da",
                         end=runs["in16_256"]["tensors_sha256"], end_expected="88cad76a4806601be9a859832b9167b95a967fb3cac4a2f7ee8de8b641cb6f34")
    b["in16_256"]["equal"] = b["in16_256"]["start"] == b["in16_256"]["start_expected"] and b["in16_256"]["end"] == b["in16_256"]["end_expected"]
    b["in16_2048"] = dict(rows=rows_equal(runs["in16_2048"], in16_ref, 2048))
    b_ok = b["G_2048"]["equal"] and b["G_2048"]["rows"]["equal"] and b["F10_2048"]["equal"] and b["in16_256"]["equal"] and b["in16_2048"]["rows"]["equal"]
    out["a_all_equal"] = exact_a; out["a_worst_relative_difference"] = worst_a; out["b_all_hold"] = b_ok
    out["rule_c"] = (not exact_a) and worst_a <= 1e-6
    out["pass"] = (exact_a and b_ok) or (out["rule_c"] and all(v["rows"]["digests_equal"] for v in b.values() if "rows" in v))
    out["cost"] = {k: dict(cpu_per_update=r["cpu_updates"] / r["dose"], cpu_per_row=r["cpu_scoring"] / len(r["curve"]),
                           cpu_per_cosine_set=r["cpu_cosines"] / len(r["cosines"]), dose=r["dose"]) for k, r in runs.items()}
    return out


def load_e49_weights(name, m, layer, iface):
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "w.pt"
        p.write_bytes(subprocess.run(["git", "show", f"a36de661:MOD97_LOCAL_RULE_20261106/exp45/e49/{name}"], cwd=HERE, capture_output=True,
                                     check=True).stdout)
        f = T.load(p, weights_only=False)
    assert T.equal(f["mask"], layer.mask) and T.equal(f["mask_in"], iface.mask_in) and T.equal(f["mask_out"], iface.mask_out)
    m.load_state_dict({k: f[k] for k in m.state_dict()})


EXPECTED2 = {"e49_in16_S_73129.pt": {"matrix": (0.367, 0.298, 0.603, 0.651, 0.606, 0.643), "row": (0.697, 0.624, 0.676, 0.762, 0.694, 0.710),
                                     "rowcells": (0.473, 0.527, 0.594, 0.654, 0.606, 0.643), "rowcol": (0.638, 0.721, 0.714, 0.710, 0.606, 0.643)},
             "e49_in16_B_73129.pt": {"matrix": (0.442, 0.309, 0.523, 0.582, 0.700, 0.706), "row": (0.405, 0.363, 0.567, 0.631, 0.700, 0.707)}}


def check2(qual, cmd):
    out = dict(files={}); ok = True
    cb = cosine_batches(SEED)
    for name, exp in EXPECTED2.items():
        m, layer, iface = R49.build("in16", SEED)
        load_e49_weights(name, m, layer, iface)
        cs = cosine_set(m, layer, iface, cb, scales=tuple(exp))
        r = {}
        for s, vals in exp.items():
            got = [cs[s][n] for n in COS_TENSORS]
            diff = max(abs(a - b) for a, b in zip(got, vals))
            r[s] = dict(got=got, expected=list(vals), max_diff=diff, within=diff <= 0.002)
            ok = ok and r[s]["within"]
            print(f"check 2 {name} {s}: {[round(v, 3) for v in got]} max diff {diff:.4f}", flush=True)
        if name == "e49_in16_S_73129.pt":
            all13 = tuple(n for n, _ in m.named_parameters())
            ex = cosine_set(m, layer, iface, cb, scales=("exact",), tensors=all13)["exact"]
            r["exact"] = dict(cosines=ex, min=min(ex.values()), within=min(ex.values()) >= 1 - 1e-6)
            ok = ok and r["exact"]["within"]
            print(f"check 2 exact: min {min(ex.values()):.9f}", flush=True)
        out["files"][name] = r
    out["pass"] = ok
    return out


def check3(qual, cmd):
    rec = run("in16", "none", "B", SEED, Path(qual), cmd, dose=256, ckpt=0)
    ref = gitjson("a36de661", "exp45/e49/e49_in16_B_73129.json")
    exp = "c0eb654815c62c9a6e3826b50a3e8044b12f776519d9a6278888178913a0b5b0"
    rows = rows_equal(rec, ref, 256)
    return dict(tensors_sha256=rec["tensors_sha256"], expected=exp, equal=rec["tensors_sha256"] == exp, rows=rows,
                **{"pass": rec["tensors_sha256"] == exp and rows["equal"]})


def check5(qual, cmd):
    c1 = json.loads((Path(QUAL) / f"check1_{SEED}.json").read_text())
    cost = c1["cost"]
    pu = {k: v["cpu_per_update"] for k, v in cost.items()}
    pr = sum(v["cpu_per_row"] for v in cost.values()) / len(cost)
    pc = sum(v["cpu_per_cosine_set"] for v in cost.values()) / len(cost)
    u_in16 = cost["in16_2048"]["cpu_per_update"]; u_f10 = cost["F10"]["cpu_per_update"]; u_g = cost["G"]["cpu_per_update"]
    runs = [("in16", u_in16, 49152)] * 5 + [("in16 B", u_in16, 24576), ("F10", u_f10, 24576), ("F10", u_f10, 24576), ("G", u_g, 24576)]
    per = [(n, (d * u + (d // ROW + 1) * pr + (d // 2048 + 1) * pc) / 3600) for n, u, d in runs]
    tot = sum(h for _, h in per)
    return dict(cpu_per_update=pu, cpu_per_row=pr, cpu_per_cosine_set=pc, forecast_per_run_cpu_h=per, forecast_nine_runs_cpu_h=tot,
                limit_cpu_h=21, note="the run under back-propagation is costed at the sweep's rate of an update; rows and cosine sets as in the dry runs",
                **{"pass": tot <= 21})


def check7(qual, cmd):
    x, y, mask = first_batch()
    tp = [(t, E.split(t, .5, SEED)[0]) for t in E.DEV]
    items = next(E.batches(tp, SEED, E.KMAX["M"]))
    nets = []
    for cd in (False, True):
        m, layer, iface = R49.build("in16", SEED)
        import arch26
        arch26.install()
        opt = R42.make_opt(m, layer)
        if cd:
            no_code_decay(m, opt)
        w0 = {k: v.detach().clone() for k, v in m.named_parameters()}
        step(m, layer, iface, opt, items, True, "matrix")
        nets.append((m, w0))
    (m1, w0), (m2, _) = nets
    p1, p2 = dict(m1.named_parameters()), dict(m2.named_parameters())
    others = [k for k in p1 if k not in ("l1.weight", "l1.bias")]
    eq11 = {k: bool(T.equal(p1[k], p2[k])) for k in others}
    l1 = {}
    for k in ("l1.weight", "l1.bias"):
        d = (p2[k] - p1[k]).detach(); e = 0.00025 * w0[k]
        l1[k] = dict(equal=bool(T.equal(p1[k], p2[k])), max_dev=float((d - e).abs().max()))
    a = dict(eleven_equal=eq11, all_eleven_equal=all(eq11.values()), l1=l1,
             ok=all(eq11.values()) and not l1["l1.weight"]["equal"] and all(v["max_dev"] <= 1e-6 for v in l1.values()))
    q = Path(qual)
    rec = run("in16", "matrix", "S", SEED, q, cmd, dose=256, code_decay=0, ckpt=0)
    refp = q.parent / "check1_73129_runs" / "e57_in16_matrix_S_u256_73129.json"
    ref = json.loads(refp.read_text())
    groups = {g["weight_decay"]: g["params"] for g in rec["optimizer"]["groups"]}
    rgroups = {g["weight_decay"]: (g["lr"], g["betas"]) for g in ref["optimizer"]["groups"]}
    b = dict(group_no_decay=sorted(groups[0.0]), group_decay=sorted(groups[C26.WD]),
             groups_ok=sorted(groups[0.0]) == ["book.weight", "l1.bias", "l1.weight"] and len(groups[C26.WD]) == 10,
             rates_betas_ok=all((g["lr"], g["betas"]) == rgroups[g["weight_decay"]] for g in rec["optimizer"]["groups"]),
             read_back_equal=rec["optimizer_read_back"] == rec["optimizer"], code_decay=rec["code_decay"],
             start_equal=rec["start_tensors_sha256"] == ref["start_tensors_sha256"],
             row0_counts_equal=all(rec["curve"][0][k] == ref["curve"][0][k] for k in ("A_counts", "R_counts", "taught_counts", "length_counts")),
             digest256_equal=rec["curve"][-1]["data_digest"] == ref["curve"][-1]["data_digest"],
             tensors_differ=rec["tensors_sha256"] != ref["tensors_sha256"], reference_tensors_sha256=ref["tensors_sha256"])
    b["ok"] = b["groups_ok"] and b["rates_betas_ok"] and b["read_back_equal"] and b["code_decay"] == 0 and b["start_equal"] and \
        b["row0_counts_equal"] and b["digest256_equal"] and b["tensors_differ"]
    c = {}
    for label, r, d in (("cd0", rec, q), ("reference", ref, refp.parent)):
        st = T.load(d / r["start_file"], weights_only=False); en = T.load(d / r["weights"], weights_only=False)
        c[label] = dict(start=float(st["l1.weight"].abs().mean()), end=float(en["l1.weight"].abs().mean()))
    return dict(a=a, b=b, c=c, **{"pass": a["ok"] and b["ok"]})


def check8(qual, cmd):
    import subprocess
    q = Path(qual); base = [sys.executable, str(HERE / "exp57_scale.py"), "run", "in16", "matrix", "S", str(SEED), "--dose", "8192", "--code-decay", "0"]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}    # as from a shell: the harness sets these on import
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    d1, d2 = q / "uninterrupted", q / "killed_and_resumed"
    p1 = subprocess.Popen(base + ["--out", str(d1), "--ckpt", "0"], cwd=HERE, env=env, stdout=open(q.parent / "check8_uninterrupted.log", "w"), stderr=subprocess.STDOUT)
    r2 = subprocess.run(base + ["--out", str(d2), "--ckpt", "4096", "--die-after", "7000"], cwd=HERE, env=env,
                        stdout=open(q.parent / "check8_killed.log", "w"), stderr=subprocess.STDOUT)
    rp2 = d2 / "e57_in16_matrix_S_cd0_u8192_73129.json"
    mid = json.loads(rp2.read_text())
    before = dict(exit_code=r2.returncode, last_row=mid["curve"][-1]["step"], cosines=len(mid["cosines"]),
                  ckpt_exists=rp2.with_suffix(".ckpt").exists())
    r3 = subprocess.run(base + ["--out", str(d2), "--ckpt", "4096", "--resume"], cwd=HERE, env=env,
                        stdout=open(q.parent / "check8_resumed.log", "w"), stderr=subprocess.STDOUT)
    p1.wait()
    a = json.loads((d1 / "e57_in16_matrix_S_cd0_u8192_73129.json").read_text()); b = json.loads(rp2.read_text())
    strip = lambda r: {k: v for k, v in r.items() if not (k.startswith("cpu") or k.startswith("wall"))}
    rows_eq = len(a["curve"]) == len(b["curve"]) and all(strip(x) == strip(y) for x, y in zip(a["curve"], b["curve"]))
    cos_strip = lambda c: [{k: v for k, v in e.items() if k != "cpu"} for e in c]
    row4096 = next(r for r in b["curve"] if r["step"] == 4096)
    out = dict(before_resume=before, resume_exit_code=r3.returncode, uninterrupted_exit_code=p1.returncode,
               tensors_equal=a["tensors_sha256"] == b["tensors_sha256"], start_equal=a["start_tensors_sha256"] == b["start_tensors_sha256"],
               tensors_sha256=[a["tensors_sha256"], b["tensors_sha256"]], rows_equal=rows_eq,
               cosines_entries=[len(a["cosines"]), len(b["cosines"])], cosines_updates=[[c["update"] for c in a["cosines"]], [c["update"] for c in b["cosines"]]],
               cosines_equal=cos_strip(a["cosines"]) == cos_strip(b["cosines"]),
               optimizer_steps=[a["optimizer_steps"], b["optimizer_steps"]],
               read_back_equal=[a["optimizer_read_back"] == a["optimizer"], b["optimizer_read_back"] == b["optimizer"]],
               no_decay_group=[sorted(g["params"]) for g in b["optimizer"]["groups"] if g["weight_decay"] == 0.0],
               cpu_updates_last_over_4096=b["curve"][-1]["cpu_updates"] / row4096["cpu_updates"],
               resumes=[a["resumes"], b["resumes"]], ckpt_deleted=not rp2.with_suffix(".ckpt").exists())
    out["pass"] = (before["exit_code"] == 9 and before["last_row"] == 6144 and before["cosines"] == 4 and r3.returncode == 0 and p1.returncode == 0
                   and out["tensors_equal"] and out["start_equal"] and rows_eq and out["cosines_entries"] == [5, 5] and out["cosines_equal"]
                   and out["optimizer_steps"] == [[8192], [8192]] and all(out["read_back_equal"])
                   and out["no_decay_group"] == [["book.weight", "l1.bias", "l1.weight"]] and out["cpu_updates_last_over_4096"] >= 1.5
                   and len(b["resumes"]) == 1 and b["resumes"][0]["update"] == 4096 and a["resumes"] == [])
    return out


CHECKS = {"1": check1, "2": check2, "3": check3, "5": check5, "7": check7, "8": check8}


# ---------------------------------------------------------------- main
def main(argv):
    cmd = "python3 " + " ".join(Path(a).name if i == 0 else a for i, a in enumerate(argv))
    a = argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    if a[0] == "run":
        network, scale, rule, seed = a[1], a[2], a[3], int(a[4])
        assert network in NETWORKS and scale in SCALE_AT and scale != "exact" and rule in RULES and seed == SEED, "E57 runs seed 73129 only"
        d = opt("--dose", None); cd = opt("--code-decay", None); da = opt("--die-after", None)
        run(network, scale, rule, seed, Path(opt("--out", str(OUT))), cmd, dose=int(d) if d else None,
            code_decay=int(cd) if cd is not None else None, ckpt=int(opt("--ckpt", 4096)), resume="--resume" in a,
            die_after=int(da) if da else None)
    elif a[0] == "check":
        n = a[1]
        qual = Path(opt("--qual", str(QUAL))); qual.mkdir(parents=True, exist_ok=True)
        p = qual / f"check{n}_{SEED}.json"
        R_.refuse_existing(p)
        t0 = time.process_time(); w0 = time.time()
        res = CHECKS[n](qual / f"check{n}_{SEED}_runs", cmd)
        res.update(check=n, seed=SEED, cpu=time.process_time() - t0, wall=time.time() - w0, versions=versions(), provenance=provenance(cmd))
        p.write_text(json.dumps(res, indent=1, default=str))
        print(f"e57 check {n}:", "PASS" if res["pass"] else "FAIL", flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
