"""E31 runner (brief OPUS_TASK_E31_ONE_SWITCH_LIVES_20261113.md with its addendum; worker A). A new file; every existing source unchanged.

A life here is exp26_candidate.life, unchanged, started from the admitted sweep birth exp45/e26/e26_dev_S_<seed>.pt, with one difference:
the update. exp26_candidate.step is rebound for the duration of the call to E31's step, which differs from it only in the backward pass
(arch31.forward's switch) and, for arm P, one extra loss term; and, for arm K, the after-step update of Bhh. The receipt is rewritten on its
way to disk (experiment "E31", the E31 arm, the switch's records, per-stage rows). Same schedule, streams, panels, scorer, fresh AdamW at
rate 0.001 and decay 0.25, 36,864 updates. By the pattern of exp26x_cross.py (names in exp26_candidate rebound inside try/finally).

Arms: N  no switch (checks only; must equal exp26_candidate life S)          G, H, K  arch31's switches
      P  the floor plus -lambda * mean((z - 0.5)^2) on the update gates       P0  P with lambda = 0 (checks only; must equal N)

    python3 exp31_switch.py life <N|G|H|K|P|P0> <seed> [--out DIR]     -> DIR/e31_life_<arm><DEVTAG><DRYTAG>_<seed>.json (+ .pt)
    python3 exp31_switch.py check <seed>                                -> exp45/e31/E31_CHECKS_A_<seed>.json (one-batch checks)
    python3 exp31_switch.py dry_summary                                  -> exp45/e31/E31_DRY_A.json (the dry receipts' checks and hashes)

Choices the brief leaves open, made here and recorded in every receipt (worker A):
  * G: gx = g[..., :512] @ W[:512] + g[..., 512:] @ S[512:], S = sign(W) * mean|W| over the whole matrix, for weight_ih_l0 and weight_hh_l0.
    H: gx = g @ W. K: gx = g @ Bhh. Weight and bias gradients in every switched map: SignLinear's expressions.
  * Bhh0 = torch.empty(768, 256).uniform_(-1/16, 1/16, generator=torch.Generator().manual_seed(seed + 31)), float32.
    After each step: Bhh = (1 - 0.00025) * Bhh + (W1 - (1 - 0.00025) * W0), in that order.
  * P: z_t = sigmoid(lin(z1.detach(), ih, True)[:, t, 256:512] + lin(hp_t.detach(), hh, True)[:, 256:512]) for t = 1 .. L-1 (hp_t the state
    step t reads); the mean over the positions whose input byte is not BOS (BOS is only at position 0 and in padding), and all 256 units.
    loss_total = task_loss + (-lambda) * mean. lambda = 0.25 * |g_task| / |g_term(lambda=1)| on the first life batch, each norm over the
    concatenation of rows 256-511 of weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0 (in that order), the task gradient by the floor's
    path; computed by torch.autograd.grad before the first update, float32 norms, then fixed.
  * U (arm P's stage rows): E30's measure on e29_gates.plus_minus_panel(seed): the share of z with 0.1 <= z <= 0.9 over every item, every
    position after the first, all 256 units, z from e29_gates.gates (exact forward values).
"""
import hashlib
import json as _json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C          # noqa: E402  (imports exp20_reference, which applies the environment gate)
import exp11_cell as E               # noqa: E402
import torch as T                    # noqa: E402
import arch26, arch31                # noqa: E402
from exp3_state import lin           # noqa: E402

X16, X, R = C.X16, C.X, C.R
OUT = HERE / "exp45" / "e31"
BIRTHS = HERE / "exp45" / "e26"
ARMS31 = {"N": None, "G": "G", "H": "H", "K": "K", "P": None, "P0": None}
DECAY = 0.00025                       # 0.001 * 0.25, the forward weights' own decay per update
ZROWS = slice(256, 512)
PROWS = ("rnn.weight_ih_l0", "rnn.weight_hh_l0", "rnn.bias_ih_l0", "rnn.bias_hh_l0")
CHOICES = __doc__.split("Choices the brief leaves open, made here and recorded in every receipt (worker A):")[1].strip()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def init_bhh(seed):
    """The starting Bhh: depends on the seed only (never on weight_hh_l0)."""
    return T.empty(768, 256).uniform_(-1 / 16, 1 / 16, generator=T.Generator().manual_seed(seed + 31))


def penalty(m, it, x):
    """mean((z - 0.5)^2) over items, positions after the first byte (input not BOS) and units, z the update gate from detached inputs."""
    ih = arch26.Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0); hh = arch26.Mat(m.rnn.weight_hh_l0, m.rnn.bias_hh_l0)
    GI = lin(it["z1"].detach(), ih, True)
    zs = []
    for t in range(1, x.shape[1]):
        a = GI[:, t, ZROWS] + lin(it["hp"][t].detach(), hh, True)[:, ZROWS]
        zs.append(T.sigmoid(a))
    Z = T.stack(zs, 1)
    msk = x[:, 1:] != X.BOS
    return ((Z - 0.5) ** 2)[msk].mean(), Z, msk


def rows_vec(m, grads):
    named = dict(m.named_parameters())
    return T.cat([g[ZROWS].flatten() for n, g in zip(PROWS, grads)]), named


def undecided_share(m, seed):
    import e29_gates
    items = e29_gates.plus_minus_panel(seed)
    x, _, _ = X.batch_tensor(items)
    Z, chk = e29_gates.gates(m, x)
    z = Z[:, 1:, :]
    return dict(U_percent=100.0 * float(((z >= 0.1) & (z <= 0.9)).double().mean()), items=len(items), positions_per_item=int(z.shape[1]),
                units=int(z.shape[2]), states_bit_identical=chk["states_bit_identical_to_arch26_forward"])


class Switch:
    """The E31 update for one arm of one life: the step, Bhh (arm K), lambda (arm P), the per-stage rows."""

    def __init__(self, arm, seed, m=None):
        assert arm in ARMS31
        self.arm, self.seed, self.switch = arm, seed, ARMS31[arm]
        self.n = 0; self.extras = {}; self.identity = {}; self.lam = None; self.lam_record = None
        self.Bhh = init_bhh(seed) if arm == "K" else None
        self.Bhh0_sha = (hashlib.sha256(self.Bhh.numpy().tobytes()).hexdigest() if self.Bhh is not None else None)
        if arm == "P0":
            self.lam = 0.0; self.lam_record = dict(lam=0.0, note="arm P0: lambda fixed at 0 (check: must equal arm N bit for bit)")

    def stage_row(self, m):
        r = dict(updates=self.n)
        if self.arm == "K":
            W = m.rnn.weight_hh_l0.detach(); Bv, Wv = self.Bhh.flatten(), W.flatten()
            r.update(cosine_Bhh_W=float((Bv * Wv).sum()) / (float(Bv.norm()) * float(Wv.norm())), rel_diff=float((self.Bhh - W).norm()) / float(W.norm()))
        if self.arm in ("P", "P0"):
            r.update(lam=self.lam, **undecided_share(m, self.seed))
        return r

    def calibrate(self, m, x, y, mask):
        params = [dict(m.named_parameters())[n] for n in PROWS]
        logits, it = arch31.forward(m, x, None)
        loss = E.cross_entropy(logits[mask], y[mask])
        gt = T.autograd.grad(loss, params, retain_graph=True)
        pen, _, _ = penalty(m, it, x)
        gp = T.autograd.grad(-1.0 * pen, params)
        nt, _ = rows_vec(m, gt); npn, _ = rows_vec(m, gp)
        nt, npn = float(nt.norm()), float(npn.norm())
        self.lam = 0.25 * nt / npn
        self.lam_record = dict(lam=self.lam, task_grad_norm_rows=nt, term_grad_norm_rows_at_lambda_1=npn,
                               term_grad_norm_rows_at_lambda=self.lam * npn, ratio=self.lam * npn / nt, batch="the first life batch, before the first update")

    def step(self, m, opt, items, local):
        assert local is True, "E31 lives are sweep lives with one switch"
        x, y, mask = X.batch_tensor(items)
        if self.n == 0:
            if self.arm == "P":
                self.calibrate(m, x, y, mask)
            self.extras[0] = self.stage_row(m)
        logits, it = arch31.forward(m, x, self.switch, self.Bhh)
        loss = E.cross_entropy(logits[mask], y[mask])
        total = loss
        if self.arm in ("P", "P0"):
            pen, _, _ = penalty(m, it, x)
            total = loss + (-self.lam) * pen
        opt.zero_grad()
        total.backward()
        if self.arm == "K":
            W0 = m.rnn.weight_hh_l0.detach().clone(); B0 = self.Bhh.clone()
        opt.step()
        self.n += 1
        if self.arm == "K":
            W1 = m.rnn.weight_hh_l0.detach().clone()
            self.Bhh = (1 - DECAY) * self.Bhh + (W1 - (1 - DECAY) * W0)
            if self.n in (1, 10, 100):        # the identity, checked in its difference form: Bhh - W shrinks by (1 - 0.00025)
                d = float(((self.Bhh - W1) - (1 - DECAY) * (B0 - W0)).abs().max())
                self.identity[str(self.n)] = dict(max_abs_residual=d, holds_to_1e_7=d <= 1e-7)
        if self.n % E.STAGE_UPDATES == 0:
            self.extras[self.n // E.STAGE_UPDATES] = self.stage_row(m)
        return float(loss.detach())

    def summary(self):
        return dict(arm=self.arm, switch=self.switch, birth_arm="S", worker="A", choices=CHOICES, updates_so_far=self.n, admission=ADMISSIONS,
                    Bhh=(dict(init="uniform(-1/16, 1/16), torch.Generator().manual_seed(seed + 31)", shape=[768, 256], init_sha256=self.Bhh0_sha,
                              decay=DECAY, identity_checks=self.identity) if self.arm == "K" else None),
                    lam=self.lam_record)


class JsonShim:
    """exp26_candidate's json, with the life receipt rewritten on its way to disk."""

    def __init__(self, sw, me):
        self.sw, self.me = sw, me

    def __getattr__(self, k):
        return getattr(_json, k)

    def dumps(self, obj, **kw):
        if isinstance(obj, dict) and obj.get("kind") == "life" and "stages" in obj:
            stages = []
            for s in obj["stages"]:
                s = dict(s)
                if s["stage"] in self.sw.extras:
                    s["e31"] = self.sw.extras[s["stage"]]
                if s["stage"] == 0:
                    s["settings"] = dict(s["settings"], update=f"exp31_switch.Switch.step, arm {self.sw.arm} (the sweep with arch31 switch {self.sw.switch})")
                stages.append(s)
            obj = dict(obj, experiment="E31", arm=self.sw.arm, stages=stages, e31=self.sw.summary())
        return _json.dumps(obj, **kw)


EXP16_AT_BIRTH = ("a301ae8e", "f4559971a6c1770523d641431fe5d4bd56def3f5ee3cafc87a5b869509be50ae")
EXP16_NOW = ("deea398c", "c9fcedcfe5f53ff7ac480e020caa5de8fc1a5163a1fd3d276b402132d73e6350")
ADMISSIONS = []


def admit_s(bfile, seed, orig_admit):
    """exp26_candidate.admit_e26 for the sweep birth, with one exception, recorded: the E26 births were made with exp16_integrated.py at
    a301ae8e; the branch now has it at deea398c (E21b: resume and receipt-writing machinery in exp16's own life/develop17, which
    exp26_candidate.life does not call). Every other check of admit_e26 runs unchanged; only that one file's hash may differ, and only as
    that exact pair. The no-switch dry life is then checked bit for bit against a dry life made on the births' own sources."""
    try:
        r = orig_admit(bfile, "S", seed); ADMISSIONS.append(dict(file=str(bfile), exception=None)); return r
    except AssertionError as e:
        if "made by other sources: ['exp16_integrated.py']" not in str(e):
            raise
    rec = _json.loads(Path(bfile).with_suffix(".json").read_text())["provenance"]["sources_sha256"]
    now = C.source_hashes()
    assert rec["exp16_integrated.py"] == EXP16_AT_BIRTH[1] and now["exp16_integrated.py"] == EXP16_NOW[1], "exp16_integrated.py differs in an unexpected way"
    keep = C.source_hashes
    C.source_hashes = lambda: dict(keep(), **{"exp16_integrated.py": rec["exp16_integrated.py"]})
    try:
        r = orig_admit(bfile, "S", seed)
    finally:
        C.source_hashes = keep
    ADMISSIONS.append(dict(file=str(bfile), exception="exp16_integrated.py at deea398c (c9fcedcf) instead of the birth's a301ae8e (f4559971); every other admission check passed"))
    return r[0], r[1], r[2] + "; exception: exp16_integrated.py a301ae8e -> deea398c (E21b resume machinery only), recorded by exp31_switch.admit_s"


def e31_life(arm, seed, out, cmd):
    out = Path(out)
    sw = Switch(arm, seed)
    me = dict(runner="exp31_switch.py", runner_sha256=sha(__file__), arch31_sha256=sha(HERE / "arch31.py"), e29_gates_sha256=sha(HERE / "e29_gates.py"))
    orig = (C.birth_path, C.admit_e26, C.life_path, C.provenance, C.step, C.json)
    C.birth_path = lambda _out, _arm, s: orig[0](BIRTHS, "S", s)
    C.admit_e26 = lambda bfile, _arm, s: admit_s(bfile, s, orig[1])
    C.life_path = lambda _out, _arm, s: Path(_out) / f"e31_life_{arm}{E.DEVTAG}{E.DRYTAG}_{s}.json"
    C.provenance = lambda c: dict(orig[3](c), e31=me)
    C.step = sw.step
    C.json = JsonShim(sw, me)
    try:
        C.life("S", seed, out, cmd)
    finally:
        C.birth_path, C.admit_e26, C.life_path, C.provenance, C.step, C.json = orig


# ================================================================ stage 1: one-batch checks
def life_birth_and_batch(seed):
    """The admitted sweep birth and the first batch of its life, built as exp26_candidate.life builds it."""
    arch26.install()
    bfile = BIRTHS / f"e26_dev_S_{seed}.pt"
    m, bsha, _ = admit_s(bfile, seed, C.admit_e26)
    m.load_state_dict(T.load(bfile))
    entries = X16.family_entries("F1", seed)
    joint = X16.joint_stream([X16.entry_stream(entries, seed + 100 + 1)], seed + 1)
    batch = [next(joint) for _ in range(X.BATCH)]
    return m, bsha, batch


def grads(m, fwd, x, y, mask, extra=None):
    m.zero_grad()
    logits, it = fwd()
    loss = E.cross_entropy(logits[mask], y[mask])
    if extra is not None:
        loss = loss + extra(it)
    loss.backward()
    return logits.detach(), {n: (p.grad.detach().clone() if p.grad is not None else T.zeros_like(p)) for n, p in m.named_parameters()}


def maxrel(ga, gb):
    return max(float((ga[n] - gb[n]).abs().max()) / max(float(gb[n].abs().max()), 1e-30) for n in gb)


def check(seed):
    out = dict(seed=seed, worker="A", runner_sha256=sha(__file__), arch31_sha256=sha(HERE / "arch31.py"))
    m, bsha, batch = life_birth_and_batch(seed)
    out["birth_sha256"] = bsha
    x, y, mask = X.batch_tensor(batch)
    out["batch"] = dict(items=len(batch), shape=list(x.shape), padded_positions=int((x[:, 1:] == X.BOS).sum()))
    # 1. no switch: arch31 equals arch26.forward(m, x, True, True)
    l26, g26 = grads(m, lambda: arch26.forward(m, x, True, True), x, y, mask)
    l31, g31 = grads(m, lambda: arch31.forward(m, x, None), x, y, mask)
    d_log = float((l26 - l31).abs().max()) / float(l26.abs().max()); d_g = maxrel(g31, g26)
    out["no_switch"] = dict(logits_max_rel=d_log, grads_max_rel=d_g, bit_identical=bool(T.equal(l26, l31) and all(T.equal(g26[n], g31[n]) for n in g26)),
                            ok=d_log <= 1e-6 and d_g <= 1e-6)
    # 2. G and H: the error sent back across the recurrent map (and, for G, the input map), every step, against the written formula
    res = {}
    for sw in ("G", "H"):
        arch31.DEBUG = []
        grads(m, lambda: arch31.forward(m, x, sw), x, y, mask)
        rec, arch31.DEBUG = arch31.DEBUG, None
        Wh, Wi = m.rnn.weight_hh_l0.detach(), m.rnn.weight_ih_l0.detach()
        Sh, Si = Wh.sign() * Wh.abs().mean(), Wi.sign() * Wi.abs().mean()
        worst, worst_sweep, n = 0.0, 0.0, {"hh": 0, "ih": 0}
        for tag, g, gx in rec:
            W, S = (Wh, Sh) if tag == "hh" else (Wi, Si)
            want = (g[..., :512] @ W[:512] + g[..., 512:] @ S[512:]) if sw == "G" else (g @ W)
            sweep = g @ S
            scale = max(float(want.abs().max()), 1e-30)
            worst = max(worst, float((gx - want).abs().max()) / scale); worst_sweep = max(worst_sweep, float((gx - sweep).abs().max()) / scale); n[tag] += 1
        res[sw] = dict(maps_checked=n, max_rel_vs_formula=worst, max_rel_vs_full_sweep=worst_sweep,
                       ok=worst <= 1e-6 and worst_sweep > 1e-3 and n["hh"] == x.shape[1] and n["ih"] == (1 if sw == "G" else 0))
    out["G_H"] = res
    # 3. K: the starting Bhh does not depend on the birth's recurrent matrix; with Bhh = W, K's gradients equal H's
    sw1 = Switch("K", seed, m)
    m2, _, _ = life_birth_and_batch(seed)
    with T.no_grad():
        m2.rnn.weight_hh_l0.mul_(-3.0).add_(0.5)
    sw2 = Switch("K", seed, m2)
    _, gH = grads(m, lambda: arch31.forward(m, x, "H"), x, y, mask)
    _, gK = grads(m, lambda: arch31.forward(m, x, "K", m.rnn.weight_hh_l0.detach().clone()), x, y, mask)
    _, gK0 = grads(m, lambda: arch31.forward(m, x, "K", init_bhh(seed)), x, y, mask)
    out["K"] = dict(start_unchanged_when_birth_W_changed=bool(T.equal(sw1.Bhh, sw2.Bhh)), Bhh0_sha256=sw1.Bhh0_sha,
                    Bhh0_range=[float(sw1.Bhh.min()), float(sw1.Bhh.max())],
                    K_with_Bhh_eq_W_vs_H_max_rel=maxrel(gK, gH), K_with_Bhh0_vs_H_max_rel=maxrel(gK0, gH),
                    identity_at_updates_1_10_100="recorded by the K dry life (receipt e31.Bhh.identity_checks); gathered by dry_summary")
    out["K"]["ok"] = out["K"]["start_unchanged_when_birth_W_changed"] and out["K"]["K_with_Bhh_eq_W_vs_H_max_rel"] <= 1e-6 and out["K"]["K_with_Bhh0_vs_H_max_rel"] > 1e-3
    # 4. P: the term's gradient lies only in the update-gate rows, and equals the written two-factor product
    lam = 0.37
    def term(it):
        pen, Z, msk = penalty(m, it, x); term.Z, term.msk = Z, msk
        return (-lam) * pen
    _, g0 = grads(m, lambda: arch31.forward(m, x, None), x, y, mask)
    _, g1 = grads(m, lambda: arch31.forward(m, x, None), x, y, mask, extra=term)
    gt = {n: g1[n] - g0[n] for n in g0}          # the term's own gradient, by difference
    m.zero_grad(); logits, it = arch31.forward(m, x, None); pen, Z, msk = penalty(m, it, x); ((-lam) * pen).backward()
    gterm = {n: (p.grad.detach().clone() if p.grad is not None else T.zeros_like(p)) for n, p in m.named_parameters()}
    outside = {}
    for n, g in gterm.items():
        if n in PROWS:
            mk = T.ones_like(g, dtype=T.bool); mk[ZROWS] = False
            outside[n] = float(g[mk].abs().max())
        else:
            outside[n] = float(g.abs().max())
    # written product: d(-lam * mean (z - .5)^2)/da = -lam * 2 (z - .5) z (1 - z) / N at counted entries; each weight: its input times that
    Zd = Z.detach().double(); N = int(msk.sum()) * Zd.shape[2]
    ga = (-lam * 2.0 * (Zd - 0.5) * Zd * (1 - Zd) / N) * msk.unsqueeze(-1).double()          # (B, L-1, 256)
    z1 = it["z1"].detach().double()[:, 1:]; hp = T.stack([h.detach() for h in it["hp"][1:]], 1).double()
    want = {"rnn.weight_ih_l0": T.einsum("btu,bti->ui", ga, z1), "rnn.weight_hh_l0": T.einsum("btu,bti->ui", ga, hp),
            "rnn.bias_ih_l0": ga.sum((0, 1)), "rnn.bias_hh_l0": ga.sum((0, 1))}
    prod = {n: float((gterm[n][ZROWS].double() - want[n]).abs().max()) / max(float(want[n].abs().max()), 1e-30) for n in want}
    out["P"] = dict(lam_used_for_check=lam, outside_update_gate_rows_max_abs=outside, outside_all_zero=all(v == 0.0 for v in outside.values()),
                    two_factor_product_max_rel=prod, term_by_difference_vs_direct_max_rel=maxrel({n: gt[n] for n in PROWS}, {n: gterm[n] for n in PROWS}),
                    counted_entries=N, masked_positions=int(msk.sum()))
    out["P"]["ok"] = out["P"]["outside_all_zero"] and max(prod.values()) <= 1e-6
    out["pass"] = all(v.get("ok", True) for v in (out["no_switch"], out["K"], out["P"])) and all(v["ok"] for v in res.values())
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"E31_CHECKS_A_{seed}.json"; p.write_text(_json.dumps(out, indent=1)); print(_json.dumps({k: (v.get("ok") if isinstance(v, dict) and "ok" in v else v) for k, v in out.items() if k != "G_H"}, indent=1), {k: v["ok"] for k, v in res.items()})
    return out["pass"]


def dry_summary(seed=73129, ref=None):
    """Gather the dry receipts: hashes, the bit-for-bit checks (N against exp26_candidate's dry life, P0 against N), K's identity rows."""
    tag = "_dry256"
    rd = {a: _json.loads((OUT / f"e31_life_{a}{tag}_{seed}.json").read_text()) for a in ARMS31 if (OUT / f"e31_life_{a}{tag}_{seed}.json").exists()}
    hashes = {a: r["complete"]["tensors_sha256"] for a, r in rd.items() if r.get("complete")}
    refr = _json.loads(Path(ref).read_text()) if ref else None
    out = dict(seed=seed, dose="E11_STAGE_UPDATES=256", worker="A", tensors_sha256=hashes,
               reference_exp26_candidate_life_S=(dict(file=str(ref), tensors_sha256=refr["complete"]["tensors_sha256"], experiment=refr["experiment"]) if refr else None))
    out["N_equals_exp26_candidate_life_S"] = (hashes.get("N") == refr["complete"]["tensors_sha256"]) if refr else None
    out["P0_equals_N"] = hashes.get("P0") == hashes.get("N")
    if "K" in rd:
        out["K_identity_checks"] = rd["K"]["e31"]["Bhh"]["identity_checks"]
        out["K_stage_rows"] = {s["stage"]: s.get("e31") for s in rd["K"]["stages"]}
    if "P" in rd:
        out["P_lambda"] = rd["P"]["e31"]["lam"]; out["P_stage_rows"] = {s["stage"]: s.get("e31") for s in rd["P"]["stages"]}
    out["for_agreement"] = {a: hashes.get(a) for a in ("G", "H", "K", "P")}
    (OUT / "E31_DRY_A.json").write_text(_json.dumps(out, indent=1)); print(_json.dumps(out, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    T.set_num_threads(1)
    if a[0] == "life":
        cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])
        out = Path(a[a.index("--out") + 1]) if "--out" in a else OUT
        e31_life(a[1], int(a[2]), out, cmd)
    elif a[0] == "check":
        sys.exit(0 if check(int(a[1])) else 1)
    elif a[0] == "dry_summary":
        dry_summary(ref=(a[a.index("--ref") + 1] if "--ref" in a else None))
    else:
        raise SystemExit(__doc__)
