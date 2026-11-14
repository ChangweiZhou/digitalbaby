"""E31 runner (brief OPUS_TASK_E31_ONE_SWITCH_LIVES_20261113.md, with the addendum on arm P; worker B). A new file; every existing source
stays as it is.

A life here is exp26_candidate.life, unchanged, run from the admitted sweep birth exp45/e26/e26_dev_S_<seed>.pt with the sweep's update
(arm S: fresh AdamW, rate 0.001, decay 0.25, the E26 schedule, streams, panels and scorer), by the pattern of exp26x_cross.py: names in
exp26_candidate are rebound for the duration of the call. What is rebound:

    birth_path, admit_e26    the E26 sweep birth, admitted by exp26_candidate.admit_e26 for arm S
    life_path                exp45/e31/e31_life_<arm><DEVTAG><DRYTAG>_<seed>.json (+ .pt)
    provenance               the E26 record plus an `e31` entry (arm, runner hash, the arm's rule)
    step                     the update of the arm (below); for arm "none" it is exp26_candidate.step's text with arch31.forward(switch=None)
    json                     a thin wrapper of the json module whose dumps writes experiment="E31", the E31 arm, an `e31` entry, and the
                             arm's per-stage measures into the receipt exp26_candidate.life writes; nothing else in the record changes

Arms (arch31 for the way back):
    none  the floor's update through arch31.forward(switch=None): must equal exp26_candidate's life S (check "no switch")
    G     true weights for rows 0-511 of weight_ih_l0 and weight_hh_l0 on the way back, the sweep elsewhere
    H     true weights for weight_hh_l0 on the way back, the sweep elsewhere
    K     error across the recurrent map through Bhh (768 x 256): drawn once from U(-1/16, 1/16) by torch.Generator().manual_seed(seed + 31);
          after every optimizer step Bhh <- (1 - d) Bhh + (W1 - (1 - d) W0), d = 0.001 * 0.25, W0 and W1 the recurrent matrix before and
          after the step. Every stage row records cos(Bhh, W) and |Bhh - W| / |W|.
    P     the sweep everywhere plus  - lam * mean((z - 0.5)^2)  added to the loss, z the update gate recomputed from detached inputs at
          every position after the first byte (all positions 1..L-1 of the batch tensor), over items, positions and the 256 units;
          lam set once on the first batch so that the term's gradient norm on the update-gate rows (rows 256-511 of weight_ih_l0,
          weight_hh_l0, bias_ih_l0, bias_hh_l0) is 0.25 of the task loss's gradient norm on the same rows. Every stage row records U
          (E30: percentage of update-gate values with 0.1 <= z <= 0.9 over every item, every position after the first byte and all units)
          on the plus-minus panel (e29_gates.plus_minus_panel, e29_gates.gates).

    python3 exp31_switch.py life <none|G|H|K|P> <seed> [--out DIR] [--lambda0]     (--lambda0: arm P with lam = 0, for the check only)
    python3 exp31_switch.py check                                                    -> exp45/e31/E31_CHECK_B.json
"""
import hashlib
import json as _json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C          # noqa: E402  (imports exp20_reference, which applies the environment gate)
import torch as T                    # noqa: E402
import arch26                        # noqa: E402
import arch31                        # noqa: E402

E, X, X16, R = C.E, C.X, C.X16, C.R
OUT = HERE / "exp45" / "e31"
BIRTHS = HERE / "exp45" / "e26"
ARMS = ("none", "G", "H", "K", "P")
DECAY = C.LR * C.WD                  # 0.001 * 0.25 = 0.00025, the forward weights' own decoupled decay per step
Z = slice(256, 512)                  # update-gate rows of the stacked GRU maps
SOURCES = ("arch31.py", "exp31_switch.py", "e29_gates.py")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def own_sources():
    return {s: sha_file(HERE / s) for s in SOURCES}


# ---------------------------------------------------------------- arm K: the learned way back across the recurrent map
class Feedback:
    def __init__(self, seed, shape=(768, 256)):
        g = T.Generator().manual_seed(seed + 31)
        self.B = T.empty(*shape).uniform_(-1 / 16, 1 / 16, generator=g)
        self.B0_sha = R.tensors_sha({"Bhh": self.B})
        self.steps = 0

    def after_step(self, W0, W1):
        """Bhh <- (1 - d) Bhh + (W1 - (1 - d) W0): the forward weight's additive step plus Bhh's own decay at the same rate."""
        self.B = (1 - DECAY) * self.B + (W1 - (1 - DECAY) * W0)
        self.steps += 1

    def measures(self, W):
        W = W.detach()
        cos = float(T.nn.functional.cosine_similarity(self.B.flatten().double(), W.flatten().double(), dim=0))
        rel = float((self.B.double() - W.double()).norm() / W.double().norm())
        return dict(cos_Bhh_W=cos, rel_diff_Bhh_W=rel, feedback_updates=self.steps)


# ---------------------------------------------------------------- arm P: the local push on the update gates
def gate_term(m, out):
    """mean((z - 0.5)^2) over items, positions 1..L-1 and units, z recomputed from detached z1 and previous states (arch31.forward's)."""
    z1, hp = out["z1"].detach(), T.stack(out["hp"][1:], 1).detach()
    a = arch26.lin(z1[:, 1:], arch26.Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0), True)[..., Z] + \
        arch26.lin(hp, arch26.Mat(m.rnn.weight_hh_l0, m.rnn.bias_hh_l0), True)[..., Z]
    z = T.sigmoid(a)
    return ((z - 0.5) ** 2).mean(), dict(z=z.detach(), z1=z1[:, 1:], hp=hp)


def gate_rows(m):
    return [(m.rnn.weight_ih_l0, Z), (m.rnn.weight_hh_l0, Z), (m.rnn.bias_ih_l0, Z), (m.rnn.bias_hh_l0, Z)]


def rows_norm(grads):
    return float(T.sqrt(sum((g[s].double() ** 2).sum() for g, s in grads)))


def calibrate_lambda(m, x, y, mask):
    """lam such that, on this batch and the weights before the first update, |grad of lam*term| on the update-gate rows is 0.25 of
    |grad of the task loss| there. The task loss's gradient is the floor's (the sweep everywhere)."""
    params = [p for p, _ in gate_rows(m)]
    logits, out = arch31.forward(m, x, True, True, switch="P")
    task = E.cross_entropy(logits[mask], y[mask])
    gt = T.autograd.grad(task, params, retain_graph=True)
    term, _ = gate_term(m, out)
    gg = T.autograd.grad(term, params)
    nt = rows_norm(list(zip(gt, [Z] * 4)))
    ng = rows_norm(list(zip(gg, [Z] * 4)))
    lam = 0.25 * nt / ng
    return lam, dict(lam=lam, task_grad_norm_gate_rows=nt, term_grad_norm_gate_rows_at_lam1=ng,
                     term_grad_norm_gate_rows_at_lam=lam * ng, ratio=lam * ng / nt)


def undecided_share(m, seed):
    import e29_gates
    items = e29_gates.plus_minus_panel(seed)
    x = X.batch_tensor(items)[0]
    Zs, chk = e29_gates.gates(m, x)
    z = Zs[:, 1:, :]
    und = ((z >= 0.1) & (z <= 0.9)).double().mean()
    return dict(U_plus_minus=float(100 * und), U_items=len(items), U_values=int(z.numel()),
                U_states_bit_identical=chk["states_bit_identical_to_arch26_forward"])


# ---------------------------------------------------------------- the update of each arm
class Arm:
    def __init__(self, arm, seed, lambda0=False):
        assert arm in ARMS
        self.arm, self.seed, self.lambda0 = arm, seed, lambda0
        self.fb = Feedback(seed) if arm == "K" else None
        self.lam, self.lam_record = (0.0, dict(lam=0.0, note="--lambda0: check only")) if lambda0 else (None, None)
        self.m = None
        self.stage0 = None

    def switch(self):
        return None if self.arm == "none" else self.arm

    def step(self, m, opt, items, local):
        assert local is True, "E31 lives use the sweep's update (arm S) with one switch"
        x, y, mask = X.batch_tensor(items)
        if self.m is None:                                  # before the first update: the stage-0 measures on the birth weights
            self.m = m
            self.stage0 = self.stage_measures()
        if self.arm == "P" and self.lam is None:
            self.lam, self.lam_record = calibrate_lambda(m, x, y, mask)
            print("e31 P lambda", self.lam_record, flush=True)
        logits, out = arch31.forward(m, x, local, True, switch=self.switch(), Bhh=self.fb.B if self.fb else None)
        loss = E.cross_entropy(logits[mask], y[mask])
        total = loss
        if self.arm == "P":
            term, _ = gate_term(m, out)
            total = loss - self.lam * term
        opt.zero_grad()
        total.backward()
        if self.fb is not None:
            W0 = m.rnn.weight_hh_l0.detach().clone()
        opt.step()
        if self.fb is not None:
            self.fb.after_step(W0, m.rnn.weight_hh_l0.detach().clone())
        return float(loss.detach())

    def stage_measures(self):
        m = self.m
        if self.arm == "K":
            return self.fb.measures(m.rnn.weight_hh_l0)
        if self.arm == "P":
            return undecided_share(m, self.seed)
        return {}

    def record(self):
        rule = {"none": "arch31.forward(switch=None): the floor's update",
                "G": "true weights for rows 0-511 of weight_ih_l0 and weight_hh_l0 on the way back; sign(W)*mean|W| (whole matrix) for rows 512-767 and every other map",
                "H": "true weights for weight_hh_l0 on the way back; the sweep elsewhere",
                "K": "error across weight_hh_l0 through Bhh; Bhh0 ~ U(-1/16,1/16) from torch.Generator().manual_seed(seed+31) via Tensor.uniform_; after each step Bhh <- (1-d)Bhh + (W1-(1-d)W0), d=0.00025",
                "P": "the sweep everywhere plus -lam*mean((z-0.5)^2) over items, positions 1..L-1 of the batch tensor, and 256 units; z from detached inputs; lam set on the first batch (ratio 0.25 on rows 256-511)"}[self.arm]
        r = dict(arm=self.arm, worker="B", rule=rule, runner_sources_sha256=own_sources())
        if self.fb is not None:
            r.update(Bhh_seed=self.seed + 31, Bhh_start_tensors_sha256=self.fb.B0_sha, feedback_decay=DECAY)
        if self.arm == "P":
            r.update(gate_push=self.lam_record)
        return r


class JsonShim:
    """exp26_candidate.life's json, with dumps rewriting the E26 life receipt into the E31 one (experiment, arm, e31 entry, stage measures)."""

    def __init__(self, armobj):
        self.a = armobj
        self.done = set()

    def __getattr__(self, k):
        return getattr(_json, k)

    def dumps(self, obj, **kw):
        if isinstance(obj, dict) and obj.get("experiment") == "E26" and obj.get("kind") == "life":
            st = obj["stages"]
            for s in st:
                if s["stage"] in self.done:
                    continue
                s["e31"] = self.a.stage0 if s["stage"] == 0 else self.a.stage_measures()
                self.done.add(s["stage"])
            obj = dict(obj, experiment="E31", arm=self.a.arm, update_arm="S", e31=self.a.record())
            if self.a.arm == "K" and "complete" in obj:
                obj["e31"]["Bhh_final_tensors_sha256"] = R.tensors_sha({"Bhh": self.a.fb.B})
        return _json.dumps(obj, **kw)


def life_path(out, arm, seed):
    return Path(out) / f"e31_life_{arm}{E.DEVTAG}{E.DRYTAG}_{seed}.json"


def run_life(arm, seed, out, cmd, lambda0=False):
    out = Path(out)
    a = Arm(arm, seed, lambda0)
    tag = arm + ("_lam0" if lambda0 else "")
    orig = dict(birth_path=C.birth_path, admit_e26=C.admit_e26, life_path=C.life_path, provenance=C.provenance, step=C.step, json=C.json)
    C.birth_path = lambda _out, _arm, s: orig["birth_path"](BIRTHS, "S", s)
    C.admit_e26 = lambda bfile, _arm, s: orig["admit_e26"](bfile, "S", s)
    C.life_path = lambda _out, _arm, s: life_path(_out, tag, s)
    C.provenance = lambda c: dict(orig["provenance"](c), e31=dict(arm=arm, lambda0=lambda0, runner="exp31_switch.py", runner_sources_sha256=own_sources()))
    C.step = a.step
    C.json = JsonShim(a)
    try:
        C.life("S", seed, out, cmd)
    finally:
        for k, v in orig.items():
            setattr(C, k, v)
    if a.fb is not None:
        p = life_path(out, tag, seed)
        T.save({"Bhh": a.fb.B}, p.with_name(p.stem + "_Bhh.pt"))


# ================================================================ stage 1 checks (worker B)
def birth_model(seed):
    arch26.install()
    bfile = C.birth_path(BIRTHS, "S", seed)
    m, sha, _ = C.admit_e26(bfile, "S", seed)
    m.load_state_dict(T.load(bfile))
    return m, sha


def first_batches(seed, n):
    """The life's own first n batches (stage F1.1: the F1 entry stream and the joint stream, as exp26_candidate.life builds them)."""
    entries = X16.family_entries("F1", seed)
    joint = X16.joint_stream([X16.entry_stream(entries, seed + 100 + 1)], seed + 1)
    return [[next(joint) for _ in range(X.BATCH)] for _ in range(n)]


def maxrel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def grads(m, x, y, mask, **kw):
    m.zero_grad()
    logits, out = arch31.forward(m, x, True, True, **kw)
    E.cross_entropy(logits[mask], y[mask]).backward()
    return logits.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters()}


def check_no_switch(m, x, y, mask):
    m.zero_grad()
    lo_ref = arch26.forward(m, x, True, True)[0]
    E.cross_entropy(lo_ref[mask], y[mask]).backward()
    g_ref = {n: p.grad.detach().clone() for n, p in m.named_parameters()}
    lo, g = grads(m, x, y, mask, switch=None)
    rel = {n: maxrel(g[n], g_ref[n]) for n in g}
    worst = max([maxrel(lo, lo_ref.detach())] + list(rel.values()))
    return dict(logits_max_rel=maxrel(lo, lo_ref.detach()), grads_max_rel=rel, worst=worst, ok=worst <= 1e-6)


def crossing_errors(m, x, y, mask, switch, Bhh=None, ts=None):
    """On one batch: for chosen steps t, the error the recurrent map sends back to h_{t-1}, given the real error g arriving at its output."""
    m.zero_grad()
    logits, out = arch31.forward(m, x, True, True, keep=True, switch=switch, Bhh=Bhh)
    E.cross_entropy(logits[mask], y[mask]).backward()
    L = x.shape[1]
    ts = ts or (1, L // 2, L - 1)
    cross = arch31.recurrent_cross(m, True, switch, Bhh)
    res = []
    for t in ts:
        g = out["gh"][t].grad.detach()
        hp = out["hp"][t].detach().requires_grad_()
        gx, = T.autograd.grad(cross(hp), hp, g)
        res.append((t, g, gx))
    return res


def check_G_H(m, x, y, mask):
    W = m.rnn.weight_hh_l0.detach()
    S = W.sign() * W.abs().mean()
    out = {}
    for sw, expect in (("G", lambda g: g[:, :512] @ W[:512] + g[:, 512:] @ S[512:]), ("H", lambda g: g @ W)):
        rows = [dict(t=t, max_rel=maxrel(gx, expect(g)), max_rel_vs_sweep=maxrel(gx, g @ S)) for t, g, gx in crossing_errors(m, x, y, mask, sw)]
        out[sw] = dict(steps=rows, worst=max(r["max_rel"] for r in rows), ok=max(r["max_rel"] for r in rows) <= 1e-6)
    # arm G's input map, the same rule on weight_ih_l0 (not required by the brief; recorded)
    Wi = m.rnn.weight_ih_l0.detach(); Si = Wi.sign() * Wi.abs().mean()
    z1 = arch31.forward(m, x, True, True)[1]["z1"].detach().requires_grad_()
    gi = arch31.input_map(m, True, "G")(z1)
    gg = T.randn(gi.shape, generator=T.Generator().manual_seed(1))
    gx, = T.autograd.grad(gi, z1, gg)
    out["G_input_map"] = dict(max_rel=maxrel(gx, gg[..., :512] @ Wi[:512] + gg[..., 512:] @ Si[512:]), note="random g, real z1")
    return out


def check_K(seed, batches):
    res = {}
    # (a) the starting Bhh does not depend on the birth's recurrent matrix
    m, _ = birth_model(seed)
    b_a = Feedback(seed).B.clone()
    with T.no_grad():
        m.rnn.weight_hh_l0.mul_(-3.0).add_(0.5)
    b_b = Feedback(seed).B.clone()
    m2, _ = birth_model(seed + 0)
    res["start_independent_of_birth_W"] = dict(equal=bool(T.equal(b_a, b_b)), Bhh0_tensors_sha256=R.tensors_sha({"Bhh": b_a}),
                                               note="Feedback(seed) is built twice, the second time after the loaded birth's weight_hh_l0 was changed in place")
    # (b) the update identity at updates 1, 10 and 100, along the life's own first 100 updates of arm K
    m, _ = birth_model(seed)
    opt = C.make_opt(m)
    a = Arm("K", seed); a.m = m; a.stage0 = {}
    rows = []
    for u, items in enumerate(batches[:100], 1):
        Bold = a.fb.B.clone(); W0 = m.rnn.weight_hh_l0.detach().clone()
        a.step(m, opt, items, True)
        W1 = m.rnn.weight_hh_l0.detach().clone(); Bnew = a.fb.B
        if u in (1, 10, 100):
            ident = float((Bnew - ((1 - DECAY) * Bold + (W1 - (1 - DECAY) * W0))).abs().max())
            contr = float(((Bnew - W1) - (1 - DECAY) * (Bold - W0)).abs().max())
            rows.append(dict(update=u, identity_max_abs=ident, contraction_max_abs=contr, **a.fb.measures(W1)))
    res["update_identity"] = dict(rows=rows, ok=all(r["identity_max_abs"] <= 1e-7 and r["contraction_max_abs"] <= 1e-7 for r in rows))
    # (c) with Bhh = W by hand, K's gradients equal H's
    m, _ = birth_model(seed)
    x, y, mask = X.batch_tensor(batches[0])
    _, gH = grads(m, x, y, mask, switch="H")
    _, gK = grads(m, x, y, mask, switch="K", Bhh=m.rnn.weight_hh_l0.detach().clone())
    rel = {n: maxrel(gK[n], gH[n]) for n in gH}
    res["K_equals_H_when_Bhh_is_W"] = dict(grads_max_rel=rel, worst=max(rel.values()), ok=max(rel.values()) <= 1e-6)
    return res


def check_P(m, x, y, mask):
    res = {}
    logits, out = arch31.forward(m, x, True, True, switch="P")
    term, aux = gate_term(m, out)
    m.zero_grad()
    term.backward()
    outside = {}
    for n, p in m.named_parameters():
        g = p.grad if p.grad is not None else T.zeros_like(p)
        if n in ("rnn.weight_ih_l0", "rnn.weight_hh_l0", "rnn.bias_ih_l0", "rnn.bias_hh_l0"):
            gz = g.clone(); gz[Z] = 0
            outside[n] = float(gz.abs().max())
        else:
            outside[n] = float(g.abs().max())
    res["term_grad_outside_update_gate_rows"] = dict(max_abs=outside, ok=all(v == 0.0 for v in outside.values()))
    z = aux["z"]
    N = z.numel()
    f = 2.0 * (z - 0.5) * z * (1 - z) / N                         # d term / d a, per item, position and unit
    f2 = f.reshape(-1, 256).double()
    exp = {"rnn.weight_ih_l0": f2.T @ aux["z1"].reshape(-1, aux["z1"].shape[-1]).double(),
           "rnn.weight_hh_l0": f2.T @ aux["hp"].reshape(-1, 256).double(),
           "rnn.bias_ih_l0": f2.sum(0), "rnn.bias_hh_l0": f2.sum(0)}
    pr = dict(m.named_parameters())
    rel = {n: maxrel(pr[n].grad[Z].double(), e) for n, e in exp.items()}
    res["term_grad_equals_two_factor_product"] = dict(max_rel=rel, ok=max(rel.values()) <= 1e-6,
                                                      note="d/dW[j,k] of mean((z-.5)^2) = sum over items and positions of input_k * 2 (z_j - .5) z_j (1 - z_j) / N")
    lam, rec = calibrate_lambda(m, x, y, mask)
    res["lambda_on_first_batch"] = rec
    return res


def run_checks(out):
    import platform
    seed = 73129
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    p = out / "E31_CHECK_B.json"
    R.refuse_existing(p)
    batches = first_batches(seed, 100)
    m, birth_sha = birth_model(seed)
    x, y, mask = X.batch_tensor(batches[0])
    rec = dict(experiment="E31", kind="check", worker="B", seed=seed, birth_sha256=birth_sha,
               batch_digest=hashlib.sha256(b"".join(t.contiguous().numpy().tobytes() for t in (x, y, mask))).hexdigest(),
               host=R.host(), runner_sources_sha256=own_sources(), e26_sources_sha256=C.source_hashes(), git=C.git_state())
    rec["no_switch"] = check_no_switch(m, x, y, mask)
    rec["G_H_crossing"] = check_G_H(m, x, y, mask)
    rec["K"] = check_K(seed, batches)
    m, _ = birth_model(seed)
    rec["P"] = check_P(m, x, y, mask)
    oks = [rec["no_switch"]["ok"], rec["G_H_crossing"]["G"]["ok"], rec["G_H_crossing"]["H"]["ok"], rec["K"]["start_independent_of_birth_W"]["equal"],
           rec["K"]["update_identity"]["ok"], rec["K"]["K_equals_H_when_Bhh_is_W"]["ok"], rec["P"]["term_grad_outside_update_gate_rows"]["ok"],
           rec["P"]["term_grad_equals_two_factor_product"]["ok"]]
    rec["pass"] = all(oks)
    rec["note"] = "The dry-life checks (no switch against exp26_candidate.py life S; P at lam = 0 against no switch) are in E31_DRY_B.json."
    p.write_text(_json.dumps(rec, indent=1))
    print(_json.dumps({k: rec[k] for k in ("pass",)}), "->", p, flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *args])
    out = Path(args[args.index("--out") + 1]) if "--out" in args else OUT
    if args[0] == "life":
        run_life(args[1], int(args[2]), out, cmd, lambda0="--lambda0" in args)
    elif args[0] == "check":
        run_checks(out)
    else:
        raise SystemExit(__doc__)
