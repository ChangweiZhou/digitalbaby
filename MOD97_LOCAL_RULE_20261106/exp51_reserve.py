"""E51, Part A (brief TASK_E51_A_NEW_TASK_LEARNED_IN_CELLS_HELD_IN_RESERVE_20261117.md, revision 2): a new task learned in cells held in
reserve. E41's half-width birth (H = 256, U = 128, half of the pairs taught) is placed inside a network of the present size (H = 512,
U = 256), so that cells 128..255 and units 256..511 of l1 and of z2 are in reserve. Four arms, two tasks, 4,096 updates:

    open         the start with a bias (the reserve cells' update gates start shut by a bias of 10); everything may change, as in E43
    rehearse     the same start; each item drawn with equal chance from the new task's stream or the first family's taught half
    keyed-free   the keyed start (the forward pass holds the reserve at zero on every item whose first byte is a byte in use); no rule
    keyed        the keyed start and the rule: changes of the old rows are projected away from the directions in use, l1, the old
                 columns of l3, l3's bias and the 46 rows of the byte table in use are restored after every step

    tasks: unrelated (e43_task.py)   conflict (the second family, modulus 97, alone: E43's arm `conflict`)

The forward pass is arch26.forward's text, copied (`fwd`), with the brief's two added lines and the list of z returned; the update is
exp26_candidate.step's text, copied (`step`), with the rule's two additions around opt.step(). exp43_unrelated.py and e43_addendum2.py are
imported and not edited. After arch26.install(), exp11_cell.forward is set to `fwd`, so that the scorer's generation runs the copy.

    python3 exp51_reserve.py start  <bias|keyed> <S|B> <seed>                                   # the embedded network, before any update
    python3 exp51_reserve.py run    <unrelated|conflict> <open|rehearse|keyed-free|keyed> <S|B> <seed> [--updates N] [--out DIR] [--start e26]
    python3 exp51_reserve.py check  <1|2|3|4|5|6> <seed> [--out DIR]
    python3 exp51_reserve.py forecast <dry DIR>                                                 # check 7, from the dry receipts

Every output path must not exist. One thread.
"""
import hashlib
import json
import sys
import time
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp43_unrelated as U          # noqa: E402  (not edited)
import e43_addendum2 as A2           # noqa: E402  (not edited)
import exp41_scale as S41            # noqa: E402  (not edited; new_net(seed, H, U))
from exp3_state import lin           # noqa: E402

E38, C, R, X16, E, X, arch26, T, TK = U.E38, U.C, U.R, U.X16, U.E, U.X, U.arch26, U.T, U.TK
Mat = arch26.Mat
OUT51 = X.OUT / "e51"
E41DIR = X.OUT / "e41"
H0, U0, H1, U1 = 256, 128, 512, 256          # half width (the birth), present size (the embedded network)
DOSE = 4096
ROWS = (0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
KEEP_ROWS = (64, 4096)
GEN_ROWS = (0, 64, 4096)
ARMS = {"open": dict(start="bias", rule=False, rehearse=False), "rehearse": dict(start="bias", rule=False, rehearse=True),
        "keyed-free": dict(start="keyed", rule=False, rehearse=False), "keyed": dict(start="keyed", rule=True, rehearse=False)}
TASKS = ("unrelated", "conflict")
OLD_ROWS = T.tensor([g * U1 + i for g in range(3) for i in range(U0)])
RES_ROWS = T.tensor([g * U1 + i for g in range(3) for i in range(U0, U1)])
OWN = ("exp51_reserve.py", "exp43_unrelated.py", "e43_addendum2.py", "e43_task.py", "exp41_scale.py", "exp38_erase.py", "exp3_state.py")
ITEMS_DIGEST = {73129: "d22b5733778491b1", 93683: "54bdb3c36ba2dc6e", 116089: "f29bee3306c1a5d1"}
N_DIRECTIONS = {"ih": 46, "hh": 129, "lm": 129, "l2": 11}

# the keyed forward's condition: in_use is None (no key) or a bool tensor over the 258 symbols; shut_all holds every reserve cell shut on
# every item (the second scoring of the new task)
KEY = dict(in_use=None, shut_all=False)


def provenance(cmd):
    p = U.provenance(cmd)
    p["e51_sources_sha256"] = {s: X16.sha_file(HERE / s) for s in OWN}
    import subprocess
    p["git"]["e51_sources_modified_vs_commit"] = subprocess.run(["git", "status", "--porcelain", "--", *OWN], cwd=HERE, capture_output=True,
                                                                text=True).stdout.strip().splitlines()
    return p


# ---------------------------------------------------------------- the forward pass: arch26.forward's text with the two added lines
def fwd(m, x, local, through_time, keep=False, silence=False):
    assert getattr(m, "kind", None) == "gru", "arch26 runs E20-B's G1 (arch20.NetRNN('gru')) only"
    u = m.book(x)
    z1 = T.tanh(lin(u, m.l1, local))
    B, L = x.shape
    gi = lin(z1, Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0), local)            # the input map, all steps at once
    hh = Mat(m.rnn.weight_hh_l0, m.rnn.bias_hh_l0)
    h = T.zeros(B, m.units)                                                    # every item starts from zero state
    shut = off = None
    if KEY["in_use"] is not None:
        used = KEY["in_use"][x[:, 1]]                                          # has the item's first byte occurred before?
        shut = used[:, None] & (T.arange(m.units) >= U0)[None, :]
        off = (used[:, None] & (T.arange(m.l2.out_features) >= H0)[None, :])[:, None, :].expand(B, L, -1)
    if KEY["shut_all"]:
        every = (T.arange(m.units) >= U0)[None, :].expand(B, -1)
        shut = every if shut is None else (shut | every)
    hs, zs = [], []
    for t in range(L):
        hp = h if through_time else h.detach()
        gh = lin(hp, hh, local)                                                # the recurrent map, crossed again at every step back
        i_r, i_z, i_n = gi[:, t].chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        r = T.sigmoid(i_r + h_r)
        z = T.sigmoid(i_z + h_z)
        if shut is not None:
            z = T.where(shut, T.ones_like(z), z)                               # added line 1
        nn_ = T.tanh(i_n + r * h_n)
        h = (1 - z) * nn_ + z * hp
        if keep and h.requires_grad:
            h.retain_grad()
        hs.append(h); zs.append(z)
    Hs = T.stack(hs, 1)
    if silence:
        Hs = T.zeros_like(Hs)
    z2 = T.tanh(lin(z1, m.l2, local) + lin(Hs, m.lm, local))
    if off is not None:
        z2 = T.where(off, T.zeros_like(z2), z2)                                # added line 2
    return lin(z2, m.l3, local), dict(z1=z1, h=hs, z=zs, z2=z2)


def install(in_use=None):
    arch26.install()
    E.forward = fwd
    KEY["in_use"], KEY["shut_all"] = in_use, False


# ---------------------------------------------------------------- the rule
def away(D, Vm):
    return D - (D @ Vm) @ Vm.T


class Rule:
    """The two additions to the step: the gradient before opt.step(), the step itself after it; and the restoring of what is not changed."""

    def __init__(self, m, dirs, in_use, restore_book=True):
        self.m, self.in_use, self.restore_book = m, in_use, restore_book
        self.mats = [(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0, dirs["ih"], OLD_ROWS), (m.rnn.weight_hh_l0, m.rnn.bias_hh_l0, dirs["hh"], OLD_ROWS),
                     (m.lm.weight, m.lm.bias, dirs["lm"], T.arange(H0)), (m.l2.weight, m.l2.bias, dirs["l2"], T.arange(H0))]
        self.fixed = [m.l1.weight, m.l1.bias, m.l3.bias]
        self.snap = None

    @T.no_grad()
    def before(self):
        for W, b, Vm, rows in self.mats:
            G = T.cat([W.grad, b.grad[:, None]], 1).double()
            G[rows] = away(G[rows], Vm); W.grad.copy_(G[:, :-1]); b.grad.copy_(G[:, -1])
        m = self.m
        self.snap = dict(mats=[(W.detach().clone(), b.detach().clone()) for W, b, _, _ in self.mats], fixed=[p.detach().clone() for p in self.fixed],
                         l3=m.l3.weight[:, :H0].detach().clone(), book=m.book.weight[self.in_use].detach().clone())

    @T.no_grad()
    def after(self):
        m = self.m
        for (W, b, Vm, rows), (W0, b0) in zip(self.mats, self.snap["mats"]):
            S = T.cat([W - W0, (b - b0)[:, None]], 1).double()
            Sr = away(S[rows], Vm)
            W[rows] = (W0[rows].double() + Sr[:, :-1]).float(); b[rows] = (b0[rows].double() + Sr[:, -1]).float()
        for p, p0 in zip(self.fixed, self.snap["fixed"]):
            p.copy_(p0)
        m.l3.weight[:, :H0] = self.snap["l3"]
        if self.restore_book:
            m.book.weight[self.in_use] = self.snap["book"]


def step(m, opt, items, local, rule=None):
    """exp26_candidate.step's text, the forward being the runner's copy, with the rule's two additions."""
    x, y, mask = X.batch_tensor(items)
    logits = fwd(m, x, local, True)[0]
    loss = E.cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    if rule is not None:
        rule.before()
    opt.step()
    if rule is not None:
        rule.after()
    return float(loss.detach())


# ---------------------------------------------------------------- the embedded network
def birth41(rule, seed):
    f = E41DIR / f"e41_{rule}_h256_f50_{seed}.pt"; rec = json.loads(f.with_suffix(".json").read_text())
    sha = X16.sha_file(f); sd0 = T.load(f)
    assert rec["complete"] is True and rec["H"] == H0 and rec["U"] == U0 and rec["rule"] == rule and rec["seed"] == seed and rec["fraction"] == 0.5
    assert sha == rec["weights_sha256"], "half-width birth: file hash differs from its receipt"
    assert R.tensors_sha(sd0) == rec["tensors_sha256"], "half-width birth: tensors hash differs from its receipt"
    m0 = S41.new_net(seed, H0, U0); m0.load_state_dict(sd0)
    return m0, sd0, dict(file=str(f.relative_to(HERE)), sha256=sha, tensors_sha256=rec["tensors_sha256"])


def embed(seed, sd0, form):
    m = S41.new_net(seed, H1, U1)
    with T.no_grad():
        sd = m.state_dict()
        sd["book.weight"].copy_(sd0["book.weight"])
        sd["l1.weight"][:256] = sd0["l1.weight"];  sd["l1.bias"][:256] = sd0["l1.bias"]
        for g in range(3):                                   # torch's gate order: r, z, n
            new, old = slice(g * 256, g * 256 + 128), slice(g * 128, (g + 1) * 128)
            sd["rnn.weight_ih_l0"][new, :256] = sd0["rnn.weight_ih_l0"][old]
            sd["rnn.weight_ih_l0"][new, 256:] = 0.0          # added feed-forward units do not reach old cells
            sd["rnn.weight_hh_l0"][new, :128] = sd0["rnn.weight_hh_l0"][old]   # columns 128..255 keep the fresh draws
            sd["rnn.bias_ih_l0"][new] = sd0["rnn.bias_ih_l0"][old]
            sd["rnn.bias_hh_l0"][new] = sd0["rnn.bias_hh_l0"][old]
        sd["lm.weight"][:256, :128] = sd0["lm.weight"];  sd["lm.bias"][:256] = sd0["lm.bias"]
        sd["l2.weight"][:256, :256] = sd0["l2.weight"];  sd["l2.weight"][:256, 256:] = 0.0;  sd["l2.bias"][:256] = sd0["l2.bias"]
        sd["l3.weight"][:, :256] = sd0["l3.weight"];     sd["l3.weight"][:, 256:] = 0.0;     sd["l3.bias"].copy_(sd0["l3.bias"])
        if form == "bias":
            sd["rnn.bias_ih_l0"][384:512] = 10.0
            sd["rnn.bias_hh_l0"][384:512] = 0.0
        else:
            assert form == "keyed"
    return m


def items2080(seed):
    rng = Random(f"e51|{seed}"); items = {k: [] for k in range(13)}
    for t in E.DEV:                                           # add, sub, max, min under 89
        taught = E.split(t, .5, seed)[0]
        for k in range(13):
            items[k] += [E.render(t, a, b, k, rng) for a, b in rng.sample(taught, 40)]
    flat = [[p, a] for k in range(13) for p, a in items[k]]
    dig = hashlib.sha256(json.dumps(flat).encode()).hexdigest()
    in_use = T.zeros(258, dtype=T.bool); in_use[E.BOS] = True
    for p, a in flat:
        for b in p + a[:-1]:
            in_use[b] = True
    return items, dig, in_use


@T.no_grad()
def directions(m, items):
    """The directions in use at the four matrices, from the 2,080 items under teacher forcing with the keyed forward, one number of letters
    at a time (no row is padded). float64; a constant 1 appended; every eigenvector of X'X/n whose eigenvalue exceeds 1e-12 of the largest."""
    col = {"ih": [], "hh": [], "lm": [], "l2": []}
    for k in sorted(items):
        x, y, mask = X.batch_tensor(items[k])
        assert all(len(p) + len(a) == x.shape[1] for p, a in items[k])
        it = fwd(m, x, False, True)[1]
        Hs = T.stack(it["h"], 1); prev = T.cat([T.zeros_like(Hs[:, :1]), Hs[:, :-1]], 1)
        col["ih"].append(it["z1"].reshape(-1, H1)); col["hh"].append(prev.reshape(-1, U1))
        col["lm"].append(Hs[mask]); col["l2"].append(it["z1"][mask])
    out, info = {}, {}
    for n, parts in col.items():
        Xm = T.cat(parts).double(); Xm = T.cat([Xm, T.ones(len(Xm), 1, dtype=T.float64)], 1)
        Cm = Xm.T @ Xm / len(Xm)
        w, V = T.linalg.eigh(Cm)
        keep = w > 1e-12 * w.max()
        out[n] = V[:, keep].contiguous()
        ws = w.flip(0); kk = int(keep.sum())
        info[n] = dict(vectors=len(Xm), dimension=Xm.shape[1], directions=kk, largest_eigenvalue=float(ws[0]), last_kept=float(ws[kk - 1]),
                       first_dropped=float(ws[kk]) if kk < len(ws) else None)
    return out, info


def start_paths(out, form, rule, seed):
    base = Path(out) / f"e51_start_{form}_{rule}_{seed}"
    return base.with_suffix(".pt"), Path(str(base) + "_directions.pt"), base.with_suffix(".json")


def build_start(form, rule, seed):
    """The embedded network of the form, its bytes in use and (keyed) its directions; the keyed forward is installed for a keyed form."""
    m0, sd0, b41 = birth41(rule, seed)
    m = embed(seed, sd0, form)
    items, dig, in_use = items2080(seed)
    assert dig.startswith(ITEMS_DIGEST[seed]), f"digest of the 2,080 items {dig[:16]} is not the reviewer's"
    assert int(in_use.sum()) == 46
    dirs = info = None
    install(in_use if form == "keyed" else None)
    if form == "keyed":
        dirs, info = directions(m, items)
    meta = dict(form=form, rule=rule, seed=seed, half_width_birth=b41, items_digest=dig, bytes_in_use=[int(i) for i in in_use.nonzero().flatten()],
                directions=info, tensors_sha256=R.tensors_sha(m.state_dict()), model=S41.describe(m))
    return m, m0, in_use, dirs, meta


def make_start(form, rule, seed, out, cmd):
    wp, dp, jp = start_paths(out, form, rule, seed)
    R.refuse_existing(wp, dp, jp); Path(out).mkdir(parents=True, exist_ok=True)
    m, m0, in_use, dirs, meta = build_start(form, rule, seed)
    meta["weights"] = wp.name; R.save_weights(m, wp); meta["weights_sha256"] = X16.sha_file(wp)
    if dirs is not None:
        T.save(dict(directions=dirs, in_use=in_use), dp); meta["directions_file"] = dp.name; meta["directions_file_sha256"] = X16.sha_file(dp)
        meta["directions_tensors_sha256"] = R.tensors_sha(dirs)
    meta["provenance"] = provenance(cmd)
    jp.write_text(json.dumps(dict(experiment="E51", kind="start", **meta), indent=1))
    print("start", form, rule, seed, "tensors", meta["tensors_sha256"][:16], "directions", {n: v["directions"] for n, v in (meta["directions"] or {}).items()}, flush=True)


def load_start(form, rule, seed, out=OUT51):
    """Rebuild the start and require it to be the committed one, tensor for tensor."""
    wp, dp, jp = start_paths(out, form, rule, seed)
    rec = json.loads(jp.read_text())
    m, m0, in_use, dirs, meta = build_start(form, rule, seed)
    assert X16.sha_file(wp) == rec["weights_sha256"] and meta["tensors_sha256"] == rec["tensors_sha256"] == R.tensors_sha(T.load(wp)), "start differs from the saved one"
    if form == "keyed":
        saved = T.load(dp)
        assert X16.sha_file(dp) == rec["directions_file_sha256"] and T.equal(saved["in_use"], in_use)
        assert all(saved["directions"][n].shape == dirs[n].shape for n in dirs)
        dirs = saved["directions"]                               # the saved directions are the ones used (computed once, before the first update)
        assert {n: dirs[n].shape[1] for n in dirs} == N_DIRECTIONS, {n: dirs[n].shape[1] for n in dirs}
    return m, m0, in_use, dirs, dict(start_file=wp.name, start_sha256=rec["weights_sha256"], start_tensors_sha256=rec["tensors_sha256"],
                                     half_width_birth=rec["half_width_birth"], items_digest=rec["items_digest"],
                                     directions={n: v["directions"] for n, v in (rec["directions"] or {}).items()} or None)


# ---------------------------------------------------------------- scores
@T.no_grad()
def answer_logits(m, items):
    """Outputs at the three answer positions under teacher forcing with the right answer, item by item (grouped by length, no padding)."""
    groups, res = {}, [None] * len(items)
    for i, (p, _) in enumerate(items):
        groups.setdefault(len(p), []).append(i)
    for L, idx in groups.items():
        for j in range(0, len(idx), 256):
            ids = idx[j:j + 256]
            x, y, mask = X.batch_tensor([items[i] for i in ids])
            lg = fwd(m, x, False, True)[0][mask].reshape(len(ids), 3, -1)
            for q, i in enumerate(ids):
                res[i] = lg[q]
    return T.stack(res)


def generate(m, items):
    return E.respond(m, [p for p, _ in items], None, cap=X16.CAP)


def first_family_pairs():
    """(a, b) of every item of exp43_unrelated.first_family_items(), in its order."""
    return {E.NAME(*t): [(a, b) for a in range(t[1]) for b in range(t[1])] for t in E.DEV}


def wrong_class(g, target, a, b, op):
    """The class of a wrong answer on a reduction pair, tried in the brief's order."""
    if g == [E.EOS]:
        return "1 end symbol alone"
    if any(128 <= v < 256 for v in g):
        return "2 contains a byte of 128 or above"
    two = len(g) == 3 and g[2] == E.EOS and all(48 <= v <= 57 for v in g[:2])
    if two:
        val = int(bytes(g[:2])); right = int(bytes(target[:2]))
        if val == E.apply(op, 97, a, b):
            return "3 right under modulus 97"
        if abs(val - right) in (1, 10):
            return "4 differs by exactly 1 or 10"
        return "5 other two digits"
    return "6 anything else"


CLASSES = ("1 end symbol alone", "2 contains a byte of 128 or above", "3 right under modulus 97", "4 differs by exactly 1 or 10", "5 other two digits",
           "6 anything else")


def score_first_family(m, s1, pairs, held, gen0):
    """Scores 1 and 2 with the generated bytes kept; the share of answers equal to row 0's; the classes of wrong answers on reduction pairs."""
    gens, res, same, tot = {}, {}, 0, 0
    classes = {n: {c: 0 for c in CLASSES} for n in ("add89", "sub89")}
    for n, (items, red) in s1.items():
        outs = generate(m, items); gens["k0_" + n] = outs
        ok = [o == t for o, (_, t) in zip(outs, items)]
        if n in ("add89", "sub89"):
            r = [o for o, f in zip(ok, red) if f]; nr = [o for o, f in zip(ok, red) if not f]
            res[n] = dict(reduction=dict(n=len(r), exact=sum(r), pct=100 * sum(r) / len(r)), other=dict(n=len(nr), exact=sum(nr), pct=100 * sum(nr) / len(nr)))
            for o, (_, t), f, good, (a, b) in zip(outs, items, red, ok, pairs[n]):
                if f and not good:
                    classes[n][wrong_class(o, t, a, b, n[:3])] += 1
        else:
            res[n] = dict(all=dict(n=len(ok), exact=sum(ok), pct=100 * sum(ok) / len(ok)))
    res["reduction_score"] = (res["add89"]["reduction"]["pct"] + res["sub89"]["reduction"]["pct"]) / 2
    s2 = {}
    for n, items in held.items():
        outs = generate(m, items); gens["k12_" + n] = outs
        s2[n] = dict(n=len(items), exact=sum(o == t for o, (_, t) in zip(outs, items)), final=sum(o[-3:] == t[-3:] for o, (_, t) in zip(outs, items)))
    if gen0 is None:
        gen0 = gens
    eq = {n: sum(a == b for a, b in zip(gens[n], gen0[n])) for n in gens}
    k0 = [n for n in gens if n.startswith("k0_")]; k12 = [n for n in gens if n.startswith("k12_")]
    share = dict(per_panel=eq, all_pairs_k0=sum(eq[n] for n in k0) / sum(len(gens[n]) for n in k0),
                 heldout_k12=sum(eq[n] for n in k12) / sum(len(gens[n]) for n in k12))
    return res, s2, share, classes, gens


def reduction_bytes(gens, s1):
    return {n: [g for g, f in zip(gens["k0_" + n], s1[n][1]) if f] for n in ("add89", "sub89")}


@T.no_grad()
def reserve_figures(m, new_items, ff_items):
    """Reserve cells whose update gate is open by more than 0.1 (mean of 1 - z over the positions and items of the new task's panel, teacher
    forcing); the largest size of a reserve cell's state on the first family's held-out panel."""
    tot, n = T.zeros(U1 - U0, dtype=T.float64), 0
    groups = {}
    for it in new_items:
        groups.setdefault(len(it[0]) + len(it[1]), []).append(it)
    for g in groups.values():
        for j in range(0, len(g), 256):
            x, y, mask = X.batch_tensor(g[j:j + 256])
            zs = T.stack(fwd(m, x, False, True)[1]["z"], 1)[:, :, U0:]
            tot += (1 - zs).double().sum((0, 1)); n += zs.shape[0] * zs.shape[1]
    opening = tot / n
    big = 0.0
    groups = {}
    for it in ff_items:
        groups.setdefault(len(it[0]) + len(it[1]), []).append(it)
    for g in groups.values():
        for j in range(0, len(g), 256):
            x, y, mask = X.batch_tensor(g[j:j + 256])
            hs = T.stack(fwd(m, x, False, True)[1]["h"], 1)[:, :, U0:]
            big = max(big, float(hs.abs().max()))
    return dict(reserve_cells_open_by_more_than_0p1=int((opening > 0.1).sum()), mean_opening=float(opening.mean()), largest_opening=float(opening.max()),
                largest_reserve_state_on_first_family_panel=big)


def weights_figures(m, bsd):
    sd = {n: p.detach() for n, p in m.named_parameters()}
    mv = U.movement(m, bsd)

    def part(names, rows):
        w = T.cat([sd[n][rows].flatten().double() for n in names]); w0 = T.cat([bsd[n][rows].flatten().double() for n in names])
        return dict(rel_diff=float((w - w0).norm() / w0.norm()), cosine=float(w @ w0 / (w.norm() * w0.norm())), norm=float(w.norm()), norm_start=float(w0.norm()))
    split = {}
    for g, names in (("input map", ["rnn.weight_ih_l0", "rnn.bias_ih_l0"]), ("recurrent matrix", ["rnn.weight_hh_l0", "rnn.bias_hh_l0"])):
        split[g] = dict(old_cells_rows=part(names, OLD_ROWS), reserve_cells_rows=part(names, RES_ROWS))
    return mv, split, {n: float(p.double().norm()) for n, p in sd.items()}


# ---------------------------------------------------------------- streams
def new_stream(task, seed):
    """An iterator of single items of the new task, in the order E43's arm of that name consumes them."""
    if task == "conflict":
        return X16.entry_stream(X16.family_entries("F1", seed), seed + 101)
    return TK.stream(seed)


def family_stream(seed):
    return X16.entry_stream([("op", t, a, b) for t in E.DEV for a, b in E.split(t, .5, seed)[0]], seed + 100)


def item_source(task, arm, seed):
    """nxt() -> (kind, item). Batches are 64 consecutive items. Without rehearsal this is E43's own batching (64 consecutive stream items)."""
    new = new_stream(task, seed)
    if not ARMS[arm]["rehearse"]:
        return (lambda: (task, next(new))), "the new task's stream alone"
    fam = family_stream(seed)
    key = f"e43|{seed}|joint" if task == "unrelated" else f"e51|{seed}|joint|conflict"
    rj = Random(key)
    return (lambda: (task, next(new)) if rj.randrange(2) == 0 else ("first_family", next(fam))), f"Random({key!r}).randrange(2): 0 -> the new task's stream, 1 -> the first family's taught half (X16.entry_stream, seed + 100)"


def new_panels(task, seed, P):
    if task == "conflict":
        return {n: P[n][12] for n in ("add97", "sub97")}
    return {"held": TK.panel(seed, "held"), "taught": TK.panel(seed, "taught")}


def never_taught(task, s3):
    """The new task's score on its never-taught items: the held panel (unrelated), the two panels together, 800 items (conflict)."""
    ps = ("add97", "sub97") if task == "conflict" else ("held",)
    return dict(n=sum(s3[n]["n"] for n in ps), exact=sum(s3[n]["exact"] for n in ps), pct=100 * sum(s3[n]["exact"] for n in ps) / sum(s3[n]["n"] for n in ps))


# ---------------------------------------------------------------- one run
def run(task, arm, rule, seed, out, cmd, updates=DOSE, start="embed", restore_book=True):
    local = C.ARMS[rule]; cfg = ARMS[arm]; out = Path(out)
    tag = f"e51_{task}_{arm}_{rule}_{seed}" + ("_e26birth" if start == "e26" else "") + ("" if updates == DOSE else f"_dry{updates}")
    path = out / f"{tag}.json"; R.refuse_existing(path); out.mkdir(parents=True, exist_ok=True); (out / "rows_uncommitted").mkdir(exist_ok=True)
    if start == "e26":                                           # check 6: E26's birth as it is, no embedding, no change of any bias
        assert arm == "open"
        install(None)
        bfile = C.birth_path(E38.E26DIR, rule, seed); m, birth_sha, admission = E38.admit(bfile, rule, seed)
        m.load_state_dict(T.load(bfile)); in_use = dirs = None
        start_rec = dict(birth=str(bfile.relative_to(HERE)), birth_sha256=birth_sha, birth_admission=admission)
    else:
        m, m0, in_use, dirs, start_rec = load_start(cfg["start"], rule, seed)
    bsd = {k: v.clone() for k, v in m.state_dict().items()}
    start_tsha = R.tensors_sha(bsd)
    opt = C.make_opt(m)
    opt.param_groups[0]["weight_decay"] = 0.0
    groups_read_back = R.opt_groups(m, opt)
    ruleobj = Rule(m, dirs, in_use, restore_book) if cfg["rule"] else None

    P, comp = X16.panels(seed)
    held = {n: P[n][12] for n in ("add89", "sub89", "max89", "min89")}
    s1 = U.first_family_items(); pairs = first_family_pairs()
    s2_items = [it for n in held for it in held[n]]
    npan = new_panels(task, seed, P)
    reserve_panel = [it for n in (("add97", "sub97") if task == "conflict" else ("held",)) for it in npan[n]]
    nxt, stream_note = item_source(task, arm, seed)
    dg = C.Digest(); counts = {task: 0, "first_family": 0}; losses = []
    t0 = time.process_time(); cpu_score = [0.0]; rows = []; gen0 = [None]
    rowset = [r for r in ROWS if r <= updates]

    def score_row(g):
        cb = time.process_time()
        r = dict(updates=g, data_digest=dg.value(), items_so_far=dict(counts), loss_since_last_row=(sum(losses) / len(losses)) if losses else None)
        s1r, s2r, share, classes, gens = score_first_family(m, s1, pairs, held, gen0[0])
        if g == 0:
            gen0[0] = gens
        r["score1_first_family_all_pairs_k0"] = s1r; r["score2_first_family_heldout_k12"] = s2r
        r["share_of_answers_equal_to_row0"] = share; r["wrong_answers_on_reduction_pairs"] = classes
        if g in GEN_ROWS:
            gp = out / f"{tag}_reduction_bytes_row{g}.json"; gp.write_text(json.dumps(reduction_bytes(gens, s1)))
            r["reduction_bytes"] = gp.name; r["reduction_bytes_sha256"] = X16.sha_file(gp)
        s3 = {n: X16.score(m, it, None) for n, it in npan.items()}
        r["score3_new_task"] = s3; r["score3_never_taught"] = never_taught(task, s3)
        if start != "e26":
            KEY["shut_all"] = True
            s3s = {n: X16.score(m, it, None) for n, it in npan.items()}
            KEY["shut_all"] = False
            r["score3_new_task_reserve_shut"] = s3s; r["score3_never_taught_reserve_shut"] = never_taught(task, s3s)
            r["reserve"] = reserve_figures(m, reserve_panel, s2_items)
        mv, split, norms = weights_figures(m, bsd)
        r["score4_movement"] = mv; r["tensor_norms"] = norms
        if start != "e26":
            r["score4_movement_old_and_reserve_rows"] = split
        cpu_score[0] += time.process_time() - cb
        r["tensors_sha256"] = R.tensors_sha(m.state_dict())
        if g in KEEP_ROWS:
            wp = (out if updates == DOSE else out / "rows_uncommitted") / f"{tag}_row{g}.pt"
            r["weights"] = str(wp.relative_to(out)); R.save_weights(m, wp); r["weights_sha256"] = X16.sha_file(wp)
        r["cpu"] = time.process_time() - t0
        rows.append(r); losses.clear()
        pc = lambda d: round(100 * d["exact"] / d["n"], 1)
        print(tag, "row", g, "reduction", round(s1r["reduction_score"], 1), "other", round(s1r["add89"]["other"]["pct"], 1), round(s1r["sub89"]["other"]["pct"], 1),
              "max/min", round(s1r["max89"]["all"]["pct"], 1), round(s1r["min89"]["all"]["pct"], 1), "held k12", {n: pc(v) for n, v in s2r.items()},
              "same as row 0", round(100 * share["all_pairs_k0"], 2), "new", {n: pc(v) for n, v in s3.items()},
              "shut", {n: pc(v) for n, v in r.get("score3_new_task_reserve_shut", {}).items()}, "reserve", r.get("reserve"), f"cpu {r['cpu']:.0f}s", flush=True)

    settings = dict(experiment="E51", part="A", task=task, arm=arm, rule=rule, local=local, seed=seed, updates=updates, start=start_rec,
                    start_form=cfg["start"] if start != "e26" else "E26 birth", start_tensors_sha256=start_tsha, keyed_forward=in_use is not None and cfg["start"] == "keyed",
                    rule_applied=cfg["rule"], restore_book=restore_book if cfg["rule"] else None, stream=stream_note, decay=0.0,
                    optimizer_groups_read_back=groups_read_back, model=S41.describe(m), batch=X.BATCH, rows=rowset, committed_rows=list(KEEP_ROWS),
                    update="exp51_reserve.step (exp26_candidate.step's text; forward exp51_reserve.fwd; the rule's two additions in arm keyed)",
                    forward="exp51_reserve.fwd (arch26.forward's text; z and z2 held on items whose first byte is in use, in the keyed arms)",
                    provenance=provenance(cmd))
    assert KEY["in_use"] is (in_use if (start != "e26" and cfg["start"] == "keyed") else None)

    def write(complete=None):
        path.write_text(json.dumps(dict(experiment="E51", kind="run", tag=tag, settings=settings, rows=rows, complete=complete), indent=1))

    score_row(0); write(); g = 0
    while g < updates:
        batch = []
        for _ in range(X.BATCH):
            kind, it = nxt(); counts[kind] += 1
            if kind == "unrelated":
                assert all(b >= 128 for b in it[0])
            else:
                assert all(b < 128 for b in it[0])
            batch.append(it)
        dg.add(batch)
        losses.append(step(m, opt, batch, local, ruleobj))
        g += 1
        if g in rowset:
            score_row(g); write()
    write(dict(updates=g, items=counts, optimizer_steps=R.opt_steps(opt), optimizer_groups_read_back=R.opt_groups(m, opt), data_digest=dg.value(),
               tensors_sha256=R.tensors_sha(m.state_dict()), cpu_total=time.process_time() - t0, cpu_scoring=cpu_score[0],
               cpu_updates=time.process_time() - t0 - cpu_score[0]))
    print(tag, "complete", f"cpu {time.process_time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- checks before the runs
def check_panels(seed):
    P, comp = X16.panels(seed)
    held = [it for n in ("add89", "sub89", "max89", "min89") for it in P[n][12]]
    s1 = U.first_family_items(); allitems = [it for n in s1 for it in s1[n][0]]
    assert len(allitems) == 31684
    return held, Random(f"e51|{seed}|check1").sample(allitems, 2000), s1, {n: P[n][12] for n in ("add89", "sub89", "max89", "min89")}


def check1(seed, out, cmd):
    """The embedding: outputs at the answer positions equal the half-width birth's (1e-5 keyed, 0.02 with a bias); row 0's scores equal."""
    held, samp, s1, heldp = check_panels(seed); res = {}
    for rule in ("S", "B"):
        for form in ("keyed", "bias"):
            m, m0, in_use, dirs, rec = load_start(form, rule, seed)
            d = {}
            for name, items in (("heldout_k12", held), ("sample_2000_k0", samp)):
                a = answer_logits(m, items); b = answer_logits(m0, items)
                d[name] = float((a - b).abs().max())
            lim = 1e-5 if form == "keyed" else 0.02
            r = dict(largest_difference=d, limit=lim, ok=all(v <= lim for v in d.values()))
            if form == "keyed":
                pairs = first_family_pairs()
                sa = score_first_family(m, s1, pairs, heldp, None); sb = score_first_family(m0, s1, pairs, heldp, None)
                r["row0_scores_equal_half_width_birth"] = sa[0] == sb[0] and sa[1] == sb[1]
                r["row0"] = dict(score1=sa[0], score2=sa[1]); r["half_width_birth"] = dict(score1=sb[0], score2=sb[1])
                r["ok"] = r["ok"] and r["row0_scores_equal_half_width_birth"]
            res[f"{form}_{rule}"] = r; print("check 1", form, rule, d, r["ok"], flush=True)
    return dict(ok=all(v["ok"] for v in res.values()), runs=res)


def check2(seed, out, cmd):
    res = {}
    for rule in ("S", "B"):
        _, _, jp = start_paths(OUT51, "keyed", rule, seed); rec = json.loads(jp.read_text())
        n = {k: v["directions"] for k, v in rec["directions"].items()}
        res[rule] = dict(items_digest=rec["items_digest"], digest_is_the_reviewers=rec["items_digest"].startswith(ITEMS_DIGEST[seed]), bytes_in_use=len(rec["bytes_in_use"]),
                         directions=n, spectrum=rec["directions"], ok=n == N_DIRECTIONS and rec["items_digest"].startswith(ITEMS_DIGEST[seed]) and len(rec["bytes_in_use"]) == 46)
        print("check 2", rule, n, res[rule]["ok"], flush=True)
    return dict(ok=all(v["ok"] for v in res.values()), expected=N_DIRECTIONS, rules=res)


def check3(seed, out, cmd):
    held, samp, s1, heldp = check_panels(seed); res = {}
    for rule in ("S", "B"):
        local = C.ARMS[rule]
        for task in TASKS:
            for variant in ("keyed", "keyed, byte table not restored", "keyed-free"):
                m, m0, in_use, dirs, rec = load_start("keyed", rule, seed)
                bsd = {k: v.clone() for k, v in m.state_dict().items()}
                opt = C.make_opt(m); opt.param_groups[0]["weight_decay"] = 0.0
                ro = None if variant == "keyed-free" else Rule(m, dirs, in_use, restore_book=(variant == "keyed"))
                a0 = {n: answer_logits(m, it) for n, it in (("heldout_k12", held), ("sample_2000_k0", samp))}
                st = new_stream(task, seed)
                for _ in range(8):
                    step(m, opt, [next(st) for _ in range(X.BATCH)], local, ro)
                d = {n: float((answer_logits(m, it) - a0[n]).abs().max()) for n, it in (("heldout_k12", held), ("sample_2000_k0", samp))}
                sd = m.state_dict()
                equal = {"l1.weight": bool(T.equal(sd["l1.weight"], bsd["l1.weight"])), "l1.bias": bool(T.equal(sd["l1.bias"], bsd["l1.bias"])),
                         "l3.weight[:, :256]": bool(T.equal(sd["l3.weight"][:, :H0], bsd["l3.weight"][:, :H0])), "l3.bias": bool(T.equal(sd["l3.bias"], bsd["l3.bias"])),
                         "book.weight rows in use": bool(T.equal(sd["book.weight"][in_use], bsd["book.weight"][in_use]))}
                book_rows_changed = int((sd["book.weight"][in_use] != bsd["book.weight"][in_use]).any(1).sum())
                moved = {n: bool(not T.equal(sd[n], bsd[n])) for n in ("rnn.weight_ih_l0", "rnn.weight_hh_l0", "lm.weight", "l2.weight", "l3.weight", "book.weight")}
                r = dict(largest_change_of_an_output=d, not_changed_bit_equal=equal, rows_of_the_byte_table_in_use_changed=book_rows_changed, tensors_that_moved=moved)
                if variant == "keyed":
                    r["ok"] = all(v <= 1e-4 for v in d.values()) and all(equal.values()) and all(moved.values())
                elif variant == "keyed-free":
                    r["ok"] = all(v > 0.1 for v in d.values())
                else:                                          # the conflict stream's items contain bytes in use: without restoring, a row moves
                    r["ok"] = (book_rows_changed >= 1) if task == "conflict" else True
                    r["note"] = "required on the conflict stream only; the unrelated stream has no byte in use but the start symbol"
                res[f"{rule} {task} {variant}"] = r; print("check 3", rule, task, variant, d, book_rows_changed, r["ok"], flush=True)
        # (c) reserve states and added units of z2 on a batch of the first family and on a batch of the new task
        m, m0, in_use, dirs, rec = load_start("keyed", rule, seed)
        fam = family_stream(seed); c = {}
        with T.no_grad():
            for name, items in (("first_family", [next(fam) for _ in range(X.BATCH)]), ("unrelated", next(TK.batches(seed, X.BATCH))),
                                ("conflict", [it for it, _ in zip(new_stream("conflict", seed), range(X.BATCH))])):
                x, y, mask = X.batch_tensor(items)
                it = fwd(m, x, False, True)[1]
                c[name] = dict(largest_reserve_state=float(T.stack(it["h"], 1)[:, :, U0:].abs().max()), largest_added_z2=float(it["z2"][:, :, H0:].abs().max()))
        ok = c["first_family"] == dict(largest_reserve_state=0.0, largest_added_z2=0.0) and all(c[n]["largest_reserve_state"] > 0 and c[n]["largest_added_z2"] > 0 for n in TASKS)
        res[f"{rule} (c)"] = dict(batches=c, ok=ok); print("check 3c", rule, c, ok, flush=True)
    return dict(ok=all(v["ok"] for v in res.values()), runs=res)


def check4(seed, out, cmd):
    """The scorer: in a keyed run the generated answers of the first family at row 0 equal the half-width birth's, item for item; and with the
    keyed weights and the added lines switched off they do not (the lines act in generation)."""
    held, samp, s1, heldp = check_panels(seed); res = {}
    for rule in ("S", "B"):
        m, m0, in_use, dirs, rec = load_start("keyed", rule, seed)
        ga = {n: generate(m, s1[n][0]) for n in s1}; ga.update({"k12_" + n: generate(m, heldp[n]) for n in heldp})
        gb = {n: generate(m0, s1[n][0]) for n in s1}; gb.update({"k12_" + n: generate(m0, heldp[n]) for n in heldp})
        KEY["in_use"] = None
        gc = {n: generate(m, s1[n][0]) for n in s1}; gc.update({"k12_" + n: generate(m, heldp[n]) for n in heldp})
        KEY["in_use"] = in_use
        diff = {n: sum(a != b for a, b in zip(ga[n], gb[n])) for n in ga}; off = {n: sum(a != b for a, b in zip(gc[n], gb[n])) for n in ga}
        res[rule] = dict(items={n: len(ga[n]) for n in ga}, answers_differing_from_half_width_birth=diff, the_same_without_the_added_lines=off,
                         ok=sum(diff.values()) == 0, added_lines_matter=sum(off.values()) > 0)
        print("check 4", rule, diff, "without the lines", off, flush=True)
    return dict(ok=all(v["ok"] for v in res.values()), rules=res)


def check5(seed, out, cmd):
    """The data: the batch digests at update 4,096 equal E43's receipts (unrelated, conflict) and, for rehearse on unrelated, the second addendum's."""
    res = {}
    for task, arm, ref in (("unrelated", "open", U.OUT43 / f"e43_unrelated_S_{seed}.json"), ("conflict", "open", U.OUT43 / f"e43_conflict_S_{seed}.json"),
                           ("unrelated", "rehearse", A2.OUT / f"e43_unrelated-rehearse_S_{seed}.json")):
        nxt, note = item_source(task, arm, seed); dg = C.Digest()
        for _ in range(DOSE):
            dg.add([nxt()[1] for _ in range(X.BATCH)])
        want = {}
        for rule in ("S", "B"):
            rec = json.loads(Path(str(ref).replace("_S_", f"_{rule}_")).read_text()); want[rule] = rec["complete"]["data_digest"]
        res[f"{task} {arm}"] = dict(here=dg.value(), receipts=want, ok=all(dg.value() == w for w in want.values()))
        print("check 5", task, arm, dg.value()["sha256"][:16], res[f"{task} {arm}"]["ok"], flush=True)
    return dict(ok=all(v["ok"] for v in res.values()), streams=res)


def check6(seed, out, cmd):
    """`open` is E43 in another network: the runner of `open` from E26's birth as it is reproduces E43's `unrelated` receipt at row 64."""
    res = {}
    for rule in ("S", "B"):
        d = Path(out) / "check6"; f = d / f"e51_unrelated_open_{rule}_{seed}_e26birth_dry64.json"
        if not f.exists():
            run("unrelated", "open", rule, seed, d, cmd, updates=64, start="e26")
        mine = json.loads(f.read_text()); ref = json.loads((U.OUT43 / f"e43_unrelated_{rule}_{seed}.json").read_text())
        a = [r for r in mine["rows"] if r["updates"] == 64][0]; b = [r for r in ref["rows"] if r["updates"] == 64][0]
        res[rule] = dict(first_family_counts_equal=a["score1_first_family_all_pairs_k0"] == b["score1_first_family_all_pairs_k0"]
                         and a["score2_first_family_heldout_k12"] == b["score2_first_family_heldout_k12"],
                         new_task_equal=a["score3_new_task"] == b["score3_new_task"], tensors_sha256_here=a["tensors_sha256"], tensors_sha256_e43=b["tensors_sha256"],
                         data_digest_equal=a["data_digest"] == b["data_digest"])
        res[rule]["ok"] = res[rule]["first_family_counts_equal"] and a["tensors_sha256"] == b["tensors_sha256"]
        print("check 6", rule, res[rule], flush=True)
    return dict(ok=all(v["ok"] for v in res.values()), rules=res)


CHECKS = {"1": check1, "2": check2, "3": check3, "4": check4, "5": check5, "6": check6}


def forecast(d, cmd):
    d = Path(d); rows = {}
    for f in sorted(d.glob("e51_*_dry64.json")):
        r = json.loads(f.read_text()); c = r["complete"]; n = len(r["rows"])
        per_up = c["cpu_updates"] / c["updates"]; per_row = c["cpu_scoring"] / n
        rows[r["tag"]] = dict(cpu_seconds_an_update=per_up, cpu_seconds_a_scored_row=per_row, forecast_cpu_seconds_4096=per_up * DOSE + per_row * len(ROWS))
    tot = sum(v["forecast_cpu_seconds_4096"] for v in rows.values())
    out = dict(experiment="E51", kind="check 7", dry_runs=rows, runs=len(rows), forecast_cpu_hours_sixteen_runs=tot / 3600 * (16 / max(1, len(rows))),
               wall_hours_at_three_processes=tot / 3600 * (16 / max(1, len(rows))) / 3, provenance=provenance(cmd))
    (OUT51 / "checks").mkdir(parents=True, exist_ok=True); p = OUT51 / "checks" / "e51_check_7_cost.json"; R.refuse_existing(p)
    p.write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k != "provenance"}, indent=1))


if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])

    def opt_(flag, default=None):
        return a[a.index(flag) + 1] if flag in a else default

    outdir = opt_("--out", str(OUT51))
    if a[0] == "start":
        assert a[1] in ("bias", "keyed") and a[2] in C.ARMS
        make_start(a[1], a[2], int(a[3]), outdir, cmd)
    elif a[0] == "run":
        assert a[1] in TASKS and a[2] in ARMS and a[3] in C.ARMS
        run(a[1], a[2], a[3], int(a[4]), outdir, cmd, int(opt_("--updates", DOSE)), start=opt_("--start", "embed"))
    elif a[0] == "check":
        seed = int(a[2]); (Path(outdir) / "checks").mkdir(parents=True, exist_ok=True)
        names = {"1": "embedding", "2": "directions", "3": "rule", "4": "scorer", "5": "data", "6": "open_is_e43"}
        p = Path(outdir) / "checks" / f"e51_check_{a[1]}_{names[a[1]]}_{seed}.json"; R.refuse_existing(p)
        res = CHECKS[a[1]](seed, outdir, cmd)
        p.write_text(json.dumps(dict(experiment="E51", kind=f"check {a[1]}", seed=seed, **res, provenance=provenance(cmd)), indent=1))
        print("check", a[1], "ok" if res["ok"] else "FAILED", flush=True)
    elif a[0] == "forecast":
        forecast(a[1], cmd)
    else:
        raise SystemExit(__doc__)
