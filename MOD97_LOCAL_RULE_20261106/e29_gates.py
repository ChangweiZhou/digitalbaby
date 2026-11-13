"""E29 Part A (brief OPUS_TASK_E29_SLOW_UNITS_AND_DELAYS_20261113.md, revision 2, commit a5cb6259): how the trained GRUs use their
update gates. Read-only: forward passes on saved weights, nothing trained, no existing file edited.

    python3 e29_gates.py check                     the hand-built check (no trained file is opened); writes exp45/e29/E29_HAND_CHECK.json
    python3 e29_gates.py measure <group>           group: untrained | births_seen | births_unseen | lives | cross
                                                   writes exp45/e29/e29_<tag>.json per file (per-unit arrays and the per-file summary)
    python3 e29_gates.py summary                   collects every record into exp45/e29/E29_GATES_SUMMARY.json

The step. arch26.forward is run with local=False (exact credit; forward values are identical either way) and through_time=True. It does
not return the update gate z, so `gates()` repeats its step, line for line, keeping z, and asserts that every state h it produces is
bit-identical (torch.equal) to the h that arch26.forward returns for the same batch. Arithmetic is the weights' own float32, as in the scorer.

Items. add89 at k = 12 and k = 20, 300 items each, as the scorer loads them:
    E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) and X.batch_tensor
For the lives (E26 and E26-X) also the untaught plus-minus composition panel the life scorer uses: comp["+-"] + comp["-+"] of
exp16_integrated.panels(seed) (600 + 600). That panel has no distractor letters (E16_COMPGAP = 0 in the harness), so the measures that
need letters are not defined on it and are recorded as None; the lowest z, its role and the mean z are recorded.

Roles of the input positions of x = [BOS] + prompt + answer[:-1]:  start (position 0, the BOS byte, where h_prev = 0), code, digit (operand
digit), operator, letter (distractor), '=', answer. The brief names six roles; `start` is not among them. Because h_prev = 0 at position 0,
z there neither holds nor rewrites anything, so the primary "lowest z off letters" excludes it; the variant including it is recorded too.

Per unit (pooled over the items of one panel):
    hold_share_letters   share of letter positions with z > 0.9
    lowest_off_letters   min of z over the non-letter positions (start excluded); rewrite_role = the role where that min occurs
    retention_letters    per item the product of z over its run of letters, then the mean over items (product in float64)
    mean_z               mean of z over all positions but start (revision 1's measure, for comparison only)
A unit is a gated holder if hold_share_letters >= 0.9 and lowest_off_letters <= 0.1.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp20_reference as R          # noqa: E402  (applies the harness's environment gate, as exp26_candidate does)
import torch as T                    # noqa: E402

X16, E, X, arch20 = R.X16, R.E, R.X, R.arch20
import arch26                        # noqa: E402

T.set_num_threads(1)
OUT = HERE / "exp45" / "e29"
E26, E26X = HERE / "exp45" / "e26", HERE / "exp45" / "e26x"
SEEDS = (73129, 93683, 116089)
SEEN = (73129,)
ADD89 = E.DEV[0]
assert ADD89[0] == "add" and ADD89[1] == 89
OPS = set(b"+-<>")
ROLES = ("start", "code", "digit", "operator", "letter", "=", "answer")


# ---------------------------------------------------------------- items and roles
def roles_of(x_row):
    """Role of every input position, from the bytes and the item's structure; asserts the structure."""
    b = [int(v) for v in x_row]
    assert b[0] == X.BOS
    eq = b.index(ord("="))
    out = ["start", "code"]
    assert chr(b[1]).islower(), f"task code {b[1]}"
    for v in b[2:eq]:
        if 48 <= v <= 57: out.append("digit")
        elif v in OPS: out.append("operator")
        elif 65 <= v <= 90: out.append("letter")
        else: raise AssertionError(f"unexpected prompt byte {v}")
    out.append("=")
    out += ["answer"] * (len(b) - eq - 1)
    assert X.BOS not in b[1:], "padding inside a panel: the panel's items differ in length"
    return out


def add89_panel(seed, k):
    t = ADD89
    pk = E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300)
    return pk[k]


def plus_minus_panel(seed):
    _, comp = X16.panels(seed)
    return comp["+-"] + comp["-+"]


def sha_items(items):
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


# ---------------------------------------------------------------- the step, with z kept
@T.no_grad()
def gates(m, x):
    """arch26.forward's recurrent step (local=False, through_time=True), keeping z. Returns Z (N, L, U) and the check against arch26."""
    lin = arch26.lin
    u = m.book(x)
    z1 = T.tanh(lin(u, m.l1, False))
    B, L = x.shape
    gi = lin(z1, arch26.Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0), False)
    hh = arch26.Mat(m.rnn.weight_hh_l0, m.rnn.bias_hh_l0)
    h = T.zeros(B, m.units)
    hs, zs = [], []
    for t in range(L):
        gh = lin(h, hh, False)
        i_r, i_z, i_n = gi[:, t].chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)
        r = T.sigmoid(i_r + h_r)
        z = T.sigmoid(i_z + h_z)
        nn_ = T.tanh(i_n + r * h_n)
        h = (1 - z) * nn_ + z * h
        hs.append(h); zs.append(z)
    ref = arch26.forward(m, x, False, True)[1]["h"]
    same = all(T.equal(a, b) for a, b in zip(hs, ref)) and len(hs) == len(ref)
    assert same, "the repeated step does not reproduce arch26.forward's states bit for bit"
    return T.stack(zs, 1), dict(states_bit_identical_to_arch26_forward=same, steps=L)


# ---------------------------------------------------------------- the measures
def quartiles(v):
    if v.numel() == 0:
        return None
    q = T.quantile(v.double(), T.tensor([.25, .5, .75], dtype=T.float64))
    return [round(float(a), 4) for a in q]


def pct(k, n):
    return round(100 * k / n, 1) if n else None


def measure(Z, roles):
    """Z (N, L, U); roles: list of N role lists. Per-unit arrays and the per-file summary."""
    N, L, U = Z.shape
    role_idx = T.tensor([[ROLES.index(r) for r in rr] for rr in roles])           # (N, L)
    letter = role_idx == ROLES.index("letter")
    start = role_idx == ROLES.index("start")
    off = ~letter & ~start
    has_letters = bool(letter.any())
    if has_letters:
        nl = letter.sum(1)
        assert bool((nl == nl[0]).all()), "items with different numbers of letters"
    Zf = Z.reshape(N * L, U)
    big = T.full_like(Z, 2.0)
    # lowest z off letters (start excluded) and the role where it occurs
    zo = T.where(off[..., None], Z, big).reshape(N * L, U)
    low, arg = zo.min(0)
    rewrite = [ROLES[int(role_idx.reshape(-1)[i])] for i in arg]
    zo_s = T.where((~letter)[..., None], Z, big).reshape(N * L, U)
    low_s, arg_s = zo_s.min(0)
    rewrite_s = [ROLES[int(role_idx.reshape(-1)[i])] for i in arg_s]
    mean_z = (Z * (~start)[..., None]).sum((0, 1)) / (~start).sum()
    if has_letters:
        hold = ((Z > 0.9) & letter[..., None]).sum((0, 1)).double() / letter.sum()
        Zd = Z.double()
        prod = T.where(letter[..., None], Zd, T.ones_like(Zd)).prod(1)             # (N, U): product over the item's letters
        ret = prod.mean(0)
        holder = (hold >= 0.9) & (low <= 0.1)
    else:
        hold = ret = holder = None
    units = dict(lowest_off_letters=[round(float(a), 6) for a in low], rewrite_role=rewrite,
                 lowest_incl_start=[round(float(a), 6) for a in low_s], rewrite_role_incl_start=rewrite_s,
                 mean_z=[round(float(a), 6) for a in mean_z],
                 hold_share_letters=None if hold is None else [round(float(a), 6) for a in hold],
                 retention_letters=None if ret is None else [float(a) for a in ret],
                 gated_holder=None if holder is None else [bool(a) for a in holder])

    def block(sel):
        n = int(sel.sum())
        d = dict(units=n, mean_of_mean_z=round(float(mean_z[sel].mean()), 4) if n else None)
        if has_letters:
            d.update(gated_holders=int(holder[sel].sum()), gated_holder_pct=pct(int(holder[sel].sum()), n),
                     retention_quartiles=quartiles(ret[sel]), retention_ge_half=int((ret[sel] >= .5).sum()),
                     retention_ge_half_pct=pct(int((ret[sel] >= .5).sum()), n),
                     hold_share_ge_0p9=int((hold[sel] >= .9).sum()), lowest_off_le_0p1=int((low[sel] <= .1).sum()))
        else:
            d.update(gated_holders=None, note="no distractor letters in this panel: holding and retention through letters are not defined")
        return d

    allu = T.ones(U, dtype=T.bool)
    summary = dict(items=N, positions=L, letters_per_item=int(letter[0].sum()), **block(allu))
    summary["by_rewrite_role"] = {r: block(T.tensor([x == r for x in rewrite])) for r in ROLES if r in rewrite}
    summary["by_rewrite_role_incl_start"] = {r: int(sum(x == r for x in rewrite_s)) for r in ROLES if r in rewrite_s}
    return units, summary


def run_panel(m, items):
    x, y, mask = X.batch_tensor(items)
    roles = [roles_of(row) for row in x]
    Z, chk = gates(m, x)
    units, summary = measure(Z, roles)
    return units, summary, chk, roles[0]


# ---------------------------------------------------------------- the hand-built check
def hand_net():
    """Unit 0: z = 1 on letters, z = 0 at every other position. Unit 1 (and every other unit): z = 0.5 everywhere."""
    m = arch26.new_net(0)
    U = m.units
    with T.no_grad():
        for p in m.parameters():
            p.zero_()
        flag = T.full((258,), -1.0)
        flag[65:91] = 1.0                                       # uppercase letters, the distractors
        m.book.weight[:, 0] = flag
        m.l1.weight[0, 0] = 10.0                                # z1[0] = tanh(+-10)
        m.rnn.weight_ih_l0[U + 0, 0] = 200.0                    # z of unit 0 = sigmoid(+-200 * 0.99999...) = 1 or 0 exactly in float32
    return m


def check():
    OUT.mkdir(parents=True, exist_ok=True)
    m = hand_net()
    res = {}
    for k in (12, 20):
        items = add89_panel(SEEDS[0], k)                       # item bytes only; no trained file is opened
        x, _, _ = X.batch_tensor(items)
        Z, chk = gates(m, x)
        roles = [roles_of(row) for row in x]
        L = T.tensor([[r == "letter" for r in rr] for rr in roles])
        z0_ok = bool((Z[..., 0][L] == 1).all() and (Z[..., 0][~L] == 0).all())
        z1_ok = bool((Z[..., 1] == 0.5).all())
        units, summ = measure(Z, roles)
        want = dict(unit0_holder=True, unit0_retention=1.0, unit1_holder=False, unit1_retention_is_not_1=True)
        got = dict(unit0_holder=units["gated_holder"][0], unit0_retention=units["retention_letters"][0],
                   unit1_holder=units["gated_holder"][1], unit1_retention_is_not_1=units["retention_letters"][1] != 1.0)
        # negative control: unit 0 inverted (z = 0 on letters, 1 elsewhere) must not be a holder and must retain nothing
        m2 = hand_net()
        with T.no_grad():
            m2.rnn.weight_ih_l0[m2.units, 0] = -200.0
        Z2, _ = gates(m2, x)
        u2, _ = measure(Z2, roles)
        res[k] = dict(construction_z_unit0_is_1_on_letters_0_elsewhere=z0_ok, construction_z_unit1_is_half_everywhere=z1_ok,
                      expected=want, measured=got,
                      unit0=dict(hold_share=units["hold_share_letters"][0], lowest_off=units["lowest_off_letters"][0],
                                 rewrite_role=units["rewrite_role"][0]),
                      unit1=dict(hold_share=units["hold_share_letters"][1], lowest_off=units["lowest_off_letters"][1],
                                 retention=units["retention_letters"][1], expected_retention=0.5 ** k),
                      inverted_unit0=dict(holder=u2["gated_holder"][0], retention=u2["retention_letters"][0]),
                      holders_in_net=summ["gated_holders"], arch26=chk,
                      passed=z0_ok and z1_ok and got == want and abs(units["retention_letters"][1] - 0.5 ** k) < 1e-12
                      and u2["gated_holder"][0] is False and u2["retention_letters"][0] == 0.0 and summ["gated_holders"] == 1)
    res["passed"] = all(v["passed"] for v in res.values() if isinstance(v, dict))
    res["note"] = "Built from zeroed weights; the add89 item bytes of seed 73129 are generated by the scorer's functions. No trained file opened."
    res["torch"] = T.__version__
    (OUT / "E29_HAND_CHECK.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v["passed"] if isinstance(v, dict) else v) for k, v in res.items()}, indent=1))
    return res["passed"]


# ---------------------------------------------------------------- files
def receipt_sha(rec):
    c = rec.get("complete")
    return c["weights_sha256"] if isinstance(c, dict) else rec["weights_sha256"]


def load(path):
    rec = json.loads(path.with_suffix(".json").read_text())
    want = receipt_sha(rec)
    sha = X16.sha_file(path)
    assert sha == want, f"{path.name}: sha256 {sha[:12]} is not its receipt's {want[:12]}"
    return sha


def files(group):
    out = []
    for s in SEEDS:
        if group == "untrained":
            out.append((f"untrained_G1_{s}", s, None, False))
        elif group in ("births_seen", "births_unseen"):
            if (s in SEEN) == (group == "births_seen"):
                out += [(f"birth_{a}_{s}", s, E26 / f"e26_dev_{a}_{s}.pt", False) for a in ("S", "B")]
        elif group == "lives":
            out += [(f"life_{a}{a}_{s}", s, E26 / f"e26_life_{a}_{s}.pt", True) for a in ("S", "B")]
        elif group == "cross":
            out += [(f"life_{c}_{s}", s, E26X / f"e26x_life_{c}_{s}.pt", True) for c in ("SB", "BS")]
        else:
            raise SystemExit(f"unknown group {group}")
    return out


def measure_group(group):
    OUT.mkdir(parents=True, exist_ok=True)
    hc = json.loads((OUT / "E29_HAND_CHECK.json").read_text()) if (OUT / "E29_HAND_CHECK.json").exists() else {}
    assert hc.get("passed") is True, "the hand-built check has not passed; no trained file is read"
    if group != "untrained":
        assert (HERE / "E29_PREDICTIONS.md").exists(), "E29_PREDICTIONS.md must exist before any trained file is read"
    for tag, seed, path, is_life in files(group):
        t0 = time.process_time()
        m = arch26.new_net(seed)
        src = dict(kind="untrained (arch26.new_net(seed), as E26's births start)", tensors_sha256=R.tensors_sha(m.state_dict()))
        if path is not None:
            sha = load(path)
            m.load_state_dict(T.load(path))
            src = dict(file=str(path.relative_to(HERE)), sha256=sha, matches_receipt=True)
        m.eval()
        panels = {f"add89_k{k}": add89_panel(seed, k) for k in (12, 20)}
        if is_life:
            panels["plus_minus_composition"] = plus_minus_panel(seed)
        rec = dict(tag=tag, seed=seed, source=src, judged=(group not in ("births_seen",)), panels={})
        if group == "births_seen":
            rec["note"] = "Seen by the auditor and the coordinator before the predictions: reported, not judged (brief, revision 2)."
        for name, items in panels.items():
            units, summ, chk, roles0 = run_panel(m, items)
            rec["panels"][name] = dict(items_sha256=sha_items(items), roles_of_first_item=roles0, check=chk, summary=summ, units=units)
        rec["cpu_s"] = round(time.process_time() - t0, 2)
        rec["torch"] = T.__version__
        (OUT / f"e29_{tag}.json").write_text(json.dumps(rec, indent=1))
        s12, s20 = rec["panels"]["add89_k12"]["summary"], rec["panels"]["add89_k20"]["summary"]
        print(tag, "holders% k12/k20", s12["gated_holder_pct"], s20["gated_holder_pct"], "ret>=.5% k12/k20",
              s12["retention_ge_half_pct"], s20["retention_ge_half_pct"], f"cpu {rec['cpu_s']}s", flush=True)


def summary():
    rows = {}
    for f in sorted(OUT.glob("e29_*.json")):
        r = json.loads(f.read_text())
        rows[r["tag"]] = dict(seed=r["seed"], judged=r["judged"], source=r["source"],
                              panels={k: v["summary"] for k, v in r["panels"].items()})
    (OUT / "E29_GATES_SUMMARY.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} files summarised")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "check":
        sys.exit(0 if check() else 1)
    if cmd == "measure":
        measure_group(sys.argv[2]); sys.exit(0)
    if cmd == "summary":
        summary(); sys.exit(0)
    raise SystemExit(__doc__)
