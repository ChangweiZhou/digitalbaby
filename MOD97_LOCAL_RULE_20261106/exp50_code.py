"""E50 runner (brief OPUS_TASK_E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE_20261116.md, revision 2): E42's F10 (256 cells, 10
fixed local connections) fed by a fixed input code in which a value stands for one byte, with E49's masks on the input map.

The network is the specification file e50_sparse_code.py (a copy of PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e50_sparse_code_spec.py
that differs in its two import lines only), imported and never rewritten: its code table, its value of a turned-on entry and its
forward. The cells are E42's (e42_cells.py), the masks and their start E49's (e49_interface.py, byte-identical to E49's worker's copy at
a36de661). Everything else is E49's runner exp49_interface.py (byte-identical to a36de661) and through it E42's runner exp42_cells.py,
both imported unchanged: the data, the optimizer, the panels, the digests, E42's check of the recurrent mask and E49's check of the two
masks at every scored row.

This file adds: the settings of revision 2 and the draw of `C46-one` (the audit's control, change 1); the birth loop around E50's forward;
a scorer that gives E42's scores and, from the same answers, the two answer bytes apart on R's items and the four operations apart;
the check that the code table, the byte table and l1 stay bit-equal to their start; the figure "by what a cell knows"; the two counts of
multiply-adds; the receipts; and the checks before the runs.

    python3 exp50_code.py run <setting> <S|B> <seed> [--dose N] [--out DIR] [--verify-scorer]
    python3 exp50_code.py check <1|3|4|5|figures> [--qual DIR] [--settings a,b] [--rules SB]
    settings: C-dense (512 values a cell) C16-in (16; E49's lists for K_IN = 16) C46-one (46: one value of every used byte); readout dense
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C26        # noqa: E402  first harness import (environment gate)

import torch as T                    # noqa: E402

import arch26                        # noqa: E402
import e42_cells as L42              # noqa: E402  E42's specification (the runner's copy)
import e49_interface as I49          # noqa: E402  E49's specification (the runner's copy, as E49's worker committed it)
import e50_sparse_code as S50        # noqa: E402  E50's specification (the runner's copy), imported, not rewritten
import exp42_cells as R42            # noqa: E402  E42's runner, imported unchanged
import exp49_interface as R49        # noqa: E402  E49's runner, imported unchanged

R_, X16, E, X = C26.R, C26.X16, C26.E, C26.X
T.set_num_threads(1)

U, D, ON = S50.U, S50.D, S50.ON
USED, NEEDED = S50.USED, S50.NEEDED
DIGITS = list(b"0123456789")
SEED = 73129
ROW = R42.ROW                        # 256
RULES = R42.RULES                    # S: the sweep (local=True); B: backpropagation
OUT = X.OUT / "e50"
QUAL = X.OUT / "e50_qual"
SPEC = HERE.parent / "PROJECT_DOCUMENTS" / "COORD_CHECKS_20261116" / "e50_sparse_code_spec.py"
SPEC_OUT = SPEC.parent / "OUTPUT_E50_SPEC_MAC.txt"
SOURCES = ("exp50_code.py", "e50_sparse_code.py", "exp49_interface.py", "e49_interface.py", "exp42_cells.py", "e42_cells.py") + C26.SOURCES
SETTINGS = {"C-dense": (512, 256), "C16-in": (16, 256), "C46-one": (46, 256)}
DOSES = {"C-dense": 24576, "C16-in": 49152, "C46-one": 49152}
FIRST_DOSE = 24576                   # the dose of C-dense, of F10 and of E49: a longer run is also read within it
# the brief's two counts (revision 2, the table of settings)
WEIGHTS_TAKING_PART = {"C-dense": 926464, "C16-in": 545536, "C46-one": 568576}
PRODUCTS_TURNED_ON = {"C-dense": 285184, "C16-in": 277000, "C46-one": 277504}
# the reviewer's figures for seed 73129 (the audit at ead8dcd8), which this runner must give
REVIEWER_73129 = dict(code_digest="12111044f0f8bc70", amplitude=3.2387, mean_squared_length=115.38, one_digest="6788d5475af8c847",
                      first_batch_loss={"C-dense": 5.597, "C16-in": 5.569, "C46-one": 5.591},
                      input_digest={"C-dense": "7b0701bb680e83f3", "C16-in": "0e09b9ff0545e68d"}, readout_digest="b27cc82e87e68c52",
                      groups=(95, 131, 30))
COPIES = {  # files copied byte for byte from E49's worker's branch claude/peaceful-noether-vcvrmf at a36de661, with their SHA-256
    "e49_interface.py": "92f968e092f2d27eaff4ee7e93ecbbb82b053cc88b6a2d3f967adadf7e3791f1",
    "exp49_interface.py": "cdf93ce167d562c8a3765c1ea4f98f74623469492324a85aa1eb34216b7ccbf3",
}
dg16 = lambda v: hashlib.sha256(json.dumps(v).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- the code and the draws
def draw_code(seed):
    """own[b]: the 11 values byte b turns on, in the order the shuffled list gives them (the brief's text; S50.code_table makes the
    same draw, and build() asserts that the two agree)."""
    vals = list(range(D)); Random(f"e50|code|{seed}").shuffle(vals)
    return {b: vals[ON * k: ON * k + ON] for k, b in enumerate(USED)}


def draw_one(seed):
    """`C46-one` (revision 2): for cells 0 to 255 in turn and, within a cell, for the 46 symbols in the code's order, the cell reads
    own[k][rng.randrange(11)], rng = Random(f"e50|one|{seed}"). Returns the 256 lists, each sorted."""
    own = draw_code(seed); rng = Random(f"e50|one|{seed}")
    return [sorted(own[b][rng.randrange(ON)] for b in USED) for _ in range(U)]


def input_lists(setting, seed):
    if setting == "C46-one":
        return draw_one(seed)
    return I49.draws(seed, SETTINGS[setting][0], U)[0]


class Net:
    """One setting's network at its start: the module, E42's layer, the interface (E49's object), the code table and what describes it."""


def build(setting, seed):
    """arch26.new_net; the value of a turned-on entry from it (S50.amplitude_for) and the code table (S50.code_table); E42's F10 start;
    then the start of the input map: E49's Interface for C-dense and C16-in, and for C46-one E49's Interface with full masks whose
    input lists are replaced by draw_one's, the kept weights of the input map scaled by sqrt(512 / 46) and the others set to zero."""
    k_in, k_out = SETTINGS[setting]
    n = Net()
    m = arch26.new_net(seed)
    with T.no_grad():
        z1 = T.tanh(m.l1(m.book(T.tensor(USED))))
        n.mean_squared_length = float(z1.pow(2).sum(1).mean())
    n.amplitude = S50.amplitude_for(m, seed)
    n.table, n.owner = S50.code_table(seed, n.amplitude)
    n.own = draw_code(seed)
    assert all(n.owner[v] == b for b, vs in n.own.items() for v in vs) and sum(o >= 0 for o in n.owner) == ON * len(USED)
    layer = L42.Layer(m, seed, N=10)                 # F10: period set "1", variant "step", the local neighbourhood
    if setting == "C46-one":
        iface = I49.Interface(m, seed, D, U)         # both masks full: nothing is scaled
        lists = draw_one(seed)
        iface.k_in = k_in; iface.full = False
        iface.mask_in = T.zeros(U, D, dtype=T.bool)
        for i, s in enumerate(lists):
            assert len(s) == k_in == len(set(s))
            iface.mask_in[i, s] = True
        with T.no_grad():
            m.rnn.weight_ih_l0.mul_(iface.mask_in.repeat(3, 1).float() * (D / k_in) ** 0.5)
    else:
        iface = I49.Interface(m, seed, k_in, k_out)
        lists = I49.draws(seed, k_in, k_out)[0]
    assert all(sorted(iface.mask_in[i].nonzero().flatten().tolist()) == lists[i] for i in range(U))
    n.m, n.layer, n.iface, n.lists, n.setting, n.seed = m, layer, iface, lists, setting, seed
    n.book0, n.l1w0, n.l1b0 = m.book.weight.detach().clone(), m.l1.weight.detach().clone(), m.l1.bias.detach().clone()
    n.table_sha = R42.tsha(n.table)
    return n


def fwd(n, x, local, silence=False):
    return S50.forward(n.m, n.layer, n.iface, n.table, x, local, silence=silence)


def install(n):
    """arch26.install(), then exp11_cell.forward replaced by a function of the harness's signature that runs E50's forward (scoring)."""
    arch26.install()

    def f(m, x, local, through_time, keep=False, silence=False):
        assert m is n.m
        lg, hs = fwd(n, x, local, silence=silence)
        return lg, dict(h=hs)
    E.forward = f


def step(n, opt, items, local):
    """exp26_candidate.step's text with E50's forward in place of arch26.forward (E49's step with the fixed code)."""
    x, y, mask = X.batch_tensor(items)
    logits = fwd(n, x, local)[0]
    loss = E.cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach())


# ---------------------------------------------------------------- what describes a setting
def counts(n):
    """The brief's two counts of multiply-adds a position. First: every weight of a map inside its mask, the cells' kept recurrent
    weights, l2 and l3, without l1 (E49's count without l1's 4,096). Second: products with a value that is not zero, taking each of the
    46 used bytes as the present byte in turn (their mean; a used byte turns on 11 values)."""
    m, f = n.m, n.iface
    n_rec = int(n.layer.mask3().sum())
    k_in = int(f.mask_in.sum(1).max())
    first = 3 * int(f.mask_in.sum()) + int(f.mask_out.sum()) + n_rec + m.l2.weight.numel() + m.l3.weight.numel()
    per_byte = [3 * int(f.mask_in[:, n.own[b]].sum()) for b in USED]
    rest = ON * m.l2.weight.shape[0] + int(f.mask_out.sum()) + n_rec + m.l3.weight.numel()
    weights = sum(q.numel() for q in m.parameters()) - m.book.weight.numel() - m.l1.weight.numel() - m.l1.bias.numel() \
        - m.rnn.weight_ih_l0.numel() - m.lm.weight.numel() - m.rnn.weight_hh_l0.numel() + 3 * int(f.mask_in.sum()) + int(f.mask_out.sum()) + n_rec
    return dict(weights_taking_part=first, input_map=3 * int(f.mask_in.sum()), lm=int(f.mask_out.sum()), cells=n_rec,
                l2=m.l2.weight.numel(), l3=m.l3.weight.numel(), k_in=k_in,
                products_with_a_turned_on_value=sum(per_byte) / len(per_byte) + rest,
                products_input_map_by_byte=dict(min=min(per_byte), max=max(per_byte), mean=sum(per_byte) / len(per_byte)),
                products_l2=ON * m.l2.weight.shape[0],
                parameters_trained=weights, parameters_untrained_byte_table_and_l1=m.book.weight.numel() + m.l1.weight.numel() + m.l1.bias.numel())


def sees(lists, owner):
    """For each cell, the bytes for which it reads at least one value."""
    return [sorted({owner[v] for v in s if owner[v] >= 0}) for s in lists]


def coverage(n):
    see = sees(n.lists, n.owner)
    rng = lambda v: dict(min=min(v), median=sorted(v)[len(v) // 2], max=max(v))
    return dict(bytes_of_46=rng([len(s) for s in see]), bytes_of_20=rng([len(set(s) & set(NEEDED)) for s in see]),
                digits_of_10=rng([len(set(s) & set(DIGITS)) for s in see]), cells_without_input=sum(1 for s in see if not s),
                cells_a_needed_byte_reaches=rng([sum(b in s for s in see) for b in NEEDED]),
                values_read_by_no_cell=int((n.iface.mask_in.sum(0) == 0).sum()),
                weights_behind_a_cell_and_needed_byte={str(k): sum(1 for i in range(U) for b in NEEDED if len(set(n.lists[i]) & set(n.own[b])) == k)
                                                       for k in range(0, ON + 1)})


def digit_groups(seed):
    """The three groups of "by what a cell knows": cells that have input for 0 to 2, 3 or 4, and 5 or more of the ten digits when each
    cell reads E49's 16 values (the lists of C16-in). The same groups are used in every setting."""
    own = draw_code(seed); owner = [-1] * D
    for b, vs in own.items():
        for v in vs:
            owner[v] = b
    k = [len(set(s) & set(DIGITS)) for s in sees(I49.draws(seed, 16, U)[0], owner)]
    return [[i for i in range(U) if k[i] <= 2], [i for i in range(U) if 3 <= k[i] <= 4], [i for i in range(U) if k[i] >= 5]]


def knows_figure(n, groups):
    """Mean magnitude of the kept recurrent weights into the cells of each group (three gates), with and without the cell's reading
    of itself. It describes the size of weights and is not read as use."""
    W = n.m.rnn.weight_hh_l0.detach().abs(); mk = n.layer.mask
    eye = T.eye(U, dtype=T.bool); out = {}
    for name, g in zip(("digits_0_to_2", "digits_3_to_4", "digits_5_or_more"), groups):
        idx = T.tensor(g)
        rows = T.cat([idx, idx + U, idx + 2 * U])
        kp = mk[idx].repeat(3, 1); kn = (mk & ~eye)[idx].repeat(3, 1)
        out[name] = dict(cells=len(g), with_self=float(W[rows][kp].mean()), neighbours_only=float(W[rows][kn].mean()))
    return out


def input_size(n):
    """Root mean square, over the 256 cells and three gates, of a cell's input for a byte without the bias (the code's row times the
    kept weights of the input map), for the 20 needed bytes and for the ten digits; and the share of cell-and-needed-byte pairs
    with no input."""
    with T.no_grad():
        gi = n.table[T.tensor(NEEDED)] @ (n.m.rnn.weight_ih_l0 * n.iface.mask_in.repeat(3, 1).float()).T      # (20, 768)
        gd = n.table[T.tensor(DIGITS)] @ (n.m.rnn.weight_ih_l0 * n.iface.mask_in.repeat(3, 1).float()).T
    return dict(needed_rms=float(gi.pow(2).mean().sqrt()), digits_rms=float(gd.pow(2).mean().sqrt()))


def fixed(n):
    """Check 2 and change 7: the code table has the SHA-256 of the start, and the byte table and l1 are bit-equal to their start."""
    m = n.m
    return dict(code_table_sha256_equal=R42.tsha(n.table) == n.table_sha, byte_table_bit_equal=bool(T.equal(m.book.weight.detach(), n.book0)),
                l1_bit_equal=bool(T.equal(m.l1.weight.detach(), n.l1w0)) and bool(T.equal(m.l1.bias.detach(), n.l1b0)))


def map_grads(n, x, y, mask, local):
    for q in n.m.parameters():
        q.grad = None
    lg = fwd(n, x, local)[0]
    E.cross_entropy(lg[mask], y[mask]).backward()
    P = dict(n.m.named_parameters())
    g = {k: P[k].grad.detach().clone() for k in ("rnn.weight_ih_l0", "lm.weight", "rnn.weight_hh_l0")}
    untouched = P["book.weight"].grad is None and P["l1.weight"].grad is None and P["l1.bias"].grad is None
    for q in n.m.parameters():
        q.grad = None
    return g, untouched


def cosines(n, items):
    """Cosine between the sweep's and the exact gradient of each of the three maps (over its kept entries), on one batch; the optimizer
    is not touched (E49's measure with E50's forward). Also whether the byte table and l1 are without gradient under both rules."""
    x, y, mask = X.batch_tensor(items)
    (gB, uB), (gS, uS) = map_grads(n, x, y, mask, False), map_grads(n, x, y, mask, True)
    cs = lambda a, b: float(T.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0))
    kp = R49.kept(n.layer, n.iface)
    out = {k: cs(gB[k][kp[k]], gS[k][kp[k]]) for k in gB}
    out["byte_table_and_l1_without_gradient"] = uB and uS
    out["gradients_zero_outside_masks"] = all(bool((g[k][~kp[k]] == 0).all()) for g in (gB, gS) for k in g)
    return out


def first_batch_loss(n, items):
    x, y, mask = X.batch_tensor(items)
    with T.no_grad():
        return {r: float(E.cross_entropy(fwd(n, x, loc)[0][mask], y[mask])) for r, loc in RULES.items()}


# ---------------------------------------------------------------- scoring: E42's scores, and from the same answers the brief's additions
def score_all(m, A, Rs, TA, LP):
    """E42's score_all (exp42_cells.score_all: exact answers on A, R, the taught panel and the length test), computed from the answers
    of exp11_cell.respond, with from the same answers: on R's items the share with the first answer byte right and with the second;
    and on the panel behind A the four operations apart (its counts hold all four already)."""
    ans = lambda it: E.respond(m, [p for p, _ in it])
    exact = lambda outs, it: sum(o == t for o, (_, t) in zip(outs, it))
    a = {k: exact(ans(it), it) for k, it in A.items()}
    r, b1, b2 = {}, {}, {}
    for k, it in Rs.items():
        outs = ans(it)
        r[k] = exact(outs, it)
        b1[k] = sum(len(o) > 0 and o[0] == t[0] for o, (_, t) in zip(outs, it))
        b2[k] = sum(len(o) > 1 and o[1] == t[1] for o, (_, t) in zip(outs, it))
    ta = {k: exact(ans(it), it) for k, it in TA.items()}
    ln = {str(k): exact(ans(it), it) for k, it in LP.items()}
    nR = sum(len(it) for it in Rs.values())
    return dict(A_counts=a, R_counts=r, taught_counts=ta, length_counts=ln,
                A=100 * (a["add89"] + a["sub89"]) / 600, R=100 * (r["add89"] + r["sub89"]) / 2000,
                taught=100 * (ta["add89"] + ta["sub89"]) / 600, len12=100 * ln["12"] / 300, len40=100 * ln["40"] / 300,
                R_first_byte_counts=b1, R_second_byte_counts=b2, R_first_byte=100 * sum(b1.values()) / nR, R_second_byte=100 * sum(b2.values()) / nR,
                ops={k: 100 * v / len(A[k]) for k, v in a.items()})


E42_SCORE_KEYS = ("A_counts", "R_counts", "taught_counts", "length_counts", "A", "R", "taught", "len12", "len40")


# ---------------------------------------------------------------- one run
def tag_of(setting, dose):
    return "" if dose == DOSES[setting] else f"_dry{dose}"


def receipt_path(out, setting, rule, seed, dose):
    return Path(out) / f"e50_{setting}_{rule}{tag_of(setting, dose)}_{seed}.json"


def weights_blob(n):
    """E49's weight file (the module's tensors, E42's mask and E49's two masks) with the code: `code_table` (258 x 512), `code_owner`
    (512 entries: the byte that turns the value on, or -1) and `code_amplitude` (the value of a turned-on entry)."""
    full = R49.weights_blob(n.m, n.layer, n.iface)
    full.update(code_table=n.table.clone(), code_owner=T.tensor(n.owner, dtype=T.long), code_amplitude=T.tensor(n.amplitude))
    return full


def versions():
    import numpy
    return dict(python=sys.version.split()[0], torch=T.__version__, numpy=numpy.__version__, torch_threads=T.get_num_threads(),
                OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"), MKL_NUM_THREADS=os.environ.get("MKL_NUM_THREADS"))


def run(setting, rule, seed, out, cmd, dose=None, verify_scorer=False):
    k_in, k_out = SETTINGS[setting]
    local = RULES[rule]
    dose = dose or DOSES[setting]
    assert dose % ROW == 0
    rp = receipt_path(out, setting, rule, seed, dose)
    paths = dict(receipt=rp, start=rp.with_name(rp.stem + "_start.pt"), weights=rp.with_suffix(".pt"),
                 answers=rp.with_name(rp.stem + "_answers.json"))
    mid = dose > FIRST_DOSE
    if mid:
        paths.update(weights_first_dose=rp.with_name(rp.stem + f"_at{FIRST_DOSE}.pt"), answers_first_dose=rp.with_name(rp.stem + f"_at{FIRST_DOSE}_answers.json"))
    R_.refuse_existing(*paths.values(), *(str(p) + ".tmp" for p in paths.values()))
    Path(out).mkdir(parents=True, exist_ok=True)
    n = build(setting, seed); m, layer, iface = n.m, n.layer, n.iface
    cov = coverage(n); rd = R49.readers(iface)
    print(f"e50 {setting} {rule} {seed}: value of a turned-on entry {n.amplitude:.4f}; cells without input {cov['cells_without_input']}; "
          f"a needed byte reaches at least {cov['cells_a_needed_byte_reaches']['min']} cells", flush=True)
    if cov["cells_without_input"] or cov["cells_a_needed_byte_reaches"]["min"] == 0 or rd["cell_min"] == 0:
        raise SystemExit(f"e50 {setting} {seed}: a cell without input, a needed byte that reaches no cell, or a cell that is not read; stopped")
    install(n)
    opt = R42.make_opt(m, layer)
    cnt = counts(n)
    assert cnt["weights_taking_part"] == WEIGHTS_TAKING_PART[setting], (setting, cnt)
    assert abs(cnt["products_with_a_turned_on_value"] - PRODUCTS_TURNED_ON[setting]) <= 100, (setting, cnt)
    ma = cnt["weights_taking_part"]
    groups = digit_groups(seed)
    A, Rs, TA, LP = R42.panels(seed)
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    bt = E.batches(tp, seed, E.KMAX["M"]); dg = C26.Digest()
    R49.save_pt(weights_blob(n), paths["start"])
    rec = dict(experiment="E50", kind="birth", setting=setting, config=setting, rule=rule, local=local, seed=seed, dose=dose, row_every=ROW,
               k_in=k_in, k_out=k_out, cells=U, N=layer.N, batch=X.BATCH, kmax=E.KMAX["M"], counts=cnt, multiply_adds_per_position=ma,
               multiply_adds_per_position_second_count=cnt["products_with_a_turned_on_value"],
               params_module=sum(q.numel() for q in m.parameters()), readers=rd, coverage=cov,
               code=dict(digest=dg16(n.owner), amplitude=n.amplitude, mean_squared_length_of_present_code=n.mean_squared_length,
                         used=len(USED), values_owned=sum(o >= 0 for o in n.owner), table_sha256=n.table_sha,
                         names_in_weight_files=dict(table="code_table", owners="code_owner", amplitude="code_amplitude")),
               draw_digests=dict(input=dg16(n.lists), readout=dg16(I49.draws(seed, D, k_out)[1])),
               start_scale=dict(input=(D / k_in) ** 0.5 if k_in < D else 1.0, lm=1.0),
               digit_groups=dict(sizes=[len(g) for g in groups], note="cells with input for 0 to 2, 3 or 4, 5 or more of the ten digits under E49's 16 values a cell"),
               optimizer=R42.opt_record(m, layer, opt),
               update="exp26_candidate.step's text with e50_sparse_code.forward(m, layer, iface, table, x, local) in place of arch26.forward",
               data="exp11_cell.batches([(t, split(t, .5, seed)[0]) for t in DEV], seed, 12)",
               forward="e50_sparse_code.forward (the specification's own forward)",
               start_file=paths["start"].name, start_sha256=X16.sha_file(paths["start"]), start_tensors_sha256=R_.tensors_sha(m.state_dict()),
               mask_sha256=dict(recurrent=R42.tsha(layer.mask), input=R42.tsha(iface.mask_in), readout=R42.tsha(iface.mask_out)),
               versions=versions(), provenance=provenance(cmd), complete=False, curve=[])
    curve = rec["curve"]; cpu_up = cpu_sc = 0.0; w0 = time.time(); losses = []; first = None

    def score_row(u):
        nonlocal cpu_sc
        c = time.process_time()
        sc = score_all(m, A, Rs, TA, LP)
        cpu_sc += time.process_time() - c
        row = dict(step=u, **sc)
        if verify_scorer:
            ref = R42.score_all(m, A, Rs, TA, LP)
            row["scorer_equals_e42"] = all(sc[k] == ref[k] for k in E42_SCORE_KEYS)
            assert row["scorer_equals_e42"], f"this file's scorer differs from E42's at row {u}"
        lim = R42.check_limit(m, layer)
        assert all(lim.values()), f"E42's check 4 failed at row {u}: {lim}"
        mk = R49.check_masks(m, layer, iface, opt)
        assert all(all(v.values()) for v in mk.values()), f"E49's check 3 failed at row {u}: {mk}"
        fx = fixed(n)
        assert all(fx.values()), f"check 2 failed at row {u}: {fx}"
        row.update(limit=lim, masks=mk, fixed=fx, kept_mean_magnitude=R49.magnitudes(m, layer, iface), by_what_a_cell_knows=knows_figure(n, groups),
                   input_size=input_size(n), cpu_updates=cpu_up, cpu_scoring=cpu_sc, multiply_adds=ma * X.BATCH * u, data_digest=dg.value(),
                   loss=(sum(losses) / len(losses)) if losses else None, finite_losses=all(v == v and abs(v) != float("inf") for v in losses),
                   wall=time.time() - w0)
        curve.append(row)
        o = row["ops"]
        print(f"e50 {setting} {rule} {seed} {u}: A {row['A']:.2f} R {row['R']:.2f} (bytes {row['R_first_byte']:.1f} {row['R_second_byte']:.1f}) "
              f"taught {row['taught']:.2f} len12 {row['len12']:.1f} len40 {row['len40']:.1f} max {o['max89']:.0f} min {o['min89']:.0f} "
              f"cpu {cpu_up:.0f}+{cpu_sc:.0f}s", flush=True)

    score_row(0)
    for u in range(1, dose + 1):
        items = next(bt)
        if first is None:
            first = items
            rec["first_batch_loss_start"] = first_batch_loss(n, first)
            rec["cosine_first_batch_start"] = cosines(n, first)
        dg.add(items)
        c = time.process_time(); losses.append(step(n, opt, items, local)); cpu_up += time.process_time() - c
        if u % ROW == 0:
            score_row(u); losses = []
            if mid and u == FIRST_DOSE:
                R49.save_pt(weights_blob(n), paths["weights_first_dose"])
                paths["answers_first_dose"].write_text(json.dumps({str(k): E.respond(m, [p for p, _ in it]) for k, it in LP.items()}))
                rec["first_dose"] = dict(update=FIRST_DOSE, weights=paths["weights_first_dose"].name, weights_sha256=X16.sha_file(paths["weights_first_dose"]),
                                         tensors_sha256=R_.tensors_sha(m.state_dict()), answers_file=paths["answers_first_dose"].name,
                                         cosine_first_batch=cosines(n, first))
            if u % 1024 == 0 or u == dose:
                rp.write_text(json.dumps(rec, indent=1))
    rec["cosine_first_batch_end"] = cosines(n, first)
    sd = m.state_dict()
    R49.save_pt(weights_blob(n), paths["weights"])
    paths["answers"].write_text(json.dumps({str(k): E.respond(m, [p for p, _ in it]) for k, it in LP.items()}))
    fx = fixed(n)
    rec.update(complete=True, weights=paths["weights"].name, weights_sha256=X16.sha_file(paths["weights"]),
               tensors_sha256=R_.tensors_sha(sd), answers_file=paths["answers"].name,
               byte_table_and_l1_bit_equal_to_start=fx["byte_table_bit_equal"] and fx["l1_bit_equal"],
               code_table_sha256_equal_at_every_row=all(r["fixed"]["code_table_sha256_equal"] for r in curve) and fx["code_table_sha256_equal"],
               optimizer_steps=R_.opt_steps(opt), optimizer_read_back=R42.opt_record(m, layer, opt),
               optimizer_state_for=sorted(k for k, q in m.named_parameters() if opt.state.get(q)), data_digest=dg.value(),
               cpu_updates=cpu_up, cpu_scoring=cpu_sc, wall=time.time() - w0, processes_beside=os.environ.get("E50_BESIDE", "not recorded"))
    rp.write_text(json.dumps(rec, indent=1))
    print("e50 complete", rp.name, rec["tensors_sha256"][:16], "weights sha256", rec["weights_sha256"][:16], flush=True)
    return rec


# ---------------------------------------------------------------- provenance
def provenance(cmd):
    p = C26.provenance(cmd)
    p["sources_sha256"] = {s: X16.sha_file(HERE / s) for s in SOURCES}
    p["runner_sha256"] = X16.sha_file(HERE / "exp50_code.py")
    p["spec_sha256"] = X16.sha_file(SPEC) if SPEC.exists() else None
    p["copies_from_e49_worker_a36de661"] = {f: dict(sha256=X16.sha_file(HERE / f), expected=s, equal=X16.sha_file(HERE / f) == s) for f, s in COPIES.items()}
    assert all(v["equal"] for v in p["copies_from_e49_worker_a36de661"].values()), "a file copied from E49's worker is not byte-identical"
    return p


# ---------------------------------------------------------------- checks before the runs
def check1(qual, cmd):
    """The specification file: the runner's copy differs from it in its two import lines only; run on its own place (from a shell's
    environment), its self-checks print what OUTPUT_E50_SPEC_MAC.txt has."""
    import subprocess
    a = SPEC.read_text().splitlines(); b = (HERE / "e50_sparse_code.py").read_text().splitlines()
    diff_lines = [(i + 1, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    only_imports = len(a) == len(b) and len(diff_lines) == 2 \
        and diff_lines[0][1].startswith("import e42_cells_spec as E42") and diff_lines[0][2].startswith("import e42_cells as E42") \
        and diff_lines[1][1].startswith("import e49_interface_spec as E49") and diff_lines[1][2].startswith("import e49_interface as E49")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}
    t0 = time.time()
    res = subprocess.run([sys.executable, "-I", str(SPEC)], capture_output=True, text=True, env=env, cwd=str(HERE.parent))
    mine, ref = res.stdout.splitlines(), SPEC_OUT.read_text().splitlines()
    diffs = [(x, y) for x, y in zip(mine, ref) if x != y]
    ok = only_imports and res.returncode == 0 and len(mine) == len(ref) and not diffs
    return dict(copy_differs_in_two_import_lines_only=only_imports, differing_lines=diff_lines, spec_sha256=X16.sha_file(SPEC),
                copy_sha256=X16.sha_file(HERE / "e50_sparse_code.py"), returncode=res.returncode, stdout=mine, stderr=res.stderr[-3000:],
                lines_compared=len(ref), lines_equal=len(mine) == len(ref) and not diffs, differences=diffs, wall_spec=time.time() - t0,
                **{"pass": ok})


def check3(qual, cmd):
    """E49's checks 4 and 5, by E49's runner unchanged: the batch digest at row 4,096 is E26's for the seed; the scorer on E26's six
    births through E49's forward with both masks full and every cell connected (E26's births have a learned code). E49's check 3 (the
    masks hold) is made at every scored row of every run of this file, dry runs included, with E42's check of the recurrent mask.
    E49's checks 1 and 2 passed on E49's runner at a36de661 and are not run again."""
    c4 = R49.check4(qual, cmd)
    c5 = R49.check5(qual, cmd)
    return dict(e49_check4=c4, e49_check5=c5, e49_runner_sha256=X16.sha_file(HERE / "exp49_interface.py"),
                e49_checks_1_and_2="passed on E49's runner at a36de661 (E49_STAGE0_73129.md there); not run again",
                **{"pass": c4["pass"] and c5["pass"]})


def second_form(n, x):
    """The forward written cell by cell, without the code table and without masks: for every used byte and every cell, the cell's input
    is its bias plus the value of a turned-on entry times the sum of its weights on those of the byte's 11 values that it reads (index
    lists); a byte that is not used gives the bias alone. l2's part is formed the same way from the byte's 11 columns. The cells read
    their sources by index. Exact arithmetic only (no rule)."""
    m = n.m
    Wi, bi = m.rnn.weight_ih_l0.detach(), m.rnn.bias_ih_l0.detach()
    reads = [set(s) for s in n.lists]
    GI = bi.repeat(258, 1).clone(); L2 = m.l2.bias.detach().repeat(258, 1).clone()
    amp = T.tensor(n.amplitude)
    for b, vs in draw_code(n.seed).items():
        for i in range(U):
            mine = [v for v in vs if v in reads[i]]
            if mine:
                for g in range(3):
                    GI[b, g * U + i] = GI[b, g * U + i] + amp * Wi[g * U + i, mine].sum()
        L2[b] = L2[b] + amp * m.l2.weight.detach()[:, vs].sum(1)
    Wh, bh = m.rnn.weight_hh_l0.detach(), m.rnn.bias_hh_l0.detach()
    src = T.stack([n.layer.mask[i].nonzero().flatten() for i in range(U)])            # (256, 11): its 10 sources and itself
    cells = T.arange(U)
    Wg = [Wh[g * U + cells[:, None], src] for g in range(3)]
    B, L = x.shape
    h = T.zeros(B, U); hs = []
    for t in range(L):
        gi = GI[x[:, t]]
        hin = h[:, src]
        gh = [T.einsum("buk,uk->bu", hin, Wg[g]) + bh[g * U:(g + 1) * U] for g in range(3)]
        r = T.sigmoid(gi[:, :U] + gh[0]); z = T.sigmoid(gi[:, U:2 * U] + gh[1]); nn_ = T.tanh(gi[:, 2 * U:] + r * gh[2])
        h = (1 - z) * nn_ + z * h
        hs.append(h)
    Hs = T.stack(hs, 1)
    z2 = T.tanh(L2[x] + Hs @ m.lm.weight.detach().T + m.lm.bias.detach())
    return z2 @ m.l3.weight.detach().T + m.l3.bias.detach()


def check4(qual, cmd):
    """The forward against a second form, on the first batch of the seed: for C16-in under backpropagation the logits of the
    specification's forward equal those of second_form to 1e-5 (the brief's check). Reported beside it: the same for the other two
    settings, under the sweep's forward (the same values), and with every trained weight moved off its starting value inside its mask."""
    items = next(E.batches([(t, E.split(t, .5, SEED)[0]) for t in E.DEV], SEED, E.KMAX["M"]))
    x, y, mask = X.batch_tensor(items)
    out = dict(settings={}); tol = 1e-5
    for s in SETTINGS:
        r = {}
        for label in ("start", "perturbed"):
            n = build(s, SEED)
            if label == "perturbed":
                gen = T.Generator().manual_seed(13)
                with T.no_grad():
                    for name, q in n.m.named_parameters():
                        if name.startswith(("book.", "l1.")):
                            continue
                        q.add_(0.05 * T.randn(q.shape, generator=gen))
                    n.m.rnn.weight_ih_l0.mul_(n.iface.mask_in.repeat(3, 1).float())
                    n.m.rnn.weight_hh_l0.mul_(n.layer.mask3().float())
            with T.no_grad():
                ref = second_form(n, x)
                d = {rule: float((fwd(n, x, loc)[0] - ref).abs().max()) for rule, loc in RULES.items()}
            r[label] = dict(logits_max_abs_diff=d, largest_logit=float(ref.abs().max()), **{"pass": max(d.values()) <= tol})
        out["settings"][s] = r
        print(f"check 4 {s}: start {r['start']['logits_max_abs_diff']}, perturbed {r['perturbed']['logits_max_abs_diff']}", flush=True)
    out["tolerance"] = tol
    out["the_briefs_check"] = dict(setting="C16-in", rule="B", logits_max_abs_diff=out["settings"]["C16-in"]["start"]["logits_max_abs_diff"]["B"])
    out["pass"] = out["the_briefs_check"]["logits_max_abs_diff"] <= tol
    out["all_pass"] = all(v[l]["pass"] for v in out["settings"].values() for l in v)
    return out


def check5(qual, cmd, settings=tuple(SETTINGS), rules="SB"):
    """One dry run of 256 updates per setting and rule, with CPU time and a forecast for the setting's dose; the weights (start and
    end, with the code table, the owners and the masks) and the answers on the length-test panel are kept for the reviewer's reader.
    In these runs this file's scorer is compared with E42's at both scored rows."""
    out = dict(runs={})
    for s in settings:
        for rule in rules:
            rec = run(s, rule, SEED, Path(qual), cmd, dose=256, verify_scorer=True)
            pu = rec["cpu_updates"] / 256; pr = rec["cpu_scoring"] / len(rec["curve"])
            rows = DOSES[s] // ROW + 1
            out["runs"][f"{s} {rule}"] = dict(receipt=receipt_path(".", s, rule, SEED, 256).name, cpu_per_update=pu, cpu_per_scored_row=pr,
                                              forecast_cpu_h=(DOSES[s] * pu + rows * pr) / 3600, dose_forecast=DOSES[s], coverage=rec["coverage"],
                                              draw_digests=rec["draw_digests"], code_digest=rec["code"]["digest"],
                                              first_batch_loss_start=rec["first_batch_loss_start"],
                                              masks_every_row=all(all(all(v.values()) for v in r["masks"].values()) for r in rec["curve"]),
                                              code_fixed_every_row=all(all(r["fixed"].values()) for r in rec["curve"]),
                                              byte_table_and_l1_bit_equal_to_start=rec["byte_table_and_l1_bit_equal_to_start"],
                                              scorer_equals_e42=all(r["scorer_equals_e42"] for r in rec["curve"]),
                                              finite=all(r["finite_losses"] for r in rec["curve"]), len12=rec["curve"][-1]["len12"],
                                              loss_256=rec["curve"][-1]["loss"], cosine_start=rec["cosine_first_batch_start"],
                                              weights_sha256=rec["weights_sha256"], processes_beside=rec["processes_beside"])
            print(f"cost {s} {rule}: {pu:.4f} s/update, {pr:.2f} s/row, {out['runs'][f'{s} {rule}']['forecast_cpu_h']:.2f} CPU-h for {DOSES[s]:,} updates", flush=True)
    out["pass"] = all(v["masks_every_row"] and v["code_fixed_every_row"] and v["byte_table_and_l1_bit_equal_to_start"] and v["scorer_equals_e42"]
                      and v["finite"] for v in out["runs"].values())
    return out


def check_figures(qual, cmd):
    """The reviewer's figures for seed 73129 (the audit at ead8dcd8), from this runner: the digest of the code, the value of a turned-on
    entry, the digests of the masks and of C46-one's draw, the first-batch losses, the groups of cells, the two counts."""
    V = REVIEWER_73129; out = dict(settings={}); ok = True
    items = next(E.batches([(t, E.split(t, .5, SEED)[0]) for t in E.DEV], SEED, E.KMAX["M"]))
    for s in SETTINGS:
        n = build(s, SEED)
        loss = first_batch_loss(n, items); cnt = counts(n)
        r = dict(code_digest=dg16(n.owner), amplitude=n.amplitude, mean_squared_length=n.mean_squared_length, input_digest=dg16(n.lists),
                 readout_digest=dg16(I49.draws(SEED, D, U)[1]), first_batch_loss=loss, counts=cnt, coverage=coverage(n),
                 input_size_start=input_size(n), kept_mean_magnitude_start=R49.magnitudes(n.m, n.layer, n.iface))
        exp_in = V["one_digest"] if s == "C46-one" else V["input_digest"][s]
        r["agree"] = dict(code_digest=r["code_digest"] == V["code_digest"], amplitude=f"{n.amplitude:.4f}" == f"{V['amplitude']:.4f}",
                          mean_squared_length=f"{n.mean_squared_length:.2f}" == f"{V['mean_squared_length']:.2f}",
                          input_digest=r["input_digest"] == exp_in, readout_digest=r["readout_digest"] == V["readout_digest"],
                          first_batch_loss=all(f"{v:.3f}" == f"{V['first_batch_loss'][s]:.3f}" for v in loss.values()),
                          weights_taking_part=cnt["weights_taking_part"] == WEIGHTS_TAKING_PART[s],
                          products_with_a_turned_on_value=abs(cnt["products_with_a_turned_on_value"] - PRODUCTS_TURNED_ON[s]) <= 100)
        ok = ok and all(r["agree"].values()); out["settings"][s] = r
        print(f"figures {s}: {r['agree']} loss {loss} second count {cnt['products_with_a_turned_on_value']:.0f}", flush=True)
    g = [len(x) for x in digit_groups(SEED)]
    out["digit_groups"] = dict(sizes=g, expected=list(V["groups"]), agree=tuple(g) == V["groups"])
    out["code_digests_other_seeds"] = {}
    for sd, exp in ((93683, "53fb9ffe8aae97fb"), (116089, "f822e27e800c23a6")):
        owner = [-1] * D
        for b, vs in draw_code(sd).items():
            for v in vs:
                owner[v] = b
        out["code_digests_other_seeds"][str(sd)] = dict(digest=dg16(owner), expected=exp, agree=dg16(owner) == exp)
    out["reviewer_figures"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in V.items()}
    out["pass"] = ok and out["digit_groups"]["agree"] and all(v["agree"] for v in out["code_digests_other_seeds"].values())
    return out


CHECKS = {"1": check1, "3": check3, "4": check4, "figures": check_figures}


# ---------------------------------------------------------------- main
def main(argv):
    cmd = "python3 " + " ".join(argv)
    a = argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    if a[0] == "run":
        setting, rule, seed = a[1], a[2], int(a[3])
        assert setting in SETTINGS and rule in RULES and seed == SEED, "E50 runs seed 73129 only"
        d = opt("--dose", None)
        run(setting, rule, seed, Path(opt("--out", str(OUT))), cmd, dose=int(d) if d else None, verify_scorer="--verify-scorer" in a)
    elif a[0] == "check":
        n = a[1]
        qual = Path(opt("--qual", str(QUAL))); qual.mkdir(parents=True, exist_ok=True)
        tag = opt("--tag", "")
        p = qual / f"check{n}_{SEED}{('_' + tag) if tag else ''}.json"
        R_.refuse_existing(p)
        t0 = time.process_time(); w0 = time.time()
        if n == "5":
            ss = tuple(opt("--settings", ",".join(SETTINGS)).split(",")); rules = opt("--rules", "SB")
            res = check5(qual / f"check5_{SEED}_runs", cmd, ss, rules)
        else:
            res = CHECKS[n](qual / f"check{n}_{SEED}_runs", cmd)
        res.update(check=n, seed=SEED, cpu=time.process_time() - t0, wall=time.time() - w0, versions=versions(), provenance=provenance(cmd))
        p.write_text(json.dumps(res, indent=1))
        print(f"e50 check {n}:", "PASS" if res["pass"] else "FAIL", flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
