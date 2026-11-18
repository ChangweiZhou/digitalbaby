"""E55, first part (brief TASK_E55_THE_FIRST_FAMILY_KEPT_WITHOUT_A_LINE_OF_THE_PROGRAM_THAT_LOOKS_AT_THE_ITEM_20261118.md, revision 4): the first
family kept while a new task is learned, with the plain forward pass (no line that reads the item's first byte). Two arms on E51's keyed start:

    blind   the three rows of each cell in reserve (input map, recurrent matrix) and rows 256..511 of lm and l2 are projected away from the
            directions in use at the start and kept so (the whole row after every update); no bias is added
    shut    the update-gate rows of the reserve and rows 256..511 of lm and l2 are projected away from the directions in use at the start, 10
            is added to the input map's bias of the update gates of the reserve, and these rows join E51's rule (projected by the step)

In both arms the old rows are set, after every update, to their start plus the projection of their whole change from the start.
exp51_reserve.py is imported (as R51) and not edited; its SHA-256 is asserted at import.

    python3 exp55_reserve.py start <blind|shut> <S|B> 73129
    python3 exp55_reserve.py run <unrelated|conflict> <blind|shut> <S|B> 73129 [--updates N]
    python3 exp55_reserve.py check <1|2|3> 73129
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp51_reserve as R51          # noqa: E402  (not edited)

R51_SHA256 = "84d0dad7f6e5eeb15c4518a8c591eb5fc043d72b17926913a3b3fde690d1f34a"
assert R51.X16.sha_file(HERE / "exp51_reserve.py") == R51_SHA256, "exp51_reserve.py is not the file the brief names"

T, R, X16, X, C, U, S41, E = R51.T, R51.R, R51.X16, R51.X, R51.C, R51.U, R51.S41, R51.E
OUT55 = X.OUT / "e55"
DOSE55 = 4 * R51.DOSE
ROWS55 = R51.ROWS + (8192, 16384)
KEEP55 = R51.KEEP_ROWS + (16384,)
ARMS55 = ("blind", "shut")

RES_ROWS = R51.RES_ROWS                                  # rows g * 256 + i, g = 0, 1, 2, i = 128 .. 255: the three rows of each cell in reserve
Z_RES = T.arange(R51.U1 + R51.U0, 2 * R51.U1)            # rows 384 to 511: the update gates of the cells in reserve
ADDED = T.arange(R51.H0, R51.H1)                         # rows 256 to 511 of lm and of l2
GATE_BIAS = 10.0
OWN = ("exp55_reserve.py",)


def provenance(cmd):
    p = R51.provenance(cmd)
    p["e55_sources_sha256"] = {s: X16.sha_file(HERE / s) for s in OWN}
    import subprocess
    p["git"]["e55_sources_modified_vs_commit"] = subprocess.run(["git", "status", "--porcelain", "--", *OWN], cwd=HERE, capture_output=True,
                                                                text=True).stdout.strip().splitlines()
    return p


# ---------------------------------------------------------------- 1. the two starts
def blocks(m, dirs, cells):                              # (weight, bias, directions in use, rows)
    return ((m.rnn.weight_ih_l0, m.rnn.bias_ih_l0, dirs["ih"], cells), (m.rnn.weight_hh_l0, m.rnn.bias_hh_l0, dirs["hh"], cells),
            (m.lm.weight, m.lm.bias, dirs["lm"], ADDED), (m.l2.weight, m.l2.bias, dirs["l2"], ADDED))


@T.no_grad()
def make_blind(m, dirs, cells):                          # the whole of each row away from the directions in use
    for W, b, Vm, rows in blocks(m, dirs, cells):
        D = R51.away(T.cat([W[rows], b[rows][:, None]], 1).double(), Vm)
        W[rows] = D[:, :-1].float(); b[rows] = D[:, -1].float()
    zero_blocks(m, cells)


@T.no_grad()
def zero_blocks(m, cells):                               # the projection leaves remainders of 1e-16 here; the sweep sends the error back by the sign of a weight
    m.rnn.weight_hh_l0[cells, :R51.U0] = 0.0; m.rnn.bias_hh_l0[cells] = 0.0
    m.lm.weight[ADDED, :R51.U0] = 0.0; m.lm.bias[ADDED] = 0.0


def load_start55(form, rule, seed):
    m, m0, in_use, dirs, rec = R51.load_start("keyed", rule, seed)   # asserts E51's committed keyed start and its saved directions
    R51.install(None)                                                 # the forward without the added lines
    if form == "blind":
        make_blind(m, dirs, RES_ROWS)
    else:
        assert form == "shut"
        make_blind(m, dirs, Z_RES)
        with T.no_grad():
            m.rnn.bias_ih_l0[Z_RES] += GATE_BIAS
    return m, in_use, dirs, rec


def start_paths(form, rule, seed, out=OUT55):
    base = Path(out) / f"e55_start_{form}_{rule}_{seed}"
    return base.with_suffix(".pt"), base.with_suffix(".json")


def make_start(form, rule, seed, cmd):
    wp, jp = start_paths(form, rule, seed); R.refuse_existing(wp, jp); OUT55.mkdir(parents=True, exist_ok=True)
    m, in_use, dirs, rec = load_start55(form, rule, seed)
    R.save_weights(m, wp)
    meta = dict(experiment="E55", kind="start", form=form, rule=rule, seed=seed, weights=wp.name, sha256=X16.sha_file(wp),
                tensors_sha256=R.tensors_sha(m.state_dict()), keyed_start=rec, gate_bias=GATE_BIAS if form == "shut" else None, provenance=provenance(cmd))
    jp.write_text(json.dumps(meta, indent=1))
    print("start", form, rule, seed, "tensors", meta["tensors_sha256"][:16], flush=True)


def load_checked(form, rule, seed):
    """Build the start again and assert the hash of its tensors against its receipt."""
    m, in_use, dirs, rec = load_start55(form, rule, seed)
    wp, jp = start_paths(form, rule, seed); srec = json.loads(jp.read_text())
    assert R.tensors_sha(m.state_dict()) == srec["tensors_sha256"] and X16.sha_file(wp) == srec["sha256"], "the start differs from its receipt"
    return m, in_use, dirs, rec, srec


# ---------------------------------------------------------------- 2. the rule
class Rule55(R51.Rule):
    def __init__(self, m, dirs, in_use, arm):
        super().__init__(m, dirs, in_use, True)          # E51's rule: the old rows, by the step
        self.arm, self.dirs = arm, dirs
        if arm == "shut":                                # the gate rows of the reserve and the added rows join E51's rows, by the step
            cells = T.cat([R51.OLD_ROWS, Z_RES]); units = T.arange(R51.H1)
            self.mats = [(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0, dirs["ih"], cells), (m.rnn.weight_hh_l0, m.rnn.bias_hh_l0, dirs["hh"], cells),
                         (m.lm.weight, m.lm.bias, dirs["lm"], units), (m.l2.weight, m.l2.bias, dirs["l2"], units)]
        else:
            assert arm == "blind"                         # self.mats stays E51's; the rows kept blind are treated below
        self.start = [(W[rows].detach().double().clone(), b[rows].detach().double().clone()) for W, b, Vm, rows in self.mats]

    @T.no_grad()
    def before(self):
        if self.arm == "blind":                           # the gradient of the rows kept blind, away from the directions in use
            for W, b, Vm, rows in blocks(self.m, self.dirs, RES_ROWS):
                G = R51.away(T.cat([W.grad[rows], b.grad[rows][:, None]], 1).double(), Vm)
                W.grad[rows] = G[:, :-1].float(); b.grad[rows] = G[:, -1].float()
        super().before()

    @T.no_grad()
    def after(self):
        super().after()
        for (W, b, Vm, rows), (W0, b0) in zip(self.mats, self.start):   # the whole change from the start, away from the directions in use
            D = R51.away(T.cat([W[rows].double() - W0, (b[rows].double() - b0)[:, None]], 1), Vm)
            W[rows] = (W0 + D[:, :-1]).float(); b[rows] = (b0 + D[:, -1]).float()
        if self.arm == "blind":
            make_blind(self.m, self.dirs, RES_ROWS)      # the whole rows again, so that no remainder accumulates
        else:
            zero_blocks(self.m, Z_RES)


# ---------------------------------------------------------------- 3. the gates on a set of items
@T.no_grad()
def gates_on(m, items):
    """Teacher forcing, the plain forward. Over the cells in reserve: the largest of 1 - z over cells, positions and items; the number of
    cells whose mean of 1 - z over positions and items exceeds 0.1; the number of cells for which the mean over items of the largest
    1 - z over the positions of the item exceeds 0.1, and the same number for 0.5; the largest magnitude of a state; and the largest
    magnitude of an added readout unit at the answer positions."""
    U0, U1, H0 = R51.U0, R51.U1, R51.H0
    tot, n, maxsum, nitems = T.zeros(U1 - U0, dtype=T.float64), 0, T.zeros(U1 - U0, dtype=T.float64), 0
    largest, state, added = 0.0, 0.0, 0.0
    groups = {}
    for it in items:
        groups.setdefault(len(it[0]) + len(it[1]), []).append(it)
    for g in groups.values():
        for j in range(0, len(g), 256):
            x, y, mask = X.batch_tensor(g[j:j + 256])
            out = R51.fwd(m, x, False, True)[1]
            op = 1 - T.stack(out["z"], 1)[:, :, U0:].double()             # items, positions, cells
            tot += op.sum((0, 1)); n += op.shape[0] * op.shape[1]
            maxsum += op.max(1).values.sum(0); nitems += op.shape[0]
            largest = max(largest, float(op.max()))
            state = max(state, float(T.stack(out["h"], 1)[:, :, U0:].abs().max()))
            added = max(added, float(out["z2"][mask][:, H0:].abs().max()))
    mean_open, mean_max = tot / n, maxsum / nitems
    return dict(largest_opening=largest, cells_open_by_more_than_0p1=int((mean_open > 0.1).sum()), cells_open_at_some_position=int((mean_max > 0.1).sum()),
                cells_open_by_more_than_0p5_at_some_position=int((mean_max > 0.5).sum()), largest_reserve_state=state,
                largest_added_unit_at_answer_positions=added)


# ---------------------------------------------------------------- 4. the run (the text of exp51_reserve.run, with the brief's changes)
def run(task, arm, rule, seed, out, cmd, updates=4 * R51.DOSE):
    assert arm in ("blind", "shut")
    local = R51.C.ARMS[rule]; out = Path(out)
    tag = f"e55_{task}_{arm}_{rule}_{seed}" + ("" if updates == 4 * R51.DOSE else f"_dry{updates}")
    path = out / f"{tag}.json"; R51.R.refuse_existing(path); out.mkdir(parents=True, exist_ok=True); (out / "rows_uncommitted").mkdir(exist_ok=True)
    m, in_use, dirs, start_rec, srec = load_checked(arm, rule, seed)
    bsd = {k: v.clone() for k, v in m.state_dict().items()}
    start_tsha = R51.R.tensors_sha(bsd)
    opt = R51.C.make_opt(m)
    opt.param_groups[0]["weight_decay"] = 0.0
    groups_read_back = R51.R.opt_groups(m, opt)
    ruleobj = Rule55(m, dirs, in_use, arm)

    P, comp = R51.X16.panels(seed)
    held = {n: P[n][12] for n in ("add89", "sub89", "max89", "min89")}
    s1 = R51.U.first_family_items(); pairs = R51.first_family_pairs()
    s2_items = [it for n in held for it in held[n]]
    npan = R51.new_panels(task, seed, P)
    reserve_panel = [it for n in (("add97", "sub97") if task == "conflict" else ("held",)) for it in npan[n]]
    nxt, stream_note = R51.item_source(task, "keyed", seed)
    len40 = R51.E.panel(("add", 89, b"p"), R51.E.split(("add", 89, b"p"), .5, seed)[1], seed, ks=(12, 40), n=300)[40]
    dg = R51.C.Digest(); counts = {task: 0, "first_family": 0}; losses = []
    t0 = time.process_time(); cpu_score = [0.0]; rows = []; gen0 = [None]; logits0 = [None]; gen40_0 = [None]
    rowset = [r for r in ROWS55 if r <= updates]

    def score_row(g):
        cb = time.process_time()
        r = dict(updates=g, data_digest=dg.value(), items_so_far=dict(counts), loss_since_last_row=(sum(losses) / len(losses)) if losses else None)
        s1r, s2r, share, classes, gens = R51.score_first_family(m, s1, pairs, held, gen0[0])
        if g == 0:
            gen0[0] = gens
        r["score1_first_family_all_pairs_k0"] = s1r; r["score2_first_family_heldout_k12"] = s2r
        r["share_of_answers_equal_to_row0"] = share; r["wrong_answers_on_reduction_pairs"] = classes
        if g in R51.GEN_ROWS:
            gp = out / f"{tag}_reduction_bytes_row{g}.json"; gp.write_text(json.dumps(R51.reduction_bytes(gens, s1)))
            r["reduction_bytes"] = gp.name; r["reduction_bytes_sha256"] = R51.X16.sha_file(gp)
        s3 = {n: R51.X16.score(m, it, None) for n, it in npan.items()}
        r["score3_new_task"] = s3; r["score3_never_taught"] = R51.never_taught(task, s3)
        R51.KEY["shut_all"] = True
        s3s = {n: R51.X16.score(m, it, None) for n, it in npan.items()}
        R51.KEY["shut_all"] = False
        r["score3_new_task_reserve_shut"] = s3s; r["score3_never_taught_reserve_shut"] = R51.never_taught(task, s3s)
        r["reserve"] = R51.reserve_figures(m, reserve_panel, s2_items)
        r["gates_on_first_family"] = gates_on(m, s2_items)
        r["gates_on_new_task"] = gates_on(m, reserve_panel)
        lg = R51.answer_logits(m, s2_items)
        if g == 0:
            logits0[0] = lg
        r["first_family_largest_change_of_an_output"] = float((lg - logits0[0]).abs().max())
        r["reserve_map_largest_eigenvalue"] = float(T.linalg.eigvals(0.5 * T.eye(128, dtype=T.float64) + 0.25 * m.rnn.weight_hh_l0[640:768, 128:256].double()).abs().max())
        r["gates_on_first_family_40_letters"] = gates_on(m, len40)
        g40 = R51.generate(m, len40)
        if g == 0:
            gen40_0[0] = g40
        r["first_family_40_letters"] = dict(n=len(len40), differ_from_row0=sum(a != b for a, b in zip(g40, gen40_0[0])),
                                            exact=sum(o == t for o, (_, t) in zip(g40, len40)))
        mv, split, norms = R51.weights_figures(m, bsd)
        r["score4_movement"] = mv; r["tensor_norms"] = norms
        r["score4_movement_old_and_reserve_rows"] = split
        cpu_score[0] += time.process_time() - cb
        r["tensors_sha256"] = R51.R.tensors_sha(m.state_dict())
        if g in KEEP55:
            wp = (out if updates == 4 * R51.DOSE else out / "rows_uncommitted") / f"{tag}_row{g}.pt"
            r["weights"] = str(wp.relative_to(out)); R51.R.save_weights(m, wp); r["weights_sha256"] = R51.X16.sha_file(wp)
        r["cpu"] = time.process_time() - t0
        rows.append(r); losses.clear()
        pc = lambda d: round(100 * d["exact"] / d["n"], 1)
        gf = r["gates_on_first_family"]
        print(tag, "row", g, "reduction", round(s1r["reduction_score"], 1), "max/min", round(s1r["max89"]["all"]["pct"], 1), round(s1r["min89"]["all"]["pct"], 1),
              "held k12", {n: pc(v) for n, v in s2r.items()}, "same as row 0", round(100 * share["all_pairs_k0"], 3),
              "new", r["score3_never_taught"]["pct"], "shut", r["score3_never_taught_reserve_shut"]["pct"],
              "ff state", f"{gf['largest_reserve_state']:.2e}", "ff 1-z", f"{gf['largest_opening']:.2e}", "out change", f"{r['first_family_largest_change_of_an_output']:.2e}",
              "eig", round(r["reserve_map_largest_eigenvalue"], 3), "40 letters differ", r["first_family_40_letters"]["differ_from_row0"],
              "new open at some position", r["gates_on_new_task"]["cells_open_at_some_position"], f"cpu {r['cpu']:.0f}s", flush=True)

    settings = dict(experiment="E55", part="first", task=task, arm=arm, rule=rule, local=local, seed=seed, updates=updates, start=start_rec,
                    start_e55=dict(form=arm, file=srec["weights"], sha256=srec["sha256"], tensors_sha256=srec["tensors_sha256"]),
                    start_form=arm, start_tensors_sha256=start_tsha, keyed_forward=False,
                    rule_applied=True, restore_book=True, stream=stream_note, decay=0.0,
                    rule_rows=("E51's rows (the old rows of the input map and the recurrent matrix, rows 0-255 of lm and l2) by the step, then the whole change "
                               "from the start projected; " + ("the three rows of each cell in reserve and rows 256-511 of lm and l2 kept blind (the whole row "
                               "projected after every update)" if arm == "blind" else "the update-gate rows of the reserve (384-511) and rows 256-511 of lm and l2 "
                               "joined to E51's rows, by the step, then the whole change from the start projected")),
                    gate_bias=GATE_BIAS if arm == "shut" else None,
                    optimizer_groups_read_back=groups_read_back, model=R51.S41.describe(m), batch=R51.X.BATCH, rows=rowset, committed_rows=list(KEEP55),
                    update="exp51_reserve.step (exp26_candidate.step's text; forward exp51_reserve.fwd without a key); the rule exp55_reserve.Rule55",
                    forward="exp51_reserve.fwd with KEY['in_use'] = None (no added line)",
                    provenance=provenance(cmd))
    assert R51.KEY["in_use"] is None

    def write(complete=None):
        path.write_text(json.dumps(dict(experiment="E55", kind="run", tag=tag, settings=settings, rows=rows, complete=complete), indent=1))

    score_row(0); write(); g = 0
    while g < updates:
        batch = []
        for _ in range(R51.X.BATCH):
            kind, it = nxt(); counts[kind] += 1
            if kind == "unrelated":
                assert all(b >= 128 for b in it[0])
            else:
                assert all(b < 128 for b in it[0])
            batch.append(it)
        dg.add(batch)
        losses.append(R51.step(m, opt, batch, local, ruleobj))
        g += 1
        if g in rowset:
            score_row(g); write()
    write(dict(updates=g, items=counts, optimizer_steps=R51.R.opt_steps(opt), optimizer_groups_read_back=R51.R.opt_groups(m, opt), data_digest=dg.value(),
               tensors_sha256=R51.R.tensors_sha(m.state_dict()), cpu_total=time.process_time() - t0, cpu_scoring=cpu_score[0],
               cpu_updates=time.process_time() - t0 - cpu_score[0]))
    print(tag, "complete", f"cpu {time.process_time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- checks before the runs
def pm(D, Vm):
    return float((D @ Vm).abs().max())


def blockD(W, b, rows):
    return T.cat([W[rows], b[rows][:, None]], 1).double()


@T.no_grad()
def keyed_reference(rule, seed, held, samp):
    """The keyed start under the keyed forward pass: outputs at the answer positions and the 32,884 generated answers."""
    m, m0, in_use, dirs, rec = R51.load_start("keyed", rule, seed)
    R51.install(in_use)
    ref = dict(lg={n: R51.answer_logits(m, it) for n, it in (("heldout_k12", held), ("sample_2000_k0", samp))})
    s1 = R51.U.first_family_items()
    ref["gen"] = [g for n in s1 for g in R51.generate(m, s1[n][0])] + R51.generate(m, held)
    R51.install(None)
    assert R51.KEY["in_use"] is None
    return ref


def check1(seed, cmd):
    held, samp, s1, heldp = R51.check_panels(seed); res = {}
    for rule in ("S", "B"):
        ref = keyed_reference(rule, seed, held, samp)
        mk, _, _, dirs, _ = R51.load_start("keyed", rule, seed); sdk = mk.state_dict()
        for form in ARMS55:
            m, in_use, dirs, rec, srec = load_checked(form, rule, seed)
            R51.install(None); assert R51.KEY["in_use"] is None
            sd = m.state_dict(); cells = RES_ROWS if form == "blind" else Z_RES
            # (a) bit-equal outside the four blocks; the zero entries
            outside = {}
            for n, v in sd.items():
                mask = T.ones_like(v, dtype=T.bool)
                if n.startswith("rnn.") :
                    mask[cells] = False
                elif n in ("lm.weight", "lm.bias", "l2.weight", "l2.bias"):
                    mask[ADDED] = False
                outside[n] = bool(T.equal(v[mask], sdk[n][mask]))
            zeros = dict(hh_from_old=bool((m.rnn.weight_hh_l0[cells, :R51.U0] == 0).all()), hh_bias=bool((m.rnn.bias_hh_l0[cells] == 0).all()),
                         lm_from_old=bool((m.lm.weight[ADDED, :R51.U0] == 0).all()), lm_bias=bool((m.lm.bias[ADDED] == 0).all()))
            # (b) each block away from the directions in use
            b_ = {}
            for name, (W, b, Vm, rows) in zip(("ih", "hh", "lm", "l2"), blocks(m, dirs, cells)):
                D = blockD(W, b, rows)
                if form == "shut" and name == "ih":
                    D[:, -1] -= GATE_BIAS
                b_[name] = pm(D, Vm)
            # (c) the gates and states on the two panels; the outputs against the keyed start under the keyed forward
            gh, gs = gates_on(m, held), gates_on(m, samp)
            diff = max(float((R51.answer_logits(m, it) - ref["lg"][n]).abs().max()) for n, it in (("heldout_k12", held), ("sample_2000_k0", samp)))
            first_batch = {}
            for task in R51.TASKS:
                st = R51.new_stream(task, seed); items = [next(st) for _ in range(X.BATCH)]
                first_batch[task] = gates_on(m, items)["largest_reserve_state"]
            # (d) the generated answers against the keyed start's
            gen = [g for n in s1 for g in R51.generate(m, s1[n][0])] + R51.generate(m, held)
            ndiff = sum(a != b for a, b in zip(gen, ref["gen"]))
            r = dict(a_bit_equal_outside_blocks=outside, a_zero_entries=zeros, b_largest_D_at_Vm=b_, c_gates_on_heldout_k12=gh, c_gates_on_sample_2000=gs,
                     c_largest_output_difference_from_keyed_start=diff, c_largest_reserve_state_first_batch_of_new_task=first_batch,
                     d_answers_differing_from_keyed_start=ndiff, d_answers=len(gen))
            ok_a = all(outside.values()) and all(zeros.values()); ok_b = all(v <= 1e-5 for v in b_.values())
            mx = lambda k: max(gh[k], gs[k])
            if form == "shut":
                ok_c = (4.5e-5 <= mx("largest_opening") <= 4.6e-5) and (1e-4 <= mx("largest_reserve_state") <= 1.1e-3) and \
                       (0 < mx("largest_added_unit_at_answer_positions") <= 1e-3) and (0 < diff <= 0.02)
                ok_d = ndiff <= 5
            else:
                ok_c = mx("largest_reserve_state") <= 1e-5 and mx("largest_added_unit_at_answer_positions") <= 1e-5 and diff <= 1e-4 and \
                       all(v > 1e-3 for v in first_batch.values())
                ok_d = ndiff == 0
            r.update(ok_a=ok_a, ok_b=ok_b, ok_c=ok_c, ok_d=ok_d, ok=ok_a and ok_b and ok_c and ok_d)
            res[f"{form} {rule}"] = r
            print("check 1", form, rule, "a", ok_a, "b", b_, "c", f"1-z {mx('largest_opening'):.3e} state {mx('largest_reserve_state'):.3e} "
                  f"added {mx('largest_added_unit_at_answer_positions'):.3e} diff {diff:.3e} first batch {first_batch}", "d", ndiff, "ok", r["ok"], flush=True)
    return dict(ok=all(v["ok"] for v in res.values()), starts=res)


def new_task_bytes(task):
    if task == "unrelated":
        return R51.TK.SYM + R51.TK.FILL + [R51.TK.CODE, R51.TK.ASK]
    return list(b"rtxn")


@T.no_grad()
def map_input(m, rows, byts):
    """What rows of the input map receive on the code of each byte: W code + b, rows by bytes."""
    code = T.tanh(m.l1(m.book.weight[T.tensor(byts)]))
    return (m.rnn.weight_ih_l0[rows] @ code.T + m.rnn.bias_ih_l0[rows][:, None]).double()


def check2(seed, cmd):
    held, samp, s1, heldp = R51.check_panels(seed); res = {}
    for rule in ("S", "B"):
        local = C.ARMS[rule]
        for task in R51.TASKS:
            for arm in ARMS55:
                m, in_use, dirs, rec, srec = load_checked(arm, rule, seed)
                assert R51.KEY["in_use"] is None
                bsd = {k: v.clone() for k, v in m.state_dict().items()}
                opt = C.make_opt(m); opt.param_groups[0]["weight_decay"] = 0.0
                ro = Rule55(m, dirs, in_use, arm)
                cells = RES_ROWS if arm == "blind" else Z_RES
                a0 = {n: R51.answer_logits(m, it) for n, it in (("heldout_k12", held), ("sample_2000_k0", samp))}
                byts = new_task_bytes(task); in0 = map_input(m, cells, byts)
                grad_med = [None]
                if arm == "shut":                                    # the gradient of the gate rows at the first update, after the rule's projection
                    orig = ro.before

                    def wrapped():
                        orig()
                        if grad_med[0] is None:
                            grad_med[0] = dict(median=float(m.rnn.weight_ih_l0.grad[Z_RES].abs().median()), largest=float(m.rnn.weight_ih_l0.grad[Z_RES].abs().max()),
                                               bias_median=float(m.rnn.bias_ih_l0.grad[Z_RES].abs().median()))
                    ro.before = wrapped
                st = R51.new_stream(task, seed)
                for _ in range(8):
                    R51.step(m, opt, [next(st) for _ in range(X.BATCH)], local, ro)
                sd = m.state_dict()
                equal = {"l1.weight": bool(T.equal(sd["l1.weight"], bsd["l1.weight"])), "l1.bias": bool(T.equal(sd["l1.bias"], bsd["l1.bias"])),
                         "l3.weight[:, :256]": bool(T.equal(sd["l3.weight"][:, :R51.H0], bsd["l3.weight"][:, :R51.H0])), "l3.bias": bool(T.equal(sd["l3.bias"], bsd["l3.bias"])),
                         "book.weight rows in use": bool(T.equal(sd["book.weight"][in_use], bsd["book.weight"][in_use]))}
                names = (("ih", "rnn.weight_ih_l0", "rnn.bias_ih_l0"), ("hh", "rnn.weight_hh_l0", "rnn.bias_hh_l0"), ("lm", "lm.weight", "lm.bias"), ("l2", "l2.weight", "l2.bias"))
                old = {}
                for k, wn, bn in names:
                    rows = R51.OLD_ROWS if k in ("ih", "hh") else T.arange(R51.H0)
                    old[k] = pm(blockD(sd[wn], sd[bn], rows) - blockD(bsd[wn], bsd[bn], rows), dirs[k])
                changed = {wn: bool(not T.equal(sd[wn], bsd[wn])) for _, wn, _ in names}
                zeros = dict(hh_from_old=bool((m.rnn.weight_hh_l0[cells, :R51.U0] == 0).all()), hh_bias=bool((m.rnn.bias_hh_l0[cells] == 0).all()),
                             lm_from_old=bool((m.lm.weight[ADDED, :R51.U0] == 0).all()), lm_bias=bool((m.lm.bias[ADDED] == 0).all()))
                gh, gs = gates_on(m, held), gates_on(m, samp)
                mx = lambda k: max(gh[k], gs[k])
                outd = max(float((R51.answer_logits(m, it) - a0[n]).abs().max()) for n, it in (("heldout_k12", held), ("sample_2000_k0", samp)))
                dmap = (map_input(m, cells, byts) - in0).abs()
                r = dict(a_bit_equal=equal, a_old_rows_change_at_Vm=old, a_matrices_changed=changed, a_zero_entries=zeros,
                         gates_on_heldout_k12=gh, gates_on_sample_2000=gs, largest_change_of_an_output=outd,
                         input_map_change_on_new_task_bytes=dict(rows="update-gate rows of the reserve" if arm == "shut" else "all rows of the reserve",
                                                                 bytes=len(byts), mean=float(dmap.mean()), largest=float(dmap.max())))
                ok = all(equal.values()) and all(v <= 1e-5 for v in old.values()) and all(changed.values()) and all(zeros.values())
                if arm == "shut":
                    kept = {}
                    for k, wn, bn in names:
                        rows = Z_RES if k in ("ih", "hh") else ADDED
                        kept[k] = pm(blockD(sd[wn], sd[bn], rows) - blockD(bsd[wn], bsd[bn], rows), dirs[k])
                    r.update(b_shut_rows_change_at_Vm=kept, c_gradient_first_update=grad_med[0])
                    ok_b = all(v <= 1e-5 for v in kept.values()) and (4.4e-5 <= mx("largest_opening") <= 1e-4) and mx("largest_reserve_state") <= 1.1e-3
                    r["c_ask"] = r["input_map_change_on_new_task_bytes"]["largest"] < 0.01
                    ok = ok and ok_b
                else:
                    whole = {}
                    for k, wn, bn in names:
                        rows = RES_ROWS if k in ("ih", "hh") else ADDED
                        whole[k] = pm(blockD(sd[wn], sd[bn], rows), dirs[k])
                    r.update(d_blind_rows_at_Vm=whole)
                    ok = ok and all(v <= 1e-5 for v in whole.values()) and mx("largest_reserve_state") <= 1e-5 and outd <= 1e-4
                    r["d_ask"] = r["input_map_change_on_new_task_bytes"]["largest"] <= 0.01
                r["ok"] = ok
                res[f"{arm} {rule} {task}"] = r
                print("check 2", arm, rule, task, "old", {k: f"{v:.1e}" for k, v in old.items()}, "1-z", f"{mx('largest_opening'):.2e}", "state", f"{mx('largest_reserve_state'):.2e}",
                      "out", f"{outd:.2e}", "map change", f"{r['input_map_change_on_new_task_bytes']['largest']:.3f}", "grad", grad_med[0], "ok", ok, flush=True)
    asks = [k for k, v in res.items() if v.get("c_ask") or v.get("d_ask")]
    return dict(ok=all(v["ok"] for v in res.values()), asks=asks, runs=res)


def check3(seed, cmd):
    d = OUT55 / "checks" / "dry"; res = {}
    for rule in ("S", "B"):
        for task in R51.TASKS:
            f = d / f"e55_{task}_blind_{rule}_{seed}_dry64.json"
            if not f.exists():
                run(task, "blind", rule, seed, d, cmd, updates=64)
            mine = json.loads(f.read_text()); ref = json.loads((R51.OUT51 / f"e51_{task}_keyed_{rule}_{seed}.json").read_text())
            a = [r for r in mine["rows"] if r["updates"] == 64][0]; b = [r for r in ref["rows"] if r["updates"] == 64][0]
            c = mine["complete"]
            res[f"{task} {rule}"] = dict(digest_here=a["data_digest"], digest_e51=b["data_digest"], equal=a["data_digest"] == b["data_digest"],
                                         cpu_seconds_an_update=c["cpu_updates"] / c["updates"], cpu_seconds_a_scored_row=c["cpu_scoring"] / len(mine["rows"]))
    fc = sum(v["cpu_seconds_an_update"] * DOSE55 + v["cpu_seconds_a_scored_row"] * len(ROWS55) for v in res.values())
    fc_runs = 2 * fc                                                 # the arm shut costs about as much as the arm blind
    out = dict(ok=all(v["equal"] for v in res.values()), runs=res, forecast_cpu_hours_eight_runs=fc_runs / 3600, forecast_cpu_hours_blind=fc / 3600,
               wall_hours_at_three_processes=fc_runs / 3600 / 3, ask=fc_runs / 3600 > 8,
               note="the forecast of the arm shut is taken equal to the arm blind's: the same network, stream, rows and scores; the rule differs in which rows are projected")
    print("check 3", json.dumps({k: v for k, v in out.items() if k != "runs"}), flush=True)
    return out


CHECKS = {"1": check1, "2": check2, "3": check3}

if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])
    if a[0] == "start":
        assert a[1] in ARMS55 and a[2] in C.ARMS
        make_start(a[1], a[2], int(a[3]), cmd)
    elif a[0] == "run":
        assert a[1] in R51.TASKS and a[2] in ARMS55 and a[3] in C.ARMS
        n = int(a[a.index("--updates") + 1]) if "--updates" in a else DOSE55
        run(a[1], a[2], a[3], int(a[4]), OUT55, cmd, n)
    elif a[0] == "check":
        seed = int(a[2]); (OUT55 / "checks").mkdir(parents=True, exist_ok=True)
        p = OUT55 / "checks" / f"e55_check_{a[1]}_{seed}.json"; R.refuse_existing(p)
        res = CHECKS[a[1]](seed, cmd)
        p.write_text(json.dumps(dict(experiment="E55", kind=f"check {a[1]}", seed=seed, **res, provenance=provenance(cmd)), indent=1))
        print("check", a[1], "ok" if res["ok"] else "FAILED", flush=True)
    else:
        raise SystemExit(__doc__)
