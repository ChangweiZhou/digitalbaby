"""E62 runner (brief TASK_E62_THE_NUMBER_OF_CONNECTIONS_OF_A_CELL_20261118.md, revision 2): the network of E50's stage 3 (the setting
`C46-one` of exp50_code.py) with N connections a cell, drawn from all cells (`all`) or, for N = 10 only, from the neighbourhood (`near`,
the record's network).

`build` is the text of exp50_code.build for `C46-one` with the one line `layer = L42.Layer(m, seed, N=10)` replaced by
`layer = L42.Layer(m, seed, N=N, wide=(choice == "all"))`. `run` is the text of exp50_code.run with the changes the brief lists (point 2)
and saving and resuming as in exp52_resume.py (point 4). Every other function is exp50_code's, called through R50 and unchanged.

    python3 exp62_connections.py run <N> <all|near> <S|B> 73129 [--dose N] [--out DIR] [--ckpt N] [--resume] [--die-after N]
    python3 exp62_connections.py check <1|2|3|5> [--qual DIR]
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C26        # noqa: E402  first harness import (environment gate), as in exp50_code.py

import torch as T                    # noqa: E402

import arch26                        # noqa: E402
import e42_cells as L42              # noqa: E402
import e49_interface as I49          # noqa: E402
import e50_sparse_code as S50        # noqa: E402
import exp42_cells as R42            # noqa: E402
import exp49_interface as R49        # noqa: E402
import exp50_code as R50             # noqa: E402

R_, X16, E, X = C26.R, C26.X16, C26.E, C26.X
T.set_num_threads(1)

U, D = S50.U, S50.D
SEED = 73129
ROW = R50.ROW
RULES = R50.RULES
DOSE = 49152
FIRST_DOSE = R50.FIRST_DOSE          # 24,576: the weights of this update are saved, as in E50's runs of 49,152 updates
CKPT = 4096
OUT = X.OUT / "e62"
QUAL = X.OUT / "e62_qual"
SOURCES = ("exp62_connections.py",) + R50.SOURCES
GRID = (2, 4, 8, 16, 32, 64, 128, 255)
SETTINGS = [(n, "all") for n in GRID[:3]] + [(10, "all")] + [(n, "all") for n in GRID[3:]] + [(10, "near")]
SETTING = "C46-one"
STEPS = (1, 2, 3, 4, 6, 8)
dg16 = R50.dg16


def weights_taking_part(N):
    return 560128 + 768 * (N + 1)


# ---------------------------------------------------------------- the network (point 1)
def build(N, choice, seed):
    """exp50_code.build's text for the setting C46-one, with N connections a cell drawn from all cells or from the neighbourhood."""
    assert choice in ("all", "near") and (choice == "all" or N == 10)
    setting = SETTING
    k_in, k_out = R50.SETTINGS[setting]
    n = R50.Net()
    m = arch26.new_net(seed)
    with T.no_grad():
        z1 = T.tanh(m.l1(m.book(T.tensor(R50.USED))))
        n.mean_squared_length = float(z1.pow(2).sum(1).mean())
    n.amplitude = S50.amplitude_for(m, seed)
    n.table, n.owner = S50.code_table(seed, n.amplitude)
    n.own = R50.draw_code(seed)
    assert all(n.owner[v] == b for b, vs in n.own.items() for v in vs) and sum(o >= 0 for o in n.owner) == R50.ON * len(R50.USED)
    layer = L42.Layer(m, seed, N=N, wide=(choice == "all"))
    if setting == "C46-one":
        iface = I49.Interface(m, seed, D, U)         # both masks full: nothing is scaled
        lists = R50.draw_one(seed)
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


# ---------------------------------------------------------------- what the receipt records once, from the mask (point 3)
def connections(layer):
    return [[j for j in range(U) if j != i and bool(layer.mask[i, j])] for i in range(U)]


def graph(layer):
    """A step goes from a cell to the cells that read it (layer.mask[i, j]: cell i reads cell j); every cell reads itself."""
    S = layer.mask.T.clone()                         # S[j, i]: i reads j, so j reaches i in one step
    eye = T.eye(U, dtype=T.bool)
    reach = eye.clone(); within = {}; every = None
    for k in range(1, U + 1):
        new = (reach.float() @ S.float()).bool() | reach
        grew = not bool((new == reach).all())
        reach = new
        cnt = reach.sum(1)
        if k in STEPS:
            within[str(k)] = dict(least=int(cnt.min()), mean=float(cnt.float().mean()), most=int(cnt.max()))
        if every is None and bool(reach.all()):
            every = k
        if k >= max(STEPS) and (every is not None or not grew):
            break
    readers = (layer.mask & ~eye).sum(0)             # how many other cells read cell j
    scc = reach & reach.T
    parts = {tuple(r.nonzero().flatten().tolist()) for r in scc}
    out = sum(1 for i, a in enumerate(connections(layer)) for j in a if j not in set(L42.neighbourhood(i, False)))
    return dict(read_by_no_other_cell=int((readers == 0).sum()), readers_least=int(readers.min()), readers_most=int(readers.max()),
                reached_within=within, every_cell_from_every_cell_within=every if every is not None else "none",
                strongly_connected_parts=len(parts), largest_part=max(len(p) for p in parts), connections_outside_neighbourhood=out)


# ---------------------------------------------------------------- one run (point 2)
def receipt_path(out, N, choice, rule, seed, dose):
    return Path(out) / f"e62_N{N}_{choice}_{rule}{'' if dose == DOSE else f'_dry{dose}'}_{seed}.json"


def provenance(cmd):
    p = C26.provenance(cmd)
    p["sources_sha256"] = {s: X16.sha_file(HERE / s) for s in SOURCES}
    p["runner_sha256"] = X16.sha_file(HERE / "exp62_connections.py")
    p["spec_sha256"] = X16.sha_file(R50.SPEC) if R50.SPEC.exists() else None
    p["copies_from_e49_worker_a36de661"] = {f: dict(sha256=X16.sha_file(HERE / f), expected=s, equal=X16.sha_file(HERE / f) == s) for f, s in R50.COPIES.items()}
    assert all(v["equal"] for v in p["copies_from_e49_worker_a36de661"].values()), "a file copied from E49's worker is not byte-identical"
    return p


def save_ckpt(path, m, opt, rec, u, cpu_up, cpu_sc, wall):
    tmp = Path(str(path) + ".tmp")
    T.save(dict(update=u, model=m.state_dict(), opt=opt.state_dict(), rec=rec, cpu_up=cpu_up, cpu_sc=cpu_sc, wall=wall), tmp)
    os.replace(tmp, path)


def run(N, choice, rule, seed, out, cmd, dose=None, ckpt_every=CKPT, resume=False, die_after=None):
    setting = SETTING
    k_in, k_out = R50.SETTINGS[setting]
    local = RULES[rule]
    dose = dose or DOSE
    assert dose % ROW == 0 and ckpt_every % ROW == 0
    rp = receipt_path(out, N, choice, rule, seed, dose); cp = rp.with_suffix(".ckpt")
    paths = dict(receipt=rp, start=rp.with_name(rp.stem + "_start.pt"), weights=rp.with_suffix(".pt"),
                 answers=rp.with_name(rp.stem + "_answers.json"))
    mid = dose > FIRST_DOSE
    if mid:
        paths.update(weights_first_dose=rp.with_name(rp.stem + f"_at{FIRST_DOSE}.pt"), answers_first_dose=rp.with_name(rp.stem + f"_at{FIRST_DOSE}_answers.json"))
    resuming = resume and cp.exists()
    if resuming:
        R_.refuse_existing(paths["weights"], paths["answers"], str(paths["weights"]) + ".tmp")
    else:
        R_.refuse_existing(*paths.values(), *(str(p) + ".tmp" for p in paths.values()), cp)
    Path(out).mkdir(parents=True, exist_ok=True)
    n = build(N, choice, seed); m, layer, iface = n.m, n.layer, n.iface
    cov = R50.coverage(n); rd = R49.readers(iface)
    print(f"e62 N{N} {choice} {rule} {seed}: value of a turned-on entry {n.amplitude:.4f}; cells without input {cov['cells_without_input']}; "
          f"a needed byte reaches at least {cov['cells_a_needed_byte_reaches']['min']} cells", flush=True)
    if cov["cells_without_input"] or cov["cells_a_needed_byte_reaches"]["min"] == 0 or rd["cell_min"] == 0:
        raise SystemExit(f"e62 N{N} {choice} {seed}: a cell without input, a needed byte that reaches no cell, or a cell that is not read; stopped")
    R50.install(n)
    opt = R42.make_opt(m, layer)
    cnt = R50.counts(n)
    assert cnt["weights_taking_part"] == weights_taking_part(N), (N, cnt)
    ma = cnt["weights_taking_part"]
    groups = R50.digit_groups(seed)
    A, Rs, TA, LP = R42.panels(seed)
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    bt = E.batches(tp, seed, E.KMAX["M"]); dg = C26.Digest()
    start = 0; cpu_up = cpu_sc = 0.0; wall0 = 0.0; first = None
    if resuming:
        ck = T.load(cp, weights_only=False)
        rec = ck["rec"]; start = ck["update"]; cpu_up, cpu_sc, wall0 = ck["cpu_up"], ck["cpu_sc"], ck["wall"]
        assert R_.tensors_sha(m.state_dict()) == rec["start_tensors_sha256"], "the rebuilt network is not the saved run's start"
        assert X16.sha_file(paths["start"]) == rec["start_sha256"], "the starting file on disk is not the saved run's"
        c = time.process_time(); w = time.time()
        for _ in range(start):
            items = next(bt)
            if first is None:
                first = items
            dg.add(items)
        assert dg.value() == rec["curve"][-1]["data_digest"], "the redrawn batch digest differs from the saved receipt's last row"
        assert rec["curve"][-1]["step"] == start
        m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        rec["resumes"].append(dict(update=start, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), replay_cpu=time.process_time() - c,
                                   replay_wall=time.time() - w, command=cmd))
        print(f"e62 N{N} {choice} {rule} resumed at {start}", flush=True)
    else:
        R49.save_pt(R50.weights_blob(n), paths["start"])
        rec = dict(experiment="E62", kind="birth", setting=setting, config=setting, choice=choice, rule=rule, local=local, seed=seed, dose=dose, row_every=ROW,
                   k_in=k_in, k_out=k_out, cells=U, N=layer.N, batch=X.BATCH, kmax=E.KMAX["M"], counts=cnt, multiply_adds_per_position=ma,
                   multiply_adds_per_position_second_count=cnt["products_with_a_turned_on_value"],
                   params_module=sum(q.numel() for q in m.parameters()), readers=rd, coverage=cov,
                   code=dict(digest=dg16(n.owner), amplitude=n.amplitude, mean_squared_length_of_present_code=n.mean_squared_length,
                             used=len(R50.USED), values_owned=sum(o >= 0 for o in n.owner), table_sha256=n.table_sha,
                             names_in_weight_files=dict(table="code_table", owners="code_owner", amplitude="code_amplitude")),
                   draw_digests=dict(input=dg16(n.lists), readout=dg16(I49.draws(seed, D, k_out)[1]), connections=dg16(connections(layer))),
                   graph=graph(layer),
                   start_scale=dict(input=(D / k_in) ** 0.5 if k_in < D else 1.0, lm=1.0, recurrent=1.0 if layer.full else (U / (N + 1)) ** 0.5),
                   digit_groups=dict(sizes=[len(g) for g in groups], note="cells with input for 0 to 2, 3 or 4, 5 or more of the ten digits under E49's 16 values a cell"),
                   optimizer=R42.opt_record(m, layer, opt),
                   update="exp26_candidate.step's text with e50_sparse_code.forward(m, layer, iface, table, x, local) in place of arch26.forward",
                   data="exp11_cell.batches([(t, split(t, .5, seed)[0]) for t in DEV], seed, 12)",
                   forward="e50_sparse_code.forward (the specification's own forward)",
                   start_file=paths["start"].name, start_sha256=X16.sha_file(paths["start"]), start_tensors_sha256=R_.tensors_sha(m.state_dict()),
                   mask_sha256=dict(recurrent=R42.tsha(layer.mask), input=R42.tsha(iface.mask_in), readout=R42.tsha(iface.mask_out)),
                   checkpoint_every=ckpt_every, resumes=[],
                   versions=R50.versions(), provenance=provenance(cmd), complete=False, curve=[])
    curve = rec["curve"]; w0 = time.time() - wall0; losses = []

    def score_row(u):
        nonlocal cpu_sc
        c = time.process_time()
        sc = R50.score_all(m, A, Rs, TA, LP)
        cpu_sc += time.process_time() - c
        row = dict(step=u, **sc)
        lim = R42.check_limit(m, layer)
        assert all(lim.values()), f"E42's check 4 failed at row {u}: {lim}"
        mk = R49.check_masks(m, layer, iface, opt)
        assert all(all(v.values()) for v in mk.values()), f"E49's check 3 failed at row {u}: {mk}"
        fx = R50.fixed(n)
        assert all(fx.values()), f"check 2 failed at row {u}: {fx}"
        row.update(limit=lim, masks=mk, fixed=fx, kept_mean_magnitude=R49.magnitudes(m, layer, iface), by_what_a_cell_knows=R50.knows_figure(n, groups),
                   input_size=R50.input_size(n), cpu_updates=cpu_up, cpu_scoring=cpu_sc, multiply_adds=ma * X.BATCH * u, data_digest=dg.value(),
                   loss=(sum(losses) / len(losses)) if losses else None, finite_losses=all(v == v and abs(v) != float("inf") for v in losses),
                   wall=time.time() - w0)
        curve.append(row)
        o = row["ops"]
        print(f"e62 N{N} {choice} {rule} {seed} {u}: A {row['A']:.2f} R {row['R']:.2f} (bytes {row['R_first_byte']:.1f} {row['R_second_byte']:.1f}) "
              f"taught {row['taught']:.2f} len12 {row['len12']:.1f} len40 {row['len40']:.1f} max {o['max89']:.0f} min {o['min89']:.0f} "
              f"cpu {cpu_up:.0f}+{cpu_sc:.0f}s", flush=True)

    if start == 0:
        score_row(0)
    for u in range(start + 1, dose + 1):
        items = next(bt)
        if first is None:
            first = items
            rec["first_batch_loss_start"] = R50.first_batch_loss(n, first)
            rec["cosine_first_batch_start"] = R50.cosines(n, first)
        dg.add(items)
        c = time.process_time(); losses.append(R50.step(n, opt, items, local)); cpu_up += time.process_time() - c
        if u % ROW == 0:
            score_row(u); losses = []
            if mid and u == FIRST_DOSE:
                R49.save_pt(R50.weights_blob(n), paths["weights_first_dose"])
                paths["answers_first_dose"].write_text(json.dumps({str(k): E.respond(m, [p for p, _ in it]) for k, it in LP.items()}))
                rec["first_dose"] = dict(update=FIRST_DOSE, weights=paths["weights_first_dose"].name, weights_sha256=X16.sha_file(paths["weights_first_dose"]),
                                         tensors_sha256=R_.tensors_sha(m.state_dict()), answers_file=paths["answers_first_dose"].name,
                                         cosine_first_batch=R50.cosines(n, first))
            ck_now = ckpt_every and u % ckpt_every == 0 and u < dose
            if u % 1024 == 0 or u == dose or ck_now:
                rp.write_text(json.dumps(rec, indent=1))
            if ck_now:
                save_ckpt(cp, m, opt, rec, u, cpu_up, cpu_sc, time.time() - w0)
        if die_after is not None and u == die_after:
            print(f"e62 dying after update {u} (check 3)", flush=True)
            os._exit(9)
    rec["cosine_first_batch_end"] = R50.cosines(n, first)
    sd = m.state_dict()
    R49.save_pt(R50.weights_blob(n), paths["weights"])
    paths["answers"].write_text(json.dumps({str(k): E.respond(m, [p for p, _ in it]) for k, it in LP.items()}))
    fx = R50.fixed(n)
    rec.update(complete=True, weights=paths["weights"].name, weights_sha256=X16.sha_file(paths["weights"]),
               tensors_sha256=R_.tensors_sha(sd), answers_file=paths["answers"].name,
               byte_table_and_l1_bit_equal_to_start=fx["byte_table_bit_equal"] and fx["l1_bit_equal"],
               code_table_sha256_equal_at_every_row=all(r["fixed"]["code_table_sha256_equal"] for r in curve) and fx["code_table_sha256_equal"],
               optimizer_steps=R_.opt_steps(opt), optimizer_read_back=R42.opt_record(m, layer, opt),
               optimizer_state_for=sorted(k for k, q in m.named_parameters() if opt.state.get(q)), data_digest=dg.value(),
               cpu_updates=cpu_up, cpu_scoring=cpu_sc, wall=time.time() - w0, processes_beside=os.environ.get("E62_BESIDE", "not recorded"))
    rp.write_text(json.dumps(rec, indent=1))
    if cp.exists():
        cp.unlink()
    print("e62 complete", rp.name, rec["tensors_sha256"][:16], "weights sha256", rec["weights_sha256"][:16], flush=True)
    return rec


# ---------------------------------------------------------------- checks before the runs
D_REF = "origin/codex/minifly-consolidated-20261023"
ROWKEYS = ("loss", "A_counts", "R_counts", "taught_counts", "length_counts", "R_first_byte_counts", "R_second_byte_counts", "data_digest")


def git_json(ref, path):
    out = subprocess.run(["git", "show", f"{ref}:MOD97_LOCAL_RULE_20261106/{path}"], cwd=HERE, capture_output=True, check=True).stdout
    return json.loads(out)


def check1(qual, cmd):
    """With N = 10 from the neighbourhood the run is the record: 512 updates under each rule; start_tensors_sha256 and rows 0, 256, 512."""
    res = {}
    for rule in "SB":
        r = run(10, "near", rule, SEED, Path(qual) / rule, cmd, dose=512, ckpt_every=0)
        ref = git_json(D_REF, f"exp45/e50/e50_C46-one_{rule}_73129.json")
        by = {x["step"]: x for x in ref["curve"]}
        rows = {str(x["step"]): {k: x[k] == by[x["step"]][k] for k in ROWKEYS} for x in r["curve"]}
        res[rule] = dict(record=f"{D_REF}:MOD97_LOCAL_RULE_20261106/exp45/e50/e50_C46-one_{rule}_73129.json",
                         start_tensors_sha256=dict(run=r["start_tensors_sha256"], record=ref["start_tensors_sha256"]),
                         start_equal=r["start_tensors_sha256"] == ref["start_tensors_sha256"], rows=rows,
                         rows_equal=all(all(v.values()) for v in rows.values()) and sorted(rows) == ["0", "256", "512"])
        res[rule]["pass"] = res[rule]["start_equal"] and res[rule]["rows_equal"]
    res["pass"] = all(res[r]["pass"] for r in "SB")
    return res


DIGESTS = {(2, "all"): "9622342ea3d0e56f", (4, "all"): "7501b501fcd35c94", (8, "all"): "10d7ba852a8bbb61", (10, "all"): "4a9feeeb77ff57d2",
           (16, "all"): "fbe2bf8967eefa58", (32, "all"): "3037c7cdde9bfefb", (64, "all"): "a6f88a4f5ef85709", (128, "all"): "2ad32f75b1f251ed",
           (255, "all"): "3b1a3a5516ed1271", (10, "near"): "fde78220c57da3d4"}
OUTSIDE = {(2, "all"): 431, (4, "all"): 868, (8, "all"): 1669, (10, "all"): 2049, (16, "all"): 3307, (32, "all"): 6617, (64, "all"): 13289,
           (128, "all"): 26620, (255, "all"): 52992, (10, "near"): 0}
F = lambda a, b, c: (a, b, c)
# section 2 of CHECK_OF_TASK_E62_BRIEF_20261118.md: read by no other cell; readers least to most; reached within 1, 2, 3, 4, 6, 8 steps
# (least, mean to one decimal, most); every cell from every cell; strongly connected parts (largest)
TABLE = {
    (2, "all"): (40, (0, 6), [F(1, 3.0, 7), F(1, 6.9, 20), F(1, 14.5, 50), F(1, 28.6, 98), F(1, 88.6, 224), F(1, 165.0, 256)], "none", (49, 207)),
    (4, "all"): (4, (0, 11), [F(1, 5.0, 12), F(1, 20.4, 52), F(1, 71.8, 172), F(1, 175.9, 255), F(1, 251.4, 256), F(1, 252.0, 256)], "none", (5, 252)),
    (8, "all"): (0, (2, 15), [F(3, 9.0, 16), F(14, 64.9, 123), F(78, 223.7, 256), F(241, 255.9, 256)] + [F(256, 256.0, 256)] * 2, 5, (1, 256)),
    (10, "all"): (0, (3, 19), [F(4, 11.0, 20), F(35, 91.8, 148), F(203, 249.7, 256)] + [F(256, 256.0, 256)] * 3, 4, (1, 256)),
    (16, "all"): (0, (5, 27), [F(6, 17.0, 28), F(75, 171.0, 220), F(254, 256.0, 256)] + [F(256, 256.0, 256)] * 3, 4, (1, 256)),
    (32, "all"): (0, (16, 54), [F(17, 33.0, 55), F(232, 252.9, 256)] + [F(256, 256.0, 256)] * 4, 3, (1, 256)),
    (64, "all"): (0, (48, 88), [F(49, 65.0, 89)] + [F(256, 256.0, 256)] * 5, 2, (1, 256)),
    (128, "all"): (0, (107, 149), [F(108, 129.0, 150)] + [F(256, 256.0, 256)] * 5, 2, (1, 256)),
    (255, "all"): (0, (255, 255), [F(256, 256.0, 256)] * 6, 1, (1, 256)),
    (10, "near"): (0, (3, 17), [F(4, 11.0, 18), F(29, 73.9, 103), F(137, 213.3, 242), F(247, 255.9, 256)] + [F(256, 256.0, 256)] * 2, 5, (1, 256)),
}


def check2(qual, cmd):
    ref = R50.build(SETTING, SEED)
    res = {}
    for N, choice in SETTINGS:
        n = build(N, choice, SEED); m, layer = n.m, n.layer
        A = connections(layer)
        g = graph(layer)
        t = TABLE[(N, choice)]
        within = [(g["reached_within"][str(k)]["least"], round(g["reached_within"][str(k)]["mean"], 1), g["reached_within"][str(k)]["most"]) for k in STEPS]
        scale = 1.0 if layer.full else (256 / (N + 1)) ** 0.5
        w_exp = arch26.new_net(SEED).rnn.weight_hh_l0.detach() * (layer.mask3().float() * scale) if not layer.full else arch26.new_net(SEED).rnn.weight_hh_l0.detach()
        W = m.rnn.weight_hh_l0.detach()
        P, Pr = dict(m.named_parameters()), dict(ref.m.named_parameters())
        same = {k: bool(T.equal(P[k].detach(), Pr[k].detach())) for k in ("rnn.weight_ih_l0", "rnn.bias_ih_l0", "rnn.bias_hh_l0", "lm.weight", "lm.bias",
                                                                           "l2.weight", "l2.bias", "l3.weight", "l3.bias")}
        same["code_table"] = bool(T.equal(n.table, ref.table))
        R50.install(n)
        x = [(t_, E.split(t_, .5, SEED)[0]) for t_ in E.DEV]
        first = next(E.batches(x, SEED, E.KMAX["M"]))
        fbl = R50.first_batch_loss(n, first)
        cs = R50.cosines(n, first)
        r = dict(N=N, choice=choice, exactly_N=all(len(a) == N for a in A) and bool(layer.mask.diagonal().all()),
                 digest=dg16(A), digest_expected=DIGESTS[(N, choice)], outside=g["connections_outside_neighbourhood"], outside_expected=OUTSIDE[(N, choice)],
                 outside_mask_zero=bool((W[~layer.mask3()] == 0).all()), kept_weights_bit_equal=bool(T.equal(W, w_exp)), other_maps_bit_equal=same,
                 first_batch_loss=fbl, first_batch_loss_finite=all(v == v and abs(v) != float("inf") for v in fbl.values()),
                 gradients_zero_outside_masks=cs["gradients_zero_outside_masks"], graph=g,
                 graph_equals_table=(g["read_by_no_other_cell"] == t[0] and (g["readers_least"], g["readers_most"]) == t[1] and within == [tuple(v) for v in t[2]]
                                     and g["every_cell_from_every_cell_within"] == t[3] and (g["strongly_connected_parts"], g["largest_part"]) == t[4]))
        r["pass"] = (r["exactly_N"] and r["digest"] == r["digest_expected"] and r["outside"] == r["outside_expected"] and r["outside_mask_zero"]
                     and r["kept_weights_bit_equal"] and all(same.values()) and r["first_batch_loss_finite"] and r["gradients_zero_outside_masks"]
                     and r["graph_equals_table"])
        res[f"{N} {choice}"] = r
        print(f"e62 check 2: N{N} {choice}: {'PASS' if r['pass'] else 'FAIL'}", flush=True)
    res["pass"] = all(v["pass"] for v in res.values())
    return res


def child_env():
    return {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}


def check3(qual, cmd):
    """32 all under the sweep, 2,048 updates: (i) --ckpt 0; (ii) --ckpt 1024 --die-after 1700 (exit 9), then --resume."""
    base = [sys.executable, str(HERE / "exp62_connections.py"), "run", "32", "all", "S", str(SEED), "--dose", "2048"]
    d1, d2 = Path(qual) / "i", Path(qual) / "ii"
    k1 = subprocess.run(base + ["--out", str(d1), "--ckpt", "0", "--resume"], cwd=HERE, env=child_env(), capture_output=True, text=True)
    k2 = subprocess.run(base + ["--out", str(d2), "--ckpt", "1024", "--resume", "--die-after", "1700"], cwd=HERE, env=child_env(), capture_output=True, text=True)
    k3 = subprocess.run(base + ["--out", str(d2), "--ckpt", "1024", "--resume"], cwd=HERE, env=child_env(), capture_output=True, text=True)
    name = receipt_path(".", 32, "all", "S", SEED, 2048).name
    a, b = json.loads((d1 / name).read_text()), json.loads((d2 / name).read_text())
    strip = lambda c: [{k: v for k, v in r.items() if not (k.startswith("cpu") or k.startswith("wall"))} for r in c]
    keys = ("first_batch_loss_start", "cosine_first_batch_start", "cosine_first_batch_end", "data_digest", "optimizer_steps")
    res = dict(exit_codes=dict(i=k1.returncode, ii_killed=k2.returncode, ii_resumed=k3.returncode),
               tensors_sha256=dict(i=a.get("tensors_sha256"), ii=b.get("tensors_sha256")), tensors_equal=a.get("tensors_sha256") == b.get("tensors_sha256"),
               rows=len(a["curve"]), rows_equal=strip(a["curve"]) == strip(b["curve"]), entries_equal={k: a.get(k) == b.get(k) for k in keys},
               resumes=b["resumes"], child_environment="os.environ without E11_, E16_ and EXP45_ variables; cwd " + str(HERE))
    res["pass"] = (k1.returncode == 0 and k2.returncode == 9 and k3.returncode == 0 and res["tensors_equal"] and a.get("complete") and b.get("complete")
                   and res["rows_equal"] and all(res["entries_equal"].values()) and [r["update"] for r in b["resumes"]] == [1024])
    if not res["pass"]:
        res["stderr_tail"] = [k.stderr[-1500:] for k in (k1, k2, k3)]
    return res


def check5(qual, cmd):
    """CPU-seconds an update (64 updates after 8 not counted) and a scored row (mean of two) for N = 2, 32, 255 under each rule; the forecast."""
    meas = {}
    for N in (2, 32, 255):
        for rule in "SB":
            local = RULES[rule]
            n = build(N, "all", SEED); m, layer, iface = n.m, n.layer, n.iface
            R50.install(n); opt = R42.make_opt(m, layer)
            groups = R50.digit_groups(SEED)
            A, Rs, TA, LP = R42.panels(SEED)
            bt = E.batches([(t, E.split(t, .5, SEED)[0]) for t in E.DEV], SEED, E.KMAX["M"])
            for _ in range(8):
                R50.step(n, opt, next(bt), local)
            c = time.process_time()
            for _ in range(64):
                R50.step(n, opt, next(bt), local)
            per_update = (time.process_time() - c) / 64
            rows = []; ok = True
            for _ in range(2):
                c = time.process_time()
                R50.score_all(m, A, Rs, TA, LP)
                lim = R42.check_limit(m, layer); mk = R49.check_masks(m, layer, iface, opt); fx = R50.fixed(n)
                R49.magnitudes(m, layer, iface); R50.knows_figure(n, groups); R50.input_size(n)
                rows.append(time.process_time() - c)
                ok = ok and all(lim.values()) and all(all(v.values()) for v in mk.values()) and all(fx.values())
            meas[f"{N} {rule}"] = dict(cpu_s_per_update=per_update, cpu_s_per_row=sum(rows) / 2, rows=rows, assertions_hold=ok)
            print(f"e62 check 5: N{N} {rule}: {per_update:.4f} s an update, {sum(rows) / 2:.2f} s a row", flush=True)
    runs = {}
    for N, choice in [s for s in SETTINGS if s != (10, "near")]:
        for rule in "SB":
            if N in (2, 32, 255):
                u, r = meas[f"{N} {rule}"]["cpu_s_per_update"], meas[f"{N} {rule}"]["cpu_s_per_row"]
            else:
                u = max(meas[f"2 {rule}"]["cpu_s_per_update"], meas[f"32 {rule}"]["cpu_s_per_update"])
                r = max(meas[f"2 {rule}"]["cpu_s_per_row"], meas[f"32 {rule}"]["cpu_s_per_row"])
            runs[f"{N} {choice} {rule}"] = (49152 * u + 193 * r) / 3600
    fc = sum(runs.values())
    finite = all(v == v and abs(v) != float("inf") for x in meas.values() for v in (x["cpu_s_per_update"], x["cpu_s_per_row"]))
    return dict(measured=meas, forecast_cpu_h_per_run=runs, forecast_cpu_h=fc, limit_cpu_h=44, ask_before_runs=fc > 44,
                **{"pass": finite and all(x["assertions_hold"] for x in meas.values())})


CHECKS = {"1": check1, "2": check2, "3": check3, "5": check5}


def main(argv):
    cmd = "python3 " + " ".join(argv)
    a = argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    if a and a[0] == "run":
        N, choice, rule, seed = int(a[1]), a[2], a[3], int(a[4])
        assert N in GRID + (10,) and choice in ("all", "near") and (choice == "all" or N == 10) and rule in RULES and seed == SEED
        d = opt("--dose", None); da = opt("--die-after", None)
        run(N, choice, rule, seed, Path(opt("--out", str(OUT))), cmd, dose=int(d) if d else None, ckpt_every=int(opt("--ckpt", str(CKPT))),
            resume="--resume" in a, die_after=int(da) if da else None)
    elif a and a[0] == "check":
        k = a[1]
        qual = Path(opt("--qual", str(QUAL))); qual.mkdir(parents=True, exist_ok=True)
        p = qual / f"check{k}_{SEED}.json"
        R_.refuse_existing(p)
        t0 = time.process_time(); w0 = time.time()
        res = CHECKS[k](qual / f"check{k}_{SEED}_runs", cmd)
        res.update(check=k, seed=SEED, cpu=time.process_time() - t0, wall=time.time() - w0, versions=R50.versions(), provenance=provenance(cmd))
        p.write_text(json.dumps(res, indent=1))
        print(f"e62 check {k}:", "PASS" if res["pass"] else "FAIL", flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
