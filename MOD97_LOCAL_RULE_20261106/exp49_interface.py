"""E49 runner (brief OPUS_TASK_E49_A_LIMITED_INTERFACE_20261116.md, revision 2): E42's F10 (256 cells, 10 fixed local connections) with a
limited interface. Each cell reads K_IN of the 512 values of the byte's code; each of the 512 readout units reads K_OUT of the 256 cells.

The network is the specification file e49_interface.py (a copy of PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e49_interface_spec.py that
differs in its import line only, which points at e42_cells.py, the runner's copy of E42's specification), imported and never rewritten:
its draws, its Interface (the two masks and the start) and its forward. Everything else is E42's runner exp42_cells.py, imported unchanged
(sha256 340f9cc5..., commit 4c9e52d1): the data, the optimizer (C26.make_opt with the layer's added parameters, of which F10 has none),
the dose, the scoring every 256 updates (panels, score_all), the digests and check 4 of the recurrent mask. This file adds the birth loop
around E49's forward, check 3 of the two masks at every scored row, the cosines and mean magnitudes of the masked maps, and the receipts.

    python3 exp49_interface.py run <setting> <S|B> <seed> [--dose N] [--out DIR]
    python3 exp49_interface.py check <1|2|4|5|6> [--qual DIR]
    settings: F10 (512, 256: E42's, both masks full) I64 (64, 32) I16 (16, 8) in16 (16, 256) out8 (512, 8)
"""
import json
import os
import sys
import time
from pathlib import Path
from random import Random  # noqa: F401  (E42's runner imports it; kept for symmetry)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C26        # noqa: E402  first harness import (environment gate)

import torch as T                    # noqa: E402

import arch26                        # noqa: E402
import e42_cells as L42              # noqa: E402  E42's specification (the runner's copy)
import e49_interface as I49          # noqa: E402  E49's specification (the runner's copy), imported, not rewritten
import exp42_cells as R42            # noqa: E402  E42's runner, imported unchanged

R_, X16, E, X = C26.R, C26.X16, C26.E, C26.X
T.set_num_threads(1)

U, D = I49.U, I49.D
SEED = 73129
DOSE = R42.DOSE                      # 24,576
ROW = R42.ROW                        # 256
RULES = R42.RULES                    # S: the sweep (local=True); B: backpropagation
OUT = X.OUT / "e49"
QUAL = X.OUT / "e49_qual"
SPEC = HERE.parent / "PROJECT_DOCUMENTS" / "COORD_CHECKS_20261116" / "e49_interface_spec.py"
SPEC_OUT = SPEC.parent / "OUTPUT_E49_SPEC_MAC.txt"
SOURCES = ("exp49_interface.py", "e49_interface.py", "exp42_cells.py", "e42_cells.py") + C26.SOURCES
SETTINGS = {"F10": (512, 256), "I64": (64, 32), "I16": (16, 8), "in16": (16, 256), "out8": (512, 8)}
MA_EXPECTED = {"F10": 930560, "I64": 471808, "I16": 422656, "in16": 549632, "out8": 803584}
WEIGHTS_EXPECTED = {"I64": 477201, "I16": 428049, "in16": 555025, "out8": 808977}
DIGESTS_73129 = {("in", 64): "eb1dd9417ca1a282", ("in", 16): "0e09b9ff0545e68d", ("out", 32): "e9c8783e0efa8d13", ("out", 8): "3d8ba41006917f42",
                 ("in", 512): "7b0701bb680e83f3", ("out", 256): "b27cc82e87e68c52"}
E42_RUNNER_SHA = "340f9cc5dc876a24996b75263e3022e9529590ceca0a6e07f2ec829c93a7c7b5"
E42_F10_S_DRY2048 = ("4c9e52d1", "MOD97_LOCAL_RULE_20261106/exp45/e42_qual/check8_73129_runs/e42_F10_S_dry2048_73129.json")


# ---------------------------------------------------------------- network
def build(setting, seed):
    """E42's F10 start (arch26.new_net, then the cell layer's start), then E49's start of the two masked maps."""
    k_in, k_out = SETTINGS[setting]
    m = arch26.new_net(seed)
    layer = L42.Layer(m, seed, N=10)                 # F10: period set "1", variant "step", the local neighbourhood
    iface = I49.Interface(m, seed, k_in, k_out)
    return m, layer, iface


def install(layer, iface):
    """arch26.install(), then exp11_cell.forward replaced by a function of the harness's signature that runs E49's forward (scoring)."""
    arch26.install()

    def fwd(m, x, local, through_time, keep=False, silence=False):
        lg, hs = I49.forward(m, layer, iface, x, local, silence=silence)
        return lg, dict(h=hs)
    E.forward = fwd


def step(m, layer, iface, opt, items, local):
    """exp26_candidate.step's text with E49's forward in place of arch26.forward."""
    x, y, mask = X.batch_tensor(items)
    logits = I49.forward(m, layer, iface, x, local)[0]
    loss = E.cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach())


def kept(layer, iface):
    """The kept entries of each of the three maps: input map (768 x 512), lm (512 x 256), recurrent (768 x 256)."""
    return {"rnn.weight_ih_l0": iface.mask_in.repeat(3, 1), "lm.weight": iface.mask_out, "rnn.weight_hh_l0": layer.mask3()}


def params(m):
    return dict(m.named_parameters())


def check_masks(m, layer, iface, opt):
    """Check 3, at every scored row: every weight outside a mask exactly zero, and the optimizer's two running averages there zero."""
    P = params(m); out = {}
    for name, kp in kept(layer, iface).items():
        w = P[name]; o = ~kp
        st = opt.state.get(w)
        out[name] = dict(outside_zero=bool((w.detach()[o] == 0).all()),
                         averages_zero=True if not st else bool((st["exp_avg"][o] == 0).all()) and bool((st["exp_avg_sq"][o] == 0).all()))
    return out


def magnitudes(m, layer, iface):
    """Mean magnitude of the kept weights of each map."""
    P = params(m)
    return {name: float(P[name].detach().abs()[kp].mean()) for name, kp in kept(layer, iface).items()}


def map_grads(m, layer, iface, x, y, mask, local):
    for q in m.parameters():
        q.grad = None
    lg = I49.forward(m, layer, iface, x, local)[0]
    E.cross_entropy(lg[mask], y[mask]).backward()
    P = params(m)
    g = {n: P[n].grad.detach().clone() for n in ("rnn.weight_ih_l0", "lm.weight", "rnn.weight_hh_l0")}
    for q in m.parameters():
        q.grad = None
    return g


def cosines(m, layer, iface, items):
    """Cosine between the sweep's and the exact gradient of each masked map (over its kept entries), on one batch; the optimizer is not
    touched. The recurrent matrix (E42's) is recorded beside them."""
    x, y, mask = X.batch_tensor(items)
    gB, gS = map_grads(m, layer, iface, x, y, mask, False), map_grads(m, layer, iface, x, y, mask, True)
    cs = lambda a, b: float(T.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0))
    kp = kept(layer, iface)
    return {n: cs(gB[n][kp[n]], gS[n][kp[n]]) for n in gB}


def readers(iface):
    """Least and most number of cells that read an input value, and of readout units that read a cell."""
    ri, ro = iface.mask_in.sum(0), iface.mask_out.sum(0)
    return dict(input_value_min=int(ri.min()), input_value_max=int(ri.max()), cell_min=int(ro.min()), cell_max=int(ro.max()))


def draw_digests(seed, k_in, k_out):
    dg = lambda v: __import__("hashlib").sha256(json.dumps(v).encode()).hexdigest()[:16]
    a, b = I49.draws(seed, k_in, k_out)
    return dict(input=dg(a), readout=dg(b))


def weights_blob(m, layer, iface):
    full = dict(m.state_dict())
    full.update(mask=layer.mask.clone(), made=layer.made.clone(), periods=layer.p.clone(), phases=layer.f.clone(),
                mask_in=iface.mask_in.clone(), mask_out=iface.mask_out.clone(),
                k_in=T.tensor(iface.k_in), k_out=T.tensor(iface.k_out))
    return full


def save_pt(obj, path):
    tmp = Path(str(path) + ".tmp"); T.save(obj, tmp); os.replace(tmp, path)


# ---------------------------------------------------------------- one run
def receipt_path(out, setting, rule, seed, dose):
    tag = "" if dose == DOSE else f"_dry{dose}"
    return Path(out) / f"e49_{setting}_{rule}{tag}_{seed}.json"


def run(setting, rule, seed, out, cmd, dose=None):
    k_in, k_out = SETTINGS[setting]
    local = RULES[rule]
    dose = dose or DOSE
    assert dose % ROW == 0
    rp = receipt_path(out, setting, rule, seed, dose)
    paths = dict(receipt=rp, start=rp.with_name(rp.stem + "_start.pt"), weights=rp.with_suffix(".pt"),
                 answers=rp.with_name(rp.stem + "_answers.json"))
    R_.refuse_existing(*paths.values(), str(paths["weights"]) + ".tmp", str(paths["start"]) + ".tmp")
    Path(out).mkdir(parents=True, exist_ok=True)
    m, layer, iface = build(setting, seed)
    rd = readers(iface)
    print(f"e49 {setting} {rule} {seed}: least cells reading an input value {rd['input_value_min']}, least readout units reading a cell "
          f"{rd['cell_min']}", flush=True)
    if rd["input_value_min"] == 0 or rd["cell_min"] == 0:
        raise SystemExit(f"e49 {setting} {seed}: an input value or a cell is read by nobody; stopped as the brief requires")
    install(layer, iface)
    opt = R42.make_opt(m, layer)
    n_rec = int(layer.mask3().sum())
    cnt = iface.counts(m, n_rec)
    assert cnt["multiply_adds"] == MA_EXPECTED[setting], (setting, cnt)
    ma = cnt["multiply_adds"]
    A, Rs, TA, LP = R42.panels(seed)
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    bt = E.batches(tp, seed, E.KMAX["M"]); dg = C26.Digest()
    start = weights_blob(m, layer, iface)
    save_pt(start, paths["start"])
    rec = dict(experiment="E49", kind="birth", setting=setting, config=setting, rule=rule, local=local, seed=seed, dose=dose, row_every=ROW,
               k_in=k_in, k_out=k_out, cells=U, N=layer.N, batch=X.BATCH, kmax=E.KMAX["M"], counts=cnt, multiply_adds_per_position=ma,
               params_module=sum(q.numel() for q in m.parameters()), readers=rd, draw_digests=draw_digests(seed, k_in, k_out),
               start_scale=dict(input=(D / k_in) ** 0.5 if k_in < D else 1.0, lm=(U / k_out) ** 0.5 if k_out < U else 1.0),
               optimizer=R42.opt_record(m, layer, opt),
               update="exp26_candidate.step's text with e49_interface.forward(m, layer, iface, x, local) in place of arch26.forward",
               data="exp11_cell.batches([(t, split(t, .5, seed)[0]) for t in DEV], seed, 12)",
               forward="e49_interface.forward (the specification's own forward)",
               start_file=paths["start"].name, start_tensors_sha256=R_.tensors_sha(m.state_dict()),
               mask_sha256=dict(recurrent=R42.tsha(layer.mask), input=R42.tsha(iface.mask_in), readout=R42.tsha(iface.mask_out)),
               provenance=provenance(cmd), complete=False, curve=[])
    curve = rec["curve"]; cpu_up = cpu_sc = 0.0; w0 = time.time(); losses = []; first = None

    def score_row(u):
        nonlocal cpu_sc
        c = time.process_time()
        row = dict(step=u, **R42.score_all(m, A, Rs, TA, LP))
        cpu_sc += time.process_time() - c
        lim = R42.check_limit(m, layer)
        assert all(lim.values()), f"E42's check 4 failed at row {u}: {lim}"
        mk = check_masks(m, layer, iface, opt)
        assert all(all(v.values()) for v in mk.values()), f"check 3 failed at row {u}: {mk}"
        row.update(limit=lim, masks=mk, kept_mean_magnitude=magnitudes(m, layer, iface), cpu_updates=cpu_up, cpu_scoring=cpu_sc,
                   multiply_adds=ma * X.BATCH * u, data_digest=dg.value(),
                   loss=(sum(losses) / len(losses)) if losses else None, finite_losses=all(v == v and abs(v) != float("inf") for v in losses),
                   wall=time.time() - w0)
        curve.append(row)
        print(f"e49 {setting} {rule} {seed} {u}: A {row['A']:.2f} R {row['R']:.2f} taught {row['taught']:.2f} len12 {row['len12']:.1f} "
              f"len40 {row['len40']:.1f} cpu {cpu_up:.0f}+{cpu_sc:.0f}s", flush=True)

    score_row(0)
    for u in range(1, dose + 1):
        items = next(bt)
        if first is None:
            first = items
            rec["cosine_first_batch_start"] = cosines(m, layer, iface, first)
        dg.add(items)
        c = time.process_time(); losses.append(step(m, layer, iface, opt, items, local)); cpu_up += time.process_time() - c
        if u % ROW == 0:
            score_row(u); losses = []
            if u % 2048 == 0 or u == dose:
                rp.write_text(json.dumps(rec, indent=1))
    rec["cosine_first_batch_end"] = cosines(m, layer, iface, first)
    sd = m.state_dict()
    save_pt(weights_blob(m, layer, iface), paths["weights"])
    paths["answers"].write_text(json.dumps({str(k): E.respond(m, [p for p, _ in it]) for k, it in LP.items()}))
    rec.update(complete=True, weights=paths["weights"].name, weights_sha256=X16.sha_file(paths["weights"]),
               start_sha256=X16.sha_file(paths["start"]), tensors_sha256=R_.tensors_sha(sd), answers_file=paths["answers"].name,
               optimizer_steps=R_.opt_steps(opt), optimizer_read_back=R42.opt_record(m, layer, opt), data_digest=dg.value(),
               cpu_updates=cpu_up, cpu_scoring=cpu_sc, wall=time.time() - w0, processes_beside=os.environ.get("E49_BESIDE", "not recorded"))
    rp.write_text(json.dumps(rec, indent=1))
    print("e49 complete", rp.name, rec["tensors_sha256"][:16], flush=True)
    return rec


# ---------------------------------------------------------------- provenance
def provenance(cmd):
    p = C26.provenance(cmd)
    p["sources_sha256"] = {s: X16.sha_file(HERE / s) for s in SOURCES}
    p["runner_sha256"] = X16.sha_file(HERE / "exp49_interface.py")
    p["spec_sha256"] = X16.sha_file(SPEC) if SPEC.exists() else None
    p["e42_runner_sha256"] = X16.sha_file(HERE / "exp42_cells.py")
    return p


# ---------------------------------------------------------------- checks before the runs
def check1(qual, cmd):
    """The specification file: the runner's copy differs from it in its import line only; run on its own place (from a shell's environment),
    its self-checks print what OUTPUT_E49_SPEC_MAC.txt has."""
    import subprocess
    a = SPEC.read_text().splitlines(); b = (HERE / "e49_interface.py").read_text().splitlines()
    diff_lines = [(i + 1, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    only_import = len(a) == len(b) and len(diff_lines) == 1 and diff_lines[0][1].startswith("import e42_cells_spec as E42") \
        and diff_lines[0][2].startswith("import e42_cells as E42")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}
    t0 = time.time()
    res = subprocess.run([sys.executable, "-I", str(SPEC)], capture_output=True, text=True, env=env, cwd=str(HERE.parent))
    mine, ref = res.stdout.splitlines(), SPEC_OUT.read_text().splitlines()
    diffs = [(x, y) for x, y in zip(mine, ref) if x != y]
    ok = only_import and res.returncode == 0 and len(mine) == len(ref) and not diffs
    return dict(copy_differs_in_import_line_only=only_import, differing_lines=diff_lines, spec_sha256=X16.sha_file(SPEC),
                copy_sha256=X16.sha_file(HERE / "e49_interface.py"), returncode=res.returncode, stdout=mine, stderr=res.stderr[-3000:],
                lines_compared=len(ref), lines_equal=len(mine) == len(ref) and not diffs, differences=diffs, wall_spec=time.time() - t0,
                **{"pass": ok})


SCORE_KEYS = ("A", "R", "taught", "len12", "len40", "A_counts", "R_counts", "taught_counts", "length_counts")


def curve_scores(rec):
    return [{k: r[k] for k in ("step",) + SCORE_KEYS} for r in rec["curve"]]


def check2(qual, cmd):
    """With both masks full the runner is E42's F10 under both rules. Sweep: E49 runner, F10, 2,048 updates, against E42's
    e42_F10_S_dry2048_73129 (at 4c9e52d1). Backpropagation: E42's runner unchanged first made e42_F10_B_dry2048_73129 (its receipt is in
    QUAL/check2_e42_runner); the E49 runner must give the same tensors_sha256 and scores."""
    import subprocess
    out = dict(runs={})
    ref_S = json.loads(subprocess.run(["git", "show", f"{E42_F10_S_DRY2048[0]}:{E42_F10_S_DRY2048[1]}"], cwd=HERE, capture_output=True,
                                      text=True, check=True).stdout)
    ref_B_path = QUAL / "check2_e42_runner" / "e42_F10_B_dry2048_73129.json"
    ref_B = json.loads(ref_B_path.read_text())
    assert ref_B["complete"] and ref_B["provenance"]["runner_sha256"] == E42_RUNNER_SHA, "E42's backpropagation dry run is not by E42's runner"
    for rule, ref, src in (("S", ref_S, f"{E42_F10_S_DRY2048[0]}:{E42_F10_S_DRY2048[1]}"), ("B", ref_B, str(ref_B_path.relative_to(HERE.parent)))):
        rec = run("F10", rule, SEED, Path(qual), cmd, dose=2048)
        same_t = rec["tensors_sha256"] == ref["tensors_sha256"]
        same_s = curve_scores(rec) == curve_scores(ref)
        same_d = rec["data_digest"] == ref["data_digest"]
        out["runs"][rule] = dict(reference=src, reference_runner_sha256=ref["provenance"]["runner_sha256"], tensors_sha256=rec["tensors_sha256"],
                                 reference_tensors_sha256=ref["tensors_sha256"], tensors_equal=same_t, scores_equal_every_row=same_s,
                                 data_digest_equal=same_d, rows=len(rec["curve"]),
                                 scores_last_row={k: rec["curve"][-1][k] for k in ("A", "R", "taught", "len12", "len40")})
    out["pass"] = all(v["tensors_equal"] and v["scores_equal_every_row"] and v["data_digest_equal"] for v in out["runs"].values())
    return out


def check4(qual, cmd):
    """Same data: the batch digest at row 4,096 is E26's for the seed (E42's check 2, run again here)."""
    return R42.check2(SEED, qual, cmd)


def check5(qual, cmd):
    """The scorer: E42's check 12 on E26's six births, through E49's forward with both masks full and every cell connected (G's layer)."""
    out = dict(counts={}); worst = 0
    for rule in "SB":
        for fs in R42.SEEDS:
            m = arch26.new_net(fs); layer = L42.Layer(m, fs, N=255); iface = I49.Interface(m, fs, D, U)
            f_ = X.OUT / "e26" / f"e26_dev_{rule}_{fs}.pt"
            m.load_state_dict(T.load(f_)); install(layer, iface)
            A, Rs, TA, LP = R42.panels(fs)
            sc = R42.score_all(m, A, Rs, TA, LP)
            got = ((sc["A_counts"]["add89"], sc["A_counts"]["sub89"]), (sc["R_counts"]["add89"], sc["R_counts"]["sub89"]),
                   (sc["taught_counts"]["add89"], sc["taught_counts"]["sub89"]), (sc["length_counts"]["12"], sc["length_counts"]["40"]))
            exp = R42.CHECK12_COUNTS[(rule, fs)]
            d = max(abs(g - e) for gg, ee in zip(got, exp) for g, e in zip(gg, ee)); worst = max(worst, d)
            out["counts"][f_.name] = dict(file_sha256=X16.sha_file(f_), got=got, expected=exp, max_abs_diff=d)
            print(f_.name, got, exp, flush=True)
    out["max_abs_diff"] = worst
    out["verdict"] = "equal" if worst == 0 else "differs by one or two (reported)" if worst <= 2 else "FAIL"
    out["pass"] = worst <= 2
    return out


def check6(qual, cmd, settings=("I64", "I16", "in16", "out8"), rules="SB"):
    """One dry run of 256 updates per setting and rule, with CPU time and a forecast; the weights (start and end, with the masks) and the
    answers on the length-test panel are kept for the reviewer's reader."""
    out = dict(runs={})
    for s in settings:
        for rule in rules:
            rec = run(s, rule, SEED, Path(qual), cmd, dose=256)
            pu = rec["cpu_updates"] / 256; pr = rec["cpu_scoring"] / len(rec["curve"])
            rows = DOSE // ROW + 1
            out["runs"][f"{s} {rule}"] = dict(receipt=rec_name(s, rule, 256), cpu_per_update=pu, cpu_per_scored_row=pr,
                                              forecast_cpu_h=(DOSE * pu + rows * pr) / 3600, readers=rec["readers"],
                                              draw_digests=rec["draw_digests"], masks_every_row=all(all(all(v.values()) for v in r["masks"].values()) for r in rec["curve"]),
                                              finite=all(r["finite_losses"] for r in rec["curve"]), len12=rec["curve"][-1]["len12"],
                                              cosine_start=rec["cosine_first_batch_start"], processes_beside=rec["processes_beside"])
            print(f"cost {s} {rule}: {pu:.4f} s/update, {pr:.2f} s/row, {out['runs'][f'{s} {rule}']['forecast_cpu_h']:.2f} CPU-h", flush=True)
    out["pass"] = all(v["masks_every_row"] and v["finite"] for v in out["runs"].values())
    return out


def rec_name(s, rule, dose):
    return receipt_path(".", s, rule, SEED, dose).name


CHECKS = {"1": check1, "2": check2, "4": check4, "5": check5}


# ---------------------------------------------------------------- main
def main(argv):
    cmd = "python3 " + " ".join(argv)
    a = argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    if a[0] == "run":
        setting, rule, seed = a[1], a[2], int(a[3])
        assert setting in SETTINGS and rule in RULES and seed == SEED, "E49 is released for seed 73129 only"
        d = opt("--dose", None)
        run(setting, rule, seed, Path(opt("--out", str(OUT))), cmd, dose=int(d) if d else None)
    elif a[0] == "check":
        n = a[1]
        qual = Path(opt("--qual", str(QUAL))); qual.mkdir(parents=True, exist_ok=True)
        tag = opt("--tag", "")
        p = qual / f"check{n}_{SEED}{('_' + tag) if tag else ''}.json"
        R_.refuse_existing(p)
        t0 = time.process_time(); w0 = time.time()
        if n == "6":
            ss = tuple(opt("--settings", "I64,I16,in16,out8").split(",")); rules = opt("--rules", "SB")
            res = check6(qual / f"check6_{SEED}_runs", cmd, ss, rules)
        else:
            res = CHECKS[n](qual / f"check{n}_{SEED}_runs", cmd)
        res.update(check=int(n), seed=SEED, cpu=time.process_time() - t0, wall=time.time() - w0, provenance=provenance(cmd))
        p.write_text(json.dumps(res, indent=1))
        print(f"e49 check {n}:", "PASS" if res["pass"] else "FAIL", flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
