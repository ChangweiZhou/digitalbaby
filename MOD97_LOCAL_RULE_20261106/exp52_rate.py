"""E52 (brief TASK_E52_THE_TWO_RULES_UNDER_OTHER_LEARNING_RATES_20261117.md, revision 2): the two learning rules under other learning rates.
E41's birth (exp41_scale.run: the first family alone, 0 to 12 letters, batches of 64, E41's stream, panels and scores) with the learning rate
of the optimizer as an argument, and, in the control runs only, the decay of the optimizer's first group set to 0.25 * 0.001 / rate.

    python3 exp52_rate.py run <S|B> <H: 256|512> <f: 25|50> <seed> <rate> [--product] [--dose N] [--out DIR]
    python3 exp52_rate.py check <1a|1b|2|3> [--qual DIR]

After opt = exp26_candidate.make_opt(m) (which asserts the standing groups), every parameter group's lr is set to the run's rate; the decay is
left as make_opt set it (0.25 on every parameter but the byte table), except with --product, where group 0's is 0.25 * 0.001 / rate. The groups
are read back and asserted at the start and at the end of the run. Every 256 updates (row 0 included): E41's A, R and taught panel, the seven
norms of exp46_decay.norms (imported unchanged from a byte-for-byte copy of E46's runner), and the length test (add under modulus 89 at 12 and
40 letters, n = 300) on pairs that the run's own stream never teaches. A row whose losses are not all finite stops the run (diverged).
CPU-seconds of updates exclude scoring; one thread. No existing source file is edited.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C          # noqa: E402  first harness import (environment gate)
import exp41_scale as S41            # noqa: E402  E41's runner, unchanged
import exp46_decay as D46            # noqa: E402  E46's runner, a byte-for-byte copy (norms and its groups)

import torch as T                    # noqa: E402

R_, X16, E, X, arch26 = C.R, C.X16, C.E, C.X, C.arch26
T.set_num_threads(1)

RATES = (0.0003, 0.001, 0.003, 0.01)
ROW = S41.ROW
RULES = S41.RULES
WIDTHS = {256: 128, 512: 256}
FRACS = {25: 0.25, 50: 0.50}
LONG_DOSE = {256: 136704, 512: 40960}          # E41's dose * 10 / 3, rounded up to a row, at rate 0.0003
OUT = X.OUT / "e52"
QUAL = X.OUT / "e52_qual"
SOURCES = ("exp52_rate.py", "exp46_decay.py") + S41.SOURCES
LEN_TASK = ("add", 89, b"p")
E46_COPY = dict(file="exp46_decay.py", from_branch="origin/claude/wonderful-edison-8hx6i6",
                from_commit="b76f5979", note="copied byte for byte with git show; imported unchanged")


def dose_of(H, rate):
    return LONG_DOSE[H] if rate == 0.0003 else S41.DOSE[H]


def decay_of(rate, product):
    return 0.25 * 0.001 / rate if product else C.WD


def provenance(cmd):
    p = C.provenance(cmd)
    p["sources_sha256"] = {s: S41.sha_file(HERE / s) for s in SOURCES}
    p["runner_sha256"] = S41.sha_file(HERE / "exp52_rate.py")
    p["runner"] = "exp52_rate.py"
    p["exp46_decay_copy"] = E46_COPY
    return p


def expected_groups(m, rate, wd):
    g = C.expected_optimizer(m)
    for x in g:
        x["lr"] = rate
    g[0]["weight_decay"] = wd
    return g


def make_opt(m, rate, wd, product):
    opt = C.make_opt(m)                          # asserts the standing groups (0.001; 0.25; the byte table without decay)
    for g in opt.param_groups:
        g["lr"] = rate
    if product:
        opt.param_groups[0]["weight_decay"] = wd
    assert R_.opt_groups(m, opt) == expected_groups(m, rate, wd), R_.opt_groups(m, opt)
    return opt


def opt_record(m, opt, rate, wd):
    """exp26_candidate.opt_record writes 0.001 and 0.25 whatever the groups hold; this writes the run's own numbers and the groups read back."""
    return dict(name="AdamW", lr=rate, weight_decay=wd, betas=[0.9, 0.98], groups=R_.opt_groups(m, opt))


def length_panel(f, seed):
    return E.panel(LEN_TASK, E.split(LEN_TASK, f, seed)[1], seed, ks=(12, 40), n=300)


def score_length(m, LP):
    c = {str(k): E.score(m, it)["exact"] for k, it in LP.items()}
    return dict(length_counts=c, len12=100 * c["12"] / 300, len40=100 * c["40"] / 300)


def receipt_path(out, rule, H, fp, seed, rate, product, dose):
    tag = "" if dose == dose_of(H, rate) else f"_dry{dose}"
    return Path(out) / f"e52_{rule}_h{H}_f{fp}_lr{rate:g}{'_product' if product else ''}{tag}_{seed}.json"


def e41_digest_4096(rule, H, fp, seed):
    p = S41.OUT / f"e41_{rule}_h{H}_f{fp}_{seed}.json"
    if not p.exists():
        return None
    return next(r["data_digest"]["sha256"] for r in json.loads(p.read_text())["curve"] if r["step"] == 4096)


def run(rule, H, fp, seed, rate, product, out, cmd, dose=None):
    arch26.install()
    local, U, f = RULES[rule], WIDTHS[H], FRACS[fp]
    assert rate in RATES
    wd = decay_of(rate, product)
    dose = dose or dose_of(H, rate)
    assert dose % ROW == 0
    rp = receipt_path(out, rule, H, fp, seed, rate, product, dose); wp = rp.with_suffix(".pt")
    R_.refuse_existing(rp, wp, str(wp) + ".tmp")
    Path(out).mkdir(parents=True, exist_ok=True)
    m = S41.new_net(seed, H, U)
    opt = make_opt(m, rate, wd, product)
    g_exp = expected_groups(m, rate, wd)
    A, Rs, TA = S41.panels(f, seed)
    LP = length_panel(f, seed)
    len12_is_A_add = LP[12] == A["add89"]
    tk = S41.taught_keys(f, seed)
    len_in_taught = sum(S41.item_key(p) in tk for k in LP for p, _ in LP[k])
    assert len_in_taught == 0, "the length test holds a taught pair"
    d41 = e41_digest_4096(rule, H, fp, seed)
    bt = S41.stream(f, seed); dg = C.Digest()
    rec = dict(experiment="E52", kind="birth", rule=rule, local=local, rate=rate, decay=wd, product_control=product, H=H, U=U, fraction=f,
               seed=seed, dose=dose, e41_dose=S41.DOSE[H], row_every=ROW, batch=X.BATCH, kmax=E.KMAX["M"],
               optimizer=opt_record(m, opt, rate, wd), optimizer_groups_at_start=R_.opt_groups(m, opt), optimizer_groups_expected=g_exp,
               model=S41.describe(m),
               update="exp26_candidate.step (forward(local), answer-masked cross entropy, zero_grad, backward, AdamW step)",
               data="exp11_cell.batches([(t, split(t, f, seed)[0]) for t in DEV], seed, 12) (exp41_scale.stream)",
               reduction_sample_sha256={n: S41.sample_digest([S41.item_key(p)[1:] for p, _ in it]) for n, it in Rs.items()},
               length_test="exp11_cell.panel(('add', 89, b'p'), split(('add', 89, b'p'), f, seed)[1], seed, ks=(12, 40), n=300), f the run's fraction",
               length_test_12_items_equal_A_add89=len12_is_A_add, length_test_items_in_taught=len_in_taught,
               norm_groups={g: list(ns) for g, ns in D46.GROUPS.items()}, e41_digest_4096=d41,
               provenance=provenance(cmd), complete=False, diverged=False, curve=[])
    curve = rec["curve"]; cpu_up = 0.0; cpu_sc = 0.0; w0 = time.time(); losses = []

    def score_row(u):
        nonlocal cpu_sc
        c = time.process_time()
        row = dict(step=u, **S41.score_all(m, A, Rs, TA), **score_length(m, LP), norms=D46.norms(m))
        if len12_is_A_add:
            assert row["length_counts"]["12"] == row["A_counts"]["add89"], "the length test at 12 letters and A's add count differ on the same items"
        cpu_sc += time.process_time() - c
        row.update(cpu_updates=cpu_up, cpu_scoring=cpu_sc, weight_items=S41.nparams(m) * X.BATCH * u, data_digest=dg.value(),
                   loss=(sum(losses) / len(losses)) if losses else None, finite_losses=all(v == v and abs(v) != float("inf") for v in losses),
                   wall=time.time() - w0)
        curve.append(row)
        print(f"e52 {rule} h{H} f{fp} lr{rate:g}{' product' if product else ''} {seed} {u}: A {row['A']:.2f} R {row['R']:.2f} "
              f"taught {row['taught']:.2f} len40 {row['len40']:.1f} rec-norm {row['norms']['recurrent']:.2f} cpu {cpu_up:.0f}+{cpu_sc:.0f}s", flush=True)
        return row

    score_row(0)
    done = 0
    for u in range(1, dose + 1):
        items = next(bt)
        S41.assert_taught(items, tk)
        dg.add(items)
        c = time.process_time(); losses.append(C.step(m, opt, items, local)); cpu_up += time.process_time() - c
        done = u
        if u % ROW == 0:
            row = score_row(u); losses = []
            if u == 4096 and d41 is not None:
                rec["digest_4096_equals_e41"] = row["data_digest"]["sha256"] == d41
                assert rec["digest_4096_equals_e41"], "the batch digest at row 4,096 differs from E41's receipt"
            if not row["finite_losses"]:
                rec.update(diverged=True, diverged_at_row=u)
                print(f"e52 diverged at row {u}: stopped", flush=True)
                break
            if u % 4096 == 0 or u == dose:
                rp.write_text(json.dumps(rec, indent=1))
    g_end = R_.opt_groups(m, opt)
    assert g_end == g_exp, f"groups changed during the run: {g_end}"
    wsha = R_.save_weights(m, wp)
    rec.update(complete=True, updates_done=done, weights=wp.name, weights_sha256=wsha, tensors_sha256=R_.tensors_sha(m.state_dict()),
               optimizer_steps=R_.opt_steps(opt), optimizer_read_back=opt_record(m, opt, rate, wd), optimizer_groups_at_end=g_end,
               data_digest=dg.value(), cpu_updates=cpu_up, cpu_scoring=cpu_sc, wall=time.time() - w0,
               processes_beside=os.environ.get("E52_BESIDE", "not recorded"))
    rp.write_text(json.dumps(rec, indent=1))
    print("e52 complete", rp.name, rec["tensors_sha256"][:16], "DIVERGED" if rec["diverged"] else "", flush=True)
    return rec


# ---------------------------------------------------------------- checks
def check1a(qual, cmd):
    """Rate 0.001, decay 0.25, 2,048 updates, present width, f = 0.50, seed 73129: E41 check 1's tensors_sha256 under each rule."""
    out = {}
    for rule in "SB":
        rec = run(rule, 512, 50, 73129, 0.001, False, qual, cmd, dose=2048)
        exp = S41.DRY2048[rule]
        out[rule] = dict(tensors_sha256=rec["tensors_sha256"], expected_prefix=exp, equal=rec["tensors_sha256"].startswith(exp),
                         digest_2048=rec["data_digest"]["sha256"], digest_equal=rec["data_digest"]["sha256"].startswith(S41.DIGEST_2048))
    out["pass"] = all(v["equal"] and v["digest_equal"] for v in out.values())
    return out


def check1b(qual, cmd, seed=93683):
    """After the eight runs at 0.001: final tensors_sha256 = E41's receipt of the same case; A, R and taught counts equal at every row."""
    out = {}; ok = True
    for rule in "SB":
        for H in WIDTHS:
            for fp in FRACS:
                k = f"{rule} h{H} f{fp}"
                rp = receipt_path(OUT, rule, H, fp, seed, 0.001, False, dose_of(H, 0.001))
                if not rp.exists() or not json.loads(rp.read_text()).get("complete"):
                    out[k] = "receipt missing or not complete"; ok = False; continue
                r = json.loads(rp.read_text()); e = json.loads((S41.OUT / f"e41_{rule}_h{H}_f{fp}_{seed}.json").read_text())
                e41 = {x["step"]: x for x in e["curve"]}
                bad = [x["step"] for x in r["curve"] if x["step"] not in e41
                       or any(x[c] != e41[x["step"]][c] for c in ("A_counts", "R_counts", "taught_counts"))]
                dig = [x["step"] for x in r["curve"] if x["step"] in e41 and x["data_digest"] != e41[x["step"]]["data_digest"]]
                res = dict(tensors_sha256=r["tensors_sha256"], e41_tensors_sha256=e["tensors_sha256"], final_equal=r["tensors_sha256"] == e["tensors_sha256"],
                           rows_compared=len(r["curve"]), e41_rows=len(e["curve"]), rows_differing_in_counts=bad, rows_differing_in_batch_digest=dig,
                           diverged=r["diverged"])
                res["pass"] = res["final_equal"] and not bad and not dig and len(r["curve"]) == len(e["curve"]) and not r["diverged"]
                ok = ok and res["pass"]; out[k] = res
    out["pass"] = ok
    return out


def check2(qual, cmd):
    """One update from E41's start (seed 73129, present width, f = 0.50, first batch) under back-propagation.
    (a) (rate, decay d) against (rate, 0): every tensor of group 0 differs by rate * d * start within 1%; the byte table not at all.
    (b) the change at (rate, 0) is rate / 0.001 times the change at (0.001, 0) within 1%, every tensor."""
    arch26.install()
    items = next(S41.stream(.5, 73129))

    def one(rate, wd):
        m = S41.new_net(73129, 512, 256)
        opt = C.make_opt(m)
        for g in opt.param_groups:
            g["lr"] = rate
        opt.param_groups[0]["weight_decay"] = wd
        assert R_.opt_groups(m, opt) == expected_groups(m, rate, wd)
        C.step(m, opt, items, False)
        assert R_.opt_groups(m, opt) == expected_groups(m, rate, wd)
        return {n: p.detach().clone() for n, p in m.named_parameters()}

    p0 = {n: p.detach().clone() for n, p in S41.new_net(73129, 512, 256).named_parameters()}
    base = one(0.001, 0.0)
    out = dict(start="exp41_scale.new_net(73129, 512, 256); first batch of exp41_scale.stream(.5, 73129); local=False", rates={}); ok = True
    for rate in (0.0003, 0.003, 0.01):
        free = one(rate, 0.0)
        res = dict(a={}, b={})
        decays = {"main (0.25)": 0.25}
        if rate != 0.01:
            decays["product (0.25 * 0.001 / rate)"] = 0.25 * 0.001 / rate
        for name, d in decays.items():
            pd = one(rate, d); per = {}
            for n in p0:
                diff = (free[n] - pd[n]).double(); q = p0[n].double()
                if n == "book.weight":
                    per[n] = dict(bit_equal=bool(T.equal(free[n], pd[n]))); ok = ok and per[n]["bit_equal"]
                else:
                    dh = float((diff * q).sum() / (rate * (q * q).sum()))
                    per[n] = dict(decay_recovered=dh, within_1pct=abs(dh - d) <= 0.01 * d,
                                  max_abs_residual=float((diff - rate * d * q).abs().max()))
                    ok = ok and per[n]["within_1pct"]
            res["a"][name] = dict(decay=d, per_tensor=per)
        for n in p0:
            d1 = (base[n] - p0[n]).double(); dr = (free[n] - p0[n]).double()
            ratio = float((dr * d1).sum() / (d1 * d1).sum()); want = rate / 0.001
            relres = float((dr - want * d1).abs().max() / d1.abs().max())
            res["b"][n] = dict(ratio_of_change=ratio, expected=want, within_1pct=abs(ratio - want) <= 0.01 * want, max_residual_rel_to_max_change=relres)
            ok = ok and res["b"][n]["within_1pct"] and relres <= 0.01
        out["rates"][f"{rate:g}"] = res
    out["c_and_d"] = ("in every run the groups read back at the start and at the end are asserted equal to the expected ones (rate, decay, betas "
                      "(0.9, 0.98)); the batch digest at row 4,096 is asserted equal to E41's receipt of the same rule, width, seed and fraction")
    out["pass"] = ok
    return out


def planned_runs():
    runs = []
    for rate in (0.001, 0.003, 0.0003):
        for H in (256, 512):
            for fp in (50, 25):
                for rule in "BS":
                    runs.append((rule, H, fp, rate, False))
    for rate in (0.003, 0.0003):
        for rule in "BS":
            runs.append((rule, 256, 50, rate, True))
    return runs


def check3(qual, cmd):
    """Dry runs of 256 updates at each width under each rule (rate 0.001, f = 0.50, seed 73129), serial, and the forecast for the 28 runs."""
    out = dict(runs={}, note="serial dry runs; dose 256, rows 0 and 256 scored with every score of the run (A, R, taught, length test, norms)")
    for H in (256, 512):
        for rule in "BS":
            rec = run(rule, H, 50, 73129, 0.001, False, qual, cmd, dose=256)
            out["runs"][f"{rule} h{H}"] = dict(cpu_per_update=rec["cpu_updates"] / 256, cpu_per_scored_row=rec["cpu_scoring"] / 2,
                                               tensors_sha256=rec["tensors_sha256"])
    tu = ts = 0.0; n = 0
    for rule, H, fp, rate, product in planned_runs():
        d = dose_of(H, rate); r = out["runs"][f"{rule} h{H}"]
        tu += d * r["cpu_per_update"]; ts += (d // ROW + 1) * r["cpu_per_scored_row"]; n += 1
    out["forecast"] = dict(runs=n, updates_cpu_h=tu / 3600, scoring_cpu_h=ts / 3600, total_cpu_h=(tu + ts) / 3600, brief_cpu_h=26,
                           limit_cpu_h=26 * 1.25)
    out["pass"] = n == 28 and (tu + ts) / 3600 <= 26 * 1.25
    return out


CHECKS = {"1a": check1a, "1b": check1b, "2": check2, "3": check3}


def main(argv):
    cmd = "python3 " + " ".join(argv)
    a = argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    if a and a[0] == "run":
        rule, H, fp, seed, rate = a[1], int(a[2]), int(a[3]), int(a[4]), float(a[5])
        assert rule in RULES and H in WIDTHS and fp in FRACS and seed in S41.SEEDS and rate in RATES
        product = "--product" in a
        assert not product or rate in (0.0003, 0.003)
        d = opt("--dose", None)
        run(rule, H, fp, seed, rate, product, Path(opt("--out", str(OUT))), cmd, dose=int(d) if d else None)
    elif a and a[0] == "check":
        n = a[1]
        qual = Path(opt("--qual", str(QUAL)))
        qual.mkdir(parents=True, exist_ok=True)
        p = qual / f"check{n}.json"
        R_.refuse_existing(p)
        t0 = time.process_time(); w0 = time.time()
        res = CHECKS[n](qual / f"check{n}_runs", cmd)
        res.update(check=n, cpu=time.process_time() - t0, wall=time.time() - w0, provenance=provenance(cmd))
        p.write_text(json.dumps(res, indent=1))
        print(f"e52 check {n}:", "PASS" if res["pass"] else "FAIL", flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
