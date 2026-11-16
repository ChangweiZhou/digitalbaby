"""E41 (brief OPUS_TASK_E41_COMPUTATION_TO_THRESHOLD_BY_WIDTH_AND_FRACTION_20261116.md, revision 2): computation to threshold on E26's
birth task, at three widths (feedforward H, GRU units U) and three taught fractions, under the sweep and under backpropagation.

    python3 exp41_scale.py run <S|B> <H> <f> <seed> [--dose N] [--out DIR]   # f in 10, 25, 50; H in 256, 512, 1024 (U = H / 2)
    python3 exp41_scale.py check <1|2|3|4|5|6|7|8> [seed] [--qual DIR]
    python3 exp41_scale.py tables [--out DIR]                                  # thresholds and descriptions from the receipts

The network is arch20.NetRNN("gru")'s parameters under the same names, created in the same order, at any width (NetW); arch26.forward runs
on it unchanged. The update is exp26_candidate.step; the optimizer exp26_candidate.make_opt. exp26_candidate is imported before any other
harness module (it imports exp20_reference, which sets E16_BOOKDECAY=off and refuses foreign E16_/E11_ variables).

Every 256 updates (row 0 included) three scores: A, the held-out panel at 12 letters (n = 300, four tasks; A = mean of add and sub); R, a fixed
sample of 1,000 held-out reduction pairs per task (add: a + b >= 89; sub: a - b < 0) at no letters; and the taught panel at 12 letters
(seed + 1). CPU-seconds of updates exclude scoring; one thread.
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
import exp26_candidate as C          # noqa: E402  first harness import (environment gate)

import torch as T                    # noqa: E402
from torch import nn                 # noqa: E402

R_, X16, E, X, arch20, arch26 = C.R, C.X16, C.E, C.X, C.arch20, C.arch26
T.set_num_threads(1)

SEEDS = (73129, 93683, 116089)
WIDTHS = {256: 128, 512: 256, 1024: 512}
DOSE = {256: 40960, 512: 12288, 1024: 8192}
PARAMS = {256: 317457, 512: 1124113, 1024: 4211985}
FRACS = {10: 0.10, 25: 0.25, 50: 0.50}
RULES = {"S": True, "B": False}
ROW = 256
OUT = X.OUT / "e41"
QUAL = X.OUT / "e41_qual"
SOURCES = ("exp41_scale.py",) + C.SOURCES


class NetW(nn.Module):
    """arch20.NetRNN('gru') at feedforward width H and GRU units U: same names, same creation order."""

    def __init__(self, H, U):
        super().__init__()
        self.kind, self.units = "gru", U
        self.book = nn.Embedding(258, 8, padding_idx=257)     # BOS = 257, as arch20.py line 43
        self.l1 = nn.Linear(8, H)
        self.rnn = nn.GRU(H, U, num_layers=1, batch_first=True)
        self.lm = nn.Linear(U, H)
        self.l2 = nn.Linear(H, H)
        self.l3 = nn.Linear(H, 257)


def new_net(seed, H, U):
    T.manual_seed(seed); return NetW(H, U)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def describe(m):
    return dict(cls=f"{__name__}.NetW", H=m.l1.out_features, U=m.units, params=nparams(m),
                params_by_tensor={n: p.numel() for n, p in m.named_parameters()},
                forward=f"{arch26.forward.__module__}.{arch26.forward.__name__}")


# ---------------------------------------------------------------- provenance
def sha_file(p):
    return X16.sha_file(Path(p))


def provenance(cmd):
    p = C.provenance(cmd)
    p["sources_sha256"] = {s: sha_file(HERE / s) for s in SOURCES}
    p["runner_sha256"] = sha_file(HERE / "exp41_scale.py")
    return p


# ---------------------------------------------------------------- data and panels
def taught_held(f, seed):
    return {t: E.split(t, f, seed) for t in E.DEV}


def stream(f, seed):
    th = taught_held(f, seed)
    return E.batches([(t, th[t][0]) for t in E.DEV], seed, E.KMAX["M"])


def reduction_sample(t, held, f, seed):
    op = t[0]
    pairs = [(a, b) for a, b in held if (a + b >= 89 if op == "add" else a - b < 0)]
    return Random(f"e41|{seed}|{round(f * 100)}|{t[0]}").sample(sorted(pairs), 1000)


def sample_digest(s):
    return hashlib.sha256(json.dumps([list(p) for p in s]).encode()).hexdigest()


def panels(f, seed):
    th = taught_held(f, seed)
    A = {E.NAME(*t): E.panel(t, th[t][1], seed, ks=(12,), n=300)[12] for t in E.DEV}
    TA = {E.NAME(*t): E.panel(t, th[t][0], seed + 1, ks=(12,), n=300)[12] for t in E.DEV}
    Rs = {}
    for t in E.DEV[:2]:
        s = reduction_sample(t, th[t][1], f, seed)
        Rs[E.NAME(*t)] = [E.render(t, a, b, 0, Random(0)) for a, b in s]
    return A, Rs, TA


def score_all(m, A, Rs, TA):
    a = {n: E.score(m, it)["exact"] for n, it in A.items()}
    r = {n: E.score(m, it)["exact"] for n, it in Rs.items()}
    ta = {n: E.score(m, it)["exact"] for n, it in TA.items()}
    return dict(A_counts=a, R_counts=r, taught_counts=ta,
                A=100 * (a["add89"] + a["sub89"]) / 600, R=100 * (r["add89"] + r["sub89"]) / 2000,
                taught=100 * (ta["add89"] + ta["sub89"]) / 600)


def item_key(p):
    """(task code, a, b) of a rendered prompt: code, two digits, symbol, letters, two digits, '='."""
    return p[0], int(bytes(p[1:3])), int(bytes(p[-3:-1]))


def taught_keys(f, seed):
    th = taught_held(f, seed)
    return {(t[2][0], a, b) for t in E.DEV for a, b in th[t][0]}


def assert_taught(items, tk):
    for p, _ in items:
        assert item_key(p) in tk, f"a held-out pair was presented: {item_key(p)}"


# ---------------------------------------------------------------- one run
def receipt_path(out, rule, H, fp, seed, dose):
    tag = "" if dose == DOSE[H] else f"_dry{dose}"
    return Path(out) / f"e41_{rule}_h{H}_f{fp}{tag}_{seed}.json"


def run(rule, H, fp, seed, out, cmd, dose=None, e26panel=None):
    arch26.install()
    local, U, f = RULES[rule], WIDTHS[H], FRACS[fp]
    dose = dose or DOSE[H]
    assert dose % ROW == 0
    rp = receipt_path(out, rule, H, fp, seed, dose); wp = rp.with_suffix(".pt")
    R_.refuse_existing(rp, wp, str(wp) + ".tmp")
    Path(out).mkdir(parents=True, exist_ok=True)
    m = new_net(seed, H, U)
    opt = C.make_opt(m)
    A, Rs, TA = panels(f, seed)
    tk = taught_keys(f, seed)
    if e26panel is None:
        e26panel = (H == 512 and fp == 50)
    pan26 = {E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV} if e26panel else None
    bt = stream(f, seed); dg = C.Digest()
    rec = dict(experiment="E41", kind="birth", rule=rule, local=local, H=H, U=U, fraction=f, seed=seed, dose=dose, row_every=ROW,
               batch=X.BATCH, kmax=E.KMAX["M"], optimizer=C.opt_record(m, opt), model=describe(m),
               update="exp26_candidate.step (forward(local), answer-masked cross entropy, zero_grad, backward, AdamW step)",
               data="exp11_cell.batches([(t, split(t, f, seed)[0]) for t in DEV], seed, 12)",
               reduction_sample_sha256={n: sample_digest([item_key(p)[1:] for p, _ in it]) for n, it in Rs.items()},
               provenance=provenance(cmd), complete=False, curve=[])
    curve = rec["curve"]; cpu_up = 0.0; cpu_sc = 0.0; w0 = time.time(); losses = []

    def score_row(u):
        nonlocal cpu_sc
        c = time.process_time()
        row = dict(step=u, **score_all(m, A, Rs, TA))
        if pan26 is not None and u in (4096, 8192):
            row["e26_sealed"] = {n: {str(k): E.score(m, it) for k, it in pk.items()} for n, pk in pan26.items()}
        cpu_sc += time.process_time() - c
        row.update(cpu_updates=cpu_up, cpu_scoring=cpu_sc, weight_items=nparams(m) * X.BATCH * u, data_digest=dg.value(),
                   loss=(sum(losses) / len(losses)) if losses else None, finite_losses=all(v == v and abs(v) != float("inf") for v in losses),
                   wall=time.time() - w0)
        curve.append(row)
        print(f"e41 {rule} h{H} f{fp} {seed} {u}: A {row['A']:.2f} R {row['R']:.2f} taught {row['taught']:.2f} "
              f"cpu {cpu_up:.0f}+{cpu_sc:.0f}s", flush=True)

    score_row(0)
    for u in range(1, dose + 1):
        items = next(bt)
        assert_taught(items, tk)
        dg.add(items)
        c = time.process_time(); losses.append(C.step(m, opt, items, local)); cpu_up += time.process_time() - c
        if u % ROW == 0:
            score_row(u); losses = []
            if u % 4096 == 0 or u == dose:
                rp.write_text(json.dumps(rec, indent=1))
    wsha = R_.save_weights(m, wp)
    rec.update(complete=True, weights=wp.name, weights_sha256=wsha, tensors_sha256=R_.tensors_sha(m.state_dict()),
               optimizer_steps=R_.opt_steps(opt), optimizer_read_back=C.opt_record(m, opt), data_digest=dg.value(),
               cpu_updates=cpu_up, cpu_scoring=cpu_sc, wall=time.time() - w0,
               processes_beside=os.environ.get("E41_BESIDE", "not recorded"))
    rp.write_text(json.dumps(rec, indent=1))
    print("e41 complete", rp.name, rec["tensors_sha256"][:16], flush=True)
    return rec


# ---------------------------------------------------------------- checks
CHECK7_SAMPLES = {
    73129: {10: ("2d9d251a354dc291", "92c29b4a374be126"), 25: ("3f31c9a15729adc3", "89a86290d78daddd"), 50: ("f433a55ca52ae34d", "18ac75b5013b6d47")},
    93683: {10: ("7a5f67e2fc00ed81", "c316f9e9ca56339e"), 25: ("81d35edfe17f2b54", "19b61154e17286f4"), 50: ("368844acc03dfd7a", "83274756387907f7")},
    116089: {10: ("49965d4fbf5633ee", "3efe08c20a9e5ad3"), 25: ("bef6158428f9008a", "0fdd40508ad49609"), 50: ("26b2eff70e51e033", "8ccb9c3177772cbf")},
}
CHECK7_COUNTS = {   # (A add, sub), (R add, sub), (taught add, sub)
    ("S", 73129): ((295, 293), (962, 955), (298, 298)), ("S", 93683): ((296, 293), (988, 979), (297, 298)),
    ("S", 116089): ((300, 299), (993, 996), (300, 296)), ("B", 73129): ((295, 296), (976, 987), (295, 297)),
    ("B", 93683): ((298, 294), (970, 968), (298, 298)), ("B", 116089): ((284, 279), (955, 895), (285, 288)),
}
E26_DIGEST_4096 = {73129: "bbd71ab7ca27b72e", 93683: "65b1c7e06b17459a", 116089: "1df203fb8f299dc0"}
DIGEST_2048 = "4f622d94e0156993"
DRY2048 = {"S": "79618ab2", "B": "adf933d7"}
E35A_DIGEST_4096 = {25: "a71b1ca10b24a36b", 10: "e120924e5af546f9"}


def data_digest_at(f, seed, n):
    bt = stream(f, seed); dg = C.Digest(); tk = taught_keys(f, seed)
    for _ in range(n):
        it = next(bt); assert_taught(it, tk); dg.add(it)
    return dg.value()["sha256"]


def check1(seed, qual, cmd):
    out = dict(state_dicts={})
    for s in SEEDS:
        a, b = new_net(s, 512, 256).state_dict(), arch26.new_net(s).state_dict()
        out["state_dicts"][str(s)] = list(a) == list(b) and all(T.equal(a[k], b[k]) for k in a)
    out["runs"] = {}
    for rule in "SB":
        rec = run(rule, 512, 50, 73129, qual, cmd, dose=2048, e26panel=False)
        out["runs"][rule] = dict(tensors_sha256=rec["tensors_sha256"], expected_prefix=DRY2048[rule],
                                 equal=rec["tensors_sha256"].startswith(DRY2048[rule]),
                                 digest_2048=rec["data_digest"]["sha256"], digest_equal=rec["data_digest"]["sha256"].startswith(DIGEST_2048))
    out["pass"] = all(out["state_dicts"].values()) and all(v["equal"] and v["digest_equal"] for v in out["runs"].values())
    return out


def check2(seed, qual, cmd):
    out = {}
    for fp, f in FRACS.items():
        d = data_digest_at(f, seed, 4096)
        out[f"f{fp}"] = dict(digest_4096=d)
        if fp == 50:
            out[f"f{fp}"]["e26_expected_prefix"] = E26_DIGEST_4096[seed]; out[f"f{fp}"]["equal_e26"] = d.startswith(E26_DIGEST_4096[seed])
        if seed == 73129 and fp in E35A_DIGEST_4096:   # check 8's second part (reported, not a failure)
            out[f"f{fp}"]["e35a_G_expected_prefix"] = E35A_DIGEST_4096[fp]; out[f"f{fp}"]["equal_e35a_G"] = d.startswith(E35A_DIGEST_4096[fp])
    out["note"] = ("the stream is built from (seed, fraction) only; equality across the six runs of a seed and fraction at row 4,096 is checked "
                   "again from the full receipts by `tables`")
    out["pass"] = out["f50"]["equal_e26"]
    return out


def check3(seed, qual, cmd):
    got = {H: nparams(new_net(seed, H, U)) for H, U in WIDTHS.items()}
    return dict(params={str(k): v for k, v in got.items()}, expected={str(k): v for k, v in PARAMS.items()}, **{"pass": got == PARAMS})


def check4(seed, qual, cmd):
    out = {}; ok = True
    for H in (256, 1024):
        m = new_net(seed, H, WIDTHS[H])
        items = next(stream(.5, seed)); x, y, mask = X.batch_tensor(items)
        lo_ref, g_ref = C.grads_of(m, lambda m_, x_: arch20._forward_rnn(m_, x_, False)[0], x, y, mask)
        lo_b, g_b = C.grads_of(m, lambda m_, x_: arch26.forward(m_, x_, False, True)[0], x, y, mask)
        lo_s, g_s = C.grads_of(m, lambda m_, x_: arch26.forward(m_, x_, True, True)[0], x, y, mask)
        rel = lambda ga, gb: {n: float((ga[n] - gb[n]).abs().max()) / max(float(ga[n].abs().max()), 1e-30) for n in ga}
        rb, rs = rel(g_ref, g_b), rel(g_ref, g_s)
        r = dict(logits_max_abs_diff=float((lo_b - lo_ref).abs().max()), grads_rel_B=rb, grads_rel_S=rs,
                 B_within=all(v <= 1e-5 for v in rb.values()), S_differs=[n for n, v in rs.items() if v > 1e-5])
        r["pass"] = r["logits_max_abs_diff"] <= 1e-6 and r["B_within"] and bool(r["S_differs"])
        ok = ok and r["pass"]; out[f"h{H}"] = r
    out["tolerances"] = dict(logits_abs=1e-6, grads_rel_of_tensor_max=1e-5)
    out["pass"] = ok
    return out


def check5(seed, qual, cmd):
    out = {}; ok = True
    for fp, f in FRACS.items():
        th = taught_held(f, seed); tk = taught_keys(f, seed)
        disj = {}
        for t in E.DEV[:2]:
            s = reduction_sample(t, th[t][1], f, seed)
            disj[t[0]] = dict(n=len(s), distinct=len(set(s)), in_taught=sum((t[2][0], a, b) in tk for a, b in s))
        bt = stream(f, seed); presented = 0
        for _ in range(4096):
            it = next(bt); assert_taught(it, tk); presented += len(it)
        planted = E.render(E.DEV[0], *th[E.DEV[0]][1][0], 3, Random(1))
        try:
            assert_taught([planted], tk); fires = False
        except AssertionError:
            fires = True
        r = dict(reduction_sample=disj, batches_asserted=4096, items_asserted=presented, planted_held_out_item_refused=fires)
        r["pass"] = fires and all(v["in_taught"] == 0 and v["distinct"] == 1000 for v in disj.values())
        ok = ok and r["pass"]; out[f"f{fp}"] = r
    out["note"] = "every batch of every run is asserted in the run itself (assert_taught); this check repeats it on the first 4,096 batches"
    out["pass"] = ok
    return out


def check6(seed, qual, cmd):
    out = dict(runs={}, note="serial dry runs on an otherwise idle host; dose 256, rows 0 and 256 scored")
    for H in (256, 512, 1024):
        for rule in "BS":
            reps = 2 if (H != 512 and rule == "B") else 1
            for i in range(reps):
                q = Path(qual) / f"dry256_rep{i}"
                rec = run(rule, H, 50, seed, q, cmd, dose=256, e26panel=False)
                row = rec["curve"][-1]
                out["runs"][f"{rule} h{H} rep{i}"] = dict(cpu_per_update=rec["cpu_updates"] / 256,
                                                          cpu_per_scored_row=rec["cpu_scoring"] / 2, tensors_sha256=rec["tensors_sha256"],
                                                          A=row["A"], R=row["R"])
    rr = out["runs"]
    out["repeat_equal"] = {f"h{H}": rr[f"B h{H} rep0"]["tensors_sha256"] == rr[f"B h{H} rep1"]["tensors_sha256"] for H in (256, 1024)}
    fc = {}; tot_u = tot_s = 0.0
    for H in (256, 512, 1024):
        for rule in "BS":
            per_u = rr[f"{rule} h{H} rep0"]["cpu_per_update"]; per_s = rr[f"{rule} h{H} rep0"]["cpu_per_scored_row"]
            u, s = 3 * DOSE[H] * per_u, 3 * (DOSE[H] // ROW + 1) * per_s
            fc[f"{rule} h{H}"] = dict(updates_cpu_h=u / 3600, scoring_cpu_h=s / 3600); tot_u += u; tot_s += s
    out["forecast_18_runs"] = dict(by_rule_width_three_fractions=fc, updates_cpu_h=tot_u / 3600, scoring_cpu_h=tot_s / 3600,
                                   total_cpu_h=(tot_u + tot_s) / 3600, limit_cpu_h=15)
    out["pass"] = all(out["repeat_equal"].values()) and (tot_u + tot_s) / 3600 <= 15
    return out


def check7(seed, qual, cmd):
    arch26.install()
    out = dict(samples={}, counts={}); ok = True
    for fp, f in FRACS.items():
        th = taught_held(f, seed)
        got = tuple(sample_digest(reduction_sample(t, th[t][1], f, seed)) for t in E.DEV[:2])
        exp = CHECK7_SAMPLES[seed][fp]
        r = dict(add=got[0], sub=got[1], expected=exp, equal=got[0].startswith(exp[0]) and got[1].startswith(exp[1]))
        if fp == 50 and seed == 73129:
            r["add_begins"] = [list(p) for p in reduction_sample(E.DEV[0], th[E.DEV[0]][1], f, seed)[:4]]
            r["add_begins_expected"] = [[30, 64], [84, 37], [64, 49], [46, 54]]
            r["equal"] = r["equal"] and r["add_begins"] == r["add_begins_expected"]
        ok = ok and r["equal"]; out["samples"][f"f{fp}"] = r
    worst = 0
    for rule, fs in ((r_, s_) for r_ in "SB" for s_ in SEEDS):      # the six E26 births, each scored with its own file's seed
        A, Rs, TA = panels(.5, fs)
        f_ = X.OUT / "e26" / f"e26_dev_{rule}_{fs}.pt"
        m = NetW(512, 256); m.load_state_dict(T.load(f_))
        sc = score_all(m, A, Rs, TA)
        got = ((sc["A_counts"]["add89"], sc["A_counts"]["sub89"]), (sc["R_counts"]["add89"], sc["R_counts"]["sub89"]),
               (sc["taught_counts"]["add89"], sc["taught_counts"]["sub89"]))
        exp = CHECK7_COUNTS[(rule, fs)]
        d = max(abs(g - e) for gg, ee in zip(got, exp) for g, e in zip(gg, ee)); worst = max(worst, d)
        out["counts"][f_.name] = dict(file_sha256=sha_file(f_), got=got, expected=exp, max_abs_diff=d, A=sc["A"], R=sc["R"], taught=sc["taught"])
    out["counts_max_abs_diff"] = worst
    out["counts_verdict"] = "equal" if worst == 0 else "differs by one or two (reported, proceeds)" if worst <= 2 else "FAIL"
    out["pass"] = ok and worst <= 2
    return out


def check8(seed, qual, cmd):
    """From the full present-width f = 0.50 receipts: E26's panel at rows 4,096 and 8,192 equals the `sealed` counts of e26_dev_<S|B>_<seed>."""
    out = {}; ok = True
    for rule in "SB":
        rp = receipt_path(OUT, rule, 512, 50, seed, DOSE[512])
        if not rp.exists():
            out[rule] = "receipt not yet present"; ok = False; continue
        mine = {r["step"]: r for r in json.loads(rp.read_text())["curve"] if "e26_sealed" in r}
        e26 = {r["step"]: r["sealed"] for r in json.loads((X.OUT / "e26" / f"e26_dev_{rule}_{seed}.json").read_text())["curve"]}
        res = {}
        for u in (4096, 8192):
            a = {n: {k: v["exact"] for k, v in d.items()} for n, d in mine[u]["e26_sealed"].items()}
            b = {n: {k: v["exact"] for k, v in d.items()} for n, d in e26[u].items()}
            res[str(u)] = dict(equal=a == b, mine=a, e26=b)
            ok = ok and a == b
        out[rule] = res
    out["pass"] = ok
    return out


CHECKS = {"1": check1, "2": check2, "3": check3, "4": check4, "5": check5, "6": check6, "7": check7, "8": check8}


# ---------------------------------------------------------------- thresholds and descriptions
THRESHOLDS = (50, 80, 90)


def first_row(curve, key, th, version):
    v = [(r["step"], r[key]) for r in curve]
    for i, (u, a) in enumerate(v):
        if version == "single" and a >= th:
            return i
        if version == "three" and i + 2 < len(v) and all(x >= th for _, x in v[i:i + 3]):
            return i
    return None


def summarize(rec):
    curve = rec["curve"]; out = dict(rule=rec["rule"], H=rec["H"], fraction=rec["fraction"], seed=rec["seed"], dose=rec["dose"],
                                     params=rec["model"]["params"], thresholds={})
    for key in ("A", "R"):
        for th in THRESHOLDS:
            for ver in ("single", "three"):
                i = first_row(curve, key, th, ver)
                r = curve[i] if i is not None else None
                out["thresholds"][f"{key}{th}_{ver}"] = None if r is None else dict(
                    updates=r["step"], weight_items=r["weight_items"], cpu_seconds=r["cpu_updates"], taught=r["taught"])
        best = max(curve, key=lambda r: r[key])
        out[f"max_{key}"] = dict(value=best[key], row=best["step"])
        lv = [r[key] for r in curve if r["step"] in (7680, 7936, 8192)]
        out[f"mean_{key}_7680_8192"] = sum(lv) / len(lv) if len(lv) == 3 else None
    out["digest_4096"] = next((r["data_digest"]["sha256"] for r in curve if r["step"] == 4096), None)
    out["final"] = dict(A=curve[-1]["A"], R=curve[-1]["R"], taught=curve[-1]["taught"])
    out["cpu_updates"], out["cpu_scoring"] = rec["cpu_updates"], rec["cpu_scoring"]
    out["processes_beside"] = rec.get("processes_beside")
    return out


def describe_step(xn, xw, Dn, Dw):
    if xn is not None and xw is not None:
        q = xw / xn
        return "falls" if q < 2 / 3 else "rises" if q > 1.5 else "stays"
    if xn is not None:
        return "rises" if Dw / xn >= 1.5 else "not decided"
    if xw is not None:
        return "falls" if xw / Dn <= 2 / 3 else "not decided"
    return "neither reaches"


def step_description(sn, sw, key, ver, unit):
    t = f"{key}_{ver}"
    cn, cw = sn["thresholds"][t], sw["thresholds"][t]
    mul_n, mul_w = (1, 1) if unit == "updates" else (sn["params"], sw["params"])
    Dn = (sn["dose"] - (512 if ver == "three" else 0)) * mul_n
    Dw = (sw["dose"] - (512 if ver == "three" else 0)) * mul_w
    xn = cn["updates"] * mul_n if cn else None
    xw = cw["updates"] * mul_w if cw else None
    return describe_step(xn, xw, Dn, Dw), xn, xw


def tables(out_dir):
    recs = {}
    for p in sorted(Path(out_dir).glob("e41_*_h*_f*_*.json")):
        if "_dry" in p.name:
            continue
        r = json.loads(p.read_text())
        if r.get("complete"):
            recs[(r["rule"], r["H"], round(r["fraction"] * 100), r["seed"])] = summarize(r)
    return recs


def fmt(c, unit):
    if c is None:
        return "not reached"
    return f"{c['updates']:,}" if unit == "updates" else f"{c['weight_items']:.3g}"


def write_tables(out_dir, seeds, md_path, json_path):
    S = tables(out_dir)
    lines = []
    have = sorted({k[3] for k in S})
    lines.append(f"Receipts read from `{Path(out_dir).relative_to(HERE)}`; seeds present: {', '.join(map(str, have))}; runs: {len(S)}.\n")
    # data digests
    lines.append("### Batch digest at row 4,096 (check 2)\n")
    lines.append("| Seed | f | digests equal across the six runs | digest (first 16) |\n|---|---|---|---|")
    for seed in seeds:
        for fp in FRACS:
            ds = {S[k]["digest_4096"] for k in S if k[2] == fp and k[3] == seed}
            n = sum(1 for k in S if k[2] == fp and k[3] == seed)
            if n:
                lines.append(f"| {seed} | {fp / 100:.2f} | {'yes' if len(ds) == 1 else 'NO'} ({n} runs) | {', '.join(d[:16] for d in ds if d)} |")
    lines.append("")
    # per-run table
    for seed in seeds:
        lines.append(f"### Seed {seed}: updates and computation to threshold (three-row; single-row in brackets; * where they differ in description)\n")
        for key in ("A", "R"):
            lines.append(f"**Accuracy {key}**\n")
            lines.append("| Rule | f | H | " + " | ".join(f"{key}{th}: updates [single]" for th in THRESHOLDS)
                         + f" | {key}80 weight-items | {key}80 CPU-s | taught at {key}80 | max {key} (row) | mean {key} rows 7,680-8,192 |")
            lines.append("|" + "---|" * (3 + len(THRESHOLDS) + 5))
            for rule in "SB":
                for fp in FRACS:
                    for H in WIDTHS:
                        k = (rule, H, fp, seed)
                        if k not in S:
                            continue
                        s = S[k]; cells = []
                        for th in THRESHOLDS:
                            c3, c1 = s["thresholds"][f"{key}{th}_three"], s["thresholds"][f"{key}{th}_single"]
                            cells.append(f"{fmt(c3, 'updates')} [{fmt(c1, 'updates')}]")
                        c = s["thresholds"][f"{key}80_three"]
                        lines.append(f"| {rule} | {fp / 100:.2f} | {H} | " + " | ".join(cells)
                                     + f" | {fmt(c, 'wi')} | {'' if c is None else format(c['cpu_seconds'], '.0f')}"
                                     + f" | {'' if c is None else format(c['taught'], '.1f')}"
                                     + f" | {s['max_' + key]['value']:.2f} ({s['max_' + key]['row']:,}) | {s['mean_' + key + '_7680_8192']:.2f} |")
            lines.append("")
    # descriptions per seed and three-seed cells
    for ths in ((80,), (50, 90)):
        lines.append(f"### Descriptions by width step, threshold{'s' if len(ths) > 1 else ''} {', '.join(map(str, ths))} (three-row version; "
                     "per-seed descriptions with the seeds' values; * where the single-row version describes the cell differently)\n")
        lines.append("| Threshold | Measure | Unit | Rule | f | Step | Cell | Per seed (description: narrower, wider) |\n|---|---|---|---|---|---|---|---|")
        for th in ths:
            for key in ("R", "A"):
                for unit in ("updates", "weight-items"):
                    for rule in "SB":
                        for fp in FRACS:
                            for hn, hw in ((256, 512), (512, 1024)):
                                per = []; per1 = []
                                for seed in seeds:
                                    kn, kw = (rule, hn, fp, seed), (rule, hw, fp, seed)
                                    if kn in S and kw in S:
                                        d, xn, xw = step_description(S[kn], S[kw], f"{key}{th}", "three", unit)
                                        d1, _, _ = step_description(S[kn], S[kw], f"{key}{th}", "single", unit)
                                        f_ = (lambda v: "nr" if v is None else (f"{v:,}" if unit == "updates" else f"{v:.3g}"))
                                        per.append((seed, d, f_(xn), f_(xw))); per1.append(d1)
                                if not per:
                                    continue
                                ds = [p[1] for p in per]
                                cell = next((d for d in set(ds) if ds.count(d) >= 2), "mixed") if len(per) == 3 else "(seeds missing)"
                                ds1 = per1
                                cell1 = next((d for d in set(ds1) if ds1.count(d) >= 2), "mixed") if len(per1) == 3 else None
                                mark = "*" if (len(per) == 3 and cell1 != cell) or any(a != b for a, b in zip(ds, ds1)) else ""
                                lines.append(f"| {th} | {key} | {unit} | {rule} | {fp / 100:.2f} | {hn}→{hw} | {cell}{mark} | "
                                             + "; ".join(f"{s_}: {d} ({a}, {b})" for s_, d, a, b in per) + " |")
        lines.append("")
    Path(md_path).write_text("\n".join(lines) + "\n")
    Path(json_path).write_text(json.dumps({f"{k[0]} h{k[1]} f{k[2]} {k[3]}": v for k, v in sorted(S.items())}, indent=1))
    return S


# ---------------------------------------------------------------- main
def main(argv):
    cmd = "python3 " + " ".join(argv)
    a = argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    if a[0] == "run":
        rule, H, fp, seed = a[1], int(a[2]), int(a[3]), int(a[4])
        assert rule in RULES and H in WIDTHS and fp in FRACS and seed in SEEDS
        d = opt("--dose", None)
        run(rule, H, fp, seed, Path(opt("--out", str(OUT))), cmd, dose=int(d) if d else None)
    elif a[0] == "check":
        n = a[1]; seed = int(a[2]) if len(a) > 2 and not a[2].startswith("--") else 73129
        qual = Path(opt("--qual", str(QUAL)))
        qual.mkdir(parents=True, exist_ok=True)
        p = qual / f"check{n}_{seed}.json"
        R_.refuse_existing(p)
        t0 = time.process_time(); w0 = time.time()
        res = CHECKS[n](seed, qual / f"check{n}_{seed}_runs", cmd)
        res.update(check=int(n), seed=seed, cpu=time.process_time() - t0, wall=time.time() - w0, provenance=provenance(cmd))
        p.write_text(json.dumps(res, indent=1))
        print(f"e41 check {n} seed {seed}:", "PASS" if res["pass"] else "FAIL", flush=True)
    elif a[0] == "tables":
        out = Path(opt("--out", str(OUT)))
        seeds = [int(s) for s in opt("--seeds", ",".join(map(str, SEEDS))).split(",")]
        tag = opt("--tag", "all")
        write_tables(out, seeds, out / f"E41_TABLES_{tag}.md", out / f"E41_SUMMARY_{tag}.json")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
