"""E46 (brief OPUS_TASK_E46_DOES_LEARNING_THE_FIRST_FAMILY_NEED_WEIGHT_DECAY_20261116.md, revision 2): does learning the first family need
weight decay? A copy of exp41_scale.py (E41's runner, at 88b8b7b3) with one added argument, the decay of the optimizer's first group. Present
width (H 512, U 256), taught fraction 0.50, E26's full birth dose of 24,576 updates; decays 0.25 (the reference), 0.05 and 0.

    python3 exp46_decay.py run <S|B> <wd: 000|005|025> <seed> [--dose N] [--out DIR]
    python3 exp46_decay.py check <1a|1b|2|3|4|5> [seed] [--qual DIR]

After opt = exp26_candidate.make_opt(m) (which asserts the standing groups), opt.param_groups[0]["weight_decay"] = wd; the groups are read back
and asserted at the start and the end of the run. The embedding table has no decay in any arm. Scores every 256 updates (row 0 included): E41's
A, R and taught panel, plus the length test of E30 and E42 (add89 at 12 and 40 letters, n = 300), and the norm of each of seven weight groups.
CPU-seconds of updates exclude scoring; one thread.
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
OUT = X.OUT / "e46"
QUAL = X.OUT / "e46_qual"
SOURCES = ("exp46_decay.py", "exp41_scale.py") + C.SOURCES
DECAYS = {"000": 0.0, "005": 0.05, "025": 0.25}
DOSE46 = 24576
H46, U46, FP46, F46 = 512, 256, 50, 0.50
LEN_TASK = ("add", 89, b"p")


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
    p["runner_sha256"] = sha_file(HERE / "exp46_decay.py")
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



# ---------------------------------------------------------------- E46 additions: decay, length test, norms
SAMPLES_F50 = {73129: ("f433a55ca52ae34d", "18ac75b5013b6d47"), 93683: ("368844acc03dfd7a", "83274756387907f7"),
               116089: ("26b2eff70e51e033", "8ccb9c3177772cbf")}      # E41 check 7's table, half of the pairs
GROUPS = {"book": ("book.weight",), "l1": ("l1.weight", "l1.bias"), "input_map": ("rnn.weight_ih_l0", "rnn.bias_ih_l0"),
          "recurrent": ("rnn.weight_hh_l0", "rnn.bias_hh_l0"), "lm": ("lm.weight", "lm.bias"), "l2": ("l2.weight", "l2.bias"),
          "l3": ("l3.weight", "l3.bias")}


def length_panel(seed):
    return E.panel(LEN_TASK, E.split(LEN_TASK, .5, seed)[1], seed, ks=(12, 40), n=300)      # E30's length test, as E42


def score_length(m, LP):
    c = {str(k): E.score(m, it)["exact"] for k, it in LP.items()}
    return dict(length_counts=c, len12=100 * c["12"] / 300, len40=100 * c["40"] / 300)


@T.no_grad()
def norms(m):
    sd = dict(m.named_parameters())
    return {g: float(sum(float((sd[n].double() ** 2).sum()) for n in ns) ** .5) for g, ns in GROUPS.items()}


def expected_groups(m, wd):
    g = C.expected_optimizer(m); g[0]["weight_decay"] = wd
    return g


def make_opt(m, wd):
    opt = C.make_opt(m)                          # asserts the standing groups (0.25; the byte table without decay)
    opt.param_groups[0]["weight_decay"] = wd
    assert R_.opt_groups(m, opt) == expected_groups(m, wd), R_.opt_groups(m, opt)
    return opt


def opt_record(m, opt, wd):
    """exp26_candidate.opt_record writes the constant 0.25 at the top level; this writes the run's own decay and the groups read back."""
    return dict(name="AdamW", lr=C.LR, weight_decay=wd, betas=list(C.BETAS), groups=R_.opt_groups(m, opt))


# ---------------------------------------------------------------- one run
def receipt_path(out, rule, wdt, seed, dose):
    tag = "" if dose == DOSE46 else f"_dry{dose}"
    return Path(out) / f"e46_{rule}_wd{wdt}{tag}_{seed}.json"


def run(rule, wdt, seed, out, cmd, dose=None):
    arch26.install()
    local, wd = RULES[rule], DECAYS[wdt]
    dose = dose or DOSE46
    assert dose % ROW == 0
    rp = receipt_path(out, rule, wdt, seed, dose); wp = rp.with_suffix(".pt")
    R_.refuse_existing(rp, wp, str(wp) + ".tmp")
    Path(out).mkdir(parents=True, exist_ok=True)
    m = new_net(seed, H46, U46)
    opt = make_opt(m, wd)
    A, Rs, TA = panels(F46, seed)
    LP = length_panel(seed)
    tk = taught_keys(F46, seed)
    sdig = {n: sample_digest([item_key(p)[1:] for p, _ in it]) for n, it in Rs.items()}
    assert sdig["add89"].startswith(SAMPLES_F50[seed][0]) and sdig["sub89"].startswith(SAMPLES_F50[seed][1]), f"reduction sample differs: {sdig}"
    pan26 = {E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV}
    bt = stream(F46, seed); dg = C.Digest()
    rec = dict(experiment="E46", kind="birth", rule=rule, local=local, decay=wd, H=H46, U=U46, fraction=F46, seed=seed, dose=dose, row_every=ROW,
               batch=X.BATCH, kmax=E.KMAX["M"], optimizer=opt_record(m, opt, wd), optimizer_groups_at_start=R_.opt_groups(m, opt), model=describe(m),
               update="exp26_candidate.step (forward(local), answer-masked cross entropy, zero_grad, backward, AdamW step)",
               data="exp11_cell.batches([(t, split(t, .5, seed)[0]) for t in DEV], seed, 12)",
               reduction_sample_sha256=sdig, reduction_sample_matches_e41_check7=True, length_test="exp11_cell.panel(('add', 89, b'p'), split(.5, seed)[1], seed, ks=(12, 40), n=300)",
               norm_groups={g: list(ns) for g, ns in GROUPS.items()},
               provenance=provenance(cmd), complete=False, curve=[])
    curve = rec["curve"]; cpu_up = 0.0; cpu_sc = 0.0; w0 = time.time(); losses = []

    def score_row(u):
        nonlocal cpu_sc
        c = time.process_time()
        row = dict(step=u, **score_all(m, A, Rs, TA), **score_length(m, LP), norms=norms(m))
        if u in (4096, 8192):
            row["e26_sealed"] = {n: {str(k): E.score(m, it) for k, it in pk.items()} for n, pk in pan26.items()}
        cpu_sc += time.process_time() - c
        row.update(cpu_updates=cpu_up, cpu_scoring=cpu_sc, weight_items=nparams(m) * X.BATCH * u, data_digest=dg.value(),
                   loss=(sum(losses) / len(losses)) if losses else None, finite_losses=all(v == v and abs(v) != float("inf") for v in losses),
                   wall=time.time() - w0)
        curve.append(row)
        print(f"e46 {rule} wd{wdt} {seed} {u}: A {row['A']:.2f} R {row['R']:.2f} taught {row['taught']:.2f} len40 {row['len40']:.1f} "
              f"rec-norm {row['norms']['recurrent']:.2f} cpu {cpu_up:.0f}+{cpu_sc:.0f}s", flush=True)

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
    g_end = R_.opt_groups(m, opt)
    assert g_end == expected_groups(m, wd), f"groups changed during the run: {g_end}"
    wsha = R_.save_weights(m, wp)
    rec.update(complete=True, weights=wp.name, weights_sha256=wsha, tensors_sha256=R_.tensors_sha(m.state_dict()),
               optimizer_steps=R_.opt_steps(opt), optimizer_read_back=opt_record(m, opt, wd), optimizer_groups_at_end=g_end,
               data_digest=dg.value(), cpu_updates=cpu_up, cpu_scoring=cpu_sc, wall=time.time() - w0,
               processes_beside=os.environ.get("E46_BESIDE", "not recorded"))
    rp.write_text(json.dumps(rec, indent=1))
    print("e46 complete", rp.name, rec["tensors_sha256"][:16], flush=True)
    return rec


# ---------------------------------------------------------------- checks
DRY2048 = {"S": "79618ab2", "B": "adf933d7"}
E26_FINAL = {("S", 73129): "d9064126", ("S", 93683): "1eb83bdd", ("S", 116089): "70ae363b",
             ("B", 73129): "efbdf87a", ("B", 93683): "6c4ff9ce", ("B", 116089): "a5b5ab81"}
E26_DIGEST_4096 = {73129: "bbd71ab7ca27b72e", 93683: "65b1c7e06b17459a", 116089: "1df203fb8f299dc0"}


def check1a(seed, qual, cmd):
    """Dry 2,048 at decay 0.25, seed 73129, every added score on: E26's dry tensors_sha256 under both rules."""
    out = {}
    for rule in "SB":
        rec = run(rule, "025", 73129, qual, cmd, dose=2048)
        out[rule] = dict(tensors_sha256=rec["tensors_sha256"], expected_prefix=DRY2048[rule], equal=rec["tensors_sha256"].startswith(DRY2048[rule]))
    out["pass"] = all(v["equal"] for v in out.values())
    return out


def check1b(seed, qual, cmd):
    """After the full runs at 0.25: (b) seed 73129's rows to 12,288 equal E41's counts; final tensors_sha256 of the six = E26's births;
    (c) E26's panel at 4,096 and 8,192 = E26's sealed counts; (d) seed 73129's length test = e42_G_<S|B>_73129 at every row (reported)."""
    import subprocess
    out = dict(b_rows={}, b_final={}, c={}, d={}); ok = True
    for rule in "SB":
        for s in SEEDS:
            rp = receipt_path(OUT, rule, "025", s, DOSE46)
            if not rp.exists():
                out["b_final"][f"{rule} {s}"] = "receipt missing"; ok = False; continue
            r = json.loads(rp.read_text())
            eq = r.get("complete") and r["tensors_sha256"].startswith(E26_FINAL[(rule, s)])
            out["b_final"][f"{rule} {s}"] = dict(tensors_sha256=r.get("tensors_sha256"), expected_prefix=E26_FINAL[(rule, s)], equal=bool(eq)); ok = ok and bool(eq)
            e26 = {x["step"]: x["sealed"] for x in json.loads((X.OUT / "e26" / f"e26_dev_{rule}_{s}.json").read_text())["curve"]}
            mine = {x["step"]: x for x in r["curve"] if "e26_sealed" in x}
            cc = {}
            for u in (4096, 8192):
                a = {n: {k: v["exact"] for k, v in d.items()} for n, d in mine[u]["e26_sealed"].items()}
                b = {n: {k: v["exact"] for k, v in d.items()} for n, d in e26[u].items()}
                cc[str(u)] = a == b; ok = ok and a == b
            out["c"][f"{rule} {s}"] = cc
            if s == 73129:
                e41 = {x["step"]: x for x in json.loads((X.OUT / "e41" / f"e41_{rule}_h512_f50_73129.json").read_text())["curve"]}
                bad = [x["step"] for x in r["curve"] if x["step"] <= 12288 and any(x[k] != e41[x["step"]][k] for k in ("A_counts", "R_counts", "taught_counts"))]
                out["b_rows"][rule] = dict(rows_compared=sum(1 for x in r["curve"] if x["step"] <= 12288), rows_differing=bad); ok = ok and not bad
                try:
                    src = subprocess.run(["git", "show", f"origin/claude/gallant-dijkstra-i9zgsr:MOD97_LOCAL_RULE_20261106/exp45/e42/e42_G_{rule}_73129.json"],
                                         cwd=HERE, capture_output=True, text=True, check=True).stdout
                    e42 = {x["step"]: x["length_counts"] for x in json.loads(src)["curve"]}
                    common = [x["step"] for x in r["curve"] if x["step"] in e42]
                    diff = [u for u in common if {str(k): v for k, v in e42[u].items()} != next(x for x in r["curve"] if x["step"] == u)["length_counts"]]
                    out["d"][rule] = dict(source="claude/gallant-dijkstra-i9zgsr", rows_compared=len(common), rows_differing=diff, equal=not diff)
                except Exception as e:     # noqa: BLE001  reported, not a failure
                    out["d"][rule] = dict(not_compared=str(e)[:300])
    out["pass"] = ok
    return out


def check2(seed, qual, cmd):
    """One update from the same start under backpropagation at decay d and at decay 0: d-hat from the difference."""
    arch26.install()
    items = next(stream(F46, 73129))
    out = {}; ok = True
    for wdt in ("025", "005"):
        d = DECAYS[wdt]
        res = {}
        for w in (d, 0.0):
            m = new_net(73129, H46, U46); p0 = {n: p.detach().clone() for n, p in m.named_parameters()}
            opt = make_opt(m, w); C.step(m, opt, items, False)
            res[w] = ({n: p.detach().clone() for n, p in m.named_parameters()}, R_.opt_groups(m, opt))
        p0 = {n: p.detach().clone() for n, p in new_net(73129, H46, U46).named_parameters()}
        p1d, p10 = res[d][0], res[0.0][0]
        per = {}
        for n in p0:
            diff = (p10[n] - p1d[n]).double(); q = p0[n].double()
            if n == "book.weight":
                per[n] = dict(bit_equal=bool(T.equal(p10[n], p1d[n])))
                ok = ok and per[n]["bit_equal"]
            else:
                dh = float((diff * q).sum() / (0.001 * (q * q).sum()))
                resid = float((diff - 0.001 * d * q).abs().max())
                per[n] = dict(d_hat=dh, within_1pct=abs(dh - d) <= 0.01 * d, max_abs_residual=resid)
                ok = ok and per[n]["within_1pct"]
        out[f"d={d}"] = dict(tensors=per, groups_read_back={str(w): res[w][1] for w in res})
    out["pass"] = ok
    return out


def check3(seed, qual, cmd):
    out = {}
    for s in SEEDS:
        bt = stream(F46, s); dg = C.Digest(); tk = taught_keys(F46, s)
        for _ in range(4096):
            it = next(bt); assert_taught(it, tk); dg.add(it)
        d = dg.value()["sha256"]
        out[str(s)] = dict(digest_4096=d, expected_prefix=E26_DIGEST_4096[s], equal=d.startswith(E26_DIGEST_4096[s]))
    out["note"] = "the runs' own digests at row 4,096 are compared again from the receipts by e46_tables.py"
    out["pass"] = all(v["equal"] for v in out.values() if isinstance(v, dict))
    return out


def check4(seed, qual, cmd):
    """The length test on E26's six final files gives E30's figures at 12 and 40 letters (exp45/e30/E30_UNSEEN.json), as E42's check 12."""
    arch26.install()
    e30 = {x["file"]: x["detail"]["H"]["counts"] for x in json.loads((X.OUT / "e30" / "E30_UNSEEN.json").read_text())["rows"]}
    out = {}; ok = True
    for rule in "SB":
        for s in SEEDS:
            fn = f"e26_dev_{rule}_{s}.pt"
            m = NetW(512, 256); m.load_state_dict(T.load(X.OUT / "e26" / fn))
            got = score_length(m, length_panel(s))["length_counts"]
            exp = {k: v["exact"] for k, v in e30[fn].items()}
            out[fn] = dict(got=got, e30=exp, equal=got == exp); ok = ok and got == exp
    out["pass"] = ok
    return out


def check5(seed, qual, cmd):
    out = dict(runs={}, note="serial dry runs of 256 updates (rows 0 and 256 scored) on an otherwise idle host, seed 73129")
    for wdt in DECAYS:
        for rule in "BS":
            rec = run(rule, wdt, 73129, Path(qual), cmd, dose=256)
            out["runs"][f"{rule} wd{wdt}"] = dict(cpu_per_update=rec["cpu_updates"] / 256, cpu_per_scored_row=rec["cpu_scoring"] / 2,
                                                  tensors_sha256=rec["tensors_sha256"])
    tot_u = tot_s = 0.0
    for k, v in out["runs"].items():
        tot_u += 3 * DOSE46 * v["cpu_per_update"]; tot_s += 3 * (DOSE46 // ROW + 1) * v["cpu_per_scored_row"]
    out["forecast_18_runs"] = dict(updates_cpu_h=tot_u / 3600, scoring_cpu_h=tot_s / 3600, total_cpu_h=(tot_u + tot_s) / 3600, brief_estimate_cpu_h=16.2)
    out["pass"] = True
    return out


CHECKS = {"1a": check1a, "1b": check1b, "2": check2, "3": check3, "4": check4, "5": check5}


# ---------------------------------------------------------------- main
def main(argv):
    cmd = "python3 " + " ".join(argv)
    a = argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    if a[0] == "run":
        rule, wdt, seed = a[1], a[2], int(a[3])
        assert rule in RULES and wdt in DECAYS and seed in SEEDS
        d = opt("--dose", None)
        run(rule, wdt, seed, Path(opt("--out", str(OUT))), cmd, dose=int(d) if d else None)
    elif a[0] == "check":
        n = a[1]; seed = int(a[2]) if len(a) > 2 and not a[2].startswith("--") else 73129
        qual = Path(opt("--qual", str(QUAL)))
        qual.mkdir(parents=True, exist_ok=True)
        p = qual / f"check{n}_{seed}.json"
        R_.refuse_existing(p)
        t0 = time.process_time(); w0 = time.time()
        res = CHECKS[n](seed, qual / f"check{n}_{seed}_runs", cmd)
        res.update(check=n, seed=seed, cpu=time.process_time() - t0, wall=time.time() - w0, provenance=provenance(cmd))
        p.write_text(json.dumps(res, indent=1))
        print(f"e46 check {n} seed {seed}:", "PASS" if res["pass"] else "FAIL", flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
