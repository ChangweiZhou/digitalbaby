"""E53 runner (brief TASK_E53_THE_NETWORK_WITH_AN_INTERFACE_AT_A_BOUNDARY_WITHOUT_DECAY_AND_FROM_A_LARGER_START_20261117.md, revision 4):
E47's network at 32 cells a side in its arm `one`, born on E26's task (the first family) from E47's start with the recurrent weights Wc
multiplied by 3 (all entries, or all but the interface cells' weights among themselves), with the trial's weight decay of 0.25 or with 0.
A new file; every existing source stays as it is. exp47_local.py and e47_local.py are byte-for-byte copies from
origin/claude/e47-worker-stage-0-3d0ya6 at 578a877b (E53_COPIED_SOURCES.txt), imported and used unchanged.

`run` is a copy of the text of exp47_local.run with the brief's changes 1 to 8 and nothing else (the report shows the difference).

    python3 exp53_local_start.py run <arm> <S|B> 73129 [--dose N] [--out DIR] [--resume]
    python3 exp53_local_start.py check <1|2|3|4|5|6>
    arms: one (check 1 only), x3, wd0, x3-wd0, xf3, xf3-wd0
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp47_local as X47            # noqa: E402  (imports exp26_candidate first: the environment gate)
from exp47_local import (build, install, make_opt, forward, step, score_row_all, magnitudes, cosines, weights, save_weights,  # noqa: E402
                         save_ckpt, tsha)
import torch as T                    # noqa: E402
import numpy                         # noqa: E402
import arch26                        # noqa: E402
import e47_local as L47              # noqa: E402

C26, R_, X16, E, X, R42 = X47.C26, X47.R_, X47.X16, X47.E, X47.X, X47.R42
T.set_num_threads(1)

SEED = 73129
DOSE, ROW, RULES, N = X47.DOSE, X47.ROW, X47.RULES, X47.N
CKPT_EVERY = X47.CKPT_EVERY
OUT = HERE / "exp45" / "e53"
QUAL = HERE / "exp45" / "e53_qual"
SIDE = 32
NET_ARM = "one"
ARMS = {  # arm: (multiple, entries multiplied, weight decay)
    "one": (1, "none", 0.25),
    "x3": (3, "all", 0.25),
    "wd0": (1, "none", 0.0),
    "x3-wd0": (3, "all", 0.0),
    "xf3": (3, "all but the interface cells' weights among themselves", 0.25),
    "xf3-wd0": (3, "all but the interface cells' weights among themselves", 0.0),
}
START_SHA = "62de40fb57d41d280b23d3f35c2e3943e13d59a18b183cff48d6f6acc897ef33"
STRUCT_SHA = dict(sources="f414eb354435af94dd1489e0f6414a6c1e92bae55bc5ff6fb03c79370acdb63b",
                  periods="9aa02009b29c3933c0e1277e60e997e88c80595c4a06ef905fe5e34ac89ec0de",
                  phases="57a6e9770cd2bd3d4039e4b3c41aacd8e5e1f694c7b15f3075ff3f5d030404fd")
SAVE_ROWS = (24064, 24320)
W7 = "origin/claude/e47-worker-stage-0-3d0ya6"
TRIAL = HERE / "exp45" / "e53_trial_files"          # the trial's files read with git show; not committed


# ---------------------------------------------------------------- change 2: the start
def multiply(L, arm):
    """The entries of Wc named by the arm, multiplied in place, in single precision. Returns the count of entries multiplied."""
    mult, which, _ = ARMS[arm]
    with T.no_grad():
        if which == "none":
            return 0
        if which == "all":
            L.Wc.mul_(3)
            return L.Wc.numel()
        far = T.zeros(L.C, dtype=T.bool); far[L.farx] = True
        mask = far[L.idx]; mask[L.farx] = True
        assert int(mask.sum()) == 8975, int(mask.sum())
        L.Wc[:, mask] *= 3
        return 3 * int(mask.sum())


# ---------------------------------------------------------------- change 3: the decay
def set_decay(opt, decay):
    g = [g_ for g_ in opt.param_groups if g_["weight_decay"] == C26.WD]
    assert len(g) == 1
    g[0]["weight_decay"] = decay


def assert_decay(opt, decay):
    got = [g_["weight_decay"] for g_ in opt.param_groups]
    assert got == [decay, 0.0], got


def opt_record(m, L, opt, decay):
    r = X47.opt_record(m, L, opt)
    r["weight_decay"] = decay
    return r


# ---------------------------------------------------------------- change 6: rings and gates
def rings(L):
    r = T.tensor([-(-L47.distance(i, L.side) // L47.RADIUS) for i in range(L.C)])
    cnt = [int((r == k).sum()) for k in range(4)]
    assert cnt == [256, 228, 300, 240], cnt
    return r


GROUPS = ("interface", "ring1", "ring2", "ring3")


def ring_magnitudes(L, ring):
    w = L.Wc.detach().abs()
    return {g: float(w[:, ring == k].mean()) for k, g in enumerate(GROUPS)}


@T.no_grad()
def gates(m, L, ring, x, tol=1e-5):
    """On x (every position, padding included): for each group of cells the share of reset- and update-gate values below 0.05 or above
    0.95, and the mean absolute state; the gates recomputed from the states that forward(m, L, x, False) returns."""
    assert bool((L.p == 1).all())
    _, hs = forward(m, L, x, False)
    u = m.book(x)
    z1 = T.tanh(L47.lin(u, m.l1, False))
    B, Lx = x.shape
    gi_if = L47.lin(z1, arch26.Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0), False).reshape(B, Lx, 3, L47.PATCH * L47.PATCH)
    acc = {g: dict(r_ext=0, z_ext=0, n=0, abs_state=0.0) for g in GROUPS}
    h = T.zeros(B, L.C); worst = 0.0
    for t in range(Lx):
        G = T.zeros(B, 3, L.C).index_copy(2, L.ifx, gi_if[:, t])
        G = G.index_copy(2, L.farx, L.bfar[None].expand(B, -1, -1))
        H = T.einsum("bik,gik->bgi", h[:, L.idx], L.Wc) + L.bc
        r = T.sigmoid(G[:, 0] + H[:, 0]); z = T.sigmoid(G[:, 1] + H[:, 1]); n = T.tanh(G[:, 2] + r * H[:, 2])
        hn = (1 - z) * n + z * h
        d = float((hn - hs[t]).abs().max()); worst = max(worst, d)
        assert d <= tol, (t, d)
        for k, g in enumerate(GROUPS):
            sel = ring == k
            rr, zz = r[:, sel], z[:, sel]
            acc[g]["r_ext"] += int(((rr < 0.05) | (rr > 0.95)).sum()); acc[g]["z_ext"] += int(((zz < 0.05) | (zz > 0.95)).sum())
            acc[g]["n"] += rr.numel(); acc[g]["abs_state"] += float(hs[t][:, sel].abs().sum())
        h = hs[t]
    out = {g: dict(reset_share_extreme=a["r_ext"] / a["n"], update_share_extreme=a["z_ext"] / a["n"], mean_abs_state=a["abs_state"] / a["n"])
           for g, a in acc.items()}
    out["recomputed_state_max_abs_diff"] = worst
    return out


# ---------------------------------------------------------------- change 4: names
def receipt_path(out, arm, rule, seed, dose):
    tag = "" if dose == DOSE else f"_dry{dose}"
    return Path(out) / f"e53_{SIDE}_{arm}_{rule}{tag}_{seed}.json"


# ---------------------------------------------------------------- change 5: provenance
def provenance(cmd):
    p = X47.provenance(cmd)
    p["e53_runner_sha256"] = X16.sha_file(HERE / "exp53_local_start.py")
    p["numpy"] = numpy.__version__
    return p


# ---------------------------------------------------------------- one run (a copy of exp47_local.run with changes 1 to 8)
def run(arm, rule, seed, out, cmd, dose=None, verbose=True, resume=False, stop_at=None):
    """resume=True continues from the run's checkpoint (written every CKPT_EVERY updates, after the row's scoring); stop_at ends the
    process after that update's checkpoint."""
    mult, which, decay = ARMS[arm]
    local = RULES[rule]
    dose = dose or DOSE
    assert dose % ROW == 0
    rp = receipt_path(out, arm, rule, seed, dose)
    paths = dict(receipt=rp, start=rp.with_name(rp.stem + "_start.pt"), weights=rp.with_suffix(".pt"), answers=rp.with_name(rp.stem + "_answers.json"))
    ckpt = rp.with_name(rp.stem + "_ckpt.pt")
    resume = resume and ckpt.exists()
    if not resume:
        R_.refuse_existing(*paths.values(), str(paths["weights"]) + ".tmp", str(paths["start"]) + ".tmp", ckpt)
    else:
        R_.refuse_existing(paths["weights"], paths["answers"])
    Path(out).mkdir(parents=True, exist_ok=True)
    m, L = build(SIDE, NET_ARM, seed)
    assert R_.tensors_sha(weights(m, L)) == START_SHA
    assert dict(sources=tsha(L.idx), periods=tsha(L.p), phases=tsha(L.f)) == STRUCT_SHA
    n_mult = multiply(L, arm)
    install(L)
    opt = make_opt(m, L)
    set_decay(opt, decay)
    assert_decay(opt, decay)
    cnt = L.counts(m)
    assert cnt["weights"] == X47.WEIGHT_COUNT[(SIDE, NET_ARM)], (SIDE, NET_ARM, cnt)
    ring = rings(L)
    fb_x = R42.first_batch(seed)[0]
    A, Rs, TA, LP = R42.panels(seed)
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    bt = E.batches(tp, seed, E.KMAX["M"]); dg = C26.Digest()
    if resume:
        start_sha = R_.tensors_sha({k: v for k, v in T.load(paths["start"]).items() if k not in ("sources", "periods", "phases", "interface", "far")})
    else:
        start_sha = save_weights(m, L, paths["start"])
    rec = dict(experiment="E53", kind="birth", side=SIDE, cells=L.C, arm=arm, network_arm=NET_ARM, multiple=mult, entries_multiplied=which,
               entries_multiplied_count=n_mult, weight_decay=decay, rule=rule, local=local, seed=seed, dose=dose, row_every=ROW, N=N,
               interface_cells=len(L.ifx), far_cells=len(L.farx), cells_by_period={str(q): int((L.p == q).sum()) for q in sorted(set(L.p.tolist()))},
               start="E47's start (Local(m, seed, 32, 10, 'one')) with the named entries of Wc multiplied in single precision",
               batch=X.BATCH, kmax=E.KMAX["M"], weight_count=cnt["weights"], multiply_adds=cnt["multiply_adds"],
               multiply_adds_with_periods=cnt["multiply_adds_with_periods"], optimizer=opt_record(m, L, opt, decay),
               update="exp26_candidate.step's text with exp47_local.forward(m, L, x, local) in place of arch26.forward",
               data="exp11_cell.batches([(t, split(t, .5, seed)[0]) for t in DEV], seed, 12)",
               forward="exp47_local.forward (only the updated cells; check 3 ties it to e47_local.forward)",
               scores="exp42_cells.score_all (plain); with the far cells at zero and at their mean: A, R, length test at 12 and 40",
               structure_sha256=dict(sources=tsha(L.idx), periods=tsha(L.p), phases=tsha(L.f)),
               start_weights=paths["start"].name, start_tensors_sha256=start_sha, start_file_sha256=X16.sha_file(paths["start"]),
               provenance=provenance(cmd), complete=False, curve=[])
    curve = rec["curve"]; cpu_up = cpu_sc = 0.0; w0 = time.time(); losses = []; first = None; u0 = 0
    if resume:
        st = T.load(ckpt, weights_only=False)
        u0 = st["u"]
        m.load_state_dict(st["module"])
        with T.no_grad():
            for k, v in st["layer"].items():
                getattr(L, k).copy_(v)
            if L.a is not None:
                L.a.copy_(st["a"])
        opt.load_state_dict(st["opt"])
        assert_decay(opt, decay)
        prev = st["rec"]
        assert prev["start_tensors_sha256"] == start_sha and prev["structure_sha256"] == rec["structure_sha256"]
        resumes = prev.get("resumes", []) + [dict(at_update=u0, provenance=rec["provenance"])]
        rec = prev; rec["resumes"] = resumes; curve = rec["curve"]
        cpu_up, cpu_sc, first = st["cpu_up"], st["cpu_sc"], st["first"]
        for _ in range(u0):                                                  # the data stream replayed to the same point, into the digest
            dg.add(next(bt))
        assert curve[-1]["step"] == u0 and curve[-1]["data_digest"] == dg.value(), "replayed stream differs from the receipt"
        print(f"e53 resumed {rp.name} at update {u0}", flush=True)

    def score_row(u, last=False):
        nonlocal cpu_sc
        c = time.process_time()
        sc, diag, ans = score_row_all(m, L, A, Rs, TA, LP, answers=last)
        dt = time.process_time() - c; cpu_sc += dt
        row = dict(step=u, **sc, far_off=diag["off"] if diag else None, far_mean=diag["mean"] if diag else None, magnitudes=magnitudes(L),
                   ring_magnitudes=ring_magnitudes(L, ring), gates=gates(m, L, ring, fb_x),
                   cpu_updates=cpu_up, cpu_scoring=cpu_sc, cpu_this_row=dt, weight_items=cnt["weights"] * X.BATCH * u,
                   multiply_add_items=cnt["multiply_adds_with_periods"] * u, data_digest=dg.value(),
                   loss=(sum(losses) / len(losses)) if losses else None, finite_losses=all(v == v and abs(v) != float("inf") for v in losses),
                   wall=time.time() - w0)
        curve.append(row)
        if verbose:
            d = (f" | off A {diag['off']['A']:.1f} R {diag['off']['R']:.1f} 40 {diag['off']['len40']:.1f}"
                 f" | mean A {diag['mean']['A']:.1f} R {diag['mean']['R']:.1f} 40 {diag['mean']['len40']:.1f}") if diag else ""
            print(f"e53 {SIDE} {arm} {rule} {seed} {u}: A {row['A']:.2f} R {row['R']:.2f} "
                  f"taught {row['taught']:.2f} len12 {row['len12']:.1f} len40 {row['len40']:.1f}{d} cpu {cpu_up:.0f}+{cpu_sc:.0f}s", flush=True)
        return ans

    if not resume:
        score_row(0)
    for u in range(u0 + 1, dose + 1):
        items = next(bt)
        if first is None:
            first = items
            rec["cosine_first_batch_start"] = cosines(m, L, first)
        dg.add(items)
        c = time.process_time(); losses.append(step(m, L, opt, items, local)); cpu_up += time.process_time() - c
        if u % ROW == 0:
            ans = score_row(u, last=(u == dose)); losses = []
            if dose == DOSE and u in SAVE_ROWS:
                wp = rp.with_name(f"{rp.stem}_u{u}.pt")
                rec.setdefault("saved_weights", {})[str(u)] = dict(file=wp.name, tensors_sha256=save_weights(m, L, wp), file_sha256=None)
                rec["saved_weights"][str(u)]["file_sha256"] = X16.sha_file(wp)
            if u % 4096 == 0 or u == dose:
                rp.write_text(json.dumps(rec, indent=1))
            if u % CKPT_EVERY == 0 and u < dose:
                save_ckpt(ckpt, m, L, opt, u, rec, cpu_up, cpu_sc, first)
                if stop_at == u:
                    return None
    rec["cosine_first_batch_end"] = cosines(m, L, first)
    sha = save_weights(m, L, paths["weights"])
    paths["answers"].write_text(json.dumps(dict(update=dose, panel="length test (add89, code p), 300 items at 12 and 40 letters",
                                                prompts={str(k): [p for p, _ in it] for k, it in LP.items()},
                                                targets={str(k): [t for _, t in it] for k, it in LP.items()}, answers=ans)))
    rec.update(complete=True, weights=paths["weights"].name, weights_sha256=X16.sha_file(paths["weights"]), tensors_sha256=sha,
               answers_file=paths["answers"].name, answers_sha256=X16.sha_file(paths["answers"]),
               module_recurrent_untouched=dict(weight_hh_grad_none=m.rnn.weight_hh_l0.grad is None, bias_hh_grad_none=m.rnn.bias_hh_l0.grad is None),
               optimizer_steps=R_.opt_steps(opt), optimizer_read_back=opt_record(m, L, opt, decay), data_digest=dg.value(),
               cpu_updates=cpu_up, cpu_scoring=cpu_sc, cpu_per_update=cpu_up / dose, wall=time.time() - w0,
               processes_beside=os.environ.get("E53_BESIDE", "not recorded"))
    rp.write_text(json.dumps(rec, indent=1))
    if ckpt.exists():
        ckpt.unlink()
    print("e53 complete", rp.name, sha[:16], flush=True)
    return rec


# ---------------------------------------------------------------- checks
def trial_file(rel):
    """A file of the trial, read with git show into a folder that is not committed."""
    p = TRIAL / Path(rel).name
    if not p.exists():
        TRIAL.mkdir(parents=True, exist_ok=True)
        b = subprocess.run(["git", "show", f"{W7}:MOD97_LOCAL_RULE_20261106/{rel}"], cwd=HERE, capture_output=True, check=True).stdout
        p.write_bytes(b)
    return p


ROW_FIELDS = ("loss", "A_counts", "R_counts", "taught_counts", "length_counts", "magnitudes", "data_digest")


def check1(qual, cmd):
    out = {}
    for rule in ("S", "B"):
        rec = run("one", rule, SEED, qual / "check1_runs", cmd, dose=1024, verbose=False)
        tri = json.loads(trial_file(f"exp45/e47/e47_32_one_{rule}_{SEED}.json").read_text())
        tr = {r["step"]: r for r in tri["curve"]}
        rows = {}
        for r in rec["curve"]:
            t = tr[r["step"]]
            d = {f: r[f] == t[f] for f in ROW_FIELDS}
            for mode in ("far_off", "far_mean"):
                d[mode] = all(r[mode][k] == t[mode][k] for k in ("A_counts", "R_counts", "length_counts"))
            rows[r["step"]] = d
        top = dict(start_tensors_sha256=rec["start_tensors_sha256"] == tri["start_tensors_sha256"],
                   structure_sha256=rec["structure_sha256"] == tri["structure_sha256"],
                   cosine_first_batch_start=rec["cosine_first_batch_start"] == tri["cosine_first_batch_start"])
        out[rule] = dict(rows=rows, receipt=top, steps=sorted(rows),
                         ok=sorted(rows) == [0, 256, 512, 768, 1024] and all(all(v.values()) for v in rows.values()) and all(top.values()))
    out["pass"] = all(v["ok"] for v in out.values())
    return out


CHECK2_ARMS = ("x3", "wd0", "x3-wd0", "xf3")


def check2(qual, cmd):
    out = {}
    trial = T.load(trial_file(f"exp45/e47/e47_32_one_S_{SEED}_start.pt"))
    for arm in CHECK2_ARMS:
        rec = run(arm, "S", SEED, qual / "check2_runs", cmd, dose=256, verbose=False)
        st = T.load(qual / "check2_runs" / rec["start_weights"])
        keys_equal = sorted(st) == sorted(trial)
        others = {k: bool(T.equal(st[k], trial[k])) for k in trial if k != "Wc"}
        W, W0 = st["Wc"], trial["Wc"]
        if arm == "wd0":
            wc = dict(equal_to_trial=bool(T.equal(W, W0)))
            ok = wc["equal_to_trial"]
        elif arm in ("x3", "x3-wd0"):
            wc = dict(equal_to_trial_times_3=bool(T.equal(W, W0 * 3)))
            ok = wc["equal_to_trial_times_3"]
        else:
            ifc = T.zeros(32 * 32, dtype=T.bool); ifc[st["interface"]] = True
            among = ifc[:, None] & ifc[st["sources"]]                               # [i, k]: cell i and its source k both interface cells
            among3 = among[None].expand(3, -1, -1)
            times3 = W == W0 * 3; same = W == W0
            wc = dict(entries_times_3=int(times3.sum()), entries_same=int(same.sum()), among_interface=int(among3.sum()),
                      times3_outside_among=bool(times3[~among3].all()), same_inside_among=bool(same[among3].all()))
            ok = (wc["among_interface"] == 6867 and wc["times3_outside_among"] and wc["same_inside_among"]
                  and int((times3 & ~among3).sum()) == 26925)
            wc["times3_outside_among_count"] = int((times3 & ~among3).sum())
        out[arm] = dict(keys_equal=keys_equal, others_bit_equal=others, Wc=wc, receipt=rec["start_weights"],
                        decay_start=[g["weight_decay"] for g in rec["optimizer"]["groups"]],
                        decay_end=[g["weight_decay"] for g in rec["optimizer_read_back"]["groups"]],
                        ok=keys_equal and all(others.values()) and ok)
    out["pass"] = all(v["ok"] for v in out.values())
    return out


def check3(qual, cmd):
    out = dict(a={}, b={})
    want = {"x3": [0.25, 0.0], "xf3": [0.25, 0.0], "wd0": [0.0, 0.0], "x3-wd0": [0.0, 0.0]}
    for arm in CHECK2_ARMS:
        rec = json.loads(receipt_path(qual / "check2_runs", arm, "S", SEED, 256).read_text())
        s = [g["weight_decay"] for g in rec["optimizer"]["groups"]]; e = [g["weight_decay"] for g in rec["optimizer_read_back"]["groups"]]
        out["a"][arm] = dict(start=s, end=e, ok=s == want[arm] and e == want[arm])
    x, y, mask = R42.first_batch(SEED)
    for rule in ("S", "B"):
        res = {}
        for decay in (0.25, 0.0):
            m, L = build(SIDE, NET_ARM, SEED)
            install(L)
            opt = make_opt(m, L); set_decay(opt, decay); assert_decay(opt, decay)
            nm = X47.names(m, L)
            group0 = [(nm[id(p)], p) for p in opt.param_groups[0]["params"]]
            start = {n: p.detach().clone() for n, p in group0}
            logits = forward(m, L, x, RULES[rule])[0]
            loss = E.cross_entropy(logits[mask], y[mask]); opt.zero_grad(); loss.backward(); opt.step()
            res[decay] = dict(after={n: p.detach().clone() for n, p in group0}, start=start, book=m.book.weight.detach().clone())
        per = {}
        for n, p0 in res[0.0]["start"].items():
            assert T.equal(p0, res[0.25]["start"][n])
            dev = (res[0.0]["after"][n].double() - res[0.25]["after"][n].double() - 0.00025 * p0.double()).abs().max()
            per[n] = dict(max_abs_deviation=float(dev), deviation_if_decay_not_removed=float((0.00025 * p0.double()).abs().max()))
        book_equal = bool(T.equal(res[0.0]["book"], res[0.25]["book"]))
        out["b"][rule] = dict(tensors=per, n_tensors=len(per), worst=max(v["max_abs_deviation"] for v in per.values()), book_bit_equal=book_equal,
                              ok=len(per) == 13 and book_equal and all(v["max_abs_deviation"] < 1e-6 for v in per.values()))
    out["pass"] = all(v["ok"] for v in out["a"].values()) and all(v["ok"] for v in out["b"].values())
    return out


def check4(qual, cmd):
    """x3-wd0 under the sweep, 2,048 updates: once without interruption; once killed (SIGKILL) after its log shows the row of update 1,536
    and resumed from the saved state of update 1,024."""
    plain_dir, killed_dir = qual / "check4_runs" / "plain", qual / "check4_runs" / "killed"
    a = run("x3-wd0", "S", SEED, plain_dir, cmd, dose=2048, verbose=False)
    killed_dir.mkdir(parents=True, exist_ok=True)
    log = killed_dir / "child.log"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_"))}   # as from a shell: the parent's import set these
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    args = [sys.executable, str(HERE / "exp53_local_start.py"), "run", "x3-wd0", "S", str(SEED), "--dose", "2048", "--out", str(killed_dir), "--resume"]
    with open(log, "w") as f:
        p = subprocess.Popen(args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env)
    killed_at = None
    while p.poll() is None:
        if " 1536: " in log.read_text():
            os.kill(p.pid, signal.SIGKILL); p.wait(); killed_at = "after the row of update 1,536"
            break
        time.sleep(0.5)
    assert killed_at is not None, "the child ended before it could be killed"
    rp = receipt_path(killed_dir, "x3-wd0", "S", SEED, 2048)
    ck = rp.with_name(rp.stem + "_ckpt.pt")
    st = T.load(ck, weights_only=False)
    receipt_before = rp.exists()
    with open(log, "a") as f:
        p2 = subprocess.run(args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=env)
    b = json.loads(rp.read_text())
    strip = lambda c: [{k: v for k, v in r.items() if not k.startswith(("cpu", "wall"))} for r in c]
    skip_top = ("curve", "provenance", "resumes")
    top_a = {k: v for k, v in a.items() if not k.startswith(("cpu", "wall")) and k not in skip_top}
    top_b = {k: v for k, v in b.items() if not k.startswith(("cpu", "wall")) and k not in skip_top}
    out = dict(killed=killed_at, saved_state_update=st["u"], receipt_existed_when_killed=receipt_before, resume_returncode=p2.returncode,
               resumes=b.get("resumes") and [r["at_update"] for r in b["resumes"]],
               rows_equal=strip(a["curve"]) == strip(b["curve"]), n_rows=len(b["curve"]),
               top_fields_equal=top_a == top_b, top_fields_differing=sorted(k for k in set(top_a) | set(top_b) if top_a.get(k) != top_b.get(k)),
               tensors_equal=a["tensors_sha256"] == b["tensors_sha256"], start_equal=a["start_tensors_sha256"] == b["start_tensors_sha256"],
               digest_equal=a["data_digest"] == b["data_digest"], cosine_end_equal=a["cosine_first_batch_end"] == b["cosine_first_batch_end"],
               decay_after_resume=[g["weight_decay"] for g in b["optimizer_read_back"]["groups"]])
    out["pass"] = (out["saved_state_update"] == 1024 and out["resume_returncode"] == 0 and out["resumes"] == [1024] and out["rows_equal"]
                   and out["top_fields_equal"] and out["tensors_equal"] and out["start_equal"] and out["digest_equal"] and out["cosine_end_equal"]
                   and out["decay_after_resume"] == [0.0, 0.0])
    return out


def check5(qual, cmd):
    x, y, mask = R42.first_batch(SEED)
    out = {}
    for arm in ("x3", "xf3"):
        for rule in ("S", "B"):
            m, L = build(SIDE, NET_ARM, SEED)
            multiply(L, arm)
            install(L)
            ring = rings(L)
            ps = [(n, q) for n, q in m.named_parameters()] + [("Wc", L.Wc), ("bc", L.bc), ("bfar", L.bfar)]
            for _, q in ps:
                q.grad = None
            logits = forward(m, L, x, RULES[rule])[0]
            loss = E.cross_entropy(logits[mask], y[mask]); loss.backward()
            grads_finite = {n: (None if q.grad is None else bool(T.isfinite(q.grad).all())) for n, q in ps}
            g = gates(m, L, ring, x)
            shares = {k: max(g[k]["reset_share_extreme"], g[k]["update_share_extreme"]) for k in GROUPS}
            out[f"{arm}_{rule}"] = dict(loss=float(loss), loss_finite=bool(T.isfinite(loss)), grads_finite=grads_finite, gates=g,
                                        largest_extreme_share=shares,
                                        ok=bool(T.isfinite(loss)) and all(v is not False for v in grads_finite.values()) and all(s < 0.01 for s in shares.values()))
    out["pass"] = all(v["ok"] for v in out.values())
    return out


def check6(qual, cmd):
    res = {}
    for arm in CHECK2_ARMS:
        rec = json.loads(receipt_path(qual / "check2_runs", arm, "S", SEED, 256).read_text())
        rows = rec["curve"]
        res[arm] = dict(cpu_per_update=rec["cpu_updates"] / 256, cpu_per_row=rec["cpu_scoring"] / len(rows), rows=len(rows))
    up = max(v["cpu_per_update"] for v in res.values()); row = max(v["cpu_per_row"] for v in res.values())
    n_rows = DOSE // ROW + 1
    per_run = up * DOSE + row * n_rows
    fc = dict(cpu_per_run_hours=per_run / 3600, six_runs_hours=6 * per_run / 3600, conditional_run_hours=per_run / 3600,
              basis="the largest CPU-seconds an update and a scored row of the four runs of check 2 (sweep, 256 updates), times 24,576 updates "
                    "and 97 rows; runs under back-propagation are taken to cost the same", limit_hours=30)
    trial = json.loads(trial_file(f"exp45/e47/e47_32_one_S_{SEED}.json").read_text())
    fc["trial_S_cpu_hours"] = (trial["cpu_updates"] + trial["cpu_scoring"]) / 3600
    tb = json.loads(trial_file(f"exp45/e47/e47_32_one_B_{SEED}.json").read_text())
    fc["trial_B_cpu_hours"] = (tb["cpu_updates"] + tb["cpu_scoring"]) / 3600
    return {"runs": res, "forecast": fc, "forecast_above_30_hours": fc["six_runs_hours"] > 30, "pass": True, "criterion": "none (the brief's check 6)"}


CHECKS = {"1": check1, "2": check2, "3": check3, "4": check4, "5": check5, "6": check6}


# ---------------------------------------------------------------- main (change 8)
def main(argv):
    cmd = "python3 " + " ".join(argv)
    a = argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    if a[0] == "run":
        arm, rule, seed = a[1], a[2], int(a[3])
        assert arm in ARMS and rule in RULES and seed == SEED
        d = opt("--dose", None)
        run(arm, rule, seed, Path(opt("--out", str(OUT))), cmd, dose=int(d) if d else None, resume="--resume" in a)
    elif a[0] == "check":
        n = a[1]
        QUAL.mkdir(parents=True, exist_ok=True)
        p = QUAL / f"check{n}.json"
        R_.refuse_existing(p)
        t0 = time.process_time(); w0 = time.time()
        res = CHECKS[n](QUAL, cmd)
        res.update(check=int(n), cpu=time.process_time() - t0, wall=time.time() - w0, provenance=provenance(cmd))
        p.write_text(json.dumps(res, indent=1, default=str))
        print(f"e53 check {n}:", "PASS" if res["pass"] else "FAIL", flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
