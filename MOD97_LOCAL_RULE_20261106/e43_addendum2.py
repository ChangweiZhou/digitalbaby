"""E43 addendum 2 (OPUS_TASK_E43_IS_A_FAMILY_LOST_TO_UNRELATED_LEARNING_20261116.md, addendum of 16 November): two further additions.
A new file; nothing existing is edited. e42_cells.py and exp42_cells.py are byte-identical copies from claude/gallant-dijkstra-i9zgsr at
534df8f0 (the hashes recorded in E42's receipts of seed 73129), imported, not rewritten.

    rehearse   unrelated-rehearse (no decay) and unrelated-rehearse-decay (0.25), the six E26 births: exp43_unrelated's unrelated arm with each
               item drawn from one of two streams with equal chance by Random(f"e43|{seed}|joint").randrange(2): 0 = the unrelated stream
               (e43_task.stream(seed)), 1 = the first family's taught half as E38's rehearsal stream,
               X16.entry_stream([("op", t, a, b) for t in DEV for a, b in split(t, .5, seed)[0]], seed + 100). Batches of 64 consecutive items.
    e42        the unrelated arm from E42's final F10 or P8 network of seed 73129 under its rule: E42's forward (exp42_cells.forward with the
               run's layer: mask, periods, phases and added parameters as saved), exp42_cells.step's text, a fresh exp42_cells.make_opt with
               group 0's weight_decay set to 0 (no decay); no replacement event.

Rows 0, 64, 256, 1,024, 2,048, 4,096; scores 1 to 4 and 6 as exp43_unrelated (score 6 with the run's forward); weights committed at 4,096.

    python3 e43_addendum2.py rehearse <unrelated-rehearse|unrelated-rehearse-decay> <S|B> <seed>
    python3 e43_addendum2.py e42 <F10|P8> <S|B>
"""
import json
import sys
import time
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp43_unrelated as U          # noqa: E402
import exp42_cells as X42            # noqa: E402  (byte-identical copy, E42's runner)

E38, C, R, X16, E, X, arch26, T, TK = U.E38, U.C, U.R, U.X16, U.E, U.X, U.arch26, U.T, U.TK
OUT = U.OUT43 / "addendum2"
E42DIR = OUT / "e42_start"
E42_COMMIT = "534df8f0"
OWN = ("e43_addendum2.py", "exp43_unrelated.py", "e43_task.py", "exp38_erase.py", "exp42_cells.py", "e42_cells.py")
DRY = [None, None]          # a dry run: (updates, output folder), set by --updates / --out
REH = {"unrelated-rehearse": 0.0, "unrelated-rehearse-decay": 0.25}


def provenance(cmd):
    p = U.provenance(cmd)
    p["addendum2_sources_sha256"] = {s: X16.sha_file(HERE / s) for s in OWN}
    import subprocess
    p["git"]["addendum2_sources_modified_vs_commit"] = subprocess.run(["git", "status", "--porcelain", "--", *OWN], cwd=HERE, capture_output=True,
                                                                      text=True).stdout.strip().splitlines()
    return p


@T.no_grad()
def eq_states(fwd, items):
    groups, res = {}, [None] * len(items)
    for i, (p, _) in enumerate(items):
        groups.setdefault(len(p), []).append(i)
    for L, idx in groups.items():
        for j in range(0, len(idx), 256):
            ids = idx[j:j + 256]
            hs = fwd(T.tensor([[E.BOS, *items[i][0]] for i in ids]))
            for q, i in enumerate(ids):
                res[i] = hs[-1][q].clone()
    return T.stack(res)


def loop(tag, m, bsd, opt, step_fn, nxt, fwd_states, settings, cmd, extra_params=None):
    dose = DRY[0] or U.DOSE; out = DRY[1] or OUT; tag = tag + (f"_dry{dose}" if DRY[0] else "")
    """The unrelated arm's rows and scores (exp43_unrelated.run's), for any network, step and stream."""
    seed = settings["seed"]
    out.mkdir(parents=True, exist_ok=True); (out / "rows_uncommitted").mkdir(exist_ok=True)
    path = out / f"{tag}.json"; R.refuse_existing(path)
    P, comp = X16.panels(seed)
    held = {n: P[n][12] for n in ("add89", "sub89", "max89", "min89")}
    s1 = U.first_family_items(); s2_items = [it for n in held for it in held[n]]
    new_panels = {"held": TK.panel(seed, "held"), "taught": TK.panel(seed, "taught")}
    dg = C.Digest(); h0 = fwd_states(s2_items); h0_mean = float(h0.abs().mean())
    t0 = time.process_time(); cpu_score = [0.0]; rows = []; losses = []; counts = {"unrelated": 0, "first_family": 0}

    def score_row(g):
        cb = time.process_time()
        r = dict(updates=g, data_digest=dg.value(), items_so_far=dict(counts), loss_since_last_row=(sum(losses) / len(losses)) if losses else None,
                 finite_losses=all(v == v and abs(v) != float("inf") for v in losses))
        r["score1_first_family_all_pairs_k0"] = U.score1(m, s1)
        r["score2_first_family_heldout_k12"] = {n: X16.score(m, it, None) for n, it in held.items()}
        r["score3_new_task"] = {n: X16.score(m, it, None) for n, it in new_panels.items()}
        r["score4_movement"] = U.movement(m, bsd)
        if extra_params:
            r["score4_added_parameters"] = {n: dict(rel_diff=float((p.detach() - bsd[n]).norm() / bsd[n].norm())) for n, p in extra_params.items()}
        h = fwd_states(s2_items)
        r["score6_state_change"] = dict(mean_abs_change=float((h - h0).abs().mean()), mean_abs_state_row0=h0_mean, ratio=float((h - h0).abs().mean()) / h0_mean)
        cpu_score[0] += time.process_time() - cb
        r["tensors_sha256"] = R.tensors_sha(m.state_dict())
        wp = (out if g == U.DOSE else out / "rows_uncommitted") / f"{tag}_row{g}.pt"
        r["weights"] = str(wp.relative_to(out)); r["weights_sha256"] = R.save_weights(m, wp)
        r["cpu"] = time.process_time() - t0
        rows.append(r); losses.clear()
        s = r["score1_first_family_all_pairs_k0"]; pc = lambda d: round(100 * d["exact"] / d["n"], 1)
        print(tag, "row", g, "reduction", round(s["reduction_score"], 1), "other", round(s["add89"]["other"]["pct"], 1), round(s["sub89"]["other"]["pct"], 1),
              "max/min", round(s["max89"]["all"]["pct"], 1), round(s["min89"]["all"]["pct"], 1),
              "held", {n: pc(v) for n, v in r["score2_first_family_heldout_k12"].items()}, "new", {n: pc(v) for n, v in r["score3_new_task"].items()},
              "items", counts, f"cpu {r['cpu']:.0f}s", flush=True)
        return r

    def write(complete=None):
        path.write_text(json.dumps(dict(experiment="E43 addendum 2", kind="run", tag=tag, settings=settings, rows=rows, complete=complete), indent=1))

    settings["provenance"] = provenance(cmd)
    score_row(0); write(); g = 0
    while g < dose:
        batch = []
        for _ in range(X.BATCH):
            kind, it = nxt()
            counts[kind] += 1
            p_ = it[0]
            assert all(b >= 128 for b in p_) if kind == "unrelated" else all(b < 128 for b in p_)      # check 4, both kinds
            batch.append(it)
        dg.add(batch)
        losses.append(step_fn(batch))
        g += 1
        if g in U.ROWS:
            score_row(g); write()
    write(dict(updates=g, items=counts, tensors_sha256=R.tensors_sha(m.state_dict()), data_digest=dg.value(), cpu_total=time.process_time() - t0,
               cpu_scoring=cpu_score[0]))
    print(tag, "complete", flush=True)


def rehearse(arm, rule, seed, cmd):
    local = C.ARMS[rule]
    arch26.install()
    bfile = C.birth_path(E38.E26DIR, rule, seed); m, birth_sha, admission = E38.admit(bfile, rule, seed)
    bsd = T.load(bfile); m.load_state_dict(bsd)
    opt = C.make_opt(m); opt.param_groups[0]["weight_decay"] = REH[arm]
    unrel = TK.stream(seed)
    fam = X16.entry_stream([("op", t, a, b) for t in E.DEV for a, b in E.split(t, .5, seed)[0]], seed + 100)
    rj = Random(f"e43|{seed}|joint")
    nxt = lambda: ("unrelated", next(unrel)) if rj.randrange(2) == 0 else ("first_family", next(fam))
    settings = dict(arm=arm, rule=rule, local=local, seed=seed, decay=REH[arm], optimizer_groups_read_back=R.opt_groups(m, opt),
                    birth=str(bfile.relative_to(HERE)), birth_sha256=birth_sha, birth_tensors_sha256=R.tensors_sha(m.state_dict()), birth_admission=admission,
                    stream="Random(f'e43|{seed}|joint').randrange(2): 0 -> e43_task.stream(seed), 1 -> X16.entry_stream(first family taught half, seed + 100)")
    loop(f"e43_{arm}_{rule}_{seed}", m, bsd, opt, lambda b: E38.step(m, opt, b, local), nxt,
         lambda items: eq_states(lambda x: arch26.forward(m, x, False, True)[1]["h"], items), settings, cmd)


def e42(config, rule, cmd):
    seed = 73129; local = C.ARMS[rule]
    base = E42DIR / f"e42_{config}_{rule}_{seed}"
    rec = json.loads(base.with_suffix(".json").read_text())
    sha = X16.sha_file(base.with_suffix(".pt")); assert sha == rec["weights_sha256"] and rec["complete"] is True
    full = T.load(base.with_suffix(".pt"))
    m, layer = X42.build(config, seed)
    X42.install(layer)
    sd = {k: v for k, v in full.items() if k in dict(m.named_parameters()) or k in m.state_dict()}
    m.load_state_dict(sd)
    assert T.equal(layer.p, full["periods"]) and T.equal(layer.f, full["phases"]), "periods or phases differ from E42's"
    layer.mask = full["mask"].clone(); layer.made = full["made"].clone()
    extra = {}
    if layer.a is not None:
        with T.no_grad():
            layer.a.copy_(full["a"])
        extra["a"] = layer.a
    assert layer.Wx is None
    assert R.tensors_sha(m.state_dict()) == rec["tensors_sha256"], "loaded network is not E42's"
    assert X42.tsha(layer.mask) == rec["structure_end_sha256"]["mask"] and X42.tsha(layer.made) == rec["structure_end_sha256"]["made"]
    assert all(X42.check_limit(m, layer).values())
    if extra:
        assert X42.tsha(layer.a) == rec["added_parameters_sha256"]["a"]
    # row-0 check: E42's own scores (exp42_cells.score_all) equal its receipt's final row
    A, Rs, TA, LP = X42.panels(seed)
    s0 = X42.score_all(m, A, Rs, TA, LP); last = rec["curve"][-1]
    row0_check = {k: s0[k] == last[k] for k in ("A_counts", "R_counts", "taught_counts", "length_counts")}
    assert all(row0_check.values()), f"row 0 differs from E42's final row: {row0_check}"
    opt = X42.make_opt(m, layer)
    g0 = [g for g in opt.param_groups if g["weight_decay"] == C.WD]; assert len(g0) == 1; g0[0]["weight_decay"] = 0.0
    bsd = {k: v.clone() for k, v in m.state_dict().items()}
    if extra:
        bsd["a"] = layer.a.detach().clone()
    unrel = TK.stream(seed)
    nxt = lambda: ("unrelated", next(unrel))
    settings = dict(arm="unrelated", network=f"E42 {config}", rule=rule, local=local, seed=seed, decay=0.0, replacement_events=False,
                    start=dict(file=f"MOD97_LOCAL_RULE_20261106/exp45/e42/e42_{config}_{rule}_{seed}.pt at {E42_COMMIT} on claude/gallant-dijkstra-i9zgsr",
                               weights_sha256=sha, tensors_sha256=rec["tensors_sha256"], structure_end_sha256=rec["structure_end_sha256"],
                               theta=rec.get("theta")),
                    row0_against_e42_final_row=dict(check=row0_check, e42=dict(A=last["A"], R=last["R"], taught=last["taught"], len12=last["len12"], len40=last["len40"]),
                                                    here={k: s0[k] for k in ("A", "R", "taught", "len12", "len40")}),
                    optimizer=X42.opt_record(m, layer, opt), weight_count=X42.weight_count(m, layer))
    loop(f"e43_unrelated-e42{config}_{rule}_{seed}", m, bsd, opt, lambda b: X42.step(m, layer, opt, b, local), nxt,
         lambda items: eq_states(lambda x: X42.forward(m, layer, x, False)[1], items), settings, cmd, extra_params=extra or None)


if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])
    if "--updates" in a:
        DRY[0] = int(a[a.index("--updates") + 1]); DRY[1] = Path(a[a.index("--out") + 1])
    if a[0] == "rehearse":
        assert a[1] in REH; rehearse(a[1], a[2], int(a[3]), cmd)
    elif a[0] == "e42":
        assert a[1] in ("F10", "P8"); e42(a[1], a[2], cmd)
    else:
        raise SystemExit(__doc__)
