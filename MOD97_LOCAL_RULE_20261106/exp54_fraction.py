"""E54 runner (brief TASK_E54_THE_LIMITED_AND_THE_PERIODIC_NETWORK_WITH_FEWER_PAIRS_20261117.md, revision 3): E42's configurations taught
a fraction of the pairs other than a half. This file is a copy of exp42_ckpt.py (branch claude/gallant-dijkstra-i9zgsr at 44f91edc) with
the changes that the brief lists under "The runner" (1 to 9) and the assertion of its check 5; exp42_cells.py is NOT edited.

run_ckpt() is exp42_cells.run's text with saving and resuming (see exp42_ckpt.py), and with:
  * the taught fraction f: the stream and the panels of A, of R and of taught pairs are E41's at that fraction; the length panel is drawn
    from the pairs held out at f; assertions at every start that the panels of untaught pairs hold no taught pair, that the taught panel
    holds only taught pairs, and that the length panel at 12 letters is the addition panel behind A;
  * exp41_scale.assert_taught on every batch (in the loop of updates and in the replay on resuming);
  * 49,152 updates by default; receipts e54_<config>_<rule>_f<50|25|15|10>[_dry<updates>]_<seed>.json under exp45/e54/;
  * the hash of the weights at update 12,288 (tensors_sha256_at["12288"]);
  * at every scored row, length_counts["12"] == A_counts["add89"].

usage: python3 exp54_fraction.py run <cfg> <S|B> <seed> --fraction <50|25|15|10> [--theta T] [--dose D] [--stop-after U] [--out DIR] [--ckdir DIR]
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))   # as exp42_cells.py does (python3 -I omits the script folder)
import exp42_cells as R          # imports the harness in the runner's order (exp26_candidate first)
from exp41_scale import item_key, taught_keys, assert_taught   # after exp42_cells; none of them uses exp41_scale.FRACS
import json
import os
import random
import sys
import time
from pathlib import Path
from random import Random

T = R.T
CK = 1024
HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e54"
CKDIR = OUT / "ckpt"
DOSE = 49152
FRACTIONS = (50, 25, 15, 10)


def receipt_path(out, config, rule, fp, seed, dose):
    tag = "" if dose == DOSE else f"_dry{dose}"
    return Path(out) / f"e54_{config}_{rule}_f{fp}{tag}_{seed}.json"


def _save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    T.save(obj, tmp)
    os.replace(tmp, path)


def run_ckpt(config, rule, seed, out, cmd, f, theta=None, dose=None, stop_after=None, ckdir=CKDIR):
    N, ps, variant, wide, changing = R.CONFIGS[config]
    local = R.RULES[rule]
    dose = dose or DOSE
    fp = round(f * 100)
    assert fp in FRACTIONS and f == fp / 100
    assert dose % R.ROW == 0 and CK % R.ROW == 0
    if changing:
        assert theta is not None, "a configuration with changing connections needs --theta (set once by check 10)"
    rp = receipt_path(out, config, rule, fp, seed, dose)
    paths = dict(receipt=rp, weights=rp.with_suffix(".pt"), structure=rp.with_name(rp.stem + "_structure.json"),
                 small=rp.with_name(rp.stem + "_recurrent.pt"), answers=rp.with_name(rp.stem + "_answers.json"))
    ckp = Path(ckdir) / (rp.stem + ".ckpt.pt")
    resume = ckp.exists()
    if resume:
        R.R_.refuse_existing(*[p for k, p in paths.items() if k != "receipt"], str(paths["weights"]) + ".tmp")
    else:
        R.R_.refuse_existing(*paths.values(), str(paths["weights"]) + ".tmp")
    Path(out).mkdir(parents=True, exist_ok=True)
    m, layer = R.build(config, seed)
    R.install(layer)
    opt = R.make_opt(m, layer)
    wc = R.weight_count(m, layer)
    assert wc == R.WEIGHT_COUNT[config], (config, wc)
    A, Rs, TA = R.panels(seed, f)[:3]                     # E41's panels at this fraction
    LP = R.E.panel(R.LEN_TASK, R.E.split(R.LEN_TASK, f, seed)[1], seed, ks=(12, 40), n=300)   # from the pairs held out at f
    tk = taught_keys(f, seed)
    for pan in list(A.values()) + list(Rs.values()) + list(LP.values()):
        for p, _ in pan:
            assert item_key(p) not in tk, f"a taught pair is in a panel of untaught pairs: {item_key(p)}"
    for pan in TA.values():
        assert_taught(pan, tk)
    assert LP[12] == A[R.E.NAME(*R.LEN_TASK)], "the length panel at 12 letters is not the addition panel behind A"
    bt = R.E.batches([(t, R.E.split(t, f, seed)[0]) for t in R.E.DEV], seed, R.E.KMAX["M"]); dg = R.C26.Digest()
    rr = Random(f"e42|replace|{seed}")
    struct0 = R.structure(layer)
    me = dict(wrapper="exp54_fraction.py", wrapper_sha256=R.X16.sha_file(HERE / "exp54_fraction.py"), every=CK, resumed=[])
    prov = R.provenance(cmd)
    prov["sources_sha256"].update({s: R.X16.sha_file(HERE / s) for s in ("exp54_fraction.py", "exp41_scale.py")})
    if not resume:
        rec = dict(experiment="E54", kind="birth", fraction=f, length_panel_is_addition_panel=True, config=config, rule=rule, local=local, seed=seed, dose=dose, row_every=R.ROW,
                   N=N, period_set=ps, variant=variant, wide_neighbourhood=wide, changing=changing, theta=theta if changing else None,
                   batch=R.X.BATCH, kmax=R.E.KMAX["M"], weight_count=wc, params_module=sum(q.numel() for q in m.parameters()),
                   added_parameters={k: (None if v is None else list(v.shape)) for k, v in (("a", layer.a), ("Wx", layer.Wx))},
                   optimizer=R.opt_record(m, layer, opt),
                   update="exp26_candidate.step's text with exp42_cells.forward(m, layer, x, local) in place of arch26.forward",
                   data=f"exp11_cell.batches([(t, split(t, {f}, seed)[0]) for t in DEV], seed, 12), E41's stream (exp41_scale.stream) for this fraction",
                   forward="exp42_cells.forward (the faster forward; check 9 ties it to e42_cells.forward)",
                   structure_start_sha256=dict(mask=R.tsha(layer.mask), made=R.tsha(layer.made)),
                   provenance=prov, complete=False, curve=[], checkpointing=me)
        cpu_up = cpu_sc = cpu_rep = 0.0; wall0 = 0.0; first = None; u0 = 0
    else:
        start_mask, start_made = layer.mask.clone(), layer.made.clone()      # the freshly built layer, before restoring
        start_made = R.tsha(start_made)
        st = T.load(ckp, weights_only=False)
        m.load_state_dict(st["model"])
        with T.no_grad():
            layer.mask.copy_(st["mask"]); layer.made.copy_(st["made"])
            if layer.a is not None:
                layer.a.copy_(st["a"])
            if layer.Wx is not None:
                layer.Wx.copy_(st["Wx"])
        opt.load_state_dict(st["opt"])
        rr.setstate(st["rr"]); T.set_rng_state(st["torch_rng"]); random.setstate(st["py_rng"])
        rec = st["rec"]; u0 = st["u"]; first = st["first"]
        cpu_up, cpu_sc, cpu_rep, wall0 = st["cpu_up"], st["cpu_sc"], st["cpu_rep"], st["wall"]
        assert (rec["config"], rec["rule"], rec["seed"], rec["dose"], rec["fraction"]) == (config, rule, seed, dose, f)
        assert rec["structure_start_sha256"] == dict(mask=R.tsha(start_mask), made=start_made), "checkpoint is not from this run's start"
        c = time.process_time()
        for _ in range(u0):                                   # replay the data stream to the checkpoint
            items = next(bt)
            assert_taught(items, tk)
            dg.add(items)
        assert dg.value() == st["digest"], "replayed data digest differs from the checkpoint's"
        rec["checkpointing"]["resumed"].append(dict(at_update=u0, cpu_replay=time.process_time() - c, when=time.strftime("%FT%TZ", time.gmtime()),
                                                     cmd=cmd))
        print(f"e54 resume {config} {rule} f{fp} {seed} at {u0}", flush=True)
    curve = rec["curve"]; w0 = time.time() - wall0; losses = []; n_rep_since = 0

    def score_row(u):
        nonlocal cpu_sc, n_rep_since
        c = time.process_time()
        row = dict(step=u, **R.score_all(m, A, Rs, TA, LP))
        cpu_sc += time.process_time() - c
        assert row["length_counts"]["12"] == row["A_counts"]["add89"], f"check 5 failed at row {u}: {row['length_counts']} {row['A_counts']}"
        lim = R.check_limit(m, layer)
        assert all(lim.values()), f"check 4 failed at row {u}: {lim}"
        row.update(replaced_since_previous_row=n_rep_since, limit=lim, cpu_updates=cpu_up, cpu_scoring=cpu_sc, cpu_replacement=cpu_rep,
                   weight_items=wc * R.X.BATCH * u, data_digest=dg.value(),
                   loss=(sum(losses) / len(losses)) if losses else None, finite_losses=all(v == v and abs(v) != float("inf") for v in losses),
                   wall=time.time() - w0)
        n_rep_since = 0
        curve.append(row)
        print(f"e54 {config} {rule} f{fp} {seed} {u}: A {row['A']:.2f} R {row['R']:.2f} taught {row['taught']:.2f} len12 {row['len12']:.1f} "
              f"len40 {row['len40']:.1f} replaced {row['replaced_since_previous_row']} cpu {cpu_up:.0f}+{cpu_sc:.0f}s", flush=True)

    def checkpoint(u):
        _save(ckp, dict(u=u, model=m.state_dict(), mask=layer.mask.clone(), made=layer.made.clone(),
                        a=None if layer.a is None else layer.a.detach().clone(), Wx=None if layer.Wx is None else layer.Wx.detach().clone(),
                        opt=opt.state_dict(), rr=rr.getstate(), torch_rng=T.get_rng_state(), py_rng=random.getstate(),
                        rec=rec, first=first, digest=dg.value(), cpu_up=cpu_up, cpu_sc=cpu_sc, cpu_rep=cpu_rep, wall=time.time() - w0))

    if u0 == 0:
        score_row(0)
    for u in range(u0 + 1, dose + 1):
        items = next(bt)
        if first is None:
            first = items
            rec["cosine_first_batch_start"] = R.cosines(m, layer, first)
        assert_taught(items, tk)
        dg.add(items)
        c = time.process_time(); losses.append(R.step(m, layer, opt, items, local)); cpu_up += time.process_time() - c
        if u % R.ROW == 0:
            if changing and u % R.L42.EVENT == 0:
                c = time.process_time()
                n_rep_since += R.L42.replace(layer, m, opt, u, theta, rr)
                assert R.check_opt_outside(layer, m, opt), f"optimizer averages not zero outside the mask after the event at {u}"
                cpu_rep += time.process_time() - c
            score_row(u); losses = []
            if u == 12288:
                rec.setdefault("tensors_sha256_at", {})["12288"] = R.R_.tensors_sha(m.state_dict())
            if u % 4096 == 0 or u == dose:
                rp.write_text(json.dumps(rec, indent=1))
            if u % CK == 0 and u < dose:
                checkpoint(u)
                if stop_after is not None and u >= stop_after:
                    print(f"e54 stopped after checkpoint at {u} (test)", flush=True)
                    return None
    rec["cosine_first_batch_end"] = R.cosines(m, layer, first)
    sd = m.state_dict()
    full = dict(sd)
    full.update(mask=layer.mask.clone(), periods=layer.p.clone(), phases=layer.f.clone(), made=layer.made.clone())
    if layer.a is not None:
        full["a"] = layer.a.detach().clone()
    if layer.Wx is not None:
        full["Wx"] = layer.Wx.detach().clone()
    tmp = Path(str(paths["weights"]) + ".tmp"); T.save(full, tmp); os.replace(tmp, paths["weights"])
    small = {k: v for k, v in full.items() if k in ("rnn.weight_hh_l0", "a", "Wx", "mask", "made", "periods", "phases")}
    T.save(small, paths["small"])
    stc = R.structure(layer)
    paths["structure"].write_text(json.dumps(dict(config=config, rule=rule, seed=seed, update=dose, start=struct0, end=stc)))
    rec.update(complete=True, weights=paths["weights"].name, weights_sha256=R.X16.sha_file(paths["weights"]),
               tensors_sha256=R.R_.tensors_sha(sd), small_file=paths["small"].name, structure_file=paths["structure"].name,
               structure_end_sha256=dict(mask=R.tsha(layer.mask), made=R.tsha(layer.made)),
               added_parameters_sha256={k: R.tsha(v) for k, v in (("a", layer.a), ("Wx", layer.Wx)) if v is not None},
               optimizer_steps=R.R_.opt_steps(opt), optimizer_read_back=R.opt_record(m, layer, opt), data_digest=dg.value(),
               cpu_updates=cpu_up, cpu_scoring=cpu_sc, cpu_replacement=cpu_rep, wall=time.time() - w0,
               processes_beside=os.environ.get("E54_BESIDE", "not recorded"))
    rp.write_text(json.dumps(rec, indent=1))
    if ckp.exists():
        ckp.unlink()
    print("e54 complete", rp.name, rec["tensors_sha256"][:16], flush=True)
    return rec


def main(argv):
    cmd = "python3 " + " ".join(argv)
    a = argv[1:]

    def opt(k, d=None):
        return a[a.index(k) + 1] if k in a else d
    assert a[0] == "run"
    config, rule, seed = a[1], a[2], int(a[3])
    assert config in R.CONFIGS and rule in R.RULES and seed in R.SEEDS
    th, d, s = opt("--theta"), opt("--dose"), opt("--stop-after")
    assert "--fraction" in a, "--fraction <50|25|15|10> is required"
    fp = int(opt("--fraction"))
    assert fp in FRACTIONS
    run_ckpt(config, rule, seed, Path(opt("--out", str(OUT))), cmd, fp / 100, theta=float(th) if th else None,
             dose=int(d) if d else None, stop_after=int(s) if s else None, ckdir=Path(opt("--ckdir", str(CKDIR))))


if __name__ == "__main__":
    main(sys.argv)
