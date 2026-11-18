"""E52: exp52_rate.run with saving and resuming, for runs that a restart of the host can interrupt (asked by the reviewer, 17 November 2026,
after a worker restart killed three runs). The run itself is exp52_rate.run's, line for line, with two additions:

- every CKPT updates (at a scored row, after the row and the partial receipt are written) the model's state, the optimizer's state, the
  receipt so far and the CPU counters are saved to <receipt>.ckpt (atomically; *.ckpt is ignored by git);
- `--resume` loads the checkpoint, rebuilds the stream and the batch digest by drawing and digesting the same batches again (the stream is
  a generator of Python's Random, rebuilt from (seed, fraction) and advanced; the digest is a running SHA-256 fed the same tensors), loads
  the states and goes on from the next update. The receipt records every resumption (update, wall time).

Every other function (groups, panels, length test, norms, opt_record, receipt name) is exp52_rate's, imported unchanged.

    python3 exp52_resume.py run <S|B> <H> <f> <seed> <rate> [--product] [--dose N] [--out DIR] [--ckpt N] [--resume] [--die-after N]
    python3 exp52_resume.py check 4 [--qual DIR]      # an interrupted and resumed dry run equals an uninterrupted one

--die-after N ends the process abruptly (os._exit) after update N, for check 4 only.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp52_rate as P               # noqa: E402  (imports exp26_candidate first: the environment gate)

import torch as T                    # noqa: E402

C, S41, D46 = P.C, P.S41, P.D46
R_, E, X, arch26 = C.R, C.E, C.X, C.arch26
T.set_num_threads(1)
CKPT = 4096
SOURCES = ("exp52_resume.py",) + P.SOURCES


def provenance(cmd):
    p = P.provenance(cmd)
    p["sources_sha256"] = {s: S41.sha_file(HERE / s) for s in SOURCES}
    p["runner_sha256"] = S41.sha_file(HERE / "exp52_resume.py")
    p["runner"] = "exp52_resume.py (exp52_rate.run with saving and resuming)"
    return p


def save_ckpt(path, m, opt, rec, u, cpu_up, cpu_sc, wall):
    tmp = Path(str(path) + ".tmp")
    T.save(dict(update=u, model=m.state_dict(), opt=opt.state_dict(), rec=rec, cpu_up=cpu_up, cpu_sc=cpu_sc, wall=wall), tmp)
    os.replace(tmp, path)


def run(rule, H, fp, seed, rate, product, out, cmd, dose=None, ckpt_every=CKPT, resume=False, die_after=None):
    arch26.install()
    local, U, f = P.RULES[rule], P.WIDTHS[H], P.FRACS[fp]
    assert rate in P.RATES
    wd = P.decay_of(rate, product)
    dose = dose or P.dose_of(H, rate)
    assert dose % P.ROW == 0 and ckpt_every % P.ROW == 0
    rp = P.receipt_path(out, rule, H, fp, seed, rate, product, dose); wp = rp.with_suffix(".pt"); cp = rp.with_suffix(".ckpt")
    if not resume:
        R_.refuse_existing(rp, wp, str(wp) + ".tmp", cp)
    else:
        assert cp.exists() and not wp.exists(), f"nothing to resume: {cp}"
    Path(out).mkdir(parents=True, exist_ok=True)
    m = S41.new_net(seed, H, U)
    opt = P.make_opt(m, rate, wd, product)
    g_exp = P.expected_groups(m, rate, wd)
    A, Rs, TA = S41.panels(f, seed)
    LP = P.length_panel(f, seed)
    len12_is_A_add = LP[12] == A["add89"]
    tk = S41.taught_keys(f, seed)
    len_in_taught = sum(S41.item_key(p) in tk for k in LP for p, _ in LP[k])
    assert len_in_taught == 0, "the length test holds a taught pair"
    d41 = P.e41_digest_4096(rule, H, fp, seed)
    bt = S41.stream(f, seed); dg = C.Digest()
    start = 0; cpu_up = 0.0; cpu_sc = 0.0; wall0 = 0.0
    if resume:
        ck = T.load(cp, weights_only=False)
        start = ck["update"]; rec = ck["rec"]; cpu_up, cpu_sc, wall0 = ck["cpu_up"], ck["cpu_sc"], ck["wall"]
        c = time.process_time(); w = time.time()
        for _ in range(start):
            items = next(bt); S41.assert_taught(items, tk); dg.add(items)
        assert dg.value() == rec["curve"][-1]["data_digest"], "the rebuilt batch digest differs from the checkpoint's"
        m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        assert R_.opt_groups(m, opt) == g_exp, R_.opt_groups(m, opt)
        assert rec["curve"][-1]["step"] == start
        rec.setdefault("resumed", []).append(dict(at_update=start, at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                  replay_cpu=time.process_time() - c, replay_wall=time.time() - w,
                                                  command=cmd, provenance=provenance(cmd)))
        print(f"e52 resumed at {start}; stream and digest rebuilt in {time.time() - w:.0f}s", flush=True)
    else:
        rec = dict(experiment="E52", kind="birth", rule=rule, local=local, rate=rate, decay=wd, product_control=product, H=H, U=U, fraction=f,
                   seed=seed, dose=dose, e41_dose=S41.DOSE[H], row_every=P.ROW, batch=X.BATCH, kmax=E.KMAX["M"],
                   optimizer=P.opt_record(m, opt, rate, wd), optimizer_groups_at_start=R_.opt_groups(m, opt), optimizer_groups_expected=g_exp,
                   model=S41.describe(m),
                   update="exp26_candidate.step (forward(local), answer-masked cross entropy, zero_grad, backward, AdamW step)",
                   data="exp11_cell.batches([(t, split(t, f, seed)[0]) for t in DEV], seed, 12) (exp41_scale.stream)",
                   reduction_sample_sha256={n: S41.sample_digest([S41.item_key(p)[1:] for p, _ in it]) for n, it in Rs.items()},
                   length_test="exp11_cell.panel(('add', 89, b'p'), split(('add', 89, b'p'), f, seed)[1], seed, ks=(12, 40), n=300), f the run's fraction",
                   length_test_12_items_equal_A_add89=len12_is_A_add, length_test_items_in_taught=len_in_taught,
                   norm_groups={g: list(ns) for g, ns in D46.GROUPS.items()}, e41_digest_4096=d41, checkpoint_every=ckpt_every,
                   provenance=provenance(cmd), complete=False, diverged=False, curve=[])
    curve = rec["curve"]; w0 = time.time() - wall0; losses = []

    def score_row(u):
        nonlocal cpu_sc
        c = time.process_time()
        row = dict(step=u, **S41.score_all(m, A, Rs, TA), **P.score_length(m, LP), norms=D46.norms(m))
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

    if start == 0:
        score_row(0)
    done = start
    for u in range(start + 1, dose + 1):
        items = next(bt)
        S41.assert_taught(items, tk)
        dg.add(items)
        c = time.process_time(); losses.append(C.step(m, opt, items, local)); cpu_up += time.process_time() - c
        done = u
        if u % P.ROW == 0:
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
            if u % ckpt_every == 0 and u != dose:
                save_ckpt(cp, m, opt, rec, u, cpu_up, cpu_sc, time.time() - w0)
        if die_after is not None and u == die_after:
            print(f"e52 dying after update {u} (check 4)", flush=True)
            os._exit(9)
    g_end = R_.opt_groups(m, opt)
    assert g_end == g_exp, f"groups changed during the run: {g_end}"
    wsha = R_.save_weights(m, wp)
    rec.update(complete=True, updates_done=done, weights=wp.name, weights_sha256=wsha, tensors_sha256=R_.tensors_sha(m.state_dict()),
               optimizer_steps=R_.opt_steps(opt), optimizer_read_back=P.opt_record(m, opt, rate, wd), optimizer_groups_at_end=g_end,
               data_digest=dg.value(), cpu_updates=cpu_up, cpu_scoring=cpu_sc, wall=time.time() - w0,
               processes_beside=os.environ.get("E52_BESIDE", "not recorded"))
    rp.write_text(json.dumps(rec, indent=1))
    if cp.exists():
        cp.unlink()
    print("e52 complete", rp.name, rec["tensors_sha256"][:16], "DIVERGED" if rec["diverged"] else "", flush=True)
    return rec


ROWKEYS = ("A_counts", "R_counts", "taught_counts", "length_counts", "norms", "data_digest", "loss", "finite_losses")


def check4(qual, cmd):
    """Half width, f = 0.25, seed 93683, rate 0.003, under each rule, 3,072 updates with a checkpoint every 1,024: (i) uninterrupted with
    exp52_rate.run; (ii) uninterrupted with this runner; (iii) this runner killed after update 2,000 (os._exit) and resumed from the checkpoint
    at 1,024. Equal: final tensors_sha256, every row's counts, norms, losses and batch digest."""
    import subprocess
    out = {}; ok = True
    for rule in "SB":
        base = Path(qual) / rule
        r1 = P.run(rule, 256, 25, 93683, 0.003, False, base / "i", cmd, dose=3072)
        r2 = run(rule, 256, 25, 93683, 0.003, False, base / "ii", cmd, dose=3072, ckpt_every=1024)
        args = [sys.executable, str(HERE / "exp52_resume.py"), "run", rule, "256", "25", "93683", "0.003", "--dose", "3072",
                "--out", str(base / "iii"), "--ckpt", "1024"]
        k = subprocess.run(args + ["--die-after", "2000"], cwd=HERE, capture_output=True, text=True)
        rp3 = P.receipt_path(base / "iii", rule, 256, 25, 93683, 0.003, False, 3072)
        partial = json.loads(rp3.read_text()) if rp3.exists() else None
        k2 = subprocess.run(args + ["--resume"], cwd=HERE, capture_output=True, text=True)
        r3 = json.loads(rp3.read_text())
        rows = lambda r: [{k_: x[k_] for k_ in ROWKEYS} | dict(step=x["step"]) for x in r["curve"]]
        res = dict(killed_exit_code=k.returncode, partial_rows=None if partial is None else len(partial["curve"]),
                   resume_exit_code=k2.returncode, resumed=r3.get("resumed", [{}])[0].get("at_update"),
                   tensors=dict(i=r1["tensors_sha256"], ii=r2["tensors_sha256"], iii=r3.get("tensors_sha256")),
                   rows_equal_i_ii=rows(r1) == rows(r2), rows_equal_i_iii=rows(r1) == rows(r3),
                   digest_equal=r1["data_digest"] == r2["data_digest"] == r3.get("data_digest"),
                   optimizer_steps=dict(i=r1["optimizer_steps"], iii=r3.get("optimizer_steps")), n_rows=len(r1["curve"]))
        res["pass"] = (k.returncode == 9 and k2.returncode == 0 and res["resumed"] == 1024
                       and r1["tensors_sha256"] == r2["tensors_sha256"] == r3.get("tensors_sha256")
                       and res["rows_equal_i_ii"] and res["rows_equal_i_iii"] and res["digest_equal"])
        if not res["pass"]:
            res["stderr_tail"] = (k.stderr[-1500:], k2.stderr[-1500:])
        ok = ok and res["pass"]; out[rule] = res
    out["pass"] = ok
    return out


def main(argv):
    cmd = "python3 " + " ".join(argv)
    a = argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    if a and a[0] == "run":
        rule, H, fp, seed, rate = a[1], int(a[2]), int(a[3]), int(a[4]), float(a[5])
        assert rule in P.RULES and H in P.WIDTHS and fp in P.FRACS and seed in S41.SEEDS and rate in P.RATES
        product = "--product" in a
        assert not product or rate in (0.0003, 0.003)
        d = opt("--dose", None); da = opt("--die-after", None)
        run(rule, H, fp, seed, rate, product, Path(opt("--out", str(P.OUT))), cmd, dose=int(d) if d else None,
            ckpt_every=int(opt("--ckpt", str(CKPT))), resume="--resume" in a, die_after=int(da) if da else None)
    elif a and a[0] == "check":
        assert a[1] == "4"
        qual = Path(opt("--qual", str(P.QUAL))); qual.mkdir(parents=True, exist_ok=True)
        p = qual / "check4.json"
        R_.refuse_existing(p)
        t0 = time.process_time(); w0 = time.time()
        res = check4(qual / "check4_runs", cmd)
        res.update(check="4 (saving and resuming)", cpu=time.process_time() - t0, wall=time.time() - w0, provenance=provenance(cmd))
        p.write_text(json.dumps(res, indent=1))
        print("e52 check 4:", "PASS" if res["pass"] else "FAIL", flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
