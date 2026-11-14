"""E30 (brief OPUS_TASK_E30_GATE_MEASURE_20261113.md): a registered gate measure. Read-only; trains nothing; edits nothing existing.

    python3 e30_gates.py calibrate      step 1: U on the plus-minus panel for the twelve E26 / E26-X lives -> exp45/e30/E30_CALIBRATION.json
    python3 e30_gates.py unseen <e28 dir> <branch@commit>   step 2 (only after step 1 passes and is pushed) -> exp45/e30/E30_UNSEEN.json

U, the undecided share: over every item, every position after the first byte (input positions 1 .. L-1 of x = [BOS] + prompt +
answer[:-1]; position 0 is the BOS byte) and all units, the percentage of update-gate values z with 0.1 <= z <= 0.9. Gate values come
from e29_gates.gates(m, x), which asserts bit-identity with arch26.forward.

H, the holding loss: whole-answer accuracy (exp16_integrated.score, exact) at 12 letters minus at 40 letters, 300 items each, from
exp11_cell.panel(t, exp11_cell.split(t, .5, seed)[1], seed, ks=(12, 40), n=300); t = add89 for births, add97 for lives.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e29_gates as G29              # noqa: E402  (imports exp20_reference: the harness's environment gate)
import torch as T                    # noqa: E402

R, X16, E, X, arch26 = G29.R, G29.X16, G29.E, G29.X, G29.arch26
OUT = HERE / "exp45" / "e30"
SEEDS = G29.SEEDS
CAL_RANGES = {"S": (73.0, 79.0), "B": (43.0, 53.0)}      # the review's ranges, by the update used during the life
TOL = 3.0


def U_of(m, items):
    x, _, _ = X.batch_tensor(items)
    assert not bool((x[:, 1:] == X.BOS).any()), "padding inside a panel"
    Z, chk = G29.gates(m, x)
    z = Z[:, 1:, :]
    und = ((z >= 0.1) & (z <= 0.9)).sum().item()
    return dict(U=round(100 * und / z.numel(), 2), undecided=und, values=z.numel(), items=len(items), positions_after_first=z.shape[1],
                units=z.shape[2], items_sha256=G29.sha_items(items), arch26=chk)


def H_of(m, task, seed):
    pk = E.panel(task, E.split(task, .5, seed)[1], seed, ks=(12, 40), n=300)
    s = {k: X16.score(m, pk[k]) for k in (12, 40)}
    acc = {k: 100 * s[k]["exact"] / s[k]["n"] for k in (12, 40)}
    return dict(task=E.NAME(*task), acc12=round(acc[12], 2), acc40=round(acc[40], 2), H=round(acc[12] - acc[40], 2),
                counts={str(k): s[k] for k in (12, 40)})


def load_net(path, seed):
    m = arch26.new_net(seed)
    sha = G29.load(path)                 # asserts sha256 against the receipt
    m.load_state_dict(T.load(path))
    m.eval()
    return m, sha


def calibrate():
    OUT.mkdir(parents=True, exist_ok=True)
    arch26.install()
    rows, ok = [], True
    for s in SEEDS:
        for cell, path in (("SS", G29.E26 / f"e26_life_S_{s}.pt"), ("BB", G29.E26 / f"e26_life_B_{s}.pt"),
                           ("SB", G29.E26X / f"e26x_life_SB_{s}.pt"), ("BS", G29.E26X / f"e26x_life_BS_{s}.pt")):
            t0 = time.process_time()
            m, sha = load_net(path, s)
            u = U_of(m, G29.plus_minus_panel(s))
            upd = cell[1]
            lo, hi = CAL_RANGES[upd]
            inside = lo - TOL <= u["U"] <= hi + TOL
            ok &= inside
            rows.append(dict(file=str(path.relative_to(HERE)), sha256=sha, cell=cell, birth=cell[0], life_update=upd, seed=s,
                             review_range=[lo, hi], within_range_plus_3=inside, cpu_s=round(time.process_time() - t0, 2), **u))
            print(f"{cell} {s} U={u['U']:.2f}  review {lo}-{hi}  {'ok' if inside else 'OUTSIDE'}", flush=True)
    rec = dict(brief="OPUS_TASK_E30_GATE_MEASURE_20261113.md", step=1, passed=ok, tolerance_points=TOL, rows=rows,
               python=sys.version.split()[0], torch=T.__version__,
               definition="U = % of z in [0.1, 0.9] over all items, input positions 1..L-1, all units; plus-minus panel comp['+-'] + comp['-+']")
    (OUT / "E30_CALIBRATION.json").write_text(json.dumps(rec, indent=1))
    print("calibration", "PASSED" if ok else "FAILED")
    return ok


def sha_file(p):
    return X16.sha_file(p)


def unseen(e28_dir, e28_source):
    """Step 2. e28_dir holds the E28 receipts and weight files as committed at e28_source (branch@commit); every file's sha256 is
    checked against its receipt (final: weights_sha256; rows: row_weights[step].sha256). Only receipts with complete == True are used."""
    cal = json.loads((OUT / "E30_CALIBRATION.json").read_text())
    assert cal.get("passed") is True, "step 1 has not passed"
    arch26.install()
    e28_dir = Path(e28_dir)
    jobs = []
    for rp in sorted(e28_dir.glob("e28_dev_*.json")):
        rec = json.loads(rp.read_text())
        if rec.get("complete") is not True:
            continue
        arm, seed = rec["arm"], rec["seed"]
        final_step = max(int(k) for k in rec["row_weights"])
        jobs.append(("E28", arm, seed, final_step, e28_dir / rec["weights"], rec["weights_sha256"], "final"))
        if arm == "S-all":
            for k, v in sorted(rec["row_weights"].items(), key=lambda kv: int(kv[0])):
                if int(k) < final_step:
                    jobs.append(("E28", arm, seed, int(k), e28_dir / v["file"], v["sha256"], "row"))
    for s in SEEDS:
        for a in ("S", "B"):
            p = G29.E26 / f"e26_dev_{a}_{s}.pt"
            jobs.append(("E26", a, s, None, p, json.loads(p.with_suffix(".json").read_text())["weights_sha256"], "final"))
    rows = []
    for exp, arm, seed, step, path, want, kind in jobs:
        t0 = time.process_time()
        sha = sha_file(path)
        assert sha == want, f"{path.name}: sha256 {sha[:12]} is not its receipt's {want[:12]}"
        m = arch26.new_net(seed)
        m.load_state_dict(T.load(path))
        m.eval()
        ua = U_of(m, G29.add89_panel(seed, 12))
        up = U_of(m, G29.plus_minus_panel(seed))
        h = H_of(m, E.DEV[0], seed)
        rows.append(dict(experiment=exp, arm=arm, seed=seed, row=step, kind=kind, file=path.name, sha256=sha,
                         U_add89_k12=ua["U"], U_plus_minus=up["U"], H=h["H"], acc12=h["acc12"], acc40=h["acc40"],
                         detail=dict(U_add89_k12=ua, U_plus_minus=up, H=h), cpu_s=round(time.process_time() - t0, 2)))
        print(f"{exp} {arm} {seed} row {step} {kind}: U add89 {ua['U']:.2f}  U pm {up['U']:.2f}  H {h['H']:.2f} ({h['acc12']:.1f} - {h['acc40']:.1f})", flush=True)
    rec = dict(brief="OPUS_TASK_E30_GATE_MEASURE_20261113.md", step=2, e28_source=e28_source, rows=rows,
               python=sys.version.split()[0], torch=T.__version__,
               not_measured="E28 births without a complete receipt at e28_source (stage 2b in flight): S-bal 93683, B-bal 93683 and all four arms of 116089")
    (OUT / "E30_UNSEEN.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "calibrate":
        sys.exit(0 if calibrate() else 1)
    if cmd == "unseen":
        unseen(sys.argv[2], sys.argv[3]); sys.exit(0)
    raise SystemExit(__doc__)
