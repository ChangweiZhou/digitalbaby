"""E30 (brief OPUS_TASK_E30_GATE_MEASURE_20261113.md): a registered gate measure. Read-only; trains nothing; edits nothing existing.

    python3 e30_gates.py calibrate      step 1: U on the plus-minus panel for the twelve E26 / E26-X lives -> exp45/e30/E30_CALIBRATION.json
    python3 e30_gates.py unseen         step 2 (only after step 1 passes and is pushed) -> exp45/e30/E30_UNSEEN.json

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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "calibrate":
        sys.exit(0 if calibrate() else 1)
    if cmd == "unseen":
        import e30_unseen      # noqa: F401  (written only after step 1 passes)
        sys.exit(0)
    raise SystemExit(__doc__)
