"""E38 check 2 diagnostic (not one of the brief's checks): where does the anchor-to-zero step part from torch's own decay?

Check 2 as specified failed (largest relative difference up to 5e-4 after 64 updates, the undecayed embedding table differing too).
This reports the per-update growth of the largest relative difference, in float32 as the runner runs and with every tensor in float64,
where rounding is about 1e-16; and the same for check 3's two forms of the anchor: if the two forms are the same arithmetic, the float64 differences stay near rounding.

    python3 e38_check2_diag.py <S|B> <seed> <float32|float64> [--out DIR]
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp38_erase as E38            # noqa: E402

C, R, X16, E, X, arch26, T = E38.C, E38.R, E38.X16, E38.E, E38.X, E38.arch26, E38.T
LR = E38.LR
STOPS = (1, 2, 4, 8, 16, 32, 64)


def run(rule, seed, dtype, mode, kappa):
    """mode 'torch': AdamW weight_decay = kappa on group 0. mode 'pull': weight_decay 0, p -= LR * kappa * p before the step (birth = 0)."""
    arch26.install()
    bfile = C.birth_path(E38.E26DIR, rule, seed); m, _, _ = E38.admit(bfile, rule, seed)
    m.load_state_dict(T.load(bfile)); m.to(dtype)
    opt = C.make_opt(m) if dtype == T.float32 else X16.adamw16(m, LR, C.WD, ())
    decayed = list(opt.param_groups[0]["params"])
    opt.param_groups[0]["weight_decay"] = kappa if mode == "torch" else 0.0
    if mode == "pull":
        pull = E38.make_pull(decayed, [T.zeros_like(p) for p in decayed], kappa, 1)
    elif mode in ("birth1", "birth2"):                    # check 3's pair: the anchor to the birth values, form 1 and form 2
        pull = E38.make_pull(decayed, [p.detach().clone() for p in decayed], kappa, int(mode[-1]))
    else:
        pull = None
    joint = X16.joint_stream([X16.entry_stream(X16.family_entries("F1", seed), seed + 101)], seed + 1)
    snaps = {}
    for g in range(1, 65):
        batch = [next(joint) for _ in range(X.BATCH)]
        if dtype == T.float32:
            E38.step(m, opt, batch, C.ARMS[rule], pull)
        else:
            x, y, mask = X.batch_tensor(batch)
            logits = arch26.forward(m, x, C.ARMS[rule], True)[0]
            loss = E.cross_entropy(logits[mask].double(), y[mask])
            opt.zero_grad(); loss.backward()
            if pull is not None:
                pull()
            opt.step()
        if g in STOPS:
            snaps[g] = {n: p.detach().clone() for n, p in m.named_parameters()}
    return snaps


if __name__ == "__main__":
    T.set_num_threads(1)
    rule, seed, dt = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else E38.OUT38
    rp = out / f"e38_check2_diag_{rule}_{seed}_{dt}.json"; R.refuse_existing(rp)
    dtype = T.float64 if dt == "float64" else T.float32
    if dtype == T.float64:
        T.set_default_dtype(T.float64)
    t0 = time.process_time(); res = {}
    for pair in (("torch", "pull"), ("birth1", "birth2")):
        for kappa in (0.25, 2.5):
            a, b = run(rule, seed, dtype, pair[0], kappa), run(rule, seed, dtype, pair[1], kappa)
            key = f"{pair[0]}_vs_{pair[1]}_{kappa}"
            res[key] = {g: {n: float((a[g][n] - b[g][n]).abs().max() / a[g][n].abs().max()) for n in a[g]} for g in STOPS}
            print(dt, key, {g: f"{max(v.values()):.1e}" for g, v in res[key].items()}, flush=True)
    rp.write_text(json.dumps(dict(experiment="E38", diagnostic="check 2", rule=rule, seed=seed, dtype=dt, stops=STOPS, max_rel=res,
                                  provenance=E38.provenance(" ".join(sys.argv)), cpu_seconds=time.process_time() - t0), indent=1))
