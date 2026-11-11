"""E23 check (11 Nov 2026): the far-head option (E16_FARHEAD, E16_FARCTRL) against what it should compute. Run before any E23 birth.

(1) shared init: every layer of the E18 local-heads net starts from the same weights with the far head added (same seed);
(2) isolation: the far head's loss sends no gradient outside {far head, second bank (lp2, lc2, lw2)}, under backprop and under the sign
    rule through time; and it does send a nonzero gradient to the second bank (else the arm is a no-op);
(3) targets: with d = 1 the far targets and mask equal the every-byte next-byte targets (y, y != BOS); with d = 4 the target at input
    position t is the byte at input position t + 4 of the same row; the shuffle keeps each row's multiset of targets and its mask;
(4) the full harness update (update16, rule T, E16_TRACEUP=all) with the far head changes only far-head and second-bank gradients
    relative to the same update without the far loss term.
Writes exp45/E23_CHECK.json; exits 1 on any failure.
"""
import json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
BASE = dict(E16_ARCH="bank2", E16_BOOKDECAY="off", E16_HEADS="local", E16_BIRTHRULE="T", E16_TRACEUP="all")
if len(sys.argv) > 1:
    d, ctrl = int(sys.argv[1]), sys.argv[2]
    os.environ.update(BASE, E16_FARHEAD=str(d), E16_FARCTRL=ctrl)
    from random import Random
    import torch as T
    from torch.nn.functional import cross_entropy
    import exp16_integrated as G, arch17, exp11_cell as E, exp45 as X
    seed = 73129; out = {}; fails = []
    rng = Random(seed + 9); ent = G.family_entries("F1", seed)
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    items = next(E.batches(tp, seed, E.KMAX["M"]))
    x, y, mask = X.batch_tensor(items)
    m = E.new_net(seed)
    arch17.FAR_HEAD = False; ref = arch17.new_net(seed, "bank2"); arch17.FAR_HEAD = True
    out["shared_init_max_abs_diff"] = max(float((p - m.get_parameter(n)).abs().max()) for n, p in ref.named_parameters())
    out["far_params"] = sorted(n for n, _ in m.named_parameters() if n.split(".")[0] in ("lmf", "l3f"))
    allowed = ("lmf", "l3f", "lp2", "lc2", "lw2")
    for name, local, tt in (("backprop", False, True), ("sign_through_time", True, True)):
        _, it = arch17.forward(m, x, local, tt)
        mf = y != E.BOS
        m.zero_grad(); cross_entropy(it["logits_far"][mf], y[mf]).backward()
        out[f"{name}_far_loss_outside_bank2"] = max((float(p.grad.abs().max()) if p.grad is not None else 0.0) for n, p in m.named_parameters() if n.split(".")[0] not in allowed)
        out[f"{name}_far_loss_into_bank2"] = max((float(m.get_parameter(n).grad.abs().max()) if m.get_parameter(n).grad is not None else 0.0) for n in ("lc2.weight", "lw2.weight", "lp2.weight"))
    # (3) targets, taken from the harness's own update by capturing what cross_entropy receives for the far term
    G.FAR_GEN[0] = T.Generator().manual_seed(1)
    captured = []
    orig = E.cross_entropy
    def cap(lg, tg):
        captured.append(tg.clone()); return orig(lg, tg)
    E.cross_entropy = cap
    opt = G.adamw16(m, .001, E.WD)
    m2 = E.new_net(seed); m2.load_state_dict(m.state_dict())
    G.update16(m, opt, items, "T", None)
    E.cross_entropy = orig
    far_t = captured[-1]
    data = T.cat([x, y[:, -1:]], 1)       # data[:, t] is input position t; y[:, t] = data[:, t + 1]
    L = y.shape[1]; exp_t = T.full_like(y, E.BOS); exp_t[:, :L - d + 1] = data[:, d:]
    exp_m = exp_t != E.BOS
    if ctrl == "none":
        out["targets_match_definition"] = bool(T.equal(far_t, exp_t[exp_m]))
        if d == 1:
            out["d1_equals_every_byte"] = bool(T.equal(exp_m, y != E.BOS)) and bool(T.equal(far_t, y[y != E.BOS]))
    else:
        rows_ok, moved, cnt = True, 0, 0
        off = 0
        for i in range(y.shape[0]):
            k = int(exp_m[i].sum()); seg = far_t[off:off + k]; ex = exp_t[i][exp_m[i]]; off += k
            rows_ok &= bool(T.equal(seg.sort().values, ex.sort().values)); moved += int((seg != ex).sum()); cnt += k
        out["shuffle_keeps_row_multisets"] = rows_ok and off == len(far_t)
        out["shuffle_fraction_moved"] = moved / cnt
    # (4) same update without the far term: gradients outside bank2 and the far head must be identical
    m3 = E.new_net(seed); m3.load_state_dict(m2.state_dict())
    lg, it = E.forward(m2, x, local=True, through_time=False, keep=True)
    m2.zero_grad(); (cross_entropy(lg[mask], y[mask]) + cross_entropy(it["logits1"][mask], y[mask])).backward(); E.trace_grads_scan(m2, it)
    gno = {n: (p.grad.clone() if p.grad is not None else T.zeros_like(p)) for n, p in m2.named_parameters()}
    lg, it = E.forward(m3, x, local=True, through_time=False, keep=True)
    mf = exp_m
    m3.zero_grad(); (cross_entropy(lg[mask], y[mask]) + cross_entropy(it["logits1"][mask], y[mask]) + cross_entropy(it["logits_far"][mf], exp_t[mf])).backward(); E.trace_grads_scan(m3, it)
    gyes = {n: (p.grad.clone() if p.grad is not None else T.zeros_like(p)) for n, p in m3.named_parameters()}
    out["update_far_term_outside_bank2_max_abs_diff"] = max(float((gno[n] - gyes[n]).abs().max()) for n in gno if n.split(".")[0] not in allowed)
    out["update_far_term_bank2_max_abs_diff"] = max(float((gno[n] - gyes[n]).abs().max()) for n in ("lc2.weight", "lw2.weight", "lp2.weight"))
    print(json.dumps(out))
else:
    res = {}
    for d, ctrl in ((1, "none"), (4, "none"), (4, "shuffle")):
        r = subprocess.run([sys.executable, __file__, str(d), ctrl], capture_output=True, text=True, cwd=HERE)
        assert r.returncode == 0, r.stderr[-3000:]
        res[f"d{d}_{ctrl}"] = json.loads(r.stdout.strip().splitlines()[-1])
    fails = []
    for k, v in res.items():
        if v["shared_init_max_abs_diff"] != 0.0: fails.append(f"{k}: shared init differs")
        if v["far_params"] != ["l3f.bias", "l3f.weight", "lmf.bias", "lmf.weight"]: fails.append(f"{k}: far params {v['far_params']}")
        for r_ in ("backprop", "sign_through_time"):
            if v[f"{r_}_far_loss_outside_bank2"] != 0.0: fails.append(f"{k}: far loss leaks outside bank2 under {r_}")
            if not v[f"{r_}_far_loss_into_bank2"] > 0.0: fails.append(f"{k}: far loss does not reach bank2 under {r_}")
        if v["update_far_term_outside_bank2_max_abs_diff"] != 0.0: fails.append(f"{k}: harness update changes gradients outside bank2")
        if not v["update_far_term_bank2_max_abs_diff"] > 0.0: fails.append(f"{k}: harness update's far term does not reach bank2")
    if res["d1_none"].get("d1_equals_every_byte") is not True: fails.append("d=1 targets differ from every-byte targets")
    if res["d4_none"].get("targets_match_definition") is not True: fails.append("d=4 targets differ from the definition")
    if res["d4_shuffle"].get("shuffle_keeps_row_multisets") is not True: fails.append("shuffle changes a row's targets")
    if not res["d4_shuffle"].get("shuffle_fraction_moved", 0) > 0.5: fails.append("shuffle moved too few targets")
    res["passed"] = not fails; res["failures"] = fails
    (HERE / "exp45" / "E23_CHECK.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    sys.exit(0 if not fails else 1)
