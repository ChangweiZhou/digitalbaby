"""E23 / E18c check (11 Nov 2026, v2 after the internal audit): the far-head option (E16_FARHEAD, E16_FARCTRL), the first head's
every-byte loss (E16_HEAD1LOSS=all) and the relay mode (E16_HEADS=relay), against what they should compute. Run before any birth.

Far-head modes (d, ctrl), all with E16_HEAD1LOSS=all on the B2 package (local heads, rule T, E16_TRACEUP=all):
(1) shared init: every layer of the local-heads net starts from the same weights with the far head added (same seed);
(2) isolation: the far head's loss sends no gradient outside {far head, second bank (lp2, lc2, lw2)}, under backprop and under the
    sign rule through time, and does send a nonzero gradient into the second bank;
(3) targets, captured from update16 itself: the far target at input position t is the byte at input position t + d of the same row,
    restricted to prompt bytes (no answer byte, no EOS); the shuffle keeps each row's multiset and moves most targets; the first head's
    targets are every next byte; the main head's are the answer bytes;
(4) gradients of update16 itself (optimizer step disabled) with the far term on and off (FARHEAD16 = 0) differ only in the far head and
    the second bank, under the same shuffle generator state.
Relay mode: (5) the first head's loss sends nothing into the second bank's module; the main loss reaches the first bank; with l2's
weights zeroed (no direct z1 path in either), relay and aux give identical gradients; with them intact, they differ in l1 (the z1 path
is what relay removes). Writes exp45/E23_CHECK.json; exits 1 on any failure.
"""
import json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
BASE = dict(E16_ARCH="bank2", E16_BOOKDECAY="off", E16_BIRTHRULE="T", E16_TRACEUP="all")
if len(sys.argv) > 1 and sys.argv[1] == "relay":
    import torch as T
    from torch.nn.functional import cross_entropy
    os.environ.update(BASE, E16_HEADS="relay")
    import exp16_integrated as G, arch17, exp11_cell as E, exp45 as X
    seed = 73129; out = {}
    items = next(E.batches([(t, E.split(t, .5, seed)[0]) for t in E.DEV], seed, E.KMAX["M"]))
    x, y, mask = X.batch_tensor(items)
    mod1 = ("book", "l1", "lp", "lc", "lw", "l2a", "lma", "l3a")
    def grads(which, z1det, zero_l2=False):
        arch17.Z1_DETACH = z1det
        m = E.new_net(seed)
        if zero_l2:
            with T.no_grad(): m.l2.weight.zero_()
        lg, it = arch17.forward(m, x, True, True)
        m.zero_grad(); cross_entropy((lg if which == "main" else it["logits1"])[mask], y[mask]).backward()
        return {n: (p.grad.clone() if p.grad is not None else T.zeros_like(p)) for n, p in m.named_parameters()}
    g = grads("head1", True)
    out["relay_head1_loss_into_bank2_module"] = max(float(v.abs().max()) for n, v in g.items() if n.split(".")[0] not in mod1)
    g = grads("main", True)
    out["relay_main_loss_into_bank1"] = max(float(g[n].abs().max()) for n in ("lp.weight", "lc.weight", "lw.weight", "l1.weight", "book.weight"))
    r0, a0 = grads("main", True, True), grads("main", False, True)
    out["relay_vs_aux_l2_zeroed_max_abs_diff"] = max(float((r0[n] - a0[n]).abs().max()) for n in r0)
    r1, a1 = grads("main", True), grads("main", False)
    out["relay_vs_aux_l1_diff_intact"] = float((r1["l1.weight"] - a1["l1.weight"]).abs().max())
    arch17.Z1_DETACH = True
    out["tag"] = G.HTAG
    print(json.dumps(out)); sys.exit(0)
if len(sys.argv) > 1:
    d, ctrl = int(sys.argv[1]), sys.argv[2]
    os.environ.update(BASE, E16_HEADS="local", E16_HEAD1LOSS="all", E16_FARHEAD=str(d), E16_FARCTRL=ctrl)
    import torch as T
    from torch.nn.functional import cross_entropy
    import exp16_integrated as G, arch17, exp11_cell as E, exp45 as X
    seed = 73129; out = {}
    items = next(E.batches([(t, E.split(t, .5, seed)[0]) for t in E.DEV], seed, E.KMAX["M"]))
    x, y, mask = X.batch_tensor(items)
    m = E.new_net(seed)
    arch17.FAR_HEAD = False; ref = arch17.new_net(seed, "bank2"); arch17.FAR_HEAD = True
    out["shared_init_max_abs_diff"] = max(float((p - m.get_parameter(n)).abs().max()) for n, p in ref.named_parameters())
    out["far_params"] = sorted(n for n, _ in m.named_parameters() if n.split(".")[0] in ("lmf", "l3f"))
    out["params_total"] = sum(p.numel() for p in m.parameters()); out["params_far_head"] = sum(p.numel() for n, p in m.named_parameters() if n.split(".")[0] in ("lmf", "l3f"))
    allowed = ("lmf", "l3f", "lp2", "lc2", "lw2")
    for name, local, tt in (("backprop", False, True), ("sign_through_time", True, True)):
        _, it = arch17.forward(m, x, local, tt)
        mf = y != E.BOS
        m.zero_grad(); cross_entropy(it["logits_far"][mf], y[mf]).backward()
        out[f"{name}_far_loss_outside_bank2"] = max((float(p.grad.abs().max()) if p.grad is not None else 0.0) for n, p in m.named_parameters() if n.split(".")[0] not in allowed)
        out[f"{name}_far_loss_into_bank2"] = max((float(m.get_parameter(n).grad.abs().max()) if m.get_parameter(n).grad is not None else 0.0) for n in ("lc2.weight", "lw2.weight", "lp2.weight"))
    # (3) and (4): the harness's own update, optimizer step disabled, cross_entropy's targets captured
    captured = []
    orig = E.cross_entropy
    def cap(lg, tg):
        captured.append(tg.clone()); return orig(lg, tg)
    def run_update(far_on):
        G.FARHEAD16 = d if far_on else 0
        G.FAR_GEN[0] = T.Generator().manual_seed(1)
        mm = E.new_net(seed); opt = G.adamw16(mm, .001, E.WD); opt.step = lambda *a, **k: None
        captured.clear(); E.cross_entropy = cap
        try:
            G.update16(mm, opt, items, "T", None)
        finally:
            E.cross_entropy = orig; G.FARHEAD16 = d
        return {n: (p.grad.clone() if p.grad is not None else T.zeros_like(p)) for n, p in mm.named_parameters()}, list(captured)
    gno, cno = run_update(False)
    gyes, cyes = run_update(True)
    out["update16_loss_terms"] = [len(cno), len(cyes)]
    out["main_targets_are_answer_bytes"] = bool(T.equal(cyes[0], y[mask]))
    out["head1_targets_are_every_byte"] = bool(T.equal(cyes[1], y[y != E.BOS]))
    far_t = cyes[2]
    data = T.cat([x, y[:, -1:]], 1)       # data[:, t] is input position t; y[:, t] = data[:, t + 1]
    L = y.shape[1]; exp_t = T.full_like(y, E.BOS); exp_t[:, :L - d + 1] = data[:, d:]
    ans = T.zeros_like(mask); ans[:, :L - d + 1] = mask[:, d - 1:]
    exp_m = (exp_t != E.BOS) & ~ans
    out["far_target_positions"] = int(exp_m.sum()); out["far_targets_answer_or_eos"] = int(((far_t >= 256) | (far_t < 0)).sum())
    if ctrl == "none":
        out["targets_match_definition"] = bool(T.equal(far_t, exp_t[exp_m]))
    else:
        rows_ok, moved, cnt, off = True, 0, 0, 0
        for i in range(y.shape[0]):
            k = int(exp_m[i].sum()); seg = far_t[off:off + k]; ex = exp_t[i][exp_m[i]]; off += k
            rows_ok &= bool(T.equal(seg.sort().values, ex.sort().values)); moved += int((seg != ex).sum()); cnt += k
        out["shuffle_keeps_row_multisets"] = rows_ok and off == len(far_t)
        out["shuffle_fraction_moved"] = moved / cnt
    out["update16_far_term_outside_bank2_max_abs_diff"] = max(float((gno[n] - gyes[n]).abs().max()) for n in gno if n.split(".")[0] not in allowed)
    out["update16_far_term_bank2_max_abs_diff"] = max(float((gno[n] - gyes[n]).abs().max()) for n in ("lc2.weight", "lw2.weight", "lp2.weight"))
    print(json.dumps(out))
else:
    res = {}
    for args in (("1", "none"), ("4", "none"), ("4", "shuffle"), ("relay",)):
        r = subprocess.run([sys.executable, __file__, *args], capture_output=True, text=True, cwd=HERE)
        assert r.returncode == 0, r.stderr[-3000:]
        res["_".join(args)] = json.loads(r.stdout.strip().splitlines()[-1])
    fails = []
    for k, v in res.items():
        if k == "relay":
            continue
        if v["shared_init_max_abs_diff"] != 0.0: fails.append(f"{k}: shared init differs")
        if v["far_params"] != ["l3f.bias", "l3f.weight", "lmf.bias", "lmf.weight"]: fails.append(f"{k}: far params {v['far_params']}")
        for r_ in ("backprop", "sign_through_time"):
            if v[f"{r_}_far_loss_outside_bank2"] != 0.0: fails.append(f"{k}: far loss leaks outside bank2 under {r_}")
            if not v[f"{r_}_far_loss_into_bank2"] > 0.0: fails.append(f"{k}: far loss does not reach bank2 under {r_}")
        if v["update16_loss_terms"] != [2, 3]: fails.append(f"{k}: update16 loss terms {v['update16_loss_terms']}")
        if not v["main_targets_are_answer_bytes"]: fails.append(f"{k}: main head targets are not the answer bytes")
        if not v["head1_targets_are_every_byte"]: fails.append(f"{k}: first head targets are not every next byte")
        if v["far_targets_answer_or_eos"] != 0: fails.append(f"{k}: far targets include EOS")
        if v["update16_far_term_outside_bank2_max_abs_diff"] != 0.0: fails.append(f"{k}: update16 far term changes gradients outside bank2")
        if not v["update16_far_term_bank2_max_abs_diff"] > 0.0: fails.append(f"{k}: update16 far term does not reach bank2")
    if res["4_none"].get("targets_match_definition") is not True: fails.append("d=4 targets differ from the definition")
    if res["1_none"].get("targets_match_definition") is not True: fails.append("d=1 targets differ from the definition")
    if res["4_shuffle"].get("shuffle_keeps_row_multisets") is not True: fails.append("shuffle changes a row's targets")
    if not res["4_shuffle"].get("shuffle_fraction_moved", 0) > 0.5: fails.append("shuffle moved too few targets")
    rl = res["relay"]
    if rl["relay_head1_loss_into_bank2_module"] != 0.0: fails.append("relay: first head's loss reaches the second bank")
    if not rl["relay_main_loss_into_bank1"] > 0.0: fails.append("relay: main loss does not reach the first bank")
    if rl["relay_vs_aux_l2_zeroed_max_abs_diff"] != 0.0: fails.append("relay differs from aux beyond the z1 path")
    if not rl["relay_vs_aux_l1_diff_intact"] > 0.0: fails.append("relay does not remove the z1 path")
    if rl["tag"] != "_rh": fails.append("relay tag")
    res["passed"] = not fails; res["failures"] = fails
    (HERE / "exp45" / "E23_CHECK.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    sys.exit(0 if not fails else 1)
