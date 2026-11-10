"""E18 check (10 Nov 2026): the local-heads option (E16_HEADS=local) and the birth-rule option, against what they should compute.

Mode `off` (knobs unset): on a first-block batch at a bank2 birth, the gradients of backprop, of the sign rule and of the sign rule with
credit through time must equal those of arch17.py as pushed at a3c0b675 (loaded from git), to the last bit.
Mode `on` (E16_HEADS=local): (1) every layer shared with the plain bank2 net has the same initial weights for the same seed;
(2) the main head's loss sends no gradient to the first bank's module (book, l1, lp, lc, lw, l2a, lma, l3a), and the first bank's head
sends none to the second's (lp2, lc2, lw2, lm2, lm, l2, l3), under backprop and under the sign rule through time; (3) under the plain
sign rule the first bank's write synapses get exactly the scan trace of the first head's error.
"""
import json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
if len(sys.argv) > 1:
    os.environ.update(E16_ARCH="bank2", E16_BOOKDECAY="off", E16_REPLAY="raw", E16_CAP="65536")
    if sys.argv[1] in ("on", "leak"):
        os.environ["E16_HEADS"] = "local" if sys.argv[1] == "on" else "aux"     # aux: head kept, relay kept: the fixture that MUST leak
    from random import Random
    import torch as T
    from torch.nn.functional import cross_entropy
    import exp16_integrated as G
    import arch17, exp11_cell as E, exp45 as X
    seed = 73129
    rng = Random(seed + 9); ent = G.family_entries("F1", seed)
    x, y, mask = X.batch_tensor([G.render(rng.choice(ent), rng) for _ in range(64)])
    out = {}
    if sys.argv[1] == "off":
        import importlib.util, tempfile
        src = subprocess.run(["git", "show", "a3c0b675:MOD97_LOCAL_RULE_20261106/arch17.py"], capture_output=True, text=True, cwd=HERE).stdout
        f = Path(tempfile.mkdtemp()) / "arch17_old.py"; f.write_text(src)
        spec = importlib.util.spec_from_file_location("arch17_old", f); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        worst = 0.0
        for local, tt in ((False, True), (True, False), (True, True)):
            g = []
            for mod in (arch17, old):
                m = mod.new_net(seed, "bank2"); m.load_state_dict(T.load(G.OUT / f"e17_dev_bank2_nbd_{seed}.pt"))
                lg, it = mod.forward(m, x, local, tt, keep=(local and not tt))
                m.zero_grad(); cross_entropy(lg[mask], y[mask]).backward()
                if local and not tt:
                    mod.TRACEUP, mod.FB_SIGN = False, True; mod.trace_grads_scan(m, it)
                g.append({n: p.grad.clone() for n, p in m.named_parameters()})
            worst = max(worst, max(float((g[0][n] - g[1][n]).abs().max()) for n in g[0]))
        out["off_vs_a3c0b675_max_abs_grad_diff"] = worst
    else:
        m = E.new_net(seed); plain = arch17.Net17.__new__(arch17.Net17)
        arch17.LOCAL_HEADS = False; ref = arch17.new_net(seed, "bank2"); arch17.LOCAL_HEADS = True
        out["shared_init_max_abs_diff"] = max(float((p - ref.get_parameter(n)).abs().max()) for n, p in m.named_parameters() if n.split(".")[0] not in ("l2a", "lma", "l3a"))
        out["new_params"] = sorted(n for n, _ in m.named_parameters() if n.split(".")[0] in ("l2a", "lma", "l3a"))
        mod1 = ("book", "l1", "lp", "lc", "lw", "l2a", "lma", "l3a"); leak = {}
        for name, local, tt in (("backprop", False, True), ("sign_through_time", True, True)):
            for which in ("main", "head1"):
                lg, it = arch17.forward(m, x, local, tt)
                m.zero_grad(); cross_entropy((lg if which == "main" else it["logits1"])[mask], y[mask]).backward()
                other = [n for n, p in m.named_parameters() if (n.split(".")[0] in mod1) == (which == "main")]
                leak[f"{name}_{which}_loss_into_other_module"] = max(float(m.get_parameter(n).grad.abs().max()) if m.get_parameter(n).grad is not None else 0.0 for n in other)
        out["leak" if sys.argv[1] == "on" else "leak_fixture_relay_kept"] = leak
        if sys.argv[1] == "leak":
            print(json.dumps(out)); sys.exit(0)
        # plain sign rule: the harness update's gradient on the first bank's write synapses against a scan of the first head's error alone
        lg, it = E.forward(m, x, local=True, through_time=False, keep=True)
        m.zero_grad(); (cross_entropy(lg[mask], y[mask]) + cross_entropy(it["logits1"][mask], y[mask])).backward(); E.trace_grads_scan(m, it)
        both = {n: m.get_parameter(n).grad.clone() for n in ("lc.weight", "lw.weight", "lc2.weight", "lw2.weight")}
        lg, it = E.forward(m, x, local=True, through_time=False, keep=True)
        m.zero_grad(); cross_entropy(it["logits1"][mask], y[mask]).backward(); E.trace_grads_scan(m, it)
        out["bank1_write_from_head1_only_max_abs_diff"] = max(float((both[n] - m.get_parameter(n).grad).abs().max()) for n in ("lc.weight", "lw.weight"))
        lg, it = E.forward(m, x, local=True, through_time=False, keep=True)
        m.zero_grad(); cross_entropy(lg[mask], y[mask]).backward(); E.trace_grads_scan(m, it)
        out["bank2_write_from_main_only_max_abs_diff"] = max(float((both[n] - m.get_parameter(n).grad).abs().max()) for n in ("lc2.weight", "lw2.weight"))
        out["head1_out_switch"] = bool(T.equal(G.head1(lambda: arch17.forward(m, x, False, True)[0]), arch17.forward(m, x, False, True)[1]["logits1"]))
    print(json.dumps(out))
else:
    res = {}
    for mode in ("off", "on", "leak"):
        r = subprocess.run([sys.executable, __file__, mode], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-3000:]
        res.update(json.loads(r.stdout.strip().splitlines()[-1]))
    # 10 Nov audit: the gate is on the numbers, not on the subprocess exit code, and a fixture that must fail is part of it
    fails = []
    if res["off_vs_a3c0b675_max_abs_grad_diff"] != 0.0: fails.append("options off: gradients differ from a3c0b675")
    if res["shared_init_max_abs_diff"] != 0.0: fails.append("shared layers do not start from the plain network's weights")
    fails += [f"gradient crosses between banks: {k} = {v}" for k, v in res["leak"].items() if v != 0.0]
    if res["bank1_write_from_head1_only_max_abs_diff"] != 0.0 or res["bank2_write_from_main_only_max_abs_diff"] != 0.0: fails.append("a write trace hears the other bank's head")
    if res["head1_out_switch"] is not True: fails.append("HEAD1_OUT does not return the first bank's head")
    # the fixture keeps the relay; both directions that exist in that network must show a nonzero gradient or the leak test cannot see a leak
    fx = res["leak_fixture_relay_kept"]
    fails += [f"leak fixture not detected: {k} = {v}" for k, v in fx.items() if "main_loss" in k and not v > 0.0]
    res["passed"] = not fails; res["failures"] = fails
    (HERE / "exp45" / "E18_CHECK.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    sys.exit(0 if not fails else 1)
