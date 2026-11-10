"""E17 Phase 4 check (10 Nov 2026): the harness option E16_TRACEUP=all against what it should compute, and its pace.

1. With the option set, one E.update-style gradient on a first-block batch must equal, on the layers downstream of the second bank's write
   (lc2, lw2, lm, lm2, l2, l3), the plain sign rule's gradient (scan trace): those layers already had through-time credit under sign feedback.
2. On the upstream layers it must equal a direct computation of sign-feedback credit through time (forward kept through time, backward
   through sign(W)): by construction the same call, checked here through the harness's own patched E.forward.
3. ms per update for the option, the plain sign rule and backprop (10 updates each, batch of 64).
Run with E16_TRACEUP unset for the reference gradients and with E16_TRACEUP=all for the option; this script does both in two processes.
"""
import json, os, subprocess, sys
if len(sys.argv) > 1:
    os.environ.update(E16_ARCH="bank2", E16_BOOKDECAY="off", E16_REPLAY="raw", E16_CAP="65536")
    if sys.argv[1] == "all":
        os.environ["E16_TRACEUP"] = "all"
    import time
    from random import Random
    import torch as T
    from torch.nn.functional import cross_entropy
    import exp16_integrated as G
    import arch17, exp11_cell as E, exp45 as X
    seed = 73129
    m = arch17.new_net(seed, "bank2"); m.load_state_dict(T.load(G.OUT / f"e17_dev_bank2_nbd_{seed}.pt"))
    rng = Random(seed + 9); ent = G.family_entries("F1", seed)
    items = [G.render(rng.choice(ent), rng) for _ in range(64)]
    x, y, mask = X.batch_tensor(items)
    lg, it = E.forward(m, x, local=True, through_time=False, keep=True)          # exactly E.update's call for arm T
    m.zero_grad(); cross_entropy(lg[mask], y[mask]).backward(); E.trace_grads_scan(m, it)
    g = {n: p.grad.clone() for n, p in m.named_parameters()}
    T.save(g, f"/tmp/e17p4_check_{sys.argv[1]}.pt")
    if sys.argv[1] == "all":
        lg, _ = arch17.forward(m, x, True, True); m.zero_grad(); cross_entropy(lg[mask], y[mask]).backward()
        direct = {n: p.grad.clone() for n, p in m.named_parameters()}
        print(json.dumps({"option_vs_direct_max_abs": max(float((g[n] - direct[n]).abs().max()) for n in g)}))
    opt = G.adamw16(m, .001, .25)
    ms = {}
    for arm in (("T",) if sys.argv[1] == "all" else ("T", "D")):
        t0 = time.process_time()
        for _ in range(10):
            E.update(m, opt, items, arm)
        ms[arm] = (time.process_time() - t0) * 100
    print(json.dumps({"ms_per_update_" + sys.argv[1]: ms}))
else:
    import torch as T
    out = {}
    for mode in ("plain", "all"):
        r = subprocess.run([sys.executable, __file__, mode], capture_output=True, text=True); assert r.returncode == 0, r.stderr
        for line in r.stdout.strip().splitlines():
            out.update(json.loads(line))
    a, b = T.load("/tmp/e17p4_check_plain.pt"), T.load("/tmp/e17p4_check_all.pt")
    down = [n for n in a if n.split(".")[0] in ("lc2", "lw2", "lm", "lm2", "l2", "l3")]
    up = [n for n in a if n not in down]
    out["downstream_plain_vs_option_max_rel_err"] = {n: float((a[n] - b[n]).abs().max() / a[n].abs().max()) for n in down}
    out["upstream_cosine_plain_vs_option"] = {n: float(T.dot(a[n].flatten(), b[n].flatten()) / (a[n].norm() * b[n].norm() + 1e-30)) for n in up if n.endswith("weight")}
    from pathlib import Path
    (Path(__file__).resolve().parent / "exp45" / "E17P4_CHECK.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
    os.remove("/tmp/e17p4_check_plain.pt"); os.remove("/tmp/e17p4_check_all.pt")
