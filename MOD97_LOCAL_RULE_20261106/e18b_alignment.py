"""E18b, before any run: does the short-trace local rule point the same way as the true gradient of its own objective, and does that
change when the target is every next byte instead of the answer bytes? Read-only, at the random initialisation and at a backprop birth.

For the local-heads package (no gradient between banks) and each loss mask, per parameter group: cosine between the rule's update and
the exact gradient of the same two-head objective (backprop through time inside each bank), for
    short   weight signs, credit within the step, exact trace on the write synapses only   (arm B3)
    long    weight signs, credit through time at every layer inside each bank               (arm B2)
Writes exp45/E18B_ALIGNMENT.json."""
import json, os
os.environ.update(E16_ARCH="bank2", E16_BOOKDECAY="off", E16_HEADS="local")
import torch as T
from torch.nn.functional import cross_entropy
import exp16_integrated as G, exp11_cell as E, exp45 as X, arch17

GROUPS = ("book", "l1", "lp", "lc", "lw", "l2a", "lma", "l3a", "lp2", "lc2", "lw2", "lm", "lm2", "l2", "l3")
def grads(m):
    return {g: T.cat([(p.grad if p.grad is not None else T.zeros_like(p)).flatten() for n, p in m.named_parameters() if n.startswith(g + ".")]).clone() for g in GROUPS}
def run(m, x, y, mask, local, tt, scan):
    lg, it = arch17.forward(m, x, local, tt, keep=scan)
    m.zero_grad(); (cross_entropy(lg[mask], y[mask]) + cross_entropy(it["logits1"][mask], y[mask])).backward()
    if scan:
        arch17.TRACEUP, arch17.FB_SIGN = False, True; arch17.trace_grads_scan(m, it)
    return grads(m)
out = {}
for where in ("random initialisation", "backprop local-heads birth (B1)"):
    for seed in (73129, 93683, 116089):
        m = E.new_net(seed)
        if where.startswith("backprop"):
            m.load_state_dict(T.load(G.OUT / f"e17_dev_bank2_nbd_lh_{seed}.pt"))
        bt = E.batches([(t, E.split(t, .5, seed)[0]) for t in E.DEV], seed + 5, E.KMAX["M"])
        items = [it for _ in range(4) for it in next(bt)]; x, y, ans = X.batch_tensor(items)
        for loss, mask in (("answer bytes", ans), ("every next byte", y != E.BOS)):
            ref = run(m, x, y, mask, False, True, False)
            for rule, g in (("short", run(m, x, y, mask, True, False, True)), ("long", run(m, x, y, mask, True, True, False))):
                out.setdefault(where, {}).setdefault(loss, {}).setdefault(rule, {})[seed] = {k: float(T.dot(g[k], ref[k]) / (g[k].norm() * ref[k].norm() + 1e-30)) for k in GROUPS}
(G.OUT / "E18B_ALIGNMENT.json").write_text(json.dumps(out, indent=1))
for where, a in out.items():
    print("==", where)
    for loss, b in a.items():
        for rule, c in b.items():
            print(f"  {loss:16s} {rule:5s}", " ".join(f"{k}:" + "/".join(f"{c[s][k]:+.2f}" for s in c) for k in ("book", "l1", "lp", "lc", "lw", "lp2", "lc2")))
