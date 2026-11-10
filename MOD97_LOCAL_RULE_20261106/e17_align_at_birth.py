"""E17 diagnostic (10 Nov 2026, reads only): at the birth of the two-bank learner, how well does each rule's update direction
line up with the true gradient (backprop through time), layer by layer, on first-block items (mod-97 operations with distractors)?

For each seed and each rule variant: cosine between the rule's gradient and backprop's, and the ratio of their norms, per parameter
group. No weights are changed and nothing is trained. Writes exp45/E17_ALIGN_AT_BIRTH.{json,md}.

Variants
    sign        the sign rule as run in Phase 2 (sign-of-weight feedback, credit within the step, exact trace on the write synapses)
    sign_tup    the same with the trace extended to the projection feeding each write (E16_TRACEUP=proj)
    exact       exact feedback, within-step credit, trace on the write synapses (E16_FEEDBACK=exact)
    exact_tup   exact feedback and the extended trace
    sign_time   sign-of-weight feedback with credit carried through time at EVERY layer (not a harness arm yet): the forward pass kept
                through time and the backward pass sent through sign(W) at each layer
"""
import json, os, sys
os.environ.update(E16_ARCH="bank2", E16_BOOKDECAY="off", E16_REPLAY="raw", E16_CAP="65536")
from random import Random
import torch as T
from torch.nn.functional import cross_entropy
import exp16_integrated as G
import arch17, exp11_cell as E, exp45 as X

GROUPS = ("book", "l1", "lp", "lc", "lw", "lp2", "lc2", "lw2", "lm", "lm2", "l2", "l3")
SEEDS = (73129, 93683, 116089)


def grads(m):
    return {g: T.cat([p.grad.flatten() if p.grad is not None else T.zeros(p.numel()) for n, p in m.named_parameters() if n.startswith(g + ".")]).clone() for g in GROUPS}


def variant(m, x, y, mask, name):
    arch17.TRACEUP = name.endswith("_tup"); arch17.FB_SIGN = name.startswith("sign")
    m.zero_grad()
    if name == "bptt":
        lg, _ = arch17.forward(m, x, False, True)
        cross_entropy(lg[mask], y[mask]).backward()
    elif name == "sign_time":
        lg, _ = arch17.forward(m, x, True, True)
        cross_entropy(lg[mask], y[mask]).backward()
    else:
        lg, it = arch17.forward(m, x, name.startswith("sign"), False, keep=True)
        cross_entropy(lg[mask], y[mask]).backward()
        arch17.trace_grads_scan(m, it)
    return grads(m)


out = {}
for seed in SEEDS:
    m = arch17.new_net(seed, "bank2")
    m.load_state_dict(T.load(G.OUT / f"e17_dev_bank2_nbd_{seed}.pt"))
    rng = Random(seed + 9)
    ent = G.family_entries("F1", seed)
    items = [G.render(rng.choice(ent), rng) for _ in range(256)]
    x, y, mask = X.batch_tensor(items)
    ref = variant(m, x, y, mask, "bptt")
    out[seed] = {}
    for v in ("sign", "sign_tup", "exact", "exact_tup", "sign_time"):
        g = variant(m, x, y, mask, v)
        out[seed][v] = {k: dict(cos=float(T.dot(g[k], ref[k]) / (g[k].norm() * ref[k].norm() + 1e-30)), norm_ratio=float(g[k].norm() / (ref[k].norm() + 1e-30))) for k in GROUPS}
(G.OUT / "E17_ALIGN_AT_BIRTH.json").write_text(json.dumps(out, indent=1))
L = ["# E17: alignment of each rule's update with the true gradient at birth (two-bank learner, first-block items, 256 per seed)", "",
     "Cosine with backprop through time, seeds 73129 / 93683 / 116089; in brackets the norm ratio (rule / backprop), mean of the three.", ""]
for v in ("sign", "sign_tup", "exact", "exact_tup", "sign_time"):
    L += [f"## {v}", "", "| Group | Cosine | Norm ratio |", "|---|---|---:|"]
    for k in GROUPS:
        L.append(f"| {k} | " + " / ".join(f"{out[s][v][k]['cos']:+.2f}" for s in SEEDS) + f" | {sum(out[s][v][k]['norm_ratio'] for s in SEEDS) / 3:.2f} |")
    L.append("")
(G.OUT / "E17_ALIGN_AT_BIRTH.md").write_text("\n".join(L)); print("\n".join(L))
