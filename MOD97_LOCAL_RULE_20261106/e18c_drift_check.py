"""E18c, a read-only check of one hypothesis (10 Nov 2026). No training; no weight is changed.

Observed (review 9.88): with every next byte as the target, backprop on the learner's pipeline keeps the skill and the sign-feedback
sweep loses all of it. Hypothesis: most bytes under that loss are prompt bytes that cannot be predicted. At a solution, the true
gradient of their loss averages to nothing over batches. The sweep relays error through sign(W) * mean|W|, which is not the gradient,
so the same irreducible error need not average out and can push the weights in a steady direction.

Measurement. At the end-of-life weights of the backprop every-byte lives (a solution of that objective), over 128 batches of 64 items
drawn equally from the three families: the gradient of the every-byte loss split into its prompt-byte part and its answer-byte part,
computed by backprop through time and by the sweep. For each parameter group and each part:
    coherence = norm of the mean over batches / mean over batches of the norm.
Pure noise gives about 1 / sqrt(128) = 0.09; a steady push gives a value well above that.

PREDICTION, written before the first run: for the layers below the second bank's write (embeddings, first layer, both projections,
first bank's write), the sweep's coherence on the PROMPT-byte part is at least twice backprop's on the same batches, in at least two of
the three seeds. If it is not, the hypothesis is not supported by this measurement, and the cause of the collapse is still open.
Writes exp45/E18C_DRIFT_CHECK.json."""
import json, os
os.environ.update(E16_ARCH="bank2", E16_BOOKDECAY="off", E16_REPLAY="raw", E16_CAP="65536")
from random import Random
import torch as T
from torch.nn.functional import cross_entropy
import exp16_integrated as G, exp11_cell as E, exp45 as X, arch17

GROUPS = {"below the second bank's write (book, l1, lp, lc, lw, lp2)": ("book", "l1", "lp", "lc", "lw", "lp2"), "second bank's write (lc2, lw2)": ("lc2", "lw2"), "readout (lm, lm2, l2, l3)": ("lm", "lm2", "l2", "l3")}
NB = 128
def flat(m, names):
    return T.cat([(p.grad if p.grad is not None else T.zeros_like(p)).flatten() for n, p in m.named_parameters() if n.split(".")[0] in names])
out = {}
for seed in (73129, 93683, 116089):
    m = arch17.new_net(seed, "bank2"); m.load_state_dict(T.load(G.OUT / f"e17_life_M_bank2_nbd_raw65536_lossall_{seed}.pt"))
    rng = Random(seed + 77); ents = [G.family_entries(f, seed) for f in ("F1", "F2", "F3")]
    acc = {(rule, part, g): [T.zeros(1), 0.0] for rule in ("backprop", "sweep") for part in ("prompt bytes", "answer bytes") for g in GROUPS}
    for b in range(NB):
        items = [G.render(rng.choice(ents[i % 3]), rng) for i in range(64)]
        x, y, ans = X.batch_tensor(items); allm = y != E.BOS; n_all = int(allm.sum())
        for rule, local in (("backprop", False), ("sweep", True)):
            for part, mask in (("prompt bytes", allm & ~ans), ("answer bytes", ans)):
                lg, _ = arch17.forward(m, x, local, True)
                m.zero_grad(); (cross_entropy(lg[mask], y[mask], reduction="sum") / n_all).backward()
                for g, names in GROUPS.items():
                    v = flat(m, names); a = acc[(rule, part, g)]; a[0] = a[0] + v; a[1] += float(v.norm())
    res = {}
    for (rule, part, g), (s, n) in acc.items():
        res.setdefault(g, {}).setdefault(part, {})[rule] = dict(coherence=float((s / NB).norm() / (n / NB)), norm_of_mean=float((s / NB).norm()), mean_norm=n / NB)
    out[seed] = res
    print(seed, {g: {part: {rule: round(v["coherence"], 3) for rule, v in d.items()} for part, d in r.items()} for g, r in res.items()}, flush=True)
key = "below the second bank's write (book, l1, lp, lc, lw, lp2)"
ratios = {s: out[s][key]["prompt bytes"]["sweep"]["coherence"] / out[s][key]["prompt bytes"]["backprop"]["coherence"] for s in out}
verdict = "holds" if sum(r >= 2 for r in ratios.values()) >= 2 else "misses"
json.dump(dict(batches=NB, noise_floor=NB ** -0.5, prediction="sweep coherence on prompt bytes at least twice backprop's, below the second bank's write, in two of three seeds",
               ratio_sweep_to_backprop=ratios, verdict=verdict, per_seed={str(k): v for k, v in out.items()}), open(G.OUT / "E18C_DRIFT_CHECK.json", "w"), indent=1)
print("ratio sweep / backprop, prompt bytes, below the second bank's write:", {k: round(v, 2) for k, v in ratios.items()}, "| prediction", verdict)
