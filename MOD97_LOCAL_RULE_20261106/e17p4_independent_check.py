"""E17 Phase 4, an independent check of the every-layer arm (10 Nov audit: e17p4_check.py compared the option with a call into the same
implementation, which shows dispatch, not correctness). Here the forward pass and the backward pass with sign(W) * mean|W| at every relay
are written out by hand with no autograd, in float64, and compared with what the harness computes (E16_TRACEUP=all) at the three births."""
import json, os, sys
os.environ.update(E16_ARCH="bank2", E16_BOOKDECAY="off", E16_REPLAY="raw", E16_CAP="65536", E16_TRACEUP="all")
from random import Random
import torch as T
from torch.nn.functional import cross_entropy
import exp16_integrated as G
import arch17, exp11_cell as E, exp45 as X

S = lambda W: W.sign() * W.abs().mean()
out = {}
for seed in (73129, 93683, 116089):
    m = arch17.new_net(seed, "bank2"); m.load_state_dict(T.load(G.OUT / f"e17_dev_bank2_nbd_{seed}.pt")); m = m.double()
    rng = Random(seed + 9); ent = G.family_entries("F1", seed)
    x, y, mask = X.batch_tensor([G.render(rng.choice(ent), rng) for _ in range(64)])
    lg, _ = E.forward(m, x, local=True, through_time=False, keep=True)          # the harness call for arm T; the option turns it into the every-layer arm
    m.zero_grad(); cross_entropy(lg[mask], y[mask]).backward(); E.trace_grads_scan(m, None)
    ref = {n: p.grad.clone() for n, p in m.named_parameters()}
    with T.no_grad():
        P = dict(m.named_parameters()); B, L = x.shape; R = m.R
        w_, b_ = lambda n: P[n + ".weight"], lambda n: P[n + ".bias"]
        u = P["book.weight"][x]
        z1 = T.tanh(u @ w_("l1").t() + b_("l1")); p = T.tanh(z1 @ w_("lp").t() + b_("lp"))
        c = T.tanh(p @ w_("lc").t() + b_("lc")); w = T.sigmoid(p @ w_("lw").t() + b_("lw"))
        Hs = T.zeros(B, L, c.shape[2], dtype=T.float64); h = T.zeros(B, c.shape[2], dtype=T.float64)
        for t in range(L):
            h = R * h + w[:, t] * c[:, t]; Hs[:, t] = h
        p2 = T.tanh(Hs @ w_("lp2").t() + b_("lp2")); c2 = T.tanh(p2 @ w_("lc2").t() + b_("lc2")); w2 = T.sigmoid(p2 @ w_("lw2").t() + b_("lw2"))
        Hs2 = T.zeros_like(Hs); h = T.zeros(B, c.shape[2], dtype=T.float64)
        for t in range(L):
            h = R * h + w2[:, t] * c2[:, t]; Hs2[:, t] = h
        z2 = T.tanh(z1 @ w_("l2").t() + b_("l2") + Hs @ w_("lm").t() + b_("lm") + Hs2 @ w_("lm2").t() + b_("lm2"))
        logits = z2 @ w_("l3").t() + b_("l3")
        g = T.softmax(logits, 2); g.scatter_add_(2, y.clamp(max=256).unsqueeze(2), -T.ones(B, L, 1, dtype=T.float64)); g = g * mask.unsqueeze(2) / mask.sum()
        mine = {}
        def lin_grad(name, d, inp):
            mine[name + ".weight"] = T.einsum("blo,bli->oi", d, inp); mine[name + ".bias"] = d.sum((0, 1))
        def scan(d):
            Lam = T.zeros_like(d); acc = T.zeros_like(d[:, 0])
            for t in range(L - 1, -1, -1):
                acc = d[:, t] + R * acc; Lam[:, t] = acc
            return Lam
        lin_grad("l3", g, z2)
        dpre = (g @ S(w_("l3"))) * (1 - z2 * z2)
        lin_grad("l2", dpre, z1); lin_grad("lm", dpre, Hs); lin_grad("lm2", dpre, Hs2)
        dz1 = dpre @ S(w_("l2")); dHs = dpre @ S(w_("lm"))
        Lam2 = scan(dpre @ S(w_("lm2")))
        dac2 = Lam2 * w2 * (1 - c2 * c2); daw2 = Lam2 * c2 * w2 * (1 - w2)
        lin_grad("lc2", dac2, p2); lin_grad("lw2", daw2, p2)
        dap2 = (dac2 @ S(w_("lc2")) + daw2 @ S(w_("lw2"))) * (1 - p2 * p2)
        lin_grad("lp2", dap2, Hs)
        Lam1 = scan(dHs + dap2 @ S(w_("lp2")))
        dac = Lam1 * w * (1 - c * c); daw = Lam1 * c * w * (1 - w)
        lin_grad("lc", dac, p); lin_grad("lw", daw, p)
        dap = (dac @ S(w_("lc")) + daw @ S(w_("lw"))) * (1 - p * p)
        lin_grad("lp", dap, z1)
        da1 = (dz1 + dap @ S(w_("lp"))) * (1 - z1 * z1)
        lin_grad("l1", da1, u)
        gb = T.zeros_like(P["book.weight"]); gb.index_add_(0, x.reshape(-1), (da1 @ S(w_("l1"))).reshape(B * L, -1)); gb[E.BOS] = 0      # the harness's embedding has padding_idx = BOS, so the start/pad row never receives a gradient; reproduced here and reported
        mine["book.weight"] = gb
    out[seed] = {n: float((mine[n] - ref[n]).abs().max() / ref[n].abs().max().clamp_min(1e-300)) for n in ref}
    out[seed]["max"] = max(out[seed].values())
    print(seed, "max relative error, hand-written against harness:", f"{out[seed]['max']:.2e}", flush=True)
ok = all(v["max"] < 1e-9 for v in out.values())
(G.OUT / "E17P4_INDEPENDENT_CHECK.json").write_text(json.dumps(dict(passed=ok, per_seed=out), indent=1))
sys.exit(0 if ok else 1)
