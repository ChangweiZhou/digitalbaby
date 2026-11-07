"""E6-route: learned-key routing on the canonical scratchpad (E5-C protocol, 18-byte window).

RouteNet: four attention heads over the window slots. Query = learned function of the whole window;
keys = learned slot-position embeddings; value = the slot's byte embedding. The output head sees the
four attended bytes plus a small 16-d window summary (variant "att") or the attended bytes only ("attv").
Every linear map goes through exp3_state.lin, so the sign rule applies to all projections in life.
Predictions: MOD97_LOCAL_RULE_20261106/E45_PREDICTIONS.md (E6-route section).
"""
import os
os.environ.setdefault("EXP45_WINDOW", "18")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import json
import sys
import time
from pathlib import Path
from random import Random

import torch as T
from torch import nn
from torch.nn.functional import cross_entropy, pad

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp45 as X
from exp3_state import lin

T.set_num_threads(1)
assert X.W == 18
W, EOS, BOS = X.W, X.EOS, X.BOS
OUT = X.OUT
SEEDS = X.SEEDS
ARMS = ("att", "attv", "attsh", "attbp", "attvbp")


class RouteNet(nn.Module):
    def __init__(self, heads=4, summary=16, H=128):
        super().__init__()
        self.heads, self.summary = heads, summary
        self.book = nn.Embedding(258, 8, padding_idx=BOS)
        self.pos = nn.Parameter(T.randn(W, 16) / 4)
        self.q = nn.Linear(8 * W, heads * 16)
        self.sum = nn.Linear(8 * W, summary) if summary else None
        self.h1 = nn.Linear(heads * 8 + summary, H)
        self.h2 = nn.Linear(H, H)
        self.l3 = nn.Linear(H, 257)

    def forward(self, x, local=False):
        win = pad(x, (W - 1, 0), value=BOS).unfold(1, W, 1)          # B x L x W
        u = self.book(win)                                             # B x L x W x 8
        B, L = x.shape
        flat = u.reshape(B, L, W * 8)
        q = lin(flat, self.q, local).reshape(B, L, self.heads, 16)
        scores = T.einsum("blhk,wk->blhw", q, self.pos) / 4.
        att = scores.softmax(-1)                                       # B x L x heads x W
        read = T.einsum("blhw,blwe->blhe", att, u).reshape(B, L, self.heads * 8)
        if self.sum is not None:
            read = T.cat([read, T.tanh(lin(flat, self.sum, local))], -1)
        z = T.tanh(lin(read, self.h1, local))
        z = T.tanh(lin(z, self.h2, local))
        return lin(z, self.l3, local)


def new_net(seed, summary):
    T.manual_seed(seed)
    return RouteNet(summary=summary)


def update(m, opt, items, local):
    x, y, mask = X.batch_tensor(items)
    logits = m(x, local=local)
    loss = cross_entropy(logits[mask], y[mask])
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())


@T.no_grad()
def respond(m, prompts, cap=22):
    out = [None] * len(prompts)
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)
    for L, idx in groups.items():
        for j in range(0, len(idx), 256):
            ids = idx[j:j + 256]
            p = T.tensor([[BOS, *prompts[i]] for i in ids])
            gen = [[] for _ in ids]; done = [False] * len(ids)
            for _ in range(cap):
                yy = m(p)[:, -1].argmax(1)
                for q in range(len(ids)):
                    if not done[q]:
                        gen[q].append(int(yy[q]))
                        if int(yy[q]) == EOS:
                            done[q] = True
                p = T.cat([p, yy[:, None]], 1)
                if all(done):
                    break
            for q, i in enumerate(ids):
                out[i] = gen[q]
    return out


def score(m, items, canonical=True):
    outs = respond(m, [p for p, _ in items], cap=22 if canonical else 7)
    res = dict(n=len(items), exact=sum(o == t for o, (_, t) in zip(outs, items)), final=sum(o[-3:] == t[-3:] for o, (_, t) in zip(outs, items)))
    if canonical:
        res["intermediate"] = sum(o[7:9] == t[7:9] for o, (_, t) in zip(outs, items))
        res["step1_prompt"] = sum(o[:7] == t[:7] for o, (_, t) in zip(outs, items))
        res["step2_prompt"] = sum(o[10:17] == t[10:17] for o, (_, t) in zip(outs, items))
    return res


def develop(seed, summary):
    tag = "att" if summary else "attv"
    path = OUT / f"e6_dev_{tag}_{seed}.pt"
    if path.exists():
        return
    m = new_net(seed, summary)
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=.3, betas=(.9, .98))
    base = [X.render(n, *p) for n in X.E5_DEV for p in X.split(n, .5, seed)[0]]
    rng, order = Random(seed + 21), Random(seed + 1)
    t0 = time.process_time(); curve = []
    for stop in range(X.BLOCK, X.DEV_UPDATES + 1, X.BLOCK):
        losses = [update(m, opt, X.with_left_noise(order.sample(base, X.BATCH), rng), False) for _ in range(X.BLOCK)]
        if stop % 4096 == 0 or stop == X.DEV_UPDATES:
            row = dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0,
                       sealed={n: score(m, X.sample_rows(n, X.split(n, .5, seed)[2], seed, 600), False) for n in X.E5_DEV})
            curve.append(row)
            print("e6 dev", tag, seed, stop, {n: v["exact"] for n, v in row["sealed"].items()}, flush=True)
    T.save(m.state_dict(), path)
    (OUT / f"e6_dev_{tag}_{seed}.json").write_text(json.dumps(dict(seed=seed, params=sum(p.numel() for p in m.parameters()), curve=curve), indent=1))


def life(arm, seed):
    path = OUT / f"e6_life_{arm}_{seed}.json"
    summary = 16 if arm in ("att", "attsh", "attbp") else 0
    m = new_net(seed, summary)
    m.load_state_dict(T.load(OUT / f"e6_dev_{'att' if summary else 'attv'}_{seed}.pt"))
    local = arm not in ("attbp", "attvbp")
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=(1. if local else .3), betas=(.9, .98))
    taught = {t: X.compose_triples(t, seed, 8000) for t in X.TAUGHT_TYPES}
    held = {t: X.compose_triples(t, seed + 1, 600, exclude=frozenset(taught[t])) for t in X.TAUGHT_TYPES}
    held.update({t: X.compose_triples(t, seed + 1, 600) for t in X.HELD_TYPES})
    teach = [X.render_canonical(t, *tr) for t in X.TAUGHT_TYPES for tr in taught[t]]
    panels = {t: [X.render_canonical(t, *tr) for tr in trs] for t, trs in held.items()}
    single = {n: X.sample_rows(n, X.split(n, .5, seed)[2], seed, 600) for n in X.E5_DEV}
    if arm == "attsh":
        rng = Random(seed + 5); finals = [tt[-3:-1] for _, tt in teach]; rng.shuffle(finals)
        teach = [(p, tt[:-3] + f + [EOS]) for (p, tt), f in zip(teach, finals)]
    t0 = time.process_time()
    curve = [dict(step=0, compose={t: score(m, it) for t, it in panels.items()}, single={n: score(m, it, False) for n, it in single.items()})]
    bt = X.batches(teach, seed + 3)
    for stop in range(X.BLOCK, X.STAGE_UPDATES + 1, X.BLOCK):
        losses = [update(m, opt, next(bt), local) for _ in range(X.BLOCK)]
        if stop % 2048 == 0:
            curve.append(dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0,
                              compose={t: score(m, it) for t, it in panels.items()}, single={n: score(m, it, False) for n, it in single.items()}))
            print("e6", arm, seed, stop, {t: v["final"] for t, v in curve[-1]["compose"].items()}, "steps", {t: (v["step1_prompt"], v["step2_prompt"]) for t, v in curve[-1]["compose"].items() if t in ("pm", "mp")}, f"cpu {curve[-1]['cpu']:.0f}s", flush=True)
    path.write_text(json.dumps(dict(arm=arm, seed=seed, summary=summary, shortcuts=X.compose_shortcuts(seed, panels, taught), curve=curve), indent=1))


def qualify():
    out = {}
    m = new_net(1, 16)
    out["params_att"] = sum(p.numel() for p in m.parameters()); out["params_attv"] = sum(p.numel() for p in new_net(1, 0).parameters())
    items = [X.render_canonical("pp", *tr) for tr in X.compose_triples("pp", 1, 64)]
    x, y, mask = X.batch_tensor(items)
    # sign path vs true gradient at the output; attention weights sum to one; smoke both modes
    logits = m(x, local=True); loss = cross_entropy(logits[mask], y[mask]); loss.backward()
    g = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad(); logits = m(x, local=False); loss = cross_entropy(logits[mask], y[mask]); loss.backward()
    cos = lambda a, b: float(T.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0))
    out["sign_vs_true_cosine"] = {n: round(cos(g[n], p.grad), 3) for n, p in m.named_parameters()}
    assert out["sign_vs_true_cosine"]["l3.weight"] > .999
    opt = T.optim.AdamW(m.parameters(), lr=.001)
    for local in (True, False):
        l = update(m, opt, items, local); assert l == l
    out["smoke"] = l
    (OUT / "QUALIFY_E6.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if sys.argv[1] == "qualify":
        qualify()
    elif sys.argv[1] == "dev":
        develop(int(sys.argv[2]), 16 if sys.argv[3] == "att" else 0)
    elif sys.argv[1] == "life":
        life(sys.argv[2], int(sys.argv[3]))
