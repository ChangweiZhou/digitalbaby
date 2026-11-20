#!/usr/bin/env python3
"""The loop's network taught by back-propagation (PLAN_1.md, item 4), and a GRU (optional item).
    python3 onelayer.py <label> [H=256] [D=10] [passes=12] [batch=256] [lr=1e-3] [threads=2] [seed=1337]
    python3 onelayer.py <label> gru=1 [H=256] [passes=20] [lanes=32] [bptt=64] [lr=2e-3] [threads=1] [seed=1337]
One layer: the last D bytes one-hot (65 D lines), H logistic cells, a softmax answering layer; minibatches, Adam, a cosine fall of the
rate to a tenth. This is a ceiling for the shape by back-propagation, not a twin of the loop's rule.
The stream's last 50,000 bytes are a development part that is not learned from: after every pass the loss on it is taken, and the
strengths of the pass with the lowest development loss are the ones scored on the last tenth (the strengths of the last pass are
scored as well). The last tenth takes no part in any choice."""
import copy, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import score as SC
DEV = 50000

class OneLayer(nn.Module):
    def __init__(s, H, D):
        super().__init__(); s.D = D; s.w1 = nn.EmbeddingBag(SC.S * D, H, mode="sum"); s.b1 = nn.Parameter(torch.zeros(H)); s.out = nn.Linear(H, SC.S)
        nn.init.normal_(s.w1.weight, 0.0, 1 / math.sqrt(D)); s.off = torch.arange(D) * SC.S
    def forward(s, ctx): return s.out(torch.sigmoid(s.w1(ctx + s.off) + s.b1))

class GRU(nn.Module):
    def __init__(s, H):
        super().__init__(); s.g = nn.GRU(SC.S, H, batch_first=True); s.out = nn.Linear(H, SC.S)
    def forward(s, x, h=None):
        o, h = s.g(F.one_hot(x, SC.S).float(), h); return s.out(o), h

@torch.no_grad()
def probs_window(model, lo, hi, batch=4096):
    model.eval(); ctx, y = SC.contexts(model.D, lo, hi)
    P = np.concatenate([F.softmax(model(torch.from_numpy(ctx[i:i + batch])).double(), -1).numpy() for i in range(0, len(y), batch)])
    model.train(); return SC.score(P, y)

@torch.no_grad()
def probs_gru(model, x, lo, hi, warm=5000):
    """The state is carried along the text: `warm` bytes before lo are read first, then every byte lo..hi-1 is named before it arrives."""
    model.eval(); xt = torch.from_numpy(x[lo - warm - 1: hi - 1])[None]; o, _ = model(xt)
    P = F.softmax(o[0, warm:].double(), -1).numpy(); model.train(); return SC.score(P, x[lo:hi])

def run(kw):
    g = lambda k, d: type(d)(kw.get(k, d)); gru = int(kw.get("gru", 0))
    H, D, seed = g("H", 256), g("D", 10), g("seed", 1337); passes = g("passes", 20 if gru else 12); lr = g("lr", 2e-3 if gru else 1e-3)
    batch, lanes, bptt, threads = g("batch", 256), g("lanes", 32), g("bptt", 64), g("threads", 1 if gru else 2)
    torch.set_num_threads(threads); torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    x, cut, _ = SC.load(); xt = torch.from_numpy(x); N = len(x); end = cut - DEV            # learning part: bytes 0..end-1
    model = GRU(H) if gru else OneLayer(H, D); n_all = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if gru:
        L = (end - 1) // lanes; per = (L - 1) // bptt; starts = torch.arange(lanes) * L
    else:
        per = (end - D) // batch
    total = passes * per; out = dict(label=kw["label"], args=kw, kind="gru" if gru else "one layer", H=H, D=None if gru else D, seed=seed, threads=threads,
                                     numbers_that_learn=n_all, passes=passes, learning_part=end, development_part=DEV, rows=[])
    cpu0 = time.process_time(); cpu_eval = 0.0; best = None; it = 0
    for p in range(passes):
        if gru: h = None
        else: perm = torch.randperm(end - D, generator=gen) + D
        for b in range(per):
            for gr in opt.param_groups: gr["lr"] = 0.1 * lr + 0.45 * lr * (1 + math.cos(math.pi * it / total))
            if gru:
                idx = starts[:, None] + b * bptt + torch.arange(bptt)[None, :]; o, h = model(xt[idx], h); h = h.detach()
                loss = F.cross_entropy(o.reshape(-1, SC.S), xt[idx + 1].reshape(-1))
            else:
                t = perm[b * batch:(b + 1) * batch]; ctx = xt[t[:, None] + torch.arange(-D, 0)[None, :]]
                loss = F.cross_entropy(model(ctx), xt[t])
            opt.zero_grad(set_to_none=True); loss.backward()
            if gru: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); it += 1
        c = time.process_time()
        dev = probs_gru(model, x, end, cut) if gru else probs_window(model, end, cut)
        row = dict(passes=p + 1, development=dev, cpu_learning=round(c - cpu0 - cpu_eval, 1)); out["rows"].append(row)
        if best is None or dev["nats"] < best[1]["nats"]: best = (p + 1, dev, copy.deepcopy(model.state_dict()))
        cpu_eval += time.process_time() - c; print(kw["label"], row, flush=True)
    out["cpu_learning"] = round(time.process_time() - cpu0 - cpu_eval, 1)
    out["bytes_seen"] = passes * per * (lanes * bptt if gru else batch)
    sc = (lambda: probs_gru(model, x, cut, N)) if gru else (lambda: probs_window(model, cut, N))
    out["last_pass"] = dict(passes=passes, stream=sc())
    model.load_state_dict(best[2]); out["chosen"] = dict(passes=best[0], bytes_seen=out["bytes_seen"] * best[0] // passes, development=best[1], stream=sc())
    json.dump(out, open(os.path.join(SC.HERE, "res", f"{'gru' if gru else 'onelayer'}_{kw['label']}.json"), "w"), indent=1)
    print(kw["label"], {k: v for k, v in out.items() if k not in ("rows", "args")}, flush=True); return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; os.makedirs(os.path.join(SC.HERE, "res"), exist_ok=True); run(kw)
