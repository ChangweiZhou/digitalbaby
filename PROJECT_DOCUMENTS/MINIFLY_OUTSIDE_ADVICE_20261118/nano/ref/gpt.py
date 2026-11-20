#!/usr/bin/env python3
"""A GPT taught by back-propagation on the CPU, written here in the manner of nanoGPT (PLAN_1.md, item 3). No code was fetched.
    python3 gpt.py <label> [layers=4] [heads=4] [width=128] [block=64] [dropout=0] [batch=12] [steps=2000] [lr=1e-3]
                   [decay=<steps>] [warmup=100] [minlr=1e-4] [every=250] [evalbatches=200] [finalbatches=200] [threads=4] [seed=1337]
                   [exact=1: every how many bytes the exact stream loss is taken; 0: not] [time=1: a timing test only]
As in nanoGPT's train.py and its config train_shakespeare_char.py: pre-norm blocks, GELU, no bias terms, the byte table tied to the
answering layer, learned positions; AdamW (0.9, 0.99), weight decay 0.1 on the matrices, gradient clipped at 1, warm-up then a
cosine fall to minlr; batches are random blocks of the stream. Scores: score.py (the stream loss) and nanoGPT's own estimate
(the mean loss over `evalbatches` batches of random blocks of the last tenth, every position of each block counted)."""
import json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import score as SC

class Block(nn.Module):
    def __init__(s, w, heads, layers, dropout):
        super().__init__(); s.heads = heads; s.dropout = dropout
        s.ln1 = nn.LayerNorm(w, bias=False); s.qkv = nn.Linear(w, 3 * w, bias=False); s.proj = nn.Linear(w, w, bias=False)
        s.ln2 = nn.LayerNorm(w, bias=False); s.fc = nn.Linear(w, 4 * w, bias=False); s.proj2 = nn.Linear(4 * w, w, bias=False)
        s.drop = nn.Dropout(dropout)
    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, s.heads, C // s.heads).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=s.dropout if s.training else 0.0, is_causal=True)
        x = x + s.drop(s.proj(a.transpose(1, 2).contiguous().view(B, T, C)))
        return x + s.drop(s.proj2(F.gelu(s.fc(s.ln2(x)))))

class GPT(nn.Module):
    def __init__(s, layers=4, heads=4, width=128, block=64, dropout=0.0, S=SC.S):
        super().__init__(); s.block = block
        s.wte = nn.Embedding(S, width); s.wpe = nn.Embedding(block, width); s.drop = nn.Dropout(dropout)
        s.h = nn.ModuleList([Block(width, heads, layers, dropout) for _ in range(layers)])
        s.lnf = nn.LayerNorm(width, bias=False); s.head = nn.Linear(width, S, bias=False); s.head.weight = s.wte.weight
        for n, p in s.named_parameters():
            if p.dim() >= 2: nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * layers) if n.endswith(("proj.weight", "proj2.weight")) else 0.02)
    def forward(s, idx):
        x = s.drop(s.wte(idx) + s.wpe(torch.arange(idx.shape[1])))
        for b in s.h: x = b(x)
        return s.head(s.lnf(x))

def get_batch(data, block, batch, gen):
    i = torch.randint(len(data) - block, (batch,), generator=gen)
    j = i[:, None] + torch.arange(block)[None, :]
    return data[j], data[j + 1]

@torch.no_grad()
def estimate(model, data, block, batch, n, seed):
    """nanoGPT's estimate_loss: the mean over n batches of random blocks; the same blocks at every call."""
    gen = torch.Generator().manual_seed(seed); model.eval(); tot = 0.0
    for _ in range(n):
        xb, yb = get_batch(data, block, batch, gen); tot += F.cross_entropy(model(xb).view(-1, SC.S), yb.view(-1)).item()
    model.train(); return tot / n

@torch.no_grad()
def stream_exact(model, every=1, batch=512):
    """Each byte of the last tenth (or every `every`-th) from exactly the `block` bytes before it."""
    model.eval(); ctx, y = SC.contexts(model.block); ctx = ctx[::every]; y = y[::every]
    P = np.concatenate([F.softmax(model(torch.from_numpy(ctx[i:i + batch]))[:, -1].double(), -1).numpy() for i in range(0, len(y), batch)])
    model.train(); return SC.score(P, y)

@torch.no_grad()
def stream_half(model, batch=64):
    """Cheaper: windows moved on by half a block; each byte of the last tenth from between block/2+1 and block bytes before it."""
    model.eval(); x, cut, _ = SC.load(); N = len(x); B = model.block; s = B // 2; P = np.zeros((N - cut, SC.S)); done = np.zeros(N - cut, bool)
    starts = []; a = cut - (B - s + 1)
    while a + B - s + 1 < N: starts.append(min(a, N - 1 - B)); a += s
    xt = torch.from_numpy(x)
    for i in range(0, len(starts), batch):
        st = np.array(starts[i:i + batch]); idx = torch.from_numpy(st[:, None] + np.arange(B)[None, :])
        pr = F.softmax(model(xt[idx])[:, B - s:].double(), -1).numpy()           # outputs at positions B-s..B-1 name bytes a+B-s+1..a+B
        for r, a0 in enumerate(st):
            t = np.arange(a0 + B - s + 1, a0 + B + 1) - cut; ok = (t >= 0) & (t < N - cut); P[t[ok]] = pr[r][ok]; done[t[ok]] = True
    assert done.all(); model.train(); return SC.score(P, x[cut:])

def run(kw):
    g = lambda k, d: type(d)(kw.get(k, d))
    layers, heads, width, block, batch, steps = g("layers", 4), g("heads", 4), g("width", 128), g("block", 64), g("batch", 12), g("steps", 2000)
    dropout, lr, minlr, warmup, decay = g("dropout", 0.0), g("lr", 1e-3), g("minlr", 1e-4), g("warmup", 100), g("decay", steps)
    every, evalb, seed, threads = g("every", 250), g("evalbatches", 200), g("seed", 1337), g("threads", 4)
    torch.set_num_threads(threads); torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    x, cut, _ = SC.load(); xt = torch.from_numpy(x); train, val = xt[:cut], xt[cut:]
    model = GPT(layers, heads, width, block, dropout)
    n_all = sum(p.numel() for p in model.parameters()); n_nopos = n_all - model.wpe.weight.numel()
    dec = [p for p in model.parameters() if p.dim() >= 2]; nodec = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=dec, weight_decay=0.1), dict(params=nodec, weight_decay=0.0)], lr=lr, betas=(0.9, 0.99))
    def rate(it):
        if it < warmup: return lr * (it + 1) / (warmup + 1)
        if it > decay: return minlr
        return minlr + 0.5 * (1 + math.cos(math.pi * (it - warmup) / (decay - warmup))) * (lr - minlr)
    out = dict(label=kw["label"], args=kw, layers=layers, heads=heads, width=width, block=block, dropout=dropout, batch=batch, steps=steps,
               seed=seed, threads=threads, numbers_that_learn=n_all, numbers_without_positions=n_nopos, curve=[])
    if int(kw.get("time", 0)) and not int(kw.get("go", 0)):            # a timing test only: no loss is looked at
        t0 = time.time()
        for it in range(steps):
            xb, yb = get_batch(train, block, batch, gen); loss = F.cross_entropy(model(xb).view(-1, SC.S), yb.view(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        print(kw["label"], n_all, "numbers;", round((time.time() - t0) / steps, 3), "s a step"); return None
    cpu0 = time.process_time(); wall0 = time.time(); cpu_eval = 0.0; model.train()
    for it in range(steps + 1):
        if it % every == 0 or it == steps:
            c = time.process_time()
            row = dict(step=it, train=estimate(model, train, block, batch, evalb, 4242), val=estimate(model, val, block, batch, evalb, 4243),
                       cpu_learning=round(c - cpu0 - cpu_eval, 1), wall=round(time.time() - wall0, 1))
            out["curve"].append(row); cpu_eval += time.process_time() - c
            print(kw["label"], row, flush=True)
            json.dump(out, open(os.path.join(SC.HERE, "res", f"gpt_{kw['label']}.partial.json"), "w"))
        if it == steps: break
        for gr in opt.param_groups: gr["lr"] = rate(it)
        xb, yb = get_batch(train, block, batch, gen); loss = F.cross_entropy(model(xb).view(-1, SC.S), yb.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    out["cpu_learning"] = round(time.process_time() - cpu0 - cpu_eval, 1); out["wall_learning_and_estimates"] = round(time.time() - wall0, 1)
    out["bytes_seen"] = steps * batch * block
    fb = int(kw.get("finalbatches", 200)); c = time.process_time()   # the final estimate always over `finalbatches` batches (nanoGPT: 200)
    out["nanogpt_estimate_final"] = dict(batches=fb, train=estimate(model, train, block, batch, fb, 4242), val=estimate(model, val, block, batch, fb, 4243))
    out["cpu_final_estimate"] = round(time.process_time() - c, 1)
    out["nanogpt_estimate_lowest_val"] = min(out["curve"], key=lambda r: r["val"])
    c = time.process_time()
    out["stream_half"] = stream_half(model)
    ex = int(kw.get("exact", 1))
    if ex: out["stream_exact"] = dict(every=ex, **stream_exact(model, every=ex))
    out["cpu_scoring"] = round(time.process_time() - c, 1)
    ck = kw.get("save")
    if ck: torch.save(model.state_dict(), ck)
    json.dump(out, open(os.path.join(SC.HERE, "res", f"gpt_{kw['label']}.json"), "w"), indent=1)
    os.remove(os.path.join(SC.HERE, "res", f"gpt_{kw['label']}.partial.json"))
    print(kw["label"], {k: v for k, v in out.items() if k not in ("curve", "args")}, flush=True); return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; os.makedirs(os.path.join(SC.HERE, "res"), exist_ok=True); run(kw)
