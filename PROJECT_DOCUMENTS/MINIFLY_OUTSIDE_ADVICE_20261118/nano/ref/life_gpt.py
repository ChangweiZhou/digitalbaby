#!/usr/bin/env python3
"""PLAN_1.md, addition 2: the small GPT through a life of two texts, beside nano/table's PLAN_5.md (asked for by the coordinating session).
    python3 life_gpt.py [threads=1] [stepsA=10000] [seed=1337]         -> res/life_gpt.json
A = the stream of the test. B = the first nine tenths of three other texts (Bryant, Burgess, Carroll), prepared exactly as in
nano/table/life.py, whose load_all() is copied here (a byte not among A's 65 values becomes a space; carriage returns removed).
The network, optimiser and recipe are those of gpt.py's small setting. On B it reads random blocks of B's stream only: no byte of A is shown again.
Every scoring is done with nothing changing: the exact stream loss, each byte from the 64 bytes before it in its own text."""
import json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
import score as SC, gpt as G
DATA = os.path.join(SC.HERE, "..", "..", "..", "..", "minifly_v148_pretrain_probe_dev_20261025", "data")
BT = ("bryant-stories.txt", "burgess-busterbrown.txt", "carroll-alice.txt")

def load_all():                                              # copied from nano/table/life.py (claude/friendly-darwin-8mmhfm at fcc174a)
    xa, cut, _ = SC.load(); vals = SC.byte_values(); tab = np.full(256, 1, np.int64); tab[vals] = np.arange(SC.S); parts = [("A", xa)]; replaced = {}
    for f in BT:
        raw = np.frombuffer(open(os.path.join(DATA, f), "rb").read(), np.uint8); raw = raw[raw != 13]; parts.append((f.split(".")[0], tab[raw])); replaced[f.split(".")[0]] = round(100 * float((~np.isin(raw, vals)).mean()), 2)
    x = np.concatenate([p[1] for p in parts]); segs = {}; off = 0
    for name, p in parts:
        c = cut if name == "A" else int(0.9 * len(p)); segs[name] = (off, off + c, off + len(p)); off += len(p)
    return x, segs, replaced

@torch.no_grad()
def scores(model, x, segs, batch=512):
    model.eval(); out = {}; W = model.block
    for name, (a, c, b) in segs.items():
        idx = np.arange(c, b)[:, None] + np.arange(-W, 0)[None, :]
        P = np.concatenate([F.softmax(model(torch.from_numpy(x[idx[i:i + batch]]))[:, -1].double(), -1).numpy() for i in range(0, b - c, batch)])
        r = SC.score(P, x[c:b]); out[name] = dict(nats=r["nats"], right=r["right"], n=int(b - c))
    bn = [k for k in segs if k != "A"]; tot = sum(out[k]["n"] for k in bn)
    out["B"] = dict(nats=sum(out[k]["nats"] * out[k]["n"] for k in bn) / tot, right=sum(out[k]["right"] * out[k]["n"] for k in bn) / tot, n=tot)
    model.train(); return out

def learn(model, opt, data, steps, gen, rate, batch=12):
    for it in range(steps):
        for gr in opt.param_groups: gr["lr"] = rate(it)
        xb, yb = G.get_batch(data, model.block, batch, gen); loss = F.cross_entropy(model(xb).view(-1, SC.S), yb.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

def fresh(steps, lr=1e-3, minlr=1e-4, warmup=100):          # gpt.py's schedule over a phase of `steps`
    return lambda it: lr * (it + 1) / (warmup + 1) if it < warmup else minlr + 0.5 * (1 + math.cos(math.pi * (it - warmup) / (steps - warmup))) * (lr - minlr)

def new(seed):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed); model = G.GPT(4, 4, 128, 64, 0.0)
    dec = [p for p in model.parameters() if p.dim() >= 2]; nodec = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=dec, weight_decay=0.1), dict(params=nodec, weight_decay=0.0)], lr=1e-3, betas=(0.9, 0.99)); return model, opt, gen

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:]); seed = int(kw.get("seed", 1337)); torch.set_num_threads(int(kw.get("threads", 1))); t0 = time.process_time()
    x, segs, replaced = load_all(); xt = torch.from_numpy(x); A = xt[segs["A"][0]:segs["A"][1]]; Bs = torch.cat([xt[segs[k][0]:segs[k][1]] for k in segs if k != "A"])
    stepsA = int(kw.get("stepsA", 10000)); stepsB = round(stepsA * len(Bs) / len(A))
    r = dict(seed=seed, segments={k: list(map(int, v)) for k, v in segs.items()}, replaced_percent=replaced, steps_A=stepsA, steps_B=stepsB, batch=12, block=64,
             bytes_seen_A=stepsA * 768, bytes_seen_B=stepsB * 768, stream_A=len(A), stream_B=len(Bs))
    model, opt, gen = new(seed); learn(model, opt, A, stepsA, gen, fresh(stepsA)); r["after_A"] = scores(model, x, segs); print("after A", r["after_A"], flush=True)
    state = {k: v.clone() for k, v in model.state_dict().items()}; ostate = __import__("copy").deepcopy(opt.state_dict()); gstate = gen.get_state()
    for arm, rate in (("fresh_schedule", fresh(stepsB)), ("constant_1e-4", lambda it: 1e-4)):
        model.load_state_dict(state); opt.load_state_dict(ostate); gen.set_state(gstate)
        learn(model, opt, Bs, stepsB, gen, rate); r["after_A_and_B__" + arm] = scores(model, x, segs); print(arm, r["after_A_and_B__" + arm], flush=True)
    model, opt, gen = new(seed); learn(model, opt, Bs, stepsB, gen, fresh(stepsB)); r["B_only"] = scores(model, x, segs); print("B only", r["B_only"], flush=True)
    r["cpu"] = round(time.process_time() - t0, 1); json.dump(r, open(os.path.join(SC.HERE, "res", "life_gpt.json"), "w"), indent=1)
