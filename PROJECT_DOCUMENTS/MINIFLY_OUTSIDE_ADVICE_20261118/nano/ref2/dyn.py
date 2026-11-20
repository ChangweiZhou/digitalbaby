#!/usr/bin/env python3
"""nano/ref2, PLAN_3.md: the network going on through the last tenth (dynamic evaluation). Each byte is answered before it is learned from.
    python3 dyn.py <seed label, e.g. s1337> W=64|256 [lr=1e-4] [threads=1] [limit=<windows, a timing test only>] [ckpt=ckpt/<label>.pt]
The last tenth is read in order in windows of W bytes. For each window: (1) with the strengths as they then are, in evaluation mode, every byte
of the window is answered exactly as in plan 1's score (windows of 256 bytes moved on by 8, the last 8 places used: 249 to 256 bytes before each
byte); (2) one step of teaching on that window: one block (a batch of one) whose last W places name the W bytes of the window (the block is the
256 bytes ending one before the window's last byte; for W = 256 it is an ordinary block of the schedule), the loss the mean over those W places,
dropout 0.2 as in the schedule, the same AdamW with its running averages carried over from step 4,000, the rate of the schedule's end, gradient
clipped at 1. Writes res/dyn_<label>_W<W>[_lr<lr>].f32 and .json."""
import json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
import gpt, score as SC
label = sys.argv[1]; kw = dict(a.split("=") for a in sys.argv[2:]); W = int(kw["W"]); lr = float(kw.get("lr", 1e-4)); limit = int(kw.get("limit", 0)); STR = 8
torch.set_num_threads(int(kw.get("threads", 1))); st = torch.load(kw.get("ckpt", f"ckpt/{label}.pt"), weights_only=False); a = st["args"]
assert limit or st["step"] == int(a["steps"]), "the run has not ended"
B = int(a["block"]); model = gpt.GPT(int(a["layers"]), int(a["heads"]), int(a["width"]), B, float(a["dropout"])); model.load_state_dict(st["model"])
dec = [p for p in model.parameters() if p.dim() >= 2]; nodec = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([dict(params=dec, weight_decay=0.1), dict(params=nodec, weight_decay=0.0)], lr=1e-3, betas=(0.9, 0.99)); opt.load_state_dict(st["opt"]); torch.set_rng_state(st["rng"])
for g in opt.param_groups: g["lr"] = lr
x, cut, _ = SC.load(); N = len(x); n = N - cut; xt = torch.from_numpy(x); nll = np.full(n, np.nan); right = np.zeros(n, bool); t0 = time.process_time(); cpu_step = 0.0; steps = 0
for w, lo in enumerate(range(cut, N, W)):
    if limit and w >= limit: break
    hi = min(lo + W, N); model.eval()
    with torch.no_grad():
        ends = np.arange(lo + STR - 1, hi + STR - 1, STR); ends = np.minimum(ends, hi - 1)          # a window with end e reads x[e-B:e]; its last place names byte e
        lp = F.log_softmax(model(xt[torch.from_numpy(ends[:, None] + np.arange(-B, 0)[None, :])])[:, B - STR:].double(), -1).numpy()
        for r, e in enumerate(ends):
            t = np.arange(e - STR + 1, e + 1); ok = (t >= lo) & np.isnan(nll[np.clip(t - cut, 0, n - 1)]); tt = t[ok]
            nll[tt - cut] = -lp[r][ok, x[tt]]; right[tt - cut] = lp[r][ok].argmax(1) == x[tt]
    assert not np.isnan(nll[lo - cut:hi - cut]).any()
    c = time.process_time(); model.train(); m = hi - lo
    xb = xt[hi - 1 - B:hi - 1][None]; yb = xt[hi - B:hi][None]; loss = F.cross_entropy(model(xb)[0, B - m:], yb[0, B - m:])
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); cpu_step += time.process_time() - c; steps += 1
done = ~np.isnan(nll); cpu = time.process_time() - t0
out = dict(label=label, W=W, lr=lr, steps=steps, bytes=int(done.sum()), nats=float(nll[done].mean()), right=float(100 * right[done].mean()), cpu=round(cpu, 1), cpu_a_teaching_step=round(cpu_step / steps, 3),
           every_eighth=float(nll[::8][done[::8]].mean()), bytes_before_each=[B - STR + 1, B], dropout_in_the_step=float(a["dropout"]), optimiser_state="carried over from step 4,000", batch=1)
print(json.dumps(out), flush=True)
if not limit:
    assert done.all(); tag = f"dyn_{label}_W{W}" + ("" if lr == 1e-4 else f"_lr{kw['lr']}"); nll.astype(np.float32).tofile(f"res/{tag}.f32"); json.dump(out, open(f"res/{tag}.json", "w"), indent=1)
