"""V168 library (design sections 1, 3, 4), torch environment. Loads V166's v166_lib.py read-only BY FILE PATH (V166's
folder is never put on sys.path, so V168's own runtime.py cannot be shadowed) and adds V168's world seeds to that
module's in-memory world table. Provides the reference forward pass with TBPTT-k detach points, the reference
gradients (BPTT, TBPTT-1/2/3) and a training loop that takes any gradient function (so every arm shares the forward
pass, initialisation, data order, AdamW, clipping and early stopping, and differs only in its gradient estimate)."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SET = json.loads((HERE / "SETTINGS.json").read_text())

_spec = importlib.util.spec_from_file_location("v166_lib", ROOT / SET["v166_lib"])
L166 = importlib.util.module_from_spec(_spec)
sys.modules["v166_lib"] = L166
_spec.loader.exec_module(L166)
L166.SET["worlds"].update(SET["worlds"])          # in-memory only: V168's own seeds; no V166 file changes

NI, NKC, H, T = L166.NI, L166.NKC, L166.SET["net"]["H"], L166.T
ANS = [L166.T_Q0 + 3 * e + 2 for e in range(L166.LINE["episodes"])]


def detach_steps(rule):
    """Steps at whose entry the carried state (h, r) is detached. TBPTT-k: s-k+1 for every ANS step s (windows of
    k <= 3 steps never overlap, since episodes are 3 steps long). BPTT: none."""
    if rule == "BPTT":
        return set()
    k = int(rule.split("-")[1])
    assert 1 <= k <= 3
    return {s - k + 1 for s in ANS}


def forward(net, x, K, V, ws, detach_at=()):
    """V166's LOOP forward pass (same operations in the same order), with optional detach points."""
    B = x.shape[0]
    U = x @ net.Wx.T + net.bh
    h = x.new_zeros(B, net.Wh.shape[0])
    r = x.new_zeros(B, NI)
    outs = []
    for t in range(x.shape[1]):
        if t in detach_at:
            h, r = h.detach(), r.detach()
        pre = U[:, t] + h @ net.Wh.T
        pre = pre + r @ net.Wr.T
        h = torch.tanh(pre)
        q = torch.relu(h @ net.Wq.T + net.bq)
        r = L166.read_factored(K, V, ws, q, t)
        outs.append(h @ net.Wo.T + r @ net.Wor.T + net.bo)
    return torch.stack(outs, 1)


def ref_grad(rule):
    """Gradient function of a reference: autograd through the forward pass with the rule's detach points."""
    da = detach_steps(rule)

    def g(net, x, K, V, ws, tg):
        m = tg >= 0
        loss = F.cross_entropy(forward(net, x, K, V, ws, da)[m], tg[m])
        loss.backward()
        return float(loss)
    return g


def world(kind, part, n=None):
    return L166.make_world(kind, part, "MIXED", n)


def train(name, cfg, tr, ca, grad_fn, log=None):
    """Every arm: V166's LOOP net (init seed 0), batch order seed 13, AdamW (cfg lr, wd 1e-4), clip 5, best calibration
    epoch (all scored steps) kept, patience 10. grad_fn(net, x, K, V, ws, tg) fills p.grad. Records = lines x epochs."""
    t0 = time.process_time()
    s = SET["train"]
    net = L166.Net("LOOP")
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=s["wd"])
    gen = torch.Generator().manual_seed(s["order_seed"])
    n = tr["n"]
    curve, best, since, epochs = [], (np.inf, 0, None), 0, 0
    for ep in range(1, cfg["max_epochs"] + 1):
        epochs = ep
        order = torch.randperm(n, generator=gen).numpy()
        for i in range(0, n, cfg["batch"]):
            x, K, V, ws, tg = L166.batch(tr, np.sort(order[i:i + cfg["batch"]]), "LOOP")
            opt.zero_grad()
            grad_fn(net, x, K, V, ws, tg)
            torch.nn.utils.clip_grad_norm_(net.parameters(), s["clip"])
            opt.step()
        if not all(torch.isfinite(p).all() for p in net.parameters()):
            raise FloatingPointError(f"{name}: non-finite parameters at epoch {ep}")
        cb = L166.evaluate(net, ca, "LOOP")["all_bits"]
        curve.append(round(cb, 5))
        if log:
            print(f"{name} ep {ep} calib {cb:.4f} cpu {time.process_time() - t0:.0f}", file=log, flush=True)
        if cb < best[0]:
            best, since = (cb, ep, {k: v.clone() for k, v in net.state_dict().items()}), 0
        else:
            since += 1
            if since >= s["patience"]:
                break
    net.load_state_dict(best[2])
    return net, {"name": name, "config": cfg, "best_epoch": best[1], "epochs_run": epochs, "calib_bits": best[0],
                 "calib_curve": curve, "best_epoch_is_last": best[1] == cfg["max_epochs"],
                 "line_presentations": n * epochs, "cpu_s": round(time.process_time() - t0, 1)}


def test_record(net, te):
    ev = L166.evaluate(net, te, "LOOP")
    return {"test": L166.summary(ev), "line": {k: ev[k].tolist() for k in ev if k != "all_bits"}}


def gap(worse, better, hop=2):
    """worse's bits minus better's bits on Q<hop> (positive: `better` is better), with the paired line-bootstrap SD."""
    lw, lb = worse["line"], better["line"]
    d = worse["test"][f"q{hop}_bits"] - better["test"][f"q{hop}_bits"]
    sd = L166.boot_sd(np.array(lw[f"q{hop}_sum"]), np.array(lb[f"q{hop}_sum"]), np.array(lb[f"q{hop}_n"]))
    return {"gain_bits": d, "boot_sd": sd, "z": d / sd if sd > 0 else None}
