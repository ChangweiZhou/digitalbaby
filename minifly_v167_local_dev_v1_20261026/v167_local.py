"""V167 learners (design section 1). Network as V163-V165: KC code -> H (512 ReLU) -> T (2,048 ReLU, dropout 0.5) -> readout.
LOCAL: no gradient crosses a layer. Readout: delta rule (its own gradient on a detached input). T: DFA, delta_T =
mask * ReLU'(a_T) * (e F_T). H: DFA, delta_H = ReLU'(a_H) * (e F_H). e = (p - onehot(y)) / N.
LOCAL-SHUF: the error rows sent to H and T are permuted within the batch (the readout keeps its true error).
DFA-H: V165's step (back-propagation within the top), in the same loop."""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import v167_common as C
import v165_dfa as D

V = C.V
S = C.SET["local"]


def feedback(K, width, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(K, width, generator=g) / math.sqrt(K)


def local_step(net, FH, FT, idx, c, perm=None):
    """Sets .grad on every parameter for one batch; returns (e, mask, deltas) for checks."""
    top = net.top
    aH = F.embedding_bag(idx, net.W, mode="sum", padding_idx=V.PAD) + net.b
    h = torch.relu(aH).detach()
    aT = top.first(h)
    zT = torch.relu(aT)
    msk = top.drop(torch.ones_like(zT))                  # dropout mask with its 1/(1-p) scale (ones in eval mode)
    zd = (zT * msk).detach()
    logits = top.out(zd)
    F.cross_entropy(logits, c).backward()               # readout only: the delta rule
    e = D.errors(logits, c)
    eh = e[perm] if perm is not None else e
    dT = (eh @ FT) * msk * (aT.detach() > 0).to(e.dtype)
    aT.backward(dT)
    dH = (eh @ FH) * (aH.detach() > 0).to(e.dtype)
    aH.backward(dH)
    return e, msk, dT, dH


def make(cls):
    torch.manual_seed(V.SET["probe"]["seed"])
    W, b = V.init_H()
    net = V.BPH(cls.K, W, b)
    net.cls = cls
    return net


def align_T(net, FT, split, n_pos):
    """Cosine between T's DFA update and the back-propagated gradient of T's first layer (eval mode)."""
    net.eval()
    idx, c = split["idx"][:n_pos], torch.from_numpy(net.cls.encode(split["y"][:n_pos]))
    with torch.no_grad():
        h = torch.relu(F.embedding_bag(idx, net.W, mode="sum", padding_idx=V.PAD) + net.b)
    Wt = torch.nn.Parameter(net.top.first.weight.detach().clone())
    aT = F.linear(h, Wt, net.top.first.bias.detach())
    logits = net.top.out(torch.relu(aT))
    F.cross_entropy(logits, c).backward()
    g_bp = Wt.grad.clone()
    e = D.errors(logits.detach(), c)
    g_dfa = ((e @ FT) * (aT.detach() > 0).to(e.dtype)).T @ h
    net.train()
    return float((g_bp.flatten() @ g_dfa.flatten()) / (g_bp.norm() * g_dfa.norm()))


def train(A_train, A_calib, mode, max_epochs, patience, align_split=None, zero_FH=False):
    """mode 'local' | 'local_shuf' | 'dfa_h'. Same loop, initialisation, batches and optimiser as V165's train_dfa."""
    t0 = time.process_time()
    p = dict(V.SET["probe"])
    p.update({"max_epochs": max_epochs, "patience": patience})
    cls = V.R.Classes(A_train["y"])
    net = make(cls)
    FH = D.feedback(cls.K, net.W.shape[1])                # V165's F (seed 21)
    if zero_FH:
        FH = torch.zeros_like(FH)
    FT = feedback(cls.K, net.top.first.out_features, S["F_T_seed"])
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["wd"])
    g = torch.Generator().manual_seed(p["seed"])
    gs = torch.Generator().manual_seed(S["shuffle_seed"])
    ga = torch.Generator().manual_seed(S["shuffle_seed"] + 1)
    ctr = torch.from_numpy(cls.encode(A_train["y"]))
    calib = lambda m: float(cls.bits(V.probs(m, A_calib["idx"]), A_calib["y"]).mean())  # noqa: E731
    curve, alH, alT, best, since = [], [], [], (np.inf, 0, None), 0
    for ep in range(1, max_epochs + 1):
        net.train()
        perm = torch.randperm(len(ctr), generator=g)
        for i in range(0, len(perm), p["batch"]):
            j = perm[i:i + p["batch"]]
            sh = torch.randperm(len(j), generator=gs) if mode == "local_shuf" else None
            opt.zero_grad()
            if mode == "dfa_h":
                D.dfa_step(net, FH, A_train["idx"][j], ctr[j], None)
            else:
                local_step(net, FH, FT, A_train["idx"][j], ctr[j], sh)
            opt.step()
        cb = calib(net)
        curve.append(round(cb, 5))
        if align_split is not None:
            alH.append(D.alignment(net, FH, align_split, C.SET["align_positions"], ga if mode == "local_shuf" else None))
            if mode != "dfa_h":
                alT.append(align_T(net, FT, align_split, C.SET["align_positions"]))
        if cb < best[0]:
            best, since = (cb, ep, {k: v.clone() for k, v in net.state_dict().items()}), 0
        else:
            since += 1
            if since >= patience:
                break
    net.load_state_dict(best[2])
    with torch.no_grad():
        net.W[V.PAD] = 0.0
    info = {"mode": mode, "best_epoch": best[1], "epochs_run": len(curve), "calib_bpb": best[0], "calib_curve": curve,
            "alignment_H": alH, "alignment_T": alT, "cut_off_by_budget": best[1] == max_epochs,
            "records": len(curve) * len(A_train["n"]), "cpu_s": round(time.process_time() - t0, 1)}
    return (net.W.detach().clone(), net.b.detach().clone()), net, info


def local_top(feats, ys, n_held, d_in, n_train_lines):
    """Non-gating fully local carry-over: a fresh T + readout trained locally on B over frozen features (seed 0)."""
    t0 = time.process_time()
    s = C.SET["local_top_B"]
    cls = V.R.Classes(ys["train"])
    torch.manual_seed(s["seed"])
    top = V.Probe("dense", d_in, V.SET["probe"]["hidden"], cls.K, V.SET["probe"]["dropout"])
    FT = feedback(cls.K, top.first.out_features, S["F_T_seed"])
    opt = torch.optim.AdamW(top.parameters(), lr=V.SET["probe"]["lr"], weight_decay=V.SET["probe"]["wd"])
    g = torch.Generator().manual_seed(s["seed"])
    ctr = torch.from_numpy(cls.encode(ys["train"]))
    curve, best, since = [], (np.inf, 0, None), 0
    for ep in range(1, s["max_epochs"] + 1):
        top.train()
        perm = torch.randperm(len(ctr), generator=g)
        for i in range(0, len(perm), V.SET["probe"]["batch"]):
            j = perm[i:i + V.SET["probe"]["batch"]]
            opt.zero_grad()
            x = feats["train"][j]
            aT = top.first(x)
            zT = torch.relu(aT)
            msk = top.drop(torch.ones_like(zT))
            logits = top.out((zT * msk).detach())
            F.cross_entropy(logits, ctr[j]).backward()
            e = D.errors(logits, ctr[j])
            aT.backward((e @ FT) * msk * (aT.detach() > 0).to(e.dtype))
            opt.step()
        cb = float(cls.bits(V.probs(top, feats["calib"]), ys["calib"]).mean())
        curve.append(round(cb, 5))
        if cb < best[0]:
            best, since = (cb, ep, {k: v.clone() for k, v in top.state_dict().items()}), 0
        else:
            since += 1
            if since >= s["patience"]:
                break
    top.load_state_dict(best[2])
    bits = cls.bits(V.probs(top, feats["held"]), ys["held"])
    return {"held_bpb": float(bits.mean()), "lines": V.line_sums(bits, n_held).tolist(), "best_epoch": best[1],
            "calib_curve": curve, "records": len(curve) * n_train_lines, "cpu_s": round(time.process_time() - t0, 1)}
