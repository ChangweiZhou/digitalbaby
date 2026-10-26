"""V165 learner (design section 1): direct feedback alignment for V163's layer H under a back-propagated top.
a = W x + b, h = ReLU(a); the top (V163's probe architecture) learns by back-propagation within itself on h (detached);
H's teaching signal is delta = ReLU'(a) * (e F), e = (p - onehot(y)) / N, F fixed random (K x 512, variance 1/K).
DFA-SHUF permutes the rows of e within each batch. Same optimiser, batches and initialisation as V163's BP-H."""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import v165_common as C

V = C.V


def feedback(K, H):
    g = torch.Generator().manual_seed(C.SET["dfa"]["F_seed"])
    return torch.randn(K, H, generator=g) / math.sqrt(K)


def errors(logits, c):
    with torch.no_grad():
        return (torch.softmax(logits, 1) - F.one_hot(c, logits.shape[1]).to(logits.dtype)) / len(c)


def dfa_step(net, Fb, idx, c, perm=None):
    """Sets .grad on every parameter: back-propagation inside the top, DFA for W and b."""
    a = F.embedding_bag(idx, net.W, mode="sum", padding_idx=V.PAD) + net.b
    h = torch.relu(a)
    logits = net.top(h.detach())
    F.cross_entropy(logits, c).backward()
    e = errors(logits, c)
    if perm is not None:
        e = e[perm]
    delta = (e @ Fb) * (a.detach() > 0).to(e.dtype)
    a.backward(delta)
    return e, delta


def alignment(net, Fb, split, n_pos, shuffle_gen=None):
    """Cosine between the DFA update of W and the back-propagated next-byte gradient at the same parameters
    (eval mode, first n_pos calibration positions, rows of W that receive input)."""
    net.eval()
    idx = split["idx"][:n_pos]
    cls_c = torch.from_numpy(net.cls.encode(split["y"][:n_pos]))
    W = torch.nn.Parameter(net.W.detach().clone())
    a = F.embedding_bag(idx, W, mode="sum", padding_idx=V.PAD) + net.b.detach()
    logits = net.top(torch.relu(a))
    F.cross_entropy(logits, cls_c).backward()
    g_bp = W.grad.clone()
    W2 = torch.nn.Parameter(net.W.detach().clone())
    a2 = F.embedding_bag(idx, W2, mode="sum", padding_idx=V.PAD) + net.b.detach()
    e = errors(logits.detach(), cls_c)
    if shuffle_gen is not None:
        e = e[torch.randperm(len(e), generator=shuffle_gen)]
    a2.backward((e @ Fb) * (a2.detach() > 0).to(e.dtype))
    rows = g_bp.abs().sum(1) > 0
    x, y = g_bp[rows].flatten(), W2.grad[rows].flatten()
    net.train()
    return float(x @ y / (x.norm() * y.norm()))


def train_dfa(A_train, A_calib, shuffle, max_epochs, patience, align_split=None):
    """Returns (W, b), the trained joint model and info (best epoch on A calibration, per-epoch alignment)."""
    t0 = time.process_time()
    p = C.probe_settings(max_epochs, patience)
    cls = V.R.Classes(A_train["y"])
    torch.manual_seed(p["seed"])
    W, b = V.init_H()
    net = V.BPH(cls.K, W, b)
    net.cls = cls
    Fb = feedback(cls.K, W.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["wd"])
    g = torch.Generator().manual_seed(p["seed"])
    gs = torch.Generator().manual_seed(C.SET["dfa"]["shuffle_seed"])
    ga = torch.Generator().manual_seed(C.SET["dfa"]["shuffle_seed"] + 1)
    ctr = torch.from_numpy(cls.encode(A_train["y"]))
    calib = C.calib_fn(cls, A_calib)
    curve, align, best, since = [], [], (np.inf, 0, None), 0
    for ep in range(1, max_epochs + 1):
        net.train()
        perm = torch.randperm(len(ctr), generator=g)
        for i in range(0, len(perm), p["batch"]):
            j = perm[i:i + p["batch"]]
            opt.zero_grad()
            dfa_step(net, Fb, A_train["idx"][j], ctr[j],
                     torch.randperm(len(j), generator=gs) if shuffle else None)
            opt.step()
        cb = calib(net)
        curve.append(round(cb, 5))
        if align_split is not None:
            align.append(alignment(net, Fb, align_split, C.SET["align_positions"], ga if shuffle else None))
        if cb < best[0]:
            best, since = (cb, ep, {k: v.clone() for k, v in net.state_dict().items()}), 0
        else:
            since += 1
            if since >= patience:
                break
    net.load_state_dict(best[2])
    with torch.no_grad():
        net.W[V.PAD] = 0.0
    info = {"shuffle": shuffle, "best_epoch": best[1], "epochs_run": len(curve), "calib_bpb": best[0],
            "calib_curve": curve, "alignment_per_epoch": align, "cut_off_by_budget": best[1] == max_epochs,
            "records": len(curve) * len(A_train["n"]), "cpu_s": round(time.process_time() - t0, 1)}
    return (net.W.detach().clone(), net.b.detach().clone()), net, info
