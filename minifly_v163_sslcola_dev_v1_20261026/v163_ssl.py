"""V163 lane L learner (design section 2): the CLAPP-type layer-local contrastive predictive rule on top of the KC code.
h_t = ReLU(W x_t + b); positive pair (h_t, h_{t+1}) within a line; one negative h_n from a uniformly random taught position;
L = max(0, 1 - h_{t+1}' B h_t) + max(0, 1 + h_n' B h_t); gradient of L in W, b, B only (nothing else exists below the probe,
and the probe never sends a gradient here). Adam, learning rate 1e-3, batch 256 pairs, no weight decay."""
import time

import numpy as np
import torch
import torch.nn.functional as F

import v163_lib as V

S = V.SET["lane_L"]["ssl"]


def pairs(split):
    """(t, t+1) position pairs within lines."""
    e = np.cumsum(np.concatenate([[0], split["n"]]))
    return np.concatenate([np.arange(e[i], e[i + 1] - 1) for i in range(len(split["n"]))])


class CLAPP(torch.nn.Module):
    def __init__(self, W, b, dtype=torch.float32):
        super().__init__()
        self.W = torch.nn.Parameter(W.clone().to(dtype))
        self.b = torch.nn.Parameter(b.clone().to(dtype))
        self.B = torch.nn.Parameter(torch.zeros(W.shape[1], W.shape[1], dtype=dtype))

    def h(self, idx):
        return torch.relu(F.embedding_bag(idx, self.W, mode="sum", padding_idx=V.PAD) + self.b)

    def scores(self, idx, t, neg):
        hc, hp, hn = self.h(idx[t]), self.h(idx[t + 1]), self.h(idx[neg])
        pred = hc @ self.B.T                      # B h_t for each pair
        return (hp * pred).sum(1), (hn * pred).sum(1)

    def loss(self, idx, t, neg):
        sp, sn = self.scores(idx, t, neg)
        return (torch.relu(1 - sp) + torch.relu(1 + sn)).mean()


def explicit_grads(m, idx, t, neg):
    """The same update written as the local three-factor rule (no autograd): for each of the three activities, the
    presynaptic inputs times the unit's derivative times a lateral prediction, gated by the violated margins."""
    with torch.no_grad():
        ac, ap, an = (F.embedding_bag(idx[s], m.W, mode="sum", padding_idx=V.PAD) + m.b for s in (t, t + 1, neg))
        hc, hp, hn = torch.relu(ac), torch.relu(ap), torch.relu(an)
        Bm = m.B
        sp, sn = (hp * (hc @ Bm.T)).sum(1), (hn * (hc @ Bm.T)).sum(1)
        N = len(t)
        gp, gn = ((1 - sp) > 0).to(hc.dtype) / N, ((1 + sn) > 0).to(hc.dtype) / N   # global gates
        dhp = -gp[:, None] * (hc @ Bm.T)
        dhn = gn[:, None] * (hc @ Bm.T)
        dhc = -gp[:, None] * (hp @ Bm) + gn[:, None] * (hn @ Bm)
        dW = torch.zeros_like(m.W)
        db = torch.zeros_like(m.b)
        for s, a, dh in ((t, ac, dhc), (t + 1, ap, dhp), (neg, an, dhn)):
            da = dh * (a > 0).to(a.dtype)
            db += da.sum(0)
            for j in range(len(s)):
                ix = idx[s[j]]
                ix = ix[ix != V.PAD]
                dW[ix] += da[j]
        dB = -(gp[:, None] * hp).T @ hc + (gn[:, None] * hn).T @ hc
    return dW, db, dB


def train(A, passes, checkpoints=(), lines=None, seed_offset=0):
    """Train from RAND-H's initialisation on A's taught text (or its first `lines` lines). Returns the model, the
    (W, b) at each checkpoint pass, records (passes x lines used) and the per-pass loss."""
    t0 = time.process_time()
    split = A["train"]
    if lines is not None:
        e = int(np.sum(split["n"][:lines]))
        split = {"idx": split["idx"][:e], "y": split["y"][:e], "n": split["n"][:lines]}
    W, b = V.init_H()
    m = CLAPP(W, b)
    opt = torch.optim.Adam(m.parameters(), lr=S["lr"])
    P = torch.from_numpy(pairs(split))
    g_ord = torch.Generator().manual_seed(S["order_seed"] + seed_offset)
    g_neg = torch.Generator().manual_seed(S["neg_seed"] + seed_offset)
    npos = len(split["y"])
    snaps, losses, cost = {}, [], []
    for p in range(1, passes + 1):
        tp = time.process_time()
        perm = P[torch.randperm(len(P), generator=g_ord)]
        tot = 0.0
        for i in range(0, len(perm), S["batch"]):
            t = perm[i:i + S["batch"]]
            neg = torch.randint(0, npos, (len(t),), generator=g_neg)
            loss = m.loss(split["idx"], t, neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(t)
        losses.append(tot / len(perm))
        cost.append(round(time.process_time() - tp, 1))
        if p in checkpoints:
            snaps[p] = (m.W.detach().clone(), m.b.detach().clone())
    return m, snaps, {"records": passes * len(split["n"]), "loss_per_pass": losses, "cpu_s_per_pass": cost,
                      "cpu_s": round(time.process_time() - t0, 1)}


def separation(m, split, seed=0):
    """Share of within-line pairs whose positive score exceeds the score of a random negative."""
    t = torch.from_numpy(pairs(split))
    neg = torch.randint(0, len(split["y"]), (len(t),), generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        sp, sn = m.scores(split["idx"], t, neg)
    return float((sp > sn).double().mean())
