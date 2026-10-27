"""V170 learners (design sections 2-3), written after the testbed cycle closed. Each returns a gradient function
grad_fn(net, w, idx, cache) for v170_lib.train, filling p.grad explicitly (no autograd):
* a_grad(arm): strategy A's one-step learner for A, A-SHUF and A-CUT (delta rule on the readout; the gate's one-step
  credit g(1-g) [(W_o^T e) . (r_t - r_{t-1})] z_t). The arm selects the fixed re-entry dynamics.
* b_grad(shuf, ...): strategy B's three-factor rule on the LOOP: eligibility tags (pre x post) decaying with lambda,
  the ANS-step output error broadcast through fixed random matrices B_h, B_q; readout by the delta rule. shuf=True is
  B-SHUF (feedback error from another line of the batch, a derangement per batch). The keyword `identity` replaces the
  random feedback by the true one-step learning signals (C1 check B1 only)."""
import numpy as np
import torch

import v170_lib as L


def _onehot_err(y, tg):
    """e = softmax(y) - onehot(target) at scored steps; mask (B, T)."""
    m = tg >= 0
    e = torch.softmax(y, -1)
    e = e - torch.nn.functional.one_hot(tg.clamp(min=0), L.NI).to(y.dtype)
    return e * m[..., None].to(y.dtype), m


def a_grad(arm, dtype=torch.float32):
    mode = L.a_mode(arm)

    def g(net, w, idx, cache):
        with torch.no_grad():
            x, r, tg = L.a_batch(w, idx, mode, dtype, cache)
            gt, z, rprev, mm, y = net.parts(x, r)
            e, m = _onehot_err(y, tg)
            n = m.sum().to(dtype)
            net.Wo.grad = torch.einsum("bti,btj->ij", e, mm) / n
            net.bo.grad = e.sum((0, 1)) / n
            s = gt * (1 - gt) * ((e @ net.Wo) * (r - rprev)).sum(-1)       # (B, T); zero off scored steps
            net.wg.grad = torch.einsum("bt,btd->d", s, z) / n
            net.bg.grad = s.sum() / n
            loss = -(torch.log_softmax(y, -1).gather(-1, tg.clamp(min=0)[..., None])[..., 0] * m).sum() / n
        return float(loss)
    return g


def feedback_mats(H, dtype=torch.float32):
    gen = torch.Generator().manual_seed(L.SET["b_rule"]["feedback_seed"])
    Bh = torch.randn(H, L.NI, generator=gen) / np.sqrt(L.NI)
    Bq = torch.randn(L.NKC, L.NI, generator=gen) / np.sqrt(L.NI)
    return Bh.to(dtype), Bq.to(dtype)


def mem_T(K, V, ws, t, v):
    """M_t^T v for the factored memory: sum_p [w_p <= t] K_p (V_p . v). K (B, P, NKC), V (B, P, NI), v (B, NI)."""
    a = torch.einsum("bpn,bn->bp", V, v) * (ws <= t).to(v.dtype)
    return torch.einsum("bp,bpk->bk", a, K)


def b_grad(shuf=False, lam=None, identity=False, perm_fn=None, dtype=torch.float32):
    lam = L.SET["b_rule"]["lambda"] if lam is None else lam
    gen = torch.Generator().manual_seed(L.SET["b_rule"]["shuf_seed"])
    state = {"perms": []}

    def derange(B):
        while True:
            p = torch.randperm(B, generator=gen)
            if B < 2 or not torch.any(p == torch.arange(B)):
                return p

    def g(net, w, idx, cache):
        with torch.no_grad():
            x, K, V, ws, tg = L.loop_batch(w, idx, dtype)
            B, H = x.shape[0], net.Wh.shape[0]
            Bh, Bq = feedback_mats(H, dtype)
            perm = (perm_fn(B) if perm_fn else derange(B)) if shuf else torch.arange(B)
            state["perms"].append(perm.tolist())
            din = x.shape[2]
            Ea = x.new_zeros(B, H, din + H + L.NI + 1)
            Eq = x.new_zeros(B, L.NKC, H + 1)
            ga, gq = torch.zeros_like(Ea[0]), torch.zeros_like(Eq[0])
            gWo, gWor, gbo = torch.zeros_like(net.Wo), torch.zeros_like(net.Wor), torch.zeros_like(net.bo)
            h = x.new_zeros(B, H)
            r = x.new_zeros(B, L.NI)
            m_all = tg >= 0
            n = m_all.sum().to(dtype)
            loss = 0.0
            one = x.new_ones(B, 1)
            for t in range(x.shape[1]):
                u = torch.cat([x[:, t], h, r, one], 1)
                pre = x[:, t] @ net.Wx.T + net.bh + h @ net.Wh.T + r @ net.Wr.T
                h = torch.tanh(pre)
                phi = 1 - h * h
                Ea = lam * Ea + phi[:, :, None] * u[:, None, :]
                qpre = h @ net.Wq.T + net.bq
                q = torch.relu(qpre)
                rho = (qpre > 0).to(dtype)
                hb = torch.cat([h, one], 1)
                Eq = lam * Eq + rho[:, :, None] * hb[:, None, :]
                r = L.L166.read_factored(K, V, ws, q, t)
                if not m_all[:, t].any():
                    continue
                y = h @ net.Wo.T + r @ net.Wor.T + net.bo
                mt = m_all[:, t].to(dtype)[:, None]
                e = (torch.softmax(y, -1) - torch.nn.functional.one_hot(tg[:, t].clamp(min=0), L.NI).to(dtype)) * mt
                loss -= float((torch.log_softmax(y, -1).gather(-1, tg[:, t].clamp(min=0)[:, None]) * mt).sum())
                gWo += e.T @ h
                gWor += e.T @ r
                gbo += e.sum(0)
                if identity:                                    # true one-step signals (check B1)
                    mq = mem_T(K, V, ws, t, e @ net.Wor)        # M_t^T W_or^T e
                    dh = e @ net.Wo + (rho * mq) @ net.Wq       # L_t = W_o^T e + W_q^T (rho' * M^T W_or^T e)
                    dq = mq
                else:
                    fe = e[perm]
                    dh, dq = fe @ Bh.T, fe @ Bq.T
                ga += (dh[:, :, None] * Ea).sum(0)
                gq += (dq[:, :, None] * Eq).sum(0)
            ga, gq = ga / n, gq / n
            net.Wx.grad = ga[:, :din].clone()
            net.Wh.grad = ga[:, din:din + H].clone()
            net.Wr.grad = ga[:, din + H:din + H + L.NI].clone()
            net.bh.grad = ga[:, -1].clone()
            net.Wq.grad = gq[:, :H].clone()
            net.bq.grad = gq[:, -1].clone()
            net.Wo.grad, net.Wor.grad, net.bo.grad = gWo / n, gWor / n, gbo / n
        return loss / float(n)
    g.state = state
    return g
