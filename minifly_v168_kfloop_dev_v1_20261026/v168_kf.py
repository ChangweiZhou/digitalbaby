"""V168 learner (design section 2): KF-RTRL with exact immediate terms (Mujika, Meier and Steger 2018) for the
recurrent group [W_x | W_h | W_r | b_h] and the query group [W_q | b_q] of V166's LOOP cell, and its KF-SHUF control
(carried terms use the learning signal of another line in the batch). For the C1 checks only: the explicit TBPTT-1
gradient (carried terms dropped), exact RTRL (full influence matrices) and the diagonal-trace e-prop rule.

State h_t; D_t = diag(phi'_t)(W_h + W_r M_{t-1} diag(rho'_{t-1}) W_q); B^a_t = diag(phi'_t) with u^a_t =
[x_t; h_{t-1}; r_{t-1}; 1]; B^q_t = diag(phi'_t) W_r M_{t-1} diag(rho'_{t-1}) with [h_{t-1}; 1];
L_t = W_o^T e_t + W_q^T dq_t, dq_t = rho'_t * M_t^T W_or^T e_t.
Gradient = (h, r)-TBPTT-1 gradient (exact) + sum over scored t of the carried terms
  (L_t^T D_t A^a_{t-1}) (x) u^a_{t-1} + (L_t^T D_t A^q_{t-1}) (x) u^q_{t-1} + (L_t^T B^q_t) (x) [h_{t-1}; 1].
"""
import numpy as np
import torch

import v168_lib as L

L166 = L.L166
NI, NKC, H, T, D_IN = L.NI, L.NKC, L.H, L.T, L166.D_IN
NA = D_IN + H + NI + 1                       # presynaptic size of the recurrent group
NQ = H + 1                                   # presynaptic size of the query group
NAMES = ("Wx", "Wh", "bh", "Wr", "Wq", "bq", "Wo", "Wor", "bo")


def split(gA, gQ, gO):
    """Group gradients -> V166 Net parameter names."""
    return {"Wx": gA[:, :D_IN], "Wh": gA[:, D_IN:D_IN + H], "Wr": gA[:, D_IN + H:D_IN + H + NI], "bh": gA[:, -1],
            "Wq": gQ[:, :H], "bq": gQ[:, H], "Wo": gO[0], "Wor": gO[1], "bo": gO[2]}


def mix(A1, u1, A2, u2, s):
    """Random-sign Kronecker mixing, per line: E[A (x) u] = A1 (x) u1 + A2 (x) u2. Scales rho_i = sqrt(|A_i|/|u_i|)
    minimise the variance of the cross terms. If one term is zero, the other is kept exactly."""
    nA1, nu1 = A1.flatten(1).norm(dim=1), u1.norm(dim=1)
    nA2, nu2 = A2.flatten(1).norm(dim=1), u2.norm(dim=1)
    ok1, ok2 = (nA1 > 0) & (nu1 > 0), (nA2 > 0) & (nu2 > 0)
    both = ok1 & ok2
    one = torch.ones_like(nA1)
    r1 = torch.where(both, (nA1 / torch.where(ok1, nu1, one)).sqrt(), one)
    r2 = torch.where(both, (nA2 / torch.where(ok2, nu2, one)).sqrt(), one)
    Ab = A1 / r1[:, None, None] + s[:, None, None] * A2 / r2[:, None, None]
    ub = r1[:, None] * u1 + s[:, None] * r2[:, None] * u2
    A = torch.where(both[:, None, None], Ab, torch.where(ok1[:, None, None], A1, A2))
    u = torch.where(both[:, None], ub, torch.where(ok1[:, None], u1, u2))
    return A, u


def run(net, x, K, V, ws, tg, mode="KF", perm=None, signs=None, gen=None, t_stop=None):
    """One pass over a batch of lines with the parameters fixed.
    mode: 'TB1' (carried terms dropped), 'KF', 'EXACT' (full influence matrices), 'EPROP' (diagonal traces for the
    recurrent group; e-prop's own memoryless-unit formula for the query group).
    perm: line index whose learning signal the carried terms use (KF-SHUF); None = own line.
    signs: (T, 2, B) tensor of +-1 (KF), else drawn per step from gen (2 x B draws per step, every step).
    t_stop: return the traces after this step (C1 check 3) instead of gradients.
    Returns (grads, tb1_grads): dicts by parameter name; grads includes the TBPTT-1 part."""
    P = {k: v.detach() for k, v in net.named_parameters()}
    Wx, Wh, bh, Wr, Wq, bq, Wo, Wor, bo = (P[k] for k in NAMES)
    B, dt = x.shape[0], x.dtype
    m = tg >= 0
    nsc = m.sum()
    z0 = lambda *s: torch.zeros(*s, dtype=dt)  # noqa: E731
    gA, gQ, gO = z0(H, NA), z0(NKC, NQ), [z0(NI, H), z0(NI, NI), z0(NI)]
    cA, cQ = z0(H, NA), z0(NKC, NQ)                 # carried parts
    h, r = z0(B, H), z0(B, NI)
    one = torch.ones(B, 1, dtype=dt)
    Aa, ua, Aq, uq = z0(B, H, H), z0(B, NA), z0(B, H, NKC), z0(B, NQ)
    Ja = z0(B, H, H, NA) if mode == "EXACT" else None
    Jq = z0(B, H, NKC, NQ) if mode == "EXACT" else None
    ea = z0(B, H, NA) if mode == "EPROP" else None
    rho_p = mask_p = hh_p = Dt = Bq = None
    for t in range(x.shape[1]):
        u_t = torch.cat([x[:, t], h, r, one], 1)                         # [x_t; h_{t-1}; r_{t-1}; 1]
        a = x[:, t] @ Wx.T + bh + h @ Wh.T + r @ Wr.T
        hn = torch.tanh(a)
        phi = 1 - hn * hn
        zq = hn @ Wq.T + bq
        q = torch.relu(zq)
        rho = (zq > 0).to(dt)
        mask = (ws <= t).to(dt)
        rn = torch.einsum("bp,bpn->bn", torch.einsum("bpk,bk->bp", K, q) * mask, V)
        y = hn @ Wo.T + rn @ Wor.T + bo
        hh = torch.cat([hn, one], 1)
        if t > 0:
            C = torch.einsum("bpn,bpk->bnk", V * mask_p[:, :, None], K * rho_p[:, None, :])  # M_{t-1} diag(rho_{t-1})
            Bq = phi[:, :, None] * (Wr @ C)                                                    # (B, H, NKC)
            Dt = phi[:, :, None] * (Wh + Wr @ (C @ Wq))                                        # (B, H, H)
        sc = m[:, t]
        if bool(sc.any()):
            e = (torch.softmax(y, -1) - torch.nn.functional.one_hot(tg[:, t].clamp(min=0), NI).to(dt)) \
                * sc[:, None].to(dt) / nsc
            gO[0] += e.T @ hn
            gO[1] += e.T @ rn
            gO[2] += e.sum(0)
            MTdr = torch.einsum("bpk,bp->bk", K, mask * torch.einsum("bpn,bn->bp", V, e @ Wor))   # M_t^T W_or^T e
            if mode == "EPROP":
                # e-prop, query units memoryless: eligibility rho'_t [h_t; 1]; learning signal dl_t/dq_t (immediate)
                elig = rho[:, :, None] * hh[:, None, :]                                        # (B, NKC, NQ)
                gQ += torch.einsum("bk,bki->ki", MTdr, elig)
            else:
                gQ += (rho * MTdr).T @ hh                                                      # immediate output path
            Lsig = e @ Wo + (rho * MTdr) @ Wq                                                  # dl_t/dh_t
            gA += (Lsig * phi).T @ u_t                                                         # immediate (TBPTT-1)
            if t > 0 and mode != "TB1":
                Lc = Lsig if perm is None else Lsig[perm]
                LD = torch.einsum("bh,bhj->bj", Lc, Dt)
                if mode == "KF":
                    cQ += torch.einsum("bh,bhk->bk", Lc, Bq).T @ hh_p
                    cA += torch.einsum("bj,bjo->bo", LD, Aa).T @ ua
                    cQ += torch.einsum("bj,bjk->bk", LD, Aq).T @ uq
                elif mode == "EXACT":
                    cQ += torch.einsum("bh,bhk->bk", Lc, Bq).T @ hh_p
                    cA += torch.einsum("bj,bjoi->oi", LD, Ja)
                    cQ += torch.einsum("bj,bjki->ki", LD, Jq)
                elif mode == "EPROP":
                    cA += torch.einsum("bj,bji->ji", Lc * torch.diagonal(Dt, dim1=1, dim2=2), ea)
        # traces (the gradient above used the traces of step t-1)
        if mode == "KF":
            s = signs[t].to(dt) if signs is not None else \
                (torch.randint(0, 2, (2, B), generator=gen) * 2 - 1).to(dt)
            if t == 0:
                Aa, ua = torch.diag_embed(phi), u_t
            else:
                Aa, ua = mix(Dt @ Aa, ua, torch.diag_embed(phi), u_t, s[0])
                Aq, uq = mix(Dt @ Aq, uq, Bq, hh_p, s[1])
        elif mode == "EXACT":
            Ja = (torch.einsum("bjk,bkoi->bjoi", Dt, Ja) if t > 0 else Ja) \
                + torch.diag_embed(phi)[:, :, :, None] * u_t[:, None, None, :]
            if t > 0:
                Jq = torch.einsum("bjk,bkmi->bjmi", Dt, Jq) + Bq[:, :, :, None] * hh_p[:, None, None, :]
        elif mode == "EPROP":
            ea = (torch.diagonal(Dt, dim1=1, dim2=2)[:, :, None] * ea if t > 0 else ea) + phi[:, :, None] * u_t[:, None, :]
        if t_stop is not None and t == t_stop:
            return {"Aa": Aa, "ua": ua, "Aq": Aq, "uq": uq, "Ja": Ja, "Jq": Jq}
        h, r, rho_p, mask_p, hh_p = hn, rn, rho, mask, hh
    tb1 = split(gA, gQ, gO)
    full = split(gA + cA, gQ + cQ, gO)
    return full, tb1


def derangement(rng, n):
    assert n >= 2
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def learner_grad(arm):
    """Gradient function for v168_lib.train: KF or KF-SHUF. Both draw the identical sign stream (seed 31); KF-SHUF also
    draws one derangement of the batch per batch (seed 37)."""
    gen = torch.Generator().manual_seed(L.SET["kf"]["sign_seed"])
    rng = np.random.default_rng(L.SET["kf"]["shuf_seed"])

    def g(net, x, K, V, ws, tg):
        perm = torch.from_numpy(derangement(rng, x.shape[0])) if arm == "KF-SHUF" else None
        full, _ = run(net, x, K, V, ws, tg, "KF", perm=perm, gen=gen)
        for k, p in net.named_parameters():
            p.grad = full[k].to(p.dtype).clone()
    return g
