"""V164 COLA (Wang et al. 2026, Algorithm 1; V163 design section 3) for V163's recurrent layer, in the shared schedule:
the per-step teaching signal and the per-unit statistics are computed online, step by step; the parameter gradients
(the same outer-product form as BPTT, Eqs. 32-33) are summed over the batch of lines and applied in one optimiser step.
  e_t      = dL_t/dx_t = phi'(x_t) * (W_out^T dL_t/do_t)                                   (Eq. 5, immediate error)
  delta_t  = e_t / (1 - mu * phi'(x_t)), |mu phi'| <= cap, denominator >= floor              (Eqs. 23, 31)
  alpha    = clip(S / (Q + eps)), S, Q EMAs of delta_t delta_{t-1} and delta_{t-1}^2          (Eq. 25)
  mu       = S_AB / (S_A2 + eps), EMAs of A^2 and A*B with                                    (Eqs. 28, 30)
             A = alpha phi'_t e_{t-1} - phi'_{t-1} e_t,  B = alpha e_{t-1} - e_t
Statistics are means over the lines that are inside their line at both steps, and persist across batches and epochs.
"""
import torch
import torch.nn.functional as F

import v164_lib as L

C = L.SET["cola"]
PAD = L.V.PAD


class COLA:
    def __init__(self, H, dtype=torch.float32, freeze_stats=False):
        z = lambda: torch.zeros(H, dtype=dtype)  # noqa: E731
        self.alpha, self.mu, self.S, self.Q, self.SA2, self.SAB = z(), z(), z(), z(), z(), z()
        self.freeze = freeze_stats
        self.capped, self.steps, self.trace = 0, 0, None

    def teach(self, e, ph):
        prod = torch.clamp(self.mu * ph, -C["mu_cap"], C["mu_cap"])
        self.capped += int((self.mu * ph).abs().ge(C["mu_cap"]).sum())
        self.steps += e.numel()
        return e / torch.clamp(1 - prod, min=C["denom_floor"])

    def update_alpha(self, d, d_prev, m):
        if self.freeze or not m.any():
            return
        ra = C["rho_alpha"]
        self.S = ra * self.S + (1 - ra) * (d[m] * d_prev[m]).mean(0)
        self.Q = ra * self.Q + (1 - ra) * (d_prev[m] ** 2).mean(0)
        self.alpha = torch.clamp(self.S / (self.Q + C["eps"]), -C["alpha_clip"], C["alpha_clip"])

    def update_mu(self, e, e_prev, ph, ph_prev, m):
        if self.freeze or not m.any():
            return
        rm = C["rho_mu"]
        A = self.alpha * ph * e_prev - ph_prev * e
        B = self.alpha * e_prev - e
        self.SA2 = rm * self.SA2 + (1 - rm) * (A[m] ** 2).mean(0)
        self.SAB = rm * self.SAB + (1 - rm) * (A[m] * B[m]).mean(0)
        self.mu = self.SAB / (self.SA2 + C["eps"])

    def __call__(self, net, idx, valid, ctr, record=False):
        """Sets .grad on every parameter for one batch of lines (loss: mean cross-entropy over scored positions)."""
        Bn, T, A = idx.shape
        inline = idx[:, :, 0] != PAD
        N = valid.sum()
        u = F.embedding_bag(idx.reshape(Bn * T, A), net.Wrk, mode="sum", padding_idx=PAD).view(Bn, T, -1)
        Wrr, b, Wo, bo = net.Wrr.detach(), net.b.detach(), net.out.weight.detach(), net.out.bias.detach()
        ud = u.detach()
        H = Wrr.shape[0]
        r = torch.zeros(Bn, H, dtype=ud.dtype)
        gWrr, gb = torch.zeros_like(Wrr), torch.zeros_like(b)
        gWo, gbo = torch.zeros_like(Wo), torch.zeros_like(bo)
        dU = torch.zeros_like(ud)
        e_prev = d_prev = ph_prev = None
        self.trace = [] if record else None
        for t in range(T):
            x = ud[:, t] + r @ Wrr.T + b
            rn = torch.tanh(x)
            ph = 1 - rn ** 2
            p = torch.softmax(rn @ Wo.T + bo, -1)
            y = F.one_hot(ctr[:, t].clamp(min=0), Wo.shape[0]).to(p.dtype)
            dLdo = (p - y) * valid[:, t:t + 1].to(p.dtype) / N
            e = ph * (dLdo @ Wo)
            d = self.teach(e, ph)
            if record:
                self.trace.append((e.clone(), ph.clone(), d.clone(), self.mu.clone()))
            if t >= 1:
                m = inline[:, t] & inline[:, t - 1]
                self.update_alpha(d, d_prev, m)
            gWrr += d.T @ r
            gb += d.sum(0)
            dU[:, t] = d
            gWo += dLdo.T @ rn
            gbo += dLdo.sum(0)
            if t >= 1:
                self.update_mu(e, e_prev, ph, ph_prev, m)
            r, e_prev, d_prev, ph_prev = rn, e, d, ph
        net.Wrk.grad = None
        u.backward(dU)
        net.Wrr.grad, net.b.grad, net.out.weight.grad, net.out.bias.grad = gWrr, gb, gWo, gbo

    def summary(self):
        q = lambda v: [float(torch.quantile(v.double(), x)) for x in (0.05, 0.5, 0.95)]  # noqa: E731
        return {"alpha_q05_50_95": q(self.alpha), "mu_q05_50_95": q(self.mu),
                "share_steps_mu_phi_at_cap": self.capped / max(1, self.steps)}
