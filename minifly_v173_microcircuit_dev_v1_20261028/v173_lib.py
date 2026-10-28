"""V173 library (design sections 1-3): the MC architecture (32 isolated gated circuits of 8 neurons), forward-sensitivity
gradients (RTRL, DIAG, LEAK), autograd references (BPTT, T1), and V164's task, schedule and evaluation (loaded by file
path from the V164 delivery, unchanged).

Neuron i of circuit c, with v_t = [x_t; s_{t-1}; 1] (17 entries):
  q = tanh(Wq[c,i] . v),  g = sigmoid(Wg[c,i] . v)  (LEAK: g = 0.5),  s_i = (1 - g) s_{t-1,i} + g q.
Forward sensitivity E[j, i, type, d] = ds_j / dW_type[c, i, d], propagated as E_t = J_t E_{t-1} + P_t with
  J[j,k] = (1 - g_j) delta_jk + g_j (1 - q_j^2) R[j,k] + (q_j - s_{t-1,j}) g_j (1 - g_j) U[j,k],
  P[i, i, q, d] = g_i (1 - q_i^2) v_d,  P[i, i, g, d] = (q_i - s_{t-1,i}) g_i (1 - g_i) v_d.
The gradient of the mean cross-entropy is sum_t E_t^T delta_t, delta_t = W_out^T (p_t - y_t) / N (one-hop readout error).
DIAG keeps only E[i, i] and propagates it by J[i, i].
"""
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SET = json.loads((HERE / "SETTINGS.json").read_text())
torch.set_num_threads(SET["threads"])


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


L4 = load("v164_lib", ROOT / "minifly_v164_cola_dev_v1_20261026" / "v164_lib.py")
V = L4.V
M = SET["mc"]
S = SET["schedule"]


class MC(torch.nn.Module):
    def __init__(self, gated=True, dtype=torch.float32):
        super().__init__()
        nc, n, nin = M["circuits"], M["neurons"], M["inputs"]
        D = nin + n + 1
        gen = torch.Generator().manual_seed(M["init_seed"])

        def rows():
            w = torch.zeros(nc, n, D, dtype=torch.float64)
            w[:, :, :nin] = torch.randn(nc, n, nin, generator=gen, dtype=torch.float64) * M["in_sd"]
            w[:, :, nin:nin + n] = torch.randn(nc, n, n, generator=gen, dtype=torch.float64) / math.sqrt(M["rec_var_inv"])
            return w.to(dtype)
        self.gated = gated
        self.Wq = torch.nn.Parameter(rows())
        self.Wg = torch.nn.Parameter(rows()) if gated else None
        torch.manual_seed(M["init_seed"])
        self.out = torch.nn.Linear(nc * n, M["classes"]).to(dtype)
        self.nc, self.n, self.nin, self.D = nc, n, nin, D

    def inputs(self, idx):
        sym = idx[:, :, 0]
        ok = (sym >= 0) & (sym < self.nin)
        return F.one_hot(sym.clamp(0, self.nin - 1), self.nin).to(self.Wq.dtype) * ok[..., None]

    def step(self, x, s):
        B = x.shape[0]
        v = torch.cat([x[:, None, :].expand(B, self.nc, self.nin), s, torch.ones(B, self.nc, 1, dtype=s.dtype)], -1)
        q = torch.tanh(torch.einsum("cid,bcd->bci", self.Wq, v))
        g = torch.sigmoid(torch.einsum("cid,bcd->bci", self.Wg, v)) if self.gated else torch.full_like(q, M["leak_gate"])
        return (1 - g) * s + g * q, v, q, g

    def run(self, idx, truncate=False):
        X = self.inputs(idx)
        B, T, _ = X.shape
        s = torch.zeros(B, self.nc, self.n, dtype=self.Wq.dtype)
        out = []
        for t in range(T):
            s, _, _, _ = self.step(X[:, t], s.detach() if truncate else s)
            out.append(s.reshape(B, -1))
        return torch.stack(out, 1)

    def recurrent_params(self):
        return [p for p in (self.Wq, self.Wg) if p is not None]


@torch.no_grad()
def forward_grad(model, idx, valid, ctr, mode):
    """Sets .grad on every parameter: forward sensitivities ('rtrl' full within circuit, 'diag' own-row only)."""
    X = model.inputs(idx)
    B, T, _ = X.shape
    nc, n, nin, D = model.nc, model.n, model.nin, model.D
    types = 2 if model.gated else 1
    dt = model.Wq.dtype
    N = valid.sum()
    Wo, bo = model.out.weight, model.out.bias
    R = model.Wq[:, :, nin:nin + n]
    U = model.Wg[:, :, nin:nin + n] if model.gated else None
    s = torch.zeros(B, nc, n, dtype=dt)
    E = torch.zeros(B, nc, n, n, types, D, dtype=dt) if mode == "rtrl" else torch.zeros(B, nc, n, types, D, dtype=dt)
    G = torch.zeros(nc, n, types, D, dtype=dt)
    gWo, gbo = torch.zeros_like(Wo), torch.zeros_like(bo)
    ii = torch.arange(n)
    eye = torch.eye(n, dtype=dt)
    for t in range(T):
        s_new, v, q, g = model.step(X[:, t], s)
        dq = g * (1 - q * q)
        J = (1 - g)[..., None] * eye + dq[..., None] * R
        vv = v[:, :, None, :]
        Pq = dq[..., None] * vv
        if model.gated:
            dg = (q - s) * g * (1 - g)
            J = J + dg[..., None] * U
            P = torch.stack([Pq, dg[..., None] * vv], -2)          # (B, nc, n, types, D)
        else:
            P = Pq[..., None, :]
        if mode == "rtrl":
            E = torch.einsum("bcjk,bckmd->bcjmd", J, E.reshape(B, nc, n, n * types, D)).reshape(B, nc, n, n, types, D)
            E[:, :, ii, ii] += P
        else:
            E = J[:, :, ii, ii][..., None, None] * E + P
        flat = s_new.reshape(B, -1)
        p = torch.softmax(flat @ Wo.T + bo, -1)
        y = F.one_hot(ctr[:, t].clamp(min=0), Wo.shape[0]).to(dt)
        e = (p - y) * valid[:, t:t + 1].to(dt) / N
        gWo += e.T @ flat
        gbo += e.sum(0)
        delta = (e @ Wo).reshape(B, nc, n)
        G += torch.einsum("bcj,bcjmd->cmd", delta, E.reshape(B, nc, n, n * types, D)).reshape(nc, n, types, D) \
            if mode == "rtrl" else torch.einsum("bci,bcimd->cimd", delta, E)
        s = s_new
    model.Wq.grad = G[:, :, 0].clone()
    if model.gated:
        model.Wg.grad = G[:, :, 1].clone()
    model.out.weight.grad, model.out.bias.grad = gWo, gbo


def autograd_grad(model, idx, valid, ctr, truncate):
    model.zero_grad()
    logits = model.out(model.run(idx, truncate=truncate))
    F.cross_entropy(logits[valid], ctr[valid]).backward()


def train(split_tr, split_ca, rule, epochs, gated=True):
    """V164's schedule. rule: 'rtrl' | 'diag' (forward sensitivities) | 'bptt' | 't1' (autograd references)."""
    t0 = time.process_time()
    sc = L4.Synth()
    model = MC(gated=gated)
    opt = torch.optim.AdamW(model.parameters(), lr=S["lr"], weight_decay=S["wd"])
    gen = torch.Generator().manual_seed(S["order_seed"])
    curve, best, since, ep_run = [], (np.inf, 0, None), 0, 0
    for ep in range(1, epochs + 1):
        ep_run = ep
        order = torch.randperm(len(split_tr["n"]), generator=gen).tolist()
        for i in range(0, len(order), S["batch_lines"]):
            idx, mask, pos = V.line_batches(split_tr, order[i:i + S["batch_lines"]])
            assert bool(mask.all())
            ctr, valid = L4.targets(split_tr, sc, pos)
            opt.zero_grad()
            if rule in ("bptt", "t1"):
                autograd_grad(model, idx, valid, ctr, truncate=rule == "t1")
            else:
                forward_grad(model, idx, valid, ctr, rule)
            torch.nn.utils.clip_grad_norm_(model.parameters(), S["clip"])
            opt.step()
        cs, cc = L4.evaluate(model, split_ca, sc)
        cb = float(cs.sum() / cc.sum())
        curve.append(round(cb, 5))
        if cb < best[0]:
            best, since = (cb, ep, {k: v.clone() for k, v in model.state_dict().items()}), 0
        else:
            since += 1
            if since >= S["patience"]:
                break
    model.load_state_dict(best[2])
    info = {"rule": rule, "gated": gated, "best_epoch": best[1], "epochs_run": ep_run, "calib_bpb": best[0],
            "calib_curve": curve, "cut_off_by_budget": best[1] == epochs and ep_run == epochs,
            "cpu_s": round(time.process_time() - t0, 1), "records": ep_run * len(split_tr["n"])}
    return model, info


def test(model, split_te, ref_line_sums=None):
    sums, counts = L4.evaluate(model, split_te, L4.Synth())
    out = {"test_bpb": float(sums.sum() / counts.sum()), "line_sums": sums.tolist(), "counts": counts.tolist()}
    if ref_line_sums is not None:
        out["boot_sd_vs_ref"] = L4.boot_sd(np.asarray(ref_line_sums), sums, counts)
    return out


def splits():
    return tuple(L4.synth_split(SET["task"], p) for p in ("train", "calib", "test"))


def dense_refs():
    tb = json.loads((ROOT / "minifly_v164_cola_dev_v1_20261026/results/testbed/TESTBED.json").read_text())
    return tb["results"][SET["task"]]["arms"]
