"""E19: births of kbank2 (arch19.py) by backprop through time and by the forward-only rule, with the acceptance tests.

Usage:
    python3 exp19_kbank.py check                 tests 1 and 2 on three random initialisations -> exp45/E19_CHECK.json
    python3 exp19_kbank.py check_births          test 1 again on the three Stage A births, and test 3 (alignment) -> exp45/E19_CHECK_BIRTHS.json
    python3 exp19_kbank.py dev <arm> <seed>      arm in bptt | r1 | r2  -> exp45/e19_dev_kbank2_<arm><DEVTAG>_<seed>.{json,pt}

Derivation (exact; no step looks ahead). Per item, with g_t = dL/dlogits_t (zero off the answer mask), F(W) the feedback matrix used in
place of W in every backward relay (W itself for the exact rule, sign(W) * mean|W| under weight-sign feedback, as SignLinear):

    readout, within the step:   ga_t = (F(W3)^T g_t) * (1 - z_t^2);  dW3 += g_t z_t^T, db3 += g_t, dWm1 += ga_t rho1_t^T,
                                dWm2 += ga_t rho2_t^T, dE[:, x_t] += ga_t
    errors at each bank:        d1_t = (F(Wm1)^T ga_t) * rho1'_t,   d2_t = (F(Wm2)^T ga_t) * rho2'_t
    second bank (exact, forward): h2_t = sum_{s<=t} R2^{t-s} (W2 rho1_s + b2), so
                                dW2[c, j] += d2_t[c] e2_t[g(c), j],   e2_t[g, j] = r_g e2_{t-1}[g, j] + rho1_t[j]
                                db2[c]   += d2_t[c] eb_t[g(c)],      eb_t[g]    = r_g eb_{t-1}[g] + 1
    first bank: the total error on h1_t is d1_t + rho1'_t * (W2^T Lambda2_t) + R1 Lambda1_{t+1}, with Lambda2_t = sum_{tau>=t} R2^{tau-t} d2_tau
                (future). Exchanging the sums over t and tau turns the future error into a trace carried forward:
                                dA[j, b] += d1_t[j] e1_t[j, b] + sum_g K^g_t[j] zeta^g_t[j, b]
                                e1_t[j, b]     = r1_j e1_{t-1}[j, b] + [x_t = b]
                                zeta^g_t[j, b] = r_g zeta^g_{t-1}[j, b] + rho1'_t[j] e1_t[j, b]          (one per upper rate g)
                                K^g_t[j]       = sum_{c in group g} d2_t[c] F(W2)[c, j]
    This equals backprop through time exactly when F(W) = W (h1 is linear in A and rho1' is evaluated at the step it applies to).
    R2 arm ("no deep traces"): zeta^g_t is replaced by its current-step term rho1'_t e1_t, i.e. the first bank hears the second bank's
    error of the current step only.
Traces held per sample: e1 (5 x U), zeta (5 x C x U) for the U bytes present in THAT sample (exactly zero for the others), e2 (5 x C), eb (5).
The training implementation runs one forward pass over the batch's steps holding only the two states and the traces; the streaming
version (`Stream`, test 2) feeds one item one byte at a time.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch as T
from torch.nn.functional import cross_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp11_cell as E
import exp45 as X
import arch19 as A19

OUT = E.OUT
C, H, NG = A19.C, A19.H, A19.NG
GIDX = A19.group_index(C)                      # group of each unit (both banks use the same assignment)
RG = T.tensor(A19.RATES)


def fb(W, sign):
    return W.sign() * W.abs().mean() if sign else W


def opt_for(m, wd):
    tab = [p for n, p in m.named_parameters() if n in A19.TABLES]
    rest = [p for n, p in m.named_parameters() if n not in A19.TABLES]
    return T.optim.AdamW([{"params": rest}, {"params": tab, "weight_decay": 0.0}], lr=.001, weight_decay=wd, betas=(.9, .98))


@T.no_grad()
def rule_grads(m, x, y, mask, sign=True, deep=True):
    """Forward-only rule over a batch: one pass in time, states and traces only. Writes .grad on every parameter; returns the loss."""
    B, L = x.shape
    dt = m.A.dtype
    N = int(mask.sum())
    # per-sample byte slots: each sample's traces cover only the bytes that sample contains (exactly zero for all others)
    rows_u = [T.unique(x[b]) for b in range(B)]
    Ub = max(len(u) for u in rows_u)
    slot = T.stack([T.searchsorted(rows_u[b], x[b]) for b in range(B)])          # (B, L): local slot of x[b, t]
    F3, Fm1, Fm2, F2 = fb(m.l3.weight, sign), fb(m.Wm1.weight, sign), fb(m.Wm2.weight, sign), fb(m.W2.weight, sign)
    F2g = [F2[GIDX == gi] for gi in range(NG)]
    cs_g = [(GIDX == gi).nonzero().squeeze(1) for gi in range(NG)]
    r1, r2 = m.R1, m.R2
    rg = RG.to(dt)
    g_of = GIDX
    h1 = T.zeros(B, C, dtype=dt); h2 = T.zeros(B, C, dtype=dt)
    e1g = T.zeros(B, NG, Ub, dtype=dt)                  # e1 depends on j only through r1_j's group
    zeta = T.zeros(B, NG, C, Ub, dtype=dt)
    e2 = T.zeros(B, NG, C, dtype=dt); eb = T.zeros(B, NG, dtype=dt)
    gW3 = T.zeros_like(m.l3.weight); gb3 = T.zeros_like(m.l3.bias); gWm1 = T.zeros_like(m.Wm1.weight); gWm2 = T.zeros_like(m.Wm2.weight)
    gE = T.zeros_like(m.Etab); gW2 = T.zeros_like(m.W2.weight); gb2 = T.zeros_like(m.W2.bias); gAloc = T.zeros(B, C, Ub, dtype=dt)
    loss = 0.0
    eyeU = T.eye(Ub, dtype=dt)
    rz = rg[None, :, None, None]
    for t in range(L):
        xt = x[:, t]
        e1g.mul_(rg[None, :, None]).add_(eyeU[slot[:, t]][:, None, :])
        h1 = r1 * h1 + m.A[:, xt].t()
        p1 = T.tanh(h1); d1p = 1 - p1 * p1
        h2 = r2 * h2 + p1 @ m.W2.weight.t() + m.W2.bias
        p2 = T.tanh(h2); d2p = 1 - p2 * p2
        z = T.tanh(p1 @ m.Wm1.weight.t() + p2 @ m.Wm2.weight.t() + m.Etab[:, xt].t())
        e2.mul_(rg[None, :, None]).add_(p1[:, None, :])
        eb.mul_(rg[None, :]).add_(1)
        e1j = e1g[:, g_of, :]                            # (B, C, Ub): e1_t[j, b] for each first-bank unit j
        if deep:
            zeta.mul_(rz).add_((d1p[:, :, None] * e1j)[:, None])
        mt = mask[:, t]
        if not bool(mt.any()):
            continue
        sel = mt.nonzero().squeeze(1)
        logits = z[sel] @ m.l3.weight.t() + m.l3.bias
        loss += float(cross_entropy(logits, y[sel, t], reduction="sum"))
        g = T.softmax(logits, 1); g[T.arange(len(sel)), y[sel, t]] -= 1; g /= N    # dL/dlogits on the masked rows
        zs, p1s, p2s = z[sel], p1[sel], p2[sel]
        gW3 += g.t() @ zs; gb3 += g.sum(0)
        ga = (g @ F3) * (1 - zs * zs)
        gWm1 += ga.t() @ p1s; gWm2 += ga.t() @ p2s
        gE.index_add_(1, xt[sel], ga.t())
        d1 = (ga @ Fm1) * d1p[sel]
        d2 = (ga @ Fm2) * d2p[sel]
        for gi in range(NG):
            cs = cs_g[gi]
            gW2[cs] += d2[:, cs].t() @ e2[sel, gi]
            gb2[cs] += (d2[:, cs] * eb[sel, gi][:, None]).sum(0)
        e1s = e1j[sel]
        gAloc[sel] += d1[:, :, None] * e1s
        K = T.stack([d2[:, cs_g[gi]] @ F2g[gi] for gi in range(NG)], 1)      # (b, G, C): K^g_t[j]
        if deep:
            gAloc[sel] += T.einsum("bgj,bgju->bju", K, zeta[sel])
        else:
            gAloc[sel] += (K.sum(1) * d1p[sel])[:, :, None] * e1s
    gA = T.zeros_like(m.A)
    for b in range(B):
        gA.index_add_(1, rows_u[b], gAloc[b, :, :len(rows_u[b])])
    for p_, g_ in ((m.l3.weight, gW3), (m.l3.bias, gb3), (m.Wm1.weight, gWm1), (m.Wm2.weight, gWm2), (m.Etab, gE),
                   (m.W2.weight, gW2), (m.W2.bias, gb2), (m.A, gA)):
        p_.grad = g_
    return loss / N


class Stream:
    """Test 2: the same rule fed ONE item ONE byte at a time, holding only the two bank states and the traces (dense over all 258 bytes)."""

    def __init__(self, m, sign, deep, scale):
        self.m, self.sign, self.deep, self.scale = m, sign, deep, scale
        dt = m.A.dtype
        self.h1 = T.zeros(C, dtype=dt); self.h2 = T.zeros(C, dtype=dt)
        self.e1 = T.zeros(C, A19.NSYM, dtype=dt); self.zeta = T.zeros(NG, C, A19.NSYM, dtype=dt)
        self.e2 = T.zeros(NG, C, dtype=dt); self.eb = T.zeros(NG, dtype=dt)

    @T.no_grad()
    def step(self, b, target, acc):
        m = self.m; rg = RG.to(m.A.dtype)
        self.e1 = m.R1[:, None] * self.e1; self.e1[:, b] += 1
        self.h1 = m.R1 * self.h1 + m.A[:, b]
        p1 = T.tanh(self.h1); d1p = 1 - p1 * p1
        self.h2 = m.R2 * self.h2 + m.W2.weight @ p1 + m.W2.bias
        p2 = T.tanh(self.h2); d2p = 1 - p2 * p2
        z = T.tanh(m.Wm1.weight @ p1 + m.Wm2.weight @ p2 + m.Etab[:, b])
        self.e2 = rg[:, None] * self.e2 + p1[None]
        self.eb = rg * self.eb + 1
        if self.deep:
            self.zeta = rg[:, None, None] * self.zeta + (d1p[:, None] * self.e1)[None]
        if target is None:
            return
        logits = m.l3.weight @ z + m.l3.bias
        g = T.softmax(logits, 0); g[target] -= 1; g *= self.scale
        acc["l3.weight"] += T.outer(g, z); acc["l3.bias"] += g
        ga = (fb(m.l3.weight, self.sign).t() @ g) * (1 - z * z)
        acc["Wm1.weight"] += T.outer(ga, p1); acc["Wm2.weight"] += T.outer(ga, p2); acc["Etab"][:, b] += ga
        d1 = (fb(m.Wm1.weight, self.sign).t() @ ga) * d1p
        d2 = (fb(m.Wm2.weight, self.sign).t() @ ga) * d2p
        F2 = fb(m.W2.weight, self.sign)
        for c in range(C):            # one second-bank unit at a time: its own rate's trace
            gi = int(GIDX[c])
            acc["W2.weight"][c] += d2[c] * self.e2[gi]
            acc["W2.bias"][c] += d2[c] * self.eb[gi]
        acc["A"] += d1[:, None] * self.e1
        K = T.stack([F2[GIDX == gi].t() @ d2[GIDX == gi] for gi in range(NG)], 0)       # (G, C)
        if self.deep:
            acc["A"] += (K[:, :, None] * self.zeta).sum(0)
        else:
            acc["A"] += (K.sum(0) * d1p)[:, None] * self.e1


def stream_grads(m, x, y, mask, sign, deep):
    acc = {n: T.zeros_like(p) for n, p in m.named_parameters()}
    N = int(mask.sum())
    for i in range(x.shape[0]):
        s = Stream(m, sign, deep, 1.0 / N)
        for t in range(x.shape[1]):
            s.step(int(x[i, t]), int(y[i, t]) if bool(mask[i, t]) else None, acc)
    return acc


def bptt_grads(m, x, y, mask):
    m.zero_grad()
    logits, _ = A19.forward(m, x)
    loss = cross_entropy(logits[mask], y[mask])
    loss.backward()
    return {n: p.grad.detach().clone() for n, p in m.named_parameters()}, float(loss.detach())


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def dev_batch(seed, k=0):
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    bt = E.batches(tp, seed, E.KMAX["M"])
    for _ in range(k):
        next(bt)
    return X.batch_tensor(next(bt))


def check():
    """Tests 1 and 2 on three random initialisations (float64 and float32), plus test 2 in float64 on a subset of items."""
    out = {"definition": "relative error = max |rule - reference| / max |reference|, per parameter tensor", "inits": []}
    for seed in (73129, 93683, 116089):
        for dt in (T.float64, T.float32):
            m = A19.new_net(seed + 1).to(dt)             # a random init, not the birth init (seed + 1)
            x, y, mask = dev_batch(seed)
            ref, loss_bp = bptt_grads(m, x, y, mask)
            loss_r = rule_grads(m, x, y, mask, sign=False, deep=True)
            rows = {n: rel(p.grad, ref[n]) for n, p in m.named_parameters()}
            entry = dict(seed_init=seed + 1, dtype=str(dt), batch=list(x.shape), loss_bptt=loss_bp, loss_rule=loss_r, test1_rel_err=rows,
                         test1_max=max(rows.values()), test1_pass=max(rows.values()) < 1e-4)
            if dt == T.float64:
                batched = {n: p.grad.clone() for n, p in m.named_parameters()}
                sub = slice(0, 4)                          # streaming is slow in Python: four items of the batch
                xs, ys, ms = x[sub], y[sub], mask[sub]
                rule_grads(m, xs, ys, ms, sign=False, deep=True); b4 = {n: p.grad.clone() for n, p in m.named_parameters()}
                st = stream_grads(m, xs, ys, ms, sign=False, deep=True)
                rows2 = {n: rel(st[n], b4[n]) for n in st}
                rule_grads(m, xs, ys, ms, sign=True, deep=False); b4s = {n: p.grad.clone() for n, p in m.named_parameters()}
                st2 = stream_grads(m, xs, ys, ms, sign=True, deep=False)
                rows3 = {n: rel(st2[n], b4s[n]) for n in st2}
                rule_grads(m, xs, ys, ms, sign=True, deep=True); b4d = {n: p.grad.clone() for n, p in m.named_parameters()}
                st3 = stream_grads(m, xs, ys, ms, sign=True, deep=True)
                rows4 = {n: rel(st3[n], b4d[n]) for n in st3}
                entry.update(test2_items=4, test2_exact_deep=rows2, test2_sign_nodeep=rows3, test2_sign_deep=rows4,
                             test2_max=max(max(rows2.values()), max(rows3.values()), max(rows4.values())))
                entry["test2_pass"] = entry["test2_max"] < 1e-4
                print("init", seed + 1, "test1 max", entry["test1_max"], "test2 max", entry["test2_max"], flush=True)
            else:
                print("init", seed + 1, dt, "test1 max", entry["test1_max"], flush=True)
            out["inits"].append(entry)
    out["test1_pass_all"] = all(e["test1_pass"] for e in out["inits"])
    out["test2_pass_all"] = all(e.get("test2_pass", True) for e in out["inits"])
    out["n_params"] = sum(p.numel() for p in A19.new_net(1).parameters())
    (OUT / "E19_CHECK.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("test1_pass_all", "test2_pass_all", "n_params")}))


def check_births():
    """Test 1 on the three Stage A births (float32 weights cast to float64), and test 3: cosine of the sign rule with and without deep
    traces against the true gradient, per parameter, on 4 development batches per seed."""
    out = {"test1": [], "test3": []}
    for seed in (73129, 93683, 116089):
        m = A19.KBank2(); m.load_state_dict(T.load(OUT / f"e19_dev_kbank2_bptt_{seed}.pt")); m = m.double()
        x, y, mask = dev_batch(seed, 7)
        ref, _ = bptt_grads(m, x, y, mask)
        rule_grads(m, x, y, mask, sign=False, deep=True)
        rows = {n: rel(p.grad, ref[n]) for n, p in m.named_parameters()}
        out["test1"].append(dict(seed=seed, rel_err=rows, max=max(rows.values()), passed=max(rows.values()) < 1e-4))
        acc = {arm: {n: [] for n, _ in m.named_parameters()} for arm in ("sign_deep", "sign_nodeep", "exact_nodeep")}
        for k in range(4):
            x, y, mask = dev_batch(seed, 11 + k)
            ref, _ = bptt_grads(m, x, y, mask)
            for arm, sg, dp in (("sign_deep", True, True), ("sign_nodeep", True, False), ("exact_nodeep", False, False)):
                rule_grads(m, x, y, mask, sign=sg, deep=dp)
                for n, p in m.named_parameters():
                    acc[arm][n].append((cos(p.grad, ref[n]), float(p.grad.norm() / ref[n].norm().clamp_min(1e-30))))
        out["test3"].append(dict(seed=seed, batches=4, cos_and_norm_ratio={arm: {n: [sum(v[0] for v in vs) / len(vs), sum(v[1] for v in vs) / len(vs)]
                                                                                for n, vs in d.items()} for arm, d in acc.items()}))
        print(seed, "test1 max", out["test1"][-1]["max"], "A cos sign_deep / sign_nodeep:",
              out["test3"][-1]["cos_and_norm_ratio"]["sign_deep"]["A"][0], out["test3"][-1]["cos_and_norm_ratio"]["sign_nodeep"]["A"][0], flush=True)
    (OUT / "E19_CHECK_BIRTHS.json").write_text(json.dumps(out, indent=1))


def develop(arm, seed):
    assert arm in ("bptt", "r1", "r2")
    A19.install()
    stem = f"e19_dev_kbank2_{arm}{E.DEVTAG}_{seed}"
    path = OUT / f"{stem}.pt"
    assert not path.exists(), f"{path} exists"
    m = A19.new_net(seed)
    opt = opt_for(m, .3 if arm == "bptt" else .25)
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    pan = {E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV}
    bt = E.batches(tp, seed, E.KMAX["M"])
    t0 = time.process_time(); curve = []; tupd = 0.0

    def step(items):
        x, y, mask = X.batch_tensor(items)
        if arm == "bptt":
            opt.zero_grad()
            logits, _ = A19.forward(m, x)
            loss = cross_entropy(logits[mask], y[mask]); loss.backward(); loss = float(loss)
        else:
            loss = rule_grads(m, x, y, mask, sign=True, deep=(arm == "r1"))
        opt.step()
        return loss

    for stop in range(X.BLOCK, E.DEV_DOSE + 1, X.BLOCK):
        u0 = time.process_time()
        losses = [step(next(bt)) for _ in range(X.BLOCK)]
        tupd += time.process_time() - u0
        if stop % 4096 == 0 or stop == E.DEV_DOSE:
            row = dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0, cpu_ms_per_update=1000 * tupd / stop,
                       sealed={n: {k: E.score(m, it) for k, it in pk.items()} for n, pk in pan.items()})
            curve.append(row)
            print("e19 dev", arm, seed, stop, {n: [v[k]["exact"] for k in (0, 4, 8, 12, 20)] for n, v in row["sealed"].items()},
                  f"loss {row['loss']:.3f} cpu {row['cpu']:.0f}s", flush=True)
    row["silenced_k12"] = {n: E.score(m, pk[12], silence=True) for n, pk in pan.items()}
    T.save(m.state_dict(), path)
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        peak = None
    (OUT / f"{stem}.json").write_text(json.dumps(dict(arch="kbank2", arm=arm, seed=seed, kmax=E.KMAX["M"], decay=opt.defaults["weight_decay"],
                                                      n_params=sum(p.numel() for p in m.parameters()), peak_rss_kb=peak,
                                                      feedback=("exact (autograd BPTT)" if arm == "bptt" else "sign(W)*mean|W|"),
                                                      deep_traces=(arm == "r1") if arm != "bptt" else None, curve=curve), indent=1))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if sys.argv[1] == "check":
        check()
    elif sys.argv[1] == "check_births":
        check_births()
    elif sys.argv[1] == "dev":
        develop(sys.argv[2], int(sys.argv[3]))
