"""V166 library (design sections 1-4), torch environment: the fixed KC-like code and Hebbian memory, the line generator
for ONE-HOP and MIXED, the recurrent network with a memory read (LOOP, NO-LOOP, LOOP-NULL), BPTT training, evaluation
and the paired line bootstrap. Self-contained: no other delivery's code is imported."""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SET = json.loads((HERE / "SETTINGS.json").read_text())
MEM, LINE = SET["memory"], SET["line"]
NI, NKC, S = MEM["items"], MEM["n_kc"], MEM["kc_active"]
MK = {m: i for i, m in enumerate(LINE["markers"])}
D_IN = len(MK) + 2 * NI
NPAIR = 2 * LINE["chains"]
T_STUDY = NPAIR + LINE["study_distract"]
T_Q0 = T_STUDY + LINE["delay"]
T = T_Q0 + 3 * LINE["episodes"]
LOOP_ARMS = ("LOOP", "LOOP-NULL", "LOOP-SHUF")


# ---------------------------------------------------------------- memory code (design section 1)
def kc_codes():
    """(NI, NKC) binary KC codes: random PN pattern per item, each KC sums kc_fanin PN dims (weight 1), top-kc_active."""
    rng = np.random.default_rng(MEM["code_seed"])
    P = rng.standard_normal((NI, MEM["pn_dim"]))
    J = np.zeros((NKC, MEM["pn_dim"]))
    for k in range(NKC):
        J[k, rng.choice(MEM["pn_dim"], MEM["kc_fanin"], replace=False)] = 1.0
    act = P @ J.T
    codes = np.zeros_like(act)
    np.put_along_axis(codes, np.argsort(-act, axis=1, kind="stable")[:, :S], 1.0, axis=1)
    return codes


CODES = kc_codes()


# ---------------------------------------------------------------- lines (design section 2)
def make_world(world, part, task, n=None):
    """Lines of one split. Per-line generator default_rng([seed, i]), so the study part of a line does not depend on
    the task. Returns numpy arrays: x (n, T, D_IN), keys/vals/wstep (n, NPAIR), tgt/hop (n, T) (-1 / 0 off ANS steps),
    episodes: (n, 4, 5) rows (t_query, hop, cue, f(cue), f(f(cue)) or -1)."""
    seed, n_def = SET["worlds"][world][part]
    n = n_def if n is None else n
    x = np.zeros((n, T, D_IN), np.float32)
    keys, vals, wstep = (np.zeros((n, NPAIR), np.int64) for _ in range(3))
    tgt, hop = np.full((n, T), -1, np.int64), np.zeros((n, T), np.int64)
    eps = np.full((n, LINE["episodes"], 5), -1, np.int64)
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        ch = rng.permutation(NI)[:3 * LINE["chains"]].reshape(LINE["chains"], 3)
        pairs = np.array([(c[0], c[1]) for c in ch] + [(c[1], c[2]) for c in ch])[rng.permutation(NPAIR)]
        f = {int(a): int(b) for a, b in pairs}
        slots = np.sort(rng.choice(T_STUDY, NPAIR, replace=False))
        j = 0
        for t in range(T_Q0):
            if t < T_STUDY and j < NPAIR and slots[j] == t:
                a, b = pairs[j]
                x[i, t, MK["STUDY"]] = 1
                x[i, t, len(MK) + a] = 1
                x[i, t, len(MK) + NI + b] = 1
                keys[i, j], vals[i, j], wstep[i, j] = a, b, t
                j += 1
            else:
                x[i, t, MK["DISTRACT"]] = 1
                x[i, t, len(MK) + rng.integers(NI)] = 1
        heads, allkeys = ch[:, 0], pairs[:, 0]
        if task == "ONE-HOP":
            ep = [(1, int(a)) for a in rng.choice(allkeys, LINE["episodes"], replace=False)]
        elif task == "MIXED":
            q2 = rng.choice(heads, 2, replace=False)
            rest = np.array([k for k in allkeys if k not in set(q2.tolist())])
            q1 = rng.choice(rest, 2, replace=False)
            ep = [(2, int(a)) for a in q2] + [(1, int(a)) for a in q1]
            ep = [ep[k] for k in rng.permutation(len(ep))]
        else:
            raise ValueError(task)
        for e, (hp, a) in enumerate(ep):
            t = T_Q0 + 3 * e
            x[i, t, MK["Q1" if hp == 1 else "Q2"]] = 1
            x[i, t, len(MK) + a] = 1
            x[i, t + 1, MK["WAIT"]] = 1
            x[i, t + 2, MK["ANS"]] = 1
            b = f[a]
            c = f.get(b, -1)
            tgt[i, t + 2] = b if hp == 1 else c
            hop[i, t + 2] = hp
            eps[i, e] = (t, hp, a, b, c)
    vals_shuf = np.stack([vals[i][derangement(np.random.default_rng([SET["shuffle_seed"], seed, i]))] for i in range(n)])
    return {"x": x, "keys": keys, "vals": vals, "vals_shuf": vals_shuf, "wstep": wstep, "tgt": tgt, "hop": hop,
            "episodes": eps, "n": n, "world": world, "part": part, "task": task}


def derangement(rng):
    """A random permutation of the NPAIR pairs with no fixed point (rejection sampling); LOOP-SHUF's written values.
    Its own generator, so the lines themselves are unchanged by it."""
    while True:
        p = rng.permutation(NPAIR)
        if not np.any(p == np.arange(NPAIR)):
            return p


def memory_tensors(w, idx, arm, dtype=torch.float32):
    """Key codes K (B, NPAIR, NKC) scaled by 1/S, value one-hots V (B, NPAIR, NI), write steps (B, NPAIR).
    LOOP-NULL writes nothing (V = 0); LOOP-SHUF writes the deranged values (same keys, same multiset of values)."""
    K = torch.from_numpy(CODES[w["keys"][idx]] / S).to(dtype)
    V = F.one_hot(torch.from_numpy(w["vals_shuf" if arm == "LOOP-SHUF" else "vals"][idx]), NI).to(dtype)
    if arm == "LOOP-NULL":
        V = torch.zeros_like(V)
    return K, V, torch.from_numpy(w["wstep"][idx])


def read_factored(K, V, wstep, q, t):
    """r_t = sum_i [w_i <= t] e_{b_i} (k(a_i) . q_t) / S   (K already scaled by 1/S)."""
    a = torch.einsum("bpk,bk->bp", K, q) * (wstep <= t).to(q.dtype)
    return torch.einsum("bp,bpn->bn", a, V)


def read_explicit(K, V, wstep, q, t):
    """r_t = M_t q_t with M_t = sum_i [w_i <= t] e_{b_i} k(a_i)^T / S built explicitly (check T0-a)."""
    m = (wstep <= t).to(q.dtype)[:, :, None, None]
    M = (m * V[:, :, :, None] * K[:, :, None, :]).sum(1)          # (B, NI, NKC)
    return torch.bmm(M, q[:, :, None])[:, :, 0]


# ---------------------------------------------------------------- network (design section 3)
def n_params(H, loop):
    return H * H + H * (D_IN + 1 + (NI if loop else 0) + NKC + NI) + NKC + NI * NI + NI


def width(arm):
    """LOOP-type arms use H; NO-LOOP uses the smallest width whose parameter count is at least LOOP's."""
    H = SET["net"]["H"]
    if arm in LOOP_ARMS:
        return H
    target = n_params(H, True)
    h = H
    while n_params(h, False) < target:
        h += 1
    return h


class Net(torch.nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm, self.loop = arm, arm in LOOP_ARMS
        H = width(arm)
        gen = torch.Generator().manual_seed(SET["net"]["init_seed"])

        def w(o, i):
            return torch.nn.Parameter(torch.randn(o, i, generator=gen) / math.sqrt(i))
        self.Wx, self.Wh = w(H, D_IN), w(H, H)
        self.bh = torch.nn.Parameter(torch.zeros(H))
        if self.loop:
            self.Wr = w(H, NI)
        self.Wq = w(NKC, H)
        self.bq = torch.nn.Parameter(torch.zeros(NKC))
        self.Wo, self.Wor = w(NI, H), w(NI, NI)
        self.bo = torch.nn.Parameter(torch.zeros(NI))

    def forward(self, x, K, V, wstep, collect=False, cut=False):
        """Logits (B, T, NI); collect=True also returns states (B, T, H) and queries (B, T, NKC).
        cut=True ignores the r -> h path (LOOP-CUT, a test-time ablation)."""
        B = x.shape[0]
        U = x @ self.Wx.T + self.bh
        h = x.new_zeros(B, self.Wh.shape[0])
        r = x.new_zeros(B, NI)
        outs, hs, qs = [], [], []
        for t in range(x.shape[1]):
            pre = U[:, t] + h @ self.Wh.T
            if self.loop and not cut:
                pre = pre + r @ self.Wr.T
            h = torch.tanh(pre)
            q = torch.relu(h @ self.Wq.T + self.bq)
            r = read_factored(K, V, wstep, q, t)
            outs.append(h @ self.Wo.T + r @ self.Wor.T + self.bo)
            if collect:
                hs.append(h)
                qs.append(q)
        out = torch.stack(outs, 1)
        return (out, torch.stack(hs, 1), torch.stack(qs, 1)) if collect else out


# ---------------------------------------------------------------- training and evaluation (design sections 3-4)
def batch(w, idx, arm, dtype=torch.float32):
    K, V, ws = memory_tensors(w, idx, arm, dtype)
    return torch.from_numpy(w["x"][idx]).to(dtype), K, V, ws, torch.from_numpy(w["tgt"][idx])


def evaluate(net, w, arm, bs=100, cut=False):
    """Per-line bit sums, counts and correct counts at Q1 and Q2 ANS steps, plus the mean over all scored steps.
    arm selects the memory written (a trained net can be read with another arm's memory: a test-time ablation)."""
    n = w["n"]
    out = {k: np.zeros(n) for k in ("q1_sum", "q2_sum", "q1_n", "q2_n", "q1_ok", "q2_ok")}
    with torch.no_grad():
        for i in range(0, n, bs):
            idx = np.arange(i, min(i + bs, n))
            x, K, V, ws, tg = batch(w, idx, arm)
            lp = torch.log_softmax(net(x, K, V, ws, cut=cut).double(), -1)
            hop = torch.from_numpy(w["hop"][idx])
            for hp in (1, 2):
                m = hop == hp
                bits = torch.zeros_like(hop, dtype=torch.float64)
                ok = torch.zeros_like(hop, dtype=torch.float64)
                tt = tg.clamp(min=0)
                bits[m] = -lp.gather(-1, tt[..., None])[..., 0][m] / math.log(2)
                ok[m] = (lp.argmax(-1) == tt)[m].double()
                out[f"q{hp}_sum"][idx] = bits.sum(1).numpy()
                out[f"q{hp}_n"][idx] = m.sum(1).numpy()
                out[f"q{hp}_ok"][idx] = ok.sum(1).numpy()
    tot = out["q1_n"].sum() + out["q2_n"].sum()
    out["all_bits"] = float((out["q1_sum"].sum() + out["q2_sum"].sum()) / tot)
    return out


def summary(ev):
    s = {}
    for hp in (1, 2):
        c = ev[f"q{hp}_n"].sum()
        if c:
            s[f"q{hp}_bits"] = float(ev[f"q{hp}_sum"].sum() / c)
            s[f"q{hp}_acc"] = float(ev[f"q{hp}_ok"].sum() / c)
    s["all_bits"] = ev["all_bits"]
    return s


def train(arm, tr, ca, max_epochs=None, patience=None, log=None):
    """Full BPTT; AdamW; batches of lines; clip; best calibration epoch (all scored steps) kept."""
    t0 = time.process_time()
    s = SET["train"]
    max_epochs = s["max_epochs"] if max_epochs is None else max_epochs
    patience = s["patience"] if patience is None else patience
    net = Net(arm)
    opt = torch.optim.AdamW(net.parameters(), lr=s["lr"], weight_decay=s["wd"])
    gen = torch.Generator().manual_seed(s["order_seed"])
    curve, best, since, epochs = [], (np.inf, 0, None), 0, 0
    for ep in range(1, max_epochs + 1):
        epochs = ep
        order = torch.randperm(tr["n"], generator=gen).numpy()
        for i in range(0, tr["n"], s["batch"]):
            x, K, V, ws, tg = batch(tr, np.sort(order[i:i + s["batch"]]), arm)
            m = tg >= 0
            opt.zero_grad()
            F.cross_entropy(net(x, K, V, ws)[m], tg[m]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), s["clip"])
            opt.step()
        cb = evaluate(net, ca, arm)["all_bits"]
        curve.append(round(cb, 5))
        if log:
            print(f"{arm} ep {ep} calib {cb:.4f} cpu {time.process_time() - t0:.0f}", file=log, flush=True)
        if cb < best[0]:
            best, since = (cb, ep, {k: v.clone() for k, v in net.state_dict().items()}), 0
        else:
            since += 1
            if since >= patience:
                break
    net.load_state_dict(best[2])
    return net, {"arm": arm, "width": width(arm), "params": sum(p.numel() for p in net.parameters()),
                 "best_epoch": best[1], "epochs_run": epochs, "calib_bits": best[0], "calib_curve": curve,
                 "cut_off_by_budget": best[1] == max_epochs, "cpu_s": round(time.process_time() - t0, 1)}


def boot_sd(a_sums, b_sums, counts):
    """Paired line bootstrap SD of (mean bits of A - mean bits of B)."""
    rng = np.random.default_rng(SET["boot"]["seed"])
    d = []
    for _ in range(SET["boot"]["reps"]):
        i = rng.integers(0, len(counts), len(counts))
        d.append((a_sums[i].sum() - b_sums[i].sum()) / counts[i].sum())
    return float(np.std(d))


def targeting(net, w, arm, bs=100):
    """Design section 4 diagnostic: at the Q, WAIT and ANS steps of every Q2 episode, which of the line's keys has the
    largest k_i . q_t; share of episodes where it is the cue a and where it is the intermediate b = f(a)."""
    hit = {s: {"cue": 0, "intermediate": 0} for s in ("Q", "WAIT", "ANS")}
    cnt = 0
    with torch.no_grad():
        for i in range(0, w["n"], bs):
            idx = np.arange(i, min(i + bs, w["n"]))
            x, K, V, ws, _ = batch(w, idx, arm)
            _, _, q = net(x, K, V, ws, collect=True)
            sc = torch.einsum("bpk,btk->btp", K, q).numpy()            # (B, T, NPAIR)
            for j, li in enumerate(idx):
                for t, hp, a, b, c in w["episodes"][li]:
                    if hp != 2:
                        continue
                    cnt += 1
                    for k, s in enumerate(("Q", "WAIT", "ANS")):
                        top = w["keys"][li][int(np.argmax(sc[j, t + k]))]
                        hit[s]["cue"] += int(top == a)
                        hit[s]["intermediate"] += int(top == b)
    return {s: {k: v / cnt for k, v in d.items()} for s, d in hit.items()} | {"episodes": cnt}


def oracle(w):
    """Reads with the true codes: b^ = argmax M k(a), c^ = argmax M k(b^); accuracy per hop type (T0-b)."""
    ok, cnt = {1: 0, 2: 0}, {1: 0, 2: 0}
    for i in range(w["n"]):
        M = np.zeros((NI, NKC))
        for a, b in zip(w["keys"][i], w["vals"][i]):
            M[b] += CODES[a] / S
        for t, hp, a, b, c in w["episodes"][i]:
            if t < 0:
                continue
            bh = int(np.argmax(M @ CODES[a]))
            pred = bh if hp == 1 else int(np.argmax(M @ CODES[bh]))
            ok[hp] += int(pred == (b if hp == 1 else c))
            cnt[hp] += 1
    return {f"q{h}_acc": ok[h] / cnt[h] for h in (1, 2) if cnt[h]} | {f"q{h}_n": cnt[h] for h in (1, 2)}
