"""V170 library (design sections 1-4), torch environment. Loads V166's v166_lib.py read-only BY FILE PATH (V166's
folder is never put on sys.path, so V170's own runtime.py cannot be shadowed). Provides: the COMPOSE-ORD generator
(ordered instruction pairs, held-out pairs for test-H), V166's PN patterns / PN->KC wiring / KC codes, the fixed
re-entry circuit's state dynamics (A, A-SHUF, A-CUT; no learned parameter), the A readout+gate module, V166's LOOP
cell re-implemented for any input width (with TBPTT-1 detach points), evaluation, the paired line bootstrap and a
training loop that takes any gradient function (every arm shares data order, AdamW, clipping and best-calibration
selection, and differs only in the gradient estimate)."""
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

_spec = importlib.util.spec_from_file_location("v166_lib", ROOT / SET["v166_lib"])
L166 = importlib.util.module_from_spec(_spec)
sys.modules["v166_lib"] = L166
_spec.loader.exec_module(L166)

NI, NKC, S = L166.NI, L166.NKC, L166.S
MEM, LINE = L166.MEM, L166.LINE
MK = {m: i for i, m in enumerate(SET["markers"])}
NTOK = SET["n_tokens"]
OFF_CUE, OFF_VAL, OFF_TOK = len(MK), len(MK) + NI, len(MK) + 2 * NI
D_IN = len(MK) + 2 * NI + NTOK                       # 75
NPAIR = 2 * LINE["chains"]
T_STUDY = NPAIR + LINE["study_distract"]
T_Q0 = T_STUDY + LINE["delay"]
T = T_Q0 + 3 * LINE["episodes"]                      # 34
ANS = [T_Q0 + 3 * e + 2 for e in range(LINE["episodes"])]
LAGS = 3
D_Z = LAGS * D_IN


# ---------------------------------------------------------------- fixed code (V166's, design section 2)
def pn_and_wiring():
    """V166's PN patterns P (NI, pn_dim) and PN->KC wiring J (NKC, pn_dim), same generator calls as v166_lib.kc_codes."""
    rng = np.random.default_rng(MEM["code_seed"])
    P = rng.standard_normal((NI, MEM["pn_dim"]))
    J = np.zeros((NKC, MEM["pn_dim"]))
    for k in range(NKC):
        J[k, rng.choice(MEM["pn_dim"], MEM["kc_fanin"], replace=False)] = 1.0
    return P, J


P_NP, J_NP = pn_and_wiring()


def topk_code(act):
    """Binary top-S indicator per row (stable argsort, as V166); rows that are all zero stay zero."""
    codes = np.zeros_like(act)
    np.put_along_axis(codes, np.argsort(-act, axis=1, kind="stable")[:, :S], 1.0, axis=1)
    return codes


CODES = topk_code(P_NP @ J_NP.T)                     # k(x) = top16(J p(x)) for every item


def reenter(r, Pmat):
    """K(r) = top16(J Pmat^T r) (rows of r: MBON patterns over items; Pmat (B, NI, pn) or (NI, pn) PN patterns per
    MBON). Zero where sum(r) <= 1e-9. The same expansion as the input encoding. numpy float64."""
    pn = np.einsum("bi,bip->bp", r, Pmat) if Pmat.ndim == 3 else r @ Pmat
    code = topk_code(pn @ J_NP.T)
    code[r.sum(1) <= 1e-9] = 0.0
    return code


# ---------------------------------------------------------------- lines (design section 1)
def pair_lists():
    ho = {tuple(p) for p in SET["heldout_pairs"]}
    asc = [(u, v) for u in range(NTOK) for v in range(u + 1, NTOK)]
    return {"T": [p for p in asc if p not in ho], "H": [p for p in asc if p in ho]}


PAIRS = pair_lists()


def derangement(rng, n):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def make_world(world, part, pairs="T", n=None, seeds=None):
    """Lines of one split. Structure from default_rng([seed, i]); instruction pair index from default_rng([seed, i, 1])
    (same uniform draws for test-T and test-H, so those lines differ only in the tokens). A-SHUF's item derangement
    from default_rng([shuffle_seed, seed, i])."""
    seed, n_def = (seeds or SET["worlds"])[world][part]
    n = n_def if n is None else n
    x = np.zeros((n, T, D_IN), np.float32)
    cue = np.full((n, T), -1, np.int64)
    keys, vals, wstep = (np.zeros((n, NPAIR), np.int64) for _ in range(3))
    tgt, hop = np.full((n, T), -1, np.int64), np.zeros((n, T), np.int64)
    eps = np.full((n, LINE["episodes"], 7), -1, np.int64)    # t_instr, hop, cue, f(cue), f(f(cue)), u, v
    perm = np.zeros((n, NI), np.int64)
    lst = PAIRS[pairs]
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        ch = rng.permutation(NI)[:3 * LINE["chains"]].reshape(LINE["chains"], 3)
        pr = np.array([(c[0], c[1]) for c in ch] + [(c[1], c[2]) for c in ch])[rng.permutation(NPAIR)]
        f = {int(a): int(b) for a, b in pr}
        slots = np.sort(rng.choice(T_STUDY, NPAIR, replace=False))
        j = 0
        for t in range(T_Q0):
            if t < T_STUDY and j < NPAIR and slots[j] == t:
                a, b = pr[j]
                x[i, t, MK["STUDY"]] = 1
                x[i, t, OFF_CUE + a] = 1
                x[i, t, OFF_VAL + b] = 1
                cue[i, t] = a
                keys[i, j], vals[i, j], wstep[i, j] = a, b, t
                j += 1
            else:
                c = rng.integers(NI)
                x[i, t, MK["DISTRACT"]] = 1
                x[i, t, OFF_CUE + c] = 1
                cue[i, t] = c
        heads, allkeys = ch[:, 0], pr[:, 0]
        q2 = rng.choice(heads, 2, replace=False)
        rest = np.array([k for k in allkeys if k not in set(q2.tolist())])
        q1 = rng.choice(rest, 2, replace=False)
        ep = [(2, int(a)) for a in q2] + [(1, int(a)) for a in q1]
        ep = [ep[k] for k in rng.permutation(len(ep))]
        irng = np.random.default_rng([seed, i, 1])
        draws = irng.random(len(ep))
        for e, (hp, a) in enumerate(ep):
            u, v = lst[int(draws[e] * len(lst))]
            if hp == 1:
                u, v = v, u                                   # descending pair: ONE-HOP
            t = T_Q0 + 3 * e
            x[i, t, MK["INSTR"]] = 1
            x[i, t, OFF_TOK + u] = 1
            x[i, t + 1, MK["QUERY"]] = 1
            x[i, t + 1, OFF_TOK + v] = 1
            x[i, t + 1, OFF_CUE + a] = 1
            cue[i, t + 1] = a
            x[i, t + 2, MK["ANS"]] = 1
            b = f[a]
            c = f.get(b, -1)
            tgt[i, t + 2] = b if hp == 1 else c
            hop[i, t + 2] = hp
            eps[i, e] = (t, hp, a, b, c, u, v)
        perm[i] = derangement(np.random.default_rng([SET["shuffle_seed"], seed, i]), NI)
    return {"x": x, "cue": cue, "keys": keys, "vals": vals, "wstep": wstep, "tgt": tgt, "hop": hop, "episodes": eps,
            "perm": perm, "n": n, "world": world, "part": part, "pairs": pairs}


def subset(w, n):
    return {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in w.items()} | {"n": n}


# ---------------------------------------------------------------- re-entry circuit dynamics (design section 2)
def memory_np(w, idx):
    K = CODES[w["keys"][idx]] / S                                  # (B, NPAIR, NKC)
    V = np.eye(NI)[w["vals"][idx]]                                 # (B, NPAIR, NI)
    return K, V, w["wstep"][idx]


def a_states(w, mode, idx=None, count=None):
    """Fixed dynamics of the re-entry circuit (no learned parameter): q_t = k(cue_t) at cue steps, else the re-entry
    K(r_{t-1}) (A), K with the line's item derangement (A-SHUF), or 0 (A-CUT); r_t = M_t q_t / 16.
    Returns r (B, T, NI) and q (B, T, NKC), float64. count (dict) receives the number of re-entry calls per row."""
    idx = np.arange(w["n"]) if idx is None else idx
    K, V, ws = memory_np(w, idx)
    B = len(idx)
    Pm = P_NP[w["perm"][idx]] if mode == "A-SHUF" else P_NP     # (B, NI, pn): PN pattern used for MBON i
    r = np.zeros((B, T, NI))
    q = np.zeros((B, T, NKC))
    rp = np.zeros((B, NI))
    calls = np.zeros(B, np.int64)
    for t in range(T):
        c = w["cue"][idx, t]
        has = c >= 0
        qt = np.zeros((B, NKC))
        qt[has] = CODES[c[has]]
        if (~has).any() and mode != "A-CUT":
            qt[~has] = reenter(rp[~has], Pm[~has] if Pm.ndim == 3 else Pm)
            calls[~has] += 1
        a = np.einsum("bpk,bk->bp", K, qt) * (ws <= t)
        rt = np.einsum("bp,bpn->bn", a, V)
        r[:, t], q[:, t], rp = rt, qt, rt
    if count is not None:
        count["calls"] = calls
    return r, q


def lagged(x):
    """z_t = [x_t; x_{t-1}; x_{t-2}] (zeros before the line start). x (B, T, D) tensor -> (B, T, 3D)."""
    z = [x]
    for k in range(1, LAGS):
        z.append(torch.cat([x.new_zeros(x.shape[0], k, x.shape[2]), x[:, :-k]], 1))
    return torch.cat(z, -1)


class ANet(torch.nn.Module):
    """Learned part of the re-entry circuit: gate g_t = sigmoid(w_g . z_t + b_g); y_t = W_o m_t + b_o with
    m_t = (1 - g_t) r_{t-1} + g_t r_t. Initialisation N(0, 1/fan-in), biases 0, seed 0."""

    def __init__(self):
        super().__init__()
        gen = torch.Generator().manual_seed(SET["train"]["init_seed"])
        self.wg = torch.nn.Parameter(torch.randn(D_Z, generator=gen) / math.sqrt(D_Z))
        self.bg = torch.nn.Parameter(torch.zeros(()))
        self.Wo = torch.nn.Parameter(torch.randn(NI, NI, generator=gen) / math.sqrt(NI))
        self.bo = torch.nn.Parameter(torch.zeros(NI))

    def parts(self, x, r):
        """Gate g (B, T), z (B, T, D_Z), r_prev (B, T, NI), m (B, T, NI), logits (B, T, NI)."""
        z = lagged(x)
        g = torch.sigmoid(z @ self.wg + self.bg)
        rprev = torch.cat([r.new_zeros(r.shape[0], 1, NI), r[:, :-1]], 1)
        m = (1 - g)[..., None] * rprev + g[..., None] * r
        return g, z, rprev, m, m @ self.Wo.T + self.bo

    def forward(self, x, r, gate=None):
        g, z, rprev, m, y = self.parts(x, r)
        if gate is not None:                               # test-time forced gate (diagnostic / oracle)
            g = torch.full_like(g, float(gate)) if not torch.is_tensor(gate) else gate
            m = (1 - g)[..., None] * rprev + g[..., None] * r
            y = m @ self.Wo.T + self.bo
        return y


def a_batch(w, idx, mode, dtype=torch.float32, cache=None):
    if cache is not None and mode in cache:
        r = cache[mode][idx]
    else:
        r, _ = a_states(w, mode, idx)
    return (torch.from_numpy(w["x"][idx]).to(dtype), torch.from_numpy(r).to(dtype),
            torch.from_numpy(w["tgt"][idx]))


def a_state_cache(w, mode):
    r, _ = a_states(w, mode)
    return {mode: r}


# ---------------------------------------------------------------- LOOP (V166's cell, any input width; design section 3)
class LoopNet(torch.nn.Module):
    """V166's LOOP Net with input width d_in; parameters created in V166's order from the same generator."""

    def __init__(self, d_in=D_IN):
        super().__init__()
        H = L166.SET["net"]["H"]
        gen = torch.Generator().manual_seed(L166.SET["net"]["init_seed"])

        def w(o, i):
            return torch.nn.Parameter(torch.randn(o, i, generator=gen) / math.sqrt(i))
        self.Wx, self.Wh = w(H, d_in), w(H, H)
        self.bh = torch.nn.Parameter(torch.zeros(H))
        self.Wr = w(H, NI)
        self.Wq = w(NKC, H)
        self.bq = torch.nn.Parameter(torch.zeros(NKC))
        self.Wo, self.Wor = w(NI, H), w(NI, NI)
        self.bo = torch.nn.Parameter(torch.zeros(NI))

    def forward(self, x, K, V, ws, detach_at=()):
        B = x.shape[0]
        U = x @ self.Wx.T + self.bh
        h = x.new_zeros(B, self.Wh.shape[0])
        r = x.new_zeros(B, NI)
        outs = []
        for t in range(x.shape[1]):
            if t in detach_at:
                h, r = h.detach(), r.detach()
            pre = U[:, t] + h @ self.Wh.T
            pre = pre + r @ self.Wr.T
            h = torch.tanh(pre)
            q = torch.relu(h @ self.Wq.T + self.bq)
            r = L166.read_factored(K, V, ws, q, t)
            outs.append(h @ self.Wo.T + r @ self.Wor.T + self.bo)
        return torch.stack(outs, 1)


def loop_batch(w, idx, dtype=torch.float32):
    K, V, ws = memory_np(w, idx)
    return (torch.from_numpy(w["x"][idx]).to(dtype), torch.from_numpy(K).to(dtype), torch.from_numpy(V).to(dtype),
            torch.from_numpy(ws), torch.from_numpy(w["tgt"][idx]))


def detach_steps(rule):
    return set() if rule == "BPTT" else {s for s in ANS}           # TBPTT-1: detach on entering every ANS step


# ---------------------------------------------------------------- arms: forward on a batch, reference gradients
def arm_kind(arm):
    return "A" if arm.startswith("A") else "LOOP"


def a_mode(arm):
    for m in ("A-SHUF", "A-CUT"):
        if arm.startswith(m):
            return m
    return "A"


def new_net(arm):
    return ANet() if arm_kind(arm) == "A" else LoopNet()


def logits(net, arm, w, idx, cache=None, dtype=torch.float32, gate=None):
    if arm_kind(arm) == "A":
        x, r, tg = a_batch(w, idx, a_mode(arm), dtype, cache)
        return net(x, r, gate=gate), tg
    x, K, V, ws, tg = loop_batch(w, idx, dtype)
    return net(x, K, V, ws), tg


def ref_grad(arm, rule):
    """Reference gradient: autograd through the whole forward pass (BPTT) or with TBPTT-1 detach points."""
    da = detach_steps(rule)

    def g(net, w, idx, cache):
        if arm_kind(arm) == "A":
            y, tg = logits(net, arm, w, idx, cache)        # A dynamics carry no parameter: BPTT runs through all of it
        else:
            x, K, V, ws, tg = loop_batch(w, idx)
            y = net(x, K, V, ws, da)
        m = tg >= 0
        loss = F.cross_entropy(y[m], tg[m])
        loss.backward()
        return float(loss.detach())
    return g


# ---------------------------------------------------------------- evaluation
def evaluate(net, arm, w, bs=100, cache=None, gate=None):
    n = w["n"]
    out = {k: np.zeros(n) for k in ("q1_sum", "q2_sum", "q1_n", "q2_n", "q1_ok", "q2_ok")}
    with torch.no_grad():
        for i in range(0, n, bs):
            idx = np.arange(i, min(i + bs, n))
            y, tg = logits(net, arm, w, idx, cache, gate=gate)
            lp = torch.log_softmax(y.double(), -1)
            hop = torch.from_numpy(w["hop"][idx])
            tt = tg.clamp(min=0)
            bits_all = -lp.gather(-1, tt[..., None])[..., 0] / math.log(2)
            ok_all = (lp.argmax(-1) == tt).double()
            for hp in (1, 2):
                m = (hop == hp).double()
                out[f"q{hp}_sum"][idx] = (bits_all * m).sum(1).numpy()
                out[f"q{hp}_n"][idx] = m.sum(1).numpy()
                out[f"q{hp}_ok"][idx] = (ok_all * m).sum(1).numpy()
    tot = out["q1_n"].sum() + out["q2_n"].sum()
    out["all_bits"] = float((out["q1_sum"].sum() + out["q2_sum"].sum()) / tot)
    return out


def summary(ev):
    s = {}
    for hp in (1, 2):
        c = ev[f"q{hp}_n"].sum()
        s[f"q{hp}_bits"] = float(ev[f"q{hp}_sum"].sum() / c)
        s[f"q{hp}_acc"] = float(ev[f"q{hp}_ok"].sum() / c)
    s["all_bits"] = ev["all_bits"]
    return s


def test_record(net, arm, te):
    ev = evaluate(net, arm, te, cache=a_state_cache(te, a_mode(arm)) if arm_kind(arm) == "A" else None)
    return {"test": summary(ev), "line": {k: ev[k].tolist() for k in ev if k != "all_bits"}}


def boot_sd(a_sums, b_sums, counts):
    rng = np.random.default_rng(SET["boot"]["seed"])
    d = []
    for _ in range(SET["boot"]["reps"]):
        i = rng.integers(0, len(counts), len(counts))
        d.append((a_sums[i].sum() - b_sums[i].sum()) / counts[i].sum())
    return float(np.std(d))


def gap(worse, better, hop=2):
    lw, lb = worse["line"], better["line"]
    d = worse["test"][f"q{hop}_bits"] - better["test"][f"q{hop}_bits"]
    sd = boot_sd(np.array(lw[f"q{hop}_sum"]), np.array(lb[f"q{hop}_sum"]), np.array(lb[f"q{hop}_n"]))
    return {"gain_bits": d, "boot_sd": sd, "z": d / sd if sd > 0 else None}


def oracle(w):
    """Ideal gate (1 for TWO-HOP, 0 for ONE-HOP) on the A circuit, argmax of m: accuracy per hop type (T0-c)."""
    r, _ = a_states(w, "A")
    ok, cnt = {1: 0, 2: 0}, {1: 0, 2: 0}
    for i in range(w["n"]):
        for s in ANS:
            hp = int(w["hop"][i, s])
            m = r[i, s] if hp == 2 else r[i, s - 1]
            ok[hp] += int(np.argmax(m) == w["tgt"][i, s])
            cnt[hp] += 1
    return {f"q{h}_acc": ok[h] / cnt[h] for h in (1, 2)} | {f"q{h}_n": cnt[h] for h in (1, 2)}


# ---------------------------------------------------------------- training (every arm)
def train(arm, cfg, tr, ca, grad_fn, log=None, lr_key=None, net=None):
    """Batch order seed 13, AdamW (cfg lr, wd 1e-4), clip 5, best calibration epoch (all scored steps) kept, patience 10.
    grad_fn(net, w, idx, cache) fills p.grad. Line presentations = lines x epochs run."""
    t0 = time.process_time()
    s = SET["train"]
    net = new_net(arm) if net is None else net
    lr = cfg[lr_key or ("lr_a" if arm_kind(arm) == "A" else "lr_loop")]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=s["wd"])
    gen = torch.Generator().manual_seed(s["order_seed"])
    n = tr["n"]
    ctr = a_state_cache(tr, a_mode(arm)) if arm_kind(arm) == "A" else None
    cca = a_state_cache(ca, a_mode(arm)) if arm_kind(arm) == "A" else None
    curve, best, since, epochs = [], (np.inf, 0, None), 0, 0
    for ep in range(1, cfg["max_epochs"] + 1):
        epochs = ep
        order = torch.randperm(n, generator=gen).numpy()
        for i in range(0, n, cfg["batch"]):
            opt.zero_grad()
            grad_fn(net, tr, np.sort(order[i:i + cfg["batch"]]), ctr)
            torch.nn.utils.clip_grad_norm_(net.parameters(), s["clip"])
            opt.step()
        if not all(torch.isfinite(p).all() for p in net.parameters()):
            raise FloatingPointError(f"{arm}: non-finite parameters at epoch {ep}")
        cb = evaluate(net, arm, ca, cache=cca)["all_bits"]
        curve.append(round(cb, 5))
        if log:
            print(f"{arm} ep {ep} calib {cb:.4f} cpu {time.process_time() - t0:.0f}", file=log, flush=True)
        if cb < best[0]:
            best, since = (cb, ep, {k: v.clone() for k, v in net.state_dict().items()}), 0
        else:
            since += 1
            if since >= s["patience"]:
                break
    net.load_state_dict(best[2])
    return net, {"arm": arm, "config": cfg, "best_epoch": best[1], "epochs_run": epochs, "calib_bits": best[0],
                 "calib_curve": curve, "best_epoch_is_last": best[1] == cfg["max_epochs"],
                 "line_presentations": n * epochs, "cpu_s": round(time.process_time() - t0, 1)}
