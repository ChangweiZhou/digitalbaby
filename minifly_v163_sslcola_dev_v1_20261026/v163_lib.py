"""V163 library (design sections 1-3), torch environment. Data, the probe (the headroom check's T3 learner, generalised
to the KC code and to a frozen layer's output), lane L's layer H and lane C's recurrent layer. No MiniFly import."""
import importlib.util
import json
import math
import os
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SET = json.loads((HERE / "SETTINGS.json").read_text())
_spec = importlib.util.spec_from_file_location(
    "v149_readouts", ROOT / "minifly_v149_input_ceiling_dev_20261025" / "readouts.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
D = SET["data"]
NIN, PAD = D["n_kc"] + 1, D["pad"]            # 5,177 KCs + empty-state column; pad index excluded from every sum
SPLITS = ("train", "calib", "held")


# ---------------------------------------------------------------- data and scoring
def load_codes():
    z = np.load(HERE / "cache" / "codes.npz")
    return {t: {k: {"idx": torch.from_numpy(z[f"{t}_idx_{k}"].astype(np.int64)), "y": z[f"{t}_y_{k}"],
                    "n": z[f"{t}_n_{k}"]} for k in SPLITS} for t in ("A", "B")}


def line_sums(bits, n):
    e = np.cumsum(np.concatenate([[0], n]))
    return np.array([bits[e[i]:e[i + 1]].sum() for i in range(len(n))])


def boot_sd(a, b, n, reps=None, seed=None):
    """SD of (sum a - sum b) / sum n over resampled held-out lines (a, b: per-line bit sums)."""
    reps = SET["boot"]["reps"] if reps is None else reps
    rng = np.random.default_rng(SET["boot"]["seed"] if seed is None else seed)
    a, b, n = np.asarray(a), np.asarray(b), np.asarray(n, float)
    d = []
    for _ in range(reps):
        i = rng.integers(0, len(n), len(n))
        d.append((a[i].sum() - b[i].sum()) / n[i].sum())
    return float(np.std(d))


# ---------------------------------------------------------------- probe (T3 learner)
class Probe(torch.nn.Module):
    """input -> [hidden ReLU -> dropout] -> logits. kind 'dense' (float input) or 'kc' (padded active indices, which is
    the same as a Linear layer on the binary code; initialised exactly as that Linear layer would be)."""

    def __init__(self, kind, d_in, hidden, K, dropout):
        super().__init__()
        self.kind = kind
        width = hidden if hidden else K
        if kind == "dense":
            self.first = torch.nn.Linear(d_in, width)
        else:
            lin = torch.nn.Linear(NIN, width)
            self.emb = torch.nn.EmbeddingBag(NIN + 1, width, mode="sum", padding_idx=PAD)
            with torch.no_grad():
                self.emb.weight.zero_()
                self.emb.weight[:NIN] = lin.weight.T
            self.bias = torch.nn.Parameter(lin.bias.detach().clone())
        self.hidden = hidden
        if hidden:
            self.drop = torch.nn.Dropout(dropout)
            self.out = torch.nn.Linear(hidden, K)

    def forward(self, x):
        h = self.first(x) if self.kind == "dense" else self.emb(x) + self.bias
        return self.out(self.drop(torch.relu(h))) if self.hidden else h


def probs(net, x, chunk=8192):
    net.eval()
    with torch.no_grad():
        return np.vstack([torch.softmax(net(x[i:i + chunk]).double(), 1).numpy() for i in range(0, len(x), chunk)])


def train_loop(net, Xtr, ctr, eval_calib, p=None):
    """The headroom check's loop: AdamW, shuffled batches, best calibration epoch kept, patience stop."""
    p = SET["probe"] if p is None else p
    g = torch.Generator().manual_seed(p["seed"])
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["wd"])
    curve, best, since = [], (np.inf, 0, None), 0
    for ep in range(1, p["max_epochs"] + 1):
        net.train()
        perm = torch.randperm(len(ctr), generator=g)
        for i in range(0, len(perm), p["batch"]):
            idx = perm[i:i + p["batch"]]
            loss = F.cross_entropy(net(Xtr[idx]), ctr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        cb = eval_calib(net)
        curve.append(round(cb, 5))
        if cb < best[0]:
            best, since = (cb, ep, {k: v.clone() for k, v in net.state_dict().items()}), 0
        else:
            since += 1
            if since >= p["patience"]:
                break
    net.load_state_dict(best[2])
    return {"best_epoch": best[1], "epochs_run": len(curve), "calib_bpb": best[0], "calib_curve": curve,
            "cut_off_by_budget": best[1] == p["max_epochs"]}


def fit_probe(kind, X, ys, n_held, hidden=None, seed=None, d_in=None):
    """X, ys: dicts over SPLITS. hidden None means SET['probe']['hidden']; hidden 0 is the linear probe.
    Returns scores with per-position held bits kept for later differences."""
    t0 = time.process_time()
    p = dict(SET["probe"])
    if seed is not None:
        p["seed"] = seed
    hidden = p["hidden"] if hidden is None else hidden
    cls = R.Classes(ys["train"])
    torch.manual_seed(p["seed"])
    net = Probe(kind, d_in, hidden, cls.K, p["dropout"])
    ctr = torch.from_numpy(cls.encode(ys["train"]))
    info = train_loop(net, X["train"], ctr,
                      lambda m: float(cls.bits(probs(m, X["calib"]), ys["calib"]).mean()), p)
    P = probs(net, X["held"])
    bits = cls.bits(P, ys["held"])
    info.update({"hidden": hidden, "seed": p["seed"], "held_bpb": float(bits.mean()),
                 "held_acc": cls.accuracy(P, ys["held"]), "lines": line_sums(bits, n_held).tolist(),
                 "cpu_s": round(time.process_time() - t0, 1)})
    return info


# ---------------------------------------------------------------- lane L: layer H on the KC code
def init_H(seed=None):
    """W ~ N(0, 1/184) per synapse (unit variance pre-activation for about 181 active inputs), b = 0; pad row zero."""
    s = SET["lane_L"]
    g = torch.Generator().manual_seed(s["init_seed"] if seed is None else seed)
    W = torch.randn(NIN + 1, s["H"], generator=g) / math.sqrt(s["init_var_inv"])
    W[PAD] = 0.0
    return W, torch.zeros(s["H"])


def h_of(idx, W, b, chunk=8192):
    with torch.no_grad():
        return torch.cat([torch.relu(F.embedding_bag(idx[i:i + chunk], W, mode="sum", padding_idx=PAD) + b)
                          for i in range(0, len(idx), chunk)])


class BPH(torch.nn.Module):
    """KC code -> H (ReLU, from init_H) -> probe (2048 ReLU, dropout) -> logits, trained end-to-end."""

    def __init__(self, K, W, b):
        super().__init__()
        p = SET["probe"]
        self.top = Probe("dense", W.shape[1], p["hidden"], K, p["dropout"])
        self.W = torch.nn.Parameter(W.clone())
        self.b = torch.nn.Parameter(b.clone())

    def forward(self, idx):
        return self.top(torch.relu(F.embedding_bag(idx, self.W, mode="sum", padding_idx=PAD) + self.b))


def train_bph(A):
    t0 = time.process_time()
    cls = R.Classes(A["train"]["y"])
    torch.manual_seed(SET["probe"]["seed"])
    W, b = init_H()
    net = BPH(cls.K, W, b)
    ctr = torch.from_numpy(cls.encode(A["train"]["y"]))
    info = train_loop(net, A["train"]["idx"], ctr,
                      lambda m: float(cls.bits(probs(m, A["calib"]["idx"]), A["calib"]["y"]).mean()))
    with torch.no_grad():
        net.W[PAD] = 0.0
    info["cpu_s"] = round(time.process_time() - t0, 1)
    return net.W.detach().clone(), net.b.detach().clone(), info


# ---------------------------------------------------------------- lane C: recurrent layer on the KC code
def line_batches(split, lines):
    """Padded (B, T, max_active) index tensor, (B, T) mask and (B, T) positions for the given line ids."""
    e = np.cumsum(np.concatenate([[0], split["n"]]))
    T = int(max(split["n"][i] for i in lines))
    idx = torch.full((len(lines), T, split["idx"].shape[1]), PAD, dtype=torch.long)
    mask = torch.zeros(len(lines), T, dtype=torch.bool)
    pos = torch.full((len(lines), T), -1, dtype=torch.long)
    for j, i in enumerate(lines):
        L = int(split["n"][i])
        idx[j, :L] = split["idx"][e[i]:e[i] + L]
        mask[j, :L] = True
        pos[j, :L] = torch.arange(e[i], e[i] + L)
    return idx, mask, pos


class RNN(torch.nn.Module):
    """r_t = tanh(W_rr r_{t-1} + W_rk x_t + b), reset per line; linear softmax readout during training."""

    def __init__(self, K, g):
        super().__init__()
        s = SET["lane_C"]
        gen = torch.Generator().manual_seed(s["init_seed"])
        Wrk = torch.randn(NIN + 1, s["H"], generator=gen) / math.sqrt(SET["lane_L"]["init_var_inv"])
        Wrk[PAD] = 0.0
        Wbar = torch.randn(s["H"], s["H"], generator=gen)
        Wbar = Wbar / torch.linalg.matrix_norm(Wbar, ord=2)
        self.Wrk = torch.nn.Parameter(Wrk)
        self.Wrr = torch.nn.Parameter(g * Wbar)
        self.b = torch.nn.Parameter(torch.zeros(s["H"]))
        torch.manual_seed(s["init_seed"])
        self.out = torch.nn.Linear(s["H"], K)

    def run(self, idx, truncate=False):
        """States (B, T, H) for a padded batch; truncate=True stops the gradient through r_{t-1} (TBPTT-1)."""
        B, T, A = idx.shape
        u = F.embedding_bag(idx.reshape(B * T, A), self.Wrk, mode="sum", padding_idx=PAD).view(B, T, -1)
        r = torch.zeros(B, self.Wrr.shape[0])
        out = []
        for t in range(T):
            prev = r.detach() if truncate else r
            r = torch.tanh(u[:, t] + prev @ self.Wrr.T + self.b)
            out.append(r)
        return torch.stack(out, 1)


def lyapunov(Wbar_g, Wrk, split, lines, seed=0):
    """Largest Lyapunov exponent of the input-driven map along taught lines (renormalised tangent vector)."""
    idx, mask, _ = line_batches(split, lines)
    B, T, A = idx.shape
    with torch.no_grad():
        u = F.embedding_bag(idx.reshape(B * T, A), Wrk, mode="sum", padding_idx=PAD).view(B, T, -1)
        r = torch.zeros(B, Wbar_g.shape[0])
        v = torch.randn(B, Wbar_g.shape[0], generator=torch.Generator().manual_seed(seed))
        v = v / v.norm(dim=1, keepdim=True)
        tot, steps = 0.0, 0
        for t in range(T):
            r = torch.where(mask[:, t:t + 1], torch.tanh(u[:, t] + r @ Wbar_g.T), r)
            if t == 0:
                continue
            vn = (1 - r ** 2) * (v @ Wbar_g.T)
            nrm = vn.norm(dim=1)
            m = mask[:, t]
            tot += float(torch.log(nrm[m]).sum())
            steps += int(m.sum())
            v = torch.where(m[:, None], vn / nrm[:, None], v)
    return tot / steps


def choose_gain(A):
    """Design section 3: the COLA paper's two-point estimate (Eq. 12), probes at g = 1 and 1.1."""
    s = SET["lane_C"]
    net = RNN(2, 1.0)
    Wbar = net.Wrr.detach()
    lines = list(range(s["lyap_lines"]))
    g1, g2 = s["g_probes"]
    l1, l2 = (lyapunov(g * Wbar, net.Wrk.detach(), A["train"], lines) for g in (g1, g2))
    slope = (l2 - l1) / math.log(g2 / g1)
    g = math.exp(-l1 / slope) if slope > 0 and math.isfinite(slope) else math.exp(-l1)
    return g, {"lambda_g1": l1, "lambda_g2": l2, "slope": slope, "g": g, "unit_fallback": not (slope > 0),
               "lambda_at_g": lyapunov(g * Wbar, net.Wrk.detach(), A["train"], lines)}


def rnn_bits(net, split, cls, batch=64):
    """Training-readout bits per position (calibration score during training)."""
    bits = np.zeros(len(split["y"]))
    with torch.no_grad():
        for i in range(0, len(split["n"]), batch):
            idx, mask, pos = line_batches(split, list(range(i, min(i + batch, len(split["n"])))))
            P = torch.softmax(net.out(net.run(idx)).double(), -1)[mask].numpy()
            bits[pos[mask].numpy()] = cls.bits(P, split["y"][pos[mask].numpy()])
    return bits


def rnn_states(net, split, batch=64):
    S = torch.zeros(len(split["y"]), net.Wrr.shape[0])
    with torch.no_grad():
        for i in range(0, len(split["n"]), batch):
            idx, mask, pos = line_batches(split, list(range(i, min(i + batch, len(split["n"])))))
            S[pos[mask]] = net.run(idx)[mask]
    return S


def train_rnn(A, g, rule, grad_fn=None):
    """rule 'tbptt1' or 'bptt' (autograd), or a custom grad_fn(net, idx, mask, ctr) that sets .grad (COLA).
    Same schedule for every rule: batches of lines, one AdamW step per batch, clip, best calibration epoch kept."""
    t0 = time.process_time()
    s = SET["lane_C"]
    cls = R.Classes(A["train"]["y"])
    net = RNN(cls.K, g)
    opt = torch.optim.AdamW(net.parameters(), lr=s["lr"], weight_decay=s["wd"])
    gen = torch.Generator().manual_seed(s["order_seed"])
    enc = torch.from_numpy(cls.encode(A["train"]["y"]))
    curve, best, since, epochs = [], (np.inf, 0, None), 0, 0
    for ep in range(1, s["max_epochs"] + 1):
        epochs = ep
        net.train()
        order = torch.randperm(len(A["train"]["n"]), generator=gen).tolist()
        for i in range(0, len(order), s["batch_lines"]):
            idx, mask, pos = line_batches(A["train"], order[i:i + s["batch_lines"]])
            ctr = enc[pos.clamp(min=0)]
            opt.zero_grad()
            if grad_fn is None:
                logits = net.out(net.run(idx, truncate=(rule == "tbptt1")))
                F.cross_entropy(logits[mask], ctr[mask]).backward()
            else:
                grad_fn(net, idx, mask, ctr)
            torch.nn.utils.clip_grad_norm_(net.parameters(), s["clip"])
            opt.step()
        cb = float(rnn_bits(net, A["calib"], cls).mean())
        curve.append(round(cb, 5))
        if cb < best[0]:
            best, since = (cb, ep, {k: v.clone() for k, v in net.state_dict().items()}), 0
        else:
            since += 1
            if since >= s["patience"]:
                break
    net.load_state_dict(best[2])
    return net, {"rule": rule, "best_epoch": best[1], "epochs_run": epochs, "calib_bpb": best[0], "calib_curve": curve,
                 "cut_off_by_budget": best[1] == s["max_epochs"], "cpu_s": round(time.process_time() - t0, 1)}
