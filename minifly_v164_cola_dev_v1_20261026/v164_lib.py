"""V164 library (design sections 1-2), torch environment: synthetic tasks, a generic training loop for V163's recurrent
layer (scored positions only), evaluation by the trained readout. V163's RNN, line batching and Lyapunov estimate are
imported unchanged."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
V163 = HERE.parent / "minifly_v163_sslcola_dev_v1_20261026"
sys.path.insert(1, str(V163))
import v163_lib as V  # noqa: E402

SET = json.loads((HERE / "SETTINGS.json").read_text())


# ---------------------------------------------------------------- tasks
class Synth:
    """Plain softmax over the 8 symbols; targets < 0 are unscored."""
    K = 8

    def encode(self, y):
        return np.asarray(y)

    def bits(self, P, y):
        return -np.log2(P[np.arange(len(y)), y])


def synth_split(task, part):
    s = SET["synthetic"]
    rng = np.random.default_rng(s["seeds"][part] * 100 + sorted(s["tasks"]).index(task))
    lags, L, n = s["tasks"][task]["lags"], s["length"], s["lines"][part]
    X = rng.integers(0, s["symbols"], (n, L))
    Y = np.full((n, L), -1)
    m = max(lags)
    Y[:, m:] = X[:, m - lags[0]:L - lags[0]] if len(lags) == 1 else \
        (X[:, m - lags[0]:L - lags[0]] + X[:, m - lags[1]:L - lags[1]]) % s["symbols"]
    return {"idx": torch.from_numpy(X.reshape(-1, 1).astype(np.int64)), "y": Y.reshape(-1), "n": np.full(n, L)}


def text_splits():
    A = V.load_codes()["A"]
    cls = V.R.Classes(A["train"]["y"])
    return A, cls


# ---------------------------------------------------------------- network
def make_rnn(K, g, in_var_inv):
    """V163's RNN; input weights rescaled to variance 1 / in_var_inv (V163 used 1/184 for about 181 active KCs)."""
    net = V.RNN(K, g)
    with torch.no_grad():
        net.Wrk.mul_(math.sqrt(V.SET["lane_L"]["init_var_inv"] / in_var_inv))
    return net


def gain(split, in_var_inv):
    s = SET["rnn"]
    net = make_rnn(2, 1.0, in_var_inv)
    Wbar, Wrk = net.Wrr.detach(), net.Wrk.detach()
    lines = list(range(min(s["lyap_lines"], len(split["n"]))))
    g1, g2 = s["g_probes"]
    l1, l2 = (V.lyapunov(g * Wbar, Wrk, split, lines) for g in (g1, g2))
    slope = (l2 - l1) / math.log(g2 / g1)
    g = math.exp(-l1 / slope) if slope > 0 else math.exp(-l1)
    return g, {"lambda_g1": l1, "lambda_g2": l2, "slope": slope, "g": g, "lambda_at_g": V.lyapunov(g * Wbar, Wrk, split, lines)}


def targets(split, scorer, pos):
    y = split["y"][pos.clamp(min=0).numpy()]
    valid = (pos >= 0).numpy() & (y >= 0) if isinstance(scorer, Synth) else (pos >= 0).numpy()
    yy = np.where(valid, y, 0) if isinstance(scorer, Synth) else y
    enc = np.asarray(scorer.encode(yy.reshape(-1))).reshape(yy.shape)
    return torch.from_numpy(np.asarray(enc)), torch.from_numpy(valid)


def evaluate(net, split, scorer, batch=64):
    """Per-line bit sums and scored counts from the trained readout."""
    sums, counts = np.zeros(len(split["n"])), np.zeros(len(split["n"]))
    with torch.no_grad():
        for i in range(0, len(split["n"]), batch):
            lines = list(range(i, min(i + batch, len(split["n"]))))
            idx, mask, pos = V.line_batches(split, lines)
            ctr, valid = targets(split, scorer, pos)
            P = torch.softmax(net.out(net.run(idx)).double(), -1)
            for j, l in enumerate(lines):
                v = valid[j]
                b = scorer.bits(P[j][v].numpy(), ctr[j][v].numpy())
                sums[l], counts[l] = b.sum(), v.sum()
    return sums, counts


def train(train_s, calib_s, scorer, g, rule, in_var_inv, max_epochs, patience, grad_fn=None):
    """V163's schedule (batches of 16 lines, one AdamW step per batch, clip, best calibration epoch kept); loss on
    scored positions only. rule 'tbptt1' | 'bptt' (autograd) or grad_fn(net, idx, valid, ctr) setting .grad (COLA)."""
    t0 = time.process_time()
    s = SET["rnn"]
    net = make_rnn(scorer.K, g, in_var_inv)
    if rule == "rand":                      # RAND-R: frozen recurrence, only the readout learns
        for q in (net.Wrk, net.Wrr, net.b):
            q.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=s["lr"], weight_decay=s["wd"])
    gen = torch.Generator().manual_seed(s["order_seed"])
    curve, best, since, epochs = [], (np.inf, 0, None), 0, 0
    for ep in range(1, max_epochs + 1):
        epochs = ep
        net.train()
        order = torch.randperm(len(train_s["n"]), generator=gen).tolist()
        for i in range(0, len(order), s["batch_lines"]):
            idx, mask, pos = V.line_batches(train_s, order[i:i + s["batch_lines"]])
            ctr, valid = targets(train_s, scorer, pos)
            opt.zero_grad()
            if grad_fn is None:
                logits = net.out(net.run(idx, truncate=(rule in ("tbptt1", "rand"))))
                F.cross_entropy(logits[valid], ctr[valid]).backward()
            else:
                grad_fn(net, idx, valid, ctr)
            torch.nn.utils.clip_grad_norm_(net.parameters(), s["clip"])
            opt.step()
        cs, cc = evaluate(net, calib_s, scorer)
        cb = float(cs.sum() / cc.sum())
        curve.append(round(cb, 5))
        if cb < best[0]:
            best, since = (cb, ep, {k: v.clone() for k, v in net.state_dict().items()}), 0
        else:
            since += 1
            if since >= patience:
                break
    net.load_state_dict(best[2])
    return net, {"rule": rule, "best_epoch": best[1], "epochs_run": epochs, "calib_bpb": best[0], "calib_curve": curve,
                 "cut_off_by_budget": best[1] == max_epochs, "cpu_s": round(time.process_time() - t0, 1)}


def boot_sd(a_sums, b_sums, counts):
    rng = np.random.default_rng(SET["boot"]["seed"])
    d = []
    for _ in range(SET["boot"]["reps"]):
        i = rng.integers(0, len(counts), len(counts))
        d.append((a_sums[i].sum() - b_sums[i].sum()) / counts[i].sum())
    return float(np.std(d))


def settings():
    """(task name, train, calib, test, scorer, input variance inverse, max_epochs, patience) for every task."""
    s = SET["synthetic"]
    out = [(t, synth_split(t, "train"), synth_split(t, "calib"), synth_split(t, "test"), Synth(), 1,
            s["max_epochs"], s["patience"]) for t in sorted(s["tasks"])]
    A, cls = text_splits()
    out.append(("TEXT", A["train"], A["calib"], A["held"], cls, V.SET["lane_L"]["init_var_inv"],
                SET["text"]["max_epochs"], SET["text"]["patience"]))
    return out
