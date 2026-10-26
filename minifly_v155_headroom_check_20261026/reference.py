"""V155 headroom check, step 2 (torch environment): readings T3 and T4, evaluator-only back-propagation references.

Run through attempt.py:  python attempt.py -- /tmp/venv_t/bin/python reference.py --tag main
Reads results/<tag>/states.npz; writes results/<tag>/REFERENCE.json. Nothing here feeds MiniFly.
  T3  88-channel receptor trace -> hidden layer (ReLU, width H) -> softmax over the classes
  T4  last 4 symbols, one-hot per position (summed embeddings = the same first layer on a sparse input) -> same
"""
import argparse
import importlib.util
import json
import os
import resource
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "v149_readouts", HERE.parent / "minifly_v149_input_ceiling_dev_20261025" / "readouts.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

GRID = [(512, 0.0), (512, 0.5), (2048, 0.0), (2048, 0.5)]   # (H, dropout)
LR, WD, BATCH, MAX_EPOCHS, PATIENCE, SEED = 1e-3, 1e-4, 256, 60, 6, 0
NCTX, NSYM = 4, 258                                          # 257 symbols + "before line start"


class Net(torch.nn.Module):
    def __init__(self, kind, d_in, H, K, dropout):
        super().__init__()
        self.kind = kind
        if kind == "trace":
            self.first = torch.nn.Linear(d_in, H)
        else:
            self.first = torch.nn.EmbeddingBag(NCTX * NSYM, H, mode="sum")
            self.bias = torch.nn.Parameter(torch.zeros(H))
        self.drop = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(H, K)

    def forward(self, inp):
        h = self.first(inp) if self.kind == "trace" else self.first(inp) + self.bias
        return self.out(self.drop(torch.relu(h)))


def as_input(kind, a):
    if kind == "trace":
        return torch.from_numpy(np.asarray(a, np.float32))
    return torch.from_numpy(a + np.arange(NCTX)[None, :] * NSYM)


def probs(net, inp, chunk=8192):
    net.eval()
    with torch.no_grad():
        return np.vstack([torch.softmax(net(inp[i:i + chunk]).double(), 1).numpy() for i in range(0, len(inp), chunk)])


def line_sums(bits, line_len):
    edges = np.cumsum(np.concatenate([[0], line_len]))
    return [float(bits[edges[i]:edges[i + 1]].sum()) for i in range(len(line_len))]


def fit(kind, data, cls, H, dropout, line_len):
    t0 = time.process_time()
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    Xtr, Xca, Xhe = (as_input(kind, data[k]) for k in ("train", "calib", "held"))
    ctr = torch.from_numpy(cls.encode(data["y_train"]))
    net = Net(kind, Xtr.shape[1] if kind == "trace" else None, H, cls.K, dropout)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    curve, best, since = [], (np.inf, 0, None), 0
    for ep in range(1, MAX_EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(ctr), generator=g)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = torch.nn.functional.cross_entropy(net(Xtr[idx]), ctr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        cb = float(cls.bits(probs(net, Xca), data["y_calib"]).mean())
        curve.append(round(cb, 5))
        if cb < best[0]:
            best, since = (cb, ep, {k: v.clone() for k, v in net.state_dict().items()}), 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    net.load_state_dict(best[2])
    P = probs(net, Xhe)
    bits = cls.bits(P, data["y_held"])
    Ptr = probs(net, Xtr)
    return {"H": H, "dropout": dropout, "best_epoch": best[1], "epochs_run": len(curve), "calib_bpb": best[0],
            "calib_curve": curve, "cut_off_by_budget": best[1] == MAX_EPOCHS,
            "train_class_bits": float(-np.log2(Ptr[np.arange(len(ctr)), ctr.numpy()]).mean()),
            "held_bpb": float(bits.mean()), "held_acc": cls.accuracy(P, data["y_held"]),
            "lines": line_sums(bits, line_len),
            "cpu_s": round(time.process_time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out = HERE / "results" / a.tag
    z = np.load(out / "states.npz")
    line_len = z["held_line_len"]
    cls = R.Classes(z["y_train"])
    res = {"tag": a.tag, "grid": GRID, "lr": LR, "weight_decay": WD, "batch": BATCH, "max_epochs": MAX_EPOCHS,
           "patience": PATIENCE, "seed": SEED, "torch": torch.__version__, "threads": torch.get_num_threads()}
    t0 = time.process_time()
    for name, kind, pre in (("T3", "trace", "rho_"), ("T4", "context", "ctx_")):
        data = {k: z[pre + k] for k in ("train", "calib", "held")}
        data.update({f"y_{k}": z[f"y_{k}"] for k in ("train", "calib", "held")})
        rows = []
        for H, dr in GRID:
            r = fit(kind, data, cls, H, dr, line_len)
            rows.append(r)
            print(name, H, dr, "epoch", r["best_epoch"], "calib", round(r["calib_bpb"], 4), "held",
                  round(r["held_bpb"], 4), "cpu", r["cpu_s"], flush=True)
        chosen = min(rows, key=lambda r: r["calib_bpb"])
        res[name] = {"chosen": {"H": chosen["H"], "dropout": chosen["dropout"]}, "held_bpb": chosen["held_bpb"],
                     "held_acc": chosen["held_acc"], "calib_bpb": chosen["calib_bpb"],
                     "cut_off_by_budget": chosen["cut_off_by_budget"], "lines": chosen["lines"], "grid": rows}
    res["cpu_s_total"] = round(time.process_time() - t0, 1)
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    (out / "REFERENCE.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
