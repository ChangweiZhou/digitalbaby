"""V155 references (torch environment): free hidden layers trained by back-propagation, evaluator instruments only.
Nothing here feeds MiniFly. They bound, optimistically, what a learned representation of the same receptor trace can
reach under each wiring (design section 12.9): graded ReLU units instead of a binary top-k code, unconstrained weights.

Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_t/bin/python refs.py --text thursday
  REF-N  q (302 PN inputs) -> 5,177 ReLU units, each connected only to its birth synapses (mask) -> softmax
  REF-D  the same with every PN connected to every unit (dense)
Training as the headroom check's T3: AdamW lr 1e-3, weight decay 1e-4, batch 256, seed 0, one thread, <= 60 epochs,
stopping after 6 epochs without improvement in calibration bits; dropout on the hidden layer in {0, 0.5}, chosen on
calibration bits. Scored by their own held-out bits/symbol (V149 class convention).
"""
import argparse
import hashlib
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
CACHE = HERE / "cache"
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
_spec = importlib.util.spec_from_file_location(
    "v149_readouts", HERE.parent / "minifly_v149_input_ceiling_dev_20261025" / "readouts.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
SETTINGS = json.loads((HERE / "SETTINGS.json").read_text())["refs"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Free(torch.nn.Module):
    def __init__(self, mask, K, dropout):
        super().__init__()
        self.first = torch.nn.Linear(mask.shape[0], mask.shape[1])
        self.mask = None if mask.all() else torch.from_numpy(mask.astype(np.float32)).T     # (units, inputs)
        self.drop = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(mask.shape[1], K)

    def forward(self, Q):
        W = self.first.weight if self.mask is None else self.first.weight * self.mask
        return self.out(self.drop(torch.relu(Q @ W.T + self.first.bias)))


def probs(model, Q, chunk=4096):
    model.eval()
    with torch.no_grad():
        return np.vstack([torch.softmax(model(Q[i:i + chunk]).double(), 1).numpy() for i in range(0, len(Q), chunk)])


def line_sums(bits, n):
    e = np.cumsum(np.concatenate([[0], n]))
    return [float(bits[e[i]:e[i + 1]].sum()) for i in range(len(n))]


def fit(mask, Q, y, cls, dropout, n_held):
    t0 = time.process_time()
    torch.manual_seed(SETTINGS["seed"])
    g = torch.Generator().manual_seed(SETTINGS["seed"])
    ctr = torch.from_numpy(cls.encode(y["train"]))
    model = Free(mask, cls.K, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=SETTINGS["lr"], weight_decay=SETTINGS["weight_decay"])
    curve, best, since = [], (np.inf, 0, None), 0
    for ep in range(1, SETTINGS["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(ctr), generator=g)
        for i in range(0, len(perm), SETTINGS["batch"]):
            idx = perm[i:i + SETTINGS["batch"]]
            loss = torch.nn.functional.cross_entropy(model(Q["train"][idx]), ctr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        cb = float(cls.bits(probs(model, Q["calib"]), y["calib"]).mean())
        curve.append(round(cb, 5))
        if cb < best[0]:
            best, since = (cb, ep, {k: v.clone() for k, v in model.state_dict().items()}), 0
        else:
            since += 1
            if since >= SETTINGS["patience"]:
                break
    model.load_state_dict(best[2])
    P = probs(model, Q["held"])
    bits = cls.bits(P, y["held"])
    Ptr = probs(model, Q["train"])
    return {"dropout": dropout, "best_epoch": best[1], "epochs_run": len(curve), "calib_bpb": best[0],
            "calib_curve": curve, "cut_off_by_budget": best[1] == SETTINGS["max_epochs"],
            "train_class_bits": float(-np.log2(Ptr[np.arange(len(ctr)), ctr.numpy()]).mean()),
            "held_bpb": float(bits.mean()), "held_acc": cls.accuracy(P, y["held"]), "lines": line_sums(bits, n_held),
            "cpu_s": round(time.process_time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    a = ap.parse_args()
    t0 = time.process_time()
    from codes import load_birth  # noqa: E402  (numpy/scipy only)
    birth = load_birth(CACHE / "birth.npz")
    z = np.load(CACHE / f"{a.text}_A.npz")
    y = {k: z[f"y_{k}"] for k in ("train", "calib", "held")}
    cls = R.Classes(y["train"])
    Q = {k: torch.from_numpy(z[f"P_{k}"][:, birth["pn"]].astype(np.float32)) for k in y}
    native = birth["B"].toarray() > 0
    res = {"text": a.text, "stage": STAGE, "settings": SETTINGS, "torch": torch.__version__,
           "threads": torch.get_num_threads(), "cache_sha256": sha(CACHE / f"{a.text}_A.npz"), "learner_records": 0,
           "held_line_len": z["n_held"].tolist()}
    for name, mask in (("REF-N", native), ("REF-D", np.ones_like(native))):
        runs = []
        for dr in SETTINGS["dropout"]:
            r = fit(mask, Q, y, cls, dr, z["n_held"])
            runs.append(r)
            print(name, "dropout", dr, "epoch", r["best_epoch"], "calib", round(r["calib_bpb"], 4), "held",
                  round(r["held_bpb"], 4), "cpu", r["cpu_s"], flush=True)
        chosen = min(runs, key=lambda r: r["calib_bpb"])
        res[name] = {"chosen_dropout": chosen["dropout"], "held_bpb": chosen["held_bpb"], "held_acc": chosen["held_acc"],
                     "calib_bpb": chosen["calib_bpb"], "cut_off_by_budget": chosen["cut_off_by_budget"],
                     "lines": chosen["lines"], "runs": runs}
    res["cpu_s_total"] = round(time.process_time() - t0, 1)
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / f"refs_{a.text}.json").write_text(json.dumps(res, indent=1))
    import runtime  # noqa: E402
    runtime.write_closure()


if __name__ == "__main__":
    main()
