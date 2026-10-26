"""V165 shared helpers (no learner code): V163's library and caches, line subsets, the frozen-H reference trainer."""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
V163 = HERE.parent / "minifly_v163_sslcola_dev_v1_20261026"
sys.path.insert(1, str(V163))
import v163_lib as V  # noqa: E402
from v163_testbed_r2 import mean_fit  # noqa: E402,F401

SET = json.loads((HERE / "SETTINGS.json").read_text())


def first_lines(split, k):
    e = int(split["n"][:k].sum())
    return {"idx": split["idx"][:e], "y": split["y"][:e], "n": split["n"][:k]}


def probe_settings(max_epochs, patience):
    p = dict(V.SET["probe"])
    p.update({"max_epochs": max_epochs, "patience": patience})
    return p


def calib_fn(cls, split):
    return lambda m: float(cls.bits(V.probs(m, split["idx"]), split["y"]).mean())


def frozen_h_top(A_train, A_calib, epochs):
    """Evaluator reference: V163's BP-H model with H frozen at RAND-H's initialisation (only the top learns)."""
    cls = V.R.Classes(A_train["y"])
    torch.manual_seed(V.SET["probe"]["seed"])
    W, b = V.init_H()
    net = V.BPH(cls.K, W, b)
    net.W.requires_grad_(False)
    net.b.requires_grad_(False)
    ctr = torch.from_numpy(cls.encode(A_train["y"]))
    info = V.train_loop(net, A_train["idx"], ctr, calib_fn(cls, A_calib), probe_settings(epochs, epochs))
    return net, info
