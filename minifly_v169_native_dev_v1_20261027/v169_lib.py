"""V169 learner-side library (design revision 2, section 2), torch environment: continued learning of V167's LOCAL network
on a second text, with or without replay of the first, and own-top scoring. V167's learner and V163's library unchanged."""
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V167 = ROOT / "minifly_v167_local_dev_v1_20261026"
sys.path.insert(1, str(V167))
import v167_common as C  # noqa: E402
import v167_local as L  # noqa: E402

V, D = C.V, L.D
SET = json.loads((HERE / "SETTINGS.json").read_text())


def write_closure():
    spec = importlib.util.spec_from_file_location("v169_runtime", HERE / "runtime.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.write_closure()


def feedbacks(cls):
    """The fixed feedback matrices V167's train() builds internally (same seeds)."""
    return D.feedback(cls.K, 512), L.feedback(cls.K, 2048, L.S["F_T_seed"])


def own_bits(net, cls, split):
    """Per-line bit sums and mean bits of the network's own predictions (eval mode) under classes cls."""
    bits = cls.bits(V.probs(net, split["idx"]), split["y"])
    return float(bits.mean()), V.line_sums(bits, split["n"])


def batches(B_train, A_train, replay, g_order, g_replay, batch=256):
    """Index tensors into B's positions and, with replay, an equal number of A positions (fixed generator)."""
    perm = torch.randperm(len(B_train["y"]), generator=g_order)
    for i in range(0, len(perm), batch):
        jb = perm[i:i + batch]
        ja = torch.randint(0, len(A_train["y"]), (len(jb),), generator=g_replay) if replay else None
        yield jb, ja


def cont(net0, cls, A_train, B_train, B_calib, replay, max_epochs, patience):
    """Continue learning with V167's local rule from net0 (deep-copied), fresh AdamW, best B-calibration epoch kept."""
    t0 = time.process_time()
    net = copy.deepcopy(net0)
    FH, FT = feedbacks(cls)
    p = V.SET["probe"]
    torch.manual_seed(p["seed"])                      # dropout masks: seeded at the start, as V167's train() does
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["wd"])
    g_order = torch.Generator().manual_seed(p["seed"])
    g_replay = torch.Generator().manual_seed(SET["persist"]["replay_seed"])
    cB = torch.from_numpy(cls.encode(B_train["y"]))
    cA = torch.from_numpy(cls.encode(A_train["y"]))
    curve, best, since = [], (np.inf, 0, copy.deepcopy(net.state_dict())), 0
    for ep in range(1, max_epochs + 1):
        net.train()
        for jb, ja in batches(B_train, A_train, replay, g_order, g_replay):
            idx, c = B_train["idx"][jb], cB[jb]
            if ja is not None:
                idx, c = torch.cat([idx, A_train["idx"][ja]]), torch.cat([c, cA[ja]])
            opt.zero_grad()
            L.local_step(net, FH, FT, idx, c)
            opt.step()
        cb, _ = own_bits(net, cls, B_calib)
        curve.append(round(cb, 5))
        if cb < best[0]:
            best, since = (cb, ep, copy.deepcopy(net.state_dict())), 0
        else:
            since += 1
            if since >= patience:
                break
    net.load_state_dict(best[2])
    per_epoch = len(B_train["n"]) * (2 if replay else 1)
    return net, {"replay": replay, "best_epoch": best[1], "epochs_run": len(curve), "calib_B": best[0], "calib_curve": curve,
                 "cut_off_by_budget": best[1] == max_epochs, "records": len(curve) * per_epoch,
                 "cpu_s": round(time.process_time() - t0, 1)}
