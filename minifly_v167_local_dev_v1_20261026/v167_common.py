"""V167 shared helpers (torch side, no learner code): V163's library, the rehearsal-pair codes, V165's DFA module,
and this delivery's own runtime closure (loaded by file path: V163's and V165's runtime.py are also on sys.path)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V163 = ROOT / "minifly_v163_sslcola_dev_v1_20261026"
V165 = ROOT / "minifly_v165_dfa_dev_v1_20261026"
sys.path.insert(1, str(V163))
sys.path.insert(1, str(V165))
import v163_lib as V  # noqa: E402
from v163_testbed_r2 import mean_fit  # noqa: E402,F401

SET = json.loads((HERE / "SETTINGS.json").read_text())


def write_closure():
    spec = importlib.util.spec_from_file_location("v167_runtime", HERE / "runtime.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.write_closure()


def load_rehearsal():
    """Emma (A) and The Ball and the Cross (B), in the same structure as V163's load_codes()."""
    z = np.load(HERE / "cache" / "codes_rehearsal.npz")
    return {t: {k: {"idx": torch.from_numpy(z[f"{t}_idx_{k}"].astype(np.int64)), "y": z[f"{t}_y_{k}"],
                    "n": z[f"{t}_n_{k}"]} for k in V.SPLITS} for t in ("A", "B")}


def first_lines(split, k):
    e = int(split["n"][:k].sum())
    return {"idx": split["idx"][:e], "y": split["y"][:e], "n": split["n"][:k]}


def h_feats(B, W, b):
    return {k: V.h_of(B[k]["idx"], W, b) for k in V.SPLITS}
