"""V153 NATIVE arm: the unchanged FastMemory with the FE0 front end and homeostasis off (so its KC codes equal the frozen
codes the evaluator readouts see), V148 teaching (per-bit delta, temperature 2), one pass over the taught records.
Scored exactly as in V148: native 257 logits on clones, one temperature fitted on the calibration block.
"""
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "v148_probe", HERE.parent / "minifly_v148_pretrain_probe_dev_20261025" / "probe.py")
probe = importlib.util.module_from_spec(_spec)
sys.modules["v148_probe"] = probe
_spec.loader.exec_module(probe)
fastmem = probe.fastmem


def new_native():
    return fastmem.FastMemory(front="fe0", homeo=False)


def per_position_bits(L, Y, T):
    Z = L / T
    Z = Z - Z.max(1, keepdims=True)
    return -(Z[np.arange(len(Y)), Y] - np.log(np.exp(Z).sum(1))) / math.log(2)


def run_native(train, calib, held):
    mem = new_native()
    t0 = time.process_time()
    for r in train:
        mem.train(b"", r)
    train_cpu = time.process_time() - t0
    t1 = time.process_time()
    Lc, Yc = probe.collect(mem, calib)
    T, calib_bpb = probe.calibrate(Lc, Yc)
    Lh, Yh = probe.collect(mem, held)
    bits = per_position_bits(Lh, Yh, T)
    return {"held_bpb": float(bits.mean()), "held_acc": probe.accuracy(Lh, Yh), "T": T, "calib_bpb": calib_bpb,
            "train_cpu_s": round(train_cpu, 1), "eval_cpu_s": round(time.process_time() - t1, 1),
            "taught_records": len(train), "taught_symbols": int(sum(len(r) + 1 for r in train))}, bits, mem
