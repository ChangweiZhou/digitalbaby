"""A fixed, label-free recent-visible-byte address for native Full151.

This is an engineered interface baseline.  It does not encode an answer or an
arithmetic rule: a deterministic receptor sketch is assigned to the last four
non-space bytes seen on the current line.  The inherited Full151 learner still
has to acquire every cue-to-output association from the arriving teacher byte.
"""
from __future__ import annotations

import copy
import hashlib
from functools import lru_cache

import numpy as np

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "FULL151_INPUT_STEP6_FADING_PAIR_20260927"))
import model as pair_model  # noqa: E402

bc = pair_model.bc
bb = pair_model.bb
VERSION = "FULL151-VISIBLE-CONTEXT-BRIDGE-v1"
WINDOW = 4
CONTACTS = 24


@lru_cache(maxsize=65536)
def context_receptors(window: bytes) -> np.ndarray:
    if len(window) != WINDOW:
        raise ValueError("a complete visible-byte window is required")
    seed = int.from_bytes(hashlib.sha256(VERSION.encode() + window).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    p = np.zeros(88, dtype=np.float64)
    indices = rng.choice(88, CONTACTS, replace=False)
    p[indices] = rng.uniform(0.4, 1.0, size=CONTACTS)
    p.flags.writeable = False
    return p


class VisibleContextFE(bc.FE0):
    name = VERSION

    def __init__(self):
        super().__init__()
        self.visible = b""

    def feed(self, b: int, t: float) -> None:
        super().feed(b, t)
        if b == 10:
            self.visible = b""
        elif b != 32:
            self.visible = (self.visible + bytes((int(b),)))[-WINDOW:]

    def read(self) -> np.ndarray:
        if len(self.visible) < WINDOW:
            return super().read()
        return bc._norm(context_receptors(self.visible))

    def clone(self):
        out = copy.copy(self)
        out.p = self.p.copy()
        return out

    def state_arrays(self):
        return [self.p]


class VisibleContextBrain(bb.F151ByteBrain):
    def fixed_digest(self) -> str:
        description = (VERSION, WINDOW, CONTACTS, "last-visible-bytes|space-skip|newline-reset")
        return hashlib.sha256((super().fixed_digest() + repr(description)).encode()).hexdigest()

    def state_digest(self) -> str:
        return hashlib.sha256((super().state_digest() + self.fe.visible.hex()).encode()).hexdigest()

    def reset_stream(self) -> None:
        super().reset_stream()
        self.fe = VisibleContextFE()


def from_native(base: bb.F151ByteBrain) -> VisibleContextBrain:
    out = copy.copy(base)
    out.__class__ = VisibleContextBrain
    out.fly = base.fly.clone()
    out.w = base.w.copy()
    out.bias = base.bias.copy()
    out.pending_x = None if base.pending_x is None else base.pending_x.copy()
    fe = VisibleContextFE()
    fe.t = float(base.fe.t)
    fe.p[:] = base.fe.p
    out.fe = fe
    if not np.array_equal(out.fly.m.fast, base.fly.m.fast):
        raise AssertionError("native birth state changed")
    return out
