"""One fixed 88-PN causal byte front end; Full151 learning remains inherited."""
from __future__ import annotations

import hashlib
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "BYTE_CORE_1_20260922"))
sys.path.insert(0, str(ROOT / "BYTE_CORE_V9"))
import bytecore as bc  # noqa: E402
import brain_byte as bb  # noqa: E402

VERSION = "FULL151-INPUT-STEP6-FADING-PAIR-v1"
PAIR_TAU = 30.0
PAIR_GAIN = 1.0
PAIR_CONTACTS = 8


@lru_cache(maxsize=65536)
def pair_receptors(older: int, newer: int) -> np.ndarray:
    if not (0 <= older < 256 and 0 <= newer < 256):
        raise ValueError("pair bytes out of range")
    seed = int.from_bytes(hashlib.sha256(
        VERSION.encode() + bytes((older, newer))).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    result = np.zeros(88, dtype=np.float64)
    indices = rng.choice(88, PAIR_CONTACTS, replace=False)
    result[indices] = rng.uniform(0.4, 1.0, size=PAIR_CONTACTS)
    result.flags.writeable = False
    return result


class FadingPairFE(bc.FE0):
    """FE0 plus a decaying ordered-pair trace over visible non-space bytes."""

    name = "FE0+FADING-PAIR-88"

    def __init__(self):
        super().__init__()
        self.pair = np.zeros(88, dtype=np.float64)
        self.prev_visible: int | None = None

    def advance(self, t: float) -> None:
        before = self.t
        super().advance(t)
        dt = self.t-before
        if dt:
            self.pair *= math.exp(-dt/PAIR_TAU)

    def feed(self, b: int, t: float) -> None:
        b = int(b)
        if not 0 <= b < 256:
            raise ValueError("byte out of range")
        super().feed(b, t)
        if b == 10:
            self.prev_visible = None
        elif b != 32:
            if self.prev_visible is not None:
                self.pair += pair_receptors(self.prev_visible, b)
            self.prev_visible = b

    def read(self) -> np.ndarray:
        return bc._norm(self.p + PAIR_GAIN*self.pair)

    def state_arrays(self) -> list[np.ndarray]:
        return [self.p, self.pair]


class FadingPairBrain(bb.F151ByteBrain):
    """Inherited Full151 state with only the 88-PN front end replaced."""

    def fixed_digest(self) -> str:
        description = (VERSION, PAIR_TAU, PAIR_GAIN, PAIR_CONTACTS,
                       "whitespace-skipped-pair|newline-chain-break")
        return hashlib.sha256((super().fixed_digest() + repr(description)
                               ).encode()).hexdigest()

    def state_digest(self) -> str:
        # The inherited digest tracks fe.p but not our added causal pair state.
        base = super().state_digest()
        h = hashlib.sha256()
        h.update(base.encode())
        h.update(self.fe.pair.tobytes())
        h.update(str(self.fe.prev_visible).encode())
        return h.hexdigest()

    def reset_stream(self) -> None:
        super().reset_stream()
        self.fe = FadingPairFE()


def from_native(base: bb.F151ByteBrain) -> FadingPairBrain:
    """Fork checkpoint state once; preserve inherited FE0 trace at birth."""
    import copy

    out = copy.copy(base)
    out.__class__ = FadingPairBrain
    out.fly = base.fly.clone()
    out.w = base.w.copy()
    out.bias = base.bias.copy()
    out.pending_x = None if base.pending_x is None else base.pending_x.copy()
    fe = FadingPairFE()
    fe.t = float(base.fe.t)
    fe.p[:] = base.fe.p
    out.fe = fe
    if (not np.array_equal(out.fly.m.fast, base.fly.m.fast) or
            not np.array_equal(out.fly.m.slow, base.fly.m.slow) or
            not np.array_equal(out.fly.m.adapt, base.fly.m.adapt)):
        raise AssertionError("native birth state changed")
    return out
