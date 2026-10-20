"""Standalone Full151 Z1 eligibility adapter for the FE0 four-output round.

The caller supplies a *fresh* native F151ByteBrain. The outer V9.2 predictor
checkpoint belongs to the shared runner; its weights are not in this object.
Frozen parent source files and Full151 anatomy are never modified here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "BYTE_CORE_V9"))
from brain_byte import F151ByteBrain, _digest_arrays, bc  # noqa: E402

VERSION = "MINIFLY-Z1-COORDINATE-ELIG-v1"
TAU_SECONDS = 30.0
ACTIVITY_INCREMENT = 0.20
RANDOM_SEED = 20260928
ARMS = ("Z0", "Z1", "Z1_BUDGET_RANDOM")
_BASE = bc.native().model.EvoLearner
_EXPECTED_MODEL = (ROOT / "minifly/V82E/src/model_evo.py").resolve()
if Path(sys.modules[_BASE.__module__].__file__).resolve() != _EXPECTED_MODEL:
    raise RuntimeError("Full151 did not import the frozen V82E learner")


def _sha_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int32).tobytes()).hexdigest()


def _finite_time(value: float) -> float:
    out = float(value)
    if not math.isfinite(out) or out < 0:
        raise ValueError("invalid eligibility time")
    return out


def _budget_random_gate(g: np.ndarray, native_slow: np.ndarray,
                        u: np.ndarray, side: np.ndarray, *,
                        seed: int, event_no: int, random_key: int) -> tuple[np.ndarray, dict]:
    """Shuffle gate placement, matching slow L1 within each side/sign bucket."""
    out = g.copy()
    if any(v < 0 for v in (seed, event_no, random_key)):
        raise ValueError("random-control keys must be nonnegative")
    buckets: dict[str, dict[str, float | int]] = {}
    for s in (0, 1):
        for sign in (-1, 1):
            ids = np.flatnonzero((side == s) & (np.sign(u) == sign) & (native_slow != 0))
            key = f"{s}:{sign:+d}"
            if len(ids) == 0:
                buckets[key] = {"coordinates": 0, "target_l1": 0.0,
                                "actual_l1": 0.0}
                continue
            a = np.abs(native_slow[ids])
            target = float(a @ g[ids])
            capacity = float(a.sum())
            if len(ids) > 1:
                rng = np.random.default_rng(np.random.SeedSequence(
                    [RANDOM_SEED, int(seed), int(event_no), int(random_key), s,
                     0 if sign < 0 else 1]))
                shuffled = g[ids][rng.permutation(len(ids))]
            else:
                shuffled = g[ids].copy()
            if target <= 0:
                gates = np.zeros(len(ids), dtype=np.float64)
            elif capacity - target <= 1e-14 * max(1.0, capacity):
                gates = np.ones(len(ids), dtype=np.float64)
            else:
                if not np.all(shuffled > 0):
                    raise AssertionError("current active support has a zero eligibility gate")
                lo, hi = 0.0, 1.0
                while float(a @ np.minimum(1.0, hi * shuffled)) < target:
                    hi *= 2.0
                for _ in range(64):
                    mid = (lo + hi) / 2.0
                    if float(a @ np.minimum(1.0, mid * shuffled)) < target:
                        lo = mid
                    else:
                        hi = mid
                gates = np.minimum(1.0, hi * shuffled)
            out[ids] = gates
            actual = float(a @ gates)
            tolerance = 1e-12 * max(1.0, target)
            if abs(actual - target) > tolerance:
                raise AssertionError("random gate missed side/sign write budget")
            buckets[key] = {"coordinates": int(len(ids)),
                            "target_l1": target, "actual_l1": actual}
    return out, buckets


class Z1Learner(_BASE):
    """V82E Full151 event with a post-event fast/slow split patch."""

    def event_with_gate(self, dt: float, x: np.ndarray, punishment: float,
                        plastic: bool, gate: np.ndarray, *, arm: str,
                        seed: int, event_no: int, random_key: int) -> dict:
        if arm not in ARMS:
            raise ValueError(arm)
        gate = np.asarray(gate, dtype=np.float64)
        if gate.shape != self.m.slow.shape or not np.isfinite(gate).all() or (
                (gate < 0).any() or (gate > 1).any()):
            raise ValueError("invalid coordinate eligibility gate")
        pre_fast = self.m.fast[:, 1].copy()
        pre_slow = self.m.slow.copy()
        record = super().event(dt, x, punishment, bool(plastic))
        if not record.get("active", False) or not plastic:
            return record
        u = np.asarray(record["rawalpha"], dtype=np.float64)
        old_slow = np.asarray(record["rawslow"], dtype=np.float64).copy()
        old_fast = np.asarray(record["rawfast"], dtype=np.float64).copy()
        if (u.shape != gate.shape or old_slow.shape != gate.shape or
                old_fast.shape != gate.shape):
            raise AssertionError("Full151 alpha shape drift")
        buckets: dict = {}
        if arm == "Z0":
            effective = np.ones_like(gate)
        elif arm == "Z1":
            effective = gate
        else:
            effective, buckets = _budget_random_gate(
                gate, old_slow, u, np.asarray(self.m.kc_side),
                seed=seed, event_no=event_no, random_key=random_key)
        new_slow = effective * old_slow
        new_fast = old_fast + (old_slow - new_slow)
        df = math.exp(-float(dt) / float(self.m.kernel["fast_tau"]))
        ds = math.exp(-float(dt) / float(self.m.kernel["slow_tau"]))
        self.m.fast[:, 1] += (new_fast - old_fast) * df
        self.m.slow += (new_slow - old_slow) * ds
        split_error = float(np.max(np.abs(new_fast + new_slow - u)))
        patch_error = max(float(np.max(np.abs(
            self.m.fast[:, 1] - (pre_fast + new_fast) * df))),
                          float(np.max(np.abs(
            self.m.slow - (pre_slow + new_slow) * ds))))
        if (split_error > 1e-10 or patch_error > 1e-8 or
                np.any(new_slow * u < 0) or np.any(new_fast * u < 0) or
                np.any(new_slow[u == 0] != 0) or np.any(new_fast[u == 0] != 0)):
            raise AssertionError("Z1 violated the inherited alpha split certificate")
        if not (np.isfinite(self.m.fast).all() and np.isfinite(self.m.slow).all()):
            raise FloatingPointError("nonfinite Z1 memory state")
        stats = self.stats
        stats["raw_slow_L1"] += float(np.abs(new_slow).sum() - np.abs(old_slow).sum())
        stats["raw_alpha_fast_L1"] += float(np.abs(new_fast).sum() - np.abs(old_fast).sum())
        stats["max_split_error"] = max(stats["max_split_error"], split_error)
        stats["max_state_patch_error"] = max(stats["max_state_patch_error"], patch_error)
        record.update(rawfast=new_fast, rawslow=new_slow,
                      split_error=split_error, state_patch_error=patch_error,
                      z_effective_slow_L1=float(np.abs(new_slow).sum()),
                      z_gate_min=float(effective.min()),
                      z_gate_max=float(effective.max()),
                      z_budget_buckets=buckets)
        return record


class Z1Brain(F151ByteBrain):
    """FE0 byte brain with one bounded eligibility trace per writable KC."""

    def __init__(self, *, arm: str = "Z1", random_key: int = 0, **kwargs):
        if arm not in ARMS or not isinstance(random_key, int) or random_key < 0:
            raise ValueError("invalid Z1 arm or random key")
        super().__init__(**kwargs)
        self._init_z(arm, random_key)

    def _init_z(self, arm: str, random_key: int) -> None:
        self.arm = arm
        self.random_key = random_key
        if type(self.fe) is not bc.FE0:
            raise TypeError("Z1 round requires exactly FE0")
        if type(self.fly) is not _BASE:
            raise TypeError("Z1 requires the frozen V82E Full151 learner")
        self.fly.__class__ = Z1Learner
        idx = np.flatnonzero(np.abs(np.asarray(self.fly.m.T)[:, 2:4]).sum(axis=1) > 0)
        self.z_indices = np.asarray(idx, dtype=np.int32)
        self.z_indices.flags.writeable = False
        self.z_indices_sha256 = _sha_indices(self.z_indices)
        self.z = np.zeros(len(idx), dtype=np.float64)
        self.trace_elapsed = float(self.fly.m.elapsed)
        self._capture_trace = False

    def __copy__(self):
        out = object.__new__(type(self))
        out.__dict__ = self.__dict__.copy()
        out.z = self.z.copy()
        out._capture_trace = False
        return out

    def clone(self) -> "Z1Brain":
        out = copy.copy(self)
        out.fly = self.fly.clone()
        out.fe = self.fe.clone()
        out.w = self.w.copy()
        out.bias = self.bias.copy()
        out.pending_x = None if self.pending_x is None else self.pending_x.copy()
        if out.state_digest() != self.state_digest() or out.z is self.z:
            raise AssertionError("Z1 clone mismatch or trace alias")
        return out

    def fixed_digest(self) -> str:
        base = F151ByteBrain.fixed_digest(self)
        config = {"version": VERSION, "arm": self.arm,
                  "tau_seconds": TAU_SECONDS,
                  "activity_increment": ACTIVITY_INCREMENT,
                  "random_seed": RANDOM_SEED, "random_key": self.random_key,
                  "z_indices_sha256": self.z_indices_sha256}
        return hashlib.sha256((base + json.dumps(config, sort_keys=True)).encode()).hexdigest()

    def _advance_trace(self, absolute_time: float) -> None:
        target = _finite_time(absolute_time)
        dt = target - self.trace_elapsed
        if dt < -1e-9:
            raise ValueError("eligibility time moved backwards")
        if dt > 0:
            self.z *= math.exp(-max(0.0, dt) / TAU_SECONDS)
            self.trace_elapsed = target

    def _features(self, t: float, *, fe=None, fly_m=None):
        native_x, pooled, rows = super()._features(t, fe=fe, fly_m=fly_m)
        if self._capture_trace and fe is None and fly_m is None:
            self._advance_trace(self.elapsed_base + float(t))
            x = np.asarray(native_x, dtype=np.float64)
            if x.shape != (self.n_native_kc,) or not np.isfinite(x).all() or (
                    (x < 0).any() or (x > 1).any()):
                raise ValueError("invalid native KC activity for Z1")
            local = x[self.z_indices]
            self.z += ACTIVITY_INCREMENT * local * (1.0 - self.z)
            if (self.z < -1e-14).any() or (self.z > 1 + 1e-14).any():
                raise AssertionError("eligibility left [0,1]")
            np.clip(self.z, 0.0, 1.0, out=self.z)
        return native_x, pooled, rows

    def byte(self, b: int, t: float, learn: bool = True, *,
             lesion_native: bool = False, lesion_temporal: bool = False) -> float:
        if self._capture_trace:
            raise RuntimeError("nested Z1 byte update")
        self._capture_trace = True
        try:
            return super().byte(b, t, learn=learn,
                                lesion_native=lesion_native,
                                lesion_temporal=lesion_temporal)
        finally:
            self._capture_trace = False

    def _full_gate(self) -> np.ndarray:
        gate = np.ones(self.n_native_kc, dtype=np.float64)
        gate[self.z_indices] = self.z
        return gate

    def teach(self, r: int, t: float, *, write: bool = True) -> dict:
        if not isinstance(r, (int, np.integer)) or int(r) not in (0, 1):
            raise ValueError("r must be 0 or 1")
        if (self.pending_x is None or self.pending_t is None or
                abs(float(t) - self.pending_t) > 1e-9):
            raise ValueError("Teach must immediately follow its outcome byte")
        dt = self.pending_t - self.brain_t
        if dt < -1e-9:
            raise ValueError("Teach time moved backwards")
        if abs(self.trace_elapsed - (self.elapsed_base + self.pending_t)) > 1e-6:
            raise AssertionError("eligibility not current at teacher arrival")
        gate = self._full_gate()
        record = self.fly.event_with_gate(max(0.0, dt), self.pending_x,
                                          float(r), bool(write), gate,
                                          arm=self.arm, seed=self.seed,
                                          event_no=self.teach_seen,
                                          random_key=self.random_key)
        self.brain_t = self.pending_t
        self.teach_seen += 1
        self.pending_x = None
        self.pending_t = None
        return record

    def rest(self, seconds: float) -> None:
        super().rest(seconds)
        self._advance_trace(float(self.fly.m.elapsed))

    def flush(self, t: float) -> float:
        error = super().flush(t)
        self._advance_trace(float(self.fly.m.elapsed))
        return error

    def reset_stream(self) -> None:
        super().reset_stream()
        self._advance_trace(float(self.fly.m.elapsed))

    def state_digest(self) -> str:
        base = F151ByteBrain.snapshot(self)
        names = ("w", "bias", "fly_fast", "fly_slow", "fly_adapt", "fe_p", "pending_x")
        arrays = tuple(base[k] for k in names) + (self.z,)
        meta = {k: v for k, v in base.items() if k not in names}
        meta.update(arm=self.arm, random_key=self.random_key,
                    trace_elapsed=self.trace_elapsed,
                    z_indices_sha256=self.z_indices_sha256)
        return _digest_arrays(arrays, meta)

    def snapshot(self) -> dict:
        state = F151ByteBrain.snapshot(self)
        state.update(z=self.z.copy(), trace_elapsed=self.trace_elapsed,
                     arm=self.arm, random_key=self.random_key,
                     z_indices_sha256=self.z_indices_sha256)
        return state

    def restore(self, state: dict) -> None:
        if (state.get("arm") != self.arm or
                state.get("random_key") != self.random_key or
                state.get("z_indices_sha256") != self.z_indices_sha256):
            raise ValueError("incompatible Z1 checkpoint")
        z = np.asarray(state["z"], dtype=np.float64)
        if (z.shape != self.z.shape or not np.isfinite(z).all() or
                (z < 0).any() or (z > 1).any()):
            raise ValueError("invalid Z1 trace checkpoint")
        trace_elapsed = _finite_time(state["trace_elapsed"])
        F151ByteBrain.restore(self, state)
        self.z[:] = z
        self.trace_elapsed = trace_elapsed
        self._capture_trace = False

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.snapshot()
        array_keys = ("w", "bias", "fly_fast", "fly_slow", "fly_adapt",
                      "fe_p", "pending_x", "z")
        metadata = {k: v for k, v in state.items() if k not in array_keys}
        arrays = {k: state[k] for k in array_keys}
        arrays["metadata_json"] = np.frombuffer(
            json.dumps(metadata, sort_keys=True, allow_nan=False).encode(),
            dtype=np.uint8)
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".pending-",
                                         dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    @classmethod
    def load(cls, path: str | Path) -> "Z1Brain":
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(saved["metadata_json"].tobytes().decode())
            keys = ("w", "bias", "fly_fast", "fly_slow", "fly_adapt",
                    "fe_p", "pending_x", "z")
            state = {**metadata, **{key: saved[key].copy() for key in keys}}
        out = cls(arm=state["arm"], random_key=state["random_key"],
                  seed=state["seed"], temporal=state["temporal"],
                  order_via_native=state["order_via_native"],
                  lr=state["lr"], bias_lr=state["bias_lr"])
        out.restore(state)
        return out

    def resources(self) -> dict:
        out = F151ByteBrain.resources(self)
        out.update(z_arm=self.arm, z_coordinates=len(self.z_indices),
                   z_mutable_bytes=int(self.z.nbytes),
                   z_indices_fixed_bytes=int(self.z_indices.nbytes),
                   z_trace_elapsed_bytes=8)
        return out


def from_native(base: F151ByteBrain, arm: str = "Z1", *,
                random_key: int = 0) -> Z1Brain:
    """Clone a fresh native FE0 birth and install the Z0/Z1 split adapter."""
    if arm not in ARMS or not isinstance(random_key, int) or random_key < 0:
        raise ValueError("invalid Z1 arm or random key")
    if type(base) is not F151ByteBrain or type(base.fe) is not bc.FE0 or (
            type(base.fly) is not _BASE):
        raise TypeError("Z1 conversion requires a fresh native FE0 Full151 brain")
    m = base.fly.m
    if (base.brain_t != 0 or base.elapsed_base != 0 or float(m.elapsed) != 0 or
            float(base.fe.t) != 0 or base.pending_x is not None or
            base.pending_t is not None or base.bytes_seen != 0 or
            base.teach_seen != 0 or np.any(m.fast) or np.any(m.slow) or
            np.any(base.fe.p)):
        raise ValueError("Z1 conversion requires reset value, FE0 and time birth")
    out = copy.copy(base)
    out.fly = base.fly.clone()
    out.fe = base.fe.clone()
    out.w = base.w.copy()
    out.bias = base.bias.copy()
    out.pending_x = None
    if out.state_digest() != base.state_digest():
        raise AssertionError("native birth clone mismatch")
    out.__class__ = Z1Brain
    out._init_z(arm, random_key)
    if out.z.size == 0 or out.z_indices.size != len(set(map(int, out.z_indices))):
        raise AssertionError("empty or duplicate Z1 writable coordinates")
    return out
