"""Versioned FULL_151 fork with an internal causal byte-prediction circuit.

The inherited FULL_151 mushroom-body value circuit remains intact.  A small
engineered temporal KC population and its own plastic output synapses make
the same fly predict the next raw byte.  This is a testable architecture,
not a claim that the added cells or learning rule were observed in flies.

No frozen parent source is modified.  The ordered-pair cells use only the
two *previously observed* bytes; the current target is scored before feed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "BYTE_CORE_1_20260922"))
import bytecore as bc  # noqa: E402


VERSION = "F151-BYTE-V9.1"
N_POOL = 128
N_LAST = 256
N_PAIR_HASHES = 2
N_PAIR_PER_HASH = 256
N_INPUT = N_POOL + N_LAST + N_PAIR_HASHES * N_PAIR_PER_HASH
N_OUTPUT = 256


def _digest_arrays(arrays: tuple[np.ndarray, ...], metadata: dict | None = None) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode("ascii"))
        h.update(a.tobytes())
    if metadata is not None:
        h.update(json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                            allow_nan=False).encode("utf-8"))
    return h.hexdigest()


class F151ByteBrain:
    """One fly with inherited value memory and internal byte prediction.

    The native KC code still comes from FULL_151's fixed PN→KC connectome
    projection, driven by the empirically stronger FE0 byte front end.
    Its active KCs are pooled into 128 fixed readout microcolumns to bound
    output synapse count.  Separate engineered temporal cells encode the
    immediately preceding byte and two independent hashes of the ordered
    preceding byte pair.  All 896×256 prediction synapses are owned by this
    brain and updated locally by presynaptic activity times the output error.

    ``byte`` always scores and advances the exact same sensory trajectory.
    ``learn=False`` freezes only predictive plasticity.  ``teach`` writes the
    inherited FULL_151 value state, and never the prediction synapses.
    """

    def __init__(self, *, temporal: bool = True, order_via_native: bool = True,
                 seed: int = 20260925,
                 lr: float = 0.03, bias_lr: float = 0.001):
        if not math.isfinite(lr) or lr <= 0 or not math.isfinite(bias_lr) or bias_lr < 0:
            raise ValueError("invalid learning rate")
        self.temporal = bool(temporal)
        self.order_via_native = bool(order_via_native)
        self.seed = int(seed)
        self.lr = float(lr)
        self.bias_lr = float(bias_lr)
        self.fly = bc.build_full151()
        self.model = bc.native().model
        self.common = bc.native().common
        self.fe = bc.FE0()
        self.n_native_kc = len(self.fly.m.adapt)
        self.pool_of_kc, self.pair_hash = self._fixed_maps()
        self.w = np.zeros((N_INPUT, N_OUTPUT), dtype=np.float32)
        self.bias = np.zeros(N_OUTPUT, dtype=np.float32)
        self.prev1: int | None = None
        self.prev2: int | None = None
        self.bytes_seen = 0
        self.logp_sum = 0.0
        # Brain time uses the v8 CAUSAL_DT convention: every byte interval
        # is committed once, with the outcome interval receiving Teach.
        self.brain_t = 0.0
        self.elapsed_base = float(self.fly.m.elapsed)
        self.last_byte_t: float | None = None
        self.pending_x: np.ndarray | None = None
        self.pending_t: float | None = None
        self.teach_seen = 0

    def _fixed_maps(self) -> tuple[np.ndarray, np.ndarray]:
        root = f"{VERSION}|{self.seed}|"
        pool = np.empty(self.n_native_kc, dtype=np.uint8)
        for i in range(self.n_native_kc):
            pool[i] = hashlib.blake2s(f"{root}pool|{i}".encode(), digest_size=1).digest()[0] % N_POOL
        pair = np.empty((256, 256, N_PAIR_HASHES), dtype=np.uint8)
        for a in range(256):
            for b in range(256):
                pair[a, b] = np.frombuffer(hashlib.blake2s(
                    f"{root}pair|{a}|{b}".encode(), digest_size=N_PAIR_HASHES).digest(),
                    dtype=np.uint8)
        return pool, pair

    def fixed_digest(self) -> str:
        return _digest_arrays((self.pool_of_kc, self.pair_hash),
                              {"version": VERSION, "seed": self.seed,
                               "order_via_native": self.order_via_native,
                               "full151_id": bc.FULL_151_ID})

    def _features(self, t: float, *, fe=None, fly_m=None) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
        fe = self.fe if fe is None else fe
        fly_m = self.fly.m if fly_m is None else fly_m
        fe.advance(t)
        fe_p = fe.read()
        # The retained value branch always receives unmodified FE0 so its
        # F3 write and read match the frozen FULL_151 adapter.  The byte
        # prediction branch may additionally route ordered history through
        # the *same fixed* PN→KC B before the bounded KC output pool.
        assoc_x = np.asarray(self.model.encode_sparse(fly_m, fe_p), dtype=np.float64)
        pred_p = fe_p
        if self.order_via_native and self.prev1 is not None:
            pred_p = fe_p.copy() + 0.8 * bc.R_TABLE[self.prev1]
            if self.prev2 is not None:
                p0, p1 = (int(z) for z in self.pair_hash[self.prev2, self.prev1])
                pred_p += 0.4 * bc.R_TABLE[p0] + 0.4 * bc.R_TABLE[p1]
            peak = float(pred_p.max())
            if peak > 0:
                pred_p /= peak
        pred_x = (np.asarray(self.model.encode_sparse(fly_m, pred_p), dtype=np.float64)
                  if pred_p is not fe_p else assoc_x)
        active = np.flatnonzero(pred_x)
        pooled = np.bincount(self.pool_of_kc[active].astype(np.intp),
                             minlength=N_POOL).astype(np.float32)
        norm = float(np.linalg.norm(pooled))
        if norm > 0:
            pooled /= norm
        temporal_rows: list[int] = []
        if self.temporal and self.prev1 is not None:
            temporal_rows.append(N_POOL + self.prev1)
            if self.prev2 is not None:
                p0, p1 = (int(z) for z in self.pair_hash[self.prev2, self.prev1])
                temporal_rows.extend((N_POOL + N_LAST + p0,
                                      N_POOL + N_LAST + N_PAIR_PER_HASH + p1))
        return assoc_x, pooled, tuple(temporal_rows)

    def _distribution(self, pooled: np.ndarray, rows: tuple[int, ...], *,
                      lesion_native: bool = False,
                      lesion_temporal: bool = False) -> np.ndarray:
        logits = self.bias.astype(np.float64)
        if not lesion_native:
            logits += pooled.astype(np.float64) @ self.w[:N_POOL].astype(np.float64)
        if rows and not lesion_temporal:
            logits += self.w[list(rows)].sum(axis=0, dtype=np.float64)
        logits -= float(logits.max())
        p = np.exp(logits)
        p /= float(p.sum())
        return p

    def predict(self, t: float, *, lesion_native: bool = False,
                lesion_temporal: bool = False) -> np.ndarray:
        """Pre-byte prediction with optional readout-only route lesions.

        The lesions leave sensory activity and persistent state untouched, so
        a held-out comparison localizes which input route carries prediction.
        """
        t = float(t)
        if not math.isfinite(t) or t < self.brain_t - 1e-9 or (
                self.pending_t is not None and t < self.pending_t - 1e-9):
            raise ValueError("invalid or backward prediction time")
        fe = self.fe.clone()
        if self.pending_t is None:
            m = self.fly.m
        else:
            trial = self.fly.clone()
            trial.rest(max(0.0, self.pending_t - self.brain_t))
            m = trial.m
        _, pooled, rows = self._features(t, fe=fe, fly_m=m)
        return self._distribution(pooled, rows, lesion_native=lesion_native,
                                  lesion_temporal=lesion_temporal)

    def _commit_pending(self) -> None:
        if self.pending_t is not None:
            dt = self.pending_t - self.brain_t
            if dt < -1e-9:
                raise ValueError("byte stream moved backwards")
            self.fly.rest(max(0.0, dt))
            self.brain_t = self.pending_t
            self.pending_x = None
            self.pending_t = None

    def byte(self, b: int, t: float, learn: bool = True, *,
             lesion_native: bool = False,
             lesion_temporal: bool = False) -> float:
        """Score before byte arrival, then optionally update local synapses."""
        if not isinstance(b, (int, np.integer)) or not 0 <= int(b) <= 255:
            raise ValueError("b must be an integer byte")
        if learn and (lesion_native or lesion_temporal):
            raise ValueError("readout lesions are only defined for frozen scoring")
        b, t = int(b), float(t)
        if not math.isfinite(t) or t < self.brain_t - 1e-9 or (
                self.pending_t is not None and t < self.pending_t - 1e-9):
            raise ValueError("invalid or backward byte time")
        self._commit_pending()
        native_x, pooled, rows = self._features(t)
        p = self._distribution(pooled, rows, lesion_native=lesion_native,
                               lesion_temporal=lesion_temporal)
        loss = -math.log2(float(p[b]))
        if learn:
            # Output-cell prediction error is broadcast to only its active
            # presynaptic contacts.  Normalization is a readout operation.
            error = p.astype(np.float32)
            error[b] -= 1.0
            self.w[:N_POOL] -= np.float32(self.lr) * pooled[:, None] * error[None, :]
            for row in rows:
                self.w[row] -= np.float32(self.lr) * error
            self.bias -= np.float32(self.bias_lr) * error
        self.pending_x = native_x.copy()  # pre-outcome FE0/KC state for Teach
        self.pending_t = t
        self.fe.feed(b, t)
        self.prev2, self.prev1 = self.prev1, b
        self.bytes_seen += 1
        self.last_byte_t = t
        self.logp_sum += loss
        return loss

    def association_value(self, t: float) -> float:
        """Read-only inherited value at t on the pre-byte FE0/KC state."""
        t = float(t)
        if not math.isfinite(t) or t < self.brain_t - 1e-9 or (
                self.pending_t is not None and t < self.pending_t - 1e-9):
            raise ValueError("invalid or backward read time")
        fe = self.fe.clone()
        fe.advance(t)
        native_x = np.asarray(self.model.encode_sparse(self.fly.m, fe.read()), dtype=np.float64)
        n = self.fly.clone()
        if self.pending_t is not None:
            n.rest(self.pending_t - self.brain_t)
            n.rest(t - self.pending_t)
        else:
            n.rest(t - self.brain_t)
        n = n.m
        dx = self.common.observed_activity(n, np.atleast_2d(native_x))
        alpha = n.expression(dx)
        pred = self.fly.reader.predict(dx)
        return float((alpha - pred).mean(1)[0])

    def teach(self, r: int, t: float, *, write: bool = True) -> None:
        """Update only inherited FULL_151 value state after an outcome byte.

        ``pending_x`` is the KC code captured *before* that byte entered FE0.
        The update duration is the actual uncommitted interval since the
        previous byte, matching v8's CAUSAL_DT protocol (normally 30/14 s).
        """
        if not isinstance(r, (int, np.integer)) or int(r) not in (0, 1):
            raise ValueError("r must be 0 or 1")
        if self.pending_x is None or self.pending_t is None or abs(float(t) - self.pending_t) > 1e-9:
            raise ValueError("Teach must immediately follow its outcome byte")
        dt = self.pending_t - self.brain_t
        if dt < -1e-9:
            raise ValueError("Teach time moved backwards")
        self.fly.event(max(0.0, dt), self.pending_x, float(r), bool(write))
        self.brain_t = self.pending_t
        self.teach_seen += 1
        self.pending_x = None
        self.pending_t = None

    def rest(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid rest")
        self._commit_pending()
        self.fly.rest(float(seconds))
        self.brain_t += float(seconds)
        self.fe.advance(self.fe.t + float(seconds))
        self.pending_x = None
        self.pending_t = None

    def flush(self, t: float) -> float:
        """Commit all unreinforced time through t; return native clock error."""
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("invalid flush time")
        self._commit_pending()
        dt = t - self.brain_t
        if dt < -1e-9:
            raise ValueError("flush moved backwards")
        self.fly.rest(max(0.0, dt))
        self.brain_t = t
        self.fe.advance(t)
        return abs(float(self.fly.m.elapsed) - self.elapsed_base - t)

    def reset_stream(self) -> None:
        """Rebase event times for an independent source; retain learned state.

        The final pending byte is committed.  The inherited native elapsed
        clock keeps accumulating; only the input source's local time origin
        and short sensory state are reset.
        """
        self._commit_pending()
        self.brain_t = 0.0
        self.elapsed_base = float(self.fly.m.elapsed)
        self.fe = bc.FE0()
        self.prev1 = self.prev2 = None
        self.last_byte_t = None
        self.pending_x = None
        self.pending_t = None

    def pair_collision_stats(self, pairs) -> dict[str, int]:
        """Audit hash collisions for an iterable of ordered (older, newer) bytes."""
        observed = {(int(a), int(b)) for a, b in pairs}
        if any(not (0 <= a <= 255 and 0 <= b <= 255) for a, b in observed):
            raise ValueError("pairs must consist of bytes")
        signatures = [(int(self.pair_hash[a, b, 0]), int(self.pair_hash[a, b, 1]))
                      for a, b in sorted(observed)]
        return {"observed_pairs": len(observed),
                "unique_signatures": len(set(signatures)),
                "full_signature_collisions": len(signatures) - len(set(signatures))}

    def parameter_digest(self) -> str:
        """Learned byte-prediction synapses only; frozen during text TEST.

        The inherited fast/slow association state has ordinary time decay
        even with no teaching, so it is tracked separately below.
        """
        return _digest_arrays((self.w, self.bias),
                              {"version": VERSION, "fixed_digest": self.fixed_digest()})

    def association_state_digest(self) -> str:
        m = self.fly.m
        return _digest_arrays((m.fast, m.slow, m.adapt),
                              {"full151_id": bc.FULL_151_ID})

    def state_digest(self) -> str:
        s = self.snapshot()
        arrays = tuple(s[k] for k in ("w", "bias", "fly_fast", "fly_slow",
                                       "fly_adapt", "fe_p", "pending_x"))
        meta = {k: v for k, v in s.items() if k not in
                ("w", "bias", "fly_fast", "fly_slow", "fly_adapt", "fe_p", "pending_x")}
        return _digest_arrays(arrays, meta)

    def snapshot(self) -> dict:
        m = self.fly.m
        return {"version": VERSION, "seed": self.seed, "temporal": self.temporal,
                "order_via_native": self.order_via_native,
                "lr": self.lr, "bias_lr": self.bias_lr,
                "full151_id": bc.FULL_151_ID, "fixed_digest": self.fixed_digest(),
                "w": self.w.copy(), "bias": self.bias.copy(),
                "fly_fast": m.fast.copy(), "fly_slow": m.slow.copy(),
                "fly_adapt": m.adapt.copy(), "fly_elapsed": float(m.elapsed),
                "fly_event_count": int(m.event_count),
                "fly_presentation_count": int(m.presentation_count),
                "brain_t": self.brain_t,
                "elapsed_base": self.elapsed_base,
                "fe_p": self.fe.p.copy(), "fe_t": float(self.fe.t),
                "prev1": self.prev1, "prev2": self.prev2,
                "bytes_seen": self.bytes_seen, "logp_sum": self.logp_sum,
                "last_byte_t": self.last_byte_t,
                "pending_x": (np.zeros(self.n_native_kc, np.float64) if self.pending_x is None
                              else self.pending_x.copy()),
                "pending_valid": self.pending_x is not None,
                "pending_t": self.pending_t, "teach_seen": self.teach_seen}

    def restore(self, state: dict) -> None:
        for key, expected in (("version", VERSION), ("seed", self.seed),
                              ("temporal", self.temporal),
                              ("order_via_native", self.order_via_native),
                              ("full151_id", bc.FULL_151_ID),
                              ("fixed_digest", self.fixed_digest())):
            if state[key] != expected:
                raise ValueError(f"checkpoint {key} mismatch")
        if float(state["lr"]) != self.lr or float(state["bias_lr"]) != self.bias_lr:
            raise ValueError("checkpoint learning-rate mismatch")
        for name, target in (("w", self.w), ("bias", self.bias),
                             ("fly_fast", self.fly.m.fast),
                             ("fly_slow", self.fly.m.slow),
                             ("fly_adapt", self.fly.m.adapt),
                             ("fe_p", self.fe.p)):
            source = np.asarray(state[name])
            if source.shape != target.shape or not np.isfinite(source).all():
                raise ValueError(f"invalid checkpoint {name}")
            target[...] = source
        pending = np.asarray(state["pending_x"], dtype=np.float64)
        if pending.shape != (self.n_native_kc,) or not np.isfinite(pending).all():
            raise ValueError("invalid pending KC state")
        self.fly.m.elapsed = float(state["fly_elapsed"])
        self.fly.m.event_count = np.uint64(int(state["fly_event_count"]))
        self.fly.m.presentation_count = np.uint64(int(state["fly_presentation_count"]))
        self.brain_t = float(state["brain_t"])
        self.elapsed_base = float(state["elapsed_base"])
        self.fe.t = float(state["fe_t"])
        self.prev1 = None if state["prev1"] is None else int(state["prev1"])
        self.prev2 = None if state["prev2"] is None else int(state["prev2"])
        self.bytes_seen = int(state["bytes_seen"])
        self.logp_sum = float(state["logp_sum"])
        self.last_byte_t = None if state["last_byte_t"] is None else float(state["last_byte_t"])
        self.pending_x = pending.copy() if state["pending_valid"] else None
        self.pending_t = None if state["pending_t"] is None else float(state["pending_t"])
        self.teach_seen = int(state["teach_seen"])

    def save(self, path: str | Path) -> None:
        """Atomically save the complete fly without pickle or source assets."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        s = self.snapshot()
        array_keys = ("w", "bias", "fly_fast", "fly_slow", "fly_adapt", "fe_p", "pending_x")
        metadata = {k: v for k, v in s.items() if k not in array_keys}
        arrays = {k: s[k] for k in array_keys}
        arrays["metadata_json"] = np.frombuffer(json.dumps(metadata, sort_keys=True,
                                                           allow_nan=False).encode(), dtype=np.uint8)
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".pending-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(f, **arrays)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    @classmethod
    def load(cls, path: str | Path) -> "F151ByteBrain":
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(z["metadata_json"].tobytes().decode())
            state = {**metadata, **{k: z[k].copy() for k in
                                    ("w", "bias", "fly_fast", "fly_slow",
                                     "fly_adapt", "fe_p", "pending_x")}}
        out = cls(temporal=state["temporal"], order_via_native=state["order_via_native"],
                  seed=state["seed"],
                  lr=state["lr"], bias_lr=state["bias_lr"])
        out.restore(state)
        return out

    def resources(self) -> dict:
        m = self.fly.m
        return {"version": VERSION,
                "native_pn": 88, "native_kc": self.n_native_kc,
                "order_via_native": self.order_via_native,
                "parallel_temporal_readout": self.temporal,
                "added_temporal_kc": N_LAST + N_PAIR_HASHES * N_PAIR_PER_HASH,
                "native_kc_pool_bins": N_POOL,
                "prediction_output_cells": N_OUTPUT,
                "prediction_synapses": int(self.w.size),
                "prediction_mutable_bytes": int(self.w.nbytes + self.bias.nbytes),
                "inherited_mutable_bytes": int(m.fast.nbytes + m.slow.nbytes + m.adapt.nbytes),
                "new_fixed_mapping_bytes": int(self.pool_of_kc.nbytes + self.pair_hash.nbytes),
                "sensory_transient_bytes": int(self.fe.p.nbytes),
                "fixed_B_nnz": int(m.B.nnz),
                "fixed_digest": self.fixed_digest()}
