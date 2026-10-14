"""A fixed-size, categorical plastic memory with unrestricted byte payloads.

This is an engineered E0/E1 baseline, not a claim about general intelligence,
arithmetic generalization, or native Full151. Its fixed, label-free address is
hashed from the ordered, bounded context. Similar contexts are not deliberately
given similar addresses. Learned weights, rather than a key-answer dictionary,
cause its next-event output. Address collisions and interference remain possible.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import numpy as np


EOS = 256
_BEGIN = 257
_REPLY = 258
_OUTPUTS = 257
_SCHEMA = "GENERIC_BYTE_IO_DEV_V1"


@dataclass(frozen=True)
class CoreConfig:
    cells: int = 4096
    contacts: int = 16
    max_context_events: int = 4096
    learning_rate: float = 4.0
    address_seed: int = 20261012

    def __post_init__(self):
        for name in ("cells", "contacts", "max_context_events", "address_seed"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer, not a boolean")
        if not (1 <= self.contacts <= self.cells):
            raise ValueError("contacts must be between one and cells")
        if self.max_context_events < 1 or self.address_seed < 0:
            raise ValueError("positive context bound and nonnegative seed required")
        if isinstance(self.learning_rate, bool) or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")


@dataclass(frozen=True)
class Prediction:
    # A new array is returned: editing it cannot change the learned bank.
    probabilities: np.ndarray
    event: int


@dataclass(frozen=True)
class Generation:
    payload: bytes
    status: str  # ENDED means the core selected EOS; TRUNCATED means budget only.
    selected_events: int


class ByteCore:
    def __init__(self, config: CoreConfig | None = None):
        self.config = config or CoreConfig()
        # ONE vector-valued plastic bank, not 257 copied learner instances.
        self._weights = np.zeros((self.config.cells, _OUTPUTS), dtype=np.float32)
        self._context = np.zeros(self.config.max_context_events, dtype=np.uint16)
        self._context_len = 0
        self._next = 0
        self._open = False
        self._generated_end = False
        self.learning_events = 0
        self.observed_payload_bytes = 0
        self.generated_payload_bytes = 0
        self.context_overflow_events = 0
        self.elapsed_seconds = 0.0

    @staticmethod
    def _byte(byte: int) -> int:
        if type(byte) is not int:
            raise TypeError("payload must be an integer byte; booleans are not bytes")
        if not 0 <= byte <= 255:
            raise ValueError("payload byte outside 0..255; use observe_end() for EOS")
        return byte

    @staticmethod
    def _learning_flag(learn: bool):
        if type(learn) is not bool:
            raise TypeError("learn must be a boolean")

    def _require_open(self):
        if not self._open:
            raise RuntimeError("begin_sequence() is required before input or prediction")

    def _ordered_context(self) -> np.ndarray:
        if self._context_len < len(self._context):
            return self._context[:self._context_len]
        return np.concatenate((self._context[self._next:], self._context[:self._next]))

    def _append(self, event: int):
        if self._context_len == len(self._context):
            self.context_overflow_events += 1
        else:
            self._context_len += 1
        self._context[self._next] = event
        self._next = (self._next + 1) % len(self._context)

    def begin_sequence(self):
        """Explicit out-of-band boundary; discards context, never learned weights.

        Starting a new sequence does not manufacture an EOS training target for
        the previous sequence. Call observe_end() when an ending was observed.
        """
        self._context.fill(0)
        self._context_len = 0
        self._next = 0
        self._open = True
        self._generated_end = False
        self._append(_BEGIN)

    def start_reply(self):
        """Explicit framing only, with no answer, expected length or task label."""
        self._require_open()
        self._append(_REPLY)

    def rest(self, seconds: float):
        """Advance declared simulated time without sleeping or changing weights.

        This engineered stable bank has no spontaneous decay or biological
        consolidation mechanism. Retention tests can measure interference from
        later writes; time-only stability here is a disclosed design property.
        """
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise TypeError("seconds must be a finite nonnegative number")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("seconds must be finite and nonnegative")
        elapsed = self.elapsed_seconds + seconds
        if not math.isfinite(elapsed):
            raise ValueError("elapsed time would overflow")
        self.elapsed_seconds = float(elapsed)

    def _address(self) -> np.ndarray:
        # Full ordered window, including all raw bytes and framing events.
        context = self._ordered_context().astype("<u2", copy=False).tobytes()
        header = f"BYTE-IO-ORDERED-v1/{self.config.address_seed}/".encode("ascii")
        source = hashlib.shake_256(header + context)
        count = max(32, self.config.contacts * 2)
        while True:
            words = np.frombuffer(source.digest(count * 8), dtype="<u8")
            address = []
            used = set()
            for word in words:
                cell = int(word) % self.config.cells
                if cell not in used:
                    used.add(cell)
                    address.append(cell)
                    if len(address) == self.config.contacts:
                        return np.asarray(sorted(address), dtype=np.int64)
            count *= 2

    def _prediction_at(self, address: np.ndarray) -> Prediction:
        logits = np.sum(self._weights[address], axis=0, dtype=np.float64)
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        return Prediction(probabilities=probabilities, event=int(np.argmax(probabilities)))

    def prediction(self) -> Prediction:
        """Read-only pre-observation prediction; it does not create pending state.

        observe() independently computes the same prediction before updating
        weights or adding its target to context. Multiple reads are safe.
        """
        self._require_open()
        return self._prediction_at(self._address())

    def _observe_event(self, event: int, learn: bool) -> Prediction:
        self._require_open()
        address = self._address()
        prediction = self._prediction_at(address)  # Target is not in context yet.
        if learn:
            gradient = -prediction.probabilities
            gradient[event] += 1.0
            increment = (self.config.learning_rate / self.config.contacts) * gradient
            self._weights[address] += increment.astype(np.float32)
            self.learning_events += 1
        self._append(event)
        return prediction

    def observe(self, byte: int, learn: bool = True) -> Prediction:
        """Consume actual payload byte, optionally learn, return prior prediction."""
        byte = self._byte(byte)
        self._learning_flag(learn)
        prediction = self._observe_event(byte, learn)
        self.observed_payload_bytes += 1
        return prediction

    def observe_bytes(self, payload: bytes | bytearray | memoryview, learn: bool = True):
        """Chunk convenience API with exactly the same per-byte transition.

        Supports contiguous raw buffers and one-dimensional byte views, including
        strided/signed byte views. Non-contiguous multi-byte element views must be
        explicitly serialized by the caller. Does not accumulate predictions or
        copy an entire strided byte view into a history-sized temporary buffer.
        """
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes, bytearray or memoryview")
        self._learning_flag(learn)
        view = memoryview(payload)
        if not (view.ndim == 1 and view.itemsize == 1):
            view = view.cast("B")
        for byte in view:
            value = byte[0] if isinstance(byte, bytes) else int(byte) & 255
            self.observe(value, learn=learn)

    def observe_end(self, learn: bool = True) -> Prediction:
        """Consume an actually observed out-of-band EOS and optionally learn it."""
        self._learning_flag(learn)
        prediction = self._observe_event(EOS, learn)
        self._open = False
        self._generated_end = False
        return prediction

    def generate(self, max_bytes: int) -> Generation:
        """Autoregress from own outputs; never teach model guesses as observations.

        At most max_bytes payload bytes and one terminal EOS decision are made.
        A final EOS check is allowed after the last permitted payload byte. A
        non-EOS prediction there returns TRUNCATED without consuming that byte.
        No expected answer length is supplied.
        """
        if type(max_bytes) is not int:
            raise TypeError("max_bytes must be an integer")
        if max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self._require_open()
        output = bytearray()
        decisions = 0
        while True:
            event = self.prediction().event
            decisions += 1
            if event == EOS:
                self._append(EOS)
                self._open = False
                self._generated_end = True
                return Generation(bytes(output), "ENDED", decisions)
            if len(output) == max_bytes:
                return Generation(bytes(output), "TRUNCATED", decisions)
            self._append(event)
            output.append(event)
            self.generated_payload_bytes += 1

    def learned_digest(self) -> str:
        """Audit only; no whole-bank copy/hash is needed by production IO."""
        digest = hashlib.sha256()
        digest.update(self._weights.astype("<f4", copy=False).tobytes())
        digest.update(str(self.learning_events).encode("ascii"))
        return digest.hexdigest()

    def transient_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self._context.astype("<u2", copy=False).tobytes())
        digest.update(json.dumps(self._metadata(), sort_keys=True).encode("utf8"))
        return digest.hexdigest()

    def _metadata(self) -> dict:
        return {"schema": _SCHEMA, "config": asdict(self.config),
                "context_len": self._context_len, "next": self._next,
                "open": self._open, "generated_end": self._generated_end,
                "learning_events": self.learning_events,
                "observed_payload_bytes": self.observed_payload_bytes,
                "generated_payload_bytes": self.generated_payload_bytes,
                "context_overflow_events": self.context_overflow_events,
                "elapsed_seconds": self.elapsed_seconds}

    def save_checkpoint(self, path: str | Path):
        """Explicit checkpoint IO, not a production per-byte operation."""
        metadata = np.frombuffer(json.dumps(self._metadata(), sort_keys=True).encode("utf8"), dtype=np.uint8)
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, metadata=metadata, weights=self._weights, context=self._context)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "ByteCore":
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"metadata", "weights", "context"}:
                raise ValueError("unexpected checkpoint fields")
            metadata = json.loads(archive["metadata"].tobytes().decode("utf8"))
            if metadata.get("schema") != _SCHEMA:
                raise ValueError("unsupported checkpoint schema")
            core = cls(CoreConfig(**metadata["config"]))
            weights, context = archive["weights"], archive["context"]
            if weights.dtype != np.float32 or weights.shape != core._weights.shape or not np.isfinite(weights).all():
                raise ValueError("invalid learned weights")
            if context.dtype != np.uint16 or context.shape != core._context.shape or np.any(context > _REPLY):
                raise ValueError("invalid context ring")
            integer_fields = ("context_len", "next", "learning_events", "observed_payload_bytes", "generated_payload_bytes", "context_overflow_events")
            for name in integer_fields:
                if type(metadata.get(name)) is not int or metadata[name] < 0:
                    raise ValueError(f"invalid checkpoint {name}")
            if metadata["context_len"] > len(core._context) or metadata["next"] >= len(core._context):
                raise ValueError("invalid ring bounds")
            if metadata["context_len"] < len(core._context) and metadata["next"] != metadata["context_len"]:
                raise ValueError("inconsistent incomplete ring position")
            for name in ("open", "generated_end"):
                if type(metadata.get(name)) is not bool:
                    raise ValueError(f"invalid checkpoint {name}")
            if metadata["generated_end"] and metadata["open"]:
                raise ValueError("ended generation cannot be open")
            elapsed = metadata.get("elapsed_seconds")
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("invalid checkpoint elapsed_seconds")
            core._weights[:] = weights
            core._context[:] = context
            core._context_len = metadata["context_len"]
            core._next = metadata["next"]
            core._open = metadata["open"]
            core._generated_end = metadata["generated_end"]
            core.elapsed_seconds = float(elapsed)
            for name in integer_fields[2:]:
                setattr(core, name, metadata[name])
            return core

    def storage_accounting(self) -> dict:
        """Array storage and explicit bounds, excluding Python object overhead."""
        return {"learned_weight_bytes": self._weights.nbytes,
                "fixed_context_ring_bytes": self._context.nbytes,
                "persistent_array_bytes": self._weights.nbytes + self._context.nbytes,
                "output_categories": _OUTPUTS, "payload_categories": 256,
                "active_cells_per_event": self.config.contacts,
                "context_event_bound": self.config.max_context_events,
                "learning_events": self.learning_events,
                "context_overflow_events": self.context_overflow_events,
                "elapsed_seconds": self.elapsed_seconds,
                "time_dynamics": "stable engineered bank; rest advances clock only",
                "production_context_work": "O(max_context_events)",
                "production_read_write_work": "O(contacts * 257)",
                "checkpoint_and_digest_work": "O(cells * 257 + max_context_events)"}
