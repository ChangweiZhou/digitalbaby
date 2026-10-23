"""Four-store wrappers with a per-record write ledger, and the two-bank R system.

Ledger rows are appended to ``mechanism_events`` (returned per branch by the frozen platform life):
    ["w", record, bank, store, write_flag, applied_alpha_L1]            native binary teach
    ["s", record, bank, store, write_flag, applied_alpha_L1, b, s]      signed interface teach
Mechanism rows: ["R", ...], ["Z", ...], ["P", ...] as documented in SPEC_DRAFT.md.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

import paths  # noqa: F401
from common_platform import CHANNELS, FourStore, clone_model

BANK_NATIVE, BANK_SHARED, BANK_PRIVATE = 0, 0, 1
R_ARMS = ("R0", "R1", "R1_rand", "R0_signed", "R3", "R3_randtarget")
SIGNED = ("R0_signed", "R3", "R3_randtarget")
R_WINDOW = 24


def _active(domain: str) -> int:
    return 4 if domain == "fact" else 2


def _check_teacher(answer: int, domain: str) -> None:
    if answer not in CHANNELS or (domain == "relation" and answer not in b"01"):
        raise ValueError("answer is not a permitted arriving byte")


class LedgerFourStore(FourStore):
    """Platform FourStore semantics; every store teach is logged with the flag the native event received."""

    def __init__(self, stores, bank: int = BANK_NATIVE, z_ref_table: dict | None = None):
        super().__init__(stores)
        self.bank = bank
        self.z_ref_table = z_ref_table     # Z2_rand only: paired Z2 per-bucket dose, read-only

    def clone(self):
        out = type(self)([clone_model(m) for m in self.stores], self.bank, self.z_ref_table)
        out.mechanism_events = list(self.mechanism_events)
        out.context = self.context
        return out

    def teach(self, answer: int, t: float, domain: str, *, write: bool):
        _check_teacher(answer, domain)
        if self.context is None or self.context[4] != domain:
            raise AssertionError("record context missing or domain changed")
        branch, index = self.context[1], self.context[2]
        for m in self.stores:
            m.byte(answer, t, learn=False)
        for j, m in enumerate(self.stores):
            did = bool(write and j < _active(domain))
            if getattr(m, "z_arm", None) is not None:
                m.z_ctx = f"{branch}|{index}"
                m.z_ref = None if self.z_ref_table is None else self.z_ref_table.get((branch, index, j))
            applied = m.teach_logged(int(CHANNELS[j] != answer), t, write=did)
            self.mechanism_events.append(["w", index, self.bank, j, int(did), round(applied, 12)])
            self._after_store(index, j, m)

    def _after_store(self, index, j, m):
        z = getattr(m, "last_z", None)
        if z is not None:
            self.mechanism_events.append(["Z", index, j, z["conflicts"], z["native_slow_l1"],
                                          z["gated_slow_l1"], z["gate_target_l1"],
                                          z.get("buckets")])
            m.last_z = None
        log = getattr(m, "p_log", None)
        if log:
            dl1, wl1, wmax, wmin = log.pop()
            self.mechanism_events.append(["P", index, j, dl1, wl1, wmax, wmin])
        diag = getattr(m, "p_diag", None)
        if diag:
            d = diag.pop()
            if j == 0:
                self.mechanism_events.append(["PD", index, j, d])


def _jaccard_novelty(x: np.ndarray, buffer: list) -> float:
    if not buffer:
        return 1.0
    a = x.astype(bool)
    best = 0.0
    for prior in buffer:
        union = np.count_nonzero(a | prior)
        best = max(best, (np.count_nonzero(a & prior) / union) if union else 1.0)
    return 1.0 - best


def derangement(n: int, key: str) -> list[int]:
    """SHA256-ranked derangement of range(n) (n in {2, 4})."""
    order = sorted(range(n), key=lambda j: hashlib.sha256(f"{key}|{j}".encode()).digest())
    perm = [0] * n
    for i in range(n):
        perm[order[i]] = order[(i + 1) % n]
    if any(perm[i] == i for i in range(n)):
        raise AssertionError("not a derangement")
    return perm


class RSystem:
    """Shared FE0 bank (4 stores) + private CONTENT bank (4 stores) with the arm's private write rule."""

    def __init__(self, shared: list, private: list, arm: str, *, scales, theta: float,
                 rand_gates: dict | None = None):
        if arm not in R_ARMS or len(shared) != 4 or len(private) != 4:
            raise ValueError("invalid R construction")
        if not all(math.isfinite(s) and s >= 1e-8 for s in scales):
            raise ValueError("invalid readout scales")
        self.shared, self.private = list(shared), list(private)
        self.arm, self.scales, self.theta = arm, (float(scales[0]), float(scales[1])), float(theta)
        self.rand_gates = rand_gates or {}
        self.context = None
        self.pred_shared = None
        self.buffer: list = []
        self.mechanism_events: list = []

    # platform protocol -------------------------------------------------------------------------
    def clone(self):
        out = RSystem([clone_model(m) for m in self.shared], [clone_model(m) for m in self.private],
                      self.arm, scales=self.scales, theta=self.theta, rand_gates=self.rand_gates)
        out.context = self.context
        out.pred_shared = None if self.pred_shared is None else list(self.pred_shared)
        out.buffer = [b.copy() for b in self.buffer]
        out.mechanism_events = list(self.mechanism_events)
        return out

    def _buffer_digest(self) -> str:
        h = hashlib.sha256()
        for b in self.buffer:
            h.update(np.packbits(b).tobytes())
        return h.hexdigest()

    def digests(self):
        return ([m.state_digest() for m in self.shared] + [m.state_digest() for m in self.private]
                + [self._buffer_digest()])

    def set_record_context(self, world, branch, index, stage, domain):
        self.context = (int(world), branch, int(index), stage, domain)
        self.pred_shared = None

    def feed(self, b, t):
        for m in self.shared + self.private:
            m.byte(b, t, learn=False)

    def values(self, t):
        before = self.digests()
        s = [float(m.association_value(t)) for m in self.shared]
        p = [float(m.association_value(t)) for m in self.private]
        u = [s[j] / self.scales[0] + p[j] / self.scales[1] for j in range(4)]
        if any(not math.isfinite(v) for v in u) or self.digests() != before:
            raise AssertionError("nonfinite or mutating R value read")
        if self.context is not None and self.pred_shared is None:
            self.pred_shared = s      # cached at cue completion, before the answer byte
        return u

    def flush(self, t):
        return max(float(m.flush(t)) for m in self.shared + self.private)

    # teaching ----------------------------------------------------------------------------------
    def teach(self, answer: int, t: float, domain: str, *, write: bool):
        _check_teacher(answer, domain)
        if self.context is None or self.pred_shared is None or self.context[4] != domain:
            raise AssertionError("R teach lacks context or cached pre-answer shared values")
        world, branch, index, stage, _ = self.context
        n = _active(domain)
        y_idx = CHANNELS.index(answer)
        for m in self.shared + self.private:
            m.byte(answer, t, learn=False)
        # pending_x now holds the KC code of the completed cue, captured before the answer byte entered FE0.
        x = self.private[0].pending_x
        if x is None or any(not np.array_equal(m.pending_x, x) for m in self.private[1:]):
            raise AssertionError("private pre-answer codes missing or differ across stores")
        xb = np.asarray(x) > 0
        novelty = _jaccard_novelty(xb, self.buffer)
        for j, m in enumerate(self.shared):
            did = bool(write and j < n)
            a = m.teach_logged(int(CHANNELS[j] != answer), t, write=did)
            self.mechanism_events.append(["w", index, BANK_SHARED, j, int(did), round(a, 12)])
        gate = True
        coeff = None
        if self.arm == "R1":
            gate = novelty > self.theta
        elif self.arm == "R1_rand":
            gate = bool(self.rand_gates[branch][index])
        if self.arm in SIGNED:
            v = np.asarray(self.pred_shared[:n], dtype=np.float64) / self.scales[0]
            e = np.exp(v - v.max())
            p = e / e.sum()
            y = np.zeros(n)
            y[y_idx] = 1.0
            if self.arm == "R0_signed":
                b, s = 1.0, 1.0 - y
            else:
                b, s = 0.0, p - y
                if self.arm == "R3_randtarget":
                    s = s[derangement(n, f"A3-R3RT-v1|{world}|{branch}|{index}")]
            coeff = [float(b)] + [float(z) for z in s]
            for j, m in enumerate(self.private):
                did = bool(write and j < n)
                sj = float(s[j]) if j < n else 0.0
                a = m.teach_signed(b, sj, t, write=did)
                self.mechanism_events.append(["s", index, BANK_PRIVATE, j, int(did), round(a, 12),
                                              float(b), sj])
        else:
            for j, m in enumerate(self.private):
                did = bool(write and gate and j < n)
                a = m.teach_logged(int(CHANNELS[j] != answer), t, write=did)
                self.mechanism_events.append(["w", index, BANK_PRIVATE, j, int(did), round(a, 12)])
        self.mechanism_events.append(["R", index, round(novelty, 12), int(gate), int(bool(write)),
                                      [float(z) for z in self.pred_shared], coeff])
        self.buffer.append(xb)
        if len(self.buffer) > R_WINDOW:
            self.buffer.pop(0)
        self.pred_shared = None


def r1_random_schedule(world_doc: dict, branch: str, r1_rows: list) -> dict[int, bool]:
    """R1_rand gates: R1's realised private-write count per (stage, domain, answer) stratum, SHA256 placement.

    Reads only R1's committed per-record gate and permission flags for this branch.
    """
    world = world_doc["world"]
    rows = {r[1]: r for r in r1_rows if r[0] == "R"}
    if sorted(rows) != list(range(len(world_doc["records"]))):
        raise AssertionError("incomplete R1 gate ledger")
    eligible, counts = {}, {}
    for rec in world_doc["records"]:
        i = rec["index"]
        _, _, _, gate, permitted, _, _ = rows[i]
        if permitted:
            key = (rec["stage"], rec["domain"], rec["answer"])
            eligible.setdefault(key, []).append(i)
            counts[key] = counts.get(key, 0) + int(gate)
    chosen = set()
    for (stage, domain, answer), idx in eligible.items():
        def rank(i, stage=stage, domain=domain, answer=answer):
            return hashlib.sha256(f"A3-R1RAND-v1|{world}|{branch}|{stage}|{domain}|{answer}|{i}".encode()).digest()
        chosen.update(sorted(idx, key=rank)[:counts[(stage, domain, answer)]])
    return {rec["index"]: rec["index"] in chosen for rec in world_doc["records"]}
