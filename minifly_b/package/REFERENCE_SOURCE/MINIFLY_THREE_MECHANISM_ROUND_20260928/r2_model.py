"""R2 error-gated dual memory and matched R0 control; FE1 is absent."""
from __future__ import annotations

import copy
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path

from common_platform import FourStore, CHANNELS, clone_model
from common_platform import branch_allows

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "FULL151_VISIBLE_CONTEXT_BRIDGE_20260927"))
import content_model  # noqa: E402


def gate_uniform(world: int, domain: str, branch: str, index: int) -> float:
    key = f"R2-GATE-v1|{world}|{domain}|{branch}|{index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64


class DualStore:
    """Four FE0 shared stores plus four CONTENT private stores in one life."""

    def __init__(self, shared: FourStore, private: FourStore, *, arm: str,
                 scales: tuple[float, float],
                 rand_gates: dict[str, dict[int, bool]] | None = None):
        if arm not in ("R0", "R2", "Rrand"):
            raise ValueError("unknown R arm")
        if len(shared.stores) != 4 or len(private.stores) != 4:
            raise ValueError("R needs two complete four-store banks")
        if not all(math.isfinite(s) and s > 0 for s in scales):
            raise ValueError("nonpositive calibration scale")
        self.shared, self.private = shared, private
        self.arm, self.scales = arm, tuple(float(x) for x in scales)
        self.rand_gates = {} if rand_gates is None else {
            branch: dict(gates) for branch, gates in rand_gates.items()}
        self.context = None
        self.pred_u = None
        self.mechanism_events = []

    @classmethod
    def from_native(cls, base, *, arm: str, scales: tuple[float, float],
                    rand_gates: dict[str, dict[int, bool]] | None = None):
        shared = FourStore([clone_model(base) for _ in range(4)])
        private = FourStore([content_model.from_native(base) for _ in range(4)])
        if any(not isinstance(m.fe, content_model.VisibleContextFE) for m in private.stores):
            raise AssertionError("CONTENT private route missing")
        return cls(shared, private, arm=arm, scales=scales, rand_gates=rand_gates)

    def clone(self):
        out = DualStore(self.shared.clone(), self.private.clone(), arm=self.arm,
                        scales=self.scales, rand_gates=self.rand_gates)
        out.context = self.context
        out.pred_u = None if self.pred_u is None else list(self.pred_u)
        out.mechanism_events = list(self.mechanism_events)
        return out

    def digests(self):
        return self.shared.digests() + self.private.digests()

    def set_record_context(self, world: int, branch: str, index: int, stage: str, domain: str):
        self.context = (int(world), branch, int(index), stage, domain)
        self.pred_u = None

    def feed(self, b: int, t: float):
        self.shared.feed(b, t)
        self.private.feed(b, t)

    def raw_values(self, t: float):
        return self.shared.values(t), self.private.values(t)

    def values(self, t: float):
        s, p = self.raw_values(t)
        u = [s[j] / self.scales[0] + p[j] / self.scales[1] for j in range(4)]
        if any(not math.isfinite(x) for x in u):
            raise AssertionError("nonfinite combined output")
        self.pred_u = u
        return u

    def teach(self, answer: int, t: float, domain: str, *, write: bool):
        if self.context is None or self.pred_u is None:
            raise AssertionError("missing pre-feedback prediction/context")
        world, branch, index, stage, declared = self.context
        if domain != declared or answer not in CHANNELS:
            raise AssertionError("teacher/domain mismatch")
        active = range(4 if domain == "fact" else 2)
        y = CHANNELS.index(answer)
        if y not in active:
            raise AssertionError("teacher outside active output")
        m = max(self.pred_u[j] for j in active)
        weights = [math.exp(self.pred_u[j] - m) for j in active]
        e = 1.0 - weights[y] / sum(weights)
        u = gate_uniform(world, domain, branch, index)
        if self.arm == "R0":
            gate = True
        elif self.arm == "R2":
            gate = u < e
        else:
            if branch not in self.rand_gates or index not in self.rand_gates[branch]:
                raise AssertionError("missing prespecified Rrand gate")
            gate = bool(self.rand_gates[branch][index])
        self.shared.teach(answer, t, domain, write=write)
        self.private.teach(answer, t, domain, write=write and gate)
        self.mechanism_events.append({"record": index, "stage": stage, "domain": domain,
                                      "branch": branch, "answer": answer,
                                      "allowed": bool(write), "gate": bool(gate),
                                      "error": e, "uniform": u, "prefeedback_u": list(self.pred_u)})
        self.pred_u = None

    def flush(self, t: float):
        return max(self.shared.flush(t), self.private.flush(t))


def random_gate_schedule(world_doc: dict, branch: str, r2_events: list[dict]) -> dict[int, bool]:
    """Count-matched event placement, stratified by phase/domain/arrived answer.

    Only R2's per-stratum approved *counts* are carried over, never its
    predictions, correctness, or chosen event positions.
    """
    world = world_doc["world"]
    if len(r2_events) != len(world_doc["records"]):
        raise AssertionError("incomplete R2 event ledger")
    eligible = defaultdict(list)
    counts = defaultdict(int)
    for row, event in zip(world_doc["records"], r2_events, strict=True):
        index = row["index"]
        if (event["record"] != index or event["branch"] != branch or
                event["answer"] != row["answer"] or event["domain"] != row["domain"]):
            raise AssertionError("R2 gate/fixture mismatch")
        allowed = branch_allows(branch, row["stage"], row["domain"])
        if event["allowed"] != allowed:
            raise AssertionError("R2 causal write permission mismatch")
        if allowed:
            key = (row["stage"], row["domain"], row["answer"])
            eligible[key].append(index)
            counts[key] += int(event["gate"])
    selected = set()
    for (stage, domain, answer), indices in eligible.items():
        def rank(index: int):
            raw = f"R2-RAND-v1|{world}|{branch}|{stage}|{domain}|{answer}|{index}".encode()
            return hashlib.sha256(raw).digest(), index
        selected.update(sorted(indices, key=rank)[:counts[(stage, domain, answer)]])
    if len(selected) != sum(counts.values()):
        raise AssertionError("Rrand count mismatch")
    return {row["index"]: row["index"] in selected for row in world_doc["records"]}
