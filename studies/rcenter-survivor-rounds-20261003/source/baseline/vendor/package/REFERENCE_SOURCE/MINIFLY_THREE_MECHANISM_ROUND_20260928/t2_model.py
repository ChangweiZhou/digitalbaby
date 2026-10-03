"""Local Full151 adapter binding a T2/T0_2 graph to model/checkpoint identity.

This wraps the unchanged FE0 Full151 byte brain. It adds no online learned
state and never changes PN-to-KC adjacency after the first round byte.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from common_platform import bb
from t2_graph import (GraphError, GraphPair, VERSION, graph_digest,
                      legacy_sparse_digest)

ARMS = ("T2", "T0_2")


def _clone_mutable(model):
    out = copy.copy(model)
    out.fly = model.fly.clone()
    out.fe = model.fe.clone()
    out.w = model.w.copy()
    out.bias = model.bias.copy()
    out.pending_x = None if model.pending_x is None else model.pending_x.copy()
    return out


class GraphBrain(bb.F151ByteBrain):
    """Full151 FE0 with one read-only world/arm graph in its fixed identity."""

    @classmethod
    def from_base(cls, base: bb.F151ByteBrain, pair: GraphPair,
                  arm: str) -> "GraphBrain":
        if arm not in ARMS:
            raise ValueError("unknown T-family arm")
        if (type(base.fe) is not bb.bc.FE0 or base.brain_t != 0.0 or
                base.fly.m.elapsed != 0.0 or base.pending_x is not None or
                base.bytes_seen != 0 or base.prev1 is not None or base.prev2 is not None):
            raise GraphError("T graph must be installed on a fresh FE0 native state")
        if legacy_sparse_digest(base.fly.m.B) != pair.receipt["source_digest"]:
            raise GraphError("birth source B does not match graph source")
        selected = pair.t2 if arm == "T2" else pair.t0_2
        actual = graph_digest(selected)
        if actual != pair.receipt["graph_digest"][arm]:
            raise GraphError("birth graph digest mismatch")
        for array in (selected.data, selected.indices, selected.indptr):
            if array.flags.writeable:
                raise GraphError("birth graph must be read-only")
        out = _clone_mutable(base)
        out.__class__ = cls
        out.fly.m.B = selected
        out.graph_world = int(pair.receipt["world"])
        out.graph_arm = arm
        out.graph_source_digest = str(pair.receipt["source_digest"])
        out.installed_graph_digest = actual
        out.graph_version = VERSION
        out._check_graph()
        if type(out.fe) is not bb.bc.FE0 or out.fly.m.elapsed != 0.0:
            raise GraphError("T graph altered newborn FE0 state")
        return out

    def _check_graph(self) -> None:
        b = self.fly.m.B
        if graph_digest(b) != self.installed_graph_digest:
            raise GraphError("PN-KC graph mutated after birth")
        if any(array.flags.writeable for array in (b.data, b.indices, b.indptr)):
            raise GraphError("PN-KC graph lost read-only protection")

    def fixed_digest(self) -> str:
        self._check_graph()
        parent = super().fixed_digest()
        payload = {"graph_version": self.graph_version,
                   "world": self.graph_world, "arm": self.graph_arm,
                   "source_digest": self.graph_source_digest,
                   "graph_digest": self.installed_graph_digest,
                   "parent_fixed_digest": parent}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def snapshot(self) -> dict:
        state = super().snapshot()
        state["t2_graph"] = {"version": self.graph_version,
                             "world": self.graph_world, "arm": self.graph_arm,
                             "source_digest": self.graph_source_digest,
                             "graph_digest": self.installed_graph_digest}
        return state

    def restore(self, state: dict) -> None:
        expected = {"version": self.graph_version,
                    "world": self.graph_world, "arm": self.graph_arm,
                    "source_digest": self.graph_source_digest,
                    "graph_digest": self.installed_graph_digest}
        if state.get("t2_graph") != expected:
            raise GraphError("checkpoint graph identity mismatch")
        self._check_graph()
        super().restore(state)
        self._check_graph()

    def clone_round(self) -> "GraphBrain":
        out = _clone_mutable(self)
        out._check_graph()
        if out.fly.m.B is not self.fly.m.B or out.fe is self.fe:
            raise GraphError("graph clone sharing contract failed")
        if out.state_digest() != self.state_digest():
            raise GraphError("graph clone state digest mismatch")
        return out

    @classmethod
    def load_graph(cls, path: str | Path, fresh_base: bb.F151ByteBrain,
                   pair: GraphPair, arm: str) -> "GraphBrain":
        """Restore a local checkpoint after regenerating/verifying its graph."""
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata_json"].tobytes().decode())
            state = {**metadata, **{key: archive[key].copy() for key in
                                   ("w", "bias", "fly_fast", "fly_slow",
                                    "fly_adapt", "fe_p", "pending_x")}}
        out = cls.from_base(fresh_base, pair, arm)
        out.restore(state)
        return out

    def resources(self) -> dict:
        data = dict(super().resources())
        b = self.fly.m.B
        data.update(t2_graph_version=self.graph_version,
                    t2_graph_world=self.graph_world,
                    t2_graph_arm=self.graph_arm,
                    t2_fixed_graph_bytes=int(b.data.nbytes + b.indices.nbytes + b.indptr.nbytes),
                    t2_added_mutable_bytes=0)
        return data
