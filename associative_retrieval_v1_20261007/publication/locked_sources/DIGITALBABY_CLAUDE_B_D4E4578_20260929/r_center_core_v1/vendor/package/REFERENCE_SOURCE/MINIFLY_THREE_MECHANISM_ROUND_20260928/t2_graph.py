"""Label-blind T2/T0_2 PN-to-KC graph construction; no learner or science runs.

See t2_spec.md. Call build_pair() on the verified instantiated Full151 B and
its pn_type_index before the first byte of a world. The returned CSR matrices
are read-only and must be installed into all four output stores of each arm.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

VERSION = "T2-GRAPH-v2"
FULL151_B_DIGEST = "32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964"
P, INPUT_TYPES, K, E = 302, 88, 5177, 27572
MIX_ACCEPTED = 8 * E
OPT_PROPOSALS = 16 * E
MAX_ATTEMPTS = 80 * E
AUDIT_SIZES = (4, 8, 16, 32)
AUDIT_DRAWS = 256


class GraphError(RuntimeError):
    """A graph failed technical instantiation or an independent invariant."""


def _u16(n: int) -> bytes:
    return int(n).to_bytes(2, "big", signed=False)


def _stream_key(world: int, domain: str) -> bytes:
    return hashlib.sha256(f"T2-v2|{world}|{domain}".encode("utf-8")).digest()


class CounterRng:
    """SHA256 counter stream with unbiased bounded draws."""

    def __init__(self, world: int, domain: str):
        self.key = _stream_key(world, domain)
        self.counter = 0

    def randbelow(self, n: int) -> int:
        if not 0 < n <= 2**64:
            raise ValueError("invalid RNG bound")
        cutoff = (2**64 // n) * n
        while True:
            if self.counter >= 2**64:
                raise GraphError("counter stream exhausted")
            raw = hashlib.sha256(self.key + self.counter.to_bytes(8, "big")).digest()
            self.counter += 1
            value = int.from_bytes(raw[:8], "big")
            if value < cutoff:
                return value % n


def legacy_sparse_digest(b: csr_matrix) -> str:
    """V88 sparse_digest format, including JSON float conversion."""
    b = b.tocsr()
    doc = {"shape": list(b.shape), "data": b.data.tolist(),
           "indices": b.indices.tolist(), "indptr": b.indptr.tolist()}
    raw = json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def graph_digest(b: csr_matrix) -> str:
    """Canonical byte digest including shape and array dtypes."""
    h = hashlib.sha256()
    h.update(VERSION.encode())
    h.update(np.asarray(b.shape, dtype="<i8").tobytes())
    for a in (b.indptr, b.indices, b.data):
        h.update(str(a.dtype).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _source(b_native: csr_matrix, pn_type_index: np.ndarray,
            expected_digest: str) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    b = b_native.tocsr(copy=False)
    if b.shape != (P, K) or b.nnz != E:
        raise GraphError("unexpected Full151 B shape or edge count")
    # The archived parent CSR is unsorted, though it has no duplicate edges.
    # Hash its original bytes; canonicalize only the new arm graphs.
    if any(len(set(map(int, b.indices[b.indptr[p]:b.indptr[p + 1]])))
           != int(b.indptr[p + 1] - b.indptr[p]) for p in range(P)):
        raise GraphError("duplicate PN-KC edge in Full151 source")
    if b.data.dtype != np.float64 or not np.isfinite(b.data).all() or (b.data <= 0).any():
        raise GraphError("Full151 B weights must be finite positive float64")
    if legacy_sparse_digest(b) != expected_digest:
        raise GraphError("Full151 B source digest mismatch")
    types = np.asarray(pn_type_index)
    if types.shape != (P,) or not np.issubdtype(types.dtype, np.integer):
        raise GraphError("PN type map shape/dtype mismatch")
    if not np.array_equal(np.unique(types), np.arange(88)):
        raise GraphError("expected 88 PN input types")
    col_degrees = np.diff(b.tocsc().indptr).astype(np.int32)
    if col_degrees.max() > 17 or int((col_degrees == 0).sum()) != 323:
        raise GraphError("Full151 KC degree source mismatch")
    return b, types.astype(np.int16, copy=True), col_degrees


def _row_degrees() -> np.ndarray:
    degrees = np.full(P, E // P, dtype=np.int32)
    ranked = sorted(range(P),
                    key=lambda p: (hashlib.sha256(b"T2-v2|degree|" + _u16(p)).digest(), p))
    for p in ranked[:E % P]:
        degrees[p] += 1
    return degrees


def _havel_hakimi(world: int, type_degrees: np.ndarray,
                  col_degrees: np.ndarray) -> list[tuple[int, int]]:
    key = _stream_key(world, "hh")
    order = sorted(range(K),
                   key=lambda k: (-int(col_degrees[k]),
                                  hashlib.sha256(key + _u16(k)).digest(), k))
    remaining = type_degrees.copy()
    edges: list[tuple[int, int]] = []
    for k in order:
        size = int(col_degrees[k])
        if size == 0:
            continue
        channels = sorted(range(INPUT_TYPES),
                          key=lambda a: (-int(remaining[a]),
                                         hashlib.sha256(key + _u16(k) + _u16(a)).digest(), a))
        selected = channels[:size]
        if len(selected) != size or any(remaining[a] <= 0 for a in selected):
            raise GraphError("bipartite degree realization failed")
        edges.extend((a, k) for a in selected)
        for a in selected:
            remaining[a] -= 1
    if len(edges) != E or np.any(remaining != 0) or len(set(edges)) != E:
        raise GraphError("bipartite degree realization incomplete")
    edges.sort()
    return edges


class _Graph:
    def __init__(self, edges: list[tuple[int, int]], n_left: int,
                 *, expected_edges: int = E):
        self.edges = edges.copy()
        self.n_left = n_left
        self.edge_set = set(edges)
        self.kc_neighbors = [set() for _ in range(K)]
        for left, k in edges:
            if not (0 <= left < n_left and 0 <= k < K):
                raise GraphError("graph edge out of bounds")
            self.kc_neighbors[k].add(left)
        if len(self.edge_set) != expected_edges:
            raise GraphError("parallel edge in graph")

    def proposed(self, i: int, j: int) -> tuple[int, int, int, int] | None:
        if i == j:
            return None
        p, k = self.edges[i]
        q, l = self.edges[j]
        if p == q or k == l or (p, l) in self.edge_set or (q, k) in self.edge_set:
            return None
        return p, k, q, l

    def switch(self, i: int, j: int, proposal: tuple[int, int, int, int]) -> None:
        p, k, q, l = proposal
        self.edge_set.remove((p, k))
        self.edge_set.remove((q, l))
        self.edge_set.add((p, l))
        self.edge_set.add((q, k))
        self.edges[i] = (p, l)
        self.edges[j] = (q, k)
        self.kc_neighbors[k].remove(p)
        self.kc_neighbors[k].add(q)
        self.kc_neighbors[l].remove(q)
        self.kc_neighbors[l].add(p)


def _draw_proposal(graph: _Graph, rng: CounterRng) -> tuple[int, int, tuple[int, int, int, int] | None]:
    i = rng.randbelow(E)
    j = rng.randbelow(E)
    return i, j, graph.proposed(i, j)


def _mix(world: int, initial: list[tuple[int, int]]) -> tuple[_Graph, int]:
    graph = _Graph(initial, INPUT_TYPES)
    rng = CounterRng(world, "mix")
    accepted = 0
    attempts = 0
    while accepted < MIX_ACCEPTED:
        attempts += 1
        if attempts > MAX_ATTEMPTS:
            raise GraphError("common graph mixing cap exceeded")
        i, j, proposal = _draw_proposal(graph, rng)
        if proposal is None:
            continue
        graph.switch(i, j, proposal)
        accepted += 1
    return graph, attempts


def _pair_counts(graph: _Graph) -> np.ndarray:
    counts = np.zeros((graph.n_left, graph.n_left), dtype=np.int32)
    for members in graph.kc_neighbors:
        ordered = sorted(members)
        for i, p in enumerate(ordered):
            for q in ordered[i + 1:]:
                counts[p, q] += 1
    return counts


def _jscore(counts: np.ndarray) -> int:
    return int(np.sum(counts * (counts - 1) // 2, dtype=np.int64))


def _pair_delta(graph: _Graph, proposal: tuple[int, int, int, int]) -> dict[tuple[int, int], int]:
    p, k, q, l = proposal
    delta: dict[tuple[int, int], int] = {}

    def add(a: int, b: int, value: int) -> None:
        pair = (a, b) if a < b else (b, a)
        delta[pair] = delta.get(pair, 0) + value

    for other in graph.kc_neighbors[k]:
        if other != p:
            add(p, other, -1)
            add(q, other, +1)
    for other in graph.kc_neighbors[l]:
        if other != q:
            add(q, other, -1)
            add(p, other, +1)
    return {pair: value for pair, value in delta.items() if value}


def _optimize(world: int, common: _Graph) -> tuple[_Graph, dict]:
    graph = _Graph(common.edges, INPUT_TYPES)
    counts = _pair_counts(graph)
    initial_j = _jscore(counts)
    score = initial_j
    rng = CounterRng(world, "opt")
    valid = 0
    accepted = 0
    for _ in range(OPT_PROPOSALS):
        i, j, proposal = _draw_proposal(graph, rng)
        if proposal is None:
            continue
        valid += 1
        delta = _pair_delta(graph, proposal)
        score_delta = 0
        for (a, b), change in delta.items():
            before = int(counts[a, b])
            after = before + change
            if after < 0:
                raise GraphError("negative pair count")
            score_delta += after * (after - 1) // 2 - before * (before - 1) // 2
        if score_delta >= 0:
            continue
        for (a, b), change in delta.items():
            counts[a, b] += change
        graph.switch(i, j, proposal)
        score += score_delta
        accepted += 1
    if accepted == 0 or score >= initial_j or score != _jscore(_pair_counts(graph)):
        raise GraphError("T2 graph failed strict independent objective audit")
    return graph, {"initial_j": initial_j, "optimized_j": score,
                   "valid_proposals": valid, "accepted_switches": accepted,
                   "proposal_attempts": OPT_PROPOSALS}


def _null(world: int, common: _Graph, target_switches: int) -> tuple[_Graph, int]:
    graph = _Graph(common.edges, INPUT_TYPES)
    rng = CounterRng(world, "null")
    accepted = 0
    attempts = 0
    while accepted < target_switches:
        attempts += 1
        if attempts > MAX_ATTEMPTS:
            raise GraphError("T0 random switch cap exceeded")
        i, j, proposal = _draw_proposal(graph, rng)
        if proposal is None:
            continue
        graph.switch(i, j, proposal)
        accepted += 1
    return graph, attempts


def _assign_type_edges_to_pn(type_graph: _Graph, types: np.ndarray,
                             degrees: np.ndarray, world: int) -> _Graph:
    """Allocate each effective input-type edge to one PN of that type."""
    key = _stream_key(world, "assign")
    pn_edges: list[tuple[int, int]] = []
    for channel in range(INPUT_TYPES):
        kcs = sorted((k for k in range(K) if channel in type_graph.kc_neighbors[k]),
                     key=lambda k: (hashlib.sha256(key + _u16(channel) + _u16(k)).digest(), k))
        stubs = [(p, slot) for p in range(P) if int(types[p]) == channel
                 for slot in range(int(degrees[p]))]
        stubs.sort(key=lambda item: (hashlib.sha256(
            key + _u16(channel) + _u16(item[0]) + _u16(item[1])).digest(), item))
        if len(stubs) != len(kcs):
            raise GraphError("input-type degree cannot be allocated to PN rows")
        pn_edges.extend((p, k) for (p, _), k in zip(stubs, kcs, strict=True))
    pn_edges.sort()
    graph = _Graph(pn_edges, P)
    if len(graph.edge_set) != E:
        raise GraphError("PN assignment created parallel edges")
    return graph


def _weighted_csr(graph: _Graph, native: csr_matrix, world: int) -> csr_matrix:
    source = native.tocsc()
    key = _stream_key(world, "weight")
    weighted: list[tuple[int, int, float]] = []
    for k, members in enumerate(graph.kc_neighbors):
        lo, hi = int(source.indptr[k]), int(source.indptr[k + 1])
        weights = sorted(float(v) for v in source.data[lo:hi])
        if len(weights) != len(members):
            raise GraphError("KC weight multiset size mismatch")
        rows = sorted(members, key=lambda p: (hashlib.sha256(key + _u16(k) + _u16(p)).digest(), p))
        weighted.extend((p, k, weight) for p, weight in zip(rows, weights, strict=True))
    weighted.sort(key=lambda row: (row[0], row[1]))
    rows = np.fromiter((r[0] for r in weighted), dtype=np.int32, count=E)
    cols = np.fromiter((r[1] for r in weighted), dtype=np.int32, count=E)
    data = np.fromiter((r[2] for r in weighted), dtype=np.float64, count=E)
    b = csr_matrix((data, (rows, cols)), shape=(P, K))
    b.sort_indices()
    if b.nnz != E or not b.has_canonical_format or not np.array_equal(b.data, data):
        raise GraphError("weighted CSR construction changed edge/weight ordering")
    for array in (b.data, b.indices, b.indptr):
        array.flags.writeable = False
    return b


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo = math.floor(position)
    hi = math.ceil(position)
    return ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])


def _sample_rows(rng: CounterRng, size: int, total: int) -> list[int]:
    rows = list(range(total))
    for j in range(size):
        other = j + rng.randbelow(total - j)
        rows[j], rows[other] = rows[other], rows[j]
    return rows[:size]


def _row_neighborhoods(graph: _Graph) -> list[set[int]]:
    rows = [set() for _ in range(graph.n_left)]
    for left, k in graph.edge_set:
        rows[left].add(k)
    return rows


def _expansion(rows: list[set[int]], pn_rows: list[int], row_degrees: np.ndarray) -> float:
    denominator = int(sum(int(row_degrees[p]) for p in pn_rows))
    return len(set().union(*(rows[p] for p in pn_rows))) / denominator


def _audit(world: int, candidate: _Graph, control: _Graph,
           common: _Graph, native: _Graph,
           candidate_types: _Graph, control_types: _Graph, common_types: _Graph,
           types: np.ndarray, degrees: np.ndarray, native_degrees: np.ndarray) -> dict:
    rng = CounterRng(world, "audit")
    tc = _row_neighborhoods(candidate)
    tn = _row_neighborhoods(control)
    tm = _row_neighborhoods(common)
    tb = _row_neighborhoods(native)
    type_rows = {"T2": _row_neighborhoods(candidate_types),
                 "T0_2": _row_neighborhoods(control_types),
                 "G_star": _row_neighborhoods(common_types),
                 "native": [set() for _ in range(INPUT_TYPES)]}
    for p, neighbors in enumerate(tb):
        type_rows["native"][int(types[p])].update(neighbors)
    type_degrees = np.bincount(types, weights=degrees, minlength=INPUT_TYPES).astype(np.int32)
    native_type_degrees = np.asarray([len(x) for x in type_rows["native"]], dtype=np.int32)
    for name in ("T2", "T0_2", "G_star"):
        if not np.array_equal(np.asarray([len(x) for x in type_rows[name]]), type_degrees):
            raise GraphError("input-type degree budget changed")
    row_stats = {}
    type_stats = {}
    for size in AUDIT_SIZES:
        values = {"T2": [], "T0_2": [], "G_star": [], "native": []}
        for _ in range(AUDIT_DRAWS):
            subset = _sample_rows(rng, size, P)
            for name, rows in (("T2", tc), ("T0_2", tn), ("G_star", tm)):
                values[name].append(_expansion(rows, subset, degrees))
            values["native"].append(_expansion(tb, subset, native_degrees))
        row_stats[str(size)] = {name: {"mean": float(np.mean(v)),
                                       "p05": _percentile(v, .05)}
                                for name, v in values.items()}
        type_values = {"T2": [], "T0_2": [], "G_star": [], "native": []}
        for _ in range(AUDIT_DRAWS):
            subset = _sample_rows(rng, size, INPUT_TYPES)
            for name in ("T2", "T0_2", "G_star"):
                type_values[name].append(_expansion(type_rows[name], subset, type_degrees))
            type_values["native"].append(
                _expansion(type_rows["native"], subset, native_type_degrees))
        type_stats[str(size)] = {name: {"mean": float(np.mean(v)),
                                            "p05": _percentile(v, .05)}
                                 for name, v in type_values.items()}
    pair_stats = {}
    for name, graph in (("T2", candidate), ("T0_2", control),
                        ("G_star", common), ("native", native)):
        counts = _pair_counts(graph)
        upper = counts[np.triu_indices(graph.n_left, k=1)].astype(np.int32)
        hist = np.bincount(upper)
        pair_stats[name] = {"J": _jscore(counts), "max_shared_kc": int(upper.max()),
                            "overlap_histogram": hist.tolist()}
    type_pair_stats = {}
    for name, graph in (("T2", candidate_types), ("T0_2", control_types),
                        ("G_star", common_types)):
        counts = _pair_counts(graph)
        upper = counts[np.triu_indices(INPUT_TYPES, k=1)].astype(np.int32)
        type_pair_stats[name] = {"J": _jscore(counts),
                                 "max_shared_kc": int(upper.max()),
                                 "overlap_histogram": np.bincount(upper).tolist()}
    lower_tail_gain = float(np.mean([
        type_stats[str(s)]["T2"]["p05"] - type_stats[str(s)]["T0_2"]["p05"]
        for s in AUDIT_SIZES]))
    row_lower_tail_gain = float(np.mean([
        row_stats[str(s)]["T2"]["p05"] - row_stats[str(s)]["T0_2"]["p05"]
        for s in AUDIT_SIZES]))
    return {"row_expansion": row_stats, "type_expansion": type_stats,
            "pair_overlap": pair_stats, "type_pair_overlap": type_pair_stats,
            "lower_tail_gain": lower_tail_gain,
            "row_lower_tail_gain": row_lower_tail_gain,
            "symmetric_difference_edges": len(candidate.edge_set ^ control.edge_set),
            "type_symmetric_difference_edges": len(candidate_types.edge_set ^ control_types.edge_set)}


def _verify_graph(native: csr_matrix, graph: _Graph, b: csr_matrix,
                  row_degrees: np.ndarray, col_degrees: np.ndarray,
                  types: np.ndarray, type_degrees: np.ndarray) -> None:
    if len(graph.edge_set) != E or b.shape != (P, K) or b.nnz != E or not b.has_canonical_format:
        raise GraphError("graph shape/support mismatch")
    if not np.array_equal(np.diff(b.indptr), row_degrees):
        raise GraphError("PN degree budget mismatch")
    native_csc = native.tocsc()
    new_csc = b.tocsc()
    if not np.array_equal(np.diff(new_csc.indptr), col_degrees):
        raise GraphError("KC degree budget mismatch")
    for k in range(K):
        a, c = int(native_csc.indptr[k]), int(native_csc.indptr[k + 1])
        x, y = int(new_csc.indptr[k]), int(new_csc.indptr[k + 1])
        if not np.array_equal(np.sort(native_csc.data[a:c]), np.sort(new_csc.data[x:y])):
            raise GraphError(f"KC {k} weight multiset mismatch")
    rebuilt = {(int(p), int(k)) for p in range(P)
               for k in b.indices[b.indptr[p]:b.indptr[p + 1]]}
    if rebuilt != graph.edge_set:
        raise GraphError("CSR indices differ from generated adjacency")
    type_neighbors = [set() for _ in range(INPUT_TYPES)]
    for k in range(K):
        members = graph.kc_neighbors[k]
        seen = {int(types[p]) for p in members}
        if len(seen) != len(members):
            raise GraphError("duplicate effective input type in one KC")
        for channel in seen:
            type_neighbors[channel].add(k)
    if not np.array_equal(np.asarray([len(x) for x in type_neighbors]), type_degrees):
        raise GraphError("effective input-type degrees changed")


@dataclass(frozen=True)
class GraphPair:
    t2: csr_matrix
    t0_2: csr_matrix
    receipt: dict


def build_pair(b_native: csr_matrix, pn_type_index: np.ndarray, world: int,
               *, expected_digest: str = FULL151_B_DIGEST) -> GraphPair:
    """Build and independently audit the paired graphs for one world.

    ``expected_digest`` defaults to the frozen Full151 B. A different digest is
    allowed only for graph-only fixtures and is always recorded in the receipt.
    This function reads no task fixture, labels, learner state or scientific result.
    """
    if not isinstance(world, int) or world < 0:
        raise ValueError("world must be a nonnegative integer")
    native, types, col_degrees = _source(b_native, pn_type_index, expected_digest)
    degrees = _row_degrees()
    type_degrees = np.bincount(types, weights=degrees, minlength=INPUT_TYPES).astype(np.int32)
    initial = _havel_hakimi(world, type_degrees, col_degrees)
    native_edges = [(p, int(k)) for p in range(P)
                    for k in native.indices[native.indptr[p]:native.indptr[p + 1]]]
    native_graph = _Graph(native_edges, P)
    common_types, mix_attempts = _mix(world, initial)
    t2_types, opt = _optimize(world, common_types)
    t0_types, null_attempts = _null(world, common_types, opt["accepted_switches"])
    common = _assign_type_edges_to_pn(common_types, types, degrees, world)
    t2_graph = _assign_type_edges_to_pn(t2_types, types, degrees, world)
    t0_graph = _assign_type_edges_to_pn(t0_types, types, degrees, world)
    t2 = _weighted_csr(t2_graph, native, world)
    t0 = _weighted_csr(t0_graph, native, world)
    _verify_graph(native, t2_graph, t2, degrees, col_degrees, types, type_degrees)
    _verify_graph(native, t0_graph, t0, degrees, col_degrees, types, type_degrees)
    audit = _audit(world, t2_graph, t0_graph, common, native_graph,
                   t2_types, t0_types, common_types, types, degrees,
                   np.diff(native.indptr).astype(np.int32))
    if audit["type_pair_overlap"]["T2"]["J"] != opt["optimized_j"]:
        raise GraphError("independent T2 objective disagrees with optimizer")
    if audit["type_symmetric_difference_edges"] == 0 or audit["symmetric_difference_edges"] == 0:
        raise GraphError("T2 and T0 have identical adjacency")
    receipt = {"schema": VERSION, "world": world,
               "source_digest": expected_digest,
               "shape": [P, K], "edges": E,
               "pn_degree_cap": 92,
               "pn_degree_92_rows": np.flatnonzero(degrees == 92).tolist(),
               "input_type_degrees": type_degrees.tolist(),
               "kc_zero_degree_count": int((col_degrees == 0).sum()),
               "mix_accepted": MIX_ACCEPTED, "mix_attempts": mix_attempts,
               "optimization": opt,
               "null_accepted": opt["accepted_switches"],
               "null_attempts": null_attempts,
               "seed_sha256": {domain: hashlib.sha256(_stream_key(world, domain)).hexdigest()
                               for domain in ("hh", "mix", "opt", "null", "assign", "weight", "audit")},
               "graph_digest": {"T2": graph_digest(t2), "T0_2": graph_digest(t0)},
               "audit": audit}
    return GraphPair(t2=t2, t0_2=t0, receipt=receipt)


def qualify_technical_roster(receipts: list[dict], expected_worlds: tuple[int, ...]) -> dict:
    """Label-blind source-lock gate for an entire reserved technical roster."""
    if not expected_worlds or len(set(expected_worlds)) != len(expected_worlds):
        raise GraphError("invalid technical world roster")
    by_world = {}
    for receipt in receipts:
        if receipt.get("schema") != VERSION:
            raise GraphError("wrong graph receipt version")
        world = receipt.get("world")
        if world in by_world:
            raise GraphError("duplicate technical graph world")
        by_world[world] = receipt
    if set(by_world) != set(expected_worlds):
        raise GraphError("technical graph roster incomplete or contains unreserved world")
    gains = []
    for world in expected_worlds:
        receipt = by_world[world]
        if receipt.get("source_digest") != FULL151_B_DIGEST:
            raise GraphError("technical roster did not use instantiated Full151 B")
        opt = receipt["optimization"]
        if opt["optimized_j"] >= opt["initial_j"] or opt["accepted_switches"] <= 0:
            raise GraphError("T2 objective failed in technical roster")
        if receipt["null_accepted"] != opt["accepted_switches"]:
            raise GraphError("T0 switch budget mismatch in technical roster")
        audit = receipt["audit"]
        if audit["type_pair_overlap"]["T2"]["J"] != opt["optimized_j"]:
            raise GraphError("independent effective-type objective mismatch")
        gain = float(audit["lower_tail_gain"])
        if not math.isfinite(gain):
            raise GraphError("nonfinite effective-type expansion gain")
        recomputed = float(np.mean([
            audit["type_expansion"][str(size)]["T2"]["p05"] -
            audit["type_expansion"][str(size)]["T0_2"]["p05"]
            for size in AUDIT_SIZES]))
        if abs(gain - recomputed) > 1e-12:
            raise GraphError("effective input-type lower-tail audit mismatch")
        gains.append(gain)
    median = float(np.median(gains))
    if median <= 0:
        raise GraphError("effective input-type lower-tail expansion did not improve")
    return {"schema": "T2-GRAPH-TECHNICAL-ROSTER-v2",
            "worlds": list(expected_worlds), "n": len(expected_worlds),
            "effective_type_lower_tail_gains": gains,
            "median_effective_type_lower_tail_gain": median}
