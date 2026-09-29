"""Label-blind T1/T3 birth graphs and their matched random controls (SPEC_LOCK.md §T).

Reuses the frozen T2 generator's machinery (Havel-Hakimi realisation, SHA256
counter streams, degree-preserving type-level switches, PN assignment and
per-KC weight-multiset assignment) from REFERENCE_SOURCE without editing it.
Only the objective differs.  Stream keys are ``T2-v2|<family>-v1:<world>|<domain>``
so T1, T3 and the historical T2 never share a random stream.

No fixture, label, byte history or score is read here.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

import paths  # noqa: F401  (sets sys.path to the frozen reference source)
import t2_graph as tg
from t2_graph import (E, INPUT_TYPES, K, MAX_ATTEMPTS, MIX_ACCEPTED, P, CounterRng, GraphError,
                      _Graph, _assign_type_edges_to_pn, _draw_proposal, _havel_hakimi,
                      _row_degrees, _source, _weighted_csr, graph_digest, legacy_sparse_digest)

VERSION = "TFAM-GRAPH-v1"
FAMILIES = ("T1", "T3")
N_TOP, N_LEAF = 4, 8            # T1: 4 top modules x 2 leaves = 8 leaf modules of 11 types
N_CLUSTER = 8                   # T3: 8 flat clusters of 11 types
T1_FLOOR_TOP = int(0.10 * E)    # >= 2757 edges must cross top-level modules
T1_FLOOR_SUB = int(0.15 * E)    # >= 4135 edges must cross leaves inside a top module
T1_PROPOSALS = 16 * E
T3_BRIDGES = int(0.20 * E)      # exactly 5514 cross-cluster bridge edges (random graph: ~87.5%)
T3_MAX_PROPOSALS = 80 * E


def tag(family: str, world: int) -> str:
    if family not in FAMILIES or type(world) is not int:
        raise ValueError("unknown T family or world")
    return f"{family}-v1:{world}"


def _balanced_partition(key: bytes, loads, parts: int, targets=None) -> np.ndarray:
    """Deterministic label-free load-balanced partition.

    Items are visited by descending load (tie: SHA256 rank, then index).  Without
    ``targets`` each item joins the currently lightest part (LPT scheduling).  With
    ``targets`` it joins the part with the largest remaining deficit.  Ties between
    parts go to the lower part index.  Only fixed source-side degrees are used.
    """
    loads = np.asarray(loads, dtype=np.int64)
    order = sorted(range(len(loads)),
                   key=lambda i: (-int(loads[i]), hashlib.sha256(key + tg._u16(i)).digest(), i))
    fill = np.zeros(parts, dtype=np.int64)
    count = np.zeros(parts, dtype=np.int64)
    out = np.empty(len(loads), dtype=np.int16)
    cap = -(-len(loads) // parts)
    for i in order:
        open_parts = [c for c in range(parts) if count[c] < cap]
        if targets is None:
            c = min(open_parts, key=lambda c: (fill[c], c))
        else:
            c = max(open_parts, key=lambda c: (int(targets[c]) - fill[c], -c))
        out[i] = c
        fill[c] += loads[i]
        count[c] += 1
    return out


def modules(family: str, world: int) -> dict:
    """Module/cluster labels of the 88 input types and 5177 KCs (source-side only).

    Type loads are the fixed effective type degrees D_a; KC loads are the fixed
    canonical column degrees c_k.  KCs are balanced against each part's type load.
    """
    key = tg._stream_key(tag(family, world), "modules")
    types = np.asarray(_TYPES_AND_COLS["types"])
    cols = np.asarray(_TYPES_AND_COLS["cols"])
    tdeg = np.bincount(types, weights=_row_degrees(), minlength=INPUT_TYPES).astype(np.int64)
    parts = N_LEAF if family == "T1" else N_CLUSTER
    part_t = _balanced_partition(key + b"type", tdeg, parts)
    load = np.bincount(part_t, weights=tdeg, minlength=parts).astype(np.int64)
    part_k = _balanced_partition(key + b"kc", cols, parts, targets=load)
    if family == "T1":
        return {"leaf_type": part_t, "leaf_kc": part_k,
                "top_type": part_t // (N_LEAF // N_TOP), "top_kc": part_k // (N_LEAF // N_TOP)}
    return {"cluster_type": part_t, "cluster_kc": part_k}


_TYPES_AND_COLS: dict = {}


def edge_cost(family: str, mods: dict, a: int, k: int) -> int:
    """T1: hierarchical distance 0/1/2.  T3: 1 if the edge is a bridge."""
    if family == "T1":
        if mods["leaf_type"][a] == mods["leaf_kc"][k]:
            return 0
        return 1 if mods["top_type"][a] == mods["top_kc"][k] else 2
    return int(mods["cluster_type"][a] != mods["cluster_kc"][k])


def level_counts(family: str, mods: dict, edges) -> dict:
    counts = {0: 0, 1: 0, 2: 0}
    for a, k in edges:
        counts[edge_cost(family, mods, a, k)] += 1
    return counts


def _optimize(family: str, world: int, common: _Graph, mods: dict) -> tuple[_Graph, dict]:
    graph = _Graph(common.edges, INPUT_TYPES)
    counts = level_counts(family, mods, graph.edges)
    initial = dict(counts)
    rng = CounterRng(tag(family, world), "opt")
    valid = accepted = proposals = 0
    budget = T1_PROPOSALS if family == "T1" else T3_MAX_PROPOSALS
    while proposals < budget:
        if family == "T3" and counts[1] == T3_BRIDGES:
            break
        proposals += 1
        i, j, prop = _draw_proposal(graph, rng)
        if prop is None:
            continue
        valid += 1
        a, k, b, l = prop  # (a,k),(b,l) -> (a,l),(b,k)
        old = (edge_cost(family, mods, a, k), edge_cost(family, mods, b, l))
        new = (edge_cost(family, mods, a, l), edge_cost(family, mods, b, k))
        after = dict(counts)
        for c in old:
            after[c] -= 1
        for c in new:
            after[c] += 1
        if family == "T1":
            # Weighted hierarchical cost must strictly fall; cross-level floors hold.
            if sum(new) >= sum(old):
                continue
            if after[2] < T1_FLOOR_TOP or after[1] + after[2] < T1_FLOOR_TOP + T1_FLOOR_SUB:
                continue
        else:
            if not (after[1] <= counts[1] and after[1] >= T3_BRIDGES):
                continue
        graph.switch(i, j, prop)
        counts = after
        accepted += 1
    if counts != level_counts(family, mods, graph.edges):
        raise GraphError(f"{family} objective ledger mismatch")
    if accepted == 0:
        raise GraphError(f"{family}_GRAPH_NOT_INSTANTIATED: no accepted switch")
    if family == "T3" and counts[1] != T3_BRIDGES:
        raise GraphError(f"T3_GRAPH_NOT_INSTANTIATED: bridges {counts[1]} != {T3_BRIDGES}")
    return graph, {"initial_levels": initial, "final_levels": counts, "valid_proposals": valid,
                   "accepted_switches": accepted, "proposal_attempts": proposals}


def _null(family: str, world: int, common: _Graph, target: int) -> tuple[_Graph, int]:
    graph = _Graph(common.edges, INPUT_TYPES)
    rng = CounterRng(tag(family, world), "null")
    accepted = attempts = 0
    while accepted < target:
        attempts += 1
        if attempts > MAX_ATTEMPTS:
            raise GraphError(f"T0_{family[1]} random switch cap exceeded")
        i, j, prop = _draw_proposal(graph, rng)
        if prop is None:
            continue
        graph.switch(i, j, prop)
        accepted += 1
    return graph, attempts


def _mix(family: str, world: int, initial) -> tuple[_Graph, int]:
    graph = _Graph(initial, INPUT_TYPES)
    rng = CounterRng(tag(family, world), "mix")
    accepted = attempts = 0
    while accepted < MIX_ACCEPTED:
        attempts += 1
        if attempts > MAX_ATTEMPTS:
            raise GraphError("common graph mixing cap exceeded")
        i, j, prop = _draw_proposal(graph, rng)
        if prop is None:
            continue
        graph.switch(i, j, prop)
        accepted += 1
    return graph, attempts


def modularity_q(family: str, mods: dict, edges) -> float:
    """Bipartite Barber modularity of the leaf/cluster partition (descriptive)."""
    lt = mods["leaf_type"] if family == "T1" else mods["cluster_type"]
    lk = mods["leaf_kc"] if family == "T1" else mods["cluster_kc"]
    n = int(lt.max()) + 1
    within = np.zeros(n)
    dt = np.zeros(n)
    dk = np.zeros(n)
    for a, k in edges:
        dt[lt[a]] += 1
        dk[lk[k]] += 1
        if lt[a] == lk[k]:
            within[lt[a]] += 1
    m = float(len(edges))
    return float(np.sum(within / m - (dt / m) * (dk / m)))


@dataclass(frozen=True)
class GraphPair:
    family: str
    candidate: csr_matrix
    control: csr_matrix
    receipt: dict


def build_pair(b_canonical: csr_matrix, pn_type_index: np.ndarray, family: str,
               world: int) -> GraphPair:
    """Candidate (T1/T3) and matched random control (T0_1/T0_3) for one world."""
    wtag = tag(family, world)
    native, types, col_degrees = _source(b_canonical, pn_type_index, tg.FULL151_B_DIGEST)
    row_degrees = _row_degrees()
    type_degrees = np.bincount(types, weights=row_degrees, minlength=INPUT_TYPES).astype(np.int32)
    initial = _havel_hakimi(wtag, type_degrees, col_degrees)
    common, mix_attempts = _mix(family, world, initial)
    _TYPES_AND_COLS.update(types=types, cols=col_degrees)
    mods = modules(family, world)
    cand_t, opt = _optimize(family, world, common, mods)
    null_t, null_attempts = _null(family, world, common, opt["accepted_switches"])
    out = {}
    for arm, tgraph in ((family, cand_t), (f"T0_{family[1]}", null_t)):
        pn = _assign_type_edges_to_pn(tgraph, types, row_degrees, wtag)
        out[arm] = _weighted_csr(pn, native, wtag)
    ctrl = f"T0_{family[1]}"
    receipt = {
        "version": VERSION, "family": family, "world": world, "stream_tag": wtag,
        "source_digest": legacy_sparse_digest(native),
        "graph_digest": {arm: graph_digest(g) for arm, g in out.items()},
        "edges": E, "mix_attempts": mix_attempts, "null_attempts": null_attempts,
        "optimization": opt,
        "levels": {family: level_counts(family, mods, cand_t.edges),
                   ctrl: level_counts(family, mods, null_t.edges),
                   "common": level_counts(family, mods, common.edges)},
        "modularity_q": {family: modularity_q(family, mods, cand_t.edges),
                         ctrl: modularity_q(family, mods, null_t.edges),
                         "common": modularity_q(family, mods, common.edges)},
        "type_symmetric_difference_edges": len(set(cand_t.edges) ^ set(null_t.edges)),
    }
    return GraphPair(family, out[family], out[ctrl], receipt)


def structural_audit(pair: GraphPair, b_canonical: csr_matrix, pn_type_index: np.ndarray) -> dict:
    """Independent invariant check (recomputes from CSR arrays only)."""
    native = b_canonical.tocsc()
    types = np.asarray(pn_type_index)
    res = {}
    rows = None
    for arm, g in (("candidate", pair.candidate), ("control", pair.control)):
        if g.shape != (P, K) or g.nnz != E or not g.has_canonical_format:
            raise GraphError("graph shape/edge/canonical failure")
        csc = g.tocsc()
        if not np.array_equal(np.diff(csc.indptr), np.diff(native.indptr)):
            raise GraphError("KC column degree mismatch")
        for k in range(K):
            a = np.sort(csc.data[csc.indptr[k]:csc.indptr[k + 1]])
            b = np.sort(native.data[native.indptr[k]:native.indptr[k + 1]])
            if not np.array_equal(a, b):
                raise GraphError("per-KC weight multiset mismatch")
            t = types[csc.indices[csc.indptr[k]:csc.indptr[k + 1]]]
            if len(set(t.tolist())) != len(t):
                raise GraphError("two PNs of one input type on a KC")
        r = np.diff(g.indptr)
        if rows is None:
            rows = r
        elif not np.array_equal(rows, r):
            raise GraphError("PN row degree mismatch between arms")
        if set(np.unique(r).tolist()) != {91, 92}:
            raise GraphError("PN row degree roster")
        res[arm] = {"digest": graph_digest(g), "weight_sum": float(g.data.sum())}
    sup_c = set(zip(*pair.candidate.nonzero()))
    sup_0 = set(zip(*pair.control.nonzero()))
    sup_n = set(zip(*b_canonical.nonzero()))
    res["support_symdiff_candidate_control"] = len(sup_c ^ sup_0)
    res["support_symdiff_candidate_native"] = len(sup_c ^ sup_n)
    if res["support_symdiff_candidate_control"] == 0:
        raise GraphError("candidate and control share identical support")
    return res
