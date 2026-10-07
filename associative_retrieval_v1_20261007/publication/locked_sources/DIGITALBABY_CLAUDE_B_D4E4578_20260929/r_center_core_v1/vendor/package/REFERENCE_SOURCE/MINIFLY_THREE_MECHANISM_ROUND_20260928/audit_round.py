"""Independent, read-only audit of full-life receipts; no learner is instantiated."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict

from audit_fixture import audit as audit_fixture
from common_platform import BRANCHES, CHANNELS, DT, RECORD_SECONDS, branch_allows, mode_from_bytes
from fixture import fact_variants, make_world
from t2_graph import FULL151_B_DIGEST, VERSION as GRAPH_VERSION

ARMS = ("R0", "R2", "Rrand", "Z0", "Z1", "Z1_BUDGET_RANDOM", "T0_2", "T2", "FE0")
STAGES = ("old_end", "old_day", "new_end", "final")
STAGE_COUNTS = {"old_end": 34, "old_day": 34, "new_end": 56, "final": 152}
Z_ARMS = ("Z0", "Z1", "Z1_BUDGET_RANDOM")
R_ARMS = ("R0", "R2", "Rrand")
T_ARMS = ("T0_2", "T2")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_BUCKET_KEYS = {"0:-1", "0:+1", "1:-1", "1:+1"}


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _near(a, b, *, tolerance: float = 1e-10) -> bool:
    return _finite(a) and _finite(b) and abs(float(a) - float(b)) <= tolerance * max(
        1.0, abs(float(a)), abs(float(b)))


def _values(values, n: int) -> int:
    if not isinstance(values, list) or len(values) != 4 or any(not _finite(v) for v in values):
        raise AssertionError("invalid four-channel value row")
    return CHANNELS[max(range(n), key=lambda j: values[j])]


def _expected_probes(fixture: dict):
    """Reconstruct the sealed roster without reading a model or stored scores."""
    def one_stage(stage: str, branch: str):
        for set_name in ("old_fact", "new_fact"):
            if set_name == "new_fact" and stage in ("old_end", "old_day"):
                continue
            group = fixture[set_name]
            for item, (cue_hex, label) in enumerate(
                    zip(group["cues_hex"], group["labels"], strict=True)):
                cue = bytes.fromhex(cue_hex)
                for form, shown in (("canonical", cue), *fact_variants(cue).items()):
                    if form != "canonical" and stage != "final":
                        continue
                    yield {"stage": stage, "branch": branch, "set": set_name,
                           "item": item, "form": form, "cue_hex": shown.hex(),
                           "target": 48 + label}
        for set_name in ("old_relation", "new_relation"):
            if set_name == "new_relation" and stage in ("old_end", "old_day"):
                continue
            for item, row in enumerate(fixture[set_name]["taught"]):
                yield {"stage": stage, "branch": branch, "set": set_name + "_taught",
                       "item": item, "cue_hex": row["cue_hex"],
                       "target": 48 + row["label"]}
        held = fixture["old_relation"]["heldout"]
        for edge in range(3):
            good, other = held[2 * edge]["cue_hex"], held[2 * edge + 1]["cue_hex"]
            for order, options in enumerate(((good, other), (other, good))):
                yield {"stage": stage, "branch": branch, "set": "old_relation_heldout",
                       "item": edge, "order": order, "option_hex": list(options),
                       "target": ord("L") if order == 0 else ord("R")}

    # The runner visits both checkpoints within each branch before the next
    # branch; stage-major ordering would silently accept a reordered receipt.
    for pair in (("old_end", "old_day"), ("new_end", "final")):
        for branch in BRANCHES:
            for stage in pair:
                yield from one_stage(stage, branch)


def _gate_uniform(world: int, domain: str, branch: str, index: int) -> float:
    raw = f"R2-GATE-v1|{world}|{domain}|{branch}|{index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / 2**64


def _random_schedule(fixture: dict, branch: str, r2_events: list[dict]) -> dict[int, bool]:
    """Independently reconstruct the post-hoc yoked diagnostic placement."""
    if not isinstance(r2_events, list) or len(r2_events) != len(fixture["records"]):
        raise AssertionError("Rrand needs complete paired R2 gate ledger")
    eligible = defaultdict(list)
    counts = Counter()
    world = fixture["world"]
    for spec, event in zip(fixture["records"], r2_events, strict=True):
        index = spec["index"]
        if (event["record"] != index or event["branch"] != branch or
                event["stage"] != spec["stage"] or event["domain"] != spec["domain"] or
                event["answer"] != spec["answer"] or
                event["allowed"] != branch_allows(branch, spec["stage"], spec["domain"])):
            raise AssertionError("Rrand paired R2 event mismatch")
        if event["allowed"]:
            key = (spec["stage"], spec["domain"], spec["answer"])
            eligible[key].append(index)
            counts[key] += int(event["gate"])
    selected = set()
    for (stage, domain, answer), indices in eligible.items():
        def rank(index: int):
            raw = f"R2-RAND-v1|{world}|{branch}|{stage}|{domain}|{answer}|{index}".encode()
            return hashlib.sha256(raw).digest(), index
        selected.update(sorted(indices, key=rank)[:counts[(stage, domain, answer)]])
    if len(selected) != sum(counts.values()):
        raise AssertionError("Rrand paired dose mismatch")
    return {row["index"]: row["index"] in selected for row in fixture["records"]}


def _audit_graph(receipt: dict, arm: str, world: int, graph_expected: dict | None) -> None:
    got = receipt.get("graph_receipt")
    if arm not in T_ARMS:
        if got is not None:
            raise AssertionError("non-T arm carried graph receipt")
        return
    if graph_expected is None:
        raise AssertionError("T graph needs a presealed independent expected receipt")
    for graph in (got, graph_expected):
        if (not isinstance(graph, dict) or graph.get("schema") != GRAPH_VERSION or
                graph.get("world") != world or graph.get("source_digest") != FULL151_B_DIGEST or
                graph.get("shape") != [302, 5177] or graph.get("edges") != 27572):
            raise AssertionError("T graph identity/shape")
        digests = graph.get("graph_digest")
        if (not isinstance(digests, dict) or set(digests) != set(T_ARMS) or
                any(not isinstance(v, str) or _HEX64.fullmatch(v) is None
                    for v in digests.values()) or digests["T2"] == digests["T0_2"]):
            raise AssertionError("T graph digest pair")
    if got != graph_expected:
        raise AssertionError("T graph differs from presealed graph receipt")


def _audit_z_events(events: dict, fixture: dict, arm: str) -> None:
    world = fixture["world"]
    for branch in BRANCHES:
        rows = events[branch]
        if not isinstance(rows, list) or len(rows) != 600 * 4:
            raise AssertionError("Z teaching event count")
        for spec in fixture["records"]:
            index = spec["index"]
            at = index * RECORD_SECONDS + (86400.0 if spec["stage"] == "new" else 0.0) + 12 * DT
            active = 4 if spec["domain"] == "fact" else 2
            for channel in range(4):
                event = rows[4 * index + channel]
                write = bool(branch_allows(branch, spec["stage"], spec["domain"]) and
                             channel < active)
                context = event.get("context")
                if (not isinstance(context, (list, tuple)) or
                        tuple(context) != (world, branch, index, spec["stage"], spec["domain"]) or
                        event.get("channel") != channel or event.get("write") is not write or
                        not _near(event.get("teacher_at"), at, tolerance=1e-12)):
                    raise AssertionError("Z teaching route/timing")
                alpha, fast, slow = (event.get(key) for key in
                                     ("raw_alpha_l1", "raw_fast_l1", "raw_slow_l1"))
                if (any(not _finite(x) or x < 0 for x in (alpha, fast, slow)) or
                        not _near(alpha, fast + slow, tolerance=1e-10) or
                        not _finite(event.get("split_error")) or
                        not 0 <= event["split_error"] <= 1e-10):
                    raise AssertionError("Z raw alpha split certificate")
                buckets = event.get("budget_buckets")
                if not write:
                    if any(abs(x) > 1e-12 for x in (alpha, fast, slow)) or buckets != {}:
                        raise AssertionError("Z forbidden write changed memory")
                elif arm != "Z1_BUDGET_RANDOM":
                    if buckets != {}:
                        raise AssertionError("non-random Z arm carried random budget")
                else:
                    if not isinstance(buckets, dict) or set(buckets) != _BUCKET_KEYS:
                        raise AssertionError("Z random bucket roster")
                    actual_total = 0.0
                    for bucket in buckets.values():
                        count = bucket.get("coordinates")
                        target, actual = bucket.get("target_l1"), bucket.get("actual_l1")
                        if (type(count) is not int or count < 0 or
                                any(not _finite(x) or x < 0 for x in (target, actual)) or
                                not _near(target, actual, tolerance=1e-12) or
                                (count == 0 and (target != 0 or actual != 0))):
                            raise AssertionError("Z random write-budget mismatch")
                        actual_total += actual
                    if not _near(actual_total, slow, tolerance=1e-10):
                        raise AssertionError("Z random slow budget mismatch")


def audit_life(receipt: dict, arm: str, *, expected_world: int | None = None,
               r2_receipt: dict | None = None, graph_expected: dict | None = None) -> dict:
    """Validate one receipt against the sealed fixture and paired inputs.

    Rrand is an offline yoked diagnostic and requires its same-world R2 receipt.
    T arms require a graph receipt reconstructed before science and source-sealed.
    """
    if arm not in ARMS:
        raise AssertionError("unregistered mechanism arm")
    world = receipt.get("world")
    arm_identity = (receipt.get("arm") == arm if arm != "FE0"
                    else receipt.get("arm", "FE0") == "FE0")
    if (type(world) is not int or (expected_world is not None and world != expected_world) or
            not arm_identity):
        raise AssertionError("life world/arm identity")
    fixture = make_world(world)
    audit_fixture(fixture)
    if (receipt.get("schema") != "MINIFLY-THREE-MECHANISM-FOURSTORE-LIFE-v1" or
            receipt.get("fixture_digest") != fixture["digest"] or
            receipt.get("branches") != list(BRANCHES) or
            not _finite(receipt.get("max_clock_error")) or
            not 0 <= receipt["max_clock_error"] < 1e-6):
        raise AssertionError("life header/clock/fixture mismatch")
    _audit_graph(receipt, arm, world, graph_expected)

    first = receipt.get("first")
    if not isinstance(first, list) or len(first) != 600 * len(BRANCHES):
        raise AssertionError("first-answer roster")
    first_values = {}
    for spec in fixture["records"]:
        index = spec["index"]
        expected_at = (index * RECORD_SECONDS +
                       (86400.0 if spec["stage"] == "new" else 0.0) + 12 * DT)
        domain = mode_from_bytes(bytes.fromhex(spec["cue_hex"]))
        if domain != spec["domain"]:
            raise AssertionError("byte-only mode disagreement")
        for j, branch in enumerate(BRANCHES):
            got = first[index * len(BRANCHES) + j]
            if (got.get("record") != index or got.get("branch") != branch or
                    not _near(got.get("teacher_at"), expected_at, tolerance=1e-12) or
                    got.get("emitted") != _values(got.get("values"), 4 if domain == "fact" else 2)):
                raise AssertionError("first-answer timing/readout")
            first_values[(branch, index)] = got["values"]

    expected_writes = {}
    for branch in BRANCHES:
        counts = Counter()
        for spec in fixture["records"]:
            if branch_allows(branch, spec["stage"], spec["domain"]):
                counts[f'{spec["stage"]}_{spec["domain"]}'] += (
                    4 if spec["domain"] == "fact" else 2)
        expected_writes[branch] = {name: counts[name] for name in
                                   ("old_fact", "old_relation", "new_fact", "new_relation")}
    if receipt.get("writes") != expected_writes:
        raise AssertionError("branch write ledger")

    probes = receipt.get("probes")
    expected = list(_expected_probes(fixture))
    if not isinstance(probes, list) or len(probes) != len(expected):
        raise AssertionError("probe roster count")
    stage_counts = Counter()
    for got, spec in zip(probes, expected, strict=True):
        if any(got.get(key) != value for key, value in spec.items()):
            if spec["set"] == "old_relation_heldout":
                raise AssertionError("heldout choice/target/roster")
            raise AssertionError("probe fixture roster/target")
        stage_counts[(spec["stage"], spec["branch"])] += 1
        if spec["set"] == "old_relation_heldout":
            options = got.get("option_values")
            scores = got.get("option_scores")
            if (not isinstance(options, list) or len(options) != 2 or
                    not isinstance(scores, list) or len(scores) != 2):
                raise AssertionError("heldout choice/values")
            for values in options:
                _values(values, 2)
            expected_scores = [values[1] - values[0] for values in options]
            if (any(not _near(a, b, tolerance=1e-12)
                    for a, b in zip(scores, expected_scores, strict=True)) or
                    got.get("emitted") != (ord("L") if scores[0] >= scores[1] else ord("R"))):
                raise AssertionError("heldout choice/readout")
        else:
            domain = "relation" if "relation" in spec["set"] else "fact"
            if got.get("emitted") != _values(got.get("values"), 2 if domain == "relation" else 4):
                raise AssertionError("probe class readout")
        if got.get("correct") != int(got["emitted"] == spec["target"]):
            raise AssertionError("probe correctness flag")
    if any(stage_counts[(stage, branch)] != count for stage, count in STAGE_COUNTS.items()
           for branch in BRANCHES):
        raise AssertionError("probe stage completeness")

    # A receipt exposes digests, not the raw checkpoint. Their format and
    # cross-arm invariants are auditable here; recomputing a digest needs replay.
    digests = receipt.get("end_state_digests")
    bank_count = 8 if arm in R_ARMS else 4
    if (not isinstance(digests, dict) or set(digests) != set(BRANCHES) or
            any(not isinstance(row, list) or len(row) != bank_count or
                any(not isinstance(value, str) or _HEX64.fullmatch(value) is None
                    for value in row) for row in digests.values())):
        raise AssertionError("end state digest roster")

    events = receipt.get("mechanism_events")
    if not isinstance(events, dict) or set(events) != set(BRANCHES):
        raise AssertionError("mechanism event branches")
    if arm in R_ARMS:
        if arm == "Rrand":
            if r2_receipt is None:
                raise AssertionError("Rrand requires a paired R2 receipt")
            audit_life(r2_receipt, "R2", expected_world=world)
        for branch in BRANCHES:
            rows = events[branch]
            if not isinstance(rows, list) or len(rows) != 600:
                raise AssertionError("R gate event count")
            expected_random = (_random_schedule(fixture, branch,
                               r2_receipt["mechanism_events"][branch])
                               if arm == "Rrand" else None)
            for spec, row in zip(fixture["records"], rows, strict=True):
                index = spec["index"]
                expected_u = _gate_uniform(world, spec["domain"], branch, index)
                if (row.get("record") != index or row.get("branch") != branch or
                        row.get("stage") != spec["stage"] or
                        row.get("domain") != spec["domain"] or
                        row.get("answer") != spec["answer"] or
                        row.get("allowed") is not
                        branch_allows(branch, spec["stage"], spec["domain"]) or
                        not _near(row.get("uniform"), expected_u, tolerance=1e-16) or
                        not _finite(row.get("error")) or not 0 <= row["error"] <= 1 or
                        type(row.get("gate")) is not bool):
                    raise AssertionError("R gate causal ledger")
                pred = row.get("prefeedback_u")
                if pred != first_values[(branch, index)]:
                    raise AssertionError("R gate did not use saved prefeedback output")
                active = range(4 if spec["domain"] == "fact" else 2)
                peak = max(pred[j] for j in active)
                weights = [math.exp(pred[j] - peak) for j in active]
                target = CHANNELS.index(spec["answer"])
                error = 1 - weights[target] / sum(weights)
                if not _near(error, row["error"], tolerance=1e-12):
                    raise AssertionError("R feedback error equation")
                if arm == "R2" and row["gate"] is not (row["uniform"] < row["error"]):
                    raise AssertionError("R2 gate equation")
                if arm == "R0" and row["gate"] is not True:
                    raise AssertionError("R0 ungated control")
                if arm == "Rrand" and row["gate"] is not expected_random[index]:
                    raise AssertionError("Rrand yoked gate schedule")
    elif arm in Z_ARMS:
        _audit_z_events(events, fixture, arm)
    elif any(events[branch] != [] for branch in BRANCHES):
        raise AssertionError("non-Z/T arm carried mechanism events")
    return {"pass": True, "world": world, "arm": arm,
            "first": len(first), "probes": len(probes),
            "mechanism_events": {k: len(v) for k, v in events.items()}}


def audit_r_family(r0: dict, r2: dict, rrand: dict, *, expected_world: int) -> dict:
    """Check the three R banks together, including the invariant FE0 shared bank."""
    audit_life(r0, "R0", expected_world=expected_world)
    audit_life(r2, "R2", expected_world=expected_world)
    audit_life(rrand, "Rrand", expected_world=expected_world, r2_receipt=r2)
    for branch in BRANCHES:
        shared = r0["end_state_digests"][branch][:4]
        if (r2["end_state_digests"][branch][:4] != shared or
                rrand["end_state_digests"][branch][:4] != shared):
            raise AssertionError("R shared-bank state diverged across controls")
    return {"pass": True, "world": expected_world, "arms": list(R_ARMS)}


def audit_t_pair(t0: dict, t2: dict, *, expected_world: int,
                 graph_expected: dict) -> dict:
    """Bind both topology arms to the same independently presealed graph pair."""
    audit_life(t0, "T0_2", expected_world=expected_world, graph_expected=graph_expected)
    audit_life(t2, "T2", expected_world=expected_world, graph_expected=graph_expected)
    if t0["graph_receipt"] != t2["graph_receipt"]:
        raise AssertionError("T matched graph pair identity")
    return {"pass": True, "world": expected_world, "arms": list(T_ARMS)}
