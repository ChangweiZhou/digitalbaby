"""V43 offline scoring controls on frozen G-route authority events.

No learner or policy state is mutated here. ``slot_uid`` must identify a slot
generation, rather than a reusable physical slot address. Optional ``logpF_true``
and ``logpS_true`` are natural logarithms and take precedence over probabilities.

SHUFFLED-ALPHA permutes LOCAL-BAYES-E2's pre-event slow weights within each
support-bin x age-quartile x prior-similarity-use-bin bucket. Prior-use bins are
0-1, 2-3, and 4+. Age cut points are the linear 25th, 50th and
75th percentiles of *all* G-event ages in this seed/load trajectory. Assignment
uses ``searchsorted(..., side='right')``: ties belong above their cut point;
repeated cut points can leave empty quartile bins. Buckets are visited in sorted
(support bin, age bin, prior-use bin) order, and events inside each bucket stay chronological.
One dedicated RNG is seeded with ``seed * 100000 + load + 4300000``. Each bucket
is randomly ordered once. For a nonsingleton bucket, a uniformly sampled nonzero
cyclic shift of that ordering assigns sources to destinations. Every event thus
gets a different source index. This is a deterministic, outcome-independent
derangement sampler, not a uniform sampler over all possible derangements. There
is no redraw. Singletons retain themselves and are excluded from the decisive
G_permutable subset. Equal-weight nonsingleton buckets remain included.

The event oracle selects FAST or SLOW separately on every event. The slot oracle
selects one of FAST or SLOW by cumulative G loss per slot generation, choosing
FAST on an exact cumulative tie. These are retrospective diagnostic lower bounds.
"""

import hashlib
import json
import math

import numpy as np


LN2 = math.log(2.0)


def _true_log_probability(event, probability_key, log_key):
    if log_key in event:
        value = float(event[log_key])
        if not math.isfinite(value) or value > 1e-12:
            raise ValueError(f"Invalid natural log probability {log_key}: {value}")
        return min(0.0, value)
    value = float(event[probability_key])
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"Invalid true probability {probability_key}: {value}")
    return math.log(value)


def mixture_loss(logp_fast, logp_slow, slow_weight):
    """Stable binary mixture log loss in bits from natural log probabilities."""
    weight = float(slow_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(f"Slow weight outside [0, 1]: {weight}")
    if weight == 0.0:
        return -float(logp_fast) / LN2
    if weight == 1.0:
        return -float(logp_slow) / LN2
    a = math.log1p(-weight) + float(logp_fast)
    b = math.log(weight) + float(logp_slow)
    larger = max(a, b)
    smaller = min(a, b)
    log_probability = larger + math.log1p(math.exp(smaller - larger))
    return -log_probability / LN2


def _support_bin(support):
    value = int(support)
    if value != support or value < 2:
        raise ValueError("G-route support must be an integer >= 2")
    return 0 if value == 2 else (1 if value <= 4 else 2)


def _slot_key(slot_uid):
    return json.dumps(slot_uid, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _prior_use_bin(prior_uses):
    value = int(prior_uses)
    if value != prior_uses or value < 0:
        raise ValueError("Prior similarity-use count must be a nonnegative integer")
    return 0 if value <= 1 else (1 if value <= 3 else 2)


def evaluate(events, seed, load):
    """Evaluate saved events in order; return only JSON-serializable values.

    All input events must belong to one seed/load's G route. The caller is
    responsible for selecting that route. No category or outcome is used for
    stratification, permutation construction, or eligibility.
    """
    seed, load = int(seed), int(load)
    rng_seed = seed * 100000 + load + 4300000
    if rng_seed < 0:
        raise ValueError("The dedicated RNG seed must be nonnegative")
    count = len(events)
    if count == 0:
        return {
            "shuffled_losses": [], "permutable_mask": [],
            "event_oracle_losses": [], "slot_oracle_losses": [],
            "slot_oracle_rows": [], "permutation_source_indices": [],
            "event_indices": [], "age_bins": [], "stratum_keys": [],
            "audit": {
                "total_G_n": 0, "permutable_n": 0, "singleton_n": 0,
                "coverage": None, "multiset_preserved": True,
                "mass_preserved": True, "reassigned_count": 0,
                "derangement_preserved": True,
                "weights_different_count": 0, "strata_count": 0,
                "bucket_dimension_count": 3, "total_possible_bucket_count": 36,
                "bucket_dimensions": ["support", "age_quartile", "prior_similarity_use"],
                "fraction_changed_assignments": None,
                "fraction_weights_different": None,
                "fraction_changed_assignments_permutable": None,
                "fraction_weights_different_permutable": None,
                "age_quartile_edges": [], "rng_seed": rng_seed,
                "local_loss_reconstruction_max_abs_error": 0.0,
                "strata": [],
            },
        }

    event_ids = [int(event["event_index"]) for event in events]
    if any(a >= b for a, b in zip(event_ids, event_ids[1:])):
        raise ValueError("G events must have unique, strictly increasing event_index")
    ages = np.asarray([float(event["slot_age"]) for event in events], dtype=np.float64)
    if np.any(~np.isfinite(ages)) or np.any(ages < 0):
        raise ValueError("Slot ages must be finite and nonnegative")
    edges = np.quantile(ages, [0.25, 0.5, 0.75], method="linear")
    age_bins = np.searchsorted(edges, ages, side="right")
    weights = [float(event["local_w_before"]) for event in events]
    log_fast = [_true_log_probability(event, "pF_true", "logpF_true") for event in events]
    log_slow = [_true_log_probability(event, "pS_true", "logpS_true") for event in events]
    local_losses = [mixture_loss(fast, slow, weight)
                    for fast, slow, weight in zip(log_fast, log_slow, weights)]
    reconstruction_errors = []
    for event, recomputed in zip(events, local_losses):
        supplied = float(event["local_loss"])
        if not math.isfinite(supplied):
            raise ValueError("Recorded local loss must be finite")
        error = abs(supplied - recomputed)
        reconstruction_errors.append(error)
        if error > 1e-8 * max(1.0, abs(recomputed)):
            raise ValueError("Recorded local loss disagrees with its frozen mixture")
        _prior_use_bin(event["prior_similarity_use_count"])
        if event["category"] not in ("target", "nontarget"):
            raise ValueError("Unknown authority-event category")

    buckets = {}
    for index, event in enumerate(events):
        key = (_support_bin(event["support"]), int(age_bins[index]),
               _prior_use_bin(event["prior_similarity_use_count"]))
        buckets.setdefault(key, []).append(index)
    rng = np.random.default_rng(rng_seed)
    sources = list(range(count))
    permutable = [False] * count
    strata = []
    for (support_bin, age_bin, prior_use_bin), indices in sorted(buckets.items()):
        random_order = rng.permutation(np.asarray(indices, dtype=np.int64)).tolist()
        cyclic_shift = int(rng.integers(1, len(indices))) if len(indices) > 1 else 0
        source_by_destination = {
            destination: random_order[(position + cyclic_shift) % len(indices)]
            for position, destination in enumerate(random_order)
        }
        assigned_sources = [source_by_destination[index] for index in indices]
        if len(indices) > 1 and any(a == b for a, b in zip(indices, assigned_sources)):
            raise AssertionError("Eligible shuffle bucket contains a fixed point")
        for destination, source in zip(indices, assigned_sources):
            sources[destination] = source
            permutable[destination] = len(indices) > 1
        original = [weights[index] for index in indices]
        shuffled = [weights[index] for index in assigned_sources]
        sorted_original = sorted(original)
        sorted_shuffled = sorted(shuffled)
        original_mass = math.fsum(original)
        shuffled_mass = math.fsum(shuffled)
        multiset_preserved = sorted_original == sorted_shuffled
        mass_preserved = original_mass == shuffled_mass
        if not multiset_preserved or not mass_preserved:
            raise AssertionError("Within-stratum weight conservation failed")
        encoded_multiset = json.dumps([value.hex() for value in sorted_original],
                                     separators=(",", ":")).encode("ascii")
        strata.append({
            "support_bin": ("2", "3-4", "5+")[support_bin],
            "age_bin": age_bin, "n": len(indices),
            "prior_similarity_use_bin": ("0-1", "2-3", "4+")[prior_use_bin],
            "permutable": len(indices) > 1,
            "cyclic_shift": cyclic_shift,
            "derangement_preserved": (all(a != b for a, b in zip(indices, assigned_sources))
                                       if len(indices) > 1 else True),
            "multiset_preserved": multiset_preserved,
            "weight_multiset_sha256": hashlib.sha256(encoded_multiset).hexdigest(),
            "original_mass": original_mass, "shuffled_mass": shuffled_mass,
            "original_mass_hex": original_mass.hex(),
            "shuffled_mass_hex": shuffled_mass.hex(),
            "mass_preserved": mass_preserved,
            "reassigned_count": sum(a != b for a, b in zip(indices, assigned_sources)),
            "weights_different_count": sum(a != b for a, b in zip(original, shuffled)),
        })

    shuffled_losses = [mixture_loss(log_fast[index], log_slow[index], weights[sources[index]])
                       for index in range(count)]
    fast_losses = [-value / LN2 for value in log_fast]
    slow_losses = [-value / LN2 for value in log_slow]
    event_oracle_losses = [min(fast, slow) for fast, slow in zip(fast_losses, slow_losses)]
    slots = {}
    for index, event in enumerate(events):
        slots.setdefault(_slot_key(event["slot_uid"]), []).append(index)
    slot_oracle_losses = [None] * count
    slot_oracle_rows = []
    for key, indices in sorted(slots.items()):
        fast_total = math.fsum(fast_losses[index] for index in indices)
        slow_total = math.fsum(slow_losses[index] for index in indices)
        select_fast = fast_total <= slow_total
        selected = fast_losses if select_fast else slow_losses
        for index in indices:
            slot_oracle_losses[index] = selected[index]
        slot_oracle_rows.append({
            "slot_uid": events[indices[0]]["slot_uid"], "n": len(indices),
            "fast_loss_sum": fast_total, "slow_loss_sum": slow_total,
            "selected_expert": "FAST" if select_fast else "SLOW",
            "selected_loss_sum": fast_total if select_fast else slow_total,
        })

    permutable_n = sum(permutable)
    reassigned_count = sum(index != source for index, source in enumerate(sources))
    derangement_preserved = all((source != index) == eligible
                                for index, (source, eligible) in enumerate(zip(sources, permutable)))
    if not derangement_preserved or reassigned_count != permutable_n:
        raise AssertionError("Global derangement eligibility check failed")
    weights_different_count = sum(weights[index] != weights[source]
                                  for index, source in enumerate(sources))
    audit = {
        "total_G_n": count, "permutable_n": permutable_n,
        "singleton_n": count - permutable_n,
        "coverage": permutable_n / count,
        "multiset_preserved": all(row["multiset_preserved"] for row in strata),
        "mass_preserved": all(row["mass_preserved"] for row in strata),
        "derangement_preserved": derangement_preserved,
        "reassigned_count": reassigned_count,
        "weights_different_count": weights_different_count,
        "strata_count": len(strata),
        "bucket_dimension_count": 3, "total_possible_bucket_count": 36,
        "bucket_dimensions": ["support", "age_quartile", "prior_similarity_use"],
        "fraction_changed_assignments": reassigned_count / count,
        "fraction_weights_different": weights_different_count / count,
        "fraction_changed_assignments_permutable": (reassigned_count / permutable_n
                                                    if permutable_n else None),
        "fraction_weights_different_permutable": (weights_different_count / permutable_n
                                                 if permutable_n else None),
        "age_quartile_edges": edges.tolist(), "rng_seed": rng_seed,
        "local_loss_reconstruction_max_abs_error": max(reconstruction_errors),
        "strata": strata,
    }
    return {
        "shuffled_losses": shuffled_losses, "permutable_mask": permutable,
        "event_oracle_losses": event_oracle_losses,
        "slot_oracle_losses": slot_oracle_losses,
        "slot_oracle_rows": slot_oracle_rows,
        "permutation_source_indices": sources,
        "event_indices": event_ids, "age_bins": age_bins.tolist(),
        "stratum_keys": [
            [("2", "3-4", "5+")[_support_bin(event["support"])],
             int(age_bins[index]),
             ("0-1", "2-3", "4+")[_prior_use_bin(event["prior_similarity_use_count"])]]
            for index, event in enumerate(events)
        ],
        "audit": audit,
    }


def self_test():
    """Synthetic checks only; no empirical seed or scientific outcome is used."""
    checks = []
    rows = []
    for index, (fast, slow, weight, support, slot) in enumerate([
        (0.9, 0.2, 0.1, 2, "A:0"),
        (0.2, 0.9, 0.3, 2, "A:0"),
        (0.3, 0.8, 0.7, 2, "B:0"),
        (0.7, 0.4, 0.9, 2, "B:1"),
        (0.5, 0.5, 0.8, 5, "C:0"),
    ]):
        rows.append({"event_index": index * 3, "slot_uid": slot,
                     "support": support, "slot_age": 0,
                     "prior_similarity_use_count": index,
                     "local_w_before": weight, "pF_true": fast, "pS_true": slow,
                     "local_loss": mixture_loss(math.log(fast), math.log(slow), weight),
                     "category": "target" if index % 2 else "nontarget"})
    before = json.dumps(rows, sort_keys=True)
    result = evaluate(rows, seed=43, load=1)
    assert result == evaluate(rows, seed=43, load=1)
    assert json.dumps(rows, sort_keys=True) == before
    checks.append("deterministic_and_input_immutable")
    assert result["audit"]["age_quartile_edges"] == [0.0, 0.0, 0.0]
    assert all(row["age_bin"] == 3 for row in result["audit"]["strata"])
    checks.append("repeated_age_edges_go_right")
    assert result["permutable_mask"] == [True, True, True, True, False]
    assert result["audit"]["permutable_n"] == 4
    assert result["audit"]["singleton_n"] == 1
    assert result["audit"]["coverage"] == 0.8
    checks.append("singleton_exclusion_and_coverage")
    assert result["audit"]["mass_preserved"] and result["audit"]["multiset_preserved"]
    assert result["audit"]["reassigned_count"] > 0
    for index, source in enumerate(result["permutation_source_indices"]):
        expected = -math.log2((1 - rows[source]["local_w_before"]) * rows[index]["pF_true"]
                             + rows[source]["local_w_before"] * rows[index]["pS_true"])
        assert abs(expected - result["shuffled_losses"][index]) < 1e-12
    checks.append("exact_weight_conservation_and_mixture_alignment")
    assert result["audit"]["derangement_preserved"]
    assert result["audit"]["reassigned_count"] == result["audit"]["permutable_n"]
    for destination, source in enumerate(result["permutation_source_indices"]):
        assert (destination != source) == result["permutable_mask"][destination]
        assert result["stratum_keys"][destination] == result["stratum_keys"][source]
    assert result["event_indices"] == [row["event_index"] for row in rows]
    assert result["age_bins"] == [3] * len(rows)
    checks.append("every_eligible_event_deranged_and_mapping_exposed")
    usage_fixture = [dict(rows[index % len(rows)], event_index=index,
                          support=2, slot_age=0, prior_similarity_use_count=index)
                     for index in range(6)]
    usage_result = evaluate(usage_fixture, seed=43, load=1)
    assert usage_result["audit"]["bucket_dimension_count"] == 3
    assert usage_result["audit"]["total_possible_bucket_count"] == 36
    assert usage_result["audit"]["strata_count"] == 3
    assert {row["prior_similarity_use_bin"] for row in usage_result["audit"]["strata"]} == {
        "0-1", "2-3", "4+"}
    for destination, source in enumerate(usage_result["permutation_source_indices"]):
        assert _prior_use_bin(usage_fixture[destination]["prior_similarity_use_count"]) == (
            _prior_use_bin(usage_fixture[source]["prior_similarity_use_count"]))
    checks.append("prior_use_strata_never_cross_at_fixed_support_and_age")
    slot_rows = {row["slot_uid"]: row for row in result["slot_oracle_rows"]}
    assert slot_rows["A:0"]["selected_expert"] == "FAST"
    assert slot_rows["B:0"]["selected_expert"] == "SLOW"
    assert slot_rows["B:1"]["selected_expert"] == "FAST"
    assert slot_rows["C:0"]["selected_expert"] == "FAST"
    assert math.fsum(result["event_oracle_losses"]) < math.fsum(result["slot_oracle_losses"])
    checks.append("slot_generation_oracle_and_fast_ties")
    assert mixture_loss(-1000.0, -1002.0, 0.0) == 1000.0 / LN2
    assert mixture_loss(-1000.0, -1002.0, 1.0) == 1002.0 / LN2
    assert math.isfinite(mixture_loss(-1000.0, -1002.0, 0.5))
    tiny = dict(rows[0], pF_true=0.0, pS_true=0.0,
                logpF_true=-1000.0, logpS_true=-1002.0)
    tiny["local_loss"] = mixture_loss(-1000.0, -1002.0, tiny["local_w_before"])
    assert math.isfinite(evaluate([tiny], 43, 1)["shuffled_losses"][0])
    checks.append("underflow_safe_logs_and_exact_endpoints")
    assert evaluate([], 43, 1)["audit"]["coverage"] is None
    constant = [dict(row, local_w_before=0.5,
                     local_loss=mixture_loss(math.log(row["pF_true"]),
                                             math.log(row["pS_true"]), 0.5)) for row in rows]
    equal = evaluate(constant, 43, 1)
    assert equal["audit"]["permutable_n"] == 4
    assert equal["audit"]["weights_different_count"] == 0
    assert equal["audit"]["reassigned_count"] == equal["audit"]["permutable_n"]
    assert all(source != destination for destination, source
               in enumerate(equal["permutation_source_indices"])
               if equal["permutable_mask"][destination])
    checks.append("empty_sample_and_equal_weight_buckets")
    bad = [dict(row) for row in rows]
    bad[0]["local_loss"] += 1
    try:
        evaluate(bad, 43, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched frozen loss was accepted")
    checks.append("frozen_loss_alignment_rejection")
    return {"passed": True, "checks": checks, "count": len(checks)}
