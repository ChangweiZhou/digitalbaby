"""Independent read-only audit of V43 authority-event tables.

This module imports no learner, engine, or offline-control implementation. It
reconstructs losses from frozen natural log probabilities, checks the serialized
permutation and independently rebuilds support/age/prior-use strata and hindsight
expert oracles. Top-1 values of arbitrary mixtures cannot be reconstructed from
true-byte probabilities, so their saved correctness indicators are reconciled to
aggregates, not claimed independently reproduced. No learning is performed.
"""

import collections
import json
import math

import numpy as np


POLICIES = ("SUPER", "READ-HYBRID", "SILENT-HARD", "ANCHOR-ONLY", "FIXED-50",
            "GLOBAL-BAYES", "LOCAL-BAYES-0", "LOCAL-BAYES-E2")


def audit_events(result, cols):
    """Return a JSON-serializable audit; collect errors without fail-fast asserts.

    ``result`` is completed data (an enclosing ``data`` envelope is also accepted).
    ``cols`` is the per-trajectory NPZ mapping of NumPy arrays. Individual numerical
    values use 2e-10 absolute plus 2e-12 relative tolerance; cumulative values use
    2e-8 absolute plus 2e-12 relative tolerance because production sums sequentially.
    Multiset and ``math.fsum`` permutation conservation are checked exactly.
    """
    if "groups" not in result and isinstance(result.get("data"), dict):
        result = result["data"]
    checks, errors, counters = [], [], {}
    counters.update(max_oracle_violation_bits=0.0, max_oracle_capture_excess=0.0,
                    oracle_bound_event_policy_pairs=0, positive_headroom_count=0,
                    positive_headroom_capture_count=0, max_raw_oracle_capture=None)
    maxdiff = {"event_loss": 0.0, "aggregate_loss": 0.0, "oracle_loss": 0.0,
               "quartile_edge": 0.0}
    error_count = 0

    def check(name, passed, detail=None):
        nonlocal error_count
        passed = bool(passed)
        checks.append({"check": name, "passed": passed})
        if not passed:
            error_count += 1
            if len(errors) < 100:
                errors.append({"check": name, "detail": str(detail) if detail is not None else None})
        return passed

    def close(name, actual, expected, kind="aggregate_loss"):
        try:
            a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
            if a.shape != b.shape:
                return check(name, False, f"Shape mismatch {a.shape} versus {b.shape}")
            finite = np.all(np.isfinite(a)) and np.all(np.isfinite(b))
            difference = float(np.max(np.abs(a - b))) if a.size and finite else (0.0 if not a.size else math.inf)
            if math.isfinite(difference):
                maxdiff[kind] = max(maxdiff.get(kind, 0.0), difference)
            atol = 2e-8 if kind == "aggregate_loss" else 2e-10
            return check(name, finite and np.all(np.abs(a - b) <= atol + 2e-12 * np.abs(b)),
                         f"Maximum absolute difference {difference:.17g}")
        except Exception as exc:
            return check(name, False, f"{type(exc).__name__}: {exc}")

    def finish():
        return {"passed": error_count == 0, "checks": checks, "counters": counters,
                "errors": errors, "error_count": error_count,
                "errors_truncated": error_count > len(errors), "maxdiff": maxdiff,
                "limitations": ["Mixture top-1 correctness is aggregate-reconciled, not independently reconstructed.",
                                "F-route events are absent from this authority table; whole-stream checks use recorded F contributions."]}

    required = {"route", "event_index", "slot_uid", "generation_uid", "support", "slot_age",
                "prior_similarity_use_count", "category", "logpF_true", "logpS_true",
                "local_w_before", "local_0_w_before", "global_w_before", "local_loss",
                "policy_losses", "policy_correct"}
    missing = sorted(required - set(cols))
    if not check("authority_columns_present", not missing, missing):
        return finish()
    try:
        n = len(cols["route"])
        counters.update(authority_events=n, seed=int(result["seed"]), load=int(result["load"]),
                        event_policy_losses_checked=n * len(POLICIES))
        if not check("authority_column_lengths", all(len(cols[key]) == n for key in required)):
            return finish()
        check("authority_rows_match_completed", n == int(result.get("authority_events_count", n)))
        if "__row_count__" in cols:
            check("npz_row_count", n == int(cols["__row_count__"]))
        route = np.asarray(cols["route"], dtype=str)
        category = np.asarray(cols["category"], dtype=str)
        event_ids = np.asarray(cols["event_index"], dtype=np.int64)
        check("authority_routes_only_ATG", set(route.tolist()) <= set("ATG"))
        check("authority_categories_valid", set(category.tolist()) <= {"target", "nontarget"})
        check("physical_event_indices_unique_chronological", bool(np.all(np.diff(event_ids) > 0)))
        lf = np.asarray(cols["logpF_true"], dtype=float)
        ls = np.asarray(cols["logpS_true"], dtype=float)
        if not check("finite_log_probabilities", np.all(np.isfinite(lf)) and np.all(np.isfinite(ls))
                     and np.all(lf <= 1e-12) and np.all(ls <= 1e-12)):
            return finish()
        fast, slow = -lf / math.log(2.0), -ls / math.log(2.0)
        weights = {"GLOBAL-BAYES": np.asarray(cols["global_w_before"], dtype=float),
                   "LOCAL-BAYES-0": np.asarray(cols["local_0_w_before"], dtype=float),
                   "LOCAL-BAYES-E2": np.asarray(cols["local_w_before"], dtype=float)}
        weights_ok = True
        for policy, weight in weights.items():
            weights_ok &= check("finite_unit_weights/" + policy,
                                np.all(np.isfinite(weight)) and np.all((weight >= 0) & (weight <= 1)))
        if not weights_ok:
            return finish()

        def mix(weight, indices=None):
            ff, ss = (lf, ls) if indices is None else (lf[indices], ls[indices])
            with np.errstate(divide="ignore", invalid="ignore"):
                return -np.logaddexp(np.log1p(-weight) + ff, np.log(weight) + ss) / math.log(2.0)

        primary = np.where(route == "A", slow, fast)
        reconstructed = {"SUPER": fast, "READ-HYBRID": slow,
                         "SILENT-HARD": np.where(route == "G", slow, primary),
                         "ANCHOR-ONLY": primary,
                         "FIXED-50": np.where(route == "G", mix(0.5), primary)}
        for policy, weight in weights.items():
            reconstructed[policy] = np.where(route == "G", mix(weight), primary)
        saved_losses = [json.loads(str(value)) for value in cols["policy_losses"]]
        saved_correct = [json.loads(str(value)) for value in cols["policy_correct"]]
        check("policy_loss_rosters", all(set(row) == set(POLICIES) for row in saved_losses))
        check("policy_correct_rosters", all(set(row) == set(POLICIES) for row in saved_correct))
        correct = {}
        for policy in POLICIES:
            recorded = np.asarray([row[policy] for row in saved_losses], dtype=float)
            correct[policy] = np.asarray([row[policy] for row in saved_correct], dtype=float)
            close("frozen_event_losses/" + policy, recorded, reconstructed[policy], "event_loss")
            check("binary_saved_correct/" + policy, np.all(np.isin(correct[policy], [0.0, 1.0])))
        close("local_loss_alignment", cols["local_loss"], reconstructed["LOCAL-BAYES-E2"], "event_loss")
    except Exception as exc:
        check("authority_decode_and_loss_reconstruction", False, f"{type(exc).__name__}: {exc}")
        return finish()

    groups = result.get("groups", {})

    def aggregate(group, policy, indices, losses, correctness=None):
        row = groups.get(group, {}).get(policy)
        expected_n = len(indices)
        if row is None:
            check("aggregate_present/" + group + "/" + policy, expected_n == 0,
                  f"Missing group with {expected_n} expected events")
            return
        prefix = group + "/" + policy
        check("aggregate_n/" + prefix, row.get("n") == expected_n,
              f"Recorded {row.get('n')}, expected {expected_n}")
        total = math.fsum(float(losses[index]) for index in indices)
        close("aggregate_loss/" + prefix, row["loss_sum"], total)
        if expected_n:
            close("aggregate_bpb/" + prefix, row.get("bpb"), total / expected_n)
        else:
            check("empty_bpb/" + prefix, row.get("bpb") is None)
        if correctness is not None:
            total_correct = int(sum(int(correctness[index]) for index in indices))
            check("aggregate_correct/" + prefix, row.get("correct") == total_correct)
            if expected_n:
                close("aggregate_accuracy/" + prefix, row.get("accuracy"), total_correct / expected_n)
        else:
            check("offline_accuracy_unavailable/" + prefix,
                  row.get("correct") is None and row.get("accuracy") is None)

    def oracle_bound(policy, losses, oracle_losses, indices):
        differences = np.asarray(oracle_losses)[indices] - np.asarray(losses)[indices]
        violation = max(0.0, float(np.max(differences))) if len(indices) else 0.0
        counters["max_oracle_violation_bits"] = max(counters["max_oracle_violation_bits"], violation)
        counters["oracle_bound_event_policy_pairs"] += len(indices)
        check("event_oracle_pointwise_lower_bound/" + policy,
              np.all(np.isfinite(differences)) and np.all(differences <= 1e-10),
              f"Maximum positive loss violation {violation:.17g} bits")
        oracle_sum = math.fsum(float(oracle_losses[index]) for index in indices)
        policy_sum = math.fsum(float(losses[index]) for index in indices)
        check("event_oracle_cumulative_lower_bound/" + policy,
              policy_sum + 2e-8 + 2e-12 * abs(oracle_sum) >= oracle_sum,
              f"Policy loss {policy_sum:.17g}, event-oracle loss {oracle_sum:.17g}")

    try:
        for label in "ATG":
            selected = np.flatnonzero(route == label).tolist()
            counters[label + "_events"] = len(selected)
            for policy in POLICIES:
                aggregate("route/" + label, policy, selected, reconstructed[policy], correct[policy])
            saved_n = result.get("audits", {}).get("route_counts", {}).get(label)
            if saved_n is not None:
                check("recorded_route_count/" + label, saved_n == len(selected))
    except Exception as exc:
        check("route_aggregates", False, f"{type(exc).__name__}: {exc}")

    gi = np.flatnonzero(route == "G")
    gn = len(gi)
    counters["G_events"] = gn
    try:
        generations = np.asarray(cols["generation_uid"])[gi]
        slotids = np.asarray(cols["slot_uid"])[gi]
        check("G_generation_identity", np.array_equal(generations, slotids))
        generation_members = collections.defaultdict(list)
        for index in gi:
            generation_members[int(cols["generation_uid"][index])].append(int(index))
        event_oracle = np.minimum(fast, slow)
        for policy in POLICIES:
            oracle_bound(policy, reconstructed[policy], event_oracle, gi)
        slot_oracle = np.zeros(n, dtype=float)
        oracle_rows = {}
        for generation, indices in generation_members.items():
            fast_sum = math.fsum(float(fast[index]) for index in indices)
            slow_sum = math.fsum(float(slow[index]) for index in indices)
            choose_fast = fast_sum <= slow_sum
            slot_oracle[indices] = fast[indices] if choose_fast else slow[indices]
            oracle_rows[generation] = {"n": len(indices), "fast_loss_sum": fast_sum,
                "slow_loss_sum": slow_sum, "selected_expert": "FAST" if choose_fast else "SLOW",
                "selected_loss_sum": fast_sum if choose_fast else slow_sum}
        counters["G_slot_generations"] = len(oracle_rows)
        saved_oracles = result.get("slot_oracle", [])
        check("slot_oracle_unique_generations", len({int(row["slot_uid"]) for row in saved_oracles}) == len(saved_oracles))
        saved_by_generation = {int(row["slot_uid"]): row for row in saved_oracles}
        check("slot_oracle_generation_roster", set(saved_by_generation) == set(oracle_rows))
        for generation, expected in oracle_rows.items():
            actual = saved_by_generation.get(generation)
            if actual is None:
                continue
            check("slot_oracle_choice/" + str(generation),
                  actual.get("n") == expected["n"] and actual.get("selected_expert") == expected["selected_expert"])
            for field in ("fast_loss_sum", "slow_loss_sum", "selected_loss_sum"):
                close("slot_oracle/" + str(generation) + "/" + field, actual[field], expected[field], "oracle_loss")
        aggregate("route/G", "EVENT-ORACLE", gi.tolist(), event_oracle)
        aggregate("route/G", "SLOT-ORACLE", gi.tolist(), slot_oracle)
        oracle_bound("SLOT-ORACLE", slot_oracle, event_oracle, gi)
        oracle_total = math.fsum(float(event_oracle[index]) for index in gi)
        hard_group = groups.get("route/G", {}).get("SILENT-HARD", {})
        hard_total = float(hard_group.get("loss_sum", 0.0))
        headroom = hard_total - oracle_total
        counters["independent_event_oracle_G_loss_sum"] = oracle_total
        counters["oracle_capture_headroom_sum"] = headroom
        counters["positive_headroom_count"] = int(headroom > 1e-12)
        epsilon_loss = 2e-8 + 2e-12 * abs(oracle_total)
        for policy, metric in result.get("metrics_by_arm", {}).items():
            if "oracle_capture" not in metric:
                continue
            recorded_capture = metric["oracle_capture"]
            recorded_group = groups.get("route/G", {}).get(policy)
            if not check("oracle_capture_G_group/" + policy, recorded_group is not None):
                continue
            if headroom > 1e-12:
                counters["positive_headroom_capture_count"] += 1
                expected_capture = (hard_total - float(recorded_group["loss_sum"])) / headroom
                valid_capture = recorded_capture is not None and math.isfinite(float(recorded_capture))
                if valid_capture:
                    capture_difference = abs(float(recorded_capture) - expected_capture)
                    maxdiff["oracle_capture"] = max(maxdiff.get("oracle_capture", 0.0), capture_difference)
                    counters["max_raw_oracle_capture"] = (float(recorded_capture)
                        if counters["max_raw_oracle_capture"] is None
                        else max(counters["max_raw_oracle_capture"], float(recorded_capture)))
                else:
                    capture_difference = math.inf
                capture_tolerance = epsilon_loss / headroom
                formula_tolerance = capture_tolerance * (1.0 + abs(expected_capture))
                check("oracle_capture_formula/" + policy,
                      valid_capture and capture_difference <= formula_tolerance,
                      f"Capture difference {capture_difference:.17g}, scaled tolerance {formula_tolerance:.17g}")
                excess = max(0.0, float(recorded_capture) - 1.0) if valid_capture else math.inf
                if math.isfinite(excess):
                    counters["max_oracle_capture_excess"] = max(counters["max_oracle_capture_excess"], excess)
                check("oracle_capture_numerator_loss_bound/" + policy,
                      float(recorded_group["loss_sum"]) + epsilon_loss >= oracle_total,
                      f"Policy loss {recorded_group['loss_sum']}, oracle loss {oracle_total}")
                check("oracle_capture_at_most_one/" + policy, valid_capture and excess <= capture_tolerance,
                      f"Capture {recorded_capture}; upper bound 1 with scaled tolerance {capture_tolerance:.17g}")
                if policy == "EVENT-ORACLE":
                    check("event_oracle_capture_is_one",
                          valid_capture and abs(float(recorded_capture) - 1.0) <= capture_tolerance)
            else:
                check("oracle_capture_undefined_without_positive_headroom/" + policy,
                      recorded_capture is None, f"Headroom {headroom:.17g}")
        for field, expected in (("event_oracle_loss", event_oracle), ("slot_oracle_loss", slot_oracle)):
            if field in cols:
                close("flat_" + field, np.asarray(cols[field], dtype=float)[gi], expected[gi], "oracle_loss")
    except Exception as exc:
        check("independent_generation_oracles", False, f"{type(exc).__name__}: {exc}")

    mapping = {"shuffle_source_event_index", "shuffle_age_bin", "shuffle_support_bin",
               "shuffle_prior_use_bin", "shuffle_permutable", "shuffled_loss"}
    missing_mapping = sorted(mapping - set(cols))
    if not check("flat_G_mapping_present", not missing_mapping,
                 "Missing persisted mapping columns: " + ", ".join(missing_mapping)):
        return finish()
    try:
        if not check("mapping_column_lengths", all(len(cols[key]) == n for key in mapping)):
            return finish()
        # Compute the linear quantiles directly, independently of np.quantile.
        ages = np.asarray(cols["slot_age"], dtype=float)[gi]
        use_counts = np.asarray(cols["prior_similarity_use_count"], dtype=float)[gi]
        support = np.asarray(cols["support"], dtype=float)[gi]
        valid = check("G_stratum_features_valid", np.all(np.isfinite(ages)) and np.all(ages >= 0)
                      and np.all(np.isfinite(use_counts)) and np.all(use_counts >= 0)
                      and np.all(use_counts == np.floor(use_counts)) and np.all(support >= 2)
                      and np.all(support == np.floor(support)))
        if not valid:
            return finish()
        ordered_ages = np.sort(ages)
        edges = []
        for quantile in (0.25, 0.5, 0.75):
            if gn:
                position = quantile * (gn - 1)
                lower, upper = math.floor(position), math.ceil(position)
                fraction = position - lower
                edges.append(float(ordered_ages[lower] + fraction * (ordered_ages[upper] - ordered_ages[lower])))
        age_bins = np.searchsorted(np.asarray(edges), ages, side="right") if gn else np.zeros(0, dtype=int)
        support_bins = np.where(support == 2, 0, np.where(support <= 4, 1, 2))
        use_bins = np.where(use_counts <= 1, 0, np.where(use_counts <= 3, 1, 2))
        for field, expected in (("shuffle_age_bin", age_bins), ("shuffle_support_bin", support_bins),
                                ("shuffle_prior_use_bin", use_bins)):
            check("independent_bins/" + field, np.array_equal(np.asarray(cols[field])[gi], expected))
        strata = collections.defaultdict(list)
        for position in range(gn):
            strata[(int(support_bins[position]), int(age_bins[position]), int(use_bins[position]))].append(position)
        eligible = np.zeros(gn, dtype=bool)
        for members in strata.values():
            if len(members) > 1:
                eligible[members] = True
        counters.update(strata_count=len(strata), permutable_n=int(eligible.sum()), singleton_n=int((~eligible).sum()))
        stored_eligible = np.asarray(cols["shuffle_permutable"])[gi]
        check("G_permutable_membership", np.array_equal(stored_eligible, eligible.astype(int)))
        g_event_ids = event_ids[gi]
        event_to_position = {int(value): position for position, value in enumerate(g_event_ids)}
        source_event_ids = np.asarray(cols["shuffle_source_event_index"])[gi]
        if not check("sources_known_unique_G_events", len(set(source_event_ids.tolist())) == gn
                     and set(source_event_ids.tolist()) == set(g_event_ids.tolist())):
            return finish()
        source_positions = np.asarray([event_to_position[int(value)] for value in source_event_ids], dtype=int)
        original_positions = np.arange(gn)
        check("all_nonsingleton_events_deranged", np.all(source_positions[eligible] != original_positions[eligible]))
        check("singleton_sources_retained", np.all(source_positions[~eligible] == original_positions[~eligible]))
        check("sources_stay_in_all_three_bins", np.array_equal(support_bins, support_bins[source_positions])
              and np.array_equal(age_bins, age_bins[source_positions]) and np.array_equal(use_bins, use_bins[source_positions]))
        local_weights = weights["LOCAL-BAYES-E2"][gi]
        all_mass_preserved, all_multisets_preserved = True, True
        rng = np.random.default_rng(int(result["seed"]) * 100000 + int(result["load"]) + 4300000)
        prescribed_sources = np.arange(gn)
        for key, members in sorted(strata.items()):
            original = [float(local_weights[position]) for position in members]
            reassigned = [float(local_weights[source_positions[position]]) for position in members]
            all_mass_preserved &= math.fsum(original) == math.fsum(reassigned)
            all_multisets_preserved &= sorted(original) == sorted(reassigned)
            random_order = rng.permutation(np.asarray(members, dtype=np.int64)).tolist()
            shift = int(rng.integers(1, len(members))) if len(members) > 1 else 0
            for position, destination in enumerate(random_order):
                prescribed_sources[destination] = random_order[(position + shift) % len(members)]
        check("exact_fsum_mass_in_every_stratum", all_mass_preserved)
        check("exact_weight_multiset_in_every_stratum", all_multisets_preserved)
        check("dedicated_seed_derangement_reproduction", np.array_equal(source_positions, prescribed_sources))
        shuffled = np.zeros(n, dtype=float)
        shuffled[gi] = mix(local_weights[source_positions], gi)
        close("frozen_shuffled_event_losses", np.asarray(cols["shuffled_loss"], dtype=float)[gi], shuffled[gi], "event_loss")
        oracle_bound("SHUFFLED-ALPHA", shuffled, np.minimum(fast, slow), gi)
        aggregate("route/G", "SHUFFLED-ALPHA", gi.tolist(), shuffled)
        eligible_indices = gi[eligible].tolist()
        for policy in POLICIES:
            aggregate("G_permutable", policy, eligible_indices, reconstructed[policy], correct[policy])
        aggregate("G_permutable", "SHUFFLED-ALPHA", eligible_indices, shuffled)
        non_g = np.flatnonzero(route != "G")
        for field in ("shuffle_source_event_index", "shuffle_age_bin", "shuffle_support_bin",
                      "shuffle_prior_use_bin", "shuffle_permutable"):
            check("non_G_null_mapping/" + field, np.all(np.asarray(cols[field])[non_g] == -1))
        for field in ("shuffled_loss", "event_oracle_loss", "slot_oracle_loss"):
            if field in cols:
                check("non_G_null_loss/" + field, np.all(np.isnan(np.asarray(cols[field], dtype=float)[non_g])))
        for label in ("whole", "target", "nontarget"):
            selected = gi if label == "whole" else gi[category[gi] == label]
            delta = math.fsum(float(shuffled[index] - reconstructed["LOCAL-BAYES-E2"][index]) for index in selected)
            actual_local = groups[label]["LOCAL-BAYES-E2"]
            actual_shuffle = groups[label]["SHUFFLED-ALPHA"]
            check("whole_counterfactual_n/" + label, actual_shuffle["n"] == actual_local["n"])
            close("whole_counterfactual_G_delta/" + label,
                  actual_shuffle["loss_sum"], actual_local["loss_sum"] + delta)
            close("whole_counterfactual_bpb/" + label, actual_shuffle["bpb"],
                  (actual_local["loss_sum"] + delta) / actual_local["n"])
            check("whole_counterfactual_accuracy_unavailable/" + label,
                  actual_shuffle.get("correct") is None and actual_shuffle.get("accuracy") is None)
        recorded_audit = result.get("audits", {}).get("shuffled", {})
        check("saved_shuffle_audit_present", bool(recorded_audit))
        expected_audit = {"total_G_n": gn, "permutable_n": int(eligible.sum()),
            "singleton_n": int((~eligible).sum()), "strata_count": len(strata),
            "reassigned_count": int(np.sum(source_positions != original_positions)),
            "weights_different_count": int(np.sum(local_weights != local_weights[source_positions])),
            "multiset_preserved": all_multisets_preserved, "mass_preserved": all_mass_preserved,
            "derangement_preserved": bool(np.array_equal(source_positions != original_positions, eligible))}
        for key, expected in expected_audit.items():
            check("saved_shuffle_audit/" + key, recorded_audit.get(key) == expected)
        if gn:
            close("saved_quartile_edges", recorded_audit.get("age_quartile_edges", []), edges, "quartile_edge")
            close("saved_permutable_coverage", recorded_audit.get("coverage"), float(eligible.mean()))
        counters["shuffle_reassigned_count"] = expected_audit["reassigned_count"]
        counters["shuffle_weights_different_count"] = expected_audit["weights_different_count"]
        counters["Q6_permutable_pairs"] = len(eligible_indices)
    except Exception as exc:
        check("offline_mapping_reconstruction", False, f"{type(exc).__name__}: {exc}")
    return finish()
