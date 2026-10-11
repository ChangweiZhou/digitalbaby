#!/usr/bin/env python3
"""Pure task/exposure checks; no native core, scientific runner or model learning.

Clean linear fits are environment-side solvability witnesses only. Synthetic
feedback fixtures are public technical data, never confirmation worlds.
"""

import argparse
import itertools
import json
import math
import random
from collections import Counter
from fractions import Fraction
from pathlib import Path


SYMBOLS = tuple(range(5))
ALL_ORBITS = frozenset(itertools.combinations_with_replacement(SYMBOLS, 2))


def orbit(a, b):
    return (min(a, b), max(a, b))


def families():
    paths = set()
    cycles = set()
    for p in itertools.permutations(SYMBOLS):
        paths.add(tuple(sorted(orbit(p[i], p[i + 1]) for i in range(4))))
        cycles.add(tuple(sorted(orbit(p[i], p[(i + 1) % 5]) for i in range(5))))
    path_loop = set()
    for edges in paths:
        for v in SYMBOLS:
            path_loop.add(tuple(sorted((*edges, (v, v)))))
    star_loop = {
        tuple(sorted(((c, c), *(orbit(c, v) for v in SYMBOLS if v != c))))
        for c in SYMBOLS
    }
    return {
        "PATH_LOOP": sorted(path_loop),
        "STAR_LOOP": sorted(star_loop),
        "ODD_CYCLE": sorted(cycles),
    }


def slots(mask):
    # A diagonal is deliberately duplicated, not silently given half exposure.
    return [(a, b) for u, v in mask for a, b in ((u, v), (v, u))]


def role_row(a, b, gauge=False):
    row = [int(i == a) for i in SYMBOLS]
    right = SYMBOLS[1:] if gauge else SYMBOLS
    return row + [int(i == b) for i in right]


def rref(matrix, coefficient_columns):
    m = [[Fraction(x) for x in row] for row in matrix]
    pivots = []
    pivot_row = 0
    for col in range(coefficient_columns):
        found = next((i for i in range(pivot_row, len(m)) if m[i][col]), None)
        if found is None:
            continue
        m[pivot_row], m[found] = m[found], m[pivot_row]
        scale = m[pivot_row][col]
        m[pivot_row] = [x / scale for x in m[pivot_row]]
        for i in range(len(m)):
            if i != pivot_row and m[i][col]:
                scale = m[i][col]
                m[i] = [x - scale * y for x, y in zip(m[i], m[pivot_row])]
        pivots.append(col)
        pivot_row += 1
        if pivot_row == len(m):
            break
    return m, pivots


def rank(matrix):
    return len(rref(matrix, len(matrix[0]))[1])


def clean_additive_fit(mask, meaning):
    # Fit only the ten teaching slots, fix the unidentifiable gauge R_0=0.
    matrix = [
        role_row(a, b, gauge=True) + [meaning[a] + meaning[b]]
        for a, b in slots(mask)
    ]
    reduced, pivots = rref(matrix, 9)
    if len(pivots) != 9:
        raise ValueError("training geometry is not identifiable")
    for row in reduced:
        if not any(row[:9]) and row[9]:
            raise ValueError("inconsistent clean teaching equations")
    solution = [Fraction(0) for _ in range(9)]
    for i, col in enumerate(pivots):
        solution[col] = reduced[i][9]
    return solution


def check_mask(mask):
    mask = frozenset(mask)
    held = ALL_ORBITS - mask
    assert len(mask) == 5
    assert len(held) == 10
    records = slots(sorted(mask))
    assert len(records) == 10
    assert {a for a, _ in records} == set(SYMBOLS)
    assert {b for _, b in records} == set(SYMBOLS)
    assert all(orbit(a, b) in mask for a, b in records)
    assert all((a, b) not in records and (b, a) not in records for a, b in held)
    assert rank([role_row(a, b) for a, b in records]) == 9
    return held


def nearest_digit(value):
    return min(range(9), key=lambda k: (abs(Fraction(k) - value), k))


def geometry_and_countermodels():
    groups = families()
    assert {k: len(v) for k, v in groups.items()} == {
        "PATH_LOOP": 300, "STAR_LOOP": 5, "ODD_CYCLE": 12
    }
    reports = {}
    all_meanings = list(itertools.permutations(SYMBOLS))
    witness_meanings = [(0, 1, 2, 3, 4), (4, 0, 3, 1, 2)]
    witness_case_count = 0
    for name, masks in groups.items():
        totals = Counter()
        tests = 0
        for mask in masks:
            held = check_mask(mask)
            for meaning in witness_meanings:
                fitted = clean_additive_fit(mask, meaning)
                for a in SYMBOLS:
                    for b in SYMBOLS:
                        prediction = sum(
                            x * w for x, w in zip(role_row(a, b, gauge=True), fitted)
                        )
                        assert prediction == meaning[a] + meaning[b]
                        witness_case_count += 1
            for meaning in all_meanings:
                delivered = [meaning[a] + meaning[b] for a, b in slots(mask)]
                counts = Counter(delivered)
                mode = min(counts, key=lambda y: (-counts[y], y))
                rounded_mean = nearest_digit(Fraction(sum(delivered), len(delivered)))
                table = {(a, b): meaning[a] + meaning[b] for a, b in slots(mask)}
                for a, b in held:
                    truth = meaning[a] + meaning[b]
                    assert (a, b) not in table and (b, a) not in table
                    totals["fixed_symbol_rank_addition"] += int(a + b == truth)
                    totals["target_mode_or_exact_table_mode_fallback"] += int(mode == truth)
                    totals["rounded_training_target_mean"] += int(rounded_mean == truth)
                    totals["zero_initial_prediction"] += int(truth == 0)
                    tests += 1
        reports[name] = {
            "masks": len(masks),
            "rank_all_masks": 9,
            "taught_orbits": 5,
            "held_out_orbits": 10,
            "epoch_records": 10,
            "clean_countermodel_cases": tests,
            "clean_countermodel_average_accuracy": {
                k: totals[k] / tests for k in sorted(totals)
            },
        }
    return {
        "families": reports,
        "all_masks_checked": sum(len(v) for v in groups.values()),
        "clean_fits_checked_per_mask": len(witness_meanings),
        "clean_fit_predictions_checked": witness_case_count,
        "all_120_meanings_countermodel_only": True,
        "scope": "exact geometry plus clean witness/countermodels; not learner efficacy",
    }


def feedback_stream(mask, meaning, regime, seed):
    # Separate synthetic namespaces make the fixture reproducible without
    # conditioning any byte draws on an arm's behavior.
    order_rng = random.Random(seed + 1000)
    noise_rng = random.Random(seed + 2000)
    wrong_rng = random.Random(seed + 3000)
    p_correct, p_wrong, _ = regime
    records = []
    for epoch in range(32):
        ordered = slots(mask)
        order_rng.shuffle(ordered)
        for a, b in ordered:
            truth = meaning[a] + meaning[b]
            draw = noise_rng.random()
            if draw < p_correct:
                kind, target = "correct", truth
            elif draw < p_correct + p_wrong:
                kind = "wrong"
                target = wrong_rng.choice([y for y in range(9) if y != truth])
            else:
                kind, target = "missing", None
            records.append({
                "epoch": epoch, "pair": (a, b), "kind": kind,
                "truth_private": truth, "delivered": target,
            })
    return records


def shuffled_history(stream, seed):
    rng = random.Random(seed + 4000)
    result = []
    for start in range(0, len(stream), 10):
        epoch = [dict(row) for row in stream[start:start + 10]]
        present_indices = [i for i, row in enumerate(epoch) if row["delivered"] is not None]
        labels = [epoch[i]["delivered"] for i in present_indices]
        rng.shuffle(labels)
        for i, label in zip(present_indices, labels):
            epoch[i]["delivered"] = label
        result.extend(epoch)
    return result


def allowed_learner_payload(row):
    # Private truth/kind are retained only in the fixture environment.
    return {"pair_byte_ids": row["pair"], "delivered": row["delivered"]}


def exposure_checks(contract):
    groups = families()
    snapshots = {}
    for j, name in enumerate(("STANDARD", "BLURRY")):
        regime = contract["feedback"][name]
        mask = groups["PATH_LOOP"][j * 37]
        held = check_mask(mask)
        probabilities = tuple(regime[k] for k in ("correct", "wrong", "missing"))
        assert abs(sum(probabilities) - 1) < 1e-12
        stream = feedback_stream(mask, (4, 0, 3, 1, 2), probabilities, 170 + j)
        shuf = shuffled_history(stream, 170 + j)
        assert len(stream) == 320
        for n in contract["old_sample_checkpoints"]:
            assert n % 10 == 0
            assert all(orbit(*row["pair"]) not in held for row in stream[:n])
            assert Counter(row["delivered"] for row in stream[:n]) == Counter(
                row["delivered"] for row in shuf[:n]
            )
            assert [row["delivered"] is None for row in stream[:n]] == [
                row["delivered"] is None for row in shuf[:n]
            ]
            assert stream[:n] == stream[:320][:n]
        for row in stream:
            payload = allowed_learner_payload(row)
            assert set(payload) == {"pair_byte_ids", "delivered"}
            assert row["kind"] != "wrong" or row["delivered"] != row["truth_private"]
            assert row["kind"] != "missing" or row["delivered"] is None
        snapshots[name] = {
            "synthetic_records": len(stream),
            "realized_feedback": dict(Counter(row["kind"] for row in stream)),
            "actual_training_bytes": sum(5 + (row["delivered"] is not None) for row in stream),
            "held_out_teacher_exposures": 0,
            "shuffled_prefix_histograms_equal": True,
        }
    n_values = contract["old_sample_checkpoints"]
    totals = [n + contract["new_samples_per_continuation"] for n in n_values]
    assert totals == contract["persistent_total_sample_checkpoints"]
    assert 320 + len(n_values) * 80 == contract["records_per_history"] == 800
    assert len(contract["causal_histories"]) * 800 == contract["records_per_full_job"] == 2400
    assert 3 * (1 + len(n_values)) == contract["state_trajectories_per_full_job"] == 21
    assert len(contract["roster"]) == 9
    assert sum(r["role"] != "reference" for r in contract["roster"]) == 7
    dev_jobs = 12 * 9
    confirm_jobs = 96 * 5
    jobs = dev_jobs + confirm_jobs
    records = jobs * 2400
    assert dev_jobs == contract["development"]["maximum_full_jobs"] == 108
    assert confirm_jobs == contract["confirmation"]["maximum_full_jobs"] == 480
    assert jobs == contract["conditional_maximum"]["full_jobs"] == 588
    assert records == contract["conditional_maximum"]["external_records"] == 1411200
    assert records * 6 == contract["conditional_maximum"]["upper_bound_training_bytes"] == 8467200
    assert set(contract["answer_alphabet"]) == set("012345678")
    return {
        "synthetic_streams": snapshots,
        "maximum_jobs": jobs,
        "maximum_executed_external_records": records,
        "maximum_training_bytes": records * 6,
        "conceptual_checkpoints_reuse_parent_prefix": True,
        "native_training_records_executed_by_this_check": 0,
    }


def curve_checks(contract):
    totals = contract["persistent_total_sample_checkpoints"]
    xs = [math.log2(t) for t in totals]
    span = xs[-1] - xs[0]
    weights = [0.0] * len(xs)
    for i in range(len(xs) - 1):
        half_interval = (xs[i + 1] - xs[i]) / (2 * span)
        weights[i] += half_interval
        weights[i + 1] += half_interval
    assert all(w > 0 for w in weights)
    assert abs(sum(weights) - 1) < 1e-14
    assert abs(sum(w * 0.5 for w in weights) - 0.5) < 1e-14
    assert abs(sum(w * (1 if i < 3 else 0) for i, w in enumerate(weights))) < 1
    return {
        "total_external_samples": totals,
        "log2_trapezoid_weights": weights,
        "weight_sum": sum(weights),
        "constant_curve_preserved": True,
        "every_checkpoint_has_positive_weight": True,
    }


def expect_rejection(fn):
    try:
        fn()
    except (AssertionError, ValueError):
        return True
    raise AssertionError("negative fixture incorrectly passed")


def negative_checks():
    even_cycle_disconnected = [(0, 1), (1, 2), (2, 3), (0, 3), (4, 4)]
    bipartite_connected = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3)]
    missing_symbol = [(0, 0), (0, 1), (1, 2), (2, 3), (1, 3)]
    leaked_mask = families()["STAR_LOOP"][0]
    held = sorted(ALL_ORBITS - frozenset(leaked_mask))

    def reverse_leak():
        teacher_keys = set(slots(leaked_mask))
        a, b = next(pair for pair in held if pair[0] != pair[1])
        teacher_keys.add((b, a))
        assert all((x, y) not in teacher_keys and (y, x) not in teacher_keys for x, y in held)

    def concealed_later_samples():
        assert [10, 20, 40, 80, 160, 320] == [90, 100, 120, 160, 240, 400]

    def count_replays_as_new_data():
        external_records = 320
        internal_replays = 319
        assert external_records + internal_replays == external_records

    def hidden_truth_api():
        assert set({"pair_byte_ids": (0, 1), "delivered": 3, "truth_private": 3}) == {
            "pair_byte_ids", "delivered"
        }

    return {
        "missing_symbol_coverage": expect_rejection(lambda: check_mask(missing_symbol)),
        "disconnected_even_cycle_plus_loop": expect_rejection(lambda: check_mask(even_cycle_disconnected)),
        "connected_bipartite_rank_deficiency": expect_rejection(lambda: check_mask(bipartite_connected)),
        "reverse_orientation_teacher_leak": expect_rejection(reverse_leak),
        "omitting_80_later_samples": expect_rejection(concealed_later_samples),
        "replay_counted_as_external_information": expect_rejection(count_replays_as_new_data),
        "private_truth_in_public_payload": expect_rejection(hidden_truth_api),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("BENCHMARK_FIXTURES.json"))
    args = parser.parse_args()
    contract = json.loads(Path(__file__).with_name("BENCHMARK_CONTRACT.json").read_text())
    report = {
        "status": "PASS_BENCHMARK_GEOMETRY_AND_EXPOSURE_ONLY",
        "protocol": contract["protocol"],
        "native_science_jobs": 0,
        "native_learner_runs": 0,
        "geometry": geometry_and_countermodels(),
        "exposure": exposure_checks(contract),
        "curve": curve_checks(contract),
        "negative_cases": negative_checks(),
        "not_qualified": contract["qualification_required"],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "geometry_masks": report["geometry"]["all_masks_checked"],
        "clean_witness_predictions": report["geometry"]["clean_fit_predictions_checked"],
        "negative_cases_rejected": len(report["negative_cases"]),
        "native_learner_runs": 0,
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
