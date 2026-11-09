"""Short WR2 technical checks, gated before the first learner import."""
from collections import Counter
import hashlib
import importlib.util
from itertools import product
import json
import os
from random import Random
from unittest.mock import patch

import e16wr2_common as common


SEED = 73129
BATCH = 64
REPLAY = 32
UPDATES = 20
ROOT = common.ROOT
OUT = common.OUT
FIXTURE_ENV = {
    "EXP45_HIDDEN": "512",
    "E8_BATCH": "64",
    "E8_WD": ".25",
    "E8_LR": ".001",
    "E8_REPLAY_SHARE": "1/2",
    "E8_BUFFER_CAP": "1024",
    "E8_SAVE_STAGES": "0",
    "E8_DIAGNOSTICS": "0",
    "E8_COMPOSE_CHECKPOINTS": "0",
    "E8_RUN_PREFIX": "checks_",
    "E8_RESERVOIR_CAP": "8",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_total_mechanics():
    from e16wr2_total import TotalReservoir

    class FixedReplacements:
        def __init__(self, replacements):
            self.replacements = iter(replacements)

        def randrange(self, seen):
            value = next(self.replacements)
            require(0 <= value < seen, "Replacement index outside seen range")
            return value

    # Enumerate the entire small sample space instead of estimating probabilities.
    subsets = Counter()
    for first, second in product(range(3), range(4)):
        total = TotalReservoir(SEED, 2)
        total.admit_rng = FixedReplacements([first, second])
        total.begin(1)
        total.observe_current([([0], [200]), ([1], [201])])
        require(not total.available(), "Current observations eligible before stage end")
        total.end()
        total.begin(2)
        total.observe_current([([2], [202]), ([3], [203])])
        require(all(prompt[0] < 2 for prompt, _ in total.draw(8)),
                "Current-stage observation leaked into replay")
        snapshot = total.end()
        require(snapshot["presentations_total"] == 4, "Replay advanced the admission counter")
        require(snapshot["retained_slots"] == 2, "Total cap violated")
        require(snapshot["transient_pending_records_peak"] == 2, "Transient peak not counted")
        subsets[tuple(sorted(prompt[0] for prompt, _, _ in total.rows))] += 1
    require(len(subsets) == 6 and set(subsets.values()) == {2}, "Algorithm R is nonuniform")

    first = TotalReservoir(SEED, 8)
    second = TotalReservoir(SEED, 8)
    for total, stage in ((first, 1), (second, 300)):
        total.begin(stage)
        total.observe_current([([7], [211, 256])] * 100)
        snapshot = total.end()
        require(snapshot["presentations_total"] == 100, "Duplicates lost from seen count")
        require(snapshot["retained_slots"] == 8, "Duplicate observations not admitted")
        require(snapshot["distinct_prompts"] == 1, "Duplicate prompt count incorrect")
        total.begin(stage + 1)
    require(first.draw(25) == second.draw(25), "Origin labels changed replayed bytes")
    require(first.draw_counts != second.draw_counts, "Diagnostic origin did not change")
    return {
        "cycle": 1, "status": "PASS", "algorithm_R_exact_small_enumeration": True,
        "duplicate_presentations_counted": True, "stage_end_eligibility": True,
        "diagnostic_origin_labels_do_not_change_replay": True,
    }


def run_life(folder, source, name, schedule, sampling, target, fixture_labels=False, small_pool=False):
    target_dir = folder / name
    target_dir.mkdir(exist_ok=False)
    (target_dir / f"e8_dev_{SEED}.pt").symlink_to(OUT / f"e8_dev_{SEED}.pt")
    env = dict(FIXTURE_ENV, E8_SCHEDULE=schedule, E8_REPLAY_SAMPLING=sampling,
               E8_REPLAY_TARGETS=target)
    with patch.dict(os.environ, env):
        spec = importlib.util.spec_from_file_location(f"wr2_technical_{name}", ROOT / source)
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        import e14_fast_inference as fast

        native, torch = model.X, model.T
        model.OUT = target_dir
        model.STAGE_UPDATES = 2
        model.COMPOSE_UPDATES = 4
        model.SAVE_STAGES = True
        # A fixture of 32 teaching rows exposes each row even after the replay split.
        # Hold the panel population fixed so its normal sample size remains valid.
        if small_pool:
            original_split = model.split

            def fixture_split(task, fraction, seed):
                training, panel = original_split(task, fraction, seed)
                return training[:REPLAY], panel

            model.split = fixture_split
        if fixture_labels:
            original_render = model.render

            def fixture_render(task, left, right):
                prompt, answer = original_render(task, left, right)
                # A deliberately non-arithmetic answer checks byte provenance.
                return prompt, [48 + ((answer[0] - 48 + 1) % 10), *answer[1:]]

            model.render = fixture_render

        def digest(value):
            if torch.is_tensor(value):
                raw = value.detach().cpu().contiguous().numpy().tobytes()
                return {"shape": list(value.shape), "dtype": str(value.dtype),
                        "sha256": hashlib.sha256(raw).hexdigest()}
            if isinstance(value, dict):
                # Store tie-breaking can depend on dictionary insertion order.
                return {"map": [[digest(key), digest(item)] for key, item in value.items()]}
            if isinstance(value, (list, tuple)):
                return [digest(item) for item in value]
            return value

        def state_hash(value):
            raw = json.dumps(digest(value), separators=(",", ":"), allow_nan=False).encode()
            return hashlib.sha256(raw).hexdigest()

        checkpoints, batches, trajectory, generated_replay = [], [], [], []
        original_update = native.update
        original_score = native.score
        scoring = False

        def score(model_, items, store=None):
            nonlocal scoring
            scoring = True
            try:
                return original_score(model_, items[:8], store)
            finally:
                scoring = False

        def respond(model_, prompts, store=None, cap=7):
            if not scoring:
                generated_replay.append(len(prompts))
            return fast.respond(model_, prompts, store, cap=cap)

        def update(model_, optimizer, items, local, store=None):
            require(local and len(items) == BATCH, "Fixture used wrong update or batch size")
            batches.append([(tuple(prompt), tuple(answer)) for prompt, answer in items])
            loss = original_update(model_, optimizer, items, local, store)
            trajectory.append({
                "model": state_hash(model_.state_dict()), "optimizer": state_hash(optimizer.state_dict()),
                "torch_rng": state_hash(torch.get_rng_state()), "loss": loss,
                "store": state_hash([store.mem, store.order]) if store is not None else None,
            })
            return loss

        def capture_checkpoint(state, path):
            checkpoints.append({key: state_hash(value) for key, value in state.items()})

        with patch.object(native, "score", score), patch.object(native, "respond", respond), \
                patch.object(native, "update", update), patch.object(torch, "save", capture_checkpoint):
            model.life("C", SEED)
        receipt_path = target_dir / f"{model.RUN_PREFIX}e8_life_C{model.TAG}_{SEED}.json"
        receipt = common.clean(common.read(receipt_path))
        require(len(batches) == UPDATES and len(checkpoints) == 10, "Fixture schedule incomplete")
        result = {
            "receipt": receipt,
            "trajectory": trajectory,
            "checkpoints": checkpoints,
            "batches": batches,
            "prompt_order_sha256": state_hash([[prompt for prompt, _ in items] for items in batches]),
            "generated_replay_calls": generated_replay,
            "sidecar": common.read(receipt_path.with_suffix(".replay.json")) if target == "observed" else None,
        }
        common.save(target_dir / "technical_trace.json", result)
        return result


def check_observed_batches(result, total_capacity=None):
    prior = {}
    reservoir = []
    presentations = 0
    cursor = 0
    draw_rng = Random(SEED + 9)
    admit_rng = Random(SEED * 1000003 + 1009)
    require(not result["generated_replay_calls"], "Observed arm generated a replay answer")
    sidecar = result["sidecar"]
    require(len(sidecar) == 9, "Missing replay sidecar stages")
    for stage, row in enumerate(result["receipt"]["stages"][1:], 1):
        dose = 4 if row["task"] == "compose" else 2
        current_count = BATCH if stage == 1 else BATCH - REPLAY
        observations = []
        origins = Counter()
        for items in result["batches"][cursor:cursor + dose]:
            observations.extend((p, t, stage) for p, t in items[:current_count])
            for prompt, answer in items[current_count:]:
                require(prompt in prior and answer == prior[prompt][0],
                        "Replayed answer differs from an actual prior observed answer")
                origin = prior[prompt][1]
                origins[origin] += 1
                if total_capacity is not None:
                    require((prompt, answer, origin) == draw_rng.choice(reservoir),
                            "Actual replay differs from independent total-reservoir reference")
        cursor += dose
        for prompt, answer, origin in observations:
            require(prior.get(prompt, (answer, origin)) == (answer, origin), "Ambiguous fixture provenance")
            prior[prompt] = (answer, origin)
            presentations += 1
            if total_capacity is not None:
                item = (prompt, answer, origin)
                if len(reservoir) < total_capacity:
                    reservoir.append(item)
                else:
                    index = admit_rng.randrange(presentations)
                    if index < total_capacity:
                        reservoir[index] = item
        metrics = sidecar[stage - 1]
        require(metrics["stage"] == stage and metrics["task"] == row["task"], "Sidecar stage differs")
        require(metrics["current_presentations"] == len(observations), "Current presentation count differs")
        require(metrics["presentations_total"] == presentations, "Cumulative presentation count differs")
        require(metrics["replay_count"] == sum(origins.values()), "Replay count differs")
        require(metrics["wrong_against_observed"] == 0, "Observed replay error count is nonzero")
        require({int(key): value for key, value in metrics["realized_draw_counts"].items()} == dict(origins),
                "Sidecar draw origins differ from actual inputs")
        if total_capacity is not None:
            require(metrics["retained_slots"] == len(reservoir), "Retained slot count differs from reference")
            require(metrics["distinct_prompts"] == len({p for p, _, _ in reservoir}),
                    "Distinct prompt count differs from reference")
            require({int(key): value for key, value in metrics["origin_counts"].items()}
                    == dict(Counter(origin for _, _, origin in reservoir)), "Reservoir origin counts differ")
            require(metrics["transient_pending_records_peak"] == len(observations), "Transient peak differs")
    require(cursor == UPDATES, "Replay validation did not cover every fixture update")


def check_receipt_guard(folder, result):
    validator_dir = folder / "validator"
    validator_dir.mkdir()
    selected = validator_dir / "exact_job.json"
    job = {"receipt": selected.name, "seed": SEED, "role": "A", "capacity": 8}
    common.save(selected, result["receipt"])
    common.save(selected.with_suffix(".replay.json"), result["sidecar"])
    (validator_dir / "unfinished_sibling.json").write_text("{")
    with patch.object(common, "OUT", validator_dir):
        common.validate_receipt(job, short=True, require_meta=False)
        incomplete = dict(result["receipt"])
        incomplete["stages"] = incomplete["stages"][:-1]
        common.save(selected, incomplete)
        try:
            common.validate_receipt(job, short=True, require_meta=False)
        except (AssertionError, ValueError, RuntimeError):
            return
    raise AssertionError("Validator accepted an incomplete exact-job receipt")


def main():
    common.verify_lock()
    gate = common.require_machine_free()
    cycles = [check_total_mechanics()]
    folder = OUT / "e16wr2_technical"
    folder.mkdir(exist_ok=True)

    old = run_life(folder, "exp8_e16wr.py", "old_default", "blocked", "uniform", "self")
    new = run_life(folder, "exp8_e16wr2.py", "new_default", "blocked", "uniform", "self")
    for key in ("receipt", "trajectory", "checkpoints", "batches", "generated_replay_calls"):
        require(old[key] == new[key], f"Default short parity differs in {key}")
    cycles.append({
        "cycle": 2, "status": "PASS", "updates": UPDATES, "scores_excluding_cpu_exact": True,
        "model_adam_torch_rng_python_replay_rng_store_buffers_loss_batch_targets_exact": True,
        "prompt_order_sha256": old["prompt_order_sha256"],
        "checkpoint_states": "HASHED_NOT_SAVED_AS_TENSORS",
    })

    old = run_life(folder, "exp8_e16wr.py", "blocked_self_fixture", "blocked", "uniform", "self",
                   fixture_labels=True, small_pool=True)
    observed = run_life(folder, "exp8_e16wr2.py", "blocked_observed_fixture", "blocked", "uniform", "observed",
                        fixture_labels=True, small_pool=True)
    require(old["prompt_order_sha256"] == observed["prompt_order_sha256"], "Part B input order changed")
    require(old["generated_replay_calls"] == [REPLAY] * (UPDATES - 2), "Old replay generation was not exercised")
    for left, right in zip(old["checkpoints"], observed["checkpoints"]):
        for key in ("replay_rng", "buffer", "buffer_ages"):
            require(left[key] == right[key], f"Part B changed {key}")
    check_observed_batches(observed)

    total = run_life(folder, "exp8_e16wr2.py", "total_observed_fixture", "late_family", "total", "observed",
                     fixture_labels=True)
    check_observed_batches(total, total_capacity=8)
    check_receipt_guard(folder, total)
    cycles.append({
        "cycle": 3, "status": "PASS", "updates_per_fixture": UPDATES,
        "part_b_old_self_vs_observed_prompt_order_sha256": observed["prompt_order_sha256"],
        "part_b_original_admission_and_replay_rng_and_prompt_buffers_exact": True,
        "all_observed_replay_labels_equal_actual_prior_fixture_labels": True,
        "deliberately_non_arithmetic_fixture_labels_preserved": True,
        "total_replay_and_counters_match_independent_stage_end_reference": True,
        "incomplete_exact_job_rejected_and_unfinished_sibling_ignored": True,
    })
    common.save(ROOT / "E16WR2_CHECKS.json", {
        "status": "PASS", "gate": gate, "cycles": cycles, "scientific_lives": 0,
        "scope": "Short model and mechanics parity, all nine stages, eight records per evaluation. "
                 "Part B target provenance fixture uses a 32-row operation pool and deliberately "
                 "non-arithmetic observed labels. No claim of full-life archived prompt parity.",
    })


if __name__ == "__main__":
    main()
