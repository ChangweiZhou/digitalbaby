"""E0 one-target algebra fixtures on disposable copies of saved V147 models.

No task generator, new world, trajectory, teach(), generation, or held query.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import time
START = time.process_time()
WALL = time.monotonic()
import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SAVED = ROOT / "minifly_v147_dev_20261024"
sys.path.insert(0, str(SAVED))
import kernel  # Saved class definitions only; no run/cycle/data generator imports.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def state(model):
    return (model.digest(), model.arrays_hash(), model.records, model.targets)


def feature(model, vector):
    indices = model.code(vector)[1]
    out = np.zeros(model.theta.shape[1])
    out[indices] = 1.0 / np.sqrt(len(indices))
    return out


def probe(model, vectors):
    before = state(model)
    value = np.asarray([model.margins(vector) for vector in vectors])
    assert before == state(model), "Forward probe mutated persistent state"
    return value


roster = [f"V147-D{d}_FIXED_{alpha}" for d in (1, 2) for alpha in ("0.003", "0.03")]
roster += [f"V147-D{d}_LOCAL_0.03" for d in (1, 2)]
input_paths = [SAVED / f"{ident}.pkl" for ident in roster]
input_paths += [SAVED / f"V147-D{d}_WORLD.json" for d in (1, 2)]
source_paths = {Path(module.__file__).resolve() for module in list(sys.modules.values())
                if getattr(module, "__file__", None)
                and str(module.__file__).endswith(".py")
                and Path(module.__file__).resolve().is_relative_to(ROOT)}
source_paths.add(Path(__file__).resolve())
input_paths += sorted(source_paths)
hashes_before = {str(path.relative_to(ROOT)): sha(path) for path in input_paths}
assert "v147_data" not in sys.modules

ledger = {"attempted_fixture_target_updates": 0, "completed_fixture_target_updates": 0,
          "failed_fixture_target_updates": 0, "external_record_applications": 0,
          "replayed_record_applications": 0, "new_worlds": 0, "held_inputs_used": 0}
write("FIXTURE_LEDGER.json", ledger)
fixtures = []
saved_contexts = []
try:
    for ident in roster:
        model, trace, rng, passes = pickle.loads((SAVED / f"{ident}.pkl").read_bytes())
        assert passes == 256
        assert len(trace.cache) < 4096, "Cannot certify original cache insertion order after eviction"
        original_state = state(model)
        original_cache_hash = hashlib.sha256(pickle.dumps(trace, protocol=4)).hexdigest()
        world_name = ident.split("_", 1)[0]
        # Only membership in the already saved TRAIN list is used. No held entry is queried.
        saved_world = json.loads((SAVED / f"{world_name}_WORLD.json").read_text())
        sigma, trained_pairs = saved_world["sigma"], saved_world["train"]
        trained_prompts = {bytes(sigma[index] for index in pair) for pair in trained_pairs}
        trained_prompts |= {prompt[::-1] for prompt in trained_prompts}
        ordered_prompts = []
        for prompt, prefix in trace.cache:
            if prefix == b"" and prompt not in ordered_prompts:
                assert prompt in trained_prompts
                ordered_prompts.append(prompt)
                if len(ordered_prompts) == 2:
                    break
        contexts = []
        for prompt in ordered_prompts:
            # Before any saved score(), the first pass inserts empty-prefix and observed
            # one-byte-prefix contexts in teach() order. No eviction occurred (<4096).
            observed_prefix = next(prefix for pp, prefix in trace.cache
                                   if pp == prompt and len(prefix) == 1)
            contexts += [(prompt, b"", observed_prefix[0]), (prompt, observed_prefix, 256)]
        assert len(contexts) == 4
        vectors = [trace.cache[(prompt, prefix)].copy() for prompt, prefix, _ in contexts]
        saved_contexts.append({"fit": ident, "cache_size": len(trace.cache),
                               "queries": [{"prompt_hex": p.hex(), "prefix_hex": q.hex(),
                                            "observed_target": target}
                                           for p, q, target in contexts]})
        before_margins = probe(model, vectors)
        old_phi = np.asarray([feature(model, vector) for vector in vectors])
        assert original_state == state(model)
        for teacher_index in (0, 1):
            # Each event starts from the same saved final model; events never accumulate.
            disposable = copy.deepcopy(model)
            teacher_vector = vectors[teacher_index]
            target = contexts[teacher_index][2]
            signs = np.asarray([1 if target & (1 << bit) else -1 for bit in range(9)])
            margin_teacher = before_margins[teacher_index]
            residual = (np.tanh(margin_teacher / 2) - signs) / 2
            warmup = 1 / 16 + 15 / 16 * min(1, model.records / 1024)
            overlaps = old_phi @ old_phi[teacher_index]
            fixed_term = -2 * model.alpha * warmup * overlaps[:, None] * residual[None, :]
            theta_prediction = model.theta - model.alpha * warmup * np.outer(residual, old_phi[teacher_index])
            ledger["attempted_fixture_target_updates"] += 1
            write("FIXTURE_LEDGER.json", ledger)
            try:
                disposable.observe(teacher_vector, target)
            except BaseException:
                ledger["failed_fixture_target_updates"] += 1
                write("FIXTURE_LEDGER.json", ledger)
                raise
            ledger["completed_fixture_target_updates"] += 1
            write("FIXTURE_LEDGER.json", ledger)
            after_margins = probe(disposable, vectors)
            new_phi = np.asarray([feature(disposable, vector) for vector in vectors])
            moving_term = 2 * (new_phi - old_phi) @ disposable.theta.T
            actual_delta = after_margins - before_margins
            predicted_delta = fixed_term + moving_term
            theta_error = float(np.max(np.abs(disposable.theta - theta_prediction)))
            equation_error = float(np.max(np.abs(actual_delta - predicted_delta)))
            feature_invariant = np.array_equal(model.W, disposable.W) and np.array_equal(model.b, disposable.b)
            if "_FIXED_" in ident:
                assert feature_invariant and np.array_equal(old_phi, new_phi)
            assert equation_error < 1e-11 and theta_error < 1e-11
            assert disposable.records == model.records
            assert disposable.targets == model.targets + 1
            assert state(model) == original_state
            fixtures.append({"fit": ident, "teacher_context": teacher_index,
                             "teacher_prompt_hex": contexts[teacher_index][0].hex(),
                             "teacher_prefix_hex": contexts[teacher_index][1].hex(),
                             "observed_target": target, "warmup": warmup,
                             "query_count": len(contexts), "query_kernel_overlaps": overlaps.tolist(),
                             "max_equation_absolute_error": equation_error,
                             "max_theta_update_absolute_error": theta_error,
                             "features_invariant": feature_invariant,
                             "changed_query_codes": int(np.sum(np.any(old_phi != new_phi, axis=1))),
                             "max_abs_fixed_term": float(np.max(np.abs(fixed_term))),
                             "max_abs_moving_term": float(np.max(np.abs(moving_term))),
                             "max_abs_actual_margin_change": float(np.max(np.abs(actual_delta))),
                             "actual_margin_changes": actual_delta.tolist(),
                             "fixed_terms": fixed_term.tolist(), "moving_terms": moving_term.tolist()})
        assert state(model) == original_state
        assert hashlib.sha256(pickle.dumps(trace, protocol=4)).hexdigest() == original_cache_hash
except BaseException as exc:
    write("FAILURE.json", {"error": repr(exc), "ledger": ledger, "cpu_seconds": time.process_time() - START})
    raise

hashes_after = {str(path.relative_to(ROOT)): sha(path) for path in input_paths}
assert hashes_before == hashes_after, "Historical source or saved-input hash changed"
assert ledger["attempted_fixture_target_updates"] == ledger["completed_fixture_target_updates"] == 12
assert ledger["failed_fixture_target_updates"] == 0
results = {"status": "E0_ONE_UPDATE_IDENTITIES_VERIFIED", "evidence_level": "E0",
           "scope": "Counterfactual single-target fixtures on disposable final calibration states; no learning efficacy claim",
           "ledger": ledger, "fits": roster, "contexts": saved_contexts, "fixtures": fixtures,
           "max_equation_absolute_error": max(row["max_equation_absolute_error"] for row in fixtures),
           "max_theta_update_absolute_error": max(row["max_theta_update_absolute_error"] for row in fixtures),
           "source_and_input_hashes_unchanged": True, "hashes": hashes_before,
           "original_model_and_trace_state_unchanged": True, "forward_probes_no_persistent_mutation": True,
           "held_inputs_scored": False, "world_generator_imported": "v147_data" in sys.modules,
           "warmup_scope": "All saved states are after warmup; ramp formula derived from code, not experimentally sampled",
           "cpu_seconds_including_imports_after_stdlib_timer": time.process_time() - START,
           "wall_seconds": time.monotonic() - WALL,
           "preflight_cost_boundary": "Earlier read-only saved-cache inspection took 2.9 tool wall seconds; its process CPU was not separately measured"}
write("RESULTS.json", results)
print(json.dumps({key: results[key] for key in ("status", "ledger", "max_equation_absolute_error", "max_theta_update_absolute_error", "cpu_seconds_including_imports_after_stdlib_timer")}, indent=2))
