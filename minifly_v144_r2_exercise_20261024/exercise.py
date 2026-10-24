"""Run three bounded V144-R2 preparation cycles; never opens CAP-H worlds."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import copy
import json
import math
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from r2_core import (HERE, ROOT, PARENT, MODES, Reference, admit, canonical, controls, digest,
                     expected_counts, reference_score, reduce_roster, seed, shortlist,
                     split_digest, stage_a_roster, task, trace_from_parent,
                     validate_result, write_json)
from restore_verify import verify

PARENT_COMMIT = "4c12621d0804d2495c2eacaebf4e17aaeb994771"
WORLD = "V144-R2-EXERCISE-C2"
PASSES = 4
DOSE = 40
HIDDEN = 128
LR = .03
DECAY = .001


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def source_lock():
    head = git_output("rev-parse", "HEAD")
    files = set()
    for module in sys.modules.values():
        raw = getattr(module, "__file__", None)
        if raw:
            path = Path(raw).resolve()
            if path.is_relative_to(ROOT) and path.suffix == ".py":
                files.add(path)
    files |= set(HERE.glob("*.py"))
    files |= {PARENT / "reference.py", PARENT / "task.py",
              ROOT / "minifly_a_v3/package/FULL151_CANONICAL_B.npz",
              ROOT / "minifly_a_v3/package/MANIFEST.json",
              ROOT / "minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json",
              PARENT / "NEXT_EXPERIMENT_DESIGN_R2.md"}
    entries = {}
    for path in sorted(files):
        if not path.is_file():
            raise FileNotFoundError(path)
        rel = str(path.relative_to(ROOT))
        ref = None
        for candidate in (head, PARENT_COMMIT):
            if subprocess.run(["git", "cat-file", "-e", f"{candidate}:{rel}"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                if subprocess.check_output(["git", "show", f"{candidate}:{rel}"], cwd=ROOT) == path.read_bytes():
                    ref = candidate
                    break
        entries[rel] = {"sha256": digest(path),
                        "origin": "pinned_git" if ref else ("new_bundle_source" if path.suffix in (".py", ".md") else "owner_asset")}
        if ref:
            entries[rel]["git_ref"] = ref
    import numba, scipy
    return {"schema": "V144_R2_SOURCE_LOCK_V1", "source_commit": head,
            "parent_commit": PARENT_COMMIT, "files": entries,
            "runtime": {"python": sys.version, "numpy": np.__version__,
                        "numba": numba.__version__, "scipy": scipy.__version__,
                        "numerical_threads": 1}}


def settings(mode):
    return {"world": WORLD, "mode": mode, "hidden": HIDDEN, "lr": LR,
            "decay": DECAY, "dose": DOSE, "passes": PASSES}


def spec(mode):
    return {"id": "DEV_" + mode, "settings": settings(mode),
            "checkpoints": ["birth", "pass1", "pass4"]}


def manifest():
    return json.loads((HERE / "MANIFEST.json").read_text())


def lock_hash():
    return digest(HERE / "SOURCE_LOCK.json")


def synthetic_oracle_result(world):
    class Oracle:
        def generate(self, prompt, cap=8):
            pair = (prompt[0] - 48, prompt[2] - 48)
            return task.answer(pair), "EOS"
    oracle = Oracle()
    held = task.oriented(world["held"])
    taught = task.oriented(world["doses"][str(DOSE)])
    queries = sum(len(task.answer(pair)) + 1 for pair in held)
    snap = {"label": "birth", "records": 0, "observed_target_tokens": 0,
            "cpu_s": 0.0, "state_hash": "0" * 64,
            "held": task.score(oracle, held), "taught": task.score(oracle, taught),
            "teacher_forced_diagnostic": {"teacher_forced_token_accuracy": 1.0,
                                          "teacher_forced_BCE_per_token": 0.0,
                                          "evaluator_target_queries": queries}}
    ss = {"id": "SYNTHETIC_ORACLE", "settings": dict(settings("BP"), passes=0),
          "checkpoints": ["birth"]}
    row = {"schema": "V144_R2_REFERENCE_RESULT_V1", "id": ss["id"],
           "complete": True, "settings": ss["settings"],
           "split_sha256": split_digest(world), "source_lock_sha256": lock_hash(),
           "records": 0, "observed_target_tokens": 0, "cpu_s": 0.0,
           "peak_rss_bytes": 0, "metrics": {"snapshots": [snap]}}
    return ss, row


def negative_fixtures(row, ss, world):
    mutations = {}
    bad = copy.deepcopy(row); del bad["metrics"]["snapshots"][0]["held"]["rows"]
    mutations["missing_nested_rows"] = bad
    bad = copy.deepcopy(row); bad["metrics"]["snapshots"][0]["held"]["orbit_exact"] = float("nan")
    mutations["nan"] = bad
    bad = copy.deepcopy(row); bad["metrics"]["snapshots"][0]["held"]["orbit_exact"] = float("inf")
    mutations["infinity"] = bad
    bad = copy.deepcopy(row); bad["metrics"]["snapshots"][0]["held"]["orbit_exact"] = 1.1
    mutations["out_of_range"] = bad
    bad = copy.deepcopy(row); bad["settings"]["world"] = "wrong"
    mutations["wrong_identity"] = bad
    bad = copy.deepcopy(row); bad["metrics"]["snapshots"].append(copy.deepcopy(bad["metrics"]["snapshots"][0]))
    mutations["checkpoint_count"] = bad
    bad = copy.deepcopy(row); bad["records"] = 1
    mutations["exposure_total"] = bad
    bad = copy.deepcopy(row); bad["metrics"]["snapshots"][0]["held"]["rows"][0]["exact"] = False
    mutations["wrong_row_score"] = bad
    bad = copy.deepcopy(row); bad["source_lock_sha256"] = "x" * 64
    mutations["source_identity"] = bad
    bad = copy.deepcopy(row); bad["split_sha256"] = "x" * 64
    mutations["split_identity"] = bad
    outcome = {}
    for name, bad in mutations.items():
        problems = validate_result(bad, ss, world, lock_hash())
        if not problems:
            raise AssertionError("accepted mutation: " + name)
        outcome[name] = problems
    return outcome


def cycle1():
    t0 = time.process_time()
    trace = trace_from_parent()
    sys.path.insert(0, str(ROOT / "minifly_learned_bank_v1_20261024"))
    import lb1
    for token in range(257):
        assert decode_roundtrip(token)
    for prompt in (b"5+2=", b"\x00 =\xff", b""):
        assert np.array_equal(trace.context(prompt), lb1.trace_for(prompt))
    world = task.world(WORLD)
    assert len(world["held"]) == 15 and not set(map(tuple, world["held"])) & set(map(tuple, world["training_order"]))
    assert task.coverage(world["doses"]["20"]) >= 2
    assert task.coverage(world["doses"]["40"]) >= 4
    lock = source_lock()
    write_json(HERE / "SOURCE_LOCK.json", lock)
    assert verify(HERE / "SOURCE_LOCK.json")["status"] == "QUALIFIED"
    write_json(HERE / "MANIFEST.json", {"schema": "V144_R2_EXERCISE_MANIFEST_V1",
                                      "world": world, "split_sha256": split_digest(world),
                                      "source_lock_sha256": lock_hash(),
                                      "roster": [spec(mode) for mode in MODES],
                                      "formal_screen_roster_stage_A_count": len(stage_a_roster()),
                                      "holdbacks_opened": False})
    ss, row = synthetic_oracle_result(world)
    assert not validate_result(row, ss, world, lock_hash())
    mutations = negative_fixtures(row, ss, world)
    grid = [{"hidden": h, "lr": lr, "decay": d, "held_exact": 0.0,
             "teacher_loss": 1.0 - lr, "cpu_s": h / 128}
            for h in (128, 256, 512) for lr in (.003, .01, .03) for d in (0., .001, .01)]
    assert {r["hidden"] for r in shortlist(grid)} == {128, 256, 512}
    assert len(stage_a_roster()) == 54 and not admit(21600, 1) and admit(21599, 1)
    try:
        admit(1, float("nan"))
        raise AssertionError("NaN budget accepted")
    except ValueError:
        pass
    write_json(HERE / "CYCLE1_CHECKS.json", {"status": "PASS", "source_files": len(lock["files"]),
                                             "source_lock_sha256": lock_hash(),
                                             "negative_fixtures": mutations,
                                             "stage_A_cells": len(stage_a_roster()),
                                             "holdbacks_opened": False,
                                             "process_cpu_s": time.process_time() - t0})
    print("Cycle 1 PASS: lock, byte interface, split, schema mutations, roster and admission")


def decode_roundtrip(token):
    from r2_core import bits, decode
    return decode(bits(token)) == token


def run_mode(mode, world, trace):
    out = HERE / "results" / ("DEV_" + mode)
    if out.exists():
        raise RuntimeError("immutable result already exists: " + str(out))
    out.mkdir(parents=True)
    model = Reference(trace, WORLD, HIDDEN, LR, DECAY, mode)
    rng = np.random.default_rng(seed("order:" + WORLD))
    pairs = task.oriented(world["doses"][str(DOSE)])
    t0 = time.process_time()
    snapshots = [reference_score(model, world, world["doses"][str(DOSE)], "birth", 0.0)]
    for ep in range(1, PASSES + 1):
        for idx in rng.permutation(len(pairs)):
            pair = pairs[idx]
            model.train(task.prompt(pair), task.answer(pair))
        if mode == "BP" and ep == 2:
            (out / "PASS2_LOCAL_ONLY.pkl").write_bytes(pickle.dumps((model, rng.bit_generator.state), protocol=5))
        if ep in (1, PASSES):
            snapshots.append(reference_score(model, world, world["doses"][str(DOSE)],
                                             "pass" + str(ep), time.process_time() - t0))
        if time.process_time() - t0 > 180:
            raise RuntimeError("exercise mode CPU ceiling exceeded")
    result = {"schema": "V144_R2_REFERENCE_RESULT_V1", "id": "DEV_" + mode,
              "complete": True, "settings": settings(mode),
              "split_sha256": split_digest(world), "source_lock_sha256": lock_hash(),
              "records": model.records, "observed_target_tokens": model.target_tokens,
              "cpu_s": time.process_time() - t0,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "metrics": {"snapshots": snapshots},
              "state_hash_at_finish": model.state_hash()}
    problems = validate_result(result, spec(mode), world, lock_hash())
    if problems:
        raise AssertionError(problems)
    write_json(out / "RESULT.json", result)
    return result


def cycle2():
    t0 = time.process_time()
    assert verify(HERE / "SOURCE_LOCK.json")["status"] == "QUALIFIED"
    m = manifest()
    assert not m["holdbacks_opened"] and m["source_lock_sha256"] == lock_hash()
    world = m["world"]
    trace = trace_from_parent()
    baseline = controls(world, DOSE, trace)
    write_json(HERE / "CONTROL_METRICS.json", baseline)
    summary = {}
    for mode in MODES:
        result = run_mode(mode, world, trace)
        final = result["metrics"]["snapshots"][-1]
        summary[mode] = {"records": result["records"],
                         "target_tokens": result["observed_target_tokens"],
                         "cpu_s": result["cpu_s"],
                         "held_complete_orbit": final["held"]["orbit_exact"],
                         "held_payload_orbit": sum(
                             sum(row["payload"] for row in group) / len(group)
                             for group in grouped(final["held"]["rows"]).values()) / 15,
                         "taught_complete_orbit": final["taught"]["orbit_exact"]}
    write_json(HERE / "CYCLE2_RESULTS.json", {"status": "COMPLETE_DEVELOPMENT_FIXTURE",
                                              "world": WORLD, "modes": summary,
                                              "controls": {k: v["orbit_complete"] for k, v in baseline.items()},
                                              "process_cpu_s": time.process_time() - t0,
                                              "holdbacks_opened": False})
    print("Cycle 2 PASS: three fresh 4-pass development reference fits")


def grouped(rows):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(sorted(row["pair"])), []).append(row)
    return groups


def cycle3():
    t0 = time.process_time()
    assert verify(HERE / "SOURCE_LOCK.json")["status"] == "QUALIFIED"
    m = manifest(); world = m["world"]
    reduced = reduce_roster(HERE / "results", m["roster"], {WORLD: world}, lock_hash())
    assert reduced["status"] == "COMPLETE", reduced
    partial = reduce_roster(HERE / "results", m["roster"] + [dict(spec("BP"), id="MISSING")],
                            {WORLD: world}, lock_hash())
    assert partial["status"] == "INCOMPLETE" and partial["missing"] == ["MISSING"] and not partial["results"]
    duplicate = reduce_roster(HERE / "results", m["roster"] + [m["roster"][0]],
                              {WORLD: world}, lock_hash())
    assert duplicate["status"] == "INCOMPLETE" and "roster" in duplicate["invalid"]
    fewer = reduce_roster(HERE / "results", m["roster"][:-1], {WORLD: world}, lock_hash())
    assert fewer["status"] == "INCOMPLETE" and fewer["extra"] == ["DEV_FROZEN"]
    bad_lock = json.loads((HERE / "SOURCE_LOCK.json").read_text())
    rel = next(iter(bad_lock["files"]))
    bad_lock["files"][rel]["sha256"] = "0" * 64
    write_json(HERE / "BAD_LOCK_FIXTURE.json", bad_lock)
    assert verify(HERE / "BAD_LOCK_FIXTURE.json")["status"] == "INVALID"
    # Pickle refers to the parent's fastmem module; import its closure first.
    trace = trace_from_parent()
    checkpoint = HERE / "results/DEV_BP/PASS2_LOCAL_ONLY.pkl"
    model, state = pickle.loads(checkpoint.read_bytes())
    resumed_rng = np.random.default_rng(); resumed_rng.bit_generator.state = state
    whole = Reference(trace, WORLD, HIDDEN, LR, DECAY, "BP")
    rng = np.random.default_rng(seed("order:" + WORLD))
    pairs = task.oriented(world["doses"][str(DOSE)])
    for ep in range(1, PASSES + 1):
        order = rng.permutation(len(pairs))
        if ep > 2:
            assert np.array_equal(order, resumed_rng.permutation(len(pairs)))
        for idx in order:
            pair = pairs[idx]; whole.train(task.prompt(pair), task.answer(pair))
            if ep > 2:
                model.train(task.prompt(pair), task.answer(pair))
    assert whole.state_hash() == model.state_hash()
    saved = json.loads((HERE / "results/DEV_BP/RESULT.json").read_text())
    assert saved["state_hash_at_finish"] == whole.state_hash()
    counts = expected_counts(world, DOSE, PASSES)
    audit = {"status": "THREE_CYCLES_COMPLETE_WITH_LIMITATIONS",
             "author_run": True, "formal_capacity_screen_run": False,
             "holdback_worlds_generated_or_opened": False,
             "full_roster_reduction": {k: v for k, v in reduced.items() if k != "results"},
             "missing_cell_rejected": True, "extra_cell_rejected": True,
             "duplicate_roster_rejected": True, "bad_source_hash_rejected": True,
             "BP_resume_equals_uninterrupted": True,
             "development_fit_starts": 4, "resumed_continuations": 1,
             "development_records_total": counts[0] * 4 + counts[0] // 2,
             "development_target_tokens_total": counts[1] * 4 + counts[1] // 2,
             "process_cpu_cycle3_s": time.process_time() - t0,
             "remaining": ["Full Stage C runner/admission and 4,096-pass continuation not executed or qualified",
                           "No CAP-D1/D2 grid, CAP-H1/H2 result, native addition efficacy or persistence verdict"]}
    write_json(HERE / "CYCLE3_AUDIT.json", audit)
    print("Cycle 3 PASS: complete roster, negative reducer fixtures, source hash and resume")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cycle", choices=("cycle1", "cycle2", "cycle3"))
    args = ap.parse_args()
    globals()[args.cycle]()


if __name__ == "__main__":
    main()
