"""Bounded engineering checks for the reasoning scaling design."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import resource
from statistics import mean
import time

from benchmark import benchmark_family, expected, parse, structure_key
from task import EOS, active_opcode, probes, training

ROOT = Path(__file__).resolve().parent
WORLD = "reasoning-engineering-rehearsal"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def source_receipt(archive):
    dirs = {
        "minifly_v216_experience_pretraining_design_20261104": ("engine.py", "fast.py"),
        "minifly_v181_mini_local_plasticity_dev_20261102": ("learner.py", "sensor.py"),
        "minifly_a_v3/package": ("FULL151_BIRTH_FINGERPRINT.json", "FULL151_CANONICAL_B.npz"),
        "minifly_a_v3/package/REFERENCE_SOURCE/minifly/iteration10/models": ("reference.npz",),
    }
    original = {str(Path(folder) / name): digest(archive / folder / name)
                for folder, names in dirs.items() for name in names}
    prior_lock = archive / "minifly_v216_experience_pretraining_design_20261104/campaign_20261104/LOCK.json"
    locked = json.loads(prior_lock.read_text())["sources"]
    assert all(locked[path] == actual for path, actual in original.items())
    return {
        "archived_dependencies": original,
        "cycle_sources": {p.name: digest(p) for p in sorted(ROOT.glob("*.py"))},
    }


def cycle1():
    rows = [benchmark_family(f"engineering-audit-{i}") for i in range(8)]
    diversity_sizes = {
        str(d): len({row.observed for row in training("engineering-canary", "A", d, 10_000)})
        for d in (4, 32)
    }
    assert min(diversity_sizes.values()) >= 9500
    for kind in ("structure", "polarity"):
        pairs = probes("engineering-canary", "A", kind)
        new = active_opcode("engineering-canary", "A_prime")
        flips = sum(expected(parse(row.prefix), new) != row.target for pair in pairs for row in pair)
        assert flips == (0 if kind == "structure" else 64)
    train = list(training("engineering-canary", "A", 4, 16))
    contamination = probes("engineering-canary", "A", "structure")[0][0]
    def exposure_clear(records):
        return all(structure_key(parse(row.prefix)) != (2, 2) for row in records)
    assert exposure_clear(train) and not exposure_clear([*train, contamination])
    return {
        "status": "ENGINEERING_PASS", "families": rows, "diamond_exposure_canary_rejected": True,
        "unique_records_in_10000_generated_untrained_events": diversity_sizes,
    }


def cycle2(backend):
    from backend import parameter_count, state_hash

    T = backend.T
    records = list(training(WORLD, "A", 4, 16))
    source = backend.Life(WORLD, backend.sensor)
    stock = backend.birth(WORLD, 64)
    wide = backend.birth(WORLD, 128)
    assert state_hash(source) == state_hash(stock)
    assert [parameter_count(life) for life in (stock, wide)] == [37313, 57921]
    assert T.equal(stock.model.p["H"], wide.model.p["H"])
    assert T.equal(stock.model.fh, wide.model.fh)
    assert T.equal(stock.model.ft, wide.model.ft[:, :64])
    before = state_hash(wide)
    for record in records:
        for life in (source, stock, wide):
            life.train(record.observed)
        assert state_hash(source) == state_hash(stock)
    assert before != state_hash(wide) and wide.events == 16
    sample = records[-1]
    x = wide.codes(sample.observed)
    targets = T.tensor([*sample.observed, EOS])
    assert tuple(x.shape) == (49, 5177)
    changed_record = sample.prefix + bytes((ord("0") if sample.target != ord("0") else ord("1"),))
    assert T.equal(x[:-1], wide.codes(changed_record)[:-1])
    rng = wide.model.rng.get_state()
    grads = backend.normal_gradients(wide.model, x, targets)
    wide.model.rng.set_state(rng)
    logits, (xx, ah, h, at, mask, z) = wide.model.forward(x, True)
    error = (logits.softmax(1) - T.nn.functional.one_hot(targets, num_classes=257)) / len(targets)
    assert logits.shape == (49, 257)
    checked = {}
    formulas = {
        "H": lambda r, c: ((error @ wide.model.fh[:, r]) * (ah[:, r] > 0) * xx[:, r, c]).sum(),
        "T": lambda r, c: ((error @ wide.model.ft[:, r]) * (at[:, r] > 0) * mask[:, r] * h[:, c]).sum(),
        "O": lambda r, c: (error[:, r] * z[:, c]).sum(),
    }
    for layer, formula in formulas.items():
        group = grads[layer]
        row, col = divmod(int(group.abs().argmax()), group.shape[1])
        actual = group[row, col]
        assert abs(float(actual)) > 1e-9
        assert T.allclose(actual, formula(row, col), atol=1e-7, rtol=1e-5)
        checked[layer] = {"row": row, "col": col, "gradient": float(actual)}
    probes_rows = [r for pair in probes(WORLD, "A", "structure")[:3] for r in pair]
    outputs = {}
    for width, life in ((64, stock), (128, wide)):
        before = state_hash(life)
        outputs[str(width)] = [life.generate(r.prefix) for r in probes_rows]
        assert state_hash(life) == before
        assert all(0 < len(out) <= 32 and all(0 <= token <= EOS for token in out)
                   and EOS not in out[:-1] for out in outputs[str(width)])
    return {
        "status": "ENGINEERING_PASS", "width64_original_engine_updates_and_state_bitwise_equal": 16,
        "width128_actual_updates": 16, "trainable_parameters": {"64": 37313, "128": 57921},
        "wide_update_checks": checked, "target_mutation_preserved_prior_features": True,
        "observed_native_outputs": outputs,
    }


def score(life, pairs):
    from backend import state_hash

    before = state_hash(life)
    rows = [
        {"prefix": row.prefix.hex(), "target": [row.target, EOS], "output": life.generate(row.prefix)}
        for pair in pairs for row in pair
    ]
    assert state_hash(life) == before
    hit = [row["output"] == row["target"] for row in rows]
    return {
        "n": len(rows), "exact": sum(hit), "both": sum(x and y for x, y in zip(hit[::2], hit[1::2])),
        "own_eos": sum(row["output"][-1] == EOS for row in rows), "outputs": rows,
    }


def train_segment(life, records):
    clock = time.process_time()
    for row in records:
        life.train(row.observed)
    return time.process_time() - clock


def cycle3(backend, checkpoint):
    from backend import state_hash

    begin = time.process_time()
    results = []
    for width, diversity in ((64, 4), (64, 32), (128, 4), (128, 32)):
        life = backend.birth(WORLD, width)
        stream = list(training(WORLD, "A", diversity, 1536))
        panels = {"cold": score(life, probes(WORLD, "A", "structure"))}
        cpu = {"A_first_64": train_segment(life, stream[:64])}
        panels["A_64"] = score(life, probes(WORLD, "A", "structure"))
        cpu["A_65_to_1024"] = train_segment(life, stream[64:1024])
        cpu["A_stable_512"] = train_segment(life, stream[1024:])
        panels["A_1536"] = score(life, probes(WORLD, "A", "structure"))
        panels["A_polarity"] = score(life, probes(WORLD, "A", "polarity"))
        panels["A_taught"] = score(life, list(zip(stream[-32::2], stream[-31::2])))
        b_records = list(training(WORLD, "B", diversity, 128))
        cpu["B_128"] = train_segment(life, b_records)
        panels["B_before_edit"] = score(life, probes(WORLD, "B", "structure"))
        matched = backend.clone(life, WORLD, width)
        edit = list(training(WORLD, "A_prime", diversity, 128))
        noedit = list(training(WORLD, "A_prime", diversity, 128, edit=False))
        assert state_hash(matched) == state_hash(life)
        assert [row.prefix for row in edit] == [row.prefix for row in noedit]
        assert any(a.target != b.target for a, b in zip(edit, noedit))
        if (width, diversity) == (128, 32):
            backend.checkpoint(life, checkpoint, WORLD, width)
            expected_life = backend.clone(life, WORLD, width)
            expected_life.train(edit[0].observed)
            resume = {
                "checkpoint_sha256": digest(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
                "next_record_hex": edit[0].observed.hex(), "expected_state_after_one_update": state_hash(expected_life),
            }
        cpu["A_prime_128"] = train_segment(life, edit)
        cpu["NOEDIT_128"] = train_segment(matched, noedit)
        assert state_hash(matched) != state_hash(life)
        panels["A_prime_changed"] = score(life, probes(WORLD, "A_prime", "polarity"))
        panels["NOEDIT_same_changed_gold"] = score(matched, probes(WORLD, "A_prime", "polarity"))
        panels["A_still_valid"] = score(life, probes(WORLD, "A_prime", "structure"))
        panels["B_after_edit"] = score(life, probes(WORLD, "B", "structure"))
        results.append({
            "width": width, "diversity": diversity, "events_main": life.events,
            "events_noedit": len(noedit), "training_target_sites": 49 * (life.events + len(noedit)),
            "cpu_seconds": cpu, "panels": panels,
        })
    late_cpu = mean(row["cpu_seconds"]["A_stable_512"] / 512 for row in results)
    return {
        "status": "RESTART_PENDING", "scope": "One unsealed technical family and four arms",
        "observed_training_events": sum(row["events_main"] + row["events_noedit"] for row in results),
        "observed_training_target_sites": sum(row["training_target_sites"] for row in results),
        "additional_restart_verification_events": 1,
        "restart": resume,
        "worker_cpu_seconds": time.process_time() - begin,
        "peak_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "training_only_projection_cpu_hours": late_cpu * 4_160_000 / 3600,
        "main_resource_admission": "PENDING_FULL_MEASUREMENT_AND_APPROVAL",
        "arms": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("mode", choices=("1", "2", "3", "resume"))
    parser.add_argument("checkpoint", nargs="?", type=Path)
    args = parser.parse_args()
    sources = source_receipt(args.archive)
    if args.mode == "1":
        receipt = cycle1()
    else:
        from backend import Backend, state_hash
        backend = Backend(args.archive)
        if args.mode == "resume":
            if args.checkpoint is None:
                parser.error("resume requires the saved rehearsal checkpoint")
            receipt = json.loads((ROOT / "CYCLE3_TEST.json").read_text())
            assert receipt["status"] == "RESTART_PENDING" and receipt["sources"] == sources
            restart = receipt["restart"]
            assert digest(args.checkpoint) == restart["checkpoint_sha256"]
            clock = time.process_time()
            life = backend.restore(args.checkpoint)
            life.train(bytes.fromhex(restart["next_record_hex"]))
            actual = state_hash(life)
            assert actual == restart["expected_state_after_one_update"]
            restart["actual_state_after_one_update"] = actual
            restart["worker_cpu_seconds"] = time.process_time() - clock
            receipt["additional_restart_verification_events"] += 1
            receipt["fresh_process_checkpoint_resume_equal"] = True
            receipt["status"] = "ENGINEERING_PASS"
        elif args.mode == "2":
            receipt = cycle2(backend)
        else:
            if args.checkpoint is None or args.checkpoint.exists():
                parser.error("cycle 3 requires a new checkpoint path")
            receipt = cycle3(backend, args.checkpoint)
    receipt["sources"] = sources
    number = "3" if args.mode == "resume" else args.mode
    write(ROOT / f"CYCLE{number}_TEST.json", receipt)
    print(json.dumps({"cycle": args.mode, "status": receipt["status"]}))
