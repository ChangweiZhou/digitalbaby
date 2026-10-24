"""V144-R2 exercise protocol. The Full151 and V144 learners are imported unchanged."""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT / "minifly_v144_dev_20261024"
sys.path.insert(0, str(PARENT))
import task  # evaluator only
from reference import EOS, ByteTrace, Reference, bits, decode, seed

MODES = ("BP", "DFA", "FROZEN")
FINAL_PASSES = (0, 1, 4, 16, 64, 256, 1024, 4096)
SCHEMA = "V144_R2_REFERENCE_RESULT_V1"
SHA = re.compile(r"^[0-9a-f]{64}$")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def split_digest(world):
    return hashlib.sha256(canonical(world).encode()).hexdigest()


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(p)


def trace_from_parent():
    sys.path.insert(0, str(ROOT / "minifly_learned_bank_v1_20261024"))
    import lb1
    return ByteTrace(lb1.BC.R_TABLE, lb1.fastmem.PERM, lb1.DELTA, lb1.fastmem.FEPerm)


def expected_counts(world, dose, passes):
    pairs = task.oriented(world["doses"][str(dose)])
    return (len(pairs) * passes,
            sum(len(task.answer(pair)) + 1 for pair in pairs) * passes)


def reference_score(model, world, training, label, elapsed_cpu):
    before = model.state_hash()
    held = task.score(model, task.oriented(world["held"]))
    taught = task.score(model, task.oriented(training))
    teacher = teacher_metrics(model, task.oriented(world["held"]))
    assert before == model.state_hash(), "probe mutated learner"
    return {"label": label, "records": model.records,
            "observed_target_tokens": model.target_tokens,
            "cpu_s": elapsed_cpu, "state_hash": before,
            "held": held, "taught": taught, "teacher_forced_diagnostic": teacher}


def teacher_metrics(model, pairs):
    hits = 0
    loss = 0.0
    queries = 0
    for pair in pairs:
        prefix = bytearray()
        for token in (*task.answer(pair), EOS):
            margins = model.margins(model.trace.context(task.prompt(pair), prefix))
            target = np.array([(token >> bit) & 1 for bit in range(9)])
            hits += decode(margins) == token
            loss += float(np.sum(np.logaddexp(0, margins) - target * margins))
            queries += 1
            if token != EOS:
                prefix.append(token)
    return {"teacher_forced_token_accuracy": hits / queries,
            "teacher_forced_BCE_per_token": loss / queries,
            "evaluator_target_queries": queries}


def orbit_rates(rows):
    """All rates use exactly the same orbit denominator."""
    groups = {}
    for row in rows:
        pair = tuple(row["pair"])
        groups.setdefault(tuple(sorted(pair)), []).append(row)
    def rate(key):
        return sum(sum(float(row[key]) for row in group) / len(group)
                   for group in groups.values()) / len(groups)
    return {"complete": rate("exact"), "payload": rate("payload"),
            "own_eos": rate("own_eos")}


def controls(world, dose, trace):
    """Evaluator-only controls; none of these values enter Reference.train."""
    training = task.oriented(world["doses"][str(dose)])
    held = task.oriented(world["held"])
    train_rows = [(p, task.prompt(p), task.answer(p)) for p in training]
    train_vectors = [(pair, trace.context(prompt), answer)
                     for pair, prompt, answer in train_rows]
    raw = {prompt: answer for _, prompt, answer in train_rows}
    unordered = {tuple(sorted(pair)): answer for pair, _, answer in train_rows}
    freq = {}
    for _, _, answer in train_rows:
        freq[answer] = freq.get(answer, 0) + 1
    frequent = min(freq, key=lambda answer: (-freq[answer], answer))
    out = {name: [] for name in ("raw_lookup", "unordered_lookup", "frequency",
                                "COPY_A", "COPY_B", "FE_PERM_NEAREST")}
    for pair in held:
        prompt = task.prompt(pair)
        vector = trace.context(prompt)
        near = min(train_vectors,
                   key=lambda row: (float(np.sum((vector - row[1]) ** 2)), row[0]))
        payloads = {
            "raw_lookup": raw.get(prompt),
            "unordered_lookup": unordered.get(tuple(sorted(pair))),
            "frequency": frequent,
            "COPY_A": prompt[:1],
            "COPY_B": prompt[2:3],
            "FE_PERM_NEAREST": near[2],
        }
        for name, payload in payloads.items():
            out[name].append({"pair": list(pair), "output_hex": "" if payload is None else payload.hex(),
                              "status": "ABSTAIN" if payload is None else "EOS",
                              "exact": payload == task.answer(pair) if payload is not None else False,
                              "payload": payload == task.answer(pair) if payload is not None else False,
                              "own_eos": payload is not None})
    return {name: {"orbit_complete": orbit_rates(rows)["complete"], "rows": rows}
            for name, rows in out.items()}


def finite(value):
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return False


def _score_errors(score, expected_pairs, where):
    errors = []
    if not isinstance(score, dict):
        return [where + ": score missing"]
    fields = {"rows", "orbit_exact", "oriented_exact", "payload_only",
              "own_eos", "invalid", "truncated"}
    if not fields <= score.keys() or not isinstance(score.get("rows"), list):
        return [where + ": nested score fields/rows missing"]
    rows = score["rows"]
    if len(rows) != len(expected_pairs) or [tuple(row.get("pair", [])) for row in rows] != expected_pairs:
        errors.append(where + ": prompt roster/order mismatch")
        return errors
    for row in rows:
        try:
            payload = bytes.fromhex(row["output_hex"])
            pair = tuple(row["pair"])
            status = row["status"]
            if status not in ("EOS", "INVALID", "TRUNCATED") or len(payload) > 8:
                raise ValueError("status/cap")
            if row["exact"] is not (status == "EOS" and payload == task.answer(pair)):
                raise ValueError("exact")
            if row["payload"] is not (payload == task.answer(pair)):
                raise ValueError("payload")
            if row["own_eos"] is not (status == "EOS"):
                raise ValueError("EOS")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(where + ": malformed/scored row " + str(exc))
            break
    if errors:
        return errors
    rates = orbit_rates(rows)
    calculated = {"orbit_exact": rates["complete"],
                  "oriented_exact": sum(row["exact"] for row in rows) / len(rows),
                  "payload_only": sum(row["payload"] for row in rows) / len(rows),
                  "own_eos": sum(row["own_eos"] for row in rows) / len(rows),
                  "invalid": sum(row["status"] == "INVALID" for row in rows) / len(rows),
                  "truncated": sum(row["status"] == "TRUNCATED" for row in rows) / len(rows)}
    for key, expected in calculated.items():
        value = score[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 or abs(value - expected) > 1e-10:
            errors.append(where + ": invalid " + key)
    return errors


def validate_result(row, spec, world, lock_sha):
    """Fail closed on identity, nested results, finite/ranged metrics and exposure."""
    errors = []
    if not isinstance(row, dict) or not finite(row):
        return ["result not a finite JSON object"]
    if row.get("schema") != SCHEMA or row.get("complete") is not True:
        errors.append("schema/incomplete")
    if row.get("id") != spec["id"] or row.get("settings") != spec["settings"]:
        errors.append("roster/settings identity")
    if row.get("split_sha256") != split_digest(world) or row.get("source_lock_sha256") != lock_sha:
        errors.append("split/source identity")
    settings = spec["settings"]
    dose = settings["dose"]
    passes = settings["passes"]
    counts = expected_counts(world, dose, passes)
    if row.get("records") != counts[0] or row.get("observed_target_tokens") != counts[1]:
        errors.append("exposure totals")
    if not isinstance(row.get("cpu_s"), (int, float)) or row.get("cpu_s") < 0:
        errors.append("CPU")
    if not isinstance(row.get("peak_rss_bytes"), int) or row["peak_rss_bytes"] < 0:
        errors.append("peak RSS")
    snaps = row.get("metrics", {}).get("snapshots") if isinstance(row.get("metrics"), dict) else None
    labels = spec["checkpoints"]
    if not isinstance(snaps, list) or len(snaps) != len(labels) or [s.get("label") for s in snaps] != labels:
        return errors + ["checkpoint roster"]
    held = task.oriented(world["held"])
    taught = task.oriented(world["doses"][str(dose)])
    prev_cpu = -1.0
    for snap, label in zip(snaps, labels):
        cp = 0 if label == "birth" else int(label[4:])
        expected = expected_counts(world, dose, cp)
        if snap.get("records") != expected[0] or snap.get("observed_target_tokens") != expected[1]:
            errors.append(label + ": checkpoint exposure")
        c = snap.get("cpu_s")
        if not isinstance(c, (int, float)) or not math.isfinite(c) or c < prev_cpu:
            errors.append(label + ": CPU monotonicity")
        else:
            prev_cpu = c
        if not isinstance(snap.get("state_hash"), str) or not SHA.fullmatch(snap["state_hash"]):
            errors.append(label + ": state hash")
        errors += _score_errors(snap.get("held"), held, label + ":held")
        errors += _score_errors(snap.get("taught"), taught, label + ":taught")
        teacher = snap.get("teacher_forced_diagnostic")
        nqueries = sum(len(task.answer(p)) + 1 for p in held)
        if not isinstance(teacher, dict) or teacher.get("evaluator_target_queries") != nqueries:
            errors.append(label + ": teacher roster")
        else:
            accuracy = teacher.get("teacher_forced_token_accuracy")
            loss = teacher.get("teacher_forced_BCE_per_token")
            if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                errors.append(label + ": teacher accuracy")
            if not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
                errors.append(label + ": teacher loss")
    if snaps[-1].get("records") != row.get("records") or snaps[-1].get("observed_target_tokens") != row.get("observed_target_tokens"):
        errors.append("final checkpoint totals")
    if isinstance(row.get("cpu_s"), (int, float)) and row["cpu_s"] < prev_cpu:
        errors.append("final CPU before checkpoint")
    return errors


def reduce_roster(result_dir, roster, worlds, lock_sha):
    """No final table or verdict if any declared cell is absent or invalid."""
    result_dir = Path(result_dir)
    declared = {s["id"]: s for s in roster}
    if len(declared) != len(roster):
        return {"status": "INCOMPLETE", "missing": [], "extra": [],
                "invalid": {"roster": ["duplicate declared ID"]},
                "validated_results": 0, "expected_results": len(roster), "results": []}
    present = {p.parent.name: p for p in result_dir.glob("*/RESULT.json")}
    missing = sorted(set(declared) - set(present))
    extra = sorted(set(present) - set(declared))
    invalid = {}
    rows = []
    for ident in sorted(set(declared) & set(present)):
        try:
            row = json.loads(present[ident].read_text())
            problems = validate_result(row, declared[ident], worlds[declared[ident]["settings"]["world"]], lock_sha)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            problems = ["unreadable/invalid result: " + str(exc)]
            row = None
        if problems:
            invalid[ident] = problems
        else:
            rows.append(row)
    status = "COMPLETE" if not (missing or extra or invalid) else "INCOMPLETE"
    return {"status": status, "missing": missing, "extra": extra, "invalid": invalid,
            "validated_results": len(rows), "expected_results": len(roster),
            "results": rows if status == "COMPLETE" else []}


def stage_a_roster():
    worlds = ("V144-CAP-D1", "V144-CAP-D2")
    return [{"id": f"A_{world}_{hidden}_{lr}_{decay}",
             "settings": {"world": world, "mode": "BP", "hidden": hidden,
                          "lr": lr, "decay": decay, "dose": 40, "passes": 64},
             "checkpoints": ["birth", "pass1", "pass4", "pass16", "pass64"]}
            for world in worlds for hidden in (128, 256, 512)
            for lr in (.003, .01, .03) for decay in (0., .001, .01)]


def shortlist(rows):
    """One 64-pass development finalist for each hidden size."""
    finalists = []
    for hidden in (128, 256, 512):
        options = [r for r in rows if r["hidden"] == hidden]
        if len(options) != 9:
            raise ValueError("incomplete development grid")
        finalists.append(min(options, key=lambda r: (-r["held_exact"], r["teacher_loss"],
                                                     r["cpu_s"], r["lr"], r["decay"])))
    return finalists


def admit(used_cpu_s, estimate_cpu_s, limit_s=6 * 3600):
    if not all(math.isfinite(v) and v >= 0 for v in (used_cpu_s, estimate_cpu_s)):
        raise ValueError("invalid CPU estimate")
    return used_cpu_s + estimate_cpu_s <= limit_s
