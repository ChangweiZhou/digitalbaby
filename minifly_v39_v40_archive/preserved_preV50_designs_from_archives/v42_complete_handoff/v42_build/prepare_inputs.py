#!/usr/bin/env python3
"""Prepare verified, read-only V41 inputs for V42 fixed-history analysis.

This program does not evaluate new arbitration policies or call observe/step.
It verifies local trusted V41 artifacts and re-reads final frozen checkpoints.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "v42_build"
SEEDS = (41101, 41102, 41103, 41104)
ARMS = ("BASE", "SUPER", "SLOTS", "READ-HYBRID", "FULL-HYBRID")
EXPECTED_SOURCE_SHA = "223434f801c3f51071b420bb7e60330795d8559d54bc24730c9bc424d77a4b27"
EXPECTED_PLAN_SHA = "b5a9761be79e8f541017b0a38bc069303279f57bfa6480b2114e9aa6309e9a25"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def integer_correct(group):
    value = group["n"] * group["accuracy"]
    integer = round(value)
    assert abs(value - integer) < 1e-8, (value, integer)
    return integer


def audit_groups(result):
    """Reconstruct original baselines only, never a new policy."""
    groups = result["groups"]
    common = {k: v for k, v in groups.items() if k.startswith("slots_state/")}
    allowed = {"exact_slot", "exact_trace", "supported_slot", "no_candidate", "tentative_trace"}
    for key, arms in common.items():
        prefix, kind, category = key.split("/")
        assert prefix == "slots_state" and kind in allowed
        assert category in ("target", "nontarget") and set(arms) == set(ARMS)
        assert len({v["n"] for v in arms.values()}) == 1
        for entry in arms.values():
            assert entry["n"] > 0
            assert math.isclose(entry["bpb"] * entry["n"], entry["loss_sum"], rel_tol=1e-12, abs_tol=1e-9)
            integer_correct(entry)
    rows = []
    for category in ("target", "nontarget", "whole"):
        selected = [v for k, v in common.items() if category == "whole" or k.endswith("/" + category)]
        for arm in ARMS:
            n = sum(v[arm]["n"] for v in selected)
            loss_sum = math.fsum(v[arm]["loss_sum"] for v in selected)
            correct = sum(integer_correct(v[arm]) for v in selected)
            expected = groups[category][arm]
            residual = loss_sum - expected["loss_sum"]
            assert n == expected["n"]
            assert abs(residual) <= 1e-7
            assert correct == integer_correct(expected)
            assert math.isclose(loss_sum / n, expected["bpb"], rel_tol=1e-11, abs_tol=1e-11)
            rows.append({"category": category, "arm": arm, "n": n,
                         "correct": correct, "loss_sum_residual": residual})
    nonauthority = []
    for key, arms in common.items():
        if key.split("/")[1] in ("no_candidate", "tentative_trace"):
            read, fast = arms["READ-HYBRID"], arms["SUPER"]
            residual = read["loss_sum"] - fast["loss_sum"]
            assert abs(residual) <= 1e-9
            assert integer_correct(read) == integer_correct(fast)
            nonauthority.append({"group": key, "n": read["n"], "loss_sum_residual": residual,
                                 "correct_difference": integer_correct(read) - integer_correct(fast)})
    return {"baseline_reconstruction": rows, "common_group_keys": sorted(common),
            "nonauthority_read_super_equality": nonauthority,
            "all_group_correct_counts_integer": True}


def probe_key(row):
    return row["checkpoint"], row["kind"], row["key_id"], row["alias"]


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / "elm_v41_mechanism_closures.py"
    archived_source = ROOT / "v41_results/elm_v41_mechanism_closures.py"
    assert sha(source.read_bytes()) == EXPECTED_SOURCE_SHA
    assert source.read_bytes() == archived_source.read_bytes()
    spec = importlib.util.spec_from_file_location("v42_input_reference", source)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    ref.dependency_setup()
    modules = ref.load_modules()
    hybrid = modules["hybrid"]
    plan_file = ROOT / "v41_results/PLAN.json"
    plan = json.loads(plan_file.read_bytes())
    assert ref.digest(plan) == EXPECTED_PLAN_SHA
    assert plan["source_sha256"] == EXPECTED_SOURCE_SHA
    records = []
    for seed in SEEDS:
        completed = ROOT / f"v41_results/completed/hybrid_{seed}.json"
        checkpoint = ROOT / f"v41_results/checkpoints/hybrid_{seed}.checkpoint"
        envelope = json.loads(completed.read_bytes())
        identity = envelope["identity"]
        assert identity == {"plan": EXPECTED_PLAN_SHA, "experiment": "hybrid", "seed": seed,
                            "config": ref.CONFIGS["hybrid"]}
        data = ref.completed_read(completed, identity)
        assert data == envelope["data"]
        result = data["result"]
        packed = checkpoint.read_bytes()
        size = struct.unpack(">Q", packed[:8])[0]
        header = json.loads(packed[8:8 + size])
        assert header["identity"] == identity
        checkpoint_data = ref.unpack_state(packed, identity)
        state = checkpoint_data["state"]
        assert state["done"] and state["cursor"] == len(state["stream"]["rows"])
        assert state["cursor"] == result["records"] == 1096
        assert state["stream"]["checkpoints"][-1]["name"] == "after_D"
        assert state["stream"]["checkpoints"][-1]["record_count"] == state["cursor"]
        stream = state["stream"]
        raw = stream["raw"]
        assert sha(raw) == stream["raw_sha256"] == result["stream_sha256"]
        assert len(raw) == result["byte_events"] == 15344
        learner_before = hybrid._canonical_hash(hybrid._learners(state))
        full_before = ref.state_fingerprint(state)
        assert learner_before == result["learner_full_state_hash"]

        history = bytearray()
        observed = set()
        for byte in raw:
            observed.add(bytes(history))
            history.append(byte)
            if len(history) > 12:
                del history[0]
        assert history == state["history"]
        contexts = [{"key_id": i, "context_hex": hybrid.M.context(key, 2).hex()}
                    for i, key in enumerate(stream["keys"])]
        collisions = [x for x in contexts if bytes.fromhex(x["context_hex"]) in observed]
        assert not collisions
        original_probe_keys = [(probe_key(x), x["arm"]) for x in result["probes"]]
        assert len(original_probe_keys) == len(set(original_probe_keys))
        after_d = [x for x in result["probes"] if x["checkpoint"] == "after_D"
                   and x["arm"] in ("READ-HYBRID", "SUPER")]
        by_probe = {}
        for x in after_d:
            by_probe.setdefault(probe_key(x), {})[x["arm"]] = x
        assert all(set(v) == {"READ-HYBRID", "SUPER"} for v in by_probe.values())
        enriched = []
        for key, old_arms in by_probe.items():
            _, kind, ident, alias = key
            assert kind in ("untouched_A", "untouched_A_heldout", "current", "current_heldout")
            context = hybrid.M.context(stream["keys"][ident], alias)
            target = int(stream["labels"][ident])
            scored, packets, _ = hybrid._read(state, context)
            packet = packets["SLOTS"]
            scalar_packet = {"stratum": hybrid._stratum(packet), "exact": bool(packet["exact"]),
                             "type": packet["type"], "authority": bool(packet["authority"]),
                             "support": int(packet["support"]),
                             "sim": None if packet["sim"] is None else float(packet["sim"])}
            if alias == 2:
                assert not scalar_packet["exact"]
                assert not scalar_packet["authority"] or scalar_packet["stratum"] == "supported_slot"
            for arm, old in old_arms.items():
                loss = float(hybrid.M.loss(scored[arm], target))
                accuracy = int(ref.np.argmax(scored[arm]) == target)
                assert loss == old["bpb"], (seed, key, arm, loss, old["bpb"])
                assert accuracy == old["accuracy"]
                if arm == "READ-HYBRID":
                    assert scalar_packet["authority"] == old["authority"]
                enriched.append({**old, "context_hex": context.hex(), "packet": scalar_packet.copy(),
                                 "origin": "reproduced_original_after_D"})
        # Additional terminal queries answer retention of revisions through D.
        # Their target labels come from the unchanged original workload state.
        revision_context_count = 0
        for ident in stream["revised"]:
            target = int(stream["revisions"][ident])
            for alias in (0, 1, 2):
                context = hybrid.M.context(stream["keys"][ident], alias)
                scored, packets, _ = hybrid._read(state, context)
                packet = packets["SLOTS"]
                scalar_packet = {"stratum": hybrid._stratum(packet), "exact": bool(packet["exact"]),
                                 "type": packet["type"], "authority": bool(packet["authority"]),
                                 "support": int(packet["support"]),
                                 "sim": None if packet["sim"] is None else float(packet["sim"])}
                if alias == 2:
                    assert not packet["exact"]
                    assert not packet["authority"] or scalar_packet["stratum"] == "supported_slot"
                revision_context_count += 1
                for arm in ("READ-HYBRID", "SUPER"):
                    enriched.append({"checkpoint": "after_D", "kind": "revision_retained", "arm": arm,
                                     "key_id": int(ident), "alias": alias,
                                     "accuracy": int(ref.np.argmax(scored[arm]) == target),
                                     "bpb": float(hybrid.M.loss(scored[arm], target)),
                                     "authority": bool(packet["authority"]) if arm == "READ-HYBRID" else None,
                                     "context_hex": context.hex(), "packet": scalar_packet.copy(),
                                     "origin": "new_terminal_read_only_revision_query"})
        learner_after = hybrid._canonical_hash(hybrid._learners(state))
        full_after = ref.state_fingerprint(state)
        assert learner_before == learner_after
        assert full_before == full_after
        audits = audit_groups(result)
        audits.update({"learner_hash_before": learner_before, "learner_hash_after": learner_after,
                       "full_state_hash_before": full_before, "full_state_hash_after": full_after,
                       "learner_hash_matches_original_record": True,
                       "terminal_probe_prediction_count": len(enriched),
                       "terminal_probe_context_count": len(by_probe) + revision_context_count,
                       "original_after_D_reproduction_context_count": len(by_probe),
                       "original_after_D_reproduction_prediction_count": 2 * len(by_probe),
                       "new_revision_retention_context_count": revision_context_count,
                       "new_revision_retention_prediction_count": 2 * revision_context_count,
                       "terminal_probe_losses_match_original_exactly": True,
                       "terminal_probe_accuracy_matches_original_exactly": True,
                       "all_original_probe_keys_unique": True,
                       "model_read_calls": len(by_probe) + revision_context_count,
                       "base_score_calculations": len(by_probe) + revision_context_count,
                       "super_score_calculations": len(by_probe) + revision_context_count,
                       "memory_lookups": 2 * (len(by_probe) + revision_context_count),
                       "scored_path_vectors": 5 * (len(by_probe) + revision_context_count),
                       "loss_calculations": len(enriched),
                       "observe_calls": 0, "step_calls": 0, "new_policy_evaluations": 0})
        records.append({"seed": seed, "original_completed_envelope": envelope,
                        "provenance": {"completed_file_sha256": sha(completed.read_bytes()),
                                       "completed_data_sha256": envelope["sha256"],
                                       "checkpoint_file_sha256": sha(packed),
                                       "checkpoint_payload_sha256": header["payload_sha256"],
                                       "v41_source_sha256": EXPECTED_SOURCE_SHA,
                                       "v41_plan_sha256": EXPECTED_PLAN_SHA,
                                       "stream_sha256": sha(raw)},
                        "raw_stream_base64": base64.b64encode(raw).decode("ascii"),
                        "alias2_contexts": contexts,
                        "alias2_certification": {"all_training_prebyte_count": len(raw),
                                                 "distinct_prebyte_contexts": len(observed),
                                                 "alias2_context_count": len(contexts),
                                                 "alias2_contexts_present": 0,
                                                 "observed_context_set_sha256": ref.digest(sorted(x.hex() for x in observed)),
                                                 "conclusion": "Alias 2 never has an exact trace or exact slot anchor at any historical time."},
                        "final_after_D_probes": enriched, "audits": audits})
        print(f"Verified {seed}: {len(enriched)} terminal predictions; no learning or policy evaluation.", flush=True)
        del state, checkpoint_data
    artifact = {"schema_version": 1, "seeds": list(SEEDS),
                "provenance": {"v41_source_sha256": EXPECTED_SOURCE_SHA,
                               "v41_plan_sha256": EXPECTED_PLAN_SHA,
                               "v41_plan_file_sha256": sha(plan_file.read_bytes()),
                               "v41_original_plan": plan,
                               "preparation_source_sha256": sha(Path(__file__).read_bytes()),
                               "preparation_method": "Read-only extraction of trusted local V41 completed records and final learner checkpoints."},
                "records": records}
    destination = OUT / "reference_inputs.json"
    ref.write_json(destination, artifact)
    report = ["# V42 reference-input preparation audit", "",
              "This preparation validates and extracts existing V41 evidence. It performs zero learning updates and zero new-policy evaluations.", "",
              f"V41 source SHA-256: `{EXPECTED_SOURCE_SHA}`.",
              f"V41 canonical plan SHA-256: `{EXPECTED_PLAN_SHA}`.",
              f"Reference input SHA-256: `{sha(destination.read_bytes())}`.", "",
              "## Verified source and checkpoint correspondence", "",
              "Each completed envelope passes its original V41 identity and data-digest checks. The local and archived V41 sources match byte-for-byte. Each checkpoint passes its payload checksum and identity checks. The full learner fingerprint matches the fingerprint saved in that seed's original completed result. All checkpoints are complete at record 1,096, the after_D checkpoint, after 15,344 byte observations.", "",
              "## Historical reconstruction inputs", "",
              "The original completed JSON envelopes, including their loss groups and historical probe records, are preserved without field modification. The mutually exclusive common SLOTS-state groups reconstruct every original arm's target, ordinary, and whole-stream counts, correct counts, and loss sums. Correct counts reconstructed from n × accuracy are integers to 1e-8; accumulated loss residuals are at most 1e-7. Nonauthoritative READ-HYBRID and SUPER groups have identical correct counts and loss sums to 1e-9.", "",
              "Historical probe keys (checkpoint, kind, key_id, alias, arm) are unique. No historical exact/type metadata is invented.", "",
              "## Alias-2 certification", "",
              "For each seed, every one of the 15,344 actual pre-byte history windows is scanned, including short initial histories. None of the 128 alias-2 contexts occurs anywhere. Because exact traces and anchors can be created only for observed contexts, no historical alias-2 probe can have exact authority. An authoritative alias-2 read must therefore be a non-exact supported-slot read. The raw stream and all alias-2 contexts are embedded so the standalone runner can independently repeat this check.", "",
              "## Terminal frozen-state extraction", "",
              "At each final checkpoint, all recorded after_D READ-HYBRID and SUPER probe contexts are re-read. Each recomputed loss equals its original stored floating-point value exactly, and every accuracy and READ authority flag matches. These rows are tagged reproduced_original_after_D. The extraction appends the common SLOTS packet's stratum, exact flag, type, authority, support, and similarity. Additionally, the final state is queried on each of the eight revised identities with all three aliases, using the revised labels from the unchanged workload state. Those rows are tagged new_terminal_read_only_revision_query and have no corresponding original after_D historical row. All learner and full-state fingerprints are unchanged before and after these reads.", "",
              "Each seed requires 168 calls to the inherited pure read function: 144 original contexts plus 24 added revision contexts. This computes 168 BASE score vectors, 168 SUPER score vectors, and 336 memory lookups (the inherited helper also reads the original FULL-HYBRID memory). It returns 840 path vectors; only READ-HYBRID and SUPER receive scalar loss/accuracy evaluation, for 336 losses per seed. Across four seeds: 672 model reads, 672 BASE score computations, 672 SUPER score computations, 1,344 memory lookups, and 1,344 evaluated prediction losses. There are zero observe calls, step calls, or new-policy evaluations.", "",
              "| Seed | Terminal contexts | Terminal predictions | Alias-2 training collisions |", "|---|---:|---:|---:|"]
    for row in records:
        report.append(f"| {row['seed']} | {row['audits']['terminal_probe_context_count']} | {row['audits']['terminal_probe_prediction_count']} | 0 |")
    report += ["", "## Scope", "",
               "These checks certify an offline transformation of four already observed V41 histories. They do not create four new independent seeds, provide fresh confirmation, recover unlogged historical score vectors, or establish policy behavior under altered writing. Only the final after_D probe records receive newly recovered packet metadata. Intermediate seen-alias probes retain their original data and cannot be classified retrospectively from terminal state. The added final revision-retention queries are new read-only evaluations on existing state, not new learning examples or new independent seeds.", ""]
    (OUT / "preparation_audit.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
