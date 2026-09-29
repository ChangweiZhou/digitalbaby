"""Common byte/teaching/output protocol for R2, Z1 and T2 technical runs.

This module contains no FE1 route and does not execute science on import.
All option probes operate on disposable clones of a continuing four-store life.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "BYTE_CORE_V9"))
import brain_byte as bb  # noqa: E402

from fixture import DT, RECORD_SECONDS, make_world, fact_variants

CHANNELS = b"0123"
BRANCHES = ("W", "N_old_fact", "N_old_rel", "N_new_fact", "N_new_rel")


def clone_model(model):
    """Clone mutable native, FE and adapter state; fixed connectome stays shared."""
    if hasattr(model, "clone_round"):
        out = model.clone_round()
    elif hasattr(model, "clone"):
        out = model.clone()
    else:
        out = copy.copy(model)
        out.fly = model.fly.clone()
        out.fe = model.fe.clone()
        out.w = model.w.copy()
        out.bias = model.bias.copy()
        out.pending_x = None if model.pending_x is None else model.pending_x.copy()
        for name in ("z", "trace", "eligibility"):
            if hasattr(model, name):
                setattr(out, name, getattr(model, name).copy())
    if out is model or out.fly is model.fly or out.fe is model.fe:
        raise AssertionError("mutable clone aliases model")
    if out.state_digest() != model.state_digest():
        raise AssertionError("clone state mismatch")
    return out


def fresh_native():
    """Native Full151 and FE0 start at model time zero; no V9.2 elapsed time."""
    model = bb.F151ByteBrain(temporal=False, order_via_native=False)
    if type(model.fe) is not bb.bc.FE0 or model.brain_t != 0.0 or model.fly.m.elapsed != 0.0:
        raise AssertionError("non-newborn native/FE0 state")
    return model


def mode_from_bytes(cue: bytes) -> str:
    # The classifier only selects the output alphabet. It does not recover
    # item identity or target, and it must also accept the sealed E2 forms.
    if (len(cue) == 12 and cue[-1] == 61 and cue.count(43) == 1 and
            sum(b in b"01234567" for b in cue) == 2):
        return "fact"
    if (len(cue) == 12 and cue[:2] == b"##" and cue[2:9] == b" " * 7
            and cue[9] != cue[10] and cue[11] == 32):
        return "relation"
    raise ValueError("unrecognized byte cue")


def branch_allows(branch: str, stage: str, domain: str) -> bool:
    if branch not in BRANCHES or stage not in ("old", "new") or domain not in ("fact", "relation"):
        raise ValueError("invalid branch/stage/domain")
    return branch != f"N_{stage}_{domain}"


class FourStore:
    def __init__(self, stores):
        if len(stores) != 4 or any(s is stores[0] for s in stores[1:]):
            raise ValueError("four independent output stores required")
        self.stores = list(stores)
        self.mechanism_events = []
        self.context = None

    def clone(self):
        out = FourStore([clone_model(m) for m in self.stores])
        out.mechanism_events = list(self.mechanism_events)
        out.context = self.context
        return out

    def set_record_context(self, world: int, branch: str, index: int, stage: str, domain: str):
        self.context = (int(world), branch, int(index), stage, domain)

    def digests(self):
        return [m.state_digest() for m in self.stores]

    def feed(self, b: int, t: float):
        for m in self.stores:
            m.byte(b, t, learn=False)

    def values(self, t: float):
        before = self.digests()
        values = [float(m.association_value(t)) for m in self.stores]
        if any(not math.isfinite(v) for v in values) or self.digests() != before:
            raise AssertionError("nonfinite or mutating native value read")
        return values

    def teach(self, answer: int, t: float, domain: str, *, write: bool):
        if answer not in CHANNELS or (domain == "relation" and answer not in b"01"):
            raise ValueError("answer is not a permitted arriving byte")
        active = range(4 if domain == "fact" else 2)
        for m in self.stores:
            m.byte(answer, t, learn=False)
        for j in range(4):
            # Inactive relation compartments still process the observed
            # outcome as a nonplastic native event, preserving adaptation and
            # clock semantics across all four continuing stores.
            did_write = bool(write and j in active)
            result = self.stores[j].teach(int(CHANNELS[j] != answer), t,
                                          write=did_write)
            if isinstance(result, dict):
                self.mechanism_events.append({
                    "context": self.context,
                    "channel": j, "teacher_at": t, "write": did_write,
                    "raw_alpha_l1": float(bb.np.abs(result.get("rawalpha", 0)).sum()),
                    "raw_fast_l1": float(bb.np.abs(result.get("rawfast", 0)).sum()),
                    "raw_slow_l1": float(bb.np.abs(result.get("rawslow", 0)).sum()),
                    "split_error": float(result.get("split_error", 0.0)),
                    "budget_buckets": result.get("z_budget_buckets", {})})

    def flush(self, t: float):
        return max(float(m.flush(t)) for m in self.stores)


def output_before_teacher(system: FourStore, cue: bytes, at: float):
    """Feed an already observed cue and emit before its answer byte arrives."""
    domain = mode_from_bytes(cue)
    for i, b in enumerate(cue):
        system.feed(b, at + i * DT)
    when = at + len(cue) * DT
    values = system.values(when)
    n = 4 if domain == "fact" else 2
    emitted = CHANNELS[max(range(n), key=lambda j: values[j])]
    return emitted, values, when, domain


def read_cue(system: FourStore, cue: bytes, at: float):
    """Read-only one-cue output from a full clone; never mutate the life."""
    before = system.digests()
    scratch = system.clone()
    emitted, values, when, domain = output_before_teacher(scratch, cue, at)
    if system.digests() != before:
        raise AssertionError("probe mutated continuing life")
    return {"emitted": emitted, "values": values, "when": when, "domain": domain}


def relation_choice(system: FourStore, cue_one: bytes, cue_two: bytes, at: float):
    """Fixed internal two-option L/R organ; both options read on disposable state."""
    if mode_from_bytes(cue_one) != "relation" or mode_from_bytes(cue_two) != "relation":
        raise ValueError("relation options required")
    first = read_cue(system, cue_one, at)
    second = read_cue(system, cue_two, at + 13 * DT)
    v1 = first["values"][1] - first["values"][0]
    v2 = second["values"][1] - second["values"][0]
    return {"emitted": ord("L") if v1 >= v2 else ord("R"),
            "option_scores": [v1, v2], "option_values": [first["values"], second["values"]],
            "response_time": at + 25 * DT}


def roster_probes(system: FourStore, world_doc: dict, at: float, stage: str, branch: str):
    """All prespecified E1/E2/E3 probes on a continuing single life."""
    rows = []
    for set_name in ("old_fact", "new_fact"):
        if set_name == "new_fact" and stage in ("old_end", "old_day"):
            continue
        group = world_doc[set_name]
        for item, (cue_hex, label) in enumerate(zip(group["cues_hex"], group["labels"], strict=True)):
            cue = bytes.fromhex(cue_hex)
            for form, shown in (("canonical", cue), *fact_variants(cue).items()):
                if form != "canonical" and stage != "final":
                    continue
                try:
                    result = read_cue(system, shown, at)
                except Exception as exc:
                    raise RuntimeError(f"probe failed at {stage}/{branch}/{set_name}/{item}/{form}: {shown!r}") from exc
                rows.append({"stage": stage, "branch": branch, "set": set_name,
                             "item": item, "form": form, "cue_hex": shown.hex(),
                             "target": 48 + label, "emitted": result["emitted"],
                             "values": result["values"], "correct": int(result["emitted"] == 48 + label)})
    for set_name in ("old_relation", "new_relation"):
        if set_name == "new_relation" and stage in ("old_end", "old_day"):
            continue
        taught = world_doc[set_name]["taught"]
        for item, row in enumerate(taught):
            cue = bytes.fromhex(row["cue_hex"])
            result = read_cue(system, cue, at)
            target = 48 + row["label"]
            rows.append({"stage": stage, "branch": branch, "set": set_name + "_taught",
                         "item": item, "cue_hex": cue.hex(), "target": target,
                         "emitted": result["emitted"], "values": result["values"],
                         "correct": int(result["emitted"] == target)})
    held = world_doc["old_relation"]["heldout"]
    for edge in range(3):
        good = bytes.fromhex(held[2 * edge]["cue_hex"])
        other = bytes.fromhex(held[2 * edge + 1]["cue_hex"])
        for order, options in enumerate(((good, other), (other, good))):
            result = relation_choice(system, *options, at)
            target = ord("L") if order == 0 else ord("R")
            rows.append({"stage": stage, "branch": branch, "set": "old_relation_heldout",
                         "item": edge, "order": order, "option_hex": [x.hex() for x in options],
                         "target": target, "emitted": result["emitted"],
                         "option_scores": result["option_scores"],
                         "option_values": result["option_values"],
                         "correct": int(result["emitted"] == target)})
    return rows


def run_fourstore_life(world: int, base: FourStore, *, technical: bool = False):
    """Full common life for one mechanism arm; caller owns source lock/saving."""
    doc = make_world(world)
    branches = {name: base.clone() for name in BRANCHES}
    birth = {name: sys.digests() for name, sys in branches.items()}
    if len({tuple(x) for x in birth.values()}) != 1:
        raise AssertionError("branch birth mismatch")
    first, probes = [], []
    writes = {name: {"old_fact": 0, "old_relation": 0, "new_fact": 0, "new_relation": 0}
              for name in BRANCHES}
    max_clock_error = 0.0
    for index, row in enumerate(doc["records"]):
        stage, declared = row["stage"], row["domain"]
        begin = index * RECORD_SECONDS + (86400.0 if stage == "new" else 0.0)
        cue, answer = bytes.fromhex(row["cue_hex"]), row["answer"]
        for branch, system in branches.items():
            if hasattr(system, "set_record_context"):
                system.set_record_context(world, branch, index, stage, declared)
            emitted, values, teacher_at, domain = output_before_teacher(system, cue, begin)
            if domain != declared:
                raise AssertionError("hidden task ID disagrees with bytes")
            first.append({"record": index, "branch": branch, "emitted": emitted,
                          "values": values, "teacher_at": teacher_at})
            allowed = branch_allows(branch, stage, domain)
            system.teach(answer, teacher_at, domain, write=allowed)
            if allowed:
                writes[branch][f"{stage}_{domain}"] += 4 if domain == "fact" else 2
            system.feed(10, begin + 13 * DT)
            max_clock_error = max(max_clock_error, system.flush(begin + RECORD_SECONDS))
        if index == 335:
            for branch, system in branches.items():
                probes.extend(roster_probes(system, doc, doc["old_end_s"], "old_end", branch))
                max_clock_error = max(max_clock_error, system.flush(doc["new_start_s"]))
                probes.extend(roster_probes(system, doc, doc["new_start_s"], "old_day", branch))
        if index == 599:
            for branch, system in branches.items():
                probes.extend(roster_probes(system, doc, doc["new_end_s"], "new_end", branch))
                max_clock_error = max(max_clock_error, system.flush(doc["final_s"]))
                probes.extend(roster_probes(system, doc, doc["final_s"], "final", branch))
    if max_clock_error >= 1e-6:
        raise AssertionError("native clock mismatch")
    return {"schema": "MINIFLY-THREE-MECHANISM-FOURSTORE-LIFE-v1", "world": world,
            "fixture_digest": doc["digest"], "technical": bool(technical),
            "branches": list(BRANCHES), "first": first, "probes": probes,
            "writes": writes, "max_clock_error": max_clock_error,
            "end_state_digests": {name: system.digests() for name, system in branches.items()},
            "mechanism_events": {name: getattr(system, "mechanism_events", [])
                                 for name, system in branches.items()}}
