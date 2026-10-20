"""Deterministic, label-blind byte task for the three-mechanism round.

This module generates fixtures only. It does not instantiate or run a learner.
All labels, teaching events and score probes are sealed before science.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

SCHEMA = "MINIFLY-THREE-MECHANISM-FIXTURE-v1"
DT = 30.0 / 14.0
RECORD_SECONDS = 165.0
DAY_SECONDS = 86400.0
BOUTS = 12
OLD_DIGITS = b"0123"
NEW_DIGITS = b"4567"
SYMBOL_POOL = b"ABCDEFGHJKLMNPQRSTUVWXYZ"
OLD_FACT_COUNT = 16
NEW_FACT_COUNT = 16
OLD_REL_TAUGHT = 12
OLD_REL_WITHHELD = 6
NEW_REL_TAUGHT = 6


def _rng(world: int, domain: str) -> random.Random:
    if not isinstance(world, int) or world < 0:
        raise ValueError("world must be a nonnegative integer")
    raw = hashlib.sha256(f"{SCHEMA}|{world}|{domain}".encode()).digest()
    return random.Random(int.from_bytes(raw[:16], "big"))


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def fact_cue(a: int, b: int) -> bytes:
    if not 48 <= a <= 55 or not 48 <= b <= 55:
        raise ValueError("fact operands must be ASCII 0..7")
    return b" " * 8 + bytes((a, 43, b, 61))


def relation_cue(a: int, b: int) -> bytes:
    if a == b or a not in SYMBOL_POOL or b not in SYMBOL_POOL:
        raise ValueError("invalid relation symbols")
    return b"##" + b" " * 7 + bytes((a, b, 32))


def fact_variants(cue: bytes) -> dict[str, bytes]:
    if len(cue) != 12 or cue[:8] != b" " * 8 or cue[9] != 43 or cue[11] != 61:
        raise ValueError("invalid fact cue")
    a, b = cue[8], cue[10]
    out = {
        "spacing": b" " * 7 + bytes((a, 32, 43, b, 61)),
        "prefix": b"Q" + b" " * 7 + bytes((a, 43, b, 61)),
        "inner_marker": b" " * 7 + bytes((a, 43, 126, b, 61)),
    }
    if any(len(x) != 12 or x == cue for x in out.values()):
        raise AssertionError("invalid fact transform")
    return out


def _fact_set(world: int, domain: str, digits: bytes) -> dict[str, Any]:
    pairs = [(int(a), int(b)) for a in digits for b in digits]
    labels = [j for j in range(4) for _ in range(4)]
    _rng(world, domain + "-labels").shuffle(labels)
    cues = [fact_cue(a, b) for a, b in pairs]
    if len(set(cues)) != 16 or sorted(labels) != [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4:
        raise AssertionError("invalid fact roster")
    return {"pairs": pairs, "cues": cues, "labels": labels}


def _relation_set(world: int, domain: str, symbols: bytes,
                  *, old: bool) -> dict[str, Any]:
    if len(symbols) != 6 or len(set(symbols)) != 6:
        raise ValueError("relation set needs six unique symbols")
    left, right = tuple(symbols[:3]), tuple(symbols[3:])
    edges = [(left[i], right[j]) for i in range(3) for j in range(3)]
    # The old held-out perfect matching ensures every symbol occurs in two
    # different trained pairings. Neither orientation of those edges is taught.
    withheld = [(left[i], right[i]) for i in range(3)] if old else []
    taught_edges = [e for e in edges if e not in withheld]
    if not old:
        assignment = list(right)
        _rng(world, domain + "-new-edges").shuffle(assignment)
        taught_edges = list(zip(left, assignment, strict=True))
    taught: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    for a, b in taught_edges:
        taught.extend(({"cue": relation_cue(a, b), "label": 1, "edge": (a, b)},
                       {"cue": relation_cue(b, a), "label": 0, "edge": (a, b)}))
    for a, b in withheld:
        held.extend(({"cue": relation_cue(a, b), "label": 1, "edge": (a, b)},
                     {"cue": relation_cue(b, a), "label": 0, "edge": (a, b)}))
    if (len(taught) != (12 if old else 6) or len(held) != (6 if old else 0) or
            {r["cue"] for r in taught} & {r["cue"] for r in held}):
        raise AssertionError("invalid relation roster")
    return {"left": left, "right": right, "taught": taught, "heldout": held}


def _schedule(world: int, domain: str, n: int) -> list[int]:
    rng = _rng(world, domain + "-schedule")
    order: list[int] = []
    for _ in range(BOUTS):
        bout = list(range(n))
        rng.shuffle(bout)
        order.extend(bout)
    return order


def _merge(facts: list[dict[str, Any]], relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a, b = len(facts), len(relations)
    out = []
    fi = ri = 0
    for j in range(a + b):
        take_fact = ((j + 1) * a) // (a + b) > (j * a) // (a + b)
        if take_fact:
            out.append(facts[fi]); fi += 1
        else:
            out.append(relations[ri]); ri += 1
    if (fi, ri) != (a, b):
        raise AssertionError("interleave count mismatch")
    return out


def make_world(world: int) -> dict[str, Any]:
    old_f = _fact_set(world, "old-fact", OLD_DIGITS)
    new_f = _fact_set(world, "new-fact", NEW_DIGITS)
    symbols = list(SYMBOL_POOL)
    _rng(world, "relation-symbols").shuffle(symbols)
    old_r = _relation_set(world, "old-rel", bytes(symbols[:6]), old=True)
    new_r = _relation_set(world, "new-rel", bytes(symbols[6:12]), old=False)

    def records(stage: str, f: dict[str, Any], r: dict[str, Any]) -> list[dict[str, Any]]:
        fs = [{"stage": stage, "domain": "fact", "cue": f["cues"][k],
               "answer": 48 + f["labels"][k], "item": k}
              for k in _schedule(world, stage + "-fact", len(f["cues"]))]
        rs = [{"stage": stage, "domain": "relation", "cue": r["taught"][k]["cue"],
               "answer": 48 + r["taught"][k]["label"], "item": k}
              for k in _schedule(world, stage + "-relation", len(r["taught"]))]
        return _merge(fs, rs)

    old_records = records("old", old_f, old_r)
    new_records = records("new", new_f, new_r)
    all_records = old_records + new_records
    if (len(old_records), len(new_records), len(all_records)) != (336, 264, 600):
        raise AssertionError("record count mismatch")
    for i, rec in enumerate(all_records):
        rec["index"] = i
        rec["cue_hex"] = rec.pop("cue").hex()
    heldout = [{"cue_hex": r["cue"].hex(), "label": r["label"]}
               for r in old_r["heldout"]]
    trained_cues = {r["cue_hex"] for r in all_records}
    if len(heldout) != OLD_REL_WITHHELD or any(x["cue_hex"] in trained_cues for x in heldout):
        raise AssertionError("held-out leakage")
    doc = {"schema": SCHEMA, "world": world,
           "old_fact": {"cues_hex": [x.hex() for x in old_f["cues"]],
                        "labels": old_f["labels"]},
           "new_fact": {"cues_hex": [x.hex() for x in new_f["cues"]],
                        "labels": new_f["labels"]},
           "old_relation": {"left": list(old_r["left"]), "right": list(old_r["right"]),
                            "taught": [{"cue_hex": x["cue"].hex(), "label": x["label"]}
                                        for x in old_r["taught"]],
                            "heldout": heldout},
           "new_relation": {"left": list(new_r["left"]), "right": list(new_r["right"]),
                            "taught": [{"cue_hex": x["cue"].hex(), "label": x["label"]}
                                        for x in new_r["taught"]]},
           "records": all_records,
           "old_end_s": len(old_records) * RECORD_SECONDS,
           "new_start_s": len(old_records) * RECORD_SECONDS + DAY_SECONDS,
           "new_end_s": len(all_records) * RECORD_SECONDS + DAY_SECONDS,
           "final_s": len(all_records) * RECORD_SECONDS + 2 * DAY_SECONDS}
    doc["digest"] = _sha(doc)
    return doc
