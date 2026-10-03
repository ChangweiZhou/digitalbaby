"""Independent byte/exposure audit of a generated mixed-life fixture."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from fixture import (DAY_SECONDS, NEW_DIGITS, OLD_DIGITS, RECORD_SECONDS,
                     SCHEMA, SYMBOL_POOL)


def _digest(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k != "digest"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def audit(doc: dict) -> dict:
    if doc.get("schema") != SCHEMA or doc.get("digest") != _digest(doc):
        raise AssertionError("fixture schema/digest")
    records = doc.get("records")
    if not isinstance(records, list) or len(records) != 600:
        raise AssertionError("fixture record count")
    if [(r["stage"], r["domain"]) for r in records].count(("old", "fact")) != 192:
        raise AssertionError("old fact count")
    expected_counts = {("old", "fact"): 192, ("old", "relation"): 144,
                       ("new", "fact"): 192, ("new", "relation"): 72}
    if Counter((r["stage"], r["domain"]) for r in records) != expected_counts:
        raise AssertionError("phase/domain counts")
    if any(r["index"] != i for i, r in enumerate(records)):
        raise AssertionError("record order")
    if any(r["stage"] != ("old" if i < 336 else "new")
           for i, r in enumerate(records)):
        raise AssertionError("phase boundary")

    rosters: dict[tuple[str, str], dict[str, int]] = {}
    for stage in ("old", "new"):
        fact = doc[stage + "_fact"]
        cues = [bytes.fromhex(h) for h in fact["cues_hex"]]
        digits = OLD_DIGITS if stage == "old" else NEW_DIGITS
        if (len(cues) != 16 or len(set(cues)) != 16 or
                sorted(fact["labels"]) != [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 or
                any(c[:8] != b" " * 8 or c[8] not in digits or
                    c[9] != ord("+") or c[10] not in digits or c[11] != ord("=")
                    for c in cues)):
            raise AssertionError("fact roster")
        rosters[(stage, "fact")] = {c.hex(): 48 + y
                                       for c, y in zip(cues, fact["labels"], strict=True)}

        relation = doc[stage + "_relation"]
        left, right = relation["left"], relation["right"]
        if (len(left) != 3 or len(right) != 3 or len(set(left + right)) != 6 or
                any(x not in SYMBOL_POOL for x in left + right)):
            raise AssertionError("relation symbol roster")
        mapping: dict[str, int] = {}
        edge_pairs = set()
        for item in relation["taught"]:
            c = bytes.fromhex(item["cue_hex"])
            if (len(c) != 12 or c[:9] != b"##" + b" " * 7 or c[11] != 32 or
                    c[9] == c[10] or
                    not ((c[9] in left and c[10] in right) or
                         (c[9] in right and c[10] in left)) or
                    item["label"] != int(c[9] in left) or c.hex() in mapping):
                raise AssertionError("relation training roster")
            mapping[c.hex()] = 48 + item["label"]
            edge_pairs.add(tuple(sorted(c[9:11])))
        if len(mapping) != (12 if stage == "old" else 6) or len(edge_pairs) * 2 != len(mapping):
            raise AssertionError("relation orientation coverage")
        if stage == "old" and any(
                sum(c[9] == symbol or c[10] == symbol for c in map(bytes.fromhex, mapping)) != 4
                for symbol in left + right):
            raise AssertionError("old symbol teaching degree")
        rosters[(stage, "relation")] = mapping

    old_symbols = set(doc["old_relation"]["left"] + doc["old_relation"]["right"])
    new_symbols = set(doc["new_relation"]["left"] + doc["new_relation"]["right"])
    if old_symbols & new_symbols:
        raise AssertionError("old/new relation symbols overlap")
    for r in records:
        key = (r["stage"], r["domain"])
        if r["cue_hex"] not in rosters[key] or r["answer"] != rosters[key][r["cue_hex"]]:
            raise AssertionError("record cue/answer mismatch")
    for key, roster in rosters.items():
        counts = Counter(r["cue_hex"] for r in records
                         if (r["stage"], r["domain"]) == key)
        if set(counts) != set(roster) or set(counts.values()) != {12}:
            raise AssertionError("teaching exposure count")

    held = doc["old_relation"]["heldout"]
    if len(held) != 6 or len({h["cue_hex"] for h in held}) != 6:
        raise AssertionError("heldout roster size")
    for h in held:
        c = bytes.fromhex(h["cue_hex"])
        if (len(c) != 12 or c[:9] != b"##" + b" " * 7 or c[11] != 32 or
                not (c[9] in old_symbols and c[10] in old_symbols) or
                h["label"] != int(c[9] in doc["old_relation"]["left"]) or
                h["cue_hex"] in rosters[("old", "relation")]):
            raise AssertionError("heldout leakage/label")
    for i in range(0, 6, 2):
        a, b = held[i], held[i + 1]
        ca, cb = bytes.fromhex(a["cue_hex"]), bytes.fromhex(b["cue_hex"])
        if (a["label"] != 1 or b["label"] != 0 or
                ca[9] != cb[10] or ca[10] != cb[9] or
                ca[:9] != cb[:9] or ca[11:] != cb[11:]):
            raise AssertionError("heldout option pairing/order")
    if (doc["old_end_s"] != 336 * RECORD_SECONDS or
            doc["new_start_s"] != doc["old_end_s"] + DAY_SECONDS or
            doc["new_end_s"] != doc["new_start_s"] + 264 * RECORD_SECONDS or
            doc["final_s"] != doc["new_end_s"] + DAY_SECONDS):
        raise AssertionError("life clock")
    return {"pass": True, "world": doc["world"], "records": len(records),
            "heldout_orientations": len(held), "fixture_digest": doc["digest"]}
