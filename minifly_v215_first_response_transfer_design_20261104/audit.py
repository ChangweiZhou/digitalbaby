"""Recompute engineering receipts, including inference and control boundaries."""
import json
import re
from pathlib import Path

from controls import qualified
from cycle1 import exact
from worlds import family

report = json.loads(Path("CYCLE3_TEST.json").read_text())
assert report["science"] == "NOT_RUN"
for row in report["families"]:
    histories, interference, old, pairs = family(row["seed"])
    assert len(histories[0]) == len(histories[1])
    assert len(interference) <= 2048
    probes = [p for twin in pairs for p in twin]
    shortcuts = []
    for probe in probes:
        locations = re.findall(rb" (?:went|moved) to the ([a-z]+)\.", probe.prompt)
        # Last mentioned location and last non-distractor location.
        assert len(locations) >= 3
        shortcuts.append(locations[1] == probe.answer)
    assert sum(shortcuts) == 8
    assert all(sum(shortcuts[i:i+2]) == 1 for i in range(0, 16, 2))
    for result in row["arms"].values():
        hits = []
        for reply, probe in zip(sum(result["responses"], []), probes, strict=True):
            raw = bytes.fromhex(reply["hex"])
            hits.append(exact(raw, probe.answer))
            assert len(raw) <= 32 and reply["correct"] == hits[-1]
            assert reply["terminated"] == raw.endswith(b"\n")
        assert sum(hits[:8]) == result["short"] and sum(hits[8:]) == result["long"]
        assert sum(all(hits[i:i+2]) for i in range(0, 16, 2)) == result["both"]
    matched, passed = qualified(row["control_counters"]["TRUE"], row["control_counters"]["SHAM"])
    assert matched == row["matching"] and passed == row["sham_qualified"]
print(json.dumps(dict(verified_families=len(report["families"]),
                      verified_main_probes=len(report["families"])*4*16,
                      recency_correct_per_family=8, recency_both_correct=0)))
