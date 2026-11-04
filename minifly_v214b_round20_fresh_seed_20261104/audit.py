"""Check archived provenance, stream alignment and committed scoring outputs."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

out = Path("results")
lock = json.loads((out / "LOCK.json").read_text())
recovery = json.loads(Path("RECOVERY.json").read_text())
summary = json.loads((out / "SUMMARY.json").read_text())
spans = json.loads((out / "SPAN_INDEX.json").read_text())

for path, info in recovery["sources"].items():
    assert sha256(Path(path).read_bytes()).hexdigest() == info["sha256"]
streams = {}
for path, info in lock["files"].items():
    raw = (out / path).read_bytes()
    assert len(raw) == info["bytes"]
    assert sha256(raw).hexdigest() == info["sha256"]
    streams[path] = raw

assert len(spans["mixed"]) == 1500 and len(spans["target"]) == 80
assert len(lock["jobs"]) == 7
assert sum(len(streams[f"{stem}.txt"]) for stem, _ in lock["jobs"]) == summary["charged_bytes"]
arrays = {}
for stem, mode in lock["jobs"]:
    scores = np.load(out / f"{stem}.txt.bits_{mode}.npy", allow_pickle=False)
    assert len(scores) == len(streams[f"{stem}.txt"])
    assert np.isfinite(scores).all() and (scores >= 0).all()
    receipt = json.loads((out / f"{stem}_{mode}.log").read_text())
    assert receipt["REC"] == mode and receipt["file"] == f"{stem}.txt"
    assert round(float(scores.mean()), 4) == receipt["bpb"]
    arrays[stem, mode] = scores

target = streams["tr_0.txt"]
for k, offset in lock["offsets"].items():
    stem = f"tr_{k}"
    assert streams[f"{stem}.txt"][offset:] == target
    assert streams[f"{stem}.txt"][:offset] == streams["tr_24.txt"][:offset]
    assert np.array_equal(arrays[stem, "1"][:offset], arrays["tr_24", "1"][:offset])
    assert np.isclose(arrays[stem, "1"][offset:].sum(), summary["transfer"][k]["target_bits"])

audit = {
    "status": "PASS_LOCAL",
    "verified_source_hashes": len(recovery["sources"]),
    "verified_stream_hashes": len(streams),
    "verified_scoring_receipts": len(arrays),
    "mixed_answer_spans": len(spans["mixed"]),
    "target_answer_spans": len(spans["target"]),
    "charged_bytes": summary["charged_bytes"],
    "shared_history_prefix_losses_identical": True,
    "target_bytes_identical_across_checkpoints": True,
}
(out / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
