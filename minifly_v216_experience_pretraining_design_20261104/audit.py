"""Independent reduction of saved exact answers and resource arithmetic."""
import json
from pathlib import Path

def read(file):
    return json.loads(Path(file).read_text())

r = read("CYCLE3_TEST.json")
tail = read("TAIL_TEST.json")
assert r["science"] == tail["science"] == "NOT_RUN"
groups = [x for a in ("learning", "adaptation") for arm in r[a].values() for x in arm.values()]
for row in groups:
    outputs, targets = row["outputs"], row["targets"]
    hits = [a == b for a, b in zip(outputs, targets, strict=True)]
    assert len(hits) == row["count"] == 8
    assert sum(hits) == row["exact"]
    assert sum(x[-1] == 256 for x in outputs) == row["eos"]
    assert sum(a and b for a, b in zip(hits[::2], hits[1::2])) == row["both"]
    assert all(0 < len(x) <= 32 and 256 not in x[:-1] for x in outputs)
projection = 1.25 * 1_600_000 * tail["tail_cpu_seconds"] / tail["tail_events"]
assert tail["checkpoint_continuation"]
print(json.dumps(dict(verified_panels=len(groups), verified_responses=len(groups)*8,
                      tail_projection_cpu_seconds=projection, admission="ADMISSION_PENDING",
                      science="NOT_RUN"), indent=2))
