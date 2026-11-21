"""Are two runs' records the same? Receipts compared field by field, leaving out times, file digests and the switch list."""
import sys, json, numpy as np
from pathlib import Path
a, b = Path(sys.argv[1]), Path(sys.argv[2])
skip = {"cpu_total_s", "cpu_per_byte_with_records", "wall_s", "cpu_core_only_s", "cpu_per_byte_core_only", "cpu_by_stretch",
        "command", "switches", "core_sha256", "wrapper_sha256", "runner_sha256", "versions"}
ra, rb = json.loads((a / "receipt.json").read_text()), json.loads((b / "receipt.json").read_text())
bad = [k for k in sorted(set(ra) | set(rb)) if k not in skip and ra.get(k) != rb.get(k)]
for f in sorted(p.name for p in a.iterdir() if p.suffix in (".npy", ".npz")):
    x, y = np.load(a / f), np.load(b / f)
    if f.endswith(".npy"):
        same = np.array_equal(x, y)
    else:
        same = set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
    if not same:
        bad.append(f)
ca = [{k: v for k, v in d.items() if k not in ("cpu_s", "per_byte")} for d in ra["cpu_by_stretch"]]
cb = [{k: v for k, v in d.items() if k not in ("cpu_s", "per_byte")} for d in rb["cpu_by_stretch"]]
if ca != cb: bad.append("cpu_by_stretch (non-time fields)")
print("SAME RECORDS" if not bad else "DIFFER: " + ", ".join(bad))
