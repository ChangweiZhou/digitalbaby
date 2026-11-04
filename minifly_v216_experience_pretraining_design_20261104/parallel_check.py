"""Compare serial and parallel scheduling of isolated engineering lives."""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def job(_):
    result = subprocess.run([sys.executable, "-B", "run.py", "--cycle", "3"],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def schedule(workers):
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(job, range(2)))
    return time.monotonic()-start, rows


serial, before = schedule(1)
parallel, after = schedule(2)
for a, b in zip(before, after):
    assert a["learning"] == b["learning"] and a["adaptation"] == b["adaptation"]
rss = sum(r["peak_rss_native_units"] for r in after)
print(json.dumps(dict(jobs=2, serial_wall_seconds=serial, parallel_wall_seconds=parallel,
                      wall_speedup=serial/parallel, sum_child_peak_rss_bytes=rss,
                      same_outputs=True, science="NOT_RUN"), indent=2))
