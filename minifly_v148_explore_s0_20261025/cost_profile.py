"""S0 cost profile (no scores): where does one FastMemory teaching event spend its time? Dev text (Buster), 60 records.
Answers whether cost scales with the number of output compartments. python cost_profile.py
"""
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minifly_v148_pretrain_probe_dev_20261025"))
import corpus  # noqa: E402
import probe  # noqa: E402

recs = corpus.records(corpus.load("buster"))[:60]
mem = probe.new_memory()
pr = cProfile.Profile()
t0 = time.process_time()
pr.enable()
n = 0
for r in recs:
    mem.train(b"", r)
    n += len(r) + 1
pr.disable()
dt = time.process_time() - t0
st = pstats.Stats(pr)
cum = {f"{Path(k[0]).name}:{k[2]}": v[3] for k, v in st.stats.items()}
total = cum["fastmem.py:train"]
per_comp = cum["stores.py:teach_signed"] + cum["fastmem.py:values"] - cum["fastmem.py:encode"]
print(f"events {n}; CPU-s/event under the profiler {dt / n:.4f}")
print(f"per-compartment share (teach_signed + values - encode) {per_comp / total:.3f}; encode share {cum['fastmem.py:encode'] / total:.3f}")
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12)
print(s.getvalue())
