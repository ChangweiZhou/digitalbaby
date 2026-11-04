"""Interleaved reference/optimized comparison on the same engineering stream."""
import json
import resource
import time

import numpy as np

from engine import Life, T, new_sensor
from environment import seed, stream, twins

plain, fast = new_sensor(fast=False), new_sensor()
birth = seed("OPT|birth")
old, new = Life(birth, plain, fast=False), Life(birth, fast)
records = list(stream(seed("OPT|stream"), 1200, 32, ord("X")))
cpu = [0., 0.]
for i, event in enumerate(records):
    assert np.array_equal(plain.codes(event.record), fast.codes(event.record))
    for j in (0, 1) if i % 2 else (1, 0):
        before = time.process_time()
        (old, new)[j].train(event.record)
        cpu[j] += time.process_time() - before
    for key in old.model.p:
        assert all(T.equal(getattr(old.model, attr)[key], getattr(new.model, attr)[key])
                   for attr in ("p", "m", "v")), (i, key)
assert T.equal(old.model.rng.get_state(), new.model.rng.get_state())
assert old.model.clock == new.model.clock == old.events == new.events
for probe in twins(seed("OPT|target"), ord("X"), 16):
    assert old.generate(probe.prefix) == new.generate(probe.prefix)
    assert old.loss(probe.record) == new.loss(probe.record)
# Exercise long-record path, unseen byte values and the all-zero birth state.
for length in (0, 1, 63, 64, 127):
    record = bytes(np.random.default_rng(length).integers(0, 256, length, dtype=np.uint8))
    assert np.array_equal(plain.codes(record), fast.codes(record))
print(json.dumps(dict(events=1200, held_probes=32, bitwise_training_parity=True,
                      baseline_cpu_seconds=cpu[0], optimized_cpu_seconds=cpu[1],
                      speedup=cpu[0]/cpu[1], peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      cache=fast.prefix_state.cache_info()._asdict()), indent=2))
