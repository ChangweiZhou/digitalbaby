"""Profile an actual post-warmup block with a disk checkpoint boundary."""
import json
from pathlib import Path
import resource
import tempfile
import time

from engine import Life, T, new_sensor
from environment import seed, stream

birth = seed("3|birth")
life = Life(birth, new_sensor())
events = iter(stream(seed("3|train"), 1256, 32, ord("X")))
for _ in range(1000):
    life.train(next(events).record)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "checkpoint.pt"
    T.save(dict(events=life.events, state=life.model.state()), path)
    size = path.stat().st_size
    restored = Life(birth, life.sensor)
    saved = T.load(path, weights_only=True)
    restored.model.restore(saved["state"])
    restored.events = saved["events"]
    first = next(events)
    life.train(first.record)
    restored.train(first.record)
    assert all(T.equal(restored.model.p[k], life.model.p[k]) for k in life.model.p)
    assert T.equal(restored.model.rng.get_state(), life.model.rng.get_state())
    before = time.process_time()
    for event in events:
        restored.train(event.record)
    elapsed = time.process_time() - before
assert restored.events == 1256
print(json.dumps(dict(checkpoint_bytes=size, checkpoint_continuation=True, tail_events=255,
                      tail_cpu_seconds=elapsed, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      science="NOT_RUN"), indent=2))
