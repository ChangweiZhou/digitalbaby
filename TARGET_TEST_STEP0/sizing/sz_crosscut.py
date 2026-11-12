"""Sizing sweep, stage 1: a dependency across a credit cut, on a fixture stream (brief rev. 3, arm B).

The fixture stream: episodes  K <filler of D - 2 bytes> = K \\n  back to back, K a letter A-X drawn at random, the filler from step 0's
filler bytes. The byte after '=' must repeat the key, which stands exactly D bytes earlier. Fed to the arm-B learner on its own training
path (sz_learner.StreamLearner: credit cut every 256 target positions, state carried, AdamW at E20-B's settings, every next byte a
target), from random weights.

Reading (prequential, no extra cuts, nothing changes the training): each chunk's predictions at the repeat positions are taken from its
forward pass, before its update, so no instance has been trained on when it is read. An instance CROSSES a cut when its key was read as
input in an earlier chunk than the one whose targets hold the repeat (the gradient from the repeat cannot reach the key's input).

Criterion, written before the run (D = 60 only): the dependency is shown learnable across a cut if, in the last quarter of the stream, at
least 100 crossing instances are read and at least 50% of them are right (chance 1/24). D = 600 (every instance crosses two or three cuts)
is run beside it and reported without a criterion.

Diagnostics added after the registered D = 60 run failed its criterion (both crossing and non-crossing instances at chance after 3,000
updates; the failure is recorded as it ran): D = 12, 24, 40 at 3,000 chunks and D = 60 at 10,000 chunks, reported without a criterion.
They do not replace the registered fixture.

    python3 sz_crosscut.py <D> <chunks> <seed> <out.json>
"""
import json
import random
import sys
import time
from pathlib import Path

import torch as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sz_learner as L     # noqa: E402

FILL = b"!#%&*+,-"


def fixture_stream(D, n_bytes, seed):
    r = random.Random(f"sizing-crosscut|{D}|{seed}")
    out = bytearray(); keys = []
    while len(out) < n_bytes:
        k = 65 + r.randrange(24)
        keys.append((len(out), len(out) + D, k))      # key byte position, repeat byte position, key
        out += bytes([k]) + bytes(r.choice(FILL) for _ in range(D - 2)) + b"=" + bytes([k]) + b"\n"
    return bytes(out), keys


def run(D, chunks, seed):
    T.set_num_threads(1)
    data, keys = fixture_stream(D, chunks * L.CHUNK, seed)
    data = data[:chunks * L.CHUNK]; keys = [k for k in keys if k[1] < len(data)]
    rep = {r: (p, k) for p, r, k in keys}
    lr = L.StreamLearner(seed, data, [])
    recs = []

    def observe(a, e, lg):
        pred = lg.argmax(1)
        for r in range(a, e):
            if r in rep:
                p, k = rep[r]
                recs.append((r, int(pred[r - a]) == k, p + 1 < a))     # key read as input at target p + 1; crossing if that is before this chunk
    lr.observe = observe
    t0 = time.process_time()
    lr.advance_to(len(data))
    cpu = time.process_time() - t0
    n = len(data)
    quart = []
    for qi in range(4):
        lo, hi = qi * n // 4, (qi + 1) * n // 4
        sub = [x for x in recs if lo <= x[0] < hi]
        cr = [x for x in sub if x[2]]; nc = [x for x in sub if not x[2]]
        quart.append(dict(quarter=qi + 1, crossing=len(cr), crossing_right=sum(x[1] for x in cr),
                          not_crossing=len(nc), not_crossing_right=sum(x[1] for x in nc)))
    last = quart[-1]
    out = dict(D=D, chunks=chunks, bytes=n, model_seed=seed, updates=lr.updates, instances=len(recs),
               crossing_share=sum(x[2] for x in recs) / max(1, len(recs)), chance=1 / 24, by_quarter=quart,
               last_quarter_crossing_accuracy=last["crossing_right"] / max(1, last["crossing"]),
               last_quarter_not_crossing_accuracy=last["not_crossing_right"] / max(1, last["not_crossing"]),
               cpu_seconds=round(cpu, 1), cpu_per_update=cpu / lr.updates)
    if D == 60 and chunks == 3000:          # the registered fixture; every other (D, chunks) is a diagnostic without a criterion
        out["criterion"] = "last quarter: at least 100 crossing instances, at least 50% right"
        out["passed"] = last["crossing"] >= 100 and last["crossing_right"] >= 0.5 * last["crossing"]
    return out


if __name__ == "__main__":
    D, chunks, seed, path = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    r = run(D, chunks, seed)
    Path(path).write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k != "by_quarter"}))
