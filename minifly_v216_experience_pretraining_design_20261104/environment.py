"""Engineering byte episodes, with truth kept outside the learner."""
from dataclasses import dataclass
from hashlib import sha256
from random import Random


@dataclass(frozen=True)
class Episode:
    prefix: bytes
    target: bytes

    @property
    def record(self):
        return self.prefix + self.target


def make(rng, actors, op, attach, held=False):
    a, b, c, d = actors
    left, right, dest = rng.sample(b"01234567", 3)
    # The held motif contains a second, irrelevant pair action before the move.
    prefix = bytes([79, a, left, 79, b, right, op, a, b])
    if held:
        prefix += bytes([88, c, d])
    prefix += bytes([77, a, dest, 81, b])
    return Episode(prefix, bytes([dest if op == attach else right]))


def oracle(prefix, attach):
    initial = dict(zip(prefix[1:6:3], prefix[2:6:3]))
    op, a, b = prefix[6:9]
    assert prefix[-5:-3] == bytes([77, a]) and prefix[-1] == b
    return bytes([prefix[-3] if op == attach else initial[b]])


def stream(seed, count, worlds, attach, namespace=0):
    rng = Random(seed)
    actors = list(range(128 + namespace, 160 + namespace))
    panels = [rng.sample(actors, 4) for _ in range(worlds)]
    assert len({tuple(panel) for panel in panels}) == worlds
    for i in range(count):
        # Cycle through every world before advancing its within-world dose.
        yield make(rng, panels[i % worlds], b"XY"[(i//worlds+i%worlds)%2], attach)


def twins(seed, attach, count=4):
    rng = Random(seed)
    pairs = []
    for _ in range(count):
        actors = rng.sample(range(192, 224), 4)
        key = rng.randrange(1 << 30)
        left, right = (make(Random(key), actors, op, attach, held=True) for op in b"XY")
        assert left.prefix[7:] == right.prefix[7:] and left.target != right.target
        assert all(oracle(x.prefix, attach) == x.target for x in (left, right))
        pairs.extend((left, right))
    return pairs


def exposure(records, panel):
    # Verify the focal actor namespace and the distractor motif separately.
    assert all(len(x.prefix) == 14 for x in records)
    assert all(len(x.prefix) == 17 and 192 <= x.prefix[1] < 224 for x in panel)
    assert all(x.prefix[1] < 192 for x in records)
    assert not ({x.record for x in records} & {x.record for x in panel})


def seed(label):
    return int.from_bytes(sha256(("V216|ENGINEERING|" + label).encode()).digest()[:8], "big")
