"""Engineering worlds. IDs and gold values stay outside the model."""
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class Probe:
    prompt: bytes
    answer: bytes


def story(person, item, a, b, before_move):
    move = f"{person} moved to the {b}.\n"
    drop = f"{person} dropped the {item}.\n"
    order = drop + move if before_move else move + drop
    return f"{person} went to the {a}.\n{person} got the {item}.\n" + order


def oracle(events, person, item):
    """Independent parser for these engineering fixture sentences."""
    position = None
    held = False
    location = None
    for line in events.splitlines():
        if line.startswith((f"{person} went to the ", f"{person} moved to the ")):
            position = line.split(" the ", 1)[1].rstrip(".")
        elif line == f"{person} got the {item}.":
            held = True
        elif line == f"{person} dropped the {item}.":
            assert held
            held, location = False, position
    assert location and not held
    return location.encode()


def family(seed):
    rng = Random(seed)
    tokens = ["".join(rng.sample("abcdefghijklmnopqrstuvxyz", 5)) for _ in range(40)]
    assert len(set(tokens)) == len(tokens)
    a, b, c, d = tokens[:4]
    # The taught object-location query concerns a stationary pickup/drop.
    # The person-location query has exactly the same byte length.
    train = f"{a} went to the {b}.\n{a} got the {c}.\n{a} dropped the {c}.\n"
    teach = f"Where is the {c}? {b}\n"
    unrelated = f"Where is now {a}? {b}\n"
    assert len(teach) == len(unrelated)
    histories = ((train + teach).encode() * 4, (train + unrelated).encode() * 4)
    interference = f"{d} went to the {b}.\n".encode() * 3
    old = Probe(f"Where is the {c}? ".encode(), b.encode())
    pairs = []
    for index in range(8):
        person, item, first, second = tokens[4 + index * 4:8 + index * 4]
        probes = []
        for drop_first in (True, False):
            events = story(person, item, first, second, drop_first)
            label = oracle(events, person, item)
            assert label == (first if drop_first else second).encode()
            delay = f"{tokens[-2]} went to the {tokens[-1]}.\n" * (3 if index >= 4 else 1)
            probes.append(Probe((events + delay + f"Where is the {item}? ").encode(), label))
        assert sorted(probes[0].prompt.splitlines()) == sorted(probes[1].prompt.splitlines())
        assert probes[0].prompt.splitlines()[-1] == probes[1].prompt.splitlines()[-1]
        assert probes[0].answer != probes[1].answer
        assert all(probe.prompt.splitlines()[-1] not in histories[0] + histories[1] for probe in probes)
        pairs.append(tuple(probes))
    return histories, interference, old, pairs
