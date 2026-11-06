"""Raw-byte episodes and paired probes for the reasoning engineering cycles."""

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from random import Random

EOS = 256
CHAIN = ((0, 1), (1, 3))
ALTERNATE = CHAIN + ((0, 3),)
DIAMOND = CHAIN + ((0, 2), (2, 3))
DISTRACTORS = ((4, 5), (5, 6), (4, 6), (6, 7))
ALPHABETS = {"A": b"XY", "B": b"UV", "A_prime": b"XY"}


def seed(*parts):
    payload = "|".join(str(part) for part in ("REASONING-SCALING", *parts)).encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class Record:
    prefix: bytes
    target: int

    @property
    def observed(self):
        return self.prefix + bytes((self.target,))


def active_opcode(world, phase, edit=True):
    alphabet = ALPHABETS[phase]
    old_index = seed(world, alphabet.hex(), "rule") % 2
    index = old_index ^ (phase == "A_prime" and edit)
    return alphabet[index]


def frame(edges, actors, locations, alphabet, active, active_edges, order, moves):
    inactive = next(op for op in alphabet if op != active)
    state = [token for who, where in zip(actors, locations[:8]) for token in (ord("O"), who, where)]
    links = [
        token for i in order for token in (
            active if edges[i] in active_edges else inactive,
            actors[edges[i][0]], actors[edges[i][1]],
        )
    ]
    prefix = bytes((*state, *links, ord("M"), actors[0], locations[8], ord("Q"), actors[3]))
    assert len(prefix) == 47
    return Record(prefix, locations[8] if moves else locations[3])


def training(world, phase, diversity, events, *, edit=True):
    """A dose prefix is literal. Each 16-record block has all taught witnesses."""
    if diversity not in (4, 32) or events <= 0 or events % 16:
        raise ValueError("Use 4 or 32 instances and a positive multiple of 16 records")
    namespace = 160 if phase == "B" else 128
    alphabet = ALPHABETS[phase]
    old_active = active_opcode(world, phase, edit=False)
    locations = tuple(b"0123456789")
    templates = []
    used = set()
    for instance in range(diversity):
        rng = Random(seed(world, phase, "environment", instance))
        actors = tuple(rng.sample(range(namespace, namespace + 8), 8))
        layout = tuple(rng.sample(locations, len(locations)))
        order = tuple(rng.sample(range(6), 6))
        template = actors, layout, order
        assert template not in used
        used.add(template)
        templates.append(template)
    variation = Random(seed(world, phase, "location-relabeling"))
    for index in range(events):
        block, slot = divmod(index, 16)
        if slot == 0:
            relabel = dict(zip(locations, variation.sample(locations, len(locations))))
        actors, layout, order = templates[block % diversity]
        offset = slot % 8
        rotated_actors = actors[offset:] + actors[:offset]
        offset = (slot + block) % len(layout)
        rotated_locations = tuple(relabel[byte] for byte in layout[offset:] + layout[:offset])
        edges = CHAIN if slot < 8 else ALTERNATE
        bits = (slot % 4) if slot < 8 else slot - 8
        mask = tuple(bool(bits & (1 << i)) for i in range(len(edges)))
        full_edges = edges + DISTRACTORS[:6 - len(edges)]
        live_edges = {edge for edge, flag in zip(edges, mask) if flag}
        live_edges.update(edge for i, edge in enumerate(full_edges[len(edges):]) if (block + slot + i) % 2)
        if phase == "A_prime" and edit:
            mask = tuple(not flag for flag in mask)
        moves = all(mask[:2]) or (len(edges) == 3 and mask[2])
        yield frame(full_edges, rotated_actors, rotated_locations, alphabet, old_active, live_edges, order, moves)


def probes(world, phase, kind, *, edit=True):
    """Matched panels. The polarity probes keep their literal prefixes after reversal."""
    if kind not in ("structure", "invariant", "polarity"):
        raise ValueError(kind)
    namespace = 208 if phase == "B" else 192
    alphabet = ALPHABETS[phase]
    original_active = active_opcode(world, phase, edit=False)
    edges = DIAMOND + DISTRACTORS[:2]
    pairs = []
    for group in range(4):
        rng = Random(seed(world, alphabet.hex(), "held", kind, group))
        actor_order = rng.sample(range(namespace, namespace + 8), 8)
        locations = rng.sample(list(b"0123456789"), 10)
        order = rng.sample(range(6), 6)
        for route, hop, reverse in product(range(2), repeat=3):
            offset = 4 * route + 2 * hop + reverse
            actors = actor_order[offset:] + actor_order[:offset]
            places = locations[offset:] + locations[:offset]
            original = set(DIAMOND[2 * route:2 * route + 2]) | {edges[4]}
            if kind == "structure":
                changed = original - {DIAMOND[2 * route + hop]} | {DIAMOND[2 * (1 - route) + hop]}
                moves = (True, False)
            elif kind == "invariant":
                changed = original - {edges[4]} | {edges[5]}
                moves = (True, True)
            else:
                original = {DIAMOND[2 * route + hop], edges[4]}
                changed = (set(DIAMOND) - original) | {edges[5]}
                moves = (False, True) if phase != "A_prime" or not edit else (True, False)
            pair = tuple(
                frame(edges, actors, places, alphabet, original_active, live, order, move)
                for live, move in zip((original, changed), moves)
            )
            pairs.append(pair[::-1] if reverse else pair)
    return pairs
