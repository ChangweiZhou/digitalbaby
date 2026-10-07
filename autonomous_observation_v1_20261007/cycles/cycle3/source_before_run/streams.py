"""External stream generator. The learner receives only byte and clock, never this metadata."""
import hashlib
import itertools

import numpy as np
import spec


def world_inputs(world):
    spec.require_dev(world)
    rng = np.random.default_rng(world)
    keys = [b'3' + bytes(x) for x in itertools.product(b'012', repeat=3)]
    keys = [keys[i] for i in rng.permutation(len(keys))[:24]]
    old = dict(zip(keys[:16], rng.permutation(np.tile(list(spec.ALPHABET), 4)).tolist()))
    new = dict(zip(keys[16:], rng.permutation(np.tile(list(spec.ALPHABET), 2)).tolist()))
    revised = {k: spec.ALPHABET[(spec.ALPHABET.index(old[k]) + 1) % 4] for k in keys[:8]}
    return dict(old=old, new=new, revised=revised)


def stream(mapping, repeats, seed):
    rng = np.random.default_rng(seed)
    keys = list(mapping)
    out = bytearray()
    for _ in range(repeats):
        for i in rng.permutation(len(keys)):
            k = keys[i]
            out.extend(k)
            out.append(mapping[k])
    raw = bytes(out)
    return raw


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def assert_contexts(raw, mapping, initial=b''):
    """Independent sliding window audit: focal contexts appear ONLY with declared actual successors."""
    h = initial
    seen = {k: 0 for k in mapping}
    for b in raw:
        if h in mapping:
            if b != mapping[h]:
                raise AssertionError('focal context has contradictory successor')
            seen[h] += 1
        h = (h + bytes([b]))[-4:]
    if any(n == 0 for n in seen.values()):
        raise AssertionError('focal context not observed')
    return seen
