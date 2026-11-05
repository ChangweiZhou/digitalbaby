"""Raw-symbol records and evaluator-only manifests for the re-entry experiment."""
import hashlib
import random
import string

EOS = 256
WIDTHS = [2] + [3] * 15 + [4] * 16


def addition(a, b):
    return (f'{a}+{b}='.encode(), str(a + b).encode())


def fact(name, value):
    return (f'f/{name}='.encode(), f'{value:03d}'.encode())


def orbit(a, b):
    return tuple(sorted((a, b)))


def tokens(records):
    return [y for prompt, answer in records for y in (*prompt, *answer, EOS)]


def exposure_digest(records):
    import numpy as np
    return hashlib.sha256(np.asarray(tokens(records), dtype='<u2').tobytes()).hexdigest()


def columns(a, b):
    width = len(str(a))
    carry = 0
    result, outgoing = set(), []
    for index in range(width):
        left, right = a % 10, b % 10
        role = 'units' if index == 0 else 'leading' if index == width - 1 else 'interior'
        result.add((role, *sorted((left, right)), carry))
        carry = int(left + right + carry >= 10)
        outgoing.append(carry)
        a, b = a // 10, b // 10
    stratum = 3 if outgoing[-1] else min(2, sum(outgoing))
    return result, stratum


def sample_pair(rng, width):
    return tuple(rng.randrange(10 ** (width - 1), 10 ** width) for _ in range(2))


def unique_pair(rng, width, used):
    while True:
        pair = sample_pair(rng, width)
        if orbit(*pair) not in used:
            used.add(orbit(*pair))
            return pair


def make_world(seed):
    """Construct one full world. Call on development seeds before sealing any final roster."""
    rng = random.Random(seed)
    used = set()
    first = []
    for _ in range(128):
        widths = WIDTHS.copy()
        rng.shuffle(widths)
        first.extend(unique_pair(rng, width, used) for width in widths)
    witness = set().union(*(columns(*pair)[0] for pair in first))
    primary, extrapolation, counterfactual = [], [], []
    for width in (3, 4, 5):
        for stratum in range(4):
            count = 0
            for _ in range(50000):
                pair = sample_pair(rng, width)
                required, kind = columns(*pair)
                if kind != stratum or orbit(*pair) in used or (width < 5 and not required <= witness):
                    continue
                used.add(orbit(*pair))
                (primary if width < 5 else extrapolation).append((width, stratum, pair))
                count += 1
                if count == 64:
                    break
            if count != 64:
                raise ValueError(f"Insufficient held support for seed {seed}, width {width}, stratum {stratum}")
    for width in (3, 4):
        count = 0
        for _ in range(50000):
            a, b = sample_pair(rng, width)
            replacement = a // 10 * 10 + rng.randrange(10)
            left, right = (a, b), (replacement, b)
            # Keep the edit in the units column and require a changed higher output column.
            if (a + b) // 10 == (replacement + b) // 10:
                continue
            if orbit(*left) in used or orbit(*right) in used or orbit(*left) == orbit(*right):
                continue
            if not (columns(*left)[0] | columns(*right)[0]) <= witness:
                continue
            used.update((orbit(*left), orbit(*right)))
            counterfactual.append((left, right))
            count += 1
            if count == 64:
                break
        if count != 64:
            raise ValueError(f"Insufficient counterfactual support for seed {seed}")
    adds = first.copy()
    for _ in range(2048 - 128):
        widths = WIDTHS.copy()
        rng.shuffle(widths)
        adds.extend(unique_pair(rng, width, used) for width in widths)
    additions = [adds, [unique_pair(rng, 4, used) for _ in range(8192)],
                 [unique_pair(rng, 4, used) for _ in range(4096)]]
    names = {}
    while len(names) < 128:
        names[''.join(rng.choices(string.ascii_lowercase, k=5))] = None
    names = list(names)
    values = rng.sample(range(1000), 160)
    assignments = [list(zip(names[:64], values[:64])), list(zip(names[64:], values[64:128])),
                   list(zip(names[:32], values[128:]))]
    phases = []
    for pairs, assignments_for_phase, stride in zip(additions, assignments, (16, 4, 4)):
        rounds = []
        while len(rounds) * stride < len(pairs):
            current = assignments_for_phase.copy()
            rng.shuffle(current)
            rounds.extend(current)
        records = []
        for index, pair in enumerate(pairs, 1):
            records.append(addition(*pair))
            if index % stride == 0:
                records.append(fact(*rounds[index // stride - 1]))
        phases.append(records)
    training = {orbit(*pair) for phase in additions for pair in phase}
    held = {orbit(*pair) for _, _, pair in primary + extrapolation}
    held.update(orbit(*pair) for pairset in counterfactual for pair in pairset)
    assert len(training) == 77824 and len(held) == 1024 and training.isdisjoint(held)
    return {'phases': phases, 'assignments': assignments, 'primary': primary,
            'extrapolation': extrapolation, 'counterfactual': counterfactual,
            'taught': first[:128], 'witness_units': len(witness)}


def small_world(seed):
    """Short cycle fixture, with all three phases and withheld disjoint arithmetic."""
    rng = random.Random(seed)
    used = set()
    phases = [[addition(*unique_pair(rng, 2, used)) for _ in range(n)] for n in (16, 4, 4)]
    phases[0] += [fact('apple', 123), fact('basil', 456)]
    phases[1] += [fact('cedar', 781)]
    phases[2] += [fact('apple', 904)]
    held = [addition(*unique_pair(rng, 2, used)) for _ in range(8)]
    return phases, held
