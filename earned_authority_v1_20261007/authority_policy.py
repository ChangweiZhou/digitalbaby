"""Stateless, pre-feedback authority organ. No tasks, identities or labels."""
from dataclasses import dataclass
import math
import numpy as np

RULE_VERSION = 'Q_EARNED_MARGIN_WINDOW_V1'
ALPHABET = b'0123'
SCALES = (1.4911274663291492, 1.3452365735750882)
POLICIES = ('ERROR', 'Q_HALF', 'Q_EARNED')


@dataclass(frozen=True)
class Observation:
    shared: tuple
    private: tuple

    def __post_init__(self):
        for values in (self.shared, self.private):
            if len(values) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                       or not math.isfinite(x) for x in values):
                raise ValueError('four finite bank scores required')


def vote(values):
    values = tuple(float(x) for x in values)
    if len(values) < 2 or not all(math.isfinite(x) for x in values):
        raise ValueError('finite decision evidence required')
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    return order[0], values[order[0]] - values[order[1]]


def authority(shared, private):
    if len(shared) != len(private):
        raise ValueError('decision roster mismatch')
    sw, sm = vote(shared)
    pw, pm = vote(private)
    # Only a close conflict is reversible by the already fixed half-weight.
    # Bounds are derived from alpha=1 and alpha=.5, not fitted to outcomes.
    trigger = sw != pw and 0.5 * pm < sm < pm
    return {'alpha': 0.5 if trigger else 1.0, 'trigger': trigger,
            'shared_winner': sw, 'private_winner': pw,
            'shared_margin': sm, 'private_margin': pm,
            'rule_version': RULE_VERSION}


def emit(observations, policy):
    """One common alpha per decision; binary evidence uses both visible options.

    Arithmetic deliberately matches the frozen source: combine the four bank
    channels first, then take channel 1 minus 0 for each binary option. This
    retains its exact baseline tie decisions as well as its information budget.
    """
    if policy not in POLICIES or len(observations) not in (1, 2):
        raise ValueError('unknown policy or sensory protocol')
    if any(type(o) is not Observation for o in observations):
        raise TypeError('only typed pre-answer bank observations are accepted')
    s = [np.asarray(o.shared, dtype=float) / SCALES[0] for o in observations]
    p = [np.asarray(o.private, dtype=float) / SCALES[1] for o in observations]
    paired = len(observations) == 2
    ds = tuple(float(x[1] - x[0]) for x in s) if paired else tuple(map(float, s[0]))
    dp = tuple(float(x[1] - x[0]) for x in p) if paired else tuple(map(float, p[0]))
    evidence = authority(ds, dp)
    alpha = evidence['alpha'] if policy == 'Q_EARNED' else (0.5 if policy == 'Q_HALF' else 1.0)
    vectors = [a + alpha * b for a, b in zip(s, p)]
    combined = tuple(float(x[1] - x[0]) for x in vectors) if paired else tuple(map(float, vectors[0]))
    winner, _ = vote(combined)
    byte = b'LR'[winner] if paired else ALPHABET[winner]
    return {**evidence, 'alpha': alpha, 'policy': policy, 'emitted': byte,
            'shared': ds, 'private': dp, 'combined': combined,
            'observations': [{'shared': o.shared, 'private': o.private} for o in observations]}
