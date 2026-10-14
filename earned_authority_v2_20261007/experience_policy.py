"""Bounded past-feedback authority hypothesis; no native learner is changed.

The support cache stores four-byte addresses only, never answers or margins.
Help/hurt counters summarize pre-feedback errors and are not confidence bounds.
The sensory protocol and its inherited tie rule remain explicit.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np

SCALES = (1.4911274663291492, 1.3452365735750882)
CAPACITY = 64
MAX_COUNTER = (1 << 64) - 1
RULE = 'SUPPORT_CONDITIONED_PREQUENTIAL_HELP_HURT_V2'
POLICIES = ('ERROR', 'Q_HALF', 'EXPERIENCE', 'GLOBAL_CALIBRATION', 'SUPPORT_ONLY')


@dataclass(frozen=True)
class Observation:
    shared: tuple
    private: tuple

    def __post_init__(self):
        for scores in (self.shared, self.private):
            if len(scores) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                       or not math.isfinite(x) for x in scores):
                raise ValueError('four finite scores required per bank')


def sensory_key(cue):
    """Same visible-byte definition as CONTENT; no parser or semantic ID."""
    if type(cue) is not bytes:
        raise TypeError('sensory bytes required')
    visible = b''
    for b in cue:
        if b == 10:
            visible = b''
        elif b != 32:
            visible = (visible + bytes((b,)))[-4:]
    if len(visible) != 4:
        raise ValueError('short CONTENT context is outside this design')
    return visible


def _keys(keys):
    keys = tuple(keys)
    if len(keys) not in (1, 2) or any(type(k) is not bytes or len(k) != 4 for k in keys):
        raise ValueError('one or two four-byte sensory addresses required')
    return keys


def _vote(values):
    return max(range(len(values)), key=lambda i: (values[i], -i))


def scores_emit(observations, alpha):
    """Match frozen arithmetic: combine four channels, then binary subtraction."""
    observations = tuple(observations)
    if len(observations) not in (1, 2) or any(type(o) is not Observation for o in observations):
        raise TypeError('one or two typed pre-feedback observations required')
    if alpha not in (0.5, 1.0):
        raise ValueError('only the two locked weights are allowed')
    with np.errstate(over='ignore', invalid='ignore'):
        vectors = [np.asarray(o.shared, dtype=float) / SCALES[0]
                   + alpha * np.asarray(o.private, dtype=float) / SCALES[1] for o in observations]
        paired = len(vectors) == 2
        combined = tuple(float(v[1] - v[0]) for v in vectors) if paired else tuple(map(float, vectors[0]))
    if not all(math.isfinite(x) for x in combined):
        raise ValueError('combined decision evidence overflow')
    return {'alpha': alpha, 'emitted': (b'LR' if paired else b'0123')[_vote(combined)],
            'combined': combined}


class ExperienceAuthority:
    """Fixed 64-key FIFO and 6 uint64 counters, plus one pending forecast.

    No direct target is accepted at query. A real byte after predict is required
    to change learning state. Disabling a write disables every observer update.
    """
    def __init__(self):
        self.cache = []
        self.stats = [[0, 0, 0], [0, 0, 0]]  # count, help, hurt
        self.pending = None

    def stratum(self, keys):
        keys = _keys(keys)
        return int(all(k in self.cache for k in keys))

    def _alpha(self, stratum, policy):
        if policy == 'ERROR':
            return 1.0
        if policy == 'Q_HALF':
            return 0.5
        if policy == 'SUPPORT_ONLY':
            return 1.0 if stratum else 0.5
        if policy == 'GLOBAL_CALIBRATION':
            help_, hurt = sum(s[1] for s in self.stats), sum(s[2] for s in self.stats)
        elif policy == 'EXPERIENCE':
            _, help_, hurt = self.stats[stratum]
        else:
            raise ValueError('unknown locked policy')
        return 0.5 if help_ > hurt else 1.0

    def query(self, keys, observations):
        keys, observations = _keys(keys), tuple(observations)
        if len(keys) != len(observations):
            raise ValueError('key/observation mismatch')
        stratum = self.stratum(keys)
        return {p: {**scores_emit(observations, self._alpha(stratum, p)),
                    'supported': bool(stratum), 'policy': p} for p in POLICIES}

    def predict(self, key, observation):
        if self.pending is not None:
            raise RuntimeError('previous real teacher not consumed')
        result = self.query((key,), (observation,))
        self.pending = {'key': key.hex(), 'stratum': int(result['ERROR']['supported']),
                        'error': result['ERROR']['emitted'], 'half': result['Q_HALF']['emitted']}
        return result

    def observe(self, teacher_byte, *, learn):
        if type(teacher_byte) is not int or teacher_byte not in b'0123' or type(learn) is not bool:
            raise ValueError('actual digit teacher byte and actual boolean write flag required')
        if self.pending is None:
            raise RuntimeError('teacher before prediction or duplicate teacher')
        pending, self.pending = self.pending, None
        if not learn:
            return {'updated': False}
        stats = self.stats[pending['stratum']]
        # Stop rather than silently saturate and alter the statistical rule.
        if any(x == MAX_COUNTER for x in stats):
            raise OverflowError('bounded authority counter exhausted')
        e, h = pending['error'] == teacher_byte, pending['half'] == teacher_byte
        stats[0] += 1
        stats[1] += int(h and not e)
        stats[2] += int(e and not h)
        key = bytes.fromhex(pending['key'])
        evicted = None
        if key not in self.cache:
            if len(self.cache) == CAPACITY:
                evicted = self.cache.pop(0).hex()
            self.cache.append(key)
        return {'updated': True, 'stratum_before_teacher': pending['stratum'],
                'help': int(h and not e), 'hurt': int(e and not h), 'evicted': evicted}

    def snapshot(self):
        return {'schema': RULE, 'cache': [k.hex() for k in self.cache],
                'stats': [list(s) for s in self.stats],
                'pending': None if self.pending is None else dict(self.pending)}

    @classmethod
    def restore(cls, d):
        if not isinstance(d, dict) or set(d) != {'schema', 'cache', 'stats', 'pending'} or d['schema'] != RULE:
            raise ValueError('authority snapshot schema')
        c = cls()
        try:
            c.cache = [bytes.fromhex(k) for k in d['cache']]
        except (TypeError, ValueError):
            raise ValueError('authority keys')
        if len(c.cache) > CAPACITY or len(set(c.cache)) != len(c.cache) or any(len(k) != 4 for k in c.cache):
            raise ValueError('authority keys')
        s = d['stats']
        if not isinstance(s, list) or len(s) != 2 or any(not isinstance(x, list) or len(x) != 3 for x in s):
            raise ValueError('authority statistics')
        if any(type(x) is not int or not 0 <= x <= MAX_COUNTER for row in s for x in row):
            raise ValueError('authority statistics')
        if any(row[1] + row[2] > row[0] for row in s):
            raise ValueError('authority statistics')
        c.stats = [list(row) for row in s]
        p = d['pending']
        if p is not None:
            if not isinstance(p, dict) or set(p) != {'key', 'stratum', 'error', 'half'}:
                raise ValueError('authority pending state')
            try:
                key = bytes.fromhex(p['key'])
            except (TypeError, ValueError):
                raise ValueError('authority pending key')
            if len(key) != 4 or type(p['stratum']) is not int or p['stratum'] not in (0, 1):
                raise ValueError('authority pending state')
            if any(type(p[k]) is not int or p[k] not in b'0123' for k in ('error', 'half')):
                raise ValueError('authority pending byte')
            if p['stratum'] != int(key in c.cache):
                raise ValueError('pending familiarity disagrees with past cache')
            c.pending = dict(p)
        return c

    def clone(self):
        return self.restore(self.snapshot())

    def digest(self):
        return hashlib.sha256(json.dumps(self.snapshot(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def packed_bytes(self):
        # Fixed deployment layout: 64 four-byte keys + length + six uint64s.
        # Pending: valid, key4, stratum, ERROR/half bytes (8); no native state.
        return 64 * 4 + 1 + 6 * 8 + 8
