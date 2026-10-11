"""Nine fixed engineered content cores. No world, truth or phase API."""
import copy
import hashlib
import json
import math
import random
import numpy as np
import config
from assets import canonical


def emitted(value):
    if not math.isfinite(value):
        raise ValueError('nonfinite output')
    return min(range(9), key=lambda k: ((k - value) ** 2, k))


def features(left, right):
    if type(left) is not int or type(right) is not int:
        raise ValueError('operand byte type')
    return (config.OPERANDS.index(left), 15 + config.OPERANDS.index(right), 30)


def state_hash(state):
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        h.update(name.encode())
        if isinstance(value, np.ndarray):
            h.update(str((value.shape, value.dtype.str)).encode())
            a = value.copy()
            if a.dtype.kind == 'f':
                a[a == 0] = 0  # canonical signed zero only
            h.update(a.tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                allow_nan=False).encode())
    return h.hexdigest()


class Core:
    def __init__(self, arm):
        if arm not in config.ARMS:
            raise ValueError('arm roster')
        self.arm = arm
        self.rng = random.Random(20261011047)
        self.updates = 0
        self.cache = []
        self.mutations = 0
        self.noops = 0
        self.scalar_ops = 0
        self.traffic_bytes = 0
        if arm in ('STABLE-ID', 'COMPACT-ID', 'REPLAY-P', 'REPLAY-U'):
            n = 5177 if arm == 'STABLE-ID' else 31
            self.f = np.zeros(n)
            self.s = np.zeros(n)
            self.visits = np.zeros(n, np.uint32)
        elif arm in ('COV-R', 'DIAG-R'):
            self.w = np.zeros(31)
            self.p = np.eye(31) * 10 if arm == 'COV-R' else np.ones(31) * 10
        else:
            birth = random.Random(20261011043)
            self.edges = np.zeros((16, 4), np.int16)
            for role in range(2):
                permutation = list(range(16))
                birth.shuffle(permutation)
                for unit in range(8):
                    chosen = permutation[unit*4:unit*4+4] if unit < 4 else birth.sample(range(16), 4)
                    self.edges[8*role+unit] = chosen
            self.theta = np.array([[birth.uniform(-.5, .5) for _ in range(4)] for _ in range(16)])
            self.beta = np.zeros(16)
            self.bias = 0.
            self.frontier = np.empty((16, 4), np.int16)
            self.cursor = np.empty(16, np.int16)
            self.utility = np.zeros((16, 4))
            for unit in range(16):
                cursor = 4*unit % 16
                chosen = []
                while len(chosen) < 4:
                    if cursor not in self.edges[unit] and cursor not in chosen:
                        chosen.append(cursor)
                    cursor = (cursor+1) % 16
                self.frontier[unit] = chosen
                self.cursor[unit] = cursor
        self.birth = copy.deepcopy(self.state())
        for value in self.birth.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        self.birth['cache'] = ()

    def state(self):
        names = ('f', 's', 'visits', 'w', 'p', 'edges', 'theta', 'beta',
                 'bias', 'frontier', 'cursor', 'utility')
        out = {name: getattr(self, name) for name in names if hasattr(self, name)}
        out.update(cache=self.cache, updates=self.updates, mutations=self.mutations,
                   noops=self.noops, rng=self.rng.getstate())
        return out

    def digest(self):
        return state_hash(self.state())

    def clone(self):
        # Every mutable component, including RNG/cache, belongs to this branch.
        return copy.deepcopy(self)

    def restore(self, state):
        for name, value in state.items():
            if name == 'rng':
                def tuples(x):
                    return tuple(tuples(v) for v in x) if isinstance(x, (list, tuple)) else x
                self.rng.setstate(tuples(value))
            elif name == 'cache':
                self.cache = list(copy.deepcopy(value))
            else:
                setattr(self, name, copy.deepcopy(value))

    def erase(self):
        self.restore(self.birth)

    def payload_bytes(self):
        arrays = sum(v.nbytes for v in self.state().values() if isinstance(v, np.ndarray))
        rest = {k: v for k, v in self.state().items() if not isinstance(v, np.ndarray)}
        return arrays + len(json.dumps(rest, separators=(',', ':')).encode())

    @staticmethod
    def local_feature(unit, feature):
        return 30 if feature == 15 else (0 if unit < 8 else 15) + feature

    def hidden(self, ids):
        active = set(ids)
        u = np.array([.25 + math.fsum(float(self.theta[j, k]) for k, f in enumerate(self.edges[j])
                                      if self.local_feature(j, int(f)) in active)
                      for j in range(16)])
        return np.clip(u, 0, 2), ((u > 0) & (u < 2)).astype(float)

    def value(self, ids):
        if self.arm == 'STABLE-ID':
            # Prospective deterministic group reduction; expanded coordinates
            # retain independent mutable arrays and all 15 equal updates.
            first = canonical()['groups'][list(ids), 0]
            return config.KAPPA * math.fsum(float(self.f[i] + self.s[i]) for i in first)
        if hasattr(self, 'f'):
            return config.KAPPA * math.fsum(float(self.f[i] + self.s[i]) for i in ids)
        if hasattr(self, 'w'):
            return math.fsum(float(self.w[i]) for i in ids)
        h, _ = self.hidden(ids)
        return self.bias + math.fsum(float(a*b) for a, b in zip(self.beta, h))

    def compact_step(self, ids, target, value):
        active = canonical()['groups'][list(ids)].ravel() if self.arm == 'STABLE-ID' else np.array(ids)
        e = max(-1., min(1., target - value))
        step = e / math.sqrt(45) / np.sqrt(1 + self.visits[active].astype(float))
        self.f[active] = np.clip(self.f[active] + .2 * step, -2, 2)
        self.s[active] = np.clip(self.s[active] + .8 * step, -2, 2)
        self.visits[active] = np.minimum(self.visits[active] + 1, 65535)
        self.updates += 1
        self.scalar_ops += 18 * len(active)
        self.traffic_bytes += 72 * len(active)

    def update(self, ids, target, allowed, cached_value):
        if type(allowed) is not bool or type(target) is not int or not 0 <= target <= 8:
            raise ValueError('observed target/permission')
        if not allowed:
            return {'performed': False, 'replay_index': None, 'mutation': None}
        before_ops = self.scalar_ops
        replay_index = mutation = None
        residual = target - cached_value
        if hasattr(self, 'f'):
            self.compact_step(ids, target, cached_value)
            if self.arm.startswith('REPLAY'):
                self.cache.append({'ids': list(ids), 'target': target, 'priority': abs(residual)})
                self.cache = self.cache[-32:]
                previous = self.cache[:-1]
                if previous:
                    if self.arm == 'REPLAY-P':
                        weights = [row['priority'] + .1 for row in previous]
                        draw = self.rng.random() * math.fsum(weights)
                        acc = 0.
                        replay_index = len(previous)-1
                        for i, weight in enumerate(weights):
                            acc += weight
                            if draw < acc:
                                replay_index = i
                                break
                    else:
                        replay_index = self.rng.randrange(len(previous))
                    row = self.cache[replay_index]
                    replay_value = self.value(row['ids'])
                    row['priority'] = abs(row['target'] - replay_value)
                    self.compact_step(row['ids'], row['target'], replay_value)
                    self.scalar_ops += 10 * len(previous) + 100
                    self.traffic_bytes += 64 * len(previous)
        elif hasattr(self, 'w'):
            gamma = 1. if abs(residual) <= 1 else 1./abs(residual)
            if self.arm == 'COV-R':
                q = self.p[:, ids[0]] + self.p[:, ids[1]] + self.p[:, ids[2]]
            else:
                q = np.zeros(31)
                q[list(ids)] = self.p[list(ids)]
            d = 1 + gamma * math.fsum(float(q[i]) for i in ids)
            self.w = np.clip(self.w + (gamma * residual / d) * q,
                             -4*config.KAPPA, 4*config.KAPPA)
            factor = gamma / d
            if self.arm == 'COV-R':
                self.p = self.p - factor * np.outer(q, q)
                self.p = (self.p + self.p.T) / 2
            else:
                self.p = self.p - factor * q**2
            self.updates += 1
            self.scalar_ops += 11000 if self.arm == 'COV-R' else 600
            self.traffic_bytes += 96000 if self.arm == 'COV-R' else 2400
        else:
            h, derivative = self.hidden(ids)
            old_beta = self.beta.copy()
            e = max(-1., min(1., residual))
            eta_out = .1/math.sqrt(1+self.updates)
            eta_in = .02/math.sqrt(1+self.updates)
            self.beta = np.clip(self.beta + eta_out*e*h, -4, 4)
            self.bias = max(-8., min(8., self.bias + eta_out*e))
            active = set(ids)
            for unit in range(16):
                for k, f in enumerate(self.edges[unit]):
                    x = int(self.local_feature(unit, int(f)) in active)
                    self.theta[unit, k] = max(-2., min(2., self.theta[unit, k] +
                                               eta_in*e*old_beta[unit]*derivative[unit]*x))
                for k, f in enumerate(self.frontier[unit]):
                    x = int(self.local_feature(unit, int(f)) in active)
                    g = abs(e*old_beta[unit]*derivative[unit]*x)
                    self.utility[unit, k] = (7*self.utility[unit, k] + g)/8
            self.updates += 1
            if self.updates % 8 == 0 and self.arm != 'TOPO-S':
                unit = (self.updates//8-1) % 16
                role = self.edges[0:8] if unit < 8 else self.edges[8:16]
                counts = np.bincount(role.ravel(), minlength=16)
                legal = [(k, g) for k, f in enumerate(self.edges[unit]) if counts[f] > 1
                         for g in range(4)]
                if legal:
                    if self.arm == 'TOPO-G':
                        k, g = min(legal, key=lambda kg: (
                            abs(self.beta[unit]*self.theta[unit, kg[0]]),
                            -self.utility[unit, kg[1]],
                            int(self.edges[unit, kg[0]]), int(self.frontier[unit, kg[1]])))
                    else:
                        k, g = legal[self.rng.randrange(len(legal))]
                    deleted = int(self.edges[unit, k])
                    grown = int(self.frontier[unit, g])
                    self.edges[unit, k] = grown
                    self.theta[unit, k] = 0.
                    cursor = int(self.cursor[unit])
                    other_frontier = set(map(int, self.frontier[unit])) - {grown}
                    while cursor in self.edges[unit] or cursor in other_frontier:
                        cursor = (cursor+1) % 16
                    self.frontier[unit, g] = cursor
                    self.cursor[unit] = (cursor+1) % 16
                    self.utility[unit, g] = 0.
                    self.mutations += 1
                    mutation = {'unit': unit, 'deleted': deleted, 'grown': grown}
                else:
                    self.noops += 1
            self.scalar_ops += 8000
            self.traffic_bytes += 15000
        if self.scalar_ops - before_ops > 20000:
            raise AssertionError('declared arithmetic bound')
        return {'performed': True, 'replay_index': replay_index, 'mutation': mutation}


class ByteLearner:
    """Only feed(byte, ticks, allowed) receives experimental input."""
    def __init__(self, arm):
        self.core = Core(arm)
        self.line = b''
        self.pending = None
        self.ticks = 0
        self.bytes_seen = 0

    def clone(self):
        return copy.deepcopy(self)

    def rest(self, ticks):
        if type(ticks) is not int or ticks < 0 or self.line or self.pending is not None:
            raise ValueError('rest must occur at complete-record boundary')
        self.ticks += ticks

    def feed(self, byte, ticks, allowed):
        if type(byte) is not int or type(ticks) is not int or ticks <= self.ticks or type(allowed) is not bool:
            raise ValueError('public input type/time')
        prediction = update = None
        if byte == 10:
            if len(self.line) not in (4, 5):
                raise ValueError('incomplete line')
            self.line = b''
            self.pending = None
        elif len(self.line) == 0:
            if byte not in config.OPERANDS:
                raise ValueError('operand')
            self.line += bytes([byte])
        elif len(self.line) == 1:
            if byte != 43:
                raise ValueError('plus')
            self.line += b'+'
        elif len(self.line) == 2:
            if byte not in config.OPERANDS:
                raise ValueError('operand')
            self.line += bytes([byte])
        elif len(self.line) == 3:
            if byte != 61:
                raise ValueError('prompt')
            self.line += b'='
            ids = features(self.line[0], self.line[2])
            value = self.core.value(ids)
            prediction = {'value': value, 'emitted': emitted(value) + 48}
            self.pending = (ids, value)
        elif len(self.line) == 4:
            if byte not in config.ANSWERS or self.pending is None:
                raise ValueError('actual observed answer required')
            ids, value = self.pending
            update = self.core.update(ids, byte-48, allowed, value)
            self.pending = None
            self.line += bytes([byte])
        else:
            raise ValueError('overlong record')
        self.ticks = ticks
        self.bytes_seen += 1
        return {'prediction': prediction, 'update': update}
