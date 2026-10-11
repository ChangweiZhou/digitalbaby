"""Independent equations and exact transition seals; imports no production core."""
import copy
import hashlib
import json
import math
import random
import numpy as np
import config
from assets import canonical


def checksum(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        if isinstance(value, np.ndarray):
            digest.update(str((value.shape, value.dtype.str)).encode())
            normalized = np.array(value, copy=True)
            if normalized.dtype.kind == 'f':
                normalized[normalized == 0.] = 0.
            digest.update(normalized.tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode())
    return digest.hexdigest()


def answer(value):
    scores = [(k-value)*(k-value) for k in range(9)]
    return scores.index(min(scores))


class Reference:
    def __init__(self, arm):
        self.arm = arm
        self.a = {}
        self.rng = random.Random(20261011047)
        self.cache = []
        self.updates = self.mutations = self.noops = 0
        if arm in ('STABLE-ID', 'COMPACT-ID', 'REPLAY-P', 'REPLAY-U'):
            length = 5177 if arm == 'STABLE-ID' else 31
            self.a = {'f': np.zeros(length), 's': np.zeros(length),
                      'visits': np.zeros(length, dtype=np.uint32)}
        elif arm in ('COV-R', 'DIAG-R'):
            self.a = {'w': np.zeros(31),
                      'p': np.diag([10.]*31) if arm == 'COV-R' else np.full(31, 10.)}
        else:
            birth_rng = random.Random(20261011043)
            connections = []
            for role in (0, 1):
                labels = list(range(16))
                birth_rng.shuffle(labels)
                connections.extend([labels[4*j:4*j+4] if j < 4 else birth_rng.sample(range(16), 4)
                                    for j in range(8)])
            incoming = np.asarray(connections, dtype=np.int16)
            weights = np.asarray([[birth_rng.uniform(-.5, .5) for k in range(4)]
                                  for j in range(16)])
            candidates, cursors = [], []
            for j in range(16):
                cursor = (j*4) % 16
                pending = []
                while len(pending) != 4:
                    if cursor not in connections[j] and cursor not in pending:
                        pending.append(cursor)
                    cursor = (cursor+1) % 16
                candidates.append(pending)
                cursors.append(cursor)
            self.a = {'edges': incoming, 'theta': weights, 'beta': np.zeros(16),
                      'bias': 0., 'frontier': np.array(candidates, np.int16),
                      'cursor': np.array(cursors, np.int16), 'utility': np.zeros((16, 4))}

    def state(self):
        result = dict(self.a)
        result.update(cache=self.cache, updates=self.updates, mutations=self.mutations,
                      noops=self.noops, rng=self.rng.getstate())
        return result

    def clone(self):
        return copy.deepcopy(self)

    def digest(self):
        return checksum(self.state())

    @staticmethod
    def global_id(unit, local):
        return 30 if local == 15 else local + (15 if unit >= 8 else 0)

    def hidden(self, ids):
        activities, derivatives = [], []
        for unit in range(16):
            terms = [float(self.a['theta'][unit, k])
                     for k, label in enumerate(self.a['edges'][unit])
                     if self.global_id(unit, int(label)) in ids]
            total = .25 + math.fsum(terms)
            activities.append(min(2., max(0., total)))
            derivatives.append(float(0. < total < 2.))
        return np.array(activities), np.array(derivatives)

    def predict(self, ids):
        a = self.a
        if 'f' in a:
            indices = canonical()['groups'][list(ids), 0] if self.arm == 'STABLE-ID' else ids
            return (15 / math.sqrt(45)) * math.fsum(float(a['f'][i]+a['s'][i]) for i in indices)
        if 'w' in a:
            return math.fsum(float(a['w'][i]) for i in ids)
        activities, _ = self.hidden(ids)
        return a['bias'] + math.fsum(float(a['beta'][j]*activities[j]) for j in range(16))

    def local_update(self, ids, y, prediction):
        a = self.a
        selected = canonical()['groups'][list(ids)].reshape(-1) if self.arm == 'STABLE-ID' else np.array(ids)
        residual = y - prediction
        capped = min(1., max(-1., residual))
        gain = capped / math.sqrt(45) / np.sqrt(1. + a['visits'][selected].astype(float))
        a['f'][selected] = np.maximum(-2., np.minimum(2., a['f'][selected] + gain*.2))
        a['s'][selected] = np.maximum(-2., np.minimum(2., a['s'][selected] + gain*.8))
        a['visits'][selected] = np.minimum(65535, a['visits'][selected] + 1)
        self.updates += 1

    def step(self, ids, y, allowed, prediction):
        if not allowed:
            return {'performed': False, 'replay_index': None, 'mutation': None}
        a = self.a
        replay_index = mutation = None
        residual = y - prediction
        if 'f' in a:
            self.local_update(ids, y, prediction)
            if self.arm in ('REPLAY-P', 'REPLAY-U'):
                self.cache.append({'ids': list(ids), 'target': y, 'priority': abs(residual)})
                if len(self.cache) > 32:
                    self.cache.pop(0)
                count = len(self.cache)-1
                if count:
                    if self.arm == 'REPLAY-U':
                        replay_index = self.rng.randrange(count)
                    else:
                        probabilities = [self.cache[j]['priority']+.1 for j in range(count)]
                        threshold = self.rng.random() * math.fsum(probabilities)
                        replay_index = count-1
                        cumulative = 0.
                        for j in range(count):
                            cumulative += probabilities[j]
                            if cumulative > threshold:
                                replay_index = j
                                break
                    row = self.cache[replay_index]
                    current = self.predict(row['ids'])
                    row['priority'] = abs(row['target']-current)
                    self.local_update(row['ids'], row['target'], current)
        elif 'w' in a:
            robust = 1. if abs(residual) <= 1. else 1. / abs(residual)
            if self.arm == 'COV-R':
                columns = [a['p'][:, i] for i in ids]
                q = columns[0] + columns[1] + columns[2]
            else:
                q = np.zeros(31)
                for i in ids:
                    q[i] = a['p'][i]
            denominator = 1. + robust * math.fsum(float(q[i]) for i in ids)
            increment = (robust * residual / denominator) * q
            a['w'] = np.minimum(4*config.KAPPA, np.maximum(-4*config.KAPPA, a['w'] + increment))
            shrink = robust / denominator
            if self.arm == 'COV-R':
                delta = shrink * (q[:, None] * q[None, :])
                decreased = a['p'] - delta
                a['p'] = (decreased + decreased.transpose()) / 2.
            else:
                a['p'] = a['p'] - shrink * np.square(q)
            self.updates += 1
        else:
            activities, derivative = self.hidden(ids)
            previous = np.copy(a['beta'])
            error = min(1., max(-1., residual))
            lr_output = .1 / math.sqrt(1.+self.updates)
            lr_input = .02 / math.sqrt(1.+self.updates)
            a['beta'] = np.minimum(4., np.maximum(-4., a['beta']+lr_output*error*activities))
            a['bias'] = min(8., max(-8., a['bias']+lr_output*error))
            for j in range(16):
                for k in range(4):
                    old = a['theta'][j, k]
                    present = int(self.global_id(j, int(a['edges'][j, k])) in ids)
                    delta = lr_input*error*previous[j]*derivative[j]*present
                    a['theta'][j, k] = min(2., max(-2., old+delta))
                for k in range(4):
                    present = int(self.global_id(j, int(a['frontier'][j, k])) in ids)
                    gradient = abs(error*previous[j]*derivative[j]*present)
                    a['utility'][j, k] = (7*a['utility'][j, k]+gradient)/8
            self.updates += 1
            if self.updates % 8 == 0 and self.arm != 'TOPO-S':
                j = (self.updates//8-1) % 16
                start = 0 if j < 8 else 8
                counts = {i: sum(int(i in a['edges'][unit]) for unit in range(start, start+8))
                          for i in range(16)}
                valid = [(k, g) for k in range(4) if counts[int(a['edges'][j, k])] > 1
                         for g in range(4)]
                if valid:
                    if self.arm == 'TOPO-R':
                        k, g = valid[self.rng.randrange(len(valid))]
                    else:
                        ordering = lambda kg: (abs(a['beta'][j]*a['theta'][j, kg[0]]),
                                               -a['utility'][j, kg[1]],
                                               int(a['edges'][j, kg[0]]),
                                               int(a['frontier'][j, kg[1]]))
                        k, g = sorted(valid, key=ordering)[0]
                    old_feature = int(a['edges'][j, k])
                    new_feature = int(a['frontier'][j, g])
                    a['edges'][j, k] = new_feature
                    a['theta'][j, k] = 0.
                    cursor = int(a['cursor'][j])
                    excluded = set(map(int, a['edges'][j])) | (set(map(int, a['frontier'][j])) - {new_feature})
                    while cursor in excluded:
                        cursor = (cursor+1) % 16
                    a['frontier'][j, g] = cursor
                    a['cursor'][j] = (cursor+1) % 16
                    a['utility'][j, g] = 0.
                    self.mutations += 1
                    mutation = {'unit': j, 'deleted': old_feature, 'grown': new_feature}
                else:
                    self.noops += 1
        return {'performed': True, 'replay_index': replay_index, 'mutation': mutation}

    def require_seal(self, actual):
        if actual != self.digest():
            raise AssertionError('independent full learned-state transition seal mismatch')

    def invariants(self):
        a = self.a
        for value in a.values():
            if isinstance(value, np.ndarray) and value.dtype.kind == 'f':
                if not np.isfinite(value).all():
                    raise AssertionError('nonfinite learned state')
        if 'p' in a:
            if self.arm == 'COV-R':
                np.linalg.cholesky(a['p'])
            elif np.min(a['p']) <= 0:
                raise AssertionError('nonpositive diagonal precision')
        if 'edges' in a:
            for role in range(2):
                counts = np.bincount(a['edges'][8*role:8*role+8].ravel(), minlength=16)
                if np.min(counts) < 1:
                    raise AssertionError('sensory coverage erased')
            for unit in range(16):
                if len(set(map(int, a['edges'][unit]))) != 4:
                    raise AssertionError('fan-in duplicate')
                if len(set(map(int, a['frontier'][unit]))) != 4:
                    raise AssertionError('frontier duplicate')
                if set(map(int, a['frontier'][unit])) & set(map(int, a['edges'][unit])):
                    raise AssertionError('frontier active leakage')
        if len(self.cache) > 32:
            raise AssertionError('unbounded replay')
