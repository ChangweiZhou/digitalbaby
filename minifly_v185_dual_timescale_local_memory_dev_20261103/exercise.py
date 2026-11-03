"""V185 technical qualification. The scientific campaign has no entry point here."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'minifly_v184_h_residual_deployment_dev_20261102'
sys.path.insert(0, str(PARENT))
from bootstrap import sensor, np, P
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

SEED = hashlib.sha256(b'V185|DEV|20261103').hexdigest()
ARMS = ('FULL', 'UNIFORM', 'SELECTIVE')

def digest(value):
    return hashlib.sha256(value).hexdigest()

def save(name, value):
    (HERE / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')

def draw(salt, index, attempt=0):
    return hashlib.sha256(f'{SEED}|{salt}|{index}|{attempt}'.encode()).digest()

def roster():
    rows, keys, answers = [], set(), set()
    for i in range(80):
        key = bytes(33 + b % 93 for b in draw('key', i)[:6 + draw('length', i)[0] % 5]) + b'~'
        assert key not in keys
        keys.add(key)
        attempt = 0
        while True:
            value = bytes(33 + b % 93 for b in draw('value', i, attempt)[:2])
            if value not in answers:
                break
            attempt += 1
        answers.add(value)
        rows.append((key, value))
    edits = {}
    for i in sorted(range(64), key=lambda i: draw('edit-id', i))[:8]:
        attempt = 0
        while True:
            value = bytes(33 + b % 93 for b in draw('edit-value', i, attempt)[:2])
            if value not in answers:
                break
            attempt += 1
        answers.add(value)
        edits[i] = value
    return rows, edits

class Projection:
    def __init__(self):
        self.sensor = sensor.Sensor()
        self.indices = P.Learner().indices
        self.signs = np.array([[1 if hashlib.sha256(f'V185|projection-sign|{j}|{r}'.encode()).digest()[-1] & 1 else -1
                               for r in range(256)] for j in range(64)], dtype=np.int8)

    def feature(self, prefix):
        code = self.sensor.codes(prefix)[-1]
        return np.tanh((code[self.indices] * self.signs).sum(axis=1) / np.float32(3))

class Learner:
    def __init__(self, arm):
        assert arm in ARMS
        self.arm = arm
        self.fast = np.zeros((257, 64), np.float32)
        self.slow = np.zeros_like(self.fast)
        self.bias = np.zeros(257, np.float32)
        self.clock = 0

    def logits(self, feature):
        return (self.fast + self.slow) @ feature + self.bias

    def teach(self, feature, target):
        logits = self.logits(feature)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        error = -p
        error[target] += 1
        direction = np.outer(error, feature) / (1 + feature @ feature)
        release = (self.fast * self.slow < 0) & (abs(self.fast) > 2 * abs(self.slow))
        gate = np.where((self.slow * direction >= 0) | release, np.float32(1), np.float32(.1))
        dose = np.abs(direction).astype(np.float64)
        total = dose.sum()
        uniform = (dose * gate).sum() / total if total else 1.
        multiplier = {'FULL': 1., 'UNIFORM': uniform, 'SELECTIVE': gate}[self.arm]
        self.fast *= 1 - 1/1024
        self.fast += .25 * direction
        self.slow += .25 * multiplier * direction
        self.bias += .03 * error
        self.clock += 1
        return direction, gate, uniform

    def state(self):
        return self.clock, self.fast.copy(), self.slow.copy(), self.bias.copy()

def decode(score, projection, key, cap=3):
    output = bytearray()
    for _ in range(cap):
        byte = int(np.argmax(score(projection.feature(key + output))))
        if byte == 256:
            return bytes(output), True
        output.append(byte)
    return bytes(output), False

def ceiling(projection, rows):
    features = np.array([projection.feature(key + value[:pos]) for key, value in rows for pos in range(3)])
    labels = np.array([target for _, value in rows for target in (*value, 256)])
    classes = np.unique(labels)
    remapped = np.searchsorted(classes, labels)
    augmented = csr_matrix(np.column_stack((features, np.ones(len(labels)))))
    count = len(classes)
    competitors = np.tile(np.arange(count + 1), (len(labels), 1))
    keep = competitors != remapped[:, None]
    site, col = np.where(keep)
    index = np.arange(len(site))
    contrast = csr_matrix(
        (np.r_[np.ones(len(site)), -np.ones(len(site))],
         (np.r_[index, index], np.r_[col, remapped[site]])),
        shape=(len(site), count + 1))
    selected = augmented[site]
    # Each inequality compares the correct logit to one alternative with unit margin.
    matrices = [selected.multiply(contrast[:, c].toarray()) for c in range(count)]
    from scipy.sparse import hstack
    constraints = hstack(matrices, format='csr')
    result = linprog(np.zeros(count * 65), A_ub=constraints, b_ub=-np.ones(len(site)),
                     bounds=(None, None), method='highs-ipm', options={'time_limit': 45})
    info = {'solver_status': int(result.status), 'message': result.message,
            'supervised_sites': len(labels), 'unique_feature_rows': len(np.unique(features, axis=0)),
            'complete': None}
    if result.success:
        weights = result.x.reshape(count, 65).astype(np.float32)
        def score(feature):
            logits = np.zeros(257, np.float32)
            logits[classes] = weights[:, :64] @ feature + weights[:, 64]
            return logits
        info['complete'] = sum(decode(score, projection, key) == (value, True) for key, value in rows)
    return info

def main():
    start = time.monotonic()
    projection = Projection()
    rows, edits = roster()
    assert projection.sensor.digest == '7e0b78ae71cd9f508e9069018ec2343a88ab8bfd9263067506ee6eb3fae4962b'
    manifest = {
        'seed': SEED, 'alphabet': 'ASCII 33–125; tilde terminator',
        'key_hashes': [digest(k) for k, _ in rows],
        'mapping_sha256': digest(b'\n'.join(k + v for k, v in rows)),
        'edits': sorted(edits),
        'edit_mapping_sha256': digest(b''.join(edits[i] for i in sorted(edits))),
        'orders': {phase: [sorted(ids, key=lambda i: draw(f'{phase}|pass|{p}', i)) for p in range(passes)]
                   for phase, ids, passes in [('A', range(64), 16), ('B', range(64, 80), 8),
                                             ('C', sorted(edits), 4)]},
        'sensor_digest': projection.sensor.digest,
        'projection_sha256': digest(projection.indices.tobytes() + projection.signs.tobytes()),
        'source_sha256': digest(Path(__file__).read_bytes()),
    }
    save('MANIFEST', manifest)
    model = Learner('SELECTIVE')
    feature = projection.feature(rows[0][0])
    assert np.count_nonzero(feature) and all(np.array_equal(Learner(arm).logits(feature), model.logits(feature)) for arm in ARMS)
    model.slow[0, :2] = [1, -1]
    model.fast[0, :2] = [-3, 3]
    original = model.state()
    direction, gate, uniform = model.teach(feature, 256)
    assert np.isclose(np.abs(direction * gate).sum(), np.abs(direction).sum() * uniform, rtol=1e-6)
    restored = Learner('SELECTIVE')
    restored.clock, restored.fast, restored.slow, restored.bias = original
    restored.teach(feature, 256)
    assert all(np.array_equal(a, b) for a, b in zip(model.state(), restored.state()))
    assert decode(lambda _: np.eye(1, 257, 256)[0], projection, rows[0][0]) == (b'', True)
    assert decode(lambda _: np.zeros(257), projection, rows[0][0]) == (b'\0'*3, False)
    save('CYCLE1', {'status': 'PASS', 'sensor_parity': True, 'learned_scalars': 33153,
                    'fixture_teaching_calls': 2, 'birth_parity': True,
                    'same_state_L1': True, 'fork_continuation': True, 'EOS_and_cap': True})
    result = ceiling(projection, rows)
    save('CYCLE2', result)
    status = ('C0_CEILING_PASSED_ADMISSION_PENDING' if result['complete'] == 80 else
              'REPRESENTATION_CEILING_FAILED' if result['solver_status'] in (0, 2) else
              'REPRESENTATION_CEILING_UNRESOLVED')
    save('CYCLE3', {'status': status, 'scientific_optimizer_records': 0,
                    'oracle_record_exposures': 80, 'oracle_target_exposures': 240,
                    'fixture_target_exposures': 2, 'wall_seconds': time.monotonic()-start,
                    'cpu_seconds': time.process_time(), 'peak_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    print(status, result, flush=True)

if __name__ == '__main__':
    main()
