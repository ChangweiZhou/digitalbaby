"""Synthetic technical qualification for V186."""
import copy
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'minifly_v181_mini_local_plasticity_dev_20261102'
sys.path[:0] = ['/private/tmp/minifly_v183_runtime', str(PARENT)]
from learner import Learner, T, np
from sensor import Sensor

class DecaySensor(Sensor):
    def codes(self, record, decay=.95):
        p = np.zeros(88)
        output = []
        for i in range(len(record) + 1):
            faded = p * decay
            output.append(self.encode(faded))
            if i < len(record):
                p = faded[self.perm] + self.R[record[i]]
        return np.array(output)

def save(name, data):
    Path(name + '.json').write_text(json.dumps(data, indent=2) + '\n')

def state_bytes(model):
    buffer = io.BytesIO()
    T.save(model.state(), buffer)
    return buffer.getvalue()

def teach(model, sensor, decay, batch, records):
    codes, labels, ids = [], [], []
    for i, record in enumerate(records):
        codes.extend(sensor.codes(record, decay))
        labels.extend([*record, 256])
        ids.extend([str(i)] * (len(record) + 1))
    for offset in range(0, len(labels), batch):
        end = offset + batch
        model.step(T.from_numpy(np.array(codes[offset:end])), T.tensor(labels[offset:end]),
                   ids[offset:end], 'NORMAL')
    return len(labels)

def main():
    started = time.monotonic()
    sensor = DecaySensor()
    parent = Sensor()
    assert sensor.digest == parent.digest
    records = [bytes(range(256)), b'prefix\x00\xff', b'prefix\x00\x01', b'']
    for record in records:
        assert np.array_equal(parent.codes(record), sensor.codes(record))
    for decay in (.95, .5):
        assert np.array_equal(sensor.codes(records[1], decay)[:8], sensor.codes(records[2], decay)[:8])
    changes = int(np.count_nonzero(sensor.codes(records[0], .5) != parent.codes(records[0])))
    assert changes
    save('CYCLE1', {'parity': True, 'digest': sensor.digest, 'changed_KC_entries': changes,
                    'source_sha256': {n:hashlib.sha256((PARENT/n).read_bytes()).hexdigest()
                                      for n in ('learner.py', 'sensor.py')}})
    results = []
    for decay in (.95, .5):
        for batch in (256, 32):
            model = Learner()
            a = teach(model, sensor, decay, batch, records)
            assert model.clock == (a + batch - 1) // batch
            b = teach(model, sensor, decay, batch, [b'intervening'])
            checkpoint = state_bytes(model)
            outputs = []
            total = a + b
            for branch in ([b'original'], [b'corrected'], [b'original']):
                child = Learner()
                child.restore(T.load(io.BytesIO(checkpoint), weights_only=True))
                total += teach(child, sensor, decay, batch, branch)
                outputs.append(child)
            for field in ('p', 'm', 'v'):
                assert all(T.equal(v, getattr(outputs[2], field)[k])
                           for k,v in getattr(outputs[0], field).items())
            assert outputs[0].clock == outputs[2].clock
            assert T.equal(outputs[0].rng.get_state(), outputs[2].rng.get_state())
            assert any(not T.equal(v, outputs[1].p[k]) for k,v in outputs[0].p.items())
            assert state_bytes(model) == checkpoint
            results.append({'decay':decay, 'batch':batch, 'targets':total, 'records':8,
                            'A_updates':(a+batch-1)//batch, 'checkpoint_bytes':len(checkpoint)})
    save('CYCLE2', {'arms':results,'candidate_science_records':0,
                    'fixture_records':sum(r['records'] for r in results),
                    'fixture_targets':sum(r['targets'] for r in results)})
    save('CYCLE3', {'status':'FIXTURE_PASS_FULL_ADMISSION_PENDING',
                    'cpu_seconds':time.process_time(),'wall_seconds':time.monotonic()-started,
                    'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
if __name__ == '__main__':
    main()
