"""Causal and integrated native-learning DEV qualification, not old science replay."""
import inspect
import json
from pathlib import Path

import numpy as np

import spec
from core import ObservationCore
from runner import consume, probe
from streams import world_inputs, stream


def rejects(fn):
    try:
        fn()
    except (ValueError, TypeError, AssertionError):
        return
    raise AssertionError('negative control was accepted')


def main():
    core = ObservationCore()
    checks = []
    assert tuple(inspect.signature(core.observe).parameters) == ('byte', 't')
    assert len({s.birth_record['fly_id'] for s in core.stores}) == 4
    assert len({s.birth_record['canonical_B_sha256'] for s in core.stores}) == 1
    checks.append('four independently verified canonical births')
    rejects(lambda: core.observe(48, 1.))
    rejects(lambda: core.predict(float('nan')))
    rejects(lambda: core.observe(48, 1., label=0))
    checks.append('unpredicted observation, invalid clock and hidden label rejected')
    before = core.native_digest()
    pred = core.predict(spec.BYTE_SECONDS)
    assert core.native_digest() == before
    rejects(lambda: core.predict(spec.BYTE_SECONDS))
    rejects(lambda: core.observe(48, spec.BYTE_SECONDS + 1))
    rejects(lambda: core.rest(10))
    assert pred['probabilities'] == [.25] * 4
    core.observe(48, spec.BYTE_SECONDS)
    checks.append('read-only pre-arrival prediction and transaction ordering')
    assert core.history == b'0'
    left, right = core.clone(), core.clone()
    left_pred = left.predict(left.t + spec.BYTE_SECONDS)
    right_pred = right.predict(right.t + spec.BYTE_SECONDS)
    assert left_pred == right_pred
    assert all(np.array_equal(a, b) for a, b in zip(left.cached['codes'], right.cached['codes']))
    left.observe(49, left.t + spec.BYTE_SECONDS)
    right.observe(50, right.t + spec.BYTE_SECONDS)
    assert left.native_digest() != right.native_digest()
    assert core.bytes_seen == 1
    checks.append('same past has identical prediction/address; different actual bytes cause different native writes')
    rejects(lambda: spec.require_dev(811001))
    checks.append('formal science IDs hard rejected')
    inputs = world_inputs(810001)
    raw = stream(inputs['old'], 4, 810001)
    w, n = ObservationCore(), ObservationCore(plastic=False)
    rows_w, rows_n = consume(w, raw), consume(n, raw)
    assert sum(sum(r['native_write_l1']) for r in rows_w) > 0
    assert all(sum(r['native_write_l1']) == 0 for r in rows_n)
    wp, np_ = probe(w, inputs['old'], 810501), probe(n, inputs['old'], 810501)
    ep = probe(w, inputs['old'], 810501, erase=True)
    assert all(abs(v - .25) < 1e-12 for row in ep['rows'] for v in row['probabilities'])
    assert wp['metrics']['all_byte_bpb'] < np_['metrics']['all_byte_bpb']
    assert wp['metrics']['focal_accuracy'] > np_['metrics']['focal_accuracy']
    checks.append('continuous ordinary-byte learning improves native output; no-write and erase controls')
    result = dict(verdict='PASS', checks=checks, W=wp['metrics'], N_ALL=np_['metrics'],
                  NATIVE_ERASE=ep['metrics'], ordinary_bytes_learned=len(raw),
                  native_mutable_bytes=w.mutable_bytes(), claims='DEV E0/E1 only')
    out = Path(__file__).parent / 'cycles' / 'cycle1' / 'INTERFACE_TEST.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError('test receipt already exists')
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
