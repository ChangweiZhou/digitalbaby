import json
import platform
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from centered_core import CenteredCore
from centered_core.core import SCALES
from tests.reference.learner import Learner
from tests.reference.environment import make_world, DT


def rejects(core, call):
    before = core.state_digest()
    try:
        call()
    except ValueError:
        assert before == core.state_digest(), 'rejected operation mutated state'
    else:
        raise AssertionError('invalid operation accepted')


def record(core, event, learn=True):
    at = event['at']
    for i, b in enumerate(bytes.fromhex(event['cue_hex'])):
        core.feed(b, at + i * DT)
    result = core.predict(at + 12 * DT)
    receipt = core.observe_outcome(event['outcome'], at + 12 * DT, learn=learn)
    core.feed(10, at + 13 * DT)
    assert core.flush(at + 165.) < 1e-8
    return result, receipt


def reference_record(ref, event, learn=True):
    at = event['at']
    for i, b in enumerate(bytes.fromhex(event['cue_hex'])):
        ref.feed(b, at + i * DT)
    result = ref.predict(at + 12 * DT)
    if learn:
        ref.observe_outcome(event['outcome'], at + 12 * DT)
    else:
        # External intervention covers BOTH APIs, all eight stores.
        patches = []
        for m in ref.shared + ref.private:
            for name in ('teach_logged', 'teach_signed'):
                orig = getattr(m, name)
                def blocked(*args, _orig=orig, **kwargs):
                    kwargs['write'] = False
                    return _orig(*args, **kwargs)
                p = patch.object(m, name, blocked)
                p.start(); patches.append(p)
        try:
            ref.observe_outcome(event['outcome'], at + 12 * DT)
        finally:
            for p in reversed(patches): p.stop()
    ref.feed(10, at + 13 * DT)
    assert ref.flush(at + 165.) < 1e-8
    return result


def main():
    started = time.monotonic()
    core = CenteredCore()
    ref = Learner('R_center', SCALES)
    assert core.bank_digests() == ref.digests()
    assert len({id(m.fly) for m in core.models}) == 8
    assert len({id(m.fly.m.B.data) for m in core.models}) == 8
    rejects(core, lambda: core.predict(0.))
    rejects(core, lambda: core.feed(256, 0.))
    rejects(core, lambda: core.feed(True, 0.))
    rejects(core, lambda: core.feed(32, float('nan')))
    rejects(core, lambda: core.observe_outcome(48, 0.))
    events = make_world(390101)['events'][:12]
    for i, e in enumerate(events):
        a, receipt = record(core, e, learn=(i % 3 != 0))
        b = reference_record(ref, e, learn=(i % 3 != 0))
        assert a.emitted == b['emitted']
        for key in ('shared', 'private', 'combined'):
            assert np.array_equal(getattr(a, key), b[key]), key
        assert core.bank_digests() == ref.digests(), f'record {i}'
        if i % 3 == 0:
            assert receipt['shared_l1'] + receipt['private_l1'] == [0.] * 8
    clone = core.clone()
    assert clone.state_digest() == core.state_digest()
    before = core.state_digest()
    clone.rest(86400.)
    assert core.state_digest() == before
    rejects(core, lambda: core.feed(32, 0.))
    # Validate pending-operation guards, including byte/time rejection before mutation.
    e = make_world(390101)['events'][12]
    for i, b in enumerate(bytes.fromhex(e['cue_hex'])): core.feed(b, e['at'] + i * DT)
    rejects(core, lambda: core.feed(32, e['at'] + 12 * DT))
    core.predict(e['at'] + 12 * DT)
    rejects(core, lambda: core.predict(e['at'] + 12 * DT))
    rejects(core, lambda: core.rest(1.))
    rejects(core, lambda: core.observe_outcome(48, e['at'] + 13 * DT))
    rejects(core, lambda: core.observe_outcome(255, e['at'] + 12 * DT))
    core.observe_outcome(e['outcome'], e['at'] + 12 * DT)
    rejects(core, lambda: core.feed(32, e['at'] + 13 * DT))
    result = {'cycle': 1, 'verdict': 'PASS', 'evidence': 'E0 technical parity',
              'world': 390101, 'records_compared': 12, 'predictions_compared': 12,
              'bank_state_comparisons': 96, 'disabled_records': 4,
              'disabled_store_writes_checked': 32, 'eight_independent_births': True,
              'clone_isolation': True, 'invalid_operations_rejected_without_mutation': True,
              'elapsed_s': time.monotonic() - started, 'python': platform.python_version()}
    Path('results/CYCLE1.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)

if __name__ == '__main__': main()
