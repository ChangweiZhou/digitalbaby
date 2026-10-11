"""Unresolved boundary fixtures only; no benchmark world or native job is run."""
import json
import copy
import numpy as np
from pathlib import Path
import config
from assets import canonical
from cores import Core, features
from reference import Reference


def main():
    ids = features(65, 66)
    cases = []
    for arm in ('STABLE-ID', 'COMPACT-ID', 'REPLAY-P', 'REPLAY-U'):
        for positive in (True, False):
            model, ref = Core(arm), Reference(arm)
            active = canonical()['groups'][list(ids)].ravel() if arm == 'STABLE-ID' else np.array(ids)
            model.f[active] = 2 if positive else -2
            model.s[active] = -2 if positive else 2
            if not positive:
                constant = canonical()['groups'][30] if arm == 'STABLE-ID' else [30]
                model.f[constant] = 2
                model.s[constant] = 2
            model.visits[active] = 65535
            for name in ('f', 's', 'visits'):
                ref.a[name] = np.copy(getattr(model, name))
            y = 8 if positive else 0
            v = model.value(ids)
            actual = model.update(ids, y, True, v)
            expected = ref.step(ids, y, True, ref.predict(ids))
            if actual != expected:
                raise AssertionError('boundary transition')
            ref.require_seal(model.digest())
            if np.max(np.abs(model.f)) > 2 or np.max(np.abs(model.s)) > 2 or np.max(model.visits) != 65535:
                raise AssertionError('independent bounds/visit cap')
            cases.append(arm+('_upper' if positive else '_lower'))
    for arm in ('TOPO-G', 'TOPO-R'):
        model, ref = Core(arm), Reference(arm)
        # Unit0 owns four uniquely covered identities; deleting any would erase
        # coverage, so its next scheduled mutation must be an audited no-op.
        model.edges[0] = [0, 1, 2, 3]
        for j in range(1, 8):
            model.edges[j] = [4+(4*(j-1)+k)%12 for k in range(4)]
        for j in range(8):
            unused = [i for i in range(16) if i not in model.edges[j]]
            model.frontier[j] = unused[:4]
            model.cursor[j] = (unused[3]+1)%16
        model.updates = 7
        for key in ('edges', 'frontier', 'cursor'):
            ref.a[key] = np.copy(getattr(model, key))
        ref.updates = 7
        before = np.copy(model.edges)
        actual = model.update(ids, 8, True, model.value(ids))
        expected = ref.step(ids, 8, True, ref.predict(ids))
        ref.require_seal(model.digest())
        ref.invariants()
        if actual != expected or actual['mutation'] is not None or model.noops != 1 or not np.array_equal(before, model.edges):
            raise AssertionError('legal empty-replacement policy')
        cases.append(arm+'_empty_legal_set')
    path = config.ROOT/'POST_REVIEW_AUDIT.json'
    path.write_text(json.dumps({'verdict': 'PASS', 'cases': cases,
                                'native_benchmark_jobs': 0, 'source': config.lock()},
                               indent=2, sort_keys=True)+'\n')
    print(json.dumps({'verdict': 'PASS', 'boundary_cases': len(cases),
                      'native_benchmark_jobs': 0}))


if __name__ == '__main__':
    main()
