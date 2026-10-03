"""Deliberate developer sealing after three accepted cycles; never launches science."""
import hashlib
import json
import platform
import sys
from compact_bridge import ROOT
from compact_integrity import identity, source_map
from compact_storage import atomic_json


def main():
    current = identity(); bound = {}
    for cycle in (1, 2, 3):
        path = ROOT / 'results' / f'CYCLE{cycle}.json'
        doc = json.loads(path.read_text())
        if doc['verdict'] != 'PASS' or doc['identity_at_trial'] != current:
            raise ValueError('all three final cycles must pass under identical final sources')
        bound[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    import numpy, scipy, numba
    if (platform.python_version(), numpy.__version__, scipy.__version__, numba.__version__) != ('3.11.5', '2.2.6', '1.14.1', '0.61.2'):
        raise ValueError('unqualified runtime')
    atomic_json(ROOT / 'SOURCE_LOCK.json', {'schema': 'COMPACT_SCIENCE_SOURCE_V1', 'identity': current, 'files': source_map()})
    atomic_json(ROOT / 'results/QUALIFICATION.json', {'verdict': 'PASS', 'identity': current, 'cycles_completed': 3,
                'cycle_receipts': bound, 'evidence': 'E0 qualified physical deletion and execution',
                'python': platform.python_version(), 'numpy': numpy.__version__, 'scipy': scipy.__version__,
                'numba': numba.__version__, 'platform': platform.platform(),
                'final_source_frozen': True, 'science_not_yet_launched': True})
    print(json.dumps({'verdict': 'PASS', 'identity': current, 'files': len(source_map())}))


if __name__ == '__main__': main()
