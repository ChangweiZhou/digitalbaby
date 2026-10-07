import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from centered_core import CenteredCore
from centered_core.checkpoint import sealed, array_info
from tests.check_cycle1 import record
from tests.reference.environment import make_world, DT


def continuation(core, e, start, prediction=False, outcome=False, newline=False):
    at = e['at']
    if not (prediction or outcome or newline):
        for i, b in list(enumerate(bytes.fromhex(e['cue_hex'])))[start:]: core.feed(b, at + i * DT)
        core.predict(at + 12 * DT)
    if not (outcome or newline): core.observe_outcome(e['outcome'], at + 12 * DT)
    if not newline: core.feed(10, at + 13 * DT)
    core.flush(at + 165.)
    core.rest(86400.)
    return core.state_digest()


def main():
    began = time.monotonic()
    original = CenteredCore()
    events = make_world(390102)['events']
    for e in events[:8]: record(original, e)
    restored_points, tamper_cases = [], []
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td); path = folder / 'model.npz'
        for label, count, prediction, outcome, newline in [
            ('mid_cue', 9, False, False, False), ('pending_prediction', 12, True, False, False),
            ('post_outcome', 12, True, True, False), ('newline', 12, True, True, True)]:
            c = original.clone(); e = events[8]; at = e['at']
            for i, b in list(enumerate(bytes.fromhex(e['cue_hex'])))[:count]: c.feed(b, at + i * DT)
            if prediction: c.predict(at + 12 * DT)
            if outcome: c.observe_outcome(e['outcome'], at + 12 * DT)
            if newline: c.feed(10, at + 13 * DT)
            c.save(path); r = CenteredCore.load(path)
            assert r.state_digest() == c.state_digest(), label
            assert continuation(r, e, count, prediction, outcome, newline) == continuation(c, e, count, prediction, outcome, newline)
            restored_points.append(label)
        original.rest(86400.); original.save(path)
        assert CenteredCore.load(path).state_digest() == original.state_digest()
        restored_points.append('rested_boundary')
        command = [sys.executable, '-c', 'from centered_core import CenteredCore; import sys; print(CenteredCore.load(sys.argv[1]).state_digest())', str(path)]
        child = subprocess.run(command, check=True, capture_output=True, text=True)
        assert child.stdout.strip() == original.state_digest()
        before = path.read_bytes()
        with patch('centered_core.checkpoint.np.savez_compressed', side_effect=OSError('simulated interrupted save')):
            try: original.save(path)
            except OSError: pass
            else: raise AssertionError('save failure not propagated')
        assert path.read_bytes() == before and not list(folder.glob('*.pending-*'))
        with np.load(path, allow_pickle=False) as z: baseline = {k: z[k].copy() for k in z.files}
        for case in ['array_bytes', 'missing_array', 'source', 'shape', 'nonfinite', 'cursor', 'visible_suffix', 'dtype', 'negative_clock']:
            d = {k: v.copy() for k, v in baseline.items()}
            m = json.loads(d['metadata_json'].tobytes())
            if case == 'array_bytes': d['s0_fly_fast'].flat[0] += 1.
            elif case == 'missing_array': del d['s4_fe_p']
            elif case == 'source': m['source_identity'] = '0' * 64
            elif case == 'shape':
                d['s0_fly_fast'] = d['s0_fly_fast'][:1].copy(); m['arrays']['s0_fly_fast'] = array_info(d['s0_fly_fast'])
            elif case == 'nonfinite': d['s0_fly_fast'].flat[0] = float('nan')
            elif case == 'cursor': m['cue_count'] = 13
            elif case == 'visible_suffix': m['states'][4]['visible_hex'] = '20'
            elif case == 'dtype':
                d['s0_fly_fast'] = d['s0_fly_fast'].astype(np.float32); m['arrays']['s0_fly_fast'] = array_info(d['s0_fly_fast'])
            elif case == 'negative_clock': m['states'][0]['brain_t'] = -1.
            m.pop('seal'); m['seal'] = sealed(m)
            d['metadata_json'] = np.frombuffer(json.dumps(m).encode(), dtype=np.uint8)
            bad = folder / (case + '.npz'); np.savez_compressed(bad, **d)
            try: CenteredCore.load(bad)
            except ValueError: tamper_cases.append(case)
            else: raise AssertionError('tampered checkpoint accepted: ' + case)
    result = {'cycle': 2, 'verdict': 'PASS', 'evidence': 'E0 interruption recovery', 'world': 390102,
              'exact_resumed_points': restored_points, 'fresh_process_restore': True,
              'atomic_save_failure_preserved_previous_checkpoint': True,
              'tamper_cases_rejected': tamper_cases, 'elapsed_s': time.monotonic() - began}
    Path('results/CYCLE2.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)

if __name__ == '__main__': main()
