import dataclasses
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from centered_core import CenteredCore
from centered_core.bootstrap import ROOT
from centered_core.core import SCALES
from centered_core.__main__ import dispatch
from tests.check_cycle1 import record, reference_record, rejects
from tests.reference.learner import Learner
from tests.reference.environment import make_world, DT


def probe_parity(c, r, world, at):
    count = 0
    before = c.state_digest()
    for cohort in world['sets'].values():
        for cue in cohort['cues']:
            a, b = c.clone(), r.clone()
            for i, byte in enumerate(bytes.fromhex(cue)):
                a.feed(byte, at + i * DT); b.feed(byte, at + i * DT)
            pa, pb = a.predict(at + 12 * DT), b.predict(at + 12 * DT)
            assert pa.emitted == pb['emitted']
            for key in ('shared', 'private', 'combined'): assert np.array_equal(getattr(pa, key), pb[key])
            assert a.bank_digests() == b.digests()
            count += 1
    assert before == c.state_digest()
    return count


def clean_cli_trial():
    with tempfile.TemporaryDirectory(prefix='r-center-clean-') as td:
        folder = Path(td)
        for name in ('centered_core', 'vendor', 'tests', 'scripts'):
            shutil.copytree(ROOT / name, folder / name, ignore=shutil.ignore_patterns('scratch', '__pycache__'))
        for name in ('requirements.txt', 'SOURCE_LOCK.json'): shutil.copy2(ROOT / name, folder / name)
        checkpoint = folder / 'model.npz'
        cue = b'        1+2='
        first = [{'op': 'feed', 'byte': b, 't': i * DT} for i, b in enumerate(cue)]
        first += [{'op': 'predict', 't': 12 * DT}]
        second = [{'op': 'observe', 'byte': 51, 't': 12 * DT}, {'op': 'feed', 'byte': 10, 't': 13 * DT},
                  {'op': 'flush', 't': 165.}, {'op': 'rest', 'seconds': 86400.}]
        core = CenteredCore()
        expected = [dispatch(core, e) for e in first + second]
        def run(events, load=False):
            args = [sys.executable, '-m', 'centered_core', '--save', str(checkpoint)]
            if load: args += ['--load', str(checkpoint)]
            child = subprocess.run(args, cwd=folder, input=''.join(json.dumps(e) + '\n' for e in events),
                                   text=True, capture_output=True, check=True)
            return [json.loads(line) for line in child.stdout.splitlines()]
        actual = run(first[:9]) + run(first[9:], True) + run(second, True)
        assert actual == json.loads(json.dumps(expected))
        assert CenteredCore.load(checkpoint).state_digest() == core.state_digest()
        code = '''from centered_core import CenteredCore
from pathlib import Path
import bytecore
c=CenteredCore()
paths=bytecore.loaded_project_files()
root=Path.cwd().resolve()
assert paths and all(Path(p).resolve().is_relative_to(root) for p in paths)
print(len(paths))
'''
        origins = subprocess.run([sys.executable, '-c', code], cwd=folder, capture_output=True, text=True, check=True)
        bad = subprocess.run([sys.executable, '-m', 'centered_core'], cwd=folder,
                             input='{"op":"feed","byte":32,"t":0,"target":51}\n',
                             capture_output=True, text=True)
        assert bad.returncode == 1 and 'unknown operation or field' in bad.stderr
        target = folder / 'centered_core/core.py'
        target.write_text(target.read_text() + '\n# source tamper\n')
        broken = subprocess.run([sys.executable, '-m', 'centered_core'], cwd=folder,
                                input='', capture_output=True, text=True)
        assert broken.returncode != 0 and 'engineering source lock mismatch' in broken.stderr
        target.write_bytes((ROOT / 'centered_core/core.py').read_bytes())
        (folder / 'SOURCE_LOCK.json').unlink()
        missing = subprocess.run([sys.executable, '-m', 'centered_core'], cwd=folder,
                                 input='', capture_output=True, text=True)
        assert missing.returncode != 0 and 'SOURCE_LOCK.json missing' in missing.stderr
        return {'saved_partial_cue': True, 'saved_pending_prediction': True, 'fresh_process_continuation_equal': True,
                'project_assets_inside_clean_copy': int(origins.stdout.strip()), 'hidden_target_field_rejected': True}


def main():
    began = time.monotonic()
    world = make_world(390103)
    parent, ref = CenteredCore(), Learner('R_center', SCALES)
    branch_results = {}
    for branch in ('W', 'N_old', 'N_new'):
        c, r = parent.clone(), ref.clone()
        disabled, probes = 0, 0
        for i, e in enumerate(world['events']):
            if i == 192:
                c.flush(world['new_start']); r.flush(world['new_start'])
                assert c.bank_digests() == r.digests()
            learn = not ((branch == 'N_old' and e['stage'] == 'old') or (branch == 'N_new' and e['stage'] == 'new'))
            pa, receipt = record(c, e, learn=learn); pb = reference_record(r, e, learn=learn)
            assert pa.emitted == pb['emitted'], (branch, i)
            for key in ('shared', 'private', 'combined'): assert np.array_equal(getattr(pa, key), pb[key]), (branch, i, key)
            assert c.bank_digests() == r.digests(), (branch, i, 'state')
            if not learn:
                assert receipt['shared_l1'] + receipt['private_l1'] == [0.] * 8
                disabled += 1
            if (i + 1) % 64 == 0:
                print(json.dumps({'progress': branch, 'completed_records': i + 1}), flush=True)
        c.flush(world['final']); r.flush(world['final'])
        assert c.bank_digests() == r.digests()
        probes += probe_parity(c, r, world, world['final'])
        branch_results[branch] = {'records': 384, 'disabled_records': disabled,
                                  'disabled_store_writes': disabled * 8, 'final_probe_predictions': probes,
                                  'bank_state_parity': True, 'bank_digests': c.bank_digests()}
    assert parent.records == 0 and ref.audit == [], 'branch writes leaked into parent'
    result = {'cycle': 3, 'verdict': 'PASS', 'evidence': 'E0 engineering deployment/parity',
              'world': 390103, 'branches': branch_results, 'records_compared': 1152,
              'final_probe_predictions': 96, 'clean_deployment': clean_cli_trial(),
              'elapsed_s': time.monotonic() - began}
    Path('results/CYCLE3.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)

if __name__ == '__main__': main()
