"""Non-pickle whole-record checkpoint and synthetic operations qualification."""
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
import config
from cores import ByteLearner
from runner import atomic_json, deserialize, serialize


def save_checkpoint(model, path):
    if model.line or model.pending is not None:
        raise ValueError('checkpoint requires complete record')
    path = Path(path)
    if path.exists():
        raise ValueError('duplicate checkpoint')
    payload = {'schema': 'DIRECTION_TECH_WHOLE_RECORD_CHECKPOINT', 'source': config.lock(),
               
               'arm': model.core.arm, 'state': serialize(model.core.state()),
               'seal': model.core.digest(), 'ticks': model.ticks, 'bytes_seen': model.bytes_seen,
               'scalar_ops': model.core.scalar_ops, 'traffic_bytes': model.core.traffic_bytes}
    temporary = path.with_name(path.name+'.pending')
    with gzip.open(temporary, 'wt') as f:
        json.dump(payload, f, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name+'.sha256').write_text(digest+'\n')
    return digest


def load_checkpoint(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != path.with_name(path.name+'.sha256').read_text().strip():
        raise ValueError('checkpoint file integrity')
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    if data['schema'] != 'DIRECTION_TECH_WHOLE_RECORD_CHECKPOINT' or data['source'] != config.lock():
        raise ValueError('checkpoint source/schema')
    model = ByteLearner(data['arm'],data['state'].get('world_id',0))
    model.core.restore(deserialize(data['state']))
    model.ticks = data['ticks']
    model.bytes_seen = data['bytes_seen']
    model.core.scalar_ops = data['scalar_ops']
    model.core.traffic_bytes = data['traffic_bytes']
    if model.core.digest() != data['seal']:
        raise ValueError('checkpoint operative state seal')
    return model


def can_admit(elapsed, predicted_seconds):
    if isinstance(elapsed, bool) or isinstance(predicted_seconds, bool):
        raise ValueError('time type')
    if not 0 <= elapsed < float('inf') or not 0 < predicted_seconds < float('inf'):
        raise ValueError('finite budget')
    return elapsed + predicted_seconds <= 8*3600


def package_synthetic(folder, target, accepted=True):
    if not accepted:
        raise ValueError('partial or failed data cannot be packaged as accepted')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for file in sorted(Path(folder).glob('*.json')):
            z.write(file, file.name)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:
            raise ValueError('package integrity')


def suite(folder, checkpoint_folder=None, expected=None):
    folder = Path(folder)
    folder.mkdir(exist_ok=False)
    if not can_admit(0, 100) or can_admit(8*3600, 1):
        raise AssertionError('safe session admission')
    negatives = 0
    for args in ((True, 1), (float('nan'), 1), (1, float('inf')), (1, -1)):
        try:
            can_admit(*args)
        except ValueError:
            negatives += 1
        else:
            raise AssertionError('bad budget admitted')
    atomic_json({'stage': 'PAUSED_SESSION_LIMIT', 'science_jobs': 0}, folder/'STATUS.json')
    try:
        package_synthetic(folder, folder/'partial.zip', accepted=False)
    except ValueError:
        negatives += 1
    else:
        raise AssertionError('partial package accepted')
    package_synthetic(folder, folder/'synthetic.zip')
    claim = folder/'SYNTHETIC_RECEIPT.claim'
    fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    try:
        fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        negatives += 1
    else:
        os.close(fd)
        raise AssertionError('duplicate dispatcher claim accepted')
    # Detached admission/process liveness, with a synthetic payload only.
    completed = folder/'CHILD.json'
    child = subprocess.Popen([sys.executable, str(Path(__file__)), '--synthetic-child', str(completed)],
                             start_new_session=True, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE)
    code = child.wait(timeout=60)
    if code or not completed.exists() or json.loads(completed.read_text())['native_records'] != 0:
        raise AssertionError('detached synthetic child')
    cold = []
    if checkpoint_folder is not None:
        for arm in config.ARMS:
            task = folder/(arm+'.task.json')
            records = [bytes((b'ABCDE'[i%5], 43, b'ABCDE'[(2*i+1)%5], 61,
                              48+i%5, 10)).hex() for i in range(24, 64)]
            destination = folder/(arm+'.result.json')
            atomic_json({'checkpoint': str(Path(checkpoint_folder)/(arm+'.json.gz')),
                         'records': records, 'output': str(destination)}, task)
            process = subprocess.Popen([sys.executable, str(Path(__file__)), '--resume-task', str(task)],
                                       start_new_session=True, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE)
            code = process.wait(timeout=60)
            if code:
                raise AssertionError('cold detached checkpoint worker: '+process.stderr.read().decode())
            result = json.loads(destination.read_text())
            want = expected[arm]
            if result['seal'] != want['final_core_seal'] or result['ticks'] != want['final_ticks'] or result['bytes_seen'] != want['final_bytes_seen']:
                raise AssertionError('cold-process state parity')
            cold.append(arm)
    return {'verdict': 'PASS', 'rejected_bad_inputs': negatives,
            'detached_child_pid': child.pid, 'native_records': 0,
            'safe_pause': True, 'package_test': True, 'cold_restart_parity_arms': cold}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--synthetic-child', type=Path)
    p.add_argument('--resume-task', type=Path)
    a = p.parse_args()
    if a.resume_task is not None:
        request = json.loads(a.resume_task.read_text())
        model = load_checkpoint(request['checkpoint'])
        for raw in request['records']:
            for b in bytes.fromhex(raw):
                model.feed(int(b), model.ticks+1, True)
        atomic_json({'seal': model.core.digest(), 'ticks': model.ticks,
                     'bytes_seen': model.bytes_seen, 'technical_only': True}, request['output'])
    elif a.synthetic_child is not None:
        atomic_json({'native_records': 0, 'pid': os.getpid(), 'synthetic': True}, a.synthetic_child)
    else:
        raise ValueError('synthetic child only')
