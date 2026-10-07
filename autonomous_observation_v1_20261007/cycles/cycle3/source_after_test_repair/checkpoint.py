"""Bounded, non-pickle native checkpoints with source, schema and byte integrity."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import bootstrap
from core import ObservationCore

SCHEMA = 'OBSERVATION-NATIVE-CHECKPOINT-1'
SOURCE_NAMES = ('core.py', 'spec.py', 'bootstrap.py', 'checkpoint.py')
PARENT_NAMES = ('vendor/src/stores.py', 'vendor/src/paths.py',
                'vendor/package/portable_birth.py', 'vendor/package/FULL151_CANONICAL_B.npz',
                'vendor/package/FULL151_BIRTH_FINGERPRINT.json', 'SOURCE_LOCK.json')


def source_identity():
    h = hashlib.sha256()
    for base, names in ((bootstrap.ROOT, SOURCE_NAMES), (bootstrap.PARENT, PARENT_NAMES)):
        for name in names:
            h.update(name.encode())
            h.update((base / name).read_bytes())
    parent_lock = json.loads((bootstrap.PARENT / 'SOURCE_LOCK.json').read_text())
    for name, expected in sorted(parent_lock['files'].items()):
        actual = hashlib.sha256((bootstrap.PARENT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('immutable native dependency changed: ' + name)
        h.update(name.encode())
        h.update(actual.encode())
    return h.hexdigest()


def payload(model):
    arrays = {}
    state = dict(schema=SCHEMA, source_identity=source_identity(), history=model.history.hex(),
                 t=model.t, bytes_seen=model.bytes_seen, plastic=model.plastic,
                 stores=[], cached=None, expected_state_digest=model.state_digest())
    for i, s in enumerate(model.stores):
        if s.pending_x is not None or s.pending_t is not None:
            raise ValueError('cannot save an incomplete native update')
        for name in ('fast', 'slow', 'adapt'):
            arrays[f'{i}_{name}'] = np.asarray(getattr(s.fly.m, name)).copy()
        state['stores'].append(dict(brain_t=s.brain_t, elapsed_base=s.elapsed_base, teach_seen=s.teach_seen,
            elapsed=float(s.fly.m.elapsed), events=int(s.fly.m.event_count),
            presentations=int(s.fly.m.presentation_count)))
    if model.cached is not None:
        state['cached'] = dict(t=model.cached['t'], history=model.cached['history'].hex())
        arrays['cached_p'] = model.cached['p'].copy()
        for i, x in enumerate(model.cached['codes']):
            arrays[f'cached_code_{i}'] = x.copy()
    arrays['metadata'] = np.frombuffer(json.dumps(state, sort_keys=True, allow_nan=False).encode(), dtype=np.uint8)
    return arrays


def save(model, path):
    path = Path(path)
    if path.exists() or path.with_suffix(path.suffix + '.sha256').exists():
        raise ValueError('checkpoint already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **payload(model))
        f.flush()
        os.fsync(f.fileno())
    digest = hashlib.sha256(temp.read_bytes()).hexdigest()
    os.replace(temp, path)
    path.with_suffix(path.suffix + '.sha256').write_text(digest + '\n')
    return digest


def load(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != path.with_suffix(path.suffix + '.sha256').read_text().strip():
        raise ValueError('checkpoint byte integrity mismatch')
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(z['metadata'].tobytes())
        if metadata['schema'] != SCHEMA or metadata['source_identity'] != source_identity():
            raise ValueError('checkpoint source/schema mismatch')
        if type(metadata['plastic']) is not bool or type(metadata['bytes_seen']) is not int or metadata['bytes_seen'] < 0:
            raise ValueError('invalid checkpoint policy/counter')
        model = ObservationCore(plastic=metadata['plastic'])
        h = bytes.fromhex(metadata['history'])
        if len(h) > 4 or any(b not in b'0123' for b in h) or len(metadata['stores']) != 4:
            raise ValueError('invalid checkpoint history/stores')
        if not np.isfinite(metadata['t']) or metadata['t'] < 0:
            raise ValueError('invalid checkpoint time')
        for i, s in enumerate(model.stores):
            entry = metadata['stores'][i]
            for name in ('fast', 'slow', 'adapt'):
                dest = getattr(s.fly.m, name)
                arr = z[f'{i}_{name}']
                if arr.shape != dest.shape or arr.dtype != dest.dtype or not np.isfinite(arr).all():
                    raise ValueError('invalid native checkpoint array')
                dest[...] = arr
            if entry['brain_t'] != metadata['t'] or abs(entry['elapsed'] - entry['elapsed_base'] - metadata['t']) > 1e-7:
                raise ValueError('checkpoint native clock mismatch')
            s.brain_t, s.elapsed_base, s.teach_seen = entry['brain_t'], entry['elapsed_base'], entry['teach_seen']
            s.fly.m.elapsed = np.float64(entry['elapsed'])
            s.fly.m.event_count = np.uint64(entry['events'])
            s.fly.m.presentation_count = np.uint64(entry['presentations'])
        model.history, model.t, model.bytes_seen = h, float(metadata['t']), metadata['bytes_seen']
        cached = metadata['cached']
        if cached is not None:
            p = z['cached_p'].copy()
            codes = [z[f'cached_code_{i}'].copy() for i in range(4)]
            if p.shape != (4,) or not np.isfinite(p).all() or np.any(p <= 0) or abs(p.sum() - 1.) > 1e-12:
                raise ValueError('invalid cached probability')
            if any(x.shape != model.stores[0].fly.m.adapt.shape or not np.isfinite(x).all() or np.any(x < 0) for x in codes):
                raise ValueError('invalid cached native code')
            if bytes.fromhex(cached['history']) != h or not np.isfinite(cached['t']) or cached['t'] < model.t:
                raise ValueError('cached pre-arrival context/time mismatch')
            model.cached = dict(t=cached['t'], history=h, p=p, codes=codes)
        if model.state_digest() != metadata['expected_state_digest']:
            raise ValueError('restored native state mismatch')
    return model
