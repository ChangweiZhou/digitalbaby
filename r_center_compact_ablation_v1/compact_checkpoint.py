"""Atomic array checkpoint for an arm plus its exact cursor and receipt history."""
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
import numpy as np
from compact_bridge import make_core

ARRAYS = ('w', 'bias', 'fly_fast', 'fly_slow', 'fly_adapt', 'fe_p', 'pending_x')


def seal(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(core, path, cursor, identity):
    arrays, states = {}, []
    for i, m in enumerate(core.models):
        state = m.snapshot()
        states.append({k: v for k, v in state.items() if k not in ARRAYS})
        if m in core.private: states[-1]['visible_hex'] = m.fe.visible.hex()
        for k in ARRAYS: arrays[f's{i}_{k}'] = state[k]
    arrays['cached'] = np.zeros(4) if core.cached is None else core.cached.copy()
    keys = ('prediction_time', 'cue_count', 'awaiting_newline', 'last_time', 'records', 'last_write')
    meta = {'schema': 'COMPACT_JOB_CHECKPOINT_V1', 'identity': identity, 'cursor': cursor,
            'states': states, 'adapter': {k: getattr(core, k) for k in keys}, 'cached_valid': core.cached is not None,
            'digest': core.state_digest(), 'arrays': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}}
    meta['seal'] = seal(meta)
    arrays['metadata_json'] = np.frombuffer(json.dumps(meta, sort_keys=True, allow_nan=False).encode(), dtype=np.uint8)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def load(path, arm, identity):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or sum(i.file_size for i in z.infolist()) > 128 * 1024**2:
            raise ValueError('invalid checkpoint container')
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z['metadata_json'].tobytes().decode(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
        saved = meta.pop('seal')
        if saved != seal(meta) or meta['schema'] != 'COMPACT_JOB_CHECKPOINT_V1' or meta['identity'] != identity:
            raise ValueError('checkpoint identity/seal mismatch')
        core = make_core(arm)
        count = len(core.models)
        expected = {f's{i}_{k}' for i in range(count) for k in ARRAYS} | {'cached'}
        if set(z.files) != expected | {'metadata_json'} or set(meta['arrays']) != expected or len(meta['states']) != count:
            raise ValueError('checkpoint roster mismatch')
        for key in expected:
            a = z[key]
            if a.dtype.hasobject or not np.isfinite(a).all() or hashlib.sha256(a.tobytes()).hexdigest() != meta['arrays'][key]:
                raise ValueError('checkpoint array mismatch')
        for i, m in enumerate(core.models):
            state = dict(meta['states'][i]); visible = state.pop('visible_hex', None)
            initial = m.snapshot()
            if set(state) != set(initial) - set(ARRAYS): raise ValueError('checkpoint state fields')
            for k in ARRAYS:
                a = z[f's{i}_{k}']
                if a.shape != initial[k].shape or a.dtype != initial[k].dtype: raise ValueError('checkpoint array layout')
                state[k] = a.copy()
            m.restore(state)
            if m in core.private:
                b = bytes.fromhex(visible)
                if len(b) > 4 or 10 in b or 32 in b: raise ValueError('bad CONTENT suffix')
                m.fe.visible = b
            elif visible is not None: raise ValueError('misplaced CONTENT suffix')
        core.cached = z['cached'].copy() if meta['cached_valid'] else None
    for k, value in meta['adapter'].items(): setattr(core, k, value)
    if core.state_digest() != meta['digest']: raise ValueError('restored digest mismatch')
    cursor = meta['cursor']
    if 'world' in cursor:
        from compact_fixture import BRANCHES, make_world
        world = make_world(cursor['world'])
        limit, branches = cursor['limits']
        bi, index = cursor['branch_index'], cursor['next_record']
        if (type(bi) is not int or type(index) is not int or not 0 <= bi <= branches <= 4
                or not 0 <= index <= limit <= 864 or cursor['arm'] != arm
                or cursor['fixture_sha256'] != world['sha256']):
            raise ValueError('invalid checkpoint cursor')
        expected_names = list(BRANCHES[:bi])
        current = BRANCHES[bi] if bi < branches else None
        if current in cursor['branches']: expected_names.append(current)
        if set(cursor['branches']) != set(expected_names):
            raise ValueError('cursor branch roster mismatch')
        for branch, history in cursor['branches'].items():
            wanted = index if branch == current else limit
            if len(history['records']) != wanted or [r['index'] for r in history['records']] != list(range(wanted)):
                raise ValueError('cursor/history mismatch')
        if (core.records != (index if current is not None else limit) or core.cached is not None
                or core.cue_count != 0 or core.awaiting_newline):
            raise ValueError('cursor/model state mismatch')
    elif cursor != {'component_only': True}:
        raise ValueError('invalid component checkpoint cursor')
    return core, cursor
