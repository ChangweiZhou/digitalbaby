"""Non-pickle atomic full-state checkpoints, including local conflict evidence."""
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
import numpy as np
from v2_core import TrialCore
from v2_fixture import make_world

ARRAYS = ('w', 'bias', 'fly_fast', 'fly_slow', 'fly_adapt', 'fe_p', 'pending_x')


def seal(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(core, path, cursor, identity):
    arrays, states = {}, []
    for i, m in enumerate(core.models):
        state = m.snapshot()
        states.append({k: v for k, v in state.items() if k not in ARRAYS})
        if i >= 4: states[-1]['visible_hex'] = m.fe.visible.hex()
        for k in ARRAYS: arrays[f's{i}_{k}'] = state[k]
    if core.cached is not None or core.private_prediction is not None: raise ValueError('checkpoint inside pending outcome')
    arrays['conflicts'] = core.conflicts.copy()
    keys = ('prediction_time', 'cue_count', 'awaiting_newline', 'last_time', 'records', 'last_write')
    meta = {'schema': 'PERSISTENT_V2_CHECKPOINT', 'identity': identity, 'arm': core.arm, 'cursor': cursor,
            'states': states, 'adapter': {k: getattr(core, k) for k in keys}, 'digest': core.state_digest(),
            'arrays': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}}
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
        if len(z.namelist()) != len(set(z.namelist())) or sum(i.file_size for i in z.infolist()) > 256 * 1024**2:
            raise ValueError('invalid checkpoint container')
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z['metadata_json'].tobytes(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
        saved = meta.pop('seal')
        if saved != seal(meta) or meta['identity'] != identity or meta['arm'] != arm or meta['schema'] != 'PERSISTENT_V2_CHECKPOINT':
            raise ValueError('checkpoint identity/seal')
        core = TrialCore(arm)
        expected = {f's{i}_{k}' for i in range(8) for k in ARRAYS} | {'conflicts'}
        if set(z.files) != expected | {'metadata_json'} or set(meta['arrays']) != expected or len(meta['states']) != 8:
            raise ValueError('checkpoint arrays roster')
        for key in expected:
            a = z[key]
            if a.dtype.hasobject or not np.isfinite(a).all() or hashlib.sha256(a.tobytes()).hexdigest() != meta['arrays'][key]:
                raise ValueError('checkpoint array corruption')
        for i, m in enumerate(core.models):
            initial = m.snapshot(); state = dict(meta['states'][i]); visible = state.pop('visible_hex', None)
            if set(state) != set(initial) - set(ARRAYS): raise ValueError('snapshot field mismatch')
            for k in ARRAYS:
                a = z[f's{i}_{k}']
                if a.shape != initial[k].shape or a.dtype != initial[k].dtype: raise ValueError('snapshot array layout')
                state[k] = a.copy()
            m.restore(state)
            if i >= 4:
                b = bytes.fromhex(visible)
                if len(b) > 4 or 10 in b or 32 in b: raise ValueError('invalid visible suffix')
                m.fe.visible = b
            elif visible is not None: raise ValueError('unexpected suffix')
        if z['conflicts'].shape != core.conflicts.shape or z['conflicts'].dtype != core.conflicts.dtype or (z['conflicts'] > 3).any():
            raise ValueError('conflict evidence layout')
        core.conflicts = z['conflicts'].copy()
    for k, v in meta['adapter'].items(): setattr(core, k, v)
    if core.state_digest() != meta['digest']: raise ValueError('restored state mismatch')
    c = meta['cursor']; world = make_world(c['world'], c['assay'])
    bi, index = c['branch_index'], c['next_record']; limit, count = c['limits']
    if (type(bi) is not int or type(index) is not int or not 0 <= bi <= count <= len(world['branches'])
            or not 0 <= index <= limit <= len(world['events']) or c['fixture_sha256'] != world['sha256'] or c['arm'] != arm):
        raise ValueError('checkpoint cursor identity')
    names = world['branches'][:bi]; current = world['branches'][bi] if bi < count else None
    if current in c['branches']: names = names + [current]
    if set(c['branches']) != set(names): raise ValueError('cursor branch roster')
    for branch, history in c['branches'].items():
        wanted = index if branch == current else limit
        if len(history['records']) != wanted or [r['index'] for r in history['records']] != list(range(wanted)):
            raise ValueError('cursor/history mismatch')
    if core.records != (index if current else limit) or core.cue_count or core.awaiting_newline or core.cached is not None:
        raise ValueError('cursor/core mismatch')
    return core, c
