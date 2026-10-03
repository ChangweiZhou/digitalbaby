"""Atomic, source-bound NPZ checkpoint. No pickle, arbitrary classes or code."""
import hashlib
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from .bootstrap import ROOT
from .core import CenteredCore, VERSION, SCALES, ALPHABET
from .integrity import source_identity

SCHEMA = 'R_CENTER_CHECKPOINT_V1'
ARRAYS = ('w', 'bias', 'fly_fast', 'fly_slow', 'fly_adapt', 'fe_p', 'pending_x')


def array_info(a):
    return {'shape': list(a.shape), 'dtype': a.dtype.str,
            'sha256': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


def sealed(meta):
    return hashlib.sha256(json.dumps(meta, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def save(core, path):
    arrays, states = {}, []
    for i, m in enumerate(core.models):
        s = m.snapshot()
        states.append({k: v for k, v in s.items() if k not in ARRAYS})
        if i >= 4:
            # Native snapshot() omits the CONTENT route's visible suffix.
            states[-1]['visible_hex'] = m.fe.visible.hex()
        for k in ARRAYS: arrays[f's{i}_{k}'] = s[k]
    arrays['cached'] = np.zeros(4, dtype=np.float64) if core.cached is None else core.cached.copy()
    meta = {'schema': SCHEMA, 'version': VERSION, 'source_identity': source_identity(), 'scales': list(SCALES),
            'states': states, 'cached_valid': core.cached is not None,
            'prediction_time': core.prediction_time, 'cue_count': core.cue_count,
            'awaiting_newline': core.awaiting_newline, 'last_time': core.last_time,
            'records': core.records, 'last_write': core.last_write, 'state_digest': core.state_digest(),
            'arrays': {k: array_info(v) for k, v in arrays.items()}}
    meta['seal'] = sealed(meta)
    arrays['metadata_json'] = np.frombuffer(json.dumps(meta, sort_keys=True, allow_nan=False).encode(), dtype=np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **arrays)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def nonfinite(_):
    raise ValueError('nonfinite checkpoint JSON')


def numeric(v, *, integer=False, optional=False):
    if optional and v is None: return
    if type(v) not in ((int,) if integer else (int, float)) or not math.isfinite(v) or v < 0:
        raise ValueError('invalid checkpoint scalar')


def load(path):
    # Bound decompressed input; reject duplicate members rather than ambiguous reads.
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(set(names)) != len(names) or sum(i.file_size for i in z.infolist()) > 64 * 1024 * 1024:
            raise ValueError('ambiguous or oversized checkpoint')
    with np.load(path, allow_pickle=False) as z:
        raw = z['metadata_json']
        if raw.dtype != np.uint8 or raw.ndim != 1: raise ValueError('invalid checkpoint metadata')
        meta = json.loads(raw.tobytes().decode(), parse_constant=nonfinite)
        seal = meta.pop('seal')
        if seal != sealed(meta): raise ValueError('checkpoint metadata seal mismatch')
        if meta['schema'] != SCHEMA or meta['version'] != VERSION or meta['scales'] != list(SCALES):
            raise ValueError('checkpoint contract mismatch')
        if meta['source_identity'] != source_identity(): raise ValueError('checkpoint source mismatch')
        expected = {f's{i}_{k}' for i in range(8) for k in ARRAYS} | {'cached'}
        if set(z.files) != expected | {'metadata_json'} or set(meta['arrays']) != expected:
            raise ValueError('checkpoint array roster mismatch')
        data = {k: z[k].copy() for k in expected}
        for k, a in data.items():
            if a.dtype.hasobject or not np.isfinite(a).all() or array_info(a) != meta['arrays'][k]:
                raise ValueError('checkpoint array mismatch: ' + k)
    if len(meta['states']) != 8: raise ValueError('checkpoint store count mismatch')
    for k in ('cached_valid', 'awaiting_newline'):
        if type(meta[k]) is not bool: raise ValueError('checkpoint bool mismatch')
    for k in ('cue_count', 'records'): numeric(meta[k], integer=True)
    if meta['cue_count'] > 12: raise ValueError('invalid cue cursor')
    numeric(meta['last_time']); numeric(meta['prediction_time'], optional=True)
    if meta['cached_valid'] != (meta['prediction_time'] is not None): raise ValueError('invalid prediction state')
    if meta['cached_valid'] and (meta['cue_count'] != 12 or meta['awaiting_newline']
                                or meta['prediction_time'] != meta['last_time']):
        raise ValueError('invalid pending outcome')
    if meta['awaiting_newline'] and meta['cue_count'] != 12: raise ValueError('invalid newline state')
    if data['cached'].shape != (4,) or data['cached'].dtype != np.float64:
        raise ValueError('invalid cached prediction')
    out = CenteredCore()  # Each store born independently; canonical B only installed at birth.
    for i, m in enumerate(out.models):
        state = dict(meta['states'][i])
        visible = state.pop('visible_hex', None)
        if (i >= 4) != (visible is not None): raise ValueError('CONTENT suffix missing or misplaced')
        initial = m.snapshot()
        if set(state) != set(initial) - set(ARRAYS): raise ValueError('checkpoint state fields mismatch')
        for k in ARRAYS:
            a = data[f's{i}_{k}']
            if a.shape != initial[k].shape or a.dtype != initial[k].dtype:
                raise ValueError('checkpoint native shape/dtype mismatch')
            state[k] = a
        for k in ('fly_elapsed', 'brain_t', 'elapsed_base', 'fe_t'): numeric(state[k])
        for k in ('fly_event_count', 'fly_presentation_count', 'bytes_seen', 'teach_seen'):
            numeric(state[k], integer=True)
            if state[k] >= 2**64: raise ValueError('checkpoint counter overflow')
        for k in ('last_byte_t', 'pending_t'): numeric(state[k], optional=True)
        for k in ('prev1', 'prev2'):
            if state[k] is not None and (type(state[k]) is not int or not 0 <= state[k] <= 255):
                raise ValueError('invalid prior byte')
        if type(state['pending_valid']) is not bool or state['pending_valid'] != (state['pending_t'] is not None):
            raise ValueError('invalid pending native event')
        if not math.isfinite(state['logp_sum']): raise ValueError('invalid predictor accounting')
        m.restore(state)
        if i >= 4:
            b = bytes.fromhex(visible)
            if len(b) > 4 or 10 in b or 32 in b: raise ValueError('invalid CONTENT suffix')
            m.fe.visible = b
    out.cached = data['cached'] if meta['cached_valid'] else None
    for k in ('prediction_time', 'cue_count', 'awaiting_newline', 'last_time', 'records', 'last_write'):
        setattr(out, k, meta[k])
    if out.state_digest() != meta['state_digest']: raise ValueError('checkpoint restored-state mismatch')
    return out
