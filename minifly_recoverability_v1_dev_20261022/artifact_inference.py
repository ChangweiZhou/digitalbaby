"""Independent frozen inference for compact-bundle reduction; no fitting/native code.

Reconstructing omitted derived logits reads saved native responses and frozen
weights only. It never calls the experiment's head, optimizer, or native engine.
"""
from pathlib import Path
import numpy as np


def logits_for(root, meta, row, raw):
    i = meta['pools'].index(row['pool'])
    source = raw[f'D{i}']
    if row['readout'] == 'DIRECT':
        return source.mean(-1)
    if row['lesion'] == 'ZERO':
        source = np.zeros_like(source)
    elif row['lesion'] == 'FOREIGN':
        source = np.roll(source, 1, axis=0)
    else:
        assert row['lesion'] == 'NONE'
    root = Path(root)
    scale = np.load(root / 'heads' / f"scale_{int(meta['STD'])}.npy")
    q = source / scale[None, None, :, None]
    if row['readout'] == 'MEAN':
        q = np.repeat(q.mean(-1, keepdims=True), 2, axis=-1)
    x = np.empty((len(q), 9, 171))
    x[:, :, :162] = q.reshape(len(q), 1, 162)
    x[:, :, 162:] = np.eye(9)
    name = f"{int(meta['STD'])}_{row['readout']}_{row['head_seed']}.npz"
    with np.load(root / 'heads' / name) as p:
        return np.tanh(x @ p['w1'] + p['b1']) @ p['w2'] + p['b2'] + q.mean(-1)
