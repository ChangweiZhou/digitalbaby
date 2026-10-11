"""Canonical immutable inherited anatomy; no parent files are edited."""
import os
import sys
import hashlib
from functools import lru_cache
from pathlib import Path
import numpy as np
import config

sys.dont_write_bytecode = True
os.environ.setdefault('NUMBA_CACHE_DIR', str(config.ROOT / 'scratch' / 'numba'))


@lru_cache(maxsize=1)
def canonical():
    parent = config.ROOT.parent / 'autonomous_observation_v1_20261007'
    sys.path.insert(0, str(parent))
    import core as inherited
    import checkpoint as inherited_checkpoint
    source = inherited_checkpoint.source_identity()
    native = inherited.Compartment()
    model = native.fly.m
    eligible = np.flatnonzero(
        (np.abs(np.asarray(model.T)[:, 2:]).sum(1) > 0) &
        (np.abs(np.asarray(model.Q)[:, 2:]).sum(1) > 0)
    )
    groups = np.random.default_rng(20261011).permutation(eligible)[:465].reshape(31, 15)
    if len(np.unique(groups)) != 465 or len(model.fast) != 5177:
        raise ValueError('canonical projection geometry')
    groups.flags.writeable = False
    sys.path.insert(0, str(config.ROOT))
    fingerprint = dict(native.birth_record)
    fingerprint.update(n_kc=len(model.fast),
                       Q_sha256=hashlib.sha256(np.asarray(model.Q).tobytes()).hexdigest(),
                       T_sha256=hashlib.sha256(np.asarray(model.T).tobytes()).hexdigest(),
                       projection_sha256=hashlib.sha256(groups.tobytes()).hexdigest())
    return dict(n=len(model.fast), groups=groups, birth=fingerprint,
                inherited_source=source)
