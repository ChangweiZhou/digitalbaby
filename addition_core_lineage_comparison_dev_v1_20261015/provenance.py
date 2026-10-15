"""Snapshot runtime identity before each technical cycle, then reject edits."""
import json
import platform
import shutil
import sys
from pathlib import Path
import bootstrap
import core
import numpy, scipy, numba, torch
from io_utils import digest, write, now, read


def snapshot(cycle):
    core.Compartment()  # zero-event birth completes lazy native module discovery
    folder=bootstrap.HERE/'cycles'/f'cycle{cycle}'
    paths=set(bootstrap.HERE.glob('*.py'))
    paths.update(bootstrap.HERE/name for name in ('DESIGN.json','OUTPUT_CODEBOOK.json','EXPERIMENT.md','VERSION_LINEAGE.json','DATA_MANIFEST.json')
                 if (bootstrap.HERE/name).is_file())
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p:
            p=Path(p).resolve()
            if p.suffix=='.py' and p.is_file() and p.is_relative_to(bootstrap.ROOT):
                paths.add(p)
    vendor=bootstrap.ROOT/'r_center_core_v1/vendor/package'
    for suffix in ('*.json','*.npz','*.csv'):
        paths.update(p for p in vendor.rglob(suffix) if p.is_file())
    paths.add(bootstrap.ROOT/'r_center_core_v1/centered_core/core.py')
    lock=dict(utc=now(),cycle=cycle,science=False,python=sys.version,executable=sys.executable,
              platform=platform.platform(),versions={m.__name__:m.__version__ for m in (numpy,scipy,numba,torch)},
              module_locations={m.__name__:m.__file__ for m in (numpy,scipy,numba,torch)},
              files={str(p.relative_to(bootstrap.ROOT)):digest(p) for p in sorted(paths)})
    for p in paths:
        if p.parent==bootstrap.HERE:
            target=folder/'source'/p.name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target)
    write(folder/'SOURCE_LOCK.json',lock)
    return lock


def verify(cycle, local_only=False):
    lock=read(bootstrap.HERE/'cycles'/f'cycle{cycle}'/'SOURCE_LOCK.json')
    for path,expected in lock['files'].items():
        if local_only and not (bootstrap.ROOT/path).is_relative_to(bootstrap.HERE):
            continue
        if digest(bootstrap.ROOT/path)!=expected:
            raise AssertionError('source/asset changed: '+path)
    return True
