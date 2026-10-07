import hashlib
import json
import os
from pathlib import Path
import bootstrap_ops as boot
from identity import identity as native_identity, source_map
from io_utils import digest, atomic_json

SCREEN = tuple(range(71009001, 71009009))
CONFIRM = tuple(range(71010001, 71010065))
DEV = (71008001, 71008002, 71008003)
ASSAYS = ('lifetime', 'reuse')
CONDITIONS = ('ERROR', 'LINK', 'PERM')
WORKERS = 8
DISK_CAP = 8 * 1024**3
RSS_CAP = 1024**3
DISPATCH_SECONDS = 9 * 3600
OUTER_SECONDS = 10 * 3600

def roster(stage):
    worlds = SCREEN if stage == 'screen' else CONFIRM if stage == 'confirm' else ()
    if not worlds:
        raise ValueError('invalid science stage')
    return [{'world': w, 'assay': a, 'job': f'{w}_{a}'} for w in worlds for a in ASSAYS]

def require_scope(world, assay, stage, qualification=False):
    if assay not in ASSAYS:
        raise ValueError('assay outside frozen design')
    if qualification:
        if stage != 'qualification' or world not in DEV:
            raise ValueError('qualification world restriction')
    elif not any(j['world'] == world and j['assay'] == assay for j in roster(stage)):
        raise ValueError('science world outside preregistered stage')

def operation_sources():
    paths = list(boot.OPS.glob('*.py')) + [boot.OPS / 'EXECUTION_LOCK.md']
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

def lock_sources():
    original = json.loads((boot.ROOT / 'SOURCE_LOCK.json').read_text())
    current = source_map()
    if current != original['files'] or native_identity() != original['identity']:
        raise ValueError('qualified frozen source differs from original SOURCE_LOCK')
    sources = operation_sources()
    d = {'schema': 'LINK_SCIENCE_OPERATIONS_LOCK_V1',
         'native_identity': original['identity'], 'operations_identity': digest(sources),
         'operations_files': sources,
         'experiment_sha256': hashlib.sha256((boot.ROOT / 'EXPERIMENT.md').read_bytes()).hexdigest(),
         'unchanged_native_files': True, 'workers': WORKERS,
         'screen_worlds': list(SCREEN), 'confirm_worlds': list(CONFIRM)}
    atomic_json(boot.OPS / 'SOURCE_LOCK.json', d)
    return d

def require_sources():
    d = json.loads((boot.OPS / 'SOURCE_LOCK.json').read_text())
    if native_identity() != d['native_identity']:
        raise ValueError('native source changed')
    if operation_sources() != d['operations_files']:
        raise ValueError('operations source changed')
    if hashlib.sha256((boot.ROOT / 'EXPERIMENT.md').read_bytes()).hexdigest() != d['experiment_sha256']:
        raise ValueError('scientific protocol changed')
    return d

def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False

