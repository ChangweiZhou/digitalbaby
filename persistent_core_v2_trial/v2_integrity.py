import hashlib
import json
import bootstrap
from centered_core.integrity import verify_sources as parent_verify


def source_map():
    files = sorted(bootstrap.ROOT.glob('*.py')) + [bootstrap.ROOT / 'PROTOCOL.md', bootstrap.ROOT / 'SPEC.json']
    # Imported experiment helpers and both reused fixtures are locked as well.
    files += [bootstrap.COMPACT / name for name in ('compact_fixture.py', 'compact_experiment.py',
                                                   'compact_storage.py', 'compact_bridge.py')]
    return {str(p.relative_to(bootstrap.ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def identity():
    return hashlib.sha256(json.dumps({'files': source_map(), 'parent': parent_verify()}, sort_keys=True).encode()).hexdigest()


def verify():
    lock = json.loads((bootstrap.ROOT / 'SOURCE_LOCK.json').read_text())
    if lock['identity'] != identity() or lock['files'] != source_map(): raise ValueError('source lock changed')
    return lock['identity']
