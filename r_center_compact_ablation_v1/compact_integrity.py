import hashlib
import json
from pathlib import Path
from compact_bridge import ROOT, PARENT, verify_parent


def source_map():
    files = sorted(ROOT.glob('*.py')) + [ROOT / 'PROTOCOL.md', ROOT / 'SPEC.json']
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def identity():
    parent = verify_parent()
    return hashlib.sha256(json.dumps({'files': source_map(), 'parent': parent}, sort_keys=True).encode()).hexdigest()


def verify():
    doc = json.loads((ROOT / 'SOURCE_LOCK.json').read_text())
    if doc['schema'] != 'COMPACT_SCIENCE_SOURCE_V1' or doc['files'] != source_map() or doc['identity'] != identity():
        raise ValueError('compact source lock mismatch')
    return doc['identity']
