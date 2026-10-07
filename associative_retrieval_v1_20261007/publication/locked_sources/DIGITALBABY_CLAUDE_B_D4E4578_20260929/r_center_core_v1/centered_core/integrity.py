"""Fail closed on missing/stale executable source locks for this NEW version."""
import hashlib
import json
from pathlib import Path
from .bootstrap import ROOT


def source_files():
    files = [p for folder in ('centered_core', 'vendor', 'tests', 'scripts') for p in (ROOT / folder).rglob('*')
             if p.is_file() and 'scratch' not in p.parts and '__pycache__' not in p.parts
             and p.suffix in ('.py', '.json', '.npz')]
    return sorted(files + [ROOT / 'requirements.txt'])


def source_map():
    files = source_files()
    if any(p.is_symlink() for p in files): raise ValueError('symlinked source is not supported')
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def source_identity():
    return hashlib.sha256(json.dumps(source_map(), sort_keys=True).encode()).hexdigest()


def verify_sources():
    path = ROOT / 'SOURCE_LOCK.json'
    if not path.is_file(): raise ValueError('engineering SOURCE_LOCK.json missing')
    doc = json.loads(path.read_text())
    actual = source_map()
    if doc.get('schema') != 'R_CENTER_ENGINEERING_SOURCE_LOCK_V1' or doc.get('files') != actual:
        raise ValueError('engineering source lock mismatch')
    if doc.get('source_identity') != source_identity(): raise ValueError('engineering source identity mismatch')
    return doc['source_identity']
