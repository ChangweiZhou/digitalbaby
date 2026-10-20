"""Artifact I/O and source identities, separate from the learner."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.pending')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(temp, path)


def source_identity():
    files = list(HERE.glob('*.py'))+[HERE/'SPEC.json']+[
        p for p in (HERE/'upstream').rglob('*') if p.suffix in ('.py','.json','.npz') and '__pycache__' not in p.parts]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files) if p.is_file()}


def source_digest():
    return hashlib.sha256(json.dumps(source_identity(), sort_keys=True).encode()).hexdigest()
