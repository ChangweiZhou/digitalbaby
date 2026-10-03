import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path


def atomic_json(path, doc):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(doc, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def commit_receipt(path, doc):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise ValueError('cannot overwrite a completed receipt')
    doc = dict(doc)
    doc['receipt_sha256'] = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    fd, temp = tempfile.mkstemp(prefix=path.name + '.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as f:
                f.write(json.dumps(doc, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
            raw.flush(); os.fsync(raw.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)
