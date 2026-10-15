import gzip
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.pending')
    text = json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if path.suffix == '.gz':
        with gzip.open(temporary,'wt') as f:
            f.write(text)
    else:
        temporary.write_text(text)
    temporary.replace(path)


def read(path):
    path=Path(path)
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
        return json.load(f)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
