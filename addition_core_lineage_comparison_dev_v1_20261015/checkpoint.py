"""Internal trusted checkpoints only. Never load user-provided pickle files."""
import gzip
import os
import pickle
from pathlib import Path


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+f'.{os.getpid()}.pending')
    with gzip.open(pending,'wb',compresslevel=1) as f:
        pickle.dump(value,f,protocol=5)
    pending.replace(path)


def load_internal(path):
    with gzip.open(path,'rb') as f:
        return pickle.load(f)
