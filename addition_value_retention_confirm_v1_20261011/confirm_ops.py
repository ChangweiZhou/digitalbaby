"""Atomic operational I/O, source verification and committed-world accounting."""
import gzip,hashlib,json,os,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results';OPS=ROOT/'ops'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    p=Path(path);raw=p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)
def atomic(path,value,exclusive=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.pending')
    raw=json.dumps(value,indent=2,sort_keys=True,allow_nan=False).encode()
    if p.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
    with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    try:
        if exclusive:os.link(tmp,p)
        else:os.replace(tmp,p)
    finally:tmp.unlink(missing_ok=True)
def counts():
    paths=list((RESULTS/'receipts').glob('WORLD_*.json.gz'))
    return dict(committed_worlds=len(paths),total_worlds=64,committed_world_arm_jobs=2*len(paths),total_world_arm_jobs=128,
                committed_histories=6*len(paths),total_histories=384,committed_training_bytes=24072*len(paths),total_training_bytes=1540608)
def verify_sources():
    lock=read(ROOT/'SOURCE_LOCK.json')
    for name,digest in lock['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('locked file changed: '+name)
    from runner import source_lock
    if source_lock()!=lock['runner_identity']:raise ValueError('scientific parent/source identity changed')
    return sha(ROOT/'SOURCE_LOCK.json')
