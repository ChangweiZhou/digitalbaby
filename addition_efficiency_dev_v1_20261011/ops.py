"""Atomic operations tests only. No science dispatcher or automatic retry."""
import hashlib,json,os,time,gzip,math
from pathlib import Path
def atomic_json(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('x') as f:
        json.dump(obj,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def read_receipt(path):
    with gzip.open(path,'rt') as f:return json.load(f)
def validate_cycle_sources(path):
    p=Path(path);lock=json.loads((p/'SOURCE_LOCK.json').read_text())
    for name,want in lock['files'].items():
        actual=hashlib.sha256((p/'source'/name).read_bytes()).hexdigest()
        if actual!=want:raise ValueError('archived source mismatch '+name)
    return True
def admit(elapsed):
    if isinstance(elapsed,bool) or not isinstance(elapsed,(float,int)) or not math.isfinite(elapsed) or elapsed<0:raise ValueError('elapsed')
    return elapsed<32400
def liveness(pid):
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    return True
