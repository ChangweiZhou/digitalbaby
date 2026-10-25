import json,os,time,resource,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parent;START=time.process_time()
def save(n,x):(H/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(n):return json.loads((H/n).read_text())
def event(**x):
 with (H/'LEDGER.jsonl').open('a') as f:f.write(json.dumps(dict(cpu=time.process_time(),**x))+'\n');f.flush();os.fsync(f.fileno())
def cpu():
 r=resource.getrusage(resource.RUSAGE_CHILDREN);return time.process_time()+r.ru_utime+r.ru_stime

def budget(reserve=0):
 if cpu()+reserve>7100:raise RuntimeError('V150 CPU admission reserve exhausted')
