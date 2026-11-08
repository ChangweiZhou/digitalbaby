"""E16-WR orchestration primitives; importing this file never imports a learner."""
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'exp45'
SEEDS = (73129, 93683, 116089)
NAMES = ('add97','sub97','max97','min97','compose','add73','sub73','max73','min73')
DOSES = (6144,6144,6144,6144,8192,6144,6144,6144,6144)
DEADLINE = datetime(2026,11,8,21,30,tzinfo=timezone.utc)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k!='cpu'}
    if isinstance(value,list):return [clean(v) for v in value]
    return value

def producers():
    p=subprocess.run(['ps','-axo','pid=,command='],capture_output=True,text=True,check=True)
    rows=[]
    for line in p.stdout.splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2:continue
        try: args=shlex.split(fields[1])
        except ValueError:continue
        if not args:continue
        if Path(args[0]).name.lower().startswith('python') and any(Path(a).name=='exp16_integrated.py' for a in args[1:]):
            rows.append(dict(pid=int(fields[0]),command=fields[1]))
    return rows

def require_machine_free():
    now=datetime.now(timezone.utc)
    if now<DEADLINE:raise RuntimeError('Before16:30EST; no learner qualification or training allowed')
    active=producers()
    if active:raise RuntimeError('Another producer remains active: '+json.dumps(active))
    return dict(observed_at=now.isoformat(),active_other_producers=active)

def verify_lock():
    lock=read(ROOT/'E16WR_LOCK.json')
    for rel,digest in {**lock['source_hashes'],**lock['input_hashes']}.items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('Locked file changed: '+rel)
    return lock

def validate_receipt(job, short=False, require_meta=True):
    p=OUT/job['receipt'];d=read(p)
    assert d['arm']=='C' and d['seed']==job['seed']
    assert len(d['stages'])==10
    names=list(NAMES) if job['role']!='profile' else [*NAMES[:4],*NAMES[5:],'compose']
    assert [s['task'] for s in d['stages'][1:]]==names
    for stage in d['stages']:
        assert all(k in stage['scores'] for k in ('add97','min97','add73','min73'))
        if not short:
            for name in ('add97','sub97','max97','min97','add73','sub73','max73','min73'):
                v=stage['scores'][name];assert v['n']==600 and 0<=v['exact']<=v['n']
        if job['role']!='profile':assert all(k in stage['scores']['compose'] for k in ('+-','-+','><','<>','within'))
    if job['role']=='profile':
        assert clean(d)==clean(read(OUT/'e16w_profile_C_73129_e8_life_C_73129.json'))
    elif require_meta:
        meta=read(OUT/job['metadata'])
        assert meta['status']=='COMPLETE' and meta['updates']==sum(DOSES)
        assert meta['parameters']==470801 and len(meta['stages'])==9
        assert meta['decay']==.25 and meta['learning_rule']=='sign'
        assert meta['target_source']==job['targets'] and meta['capacity_per_stage']==job['capacity']
        h=read(p.with_suffix('.reservoir.json'))
        assert len(h)==9
        for i,(row,expected) in enumerate(zip(h,DOSES),1):
            r=row['reservoirs'][-1]
            presented=expected*(64 if i==1 else 32)
            assert r['stage']==i and r['presentations']==presented
            assert r['slots']==min(job['capacity'],presented)
            assert 0<r['distinct_prompts']<=r['slots']
            assert set(map(int,row['replay']['draws']))<=set(range(1,i))
            assert sum(row['replay']['draws'].values())==(0 if i==1 else expected*32)
            if job['targets']=='observed':
                assert row['replay']['composition']['wrong_full']==0 and row['replay']['operation']['wrong_full']==0
    return d
