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
DEADLINE = datetime(2026,11,9,0,35,tzinfo=timezone.utc)

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
        if Path(args[0]).name.lower().startswith('python') and any(Path(a).name in {'exp16_integrated.py','e16wr_worker.py','e16wr_campaign.py'} for a in args[1:]):
            rows.append(dict(pid=int(fields[0]),command=fields[1]))
    return rows

def require_machine_free():
    now=datetime.now(timezone.utc)
    if now<DEADLINE:raise RuntimeError('Before19:35EST; no learner qualification or training allowed')
    active=producers()
    if active:raise RuntimeError('Another producer remains active: '+json.dumps(active))
    return dict(observed_at=now.isoformat(),active_other_producers=active)

def verify_lock():
    lock=read(ROOT/'E16WR2_LOCK.json')
    for rel,digest in {**lock['source_hashes'],**lock['input_hashes']}.items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('Locked file changed: '+rel)
    return lock

def validate_receipt(job, short=False, require_meta=True):
    p=OUT/job['receipt'];d=read(p)
    assert d['arm']=='C' and d['seed']==job['seed'] and len(d['stages'])==10
    is_a=job['role']=='A'
    names=list(NAMES) if is_a else [*NAMES[:4],*NAMES[5:],'compose']
    doses=list(DOSES) if is_a else [6144]*8+[8192]
    assert [s['task'] for s in d['stages'][1:]]==names
    ops=[n for n in NAMES if n!='compose']
    for i,s in enumerate(d['stages']):
        for n in ops:
            v=s['scores'][n];assert (short or v['n']==600) and 0<=v['exact']<=v['n']
        if is_a or i==9:
            for k in ('+-','-+','><','<>','within'):
                v=s['scores']['compose'][k];assert (short or v['n']==600) and 0<=v['exact']<=v['final']<=v['n']
    if require_meta:
        m=read(OUT/job['metadata'])
        assert m['status']=='COMPLETE' and m['updates']==57344 and m['parameters']==470801
        assert m['key']==job['key'] and m['seed']==job['seed'] and m['target_source']=='observed'
        assert m['capacity']==job['capacity'] and m['decay']==.25
        assert [x['updates'] for x in m['stages']]==doses
        assert [x['task'] for x in m['stages']]==names
        assert m['lock_sha256']==sha(ROOT/'E16WR2_LOCK.json') and not m['gate']['active_other_producers']
        if not is_a:
            ref=next(x for x in read(ROOT/'E16WR2_B_INPUT_REFERENCE.json')['seeds'] if x['seed']==job['seed'])
            assert m['prompt_sha256']==ref['prompt_sha256']
            assert [s['prompt_sha256'] for s in m['stages']]==[s['prompt_sha256'] for s in ref['stages']]
        hist=read(p.with_suffix('.replay.json'));assert len(hist)==9
        total=0
        for i,(h,n,dose) in enumerate(zip(hist,names,doses),1):
            count=dose*(64 if i==1 else 32);total+=count
            assert h['stage']==i and h['task']==n and h['current_presentations']==count
            assert h['presentations_total']==total
            assert h['replay_count']==(0 if i==1 else dose*32)
            assert m['stages'][i-1]['replay_targets_verified_against_prior_observations']==h['replay_count']
            assert m['stages'][i-1]['store_write_updates']==(0 if not is_a and i==9 else dose)
            assert sum(h['realized_draw_counts'].values())==h['replay_count']
            assert set(map(int,h['realized_draw_counts']))<=set(range(1,i))
            assert h['wrong_against_observed']==0
            expected_slots=min(total,job['capacity']) if is_a else min(i,8)*1024
            assert h['retained_slots']==expected_slots
            assert sum(h['origin_counts'].values())==expected_slots
            assert 0<h['distinct_prompts']<=expected_slots
    return d
