"""Bounded dispatcher primitives. Importing this module never launches science."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from core import CONFIGS
from worlds import SCI_MIN, SCI_MAX
from runner import source_lock


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('w') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def manifest():
    return [{'world':w,'config':c,'job_id':f'{w}_{c}'}
            for w in range(SCI_MIN,SCI_MAX+1) for c in CONFIGS]


def may_admit(elapsed,estimated_job_seconds):
    if elapsed<0 or estimated_job_seconds<=0:
        raise ValueError('invalid operational budget')
    # Explicit stop at eight hours; predicted whole job must fit before hard ten.
    return elapsed<8*3600 and elapsed+3*estimated_job_seconds<10*3600


def committed_index(directory,expected,technical=False,validate=False):
    seen=set();rows=[]
    for path in sorted(Path(directory).glob('*.json.gz')):
        with gzip.open(path,'rt') as f:r=json.load(f)
        key=(r['world'],r['config'])
        if key in seen or key not in expected or r['technical']!=technical:
            raise ValueError('duplicate/unplanned/misclassified committed receipt')
        cert_path=Path(directory).parent/'certificates'/f'{r["world"]}_{r["config"]}.json'
        cert=json.loads(cert_path.read_text())
        if not cert.get('accepted') or cert.get('receipt_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('missing/mismatched accepted certificate')
        if cert.get('source_lock')!=r['source_lock'] or (validate and cert.get('source_lock')!=source_lock()):
            raise ValueError('certificate/source mismatch')
        seen.add(key)
        rows.append({'world':r['world'],'config':r['config'],
                     'records':r['records_applied'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'committed_jobs':len(rows),'committed_histories':3*len(rows),
            'committed_records':sum(r['records'] for r in rows),'rows':rows}


def require_authorization(path):
    auth=json.loads(Path(path).read_text())
    if auth.get('science_authorized') is not True or auth.get('source_lock')!=source_lock():
        raise PermissionError('authorization/frozen source mismatch')
    return auth
