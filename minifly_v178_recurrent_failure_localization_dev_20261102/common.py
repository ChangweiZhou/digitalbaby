"""V178 historical-source import and sequential resource ledger."""
import os,sys,time,json,hashlib,resource
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.path.insert(0,str(H/'local/compat'))
import numpy as np
import scipy
import torch
torch.set_num_threads(1)
sys.path.insert(0,str(H/'local/source/minifly_v164_cola_dev_v1_20261026'))
import v164_lib as L
AUTH=json.loads((H/'AUTHORIZATION.json').read_text());GAIN=3.1475108458573424
def save(n,obj):
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cpu():r=resource.getrusage(resource.RUSAGE_SELF);return r.ru_utime+r.ru_stime
def budget():
 if cpu()+AUTH.get('preflight_CPU_allowance',120)>AUTH['CPU_cap']:raise RuntimeError('CPU_CAP')
 if time.time()-AUTH['epoch']>AUTH['wall_cap']:raise RuntimeError('WALL_CAP')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>AUTH['RSS_cap']:raise RuntimeError('RSS_CAP')
 if sum(p.stat().st_size for p in H.rglob('*') if p.is_file())>AUTH['output_cap']:raise RuntimeError('OUTPUT_CAP')
def charge(n):
 p=H/'COUNTS.json';d=json.loads(p.read_text()) if p.exists() else dict(training_records=0,scored_target_applications=0)
 d['training_records']+=n;d['scored_target_applications']+=n*112;save('COUNTS.json',d)
 if d['training_records']>5100:raise RuntimeError('RECORD_CAP')
 budget()
def splits(guard=False):
 if not guard:return {k:L.synth_split('S-COPY',k) for k in ('train','calib','test')}
 out={}
 for k,n in [('train',100),('calib',20),('test',50)]:
  seed=int.from_bytes(hashlib.sha256(f'V178|202611027811|{k}|body'.encode()).digest()[:16],'big');g=np.random.default_rng(seed);x=g.integers(0,8,(n,120));y=np.full_like(x,-1);y[:,8:]=x[:,:112];out[k]=dict(idx=torch.from_numpy(x.reshape(-1,1)),y=y.ravel(),n=np.full(n,120))
 return out
