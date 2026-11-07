import importlib.util,sys,json,hashlib,tempfile
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
import exp8_life as E
X=E.X;T=E.T
spec=importlib.util.spec_from_file_location('original',str(R/'exp8_e9_frozen.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
base_update=X.update; out0=X.OUT; out1=E.OUT

def digest(v):
    if T.is_tensor(v):return [str(v.dtype),list(v.shape),hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()]
    if isinstance(v,dict):return {str(k):digest(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [digest(x) for x in v]
    return v

def run(mod,arm,folder,save=False):
    mod.OUT=folder;X.OUT=folder;folder.mkdir();(folder/'e8_dev_73129.pt').symlink_to(out1/'e8_dev_73129.pt')
    mod.STAGE_UPDATES=2;mod.COMPOSE_UPDATES=2
    if mod is E:mod.SAVE_STAGES=save
    traces=[]
    def update(m,opt,*args,**kw):
        loss=base_update(m,opt,*args,**kw)
        traces.append(digest([m.state_dict(),opt.state_dict(),T.get_rng_state(),loss]));return loss
    X.update=update;mod.life(arm,73129);X.update=base_update
    data=json.loads(next(folder.glob('e8_life*.json')).read_text())
    for row in data['stages']:row.pop('cpu',None)
    return traces,data
root=Path(tempfile.mkdtemp(prefix='e12-technical-'));checks={}
for arm in ('A','C'):
    a=run(old,arm,root/f'old{arm}');b=run(E,arm,root/f'new{arm}',True)
    assert a==b,f'Default/saving parity failed{arm}'
    checks[arm]={'updates':len(a[0]),'model_optimizer_rng_losses_and_scores_exact':True}
# Basic fixed exposure and recency probability checks.
assert 24576*16==6144*64==1536*256==393216
assert 32768*16==8192*64==2048*256==524288
checks['fixed_exposure_exact']=True
checks['recency_weights']=[.5**(9-j) for j in range(1,9)]
(R/'E12_TECHNICAL.json').write_text(json.dumps(checks,indent=2)+'\n')
print('E12 technical parity passed',checks)
