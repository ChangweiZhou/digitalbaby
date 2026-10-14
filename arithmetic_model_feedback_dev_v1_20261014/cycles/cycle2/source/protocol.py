"""External stream/feedback protocol. Not imported by the learner."""
import hashlib,json,random,os
from pathlib import Path
from worlds import make_world,schedule,shuffle_targets,probes
ROOT=Path(__file__).resolve().parent
D=json.loads((ROOT/'DESIGN.json').read_text())

def seed(world,index,kind='action'):
    return int.from_bytes(hashlib.sha256(f'MF|{world}|{index}|{kind}'.encode()).digest()[:8],'big')%(2**63)
def show(world,index,phase):
    p={'WARM':1.,'HALF':.5,'SPARSE':.125,'SCALAR':0.}[phase]
    return random.Random(seed(world,index,'visibility')).random()<p
def plans(world,nb,nm,stages):
    w=make_world(world);b=schedule(w,'binary',nb);m=schedule(w,'mixed',nm)
    first=nb-3*nm;assert first>=0 and first%4==0
    warm=b[:first];j=first
    for r in m:warm.extend(b[j:j+3]);warm.append(r);j+=3
    assert len(warm)==nb+nm
    bt=schedule(w,'binary',sum(stages)//2);mt=schedule(w,'mixed',sum(stages)//2)
    tail=[]
    for x,y in zip(bt,mt):tail.extend((x,y))
    ends=[];pos=0
    for n in stages:ends.append(tail[pos:pos+n]);pos+=n
    return w,warm,shuffle_targets(warm,world+19),dict(zip(('HALF','SPARSE','SCALAR'),ends))
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.pending-{os.getpid()}');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def identity():
    local=('core.py','protocol.py','worlds.py','runner.py','auditor.py','analysis.py','tests.py','DESIGN.json','EXPERIMENT.md','supervisor.py')
    files={x:hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in local}
    for f in ('arithmetic_source_derived_kernel_dev_v1_20261013/core.py','arithmetic_source_derived_kernel_dev_v1_20261013/auditor.py','arithmetic_end_to_end_design_v1_20261013/validate_design.py'):
        files['external:'+f]=hashlib.sha256((ROOT.parent/f).read_bytes()).hexdigest()
    return files
