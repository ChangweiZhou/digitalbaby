"""Technical preparation runner; cannot submit or classify formal science."""
import argparse, collections, gzip, hashlib, json, time
from pathlib import Path
import torch
from core import Learner, configuration
from worlds import make_world, schedule, shuffle_targets, probes

ARMS=('G0','G1','G4','R1','R8','RU8','C','P','P_STOP','P_RESET')
def model_seed(world,arm):
    family='G' if arm.startswith('G') else 'R' if arm.startswith('R') else 'P' if arm.startswith('P') else 'C'
    return int.from_bytes(hashlib.sha256(f'KERNEL_V3|{world}|{family}'.encode()).digest()[:8],'big')%(2**63)

def score(core,rows,label):
    before=core.digest();out=[]
    for r in rows:
        answer,status=core.generate(bytes.fromhex(r['prompt']))
        out.append({'prompt':r['prompt'],'expected':r['reply'],'operation':r['operation'],'group':r['group'],
                    'actual':answer.hex(),'status':status,'exact':answer.hex()==r['reply'] and status=='EOS'})
    after=core.digest();assert before==after,'probe changed learning state'
    return {'label':label,'before':before,'after':after,'rows':out}

def train(core,records,allowed,label):
    before=core.digest();counts_before=dict(core.resource);ledger=[]
    for i,r in enumerate(records):
        rows=core.observe(bytes.fromhex(r['prompt']),bytes.fromhex(r['reply']),allowed)
        ledger.append({'index':i,'prompt':r['prompt'],'reply':r['reply'],'allowed':allowed,'predictions':rows})
    if allowed:core.flush()
    after=core.digest()
    if not allowed:assert before==after,'no-write changed learning state'
    return {'label':label,'allowed':allowed,'before':before,'after':after,'counts_before':counts_before,'counts_after':dict(core.resource),'records':ledger}

def job(world_id,arm,binary_n,mixed_n):
    start=time.process_time();wall=time.monotonic();world=make_world(world_id)
    records=schedule(world,'binary',binary_n);mixed=schedule(world,'mixed',mixed_n)
    shuffled=shuffle_targets(records,world_id+19)
    birth=Learner(arm,model_seed(world_id,arm));initial=birth.digest();prefixes={};receipt={
      'schema':'KERNEL_V3_TECHNICAL_JOB','technical':True,'world':world_id,'arm':arm,'binary_n':binary_n,'mixed_n':mixed_n,
      'data_digest':world['data_digest'],'source':source_identity(),'framework':torch.__version__,
      'parameters':sum(p.numel() for p in birth.net.parameters()),'initial_digest':initial,'branches':{},'completed':False}
    for name,stream,allowed in [('W',records,True),('N_BINARY',records,False),('SH_BINARY',shuffled,True)]:
        c=birth.clone();assert c.digest()==initial
        seg=train(c,stream,allowed,'binary');taught=probes(stream);held=probes(world['binary_held'])
        before=[score(c,taught,'old_taught'),score(c,held,'old_held')]
        prefixes[name]=c;receipt['branches'][name]={'segments':[seg],'before_mixed':before}
    w_prefix=prefixes['W'].clone()
    for name in ('W','N_BINARY','SH_BINARY','N_MIXED'):
        c=w_prefix.clone() if name=='N_MIXED' else prefixes[name]
        seg=train(c,mixed,name!='N_MIXED','mixed')
        branch=receipt['branches'].setdefault(name,{'segments':[]});branch['segments'].append(seg)
        branch['final']=[score(c,probes(records),'old_taught'),score(c,probes(world['binary_held']),'old_held'),
                         score(c,probes(mixed),'new_taught'),score(c,probes(world['mixed_held']),'new_held')]
        branch['resource']=dict(c.resource);branch['state_bytes']=c.state_bytes()
        branch['last_gradient']=dict(c.last_gradient);branch['last_update']=dict(c.last_update)
    receipt['executed_external_records']=3*binary_n+4*mixed_n
    receipt['correlated_final_histories']=4
    receipt['total_process_cpu']=time.process_time()-start;receipt['total_wall']=time.monotonic()-wall
    receipt['completed']=True
    return receipt

def source_identity():
    return {f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in ('core.py','worlds.py','runner.py','auditor.py')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--cycle',type=int,required=True);p.add_argument('--world',type=int,required=True)
    p.add_argument('--arms',nargs='+',default=list(ARMS));p.add_argument('--binary',type=int,required=True);p.add_argument('--mixed',type=int,required=True)
    a=p.parse_args();out=Path(__file__).parent/f'cycles/cycle{a.cycle}/receipts';out.mkdir(parents=True,exist_ok=True)
    from auditor import audit
    for arm in a.arms:
        target=out/f'{a.world}_{arm}.json.gz'
        if target.exists():raise RuntimeError('technical receipt already exists; do not silently repeat')
        r=job(a.world,arm,a.binary,a.mixed);certificate=audit(r)
        with gzip.open(target,'wt') as f:json.dump(r,f)
        target.with_suffix('.audit.json').write_text(json.dumps(certificate,indent=2)+'\n')
        print(json.dumps({'cycle':a.cycle,'world':a.world,'arm':arm,'records':r['executed_external_records'],'cpu':round(r['total_process_cpu'],2),'audit':certificate['accepted']}),flush=True)
if __name__=='__main__':main()
