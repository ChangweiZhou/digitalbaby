"""Locked native jobs; answer grading lives here, never in core.py."""
import argparse,collections,gzip,hashlib,json,os,resource,sys,time
from pathlib import Path
from core import Learner,MODELS
from protocol import ROOT,D,seed,show,plans,probes,atomic,identity

class Shortcuts:
    def __init__(self,feedback):
        self.tables={k:collections.defaultdict(collections.Counter) for k in ('exact','canonical','length','suffix','bag')}
        self.glob=collections.Counter()
        for r in feedback:
            answer=r['observed']
            if answer is None and r['reward']==1:answer=r['actual']
            if answer is None:continue
            p=bytes.fromhex(r['prompt']);keys=(r['prompt'],r['group'],len(p),p[-2:],tuple(sorted(p)))
            self.glob[answer]+=1
            for k,x in zip(self.tables,keys):self.tables[k][x][answer]+=1
    def predict(self,r):
        mode=lambda c:min(c,key=lambda a:(-c[a],a)) if c else None
        p=bytes.fromhex(r['prompt']);keys=(r['prompt'],r['group'],len(p),p[-2:],tuple(sorted(p)))
        a={k:mode(self.tables[k].get(x,{})) for k,x in zip(self.tables,keys)};a['global']=mode(self.glob)
        a['length']=a['length'] or a['global'];a['suffix']=a['suffix'] or a['global'];a['bag']=a['bag'] or a['length'];return a

def checkpoint(c,n,w,feedback,taught=False):
    before=c.digest();counter=Shortcuts(feedback);cpu_before=c.resource['production_cpu']
    exposed={r['prompt']:dict(prompt=r['prompt'],reply=r['expected'],operation=r['operation'],group=r['group']) for r in feedback}
    rows=[(label,probes([r for r in exposed.values() if len(r['operation'])==(1 if label=='binary' else 2)],None) if taught else probes(w[label+'_held'],None)) for label in ('binary','mixed')]
    scores={}
    for label,rs in rows:
        answers=c.generate_batch([bytes.fromhex(x['prompt']) for x in rs]);scores[label]=[]
        for x,(answer,status) in zip(rs,answers):
            scores[label].append(dict(x,actual=answer.hex(),status=status,exact=answer.hex()==x['reply'] and status=='EOS',shortcuts=counter.predict(x)))
    assert before==c.digest()
    if taught:
        c.resource['diagnostic_cpu']=c.resource.get('diagnostic_cpu',0.)+c.resource['production_cpu']-cpu_before
        c.resource['production_cpu']=cpu_before
    return dict(n=n,scores=scores,before=before,after=c.digest(),cpu=c.resource['production_cpu'],resource=dict(c.resource))

def segment(c,rows,phase,world,offset,kind,allowed,feedback,hb,checks,w):
    before=c.digest();before_resources=dict(c.resource);ledger=[];curve=[]
    for i,r in enumerate(rows):
        index=offset+i
        observable=(r['reply'] if kind=='DENSE' or show(world,index,phase) else None)
        answer,status,cache=c.act(bytes.fromhex(r['prompt']),seed(world,index),credit=allowed and phase!='WARM')
        reward=int(answer.hex()==r['truth'] and status=='EOS')
        learning_mode='DENSE' if kind=='DENSE' and phase!='WARM' else phase
        update=c.update(bytes.fromhex(r['prompt']),None if observable is None else bytes.fromhex(observable),reward,cache,learning_mode,allowed)
        row=dict(index=index,prompt=r['prompt'],expected=r['truth'],observed=observable,operation=r['operation'],group=r['group'],
            actual=answer.hex(),status=status,reward=reward,actions=cache['actions'],baseline=float(cache['baseline']),allowed=allowed,learning_mode=learning_mode,update=update)
        ledger.append(row);feedback.append(row)
        if i+1 in checks:curve.append(checkpoint(c,index+1,w,feedback))
        if i%64==0 or i+1==len(rows):hb(dict(phase=phase,branch=kind,completed_records=i+1,segment_records=len(rows),partial=True))
    after=c.digest()
    if not allowed:assert before==after,'complete no-write state changed'
    return dict(phase=phase,kind=kind,offset=offset,allowed=allowed,before=before,after=after,
        resources_before=before_resources,resources_after=dict(c.resource),records=ledger,curve=curve)

def job(world,model,nb,nm,stages,out,hb):
    start=time.process_time();wall=time.monotonic();w,warm,shuffled,tails=plans(world,nb,nm,stages)
    for x,t in zip(warm,shuffled):x['truth']=x['reply'];t['truth']=x['reply']
    for rows in tails.values():
        for x in rows:x['truth']=x['reply']
    born=Learner(model,seed(world,0,'model'));initial=born.digest()
    states=out/'states'/f'{world}_{model}';states.mkdir(parents=True,exist_ok=True);born.save(states/'BIRTH.npz')
    r=dict(schema='MODEL_FEEDBACK_JOB_V1',world=world,model=model,binary=nb,mixed=nm,stages=stages,
        data_digest=w['data_digest'],source=identity(),initial=initial,model_seed=seed(world,0,'model'),prefixes={},branches={},archives={},completed=False)
    r['parameters']={k:p.numel() for k,p in born.net.p.items()};r['learning_bytes']=sum(p.numel()*p.element_size() for p in born.net.p.values())*2
    parents={};fw=nb+nm
    for name,stream,allowed in [('TRUE',warm,True),('NO_WARM',warm,False),('SH_WARM',shuffled,True)]:
        c=born.clone();feedback=[];birth_curve=checkpoint(c,0,w,feedback)
        seg=segment(c,stream,'WARM',world,0,name,allowed,feedback,hb,set(fw*i//4 for i in range(1,5)),w)
        c.save(states/(name+'_WARM.npz'));r['prefixes'][name]=dict(segment=seg,curve=[birth_curve]+seg['curve'],
            taught=checkpoint(c,fw,w,feedback,True),state=c.digest())
        parents[name]=(c,feedback)
    scalar_parent=None
    for name,parent in [('FADING','TRUE'),('DENSE','TRUE'),('NO_WARM','NO_WARM'),('SH_WARM','SH_WARM')]:
        c=parents[parent][0].clone();feedback=list(parents[parent][1]);segments=[];curve=list(r['prefixes'][parent]['curve']);offset=fw
        for phase,n in zip(('HALF','SPARSE','SCALAR'),stages):
            if phase=='SCALAR':
                c.save(states/(name+'_SCALAR_START.npz'))
                if name=='FADING':scalar_parent=(c.clone(),list(feedback),list(curve))
            checks={n} if phase!='SCALAR' else {n//2,n}
            seg=segment(c,tails[phase],phase,world,offset,name,True,feedback,hb,checks,w)
            segments.append(seg);curve.extend(seg['curve']);offset+=n
        c.save(states/(name+'_FINAL.npz'))
        r['branches'][name]=dict(prefix=parent,segments=segments,curve=curve,final=c.digest(),resource=dict(c.resource),
            taught=checkpoint(c,offset,w,feedback,True),last=dict(c.last))
    c,feedback,curve=scalar_parent;scalar_before=c.digest();offset=fw+sum(stages[:2]);n=stages[-1]
    seg=segment(c,tails['SCALAR'],'SCALAR',world,offset,'FREEZE_SCALAR',False,feedback,hb,{n//2,n},w)
    curve.extend(seg['curve']);c.save(states/'FREEZE_SCALAR_FINAL.npz')
    r['branches']['FREEZE_SCALAR']=dict(prefix='TRUE',curve_parent='FADING_BEFORE_SCALAR',segments=[seg],curve=curve,
        fork_state=scalar_before,final=c.digest(),resource=dict(c.resource),taught=checkpoint(c,fw+sum(stages),w,feedback,True),last=dict(c.last))
    assert scalar_before==r['branches']['FADING']['segments'][-1]['before']==c.digest()
    r['executed_records']=3*fw+4*sum(stages[:2])+5*stages[-1]
    r['correlated_final_histories']=5;r['total_cpu']=time.process_time()-start;r['wall']=time.monotonic()-wall
    r['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
    r['archives']={p.name:str(p.relative_to(ROOT)) for p in states.glob('*.npz')};r['completed']=True
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('cycle2','cycle3','science'),required=True);p.add_argument('--world',type=int,required=True);p.add_argument('--model',choices=MODELS,required=True);a=p.parse_args()
    if a.stage=='science':
        admit=json.loads((ROOT/'SCIENCE_ADMISSION.json').read_text());assert admit['accepted'] is True
        assert a.world in D['science_worlds'];nb,nm,stages=D['science_binary'],D['science_mixed'],D['science_stages']
        assert identity()==admit['source'];assert (ROOT/'AUTHORIZATION.json').is_file()
    else:
        config=D['cycles'][a.stage[-1]];assert a.world in config['worlds'];nb,nm,stages=config['binary'],config['mixed'],config['stages']
    out=ROOT/('results' if a.stage=='science' else 'cycles/'+a.stage);target=out/'receipts'/f'{a.world}_{a.model}.json.gz'
    if target.exists():raise RuntimeError('committed job exists; no repeat')
    if not target.parent.exists():target.parent.mkdir(parents=True,exist_ok=True)
    active=out/'active'/f'{a.world}_{a.model}.json'
    def hb(x):atomic(active,dict(x,pid=os.getpid(),world=a.world,model=a.model,heartbeat=time.time()))
    r=job(a.world,a.model,nb,nm,stages,out,hb)
    from auditor import audit
    audit_start=time.process_time();certificate=audit(r,identity());certificate['audit_cpu']=time.process_time()-audit_start
    temp=target.with_name(target.name+'.pending')
    with gzip.open(temp,'wt') as f:json.dump(r,f,allow_nan=False)
    temp.replace(target);atomic(target.with_suffix('.audit.json'),certificate)
    atomic(active,dict(committed=True,world=a.world,model=a.model,pid=os.getpid(),heartbeat=time.time(),records=r['executed_records']))
    print(json.dumps(dict(world=a.world,model=a.model,records=r['executed_records'],cpu=r['total_cpu'],wall=r['wall'],audit_cpu=certificate['audit_cpu'])),flush=True)
if __name__=='__main__':main()
