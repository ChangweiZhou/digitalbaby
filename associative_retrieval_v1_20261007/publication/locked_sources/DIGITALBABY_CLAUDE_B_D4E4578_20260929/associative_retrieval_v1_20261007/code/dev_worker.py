"""Only registered development worlds; committed receipts never rerun."""
import argparse
import copy
import dataclasses
import json
import os
import resource
import time
from pathlib import Path
import runtime
import numpy as np
from bridge_core import RetrievalCore, ChoiceOrgan
from association import code_ids
from assays import make_world, permitted, DT, RECORD_SECONDS
from auditor import instrument
from io_utils import atomic_json, canonical, write_receipt, read_receipt
from identity import identity, require
from independent_audit import audit
import bridge_checkpoint

DEVELOPMENT_WORLDS=(71008001,71008002,71008003)

def plain(x): return json.loads(canonical(x))

def record(c,e,branch):
    at=e['at']; parent_cpu=0.; started=time.process_time()
    with instrument(c) as tr:
        start=time.process_time()
        for i,b in enumerate(bytes.fromhex(e['cue_hex'])): c.feed(b,at+i*DT)
        parent_cpu+=time.process_time()-start
        p=c.predict(at+12*DT)
        parent_cpu+=p.costs['parent_prediction_cpu_s']
        observed=[]
        for i,m in enumerate(c.models):
            name='teach_logged' if i<4 else 'teach_signed'; original=getattr(m,name)
            def capture(*args,_m=m,_original=original,_i=i,**kw):
                observed.append((_i,code_ids(_m.pending_x,_m.n_native_kc).tolist()))
                return _original(*args,**kw)
            setattr(m,name,capture)
        flag=permitted(branch,e['stage'])
        w=c.observe_outcome(e['outcome'],at+12*DT,learn=flag)
        parent_cpu+=c.last_costs['parent_outcome_cpu_s']
        start=time.process_time();c.feed(10,at+13*DT);error=c.flush(at+RECORD_SECONDS)
        parent_cpu+=time.process_time()-start
        calls=copy.deepcopy(tr['calls'])
        if [i for i,_ in observed]!=list(range(8)): raise AssertionError('missing actual address observation')
        for call,(_,ids) in zip(calls,observed): call['address_ids']=ids
    total=time.process_time()-started-tr['audit_cpu_s']
    row={'index':e['index'],'stage':e['stage'],'item':e['item'],'learn':flag,
         'prediction_precedes_outcome':True,'predicted_at':at+12*DT,'observed_at':at+12*DT,
         'prediction':plain(dataclasses.asdict(p)),'write':w,'actual_calls':calls,
         'flush_error':float(error),'parent_online_cpu_s':max(0.,parent_cpu-tr['audit_cpu_s']),
         'LINK_online_cpu_s':total-p.costs['control_retrieval_cpu_s'],
         'PERM_online_cpu_s':total-p.costs['retrieval_cpu_s'],
         'physical_online_cpu_s':total,'external_audit_cpu_s':tr['audit_cpu_s']}
    if error>=1e-8: raise AssertionError('native flush mismatch')
    return row

def probe(c,w,at,name,after_record):
    before=c.state_digest(); rows=[]; started=time.process_time()
    for group in ('old','heldout','new','revision'):
        if group not in w['sets'] or (name.startswith('old_') and group in ('new','revision')): continue
        rs=w['sets'][group]
        if w['assay']=='reuse':
            for i in range(len(rs)//2):
                a,n=rs[2*i:2*i+2]
                if (a['outcome'],n['outcome'])!=(49,48): raise AssertionError('relation source orientation')
                for order in (0,1):
                    first,second=(a,n) if order==0 else (n,a)
                    policies,ps=ChoiceOrgan(c).choose(bytes.fromhex(first['cue_hex']),bytes.fromhex(second['cue_hex']),at,DT)
                    target=ord('L') if order==0 else ord('R')
                    for v in policies.values(): v['correct']=int(v['emitted']==target)
                    rows.append({'stage':group,'pair':i,'order':order,'target':target,
                                 'predictions':plain(ps),'policies':policies,'first_pre_feedback':True})
        else:
            for i,r in enumerate(rs):
                clone=c.clone()
                for j,b in enumerate(bytes.fromhex(r['cue_hex'])): clone.feed(b,at+j*DT)
                p=clone.predict(at+12*DT); policies=copy.deepcopy(p.policies)
                for v in policies.values(): v['correct']=int(v['emitted']==r['outcome'])
                rows.append({'stage':group,'item':i,'target':r['outcome'],
                             'predictions':[plain(dataclasses.asdict(p))],'policies':policies,'first_pre_feedback':True})
    after=c.state_digest()
    if before!=after: raise AssertionError('probe mutated continuing native or association state')
    return {'name':name,'at':at,'after_record':after_record,'rows':rows,
            'state_before':before,'state_after':after,'physical_probe_cpu_s':time.process_time()-started}

def run_branch(world,assay,branch,folder,*,limit=None,stop=None,resume=False):
    if world not in DEVELOPMENT_WORLDS or assay not in ('lifetime','reuse'): raise ValueError('development scope only')
    w=make_world(world,assay)
    if branch not in w['branches']: raise ValueError('unknown branch')
    full=limit is None; limit=len(w['events']) if full else limit
    if type(limit) is not int or not 0<limit<=len(w['events']): raise ValueError('record limit')
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    key=f'{world}_{assay}_{branch}'; receipt=folder/f'{key}.json.gz'; cp=folder/f'{key}.npz'
    ident=identity()
    if receipt.exists():
        d=read_receipt(receipt); audit(d,ident); return {'status':'SKIPPED_COMMITTED','path':str(receipt)}
    lock=folder/f'{key}.lock'; fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    start=time.monotonic(); cpu=time.process_time()
    try:
        if resume:
            c,cur=bridge_checkpoint.load(cp,ident)
            if (cur['world'],cur['assay'],cur['branch'],cur['limit'])!=(world,assay,branch,limit): raise ValueError('resume job mismatch')
        else:
            if cp.exists(): raise ValueError('explicit resume required')
            c=RetrievalCore();cur={'world':world,'assay':assay,'branch':branch,'limit':limit,
                                  'fixture_sha256':w['sha256'],'next_record':0,'records':[],'probes':[],
                                  'births':c.births,'prior_cpu_s':0.,'prior_worker_s':0.}
        prior_cpu,prior_wall=cur['prior_cpu_s'],cur['prior_worker_s']
        for i in range(cur['next_record'],limit):
            cur['records'].append(record(c,w['events'][i],branch));cur['next_record']=i+1
            if full:
                for name in w['boundaries'].get(str(i+1),[]):
                    c.flush(w['clocks'][name]);cur['probes'].append(probe(c,w,w['clocks'][name],name,i+1))
            cur['prior_cpu_s']=prior_cpu+time.process_time()-cpu;cur['prior_worker_s']=prior_wall+time.monotonic()-start
            if (i+1)%48==0 or stop==i+1 or i+1==limit:
                bridge_checkpoint.save(c,cp,cur,ident)
                atomic_json(folder/f'{key}.active.json',{'pid':os.getpid(),'world':world,'assay':assay,'branch':branch,
                    'committed_records_checkpoint':i+1,'total_records':limit,'updated_unix':time.time()})
            if stop==i+1: return {'status':'CHECKPOINT_STOP','path':str(cp),'records':i+1}
        if not full:cur['probes'].append(probe(c,w,c.last_time,'short_end',limit))
        bd={'births':cur['births'],'records':cur['records'],'probes':cur['probes'],
            'final_state_digest':c.state_digest(),'native_final_digest':c.native_digest(),
            'association_final_digest':c.associations.digest(),'final_time':c.last_time,
            'association_mutable_bytes':c.associations.mutable_bytes(),'fixed_permutation_bytes':c.permutation.nbytes}
        doc={'schema':'ASSOCIATIVE_DEVELOPMENT_V1','development':True,'world':world,'assay':assay,
             'fixture_sha256':w['sha256'],'source_identity':ident,'limit':limit,'complete':False,
             'complete_branch':full,'branches':{branch:bd},'cpu_s':prior_cpu+time.process_time()-cpu,
             'worker_s':prior_wall+time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        audit(doc,ident); require(ident);write_receipt(receipt,doc)
        (folder/f'{key}.active.json').unlink(missing_ok=True)
        return {'status':'COMMITTED','path':str(receipt),'records':limit,'native_lives':int(full)}
    finally:lock.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--world',type=int,required=True);p.add_argument('--assay',required=True,choices=('lifetime','reuse'))
    p.add_argument('--branch');p.add_argument('--folder',required=True);p.add_argument('--limit',type=int);p.add_argument('--stop',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args()
    w=make_world(a.world,a.assay);bs=[a.branch] if a.branch else w['branches']
    if (a.stop or a.resume) and not a.branch: raise ValueError('restart fixture needs explicit branch')
    for b in bs:
        result=run_branch(a.world,a.assay,b,a.folder,limit=a.limit,stop=a.stop,resume=a.resume)
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()

