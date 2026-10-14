"""Technical development jobs ONLY. One ERROR life, three read-only policies.

Record-boundary checkpoints preserve all operative state and prior receipts.
Formal world IDs cannot be passed to this executable.
"""
import argparse,dataclasses,json,os,resource,signal,time
from pathlib import Path
import bridge_runtime as rt
from authority_policy import Observation,emit,POLICIES
from local_integrity import locked_identity,require
from independent_audit import audit_record,audit_job,flag_for
ROOT=Path(__file__).resolve().parent
DEVELOPMENT_WORLDS=tuple(range(62010001,62010009))
STOP_REQUESTED=False

def stop_signal(signum,frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True


def observations(predictions):
    return [Observation(tuple(p.shared),tuple(p.private)) for p in predictions]


def score_policies(predictions):
    obs=observations(predictions);out={};cost={}
    for policy in POLICIES:
        t=time.process_time();out[policy]=emit(obs,policy);cost[policy]=time.process_time()-t
    return out,cost


def record(core,event,branch):
    at=event['at'];flag=flag_for(branch,event['stage'])
    with rt.instrument(core) as tr:
        start=time.process_time()
        for i,b in enumerate(bytes.fromhex(event['cue_hex'])):core.feed(b,at+i*rt.DT)
        pred=core.predict(at+12*rt.DT)
        policy,cost=score_policies([pred])  # emit BEFORE the arriving teacher byte
        primary_emitted=policy['Q_EARNED']['emitted']
        write=core.observe_outcome(event['outcome'],at+12*rt.DT,learn=flag)
        core.feed(10,at+13*rt.DT);error=core.flush(at+rt.RECORD_SECONDS)
        native_cpu=time.process_time()-start-tr['audit_cpu_s']-sum(cost.values())
    row={'index':event['index'],'stage':event['stage'],'item':event['item'],'learn':flag,
         'prediction_precedes_outcome':True,'predicted_at':at+12*rt.DT,'observed_at':at+12*rt.DT,
         'prediction':dataclasses.asdict(pred),'write':write,'actual_calls':tr['calls'],
         'flush_error':float(error),'online_cpu_s':native_cpu,'external_audit_cpu_s':tr['audit_cpu_s'],
         'model_emitted':primary_emitted,'readouts':policy,'policy_cpu_s':cost,
         'operation_order':['cue','native_predict','policy_emit','outcome','newline','flush']}
    rt.check_record(row,'ERROR',event,branch,8)
    audit_record(row,event,branch)
    return row


class EarnedChoiceOrgan:
    """Serial sensory presentation and in-model comparison; no answer argument."""
    def __init__(self,core):self.core=core.clone()
    def choose(self,cues,at):
        preds=[]
        for k,cue in enumerate(cues):
            start=at+13*k*rt.DT
            for i,b in enumerate(cue):self.core.feed(b,start+i*rt.DT)
            preds.append(self.core.predict(start+12*rt.DT))
            self.core.clear_unreinforced_prediction()
            self.core.feed(ord('|') if k==0 else ord('?'),start+12*rt.DT);self.core.cue_count=0
        policy,cost=score_policies(preds)
        return preds,policy,cost


def probe(core,world,at,name):
    before=core.state_digest();rows=[];native_cpu=0.;cost={p:0. for p in POLICIES}
    for group in ('old','heldout','new','revision'):
        if group not in world['sets'] or (name.startswith('old_') and group in ('new','revision')):continue
        rs=world['sets'][group]
        if world['assay']=='reuse':
            for i in range(len(rs)//2):
                a,n=rs[2*i:2*i+2]
                for order in (0,1):
                    first,second=(a,n) if order==0 else (n,a)
                    organ=EarnedChoiceOrgan(core);begin=time.process_time()
                    preds,policy,pc=organ.choose((bytes.fromhex(first['cue_hex']),bytes.fromhex(second['cue_hex'])),at)
                    native_cpu+=time.process_time()-begin-sum(pc.values())
                    for p in POLICIES:cost[p]+=pc[p]
                    # Evaluator target is only consulted after the organ emitted.
                    target=ord('L') if order==0 else ord('R')
                    rows.append({'stage':group,'item':i,'order':order,'target':target,
                                 'predictions':[dataclasses.asdict(p) for p in preds],
                                 'readouts':policy,'cue_hex':[first['cue_hex'],second['cue_hex']],'first_pre_feedback':True})
        else:
            for i,r in enumerate(rs):
                clone=core.clone();begin=time.process_time()
                for j,b in enumerate(bytes.fromhex(r['cue_hex'])):clone.feed(b,at+j*rt.DT)
                pred=clone.predict(at+12*rt.DT);policy,pc=score_policies([pred])
                native_cpu+=time.process_time()-begin-sum(pc.values())
                for p in POLICIES:cost[p]+=pc[p]
                rows.append({'stage':group,'item':i,'target':r['outcome'],
                             'predictions':[dataclasses.asdict(pred)],'readouts':policy,'cue_hex':[r['cue_hex']],'first_pre_feedback':True})
    after=core.state_digest()
    if before!=after:raise AssertionError('probe changed continuing learner')
    return {'name':name,'at':at,'rows':rows,'state_before':before,'state_after':after,
            'online_api_cpu_s':native_cpu,'policy_cpu_s':cost}


def run(world_id,assay,folder,short=None,resume=False,stop_at=None):
    if world_id not in DEVELOPMENT_WORLDS:raise ValueError('formal science is not authorized by this runner')
    ident=locked_identity();w=rt.make_world(world_id,assay);folder=Path(folder)
    key=f'{world_id}_ERROR_{assay}';receipt=folder/'receipts'/f'{key}.json.gz';cp=folder/'checkpoints'/f'{key}.npz'
    if receipt.exists():
        audit_job(rt.read_receipt(receipt),ident)
        return {'status':'SKIPPED_COMMITTED','receipt':str(receipt)}
    limit=len(w['events']) if short is None else int(short)
    if not 0<limit<=len(w['events']):raise ValueError('record limit')
    for name in ('locks','active','checkpoints','receipts'): (folder/name).mkdir(parents=True,exist_ok=True)
    lock=folder/'locks'/f'{key}.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'key':key},f)
    t=time.monotonic();cpu=time.process_time();limit=len(w['events']) if short is None else int(short)
    if not 0<limit<=len(w['events']):raise ValueError('record limit')
    branches=w['branches'] if short is None else ['W'];active=folder/'active'/f'{key}.json'
    try:
        if resume:
            c,cur=rt.checkpoint.load(cp,'ERROR',ident)
            if (cur['world'],cur['assay'],cur['limit'],cur['branch_names'])!=(world_id,assay,limit,branches):raise ValueError('resume job mismatch')
        else:
            if cp.exists():raise ValueError('explicit resume required; checkpoint is not disposable')
            c=None;cur={'world':world_id,'assay':assay,'fixture_sha256':w['sha256'],'branch_names':branches,
                        'branch_index':0,'next_record':0,'limit':limit,'branches':{},'completed_cpu_s':0.,'completed_worker_s':0.}
        old_cpu,old_worker=cur['completed_cpu_s'],cur['completed_worker_s']
        for bi in range(cur['branch_index'],len(branches)):
            branch=branches[bi]
            if c is None:c=rt.Core('ERROR');cur['branch_index']=bi;cur['next_record']=0
            if branch not in cur['branches']:cur['branches'][branch]={'births':c.births,'records':[],'probes':[], 'peak_mutable_bytes':rt.mutable_bytes(c)}
            bd=cur['branches'][branch]
            for i in range(cur['next_record'],limit):
                bd['records'].append(record(c,w['events'][i],branch));cur['next_record']=i+1
                if short is None:
                    for name in w['boundaries'].get(str(i+1),[]):
                        c.flush(w['clocks'][name]);bd['probes'].append(probe(c,w,w['clocks'][name],name))
                paused=STOP_REQUESTED or (stop_at is not None and i+1==stop_at)
                if (i+1)%48==0 or paused or i+1==limit:
                    cur['completed_cpu_s']=old_cpu+time.process_time()-cpu;cur['completed_worker_s']=old_worker+time.monotonic()-t
                    rt.checkpoint.save(c,cp,cur,ident)
                rt.atomic_json(active,{'pid':os.getpid(),'world':world_id,'assay':assay,'branch':branch,
                                     'records':i+1,'limit':limit,'updated_unix':time.time(),'partial_only':True})
                if paused:
                    rt.atomic_json(folder/'SESSION_RESULT.json',{'status':'PAUSED_TECHNICAL_CHECKPOINT','key':key,'branch':branch,'next_record':i+1})
                    active.unlink(missing_ok=True)
                    return {'status':'PAUSED_TECHNICAL_CHECKPOINT','checkpoint':str(cp),'next_record':i+1}
            if short is not None:bd['probes'].append(probe(c,w,c.last_time,'short_end'))
            bd.update(records_completed=limit,final_state_digest=c.state_digest(),final_time=c.last_time,
                      online_cpu_s=sum(r['online_cpu_s'] for r in bd['records'])+sum(p['online_api_cpu_s'] for p in bd['probes'] if p['name']=='final'))
            bd['policy_cpu_s']={p:sum(r['policy_cpu_s'][p] for r in bd['records'])+sum(x['policy_cpu_s'][p] for x in bd['probes'] if x['name']=='final') for p in POLICIES}
            cur['branch_index']=bi+1;cur['next_record']=0;c=None
        require(ident)
        d={'schema':'EARNED_AUTHORITY_DEVELOPMENT_JOB_V1','development':True,'world':world_id,'arm':'ERROR','assay':assay,
           'source_identity':ident,'fixture_sha256':w['sha256'],'branches':cur['branches'],
           'limits':{'records':limit,'branches':branches},'complete':short is None,
           'cpu_s':old_cpu+time.process_time()-cpu,'worker_s':old_worker+time.monotonic()-t,
           'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        audit_job(d,ident);rt.write_receipt(receipt,d);active.unlink(missing_ok=True)
        return {'status':'COMMITTED_TECHNICAL','receipt':str(receipt),'lives':len(branches),'records':len(branches)*limit}
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--world',type=int,required=True);p.add_argument('--assay',choices=('lifetime','reuse','latin'),required=True)
    p.add_argument('--folder',required=True);p.add_argument('--short',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--stop-at',type=int)
    a=p.parse_args();signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    print(json.dumps(run(a.world,a.assay,a.folder,a.short,a.resume,a.stop_at)),flush=True)
