"""Qualification jobs only. No science dispatch interface exists in this release."""
import argparse,contextlib,dataclasses,json,os,resource,time
MODULE_CPU=time.process_time()
MODULE_WALL=time.monotonic()
from pathlib import Path
import bootstrap
from core import Core,ChoiceOrgan,policies
from assays import make_world,permitted,DT,RECORD_SECONDS
from spec import DEVELOPMENT_WORLDS,DOSES,LATIN_ARMS
from mechanisms import sha,relation_q
from auditor import instrument,check_record,audit_receipt
from io_utils import digest,atomic_json,write_receipt,read_receipt
from integrity import identity,require
import checkpoint

def record(c,e,branch):
    at=e['at']
    with instrument(c) as tr:
        begin=time.process_time()
        for i,b in enumerate(bytes.fromhex(e['cue_hex'])):c.feed(b,at+i*DT)
        p=c.predict(at+12*DT)
        flag=permitted(branch,e['stage']);w=c.observe_outcome(e['outcome'],at+12*DT,learn=flag)
        c.feed(10,at+13*DT);err=c.flush(at+RECORD_SECONDS)
        online=time.process_time()-begin-tr['audit_cpu_s']
    d={'index':e['index'],'stage':e['stage'],'item':e['item'],'learn':flag,
       'prediction_precedes_outcome':True,'predicted_at':at+12*DT,'observed_at':at+12*DT,
       'prediction':dataclasses.asdict(p),'policies':policies(p),'write':w,
       'actual_calls':tr['calls'],'flush_error':float(err),'online_cpu_s':online,'external_audit_cpu_s':tr['audit_cpu_s']}
    check_record(d,c.arm,e,branch,len(c.models));return d

def probe(c,w,at,name,branch):
    before=c.state_digest();rows=[];online=0.;half_cpu=0.;beg=time.process_time()
    for group in ('old','heldout','new','revision'):
        if group not in w['sets'] or (name.startswith('old_') and group in ('new','revision')):continue
        rs=w['sets'][group]
        if w['assay']=='reuse':
            for i in range(len(rs)//2):
                a,n=rs[2*i:2*i+2];assert (a['outcome'],n['outcome'])==(49,48)
                for order in (0,1):
                    first,second=(a,n) if order==0 else (n,a);o=ChoiceOrgan(c)
                    start=time.process_time();emit,vs,ps,half=o.choose(bytes.fromhex(first['cue_hex']),bytes.fromhex(second['cue_hex']),at,DT);online+=time.process_time()-start-o.half_cpu_s;half_cpu+=o.half_cpu_s
                    target=ord('L') if order==0 else ord('R')
                    rows.append({'stage':group,'pair':i,'order':order,'target':target,'emitted':emit,
                                 'correct':int(emit==target),'values':vs,'predictions':[dataclasses.asdict(p) for p in ps],
                                 'Q_HALF_emitted':half,'Q_HALF_correct':int(half==target),'first_pre_feedback':True})
        else:
            for i,r in enumerate(rs):
                clone=c.clone();start=time.process_time()
                for j,b in enumerate(bytes.fromhex(r['cue_hex'])):clone.feed(b,at+j*DT)
                p=clone.predict(at+12*DT);online+=time.process_time()-start
                extra_start=time.process_time();policy=policies(p);half_cpu+=time.process_time()-extra_start
                row={'stage':group,'item':i,'target':r['outcome'],'emitted':p.emitted,'correct':int(p.emitted==r['outcome']),
                     'prediction':dataclasses.asdict(p),'policies':policy,'first_pre_feedback':True}
                if name=='final' or (w['assay']=='latin' and c.arm=='CENTER' and branch=='W' and name=='old_end'):
                    fe=clone.shared[0].fe.clone();fe.advance(at+12*DT);h=fe.read()
                    row['sidecar']={'h':h.tolist(),'h_sha256':sha(h),'q_sha256':sha(relation_q(h))}
                rows.append(row)
    after=c.state_digest()
    if before!=after:raise AssertionError('probe changed continuing state')
    return {'name':name,'at':at,'rows':rows,'state_before':before,'state_after':after,
            'online_api_cpu_s':online,'Q_HALF_additional_readout_cpu_s':half_cpu,'all_probe_cpu_s':time.process_time()-beg}

def mutable_bytes(c):
    total=0
    for m in c.models:
        total+=sum(getattr(m.fly.m,k).nbytes for k in ('fast','slow','adapt'))
        total+=m.fe.p.nbytes+m.w.nbytes+m.bias.nbytes
        for k in ('on','epos','sw','U','A','cue_h','cue_x','last_h','last_q','last_x','pending_x'):
            a=getattr(m,k,None)
            if a is not None:total+=a.nbytes
        if hasattr(m,'graph_hash'):total+=m.fly.m.B.data.nbytes+m.fly.m.B.indices.nbytes+m.fly.m.B.indptr.nbytes
    return total

def run(world,arm,assay,folder,*,short=None,stop=None,resume=False):
    if world not in DEVELOPMENT_WORLDS:raise ValueError('this runner is restricted to registered development worlds')
    if assay=='latin' and arm not in LATIN_ARMS:raise ValueError('unregistered Latin configuration')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    key=f'{world}_{arm}_{assay}';path=folder/'receipts'/f'{key}.json.gz';cp=folder/'checkpoints'/f'{key}.npz'
    ident=identity();w=make_world(world,assay)
    limit=len(w['events']) if short is None else int(short)
    if not 0<limit<=len(w['events']):raise ValueError('explicit short-fixture limit')
    branches=w['branches'] if short is None else ['W']
    if path.exists():
        d=read_receipt(path);audit_receipt(d,ident,complete=short is None);return {'status':'SKIPPED_COMMITTED','path':str(path)}
    lockpath=folder/'locks'/f'{key}.lock';lockpath.parent.mkdir(exist_ok=True)
    fd=os.open(lockpath,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=MODULE_WALL;cpu_start=MODULE_CPU
    try:
        if resume:
            c,cur=checkpoint.load(cp,arm,ident)
            if cur['world']!=world or cur['assay']!=assay or cur['fixture_sha256']!=w['sha256'] or cur['limit']!=limit or cur['branch_names']!=branches:raise ValueError('resume fixture/cursor mismatch')
        else:
            if cp.exists():raise ValueError('checkpoint exists; explicit resume required')
            c=None;cur={'world':world,'assay':assay,'fixture_sha256':w['sha256'],'branch_names':branches,'branch_index':0,
                        'next_record':0,'limit':limit,'branches':{},'completed_cpu_s':0.,'completed_worker_s':0.}
        prior_cpu,prior_worker=cur['completed_cpu_s'],cur['completed_worker_s']
        yoke=None
        if arm=='S3_RAND':
            yp=folder/'receipts'/f'{world}_S3_CUE_{assay}.json.gz'
            yoke=read_receipt(yp);audit_receipt(yoke,ident,complete=short is None)
        for bi in range(cur['branch_index'],len(branches)):
            branch=branches[bi]
            if c is None or cur['branch_index']!=bi:
                c=Core(arm);cur['branch_index']=bi;cur['next_record']=0
            if branch not in cur['branches']:
                cur['branches'][branch]={'births':c.births,'records':[],'probes':[],'peak_mutable_bytes':mutable_bytes(c)}
            bd=cur['branches'][branch]
            if yoke is not None:c.yoked_counts=[[a['moves'] for a in r['write']['adaptations']] for r in yoke['branches'][branch]['records']]
            for i in range(cur['next_record'],limit):
                bd['records'].append(record(c,w['events'][i],branch));cur['next_record']=i+1
                for name in w['boundaries'].get(str(i+1),[]):
                    if short is not None:continue
                    c.flush(w['clocks'][name]);bd['probes'].append(probe(c,w,w['clocks'][name],name,branch))
                bd['peak_mutable_bytes']=max(bd['peak_mutable_bytes'],mutable_bytes(c))
                if (i+1)%48==0 or stop==i+1 or i+1==limit:
                    cur['completed_cpu_s']=prior_cpu+time.process_time()-cpu_start
                    cur['completed_worker_s']=prior_worker+time.monotonic()-started
                    checkpoint.save(c,cp,cur,ident)
                    atomic_json(folder/'active'/f'{key}.json',{'pid':os.getpid(),'world':world,'arm':arm,'assay':assay,'branch':branch,'records':i+1,'total':limit,'updated_unix':time.time()})
                if stop==i+1:return {'status':'CHECKPOINT_STOP','path':str(cp),'next_record':i+1}
            if short is not None:bd['probes'].append(probe(c,w,c.last_time,'short_end',branch))
            bd.update(records_completed=limit,final_state_digest=c.state_digest(),final_time=c.last_time,
                      online_cpu_s=sum(r['online_cpu_s'] for r in bd['records'])+sum(p['online_api_cpu_s'] for p in bd['probes'] if p['name']=='final'))
            if arm=='ERROR':bd['Q_HALF_online_cpu_s']=bd['online_cpu_s']+sum(p['Q_HALF_additional_readout_cpu_s'] for p in bd['probes'] if p['name']=='final')
            cur['branch_index']=bi+1;cur['next_record']=0;c=None
        require(ident)
        doc={'schema':'NEXT_CORE_DEVELOPMENT_JOB_V1','development':True,'world':world,'arm':arm,'assay':assay,
             'source_identity':ident,'fixture_sha256':w['sha256'],'branches':cur['branches'],
             'limits':{'records':limit,'branches':branches},'cpu_s':time.process_time()-cpu_start+prior_cpu,
             'worker_s':time.monotonic()-started+prior_worker,
             'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'complete':short is None}
        audit_receipt(doc,ident,complete=short is None);write_receipt(path,doc)
        (folder/'active'/f'{key}.json').unlink(missing_ok=True)
        return {'status':'COMMITTED','path':str(path),'lives':len(branches),'records':limit*len(branches)}
    finally:lockpath.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--world',type=int,required=True);p.add_argument('--arm',required=True);p.add_argument('--assay',choices=('lifetime','reuse','latin'),required=True)
    p.add_argument('--folder',required=True);p.add_argument('--short',type=int);p.add_argument('--stop',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--hold-after-stop',action='store_true');a=p.parse_args()
    result=run(a.world,a.arm,a.assay,a.folder,short=a.short,stop=a.stop,resume=a.resume)
    print(json.dumps(result),flush=True)
    if a.hold_after_stop:
        if result['status']!='CHECKPOINT_STOP':raise ValueError('hold requires a checkpoint stop')
        while True:time.sleep(.1)
if __name__=='__main__':main()
