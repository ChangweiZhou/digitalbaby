"""One frozen world/arm/assay job, with durable branch and record cursors."""
import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path
import bootstrap
from v2_core import TrialCore, ARMS
from v2_fixture import make_world, SCIENCE_WORLDS, DEVELOPMENT_WORLDS
from v2_experiment import instrument, record, probe
from v2_checkpoint import save, load
from v2_integrity import identity, verify
from v2_audit import read, audit_receipt
from compact_storage import atomic_json, commit_receipt


def run_job(world_id, arm, assay, folder, source_identity, *, resume=False, stop_after=None,
            record_limit=None, branch_limit=None, progress=None):
    began = time.monotonic(); cpu_began = 0.  # Charge process startup/import CPU too.
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    key=f'{world_id}_{arm}_{assay}'
    import fcntl
    (folder/'locks').mkdir(exist_ok=True)
    mutex=(folder/'locks'/f'{key}.lock').open('a'); fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=folder/'receipts'/f'{key}.json.gz'; cp=folder/'checkpoints'/f'{key}.npz'; active=folder/'active'/f'{key}.json'
    world=make_world(world_id,assay)
    limit=len(world['events']) if record_limit is None else record_limit
    count=len(world['branches']) if branch_limit is None else branch_limit
    complete=limit==len(world['events']) and count==len(world['branches'])
    if not complete and world_id not in DEVELOPMENT_WORLDS: raise ValueError('short science forbidden')
    if receipt.exists():
        doc=read(receipt); audit_receipt(doc,source_identity,complete=complete)
        return doc,{'already_completed':True}
    if cp.exists():
        if not resume: raise ValueError('partial job requires explicit resume')
        core,cursor=load(cp,arm,source_identity)
        if cursor['world']!=world_id or cursor['assay']!=assay or cursor['limits']!=[limit,count]: raise ValueError('resume job mismatch')
    else:
        core=TrialCore(arm)
        cursor={'world':world_id,'arm':arm,'assay':assay,'fixture_sha256':world['sha256'],
                'branch_index':0,'next_record':0,'branches':{},'limits':[limit,count],
                'worker_s_prior':0.,'cpu_s_prior':0.}
    prior=cursor['worker_s_prior']; cpu_prior=cursor['cpu_s_prior']; attempted=0
    trace=instrument(core)
    for bi in range(cursor['branch_index'],count):
        branch=world['branches'][bi]
        cursor['branches'].setdefault(branch,{'births':core.births,'records':[],'probes':[]})
        history=cursor['branches'][branch]
        for i in range(cursor['next_record'],limit):
            history['records'].append(record(core,world['events'][i],branch,trace))
            for name in world['boundaries'].get(str(i+1),[]):
                core.flush(world['clocks'][name]); history['probes'].append(probe(core,world,world['clocks'][name],name))
            cursor['next_record']=i+1; attempted+=1
            if (i+1)%32==0 or i+1==limit:
                cursor.update(worker_s_prior=prior+time.monotonic()-began,cpu_s_prior=cpu_prior+time.process_time()-cpu_began)
                save(core,cp,cursor,source_identity)
                atomic_json(active,{'pid':os.getpid(),'world':world_id,'arm':arm,'assay':assay,'branch':branch,
                                    'committed_record_cursor':i+1,'heartbeat_unix':time.time()})
                if progress: progress(branch,i+1)
                if stop_after and attempted>=stop_after: return None,{'deliberate_stop':True,'cursor':i+1}
        history['final_state_digest']=core.state_digest()
        cursor['branch_index']=bi+1; cursor['next_record']=0
        if bi+1<count: core=TrialCore(arm); trace=instrument(core)
        cursor.update(worker_s_prior=prior+time.monotonic()-began,cpu_s_prior=cpu_prior+time.process_time()-cpu_began)
        save(core,cp,cursor,source_identity)
    doc={k:cursor[k] for k in ('world','arm','assay','fixture_sha256','branches')}
    doc.update(schema='PERSISTENT_V2_RECEIPT',identity=source_identity,
               worker_s=prior+time.monotonic()-began,cpu_s=cpu_prior+time.process_time()-cpu_began,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
               runtime={'python':platform.python_version()},logical_lives=count,
               engineering_only=world_id in DEVELOPMENT_WORLDS,completed_unix=time.time())
    audit_began=time.process_time()
    audit_receipt(doc,source_identity,complete=complete)
    doc['cpu_receipt_audit_s']=time.process_time()-audit_began
    doc['cpu_s']=cpu_prior+time.process_time()-cpu_began
    doc['worker_s']=prior+time.monotonic()-began
    commit_receipt(receipt,doc)
    if active.exists(): active.unlink()
    # A committed complete receipt is the restartable unit; its obsolete cursor
    # can be removed after the receipt has been fsynced and audited.
    if cp.exists(): cp.unlink()
    return doc,{'already_completed':False}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--world',type=int,required=True); p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--assay',choices=('lifetime','reuse'),required=True); p.add_argument('--development',action='store_true')
    p.add_argument('--folder',type=Path); p.add_argument('--short',action='store_true'); p.add_argument('--stop',type=int)
    p.add_argument('--resume',action='store_true'); a=p.parse_args()
    if a.development:
        if a.world not in DEVELOPMENT_WORLDS or a.folder is None: raise ValueError('development roster')
        source=identity(); folder=a.folder
    else:
        if a.world not in SCIENCE_WORLDS or a.short or a.stop: raise ValueError('science roster')
        source=verify(); q=json.loads((bootstrap.ROOT/'results/QUALIFICATION.json').read_text())
        if q['verdict']!='PASS' or q['identity']!=source: raise ValueError('qualification missing')
        folder=bootstrap.ROOT/'results/science'
    doc,state=run_job(a.world,a.arm,a.assay,folder,source,resume=a.resume,stop_after=a.stop,
                      record_limit=64 if a.short else None,branch_limit=1 if a.short else None,
                      progress=lambda b,c:print(json.dumps({'branch':b,'cursor':c}),flush=True))
    print(json.dumps({'world':a.world,'arm':a.arm,'assay':a.assay,**state}),flush=True)


if __name__=='__main__': main()
