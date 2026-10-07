"""Three-cycle technical exercise. No science or publication operation."""
import argparse
import copy
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path
import runtime
import numpy as np
from core import Core as ParentCore
from bridge_core import RetrievalCore, address, read_addresses
from dev_worker import record, probe, plain
from assays import make_world, DT, RECORD_SECONDS
from independent_audit import audit, check_prediction, assert_read_activity
from association import Associations
from identity import identity, source_map
from io_utils import atomic_json, read_receipt

def rejected(fn):
    try: fn()
    except (ValueError, AssertionError, TypeError, KeyError, IndexError): return True
    raise AssertionError('hostile case was accepted')

def parent_record(c,e,flag=True):
    for i,b in enumerate(bytes.fromhex(e['cue_hex'])): c.feed(b,e['at']+i*DT)
    p=c.predict(e['at']+12*DT);c.observe_outcome(e['outcome'],e['at']+12*DT,learn=flag)
    c.feed(10,e['at']+13*DT);c.flush(e['at']+RECORD_SECONDS)
    return p

def cycle2():
    start=time.monotonic();cpu=time.process_time();checks={}; attacks=0
    w=make_world(71008002,'lifetime');a=RetrievalCore();b=ParentCore('ERROR')
    for e in w['events'][:8]:
        r=record(a,e,'W');p=parent_record(b,e)
        assert r['prediction']['policies']['ERROR']['combined']==list(p.combined)
        assert a.bank_digests()==b.bank_digests() and a.native_digest()==b.state_digest()
    checks['ERROR_exact_native_state_and_output_parity_records']=8
    assert len({id(m.fly) for m in a.models})==8 and len({id(m.fly.m.fast) for m in a.models})==8
    assert len({id(m.fly.m.B.data) for m in a.models})==8
    checks['eight_independent_canonical_births']=True
    a=RetrievalCore();b=RetrievalCore()
    for e in w['events'][:12]:
        record(a,e,'W');r=record(b,e,'N_old')
        assert a.associations.digest()==b.associations.digest()
        assert all(x['no_write_reference_equal'] and x['delta_l1']==0 for x in r['actual_calls'])
    checks['matched_W_N_association_and_actual_native_clamp_records']=12
    clone=a.clone();before=a.state_digest();record(clone,w['events'][12],'W');assert a.state_digest()==before
    assert clone.associations.shared is not a.associations.shared
    checks['native_and_association_clone_isolation']=True
    # Exact read operation on current native state, including elapsed-time handling.
    c=RetrievalCore();e=w['events'][0]
    for i,z in enumerate(bytes.fromhex(e['cue_hex'])):c.feed(z,e['at']+i*DT)
    t=e['at']+12*DT
    for m in c.models:
        ids=address(m,t);v=read_addresses(m,t,[ids])[0]
        assert np.isclose(v,m.association_value(t),rtol=0,atol=1e-12)
    checks['coherent_address_read_matches_inherited_native_reader']=8
    p=c.predict(t)
    assert p.policies['ERROR']['combined']==p.policies['LINK']['combined']==p.policies['PERM']['combined']
    rejected(lambda:c.predict(t,heldout=True));attacks+=1
    rejected(lambda:c.feed(e['outcome'],t));attacks+=1
    rejected(lambda:c.observe_outcome(e['outcome'],t+1));attacks+=1
    c.observe_outcome(e['outcome'],t,learn=False)
    # Future answer changes native learning, never a prior prediction or address pair.
    a=RetrievalCore();b=RetrievalCore()
    for i,z in enumerate(bytes.fromhex(e['cue_hex'])):a.feed(z,e['at']+i*DT);b.feed(z,e['at']+i*DT)
    pa,pb=a.predict(t),b.predict(t)
    assert pa.policies==pb.policies and pa.shared_ids==pb.shared_ids and pa.private_ids==pb.private_ids
    a.observe_outcome(48,t);b.observe_outcome(49,t)
    assert a.associations.digest()==b.associations.digest()
    checks['future_teacher_cannot_select_associations']=True
    # Observe actual activity passed to the reader, rather than trusting logged slot names.
    c=a;query=c.associations.query(np.asarray(pa.shared_ids,np.int32)); captured=[]
    common=c.private[0].common;original=common.observed_activity
    def spy(m,x):captured.append(np.asarray(x).copy());return original(m,x)
    common.observed_activity=spy
    try:c._retrieve(query,t,False)
    finally:common.observed_activity=original
    assert_read_activity(captured,query,c.private[0].n_native_kc)
    wrong=[np.roll(x,1,axis=1) for x in captured]
    rejected(lambda:assert_read_activity(wrong,query,c.private[0].n_native_kc))
    attacks+=1;checks['actual_retrieval_activity_observed']=4
    captured=[];common.observed_activity=spy
    try:c._retrieve(query,t,True)
    finally:common.observed_activity=original
    assert_read_activity(captured,query,c.private[0].n_native_kc,c.permutation)
    checks['actual_permuted_retrieval_activity_observed']=4
    m=c.private[0].fly.m;perm=c.permutation
    assert sorted(perm.tolist())==list(range(len(perm)))
    for arr in (m.kc_side,np.asarray(m.B.sum(axis=0)).ravel()>0,np.abs(m.T[:,2:4]).sum(axis=1)>0):
        assert np.array_equal(arr,arr[perm])
    checks['permutation_preserves_declared_strata']=True
    # A real bypass of native no-write cannot hide behind a correct receipt flag.
    c=RetrievalCore();cls=type(c.shared[0]);original=cls.teach_logged
    def bypass(self,*args,**kw):kw['write']=True;return original(self,*args,**kw)
    cls.teach_logged=bypass
    try:rejected(lambda:record(c,e,'N_old'));attacks+=1
    finally:cls.teach_logged=original
    # Independent receipt audit and mutations to causal evidence.
    c=RetrievalCore();rows=[record(c,z,'N_old') for z in w['events'][:4]]
    d={'source_identity':identity(),'development':True,'world':71008002,'assay':'lifetime','fixture_sha256':w['sha256'],
       'limit':4,'complete':False,'branches':{'N_old':{'records':rows,'births':c.births,'probes':[]}}}
    audit(d,identity())
    mutations=[]
    for field,value in [('learn',True),('observed_at',-1)]:
        z=copy.deepcopy(d);z['branches']['N_old']['records'][0][field]=value;mutations.append(z)
    z=copy.deepcopy(d);z['branches']['N_old']['records'][0]['actual_calls'][0]['delta_l1']=1.;mutations.append(z)
    z=copy.deepcopy(d);z['branches']['N_old']['records'][1]['prediction']['query'][0]['private_ids'][0]+=1;mutations.append(z)
    z=copy.deepcopy(d);z['branches']['N_old']['records'][0]['write']['association']['observations']=5;mutations.append(z)
    z=copy.deepcopy(d);z['branches']['N_old']['records'][0]['write']['s'][0]+=.1;mutations.append(z)
    z=copy.deepcopy(d);z['branches']['N_old']['records'][0]['actual_calls'][0]['address_ids'][0]+=1;mutations.append(z)
    z=copy.deepcopy(d);z['branches']['N_old']['records'][0]['prediction']['policies']['LINK']['combined'][0]+=.1;mutations.append(z)
    z=copy.deepcopy(d);z['branches']['N_old']['births'][1]['fly_id']=z['branches']['N_old']['births'][0]['fly_id'];mutations.append(z)
    for z in mutations:rejected(lambda z=z:audit(z,identity()));attacks+=1
    checks['rejected_hostile_cases']=attacks
    out={'cycle':2,'verdict':'PASS','checks':checks,'source_identity':identity(),
         'cpu_s':time.process_time()-cpu,'worker_s':time.monotonic()-start,'evidence_level':'E0'}
    atomic_json(runtime.ROOT/'evidence/CYCLE2.json',out)
    print(json.dumps(out),flush=True)

def child(args,log):
    with open(log,'w') as f:
        p=subprocess.Popen([sys.executable,'-u',str(runtime.ROOT/'code/dev_worker.py'),*args],stdout=f,stderr=subprocess.STDOUT)
    return p

def cycle3():
    start=time.monotonic();folder=runtime.ROOT/'technical/cycle3';folder.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for assay in ('lifetime','reuse'):
        args=['--world','71008003','--assay',assay,'--folder',str(folder)]
        jobs.append((assay,child(args,folder/f'{assay}.log')))
    for name,p in jobs:
        code=p.wait(timeout=5400)
        if code:raise RuntimeError(f'{name} full development job failed; inspect {folder/name}.log')
    docs=[read_receipt(p) for p in sorted(folder.glob('*.json.gz'))]
    checks=[audit(d,identity()) for d in docs]
    # Independent full branch roster and matched unsupervised histories.
    for assay in ('lifetime','reuse'):
        ds=[d for d in docs if d['assay']==assay]; merged=copy.deepcopy(ds[0]);merged['branches']={}
        for d in ds:merged['branches'].update(d['branches'])
        merged['complete']=True;audit(merged,identity())
    # Fresh-process resume at a safe whole-record boundary; no completed native life is rerun.
    base=['--world','71008003','--assay','reuse','--branch','W','--limit','32']
    reference=folder/'resume_reference';resumed=folder/'resume_actual'
    commands=[(base+['--folder',str(reference)],'resume_reference.log'),
              (base+['--folder',str(resumed),'--stop','24'],'resume_stop.log'),
              (base+['--folder',str(resumed),'--resume'],'resume_continue.log')]
    for args,log in commands:
        p=child(args,folder/log)
        if p.wait(timeout=600):raise RuntimeError(f'resume fixture failed: {log}')
    a=read_receipt(next(reference.glob('*.json.gz')));b=read_receipt(next(resumed.glob('*.json.gz')))
    ba,bb=a['branches']['W'],b['branches']['W']
    assert ba['final_state_digest']==bb['final_state_digest'] and ba['association_final_digest']==bb['association_final_digest']
    for ra,rb in zip(ba['records'],bb['records']):
        for k in ('policies','shared','private','combined','retrieval','permutation_retrieval','shared_ids','private_ids','query'):
            assert ra['prediction'][k]==rb['prediction'][k]
        assert ra['write']==rb['write'] and ra['actual_calls']==rb['actual_calls']
    # A repeated request skips the committed reference rather than executing records again.
    result=subprocess.run([sys.executable,str(runtime.ROOT/'code/dev_worker.py'),*commands[0][0]],capture_output=True,text=True,timeout=120)
    assert result.returncode==0 and 'SKIPPED_COMMITTED' in result.stdout
    totals={'native_full_lives':sum(x['lives'] for x in checks),'native_full_teaching_records':sum(x['records'] for x in checks),
            'full_probe_rows':sum(x['probe_rows'] for x in checks),'full_cpu_s':sum(d['cpu_s'] for d in docs),
            'full_worker_s':sum(d['worker_s'] for d in docs),'max_rss_bytes':max(d['peak_rss_bytes'] for d in docs),
            'qualification_disk_bytes':sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())}
    metrics={}
    for d in docs:
        for branch,bd in d['branches'].items():
            if branch!='W':continue
            final=next(p for p in bd['probes'] if p['name']=='final')
            metrics[d['assay']]={}
            for name in ('ERROR','LINK','PERM'):
                metrics[d['assay']][name]={}
                for group in sorted({r['stage'] for r in final['rows']}):
                    rows=[r for r in final['rows'] if r['stage']==group]
                    if d['assay']=='lifetime' and group=='old':
                        w=make_world(d['world'],d['assay'])
                        rows=[r for r in rows if r['item'] in w['intact_old_items']]
                    metrics[d['assay']][name][group]=sum(r['policies'][name]['correct'] for r in rows)/len(rows)
    costs={}
    for assay in ('lifetime','reuse'):
        bd=next(d['branches']['W'] for d in docs if d['assay']==assay and 'W' in d['branches'])
        costs[assay]={k:sum(r[k] for r in bd['records']) for k in ('parent_online_cpu_s','LINK_online_cpu_s','PERM_online_cpu_s','physical_online_cpu_s')}
    out={'cycle':3,'verdict':'PASS','source_identity':identity(),'full_jobs':2,'comparison_conditions':3,
         'independent_audits':checks,'totals':totals,'development_only_final_W_metrics':metrics,
         'training_API_costs':costs,'safe_boundary_fresh_process_resume':'BIT_IDENTICAL',
         'committed_receipt_guard':'SKIPPED_COMMITTED','worker_s':time.monotonic()-start,
         'science_worlds':0,'note':'one development world, not scientific inference; no parameter selection'}
    atomic_json(runtime.ROOT/'evidence/CYCLE3.json',out);print(json.dumps(out),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cycle',choices=('2','3'));a=p.parse_args()
    (cycle2 if a.cycle=='2' else cycle3)()
