"""Three bounded E0/development cycles, never scientific confirmation."""
import argparse,concurrent.futures,copy,dataclasses,json,os,subprocess,sys,time,types,resource
from pathlib import Path
import numpy as np
import bootstrap
from core import Core,policies
from centered_core import CenteredCore
from mechanisms import projections,relation_q,MechanismBrain,graph_digest,DiagnosticUnavailable,sha
from assays import make_world,permitted,DT,RECORD_SECONDS,module
from spec import PHYSICAL_ARMS,LATIN_ARMS,DEVELOPMENT_WORLDS,EPSILONS,DOSES,CPU_LIMIT,WORKER_LIMIT,RSS_LIMIT,STORAGE_LIMIT
from worker import record
from auditor import audit_receipt,check_record
from io_utils import atomic_json,read_receipt,digest
from integrity import identity,lock
from checkpoint import load
from analysis_math import lower,harm_upper
ROOT=bootstrap.ROOT
def cpu_total():
    r=resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time()+r.ru_utime+r.ru_stime

def reject(f):
    try:f()
    except (ValueError,AssertionError,KeyError,IndexError,TypeError,DiagnosticUnavailable):return True
    raise AssertionError('hostile case accepted')

def child(world,arm,assay,folder,short=None,stop=None,resume=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(ROOT/'worker.py'),'--world',str(world),'--arm',arm,'--assay',assay,'--folder',str(folder)]
    if short is not None:cmd+=['--short',str(short)]
    if stop is not None:cmd+=['--stop',str(stop)]
    if resume:cmd+=['--resume']
    logfile=folder/f'{world}_{arm}_{assay}_{time.time_ns()}.log'
    with logfile.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=3600 if assay!='latin' else 900)
    if r.returncode:raise RuntimeError(f'development job failed: {logfile}')
    return folder/'receipts'/f'{world}_{arm}_{assay}.json.gz'

def killed_child(world,arm,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(ROOT/'worker.py'),'--world',str(world),'--arm',arm,'--assay','lifetime','--folder',str(folder),'--short','32','--stop','24','--hold-after-stop']
    logfile=folder/'killed_child.log';cp=folder/'checkpoints'/f'{world}_{arm}_lifetime.npz'
    with logfile.open('w') as f:
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+240
        try:
            while True:
                if proc.poll() is not None:raise RuntimeError(f'child exited before kill fixture: {logfile}')
                if cp.exists() and 'CHECKPOINT_STOP' in logfile.read_text():break
                if time.monotonic()>deadline:raise RuntimeError('checkpoint kill fixture timeout')
                time.sleep(.1)
            proc.terminate();code=proc.wait(timeout=10)
            if code!=-15:raise AssertionError('SIGTERM stop not observed')
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
    atomic_json(folder/'KILL_RECORD.json',{'development':True,'pid':proc.pid,'exit_code':code,'signal':'SIGTERM','persisted_cursor':24})

def parent_record(c,e):
    for i,b in enumerate(bytes.fromhex(e['cue_hex'])):c.feed(b,e['at']+i*DT)
    p=c.predict(e['at']+12*DT);c.observe_outcome(e['outcome'],e['at']+12*DT,learn=True)
    c.feed(10,e['at']+13*DT);c.flush(e['at']+RECORD_SECONDS);return p

def read_address_check(brain,t):
    expected=brain.rel_features(t)[2];observed=[];common=brain.common;orig=common.observed_activity
    def spy(n,x):observed.append(np.asarray(x).copy());return orig(n,x)
    common.observed_activity=spy
    try:brain.association_value(t)
    finally:common.observed_activity=orig
    assert len(observed)==1 and np.array_equal(observed[0],np.atleast_2d(expected))

def byte_clock_check(native,rel,cue):
    for i,byte in enumerate(cue):
        native.byte(byte,i*DT,learn=False);rel.byte(byte,i*DT,learn=False)
        assert native.fly.m.event_count==rel.fly.m.event_count
        assert native.fly.m.elapsed==rel.fly.m.elapsed
        assert native.brain_t==rel.brain_t and native.pending_t==rel.pending_t
        assert np.array_equal(native.w,rel.w) and np.array_equal(native.bias,rel.bias)
        assert np.array_equal(native.fe.p,rel.fe.p)

def cycle1():
    start=time.monotonic();cpu_start=cpu_total();checks={};w=make_world(61005001,'lifetime')
    original=CenteredCore();new=Core('CENTER')
    for e in w['events'][:8]:
        p=parent_record(original,e);r=record(new,e,'W')
        assert r['prediction']['emitted']==p.emitted and tuple(r['prediction']['combined'])==p.combined
        assert original.bank_digests()==new.bank_digests()
    checks['CENTER_exact_source_parity_records']=8
    v2core=module('_next_core_parent_v2',bootstrap.V2/'v2_core.py');old=v2core.TrialCore('ERROR');c=Core('ERROR')
    for e in w['events'][:8]:
        p=parent_record(old,e);r=record(c,e,'W')
        assert old.bank_digests()==c.bank_digests() and tuple(r['prediction']['combined'])==p.combined
    checks['ERROR_exact_source_parity_records']=8
    a,b,perm=projections()
    assert a.shape==b.shape==(44,88) and (np.count_nonzero(a,axis=1)==5).all() and (np.count_nonzero(b,axis=1)==5).all()
    assert set(np.unique(a))=={-.2,0.,.2} and sorted(perm.tolist())==list(range(88)) and (perm!=np.arange(88)).all()
    h=np.linspace(0,1,88);z=(a@h)*(b@h)
    assert np.array_equal(z,(-a@h)*(-b@h)) and np.array_equal(-z,(-a@h)*(b@h))
    assert np.array_equal(relation_q(np.zeros(88)),np.zeros(88))
    checks['REL_projection_sign_zero_contract']=True
    for arm in PHYSICAL_ARMS:
        c=Core(arm);assert len({id(m.fly) for m in c.models})==len(c.models)
        assert len({id(m.fly.m.B.data) for m in c.models})==len(c.models)
        if arm=='S3_RAND':c.yoked_counts=[[0]*4 for _ in range(64)]
        d=c.clone();d.private[0].fly.m.slow[0]+=1
        assert c.private[0].fly.m.slow[0]!=d.private[0].fly.m.slow[0]
        if arm=='S3_RAND':
            d.yoked_counts[0][0]=1
            assert c.yoked_counts[0][0]==0
        e=w['events'][0]
        for stage in ('old','new','revision'):
            d=c.clone();ev=dict(e,stage=stage)
            row=record(d,ev,'N_'+stage)
            assert all(x['no_write_reference_equal'] for x in row['actual_calls'])
        if arm in DOSES:
            e=w['events'][0];d=c.clone()
            for i,byte in enumerate(bytes.fromhex(e['cue_hex'])):d.feed(byte,e['at']+i*DT)
            t=e['at']+12*DT;m=d.extra[0];read_address_check(m,t)
            orig=MechanismBrain.association_value
            MechanismBrain.association_value=lambda self,t:__import__('stores').F151.association_value(self,t)
            try:reject(lambda:read_address_check(m,t))
            finally:MechanismBrain.association_value=orig
            # Compare inherited predictor features and exact byte-clock transitions.
            native=__import__('stores').birth('native')[0];rel=c.extra[0].clone()
            byte_clock_check(native,rel,bytes.fromhex(e['cue_hex']))
            _,q,x=rel.rel_features(t);rel.byte(e['outcome'],t,learn=False)
            assert np.array_equal(rel.pending_x,x)
            reference=rel.fly.clone();reference.alpha_scale=rel.original_alpha
            test=rel.fly.clone();r0=reference.event(t-reference.m.elapsed,x,1.,True);r1=test.event(t-test.m.elapsed,x,1.,True)
            assert np.allclose(r1['rawalpha'],r0['rawalpha']*DOSES[arm],rtol=1e-12,atol=1e-12)
    checks['all_arm_independent_birth_clone_and_stage_clamps']=len(PHYSICAL_ARMS)
    # Three scientific fixtures and exposure keys are checked, without running science worlds.
    for assay in ('lifetime','reuse','latin'):
        f=make_world(61005001,assay);taught={r['cue_hex'] for r in f['sets']['old']+f['sets']['new']}
        assert not taught & {r['cue_hex'] for r in f['sets'].get('heldout',[])}
    checks['registered_exposure_audit']=True
    for arm in DOSES:
        c=Core(arm);e=w['events'][0];before=c.base_state_digest();row=record(c,e,'W')
        plain=Core('ERROR');record(plain,e,'W');assert c.base_state_digest()==plain.base_state_digest()
    checks['extra_bank_does_not_change_parent_banks']=True
    from core import Prediction
    from spec import SCALES
    example=Prediction(49,(SCALES[0],0.,0.,0.),(-1.5*SCALES[1],0.,0.,0.),(-.5,0.,0.,0.))
    before=c.state_digest();p=policies(example)
    assert p['native']['emitted']==49 and p['Q_HALF']['emitted']==48
    assert c.state_digest()==before
    checks['Q_HALF_state_free_readout']=True
    out={'cycle':1,'verdict':'PASS','checks':checks,'source_identity':identity(),'worker_s':time.monotonic()-start,'cpu_s':cpu_total()-cpu_start,'evidence':'E0 only; parameters not selected from outcomes'}
    atomic_json(ROOT/'results/CYCLE1.json',out);return out

def cycle2():
    start=time.monotonic();cpu_start=cpu_total();cases=0;w=make_world(61005002,'lifetime');e=w['events'][0]
    # Supervised interventions must not change graph/evidence evolution.
    for arm in ('P005','P0005','S3_CUE','S3_RAND'):
        a,b=Core(arm),Core(arm);yoke=[]
        if arm=='S3_RAND':
            s=Core('S3_CUE')
            for ev in w['events'][:33]:yoke.append([x['moves'] for x in record(s,ev,'W')['write']['adaptations']])
            a.yoked_counts=b.yoked_counts=yoke
        for ev in w['events'][:32]:
            ra,rb=record(a,ev,'W'),record(b,ev,'N_old')
            assert ra['write']['adaptations']==rb['write']['adaptations']
        before=a.state_digest();clone=a.clone();parent_record(clone,w['events'][32]);assert a.state_digest()==before
    base=Core('REL10');row=record(base,e,'N_old')
    variants=[]
    for k,v in [('learn',True),('prediction_precedes_outcome',False),('predicted_at',-1.)]:
        d=copy.deepcopy(row);d[k]=v;variants.append(d)
    d=copy.deepcopy(row);d['actual_calls'][8]['write']=True;variants.append(d)
    d=copy.deepcopy(row);d['write']['s'][0]+=.1;variants.append(d)
    d=copy.deepcopy(row);d['actual_calls'][0]['coefficients']=[2.];variants.append(d)
    d=copy.deepcopy(row);d['actual_calls'][4]['api']='punishment=-s';variants.append(d)
    d=copy.deepcopy(row);d['write']['outcome']=51 if e['outcome']!=51 else 48;variants.append(d)
    d=copy.deepcopy(row);d['actual_calls'][8]['no_write_reference_equal']=False;variants.append(d)
    d=copy.deepcopy(row);d['prediction']['emitted']=255;variants.append(d)
    for d in variants:reject(lambda d=d:check_record(d,'REL10',e,'N_old',12));cases+=1
    reject(lambda:permitted('N_old_rel','old'));cases+=1
    # Actual, not merely metadata, no-write bypass is rejected by the native reference.
    c=Core('REL10');orig=type(c.extra[0]).teach_logged
    def bypass(self,*args,**kw):kw['write']=True;return orig(self,*args,**kw)
    c.extra[0].teach_logged=types.MethodType(bypass,c.extra[0])
    # instrument installs from the class, so inject into the class for this attack only.
    cls=type(c.extra[0]);cls.teach_logged=bypass
    try:reject(lambda:record(c,e,'N_old'));cases+=1
    finally:cls.teach_logged=orig
    # Target leakage: changed future answer must leave the prior prediction/address unchanged.
    a,b=Core('REL10'),Core('REL10')
    for i,byte in enumerate(bytes.fromhex(e['cue_hex'])):a.feed(byte,e['at']+i*DT);b.feed(byte,e['at']+i*DT)
    pa,pb=a.predict(e['at']+12*DT),b.predict(e['at']+12*DT);assert pa==pb and a.bank_digests()==b.bank_digests()
    reject(lambda:a.feed(e['outcome'],e['at']+12*DT));cases+=1
    reject(lambda:a.predict(e['at']+12*DT,heldout=True));cases+=1
    # Late outcome must use exact current cache, not post-answer sensory features.
    a.observe_outcome(e['outcome'],e['at']+12*DT);b.observe_outcome(48+(e['outcome']-48+1)%4,e['at']+12*DT)
    assert a.bank_digests()!=b.bank_digests()
    # Synthetic decoder/native poison contracts cannot silently qualify.
    m=Core('REL10').extra[0]
    reject(lambda:m.fly.event(0.,-np.ones(m.n_native_kc),1.,True));cases+=1
    for i in (0,1):
        bad=Core('REL10');old=type(bad.extra[0])._features
        def wrong(self,t,**kw):return __import__('stores').F151._features(self,t,**kw)
        type(bad.extra[0])._features=wrong
        try:reject(lambda:record(bad,e,'W'));cases+=1
        finally:type(bad.extra[0])._features=old
        break
    native=__import__('stores').birth('native')[0];rel=Core('REL10').extra[0]
    old_byte=MechanismBrain.byte
    def extra_stimulus(self,byte,t,**kw):
        value=old_byte(self,byte,t,**kw)
        self.fly.event(0.,self.pending_x,0.,False)
        return value
    MechanismBrain.byte=extra_stimulus
    try:reject(lambda:byte_clock_check(native,rel,bytes.fromhex(e['cue_hex'])));cases+=1
    finally:MechanismBrain.byte=old_byte
    for branch in ('W','N_old','N_new','N_revision'):
        for stage in ('old','new','revision'):assert permitted(branch,stage)==(branch=='W' or branch!='N_'+stage)
    assert lower([0.]*72,.001)['lower'] is None
    assert lower([0.]*72,.001,width=2)['lower']<0 and harm_upper([0]*72,.001)<.1
    assert lower([0.]*64,.002,width=4)['lower']<lower([0.]*64,.002,width=2)['lower']
    out={'cycle':2,'verdict':'PASS','hostile_cases_rejected':cases,'graph_W_N_bit_parity_records_per_arm':32,
         'source_identity':identity(),'worker_s':time.monotonic()-start,'cpu_s':cpu_total()-cpu_start,'evidence':'E0 interventions, no official outcomes inspected'}
    atomic_json(ROOT/'results/CYCLE2.json',out);return out

def scientific_projection(doc):
    d=copy.deepcopy(doc)
    for k in ('cpu_s','worker_s','peak_rss_bytes'):d.pop(k,None)
    for b in d['branches'].values():
        b.pop('online_cpu_s',None)
        b.pop('Q_HALF_online_cpu_s',None)
        for birth in b['births']:birth.pop('fly_id',None)
        for r in b['records']:
            r.pop('online_cpu_s',None);r.pop('external_audit_cpu_s',None)
        for p in b['probes']:
            p.pop('online_api_cpu_s',None);p.pop('all_probe_cpu_s',None);p.pop('Q_HALF_additional_readout_cpu_s',None)
    return d

def cycle3():
    start=time.monotonic();cpu_start=cpu_total();folder=ROOT/'results/development/cycle3';paths=[]
    jobs=[(a,s) for a in PHYSICAL_ARMS if a!='S3_RAND' for s in ('lifetime','reuse')]+[(a,'latin') for a in LATIN_ARMS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(child,61005003,a,s,folder):(a,s) for a,s in jobs}
        for f in concurrent.futures.as_completed(fs):
            path=f.result();paths.append(path);print(json.dumps({'full_development_committed':len(paths),'total':29,'job':fs[f]}),flush=True)
    for s in ('lifetime','reuse'):paths.append(child(61005003,'S3_RAND',s,folder))
    docs=[read_receipt(p) for p in paths]
    audits=[audit_receipt(d,identity()) for d in docs]
    # Full cursor, graph, dose and output parity across an actual killed child process.
    recovery={}
    for arm in ('CENTER','ERROR','P005','P0005','S3_CUE','S3_RAND','REL05','REL10','REL20','REL_PERM10','FIRST10'):
        base=ROOT/'results/development/recovery'/arm
        if arm=='S3_RAND':
            for name in ('reference','resumed'):child(61005004,'S3_CUE','lifetime',base/name,short=32)
        a=child(61005004,arm,'lifetime',base/'reference',short=32)
        killed_child(61005004,arm,base/'resumed')
        cp=base/'resumed'/'checkpoints'/f'61005004_{arm}_lifetime.npz'
        core,cursor=load(cp,arm,identity());assert cursor['next_record']==24
        reject(lambda:load(cp,arm,'0'*64))
        with np.load(cp,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        k=next(k for k,v in arrays.items() if k!='metadata_json' and v.dtype.kind=='f' and v.size)
        arrays[k].flat[0]+=1
        corrupt=base/'corrupt.npz';np.savez_compressed(corrupt,**arrays);reject(lambda:load(corrupt,arm,identity()))
        with np.load(cp,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        meta=json.loads(arrays['metadata_json'].tobytes());meta.pop('seal');meta['cursor']['branch_index']=77;meta['seal']=digest(meta)
        from io_utils import canonical
        arrays['metadata_json']=np.frombuffer(canonical(meta),dtype=np.uint8)
        bad_cursor=base/'resealed_bad_cursor.npz';np.savez_compressed(bad_cursor,**arrays);reject(lambda:load(bad_cursor,arm,identity()))
        b=child(61005004,arm,'lifetime',base/'resumed',short=32,resume=True)
        assert scientific_projection(read_receipt(a))==scientific_projection(read_receipt(b))
        unchanged=b.read_bytes();child(61005004,arm,'lifetime',base/'resumed',short=32,resume=True);assert b.read_bytes()==unchanged
        recovery[arm]={'cursor':24,'compared_records':32,'fresh_process_bit_parity':True,'tamper_rejected':True,'committed_skip':True}
    # Explicitly compare four/eight-worker scheduling on a common dev input.
    concurrent_checks=[]
    for workers in (4,8):
        dst=ROOT/f'results/development/concurrency{workers}'
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            fs=[pool.submit(child,61005005,a,'reuse',dst,short=2) for a in PHYSICAL_ARMS if a!='S3_RAND']
            concurrent_checks.append([read_receipt(f.result()) for f in fs])
    assert [scientific_projection(d) for d in concurrent_checks[0]]==[scientific_projection(d) for d in concurrent_checks[1]]
    # One complete dev job per registered physical arm/assay: forecast, not science inference.
    cpu_est=worker_est=size_est=checkpoint_est=0.
    for d,p in zip(docs,paths):
        n=64 if d['assay']=='latin' else 72
        cpu_est+=d['cpu_s']*n;worker_est+=d['worker_s']*n;size_est+=p.stat().st_size*n
        cp=folder/'checkpoints'/f"{d['world']}_{d['arm']}_{d['assay']}.npz"
        checkpoint_est+=cp.stat().st_size*n
    maxrss=max(d['peak_rss_bytes'] for d in docs+concurrent_checks[0]+concurrent_checks[1])
    forecast={'science_cpu_h_point':cpu_est/3600,'science_worker_h_point':worker_est/3600,
              'conservative_multiplier':1.5,'science_cpu_h_conservative':1.5*cpu_est/3600,
              'science_worker_h_conservative':1.5*worker_est/3600,'science_receipt_bytes_point':size_est,
              'science_checkpoint_bytes_point':checkpoint_est,
              'peak_rss_bytes':maxrss,'p95_note':'One full job per recipe; no reliable tail quantile. 1.5x is a planning margin, not a confidence bound.'}
    earlier_cpu=sum(json.loads((ROOT/f'results/CYCLE{i}.json').read_text()).get('cpu_s',0.) for i in (1,2))
    qualification_cpu=earlier_cpu+cpu_total()-cpu_start
    forecast['qualification_cpu_h']=qualification_cpu/3600
    # At most eight qualification child processes run at once; charge the entire
    # qualification wall interval at eight slots as a conservative upper bound.
    qualification_worker=sum(json.loads((ROOT/f'results/CYCLE{i}.json').read_text()).get('worker_s',0.) for i in (1,2))+8*(time.monotonic()-start)
    forecast['qualification_worker_h_upper']=qualification_worker/3600
    forecast['receipt_checkpoint_bytes_conservative']=1.5*(size_est+checkpoint_est)
    forecast['log_reserve_bytes']=1024**3
    forecast['qualification_bytes']=sum(p.stat().st_size for p in (ROOT/'results').rglob('*') if p.is_file())
    resource_ok=(qualification_cpu+1.5*cpu_est<CPU_LIMIT and qualification_worker+1.5*worker_est<WORKER_LIMIT
                 and forecast['qualification_bytes']+forecast['receipt_checkpoint_bytes_conservative']+forecast['log_reserve_bytes']<STORAGE_LIMIT and maxrss<RSS_LIMIT)
    out={'cycle':3,'verdict':'PASS' if resource_ok else 'RESOURCE_NOT_QUALIFIED','full_development_jobs':len(docs),
         'full_development_lives':sum(x['lives'] for x in audits),'audits':audits,'recovery':recovery,
         'four_eight_worker_state_parity':True,'forecast':forecast,'source_identity':identity(),
         'worker_s':time.monotonic()-start,'cpu_s':cpu_total()-cpu_start,'science_trajectories':0}
    atomic_json(ROOT/'results/CYCLE3.json',out);return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--cycle',type=int,choices=(1,2,3));p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:
        ident=identity();cs=[json.loads((ROOT/f'results/CYCLE{i}.json').read_text()) for i in (1,2,3)]
        if any(c['verdict']!='PASS' or c['source_identity']!=ident for c in cs):raise ValueError('three current-source cycles not passed')
        lock();out={'verdict':'PASS','cycles_completed':3,'source_identity':ident,'science_authorized':False,'science_trajectories':0}
        atomic_json(ROOT/'results/QUALIFICATION.json',out)
    else:
        try:out={1:cycle1,2:cycle2,3:cycle3}[a.cycle]()
        except Exception as exc:
            atomic_json(ROOT/f'results/CYCLE{a.cycle}_FAILURE.json',{'cycle':a.cycle,'development':True,
                        'error_type':type(exc).__name__,'error':str(exc),'source_identity':identity(),'science_trajectories':0})
            raise
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()
