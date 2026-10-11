"""Read accepted artifacts once; never launch or execute a learning trajectory."""
import argparse, copy, gzip, hashlib, json, os, signal, subprocess, sys, threading, time, traceback, zipfile
from pathlib import Path
import numpy as np
RECOVERY=Path(__file__).resolve().parent
ROOT=RECOVERY.parent.parent
OUT=ROOT/'results'
sys.path.insert(0,str(ROOT/'ops/science_20261011'))
import science as original
import decision, config


def dump(x,p): original.dump(x,p)
def load(p): return original.load(p)
def sha(p): return original.digest(p)

def compact(e):
    r=e['kernel_receipt']; branches={}
    for h,b in r['branches'].items():
        cpu=sum(float(x['production_cpu_seconds']) for rows in [b['old_records']]+[f['new_records'] for f in b['forks'].values()] for x in rows)
        forks={}
        for n,f in b['forks'].items():
            forks[n]={k:{'accuracy':f[k]['accuracy']} for k in ('final_old_held','final_new_held','immediate_old_held')}
            forks[n].update(final_checkpoint={'payload_bytes':f['final_checkpoint']['payload_bytes']},new_records=[])
        branches[h]={'birth':{'payload_bytes':b['birth']['payload_bytes']},'old_records':[{'production_cpu_seconds':cpu}],'forks':forks}
    return {'world':e['world'],'arm':e['arm'],'kernel_receipt':{'world':e['world'],'ordinary':r['ordinary'],'noise':r['noise'],'family':r['family'],'branches':branches}}


def index_inputs():
    original.verify_sources()
    lock=load(OUT/'SOURCE_LOCK.json')['files']
    selection=load(OUT/'SELECTION_LOCK.json')
    if selection['source']!=lock: raise AssertionError('selection source changed')
    admission=load(OUT/'CONFIRM_ADMISSION.json')
    if admission['selection_sha256']!=sha(OUT/'SELECTION_LOCK.json') or admission['generated_epoch']<selection['locked_epoch']: raise AssertionError('confirmation admission chronology')
    manifests={}; rows=[]; counters={'committed_full_jobs':0,'committed_external_records':0,'committed_training_bytes':0,'state_trajectories_with_shared_prefix':0}
    for stage,n in [('dev',12),('confirm',96)]:
        m=original.read_manifest(OUT/stage/'MANIFEST.json'); manifests[stage]=m
        if len(m['worlds'])!=n: raise AssertionError('world roster changed')
        expected={(t['world'],a) for t in m['worlds'] for a in m['arms']}
        actual=set()
        for p in sorted((OUT/stage/'receipts').glob('*.json.gz')):
            cert=load(p.with_name(p.name+'.audit.json'))
            key=p.name[:-8].split('_',1); key=(int(key[0]),key[1])
            if key not in expected or key in actual or cert['verdict']!='PASS' or not cert['independent_record_equations']: raise AssertionError('roster/accepted audit certificate')
            actual.add(key)
            if cert['training_records']!=2400 or cert['state_trajectories']!=21 or cert['science_jobs']!=1: raise AssertionError('accepted count certificate')
            rows.append({'stage':stage,'path':str(p),'world':key[0],'arm':key[1],'receipt_sha256':cert['receipt_sha256'],'certificate_sha256':sha(p.with_name(p.name+'.audit.json')),'training_bytes':cert['training_bytes']})
            counters['committed_full_jobs']+=1; counters['committed_external_records']+=2400; counters['committed_training_bytes']+=cert['training_bytes']; counters['state_trajectories_with_shared_prefix']+=21
        if actual!=expected: raise AssertionError('missing committed whole job; recovery never trains it')
    if manifests['dev']['arms']!=list(config.ARMS) or manifests['confirm']['arms']!=original.roster(selection['selected_candidate']): raise AssertionError('locked arm roster')
    if {r['world'] for r in manifests['dev']['worlds']} & {r['world'] for r in manifests['confirm']['worlds']}: raise AssertionError('world namespace overlap')
    assert counters['committed_full_jobs']==492
    return manifests,rows,counters


class Monitor:
    """No filesystem reads while holding the status lock."""
    def __init__(self,counts,out=OUT):
        self.counts=dict(counts); self.out=out; self.lock=threading.Lock(); self.data={}; self.stop=threading.Event(); self.thread=None
    def update(self,stage,**kw):
        with self.lock:
            self.data.update(stage=stage,heartbeat_epoch=time.time(),supervisor_pid=os.getpid(),workers=0,**kw)
            dump({**self.counts,**self.data},self.out/'STATUS.json')
    def start(self):
        def beat():
            while not self.stop.wait(5):
                with self.lock:
                    self.data['heartbeat_epoch']=time.time()
                    dump({**self.counts,**self.data},self.out/'STATUS.json')
        self.thread=threading.Thread(target=beat,daemon=True); self.thread.start()
    def end(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=10)


def validate_and_reduce(row,m):
    path=Path(row['path'])
    if sha(path)!=row['receipt_sha256'] or sha(path.with_name(path.name+'.audit.json'))!=row['certificate_sha256']: raise AssertionError('accepted artifact modified')
    with gzip.open(path,'rt') as f:e=json.load(f)
    r=e['kernel_receipt']; task=next(t for t in m['worlds'] if t['world']==row['world'])
    if e['schema']!='SAMPLE_GEN_SCIENCE_ENVELOPE_V2' or e['stage']!=m['stage'] or e['manifest_sha256']!=m['sha256'] or e['source']!=m['source']: raise AssertionError('immutable envelope admission')
    if e['world']!=row['world'] or e['arm']!=row['arm'] or r['arm']!=row['arm'] or r['source']!=m['source']: raise AssertionError('receipt key/source')
    if any(r[k]!=task[k] for k in ('world','family','noise','ordinary')):raise AssertionError('manifest population')
    if e['training_bytes']!=row['training_bytes'] or e['training_records']!=2400 or e['state_trajectories']!=21:raise AssertionError('external count mismatch')
    # Recompute reported score arithmetic only, not state transitions or learning.
    for b in r['branches'].values():
        for f in b['forks'].values():
            for name in ('immediate_old_held','final_old_held','final_new_held','final_old_taught','final_new_taught','erased_old_held'):
                q=f[name]; scores=[]
                for orbit in q['rows']:
                    observed=[int(response['emitted']==48+orbit['truth_private']) for response in orbit['responses']]
                    if any(bool(response['exact'])!=bool(v) for response,v in zip(orbit['responses'],observed)):raise AssertionError('stored exact-output score')
                    average=sum(observed)/len(observed)
                    if average!=orbit['accuracy']:raise AssertionError('orientation scoring')
                    scores.append(average)
                if sum(scores)/len(scores)!=q['accuracy']:raise AssertionError('orbit mean scoring')
    slim=compact(e)
    if original.receipt_metric(e)!=original.receipt_metric(slim):raise AssertionError('compact reduction changes metric')
    mm=original.receipt_metric(slim); thresholds={}
    for limit in (.75,.90):
        reached=[n+80 for n,v in zip(config.CHECKPOINTS,mm['W']['persistent_old_held']) if v>=limit]
        thresholds[str(limit)]={'first_observed_total_samples':min(reached) if reached else None,'right_censored_at_400':not bool(reached)}
    diagnostic={'world':e['world'],'arm':e['arm'],'noise':r['noise'],'ordinary':r['ordinary'],'metrics':mm,'thresholds':thresholds,'countermodels':decision.countermodels(r),
        'inclusive_job_cpu_seconds':r['inclusive_cpu_seconds'],'job_wall_seconds':r['wall_seconds'],'peak_worker_rss_bytes':r['peak_rss_bytes']}
    return slim,diagnostic


def audit_final(dev,confirm,manifests):
    selection=load(OUT/'SELECTION_LOCK.json')
    m=original.means(dev,manifests['dev']['arms'])
    if original.choose(m)!=selection['selected_candidate']:raise AssertionError('development selection differs')
    for a,old in selection['means'].items():
        for k,v in old.items():
            if not np.isclose(m[a][k],v,rtol=1e-12,atol=1e-12):raise AssertionError('development aggregate differs: '+a+'/'+k)
    return original.confirm_result(confirm,selection['selected_candidate'],manifests['confirm']['arms'])


def package(summary,counts,monitor):
    monitor.update('PACKAGING',verdict=summary['verdict'])
    target=OUT/'RESULT_BUNDLE.zip'; pending=OUT/'RESULT_BUNDLE.zip.pending'
    with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
        for base in (OUT,ROOT/'ops/science_20261011',RECOVERY):
            for p in sorted(base.rglob('*')):
                if p.is_file() and p.name not in ('RESULT_BUNDLE.zip','RESULT_BUNDLE.zip.pending','PACKAGE.json') and 'active' not in p.parts and '__pycache__' not in p.parts:
                    z.write(p,str(p.relative_to(ROOT)))
        for name in original.ORIGINAL_LOCK():z.write(ROOT.parent/name,'source/'+name)
    with zipfile.ZipFile(pending) as z:
        if z.testzip() is not None:raise AssertionError('package CRC failure')
    amount=sum(p.stat().st_size for base in (OUT,ROOT/'ops/science_20261011',RECOVERY) for p in base.rglob('*') if p.is_file())
    if amount>2*1024**3:raise AssertionError('artifact cap')
    os.replace(pending,target)
    dump({'status':'COMPLETE','bytes':target.stat().st_size,'sha256':sha(target),'verdict':summary['verdict'],'operations_only_recovery':True},OUT/'PACKAGE.json')
    monitor.update('COMPLETE',verdict=summary['verdict'],package_complete=True)


def execute():
    lock=load(RECOVERY/'RECOVERY_SOURCE_LOCK.json')
    if lock['sha256']!=sha(Path(__file__)):raise AssertionError('recovery source changed')
    descriptor=os.open(RECOVERY/'FINISH.claim',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(descriptor)
    manifests,rows,counts=index_inputs()
    caffeinate=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    dump({'supervisor_pid':os.getpid(),'caffeinate_pid':caffeinate.pid,'workers':0,'analysis_only':True,'log':str(RECOVERY/'supervisor.log'),'started_epoch':time.time(),'detached':True},OUT/'LAUNCH.json')
    monitor=Monitor(counts);monitor.update('POSTPROCESS_RECOVERY',processed_jobs=0,total_jobs=len(rows));monitor.start();start=time.monotonic()
    try:
        reduced={'dev':[],'confirm':[]}; diagnostics={'dev':[],'confirm':[]}
        for i,row in enumerate(rows,1):
            slim,diag=validate_and_reduce(row,manifests[row['stage']]);reduced[row['stage']].append(slim);diagnostics[row['stage']].append(diag)
            monitor.update('POSTPROCESS_RECOVERY',processed_jobs=i,total_jobs=len(rows))
            if i%24==0:print(json.dumps({'processed_jobs':i,'total_jobs':len(rows),'elapsed_seconds':time.monotonic()-start}),flush=True)
        for stage in ('dev','confirm'):
            with gzip.open(OUT/stage/'WORLD_METRICS.json.gz','wt') as f:json.dump(diagnostics[stage],f,sort_keys=True,allow_nan=False)
            dump({'stage':'COMPLETE','jobs':len(reduced[stage]),'recovered_from_existing_receipts':True},OUT/stage/'STATUS.json')
            dump({'stage':stage.upper(),'committed_jobs':[{'world':e['world'],'arm':e['arm']} for e in reduced[stage]],'operations_recovery':True},OUT/stage/'COMMIT_INDEX.json')
        monitor.update('CONFIRM_ANALYSIS',processed_jobs=len(rows),total_jobs=len(rows))
        summary=audit_final(reduced['dev'],reduced['confirm'],manifests)
        original.verify_sources()
        dump(summary,OUT/'CONFIRM_SUMMARY.json');dump(summary,OUT/'SUMMARY.json');original.report(summary,'CONFIRM_REPORT')
        audit={'verdict':'PASS','sources_verified':True,'immutable_manifests_and_selection_verified':True,'accepted_artifact_seals_verified':True,'independent_record_equations':'reused sealed PASS certificates; no transition rerun','emitted_score_arithmetic_verified':True,'original_statistical_and_adoption_equations':True,'complete_stages_only':True,'no_new_native_records':True,'recovery_new_training_records':0,**counts}
        dump(audit,OUT/'FINAL_AUDIT.json')
        original.report({**summary,**counts},'REPORT')
        with (OUT/'REPORT.md').open('a') as f:f.write('\nPostprocessing-only recovery: the original status monitor repeatedly decoded all receipts while holding a shared lock. The replacement reads each accepted receipt once, retains compact metric summaries and reports counters from memory. Original learning/design sources, manifests, selection, receipts and acceptance certificates are unchanged; zero new training records were executed.\n')
        dump({'verdict':'PASS','old_source_preserved':True,'receipt_digests_preserved':True,'new_native_records':0,'processed_jobs':len(rows),'elapsed_seconds':time.monotonic()-start,'scientific_verdict':summary['verdict']},RECOVERY/'RECOVERY_AUDIT.json')
        monitor.end();package(summary,counts,monitor)
        print(json.dumps({'stage':'COMPLETE','verdict':summary['verdict'],**counts}),flush=True)
    except BaseException as exc:
        monitor.end();dump({'error':str(exc),'traceback':traceback.format_exc(),'operations_only':True,'new_native_records':0,**counts},OUT/'FAILURE.json');monitor.update('FAILED',error=str(exc));raise
    finally:monitor.end();caffeinate.terminate()


def tests():
    # Fixtures contain score/cost summaries only, no real benchmark or learner execution.
    records=[]
    for ordinary,count in ((False,64),(True,32)):
        for i in range(count):
            for arm in original.roster('COV-R'):
                branches={}
                for h in config.HISTORIES:
                    score=.95 if arm=='COV-R' and h=='W' else .1
                    branches[h]={'birth':{'payload_bytes':90},'old_records':[{'production_cpu_seconds':.01}],
                        'forks':{str(n):{'final_old_held':{'accuracy':score},'final_new_held':{'accuracy':.7},'immediate_old_held':{'accuracy':score},'final_checkpoint':{'payload_bytes':100},'new_records':[{'production_cpu_seconds':.02}]} for n in config.CHECKPOINTS}}
                e={'world':i+(1000 if ordinary else 0),'arm':arm,'kernel_receipt':{'world':i,'ordinary':ordinary,'noise':'STANDARD' if i<count//2 else 'BLURRY','family':'ODD_CYCLE','branches':branches}}
                records.append(e)
    slim=[compact(e) for e in records]
    assert all(original.receipt_metric(e)==original.receipt_metric(s) for e,s in zip(records,slim))
    left=original.confirm_result(records,'COV-R',original.roster('COV-R'));right=original.confirm_result(slim,'COV-R',original.roster('COV-R'))
    for result in (left,right):
        for population in ('opaque_means','ordinary_guard_means'):
            for row in result[population].values():row.pop('production_cpu_seconds')
    assert left==right
    directory=RECOVERY/'synthetic_monitor';directory.mkdir(exist_ok=False)
    monitor=Monitor({'committed_full_jobs':492},directory);monitor.start();t=time.monotonic()
    for i in range(100):monitor.update('SYNTHETIC',processed_jobs=i)
    monitor.end()
    assert time.monotonic()-t<5 and load(directory/'STATUS.json')['processed_jobs']==99
    # Full decoded receipt reads in status/count hot path are forbidden.
    import inspect
    assert 'runner.read' not in inspect.getsource(Monitor) and 'glob(' not in inspect.getsource(Monitor)
    return {'verdict':'PASS','synthetic_scientific_metrics_and_decision_exactly_equal':True,'CPU_reduction_regrouping_only':True,'100_status_updates_seconds':time.monotonic()-t,'hot_path_receipt_reads':0,'new_native_records':0,'science_jobs':0}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');p.add_argument('--finish',action='store_true');a=p.parse_args()
    if a.test:print(json.dumps(tests(),sort_keys=True))
    elif a.finish:execute()
    else:raise ValueError('explicit mode required')
