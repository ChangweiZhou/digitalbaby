"""Operations-only continuation of unstarted diagnostic cells; no scientific replay."""
import os, sys, time, subprocess, resource, traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
sys.dont_write_bytecode=True
import support154 as s
import native154 as n
import features154, probes154
import numpy as np
from scipy import sparse

def main():
    assert not (s.H/'OPERATIONS_CONTINUATION.json').exists(), 'No automatic second continuation'
    failure=s.load(s.H/'FAILURE.json');assert failure['error']=='''TypeError("dict() got multiple values for keyword argument 'CPU'")'''
    old=s.load(s.H/'COUNTS.json');launch=s.load(s.H/'LAUNCH.json')
    for name in ['FAILURE.json','STATUS.json','COUNTS.json','LAUNCH.json']:
        s.save('cycles/logging_failure/'+name,s.load(s.H/name))
    frozen=s.load(s.H/'SOURCE_LOCK.json');assert all(s.sha(s.ROOT/p)==v for p,v in frozen.items())
    groups=[s.load(p) for p in sorted((s.H/'groups').glob('*/RESULT.json'))];assert len(groups)==10
    assert len(list((s.H/'groups').glob('*/HELD.json')))==10
    existing=list(s.H.glob('PROBE_*.json'));assert len(existing)==1
    first=s.load(existing[0]);raw=s.load(s.H/'local'/existing[0].name)
    assert first['id']==raw['id']=='I_FAST_RHO_202610255401_OFF_257' and len(raw['scoring'])==3
    assert old['counts']['native_records']==25600 and old['counts']['batch_fits']==1
    # Carry original CPU, wall, counters, timings and limits. No budget reset.
    base_cpu=s.cpu;carry=old['CPU'];s.cpu=lambda:carry+base_cpu()
    s.START=time.monotonic()-(time.time()-launch['wall_start'])
    s.COUNTS.update(old['counts']);s.TIMES.update(old['timings'])
    def log(event,**fields):
        if 'CPU' in fields:fields['cell_CPU']=fields.pop('CPU')
        with (s.H/'LEDGER.jsonl').open('a') as f:
            import json
            f.write(json.dumps(s.plain(dict(event=event,CPU=s.cpu(),wall=time.monotonic()-s.START,**fields)),allow_nan=False)+'\n')
    s.log=log
    log('LOGGER_QUALIFICATION_ONLY',CPU=0.,no_model_queries=True)
    s.save('OPERATIONS_CONTINUATION.json',dict(reason=failure['error'],scope='Rename diagnostic cell CPU log field only; continue unstarted registered cells',
        frozen_scientific_sources_unchanged=True,native_histories_replayed=0,completed_fits_replayed=0,
        native_record_applications_at_boundary=25600,prior_CPU=carry,original_wall_start=launch['wall_start'],
        completed_probe_sha256=s.sha(existing[0]),completed_probe_raw_sha256=s.sha(s.H/'local'/existing[0].name),
        source_sha256=s.sha(s.H/'continue154.py'),authorization='Original user-authorized design/audit/test/revise exercise; technical operations repair, no science extension',
        unchanged_budget=True,no_new_rates_passes_worlds_or_gates=True))
    log('PROBE_COMMIT_RECOVERED',id=first['id'],already_saved=True,no_replay=True)
    fits=[];data=s.load(s.H/'local/DATA_MANIFEST.json')
    for world_text,dd in data.items():
        world=int(world_text)
        for front in n.FRONTS:
            s.save('STATUS.json',dict(version='V154',status='CYCLE3_FEATURES_AND_PROBES',world=world,front=front,operations_continuation=True))
            key=f'{front}_{world}';cache=s.H/'local'/f'FEATURE_{key}.npz'
            if cache.exists():
                assert key=='I_FAST_202610255401'
                assert s.sha(cache)==s.load(s.H/f'FEATURE_{key}.json')['cache_sha256']
                with np.load(cache) as z:y=z['y'].copy();ids=z['record_ids'].tolist()
                X={tap:sparse.load_npz(s.H/'local'/f'FEATURE_{key}_{tap}.npz') for tap in ['RHO','X','OBS']}
                log('FEATURE_CACHE_RECOVERED',front=front,world=world,no_replay=True)
            else:X,y,ids=features154.collect(dd,front)
            for tap,x in X.items():
                kinds=['OFF_257']+(['LOC_257','OFF_C16','LOC_C16'] if front in ['I_FAST','P_SLOW'] and tap in ['X','OBS'] else [])
                for kind in kinds:
                    s.budget();fitid=f'{front}_{tap}_{world}_{kind}'
                    if fitid==first['id']:fits.append(raw);continue
                    assert not (s.H/f'PROBE_{fitid}.json').exists()
                    fits.append(probes154.fit(x,y,ids,dd,kind,f'{front}_{tap}_{world}'))
    assert len(fits)==48
    held=[]
    for r in groups:
        name=f"{r['config']['id']}_{r['world']}";h=s.load(s.H/'groups'/name/'HELD.json')
        held.append(dict(group=name,ready=r['ready'],branches=[dict(branch=x['branch'],metrics=x['metrics']) for x in h['branches']]))
    verdict='DESCRIPTIVE_NATIVE_DEVELOPMENT_COMPLETE'
    summary=dict(version='V154',verdict=verdict,port_ready=True,native_groups=10,correlated_native_histories=30,
        native_record_applications=25600,native_observed_targets=sum(r['targets'] for r in groups),
        native_readiness=[dict(config=r['config'],world=r['world'],ready=r['ready'],counts=r['readiness_counts'],immediate_exact=r['immediate_exact']) for r in groups],
        held=held,checkpoint_lane_status='COMPLETE',diagnostic_fits=len(fits),qualified_fits=sum(x['status']=='QUALIFIED' for x in fits),
        optimizer_unresolved=sum(x['status']=='OPTIMIZER_UNRESOLVED' for x in fits),new_core_adopted=False,
        operations_failure_preserved='cycles/logging_failure/FAILURE.json',completed_science_replayed=False)
    s.save('SUMMARY.json',summary)
    s.save('COUNTS.json',dict(counts=dict(s.COUNTS),timings=dict(s.TIMES),CPU=s.cpu(),wall_seconds=time.monotonic()-s.START,
        RSS_peak_bytes=max(old['RSS_peak_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),original_clock_carried=True))
    result=subprocess.run([sys.executable,str(s.H/'audit154.py')],cwd=s.ROOT,capture_output=True,text=True)
    (s.H/'audit.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
    s.save('cycles/CYCLE3_REVISION.json',dict(status='COMPLETE',operations_logger_fixed=True,science_unchanged=True,completed_fits_replayed=0,native_histories_replayed=0,independent_audit='ACCEPTED'))
    log('CYCLE3_COMPLETE',verdict=verdict,operations_continuation=True)
    s.save('STATUS.json',dict(version='V154',status='THREE_CYCLES_COMPLETE',verdict=verdict,audit='ACCEPTED',package_complete=False,new_core_adopted=False,logging_failure_resolved_without_replay=True))

if __name__=='__main__':
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:main()
    except Exception as e:
        s.save('CONTINUATION_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),counts=dict(s.COUNTS),CPU=s.cpu(),no_retry=True))
        s.save('STATUS.json',dict(version='V154',status='CONTINUATION_FAILURE',error=repr(e)));raise
    finally:
        if (s.H/'OPERATIONS_CONTINUATION.json').exists():
            prior=s.load(s.H/'cycles/logging_failure/COUNTS.json')
            s.save('COUNTS.json',dict(counts=dict(s.COUNTS),timings=dict(s.TIMES),CPU=s.cpu(),wall_seconds=time.monotonic()-s.START,
                RSS_peak_bytes=max(prior['RSS_peak_bytes'],resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),original_clock_carried=True))
        if awake.poll() is None:awake.terminate()
