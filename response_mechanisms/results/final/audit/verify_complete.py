"""Independent, complete-only receipt/statistical/operational audit. Never simulates.

Run only after parent confirms all 224 primary lives and final locked analysis.
Outputs a separate audit JSON; never rewrites science, receipts, ledger or report.
"""
from __future__ import annotations
import argparse, fcntl, gzip, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path('/workspace/scratch/cbb599bc73b9/minifly-response/response_mechanisms')
LOCK_SHA = '24943155d4b010c8e17746a265803155e13585a7b0b23fe261f3c0bae95a63fe'
PARALLEL_SHA = 'f1a775bbecbad25d0adafbd2dcf96cebe3d1a49ee86176cfd69555d16511f06a'
START = 1790872804.660057
DEADLINE = START + 43200
PRIOR_CHARGE = 17245.33994293213
PRIMARY = [('T','T_OFF'),('H','FE0'),('J','J_ADD'),('J','J_SHUFFLE')]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def receipt(path): return json.loads(gzip.decompress(Path(path).read_bytes()))
def equal(a, b, path='root'):
    if isinstance(a, dict):
        assert set(a) == set(b), ('keys', path, set(a), set(b))
        for key in a: equal(a[key], b[key], path+'/'+str(key))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), ('length',path)
        for i,(x,y) in enumerate(zip(a,b)): equal(x,y,path+'/'+str(i))
    elif a is None or isinstance(a,(str,bool)):
        assert a == b, (path,a,b)
    else:
        assert math.isfinite(float(a)) and math.isfinite(float(b)), ('nonfinite',path)
        assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10), (path,a,b)


def components(values):
    v=np.asarray(values,dtype=float).reshape(4,4,4)
    base=np.average(v,axis=(0,1))
    first=np.average(v,axis=1)-base
    second=np.average(v,axis=0)-base
    interaction=v-first[:,None,:]-second[None,:,:]-base
    return v,base,first,second,interaction


def decomposition(values, labels):
    v,b,f,g,h=components(values)
    target=-np.ones((16,4));target[np.arange(16),np.asarray(labels,dtype=int)]=1
    _,tb,tf,tg,th=components(target)
    rms=lambda x: float(np.sqrt(np.average(np.square(x))))
    aligned={}
    for name,c,t in zip(('B','F','G','H'),(b-b.mean(),f,g,h),(tb-tb.mean(),tf,tg,th)):
        norm=rms(t);cov=float(np.average(c*t))
        aligned[name]=dict(target_rms=norm,covariance=cov,projection=cov/norm if norm>1e-12 else None)
    g1,g2,g12=[rms(x) for x in (f,g,h)]
    return dict(bias_rms=rms(b-b.mean()),g1=g1,g2=g2,g12=g12,
        g12_g1=g12/g1 if g1>1e-12 else None,g12_g2=g12/g2 if g2>1e-12 else None,
        reconstruction_max=float(np.max(np.abs(v-(b+f[:,None,:]+g[None,:,:]+h)))),
        target_alignment=aligned,components={n:x.tolist() for n,x in zip(('B','F','G','H'),(b,f,g,h))})


def summary(values):
    a=np.asarray(values,dtype=float);assert len(a)>0 and np.isfinite(a).all()
    mean=float(np.average(a))
    if len(a)==1:return dict(n=1,mean=mean,sd=None,ci95=None,p_two_sided=None)
    sd=float(np.std(a,ddof=1))
    if sd==0: ci=[mean,mean];p=1. if mean==0 else 0.
    else:
        ci=list(map(float,stats.t.interval(.95,len(a)-1,loc=mean,scale=stats.sem(a))))
        p=float(stats.ttest_1samp(a,0).pvalue)
    return dict(n=len(a),mean=mean,sd=sd,ci95=ci,p_two_sided=p)


def optional(values):
    finite=[v for v in values if v is not None]
    return dict(total=len(values),undefined=len(values)-len(finite),statistics=summary(finite) if finite else None)


def holm(values):
    ranked=sorted(enumerate(values),key=lambda pair:pair[1]);result=[None]*len(values);largest=0.
    for rank,(idx,value) in enumerate(ranked):
        largest=max(largest,(len(values)-rank)*value);result[idx]=min(1.,largest)
    return result


def independent_metrics(doc):
    # Recompute EVERY saved reduction directly, without the experiment's decompose().
    for phase,branches in doc['probes'].items():
        for branch in ('W','N_old'):
            for group,o in branches[branch].items():
                labels=doc['fixture'][group+'_fact']['labels'];v=np.asarray(o['values']);raw=np.asarray(o['raw_values'])
                equal(o['accuracy'],float(np.average(v.argmax(1)==labels)))
                equal(o['raw_accuracy'],float(np.average(raw.argmax(1)==labels)))
                equal(o['predictions'],v.argmax(1).tolist())
                equal(o['decomposition'],decomposition(v,labels))
                equal(o['raw_decomposition'],decomposition(raw,labels))
                equal(o['decision_decomposition'],decomposition(v-v.mean(axis=1,keepdims=True),labels))
        for group,o in branches['paired_delta'].items():
            labels=doc['fixture'][group+'_fact']['labels']
            delta=np.asarray(branches['W'][group]['values'])-np.asarray(branches['N_old'][group]['values'])
            equal(o['values'],delta.tolist());equal(o['decomposition'],decomposition(delta,labels))
            equal(o['decision_decomposition'],decomposition(delta-delta.mean(axis=1,keepdims=True),labels))
    final=doc['probes']['final'];old=final['W']['old'];labels=doc['fixture']['old_fact']['labels']
    v=np.asarray(old['values']);raw=np.asarray(old['raw_values']);n=np.asarray(final['N_old']['old']['values'])
    delta=v-n;learned=decomposition(delta-delta.mean(axis=1,keepdims=True),labels)
    raw_d=decomposition(raw,labels);dec_d=decomposition(v-v.mean(axis=1,keepdims=True),labels)
    accuracy=lambda x:float(np.average(x.argmax(1)==labels))
    return dict(accuracy=accuracy(v),raw_accuracy=accuracy(raw),bias=dec_d['bias_rms'],raw_bias=raw_d['bias_rms'],
        H_projection=learned['target_alignment']['H']['projection'],
        **{k:learned[k] for k in ('g1','g2','g12','g12_g1','g12_g2')},
        centered_accuracy=accuracy(raw-raw.mean(axis=0)),raw_g2_g1=raw_d['g2']/raw_d['g1'] if raw_d['g1']>1e-12 else None,
        benefit=accuracy(v)-accuracy(n),retention_loss=accuracy(v)-accuracy(np.asarray(doc['probes']['old_day']['W']['old']['values'])),
        birth_H=decomposition(np.asarray(doc['probes']['birth']['W']['old']['values'])-np.asarray(doc['probes']['birth']['W']['old']['values']).mean(axis=1,keepdims=True),labels)['g12'])


def numerical(docs, out):
    computed={a:[independent_metrics(d) for d in rows] for a,rows in docs.items()}
    for arm,rows in computed.items():
        for name in rows[0]: equal(out['arms'][arm][name],optional([r[name] for r in rows]),arm+'/'+name)
        equal(out['world_level'][arm],[dict(world=d['world'],metrics=m) for d,m in zip(docs[arm],rows)],arm+'/world_level')
    primary=[]
    for a,b in PRIMARY:
        primary.append(dict(candidate=a,control=b,**summary([x['accuracy']-y['accuracy'] for x,y in zip(computed[a],computed[b])])))
    for row,p in zip(primary,holm([r['p_two_sided'] for r in primary])):row['p_holm']=p
    equal(out['primary'],primary,'primary')
    mechanisms={}
    for a,b in PRIMARY:
        mechanisms[a+'-'+b]={}
        for name in ('g1','g2','g12','g12_g1','g12_g2','H_projection','bias','benefit'):
            mechanisms[a+'-'+b][name]=optional([None if x[name] is None or y[name] is None else x[name]-y[name] for x,y in zip(computed[a],computed[b])])
    equal(out['mechanistic_contrasts'],mechanisms,'mechanisms')
    behavioral=[r['mean']>0 and r['p_holm']<.05 for r in primary]
    def signature(a,b,name):
        s=mechanisms[a+'-'+b][name]
        return s['undefined']==0 and s['statistics'] is not None and s['statistics']['ci95'] is not None and s['statistics']['ci95'][0]>0
    floor={name:np.mean([r[name] for r in computed['T']])>=.9*np.mean([r[name] for r in computed['T_OFF']]) for name in ('g1','g2')}
    floor={k:bool(v) for k,v in floor.items()}
    h=summary([r['bias']-r['raw_bias'] for r in computed['H']])
    qualification=dict(T=dict(behavior_pass=behavioral[0],both_main_effects_at_least_90_percent=floor,qualified=behavioral[0] and all(floor.values()) and all(signature('T','T_OFF',m) for m in ('g12_g1','g12_g2','g12','H_projection'))),
        H=dict(behavior_pass=behavioral[1],within_state_bias_change=h,qualified=behavioral[1] and h['ci95'][1]<0),
        J=dict(behavior_vs_additive=behavioral[2],behavior_vs_shuffle=behavioral[3],qualified=behavioral[2] and behavioral[3] and all(signature('J',b,m) for b in ('J_ADD','J_SHUFFLE') for m in ('g12','H_projection'))))
    equal(out['qualification'],qualification,'qualification')
    return dict(independent_reductions=True,independent_primary=primary,independent_qualification=qualification,
                mechanistic_components_remain_nominal_supportive=True)


def operational(root, docs, replays, out):
    dest=root/'results/final';h=read(dest/'RUN_LEDGER.json');status=read(dest/'RUN_STATUS.json')
    assert status['state']=='complete_pending_final_audit' and status['completed']==224
    assert not h.get('active_job') and not h.get('active_jobs') and not h.get('terminal_failure')
    assert h['lock_sha256']==LOCK_SHA and h['started_unix']==START
    assert h['original_elapsed_deadline_unix']==DEADLINE and h['effective_worker_hours_cap']==12 and h['effective_workers']==2
    assert h['original_worker_hours_cap_superseded']==8 and h['original_workers_cap_superseded']==1
    assert sha(root/'ops/recover_parallel.py')==PARALLEL_SHA==h['operational_launcher_sha256']
    assert all(math.isfinite(j['seconds']) and j['seconds']>=0 for j in h['jobs'])
    equal(math.fsum(j['seconds'] for j in h['jobs']),h['worker_seconds'],'ledger sum')
    assert h['worker_seconds']<=43200
    prior=read(dest/'operations/PRE_RESET_REMOTE_LEDGER.json');loss=read(dest/'operations/RESET_RECOVERY_20261001.json')
    equal(h['jobs'][:14],prior['jobs']);assert prior['started_unix']==START
    equal(loss['accounting']['conservative_prior_charge_seconds'],PRIOR_CHARGE)
    before_parallel=list((dest/'operations').glob('PRE_PARALLEL_LEDGER-*.json'))
    assert before_parallel, 'Missing immutable pre-parallel ledger'
    for path in before_parallel:
        old=read(path);assert old['started_unix']==START
        equal(h['jobs'][:len(old['jobs'])],old['jobs'],'preserved pre-parallel ledger')
    adjustments=[j for j in h['jobs'] if j.get('accounting_only')]
    assert len(adjustments)==1
    equal(prior['worker_seconds']+adjustments[0]['seconds'],PRIOR_CHARGE,'prior conservative baseline')
    auth=h['parallel_authorization'];approval=auth['approval'];review=auth['independent_review']
    assert approval['approved'] is True and approval['effective_workers']==2 and approval['effective_worker_hours_cap']==12
    assert approval['operational_launcher_sha256']==review['operational_launcher_sha256']==PARALLEL_SHA
    assert approval['lock_sha256']==LOCK_SHA and approval['original_started_unix']==START and approval['deadline_unix']==DEADLINE
    assert review['pass_all'] is True and approval['user_approval_reference'] and review['review_reference']
    assert (root/'RECOVERY_AMENDMENT_20261001.md').exists() and (root/'PARALLEL_AMENDMENT_20261001.md').exists()
    assert 'Ran 24 tests' in (root/'ops/PARALLEL_MOCK_TEST_RESULTS.txt').read_text()
    successful=[j for j in h['jobs'] if j.get('exit_code')==0 and not j.get('reason')]
    failed=[j for j in h['jobs'] if j.get('exit_code') not in (None,0) or (j.get('reason') and not j.get('accounting_only') and not j.get('unobserved_interruption'))]
    assert not failed, failed
    assert all(j['seconds']<=300 for j in successful)
    expected={(d['world'],d['arm'],False) for rows in docs.values() for d in rows}|{(d['world'],d['arm'],True) for d in replays}
    actual=[(j['world'],j['arm'],j['replay']) for j in successful]
    assert len(actual)==len(set(actual))==231 and set(actual)==expected
    parallel=[j for j in successful if j.get('parallel_recovery')]
    for j in parallel:
        assert j['reserved_seconds']==300 and j['effective_workers']==2 and j['receipt_validated'] is True
        assert j['operational_launcher_sha256']==PARALLEL_SHA and j['observed_peak_rss_bytes']<=800000000
        assert START<=j['started_unix']<=j['finished_unix']<=DEADLINE
        assert (dest/j['stdout']).exists() and (dest/j['stderr']).exists()
    # finished_unix is validation completion, so these are conservative reservation intervals.
    events=sorted([(j['started_unix'],1) for j in parallel]+[(j['finished_unix'],-1) for j in parallel])
    concurrent=peak=0
    for _,delta in events:concurrent+=delta;peak=max(peak,concurrent)
    assert peak<=2
    all_docs=[d for rows in docs.values() for d in rows]+replays
    assert all(0<=d['resource']['wall_seconds']<=300 and 0<=d['resource']['peak_rss_bytes']<=800000000 for d in all_docs)
    primary_seconds=math.fsum(d['resource']['wall_seconds'] for rows in docs.values() for d in rows)
    replay_seconds=math.fsum(d['resource']['wall_seconds'] for d in replays)
    equal(out['resources']['total_worker_seconds'],primary_seconds)
    equal(out['resources']['maximum_peak_rss_bytes'],max(d['resource']['peak_rss_bytes'] for rows in docs.values() for d in rows))
    size=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file());assert size<=200000000
    provenance={}
    def inspect(value):
        if isinstance(value,dict):
            name=value.get('path') or value.get('file');expected_hash=value.get('sha256')
            if isinstance(name,str) and name.endswith('.json.gz') and isinstance(expected_hash,str):
                path=root.parent/name
                assert path.is_relative_to(dest) and path.exists(), ('provenance path',name)
                assert sha(path)==expected_hash, ('changed inspected receipt',name)
                provenance[name]=expected_hash
            for child in value.values():inspect(child)
        elif isinstance(value,list):
            for child in value:inspect(child)
    for path in (dest/'publication_inspection').glob('*.json'):inspect(read(path))
    assert len(provenance)>=14, 'Missing byte-integrity evidence for original primary/replay checkpoint'
    interruptions=[j for j in h['jobs'] if j.get('unobserved_interruption')]
    return dict(ledger_total_seconds=h['worker_seconds'],ledger_hours=h['worker_seconds']/3600,
        prior_conservative_seconds=PRIOR_CHARGE,primary_receipt_seconds=primary_seconds,replay_receipt_seconds=replay_seconds,
        successful_ledger_jobs=len(successful),recovery_primary_lives=217,retained_original_primary_lives=7,
        completed_job_max_seconds=max(j['seconds'] for j in successful),receipt_peak_rss_bytes=max(d['resource']['peak_rss_bytes'] for d in all_docs),
        parallel_reservation_interval_peak=peak,results_bytes_before_audit_output=size,
        prior_inspection_receipt_hashes_verified=len(provenance),
        interruptions=interruptions,original_start=START,original_deadline=DEADLINE,
        audit_observed_unix=time.time(),audit_before_original_deadline=time.time()<=DEADLINE,
        limitations=['Lost pre-reset job durations/individual cap compliance cannot be reconstructed',
          '374.0029s historical interruption charge included downtime and is not measured life runtime',
          'Historical single-worker jobs lack exact start timestamps; universal historical concurrency is not independently established',
          'Primary receipt timing excludes replay, interrupted/lost-work accounting and process overhead',
          'Final sample is original 7 plus recomputed 217 fixed-roster primary receipts; lost earlier 202-count is not additional data',
          'Userspace cap monitoring does not prove continuous RSS bounds between samples; completed receipt peaks and retained measurements were checked'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    root=ROOT;dest=root/'results/final';lock=read(root/'SOURCE_LOCK.json')
    assert sha(root/'SOURCE_LOCK.json')==LOCK_SHA
    expected={f'{a}/{w}.json.gz' for w in lock['worlds'] for a in lock['arms']}|{f'replays/{a}/{w}.json.gz' for w,a in lock['replay_jobs']}
    actual={str(x.relative_to(dest)) for x in dest.rglob('*.json.gz')}
    assert actual==expected and len(actual)==231, 'STOP: requires exact COMPLETE 224+7 roster before reading efficacy'
    assert (dest/'FINAL_METRICS.json').exists() and (dest/'REPORT.md').exists(), 'STOP: locked final analysis not available'
    guard=(root/'scratch/response-supervisor.lock').open('r')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        sys.path.insert(0,str(root/'src'))
        from audit_receipts import audit_suite
        from analyze import render
        audit=audit_suite('final',lock['worlds'],expected_bouts=6)
        assert audit['pass_all'] and audit['validated_count']==224
        docs={a:[receipt(dest/a/f'{w}.json.gz') for w in lock['worlds']] for a in lock['arms']}
        replays=[receipt(dest/'replays'/a/f'{w}.json.gz') for w,a in lock['replay_jobs']]
        out=read(dest/'FINAL_METRICS.json');assert out['kind']=='final' and out['exploratory'] is False
        equal(out['worlds'],lock['worlds']);assert out['bouts']==6
        assert render(out)==(dest/'REPORT.md').read_text(), 'Final report differs from locked render'
        nums=numerical(docs,out);ops=operational(root,docs,replays,out)
        report=dict(pass_all=True,scientific_lock_sha256=LOCK_SHA,audit_helper_sha256=sha(__file__),
            complete_roster=231,locked_receipt_audit=audit,independent_numerical=nums,operational=ops,
            termination_evidence='Exclusive inherited coordinator flock acquired; final status complete; no ledger active jobs. Parent must additionally confirm executor session returned.',
            publication='Not established by this local audit; parent must attach exact current verified GitHub and private-backup evidence separately')
        target=Path(args.output);assert not target.exists(),'Write-once audit output already exists'
        with target.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps(dict(pass_all=True,output=str(target),ledger_hours=ops['ledger_hours'],primary=nums['independent_primary'])))
    finally:guard.close()

if __name__=='__main__':main()
