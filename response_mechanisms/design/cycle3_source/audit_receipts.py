"""Receipt-only verification, deliberately does not trust saved reductions."""
from __future__ import annotations
import gzip,json,math
from pathlib import Path
import numpy as np
from scipy import stats
from assay import ROOT,ARMS,decompose,world_doc


def load(path):return json.loads(gzip.decompress(Path(path).read_bytes()))

def near(a,b,tol=1e-10):
    if isinstance(a,dict):
        if set(a)!=set(b):raise AssertionError('reduction fields changed')
        for k in a:near(a[k],b[k],tol)
    elif isinstance(a,list):
        if len(a)!=len(b):raise AssertionError('reduction length changed')
        for x,y in zip(a,b):near(x,y,tol)
    elif isinstance(a,(float,int)):
        if not math.isfinite(float(a)) or abs(float(a)-float(b))>tol*max(1.,abs(float(b))):
            raise AssertionError('saved reduction mismatch')
    elif a!=b:raise AssertionError('saved metadata mismatch')


def validate(doc,*,expected_world=None,expected_arm=None,expected_bouts=None,expected_sources=None,expected_params=None,expected_runtime=None,lock_sha256=None):
    assert doc['schema']=='RESPONSE-MECHANISMS-RECEIPT-v1'
    assert doc['arm'] in ARMS
    if expected_world is not None:assert doc['world']==expected_world
    if expected_arm is not None:assert doc['arm']==expected_arm
    if expected_bouts is not None:assert doc['bouts']==expected_bouts
    if expected_sources is not None:assert doc['source_hashes']==expected_sources
    if expected_params is not None:assert doc['params']==expected_params
    if expected_runtime is not None:assert doc['runtime']==expected_runtime
    if lock_sha256 is not None:assert doc['lock_sha256']==lock_sha256
    fixture=world_doc(doc['world'],doc['bouts'])
    assert doc['fixture']==fixture
    rows=doc['records'];assert len(rows)==len(fixture['records'])*2
    for i,r in enumerate(rows):
        ref=fixture['records'][i//2];branch=('W','N_old')[i%2]
        assert r['record']==ref['index'] and r['branch']==branch
        assert r['cue_hex']==ref['cue_hex'] and r['answer']==ref['answer'] and r['stage']==ref['stage']
        permission=not(branch=='N_old' and r['stage']=='old')
        assert r['write']==int(permission)
        if not permission:assert r['native_l1']==[0.]*4 and r['j_write_l1']==0.
        assert np.isfinite(r['values']).all() and np.isfinite(r['raw_values']).all()
        assert r['predicted']==int(np.argmax(r['values']))
        assert len(r['native_l1'])==4 and min(r['native_l1'])>=0
        if i%2:assert r['sensory_before_teacher']==rows[i-1]['sensory_before_teacher']
        assert r['j_max']<=doc['params']['j_bound']+1e-12
        if doc['params']['revision']>=3:
            from fixture import RECORD_SECONDS,DT
            start=ref['index']*RECORD_SECONDS+(86400 if r['stage']=='new' else 0)
            assert r['teacher_at']==start+12*DT and r['end_at']==start+RECORD_SECONDS
            raw=np.array(r['raw_values'])
            expected=raw-np.array(r['homeostasis_before']) if doc['arm']=='H' else raw+np.array(r['extra_values']) if doc['arm'].startswith('J') else raw
            near(r['values'],expected.tolist())
    expected_phases={'birth','old_end','old_day','new_end','final'}
    assert set(doc['probes'])==expected_phases
    for phase,allbranches in doc['probes'].items():
        groups=('old','new') if phase in ('birth','new_end','final') else ('old',)
        assert set(allbranches)=={'W','N_old','paired_delta'}
        for branch in ('W','N_old'):
            assert set(allbranches[branch])==set(groups)
            for group in groups:
                o=allbranches[branch][group];labels=fixture[group+'_fact']['labels']
                if doc['params']['revision']>=3:
                    n=16*doc['bouts'];R=165.
                    expected_at={'birth':0.,'old_end':n*R,'old_day':n*R+86400,
                                 'new_end':2*n*R+86400,'final':2*n*R+172800}[phase]
                    assert o['probe_at']==expected_at and o['response_at']==expected_at+12*(30/14)
                for key in ('values','raw_values','extra_values'):
                    assert np.asarray(o[key]).shape==(16,4) and np.isfinite(o[key]).all()
                v=np.array(o['values']);raw=np.array(o['raw_values'])
                assert o['predictions']==v.argmax(1).tolist()
                assert o['accuracy']==float(np.mean(v.argmax(1)==labels))
                assert o['raw_accuracy']==float(np.mean(raw.argmax(1)==labels))
                near(o['decomposition'],decompose(v,labels))
                near(o['raw_decomposition'],decompose(raw,labels))
                near(o['decision_decomposition'],decompose(v-v.mean(1,keepdims=True),labels))
                if doc['arm']=='H':
                    near(v.tolist(),(raw-np.array(o['homeostasis'])).tolist())
                    for k in ('g1','g2','g12'):near(o['decomposition'][k],o['raw_decomposition'][k])
                elif doc['arm'].startswith('J'):
                    near(v.tolist(),(raw+np.array(o['extra_values'])).tolist())
                else:near(v.tolist(),raw.tolist())
                if doc['arm']=='J_ADD':
                    assert o['feature_rank']<=7
                    assert decompose(o['extra_values'],labels)['g12']<1e-10
        for group in groups:
            delta=np.array(allbranches['W'][group]['values'])-np.array(allbranches['N_old'][group]['values'])
            d=allbranches['paired_delta'][group]
            near(d['values'],delta.tolist())
            near(d['decomposition'],decompose(delta,fixture[group+'_fact']['labels']))
            near(d['decision_decomposition'],decompose(delta-delta.mean(1,keepdims=True),fixture[group+'_fact']['labels']))
    assert doc['runtime']['python']=='3.11.15' and doc['runtime']['numpy']=='2.2.6'
    assert doc['runtime']['scipy']=='1.14.1' and doc['runtime']['numba']=='0.61.2'
    assert doc['runtime']['pandas']=='2.2.3'
    assert set(doc['runtime']['thread_environment'].values())=={'1'}
    if doc['params']['revision']>=3:
        assert doc['predictor_frozen'] is True
        assert len(doc['births'])==4
        for b in doc['births']:
            assert b['canonical_B_sha256']=='32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964'
        for b in doc['end_timing']:
            assert b['support_unchanged'] and b['incoming_mass_max_error']<1e-9
            assert b['finite_positive']
    return True


def audit_suite(kind,worlds,arms=ARMS,expected_bouts=None,expected_sources=None):
    params=runtime=locksha=None
    if kind=='final':
        from locked_run import verify_lock
        lock,locksha=verify_lock()
        if list(worlds)!=lock['worlds'] or list(arms)!=lock['arms']:
            raise AssertionError('final audit requires exact ordered complete roster')
        if expected_bouts is not None and expected_bouts!=lock['bouts']:raise AssertionError('wrong final dose')
        expected_bouts=lock['bouts'];expected_sources=lock['source_hashes']
        params=lock['params'];runtime=lock['expected_receipt_runtime']
        actual={str(p.relative_to(ROOT/'results/final')) for p in (ROOT/'results/final').rglob('*.json.gz')}
        expected={f'{a}/{w}.json.gz' for w in worlds for a in arms}
        expected|={f'replays/{a}/{w}.json.gz' for w,a in lock['replay_jobs']}
        if actual!=expected:raise AssertionError('missing or extra final receipts')
    validated=[];failures=[];byworld={}
    for w in worlds:
        byworld[w]={}
        for arm in arms:
            path=ROOT/'results'/kind/arm/f'{w}.json.gz'
            try:
                d=load(path);validate(d,expected_world=w,expected_arm=arm,expected_bouts=expected_bouts,expected_sources=expected_sources,expected_params=params,expected_runtime=runtime,lock_sha256=locksha)
                byworld[w][arm]=d;validated.append(str(path.relative_to(ROOT)))
            except Exception as exc:failures.append(dict(world=w,arm=arm,error=repr(exc)))
        if 'FE0' in byworld[w] and 'T_OFF' in byworld[w]:
            for p in byworld[w]['FE0']['probes']:
                for b in ('W','N_old'):
                    for group in byworld[w]['FE0']['probes'][p][b]:
                        if byworld[w]['FE0']['probes'][p][b][group]['values']!=byworld[w]['T_OFF']['probes'][p][b][group]['values']:
                            failures.append(dict(world=w,error='T_OFF differs from FE0',phase=p))
    if kind=='final':
        for w,a in lock['replay_jobs']:
            first=load(ROOT/'results/final'/a/f'{w}.json.gz')
            replay=load(ROOT/'results/final/replays'/a/f'{w}.json.gz')
            if first['resource']['process_id']==replay['resource']['process_id']:failures.append(dict(world=w,arm=a,error='replay process reused'))
            first.pop('resource');replay.pop('resource')
            if first!=replay:failures.append(dict(world=w,arm=a,error='replay mismatch'))
    return dict(pass_all=not failures,validated_count=len(validated),validated=validated,failures=failures)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('kind');p.add_argument('world',type=int);p.add_argument('--bouts',type=int)
    a=p.parse_args();out=audit_suite(a.kind,[a.world],expected_bouts=a.bouts)
    print(json.dumps(out,indent=2));raise SystemExit(0 if out['pass_all'] else 1)
