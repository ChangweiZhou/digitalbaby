"""Synthetic qualification only; no admitted DEV worlds or science lives."""
import argparse,importlib.util,json,traceback,hashlib
from pathlib import Path
import numpy as np
import dependencies,settings,data,addition_checkpoint
from learners import build
from temporal import advance

def consume(m,raw):return [m.feed(int(b),m.t+settings.DT) for b in raw]
def run(path):
    out=Path(path);out.mkdir(parents=True,exist_ok=True);checks=[]
    models={a:build(a) for a in settings.ARMS}
    for pair in data.ALL:assert all(np.array_equal(models[a].encoder.code(pair),models['DECAY'].encoder.code(pair)) for a in settings.ARMS)
    checks.append('identical_fixed_codes_all_25_questions')
    rng=np.random.default_rng(20261011);f=rng.uniform(-2,2,(100,1));s=rng.uniform(-2,2,(100,1));dt=rng.uniform(0,1e6,100)
    nf,ns=advance(f,s,dt,'TRANSFER');assert np.max(np.abs(nf+ns-f-s))<1e-15
    assert np.abs(nf).max()<=2 and np.abs(ns).max()<=2
    f1,s1=advance(f,s,dt*.31,'TRANSFER');f2,s2=advance(f1,s1,dt*.69,'TRANSFER')
    assert np.max(np.abs(f2-nf))<1e-14 and np.max(np.abs(s2-ns))<1e-14
    zf,zs=advance(f,s,np.zeros(100),'TRANSFER');assert np.array_equal(zf,f) and np.array_equal(zs,s)
    checks+=['signed_bounded_transfer_conservation','split_time_semigroup','zero_time_identity']
    for ff,ss in [(1.,2.),(-1.,-2.),(1.,1.9),(-1.,-1.9)]:
        a,b=advance(np.array([[ff]]),np.array([[ss]]),np.array([86400.]),'TRANSFER');assert abs(float(a[0,0]+b[0,0])-ff-ss)<1e-15
        assert abs(a[0,0])<=2 and abs(b[0,0])<=2
    checks.append('saturated_slow_capacity_keeps_remainder')
    for invalid in (-1.,float('nan'),float('inf')):
        try:advance(f[:1],s[:1],np.array([invalid]),'TRANSFER')
        except ValueError:pass
        else:raise AssertionError('invalid time admitted')
    checks.append('invalid_elapsed_time_rejected')
    for arm,m0 in models.items():
        for teacher in settings.ALPHABET:
            m=m0.clone();tr=consume(m,b'1+2='+bytes([teacher])+b'\n');assert sum(x['answer_update'] for x in tr)==1
            before=next(x['prediction'] for x in tr if x['prediction'])['probabilities'][teacher-48]
            pred=next(x['prediction'] for x in consume(m,b'1+2=\n') if x['prediction'])
            assert pred['probabilities'][teacher-48]>=before-1e-12
        a=m0.clone();b=m0.clone();a.plastic=b.plastic=False
        consume(a,b'1+2=0\n');consume(b,b'1+2=8\n');assert a.state_digest()==b.state_digest()
        assert not a.fast.any() and not a.slow.any() and not a.visits.any()
        m=m0.clone();before=m.state_digest();consume(m,b'1+2=\n');assert not m.visits.any() and not m.fast.any() and not m.slow.any()
        c=m0.clone();original=m0.state_digest();consume(c,b'1+2=3\n');assert m0.state_digest()==original
        assert all(not np.shares_memory(x,y) for x,y in zip(m0.content_arrays(),c.content_arrays()))
        m=c.clone();m.rest(settings.DAY);digest=m.state_digest();first=m._prediction((1,2),m.t)[0];second=m._prediction((1,2),m.t)[0]
        assert np.array_equal(first,second) and m.state_digest()==digest
        eager=m.clone();f1,s1=advance(eager.fast,eager.slow,eager.t-eager.last,arm);eager.fast[:]=f1;eager.slow[:]=s1;eager.last.fill(eager.t)
        assert np.max(np.abs(m._prediction((1,2),m.t)[0]-eager._prediction((1,2),eager.t)[0]))<1e-14
        m=c.clone();consume(m,b'2+1=');file=out/(arm+'_pending.npz');addition_checkpoint.save(m,arm,file);restored=addition_checkpoint.load(file)
        assert m.state_digest()==restored.state_digest();assert consume(m,b'3\n0+1=1\n')==consume(restored,b'3\n0+1=1\n')
        assert m.state_digest()==restored.state_digest()
        try:m0.feed(48,0.,truth=0)
        except TypeError:pass
        else:raise AssertionError('truth accepted')
        checks.extend([arm+'_actual_teacher_and_no_write',arm+'_clone_query_lazy_eager',arm+'_pending_checkpoint_equality'])
    # Saturating teacher update uses the same bounded content policy in every rule.
    for arm in settings.ARMS:
        m=build(arm);ids=np.flatnonzero((m.encoder.code((4,4))>0)&m.encoder.mask);m.fast[ids]=1.999;m.slow[ids]=1.999
        rows=consume(m,b'4+4=0\n');assert np.abs(m.fast).max()<=2 and np.abs(m.slow).max()<=2
        assert sum(x['answer_update'] for x in rows)==1 and np.isfinite(m.fast).all()
    checks.append('signed_teaching_at_content_bounds')
    from decision import lower_bound
    zero=lower_bound([.3]*64);assert zero['method']=='HOEFFDING_ZERO_VARIANCE' and zero['lower']<.3
    tiny=lower_bound([.3+1e-14*(i%2) for i in range(64)]);assert tiny['method']=='HOEFFDING_ZERO_VARIANCE'
    assert lower_bound([0.]*64)['lower']<0
    nonzero=lower_bound([0.,.2,.5,1.]);assert nonzero['method']=='PAIRED_STUDENT_ONE_SIDED_95'
    checks.append('bounded_confirmation_interval_no_zero_width_claim')
    from runner import save
    target=out/'synthetic_commit.json.gz';save(dict(synthetic=True,science_worlds=0),target)
    try:save(dict(synthetic=True),target)
    except ValueError:pass
    else:raise AssertionError('duplicate receipt overwritten')
    checks.append('duplicate_complete_commit_rejected')
    # Numerical baseline parity against untouched previous source, using the same fixed representation.
    spec=importlib.util.spec_from_file_location('previous_ordinal_learner',dependencies.ROOT.parent/'addition_rule_dev_v1_20261007/learners.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);a=old.Residual();b=build('DECAY')
    for raw in (b'1+2=3\n',b'4+4=8\n',b'2+0=0\n',b'1+0=\n'):
        ra=consume(a,raw);rb=consume(b,raw)
        for xa,xb in zip(ra,rb):
            for key in ('byte','t','pre_line','post_line','observed_target','plastic','answer_update','write_l1'):assert xa[key]==xb[key]
            if xa['prediction']:
                for key in ('emitted','probabilities','values'):assert xa['prediction'][key]==xb['prediction'][key]
        assert all(np.array_equal(x,y) for x,y in zip(a.content_arrays(),b.content_arrays()))
    a.rest(settings.DAY);b.rest(settings.DAY)
    assert np.array_equal(a._prediction((1,2),a.t)[0],b._prediction((1,2),b.t)[0])
    checks.append('DECAY_previous_frozen_equations_bitwise_parity')
    for bad in (True,831201,832100,833001,-1):
        try:settings.require_dev(bad)
        except ValueError:pass
        else:raise AssertionError('extra/science ID admitted')
    checks.append('science_and_extra_DEV_rejected')
    # Rehashed checkpoint corruption must still fail semantic checks.
    with np.load(out/'TRANSFER_pending.npz',allow_pickle=False) as z:original={k:z[k].copy() for k in z.files}
    rejected=[]
    for name in ('source','row_time','visits','pending_value','rule'):
        payload={k:v.copy() for k,v in original.items()};meta=json.loads(payload['metadata'].tobytes())
        if name=='source':meta['source']['files']['temporal.py']='wrong'
        elif name=='row_time':payload['a2'][0]=meta['t']+1
        elif name=='visits':payload['a3'][0]=65536
        elif name=='pending_value':meta['pending']['value']+=1
        else:meta['arm']='STABLE'
        payload['metadata']=np.frombuffer(json.dumps(meta,sort_keys=True).encode(),np.uint8)
        target=out/('tamper_'+name+'.npz')
        with target.open('wb') as f:np.savez_compressed(f,**payload)
        target.with_suffix('.sha256').write_text(hashlib.sha256(target.read_bytes()).hexdigest()+'\n')
        try:addition_checkpoint.load(target)
        except ValueError:rejected.append(name)
        else:raise AssertionError('rehashed checkpoint tamper accepted '+name)
    return dict(verdict='PASS',checks=checks,checkpoint_tamper_rejected=rejected,science_worlds=0,DEV_worlds=0)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    try:result=run(args.out)
    except Exception as exc:
        p=Path(args.out);p.mkdir(parents=True,exist_ok=True);(p/'TECHNICAL_FAILURE.json').write_text(json.dumps(dict(verdict='FAIL',error=repr(exc),traceback=traceback.format_exc(),science_worlds=0),indent=2));raise
    (Path(args.out)/'TEST_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
