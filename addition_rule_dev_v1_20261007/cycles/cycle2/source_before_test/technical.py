"""Structural checks and small synthetic fixtures, never science lives."""
import argparse,copy,gzip,hashlib,json,traceback
from pathlib import Path
import numpy as np
import dependencies,settings,data,addition_checkpoint
from learners import build
from representation import receptors

def consume(m,raw):return [m.feed(int(b),m.t+settings.DT) for b in raw]
def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);passed=[]
    codes={arm:build(arm) for arm in settings.ARMS}
    for pair in data.ALL:
        assert np.array_equal(codes['NATIVE'].encoder.code(pair),codes['RESIDUAL'].encoder.code(pair))
    # No cross-role feature depends on the other operand.
    for a in range(5):
        first=receptors((a,0))[:32]
        assert all(np.array_equal(first,receptors((a,b))[:32]) for b in range(5))
    for b in range(5):
        second=receptors((0,b))[32:64]
        assert all(np.array_equal(second,receptors((a,b))[32:64]) for a in range(5))
    passed+=['same_KC_codes_all_25_pairs','independent_operand_features_no_sum']
    for arm,initial in codes.items():
        # Every output channel, including 8, follows the actual teacher byte.
        for digit in settings.ALPHABET:
            m=initial.clone();rows=consume(m,b'1+2='+bytes([digit])+b'\n')
            assert sum(r['answer_update'] for r in rows)==1
            q=next(r['prediction'] for r in consume(m,b'1+2=\n') if r['prediction'] is not None)
            assert q['probabilities'][digit-48]>1/9, (arm,digit,'wrong teaching sign')
        passed.append(arm+'_all_nine_actual_teacher_signs')
        a=initial.clone();b=initial.clone();a.plastic=b.plastic=False
        consume(a,b'1+2=0\n');consume(b,b'1+2=8\n')
        assert a.state_digest()==b.state_digest(),'disabled teacher leaked into state'
        if arm=='RESIDUAL':assert not a.visits.any() and not a.fast.any() and not a.slow.any()
        else:assert all(not s.fly.m.fast.any() and not s.fly.m.slow.any() for s in a.stores)
        passed.append(arm+'_actual_no_write_arrays_and_label_invariance')
        m=initial.clone();before=[a.copy() for a in m.content_arrays()];rows=consume(m,b'1+2=\n')
        assert all(np.array_equal(x,y) for x,y in zip(before,m.content_arrays()))
        assert not any(r['answer_update'] or any(r['write_l1']) for r in rows)
        passed.append(arm+'_missing_answer_no_actual_write')
        m=initial.clone();consume(m,b'0+0=0\n1+0=1\n1+2=')
        path=out/(arm+'_pending.npz');addition_checkpoint.save(m,arm,path);restored=addition_checkpoint.load(path)
        assert restored.state_digest()==m.state_digest()
        r1=consume(m,b'3\n2+2=4\n');r2=consume(restored,b'3\n2+2=4\n')
        assert r1==r2 and m.state_digest()==restored.state_digest()
        m.rest(settings.DAY);restored.rest(settings.DAY)
        assert consume(m,b'1+2=\n')==consume(restored,b'1+2=\n')
        passed.append(arm+'_pending_checkpoint_resumed_equal')
        before=initial.state_digest();c=initial.clone();consume(c,b'1+2=3\n');assert initial.state_digest()==before
        for x,y in zip(initial.content_arrays(),c.content_arrays()):assert not np.shares_memory(x,y)
        assert c.encoder.cache is not initial.encoder.cache
        passed.append(arm+'_clone_learning_and_cache_isolation')
        try: initial.feed(48,0.,truth=3)
        except TypeError:pass
        else:raise AssertionError('model accepted truth metadata')
    # Residual outer-product arithmetic checked independently.
    m=build('RESIDUAL');consume(m,b'1+2=');ids=m.pending['ids'];p=m.pending['p'].copy();z=m.pending['z'].copy()
    teacher=51;y=np.zeros(9);y[teacher-48]=1;delta=settings.ETA*z[:,None]*(y-p)[None,:]
    consume(m,bytes([teacher]));assert np.array_equal(m.fast[ids],np.clip((1-settings.SLOW_SHARE)*delta,-2,2))
    assert np.array_equal(m.slow[ids],np.clip(settings.SLOW_SHARE*delta,-2,2))
    assert np.all(m.visits[ids]==1) and m.visits.sum()==len(ids)
    passed.append('residual_independent_outer_product')
    eager=m.clone();dt=settings.DAY;m.pending=None;eager.pending=None;m.rest(dt);eager.rest(dt)
    eager.fast*=np.exp(-(eager.t-eager.last)[:,None]/settings.FAST_TAU)
    eager.slow*=np.exp(-(eager.t-eager.last)[:,None]/settings.SLOW_TAU);eager.last.fill(eager.t)
    for pair in ((1,2),(4,4),(0,4)):
        p1=m._prediction(pair,m.t)[0];p2=eager._prediction(pair,eager.t)[0];assert np.max(np.abs(p1-p2))<1e-15
    passed.append('lazy_eager_equivalence')
    for invalid in (True,831003,832001,-1):
        try:settings.require_dev(invalid)
        except ValueError:pass
        else:raise AssertionError('unauthorized world admitted')
    passed.append('science_and_extra_DEV_rejected')
    tamper_count=0
    original=out/'RESIDUAL_pending.npz'
    with np.load(original,allow_pickle=False) as z:payload={k:z[k].copy() for k in z.files}
    for kind in ('source','probability','count','row_time'):
        altered={k:v.copy() for k,v in payload.items()};meta=json.loads(altered['metadata'].tobytes())
        if kind=='source':meta['source']['files']['learners.py']='wrong'
        elif kind=='probability':altered['pending_p'][0]+=.01
        elif kind=='count':altered['a3'][0]=65536
        else:altered['a2'][0]=meta['t']+1
        altered['metadata']=np.frombuffer(json.dumps(meta,sort_keys=True).encode(),np.uint8)
        path=out/('tamper_'+kind+'.npz')
        with path.open('wb') as f:np.savez_compressed(f,**altered)
        path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
        try:addition_checkpoint.load(path)
        except ValueError:tamper_count+=1
        else:raise AssertionError('checkpoint tamper accepted')
    return dict(verdict='PASS',checks=passed,checkpoint_tamper_rejected=tamper_count,science_worlds=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    try:r=run(a.out)
    except Exception as exc:
        Path(a.out).mkdir(parents=True,exist_ok=True);r=dict(verdict='FAIL',error=repr(exc),traceback=traceback.format_exc(),science_worlds=0)
        (Path(a.out)/'TECHNICAL_FAILURE.json').write_text(json.dumps(r,indent=2));raise
    (Path(a.out)/'TEST_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
