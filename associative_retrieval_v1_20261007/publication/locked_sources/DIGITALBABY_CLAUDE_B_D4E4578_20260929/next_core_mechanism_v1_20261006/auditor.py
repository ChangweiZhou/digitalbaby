"""Independent observations of real native transitions and hostile receipts."""
import contextlib,time,types,math
import numpy as np
from assays import make_world,permitted,DT,RECORD_SECONDS
from spec import ALPHABET,SCALES,DOSES,EPSILONS

@contextlib.contextmanager
def instrument(core):
    evidence={'calls':[],'audit_cpu_s':0.}
    changed=[]
    for i,m in enumerate(core.models):
        for name in ('teach_logged','teach_signed'):
            method=getattr(type(m),name)
            def wrapped(self,*args,_method=method,_name=name,_i=i,**kw):
                start=time.process_time();flag=kw['write'];t=args[-1]
                # Compare against the same actual prestate with native plasticity off.
                reference=self.fly.clone();reference.event(self._teach_prologue(t),self.pending_x,0.,False)
                before_call=time.process_time();evidence['audit_cpu_s']+=before_call-start
                value=_method(self,*args,**kw)
                end_call=time.process_time()
                df=self.fly.m.fast-reference.m.fast;ds=self.fly.m.slow-reference.m.slow
                delta=np.concatenate((df.ravel(),ds.ravel()))
                equal=all(np.array_equal(getattr(self.fly.m,k),getattr(reference.m,k)) for k in ('fast','slow','adapt'))
                if not flag and (not equal or value!=0.):raise AssertionError('actual disabled plasticity changed state')
                if not np.array_equal(self.fly.m.adapt,reference.m.adapt) or self.fly.m.elapsed!=reference.m.elapsed:raise AssertionError('native nonplastic clock/adaptation mismatch')
                evidence['calls'].append({'store':_i,'api':_name,'write':flag,'coefficients':list(map(float,args[:-1])),
                    'applied_l1':float(value),'delta_l1':float(abs(delta).sum()),'delta_l2':float(np.linalg.norm(delta)),
                    'delta_max':float(abs(delta).max()),'no_write_reference_equal':bool(equal) if not flag else None})
                evidence['audit_cpu_s']+=time.process_time()-end_call
                return value
            setattr(m,name,types.MethodType(wrapped,m));changed.append((m,name))
    try:yield evidence
    finally:
        for m,n in changed:m.__dict__.pop(n,None)

def check_record(row,arm,event,branch,nstores):
    assert row['index']==event['index'] and row['stage']==event['stage'] and row['item']==event['item']
    flag=permitted(branch,event['stage']);w=row['write'];calls=row['actual_calls']
    assert row['learn'] is flag and w['learn'] is flag and row['prediction_precedes_outcome'] is True
    assert row['predicted_at']==row['observed_at']==event['at']+12*DT
    assert w['outcome']==event['outcome'] and w['time']==row['observed_at']
    assert len(calls)==nstores and [r['store'] for r in calls]==list(range(nstores))
    p=row['prediction'];s=np.asarray(p['shared']);v=np.asarray(p['private']);extra=np.asarray(p['extra'])
    u=s/SCALES[0]+v/SCALES[1]+(extra/SCALES[0] if len(extra) else 0)
    assert np.array_equal(np.asarray(p['combined']),u)
    assert p['emitted']==ALPHABET[int(np.argmax(u))]
    logits=v/SCALES[1];pi=np.exp(logits-logits.max());pi/=pi.sum()
    assert np.array_equal(np.asarray(w['private_pre_outcome_probabilities']),pi)
    y=np.array([float(b==event['outcome']) for b in ALPHABET])
    expected=np.full(4,.25)-y if arm=='CENTER' else pi-y
    assert np.array_equal(np.asarray(w['s']),expected)
    for i,r in enumerate(calls):
        assert r['write'] is flag
        if i<4 or i>=8:
            assert r['api']=='teach_logged' and r['coefficients']==[float(ALPHABET[i%4]!=event['outcome'])]
        else:assert r['api']=='teach_signed' and r['coefficients']==[0.,float(expected[i-4])]
        assert all(math.isfinite(r[k]) and r[k]>=0 for k in ('applied_l1','delta_l1','delta_l2','delta_max'))
        if not flag:
            assert r['no_write_reference_equal'] is True
            assert r['applied_l1']==r['delta_l1']==r['delta_l2']==r['delta_max']==0
    assert row['flush_error']<1e-8 and row['online_cpu_s']>0
    if arm in EPSILONS or arm.startswith('S3_'):
        assert len(w['adaptations'])==4
        for a in w['adaptations']:assert a['hooks']==row['index']+1

def audit_receipt(d,identity,complete=True):
    assert d['source_identity']==identity and d['development'] is True
    w=make_world(d['world'],d['assay']);assert d['fixture_sha256']==w['sha256']
    n=12 if d['arm'] in DOSES else 8
    branches=w['branches'] if complete else d['limits']['branches']
    assert set(d['branches'])==set(branches)
    limit=len(w['events']) if complete else d['limits']['records']
    for branch,b in d['branches'].items():
        assert len(b['records'])==limit and b['records_completed']==limit
        assert len(b['births'])==n and len({z['fly_id'] for z in b['births']})==n
        assert len({z['canonical_B_sha256'] for z in b['births']})==1
        for e,r in zip(w['events'][:limit],b['records']):check_record(r,d['arm'],e,branch,n)
        for p in b['probes']:
            assert p['state_before']==p['state_after']
            for r in p['rows']:
                assert r['first_pre_feedback'] is True
                assert r['correct']==int(r['emitted']==r['target'])
        assert b['final_time']==(w['clocks']['final'] if complete else w['events'][limit-1]['at']+RECORD_SECONDS)
    if d['arm'] in EPSILONS or d['arm'] in ('S3_CUE','S3_RAND'):
        # These mechanisms read bytes/clock/x, not supervised value states.
        ref=d['branches']['W']['records']
        for branch,b in d['branches'].items():
            for a,r in zip(ref,b['records']):assert a['write']['adaptations']==r['write']['adaptations']
    assert d['peak_rss_bytes']>0 and d['cpu_s']>0 and d['worker_s']>0
    return {'verdict':'PASS','world':d['world'],'arm':d['arm'],'assay':d['assay'],'lives':len(branches),'records':limit*len(branches)}
