"""Cycle-1 independent fixtures: audit rejection, serialization and gradients."""
import copy
import json
import sys
import time
import numpy as np
from v144_audit import HERE, ROOT, Ledger, source_lock, validate, write_json
from reference import ByteTrace, Reference, bits, decode
from task import world
import native_diagnostic as native


def main():
    ledger=Ledger('cycle1_checks_qualified','A'); tests={}
    for token in range(257):assert decode(bits(token))==token
    tests['257_token_roundtrip']=True
    assert decode(np.ones(9))==511
    tests['invalid_codes_unmasked']=True
    trace=ByteTrace(native.lb1.BC.R_TABLE,native.lb1.fastmem.PERM,native.lb1.DELTA,native.lb1.fastmem.FEPerm)
    for prompt in [b'5+2=',b'\x00 =\xff',b'']:
        assert np.array_equal(trace.context(prompt),native.lb1.trace_for(prompt))
    tests['parent_trace_exact']=True
    # Incremental feed independently uses parent FE-PERM API.
    f=native.lb1.fastmem.FEPerm();f.t=0;t=0
    for b in b'x=\x00':t+=native.lb1.DELTA;f.advance(t);f.feed(b,t)
    t+=2*native.lb1.DELTA;f.advance(t)
    for b in b'\xff =':f.feed(b,t);t+=native.lb1.DELTA;f.advance(t)
    assert np.allclose(f.read(),trace.context(b'x=\x00',b'\xff ='),atol=2e-15,rtol=0)
    tests['answer_prefix_trace_parity']=True
    model=Reference(trace,'fixture',hidden=8);model.U[:]=np.random.default_rng(4).normal(0,.1,model.U.shape)
    p=trace.context(b'\x00=\xff');dw,db,du=model.gradients(p,256)
    def loss():
        m=model.margins(p);y=(bits(256)+1)/2
        return float(np.sum(np.logaddexp(0,m)-y*m))
    errors=[]
    for array,grad in [(model.W,dw),(model.U,du),(model.b,db)]:
        ix=np.unravel_index(np.abs(grad).argmax(),grad.shape)
        assert abs(grad[ix])>1e-5,'finite-difference fixture must exercise a nonzero gradient'
        orig=array[ix];eps=1e-5;array[ix]=orig+eps;a=loss();array[ix]=orig-eps;b=loss();array[ix]=orig
        errors.append(abs((a-b)/(2*eps)-grad[ix]))
    assert max(errors)<1e-7,errors
    tests['BP_finite_difference_max_error']=max(errors)
    for mode in ['BP','DFA','FROZEN']:
        m=Reference(trace,'fixture',hidden=8,mode=mode);before=m.state_hash();m.train(b'a',b'\x00\xff')
        assert np.array_equal(m.U,np.zeros_like(m.U)), 'first record must have zero ramp'
        assert m.records==1 and m.target_tokens==3
        before=m.state_hash();m.generate(b'a');assert m.state_hash()==before
    tests['warmup_and_nonmutating_generation']=True
    # Next-token oracle is test plumbing only, never wired into Reference.
    for payload in [b'',b'\x00',b'\xff =',b'\x00\xff =ab']:
        seq=[decode(bits(t)) for t in (*payload,256)]
        assert bytes(seq[:-1])==payload and seq[-1]==256 and len(seq)<=8
    tests['arbitrary_payload_EOS_fixture']=True
    w=world('V144-EXERCISE-C2'); assert len(w['held'])==15
    assert all(not(set(w['held']) & set(d)) for d in w['doses'].values())
    tests['nested_split_no_overlap']=True
    tests['native_instrumentation']=native.parity_fixture()
    lock=source_lock(HERE/'cycles/CYCLE1_SOURCE_LOCK.json')
    good=[{'id':k,'complete':True,'metrics':{'x':1}} for k in ['LEARN','FROZEN']]
    assert validate(lock,good,['LEARN','FROZEN'])['valid']
    rejects=[]
    for name,rows,lk,closure in [
        ('missing_control',good[:1],lock,True),('extra',good+[{'id':'other'}],lock,True),
        ('duplicate',good+good[:1],lock,True),('closure',good,lock,False),
        ('hash',good,{'files':{'minifly_v144_dev_20261024/checks.py':'bad'}},True),
        ('metrics',[{'id':x,'complete':True} for x in ['LEARN','FROZEN']],lock,True),
        ('partial',[dict(r,complete=False) for r in good],lock,True),
        ('one_world_substitute',good,lock,True)]:
        expected=['LEARN','FROZEN','second_world'] if name=='one_world_substitute' else ['LEARN','FROZEN']
        v=validate(lk,rows,expected,closure)
        assert v['verdict']=='INVALID_FOR_VERDICT',name;rejects.append(name)
    tests['fail_closed_mutations']=rejects
    assert not validate(lock,good,['LEARN','FROZEN'],expected_settings={'hidden':128})['valid']
    tests['invalid_settings_rejected']=True
    write_json(HERE/'cycles/CYCLE1_CHECKS.json',tests);ledger.emit('complete',tests=len(tests),synthetic_native_records=16,synthetic_reference_records=3,synthetic_reference_target_tokens=9)
    print(json.dumps(tests,indent=2))


if __name__=='__main__':main()
