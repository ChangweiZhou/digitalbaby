import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from mechanism_model import System,ARMS,digest
from assay import decompose,source_hashes,world_doc
from fixture import DT,RECORD_SECONDS

@pytest.fixture(scope='module')
def base():return System('FE0')

def show(s,cue=b'        0+1=',i=0):
    at=i*RECORD_SECONDS;s.begin_cue(at)
    for k,b in enumerate(cue):s.cue_byte(b,at+k*DT)
    t=at+12*DT
    u,r,e,_=s.read(t);dw=s.observe_cue(t,r)
    return t,u,r,e,dw

def teach(s,answer,i,write=True,cue=b'        0+1='):
    t,u,r,e,dw=show(s,cue,i)
    log=s.teach(answer,t,write,290099,i,r,e)
    s.end_record((i+1)*RECORD_SECONDS,i*RECORD_SECONDS+13*DT)
    return u,log,e

def test_upstream_closure():assert source_hashes()['a3_source_lock']

def test_decomposition():
    rng=np.random.default_rng(2);v=rng.normal(size=(4,4,4));labels=np.tile(np.arange(4),4)
    d=decompose(v,labels);assert d['reconstruction_max']<1e-14
    x=decompose(v-np.arange(4),labels)
    for n in ('g1','g2','g12'):assert abs(d[n]-x[n])<1e-14

def test_branch_fixture():
    d=world_doc(290000,2);assert len(d['records'])==64
    assert all(r['domain']=='fact' for r in d['records'])
    assert sorted(d['old_fact']['labels'])==[0]*4+[1]*4+[2]*4+[3]*4

@pytest.mark.parametrize('arm',ARMS)
def test_clone_readonly_and_no_write(base,arm):
    s=base.clone();s.arm=arm;b=s.state_digest();c=s.clone()
    assert b==c.state_digest()
    _,log,_=teach(c,48,0,False)
    assert log['native_l1']==[0]*4 and log['j_write_l1']==0
    assert s.state_digest()==b
    assert not np.shares_memory(s.jw,c.jw)
    assert s.state_budget()['added_allocated_bytes']==c.state_budget()['added_allocated_bytes']

@pytest.mark.parametrize('arm',('T','J','J_ADD','J_SHUFFLE'))
def test_previous_teacher_cannot_reach_sensory_state(base,arm):
    a=base.clone();b=base.clone();a.arm=b.arm=arm
    for i,cue in enumerate((b'        0+1=',b'        2+3=',b'        1+2=')):
        _,_,ea=teach(a,48,i,cue=cue)
        _,_,eb=teach(b,51,i,cue=cue)
        assert np.array_equal(ea,eb)
        assert a.sensory_digest()==b.sensory_digest()

def test_disabled_timing_exact_parity(base):
    a=base.clone();b=base.clone();b.arm='T_OFF'
    for i in range(2):
        ua,_,_=teach(a,48+i,i);ub,_,_=teach(b,48+i,i)
        assert np.array_equal(ua,ub)
        for x,y in zip(a.stores,b.stores):assert x.state_digest()==y.state_digest()

def test_timing_support_budget_and_nonzero(base):
    s=base.clone();s.arm='T';B=s.stores[0].fly.m.B.copy()
    teach(s,48,0);teach(s,48,1,cue=b'        2+3=')
    out=s.stores[0].fly.m.B
    assert np.array_equal(B.indices,out.indices) and np.array_equal(B.indptr,out.indptr)
    assert np.max(np.abs(np.asarray(B.sum(0))-np.asarray(out.sum(0))))<1e-10
    assert np.any(B.data!=out.data)

def test_homeostasis_only_own_activity(base):
    a=base.clone();a.arm='H';b=a.clone()
    t,_,r,_,_=show(a);t2,_,r2,_,_=show(b)
    assert np.array_equal(a.h,b.h)
    assert np.array_equal(a.h,r*a.p['homeo_beta'])
    # Held-out label changes cannot enter this method: it only takes time/raw activity.
    before=a.h.copy();a.read(t);assert np.array_equal(a.h,before)

def test_j_write_disabled_not_hidden(base):
    s=base.clone();s.arm='J';teach(s,48,0,False)
    assert np.all(s.jw==0)
    teach(s,48,1,True);assert np.any(s.jw!=0)
    old=s.jw.copy();last=s.jlast;teach(s,51,2,False)
    decay=np.exp(-(s.jlast-last)/s.p['j_tau'])
    assert np.allclose(s.jw,old*decay,rtol=1e-14,atol=1e-14)

def test_timing_within_first_record_and_order(base):
    a=base.clone();b=base.clone();a.arm=b.arm='T'
    native=a.stores[0].fly.m.B.data.copy()
    teach(a,48,0,cue=b'        0+1=');teach(b,48,0,cue=b'        1+0=')
    assert np.any(native!=a.stores[0].fly.m.B.data)
    assert np.any(a.stores[0].fly.m.B.data!=b.stores[0].fly.m.B.data)
    a.begin_cue(RECORD_SECONDS)
    assert np.all(a.tpre==0) and np.all(a.tpost==0) and np.all(a.telig==0)

def test_homeostasis_tensor_invariant_same_frozen_state(base):
    from assay import probe
    s=base.clone();s.arm='H'
    for i in range(3):teach(s,48+i,i)
    f=world_doc(290099,2)['old_fact']
    o=probe(s,f['cues_hex'],f['labels'],3*RECORD_SECONDS)
    for key in ('g1','g2','g12'):
        assert abs(o['decomposition'][key]-o['raw_decomposition'][key])<1e-12

def test_j_clipping_ledger_uses_actual_step(base):
    s=base.clone();s.arm='J';s.p={**s.p,'j_bound':1e-8}
    _,log,_=teach(s,48,0)
    assert log['j_clipped_coordinates']>0
    assert abs(log['j_write_l1']-np.abs(s.jw).sum())<1e-12
    assert log['j_proposed_l1']>log['j_write_l1']

def test_disabled_timing_delayed_checkpoint_parity(base):
    from assay import probe
    a=base.clone();b=base.clone();b.arm='T_OFF'
    for i,cue in enumerate((b'        0+1=',b'        3+2=',b'        1+3=')):
        teach(a,48+i,i,cue=cue);teach(b,48+i,i,cue=cue)
    at=3*RECORD_SECONDS+86400
    a.flush(at);b.flush(at)
    f=world_doc(290099,2)['old_fact']
    aa=probe(a,f['cues_hex'],f['labels'],at);bb=probe(b,f['cues_hex'],f['labels'],at)
    assert aa['values']==bb['values']

def test_pending_timing_clone_is_independent(base):
    a=base.clone();a.arm='T';show(a)
    assert a.pending_timing is not None
    b=a.clone();assert np.array_equal(a.pending_timing,b.pending_timing)
    assert not np.shares_memory(a.pending_timing,b.pending_timing)
