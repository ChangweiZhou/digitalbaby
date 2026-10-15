"""Executable source/math/causality fixtures; every training exposure is charged."""
import copy
import json
import math
import time
import core
import numpy as np
import torch
from io_utils import write, now


def run():
    checks=[]
    exposures=0
    start=time.monotonic()
    C=2*np.eye(4)-1
    u=np.array([.3,-.9,.5,1.7])
    assert np.allclose(core.logits(u,C),u-u.mean())
    assert np.array_equal((C.mean(0)-C[2])/2,np.array([.25,.25,-.75,.25]))
    assert len(set(core.WORDS))==257 and np.all(core.CODE.sum(1)==0)
    checks.append('four-class old logits and centered coefficients; balanced257-code port')
    # Exact native signed source formula and explicit opponent teaching fixture.
    s=core.Compartment()
    x=core.stores.bb.bc.native().model.encode_sparse(s.fly.m,core.stores.bb.bc.R_TABLE[51])
    for b,coefficient in [(1.,0.),(1.,1.),(0.,.25),(0.,-.75)]:
        base=s.fly.clone()
        actual=s.clone()
        actual.pending_t=core.DELTA
        actual.pending_x=x
        actual.teach_signed(b,coefficient,core.DELTA,write=True)
        z,f0,f1=base.clone(),base.clone(),base.clone()
        z.event(core.DELTA,x,0.,False)
        f0.event(core.DELTA,x,0.,True)
        f1.event(core.DELTA,x,1.,True)
        expected=z.m.fast+b*(f0.m.fast-z.m.fast)+coefficient*(f1.m.fast-f0.m.fast)
        assert np.array_equal(actual.fly.m.fast,expected)
        assert np.array_equal(actual.fly.m.slow,z.m.slow+b*(f0.m.slow-z.m.slow)+coefficient*(f1.m.slow-f0.m.slow))
    checks.append('signed teaching exact source state formula including four-class target/non-target')
    for version in core.VERSIONS:
        m=core.create(version,1234)
        native=isinstance(m,core.Native)
        if native:
            stores=[s for bank in m.banks for s in bank]
            assert len(stores)==(32 if version=='RC151-C16' else 16)
            assert not hasattr(m,'w')
            assert len({s.birth['fly_id'] for s in stores})==len(stores)
            for attr in ('fast','slow','adapt'):
                for a,b in zip(stores,stores[1:]):
                    assert not np.shares_memory(getattr(a.fly.m,attr),getattr(b.fly.m,attr))
        else:
            assert 'critic' not in m.p and m.p['V'].shape==(m.width+1,16)
            parent=core.OLD.Learner(m.model,1234)
            for k in m.p:
                if k!='V':
                    assert torch.equal(m.p[k],parent.p[k])
            # Inherited recurrence tested against a separately born parent.
            a,b=m.initial(),parent.initial()
            for token in (257,0,255,258):
                a=m.step(a,torch.tensor([token])); b=parent.step(b,torch.tensor([token]))
                assert torch.equal(m.hidden(a),parent.hidden(b))
            # CE optimizer independently calculated, with no critic or PG path.
            loss=m.ce_loss(b'raw\x00\xff',b'\x80')
            grads=torch.autograd.grad(loss,list(m.p.values()))
            norm=math.sqrt(sum(float((g*g).sum()) for g in grads))
            expected={k:p.detach().clone()-.001*(g*min(1,1/max(norm,1e-30)))/(g.abs()*min(1,1/max(norm,1e-30))+1e-8)
                      for (k,p),g in zip(m.p.items(),grads)}
            m.train(b'raw\x00\xff',b'\x80',True); exposures+=1
            for k in m.p:
                assert torch.allclose(m.p[k],expected[k],rtol=1e-12,atol=1e-12)
            # V gradient finite difference using fixed, task-agnostic byte fixtures.
            loss=m.ce_loss(b'raw\x00\xff',b'\x80')
            g=torch.autograd.grad(loss,m.p['V'])[0]
            index=np.unravel_index(int(g.abs().argmax()),g.shape)
            with torch.no_grad():
                original=float(m.p['V'][index]); eps=1e-6
                m.p['V'][index]=original+eps
                plus=float(m.ce_loss(b'raw\x00\xff',b'\x80'))
                m.p['V'][index]=original-eps
                minus=float(m.ce_loss(b'raw\x00\xff',b'\x80'))
                m.p['V'][index]=original
            assert abs((plus-minus)/(2*eps)-float(g[index]))<1e-7
        m.train(b'1+2=',b'3',True); exposures+=1
        digest=m.digest()
        off=m.train(b'1+2=',b'3',False); exposures+=1
        assert off['forbidden_delta']<=1e-9 and off['actor_steps']==0
        if not native:
            assert digest==m.digest()
        digest=m.digest()
        outputs=m.generate_batch([b'1+2=',b'\x00\xff'],cap=1)
        assert len(outputs)==2 and all(status in ('EOS','TRUNCATED') for _,status in outputs)
        assert m.digest()==digest
        c=m.clone()
        assert c.digest()==digest
        c.train(b'1+2=',b'4',True); exposures+=1
        assert m.digest()==digest and c.digest()!=digest
        checks.append(version+': lineage, generic bytes, forbid-write, inference purity, independent clone')
    # RC shared is independently born, yet identical to N151 under matched bytes.
    n,r=core.create('N151-C16'),core.create('RC151-C16')
    for prompt,answer in [(b'1+2=',b'3'),(b'0+9=',b'9')]:
        n.train(prompt,answer); r.train(prompt,answer); exposures+=2
    for a,b in zip(n.banks[0],r.banks[0]):
        for name in ('fast','slow','adapt'):
            assert np.array_equal(getattr(a.fly.m,name),getattr(b.fly.m,name))
    # Four-class actual readout/update bank, using native source state.
    channels=[core.Compartment() for _ in range(4)]
    for j,s in enumerate(channels):
        s.pending_t=core.DELTA;s.pending_x=x
        s.teach_signed(0.,.25-float(j==2),core.DELTA,write=True)
    values=np.array([float((s.fly.m.expression(core.stores.bb.bc.native().common.observed_activity(s.fly.m,x[None,:]))-
                           s.fly.reader.predict(core.stores.bb.bc.native().common.observed_activity(s.fly.m,x[None,:]))).mean()) for s in channels])
    assert int(core.logits(values,C).argmax())==int(values.argmax())
    checks.append('RC shared bank matches N151 independently; source four-class update/readout compatibility')
    # Split checks are evaluator-side, never imported by the operative learner.
    from data import prepare
    for world in (202610143901,202610143902,202610143903):
        d=prepare(world,128,32)
        assert not set(d['groups']['held']) & (set(d['groups']['old'])|set(d['groups']['new']))
        assert sorted(y for _,y in d['old'])==sorted(y for _,y in d['shuffled'])
    checks.append('whole-orbit split and interval-preserving shuffled-answer controls')
    result=dict(accepted=True,utc=now(),science=False,checks=checks,training_record_applications=exposures,
                native_source_event_fixtures=36,elapsed=time.monotonic()-start,
                note='Source/event fixtures and differentiations are technical computation, not science; smoke arithmetic is not an efficacy estimate.')
    write(core.bootstrap.HERE/'cycles/cycle1/TEST_RESULTS.json',result)
    return result


if __name__=='__main__':
    print(json.dumps(run(),indent=2),flush=True)
