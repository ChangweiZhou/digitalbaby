"""Three technical cycles, with fixed synthetic records and no scientific exposure."""
import copy
import io
import json
from pathlib import Path
import resource
import time
from head import Factorized, Learner, T, joint, error
from qualify import DecaySensor, teach, np
from generation import generate

def save(name,value):
    Path(name+'.json').write_text(json.dumps(value,indent=2)+'\n')

def main():
    start=time.process_time()
    rng=T.Generator().manual_seed(187)
    raw=T.randn(5,257,generator=rng,dtype=T.float64,requires_grad=True)
    y=T.tensor([0,255,256,2,256])
    lp=joint(raw)
    assert T.allclose(lp.exp().sum(1),T.ones(5,dtype=raw.dtype))
    (-lp[T.arange(5),y].mean()).backward()
    diff=float((raw.grad-error(raw.detach(),y)).abs().max())
    assert diff<1e-12
    assert not T.count_nonzero(error(raw.detach(),y)[y==256,:256])
    extreme=T.full((2,257),1000.);extreme[0,256]=-1000.
    assert T.isfinite(joint(extreme)).all()
    save('CYCLE1',{'status':'PASS','max_reference_gradient_error':diff})
    sensor=DecaySensor()
    rows=[bytes(range(64)),b'abc\0\xff',b'']
    outcomes={}
    records=targets=0
    for kind in (Learner,Factorized):
        net=kind()
        assert sum(p.numel() for p in net.p.values())==37313
        for phase in (rows,[b'intervening']):
            targets+=teach(net,sensor,.5,32,phase);records+=len(phase)
        checkpoint=io.BytesIO()
        T.save(net.state(),checkpoint)
        checkpoint.seek(0)
        state=T.load(checkpoint,weights_only=True)
        children=[]
        for payload in (b'original',b'corrected',b'original'):
            child=kind()
            child.restore(state)
            targets+=teach(child,sensor,.5,32,[payload]);records+=1
            children.append(child)
        for attr in ('p','m','v'):
            assert all(T.equal(v,getattr(children[2],attr)[k]) for k,v in getattr(children[0],attr).items())
        assert children[0].clock==children[2].clock and T.equal(children[0].rng.get_state(),children[2].rng.get_state())
        assert not T.equal(children[0].p['Ob'],children[1].p['Ob'])
        frozen=copy.deepcopy(net.state())
        def next_class(code):
            return int(net.forward(T.from_numpy(code[None,:]))[0].argmax())
        payload,eos=generate(next_class,sensor,b'abc',.5)
        assert len(payload)+int(eos)<=64
        assert all(T.equal(p,frozen['weights'][k]) for k,p in net.p.items())
        assert T.equal(frozen['dropout_rng'],net.rng.get_state())
        outcomes[kind.__name__]={'checkpoint_bytes':len(checkpoint.getvalue()),'actual_fixture_own_EOS':eos}
    save('CYCLE2',{'status':'PASS','arms':outcomes,'fixture_records':records,'fixture_targets':targets})
    save('CYCLE3',{'status':'COMPONENT_FIXTURE_PASS_CAMPAIGN_NOT_ADMITTED',
                   'scientific_records':0,'cpu_seconds':time.process_time()-start,
                   'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
if __name__=='__main__':
    main()
