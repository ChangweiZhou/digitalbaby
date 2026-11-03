import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

from head import Factorized, Learner, T, joint, error
from qualify import DecaySensor, teach
from generation import generate

def audit_head():
    x=T.randn(4,257,generator=T.Generator().manual_seed(187),dtype=T.float64,requires_grad=True)
    for labels in ([0,255,256,256],[256]*4,[0,1,2,255]):
        y=T.tensor(labels)
        lp=joint(x)
        T.testing.assert_close(lp.exp().sum(1),T.ones(4,dtype=x.dtype))
        ref=T.autograd.grad(-lp[T.arange(4),y].mean(),x)[0]
        T.testing.assert_close(ref,error(x.detach(),y),atol=1e-12,rtol=0)
    parent,candidate=Learner(),Factorized()
    for key in parent.p:
        if key!='Ob':
            assert T.equal(parent.p[key],candidate.p[key])
    assert abs(float(candidate.p['Ob'][256]-parent.p['Ob'][256])+__import__('math').log(256))<1e-6
    return {'gradient_cases':3,'learned_scalars':sum(p.numel() for p in candidate.p.values())}

def stop_score(reference,prefix,payload,eos,cap=64):
    expected=len(reference)-len(prefix)
    if not 0<=expected<cap:
        return 'ineligible',False
    if not eos:
        return 'truncated',False
    if len(payload)<expected:
        return 'early',False
    if len(payload)>expected:
        return 'late',False
    return 'on_length',prefix+payload==reference

def audit_decoder():
    sensor=DecaySensor()
    for reference,prefix,choices,status,exact in [
        (b'ab',b'a',[98,256],'on_length',True),
        (b'ab',b'a',[99,256],'on_length',False),
        (b'ab',b'a',[256],'early',False),
        (b'ab',b'a',[98,99,256],'late',False),
        (b'',b'',[256],'on_length',True),
        (bytes(63),b'',[0]*63+[256],'on_length',True),
        (bytes(64),b'',[0]*64,'ineligible',False),
        (b'ab',b'a',[0]*64,'truncated',False)]:
        stream=iter(choices)
        payload,eos=generate(lambda _:next(stream),sensor,prefix,.5)
        assert stop_score(reference,prefix,payload,eos)==(status,exact)
    return {'cases':8,'scientific_records':0}

def checkpoint(model,arm):
    stream=io.BytesIO()
    T.save({'arm':arm,'decay':.5,'batch':32,'state':model.state()},stream)
    return stream.getvalue()

def restore(model,blob,arm):
    value=T.load(io.BytesIO(blob),weights_only=True)
    if (value['arm'],value['decay'],value['batch'])!=(arm,.5,32):
        raise ValueError('checkpoint configuration mismatch')
    model.restore(value['state'])

def integration():
    sensor=DecaySensor()
    receipts=[]
    records=targets=0
    for world in ('fixture1','fixture2'):
        for cls in (Learner,Factorized):
            arm=cls.__name__
            net=cls(world)
            for phase in ([bytes(range(40)),b'ab'],[b'bc']):
                targets+=teach(net,sensor,.5,32,phase)
                records+=len(phase)
            blob=checkpoint(net,arm)
            try:
                restore(net,blob,'wrong-arm')
            except ValueError:
                pass
            else:
                raise AssertionError('cross-arm restore accepted')
            forks=[]
            for data in (b'abc',b'abd',b'abc'):
                child=cls(world)
                restore(child,blob,arm)
                targets+=teach(child,sensor,.5,32,[data])
                records+=1
                forks.append(checkpoint(child,arm))
            assert forks[0]==forks[2] and forks[0]!=forks[1]
            assert checkpoint(net,arm)==blob
            assert max(map(len,[blob,*forks]))<=2**20
            payload,eos=generate(lambda code:int(net.forward(T.from_numpy(code[None,:]))[0].argmax()),
                                 sensor,b'a',.5)
            outcome,exact=stop_score(b'ab',b'a',payload,eos)
            receipts.append({'world':world,'arm':arm,'stop':outcome,'exact':exact})
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/'report.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('receipts.json',json.dumps(receipts))
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and json.loads(z.read('receipts.json'))==receipts
    return {'lives':len(receipts),'records':records,'targets':targets,'archive_readback':True,
            'scientific_records':0}

if __name__=='__main__':
    cycle=int(sys.argv[1])
    result=[audit_head,audit_decoder,integration][cycle-1]()
    Path(f'cycles/{cycle}_test.json').write_text(json.dumps({'status':'PASS',**result},indent=2)+'\n')
