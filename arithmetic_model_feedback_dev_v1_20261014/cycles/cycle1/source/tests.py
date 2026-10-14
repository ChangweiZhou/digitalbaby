"""Numerical, byte, causal and archive tests; no arithmetic model training."""
import ast,importlib.util,json,tempfile,time
from pathlib import Path
import torch
from core import Learner,MODELS,ROOT,EOS,S

def run():
    checks=[];fixtures=0
    def ok(name,**data):checks.append(dict(name=name,**data))
    for model in MODELS:
        c=Learner(model,31);ref=S.Learner('B24',31)
        for k,p in ref.net.p.items():assert torch.equal(p,c.net.p[k])
        prompts=[b'',b'\x00\xff\n=',bytes(range(30))]
        assert c.generate_batch(prompts)==ref.generate_batch(prompts)
        ok(model+' source birth and zero-condition own generation parity')
        before=c.digest();clone=c.clone()
        assert all(c.net.p[k].data_ptr()!=clone.net.p[k].data_ptr() for k in c.net.p)
        clone.net.p['V'].data[-1,2]+=1;assert c.digest()==before and clone.digest()!=before
        ok(model+' independent mutable state')
        for token in (0,10,32,61,255,EOS):
            with torch.no_grad():c.net.p['V'].zero_();c.net.p['V'][-1,token]=100
            out=c.generate_batch([b'\x00\xff'],2)[0]
            assert out==(b'','EOS') if token==EOS else out==(bytes([token,token]),'TRUNCATED')
        ok(model+' unrestricted payload and own EOS versus cap')
        c=Learner(model,21)
        with torch.no_grad():c.net.p['V'].copy_(torch.randn(c.net.p['V'].shape,generator=torch.Generator().manual_seed(17),dtype=torch.float64)*.15)
        actor={k:p for k,p in c.net.p.items() if k!='critic'}
        for objective in ('CE','PG'):
            def loss():
                return c.ce_loss(b'\x00\xff\x07=',b'\x0a\x3d') if objective=='CE' else -.5*c.sequence_logp(b'\x00\xff\x07=',[10,61,EOS])
            l=loss();grads=torch.autograd.grad(l,list(actor.values()))
            for (k,p),g in zip(actor.items(),grads):
                index=int(g.abs().reshape(-1).argmax());value=float(p.detach().reshape(-1)[index]);delta=2e-6
                with torch.no_grad():p.reshape(-1)[index]=value+delta
                plus=float(loss().detach())
                with torch.no_grad():p.reshape(-1)[index]=value-delta
                minus=float(loss().detach())
                with torch.no_grad():p.reshape(-1)[index]=value
                fd=(plus-minus)/(2*delta);an=float(g.reshape(-1)[index])
                assert abs(fd-an)<1e-7+1e-4*abs(an),(model,objective,k,fd,an)
                ok(model+' '+objective+' derivative '+k,finite_difference=fd,analytic=an)
        # Critic value has no representation gradient; PG has no critic gradient.
        _,_,cache=c.act(b'abc',5,True)
        vg=torch.autograd.grad(c.critic(cache['c'])[0],list(actor.values()),allow_unused=True)
        assert all(g is None for g in vg)
        assert torch.autograd.grad(cache['logp'],c.net.p['critic'],allow_unused=True)[0] is None
        ok(model+' critic detach and action-independent baseline credit')
        before=c.digest();a,s,cache=c.act(b'abc',9,False);c.update(b'abc',None,0,cache,'SCALAR',False)
        assert before==c.digest();ok(model+' no-write freezes actor critic moments and counters')
        a,s,cache=c.act(b'abc',9,True)
        try:c.update(b'abc',b'forbidden',0,cache,'SCALAR');raise AssertionError('hidden target accepted')
        except ValueError:pass
        ok(model+' scalar phase rejects answer bytes')
        a=c.clone();b=c.clone();aa,sa,ca=a.act(b'abc',191,True);ab,sb,cb=b.act(b'abc',191,True)
        assert (aa,sa)==(ab,sb)
        a.update(b'abc',None,1,ca,'SCALAR');b.update(b'abc',None,1,cb,'SCALAR')
        assert a.digest()==b.digest();ok(model+' scalar update depends only on actual action and given reward')
        before=c.digest();rng=torch.random.get_rng_state().clone()
        x=c.generate_batch(prompts);y=[c.generate_batch([p])[0] for p in prompts]
        assert x==y and c.digest()==before and torch.equal(rng,torch.random.get_rng_state())
        ok(model+' batched serial parity and inference purity')
        with tempfile.TemporaryDirectory(dir=ROOT/'ops') as tmp:
            p=Path(tmp)/'state.npz';a.save(p);b=Learner.load(p);assert a.digest()==b.digest()
            assert all(a.net.p[k].data_ptr()!=b.net.p[k].data_ptr() for k in a.net.p)
        ok(model+' complete state archive roundtrip')
    # Source-faithful CE actor update, despite a separately updated critic.
    c=Learner('R0',81);ref=S.Learner('B24',81)
    for i in range(5):
        prompt=bytes([i,255-i]);reply=bytes([21+i,10])
        a,s,cache=c.act(prompt,i,False);c.update(prompt,reply,int(a==reply and s=='EOS'),cache,'WARM');ref.observe(prompt,reply,True);fixtures+=1
        for k,p in ref.net.p.items():
            assert torch.allclose(c.net.p[k],p,atol=1e-12,rtol=1e-12)
            assert torch.allclose(c.mom[k],ref.mom[k],atol=1e-12,rtol=1e-12)
    ok('R0 faithful CE RMS actor source parity with separate critic')
    # Fixed generic acquisition diagnostic; no unbounded retry to get a pass.
    fixtures_rows=[(bytes([i*31,255-i]),bytes([201-i])) for i in range(8)]
    acquisition={}
    for model in MODELS:
        c=Learner(model,109)
        for i in range(4096):
            p,a=fixtures_rows[i%8];got,status,cache=c.act(p,777+i,False);c.update(p,a,int(got==a and status=='EOS'),cache,'WARM');fixtures+=1
        generated=c.generate_batch([p for p,a in fixtures_rows]);count=sum(x==(a,'EOS') for x,(p,a) in zip(generated,fixtures_rows))
        acquisition[model]=dict(correct=count,total=8,outputs=[dict(actual=a.hex(),status=s) for a,s in generated])
        assert count==8,(model,'generic acquisition failed',acquisition[model])
        ok(model+' fixed-budget generic taught-byte acquisition',correct=count,total=8)
    tree=ast.parse((ROOT/'core.py').read_text());modules={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
    assert not any(x and ('world' in x or 'audit' in x or 'protocol' in x) for x in modules)
    ok('learner has no evaluator or arithmetic module import')
    return dict(accepted=True,checks=checks,check_count=len(checks),generic_record_applications=fixtures,
        acquisition=acquisition,scope='numerical/source/byte/learning-signal fixtures, not arithmetic efficacy')
if __name__=='__main__':
    t=time.time();r=run();r['wall']=time.time()-t;(ROOT/'CONTRACT_TESTS.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('accepted','check_count','generic_record_applications','acquisition','wall')}))
