"""Cycle 1: native bridge, causal contract and strict-gate mechanism fixtures."""
from v51_common import *
from engine import Engine,DT,targets,decode,residual,confirm
from representation import Codes,Sequence,BITS
from v51_core import Learner
import ast,pickle,resource,copy

def run():
    start=time.process_time();save(Path('cycles/cycle1/SOURCE_BEFORE.json'),source_lock())
    checks={};partial=0;records=0;maximum=0.;maximum_ref=0.;rng=np.random.default_rng(5101)
    for n in (5177,65536):
        e=Engine(n,5101,False);ref=e.bank.clone()
        for step in range(6):
            x=np.zeros(n);x[rng.choice(n,max(1,int(n*.02)),replace=False)]=1
            coeff=rng.uniform(-.3,.3,9) if step%2 else np.zeros(9)
            m,z,r=e.margins(x)
            actual,drive=query_batch(ref.fast,ref.slow,ref.adapt,x,DT,ref.Q,ref.reader_Q,ref.MM,ref.k0,ref.ta,ref.tg,ref.tf,ref.ts)
            neutral,drive0=query_batch(np.zeros_like(ref.fast),np.zeros_like(ref.slow),ref.adapt,x,DT,ref.Q,ref.reader_Q,ref.MM,ref.k0,ref.ta,ref.tg,ref.tf,ref.ts)
            ref_m=(actual-neutral).mean(1);maximum_ref=max(maximum_ref,float(abs(m-ref_m).max()))
            assert np.allclose(m,ref_m,atol=1e-12,rtol=0)
            e.tick(x,coeff);partial+=1
            event_batch(ref.fast,ref.slow,ref.adapt,x,DT,coeff,True,ref.Q,ref.T,ref.F,ref.MM,ref.k0,ref.fw0,ref.fwd,ref.ta,ref.tg,ref.tf,ref.ts,ref.fraction,ref.alpha_scale,ref.genes,ref.sf,ref.ss)
            for key in ('fast','slow','adapt'):
                error=float(abs(getattr(e.bank,key)-getattr(ref,key)).max());maximum=max(maximum,error);assert error<1e-9
        e.rest(3600);ref.rest(3600)
        assert np.allclose(e.bank.fast,ref.fast,atol=1e-12,rtol=0)
    checks['native_event_parity_5177_and_65536']=True;checks['max_native_state_error']=maximum
    checks['max_neutral_reference_error']=maximum_ref
    e=Engine(5177,5101);passive=Engine(5177,5101);x=np.zeros(5177);x[100:203]=1
    m,z,r=e.margins(x);assert np.max(abs(m))==0
    y=targets(137);s=residual(m,137,1.)
    e.tick(x,s);passive.tick(x);partial+=2
    after=e.margins(x)[0];control=passive.margins(x)[0]
    assert np.mean((np.tanh(after/2)-y)**2)<np.mean((np.tanh(control/2)-y)**2)
    assert np.all(after*y>0),after
    assert np.all(residual(np.zeros(9),256,1)[:8]==0)
    for i in range(257):assert decode(targets(i))==i
    checks['all_bytes_eos_and_mask']=True;checks['matched_passive_direction_correct']=True
    checks['one_event_correct_direction_margin']=after.tolist()
    # Strict floor, delayed eligibility, replay, mixed signs and repeated prompt.
    g=Engine(64,5101);x=np.ones(64);slow=[]
    for i in range(5):
        g.begin();g.tick(x,np.ones(9)*.1);partial+=1;slow.append(g.slow_write_l1)
        g.end(bytes([i]));g.rest(61)
    assert slow[:4]==[0.]*4,slow
    assert slow[4]>0,slow
    before=g.q.copy();g.begin();g.tick(x,np.ones(9)*.1);partial+=1;g.end(bytes([4]));assert np.array_equal(before,g.q)
    before=g.q.copy();mass=g.slow_write_l1;g.tick(x,np.ones(9)*.1,replay=True);partial+=1;g.end(b'replay',replay=True)
    assert np.array_equal(before,g.q) and g.slow_write_l1==mass
    g.signs.fill(2);before=g.q.copy();g.end(b'mixed');assert np.array_equal(before,g.q)
    g.rest(61);g.signs.fill(-1);g.end(b'contradiction');assert np.mean(g.q)<np.mean(before)
    checks['strict_slow_floor']=True;checks['first_slow_write_after_completed_confirmations']=slow
    checks['repeat_mixed_replay_and_negative_confirmation']=True
    # Generic synthetic identity/class fixture, not learned segmentation evidence.
    codes=Codes(5101);payload=[0,255,137,44,18,203]
    for i,b in enumerate(payload):codes.w[i]=BITS[b]
    codes.bias.fill(0);codes.classes[:]=np.arange(64)%2
    codes.identity_frozen=codes.context_frozen=True
    state=Sequence(codes);previous=None
    for b in payload:
        old=list(state.open);old_class=state.last_class;state.feed(b)
        assert state.open[-1]==b
        if old_class is not None and codes.byte_class(b)!=old_class:assert state.closed[0]==old
    state.begin_reply();assert state.reply==0
    state.feed_output(18);assert state.reply==1
    checks['boundary_byte_retained_and_reply_counter_separate']=True
    # Prepared addresses are target-independent; full record fixtures use a tiny
    # structural population, never graded as the 65,536-cell candidate.
    model=Learner(codes,256,5101,1/16,1024);model.geometry.frozen=True
    e1=model.engine.clone();e2=model.engine.clone();s1=model._prompt(b'\x00\xff',e1);s2=model._prompt(b'\x00\xff',e2)
    assert np.array_equal(model.geometry.encode(s1)[0],model.geometry.encode(s2)[0])
    for p,a in [(b'\x00\xff',b'\x80'),(b'\x89',b'')]:model.train(p,a);records+=1
    before=model.digest();model.generate(b'\x00\xff');model.generate(b'\x89');assert model.digest()==before
    restored=pickle.loads(pickle.dumps(model));model.train(b'\x00',b'\xff');restored.train(b'\x00',b'\xff');records+=2
    assert model.digest()==restored.digest()
    nw=Learner(codes,256,5101,1/16,1024);nw.geometry.frozen=True;nw.train(b'\x00',b'\xff',write=False);records+=1
    assert np.max(abs(nw.engine.bank.fast))==np.max(abs(nw.engine.bank.slow))==np.max(abs(nw.engine.q))==0
    assert model.generate(bytes(257))['error']=='INPUT_LIMIT'
    checks['readonly_query_and_checkpoint_exact']=True;checks['target_independent_frame']=True;checks['no_write_all_value_and_confirmation_zero']=True
    checks['input_output_bounds']=True
    for f in ['engine.py','representation.py','v51_core.py']:
        tree=ast.parse((HERE/f).read_text());imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
            if isinstance(node,ast.ImportFrom):imports.append(node.module)
        assert not set(imports)&{'data','torch','jax','tensorflow','sklearn'}
    checks['no_task_import_or_gradient_framework']=True
    out=dict(accepted=True,scope='E0 plus directional/structural fixtures; not taught-reply qualification or E3',checks=checks,
        full_record_applications=records,partial_native_events=partial,scientific_records=0,
        CPU_seconds=time.process_time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_after=source_lock())
    save(Path('cycles/cycle1/AUDIT.json'),out)
    print(json.dumps({k:v for k,v in out.items() if k!='source_after'}),flush=True)
    return out
if __name__=='__main__':run()
