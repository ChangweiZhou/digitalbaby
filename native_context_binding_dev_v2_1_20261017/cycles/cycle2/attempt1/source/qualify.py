"""Three bounded technical cycles. No science-world admission or launch API."""
import argparse
import ast
import copy
import json
import math
import os
import resource
import shutil
import time
import traceback
from pathlib import Path
import bootstrap
import numpy as np
import torch
from bind_kernel import Learner, ARMS, CODE, MU, EOS, DELTA
from bind_native import State,tensor
from bind_artifacts import Ledger,write,source_identity
import bind_evaluator as evaluator
import protocol

TECH_WORLDS=(202610177001,202610177002,202610177003)

def close(a,b,label,atol=1e-9):
    if isinstance(a,torch.Tensor):a=a.detach().numpy()
    if isinstance(b,torch.Tensor):b=b.detach().numpy()
    np.testing.assert_allclose(a,b,rtol=1e-9,atol=atol,err_msg=label)
    return float(np.max(np.abs(a-b)))

def state_close(a,b):
    return {k:close(getattr(a,k),getattr(b,k),k) for k in ('fast','slow','adapt')}

def reject(fn):
    try:fn()
    except (ValueError,AssertionError):return True
    raise AssertionError('tamper unexpectedly accepted')

def raw_records():
    a=(b'\x00\x81?A',b'\x93');b=(b'\xff\x0a!B',b'\x2b')
    return [a,b,a,b]

def cycle1(ledger,out,checks):
    world=TECH_WORLDS[0]
    old=bootstrap.source_module('native_bind_v1_parity','native_context_binding_dev_v1_20261016/bind_kernel.py')
    model=Learner('NATIVE',world);parent=old.Learner('NATIVE',world)
    assert len(model.births)==32
    assert len({model.state.fast[b,j].data_ptr() for b in range(2) for j in range(16)})==32
    checks['independent_mutable_stores']=32
    for allowed in (True,False):
        ledger.call(f'new-parity-{allowed}',model,lambda:model.train_window(raw_records(),allowed))
        ledger.call(f'parent-parity-{allowed}',parent,lambda:parent.train_window(raw_records(),allowed))
        checks[f'parent_{allowed}']=state_close(model.state,parent.state)
        with torch.no_grad():
            checks[f'MEM_parity_{allowed}']=close(model.forecast(),parent.forecast(),'MEM')
            checks[f'LEGACY_parity_{allowed}']=close(model.forecast('LEGACY'),parent.forecast('LEGACY'),'LEGACY')
    def sign_fixtures():
        anatomy=model.anatomy
        state=State(torch.zeros_like(model.state.fast),torch.zeros_like(model.state.slow),torch.ones_like(model.state.adapt))
        x=torch.full((2,1,anatomy.N),.002,dtype=torch.float64)
        min_improvement=math.inf
        with torch.no_grad():
            for target in range(257):
                c=(MU-CODE[target])/2
                written,rm,info=anatomy.teach_error(state,x,DELTA,c,True)
                s,dx=anatomy.query_activity(written,x,DELTA)
                bank=(anatomy.expression(s,dx)-anatomy.expression(s,dx,True)).mean(-1)
                v=bank[0]/1.4911274663291492+bank[1]/1.3452365735750882
                logits=CODE@(v-v.mean())/4
                improvement=float(logits[target]-logits.mean())
                assert improvement>0 and int(logits.argmax())==target,(target,improvement)
                min_improvement=min(min_improvement,improvement)
            zero,_,_=anatomy.teach_error(state,x,DELTA,torch.zeros(16),True)
            denied,_,_=anatomy.teach_error(state,x,DELTA,torch.ones(16),False)
            sn,_,_=anatomy.event(state,x,DELTA,torch.zeros((2,16)),False)
            state_close(zero,sn);state_close(denied,sn)
        return dict(all257_target_margins_positive=True,all257_argmax_correct=True,
                    min_target_margin=improvement,minimum_over_targets=min_improvement,zero_error_equals_passive=True)
    checks['ERROR_sign']=ledger.fixture('257-target independent-activity equation fixtures',sign_fixtures,native_events=257*3+5)
    model=Learner('F111',world)
    assert sum(p.numel() for p in model.p.values())==22016
    with torch.no_grad():
        model.prompt(b'\x00?')
        frame=model.prediction_frame();x_before=frame['x'].clone()
        assert not frame['consumed']
        model.event(0x93,0x93,True,frame)
        assert frame['consumed'] and torch.equal(frame['x'],x_before)
        assert reject(lambda:model.event(0x93,0x93,True,frame))
        stale=model.prediction_frame();model.rest(1)
        assert reject(lambda:model.event(0x93,0x93,True,stale))
        other=model.clone();frame=model.prediction_frame()
        assert reject(lambda:other.event(0x93,0x93,True,frame))
        assert reject(lambda:model.event(1,1,True))
        assert reject(lambda:model.event(1,None,True))
    checks['prediction_frame_tamper_rejected']=5
    # Generic byte access, not additional reinforced records.
    @torch.no_grad()
    def io_fixture():
        model.prompt(bytes(range(256)))
        assert all(int((CODE@CODE[y]).argmax())==y for y in range(257))
        before=model.digest();model.generate(b'\x00\n =\xff',cap=1)
        assert before==model.digest()
        return dict(payload_bytes=256,classes=257,readonly=True)
    checks['byte_IO']=ledger.fixture('all256 payload plusEOS and readonly generation',io_fixture,native_events=258,payload_bytes=256)
    for arm in ARMS:
        model=Learner(arm,world)
        result=ledger.call(f'{arm}-one-ABAB-window',model,lambda:model.train_window(raw_records()))
        assert result['records']==4 and result['applied_update']==model.learns
        assert all(t['frame_address_reused'] and t['prefeedback'] for t in model.last_trace)
        if model.learns:assert result['gradient_norm']>0
    checks['all9_short_windows']=True
    # Syntax/import boundary is checked on the actual source, not a name-only manifest.
    for name in ('bind_kernel.py','bind_native.py'):
        tree=ast.parse((bootstrap.HERE/name).read_text())
        imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
        assert not imports&{'bind_evaluator','protocol','tasks','data'}
    checks['no_task_import_in_kernel']=True

def manual_credit_probe(model,retain):
    """Lambda1 changes no forward value; isolates first-record fact-state credit."""
    lam=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    read_loss=None
    for index,(prompt,answer) in enumerate(raw_records()):
        model.prompt(prompt)
        for position,target in enumerate((*answer,EOS)):
            frame=model.prediction_frame()
            if index==2 and position==0:
                read_loss=torch.nn.functional.cross_entropy(frame['logits'][None,:],torch.tensor([target]))
            model.event(target,target,True,frame)
        model.resource['records']+=1
        if index==0:
            # Birth fast/slow are exactly0: these arrays contain only first-record fact writes.
            model.state=State(model.state.fast*lam,model.state.slow*lam,model.state.adapt)
        if not retain:model.state=model.state.detached()
    grad=torch.autograd.grad(read_loss,lam,allow_unused=True)[0]
    value=0. if grad is None else float(grad)
    model.state=model.state.detached();model.h=model.h.detach()
    return value

def cycle2(ledger,out,checks):
    world=TECH_WORLDS[1]
    for cut_name,through_name in [('F000','F001'),('F110','F111')]:
        cut=Learner(cut_name,world);full=Learner(through_name,world)
        a=ledger.call(cut_name+'-cut-no-step',cut,lambda:cut.train_window(raw_records(),apply_update=False))
        b=ledger.call(through_name+'-through-no-step',full,lambda:full.train_window(raw_records(),apply_update=False))
        state_close(cut.state,full.state)
        assert cut.digest()==full.digest()
        difference=math.sqrt(sum(float(((cut.last_grads[k]-full.last_grads[k])**2).sum()) for k in cut.p))
        assert difference>1e-10
        probes=[]
        for arm,retain in [(cut_name,False),(through_name,True)]:
            m=Learner(arm,world)
            probes.append(ledger.call(arm+'-first-write-only-lambda',m,lambda:manual_credit_probe(m,retain)))
        assert probes[0]==0 and abs(probes[1])>1e-12,probes
        checks[cut_name+'_credit']=dict(forward_identical=True,gradient_difference=difference,first_fact_write_gradient=probes)
    no_write={}
    for arm in ARMS:
        model=Learner(arm,world)
        ledger.call(arm+'-nonzero-preparation',model,lambda:model.train_window(raw_records()))
        before=model.theta_digest();state=model.state.detached(True);time0=model.time
        denied=ledger.call(arm+'-nonzero-denied',model,lambda:model.train_window(raw_records(),False))
        expected=model.anatomy.rest(state,model.time-time0)
        assert before==model.theta_digest() and not denied['applied_update']
        close(model.state.fast,expected.fast,'nonzero denied fast')
        close(model.state.slow,expected.slow,'nonzero denied slow')
        no_write[arm]=True
    checks['all9_nonzero_denial']=no_write
    # Independent fixed-activity derivative checks; keep cached pi constant.
    m=Learner('F111',world);a=m.anatomy
    fixed=State(torch.zeros_like(m.state.fast),torch.zeros_like(m.state.slow),torch.ones_like(m.state.adapt))
    x=torch.full((2,1,a.N),.002,dtype=torch.float64,requires_grad=True);c=(MU-CODE[0x93])/2
    def objective(xx):
        state,_,_=a.teach_error(fixed,xx,DELTA,c,True)
        return state.fast.square().sum()+state.slow.square().sum()
    g=torch.autograd.grad(objective(x),x)[0];flat=int(g.abs().argmax());idx=np.unravel_index(flat,tuple(x.shape))
    aa=x.detach().clone();bb=aa.clone();eps=1e-7;aa[idx]+=eps;bb[idx]-=eps
    finite=float((objective(aa)-objective(bb))/(2*eps));error=abs(finite-float(g[idx]))/max(1,abs(finite),abs(float(g[idx])))
    assert error<1e-5
    checks['fixed_pi_write_derivative_relative_error']=error
    for group,dose,paired,tail in [('old',768,720,48),('new',256,240,16),('old',1536,1512,24),('new',512,504,8)]:
        rows=evaluator.stream(world,group,dose);counts={p:sum(pp==p for pp,_ in rows) for p,_ in rows}
        assert max(counts.values())-min(counts.values())<=1
        for j in range(0,paired,4):
            w=[p for p,_ in rows[j:j+4]];assert w[0]==w[2] and w[1]==w[3] and w[0]!=w[1]
        assert len(rows[paired:])==tail
        for j in range(paired,dose,4):assert len({p for p,_ in rows[j:j+4]})==4
    checks['ABAB_dose_tail_and_floor_ceil']=True
    groups=evaluator.split(world);assert not set(groups['old'])&set(groups['held']) and not set(groups['new'])&set(groups['held'])
    for a,b in groups['held']:assert (b,a) in groups['held']
    checks['orbit_exposure_isolation']=True
    # Both interfaces use the literal promised formula; zero-support distinction is real.
    p=torch.tensor([0.,1.,0.,.5],dtype=torch.float64);eta=torch.tensor([.2,-.1,.1,-.2])
    mult=p*torch.exp(eta);add=torch.relu(p+eta)
    assert mult[0]==0 and add[0]>0
    checks['additive_can_activate_zero_PN']=True

def synthetic_protocol_checks():
    root=dict(G=.2,Q=.2,OLD_FINAL=.5,G_N_OLD=0,G_SHORTCUT=.05,HELD_FINAL=.2,HELD_N_OLD_FINAL=0,HELD_SHORTCUT_FINAL=.05,
              OLD_END=.5,OLD_REST=.5,NEW_END=.5,NEW_REST=.5,OLD_END_CONTROL=.1,OLD_REST_CONTROL=.1,NEW_END_CONTROL=.1,NEW_REST_CONTROL=.1)
    scores={a:dict(root,G=.1,Q=.1) for a in ARMS};scores['F000']=root.copy()
    assert protocol.select(scores)=='F000'
    for arm in protocol.FACTORIAL:
        trial=copy.deepcopy(scores)
        if arm!='F000':trial[arm].update(G=.4,Q=.4)
        assert protocol.select(trial)==arm
        roster=protocol.confirm_roster(arm);assert 2<=len(roster)<=6 and len(set(roster))==len(roster)
    bad=copy.deepcopy(scores);bad['F000']['HELD_N_OLD_FINAL']=.2
    assert protocol.select(bad) is None
    keys=['G_NATIVE','Q_NATIVE','G_CAUSAL','HELD_CAUSAL','HELD_SHORTCUT','OLD_END_CAUSAL','OLD_END_MODE',
          'OLD_REST_CAUSAL','OLD_REST_MODE','OLD_FINAL_MODE','NEW_END_MODE','NEW_REST_MODE','OLD_PRE_RETAIN','OLD_FINAL_RETAIN','G_ROOT','Q_ROOT']
    rows=[dict(world=j,**{k:.2 for k in keys}) for j in range(16)]
    assert protocol.confirm_gates('F111',rows)['passed']
    for key in keys:
        bad=copy.deepcopy(rows)
        for r in bad:r[key]=-.01
        assert not protocol.confirm_gates('F111',bad)['passed'],key
    assert protocol.holm({'a':.001,'b':.02,'c':.1})=={'a':True,'b':True,'c':False}
    suffix=evaluator.SuffixBackoff();suffix.observe(b'x',b'\x00');assert suffix.generate(b'x')==(b'\x00','EOS')
    return dict(all8_selection_cases=True,root_promotable=True,negative_gates_rejected=len(keys),suffix_byte_and_EOS=True)

def cycle3(ledger,out,checks):
    world=TECH_WORLDS[2];jobs={}
    for arm in ARMS:
        m=Learner(arm,world);rows=[];traces=[];start=time.process_time()
        before=evaluator.score(m,raw_records()[:2]);init_digest=m.digest()
        for k in range(6):
            rows.append(ledger.call(arm+f'-arbitrary-E1-window{k}',m,lambda:m.train_window(raw_records())))
            traces.append(copy.deepcopy(m.last_trace))
        after=evaluator.score(m,raw_records()[:2]);elapsed=time.process_time()-start
        m.checkpoint(out/f'{arm}.pt',dict(next_record=24,history='TECH_ONLY'))
        jobs[arm]=dict(records=24,parameters=sum(p.numel() for p in m.p.values()),before=before,after=after,
                       resource=m.resource.copy(),wall_process_CPU_with_probes=elapsed,initial_digest=init_digest,updates=rows,traces=traces)
        write(out/f'{arm}_RECEIPT.json',jobs[arm])
        del m
    checks['arbitrary_E1']=jobs
    # Never replay a trained prefix: three actual four-record suffix applications.
    base,cursor=Learner.restore(out/'F111.pt')
    suffix=list(reversed(raw_records()));recovered,cursor2=Learner.restore(out/'F111.pt')
    ledger.call('continuation-reference',base,lambda:base.train_window(suffix))
    ledger.call('continuation-restored',recovered,lambda:recovered.train_window(suffix))
    assert base.digest()==recovered.digest() and cursor==cursor2
    checks['boundary_restore_suffix_parity']=True
    denied,cursor=Learner.restore(out/'F111.pt')
    theta=denied.theta_digest();state=denied.state.detached(True);time0=denied.time
    ledger.call('restored-nonzero-denial',denied,lambda:denied.train_window(suffix,False))
    assert theta==denied.theta_digest()
    expected=denied.anatomy.rest(state,denied.time-time0)
    close(denied.state.fast,expected.fast,'restored denied fast');close(denied.state.slow,expected.slow,'restored denied slow')
    old=bootstrap.source_module('native_bind_v1_final_parity','native_context_binding_dev_v1_20261016/bind_kernel.py')
    native=Learner('NATIVE',world);reference=old.Learner('NATIVE',world)
    ledger.call('final-source-native-parity',native,lambda:native.train_window(raw_records()))
    ledger.call('final-source-parent-parity',reference,lambda:reference.train_window(raw_records()))
    checks['final_source_native_forward_parity']=state_close(native.state,reference.state)
    checks['synthetic_pipeline']=synthetic_protocol_checks()
    checks['peak_process_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def main(cycle,out):
    if cycle not in (1,2,3):raise ValueError('technical cycles only')
    out=Path(out)
    if out.exists():raise ValueError('fresh immutable attempt directory required')
    out.mkdir(parents=True);(out/'source').mkdir()
    for p in bootstrap.HERE.glob('*.py'):shutil.copyfile(p,out/'source'/p.name)
    shutil.copyfile(bootstrap.HERE/'SPEC.json',out/'source/SPEC.json')
    shutil.copyfile(bootstrap.HERE/'OUTPUT_CODEBOOK.json',out/'source/OUTPUT_CODEBOOK.json')
    write(out/'DESIGN.json',dict(cycle=cycle,world=TECH_WORLDS[cycle-1],record_cap={1:128,2:128,3:256}[cycle],science=False,identity=source_identity()))
    ledger=Ledger(out,cycle);checks={};started=time.time()
    try:
        (cycle1,cycle2,cycle3)[cycle-1](ledger,out,checks)
        result=dict(passed=True,cycle=cycle,checks=checks,source_identity=source_identity(),science_jobs=0,
                    full_record_applications=sum(r['records_executed'] for r in ledger.rows),elapsed_seconds=time.time()-started,
                    evidence='E0 implementation qualification;arbitrary taught E1 only;no addition efficacy claim')
        write(out/'RESULT.json',result)
        print(json.dumps(dict(passed=True,cycle=cycle,records=result['full_record_applications'],seconds=result['elapsed_seconds'])),flush=True)
    except Exception as exc:
        write(out/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),checks=checks,identity=source_identity()))
        print(json.dumps(dict(passed=False,cycle=cycle,error=repr(exc))),flush=True);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cycle',type=int,required=True);p.add_argument('--out',required=True)
    a=p.parse_args();main(a.cycle,a.out)
