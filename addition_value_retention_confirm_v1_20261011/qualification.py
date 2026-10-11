"""Synthetic operations/parity tests and read-only DEV-to-formal auditor fixtures."""
import argparse,copy,importlib.util,json,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
import settings,data
from learners import build
from confirm_ops import ROOT,OPS,read,atomic,sha
from runner import raw_consume,source_lock
from analysis import decide
from confirm_audit import verify

def synthetic():
    states={}
    for mode in settings.ARMS:
        m=build(mode);tr=[]
        for raw in (b'0+1=1\n',b'3+4=7\n',b'2+3=6\n',b'1+1=\n')*3:tr+=raw_consume(m,raw)
        m.rest(86400);c=m.clone();before=m.state_digest();c.plastic=False
        tr+=raw_consume(c,b'1+2=\n');assert m.state_digest()==before
        states[mode]=dict(digest=m.state_digest(),probe=c.state_digest(),trace=tr)
    return states

def main():
    OPS.mkdir(exist_ok=True);out=OPS/'qualification';out.mkdir(exist_ok=False)
    tests=[];prov=read(ROOT/'DEV_PROVENANCE.json')
    for name,digest in prov['qualified_model_files'].items():assert sha(ROOT/name)==digest,name
    tests.append('all_operative_model_and_equation_files_identical_to_qualified_DEV')
    # Check every mathematical setting against DEV; scope/version are the only settings differences.
    env={};exec((Path(prov['dev_folder'])/'settings.py').read_text(),env)
    for key in ('ALPHABET','DT','DAY','REPEATS','BRANCHES','SCALE','FAST_TAU','SLOW_TAU','SLOW_SHARE','ROW_BOUND','VISIT_CAP','ORDINAL_ETA','PHASES'):assert getattr(settings,key)==env[key],key
    tests.append('feedback_clock_parameters_and_representation_frozen')
    for world in (832101,833000,833065,True):
        try:settings.require_dev(world)
        except ValueError:pass
        else:raise AssertionError('foreign world admitted')
    tests.append('DEV_and_extra_worlds_rejected')
    baseline=synthetic()
    procs=[]
    for i in range(2):
        p=out/f'parity{i}.json';procs.append((subprocess.Popen([sys.executable,str(ROOT/'qualification.py'),'--child',str(p)]),p))
    for proc,p in procs:assert proc.wait()==0 and read(p)==baseline
    tests.append('two_worker_E0_state_and_output_bitwise_parity')
    import addition_checkpoint
    for mode in settings.ARMS:
        m=build(mode);raw_consume(m,b'1+2=')
        p=out/f'{mode}_pending.npz';addition_checkpoint.save(m,mode,p)
        restored=addition_checkpoint.load(p)
        assert restored.state_digest()==m.state_digest()
        assert raw_consume(restored,b'3\n')==raw_consume(m,b'3\n')
        assert restored.state_digest()==m.state_digest()
    tests.append('formal_identity_pending_checkpoint_exact_recovery')
    # Saved DEV bytes are unchanged; transformed scope fixture is technical only.
    dev=read(Path(prov['dev_folder'])/'cycles/cycle3/WORLD_832106.json.gz')
    r=copy.deepcopy(dev);r['arms'].pop('TRANSFER');r.update(schema='ADDITION_CONFIRM_RECEIPT',science_worlds=1,training_histories=6,training_bytes=24072,version=settings.VERSION,source=source_lock())
    admission=settings.require_dev
    settings.require_dev=lambda world:None if world==dev['world'] else admission(world)
    try:
        verdict=verify(r);assert verdict['basic']['audited_rows']==27322
        cases=[]
        def reject(name,mutate):
            x=copy.deepcopy(r);mutate(x)
            try:verify(x)
            except (AssertionError,ValueError,KeyError,IndexError):cases.append(name)
            else:raise AssertionError('tamper accepted: '+name)
        def answer(x,b='W'):return next(t for tr in x['arms']['STABLE']['training']['old'][b] for t in tr if t['observed_target'] is not None)
        reject('N_OLD_forbidden_write',lambda x:answer(x,'N_OLD').update(answer_update=True,write_l1=[1.]))
        reject('actual_answer_byte',lambda x:answer(x).update(observed_target=57))
        reject('content_snapshot',lambda x:x['arms']['STABLE']['content_snapshots']['day2']['W']['slow'].__setitem__(0,1.))
        reject('held_teaching',lambda x:x['fixture']['records']['old'][0].update(pair=[1,1]))
        reject('missing_record',lambda x:x['arms']['DECAY']['training']['old']['W'][0].pop())
        reject('dropped_arm',lambda x:x['arms'].pop('DECAY'))
        reject('wrong_count',lambda x:x.update(training_histories=9))
        reject('source',lambda x:x['source']['files'].update(temporal_py='wrong'))
        reject('probe_hidden_feedback',lambda x:x['arms']['STABLE']['erased_final']['rows'][0]['trace'][0].update(observed_target=50))
        reject('probe_numeric_value',lambda x:next(t['prediction'] for t in x['arms']['STABLE']['probes']['day2']['W']['rows'][0]['trace'] if t['prediction']).update(numeric_value=99))
        reject('output',lambda x:x['arms']['STABLE']['probes']['day2']['W']['rows'][0].update(emitted=57))
        reject('scope',lambda x:x.update(science_worlds=0))
    finally:settings.require_dev=admission
    tests.append('formal_receipt_adaptor_raw_byte_numeric_audit_and_12_tampers')
    # Test prospective intersection, including no-zero-width fallback on a constant sample.
    fixture=dict(world=0,held=1.,old=1.,new=1.,held_mae=0.,gain_no_old=.75,gain_shuffled=.85,old_gain_vs_decay=.3,held_difference_vs_decay=.05,shortcut_gap=.75)
    positive=[dict(fixture,world=i,old_gain_vs_decay=.28+.005*(i%5)) for i in range(64)]
    assert decide(positive)['verdict'].startswith('ADOPT')
    bad=copy.deepcopy(positive)
    for x in bad:x['gain_no_old']=0.
    assert not decide(bad)['verdict'].startswith('ADOPT')
    for x in bad:x['gain_no_old']=.75;x['old']=.7
    assert not decide(bad)['verdict'].startswith('ADOPT')
    assert decide([dict(fixture,world=i) for i in range(64)])['contrasts']['old_gain_vs_decay']['lower']<0
    tests.append('intersection_gates_negative_and_zero_variance_conservative_decisions')
    from supervisor import dispatch
    done=[]
    plan=[(i,[sys.executable,'-c','import time;time.sleep(.15)']) for i in range(4)]
    result=dispatch(plan,2,out/'dispatch_ok',lambda *a:None,done.append)
    assert done==[0,1,2,3] and result['max_workers']==2
    attempts=[]
    try:dispatch([(1,[sys.executable,'-c','raise SystemExit(7)']),(2,[sys.executable,'-c','import time;time.sleep(3)'])],2,out/'dispatch_fail',lambda active,e:attempts.append(tuple(active)),lambda w:None)
    except RuntimeError:pass
    else:raise AssertionError('worker failure did not stop dispatcher')
    assert len(attempts)<10
    assert dispatch(plan,2,out/'dispatch_pause',lambda *a:None,lambda w:None,drain_after=0)['stage']=='PAUSED_SESSION_LIMIT'
    tests.append('two_worker_dispatch_stop_on_failure_no_retry_and_whole_job_pause')
    p=out/'exclusive.json';atomic(p,dict(value=1),exclusive=True)
    try:atomic(p,dict(value=2),exclusive=True)
    except FileExistsError:pass
    else:raise AssertionError('duplicate commit overwrote accepted data')
    assert read(p)==dict(value=1);tests.append('atomic_no_overwrite_receipt')
    atomic(ROOT/'QUALIFICATION.json',dict(verdict='PASS',tests=tests,tamper_cases=cases,scientific_worlds_executed=0,DEV_worlds_rerun=0,synthetic_fixture_only=True,workers=2,unchanged_learning_model=True),exclusive=True)
    print(json.dumps(dict(verdict='PASS',tests=tests)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child');args=p.parse_args()
    if args.child:atomic(args.child,synthetic(),exclusive=True)
    else:main()
