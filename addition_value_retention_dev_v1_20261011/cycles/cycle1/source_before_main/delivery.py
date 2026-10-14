"""Read-only final reducer and receipt tamper suite; no model trajectories."""
import copy,gzip,json
from pathlib import Path
import numpy as np
import settings
from audit import audit,countermodels
from equation_audit import verify_equations

def verify(r,check_source=True):
    basic=audit(r,check_source=check_source);numeric=verify_equations(r)
    return dict(verdict='PASS',basic=basic,numeric=numeric)
def tamper_suite(receipt):
    rejected=[]
    def test(name,mutate):
        x=copy.deepcopy(receipt);mutate(x)
        try:verify(x)
        except (ValueError,AssertionError,KeyError,IndexError):rejected.append(name)
        else:raise AssertionError('accepted tamper '+name)
    def answer(x,branch='W'):return next(r for tr in x['arms']['TRANSFER']['training']['old'][branch] for r in tr if r['observed_target'] is not None)
    test('forbidden_old_write',lambda x:answer(x,'N_OLD').update(answer_update=True,write_l1=[1.]))
    test('hidden_answer',lambda x:answer(x).update(observed_target=57))
    test('lost_transfer_ledger',lambda x:next(r for traces in x['arms']['TRANSFER']['training'].values() for tr in traces['W'] for r in tr if r['time_transform_l1']>0).update(time_transform_l1=0))
    test('clipping_ledger',lambda x:answer(x).update(clipped_rows=999))
    test('actual_content',lambda x:x['arms']['STABLE']['content_snapshots']['day2']['W']['slow'].__setitem__(0,1.))
    test('numeric_readout',lambda x:next(r for tr in x['arms']['STABLE']['training']['old']['W'] for r in tr if r['prediction']).get('prediction').update(numeric_value=20))
    test('output',lambda x:x['arms']['STABLE']['probes']['day2']['W']['rows'][0].update(emitted=57))
    test('hidden_held_teaching',lambda x:x['fixture']['records']['old'][0].update(pair=[1,1]))
    test('missing_byte',lambda x:x['arms']['DECAY']['training']['old']['W'][0].pop())
    test('world_or_source',lambda x:x['source']['files'].update(temporal_py='wrong'))
    test('science_count',lambda x:x.update(science_worlds=1))
    test('dropped_arm',lambda x:x['arms'].pop('DECAY'))
    return dict(verdict='PASS',rejected=len(rejected),cases=rejected,model_trajectory_executions=0)

def summarize(receipts):
    results={}
    for arm in settings.ARMS:
        worlds=[]
        for r in receipts:
            out=r['arms'][arm];probe=out['probes']['day2'];w=probe['W']['groups'];cm=countermodels(r['fixture'])
            worlds.append(dict(world=r['world'],held=w['held']['exact'],old=w['old']['exact'],new=w['new']['exact'],held_mae=w['held']['mae'],gain_no_old=w['held']['exact']-probe['N_OLD']['groups']['held']['exact'],gain_shuffled=w['held']['exact']-probe['SHUFFLED']['groups']['held']['exact'],no_old=probe['N_OLD']['groups']['held']['exact'],shuffled=probe['SHUFFLED']['groups']['held']['exact'],countermodels=cm,shortcut_gap=w['held']['exact']-max(v for k,v in cm.items() if k!='learned_linear_solvability_witness'),training_cpu=sum(v['cpu_seconds'] for k,v in out['costs'].items() if k.endswith('/W')),mutable_bytes=out['states']['day2']['W']['mutable_bytes'],time_transport_l1=sum(row['time_transform_l1'] for histories in out['training'].values() for tr in histories['W'] for row in tr),clipped_rows=sum(row['clipped_rows'] for histories in out['training'].values() for tr in histories['W'] for row in tr)))
        fields=['held','old','new','held_mae','gain_no_old','gain_shuffled','no_old','shuffled','shortcut_gap','training_cpu','mutable_bytes','time_transport_l1','clipped_rows']
        means={k:float(np.mean([x[k] for x in worlds])) for k in fields};results[arm]=dict(worlds=worlds,means=means)
    for arm in ('STABLE','TRANSFER'):
        a=results[arm]['means'];b=results['DECAY']['means'];a['old_gain_vs_decay']=a['old']-b['old'];a['held_difference_vs_decay']=a['held']-b['held']
        guards={k:a[k]>=v for k,v in [('held',.75),('old',.75),('new',.75),('gain_no_old',.2),('gain_shuffled',.2),('old_gain_vs_decay',.1),('shortcut_gap',.1)]};guards['held_vs_decay']=a['held_difference_vs_decay']>=-.05
        results[arm]['guards']=guards;results[arm]['DEV_feasible']=all(guards.values())
    selected='STABLE' if results['STABLE']['DEV_feasible'] else 'TRANSFER' if results['TRANSFER']['DEV_feasible'] else None
    return dict(schema='VALUE_RETENTION_THREE_CYCLE_DEV',cycles_completed=3,DEV_worlds=len(receipts),training_histories=sum(r['training_histories'] for r in receipts),training_bytes=sum(r['training_bytes'] for r in receipts),audited_rows=sum(r['independent_audit']['audited_rows'] for r in receipts),results=results,selected_candidate=selected,verdict='DEV_FEASIBLE_NOT_CONFIRMED' if selected else 'HOLD_NO_CANDIDATE',science_worlds=0,new_V2_adopted=False,formal_launch_authorized=False)
