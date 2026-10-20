"""Cycle 2 finite parameter choice and warmup/acquisition checks; HELD unused."""
import bootstrap
from dataclasses import asdict
import json
import math
import time
import subprocess

import numpy as np
from core import Learner,Limits,CODE,UP,DELTA,hash_arrays,multipliers
from data import records,split
from evaluate import score
from checkpoint import save,load
from artifacts import HERE,atomic_json,source_identity


def confidence(model,recs):
    values=[]
    for prompt,answer in recs:
        view=model.clone();view._prompt(prompt);z=UP.logits(view._frame()['z'])
        target=answer[0]
        values.append(float(z[target]-np.max(np.delete(z,target))))
    return float(np.mean(values))


def main():
    awake=subprocess.Popen(['caffeinate','-i','-w',str(__import__('os').getpid())])
    output=HERE/'cycles/cycle2';start=time.process_time();spec=json.loads((HERE/'SPEC.json').read_text())
    generic=[(b'\x00\x80\xff',b'\x11'),(b'\xff\x00\x80',b'\xee'),
             (b'\x81\x01\x00',b'\x22'),(b'\x01\x81\x00',b'\xdd')]
    choices=[];selected=None;total_records=0
    for candidate in spec['calibration_candidates']:
        limits=Limits(**{k:v for k,v in candidate.items() if k!='id'})
        model=Learner(202610208901,plastic=True,gradual=True,limits=limits)
        before=model.parameter_hash();initial=confidence(model,generic)
        for _ in range(16):
            for p,a in generic:model.train(p,a)
        total_records+=64
        final=confidence(model,generic);d=model.diagnostics()
        checks=dict(finite=d['finite'],nonzero_credit=model.parameter_hash()!=before,
                    bounded_updates=all(v['max_row_norm']<=v['max_allowed']+1e-12 for v in model.last_update.values()),
                    improves_taught_target_margin=final>initial,
                    zero_diagonal=all(np.all(np.diag(model.params[k])==0) for k in ('U','H')))
        choice=dict(candidate=candidate,initial_margin=initial,final_margin=final,checks=checks,record_applications=64)
        choices.append(choice);atomic_json(output/'CALIBRATION.json',dict(attempts=choices,held_used=False))
        print('calibration',candidate['id'],checks,initial,final,flush=True)
        if all(checks.values()):selected=limits;break
    if selected is None:raise RuntimeError('finite calibration list exhausted; no permitted parameter candidate qualifies')
    atomic_json(HERE/'PARAMETER_LOCK.json',dict(schema='V3_ENGINEERING_PARAMETER_LOCK',limits=asdict(selected),
                                               selection='first qualifying candidate in preregistered increasing-eta order',
                                               held_used=False,initialization=spec['initialization'],
                                               a0=spec['a0'],H=spec['H'],old_budget=7680,
                                               source_identity_at_lock=source_identity()))
    world=202610209901;sets,_=split(world);results={}
    for arm in spec['arms']:
        m=Learner(202610208902,plastic=arm[0]=='P',gradual=arm[-1]=='G',limits=selected)
        birth=m.parameter_hash();base=m.clone();probes=[p for p,_ in sets['OLD'][:8]]
        codes=[]
        for p in probes:
            v=m.clone();v._prompt(p);codes.append(v._frame()['x'])
        curve=[];cpu=0.
        for n,row in enumerate(records(world,'W','OLD'),1):
            t=time.process_time();m.train(row['prompt'],row['answer']);cpu+=time.process_time()-t
            if n in (60,240,480,960):
                outcome=score(m,sets['OLD']);curve.append(dict(records=n,accuracy=outcome['accuracy']))
                print('acquisition',arm,n,outcome['accuracy'],'production_cpu',round(cpu,2),flush=True)
            if n==960:break
        total_records+=960
        assert (m.parameter_hash()!=birth)==(arm[0]=='P')
        overlap=[]
        for p,old in zip(probes,codes):
            v=m.clone();v._prompt(p);new=v._frame()['x'];overlap.append(float((old*new).sum()/max(1,((old+new)>0).sum())))
        save(m,output/f'{arm}_CHECKPOINT.npz');restored=load(output/f'{arm}_CHECKPOINT.npz')
        assert restored.digest()==m.digest()
        if arm=='P-G':
            # Whole-record resume equivalence with three additional taught records, counted on both branches.
            for p,a in sets['OLD'][:3]:m.train(p,a);restored.train(p,a)
            total_records+=6
            assert m.digest()==restored.digest(),'resume differs'
        results[arm]=dict(curve=curve,production_cpu=cpu,mean_birth_code_jaccard=float(np.mean(overlap)),
                          diagnostics=m.diagnostics(),initial_parameter_hash=birth,
                          last_checkpoint_parameter_hash=restored.parameter_hash())
        atomic_json(output/'ACQUISITION.json',dict(world=world,held_used=False,arms=results))
    # Frozen and plastic pairs have identical birth forward outputs and all constants other than plasticity permission.
    f=Learner(202610208903,plastic=False,gradual=True,limits=selected)
    p=Learner(202610208903,plastic=True,gradual=True,limits=selected)
    assert f.parameter_hash()==p.parameter_hash()
    assert np.array_equal(f.D,p.D) and np.array_equal(f.Fback,p.Fback)
    f._prompt(generic[0][0]);p._prompt(generic[0][0]);assert np.array_equal(f._frame()['x'],p._frame()['x'])
    # Two distinct incoming targets cannot influence their own prefeedback prediction/address.
    a=f.clone();b=p.clone();fa=a._frame();fb=b._frame();assert np.array_equal(fa['z'],fb['z'])
    a._event(0,0,True,fa);b._event(255,255,True,fb)
    assert hash_arrays([fa['x']])==hash_arrays([fb['x']])
    report=dict(schema='MINIFLY_V3_CYCLE_2',accepted=True,calibration=choices,selected_limits=asdict(selected),
                arms=results,held_used=False,birth_forward_equivalence=True,incoming_target_frame_independence=True,
                whole_record_resume=True,record_applications=total_records,
                partial_record_target_fixtures=2,science_records=0,cpu_s=time.process_time()-start,
                source_identity=source_identity())
    atomic_json(output/'AUDIT.json',report)
    print('CYCLE_2_ACCEPTED',total_records,'cpu',round(report['cpu_s'],2),flush=True)


if __name__=='__main__':main()
