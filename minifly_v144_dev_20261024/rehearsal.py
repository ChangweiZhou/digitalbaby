"""Cycle-3 synthetic resume, checker exit and prospective selection fixtures."""
import json
import pickle
import subprocess
import sys
import numpy as np
from v144_audit import HERE,source_lock,write_json,Ledger
from reference import Reference,ByteTrace
import capacity


def shortlist(rows):
    # Prospective R1: preserve all three hidden sizes; early stopping can miss late learners.
    def key(r):
        return (-r['held_exact'],r['teacher_loss'],r['cpu_s'],r['hidden'],r['lr'],r['decay'])
    return [min([r for r in rows if r['hidden']==h],key=key) for h in [128,256,512]]


def main():
    ledger=Ledger('cycle3_synthetic','A_C_REHEARSAL');tests={}
    trace=ByteTrace(capacity.lb1.BC.R_TABLE,capacity.lb1.fastmem.PERM,capacity.lb1.DELTA,capacity.lb1.fastmem.FEPerm)
    data=[(b'a\x00',b'\xff ='),(b'\xffb',b''),(b' a=',b'\x00')]*4
    whole=Reference(trace,'resume',hidden=16)
    for p,y in data:whole.train(p,y)
    part=Reference(trace,'resume',hidden=16)
    for p,y in data[:6]:part.train(p,y)
    restored=pickle.loads(pickle.dumps(part,protocol=5))
    for p,y in data[6:]:restored.train(p,y)
    assert whole.state_hash()==restored.state_hash()
    tests['synthetic_resume_state_exact']=True
    tests['synthetic_record_applications']=24
    tests['synthetic_target_tokens']=whole.target_tokens+restored.target_tokens
    lock=source_lock(HERE/'cycles/CYCLE3_SOURCE_LOCK.json')
    req=['B_LEARN','B_FROZEN']
    good=[{'id':x,'complete':True,'metrics':{'fixture':True},'settings':{'hidden':128}} for x in req]
    variants={'good':dict(rows=good,required=req,closure_ok=True,expected_settings={'hidden':128}),
              'closure_failure':dict(rows=good,required=req,closure_ok=False),
              'missing_control':dict(rows=good[:1],required=req,closure_ok=True),
              'wrong_settings':dict(rows=good,required=req,closure_ok=True,expected_settings={'hidden':256})}
    exits={}
    for label,value in variants.items():
        path=HERE/'cycles'/('checker_'+label+'.json');write_json(path,value)
        p=subprocess.run([sys.executable,str(HERE/'checker.py'),'--lock',str(HERE/'cycles/CYCLE3_SOURCE_LOCK.json'),
                          '--input',str(path)],capture_output=True,text=True)
        exits[label]=p.returncode
        assert p.returncode==(0 if label=='good' else 2),(label,p.stdout,p.stderr)
    bad_lock=dict(lock,files=dict(lock['files']));bad_lock['files']['minifly_v144_dev_20261024/checker.py']='wrong'
    write_json(HERE/'cycles/MUTATED_FIXTURE_LOCK.json',bad_lock)
    p=subprocess.run([sys.executable,str(HERE/'checker.py'),'--lock',str(HERE/'cycles/MUTATED_FIXTURE_LOCK.json'),
                      '--input',str(HERE/'cycles/checker_good.json')],capture_output=True,text=True)
    assert p.returncode==2;exits['hash_failure']=p.returncode;tests['checker_exit_codes']=exits
    grid=[dict(hidden=h,lr=lr,decay=d,held_exact=0,teacher_loss=10-lr,cpu_s=h/128)
          for h in [128,256,512] for lr in [.003,.01,.03] for d in [0,.001,.01]]
    picked=shortlist(grid);assert {r['hidden'] for r in picked}=={128,256,512}
    tests['zero_accuracy_shortlist_keeps_capacity_breadth']=picked
    tests['holdback_generated_or_trained']=False
    write_json(HERE/'cycles/CYCLE3_CHECKS.json',tests);ledger.emit('complete',synthetic_reference_records=24,
          synthetic_reference_target_tokens=tests['synthetic_target_tokens']);print(json.dumps(tests,indent=2))


if __name__=='__main__':main()
