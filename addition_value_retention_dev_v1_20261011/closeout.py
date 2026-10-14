"""Read-only closure of saved DEV records; never executes model trajectories."""
import argparse, copy, gzip, hashlib, json
from pathlib import Path
import numpy as np
import delivery, settings
from runner import source_lock

ROOT=Path(__file__).resolve().parent
def write(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def closure(cycle):
    receipts=[];checks=[]
    for c in range(1,cycle+1):
        folder=ROOT/'cycles'/f'cycle{c}'
        snap=json.loads((folder/('SOURCE_BEFORE_MAIN.json' if c==1 else 'SOURCE_SNAPSHOT.json')).read_text())
        archive=folder/('source_before_main' if c==1 else 'source_before_test')
        for name,digest in snap.items():
            assert hashlib.sha256((archive/name).read_bytes()).hexdigest()==digest,(c,name)
        for world in (832099+2*c,832100+2*c):
            path=folder/f'WORLD_{world}.json.gz'
            r=json.loads(gzip.decompress(path.read_bytes()))
            assert r['world']==world and r['source']['files']==snap
            assert r['source']['parent_identity']==source_lock()['parent_identity']
            for name in ('learners.py','temporal.py','representation.py','settings.py','data.py','dependencies.py','addition_checkpoint.py'):
                assert r['source']['files'][name]==source_lock()['files'][name],name
            v=delivery.verify(r,check_source=False)
            checks.append(dict(world=world,source_archive_verified=True,**v));receipts.append(r)
    summary=delivery.summarize(receipts);summary['cycles_completed']=cycle
    assert summary['DEV_worlds']==cycle*2 and summary['training_histories']==cycle*18
    # Strong equality check across every checkpoint/history/probe, not just accuracy.
    max_value=0.;max_probability=0.;different_outputs=0;tested=0
    for r in receipts:
        s=r['arms']['STABLE'];t=r['arms']['TRANSFER']
        for phase in settings.PHASES:
            for branch in settings.BRANCHES:
                for a,b in zip(s['probes'][phase][branch]['rows'],t['probes'][phase][branch]['rows']):
                    assert a['pair']==b['pair']
                    pa=next(x['prediction'] for x in a['trace'] if x['prediction'])
                    pb=next(x['prediction'] for x in b['trace'] if x['prediction'])
                    max_value=max(max_value,abs(pa['numeric_value']-pb['numeric_value']))
                    max_probability=max(max_probability,max(abs(x-y) for x,y in zip(pa['probabilities'],pb['probabilities'])))
                    different_outputs+=int(a['emitted']!=b['emitted']);tested+=1
    equivalence=dict(probe_outputs_compared=tested,different_outputs=different_outputs,max_value_difference=max_value,max_probability_difference=max_probability)
    tamper=delivery.tamper_suite(receipts[-1])
    # Selection tests reuse saved statistics, with no fitting and no model run.
    bad=copy.deepcopy(receipts)
    for r in bad:
        for arm in ('STABLE','TRANSFER'):
            r['arms'][arm]['probes']['day2']['W']['groups']['old']['exact']=0.
    assert delivery.summarize(bad)['selected_candidate'] is None
    assert delivery.summarize(receipts)['selected_candidate']==summary['selected_candidate']
    out=ROOT/'cycles'/f'cycle{cycle}'
    write(out/'RESULT.json',summary)
    audit=dict(verdict='PASS',world_audits=checks,tamper=tamper,equivalence=equivalence,
               failed_gates_reject_selection=True,model_trajectory_executions=0,
               science_worlds=0,formal_launch_authorized=False)
    write(out/'POST_TEST_AUDIT.json',audit)
    if cycle==3:
        write(ROOT/'SUMMARY.json',summary);write(ROOT/'FINAL_AUDIT.json',audit)
    print(json.dumps(dict(cycle=cycle,selected=summary['selected_candidate'],means={k:v['means'] for k,v in summary['results'].items()},equivalence=equivalence)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cycle',type=int,choices=(1,2,3),required=True)
    closure(parser.parse_args().cycle)
