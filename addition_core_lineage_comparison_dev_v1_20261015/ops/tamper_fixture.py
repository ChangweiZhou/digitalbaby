"""Corrupt copies of an actual technical receipt; never generates learner data."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit import audit_job
from io_utils import read,write,now
import bootstrap


def run():
    original=read(bootstrap.HERE/'cycles/cycle2/receipts/202610143901_N151-C16.json.gz')
    assert audit_job(original)['accepted']
    cases=[]
    for label in ('wrong_permission','forbidden_delta','forbidden_alpha','forbidden_actor','changed_target',
                  'held_cue_teaching','wrong_count','missing_checkpoint','false_scoring','aliased_birth',
                  'wrong_anatomy','missing_control'):
        r=copy.deepcopy(original)
        off=r['histories']['N_OLD']['rows'][0]
        if label=='wrong_permission':off['allowed']=True
        if label=='forbidden_delta':off['forbidden_delta']=.01
        if label=='forbidden_alpha':off['raw_alpha_l1']=1.
        if label=='forbidden_actor':off['actor_steps']=1
        if label=='changed_target':r['histories']['W']['rows'][0]['answer']='ff'
        if label=='held_cue_teaching':r['histories']['W']['rows'][0]['prompt']='ff'
        if label=='wrong_count':r['records']-=1
        if label=='missing_checkpoint':r['histories']['W']['metrics'].pop(0)
        if label=='false_scoring':
            row=r['histories']['W']['metrics'][0]['scores']['held'][0];row['correct']=not row['correct']
        if label=='aliased_birth':r['histories']['W']['births'][1]['fly_id']=r['histories']['W']['births'][0]['fly_id']
        if label=='wrong_anatomy':r['histories']['W']['births'][0]['canonical_B_sha256']='wrong'
        if label=='missing_control':del r['histories']['SH_OLD']
        try:
            audit_job(r)
        except (AssertionError,ValueError,KeyError):
            cases.append(dict(case=label,rejected=True))
        else:
            raise AssertionError('auditor accepted '+label)
    result=dict(utc=now(),accepted=True,cases=cases,science=False,training_records=0,
                note='Tests independent reconstruction/count checks; not a claim of cryptographic trust in arbitrary runner instrumentation.')
    write(bootstrap.HERE/'cycles/cycle2/TAMPER_SUITE.json',result)
    return result


if __name__=='__main__':print(run(),flush=True)
