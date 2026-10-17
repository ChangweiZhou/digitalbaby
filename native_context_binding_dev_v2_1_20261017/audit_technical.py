"""Independent receipt/ledger reducer. Does not import or execute learners."""
import copy
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
WORDS=json.loads((HERE/'OUTPUT_CODEBOOK.json').read_text())['words_by_class']
CODE=np.array([[1. if word&(1<<j) else -1. for j in range(16)] for word in WORDS])
MU=CODE.mean(0)
ARMS=('NATIVE',)+tuple(f'F{e}{a}{c}' for e in (0,1) for a in (0,1) for c in (0,1))

def audit_traces(arm,traces,updates,permissions):
    checked=0
    if len(traces)!=len(updates) or len(traces)!=len(permissions):raise ValueError('window roster mismatch')
    for window,summary,allowed in zip(traces,updates,permissions):
        if summary['records']!=4 or summary['allowed']!=allowed:raise ValueError('permission/count mismatch')
        losses=[];record_terms=[]
        for row in window:
            if row['native_write']!=allowed or not row['prefeedback'] or not row['frame_address_reused']:
                raise ValueError('causal frame or write flag violated')
            logits=np.array(row['logits'],dtype=float)
            if logits.shape!=(257,) or not np.isfinite(logits).all():raise ValueError('bad full vocabulary logits')
            target=row['target']
            if type(target) is not int or not 0<=target<=256 or int(logits.argmax())!=row['predicted']:
                raise ValueError('bad target or first-choice reconstruction')
            exp=np.exp(logits-logits.max());pi=exp/exp.sum()
            if arm!='NATIVE' and arm[1]=='1':
                expected=(pi@CODE-CODE[target])/2
                np.testing.assert_allclose(row['signal'],expected,atol=1e-10,rtol=1e-10)
            else:
                np.testing.assert_allclose(row['signal']['shared'],(1-CODE[target])/2,atol=1e-12,rtol=0)
                np.testing.assert_allclose(row['signal']['private'],(MU-CODE[target])/2,atol=1e-12,rtol=0)
            record_terms.append(float(logits.max()+math.log(exp.sum())-logits[target]))
            if target==256:
                losses.append(sum(record_terms)/len(record_terms));record_terms=[]
            checked+=1
        if record_terms or len(losses)!=4:raise ValueError('incomplete answer/EOS record')
        if abs(sum(losses)/4-summary['loss'])>1e-9:raise ValueError('loss not reconstructed from real predictions')
        expected_update=arm!='NATIVE' and allowed
        if summary['applied_update']!=expected_update:raise ValueError('hidden optimizer update')
    return checked

def audit_score(score):
    correct=[]
    for r in score['rows']:
        ok=r['status']=='EOS' and r['emitted']==r['answer']
        if ok!=r['correct']:raise ValueError('teacher-forced or fake exact answer')
        correct.append(ok)
    if not correct or abs(sum(correct)/len(correct)-score['accuracy'])>1e-12:
        raise ValueError('wrong own-output accuracy')

def audit_job(job):
    arm=job['arm']
    if arm not in ARMS or job['records']!=16 or job['science_jobs']!=0:raise ValueError('technical job roster/count')
    if set(job['histories'])!={'W','N_OLD'}:raise ValueError('missing causal history')
    tokens=0
    for history,row in job['histories'].items():
        if row['records']!=8 or row['resource']['records']!=8:raise ValueError('history count')
        tokens+=audit_traces(arm,row['traces'],row['updates'],[history=='W',True])
        expected_steps=[1,2] if history=='W' else [0,1]
        if arm=='NATIVE':expected_steps=[0,0]
        if [v['actor_steps'] for v in row['updates']]!=expected_steps:raise ValueError('OLD denial updated theta/RMS')
        # State version/time checks independently reconstruct feedback ordering.
        version=0;clock=0.
        for window in row['traces']:
            next_record=True
            for token in window:
                if next_record:version+=6;clock+=6*(30/14)
                if token['frame_version']!=version or abs(token['pretime']-clock)>1e-8:
                    raise ValueError('prediction frame from wrong event or clock')
                version+=1;clock+=30/14;next_record=token['target']==256
            version+=1;clock+=86400
        for score in row['scores'].values():audit_score(score)
        r=row['resource']
        if not 0<=r['initialization_cpu']<=r['production_cpu']:raise ValueError('missing initialization cost')
        if abs(r['logical_MEM_cpu']+r['legacy_readout_cpu']-r['production_cpu'])>1e-7:raise ValueError('MEM CPU double/subtract charge')
        if arm=='NATIVE' and abs(r['logical_LEGACY_cpu']+r['mem_readout_cpu']-r['production_cpu'])>1e-7:
            raise ValueError('LEGACY CPU double/subtract charge')
    return tokens

def rejected(fn):
    try:fn()
    except (ValueError,AssertionError,KeyError):return True
    return False

def main():
    folder=HERE/'cycles/cycle3/attempt1'
    jobs={arm:json.loads((folder/f'{arm}_RECEIPT.json').read_text()) for arm in ARMS}
    tokens=sum(audit_job(job) for job in jobs.values())
    e1={}
    for arm in ('NATIVE','F000','F111'):
        row=json.loads((folder/f'{arm}_E1_RECEIPT.json').read_text())
        tokens+=audit_traces(arm,row['traces'],row['updates'],[True]*6)
        audit_score(row['before']);audit_score(row['after'])
        e1[arm]={'before':row['before']['accuracy'],'after':row['after']['accuracy'],'records':row['records']}
    tamper={}
    bad=copy.deepcopy(jobs['F111']);bad['histories']['W']['traces'][0][0]['signal'][0]+=.01
    tamper['wrong_residual']=rejected(lambda:audit_job(bad))
    bad=copy.deepcopy(jobs['F000']);bad['histories']['N_OLD']['traces'][0][0]['native_write']=True
    tamper['OLD_denial_not_disabled']=rejected(lambda:audit_job(bad))
    bad=copy.deepcopy(jobs['F111']);bad['histories']['W']['traces'][0][0]['frame_version']+=1
    tamper['stale_frame']=rejected(lambda:audit_job(bad))
    bad=copy.deepcopy(jobs['NATIVE']);bad['histories']['W']['scores']['OLD_END']['accuracy']=.12345
    tamper['fake_accuracy']=rejected(lambda:audit_job(bad))
    bad=copy.deepcopy(jobs['F111']);bad['histories']['W']['resource']['initialization_cpu']=-1
    tamper['negative_setup_cost']=rejected(lambda:audit_job(bad))
    if not all(tamper.values()):raise AssertionError(tamper)
    ledgers=[json.loads(p.read_text()) for p in (HERE/'cycles').glob('cycle*/attempt*/EXECUTION_LEDGER.json')]
    counts={cycle:sum(l['full_record_applications'] for l in ledgers if l['cycle']==cycle) for cycle in (1,2,3)}
    if any(counts[c]>{1:128,2:128,3:256}[c] for c in counts) or sum(counts.values())>512:
        raise ValueError('technical cap exceeded')
    result=dict(passed=True,independent_auditor_imports_learner=False,model_execution=False,
                full_record_applications_by_cycle=counts,total_full_records=sum(counts.values()),
                audited_prediction_tokens=tokens,technical_addition_jobs=9,scientific_jobs=0,
                arbitrary_E1=e1,tamper_cases_rejected=tamper)
    path=HERE/'TECHNICAL_AUDIT.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
