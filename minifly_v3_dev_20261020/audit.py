"""Independent receipt/trace reducer; does not import or execute the learner."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from artifacts import HERE,atomic_json,sha


def one_sided_p(differences):
    x=np.asarray(differences,float)
    if len(x)<2 or not np.isfinite(x).all():return 1.
    mean=float(x.mean());sd=float(x.std(ddof=1))
    if sd==0:return 2.**(-len(x)) if mean>0 else 1.
    return float(student_t.sf(mean/(sd/math.sqrt(len(x))),len(x)-1))


def verify_scores(point):
    for name,score in point['scores'].items():
        correct=0
        for row in score['rows']:
            exact=(row['got']==row['want'] and row['ended_with_EOS'] and not row['truncated'])
            assert exact==row['exact']
            if row['ended_with_EOS']:assert row['tokens'][-1]==256
            if row['truncated']:assert not row['ended_with_EOS'] and len(bytes.fromhex(row['got']))==16
            correct+=int(exact)
        assert score['n']==len(score['rows']) and score['correct']==correct
        assert abs(score['accuracy']-correct/len(score['rows']))<1e-12


def sample_integral(points):
    xy=[(math.log2(p['count']+1),p['scores']['HELD']['accuracy']) for p in points]
    return sum((x1-x0)*(y0+y1)/2 for (x0,y0),(x1,y1) in zip(xy,xy[1:]))/(xy[-1][0]-xy[0][0])


def cpu_integral(points,cstar):
    last_time=0.;last_score=points[0]['scores']['HELD']['accuracy'];area=0.
    for point in points:
        now=min(cstar,point['production_cpu'])
        area+=max(0.,now-last_time)*last_score
        last_time=now;last_score=point['scores']['HELD']['accuracy']
        if now==cstar:return area/cstar
    return (area+max(0.,cstar-last_time)*last_score)/cstar


def audit_job(path):
    path=Path(path);r=json.loads((path/'RECEIPT.json').read_text())
    cert=json.loads((path/'COMPLETE.json').read_text());assert cert['receipt_sha256']==sha(path/'RECEIPT.json')
    trace=HERE/r['trace'];assert sha(trace)==r['trace_sha256']
    code_doc=json.loads((HERE/'upstream/OUTPUT_CODEBOOK.json').read_text())
    code=np.array([[1. if w&(1<<j) else -1. for j in range(16)] for w in code_doc['words_by_class']])
    spec=json.loads((HERE/'SPEC.json').read_text());target_count=write_count=rows=0;hist_counts={}
    group_answers={};true_answers={};last_time={};last_record={};train_prompts=set()
    gradual=[(1/16+(1-1/16)*(3*min(n/960,1)**2-2*min(n/960,1)**3)) for n in range(1,7681)]
    mean=sum(gradual)/7680;square=sum(a*a for a in gradual)/7680
    for line in gzip.open(trace,'rt'):
        row=json.loads(line);rows+=1;h=row['history'];phase=row['phase'];hist_counts[h]=hist_counts.get(h,0)+1
        assert row['record']==last_record.get(h,0)+1;last_record[h]=row['record']
        if h in last_time and row['phase_record']!=1:assert abs(row['time_before']-last_time[h])<1e-8
        if phase=='NEW' and row['phase_record']==1:
            assert abs(row['time_before']-last_time[h]-86400)<1e-7
        last_time[h]=row['time_after']
        prompt=bytes.fromhex(row['prompt']);answer=bytes.fromhex(row['answer']);truth=bytes.fromhex(row['truth'])
        if row['record']>7680:expected_scale,expected_rep=1.,1.
        elif r['arm'][-1]=='G':expected_scale=gradual[row['record']-1];expected_rep=expected_scale**2
        else:expected_scale,expected_rep=mean,square
        assert abs(row['value_scale']-expected_scale)<1e-12
        assert abs(row['representation_scale']-expected_rep)<1e-12
        expected=(2+len(prompt)+len(answer)+1)*(30/14)
        assert abs(row['time_after']-row['time_before']-expected)<1e-7
        assert row['version_after']-row['version_before']==2+len(prompt)+len(answer)+1
        assert row['allowed']==(not(h=='N_OLD' and phase=='OLD'))
        assert len(row['target_events'])==len(answer)+1
        for event,target in zip(row['target_events'],[*answer,256]):
            target_count+=1;assert event['target']==target
            assert event['post_version']==event['pre_version']+1
            assert abs(event['posttime']-event['pretime']-30/14)<1e-8
            z=np.array(event['values']);logits=code@(z-z.mean())/4
            assert int(logits.argmax())==event['predicted']
            expected_s=(np.tanh((z-z.mean())/2)-code[target])/2*row['value_scale']
            c=event['coefficient']
            assert (c is not None)==row['allowed']
            if c is not None:
                write_count+=1;assert np.max(abs(np.array(c)-expected_s))<1e-12
                assert max(abs(x) for x in c)<=1+1e-12
        size=60 if phase=='OLD' else 20
        epoch=(row['phase_record']-1)//size
        key=(h,phase,epoch);group_answers.setdefault(key,[]).append(answer);true_answers.setdefault(key,[]).append(truth)
        if h=='W':train_prompts.add(row['prompt'])
        if phase=='OLD' and h=='N_OLD':assert not row['local_update']
    assert rows==r['record_applications']==24960
    assert hist_counts=={'W':8320,'N_OLD':8320,'SH_OLD':8320}
    assert target_count==r['target_events'] and write_count==r['actual_native_target_writes']
    for k,answers in group_answers.items():assert sorted(answers)==sorted(true_answers[k])
    for history,tr in r['trajectories'].items():
        for point in tr['points']+[tr['timeonly_endpoint']]:verify_scores(point)
        old=[p for p in tr['points'] if p['phase']=='OLD'];assert [p['count'] for p in old]==spec['old_checkpoints']
        new=[p for p in tr['points'] if p['phase']=='NEW'];assert [p['count'] for p in new]==spec['new_checkpoints']
        assert abs(sample_integral(old)-tr['sample_area'])<1e-12
        final=tr['points'][-1];assert final['phase']=='FINAL' and final['records']==8320
        assert abs(final['time']-tr['timeonly_endpoint']['time'])<1e-8
        for held in final['scores']['HELD']['rows']:assert held['prompt'] not in train_prompts
        if r['arm'][0]=='F':assert tr['initial_parameter_hash']==tr['final_parameter_hash']
        expected_steps=(640 if history=='N_OLD' else 8320) if r['arm'][0]=='P' else 0
        assert tr['final_diagnostics']['parameter_steps']==expected_steps
        cp=HERE/tr['final_checkpoint'];assert sha(cp)==tr['final_checkpoint_sha256']
        certificate=json.loads(cp.with_suffix('.certificate.json').read_text());assert certificate['sha256']==sha(cp)
    report=dict(accepted=True,receipt_sha256=sha(path/'RECEIPT.json'),technical_records=rows,scientific_records=0,
                continuous_histories=3,target_events=target_count,actual_native_target_writes=write_count,
                independent_score_and_residual_reduction=True,trace_hash_verified=True)
    atomic_json(path/'AUDIT.json',report);return report


def metric_fixtures():
    from evaluate import sample_area,compute_area
    points=[dict(count=n,production_cpu=c,scores={'HELD':{'accuracy':a}})
            for n,c,a in [(0,2.,.25),(60,5.,.5),(120,8.,.75)]]
    assert abs(sample_integral(points)-sample_area(points))<1e-12
    for cstar in (1.,3.,6.,10.):assert abs(cpu_integral(points,cstar)-compute_area(points,cstar))<1e-12
    assert one_sided_p([0.,0.,0.,0.])==1.
    assert one_sided_p([1.,1.,1.,1.])==1/16
    assert one_sided_p([-1.,-1.])==1.
    assert one_sided_p([1.])==1.
    return dict(sample_integral=True,compute_integral_including_birth_time=True,zero_variance_statistics=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('job');a=ap.parse_args();print(json.dumps(audit_job(a.job)))
