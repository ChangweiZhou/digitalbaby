import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy,json,time,traceback,ast
import bootstrap,config,data,models,job,analysis,audits
from artifacts import read,write,now


def rejects(call):
    try:call()
    except (ValueError,AssertionError,PermissionError,FileExistsError,FileNotFoundError):return
    raise AssertionError('unsafe action was not rejected')


def signature(receipt):
    return {h:dict(final_digest=r['final_digest'],
        curve=[(q['T'],q['phase'],q.get('operative_digest'),q['scores']) for q in r['curve']],
        sequence=[(r['id'],r['prompt'],r['answer'],r['loss'],r['allowed']) for r in r['rows']])
            for h,r in receipt['histories'].items()}


def run():
    folder=config.HERE/'cycles/cycle3';w=config.TECH_WORLDS[2];checks=[]
    rejects(lambda:job.run_addition(folder/'FORBIDDEN_SCIENCE',config.SCIENCE_WORLDS[0],'BLOCKED'))
    checks.append('formal_admission_refused_before_any_model_record')
    original=job.run_addition(folder/'UNINTERRUPTED',w,'INTERLEAVED',technical=True)
    paused=job.run_addition(folder/'RESTORED',w,'INTERLEAVED',technical=True,stop_after=2)
    assert paused['state']=='PAUSED' and paused['cursor']==2
    restored=job.run_addition(folder/'RESTORED',w,'INTERLEAVED',technical=True,resume=True)
    assert signature(original)==signature(restored)
    assert job.journal_cursor(folder/'RESTORED/journal.jsonl')==12
    rejects(lambda:job.run_addition(folder/'RESTORED',w,'INTERLEAVED',technical=True,resume=True))
    checks.append('real_three_history_job_uninterrupted_equals_safe_resumed_without_record_replay')
    bad=folder/'UNSAFE_JOURNAL.jsonl'
    job.journal(bad,dict(event='START',cursor=1,id='X'))
    rejects(lambda:job.journal_cursor(bad))
    checks.append('ambiguous_partial_record_refused')
    for variant in config.NATIVE_VARIANTS:
        p=folder/variant
        paused=job.run_native(p,w,variant,technical=True,stop_after=1)
        assert paused['state']=='PAUSED' and paused['cursor']==1
        result=job.run_native(p,w,variant,technical=True,resume=True)
        assert result['record_applications']==8 and read(p/'AUDIT.json')['accepted']
        assert job.journal_cursor(p/'journal.jsonl')==8
        assert len(result['histories']['W']['geometry']['rows'])>16
        rejects(lambda:job.run_native(p,w,variant,technical=True,resume=True))
    checks.append('both_real_native_job_pipelines_pause_restore_query_geometry_and_independent_audit')
    rejects(lambda:analysis.profile(restored))
    assert abs(analysis.area([(0,0.),(1,1.)])-.5)<1e-12
    rejects(lambda:analysis.area([(0,0.),(0,1.)]))
    profiles={(world,s):{'W':{'HELD_AUC':.2 if s=='INTERLEAVED' else .1}} for world in config.SCIENCE_WORLDS for s in config.SCHEDULES}
    reduced=analysis.paired(profiles)
    assert abs(reduced['mean']-.1)<1e-12 and reduced['n_worlds']==8
    profiles.pop(next(iter(profiles)));rejects(lambda:analysis.paired(profiles))
    checks.append('fixed_HELD_integral_paired_roster_and_technical_science_boundary')
    bad=copy.deepcopy(restored);bad['histories']['W']['curve'][0]['shortcuts']['envelope']+=.1
    rejects(lambda:audits.audit_addition_job(bad))
    checks.append('independent_complete_job_audit_rejects_shortcut_tampering')
    for path in config.HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    return dict(accepted=True,checks=checks,technical_only=True,training_record_applications_planned=40,
       full_science_jobs_executed=0,full_dose_behavior_tested=False,capability_improvement_demonstrated=False,
       production_data_untrained=True)

if __name__=='__main__':
    start=time.monotonic()
    try:r=run()
    except BaseException as e:
        write(config.HERE/'cycles/cycle3/RESULT.json',dict(accepted=False,error=repr(e),traceback=traceback.format_exc(),utc=now()));raise
    r.update(utc=now(),elapsed=time.monotonic()-start)
    write(config.HERE/'cycles/cycle3/RESULT.json',r);print(r)
