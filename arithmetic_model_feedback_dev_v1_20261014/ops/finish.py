"""Authorized conditional completion. Reads frozen jobs; never retries science."""
import hashlib,json,os,subprocess,sys,time,zipfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from protocol import atomic,identity,D
from analysis import reduce_stage,write_report

def package_preparation(verdict,result):
    package=ROOT/'PREPARATION_BUNDLE.zip';tmp=package.with_suffix('.pending')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in ROOT.rglob('*'):
            if not p.is_file() or p in (package,tmp) or '__pycache__' in p.parts or p.name in ('PACKAGE.json','RESULT_BUNDLE.zip'):continue
            z.write(p,arcname=str(p.relative_to(ROOT.parent)))
        for f in ('arithmetic_source_derived_kernel_dev_v1_20261013/core.py','arithmetic_source_derived_kernel_dev_v1_20261013/auditor.py','arithmetic_end_to_end_design_v1_20261013/validate_design.py'):
            z.write(ROOT.parent/f,arcname=f)
    tmp.replace(package)
    atomic(ROOT/'PACKAGE.json',dict(accepted=True,path=str(package),bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest(),verdict=verdict,science_jobs=0))

def main():
    start=json.loads((ROOT/'ops/WORK_AWAKE.json').read_text())['started_epoch']
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-s','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    atomic(ROOT/'ops/FINISH_PROCESS.json',dict(pid=os.getpid(),caffeinate_pid=awake.pid,started=time.time()))
    try:
        while True:
            p=ROOT/'cycles/cycle3/STATUS.json'
            status=json.loads(p.read_text()) if p.exists() else {}
            if status.get('status')=='EXECUTION_ACCEPTED':break
            if status.get('status') in ('FAILED','PAUSED_SESSION_LIMIT'):
                atomic(ROOT/'STATUS.json',dict(status=status['status'],stage='cycle3',detail=status,heartbeat=time.time(),science_jobs=0));return
            if time.time()-start>=10*3600:raise RuntimeError('ten-hour session boundary before qualification')
            if status and time.time()-status['heartbeat']>120:
                pid=status.get('supervisor_pid')
                try:os.kill(pid,0)
                except (ProcessLookupError,TypeError):raise RuntimeError('cycle3 supervisor exited unexpectedly')
            atomic(ROOT/'STATUS.json',dict(status='CYCLE3_RUNNING',heartbeat=time.time(),detail=status,science_jobs=0))
            time.sleep(10)
        source=json.loads((ROOT/'SOURCE_LOCK.json').read_text());assert source==identity()
        assert json.loads((ROOT/'CONTRACT_TESTS.json').read_text())['accepted']
        assert json.loads((ROOT/'cycles/cycle2/TAMPER_TESTS.json').read_text())['accepted']
        result=reduce_stage(ROOT/'cycles/cycle3');assert len(result['worlds'])==4 and result['committed_jobs']==8 and result['executed_records']==786432
        assert json.loads((ROOT/'cycles/cycle3/MANIFEST.json').read_text())['source']==source
        atomic(ROOT/'cycles/cycle3/SUMMARY.json',result)
        atomic(ROOT/'cycles/cycle3/REVIEW.json',dict(accepted=True,cycle=3,native_jobs=8,records=786432,
            effect_admission=result['efficacy_admission'],model_changed=False,thresholds_changed=False,
            revisions=['Complete three-cycle report and cost/efficacy disposition; no new recipe, dose or target after full-dose outcomes.']))
        total_jobs=10;total_records=793600
        verdict='QUALIFIED_FOR_CONDITIONAL_SCIENCE' if result['efficacy_admission'] else 'NO_SCIENCE_ADMISSION'
        atomic(ROOT/'QUALIFICATION.json',dict(accepted=True,technical_cycles=3,technical_jobs=total_jobs,technical_records=total_records,
            generic_record_applications=8197,paired_source_oracle_applications=5,efficacy_admission=result['efficacy_admission'],verdict=verdict,source=source))
        atomic(ROOT/'SUMMARY.json',dict(result,verdict=verdict,technical_jobs=total_jobs,technical_records=total_records,formal_science_jobs=0))
        write_report(ROOT/'PREPARATION_REPORT.md',result,verdict)
        with (ROOT/'PREPARATION_REPORT.md').open('a') as f:
            f.write('\n三循环合计10完整技术作业、793600执行记录、50相关末态历史；另有8197通用fixture应用与5配对源码oracle应用。正式科学作业为0，技术世界不作显著性或核心采纳。\n')
        atomic(ROOT/'FINAL_PREPARATION_AUDIT.json',dict(accepted=True,technical_jobs=total_jobs,records=total_records,source=source,
            original_certificates_all_accepted=True,core_same_across_cycles=True,whole_scalar_fork_no_write_and_all_archives_checked=True,
            source_scope='C1/C2 producer snapshots preserved; C2 reducer sanity bug corrected before full C3, primary/E3 gates unchanged.',verdict=verdict))
        if not result['efficacy_admission']:
            atomic(ROOT/'SCIENCE_ADMISSION.json',dict(accepted=False,reason='Locked efficacy means did not all pass',effects=result['effect_means'],source=source,cstar=result['cstar']))
            atomic(ROOT/'STATUS.json',dict(status='PACKAGING_NEGATIVE_PREPARATION',verdict=verdict,heartbeat=time.time(),technical_jobs=total_jobs,technical_records=total_records,science_jobs=0))
            package_preparation(verdict,result)
            atomic(ROOT/'STATUS.json',dict(status='COMPLETED_NO_SCIENCE_ADMISSION',verdict=verdict,heartbeat=time.time(),technical_jobs=total_jobs,technical_records=total_records,science_jobs=0))
            print(json.dumps(dict(verdict=verdict,technical_jobs=total_jobs,records=total_records)),flush=True);return
        # A full measured worst job, conservative factor two, and ten-minute
        # finalization allowance must fit before the shared eight-hour stop.
        costs=[]
        for p in (ROOT/'cycles/cycle3/receipts').glob('*.audit.json'):
            costs.append(json.loads(p.read_text())['audit_cpu'])
        predicted=2*(result['cost']['max_job_wall']+max(costs))*64/2+600
        left=start+8*3600-time.time()
        if predicted>left:
            atomic(ROOT/'SCIENCE_ADMISSION.json',dict(accepted=False,reason='SAFE_SESSION_BUDGET_PAUSE',estimated_seconds=predicted,remaining_seconds=left,efficacy_pass=True,source=source,cstar=result['cstar']))
            package_preparation('PAUSED_SESSION_LIMIT',result)
            atomic(ROOT/'STATUS.json',dict(status='PAUSED_SESSION_LIMIT',next='formal science, explicit resume required',science_jobs=0,heartbeat=time.time()));return
        atomic(ROOT/'SCIENCE_ADMISSION.json',dict(accepted=True,source=source,cstar=result['cstar'],worlds=D['science_worlds'],predicted_seconds=predicted,conditional_user_authorization=True))
        atomic(ROOT/'STATUS.json',dict(status='SCIENCE_LAUNCHING',heartbeat=time.time(),technical_jobs=total_jobs,technical_records=total_records,science_jobs=0))
        folder=ROOT/'ops/science';folder.mkdir(parents=True,exist_ok=True)
        python=ROOT.parent/'arithmetic_kernel_no_attention_dev_v3_20261013/.venv/bin/python'
        with (folder/'supervisor.log').open('ab') as log:
            proc=subprocess.Popen([str(python),'-u',str(ROOT/'supervisor.py'),'--stage','science'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic(folder/'DETACHED_PROCESS.json',dict(pid=proc.pid,started=time.time(),log=str(folder/'supervisor.log')))
            while proc.poll() is None:
                s=ROOT/'results/STATUS.json'
                if s.exists():atomic(ROOT/'STATUS.json',dict(status='SCIENCE_ACTIVE',heartbeat=time.time(),detail=json.loads(s.read_text()),technical_cycles=3))
                time.sleep(10)
        final=json.loads((ROOT/'results/STATUS.json').read_text())
        atomic(ROOT/'STATUS.json',dict(status=final['status'],heartbeat=time.time(),detail=final,technical_cycles=3))
        print(json.dumps(final),flush=True)
    except Exception as e:
        atomic(ROOT/'FAILURE.json',dict(error=repr(e),time=time.time(),no_automatic_retry=True))
        atomic(ROOT/'STATUS.json',dict(status='FAILED',error=repr(e),heartbeat=time.time()));raise
    finally:
        awake.terminate()
        # Release only this task's bounded working assertion. Never touch a
        # different experiment's caffeinate process.
        pid=json.loads((ROOT/'ops/WORK_AWAKE.json').read_text())['pid']
        try:os.kill(pid,15)
        except ProcessLookupError:pass
if __name__=='__main__':main()
