"""One sequential detached worker; no automatic retries, with caffeinate and audits."""
import os,sys,json,time,subprocess,resource,signal
from pathlib import Path
import runtime as C
TORCH='/Users/pencilbard/Documents/second_try/.venv/bin/python'
NATIVE='/Users/pencilbard/Downloads/second_try_copy/V88/.venv-rce-v87-followup/bin/python'
CHILD=None;START=C.usage();PEAK=0
def prior_cpu():return C.AUTH['base_CPU_allowance']+sum(json.loads(p.read_text())['CPU'] for p in (C.R/'ops').glob('qual_*_DONE.json'))+C.usage()+resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
def verify_lock():
 for p,d in json.loads((C.R/'SOURCE_LOCK.json').read_text())['files'].items():assert C.sha(C.ROOT/p)==d,p
def disk_bytes():return sum(p.stat().st_size for p in C.R.rglob('*') if p.is_file())+C.AUTH.get('base_output_bytes',0)
def worker(stage,python,script,args):
 global CHILD,PEAK
 verify_lock();env=dict(os.environ,V176_STAGE=stage,V176_PRIOR_CPU=str(prior_cpu()),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 C.save('STATUS.json',dict(status='RUNNING',stage=stage,supervisor_pid=os.getpid(),epoch=time.time()))
 with (C.R/f'ops/{stage}.log').open('ab') as log:
  CHILD=subprocess.Popen([python,str(C.R/script),*args],cwd=C.ROOT,env=env,stdout=log,stderr=log)
  C.save('PROCESS.json',dict(supervisor_pid=os.getpid(),worker_pid=CHILD.pid,stage=stage,epoch=time.time()))
  while CHILD.poll() is None:
   time.sleep(2)
   active=json.loads((C.R/'ACTIVE.json').read_text()) if (C.R/'ACTIVE.json').exists() else {}
   if active.get('pid')==CHILD.pid:PEAK=max(PEAK,active.get('peak_RSS_bytes',0))
   rss=0
   try:rss=int(subprocess.check_output(['/bin/ps','-o','rss=','-p',str(CHILD.pid)],text=True).strip())*1024
   except (ValueError,subprocess.CalledProcessError):pass
   PEAK=max(PEAK,rss)
   total=prior_cpu()+active.get('process_CPU',0) if active.get('pid')==CHILD.pid else prior_cpu()
   reason=None
   if total>C.AUTH['CPU_cap']:reason='CPU_CAP'
   if time.time()-C.AUTH['epoch']>C.AUTH['wall_cap']:reason='WALL_CAP'
   if rss>C.AUTH['RSS_cap']:reason='RSS_CAP'
   if disk_bytes()>C.AUTH['output_cap']:reason='OUTPUT_CAP'
   C.save('MONITOR.json',dict(status='RUNNING',stage=stage,worker_pid=CHILD.pid,epoch=time.time(),CPU_estimate=total,wall=time.time()-C.AUTH['epoch'],peak_RSS_bytes=PEAK,output_bytes=disk_bytes(),active=active))
   if reason:
    CHILD.send_signal(signal.SIGTERM);CHILD.wait(timeout=30);raise RuntimeError(reason)
  code=CHILD.returncode;CHILD=None
  if code:raise RuntimeError(f'{stage} exited {code}; see ops/{stage}.log and FAILURE.json')
 if not (C.R/f'ops/{stage}_DONE.json').exists():raise RuntimeError('MISSING_STAGE_RECEIPT:'+stage)
def main():
 global CHILD
 assert json.loads((C.R/'QUALIFICATION.json').read_text())['status']=='PASS'
 if (C.R/'LAUNCH.json').exists():raise RuntimeError('LAUNCH_ALREADY_EXISTS_NO_RETRY')
 caffeinate=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
 C.save('LAUNCH.json',dict(supervisor_pid=os.getpid(),caffeinate_pid=caffeinate.pid,epoch=time.time(),original_session_epoch=C.AUTH['epoch'],original_session_boundaries_carried=True,workers=1,numerical_threads=1,automatic_retry=False,qualified=True))
 try:
  worker('source',TORCH,'source_worker.py',['fit']);worker('bridge',NATIVE,'bridge_worker.py',['export']);worker('probe',TORCH,'source_worker.py',['probe'])
  gate=json.loads((C.R/'ADMISSION.json').read_text())
  if gate['native_B_admitted']:worker('native',NATIVE,'native_worker.py',[])
  worker('analysis',NATIVE,'analyze.py',[])
  summary=json.loads((C.R/'SUMMARY.json').read_text());assert json.loads((C.R/'FINAL_AUDIT.json').read_text())['status']=='ACCEPTED' and json.loads((C.R/'PACKAGE.json').read_text())['status']=='COMPLETE'
  C.save('STATUS.json',dict(status='COMPLETED',verdict=summary['verdict'],epoch=time.time(),CPU_total=prior_cpu(),peak_RSS_bytes=PEAK,source_records=summary['source_record_applications'],native_records=summary['native_records'],package='RESULT_BUNDLE.zip'))
 except BaseException as e:
  if CHILD is not None and CHILD.poll() is None:
   CHILD.send_signal(signal.SIGTERM)
   try:CHILD.wait(timeout=30)
   except subprocess.TimeoutExpired:CHILD.kill();CHILD.wait()
  failure=json.loads((C.R/'FAILURE.json').read_text()) if (C.R/'FAILURE.json').exists() else {}
  C.save('SUPERVISOR_FAILURE.json',dict(error=repr(e),worker_failure=failure,epoch=time.time(),CPU_total=prior_cpu(),automatic_retry=False))
  C.save('STATUS.json',dict(status='FAILED',error=repr(e),epoch=time.time(),preserve_files=True,automatic_retry=False));raise
 finally:
  if caffeinate.poll() is None:caffeinate.terminate();caffeinate.wait()
if __name__=='__main__':main()
