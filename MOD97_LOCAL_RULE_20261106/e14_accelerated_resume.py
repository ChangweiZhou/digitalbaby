"""Qualified inference handover after in-flight births finish; no training worker is interrupted."""
import fcntl,os,signal,subprocess,time,json,shutil,datetime
from pathlib import Path
import e14_campaign as C
R,O=C.R,C.O

def process_states(pids):
 p=subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid=,state='],capture_output=True,text=True)
 return {int(line.split()[0]):line.split()[1] for line in p.stdout.splitlines() if line.split()}

def main():
 hand=(O/'E14_ACCEL_HANDOVER.lock').open('a');fcntl.flock(hand,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status=C.read(O/'E14_STATUS.json');old_pid=status['pid'];paused=False;replaced=False
 assert status['status']=='births','Use only the birth-to-life boundary'
 qual=C.read(R/'E14_ACCELERATION_STATUS.json');assert qual['scientific_receipt_equal'] and qual['full_112_diagnostic_checkpoints_exact']
 for n,h in C.read(R/'E14_LOCK.json')['files'].items():assert C.sha(R/n)==h,n
 for n,h in C.read(R/'E14_ACCELERATION_LOCK.json')['files'].items():assert C.sha(R/n)==h,n
 births,jobs,profile=C.plan();birth_keys={j['key'] for j in births}
 assert all(j['key'] in birth_keys or j['key'] in ('C_wd1_73129','profile_E7_73129') for j in status['completed'])
 command=subprocess.check_output(['ps','-p',str(old_pid),'-o','command='],text=True);assert 'e14_campaign.py' in command
 workers={j['key']:C.read(O/f'E14_JOB_{j["key"]}.json') for j in births}
 starts={key:(O/f'E14_JOB_{key}.json').stat().st_mtime for key in workers}
 archive=O/'e14_execution_before_acceleration';archive.mkdir()
 for p in [O/'E14_STATUS.json',O/'logs/e14_campaign.log',*[O/f'E14_JOB_{j["key"]}.json' for j in status['completed']],*[O/f'E14_JOB_{j["key"]}.json' for j in births]]:
  dest=archive/p.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 original_log=(O/'logs/e14_campaign.log').stat()
 original_elapsed=time.time()-getattr(original_log,'st_birthtime',original_log.st_ctime);handover_start=time.monotonic()
 try:
  # Suspend only the scheduler so no future life starts during the handover.
  os.kill(old_pid,signal.SIGSTOP);paused=True
  # If a life raced ahead before suspension, abort the handover and resume the old scheduler.
  for j in jobs:
   if j['key']=='C_wd1_73129':continue
   assert not (O/f'E14_JOB_{j["key"]}.json').exists(),'A scientific life already started; do not interrupt it'
  C.save(R/'E14_SPEED_HANDOVER.json',dict(status='waiting_for_existing_births',old_controller_pid=old_pid,new_controller_pid=os.getpid(),scientific_workers_interrupted=False,qualification=qual,notes='Only original scheduler suspended. All six in-flight training workers continue.'))
  worker_ids=[w['pid'] for w in workers.values() if w.get('status')=='running']
  while True:
   states=process_states(worker_ids)
   if all(pid not in states or states[pid].startswith('Z') for pid in worker_ids):break
   time.sleep(5)
  # Every birth worker is now exited/zombie; do not kill any model trajectory.
  complete={j['key']:j for j in status['completed']}
  born_files={}
  for j in births:
   C.validate(j);row=C.read(O/f'E14_JOB_{j["key"]}.json')
   if row.get('status')!='complete':
    completed_time=(O/j['metadata']).stat().st_mtime
    row=dict(**j,status='complete',returncode=0,log=workers[j['key']]['log'],wall_seconds=max(0.,completed_time-starts[j['key']]),wall_seconds_method='Lower bound: completed metadata timestamp minus original job-start metadata timestamp; excludes exit cleanup',adopted_after_worker_exit=True)
    C.save(O/f'E14_JOB_{j["key"]}.json',row)
   complete[j['key']]=row;born_files[j['key']]={n:C.sha(O/n) for n in (j['receipt'],j['metadata'])}
  assert len(complete)==8
  C.save(R/'E14_BIRTHS.json',dict(status='complete',files=born_files,adopted_without_retraining=True))
  # No worker remains alive. Terminate the suspended old scheduler and release its lock.
  os.kill(old_pid,signal.SIGKILL);replaced=True;paused=False
  guard=(O/'E14_CONTROLLER.lock').open('a')
  for _ in range(100):
   try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:time.sleep(.1)
  else:raise RuntimeError('Old scheduler lock not released')
  runtime=['e14_fast_inference.py','e14_acceleration_worker.py','e14_accelerated_resume.py']
  C.save(R/'E14_ACCELERATION_DEPLOYMENT_LOCK.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='User asked to find and apply ways to speed E14; full equivalence qualification passed before deployment',scope='Inference only for future unstarted lives; original training source and all scientific settings unchanged',qualified_path='C73129 full scientific receipt and112 diagnostic checkpoints exact;128-update weights/optimizer/RNG/store/reply parity. Not a proof for every future state.',backend='Compute final window only in stateless response generation; no change to training forward/backward, optimizer, data, seeds, dose, gradients or diagnostic cadence',runtime_source_sha256={n:C.sha(R/n) for n in runtime},original_lock_sha256=C.sha(R/'E14_LOCK.json'),births_adopted=6,interrupted_training_workers=0,original_controller_pid=old_pid,first_8_completed_jobs=list(complete),policy='Exact parity remains mandatory on all logged reruns. Any technical/parity failure stops its affected part without tuning or relaxing tolerances. Qualification overhead separately reported.'))
  completed=list(complete.values());remaining=[j for j in jobs if j['key'] not in complete]
  assert len(remaining)==38
  def worker(j):
   for n,h in C.read(R/'E14_ACCELERATION_DEPLOYMENT_LOCK.json')['runtime_source_sha256'].items():assert C.sha(R/n)==h,n
   actual={**j,'command':['e14_acceleration_worker.py',*j['command']],'original_command':j['command'],'inference_backend':'qualified_final_window_v1'}
   return C.run(actual)
  def update(done,active,queued,failed):
   C.save(O/'E14_STATUS.json',dict(status='running',pid=os.getpid(),completed=completed+done,active=active,queued=queued,failed_parts=sorted(failed),planned_subprocesses=46,elapsed_seconds=original_elapsed+time.monotonic()-handover_start,controller='e14_accelerated_resume.py',acceleration_deployment='E14_ACCELERATION_DEPLOYMENT_LOCK.json',training_workers_interrupted=0))
  C.save(R/'E14_SPEED_HANDOVER.json',dict(status='deployed_for_future_jobs',old_controller_pid=old_pid,new_controller_pid=os.getpid(),births_adopted=6,scientific_workers_interrupted=False,extra_validation_run_count=1))
  done,failed=C.queue(remaining,worker,C.validate,update);completed+=done
  C.save(O/'E14_STATUS.json',dict(status='complete' if len(completed)==46 and not failed else 'partial_failed',completed=completed,failed_parts=sorted(failed),planned_subprocesses=46,elapsed_seconds=original_elapsed+time.monotonic()-handover_start,controller='e14_accelerated_resume.py',acceleration_deployment='E14_ACCELERATION_DEPLOYMENT_LOCK.json',training_workers_interrupted=0))
 except BaseException as exc:
  C.save(R/'E14_SPEED_HANDOVER_ERROR.json',dict(error=repr(exc),old_controller_replaced=replaced))
  raise
 finally:
  if paused and not replaced:os.kill(old_pid,signal.SIGCONT)
if __name__=='__main__':main()
