"""Single detached controller, one worker, no automatic retry; caffeinate follows controller."""
from common import *
import subprocess,signal

def main():
 source_check();assert not (CAM/'RUN_STARTED').exists()
 started=time.time();launch={'started_unix':started,'supervisor_pid':os.getpid(),'worker_pid':None,'caffeinate_pid':None,'carried_CPU_seconds':30.,'python':sys.executable,'threads':1,'science_records':4480,'science_targets':227148,'CPU_limit_seconds':900,'wall_limit_seconds':1800,'no_retries':True}
 save(CAM/'LAUNCH.json',launch);save(CAM/'STATUS.json',{'status':'STARTING'})
 caffeinate=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 launch['caffeinate_pid']=caffeinate.pid;save(CAM/'LAUNCH.json',launch)
 env=os.environ.copy()
 for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
 def limits():resource.setrlimit(resource.RLIMIT_CPU,(870,871))
 with (CAM/'ops/worker.log').open('x') as f:
  child=subprocess.Popen([sys.executable,'-u',str(OPS/'run.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,preexec_fn=limits)
  launch['worker_pid']=child.pid;save(CAM/'LAUNCH.json',launch);save(CAM/'PROCESS.json',{'supervisor_pid':os.getpid(),'worker_pid':child.pid,'caffeinate_pid':caffeinate.pid,'running':True})
  reason=None
  while child.poll() is None:
   wall=time.time()-started
   if wall>1800 or footprint()>268435456:
    reason='WALL_OR_OUTPUT_LIMIT';child.terminate()
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:child.kill();child.wait()
    break
   save(CAM/'MONITOR.json',{'updated_unix':time.time(),'worker_liveness':'Popen.poll active','supervisor_pid':os.getpid(),'worker_pid':child.pid,'wall_seconds':wall,'output_bytes':footprint()});time.sleep(2)
  rc=child.wait();save(CAM/'PROCESS.json',{'supervisor_pid':os.getpid(),'worker_pid':child.pid,'caffeinate_pid':caffeinate.pid,'running':False,'returncode':rc,'ended_unix':time.time()})
  if rc!=0 or reason:
   save(CAM/'SUPERVISOR_FAILURE.json',{'error':reason or f'worker exited {rc}','returncode':rc,'counters':load(CAM/'COUNTERS.json') if (CAM/'COUNTERS.json').exists() else {}});save(CAM/'STATUS.json',{'status':'FAILED','error':reason or f'worker exited {rc}'});print('V181_FAILED',rc,reason,flush=True)
  else:assert load(CAM/'STATUS.json')['status']=='COMPLETED';print('V181_CONTROLLER_COMPLETED',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  import traceback
  save(CAM/'SUPERVISOR_FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc()});save(CAM/'STATUS.json',{'status':'FAILED','error':repr(e)});raise
