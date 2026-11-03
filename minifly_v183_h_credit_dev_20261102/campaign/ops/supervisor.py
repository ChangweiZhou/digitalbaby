"""One detached worker, original ceilings, caffeinate, no retries."""
from pathlib import Path
import subprocess,sys,os,time,json,resource
OPS=Path(__file__).resolve().parent;CAM=OPS.parent;DEV=CAM.parent
RUNTIME=Path('/private/tmp/minifly_v183_runtime')
def read(p):return json.loads(Path(p).read_text())
def save(name,v):
 p=CAM/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def disk():return sum(p.stat().st_size for base in (DEV,RUNTIME) for p in base.rglob('*') if p.is_file())
def main():
 assert not (CAM/'RUN_STARTED').exists(),'No retry'
 auth=read(CAM/'AUTHORIZATION.json');started=time.time();launch={'started_unix':started,'supervisor_pid':os.getpid(),'worker_pid':None,'caffeinate_pid':None,'CPU_carried':auth['charged_CPU_carried'],'active_wall_carried':auth['active_wall_carried'],'CPU_cap':900,'wall_cap':1800,'RSS_cap_bytes':1073741824,'output_cap_bytes_including_runtime':268435456,'threads':1,'no_automatic_retries':True};save('LAUNCH.json',launch)
 awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);launch['caffeinate_pid']=awake.pid;save('LAUNCH.json',launch)
 env=os.environ.copy()
 for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
 def limits():
  left=int(900-auth['charged_CPU_carried']);resource.setrlimit(resource.RLIMIT_CPU,(left,left+1))
 with (OPS/'worker.log').open('x') as out:
  child=subprocess.Popen([sys.executable,'-u',str(OPS/'worker.py')],cwd=DEV.parent,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,preexec_fn=limits)
  launch['worker_pid']=child.pid;save('LAUNCH.json',launch);reason=None
  while child.poll() is None:
   wall=time.time()-started+auth['active_wall_carried'];size=disk();save('PROCESS.json',{'updated_unix':time.time(),'supervisor_pid':os.getpid(),'worker_pid':child.pid,'caffeinate_pid':awake.pid,'worker_running':True,'actual_liveness':'Popen.poll active','charged_execution_wall_seconds':wall,'output_bytes_including_runtime':size})
   if wall>1800 or size>268435456:
    reason='WALL_OR_OUTPUT_LIMIT';child.terminate()
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:child.kill();child.wait()
    break
   time.sleep(2)
  rc=child.wait();save('PROCESS.json',{'updated_unix':time.time(),'supervisor_pid':os.getpid(),'worker_pid':child.pid,'caffeinate_pid':awake.pid,'worker_running':False,'returncode':rc})
  if rc!=0 or reason:
   save('SUPERVISOR_FAILURE.json',{'error':reason or f'worker exited {rc}','returncode':rc});save('STATUS.json',{'status':'FAILED','error':reason or f'worker exited {rc}'});print('V183_FAILED',rc,reason,flush=True)
  else:assert read(CAM/'STATUS.json')['status']=='COMPLETED';assert read(CAM/'PACKAGE.json')['status']=='COMPLETE';print('V183_CONTROLLER_COMPLETED',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  import traceback
  save('SUPERVISOR_FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc()});save('STATUS.json',{'status':'FAILED','error':repr(e)});raise
