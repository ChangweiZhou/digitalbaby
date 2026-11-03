"""One detached worker, original hard limits, awake controller; never retries."""
from pathlib import Path
import sys,subprocess,resource,time,os,json
OPS=Path(__file__).resolve().parent;CAM=OPS.parent;DEV=CAM.parent;ROOT=DEV.parent;RUNTIME=Path('/private/tmp/minifly_v183_runtime')
def read(n):return json.loads((CAM/n).read_text())
def save(n,v):
 p=CAM/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def disk():return sum(p.stat().st_size for b in (DEV,RUNTIME) for p in b.rglob('*') if p.is_file())
def main():
 assert not (CAM/'RUN_STARTED').exists(),'No retry';a=read('AUTHORIZATION.json');started=time.time();awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 launch={'started_unix':started,'supervisor_pid':os.getpid(),'worker_pid':None,'caffeinate_pid':awake.pid,'CPU_carried':a['charged_CPU_carried'],'active_execution_wall_carried':a['active_execution_wall_carried'],'CPU_cap':900,'wall_cap':1800,'RSS_cap_bytes':1073741824,'output_cap_bytes':268435456,'threads':1,'no_automatic_retries':True};save('LAUNCH.json',launch)
 env=os.environ.copy()
 for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
 def limits():
  left=int(900-a['charged_CPU_carried']);resource.setrlimit(resource.RLIMIT_CPU,(left,left+1))
 with (OPS/'worker.log').open('x') as f:
  p=subprocess.Popen([sys.executable,'-u',str(OPS/'worker.py')],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,preexec_fn=limits);launch['worker_pid']=p.pid;save('LAUNCH.json',launch);reason=None
  while p.poll() is None:
   wall=time.time()-started+a['active_execution_wall_carried'];size=disk();save('PROCESS.json',{'updated_unix':time.time(),'supervisor_pid':os.getpid(),'worker_pid':p.pid,'caffeinate_pid':awake.pid,'worker_running':True,'actual_liveness':'Popen.poll active','charged_execution_wall_seconds':wall,'output_bytes_including_runtime':size})
   if wall>1800 or size>268435456:
    reason='WALL_OR_OUTPUT_LIMIT';p.terminate()
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill();p.wait()
    break
   time.sleep(2)
  rc=p.wait();save('PROCESS.json',{'updated_unix':time.time(),'supervisor_pid':os.getpid(),'worker_pid':p.pid,'caffeinate_pid':awake.pid,'worker_running':False,'returncode':rc})
  if rc!=0 or reason:
   state=read('STATUS.json') if (CAM/'STATUS.json').exists() else {};save('SUPERVISOR_EXIT.json',{'returncode':rc,'reason':reason or state.get('status') or f'worker exited {rc}'})
   if state.get('status')!='STOPPED_CONTROL_NOT_QUALIFIED':save('STATUS.json',{'status':'FAILED','error':reason or f'worker exited {rc}'})
   print('V184_STOPPED',rc,reason or state.get('status'),flush=True)
  else:assert read('STATUS.json')['status']=='COMPLETED';assert read('PACKAGE.json')['status']=='COMPLETE';print('V184_CONTROLLER_COMPLETED',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  import traceback
  save('SUPERVISOR_FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc()});save('STATUS.json',{'status':'FAILED','error':repr(e)});raise
