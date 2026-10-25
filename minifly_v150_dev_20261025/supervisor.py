import sys,json,resource,subprocess,time,os
from pathlib import Path
H=Path(__file__).resolve().parent;t=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
with (H/'SUPERVISOR.jsonl').open('a') as f:
 f.write(json.dumps({'event':'START','pid':os.getpid(),'time':time.time()})+'\n');f.flush();os.fsync(f.fileno())
 wake=subprocess.Popen(['caffeinate','-w',str(os.getpid())]) if sys.platform=='darwin' else None
 with (H/'run.log').open('w') as log:q=subprocess.run([sys.executable,str(H/'run.py')],stdout=log,stderr=subprocess.STDOUT)
 rr=resource.getrusage(resource.RUSAGE_CHILDREN);f.write(json.dumps({'event':'EXIT','code':q.returncode,'wall':time.monotonic()-t,'child_CPU':rr.ru_utime+rr.ru_stime-before.ru_utime-before.ru_stime,'peakRSS':rr.ru_maxrss})+'\n');f.flush();os.fsync(f.fileno())
 if wake is not None:wake.terminate()
sys.exit(q.returncode)
