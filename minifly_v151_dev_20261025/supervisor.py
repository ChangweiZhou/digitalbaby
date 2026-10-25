"""One supervised finite session; no retry/resume or scientific changes."""
import os,sys,time,subprocess,resource,json
from pathlib import Path
H=Path(__file__).resolve().parent
start=time.monotonic();env=os.environ.copy();env['V151_WALL_START']=str(time.time())
with (H/'SUPERVISOR.jsonl').open('a') as f:
    f.write(json.dumps({'event':'START','pid':os.getpid(),'time':time.time()})+'\n');f.flush();os.fsync(f.fileno())
    awake=subprocess.Popen(['caffeinate','-w',str(os.getpid())]) if sys.platform=='darwin' else None
    try:
        with (H/'run.log').open('w') as log:p=subprocess.run([sys.executable,str(H/'run.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        f.write(json.dumps({'event':'EXIT','code':p.returncode,'wall':time.monotonic()-start,'CPU_children':usage.ru_utime+usage.ru_stime,'RSS_peak_bytes':usage.ru_maxrss})+'\n');f.flush();os.fsync(f.fileno())
    finally:
        if awake is not None:awake.terminate()
sys.exit(p.returncode)
