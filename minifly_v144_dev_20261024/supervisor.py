"""Bounded foreground Stage-B coordinator, two workers, no retries."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from v144_audit import HERE, write_json


def main():
    if (HERE/'ops/B_LAUNCH.json').exists():raise RuntimeError('no automatic repeat')
    wake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    workers=[]
    for arm in ['LEARN','FROZEN']:
        f=(HERE/'ops'/('B_'+arm+'.log')).open('x')
        p=subprocess.Popen([sys.executable,str(HERE/'native_diagnostic.py'),'--arm',arm],stdout=f,stderr=subprocess.STDOUT,env=env)
        workers.append((arm,p,f))
    write_json(HERE/'ops/B_LAUNCH.json',{'supervisor_pid':os.getpid(),'caffeinate_pid':wake.pid,
               'workers':{a:p.pid for a,p,f in workers},'source_lock':'cycles/CYCLE1_SOURCE_LOCK.json','retry':False})
    while True:
        status={a:p.poll() for a,p,f in workers}
        write_json(HERE/'ops/B_STATUS.json',{'workers_exit':status,'timestamp_unix':time.time(),
                   'running':any(x is None for x in status.values())})
        if all(x is not None for x in status.values()):break
        time.sleep(5)
    for a,p,f in workers:f.close()
    return 0 if all(x==0 for x in status.values()) else 1


if __name__=='__main__':raise SystemExit(main())
