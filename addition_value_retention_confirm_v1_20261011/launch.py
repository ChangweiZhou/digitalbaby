"""One-time detached launch; no shell terminal ownership and no auto restart."""
import os,subprocess,sys,time
from confirm_ops import ROOT,RESULTS,OPS,read,atomic,verify_sources
def launch():
    verify_sources()
    if read(ROOT/'QUALIFICATION.json')['verdict']!='PASS':raise ValueError('not qualified')
    if (RESULTS/'LAUNCH.json').exists() or (OPS/'DETACHED_PROCESS.json').exists():raise ValueError('already launched')
    OPS.mkdir(exist_ok=True)
    with (OPS/'supervisor.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'supervisor.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    atomic(OPS/'DETACHED_PROCESS.json',dict(pid=p.pid,time_unix=time.time(),start_new_session=True,stdin='DEVNULL',stdout=str(OPS/'supervisor.log'),interpreter=sys.executable),exclusive=True)
    for i in range(60):
        if p.poll() is not None:raise RuntimeError('supervisor exited: '+(OPS/'supervisor.log').read_text()[-4000:])
        if (RESULTS/'LAUNCH.json').exists():
            print(read(RESULTS/'LAUNCH.json'));return
        time.sleep(.2)
    raise TimeoutError('launch not acknowledged; preserve process, do not launch a duplicate')
if __name__=='__main__':launch()
