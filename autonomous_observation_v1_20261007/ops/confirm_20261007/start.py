"""One explicitly authorized detached launch; no interactive terminal dependency."""
import os
import subprocess
import sys
import time
from ops_common import OPS, ROOT, RESULTS, atomic, counts, read, verify_sources


def main():
    source = verify_sources()
    if read(OPS/'QUALIFICATION.json')['verdict'] != 'PASS' or read(OPS/'FINISH_TEST.json')['verdict'] != 'PASS':
        raise ValueError('launch qualification incomplete')
    if (RESULTS/'LAUNCH.json').exists() or counts()['committed_worlds']:
        raise ValueError('experiment already launched; do not repeat')
    atomic(RESULTS/'LAUNCH_REQUEST.json',dict(time_unix=time.time(),source_lock_sha256=source,
           authorization='用户：替我跑起来。',detached=True),exclusive=True)
    logpath=OPS/'supervisor.log'
    with logpath.open('x') as log:
        p=subprocess.Popen([sys.executable,'-u',str(OPS/'supervisor.py')],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True,env=os.environ.copy())
    atomic(RESULTS/'DETACHED_PROCESS.json',dict(supervisor_pid=p.pid,
           log=str(logpath),start_new_session=True,stdin='DEVNULL',time_unix=time.time()),exclusive=True)
    print(f'Detached supervisor PID {p.pid}; log={logpath}',flush=True)


if __name__=='__main__':
    main()
