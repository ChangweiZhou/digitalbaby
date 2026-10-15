"""Launch the fixed DEV attempt detached, not a new scientific campaign."""
import subprocess
import os
import bootstrap
from artifacts import write,now

if __name__=='__main__':
    root=bootstrap.HERE;record=root/'ops/DETACHED_PROCESS.json'
    if record.exists() or (root/'results/LAUNCH.json').exists():raise FileExistsError('already launched')
    log=(root/'ops/supervisor.log').open('x')
    process=subprocess.Popen([os.sys.executable,str(root/'run_dev.py')],cwd=root,stdin=subprocess.DEVNULL,
                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    write(record,dict(utc=now(),pid=process.pid,detached=True,science=False,log='ops/supervisor.log'))
    print(process.pid)
