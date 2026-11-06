"""Dispatch local experiment workers with bounded concurrency."""
from collections import deque
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def launch(spec,limit):
    queue=deque(spec)
    running=[]
    failures=[]
    while queue or running:
        while queue and len(running)<limit:
            script,args,path=queue.popleft()
            target=ROOT/path
            target.parent.mkdir(parents=True,exist_ok=True)
            log=target.open("a")
            p=subprocess.Popen([sys.executable,"-B",str(ROOT/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT)
            running.append((p,log,path))
        time.sleep(1)
        active=[]
        for p,log,path in running:
            result=p.poll()
            if result is None:active.append((p,log,path))
            else:
                log.close()
                print(path,result,flush=True)
                if result: failures.append(path)
        running=active
    return failures


if __name__=="__main__":
    jobs=json.loads(Path(sys.argv[1]).read_text())
    errors=launch(jobs,int(sys.argv[2]))
    if errors: raise SystemExit(f"Failed workers: {errors}")
