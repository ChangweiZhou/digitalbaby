"""Start qualification followed by science in a durable detached session."""
import argparse
import json
import os
import subprocess
import sys
import bootstrap
from v2_integrity import identity
from compact_storage import atomic_json


def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); a=p.parse_args()
    r=bootstrap.ROOT/'results'; r.mkdir(exist_ok=True)
    if (r/'STATUS.json').exists(): raise ValueError('existing launch requires review; no automatic restart')
    log=(r/'supervisor.log').open('a')
    args=[sys.executable,'-u',str(bootstrap.ROOT/'supervisor.py')]+(['--prepare'] if a.prepare else [])
    proc=subprocess.Popen(args,cwd=bootstrap.ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                          start_new_session=True,close_fds=True)
    doc={'pid':proc.pid,'identity':identity(),'detached_session':True,'python':sys.executable,
         'log':str(r/'supervisor.log'),'authorized_science_after_three_cycles':True,'workers':4}
    atomic_json(r/'LAUNCH.json',doc); print(json.dumps(doc),flush=True)


if __name__=='__main__': main()
