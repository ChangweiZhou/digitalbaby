"""Cycle3 four-process compatibility test; one4-record train window total."""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
import bootstrap
import torch
from bind_kernel import Learner,science_identity
from bind_artifacts import Ledger,write
from qualify import raw_records

OUT=bootstrap.HERE/'cycles/cycle3/attempt3'

def worker(index):
    if index not in range(4):raise ValueError('fixed technical workers only')
    arm=('NATIVE','F000','F111','F111')[index]
    model=Learner(arm,202610177003)
    before=model.digest();cpu=time.process_time()
    answer,status,_=model.generate(raw_records()[0][0],cap=16)
    probe_cpu=time.process_time()-cpu
    assert before==model.digest() and answer==b'\x00'*16 and status=='TRUNCATED'
    training=0
    if index==3:
        ledger=Ledger(OUT,3)
        ledger.call('one-four-worker-gradient-window',model,lambda:model.train_window(raw_records()))
        model.rest(86400)
        reference=torch.load(bootstrap.HERE/'cycles/cycle3/attempt2/valid.pt',map_location='cpu',weights_only=False)
        assert model.digest()==reference['digest']
        training=4
    write(OUT/f'WORKER_{index}.json',dict(passed=True,pid=os.getpid(),session=os.getsid(0),
          arm=arm,world=202610177003,numerical_threads=torch.get_num_threads(),
          records=training,full_cap_generation_CPU=probe_cpu,
          peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          production_resources=model.resource,science_jobs=0,model_identity=science_identity()))

def main():
    if OUT.exists():raise ValueError('fresh operations attempt required')
    OUT.mkdir();started=time.time();children=[];files=[]
    try:
        for index in range(4):
            log=(OUT/f'worker{index}.log').open('w');files.append(log)
            children.append(subprocess.Popen([sys.executable,__file__,'--worker',str(index)],
                         stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        write(OUT/'LAUNCH.json',dict(technical_only=True,science=False,pids=[p.pid for p in children],workers=4))
        for child in children:
            if child.wait(timeout=120)!=0:raise RuntimeError('technical worker failed;see immutable logs')
        results=[json.loads((OUT/f'WORKER_{i}.json').read_text()) for i in range(4)]
        assert all(r['passed'] and r['session']==r['pid'] and r['numerical_threads']==1 and r['science_jobs']==0 for r in results)
        assert sum(r['records'] for r in results)==4
        write(OUT/'RESULT.json',dict(passed=True,cycle=3,part='FOUR_PROCESS_COMPATIBILITY',workers=4,
             full_record_applications=4,science_jobs=0,elapsed_seconds=time.time()-started,results=results))
        print(json.dumps(dict(passed=True,workers=4,records=4,seconds=time.time()-started)))
    except Exception as exc:
        for child in children:
            if child.poll() is None:child.terminate()
        write(OUT/'FAILURE.json',dict(error=repr(exc),science_jobs=0));raise
    finally:
        for f in files:f.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);a=p.parse_args()
    main() if a.worker is None else worker(a.worker)
