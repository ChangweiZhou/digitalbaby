"""Final cycle3 resource/concurrency probe, 16 explicitly charged records."""
import argparse
import json
import multiprocessing
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import bootstrap
from bind_kernel import Learner
from bind_artifacts import Ledger,write,admit_technical,source_identity


def one_worker(index,directory):
    directory=Path(directory)/f"worker{index}"
    if (directory/"EXECUTION_LEDGER.json").exists():
        raise ValueError("technical resource probe is never replayed into same directory")
    ledger=Ledger(directory);begin=time.time();cpu=time.process_time()
    model=Learner("BIND-FB",202610166003)
    init_cpu=time.process_time()-cpu
    records=[(b"9+9=",b"18"),(b"8+8=",b"16"),(b"9+8=",b"17"),(b"8+9=",b"17")]
    result=ledger.call("full-length-addition-native-write-tape",model,lambda:model.train_window(records))
    start=time.process_time();model.generate(b"9+7=",cap=16);query_cpu=time.process_time()-start
    maximum=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maximum_bytes=maximum if sys.platform=="darwin" else maximum*1024
    out=dict(worker=index,pid=os.getpid(),records=4,digest=model.digest(),
             init_cpu=init_cpu,production_cpu=model.resource["production_cpu"],one_query_cpu=query_cpu,
             peak_rss_bytes=maximum_bytes,wall_seconds=time.time()-begin,summary=result)
    write(directory/"RESULT.json",out)
    return out


def main(out):
    remaining=256-admit_technical()
    if remaining<16:raise ValueError("technical budget cannot admit resource probe")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    began=time.time()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context("spawn")) as pool:
        results=list(pool.map(one_worker,range(4),[str(out)]*4))
    assert len({r["digest"] for r in results})==1
    assert len({r["pid"] for r in results})==4
    # Admission is a conservative planning estimate, never a performance result.
    rho=max(r["production_cpu"]/4 for r in results)
    query=max(r["one_query_cpu"] for r in results)
    init=max(r["init_cpu"] for r in results)
    # 8 sample and at most 6 CPU checkpoints; all100 facts in5 boundary probes.
    # The sum intentionally double-counts overlapping checkpoints conservatively.
    queries_per_history=20*14+100*5
    estimates={}
    for stage,jobs,records in (("DEV",24,49152),("CONFIRM",64,262144)):
        histories=jobs*2
        cpu=records*rho+histories*queries_per_history*query+jobs*init
        # 2x measured slowest concurrent cost, plus30min analysis/IO allowance.
        hours=(2*cpu/4+1800)/3600
        estimates[stage]=dict(record_applications=records,full_jobs=jobs,
                             conservative_wall_hours=hours,eight_hour_projection_pass=hours<=8)
    report=dict(passed=True,concurrency_state_parity=True,workers=results,source_identity=source_identity(),
                records_executed=16,science_jobs=0,elapsed_seconds=time.time()-began,
                slowest_cpu_seconds_per_record=rho,slowest_query_cpu=query,
                concurrent_worker_peak_sum_bytes=sum(r["peak_rss_bytes"] for r in results),
                projections=estimates,scientific_launch_authorized=False,
                projection_limitations="Short technical profile, one host. Includes2x slowest concurrent cost and30minIO/analysis allowance; not a guaranteed runtime.")
    write(out/"RESULT.json",report)
    print({k:report[k] for k in ("concurrency_state_parity","records_executed","slowest_cpu_seconds_per_record","concurrent_worker_peak_sum_bytes","projections")})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);main(p.parse_args().out)
