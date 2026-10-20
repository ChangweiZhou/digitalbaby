"""Conservative read-only memory profile; no training or trajectory replay."""
from pathlib import Path
import sys,time,json,resource,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
start_cpu=time.process_time();start_wall=time.monotonic()
import bootstrap
from artifacts import HERE,atomic_json
from checkpoint import load
from data import split
models=[load(HERE/f'cycles/cycle3/jobs/202610209902_P-G/{h}_FINAL.npz') for h in ('W','N_OLD','SH_OLD')]
receipt=json.loads((HERE/'cycles/cycle3/jobs/202610209902_P-G/RECEIPT.json').read_text())
sets,_=split(202610209902)
for m in models:
 before=m.digest()
 for p,_ in sets['OLD'][:8]:m.generate(p)
 assert m.digest()==before
usage=resource.getrusage(resource.RUSAGE_SELF)
peak=usage.ru_maxrss if sys.platform=='darwin' else usage.ru_maxrss*1024
result=dict(record_applications=0,scientific_records=0,readonly_queries=24,
 CPU_seconds=time.process_time()-start_cpu,wall_seconds=time.monotonic()-start_wall,
 peak_RSS_bytes=int(peak),mutable_state_bytes_each=[m.diagnostics()['mutable_bytes'] for m in models],
 scope='Read-only conservative profiling workload holding three complete native histories plus a full receipt. This is not a retrospective peak-RSS measurement of the earlier full workers.',
 historical_worker_peak_RSS_measured=False,
 proposed_future_worker_reservation_bytes=int(peak*1.25),system_reserve_bytes=512*1024**2)
atomic_json(HERE/'cycles/cycle3/RESOURCE_PROFILE.json',result);print(json.dumps(result))
