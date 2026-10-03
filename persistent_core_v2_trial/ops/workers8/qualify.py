"""Eight-worker E0 execution test; no science worlds or endpoint selection."""
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OPS=Path(__file__).resolve().parent
ROOT=OPS.parent.parent
sys.path.insert(0,str(ROOT))
import bootstrap
from v2_audit import audit_receipt, read
from v2_integrity import verify
from compact_storage import atomic_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scientific_history(doc):
    rows=copy.deepcopy(doc['branches']['W'])
    # Object IDs and per-process timings differ; neither changes the transition.
    rows.pop('births')
    for row in rows['records']:
        row.pop('cpu_input_predict_s')
        row.pop('cpu_observe_flush_s')
    return rows


def main():
    source=verify()
    lock=json.loads((OPS/'SOURCE_LOCK.json').read_text())
    assert digest(ROOT/'supervisor.py')==lock['original_supervisor_sha256']
    assert digest(OPS/'supervisor8.py')==lock['supervisor8_sha256']
    amended=(ROOT/'supervisor.py').read_text()
    for before,after in lock['exact_dispatch_changes']:
        assert amended.count(before)==1
        amended=amended.replace(before,after)
    adapter=(OPS/'supervisor8.py').read_text()
    assert adapter.endswith(amended)
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    began=time.monotonic()
    def child(arm,replica):
        folder=OPS/'technical'/f'{arm}_{replica}'
        folder.mkdir(parents=True,exist_ok=True)
        receipt=folder/'receipts'/f'500103_{arm}_lifetime.json.gz'
        assert not receipt.exists(), 'Do not rerun an existing technical test'
        with (folder/'worker.log').open('w') as f:
            result=subprocess.run([sys.executable,'-u',str(ROOT/'worker.py'),'--development','--world','500103','--arm',arm,'--assay','lifetime','--folder',str(folder),'--short'],stdout=f,stderr=subprocess.STDOUT,env=environment,timeout=300)
        assert result.returncode==0, f'Worker failed: {folder}'
        doc=read(receipt)
        audit_receipt(doc,source,complete=False)
        reference=ROOT/'results/development/cycle3'/arm/'uninterrupted/receipts'/f'500103_{arm}_lifetime.json.gz'
        assert scientific_history(doc)==scientific_history(read(reference)), f'Concurrent transition mismatch: {arm}/{replica}'
        assert doc['peak_rss_bytes']<=1073741824
        return {'arm':arm,'replica':replica,'records_compared':64,'exact_scientific_history_and_state':True,'peak_rss_bytes':doc['peak_rss_bytes'],'worker_s':doc['worker_s'],'receipt':str(receipt.relative_to(ROOT))}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        jobs=[pool.submit(child,arm,r) for arm in ('V1','ERROR','REPLACE','BOTH') for r in (1,2)]
        trials=[job.result() for job in jobs]
    assert verify()==source
    result={'verdict':'PASS','workers':8,'identity':source,'supervisor8_sha256':lock['supervisor8_sha256'],'technical_world':500103,'no_science_trajectories_run':True,'trials':trials,'concurrent_state_parity':True,'dispatch_only_diff_audited':True,'elapsed_s':time.monotonic()-began,'limitation':'Short E0 concurrency test; full-life qualification remains the previously accepted frozen three-cycle qualification. No twofold speedup or hard runtime guarantee.'}
    atomic_json(OPS/'QUALIFICATION.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
