"""Synthetic constant-time status and fail-closed finishing qualification."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from runner import atomic_json

class Monitor:
    def __init__(self):
        self.lock=threading.Lock();self.count=0;self.receipt_reads=0
    def commit(self):
        with self.lock:self.count+=1
    def status(self):
        # The monitor never opens or decompresses receipts under its lock.
        with self.lock:return {'committed_jobs':self.count,'receipt_reads':self.receipt_reads}

def finish_allowed(count,expected,audit,failed=False):
    return count==expected and audit is True and failed is False

def suite(folder):
    folder.mkdir(parents=True,exist_ok=False)
    m=Monitor();begin=time.monotonic()
    def writer():
        for i in range(500):m.commit()
    def reader():
        for i in range(1000):
            s=m.status()
            if s['receipt_reads']!=0:raise AssertionError('monitor opened receipts')
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(writer) for _ in range(4)]+[pool.submit(reader) for _ in range(4)]
        for f in futures:f.result(timeout=10)
    if m.count!=2000:raise AssertionError('lost status commit')
    accepted=finish_allowed(12,12,True)
    negatives=[finish_allowed(11,12,True),finish_allowed(12,12,False),finish_allowed(12,12,True,True)]
    if not accepted or any(negatives):raise AssertionError('finishing gate')
    result={'verdict':'PASS_SYNTHETIC_ONLY','concurrent_status_reads':4000,
            'synthetic_commits':2000,'receipt_decompressions':0,
            'negative_finish_cases':3,'science_records':0,
            'wall_seconds':time.monotonic()-begin}
    atomic_json(result,folder/'RESULT.json');return result
