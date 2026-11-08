"""Wait without importing a learner, then hand over to the locked controller."""
import os
import sys
import time
from datetime import datetime,timezone
from e16wr_common import ROOT,OUT,save,sha,verify_lock,require_machine_free

def main():
    verify_lock()
    if (OUT/'e16wr_campaign.guard').exists():raise RuntimeError('Campaign guard exists; investigate instead of launching twice')
    guard=OUT/'e16wr_wait.guard';guard.mkdir()
    owner=dict(status='WAITING_FOR_MACHINE',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
               launcher_sha256=sha(ROOT/'e16wr_wait.py'),lock_sha256=sha(ROOT/'E16WR_LOCK.json'),network_imported=False)
    save(guard/'owner.json',owner)
    last=None
    while True:
        try:
            gate=require_machine_free()
            break
        except RuntimeError as ex:
            message=str(ex)
            # Distinct PIDs can change; unchanged busy status needs no repeated log spam.
            if message!=last:
                print(datetime.now(timezone.utc).isoformat(),message,flush=True);last=message
            time.sleep(15)
    verify_lock()
    owner.update(status='HANDING_OVER_TO_CONTROLLER',gate=gate)
    save(guard/'owner.json',owner)
    print('Gate clear; starting locked technical qualification and parity gate.',flush=True)
    os.chdir(ROOT)
    os.execv(sys.executable,[sys.executable,str(ROOT/'e16wr_campaign.py')])

if __name__=='__main__':main()
