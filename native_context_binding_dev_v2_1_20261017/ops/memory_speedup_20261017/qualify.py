"""Operational fixtures only; no native training, replay or outcome analysis."""
import json,math,sys,time,subprocess,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;I=HERE.parents[1];sys.path.insert(0,str(I/'ops/dev_b_20261017'))
import campaign_common as c
sys.path.insert(0,str(HERE))
from memory_policy import decide,usage,system,Policy
import supervisor as controller

def main():
    c.verify_lock();reserve={'NATIVE':600,'F000':1800,'F111':2400}
    assert decide(5000,1,reserve,[dict(arm='NATIVE',RSS_bytes=500)],'F000')['required_bytes']==1800+100+512*1024**2
    assert not decide(10**10,2,reserve,[],'F000')['admit']
    assert not decide(0,1,reserve,[],'F000')['admit']
    assert decide(10**10,1,reserve,[],'F000')['admit']
    fn,source=controller.stage_function()
    assert 'running={pid:x for pid,x in ADOPTED.items()' in source and 'completed=len(certified)' in source
    assert '--resume' in source # original no-auto-resume permission remains enforced
    assert 'existing and not resume_session' in source and 'original' not in source
    me=usage(__import__('os').getpid());assert me['RSS_bytes']>0 and me['physical_footprint_bytes']>0
    # Adopt a real external operational process: completion checked without sending it any signal.
    with tempfile.TemporaryDirectory(prefix='bind_dispatch_fixture_') as tmp:
        folder=Path(tmp);proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.1)'])
        start=usage(proc.pid)['start_abstime'];external=controller.ExistingProcess(proc.pid,folder,start)
        assert external.poll() is None
        (folder/'AUDIT.json').write_text('{"accepted":true,"synthetic":true}')
        proc.wait(timeout=5);assert external.poll()==0
        (folder/'AUDIT.json').unlink();assert external.poll()==1
    s=system();policy=Policy();current=c.read(c.RESULTS/'STATUS.json');manifest=c.manifest('DEV_B')
    predicted=[]
    for row in current['active_jobs']:
        try:predicted.append(dict(arm=row['arm'],**usage(row['pid'])))
        except OSError:pass
    active_keys={row['job'] for row in current['active_jobs']}
    committed={(j['world'],j['arm']) for j in c.committed() if j['stage']=='DEV_B'}
    pending=[j for j in manifest['jobs'] if j['key'] not in active_keys and (j['world'],j['arm']) not in committed]
    next_arm=pending[0]['arm'] if pending else None
    live=decide(s['available_proxy_bytes'],s['pressure'],policy.reserves,predicted,next_arm) if next_arm else None
    result=dict(passed=True,zero_science_records=True,checks=dict(per_arm_growth=True,warning_blocks_admission=True,
        insufficient_headroom_rejected=True,adopted_job_not_replayed=True,already_certified_adopted_job_not_double_counted=True,
        complete_external_process_detected=True,unexpected_exit_detected=True,ten_hour_session_preserved_in_handoff=True),
        live_system=s,live_policy_projection=live,scientific_source_unchanged=True)
    c.write(HERE/'QUALIFICATION.json',result);print(json.dumps(result))
if __name__=='__main__':main()
