"""E12 orchestration repair: validate only the completed job; preserve scientific source/settings."""
import fcntl, json, os, time
from pathlib import Path
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor,as_completed
import e12_campaign as C
R,O=C.R,C.O

def receipt_path(item):
    key,cmd,env,part=item
    if cmd[0]=='exp12_reminder.py':return O/f'e12_reminder_{cmd[2]}_r{cmd[3]}_{cmd[1]}.json'
    share=Fraction(env.get('E8_REPLAY_SHARE','1/2'));schedule=env.get('E8_SCHEDULE','blocked')
    tag='' if share==Fraction(1,2) and schedule=='blocked' else f'_rp{share.numerator}of{share.denominator}_{schedule}'
    batch=int(env.get('E8_BATCH','64')); cap=int(env.get('E8_BUFFER_CAP','1024'))
    tag+=(f'_b{batch}' if batch!=64 else '')+(f'_cap{cap}' if cap!=1024 else '')+('_recency' if env.get('E8_REPLAY_SAMPLING','uniform')!='uniform' else '')
    return O/f'{env["E8_RUN_PREFIX"]}e8_life_{cmd[1]}{tag}_{cmd[-1]}.json'

def validate(item):
    p=receipt_path(item);d=C.read(p)
    if item[1][0]=='exp12_reminder.py':
        assert d['updates']==int(item[1][-1]) and len(d['after'])==8
    else:
        expected=3 if item[3]=='A' else 10
        assert d['seed']==int(item[1][-1]) and len(d['stages'])==expected and d['stages'][-1]['task']=='compose',str(p)
        for stage in d['stages']:
            for key,row in stage['scores'].items():
                if key=='compose':continue
                assert 0<=row['exact']<=row['n'] and 0<=row['final']<=row['n']
        assert set(d['stages'][-1]['scores']['compose'])=={'+-','-+','><','<>','within'}
    return p

def main():
    guard=(O/'E12_CAMPAIGN.pidlock').open('w');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for n,h in C.read(R/'E12_LOCK.json')['files'].items():assert C.sha(R/n)==h,n
    lock=C.read(R/'E12_RESUME_LOCK.json')
    assert C.sha(R/'e12_resume.py')==lock['resume_source_sha256']
    manifest=lock['jobs']; items=[(d['key'],d['command'],d['env'],d['part']) for d in manifest]
    completed=[];waiting=[]
    prior=C.read(O/'E12_STATUS.json')
    if prior['status']!='partial_failed':raise RuntimeError('Resume expected stopped first attempt')
    profile=C.read(O/'E12_JOB_profile.json');assert profile['status']=='complete';completed.append(profile)
    for item in items:
        j=O/f'E12_JOB_{item[0]}.json'
        row=C.read(j) if j.exists() else None
        if row and row['status']=='complete':validate(item);completed.append(row)
        else:waiting.append(item)
    assert len(completed)==34 and len(waiting)==15,(len(completed),len(waiting))
    t0=time.monotonic();failed=set()
    def status(active):
        C.save(O/'E12_STATUS.json',dict(status='running',pid=os.getpid(),completed=completed,active=active,queued=[i[0] for i in waiting],failed_parts=sorted(failed),resume_from='E12_RESUME_LOCK.json',elapsed_seconds=prior['elapsed_seconds']+time.monotonic()-t0))
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending={}
        while waiting or pending:
            while waiting and len(pending)<6:
                item=waiting.pop(0)
                if item[3] not in failed:pending[pool.submit(C.run,*item)]=item
            status([i[0] for i in pending.values()]);C.e12_report.main()
            if not pending:break
            future=next(as_completed(pending));item=pending.pop(future)
            try:
                row=future.result()
                if row['status']=='complete':validate(item)
            except Exception as exc:row=dict(key=item[0],part=item[3],status='failed',error=repr(exc))
            completed.append(row)
            if row['status']=='failed':
                failed.add(item[3])
                for other in pending.values():
                    if other[3]==item[3]:
                        p=O/f'E12_JOB_{other[0]}.json'
                        if p.exists():
                            j=C.read(p)
                            if j.get('status')=='running':
                                try:os.kill(j['pid'],15)
                                except ProcessLookupError:pass
    C.save(O/'E12_STATUS.json',dict(status='complete' if len(completed)==49 and not failed else 'partial_failed',completed=completed,failed_parts=sorted(failed),resume_from='E12_RESUME_LOCK.json',elapsed_seconds=prior['elapsed_seconds']+time.monotonic()-t0))
    C.e12_report.main()
if __name__=='__main__':main()
