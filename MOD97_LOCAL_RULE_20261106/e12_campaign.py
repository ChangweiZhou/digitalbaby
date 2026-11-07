"""User-authorized E12. No tuning. Baseline parity gates the campaign."""
import fcntl, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from e9_campaign import stable, save, read
import e12_report
R=Path(__file__).resolve().parent; O=R/'exp45'; SEEDS=(73129,93683,116089)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(key,cmd,env,part):
    start=time.monotonic(); log=O/'logs'/f'e12_{key}.log';log.parent.mkdir(exist_ok=True)
    receipt=dict(key=key,part=part,command=cmd,env=env,log=str(log.relative_to(R)))
    with log.open('w') as f:
        p=subprocess.Popen([sys.executable,'-u',*cmd],cwd=R,env={**os.environ,**env},stdout=f,stderr=subprocess.STDOUT)
        save(O/f'E12_JOB_{key}.json',{**receipt,'status':'running','pid':p.pid})
        code=p.wait()
    receipt.update(status='complete' if code==0 else 'failed',returncode=code,wall_seconds=time.monotonic()-start)
    save(O/f'E12_JOB_{key}.json',receipt)
    return receipt

def plan():
    jobs=[]
    for s in SEEDS:
        jobs.append((f'A_{s}',['exp8_life.py','A',str(s)],{'E8_RUN_PREFIX':'e12_A_','E8_SAVE_STAGES':'1','E8_SCHEDULE':'interleaved','E8_REPLAY_SHARE':'0'},'A'))
        for batch in (16,256):jobs.append((f'B_b{batch}_{s}',['exp8_life.py','C',str(s)],{'E8_RUN_PREFIX':'e12_B_','E8_BATCH':str(batch)},'B'))
        for share in ('1/2','1/4'):jobs.append((f'C_rp{share.replace("/","of")}_{s}',['exp8_life.py','C',str(s)],{'E8_RUN_PREFIX':'e12_C_','E8_REPLAY_SAMPLING':'recency','E8_REPLAY_SHARE':share},'C'))
        for cap in (128,8192):jobs.append((f'D_cap{cap}_{s}',['exp8_life.py','C',str(s)],{'E8_RUN_PREFIX':'e12_D_','E8_BUFFER_CAP':str(cap)},'D'))
    return jobs

def validate(item):
    key,cmd,env,part=item
    if part=='A' and cmd[0]=='exp8_life.py':
        s=cmd[-1]; new=read(O/f'e12_A_e8_life_A_rp0of1_interleaved_{s}.json');old=read(O/f'e8_life_A_rp0of1_interleaved_{s}.json')
        if stable(new)!=stable(old): raise ValueError(f'A receipt scientific parity mismatch seed{s}')
        return
    if cmd[0]=='exp12_reminder.py':
        s,origin,dose=cmd[1:]; d=read(O/f'e12_reminder_{origin}_r{dose}_{s}.json')
        assert d['updates']==int(dose) and len(d['after'])==8
    else:
        matches=list(O.glob(f'{env["E8_RUN_PREFIX"]}e8_life_C*_{cmd[-1]}.json'))
        # Settings may coexist; every file must be a completed 10-stage life.
        for p in matches:
            d=read(p);assert len(d['stages'])==10 and d['stages'][-1]['task']=='compose'

def main():
    guard=(O/'E12_CAMPAIGN.pidlock').open('w');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read(O/'E12_STATUS.json') if (O/'E12_STATUS.json').exists() else {}
    if old.get('completed'):raise RuntimeError('Existing receipts: inspect/resume explicitly; do not relaunch')
    lock=read(R/'E12_LOCK.json')
    for name,h in lock['files'].items():
        if sha(R/name)!=h:raise RuntimeError(f'Locked input changed: {name}')
    t0=time.monotonic(); completed=[]
    save(O/'E12_STATUS.json',dict(status='profiling',pid=os.getpid(),completed=[]))
    profile=run('profile',['exp8_life.py','C','73129'],{'E8_RUN_PREFIX':'e12_profile_'},'profile');completed.append(profile)
    if profile['returncode'] or stable(read(O/'e12_profile_e8_life_C_73129.json'))!=stable(read(O/'e9_archived_C_73129.json')):
        save(O/'E12_STATUS.json',dict(status='stopped_profile_or_parity_failure',completed=completed));e12_report.main();return
    jobs=plan()
    cpu=read(O/'e12_profile_e8_life_C_73129.json')['stages'][-1]['cpu']
    # Six batch16 lives-equivalents plus remaining lives and 27 independent reminders.
    equivalents=3+6+6+3*(4+.25)+3*3*(64+256+1024)/(8*6144+8192)
    save(R/'E12_PROFILE.json',dict(profile=profile,default_scientific_parity=True,cpu_seconds=cpu,
         new_full_lives=len(jobs),reminders=27,forecast_cpu_seconds=cpu*equivalents,
         forecast_wall_seconds_six_workers=profile['wall_seconds']*equivalents/6,
         caveat='Update-count approximation; large-batch cost, scoring, concurrency and E11 contention vary.'))
    failed=set();waiting=jobs[:]
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending={}
        while waiting or pending:
            while waiting and len(pending)<6:
                item=waiting.pop(0)
                if item[3] not in failed:pending[pool.submit(run,*item)]=item
            if not pending:break
            future=next(as_completed(pending));item=pending.pop(future)
            try:
                receipt=future.result()
                if receipt['status']=='complete':validate(item)
            except Exception as exc:receipt=dict(key=item[0],part=item[3],status='failed',error=str(exc))
            completed.append(receipt)
            if receipt['status']=='failed':
                failed.add(item[3])
                for active in pending.values():
                    if active[3]==item[3]:
                        f=O/f'E12_JOB_{active[0]}.json'
                        if f.exists():
                            d=read(f)
                            if d.get('status')=='running':
                                try:os.kill(d['pid'],15)
                                except ProcessLookupError:pass
            elif item[3]=='A' and item[1][0]=='exp8_life.py':
                s=item[1][-1]
                waiting += [(f'A_rem_{origin}_{dose}_{s}',['exp12_reminder.py',s,origin,str(dose)],{},'A') for origin in ('erased','birth','pre') for dose in (64,256,1024)]
            save(O/'E12_STATUS.json',dict(status='running',pid=os.getpid(),completed=completed,
                 failed_parts=sorted(failed),queued=[i[0] for i in waiting],active=[i[0] for i in pending.values()],elapsed_seconds=time.monotonic()-t0))
            e12_report.main()
    save(O/'E12_STATUS.json',dict(status='complete' if len(completed)==49 and not failed else 'partial_failed',completed=completed,failed_parts=sorted(failed),elapsed_seconds=time.monotonic()-t0))
    e12_report.main()
if __name__=='__main__':main()
