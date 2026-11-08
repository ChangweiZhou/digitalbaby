"""Run and collect the authorized E9 campaign without modifying its scientific settings."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'exp45'
SEEDS = (73129, 93683, 116089)


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in ('cpu', 'seconds', 'wall')}
    if isinstance(value, list):
        return [stable(x) for x in value]
    return value


def run_job(key, command, env, part):
    start = time.monotonic()
    log = OUT / 'logs' / f'e9_{key}.log'
    log.parent.mkdir(exist_ok=True)
    with log.open('w') as stream:
        process = subprocess.Popen([sys.executable, '-u', *command], cwd=ROOT, env={**os.environ, **env},
                                   stdout=stream, stderr=subprocess.STDOUT)
        save(OUT/f'E9_JOB_{key}.json', dict(key=key, part=part, pid=process.pid,
             command=command, env=env, status='running', log=str(log.relative_to(ROOT))))
        code = process.wait()
    receipt = dict(key=key, part=part, command=command, env=env, returncode=code,
                   wall_seconds=time.monotonic()-start, log=str(log.relative_to(ROOT)),
                   status='complete' if code == 0 else 'failed')
    save(OUT/f'E9_JOB_{key}.json', receipt)
    return receipt


def report(status='running'):
    lines = ['# E9 price of persistence', '', f'Campaign status **{status}**.', '',
             'Source and birth hashes are recorded in E9_LOCK.json. Archived baselines are reused only after '
             'default scientific parity is checked. CPU fields vary by execution.', '',
             '| Schedule | Replay | Seed | Mean eight rules before composition | After composition | Held composition mean |',
             '| --- | --- | --- | ---: | ---: | ---: |']
    detail = []
    for schedule, share, tag, arm in (
        ('blocked','1/8','_rp1of8_blocked','C'), ('blocked','1/4','_rp1of4_blocked','C'),
        ('blocked','1/2','','C'), ('blocked','0','','A'),
        ('interleaved','1/2','_rp1of2_interleaved','C'), ('interleaved','0','_rp0of1_interleaved','A')):
        for seed in SEEDS:
            file = OUT/f'e8_life_{arm}{tag}_{seed}.json'
            if not file.exists():
                lines.append(f'| {schedule} | {share} | {seed} | Pending | Pending | Pending |')
                continue
            try:
                row = read(file)
            except json.JSONDecodeError:
                continue
            stages = row['stages']
            final = next((s for s in stages if s['task']=='compose'),None)
            pre = next((s for s in stages if s['task']==('min73' if schedule=='blocked' else 'interleaved')),None)
            names = ('add97','sub97','max97','min97','add73','sub73','max73','min73')
            pct = lambda s,k: 100*s['scores'][k]['exact']/s['scores'][k]['n']
            mean = lambda s: sum(pct(s,k) for k in names)/8
            comp = (sum(100*final['scores']['compose'][k]['final']/final['scores']['compose'][k]['n']
                        for k in ('+-','-+','><','<>'))/4 if final else None)
            show = lambda x: f'{x:.2f}%' if x is not None else 'Pending'
            lines.append(f'| {schedule} | {share} | {seed} | {show(mean(pre) if pre else None)} | '
                         f'{show(mean(final) if final else None)} | {show(comp)} |')
            if pre or final:
                detail.extend(['', f'## {schedule}, replay {share}, seed {seed}', '',
                               '| Operation | Before composition | After composition |',
                               '| --- | ---: | ---: |'])
                for k in names:
                    detail.append(f'| {k} | {show(pct(pre,k) if pre else None)} | {show(pct(final,k) if final else None)} |')
                if final:
                    for k,v in final['scores']['compose'].items():
                        detail.append(f'\nComposition {k} · final {v["final"]}/{v["n"]}, full output {v["exact"]}/{v["n"]}.')
    lines.extend(['', '## Part C', '',
                  '| Seed | Held pair means at endpoint | Within | Final minus previous held mean |',
                  '| --- | ---: | ---: | ---: |'])
    for seed in SEEDS:
        p = OUT/f'e7_life_SCR_h512_d16384_{seed}.json'
        if not p.exists():
            lines.append(f'| {seed} | Pending | Pending | Pending |')
            continue
        curves = read(p)['curve']
        def avg(c):
            return sum(100*c['compose'][k]['final']/c['compose'][k]['n'] for k in ('+-','-+','><','<>'))/4
        last = curves[-1]['compose']
        within = 100*last['within']['final']/last['within']['n']
        lines.append(f'| {seed} | {avg(curves[-1]):.2f}% | {within:.2f}% | {avg(curves[-1])-avg(curves[-2]):+.2f} points |')
        detail.extend(['',f'## Part C, seed {seed}', ''])
        detail.extend(f'- {k} · final {v["final"]}/{v["n"]}, full output {v["exact"]}/{v["n"]}.'
                      for k,v in last.items())
    lines.extend(['', '## Prediction checks', ''])
    def load_e8(arm, seed, tag=''):
        p = OUT/f'e8_life_{arm}{tag}_{seed}.json'
        if not p.exists():
            return None
        try:
            stages = read(p)['stages']
        except json.JSONDecodeError:
            return None
        if stages[-1]['task'] != 'compose':
            return None
        return stages[-2]['scores'], stages[-1]['scores']
    names = ('add97','sub97','max97','min97','add73','sub73','max73','min73')
    carry = ('add97','sub97','add73','sub73')
    def average(scores, keys):
        return sum(100*scores[k]['exact']/scores[k]['n'] for k in keys)/len(keys)
    for seed in SEEDS:
        base = load_e8('C', seed)
        quarter = load_e8('C', seed, '_rp1of4_blocked')
        eighth = load_e8('C', seed, '_rp1of8_blocked')
        inter = load_e8('A', seed, '_rp0of1_interleaved')
        if base and quarter:
            delta = average(quarter[0], names) - average(base[0], names)
            erosion_q = average(quarter[0], carry) - average(quarter[1], carry)
            erosion_b = average(base[0], carry) - average(base[1], carry)
            lines.append(f'- A quarter, seed {seed} · stage-8 mean delta {delta:+.2f} points, '
                         f'carry erosion {erosion_q:.2f} vs {erosion_b:.2f} points, '
                         f'prediction {abs(delta) <= 5 and erosion_q < erosion_b}.')
        if base and eighth:
            drop = average(base[0], names[:4]) - average(eighth[0], names[:4])
            lines.append(f'- A eighth, seed {seed} · earliest-four mean drop {drop:.2f} points, prediction {drop > 15}.')
        if base and inter:
            differences = [average(inter[0], (k,)) - average(base[0], (k,)) for k in names]
            erosion = average(inter[0], carry) - average(inter[1], carry)
            lines.append(f'- B no replay, seed {seed} · weakest per-operation delta {min(differences):+.2f}, '
                         f'carry erosion {erosion:.2f} points, prediction {min(differences) >= 0 and erosion < 10}.')
    lines.extend(['', '## Archived backprop ceiling', ''])
    for seed in SEEDS:
        ref = load_e8('D', seed)
        if ref:
            lines.append(f'- Seed {seed} · mean operations at stage 8 {average(ref[0], names):.2f}%, '
                         f'after composition {average(ref[1], names):.2f}%. Receipt exp45/e8_life_D_{seed}.json.')
    (ROOT/'E9_REPORT.md').write_text('\n'.join(lines+detail)+'\n')


def main():
    lock = read(ROOT/'E9_LOCK.json')
    for name,h in lock['files'].items():
        if sha256((ROOT/name).read_bytes()).hexdigest()!=h:
            raise ValueError(f'Source or birth changed: {name}')
    save(OUT/'E9_STATUS.json', dict(status='profiling', pid=os.getpid(), jobs=[]))
    # Profile one full life before launching any sweep jobs.
    baseline = read(OUT/'e9_archived_C_73129.json')
    profile = run_job('profile_C_73129', ['exp8_life.py','C','73129'], {}, 'profile')
    if profile['returncode'] or stable(read(OUT/'e8_life_C_73129.json')) != stable(baseline):
        save(OUT/'E9_STATUS.json', dict(status='stopped_profile_or_parity_failure', profile=profile))
        report('stopped_profile_or_parity_failure')
        return
    cpu = read(OUT/'e8_life_C_73129.json')['stages'][-1]['cpu']
    jobs=[]
    for seed in SEEDS:
        for share in ('1/8','1/4'):
            jobs.append((f'A_{share.replace("/","of")}_{seed}', ['exp8_life.py','C',str(seed)],
                         {'E8_REPLAY_SHARE':share}, 'A'))
        jobs.extend([
            (f'B_C_{seed}', ['exp8_life.py','C',str(seed)],
             {'E8_SCHEDULE':'interleaved','E8_REPLAY_SHARE':'1/2'}, 'B'),
            (f'B_A_{seed}', ['exp8_life.py','A',str(seed)],
             {'E8_SCHEDULE':'interleaved','E8_REPLAY_SHARE':'0'}, 'B'),
            (f'C_{seed}', ['exp7_decon.py','SCR',str(seed)],
             {'EXP45_HIDDEN':'512','E7_DOSE':'16384'}, 'C')])
    save(ROOT/'E9_PROFILE.json', dict(profile=profile, cpu_seconds=cpu,
         new_jobs=len(jobs), forecast_cpu_seconds=cpu*len(jobs),
         forecast_wall_seconds_at_six_workers=profile['wall_seconds']*len(jobs)/6,
         default_scientific_receipts_equal=True))
    save(OUT/'E9_STATUS.json', dict(status='running', pid=os.getpid(), planned_jobs=len(jobs), completed=[]))
    report()
    failed_parts=set()
    completed=[]
    # At most one worker per part/seed is outstanding in a bounded queue.
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending={}
        waiting=list(jobs)
        while waiting or pending:
            while waiting and len(pending)<6:
                item=waiting.pop(0)
                if item[3] not in failed_parts:
                    pending[pool.submit(run_job,*item)]=item
            if not pending: break
            future=next(as_completed(pending))
            item=pending.pop(future)
            try:
                record=future.result()
            except Exception as exc:
                record=dict(key=item[0],part=item[3],status='failed',error=str(exc))
            completed.append(record)
            if record['status']=='failed':
                failed_parts.add(item[3])
                for other in pending.values():
                    if other[3] == item[3]:
                        job = OUT/f'E9_JOB_{other[0]}.json'
                        if job.exists():
                            active = read(job)
                            if active.get('status') == 'running':
                                try:
                                    os.kill(active['pid'], 15)
                                except ProcessLookupError:
                                    pass
            save(OUT/'E9_STATUS.json',dict(status='running',pid=os.getpid(),
                 completed=completed,failed_parts=sorted(failed_parts),
                 queued=[x[0] for x in waiting],active=[x[0] for x in pending.values()]))
            report()
    status='complete' if len(completed)==len(jobs) and not failed_parts else 'partial_failed'
    save(OUT/'E9_STATUS.json',dict(status=status,completed=completed,failed_parts=sorted(failed_parts)))
    report(status)


if __name__=='__main__':
    main()
