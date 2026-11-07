import json
from pathlib import Path
R=Path(__file__).resolve().parent; O=R/'exp45'; SEEDS=(73129,93683,116089)
NAMES=('add97','sub97','max97','min97','add73','sub73','max73','min73')
def read(p): return json.loads(p.read_text())
def pct(sc,k): return 100*sc[k]['exact']/sc[k]['n']
def avg(sc,keys=NAMES): return sum(pct(sc,k) for k in keys)/len(keys)
def metrics(p):
    d=read(p); pre=d['stages'][-2]['scores']; end=d['stages'][-1]['scores']
    if d['stages'][-1]['task']!='compose': raise ValueError('Incomplete life')
    comp=sum(100*end['compose'][k]['final']/end['compose'][k]['n'] for k in ('+-','-+','><','<>'))/4
    return pre,end,comp,d['stages'][-1]['cpu']
def main():
    status=read(O/'E12_STATUS.json') if (O/'E12_STATUS.json').exists() else {'status':'prepared'}
    lines=['# E12 replay mechanics','',f"Campaign status: **{status['status']}**.",'','Source/authorization/hashes: E12_LOCK.json. Predictions were written before gradients. Existing E8/E9 controls are reused, not new replications. Own-byte/EOS endpoints unchanged.','', '## A: reminder savings','', '| Seed | Updates | Erased | Birth | Pre-composition | Savings |','|---|---:|---:|---:|---:|---:|']
    verdict=[]; details=[]
    for s in SEEDS:
        savings=[]
        for dose in (64,256,1024):
            files=[O/f'e12_reminder_{a}_r{dose}_{s}.json' for a in ('erased','birth','pre')]
            if all(p.exists() for p in files):
                rows=[read(p) for p in files]; v=[avg(x['after']) for x in rows]
                lines.append(f'|{s}|{dose}|{v[0]:.2f}|{v[1]:.2f}|{v[2]:.2f}|{v[0]-v[1]:+.2f}|'); savings.append(v[0]-v[1])
                if dose==256: verdict.append(f'A seed{s}: recovery>=60 and birth<=20: {v[0]>=60 and v[1]<=20}.')
                details += [f'\n### Reminder seed{s}, updates{dose}',str({a:{k:round(pct(row['after'],k),2) for k in NAMES} for a,row in zip(('erased','birth','pre'),rows)})]
            else: lines.append(f'|{s}|{dose}|Pending|Pending|Pending|Pending|')
        if len(savings)==3: verdict.append(f'A seed{s}: within10 at all doses: {all(abs(x)<=10 for x in savings)} (not a proof of complete information loss).')
    lines += ['','## B/C/D: full lives','','|Part|Setting|Seed|Stage8 mean|Final mean|Held composition|CPU seconds|Origin|','|---|---|---:|---:|---:|---:|---:|---|']
    for s in SEEDS:
        paths=[('B','batch16',f'e12_B_e8_life_C_b16_{s}.json','new'),('B','batch64',f'e8_life_C_{s}.json','reused'),('B','batch256',f'e12_B_e8_life_C_b256_{s}.json','new'),('C','recency half',f'e12_C_e8_life_C_recency_{s}.json','new'),('C','recency quarter',f'e12_C_e8_life_C_rp1of4_blocked_recency_{s}.json','new'),('C','uniform quarter',f'e8_life_C_rp1of4_blocked_{s}.json','reused'),('D','cap128',f'e12_D_e8_life_C_cap128_{s}.json','new'),('D','cap1024',f'e8_life_C_{s}.json','reused'),('D','cap8192',f'e12_D_e8_life_C_cap8192_{s}.json','new')]
        baseline=metrics(O/f'e8_life_C_{s}.json')
        for part,setting,name,origin in paths:
            p=O/name
            try: pre,end,c,cpu=metrics(p)
            except (FileNotFoundError,ValueError,json.JSONDecodeError):
                lines.append(f'|{part}|{setting}|{s}|Pending|Pending|Pending|Pending|{origin}|');continue
            lines.append(f'|{part}|{setting}|{s}|{avg(pre):.2f}|{avg(end):.2f}|{c:.2f}|{cpu:.2f}|{origin}|')
            details += [f'\n### {part} {setting} seed{s}', '|Operation|Stage8|Final|','|---|---:|---:|']+[f'|{k}|{pct(pre,k):.2f}|{pct(end,k):.2f}|' for k in pre]
            details += [str(end['compose'])]
            if setting=='batch16': verdict.append(f'B seed{s}: retention within3: {abs(avg(pre)-avg(baseline[0]))<=3}; composition delta {c-baseline[2]:+.2f}.')
            if setting=='batch256': verdict.append(f'B seed{s}: carry loss>10: {avg(baseline[0],NAMES[0:2]+NAMES[4:6])-avg(pre,NAMES[0:2]+NAMES[4:6])>10}.')
            if part=='C' and setting.startswith('recency'):
                ref=baseline if setting.endswith('half') else metrics(O/f'e8_life_C_rp1of4_blocked_{s}.json')
                early=avg(pre,NAMES[:4])-avg(ref[0],NAMES[:4]);late=avg(pre,NAMES[4:])-avg(ref[0],NAMES[4:])
                verdict.append(f'C {setting} seed{s}: early delta{early:+.2f}, late delta{late:+.2f}; predicted {early < -10 and late<5}; all operation deltas within5: {all(abs(pct(pre,k)-pct(ref[0],k))<=5 for k in NAMES)}.')
            if part=='D': verdict.append(f'D {setting} seed{s}: stage8 delta {avg(pre)-avg(baseline[0]):+.2f}, final delta {avg(end)-avg(baseline[1]):+.2f}.')
    lines += ['','## Prediction verdicts','']+verdict
    jobs=status.get('completed',[]); lines+=['','## Actual compute','',f"Completed subprocesses: {len(jobs)}. Summed wall seconds: {sum(x.get('wall_seconds',0) for x in jobs):.2f}. Profile forecast in E12_PROFILE.json is not measured total cost.", '','## Limits','','Savings does not localize residual information to a specific layer. Batch changes update count and cumulative weight decay. Recency tests a nonstationary task sequence. Buffer coverage differs from exposure repetition. No tuning or promotion is inferred. Path-integration follow-up remains NOT_RUN; E11 is untouched.']
    (R/'E12_REPORT.md').write_text('\n'.join(lines+details)+'\n')
if __name__=='__main__':main()
