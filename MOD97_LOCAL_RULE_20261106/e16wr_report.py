"""Receipt-only E16-WR scorecard; incomplete jobs cannot produce final verdicts."""
import csv
import statistics as st
from e16wr_common import ROOT,OUT,read,save,sha,verify_lock,validate_receipt,SEEDS

PAIRS=('+-','-+','><','<>')
OPS=('add','sub','max','min')
def pct(v,key='exact'):return 100*v[key]/v['n']
def held(q,key='final'):return st.mean(pct(q['scores']['compose'][p],key) for p in PAIRS)
def ops(q,mod):return st.mean(pct(q['scores'][o+str(mod)]) for o in OPS)
def table(head,rows):
    return '\n'.join(['|'+'|'.join(map(str,head))+'|','|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def csvsave(name,rows):
    if not rows:return
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    lock=verify_lock();state=read(ROOT/'E16WR_STATUS.json')
    rows=[];full=[];exposure=[];completed={};costs=[]
    for j in lock['jobs']:
        jp=OUT/f'E16WR_JOB_{j["key"]}.json'
        if not jp.exists():continue
        job=read(jp)
        if job['status']!='complete':continue
        d=validate_receipt(job);meta=read(OUT/job['metadata'])
        costs.append(dict(key=job['key'],role=job['role'],seed=job['seed'],
                          measured_life_cpu=meta['process_cpu_seconds'],controller_measured_wall=job['wall_seconds'],
                          peak_rss_raw=meta['peak_rss_raw'],peak_rss_units=meta['peak_rss_units']))
        if job['role']=='profile':continue
        completed[(job['capacity'],job['targets'],job['seed'])]=d
        for s in d['stages']:
            q=s['scores']['compose']
            rows.append(dict(capacity=job['capacity'],targets=job['targets'],seed=job['seed'],stage=s['stage'],task=s['task'],
                             held_final=held(s),held_full=held(s,'exact'),old97_mean=ops(s,97),new73_mean=ops(s,73),
                             within_final=pct(q['within'],'final'),within_full=pct(q['within']),
                             **{p+'_final':pct(q[p],'final') for p in PAIRS},**{p+'_full':pct(q[p]) for p in PAIRS}))
            for panel,v in s['scores'].items():
                g=v if panel=='compose' else {panel:v}
                for sub,vv in g.items():
                    for key in ('exact','final','intermediate'):
                        if vv.get(key) is not None:full.append(dict(capacity=job['capacity'],targets=job['targets'],seed=job['seed'],stage=s['stage'],panel=panel,subpanel=sub,metric=key,n=vv['n'],correct=vv[key],percent=pct(vv,key)))
        snapshots=read((OUT/job['receipt']).with_suffix('.reservoir.json'))
        for s in snapshots:
            for category in ('composition','operation'):
                v=s['replay'][category]
                exposure.append(dict(capacity=job['capacity'],targets=job['targets'],seed=job['seed'],stage=s['stage'],
                                     category=category,draws=v['n'],wrong_full=v['wrong_full'],wrong_final=v['wrong_final'],malformed=v['malformed'],
                                     wrong_full_fraction=v['wrong_full']/v['n'] if v['n'] else None,
                                     wrong_final_fraction=v['wrong_final']/v['n'] if v['n'] else None,
                                     per_stratum_draws=s['replay']['draws'],reservoirs=s['reservoirs']))
    baselines={}
    for b in lock['reused_controls']:baselines[(b['role'],b['seed'])]=read(OUT/b['receipt'])
    predictions=[]
    for z in SEEDS:
        base=baselines[('C_uni',z)];base_loss=held(base['stages'][5])-held(base['stages'][-1])
        selfarms=[completed.get((c,'self',z)) for c in (1024,4096,16384)]
        if all(selfarms):
            finals=[held(d['stages'][-1]) for d in selfarms]
            loss=held(selfarms[-1]['stages'][5])-finals[-1]
            predictions.append(dict(seed=z,prediction='P1',passed=finals[0]<finals[1]<finals[2] and loss<=base_loss/2,
                                    value=dict(finals=finals,largest_loss=loss,reused_loss=base_loss)))
        largest=completed.get((16384,'self',z))
        if largest:
            v=ops(largest['stages'][-1],97);predictions.append(dict(seed=z,prediction='P2',passed=v>=88,value=v))
        small=completed.get((1024,'self',z))
        if small:
            a=held(small['stages'][-1])-held(base['stages'][-1]);b=ops(small['stages'][-1],97)-ops(base['stages'][-1],97)
            predictions.append(dict(seed=z,prediction='P3',passed=abs(a)<=5 and abs(b)<=5,value=dict(held_delta=a,old97_delta=b)))
        for (cap,target,zz),d in completed.items():
            if zz==z:
                v=held(d['stages'][5])-held(base['stages'][5])
                predictions.append(dict(seed=z,prediction=f'P4_{cap}_{target}',passed=abs(v)<=3,value=v))
        observed=completed.get((16384,'observed',z))
        if largest and observed:
            a=held(largest['stages'][5])-held(largest['stages'][-1]);b=held(observed['stages'][5])-held(observed['stages'][-1])
            predictions.append(dict(seed=z,prediction='P5',passed=a-b>=10,value=dict(self_loss=a,observed_loss=b,loss_reduction_pp=a-b)))
    csvsave('E16WR_CURVES.csv',rows);csvsave('E16WR_ALL_SCORES.csv',full)
    save(ROOT/'E16WR_EXPOSURES.json',exposure);save(ROOT/'E16WR_PREDICTION_VERDICTS.json',predictions);save(ROOT/'E16WR_COSTS.json',costs)
    final=len(completed)==12 and state['status']=='RUNS_COMPLETE_VERIFICATION_PENDING'
    lines=['# E16-WR — bounded stage reservoirs and replay targets','',
           f'Status: {"RUNS_COMPLETE_FINAL_VERIFICATION_PENDING" if final else "IN_PROGRESS"}; {len(completed)}/12 scientific lives have complete validated receipts. No incomplete life is scored as a final result.','',
           'Four arms: self-generated replay at1,024/4,096/16,384slots per stage and observed-answer replay at16,384. Three seeds each. Nine stage strata,57344updates,sign rule,decay0.25,batch64. Comparators and births are reused, not rerun. One full default-profile reproduction is charged separately.','',
           'The largest pair must match prompt-input hashes and every reservoir/draw snapshot. Observed answers are saved with current presented examples; neither arm invents new observations. The observed treatment changes replay targets throughout life, including operations, so it does not isolate only late composition errors.','',
           table(['Capacity','Targets','Seed','Post-compose held','Final held','Loss pp','Final full','Final old97'],
                 [[c,t,z,f'{held(d["stages"][5]):.4f}',f'{held(d["stages"][-1]):.4f}',
                   f'{held(d["stages"][5])-held(d["stages"][-1]):.4f}',f'{held(d["stages"][-1],"exact"):.4f}',f'{ops(d["stages"][-1],97):.4f}']
                  for (c,t,z),d in sorted(completed.items())]),'',
           table(['Seed','Prediction','Pass','Exact values'],[[x['seed'],x['prediction'],x['passed'],x['value']] for x in predictions]),'',
           'Full per-stage operation/pair/within and full-scratchpad curves: E16WR_CURVES.csv and E16WR_ALL_SCORES.csv. Per-stage actual draw counts, reservoir support and replay target errors: E16WR_EXPOSURES.json. Raw receipts retain every source score.','',
           '## Reused controls','',
           table(['Role','Seed','Final held','Final old97','New run?'],[[a,z,f'{held(d["stages"][-1]):.4f}',f'{ops(d["stages"][-1],97):.4f}','No'] for (a,z),d in sorted(baselines.items())]),'',
           '## Costs and limits','',
           'E16WR_COSTS.json gives measured life CPU from before model construction through final scoring (excluding imports and metadata writing), controller wall and observed peakRSS with platform units. E16WR_STATUS.json gives exact CPU of reaped child processes plus controller CPU; technical cost is separate in E16WR_TECHNICAL_JOB.json. Reused birth/control training costs are historical and are not counted as new.','',
           'This is a synthetic development experiment. Per-stage caps are not a total bound for an unlimited number of stages. AlgorithmR admits duplicate presentations; capacity is not guaranteed distinct coverage. The window and cell studies differ in strata, data and replay targets, preventing a cell-only causal comparison. No new mechanism or autonomous learning is established solely by these scores.']
    (ROOT/'E16WR_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(f'Validated {len(completed)}/12 scientific lives; report updated.')

if __name__=='__main__':main()
