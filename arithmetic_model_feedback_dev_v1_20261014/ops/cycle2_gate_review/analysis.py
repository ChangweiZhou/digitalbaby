"""Saved-output reduction and fixed admission/confirmation rules. No training."""
import collections,gzip,json,math,statistics
from pathlib import Path
from protocol import ROOT,D,atomic
def rates(rows):
    by=collections.defaultdict(list)
    for x in rows:by[x['operation']].append(float(x['exact']))
    return {k:sum(v)/len(v) for k,v in by.items()}
def macro(values):return sum(values.values())/len(values) if values else 0.
def ability(c):return .5*macro(rates(c['scores']['binary']))+.5*macro(rates(c['scores']['mixed']))
def staircase(points,end):
    x0=a0=area=0.
    for x,a in points:
        assert x>=x0 and 0<=a<=1
        cut=min(x,end);area+=(cut-x0)*a0
        if x>=end:return area/end
        x0,a0=x,a
    return (area+(end-x0)*a0)/end
def shortcut(entry):
    out={}
    for label,rows in entry['scores'].items():
        by=collections.defaultdict(lambda:collections.defaultdict(list))
        for row in rows:
            for key,pred in row['shortcuts'].items():by[row['operation']][key].append(float(pred==row['reply']))
        out[label]={op:max(sum(v)/len(v) for v in models.values()) for op,models in by.items()}
    return out
def metrics(r,cstar):
    out={};end=r['binary']+r['mixed']+sum(r['stages'])
    for name,b in r['branches'].items():
        curve=b['curve'];last=curve[-1]
        out[name]=dict(J_sample=staircase([(x['n'],ability(x)) for x in curve],end),
            J_compute=staircase([(x['cpu'],ability(x)) for x in curve],cstar),A=ability(last),
            binary=rates(last['scores']['binary']),mixed=rates(last['scores']['mixed']),shortcut=shortcut(last),
            CPU=last['cpu'],answers=b['resource']['observed_answer_bytes'])
    out['warm']=dict(binary=rates(r['prefixes']['TRUE']['taught']['scores']['binary']),shortcut=shortcut(r['prefixes']['TRUE']['taught']))
    return out
def load(folder):
    rows=[]
    for p in sorted((Path(folder)/'receipts').glob('*.json.gz')):
        cert=json.loads(p.with_suffix('.audit.json').read_text());assert cert['accepted']
        with gzip.open(p,'rt') as f:r=json.load(f)
        assert r['completed'];rows.append(r)
    return rows
def effects(by):
    result=collections.defaultdict(list)
    for world,models in sorted(by.items()):
        k,b=models['KEEP'],models['R0'];f=k['FADING']
        for key in ('J_sample','J_compute'):result[key].append(f[key]-b['FADING'][key])
        result['scalar_increment'].append(f['A']-k['FREEZE_SCALAR']['A'])
        for op in ('+','-','*','/'):
            result['warm_input_'+op].append(k['warm']['binary'][op]-k['warm']['shortcut']['binary'][op])
            for control in ('NO_WARM','SH_WARM'):
                result['binary_'+op+'_'+control].append(f['binary'][op]-k[control]['binary'][op])
            result['binary_'+op+'_shortcut'].append(f['binary'][op]-f['shortcut']['binary'][op])
        for control in ('NO_WARM','SH_WARM'):
            result['mixed_'+control].append(macro(f['mixed'])-macro(k[control]['mixed']))
        result['mixed_shortcut'].append(macro(f['mixed'])-macro(f['shortcut']['mixed']))
    return dict(result)
def reduce_stage(folder,cstar=None):
    rows=load(folder);assert rows
    if cstar is None:cstar=statistics.median(r['branches']['FADING']['curve'][-1]['cpu'] for r in rows if r['model']=='R0')
    by={}
    for r in rows:by.setdefault(r['world'],{})[r['model']]=metrics(r,cstar)
    assert all(set(v)=={'R0','KEEP'} for v in by.values())
    es=effects(by);means={k:statistics.mean(v) for k,v in es.items()}
    gate=all(v>0 for v in means.values())
    return dict(worlds=list(by),cstar=cstar,metrics=by,effects=es,effect_means=means,efficacy_admission=gate,
        committed_jobs=len(rows),executed_records=sum(r['executed_records'] for r in rows),correlated_histories=5*len(rows),
        cost=dict(max_job_wall=max(r['wall'] for r in rows),max_job_cpu=max(r['total_cpu'] for r in rows)),
        scope='Technical effect means are descriptive admission evidence, not significance or adoption.')
def confirm(result):
    from scipy.stats import t
    intervals={}
    # Warm acquisition is an admission sanity condition, not a substitute for
    # either primary efficiency outcome in fresh confirmation.
    for k,v in result['effects'].items():
        n=len(v);m=statistics.mean(v);se=statistics.stdev(v)/math.sqrt(n)
        lower=m-float(t.ppf(.95,n-1))*se
        intervals[k]=dict(mean=m,lower_one_sided95=lower,worlds=n)
    passed=all(x['lower_one_sided95']>0 for k,x in intervals.items() if not k.startswith('warm_input'))
    return dict(verdict='ADOPT_BOUNDED_MODEL_FEEDBACK_CORE' if passed else 'NO_CONFIRMED_REPLACEMENT',
                intervals=intervals,all_required=passed,method='world-paired one-sided Student intersection-union; one fixed candidate')
def write_report(path,result,verdict):
    lines=['# 模型与反馈联合实验结果','',f'结论：**{verdict}**。',
        f"完整作业 {result['committed_jobs']}；执行外部记录 {result['executed_records']}；相关末态历史 {result['correlated_histories']}。",'',
        '整串自身生成与EOS评分，四操作和混合题保持原权重。预热和全部后续题目计入样本面积，抽样、critic和查询计入计算。',
        '', '| 固定效应 | 配对世界均值 |','|---|---:|']
    lines += [f'| {k} | {v:.6f} |' for k,v in result['effect_means'].items()]
    lines+=['','DENSE为信息参照；FREEZE_SCALAR从相同的过渡期末态分叉。未见题没有接受反馈。',
        '技术世界不能提供显著性或采纳结论；启动失败不会追加dose、降低门槛或替换主要任务。']
    Path(path).write_text('\n'.join(lines)+'\n')

def finalize():
    import hashlib,zipfile,time
    from protocol import identity
    out=ROOT/'results';manifest=json.loads((out/'MANIFEST.json').read_text());admit=json.loads((ROOT/'SCIENCE_ADMISSION.json').read_text())
    rows=load(out)
    actual={(r['world'],r['model']) for r in rows};wanted={(x['world'],x['model']) for x in manifest['jobs']}
    assert actual==wanted and len(rows)==64 and len(actual)==64
    assert all(r['source']==admit['source']==manifest['source']==identity() for r in rows)
    result=reduce_stage(out,admit['cstar']);verdict=confirm(result)
    atomic(out/'SUMMARY.json',dict(result,confirmation=verdict))
    write_report(out/'REPORT.md',result,verdict['verdict'])
    audit=dict(accepted=True,source=identity(),jobs=len(rows),records=result['executed_records'],histories=result['correlated_histories'],
        fresh_worlds=32,all_certificates_accepted=True,all_required_effects=verdict['all_required'],
        scope='unique locked roster, original independent certificates, source identity and saved-output confirmation reduction; no training replay')
    atomic(out/'FINAL_AUDIT.json',audit)
    atomic(out/'STATUS.json',dict(status='AUDIT_ACCEPTED_PACKAGING',heartbeat=time.time(),verdict=verdict['verdict'],committed_jobs=len(rows),committed_records=result['executed_records']))
    package=ROOT/'RESULT_BUNDLE.zip';temp=package.with_suffix('.pending')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in ROOT.rglob('*'):
            if not p.is_file() or p in (temp,package) or '__pycache__' in p.parts or p.name=='PACKAGE.json':continue
            z.write(p,arcname=str(p.relative_to(ROOT.parent)))
        for p in [ROOT.parent/'arithmetic_source_derived_kernel_dev_v1_20261013/core.py',ROOT.parent/'arithmetic_source_derived_kernel_dev_v1_20261013/auditor.py',ROOT.parent/'arithmetic_end_to_end_design_v1_20261013/validate_design.py']:
            z.write(p,arcname=str(p.relative_to(ROOT.parent)))
    temp.replace(package)
    atomic(out/'PACKAGE.json',dict(accepted=True,path=str(package),bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest(),science_rerun=False))
    atomic(out/'STATUS.json',dict(status='COMPLETED',heartbeat=time.time(),verdict=verdict['verdict'],committed_jobs=len(rows),committed_records=result['executed_records'],package=str(package)))
    print(json.dumps(dict(verdict=verdict['verdict'],jobs=len(rows),records=result['executed_records'])),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args()
    if a.final:finalize()
