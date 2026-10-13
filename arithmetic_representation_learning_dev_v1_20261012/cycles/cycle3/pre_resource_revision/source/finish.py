"""Audit, reduce and package only a complete, source-matched roster."""
import gzip
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from core import CONFIGS
from auditor import audit
from analysis import reduce
from operations import atomic_json, manifest
from runner import source_lock


def finish(receipt_directory,output,technical=False,expected_worlds=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    groups={};certs=[];paths=[]
    atomic_json(output/'ANALYSIS_HEARTBEAT.json',{'stage':'audit','epoch':time.time()})
    for path in sorted(Path(receipt_directory).glob('*.json.gz')):
        with gzip.open(path,'rt') as f:r=json.load(f)
        if r['technical']!=technical:raise ValueError('science/technical contamination')
        w,c=r['world'],r['config']
        if c in groups.setdefault(w,{}):raise ValueError('duplicate world/config')
        certs.append(audit(r,check_sources=True));groups[w][c]=r;paths.append(path)
    if expected_worlds is None:expected_worlds=sorted({x['world'] for x in manifest()})
    if sorted(groups)!=sorted(expected_worlds) or any(set(x)!=set(CONFIGS) for x in groups.values()):
        raise ValueError('incomplete roster; partial results cannot be promoted')
    if any(len({r['plan']['stream_digest'] for r in x.values()})!=1 for x in groups.values()):
        raise ValueError('unpaired streams')
    if any(len(r['plan']['records'])!=512 for x in groups.values() for r in x.values()):
        raise ValueError('incomplete full budget')
    atomic_json(output/'ANALYSIS_HEARTBEAT.json',{'stage':'reduce','epoch':time.time()})
    summary=reduce([groups[w] for w in sorted(groups)],technical)
    counts={'committed_jobs':len(certs),'correlated_histories':3*len(certs),
            'record_applications':sum(c['records_applied'] for c in certs),
            'scientific_jobs':sum(c['science_jobs'] for c in certs)}
    summary['counts']=counts
    atomic_json(output/'SUMMARY.json',summary)
    atomic_json(output/'FINAL_AUDIT.json',{'accepted':True,'technical':technical,
        'counts':counts,'source_lock':source_lock(),'certificates':certs,
        'scientific_ability_claim':False if technical else any(
            any(p['final_E3_operations'].values()) for p in summary['candidate_profiles'].values())})
    lines=['# ADAPT-2 — '+('技术管线结果' if technical else '科学结果'),'',
        f"判定：{summary['verdict']}；选择：{summary['selected']}。",'',
        f"完成 {len(certs)} 任务、{counts['correlated_histories']} 条相关历史、{counts['record_applications']} 次记录应用。",'',
        '技术夹具不支持科学 E3，也不参与正式世界统计。' if technical else
        '按事实家族留出；评分是模型完整生成的回答及EOS。机制、能力和替换建议分开。','',
        '| 配置 | 4/8/16/32/64/128样本未见题宏曲线 | W训练CPU中位秒 |',
        '| --- | --- | ---: |']
    for c,d in summary['descriptive_results'].items():
        lines.append(f"| {c} | "+' / '.join(f'{v:.2%}' for v in d['W_macro_curve'])+
                     f" | {np.median(d['W_training_cpu_by_world']):.3f} |")
    lines += ['', '| 配置 | 运算 | 最终 W | N_ALL | SHUFFLE |','| --- | --- | ---: | ---: | ---: |']
    for c,d in summary['descriptive_results'].items():
        for op,o in d['per_operation'].items():
            h=o['final_by_history'];lines.append(f"| {c} | {op} | {h['W']:.2%} | {h['N_ALL']:.2%} | {h['SHUFFLE']:.2%} |")
    lines += ['', '全部41项配对推断、分运算曲线、同运算干扰、真实状态成本和不可判项见 SUMMARY.json。',
              '静置仅推进模拟时钟；没有生物巩固、混合表达式或自主无教学信号学习结论。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic_json(output/'ANALYSIS_HEARTBEAT.json',{'stage':'package','epoch':time.time()})
    bundle=output/'RESULT_BUNDLE.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for path in paths:z.write(path,'receipts/'+path.name)
        for name in ('REPORT.md','SUMMARY.json','FINAL_AUDIT.json'):z.write(output/name,name)
        root=Path(__file__).parent
        for rel in source_lock():z.write(root/rel,'locked_source/'+rel)
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None:raise ValueError('package corruption')
    atomic_json(output/'PACKAGE.json',{'complete':True,'technical':technical,
        'zip_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'counts':counts})
    atomic_json(output/'ANALYSIS_HEARTBEAT.json',{'stage':'complete','epoch':time.time()})
    return summary
