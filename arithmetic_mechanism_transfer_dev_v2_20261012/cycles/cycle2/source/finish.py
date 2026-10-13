"""Immutable-roster analysis/audit/package, or explicitly synthetic pipeline test."""
import gzip
import hashlib
import json
from pathlib import Path
import zipfile
from core import CONFIGS
from analysis import reduce
from auditor import audit
from operations import atomic_json, manifest
from runner import source_lock


def finish(receipt_directory,output,technical=False,expected_worlds=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    groups={};certs=[];paths=[]
    for path in sorted(Path(receipt_directory).glob('*.json.gz')):
        with gzip.open(path,'rt') as f:r=json.load(f)
        if r['technical']!=technical:
            raise ValueError('science/technical isolation')
        world,arm=r['world'],r['config']
        if arm in groups.setdefault(world,{}):
            raise ValueError('duplicate world/config')
        groups[world][arm]=r;paths.append(path)
        certs.append(audit(r,check_sources=True))
    if expected_worlds is None:
        expected_worlds=sorted({row['world'] for row in manifest()})
    if sorted(groups)!=sorted(expected_worlds) or any(set(v)!=set(CONFIGS) for v in groups.values()):
        raise ValueError('incomplete frozen roster; cannot analyze partial completion')
    for world,items in groups.items():
        if len({r['plan']['stream_digest'] for r in items.values()})!=1:
            raise ValueError('unpaired stream')
        if any(len(r['plan']['records'])!=512 for r in items.values()):
            raise ValueError('incomplete per-history budget')
    summary=reduce([groups[w] for w in sorted(groups)],technical=technical)
    counts={'committed_jobs':len(certs),'correlated_histories':3*len(certs),
            'record_applications':sum(r['records_applied'] for r in certs),
            'scientific_jobs':sum(r['science_jobs'] for r in certs)}
    summary['counts']=counts
    atomic_json(output/'SUMMARY.json',summary)
    atomic_json(output/'FINAL_AUDIT.json',{'accepted':True,'technical':technical,
        'counts':counts,'source_lock':source_lock(),'certificates':certs,
        'ability_demonstrated':False if technical else summary['selected'] is not None})
    report=('# 四则机制实验 — '+('技术管线验证' if technical else '科学结果')+'\n\n'+
        f"判定：{summary['verdict']}。\n\n"+
        f"完成{len(certs)}个任务，{counts['correlated_histories']}条相关历史，"
        f"{counts['record_applications']}次记录应用。统计单位是配对世界。\n\n"+
        ('技术世界只证明程序可运行；不支持E3或新核心采纳。\n' if technical else
         '未见题按事实家族留出，完整生成答案及EOS；学习因果、样本效率、计算效率与保持分别报告。\n')+
        '\n全部116项预声明比较、零方差未决和空CPU区间见SUMMARY.json。\n')
    (output/'REPORT.md').write_text(report)
    bundle=output/'RESULT_BUNDLE.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for path in paths:z.write(path,'receipts/'+path.name)
        for name in ('REPORT.md','SUMMARY.json','FINAL_AUDIT.json'):z.write(output/name,name)
        for path in Path(__file__).parent.glob('*.py'):z.write(path,'locked_sources/'+path.name)
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None:raise ValueError('package corruption')
    atomic_json(output/'PACKAGE.json',{'complete':True,'technical':technical,
        'zip_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'counts':counts})
    return summary
