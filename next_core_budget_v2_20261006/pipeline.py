"""Explicitly authorized screen -> immutable one-winner confirmation -> package."""
import argparse,time,json,zipfile,os,signal,subprocess
ORIGIN=time.monotonic()
from pathlib import Path
from runtime import ROOT,FROZEN,load_plan,read_receipt,atomic_json,atomic_bytes,write_receipt,execution_identity,file_sha,source_files,digest
from locks import require_authorized,manifest,seal_selection,verify_selection_lock
from dispatcher import dispatch,decision,disk_bytes
from metrics import collect
from selection import choose
from confirmation import confirm
from protocol import screen_worlds,confirmation_worlds,roster,job_key
from job_audit import audit,audit_unavailable

class DeadlinePause(Exception):pass
def deadline():
    if time.monotonic()-ORIGIN>=load_plan()['session']['safe_pause_request_seconds']:raise DeadlinePause('analysis/package session limit')

def render_report(stage,rows,out,evidence):
    lines=[f'# MiniFly budget v2 {stage}', '', '固定字节接口与既有 Full151 机制；不扩展任务、样本或候选。', '', '| 配置 | n | old E1 | new E1 | revision E1 | reuse W | reuse W−N | W online CPU s |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,rs in rows.items():
        n=len(rs)
        mean=lambda k:sum(x[k] for x in rs)/n
        lines.append(f"| {arm} | {n} | {mean('old'):.4f} | {mean('new'):.4f} | {mean('revision'):.4f} | {mean('reuse_W'):.4f} | {mean('reuse_W')-mean('reuse_N'):.4f} | {mean('tauL')+mean('tauR'):.4f} |")
    lines+=['',f"裁决：`{out['verdict']}`。",'', 'E1 是教过键的回忆/保留；E3 是未强化关系的有限复用。精确键查表可解决 E1，故 E1 单独通过不证明 E3、推理或自主学习。所有测试仍采用提示输出与到来的答案字节教学。筛选结果是探索性的；确认只检验预先锁定的一组候选和目标。', '', f'完整收据/不可用标记：{len(evidence)}；所有配置均保留。', '', '```json',json.dumps(out,ensure_ascii=False,indent=2),'```','']
    return '\n'.join(lines)

def final_audit(selection,summary=None):
    deadline();verify_selection_lock(selection);p=load_plan();ident=execution_identity();counts={};evidence=[]
    progress_file=ROOT/'results/FINAL_AUDIT_PROGRESS.json'
    progress=json.loads(progress_file.read_text()) if progress_file.exists() else {'execution_source_identity':ident,'validated':{}}
    if progress['execution_source_identity']!=ident:raise ValueError('audit progress source changed')
    for stage in ('screen','confirm'):
        if stage=='confirm' and not selection['winner']:continue
        m=manifest(stage,selection=selection if stage=='confirm' else None);folder=ROOT/'science'/stage
        actual=set(x.stem.replace('.json','') for x in (folder/'receipts').glob('*.json.gz'))|set(x.stem.replace('.json','') for x in (folder/'unavailable').glob('*.json.gz'))
        expected={job_key(j) for j in m['jobs']}
        if actual!=expected:raise ValueError('final receipt/manifest census')
        complete=unavailable=lives=0
        for j in m['jobs']:
            k=job_key(j);f=folder/'receipts'/f'{k}.json.gz'
            if f.exists():
                deadline();d=read_receipt(f);model=folder/'models'/d['W_export']['file'];token=digest([file_sha(f),file_sha(model)])
                if progress['validated'].get(k)!=token:
                    audit(d,ident,folder,load_export=True);progress['validated'][k]=token;atomic_json(progress_file,progress)
                complete+=1;lives+=p['assays'][j['assay']]['lives_per_job']
            else:f=folder/'unavailable'/f'{k}.json.gz';audit_unavailable(read_receipt(f),ident,folder);unavailable+=1
            evidence.append({'file':str(f.relative_to(ROOT)),'sha256':file_sha(f)})
        counts[stage]={'complete_jobs':complete,'diagnostic_unavailable':unavailable,'complete_lives':lives,'planned_jobs':len(m['jobs'])}
    if selection['winner']:
        rows,files=collect(ROOT/'science/confirm',confirmation_worlds(p),selection['confirmation_roster'],'confirm')
        rebuilt=confirm(rows,p,selection,confirmation_worlds(p))
        if rebuilt!=summary:raise ValueError('confirmation independent reduction mismatch')
    out={'verdict':'ACCEPTED_COMPLETED_EXPERIMENT','scientific_source_identity':__import__('runtime').SCIENTIFIC_ID,'execution_source_identity':ident,'counts':counts,'receipt_evidence':evidence,'selection_sha256':file_sha(ROOT/'SELECTION_LOCK.json.gz'),'science_extended':False,'current_run_only':True,'GitHub_upload_authorized':False}
    atomic_json(ROOT/'results/FINAL_AUDIT.json',out);return out

def package():
    deadline()
    accepted=json.loads((ROOT/'results/FINAL_AUDIT.json').read_text())
    if accepted['verdict']!='ACCEPTED_COMPLETED_EXPERIMENT' or accepted['execution_source_identity']!=execution_identity():raise ValueError('cannot package partial/unaccepted science')
    dest=ROOT/'RESULT_BUNDLE.zip';tmp=ROOT/'RESULT_BUNDLE.zip.tmp';files=[]
    for f in ROOT.rglob('*'):
        if not f.is_file() or f in (dest,tmp) or '.pending-' in f.name:continue
        parts=f.relative_to(ROOT).parts
        if any(x in parts for x in ('scratch','__pycache__','checkpoints','logs','qualification')) or f.name.endswith('.lock'):continue
        files.append((f,'experiment/'+str(f.relative_to(ROOT))))
    # Scientific dependencies and immutable assets referenced by source_map are included once.
    for f in __import__('runtime').frozen_integrity.source_map():
        f=Path(f)
        if f.is_file():files.append((f,'dependencies/'+str(f).lstrip('/')))
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        names=set()
        for f,n in files:
            deadline()
            if n in names:continue
            names.add(n);z.write(f,n)
        z.writestr('README.txt','Immutable experiment evidence and final W models. Local recovery checkpoints and verbose job stdout are excluded; source path manifests retain original locations. No executable pickle.\n')
    local_bytes=disk_bytes(ROOT,exclude=(tmp,dest))
    if local_bytes+tmp.stat().st_size>load_plan()['resource']['all_deliverables_including_archive_GiB_cap']*1024**3:tmp.unlink();raise ValueError('total deliverable budget including archive exceeded')
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,dest)
    total=disk_bytes(ROOT)
    if total>load_plan()['resource']['all_deliverables_including_archive_GiB_cap']*1024**3:raise ValueError('deliverable disk cap exceeded')
    atomic_json(ROOT/'results/PACKAGE.json',{'file':dest.name,'sha256':file_sha(dest),'bytes':dest.stat().st_size,'all_deliverables_bytes':total});return dest

def run(resume=False):
    require_authorized();p=load_plan();ROOT.joinpath('results').mkdir(exist_ok=True)
    # Keep an attached inhibitor during dispatch, reduction, auditing and packaging.
    c=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]) if Path('/usr/bin/caffeinate').exists() else None
    try:
        sm=manifest('screen');result=dispatch(sm,ROOT/'science/screen',resume=resume,session_origin=ORIGIN)
        if result['state']!='STAGE_COMPLETE':return result
        rows,evidence=collect(ROOT/'science/screen',screen_worlds(p),p['physical_configurations'],'screen')
        s=choose(rows,p,screen_worlds(p));s=seal_selection(s,evidence)
        atomic_bytes(ROOT/'SCREEN_REPORT.md',render_report('screen',rows,s,evidence).encode())
        if not s['winner']:
            final_audit(s);bundle=package();out={'state':'COMPLETE','verdict':s['verdict'],'bundle':str(bundle)}
        elif decision(time.monotonic()-ORIGIN,p['session'])!='DISPATCH':out={'state':'PAUSED_SESSION_LIMIT','selection_frozen':True,'next_stage':'confirm'}
        else:
            cm=manifest('confirm',selection=s);result=dispatch(cm,ROOT/'science/confirm',resume=resume,session_origin=ORIGIN)
            if result['state']!='STAGE_COMPLETE':return result
            rows,evidence=collect(ROOT/'science/confirm',confirmation_worlds(p),s['confirmation_roster'],'confirm')
            summary=confirm(rows,p,s,confirmation_worlds(p));atomic_json(ROOT/'results/SUMMARY.json',summary)
            atomic_bytes(ROOT/'CONFIRM_REPORT.md',render_report('confirm',rows,summary,evidence).encode());final_audit(s,summary)
            bundle=package();out={'state':'COMPLETE','verdict':summary['verdict'],'bundle':str(bundle)}
        atomic_json(ROOT/'results/STATUS.json',out);return out
    except DeadlinePause:
        out={'state':'PAUSED_SESSION_LIMIT','next_stage':'analysis_audit_package','science_repeat_required':False};atomic_json(ROOT/'results/STATUS.json',out);return out
    finally:
        if c:c.terminate();c.wait(timeout=5)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--resume-session',action='store_true');a=ap.parse_args();print(json.dumps(run(a.resume_session)))
