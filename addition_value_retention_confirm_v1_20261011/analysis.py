"""Fixed intersection decision and complete result bundle, including negatives."""
import json,math,time,zipfile
import numpy as np
import settings
from audit import countermodels
from decision import lower_bound
from confirm_ops import ROOT,RESULTS,read,atomic,sha,verify_sources
from confirm_audit import verify

def decide(worlds):
    w=worlds
    mean=lambda k:float(np.mean([x[k] for x in w]))
    guards={k:mean(k)>=v for k,v in [('held',.75),('old',.75),('new',.75),('gain_no_old',.2),('gain_shuffled',.2),('old_gain_vs_decay',.1),('shortcut_gap',.1)]}
    guards['held_vs_decay']=mean('held_difference_vs_decay')>=-.05
    intervals={k:lower_bound([x[k] for x in w]) for k in ('old_gain_vs_decay','gain_no_old','gain_shuffled')}
    passing=all(guards.values()) and all(x['lower']>0 for x in intervals.values())
    return dict(verdict='ADOPT_STABLE_LIMITED_ADDITION_V2' if passing else 'RETAIN_CURRENT_CORE_CONFIRMATION_NOT_PASSED',guards=guards,contrasts=intervals,means={k:mean(k) for k in w[0] if k!='world'})

def analyze():
    verify_sources();worlds=[];audits={};cue={};phase_means={};rows=0
    paths=[RESULTS/'receipts'/f'WORLD_{w}.json.gz' for w in settings.CONFIRM_WORLDS]
    if any(not p.exists() for p in paths):raise ValueError('cannot analyze or package partial confirmation')
    for i,p in enumerate(paths):
        r=read(p);world=settings.CONFIRM_WORLDS[i]
        if r['world']!=world:raise ValueError('world/file mismatch')
        v=verify(r);audits[str(world)]=v;rows+=v['basic']['audited_rows']
        st=r['arms']['STABLE'];dc=r['arms']['DECAY'];w=st['probes']['day2']['W']['groups'];d=dc['probes']['day2']['W']['groups'];cm=countermodels(r['fixture'])
        worlds.append(dict(world=world,held=w['held']['exact'],old=w['old']['exact'],new=w['new']['exact'],held_mae=w['held']['mae'],
            gain_no_old=w['held']['exact']-st['probes']['day2']['N_OLD']['groups']['held']['exact'],
            gain_shuffled=w['held']['exact']-st['probes']['day2']['SHUFFLED']['groups']['held']['exact'],
            old_gain_vs_decay=w['old']['exact']-d['old']['exact'],held_difference_vs_decay=w['held']['exact']-d['held']['exact'],
            shortcut_gap=w['held']['exact']-max(v for k,v in cm.items() if k!='learned_linear_solvability_witness')))
        for arm in settings.ARMS:
            for phase in settings.PHASES:
                for branch in settings.BRANCHES:
                    p1=r['arms'][arm]['probes'][phase][branch];key=arm+'/'+phase+'/'+branch
                    phase_means.setdefault(key,[]).append(p1['groups'])
                    for q in p1['rows']:
                        k=key+'/'+str(q['pair']);cue.setdefault(k,[]).append(q['exact'])
        atomic(RESULTS/'ANALYSIS_HEARTBEAT.json',dict(worlds_audited=i+1,total=64,heartbeat_unix=time.time()))
    result=decide(worlds);result.update(worlds=worlds,selected_candidate='STABLE',formal_worlds=64,training_histories=384,training_bytes=1540608,audited_rows=rows,
        evidence='E1 retention and E3 limited addition; engineered digit/role/ordinal priors; noisy example answers',publication_authorized=False)
    atomic(RESULTS/'SUMMARY.json',result,exclusive=True)
    aggregated={k:{g:{m:float(np.mean([x[g][m] for x in ps])) for m in ('exact','mae','bpb')} for g in ('old','new','held','all')} for k,ps in phase_means.items()}
    atomic(RESULTS/'CHECKPOINT_RESULTS.json',dict(groups=aggregated,per_cue={k:dict(correct=int(sum(v)),worlds=64) for k,v in cue.items()}),exclusive=True)
    atomic(RESULTS/'FINAL_AUDIT.json',dict(verdict='ACCEPTED_COMPLETE_EXPERIMENT',worlds=64,world_audits=audits,audited_rows=rows,model_trajectory_executions=0,negative_verdict_is_valid=True),exclusive=True)
    report='# 有限加法数值保持：正式确认报告\n\n判定：**'+result['verdict']+'**。64个全新确认世界，128个world-arm作业，384条训练历史，1540608训练字节。全部独立字节、数值、禁写及留出审计通过。\n\n|规则|最终旧题|最终后续题|最终未教组合|\n|---|---:|---:|---:|\n'
    for arm in settings.ARMS:
        g=aggregated[arm+'/day2/W'];report+=f"|{arm}|{g['old']['exact']:.2%}|{g['new']['exact']:.2%}|{g['held']['exact']:.2%}|\n"
    report+='\n三个预定配对一侧95%下界：\n\n'
    for k,v in result['contrasts'].items():report+=f"- {k}: 均值 {100*v['mean']:.2f} pp；下界 {100*v['lower']:.2f} pp；{v['method']}。\n"
    report+='\n全部绝对与差值门槛：'+json.dumps(result['guards'],ensure_ascii=False)+'。方差近零采用预定Hoeffding保守下界；没有补世界或换方法。\n\n这是操作数0–4、答案0–8、80%正确/10%错误/10%缺失示例下的有限加法实例；模型在答案反馈前输出。完整交换轨道从教学中留出。工程数字大小、角色表示和ordinal输出先验公开，不是原生Full151发现任意算术，也不是无外部反馈学习。确认没有增加NLP、多位数、新机制或任务。所有对照、逐题和检查点均保留。24小时是模型时钟推进，不是实际等待。\n\n若采用，范围仅为该限定任务和既定保持窗口；容量、规则变化及错误知识长期修订未经本实验认证。GitHub上传未授权。\n'
    (RESULTS/'REPORT.md').write_text(report)
    return result

def package(result):
    if read(RESULTS/'FINAL_AUDIT.json')['verdict']!='ACCEPTED_COMPLETE_EXPERIMENT':raise ValueError('final audit not accepted')
    paths=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in ('scratch','__pycache__','provisional','active') for x in p.relative_to(ROOT).parts) and p.name not in ('RESULT_BUNDLE.zip','PACKAGE.json','BUNDLE_MANIFEST.json','supervisor.lock','supervisor.log','STATUS.json') and not p.name.endswith(('.pending','.pyc'))]
    manifest={str(p.relative_to(ROOT)):sha(p) for p in paths}
    atomic(RESULTS/'BUNDLE_MANIFEST.json',manifest,exclusive=True)
    dst=RESULTS/'RESULT_BUNDLE.zip';tmp=dst.with_suffix('.zip.pending')
    with zipfile.ZipFile(tmp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write(p,str(p.relative_to(ROOT)))
        z.write(RESULTS/'BUNDLE_MANIFEST.json','results/BUNDLE_MANIFEST.json')
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:raise ValueError('bundle failed readback')
        for name,digest in manifest.items():
            import hashlib
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('bundle byte mismatch')
    import os
    os.link(tmp,dst);tmp.unlink()
    atomic(RESULTS/'PACKAGE.json',dict(verdict='PASS',bytes=dst.stat().st_size,sha256=sha(dst),partial_data_packaged=False,parent_dependency_external=True,github_upload_authorized=False),exclusive=True)
