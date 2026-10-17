"""Prespecified eight-world DEV reduction; no learning or new evaluations."""
import collections,statistics,math
import bootstrap
import bind_evaluator as evaluator
import protocol
import campaign_common as c

def most(counter):
    return min(counter,key=lambda k:(-counter[k],k)) if counter else b""


def shortcut_scores(records,items):
    frequency=collections.Counter(a for p,a in records)
    fallback=most(frequency)
    exact={};canonical={};positions={}
    for p,a in records:
        exact.setdefault(p,collections.Counter())[a]+=1
        canonical.setdefault(p.replace(b" ",b""),collections.Counter())[a]+=1
        for j,b in enumerate(p):positions.setdefault((j,b),collections.Counter())[a]+=1
    outcomes={"frequency":[],"exact":[],"canonical":[]}
    for j in range(max((len(p) for p,a in items),default=0)):
        outcomes[f"byte_position_{j}"]=[]
    for p,a in items:
        outcomes["frequency"].append(fallback==a)
        outcomes["exact"].append(most(exact[p])==a if p in exact else fallback==a)
        key=p.replace(b" ",b"")
        outcomes["canonical"].append(most(canonical[key])==a if key in canonical else fallback==a)
        for name in outcomes:
            if name.startswith("byte_position_"):
                j=int(name.rsplit("_",1)[1]);key=(j,p[j]) if j<len(p) else None
                outcomes[name].append((most(positions[key]) if key in positions else fallback)==a)
    suffix=evaluator.SuffixBackoff()
    for p,a in records:suffix.observe(p,a)
    outcomes['suffix_backoff_0_4']=[suffix.generate(p)==(a,'EOS') for p,a in items]
    scores={k:sum(v)/len(v) for k,v in outcomes.items()}
    return dict(components=scores,envelope=max(scores.values()))


def boundary(h,phase,tag,reading,group):
    return next(b for b in h['boundaries'] if (b['phase'],b['tag'])==(phase,tag))['readings'][reading][group]['accuracy']

def history_metrics(h,reading,budget):
    points=h['points']
    sample=[(p['records'],p['readings'][reading]['accuracy']) for p in points if p['tag']=='SAMPLE']
    # Rest consumes CPU and changes the available state; it is excluded from sample G but retained in CPU Q.
    cpu=[(p['CPU_'+reading],p['readings'][reading]['accuracy']) for p in points]
    return dict(G=evaluator.sample_auc(sample),Q=evaluator.cpu_auc(cpu,budget),
        HELD_FINAL=next(p for p in points if p['tag']=='NEW_REST')['readings'][reading]['accuracy'],
        OLD_END=boundary(h,'OLD','BEFORE_REST',reading,'old'),OLD_REST=boundary(h,'OLD','AFTER_REST',reading,'old'),
        OLD_FINAL=boundary(h,'NEW','AFTER_REST',reading,'old'),NEW_END=boundary(h,'NEW','BEFORE_REST',reading,'new'),
        NEW_REST=boundary(h,'NEW','AFTER_REST',reading,'new'),sample_curve=sample,CPU_curve=cpu)

def shortcuts(world):
    old=evaluator.stream(world,'old',768);new=evaluator.stream(world,'new',256);allrows=old+new
    groups=evaluator.split(world)
    render=lambda name:[(f'{a}+{b}='.encode(),str(a+b).encode()) for a,b in groups[name]]
    held=render('held');curves=[];details=[]
    for n in (0,16,64,256,512,768,1024):
        value=shortcut_scores(allrows[:n],held);curves.append((n,value['envelope']));details.append(dict(records=n,**value))
    return dict(G=evaluator.sample_auc(curves),HELD_FINAL=shortcut_scores(allrows,held)['envelope'],
        OLD_MODE=shortcut_scores(old,render('old'))['components']['frequency'],
        OLD_FINAL_MODE=shortcut_scores(allrows,render('old'))['components']['frequency'],
        NEW_MODE=shortcut_scores(allrows,render('new'))['components']['frequency'],
        curves=details,exact_taught_key_upper_bound=1.)

def choose(worlds):
    mean=lambda fn:statistics.mean(fn(w) for w in worlds.values())
    details={};eligible=[]
    for arm in protocol.FACTORIAL:
        def wr(w):return w['arms'][arm]['W']['MEM']
        def nr(w):return w['arms'][arm]['N_OLD']['MEM']
        def baseline(w,key):return max(v[key] for v in w['arms']['NATIVE']['W'].values())
        refs=['NATIVE']+(['F000'] if arm!='F000' else [])
        gaps={}
        for key in ('G','Q'):
            gaps[key+'_NATIVE']=mean(lambda w:wr(w)[key]-baseline(w,key))
            if arm!='F000':gaps[key+'_ROOT']=mean(lambda w:wr(w)[key]-w['arms']['F000']['W']['MEM'][key])
        gaps['OLD_FINAL_RETAIN']=mean(lambda w:wr(w)['OLD_FINAL']-max([baseline(w,'OLD_FINAL')]+([w['arms']['F000']['W']['MEM']['OLD_FINAL']] if arm!='F000' else [])))
        for key in ('G','HELD_FINAL'):
            gaps[key+'_CAUSAL']=mean(lambda w:wr(w)[key]-nr(w)[key])
            gaps[key+'_SHORTCUT']=mean(lambda w:wr(w)[key]-w['shortcut'][key])
        for key in ('OLD_END','OLD_REST'):
            gaps[key+'_CAUSAL']=mean(lambda w:wr(w)[key]-nr(w)[key])
            gaps[key+'_MODE']=mean(lambda w:wr(w)[key]-w['shortcut']['OLD_MODE'])
        for key in ('NEW_END','NEW_REST'):
            gaps[key+'_MODE']=mean(lambda w:wr(w)[key]-w['shortcut']['NEW_MODE'])
        passes=all(v>=0 if k=='OLD_FINAL_RETAIN' else v>0 for k,v in gaps.items())
        rank=min(gaps['G_NATIVE'],gaps['Q_NATIVE'])
        details[arm]=dict(eligible=passes,mean_differences=gaps,rank_score=rank)
        if passes:eligible.append((rank,-arm.count('1'),-protocol.FACTORIAL.index(arm),arm))
    selected=max(eligible)[-1] if eligible else None
    return dict(schema='NATIVE_BIND_SELECTION_LOCK_V2_1',selected=selected,verdict='ADVANCE_ONE' if selected else 'NO_ADVANCE',
        details=details,worlds=sorted(worlds),selection_is_confirmatory=False,automatic_adoption=False)

def analyze():
    # Inspect only required runtime fields before persisting the common CPU lock.
    entries=c.committed();assert len(entries)==72
    paths={(e['world'],e['arm']):e['receipt'] for e in entries}
    dev=c.read(c.HERE/'SPEC.json')['dev'];assert set(paths)=={(w,a) for w in dev['worlds'] for a in dev['roster']}
    medians={}
    for arm,reading in [('NATIVE','MEM'),('NATIVE','LEGACY'),('F000','MEM')]:
        values=[c.read(paths[(world,arm)])['histories']['W']['resource']['logical_'+reading+'_cpu'] for world in dev['worlds']]
        if not all(math.isfinite(x) and x>0 for x in values):raise AssertionError('Invalid CPU charge')
        medians[arm+'/'+reading]=statistics.median(values)
    budget=min(medians.values());lock=dict(C_dev=budget,C_star=2*budget,runtime_medians=medians,
        formula='Minimum raw median W CPU of NATIVE/MEM,NATIVE/LEGACY,F000 across all8 DEV worlds',computed_without_scores=True)
    path=c.RESULTS/'DEV/CPU_BUDGET_LOCK.json'
    if path.exists():raise PermissionError('Never relock an analyzed DEV budget')
    c.write(path,lock)
    worlds={}
    for world in dev['worlds']:
        arms={}
        for arm in dev['roster']:
            r=c.read(paths[(world,arm)])
            arms[arm]={h:{reading:history_metrics(v,reading,budget) for reading in (('MEM','LEGACY') if arm=='NATIVE' else ('MEM',))} for h,v in r['histories'].items()}
        worlds[str(world)]=dict(world=world,arms=arms,shortcut=shortcuts(world))
    selection=choose(worlds);selection.update(C_dev=budget,confirmation_CPU_budget=2*budget,source_lock_sha256=c.sha(c.LOCK))
    path=c.RESULTS/'DEV/SELECTION_LOCK.json'
    if path.exists():raise PermissionError('Never repeat locked selection')
    c.write(path,selection);c.write(c.RESULTS/'DEV/WORLD_METRICS.json',worlds)
    roster=protocol.confirm_roster(selection['selected']) if selection['selected'] else []
    c.write(c.RESULTS/'DEV/CONFIRM_ADMISSION_LOCK.json',dict(selected=selection['selected'],roster=roster,
        C_dev=budget,C_star=2*budget,selection_sha256=c.sha(path),CPU_budget_sha256=c.sha(c.RESULTS/'DEV/CPU_BUDGET_LOCK.json'),
        launch_authorized=False,confirmation_not_started=True))
    lines=['# NATIVE-BIND V2.1 · 完整 DEV','',f"预定筛选结果：**{selection['verdict']}**；候选：{selection['selected'] or '无'}。8 个 DEV 世界用于配方筛选，不是确认性证据。",'',
        '72 个完整任务，144 条连续历史，147,456 条外部记录应用。DEV_A 与 DEV_B 的冻结科学核心、数据剂量和评分相同。',
        '',f'C_dev = {budget:.6f} CPU 秒，条件确认 C* = {2*budget:.6f} CPU 秒。预算先按参考条件运行时间锁定，再计算行为指标。','',
        '| 配置/读出 | HELD G | HELD Q | 最终延迟 HELD | 最终 OLD | 最终 NEW |','|---|---:|---:|---:|---:|---:|']
    for arm in dev['roster']:
        for reading in (('MEM','LEGACY') if arm=='NATIVE' else ('MEM',)):
            means={k:statistics.mean(w['arms'][arm]['W'][reading][k] for w in worlds.values()) for k in ('G','Q','HELD_FINAL','OLD_FINAL','NEW_REST')}
            lines.append(f"| {arm}/{reading} | {means['G']:.4f} | {means['Q']:.4f} | {means['HELD_FINAL']:.1%} | {means['OLD_FINAL']:.1%} | {means['NEW_REST']:.1%} |")
    lines+=['','HELD 的完整交换轨道从未获得答案反馈；评分为模型自主生成的完整字节串及 EOS。OLD/NEW 成绩属于已教题 acquisition/retention。训练记录数包括重复和禁写呈现。',
        '', '捷径包含仅从已观察记录拟合的答案众数、精确键、空格规范化键、单字节位置众数和 0–4 字节后缀 backoff。所有门和差值见 DEV/SELECTION_LOCK.json。',
        '', 'NO_ADVANCE 表示本轮固定配方未通过全部推进门，不代表整个机制家族或任务不可能。ADVANCE_ONE 只允许后续独立确认，不构成新核心采用。未启动确认。']
    report='\n'.join(lines)+'\n';(c.RESULTS/'DEV_REPORT.md').write_text(report);(c.RESULTS/'REPORT.md').write_text(report)
    return selection,worlds,lock
