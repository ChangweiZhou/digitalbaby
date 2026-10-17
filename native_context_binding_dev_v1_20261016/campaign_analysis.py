"""Frozen-design world-level metrics, selection and confirmation. No learning."""
import collections
import math
import statistics
import bootstrap
import bind_evaluator as evaluator
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
    scores={k:sum(v)/len(v) for k,v in outcomes.items()}
    return dict(components=scores,envelope=max(scores.values()))


def boundary(history,phase,tag,reading="MEM",group="old"):
    return next(b for b in history["boundaries"] if b["phase"]==phase and b["tag"]==tag)["readings"][reading][group]["accuracy"]


def history_metrics(history,reading,budget):
    sample=[(p["records"],p["readings"][reading]["accuracy"]) for p in history["points"] if p["tag"]=="SAMPLE"]
    # Delayed probes have their own endpoint, never a second sample/CPU coordinate.
    cpu=[(p["CPU_"+reading],p["readings"][reading]["accuracy"]) for p in history["points"] if p["tag"] in ("SAMPLE","CPU")]
    final=next(p for p in history["points"] if p["tag"]=="NEW_REST")["readings"][reading]["accuracy"]
    return dict(G=evaluator.sample_auc(sample),Q=evaluator.cpu_auc(cpu,budget),final_HELD=final,
                OLD_end=boundary(history,"OLD","BEFORE_REST",reading),
                OLD_rest=boundary(history,"OLD","AFTER_REST",reading),
                OLD_final=boundary(history,"NEW","AFTER_REST",reading),
                NEW_final=boundary(history,"NEW","AFTER_REST",reading,"new"),
                samples=sample,CPU_curve=cpu)


def receipt_metrics(receipt,budget):
    readings=("MEM","LEGACY") if receipt["arm"]=="NATIVE" else ("MEM",)
    return {h:{r:history_metrics(v,r,budget) for r in readings} for h,v in receipt["histories"].items()}


def shortcut_metrics(receipt):
    old=evaluator.stream(receipt["world"],"old",receipt["old_n"])
    new=evaluator.stream(receipt["world"],"new",receipt["new_n"])
    groups=evaluator.split(receipt["world"])
    held=[(f"{a}+{b}=".encode(),str(a+b).encode()) for a,b in groups["held"]]
    taught=[(f"{a}+{b}=".encode(),str(a+b).encode()) for a,b in groups["old"]]
    observed=old+new;curves=[];components=[]
    samples=[p["records"] for p in receipt["histories"]["W"]["points"] if p["tag"]=="SAMPLE"]
    for n in samples:
        value=shortcut_scores(observed[:n],held)
        curves.append((n,value["envelope"]));components.append(dict(records=n,**value))
    return dict(G=evaluator.sample_auc(curves),final_HELD=shortcut_scores(observed,held)["envelope"],
                OLD_frequency_end=shortcut_scores(old,taught)["components"]["frequency"],
                OLD_frequency_final=shortcut_scores(observed,taught)["components"]["frequency"],
                curves=components,exact_taught_key_upper_bound=1.)


def cpu_budget(receipts):
    medians={}
    for arm in c.read(c.DESIGN)["dev"]["roster"]:
        for r in (("MEM","LEGACY") if arm=="NATIVE" else ("MEM",)):
            values=[j["histories"]["W"]["resource"]["logical_"+r+"_cpu"] for j in receipts if j["arm"]==arm]
            if len(values)!=4 or any(not math.isfinite(x) or x<=0 for x in values):
                raise AssertionError("Incomplete/invalid DEV runtime-only budget")
            medians[arm+"/"+r]=statistics.median(values)
    value=2**math.floor(math.log2(min(medians.values())))
    return dict(C_dev=value,C_star=2*value,runtime_medians=medians,
                formula="Power-of-two floor of minimum logical-condition median W CPU; computed without scores")


def build_worlds(receipts,budget):
    result={}
    for job in receipts:
        key=str(job["world"])
        world=result.setdefault(key,dict(arms={},shortcut=shortcut_metrics(job)))
        world["arms"][job["arm"]]=receipt_metrics(job,budget)
    return result


def deltas(world,selected):
    s=world["arms"][selected];w=s["W"]["MEM"];n=s["N_OLD"]["MEM"]
    native=world["arms"]["NATIVE"]["W"]
    base=lambda key:max(v[key] for v in native.values())
    shortcut=world["shortcut"]
    return dict(G=w["G"]-base("G"),Q=w["Q"]-base("Q"),
                OLD_final=w["OLD_final"]-base("OLD_final"),
                OLD_rest=w["OLD_rest"]-base("OLD_rest"),
                prior_G=w["G"]-n["G"],prior_final=w["final_HELD"]-n["final_HELD"],
                shortcut_G=w["G"]-shortcut["G"],shortcut_final=w["final_HELD"]-shortcut["final_HELD"],
                OLD_end_prior=w["OLD_end"]-n["OLD_end"],OLD_rest_prior=w["OLD_rest"]-n["OLD_rest"],
                OLD_end_frequency=w["OLD_end"]-shortcut["OLD_frequency_end"],
                OLD_rest_frequency=w["OLD_rest"]-shortcut["OLD_frequency_end"],
                OLD_final_frequency=w["OLD_final"]-shortcut["OLD_frequency_final"])


def select(worlds):
    details={};eligible=[]
    for candidate in ("BIND","BIND-ADAPT","BIND-FB"):
        differences=[deltas(w,candidate) for w in worlds.values()]
        means={k:statistics.mean(d[k] for d in differences) for k in differences[0]}
        passes=means["G"]>0 and means["Q"]>0 and means["OLD_final"]>=0 and means["prior_G"]>0 and means["shortcut_G"]>0
        special={}
        if candidate!="BIND":
            for metric in ("G","Q","OLD_final"):
                special[metric]=statistics.mean(w["arms"][candidate]["W"]["MEM"][metric]-w["arms"]["BIND"]["W"]["MEM"][metric] for w in worlds.values())
            passes=passes and special["G"]>0 and special["Q"]>0 and special["OLD_final"]>=0
        details[candidate]=dict(eligible=passes,mean_differences=means,special_over_BIND=special,
                                rank_score=min(means["G"],means["Q"]))
        if passes:eligible.append(candidate)
    parameters={"BIND":23424,"BIND-ADAPT":23552,"BIND-FB":24192}
    eligible.sort(key=lambda k:(-details[k]["rank_score"],parameters[k],("BIND","BIND-ADAPT","BIND-FB").index(k)))
    return dict(selected=eligible[0] if eligible else None,details=details,
                verdict="ADVANCE_ONE" if eligible else "NO_ADVANCE")


def interval(values):
    from scipy.stats import t
    n=len(values);mean=statistics.mean(values)
    if n<2:raise AssertionError("World-level interval requires multiple worlds")
    sd=statistics.stdev(values)
    if sd==0:
        lo=hi=mean;p=0. if mean else 1.
    else:
        se=sd/math.sqrt(n);half=float(t.ppf(.975,n-1))*se
        lo,hi=mean-half,mean+half;p=float(2*t.sf(abs(mean/se),n-1))
    return dict(n=n,mean=mean,lower=lo,upper=hi,two_sided_p=p,world_differences=values)


def confirm(worlds,selected):
    if len(worlds)!=16:raise AssertionError("Confirmation requires all 16 fixed worlds")
    all_d=[deltas(w,selected) for w in worlds.values()]
    strict=("G","Q","prior_G","prior_final","shortcut_final","OLD_end_prior","OLD_rest_prior",
            "OLD_end_frequency","OLD_rest_frequency","OLD_final_frequency")
    nonnegative=("OLD_rest","OLD_final")
    gates={k:dict(**interval([d[k] for d in all_d]),comparison=">0" if k in strict else ">=0") for k in strict+nonnegative}
    for key,g in gates.items():g["passed"]=g["lower"]>0 if key in strict else g["lower"]>=0
    frozen={"BIND":"BIND-FROZEN","BIND-ADAPT":"FROZEN-ADAPT","BIND-FB":"FROZEN-FB"}[selected]
    peer="BIND-1STEP" if selected=="BIND" else "BIND"
    attributions={name:interval([w["arms"][selected]["W"]["MEM"]["G"]-w["arms"][control]["W"]["MEM"]["G"] for w in worlds.values()])
                  for name,control in (("internal_learning",frozen),("temporal_or_special_increment",peer))}
    order=sorted(attributions,key=lambda k:attributions[k]["two_sided_p"]);alive=True;running_adjusted=0.
    for rank,key in enumerate(order):
        item=attributions[key];running_adjusted=max(running_adjusted,min(1.,(2-rank)*item["two_sided_p"]))
        item["holm_adjusted_p"]=running_adjusted
        reject=alive and item["two_sided_p"]<=.05/(2-rank)
        item["supported"]=reject and item["mean"]>0
        if not reject:alive=False
    global_pass=all(v["passed"] for v in gates.values())
    return dict(verdict="CONFIRMED_BOUNDED_IMPROVEMENT" if global_pass else "NO_CONFIRMED_REPLACEMENT",
                global_conjunctive_claim_passed=global_pass,gates=gates,attributions=attributions,
                learned_binding_supported=global_pass and attributions["internal_learning"]["supported"],
                automatic_core_adoption=False)


def load_jobs(stages,selected=None):
    receipts=[]
    for stage in stages:
        manifest=c.manifest(stage,selected)
        for job in manifest["jobs"]:
            folder=c.job_folder(stage,job)
            certificate=c.read(folder/"AUDIT.json")
            if not certificate["accepted"] or c.sha(folder/"RECEIPT.json")!=certificate["receipt_sha256"]:
                raise AssertionError("Uncertified job")
            receipts.append(c.read(folder/"RECEIPT.json"))
    return receipts


def table(worlds):
    arms=next(iter(worlds.values()))["arms"]
    lines=["| 配置/读出 | HELD G | HELD Q | OLD最终 | NEW最终 | 延迟HELD |","|---|---:|---:|---:|---:|---:|"]
    for arm in arms:
        for reading in next(iter(worlds.values()))["arms"][arm]["W"]:
            values=[w["arms"][arm]["W"][reading] for w in worlds.values()]
            means={k:statistics.mean(v[k] for v in values) for k in ("G","Q","OLD_final","NEW_final","final_HELD")}
            lines.append(f"| {arm}/{reading} | {means['G']:.4f} | {means['Q']:.4f} | {means['OLD_final']:.1%} | {means['NEW_final']:.1%} | {means['final_HELD']:.1%} |")
    return "\n".join(lines)


def analyze_dev():
    receipts=load_jobs(["DEV"])
    budget=cpu_budget(receipts)  # Runtime only; persist before inspecting/selecting scores.
    path=c.RESULTS/"DEV/CPU_BUDGET_LOCK.json"
    if path.exists():
        if c.read(path)!=budget:raise AssertionError("Locked DEV runtime changed")
    else:c.write(path,budget)
    worlds=build_worlds(receipts,budget["C_dev"]);selection=select(worlds)
    selection.update(schema="NATIVE_BIND_SELECTION_LOCK_V1",C_dev=budget["C_dev"],
                     confirmation_CPU_budget=budget["C_star"],worlds=sorted(worlds),
                     source_lock_sha256=c.sha(c.HERE/"SOURCE_LOCK.json"),selection_is_confirmatory=False)
    path=c.RESULTS/"DEV/SELECTION_LOCK.json"
    if path.exists():raise PermissionError("Selection lock is immutable; do not rerun selection")
    c.write(path,selection);c.write(c.RESULTS/"DEV/WORLD_METRICS.json",worlds)
    c.write(c.RESULTS/"DEV/CONFIRM_ADMISSION_LOCK.json",dict(
        selected=selection["selected"],selection_sha256=c.sha(path),
        CPU_budget_sha256=c.sha(c.RESULTS/"DEV/CPU_BUDGET_LOCK.json"),
        C_dev=budget["C_dev"],C_star=budget["C_star"],
        source_lock_sha256=c.sha(c.HERE/"SOURCE_LOCK.json"),
        roster=None if selection["selected"] is None else c.read(c.DESIGN)["confirm"]["rosters"][selection["selected"]]))
    text=("# NATIVE-BIND DEV\n\n"+f"结论：**{selection['verdict']}**。所选候选：{selection['selected'] or '无'}。4个DEV世界只作预定筛选，不作显著性主张。\n\n"
          +f"C_dev={budget['C_dev']} CPU秒；条件确认C*={budget['C_star']} CPU秒。所有重复、warm-up和禁写呈现均计入横轴。\n\n"+table(worlds)
          +"\n\nHELD交换轨道从未提供答案反馈；成绩为自主完整字节串及EOS。OLD/NEW是已教题诊断。\n\n无合格候选意味着这三种冻结配方在该预算下未通过全部推进门，不是整个机制家族不可能。\n")
    (c.RESULTS/"DEV_REPORT.md").write_text(text)
    return selection,worlds


def analyze_confirm(selected,budget):
    receipts=load_jobs(["CONFIRM_A","CONFIRM_B"],selected)
    worlds=build_worlds(receipts,budget);verdict=confirm(worlds,selected)
    c.write(c.RESULTS/"CONFIRM_WORLD_METRICS.json",worlds)
    c.write(c.RESULTS/"CONFIRM_INFERENCE.json",verdict)
    lines=["# NATIVE-BIND CONFIRM", "",f"结论：**{verdict['verdict']}**。16个全新配对世界；一个预选候选；无自动核心采用。", "",table(worlds),"", "| 预定门 | 世界均值差 | 双侧95%区间 | 通过 |","|---|---:|---|---|"]
    for key,g in verdict["gates"].items():
        lines.append(f"| {key} | {g['mean']*100:+.2f} pp | [{g['lower']*100:+.2f}, {g['upper']*100:+.2f}] | {g['passed']} |")
    lines.append("\n主张是全部门的合取，不是挑选显著指标。题目和字节不是独立统计样本；内部归因使用固定两项Holm家族。E3范围仅为0–9加法组合内插，不能继承减法、混合算术、语言或扩大数值范围。\n")
    (c.RESULTS/"CONFIRM_REPORT.md").write_text("\n".join(lines))
    return verdict,worlds
