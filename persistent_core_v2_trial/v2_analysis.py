"""Frozen world-level paired analysis. No learner is constructed."""
import json
import math
import statistics as st
from pathlib import Path
import bootstrap
from scipy.stats import t as student_t
from v2_core import ARMS
from v2_fixture import SCIENCE_WORLDS, ASSAYS, make_world
from v2_audit import read, audit_receipt
from compact_storage import atomic_json
from v2_integrity import verify


def interval(values, alpha=.05/21, *, two_sided=False):
    center=st.mean(values); sd=st.stdev(values)
    if sd==0:
        return {'mean':center,'lower':None,'upper':None,'sd':0.,'status':'UNRESOLVED_ZERO_VARIANCE'}
    half=float(student_t.ppf(1-alpha/(2 if two_sided else 1),len(values)-1))*sd/math.sqrt(len(values))
    return {'mean':center,'lower':center-half,'upper':center+half,'sd':sd,'status':'APPROXIMATE_WORLD_T'}


def score(doc, branch, checkpoint, stage):
    world=make_world(doc['world'],doc['assay'])
    rows=next(p for p in doc['branches'][branch]['probes'] if p['name']==checkpoint)['rows']
    rows=[r for r in rows if r['stage']==stage and
          not (doc['assay']=='lifetime' and stage=='old' and r['item'] not in world['intact_old_items'])]
    return st.mean(r['correct'] for r in rows)


def vectors(docs):
    result={}
    for arm in ARMS:
        v={k:[] for k in ('old','new','revision','revision_end','e3','e3_W','e3_N',
                         'e3_old_end','e3_taught_end','e3_taught_final','e3_new_W','e3_new_N',
                         'cpu','worker','efficiency','rss','native_write_l1','removed_slow_l1')}
        for world in SCIENCE_WORLDS:
            a=docs[(world,arm,'lifetime')]; b=docs[(world,arm,'reuse')]
            for stage in ('old','new','revision'): v[stage].append(score(a,'W','final',stage))
            v['revision_end'].append(score(a,'W','revision_end','revision'))
            w=score(b,'W','final','heldout'); n=score(b,'N_old','final','heldout'); effect=w-n
            v['e3'].append(effect); v['e3_W'].append(w); v['e3_N'].append(n)
            v['e3_old_end'].append(score(b,'W','old_end','heldout')-score(b,'N_old','old_end','heldout'))
            v['e3_taught_end'].append(score(b,'W','old_end','old'))
            v['e3_taught_final'].append(score(b,'W','final','old'))
            v['e3_new_W'].append(score(b,'W','new_end','new')); v['e3_new_N'].append(score(b,'N_old','new_end','new'))
            v['efficiency'].append(effect/(b['cpu_s']/2)*3600.)
            v['cpu'].append(a['cpu_s']+b['cpu_s']); v['worker'].append(a['worker_s']+b['worker_s'])
            v['rss'].append(max(a['peak_rss_bytes'],b['peak_rss_bytes']))
            rows=[r for d in (a,b) for h in d['branches'].values() for r in h['records']]
            v['native_write_l1'].append(sum(sum(r['write']['shared_l1']+r['write']['private_l1']) for r in rows))
            v['removed_slow_l1'].append(sum(rep['removed_l1'] for r in rows for rep in r['write']['replacements']))
        result[arm]=v
    return result


def summarize(docs, source):
    vs=vectors(docs); summaries={}; decisions={}; ref=vs['V1']
    for arm,v in vs.items():
        summaries[arm]={k:{'mean':st.mean(vals),'interval95_descriptive':interval(vals,.05,two_sided=True)}
                        for k,vals in v.items() if k not in ('cpu','worker','rss','native_write_l1','removed_slow_l1')}
        summaries[arm]['resources']={'total_cpu_s':sum(v['cpu']),'total_worker_s':sum(v['worker']),
                                    'peak_worker_rss_bytes':max(v['rss']),
                                    'native_signed_alpha_l1':sum(v['native_write_l1']),
                                    'additional_removed_slow_l1':sum(v['removed_slow_l1']),
                                    'cpu_ratio_to_V1':sum(v['cpu'])/sum(ref['cpu'])}
        summaries[arm]['effects']={stage:interval([score(docs[(w,arm,'lifetime')],'W','final',stage)-
                                                  score(docs[(w,arm,'lifetime')],f'N_{stage}','final',stage)
                                                  for w in SCIENCE_WORLDS],.05,two_sided=True)
                                  for stage in ('old','new','revision')}
        if arm=='V1': continue
        tests={}
        for name,key,threshold,absolute in [('old_preservation','old',-.03,False),('new_preservation','new',-.03,False),
                                           ('revision_gain','revision',.20,False),('revision_reliability','revision',.90,True),
                                           ('prior_knowledge_reuse','e3',0.,True),('reuse_preservation','e3',-.05,False),
                                           ('capability_per_compute_gain','efficiency',0.,False)]:
            values=v[key] if absolute else [a-b for a,b in zip(v[key],ref[key])]
            bound=interval(values); bound.update(threshold=threshold,pass_gate=bound['lower'] is not None and bound['lower']>threshold)
            tests[name]=bound
        guards={'old_mean_ge_90':st.mean(v['old'])>=.90,'new_mean_ge_90':st.mean(v['new'])>=.90,
                'compute_ratio_le_1_5':sum(v['cpu'])/sum(ref['cpu'])<=1.5,
                'relation_taught_old_end_ge_80':st.mean(v['e3_taught_end'])>=.80,
                'relation_taught_final_ge_70':st.mean(v['e3_taught_final'])>=.70,
                'relation_new_W_ge_65':st.mean(v['e3_new_W'])>=.65,
                'relation_new_N_ge_65':st.mean(v['e3_new_N'])>=.65}
        decisions[arm]={'tests':tests,'guards':guards,'qualified':all(t['pass_gate'] for t in tests.values()) and all(guards.values())}
    qualified=[a for a in ARMS[1:] if decisions[a]['qualified']]
    chosen=max(qualified,key=lambda a:(summaries[a]['efficiency']['mean'],-summaries[a]['resources']['total_cpu_s'])) if qualified else None
    return {'version':'PERSISTENT_CORE_V2_2X2','identity':source,'n':96,'jobs':768,'lives':2304,
            'summaries':summaries,'decisions':decisions,'adopted_candidate':chosen,
            'verdict':'ADOPT_QUALIFIED_V2' if chosen else 'RETAIN_V1',
            'evidence':'E1 lifecycle and qualified within-family E3 only when its guards/effect pass',
            'knowledge':{'lifetime_unique_keys':64,'changed_answers':8,'lifetime_records':864,
                         'reuse_prior_taught_cues':12,'reuse_later_taught_cues':6,'reuse_records':216},
            'family':{'comparisons':21,'one_sided_alpha':.05/21,'zero_variance':'UNRESOLVED'}}


def main():
    source=verify(); folder=bootstrap.ROOT/'results/science/receipts'
    expected={f'{w}_{a}_{s}.json.gz' for w in SCIENCE_WORLDS for a in ARMS for s in ASSAYS}
    if {p.name for p in folder.glob('*.json.gz')}!=expected: raise ValueError('incomplete/excess receipt roster')
    docs={}
    for w in SCIENCE_WORLDS:
        for a in ARMS:
            for s in ASSAYS:
                d=read(folder/f'{w}_{a}_{s}.json.gz'); audit_receipt(d,source)
                if (d['world'],d['arm'],d['assay'])!=(w,a,s) or d['engineering_only']: raise ValueError('science identity')
                docs[(w,a,s)]=d
    result=summarize(docs,source); atomic_json(bootstrap.ROOT/'results/SUMMARY.json',result)
    lines=['# Persistent core V2 comparison','',f"**Verdict: {result['verdict']}**. Candidate: {result['adopted_candidate'] or 'none'}.",
           '', '96 paired worlds, 768 committed world-arm-assay jobs, 2304 lives. No extra seeds or rescue variant.',
           '', '| Core | Old unchanged | New acquired | Revised immediately | Revised after day | Held-out W−N | CPU hours |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        r=result['summaries'][a]
        lines.append(f"| {a} | {r['old']['mean']:.2%} | {r['new']['mean']:.2%} | {r['revision_end']['mean']:.2%} | {r['revision']['mean']:.2%} | {100*r['e3']['mean']:+.2f} pp | {r['resources']['total_cpu_s']/3600:.2f} |")
    lines+=['','All inference uses world-level means. The 21 one-sided family bounds and hard guards are in SUMMARY.json.',
            'Zero observed variance is unresolved. Failure to qualify does not erase a valid acquisition/revision effect.',
            'No E3 claim is made when taught competence fails or prior teaching lacks a positive effect.',
            '', 'The same configuration was tested on two separate lifetimes. Concurrent multitask use was not tested.',
            'The relation can be solved by a simple learned symbol-class rule. No broad intelligence/NLP claim follows.',
            'Teaching is the arriving byte; teacher-free learning and unprompted generation were not tested.',
            '', 'Compute-normalized effects use mean process CPU seconds per reuse life including birth, audit and checkpoint/probe work.',
            'Raw effects and CPU are reported separately; extra erasure dose is not hidden as conserved native plasticity.',
            'V1 remains the reference unless every declared qualification requirement passes. No automatic next experiment.']
    (bootstrap.ROOT/'results/REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'verdict':result['verdict'],'candidate':result['adopted_candidate']}),flush=True)


if __name__=='__main__': main()
