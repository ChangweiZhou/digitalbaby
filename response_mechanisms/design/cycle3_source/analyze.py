"""Locked paired-world analysis; all seven arms and all final worlds are reported."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats
from assay import ROOT,ARMS
from audit_receipts import load,audit_suite
PRIMARY=(('T','T_OFF'),('H','FE0'),('J','J_ADD'),('J','J_SHUFFLE'))


def summary(values):
    a=np.asarray(values,float)
    if len(a)==0 or not np.isfinite(a).all():raise ValueError('empty/nonfinite sample')
    mean=float(a.mean());n=len(a)
    if n<2:return dict(n=n,mean=mean,sd=None,ci95=None,p_two_sided=None)
    sd=float(a.std(ddof=1));se=sd/np.sqrt(n)
    ci=[mean-float(stats.t.ppf(.975,n-1))*se,mean+float(stats.t.ppf(.975,n-1))*se]
    p=float(2*stats.t.sf(abs(mean/se),n-1)) if se>0 else (1. if mean==0 else 0.)
    return dict(n=n,mean=mean,sd=sd,ci95=ci,p_two_sided=p)


def holm(ps):
    order=np.argsort(ps);out=np.zeros(len(ps));last=0.
    for i,k in enumerate(order):
        last=max(last,(len(ps)-i)*ps[k]);out[k]=min(1.,last)
    return out.tolist()


def metric(d,name):
    o=d['probes']['final']['W']['old']
    delta=d['probes']['final']['paired_delta']['old']['decision_decomposition']
    if name=='accuracy':return o['accuracy']
    if name=='raw_accuracy':return o['raw_accuracy']
    if name=='bias':return o['decision_decomposition']['bias_rms']
    if name=='raw_bias':return o['raw_decomposition']['bias_rms']
    if name=='H_projection':return delta['target_alignment']['H']['projection']
    if name in ('g1','g2','g12','g12_g1','g12_g2'):return delta[name]
    if name=='centered_accuracy':
        v=np.array(o['raw_values']);v-=v.mean(0)
        return float(np.mean(v.argmax(1)==d['fixture']['old_fact']['labels']))
    if name=='raw_g2_g1':
        x=o['raw_decomposition'];return x['g2']/x['g1'] if x['g1']>1e-12 else None
    if name=='benefit':return o['accuracy']-d['probes']['final']['N_old']['old']['accuracy']
    if name=='retention_loss':return o['accuracy']-d['probes']['old_day']['W']['old']['accuracy']
    if name=='birth_H':return d['probes']['birth']['W']['old']['decision_decomposition']['g12']
    raise KeyError(name)


def summarize_optional(values):
    finite=[v for v in values if v is not None]
    return dict(total=len(values),undefined=len(values)-len(finite),statistics=summary(finite) if finite else None)


def analyze(kind,worlds,*,bouts,sources=None):
    if kind=='final':
        from locked_run import verify_lock
        locked,_=verify_lock()
        if list(worlds)!=locked['worlds'] or bouts!=locked['bouts']:raise AssertionError('final analysis requires complete locked roster/dose')
        sources=locked['source_hashes']
    audit=audit_suite(kind,worlds,expected_bouts=bouts,expected_sources=sources)
    if not audit['pass_all']:raise AssertionError(json.dumps(audit['failures']))
    ds={a:[load(ROOT/'results'/kind/a/f'{w}.json.gz') for w in worlds] for a in ARMS}
    armstats={}
    metrics=('accuracy','raw_accuracy','bias','raw_bias','H_projection','g1','g2','g12',
             'g12_g1','g12_g2','centered_accuracy','raw_g2_g1','benefit','retention_loss','birth_H')
    for a,docs in ds.items():
        armstats[a]={m:summarize_optional([metric(d,m) for d in docs]) for m in metrics}
        armstats[a]['checkpoints']={phase:{g:summary([d['probes'][phase]['W'][g]['accuracy'] for d in docs])
            for g in docs[0]['probes'][phase]['W']} for phase in docs[0]['probes']}
        armstats[a]['total_j_write_l1']=summary([sum(r['j_write_l1'] for r in d['records']) for d in docs])
        armstats[a]['total_j_step_l2']=summary([sum(r.get('j_step_l2',0) for r in d['records']) for d in docs])
        armstats[a]['total_timing_l1']=summary([sum(r['timing_l1'] for r in d['records']) for d in docs])
        armstats[a]['clip_coordinates']=sum(sum(r.get('j_clipped_coordinates',0) for r in d['records']) for d in docs)
        armstats[a]['state_budget']=docs[0]['state_budget']
    primary=[]
    for a,b in PRIMARY:
        s=summary([metric(x,'accuracy')-metric(y,'accuracy') for x,y in zip(ds[a],ds[b])])
        primary.append(dict(candidate=a,control=b,**s))
    ps=[x['p_two_sided'] for x in primary]
    adjusted=holm(ps) if len(worlds)>1 else [None]*4
    for row,p in zip(primary,adjusted):row['p_holm']=p
    def contrast(a,b,m):
        v=[None if metric(x,m) is None or metric(y,m) is None else metric(x,m)-metric(y,m)
           for x,y in zip(ds[a],ds[b])]
        return summarize_optional(v)
    mechanistic={a+'-'+b:{m:contrast(a,b,m) for m in ('g1','g2','g12','g12_g1','g12_g2','H_projection','bias','benefit')}
                 for a,b in PRIMARY}
    h_bias=summary([metric(d,'bias')-metric(d,'raw_bias') for d in ds['H']])
    def positive(s):return s['ci95'] is not None and s['ci95'][0]>0
    def signature(a,b,m):
        z=mechanistic[a+'-'+b][m]
        return z['undefined']==0 and z['statistics'] is not None and positive(z['statistics'])
    behavioral=[x['mean']>0 and x['p_holm'] is not None and x['p_holm']<.05 for x in primary]
    t_preserve={m:armstats['T'][m]['statistics']['mean']>=.9*armstats['T_OFF'][m]['statistics']['mean'] for m in ('g1','g2')}
    outcomes={
        'T':dict(behavior_pass=behavioral[0],both_main_effects_at_least_90_percent=t_preserve,
                 qualified=behavioral[0] and all(t_preserve.values()) and
                 all(signature('T','T_OFF',m) for m in ('g12_g1','g12_g2','g12','H_projection'))),
        'H':dict(behavior_pass=behavioral[1],within_state_bias_change=h_bias,
                 qualified=behavioral[1] and h_bias['ci95'] is not None and h_bias['ci95'][1]<0),
        'J':dict(behavior_vs_additive=behavioral[2],behavior_vs_shuffle=behavioral[3],
                 qualified=behavioral[2] and behavioral[3] and all(signature('J',b,m)
                    for b in ('J_ADD','J_SHUFFLE') for m in ('g12','H_projection')))}
    raw_worlds={a:[dict(world=d['world'],metrics={m:metric(d,m) for m in metrics}) for d in docs] for a,docs in ds.items()}
    target_summary={k:summary([ds['FE0'][i]['probes']['final']['W']['old']['decomposition']['target_alignment'][k]['target_rms']
                              for i in range(len(worlds))]) for k in ('F','G','H')}
    return dict(kind=kind,exploratory=kind!='final',worlds=list(worlds),bouts=bouts,
        audit=audit,arms=armstats,primary=primary,mechanistic_contrasts=mechanistic,
        qualification=outcomes,world_level=raw_worlds,target_component_distribution=target_summary,
        inference='Paired worlds; two-sided Student-t intervals and tests; Holm correction across four primary behavioral contrasts; mechanistic signature gates use nominal95% supportive components and are NOT familywise-confirmed mechanistic claims',
        resources=dict(total_worker_seconds=sum(d['resource']['wall_seconds'] for docs in ds.values() for d in docs),
                       maximum_peak_rss_bytes=max(d['resource']['peak_rss_bytes'] for docs in ds.values() for d in docs)))


def render(out):
    def ci(row,scale=1):
        if row['ci95'] is None:return f"{row['mean']*scale:.3f} (one pilot world)"
        return f"{row['mean']*scale:.3f} [{row['ci95'][0]*scale:.3f}, {row['ci95'][1]*scale:.3f}]"
    lines=['# Response-mechanism results', '',
        f"Status: {'EXPLORATORY PILOT ONLY' if out['exploratory'] else 'PROSPECTIVE FRESH-WORLD FINAL RUN'}. {len(out['worlds'])} worlds; all seven arms; {out['bouts']} repetitions per cohort.",
        '', '## Primary behavioral endpoint: final old-pair accuracy', '',
        '| Arm | Accuracy %, mean [95% interval] | W−N_old percentage points |',
        '|---|---:|---:|']
    for a in ARMS:lines.append(f"| {a} | {ci(out['arms'][a]['accuracy']['statistics'],100)} | {ci(out['arms'][a]['benefit']['statistics'],100)} |")
    lines+=['','## Prespecified paired primary contrasts','','| Comparison | Difference, percentage points | Holm p |','|---|---:|---:|']
    for r in out['primary']:
        lines.append(f"| {r['candidate']} − {r['control']} | {ci(r,100)} | {r['p_holm']} |")
    lines+=['','## Mechanistic qualification','']
    for a,q in out['qualification'].items():lines.append(f"- {a}: {'meets prespecified supportive signature' if q['qualified'] else 'does not meet prespecified supportive signature'}")
    lines+=['','Mechanistic signature checks are supportive and do not have a joint familywise confirmation guarantee. Full component magnitudes, target-aligned projections of decision-centered W−N_old response, denominator flags, state budgets, exact fixtures, doses and all world-level contrasts are in FINAL_METRICS.json and immutable receipts.',
            '', '## Interpretation limits','',
            '- H preserving F/G/H is an algebraic implementation check; behavioral benefit and bias reduction are independently required',
            '- J comparisons control allocation and learning equation, not exact realized update dose; the bank has extra separately writable synapses relative to FE0',
            '- Birth tensors and N_old controls distinguish preexisting activity from old-learning-induced response; interaction RMS alone is insufficient',
            '- Offline grand-mean centering is reported as a diagnostic, not a deployed or label-tuned learner',
            '- Random mappings are globally balanced but have variable row and column effects, all reported',
            '- This assay tests taught random pairs and retention under interference, not withheld-pair transfer or general reasoning',
            '- A failure to qualify is not proof of no possible benefit; intervals and all negative outcomes are retained',
            '', '## Verification and resources','',
            f"All {out['audit']['validated_count']} receipts passed receipt-only reconstruction (not independent simulation replay). Total measured worker time {out['resources']['total_worker_seconds']/3600:.3f} h; maximum RSS {out['resources']['maximum_peak_rss_bytes']/1e6:.1f} MB.",
            '',out['inference'], '']
    return '\n'.join(lines)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('kind');p.add_argument('--worlds',nargs='+',type=int,required=True);p.add_argument('--bouts',type=int,required=True)
    a=p.parse_args();out=analyze(a.kind,a.worlds,bouts=a.bouts)
    dest=ROOT/'results'/a.kind;dest.mkdir(parents=True,exist_ok=True)
    (dest/'FINAL_METRICS.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    (dest/'REPORT.md').write_text(render(out));print(json.dumps(dict(audit_pass=out['audit']['pass_all'],kind=a.kind,worlds=len(a.worlds))))
