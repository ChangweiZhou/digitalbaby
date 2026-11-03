#!/usr/bin/env python3
"""Reproduce the V41 analysis from checksummed completed records. NumPy, SciPy, Matplotlib."""
from pathlib import Path
import argparse, csv, hashlib, itertools, json, math
import numpy as np
from scipy.stats import t

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def sha(x):
    return hashlib.sha256(x).hexdigest()

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '|'+ '|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=Path('v41_results'))
    ap.add_argument('--out',type=Path,default=Path('v41_analysis'))
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((a.results/'PLAN.json').read_text());pid=sha(canonical(plan))
    records={};audit={'plan_sha256':pid,'source_sha256':plan['source_sha256'],'completed_records':0,'checks':{}}
    source=a.results/'elm_v41_mechanism_closures.py'
    assert sha(source.read_bytes())==plan['source_sha256']
    assert plan['mode']=='science' and len(plan['jobs'])==12
    for job in plan['jobs']:
        e,s=job['experiment'],job['seed']
        p=a.results/'completed'/f'{e}_{s}.json';envelope=json.loads(p.read_text())
        assert envelope['sha256']==sha(canonical(envelope['data']))
        assert envelope['identity']==dict(plan=pid,experiment=e,seed=s,config=job['config'])
        r=envelope['data']['result'];records[e,s]=r;audit['completed_records']+=1
        keys=[(x['arm'],x['metric'],x.get('group','')) for x in r['metrics']]
        assert len(keys)==len(set(keys))
        if e in ('hybrid','stationarity'):
            for arm in r['metrics_by_arm']:
                g=r['groups'];assert g['whole'][arm]['n']==r['byte_events']==15344
                assert g['target'][arm]['n']+g['nontarget'][arm]['n']==15344
                assert math.isclose(g['whole'][arm]['loss_sum'],g['target'][arm]['loss_sum']+g['nontarget'][arm]['loss_sum'],rel_tol=1e-11)
            assert r['probe_invariance_checks']>0
        if e=='stationarity':
            assert r['total_byte_presentations']==19432 and r['warmup_byte_events']==4088
            assert r['frozen_normalizer_unchanged'] and r['frozen_output_changed_after_fork']
            assert r['frozen_output_updates_after_fork']==r['live_output_updates_after_fork']==15344
            assert len({c['matched_store_projection_normalizer_bytes'] for c in r['costs'].values()})==1
        if e=='hybrid':
            assert r['physical_slot_states']==2
            assert all(x['slots']<=384 and x['traces']<=512 for x in r['checkpoint_memory'])
            for d in r['read_diagnostics'].values():
                assert math.isclose(d['full_minus_slots_loss_sum'],d['full_minus_read_loss_sum']+d['read_minus_slots_loss_sum'],abs_tol=1e-8)
        if e=='utility':
            assert r['validity']['intervention_target_presentation']==320
            if r['validity']['valid']:
                assert r['selection_audit']['equal_budget'] and r['selection_audit']['identical_score_multiset']
                assert r['sample_accounting']['followup_target_presentations']==224
                assert r['sample_accounting']['followup_learner_writes']==0
            else:assert r['status']=='invalid' and 'selection_audit' not in r
    seeds=sorted(s for e,s in records if e=='hybrid')
    def values(e,arm,metric):
        return np.array([next(x['value'] for x in records[e,s]['metrics'] if x['arm']==arm and x['metric']==metric and not x.get('group')) for s in seeds])
    def mean(e,arm,metric):return float(values(e,arm,metric).mean())
    contrasts=[]
    selections=[('hybrid','READ-HYBRID','SLOTS',['whole_bpb','target_bpb','nontarget_bpb','E1_all','A_after_D_heldout']),
                ('hybrid','FULL-HYBRID','READ-HYBRID',['whole_bpb','target_bpb','nontarget_bpb','E1_all','A_after_D_heldout']),
                ('stationarity','PHASE-FROZEN','PHASE-LIVE',['whole_bpb','E1_all','A_after_D_heldout','fixed_context_active_fraction']),
                ('stationarity','PHASE-FROZEN','LAG',['whole_bpb','E1_all','A_after_D_heldout'])]
    for e,left,right,metrics in selections:
        for metric in metrics:
            d=values(e,left,metric)-values(e,right,metric);mu=float(d.mean())
            half=float(t.ppf(.975,3)*d.std(ddof=1)/2)
            null=[abs(float((d*np.array(signs)).mean())) for signs in itertools.product((-1,1),repeat=4)]
            p=float(np.mean(np.array(null)>=abs(mu)-1e-12))
            contrasts.append(dict(experiment=e,contrast=f'{left} minus {right}',metric=metric,n=4,
                mean=mu,ci_low=mu-half,ci_high=mu+half,exact_sign_flip_p=p,
                **{f'seed_{s}':float(v) for s,v in zip(seeds,d)}))
    write_csv(a.out/'v41_primary_contrasts.csv',contrasts)
    def contrast(e,left,right,metric):
        return next(x for x in contrasts if (x['experiment'],x['contrast'],x['metric'])==(e,f'{left} minus {right}',metric))
    def fmt_contrast(e,left,right,metric):
        x=contrast(e,left,right,metric)
        return f"{x['mean']:+.4f} [{x['ci_low']:+.4f}, {x['ci_high']:+.4f}]"
    utility=[records['utility',s] for s in seeds];valid=[r for r in utility if r['validity']['valid']]
    assert len(valid)==1  # This report's narrative describes the locked V41 outcome.
    ur=valid[0]
    um={(m['arm'],m['metric']):m['value'] for m in ur['metrics'] if not m.get('group')}
    hb=mean('hybrid','READ-HYBRID','whole_bpb')-mean('hybrid','SLOTS','whole_bpb')
    target=(mean('hybrid','READ-HYBRID','target_bpb')-mean('hybrid','SLOTS','target_bpb'))/14
    background=13*(mean('hybrid','READ-HYBRID','nontarget_bpb')-mean('hybrid','SLOTS','nontarget_bpb'))/14
    assert math.isclose(hb,target+background,abs_tol=1e-11)
    parts=[
      '**V41: what the three mechanism experiments establish**',
      'All 12 scheduled jobs finished and were recorded. Eight hybrid/stationarity jobs completed their full workloads; three utility jobs reported failed predeclared gates, and one executed its eviction comparison. No thresholds, seeds or workloads were changed after seeing scientific results.',
      'My strongest recommendation is to retain the separation between retrieval and writing. READ-HYBRID combines immediate transfer with SLOTS’ durable history on this task. Feeding the stronger fallback into admission changes which memories survive and sacrifices some old recall. Freezing normalization helps, but the stationary lag representation remains the better reference. Utility-based eviction remains unresolved because the assay was valid in only one seed.',
      'The latest recoverable chat recommendations were three small closure experiments: a common eviction event, a read-versus-writing hybrid split, and lag/live/frozen representations. The complete numerical protocol was unavailable. DESIGN.md therefore identifies our numerical choices explicitly as refinements, rather than claiming exact reproduction of an unseen design. These are new focused experiments, not directly comparable reruns of V40’s larger workloads.',
      table(['Experiment','Independent seeds','Training per seed','Status'],[
        ['Utility','4','320 contextual next-byte observations','1 valid; 3 gate failures'],
        ['Hybrid','4','15,344 continuous bytes / 1,096 records','4 completed'],
        ['Stationarity','4','4,088 warmup + 15,344 main bytes','4 completed']]),
      'Utility contexts contain twelve supplied bytes that are covariates, not twelve extra learning events. Its valid seed evaluates 224 frozen target predictions over 48 unique followup contexts. Repeated predictions, identities, aliases and checkpoints are not independent replications. Across all jobs, 139,104 continuous byte presentations plus 1,280 contextual target observations were used for training; the two kinds of samples should remain separately reported.',
      '**The read-only hybrid preserves recall while improving one-shot transfer.** Means below are over the same four seeds. E1 is the held-out rendering probe after the first training presentation; old recall tests untouched first-wave identities after wave D.',
      table(['Arm','Whole bits/byte ↓','Target bits/byte ↓','E1 accuracy','Old held-out recall'],[
        [arm,f"{mean('hybrid',arm,'whole_bpb'):.4f}",f"{mean('hybrid',arm,'target_bpb'):.4f}",f"{100*mean('hybrid',arm,'E1_all'):.2f}%",f"{100*mean('hybrid',arm,'A_after_D_heldout'):.2f}%"]
        for arm in ('BASE','SUPER','SLOTS','READ-HYBRID','FULL-HYBRID')]),
      f"READ-HYBRID literally shares SLOTS’ memory and writes. Its whole-stream change is {fmt_contrast('hybrid','READ-HYBRID','SLOTS','whole_bpb')} bits/byte (paired mean and pointwise Student 95% interval). The target contribution is {target:+.4f} bits per whole-stream byte, while non-target bytes contribute {background:+.4f}; they sum to {hb:+.4f}. Thus strong target transfer does not translate into an equally large whole-stream gain. Each record has one target byte and thirteen non-target bytes.",
      f"FULL-HYBRID minus READ-HYBRID changes whole loss by {fmt_contrast('hybrid','FULL-HYBRID','READ-HYBRID','whole_bpb')} bits/byte and old held-out recall by {100*contrast('hybrid','FULL-HYBRID','READ-HYBRID','A_after_D_heldout')['mean']:+.2f} percentage points. The old-recall interval is −43.11 to +5.61 percentage points, so this remains a four-seed descriptive warning. This comparison measures the total consequence of the different writing history; it does not attribute the loss to one particular admission event. FULL is not uniformly better simply because its average bits/byte is lower.",
      'The independent event audit makes the durability concern more specific: FULL has zero consolidated-slot evictions in all four seeds, yet all thirteen of its old held-out failures lack memory authority. READ answers twelve of those thirteen correctly. FULL also has substantially fewer target promotions and unused slot capacity. This supports investigating suppressed consolidation; it does not prove an identity-level causal chain because trace eviction and failed matching remain possible.',
      'READ still costs 0.10658 whole-stream bits/byte relative to SUPER. The authority-stratified audit localizes its non-target penalty to exact probationary traces and similarity reads from supported slots; fallback reads add zero relative to SUPER because they are identical there. This identifies the next read-authority question. Target/non-target evaluator labels must not become an oracle input to a future policy. See hybrid_scientific_audit.md for the full decomposition and seed values.',
      'The hybrid uses the inherited 4,096-unit superposed layer: 8,388,608 output bytes plus 786,432 projection bytes, in addition to its slot memory and common learner. This is not a reduced-neuron or lower-memory result. Shared execution also cannot establish standalone runtime. Slot/traces stay within the fixed capacities of 384/512.',
      '**Freezing normalization improves this phase model, but does not close the gap to LAG.** Both phase branches start with identical weights and normalizer state after causal warmup. Only normalizer updates stop in FROZEN; all 15,344 subsequent output updates continue in both branches.',
      table(['Arm','Main bits/byte ↓','Warmup-inclusive bits/byte ↓','E1 accuracy','Old held-out recall','Fixed-context active overlap'],[
        [arm,f"{mean('stationarity',arm,'whole_bpb'):.4f}",f"{mean('stationarity',arm,'including_warmup_bpb'):.4f}",f"{100*mean('stationarity',arm,'E1_all'):.2f}%",f"{100*mean('stationarity',arm,'A_after_D_heldout'):.2f}%",f"{mean('stationarity',arm,'fixed_context_active_fraction'):.4f}"]
        for arm in ('LAG','PHASE-LIVE','PHASE-FROZEN')]),
      f"FROZEN minus LIVE changes main loss by {fmt_contrast('stationarity','PHASE-FROZEN','PHASE-LIVE','whole_bpb')} bits/byte; FROZEN minus LAG is {fmt_contrast('stationarity','PHASE-FROZEN','LAG','whole_bpb')}. The discovered periods were {[records['stationarity',s]['period'] for s in seeds]} bytes. Fixed-context overlap is measured at matched phase. Freezing removes normalizer drift there, but recurring contexts at their actual stream times can still receive different codes when period discovery selects a multiple of the fourteen-byte record length. Mean actual-time recurring-context overlap is 1.0000 for LAG, 0.9414 for LIVE and 0.9764 for FROZEN; FROZEN reaches only 0.9056 in the period-28 seed. The raw report includes both measurements.",
      'Geometry changes and removed drift occur together in this intervention, so the results do not establish that drift alone mediates every prediction difference. All arms retain output plasticity and reach full current-wave acquisition in this run. Equal charged representation budgets include explicitly reported inert padding for LAG; that allowance does not improve its functional model.',
      '**The utility comparison did not qualify as a four-seed experiment.** The fixed event occurred after 320 target observations in every seed, with no early eviction. Gates required at least sixteen resident slots, at least two postbirth utility observations for every resident, and at least 90% held-out acquisition.',
      table(['Seed','Resident slots','Held-out acquisition','Minimum postbirth uses','Outcome'],[
        [r['seed'],r['validity']['resident_slots'],f"{100*r['validity']['heldout_accuracy']:.2f}%",r['validity']['minimum_postbirth_uses'],
         'Valid' if r['validity']['valid'] else '; '.join({'all_residents_observed_after_birth':'A resident has fewer than two postbirth uses','heldout_acquisition':'Acquisition below 90%','minimum_resident_count':'Fewer than sixteen resident slots','no_preintervention_eviction':'Early eviction'}[k] for k,v in r['validity']['gates'].items() if not v)] for r in utility]),
      f"In the sole valid seed ({ur['seed']}), every policy demoted eleven of forty-seven resident slots at the same time. Utility-minus-LFU future loss was {um['UTILITY-RATE','future_loss_per_target']-um['LFU-RATE','future_loss_per_target']:+.6f} bits/target; utility-minus-random was {um['UTILITY-RATE','future_loss_per_target']-um['RANDOM','future_loss_per_target']:+.6f}; utility tied shuffled utility to numerical precision. These are single-seed descriptions, with no confidence interval or general ranking.",
      'The surviving utility fixture has a further limitation: base prediction loss is exactly eight bits on every held-out identity. It therefore supplies little independent variation in future predictive value beyond recurrence. It cannot establish an advantage of value over frequency. Demotion restores two exact traces: all five eviction arms use 152,988 semantic bytes versus 138,732 for no eviction. Equal victim counts isolate slot authority, but do not imply a RAM saving.',
      '**What I would carry forward.** Keep READ-HYBRID’s firewall between fallback reads and corroborated writing as the reference composition. Keep LAG as the stationary representation reference. A future utility test should first make acquisition and postbirth exposure reliable across separate engineering seeds and vary base predictability independently of demand; then lock one new event and its controls. The present gates should remain visible as failed outcomes, rather than being weakened after inspection. A small phase-alignment test could address the residual time dependence before spending compute on another capacity sweep. Those followups were not run here.',
      'All effect estimates are conditional on these synthetic workloads. The intervals are pointwise, exploratory, and based on four paired seeds; no multiplicity adjustment or broad qualification is claimed. v41_primary_contrasts.csv contains every seed difference and an exact sign-reversal sensitivity calculation. With four seeds the exact two-sided sign-reversal test has a minimum p-value of 0.125 and cannot attain p<0.05. A parametric Student test can produce smaller p-values under its distributional assumptions; those assumptions cannot be assessed reliably from four seeds. Utility has only one eligible comparison and is excluded from that calculation.',
      '**Execution and reproducibility.** The unchanged runner embeds its source dependencies, saves complete state every thirty-two records, atomically replaces checksummed checkpoints, retains a previous generation, validates source/configuration identities, rejects concurrent writers, and skips verified completed jobs. SIGINT/SIGTERM stop after a block; rerunning the same command resumes. macOS caffeinate is automatic. Initial environment and invocation records are preserved. A checkpoint snapshot was saved during scientific execution. The initial three-worker run hit the environment memory limit after nine completions; its broken pool was interrupted and the three remaining jobs resumed sequentially from checkpoints with the same source and plan. The default is one worker. A fatal process-pool failure can require an interrupt before resuming; automatic recovery of a broken pool is not implemented. The final compact results ZIP omits completed checkpoint states.',
      'Engineering checks include original numerical equivalence, read-only probe invariance, freeze-only-normalizer behavior, exact victim budgets, shuffled-score preservation, serialization/resume equivalence, and rejection of corrupted or mismatched records. The command-line smoke workflow paused, resumed all three jobs, and then reused all completed jobs with zero new learning. A final scientific rerun also verified and reused all twelve completed jobs with zero new learning. The numeric-key JSON checksum issue found during smoke testing was repaired before scientific execution.',
      f"Source SHA-256: `{plan['source_sha256']}`. The analysis independently verifies all twelve result checksums, identity/plan links, denominators, read-loss decomposition, observed capacity bounds, normalization state, and validity-gate handling. See v41_integrity_audit.json, verification.json, and the execution logs for evidence.",
      'Files: elm_v41_mechanism_closures.py runs or resumes the science and regenerates raw reports; analyze_v41_results.py regenerates this focused analysis and figure from the completed JSON records. v41_results_results.zip contains the raw records, plan, source, environment, per-seed metrics and paired summaries. The complete handoff archive also contains the protocol, verification evidence and focused reports.'
    ]
    (a.out/'v41_analysis_report.md').write_text('\n\n'.join(parts)+'\n')
    audit['checks']={k:True for k in ['completed_record_hashes','plan_identities','source_hash','unique_metric_keys','continuous_sample_denominators','loss_decomposition','observed_memory_capacities','frozen_normalizer_and_plastic_output','matched_stationarity_state_budgets','utility_gate_failures_preserved','equal_valid_utility_interventions']}
    audit['utility_valid_seeds']=[r['seed'] for r in valid]
    (a.out/'v41_integrity_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors=['#657b91','#386b91','#d09638','#268878','#ad5768']
    def panel(ax,e,arms,metric,title,ylabel,scale=1):
        for i,arm in enumerate(arms):
            v=values(e,arm,metric)*scale
            ax.bar(i,v.mean(),color=colors[i],alpha=.82,width=.65)
            ax.scatter(i+np.linspace(-.13,.13,4),v,s=22,facecolors='white',edgecolors='#26313d',linewidth=.8,zorder=4)
        ax.set_xticks(range(len(arms)),[x.replace('-','\n') for x in arms],fontsize=9)
        ax.set_ylabel(ylabel);ax.set_title(title,loc='left',fontweight='bold',fontsize=12)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    panel(axes[0,0],'hybrid',['SUPER','SLOTS','READ-HYBRID','FULL-HYBRID'],'whole_bpb','A  Hybrid prediction loss','Bits per byte (lower is better)')
    panel(axes[0,1],'hybrid',['SUPER','SLOTS','READ-HYBRID','FULL-HYBRID'],'A_after_D_heldout','B  Old held-out recall','Accuracy (%)',100)
    axes[0,1].set_ylim(0,105)
    panel(axes[1,0],'stationarity',['LAG','PHASE-LIVE','PHASE-FROZEN'],'whole_bpb','C  Stationarity prediction loss','Main-stream bits per byte')
    panel(axes[1,1],'stationarity',['LAG','PHASE-LIVE','PHASE-FROZEN'],'fixed_context_active_fraction','D  Matched-phase code stability','Active overlap after wave D')
    axes[1,1].set_ylim(0,1.05)
    fig.suptitle('V41 mechanism experiments: means and all four seed observations',fontsize=15,fontweight='bold')
    fig.savefig(a.out/'v41_results_figure.png',dpi=180);plt.close(fig)
    print(json.dumps({'completed_records':12,'utility_valid':len(valid),'report':str(a.out/'v41_analysis_report.md')}))

if __name__=='__main__':main()
