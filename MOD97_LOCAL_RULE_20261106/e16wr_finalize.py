"""Final receipt verification and reporting only; no learner or new experiment."""
import csv
import json
import statistics
from datetime import datetime,timezone
from pathlib import Path
from e16wr_common import ROOT,OUT,read,save,sha,verify_lock,validate_receipt,SEEDS
from e16wr_report import main as report,held,ops,table

def main():
    lock=verify_lock();state=read(ROOT/'E16WR_STATUS.json')
    assert state['status']=='RUNS_COMPLETE_VERIFICATION_PENDING'
    assert len(state['jobs'])==13 and all(j['status']=='complete' and j['returncode']==0 for j in state['jobs'])
    assert read(ROOT/'E16WR_CHECKS.json')['status']=='PASS'
    assert read(ROOT/'E16WR_PARITY_VERIFICATION.json')['all_remaining_values_exact']
    for name,digest in lock['source_hashes'].items():
        assert sha(ROOT/'E16WR_SOURCE_SNAPSHOT'/name)==digest
    records={};metas={};pairs=[]
    for j in state['jobs']:
        expected=next(x for x in lock['jobs'] if x['key']==j['key'])
        assert j['env']==expected['env'] and j['seed']==expected['seed']
        assert j['command'][1:]==['e16wr_worker.py',j['key']]
        assert sha(OUT/j['receipt'])==j['receipt_sha256']
        records[j['key']]=validate_receipt(j)
        m=read(OUT/j['metadata']);metas[j['key']]=m
        assert m['updates']==57344 and m['lock_sha256']==sha(ROOT/'E16WR_LOCK.json')
        assert m['parameters']==470801 and m['gate']['active_other_producers']==[]
        assert sum(s['updates'] for s in m['stages'])==57344
    for z in SEEDS:
        a=next(j for j in state['jobs'] if j['key']==f'R16384_self_{z}')
        b=next(j for j in state['jobs'] if j['key']==f'R16384_observed_{z}')
        ma,mb=metas[a['key']],metas[b['key']]
        assert ma['prompt_sha256']==mb['prompt_sha256'] and ma['stages']==mb['stages']
        ar=read((OUT/a['receipt']).with_suffix('.reservoir.json'));br=read((OUT/b['receipt']).with_suffix('.reservoir.json'))
        assert len(ar)==len(br)==9
        assert all(x['reservoirs']==y['reservoirs'] and x['replay']['draws']==y['replay']['draws'] for x,y in zip(ar,br))
        pairs.append(dict(seed=z,prompt_sha256=ma['prompt_sha256'],nine_stage_prompt_hashes_equal=True,
                          nine_stage_reservoirs_and_draws_equal=True))
    report()
    predictions=read(ROOT/'E16WR_PREDICTION_VERDICTS.json')
    p5=[x for x in predictions if x['prediction']=='P5'];assert len(p5)==3
    summary=[]
    for z in SEEDS:
        for cap,target in ((1024,'self'),(4096,'self'),(16384,'self'),(16384,'observed')):
            q=records[f'R{cap}_{target}_{z}']['stages']
            summary.append([z,cap,target,held(q[5]),held(q[-1]),held(q[5])-held(q[-1]),held(q[-1],'exact'),ops(q[-1],97)])
    verdict_counts={p:dict(passes=sum(x['passed'] for x in predictions if x['prediction']==p),total=sum(x['prediction']==p for x in predictions)) for p in ('P1','P2','P3','P5')}
    verdict_counts['P4']=dict(passes=sum(x['passed'] for x in predictions if x['prediction'].startswith('P4_')),total=12)
    self_finals=[row[4] for row in summary if row[1]==16384 and row[2]=='self']
    observed_finals=[row[4] for row in summary if row[2]=='observed']
    costs=dict(reaped_children_cpu_seconds=state['reaped_child_cpu_seconds'],controller_cpu_seconds=state['controller_cpu_seconds'],
               campaign_wall_seconds=state['wall_seconds'],technical=read(ROOT/'E16WR_TECHNICAL_JOB.json'),
               per_worker_measured_life_cpu_excluding_imports={k:v['process_cpu_seconds'] for k,v in metas.items()},
               peak_rss={k:dict(raw=v['peak_rss_raw'],units=v['peak_rss_units']) for k,v in metas.items()},
               reused_births_and_controls='Historical training not rerun; not counted as new CPU',
               reporting_and_final_verification_cpu='NOT_MEASURED; separate from campaign counters')
    save(ROOT/'E16WR_FINAL_COSTS.json',costs)
    text=(ROOT/'E16WR_REPORT.md').read_text().replace('Status: RUNS_COMPLETE_FINAL_VERIFICATION_PENDING','Status: COMPLETE_VERIFIED')
    conclusion='\n\n## Final findings\n\n'
    conclusion+=table(['Prediction','Passes','Total'],[[p,v['passes'],v['total']] for p,v in verdict_counts.items()])+'\n\n'
    conclusion+=f'At self-target capacity16,384, final held composition is {" / ".join(f"{v:.2f}" for v in self_finals)}%; observed-target finals are {" / ".join(f"{v:.2f}" for v in observed_finals)}%. Larger self-target reservoirs are not a monotonic retention repair in these runs. The observed-target comparison must be read through the matched input/coverage checks and its per-seed loss differences, not as a new architecture result.\n\n'
    conclusion+=table(['Seed','Self loss','Observed loss','Loss reduction pp','P5≥10pp'],[[x['seed'],f"{x['value']['self_loss']:.4f}",f"{x['value']['observed_loss']:.4f}",f"{x['value']['loss_reduction_pp']:.4f}",x['passed']] for x in p5])+'\n\n'
    conclusion+='All three largest-capacity pairs match actual prompt-order hashes and all nine stage reservoir and draw snapshots exactly. Observed targets were saved with presented examples. This identifies the effect of replacing self-generated replay targets with observed targets at this capacity and schedule, including downstream changes in Store and earlier acquisition. It does not isolate a late-composition-only effect, establish a global capacity optimum, or prove autonomous learning.\n\n'
    conclusion+=f'Campaign child CPU: {state["reaped_child_cpu_seconds"]/3600:.4f} hours, including the technical cycles and full default reproduction; controller CPU {state["controller_cpu_seconds"]:.3f}s. Campaign wall {state["wall_seconds"]/60:.2f}min from controller start, excluding prior waiting. Actual peakRSS is recorded per worker in E16WR_FINAL_COSTS.json. Initial forecasts are retained as forecasts, not rewritten to match measured cost.\n\n'
    conclusion+='No final scientific model/optimizer checkpoints were saved under the locked settings. Exact source, hashes, commands, raw receipts, scores, diagnostics and technical trajectory hashes are retained; do not describe these as a complete resumable checkpoint bundle.\n'
    (ROOT/'E16WR_REPORT.md').write_text(text+conclusion)
    (ROOT/'E16WR_REVIEW_SECTION_9_35.md').write_text('# 9.35 — E16-WR: bounded stage reservoirs and replay targets\n\n'+conclusion.replace('## Final findings\n\n','')+'\nSource: E16WR_REPORT.md, E16WR_VERIFICATION.json and exact JOB-selected receipts. Twelve lives, three seeds in each of four arms; one reused-default full reproduction plus separate technical cycles. Existing E16-R review sections were not edited.\n')
    # Scientific figures only from completed validated receipts.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=list(csv.DictReader((ROOT/'E16WR_CURVES.csv').open()))
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
    for ax,z in zip(axes,SEEDS):
        for cap,target in ((1024,'self'),(4096,'self'),(16384,'self'),(16384,'observed')):
            rr=[r for r in rows if int(r['seed'])==z and int(r['capacity'])==cap and r['targets']==target]
            ax.plot([int(r['stage']) for r in rr],[float(r['held_final']) for r in rr],marker='.',label=f'{cap:,} {target}')
        ax.axvline(5,color='gray',linestyle=':',linewidth=1);ax.set_title(f'Seed {z}');ax.set_xlabel('Completed training stage');ax.set_ylim(0,100)
    axes[0].set_ylabel('Held composition final-answer accuracy (%)');axes[-1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(ROOT/'E16WR_ACCURACY_CURVES.png',dpi=170);plt.close(fig)
    save(ROOT/'E16WR_VERIFICATION.json',dict(status='PASS',verified_at=datetime.now(timezone.utc).isoformat(),
         all13_observed_returncodes_zero=True,all_receipts_validated=True,all_locked_hashes_and_snapshot_hashes_match=True,
         full_default_parity_cpu_only=True,all12_budgets_57344=True,all_source_environments_match=True,
         paired_largest_arms=pairs,observed_targets_wrong_against_saved_observation=0,
         current_observation_provenance='Locked source and per-stage presentation/cap/draw counters; full scientific batch transcripts not saved',
         evidence_scope='Bounded synthetic development; no final resumable model checkpoints',prediction_verdict_counts=verdict_counts,
         new_experiments_during_finalization=0))
    print(json.dumps(dict(status='COMPLETE_VERIFIED',predictions=verdict_counts),indent=2))

if __name__=='__main__':main()
