"""Final same-agent recount and report: no training, no source or outcome selection."""
import json,hashlib,datetime,csv
from pathlib import Path
import e13_campaign as C
R=C.R;O=C.O;NAV=R.parent/'output/minifly-connectome-navigation';N=('add97','sub97','max97','min97','add73','sub73','max73','min73');K=('+-','-+','><','<>')
def load(p):return json.loads(p.read_text())
def mean(s,keys=N):return sum(100*s[k]['exact']/s[k]['n'] for k in keys)/len(keys)
def life(p):
 d=load(p);a=d['stages'][-2]['scores'];b=d['stages'][-1]['scores'];return a,b,sum(100*b['compose'][k]['final']/b['compose'][k]['n'] for k in K)
def average_comp(s,key='final'):return sum(100*s[k][key]/s[k]['n'] for k in K)/4

def main():
 lock=load(R/'E13_LOCK.json');status=load(O/'E13_STATUS.json');assert status['status']=='ABC_complete' and len(status['completed'])==20 and not status['failed_parts']
 for n,h in {**lock['files'],**lock['technical_artifacts']}.items():assert C.sha(R/n)==h,n
 assert load(R/'E13_CYCLES.json')['all_passed'] and load(R/'E13_CONTROLLER_REHEARSAL.json')['passed']
 complete={j['key']:j for j in status['completed']};assert len(complete)==20
 hashes={};cpu=0.;candidates=[]
 for key,j in complete.items():
  assert j['status']=='complete' and j['returncode']==0,key
  digest=C.validate(j);p=O/j['receipt'];d=load(p);hashes[str(p.relative_to(R))]=digest
  cpu+=(d['stages'][-1]['cpu'] if 'stages' in j else d['curve'][-1]['cpu'])
  if j['part']=='profile':assert C.scientific(d)==C.scientific(load(O/j['reference'])),key
  else:candidates.append(j)
 assert {j['key'] for j in candidates}=={j['key'] for j in lock['jobs']}
 pi_lock=load(NAV/'PI_FOLLOWUP_LOCK.json');pi=load(NAV/'PI_FOLLOWUP_RESULTS.json');assert C.sha(NAV/'PI_FOLLOWUP_LOCK.json')==pi['lock_sha256']
 for n,h in pi_lock['files'].items():assert C.sha(NAV/n)==h,n
 assert pi['original_PI_score_parity'] and pi['clamped_return_stop']
 assert len(pi['rows'])==6 and len(pi['episode_rows'])==288
 assert len(list(csv.DictReader((NAV/'PI_FOLLOWUP_RETURN_EPISODES.csv').open())))==288
 for row in pi['rows']:
  for panel in ('main','returning'):assert row['scores'][panel]['max_clamp_encoding_error_deg']<1e-6
  if row['arm']=='native':assert row['scores']['returning']['true_heading_clamped']['median_residual_over_outbound']>.3
 assert load(NAV/'PI_FOLLOWUP_STATUS.json')['refit']=='NOT_RUN'
 a4_losses={}
 for seed in C.SEEDS:
  d=load(O/f'e13_A4_e8_life_C_b1024_{seed}.json')
  a4_losses[str(seed)]=[dict(stage=s['task'],raw_mean=s['loss'],corrected_mean=s['loss']*(512/384 if s['task']!='compose' else 1)) for s in d['stages'][1:]]
 wall=sum(j['wall_seconds'] for j in complete.values())
 v=dict(status='ALL20_SUBPROCESSES_VERIFIED_AND_D_PRESCRIBED_STOP_DELIVERED',verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),method='Same-agent source/hash verification and raw receipt recount; not independent implementation confirmation',original_locked_inputs_unchanged=True,technical_artifact_hashes_match=True,baseline_scientific_parity_excluding_cpu_only=True,seed_count=3,candidate_jobs=18,profiles=2,part_D=dict(all_three_seeds=True,corrected_clamp_max_error_below_1e_minus6_degree=True,prescribed_stop=True,refit='NOT_RUN',diagnostic_seconds=pi['diagnostic_seconds']),cpu_seconds=cpu,summed_process_wall_seconds=wall,elapsed_campaign_seconds=status['elapsed_seconds'],peak_RSS='NOT_MEASURED',new_development_training=False,new_life_checkpoints_saved=False,A4_corrected_stage_mean_losses=a4_losses,receipt_sha256=hashes)
 (R/'E13_VERIFICATION.json').write_text(json.dumps(v,indent=2)+'\n')
 text=(R/'E13_REPORT.md').read_text();text=text[text.index('## Part A'):]
 cut=text.index('## Costs and limitations');end=text.index('### A1',cut)
 text=text[:cut]+f'''## Final compute and resources

20 successful subprocesses (18 candidates, two full baseline reproductions), no failed/interrupted candidate runs. Recorded training/evaluation CPU: {cpu:.2f} seconds ({cpu/3600:.2f} CPU hours). Summed subprocess wall: {wall:.2f} seconds ({wall/3600:.2f} process hours); parallel campaign elapsed: {status['elapsed_seconds']:.2f} seconds ({status['elapsed_seconds']/3600:.2f} hours). The last two quantities measure different things. CPU clocks follow the original scripts and exclude some startup/initial scoring. Technical-cycle work, final audit, figure generation and Part D graph setup are additional costs; Part D records {pi['diagnostic_seconds']:.3f} seconds for its accepted diagnostic loop. Initial invalid-clamp work is preserved and also additional. Peak RSS: NOT_MEASURED.

Model: 470,801 trainable parameters, 1,883,204 bytes of weight tensors plus 3,766,408 bytes of Adam moments per live learner before gradients/activations/store. Six existing development checkpoints reused (~11.3 MB); no new development training and no new scientific life checkpoints saved. Receipts, exact launch environments, losses, score curves and logs are preserved.

A4 reporting correction: operation stages contain 384 updates, but the inherited raw loss field divides the sum by 512. Its actual stage mean is raw_loss *512/384. Composition has512 updates and needs no correction. This aggregate never feeds an optimizer step or a prediction gate. Raw receipts are unchanged; E13_VERIFICATION.json preserves both raw/corrected means for every A4 stage. Other conditions have at least512 updates/stage.

''' +text[end:]
 intro='''# E13 batch-package experiment — final results

**Complete and verified:** all 18 candidate runs, both full default reproductions, and Part D's prescribed scientific stop. Three seeds in every candidate condition. No new candidate mechanisms, tasks, panels, tuning or branches. These are development results on already-used panels, not fresh independent confirmation. E13_VERIFICATION.json records the source/receipt recount.

The strongest result is lower decay at the original batch64: wd0.25 produces final retention93.75/94.06/94.58% and held composition82.29/81.00/85.00%, compared with76.17–79.79% and48.29–54.92% at wd1.0. This demonstrates a substantial decay effect at fixed batch size and update count. Lowering decay improves both endpoints in all three seeds without adding a mechanism.

Conversely, batch256 with wd4.0 collapses retention to24.48–30.06% and composition to11.33–13.25%. These arms approximately match cumulative multiplicative decay in each direction, but the high-decay batch256 results are much worse than batch64/wd1.0. Consequently, cumulative decay alone is not a complete quantitative explanation: decay per step, Adam state, gradient statistics and update count interact. The loose A2<=80 gate passes, but that does not mean it reproduced the batch64 baseline.

Batch256/lr0.002 retains90.75–91.58% and composes at75.42–77.12%. Batch1024 retains88.81–91.19%, but composition falls to39.54–42.21%; increasing batch size further is not uniformly beneficial. The saturation prediction fails in every seed.

Part B transfers the unmodified E12 batch256 package to isolated composition: held-pair final means88.33/83.04/87.71%, full-scratchpad means86.88/81.42/86.13%, versus E9 final means74.96/62.33/75.88%. All three pass the predeclared >=80 held-pair MEAN gate; individual pairs may score below80. The first two training curves still rise over their last matched-exposure interval; the third changes by-0.46points. No later/best checkpoint is substituted.

Part C does not fully preserve the ceiling: final old-operation mean improves by+1.75/+0.67/+2.44points. The strict <2point change prediction misses in one seed. BP held composition changes from89.29/91.25/88.58% to83.83/85.75/87.42% (see raw controls for exact values). Thus the optimization effect is not exclusive to the local rule, and 'the backprop ceiling did not move' is not supported globally.

Part D: the wiring gives an instantaneous displacement-direction signal; temporal accumulation is supplied by the model. Corrected true-heading clamping still leaves return residual0.710–0.717 of outbound distance in every native seed, above0.30. The unchanged follow-up explicitly requires STOP in this case, so no refit was performed. Turn-only compass errors38.1–41.1degrees are also real, but fixing heading alone cannot repair the full accumulator/encoding/readout package.

## Prediction summary

| Prediction | All-three-seed verdict |
|---|---|
| A1 final retention>=90 | Supported |
| A2 final retention<=80 | Supported, but collapse is much worse than the proposed matching baseline |
| A4 retention within3points AND composition no lower than batch256 | Refuted |
| B held-pair mean>=80 | Supported |
| C absolute final retention change<2points | Refuted: two seeds pass, one misses |
| D clamped return residual<=0.30 | Fails all seeds; required stop delivered |

## Controls, limits and interpretation

Three executed technical cycles and a production-subprocess rehearsal preceded candidates. The E12 unfinished-sibling bug, deliberate subprocess failure isolation and duplicate-launch locking were exercised. Default E8 and E7 full scientific receipts match the archived controls exactly when cpu timing fields alone are removed. Original locked scientific source and technical artifact hashes still match.

The initial Part D nominal-phase clamp was invalid (up to12.697925degrees decoded error); its script, lock, outputs and report remain archived under PI_FOLLOWUP_CYCLE1_INVALID_CLAMP and are not admitted as ceiling evidence. The accepted diagnostic numerically inverts only the fixed geometric observer to make decoded heading equal the supplied heading (maximum error<1e-6degree). It changes no learned gain. All prior PI summary scores reproduce within1e-7. The accepted lock, per-episode CSV, scatter and stop report are in output/minifly-connectome-navigation.

Historical E8/E9/E12 controls are reused, not mislabeled as new runs. The historical E7 BP comparator differs in width/dose from Part B; its gap cannot establish matched-architecture closure. The joint-ground-truth BP ceiling also has different access to old targets than self-generated replay; it is an upper reference, not a pure learning-rule ablation. Old development operations not rehearsed can still be lost; neither high new-operation means nor composition averages prove general autonomous lifelong learning. No model is promoted from these results.

'''
 # Recount the historical Part C composition numbers rather than trusting prose constants.
 vals=[]
 for seed in C.SEEDS:
  old=load(O/f'e8_life_D_{seed}.json')['stages'][-1]['scores']['compose'];vals.append(average_comp(old))
 intro=intro.replace('89.29/91.25/88.58%', '/'.join(f'{x:.2f}' for x in vals)+'%')
 (R/'E13_REPORT.md').write_text(intro+text)
 print(json.dumps({k:v for k,v in v.items() if k not in ('receipt_sha256','A4_corrected_stage_mean_losses')},indent=2))
if __name__=='__main__':main()
