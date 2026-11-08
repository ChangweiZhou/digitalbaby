"""Finish the authorized E16-V source/receipt audit, without torch or rerunning lives."""
import ast
import csv
import hashlib
import json
import statistics as st
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import e16v_verify as base
from e16v_crosswalk import SEEDS, ARMS, OPS, PAIRS, pct, meanops, comp, pair, crosswalk, more_audits, csvwrite

R = Path(__file__).resolve().parent
S = R/'E16V_SOURCE_SNAPSHOT'
D = S/'MOD97_LOCAL_RULE_20261106'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj): (R/name).write_text(json.dumps(obj,indent=2)+'\n')
def table(h, rows): return '\n'.join(base.tab(h, rows))
def source_calls(path, function):
    tree=ast.parse(path.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==function)
    return sorted(set(ast.unparse(n.func) for n in ast.walk(node) if isinstance(n,ast.Call)))

def main():
    base.main()  # Retains the independent-panel and fraction calculations and their CSVs.
    data, numbers=crosswalk(R)
    claims=more_audits(R,data)
    def stage(a,z,b=6): return data[f'{a}_{z}']['stages'][b]
    # Literal pre-result predictions; improvements also count as deviations for a two-sided "within" gate.
    gates=[]
    def gate(name,arm,seed,measurement,value,criterion,passed):
        gates.append(dict(prediction=name,arm=arm,seed=seed,measurement=measurement,value=value,criterion=criterion,passed=bool(passed)))
    for z in SEEDS:
        for label,fn in [('F1',meanops),('F2',comp),('F3',lambda q:meanops(q,73))]:
            ratio=fn(stage('T',z))/fn(stage('D',z))
            gate('P16.1','T',z,label+'_ratio',ratio,'>=0.8',ratio>=.8)
        loss=meanops(stage('T',z,2))-meanops(stage('T',z))
        gate('P16.2','T',z,'F1_loss_pp',loss,'<=10',loss<=10)
        for op in ('add','sub'):
            loss=pct(stage('T',z,2)['scores'][op+'97']['12'])-pct(stage('T',z)['scores'][op+'97']['12'])
            gate('P16.2','T',z,op+'_loss_pp',loss,'<=20',loss<=20)
        v=comp(stage('T',z,4));gate('P16.3','T',z,'held_acquisition',v,'>=70',v>=70)
        for p in PAIRS[:2]:
            v=pair(stage('T',z,4),p);gate('P16.3','T',z,p+'_acquisition',v,'>=60',v>=60)
        cost=meanops(stage('T',z))-meanops(stage('T',z),silence=True)
        gap=meanops(stage('T',z))-meanops(stage('T',z),k='20')
        gate('P16.4','T',z,'F1_silencing_cost_pp',cost,'>=30',cost>=30)
        gate('P16.4','T',z,'F1_k12_minus_k20_pp',gap,'abs<=10',abs(gap)<=10)
        for label,fn,threshold in [('F1',meanops,90),('F2',comp,80),('F3',lambda q:meanops(q,73),90)]:
            v=fn(stage('D',z));gate('P16.5','D',z,label,v,f'>={threshold}',v>=threshold)
        for label,fn,b,t in [('F1',meanops,2,5),('F2',comp,4,10)]:
            v=fn(stage('T_oracle',z,b))-fn(stage('T_oracle',z))
            gate('P16D.1','T_oracle',z,label+'_loss_pp',v,f'abs<={t}',abs(v)<=t)
        for label,fn in [('F1',meanops),('F2',comp),('F3',lambda q:meanops(q,73))]:
            v=fn(stage('T_all_balanced',z))-fn(stage('T_oracle',z))
            gate('P16D.3','T_all_balanced',z,label+'_minus_oracle_pp',v,'abs<=5',abs(v)<=5)
        for a in ('T_oracle','T_balanced','T_all_balanced'):
            for label,fn,b in [('F1',meanops,2),('F2',comp,4),('F3',lambda q:meanops(q,73),6)]:
                v=fn(stage(a,z,b))-fn(stage('T',z,b))
                gate('P16D.4',a,z,label+'_acquisition_delta_pp',v,'abs<=3',abs(v)<=3)
        for label,fn,b0,b1 in [('F1',meanops,2,4),('F2',comp,4,6)]:
            old=fn(stage('T',z,b0))-fn(stage('T',z,b1))
            new=fn(stage('T_share',z,b0))-fn(stage('T_share',z,b1))
            v=(old-new)/old
            gate('P16D2.1','T_share',z,label+'_relative_loss_reduction',v,'>=1/3',v>=1/3)
        for label,fn,b,t in [('F1',meanops,2,5),('F2',comp,4,10)]:
            v=fn(stage('M',z,b))-fn(stage('M',z))
            gate('P16D2.2','M',z,label+'_loss_pp',v,f'abs<={t}',abs(v)<=t)
        for a in ('T_share','M'):
            for label,fn,b in [('F1',meanops,2),('F2',comp,4),('F3',lambda q:meanops(q,73),6)]:
                v=fn(stage(a,z,b))-fn(stage('T',z,b))
                gate('P16D2.3',a,z,label+'_acquisition_delta_pp',v,'abs<=3',abs(v)<=3)
    csvwrite(R/'E16V_GATES.csv',gates)
    # Buffer snapshots are written after scoring and after the first path.write_text.
    # Thus the next stage's save persists previous-stage counts, while final-block counts are absent.
    buffers=[]
    for a in ARMS:
        for z in SEEDS:
            for b in range(1,7):
                actual=stage(a,z,b).get('buffer')
                expected=None
                if a in ('T_balanced','T_share'):
                    expected={f'F{f}':1024*min(2,max(0,b-2*(f-1))) for f in range(1,4) if b>2*(f-1)}
                elif a=='T_all_balanced': expected={f'F{f}':v for f,v in enumerate((9408,24000,5328),1) if b>2*(f-1)}
                elif a in ('M','T'): expected=1024*b
                verdict='NOT_LOGGED'
                if actual is not None:
                    assert actual==expected,(a,z,b,actual,expected)
                    verdict='CONFIRMED'
                buffers.append(dict(arm=a,seed=z,block=b,observed=json.dumps(actual),source_expected=json.dumps(expected),
                                    verdict=verdict,realized_draw_frequencies='NOT_MEASURED'))
    csvwrite(R/'E16V_BUFFERS.csv',buffers)
    calls={}
    for file,names in [('exp16_integrated.py',('life','render','render_pair','entry_stream','joint_stream')),
                       ('exp11_cell.py',('update','forward','respond','render','split')),
                       ('exp45.py',('batch_tensor','write','logits','_keys'))]:
        for f in names:calls[file+':'+f]=source_calls(D/file,f)
    save('E16V_SOURCE_CALLS.json',calls)
    settings=[]
    for a in ARMS:
        for z in SEEDS:
            q=stage(a,z,0)['settings'];expected_mode=a.removeprefix('T_') if a.startswith('T_') else 'uniform'
            assert q['wd']==.25 and q['replay']=='fresh' and q['trace']=='scan' and q['stage_updates']==6144
            # The original T/D receipts predate the additional replay16 setting field.
            assert q.get('replay16',expected_mode)==expected_mode
            settings.append(dict(arm=a,seed=z,receipt_settings=json.dumps(q,sort_keys=True),
                                 actual_decay_from_source=(.3 if a=='D' else .25),
                                 store_from_source=(a not in ('D','M')),
                                 settings_discrepancy=('receipt wd=.25 but optimizer D=.3' if a=='D' else ''),
                                 runtime_source_hash='NOT_RECORDED',birth_hash_in_receipt='NOT_RECORDED'))
    csvwrite(R/'E16V_SETTINGS.csv',settings)
    # Assert snapshot bytes against pinned git blobs without invoking experiment source.
    repo=Path('/private/tmp/minifly-navigation-publish-path').read_text().strip()
    revision=(S/'COMMIT.txt').read_text().strip()
    files=[p for p in S.rglob('*') if p.is_file() and p.name not in ('COMMIT.txt','REPOSITORY_PATHS.json')]
    relative=[str(p.relative_to(S)) for p in files]
    out=subprocess.run(['git','-c','gc.auto=0','ls-tree','-rz',revision,'--',*relative],cwd=repo,capture_output=True,check=True).stdout
    gitids={}
    for rec in out.split(b'\0'):
        if not rec:continue
        meta,path=rec.split(b'\t',1);gitids[path.decode()]=meta.split()[2].decode()
    manifest=[]
    for p,rel in zip(files,relative):
        raw=p.read_bytes();gid=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert gitids.get(rel)==gid,(rel,gid,gitids.get(rel))
        manifest.append(dict(path=rel,bytes=len(raw),sha256=sha(p),git_blob_sha1=gid))
    save('E16V_INPUT_MANIFEST.json',dict(commit=revision,all_snapshot_blobs_match_commit=True,files=manifest,
                                         metadata={p.name:sha(p) for p in (S/'COMMIT.txt',S/'REPOSITORY_PATHS.json')}))
    tree=json.loads((S/'REPOSITORY_PATHS.json').read_text())
    dry=sorted(p for p in tree if '/archive_e16_dry/' in p)
    births=[]
    for z in SEEDS:
        p=R/'exp45'/f'e11_dev_M_{z}.pt';relative='MOD97_LOCAL_RULE_20261106/exp45/'+p.name
        births.append(dict(seed=z,path=relative,in_pinned_repository=relative in tree,
                           local_present=p.exists(),local_sha256=sha(p) if p.exists() else None,
                           status='LOCAL_ONLY_NOT_SOURCE_AT_RUN_ATTESTED' if p.exists() else 'MISSING'))
    save('E16V_PROVENANCE.json',dict(births=births,dry_paths=dry,source_at_run_hashes='NOT_RECORDED',
                                   batch_stream_hashes='NOT_RECORDED',runtime_versions='NOT_RECORDED',
                                   wall_time_claims='NOT_INDEPENDENTLY_VERIFIED_FROM_PROCESS_CPU',
                                   source_commit=revision,life_reruns=0,network_executions=0))
    cell=[];cellcomp=[]
    for a in ARMS:
        for z in SEEDS:
            q=stage(a,z)
            for name,v in q['scores']['silenced_k12'].items():
                cell.append(dict(arm=a,seed=z,operation=name,n=v['n'],exact=v['exact'],percent=pct(v),above10=pct(v)>10))
            for p,v in q['scores']['silenced_compose'].items():
                cellcomp.append(dict(arm=a,seed=z,pair=p,n=v['n'],exact=v['exact'],final=v['final'],
                                     full_percent=pct(v),final_percent=pct(v,'final')))
    csvwrite(R/'E16V_CELL.csv',cell);csvwrite(R/'E16V_CELL_COMPOSITIONS.csv',cellcomp)
    # Exhaustive normalized score export, preserving exact counts and all checkpoints/panels.
    allscores=[]
    for a in ARMS:
        for z in SEEDS:
            for b in range(7):
                for panel,values in stage(a,z,b)['scores'].items():
                    groups=values if panel in ('compose','silenced_k12','silenced_compose') else values
                    for sub,v in groups.items():
                        if not isinstance(v,dict) or 'n' not in v: continue
                        for k in ('exact','final','intermediate'):
                            if v.get(k) is not None:
                                allscores.append(dict(arm=a,seed=z,block=b,panel=panel,subpanel=sub,metric=k,n=v['n'],correct=v[k],percent=pct(v,k)))
    csvwrite(R/'E16V_ALL_SCORES.csv',allscores)
    metrics=list(csv.DictReader((R/'E16V_METRICS.csv').open()))
    ceilings=list(csv.DictReader((R/'E16V_SHORTCUTS.csv').open()))
    exposures=list(csv.DictReader((R/'E16V_EXPOSURES.csv').open()))
    verdicts=[
        ('1 Numbers and literal predictions','DISCREPANCY',f'All {len(numbers)} numeric table components match within displayed rounding. Prose bounds and the literal acquisition gates have discrepancies; see claims and gates CSVs.'),
        ('2 Independent shortcut ceilings','CONFIRMED','12 panels ×4 existing predictors =48 exact fraction comparisons. Extra intermediate-copy is identical to ignore_o2; modal answer is a panel-specific constant upper bound.'),
        ('3 Exposure and buffers','DISCREPANCY','36 source-derived whole-batch family-share rows;90 logged buffer snapshots match source counts. Realized draw frequencies are NOT_MEASURED. Share did not double exposure or alter first loss-bearing blocks.'),
        ('4 Stream construction','CONFIRMED_WITH_LIMITS','Oracle/D and T/M have identical source-level input/target construction under the pinned helpers. No recorded batch digests attest the actual historical streams. Store and decay differences remain.'),
        ('5 Cell ablation','DISCREPANCY',f'{sum(x["above10"] for x in cell)}/{len(cell)} individual operation scores exceed10%; maximum32.25%. Low family means do not establish≤10% everywhere. Store was not ablated.'),
        ('6 Settings and provenance','DISCREPANCY','Standing births absent from pinned tree;9 dry receipts present. All D receipts label wd=.25 despite source optimizer=.3. Source-at-run, birth and batch hashes not recorded.')]
    replay=json.loads((R/'E16V_REPLAY_AUDIT.json').read_text())
    replay_costs=json.loads((R/'E16V_REPLAY_AUDIT_HISTORY.json').read_text())
    coverage=list(csv.DictReader((R/'E16V_REPLAY_COVERAGE_ERRORS.csv').open()))
    verdicts.append(('7 Added replay coverage/target-error audit','DISCREPANCY / MEASURED LIMIT',
                     'Both arms replay only1,024 distinct compositions cumulatively. Uniform has13.6–21.4% wrong full-scratchpad replay targets per late stage. Share targets were not saved; its correctness is NOT_RECORDED.'))
    # Derive counts; no hard-coded successful buffer count is allowed to drift.
    bufcount=sum(x['verdict']=='CONFIRMED' for x in buffers)
    verdicts[2]=(verdicts[2][0],verdicts[2][1],verdicts[2][2].replace('90 logged',f'{bufcount} logged'))
    lines=['# E16-V — independent source and receipt verification','',
           'Status: COMPLETE_WITH_DISCREPANCIES. Audit only: zero life reruns, zero network executions. No original scientific receipt, harness, prediction or report was edited.', '',
           f'Pinned branch: `codex/minifly-consolidated-20261023`; source commit `{revision}`. This audit covers the E16/E16-D/E16-D2 sections as they existed at that commit, all21 target lives, the complete SUMMARY_E16 tables, and the12 JOB-selected E16W receipts used in cross-study numerical claims. The added check7 from brief amendment `4e87196` is also delivered, with its amended brief preserved as E16V_AMENDED_SOURCE_BRIEF.md. Later E16-R results are outside this frozen audit.','',
           'The acquisition and retention results are real as recorded. The audit supports a benefit from richer replay, but does not establish operand coverage as the unique cause, clear the learning rule, or isolate a cell-specific cause. Several literal predictions were reported as holding after their thresholds were widened or their scope narrowed.','',
           table(['Check','Verdict','Finding'],verdicts),'',
           '## 1. Report numbers and pre-result gates','',
           f'Every numeric component of all five SUMMARY_E16 tables and all four E45 E16/D/D2 tables was independently recomputed: {len(numbers)} comparisons, zero beyond the displayed half-unit rounding interval. That agreement does not validate every prose inference. `E16V_TABLE_CROSSWALK.csv` gives source line, column, published token, exact recomputation and tolerance.','',
           table(['Arm/seed','F1 b2→b4→b6','Held b4→b6','F3 b6','F1/held/F3 vs twin'],
                 [[x['arm'], '→'.join(f'{float(x[k]):.4f}' for k in ('F1_block2','F1_block4','F1_block6')),
                   '→'.join(f'{float(x[k]):.4f}' for k in ('comp_block4','comp_block6')),f'{float(x["F3_block6"]):.4f}',
                   '/'.join(f'{float(x[k]):.4f}' for k in ('F1_ratio','comp_ratio','F3_ratio'))] for x in metrics]),'',
           '**Literal gate corrections.** P16D.4 required every control\'s acquisition on each current family to be within3pp of T. It does not hold as written: improvements beyond3pp also violate this two-sided prediction. P16D2.3 likewise required both new arms to be within3pp on every family, and does not hold. These are prediction misses, not evidence that improved acquisition is bad. P16.4 explicitly concerned F1: its≥30pp ablation cost and≤10pp family-mean extrapolation gap both pass. Do not broaden that gate retrospectively; the report\'s broader k20 wording does not hold for F3 seed116089 (10.375pp drop).', '',
           table(['Prediction','Arm','Seed','Measurement','Exact value','Criterion','Pass'],
                 [[x['prediction'],x['arm'],x['seed'],x['measurement'],f'{x["value"]:.6f}',x['criterion'],x['passed']] for x in gates if not x['passed'] or x['prediction']=='P16.4']), '',
           'P16D.2 was withdrawn before final retention endpoints and is not scored as a successful share intervention. All other literal predictions and every seed are in `E16V_GATES.csv`. The first E16 coexistence gate still fails; richer replay is a bounded development positive, not autonomous learning.','',
           table(['Section','Quoted claim','Recomputed','Verdict','Explanation'],[[x[k] for k in ('section','quoted_claim','recomputed','verdict','note')] for x in claims]),'',
           'Verdict: DISCREPANCY in prose/interpretation and literal gate reporting; numerical tables confirmed. Historical wall forecasts80min/115min/2h25 are not independently verified from process-CPU fields.','',
           '## 2. Independently rebuilt shortcut ceilings','',
           'Rebuilt the arithmetic panel construction using only Python Random and arithmetic, without importing the harness or model. Each of12 held panels has600 distinct triples. All48 existing predictor fractions match exactly. The earlier WORKING status incorrectly counted36 checks; corrected here. Copy-intermediate is exactly the already-recorded ignore-second-operation predictor, not an independent control. Modal answer is the most common correct output in the sealed panel; its reported score is an optimistic panel-derived constant upper bound, not an independently trained predictor.','',
           table(['Seed','Pair','Ignore1/a','Ignore1/b','Ignore2','Copy c','Copy intermediate','Modal constant'],
                 [[x['seed'],x['pair'],*[f'{float(x[k]):.4f}' for k in ('ignore_o1_use_a','ignore_o1_use_b','ignore_o2','answer_is_c','copy_intermediate','modal_answer')]] for x in ceilings]),'',
           'Verdict: CONFIRMED. Carry-pair shortcuts are low; comparison-pair visible-only scores reach≈65–69%, so comparison accuracy alone cannot establish composition.','',
           '## 3. Expected exposure, recorded buffer counts, missing realized counters','',
           'F1=mod97 operations, F2=composition, F3=mod73 operations. Fractions are EXPECTED WHOLE-BATCH exposure; they are not realized historical frequencies. T and M use uniform rows; D and oracle use their own equal-seen-family stream. Each replay half has32 items except branches that draw the whole batch from the current or joint stream.','',
           table(['Arm','Block','F1','F2','F3'],[[x[k] for k in ('arm','block','F1','F2','F3')] for x in exposures]),'',
           f'All{bufcount} recorded snapshots in `E16V_BUFFERS.csv` agree with the source. Balanced/share retain1,024 presentation samples per block (2,048 per completed family); all-entry buffers show F1=9,408, F2=24,000, F3=5,328 distinct entries. M logs1,024/2,048/3,072/4,096/5,120 after blocks1–5. Original T has no buffer fields. No arm persists a block6 buffer count: source saves the receipt before its final buffer mutation, and only the following stage save persists earlier mutations. Missing counters cannot be reconstructed as measurements.','',
           'Uniform→share F1 exposure during the two F2 blocks is1/2→1/2 and1/3→1/2; F2 exposure during F3 is1/4→1/4 and1/5→1/4. The first large-loss block is unchanged in both transitions. Relative to balanced, share changes F1 during F2 block2 from1/4 to1/2. The claim that it doubled exposure throughout the loss-bearing period is false. Balanced/all-entry have identical expected family shares; their comparison changes support/coverage and repetition per item, not just a biological mechanism.','',
           'Verdict: DISCREPANCY in exposure interpretation. Realized draw frequencies: NOT_MEASURED.','',
           '## 4. Same-stream assertions and remaining differences','',
           table(['Comparison','Matching source construction','Remaining differences / limits'],[
               ['T_oracle vs D','Identical family_entries, entry_stream seeds, seen-family order, joint_stream(seed+si), batch length and ground-truth render','D usesBP/.3/noStore; oracle useslocal trace/.25/Store. Data construction identical, actual runtime stream hashes absent.'],
               ['M vs T_uniform','Same Random(seed+9), generators, rng.sample buffer admission/draws, ground-truth rerender; model outputs never construct training labels','M usesBP/.25/noStore; T useslocal trace/.25/Store. Store changes scoring, not these training batches. No actual historical batch digests.'],
               ['Helpers','update/forward/trace/Store paths do not consume the Python replay RNG; render receives explicit rng; entry_stream owns its own Random','new_net seedsTorch only; no model-output-dependent branch alters data selection. AST call inventory in E16V_SOURCE_CALLS.json; current pinned source is not an immutable source-at-run attestation.']]),'',
           'The phrase “exactly the learner\'s batches” is supported by deterministic source construction, not by recorded per-batch hashes or an independent execution. This is stronger than merely similar distributions but weaker than an audited historical stream transcript. Local-vs-M training compares the backward rule under the same source-generated inputs/labels and optimizer settings; final accuracy also includes the Store difference. Store gain is6 and capacity65,536 exact context keys; do not confuse it with the replay buffer.','',
           'Verdict: CONFIRMED_WITH_LIMITS at source-construction level; exact historical stream identity NOT_ATTESTED.','',
           '## 5. Cell ablation, every arm and seed','',
           'All scores below are exact full-operation answers at k12 with the cell silenced. Silencing happens in the cell pathway; it does not disable the window pathway or the Store. No no-Store ablation was run for this audit.','',
           table(['Arm','Seed',*[o+str(m) for m in (97,73) for o in OPS],'Any>10%'],
                 [[a,z,*[f'{pct(stage(a,z)["scores"]["silenced_k12"][o+str(m)]):.2f}' for m in (97,73) for o in OPS],
                   any(x['above10'] for x in cell if x['arm']==a and x['seed']==z)] for a in ARMS for z in SEEDS]),'',
           'Silenced composition scores below are final-answer / full-scratchpad percentages, separately for all held pairs. They are not interchangeable with complete-response scores.','',
           table(['Arm','Seed',*PAIRS],[[a,z,*[f'{pair(stage(a,z),p,True):.4f}/{pair(stage(a,z),p,True,"exact"):.4f}' for p in PAIRS]] for a in ARMS for z in SEEDS]),'',
           f'Verdict: DISCREPANCY. {sum(x["above10"] for x in cell)} of168 operation/arm/seed scores exceed10%. Largest is32.25% (T_all_balanced,116089,min73); its max73 is31.5%. Low aggregate means hide these exceptions. The original T F1≥30pp cell-importance prediction still passes all3 seeds; this does not prove all retained performance originates exclusively in the cell.','',
           '## 6. Deviations and reproducibility provenance','',
           table(['Item','Finding'],[
               ['Standing memory births','All3 e11_dev_M_{seed}.pt absent from pinned repository tree. Local availability/hashes are in E16V_PROVENANCE.json, not evidence that they are the exact historical run inputs.'],
               ['Dry receipts',f'{len(dry)} files exist in archive_e16_dry/, including T,D,M,oracle,balanced,all-entry,share and2 reservoir dry controls. They record4 updates/block; do not claim dry files are missing.'],
               ['D settings','All3 D stages[0].settings.wd=.25, but source actually creates D AdamW with .3. The settings dictionary reports shared E.WD, not actual arm decay.'],
               ['Other settings','All21 receipts record fresh/scan/6144/.25, consistent with local and M source. OldT/D omit replay16 field; later arms record it. W10,H512,B64 inferred from pinned helper defaults/assertions, not runtime hash/version attestation.'],
               ['Source versions','Pinned exp16_integrated.py already includes later reservoir code. Earlier receipts omit fields now emitted; cannot pretend this exact later blob generated every earlier receipt. Helpers are pinned for this static audit.'],
               ['Missing replay telemetry','No actual per-batch digest, realized family draw count, target accuracy or historical Store occupancy/lookup-hit counter. Buffer sizes support intended eligibility, not actual exposure or an isolated cause.'],
               ['Compute provenance','Receipt processCPU excludes final evaluation and is not walltime; this audit does not certify quoted walltimes or invent RSS. No scientific compute performed.'],
               ['Immutable inputs',f'{len(manifest)} snapshot source/report/receipt blobs match pinned git objects; SHA256/byte counts are in E16V_INPUT_MANIFEST.json.']]),'',
           'Verdict: DISCREPANCY / incomplete independent-replication bundle. These gaps do not erase the observed receipt outcomes; they limit claims of independent reproducibility and unique causal explanation.','',
           '## 7. Added check: composition coverage and self-generated target errors','',
           'This check reads the existing E16W batch archives; it does not run a learner. All3 uniform streams match their compressed-file hashes and original canonical full-batch digests, with57,344 batches per seed. The truth calculator also agrees with all262,144 newly presented ground-truth composition targets per seed (786,432 checks total). All uniform replay prompts match an independent replay-RNG reconstruction exactly. Both arms\' reconstructed stage draw counts match their recorded counters.','',
           '**New finding:** during each late mod73 stage,13.59–21.40% of uniform replay\'s full six-byte composition targets are mathematically wrong;11.29–18.88% have a wrong final answer. These are targets actually passed to update, not merely a model\'s untrained outputs. Every late stage still samples only the same1,024 distinct composition prompts in each seed (4.27% of24,000 taught triples across12 taught operator pairs). Increasing share increases repetition of that support; it does not broaden it.','',
           'C_share saved aggregate exposures, malformed-output counters and buffer membership, but did not save its generated targets. Its exact correctness cannot be recovered without executing a network, which this assignment forbids. We reconstruct its prompt selection with the original Python RNG/admission ordering and verify every stage draw count. This supports the coverage calculation conditionally on that source, but is not an actual target transcript. Malformed=0 does not imply mathematically correct.','',
           table(['Arm','Seed','Stage','Composition replay count','Cumulative distinct','Wrong full target %','Wrong final target %','Held final %'],
                 [[x['arm'],x['seed'],x['task'],x['replayed_composition_presentations'],x['distinct_compositions_replayed_cumulative'],
                   f'{100*float(x["wrong_full_target_fraction"]):.4f}' if x['wrong_full_target_fraction'] not in ('NOT_RECORDED','NO_REPLAY') else x['wrong_full_target_fraction'],
                   f'{100*float(x["wrong_final_target_fraction"]):.4f}' if x['wrong_final_target_fraction'] not in ('NOT_RECORDED','NO_REPLAY') else x['wrong_final_target_fraction'],
                   f'{float(x["held_final_percent"]):.4f}'] for x in coverage if int(x['stage'])>=5]),'',
           'Verdict: coverage and wrong-target measurements delivered for uniform; coverage reconstructed with counter corroboration for share, whose target error is NOT_RECORDED. This challenges a coverage-only explanation for E16W. It does not isolate whether wrong replay targets initiate forgetting, result from forgetting, or amplify it. E16 cell replay re-renders ground-truth answers, so these window-net error rates cannot be transferred to E16/D/D2. Full54-row accounting is in E16V_REPLAY_COVERAGE_ERRORS.csv; evidence is in E16V_REPLAY_AUDIT.json.','',
           f'Read-only batch analysis cost: two file passes, {replay_costs["recorded_audit_process_cpu_seconds_total"]:.3f} process-CPU seconds and {replay_costs["recorded_audit_wall_seconds_total"]:.3f} summed wall seconds, including the validation pass. Other reporting/Git overhead and peakRSS were not measured. No scientific training or inference was performed. E16V_REPLAY_AUDIT_HISTORY.json preserves both measurements.','',
           '## Scope of the next assigned part','',
           'E16-WR is a separate authorized compute task. It is NOT_RUN by this source-only verification. Earliest start16:30EST on8November2026, and only after process inspection confirms no exp16_integrated.py workers. The capacity test will use9 stage strata and model-generated replay targets, unlike E16-R\'s3 families and recomputed ground-truth targets. AlgorithmR retains duplicate presentations, so4,096 slots do not guarantee full operation coverage. No E16-R/E16-F worker was altered. This audit has not launched any training.']
    prose='\n'.join(lines)+'\n'
    # Keep prose readable without changing operation/family identifiers or CSV keys.
    for old,new in [('all21','all 21'),('the12','the 12'),('All48','All 48'),('All60','All 60'),
                    ('All21','All 21'),('All3','All 3'),('all3','all 3'),('All9','All 9'),
                    ('All168','All 168'),('of168','of 168'),('eachof12','each of 12'),
                    ('Each of12','Each of 12'),('within3pp','within 3 pp'),('within10pp','within 10 pp'),
                    ('beyond3pp','beyond 3 pp'),('beyond3','beyond 3'),('and12','and 12'),
                    ('their≥','their ≥'),('its≥','its ≥'),('and≤','and ≤'),('with the cell','with the cell'),
                    ('block6','block 6'),('blocks1–5','blocks 1–5'),('all4','all 4'),
                    ('from1/','from 1/'),('is1/','is 1/'),('and1/','and 1/'),('than10','than 10')]:
        prose=prose.replace(old,new)
    (R/'E16V_REPORT.md').write_text(prose)
    save('E16V_STATUS.json',dict(status='COMPLETE_WITH_DISCREPANCIES',completed_at=datetime.now(timezone.utc).isoformat(),
                              source_commit=revision,life_reruns=0,network_executions=0,receipts=21,
                              cross_study_receipts=12,numeric_table_components=len(numbers),table_discrepancies=0,
                              shortcut_panels=12,shortcut_comparisons=48,buffer_snapshot_matches=bufcount,
                              cell_operation_exceptions=sum(x['above10'] for x in cell),all_score_rows=len(allscores),
                              added_check7='DELIVERED_WITH_C_SHARE_TARGETS_NOT_RECORDED',
                              replay_audit_process_cpu_seconds_total=replay_costs['recorded_audit_process_cpu_seconds_total'],
                              check_verdicts=[dict(check=a,verdict=b,summary=c) for a,b,c in verdicts],
                              part_b='NOT_RUN_TIME_AND_PROCESS_GATE_PENDING',publication='LOCAL_ONLY_NOT_PUSHED',push_authorized=False))
    print(json.dumps(dict(status='COMPLETE_WITH_DISCREPANCIES',numeric_table_components=len(numbers),buffer_snapshots=bufcount,
                          operation_exceptions=sum(x['above10'] for x in cell),source_blobs=len(manifest)),indent=2))

if __name__=='__main__':main()
