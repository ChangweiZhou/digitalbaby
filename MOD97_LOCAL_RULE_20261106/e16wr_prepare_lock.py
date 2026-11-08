"""Source/data preparation only; this script does not import or execute a learner."""
import hashlib
import shutil
from datetime import datetime,timezone
from e16wr_common import ROOT,OUT,SEEDS,read,save,sha

def main():
    if (ROOT/'E16WR_LOCK.json').exists():raise RuntimeError('Lock already exists; do not replace')
    auth=read(ROOT/'E16WR_AUTHORIZATION.json');assert auth['scientific_lives']==12
    brief=read(ROOT/'E16WR_BRIEF_PROVENANCE.json')
    assert sha(ROOT/'E16WR_SOURCE_BRIEF.md')==brief['sha256']
    source=['exp8_e16wr.py','e16wr_reservoir.py','e16wr_common.py','e16wr_worker.py','e16wr_campaign.py','e16wr_checks.py','e16wr_prepare_lock.py',
            'exp8_e16w2.py','e14_fast_inference.py','exp45.py','exp3_state.py','mod97_local.py']
    input_paths=['E16WR_AUTHORIZATION.json','E16WR_SOURCE_BRIEF.md','E16WR_BRIEF_PROVENANCE.json']
    baselines=[];pace=[]
    for z in SEEDS:
        input_paths.append(f'exp45/e8_dev_{z}.pt')
        for role in ('C_uni','D_ceiling'):
            j=read(OUT/f'E16W2_JOB_{role}_{z}.json')
            assert j['status']=='complete' and sha(OUT/j['receipt'])==j['receipt_sha256']
            input_paths.extend([f'exp45/E16W2_JOB_{role}_{z}.json','exp45/'+j['receipt']])
            baselines.append(dict(role=role,seed=z,receipt=j['receipt'],sha256=j['receipt_sha256'],reused=True))
            if role=='C_uni':pace.append(dict(seed=z,receipt_cpu_lower_bound=read(OUT/j['receipt'])['stages'][-1]['cpu'],observed_wall=j['wall_seconds']))
    input_paths.append('exp45/e16w_profile_C_73129_e8_life_C_73129.json')
    profile=dict(key='profile_73129',role='profile',seed=73129,capacity=1024,targets='self')
    jobs=[profile]+[dict(key=f'R{cap}_{target}_{z}',role='reservoir',seed=z,capacity=cap,targets=target)
                    for cap,target in ((1024,'self'),(4096,'self'),(16384,'self'),(16384,'observed')) for z in SEEDS]
    for j in jobs:
        profile=j['role']=='profile';schedule='blocked' if profile else 'late_family';sample='uniform' if profile else 'reservoir'
        prefix='e16wr_'+j['key']+'_'
        tag='' if profile else f'_rp1of2_late_family_reservoir_res{j["capacity"]}_{j["targets"]}'
        j['env']=dict(EXP45_HIDDEN='512',E8_BATCH='64',E8_WD='.25',E8_LR='.001',E8_SCHEDULE=schedule,E8_REPLAY_SHARE='1/2',
                      E8_BUFFER_CAP='1024',E8_REPLAY_SAMPLING=sample,E8_RESERVOIR_CAP=str(j['capacity']),E8_REPLAY_TARGETS=j['targets'],
                      E8_SAVE_STAGES='0',E8_DIAGNOSTICS='0',E8_COMPOSE_CHECKPOINTS='0',E8_RUN_PREFIX=prefix)
        j['receipt']=f'{prefix}e8_life_C{tag}_{j["seed"]}.json';j['metadata']=f'E16WR_META_{j["key"]}.json'
    text='''# E16-WR locked predictions

User confirmed all12 scientific lives, including observed-target C16384, before any E16WR training.
P1: final held composition strictly rises across self-target C1024<C4096<C16384 in every seed; largest-capacity loss is at most half the exact reused C_uni loss in every seed.
P2: self-target C16384 final mod97 operation mean is at least88% in every seed.
P3: C1024 is within5 percentage points of reused C_uni on final held composition and final old-mod97 mean, every seed. Similarity would not uniquely prove capacity rather than sampling caused prior loss.
P4: post-composition held final-answer mean is within3 percentage points of C_uni for every arm and seed, including observed-target. This is a literal two-sided prediction; improvements outside the band also count as misses.
P5: observed-target C16384 loses at least10 percentage points less held composition than self-target C16384 in every seed. Loss is each arm's own post-composition score minus final score. Within5pp is a null benefit at this capacity, not proof wrong targets never matter;5–10pp is intermediate.

No tuning, seed replacement, new tasks or additional scientific arms. Carry pairs, comparison shortcut limits, exact full scratchpads and operations all reported separately.
AlgorithmR samples duplicate presentations;4096slots do not guarantee all operation identities. Nine per-stage strata differ from the cell study's3families, and observed/self-target replay differ. No cell-only inference.
Observed treatment changes targets for all replayed operations and compositions throughout life, not only late composition targets. It tests that target-source package, including downstream Store writes and differences in acquisition.
Both largest arms save identical observed-byte payloads and use independent admission RNG plus common sampling RNG, so input order, coverage and reservoir hashes must match exactly.

## Provisional cost forecast
Prior C_uni receipts imply6.315 CPU-hours if all12cost the same; those CPU counters exclude final evaluation. Observed targets skip generation and may cost less; larger reservoirs add overhead. Budget forecast6–9 CPU-hours plus approximately0.4h full parity and short qualification, with1.5–3h wall AFTER the machine is free, provisional until measured. No promise of an under-one-hour finish. Qualification/technical costs, source hashes, reused births and comparator receipts recorded separately.

Raw model:470801 trainable scalars(1883204 float32 bytes); Adam moments approximately twice the weights plus scalar clocks, activations and gradients additional. Inference includes an unchanged65536-key exact-window Store. Reservoir cap is per stage: largest147456slots, each carrying prompt plus observed answer; Python object/RAM costs exceed byte payload. Per-worker peakRSS will be measured, not inferred from payload alone. Twelve scientific lives each57344updates×64items; one additional full default parity life. No repeated development births.
'''
    (ROOT/'E16WR_PREDICTIONS.md').write_text(text)
    source.append('E16WR_PREDICTIONS.md')
    snapshot=ROOT/'E16WR_SOURCE_SNAPSHOT';snapshot.mkdir()
    for name in source:shutil.copy2(ROOT/name,snapshot/name)
    lock=dict(status='LOCKED_BEFORE_GRADIENT_STEPS',created_at=datetime.now(timezone.utc).isoformat(),brief_commit=brief['source_commit'],
              user_confirmed=True,max_workers=9,jobs=jobs,scientific_lives=12,full_default_profiles=1,
              source_hashes={p:sha(ROOT/p) for p in source},input_hashes={p:sha(ROOT/p) for p in input_paths},
              reused_controls=baselines,measured_old_pace=pace,
              optimizer=dict(type='AdamW',lr=.001,decay=.25,betas=[.9,.98]),model=dict(window=18,hidden=512,parameters=470801,classes=257),
              sampling=dict(kind='AlgorithmR',strata=9,current_presentations_only=True,duplicates=True,completed_stages_only=True),
              target_comparison='Saved observed answer versus learner-generated replay; equal payload and matched input order atC16384',
              parity=dict(reference='exp45/e16w_profile_C_73129_e8_life_C_73129.json',excluded_fields=['cpu'],scientific_jobs_require_full_pass=True),
              technical_cycles='E16WR_CHECKS.json PASS required before full parity; no learner execution while external producer active',
              repeated_item_presentations_per_scientific_life=57344*64,
              boundaries=['synthetic byte tasks','not autonomous learning','not globally bounded for unlimited stages','not a cell-only causal comparison'])
    save(ROOT/'E16WR_LOCK.json',lock)
    print('Prepared immutable lock and predictions; no learner imported or run.')

if __name__=='__main__':main()
