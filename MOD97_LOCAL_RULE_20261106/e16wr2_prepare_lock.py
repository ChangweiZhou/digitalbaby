"""Freeze WR2 sources, inputs, and predictions without importing a learner."""
import ast
import importlib.metadata
import shutil
import statistics
import sys
from datetime import datetime,timezone
from e16wr2_common import ROOT,OUT,SEEDS,read,save,sha

def main():
    dest=ROOT/'E16WR2_LOCK.json'
    if dest.exists():raise RuntimeError('Existing WR2 lock; do not overwrite')
    prep=read(ROOT/'E16WR2_PREPARATION.json')
    assert sha(ROOT/'E16WR2_SOURCE_BRIEF.md')==prep['brief_sha256']
    source=['exp8_e16wr2.py','exp8_e16wr.py','e16wr_reservoir.py','e16wr2_total.py','e16wr2_common.py','e16wr2_worker.py',
            'e16wr2_campaign.py','e16wr2_checks.py','e16wr2_wait.py','e16wr2_prepare_lock.py','e14_fast_inference.py','exp45.py',
            'exp3_state.py','mod97_local.py','e16wr2_input_reference.py','E16WR2_PREDICTIONS.md']
    for name in source:
        if name.endswith('.py'):ast.parse((ROOT/name).read_text(),filename=name)
    # Static source check: task construction, panel generation, and birth functions must match WR.
    def definitions(name):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse((ROOT/name).read_text()).body if isinstance(n,ast.FunctionDef)}
    prior=definitions('exp8_e16wr.py');new=definitions('exp8_e16wr2.py')
    unchanged=[n for n in ('apply','render','split','render_pair','triples','panel','develop') if n in prior]
    assert unchanged and all(prior[n]==new[n] for n in unchanged)
    inputs=['E16WR2_SOURCE_BRIEF.md','E16WR2_PREPARATION.json','E16WR_PARITY_VERIFICATION.json','E16WR_FINAL_COSTS.json','E16WR2_B_INPUT_REFERENCE.json']
    reused=[]
    for seed in SEEDS:
        inputs.append(f'exp45/e8_dev_{seed}.pt')
        for role,job in [('WR_observed',f'E16WR_JOB_R16384_observed_{seed}.json'),('W_ceiling',f'E16W2_JOB_D_ceiling_{seed}.json'),('W_uniform',f'E16W2_JOB_C_uni_{seed}.json')]:
            j=read(OUT/job);assert j['status']=='complete' and sha(OUT/j['receipt'])==j['receipt_sha256']
            inputs.extend(['exp45/'+job,'exp45/'+j['receipt']]);reused.append(dict(role=role,seed=seed,receipt=j['receipt'],sha256=j['receipt_sha256']))
        name=f'e13_A1_e8_life_C_{seed}.json';d=read(OUT/name)
        assert d['seed']==seed and d['arm']=='C' and len(d['stages'])==10 and d['stages'][-1]['task']=='compose'
        inputs.append('exp45/'+name);reused.append(dict(role='E13_A1',seed=seed,receipt=name,sha256=sha(OUT/name)))
    pace=read(ROOT/'E16WR_FINAL_COSTS.json')['per_worker_measured_life_cpu_excluding_imports']
    observed=[pace[f'R16384_observed_{seed}'] for seed in SEEDS]
    prediction=ROOT/'E16WR2_PREDICTIONS.md'
    with prediction.open('a') as f:
        f.write(f'\nMeasured WR observed-target reference CPU seconds: {observed}. Same-cost extrapolation for nine WR2 lives: {statistics.mean(observed)*9/3600:.3f} CPU-hours (excludes imports and metadata). The larger total reservoir and stage-end observation buffer may add overhead. Provisional planning range 1.8–3 CPU-hours and 15–35 minutes wall after the machine is free; qualification is separately measured. These are estimates, not limits.\n')
    jobs=[]
    for role,cap in [('A',16384),('A',65536),('B',1024)]:
        for seed in SEEDS:
            key=f'A{cap}_{seed}' if role=='A' else f'B_{seed}';prefix='e16wr2_'+key+'_'
            tag=f'_rp1of2_late_family_total_res{cap}_observed' if role=='A' else '_observed'
            env=dict(EXP45_HIDDEN='512',E8_BATCH='64',E8_WD='.25',E8_LR='.001',E8_REPLAY_SHARE='1/2',E8_BUFFER_CAP='1024',
                     E8_SCHEDULE='late_family' if role=='A' else 'blocked',E8_REPLAY_SAMPLING='total' if role=='A' else 'uniform',
                     E8_REPLAY_TARGETS='observed',E8_RESERVOIR_CAP=str(cap),E8_RUN_PREFIX=prefix,E8_SAVE_STAGES='0',
                     E8_DIAGNOSTICS='0',E8_COMPOSE_CHECKPOINTS='0')
            jobs.append(dict(key=key,role=role,capacity=cap,seed=seed,targets='observed',env=env,
                             receipt=f'{prefix}e8_life_C{tag}_{seed}.json',metadata=f'E16WR2_META_{key}.json'))
    snapshot=ROOT/'E16WR2_SOURCE_SNAPSHOT';snapshot.mkdir()
    for name in source:shutil.copy2(ROOT/name,snapshot/name)
    save(dest,dict(status='LOCKED_BEFORE_GRADIENT_STEPS',created_at=datetime.now(timezone.utc).isoformat(),
                   brief_commit=prep['source_commit'],scientific_lives=9,max_workers=9,jobs=jobs,reused_controls=reused,
                   source_hashes={p:sha(ROOT/p) for p in source},input_hashes={p:sha(ROOT/p) for p in inputs},
                   static_default_check=dict(unchanged_definition_AST=unchanged,qualification='NOT_RUN'),
                   runtime=dict(executable=sys.executable,python_version=sys.version,torch_distribution=importlib.metadata.version('torch')),
                   resource_census=dict(trainable_parameters=470801,float32_parameter_bytes=1883204,store_keys=65536,
                       max_retained_A_slots=65536,max_retained_B_slots=8192,transient_current_observation_records_additional=True,
                       updates_per_life=57344,batch=64,new_full_profiles=0,final_checkpoints_saved=False),
                   forecast=dict(old_observed_cpu_seconds=observed,same_cost_nine_life_cpu_hours=statistics.mean(observed)*9/3600),
                   machine_gate='Not before 2026-11-08 19:35 EST; no competing exp16_integrated.py or prior WR producer',
                   scientific_boundary='Synthetic scaffolded byte training; no autonomous learning claim; observed bytes saved at exposure'))
    save(ROOT/'E16WR2_STATUS.json',dict(status='LOCKED_WAITING_FOR_MACHINE',jobs=[dict(j,status='queued') for j in jobs],
         source_lock_sha256=sha(dest),scientific_lives=9,qualification='NOT_RUN',full_default_profile='REUSED_WR_NOT_RERUN'))
    print('WR2 lock and snapshots saved; no learner imported or executed.')

if __name__=='__main__':main()
