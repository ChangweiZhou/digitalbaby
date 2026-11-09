"""Locked E16-WR2 worker; hashes actual inputs and targets without extra RNG draws."""
import os
import sys
import json
import hashlib
import resource
import time
from e16wr2_common import ROOT,OUT,read,save,sha,verify_lock,require_machine_free,DOSES,NAMES

def main():
    key=sys.argv[1]
    lock=verify_lock();job=next(j for j in lock['jobs'] if j['key']==key)
    gate=require_machine_free()  # Runs before importing torch or constructing any learner.
    for name,value in job['env'].items():
        if os.environ.get(name)!=value:raise RuntimeError('Environment mismatch: '+name)
    import exp8_e16wr2 as E
    import e14_fast_inference as F
    E.X.respond=F.respond
    original=E.X.update
    prompt_hash=hashlib.sha256();target_hash=hashlib.sha256();stage_hash=hashlib.sha256()
    count=0;stage_rows=[];parameters=None
    budgets=list(DOSES) if job['role']=='A' else [6144]*8+[8192]
    names=list(NAMES) if job['role']=='A' else [*NAMES[:4],*NAMES[5:],'compose']
    stage_index=0;within=0;total_bytes=0
    # Read-only diagnostics: never available to the learner or replay selector.
    past_observations={};current_observations={};verified_targets=0;store_updates=0
    process0=time.process_time();wall0=time.monotonic()
    def update(m,opt,items,local,store=None):
        nonlocal count,stage_index,within,stage_hash,parameters,total_bytes,verified_targets,store_updates
        if parameters is None:parameters=sum(p.numel() for p in m.parameters())
        assert parameters==470801 and local and len(items)==64
        assert all(g['weight_decay']==.25 and g['lr']==.001 and g['betas']==(.9,.98) for g in opt.param_groups)
        current_count=64 if stage_index==0 else 32
        for p,t in items[:current_count]:
            key=tuple(p);value=tuple(t)
            assert current_observations.setdefault(key,value)==value
        for p,t in items[current_count:]:
            assert past_observations[tuple(p)]==tuple(t), 'Replay target did not match a prior observed answer'
            verified_targets+=1
        store_updates+=store is not None
        prompts=[p for p,t in items]
        raw=json.dumps(prompts,separators=(',',':')).encode()
        frame=len(raw).to_bytes(8,'little')+raw
        prompt_hash.update(frame);stage_hash.update(frame);total_bytes+=len(frame)
        raw_t=json.dumps(items,separators=(',',':')).encode()
        target_hash.update(len(raw_t).to_bytes(8,'little'));target_hash.update(raw_t)
        loss=original(m,opt,items,local,store)
        count+=1;within+=1
        if within==budgets[stage_index]:
            stage_rows.append(dict(stage=stage_index+1,task=names[stage_index],updates=within,prompt_sha256=stage_hash.hexdigest(),
                                   replay_targets_verified_against_prior_observations=verified_targets,store_write_updates=store_updates,
                                   diagnostic_distinct_current_prompts=len(current_observations)))
            for key,value in current_observations.items():assert past_observations.setdefault(key,value)==value
            current_observations.clear();verified_targets=0;store_updates=0
            within=0;stage_index+=1;stage_hash=hashlib.sha256()
        return loss
    E.X.update=update
    E.life('C',job['seed'])
    assert count==57344 and stage_index==9 and within==0
    meta=dict(status='COMPLETE',key=key,seed=job['seed'],parameters=parameters,
              updates=count,prompt_sha256=prompt_hash.hexdigest(),training_batch_sha256=target_hash.hexdigest(),
              hashed_prompt_bytes=total_bytes,stages=stage_rows,target_source=job['targets'],
              capacity=job['capacity'],learning_rule='sign',decay=.25,
              optimizer='AdamW',learning_rate=.001,betas=[.9,.98],store_used=True,
              diagnostic_observation_map='Not part of replay; retained for independent target fidelity checks; included in measured RSS',
              python_version=sys.version,torch_version=str(E.T.__version__),platform=sys.platform,
              gate=gate,lock_sha256=sha(ROOT/'E16WR2_LOCK.json'),
              process_cpu_seconds=time.process_time()-process0,wall_seconds=time.monotonic()-wall0,
              peak_rss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              peak_rss_units=('bytes' if sys.platform=='darwin' else 'KiB'))
    save(OUT/job['metadata'],meta)
    print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
