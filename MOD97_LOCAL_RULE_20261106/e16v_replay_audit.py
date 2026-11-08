"""Read E16W recorded batches; reconstruct sampling only, never execute networks.

C_share did not save targets. Its exposure counters and reconstructed prompt coverage
are reported separately, never substituted for a measurement of target correctness.
"""
import csv
import gzip
import hashlib
import json
import pickle
import statistics
import subprocess
import time
from pathlib import Path
from random import Random
from e16v_crosswalk import SEEDS, PAIRS, csvwrite, pct

R=Path(__file__).resolve().parent
S=R/'E16V_SOURCE_SNAPSHOT'/'MOD97_LOCAL_RULE_20261106'
O=S/'exp45'
NAMES=('add97','sub97','max97','min97','compose','add73','sub73','max73','min73')
DOSES=(6144,6144,6144,6144,8192,6144,6144,6144,6144)
CODE={114:0,116:1,120:2,110:3,99:4,115:5,117:6,121:7,122:8}

class DataOnly(pickle.Unpickler):
    def find_class(self,module,name):raise ValueError('Only primitive batch data permitted')

def truth(p):
    assert p[0]==99 and len(p)==10 and p[-1]==61
    a=int(bytes(p[1:3]));b=int(bytes(p[4:6]));c=int(bytes(p[7:9]))
    def op(code,a,b):
        return {43:(a+b)%97,45:(a-b)%97,62:max(a,b),60:min(a,b)}[code]
    i=op(p[3],a,b);f=op(p[6],i,c)
    return [*f'{i:02d}'.encode(),59,*f'{f:02d}'.encode(),256]

def main():
    t0=time.monotonic();c0=time.process_time();out=[];inputs=[];evidence=[]
    repo=Path('/private/tmp/minifly-navigation-publish-path').read_text().strip()
    rev=(R/'E16V_SOURCE_SNAPSHOT'/'COMMIT.txt').read_text().strip()
    for z in SEEDS:
        stream=R/'exp45'/f'e16w2_batches_{z}.pkl.gz'
        meta=json.loads((O/f'E16W2_BATCHES_C_uni_{z}.json').read_text())
        raw=stream.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        assert digest==meta['teacher_stream_sha256']
        rel='MOD97_LOCAL_RULE_20261106/exp45/'+stream.name
        gitid=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        ls=subprocess.run(['git','-c','gc.auto=0','ls-tree',rev,'--',rel],cwd=repo,capture_output=True,text=True,check=True).stdout
        assert ls.split()[2]==gitid
        inputs.append(dict(path=rel,sha256=digest,git_blob_sha1=gitid,bytes=len(raw),source_commit=rev))
        del raw
        for arm in ('C_uni','C_share'):
            meta=json.loads((O/f'E16W2_BATCHES_{arm}_{z}.json').read_text())
            buf=json.loads((O/meta['receipt'].replace('.json','.buffer.json')).read_text())
            scores=json.loads((O/meta['receipt']).read_text())['stages']
            assert len(buf)==9 and meta['training_batches']==57344
            rng=Random(z+9);pool=[];seen=set();cumulative_n=0;cum_wrong=0;cum_final=0
            batch_hash=hashlib.sha256();count=0
            observed_ground_truth_composition_checks=0
            fh=gzip.open(stream,'rb') if arm=='C_uni' else None
            try:
                for si,(name,dose) in enumerate(zip(NAMES,DOSES),1):
                    draws=[0]*(si-1);distinct=set();n=0;wrong=0;wrong_final=0
                    comp_pool=[p for p in pool if p[0]==99]
                    op_pool=[p for p in pool if p[0]!=99]
                    for _ in range(dose):
                        if not pool:rp=[]
                        elif arm=='C_share' and si>=6:
                            rp=rng.sample(comp_pool,16)+rng.sample(op_pool,16)
                        else:rp=rng.sample(pool,32)
                        for p in rp:
                            draws[CODE[p[0]]]+=1
                            if p[0]==99:distinct.add(tuple(p));seen.add(tuple(p))
                        n+=sum(p[0]==99 for p in rp)
                        if fh:
                            items,flag=DataOnly(fh).load();assert len(items)==64 and isinstance(flag,bool)
                            assert [p for p,t in items[len(items)-len(rp):]]==rp if rp else len(items)==64
                            encoded=json.dumps([items,flag],separators=(',',':')).encode()
                            batch_hash.update(len(encoded).to_bytes(8,'little'));batch_hash.update(encoded)
                            for p,t in items[64-len(rp):] if rp else []:
                                if p[0]==99:
                                    expected=truth(p);wrong+=t!=expected;wrong_final+=t[-3:]!=expected[-3:]
                            if si==5:
                                for p,t in items[:32]:
                                    assert p[0]==99 and t==truth(p), 'Ground-truth parser disagreement'
                                    observed_ground_truth_composition_checks+=1
                        count+=1
                    assert draws==buf[si-1]['replay_draws_by_prior_stage'],(arm,z,si,draws,buf[si-1]['replay_draws_by_prior_stage'])
                    assert sum(draws)==(0 if si==1 else dose*32)
                    assert n==(meta['exposures'][si-1]['composition'] if si>=6 else 0)
                    cumulative_n+=n;cum_wrong+=wrong;cum_final+=wrong_final
                    q=scores[si]['scores']['compose'];held=statistics.mean(pct(q[p],'final') for p in PAIRS)
                    # Advance only the replay RNG through admission; list values do not affect sample indices.
                    rng.sample(range(buf[si-1]['presented_unique']),1024)
                    additions=buf[si-1]['prompts']
                    assert len(additions)==1024 and len(set(map(tuple,additions)))==1024
                    pool+=additions
                    out.append(dict(arm=arm,seed=z,stage=si,task=name,
                                    replayed_composition_presentations=n,cumulative_replayed_composition_presentations=cumulative_n,
                                    distinct_compositions_replayed_stage=len(distinct),distinct_compositions_replayed_cumulative=len(seen),
                                    coverage_evidence=('RECORDED_BATCHES_EXACTLY_MATCH_RECONSTRUCTION' if fh else 'SOURCE_RECONSTRUCTION_MATCHES_RECORDED_DRAW_COUNTS'),
                                    composition_buffer_slots=(1024 if si>=5 else 0),
                                    wrong_full_targets=(wrong if fh else 'NOT_RECORDED'),
                                    wrong_full_target_fraction=(wrong/n if fh and n else ('NO_REPLAY' if fh else 'NOT_RECORDED')),
                                    wrong_final_target_fraction=(wrong_final/n if fh and n else ('NO_REPLAY' if fh else 'NOT_RECORDED')),
                                    cumulative_wrong_full_target_fraction=(cum_wrong/cumulative_n if fh and cumulative_n else ('NO_REPLAY' if fh else 'NOT_RECORDED')),
                                    held_final_percent=held,
                                    malformed_composition=buf[si-1]['malformed']['composition']))
                if fh:
                    try:DataOnly(fh).load()
                    except EOFError:pass
                    else:raise AssertionError('Extra teacher batch')
                    assert batch_hash.hexdigest()==meta['batch_sha256']
                assert count==57344
            finally:
                if fh:fh.close()
            evidence.append(dict(arm=arm,seed=z,all_stage_draw_counts_exact=True,
                                 all_replay_prompts_exact=(True if arm=='C_uni' else 'NOT_RECORDED'),
                                 exact_batch_digest=(batch_hash.hexdigest() if arm=='C_uni' else 'TARGETS_NOT_RECORDED'),
                                 current_ground_truth_composition_targets_verified=observed_ground_truth_composition_checks))
    csvwrite(R/'E16V_REPLAY_COVERAGE_ERRORS.csv',out)
    result=dict(status='COMPLETE_WITH_MISSING_C_SHARE_TARGETS',new_runs=0,network_executions=0,rows=len(out),
                source_commit=rev,amended_brief_commit='4e87196',inputs=inputs,checks=evidence,
                audit_process_cpu_seconds=time.process_time()-c0,audit_wall_seconds=time.monotonic()-t0,
                C_share_target_error='NOT_RECORDED; malformed is not correctness',
                causal_scope='Joint observational accounting, not an intervention that separates coverage from target error')
    (R/'E16V_REPLAY_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
