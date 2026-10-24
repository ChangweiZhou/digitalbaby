"""Post-exercise artifact audit, no training or trajectory replay."""
import hashlib
import json
import math
import numpy as np
from pathlib import Path
from v144_audit import HERE,ROOT,write_json


def finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,list):return all(finite(v) for v in value)
    return True


def main():
    locks=[HERE/'cycles/CYCLE1_SOURCE_LOCK.json',HERE/'cycles/CYCLE3_SOURCE_LOCK.json']
    locks+=list((HERE/'cycles').glob('C2_*_SOURCE_LOCK.json'))
    mismatches=[]
    for p in locks:
        for rel,digest in json.loads(p.read_text())['files'].items():
            if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:mismatches.append([p.name,rel])
    assert not mismatches,mismatches
    supplement={};records=targets=0
    for mode in ['BP','DFA','FROZEN']:
        v=json.loads((HERE/'results'/('C2_'+mode)/'RESULT.json').read_text());assert finite(v)
        assert v['complete'] and v['settings']['world']=='V144-EXERCISE-C2'
        assert v['settings']['passes']==64 and v['settings']['hidden']==128
        assert v['settings']['lr']==.03 and v['settings']['decay']==.001
        m=json.loads((HERE/'cycles'/('C2_'+mode+'_MANIFEST.json')).read_text())
        assert not set(map(tuple,m['world']['held'])) & set(map(tuple,m['world']['training_order']))
        rows=v['metrics']['snapshots'][-1]['held']['rows'];groups={}
        for row in rows:
            a,b=row['pair'];payload=bytes.fromhex(row['output_hex'])
            groups.setdefault(tuple(sorted((a,b))),[]).append(row['status']=='EOS' and len(payload)==len(str(a+b)))
        supplement[mode]={'held_EOS_at_correct_payload_length_orbit_rate':sum(sum(x)/len(x) for x in groups.values())/15,
                          'termination_is_not_complete_answer_accuracy':True}
        records+=v['records'];targets+=v['observed_target_tokens']
    native_records=0;code_arrays_verified=0
    for arm in ['LEARN','FROZEN']:
        p=HERE/'results'/('B_'+arm);v=json.loads((p/'RESULT.json').read_text())
        assert finite(v) and v['complete'] and v['records']==11250
        events=[json.loads(line) for line in (p/'EVENTS.jsonl').read_text().splitlines()]
        assert len(events)==v['records'] and all(e['target'] in (48,49) for e in events)
        assert all(finite(e) for e in events)
        for cp in v['metrics']['snapshots']:
            with np.load(p/(cp['label']+'_codes.npz'),allow_pickle=False) as arrays:
                for bank,digest in cp['code_hashes'].items():
                    assert hashlib.sha256(arrays[bank].tobytes()).hexdigest()==digest
                    code_arrays_verified+=1
        native_records+=v['records']
    c3=json.loads((HERE/'cycles/CYCLE3_CHECKS.json').read_text())
    assert c3['synthetic_resume_state_exact'] and not c3['holdback_generated_or_trained']
    assert all(v==(0 if k=='good' else 2) for k,v in c3['checker_exit_codes'].items())
    # Qualification attempts are retained, including the early fixture that stopped at birth.
    synthetic_native=48;synthetic_reference=12+24;synthetic_reference_targets=36+56
    result={'status':'THREE_CYCLES_COMPLETED_WITH_LIMITATIONS','audit_kind':'AUTHOR_RUN_ARTIFACT_AUDIT_NOT_INDEPENDENT_SCIENCE','source_locks_verified':len(locks),
      'saved_code_arrays_hash_verified':code_arrays_verified,
      'source_mismatches':mismatches,'native_records':native_records,'addition_reference_records':records,
      'addition_reference_target_tokens':targets,'synthetic_native_records':synthetic_native,
      'synthetic_reference_records':synthetic_reference,'synthetic_reference_target_tokens':synthetic_reference_targets,
      'total_record_applications':native_records+records+synthetic_native+synthetic_reference,
      'formal_scientific_jobs':0,'full_capacity_screen_executed':False,'holdback_opened':False,
      'native_addition_efficacy_claim':False,'supplementary_stopping_metrics':supplement,
      'limits':['Setup CPU incompletely measured; retained CPU is not all-in cost.',
                'Native telemetry preserves norms, not raw per-event credit vectors/gates or raw-alpha trajectories.',
                'One development world per task; no statistical efficacy confirmation.',
                'Full-screen dispatcher and complete holdback protocol are not yet qualified.']}
    write_json(HERE/'FINAL_AUDIT.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
