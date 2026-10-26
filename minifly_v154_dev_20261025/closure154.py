"""Postrun closure checks of saved evidence only: no model import/inference."""
from pathlib import Path
from collections import Counter
import json, hashlib, time, resource

H = Path(__file__).resolve().parent
R = H.parent
load = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    start = time.process_time()
    assert load(H/'FINAL_AUDIT.json')['status'] == 'ACCEPTED'
    ledger = [json.loads(x) for x in (H/'LEDGER.jsonl').read_text().splitlines()]
    events = [x['event'] for x in ledger]
    freeze = events.index('GLOBAL_ACQUISITION_FROZEN')
    assert events[:freeze].count('NATIVE_GROUP_COMMIT') == 10
    assert events.index('CYCLE2_COMPLETE') > freeze
    lock = load(H/'ACQUISITION_LOCK.json')
    assert len(lock) == 10
    ops=load(H/'OPERATIONS_SOURCE_LOCK.json');continuation=load(H/'OPERATIONS_CONTINUATION.json')
    assert sha(R/ops['source'])==ops['sha256']==continuation['source_sha256']
    assert sha(H/'SOURCE_LOCK.json')==ops['original_science_source_lock_sha256']
    assert events.count('PROBE_COMMIT')==47 and events.count('PROBE_COMMIT_RECOVERED')==1
    assert events.count('FEATURE_COMMIT')==8
    assert sha(H/'PROBE_I_FAST_RHO_202610255401_OFF_257.json')==continuation['completed_probe_sha256']
    assert sha(H/'local/PROBE_I_FAST_RHO_202610255401_OFF_257.json')==continuation['completed_probe_raw_sha256']
    probes=[load(p) for p in H.glob('PROBE_*.json')]
    assert len(probes)==48 and len({p['id'] for p in probes})==48
    assert Counter(p['kind'] for p in probes)==Counter(OFF_257=24,LOC_257=8,OFF_C16=8,LOC_C16=8)
    checks = []
    for path in sorted((H/'groups').glob('*/RECEIPT.json')):
        d = path.parent
        a, data, cm = [load(d/n) for n in ['ACQUISITION.json','DATA.json','COUNTERMODELS.json']]
        by_id = {v['id']:v for part in ['OLD','NEW'] for v in data[part]}
        for result in cm['branch_results']:
            branch = result['branch']
            presentations = [v for v in a['training_records'] if v['write'] and (
                (branch == 'W' and v['branch'] == 'W') or
                (branch == 'SH_OLD' and v['branch'] == 'SH_OLD') or
                (branch == 'N_OLD' and v['branch'] == 'W' and v['phase'] == 'OLD'))]
            assert len(presentations) == result['learned_record_presentations']
            pairs = [(by_id[v['id']]['prompt'], by_id[v['id']]['answer']) for v in presentations]
            exact = dict(pairs)
            assert len(exact) == result['unique_raw_keys']
            assert all(exact[p] == y for p,y in pairs) # Saved-response recitation witness.
            seen_orbits = {tuple(by_id[v['id']]['orbit']) for v in presentations}
            assert len(seen_orbits) == result['unique_canonical_keys']
            assert not seen_orbits.intersection(tuple(v['orbit']) for v in data['assessment'])
            assert all(v['prompt'] not in exact for v in data['assessment'])
            favorite = Counter(y for _,y in pairs)
            fav = min(favorite, key=lambda x:(-favorite[x], bytes.fromhex(x))) if favorite else ''
            positions = [Counter() for _ in range(4)]
            for _, y in pairs:
                for i,byte in enumerate(list(bytes.fromhex(y))+[256]): positions[i][byte] += 1
            pos = []
            for c in positions:
                byte = min(c, key=lambda x:(-c[x], x)) if c else 256
                pos.append(byte)
                if byte == 256: break
            for kind, score in result['scores'].items():
                expected_rows = data['assessment'] if a['ready'] else []
                assert len(score['rows']) == score['records'] == len(expected_rows)
                correct = 0
                for observed, question in zip(score['rows'], expected_rows):
                    choices = pos if kind == 'POSITION' else list(bytes.fromhex(fav))+[256]
                    eos = choices[-1] == 256
                    output = bytes(choices[:-1] if eos else choices).hex()
                    assert observed['id'] == question['id']
                    assert observed['output_hex'] == output and observed['EOS'] == eos
                    right = eos and output == question['answer']
                    assert observed['correct'] == right
                    correct += right
                assert correct == score['correct']
            checks.append(dict(group=d.name,branch=branch,learned_presentations=len(pairs),
                               recitation_taught_correct=len(exact),held_key_overlap=0))
    counts = load(H/'COUNTS.json')
    assert counts['counts']['online_probe_records']==16384 and counts['counts']['feature_records']==1280
    assert counts['CPU'] < 7200 and counts['wall_seconds'] < 10800
    assert counts['RSS_peak_bytes'] <= 1073741824
    out = dict(status='ACCEPTED',scope='saved-response countermodels, recitation, leakage, global order and resource closure',
               checks=checks,global_acquisition_lock_before_any_held=True,
               new_training=False,new_model_queries=False,CPU=time.process_time()-start,
               RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (H/'CLOSURE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'closure':'ACCEPTED','branches':len(checks),'CPU':out['CPU']}))

if __name__ == '__main__': main()
